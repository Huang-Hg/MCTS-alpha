"""GPU 因子评估度量(device=cuda 路径)—— 镜像 backtest/ops 的 C 度量语义,消费**常驻 cupy**
因子 + 常驻 cupy y_future,使因子在 device=cuda 下从求值到评分全程不离显存(消灭每候选 1776MB D2H)。

与 CPU 侧对照(逐位语义对齐,容差 ~1e-6):
    finite_validstd  ← fm_cs_finite_validstd   (finite_ratio=行内≥2 finite 的行占比;valid=var>thr² 行占比)
    holdable_coverage← fm_cs_holdable_coverage  (mean_t over h>0 of #(x∧y finite)/#(y finite))
    per_t_ic / ic    ← fm_per_t_ic / fm_ic      (逐行 Pearson;n<3 或 dx/dy≤0 → NaN/跳过)
    icir             ← metrics.icir             (nanmean/nanstd(ddof=0);non-finite mean 或 std<1e-9 → 0)
    rank_ic          ← fm_rank_ic               (ic(cs_rank(x), cs_rank(y));y 的 rank 可缓存)
    turnover         ← fm_turnover              (Σ|Δx| / 跨列均 std;ddof=0)
    per_t_pnl        ← fm_per_t_pnl             (cs-demean + L1=1 加权 · y;返回 (T,) cupy)

归约一律 `dtype=cupy.float64` 累加器:消费卡 panel 为 f32 时仍保住全盘 reduction 精度(像全局 scan
那样 f32 累加会炸),且不物化 f64 整盘。标量度量 D2H 回 float;per_t_pnl 留 (T,) cupy(evaluator
仅此一条 (T,) D2H 喂 CPU 池 corr)。
"""
from __future__ import annotations

import cupy as cp

from backtest.ops_cuda import cs_rank, k_per_t_ic, k_per_t_pnl

_F64 = cp.float64


def finite_validstd(x, thr_std: float = 1e-9):
    """→ (finite_ratio, valid_ratio_cs)。finite_ratio = 行内有≥2 个 finite 的行占比(非元素级!);
    valid = 这些行中 var(ddof=0) > thr_std² 的占比。两者分母均为 T。"""
    T = x.shape[0]
    f = cp.isfinite(x)
    xf = cp.where(f, x, _F64(0))
    n = f.sum(1, dtype=_F64)
    s = xf.sum(1, dtype=_F64)
    ss = (xf * xf).sum(1, dtype=_F64)
    has = n >= 2
    nsafe = cp.maximum(n, 1)
    mean = s / nsafe
    var = ss / nsafe - mean * mean
    n_has_var = float(has.sum())
    n_valid = float((has & (var > thr_std * thr_std)).sum())
    return (n_has_var / T if T > 0 else 0.0), (n_valid / T if T > 0 else 0.0)


def holdable_coverage(x, y) -> float:
    """mean_t over (h>0) of #(x∧y finite)/#(y finite)。y 已 NaN-mask 到 holdable。"""
    fy = cp.isfinite(y)
    h = fy.sum(1, dtype=_F64)
    c = (fy & cp.isfinite(x)).sum(1, dtype=_F64)
    pos = h > 0
    nb = float(pos.sum())
    if nb == 0:
        return 0.0
    ratio = cp.where(pos, c / cp.maximum(h, 1), _F64(0))
    return float(ratio.sum(dtype=_F64) / nb)


def per_t_ic(x, y):
    """(T,) 逐行 Pearson IC;n<3 或 dx≤0 或 dy≤0 → NaN(下游 nan-aware)。公式贴 C:
    num=sxy−mx·sy;dx=sxx−mx·sx;dy=syy−my·sy。**融合单遍 warp-shuffle 核**(k_per_t_ic)替原
    ~10 遍整盘 .sum(1) 归约;f64 累加器同精度(消费卡 f32 panel 仍全盘 f64 reduction)。"""
    return k_per_t_ic(x, y)


def _mean_over_finite(v):
    """nanmean,全 NaN → None;返回 (mean, finite_vals) 供 ic/icir 复用。"""
    f = cp.isfinite(v)
    cnt = float(f.sum())
    if cnt == 0:
        return None, None, 0
    vv = cp.where(f, v, _F64(0))
    return float(vv.sum(dtype=_F64) / cnt), f, cnt


def ic(x, y) -> float:
    """fm_ic = mean over valid-t of per_t_ic。无有效 t → 0。"""
    pti = per_t_ic(x, y)
    mean, _f, cnt = _mean_over_finite(pti)
    return mean if cnt > 0 else 0.0


def icir(x, y):
    """→ (icir, mean_ic, std_ic),metrics.icir 语义:nanstd(ddof=0);non-finite mean 或 std<1e-9 → icir=0。"""
    pti = per_t_ic(x, y)
    f = cp.isfinite(pti)
    cnt = float(f.sum())
    if cnt == 0:
        return 0.0, 0.0, 0.0
    vals = pti[f]
    mean_ic = float(vals.mean(dtype=_F64))
    std_ic = float(vals.std(dtype=_F64))            # ddof=0
    if not cp.isfinite(cp.asarray(mean_ic)) or std_ic < 1e-9:
        return 0.0, (mean_ic if mean_ic == mean_ic else 0.0), std_ic
    return mean_ic / std_ic, mean_ic, std_ic


def topk_gross_sharpe(x, y, k):
    """per-bar top-k long / bottom-k short **等权** gross PnL 的 Sharpe(mean/std)。匹配部署 top-K 多空构造
    (无换手缓冲、无费),奖励"极端押注真赚钱"的 alpha → 治 |IC| 的 IC↔PnL 背离(rv_* 类 IC 高但极端 PnL 烂)。
    注:rank-conviction selection reward 2026-06-25 证否(选高换手 alpha → net 更差,diag_pool_rankconv)→ 回退等权;
    deploy 端 conviction 仍留 rl.sizing.topk_ls_weights。每 bar 有效截面 <2k → 跳过;有效 bar <5 或 std≈0 → 0。带符号。"""
    fin = cp.isfinite(x) & cp.isfinite(y)
    nfin = fin.sum(1)
    S = x.shape[1]
    xl = cp.where(fin, x, _F64(-cp.inf))
    xs = cp.where(fin, x, _F64(cp.inf))
    thr_hi = cp.partition(xl, S - k, axis=1)[:, S - k]      # 每行第 k 大(finite)
    thr_lo = cp.partition(xs, k - 1, axis=1)[:, k - 1]      # 每行第 k 小(finite)
    long_m = fin & (x >= thr_hi[:, None])
    short_m = fin & (x <= thr_lo[:, None])
    # 腿内等权 gross PnL(rank-conviction selection reward 已证否 net 更差 → 回退 equal;deploy conviction 留 topk_ls_weights)
    nl = long_m.sum(1, dtype=_F64); ns = short_m.sum(1, dtype=_F64)
    yl = cp.where(long_m, y, _F64(0)).sum(1, dtype=_F64) / cp.maximum(nl, _F64(1))
    ys = cp.where(short_m, y, _F64(0)).sum(1, dtype=_F64) / cp.maximum(ns, _F64(1))
    pnl = cp.where(nfin >= 2 * k, yl - ys, cp.nan)
    f = cp.isfinite(pnl)
    cnt = float(f.sum())
    if cnt < 5:
        return 0.0
    v = pnl[f]
    mu = float(v.mean(dtype=_F64)); sd = float(v.std(dtype=_F64))
    return mu / sd if sd > 1e-12 else 0.0


def rank_cache(y):
    """y_future 的 cs_rank panel(每候选不变 → evaluator 缓存一次,喂 rank_ic 省一次排序)。"""
    return cs_rank(y)


def rank_ic(x, y, ry=None) -> float:
    """fm_rank_ic = ic(cs_rank(x), cs_rank(y))。ry 传入缓存的 rank_cache(y) 则省 y 排序。"""
    rx = cs_rank(x)
    if ry is None:
        ry = cs_rank(y)
    return ic(rx, ry)


def turnover(x) -> float:
    """fm_turnover = mean_{Δ both-finite}|Δx| / mean_{col std>0} std(ddof=0)。"""
    a = x[1:]; b = x[:-1]
    both = cp.isfinite(a) & cp.isfinite(b)
    d = cp.where(both, a - b, _F64(0))
    sum_abs_d = float(cp.abs(d).sum(dtype=_F64))
    cnt_d = float(both.sum())
    xf = cp.isfinite(x)
    xx = cp.where(xf, x, _F64(0))
    cc = xf.sum(0, dtype=_F64)
    cs = xx.sum(0, dtype=_F64)
    css = (xx * xx).sum(0, dtype=_F64)
    ccsafe = cp.maximum(cc, 1)
    mean = cs / ccsafe
    var = (css - mean * cs) / ccsafe                # ddof=0(贴 C 形式)
    sd = cp.sqrt(cp.where(var > 0, var, _F64(0)))
    valid_sd = (cc >= 2) & (var > 0) & cp.isfinite(sd) & (sd > 0)
    sum_sd = float(cp.where(valid_sd, sd, _F64(0)).sum(dtype=_F64))
    cnt_sd = float(valid_sd.sum())
    if cnt_d == 0 or cnt_sd == 0:
        return 0.0
    norm = sum_sd / cnt_sd
    if norm <= 0:
        return 0.0
    return (sum_abs_d / cnt_d) / norm


def per_t_pnl(values, y):
    """(T,) cupy。v=values−cs_mean;w=v/Σ|v|;out[t]=Σ w·y(v 与 y 都 finite 的 s)。
    n_v==0 或 Σ|v|<1e-12 → 0。evaluator 唯一 D2H 的 (T,) 数组(喂 CPU 池 pnl_corr)。
    **融合两轮归约核**(k_per_t_pnl)替原 ~7 遍整盘归约;f64 累加器同精度。"""
    return k_per_t_pnl(values, y)
