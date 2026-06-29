"""
Alpha 评估 — 标准 IC 口径(唯一质量 = rank IC;池不定权,deploy/eval 走 AFF 融合;2026-06-27 删 icir)。

门通路(任一门拒 → _bad;below-gate → per_t_pnl=None,gp 侧落 _REWARD_FLOOR 不收集):
  G0         无 operand / raw-price-wrapper(无 ts_*/pair 且仅 OHLCV leaf)结构静态拒
  G1         finite_ratio + valid_cs 数值有效性硬门
  G1.5       holdable 覆盖率门(挡稀疏信号在 1-3 截面上的伪高 IC)
  G2         max_cos_sim_with_pool > cos_sim_threshold(结构源头去重)
  FailCache  候选 embedding cos>0.995 命中 hard-fail 簇(性能短路)
  --- above-gate(|rank_ic| > admit_rankic_min)---
  eval_tree(tree, panels) → values (T,S) → rank_ic / per_t_pnl(C ops 或 GPU)
  per_t_pnl 暴露给 gp:obj1 = |rank_ic| 搜索 reward,+ 池准入 Δ=|rank_ic|·(1−R²) / 行为多样性。

device=cpu 走 C ops;device=cuda 因子常驻显存 + GPU 度量(消灭每候选 (T,S) D2H)。

输出 dict:rank_ic / per_t_pnl(供 gp + try_new)+ topk_sharpe / turnover / ic(仅 diagnostics=True)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config.config import ini, NeutralizeConfig
from backtest import ops
from evaluation.cache import EvalCache
from evaluation.ast import AlphaTree
from evaluation.expression import evaluate as eval_tree
from rl.alpha_pool import AlphaPool
from rl.sizing import build_crowding_basis, crowding_neutralize, _CROWD_TOKENS

# device 评分后端(device=cuda:因子全程留显存 —— resident 求值 + GPU 度量,消灭每候选 1776MB D2H;
#   per_t_pnl (T,) D2H 一次喂 CPU 池 corr)。cpu 路径完全不引 cupy/ops_cuda。
#   INI [evaluator] device 与 expression.py 同源。
_DEVICE = ini('evaluator', 'device', 'cpu')
if _DEVICE == 'cuda':
    import cupy as cp
    import evaluation.expression as _ec
    import backtest.ops_cuda as _gops          # DTYPE(T1000=f32 / V100=f64),拥挤基跟随 resident 因子精度
    import backtest.ops_cuda.metrics as _gm

    _RY_CACHE: Dict[int, tuple] = {}

    def _ry(y_future, y_dev):
        """y_future 的 cs_rank panel 缓存(rank_ic 用;y 整 run 不变 → 算一次)。"""
        key = id(y_future)
        hit = _RY_CACHE.get(key)
        if hit is not None and hit[0] is y_future:
            return hit[1]
        r = _gm.rank_cache(y_dev)
        _RY_CACHE[key] = (y_future, r)
        return r


# 拥挤基 Gz 缓存(panels 整 run 不变 → 算一次);cpu 复用 sizing numpy,cuda cupy 镜像(逐位对齐)。
_GZ_CACHE: Dict[int, tuple] = {}


def _crowd_gz(panels, cuda):
    key = id(panels)
    hit = _GZ_CACHE.get(key)
    if hit is not None and hit[0] is panels:
        return hit[1]
    if cuda:
        rows = []
        for tok in _CROWD_TOKENS:
            x = cp.asarray(panels[tok], dtype=_gops.DTYPE)      # f32(T1000)避免 f64 翻倍显存 + 1/32 吞吐
            f = cp.isfinite(x)
            n = f.sum(1, keepdims=True).astype(x.dtype)         # cast:否则 f32/int64 被 cupy 上抬回 f64 污染 gz
            mu = cp.where(f, x, 0.0).sum(1, keepdims=True) / cp.maximum(n, 1)
            var = cp.where(f, (x - mu) ** 2, 0.0).sum(1, keepdims=True) / cp.maximum(n, 1)
            sd = cp.sqrt(var)
            gz = cp.where(f & (sd > 1e-12), (x - mu) / cp.where(sd > 1e-12, sd, 1.0), 0.0)
            rows.append(gz)
        gz = cp.stack(rows, axis=1)
    else:
        gz = build_crowding_basis(panels)
    _GZ_CACHE[key] = (panels, gz)
    return gz


def _crowd_resid(arr, panels, cuda):
    """候选因子 arr (T,S) ⊥ 拥挤基。cuda→cupy 镜像 sizing.crowding_neutralize 逐位语义。"""
    gz = _crowd_gz(panels, cuda)
    if not cuda:
        return crowding_neutralize(arr, gz)
    m = cp.isfinite(arr)
    gzm = gz * m[:, None, :]
    GGt = cp.einsum('tis,tjs->tij', gzm, gzm) + 1e-6 * cp.eye(gz.shape[1], dtype=gz.dtype)   # eye 跟随 DTYPE,不上抬 f64
    Gf = cp.einsum('tis,ts->ti', gzm, cp.where(m, arr, 0.0))
    c = cp.linalg.solve(GGt, Gf[..., None])[..., 0]
    return cp.where(m, arr - cp.einsum('ti,tis->ts', c, gz), cp.nan)

_PROF: Dict[str, float] = {
    'eval_tree':   0.0,
    'cs_std':      0.0,
    'ic_diag':     0.0,
    'total':       0.0,
}
_PROF_N: Dict[str, int] = {
    'calls':           0,
    'gate1_reject':    0,
    'fail_cache_hit':  0,
    'cos_sim_dup':     0,
}


# ============================================================================
# Fail-fast embedding cache(性能加速,非语义门)
#   evaluator 顶部把候选 embedding 跟已知 hard-fail 簇做 cos sim,>0.995 直接返回 _bad,
#   省 eval_tree(reward_fn 大头)。缓存来源:G1 gate1_reject(结构性退化)。
#   阈值 0.995 比入池 cos_sim_threshold(INI)严;net 权重改 → 嵌入漂 → on_net_update_hook 清。
# ============================================================================

class _FailCache:
    def __init__(self, threshold: float = 0.995):
        self.threshold = threshold
        self.embs: List[np.ndarray] = []
        self._mat: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.embs.clear()
        self._mat = None

    def _matrix(self) -> np.ndarray:
        if self._mat is None or self._mat.shape[0] != len(self.embs):
            M = np.stack(self.embs)
            n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
            self._mat = (M / n).astype(np.float32)
        return self._mat

    def hit(self, emb: np.ndarray) -> bool:
        if not self.embs:
            return False
        M = self._matrix()
        nc = emb / (np.linalg.norm(emb) + 1e-12)
        return float(np.max(M @ nc.astype(np.float32))) > self.threshold

    def add(self, emb: np.ndarray) -> None:
        self.embs.append(emb.astype(np.float32, copy=True))
        self._mat = None


_FAIL_CACHE = _FailCache(threshold=0.995)

_REWARD_FLOOR = -2.0      # gp_baseline 无效个体(门拒 / cost 越界)的 fitness 下界(< 任何 |rank_ic|)


def on_net_update_hook() -> None:
    """policy net 权重更新时由 trainer 调一次。清 _FAIL_CACHE(嵌入漂)。"""
    _FAIL_CACHE.reset()


def reset_prof() -> None:
    for k in _PROF: _PROF[k] = 0.0
    for k in _PROF_N: _PROF_N[k] = 0


def get_prof() -> Dict[str, float]:
    return {**_PROF, **{f'n_{k}': v for k, v in _PROF_N.items()}}


def merge_prof(d: Dict[str, float]) -> None:
    """累加 worker 进程回传的 prof 增量(局级并行 merge,key 口径同 get_prof)。"""
    for k in _PROF: _PROF[k] += d[k]
    for k in _PROF_N: _PROF_N[k] += int(d[f'n_{k}'])


@dataclass
class EvalConfig:
    # 数值有效性 / 覆盖率门(固定实现常量)
    min_finite_ratio:      float = 0.10
    min_cs_valid_ratio:    float = 0.5
    min_holdable_coverage: float = 0.70    # holdable 截面覆盖率下限(挡稀疏信号伪高 IC)
    cos_sim_dup_threshold: float = ini('alpha_pool', 'cos_sim_threshold', 0.99)   # G2 结构去重
    # 准入便宜门:|rank_ic| < 此 → 跳过财富路径(reward 已现算,免费)
    admit_rankic_min:      float = 0.01
    top_k:                 int   = ini('backtest_reward', 'top_k', 8)   # top-K gross Sharpe 度量每腿仓数
    # 候选 ⊥ 拥挤子空间(funding/lsr/基差/OI)再评分 → 强制方向 alpha、剿 carry 收割(2026-06-26)。
    # 统一中性化单一源:[neutralize].neutralize 含 crowding 即开(见 config.NeutralizeConfig)。
    crowding_neutral:      bool  = NeutralizeConfig().crowding


def evaluate_alpha(
    tree: AlphaTree,
    panels: Dict[str, np.ndarray],
    y_future: np.ndarray,
    pool_obj: AlphaPool,
    cfg: EvalConfig = EvalConfig(),
    cache: Optional[EvalCache] = None,
    diagnostics: bool = False,
) -> Dict[str, float]:
    """因子评估 path。pool_obj 必传(候选 embedding / G2 cos-dup 用)。

    返回 dict 字段:
        rank_ic   — 候选秩 IC(gp obj1 = |rank_ic|;= 唯一标准质量口径)
        per_t_pnl — (T,) 候选 per-bar L1-norm signed PnL(above-gate 才暴露;供池 R²/Δ + 行为多样性)
        ic        — 仅 diagnostics=True(等权 Pearson,只作日志对照)
    """
    if not tree.has_operand():
        return _bad()
    # G0: raw-price-wrapper 结构静态拒(无 ts_*/pair 且仅 OHLCV leaf,cs 截面等价 symbol identity)
    if tree.is_raw_price_wrapper():
        _PROF_N['gate1_reject'] += 1
        return _bad()

    _t_total = time.perf_counter()
    _PROF_N['calls'] += 1

    # FailCache(emb cos > 0.995 → 同源 hard-fail 簇,短路)
    cand_emb = pool_obj.candidate_embedding(tree)
    if cand_emb is not None and _FAIL_CACHE.hit(cand_emb):
        _PROF_N['fail_cache_hit'] += 1
        _PROF['total'] += time.perf_counter() - _t_total
        return _bad()

    cuda = _DEVICE == 'cuda'
    _t = time.perf_counter()
    if cuda:
        fac = _ec.evaluate_resident(tree, panels)        # 常驻 cupy,不 D2H
        y_dev = _ec.gy(y_future)
    else:
        values = eval_tree(tree, panels, cache)
    _PROF['eval_tree'] += time.perf_counter() - _t

    # G1: numeric validity 硬门
    _t = time.perf_counter()
    if cuda:
        finite_ratio, valid_ratio_cs = _gm.finite_validstd(fac, 1e-9)
    else:
        finite_ratio, valid_ratio_cs = ops.cs_finite_validstd(values, 1e-9)
    _PROF['cs_std'] += time.perf_counter() - _t
    if finite_ratio < cfg.min_finite_ratio:
        _PROF_N['gate1_reject'] += 1
        if cand_emb is not None: _FAIL_CACHE.add(cand_emb)
        _PROF['total'] += time.perf_counter() - _t_total
        return _bad(finite_ratio=finite_ratio)
    if valid_ratio_cs < cfg.min_cs_valid_ratio:
        _PROF_N['gate1_reject'] += 1
        if cand_emb is not None: _FAIL_CACHE.add(cand_emb)
        _PROF['total'] += time.perf_counter() - _t_total
        return _bad(finite_ratio=finite_ratio, valid_ratio_cs=valid_ratio_cs)

    # cuda 下因子留 cupy(fac,GPU 度量直吃);cpu 路 values_c 供 C-ops 度量(rank_ic/topk_sharpe/per_t_pnl)。
    values_c = None if cuda else np.ascontiguousarray(values, dtype=np.float64)

    # G1.5: holdable 覆盖率门 — 信号在 holdable 截面(y 非 NaN)的 per-bar 平均覆盖率 < 阈值 → 拒。
    # 挡 sqrt(cs_z−2) 类近全 NaN 稀疏信号:每 bar 只覆盖极少 holdable → per_t_ic 在 1-3 点上算出
    # 伪高 |IC|(实测 [0] ic=−0.695 覆盖率仅 ~7%),污染 pool ensemble + policy reward。
    coverage = (_gm.holdable_coverage(fac, y_dev) if cuda
                else ops.cs_holdable_coverage(values_c, y_future))
    if coverage < cfg.min_holdable_coverage:
        _PROF_N['gate1_reject'] += 1
        if cand_emb is not None: _FAIL_CACHE.add(cand_emb)
        _PROF['total'] += time.perf_counter() - _t_total
        return _bad(finite_ratio=finite_ratio, valid_ratio_cs=valid_ratio_cs)

    # G2: 结构近完全重复 step 惩罚 — max_cos_sim_with_pool > cos_sim_threshold → score=0(结构源头去重)
    if cand_emb is not None and pool_obj.members:
        if pool_obj.max_cos_sim_with(cand_emb) > cfg.cos_sim_dup_threshold:
            _PROF_N['cos_sim_dup'] += 1
            _PROF['total'] += time.perf_counter() - _t_total
            return _bad(finite_ratio=finite_ratio, valid_ratio_cs=valid_ratio_cs)

    # 拥挤中性化:候选 ⊥ 拥挤子空间(funding/lsr/基差/OI)→ 强制方向 alpha(carry 因子残差≈0,
    # reward 塌)。门(G1/G1.5 数值有效性)在原始因子上,此处对残差打分。线性 → 集成可交换,
    # 池存树 → deploy 须同样中性化(main.eval_pool_val + diag 已加 crowding_neutralize)。
    if cfg.crowding_neutral:
        if cuda:
            fac = _crowd_resid(fac, panels, True)
        else:
            values_c = _crowd_resid(values_c, panels, False)

    # 度量:rank_ic / topk_sharpe / turnover / per_t_pnl(ic 仅 diagnostics)。device=cuda 走 GPU 度量
    #   (因子留显存,per_t_pnl 是唯一 D2H 的 (T,) 数组);cpu 走 C-ops。per_t_pnl 供 admission
    #   Δ/holdout;仅越准入便宜门才 expose 进 dict(worker 靠 `per_t_pnl is not None` 收准入)。
    _t = time.perf_counter()
    if cuda:
        rank_ic_val = _gm.rank_ic(fac, y_dev, _ry(y_future, y_dev))
        turnover_val = _gm.turnover(fac)
        ic_val = _gm.ic(fac, y_dev) if diagnostics else 0.0
        topk_sharpe_val = _gm.topk_gross_sharpe(fac, y_dev, cfg.top_k)
        per_t_pnl_full = cp.asnumpy(_gm.per_t_pnl(fac, y_dev)).astype(np.float64, copy=False)
    else:
        rank_ic_val = ops.rank_ic(values_c, y_future)
        turnover_val = float(ops.turnover(values_c))
        ic_val = ops.ic(values_c, y_future) if diagnostics else 0.0
        topk_sharpe_val = _topk_gross_sharpe_np(values_c, y_future, cfg.top_k)
        per_t_pnl_full = ops.per_t_pnl(values_c, y_future)
    _PROF['ic_diag'] += time.perf_counter() - _t

    complexity_val = float(tree.total_cost())

    # 准入便宜门:|rank_ic| > admit_rankic_min 才暴露 per_t_pnl(供 collect + 池 R²/Δ);余短路不收集。
    if diagnostics or abs(rank_ic_val) > cfg.admit_rankic_min:
        per_t_pnl = per_t_pnl_full                     # 复用上方恒算结果
    else:
        per_t_pnl = None                               # below-gate:不收集不准入(≤admit_rankic_min)
    _PROF['total']  += time.perf_counter() - _t_total

    return {
        'ic':              float(ic_val),
        'rank_ic':         float(rank_ic_val),
        'topk_sharpe':     float(topk_sharpe_val),
        'turnover':        float(turnover_val),
        'complexity':      complexity_val,
        'finite_ratio':    finite_ratio,
        'valid_ratio_cs':  valid_ratio_cs,
        'per_t_pnl':       per_t_pnl,
    }


def _bad(**fields) -> Dict[str, float]:
    # 坏结构(G0/G1/G2/coverage 拒):rank_ic=0 → per_t_pnl=None,gp 侧 fitness 落 _REWARD_FLOOR、不收集。
    out = {
        'ic': 0.0, 'rank_ic': 0.0, 'topk_sharpe': 0.0, 'turnover': 0.0,
        'complexity': 0.0, 'finite_ratio': 0.0, 'valid_ratio_cs': 0.0,
        'per_t_pnl': None,
    }
    out.update(fields)
    return out


def _topk_gross_sharpe_np(x, y, k):
    """cpu 版 per-bar top-k 多空 **等权** gross PnL Sharpe(向量化 numpy,语义同 ops_cuda.metrics.topk_gross_sharpe)。"""
    fin = np.isfinite(x) & np.isfinite(y)
    S = x.shape[1]
    if S <= 2 * k:
        return 0.0
    xl = np.where(fin, x, -np.inf)
    xs = np.where(fin, x, np.inf)
    thr_hi = np.partition(xl, S - k, axis=1)[:, S - k]
    thr_lo = np.partition(xs, k - 1, axis=1)[:, k - 1]
    long_m = fin & (x >= thr_hi[:, None])
    short_m = fin & (x <= thr_lo[:, None])
    # 腿内等权 gross PnL(rank-conviction **selection reward** 已证否:选出高换手 alpha、net 更差,
    # 2026-06-25 diag_pool_rankconv 两窗一致 → 回退 equal;deploy 端 conviction 仍留 topk_ls_weights)
    nl = long_m.sum(1); ns = short_m.sum(1)
    yl = np.where(long_m, y, 0.0).sum(1) / np.maximum(nl, 1)
    ys = np.where(short_m, y, 0.0).sum(1) / np.maximum(ns, 1)
    pnl = np.where(fin.sum(1) >= 2 * k, yl - ys, np.nan)
    v = pnl[np.isfinite(pnl)]
    if v.size < 5:
        return 0.0
    sd = v.std()
    return float(v.mean() / sd) if sd > 1e-12 else 0.0


