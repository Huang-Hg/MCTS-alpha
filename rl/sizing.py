"""确定性 sizing / 信号变换层 — 均与 bt eval / live 共享、无 policy 反馈环(2026-06-27 删 RL 孪生):

1. **alpha → weight 转换**(`signal_to_weight`):cs_zscore → clip(±3) → post-clip demean
   (dollar-neutral)→ L1 normalize with cap,与 bt C kernel `_row_signal_to_weight` 逐字一致。
1b. **top-K 多空构造**(`_topk_ls_step` / `topk_ls_weights`):swap_n/min_hold 换手缓冲,conviction-rank 腿权。
1c. **等权动态 β 中性化**(`beta_neutralize`):信号 ⊥ 因果滚动 β_s。
1d. **crowding/funding 暴露中性化**(`build_crowding_basis` / `crowding_neutralize`):候选 ⊥ 拥挤子空间。
1e. **AFF 自适应融合**(`aff_fuse`):因果滚动 lstsq,取代静态 ICIR/rank_ic 池组合。
2. **组合层 vol-target 杠杆 + gross backstop**(`VolTargetConfig` / `vol_target_scale_np` /
   `gross_backstop_np`):杠杆只是组合层一个标量,按 vol_24h 择时缩放、均匀作用所有名,不破坏截面相对配置。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtest import ops
from evaluation.grammar import OperandToken


# ============================================================================
# 1. alpha → weight 转换
#
# Canonical alpha → weight 转换,**与 bt C kernel `_row_signal_to_weight` 完全一致**。
#
# bt 内核(`backtest/portfolio_bt.c:49`)与 live(`trade/signal.py`)必须用同一份转换逻辑,
# 否则 deploy bundle 在 bt 上 sharpe 与 live 行为不可比(已踩过坑:live 缺 post-clip
# demean → 非 dollar-neutral,16h -11% 部分由此引发)。
#
# 四步:
#   1. cs_zscore(NaN-aware,只在 finite sym 上算 mu/sd)
#   2. clip(±3)
#   3. post-clip demean(强制 Σw=0,dollar-neutral)
#   4. L1 normalize with cap(Σ|w| ≤ leverage_cap)
# ============================================================================

def signal_to_weight(alpha: np.ndarray,
                     leverage_cap: float = 1.0,
                     per_sym_cap: float | None = None,
                     valid_mask: np.ndarray | None = None) -> np.ndarray:
    """alpha (T, S) → weight (T, S) 或 alpha (S,) → weight (S,)。

    NaN sym → weight=0(不参与 mu/sd/L1);全行无效(n<2 或 sd≈0) → 整行 0。
    leverage_cap=1.0 + per_sym_cap=None 时与 bt C kernel 完全一致(Σ|w| ≤ 1)。

    per_sym_cap(H7 防御):L1 norm 后若 max|w|>cap,把超额削回 cap 并重 demean +
    重 L1 norm。会使 Σ|w| 略小于 leverage_cap(把"集中头寸"换成"分散 + 略低 gross")。
    bt 不传此参数,保持 dollar-neutral 不变量。
    """
    one_d = (alpha.ndim == 1)
    a = alpha[None, :] if one_d else alpha
    T, S = a.shape

    # valid_mask:未上市/无量 symbol → NaN(cs_zscore NaN-safe → 权重 0),listed 截面重 demean/L1。
    if valid_mask is not None:
        vm = valid_mask[None, :] if valid_mask.ndim == 1 else valid_mask
        a = np.where(vm, a, np.nan)

    # Step 1: cs_zscore(C kernel,NaN-preserving)
    z = ops.cs_zscore(a)

    # Step 2: clip(±3) — NaN 在 np.clip 下保持 NaN
    z = np.clip(z, -3.0, 3.0)

    # Step 3: post-clip demean(C kernel,NaN-preserving)
    z = ops.cs_demean(z)

    # Step 4: NaN → 0 用于 L1 norm
    w = np.where(np.isfinite(z), z, 0.0)
    l1 = np.abs(w).sum(axis=1, keepdims=True)
    scale = np.where(l1 > leverage_cap,
                     leverage_cap / np.maximum(l1, 1e-12),
                     1.0)
    w = w * scale

    # Step 5(H7,仅 live 用):per-sym cap → 重 demean → 收敛。clip+demean 是 contraction,
    # 3 轮足够收敛到 max|w| ≤ cap 且 Σw≈0(实测 24 sym basket,3 轮 max|w|-cap < 1e-12)。
    if per_sym_cap is not None and per_sym_cap > 0:
        for _ in range(3):
            w = np.clip(w, -per_sym_cap, per_sym_cap)
            w = w - w.mean(axis=1, keepdims=True)
        # final 再 clip 一次保严格 cap(此时 Σw 可能 ~1e-N 偏离 0,可接受)
        w = np.clip(w, -per_sym_cap, per_sym_cap)
        l1 = np.abs(w).sum(axis=1, keepdims=True)
        scale = np.where(l1 > leverage_cap,
                         leverage_cap / np.maximum(l1, 1e-12),
                         1.0)
        w = w * scale

    return w[0] if one_d else w


# ============================================================================
# 1b. top-K 多空组合构造(swap-n / min-hold 缓冲)
#
# 贴 alphasage/alphacfg 的 top-K 选股约定(TopKSwapN / qlib TopkDropout),但**允许做空**:
#   每个决策 bar 做多 signal 最高 k 名、做空最低 k 名(共 2k 仓),腿内等权、dollar-neutral、
#   Σ|w| = leverage_cap(多腿合 +cap/2、空腿合 −cap/2)。等权天然抗尾部(替代 z 加权把敞口
#   堆到极端分位的少数名 → 修 6 月 IC↔PnL 背离)。
#
# 换手缓冲(降 net 成本,成本是组合天花板的主因):
#   - 每腿每次 rebalance 至多换出 n_swap 名(掉出新鲜 top-k 的在持名,按最差优先)
#   - 在持满 min_hold 个决策 bar 才允许被换出;未到期即使掉出 top-k 也保留
#   - 失去 valid(退市/掉出因果 universe)的在持名强制平,不受 min_hold 约束
#
# 有状态:跨决策 bar 维护多/空持仓集 + 每名持有时长。backtest 走 path(顺序循环),
# live 单步复用同一 _step → 保证 eval≡live。
# ============================================================================

def _topk_ls_step(sig, valid, long_held, short_held, age, k, n_swap, min_hold):
    """单决策步 long-short top-k swap 更新。返回 (new_long, new_short, new_age) 三个 (S,) array。
    sig (S,) 信号;valid (S,) bool 因果可持仓;long_held/short_held (S,) bool 上期持仓;age (S,) 持有 bar 数。"""
    S = sig.shape[0]
    valid = valid & np.isfinite(sig)
    long_held = long_held & valid                       # 强制退出失效持名
    short_held = short_held & valid

    def _update(held, other_held, score):
        vi = np.where(valid)[0]
        vi_sorted = vi[np.argsort(score[vi])[::-1]]     # valid 名按 score 降序(最想要的在前)
        desired = np.zeros(S, dtype=bool)
        desired[vi_sorted[:k]] = True                   # 新鲜 top-k
        droppable = held & ~desired & (age >= min_hold)
        drop = np.where(droppable)[0]
        if drop.size > n_swap:
            drop = drop[np.argsort(score[drop])][:n_swap]   # 最差(score 最低)优先换出
        new_held = held.copy()
        new_held[drop] = False
        slots = k - int(new_held.sum())
        if slots > 0:
            takeable = ~new_held[vi_sorted] & ~other_held[vi_sorted]
            add = vi_sorted[takeable][:slots]           # 最优可入(未持本腿/对腿)补满
            new_held[add] = True
        return new_held

    new_long = _update(long_held, short_held, sig)
    new_short = _update(short_held, new_long, -sig)     # 对腿排除新多头,防同名双向
    stayed = (new_long & long_held) | (new_short & short_held)
    new_age = np.where(stayed, age + 1, 0)
    return new_long, new_short, new_age


def topk_ls_weights(signal_dec: np.ndarray, valid_dec: np.ndarray,
                    k: int, n_swap: int, min_hold: int,
                    leverage_cap: float = 1.0) -> np.ndarray:
    """signal_dec (n_dec, S) + valid_dec (n_dec, S) bool → weights (n_dec, S)。
    顺序应用 _topk_ls_step 选名;腿内按 signal **conviction rank 加权**(w ∝ 腿内 rank:long 越高 /
    short 越低权越大,最高信念=n、最低=1),dollar-neutral、Σ|w|=leverage_cap(满腿时)。
    等权口径已证否(net 更差,val/test 双窗 −0.16/+0.10 → +0.29/+0.24;尺度无关、gross 双窗都升;
    全栈同口径(eval + reward kernel 一致),见 scripts/diag_topk_weights。"""
    n_dec, S = signal_dec.shape
    W = np.zeros((n_dec, S), dtype=np.float64)
    long_held = np.zeros(S, dtype=bool)
    short_held = np.zeros(S, dtype=bool)
    age = np.zeros(S, dtype=np.int64)
    half = 0.5 * leverage_cap
    for t in range(n_dec):
        long_held, short_held, age = _topk_ls_step(
            signal_dec[t], valid_dec[t], long_held, short_held, age, k, n_swap, min_hold)
        for held, sgn in ((long_held, 1.0), (short_held, -1.0)):
            idx = np.where(held)[0]
            if idx.size == 0:
                continue
            c = signal_dec[t, idx] * sgn            # conviction:long +signal / short −signal
            rk = np.argsort(np.argsort(c)).astype(np.float64) + 1.0   # 腿内 rank 1..n(最高信念=n)
            W[t, idx] = sgn * (rk / rk.sum()) * half
    return W


# ============================================================================
# 1c. 等权动态 β 中性化(部署/eval 构造前的信号变换)
#
# 逐 bar 把 ensemble 信号截面正交于因果滚动 β_s(信号⊥β → 选出的多空腿净 β≈0),topk 选名/
# conviction 加权口径不变。**唯一保留的系统风险正交**:
#   · 系统风险因子扫描(scripts/diag_sysrisk_maxdd + diag_beta_split,val+test 双窗)结论:
#     等权 β 是唯一两窗都稳健的正交 —— net-ret +0.40/+0.25(vs base +0.17/+0.07)、gross 还升,
#     把 −0.5~−1.0 的**时变**系统 β 压到 ≈0(防 regime 翻转尾部:base 隐含 +1.5 BTC-β/−1.6 alt-β)。
#   · funding 正交(net val↔test 反号、叠加反拖累)/ size/vol/mom/basis(是 alpha 本体)/
#     BTC-alt 拆分(反号或缠 alpha)全证否,一概不做。
#   · max-DD 是 turnover×fee+funding 慢蚀,正交化救不了(另一条降成本线),勿误期望。
#
# β_s,d = 过去 w_beta 个决策 bar 已实现收益对等权市场的 cov/var,**只用 ≤d 数据 → 无 lookahead**。
# 单因子 → 闭式截面残差(向量化,与 scripts 的 per-bar lstsq orth 代数逐位等价:Σfc=0 消常数项)。
# ============================================================================

def beta_neutralize(signal_dec: np.ndarray, close_dec: np.ndarray,
                    valid_dec: np.ndarray, w_beta: int = 168) -> np.ndarray:
    """signal_dec (n_dec, S) + close_dec (n_dec, S) 决策网格 close + valid_dec (n_dec, S) bool
    → β-中性化残差信号 (n_dec, S)。w_beta=168(dec≈1h → 7d 因果滚动 β 窗)。
    非 valid / 无 β 名 → NaN(topk_ls_weights 的 isfinite 门自动跳过)。"""
    n_dec = signal_dec.shape[0]
    ret = np.full_like(close_dec, np.nan)
    ret[1:] = close_dec[1:] / close_dec[:-1] - 1.0
    mkt = np.nanmean(np.where(valid_dec, ret, np.nan), axis=1)             # (n_dec,) 等权市场

    def _rs(a):                                                            # 滚动和(前 w 行 expanding)
        c = np.cumsum(a, 0); o = c.copy(); o[w_beta:] = c[w_beta:] - c[:-w_beta]; return o

    cnt = np.arange(1, n_dec + 1).clip(max=w_beta).astype(np.float64)[:, None]
    x0 = np.nan_to_num(mkt); Y0 = np.nan_to_num(ret)
    Sx = _rs(x0)[:, None]; Sxx = _rs(x0 * x0)[:, None]
    Sy = _rs(Y0); Sxy = _rs(Y0 * x0[:, None])
    cov = Sxy / cnt - (Sx / cnt) * (Sy / cnt)
    varx = Sxx / cnt - (Sx / cnt) ** 2
    beta = cov / np.where(varx > 1e-12, varx, np.nan)                      # (n_dec,S) 因果 β_s

    # 截面 z(β)(非 valid / 非 finite → 0),再在 m=valid∧finite(sig) 上闭式回归取残差
    bm = np.where(valid_dec & np.isfinite(beta), beta, np.nan)
    mu = np.nanmean(bm, 1, keepdims=True); sd = np.nanstd(bm, 1, keepdims=True)
    zb = (bm - mu) / np.where(sd > 1e-12, sd, np.nan)
    zb = np.where(np.isfinite(zb), zb, 0.0)
    m = valid_dec & np.isfinite(signal_dec)
    nb = m.sum(1, keepdims=True).clip(min=1)
    fb = np.where(m, zb, 0.0).sum(1, keepdims=True) / nb                   # m 内重新中心化
    fc = zb - fb
    cov2 = np.where(m, fc * signal_dec, 0.0).sum(1, keepdims=True) / nb    # Σfc=0 → sig 常数项消失
    var2 = np.where(m, fc * fc, 0.0).sum(1, keepdims=True) / nb
    slope = cov2 / np.where(var2 > 1e-12, var2, np.nan)
    return np.where(m, signal_dec - slope * fc, np.nan)


# ============================================================================
# 1d. crowding/funding 暴露中性化(候选信号 ⊥ 拥挤子空间)
#
# 病根(2026-06-26 实证):alpha ≡ funding 暴露(截面共线)≡ funding 事件(时间共线)——做空
# 拥挤负 funding 小币,在结算时点收"拥挤释放"的回归、同时付 funding,盈利与成本同事件不可分。
# net-reward(per-bar 减 funding)被证无效:funding 是低方差慢漂移,per-bar 被收益噪声冲没。
# 唯一有效杠杆 = 把候选投到拥挤子空间正交补:carry 因子投影→0 reward 塌,方向因子存活 → 强制
# 搜索找方向 alpha。截面拥挤中性顺带解决时间集中(不站在结算跳价的名上)。线性 → 集成可交换
# (Σ neutralize = neutralize Σ),train(逐候选)与 deploy(集成)一致。接 [[project_cost_erosion_bottleneck]]。
# ============================================================================

# 拥挤子空间 = README §4b 实证的 funding-成本流 co-exposure 轴(LSR + 基差/funding + OI)
_CROWD_TOKENS = (OperandToken.FUNDING_RATE_INTERP, OperandToken.PREMIUM_INDEX_5M,
                 OperandToken.SUM_TOP_LSR, OperandToken.SUM_OI_VALUE)


def build_crowding_basis(panels) -> np.ndarray:
    """panels(决策网格 operand)→ 拥挤基 Gz (n, k, S),逐 bar cs-z(finite 上),非 finite → 0。"""
    rows = []
    for tok in _CROWD_TOKENS:
        x = np.ascontiguousarray(panels[tok], dtype=np.float64)
        f = np.isfinite(x)
        n = f.sum(1, keepdims=True)
        mu = np.where(f, x, 0.0).sum(1, keepdims=True) / np.maximum(n, 1)
        var = np.where(f, (x - mu) ** 2, 0.0).sum(1, keepdims=True) / np.maximum(n, 1)
        sd = np.sqrt(var)
        z = np.where(f & (sd > 1e-12), (x - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0)
        rows.append(z)
    return np.stack(rows, axis=1)


def crowding_neutralize(signal: np.ndarray, Gz: np.ndarray) -> np.ndarray:
    """signal (n, S) ⊥ 拥挤基 Gz (n, k, S) → 逐 bar 在 finite(signal) cell 上多因子残差。
    保 NaN 结构(非 finite → NaN);空截面/奇异 bar 由 ridge(数值边界)使解为 0。"""
    m = np.isfinite(signal)
    Gzm = Gz * m[:, None, :]
    GGt = np.einsum('tis,tjs->tij', Gzm, Gzm) + 1e-6 * np.eye(Gz.shape[1])      # (n,k,k)+ridge
    Gf = np.einsum('tis,ts->ti', Gzm, np.where(m, signal, 0.0))                # (n,k)
    c = np.linalg.solve(GGt, Gf[..., None])[..., 0]                           # (n,k)
    resid = signal - np.einsum('ti,tis->ts', c, Gz)
    return np.where(m, resid, np.nan)


# ============================================================================
# 1e. AFF 自适应融合(因果滚动 lstsq)—— 取代静态 ICIR/rank_ic 池组合
#
# 2026-06-26 实证:静态 IC/ICIR 加权对弱方向因子标定差(rank_ic/icir 是秩/标准化代理,非真回归系数);
# 多元 lstsq(真收益回归斜率)+ 时变重拟把方向池 gross 从 +1.0 拉到 +4.5(test)。两层:① 系数标定
# (frozen lstsq 已大胜静态代理)② 时变(test 上 regime 轮动靠重拟再翻倍)。代价=换手↑(权重在漂)。
# 全因果(只用 ≤t−lag 的已实现 (因子,y) 对拟合),warmup 前无仓。接 [[project_cost_erosion_bottleneck]]。
# ============================================================================

def aff_fuse(mz: np.ndarray, y_dec: np.ndarray, valid: np.ndarray,
             refit: int = 24, lag: int = 2, warmup: int = 480) -> np.ndarray:
    """mz (n, S, K) 成员信号面板(各成员 cs_z,方向池先 crowding_neutralize)→ 因果滚动 lstsq 融合 (n, S)。
    每 refit dec 在 ≤t−lag 的 valid∧finite (因子,y) 对上重拟多元系数;warmup 前 NaN(无仓);系数冻结直到下次重拟。
    lag=2 匹配 2h 前向视界(只用已实现 y);K+200 最小样本门。"""
    n, S, K = mz.shape
    out = np.full((n, S), np.nan)
    coef = None
    finy = np.isfinite(y_dec); finm = np.isfinite(mz).all(-1)
    for t in range(n):
        if t % refit == 0 and t >= warmup:
            cut = t - lag
            m = valid[:cut] & finm[:cut] & finy[:cut]
            if int(m.sum()) > K + 200:
                coef, *_ = np.linalg.lstsq(mz[:cut][m], y_dec[:cut][m], rcond=None)
        if coef is not None:
            out[t] = mz[t] @ coef
    return out


# ============================================================================
# 2. 组合层标量 vol-target 杠杆 + gross backstop
#
# 替代自由 lev_head 与一切 per-sym reweight。
#
# 实证结论(scripts/rl_dynlev_ablation + rl_voltarget_ablation,2026-05-25):
#   - **截面相对配置神圣不可动**:任何 per-sym reweight(自由 lev_head / 逆下行波动 /
#     tail-derisk / apply_sizing 的 liq+tail 收紧)都杀收益或过拟合。P1 alpha 走干净
#     cap0.20 → test 988x;走 apply_sizing(None+wcap0.1+liq+tail) → test 50x(砍 20x)。
#   - **杠杆只能是组合层一个标量**,按 vol_24h 择时缩放,均匀作用所有名 → 不破坏配置。
#     val/test 同步几何放大(比例 1.4–2.3,健康),不像旧 lev_head 撕裂(14.4 = 过拟合)。
#   - vol-target 价值主要在压尾部 MDD;趋势市里降 gross 反损收益,故 mult 是收益/MDD 前沿旋钮。
#
# scale_t = max(mult · target_vol / vol_24h_t, scale_min);target_vol 取 train 段 vol_24h 中位
# ([trade_signal] target_vol;≤0 → 不缩放,满仓收益最优锚)。gross_backstop 只缩(ruin 栏)。
# 上限 clip(原 scale_max=4.0)2026-06-05 删:k=4 时 82% dec 被它钳死成伪常数;上栏职责
# 归 gross_backstop([trade_signal] sizing_gross_cap,= 保证金可行线 0.9×account_leverage)。
# ============================================================================

@dataclass
class VolTargetConfig:
    mult:      float = 1.0     # 收益/MDD 前沿旋钮:<1 偏低杠杆压 MDD,>1 加杠杆冲收益
    scale_min: float = 0.25
    eps:       float = 1e-9


def vol_target_scale_np(vol24: np.ndarray, target_vol: float, cfg: VolTargetConfig) -> np.ndarray:
    """vol24 (T,) 绝对 portfolio_realized_vol_24h → 组合标量缩放 (T,)。"""
    return np.maximum(cfg.mult * target_vol / np.maximum(vol24, cfg.eps), cfg.scale_min)


def gross_backstop_np(w: np.ndarray, gross_cap: float) -> np.ndarray:
    """Σ|w| > gross_cap → 缩回(ruin 外栏,只缩不放;不改相对配置)。w (S,) 或 (T,S)。"""
    one_d = (w.ndim == 1)
    a = w[None, :] if one_d else w
    g = np.abs(a).sum(axis=1, keepdims=True)
    a = a * np.minimum(1.0, gross_cap / np.maximum(g, 1e-12))
    return a[0] if one_d else a
