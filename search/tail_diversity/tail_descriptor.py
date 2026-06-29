"""
尾部条件 / 净成本 / 剥离共因 的多样性描述子(真多样 vs 伪多样)。

动机(见 [[project_cost_erosion_bottleneck]] + 2026-06-25 文献调研):
    现 `rl/alpha_pool.py` 的多样性度量 = **全样本 Pearson R²(per_t_pnl)+ gross 口径**:
        R² = max_k pearson(cand_pnl, member_k_pnl)²    (LinearAlphaPool._proj_r2)
    这是数学上的"伪多样"——两个因子平时(全样本)去相关,崩盘里却一起死(corr→1)。
    实证:Mallela & Leonelli 2026(arXiv 2606.16840)证加密下尾依赖图崩盘时趋近全连通,
    高斯/协方差口径低估联合崩盘概率 ~8×。

本模块把"多样性"重建在三根支柱上(全 numpy,零 C 扩展依赖,零 load_panel):
    ① 尾部条件:只在最差 q% bar(crash 窗)上度相关,而非全样本。
    ② 净成本口径:per_t_pnl 传入净值(gross − funding carry − turnover·fee),
       否则会在"成本轴"上虚报多样性 —— 正是 net 归零的那条轴。
    ③ 剥离共因:对池的尾部签名做 SVD,取 top-r 奇异方向 = 隐崩盘驱动;候选对其
       投影残差 = 真新颖尾部行为(只两两 χ 低还不够,要扣掉共同隐驱动)。
    eglatent(`eglatent.py`)是 ③ 的"周期性精确版"(稀疏+低秩 HR 分解);本模块给
    SVD 廉价版供搜索内循环每候选用。

no-lookahead 说明:tail_mask 的 market_loss 指标在**评估窗内 post-hoc** 算(与 IC 同源,
    研究期度量,不做前向交易决策)。若挪到 live 因果门,指标须改为滚动因果版。见 [[feedback_no_lookahead]]。
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


# ============================================================================
# 尾部窗
# ============================================================================

def tail_mask(market_loss: np.ndarray, q: float = 0.10) -> np.ndarray:
    """最差 q% bar 的布尔掩码(crash 窗)。

    market_loss: (T,) 市场压力指标,**越小越糟**(如等权 universe / BTC 的 bar 收益,
        或聚合已实现损失的相反数)。𝒯 = { t : market_loss[t] ≤ quantile_q }。
    返回 (T,) bool。"""
    thr = np.nanquantile(market_loss, q)
    return market_loss <= thr


# ============================================================================
# 尾部签名 + 尾部条件相关(廉价,搜索热路径)
# ============================================================================

def tail_signature(per_t_pnl_net: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """因子在 crash 窗上的标准化净 PnL 签名,(m,) where m=|𝒯|。
    NaN→0(空仓 bar);零方差 → 全零(无尾部行为)。"""
    sig = np.where(np.isfinite(per_t_pnl_net), per_t_pnl_net, 0.0)[mask]
    sd = sig.std()
    if sd < _EPS:
        return np.zeros_like(sig)
    return (sig - sig.mean()) / sd


def tail_corr_vec(cand_sig: np.ndarray, member_sigs: np.ndarray) -> np.ndarray:
    """候选尾部签名 vs 各池成员尾部签名的 Pearson,(n,)。
    cand_sig: (m,) 已标准化;member_sigs: (n, m) 各行已标准化。空池 → (0,)。
    (镜像 ops.pnl_corr_vec 但限定在 crash 窗 + 纯 numpy。)"""
    n = member_sigs.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    m = cand_sig.shape[0]
    # 标准化签名的 pearson = 点积 / m(各自已零均值单位方差);零向量行 → 0。
    return (member_sigs @ cand_sig) / max(m, 1)


def proj_r2(c: np.ndarray) -> float:
    """冗余度 R² = max_k c_k²(与"最像单成员"的平方相关)。空 → 0。
    与 LinearAlphaPool._proj_r2 同口径,但 c 来自 tail_corr_vec(crash 窗)。"""
    if c.size == 0:
        return 0.0
    return float(min(np.max(c * c), 1.0))


# ============================================================================
# 隐崩盘驱动 + 投影残差(剥离共因,SVD 廉价版)
# ============================================================================

def latent_drivers(member_sigs: np.ndarray, rank: int = 3,
                   energy: float = 0.90) -> np.ndarray:
    """池尾部签名的 top-r 奇异方向 = 隐崩盘驱动(时间-签名空间,长度 m)。

    member_sigs: (n, m) 各行已标准化尾部签名。返回 V_r: (m, r_eff),正交列。
    r_eff = min(rank, 满足累计能量≥energy 的奇异值数, n)。n=0 → (m,0)。
    这对应 eglatent 的低秩块 L,但用廉价 SVD(无凸优化);搜索内循环用此,
    周期性(每 N 迭代)再用 eglatent.py 精算 S+L。"""
    n = member_sigs.shape[0]
    if n == 0:
        return np.zeros((member_sigs.shape[1] if member_sigs.ndim == 2 else 0, 0))
    # SVD: member_sigs (n,m) = U Σ Vt;V 行 = 时间方向。取能解释最多崩盘方差的方向。
    U, S, Vt = np.linalg.svd(member_sigs, full_matrices=False)
    if S.sum() < _EPS:
        return np.zeros((member_sigs.shape[1], 0))
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    r_energy = int(np.searchsorted(cum, energy) + 1)
    r_eff = min(rank, r_energy, n)
    return np.ascontiguousarray(Vt[:r_eff].T)        # (m, r_eff)


def latent_residual(cand_sig: np.ndarray, drivers: np.ndarray) -> float:
    """候选尾部签名扣掉隐崩盘驱动后的残差能量占比 ∈ [0,1]。
    1.0 = 与所有已知崩盘驱动正交(真新颖尾部行为);0.0 = 纯属又一个共驱动 clone
    (即便两两 χ 不高也会被抓出)。drivers: (m, r) 正交列;无驱动 → 残差=1。"""
    norm2 = float(cand_sig @ cand_sig)
    if norm2 < _EPS:
        return 0.0                                   # 无尾部行为 → 不算多样
    if drivers.shape[1] == 0:
        return 1.0
    proj = drivers.T @ cand_sig                       # (r,)
    return float(max(0.0, 1.0 - (proj @ proj) / norm2))


# ============================================================================
# 轴 B — funding-成本流 co-exposure(真数据 §4b 证实的主问题轴)
#   成本是慢漂移、逐 bar << 收益波动 → net 逐 bar 收益 ≈ gross,收益相关里看不到 funding
#   (rankconv 池 val 实测:收益伪多样 0/276 vs funding 30/276,funding 占成本相关 97%)。
#   共享 funding 暴露活在成本流本身:cost_i[t]=gross_ret_i−net_ret_i(≈纯 funding)。
#   度量复用尾部签名/相关原语(tail_signature/tail_corr_vec/proj_r2),只是输入换成成本流。
# ============================================================================

def cost_stream(gross_ret: np.ndarray, net_ret: np.ndarray) -> np.ndarray:
    """逐 bar 成本流 = gross_ret − net_ret(≈纯 funding,实测占成本相关 97%)。
    集成时由 run_bt(with_equity) gross(funding=0/fee=0)与 net 双跑取 equity 逐 bar 收益之差。
    签名走 tail_signature(同收益轴),crash 窗标准化。"""
    return np.asarray(gross_ret, dtype=np.float64) - np.asarray(net_ret, dtype=np.float64)


def cost_corr_vec(cand_cost_sig: np.ndarray, member_cost_sigs: np.ndarray) -> np.ndarray:
    """候选成本流尾部签名 vs 各成员的 Pearson,(n,)(= tail_corr_vec,成本轴语义命名)。
    cand_cost_sig / member_cost_sigs 均为 tail_signature(cost_stream, mask) 的输出(已标准化)。"""
    return tail_corr_vec(cand_cost_sig, member_cost_sigs)


def cost_r2_tail(cand_cost_sig: np.ndarray, member_cost_sigs: np.ndarray) -> float:
    """Rc²_tail = max_k pearson(cost_cand, cost_k)²(crash 窗):候选与"最像成员"的共享 funding 暴露。
    高 = 收益上看着分散、却和某成员一起被 funding 薅(伪多样的 funding 轴)。空池 → 0。"""
    return proj_r2(cost_corr_vec(cand_cost_sig, member_cost_sigs))


# ============================================================================
# 组合多样性分(搜索 reward / 准入 Δ:两轴 = 收益尾部 + 共因 + funding-成本流)
# ============================================================================

def tail_diversity_delta(quality: float, cand_sig: np.ndarray,
                         member_sigs: np.ndarray, drivers: np.ndarray,
                         cand_cost_sig: np.ndarray = None,
                         member_cost_sigs: np.ndarray = None) -> float:
    """两轴正交贡献 Δ,drop-in 替换 LinearAlphaPool 的 Δ=|IC|·(1−R²)。

        Δ_tail = quality · (1 − R²_tail) · latent_residual · (1 − Rc²_tail)
                 └质量┘ └轴A 收益崩盘共动┘ └轴A 共同隐驱动┘  └★轴B funding-成本流┘

    quality = 质量幅度(|rank IC| 或净 CVaR 幅值)。
    轴 A(必传):cand_sig/member_sigs = 收益 tail_signature;drivers = latent_drivers。
    轴 B(可选,真数据证为主问题轴):传 cand_cost_sig/member_cost_sigs = 成本流 tail_signature 才计;
      省略 → 退回纯收益轴(向后兼容)。空池下各项自动→1,不影响。"""
    r2 = proj_r2(tail_corr_vec(cand_sig, member_sigs))
    res = latent_residual(cand_sig, drivers)
    delta = quality * (1.0 - r2) * res
    if cand_cost_sig is not None and member_cost_sigs is not None:
        delta *= (1.0 - cost_r2_tail(cand_cost_sig, member_cost_sigs))
    return float(delta)


# ============================================================================
# fake-vs-real 验收测试(整个方案的护栏 —— 文献无人做)
# ============================================================================

def fake_vs_real_collapse(member_pnls_net: np.ndarray, market_loss: np.ndarray,
                          q: float = 0.10) -> dict:
    """对一组因子净 PnL,对照"全样本 R²"vs"尾部 R²",量化伪多样塌缩。

    member_pnls_net: (n, T) 各因子净 per_t_pnl;market_loss: (T,)。
    返回 {full_offdiag_mean, tail_offdiag_mean, collapse}:
        collapse = tail − full 的平均 off-diagonal |corr| 提升;>0 即"平时分散、崩盘共动"
        = 伪多样的强度。真多样池此值 ≈ 0。"""
    n = member_pnls_net.shape[0]
    mask = tail_mask(market_loss, q)
    full = np.stack([_std_full(p) for p in member_pnls_net])
    tail = np.stack([tail_signature(p, mask) for p in member_pnls_net])
    Cf = np.abs(np.corrcoef(full))
    Ct = np.abs(np.corrcoef(tail))
    iu = np.triu_indices(n, k=1)
    full_m = float(np.nanmean(Cf[iu]))
    tail_m = float(np.nanmean(Ct[iu]))
    return {'full_offdiag_mean': full_m, 'tail_offdiag_mean': tail_m,
            'collapse': tail_m - full_m}


def _std_full(per_t_pnl_net: np.ndarray) -> np.ndarray:
    """全样本标准化净签名(NaN→0),供 fake-vs-real 对照用。"""
    sig = np.where(np.isfinite(per_t_pnl_net), per_t_pnl_net, 0.0)
    sd = sig.std()
    if sd < _EPS:
        return np.zeros_like(sig)
    return (sig - sig.mean()) / sd
