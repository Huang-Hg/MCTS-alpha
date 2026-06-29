"""
eglatent — 极值图模型的稀疏+低秩凸分解(剥离共因的"精确版",周期性用)。

来源:Engelke & Taeb, "Extremal graphical modeling with latent variables via convex
optimization", JMLR 2025(arXiv:2403.09604);算法对照其 R 包 `graphicalExtremes`
源码(eglatent.R / matrix_transformations.R)逐式重写为 numpy + cvxpy。

它解决 `tail_descriptor.latent_drivers`(SVD 廉价版)解决不了的问题:把池的尾部依赖
精度矩阵分解为 Θ̃ = S − L,S 稀疏(剥掉共同崩盘驱动后的**真**条件尾独立图),
L 低秩 PSD(少数隐崩盘驱动,= corr→1 的真凶)。秩自动推断(不需指定隐变量数)。

工程定位(2026-06-25 代码级核对):样本复杂度 k ≳ p²log(p) 尾部越界样本 + 每次解一个
log-det 锥规划(p~30-80 几秒)→ **不能放每候选热路径,只能周期性在已接受池上跑**。
搜索内循环用 tail_descriptor 的 SVD 版;每 N 迭代/checkpoint 用本模块精算 S+L,
刷新隐崩盘驱动子空间。

下尾约定:HR 模型建在**上尾**(multivariate-Pareto,极值=大)。要捕"一起崩"须把净
PnL 取反(亏损→大正值)再喂 `emp_variogram(-net_pnl)`。

依赖:cvxpy + 锥求解器(SCS / Clarabel)。本模块顶层 import cvxpy(缺则大声崩);
smoke 默认不导入本模块(只在系统边界 try ImportError 选跑)。
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

_EPS = 1e-12


# ============================================================================
# 经验变差图 Γ̂  +  Γ→Θ
# ============================================================================

def _to_mpareto(data: np.ndarray, p: float) -> np.ndarray:
    """各列经验 CDF → 标准 Pareto X̃=1/(1−F̂),除以阈值后保留 ∞-范数>1 的极值行。
    data: (T, d)(极值=大;捕崩盘须先取反)。返回 (k, d),k=尾部样本数。"""
    T = data.shape[0]
    ranks = np.argsort(np.argsort(data, axis=0), axis=0) + 1      # 1..T 每列
    F = ranks / (T + 1.0)
    X = 1.0 / (1.0 - F)
    thr = 1.0 / (1.0 - p)
    Xn = X / thr
    return Xn[Xn.max(axis=1) > 1.0]


def _sigma2gamma(Sigma: np.ndarray) -> np.ndarray:
    diag = np.diag(Sigma)
    return diag[:, None] + diag[None, :] - 2.0 * Sigma


def emp_variogram(data: np.ndarray, p: float = 0.90) -> np.ndarray:
    """HR 经验变差图 Γ̂ = (1/d) Σ_m Sigma2Gamma( cov( log X̃[X̃[:,m]>1] ) )(Eq.7)。
    data: (T, d) 极值=大。返回 (d, d) 对称、零对角。"""
    X = _to_mpareto(data, p)
    d = X.shape[1]
    G = np.zeros((d, d), dtype=np.float64)
    for m in range(d):
        sub = np.log(X[X[:, m] > 1.0])
        G += _sigma2gamma(np.cov(sub, rowvar=False))
    return G / d


def gamma_to_theta(Gamma: np.ndarray) -> np.ndarray:
    """Θ = ( Π(−Γ/2)Π )⁺,Π = I − 11ᵀ/d(Definition 1)。Θ1=0,rank d−1。"""
    d = Gamma.shape[0]
    Pi = np.eye(d) - np.ones((d, d)) / d
    Sigma = Pi @ (-0.5 * Gamma) @ Pi
    return np.linalg.pinv(Sigma)


# ============================================================================
# 凸分解  Θ̃ = S − L
# ============================================================================

def fit_eglatent(Gamma_hat: np.ndarray, lam1: float = 0.15,
                 lam2: float = 2.0) -> dict:
    """解 eglatent 凸规划(Eq.9 / R 包目标):

        min_{P,S,L⪰0}  −logdet(Uᵀ P U) − ½ tr(P Γ̂) + lam1·( ‖S‖₁ + lam2·tr(L) )
        s.t.  P = S − L ,  UUᵀ P UUᵀ = P            (UUᵀ=Π,强制 P1=0)

    U = Π 的前 d−1 个左奇异向量。lam1=λ_n(整体正则),lam2=γ(隐变量惩罚相对权重)。
    返回 {S, L, P}(均 (d,d))。集成时按论文用 held-out HR 似然在 (lam1,lam2) 网格选优。"""
    d = Gamma_hat.shape[0]
    Pi = np.eye(d) - np.ones((d, d)) / d
    Uf, _, _ = np.linalg.svd(Pi)
    U = Uf[:, :d - 1]                                  # (d, d-1), U Uᵀ = Π

    P = cp.Variable((d, d), PSD=True)
    S = cp.Variable((d, d), PSD=True)
    L = cp.Variable((d, d), PSD=True)
    UUt = U @ U.T
    cons = [P == S - L, UUt @ P @ UUt == P]
    obj = (-cp.log_det(U.T @ P @ U) - 0.5 * cp.trace(P @ Gamma_hat)
           + lam1 * (cp.sum(cp.abs(S)) + lam2 * cp.trace(L)))
    cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.SCS)
    return {'S': S.value, 'L': L.value, 'P': P.value}


# ============================================================================
# 读出多样性:隐崩盘载荷(从 L) + 真条件尾图(从 S)
# ============================================================================

def latent_loadings(L: np.ndarray, thr: float = 1e-3):
    """L 特征分解 → (V_r, eigvals):rk 个特征值≥thr 的方向 = 隐崩盘驱动载荷。
    V_r: (d, rk) 列;每列是各因子对该隐驱动的载荷。rk 自动推断(非超参)。"""
    w, V = np.linalg.eigh(L)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    rk = int(np.sum(w >= thr))
    return np.ascontiguousarray(V[:, :rk]), w[:rk]


def sparse_graph(S: np.ndarray, thr: float = 1e-3) -> np.ndarray:
    """|S_ij|>thr → 边:剥掉隐崩盘驱动后**仍**条件尾相依的因子对(真特异尾连接)。
    返回 (d,d) 0/1 邻接(零对角)。在 S 图里孤立(度=0)= 真尾去相关。"""
    A = (np.abs(S) > thr).astype(np.int64)
    np.fill_diagonal(A, 0)
    return A


def member_diversity_scores(S: np.ndarray, L: np.ndarray, thr: float = 1e-3) -> np.ndarray:
    """每个池成员的尾部多样性分 = residual_latent × 1/(1+deg_S)(论文推荐合成式)。
    residual_latent_i = 1 − 该成员在隐载荷上的能量占比;deg_S_i = S 图度数。
    高分 = 既非隐驱动 clone、又无特异尾连接 = 真尾去相关。"""
    V_r, _ = latent_loadings(L, thr)
    A = sparse_graph(S, thr)
    d = S.shape[0]
    deg = A.sum(axis=1)
    if V_r.shape[1] == 0:
        res = np.ones(d)
    else:
        # 各因子在隐子空间的能量占比(行投影);L 对角能量近似总尾能量。
        proj_energy = (V_r ** 2).sum(axis=1)
        res = 1.0 - proj_energy / (proj_energy.max() + _EPS)
    return res / (1.0 + deg)
