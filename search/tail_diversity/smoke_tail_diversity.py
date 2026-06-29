"""smoke:尾部条件多样性描述子 + DNS archive 的 fake-vs-real 验收(2026-06-25)。

自包含、纯 numpy、零 load_panel(合成可控 per_t_pnl)。证明核心命题:
    全样本 Pearson R²(现 LinearAlphaPool 口径)把"平时分散、崩盘共动"的伪多样因子
    误判为多样;尾部条件 + 剥离共因描述子把它们抓出来;DNS 在等质量下保真多样、剔伪多样。

合成设定:
    - crash 窗 = 最差 PHI 比例 bar(market_loss 控)。
    - FAKE 因子:独立 idiosyncratic 噪声 + 共享崩盘 shock(只在 crash 窗,所有 fake 共用一个 s)
      → 全样本两两相关低,但 crash 窗里一起死(corr→1)= 伪多样。
    - REAL 因子:纯独立噪声(crash 窗里也独立)= 真多样。

跑:/mnt/d/03learn/machine/.venv/Scripts/python.exe search/tail_diversity/smoke_tail_diversity.py
   (或任意装了 numpy 的解释器;eglatent 精确路径需 cvxpy,缺则自动跳过。)
"""

from __future__ import annotations

import numpy as np

from tail_descriptor import (
    tail_mask, tail_signature, tail_corr_vec, proj_r2,
    latent_drivers, latent_residual, fake_vs_real_collapse, _std_full,
    cost_r2_tail, tail_diversity_delta,
)
from dns_archive import DNSArchive

RNG = np.random.default_rng(0)
T = 4000
PHI = 0.04                         # crash 窗比例
N_CRASH = int(T * PHI)             # 160
R2_CAP = 0.5                       # = config [alpha_pool] r2_cap(现池冗余门)


def check(name, cond):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    assert cond, name


# ---- 合成 crash 窗 + 共享崩盘驱动 ----
crash_idx = RNG.choice(T, N_CRASH, replace=False)
crash = np.zeros(T, dtype=bool)
crash[crash_idx] = True
market_loss = RNG.standard_normal(T)
market_loss[crash] = -10.0 - np.abs(RNG.standard_normal(N_CRASH))
s_shared = RNG.standard_normal(N_CRASH)        # 所有 fake 共用的崩盘 shock(每 crash bar 一个)

MASK = tail_mask(market_loss, PHI)


def make_fake() -> np.ndarray:
    pnl = 0.0005 + 0.03 * RNG.standard_normal(T)       # 独立 idiosyncratic
    shock = np.zeros(T)
    shock[crash] = 0.06 * s_shared                      # 共享崩盘 shock
    return pnl + shock


def make_real() -> np.ndarray:
    return 0.0005 + 0.03 * RNG.standard_normal(T)       # 纯独立(crash 里也独立)


# ============================================================================
print("== 1. 全样本 R² 被骗:伪多样因子全样本看起来分散(< r2_cap)==")
fake_pool = [make_fake() for _ in range(8)]
fake_full = np.stack([_std_full(p) for p in fake_pool])
Cf = np.abs(np.corrcoef(fake_full))
iu = np.triu_indices(8, k=1)
full_max_r2 = float((Cf[iu] ** 2).max())
print(f"     fake 池 全样本 max off-diag R² = {full_max_r2:.4f}")
check("全样本 max R² < r2_cap(现池会当多样准入)", full_max_r2 < R2_CAP)


# ============================================================================
print("== 2. 尾部 R² 抓出共动:同一批 fake 在 crash 窗里 corr→1 ==")
fake_sigs = np.stack([tail_signature(p, MASK) for p in fake_pool])
# 新 fake 候选(共享同一 s)对现 fake 池的尾部 R²
cand_fake = make_fake()
cf_sig = tail_signature(cand_fake, MASK)
r2_tail_fake = proj_r2(tail_corr_vec(cf_sig, fake_sigs))
# 对照:该候选的全样本 R²(应低 → 现池放行)
r2_full_fake = proj_r2((fake_full @ _std_full(cand_fake)) / T)
print(f"     新 fake 候选: 全样本 R²={r2_full_fake:.4f}  尾部 R²={r2_tail_fake:.4f}")
check("新 fake 全样本 R² < r2_cap(现池放行=被骗)", r2_full_fake < R2_CAP)
check("新 fake 尾部 R² > r2_cap(尾部口径正确拒绝)", r2_tail_fake > R2_CAP)


# ============================================================================
print("== 3. 真多样对照:REAL 因子尾部 R² 仍低 ==")
real_pool = [make_real() for _ in range(8)]
real_sigs = np.stack([tail_signature(p, MASK) for p in real_pool])
cand_real = make_real()
cr_sig = tail_signature(cand_real, MASK)
r2_tail_real = proj_r2(tail_corr_vec(cr_sig, real_sigs))
print(f"     新 real 候选: 尾部 R²={r2_tail_real:.4f}")
check("新 real 尾部 R² < r2_cap(真多样不误杀)", r2_tail_real < R2_CAP)


# ============================================================================
print("== 4. collapse 度量:tail−full off-diag 提升(伪多样强度)==")
col_fake = fake_vs_real_collapse(np.stack(fake_pool), market_loss, PHI)
col_real = fake_vs_real_collapse(np.stack(real_pool), market_loss, PHI)
print(f"     fake collapse={col_fake['collapse']:.3f}  real collapse={col_real['collapse']:.3f}")
check("fake collapse > 0.3(平时分散崩盘共动)", col_fake['collapse'] > 0.3)
check("real collapse < 0.15(真多样无塌缩)", col_real['collapse'] < 0.15)


# ============================================================================
print("== 5. 剥离共因:隐崩盘驱动投影残差(两两都不高也能抓)==")
drivers = latent_drivers(fake_sigs, rank=3)          # 从 fake 池学隐崩盘驱动
res_new_fake = latent_residual(cf_sig, drivers)      # 新 fake → 又一个共驱动 clone → 低
res_new_real = latent_residual(cr_sig, drivers)      # 新 real → 与崩盘驱动正交 → 高
print(f"     隐驱动数={drivers.shape[1]}  res(new fake)={res_new_fake:.3f}  res(new real)={res_new_real:.3f}")
check("新 fake 残差 < 0.45(被识别为共驱动冗余)", res_new_fake < 0.45)
check("新 real 残差 > 0.60(真新颖尾部行为)", res_new_real > 0.60)


# ============================================================================
print("== 6. DNS archive:等质量下保真多样、剔伪多样(高维学习描述子)==")
# 描述子 = 尾部签名本身(高维 m~160;DNS 的招牌用法)。2 real + 6 fake,质量全相等。
genos = [f'real{i}' for i in range(2)] + [f'fake{i}' for i in range(6)]
descs = np.stack([tail_signature(make_real(), MASK) for _ in range(2)]
                 + [tail_signature(make_fake(), MASK) for _ in range(6)])
fitness = np.ones(len(genos))                         # 等质量 → 纯按多样性选
arch = DNSArchive(capacity=4, k=3)
dropped = arch.add_batch(genos, fitness, descs)
print(f"     保留={sorted(arch.genos)}  淘汰={sorted(dropped)}")
check("两个 real 都存活(真多样占独立 niche)",
      'real0' in arch.genos and 'real1' in arch.genos)
check("淘汰的全是 fake(伪多样聚簇被剔)", all(g.startswith('fake') for g in dropped))


# ============================================================================
print("== 7. (可选)eglatent 精确路径:稀疏+低秩 HR 分解 ==")
try:
    import eglatent as eg
except ImportError:
    print("     [SKIP] 未装 cvxpy → 跳过精确路径(SVD 廉价版已验)")
else:
    # 6 因子(4 fake + 2 real);HR 建上尾 → 取反净 PnL(亏损→大)。
    pool6 = [make_fake() for _ in range(4)] + [make_real() for _ in range(2)]
    data = -np.stack(pool6, axis=1)                   # (T, 6) 极值=大
    Gamma = eg.emp_variogram(data, p=0.90)
    fit = eg.fit_eglatent(Gamma, lam1=0.15, lam2=2.0)
    V_r, eigs = eg.latent_loadings(fit['L'])
    scores = eg.member_diversity_scores(fit['S'], fit['L'])
    print(f"     隐崩盘驱动 rank={V_r.shape[1]}  eig={np.round(eigs, 4)}")
    print(f"     多样性分 fake4={np.round(scores[:4], 3)}  real2={np.round(scores[4:], 3)}")
    print(f"     mean(real)={scores[4:].mean():.3f}  mean(fake)={scores[:4].mean():.3f}"
          " (real 应 ≥ fake;print-only,不 assert solver 行为)")


# ============================================================================
print("== 8. 轴 B funding-成本流:收益分散但 funding 共担 → 轴 B 抓 + 两轴 Δ 压低 ==")
s_fund = RNG.standard_normal(N_CRASH)            # 所有"共担"因子共用的 crash 窗 funding spike


def cost_shared() -> np.ndarray:
    c = 1e-4 * RNG.standard_normal(T)
    c[crash] += 0.02 * s_fund                    # 共享 funding 成本(crash 窗)
    return c


def cost_indep() -> np.ndarray:
    c = 1e-4 * RNG.standard_normal(T)
    c[crash] += 0.02 * RNG.standard_normal(N_CRASH)   # 独立 funding 成本
    return c


# 池:收益独立(轴A 干净),但 funding 成本共担
ret_pool_sigs = np.stack([tail_signature(make_real(), MASK) for _ in range(8)])
cost_pool_sigs = np.stack([tail_signature(cost_shared(), MASK) for _ in range(8)])
cand_ret_sig = tail_signature(make_real(), MASK)             # 收益分散的候选
cand_cost_shared = tail_signature(cost_shared(), MASK)       # 同一候选:funding 共担版
cand_cost_indep = tail_signature(cost_indep(), MASK)         # 同一候选:funding 独立版

r2A = proj_r2(tail_corr_vec(cand_ret_sig, ret_pool_sigs))
rcB_shared = cost_r2_tail(cand_cost_shared, cost_pool_sigs)
rcB_indep = cost_r2_tail(cand_cost_indep, cost_pool_sigs)
print(f"     轴A 收益 R²_tail={r2A:.3f}  轴B funding Rc²(shared)={rcB_shared:.3f}  Rc²(indep)={rcB_indep:.3f}")
check("收益轴看着分散(R²_tail<0.5)", r2A < 0.5)
check("funding 共担候选被轴B抓(Rc²>0.5)", rcB_shared > 0.5)
check("funding 独立候选轴B放行(Rc²<0.5)", rcB_indep < 0.5)

drivers_ret = latent_drivers(ret_pool_sigs, rank=3)
d_shared = tail_diversity_delta(1.0, cand_ret_sig, ret_pool_sigs, drivers_ret, cand_cost_shared, cost_pool_sigs)
d_indep = tail_diversity_delta(1.0, cand_ret_sig, ret_pool_sigs, drivers_ret, cand_cost_indep, cost_pool_sigs)
print(f"     两轴 Δ: funding共担={d_shared:.3f}  funding独立={d_indep:.3f}(同收益同质量,仅 funding 异)")
check("funding 共担候选 Δ 被显著压低(< 0.5×独立)", d_shared < 0.5 * d_indep)


print("\nALL SMOKE PASSED")
