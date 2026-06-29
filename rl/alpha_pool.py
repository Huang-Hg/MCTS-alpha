"""
LinearAlphaPool — AlphaCFG paper-style 因子池:质量口径统一为标准 rank IC(2026-06-22)。

admission / 留池:质量幅度 q = 候选 **|rank IC|**(标准秩 IC 幅值)。正交贡献 Δ = q·(1−R²),
  R² = max_k corr(cand, member_k)²(候选 per_t_pnl 与"最像的单成员"的平方 Pearson,度行为多样性)。三道门:
    ① 近克隆门:R² > r2_cap 且候选 |rank_ic| ≤ 最像成员 → 拒(更弱的近重复;更强则放行,留给 prune 替换);
    ② 正交贡献门:Δ < delta_floor → 拒(正交贡献太弱);
    ③ OOS holdout:末 holdout_frac 段 fit 方向不盈利 → 拒。
  每次 admit 后 leave-one-out 重算 prune:冗余簇(R²_loo>r2_cap)非空 → 踢簇内 |rank_ic| 最小者
  (留最强代表);否则超容量 / Δ_loo<floor → 踢 argmin Δ_loo(= q·(1−R²_loo))
  → 更强候选挤掉弱钉子户,连续自清洁。池**不定权**(deploy/eval/live 全走 AFF 因果滚动融合,见 rl.sizing.aff_fuse)。

per_t_pnl(evaluator 里 ops.per_t_pnl 算):v = values − cs_mean;w = v/Σ|v|;per_t_pnl[t] = Σ_s w[t,s]·y[t,s]
  (cs-demean + L1=1 normalize;仅供 R² 正交度量 + OOS holdout,质量幅度走 |rank IC|)。

gp obj1(搜索 reward)= 候选 |rank IC|(evaluator 侧只算候选自身、不读池);两者口径一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from config.config import ini
from backtest import ops
from evaluation.ast import AlphaTree
from markets import CALENDAR as _CAL


# ============================================================================
# Pool member
# ============================================================================

@dataclass
class PoolMember:
    tree:       AlphaTree
    rank_ic:    float                          # 带符号秩 IC;质量口径(admission/prune 用 |rank_ic|)+ bundle 记录
    embedding:  Optional[np.ndarray] = None    # (H,) tree-lstm root embedding,FailCache / G2 用
    per_t_pnl:  Optional[np.ndarray] = None    # (T,) per-bar L1-normalized signed PnL,R² 正交度量用


# ============================================================================
# LinearAlphaPool
# ============================================================================

@dataclass
class LinearAlphaPool:
    capacity: int = 16
    # 最小正交贡献 Δ = |IC|·(1−R²):admission/prune 门 = Δ < delta_floor → reject/evict。
    delta_floor:       float = ini('alpha_pool', 'delta_floor',       0.001)
    # 冗余硬门(与 |IC| 解耦):R² > r2_cap → reject / evict,强冗余强 alpha 也踢。
    r2_cap:            float = ini('alpha_pool', 'r2_cap',            0.5)
    # OOS holdout 边际门:末 holdout_frac 比例 per_t_pnl bars 作嵌入式 holdout,候选用 fit 段估的
    # 部署方向必须在 holdout 段真盈利才入池(挡正交+有 |IC| 但 OOS 退化的过度生长)。
    holdout_frac:      float = ini('alpha_pool', 'holdout_frac',      0.25)
    # live panel 物理深度(= trade_panel.cold_load_days × hours_per_day,1h bars):required_depth
    # 超此的树上线后每次决策整树 NaN→哑火,结构性硬拒。hours_per_day 由 active MarketProfile 派生(crypto 24)。
    max_panel_depth:   int   = ini('trade_panel', 'cold_load_days', 21) * int(_CAL.hours_per_day)

    members:  List[PoolMember] = field(default_factory=list)
    # 平行数组(与 members 同序)。single_q = 各成员 |rank IC|(admission/prune 质量幅度)。
    # 池不再定权(deploy/eval/live 全走 AFF 因果滚动融合,见 rl.sizing.aff_fuse)。
    single_q:   np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    # PnL Pearson 矩阵(对角 1,off-diagonal = pearson(per_t_pnl_i, per_t_pnl_j))
    pnl_corrs:  np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    _failure_cache: Set[int] = field(default_factory=set, repr=False)
    # 由 main.py 在构造 net 后注入。签名:List[AlphaTree] -> np.ndarray (n, H)
    embed_fn: Optional[Callable[[List[AlphaTree]], np.ndarray]] = None

    # ----- 基本 query -----

    @property
    def size(self) -> int:
        return len(self.members)

    def to_jsonable(self) -> List[Dict]:
        out = []
        for m in self.members:
            out.append({
                'tree':       m.tree.to_dict(),
                'pretty':     m.tree.pretty(),
                'rank_ic':    m.rank_ic,
            })
        return out

    # ----- embedding helpers(evaluator G2 / FailCache 用)-----

    def candidate_embedding(self, tree: AlphaTree) -> Optional[np.ndarray]:
        if self.embed_fn is None:
            return None
        return self.embed_fn([tree])[0]

    def max_cos_sim_with(self, cand_emb: np.ndarray) -> float:
        if not self.members or self.embed_fn is None:
            return 0.0
        member_embs = [m.embedding for m in self.members if m.embedding is not None]
        if not member_embs:
            return 0.0
        M = np.stack(member_embs)
        nc = cand_emb / (np.linalg.norm(cand_emb) + 1e-12)
        nM = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return float(np.max(nM @ nc))

    def refresh_embeddings(self) -> None:
        if self.embed_fn is None or not self.members:
            return
        embs = self.embed_fn([m.tree for m in self.members])
        for m, e in zip(self.members, embs):
            m.embedding = e

    # ----- 正交 Δ 度量(reward-无关准入)-----

    def _corr_with_pool(self, cand_per_t_pnl: np.ndarray) -> np.ndarray:
        """c = [pearson(cand, member_k)]ₖ,(n,) 向量(NaN-safe,finite<30→0)。C 内核 ops.pnl_corr_vec
        (OMP 跨成员,worker 热路径每候选都调)。空池 → (0,)。"""
        n = self.size
        if n == 0:
            return np.empty(0, dtype=np.float64)
        members = np.ascontiguousarray(
            np.stack([m.per_t_pnl for m in self.members]), dtype=np.float64)
        return ops.pnl_corr_vec(
            np.ascontiguousarray(cand_per_t_pnl, dtype=np.float64), members)

    @staticmethod
    def _proj_r2(c: np.ndarray) -> float:
        """冗余度 R² = max_k c_k²(pairwise PnL 相关):候选 per_t_pnl 与"任一单成员"的最大平方
        Pearson。clone(c 某项=±1)→ R²=1、与任一成员都不像(max|c|=0)→ R²=0。空池(c 空)→ 0。"""
        if c.size == 0:
            return 0.0
        return float(min(np.max(c * c), 1.0))

    def _leave_one_out_r2(self) -> np.ndarray:
        """每个成员的 leave-one-out R²_i = 成员 i 对"其余 n-1 个成员"的最大平方 PnL 相关
        = 把成员 i 当新候选,对其余成员重算一遍 admission R²。n≤1 → 0。"""
        n = self.size
        out = np.zeros(n, dtype=np.float64)
        if n <= 1:
            return out
        M = self.pnl_corrs
        for i in range(n):
            others = [k for k in range(n) if k != i]
            out[i] = self._proj_r2(M[i, others])
        return out

    def _leave_one_out_deltas(self) -> np.ndarray:
        """每个成员的 leave-one-out Δ_i = |rank_ic|_i·(1−R²_loo_i):把成员 i 当新候选,对其余成员
        重算 admission Δ。小 = IC 弱 / 冗余 → prune 优先踢。single_q 存 |rank_ic|≥0;n≤1 → R²=0 → Δ=|rank_ic|。"""
        return self.single_q * (1.0 - self._leave_one_out_r2())

    # ----- admission(commit-then-revert)-----

    def try_new(
        self, cand_tree: AlphaTree, cand_rank_ic: float, cand_per_t_pnl: np.ndarray,
        cand_quality: float = None,
    ) -> Tuple[bool, float]:
        """commit-then-revert admission,三道前置门:近克隆门 + 正交 Δ 门 + OOS holdout。
        返回 (accepted, post_ensemble_ic)。cand_per_t_pnl 必传(R² 正交度量 + 切 fit/holdout 验 OOS 盈利);
        cand_rank_ic = 候选带符号秩 IC(落 PoolMember.rank_ic / bundle,供质量幅度与参考);
        cand_quality = 质量幅度源(admission/prune 的 single_q = |cand_quality|;None → 回落 |rank_ic|。
          gross-aware 搜索传 |top-K gross Sharpe| → 池按 Sharpe 质量留/剔,治 IC↔PnL 背离)。
        池不定权(deploy/eval/live 全走 AFF 融合)。"""
        q = abs(cand_quality) if cand_quality is not None else abs(cand_rank_ic)   # 质量幅度
        sig = cand_tree.hash
        if sig in self._failure_cache:
            return False, 0.0
        # 深度硬门:嵌套窗+operand warmup 超 live panel 物理深度的树,离线 T 充裕评得动、
        # live 上每次决策整树 NaN(深度死)→ 结构性拒,不进池。
        if cand_tree.required_depth() > self.max_panel_depth:
            self._failure_cache.add(sig)
            return False, 0.0

        # 多样性门(近克隆 cap + 正交贡献 Δ):c = corr 向量(顺带给下方 tentative-admit 扩 M 用)。
        # R² = max_k c_k²(候选 vs "最像成员"的平方 PnL 相关);Δ = |rank_ic|·(1−R²):IC 强且与任一
        # 成员都不像 = 大;clone(R²→1)→ Δ→0。质量幅度 q = |rank IC|。
        n = self.size
        pnl_corrs_new = self._corr_with_pool(cand_per_t_pnl)     # rho:候选 vs 各成员 pnl 相关
        r2 = self._proj_r2(pnl_corrs_new)
        # 近克隆门:R²>cap 且候选 |rank_ic| ≤ 最像成员 → 拒(更弱的近重复);候选更强
        #   则放行(prune 会踢掉被它克隆的弱成员)。空池 n=0 → r2=0,不触发。
        if r2 > self.r2_cap:
            k_star = int(np.argmax(pnl_corrs_new * pnl_corrs_new))
            if q <= self.single_q[k_star]:
                self._failure_cache.add(sig)
                return False, 0.0
        # 正交贡献门(Δ<floor,池没满也拒;门在 capacity prune 之前)→ 拒
        delta = q * (1.0 - r2)                                    # q = |rank_ic| ≥ 0
        if delta < self.delta_floor:
            self._failure_cache.add(sig)
            return False, 0.0

        # OOS holdout 边际门:per_t_pnl 切 [fit | holdout(末 holdout_frac)],方向只用 fit 段估
        # (holdout 严格 OOS)。线性 ensemble 下"候选对池 holdout 均值回报的边际贡献"= 自身 signed
        # holdout PnL(池其余项不变)→ sign(fit 均值)·(holdout 均值) > 0 才入。挡正交但 OOS 退化。
        h0 = int(cand_per_t_pnl.shape[0] * (1.0 - self.holdout_frac))
        sign_fit = np.sign(np.nanmean(cand_per_t_pnl[:h0]))
        if not (sign_fit * np.nanmean(cand_per_t_pnl[h0:]) > 0.0):
            self._failure_cache.add(sig)
            return False, 0.0

        # tentative admit(先无条件 commit,prune loop 再裁谁留)
        cand_emb = self.candidate_embedding(cand_tree)
        new_member = PoolMember(
            tree=cand_tree, rank_ic=cand_rank_ic,
            embedding=cand_emb, per_t_pnl=cand_per_t_pnl.astype(np.float64, copy=True),
        )
        cand_idx = n                               # 候选下标(prune 中随删随减)
        self.members.append(new_member)
        self.single_q = np.concatenate([self.single_q, [q]])
        new_M = np.empty((n + 1, n + 1), dtype=np.float64)
        new_M[:n, :n] = self.pnl_corrs
        new_M[:n, n] = pnl_corrs_new
        new_M[n, :n] = pnl_corrs_new
        new_M[n, n] = 1.0
        self.pnl_corrs = new_M

        # 连续 net-aware prune(2026-06-22):不论满没满,leave-one-out 当留池判据,growth 阶段
        #   就剔除净弱 / 变冗余的旧成员,不再永久占坑。三档优先级:
        #   1) R²_loo > r2_cap → 踢冗余簇内 **|rank_ic| 最小**者(留最强代表;克隆瘦身,
        #      非旧 argmax R² 口径 → 更强近克隆候选可挤掉弱钉子户)
        #   2) 超容量 / Δ_loo < floor → 踢 argmin Δ_loo(= |rank_ic|·(1−R²_loo);弱/冗余先走)
        #   3) 全过 → break
        while self.size > 0:
            r2s = self._leave_one_out_r2()
            deltas = self._leave_one_out_deltas()      # 成员 Δ=|rank_ic|·(1−R²_loo) 留池贡献
            over_cap = self.size > self.capacity
            redundant = r2s > self.r2_cap
            if redundant.any():
                # 冗余簇内踢 |rank_ic| 最小者(非簇内成员 single_q 屏蔽到 +inf,不参与 argmin)
                masked_q = np.where(redundant, self.single_q, np.inf)
                worst = int(np.argmin(masked_q))
            elif over_cap or (deltas < self.delta_floor).any():
                worst = int(np.argmin(deltas))         # 超容量 / 弱贡献成员 → 踢 argmin Δ_loo
            else:
                break
            if worst == cand_idx:
                self._evict(cand_idx)               # 候选自己最该踢(净最弱/最冗余)→ revert
                self._failure_cache.add(sig)
                return False, 0.0
            self._evict(worst)
            if worst < cand_idx:
                cand_idx -= 1

        self._failure_cache = set()
        return True, 0.0

    def _evict(self, idx: int) -> None:
        """删除 members[idx] + 同步删 single_q / pnl_corrs。"""
        n = self.size
        self.members.pop(idx)
        keep = [i for i in range(n) if i != idx]
        self.single_q = self.single_q[keep]
        self.pnl_corrs = self.pnl_corrs[np.ix_(keep, keep)]


# 向后兼容别名(main.py / evaluator.py 用 AlphaPool 名字 import)
AlphaPool = LinearAlphaPool


# ============================================================================
# 跨 run 累积的正交 alpha 库(2026-06-22,原 alpha_library 并入)
#
# 只存因子树(tree/pretty/rank_ic/源 run),**不存 pnl 序列**(太大)。
# - `_greedy_orthogonal`:按 |rank_ic| 降序贪心,保留与已选 pnl |pearson| < corr_max 的(C 算子 pnl_corr_vec)。
# - `append_run`:run 尾用内存中 collected 的 pnl 贪心挑 within-run 正交高 IC,append 到库
#   (只 within-run 正交;跨 run 冗余 + 衰减 + 封顶留给定时清理 `scripts/clean_alpha_library.py`)。
#
# IC 门(>0.01)继承 collected 已有的 `admit_rankic_min` 过滤(below-gate 候选 per_t_pnl=None 根本不进
# collected),不另设 knob。
# ============================================================================

def _greedy_orthogonal(items: List[Tuple], corr_max: float, cap: int = 0) -> List[Tuple]:
    """items=[(tree, rank_ic, per_t_pnl), …] 已按 |rank_ic| 降序(pnl 在末位)。
    贪心保留与已选 pnl |pearson| < corr_max 的;cap>0 时满 cap 即止。返回保留项(原 tuple 不变)。"""
    kept: List[Tuple] = []
    kept_pnl: List[np.ndarray] = []
    for it in items:
        ptp = np.ascontiguousarray(it[-1], dtype=np.float64)
        if kept_pnl:
            M = np.ascontiguousarray(np.stack(kept_pnl))
            if float(np.abs(ops.pnl_corr_vec(ptp, M)).max()) >= corr_max:
                continue
        kept.append(it)
        kept_pnl.append(ptp)
        if cap and len(kept) >= cap:
            break
    return kept


def append_run(collected: Dict, lib_path: Path, corr_max: float, max_size: int, run_ts: str) -> int:
    """从本轮 collected(值=(tree, q, ptp, rank_ic),pnl 在内存)贪心挑 within-run 正交高 IC,
    append 到库文件(只追加,跨 run 冗余留给清理)。返回本轮入库数。"""
    items = sorted(((c[0], c[3], c[2]) for c in collected.values()), key=lambda x: -abs(x[1]))
    kept = _greedy_orthogonal(items, corr_max, cap=max_size)
    existing = json.loads(lib_path.read_text(encoding='utf-8')) if lib_path.exists() else []
    for tree, rank_ic, _ in kept:
        existing.append({'tree': tree.to_dict(), 'pretty': tree.pretty(),
                         'rank_ic': float(rank_ic), 'run': run_ts})
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    lib_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
    return len(kept)
