"""
Dominated Novelty Search archive(numpy 移植,供 alpha pool 用)。

来源:Lim, Faldor, Cully, Grillotti — Dominated Novelty Search: Rethinking Local
Competition in Quality-Diversity, GECCO 2025(arXiv:2502.00593)。参考实现 JAX+QDax
(`qdax/core/populations/adaptive_population.py`),此处纯 numpy 重写。

为什么用它当 archive(替/补现 LinearAlphaPool 的 leave-one-out prune):
    - 描述子是**高维学习量**(隐尾载荷向量 D~10-50),网格 MAP-Elites 在高维爆格不可用
      (格数 = 分辨率^D);DNS 无网格、无界、无分辨率,成本仅 O(N²·D) 线性于 D,
      论文实测撑到 1000 维、随维度增长以 p<1e-9 碾压网格。
    - 天然实现"IC 当准入闸、多样性当选择压":一个低质量因子只有当它在描述子空间里
      **远离所有更优因子**(占新 niche)才存活;若与某更优因子相近则被剔 —— 正是
      "绝不保留一个更优近邻已覆盖的弱因子"。

核心:competition fitness(dominated novelty score)= 到 **k 个最近"更优"邻居** 的平均
距离;无更优邻居(=全局最优)→ +∞ 永留。选择 = 按此分数截断到 max_size。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

_EPS = 1e-12


def whiten(descriptors: np.ndarray) -> np.ndarray:
    """列 z-score(论文未做,但学习描述子各维尺度不一,裸 L2 会被高方差维主导 —— 文献
    标注的头号坑)。(N, D) → (N, D);零方差列保持 0。"""
    mu = descriptors.mean(axis=0, keepdims=True)
    sd = descriptors.std(axis=0, keepdims=True)
    sd = np.where(sd < _EPS, 1.0, sd)
    return (descriptors - mu) / sd


def dominated_novelty(fitness: np.ndarray, descriptors: np.ndarray,
                      k: int = 3) -> np.ndarray:
    """competition fitness = 到 k 个最近"更优(fitness≥自身)"邻居的平均 L2 距离。

    fitness: (N,) 质量(净 CVaR / |rank IC|),越大越好。
    descriptors: (N, D) 行为描述子(隐尾载荷向量;调用方应先 whiten)。
    无更优邻居(唯一全局最优)→ +∞(永留);更优邻居数 <k → 用全部可得邻居。
    k=3 为论文稳健默认(k∈{1,3,5} 无显著差异)。"""
    N = fitness.shape[0]
    # fitter[i,j] = j 不差于 i(≥,排除对角):等质量互为"更优"(论文 <= 约定)。
    fitter = fitness[None, :] >= fitness[:, None]
    np.fill_diagonal(fitter, False)
    dist = np.linalg.norm(descriptors[:, None, :] - descriptors[None, :, :], axis=-1)
    dist = np.where(fitter, dist, np.inf)
    dist.sort(axis=1)                                  # 升序,更优邻居在前
    score = np.empty(N, dtype=np.float64)
    for i in range(N):
        finite = dist[i][np.isfinite(dist[i])][:k]
        score[i] = finite.mean() if finite.size else np.inf
    return score


@dataclass
class DNSArchive:
    """固定容量种群,competition = dominated novelty。

    每代:concat(现群, 候选) → 算 dominated novelty → 截断保 top-max_size。
    与 LinearAlphaPool 的差异:不靠 R²/Δ 阈值,靠"在更优邻居中是否够新"全局排序。
    quality 与 descriptor 平行存;genos 仅占位(集成时换 AlphaTree)。"""
    capacity: int = 24
    k:        int = 3

    genos:       List[object] = field(default_factory=list)
    fitness:     np.ndarray   = field(default_factory=lambda: np.zeros(0))
    descriptors: np.ndarray   = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def size(self) -> int:
        return len(self.genos)

    def add_batch(self, genos: List[object], fitness: np.ndarray,
                  descriptors: np.ndarray) -> List[object]:
        """加入一批候选并截断回 capacity。返回**被淘汰**的 genos(供调用方记账)。"""
        if self.size == 0:
            all_g, all_f, all_d = list(genos), np.asarray(fitness, float), np.asarray(descriptors, float)
        else:
            all_g = self.genos + list(genos)
            all_f = np.concatenate([self.fitness, np.asarray(fitness, float)])
            all_d = np.vstack([self.descriptors, np.asarray(descriptors, float)])

        if len(all_g) <= self.capacity:
            self.genos, self.fitness, self.descriptors = all_g, all_f, all_d
            return []

        score = dominated_novelty(all_f, whiten(all_d), self.k)
        keep = np.argsort(score)[::-1][:self.capacity]          # 分高者留
        keep_set = set(keep.tolist())
        dropped = [all_g[i] for i in range(len(all_g)) if i not in keep_set]
        self.genos = [all_g[i] for i in keep]
        self.fitness = all_f[keep]
        self.descriptors = all_d[keep]
        return dropped
