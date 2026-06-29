"""配置 — 单一入口:读 `config/config.ini`。

只暴露一个助手 `ini(section, key, default)`:类型按 default 自动推断,INI 缺失或留空
时回落 default。子模块在 dataclass 字段上直接调用,例:

    @dataclass
    class FactorMiningConfig:
        pool_capacity: int = ini('factor_mining', 'pool_capacity', 24)

INI 在 import 阶段读一次,值固化为字段默认。
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar


_INI = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
_INI.read(Path(__file__).resolve().parent / 'config.ini', encoding='utf-8')


T = TypeVar('T')


def ini(section: str, key: str, default: T) -> T:
    """从 config.ini[section][key] 读值;类型按 default 自动转换。
    缺失 / 留空 → 返回 default。"""
    if section not in _INI or key not in _INI[section]:
        return default
    raw = _INI[section][key].strip()
    if not raw:
        return default
    t = type(default)
    if t is bool:
        return raw.lower() in ('true', 'yes', '1', 'on')   # type: ignore[return-value]
    if t is int:
        return int(raw)                                    # type: ignore[return-value]
    if t is float:
        return float(raw)                                  # type: ignore[return-value]
    return raw                                             # type: ignore[return-value]


# ============================================================================
# 顶层 orchestration 配置
# ============================================================================

@dataclass
class FactorMiningConfig:
    parquet_5m_root: str = ini('factor_mining', 'parquet_5m_root', './data/parquet_5m')
    pool_capacity:   int = ini('factor_mining', 'pool_capacity', 24)


@dataclass
class AlphaSAGEConfig:
    """AlphaSAGE GFlowNet alpha-mining baseline(main.py train-alphasage 驱动,独立于 GP)。
    parquet_root / pool_capacity 复用 [factor_mining];GFN/reward 超参在 [alphasage]。"""
    parquet_root:        str   = ini('factor_mining', 'parquet_5m_root', './data/parquet_5m')
    pool_capacity:       int   = ini('factor_mining', 'pool_capacity', 24)
    hidden_dim:          int   = ini('alphasage', 'hidden_dim', 64)
    n_layers:            int   = ini('alphasage', 'n_layers', 2)
    lr:                  float = ini('alphasage', 'lr', 0.001)
    logz_lr:             float = ini('alphasage', 'logz_lr', 0.1)       # logZ 单独高 lr(GFlowNet 铁律,防 TB 发散)
    batch_size:          int   = ini('alphasage', 'batch_size', 16)
    n_episodes:          int   = ini('alphasage', 'n_episodes', 4000)
    nov_weight:          float = ini('alphasage', 'nov_weight', 0.1)
    entropy_coef:        float = ini('alphasage', 'entropy_coef', 0.01)
    entropy_temperature: float = ini('alphasage', 'entropy_temperature', 1.0)
    r_sa_weight:         float = ini('alphasage', 'r_sa_weight', 0.5)     # λ(0):R_SA 结构稠密奖励初值(退火→0)
    r_sa_k:              int   = ini('alphasage', 'r_sa_k', 16)           # R_SA kNN 结构邻居数(池内)
    r_sa_tau:            float = ini('alphasage', 'r_sa_tau', 1.0)        # R_SA 结构相似度 softmax 温度
