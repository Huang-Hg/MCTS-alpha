"""PIT 指数成分 —— **建 parquet 阶段**的成分过滤(消除纳入前视)。

架构:项目只读 panel.parquet,universe = parquet 的 symbol,因果 T-1 构建天然无 lookahead。
membership 是「**构建 parquet**」的输入,**不是**消费端(loader/挖矿/回测)的输入 —— 在建 parquet
时按 PIT 成分把非成员 (date,symbol) 行剔掉,产出的 parquet 本身即 PIT-correct,消费端完全不感知。

membership 长表(parquet/csv:symbol, start, end;end 空=仍在册;同 symbol 可多段区间)用官方
**生效日**作半开区间 [start,end) 边界(决策时刻可知,无 lookahead)。pull 脚本建 panel.parquet 时调
`pit_filter_long(df, membership)` 过滤。

注:此过滤修**纳入前视**;**幸存者偏差**还需 pull 按 membership 的全历史 symbol union 拉价(含已剔除名),
否则当期成分套全历史时退市名缺价数据仍缺席。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_RENAME = {'ticker': 'symbol', 'start_date': 'start', 'end_date': 'end'}


def load_membership(path: Path) -> pd.DataFrame:
    """成分 membership 长表 → DataFrame[symbol, start(datetime64), end(datetime64|NaT)]。
    接受 ticker/start_date/end_date 或 symbol/start/end 列名;end 空 → NaT(仍在册)。
    start 含 NaT(纳入日缺失)→ fail-fast(数据损坏,非静默丢区间)。"""
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    df = df.rename(columns=_RENAME)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])                # 空 → NaT(end 允许 NaT=仍在册)
    assert not df['start'].isna().any(), 'membership start 列含 NaT(纳入日缺失)→ fail-fast'
    return df[['symbol', 'start', 'end']]


def pit_filter_long(df: pd.DataFrame, membership: pd.DataFrame,
                    date_col: str = 'date', symbol_col: str = 'symbol') -> pd.DataFrame:
    """建 parquet 期的 PIT 成分过滤:长表 panel(date,symbol,…)只保留 symbol 在 date ∈ 指数的行
    (半开区间 [start,end);membership **无记录** 的 symbol 全程剔除 = strict PIT)。
    数据边界 fail-fast:匹配 symbol < 半数 → 大声崩(membership 与 panel 符号体系不一致,防静默全剔)。"""
    panel_syms = df[symbol_col].unique()
    mb = membership[membership['symbol'].isin(panel_syms)].copy()
    matched = mb['symbol'].nunique()
    assert matched >= 0.5 * len(panel_syms), (
        f"PIT membership 仅匹配 {matched}/{len(panel_syms)} symbol → 疑符号体系不一致"
        f"(membership symbol vs panel symbol),拒绝静默全剔除;核对符号编码")
    mb['end'] = mb['end'].fillna(pd.Timestamp.max)
    m = df.merge(mb, on=symbol_col, how='left')          # 无记录 symbol → start/end NaT → 下面 keep 全 False
    keep = (m[date_col] >= m['start']) & (m[date_col] < m['end'])
    out = m[keep].drop(columns=['start', 'end']).drop_duplicates(subset=[date_col, symbol_col])
    return out.sort_values([date_col, symbol_col]).reset_index(drop=True)
