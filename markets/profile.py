"""市场画像 —— 把市场机制隔离在引擎之外的薄抽象层(P0:仅日历/常量)。

引擎(backtest / rl / evaluation / search)从不直接知道「市场」,只消费一个
`MarketProfile`。crypto 注入 `CONTINUOUS_24X7`(逐位复现旧行为),权益注入
`US_EQUITY_DAILY` 等。P0 只落地 `TradingCalendar` + warmup;CostModel /
ShortingPolicy / UniverseProvider / CorporateActionAdjuster 等子模型 P1-P3 续接。

设计要点(critique 已校正):
  - `bars_per_year` = **回测权益曲线采样 cadence** 的年 bar 数(= bars_per_day × 交易日),
    Sharpe 年化 = mean/std × sqrt(bars_per_year)。日线 US = 252;5m US = 252×78。
    不是 sqrt(交易日数) —— 年化必须匹配权益曲线采样频率,而非决策频率。
  - ts-算子按数组行索引开窗,正确性取决于「相邻行是否相邻时间」。本类只提供常量;
    隔夜/午休 gap 的物化(决定相邻行是否跨 session)属数据层(P1),不在此。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingCalendar:
    """市场时间模型 —— 所有 bars-per-day / 年化 / panel 深度常量的唯一来源。"""
    name:                  str
    bars_per_day:          int      # 回测 bar cadence 下每交易日 bar 数(crypto 5m→288;US 日线→1)
    hours_per_day:         float    # 每交易日小时数(crypto 24;US RTH 6.5;A股 4)
    trading_days_per_year: float    # 年交易日(crypto 365;US ~252;A股 ~242)
    has_overnight_gap:     bool     # 是否存在隔夜/周末跳空(crypto False;权益 True)
    tz:                    str      # 本地时区(crypto UTC)

    @property
    def bars_per_year(self) -> float:
        """年化用年 bar 数 = 权益曲线采样 cadence 的年 bar 数 = bars_per_day × 交易日。"""
        return self.bars_per_day * self.trading_days_per_year


@dataclass(frozen=True)
class MarketProfile:
    """注入引擎的市场画像。P0 含 calendar + warmup_bars;子模型 P1-P3 续接。"""
    name:        str
    calendar:    TradingCalendar
    warmup_bars: int    # bt 跳过的初始不稳定 bar 数(crypto 288 = 1 日)
