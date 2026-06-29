"""Continuous24x7 画像 —— Binance 永续 crypto(现状)。

**逐位复现** P0 之前的所有硬编码常量(288 bars/day、365 交易日、
bars_per_year=105120、skip_warmup=288),作为 MarketProfile 抽象层的回归门:
注入此画像时引擎行为必须与改造前 byte-identical。
"""

from __future__ import annotations

from markets.profile import MarketProfile, TradingCalendar

CONTINUOUS_24X7 = MarketProfile(
    name='continuous24x7',
    calendar=TradingCalendar(
        name='continuous24x7',
        bars_per_day=288,            # 24h × 12 (5m bars)
        hours_per_day=24.0,
        trading_days_per_year=365.0,
        has_overnight_gap=False,
        tz='UTC',
    ),
    warmup_bars=288,                 # = 旧 skip_warmup_bars(1 日)
)
