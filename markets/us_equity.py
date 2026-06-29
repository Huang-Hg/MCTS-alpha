"""US 权益画像(P0 桩:仅日线日历)。

P0 只填日历片以验证抽象层可注入不同市场常量;CostModel(非对称佣金 + SEC/TAF)、
ShortingPolicy(locate + borrow gate)、CorporateActionAdjuster(复权)、
UniverseProvider(PIT 指数成分)、CarryModel(分红 + 借券费)等子模型在 P1-P3 续接。
warmup/trading_days 等数值 P2 经实测再校准。
"""

from __future__ import annotations

from markets.profile import MarketProfile, TradingCalendar

US_EQUITY_DAILY = MarketProfile(
    name='us_equity_daily',
    calendar=TradingCalendar(
        name='us_equity_daily',
        bars_per_day=1,              # 日线(P3 可加 us_equity_5m=78 bars/day)
        hours_per_day=6.5,           # RTH 09:30-16:00 ET
        trading_days_per_year=252.0,
        has_overnight_gap=True,
        tz='America/New_York',
    ),
    warmup_bars=20,                  # ~1 月日线 warmup(P2 校准)
)
