"""市场画像注册表 —— 引擎经此拿 active `MarketProfile`(由 `config.ini [calendar].profile` 选)。

re-export:`get_profile()` / 模块级 `ACTIVE` / `CALENDAR`。crypto 默认 `continuous24x7`
(逐位复现旧行为);权益注入 `us_equity_daily` 等。import 阶段按 INI 固化(与
config dataclass 同生命周期)。
"""

from __future__ import annotations

from config.config import ini
from markets.profile import MarketProfile, TradingCalendar
from markets.continuous import CONTINUOUS_24X7
from markets.us_equity import US_EQUITY_DAILY

__all__ = ['MarketProfile', 'TradingCalendar', 'get_profile', 'ACTIVE', 'CALENDAR']

_PROFILES = {
    'continuous24x7':  CONTINUOUS_24X7,
    'us_equity_daily': US_EQUITY_DAILY,
}


def get_profile(name: str | None = None) -> MarketProfile:
    """按名取画像;None → 读 config.ini [calendar].profile(默认 continuous24x7)。"""
    key = name if name is not None else ini('calendar', 'profile', 'continuous24x7')
    return _PROFILES[key]


ACTIVE: MarketProfile = get_profile()
CALENDAR: TradingCalendar = ACTIVE.calendar
