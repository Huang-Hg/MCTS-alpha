"""市场画像 —— 把市场机制隔离在引擎之外的薄抽象层(日历/常量 + 复权 + 注册表)。

引擎(backtest / rl / evaluation / search)从不直接知道「市场」,只消费一个
`MarketProfile`。crypto 注入 `CONTINUOUS_24X7`(逐位复现旧行为),权益注入
`US_EQUITY_DAILY` 等。本文件聚合(无 __init__.py,namespace package):
  - `TradingCalendar` / `MarketProfile` 数据类
  - `CorporateActionAdjuster` / `IdentityAdjuster`(复权口径;信号用复权价、成交用原始价)
  - `CONTINUOUS_24X7` / `US_EQUITY_DAILY` 具体画像
  - `get_profile()` / 模块级 `ACTIVE` / `CALENDAR`(由 `config.ini [calendar].profile` 选)
CostModel / ShortingPolicy / UniverseProvider 等子模型 P2-P3 续接;operand 词表见
`markets.vocabulary`(数据驱动,不在画像里固化)。

设计要点(critique 已校正):
  - `bars_per_year` = **回测权益曲线采样 cadence** 的年 bar 数(= bars_per_day × 交易日),
    Sharpe 年化 = mean/std × sqrt(bars_per_year)。日线 US = 252;5m US = 252×78。
    不是 sqrt(交易日数) —— 年化必须匹配权益曲线采样频率,而非决策频率。
  - ts-算子按数组行索引开窗,正确性取决于「相邻行是否相邻时间」。本类只提供常量;
    隔夜/午休 gap 的物化(决定相邻行是否跨 session)属数据层,不在此。
  - 复权口径(critique #4 双计数):复权价已抹除息跳空 → 复权价 MTM(无 dividend_TS)XOR
    原始价 MTM(+dividend_TS),单一口径二选一;数据层只产两条价,口径在回测核强制。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.config import ini


# ============================================================================
# 日历 + 画像数据类
# ============================================================================

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
    """注入引擎的市场画像。含 calendar + warmup_bars + windows;子模型 P2-P3 续接。"""
    name:        str
    calendar:    TradingCalendar
    warmup_bars: int                 # bt 跳过的初始不稳定 bar 数(crypto 288 = 1 日)
    windows:     tuple               # ts/pair 回看窗集(单位=bar cadence);挖矿入口注入 grammar.set_windows


# ============================================================================
# 公司行为复权口径
# ============================================================================
# 铁律:**信号/特征用复权价(总收益,除权除息跳空已抹平),成交/fill 用原始价**,两条价分离。
# adj_factor = adj_close / raw_close,对同日 O/H/L/C 统一乘法缩放。

@dataclass(frozen=True)
class CorporateActionAdjuster:
    """raw 价 + adj_factor → 复权价(信号)vs 原始价(成交)。权益注入此版。"""

    @staticmethod
    def adjust(raw_px: np.ndarray, adj_factor: np.ndarray) -> np.ndarray:
        """复权价(喂特征/收益):raw × factor。NaN 传播(未上市/停牌)。"""
        return raw_px * adj_factor

    @staticmethod
    def raw(raw_px: np.ndarray) -> np.ndarray:
        """原始价(喂成交/fill):恒等。"""
        return raw_px


@dataclass(frozen=True)
class IdentityAdjuster:
    """crypto 注入:无 split/dividend,原始价天然连续正确(raw == adjusted)。"""

    @staticmethod
    def adjust(raw_px: np.ndarray, adj_factor: np.ndarray | None = None) -> np.ndarray:
        return raw_px

    @staticmethod
    def raw(raw_px: np.ndarray) -> np.ndarray:
        return raw_px


# ============================================================================
# 具体画像
# ============================================================================
# ts/pair 回看窗集(单位 = 各市场 bar cadence)的**唯一来源**(grammar.ACTIVE_WINDOWS 默认从此取)。
# 圆整周期(非 2 的幂网格),贴正常 alpha 习惯:
#   crypto(1h bars):1h/2h/4h/6h/12h/1d/2d/4d/7d/14d。
#   daily(equity/ashare 共用):3d/1w/2w/1mo/2mo/季/半年/年。
_CRYPTO_WINDOWS_1H: tuple = (1, 2, 4, 6, 12, 24, 48, 96, 168, 336)
_DAILY_WINDOWS:     tuple = (3, 5, 10, 20, 40, 60, 120, 240)

# Continuous24x7 —— Binance 永续 crypto(现状):288 bars/day、365 交易日、bars_per_year=105120、
# skip_warmup=288(MTM/成本/强平骨架仍 bit-identical 回归门;窗口集 1h 圆整档)。
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
    windows=_CRYPTO_WINDOWS_1H,
)

# US 权益(日线):日历片已落地;CostModel(非对称佣金 + SEC/TAF)、ShortingPolicy、
# UniverseProvider(PIT 指数成分)、CarryModel(分红 + 借券费)等子模型 P2-P3 续接。
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
    windows=_DAILY_WINDOWS,
)

# 中国 A 股(日线):242 交易日 / 4h RTH(09:30-11:30 + 13:00-15:00)/ T+1(daily cadence 天然满足)/
# 涨跌停(执行层 tradeable-mask,不在 calendar)/ CNY。成本机制(印花税卖出单边)在 ASharePolicy(C++)。
A_SHARE_DAILY = MarketProfile(
    name='a_share_daily',
    calendar=TradingCalendar(
        name='a_share_daily',
        bars_per_day=1,              # 日线
        hours_per_day=4.0,           # RTH 09:30-11:30 + 13:00-15:00
        trading_days_per_year=242.0, # 年交易日(~242)
        has_overnight_gap=True,
        tz='Asia/Shanghai',
    ),
    warmup_bars=20,
    windows=_DAILY_WINDOWS,
)


# ============================================================================
# 注册表 —— 引擎经此拿 active MarketProfile(由 config.ini [calendar].profile 选)
# ============================================================================

_PROFILES = {
    'continuous24x7':  CONTINUOUS_24X7,
    'us_equity_daily': US_EQUITY_DAILY,
    'a_share_daily':   A_SHARE_DAILY,
}


def get_profile(name: str | None = None) -> MarketProfile:
    """按名取画像;None → 读 config.ini [calendar].profile(默认 continuous24x7)。"""
    key = name if name is not None else ini('calendar', 'profile', 'continuous24x7')
    return _PROFILES[key]


# import 阶段按 INI 固化(与 config dataclass 同生命周期)。
ACTIVE: MarketProfile = get_profile()
CALENDAR: TradingCalendar = ACTIVE.calendar
