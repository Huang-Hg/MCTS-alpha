"""
组合级 backtest 配置与一次性面板预处理。

evaluator 调 _bt.portfolio_bt → single equity 曲线 → reward。
本模块负责:
    - BacktestRewardConfig:portfolio / cost / risk 参数,集中读 INI
    - prepare_close_TS / prepare_open_TS: per-symbol ffill,确保 NaN-free 喂 C 内核
    - prepare_panel_TS:funding / vol / quote_volume 的 contig 化(无 ffill)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from backtest import _bt as _bt_kernel
from config.config import ini
from markets.profile import ACTIVE as _MKT, CALENDAR as _CAL, get_profile


@dataclass
class BacktestRewardConfig:
    """单 bar 直达 target、全 taker @ open 的 bt 配置(费率模型:half_spread+fee+impact+funding+强平+
    trailing-stop)。费后 PnL 仅用于 val 诊断,不进训练 reward(reward 已纯 IC × diversity)。
    cost 三项均为 fraction:half_spread(半点差)/ fee(taker 单边)/ impact_Y(sqrt-impact 系数)。"""
    initial_cash:      float = 100_000.0
    half_spread_rate:  float = ini('backtest_reward', 'half_spread_rate',  0.0)
    fee_rate:          float = ini('backtest_reward', 'fee_rate',          0.00050)
    impact_Y:          float = ini('backtest_reward', 'impact_Y',          0.5)
    maint_margin_rate: float = 0.005
    skip_warmup_bars:  int   = _MKT.warmup_bars        # crypto 288;由 active MarketProfile 派生
    leverage_cap:      float = ini('backtest_reward', 'leverage_cap',      2.0)    # top-K 多空 gross 目标 Σ|w|
    max_concentration: float = ini('backtest_reward', 'max_concentration', 0.20)   # 旧 z 加权 clip;top-K 下不触及
    # top-K 多空组合构造(eval≡live 共享口径;贴 alphasage/alphacfg top-K 等权 + 允许做空 + swap 缓冲)
    top_k:             int   = ini('backtest_reward', 'top_k',    8)               # 每腿仓数(多 top_k + 空 bottom_k)
    swap_n:            int   = ini('backtest_reward', 'swap_n',   2)               # 每腿每 rebalance 最多换出名数
    min_hold:          int   = ini('backtest_reward', 'min_hold', 4)               # 在持满 N 决策 bar 才可换出
    # per-name trailing stop + latch:与 live 同源读 [trade_execute](bt 环境 = 部署环境);0 = 关闭。
    stop_trail_pct:    float = ini('trade_execute', 'stop_trail_pct', 0.0)
    stop_ratchet:      str   = ini('trade_execute', 'stop_ratchet',   '')

    def stop_ratchet_flat(self) -> np.ndarray | None:
        """'0.5:0.12,1.0:0.08' → flat (2k,) 升序 [gain, trail, ...];'' → None。"""
        pairs = sorted((float(g), float(t)) for g, t in
                       (p.split(':') for p in self.stop_ratchet.split(',') if p))
        if not pairs:
            return None
        return np.array([x for pr in pairs for x in pr], dtype=np.float64)


_EQUITY_PROFILE = 'us_equity_daily'


@dataclass
class EquityCostsConfig:
    """US 权益回测口径(EquityPolicy:佣金+半点差单边对称 cost_rate、空腿日借券费;
    无 funding / 强平 / sqrt-impact)。MTM 与 rebalance 全走**复权 OHLC**(单一总收益口径,
    与 EquityPanel.fwd_ret 一致;不混原始价以免 split/分红跳空双计数)。年化走 us_equity_daily=252。"""
    initial_cash:      float = 100_000.0
    cost_rate:         float = ini('equity_costs', 'cost_rate',    0.0005)   # 佣金+半点差,单边 fraction
    daily_borrow:      float = ini('equity_costs', 'daily_borrow', 0.0)      # 空腿日借券费(年化/252);0=关

    @property
    def skip_warmup_bars(self) -> int:
        return get_profile(_EQUITY_PROFILE).warmup_bars

    @property
    def bars_per_year(self) -> float:
        return get_profile(_EQUITY_PROFILE).calendar.bars_per_year


def run_equity_bt(weights_TS: np.ndarray,
                  adj_close_TS: np.ndarray, adj_open_TS: np.ndarray,
                  ecfg: EquityCostsConfig,
                  raw_weights: bool = True,
                  with_equity: bool = False) -> Dict[str, float]:
    """US 权益组合回测(EquityPolicy)→ 字典 (total_return / sharpe / mean_turnover [/ equity])。
    weights_TS:(T,S) 最终目标权重(market-neutral long-short,Σw≈0;raw_weights=True 直接用)。
    adj_close/adj_open:(T,S) **复权** OHLC(总收益口径)——MTM 用 adj_close,rebalance @ adj_open。
    成本 = cost_rate×|Δnotional|(买卖对称);空腿借券 = daily_borrow×|notional|;无 funding/强平。
    Sharpe 年化 = √bars_per_year(日线 US = √252)。"""
    res = _bt_kernel.portfolio_bt_equity(
        np.ascontiguousarray(weights_TS, dtype=np.float64),
        np.ascontiguousarray(adj_close_TS, dtype=np.float64),
        np.ascontiguousarray(adj_open_TS, dtype=np.float64),
        ecfg.initial_cash, int(ecfg.skip_warmup_bars),
        ecfg.cost_rate, ecfg.daily_borrow,
        1 if raw_weights else 0,
        0.0, None,                         # equity 暂不启用 per-name trailing-stop
        float(ecfg.bars_per_year),
    )
    out = {
        'total_return':  float(res['total_return']),
        'sharpe':        float(res['sharpe']),
        'mean_turnover': float(res['mean_turnover']),
    }
    if with_equity:
        out['equity'] = np.asarray(res['equity'], dtype=np.float64)
    return out


_ASHARE_PROFILE = 'a_share_daily'


@dataclass
class AShareCostsConfig:
    """中国 A 股回测口径(ASharePolicy:佣金双边 + 印花税卖出单边 + 过户费双边;**long-only**,无 funding/
    强平/impact)。MTM/rebalance 全走 qfq 复权 OHLC(总收益)。年化 √242;daily cadence 天然满足 T+1。"""
    initial_cash:      float = 100_000.0
    commission:        float = ini('ashare_costs', 'commission',       0.00025)  # 佣金单边(万2.5)
    stamp_tax_sell:    float = ini('ashare_costs', 'stamp_tax_sell',   0.0005)   # 印花税卖出单边(万5,2023 减半)
    transfer_fee:      float = ini('ashare_costs', 'transfer_fee',     0.00001)  # 过户费双边(万0.1)
    price_limit_pct:   float = ini('ashare_costs', 'price_limit_pct',  0.10)     # 涨跌停幅度(沪深300 主板 ±10%)

    @property
    def skip_warmup_bars(self) -> int:
        return get_profile(_ASHARE_PROFILE).warmup_bars

    @property
    def bars_per_year(self) -> float:
        return get_profile(_ASHARE_PROFILE).calendar.bars_per_year


def ashare_trade_block(adj_close_TS: np.ndarray, adj_open_TS: np.ndarray,
                       raw_volume_TS: np.ndarray, price_limit_pct: float) -> np.ndarray:
    """涨跌停 / 停牌方向冻结掩码 (T,S) int8 —— 喂 run_ashare_bt 的 C 引擎(因果,只用 ≤t 信息)。
    位语义:bit0(=1)开盘即涨停→禁买、bit1(=2)开盘即跌停→禁卖、3(=1|2)停牌→双禁。
    判据(open-fill 口径):ratio=adj_open[t]/adj_close[t-1](复权抹除除权跳空 → 真涨跌幅);
      涨停 ratio≥1+pct−1e-3、跌停 ratio≤1−pct+1e-3(1e-3 容 0.01 元 tick 取整误差);
      停牌 = open 非有限 ∨ 当日 raw 成交量≤0。**用原始(含 NaN)面板**派生,非 ffill 后的。"""
    T, S = adj_close_TS.shape
    block = np.zeros((T, S), dtype=np.int8)
    ratio = np.full((T, S), np.nan)
    ratio[1:] = adj_open_TS[1:] / adj_close_TS[:-1]
    block[ratio >= (1.0 + price_limit_pct) - 1e-3] |= 1           # 开盘涨停 → 禁买
    block[ratio <= (1.0 - price_limit_pct) + 1e-3] |= 2           # 开盘跌停 → 禁卖
    susp = ~np.isfinite(adj_open_TS) | ~np.isfinite(raw_volume_TS) | (raw_volume_TS <= 0.0)
    block[susp] |= 3                                              # 停牌 → 双禁
    return np.ascontiguousarray(block)


def run_ashare_bt(weights_TS: np.ndarray,
                  adj_close_TS: np.ndarray, adj_open_TS: np.ndarray,
                  acfg: AShareCostsConfig,
                  raw_weights: bool = True,
                  trade_block: np.ndarray | None = None,
                  with_equity: bool = False) -> Dict[str, float]:
    """中国 A 股组合回测(ASharePolicy)→ 字典 (total_return / sharpe / mean_turnover [/ equity])。
    weights_TS:(T,S) 最终目标权重(**long-only**,w≥0;A 股不可做空)。adj_*:qfq 复权 OHLC。
    成本 = 佣金×|Δ|(双边)+ **印花税×卖出 notional(单边)** + 过户费×|Δ|;无 funding/强平。Sharpe √242。
    trade_block((T,S) int8,可选):涨跌停 / 停牌方向冻结掩码(见 ashare_trade_block);受限方向不成交。"""
    res = _bt_kernel.portfolio_bt_ashare(
        np.ascontiguousarray(weights_TS, dtype=np.float64),
        np.ascontiguousarray(adj_close_TS, dtype=np.float64),
        np.ascontiguousarray(adj_open_TS, dtype=np.float64),
        acfg.initial_cash, int(acfg.skip_warmup_bars),
        acfg.commission, acfg.stamp_tax_sell, acfg.transfer_fee,
        1 if raw_weights else 0,
        None if trade_block is None else np.ascontiguousarray(trade_block, dtype=np.int8),
        float(acfg.bars_per_year),
    )
    out = {
        'total_return':  float(res['total_return']),
        'sharpe':        float(res['sharpe']),
        'mean_turnover': float(res['mean_turnover']),
    }
    if with_equity:
        out['equity'] = np.asarray(res['equity'], dtype=np.float64)
    return out


def _ffill_panel(panel: np.ndarray) -> np.ndarray:
    """per-symbol ffill + 前导 bfill,整列全 NaN → 填 1.0。Fully vectorized.
    旧版 Python 双层 loop 在 T=263k × S=387 上单线程跑 ~7 min,vectorized 版 < 5s。
    """
    T, S = panel.shape
    out = panel.astype(np.float64, copy=True) if panel.dtype != np.float64 else panel.copy()
    valid = np.isfinite(out) & (out > 0)
    has_any_col = valid.any(axis=0)                     # (S,) bool
    # row_idx[t,s] = 行号 if valid[t,s] else -1;cummax 沿 t 传播 → 最近 valid 行号
    row_idx = np.where(valid, np.arange(T, dtype=np.int64)[:, None], -1)
    np.maximum.accumulate(row_idx, axis=0, out=row_idx)
    leading_mask = (row_idx < 0)                        # 首个 valid 之前的位置
    fwd_src = np.where(row_idx >= 0, row_idx, 0)
    out_fwd = out[fwd_src, np.arange(S, dtype=np.int64)[None, :]]
    # bfill 前导:每列首个 valid 索引 f,用 out[f, j] 填 t < f 的所有行
    if leading_mask.any():
        first_idx = np.argmax(valid, axis=0)            # (S,) 首个 True 的 t;全 False 时为 0
        fv_vals = out[first_idx, np.arange(S, dtype=np.int64)]  # (S,) 首个 valid 值(无 valid 列是 NaN)
        out_fwd = np.where(leading_mask, fv_vals[None, :], out_fwd)
    # 整列无 valid → 填 1.0(对应整 sym 全 NaN)
    if not has_any_col.all():
        out_fwd[:, ~has_any_col] = 1.0
    return out_fwd


def prepare_close_TS(close_panel: np.ndarray) -> np.ndarray:
    """close (T, S) → ffilled (T, S) C-连续。"""
    return np.ascontiguousarray(_ffill_panel(close_panel))


def prepare_open_TS(open_panel: np.ndarray) -> np.ndarray:
    """open (T, S) → ffilled (T, S) C-连续(next-bar fill 用)。"""
    return np.ascontiguousarray(_ffill_panel(open_panel))


def prepare_panel_TS(panel: np.ndarray | None) -> np.ndarray | None:
    """funding / vol / quote_volume 面板 (T, S) C-连续(无 ffill,kernel 自动跳过 0/NaN lane)。"""
    return np.ascontiguousarray(panel) if panel is not None else None


_BARS_PER_DEC = 12      # 60min 决策 / 5m bar(dec 信号广播步长)


def broadcast_dec_to_5m(dec_vals: np.ndarray, T_gate: int) -> np.ndarray:
    """(n_dec, S) 决策网格信号 → (T_gate, S) 5m 网格,每 dec 值持平 _BARS_PER_DEC bar
    (target-position;首 dec 前无效区由 valid_mask 屏蔽,末段持平到窗尾)。val_eval 的池
    ensemble dec 信号经此持平到 5m 价路径再喂 C portfolio_bt(stop/liq 精度走 5m)。"""
    rep = np.repeat(dec_vals, _BARS_PER_DEC, axis=0)
    if rep.shape[0] >= T_gate:
        return np.ascontiguousarray(rep[:T_gate])
    pad = np.broadcast_to(rep[-1], (T_gate - rep.shape[0], rep.shape[1]))
    return np.ascontiguousarray(np.concatenate([rep, pad], axis=0))


def run_bt(deployed_signal: np.ndarray,
           close_TS: np.ndarray, open_TS: np.ndarray, qvol_TS: np.ndarray,
           vol_TS: np.ndarray, funding_TS: np.ndarray,
           bcfg: BacktestRewardConfig,
           with_equity: bool = False,
           leverage_cap: float | None = None,
           max_concentration: float | None = None,
           valid_mask: np.ndarray | None = None,
           raw_weights: bool = False) -> Dict[str, float]:
    """跑 _bt_kernel.portfolio_bt → 字典 (total_return / sharpe / mean_turnover / liq_at)。
    deployed_signal: caller 已取过 sign 的 (T,S) panel(C 内核内部走 signal_to_weight)。
    valid_mask((T,S) bool,可选):因果 universe 持仓资格(listed ∧ member)。非 valid 单元
    信号置 NaN → C 内核 _row_signal_to_weight cs 算子 NaN-aware 自动排除 + 权重归 0。
    raw_weights=True:deployed_signal 已是最终目标权重(如 topk_ls_weights 输出 broadcast 到 5m),
    内核跳过 signal_to_weight 直接用;此时 valid 资格已在权重构造里处理,不再 NaN-mask。
    费率模型层级(half_spread + fee + impact + funding + cross-margin 强平 + trailing-stop)全走。
    with_equity=True 时附加 'equity' (T,) 序列。
    """
    if valid_mask is not None and not raw_weights:
        deployed_signal = np.ascontiguousarray(
            np.where(valid_mask, deployed_signal, np.nan), dtype=np.float64)
    res = _bt_kernel.portfolio_bt(
        deployed_signal, close_TS, open_TS,
        None, None,                      # high_TS / low_TS:isolated wick 强平未启用(cross-margin)
        qvol_TS, vol_TS, funding_TS,
        bcfg.initial_cash, int(bcfg.skip_warmup_bars),
        bcfg.half_spread_rate, bcfg.fee_rate, bcfg.impact_Y,
        bcfg.maint_margin_rate,
        float(leverage_cap if leverage_cap is not None else bcfg.leverage_cap),
        float(max_concentration if max_concentration is not None else bcfg.max_concentration),
        1 if raw_weights else 0,         # raw_weights:1=直接用目标权重;0=走 signal_to_weight
        None,                            # leverage_TS:isolated 杠杆未启用
        float(bcfg.stop_trail_pct),
        bcfg.stop_ratchet_flat(),
        float(_CAL.bars_per_year),       # Sharpe 年化 = sqrt(bars_per_year);crypto 105120
    )
    out = {
        'total_return':  float(res['total_return']),
        'sharpe':        float(res['sharpe']),
        'mean_turnover': float(res['mean_turnover']),
        'liq_at':        int(res['liq_at']),
    }
    if with_equity:
        out['equity'] = np.asarray(res['equity'], dtype=np.float64)
    return out
