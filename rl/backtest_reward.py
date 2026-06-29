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
from markets import ACTIVE as _MKT, CALENDAR as _CAL


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
