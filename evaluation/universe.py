"""因果 universe 内核:trailing qv-window hourly top-N membership + 边界 grace(纯函数,无 IO)。

与 live(trade/runner.task_universe_rotation / _plan_rotation)同语义:
  - membership[h, s] ⟺ sym s ∈ top-N by Σ quote_volume over (hours[h]−window, hours[h]]
    (live exchange.fetch_quote_volume_ranks 同为 1h kline 滚 window_hours 和、丢未收盘
     当前小时、min_periods=1 —— 与本内核同语义同粒度,精确对齐。
     窗宽 = INI [trade_universe] qv_window_hours,bt/live 单一参数源。)
  - grace:连续掉出 top-N 不满 hold_max_persist 小时仍可持有(live 边界 grace);
    满 persist 转 wind-down(live = hold-only 只卖不买;bt 一阶近似 = 不可持有)。
  - 决策时刻 ts 用 stamp = 最后一个 ≤ ts 的整点 hour 的 membership(live rotation
    每小时整点拉 rank,L1 决策用最近一次 rotation 结果,同构)。

因果性约定(铁律):hour H 的 membership 只用 close ≤ H 的
bar 量;decision_time = bar close,故 stamp ≤ ts 的 membership 在 ts 时刻全部可知。

参数不在此沉淀:top_n / hold_max_persist 由 caller 从 INI [trade_universe] 读
(与 live 同源,不另设键)。
"""
from __future__ import annotations

import numpy as np

HOUR_MS = 3_600_000


def trailing_membership(qv_h: np.ndarray, top_n: int, window_hours: int = 24) -> np.ndarray:
    """member[i, j] ⟺ sym j ∈ top-N by Σ qv_h over (i−window, i](rolling sum, min_periods=1)。

    量为 0 的 sym 永不入选(即使在场 sym < top_n);新上币按已上市部分量参与排名(同 live
    24h ticker 对上市 <24h 标的的语义)。前置条件:qv_h 无 NaN(缺失=0,caller 已 fillna;
    NaN 会毒化 cumsum 使该列永久退出排名)。
    """
    H, S = qv_h.shape
    cum = np.vstack([np.zeros((1, S)), qv_h.cumsum(axis=0)])             # (H+1, S)
    lo = np.maximum(np.arange(1, H + 1) - window_hours, 0)
    trail = cum[1:] - cum[lo]                                            # (H, S) 滚动 window 和
    member = np.zeros((H, S), dtype=bool)
    k = min(top_n, S)
    top = np.argpartition(trail, -k, axis=1)[:, -k:]                     # (H, k)
    rows = np.repeat(np.arange(H), k)
    cols = top.ravel()
    pos = trail[rows, cols] > 0.0
    member[rows[pos], cols[pos]] = True
    return member


def grace_mask(member: np.ndarray, persist: int) -> np.ndarray:
    """边界 grace:holdable[i, s] = member[i, s] ∨ 连续掉出不满 persist 小时。

    与 live _plan_rotation 的 hold_dwell 同语义:rank ≤ top_n 任一小时即清零 dwell;
    连续掉出满 persist 小时转 wind-down(bt 近似为不可持有)。从未入选过 → False。
    """
    H, S = member.shape
    idx = np.arange(H, dtype=np.int64)[:, None]
    last = np.maximum.accumulate(np.where(member, idx, np.int64(-1)), axis=0)   # 最近 member 小时
    return (last >= 0) & ((idx - last) < persist)


def member_at(hours_ms: np.ndarray, mask_h: np.ndarray, ts_ms: np.ndarray) -> np.ndarray:
    """把小时级 mask 映射到任意决策时刻:stamp = 最后一个 ≤ ts 的 hour。

    ts 早于首个 hour stamp → 整行 False(无 membership 历史 = 不可持有)。
    返回 (len(ts), S) bool。
    """
    idx = np.searchsorted(hours_ms, ts_ms, side='right') - 1             # (T,)
    out = np.zeros((len(ts_ms), mask_h.shape[1]), dtype=bool)
    ok = idx >= 0
    out[ok] = mask_h[idx[ok]]
    return out
