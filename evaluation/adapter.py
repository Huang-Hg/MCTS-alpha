"""
读 parquet_5m 月文件 → (T, S) 面板 + future-return 标签
=======================================================

输出:
    PanelBundle:
        panels:        dict[列名 str -> (T,S) float64 C-contiguous]
        y_future:      (T, S) float64,future log-return at primary horizon
        timestamps:    pd.DatetimeIndex (length T,UTC)
        symbols:       list[str] (length S)

关键决策:
    - 月文件已经是按 (decision_time, symbol) 排序、每行 = 一个 (T, sym) 单元的长表;
      我们 pivot 到宽表 → 每个 operand 列拿 .values。
    - **同一时间窗口内出现过的所有 symbol 都进 panel**;某 symbol 不在某天 universe →
      该天的 row 在 panel 里是 NaN(由 dataset.py 已自然 reindex)。
    - future return 用 close:`y_h = log(close).shift(-h) - log(close)`(panel 级 shift,
      跨 day 边界时 NaN 自然传播)。
    - 加载范围 = [extra_lead_days 之前, end];前导日给 rolling 算子暖机用。
    - parquet 文件名约定 'YYYY-MM.parquet'。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.config import ini
from evaluation import universe as univ
from markets.vocabulary import CRYPTO_PARQUET_COLUMNS

# parquet 月文件读并行度:pyarrow 读时释放 GIL → ThreadPool 近线性 overlap I/O。
# Pass2 瞬时 transient(每 worker 一个月 df)~百 MB 级,远低于 panels_full 常驻。
_N_IO_WORKERS = min(8, (os.cpu_count() or 4))


# ============================================================================
# 配置
# ============================================================================

DEFAULT_PARQUET_FUNDING_ROOT: str = './data/parquet_funding'
DEFAULT_PRIMARY_HORIZON_BARS: int = 24      # 24 × 5min = 2 hours
DEFAULT_LEAD_DAYS: int = 2                  # 暖机窗口前置天数


# ============================================================================
# 输出 bundle
# ============================================================================

@dataclass
class PanelBundle:
    panels: Dict[str, np.ndarray]   # 每个 operand 的 (T, S) 面板
    y_future: np.ndarray                      # (T, S) future log-return
    timestamps: pd.DatetimeIndex              # 长度 T
    symbols: List[str]                        # 长度 S
    train_mask: np.ndarray                    # (T,) bool — 实际归属"训练区间"的行
    val_mask:   np.ndarray                    # (T,) bool — 验证区间
    funding_rate: np.ndarray                  # (T, S) float32 — 大多 0,funding bar 处填实际 rate(8h/4h/1h 自动对齐 5min 整点)
    slippage_vol: np.ndarray                  # (T, S) float32 — intra_rv,sqrt-impact 模型的 σ_5m(C bt 入口按需升 f64)
    opens:        np.ndarray                  # (T, S) float64 — next-bar fill 用,ffilled
    bar_quote_volume: np.ndarray              # (T, S) float64 — USDT 计成交量,sqrt-impact 分母
    highs:        np.ndarray                  # (T, S) float64 — bar high,L3 maker intra-bar fill 判定
    lows:         np.ndarray                  # (T, S) float64 — bar low,同上
    member_mask:  np.ndarray = None           # (T, S) bool — 因果 universe 可持仓资格
                                              # (trailing qv-window hourly top-N + grace,live rotation 复刻);
                                              # 持仓 valid = listed(qvol>0) ∧ member。行数据 ≠ 资格:
                                              # 行只供特征/ts 历史,可交易性由本 mask 把关。

    @property
    def T(self) -> int: return len(self.timestamps)

    @property
    def S(self) -> int: return len(self.symbols)


# ============================================================================
# 月份迭代
# ============================================================================

def month_iter(start: str, end: str) -> List[str]:
    """返回 'YYYY-MM' 字符串序列,inclusive。start/end 都是 'YYYY-MM' 或 'YYYY-MM-DD'。"""
    if len(start) > 7: start = start[:7]
    if len(end) > 7: end = end[:7]
    sy, sm = map(int, start.split('-'))
    ey, em = map(int, end.split('-'))
    out: List[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f'{y:04d}-{m:02d}')
        m += 1
        if m == 13:
            y += 1; m = 1
    return out


# ============================================================================
# 加载
# ============================================================================

def load_panel(
    train_start: str,
    train_end: str,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    *,
    parquet_root: str,
    primary_horizon: int = DEFAULT_PRIMARY_HORIZON_BARS,
    lead_days: int = DEFAULT_LEAD_DAYS,
) -> PanelBundle:
    """加载 [train_start, train_end] ∪ [val_start, val_end] 范围的 5m 面板。

    传日期时仅保留月份级精度(月文件已经是月分区)。返回的 panel 横跨整个范围,
    train_mask / val_mask 标识每行属于哪个 split。

    universe/alpha 分离:行数据(parquet 稀疏性)只决定特征/ts 历史的可用性;
    **可持仓资格由 member_mask 单独承载**(因果 trailing qv-window hourly top-N + grace,
    evaluation/universe.py)。消费方持仓 valid = listed(qvol>0)∧ member。
    membership 从 parquet 自身 qvol 派生因果 trailing top-N。
    bt 端 close ffill → sym 退出后下一 bar 按上次 close 交割。
    """
    root = Path(parquet_root)

    # 加载月份范围:从 train_start 提前 lead_days(向前一个月足够)
    pre_start = (datetime.fromisoformat(train_start) - timedelta(days=lead_days)).strftime('%Y-%m')

    if val_start is None:
        full_end = train_end
    else:
        full_end = val_end if val_end else val_start

    months = month_iter(pre_start, full_end)

    needed_cols = ['decision_time', 'symbol',
                   'close', 'open', 'high', 'low', 'quote_volume', 'intra_rv'] + list(CRYPTO_PARQUET_COLUMNS)
    needed_cols = list(dict.fromkeys(needed_cols))
    panel_cols = [c for c in needed_cols if c not in ('decision_time', 'symbol')]

    train_lo = pd.Timestamp(train_start, tz='UTC')
    train_hi = pd.Timestamp(train_end, tz='UTC') + pd.Timedelta(days=1)
    if val_start:
        val_lo = pd.Timestamp(val_start, tz='UTC')
        val_hi = pd.Timestamp(val_end, tz='UTC') + pd.Timedelta(days=1) if val_end else val_lo + pd.Timedelta(days=1)
    else:
        val_lo = val_hi = None
    keep_lo = pd.Timestamp(pre_start + '-01', tz='UTC')
    keep_hi = max(train_hi, val_hi) if val_hi is not None else train_hi

    # ---------- Pass 1:因果 universe(trailing qv-window hourly top-N + grace,live rotation 复刻) ----------
    # universe 是一等因果 artifact(evaluation/universe.py,与 live 同语义内核:
    # live fetch_quote_volume_ranks 同为 1h kline 滚 qv_window_hours 和,精确对齐非近似):
    #   - membership = 从 parquet 自身 qvol 派生的**因果 trailing top-N**。行存在性
    #     = 历史候选集:候选缺失只致边缘 rank 噪声,不授予未来信息 membership。
    #   - 列并集 = 加载窗内 ever-holdable(列存在 ≠ 资格;逐 bar 资格由 member_mask 把关)。
    # 28 个 operand panel(10 f64 + 18 f32)× T_full × S 常驻内存;全史 ever-member ~669,
    # S=669 峰值 ≈ 46.8 GB(110 GB 容器富余)。全保留 ever-holdable,不按未来在场度砍名
    # (按未来 holdable-hours 砍尾 = universe-selection lookahead)。
    univ_top_n   = int(ini('trade_universe', 'top_n', 30))
    univ_persist = int(ini('trade_universe', 'hold_max_persist', 4))
    univ_qvw     = int(ini('trade_universe', 'qv_window_hours', 4))   # 与 live rotation 同窗(单一参数源)

    def _qv_one(mo: str):
        p = root / f'{mo}.parquet'
        if not p.exists():
            return None
        df = pd.read_parquet(p, columns=['decision_time', 'symbol', 'quote_volume'])
        ts_ms = (pd.to_datetime(df['decision_time'], utc=True)
                 .values.astype('datetime64[ms]').astype(np.int64))
        bucket = ((ts_ms - 1) // univ.HOUR_MS + 1) * univ.HOUR_MS    # ceil 到整点(hour H 含 close∈(H−1h,H])
        qv = pd.to_numeric(df['quote_volume'], errors='coerce').fillna(0.0).to_numpy()
        return (pd.DataFrame({'h': bucket, 'symbol': df['symbol'].to_numpy(), 'qv': qv})
                .groupby(['h', 'symbol'], sort=False)['qv'].sum())

    _t = time.time()
    qv_parts = []
    with ThreadPoolExecutor(max_workers=_N_IO_WORKERS) as _ex:
        for res in _ex.map(_qv_one, months):
            if res is not None:
                qv_parts.append(res)
    if not qv_parts:
        raise FileNotFoundError(f'no parquet months in {months} under {parquet_root}')
    qv_long = pd.concat(qv_parts).groupby(level=[0, 1]).sum()
    panel_syms = sorted(qv_long.index.get_level_values('symbol').unique())

    # membership = 因果 trailing top-N(parquet 自身 qv,trailing 窗 PIT;行存在性 = 历史候选集,
    # 不授予未来信息)。
    h_arr = qv_long.index.get_level_values('h').to_numpy(np.int64)
    s_arr = qv_long.index.get_level_values('symbol').map(
        {s: j for j, s in enumerate(panel_syms)}).to_numpy(np.int64)
    h0 = h_arr.min()
    hours_ms = np.arange(h0, h_arr.max() + univ.HOUR_MS, univ.HOUR_MS, dtype=np.int64)
    qv_h = np.zeros((len(hours_ms), len(panel_syms)))
    qv_h[(h_arr - h0) // univ.HOUR_MS, s_arr] = qv_long.to_numpy()
    member = univ.trailing_membership(qv_h, univ_top_n, window_hours=univ_qvw)
    mem_syms = panel_syms
    del qv_h
    holdable = univ.grace_mask(member, univ_persist)                  # (H, S_mem)
    ever = holdable.any(axis=0)
    panel_set = set(panel_syms)
    kept_syms = sorted(s for s, e in zip(mem_syms, ever) if e and s in panel_set)
    n_no_rows = sum(1 for s, e in zip(mem_syms, ever) if e and s not in panel_set)
    if n_no_rows:
        print(f'[adapter] membership 覆盖缺口:{n_no_rows} 个 ever-member sym 无 parquet 行(bt 不可持)', flush=True)
    _kept_j = np.array([mem_syms.index(s) for s in kept_syms], dtype=np.int64)
    holdable_kept = holdable[:, _kept_j]                              # (H, S_kept)
    print(f'[adapter] Pass1 causal universe {time.time()-_t:.0f}s: '
          f'member/hour={member.sum(1).mean():.1f} ever-holdable={int(ever.sum())} kept S={len(kept_syms)}',
          flush=True)
    del qv_long, member, holdable

    # Pass 1.5:per-month 仅读 decision_time 列,枚举 timestamps 与 T_mo,以便 Pass 2 预分配。
    # panels_full 体量大,vstack 双倍峰值会 OOM,必须 in-place 写入到预分配
    # (T_total, S_full) 数组,避免 list-vstack 拷贝。
    def _times_one(mo: str):
        p = root / f'{mo}.parquet'
        if not p.exists():
            return mo, None
        df = pd.read_parquet(p, columns=['decision_time'])
        dt = pd.to_datetime(df['decision_time'], utc=True)
        dt = dt[(dt >= keep_lo) & (dt < keep_hi)]
        times_mo = pd.DatetimeIndex(dt.unique()).sort_values()
        return mo, (times_mo if len(times_mo) else None)

    _t = time.time()
    times_per_mo: Dict[str, pd.DatetimeIndex] = {}
    total_t = 0
    with ThreadPoolExecutor(max_workers=_N_IO_WORKERS) as _ex:
        for mo, times_mo in _ex.map(_times_one, months):   # map 保序 → 插入序 = month 序
            if times_mo is not None:
                times_per_mo[mo] = times_mo
                total_t += len(times_mo)
    print(f'[adapter] Pass1.5 timestamps {time.time()-_t:.0f}s', flush=True)
    if total_t == 0:
        raise FileNotFoundError(f'no parquet rows in [{keep_lo}, {keep_hi}) under {parquet_root}')

    sym_full_idx = {s: j for j, s in enumerate(kept_syms)}
    S_full = len(kept_syms)

    # Pass 2:预分配 (total_t, S_full) per-col,per-month pivot 后 in-place 写入。
    # 混合 dtype(端到端 IC 定标过):
    #   f64 必留 — 价格族(open/high/low/close:ts_delta(log) 相邻相减抵消 + bt equity/强平 accounting)、
    #     量列(volume/quote_volume/taker_buy_quote:ts_corr(·, close) 大数相消把 f32 的 6e-8 舍入
    #     放大到 ΔIC 1e-5~4e-5)、oi_log_ret(对数收益,在 div/max(·,price) 里病态放大)。
    #   sum_oi/sum_oi_value 可降 f32(max|ΔIC|=1.4e-7;parquet 存储仍 f64,此处 fill 时 downcast)。
    #   其余 operand + 派生 TENURE_NORM 本就 f32(ratio/归一化/bounded)→ 常驻内存砍半。
    #   C 算子/bt 入口 PyArray_FROMANY(NPY_DOUBLE) 按需临时升 f64,内核零改;operand 直返 panel
    #   view 不进 cache → f32 不触发二次 alloc。
    # cuda 路径:ops_cuda DTYPE=f32,panel 传 GPU 必 f64→f32 下转 → 存 f64 纯浪费(GPU 结果逐位相同),
    #   全 f32 把全 5m 常驻从 ~26GB 砍到 ~20GB(适配 40GB GPU gear,免 OOM)。
    # cpu 路径:C-ops ts_corr(vol,close) 大数相消需 f64(否则 ΔIC 1e-5~4e-5)→ 保留 8 个价/量/oi 列 f64。
    PANEL_F64_COLS = frozenset() if ini('evaluator', 'device', 'cpu') == 'cuda' else frozenset(
        ('open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_quote_volume', 'oi_log_ret'))
    panels_full: Dict[str, np.ndarray] = {
        col: np.full((total_t, S_full), np.nan,
                     dtype=(np.float64 if col in PANEL_F64_COLS else np.float32))
        for col in panel_cols
    }
    # 各 present month 在 panels_full 的行偏移(month 序累加)。
    present_months = [mo for mo in months if mo in times_per_mo]
    offsets: Dict[str, int] = {}
    _acc = 0
    for mo in present_months:
        offsets[mo] = _acc
        _acc += len(times_per_mo[mo])

    def _fill_one(mo: str) -> None:
        # 读月文件 → numpy scatter 写 panels_full 的 disjoint 行段(代替 per-col unstack/reindex)。
        # 各月行段互不重叠 → 多线程写同一数组不同区域安全;pyarrow 读时释放 GIL → overlap I/O。
        df = pd.read_parquet(root / f'{mo}.parquet', columns=needed_cols,
                             filters=[('symbol', 'in', kept_syms)])
        dt = pd.to_datetime(df['decision_time'], utc=True)
        m = ((dt >= keep_lo) & (dt < keep_hi)).to_numpy()
        if not m.any():
            return
        times_mo = times_per_mo[mo]
        # 每行 (ts,sym) → (全局行, 全局列);times_mo ⊇ 本月 ts 故 get_indexer 全命中,缺失 (ts,sym) cell 保持 NaN(=reindex 语义)。
        row = offsets[mo] + times_mo.get_indexer(pd.DatetimeIndex(dt[m]))
        col = df['symbol'].map(sym_full_idx).to_numpy(np.int64)[m]
        for c in panel_cols:
            panels_full[c][row, col] = df[c].to_numpy()[m].astype(panels_full[c].dtype, copy=False)

    _t = time.time()
    with ThreadPoolExecutor(max_workers=_N_IO_WORKERS) as _ex:
        for _ in _ex.map(_fill_one, present_months):   # 消费迭代器 → 等齐所有写入 + 抛出 worker 异常
            pass
    print(f'[adapter] Pass2 panel fill {time.time()-_t:.0f}s', flush=True)

    _parts = [times_per_mo[mo] for mo in present_months]
    timestamps = _parts[0].append(_parts[1:]) if len(_parts) > 1 else _parts[0]
    timestamps = pd.DatetimeIndex(timestamps)
    all_symbols = kept_syms

    # ---------- member_mask:小时级 holdable → bar 级(stamp = 最后一个 ≤ ts 的整点) ----------
    _ts_ms = (timestamps.tz_convert('UTC').tz_localize(None)
              .values.astype('datetime64[ms]').astype(np.int64))
    member_mask = univ.member_at(hours_ms, holdable_kept, _ts_ms)     # (T, S) bool
    print(f'[adapter] member_mask: per-bar holdable mean={member_mask.sum(1).mean():.1f}/{len(kept_syms)}',
          flush=True)
    del holdable_kept
    # close/open/high/low/quote_volume/intra_rv 同时是 operand 列(进 panels)和 bt 辅助列,
    # 复用同一份 ndarray 引用,无 copy(intra_rv 2026-06-24 起也作 operand,故不再 pop)。
    close_panel       = panels_full['close']
    open_panel_full   = panels_full['open']
    high_panel_full   = panels_full['high']
    low_panel_full    = panels_full['low']
    qvol_panel_full   = panels_full['quote_volume']
    vol_panel_full    = panels_full['intra_rv']

    # symbols 已在 Pass 1 按"train 区有任一 valid close"预过滤,无需二次卡。
    symbols_kept = all_symbols
    t_arr = timestamps.tz_convert('UTC').tz_localize(None).values
    train_lo_ns = np.datetime64(train_lo.tz_convert('UTC').tz_localize(None))
    train_hi_ns = np.datetime64(train_hi.tz_convert('UTC').tz_localize(None))
    train_idx = (t_arr >= train_lo_ns) & (t_arr < train_hi_ns)

    # panels_full 由 np.full 直接预分配 → 已 C-contig,ascontiguousarray 是 no-op view,无 copy。
    # panels 键 = parquet 列名(= operand 身份);tenure_norm 为合成 operand 另行注入。
    panels_arr: Dict[str, np.ndarray] = {
        c: panels_full[c] for c in CRYPTO_PARQUET_COLUMNS
    }

    # ---------- metrics 因果对齐:发布延迟统一 lag-1 ----------
    # /futures/data/* bin T 实际发布于 T+0.8–3.1min(takerlongshortRatio 6.4–8.4min),
    # vision zip 落终值 → 不 shift 即默认偷看未来发布。live 端 panel lag-1 merge 后,
    # 此处同构 shift 使训练/回测与 live 决策因果严格一致(全列统一 lag-1,不为 taker 开特例)。
    # 派生列(oi_log_ret/oi_chg_N)对全局 shift 与"先 shift raw 再重算"严格交换,直接 shift。
    for c in ('sum_oi', 'sum_oi_value',
              'oi_log_ret', 'oi_chg_48', 'oi_chg_288',
              'count_top_lsr', 'sum_top_lsr',
              'count_lsr', 'sum_taker_lsr'):
        arr = panels_arr[c]
        arr[1:, :] = arr[:-1, :]              # numpy 重叠赋值自带 overlap 检测,安全
        arr[:1, :] = np.nan
    open_panel = open_panel_full
    # qvol/vol 需 NaN→0:就地修改(不再 copy)。后续 close 不再被改,共享同一 buffer。
    np.nan_to_num(qvol_panel_full, copy=False, nan=0.0)
    np.nan_to_num(vol_panel_full,  copy=False, nan=0.0)
    qvol_panel = qvol_panel_full
    vol_panel  = vol_panel_full

    # ---------- 派生 operand: TENURE_NORM (recency) ----------
    # tenure = 连续 qvol>0 的 bar 数(run-length);qvol==0(未上市/掉出 top-30)处重置归零。
    # = exp(−tenure/288):tenure=1→0.997,1d(288)→0.368,3d→0.05。把"刚进 top-30"显式成
    # 可挖矿特征(此前经 universe membership 隐式入)。
    # 未上市处置 NaN(与其它 operand 同约定:cs 算子 NaN-safe 自动排除)。合成 operand(无 parquet 列)。
    _idx = np.arange(total_t, dtype=np.int64)[:, None]
    _last_out = np.maximum.accumulate(np.where(qvol_panel > 0.0, -1, _idx), axis=0)
    _tenure = (_idx - _last_out).astype(np.float64)
    panels_arr['tenure_norm'] = np.where(qvol_panel > 0.0,
                                         np.exp(_tenure / -288.0), np.nan).astype(np.float32)
    del _idx, _last_out, _tenure

    # ---------- future return ----------
    # log_close 一次 alloc(代替 np.where + np.log 两次 alloc):mask + 直接 in-place log。
    log_close = np.full_like(close_panel, np.nan)
    pos = close_panel > 0.0
    np.log(close_panel, where=pos, out=log_close)
    y_future = np.full_like(log_close, np.nan)
    h = primary_horizon
    if h > 0:
        np.subtract(log_close[h:, :], log_close[:-h, :], out=y_future[:-h, :])
    del log_close

    # ---------- masks ----------
    train_mask = train_idx
    if val_lo is not None:
        val_lo_ns = np.datetime64(val_lo.tz_convert('UTC').tz_localize(None))
        val_hi_ns = np.datetime64(val_hi.tz_convert('UTC').tz_localize(None))
        val_mask = (t_arr >= val_lo_ns) & (t_arr < val_hi_ns)
    else:
        val_mask = np.zeros_like(train_mask)

    # ---------- funding rate ----------
    funding_panel = _load_funding(months, timestamps, symbols_kept,
                                  Path(DEFAULT_PARQUET_FUNDING_ROOT))

    # pyarrow mimalloc 池归还:parquet 读流的 Arrow 缓冲 del 后 pool 逻辑归零但物理页不还 OS
    # (峰值数 GB 钉死 RSS,OOM 帮凶)。读流已尽,放掉。
    import pyarrow as _pa
    _pa.default_memory_pool().release_unused()

    return PanelBundle(
        panels=panels_arr,
        y_future=y_future,
        timestamps=timestamps,
        symbols=symbols_kept,
        train_mask=train_mask,
        val_mask=val_mask,
        funding_rate=funding_panel,
        slippage_vol=vol_panel,
        opens=open_panel,
        bar_quote_volume=qvol_panel,
        highs=high_panel_full,
        lows=low_panel_full,
        member_mask=member_mask,
    )


def _load_funding(months: List[str], timestamps: pd.DatetimeIndex,
                  symbols: List[str], funding_root: Path) -> np.ndarray:
    """读 parquet_funding/{ym}.parquet,对齐到 (T, S) 面板。
    Binance perp funding 间隔随 symbol/时段变化(主流 8h;2024 起部分 alt 切 4h;
    2025-Q2 起少量 1h),全部都是 5min 整点 → `floor('5min')` 后查 ts dict 自动对齐。
    无数据 → 该 cell = 0。"""
    T, S = len(timestamps), len(symbols)
    # f32:funding 是 1e-4 级 ratio(rel err 6e-8 → abs 6e-12 可忽略),full-T f64 白占 1GB;
    # C bt 入口 FROMANY 按需升格(仅 gate/stage4/val 低频路径付瞬时),torch bt_lite 本就 f32。
    out = np.zeros((T, S), dtype=np.float32)

    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    # 把 timestamps round 到 5min(应已对齐),做 dict 查询
    ts_arr = timestamps.tz_convert('UTC').values  # np.datetime64[ns]
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    for mo in months:
        p = funding_root / f'{mo}.parquet'
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        # 把 fundingTime round 到 5min(吸收 16:00:00.001 这种边界毛刺)
        ft = pd.to_datetime(df['fundingTime'], utc=True).dt.floor('5min')
        ft_arr = ft.values
        for ts, sym, rate in zip(ft_arr, df['symbol'].values, df['fundingRate'].values):
            i = ts_to_idx.get(ts)
            j = sym_to_idx.get(sym)
            if i is not None and j is not None:
                out[i, j] = rate
    return np.ascontiguousarray(out)


# ============================================================================
# 工具:截取 mask 内的子面板(view,不拷贝)
# ============================================================================

def split_train_val(bundle: PanelBundle
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """train/val 拆分,两段各独立 anon 拷贝。

    train 段拷贝供局级 worker fork COW 共享;两段都 .copy():连续行切片 view 已 C-contig,
    ascontiguousarray 是 no-op,返回原 view 会让 base 链钉死全长 buffer 永不释放。逐列 pop +
    copy 后 del full → bundle.panels/y_future 全长 anon 随之逐列释放。要求 train/val mask 各自
    连续(load_panel 构造保证);消费完 y_future 置 None。
    """
    tr_idx = np.where(bundle.train_mask)[0]
    va_idx = np.where(bundle.val_mask)[0]
    tr_sl = slice(int(tr_idx[0]), int(tr_idx[-1]) + 1)
    va_sl = slice(int(va_idx[0]), int(va_idx[-1]) + 1)

    train_panels: Dict[str, np.ndarray] = {}
    val_panels:   Dict[str, np.ndarray] = {}
    for tok in list(bundle.panels):
        full = bundle.panels.pop(tok)
        train_panels[tok] = full[tr_sl].copy()
        val_panels[tok] = full[va_sl].copy()
        del full
    train_y = bundle.y_future[tr_sl].copy()
    val_y = bundle.y_future[va_sl].copy()
    bundle.y_future = None
    return train_panels, train_y, val_panels, val_y


def slice_bundle(bundle: PanelBundle, mask: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """返回 (panels_subset, y_subset)。panels_subset 与 y_subset 都为 mask 区间的连续切片。
    若 mask 不是连续区间(比如 train+val 之间有 hole),用 fancy indexing(会拷贝)。
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError('empty mask')
    contiguous = (idx[-1] - idx[0] + 1) == len(idx)
    if contiguous:
        sl = slice(idx[0], idx[-1] + 1)
        panels_sub = {tok: arr[sl] for tok, arr in bundle.panels.items()}
        y_sub = bundle.y_future[sl]
    else:
        panels_sub = {tok: np.ascontiguousarray(arr[idx]) for tok, arr in bundle.panels.items()}
        y_sub = np.ascontiguousarray(bundle.y_future[idx])
    return panels_sub, y_sub


def slice_bt_aux(bundle: PanelBundle, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """切 (funding_rate, slippage_vol, opens, bar_quote_volume) 与 slice_bundle 同步。
    返回顺序固定,caller 解包 4 个数组。"""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError('empty mask')
    contiguous = (idx[-1] - idx[0] + 1) == len(idx)
    if contiguous:
        sl = slice(idx[0], idx[-1] + 1)
        return (bundle.funding_rate[sl], bundle.slippage_vol[sl],
                bundle.opens[sl], bundle.bar_quote_volume[sl])
    return (np.ascontiguousarray(bundle.funding_rate[idx]),
            np.ascontiguousarray(bundle.slippage_vol[idx]),
            np.ascontiguousarray(bundle.opens[idx]),
            np.ascontiguousarray(bundle.bar_quote_volume[idx]))
