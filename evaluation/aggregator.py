"""1m kline + binance metrics → 5m panel(47 列)聚合,纯函数.

每符号独立处理。跨月 rolling features 由 caller 在前面拼接 prev-month 尾部 5m bars
再调用 add_rolling_features,然后切回当月。

公式来源:逆向 data/parquet_5m/*.parquet vs data/parquet_1m + parquet_metrics 实测对齐。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1) 1m → 5m bar (basic OHLCV + intra-bar features)
# ---------------------------------------------------------------------------

def aggregate_1m_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """单符号 1m → 5m。decision_time = bin close_time = 5m 窗口右端(open+5min)。

    1m schema 必须含: open_time(tz-UTC), open, high, low, close, volume, quote_volume,
                    number_of_trades, taker_buy_quote_volume, symbol
    5m 输出: decision_time, symbol, OHLCV, QV, NT, TBQV, vwap +
              intra_rv, intra_up_minute_ratio, intra_vol_concentration,
              intra_taker_imb_mean, intra_taker_imb_std
    """
    if df_1m.empty:
        return pd.DataFrame()

    df = df_1m.sort_values('open_time').reset_index(drop=True)
    sym = df['symbol'].iloc[0]
    # bin5 = 1m bar 所属的 5m bar 的 decision_time(右端,即 open_time floor + 5min)
    bin5 = (df['open_time'].dt.floor('5min') + pd.Timedelta(minutes=5)).rename('decision_time')
    # 1m close-to-close log return(全局,跨 bin 桥接;intra_rv 用)
    df['_log_cc_1m'] = np.log(df['close'] / df['close'].shift(1))

    # 1m intra-bar 派生
    qv = df['quote_volume']
    tbqv = df['taker_buy_quote_volume']

    g = df.groupby(bin5, sort=True)
    agg = g.agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        quote_volume=('quote_volume', 'sum'),
        number_of_trades=('number_of_trades', 'sum'),
        taker_buy_quote_volume=('taker_buy_quote_volume', 'sum'),
    )
    # intra_*: per-5m-bar 的 1m 聚合量。2026-05-26 互控增量 IC 否决删除:up_minute_ratio /
    # vol_concentration / taker_imb_mean / taker_imb_std(被 KYLE_12/VPIN_12 吸收或噪声致命)。
    # 保留 intra_rv(bt slippage);新增 sum_abs_sf/ret 给 vpin/kyle。
    intra = pd.DataFrame(index=agg.index)
    intra['intra_rv'] = np.sqrt(g.apply(lambda d: float((d['_log_cc_1m'] ** 2).sum())))
    # vpin/kyle building blocks(真 1m Σ|·|,与 offline build_5m_panel 同源;1m 粒度 Σ|·| ≠ 5m 后取 abs):
    #   intra_sum_abs_sf = Σ|2·taker_buy − qv|_1m(VPIN 分子 / KYLE 分母);intra_sum_abs_ret = Σ|log_cc_1m|(KYLE 分子)
    intra['intra_sum_abs_sf']  = df.assign(_absf=(2.0 * tbqv - qv).abs()).groupby(bin5)['_absf'].sum()
    intra['intra_sum_abs_ret'] = df.assign(_absr=df['_log_cc_1m'].abs()).groupby(bin5)['_absr'].sum()
    df.drop(columns=['_log_cc_1m'], inplace=True)

    out = pd.concat([agg, intra], axis=1).reset_index()
    if out['decision_time'].dt.tz is None:
        out['decision_time'] = out['decision_time'].dt.tz_localize('UTC')
    out['symbol'] = sym

    # 单 bar 派生(不依赖 lag)。vwap 已删(2026-05-26:= qv/volume,close 近别名,冗余)。
    out['log_ret_oc'] = np.log(out['close'] / out['open'])
    # 实测 parquet_5m: range_pct = (H-L)/C, body_pct = (C-O)/C(signed,非 abs)
    out['range_pct'] = (out['high'] - out['low']) / out['close']
    out['body_pct'] = (out['close'] - out['open']) / out['close']
    out['log_volume'] = np.log(out['volume'] + 1.0)
    out['avg_trade_size'] = out['quote_volume'] / out['number_of_trades']
    out['taker_imbalance'] = (2.0 * out['taker_buy_quote_volume'] - out['quote_volume']) / out['quote_volume']

    return out


# ---------------------------------------------------------------------------
# 2) merge metrics (OI / LSR)
# ---------------------------------------------------------------------------

def merge_metrics(df_5m: pd.DataFrame, df_metrics: pd.DataFrame) -> pd.DataFrame:
    """metrics(5m freq, create_time)joined 进 5m panel。

    metrics schema: create_time, symbol, sum_open_interest, sum_open_interest_value,
                    count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
                    count_long_short_ratio, sum_taker_long_short_vol_ratio
    输出列 rename 对齐 parquet_5m 命名:
        sum_oi, sum_oi_value, count_top_lsr, sum_top_lsr,
        count_lsr, sum_taker_lsr, oi_value_per_count
    """
    rename = {
        'sum_open_interest': 'sum_oi',
        'sum_open_interest_value': 'sum_oi_value',
        'count_toptrader_long_short_ratio': 'count_top_lsr',
        'sum_toptrader_long_short_ratio': 'sum_top_lsr',
        'count_long_short_ratio': 'count_lsr',
        'sum_taker_long_short_vol_ratio': 'sum_taker_lsr',
    }
    m = df_metrics[['create_time', 'symbol', *rename.keys()]].copy()
    # 实测 metrics create_time 偶有 +2s 漂移(8636/8640 对齐 5min 整点),merge 前 floor 修齐
    m['create_time'] = m['create_time'].dt.floor('5min')
    m = m.rename(columns={**rename, 'create_time': 'decision_time'})
    # 同 (decision_time, symbol) 若多行(漂移后冲突):取首条
    m = m.drop_duplicates(subset=['decision_time', 'symbol'], keep='first')
    out = df_5m.merge(m, on=['decision_time', 'symbol'], how='left')
    out['oi_value_per_count'] = out['sum_oi_value'] / out['sum_oi']
    return out


# ---------------------------------------------------------------------------
# 3) rolling / lagged features (per-symbol time series)
# ---------------------------------------------------------------------------

def add_lag_and_oi_logret(df: pd.DataFrame) -> pd.DataFrame:
    """计算 log_ret_1, oi_log_ret, oi_value_log_ret(per symbol,前一行 lag)。"""
    df = df.sort_values(['symbol', 'decision_time']).reset_index(drop=True)
    g = df.groupby('symbol', sort=False)
    df['log_ret_1'] = np.log(df['close'] / g['close'].shift(1))
    df['oi_log_ret'] = np.log(df['sum_oi'] / g['sum_oi'].shift(1))
    df['oi_value_log_ret'] = np.log(df['sum_oi_value'] / g['sum_oi_value'].shift(1))
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """rolling per-symbol。须 caller 已 prepend 上一月尾部供 288-bar warmup。

    rv_N      = sqrt(sum(log_ret_oc^2 last N))            N ∈ {12, 48, 144, 288}
    parkinson_48 = sqrt(sum(ln(H/L)^2 last 48) / (4·ln2))
    ret_N     = ln(C[t] / C[t-N])                         N ∈ {12, 48, 144, 288}
    vol_zscore_288 = z-score(log_volume, 288, ddof=1)
    taker_imb_ema_48 = EMA(taker_imbalance, span=48, adjust=True)
    oi_chg_N  = sum(oi_log_ret last N)                    N ∈ {48, 288}
    ma_dist_N = (close - MA(close, N)) / MA(close, N)     N ∈ {48, 288}
    """
    df = df.sort_values(['symbol', 'decision_time']).reset_index(drop=True)

    log_ret_1_sq = df['log_ret_1'] ** 2  # 5m close-to-close 平方(实测 rv_N 用此)

    def _g(col):
        return df.groupby('symbol', sort=False)[col]

    # rv_N = sqrt(sum(log_ret_1² last N))(close-to-close 5m,非 oc)
    for n in (12, 48, 144, 288):
        df[f'rv_{n}'] = np.sqrt(log_ret_1_sq.groupby(df['symbol']).transform(
            lambda s: s.rolling(n, min_periods=n).sum()))

    # parkinson_48 已删(2026-05-26:rv_N 已覆盖波动估计,冗余)

    # ret_N (cumulative log return last N: ln(C[t]/C[t-N]))
    g_close = _g('close')
    for n in (12, 48, 144, 288):
        df[f'ret_{n}'] = np.log(df['close'] / g_close.shift(n))

    # vol_zscore_288 on log_volume (ddof=1)
    g_lv = _g('log_volume')
    lv_mean = g_lv.transform(lambda s: s.rolling(288, min_periods=288).mean())
    lv_std = g_lv.transform(lambda s: s.rolling(288, min_periods=288).std(ddof=1))
    df['vol_zscore_288'] = (df['log_volume'] - lv_mean) / lv_std

    # taker_imb_ema_48 (per-symbol EWM)
    df['taker_imb_ema_48'] = _g('taker_imbalance').transform(
        lambda s: s.ewm(span=48, adjust=True).mean())

    # oi_chg_N = sum(oi_log_ret last N)
    g_oilr = _g('oi_log_ret')
    for n in (48, 288):
        df[f'oi_chg_{n}'] = g_oilr.transform(lambda s: s.rolling(n, min_periods=n).sum())

    # ma_dist_N
    for n in (48, 288):
        ma = g_close.transform(lambda s: s.rolling(n, min_periods=n).mean())
        df[f'ma_dist_{n}'] = (df['close'] - ma) / ma

    # 真 1m trailing-window 风险特征(W=12 个 5m bar = 60 个 1m bar = 1h),= offline build_5m_panel 同源:
    #   vpin_12 = Σ|sf_1m| / Σqv  (order-flow toxicity);kyle_12 = Σ|r_1m| / Σ|sf_1m|  (价格冲击/illiq)
    # building block(intra_sum_abs_sf/ret)已是 per-5m-bar 的 1m Σ|·| → rolling-12-sum 即 60-1m 真值。
    eps = 1e-12
    sum_sf_12 = _g('intra_sum_abs_sf').transform(lambda s: s.rolling(12, min_periods=12).sum())
    sum_r_12  = _g('intra_sum_abs_ret').transform(lambda s: s.rolling(12, min_periods=12).sum())
    sum_qv_12 = _g('quote_volume').transform(lambda s: s.rolling(12, min_periods=12).sum())
    df['vpin_12'] = sum_sf_12 / (sum_qv_12 + eps)
    df['kyle_12'] = sum_r_12 / (sum_sf_12 + eps)

    return df


# ---------------------------------------------------------------------------
# 4) full pipeline (single-month, multi-symbol, with warmup prepend)
# ---------------------------------------------------------------------------

# 47 列最终 schema(对齐现有 parquet_5m)
SCHEMA_COLUMNS = [
    'decision_time', 'symbol',
    'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'number_of_trades',
    'taker_buy_quote_volume',
    'log_ret_1', 'log_ret_oc', 'range_pct', 'body_pct', 'log_volume',
    'avg_trade_size', 'taker_imbalance',
    'intra_rv', 'intra_sum_abs_sf', 'intra_sum_abs_ret',
    'sum_oi', 'sum_oi_value', 'count_top_lsr', 'sum_top_lsr',
    'count_lsr', 'sum_taker_lsr',
    'oi_value_per_count', 'oi_log_ret', 'oi_value_log_ret',
    'rv_12', 'rv_48', 'rv_144', 'rv_288',
    'ret_12', 'ret_48', 'ret_144', 'ret_288',
    'vol_zscore_288', 'taker_imb_ema_48',
    'oi_chg_48', 'oi_chg_288',
    'ma_dist_48', 'ma_dist_288',
    'vpin_12', 'kyle_12',
]


# ---------------------------------------------------------------------------
# 4b) premium / funding 派生(P0 enrich 列;offline enrich + live LivePanel 共享数学源)
#     parquet_5m 默认不含这 3 列(SCHEMA_COLUMNS 之外),由 enrich_5m_panel in-place 加;
#     live 由 LivePanel 用同一对函数算 → train==live。2026-05-24 bit-exact 复刻验证通过。
# ---------------------------------------------------------------------------

PREMIUM_FUNDING_COLUMNS = ['premium_index_5m', 'funding_rate_interp', 'funding_countdown_norm']


def agg_premium(df_panel: pd.DataFrame, premium_by_sym: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """premium_index_5m = premiumIndexKlines.close,(mark − index)/index(币安已算好)。
    premium_by_sym: {sym -> df(open_time, close)};decision_time = open_time + 5min(右端对齐)。
    返回 (decision_time, symbol, premium_index_5m)。"""
    out = df_panel[['decision_time', 'symbol']].copy()
    pieces = []
    for sym, df in premium_by_sym.items():
        d = df[['open_time', 'close']].copy()
        d['decision_time'] = d['open_time'] + pd.Timedelta(minutes=5)
        d['symbol'] = sym
        d = d.rename(columns={'close': 'premium_index_5m'})[
            ['decision_time', 'symbol', 'premium_index_5m']]
        pieces.append(d)
    if not pieces:
        out['premium_index_5m'] = np.nan
        return out
    prm = pd.concat(pieces, ignore_index=True).drop_duplicates(['decision_time', 'symbol'], keep='first')
    return out.merge(prm, on=['decision_time', 'symbol'], how='left')


def agg_funding(df_panel: pd.DataFrame, df_funding: pd.DataFrame) -> pd.DataFrame:
    """funding_rate_interp / funding_countdown_norm 2 col。
    df_funding: (decision_time=raw funding_time[未floor], symbol, funding_rate)。
    period per-symbol 实测(8h 或 4h):next_event − prev_event,末行 fallback prev+8h。
    interp = 当前周期 carry(上次 funding rate,常数);countdown = clip((t−prev)/(next−prev),0,1)。
    关键:用 raw 未 floor 的 funding_time + backward asof(floor 会抹掉 sub-ms 抖动致错位)。"""
    out = df_panel[['decision_time', 'symbol']].copy()
    # merge_asof 要求两侧 join key 分辨率一致(新版 pandas 严格);panel 与 funding 事件的
    # datetime 可能 us/ms 不同 → 统一到 ns(最细,无损,tz 保留,sub-ms 抖动不丢)。
    out['decision_time'] = out['decision_time'].dt.as_unit('ns')
    if df_funding is None or df_funding.empty:
        out['funding_rate_interp'] = np.nan
        out['funding_countdown_norm'] = np.nan
        return out
    f = df_funding[['decision_time', 'symbol', 'funding_rate']].sort_values(
        ['symbol', 'decision_time']).reset_index(drop=True)
    f['decision_time'] = f['decision_time'].dt.as_unit('ns')
    f['next_funding_time'] = f.groupby('symbol', sort=False)['decision_time'].shift(-1)
    f['next_funding_time'] = f['next_funding_time'].fillna(f['decision_time'] + pd.Timedelta(hours=8))
    out_pieces = []
    for sym, gp in out.groupby('symbol', sort=False):
        fp = f[f['symbol'] == sym][['decision_time', 'funding_rate', 'next_funding_time']].rename(
            columns={'decision_time': 'funding_time'})
        if fp.empty:
            out_pieces.append(gp.assign(funding_rate_interp=np.nan, funding_countdown_norm=np.nan))
            continue
        merged = pd.merge_asof(
            gp.sort_values('decision_time'), fp.sort_values('funding_time'),
            left_on='decision_time', right_on='funding_time', direction='backward')
        dt       = (merged['decision_time']     - merged['funding_time']).dt.total_seconds()
        period_s = (merged['next_funding_time'] - merged['funding_time']).dt.total_seconds()
        merged['funding_countdown_norm'] = (dt / period_s.replace(0, np.nan)).clip(0.0, 1.0)
        merged['funding_rate_interp']    = merged['funding_rate']
        out_pieces.append(merged[['decision_time', 'symbol',
                                  'funding_rate_interp', 'funding_countdown_norm']])
    return pd.concat(out_pieces, ignore_index=True)


def build_5m_for_month(
    df_1m_window: pd.DataFrame,        # 1m: target month + ≥1 day prepend (实际 caller 给本月 1m)
    df_metrics_window: pd.DataFrame,   # metrics: 同窗口
    df_warmup_5m: pd.DataFrame | None, # 上月最后 N 个 5m bar(N≥288),用于 rolling warmup
) -> pd.DataFrame:
    """target month 1m + metrics → 5m,返回 47 列(rolling 已用 warmup 计算)。

    df_warmup_5m: schema 与 SCHEMA_COLUMNS 一致(可缺 lag 列只要含 close/log_ret_oc/log_volume/sum_oi 等)。
                  None 时 rolling 列前 288 行 NaN。
    输出范围:caller 切片 decision_time ∈ target month。
    """
    pieces = []
    for sym, g1 in df_1m_window.groupby('symbol', sort=True):
        five = aggregate_1m_to_5m(g1)
        if five.empty:
            continue
        m_sym = df_metrics_window[df_metrics_window['symbol'] == sym]
        five = merge_metrics(five, m_sym)
        pieces.append(five)
    if not pieces:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    cur = pd.concat(pieces, ignore_index=True)

    # 拼 warmup(只取与本月 symbol 集合相交的)+ 当月,统一计算 lag/rolling,再切回当月。
    # 重叠去重:warm 已含完整 metrics + 派生列,边界 5m bar 由 warm 兜底(cur 在跨月桥接时
    # 可能 metrics 缺失);keep='first' 保 warm 行,新月份非重叠行用 cur。
    if df_warmup_5m is not None and not df_warmup_5m.empty:
        warm = df_warmup_5m[df_warmup_5m['symbol'].isin(cur['symbol'].unique())].copy()
        full = pd.concat([warm, cur], ignore_index=True)
        full = full.drop_duplicates(subset=['decision_time', 'symbol'], keep='first').reset_index(drop=True)
    else:
        full = cur

    full = add_lag_and_oi_logret(full)
    full = add_rolling_features(full)

    # 切回 target month 范围(caller 通过 decision_time 切;此处保留全 set,由 caller 决定切片)
    return full[SCHEMA_COLUMNS]


# ---------------------------------------------------------------------------
# 5) dense (T,S) 5m → 1h operand-panel 下采样(alpha-eval 提速;train adapter + live panel 共享)
# ---------------------------------------------------------------------------
# 动机:alpha 决策本就 1h cadence(gate_dec 步长 12),公式在全 5m 面板上求值是唯一仍在 5m 的热
# 路径(eval_tree ~82% 由 ts/pair 滚动窗口主导)。把 alpha 算子读的 operand 面板降到 1h →
# ts/pair O(T) 缩 12×。data/ 的 5m 量价特征一个不删:此函数只在 load 时下采样,5m parquet 仍是源。
#
# 因果对齐(无 lookahead 铁律):1h row k = 截至决策 bar dec5m[k] 的尾部 12-bar(=1h)聚合
# (右沿 = 决策点 → row k 只含 ≤ dec5m[k] 的数据),与当前"在 bar dec5m[k] 上算 alpha(ts 历史
# ≤ dec5m[k])"严格同因果,只是分辨率由 5m 变 1h。
#
# 逐字段规则:原始流量 sum、OHLC first/max/min/last、其余(CLOSE + 全部派生/比率/快照/微结构
# 特征)右沿采样(= 决策 bar 的值)→ 保留 5min/1m 算出的派生特征值(vpin/kyle/rv/vol_zscore/…)。
from evaluation.grammar import OperandToken as _OT

_AGG_SUM   = frozenset({_OT.VOLUME, _OT.QUOTE_VOLUME, _OT.NUMBER_OF_TRADES, _OT.TAKER_BUY_QUOTE})
_AGG_FIRST = frozenset({_OT.OPEN})
_AGG_MAX   = frozenset({_OT.HIGH})
_AGG_MIN   = frozenset({_OT.LOW})
# 其余(CLOSE + 全部派生)→ 右沿采样(last),保留 5min-派生值


def aggregate_5m_to_1h(panels_5m: dict) -> dict:
    """dense (T,S) 5m operand 面板 → (n,S) 1h,n = T//12。row k = 截至 **block-start** bar 12k 的
    尾部 12-bar 聚合(右沿 = 12k = 决策点 = 当前 5m bt 的 dec block-start)。

    决策格 = arange(0, T, 12)(block-start,与 broadcast_dec_to_5m 的 [12k:12k+12) 持仓块、
    build_wealth_ctx 的 di=gate_lo+12k 同相)→ 调用侧 gate_dec_1h = gate_dec // 12,broadcast/
    wealth_ctx/gate_bt 全不动。因果:row k 用 bars [12k−11, 12k](≤ 12k),与当前"在 bar 12k
    上算 alpha(ts 历史 ≤ 12k)、持仓 [12k, 12k+12)"严格同因果;前 pad 11 NaN 行使 row 0 =
    [仅 bar 0](warmup,与 ts 暖机区一致)。scripts/smoke_1h_panel_eval.py future-perturbation 钉死无 lookahead。

    逐字段:原始流量 sum、OHLC first/max/min/last、其余(CLOSE + 全部派生)右沿采样(保留 5m 值)。"""
    T = next(iter(panels_5m.values())).shape[0]
    n = T // 12
    out = {}
    for tok, arr in panels_5m.items():
        S = arr.shape[1]
        pad = np.full((11, S), np.nan, dtype=arr.dtype)
        a = np.concatenate([pad, arr], axis=0)[:12 * n]           # row k = a[12k:12k+12] = bars [12k−11, 12k]
        blk = a.reshape(n, 12, S)                                 # (n,12,S);blk[:,-1] = 右沿 = bar 12k
        if   tok in _AGG_SUM:   r = blk.sum(axis=1)
        elif tok in _AGG_MAX:   r = blk.max(axis=1)
        elif tok in _AGG_MIN:   r = blk.min(axis=1)
        elif tok in _AGG_FIRST: r = blk[:, 0]
        else:                   r = blk[:, -1]                    # CLOSE + 派生:右沿采样
        out[tok] = np.ascontiguousarray(r, dtype=arr.dtype)
    return out
