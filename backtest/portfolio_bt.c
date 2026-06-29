/* backtest/portfolio_bt.c
 *
 * 组合级 backtest 内核 — 单 bar 直达 target,全 taker @ open
 *
 * 步骤 1(alpha → weight,(T,S) → (T,S))跨 t 并行(OMP)
 * 步骤 2(组合 bt,T 串行,inner loop S 由编译器 SIMD):
 *   每 bar 单 bar 直达 pos=target;cost_rate = half_spread + fee + sqrt-impact。
 *
 * 持久 scratch(grow-only,_Thread_local 多线程独占):
 *   weight_TS (T,S) / pos (S,) / target (S,) / stop 状态 (S,)
 */

#include "portfolio_bt.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/* ---------------- persistent scratch(thread-local 防并发 BT race)----------------
 * Py_BEGIN_ALLOW_THREADS 释放 GIL 后,多 Python 线程可同时进 bt_portfolio。
 * scratch 用 _Thread_local 让每线程独占一份;grow-only 跨调用复用。
 */

static _Thread_local double*  g_weight_TS = NULL;
static _Thread_local double*  g_pos       = NULL;
static _Thread_local double*  g_target    = NULL;   /* (S,) 当 bar target */
static _Thread_local double*  g_stp_ext   = NULL;   /* (S,) trailing stop 有利极值价 */
static _Thread_local double*  g_stp_entry = NULL;   /* (S,) 入场锚价(ratchet 浮盈基准)*/
static _Thread_local double*  g_stp_sgn   = NULL;   /* (S,) 锚定方向 ±1 */
static _Thread_local uint8_t* g_stp_flags = NULL;   /* (S,) bit0=anchored bit1=stopped(latch)*/
static _Thread_local int64_t  g_T_cap     = 0;
static _Thread_local int64_t  g_S_cap     = 0;

#define BT_STOP_WMIN 0.02   /* latch 释放阈 = executor._STOP_WMIN */

static void _ensure_scratch(int64_t T, int64_t S) {
    if (T > g_T_cap || S > g_S_cap) {
        free(g_weight_TS); free(g_pos); free(g_target);
        free(g_stp_ext); free(g_stp_entry); free(g_stp_sgn); free(g_stp_flags);
        g_weight_TS = (double*)malloc((size_t)T * (size_t)S * sizeof(double));
        g_pos       = (double*)malloc((size_t)S * sizeof(double));
        g_target    = (double*)malloc((size_t)S * sizeof(double));
        g_stp_ext   = (double*)malloc((size_t)S * sizeof(double));
        g_stp_entry = (double*)malloc((size_t)S * sizeof(double));
        g_stp_sgn   = (double*)malloc((size_t)S * sizeof(double));
        g_stp_flags = (uint8_t*)malloc((size_t)S * sizeof(uint8_t));
        g_T_cap = T;
        g_S_cap = S;
    }
}


/* ---------------- step 1: alpha → weight 行内变换 ---------------- */
static inline void _row_signal_to_weight(
    const double* a,
    double* w,
    int64_t S,
    double cap,
    double max_concentration
) {
    double sum = 0.0, sumsq = 0.0;
    int64_t n = 0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        if (isfinite(v)) { sum += v; sumsq += v * v; n++; }
    }
    if (n < 2) {
        for (int64_t s = 0; s < S; s++) w[s] = 0.0;
        return;
    }
    double m   = sum / (double)n;
    double var = (sumsq - (double)n * m * m) / (double)(n - 1);
    double sd  = (var > 0.0) ? sqrt(var) : 0.0;
    if (sd < 1e-12) {
        for (int64_t s = 0; s < S; s++) w[s] = 0.0;
        return;
    }
    double inv_sd = 1.0 / sd;

    double clip_sum = 0.0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        double z = isfinite(v) ? (v - m) * inv_sd : 0.0;
        if (z >  3.0) z =  3.0;
        if (z < -3.0) z = -3.0;
        w[s] = z;
        clip_sum += z;
    }

    const double clip_mean = clip_sum / (double)n;   /* dollar-neutral demean(与 live signal 一致)*/
    double l1 = 0.0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        if (!isfinite(v)) { w[s] = 0.0; continue; }
        w[s] -= clip_mean;
        l1   += fabs(w[s]);
    }
    /* L1 cap:Σ|w| ≤ cap。l1 > cap 时缩到 cap;否则保留(允许 net long/short 的细 alpha)*/
    double inv_l1 = (l1 > cap && cap > 0.0) ? (cap / l1) : 1.0;
    for (int64_t s = 0; s < S; s++) w[s] *= inv_l1;

    /* Max concentration cap:|w[s]| ≤ max_concentration。max_concentration ≤ 0 → 禁用。
     * 在 L1 cap 之后做 per-symbol clip;clip 后 Σ|w| 可能 < cap(允许),
     * 集中度限制本质即"用 leverage 换分散度"的 trade-off。
     * sum_oi_value monoculture 派生 alpha 在 cs 后通常把 long/short 全压到 ±1-2 个最极端 symbol,
     * 0.20 cap = 单 symbol 最多 20% gross → 强制至少 5 个 symbol 共享 long/short。*/
    if (max_concentration > 0.0) {
        for (int64_t s = 0; s < S; s++) {
            if (w[s] >  max_concentration) w[s] =  max_concentration;
            if (w[s] < -max_concentration) w[s] = -max_concentration;
        }
    }
}


/* ============================================================================
 * 主入口
 * ============================================================================ */
void bt_portfolio(
    const double* alpha,
    const double* close_TS,
    const double* open_TS,
    const double* high_TS,
    const double* low_TS,
    const double* bar_volume_TS,
    const double* sigma_TS,
    const double* funding_TS,
    int64_t T, int64_t S,
    double initial_cash,
    int64_t skip_warmup,
    double half_spread_rate,
    double fee_rate,
    double impact_Y,
    double mmr,
    double leverage_cap,
    double max_concentration,
    int raw_weights,
    const double* leverage_TS,
    double stop_trail_pct,
    const double* stop_ratchet,
    int64_t n_ratchet,
    double* equity,
    double* turnover,
    double* cost_drag,
    int64_t* liq_at_out
) {
    _ensure_scratch(T, S);
    double*  weight_TS = g_weight_TS;
    double*  pos       = g_pos;
    double*  target    = g_target;
    double*  stp_ext   = g_stp_ext;
    double*  stp_entry = g_stp_entry;
    double*  stp_sgn   = g_stp_sgn;
    uint8_t* stp_flags = g_stp_flags;   /* bit0=anchored bit1=stopped(latch) */

    /* per-symbol isolated wick 强平:leverage_TS + high/low 三者齐 → 启用,取代组合层 cross-margin */
    const int iso_active = (leverage_TS != NULL && high_TS != NULL && low_TS != NULL);
    const int stop_active = (stop_trail_pct > 0.0);
    const double taker_rate = half_spread_rate + fee_rate;

    /* === Step 1: alpha → weight,跨 t 并行 ================================ */
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double* w_row = &weight_TS[t * S];
        if (t < skip_warmup) {
            for (int64_t s = 0; s < S; s++) w_row[s] = 0.0;
        } else if (raw_weights) {
            /* RL 路径:caller 已给 final weights,直接 copy。NaN → 0 防污染下游 PnL。*/
            const double* a_row = &alpha[t * S];
            for (int64_t s = 0; s < S; s++) {
                double v = a_row[s];
                w_row[s] = isfinite(v) ? v : 0.0;
            }
        } else {
            _row_signal_to_weight(&alpha[t * S], w_row, S, leverage_cap, max_concentration);
        }
    }

    /* === Step 2: 组合 bt,串行 T,inner SIMD over S ======================= */
    for (int64_t s = 0; s < S; s++) {
        pos[s] = 0.0; target[s] = 0.0;
        stp_ext[s] = 0.0; stp_entry[s] = 0.0; stp_sgn[s] = 0.0; stp_flags[s] = 0;
    }

    double cash = initial_cash;
    *liq_at_out = -1;

    equity[0]    = cash;
    turnover[0]  = 0.0;
    cost_drag[0] = 0.0;

    for (int64_t t = 1; t < T; t++) {
        const double* p_open  = &open_TS[t * S];
        const double* p_close = &close_TS[t * S];
        const double* p_prev  = &close_TS[(t - 1) * S];
        const double* p_high  = high_TS ? &high_TS[t * S] : NULL;   /* isolated wick 用 */
        const double* p_low   = low_TS  ? &low_TS [t * S] : NULL;
        const double* w_lag   = &weight_TS[(t - 1) * S];
        const double* sigma_r = sigma_TS      ? &sigma_TS[t * S]      : NULL;
        const double* qvol_r  = bar_volume_TS ? &bar_volume_TS[t * S] : NULL;
        const double* fund_r  = funding_TS    ? &funding_TS[t * S]    : NULL;

        /* (a) MTM 跨 open gap */
        for (int64_t s = 0; s < S; s++) {
            cash += pos[s] * (p_open[s] - p_prev[s]);
        }

        if (cash <= 0.0) {
            cash = 0.0;
            for (int64_t s = 0; s < S; s++) pos[s] = 0.0;
            *liq_at_out = t;
            equity[t]    = 0.0;
            turnover[t]  = 0.0;
            cost_drag[t] = 0.0;
            for (int64_t j = t + 1; j < T; j++) {
                equity[j] = 0.0; turnover[j] = 0.0; cost_drag[j] = 0.0;
            }
            return;
        }

        /* (b) rebal + 执行:每 bar target = w_lag · equity / op,单 bar 直达 */
        const double equity_pre_trade = cash;
        double notional_total = 0.0;
        double cost_total     = 0.0;

        for (int64_t s = 0; s < S; s++) {
            double op = p_open[s];
            target[s] = (op > 0.0) ? (w_lag[s] * equity_pre_trade / op) : 0.0;
            /* 未上市/停牌(无真实成交 bar_volume≤0)→ 不持仓:防 ffill 幽灵 K 线(close 恒常数被
             * 前向填充)拿到权重,且其坏 low(填 0)在 isolated wick 上假强平。qvol_r=NULL 时不启用。*/
            if (qvol_r && !(qvol_r[s] > 0.0)) target[s] = 0.0;
        }

        for (int64_t s = 0; s < S; s++) {
            double op = p_open[s];
            if (!(op > 0.0)) continue;

            /* trailing-stop latch:钳 target=0 */
            const double tgt = (stop_active && (stp_flags[s] & 2)) ? 0.0 : target[s];
            const double delta  = tgt - pos[s];
            const double notional = fabs(delta) * op;
            if (notional <= 0.0) {
                pos[s] = tgt;
                continue;
            }

            /* taker @ open + sqrt-impact */
            const double sig = (sigma_r && isfinite(sigma_r[s])) ? sigma_r[s] : 0.0;
            const double bv  = qvol_r ? qvol_r[s] : 0.0;
            const double impact = (bv > 0.0 && impact_Y > 0.0 && sig > 0.0)
                ? impact_Y * sig * sqrt(notional / bv)
                : 0.0;
            const double cost = notional * (taker_rate + impact);
            cash    -= cost;
            pos[s]   = tgt;
            notional_total += notional;
            cost_total     += cost;
        }
        turnover[t]  = (equity_pre_trade > 0.0) ? notional_total / equity_pre_trade : 0.0;
        cost_drag[t] = cost_total;

        /* (c) MTM 余下 */
        for (int64_t s = 0; s < S; s++) {
            cash += pos[s] * (p_close[s] - p_open[s]);
        }

        /* (c2) per-symbol isolated wick 强平(iso_active → 取代组合层 cross-margin step e)。
         * 单名独立:bar 内 low(多)/high(空)击穿 entry·(1∓(1/L_s−mmr)) → 平该名锁定保证金损失,
         * 其它腿 net 不受影响(对齐 binance isolated;捕 close-based 抹掉的 bar 内插针)。*/
        if (iso_active) {
            const double* lev_r = &leverage_TS[t * S];
            for (int64_t s = 0; s < S; s++) {
                if (pos[s] == 0.0) continue;
                double L = lev_r[s];
                if (!(L > 1.0)) continue;                 /* L≤1 → 阈值≥~1,实质不爆 */
                double thr = 1.0 / L - mmr;               /* 单名不利幅度强平阈值 */
                if (!(thr > 0.0)) continue;
                /* 参考价 = prev_close(close[t-1]):close 链不 stale(open 面板有 ffill stale 值,
                 * 会让 low/open 假性极小 → 假强平;cross 路径里 open 项 MTM 抵消故无碍,isolated
                 * 单独用 open 作参考则被污染)。逐 bar 相对前收的 wick 强平,贴近交易所 mark-price。*/
                double ref = p_prev[s];
                if (!(ref > 0.0)) continue;
                double liq_price; int hit = 0;
                if (pos[s] > 0.0) {                        /* 多头:跌破 ref·(1−thr) */
                    liq_price = ref * (1.0 - thr);
                    if (isfinite(p_low[s])  && p_low[s]  > 0.0 && p_low[s]  <= liq_price) hit = 1;
                } else {                                  /* 空头:涨破 ref·(1+thr) */
                    liq_price = ref * (1.0 + thr);
                    if (isfinite(p_high[s]) && p_high[s] > 0.0 && p_high[s] >= liq_price) hit = 1;
                }
                if (!hit) continue;
                /* step(c) 已按 close MTM,改记到 liq_price 平仓:cash += pos·(liq−close) */
                cash         += pos[s] * (liq_price - p_close[s]);
                double liqcost = fabs(pos[s]) * liq_price * (half_spread_rate + fee_rate);
                cash         -= liqcost;
                cost_drag[t] += liqcost;
                pos[s]      = 0.0;
                if (*liq_at_out < 0) *liq_at_out = t;     /* 首个单名强平 bar(诊断,不 flatline)*/
            }
        }

        /* (c3) per-name 自入场峰值 trailing stop + latch(live executor._eval_stops 复刻)。
         * 以 close[t] 评估(≈ live 60s stop_tick 的 5m 近似);held = |w_lag| > WMIN(与
         * executor 用 latch 覆盖前的 _raw_weights 判释放同构):
         *   - latch 释放:alpha 撤名(|w| ≤ WMIN)→ 解锁 + 复位锚(此后允许全新入场)
         *   - 新持仓 / 未 latch 翻向 → 重锚 ext=entry=open[t](≈fill 价),sgn=sign(w);
         *     锚 bar 仍用 close[t] 更新 ext 并可触发(开盘即崩的 bar 不豁免)
         *   - 多头 adverse = 1 − close/ext;空头 adverse = close/ext − 1(价格比空间,
         *     与 executor 一致;注意非 log 对称式)
         *   - ratchet:gain(多 ext/entry−1,空 entry/ext−1)过里程碑 → thr 收紧(升序覆盖)
         *   - 触发 → MARKET 平 @ close(taker 成本)+ latch(翻向不解锁)*/
        if (stop_active) {
            const double* w_now = w_lag;   /* 当前持仓由 w_lag 行生成 */
            for (int64_t s = 0; s < S; s++) {
                const double w = w_now[s];
                const int held = isfinite(w) && (fabs(w) > BT_STOP_WMIN);
                if (!held) { stp_flags[s] = 0; continue; }       /* 撤名:释放 latch + 复位锚 */
                if (stp_flags[s] & 2) continue;                  /* 已 latch:冻结直到撤名 */
                const double c = p_close[s];
                if (!(c > 0.0)) continue;
                const double sgn = (w > 0.0) ? 1.0 : -1.0;
                if (!(stp_flags[s] & 1) || sgn != stp_sgn[s]) {  /* 新持仓 / 未 latch 翻向 → 重锚 */
                    const double op = p_open[s];
                    const double anchor_px = (op > 0.0) ? op : c;
                    stp_ext[s] = anchor_px; stp_entry[s] = anchor_px;
                    stp_sgn[s] = sgn; stp_flags[s] = 1;
                }
                double adverse, gain;
                if (sgn > 0.0) {
                    if (c > stp_ext[s]) stp_ext[s] = c;
                    adverse = 1.0 - c / stp_ext[s];
                    gain    = stp_ext[s] / stp_entry[s] - 1.0;
                } else {
                    if (c < stp_ext[s]) stp_ext[s] = c;
                    adverse = c / stp_ext[s] - 1.0;
                    gain    = stp_entry[s] / stp_ext[s] - 1.0;
                }
                double thr = stop_trail_pct;
                for (int64_t k = 0; k < n_ratchet; k++) {        /* 升序里程碑,最高已过者生效 */
                    if (gain >= stop_ratchet[2 * k]) thr = stop_ratchet[2 * k + 1];
                }
                if (adverse >= thr) {
                    stp_flags[s] |= 2;                           /* latch */
                    if (pos[s] != 0.0) {                         /* MARKET 平 @ close[t] */
                        const double notl = fabs(pos[s]) * c;
                        const double stop_cost = notl * (half_spread_rate + fee_rate);
                        cash -= stop_cost;
                        cost_drag[t]   += stop_cost;
                        notional_total += notl;
                        pos[s] = 0.0;
                    }
                }
            }
            turnover[t] = (equity_pre_trade > 0.0) ? notional_total / equity_pre_trade : 0.0;
        }

        /* (d) funding */
        if (fund_r) {
            for (int64_t s = 0; s < S; s++) {
                double f = fund_r[s];
                if (f != 0.0) cash -= pos[s] * p_close[s] * f;
            }
        }

        /* (e) cross-margin 强平(组合层一刀切;iso_active 时由 (c2) per-sym isolated 取代)*/
        double gmv = 0.0;
        for (int64_t s = 0; s < S; s++) {
            gmv += fabs(pos[s]) * p_close[s];
        }
        if (!iso_active && gmv > 0.0 && cash < mmr * gmv) {
            const double slip5 = 5.0 * (half_spread_rate + fee_rate);
            for (int64_t s = 0; s < S; s++) {
                if (pos[s] == 0.0) continue;
                double exec = (pos[s] > 0.0) ? p_close[s] * (1.0 - slip5)
                                             : p_close[s] * (1.0 + slip5);
                cash += pos[s] * (exec - p_close[s]);
                cash -= fabs(pos[s]) * exec * (half_spread_rate + fee_rate);
                cost_drag[t] += fabs(pos[s]) * exec * (half_spread_rate + fee_rate);
                pos[s] = 0.0;
            }
            if (cash < 0.0) cash = 0.0;
            *liq_at_out = t;
            equity[t] = cash;
            for (int64_t j = t + 1; j < T; j++) {
                equity[j] = cash; turnover[j] = 0.0; cost_drag[j] = 0.0;
            }
            return;
        }

        equity[t] = cash;
    }
}


/* ============================================================================
 * bt_lite 决策窗 trailing-stop latch 扫描(rl_bt_torch._stop_latch_scan 的 C 实现)
 * ============================================================================
 * per-symbol 列自动机独立 → OMP 并行 over S。语义逐条对齐 torch 版(见 .h 注释):
 *   - stopped/anchored 撤名释放 → keep 写出 → 锚定判定(新持仓/未 latch 翻向)
 *   - 窗内 j=1..N 逐 close 评:running ext(cummax/cummin from 窗首 ext_in)、
 *     adverse 价格比、ratchet 升序覆盖 thr、首触发 j 截断 + latch
 *   - 存活名 ext 滚动到窗末 running 极值
 * REAL=float 变体常量全 (REAL) cast,与 torch float32 逐位一致。
 */
#define DEFINE_STOP_SCAN(SUFFIX, REAL, FABS)                                       \
void bt_stop_scan_##SUFFIX(                                                        \
    const REAL* w_TS, const REAL* close_TS, const int64_t* start_idx,              \
    int64_t T_dec, int64_t S, int64_t bars_per_dec,                                \
    REAL trail, const REAL* ratchet, int64_t n_ratchet,                            \
    REAL* keep, int64_t* jstar, uint8_t* fired)                                    \
{                                                                                  \
    _Pragma("omp parallel for schedule(static)")                                   \
    for (int64_t s = 0; s < S; s++) {                                              \
        int anchored = 0, stopped = 0;                                             \
        REAL ext = (REAL)0, entry = (REAL)1, sgn = (REAL)0;                        \
        for (int64_t t = 0; t < T_dec; t++) {                                      \
            const REAL w = w_TS[t * S + s];                                        \
            const int held = isfinite(w) && (FABS(w) > (REAL)0.02);                \
            stopped = stopped && held;            /* 撤名 → 释放 latch */          \
            anchored = anchored && held;          /* 撤名 → 复位锚 */              \
            keep[t * S + s] = stopped ? (REAL)0 : (REAL)1;                         \
            const REAL* cl = &close_TS[start_idx[t] * S + s];   /* 行距 S */       \
            const REAL c0 = cl[0];                                                 \
            const REAL w_sgn = (w > (REAL)0) ? (REAL)1                             \
                             : ((w < (REAL)0) ? (REAL)-1 : (REAL)0);               \
            const int anchor = (held && !anchored) ||                              \
                               (held && anchored && !stopped && w_sgn != sgn);     \
            if (anchor) {                                                          \
                ext   = c0;                                                        \
                entry = (c0 > (REAL)1e-12) ? c0 : (REAL)1e-12;                     \
                sgn   = w_sgn;                                                     \
            }                                                                      \
            anchored = anchored || held;                                           \
            const int act = held && !stopped;                                      \
            int64_t fj = bars_per_dec;                                             \
            int trig = 0;                                                          \
            if (act) {                                                             \
                const int is_long = sgn > (REAL)0;                                 \
                REAL e = ext;                                                      \
                for (int64_t j = 1; j <= bars_per_dec; j++) {                      \
                    const REAL c = cl[j * S];                                      \
                    if (is_long) { if (c > e) e = c; }                             \
                    else         { if (c < e) e = c; }                             \
                    const REAL c_safe = (c > (REAL)1e-12) ? c : (REAL)1e-12;       \
                    const REAL e_safe = (e > (REAL)1e-12) ? e : (REAL)1e-12;       \
                    const REAL adverse = is_long ? (REAL)1 - c_safe / e_safe       \
                                                 : c_safe / e_safe - (REAL)1;      \
                    const REAL gain = is_long ? e_safe / entry - (REAL)1           \
                                              : entry / e_safe - (REAL)1;          \
                    REAL thr = trail;                                              \
                    for (int64_t k = 0; k < n_ratchet; k++)                        \
                        if (gain >= ratchet[2 * k]) thr = ratchet[2 * k + 1];      \
                    if (adverse >= thr) { trig = 1; fj = j; break; }               \
                }                                                                  \
                if (trig) stopped = 1;                                             \
                else      ext = e;            /* 存活名滚动峰值 */                  \
            }                                                                      \
            jstar[t * S + s] = fj;                                                 \
            fired[t * S + s] = (uint8_t)trig;                                      \
        }                                                                          \
    }                                                                              \
}

DEFINE_STOP_SCAN(f64, double, fabs)
DEFINE_STOP_SCAN(f32, float,  fabsf)
#undef DEFINE_STOP_SCAN
