/* backtest/engine/engine.cpp —— 市场无关组合回测引擎核心 + 决策窗 stop 扫描。
 *
 * bar 循环与算术顺序逐段对照旧 portfolio_bt.c(回归门 bit-identical);市场专属块
 * (成本 / funding·carry / isolated·cross 强平)改为 MarketPolicy 回调:
 *   (a) MTM open gap → cash 底线 → (b) rebalance(policy.trade_cost)→ (c) MTM 余 →
 *   (c2) policy.isolated_liq → (c3) trailing-stop(policy.trade_cost 出场)→
 *   (d) policy.carry → (e) policy.cross_margin_liq。
 * stop_scan = 决策窗 trailing-stop latch 自动机(per-symbol 列独立,OMP over S;f32/f64 双路)。
 */
#include "engine.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bt {

namespace {

constexpr double BT_STOP_WMIN = 0.02;   /* latch 释放阈 = executor._STOP_WMIN */

/* 持久 scratch(thread_local std::vector:防并发 BT race + RAII;resize 保 capacity → grow-only 复用)。 */
thread_local std::vector<double>  g_weight_TS, g_pos, g_target, g_stp_ext, g_stp_entry, g_stp_sgn;
thread_local std::vector<uint8_t> g_stp_flags;

/* step 1: alpha → weight 行内变换(市场无关:cs_zscore → clip(±3) → demean → L1 cap → conc cap)。 */
inline void row_signal_to_weight(const double* a, double* w, int64_t S,
                                 double cap, double max_concentration) {
    double sum = 0.0, sumsq = 0.0;
    int64_t n = 0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        if (std::isfinite(v)) { sum += v; sumsq += v * v; n++; }
    }
    if (n < 2) { for (int64_t s = 0; s < S; s++) w[s] = 0.0; return; }
    double m   = sum / (double)n;
    double var = (sumsq - (double)n * m * m) / (double)(n - 1);
    double sd  = (var > 0.0) ? std::sqrt(var) : 0.0;
    if (sd < 1e-12) { for (int64_t s = 0; s < S; s++) w[s] = 0.0; return; }
    double inv_sd = 1.0 / sd;

    double clip_sum = 0.0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        double z = std::isfinite(v) ? (v - m) * inv_sd : 0.0;
        if (z >  3.0) z =  3.0;
        if (z < -3.0) z = -3.0;
        w[s] = z;
        clip_sum += z;
    }
    const double clip_mean = clip_sum / (double)n;
    double l1 = 0.0;
    for (int64_t s = 0; s < S; s++) {
        double v = a[s];
        if (!std::isfinite(v)) { w[s] = 0.0; continue; }
        w[s] -= clip_mean;
        l1   += std::fabs(w[s]);
    }
    double inv_l1 = (l1 > cap && cap > 0.0) ? (cap / l1) : 1.0;
    for (int64_t s = 0; s < S; s++) w[s] *= inv_l1;

    if (max_concentration > 0.0) {
        for (int64_t s = 0; s < S; s++) {
            if (w[s] >  max_concentration) w[s] =  max_concentration;
            if (w[s] < -max_concentration) w[s] = -max_concentration;
        }
    }
}

} // anonymous namespace


void run_portfolio(
    const double* alpha, const double* close_TS, const double* open_TS,
    const double* high_TS, const double* low_TS,
    const double* bar_volume_TS, const double* sigma_TS,
    const double* funding_TS, const double* leverage_TS,
    const int8_t* trade_block,
    int64_t T, int64_t S,
    double initial_cash, int64_t skip_warmup,
    double leverage_cap, double max_concentration, int raw_weights,
    double stop_trail_pct, const double* stop_ratchet, int64_t n_ratchet,
    const MarketPolicy& policy,
    double* equity, double* turnover, double* cost_drag, int64_t* liq_at_out)
{
    g_weight_TS.resize((size_t)T * (size_t)S);
    g_pos.resize(S); g_target.resize(S); g_stp_ext.resize(S);
    g_stp_entry.resize(S); g_stp_sgn.resize(S); g_stp_flags.resize(S);
    double*  weight_TS = g_weight_TS.data();
    double*  pos       = g_pos.data();
    double*  target    = g_target.data();
    double*  stp_ext   = g_stp_ext.data();
    double*  stp_entry = g_stp_entry.data();
    double*  stp_sgn   = g_stp_sgn.data();
    uint8_t* stp_flags = g_stp_flags.data();

    const bool iso_active = policy.isolated_active();
    const int  stop_active = (stop_trail_pct > 0.0);

    /* === Step 1: alpha → weight,跨 t 并行 === */
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double* w_row = &weight_TS[t * S];
        if (t < skip_warmup) {
            for (int64_t s = 0; s < S; s++) w_row[s] = 0.0;
        } else if (raw_weights) {
            const double* a_row = &alpha[t * S];
            for (int64_t s = 0; s < S; s++) {
                double v = a_row[s];
                w_row[s] = std::isfinite(v) ? v : 0.0;
            }
        } else {
            row_signal_to_weight(&alpha[t * S], w_row, S, leverage_cap, max_concentration);
        }
    }

    /* === Step 2: 组合 bt,串行 T,inner SIMD over S === */
    for (int64_t s = 0; s < S; s++) {
        pos[s] = 0.0; target[s] = 0.0;
        stp_ext[s] = 0.0; stp_entry[s] = 0.0; stp_sgn[s] = 0.0; stp_flags[s] = 0;
    }

    double cash = initial_cash;
    *liq_at_out = -1;
    equity[0] = cash; turnover[0] = 0.0; cost_drag[0] = 0.0;

    for (int64_t t = 1; t < T; t++) {
        const double* p_open  = &open_TS[t * S];
        const double* p_close = &close_TS[t * S];
        const double* p_prev  = &close_TS[(t - 1) * S];
        const double* sigma_r = sigma_TS      ? &sigma_TS[t * S]      : nullptr;
        const double* qvol_r  = bar_volume_TS ? &bar_volume_TS[t * S] : nullptr;
        const double* w_lag   = &weight_TS[(t - 1) * S];

        BarRow bar;
        bar.close      = p_close;
        bar.prev_close = p_prev;
        bar.high       = high_TS     ? &high_TS[t * S]     : nullptr;
        bar.low        = low_TS      ? &low_TS [t * S]     : nullptr;
        bar.funding    = funding_TS  ? &funding_TS[t * S]  : nullptr;
        bar.leverage   = leverage_TS ? &leverage_TS[t * S] : nullptr;

        /* (a) MTM 跨 open gap */
        for (int64_t s = 0; s < S; s++) cash += pos[s] * (p_open[s] - p_prev[s]);

        if (cash <= 0.0) {                                  /* cash 底线(市场无关硬停)*/
            cash = 0.0;
            for (int64_t s = 0; s < S; s++) pos[s] = 0.0;
            *liq_at_out = t;
            equity[t] = 0.0; turnover[t] = 0.0; cost_drag[t] = 0.0;
            for (int64_t j = t + 1; j < T; j++) { equity[j] = 0.0; turnover[j] = 0.0; cost_drag[j] = 0.0; }
            return;
        }

        /* (b) rebal + 执行:单 bar 直达 target(成本经 policy)*/
        const double equity_pre_trade = cash;
        double notional_total = 0.0;
        double cost_total     = 0.0;

        for (int64_t s = 0; s < S; s++) {
            double op = p_open[s];
            target[s] = (op > 0.0) ? (w_lag[s] * equity_pre_trade / op) : 0.0;
            if (qvol_r && !(qvol_r[s] > 0.0)) target[s] = 0.0;
        }
        const int8_t* tb_row = trade_block ? &trade_block[t * S] : nullptr;
        for (int64_t s = 0; s < S; s++) {
            double op = p_open[s];
            if (!(op > 0.0)) continue;
            double tgt = (stop_active && (stp_flags[s] & 2)) ? 0.0 : target[s];
            if (tb_row) {                                   /* 涨跌停/停牌:受限方向冻结仓位(target←pos)*/
                const int8_t b = tb_row[s];
                const double d = tgt - pos[s];
                if (((b & 1) && d > 0.0) || ((b & 2) && d < 0.0)) tgt = pos[s];
            }
            const double delta = tgt - pos[s];
            const double notional = std::fabs(delta) * op;
            if (notional <= 0.0) { pos[s] = tgt; continue; }
            const double sig = (sigma_r && std::isfinite(sigma_r[s])) ? sigma_r[s] : 0.0;
            const double bv  = qvol_r ? qvol_r[s] : 0.0;
            const double cost = policy.trade_cost(notional, sig, bv, delta < 0.0);   /* delta<0=减仓(卖)*/
            cash    -= cost;
            pos[s]   = tgt;
            notional_total += notional;
            cost_total     += cost;
        }
        turnover[t]  = (equity_pre_trade > 0.0) ? notional_total / equity_pre_trade : 0.0;
        cost_drag[t] = cost_total;

        /* (c) MTM 余下 */
        for (int64_t s = 0; s < S; s++) cash += pos[s] * (p_close[s] - p_open[s]);

        /* (c2) isolated-wick 强平(policy)*/
        if (iso_active) policy.isolated_liq(pos, bar, S, t, cash, cost_drag[t], *liq_at_out);

        /* (c3) per-name 自入场峰值 trailing stop + latch(市场无关;出场成本经 policy)。
         * 注:stop-out 强平出场**不查 trade_block**(受限方向理论上挡不住 stop 平仓)。当前无冲突 —
         * trade_block 仅 ASharePolicy 用(其 binding 恒传 stop_trail_pct=0 → stop_active=0),而 perp/equity
         * 用 stop 但 trade_block=nullptr;两者**构造上互斥**,绝不同 bar 同时活。若未来某市场二者并用,
         * 须在此显式让 stop-out 也尊重 trade_block 的跌停禁卖。 */
        if (stop_active) {
            const double* w_now = w_lag;
            for (int64_t s = 0; s < S; s++) {
                const double w = w_now[s];
                const int held = std::isfinite(w) && (std::fabs(w) > BT_STOP_WMIN);
                if (!held) { stp_flags[s] = 0; continue; }
                if (stp_flags[s] & 2) continue;
                const double c = p_close[s];
                if (!(c > 0.0)) continue;
                const double sgn = (w > 0.0) ? 1.0 : -1.0;
                if (!(stp_flags[s] & 1) || sgn != stp_sgn[s]) {
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
                for (int64_t k = 0; k < n_ratchet; k++)
                    if (gain >= stop_ratchet[2 * k]) thr = stop_ratchet[2 * k + 1];
                if (adverse >= thr) {
                    stp_flags[s] |= 2;
                    if (pos[s] != 0.0) {
                        const double notl = std::fabs(pos[s]) * c;
                        const double stop_cost = policy.trade_cost(notl, 0.0, 0.0, pos[s] > 0.0);  /* 平多=卖 */
                        cash -= stop_cost;
                        cost_drag[t]   += stop_cost;
                        notional_total += notl;
                        pos[s] = 0.0;
                    }
                }
            }
            turnover[t] = (equity_pre_trade > 0.0) ? notional_total / equity_pre_trade : 0.0;
        }

        /* (d) carry(funding / 借券费)*/
        policy.carry(pos, bar, S, cash);

        /* (e) cross-margin 强平(policy;iso_active 时由 c2 取代故跳过)*/
        if (!iso_active) {
            if (policy.cross_margin_liq(pos, p_close, S, t, cash, cost_drag[t], *liq_at_out)) {
                equity[t] = cash;
                for (int64_t j = t + 1; j < T; j++) { equity[j] = cash; turnover[j] = 0.0; cost_drag[j] = 0.0; }
                return;
            }
        }

        equity[t] = cash;
    }
}


/* ============================ 决策窗 stop latch 扫描 ============================ */
#define BT_DEFINE_STOP_SCAN(NAME, REAL, FABS)                                       \
void NAME(                                                                          \
    const REAL* w_TS, const REAL* close_TS, const int64_t* start_idx,               \
    int64_t T_dec, int64_t S, int64_t bars_per_dec,                                 \
    REAL trail, const REAL* ratchet, int64_t n_ratchet,                             \
    REAL* keep, int64_t* jstar, uint8_t* fired)                                     \
{                                                                                   \
    _Pragma("omp parallel for schedule(static)")                                    \
    for (int64_t s = 0; s < S; s++) {                                               \
        int anchored = 0, stopped = 0;                                              \
        REAL ext = (REAL)0, entry = (REAL)1, sgn = (REAL)0;                         \
        for (int64_t t = 0; t < T_dec; t++) {                                       \
            const REAL w = w_TS[t * S + s];                                         \
            const int held = std::isfinite(w) && (FABS(w) > (REAL)0.02);            \
            stopped = stopped && held;                                              \
            anchored = anchored && held;                                            \
            keep[t * S + s] = stopped ? (REAL)0 : (REAL)1;                          \
            const REAL* cl = &close_TS[start_idx[t] * S + s];                       \
            const REAL c0 = cl[0];                                                  \
            const REAL w_sgn = (w > (REAL)0) ? (REAL)1                              \
                             : ((w < (REAL)0) ? (REAL)-1 : (REAL)0);                \
            const int anchor = (held && !anchored) ||                               \
                               (held && anchored && !stopped && w_sgn != sgn);      \
            if (anchor) {                                                           \
                ext   = c0;                                                         \
                entry = (c0 > (REAL)1e-12) ? c0 : (REAL)1e-12;                      \
                sgn   = w_sgn;                                                      \
            }                                                                       \
            anchored = anchored || held;                                            \
            const int act = held && !stopped;                                       \
            int64_t fj = bars_per_dec;                                              \
            int trig = 0;                                                           \
            if (act) {                                                              \
                const int is_long = sgn > (REAL)0;                                  \
                REAL e = ext;                                                       \
                for (int64_t j = 1; j <= bars_per_dec; j++) {                       \
                    const REAL c = cl[j * S];                                       \
                    if (is_long) { if (c > e) e = c; }                              \
                    else         { if (c < e) e = c; }                              \
                    const REAL c_safe = (c > (REAL)1e-12) ? c : (REAL)1e-12;        \
                    const REAL e_safe = (e > (REAL)1e-12) ? e : (REAL)1e-12;        \
                    const REAL adverse = is_long ? (REAL)1 - c_safe / e_safe        \
                                                 : c_safe / e_safe - (REAL)1;       \
                    const REAL gain = is_long ? e_safe / entry - (REAL)1            \
                                              : entry / e_safe - (REAL)1;           \
                    REAL thr = trail;                                               \
                    for (int64_t k = 0; k < n_ratchet; k++)                         \
                        if (gain >= ratchet[2 * k]) thr = ratchet[2 * k + 1];       \
                    if (adverse >= thr) { trig = 1; fj = j; break; }                \
                }                                                                   \
                if (trig) stopped = 1;                                              \
                else      ext = e;                                                  \
            }                                                                       \
            jstar[t * S + s] = fj;                                                  \
            fired[t * S + s] = (uint8_t)trig;                                       \
        }                                                                           \
    }                                                                               \
}

BT_DEFINE_STOP_SCAN(stop_scan_f64, double, std::fabs)
BT_DEFINE_STOP_SCAN(stop_scan_f32, float,  std::fabs)
#undef BT_DEFINE_STOP_SCAN

} // namespace bt
