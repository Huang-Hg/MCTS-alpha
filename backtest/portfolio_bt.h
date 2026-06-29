/* backtest/portfolio_bt.h
 *
 * 组合级 backtest 内核 — 单 bar 直达 target,全 taker @ open
 * ============================================================================
 *
 * 实现要点:
 *   - scratch 用 _Thread_local:Py_BEGIN_ALLOW_THREADS 释放 GIL,多 Python 线程并发 BT 安全。
 *
 * 流程(per bar t,t ≥ 1):
 *   1. weight[t-1, :] = cs_zscore(α) → clip(±3) → demean → L1_norm
 *   2. MTM cash += pos_prev · (open[t] − close[t-1])
 *   3. rebal:target = w_lag · cash / open;delta = target − pos;notional = |delta|·op
 *      cost_rate = half_spread + fee + sqrt-impact;cash -= notional · cost_rate;pos = target
 *   4. MTM cash += pos · (close[t] − open[t])
 *   5. trailing stop;6. funding;7. 强平(isolated wick 或 cross-margin)
 *
 * 输出 caller 预分配:
 *   equity (T,) / turnover (T,) / cost_drag (T,) / liq_at int64
 */

#ifndef BT_PORTFOLIO_H
#define BT_PORTFOLIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void bt_portfolio(
    const double* alpha,                /* (T, S) row-major */
    const double* close_TS,             /* (T, S) ffilled */
    const double* open_TS,              /* (T, S) ffilled */
    const double* high_TS,              /* (T, S),isolated wick 强平用;NULL → cross-margin */
    const double* low_TS,               /* (T, S),同上 */
    const double* bar_volume_TS,        /* (T, S) USDT,NULL → 跳 impact */
    const double* sigma_TS,             /* (T, S) intra_rv,NULL → impact 用 0 */
    const double* funding_TS,           /* (T, S),0 = 非 funding bar,NULL → 不扣 */
    int64_t T, int64_t S,
    /* portfolio params */
    double initial_cash,
    int64_t skip_warmup,
    /* cost params(均为 fraction)*/
    double half_spread_rate,
    double fee_rate,
    double impact_Y,
    /* risk */
    double mmr,
    /* L1 norm cap:Σ|w| ≤ leverage_cap(1.0=1×gross;5.0=模拟 live 5× 杠杆)*/
    double leverage_cap,
    /* Per-symbol weight cap:|w[s]| ≤ max_concentration。≤ 0 = 禁用。
     * L1 cap 之后做 per-symbol clip;clip 后 Σ|w| 可能 < cap。*/
    double max_concentration,
    /* raw_weights bypass:1 = 跳过 step 1(_row_signal_to_weight),caller 已给 final weights;
     * 0 = 走 cs_zscore+clip+demean+L1+per_sym。
     * RL 路径(per-sym lev + bias)用 raw_weights=1 完全 bypass C 内核的 signal-to-weight。
     * leverage_cap / max_concentration 在 raw=1 时被忽略。*/
    int raw_weights,
    /* per-symbol isolated 杠杆 L_s:(T, S),NULL → cross-margin 组合层强平(step e)。
     * 非 NULL → per-symbol isolated wick 强平:单名独立,逐 bar 用 low(多)/high(空)判
     * prev_close·(1∓(1/L_s−mmr)) 击穿 → 平该单名(锁定保证金损失),不拖累其它腿。参考用 prev_close
     * (close 链不 stale;open 面板 ffill 有 stale 值会假触发)。取代组合层 cross-margin(两者互斥);
     * close-only step e 系统性低估单名插针强平,isolated wick 修正。 */
    const double* leverage_TS,
    /* per-name 自入场峰值 trailing stop + latch(live executor._eval_stops 复刻)。
     * stop_trail_pct ≤ 0 → 关闭。每 bar 以 close 评估(≈ live 60s stop_tick 的 5m 近似):
     *   held = |w_lag| > 0.02(WMIN,= executor._STOP_WMIN);新持仓/未 latch 翻向 → 锚
     *   ext=entry=open[入场 bar](≈fill 价);多头 adverse = 1−close/ext ≥ thr、空头
     *   adverse = close/ext−1 ≥ thr → MARKET 平 @ close + latch(钳 target=0,翻向不解锁),
     *   直到 |w_lag| ≤ WMIN(alpha 撤名)才释放。ratchet:峰值浮盈过里程碑收紧 thr,
     *   stop_ratchet = flattened 升序 [gain_0, trail_0, gain_1, trail_1, ...],NULL → 无。 */
    double stop_trail_pct,
    const double* stop_ratchet,
    int64_t n_ratchet,
    /* outputs(caller 预分配,长度 T)*/
    double* equity,
    double* turnover,
    double* cost_drag,
    int64_t* liq_at_out
);

/* bt_lite 决策窗 trailing-stop latch 扫描(rl_bt_torch._stop_latch_scan 的 C 实现)。
 * 与 (c3) 同一自动机但不同口径:锚 ext=entry=close[fill bar](= 窗首 close,start_idx[t] 行),
 * 窗内 j=1..bars_per_dec 逐 close 评,跨窗携带 (anchored, stopped, ext, entry, sgn)。
 * per-symbol 列独立 → OMP 并行。f32 变体逐位对齐 torch float32 语义(常量按 REAL cast)。
 * 输出:keep (T_dec,S) 0/1;jstar (T_dec,S) ∈[1,bars_per_dec] 有效路径终点;fired (T_dec,S) 0/1。
 * 前置条件(binding 校验):start_idx[t] + bars_per_dec < T5m。 */
void bt_stop_scan_f64(
    const double* w_TS, const double* close_TS, const int64_t* start_idx,
    int64_t T_dec, int64_t S, int64_t bars_per_dec,
    double trail, const double* ratchet, int64_t n_ratchet,
    double* keep, int64_t* jstar, uint8_t* fired);

void bt_stop_scan_f32(
    const float* w_TS, const float* close_TS, const int64_t* start_idx,
    int64_t T_dec, int64_t S, int64_t bars_per_dec,
    float trail, const float* ratchet, int64_t n_ratchet,
    float* keep, int64_t* jstar, uint8_t* fired);

#ifdef __cplusplus
}
#endif

#endif
