/* backtest/engine/engine.hpp
 *
 * Policy 抽象组合回测引擎 —— 接口头(市场 policy + 引擎/扫描声明)。
 *
 * 市场机制(成本 / carry(funding·借券·分红)/ 强平)抽象成可插拔 MarketPolicy;引擎核心
 * (engine.cpp)市场无关:MTM / rebalance 骨架 / trailing-stop bar 循环,在固定 hook 点回调。
 *   PerpPolicy   —— Binance USDⓈ 永续(crypto):对称 taker + sqrt-impact、funding、
 *                   isolated-wick / cross-margin 强平。**逐位复现** 旧 portfolio_bt.c(回归门)。
 *   EquityPolicy —— US 权益:佣金(+spread)、空腿借券费、无强平(leverage_cap≤1 + cash 底线)。
 *                   未来 A 股 = 新 policy(印花税 / T+1 / 涨跌停),引擎核心不动。
 * 纯 C++(无 Python/numpy),只碰 double*;binding(_bt_module.cpp)构造具体 policy 后传引擎。
 */
#ifndef BT_ENGINE_HPP
#define BT_ENGINE_HPP

#include <cmath>
#include <cstdint>

namespace bt {

/* per-bar 行视图(引擎填,policy 读)。指针为 NULL 表该列未提供。 */
struct BarRow {
    const double* close;       /* close[t]   */
    const double* prev_close;  /* close[t-1] */
    const double* high;        /* high[t] | NULL */
    const double* low;         /* low[t]  | NULL */
    const double* funding;     /* funding[t] | NULL */
    const double* leverage;    /* leverage_TS[t] | NULL(isolated)*/
};


class MarketPolicy {
public:
    virtual ~MarketPolicy() = default;

    /* 单笔 rebalance / 平仓成本。notional=|delta|·price;sig/bv 给 impact(无则传 0)。
     * is_sell=减仓方向(delta<0)→ 非对称成本(如 A 股印花税卖出单边)用;对称成本 policy 忽略。 */
    virtual double trade_cost(double notional, double sigma, double bar_vol, bool is_sell) const = 0;

    /* bar 末 carry:funding(crypto)/ 借券费(equity 空腿)/ 分红;就地改 cash。 */
    virtual void carry(double* pos, const BarRow& bar, int64_t S, double& cash) const = 0;

    /* isolated 强平是否启用(true → 引擎 MTM 后调 isolated_liq、跳过 cross_margin)。 */
    virtual bool isolated_active() const = 0;

    /* (c2) 单名 isolated-wick 强平(MTM-close 后);仅 isolated_active() 时被调。 */
    virtual void isolated_liq(double* pos, const BarRow& bar, int64_t S, int64_t t,
                              double& cash, double& cost_drag_t, int64_t& liq_at) const = 0;

    /* (e) 组合层 cross-margin 强平(carry 后);仅 !isolated_active() 时被调。
     *     返回 true 表已强平 → 引擎 flatline equity[t..T]=cash 并 return。 */
    virtual bool cross_margin_liq(double* pos, const double* close, int64_t S, int64_t t,
                                  double& cash, double& cost_drag_t, int64_t& liq_at) const = 0;
};


/* ============================ Perp(crypto)—— 逐位复现旧内核 ============================ */
class PerpPolicy final : public MarketPolicy {
public:
    PerpPolicy(double half_spread, double fee, double impact_Y, double mmr, bool iso_active)
        : impact_Y_(impact_Y), mmr_(mmr), taker_(half_spread + fee), iso_active_(iso_active) {}

    double trade_cost(double notional, double sigma, double bar_vol, bool /*is_sell*/) const override {
        const double impact = (bar_vol > 0.0 && impact_Y_ > 0.0 && sigma > 0.0)
            ? impact_Y_ * sigma * std::sqrt(notional / bar_vol) : 0.0;
        return notional * (taker_ + impact);            /* 对称(taker 双边),忽略 is_sell */
    }

    void carry(double* pos, const BarRow& bar, int64_t S, double& cash) const override {
        if (!bar.funding) return;
        for (int64_t s = 0; s < S; s++) {
            const double f = bar.funding[s];
            if (f != 0.0) cash -= pos[s] * bar.close[s] * f;
        }
    }

    bool isolated_active() const override { return iso_active_; }

    void isolated_liq(double* pos, const BarRow& bar, int64_t S, int64_t t,
                      double& cash, double& cost_drag_t, int64_t& liq_at) const override {
        const double* lev = bar.leverage;
        for (int64_t s = 0; s < S; s++) {
            if (pos[s] == 0.0) continue;
            const double L = lev[s];
            if (!(L > 1.0)) continue;
            const double thr = 1.0 / L - mmr_;
            if (!(thr > 0.0)) continue;
            const double ref = bar.prev_close[s];
            if (!(ref > 0.0)) continue;
            double liq_price; int hit = 0;
            if (pos[s] > 0.0) {
                liq_price = ref * (1.0 - thr);
                if (std::isfinite(bar.low[s])  && bar.low[s]  > 0.0 && bar.low[s]  <= liq_price) hit = 1;
            } else {
                liq_price = ref * (1.0 + thr);
                if (std::isfinite(bar.high[s]) && bar.high[s] > 0.0 && bar.high[s] >= liq_price) hit = 1;
            }
            if (!hit) continue;
            cash         += pos[s] * (liq_price - bar.close[s]);
            const double liqcost = std::fabs(pos[s]) * liq_price * taker_;
            cash         -= liqcost;
            cost_drag_t  += liqcost;
            pos[s]        = 0.0;
            if (liq_at < 0) liq_at = t;
        }
    }

    bool cross_margin_liq(double* pos, const double* close, int64_t S, int64_t t,
                          double& cash, double& cost_drag_t, int64_t& liq_at) const override {
        double gmv = 0.0;
        for (int64_t s = 0; s < S; s++) gmv += std::fabs(pos[s]) * close[s];
        if (!(gmv > 0.0) || !(cash < mmr_ * gmv)) return false;
        const double slip5 = 5.0 * taker_;
        for (int64_t s = 0; s < S; s++) {
            if (pos[s] == 0.0) continue;
            const double exec = (pos[s] > 0.0) ? close[s] * (1.0 - slip5)
                                               : close[s] * (1.0 + slip5);
            cash        += pos[s] * (exec - close[s]);
            cash        -= std::fabs(pos[s]) * exec * taker_;
            cost_drag_t += std::fabs(pos[s]) * exec * taker_;
            pos[s]       = 0.0;
        }
        if (cash < 0.0) cash = 0.0;
        liq_at = t;
        return true;
    }

private:
    double impact_Y_, mmr_, taker_;
    bool iso_active_;
};


/* ============================ Equity(US 权益)============================
 * 佣金(+半点差)对称单边成本;空腿按日借券费(年化→日);无强平(leverage_cap≤1 + cash 底线
 * 已在引擎处理)。MTM/rebalance 喂**复权** OHLC(总收益口径)。**long-short**(空腿计借券费);
 * 唯一归零路径 = 引擎 cash 底线(cash≤0,与 policy 无关),非 margin-call 强平。 */
class EquityPolicy final : public MarketPolicy {
public:
    EquityPolicy(double cost_rate, double daily_borrow)
        : cost_rate_(cost_rate), daily_borrow_(daily_borrow) {}

    double trade_cost(double notional, double /*sigma*/, double /*bar_vol*/, bool /*is_sell*/) const override {
        return notional * cost_rate_;                   /* 对称佣金(美股,买卖同费),忽略 is_sell */
    }

    void carry(double* pos, const BarRow& bar, int64_t S, double& cash) const override {
        if (daily_borrow_ <= 0.0) return;
        for (int64_t s = 0; s < S; s++)
            if (pos[s] < 0.0) cash -= std::fabs(pos[s]) * bar.close[s] * daily_borrow_;  /* 空腿借券费 */
    }

    bool isolated_active() const override { return false; }
    void isolated_liq(double*, const BarRow&, int64_t, int64_t,
                      double&, double&, int64_t&) const override {}
    bool cross_margin_liq(double*, const double*, int64_t, int64_t,
                          double&, double&, int64_t&) const override { return false; }

private:
    double cost_rate_, daily_borrow_;
};


/* ============================ A-share(中国 A 股)============================
 * 佣金(buy+sell 对称)+ **印花税卖出单边**(asymmetric)+ 过户费(双边);**long-only**(无做空 →
 * 无 carry/借券)、无 margin-call 强平。daily cadence 天然满足 T+1(开盘买、最早次日开盘卖)。
 * 涨跌停 fill 限制属执行层(由权重端 tradeable-mask / valid 处理,不在 policy)。MTM/rebalance 喂 qfq 复权 OHLC。 */
class ASharePolicy final : public MarketPolicy {
public:
    ASharePolicy(double commission, double stamp_tax_sell, double transfer_fee)
        : buy_rate_(commission + transfer_fee),
          sell_rate_(commission + transfer_fee + stamp_tax_sell) {}

    double trade_cost(double notional, double /*sigma*/, double /*bar_vol*/, bool is_sell) const override {
        return notional * (is_sell ? sell_rate_ : buy_rate_);   /* 卖单含印花税 → 非对称 */
    }
    void carry(double*, const BarRow&, int64_t, double&) const override {}   /* long-only:无 funding/借券 */
    bool isolated_active() const override { return false; }
    void isolated_liq(double*, const BarRow&, int64_t, int64_t,
                      double&, double&, int64_t&) const override {}
    bool cross_margin_liq(double*, const double*, int64_t, int64_t,
                          double&, double&, int64_t&) const override { return false; }

private:
    double buy_rate_, sell_rate_;
};


/* ============================ 引擎 / 扫描入口 ============================ */

/* 组合回测:单 bar 直达 target、rebalance @ open。outputs caller 预分配(长度 T)。
 * 成本 / carry / 强平 全经 policy;leverage_cap / max_concentration 仅 raw_weights=0 时用。
 * trade_block:(T,S) int8 方向冻结掩码 | NULL(=全可交易)。bit0=禁增仓(涨停/停牌),
 *   bit1=禁减仓(跌停/停牌);受限方向的 rebalance 被冻结(target←pos,不成交不计费)——
 *   A 股涨跌停 / 停牌不可成交建模,市场无关(NULL 时 perp/equity 逐位复现)。 */
void run_portfolio(
    const double* alpha, const double* close, const double* open,
    const double* high, const double* low,
    const double* bar_volume, const double* sigma,
    const double* funding, const double* leverage,
    const int8_t* trade_block,
    int64_t T, int64_t S,
    double initial_cash, int64_t skip_warmup,
    double leverage_cap, double max_concentration, int raw_weights,
    double stop_trail_pct, const double* stop_ratchet, int64_t n_ratchet,
    const MarketPolicy& policy,
    double* equity, double* turnover, double* cost_drag, int64_t* liq_at);

/* 决策窗 stop 自动机(f64/f32 双路;逐位对齐 rl_bt_torch._stop_latch_scan)。 */
void stop_scan_f64(const double* w_TS, const double* close_TS, const int64_t* start_idx,
                   int64_t T_dec, int64_t S, int64_t bars_per_dec,
                   double trail, const double* ratchet, int64_t n_ratchet,
                   double* keep, int64_t* jstar, uint8_t* fired);
void stop_scan_f32(const float* w_TS, const float* close_TS, const int64_t* start_idx,
                   int64_t T_dec, int64_t S, int64_t bars_per_dec,
                   float trail, const float* ratchet, int64_t n_ratchet,
                   float* keep, int64_t* jstar, uint8_t* fired);

} // namespace bt

#endif /* BT_ENGINE_HPP */
