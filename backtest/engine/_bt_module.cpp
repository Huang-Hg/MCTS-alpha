/* backtest/engine/_bt_module.cpp —— backtest._bt Python 扩展 binding(C++)。
 *
 * 导出:
 *   portfolio_bt(...)         crypto 永续(PerpPolicy;签名/行为与旧内核不变 → 回归门 bit-identical)
 *   portfolio_bt_equity(...)  US 权益(EquityPolicy;佣金+借券费,无 funding/强平)
 *   stop_latch_scan(...)      决策窗 stop 自动机(f32/f64)
 * numpy glue / module 样板复用 common/npy_glue.h;引擎核心 engine/portfolio。
 */
#include "npy_glue.h"

#include <cmath>

#include "engine.hpp"


/* equity/turnover/cost (T,) + 标量汇总 → 结果 dict。消费(steal)三个 (T,) 数组的引用。 */
static PyObject* pack_result(PyArrayObject* equity_arr, PyArrayObject* turnover_arr,
                             PyArrayObject* cost_arr, int64_t liq_at,
                             double init_cash, double bars_per_year, npy_intp T) {
    const double* eq = (const double*)PyArray_DATA(equity_arr);
    const double* tu = (const double*)PyArray_DATA(turnover_arr);
    double total_return = (init_cash > 0.0) ? (eq[T - 1] / init_cash - 1.0) : 0.0;

    double sum_lr = 0.0, sum_lr_sq = 0.0;
    int64_t n_r = 0;
    for (npy_intp i = 1; i < T; i++) {
        if (eq[i - 1] <= 0.0 || eq[i] <= 0.0) break;
        double lr = std::log(eq[i] / eq[i - 1]);
        sum_lr += lr; sum_lr_sq += lr * lr; n_r++;
    }
    double sharpe = 0.0;
    if (n_r > 1) {
        double mean = sum_lr / (double)n_r;
        double var  = (sum_lr_sq - mean * sum_lr) / (double)(n_r - 1);
        if (var > 0.0) sharpe = mean / std::sqrt(var) * std::sqrt(bars_per_year);
    }
    double mean_turnover = 0.0;
    if (T > 1) {
        double s = 0.0;
        for (npy_intp i = 1; i < T; i++) s += tu[i];
        mean_turnover = s / (double)(T - 1);
    }

    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "equity",    (PyObject*)equity_arr);
    PyDict_SetItemString(d, "turnover",  (PyObject*)turnover_arr);
    PyDict_SetItemString(d, "cost_drag", (PyObject*)cost_arr);
    PyDict_SetItemString(d, "liq_at",        PyLong_FromLongLong((long long)liq_at));
    PyDict_SetItemString(d, "total_return",  PyFloat_FromDouble(total_return));
    PyDict_SetItemString(d, "sharpe",        PyFloat_FromDouble(sharpe));
    PyDict_SetItemString(d, "mean_turnover", PyFloat_FromDouble(mean_turnover));
    Py_DECREF(equity_arr); Py_DECREF(turnover_arr); Py_DECREF(cost_arr);
    return d;
}

#define _DAT(a)  ((a) ? (const double*)PyArray_DATA(a) : nullptr)


/* ============================ portfolio_bt(crypto / PerpPolicy)============================ */
static PyObject* py_portfolio_bt(PyObject* self, PyObject* args) {
    PyObject *o_alpha, *o_close, *o_open;
    PyObject *o_high = Py_None, *o_low = Py_None;
    PyObject *o_bvol = Py_None, *o_sigma = Py_None, *o_funding = Py_None;
    PyObject *o_leverage = Py_None, *o_ratchet = Py_None;
    double init_cash, half_spread, fee, impact_Y, mmr;
    long long skip_warmup_ll;
    double leverage_cap = 1.0, max_concentration = 0.0;
    int raw_weights = 0;
    double stop_trail_pct = 0.0;
    double bars_per_year = 365.0 * 24.0 * 12.0;

    if (!PyArg_ParseTuple(args, "OOOOOOOOdLdddd|ddiOdOd",
                          &o_alpha, &o_close, &o_open, &o_high, &o_low,
                          &o_bvol, &o_sigma, &o_funding,
                          &init_cash, &skip_warmup_ll,
                          &half_spread, &fee, &impact_Y, &mmr,
                          &leverage_cap, &max_concentration, &raw_weights,
                          &o_leverage, &stop_trail_pct, &o_ratchet, &bars_per_year)) {
        return NULL;
    }

    PyArrayObject* alpha_arr = bt_as_2d(o_alpha);
    if (!alpha_arr) return NULL;
    PyArrayObject* close_arr = bt_as_2d(o_close);
    if (!close_arr) { Py_DECREF(alpha_arr); return NULL; }
    PyArrayObject* open_arr = bt_as_2d(o_open);
    if (!open_arr) { Py_DECREF(alpha_arr); Py_DECREF(close_arr); return NULL; }

    npy_intp T = PyArray_DIM(alpha_arr, 0), S = PyArray_DIM(alpha_arr, 1);
    if (PyArray_DIM(close_arr, 0) != T || PyArray_DIM(close_arr, 1) != S ||
        PyArray_DIM(open_arr, 0) != T || PyArray_DIM(open_arr, 1) != S) {
        PyErr_SetString(PyExc_ValueError, "close/open shape must match alpha (T,S)");
        Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
    }

    PyArrayObject* high_arr  = bt_as_2d_or_none(o_high);
    PyArrayObject* low_arr   = bt_as_2d_or_none(o_low);
    PyArrayObject* bvol_arr  = bt_as_2d_or_none(o_bvol);
    PyArrayObject* sigma_arr = bt_as_2d_or_none(o_sigma);
    PyArrayObject* fund_arr  = bt_as_2d_or_none(o_funding);
    PyArrayObject* lev_arr   = bt_as_2d_or_none(o_leverage);
    PyArrayObject* ratchet_arr = NULL;
    if (o_ratchet != Py_None) {
        ratchet_arr = bt_as_1d_f64(o_ratchet);
        if (!ratchet_arr || (PyArray_DIM(ratchet_arr, 0) % 2) != 0) {
            PyErr_SetString(PyExc_ValueError, "stop_ratchet must be flat (2k,) [gain, trail] pairs");
            Py_XDECREF(ratchet_arr);
            Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
        }
    }

    const bool iso_active = (lev_arr != NULL && high_arr != NULL && low_arr != NULL);
    bt::PerpPolicy policy(half_spread, fee, impact_Y, mmr, iso_active);

    npy_intp shape_T[1] = { T };
    PyArrayObject* equity_arr   = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* turnover_arr = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* cost_arr     = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    int64_t liq_at = -1;

    Py_BEGIN_ALLOW_THREADS
    bt::run_portfolio(
        _DAT(alpha_arr), _DAT(close_arr), _DAT(open_arr),
        _DAT(high_arr), _DAT(low_arr), _DAT(bvol_arr), _DAT(sigma_arr),
        _DAT(fund_arr), _DAT(lev_arr),
        nullptr,                                /* crypto 永续无涨跌停 */
        (int64_t)T, (int64_t)S, init_cash, (int64_t)skip_warmup_ll,
        leverage_cap, max_concentration, raw_weights,
        stop_trail_pct, _DAT(ratchet_arr),
        ratchet_arr ? (int64_t)(PyArray_DIM(ratchet_arr, 0) / 2) : 0,
        policy,
        (double*)PyArray_DATA(equity_arr), (double*)PyArray_DATA(turnover_arr),
        (double*)PyArray_DATA(cost_arr), &liq_at);
    Py_END_ALLOW_THREADS

    Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr);
    Py_XDECREF(high_arr); Py_XDECREF(low_arr); Py_XDECREF(bvol_arr);
    Py_XDECREF(sigma_arr); Py_XDECREF(fund_arr); Py_XDECREF(lev_arr); Py_XDECREF(ratchet_arr);
    return pack_result(equity_arr, turnover_arr, cost_arr, liq_at, init_cash, bars_per_year, T);
}


/* ============================ portfolio_bt_equity(US 权益 / EquityPolicy)============================
 * portfolio_bt_equity(alpha, close, open, init_cash, skip_warmup, cost_rate, daily_borrow,
 *                     [raw_weights=0, stop_trail_pct=0.0, stop_ratchet=None, bars_per_year=252.0]) -> dict
 * close/open = 复权 OHLC(总收益);无 funding/margin-call 强平/impact。long-short(空腿计借券费)。
 * 权重外部构造(raw_weights=1 直传)→ leverage_cap/max_concentration 无意义,不暴露。 */
static PyObject* py_portfolio_bt_equity(PyObject* self, PyObject* args) {
    PyObject *o_alpha, *o_close, *o_open, *o_ratchet = Py_None;
    double init_cash, cost_rate, daily_borrow;
    long long skip_warmup_ll;
    int raw_weights = 0;
    double stop_trail_pct = 0.0, bars_per_year = 252.0;

    if (!PyArg_ParseTuple(args, "OOOdLdd|idOd",
                          &o_alpha, &o_close, &o_open,
                          &init_cash, &skip_warmup_ll, &cost_rate, &daily_borrow,
                          &raw_weights, &stop_trail_pct, &o_ratchet, &bars_per_year)) {
        return NULL;
    }

    PyArrayObject* alpha_arr = bt_as_2d(o_alpha);
    if (!alpha_arr) return NULL;
    PyArrayObject* close_arr = bt_as_2d(o_close);
    if (!close_arr) { Py_DECREF(alpha_arr); return NULL; }
    PyArrayObject* open_arr = bt_as_2d(o_open);
    if (!open_arr) { Py_DECREF(alpha_arr); Py_DECREF(close_arr); return NULL; }

    npy_intp T = PyArray_DIM(alpha_arr, 0), S = PyArray_DIM(alpha_arr, 1);
    if (PyArray_DIM(close_arr, 0) != T || PyArray_DIM(close_arr, 1) != S ||
        PyArray_DIM(open_arr, 0) != T || PyArray_DIM(open_arr, 1) != S) {
        PyErr_SetString(PyExc_ValueError, "close/open shape must match alpha (T,S)");
        Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
    }
    PyArrayObject* ratchet_arr = NULL;
    if (o_ratchet != Py_None) {
        ratchet_arr = bt_as_1d_f64(o_ratchet);
        if (!ratchet_arr || (PyArray_DIM(ratchet_arr, 0) % 2) != 0) {
            PyErr_SetString(PyExc_ValueError, "stop_ratchet must be flat (2k,) [gain, trail] pairs");
            Py_XDECREF(ratchet_arr);
            Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
        }
    }

    bt::EquityPolicy policy(cost_rate, daily_borrow);

    npy_intp shape_T[1] = { T };
    PyArrayObject* equity_arr   = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* turnover_arr = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* cost_arr     = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    int64_t liq_at = -1;

    Py_BEGIN_ALLOW_THREADS
    bt::run_portfolio(
        _DAT(alpha_arr), _DAT(close_arr), _DAT(open_arr),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr,                                /* 美股研究口径不建模 halt 冻结 */
        (int64_t)T, (int64_t)S, init_cash, (int64_t)skip_warmup_ll,
        1.0, 0.0, raw_weights,                  /* lev/conc raw_weights 旁路不触及 */
        stop_trail_pct, _DAT(ratchet_arr),
        ratchet_arr ? (int64_t)(PyArray_DIM(ratchet_arr, 0) / 2) : 0,
        policy,
        (double*)PyArray_DATA(equity_arr), (double*)PyArray_DATA(turnover_arr),
        (double*)PyArray_DATA(cost_arr), &liq_at);
    Py_END_ALLOW_THREADS

    Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); Py_XDECREF(ratchet_arr);
    return pack_result(equity_arr, turnover_arr, cost_arr, liq_at, init_cash, bars_per_year, T);
}


/* ============================ stop_latch_scan ============================ */
static PyObject* py_stop_latch_scan(PyObject* self, PyObject* args) {
    PyObject *o_w, *o_close, *o_start, *o_ratchet = Py_None;
    long long bars_ll;
    double trail;
    if (!PyArg_ParseTuple(args, "OOOLd|O", &o_w, &o_close, &o_start, &bars_ll, &trail, &o_ratchet))
        return NULL;

    int both_f32 = 0;
    {
        PyArrayObject* w_probe = (PyArrayObject*)PyArray_FROM_OF(o_w, NPY_ARRAY_ALIGNED);
        PyArrayObject* c_probe = (PyArrayObject*)PyArray_FROM_OF(o_close, NPY_ARRAY_ALIGNED);
        if (!w_probe || !c_probe) { Py_XDECREF(w_probe); Py_XDECREF(c_probe); return NULL; }
        both_f32 = (PyArray_TYPE(w_probe) == NPY_FLOAT32 && PyArray_TYPE(c_probe) == NPY_FLOAT32);
        Py_DECREF(w_probe); Py_DECREF(c_probe);
    }
    const int real_t = both_f32 ? NPY_FLOAT32 : NPY_FLOAT64;

    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROMANY(
        o_w, real_t, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!w_arr) return NULL;
    PyArrayObject* close_arr = (PyArrayObject*)PyArray_FROMANY(
        o_close, real_t, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!close_arr) { Py_DECREF(w_arr); return NULL; }
    PyArrayObject* start_arr = bt_as_1d_i64(o_start);
    if (!start_arr) { Py_DECREF(w_arr); Py_DECREF(close_arr); return NULL; }

    PyArrayObject* ratchet_arr = NULL;
    if (o_ratchet != Py_None) {
        ratchet_arr = (PyArrayObject*)PyArray_FROMANY(
            o_ratchet, real_t, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
        if (!ratchet_arr || (PyArray_DIM(ratchet_arr, 0) % 2) != 0) {
            PyErr_SetString(PyExc_ValueError, "ratchet must be flat (2k,) [gain, trail] pairs");
            Py_XDECREF(ratchet_arr);
            Py_DECREF(w_arr); Py_DECREF(close_arr); Py_DECREF(start_arr); return NULL;
        }
    }

    const npy_intp T_dec = PyArray_DIM(w_arr, 0), S = PyArray_DIM(w_arr, 1);
    const npy_intp T5 = PyArray_DIM(close_arr, 0);
    const int64_t* start = (const int64_t*)PyArray_DATA(start_arr);
    int shape_ok = (PyArray_DIM(close_arr, 1) == S) && (PyArray_DIM(start_arr, 0) == T_dec);
    for (npy_intp t = 0; shape_ok && t < T_dec; t++)
        if (start[t] < 0 || start[t] + bars_ll >= T5) shape_ok = 0;
    if (!shape_ok) {
        PyErr_SetString(PyExc_ValueError, "stop_latch_scan: shape/bounds mismatch");
        Py_DECREF(w_arr); Py_DECREF(close_arr); Py_DECREF(start_arr); Py_XDECREF(ratchet_arr);
        return NULL;
    }

    npy_intp shape2[2] = { T_dec, S };
    PyArrayObject* keep_arr  = (PyArrayObject*)PyArray_EMPTY(2, shape2, real_t, 0);
    PyArrayObject* jstar_arr = (PyArrayObject*)PyArray_EMPTY(2, shape2, NPY_INT64, 0);
    PyArrayObject* fired_arr = (PyArrayObject*)PyArray_EMPTY(2, shape2, NPY_BOOL, 0);
    const int64_t n_ratchet = ratchet_arr ? (int64_t)(PyArray_DIM(ratchet_arr, 0) / 2) : 0;

    Py_BEGIN_ALLOW_THREADS
    if (both_f32) {
        bt::stop_scan_f32(
            (const float*)PyArray_DATA(w_arr), (const float*)PyArray_DATA(close_arr),
            start, (int64_t)T_dec, (int64_t)S, (int64_t)bars_ll, (float)trail,
            ratchet_arr ? (const float*)PyArray_DATA(ratchet_arr) : nullptr, n_ratchet,
            (float*)PyArray_DATA(keep_arr), (int64_t*)PyArray_DATA(jstar_arr),
            (uint8_t*)PyArray_DATA(fired_arr));
    } else {
        bt::stop_scan_f64(
            (const double*)PyArray_DATA(w_arr), (const double*)PyArray_DATA(close_arr),
            start, (int64_t)T_dec, (int64_t)S, (int64_t)bars_ll, trail,
            ratchet_arr ? (const double*)PyArray_DATA(ratchet_arr) : nullptr, n_ratchet,
            (double*)PyArray_DATA(keep_arr), (int64_t*)PyArray_DATA(jstar_arr),
            (uint8_t*)PyArray_DATA(fired_arr));
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(w_arr); Py_DECREF(close_arr); Py_DECREF(start_arr); Py_XDECREF(ratchet_arr);
    PyObject* out = PyTuple_Pack(3, (PyObject*)keep_arr, (PyObject*)jstar_arr, (PyObject*)fired_arr);
    Py_DECREF(keep_arr); Py_DECREF(jstar_arr); Py_DECREF(fired_arr);
    return out;
}


/* ============================ portfolio_bt_ashare(中国 A 股 / ASharePolicy)============================
 * portfolio_bt_ashare(alpha, close, open, init_cash, skip_warmup, commission, stamp_tax_sell, transfer_fee,
 *                     [raw_weights=0, trade_block=None, bars_per_year=242.0]) -> dict
 * close/open = qfq 复权 OHLC(总收益);佣金双边 + 印花税卖出单边 + 过户费双边;long-only,无 funding/强平。
 * trade_block:(T,S) int8 涨跌停/停牌方向冻结掩码 | None(bit0=涨停禁买、bit1=跌停禁卖、3=停牌双禁)。
 * 权重外部构造(raw_weights=1 直传)→ leverage_cap/max_concentration 无意义,不暴露。 */
static PyObject* py_portfolio_bt_ashare(PyObject* self, PyObject* args) {
    PyObject *o_alpha, *o_close, *o_open, *o_tblock = Py_None;
    double init_cash, commission, stamp_tax_sell, transfer_fee;
    long long skip_warmup_ll;
    double bars_per_year = 242.0;
    int raw_weights = 0;

    if (!PyArg_ParseTuple(args, "OOOdLddd|iOd",
                          &o_alpha, &o_close, &o_open,
                          &init_cash, &skip_warmup_ll, &commission, &stamp_tax_sell, &transfer_fee,
                          &raw_weights, &o_tblock, &bars_per_year)) {
        return NULL;
    }

    PyArrayObject* alpha_arr = bt_as_2d(o_alpha);
    if (!alpha_arr) return NULL;
    PyArrayObject* close_arr = bt_as_2d(o_close);
    if (!close_arr) { Py_DECREF(alpha_arr); return NULL; }
    PyArrayObject* open_arr = bt_as_2d(o_open);
    if (!open_arr) { Py_DECREF(alpha_arr); Py_DECREF(close_arr); return NULL; }

    npy_intp T = PyArray_DIM(alpha_arr, 0), S = PyArray_DIM(alpha_arr, 1);
    if (PyArray_DIM(close_arr, 0) != T || PyArray_DIM(close_arr, 1) != S ||
        PyArray_DIM(open_arr, 0) != T || PyArray_DIM(open_arr, 1) != S) {
        PyErr_SetString(PyExc_ValueError, "close/open shape must match alpha (T,S)");
        Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
    }

    PyArrayObject* tblock_arr = NULL;
    if (o_tblock != Py_None) {
        tblock_arr = (PyArrayObject*)PyArray_FROMANY(
            o_tblock, NPY_INT8, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
        if (!tblock_arr ||
            PyArray_DIM(tblock_arr, 0) != T || PyArray_DIM(tblock_arr, 1) != S) {
            PyErr_SetString(PyExc_ValueError, "trade_block must be int8 (T,S) matching alpha");
            Py_XDECREF(tblock_arr);
            Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
        }
    }

    bt::ASharePolicy policy(commission, stamp_tax_sell, transfer_fee);

    npy_intp shape_T[1] = { T };
    PyArrayObject* equity_arr   = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* turnover_arr = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* cost_arr     = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    int64_t liq_at = -1;

    Py_BEGIN_ALLOW_THREADS
    bt::run_portfolio(
        _DAT(alpha_arr), _DAT(close_arr), _DAT(open_arr),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        tblock_arr ? (const int8_t*)PyArray_DATA(tblock_arr) : nullptr,
        (int64_t)T, (int64_t)S, init_cash, (int64_t)skip_warmup_ll,
        1.0, 0.0, raw_weights,                  /* lev/conc raw_weights 旁路不触及 */
        0.0, nullptr, 0,
        policy,
        (double*)PyArray_DATA(equity_arr), (double*)PyArray_DATA(turnover_arr),
        (double*)PyArray_DATA(cost_arr), &liq_at);
    Py_END_ALLOW_THREADS

    Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); Py_XDECREF(tblock_arr);
    return pack_result(equity_arr, turnover_arr, cost_arr, liq_at, init_cash, bars_per_year, T);
}


static PyMethodDef methods[] = {
    {"portfolio_bt", py_portfolio_bt, METH_VARARGS,
     "portfolio_bt(alpha, close, open, high|None, low|None, bar_volume|None, sigma|None, funding|None, "
     "init_cash, skip_warmup, half_spread, fee, impact_Y, mmr, "
     "[leverage_cap=1.0, max_concentration=0.0, raw_weights=0, leverage_TS=None, stop_trail_pct=0.0, "
     "stop_ratchet=None, bars_per_year=105120.0]) -> dict  (crypto perp / PerpPolicy)"},
    {"portfolio_bt_equity", py_portfolio_bt_equity, METH_VARARGS,
     "portfolio_bt_equity(alpha, close, open, init_cash, skip_warmup, cost_rate, daily_borrow, "
     "[raw_weights=0, stop_trail_pct=0.0, stop_ratchet=None, bars_per_year=252.0]) -> dict  (US equity / EquityPolicy)"},
    {"portfolio_bt_ashare", py_portfolio_bt_ashare, METH_VARARGS,
     "portfolio_bt_ashare(alpha, close, open, init_cash, skip_warmup, commission, stamp_tax_sell, transfer_fee, "
     "[raw_weights=0, trade_block=None, bars_per_year=242.0]) -> dict  (China A-share / ASharePolicy)"},
    {"stop_latch_scan", py_stop_latch_scan, METH_VARARGS,
     "stop_latch_scan(w_dec, close_5m, start_idx, bars_per_dec, trail, ratchet|None) -> (keep, jstar, fired)"},
    {NULL, NULL, 0, NULL}
};

BT_MODULE(_bt, methods, "Backtest C++ engine (policy-abstracted: PerpPolicy / EquityPolicy / ASharePolicy).")
