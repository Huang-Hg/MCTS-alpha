/* backtest/_bt_module.c
 *
 * Python C 扩展 → 暴露 backtest API。
 *
 * 导出:
 *   portfolio_bt(alpha, close, open, high|None, low|None,
 *                bar_volume|None, sigma|None, funding|None,
 *                init_cash, skip_warmup, half_spread, fee, impact_Y, mmr, [..., bars_per_year])
 *       → dict(equity, turnover, cost_drag, liq_at,
 *              total_return, sharpe, mean_turnover)
 *   stop_latch_scan(w_dec, close_5m, start_idx, bars_per_dec, trail, ratchet|None)
 *       → (keep, jstar, fired)   bt_lite 决策窗 stop 自动机(f32/f64 双路)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "portfolio_bt.h"

#include <math.h>
#include <string.h>


static inline PyArrayObject* as_2d(PyObject* obj) {
    return (PyArrayObject*)PyArray_FROMANY(
        obj, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

static inline PyArrayObject* as_2d_or_none(PyObject* obj) {
    if (obj == Py_None) return NULL;
    return as_2d(obj);
}


static PyObject* py_portfolio_bt(PyObject* self, PyObject* args) {
    PyObject *o_alpha, *o_close, *o_open;
    PyObject *o_high = Py_None, *o_low = Py_None;
    PyObject *o_bvol = Py_None, *o_sigma = Py_None, *o_funding = Py_None;
    PyObject *o_leverage = Py_None;   /* (T,S) per-sym isolated 杠杆;None → cross-margin */
    PyObject *o_ratchet  = Py_None;   /* (2k,) flattened [gain, trail] 升序;None → 无 ratchet */
    double init_cash, half_spread, fee, impact_Y, mmr;
    long long skip_warmup_ll;
    double leverage_cap = 1.0;     /* default 1× gross */
    double max_concentration = 0.0; /* default 禁用 */
    int raw_weights = 0;            /* default 0;1 = 跳过 step 1,RL 路径用 */
    double stop_trail_pct = 0.0;    /* default 关闭 */
    /* Sharpe 年化 bar 数(= 权益曲线采样 cadence 的年 bar 数)。
     * default 365×24×12=105120 = 24/7 5m(逐位复现旧行为);US 日线注入 252、5m 注入 252×78。
     * 由 markets.TradingCalendar.bars_per_year 提供。 */
    double bars_per_year = 365.0 * 24.0 * 12.0;

    /* O×8 d L d×4 | d d i O d O d
     *  alpha close open high low bvol sigma fund | cash skip | hs fee Y mmr |
     *  leverage_cap(opt) max_concentration(opt)
     *  raw_weights(opt) leverage_TS(opt) stop_trail_pct(opt) stop_ratchet(opt) bars_per_year(opt) */
    if (!PyArg_ParseTuple(args, "OOOOOOOOdLdddd|ddiOdOd",
                          &o_alpha, &o_close, &o_open,
                          &o_high, &o_low,
                          &o_bvol, &o_sigma, &o_funding,
                          &init_cash, &skip_warmup_ll,
                          &half_spread, &fee, &impact_Y, &mmr,
                          &leverage_cap, &max_concentration, &raw_weights,
                          &o_leverage, &stop_trail_pct, &o_ratchet,
                          &bars_per_year)) {
        return NULL;
    }

    PyArrayObject* alpha_arr = as_2d(o_alpha);
    if (!alpha_arr) return NULL;
    PyArrayObject* close_arr = as_2d(o_close);
    if (!close_arr) { Py_DECREF(alpha_arr); return NULL; }
    PyArrayObject* open_arr = as_2d(o_open);
    if (!open_arr)  { Py_DECREF(alpha_arr); Py_DECREF(close_arr); return NULL; }

    npy_intp T = PyArray_DIM(alpha_arr, 0);
    npy_intp S = PyArray_DIM(alpha_arr, 1);

    if (PyArray_DIM(close_arr, 0) != T || PyArray_DIM(close_arr, 1) != S ||
        PyArray_DIM(open_arr,  0) != T || PyArray_DIM(open_arr,  1) != S) {
        PyErr_SetString(PyExc_ValueError, "close/open shape must match alpha (T,S)");
        Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr);
        return NULL;
    }

    PyArrayObject* high_arr  = as_2d_or_none(o_high);
    PyArrayObject* low_arr   = as_2d_or_none(o_low);
    PyArrayObject* bvol_arr  = as_2d_or_none(o_bvol);
    PyArrayObject* sigma_arr = as_2d_or_none(o_sigma);
    PyArrayObject* fund_arr  = as_2d_or_none(o_funding);
    PyArrayObject* lev_arr   = as_2d_or_none(o_leverage);
    PyArrayObject* ratchet_arr = NULL;
    if (o_ratchet != Py_None) {
        ratchet_arr = (PyArrayObject*)PyArray_FROMANY(
            o_ratchet, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (!ratchet_arr || (PyArray_DIM(ratchet_arr, 0) % 2) != 0) {
            PyErr_SetString(PyExc_ValueError, "stop_ratchet must be flat (2k,) [gain, trail] pairs");
            Py_XDECREF(ratchet_arr);
            Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr);
            return NULL;
        }
    }

    #define _CLEANUP_OPTIONALS \
        do { \
            if (high_arr)  Py_DECREF(high_arr); \
            if (low_arr)   Py_DECREF(low_arr); \
            if (bvol_arr)  Py_DECREF(bvol_arr); \
            if (sigma_arr) Py_DECREF(sigma_arr); \
            if (fund_arr)  Py_DECREF(fund_arr); \
            if (lev_arr)   Py_DECREF(lev_arr); \
            if (ratchet_arr) Py_DECREF(ratchet_arr); \
        } while (0)

    PyArrayObject* shape_checks[] = { high_arr, low_arr, bvol_arr, sigma_arr, fund_arr, lev_arr };
    for (size_t i = 0; i < sizeof(shape_checks)/sizeof(shape_checks[0]); i++) {
        PyArrayObject* a = shape_checks[i];
        if (a && (PyArray_DIM(a, 0) != T || PyArray_DIM(a, 1) != S)) {
            PyErr_SetString(PyExc_ValueError, "optional (T,S) arrays must match alpha shape");
            Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr);
            _CLEANUP_OPTIONALS;
            return NULL;
        }
    }

    npy_intp shape_T[1] = { T };
    PyArrayObject* equity_arr   = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* turnover_arr = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* cost_arr     = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);

    int64_t liq_at = -1;

    Py_BEGIN_ALLOW_THREADS
    bt_portfolio(
        (const double*)PyArray_DATA(alpha_arr),
        (const double*)PyArray_DATA(close_arr),
        (const double*)PyArray_DATA(open_arr),
        high_arr  ? (const double*)PyArray_DATA(high_arr)  : NULL,
        low_arr   ? (const double*)PyArray_DATA(low_arr)   : NULL,
        bvol_arr  ? (const double*)PyArray_DATA(bvol_arr)  : NULL,
        sigma_arr ? (const double*)PyArray_DATA(sigma_arr) : NULL,
        fund_arr  ? (const double*)PyArray_DATA(fund_arr)  : NULL,
        (int64_t)T, (int64_t)S,
        init_cash, (int64_t)skip_warmup_ll,
        half_spread, fee, impact_Y,
        mmr,
        leverage_cap,
        max_concentration,
        raw_weights,
        lev_arr   ? (const double*)PyArray_DATA(lev_arr)   : NULL,
        stop_trail_pct,
        ratchet_arr ? (const double*)PyArray_DATA(ratchet_arr) : NULL,
        ratchet_arr ? (int64_t)(PyArray_DIM(ratchet_arr, 0) / 2) : 0,
        (double*)PyArray_DATA(equity_arr),
        (double*)PyArray_DATA(turnover_arr),
        (double*)PyArray_DATA(cost_arr),
        &liq_at
    );
    Py_END_ALLOW_THREADS

    Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr);
    _CLEANUP_OPTIONALS;
    #undef _CLEANUP_OPTIONALS

    /* 汇总 */
    const double* eq = (const double*)PyArray_DATA(equity_arr);
    const double* tu = (const double*)PyArray_DATA(turnover_arr);
    double total_return = (init_cash > 0.0) ? (eq[T - 1] / init_cash - 1.0) : 0.0;

    double sum_lr = 0.0, sum_lr_sq = 0.0;
    int64_t n_r = 0;
    for (npy_intp i = 1; i < T; i++) {
        if (eq[i - 1] <= 0.0 || eq[i] <= 0.0) break;
        double lr = log(eq[i] / eq[i - 1]);
        sum_lr    += lr;
        sum_lr_sq += lr * lr;
        n_r++;
    }
    double sharpe = 0.0;
    if (n_r > 1) {
        double mean = sum_lr / (double)n_r;
        double var  = (sum_lr_sq - mean * sum_lr) / (double)(n_r - 1);
        if (var > 0.0) {
            sharpe = mean / sqrt(var) * sqrt(bars_per_year);
        }
    }

    double mean_turnover = 0.0;
    if (T > 1) {
        double s = 0.0;
        for (npy_intp i = 1; i < T; i++) s += tu[i];
        mean_turnover = s / (double)(T - 1);
    }

    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "equity",   (PyObject*)equity_arr);
    PyDict_SetItemString(d, "turnover", (PyObject*)turnover_arr);
    PyDict_SetItemString(d, "cost_drag",(PyObject*)cost_arr);
    PyDict_SetItemString(d, "liq_at",          PyLong_FromLongLong((long long)liq_at));
    PyDict_SetItemString(d, "total_return",    PyFloat_FromDouble(total_return));
    PyDict_SetItemString(d, "sharpe",          PyFloat_FromDouble(sharpe));
    PyDict_SetItemString(d, "mean_turnover",   PyFloat_FromDouble(mean_turnover));
    Py_DECREF(equity_arr); Py_DECREF(turnover_arr); Py_DECREF(cost_arr);
    return d;
}


/* stop_latch_scan(w_dec, close_5m, start_idx, bars_per_dec, trail, ratchet|None)
 *   → (keep, jstar, fired)
 * f32/f64 双路:w 与 close 均 float32 → f32 内核(逐位对齐 torch float32);否则统一 f64。
 * 形状:w (T_dec,S);close (T5,S);start_idx (T_dec,) int64,要求 max+bars_per_dec < T5。 */
static PyObject* py_stop_latch_scan(PyObject* self, PyObject* args) {
    PyObject *o_w, *o_close, *o_start, *o_ratchet = Py_None;
    long long bars_ll;
    double trail;
    if (!PyArg_ParseTuple(args, "OOOLd|O", &o_w, &o_close, &o_start,
                          &bars_ll, &trail, &o_ratchet)) {
        return NULL;
    }

    /* dtype 决策:两者皆 f32 → f32 路;否则 f64 路 */
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
    PyArrayObject* start_arr = (PyArrayObject*)PyArray_FROMANY(
        o_start, NPY_INT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!start_arr) { Py_DECREF(w_arr); Py_DECREF(close_arr); return NULL; }

    PyArrayObject* ratchet_arr = NULL;
    if (o_ratchet != Py_None) {
        /* f32 路:f64 ratchet 下转 f32 需 FORCECAST(与 torch 标量→tensor-dtype cast 同语义) */
        ratchet_arr = (PyArrayObject*)PyArray_FROMANY(
            o_ratchet, real_t, 1, 1,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
        if (!ratchet_arr || (PyArray_DIM(ratchet_arr, 0) % 2) != 0) {
            PyErr_SetString(PyExc_ValueError, "ratchet must be flat (2k,) [gain, trail] pairs");
            Py_XDECREF(ratchet_arr);
            Py_DECREF(w_arr); Py_DECREF(close_arr); Py_DECREF(start_arr);
            return NULL;
        }
    }

    const npy_intp T_dec = PyArray_DIM(w_arr, 0);
    const npy_intp S     = PyArray_DIM(w_arr, 1);
    const npy_intp T5    = PyArray_DIM(close_arr, 0);
    const int64_t* start = (const int64_t*)PyArray_DATA(start_arr);
    int shape_ok = (PyArray_DIM(close_arr, 1) == S) && (PyArray_DIM(start_arr, 0) == T_dec);
    for (npy_intp t = 0; shape_ok && t < T_dec; t++) {
        if (start[t] < 0 || start[t] + bars_ll >= T5) shape_ok = 0;
    }
    if (!shape_ok) {
        PyErr_SetString(PyExc_ValueError,
                        "stop_latch_scan: shape/bounds mismatch (need close (T5,S), start (T_dec,), "
                        "start[t]+bars_per_dec < T5)");
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
        bt_stop_scan_f32(
            (const float*)PyArray_DATA(w_arr), (const float*)PyArray_DATA(close_arr),
            start, (int64_t)T_dec, (int64_t)S, (int64_t)bars_ll,
            (float)trail,
            ratchet_arr ? (const float*)PyArray_DATA(ratchet_arr) : NULL, n_ratchet,
            (float*)PyArray_DATA(keep_arr), (int64_t*)PyArray_DATA(jstar_arr),
            (uint8_t*)PyArray_DATA(fired_arr));
    } else {
        bt_stop_scan_f64(
            (const double*)PyArray_DATA(w_arr), (const double*)PyArray_DATA(close_arr),
            start, (int64_t)T_dec, (int64_t)S, (int64_t)bars_ll,
            trail,
            ratchet_arr ? (const double*)PyArray_DATA(ratchet_arr) : NULL, n_ratchet,
            (double*)PyArray_DATA(keep_arr), (int64_t*)PyArray_DATA(jstar_arr),
            (uint8_t*)PyArray_DATA(fired_arr));
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(w_arr); Py_DECREF(close_arr); Py_DECREF(start_arr); Py_XDECREF(ratchet_arr);
    PyObject* out = PyTuple_Pack(3, (PyObject*)keep_arr, (PyObject*)jstar_arr, (PyObject*)fired_arr);
    Py_DECREF(keep_arr); Py_DECREF(jstar_arr); Py_DECREF(fired_arr);
    return out;
}


static PyMethodDef methods[] = {
    {"portfolio_bt", py_portfolio_bt, METH_VARARGS,
     "portfolio_bt(alpha, close, open, high|None, low|None, "
     "bar_volume|None, sigma|None, funding|None, "
     "init_cash, skip_warmup, half_spread, fee, impact_Y, mmr, "
     "[leverage_cap=1.0, max_concentration=0.0, raw_weights=0, leverage_TS=None, stop_trail_pct=0.0, stop_ratchet=None, bars_per_year=105120.0]) -> dict"},
    {"stop_latch_scan", py_stop_latch_scan, METH_VARARGS,
     "stop_latch_scan(w_dec, close_5m, start_idx, bars_per_dec, trail, ratchet|None) -> (keep, jstar, fired)"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_bt",
    "Backtest C kernel.",
    -1, methods, NULL, NULL, NULL, NULL,
};


PyMODINIT_FUNC PyInit__bt(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
