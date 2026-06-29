/* factor_mining/ops/_ops_module.c
 *
 * Python C 扩展 → 暴露 ops.c 的全部算子为 _ops.<funcname>。
 *
 * 约定:
 *   - elementwise / ts / pair 接受 (T,S) float64 行步长视图(列连续即可,行步长任意正)
 *     → 零拷贝直通 C 内核(块化求值器切列块);其余形态(f32 / 列不连续)由
 *     PyArray_FROMANY 物化成 C 连续拷贝。cs / metrics 仍要求 C-连续全宽面板。
 *   - elementwise / ts / pair 末尾可选 out= 位置参数:可写 float64 (T,S) 列连续视图,
 *     内核直写(块化求值器把列块结果落进全宽数组,免回拷);返回 out 本身。
 *     缺省时返回新 C-连续数组。
 *   - 错误用 PyExc_TypeError / PyExc_ValueError 抛出,但内部假设输入合规(严禁防御性);只校验
 *     numpy 不能直接 reinterpret 的硬约束(dtype、ndim、shape 一致性、窗口正)。
 */

#include "npy_glue.h"   /* 共享 numpy 胶水 + module 样板(消除本地 as_panel / import_array 重复)*/
#include "ops.hpp"

/* 沿用旧调用名 → backtest/npy_glue.h 共享胶水(C-连续 / 零拷贝行步长视图 / out 视图 / 分配)。 */
#define as_panel     bt_as_2d
#define as_panel_ld  bt_as_panel_ld
#define as_out_ld    bt_as_out_ld
#define alloc_panel  bt_alloc_panel


/* ---------- Elementwise unary 通用模板 ---------- */

#define DEF_UNARY(py_name, c_fn)                                                   \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                    \
    PyObject* o_x; PyObject* o_out = NULL;                                          \
    if (!PyArg_ParseTuple(args, "O|O", &o_x, &o_out)) return NULL;                  \
    int64_t ldx, ldo;                                                                \
    PyArrayObject* x = as_panel_ld(o_x, &ldx);                                       \
    if (!x) return NULL;                                                             \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                           \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                        \
                               : alloc_panel(T, S, &ldo);                            \
    if (!out) { Py_DECREF(x); return NULL; }                                         \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                 \
         (int64_t)T, (int64_t)S, ldx, ldo);                                          \
    Py_DECREF(x);                                                                    \
    return (PyObject*)out;                                                           \
}

DEF_UNARY(abs_,    fm_abs)
DEF_UNARY(neg,     fm_neg)
DEF_UNARY(sign,    fm_sign)
DEF_UNARY(log_,    fm_log)
DEF_UNARY(sqrt_,   fm_sqrt_)
DEF_UNARY(square,  fm_square)
DEF_UNARY(tanh_,   fm_tanh)
DEF_UNARY(inv,     fm_inv)
DEF_UNARY(s_log_1p,fm_s_log_1p)


/* ---------- f32 → f64 块升格(operand 叶,必带 out)---------- */
static PyObject* py_upcast32(PyObject* self, PyObject* args) {
    PyObject *o_x, *o_out;
    if (!PyArg_ParseTuple(args, "OO", &o_x, &o_out)) return NULL;
    PyArrayObject* x = (PyArrayObject*)o_x;
    if (!PyArray_Check(o_x) || PyArray_TYPE(x) != NPY_FLOAT || PyArray_NDIM(x) != 2 ||
        PyArray_STRIDE(x, 1) != (npy_intp)sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "upcast32: x must be float32 (T,S) with contiguous columns");
        return NULL;
    }
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);
    int64_t ldx = (int64_t)(PyArray_STRIDE(x, 0) / (npy_intp)sizeof(float));
    int64_t ldo;
    PyArrayObject* out = as_out_ld(o_out, T, S, &ldo);
    if (!out) return NULL;
    fm_upcast32((const float*)PyArray_DATA(x), (double*)PyArray_DATA(out),
                (int64_t)T, (int64_t)S, ldx, ldo);
    return (PyObject*)out;
}


/* ---------- EvalCache dense-pack ---------- */
static PyObject* py_pack_sparse(PyObject* self, PyObject* args) {
    PyObject* o_x;
    if (!PyArg_ParseTuple(args, "O", &o_x)) return NULL;
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);
    npy_intp d_off[1] = { T + 1 };
    PyArrayObject* off = (PyArrayObject*)PyArray_EMPTY(1, d_off, NPY_INT64, 0);
    if (!off) { Py_DECREF(x); return NULL; }
    fm_pack_sparse_scan((const double*)PyArray_DATA(x), (int64_t)T, (int64_t)S,
                        (int64_t*)PyArray_DATA(off));
    int64_t n = ((int64_t*)PyArray_DATA(off))[T];
    npy_intp d_bm[2] = { T, (S + 63) / 64 };
    PyArrayObject* bm = (PyArrayObject*)PyArray_EMPTY(2, d_bm, NPY_UINT64, 0);
    npy_intp d_v[1] = { (npy_intp)n };
    PyArrayObject* vals = (PyArrayObject*)PyArray_EMPTY(1, d_v, NPY_DOUBLE, 0);
    if (!bm || !vals) { Py_XDECREF(bm); Py_XDECREF(vals); Py_DECREF(off); Py_DECREF(x); return NULL; }
    fm_pack_sparse_fill((const double*)PyArray_DATA(x), (int64_t)T, (int64_t)S,
                        (const int64_t*)PyArray_DATA(off),
                        (uint64_t*)PyArray_DATA(bm), (double*)PyArray_DATA(vals));
    Py_DECREF(x);
    return Py_BuildValue("(NNN)", bm, vals, off);    /* N: steal refs */
}

static PyObject* py_unpack_sparse(PyObject* self, PyObject* args) {
    PyObject *o_bm, *o_v, *o_off; long long S;
    if (!PyArg_ParseTuple(args, "OOOL", &o_bm, &o_v, &o_off, &S)) return NULL;
    PyArrayObject* bm = (PyArrayObject*)PyArray_FROMANY(
        o_bm, NPY_UINT64, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!bm) return NULL;
    PyArrayObject* v = (PyArrayObject*)PyArray_FROMANY(
        o_v, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!v) { Py_DECREF(bm); return NULL; }
    PyArrayObject* off = (PyArrayObject*)PyArray_FROMANY(
        o_off, NPY_INT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!off) { Py_DECREF(bm); Py_DECREF(v); return NULL; }
    npy_intp T = PyArray_DIM(bm, 0);
    if (PyArray_DIM(bm, 1) != (S + 63) / 64 || PyArray_DIM(off, 0) != T + 1) {
        PyErr_SetString(PyExc_ValueError, "unpack_sparse: bitmap/row_off shape mismatch");
        Py_DECREF(bm); Py_DECREF(v); Py_DECREF(off); return NULL;
    }
    int64_t ld_unused;
    PyArrayObject* out = alloc_panel(T, (npy_intp)S, &ld_unused);
    if (!out) { Py_DECREF(bm); Py_DECREF(v); Py_DECREF(off); return NULL; }
    fm_unpack_sparse((const uint64_t*)PyArray_DATA(bm), (const double*)PyArray_DATA(v),
                     (const int64_t*)PyArray_DATA(off), (int64_t)T, (int64_t)S,
                     (double*)PyArray_DATA(out));
    Py_DECREF(bm); Py_DECREF(v); Py_DECREF(off);
    return (PyObject*)out;
}


/* ---------- Elementwise binary ---------- */

#define DEF_BINARY(py_name, c_fn)                                                   \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject *o_a, *o_b; PyObject* o_out = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OO|O", &o_a, &o_b, &o_out)) return NULL;            \
    int64_t lda, ldb, ldo;                                                            \
    PyArrayObject* a = as_panel_ld(o_a, &lda); if (!a) return NULL;                   \
    PyArrayObject* b = as_panel_ld(o_b, &ldb); if (!b) { Py_DECREF(a); return NULL; } \
    if (PyArray_DIM(a, 0) != PyArray_DIM(b, 0) || PyArray_DIM(a, 1) != PyArray_DIM(b, 1)) { \
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");                   \
        Py_DECREF(a); Py_DECREF(b); return NULL;                                     \
    }                                                                                 \
    npy_intp T = PyArray_DIM(a, 0), S = PyArray_DIM(a, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(a); Py_DECREF(b); return NULL; }                            \
    c_fn((const double*)PyArray_DATA(a), (const double*)PyArray_DATA(b),              \
         (double*)PyArray_DATA(out), (int64_t)T, (int64_t)S, lda, ldb, ldo);          \
    Py_DECREF(a); Py_DECREF(b);                                                       \
    return (PyObject*)out;                                                            \
}

DEF_BINARY(add,   fm_add)
DEF_BINARY(sub,   fm_sub)
DEF_BINARY(mul,   fm_mul)
DEF_BINARY(div_,  fm_div)
DEF_BINARY(max_b, fm_max_b)
DEF_BINARY(min_b, fm_min_b)


/* ---------- Elementwise binary_const(panel ⊕ scalar k)---------- */

#define DEF_BIN_CONST(py_name, c_fn)                                                \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x; double k; PyObject* o_out = NULL;                                 \
    if (!PyArg_ParseTuple(args, "Od|O", &o_x, &k, &o_out)) return NULL;              \
    int64_t ldx, ldo;                                                                 \
    PyArrayObject* x = as_panel_ld(o_x, &ldx); if (!x) return NULL;                   \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                          \
    c_fn((const double*)PyArray_DATA(x), k, (double*)PyArray_DATA(out),               \
         (int64_t)T, (int64_t)S, ldx, ldo);                                           \
    Py_DECREF(x);                                                                     \
    return (PyObject*)out;                                                            \
}

DEF_BIN_CONST(add_const, fm_add_const)
DEF_BIN_CONST(mul_const, fm_mul_const)
DEF_BIN_CONST(pow_const, fm_pow_const)


/* ---------- Rolling unary ---------- */

/* 所有 ts unary 绑定统一接受可选尾参 (out, vlo, vhi)(Tier2b 预算 active-range,int64×S):
 * 非裁剪核(DEF_TS_UNARY)解析后忽略 vlo/vhi;裁剪核(DEF_TS_UNARY_CLIP)传给 c_fn
 * (NULL → 核内扫描 fallback)。统一签名让 executor 派发对所有 ts op 一致传 5 参。
 * vlo/vhi 须 int64 C-contiguous(executor 保证);直接借数据指针不管引用(调用期有效)。*/
#define _AS_I64_OR_NULL(o) \
    ((o) && (o) != Py_None ? (const int64_t*)PyArray_DATA((PyArrayObject*)(o)) : NULL)

#define DEF_TS_UNARY(py_name, c_fn)                                                 \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x; long long w; PyObject* o_out = NULL;                              \
    PyObject* o_lo = NULL; PyObject* o_hi = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OL|OOO", &o_x, &w, &o_out, &o_lo, &o_hi)) return NULL; \
    (void)o_lo; (void)o_hi;                          /* 非裁剪核忽略范围 */          \
    int64_t ldx, ldo;                                                                 \
    PyArrayObject* x = as_panel_ld(o_x, &ldx);                                        \
    if (!x) return NULL;                                                              \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                          \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                  \
         (int64_t)T, (int64_t)S, (int64_t)w, ldx, ldo);                               \
    Py_DECREF(x);                                                                     \
    return (PyObject*)out;                                                            \
}

#define DEF_TS_UNARY_CLIP(py_name, c_fn)                                            \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x; long long w; PyObject* o_out = NULL;                              \
    PyObject* o_lo = NULL; PyObject* o_hi = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OL|OOO", &o_x, &w, &o_out, &o_lo, &o_hi)) return NULL; \
    int64_t ldx, ldo;                                                                 \
    PyArrayObject* x = as_panel_ld(o_x, &ldx);                                        \
    if (!x) return NULL;                                                              \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                          \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                  \
         (int64_t)T, (int64_t)S, (int64_t)w, ldx, ldo,                                \
         _AS_I64_OR_NULL(o_lo), _AS_I64_OR_NULL(o_hi));                               \
    Py_DECREF(x);                                                                     \
    return (PyObject*)out;                                                            \
}

DEF_TS_UNARY(ts_mean,  fm_ts_mean)
DEF_TS_UNARY_CLIP(ts_std,   fm_ts_std)
DEF_TS_UNARY(ts_sum,   fm_ts_sum)
DEF_TS_UNARY_CLIP(ts_max,   fm_ts_max)
DEF_TS_UNARY_CLIP(ts_min,   fm_ts_min)
DEF_TS_UNARY(ts_ref,   fm_ts_ref)
DEF_TS_UNARY(ts_delta, fm_ts_delta)
DEF_TS_UNARY(ts_ema,   fm_ts_ema)
DEF_TS_UNARY(ts_wma,   fm_ts_wma)
DEF_TS_UNARY(ts_rank,    fm_ts_rank)
DEF_TS_UNARY(ts_arg_max, fm_ts_arg_max)
DEF_TS_UNARY(ts_arg_min, fm_ts_arg_min)
DEF_TS_UNARY_CLIP(ts_skew,    fm_ts_skew)
DEF_TS_UNARY_CLIP(ts_kurt,    fm_ts_kurt)
DEF_TS_UNARY_CLIP(ts_mad,     fm_ts_mad)
DEF_TS_UNARY(ts_slope,   fm_ts_slope)        /* 增量值随起点漂移 → 留扫描裁剪保 bit-identity */


/* ---------- Rolling pair（corr/cov 均裁剪,统一收 vlo/vhi）---------- */

#define DEF_TS_PAIR(py_name, c_fn)                                                  \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject *o_a, *o_b; long long w; PyObject* o_out = NULL;                        \
    PyObject* o_lo = NULL; PyObject* o_hi = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OOL|OOO", &o_a, &o_b, &w, &o_out, &o_lo, &o_hi)) return NULL; \
    int64_t lda, ldb, ldo;                                                            \
    PyArrayObject* a = as_panel_ld(o_a, &lda); if (!a) return NULL;                   \
    PyArrayObject* b = as_panel_ld(o_b, &ldb); if (!b) { Py_DECREF(a); return NULL; } \
    if (PyArray_DIM(a, 0) != PyArray_DIM(b, 0) || PyArray_DIM(a, 1) != PyArray_DIM(b, 1)) { \
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");                   \
        Py_DECREF(a); Py_DECREF(b); return NULL;                                     \
    }                                                                                 \
    npy_intp T = PyArray_DIM(a, 0), S = PyArray_DIM(a, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(a); Py_DECREF(b); return NULL; }                            \
    c_fn((const double*)PyArray_DATA(a), (const double*)PyArray_DATA(b),              \
         (double*)PyArray_DATA(out), (int64_t)T, (int64_t)S, (int64_t)w,              \
         lda, ldb, ldo, _AS_I64_OR_NULL(o_lo), _AS_I64_OR_NULL(o_hi));               \
    Py_DECREF(a); Py_DECREF(b);                                                       \
    return (PyObject*)out;                                                            \
}

DEF_TS_PAIR(ts_corr, fm_ts_corr)
DEF_TS_PAIR(ts_cov,  fm_ts_cov)


/* ---------- Cross-sectional ---------- */

#define DEF_CS(py_name, c_fn)                                                       \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x;                                                                  \
    if (!PyArg_ParseTuple(args, "O", &o_x)) return NULL;                            \
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;                          \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                          \
    int64_t ld_unused;                                                              \
    PyArrayObject* out = alloc_panel(T, S, &ld_unused);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                        \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                \
         (int64_t)T, (int64_t)S);                                                   \
    Py_DECREF(x);                                                                   \
    return (PyObject*)out;                                                          \
}

DEF_CS(cs_rank,   fm_cs_rank)
DEF_CS(cs_zscore, fm_cs_zscore)
DEF_CS(cs_zscore_np, fm_cs_zscore_np)
DEF_CS(cs_demean, fm_cs_demean)
DEF_CS(cs_scale,  fm_cs_scale)


/* ---------- cs_finite_validstd → (finite_ratio, valid_ratio_cs) tuple ---------- */
static PyObject* py_cs_finite_validstd(PyObject* self, PyObject* args) {
    PyObject* o_x; double thr_std;
    if (!PyArg_ParseTuple(args, "Od", &o_x, &thr_std)) return NULL;
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);
    double finite_ratio = 0.0, valid_ratio = 0.0;
    fm_cs_finite_validstd((const double*)PyArray_DATA(x), (int64_t)T, (int64_t)S,
                          thr_std, &finite_ratio, &valid_ratio);
    Py_DECREF(x);
    return Py_BuildValue("(dd)", finite_ratio, valid_ratio);
}


/* ---------- cs_holdable_coverage(x, y) → mean per-bar holdable coverage (double) ---------- */
static PyObject* py_cs_holdable_coverage(PyObject* self, PyObject* args) {
    PyObject *o_x, *o_y;
    if (!PyArg_ParseTuple(args, "OO", &o_x, &o_y)) return NULL;
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;
    PyArrayObject* y = as_panel(o_y); if (!y) { Py_DECREF(x); return NULL; }
    if (PyArray_DIM(x, 0) != PyArray_DIM(y, 0) || PyArray_DIM(x, 1) != PyArray_DIM(y, 1)) {
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");
        Py_DECREF(x); Py_DECREF(y); return NULL;
    }
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);
    double cov = fm_cs_holdable_coverage((const double*)PyArray_DATA(x),
                                         (const double*)PyArray_DATA(y), (int64_t)T, (int64_t)S);
    Py_DECREF(x); Py_DECREF(y);
    return PyFloat_FromDouble(cov);
}


/* ---------- Metrics ---------- */

#define DEF_METRIC(py_name, c_fn)                                                   \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject *o_x, *o_y;                                                            \
    if (!PyArg_ParseTuple(args, "OO", &o_x, &o_y)) return NULL;                     \
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;                          \
    PyArrayObject* y = as_panel(o_y); if (!y) { Py_DECREF(x); return NULL; }        \
    if (PyArray_DIM(x, 0) != PyArray_DIM(y, 0) || PyArray_DIM(x, 1) != PyArray_DIM(y, 1)) { \
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");                  \
        Py_DECREF(x); Py_DECREF(y); return NULL;                                    \
    }                                                                                \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                          \
    double r = c_fn((const double*)PyArray_DATA(x), (const double*)PyArray_DATA(y), \
                    (int64_t)T, (int64_t)S);                                        \
    Py_DECREF(x); Py_DECREF(y);                                                     \
    return PyFloat_FromDouble(r);                                                   \
}

DEF_METRIC(ic,       fm_ic)
DEF_METRIC(rank_ic,  fm_rank_ic)


/* ---------- icir(x, y) → (icir, mean_ic, std_ic) tuple ---------- */
static PyObject* py_icir(PyObject* self, PyObject* args) {
    PyObject *o_x, *o_y;
    if (!PyArg_ParseTuple(args, "OO", &o_x, &o_y)) return NULL;
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;
    PyArrayObject* y = as_panel(o_y); if (!y) { Py_DECREF(x); return NULL; }
    if (PyArray_DIM(x, 0) != PyArray_DIM(y, 0) || PyArray_DIM(x, 1) != PyArray_DIM(y, 1)) {
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");
        Py_DECREF(x); Py_DECREF(y); return NULL;
    }
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);
    double mean_ic = 0.0, std_ic = 0.0;
    double r = fm_icir((const double*)PyArray_DATA(x), (const double*)PyArray_DATA(y),
                       (int64_t)T, (int64_t)S, &mean_ic, &std_ic);
    Py_DECREF(x); Py_DECREF(y);
    return Py_BuildValue("(ddd)", r, mean_ic, std_ic);
}


/* per-t IC — 返回 (T,) np.ndarray of cross-sec IC at each t */
static PyObject* py_per_t_ic(PyObject* self, PyObject* args) {
    PyObject *o_x, *o_y;
    if (!PyArg_ParseTuple(args, "OO", &o_x, &o_y)) return NULL;
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;
    PyArrayObject* y = as_panel(o_y); if (!y) { Py_DECREF(x); return NULL; }
    if (PyArray_DIM(x, 0) != PyArray_DIM(y, 0) || PyArray_DIM(x, 1) != PyArray_DIM(y, 1)) {
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");
        Py_DECREF(x); Py_DECREF(y); return NULL;
    }
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);
    npy_intp shape_T[1] = { T };
    PyArrayObject* out = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    if (!out) { Py_DECREF(x); Py_DECREF(y); return NULL; }
    fm_per_t_ic((const double*)PyArray_DATA(x), (const double*)PyArray_DATA(y),
                (int64_t)T, (int64_t)S, (double*)PyArray_DATA(out));
    Py_DECREF(x); Py_DECREF(y);
    return (PyObject*)out;
}


/* per-t pseudo-PnL — 返回 (T,) np.ndarray of L1-norm signed PnL per t */
static PyObject* py_per_t_pnl(PyObject* self, PyObject* args) {
    PyObject *o_v, *o_y;
    if (!PyArg_ParseTuple(args, "OO", &o_v, &o_y)) return NULL;
    PyArrayObject* v = as_panel(o_v); if (!v) return NULL;
    PyArrayObject* y = as_panel(o_y); if (!y) { Py_DECREF(v); return NULL; }
    if (PyArray_DIM(v, 0) != PyArray_DIM(y, 0) || PyArray_DIM(v, 1) != PyArray_DIM(y, 1)) {
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");
        Py_DECREF(v); Py_DECREF(y); return NULL;
    }
    npy_intp T = PyArray_DIM(v, 0), S = PyArray_DIM(v, 1);
    npy_intp shape_T[1] = { T };
    PyArrayObject* out = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    if (!out) { Py_DECREF(v); Py_DECREF(y); return NULL; }
    fm_per_t_pnl((const double*)PyArray_DATA(v), (const double*)PyArray_DATA(y),
                 (int64_t)T, (int64_t)S, (double*)PyArray_DATA(out));
    Py_DECREF(v); Py_DECREF(y);
    return (PyObject*)out;
}





/* ---------- turnover(单 panel 标量)---------- */
static PyObject* py_turnover(PyObject* self, PyObject* args) {
    PyObject* o_x;
    if (!PyArg_ParseTuple(args, "O", &o_x)) return NULL;
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);
    double r = fm_turnover((const double*)PyArray_DATA(x), (int64_t)T, (int64_t)S);
    Py_DECREF(x);
    return PyFloat_FromDouble(r);
}



/* ---------- AlphaPool 边际 Δens 的 pnl 相关向量 ---------- */
/* pnl_corr_vec(cand (T,), members (n,T)) → (n,) Pearson(cand, member_j),NaN-aware,
 *   共同 finite<30 → 0.0。alpha_pool._corr_with_pool 的 C 内核(OMP 跨成员)。 */
static PyObject* py_pnl_corr_vec(PyObject* self, PyObject* args) {
    PyObject *o_cand, *o_mem;
    if (!PyArg_ParseTuple(args, "OO", &o_cand, &o_mem)) return NULL;
    PyArrayObject* cand = (PyArrayObject*)PyArray_FROMANY(
        o_cand, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!cand) return NULL;
    PyArrayObject* mem = (PyArrayObject*)PyArray_FROMANY(
        o_mem, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!mem) { Py_DECREF(cand); return NULL; }
    npy_intp T = PyArray_DIM(cand, 0);
    npy_intp n = PyArray_DIM(mem, 0);
    if (PyArray_DIM(mem, 1) != T) {
        PyErr_SetString(PyExc_ValueError, "members (n,T) must match cand (T,)");
        Py_DECREF(cand); Py_DECREF(mem); return NULL;
    }
    npy_intp shape_n[1] = { n };
    PyArrayObject* out = (PyArrayObject*)PyArray_EMPTY(1, shape_n, NPY_DOUBLE, 0);
    if (!out) { Py_DECREF(cand); Py_DECREF(mem); return NULL; }
    fm_pnl_corr_vec((const double*)PyArray_DATA(cand), (const double*)PyArray_DATA(mem),
                    (double*)PyArray_DATA(out), (int64_t)n, (int64_t)T);
    Py_DECREF(cand); Py_DECREF(mem);
    return (PyObject*)out;
}


/* ---------- 杂项 ---------- */

static PyObject* py_omp_max_threads(PyObject* self, PyObject* args) {
    return PyLong_FromLong(fm_omp_max_threads());
}

static PyObject* py_omp_set_num_threads(PyObject* self, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;
    fm_omp_set_num_threads(n);
    Py_RETURN_NONE;
}

static PyObject* py_malloc_trim(PyObject* self, PyObject* args) {
    return PyLong_FromLong(fm_malloc_trim());
}


/* ---------- Method table ---------- */

static PyMethodDef methods[] = {
    {"abs_",   py_abs_,   METH_VARARGS, "|x|"},
    {"neg",    py_neg,    METH_VARARGS, "-x"},
    {"sign",   py_sign,   METH_VARARGS, "sign(x)"},
    {"log_",   py_log_,   METH_VARARGS, "log(x)"},
    {"sqrt_",  py_sqrt_,  METH_VARARGS, "sqrt(x)"},
    {"square", py_square, METH_VARARGS, "x**2"},
    {"tanh_",  py_tanh_,  METH_VARARGS, "tanh(x)"},
    {"inv",      py_inv,      METH_VARARGS, "1/x (NaN if x==0)"},
    {"s_log_1p", py_s_log_1p, METH_VARARGS, "sign(x)·log(1+|x|)"},
    {"add",    py_add,    METH_VARARGS, "a+b"},
    {"sub",    py_sub,    METH_VARARGS, "a-b"},
    {"mul",    py_mul,    METH_VARARGS, "a*b"},
    {"div_",   py_div_,   METH_VARARGS, "a/b (NaN if b==0)"},
    {"max_b",  py_max_b,  METH_VARARGS, "max(a,b) elementwise"},
    {"min_b",  py_min_b,  METH_VARARGS, "min(a,b) elementwise"},

    {"upcast32",  py_upcast32,  METH_VARARGS, "upcast32(x_f32, out_f64): OMP f32→f64 block cast"},
    {"pack_sparse",   py_pack_sparse,   METH_VARARGS,
     "pack_sparse(x) -> (bitmap u64 (T,ceil(S/64)), values f64 (n,), row_off i64 (T+1,)); lossless non-NaN pack"},
    {"unpack_sparse", py_unpack_sparse, METH_VARARGS,
     "unpack_sparse(bitmap, values, row_off, S) -> (T,S) f64 with NaN holes"},

    {"add_const", py_add_const, METH_VARARGS, "x + k (1 SIMD pass, NaN propagates)"},
    {"mul_const", py_mul_const, METH_VARARGS, "x * k (1 SIMD pass, NaN propagates)"},
    {"pow_const", py_pow_const, METH_VARARGS, "sign(x)·|x|^k; 0→0; NaN propagates"},

    {"ts_mean",  py_ts_mean,  METH_VARARGS, "rolling mean (T,S,w)"},
    {"ts_std",   py_ts_std,   METH_VARARGS, "rolling std"},
    {"ts_sum",   py_ts_sum,   METH_VARARGS, "rolling sum"},
    {"ts_max",   py_ts_max,   METH_VARARGS, "rolling max"},
    {"ts_min",   py_ts_min,   METH_VARARGS, "rolling min"},
    {"ts_ref",   py_ts_ref,   METH_VARARGS, "x.shift(w)"},
    {"ts_delta", py_ts_delta, METH_VARARGS, "x - x.shift(w)"},
    {"ts_ema",   py_ts_ema,   METH_VARARGS, "ewm(span=w, adjust=False, min_periods=w).mean"},
    {"ts_wma",   py_ts_wma,   METH_VARARGS, "linear-weighted MA"},
    {"ts_rank",    py_ts_rank,    METH_VARARGS, "rolling avg-rank in [0,1]"},
    {"ts_arg_max", py_ts_arg_max, METH_VARARGS, "bars-since rolling-max"},
    {"ts_arg_min", py_ts_arg_min, METH_VARARGS, "bars-since rolling-min"},
    {"ts_skew",    py_ts_skew,    METH_VARARGS, "rolling 3rd standardized moment"},
    {"ts_kurt",    py_ts_kurt,    METH_VARARGS, "rolling excess kurtosis (m4/m2² − 3)"},
    {"ts_mad",     py_ts_mad,     METH_VARARGS, "rolling mean absolute deviation"},
    {"ts_slope",   py_ts_slope,   METH_VARARGS, "rolling OLS slope vs t"},

    {"ts_corr",  py_ts_corr,  METH_VARARGS, "rolling corr (T,S,w)"},
    {"ts_cov",   py_ts_cov,   METH_VARARGS, "rolling cov"},

    {"cs_rank",   py_cs_rank,   METH_VARARGS, "cross-sectional avg-rank in [0,1]"},
    {"cs_zscore", py_cs_zscore, METH_VARARGS, "cross-sectional z-score"},
    {"cs_zscore_np", py_cs_zscore_np, METH_VARARGS, "cs z-score, NaN→0 fill, ddof=0 (PCA prefill)"},
    {"cs_demean", py_cs_demean, METH_VARARGS, "cross-sectional row-demean"},
    {"cs_scale",  py_cs_scale,  METH_VARARGS, "cross-sectional L1-normalize x/Σ|x|"},
    {"cs_finite_validstd", py_cs_finite_validstd, METH_VARARGS,
     "(finite_ratio, valid_ratio_cs) single-pass for evaluator gate1"},
    {"cs_holdable_coverage", py_cs_holdable_coverage, METH_VARARGS,
     "mean per-bar holdable coverage #(x&y finite)/#(y finite) for evaluator gate"},

    {"ic",        py_ic,        METH_VARARGS, "mean cross-sectional Pearson IC"},
    {"rank_ic",   py_rank_ic,   METH_VARARGS, "RankIC = IC after cs_rank"},
    {"icir",      py_icir,      METH_VARARGS, "(icir, mean_ic, std_ic) = ICIR over per-t IC, ddof=0"},
    {"turnover",  py_turnover,  METH_VARARGS, "mean(|Δx|) / mean(per-col std)"},
    {"per_t_ic",  py_per_t_ic, METH_VARARGS,
     "per_t_ic(panel, y) → (T,) np.ndarray of cross-sec Pearson IC per t"},
    {"per_t_pnl", py_per_t_pnl, METH_VARARGS,
     "per_t_pnl(values, y) → (T,) np.ndarray of L1-norm signed PnL per t"},
    {"pnl_corr_vec", py_pnl_corr_vec, METH_VARARGS,
     "pnl_corr_vec(cand (T,), members (n,T)) -> (n,) NaN-aware Pearson corr of cand vs each member"},

    {"omp_max_threads", py_omp_max_threads, METH_NOARGS, "max OMP threads (0 if no OMP)"},
    {"omp_set_num_threads", py_omp_set_num_threads, METH_VARARGS,
     "set OMP ICV thread count (worker raises threads after fork)"},
    {"malloc_trim", py_malloc_trim, METH_NOARGS,
     "glibc malloc_trim(0): return brk-heap holes to OS (no-op elsewhere)"},
    {NULL, NULL, 0, NULL}
};


BT_MODULE(_ops, methods, "Formulaic alpha C ops kernel.")
