/* backtest/common/npy_glue.h
 *
 * 共享 numpy↔C 胶水层 + Python 模块样板 —— 被各 C/C++ 扩展 binding 复用(消除 _bt_module /
 * _ops_module 各写一份 as_2d / import_array / PyModuleDef 的重复)。
 *
 * 仅在**做 import_array 的 binding 文件**里 include(numpy PyArray_API 是 per-TU 指针表;
 * 每个扩展模块恰有一个 numpy TU = 其 binding 文件)。纯算子/引擎 .c/.cpp(只碰 double*)
 * 不 include 本头。所有 helper 为 static inline(内部链接,无 ODR 冲突)。
 *
 * 约定(严禁防御性编程):内核假设输入合规,只校验 numpy 不能直接 reinterpret 的硬约束
 * (dtype / ndim / shape / 步长 / 可写)—— 这些是 segfault 边界。
 */
#ifndef BT_COMMON_NPY_GLUE_H
#define BT_COMMON_NPY_GLUE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>


/* ---------- (T,S) C-连续 float64(cs / metrics / 价量面板用)---------- */
static inline PyArrayObject* bt_as_2d(PyObject* obj) {
    return (PyArrayObject*)PyArray_FROMANY(
        obj, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

static inline PyArrayObject* bt_as_2d_or_none(PyObject* obj) {
    if (obj == Py_None) return NULL;
    return bt_as_2d(obj);
}

/* ---------- (k,) 1D C-连续 float64 / int64(ratchet / start_idx 用)---------- */
static inline PyArrayObject* bt_as_1d_f64(PyObject* obj) {
    return (PyArrayObject*)PyArray_FROMANY(
        obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

static inline PyArrayObject* bt_as_1d_i64(PyObject* obj) {
    return (PyArrayObject*)PyArray_FROMANY(
        obj, NPY_INT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

/* ---------- (T,S) float64 行步长视图(零拷贝)----------
 * f64 + 2D + 列步长==8B + 行步长 ≥ S·8B(正、行不重叠)→ 借引用,*ld=行步长(元素数);
 * 其余形态 FROMANY 物化 C-连续拷贝(ld=S)。块化求值器切列块直通。 */
static inline PyArrayObject* bt_as_panel_ld(PyObject* obj, int64_t* ld_out) {
    if (PyArray_Check(obj)) {
        PyArrayObject* a = (PyArrayObject*)obj;
        if (PyArray_TYPE(a) == NPY_DOUBLE && PyArray_NDIM(a) == 2 &&
            PyArray_STRIDE(a, 1) == (npy_intp)sizeof(double) &&
            PyArray_STRIDE(a, 0) >= (npy_intp)(PyArray_DIM(a, 1) * sizeof(double)) &&
            PyArray_ISALIGNED(a)) {
            Py_INCREF(a);
            *ld_out = (int64_t)(PyArray_STRIDE(a, 0) / (npy_intp)sizeof(double));
            return a;
        }
    }
    PyArrayObject* arr = bt_as_2d(obj);
    if (arr) *ld_out = (int64_t)PyArray_DIM(arr, 1);
    return arr;
}

/* out= 目标:可写 float64 (T,S) 列连续视图。shape/dtype/步长是 segfault 边界,必须校验。 */
static inline PyArrayObject* bt_as_out_ld(PyObject* obj, npy_intp T, npy_intp S, int64_t* ld_out) {
    PyArrayObject* a = (PyArrayObject*)obj;
    if (!PyArray_Check(obj) || PyArray_TYPE(a) != NPY_DOUBLE || PyArray_NDIM(a) != 2 ||
        PyArray_DIM(a, 0) != T || PyArray_DIM(a, 1) != S ||
        PyArray_STRIDE(a, 1) != (npy_intp)sizeof(double) ||
        PyArray_STRIDE(a, 0) < (npy_intp)(S * (npy_intp)sizeof(double)) ||
        !PyArray_ISWRITEABLE(a) || !PyArray_ISALIGNED(a)) {
        PyErr_SetString(PyExc_ValueError,
                        "out= must be a writable float64 (T,S) view with contiguous columns");
        return NULL;
    }
    Py_INCREF(a);
    *ld_out = (int64_t)(PyArray_STRIDE(a, 0) / (npy_intp)sizeof(double));
    return a;
}

static inline PyArrayObject* bt_alloc_panel(npy_intp T, npy_intp S, int64_t* ld_out) {
    npy_intp dims[2] = { T, S };
    *ld_out = (int64_t)S;
    return (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
}

static inline PyArrayObject* bt_alloc_1d(npy_intp n, int typenum) {
    npy_intp dims[1] = { n };
    return (PyArrayObject*)PyArray_EMPTY(1, dims, typenum, 0);
}


/* ---------- 模块样板:PyModuleDef + PyInit(含 import_array)----------
 * 用法(binding 文件末尾):  BT_MODULE(_bt, methods, "Backtest C/C++ kernel.")
 * C++ binding 自动加 extern "C" 链接(PyInit 必须 C ABI)。 */
#ifdef __cplusplus
#define BT_MODULE_EXTERN_C extern "C"
#else
#define BT_MODULE_EXTERN_C
#endif

#define BT_MODULE(modname, methods_arr, docstr)                                    \
    static struct PyModuleDef _bt_moduledef_##modname = {                          \
        PyModuleDef_HEAD_INIT, #modname, docstr, -1, methods_arr,                  \
        NULL, NULL, NULL, NULL };                                                  \
    BT_MODULE_EXTERN_C PyMODINIT_FUNC PyInit_##modname(void) {                     \
        import_array();                                                            \
        return PyModule_Create(&_bt_moduledef_##modname);                          \
    }

#endif /* BT_COMMON_NPY_GLUE_H */
