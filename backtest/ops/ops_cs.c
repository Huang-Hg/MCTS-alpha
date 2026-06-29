/* factor_mining/ops/ops_cs.c — Cross-sectional kernels
 *
 * fm_cs_rank / fm_cs_zscore / fm_cs_demean / fm_cs_finite_validstd / fm_cs_scale
 *
 * 每 t 一行截面运算,跨 t OMP 并行(每 t 独立)。
 * cs_rank:活跃集 compaction(per-bar carried ~220,非旧 top-30 的 ≤25)后走 3-way
 * quicksort(Dutch-flag 三分 → tie-heavy alpha 不退化;尾递归消除栈深 O(log n))。
 */

#include "ops.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif


static inline void _swap_vi(double* v, int64_t* idx, int64_t i, int64_t j) {
    double td = v[i]; v[i] = v[j]; v[j] = td;
    int64_t ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
}

/* 3-way quicksort 同步重排 (v, idx),按 v 升序。median-of-3 pivot + Dutch-flag 三分
 * (等值密集 alpha 不退化 O(n²));尾递归消除(递归小区、循环大区)→ 栈深 O(log n)。
 * 仅重排不改 FP 值 → cs_rank 输出与原插入排序逐位一致(平均-rank 对组内顺序无关)。 */
static void _qsort_vi(double* v, int64_t* idx, int64_t lo, int64_t hi) {
    while (hi - lo > 16) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (v[mid] < v[lo])  _swap_vi(v, idx, lo, mid);
        if (v[hi]  < v[lo])  _swap_vi(v, idx, lo, hi);
        if (v[hi]  < v[mid]) _swap_vi(v, idx, mid, hi);
        double piv = v[mid];
        int64_t lt = lo, gt = hi, i = lo;
        while (i <= gt) {
            if      (v[i] < piv) _swap_vi(v, idx, lt++, i++);
            else if (v[i] > piv) _swap_vi(v, idx, i, gt--);
            else                 i++;
        }
        if (lt - lo < hi - gt) { _qsort_vi(v, idx, lo, lt - 1); lo = gt + 1; }
        else                   { _qsort_vi(v, idx, gt + 1, hi); hi = lt - 1; }
    }
    for (int64_t i = lo + 1; i <= hi; i++) {   /* 小区插入排序 base case */
        double k = v[i]; int64_t ki = idx[i];
        int64_t j = i - 1;
        while (j >= lo && v[j] > k) { v[j+1] = v[j]; idx[j+1] = idx[j]; j--; }
        v[j+1] = k; idx[j+1] = ki;
    }
}

static void cs_rank_one_row(const double* row, double* out_row, int64_t S,
                             double* vals_buf, int64_t* idx_buf) {
    int64_t n_valid = 0;
    for (int64_t s = 0; s < S; s++) {
        double v = row[s];
        if (!isnan(v)) {
            vals_buf[n_valid] = v;
            idx_buf[n_valid]  = s;
            n_valid++;
        }
    }
    if (n_valid < 2) {
        for (int64_t s = 0; s < S; s++) out_row[s] = NAN;
        return;
    }
    _qsort_vi(vals_buf, idx_buf, 0, n_valid - 1);   /* 升序 */
    /* 同值组取平均 rank */
    int64_t i = 0;
    while (i < n_valid) {
        int64_t j = i;
        while (j + 1 < n_valid && vals_buf[j+1] == vals_buf[i]) j++;
        double avg_norm = ((double)(i + j) / 2.0) / (double)(n_valid - 1);
        for (int64_t k = i; k <= j; k++) out_row[idx_buf[k]] = avg_norm;
        i = j + 1;
    }
    /* NaN 位置回填 */
    for (int64_t s = 0; s < S; s++) {
        if (isnan(row[s])) out_row[s] = NAN;
    }
}

void fm_cs_rank(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel
    {
        double*  vals = (double*)malloc((size_t)S * sizeof(double));
        int64_t* idx  = (int64_t*)malloc((size_t)S * sizeof(int64_t));
        #pragma omp for schedule(static)
        for (int64_t t = 0; t < T; t++) {
            cs_rank_one_row(&x[t * S], &out[t * S], S, vals, idx);
        }
        free(vals); free(idx);
    }
}

void fm_cs_zscore(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double sum = 0.0, sum_sq = 0.0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            if (!isnan(v)) { sum += v; sum_sq += v * v; n++; }
        }
        if (n < 2) {
            for (int64_t s = 0; s < S; s++) out[t * S + s] = NAN;
            continue;
        }
        double mean = sum / (double)n;
        double var  = (sum_sq - mean * sum) / (double)(n - 1);
        if (var <= 0.0) {
            for (int64_t s = 0; s < S; s++) out[t * S + s] = NAN;
            continue;
        }
        double sd = sqrt(var);
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            out[t * S + s] = isnan(v) ? NAN : ((v - mean) / sd);
        }
    }
}


/* fm_cs_zscore_np — cs z-score,**NaN→0 填充 + ddof=0**(复刻旧 metrics.cs_zscore_np)。
 * 与 fm_cs_zscore 区别:后者 NaN 传播 + ddof=1 + 退化行 → NaN;本函数给 PCA 前置用,
 * 需无 NaN 输出。每行:finite 上算 mean/var(两 pass 中心化,匹配 numpy);
 * out[t,s] = (finite(x) && sd>1e-12) ? (x−mean)/sd : 0。退化行(n=0 或 sd≤1e-12)整行 0。 */
void fm_cs_zscore_np(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        const double* xr = x + t * S;
        double* outr = out + t * S;
        double sum = 0.0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double v = xr[s];
            if (isfinite(v)) { sum += v; n++; }
        }
        double mean = (n > 0) ? sum / (double)n : 0.0;
        double var = 0.0;
        if (n > 0) {
            double ss = 0.0;
            for (int64_t s = 0; s < S; s++) {
                double v = xr[s];
                if (isfinite(v)) { double d = v - mean; ss += d * d; }
            }
            var = ss / (double)n;
        }
        double sd = (var > 0.0) ? sqrt(var) : 0.0;
        for (int64_t s = 0; s < S; s++) {
            double v = xr[s];
            outr[s] = (isfinite(v) && sd > 1e-12) ? ((v - mean) / sd) : 0.0;
        }
    }
}

void fm_cs_demean(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double sum = 0.0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            if (!isnan(v)) { sum += v; n++; }
        }
        if (n == 0) {
            for (int64_t s = 0; s < S; s++) out[t * S + s] = NAN;
            continue;
        }
        double mean = sum / (double)n;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            out[t * S + s] = isnan(v) ? NAN : (v - mean);
        }
    }
}


/* finite_ratio + valid_ratio_cs 单 pass:evaluator gate1 用。
 * 替代 numpy 多 pass(isfinite + sum + sumsq + sqrt + valid mask),省 5×(T·S) 内存带宽。
 * biased var(分母 n,与 evaluator 原行为一致)。 */
void fm_cs_finite_validstd(const double* x, int64_t T, int64_t S, double thr_std,
                           double* finite_ratio_out, double* valid_ratio_cs_out) {
    int64_t n_has_var = 0;
    int64_t n_valid = 0;
    double thr_var = thr_std * thr_std;
    #pragma omp parallel for reduction(+:n_has_var,n_valid) schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double sum = 0.0, sumsq = 0.0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            /* isfinite — 同时挡 NaN + ±Inf。Inf 会让下游 ic 计算 num=Inf−Inf=NaN 污染 Lasso */
            if (isfinite(v)) { sum += v; sumsq += v * v; n++; }
        }
        if (n >= 2) {
            n_has_var++;
            double mean = sum / (double)n;
            double var  = sumsq / (double)n - mean * mean;
            if (var > thr_var) n_valid++;
        }
    }
    *finite_ratio_out   = (T > 0) ? (double)n_has_var / (double)T : 0.0;
    *valid_ratio_cs_out = (T > 0) ? (double)n_valid   / (double)T : 0.0;
}


/* cs_scale: 横截面 L1 归一化 = x[t,s] / sum_s |x[t,s]|。
 * WQ101 #28/#31/#32/#36/#60/#100 等 ~12 alpha 用,生产 dollar-neutral 信号。 */
void fm_cs_scale(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double sum_abs = 0.0;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            if (!isnan(v)) sum_abs += fabs(v);
        }
        if (sum_abs <= 1e-12) {
            for (int64_t s = 0; s < S; s++) out[t * S + s] = NAN;
            continue;
        }
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            out[t * S + s] = isnan(v) ? NAN : (v / sum_abs);
        }
    }
}
