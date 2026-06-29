/* factor_mining/ops/ops_metrics.c — 因子评估指标
 *
 * fm_ic / fm_per_t_ic / fm_per_t_pnl / fm_rank_ic / fm_turnover
 * fm_omp_max_threads
 *
 * 设计原则:全部跨 t 或跨 (m,k) OMP 并行,reduction(+) 合 sum_corr/cnt;
 * isfinite 同时挡 NaN + ±Inf(Inf 会让 num=Inf−Inf=NaN 污染 reduction)。
 */

#include "ops.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif


double fm_ic(const double* x, const double* y, int64_t T, int64_t S) {
    double total = 0.0;
    int64_t n_valid_t = 0;
    #pragma omp parallel for schedule(static) reduction(+:total) reduction(+:n_valid_t)
    for (int64_t t = 0; t < T; t++) {
        double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double vx = x[t * S + s], vy = y[t * S + s];
            if (isfinite(vx) && isfinite(vy)) {
                sx += vx; sy += vy; sxy += vx * vy; sxx += vx * vx; syy += vy * vy; n++;
            }
        }
        if (n < 3) continue;
        double mx = sx / n, my = sy / n;
        double num = sxy - mx * sy;
        double dx  = sxx - mx * sx;
        double dy  = syy - my * sy;
        if (dx > 0.0 && dy > 0.0) {
            total += num / sqrt(dx * dy);
            n_valid_t++;
        }
    }
    return (n_valid_t > 0) ? (total / (double)n_valid_t) : 0.0;
}


/* per-t pseudo-PnL — L1-normalized signed PnL,模拟 L1=1 deploy 的 per-bar 收益。
 *   v[s] = values[t,s] − cs_mean(values[t,:]),  w[s] = v[s] / Σ|v|
 *   out[t] = Σ_s w[s] · y[t,s]   (任一为 NaN 的 (s) 跳过 / 全 NaN 行 / Σ|v|≈0 → 0)
 * AlphaPool D5 行为级 diversity 度量,Pearson(out_a, out_b) 即 pnl_corr。
 * 2 passes/row:第一 pass 算 mean_v,第二 pass fused (sum|v| + Σ v·y)。 */
void fm_per_t_pnl(const double* values, const double* y, int64_t T, int64_t S, double* out) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        const double* vrow = values + t * S;
        const double* yrow = y      + t * S;
        /* pass 1: cross-sec mean of finite values */
        double sum_v = 0.0;
        int64_t n_v = 0;
        for (int64_t s = 0; s < S; s++) {
            double v = vrow[s];
            if (isfinite(v)) { sum_v += v; n_v++; }
        }
        if (n_v == 0) { out[t] = 0.0; continue; }
        double mean_v = sum_v / (double)n_v;
        /* pass 2: fused Σ|v−mean| + Σ (v−mean)·y(只对 v 和 y 都 finite 的 s 累 pnl 分子) */
        double sum_abs = 0.0, pnl_num = 0.0;
        for (int64_t s = 0; s < S; s++) {
            double v = vrow[s];
            if (!isfinite(v)) continue;
            double dv = v - mean_v;
            sum_abs += fabs(dv);
            double yv = yrow[s];
            if (isfinite(yv)) pnl_num += dv * yv;
        }
        out[t] = (sum_abs < 1e-12) ? 0.0 : (pnl_num / sum_abs);
    }
}


/* per-t IC — 跟 fm_ic 同算法,每 t 写入 out[t](不平均)。
 * n<3 / std<=0 → out[t] = NaN(下游 nan-aware 取 mean/cov 时跳过)。 */
void fm_per_t_ic(const double* x, const double* y, int64_t T, int64_t S, double* out) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double vx = x[t * S + s], vy = y[t * S + s];
            if (isfinite(vx) && isfinite(vy)) {
                sx += vx; sy += vy; sxy += vx * vy; sxx += vx * vx; syy += vy * vy; n++;
            }
        }
        if (n < 3) { out[t] = NAN; continue; }
        double mx = sx / n, my = sy / n;
        double num = sxy - mx * sy;
        double dx  = sxx - mx * sx;
        double dy  = syy - my * sy;
        if (dx > 0.0 && dy > 0.0) {
            out[t] = num / sqrt(dx * dy);
        } else {
            out[t] = NAN;
        }
    }
}


/* holdable 覆盖率门:mean_t[ #(x finite ∧ y finite) / #(y finite) ]。
 * y 已 NaN-mask 到 holdable(listed∧member)→ 量"信号覆盖了多少可交易截面"。
 * 挡 sqrt(cs_z−2) 类近全 NaN 信号:每 bar 只覆盖极少 holdable → per_t_ic 在 1-3 点上算出
 * 伪高 |IC|。返回 [0,1];无 holdable 的退化盘返 0(必被拒)。*/
double fm_cs_holdable_coverage(const double* x, const double* y, int64_t T, int64_t S) {
    double acc = 0.0;
    int64_t nb = 0;
    #pragma omp parallel for reduction(+:acc,nb) schedule(static)
    for (int64_t t = 0; t < T; t++) {
        int64_t h = 0, c = 0;
        for (int64_t s = 0; s < S; s++) {
            if (isfinite(y[t * S + s])) { h++; if (isfinite(x[t * S + s])) c++; }
        }
        if (h > 0) { acc += (double)c / (double)h; nb++; }
    }
    return (nb > 0) ? acc / (double)nb : 0.0;
}


/* fused per-t cs_rank(行内插入排序 + 平均-rank) + Pearson reduction。
 * 每 thread 仅 4×S double 暂存,**省去 2×T·S = ~60MB 临时面板**(T=245k,S=16)。
 * 算法等价于 fm_cs_rank(x→rx) + fm_cs_rank(y→ry) + fm_ic(rx,ry),bit-exact。 */
static inline void _rank_row(
    const double* row, double* out_row, int64_t S,
    double* vals_buf, int64_t* idx_buf
) {
    int64_t n_valid = 0;
    for (int64_t s = 0; s < S; s++) {
        double v = row[s];
        if (!isnan(v)) { vals_buf[n_valid] = v; idx_buf[n_valid] = s; n_valid++; }
    }
    if (n_valid < 2) {
        for (int64_t s = 0; s < S; s++) out_row[s] = NAN;
        return;
    }
    /* 升序插入排序(S≤25,N² 比 qsort 快)。 */
    for (int64_t i = 1; i < n_valid; i++) {
        double k = vals_buf[i]; int64_t ki = idx_buf[i];
        int64_t j = i - 1;
        while (j >= 0 && vals_buf[j] > k) { vals_buf[j+1] = vals_buf[j]; idx_buf[j+1] = idx_buf[j]; j--; }
        vals_buf[j+1] = k; idx_buf[j+1] = ki;
    }
    /* 同值组取平均 rank,归一化到 [0,1]。 */
    int64_t i = 0;
    while (i < n_valid) {
        int64_t j = i;
        while (j + 1 < n_valid && vals_buf[j+1] == vals_buf[i]) j++;
        double avg_norm = ((double)(i + j) / 2.0) / (double)(n_valid - 1);
        for (int64_t k = i; k <= j; k++) out_row[idx_buf[k]] = avg_norm;
        i = j + 1;
    }
    for (int64_t s = 0; s < S; s++) {
        if (isnan(row[s])) out_row[s] = NAN;
    }
}

double fm_rank_ic(const double* x, const double* y, int64_t T, int64_t S) {
    double total = 0.0;
    int64_t n_valid_t = 0;
    #pragma omp parallel reduction(+:total) reduction(+:n_valid_t)
    {
        double*  rx   = (double*)malloc((size_t)S * sizeof(double));
        double*  ry   = (double*)malloc((size_t)S * sizeof(double));
        double*  vals = (double*)malloc((size_t)S * sizeof(double));
        int64_t* idx  = (int64_t*)malloc((size_t)S * sizeof(int64_t));
        #pragma omp for schedule(static)
        for (int64_t t = 0; t < T; t++) {
            _rank_row(&x[t * S], rx, S, vals, idx);
            _rank_row(&y[t * S], ry, S, vals, idx);
            double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
            int64_t n = 0;
            for (int64_t s = 0; s < S; s++) {
                double vx = rx[s], vy = ry[s];
                if (isfinite(vx) && isfinite(vy)) {
                    sx += vx; sy += vy; sxy += vx * vy; sxx += vx * vx; syy += vy * vy; n++;
                }
            }
            if (n >= 3) {
                double mx = sx / n, my = sy / n;
                double num = sxy - mx * sy;
                double dx  = sxx - mx * sx;
                double dy  = syy - my * sy;
                if (dx > 0.0 && dy > 0.0) {
                    total += num / sqrt(dx * dy);
                    n_valid_t++;
                }
            }
        }
        free(rx); free(ry); free(vals); free(idx);
    }
    return (n_valid_t > 0) ? (total / (double)n_valid_t) : 0.0;
}


/* fm_icir — 单因子 ICIR = nanmean(per_t_ic) / nanstd(per_t_ic, ddof=0)。
 * 复刻旧 metrics.icir:per_t_ic 公式同 fm_per_t_ic(n<3 或 dx/dy≤0 → 该 t 无效跳过);
 * 对有效 t 累 Σic、Σic²、cnt → mean = Σic/cnt,var = Σic²/cnt − mean²,std = sqrt(max(var,0))。
 * non-finite mean 或 std<1e-9 → icir=0(此时 mean 非有限回 0);cnt==0 → (0,0,0)。
 * 返回 icir;mean_ic / std_ic 经 out 指针回传(binding 组 3-tuple)。 */
double fm_icir(const double* x, const double* y, int64_t T, int64_t S,
               double* out_mean, double* out_std) {
    double sum_ic = 0.0, sum_ic2 = 0.0;
    int64_t cnt = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum_ic) reduction(+:sum_ic2) reduction(+:cnt)
    for (int64_t t = 0; t < T; t++) {
        double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double vx = x[t * S + s], vy = y[t * S + s];
            if (isfinite(vx) && isfinite(vy)) {
                sx += vx; sy += vy; sxy += vx * vy; sxx += vx * vx; syy += vy * vy; n++;
            }
        }
        if (n < 3) continue;
        double mx = sx / n, my = sy / n;
        double num = sxy - mx * sy;
        double dx  = sxx - mx * sx;
        double dy  = syy - my * sy;
        if (dx > 0.0 && dy > 0.0) {
            double ic = num / sqrt(dx * dy);
            sum_ic += ic; sum_ic2 += ic * ic; cnt++;
        }
    }
    if (cnt == 0) { *out_mean = 0.0; *out_std = 0.0; return 0.0; }
    double mean_ic = sum_ic / (double)cnt;
    double var = sum_ic2 / (double)cnt - mean_ic * mean_ic;
    double std_ic = (var > 0.0) ? sqrt(var) : 0.0;
    if (!isfinite(mean_ic) || std_ic < 1e-9) {
        *out_mean = isfinite(mean_ic) ? mean_ic : 0.0;
        *out_std  = std_ic;
        return 0.0;
    }
    *out_mean = mean_ic;
    *out_std  = std_ic;
    return mean_ic / std_ic;
}


/* turnover 估计 — 单 pass per-column,O(T·S)。
 *   mean_abs_d = mean_{t≥1, both finite}( |x[t,s] - x[t-1,s]| )
 *   per-col std = sqrt( (Σx² - mean·Σx) / n )      (ddof=0,population)
 *   norm = mean_{s, std>0 finite}( std )
 *   return (norm > 0 && cnt_d > 0) ? mean_abs_d / norm : 0.0
 */
double fm_turnover(const double* x, int64_t T, int64_t S) {
    if (T < 2 || S < 1) return 0.0;

    double sum_abs_d = 0.0;
    int64_t cnt_d = 0;
    double sum_sd = 0.0;
    int64_t cnt_sd = 0;

    #pragma omp parallel for schedule(static) \
        reduction(+:sum_abs_d) reduction(+:cnt_d) \
        reduction(+:sum_sd)    reduction(+:cnt_sd)
    for (int64_t s = 0; s < S; s++) {
        double cs = 0.0, css = 0.0;
        int64_t cc = 0;
        double prev = x[0 * S + s];
        if (!isnan(prev)) { cs += prev; css += prev * prev; cc++; }
        for (int64_t t = 1; t < T; t++) {
            double cur = x[t * S + s];
            if (!isnan(cur)) { cs += cur; css += cur * cur; cc++; }
            if (!isnan(cur) && !isnan(prev)) {
                double d = cur - prev;
                sum_abs_d += (d >= 0.0) ? d : -d;
                cnt_d++;
            }
            prev = cur;
        }
        if (cc >= 2) {
            double mean = cs / (double)cc;
            double var  = (css - mean * cs) / (double)cc;       /* ddof=0 */
            if (var > 0.0) {
                double sd = sqrt(var);
                if (isfinite(sd) && sd > 0.0) {
                    sum_sd += sd;
                    cnt_sd++;
                }
            }
        }
    }

    if (cnt_d == 0 || cnt_sd == 0) return 0.0;
    double mean_abs_d = sum_abs_d / (double)cnt_d;
    double norm = sum_sd / (double)cnt_sd;
    if (norm <= 0.0) return 0.0;
    return mean_abs_d / norm;
}


/* 候选 per_t_pnl 对池各成员的 Pearson 相关向量(AlphaPool 边际 Δens 的 _corr_with_pool C 化）。
 *   cand (T,);  members (n,T) row-major;  out (n,) 预分配。OMP 跨成员并行。
 *   每成员仅取与 cand 同 finite 的 bar,中心化后 Pearson;共同 finite<30 → 0.0;
 *   den = sqrt(Σa²·Σb²)+1e-12(与 Python _calc_pnl_corr 逐项一致:da/db≈0 时 num=0 → 0)。 */
void fm_pnl_corr_vec(const double* cand, const double* members, double* out, int64_t n, int64_t T) {
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < n; j++) {
        const double* mj = members + j * T;
        double sa = 0.0, sb = 0.0;
        int64_t cnt = 0;
        for (int64_t t = 0; t < T; t++) {
            double a = cand[t], b = mj[t];
            if (isfinite(a) && isfinite(b)) { sa += a; sb += b; cnt++; }
        }
        if (cnt < 30) { out[j] = 0.0; continue; }
        double ma = sa / (double)cnt, mb = sb / (double)cnt;
        double num = 0.0, da = 0.0, db = 0.0;
        for (int64_t t = 0; t < T; t++) {
            double a = cand[t], b = mj[t];
            if (isfinite(a) && isfinite(b)) {
                double xa = a - ma, xb = b - mb;
                num += xa * xb; da += xa * xa; db += xb * xb;
            }
        }
        out[j] = num / (sqrt(da * db) + 1e-12);
    }
}


int fm_omp_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

/* 局级多进程 worker 升线程通道:主进程全程 OMP_NUM_THREADS=1(libgomp 零线程池
 * → fork 无毒),fork 后 worker 调这里把 ICV 升回 N → 子进程 fresh 起线程。 */
void fm_omp_set_num_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

/* glibc brk 堆空洞归还(2026-06-07 OOM 病灶):Pass1 pandas 把 mmap_threshold 动态抬满
 * (32MB)+ brk 抬高数 GB 后,Pass2 的 GB 级 np.full 从 brk top 切出 → free 后钉在堆里
 * 永不还 OS(diag 实测 7GB 幽灵)。malloc_trim(0) 遍历 arena 空闲块 madvise(DONTNEED)
 * 归还物理页。fork 前 / worker 每局末调。非 glibc(Windows .pyd)no-op 返 0。 */
int fm_malloc_trim(void) {
#ifdef __GLIBC__
    extern int malloc_trim(size_t);
    return malloc_trim(0);
#else
    return 0;
#endif
}
