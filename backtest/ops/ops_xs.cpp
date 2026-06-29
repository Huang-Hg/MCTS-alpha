/* factor_mining/ops/ops_elementwise.c — 元素逐位算子(无窗口、无截面、无 reduction)
 *
 * unary:fm_abs / fm_neg / fm_sign / fm_log / fm_sqrt_ / fm_square / fm_tanh / fm_inv / fm_s_log_1p
 * binary:fm_add / fm_sub / fm_mul / fm_div / fm_max_b / fm_min_b
 * binary_const:fm_add_const / fm_mul_const / fm_pow_const(signed_power)
 *
 * 设计原则:
 *   - 算术 ops (add/sub/mul,unary 的 abs/neg/square/tanh):IEEE 754 NaN 隐式传播,
 *     无显式 isnan 分支 → -O3 -march=native 自动 SIMD vectorize。
 *   - 比较/选择 ops (max/min/log/sqrt/inv/sign):NaN 比较恒 false 会"吞 NaN"成
 *     0/选错值,必须显式 isnan 检查。
 *   - div/pow_const:特殊值(b==0,base==0)显式分支,其余靠 IEEE 自然传播。
 *   - build.sh 须保留 -fno-finite-math-only;否则 GCC ≥ 14 会优化掉 isnan/isfinite。
 *   - 2D strided 寻址(ldx/ldo 行步长,块化求值器零拷贝切列):逐元素表达式与
 *     旧 flat 版完全相同 → 数值逐位一致;ld == S 即原 C-连续语义。
 */

#include "ops.hpp"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>     /* malloc_trim(fm_malloc_trim) */

#include <algorithm>    /* std::sort */
#include <utility>      /* std::pair */
#include <vector>       /* std::vector scratch(RAII,取代手动 malloc/free) */

#ifdef _OPENMP
#include <omp.h>
#endif


/* ----- unary ----- */

#define FM_UNARY(name, expr) \
void name(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo) { \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* xr  = x   + t * ldx; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double v = xr[s]; \
            outr[s] = (expr); \
        } \
    } \
}

FM_UNARY(fm_abs,    fabs(v))
FM_UNARY(fm_neg,    -v)
FM_UNARY(fm_sign,   isnan(v) ? NAN : (v > 0.0) ? 1.0 : (v < 0.0) ? -1.0 : 0.0)
FM_UNARY(fm_log,    (isnan(v) || v <= 0.0) ? NAN : log(v))
FM_UNARY(fm_sqrt_,  (isnan(v) || v < 0.0) ? NAN : sqrt(v))
FM_UNARY(fm_square, isnan(v) ? NAN : v * v)
FM_UNARY(fm_tanh,   isnan(v) ? NAN : tanh(v))
FM_UNARY(fm_inv,    (isnan(v) || v == 0.0) ? NAN : (1.0 / v))
/* s_log_1p: sign(x) · log(1 + |x|),压缩重尾保留符号,定义域全实数。 */
FM_UNARY(fm_s_log_1p, isnan(v) ? NAN : (v >= 0.0) ? log1p(v) : -log1p(-v))


/* f32 → f64 块升格(块化求值器 f32 operand 叶)。cast 精确无舍入。 */
void fm_upcast32(const float* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        const float* xr  = x   + t * ldx;
        double*      outr = out + t * ldo;
        for (int64_t s = 0; s < S; s++) outr[s] = (double)xr[s];
    }
}


/* ----- EvalCache dense-pack(无损稀疏存储)----- */

/* pass1:row_off[t+1] = 行 t 非 NaN 数(OMP),随后串行前缀和(T~21 万,<1ms)。 */
void fm_pack_sparse_scan(const double* x, int64_t T, int64_t S, int64_t* row_off) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        const double* xr = x + t * S;
        int64_t c = 0;
        for (int64_t s = 0; s < S; s++) c += !isnan(xr[s]);
        row_off[t + 1] = c;
    }
    row_off[0] = 0;
    for (int64_t t = 0; t < T; t++) row_off[t + 1] += row_off[t];
}

/* pass2:bitmap 行对齐(跨行无共享字 → 行并行安全);每字寄存器内拼好整字一次落地。 */
void fm_pack_sparse_fill(const double* x, int64_t T, int64_t S,
                         const int64_t* row_off, uint64_t* bitmap, double* values) {
    int64_t words = (S + 63) >> 6;
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        const double* xr = x + t * S;
        uint64_t* bm = bitmap + t * words;
        double* v = values + row_off[t];
        int64_t k = 0;
        for (int64_t w = 0; w < words; w++) {
            uint64_t word = 0;
            int64_t s0 = w << 6;
            int64_t s1 = (s0 + 64 < S) ? s0 + 64 : S;
            for (int64_t s = s0; s < s1; s++) {
                double val = xr[s];
                if (!isnan(val)) { word |= 1ULL << (s - s0); v[k++] = val; }
            }
            bm[w] = word;
        }
    }
}

void fm_unpack_sparse(const uint64_t* bitmap, const double* values, const int64_t* row_off,
                      int64_t T, int64_t S, double* out) {
    int64_t words = (S + 63) >> 6;
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        const uint64_t* bm = bitmap + t * words;
        const double* v = values + row_off[t];
        double* outr = out + t * S;
        int64_t k = 0;
        for (int64_t s = 0; s < S; s++) {
            outr[s] = ((bm[s >> 6] >> (s & 63)) & 1) ? v[k++] : NAN;
        }
    }
}


/* ----- binary ----- */

#define FM_BIN_ARITH(name, expr) \
void name(const double* a, const double* b, double* out, int64_t T, int64_t S, \
          int64_t lda, int64_t ldb, int64_t ldo) { \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* ar  = a   + t * lda; \
        const double* br  = b   + t * ldb; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double va = ar[s], vb = br[s]; \
            outr[s] = (expr); \
        } \
    } \
}

#define FM_BIN_NAN(name, expr) \
void name(const double* a, const double* b, double* out, int64_t T, int64_t S, \
          int64_t lda, int64_t ldb, int64_t ldo) { \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* ar  = a   + t * lda; \
        const double* br  = b   + t * ldb; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double va = ar[s], vb = br[s]; \
            outr[s] = (isnan(va) || isnan(vb)) ? NAN : (expr); \
        } \
    } \
}

FM_BIN_ARITH(fm_add, va + vb)
FM_BIN_ARITH(fm_sub, va - vb)
FM_BIN_ARITH(fm_mul, va * vb)
/* div:vb == 0.0 须显式检查(IEEE 754 div-by-0 → ±Inf,而我们要 NaN);NaN 由 va/vb 自然传播。 */
FM_BIN_ARITH(fm_div, (vb == 0.0) ? NAN : (va / vb))

FM_BIN_NAN(fm_max_b, (va > vb) ? va : vb)
FM_BIN_NAN(fm_min_b, (va < vb) ? va : vb)


/* ----- binary_const(panel ⊕ scalar k,1 SIMD pass) ----- */

#define FM_BIN_CONST(name, expr) \
void name(const double* x, double k, double* out, int64_t T, int64_t S, \
          int64_t ldx, int64_t ldo) { \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* xr  = x   + t * ldx; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double v = xr[s]; \
            outr[s] = (expr); \
        } \
    } \
}

FM_BIN_CONST(fm_add_const, v + k)
FM_BIN_CONST(fm_mul_const, v * k)

/* signed_power 内循环模板:0→0(防 0^k=inf/NaN @ k<0);其他 sign(v)·P(av),av=|v|。
 * NaN 由 fabs/P 自然传播(fabs(NaN)=NaN);0 检查不可省。P 表达式按 k 特化。 */
#define FM_SIGNED_POW_LOOP(P) \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* xr  = x   + t * ldx; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double v = xr[s]; \
            double av = fabs(v); \
            if (av == 0.0) { outr[s] = 0.0; continue; } \
            double p = (P); \
            outr[s] = (v < 0.0) ? -p : p; \
        } \
    }

void fm_pow_const(const double* x, double k, double* out, int64_t T, int64_t S,
                  int64_t ldx, int64_t ldo) {
    /* k ∈ MULPOW_CONSTANTS {±2, ±0.5}(grammar 唯一来源):用 mul/sqrt 替 libm pow
     * (~5-11× on pow_const,pow_const 主导 bin_c)。0/sign/NaN 处理与原 pow 版逐位等价:
     * av*av = pow(av,2) 精确;sqrt = pow(av,.5) IEEE 正确舍入;负指数同式取倒。
     * 其余 k 退通用 pow(kernel 契约,当前 grammar 不触发)。 */
    if      (k ==  2.0) { FM_SIGNED_POW_LOOP(av * av) }
    else if (k == -2.0) { FM_SIGNED_POW_LOOP(1.0 / (av * av)) }
    else if (k ==  0.5) { FM_SIGNED_POW_LOOP(sqrt(av)) }
    else if (k == -0.5) { FM_SIGNED_POW_LOOP(1.0 / sqrt(av)) }
    else                { FM_SIGNED_POW_LOOP(pow(av, k)) }
}

/* factor_mining/ops/ops_cs.c — Cross-sectional kernels
 *
 * fm_cs_rank / fm_cs_zscore / fm_cs_demean / fm_cs_finite_validstd / fm_cs_scale
 *
 * 每 t 一行截面运算,跨 t OMP 并行(每 t 独立)。
 * cs_rank:活跃集 compaction(per-bar carried ~220)后 std::sort 按值升序 + 等值组平均-rank。
 * tie-invariant:组内顺序不影响输出(等值组赋同一 avg-rank)→ value-only 比较即逐位一致。
 */

/* (value, orig_s) 对;std::sort 按 value 升序(tie 顺序未定但输出 tie-invariant)。 */
typedef std::pair<double, int64_t> _vi_pair;

static void cs_rank_one_row(const double* row, double* out_row, int64_t S, _vi_pair* buf) {
    int64_t n_valid = 0;
    for (int64_t s = 0; s < S; s++) {
        double v = row[s];
        if (!isnan(v)) buf[n_valid++] = { v, s };
    }
    if (n_valid < 2) {
        for (int64_t s = 0; s < S; s++) out_row[s] = NAN;
        return;
    }
    std::sort(buf, buf + n_valid,
              [](const _vi_pair& a, const _vi_pair& b) { return a.first < b.first; });
    /* 同值组取平均 rank(组内顺序无关 → bit-identical) */
    int64_t i = 0;
    while (i < n_valid) {
        int64_t j = i;
        while (j + 1 < n_valid && buf[j+1].first == buf[i].first) j++;
        double avg_norm = ((double)(i + j) / 2.0) / (double)(n_valid - 1);
        for (int64_t k = i; k <= j; k++) out_row[buf[k].second] = avg_norm;
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
        std::vector<_vi_pair> buf(S);            /* per-thread scratch(RAII) */
        #pragma omp for schedule(static)
        for (int64_t t = 0; t < T; t++)
            cs_rank_one_row(&x[t * S], &out[t * S], S, buf.data());
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

/* factor_mining/ops/ops_metrics.c — 因子评估指标
 *
 * fm_ic / fm_per_t_ic / fm_per_t_pnl / fm_rank_ic / fm_turnover
 * fm_omp_max_threads
 *
 * 设计原则:全部跨 t 或跨 (m,k) OMP 并行,reduction(+) 合 sum_corr/cnt;
 * isfinite 同时挡 NaN + ±Inf(Inf 会让 num=Inf−Inf=NaN 污染 reduction)。
 */


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
        std::vector<double>  rx(S), ry(S), vals(S);    /* per-thread scratch(RAII) */
        std::vector<int64_t> idx(S);
        #pragma omp for schedule(static)
        for (int64_t t = 0; t < T; t++) {
            _rank_row(&x[t * S], rx.data(), S, vals.data(), idx.data());
            _rank_row(&y[t * S], ry.data(), S, vals.data(), idx.data());
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
