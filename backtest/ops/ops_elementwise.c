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

#include "ops.h"

#include <math.h>
#include <stdint.h>

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
