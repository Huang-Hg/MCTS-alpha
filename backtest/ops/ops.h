/* factor_mining/ops/ops.h
 *
 * 公式化 alpha 因子算子库 — 公共 C API
 *
 * Layout 约定:
 *   所有 (T, S) 面板用 double 行优先:data[t * ld + s] = value at (timestamp t, symbol s)。
 *   T = 时间步数,S = 列数(symbol 或列块宽),ld = 行步长(元素数,ld ≥ S)。
 *   ld == S 即 C-连续;ld > S 表示宽面板的列块视图(块化求值器零拷贝切列)。
 *   elementwise / ts / pair 每个输入与 out 各带独立 ld(追加在参数尾部);
 *   cs / metrics 仍只收 C-连续全宽面板(截面语义需要整行)。
 *   空缺值用 NaN(IEEE 754),逐元素 NaN 传播。
 *
 * Rolling 语义:
 *   pandas `rolling(w).f()` with `min_periods=w`。窗口内有任何 NaN → 输出 NaN。
 *   t < w-1 时(暖机期)输出 NaN。
 *
 * NaN 传播:
 *   binary 算子(add/mul/...)任一侧 NaN → NaN。
 *   div 中 b==0 → NaN(不抛 inf,因子表达式需要 IEEE 数值,NaN 让评估器统一过滤)。
 */

#ifndef FM_OPS_H
#define FM_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Elementwise unary ===== */
void fm_abs   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_neg   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_sign  (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_log   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* log(x) for x>0, else NaN */
void fm_sqrt_ (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* x>=0 else NaN */
void fm_square(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_tanh  (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* tanh(x) ∈ (-1, 1) */
void fm_inv   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* 1/x; x==0 → NaN */
void fm_s_log_1p(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* sign(x)·log(1+|x|) */

/* f32 → f64 块升格(块化求值器 f32 operand 叶专用;OMP 并行,
 * numpy 单线程 strided copyto 在列块上 ~2.5× 慢)。cast 精确,与 FROMANY 升格逐位一致。 */
void fm_upcast32(const float* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);

/* ===== EvalCache dense-pack(无损稀疏存储,alpha 整列 NaN 占比高)=====
 * bitmap 行对齐:words_per_row = ceil(S/64),bit s=1 ⇔ x[t,s] 非 NaN(±inf 照存 values);
 * values 按行序紧凑;row_off[t] = 行 t 在 values 的起始下标,row_off[T] = 总数。
 * 非 NaN 值 memcpy 级无损;NaN 还原为 canonical NAN(下游只 isnan,payload 无读者)。 */
void fm_pack_sparse_scan(const double* x, int64_t T, int64_t S, int64_t* row_off);  /* 计数+前缀和 */
void fm_pack_sparse_fill(const double* x, int64_t T, int64_t S,
                         const int64_t* row_off, uint64_t* bitmap, double* values);
void fm_unpack_sparse(const uint64_t* bitmap, const double* values, const int64_t* row_off,
                      int64_t T, int64_t S, double* out);

/* ===== Elementwise binary ===== */
void fm_add  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_sub  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_mul  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_div  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_max_b(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* max(a,b) */
void fm_min_b(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* min(a,b) */

/* ===== Elementwise binary_const(panel ⊕ scalar k,1 SIMD pass) ===== */
void fm_add_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_mul_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
/* signed_power(x, k) = sign(x)·|x|^k;0→0(避免 0^k=NaN/inf @ k<0)。NaN 透传。 */
void fm_pow_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);

/* ===== Rolling unary, per-symbol time series =====
 * Tier2b:std/max/min/skew/kurt/mad/slope/corr/cov 末尾收 rng_lo/rng_hi(每列 [vlo,vhi]
 * 预算的 active-range,int64×S);非 NULL 则跳过内部扫描直接用,NULL 则就地扫描(fallback)。
 * 其余 ts(mean/sum/rank/ref/delta/ema/wma/arg)签名不变(不裁剪)。 */
void fm_ts_mean (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_std  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_sum  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_max  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_min  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_ref  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* x.shift(w) */
void fm_ts_delta(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* x - x.shift(w) */
void fm_ts_ema  (const double* x, double* out, int64_t T, int64_t S, int64_t span, int64_t ldx, int64_t ldo);
void fm_ts_wma  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_rank   (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* in [0,1] */
void fm_ts_arg_max(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* bars-since-max ∈ [0,w-1] */
void fm_ts_arg_min(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_skew   (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* 3rd standardized moment */
void fm_ts_kurt   (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* excess kurtosis */
void fm_ts_mad    (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* mean abs deviation */
void fm_ts_slope  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* OLS slope vs t(增量,留扫描裁剪:值随起点漂移,Tier2b 全局起点会破 bit-identity)*/

/* ===== Rolling pair ===== */
void fm_ts_corr(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w, int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_cov (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w, int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);

/* ===== Cross-sectional ===== */
/* 每个 t 在 S symbols 上做(平均-rank)排名,归一化到 [0,1]。NaN 位置输出 NaN。 */
void fm_cs_rank   (const double* x, double* out, int64_t T, int64_t S);
/* 每个 t 标准化:(x - mean) / std,跨 symbol。NaN 位置输出 NaN。 */
void fm_cs_zscore (const double* x, double* out, int64_t T, int64_t S);
/* cs z-score,**NaN→0 填充 + ddof=0**(旧 metrics.cs_zscore_np 的 C 化;PCA 前置需无 NaN)。
 * 退化行(<1 finite 或 sd≤1e-12)整行 0。区别于 fm_cs_zscore(NaN 传播/ddof=1)。 */
void fm_cs_zscore_np(const double* x, double* out, int64_t T, int64_t S);
/* 单 pass 算 (finite_ratio, valid_ratio_cs):evaluator gate1 用。
 *   finite_ratio   = mean over t of (n_finite_per_t >= 2)
 *   valid_ratio_cs = mean over t of (cs_std > thr_std)  (biased var, n 除数)
 * 两个标量结果通过 out 指针返回,避免在 Python 侧重新对 (T,S) 走 4-5 轮 numpy。 */
void fm_cs_finite_validstd(const double* x, int64_t T, int64_t S, double thr_std,
                           double* finite_ratio_out, double* valid_ratio_cs_out);
/* 每个 t 中心化:x - row_mean,跨 symbol。NaN 位置输出 NaN。 */
void fm_cs_demean (const double* x, double* out, int64_t T, int64_t S);
/* 每个 t L1 归一化:x / Σ_s|x|。NaN 位置输出 NaN。WQ101 高频(#28/#31/#32 等)。 */
void fm_cs_scale  (const double* x, double* out, int64_t T, int64_t S);

/* ===== 因子评估指标(供 evaluator 内循环复用)=====
 * 每个 t 跨 S 做 Pearson(x_t, y_t),平均到标量 IC。
 * 跳过 valid pair < 3 或 var=0 的 t。
 */
double fm_ic     (const double* x, const double* y, int64_t T, int64_t S);
double fm_rank_ic(const double* x, const double* y, int64_t T, int64_t S);

/* 单因子 ICIR = nanmean(per_t_ic)/nanstd(per_t_ic, ddof=0)。复刻旧 metrics.icir。
 * 返回 icir;mean_ic / std_ic 经 out 指针回传。non-finite mean 或 std<1e-9 → icir=0。 */
double fm_icir(const double* x, const double* y, int64_t T, int64_t S,
               double* out_mean, double* out_std);

/* per-t IC 数组 — 给每个 t 算一次 cross-sec Pearson(α[t,:], y[t,:]),
 * 写入 out[t]。NaN 不足或 std=0 的 t → out[t] = NaN。caller 预分配 out (T,)。
 * 用途:AlphaPool ICIR-style Lasso M = cov_t(per_t_ic_i, per_t_ic_j),r = mean。
 * O(T·S),OMP 跨 t 并行。 */
void fm_per_t_ic(const double* x, const double* y, int64_t T, int64_t S, double* out);

/* holdable 覆盖率门:mean_t[#(x finite ∧ y finite)/#(y finite)] ∈ [0,1]。y NaN-mask 到
 * holdable → 量信号覆盖多少可交易截面;挡近全 NaN 稀疏信号的伪高 IC。O(T·S),OMP 跨 t。 */
double fm_cs_holdable_coverage(const double* x, const double* y, int64_t T, int64_t S);

/* per-t pseudo-PnL — 给每个 t 算 L1-normalized signed PnL(L1=1 deploy 行为代理):
 *   v[s] = values[t,s] - cs_mean(values[t,:]),  w[s] = v[s] / Σ|v|
 *   out[t] = Σ_s w[s] · y[t,s]  (NaN-safe,全 NaN 行 / Σ|v|≈0 → out[t] = 0)
 * 用途:AlphaPool D5 PnL Pearson 行为级 diversity 度量。 *
 * O(T·S),OMP 跨 t 并行,2 passes/row(mean → demean+sumabs+pnl)。 */
void fm_per_t_pnl(const double* values, const double* y, int64_t T, int64_t S, double* out);

/* turnover 估计:
 *   mean_abs_d = avg_{t≥1, finite}( |x[t,s] - x[t-1,s]| )
 *   norm       = avg_{s, std>0}( std_pop(x[:,s]) )      // ddof=0
 *   return mean_abs_d / norm  (任意条件不满足 → 0.0)
 * 单 pass per-column,O(T·S)。
 */
double fm_turnover(const double* x, int64_t T, int64_t S);



/* 候选 per_t_pnl 对池各成员的 Pearson 相关向量(AlphaPool 边际 Δens 的 _corr_with_pool C 化)。
 *   cand (T,), members (n,T) row-major, out (n,) 预分配。NaN-aware,共同 finite<30 → 0。OMP 跨成员。 */
void fm_pnl_corr_vec(const double* cand, const double* members, double* out, int64_t n, int64_t T);


/* OpenMP 编译时的最大线程数(0 = 未启用) */
int fm_omp_max_threads(void);

/* 设 OMP ICV 线程数(局级多进程 worker fork 后升线程用) */
void fm_omp_set_num_threads(int n);

/* glibc malloc_trim(0):brk 堆空洞归还 OS(非 glibc no-op 返 0) */
int fm_malloc_trim(void);

#ifdef __cplusplus
}
#endif

#endif
