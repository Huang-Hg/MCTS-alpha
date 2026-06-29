/* factor_mining/ops_cuda/kernels.cu — 主程序级 CUDA alpha 算子(templated C++,NVRTC 经 cupy 编译)
 *
 * 与 CPU 侧 factor_mining/ops/*.c 一一对应、语义对齐(真 C = ground truth,容差 ~1e-6;
 * GPU/CPU FP 累加序不同 → 非 bitwise)。**分开写程序**:CPU=C 扩展,CUDA=此 .cu。
 *
 * 设计(源自 FACT + V100/T1000 实证,见 reference_openbayes_v100_cuda):
 *   - 行优先 (T,S):x[t*S+s];面板瘦高(T≫S)→ 一线程一输出 (t,s) 喂满 SM。
 *   - 滑窗 shared-mem tile:一 block=(SS symbol)×(TT 时间)协作载入 (TT+w-1)×SS tile+halo。
 *   - rolling 规则化简:out[t,s] = (t>=w-1 且窗口[t-w+1,t]无 NaN) ? stat : NaN
 *     (列 range-trim 在 C 里只是性能优化;跨上市边界窗口必含 NaN → 语义等价此简单规则)。
 *   - 中心化锚 K:kmode0=窗口最旧值(sum/mean/std/mad,数值干净);kmode1=列首有效值
 *     Kcol[s](skew/kurt,复刻 C 大偏移抵消 → 对齐 m2<=1e-12 退化门)。
 *
 * **templated C++**:template<typename T> __device__ 核心 + extern "C" f64/f32 实例化
 *   (cupy RawModule 按平名 get_function('k_*_f64'/'k_*_f32'));替掉旧 .replace('double','float')。
 *   消费卡(FP64 1:32)走 f32 实例(2-3.6× 快),数据中心卡走 f64。
 *   scan(flat-in-w)是 f64-only 性能优化、且 fp32 全局 cumsum 已实证灾难 → 不在此文件,
 *   主程序求值器用 tile 核(全 dtype 安全)。
 */

/* 动态 shared 的 typed 视图:模板 T 下避开多个 extern __shared__ 同名异型冲突 */
template<typename T> __device__ __forceinline__ T* shmem() {
    extern __shared__ __align__(8) unsigned char smem[];
    return reinterpret_cast<T*>(smem);
}


/* ============================================================================
 * 滑窗矩核:op 0=sum 1=mean 2=std 3=skew 4=kurt 5=mad
 * ========================================================================== */
template<typename T>
__device__ void moment_core(const T* __restrict__ x, T* __restrict__ out,
                            int Tt, int S, int w, int op, const T* __restrict__ Kcol, int kmode) {
    T* sh = shmem<T>();
    int SS = blockDim.x, TT = blockDim.y;
    int strip_s0 = blockIdx.x * SS, tile_t0 = blockIdx.y * TT;
    int rows = TT + w - 1, base_t = tile_t0 - (w - 1);
    int tid = threadIdx.y * SS + threadIdx.x, nth = SS * TT;
    for (int idx = tid; idx < rows * SS; idx += nth) {
        int i = idx / SS, sx = idx - i * SS;
        int gt = base_t + i, gs = strip_s0 + sx;
        sh[i * SS + sx] = (gt < 0 || gt >= Tt || gs >= S) ? (T)nan("")
                                                          : x[(long long)gt * S + gs];
    }
    __syncthreads();

    int sx = threadIdx.x, ty = threadIdx.y;
    int s = strip_s0 + sx, t = tile_t0 + ty;
    if (s >= S || t >= Tt) return;
    long long o = (long long)t * S + s;

    T K = (kmode == 1) ? Kcol[s] : sh[ty * SS + sx];
    T s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    int bad = 0;
    for (int k = 0; k < w; k++) {
        T v = sh[(ty + k) * SS + sx];
        if (isnan(v)) { bad = 1; break; }
        T d = v - K;
        s1 += d;
        if (op >= 2 && op != 5) { T d2 = d * d; s2 += d2;
            if (op >= 3) { s3 += d2 * d;
                if (op >= 4) s4 += d2 * d2; } }
    }
    if (bad) { out[o] = (T)nan(""); return; }

    T wd = (T)w, res;
    if (op == 0)      res = s1 + wd * K;                          /* sum  */
    else if (op == 1) res = (s1 + wd * K) / wd;                   /* mean */
    else if (op == 5) {                                           /* mad = mean(|x-mean|) */
        T mean = (s1 + wd * K) / wd, mad = 0;
        for (int k = 0; k < w; k++) mad += fabs(sh[(ty + k) * SS + sx] - mean);
        res = mad / wd;
    } else if (op == 2) {                                         /* std  */
        T Sx = s1 + wd * K, Sxx = s2 + (T)2 * K * s1 + wd * K * K;
        T mean = Sx / wd, var = (Sxx - mean * Sx) / (wd - (T)1);
        if (var < 0) var = 0;
        res = sqrt(var);
    } else {                                                      /* skew/kurt:窗内两遍中心矩 */
        /* pass1 窗均值 → pass2 直接中心矩 Σ(v-mean)^p(消 e2−mu² 抵消)。与 C fm_ts_skew/kurt
         * 同算法同顺序(窗最旧→最新)→ 逐位一致(替旧增量幂和:真盘 f64 与 GPU 分叉 kurt~40%,
         * m2<=1e-8·e2 floor 证伪;两遍稳定后近常数窗也产同一(大)值 → 不再分叉)。 */
        T mean = 0;
        for (int k = 0; k < w; k++) mean += sh[(ty + k) * SS + sx];
        mean /= wd;
        T m2 = 0, m3 = 0, m4 = 0;
        for (int k = 0; k < w; k++) {
            T d = sh[(ty + k) * SS + sx] - mean, d2 = d * d;
            m2 += d2; m3 += d2 * d; m4 += d2 * d2;
        }
        m2 /= wd; m3 /= wd; m4 /= wd;
        if (m2 <= (T)0) { out[o] = (T)nan(""); return; }            /* 常数窗 → NaN */
        res = (op == 3) ? m3 / pow(m2, (T)1.5) : m4 / (m2 * m2) - (T)3;
    }
    out[o] = res;
}


/* ============================================================================
 * 滑窗扩展核:op 0=max 1=min 2=wma 3=slope 4=rank 5=arg_max 6=arg_min
 *   arg_*:emit (w-1)-argk = 距今多少 bar(∈[0,w-1]),ties→最近(>= / <=,贴 C deque)。
 * ========================================================================== */
template<typename T>
__device__ void ext_core(const T* __restrict__ x, T* __restrict__ out,
                         int Tt, int S, int w, int op) {
    T* sh = shmem<T>();
    int SS = blockDim.x, TT = blockDim.y;
    int strip_s0 = blockIdx.x * SS, tile_t0 = blockIdx.y * TT;
    int rows = TT + w - 1, base_t = tile_t0 - (w - 1);
    int tid = threadIdx.y * SS + threadIdx.x, nth = SS * TT;
    for (int idx = tid; idx < rows * SS; idx += nth) {
        int i = idx / SS, sx = idx - i * SS, gt = base_t + i, gs = strip_s0 + sx;
        sh[i * SS + sx] = (gt < 0 || gt >= Tt || gs >= S) ? (T)nan("") : x[(long long)gt * S + gs];
    }
    __syncthreads();
    int sx = threadIdx.x, ty = threadIdx.y, s = strip_s0 + sx, t = tile_t0 + ty;
    if (s >= S || t >= Tt) return;
    long long o = (long long)t * S + s;
    int bad = 0;
    if (op == 0 || op == 1) {                                    /* max / min */
        T m = sh[ty * SS + sx];                                  /* 窗口最旧播种(避 NVRTC 无 INFINITY) */
        if (isnan(m)) { out[o] = (T)nan(""); return; }
        for (int k = 1; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } m = (op == 0) ? fmax(m, v) : fmin(m, v); }
        out[o] = bad ? (T)nan("") : m;
    } else if (op == 5 || op == 6) {                             /* arg_max / arg_min */
        T m = sh[ty * SS + sx]; int argk = 0;
        if (isnan(m)) { out[o] = (T)nan(""); return; }
        for (int k = 1; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; }
            if ((op == 5 && v >= m) || (op == 6 && v <= m)) { m = v; argk = k; } }
        out[o] = bad ? (T)nan("") : (T)(w - 1 - argk);
    } else if (op == 2) {                                        /* wma */
        T wsum = 0, denom = (T)w * ((T)w + (T)1) / (T)2;
        for (int k = 0; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } wsum += (T)(k + 1) * v; }
        out[o] = bad ? (T)nan("") : wsum / denom;
    } else if (op == 3) {                                        /* slope */
        T Sx = 0, Skx = 0;
        for (int k = 0; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } Sx += v; Skx += (T)k * v; }
        if (bad) { out[o] = (T)nan(""); return; }
        T mean_t = (T)(w - 1) / (T)2, var_t = (T)((long long)w * w - 1) / (T)12;
        out[o] = (Skx - mean_t * Sx) / (T)w / var_t;
    } else {                                                     /* rank */
        T cur = sh[(ty + w - 1) * SS + sx];
        if (isnan(cur)) { out[o] = (T)nan(""); return; }
        int cl = 0, ce = 0;
        for (int k = 0; k < w; k++) { T v = sh[(ty + k) * SS + sx];
            if (isnan(v)) { bad = 1; break; } if (v < cur) cl++; else if (v == cur) ce++; }
        out[o] = bad ? (T)nan("") : ((T)cl + (T)0.5 * (T)(ce - 1)) / (T)(w - 1);
    }
}


/* ts_corr / ts_cov:成对 tile;op 0=corr 1=cov;任一侧 NaN→NaN。shared=2*rows*SS*sizeof(T)。 */
template<typename T>
__device__ void pair_core(const T* __restrict__ a, const T* __restrict__ b,
                         T* __restrict__ out, int Tt, int S, int w, int op) {
    T* sh = shmem<T>();
    int SS = blockDim.x, TT = blockDim.y, rows = TT + w - 1;
    T* sha = sh; T* shb = sh + rows * SS;
    int strip_s0 = blockIdx.x * SS, tile_t0 = blockIdx.y * TT, base_t = tile_t0 - (w - 1);
    int tid = threadIdx.y * SS + threadIdx.x, nth = SS * TT;
    for (int idx = tid; idx < rows * SS; idx += nth) {
        int i = idx / SS, sx = idx - i * SS, gt = base_t + i, gs = strip_s0 + sx;
        int oob = (gt < 0 || gt >= Tt || gs >= S);
        sha[i * SS + sx] = oob ? (T)nan("") : a[(long long)gt * S + gs];
        shb[i * SS + sx] = oob ? (T)nan("") : b[(long long)gt * S + gs];
    }
    __syncthreads();
    int sx = threadIdx.x, ty = threadIdx.y, s = strip_s0 + sx, t = tile_t0 + ty;
    if (s >= S || t >= Tt) return;
    long long o = (long long)t * S + s;
    /* K-centering(窗口最旧值,平移不变 → cov/var 不变):压住大幅值(如 close~1e2 vs vol~1e6)
     * 的乘积 Σab 灾难性抵消,这是 f32 下 corr/cov 数值稳定的关键(同 moment 核 std 思路)。*/
    T Ka = sha[ty * SS + sx], Kb = shb[ty * SS + sx];
    T Sa = 0, Sb = 0, Sab = 0, Saa = 0, Sbb = 0; int bad = 0;
    for (int k = 0; k < w; k++) {
        T va = sha[(ty + k) * SS + sx], vb = shb[(ty + k) * SS + sx];
        if (isnan(va) || isnan(vb)) { bad = 1; break; }
        T da = va - Ka, db = vb - Kb;
        Sa += da; Sb += db; Sab += da * db; Saa += da * da; Sbb += db * db;
    }
    if (bad) { out[o] = (T)nan(""); return; }
    T wd = (T)w, ma = Sa / wd, mb = Sb / wd, cov = (Sab - ma * Sb) / (wd - (T)1);
    if (op == 1) { out[o] = cov; return; }
    T va_ = (Saa - ma * Sa) / (wd - (T)1), vb_ = (Sbb - mb * Sb) / (wd - (T)1);
    out[o] = (va_ > 0 && vb_ > 0) ? cov / sqrt(va_ * vb_) : (T)nan("");
}


/* ts_ema:pandas ewm(span,adjust=False,min_periods=span)。序列递推 → 一线程一列(沿 T 串行)。 */
template<typename T>
__device__ void ema_core(const T* __restrict__ x, T* __restrict__ out, int Tt, int S, int span) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    T alpha = (T)2 / (T)(span + 1);
    T ema = 0; int has_seed = 0; long long cnt = 0;
    for (int t = 0; t < Tt; t++) {
        long long o = (long long)t * S + s;
        T v = x[o];
        if (isnan(v)) { out[o] = (T)nan(""); continue; }
        ema = has_seed ? (alpha * v + ((T)1 - alpha) * ema) : v;
        has_seed = 1; cnt++;
        out[o] = (cnt >= span) ? ema : (T)nan("");
    }
}


/* ============================================================================
 * cross-sectional warp-shuffle 核:op 0=zscore 1=demean 2=scale(L1)
 * ========================================================================== */
template<typename T> __device__ __forceinline__ T warp_reduce_sum(T v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

template<typename T>
__device__ void cs_core(const T* __restrict__ x, T* __restrict__ out, int S, int op) {
    int t = blockIdx.x;
    const T* row = x + (long long)t * S;
    T* orow = out + (long long)t * S;
    int tid = threadIdx.x, B = blockDim.x;
    int lane = tid & 31, wid = tid >> 5, nwarp = B >> 5;

    T lsum = 0, lsq = 0; int ln = 0;
    for (int s = tid; s < S; s += B) {
        T v = row[s];
        if (!isnan(v)) { lsum += v; ln++;
            if (op == 0) lsq += v * v; else if (op == 2) lsq += fabs(v); }
    }
    lsum = warp_reduce_sum<T>(lsum);
    T lsqr = warp_reduce_sum<T>(lsq);
    T lnr  = warp_reduce_sum<T>((T)ln);
    __shared__ T wsum[32], wsq[32], wn[32];
    if (lane == 0) { wsum[wid] = lsum; wsq[wid] = lsqr; wn[wid] = lnr; }
    __syncthreads();
    __shared__ T mean_sh, p_sh; __shared__ int ok_sh;
    if (wid == 0) {
        T sm = (lane < nwarp) ? wsum[lane] : (T)0;
        T sq = (lane < nwarp) ? wsq[lane]  : (T)0;
        T nn = (lane < nwarp) ? wn[lane]   : (T)0;
        sm = warp_reduce_sum<T>(sm); sq = warp_reduce_sum<T>(sq); nn = warp_reduce_sum<T>(nn);
        if (lane == 0) {
            long long n = (long long)nn; ok_sh = 0;
            if (op == 1) { if (n >= 1) { mean_sh = sm / (T)n; ok_sh = 1; } }
            else if (op == 2) { if (sq > (T)1e-12) { p_sh = sq; ok_sh = 1; } }
            else { if (n >= 2) { T mean = sm / (T)n;
                       T var = (sq - mean * sm) / (T)(n - 1);
                       if (var > 0) { mean_sh = mean; p_sh = sqrt(var); ok_sh = 1; } } }
        }
    }
    __syncthreads();
    if (!ok_sh) { for (int s = tid; s < S; s += B) orow[s] = (T)nan(""); return; }
    T mean = mean_sh, p = p_sh;
    for (int s = tid; s < S; s += B) {
        T v = row[s];
        if (isnan(v)) { orow[s] = (T)nan(""); continue; }
        if (op == 0)      orow[s] = (v - mean) / p;
        else if (op == 1) orow[s] = v - mean;
        else              orow[s] = v / p;
    }
}


/* cs_rank:逐 t 截面平均-rank ∈[0,1];n<2 整行 NaN。一 block 一 t,行入 shared,O(n²) count。 */
template<typename T>
__device__ void cs_rank_core(const T* __restrict__ x, T* __restrict__ out, int S) {
    T* r = shmem<T>();
    int t = blockIdx.x; const T* row = x + (long long)t * S; T* orow = out + (long long)t * S;
    int tid = threadIdx.x, B = blockDim.x;
    for (int s = tid; s < S; s += B) r[s] = row[s];
    __shared__ int nval; if (tid == 0) nval = 0;
    __syncthreads();
    int ln = 0; for (int s = tid; s < S; s += B) if (!isnan(r[s])) ln++;
    atomicAdd(&nval, ln); __syncthreads();
    int n = nval;
    if (n < 2) { for (int s = tid; s < S; s += B) orow[s] = (T)nan(""); return; }
    for (int s = tid; s < S; s += B) {
        T v = r[s];
        if (isnan(v)) { orow[s] = (T)nan(""); continue; }
        int cl = 0, ce = 0;
        for (int j = 0; j < S; j++) { T u = r[j]; if (isnan(u)) continue;
            if (u < v) cl++; else if (u == v) ce++; }
        orow[s] = ((T)cl + (T)0.5 * (T)(ce - 1)) / (T)(n - 1);
    }
}


/* ============================================================================
 * extern "C" f64/f32 实例化(cupy 按平名 get_function)
 * ========================================================================== */
#define INST_TS(name, core) \
    extern "C" __global__ void name##_f64(const double* x, double* o, int T, int S, int w, int op, const double* K, int km) { core<double>(x, o, T, S, w, op, K, km); } \
    extern "C" __global__ void name##_f32(const float*  x, float*  o, int T, int S, int w, int op, const float*  K, int km) { core<float >(x, o, T, S, w, op, K, km); }
INST_TS(k_moment, moment_core)

#define INST_EXT(name, core) \
    extern "C" __global__ void name##_f64(const double* x, double* o, int T, int S, int w, int op) { core<double>(x, o, T, S, w, op); } \
    extern "C" __global__ void name##_f32(const float*  x, float*  o, int T, int S, int w, int op) { core<float >(x, o, T, S, w, op); }
INST_EXT(k_ext, ext_core)

#define INST_PAIR(name, core) \
    extern "C" __global__ void name##_f64(const double* a, const double* b, double* o, int T, int S, int w, int op) { core<double>(a, b, o, T, S, w, op); } \
    extern "C" __global__ void name##_f32(const float*  a, const float*  b, float*  o, int T, int S, int w, int op) { core<float >(a, b, o, T, S, w, op); }
INST_PAIR(k_pair, pair_core)

extern "C" __global__ void k_ema_f64(const double* x, double* o, int T, int S, int span) { ema_core<double>(x, o, T, S, span); }
extern "C" __global__ void k_ema_f32(const float*  x, float*  o, int T, int S, int span) { ema_core<float >(x, o, T, S, span); }

extern "C" __global__ void k_cs_f64(const double* x, double* o, int S, int op) { cs_core<double>(x, o, S, op); }
extern "C" __global__ void k_cs_f32(const float*  x, float*  o, int S, int op) { cs_core<float >(x, o, S, op); }

extern "C" __global__ void k_cs_rank_f64(const double* x, double* o, int S) { cs_rank_core<double>(x, o, S); }
extern "C" __global__ void k_cs_rank_f32(const float*  x, float*  o, int S) { cs_rank_core<float >(x, o, S); }
