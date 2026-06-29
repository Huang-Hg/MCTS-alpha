"""CUDA alpha 算子 —— 主程序级 GPU 后端(cupy RawModule,NVRTC,无需 nvcc)。

与 CPU 侧 backtest.ops 一一对应、**分开写程序**(CPU=C 扩展,CUDA=此包 + kernels.cu)。
全 panel 常驻 cupy in → cupy out;rolling/cs 走 kernels.cu 模板核,elementwise/binary/const
直接 cupy ufunc(语义对齐 ops_elementwise.c 的 NaN/边界处理)。

dtype 策略(reference_openbayes_v100_cuda 实证):消费卡 FP64 1:32 → DTYPE=float32(tile 核 2-3.6×,
窗口内有界累加 fp32 安全;skew/kurt 小窗 fp32 偏差较大但落用户接受的容差);数据中心卡 → float64。
launch 全 device-prop 驱动(candidate_tiles,sharedMemPerBlockOptin),跨卡自适应。
"""
from __future__ import annotations

import os
import numpy as np
import cupy as cp

# ===== 模块编译(templated C++,f64/f32 实例化按平名 get_function)=====
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels.cu')) as _f:
    _MOD = cp.RawModule(code=_f.read(), options=('--std=c++14',))

# ===== device-prop 驱动 launch + dtype 策略 =====
_PROPS    = cp.cuda.runtime.getDeviceProperties(0)
_WARP     = _PROPS['warpSize']
_OPTIN    = _PROPS['sharedMemPerBlockOptin']
_SMEM_SM  = _PROPS['sharedMemPerMultiprocessor']
_MAXTHR   = _PROPS['maxThreadsPerBlock']
_SHBUDGET = min(_OPTIN, _SMEM_SM // 3)
_FP64_RATIO = cp.cuda.Device(0).attributes['SingleToDoublePrecisionPerfRatio']

DTYPE = cp.float32 if _FP64_RATIO >= 8 else cp.float64
_SFX  = 'f32' if DTYPE == cp.float32 else 'f64'
_ELEM = np.dtype(DTYPE).itemsize

for _fn in ('k_moment', 'k_ext', 'k_pair', 'k_cs_rank'):
    _MOD.get_function(f'{_fn}_{_SFX}').max_dynamic_shared_size_bytes = _SHBUDGET

_k_moment  = _MOD.get_function(f'k_moment_{_SFX}')
_k_ext     = _MOD.get_function(f'k_ext_{_SFX}')
_k_pair    = _MOD.get_function(f'k_pair_{_SFX}')
_k_ema     = _MOD.get_function(f'k_ema_{_SFX}')
_k_cs      = _MOD.get_function(f'k_cs_{_SFX}')
_k_cs_rank = _MOD.get_function(f'k_cs_rank_{_SFX}')
_k_per_t_ic  = _MOD.get_function(f'k_per_t_ic_{_SFX}')
_k_per_t_pnl = _MOD.get_function(f'k_per_t_pnl_{_SFX}')
_k_pair.max_dynamic_shared_size_bytes = _OPTIN    # pair=2 数组,大 w shared 到 optin(单数组核留 _SHBUDGET)

_DUMMY = cp.zeros(1, dtype=DTYPE)        # kmode=0 占位(kernel 不解引用)


def candidate_tiles(w, budget=_SHBUDGET, mult=1):
    """device-prop 驱动 tile:shared 预算 budget 内,大 tile(复用高)优先。
    mult=每元素 shared 份数(pair 核 2 数组 → mult=2);budget 默认 _SHBUDGET(保占用),
    pair 大 w 用 _OPTIN(2 数组 ×大窗超 _SHBUDGET,牺牲占用换正确)。"""
    out = []
    for ss in (8, 16, 32):
        for tt in (256, 192, 128, 96, 64, 48, 32, 16):
            bs = ss * tt
            if bs > _MAXTHR or bs % _WARP != 0:
                continue
            if (tt + w - 1) * ss * _ELEM * mult > budget:
                continue
            out.append((ss, tt))
    out.sort(key=lambda st: -st[0] * st[1])
    return out[:4]


def col_first_valid(d_x):
    """每列首个有效值(skew/kurt 中心化锚 Kcol);全 NaN 列 → 0(对齐 C)。"""
    fin = ~cp.isnan(d_x)
    has = fin.any(axis=0)
    idx = fin.argmax(axis=0)
    K = d_x[idx, cp.arange(d_x.shape[1])]
    return cp.where(has, K, 0).astype(DTYPE)


# ===== rolling 矩核:sum/mean/std/skew/kurt/mad =====
def _moment(d_x, w, op, kmode, d_K):
    T, S = d_x.shape
    d_o = cp.empty_like(d_x)
    ss, tt = candidate_tiles(w)[0]
    grid = ((S + ss - 1) // ss, (T + tt - 1) // tt)
    _k_moment(grid, (ss, tt), (d_x, d_o, np.int32(T), np.int32(S), np.int32(w),
                               np.int32(op), d_K, np.int32(kmode)),
              shared_mem=(tt + w - 1) * ss * _ELEM)
    return d_o


def ts_sum(x, w):  return _moment(x, w, 0, 0, _DUMMY)
def ts_mean(x, w): return _moment(x, w, 1, 0, _DUMMY)
def ts_std(x, w):  return _moment(x, w, 2, 0, _DUMMY)
def ts_mad(x, w):  return _moment(x, w, 5, 0, _DUMMY)
def ts_skew(x, w): return _moment(x, w, 3, 1, col_first_valid(x))
def ts_kurt(x, w): return _moment(x, w, 4, 1, col_first_valid(x))


# ===== rolling 扩展核:max/min/wma/slope/rank/arg_max/arg_min =====
def _ext(d_x, w, op):
    T, S = d_x.shape
    d_o = cp.empty_like(d_x)
    ss, tt = candidate_tiles(w)[0]
    grid = ((S + ss - 1) // ss, (T + tt - 1) // tt)
    _k_ext(grid, (ss, tt), (d_x, d_o, np.int32(T), np.int32(S), np.int32(w), np.int32(op)),
           shared_mem=(tt + w - 1) * ss * _ELEM)
    return d_o


def ts_max(x, w):     return _ext(x, w, 0)
def ts_min(x, w):     return _ext(x, w, 1)
def ts_wma(x, w):     return _ext(x, w, 2)
def ts_slope(x, w):   return _ext(x, w, 3)
def ts_rank(x, w):    return _ext(x, w, 4)
def ts_arg_max(x, w): return _ext(x, w, 5)
def ts_arg_min(x, w): return _ext(x, w, 6)


# ===== rolling pair:corr/cov =====
def _pair(d_a, d_b, w, op):
    T, S = d_a.shape
    d_o = cp.empty_like(d_a)
    ss, tt = candidate_tiles(w, _OPTIN, 2)[0]      # 自适应(2 数组,_OPTIN 预算)→ 大 w 不撞 shared 上限
    grid = ((S + ss - 1) // ss, (T + tt - 1) // tt)
    _k_pair(grid, (ss, tt), (d_a, d_b, d_o, np.int32(T), np.int32(S), np.int32(w), np.int32(op)),
            shared_mem=2 * (tt + w - 1) * ss * _ELEM)
    return d_o


def ts_corr(a, b, w): return _pair(a, b, w, 0)
def ts_cov(a, b, w):  return _pair(a, b, w, 1)


# ===== ema(一线程一列,序列递推)=====
def ts_ema(d_x, span):
    T, S = d_x.shape
    d_o = cp.empty_like(d_x)
    tpb = 128
    _k_ema(((S + tpb - 1) // tpb,), (tpb,), (d_x, d_o, np.int32(T), np.int32(S), np.int32(span)))
    return d_o


# ===== 序列引用/差分(纯切片,无核)=====
def ts_ref(d_x, w):
    d_o = cp.full_like(d_x, cp.nan)
    if w < d_x.shape[0]:
        d_o[w:] = d_x[:-w]
    return d_o


def ts_delta(d_x, w):
    return d_x - ts_ref(d_x, w)


# ===== cross-sectional:zscore/demean/scale =====
def _cs(d_x, op):
    T, S = d_x.shape
    d_o = cp.empty_like(d_x)
    _k_cs((T,), (256,), (d_x, d_o, np.int32(S), np.int32(op)))
    return d_o


def cs_zscore(x): return _cs(x, 0)
def cs_demean(x): return _cs(x, 1)
def cs_scale(x):  return _cs(x, 2)


def cs_rank(d_x):
    """逐 t 截面平均-rank ∈[0,1];n<2 整行 NaN。一 block 一 t,行入 shared,O(S²) count。
    (2026-06-29:block-bitonic O(S log²S) 替代经 crypto/A股/美股三市真实 S(299-666)实测慢 8-12×,
    真实列 tile 扫到 S=2048 仍慢 3.7×,crossover >S~3000 超出全部市场 universe——36-45 个 __syncthreads
    屏障 + 串行 tie 尾压过渐近优势;O(S²) 满并行(近零屏障)在本引擎全 S 区间即最优 → 证否删除。)"""
    T, S = d_x.shape
    d_o = cp.empty_like(d_x)
    _k_cs_rank((T,), (256,), (d_x, d_o, np.int32(S)), shared_mem=S * _ELEM)
    return d_o


# ===== metrics 融合核(逐 t 一 block,单遍 warp-shuffle 归约;metrics.py 调)=====
def k_per_t_ic(x, y):
    """逐 t Pearson IC → (T,) f64 常驻 cupy(替 metrics.py 多遍整盘 .sum 归约)。x,y 须 DTYPE 连续 (T,S)。"""
    T, S = x.shape
    o = cp.empty(T, dtype=cp.float64)
    _k_per_t_ic((T,), (256,), (x, y, o, np.int32(S)))
    return o


def k_per_t_pnl(values, y):
    """逐 t L1-norm signed PnL → (T,) f64 常驻 cupy(融合两轮归约核;values,y 须 DTYPE 连续 (T,S))。"""
    T, S = values.shape
    o = cp.empty(T, dtype=cp.float64)
    _k_per_t_pnl((T,), (256,), (values, y, o, np.int32(S)))
    return o


# ===== elementwise / binary / binary_const =====
# 算术/隐式-NaN 传播 op(abs/neg/square/tanh/add/sub/mul/max/min/add_const/mul_const)直接 cupy
# ufunc(本就单核)。**带显式 NaN where-mask 的 op(sign/log/sqrt/inv/s_log_1p/div/pow_const)**原
# 各是 3-7 个 cp.where/比较 ufunc 链 = 每 op 多次整盘全局读写 + 多张 (T,S) 临时盘;FACT 融合 →
# 各塌成**单个 cp.ElementwiseKernel**(读一遍、1 launch、0 中间临时):标量 body 内联 NaN 语义,
# 逐元素无 reduction → 不改累加序,落 GPU ~1e-6 容差(scripts/verify_fused 守门)。
# body 模板 T = 输入 dtype(f32/f64),log/sqrt/pow/log1p/fabs 自动选对应重载。

def abs_(x):    return cp.abs(x)
def neg(x):     return -x
def square(x):  return x * x
def tanh_(x):   return cp.tanh(x)
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def max_b(a, b): return cp.maximum(a, b)                              # nan 任一 → nan
def min_b(a, b): return cp.minimum(a, b)
def add_const(x, k): return x + DTYPE(k)
def mul_const(x, k): return x * DTYPE(k)


_k_sign     = cp.ElementwiseKernel('T x', 'T o',
    'o = isnan(x) ? x : (x > (T)0 ? (T)1 : (x < (T)0 ? (T)-1 : (T)0));', 'fm_sign')
_k_log      = cp.ElementwiseKernel('T x', 'T o',
    'o = (isnan(x) || x <= (T)0) ? (T)nan("") : log(x);', 'fm_log')          # x<=0/nan → nan
_k_sqrt     = cp.ElementwiseKernel('T x', 'T o',
    'o = (isnan(x) || x < (T)0) ? (T)nan("") : sqrt(x);', 'fm_sqrt')         # x<0/nan → nan
_k_inv      = cp.ElementwiseKernel('T x', 'T o',
    'o = (isnan(x) || x == (T)0) ? (T)nan("") : (T)1 / x;', 'fm_inv')        # x==0/nan → nan
_k_s_log_1p = cp.ElementwiseKernel('T x', 'T o',
    'o = isnan(x) ? x : (x >= (T)0 ? log1p(x) : -log1p(-x));', 'fm_s_log_1p')
_k_div      = cp.ElementwiseKernel('T a, T b', 'T o',
    'o = (b == (T)0) ? (T)nan("") : a / b;', 'fm_div')                       # b==0 → nan(a/b 自然传 nan)
_k_pow_const = cp.ElementwiseKernel('T x, T k', 'T o',
    'T av = fabs(x); if (av == (T)0) { o = (T)0; } else { T p = pow(av, k); o = (x < (T)0) ? -p : p; }',
    'fm_pow_const')                                                          # signed_power,0→0


def sign(x):     return _k_sign(x)
def log(x):      return _k_log(x)
def sqrt_(x):    return _k_sqrt(x)
def inv(x):      return _k_inv(x)
def s_log_1p(x): return _k_s_log_1p(x)
def div(a, b):   return _k_div(a, b)
def pow_const(x, k): return _k_pow_const(x, DTYPE(k))
