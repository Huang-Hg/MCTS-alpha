"""
AST 求值:把 AlphaTree 在 (T, S) 面板上算成一张同形 ndarray。**块化流式执行器**。

调用约定:
    evaluate(tree, panels, cache=None) -> np.ndarray (T, S) float64
        panels: dict[OperandToken -> ndarray (T, S)](f64 或 f32,f32 按块升格)
        cache:  EvalCache(hash → 整列 ndarray)或 None;命中直接返回

执行模型(空间复杂度优化,时间复杂度不变):
    - cs / 链根是**物化屏障**:整列 (T,S) f64 落地 + 入 cache。
    - 屏障之间的 colwise 链(operand/unary/binary/binary_const/ts/pair,全部逐列独立)
      按 _BLOCK_COLS 列块流水:每块自底向上递归,中间结果只活 (T,B),
      链根块直写整列输出(ops out=)。in-flight 从 O(链长)×(T,S) 降到 O(深度)×(T,B)。
    - cache 准入:链内点 miss 首次只计数(cache.bump),**二次请求才物化整列 + 入 cache**
      (虚拟叶);cache 命中的内点同样以虚拟叶接入,块内零拷贝切列。
    - 逐位一致性:per-column 计算顺序与全宽一致;_BLOCK_COLS 必须 4 的倍数,
      保证 ts AVX2 4 列分组与全宽调用相同(scalar 尾列也落在相同全局列上)。

注意:
    所有内部算子调用 backtest.ops.* 的 PYD 实现,**不写 numpy 的 fallback**。
    NaN 由算子层自动传播。operand 不进 cache(panels 视图,零 alloc)。
    _PROF 计时按 kind 跨块累计;_PROF_N kind 计数每节点每链一次(首块计)。
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from config.config import ini
from backtest import ops
from evaluation.ast import AlphaTree, Node
from evaluation.grammar import (
    BinaryOp, ConstBinaryOp, CsOp, OperandToken, PairOp, TsOp, UnaryOp,
)

# 求值后端:cpu=块化 C 执行器(本文件上半);cuda=全panel常驻 GPU 执行器(本文件下半,lazy)。
# config.ini[evaluator] device 单一来源,import 阶段固化。
_DEVICE = ini('evaluator', 'device', 'cpu')


# 模块级 phase profiler:eval_tree 内 cache 命中 + 各 kind 算子耗时。
# 调用方 reset_prof() / get_prof() 取数据。
_PROF: Dict[str, float] = {
    'cache_lookup': 0.0,  # 入口 hash → cache 查找
    'operand': 0.0,       # 叶子取 panels[op]
    'const': 0.0,         # np.full 标量广播
    'unary': 0.0,         # ops.abs/neg/sqrt/...
    'cs': 0.0,            # ops.cs_rank/cs_zscore/...
    'binary': 0.0,        # ops.add/sub/mul/...
    'binary_const': 0.0,  # _add_const / _mul_const / _pow_const
    'ts': 0.0,            # ops.ts_mean/ts_std/...(带 window)
    'pair': 0.0,          # ops.ts_corr / ts_cov(双 panel + window)
    'cache_store': 0.0,   # cache[hash] = result
}
_PROF_N: Dict[str, int] = {
    'cache_hit': 0,
    'cache_miss': 0,
    'operand': 0, 'const': 0, 'unary': 0, 'cs': 0,
    'binary': 0, 'binary_const': 0, 'ts': 0, 'pair': 0,
}


def reset_prof() -> None:
    for k in _PROF: _PROF[k] = 0.0
    for k in _PROF_N: _PROF_N[k] = 0


def get_prof() -> Dict[str, float]:
    return {**_PROF, **{f'n_{k}': v for k, v in _PROF_N.items()}}


def merge_prof(d: Dict[str, float]) -> None:
    """累加 worker 进程回传的 prof 增量(局级并行 merge,key 口径同 get_prof)。"""
    for k in _PROF: _PROF[k] += d[k]
    for k in _PROF_N: _PROF_N[k] += int(d[f'n_{k}'])


# ============================================================================
# 算子分发表
# ============================================================================

_UNARY_FN = {
    UnaryOp.ABS:      ops.abs_,
    UnaryOp.NEG:      ops.neg,
    UnaryOp.SIGN:     ops.sign,
    UnaryOp.LOG:      ops.log,
    UnaryOp.SQUARE:   ops.square,
    UnaryOp.SQRT:     ops.sqrt_,
    UnaryOp.TANH:     ops.tanh_,
    UnaryOp.INV:      ops.inv,
    UnaryOp.S_LOG_1P: ops.s_log_1p,
}

_BINARY_FN = {
    BinaryOp.ADD: ops.add,
    BinaryOp.SUB: ops.sub,
    BinaryOp.MUL: ops.mul,
    BinaryOp.DIV: ops.div,
    BinaryOp.MAX: ops.max_b,
    BinaryOp.MIN: ops.min_b,
}

# binary_const:panel ⊕ scalar k → ops C kernel(1 SIMD pass,替换 numpy 多 pass)。
# 旧 numpy 实现 _pow_const 在 (T,S)=193K×16 panel 上要走 6 个 panel pass(abs/power/sign/where/...)
# C kernel 单 pass,约 5-10x 提速。
_CONST_BIN_FN = {
    ConstBinaryOp.ADD_CONST: ops.add_const,
    ConstBinaryOp.MUL_CONST: ops.mul_const,
    ConstBinaryOp.POW_CONST: ops.pow_const,
}

_CS_FN = {
    CsOp.CS_RANK:   ops.cs_rank,
    CsOp.CS_ZSCORE: ops.cs_zscore,
    CsOp.CS_DEMEAN: ops.cs_demean,
    CsOp.CS_SCALE:  ops.cs_scale,
}

_TS_FN = {
    TsOp.TS_MEAN:    ops.ts_mean,
    TsOp.TS_STD:     ops.ts_std,
    TsOp.TS_MAX:     ops.ts_max,
    TsOp.TS_MIN:     ops.ts_min,
    TsOp.TS_SUM:     ops.ts_sum,
    TsOp.TS_RANK:    ops.ts_rank,
    TsOp.TS_ARG_MAX: ops.ts_arg_max,
    TsOp.TS_ARG_MIN: ops.ts_arg_min,
    TsOp.TS_EMA:     ops.ts_ema,
    TsOp.TS_WMA:     ops.ts_wma,
    TsOp.TS_REF:     ops.ts_ref,
    TsOp.TS_DELTA:   ops.ts_delta,
    TsOp.TS_SKEW:    ops.ts_skew,
    TsOp.TS_KURT:    ops.ts_kurt,
    TsOp.TS_MAD:     ops.ts_mad,
    TsOp.TS_SLOPE:   ops.ts_slope,
}

_PAIR_FN = {
    PairOp.TS_CORR: ops.ts_corr,
    PairOp.TS_COV:  ops.ts_cov,
}


# ============================================================================
# 求值 — 块化流式执行器
# ============================================================================

# 列块宽。必须 4 的倍数:ts 内核 AVX2 按 4 列一组分派,块起点/宽 4 对齐才能让
# 同一全局列在块化与全宽调用走同一条 SIMD/scalar 代码路径(fast-math 下 rounding
# 路径相关,这是逐位一致的前提)。64 列 × T=210k f64 ≈ 108MB/块缓冲。
_BLOCK_COLS = 64

_COLWISE = frozenset(('unary', 'binary', 'binary_const', 'ts', 'pair'))


class _BlockPool:
    """(T, width) f64 块缓冲 free-list:链式求值每节点每块复用,免反复 mmap+首触 page-fault。
    生命周期 = 一次链根物化;池内最多 ~链深 个缓冲。"""

    def __init__(self, T: int):
        self._T = T
        self._free: Dict[int, list] = {}

    def get(self, width: int) -> np.ndarray:
        lst = self._free.get(width)
        if lst:
            return lst.pop()
        return np.empty((self._T, width), dtype=np.float64)

    def put(self, arr: np.ndarray) -> None:
        self._free.setdefault(arr.shape[1], []).append(arr)


# Tier2b active-range 预算:每列 [vlo, vhi] = 所有 operand 的 first/last 有效行的**并集**
# (vlo=min first-valid, vhi=max last-valid)。证:t<vlo ⟹ 全 operand 在该 (t,s) NaN ⟹ 任意
# 由 operand 复合的中间节点也 NaN;t>vhi 同理。故此并集界对树中**每个**节点都是正确裁剪界
# (无需逐数组传播范围)。panel 整个 run 不变 → 按 id(panels) memoize(sentinel 身份校验防
# id 复用),算一次 O(operands·T·S),之后每次 evaluate 零成本复用。
_RANGE_CACHE: Dict[int, tuple] = {}


def _compute_ranges(panels: Dict[OperandToken, np.ndarray]):
    arrs = list(panels.values())
    T, S = arrs[0].shape
    vlo = np.full(S, T, dtype=np.int64)
    vhi = np.full(S, -1, dtype=np.int64)
    for a in arrs:
        fin = ~np.isnan(a)
        has = fin.any(axis=0)
        first = fin.argmax(axis=0)                 # 首个 True(无 True → 0,用 has 屏蔽)
        last = T - 1 - fin[::-1].argmax(axis=0)     # 末个 True
        np.minimum(vlo, np.where(has, first, T), out=vlo)
        np.maximum(vhi, np.where(has, last, -1), out=vhi)
    return np.ascontiguousarray(vlo), np.ascontiguousarray(vhi)


def _col_ranges(panels: Dict[OperandToken, np.ndarray]):
    key = id(panels)
    sentinel = next(iter(panels.values()))
    hit = _RANGE_CACHE.get(key)
    if hit is not None and hit[0] is sentinel:      # 身份校验:id 复用则 sentinel 不同 → 重算
        return hit[1], hit[2]
    vlo, vhi = _compute_ranges(panels)
    _RANGE_CACHE[key] = (sentinel, vlo, vhi)
    return vlo, vhi


def evaluate(
    tree: AlphaTree,
    panels: Dict[OperandToken, np.ndarray],
    cache=None,
) -> np.ndarray:
    """求值整树,返回 (T, S) ndarray。cache 须为 EvalCache(bump 准入)或 None。"""
    if _DEVICE == 'cuda':
        return _gpu_evaluate(tree, panels, cache)   # 同文件 GPU 执行器(lazy cupy/ops_cuda)
    return _eval_full(tree.root, panels, cache, _col_ranges(panels))


def _eval_full(n: Node, panels: Dict[OperandToken, np.ndarray], cache, ranges=None, force_store: bool = False) -> np.ndarray:
    """物化一个节点为整列 (T, S) 数组:cs / 链根 / 热内点走这里。
    入库同样走 bump≥2 准入(pack 有成本 ~60-90ms/条,一次性 root/cs 不值得占 LRU);
    force_store = 热内点(_prepare 已证复用,直接入)。"""
    _t = time.perf_counter()
    if cache is not None and n.hash in cache:
        _PROF['cache_lookup'] += time.perf_counter() - _t
        _PROF_N['cache_hit'] += 1
        return cache[n.hash]
    _PROF['cache_lookup'] += time.perf_counter() - _t
    _PROF_N['cache_miss'] += 1

    if n.kind == 'operand':
        # operand 直接返 panels[op] 引用,**不进 cache**:cache 用 nbytes 计 0.76 GB/sym,
        # 但实际是 panels_full view 共享内存 0 alloc。进 cache 会让 LRU 把 nbytes 算入预算
        # → 误评 → 错误驱逐真正占内存的 cs/ts 中间结果。直接返回比缓存更准确。
        _t = time.perf_counter()
        result = panels[n.op]
        _PROF['operand'] += time.perf_counter() - _t
        _PROF_N['operand'] += 1
        return result

    if n.kind in ('const_add', 'const_mul'):
        _t = time.perf_counter()
        ref = next(iter(panels.values()))
        result = np.full(ref.shape, float(n.op), dtype=np.float64, order='C')
        _PROF['const'] += time.perf_counter() - _t
        _PROF_N['const'] += 1
    elif n.kind == 'cs':
        # cs 屏障:截面语义需要整行 → 子树先整列物化(f32 operand 由 C 绑定升格)。
        x = _eval_full(n.children[0], panels, cache, ranges)
        _t = time.perf_counter()
        result = _CS_FN[n.op](x)
        _PROF['cs'] += time.perf_counter() - _t
        _PROF_N['cs'] += 1
    else:
        # colwise 链根:列块流水,块结果直写整列输出。
        ref = next(iter(panels.values()))
        T, S = ref.shape
        vleaves: Dict[int, np.ndarray] = {}
        if n.kind == 'binary_const':
            _prepare(n.children[0], panels, cache, vleaves, ranges)
        else:
            for c in n.children:
                _prepare(c, panels, cache, vleaves, ranges)
        result = np.empty((T, S), dtype=np.float64)
        pool = _BlockPool(T)
        rlo, rhi = (ranges if ranges is not None else (None, None))
        for j0 in range(0, S, _BLOCK_COLS):
            j1 = min(j0 + _BLOCK_COLS, S)
            blo = rlo[j0:j1] if rlo is not None else None   # 列块切片(j0/j1 块内恒定)
            bhi = rhi[j0:j1] if rhi is not None else None
            _eval_block(n, panels, vleaves, pool, j0, j1, result[:, j0:j1], blo, bhi)

    if cache is not None:
        _t = time.perf_counter()
        if force_store or cache.bump(n.hash) >= 2:
            cache[n.hash] = result
        _PROF['cache_store'] += time.perf_counter() - _t
    return result


def _prepare(n: Node, panels: Dict[OperandToken, np.ndarray], cache, vleaves: Dict[int, np.ndarray], ranges=None) -> None:
    """链根以下的 colwise 子树扫描:把必须整列存在的节点解析成虚拟叶。
    虚拟叶 = cs 子结果 | cache 命中的内点 | 二次使用(bump≥2)的热内点。
    其余内点留给 _eval_block 按块流式算,不物化。"""
    if n.kind == 'operand' or n.kind in ('const_add', 'const_mul'):
        return
    if n.kind == 'cs':
        vleaves[n.hash] = _eval_full(n, panels, cache, ranges)
        return
    if cache is not None:
        _t = time.perf_counter()
        hit = n.hash in cache
        _PROF['cache_lookup'] += time.perf_counter() - _t
        if hit:
            _PROF_N['cache_hit'] += 1
            vleaves[n.hash] = cache[n.hash]
            return
        _PROF_N['cache_miss'] += 1
        if cache.bump(n.hash) >= 2:
            vleaves[n.hash] = _eval_full(n, panels, cache, ranges, force_store=True)
            return
    if n.kind == 'binary_const':
        _prepare(n.children[0], panels, cache, vleaves, ranges)
    else:
        for c in n.children:
            _prepare(c, panels, cache, vleaves, ranges)


def _eval_block(
    n: Node,
    panels: Dict[OperandToken, np.ndarray],
    vleaves: Dict[int, np.ndarray],
    pool: _BlockPool,
    j0: int,
    j1: int,
    out: Optional[np.ndarray] = None,
    vlo=None,
    vhi=None,
):
    """递归求一个节点的列块 [j0:j1)。返回 (blk, owned):owned=True 表示 blk 来自 pool,
    由消费方用毕归还。out 非 None(仅链根)时内核直写该整列切片并返回 (out, False)。
    虚拟叶 / f64 operand 以 strided 视图直入 C 内核(绑定零拷贝)。
    vlo/vhi:本块列的 Tier2b active-range 切片(int64,长 j1-j0),透传给 ts/pair 内核;
    块内 j0/j1 恒定 → 整条递归共用同一切片。None → 内核扫描 fallback。"""
    first = j0 == 0
    if n.hash in vleaves:
        return vleaves[n.hash][:, j0:j1], False
    kind = n.kind
    if kind == 'operand':
        _t = time.perf_counter()
        src = panels[n.op][:, j0:j1]
        if src.dtype == np.float64:
            blk, owned = src, False
        else:
            blk = pool.get(j1 - j0)          # f32 → f64 块内升格(与 FROMANY 升格逐位一致)
            ops.upcast32(src, blk)           # OMP 并行;numpy 单线程 strided copyto ~2.5× 慢
            owned = True
        _PROF['operand'] += time.perf_counter() - _t
        if first: _PROF_N['operand'] += 1
        return blk, owned
    if kind in ('const_add', 'const_mul'):
        _t = time.perf_counter()
        blk = pool.get(j1 - j0)
        blk.fill(float(n.op))
        _PROF['const'] += time.perf_counter() - _t
        if first: _PROF_N['const'] += 1
        return blk, True
    if kind == 'unary':
        xb, xo = _eval_block(n.children[0], panels, vleaves, pool, j0, j1, vlo=vlo, vhi=vhi)
        _t = time.perf_counter()
        dest = out if out is not None else pool.get(j1 - j0)
        _UNARY_FN[n.op](xb, dest)
        _PROF['unary'] += time.perf_counter() - _t
        if first: _PROF_N['unary'] += 1
        if xo: pool.put(xb)
        return dest, out is None
    if kind == 'binary':
        ab, ao = _eval_block(n.children[0], panels, vleaves, pool, j0, j1, vlo=vlo, vhi=vhi)
        bb, bo = _eval_block(n.children[1], panels, vleaves, pool, j0, j1, vlo=vlo, vhi=vhi)
        _t = time.perf_counter()
        dest = out if out is not None else pool.get(j1 - j0)
        _BINARY_FN[n.op](ab, bb, dest)
        _PROF['binary'] += time.perf_counter() - _t
        if first: _PROF_N['binary'] += 1
        if ao: pool.put(ab)
        if bo: pool.put(bb)
        return dest, out is None
    if kind == 'binary_const':
        ab, ao = _eval_block(n.children[0], panels, vleaves, pool, j0, j1, vlo=vlo, vhi=vhi)
        _t = time.perf_counter()
        dest = out if out is not None else pool.get(j1 - j0)
        _CONST_BIN_FN[n.op](ab, float(n.children[1].op), dest)
        _PROF['binary_const'] += time.perf_counter() - _t
        if first: _PROF_N['binary_const'] += 1
        if ao: pool.put(ab)
        return dest, out is None
    if kind == 'ts':
        xb, xo = _eval_block(n.children[0], panels, vleaves, pool, j0, j1, vlo=vlo, vhi=vhi)
        _t = time.perf_counter()
        dest = out if out is not None else pool.get(j1 - j0)
        _TS_FN[n.op](xb, n.window, dest, vlo, vhi)
        _PROF['ts'] += time.perf_counter() - _t
        if first: _PROF_N['ts'] += 1
        if xo: pool.put(xb)
        return dest, out is None
    if kind == 'pair':
        ab, ao = _eval_block(n.children[0], panels, vleaves, pool, j0, j1, vlo=vlo, vhi=vhi)
        bb, bo = _eval_block(n.children[1], panels, vleaves, pool, j0, j1, vlo=vlo, vhi=vhi)
        _t = time.perf_counter()
        dest = out if out is not None else pool.get(j1 - j0)
        _PAIR_FN[n.op](ab, bb, n.window, dest, vlo, vhi)
        _PROF['pair'] += time.perf_counter() - _t
        if first: _PROF_N['pair'] += 1
        if ao: pool.put(ab)
        if bo: pool.put(bb)
        return dest, out is None
    raise ValueError(f'unknown kind {kind}')


# ============================================================================
# GPU 全panel常驻执行器(device=cuda 路径)—— 与上面 CPU 块化执行器正交。
# ============================================================================
# 执行模型(GPU 专用):
#   - panels 一次 H2D 常驻显存(按 id(panels) 缓存,整 run 复用 → 不每树重传 1.78GB)。
#   - 整树自底向上递归,中间结果全 (T,S) 常驻 cupy(显存富余:实测 1.78GB/32GB)。
#   - 算子全走 backtest.ops_cuda(语义对齐真 C,容差 ~1e-6;dtype=消费卡 f32/数据中心 f64)。
#   - 链根 D2H 回 numpy f64 → evaluator 下游零改;cache 入参兼容但 GPU 路不用(整树常驻够快)。
# **cupy / ops_cuda / GPU 分发表全部 lazy**:device=cpu 进程(含 live 无 cupy)import 本模块零触
# cupy;首次 cuda 求值时 _gpu_init() 构建一次。GPU/CPU FP 累加序不同 → 非 bitwise,两路各自自洽。
_GPU = None                                   # lazy GPU state(cupy/ops_cuda/分发表)
_GPANEL_CACHE: Dict[int, tuple] = {}          # id(panels) → (sentinel, {tok: cupy})
_GY_CACHE: Dict[int, tuple] = {}              # id(y_future) → (y_future, cupy)


class _GpuState:
    __slots__ = ('cp', 'gops', 'unary', 'binary', 'const_bin', 'cs', 'ts', 'pair')


def _gpu_init() -> '_GpuState':
    """首次 cuda 求值时构建 cupy/ops_cuda 句柄 + GPU 算子分发表(之后零成本复用)。"""
    global _GPU
    if _GPU is not None:
        return _GPU
    import cupy as cp
    from backtest import ops_cuda as gops
    st = _GpuState()
    st.cp = cp
    st.gops = gops
    st.unary = {
        UnaryOp.ABS: gops.abs_, UnaryOp.NEG: gops.neg, UnaryOp.SIGN: gops.sign,
        UnaryOp.LOG: gops.log, UnaryOp.SQUARE: gops.square, UnaryOp.SQRT: gops.sqrt_,
        UnaryOp.TANH: gops.tanh_, UnaryOp.INV: gops.inv, UnaryOp.S_LOG_1P: gops.s_log_1p,
    }
    st.binary = {
        BinaryOp.ADD: gops.add, BinaryOp.SUB: gops.sub, BinaryOp.MUL: gops.mul,
        BinaryOp.DIV: gops.div, BinaryOp.MAX: gops.max_b, BinaryOp.MIN: gops.min_b,
    }
    st.const_bin = {
        ConstBinaryOp.ADD_CONST: gops.add_const, ConstBinaryOp.MUL_CONST: gops.mul_const,
        ConstBinaryOp.POW_CONST: gops.pow_const,
    }
    st.cs = {
        CsOp.CS_RANK: gops.cs_rank, CsOp.CS_ZSCORE: gops.cs_zscore,
        CsOp.CS_DEMEAN: gops.cs_demean, CsOp.CS_SCALE: gops.cs_scale,
    }
    st.ts = {
        TsOp.TS_MEAN: gops.ts_mean, TsOp.TS_STD: gops.ts_std, TsOp.TS_MAX: gops.ts_max,
        TsOp.TS_MIN: gops.ts_min, TsOp.TS_SUM: gops.ts_sum, TsOp.TS_RANK: gops.ts_rank,
        TsOp.TS_ARG_MAX: gops.ts_arg_max, TsOp.TS_ARG_MIN: gops.ts_arg_min,
        TsOp.TS_EMA: gops.ts_ema, TsOp.TS_WMA: gops.ts_wma, TsOp.TS_REF: gops.ts_ref,
        TsOp.TS_DELTA: gops.ts_delta, TsOp.TS_SKEW: gops.ts_skew, TsOp.TS_KURT: gops.ts_kurt,
        TsOp.TS_MAD: gops.ts_mad, TsOp.TS_SLOPE: gops.ts_slope,
    }
    st.pair = {PairOp.TS_CORR: gops.ts_corr, PairOp.TS_COV: gops.ts_cov}
    _GPU = st
    return _GPU


def _gpanels(panels: Dict[OperandToken, np.ndarray]):
    """panels 一次 H2D 常驻,按 id 缓存(sentinel 身份校验防 id 复用),整 run 复用。"""
    g = _gpu_init()
    key = id(panels)
    sentinel = next(iter(panels.values()))
    hit = _GPANEL_CACHE.get(key)
    if hit is not None and hit[0] is sentinel:
        return hit[1]
    gp = {tok: g.cp.asarray(a.astype(g.gops.DTYPE, copy=False)) for tok, a in panels.items()}
    _GPANEL_CACHE[key] = (sentinel, gp)
    return gp


def _gpu_eval(n: Node, panels):
    g = _GPU
    k = n.kind
    if k == 'operand':
        return panels[n.op]
    if k in ('const_add', 'const_mul'):
        ref = next(iter(panels.values()))
        return g.cp.full(ref.shape, g.gops.DTYPE(n.op), dtype=g.gops.DTYPE)
    if k == 'unary':
        return g.unary[n.op](_gpu_eval(n.children[0], panels))
    if k == 'binary':
        return g.binary[n.op](_gpu_eval(n.children[0], panels), _gpu_eval(n.children[1], panels))
    if k == 'binary_const':
        return g.const_bin[n.op](_gpu_eval(n.children[0], panels), float(n.children[1].op))
    if k == 'cs':
        return g.cs[n.op](_gpu_eval(n.children[0], panels))
    if k == 'ts':
        return g.ts[n.op](_gpu_eval(n.children[0], panels), n.window)
    if k == 'pair':
        return g.pair[n.op](_gpu_eval(n.children[0], panels), _gpu_eval(n.children[1], panels), n.window)
    raise ValueError(f'unknown kind {k}')


def _gpu_evaluate(tree: AlphaTree, panels: Dict[OperandToken, np.ndarray], cache=None) -> np.ndarray:
    """整树 GPU 求值,链根 D2H 回 numpy f64(evaluate() 在 device=cuda 时调本函数)。"""
    gp = _gpanels(panels)
    res = _gpu_eval(tree.root, gp)
    return _GPU.cp.asnumpy(res).astype(np.float64, copy=False)


def evaluate_resident(tree: AlphaTree, panels: Dict[OperandToken, np.ndarray]):
    """返回**常驻 cupy** (T,S),不 D2H —— device=cuda 评分路径用(因子留显存,直接喂 GPU 度量,
    消灭每候选 1776MB D2H)。下游 ops_cuda.metrics 消费此 cupy handle。"""
    return _gpu_eval(tree.root, _gpanels(panels))


def gy(y_future: np.ndarray):
    """y_future H2D 缓存(整 run 不变 → 1 份常驻)。GPU 度量(ic/rank_ic/per_t_pnl/coverage)用。"""
    g = _gpu_init()
    key = id(y_future)
    hit = _GY_CACHE.get(key)
    if hit is not None and hit[0] is y_future:
        return hit[1]
    gv = g.cp.asarray(y_future.astype(g.gops.DTYPE, copy=False))
    _GY_CACHE[key] = (y_future, gv)
    return gv
