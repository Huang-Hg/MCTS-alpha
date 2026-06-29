"""DEAP 强类型 GP 因子挖掘 baseline(2026-06-22)。

目的:公式化 alpha 搜索 —— 评估侧全复用共享管线(同 panel/y、同 `evaluate_alpha` standalone 净
Sharpe reward + G0/G1/G2 门、同 `alpha_pool.try_new` net 感知准入、同 val/test pool_ret 口径),
搜索器 = DEAP NSGA-II 强类型 GP。

强类型(STGP):DEAP `PrimitiveSetTyped`,4 个类型 {Panel, WinT, AddC, MulC}——panel 与
窗口/两类常数分型,结构合法(杜绝 `cs_demean(0.5)`/`ts_min(3,w)`)。文法 SemKind 的更细反冗余
forbid 集 DEAP 单类型/槽表达不了,故 GP 探索空间略大于文法严格合法空间,由有效性门 + cost 预算剪。
primitive/terminal 一一镜像 DSL 算子(unary/cs/binary/binary_const/ts/pair
+ operands/WINDOWS/ADD/MULPOW consts)。

fitness = NSGA-II 双目标(均最大化):
  obj1 = 候选 standalone |rank IC|(= |evaluate_alpha['rank_ic']|;秩 IC 抗极端值过拟合)
  obj2 = 行为多样性 = 1 − max_{j≠i}|pnl_corr(per_t_pnl_i, per_t_pnl_j)|(种群内,C 算子 pnl_corr_vec)
无效树(cost 越界 / 门拒 / per_t_pnl=None)→ (_REWARD_FLOOR, 0) 被支配。

评估 device 由 `[evaluator] device` 派发:本地走 cpu;V100 走 cuda(因子常驻显存,单进程,
CUDA context 不跨 fork)。pool_obj.embed_fn=None → 无嵌入网络(G2/FailCache 走嵌入的支路跳过)。

桥:DEAP PrimitiveTree(prefix)→ Node.make → AlphaTree。ts/pair 的 window、binary_const 的 const
都是末位 terminal,按类型解析进父节点(const → const_add/const_mul 叶,window → Node.window attr)。
"""
from __future__ import annotations

import operator
import random
from inspect import isclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from deap import algorithms, base, creator, gp, tools

from config.config import ini
from backtest import ops
from evaluation.cache import EvalCache
from evaluation.ast import AlphaTree, Node
from evaluation.grammar import (
    ADD_CONSTANTS, MULPOW_CONSTANTS, WINDOWS,
    BinaryOp, ConstBinaryOp, CsOp, OperandToken, PairOp, TsOp, UnaryOp,
)
from rl.alpha_pool import AlphaPool
from rl.evaluator import EvalConfig, _REWARD_FLOOR, evaluate_alpha


# ============================================================================
# config([gp_baseline];cost 上限复用 [dsl])
# ============================================================================
_POP        = int(ini('gp_baseline', 'pop_size', 300))
_NGEN       = int(ini('gp_baseline', 'n_gen', 40))
_CXPB       = ini('gp_baseline', 'cxpb', 0.7)
_MUTPB      = ini('gp_baseline', 'mutpb', 0.2)
_INIT_MIN_H = int(ini('gp_baseline', 'init_min_height', 1))
_INIT_MAX_H = int(ini('gp_baseline', 'init_max_height', 4))
_MAX_H      = int(ini('gp_baseline', 'max_height', 8))
_MAX_COST   = int(ini('dsl', 'max_cost', 25))
_MIN_COST   = int(ini('dsl', 'min_cost', 3))


# ============================================================================
# 强类型 primitive set + DEAP→AlphaTree 桥映射
# ============================================================================
class Panel: pass      # 主因子类型((T,S) panel);所有算子输出 + operand 输入
class WinT:  pass      # 窗口整数(ts/pair 末槽)
class AddC:  pass      # add_const 常数槽(ADD_CONSTANTS)
class MulC:  pass      # mul_const / pow_const 常数槽(MULPOW_CONSTANTS)


def _cname(c: float) -> str:
    return ('n' if c < 0 else '') + str(abs(c)).replace('.', 'p')


def _build_pset() -> Tuple[gp.PrimitiveSetTyped, Dict[str, tuple], Dict[str, tuple]]:
    pset = gp.PrimitiveSetTyped("alpha", [], Panel)
    prim_map: Dict[str, tuple] = {}    # primitive .name → (kind, op)
    term_map: Dict[str, tuple] = {}    # terminal  .name → (tag, value)
    _id = lambda *a: None              # 占位 func:桥手工解析,gp.compile 不用

    for op in UnaryOp:
        nm = 'u_' + op.value
        pset.addPrimitive(_id, [Panel], Panel, name=nm); prim_map[nm] = ('unary', op)
    for op in CsOp:
        nm = 'c_' + op.value
        pset.addPrimitive(_id, [Panel], Panel, name=nm); prim_map[nm] = ('cs', op)
    for op in BinaryOp:
        nm = 'b_' + op.value
        pset.addPrimitive(_id, [Panel, Panel], Panel, name=nm); prim_map[nm] = ('binary', op)
    for op in ConstBinaryOp:
        ct = AddC if op == ConstBinaryOp.ADD_CONST else MulC
        nm = 'bc_' + op.value
        pset.addPrimitive(_id, [Panel, ct], Panel, name=nm); prim_map[nm] = ('binary_const', op)
    for op in TsOp:
        nm = 't_' + op.value
        pset.addPrimitive(_id, [Panel, WinT], Panel, name=nm); prim_map[nm] = ('ts', op)
    for op in PairOp:
        nm = 'p_' + op.value
        pset.addPrimitive(_id, [Panel, Panel, WinT], Panel, name=nm); prim_map[nm] = ('pair', op)

    for tok in OperandToken:
        nm = 'op_' + tok.name
        pset.addTerminal(tok, Panel, name=nm); term_map[nm] = ('operand', tok)
    for w in WINDOWS:
        nm = 'w%d' % w
        pset.addTerminal(w, WinT, name=nm); term_map[nm] = ('window', int(w))
    for c in ADD_CONSTANTS:
        nm = 'ac_' + _cname(c)
        pset.addTerminal(c, AddC, name=nm); term_map[nm] = ('const_add', float(c))
    for c in MULPOW_CONSTANTS:
        nm = 'mc_' + _cname(c)
        pset.addTerminal(c, MulC, name=nm); term_map[nm] = ('const_mul', float(c))
    return pset, prim_map, term_map


_PSET, _PRIM_MAP, _TERM_MAP = _build_pset()


# ---- typed 树生成(DEAP genFull/HalfAndHalf 对 terminal-only 类型 WinT/AddC/MulC 会崩:
#      非叶深度强求 primitive 而该类型无 primitive。这里复刻 DEAP generate + "无 primitive → 回退
#      terminal" 的标准修法,half-and-half(一半 full 一半 grow))----
def _generate(min_: int, max_: int, condition, type_=None) -> list:
    if type_ is None:
        type_ = _PSET.ret
    expr, height = [], random.randint(min_, max_)
    stack = [(0, type_)]
    while stack:
        depth, t = stack.pop()
        if condition(height, depth) or not _PSET.primitives[t]:    # 叶 / terminal-only 类型 → terminal
            term = random.choice(_PSET.terminals[t])
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            prim = random.choice(_PSET.primitives[t])
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def gen_expr(min_: int, max_: int, pset=None, type_=None) -> list:
    """half-and-half typed 生成;pset/type_ 由 mutUniform 以 kwargs 传入(此处 pset 固定 _PSET)。"""
    if random.random() < 0.5:
        cond = lambda h, d: d == h                                       # full
    else:
        cond = lambda h, d: d == h or (d >= min_ and random.random() < 0.35)   # grow
    return _generate(min_, max_, cond, type_)


def deap_to_alphatree(ind) -> AlphaTree:
    """DEAP PrimitiveTree(prefix 序列)→ AlphaTree。递归消费 prefix:panel 子树递归建 Node,
    ts/pair 末位 WinT terminal 收进 Node.window,binary_const 末位 const terminal 建 const_add/mul 叶。"""
    idx = [0]

    def build() -> Node:
        el = ind[idx[0]]; idx[0] += 1
        if isinstance(el, gp.Primitive):
            kind, op = _PRIM_MAP[el.name]
            if kind in ('unary', 'cs'):
                return Node.make(kind, op, None, (build(),))
            if kind == 'binary':
                l = build(); r = build()
                return Node.make('binary', op, None, (l, r))
            if kind == 'binary_const':
                l = build()
                cel = ind[idx[0]]; idx[0] += 1            # AddC/MulC terminal
                ckind, cval = _TERM_MAP[cel.name]         # 'const_add' | 'const_mul'
                const_leaf = Node.make(ckind, cval, None, ())
                return Node.make('binary_const', op, None, (l, const_leaf))
            if kind == 'ts':
                c = build()
                wel = ind[idx[0]]; idx[0] += 1            # WinT terminal
                return Node.make('ts', op, _TERM_MAP[wel.name][1], (c,))
            if kind == 'pair':
                l = build(); r = build()
                wel = ind[idx[0]]; idx[0] += 1
                return Node.make('pair', op, _TERM_MAP[wel.name][1], (l, r))
            raise ValueError(f'unknown primitive kind {kind}')
        # Terminal:此路径只可能是 operand(window/const 由父 primitive 直接消费)
        tag, val = _TERM_MAP[el.name]
        return Node.make('operand', val, None, ())

    return AlphaTree(build())


# ============================================================================
# DEAP creator(进程内一次)
# ============================================================================
creator.create("FitnessGP", base.Fitness, weights=(1.0, 1.0))   # 双目标均最大化
creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessGP)


# ============================================================================
# 评估:bridge → evaluate_alpha → (净 Sharpe, per_t_pnl);种群内算多样性 obj2
# ============================================================================
def _eval_and_assign(individuals, panels, y, eval_pool, cfg, cache, collected: Dict) -> int:
    """逐个体评估并写 fitness(obj1 |rank IC|, obj2 种群内行为多样性)。有效候选累进 collected
    (按 tree.hash 取最大 |rank IC|,供后续建池)。返回有效候选数。"""
    results: List[Optional[tuple]] = []
    for ind in individuals:
        tree = deap_to_alphatree(ind)
        cost = tree.total_cost()
        if not (_MIN_COST <= cost <= _MAX_COST):          # DSL cost 预算
            results.append(None); continue
        m = evaluate_alpha(tree, panels, y, eval_pool, cfg, cache, diagnostics=False)
        ptp, q = m['per_t_pnl'], abs(m['rank_ic'])        # reward = |rank IC|(唯一质量口径)
        if ptp is None or not np.isfinite(q):             # 门拒 / below-gate(per_t_pnl 仍供 obj2 + 准入 R²)
            results.append(None); continue
        rec = (tree, float(q), ptp, float(m['rank_ic']))   # rec[3]=带符号 rank_ic(= 池 q 来源 / append_run)
        results.append(rec)
        h = tree.hash
        if h not in collected or q > collected[h][1]:
            collected[h] = rec

    valid = [i for i, r in enumerate(results) if r is not None]
    M = np.ascontiguousarray(np.stack([results[i][2] for i in valid])) if valid else None
    for li, i in enumerate(valid):
        corr = np.abs(ops.pnl_corr_vec(np.ascontiguousarray(results[i][2]), M))
        corr[li] = 0.0                                    # 排除自相关(=1)
        nov = 1.0 - float(corr.max()) if M.shape[0] > 1 else 1.0
        individuals[i].fitness.values = (results[i][1], nov)
    for i, r in enumerate(results):
        if r is None:
            individuals[i].fitness.values = (_REWARD_FLOOR, 0.0)
    return len(valid)


def _build_pool(collected: Dict, capacity: int):
    """GP 收集的候选按 |rank IC| 降序过同一准入(try_new;Δ=|rank_ic|·(1−R²))建池。"""
    pool = AlphaPool(capacity=capacity, embed_fn=None)
    for tree, q, ptp, ric in sorted(collected.values(), key=lambda x: -x[1]):
        pool.try_new(tree, ric, ptp)
    return pool


# ============================================================================
# NSGA-II 主循环
# ============================================================================
def run_gp(panels, y, capacity: int, seed: int = 42,
           pop_size: Optional[int] = None, n_gen: Optional[int] = None, log=print):
    """跑 DEAP NSGA-II 强类型 GP,返回 (pool, collected)。pop_size 须被 4 整除(selTournamentDCD)。"""
    pop_size = pop_size or _POP
    n_gen = n_gen if n_gen is not None else _NGEN
    if pop_size % 4 != 0:
        pop_size += 4 - pop_size % 4
    random.seed(seed); np.random.seed(seed)

    tb = base.Toolbox()
    tb.register("expr", gen_expr, _INIT_MIN_H, _INIT_MAX_H)
    tb.register("individual", tools.initIterate, creator.IndividualGP, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gen_expr, 0, 2)                   # mutUniform 以 (pset=, type_=) kwargs 调
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=_PSET)
    tb.decorate("mate", gp.staticLimit(operator.attrgetter("height"), _MAX_H))
    tb.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), _MAX_H))
    tb.register("select", tools.selNSGA2)

    eval_pool = AlphaPool(capacity=capacity, embed_fn=None)   # 仅供 evaluate_alpha(G2 嵌入支路跳过)
    cfg = EvalConfig()
    cache = EvalCache(max_bytes=512 * 1024 ** 2)
    collected: Dict[int, tuple] = {}

    pop = tb.population(n=pop_size)
    nv = _eval_and_assign(pop, panels, y, eval_pool, cfg, cache, collected)
    pop = tb.select(pop, pop_size)                            # 赋 NSGA-II crowding
    log(f"[gp] gen00 valid={nv}/{pop_size} collected={len(collected)}")

    for gen in range(1, n_gen + 1):
        offspring = tools.selTournamentDCD(pop, pop_size)
        offspring = algorithms.varAnd(offspring, tb, _CXPB, _MUTPB)
        nv = _eval_and_assign(offspring, panels, y, eval_pool, cfg, cache, collected)
        pop = tb.select(pop + offspring, pop_size)
        best = max((c[1] for c in collected.values()), default=float('nan'))
        log(f"[gp] gen{gen:02d} valid={nv} collected={len(collected)} best_rank_ic={best:+.4f}")

    pool = _build_pool(collected, capacity)
    log(f"[gp] done: collected={len(collected)} pool_size={pool.size}/{capacity}")
    return pool, collected
