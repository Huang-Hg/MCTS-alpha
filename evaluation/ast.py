"""
Alpha 表达式 AST(由 grammar 产生式构造)
==========================================

节点形式:
    Leaf:     ('operand', OperandToken)  或  ('const_add'|'const_mul', float)
    Internal: ('unary'|'cs', op, child)
              ('binary',     op, child_l, child_r)
              ('ts',         op, w, child)
              ('pair',       op, w, child_l, child_r)

每个节点缓存 Merkle subtree-hash(子节点 hash 排序后并入 — commutative op
对子节点排序,等价表达 hash 相同)。

API:
    AlphaTree.from_action_seq(seq)         构建
    .hash                                  整树 64-bit fnv hash(int)
    .pretty()                              人类可读的 S-expression
    .total_cost()                          α-Sem-k 累计 cost
    .has_operand()                         是否含 ≥1 个 operand 叶
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from evaluation.grammar import (
    BinaryOp, ConstBinaryOp, CsOp, OperandToken, PairOp, Production, TsOp, UnaryOp,
    PRODUCTION_COSTS,
)


# ============================================================================
# Hash 工具 — FNV-1a 64-bit,稳定、便宜
# ============================================================================

_FNV_OFFSET = 0xcbf29ce484222325
_FNV_PRIME  = 0x100000001b3
_MASK64     = 0xffffffffffffffff


def _fnv_step(h: int, b: int) -> int:
    return (((h ^ b) * _FNV_PRIME) & _MASK64)


def _fnv_bytes(buf: bytes) -> int:
    h = _FNV_OFFSET
    for b in buf:
        h = _fnv_step(h, b)
    return h


def _hash_str(s: str) -> int:
    return _fnv_bytes(s.encode('utf-8'))


def _hash_combine(parts: Tuple[int, ...]) -> int:
    """混合一组 hash → 单一 hash。"""
    h = _FNV_OFFSET
    for p in parts:
        # 把 64-bit int 拆 8 字节
        for i in range(8):
            h = _fnv_step(h, (p >> (i * 8)) & 0xff)
    return h


# ============================================================================
# AST 节点
# ============================================================================

# 自然 commutative ops(子顺序不影响结果)
_COMMUTATIVE_BIN: frozenset = frozenset({BinaryOp.ADD, BinaryOp.MUL, BinaryOp.MAX, BinaryOp.MIN})
_COMMUTATIVE_PAIR: frozenset = frozenset({PairOp.TS_CORR, PairOp.TS_COV})


@dataclass(frozen=True)
class Node:
    """通用节点。kind+op+window 决定操作语义,children 是子树。"""
    kind: str                     # 'operand' | 'const_add' | 'const_mul' | 'unary' | 'cs' | 'binary' | 'binary_const' | 'ts' | 'pair'
    op:   object                  # OperandToken / float / UnaryOp / CsOp / BinaryOp / TsOp / PairOp
    window: Optional[int] = None  # 只在 ts/pair 用
    children: Tuple['Node', ...] = ()
    hash:  int = 0                # 自底向上 Merkle hash(post-init 计算)

    def __new__(cls, *a, **kw):
        return object.__new__(cls)

    @staticmethod
    def make(kind: str, op: object, window: Optional[int],
             children: Tuple['Node', ...]) -> 'Node':
        # commutative 子节点按 hash 排序,使等价表达 hash 一致
        if kind == 'binary' and op in _COMMUTATIVE_BIN and len(children) == 2:
            if children[0].hash > children[1].hash:
                children = (children[1], children[0])
        elif kind == 'pair' and op in _COMMUTATIVE_PAIR and len(children) == 2:
            if children[0].hash > children[1].hash:
                children = (children[1], children[0])

        # 计算 hash
        op_repr = op.value if hasattr(op, 'value') else (
            f'op{int(op)}' if isinstance(op, int) else str(op)
        )
        seed = _hash_str(f'{kind}|{op_repr}|{window}')
        if children:
            h = _hash_combine((seed,) + tuple(c.hash for c in children))
        else:
            h = seed
        # 直接 object.__setattr__ 绕过 frozen
        n = Node.__new__(Node)
        object.__setattr__(n, 'kind', kind)
        object.__setattr__(n, 'op', op)
        object.__setattr__(n, 'window', window)
        object.__setattr__(n, 'children', children)
        object.__setattr__(n, 'hash', h)
        return n


# operand 叶自身的 warmup 深度(bar 数):panel 构建端 rolling 列的窗长 + 派生 shift +
# metrics 列 lag-1(adapter / live merge 同构)。raw kline / premium / funding / tenure = 1。
# required_depth() 用:树最后一行非 NaN 需要的最小 panel 深度从叶往上累加。
_OPERAND_WARMUP_DEPTH = {
    OperandToken.SUM_OI:           2,    # metrics lag-1
    OperandToken.SUM_OI_VALUE:     2,
    OperandToken.COUNT_TOP_LSR:    2,
    OperandToken.SUM_TOP_LSR:      2,
    OperandToken.COUNT_LSR:        2,
    OperandToken.SUM_TAKER_LSR:    2,
    OperandToken.OI_LOG_RET:       3,    # shift(1) + lag-1
    OperandToken.OI_CHG_48:        51,   # rolling48(oi_log_ret) + lag-1
    OperandToken.OI_CHG_288:       291,
    OperandToken.VPIN_12:          14,
    OperandToken.KYLE_12:          14,
    OperandToken.VOL_ZSCORE_288:   290,
    OperandToken.TAKER_IMB_EMA_48: 50,
    OperandToken.MA_DIST_48:       49,
    OperandToken.MA_DIST_288:      289,
    # 2026-06-24 新增多 horizon 动量/波动(trailing N 5m bar + shift;同 ma_dist 约定)。
    # intra_*/body_pct/log_ret_oc/range_pct 为桶内 bar-local → 默认 warmup 1(不入表)。
    OperandToken.RET_12:           13,
    OperandToken.RET_48:           49,
    OperandToken.RET_144:          145,
    OperandToken.RET_288:          289,
    OperandToken.RV_12:            14,
    OperandToken.RV_48:            50,
    OperandToken.RV_144:           146,
    OperandToken.RV_288:           290,
}


# ============================================================================
# AlphaTree:整棵 alpha 表达式
# ============================================================================

@dataclass
class AlphaTree:
    root: Node

    # ---------- hash / 比较 ----------
    @property
    def hash(self) -> int:
        return self.root.hash

    def __hash__(self) -> int:
        return self.root.hash

    def __eq__(self, other) -> bool:
        return isinstance(other, AlphaTree) and other.root.hash == self.root.hash

    # ---------- 结构属性 ----------
    def total_cost(self) -> int:
        s = 0
        def walk(n: Node):
            nonlocal s
            s += PRODUCTION_COSTS[n.kind]
            for c in n.children:
                walk(c)
        walk(self.root)
        return s

    def has_operand(self) -> bool:
        found = False
        def walk(n: Node):
            nonlocal found
            if found: return
            if n.kind == 'operand': found = True; return
            for c in n.children:
                walk(c)
        walk(self.root)
        return found

    def is_raw_price_wrapper(self) -> bool:
        """True 当且仅当 tree 不含任何 ts_*/pair 时序 op,且所有 operand leaf 都是 OHLCV
        (open/high/low/close/volume/quote_volume)。
        这种 alpha 在 cs 截面上等价于 symbol identity 排序(BTC 永远高、DOGE 永远低),
        承载不了时序信息 — 训练阶段应作 score=0 处理,防 policy 退化到 leaf wrapper。"""
        _OHLCV = (OperandToken.OPEN, OperandToken.HIGH, OperandToken.LOW,
                  OperandToken.CLOSE,
                  OperandToken.VOLUME, OperandToken.QUOTE_VOLUME)
        bad = False
        def walk(n: Node):
            nonlocal bad
            if bad: return
            if n.kind in ('ts', 'pair'):
                bad = True; return
            if n.kind == 'operand' and n.op not in _OHLCV:
                bad = True; return
            for c in n.children:
                walk(c)
        walk(self.root)
        return not bad

    def required_depth(self) -> int:
        """最后一行非 warmup 所需最小 panel 行数:嵌套 ts/pair 沿路径累加 (w−1),
        operand 叶计自身派生窗(panel 构建端 rolling warmup + metrics lag-1)。
        live panel 深度有限(cold_load_days×288),required_depth 超深度的树 live 上
        每次决策都整树 NaN→哑火 → 选池硬拒的依据。"""
        def walk(n: Node) -> int:
            if n.kind == 'operand':
                return _OPERAND_WARMUP_DEPTH.get(n.op, 1)
            if n.kind in ('const_add', 'const_mul', 'window'):
                return 0
            d = max(walk(c) for c in n.children)
            if n.kind in ('ts', 'pair'):
                d += int(n.window) - 1
            return d
        return walk(self.root)

    # ---------- 序列化 ----------
    def pretty(self) -> str:
        return _pretty(self.root)

    def to_dict(self) -> dict:
        return _to_dict(self.root)

    @classmethod
    def from_dict(cls, d: dict) -> 'AlphaTree':
        return cls(_from_dict(d))


def _pretty(n: Node) -> str:
    if n.kind == 'operand':
        return n.op.name.lower()
    if n.kind in ('const_add', 'const_mul'):
        return f'{n.op:g}'
    if n.kind in ('unary', 'cs'):
        op = n.op.value
        return f'{op}({_pretty(n.children[0])})'
    if n.kind == 'binary':
        op = n.op.value
        return f'{op}({_pretty(n.children[0])}, {_pretty(n.children[1])})'
    if n.kind == 'binary_const':
        op = n.op.value
        return f'{op}({_pretty(n.children[0])}, {_pretty(n.children[1])})'
    if n.kind == 'ts':
        op = n.op.value
        return f'{op}({_pretty(n.children[0])}, {n.window})'
    if n.kind == 'pair':
        op = n.op.value
        return f'{op}({_pretty(n.children[0])}, {_pretty(n.children[1])}, {n.window})'
    raise ValueError(f'unknown kind {n.kind}')


def _to_dict(n: Node) -> dict:
    base = {'kind': n.kind}
    if n.kind == 'operand':
        base['op'] = n.op.name
    elif n.kind in ('const_add', 'const_mul'):
        base['op'] = float(n.op)
    else:
        base['op'] = n.op.value if hasattr(n.op, 'value') else str(n.op)
    if n.window is not None:
        base['w'] = n.window
    if n.children:
        base['c'] = [_to_dict(c) for c in n.children]
    return base


def _from_dict(d: dict) -> Node:
    kind = d['kind']
    op_raw = d['op']
    w = d.get('w')
    if kind == 'operand':
        op = OperandToken[op_raw]
    elif kind in ('const_add', 'const_mul'):
        op = float(op_raw)
    elif kind == 'unary':
        op = UnaryOp(op_raw)
    elif kind == 'cs':
        op = CsOp(op_raw)
    elif kind == 'binary':
        op = BinaryOp(op_raw)
    elif kind == 'binary_const':
        op = ConstBinaryOp(op_raw)
    elif kind == 'ts':
        op = TsOp(op_raw)
    elif kind == 'pair':
        op = PairOp(op_raw)
    else:
        raise ValueError(f'unknown kind {kind}')
    children = tuple(_from_dict(c) for c in d.get('c', []))
    return Node.make(kind, op, w, children)


# ============================================================================
# Production → Node 构造
# ============================================================================

def make_node(prod: Production, children: Tuple[Node, ...]) -> Node:
    """根据 production 和已构造好的子树拼一个新 Node。
    ts/pair:最后一个 child 是 window leaf(rhs_kind='window'),其 op = 整数窗口值。
    AST 把 window 收进 Node.window attr,不进 children — expression.py / pretty 等下游不变。
    const_add / const_mul:per-op 常数 leaf(grammar 拆分,policy 学不出 mul(*,1)=identity 的浪费)。
    """
    if prod.rhs_kind == 'operand':
        return Node.make('operand', prod.op, None, ())
    if prod.rhs_kind in ('const_add', 'const_mul'):
        return Node.make(prod.rhs_kind, prod.op, None, ())
    if prod.rhs_kind == 'window':
        return Node.make('window', prod.op, None, ())
    if prod.rhs_kind == 'unary':
        return Node.make('unary', prod.op, None, children)
    if prod.rhs_kind == 'cs':
        return Node.make('cs', prod.op, None, children)
    if prod.rhs_kind == 'binary':
        return Node.make('binary', prod.op, None, children)
    if prod.rhs_kind == 'binary_const':
        return Node.make('binary_const', prod.op, None, children)
    if prod.rhs_kind == 'ts':
        # children = (panel, window_leaf):剥 window
        window_val = int(children[-1].op)
        return Node.make('ts', prod.op, window_val, children[:-1])
    if prod.rhs_kind == 'pair':
        # children = (panel_a, panel_b, window_leaf):剥 window
        window_val = int(children[-1].op)
        return Node.make('pair', prod.op, window_val, children[:-1])
    raise ValueError(f'unknown production kind {prod.rhs_kind}')
