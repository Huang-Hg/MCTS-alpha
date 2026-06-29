"""
α-Sem-k typed CFG for formulaic alpha discovery(AlphaCFG-style)
================================================================

10 个 rhs_kind:operand / const_add / const_mul / window / unary / cs / binary / binary_const / ts / pair。
每个 production 自带:
    out_kind: SemKind             — 此节点的输出语义类型
    child_allowed: tuple[set[SemKind], ...] — 每个子槽接受的输入 SemKind 集合
    cost: int                     — α-Sem-k 长度成本(替代 token count)

α-Sem 类型系统:
    SemKind ∈ {RAW, RANKED, NORMALIZED, CENTERED, SCALED, INDEX, SIGNED, CORR, NUMERIC}
    PANEL_KINDS = ALL_KINDS \\ {NUMERIC}  — panel 类型集合
    根 slot 期望 ∈ PANEL_KINDS:整树根禁止 NUMERIC,即 const 不能当根。
    binary_const 第二槽期望 = {NUMERIC}:CONSTANTS 只能填这里,
    `cs_demean(0.5)` / `ts_min(3, w)` 等退化形式结构性禁止。
    cs 的 forbid 集合阻止冗余链:cs_rank(cs_rank(x)) / cs_zscore(cs_demean(x)) 等
    (ts_rank 已放开 RANKED 输入、tanh 已放开 NORMALIZED 输入:二阶时序分位 / 离群压缩非冗余)。

剩余结构约束(类型系统覆盖不到的):
    - ts/pair same-op same-window 嵌套禁(`ts_mean(ts_mean(x, w), w)` 退化),is_legal 检查。
    - 整树 ≥ 1 operand:has_operand=False 且即将完成最后叶子时,mask 掉 const-only 路径。
    - 树深度 / op 数由搜索器侧约束(gp_baseline 走 DEAP staticLimit height + cost 预算)。

cost 表见 PRODUCTION_COSTS;总 cost ≤ max_total_cost 由 ExpansionContext.remaining_cost 前瞻剪枝。

接口:
    PRODUCTIONS    — 全部 production 元组(NUM_PRODUCTIONS 条)
    is_legal(p, ctx) — 给定父槽 ctx,该 production 是否合法
    child_context(parent, child_idx, sibling_kinds, outer_ctx) — 构造子 slot 的 ctx
    initial_context(max_total_cost) — 根 slot 的 ctx
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, FrozenSet, List, Optional, Tuple


# ============================================================================
# 终结符:Operand / Const / Window
# ============================================================================

class OperandToken(IntEnum):
    """终端 operand:从 5m parquet 列名映射。
    分组:
      - 价格:OPEN/HIGH/LOW/CLOSE
      - 量:VOLUME/QUOTE_VOLUME/NUMBER_OF_TRADES/TAKER_IMBALANCE
      - OI:SUM_OI/SUM_OI_VALUE/OI_LOG_RET
      - LSR(币安特有 long-short ratio):COUNT_TOP_LSR/SUM_TOP_LSR/COUNT_LSR/SUM_TAKER_LSR
      - 真 1m trailing-1h 风险特征(per-5m-bar 1m building block 经 rolling-12 拼成 60-1m 真值;给 sizing 头):
        VPIN_12(toxicity)/ KYLE_12(价格冲击);均为 forward-vol 预测器(增量 fvol IC 两段符号稳)
      - 事件 / regime 物化特征(P0,数据层 build_5m_panel.py 已预算,grammar 直接消费,
        免去搜索器在深 ts 嵌套里重发现):
        VOL_ZSCORE_288 / TAKER_IMB_EMA_48 / OI_CHG_48 / OI_CHG_288 / MA_DIST_48 / MA_DIST_288
    IntEnum value 留 hole 不复用(已删 operand 的槽位)。
    """
    OPEN                  = 0
    HIGH                  = 1
    LOW                   = 2
    CLOSE                 = 3
    VOLUME                = 4
    QUOTE_VOLUME          = 5
    # 6,7,8 留空 — 已删 operand 槽位(不复用)
    NUMBER_OF_TRADES      = 9
    TAKER_BUY_QUOTE       = 10   # 留作 deployed bundle 兼容(taker_imbalance raw 版,affine 等价)
    TAKER_IMBALANCE       = 11
    SUM_OI                = 12
    OI_LOG_RET            = 13
    SUM_OI_VALUE          = 14   # = SUM_OI × close 隐含 raw price,bt max_concentration 0.20 约束 mono 影响
    # 15,16 留空 — 已删 operand 槽位(不复用)
    COUNT_TOP_LSR         = 17
    SUM_TOP_LSR           = 18
    COUNT_LSR             = 19
    SUM_TAKER_LSR         = 20
    # 21-25 留空 — 已删 operand 槽位(不复用)
    # P0 事件 / regime 物化特征
    VOL_ZSCORE_288        = 26   # 1d 成交量 z-score(成交量爆发 regime)
    TAKER_IMB_EMA_48      = 27   # 4h 主动买卖盘 EMA(主动单流 regime)
    OI_CHG_48             = 28   # 4h OI 变化率(短期杠杆建立)
    OI_CHG_288            = 29   # 1d OI 变化率(中期杠杆累积)
    MA_DIST_48            = 30   # 4h MA 距离 (close-mean)/std(短期 breakout)
    MA_DIST_288           = 31   # 1d MA 距离(中期 breakout / mean-reversion)
    # P0+ 永续独有 funding/basis 特征(enrich_5m_panel.py 物化)
    PREMIUM_INDEX_5M      = 32   # (mark − index)/index,实时 funding 压力
    FUNDING_RATE_INTERP   = 33   # 当前 funding cycle carry rate(4h or 8h discrete)
    FUNDING_COUNTDOWN_NORM = 34  # funding 周期内相对位置 ∈ [0, 1]
    # 35-38 留空 — 已删 operand 槽位(不复用)
    # 派生 operand(非 parquet 列,adapter.load_panel 合成注入;不进 OPERAND_COLUMNS)
    # 真 1m trailing-1h(W=12 5m bar)风险特征(给 sizing 头)
    VPIN_12               = 40   # Σ|1m签名流| / Σqv,order-flow toxicity → forward vol
    KYLE_12               = 41   # Σ|1m收益| / Σ|1m签名流|,价格冲击/illiq → forward vol
    TENURE_NORM           = 39   # recency = exp(−tenure/288),tenure=连续 qvol>0 bar 数;
                                 #   把"刚进 top-N"显式成可挖矿特征
    # === 2026-06-24 新增:数据里已有但此前未喂搜索的特征(抬 IC 上限;42-55)===
    # 桶内 1m 微结构(5m OHLCV 无法重建 → 纯增信息)
    INTRA_RV              = 42   # 桶内 1m 已实现波动率
    INTRA_SUM_ABS_SF      = 43   # 桶内 Σ|1m 签名流|(order-flow)
    INTRA_SUM_ABS_RET     = 44   # 桶内 Σ|1m 收益|
    # K 线体型(桶内形态)
    BODY_PCT              = 45   # 实体占比 |close−open|/range
    LOG_RET_OC            = 46   # log(close/open) 开→收内收益
    RANGE_PCT             = 47   # 振幅 (high−low)/·
    # 多 horizon 动量(12/48/144/288 ×5m = 1h/4h/12h/1d;len≤6 建不出深 ts 链 → 预算金子)
    RET_12                = 48
    RET_48                = 49
    RET_144               = 50
    RET_288               = 51
    # 多 horizon 价格已实现波动(≠ vol_zscore 的成交量 z)
    RV_12                 = 52
    RV_48                 = 53
    RV_144                = 54
    RV_288                = 55


# 列名映射(必须存在于 parquet_5m 输出 schema 中)
# 注:TENURE_NORM 不在此 — 它是 adapter 派生注入的 operand,无对应 parquet 列。
OPERAND_COLUMNS: Dict[OperandToken, str] = {
    OperandToken.OPEN:                 'open',
    OperandToken.HIGH:                 'high',
    OperandToken.LOW:                  'low',
    OperandToken.CLOSE:                'close',
    OperandToken.VOLUME:               'volume',
    OperandToken.QUOTE_VOLUME:         'quote_volume',
    OperandToken.NUMBER_OF_TRADES:     'number_of_trades',
    OperandToken.TAKER_BUY_QUOTE:      'taker_buy_quote_volume',
    OperandToken.TAKER_IMBALANCE:      'taker_imbalance',
    OperandToken.SUM_OI:               'sum_oi',
    OperandToken.OI_LOG_RET:           'oi_log_ret',
    OperandToken.SUM_OI_VALUE:         'sum_oi_value',
    OperandToken.COUNT_TOP_LSR:        'count_top_lsr',
    OperandToken.SUM_TOP_LSR:          'sum_top_lsr',
    OperandToken.COUNT_LSR:            'count_lsr',
    OperandToken.SUM_TAKER_LSR:        'sum_taker_lsr',
    OperandToken.VPIN_12:              'vpin_12',
    OperandToken.KYLE_12:              'kyle_12',
    # P0 事件 / regime 物化
    OperandToken.VOL_ZSCORE_288:       'vol_zscore_288',
    OperandToken.TAKER_IMB_EMA_48:     'taker_imb_ema_48',
    OperandToken.OI_CHG_48:            'oi_chg_48',
    OperandToken.OI_CHG_288:           'oi_chg_288',
    OperandToken.MA_DIST_48:           'ma_dist_48',
    OperandToken.MA_DIST_288:          'ma_dist_288',
    # P0+ 永续独有 funding/basis(enrich_5m_panel.py 加列)
    OperandToken.PREMIUM_INDEX_5M:      'premium_index_5m',
    OperandToken.FUNDING_RATE_INTERP:   'funding_rate_interp',
    OperandToken.FUNDING_COUNTDOWN_NORM:'funding_countdown_norm',
    # 2026-06-24 新增:1m 微结构 + K线体型 + 多 horizon 动量/波动
    OperandToken.INTRA_RV:              'intra_rv',
    OperandToken.INTRA_SUM_ABS_SF:      'intra_sum_abs_sf',
    OperandToken.INTRA_SUM_ABS_RET:     'intra_sum_abs_ret',
    OperandToken.BODY_PCT:              'body_pct',
    OperandToken.LOG_RET_OC:            'log_ret_oc',
    OperandToken.RANGE_PCT:             'range_pct',
    OperandToken.RET_12:                'ret_12',
    OperandToken.RET_48:                'ret_48',
    OperandToken.RET_144:               'ret_144',
    OperandToken.RET_288:               'ret_288',
    OperandToken.RV_12:                 'rv_12',
    OperandToken.RV_48:                 'rv_48',
    OperandToken.RV_144:                'rv_144',
    OperandToken.RV_288:                'rv_288',
}


# 常数池 — 按算子类型分两组(避免 mul/pow 的 ±1 = identity 浪费搜索空间):
#   ADD_CONSTANTS    = {±0.5, ±1, ±2}      shift 用得着 ±1
#   MULPOW_CONSTANTS = {±0.5, ±2}          mul_const(*,±1) = identity/NEG;pow_const(*,±1) = identity/inv,
#                                          均与 UnaryOp 重复 → 删
ADD_CONSTANTS:    Tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
MULPOW_CONSTANTS: Tuple[float, ...] = (-2.0, -0.5, 0.5, 2.0)

# 窗口(**1h bars**):1h / 2h / 4h / 8h / 16h / 1d / 2d / 4d / 7d / 14d
# alpha operand 面板 1h(load 时尾部小时聚合,adapter.aggregate_5m_to_1h),ts/pair 窗口为 1h 单位。
# max=336(14d)≤ live cold_load 21d×24=504 可部署。
WINDOWS: Tuple[int, ...] = (1, 2, 4, 8, 16, 24, 48, 96, 168, 336)


# ============================================================================
# 算子枚举
# ============================================================================

class UnaryOp(Enum):
    ABS      = 'abs'
    NEG      = 'neg'
    SIGN     = 'sign'
    LOG      = 'log'
    SQUARE   = 'square'
    SQRT     = 'sqrt'
    TANH     = 'tanh'
    INV      = 'inv'
    S_LOG_1P = 's_log_1p'


class BinaryOp(Enum):
    # 删 GT/LT:返回 0/1 离散值,cs_zscore + tanh 后变成 ±tanh(z) 分类器,
    # 偶然在 in-sample 拿到极高 IC 但经济意义可疑,易过拟合。
    ADD = 'add'
    SUB = 'sub'
    MUL = 'mul'
    DIV = 'div'
    MAX = 'max'
    MIN = 'min'


class CsOp(Enum):
    CS_RANK   = 'cs_rank'
    CS_ZSCORE = 'cs_zscore'
    CS_DEMEAN = 'cs_demean'
    CS_SCALE  = 'cs_scale'      # WQ101 高频:x / sum_s |x|,L1 横截面归一化


class TsOp(Enum):
    TS_MEAN    = 'ts_mean'
    TS_STD     = 'ts_std'
    TS_MAX     = 'ts_max'
    TS_MIN     = 'ts_min'
    TS_SUM     = 'ts_sum'
    TS_RANK    = 'ts_rank'
    TS_ARG_MAX = 'ts_arg_max'
    TS_ARG_MIN = 'ts_arg_min'
    TS_EMA     = 'ts_ema'
    TS_WMA     = 'ts_wma'
    TS_REF     = 'ts_ref'
    TS_DELTA   = 'ts_delta'
    TS_SKEW    = 'ts_skew'      # 滚动 3 阶中心矩
    TS_KURT    = 'ts_kurt'      # 滚动 4 阶中心矩
    TS_MAD     = 'ts_mad'       # 滚动 mean absolute deviation(robust 替代 ts_std)
    TS_SLOPE   = 'ts_slope'     # 滚动 OLS 斜率


class PairOp(Enum):
    TS_CORR = 'ts_corr'
    TS_COV  = 'ts_cov'


class ConstBinaryOp(Enum):
    """常数双目算子:第一槽 panel,第二槽 NUMERIC 常数。AlphaCFG-style typed slot。"""
    ADD_CONST = 'add_const'  # panel + k     (小偏移)
    MUL_CONST = 'mul_const'  # panel * k     (缩放)
    POW_CONST = 'pow_const'  # sign(panel)·|panel|^k(signed_power,WQ101 #1/#3/#4 等用)


# ============================================================================
# α-Sem 语义类型(typed grammar)
# ============================================================================
# 每个 production 输出某个 SemKind,每个子槽接受一个 SemKind 集合。
# 沿树由根向下传播。子产生式的 out_kind 不在父槽的 allowed 集合里 → 立即 mask。
# 替代了原来 4 条「即时父」检查(inside_unary_to_const / cs+unary same-op / etc.),
# 同时把语义信息显式化。

class SemKind(IntEnum):
    RAW            = 0   # operand 或一般运算后的连续 panel
    RANKED         = 1   # cs_rank / ts_rank → 值域 [0,1]
    NORMALIZED     = 2   # cs_zscore → 均值 0 std 1
    CENTERED       = 3   # cs_demean → 均值 0
    SCALED         = 4   # cs_scale → Σ_s|x|=1
    INDEX          = 5   # ts_arg_max / ts_arg_min → 整数 ∈ [0,w-1]
    SIGNED         = 6   # sign → {-1, 0, +1}
    CORR           = 7   # ts_corr → [-1,1]
    NUMERIC_ADD    = 8   # ADD_CONST 第二槽:{±0.5, ±1, ±2}
    NUMERIC_MULPOW = 9   # MUL_CONST/POW_CONST 第二槽:{±0.5, ±2}(去 ±1 防 identity / NEG 冗余)
    WINDOW_VAL     = 10  # 窗口整数(ts/pair 最后一槽)
    SHIFTED        = 11  # add_const(x,k):per-bar 平移像。cs_rank/zscore/demean 对平移不变 → 作其子冗余
    RESCALED       = 12  # mul_const(x,k):per-bar 缩放像。cs_rank/zscore/scale 对缩放不变(±)→ 作其子冗余
    NEGATED        = 13  # neg(x) = 已删的 mul_const(x,-1)。所有 cs 对其 ±不变;neg/abs/square∘neg 退化 → 作其子冗余


ALL_KINDS:   FrozenSet[SemKind] = frozenset(SemKind)
# panel 类型集合 = 排除非 panel 叶类型(三种 NUMERIC*/WINDOW_VAL)。
PANEL_KINDS: FrozenSet[SemKind] = ALL_KINDS - {SemKind.NUMERIC_ADD, SemKind.NUMERIC_MULPOW, SemKind.WINDOW_VAL}


# ============================================================================
# Production 定义
# ============================================================================

class NonTerminal(IntEnum):
    EXPR   = 0  # 主 NT,语义类型由 Production.out_kind / child_allowed 表达
    WINDOW = 1  # 窗口 leaf 子槽(ts/pair 最后一槽);只接 rhs_kind='window' 的 8 个 production


@dataclass(frozen=True)
class Production:
    """文法产生式 + α-Sem 类型标记。

    rhs_kind: 节点类别(用于 ast / encoder 派发)
    op:       具体算子(OperandToken / UnaryOp / BinaryOp / CsOp / TsOp / PairOp)
    extra:    ts/pair 的窗口 w(int)
    children_nts: 子 NT 序列(占位用,数量决定 children 数)
    cost:     production 自身 cost
    out_kind: 此 production 的输出语义类型(α-Sem)
    child_allowed: 每个子槽允许的输入 SemKind 集合(典型地为 ALL_KINDS \\ {redundant})
    """
    rhs_kind: str
    op: object
    extra: object = None
    children_nts: Tuple[NonTerminal, ...] = ()
    cost: int = 1
    out_kind: SemKind = SemKind.RAW
    child_allowed: Tuple[FrozenSet[SemKind], ...] = ()


PRODUCTION_COSTS: Dict[str, int] = {
    'operand':      1,
    'const_add':    1,    # NUMERIC_ADD 标量,填 add_const 第二槽
    'const_mul':    1,    # NUMERIC_MULPOW 标量,填 mul/pow_const 第二槽
    'unary':        2,
    'cs':           2,
    'binary':       3,
    'binary_const': 2,    # panel ⊕ const,比 binary(panel ⊕ panel)便宜:无需第二个 panel 子树
    'ts':           3,
    'pair':         5,
    # 'window' 不在表里 — window leaf 在 make_node 时被 strip 进父 ts/pair 的 Node.window
    # attr,不进 AST.total_cost 遍历;production cost inline 写 0(纯参数,免成本预算)。
}


# ============================================================================
# α-Sem 类型表 — 决定每个算子的 out_kind 和 child forbid
# ============================================================================
# 「forbid 集合」记录哪些 SemKind 不可作此 child slot 输入;allowed = ALL_KINDS \\ forbid。

# unary 算子分两类:
#   - 「重置回 RAW」类(neg/abs/square/tanh/s_log_1p/inv):输入随便,输出 RAW
#   - 「保 sign 标记」类(sign):输入除 SIGNED/INDEX(冗余)外都可,输出 SIGNED
#   - 「值域受限」类(log/sqrt):输入不能是 INDEX/SIGNED/CENTERED/NORMALIZED(可能 ≤0),输出 RAW
# NEGATED 加入 neg/abs/square 的 forbid:neg∘neg=id、abs∘neg=abs、square∘neg=square 全退化。
_UNARY_FORBID: Dict[UnaryOp, FrozenSet[SemKind]] = {
    UnaryOp.ABS:      frozenset({SemKind.SIGNED, SemKind.INDEX, SemKind.RANKED, SemKind.NEGATED}),
    UnaryOp.NEG:      frozenset({SemKind.SIGNED, SemKind.NEGATED}),
    UnaryOp.SIGN:     frozenset({SemKind.SIGNED, SemKind.INDEX}),
    UnaryOp.LOG:      frozenset({SemKind.INDEX, SemKind.SIGNED, SemKind.CENTERED, SemKind.NORMALIZED, SemKind.CORR}),
    UnaryOp.SQUARE:   frozenset({SemKind.NEGATED}),
    UnaryOp.SQRT:     frozenset({SemKind.INDEX, SemKind.SIGNED, SemKind.CENTERED, SemKind.NORMALIZED, SemKind.CORR}),
    UnaryOp.TANH:     frozenset({SemKind.RANKED}),  # tanh(rank∈[0,1]) 近线性≈恒等仍剪;NORMALIZED 放开:tanh(zscore) 是有效离群压缩(zscore 尾部无界、非饱和)
    UnaryOp.INV:      frozenset({SemKind.INDEX, SemKind.SIGNED, SemKind.CENTERED}),
    UnaryOp.S_LOG_1P: frozenset(),
}
_UNARY_OUT: Dict[UnaryOp, SemKind] = {
    UnaryOp.SIGN: SemKind.SIGNED,
    UnaryOp.NEG:  SemKind.NEGATED,
    # 其余都 RAW
}

# binary:输出 RAW;allowed 不限(子可任意类型;redundant case 不严格了)
# cs:每个 cs 输出特定 kind;child 不能再次产生同 kind
_CS_OUT: Dict[CsOp, SemKind] = {
    CsOp.CS_RANK:   SemKind.RANKED,
    CsOp.CS_ZSCORE: SemKind.NORMALIZED,
    CsOp.CS_DEMEAN: SemKind.CENTERED,
    CsOp.CS_SCALE:  SemKind.SCALED,
}
# forbid = 该 cs op 的「不变群」:对它不变的 per-bar 变换之子像一律冗余,生成时剪。
#   rank   ← 任意严格单调像:demean/zscore/scale(cs∘cs) + add/mul_const(cs∘const)
#   zscore ← 任意仿射像:CENTERED/SCALED/SHIFTED/RESCALED(非线性单调 tanh/s_log_1p 改 z 分布,不剪)
#   demean ← 仅平移像:CENTERED/SHIFTED(NORMALIZED 已 mean0 → demean 透明)
#   scale  ← 仅缩放像:SCALED/RESCALED
# add_const→SHIFTED、mul_const→RESCALED 让「常数仿射包裹」与「cs 归一化包裹」共用同一 forbid 机制。
# NEGATED(=mul_const(·,-1))对所有 cs 都 ±不变(rank→反序、zscore/demean/scale→取负;池吸收符号)→ 四者皆 forbid。
_CS_FORBID: Dict[CsOp, FrozenSet[SemKind]] = {
    CsOp.CS_RANK:   frozenset({SemKind.RANKED, SemKind.NORMALIZED, SemKind.CENTERED,
                               SemKind.SCALED, SemKind.SHIFTED, SemKind.RESCALED, SemKind.NEGATED}),
    CsOp.CS_ZSCORE: frozenset({SemKind.NORMALIZED, SemKind.RANKED, SemKind.CENTERED,
                               SemKind.SCALED, SemKind.SHIFTED, SemKind.RESCALED, SemKind.NEGATED}),
    CsOp.CS_DEMEAN: frozenset({SemKind.CENTERED, SemKind.NORMALIZED, SemKind.RANKED,
                               SemKind.SHIFTED, SemKind.NEGATED}),
    CsOp.CS_SCALE:  frozenset({SemKind.SCALED, SemKind.RANKED, SemKind.NORMALIZED,
                               SemKind.RESCALED, SemKind.NEGATED}),
}

# ts:绝大多数 ts 算子输出 RAW;rank/arg 例外
_TS_OUT: Dict[TsOp, SemKind] = {
    TsOp.TS_RANK:    SemKind.RANKED,
    TsOp.TS_ARG_MAX: SemKind.INDEX,
    TsOp.TS_ARG_MIN: SemKind.INDEX,
}
_TS_FORBID: Dict[TsOp, FrozenSet[SemKind]] = {
    TsOp.TS_RANK:    frozenset(),                  # RANKED 放开:ts_rank(cs_rank(x)) 是二阶时序分位、非冗余(ts_rank 对已 ranked 输入不不变)
    TsOp.TS_ARG_MAX: frozenset({SemKind.INDEX, SemKind.SIGNED}),
    TsOp.TS_ARG_MIN: frozenset({SemKind.INDEX, SemKind.SIGNED}),
}

# pair:ts_corr 输出 CORR,ts_cov RAW;子无类型限制
_PAIR_OUT: Dict[PairOp, SemKind] = {
    PairOp.TS_CORR: SemKind.CORR,
    PairOp.TS_COV:  SemKind.RAW,
}


def _panel_minus(forbid: FrozenSet[SemKind]) -> FrozenSet[SemKind]:
    """从 panel 类型集中扣掉 forbid。non-const 子槽默认 allowed。"""
    return PANEL_KINDS - forbid


def all_productions() -> Tuple[Production, ...]:
    """枚举所有合法 productions,带 α-Sem 类型标记。"""
    out: List[Production] = []

    # operand 叶 → out=RAW
    for tok in OperandToken:
        out.append(Production('operand', tok, None, (), PRODUCTION_COSTS['operand'],
                              out_kind=SemKind.RAW, child_allowed=()))

    # const 叶按 op 分组(per-op grid,#2):
    #   const_add → NUMERIC_ADD,只填 ADD_CONST 第二槽
    #   const_mul → NUMERIC_MULPOW,只填 MUL_CONST/POW_CONST 第二槽
    for c in ADD_CONSTANTS:
        out.append(Production('const_add', c, None, (), PRODUCTION_COSTS['const_add'],
                              out_kind=SemKind.NUMERIC_ADD, child_allowed=()))
    for c in MULPOW_CONSTANTS:
        out.append(Production('const_mul', c, None, (), PRODUCTION_COSTS['const_mul'],
                              out_kind=SemKind.NUMERIC_MULPOW, child_allowed=()))

    # window 叶 → out=WINDOW_VAL,**只能**填 ts/pair 最后一槽
    # 跟 const 同类设计但隔离类型,防止填错槽位。cost=0(纯参数,不占预算)。
    for w in WINDOWS:
        out.append(Production('window', w, None, (), 0,
                              out_kind=SemKind.WINDOW_VAL, child_allowed=()))

    # unary(子槽只接 panel,默认 PANEL_KINDS 减 forbid)
    for op in UnaryOp:
        forbid = _UNARY_FORBID.get(op, frozenset())
        out_k = _UNARY_OUT.get(op, SemKind.RAW)
        out.append(Production('unary', op, None, (NonTerminal.EXPR,), PRODUCTION_COSTS['unary'],
                              out_kind=out_k, child_allowed=(_panel_minus(forbid),)))

    # cs
    for op in CsOp:
        forbid = _CS_FORBID[op]
        out.append(Production('cs', op, None, (NonTerminal.EXPR,), PRODUCTION_COSTS['cs'],
                              out_kind=_CS_OUT[op],
                              child_allowed=(_panel_minus(forbid),)))

    # binary(panel + panel,两槽都只接 panel,不接 NUMERIC)
    for op in BinaryOp:
        out.append(Production('binary', op, None, (NonTerminal.EXPR, NonTerminal.EXPR), PRODUCTION_COSTS['binary'],
                              out_kind=SemKind.RAW,
                              child_allowed=(PANEL_KINDS, PANEL_KINDS)))

    # binary_const(panel + 常数,第二槽按 op 接 NUMERIC_ADD 或 NUMERIC_MULPOW)
    # ADD_CONST:add_const(panel, k),k ∈ {±0.5, ±1, ±2}
    # MUL_CONST/POW_CONST:k ∈ {±0.5, ±2}(±1 跟 NEG/inv 重复 → 已删)
    only_add: FrozenSet[SemKind] = frozenset({SemKind.NUMERIC_ADD})
    only_mul: FrozenSet[SemKind] = frozenset({SemKind.NUMERIC_MULPOW})
    # add_const→SHIFTED / mul_const→RESCALED:把仿射包裹纳入 cs forbid;pow_const 非线性 → RAW。
    _constbin_out = {ConstBinaryOp.ADD_CONST: SemKind.SHIFTED,
                     ConstBinaryOp.MUL_CONST: SemKind.RESCALED,
                     ConstBinaryOp.POW_CONST: SemKind.RAW}
    # 第一槽 forbid 自身输出 kind = 同 op 自嵌套折叠(add_const∘add_const=add_const(x,a+b) 等)声明式剪;
    # pow_const∘pow_const 非纯不变(达 x^4/x^.25)→ 留 cost budget 规则,此处第一槽不限。
    _constbin_child0 = {ConstBinaryOp.ADD_CONST: PANEL_KINDS - {SemKind.SHIFTED},
                        ConstBinaryOp.MUL_CONST: PANEL_KINDS - {SemKind.RESCALED},
                        ConstBinaryOp.POW_CONST: PANEL_KINDS}
    for op in ConstBinaryOp:
        slot_allowed = only_add if op == ConstBinaryOp.ADD_CONST else only_mul
        out.append(Production('binary_const', op, None, (NonTerminal.EXPR, NonTerminal.EXPR),
                              PRODUCTION_COSTS['binary_const'],
                              out_kind=_constbin_out[op],
                              child_allowed=(_constbin_child0[op], slot_allowed)))

    # ts(panel + window leaf):每个 op 1 production,window 由独立 len(WINDOWS) 个 productions 填
    only_window: FrozenSet[SemKind] = frozenset({SemKind.WINDOW_VAL})
    for op in TsOp:
        forbid = _TS_FORBID.get(op, frozenset())
        out_k = _TS_OUT.get(op, SemKind.RAW)
        out.append(Production('ts', op, None,
                              (NonTerminal.EXPR, NonTerminal.WINDOW),
                              PRODUCTION_COSTS['ts'],
                              out_kind=out_k,
                              child_allowed=(_panel_minus(forbid), only_window)))

    # pair(panel + panel + window leaf)
    for op in PairOp:
        out.append(Production('pair', op, None,
                              (NonTerminal.EXPR, NonTerminal.EXPR, NonTerminal.WINDOW),
                              PRODUCTION_COSTS['pair'],
                              out_kind=_PAIR_OUT[op],
                              child_allowed=(PANEL_KINDS, PANEL_KINDS, only_window)))

    return tuple(out)


PRODUCTIONS: Tuple[Production, ...] = all_productions()
NUM_PRODUCTIONS: int = len(PRODUCTIONS)


# ============================================================================
# 扩展上下文 + 合法性
# ============================================================================

@dataclass
class ExpansionContext:
    """要被扩展的 slot 的上下文。
    expected_out_kinds: 此 slot 允许的输出 SemKind 集合(从父 production 的 child_allowed 传下)。
    parent_kind / parent_op:仅留作 caller 反查诊断,不进 is_legal。
    remaining_cost:cost 前瞻剪枝。"""
    expected_out_kinds: FrozenSet[SemKind] = ALL_KINDS
    parent_kind:        Optional[str] = None
    parent_op:          Optional[object] = None
    remaining_cost:     int = 999_999_999


def is_legal(prod: Production, ctx: ExpansionContext) -> bool:
    """α-Sem 合法性:返回 True 表示该 production 可在 ctx 下扩展。"""
    # 1) cost 前瞻
    min_total = prod.cost + (1 if prod.children_nts else 0)
    if min_total > ctx.remaining_cost:
        return False

    # 2) α-Sem 类型:此 prod 的 out_kind 必须落在 slot 允许集合内
    if prod.out_kind not in ctx.expected_out_kinds:
        return False

    # ts/pair same-op-same-window 嵌套不剪:冗余如 ts_mean(ts_mean(x,24),24) 保留在搜索空间里
    # (罕见,且 inner ts_mean 仍是合法子特征)。
    return True


def child_context(parent: Production, child_idx: int, sibling_kinds: Tuple[str, ...],
                  outer_ctx: ExpansionContext) -> ExpansionContext:
    """构造扩展某个子节点时的 context。"""
    return ExpansionContext(
        expected_out_kinds=parent.child_allowed[child_idx] if child_idx < len(parent.child_allowed) else ALL_KINDS,
        parent_kind=parent.rhs_kind,
        parent_op=parent.op,
        remaining_cost=outer_ctx.remaining_cost - parent.cost,
    )


def initial_context(max_total_cost: int) -> ExpansionContext:
    """根 slot:整树根必须输出 panel,且禁仿射/sign 包裹(SHIFTED/RESCALED/NEGATED)——
    IC 对根的 per-bar 仿射/取负不变(池吸收符号)→ 根包 add_const/mul_const/neg 纯冗余。"""
    return ExpansionContext(
        expected_out_kinds=PANEL_KINDS - {SemKind.SHIFTED, SemKind.RESCALED, SemKind.NEGATED},
        parent_kind=None, parent_op=None,
        remaining_cost=max_total_cost,
    )


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_MAX_COST: int = 25     # 单棵 alpha 最大累计 cost
DEFAULT_MIN_COST: int =  3     # 整树最小 cost(防止 alpha = 单 operand,无意义)
