# operand 耦合的语义判定 —— 由使用者自定义

typed-grammar 的语义系统是**算子驱动**的:所有 operand 叶进语法时类型一致,SemKind 由算子输出向上传播,
`child_allowed` / forbid 集合只引用 SemKind、从不引用具体 operand。因此**唯一与 operand 身份耦合的语义判定**
只有一处,由每个 operand 的 `OperandKind` 决定,且**可由使用者经 `from_columns(overrides=...)` 自定义**。

这一判定管两件事:

1. **raw-level 归属**(G0 gate `is_raw_price_wrapper`):`PRICE ∪ VOLUME` = raw-level。纯 raw-level 叶、无 ts 的
   退化树在截面上 ≈ symbol identity(BTC 永远高、DOGE 永远低)→ 训练阶段 score=0 拒掉。
2. **operand 初始 SemKind**(grammar 产生式 `out_kind`):默认 `RAW`。

## 自定义

```python
from markets.vocabulary import from_columns, OperandKind

vocab = from_columns(
    columns,
    overrides={
        'oi':               OperandKind.VOLUME,      # 声明为量水平 → 计入 raw-level
        'my_xs_percentile': OperandKind.NORMALIZED,  # 声明已截面归一(见下)
    },
)
```

默认走自动分类(`classify_kind`,按列名:价→PRICE、量→VOLUME、其余→FEATURE);`overrides` 逐列覆盖。

| OperandKind | raw-level(G0) | 初始 SemKind |
|---|---|---|
| `PRICE`      | ✅ 是 | `RAW` |
| `VOLUME`     | ✅ 是 | `RAW` |
| `FEATURE`    | ❌ 否 | `RAW` |
| `NORMALIZED` | ❌ 否 | `NORMALIZED` |

## 顺带补的语义精度:`NORMALIZED`

类型系统默认把所有 operand 当 `RAW`,但有些特征本就是**截面归一**信号(已是 cs-zscore / 横截面 percentile),
再套 `cs_zscore`/`cs_rank` 是冗余。声明 `OperandKind.NORMALIZED` → 该 operand `out_kind = SemKind.NORMALIZED`
→ **直接复用现成 cs-forbid**(`cs_zscore`/`cs_rank`/`cs_demean` 均 forbid NORMALIZED 输入),无新机制,自动剪冗余。

> ⚠️ 指**截面**归一。**时序**归一的特征(如 per-symbol 滚动 z-score `vol_zscore_288`)对其做 `cs_zscore` 仍有意义
> → 应留 `FEATURE`。crypto 默认词表无任何 NORMALIZED(所有特征都不是截面预归一)→ 与去枚举前逐位一致。
