# quant — 公式化 Alpha 因子挖掘 + 回测内核

用强类型遗传规划(GP)搜索公式化 alpha,经纯 C++ / OpenMP 回测内核评估,确定性 sizing 融合成组合权重。支持 crypto 永续 / 美股日线 / A 股日线三市场。

---

## 代码逻辑

端到端是「公式 alpha 搜索 → 回测评估 → 建池 → sizing 融合」一条流水线,各层搜索引擎无关:

1. **DSL(`evaluation/`)** — 公式 alpha 表示为强类型 AST;上下文无关文法(CFG + α-Sem-k 成本预算)约束搜索空间,保证语法/语义合法。operand 词表**由 parquet 列名自动构建**(自动分类 价/量/特征),换市场即换列。求值器把整棵 token 树编译到 C(及可选 CUDA)上批量求值,subtree-hash 缓存复用公共子树。

2. **搜索(`search/`)** — `gp/`:DEAP 强类型 GP,NSGA-II 双目标(净 Sharpe + 行为多样性),在 train 段挖公式 alpha。`tail_diversity/`:尾部条件多样性搜索原型(真多样 vs 伪多样,隔离实现)。

3. **评估 + 建池(`rl/`)** — evaluator 算 rank-IC 与组合回测收益,经多道准入门(|IC| / 正交贡献 Δ / OOS holdout)收进 `LinearAlphaPool`,池不定权。

4. **回测内核(`backtest/`)** — 纯 C++ / OpenMP:`engine/` 做组合级 (T,S) 连续 target-position 回测(fee / funding / trailing-stop / 强平);`ops/` 是 C++ 因子算子,`ops_cuda/` 是 CUDA kernels(cupy NVRTC 运行时编译)。不建模订单簿。

5. **sizing(`rl/sizing.py`)** — 确定性融合:池成员 z-score → AFF 因果滚动 lstsq 融合 → top-K 多空构造 → 组合级 vol-target 标量杠杆。

6. **市场抽象(`markets/`)** — `MarketProfile` 把日历 / 年化 / 复权口径隔离在引擎外;crypto / 美股 / A 股为三种模式,词表与画像由数据自动识别。

模块速览:

| 模块 | 职责 |
|---|---|
| `evaluation/` | 公式 DSL(AST / CFG / C+CUDA 求值)+ panel adapter + subtree-hash 缓存 |
| `search/` | `gp`(DEAP 强类型 GP)+ `tail_diversity`(尾部多样性原型) |
| `rl/` | evaluator + LinearAlphaPool + sizing 融合 |
| `backtest/` | C++/OpenMP 回测内核(engine)+ 因子算子(ops C++ / ops_cuda CUDA) |
| `markets/` | MarketProfile 市场画像抽象(日历 / 年化 / 复权 / 词表) |
| `config/` | INI ↔ dataclass 单一参数源 |
| `main.py` | 训练侧 CLI |

---

## 使用方法

### 安装 + 构建扩展

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirement.txt   # numpy/pandas/pyarrow/torch/deap

bash backtest/build.sh           # 编 C++ 扩展:backtest._bt(引擎)+ backtest.ops._ops(算子);Linux→.so / WSL→Win .pyd
bash backtest/ops_cuda/build.sh  # CUDA 算子:cupy NVRTC 运行时编译 + 对 numpy 黄金参考验证(需 cupy + GPU)
```

GPU 研究路径(`[evaluator] device=cuda`)需另装 `cupy`;CPU 路径设 `device=cpu`。

### 挖因子 / 物化

`main.py` 只接位置参数(输出路径),其余全读 `config/config.ini`:

```bash
python main.py gp-baseline output/pool.json                       # DEAP 强类型 GP(NSGA-II)— crypto 永续挖矿
python main.py materialize output/pool.json output/materialized/  # alpha JSON → 月级 per-alpha parquet
```

挖矿产物 = deploy bundle JSON(`{"pool": [{tree, ...}, ...]}`)+ `logs/` 快照。

> 本仓为公开快照:挖矿需自备符合 `evaluation/adapter.py` 读取格式的 parquet 面板(数据采集管线未含)。`main.py` 另有 `alphasage` / `gp-equity` / `gp-ashare` 子命令,依赖本快照未含的组件(GFlowNet 引擎 / 权益数据加载器)。

---

## 参考论文 / 致谢

本项目站在以下工作之上,谨致谢:

- **Alpha Discovery via Grammar-Guided Learning and Search** —— `evaluation/grammar.py` 的 CFG 文法与 α-Sem-k 成本预算受其启发。
- **FACT**(arXiv:2604.26666)—— `backtest/ops_cuda/` 的算子融合吸收其 compositional kernel synthesis 范式。
- **NSGA-II**(Deb et al., 2002)+ **DEAP**(Fortin et al., 2012)—— `search/gp` 的多目标强类型 GP。
- **AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets** —— `[alphasage]` 配置与 `alphasage` 子命令对应的 GFlowNet 挖矿方法(引擎实现未含在本快照)。

`search/tail_diversity/` 原型参考:
- Mallela & Leonelli, 2026(arXiv:2606.16840)—— 崩盘下尾依赖图。
- Engelke & Taeb, JMLR 2025(arXiv:2403.09604)—— eglatent 剥离共因。
- Dominated Novelty Search, GECCO 2025(arXiv:2502.00593)—— 无界 archive 多样性。
- Campbell et al., 2026(arXiv:2603.15963)—— 收益尾部外部坐标。
