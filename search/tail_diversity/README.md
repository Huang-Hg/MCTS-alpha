# tail_diversity — 尾部条件多样性搜索(真多样 vs 伪多样)

> 隔离原型(2026-06-25)。**不改动任何现有模块**;旋钮暂用显式常量,集成时再提 `[tail_diversity]` ini section。
> 起因:`flat reward 打赢所有 reward` 的旧结论 → "大量可堪一用、多样性极佳的因子群 ≫ 个体极佳因子" →
> 新搜索策略:**reward = 多样性,IC 只作准入门槛**。本目录解决其中最难的一环:**怎么构造多样性,
> 使搜索可进行(tractable)且真的多样(黑天鹅里不塌成 corr→1)**。

---

## 0. 一句话

普通的"相关性/IC/协方差多样性"是**伪多样**——崩盘里必然 corr→1。真多样 = 把多样性建在
**① 尾部条件(只看崩盘 bar)② 净成本口径(扣 funding+fee)③ 剥离共因(partial out 共同崩盘驱动)**
之上;用 **分布级描述子 + 无界 archive(DNS)** 让搜索仍跑得动;关键戒律:尾部描述子当
**准入闸/holdout 验证器**,不当被直接优化的 reward(否则搜索会 Goodhart 它)。

接 [[project_cost_erosion_bottleneck]]:瓶颈是 net~0(funding+fee 蚕食),成本集中在拥挤负 funding
空腿。**真数据修正(§4b,2026-06-25):成本是缓慢漂移,逐 bar 量级 << 收益波动 → net 逐 bar 收益 ≈ gross,
建在"net per-bar 收益"上根本抓不到 funding 问题。** 共享 funding 暴露活在**成本流本身**里——所以要
**显式加一条"funding-成本流 co-exposure"坐标**(成本序列的尾部相关),与收益描述子正交并列。

---

## 1. 病根:现 `rl/alpha_pool.py` 的多样性度量就是伪多样

`LinearAlphaPool`(`rl/alpha_pool.py`)现有口径:

```
per_t_pnl[t] = Σ_s w[t,s]·y[t,s]     # w = cs-demean(signal)/L1-norm —— GROSS
R²           = max_k pearson(cand_pnl, member_k_pnl)²     # _proj_r2,全样本
Δ            = |rank_ic| · (1 − R²)   # 正交贡献门
```

两处致命:**(i) 全样本 Pearson** —— 平时去相关、崩盘共动的因子被判为多样;
**(ii) gross 口径** —— 共享 funding 暴露在信号层不可见。实证背书:Mallela & Leonelli 2026
(arXiv:2606.16840)证加密下尾依赖图崩盘时近全连通,高斯口径低估联合崩盘概率 ~8×。

烟测把这个 bug 复现并量化了(见 §5):伪多样因子全样本 R²=0.02(现池放行),尾部 R²=0.70。

---

## 2. (b) 两个核心机件 —— 代码级深读结论

### 2.1 eglatent(剥离共因,精确版)— Engelke & Taeb, JMLR 2025 (arXiv:2403.09604)

把池的尾部依赖精度矩阵分解为 **Θ̃ = S − L**:S 稀疏 = 剥掉共同崩盘驱动后的**真**条件尾独立图;
L 低秩 PSD = 少数隐崩盘驱动(corr→1 的真凶),**秩自动推断**(非超参,按特征值 ≥1e-3 数)。

- 模型:Hüsler–Reiss 极值图;经验变差图 `Γ̂=(1/d)Σ_m Sigma2Gamma(cov(log X̃[X̃[:,m]>1]))`;
  `Θ=(Π(−Γ/2)Π)⁺`,Π=I−11ᵀ/d。
- 凸规划:`min −logdet(UᵀPU) − ½tr(PΓ̂) + λ₁(‖S‖₁ + λ₂·tr(L))`,s.t. `P=S−L`,`UUᵀPUUᵀ=P`,`S,L⪰0`。
- **工程定位(决定架构)**:样本复杂度 **k ≳ p²log(p)** 尾部越界样本 + 每次解一个 log-det 锥规划
  (cvxpy+SCS,p~30–80 几秒)→ **不能放每候选热路径,只能周期性在已接受池上跑**。
- 读出:`latent_loadings(L)` → 隐崩盘驱动载荷;`sparse_graph(S)` → 真特异尾连接;
  `member_diversity_scores = residual_latent × 1/(1+deg_S)`。
- 实现:`eglatent.py`,对照 R 包 `graphicalExtremes` 源码逐式重写。**需 `pip install cvxpy`(+SCS);
  本地未装 → 未实跑**,代码按源码忠实移植,smoke 在系统边界 try-ImportError 选跑。

### 2.2 Dominated Novelty Search(无界 archive,可搜索性)— GECCO 2025 (arXiv:2502.00593)

- competition fitness = 到 **k 个最近"更优"邻居** 的平均 L2 距离;无更优邻居(全局最优)→ +∞ 永留。
  选择 = 按此分数截断到容量。
- **为什么适配**:描述子是**高维学习量**(隐尾载荷 D~10–50),网格 MAP-Elites 在高维爆格
  (格数=分辨率^D);DNS 无网格、成本仅 **O(N²·D)** 线性于 D,论文实测撑到 **1000 维**、随维度增长
  以 **p<1e-9** 碾压网格。
- 天然实现"IC 当闸、多样性当选择压":弱 IC 因子只有在描述子空间**远离所有更优因子**时才存活。
- 纯 numpy(无 JAX/QDax 依赖),k=3 稳健默认。坑:学习描述子各维尺度不一,**L2 前必白化**(已实现)。
- 实现:`dns_archive.py`。

---

## 3. (a) 设计:三支柱描述子 + reward + archive

### 3.1 描述子构造(`tail_descriptor.py`)

**两条正交轴**(真数据 §4b 证实:收益轴 OK、funding 成本轴才是伪多样所在):

```
轴 A — 收益尾部共动(crash 窗 corr→1):
② crash 窗:        𝒯 = { t : market_loss[t] ≤ quantile_q }     # tail_mask,q~0.02(真 cascade)
③ 尾部签名:        sig_i = standardize( per_t_pnl_i[𝒯] )        # tail_signature(gross 即可;net≈gross)
   尾部相关:        R²_tail = max_k pearson(sig_cand, sig_k)²     # tail_corr_vec + proj_r2
   隐崩盘驱动:      V_r = top-r SVD 方向(池尾部签名)            # latent_drivers(廉价)/ eglatent(精确)
   投影残差:        residual = 1 − ‖V_rᵀ sig‖²/‖sig‖²            # latent_residual

轴 B — funding-成本流 co-exposure(★ 真数据证实的主问题轴,新增):
   成本流:          cost_i[t] = gross_ret_i[t] − net_ret_i[t]    # ≈纯 funding(实测占 97%)
   成本相关:        Rc²_tail = max_k pearson(cost_i[𝒯], cost_k[𝒯])²
   集成时由 run_bt(with_equity) 双跑(gross funding=0 / net)取差,非逐 bar 信号可得。
```

⚠️ 原 "① 净 per-bar PnL 描述子" 已**作废**:net 逐 bar 收益 ≈ gross(成本是慢漂移),抓不到 funding;
改用上面 **轴 B(成本流相关)** 显式度共享 funding 暴露。

### 3.2 reward 形状(drop-in 替换 `LinearAlphaPool` 的 Δ)

```
Δ_tail = quality · (1 − R²_tail) · residual_latent · (1 − Rc²_tail)   # tail_diversity_delta(+成本轴)
         └质量幅度┘ └治收益崩盘共动┘ └治共同隐驱动┘   └★治共享 funding 暴露┘
```

- `quality` = |rank IC|(现口径)或净 CVaR 幅值(更贴 net 目标)。
- `(1 − Rc²_tail)` = **轴 B**:候选若与池共享 funding 成本流则压分(真数据证实的主问题轴)。
- IC 仍是**硬准入闸**(`|rank_ic| ≥ admit_rankic_min`,不变);多样性进 reward/选择,不进闸。

### 3.3 archive(DNS over 隐尾载荷)

- 描述子 = 每因子的**隐尾载荷向量**(`latent_drivers` 廉价版,或 eglatent 的 L 载荷,周期刷新)。
- quality = 净 CVaR / |rank IC|。DNS 截断保容量,等质量下保真多样、剔伪多样(§5 烟测验证)。
- 这替代/增强现 `try_new` 的 leave-one-out prune(后者基于全样本 R²)。

### 3.4 与现有代码的精确对接点(集成时,本原型**未触碰**)

| 现 `LinearAlphaPool` | 改为 |
|---|---|
| `_corr_with_pool`(C `ops.pnl_corr_vec`,全样本) | 在 crash 窗 + net PnL 上算 → `tail_corr_vec`(或把 mask+net 下推到 C 算子) |
| `_proj_r2` | 不变(口径相同,输入换尾部签名) |
| `Δ = q·(1−R²)` | `Δ_tail = q·(1−R²_tail)·residual` |
| leave-one-out prune | DNS `add_batch`(隐尾载荷描述子) |
| 周期性 | 每 N 迭代用 `eglatent.fit_eglatent` 刷新隐崩盘驱动子空间 |

PoolMember 需多存:`per_t_pnl_net`(net 口径)+ `tail_sig` + `latent_loading`。`market_loss` 由
evaluator 注入(等权 universe / BTC 的 bar 收益)。

---

## 4. 必须守的护栏(文献坑,接调研 §D + 补缺)

1. **估计方差**:crash bar 天生稀缺,裸 χ/相关高方差 → 会填出不可复现的"伪多样格子"(corr→1 的
   估计侧孪生)。对策:eglatent 的稀疏+低秩正则;跨 symbol 池化;**多 crash 窗可复现性**(Extract-QD)
   才准入。**慎用生成增强**:因子结构扩散会自带共因、可能制造伪 corr→1。
2. **Goodhart/reward-hacking**:强搜索被告知"最大化尾部多样 proxy" → 会过拟合定义描述子的那几个历史
   crash 窗,OOS 照样共崩。**戒律:尾部描述子当 holdout 准入闸/evaluator,不当被优化的 GFlowNet reward**
   (AlphaEval 把它当评估闸并证高尾稳健→低 maxDD);max-min over crash windows。
3. **执行/清算拥挤**(funding 之外的第二相关器):共享抵押品 / ADL 瀑布 / 拥挤小币空头薄盘平仓 —— 价格
   收益描述子看不见,且 backtest kernel 不建模订单簿 → 须当**外部坐标**喂入(Campbell et al. 2026,
   arXiv:2603.15963)。本原型未含,集成时加。**注**:§4b 真数据已证 funding(轴 B)是当前主成本相关器;
   清算拥挤是其尾部放大与第二来源。
4. **no-lookahead**:`tail_mask` 的 market_loss 在评估窗内 post-hoc 算(与 IC 同源,研究期度量)。
   挪到 live 因果门须改滚动因果指标。见 [[feedback_no_lookahead]]。

---

## 4b. 真数据验真(`alphasage_pool_rankconv_1000ep.json`,val 段,q=0.02,2026-06-25)

脚本 `scripts/diag_pool_tail_collapse.py`(收益轴)+ `scripts/diag_pool_net_collapse.py`(成本轴 + funding 隔离),
本地 val 跑(~100s,含 load_panel)。**决定性结论:收益轴多样性 OK,funding 成本轴才是伪多样所在。**

```
轴                full|corr|  tail|corr|  伪多样对(full R²<0.5 且 tail R²>0.5)
收益 gross/net      0.160      0.219       0/276        ← net≈gross,逐 bar 抓不到成本
总成本(f+fee+imp)  0.339      0.425      27/276
纯 funding         0.329      0.385      30/276        ← 占总成本相关 97% → fee/timing 非主因
```

- **收益轴**:net 逐 bar ≈ gross(Δ|corr|≈0),0/276 严格伪多样 → 收益多样性不是瓶颈。
- **★ 成本轴(funding)**:因子付的成本比赚的钱相关 ~2×(0.329 vs 0.160),尾部冲到 0.385,30/276 对
  "收益分散、funding 一起被薅"。funding-only 隔离(fee=impact=0)复现 97% 的成本相关 → **共动就是 funding 本身**,
  非 fee/调仓时点(caveat 钉死)。
- 最差 funding 对全压 `sum_top_lsr / intra_sum / count_top_lsr / premium_index / tenure_norm`——拥挤
  long/short-ratio + OI + 基差那条轴(机制 = 同名同向空头)。
- **设计含义**:这就是新增**轴 B(funding-成本流 co-exposure)** 的实证依据;轴 A(收益尾部)在本池上是次要的。

---

## 5. 烟测结果(`smoke_tail_diversity.py`,纯 numpy,毫秒级,已跑通)

```
1. 全样本 R² 被骗:    fake 池全样本 max R² = 0.025  < 0.5(现池当多样准入)
2. 尾部 R² 抓出共动:  新 fake 全样本 R²=0.020 / 尾部 R²=0.701  > 0.5(尾部口径拒绝)
3. 真多样不误杀:      新 real 尾部 R²=0.021  < 0.5
4. collapse 量化:     fake=0.668  vs  real=0.039   ("看似多样"被量化)
5. 剥离共因残差:      res(new fake)=0.204(抓出)  res(new real)=0.972(保真)
6. DNS:               保留={real0,real1,fake2,fake4}  淘汰=全 fake
7. eglatent 精确路径:  [SKIP] 本地无 cvxpy(SVD 廉价版已验)
8. ★轴B funding:      收益 R²_tail=0.027(分散) / funding Rc²=1.000(共担抓出) / 独立 0.002
                      两轴 Δ:funding共担=0.000(压到0) vs 独立=0.956(保留)
ALL SMOKE PASSED
```

跑:`/mnt/d/03learn/machine/.venv/Scripts/python.exe search/tail_diversity/smoke_tail_diversity.py`

---

## 6. 文件

| 文件 | 内容 | 依赖 |
|---|---|---|
| `tail_descriptor.py` | 轴A 尾部窗/签名/相关/隐驱动残差 + **轴B `cost_stream`/`cost_corr_vec`/`cost_r2_tail`** + 两轴 `tail_diversity_delta`/collapse | numpy |
| `dns_archive.py` | Dominated Novelty Search archive(白化+截断) | numpy |
| `eglatent.py` | 稀疏+低秩 HR 分解(剥离共因精确版,周期用) | numpy + **cvxpy(SCS)** |
| `smoke_tail_diversity.py` | fake-vs-real 自包含验收 | numpy |

---

## 7. 集成路线(分阶段,每步先 val 段回测对比 baseline,见 [[feedback_validate_before_deploy]])

1. ✅ **离线验真**(本原型已做):合成 fake-vs-real,证描述子有效(§5)。
2. ✅ **真池验真**(2026-06-25,§4b):val 段 rankconv 池,收益轴 0/276 伪多样、funding 成本轴 30/276
   (funding 占成本相关 97%)→ **轴 B(funding-成本流)是主问题轴**,轴 A 次要。
3. ✅ **把轴 B 做进描述子**(2026-06-25):`tail_descriptor.py` 加 `cost_stream`/`cost_corr_vec`/`cost_r2_tail`,
   `tail_diversity_delta` 升级两轴 `q·(1−R²_tail)·residual·(1−Rc²_tail)`;smoke §8 验真(funding 共担候选 Δ→0)。
   成本流由 `run_bt(with_equity)` gross/net 双跑取差得(见 `scripts/diag_pool_net_collapse.py`)。
4. **接入 reward/闸**:`LinearAlphaPool` 加 `Δ_tail`(含轴 B)路径(开关),val 段对比 net-Sharpe/maxDD。
5. **archive 换 DNS** + 周期 eglatent 刷隐驱动。
6. **加执行/清算拥挤坐标**(外部,funding 之外的第二尾部相关器)。
7. 证伪即按 [[feedback_delete_dead_designs]] 删干净。

---

## 8. 关键文献(已对抗式验证可解析,2024–2026)

- Mallela & Leonelli 2026, *Crashing Together, Rallying Apart* — arXiv:2606.16840(下尾近全连通;未评审 preprint)
- Engelke & Taeb 2025, *eglatent* — arXiv:2403.09604(JMLR;稀疏+低秩 HR)
- Lim, Faldor, Cully, Grillotti 2025, *Dominated Novelty Search* — arXiv:2502.00593(GECCO'25)
- Gong/Huser 2024, *Partial Tail-Correlation Coefficient* — arXiv:2210.07351(Technometrics 2024)
- Boulin & Bucher 2025, *Structured Linear Factor Models for Tail Dependence* — arXiv:2507.16340
- AlphaEval 2025 — arXiv:2508.13174(多样性熵+重尾扰动当评估闸→低 maxDD)
- Campbell/Hey/Moallemi/Nutz 2026, *Risk-Based Auto-Deleveraging* — arXiv:2603.15963(ADL 共同体尾相关器)
- ⚠️ AlphaSAGE(arXiv:2509.25055)的 novelty = `1−max|IC|`(全样本),属被批判那族,**勿照抄**。
