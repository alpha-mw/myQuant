<div align="center">

<img src="assets/logo.svg" alt="Quant-Investor" width="520"/>

**一套 fail-closed 的 A 股研究与组合决策系统。**

*量化真正难的不是找到信号，而是证明信号是真的 —— 以及在证据缺失时拒绝出手。*

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-17.0.0-FF6B35?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

[English](README.md) · **简体中文**

[核心命题](#核心命题) · [整体框架](#整体框架) · [运行流程](#运行流程) ·
[它解决了哪些量化问题](#它解决了哪些量化问题) · [特色](#特色) ·
[快速开始](#快速开始) · [当前状态](#当前状态)

</div>

---

## 这是什么

一套单人运维、仅覆盖 A 股的量化研究系统：427 个模块约 29 万行 Python，210 个测试
文件，**没有券商连接，没有下单权限**。它以严格的 point-in-time 行情与基本面数据为
输入，产出三条独立证据分支，经确定性控制链融合，并为每个策略只发布一个受治理的
结果。

它的全部结构都围绕下面这一条命题组织。

## 核心命题

多数量化系统不会大声失败。它们的失败方式是悄悄给出一个**看起来正是你想要的数字**。
真正该问的不是「我的 alpha 是多少」，而是 **「这个数字要在什么条件下才是错的，而我
会不会发现」**。

有四件事会出错，而且都是无声的：

| 失败方式 | 从系统内部看到的样子 |
|---|---|
| **扛不住自己窗口的证据** | 一个用八个月数据挖出来的因子，展示 rank IC 0.05 和八门全过，然后在五年尺度衰减到四分之一。日频 rank IC 对 30 天前瞻收益高度重叠，朴素 t 检验把显著性放大约 √30 倍 —— 足以让一个假象在 p < 1e-4 上看起来像 alpha。 |
| **会自我替换的数据** | 缺失的 Parquet 分区变成 CSV 回退、陈旧快照或推断值。链路照样返回结果，输出里没有任何一处说明它究竟由哪些字节算出。 |
| **会漂移的定义** | 定义为「对当前生产组合取残差」的因子，每当生产集变化，含义就变了。它记录在案的证据描述的是一个已经不存在的基线。 |
| **握有决策权的语言模型** | 能调权重的模型就能幻觉出一个权重。模型输出一旦进入控制路径，「为什么是这个仓位」就不再有确定性答案。 |

整个仓库的组织原则就是对这四点的回答：

> **每一个数字都携带可复现的生成证据。证据缺失等于拒绝，而不是等于取默认值。**

这就是为什么系统是 *fail-closed* 而不是 *best-effort*。best-effort 的链路优化的是
「永远能返回点什么」；这套系统优化的是「绝不返回自己辩护不了的东西」。
`V17_MAINLINE_UNINITIALIZED` 是一个**正确答案**。

## 整体框架

四层，每一层只有在能证明工作来源时才能把工作交给下一层。

```text
┌─────────────────────────────────────────────────────────────────┐
│  4. 治理与发布                                                   │
│     不可变 run -> 活跃指针 CAS -> 只读公开投影                    │
│     因子注册表、WAL + receipt、决策日志                           │
├─────────────────────────────────────────────────────────────────┤
│  3. 确定性控制链                                                  │
│     Bayesian 后验 -> RiskGuard -> ICCoordinator                  │
│     -> PortfolioConstructor -> NarratorAgent（只读）              │
├─────────────────────────────────────────────────────────────────┤
│  2. 证据生产                                                      │
│     DeterministicFunnel -> {quant | fundamental | macro}         │
│     因子池由 FactorGovernanceProtocol v4 治理                     │
├─────────────────────────────────────────────────────────────────┤
│  1. 数据权威                                                      │
│     strict Parquet + PIT 成分 + 哈希绑定 manifest                 │
│     staging -> validate -> serving，损坏字节进 quarantine          │
└─────────────────────────────────────────────────────────────────┘
```

**第 1 层 —— 数据权威。** 规范存储是 Parquet 加哈希绑定的 manifest，经
`parquet_staging -> parquet_serving` 校验后才晋级；不可能的字节进隔离区，而不是就地
修补。成分是 point-in-time 的（`results/pit_universe/`），所以一次回补的上市记录不能
追溯性地混进 2021 年的截面。CSV 只用于显式导出或迁移，永远不作为读取回退。

**第 2 层 —— 证据生产。** `DeterministicFunnel` 用硬性数据质量、可交易性、流动性闸门
加确定性排序，把全 A 压缩成候选集（默认上限 500）—— 没有模型，没有随机性。三条分支
随后在**同一个池**上研究：`quant`、`fundamental`、`macro`。喂给 quant 分支的因子池
另由一套八闸门协议治理（见下）。

**第 3 层 —— 控制链。** 分支证据变成后验，后验遇上硬约束，约束变成权重。每一段都有
类型化契约（`BranchVerdict`、`RiskDecision`、`ICDecision`、`PortfolioPlan`、
`ReportBundle`），且每一段**只能收紧**上一段允许的范围。

**第 4 层 —— 治理。** 链路跑完的结果并不公开。它先成为不可变的 `mainline-run.v1`，
随后只有一次针对预期前值的 compare-and-swap（再加精确回读）才能推进活跃指针。
**部署代码不会激活任何东西。**

第 2、3 层内部的不对称都是刻意的，每一条都堵住一种特定的自欺方式：

- **只有 `quant` 和 `fundamental` 进入 Bayesian 似然。** `macro` 是先验和上下文，它
  塑造风险预算，从不为某只标的投票。
- **Fundamental 必须自证来源。** 只有 canonical generation pointer 校验通过、
  lineage 绑定到 `tushare_primary` 时才允许进入似然；否则分支仍出诊断，但该标的的
  似然被严格中性化为 `0.50` —— 存在，且可证明不含信息。
- **Regime 只能降风险。** Markov 输出以 `min(baseline, suggested)` 应用于目标敞口与
  单票上限；换手上限只能收紧不能放松。regime 信号无法把账本说服成更大的仓位。
- **分支不可开关。** 不存在 `enable_quant`。带这种字段的请求会失败，而不是被静默忽略。
- **说明必须分桶，且分桶是承重的。** `investment_risks`、`coverage_notes`、
  `diagnostic_notes` 按契约分离，后两者被禁止进入配权。一次 provider 故障不能泄漏成
  一个仓位大小。

## 运行流程

四条不同节奏的回路。它们共享存储与哈希，任何一条都不能抄近路绕过另一条。

### 1. 数据维护 —— 每日、显式

```bash
quant-investor market maintain --market CN --staged
quant-investor market storage-validate --market CN
```

下载 → 清洗 → 暂存 → 校验 → 晋级 serving → 刷新 manifest 与 PIT 成分。这是一条独立
且必须显式触发的工作流：**任何分析命令都不会为了补一个洞而自动发起网络请求。**

### 2. 决策 —— DAG

```text
data_snapshot（strict Parquet 健康度、最新交易日）
  -> 标的池 / PIT 成分
  -> 批量读取
  -> DeterministicFunnel            硬闸门 + 确定性排序
  -> GlobalContext                  macro、regime、行业映射
  -> 逐标的研究                      quant | fundamental | macro
  -> Bayesian selection             先验 + 两条似然，log-odds 更新
  -> RiskGuard                      硬否决、敞口与单票上限
  -> ICCoordinator                  共识、分歧、结构化动作
  -> PortfolioConstructor           确定性目标权重
  -> NarratorAgent                  只读报告
  -> 报告持久化 + 各阶段耗时
```

Bayesian 这一步是在层级先验上做两条似然的 log-odds 更新，行动阈值随 regime 变化
（趋势上涨 0.52 买入，趋势下跌 0.60），并对覆盖缺口与降级后端施加显式惩罚。
`RiskGuard` 使用**双语**硬否决关键词集 —— A 股的风险文本是中文，只有英文词表的否决
会永远静默失效 —— 并默认 15% 单票上限；出现三条以上不同风险说明，动作本身就被压到
HOLD。

可选的 LLM review 层位于这条链**旁边**而不是里面。它只输出建议；没有 API key 时降级
为 handoff，确定性结果逐字节不变。**每一个权重都来自 `PortfolioConstructor`。**

### 3. 因子挖掘与治理 —— 周度只读报告，月末提案

```text
候选生成
  -> Gate 1-8          数据安全、覆盖/稳定、IC-RankIC、分组收益、成本/换手、
                       行业 x 市值中性化、purged CPCV 样本外、组合增量
  -> 试验修正          deflated Sharpe、PBO、非重叠 t > 3.0、有效试验数聚类
  -> family BH         同族内 q <= 0.10
  -> 成熟度            12 个月末 RankIC 观测，或 8 个非重叠 30 交易日 cohort，
                       基于 strict Parquet 日历
  -> canonical replay  A/B/C/D 四臂走真实控制链，逐段哈希绑定
  -> 月末提案（永远不是 apply）
```

周度运行**不能改变任何权重**。月末最多两条提案；一旦达到十因子目标，每次新增必须
一进一出，且其配对增量边际的 95% 置信下界必须严格大于零。

### 4. 发布与读取

```text
pipeline 结果 -> mainline-run.v1（不可变）
             -> mainline-active-pointer.v1 上的 CAS（预期前值 + 回读）
             -> mainline-public-run.v1（只读投影）
```

三条命令解析同一个指针，返回同一条权威链：

```bash
quant-investor research run --workspace-root /abs/path/myQuant --strategy-id <id>
quant-investor market analyze --workspace-root /abs/path/myQuant --strategy-id <id>
quant-investor market run --workspace-root /abs/path/myQuant --strategy-id <id>
```

独立的 **Shadow 通道**从内容寻址的封存请求中累积「只朝前」的证据。它的产出永远不
授予主线权威，也永远不能作为公开 run 返回：

```bash
quant-investor-v17-v4 run-forward --workspace-root /abs/path/myQuant \
  --request-path <request.json> --request-sha256 <sha256>
```

研究与生产是**结构上**分离的，不是靠约定。

## 它解决了哪些量化问题

| 经典失败 | 系统的处理 |
|---|---|
| **前视与幸存者偏差** | point-in-time 成分；基本面 generation 逐行带 `availability_date`；regime 特征先按 `as_of` 截断 dated frame 再计算任何东西。 |
| **数据静默替换** | 单一规范 Parquet 通道 + 哈希绑定 manifest；无 CSV 回退；损坏进隔离而非修补；输入缺失是显式 blocker code。 |
| **重叠标签导致的显著性膨胀** | 非重叠 30 交易日 cohort t 检验，门槛取 **3.0** 而非 2.0，且 cohort 宽度由序列自身的采样间隔推断。 |
| **多重检验 / 因子动物园** | 同族 Benjamini-Hochberg `q <= 0.10`，叠加 deflated Sharpe（下限 0.95）、PBO（上限 0.5），以及先把相关候选聚合再算 best-of-N 零假设的**有效**试验数。 |
| **回测过拟合** | 组合式 purged 交叉验证：10 个区块、全部 C(10,2)=45 条路径、以交易日计的 30 日 purge 与 30 日 embargo、内容绑定的证据哈希。 |
| **因子冗余** | 排序目标本身就是**增量**的：逐截面把生产池投影出候选，按残差化 ICIR 排序。`low_dollar_volume` 的克隆从独立 RankIC 0.12 掉到残差 0.011，名次下滑 95 位。 |
| **风格暴露冒充 alpha** | Gate 6 在行业 × 市值桶内评中性化 ICIR，市值用**截面三分位**而非固定绝对阈值（后者在牛市里会把整个截面扫进 `large`，等于不中性化）。`style_exposure_only` 标记那些在去均值后方向翻转或损失一半 ICIR 的因子。 |
| **无意中的集中度** | 单因子 20%、单族 35% 的上限在算术上互相作用：恰好五个因子时 20% 上限迫使每个都是 20%，两个同族就是 40%。**因此五因子集必须来自五个不同族** —— 集中度由算术而非判断约束。 |
| **无法复现的定义** | 依赖可变注册表状态的定义在生产闸门被拒。确实需要时，其基线被 **pin 住**：显式命名并以内容哈希绑定。 |
| **顺周期加风险** | regime 覆盖只能降。永远是 `min(baseline, suggested)`。 |
| **LLM 幻觉进入控制路径** | 建议权与决策权是结构性分离的。`NarratorAgent` 只读；LLM 层不能改候选、风险限额或权重；完全没有 key 时系统行为不变。 |
| **结果不可审计** | 不可变 run、只走 CAS 且带回读的激活、带前后注册表哈希的 append-only WAL 蓝图、当日激活 receipt，以及一份 advisory 与 human action 必须按 ID 配对才可能考虑成交的决策日志。 |
| **稀薄日历粉饰统计** | 交易日来自活跃快照**实际观测到**的日子，稀薄日被排除：1,796 个观测到的 A 股交易日中，只有 858 个带有真实截面。用其余日期凑成熟度会满足算术，但底下什么都没有。 |

## 特色

**fail-closed 是架构，不是开关。** 每个公开面只解析唯一一个指针。缺失 →
`V17_MAINLINE_UNINITIALIZED`，不写任何东西；无效 → `V17_MAINLINE_BLOCKED:<blocker>`。
没有回退、不扫目录找最近一次运行、不自举。主线回测不是降级而是拒绝：
`V17_BACKTEST_UNAVAILABLE`。

**内容寻址的权威。** 结果沿固定链路流转，每一环绑定上一环的字节。激活是
compare-and-swap 加精确回读 —— 竞态或过期预期会中止，而不是覆盖。

**AI 只建议，控制链才决策。** 系统确实是多智能体的，而且**没有一个智能体能决定任何
事**。LLM 层是叙述者和评审者，背后挂着 provider 优先级链；把它整个拿掉，权重一分不动。

**因子的门槛由搜索规模决定，而不是由最好那个结果决定。** deflated Sharpe 问的是：
N 次毫无价值的试验中最好那次会打到多少分。当前运行这个数是 1.067，而实际观测到的最佳
ICIR 是 0.724 —— 所以诚实的结论是：**枚举更多候选只会抬高门槛**，出路是更小的预注册
搜索而不是更大的搜索。这个反直觉结论是承重的设计结论，不是一个麻烦。

**冗余是目标函数，不是驳回理由。** 多数链路把冗余放在最后当否决。这里的搜索**直接
指向**增量价值：按候选给池子新增了什么来排序，近似重复品根本到不了列表顶端，也就不
需要在最后被否掉。

**系统用自己的词汇报告自己的不足。** 挖掘无产出时，输出会区分
`factor_exposure_evidence_not_ready`（管道问题）与
`no_qualified_positive_candidate`（关于候选本身的判断）。这是两种不同的问题，运行结果
会说清楚撞的是哪一种。

**研究与生产不会被混淆。** Shadow 是独立运行时、独立存储根、独立 schema 族。不存在
任何把一方提升为另一方的开关。

## 快速开始

```bash
uv sync
cp .env.example .env
```

本地验证保持离线。只填显式授权的工作流真正需要的项。随后运行
[发布与读取](#4-发布与读取) 中的任一读取命令。

## 当前状态

仅 CN，**没有券商、下单、执行或交易权限**。有两件事是敞开的，且都是设计的后果而非
未完成的收尾。

**Factor Governance v4 尚未 ready。** 在完整开放交易日日历上度量，五个候选中三个通过
`q <= 0.10`。失败的两个恰好就是记录证据来自短期近窗的那两个 —— 它们扛不住对 30 天
重叠前瞻收益的修正。门槛在正常工作，因此 `factor_governance_ready` 保持 `false`。

**Gate 1、5、8 目前还没有证据生产者**，因此 fail-closed，任何候选最高只能到 5/8。它们
分别需要：逐候选的版本化 PIT / 可交易性审计；用参与率滑点模型替换当前 1bp 平坦假设；
以及走真实控制链的完整 A/B/C/D replay。

**主线还没有发布器。** `quant_investor/v17_mainline/` 只有读侧。在决策输出生产者存在
之前，公开面返回 `V17_MAINLINE_UNINITIALIZED` —— 这是正确答案，不是 bug。

## 项目地图

```text
quant_investor/
  cli/                       公开命令路由
  data/                      数据源、PIT 成分、加工
  market/                    CN 维护、规范读取、DAG executor
  funnel/                    确定性全市场压缩
  agents/                    分支、RiskGuard、IC、PortfolioConstructor、Narrator
  bayesian/                  先验、似然、后验、校准
  macro/ regime/             上下文与只降风险的覆盖层
  factors/                   Factor Governance v4、CPCV、试验修正
  pipeline/                  研究与组合链路
  v17_mainline/              活跃指针权威与公开 run 读取
  v17_v4_contract/           V17 v4 schema 与校验
  v17_v4_runtime/            只研究的 Shadow 运行时
  learning/                  交易后复盘与记忆晋级
portfolio_dashboard/         只读看板契约
web/                         本地研究工作台 API
results/v17_mainline/        受治理的活跃主线结果
results/v17_v4_shadow/       只研究的前瞻证据
```

## 开发

Python 3.13+。先跑最小相关测试；大范围分阶段升级工作用：

```bash
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

除非任务显式授权，本地验证期间不要调用实时数据、LLM、券商、下单、执行或交易 API。

## 文档

- [文档索引](docs/README.md)
- [V17 v4 主线契约](docs/architecture/v17_v4_production_research_contract.md)
- [研究链路与协议](docs/architecture/research_pipeline_and_protocols.md)
- [Factor Governance v4](docs/factor_governance_v4.md)
- [因子挖掘机制](docs/factor_mining_mechanism.md)
- [V17 v4 运维](docs/runbooks/v17_v4_operations.md)
- [入口与版本](docs/architecture/entrypoints_and_versioning.md)
- [前瞻证据运行时](docs/architecture/v17_v4_forward_evidence_runtime.md)
- [模块地图](docs/modules/module_map.md)
- [Agent 指南](AGENTS.md)

## 许可证

[MIT](LICENSE) © 2024 alpha-mw
