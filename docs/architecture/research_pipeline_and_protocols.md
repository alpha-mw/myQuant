# Research Pipeline And Protocols

本文件区分 V17 的目标研究架构与当前已注册的只读主线协议。源码、CLI
注册和 exact artifact readback 高于本文中的目标描述。

## Governance Target

治理层目标链路固定为：

`snapshot -> DeterministicFunnel -> three branches -> Bayesian -> RiskGuard -> ICCoordinator -> PortfolioConstructor -> NarratorAgent`

含义：

- snapshot 负责披露本地数据来源、最新交易日和 strict Parquet 健康状态。
- `DeterministicFunnel` 负责 quant-only 初筛和 candidate set 收敛。
- three branches 严格只包含 `quant`、`fundamental`、`macro`。
- Bayesian selection 负责把分支证据映射为 posterior shortlist。
- `RiskGuard` 负责硬约束、hard veto、exposure cap 和 symbol-level limit。
- `ICCoordinator` 负责共识、分歧和结构化动作建议。
- `PortfolioConstructor` 负责 deterministic 配权。
- `NarratorAgent` 只读取结构化结果并生成说明。

## Runtime Shape

### 当前已注册行为

- `QuantInvestor.run()`、`quant-investor research run`、`quant-investor market
  analyze` 和 `quant-investor market run` 都只解析调用方提供的 canonical
  strategy ID 对应的 exact V17 active pointer。
- 这些入口读取同一份 formal/portfolio/source public closure；它们不扫描、不
  回退、不执行 full-A DAG，也不生产、发布或激活 mainline run。
- 当前源码没有 registered high-level full-A decision producer 或 governed
  mainline publisher。底层 `MainlineStore` exact-once/CAS primitives 是基础设施，
  不是生产发布权限。
- 因此 public DTO 的 `ACTIVE` 只证明 public artifact closure，不证明同日数据、
  持仓连续性、Factor active set、Risk/Portfolio policy 或业务 readiness。

### 目标架构（尚非当前 runtime）

- 目标 DAG 顺序是：`data_snapshot` / symbol list -> batch read ->
  `DeterministicFunnel` -> candidate branch research -> Bayesian selection ->
  deterministic control chain -> reporting artifacts。
- 目标生产 Markov regime 只可在 full-market 或合格 broad-market reference
  scope 下收紧风险预算。
- 目标 LLM review layer 只能生成 advisory hints，不得改变 deterministic 控制链。
- `final_strategy`、`final_report`、`agent_portfolio_plan`、
  `agent_report_bundle` 和 `agent_ic_decisions` 是目标结构化/兼容输出，不应被
  解读为当前已注册 producer 的存在证明。

该目标只有在一个 exact ref 同时包含可验证的纯 producer、闭合输入、immutable
publication 和独立 owner-operated activation 后，才可描述为已执行 runtime。

## Full-Market Reporting

- 当前没有 registered full-market reporting producer；`market analyze/run`
  仍只读取已经激活的 V17 public closure。
- 目标报告协议是 `NarratorAgent -> ReportBundle`。只有目标 producer 被正式实现、
  验证并闭合发布后，full-market 文档才可把 `ReportBundle` 作为事实来源。

## Research Branches

目标稳定 branch set：

- `quant`
- `fundamental`
- `macro`

`kline`、Kronos/Chronos 与 `llm_debate` 仅是 Web 分析的辅助配置面，不计入 V17 v4 mainline canonical DAG 证据分支。已退役的 `intelligence` branch 与 `enable_intelligence` 字段会被严格拒绝且不会出现在当前 artifact 中。

在目标 producer 中，三个 canonical 分支应始终执行，不提供 `enable_quant`、
`enable_fundamental`、`enable_macro` 或对应 `branches.*.enabled` 开关；出现这些
字段时请求应失败，而不是静默忽略。

Bayesian likelihood 证据仅包含 `quant` 与 `fundamental`（`x/2`）；`macro` 只作为 prior/context。

Fundamental 只有在 canonical generation pointer 验证通过、branch readiness 为
`pass`/`warn`、generation 与逐行 lineage 均绑定 `tushare_primary` 时才可进入
likelihood。任一 generation、来源、PIT 或 readiness 证据缺失/冲突时，分支仍可
输出诊断，但该标的的 Fundamental likelihood 必须严格中性化为 `0.50`。
公开 `publish_fundamental_generation()` 只能发布非 primary generation；
`tushare_primary` 必须由 live maintenance 内部能力签发，并同时绑定 provider
manifest、六张 raw table 与三张 generation output table，不能由调用方 metadata
或行字段自报获得。该证明以 `cn-fundamental-primary-provenance.v1` envelope
持久化到 generation manifest 与 pointer；缺少该 envelope 的旧 primary generation
可作为历史文件保留，但读取会 fail closed，不能确认 generation 或贡献 likelihood。
增量 primary generation 若保留旧行，父 generation 也必须通过同一 durable
provenance 校验，并把父 generation ID 与 envelope hash 绑定到新 generation；
离线或旧格式父数据不能借一次 live partial refresh 被整体升级为 primary。

## Structured Control Contracts

- `BranchVerdict`
  - 承载分支结论、`final_score`、`final_confidence` 和三桶说明。
- `RiskDecision`
  - 承载 `hard_veto`、`action_cap`、`max_weight`、`gross_exposure_cap` 等硬约束。
- `ICDecision`
  - 承载共识、冲突点和结构化动作建议，不直接生成精确权重。
- `PortfolioPlan`
  - 承载 deterministic 目标权重、敞口和执行说明。
- `ReportBundle`
  - 承载 `NarratorAgent` 只读生成的结构化报告。

## Target Markov Regime Production Contract

- Markov 默认 production-first：`MARKOV_REGIME_ENABLED=1` 启用，`MARKOV_REGIME_ENABLED=0` 是紧急 kill switch。
- `MARKOV_REGIME_EXECUTION_TARGET=shadow` 为兼容旧输入而保留，但运行时归一化为 production，并写入 `markov_shadow_deprecated_normalized_to_production` 诊断。
- Market regime input 必须带结构化 scope：`regime_scope`、`scope_key`、`production_eligible`、`source_symbol_count`、`requested_symbol_count`、`explicit_symbol_count`、`sampled`、`source_universe_key`。
- 显式小股票池、抽样过小的 universe、或缺失 broad reference 数据时，Markov 状态为 production-ineligible；DAG 保留 MacroAgent regime、baseline target exposure、baseline max single weight 和原控制链。
- 当当前 DAG 输入不是 full-market 时，Markov 使用本地 strict/canonical market reference universe（CN 默认 `full_a`，US 默认 `full_us`），按稳定排序进行 deterministic sample，最多 `MARKOV_REGIME_MAX_REFERENCE_SYMBOLS`。
- Markov 应用时只允许降低风险：target exposure 与 max single weight 都取 baseline 与 Markov 建议的 `min()`；turnover cap 只能新增或收紧，不直接写目标权重。
- DAG/Markov baseline max single weight 为 50%，以匹配集中持仓目标；RiskGuard、流动性与可交易性约束仍可给出更低的 symbol-level cap。
- Regime features 必须按 `as_of` 截断 dated frames 后再计算 returns、volatility、breadth、momentum、drawdown、breakout 和 liquidity。
- Regime history persistence 按 `market + scope_key + source_universe_key + as_of` 隔离；future records 不参与 posterior、transition estimation 或 duplicate-write decisions；无 scope 的 legacy records 不用于 production scoped history。

## Bucketization Rules

以下三类字段必须严格分桶：

- `investment_risks`
  - 只记录会影响可投资性、仓位、流动性、回撤或事件暴露的风险。
- `coverage_notes`
  - 只记录覆盖率、可得性、缺失和 provider 缺口。
- `diagnostic_notes`
  - 只记录超时、fallback、异常、解析失败和工程诊断。

`coverage_notes` 与 `diagnostic_notes` 不得直接作为最终配权输入。

## Reporting Rules

- `NarratorAgent` 是只读角色，不修改候选标的、风险限额、目标仓位或最终权重。
- `NarratorAgent -> ReportBundle` 是目标报告协议名，不是当前 producer 能力证明。
- LLM review 只能输出结构化建议，最终权重只由控制链生成。
