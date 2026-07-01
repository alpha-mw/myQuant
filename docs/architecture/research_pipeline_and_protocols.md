# Research Pipeline And Protocols

本文件描述当前主线的研究链路、结构化协议边界和报告规则。

## Governance Target

治理层目标链路固定为：

`snapshot -> DeterministicFunnel -> four branches -> Bayesian -> RiskGuard -> ICCoordinator -> PortfolioConstructor -> NarratorAgent`

含义：

- snapshot 负责披露本地数据来源、最新交易日和 strict Parquet 健康状态。
- `DeterministicFunnel` 负责 quant-only 初筛和 candidate set 收敛。
- four branches 只包含 `quant`、`fundamental`、`intelligence`、`macro`。
- Bayesian selection 负责把分支证据映射为 posterior shortlist。
- `RiskGuard` 负责硬约束、hard veto、exposure cap 和 symbol-level limit。
- `ICCoordinator` 负责共识、分歧和结构化动作建议。
- `PortfolioConstructor` 负责 deterministic 配权。
- `NarratorAgent` 只读取结构化结果并生成说明。

## Runtime Shape

- 默认入口：`QuantInvestor`
- 单标的公开主线：`QuantInvestor` 调用全市场 DAG helper，并把 DAG artifacts 转为 `QuantInvestorPipelineResult`。
- 全市场主线：`quant-investor market analyze/run` 走 `execute_market_dag()`。
- DAG 顺序：`data_snapshot` / symbol list -> batch read -> `DeterministicFunnel` -> candidate branch research -> Bayesian selection -> control chain -> reporting artifacts。
- 生产 Markov regime 在 DAG context 阶段读取市场状态；它只在 full-market 或 broad market-reference scope 生产合格时覆盖 `GlobalContext.macro_regime` 并收紧风险预算。
- 可选 LLM review layer 只提供 advisory hints；缺 key 或本地禁用时降级为 Codex handoff，不改变 deterministic 控制链。
- stage profile：`market analyze/run` 写入 snapshot、symbol list、batch read、funnel/context、candidate research、Bayesian、control chain、report persistence 等阶段耗时。
- 兼容输出：`final_strategy`、`final_report`
- 结构化事实面：`agent_portfolio_plan`、`agent_report_bundle`、`agent_ic_decisions`

当前文档应把结构化协议视为规范面，把兼容 markdown 和策略对象视为从结构化结果再生成的输出。

## Full-Market Reporting

- 入口：`quant_investor.market.analyze.run_market_analysis`
- 当前主报告协议：`NarratorAgent -> ReportBundle`
- full-market 文档应以 `ReportBundle` 为事实来源，而不是 markdown 拼接历史名词。

## Research Branches

当前稳定 branch set：

- `quant`
- `fundamental`
- `intelligence`
- `macro`

`kline`、Kronos/Chronos 和 legacy batch pipeline 不属于 v13 canonical branch set；旧 payload 中的未知分支只能被过滤或作为历史兼容读取，不能重新进入 runtime branch set。

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

## Markov Regime Production Contract

- Markov 默认 production-first：`MARKOV_REGIME_ENABLED=1` 启用，`MARKOV_REGIME_ENABLED=0` 是紧急 kill switch。
- `MARKOV_REGIME_EXECUTION_TARGET=shadow` 为兼容旧输入而保留，但运行时归一化为 production，并写入 `markov_shadow_deprecated_normalized_to_production` 诊断。
- Market regime input 必须带结构化 scope：`regime_scope`、`scope_key`、`production_eligible`、`source_symbol_count`、`requested_symbol_count`、`explicit_symbol_count`、`sampled`、`source_universe_key`。
- 显式小股票池、抽样过小的 universe、或缺失 broad reference 数据时，Markov 状态为 production-ineligible；DAG 保留 MacroAgent regime、baseline target exposure、baseline max single weight 和原控制链。
- 当当前 DAG 输入不是 full-market 时，Markov 使用本地 strict/canonical market reference universe（CN 默认 `full_a`，US 默认 `full_us`），按稳定排序进行 deterministic sample，最多 `MARKOV_REGIME_MAX_REFERENCE_SYMBOLS`。
- Markov 应用时只允许降低风险：target exposure 与 max single weight 都取 baseline 与 Markov 建议的 `min()`；turnover cap 只能新增或收紧，不直接写目标权重。
- DAG/Markov baseline max single weight 为 50%，以匹配集中持仓目标；RiskGuard、Theme overlay、流动性与可交易性约束仍可给出更低的 symbol-level cap。
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
- `NarratorAgent -> ReportBundle` 是当前公开报告协议名。
- LLM review 只能输出结构化建议，最终权重只由控制链生成。
