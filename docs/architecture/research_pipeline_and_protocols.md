# Research Pipeline And Protocols

本文件描述当前主线的研究链路、结构化协议边界和报告规则。

## Governance Target

治理层目标链路固定为：

`Research Agents -> RiskGuard -> ICCoordinator -> PortfolioConstructor -> NarratorAgent`

含义：

- Research Agents 负责生成结构化研究证据。
- `RiskGuard` 负责硬约束、hard veto、exposure cap 和 symbol-level limit。
- `ICCoordinator` 负责共识、分歧和结构化动作建议。
- `PortfolioConstructor` 负责 deterministic 配权。
- `NarratorAgent` 只读取结构化结果并生成说明。

## Runtime Shape

- 默认入口：`QuantInvestor`
- 运行方式：先执行 `ParallelResearchPipeline` 的 deterministic research core，再按配置追加 LLM review layer，最后只走一次结构化控制链。
- 兼容输出：`final_strategy`、`final_report`
- 结构化事实面：`agent_portfolio_plan`、`agent_report_bundle`、`agent_ic_decisions`

当前文档应把结构化协议视为规范面，把兼容 markdown 和策略对象视为从结构化结果再生成的输出。

## Full-Market Reporting

- 入口：`quant_investor.market.analyze.run_market_analysis`
- 当前主报告协议：`NarratorAgent -> ReportBundle`
- full-market 文档应以 `ReportBundle` 为事实来源，而不是 markdown 拼接历史名词。

## Research Branches

当前稳定 branch set：

- `kline`
- `quant`
- `fundamental`
- `intelligence`
- `macro`

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
