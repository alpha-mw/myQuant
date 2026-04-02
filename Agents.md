# Agents.md — 多智能体投研系统架构文档

> **quant_investor v12.0.0-stable**
>
> | 版本标识 | 值 |
> |---|---|
> | architecture_version | `12.0.0-stable` |
> | agent_schema_version | `2026-03-23.agent.v1` |
> | branch_schema_version | `branch-schema.v12.unified-mainline` |
> | ic_protocol_version | `ic-protocol.v12.mainline` |
> | report_protocol_version | `report-protocol.v12.mainline` |

系统采用**双层智能体架构**：确定性控制层负责硬约束与组合构建，LLM 审阅层提供定性评判与辩论。审阅层严格为 advisory-only，永远不覆盖控制层的硬否决。

---

## 1. 架构总览

```
                    ┌─────────────────────────────────────────────────┐
                    │           AgentOrchestrator (异步编排)            │
                    │           agents/orchestrator.py                │
                    └──────────┬──────────────────┬──────────┬───────┘
                               │                  │          │
              Phase 1 (并发)    │   Phase 2 (串行)  │  Phase 3  │
                               │                  │  (串行)   │
                               ▼                  ▼          ▼
                    ┌──────────────────┐  ┌────────────┐  ┌──────────────┐
                    │ BranchSubAgent×5 │  │RiskSubAgent│  │ MasterAgent  │
                    │  (LLM 审阅)      │  │ (LLM 风控) │  │ (IC 主席)    │
                    └────────┬─────────┘  └─────┬──────┘  └──────┬───────┘
                             │                  │                │
  ═══════════════════════════╪══════════════════╪════════════════╪══════════
       审阅层 (Advisory)      │                  │                │
  ───────────────────────────┼──────────────────┼────────────────┼──────────
       控制层 (Deterministic) │                  │                │
  ═══════════════════════════╪══════════════════╪════════════════╪══════════
                             ▼                  ▼                ▼
                    ┌──────────────────┐  ┌────────────┐  ┌──────────────┐
                    │ BranchAgent ×5   │  │ RiskGuard  │  │ICCoordinator │
                    │ (BranchVerdict)  │  │(RiskDecis.)│  │ (ICDecision) │
                    └────────┬─────────┘  └────────────┘  └──────┬───────┘
                             │                                   │
                             │                                   ▼
                             │                          ┌────────────────────┐
                             │                          │PortfolioConstructor│
                             │                          │  (PortfolioPlan)   │
                             │                          └────────┬───────────┘
                             │                                   │
                             └───────────────┬───────────────────┘
                                             ▼
                                    ┌─────────────────┐
                                    │  NarratorAgent   │
                                    │  (ReportBundle)  │
                                    └─────────────────┘
```

---

## 2. 快速参考表

### 控制层 (Deterministic)

| Agent | 源文件 | 输入 | 输出 | 核心行为 |
|---|---|---|---|---|
| **KlineAgent** | `agents/kline_agent.py` | `BranchResult` | `BranchVerdict` | 封装 K 线分支结果 |
| **QuantAgent** | `agents/quant_agent.py` | `BranchResult` | `BranchVerdict` | 封装量化因子分支结果 |
| **FundamentalAgent** | `agents/fundamental_agent.py` | `BranchResult` | `BranchVerdict` | 封装基本面分支结果 |
| **IntelligenceAgent** | `agents/intelligence_agent.py` | `BranchResult` | `BranchVerdict` | 封装情报分支结果 |
| **MacroAgent** | `agents/macro_agent.py` | `BranchResult` | `BranchVerdict` | 封装宏观分支结果 (market scope) |
| **RiskGuard** | `agents/risk_guard.py` | `branch_verdicts`, `constraints` | `RiskDecision` | 硬否决、暴露上限、仓位限制 |
| **ICCoordinator** | `agents/ic_coordinator.py` | `branch_verdicts`, `risk_decision` | `ICDecision` | 一致性协调，方向投票 |
| **PortfolioConstructor** | `agents/portfolio_constructor.py` | `ic_decisions`, `risk_limits` | `PortfolioPlan` | 确定性权重分配、换手约束 |
| **NarratorAgent** | `agents/narrator_agent.py` | `branch_summaries`, `ic_decisions`, `portfolio_plan` | `ReportBundle` | 只读渲染 Markdown 报告 |

### 审阅层 (LLM Advisory)

| Agent | 源文件 | 输入 | 输出 | 偏离上限 | 核心行为 |
|---|---|---|---|---|---|
| **KLineSubAgent** | `agents/subagents/kline_agent.py` | `KLineAgentInput` | `KLineAgentOutput` | ±0.25 | 技术分析审阅 |
| **QuantSubAgent** | `agents/subagents/quant_agent.py` | `QuantAgentInput` | `QuantAgentOutput` | ±0.25 | 因子分析审阅 |
| **FundamentalSubAgent** | `agents/subagents/fundamental_agent.py` | `FundamentalAgentInput` | `FundamentalAgentOutput` | ±0.35 | 基本面审阅 |
| **IntelligenceSubAgent** | `agents/subagents/intelligence_agent.py` | `IntelligenceAgentInput` | `IntelligenceAgentOutput` | ±0.30 | 情报审阅 |
| **MacroSubAgent** | `agents/subagents/macro_agent.py` | `MacroAgentInput` | `MacroAgentOutput` | ±0.25 | 宏观环境审阅 |
| **RiskSubAgent** | `agents/subagent.py` | `RiskAgentInput` | `RiskAgentOutput` | — | 组合级风险评估 |
| **MasterAgent** | `agents/master_agent.py` | `MasterAgentInput` | `MasterAgentOutput` | ±0.30 | IC 主席综合辩论 |

---

## 3. 控制层详解

### 3.1 BaseAgent 基类

**源文件**: `agents/base.py`

所有控制层 Agent 的公共基类，提供 `run(payload) -> Any` 抽象接口与工具方法：

| 方法 | 功能 |
|---|---|
| `score_to_direction(score)` | `≥0.15` → bullish, `≤-0.15` → bearish, 否则 neutral |
| `score_to_action(score)` | `≥0.25` → buy, `≤-0.35` → sell, 否则 hold |
| `confidence_to_label(conf)` | `≥0.85` very_high, `≥0.65` high, `≥0.4` medium, `≥0.2` low, 否则 very_low |
| `more_restrictive_action(a, b)` | 取优先级更低（更保守）的 action |
| `clamp_action_to_cap(action, cap)` | 将 action 约束在 action_cap 以内 |
| `branch_result_to_verdict(br)` | 将 `BranchResult` 映射为 `BranchVerdict`，自动分桶 notes |
| `partition_bucket_notes(items)` | 将文本分为 investment_risks / coverage_notes / diagnostic_notes |

### 3.2 五路分支 Agent

KlineAgent、QuantAgent、FundamentalAgent、IntelligenceAgent、MacroAgent 各自将对应分支的 `BranchResult` 封装为协议层 `BranchVerdict`。MacroAgent 的 scope 为 `CoverageScope.MARKET`（市场级别），其余为 `CoverageScope.BRANCH`。

### 3.3 RiskGuard

**源文件**: `agents/risk_guard.py`
**输入**: `branch_verdicts`, `portfolio_state`, `constraints`
**输出**: `RiskDecision`

确定性风控引擎，施加硬约束：

| 触发条件 | 约束动作 |
|---|---|
| `force_veto` 或检测到否决关键词 | 硬否决: action_cap 收紧, gross_cap→0, blocked 全部候选标的 |
| 宏观 score ≤ -0.2 | action_cap → HOLD, gross_cap ≤ 50%, max_weight ≤ 10% |
| risk_texts ≥ 3 条 | action_cap → HOLD, gross_cap ≤ 60%, max_weight ≤ 12% |

**否决关键词**: `fraud`, `halt`, `delist`, `hard veto`, `veto`, `liquidity freeze`

**RiskLevel 推断**:

| RiskLevel | 条件 |
|---|---|
| EXTREME | veto 或 gross_cap ≤ 10% |
| HIGH | gross_cap ≤ 40% 或 risk_count ≥ 3 |
| MEDIUM | gross_cap ≤ 75% 或 risk_count ≥ 1 |
| LOW | 其余 |

### 3.4 ICCoordinator

**源文件**: `agents/ic_coordinator.py`
**输入**: `branch_verdicts`, `risk_decision`, 可选 `ic_hints`
**输出**: `ICDecision`

基于结构化分支结论做一致性协调，**禁止直接读取原始市场数据**（运行时强制检查 `market_snapshot`, `symbol_data`, `ohlcv` 等字段）。

核心逻辑：
- 计算分支 score 均值，若有 LLM hint 则 70/30 混合
- Action 受 `RiskDecision.action_cap` 硬约束
- 构建 `agreement_points`（≥2 分支同方向）和 `conflict_points`（多空对冲或降级分支）

### 3.5 PortfolioConstructor

**源文件**: `agents/portfolio_constructor.py`
**输入**: `ic_decisions`, `macro_verdict`, `risk_limits`, `existing_portfolio`, `tradability_snapshot`
**输出**: `PortfolioPlan`

确定性组合构建，核心约束：
- **权重计算**: `strength = max(score, 0) × (0.5 + 0.5 × confidence) × action_multiplier`
- **Action multiplier**: BUY=1.0, HOLD=0.6, 其余=0
- **多级约束**: symbol cap、sector cap、gross_exposure_cap、turnover_cap
- **Turnover 平滑**: 超限时对 baseline 与 target 做线性混合
- **浓度指标**: top1/3/5 weight、HHI、effective N、max sector weight
- **不可变**: NarratorAgent 与 IC thesis 不可改写 target_weight

### 3.6 NarratorAgent

**源文件**: `agents/narrator_agent.py`
**输入**: `macro_verdict`, `branch_summaries`, `ic_decisions`, `portfolio_plan`, `run_diagnostics`
**输出**: `ReportBundle`

只读报告渲染器，不访问任何市场数据。委托：
- `ExecutiveSummaryBuilder` — 摘要段落
- `ConclusionRenderer` — 市场观点、分支结论、个股卡片、Markdown 报告
- `DiagnosticsBucketizer` — 覆盖率与诊断分桶

---

## 4. 审阅层详解

### 4.1 LLMClient

**源文件**: `agents/llm_client.py`

统一异步 LLM 客户端（基于 aiohttp），支持多 Provider：

| Provider | 模型前缀 | 环境变量 |
|---|---|---|
| OpenAI | `gpt-`, `o1-`, `o3-`, `o4-` | `OPENAI_API_KEY` |
| Anthropic | `claude-` | `ANTHROPIC_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Google | `gemini` | `GOOGLE_API_KEY` |

`has_any_provider()` 检查是否有可用 API key。无可用 Provider 时编排器自动回退至纯算法模式。

### 4.2 BranchSubAgent

**源文件**: `agents/subagent.py`（基类）、`agents/subagents/*.py`（特化子类）

`analyze(BranchAgentInput) -> BranchAgentOutput`

LLM 审阅各分支的量化结果，输出 conviction 评分。所有输出受偏离上限硬约束：

| 分支 | 偏离上限 | 说明 |
|---|---|---|
| kline | ±0.25 | 技术面短期信号，偏离空间较小 |
| quant | ±0.25 | 因子模型相对稳定 |
| fundamental | ±0.35 | 长期视角，允许更大偏离 |
| intelligence | ±0.30 | 事件驱动，中等弹性 |
| macro | ±0.25 | 宏观环境相对确定 |

**Conviction 映射**:
- `≥0.5` → strong_buy
- `≥0.15` → buy
- `≤-0.5` → strong_sell
- `≤-0.15` → sell
- 其余 → neutral

**特化 SubAgent**:
- `KLineSubAgent`: 趋势线、K 线形态、LSTM/Chronos 可靠性、多周期对齐
- `QuantSubAgent`: 多因子模型、Alpha 挖掘、因子衰减、regime 适配
- `FundamentalSubAgent`: 财务质量(28%)、预测修订(18%)、估值(18%)、治理(14%)、持股(12%)、文档情绪(10%)
- `IntelligenceSubAgent`: 事件风险、恐贪情绪、资金流、宽度轮动、逆向信号
- `MacroSubAgent`: 流动性、波动率结构、宽度、动量对齐、跨资产联动

### 4.3 RiskSubAgent

**源文件**: `agents/subagent.py`

`analyze(RiskAgentInput) -> RiskAgentOutput`

输出组合级风险评估：

| 字段 | 说明 |
|---|---|
| `risk_assessment` | `acceptable` / `elevated` / `high` / `extreme` |
| `max_recommended_exposure` | [0.0, 1.0]，建议最大暴露 |
| `tail_risk_assessment` | 尾部风险定性判断 |
| `correlation_breakdown_risk` | 相关性崩溃风险 |
| `drawdown_scenario` | 回撤情景分析 |

Advisory only — 不覆盖 RiskGuard 的硬否决。

### 4.4 MasterAgent (IC 主席)

**源文件**: `agents/master_agent.py`

`deliberate(MasterAgentInput) -> MasterAgentOutput`

模拟投资委员会辩论，综合 5 路分支报告 + 风控报告：

- **偏离上限**: final_score 距 ensemble baseline ±0.30
- **风险否决**: risk_assessment == "extreme" → conviction 封顶 neutral, score ≤ 0.1
- **输出结构**:
  - `final_conviction` / `final_score` / `confidence`
  - `consensus_areas` — 共识点
  - `disagreement_areas` — 分歧点
  - `debate_resolution` — 分歧调解
  - `dissenting_views` — 保留少数派意见
  - `top_picks` — 标的推荐 (symbol, action, conviction, rationale, target_weight)
  - `portfolio_narrative` — 3-5 句投资论点因果链

---

## 5. 编排流程

### 5.1 AgentOrchestrator

**源文件**: `agents/orchestrator.py`

`enhance()` 方法执行三阶段异步编排：

```
Phase 1 ─── 5 × BranchSubAgent 并发执行 ──► 收集成功的分支报告
                │
Phase 2 ─── RiskSubAgent 串行执行 ──────────► RiskAgentOutput
                │
Phase 3 ─── MasterAgent 串行执行 ───────────► MasterAgentOutput
                │                                  (需至少 1 个分支成功)
                ▼
        AgentEnhancedStrategy
        ├─ algorithmic_strategy (原始算法输出)
        ├─ agent_strategy (MasterAgentOutput | None)
        ├─ branch_agent_outputs (各分支审阅结果)
        ├─ risk_agent_output (风控审阅结果)
        ├─ agent_layer_success (bool)
        ├─ agent_layer_timings (各阶段耗时)
        └─ fallback_used (bool, 无 Provider 时为 True)
```

`enhance_sync()` 提供同步包装，兼容无事件循环的调用场景。

### 5.2 超时预算

| 组件 | 默认值 | 说明 |
|---|---|---|
| 分支 SubAgent | 15s | kline 享有 1.5× = 22.5s |
| 分支外层等待 | timeout + 5s | asyncio.wait_for 安全余量 |
| RiskSubAgent | 15s | — |
| MasterAgent | 30s | — |
| Master 外层等待 | timeout + 10s | — |
| 总超时 | 120s | — |
| max_tokens (分支) | 800 | kline 限制为 600 |
| max_tokens (Master) | 1500 | — |

---

## 6. 协议层合约

### 6.1 枚举类型

| 枚举 | 值 | 定义位置 |
|---|---|---|
| `AgentStatus` | `success`, `degraded`, `vetoed` | `agent_protocol.py` |
| `Direction` | `bullish`, `bearish`, `neutral` | `agent_protocol.py` |
| `ActionLabel` | `buy`, `hold`, `sell`, `watch`, `avoid` | `agent_protocol.py` |
| `ConfidenceLabel` | `very_high`, `high`, `medium`, `low`, `very_low` | `agent_protocol.py` |
| `CoverageScope` | `branch`, `symbol`, `market`, `portfolio` | `agent_protocol.py` |
| `RiskLevel` | `low`, `medium`, `high`, `extreme` | `agent_protocol.py` |

### 6.2 控制层协议 (Dataclasses)

**定义位置**: `agent_protocol.py`

| 类型 | 关键字段 | 版本标记 |
|---|---|---|
| `BranchVerdict` | agent_name, thesis, direction, action, final_score, final_confidence, evidence, investment_risks | architecture + branch_schema |
| `RiskDecision` | hard_veto, action_cap, max_weight, gross_exposure_cap, blocked_symbols, position_limits | + ic_protocol |
| `ICDecision` | thesis, direction, action, final_score, agreement_points, conflict_points, rationale_points | + ic_protocol |
| `PortfolioPlan` | target_weights, target_positions, position_limits, blocked_symbols, concentration_metrics, turnover_estimate | + ic_protocol |
| `ReportBundle` | headline, summary, markdown_report, executive_summary, branch_verdicts, ic_decisions, portfolio_plan | + report_protocol |

辅助类型: `EventNote`, `EvidenceItem`

### 6.3 审阅层合约 (Pydantic)

**定义位置**: `agents/agent_contracts.py`

所有模型继承 `_CompatModel`（`extra="allow"`，支持向前兼容）。

**通用合约**:

| 类型 | 关键字段 |
|---|---|
| `BaseBranchAgentInput` | branch_name, base_score, final_score, confidence, evidence_summary, bull/bear/risk_points, symbol_scores, market_regime, branch_signals |
| `BaseBranchAgentOutput` | branch_name, conviction, conviction_score, confidence, key_insights, risk_flags, disagreements_with_algo, symbol_views |
| `RiskAgentInput` | risk_metrics_summary, regime, position_sizing, branch_agent_summaries, portfolio_level_risks |
| `RiskAgentOutput` | risk_assessment, max_recommended_exposure, tail_risk_assessment, correlation_breakdown_risk, drawdown_scenario |
| `MasterAgentInput` | branch_reports, risk_report, ensemble_baseline, market_regime, candidate_symbols |
| `MasterAgentOutput` | final_conviction, final_score, confidence, consensus_areas, disagreement_areas, top_picks, portfolio_narrative, dissenting_views |
| `AgentEnhancedStrategy` | algorithmic_strategy, agent_strategy, branch_agent_outputs, risk_agent_output, agent_layer_success, fallback_used |

**分支特化输入/输出**: `KLineAgentInput/Output`, `QuantAgentInput/Output`, `FundamentalAgentInput/Output`, `IntelligenceAgentInput/Output`, `MacroAgentInput/Output` — 各自添加领域特定字段。

---

## 7. 数据流图

```
BranchResult (5路算法输出)
    │
    ├──► BranchAgent ──► BranchVerdict ──┬──► RiskGuard ──► RiskDecision
    │                                    │                       │
    │                                    │            ┌──────────┘
    │                                    │            ▼
    │                                    ├──► ICCoordinator ──► ICDecision
    │                                    │       ▲                  │
    │                                    │       │ ic_hints         │
    └──► BranchSubAgent ──► BranchAgentOutput    │                  │
              │                                  │                  ▼
              │         RiskSubAgent ──► RiskAgentOutput    PortfolioConstructor
              │              │                                      │
              │              ▼                                      ▼
              └───► MasterAgent ──► MasterAgentOutput ─┘     PortfolioPlan
                                                                    │
                                                                    ▼
                                                            NarratorAgent
                                                                    │
                                                                    ▼
                                                             ReportBundle
```

**关键交汇点**: MasterAgent 的输出可作为 `ic_hints` 传入 ICCoordinator，实现审阅层对控制层的 advisory 影响（70/30 混合权重）。

---

## 8. 分支权重

| 分支 | 权重 | 研究领域 |
|---|---|---|
| kline | 22% | K 线技术分析、趋势识别、时间序列建模 (Kronos/Chronos) |
| quant | 28% | 多因子模型、Alpha158 特征、ML 信号 |
| fundamental | 15% | 企业基本面、盈利质量、估值分析 |
| intelligence | 20% | 新闻情绪、事件驱动、资金流、催化剂识别 |
| macro | 15% | 宏观经济指标、regime 检测、跨资产联动 |

分支顺序: `kline → quant → fundamental → intelligence → macro`

---

## 9. 安全不变量

以下约束为系统核心设计，任何修改都需要架构级评审：

1. **审阅层 advisory-only**: LLM 层永远不覆盖控制层的硬否决（`RiskGuard.hard_veto`）
2. **ICCoordinator 禁止原始数据**: 运行时检查并拒绝 `market_snapshot`, `symbol_data`, `ohlcv` 等字段
3. **NarratorAgent 只读**: 不访问市场数据，不修改任何决策输出
4. **Conviction 偏离硬约束**: 所有 SubAgent 输出受分支级偏离上限钳制，MasterAgent 受 ±0.30 钳制
5. **风险否决传导**: risk_assessment == "extreme" → MasterAgent conviction 封顶 neutral, score ≤ 0.1
6. **版本戳追踪**: 所有协议输出携带 architecture_version、branch_schema_version、ic_protocol_version

---

## 10. 系统提示角色摘要

| Agent | 角色定位 | 核心专注领域 |
|---|---|---|
| KLineSubAgent | 资深技术分析专家 | 趋势线、K 线形态、LSTM/Chronos 可靠性、多周期对齐 |
| QuantSubAgent | 量化因子研究专家 | 因子暴露、Alpha mining、因子衰减、regime 适配 |
| FundamentalSubAgent | 资深基本面分析师 | 财务质量、估值合理性、公司治理、持股结构 |
| IntelligenceSubAgent | 多维信息情报分析师 | 事件风险、恐贪情绪、资金流向、宽度轮动、逆向信号 |
| MacroSubAgent | 宏观策略师 | 流动性环境、波动率结构、市场宽度、跨资产联动 |
| RiskSubAgent | 首席风控官 | VaR/CVaR、尾部风险、仓位集中度、压力测试、回撤情景 |
| MasterAgent | 投资委员会主席 | 综合辩论、共识/分歧调解、保留少数派意见、投资论点因果链 |

提示模板定义于 `agents/prompts.py`，包含 `BRANCH_SYSTEM_PROMPTS`、`RISK_SYSTEM_PROMPT`、`MASTER_SYSTEM_PROMPT`。

---

## 11. 文件索引

| 文件 | 说明 |
|---|---|
| `agents/__init__.py` | V10 多智能体层公共导出 |
| `agents/base.py` | BaseAgent 基类，工具方法集 |
| `agents/agent_contracts.py` | 审阅层 Pydantic I/O 合约 |
| `agents/llm_client.py` | 多 Provider 异步 LLM 客户端 |
| `agents/prompts.py` | 系统提示模板、偏离上限、消息构建器 |
| `agents/subagent.py` | BranchSubAgent、RiskSubAgent 基类 |
| `agents/master_agent.py` | MasterAgent (IC 主席) |
| `agents/orchestrator.py` | AgentOrchestrator 三阶段编排 |
| `agents/kline_agent.py` | KlineAgent (控制层) |
| `agents/quant_agent.py` | QuantAgent (控制层) |
| `agents/fundamental_agent.py` | FundamentalAgent (控制层) |
| `agents/intelligence_agent.py` | IntelligenceAgent (控制层) |
| `agents/macro_agent.py` | MacroAgent (控制层) |
| `agents/risk_guard.py` | RiskGuard 硬否决引擎 |
| `agents/ic_coordinator.py` | ICCoordinator 一致性协调 |
| `agents/portfolio_constructor.py` | PortfolioConstructor 确定性组合构建 |
| `agents/narrator_agent.py` | NarratorAgent 只读报告渲染 |
| `agents/subagents/kline_agent.py` | KLineSubAgent (审阅层) |
| `agents/subagents/quant_agent.py` | QuantSubAgent (审阅层) |
| `agents/subagents/fundamental_agent.py` | FundamentalSubAgent (审阅层) |
| `agents/subagents/intelligence_agent.py` | IntelligenceSubAgent (审阅层) |
| `agents/subagents/macro_agent.py` | MacroSubAgent (审阅层) |
| `agents/subagents/risk_agent.py` | SpecializedRiskSubAgent (审阅层) |
| `agent_protocol.py` | 协议层 dataclass 定义 |
| `agent_orchestrator.py` | 备选单链编排策略 |
| `versioning.py` | 版本号、分支顺序、分支权重 |
