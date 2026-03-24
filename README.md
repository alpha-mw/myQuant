# Quant-Investor

<div align="center">

**V9 current stable · V8 legacy frozen · Multi-Agent Protocol · branch-local review · RiskGuard · PortfolioConstructor · Narrator**

[![Version](https://img.shields.io/badge/Version-9.0.0-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 项目简介

**Quant-Investor 当前官方稳定主线是 V9.0，V8 已冻结为 legacy frozen；多 agent 协议层已落地，但不会偷换 V8 语义。**

当前稳定主线采用：

`数据层 -> 五路并行研究分支 -> RiskGuard -> ICCoordinator -> PortfolioConstructor -> NarratorAgent`

这次版本升级的核心不是“再加一个 LLM 分支”，而是把原独立 `llm_debate` 顶层身份下沉为所有分支共用的 `branch-local debate engine`，同时新增 `fundamental` 一级分支来承担公司基本面研究。

当前对外稳定入口：

- Python API 默认稳定入口：`from quant_investor import QuantInvestor`
- Python API 显式当前入口：`from quant_investor import QuantInvestorV9`
- pipeline 显式当前入口：`from quant_investor.pipeline.current import QuantInvestor`
- legacy frozen API：`from quant_investor import QuantInvestorV8`
- 显式最新版本入口：`from quant_investor import QuantInvestorLatest`
- CLI：建议显式传 `--architecture v9|v10|v8`，避免把 `latest` 误当作稳定默认线
- 稳定契约：`UnifiedDataBundle`、`BranchResult`、`ResearchPipelineResult`、`PortfolioStrategy`

版本语义：

- `QuantInvestor` / `quant_investor.pipeline.current.QuantInvestor` / `QuantInvestorV9`：current stable architecture
- `QuantInvestorV8` / `--architecture v8`：legacy frozen，不指向 V9 实现
- `QuantInvestorV9` / `--architecture v9`：current architecture
- `QuantInvestorLatest` / `--architecture latest`：显式 latest 版本列车，当前等于 V10，但不是稳定默认入口

---

## 多 Agent 架构分工

当前架构是多 agent 投资研究系统，不是自由聊天式 multi-agent demo。角色边界如下：

- Research Agents：`KlineAgent`、`QuantAgent`、`FundamentalAgent`、`IntelligenceAgent`、`MacroAgent`
- Control Agents：`RiskGuard`、`ICCoordinator`、`PortfolioConstructor`、`NarratorAgent`

full-market / shortlist 语义约束：

- full-market batch 的 `Kline` 主路径默认固定为 `heuristic fast-screen`
- `hybrid / chronos / kronos` 仅用于 shortlist / second-pass deep analysis
- full-market batch 汇总报告统一复用 `NarratorAgent -> ReportBundle`，不再维护独立 legacy batch renderer

关键控制职责：

- `RiskGuard`：负责 hard veto、动作上限与风险限额，IC 不得覆盖其硬约束
- `PortfolioConstructor`：只消费结构化 `ICDecision`、风险约束和可交易性快照，以 deterministic 规则生成仓位
- `NarratorAgent`：只负责解释和报告，不参与交易决策，不修改 action 或 weight

---

## 当前稳定架构图

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                              数据层 Data Layer                           │
│ UnifiedDataBundle: OHLCV / 基本面 / 事件 / 情绪 / 宏观 / provenance      │
│ + point-in-time snapshots / offline corporate semantic snapshots        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────┬──────────────┐
     │                          │                          │              │
┌────▼────┐  ┌───────────┐  ┌───▼──────────┐  ┌───────────▼───┐   ┌──────▼─────┐
│KlineAgent│ │QuantAgent │  │FundamentalAgent│ │IntelligenceAgent│   │ MacroAgent │
│快筛/深模 │ │deterministic│ │财务/预测/估值 │ │新闻/事件/情绪/流│   │市场级一次性│
└────┬────┘  └─────┬─────┘  └──────┬───────┘  └─────────┬──────┘   └──────┬─────┘
     └─────────────┴───────────────┴────────────────────┴──────────────┘
                                   │
                    BranchVerdict / structured evidence only
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                             RiskGuard 控制层                             │
│ hard veto / action_cap / gross_exposure_cap / max_weight                │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                           ICCoordinator 控制层                           │
│ agreement / conflict / action suggestion / one-line thesis              │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                       PortfolioConstructor 控制层                        │
│ deterministic weights / sector cap / liquidity / turnover              │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                           NarratorAgent 报告层                           │
│ executive summary / market view / branch conclusions / diagnostics      │
│ read-only, never writes back action or target_weight                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 五个研究分支

| 分支 | 标识 | 职责 |
|:---|:---|:---|
| K 线分支 | `kline` | K 线/Kronos/Chronos/趋势/波动/图形结构 |
| 传统量化分支 | `quant` | 因子、模型、拥挤度、稳定性、预期收益 |
| Fundamental 分支 | `fundamental` | 财务质量、盈利预测、估值、管理层、股东结构、离线文档语义 |
| 多维智能分支 | `intelligence` | 新闻、公告事件、情绪、资金流、市场广度、行业轮动 |
| 宏观分支 | `macro` | 宏观与总仓位 overlay |

所有分支统一遵循：

`raw model -> evidence packet -> branch-local debate -> branch fusion -> BranchResult`

---

## branch-local debate 规则

V9 中的 debate 不是独立分支，也不是第二套 ensemble。它只能在分支内部做有限修正：

- 修正 `confidence`
- 增加 `risk_flags`
- 做 bounded `score_adjustment`
- 触发 `caution / hard_veto`

不允许做的事：

- 从零生成交易信号
- 作为独立顶层分支参与第二次打分
- 把强多直接翻成强空，或把强空直接翻成强多

统一 JSON schema：

```json
{
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0,
  "score_adjustment": 0.0,
  "bull_points": [],
  "bear_points": [],
  "risk_flags": [],
  "unknowns": [],
  "used_features": [],
  "hard_veto": false
}
```

分支 adjustment 上限：

- `kline <= 0.10`
- `quant <= 0.10`
- `fundamental <= 0.20`
- `intelligence <= 0.15`
- `macro <= 0.10`

---

## 核心契约

V9 对 `BranchResult` 做了增量扩展，保持旧字段兼容：

```python
@dataclass
class BranchResult:
    branch_name: str
    score: float
    confidence: float
    base_score: float
    final_score: float
    base_confidence: float
    final_confidence: float
    horizon_days: int
    signals: dict[str, Any]
    risks: list[str]
    explanation: str
    symbol_scores: dict[str, float]
    evidence: EvidencePacket
    debate_verdict: DebateVerdict
    data_quality: dict[str, Any]
```

完整字段请以 `quant_investor/branch_contracts.py` 为准。旧代码继续从 `quant_investor/contracts.py` 导入也不会失效。

---

## 快速开始

### 环境配置

```bash
git clone https://github.com/alpha-mw/myQuant.git
cd myQuant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 必需环境变量

```bash
export TUSHARE_TOKEN="your-token"
```

可选能力：

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="your-gemini-key"
```

没有 LLM key、没有文档语义快照、没有 forecast provider 时，系统会平稳降级到 deterministic / neutral 结果，不会因为这些扩展能力缺失而终止主流程。

### CLI

```bash
quant-investor research run \
  --architecture v9 \
  --stocks 000001.SZ 600519.SH 000858.SZ \
  --market CN \
  --capital 1000000 \
  --risk 中等 \
  --debate-model gpt-5.4-mini \
  --output report_v9.md
```

常用开关：

- `--architecture v9|v10|latest|v8`
- `--no-kline`
- `--no-quant`
- `--no-fundamental`
- `--no-intelligence`
- `--no-macro`
- `--no-branch-debate`
- `--debate-top-k`
- `--debate-min-abs-score`
- `--debate-timeout-sec`
- `--debate-model`
- `--disable-document-semantics`
- `--allow-synthetic-for-research`
- `--no-agent-layer`
- `--agent-model`
- `--master-model`

兼容旧参数：

- `--no-kronos`
- `--no-llm-debate`

当运行 `--architecture latest` 或 `--architecture v10` 时，系统默认启动：

- `KLine SubAgent`：趋势识别、LSTM/Chronos 预测可靠性、经典形态验证
- `Quant SubAgent`：Alpha 衰减、因子拥挤度、regime 适配性
- `Fundamental SubAgent`：财务质量、估值合理性、管理层、股权结构
- `Intelligence SubAgent`：事件驱动、情绪极端值、资金流向异常
- `Macro SubAgent`：流动性环境、波动率结构、跨资产联动
- `Risk SubAgent`：VaR/CVaR 解读、尾部风险、仓位合理性
- `Master Agent IC`：综合辩论、共识提炼、分歧调解、最终决策

如果未显式传入 `--agent-model` / `--master-model`，系统会根据当前已配置的 provider 自动选择默认模型；如果没有可用 provider，Agent 层会自动降级跳过，不影响量化主流程。

### Python API

```python
from quant_investor import QuantInvestor

analyzer = QuantInvestor(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    risk_level="中等",
    enable_kline=True,
    enable_quant=True,
    enable_fundamental=True,
    enable_intelligence=True,
    enable_macro=True,
    enable_branch_debate=True,
    enable_document_semantics=True,
    debate_top_k=3,
    debate_min_abs_score=0.08,
    debate_timeout_sec=8.0,
    debate_model="gpt-5.4-mini",
    verbose=True,
)

result = analyzer.run()
print(result.final_report)
```

如果你需要显式绑定当前稳定实现，也可以直接导入：

```python
from quant_investor import QuantInvestorV9
```

冻结的 legacy 入口仍然可用：

```python
from quant_investor import QuantInvestorV8
```

显式 latest 入口：

```python
from quant_investor import QuantInvestorLatest
```

### 输出版本字段

当前稳定主线的输出对象与状态文件会携带以下字段，用于区分架构、分支 schema、IC 协议与报告协议：

- `architecture_version`
- `branch_schema_version`
- `ic_protocol_version`
- `report_protocol_version`

---

## V9 新配置项

### Pipeline / API

- `enable_fundamental`
- `enable_branch_debate`
- `debate_top_k`
- `debate_min_abs_score`
- `debate_timeout_sec`
- `debate_model`
- `enable_document_semantics`

### Data Layer Snapshot 接口

- `get_point_in_time_fundamental_snapshot(symbol, as_of)`
- `get_earnings_forecast_snapshot(symbol, as_of)`
- `get_management_snapshot(symbol, as_of)`
- `get_ownership_snapshot(symbol, as_of)`
- `get_document_semantic_snapshot(symbol, as_of)`

### 离线文档语义

V9 要求公司文档语义必须走离线 snapshot，不允许在主交易热路径里实时解析长文档：

- 读取：`quant_investor/corporate_doc_store.py`
- 写入：`quant_investor/corporate_doc_ingest.py`

---

## 代码结构

```text
myQuant/
├── README.md
├── pyproject.toml
├── quant_investor/
│   ├── branch_contracts.py
│   ├── contracts.py
│   ├── branch_debate_engine.py
│   ├── debate_templates.py
│   ├── fundamental_branch.py
│   ├── fundamental_components.py
│   ├── corporate_doc_store.py
│   ├── corporate_doc_ingest.py
│   ├── ensemble_judge.py
│   ├── enhanced_data_layer.py
│   ├── pipeline/
│   │   ├── parallel_research_pipeline.py
│   │   ├── quant_investor_v8.py
│   │   ├── quant_investor_v9.py
│   │   ├── quant_investor_v10.py
│   │   └── legacy_v8_pipeline.py
│   ├── agents/
│   ├── data/
│   ├── market/
│   ├── cli/
│   └── macro_terminal_tushare.py
├── tests/
│   └── unit/
├── web/        # 可选 Web/API 层
└── frontend/   # 可选前端层
```

说明：

- `quant_investor/pipeline/quant_investor_v8.py` 是 legacy frozen 入口
- `quant_investor/pipeline/quant_investor_v9.py` 是 current architecture 入口
- `quant_investor/pipeline/quant_investor_v10.py` 是 Multi-Agent IC 入口
- `web/` 与 `frontend/` 不属于当前研究主线的事实基准

---

## 验证

推荐的定向测试：

```bash
pytest tests/unit/test_parallel_research_pipeline.py -v
pytest tests/unit/test_data_layer.py -v
pytest tests/unit/test_full_market_batch_reports.py -v
```

如果环境缺少 `pytest` 或部分科学计算依赖，可先运行最小手工验证脚本，确认：

- branch order 已切到 `fundamental`
- 无 provider 时 debate 自动降级
- score adjustment 有上限
- Fundamental 分支在无文档数据时仍可运行
- macro debate 只执行一次市场级逻辑

---

## V8 -> V9 迁移

迁移细节见 [docs/V8_TO_V9_MIGRATION.md](docs/V8_TO_V9_MIGRATION.md)。
多 agent 协议与报告迁移见 [docs/MULTI_AGENT_PROTOCOL_MIGRATION.md](docs/MULTI_AGENT_PROTOCOL_MIGRATION.md)。

### 架构变化

- 顶层分支从 `kline / quant / llm_debate / intelligence / macro` 迁移为 `kline / quant / fundamental / intelligence / macro`
- 原 `llm_debate` 不再是独立分支，而是所有分支共用的 `branch-local debate engine`
- `fundamental` 接管财务质量、盈利预测、估值、管理层、股东结构、离线文档语义
- `intelligence` 不再承担主财务计分

### 接口变化

- 新增 `BranchResult.base_score / final_score / base_confidence / final_confidence`
- 新增 `BranchResult.evidence / debate_verdict / data_quality`
- 新增 `EnhancedDataLayer` 五个 snapshot 接口
- 新增 `EnsembleJudge`
- 新增 CLI/API 参数 `enable_fundamental`、`enable_branch_debate`、`enable_document_semantics`

### 向后兼容

- `QuantInvestor` 固定指向当前稳定主线 `QuantInvestorV9`
- `QuantInvestorV8` 仍可导入，但保持 legacy frozen，不再指向 V9
- `QuantInvestorV9` 与 `QuantInvestorV8` 保持不同实现，不共享同一入口类
- `QuantInvestorLatest` 指向 `QuantInvestorV10`，仅代表显式 latest，不代表稳定默认入口
- `enable_llm_debate` 在 V8 中保持 legacy 顶层分支语义，在 V9 中仅作为 `enable_branch_debate` 的兼容 alias
- `quant_investor/contracts.py` 仍可导入，但真实定义源迁移到了 `quant_investor/branch_contracts.py`
- 旧 `llm_debate` 历史统计只做归档保存，不再映射到 `fundamental`

## 最终迁移落点

- 旧 `branch pipeline` 仍可运行，但新的统一协议落点是 `BranchVerdict -> RiskDecision -> ICDecision -> PortfolioPlan -> ReportBundle`
- 旧 `llm_debate` 顶层分支已经迁移为分支内 `branch-local review` 和控制层 `ICCoordinator`
- 旧自由文本主报告已经迁移为严格分桶报告：`investment_risks` 正文、`coverage_notes` 数据覆盖摘要、`diagnostic_notes` 附录
- full-market 批量分析默认走当前稳定入口 `QuantInvestor`，并默认关闭 `branch_local_debate` 与 `document_semantics`，避免把 latest/V10 增强层和重型附加模块默认带入大批量分析
- full-market 批量分析的 K 线分支默认固定为 `heuristic fast-screen`，把 `hybrid / chronos` 深模型留给 shortlist / second-pass
- full-market 汇总 markdown 已统一从 `NarratorAgent -> ReportBundle` 生成，batch 与 single-symbol / shortlist 共用同一报告协议和分桶规则

---

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。量化模型、宏观规则和回测结果均不保证未来收益，投资有风险，入市需谨慎。
