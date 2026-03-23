# Quant-Investor

<div align="center">

**V9 current architecture · V8 legacy frozen · Fundamental Branch · branch-local debate · 风控层 · 组合级集成裁判**

[![Version](https://img.shields.io/badge/Version-9.0.0-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 项目简介

**Quant-Investor 当前官方主线是 V9.0，V8 已冻结为 legacy frozen。**

V9 主线采用：

`数据层 -> 五路并行研究分支 -> 风控层 -> 集成裁判层`

这次版本升级的核心不是“再加一个 LLM 分支”，而是把原独立 `llm_debate` 顶层身份下沉为所有分支共用的 `branch-local debate engine`，同时新增 `fundamental` 一级分支来承担公司基本面研究。

当前对外稳定入口：

- CLI：`quant-investor research run --architecture latest`
- Python API：`from quant_investor import QuantInvestorV9`
- 默认最新别名：`from quant_investor import QuantInvestorLatest`
- legacy frozen API：`from quant_investor import QuantInvestorV8`
- 稳定契约：`UnifiedDataBundle`、`BranchResult`、`ResearchPipelineResult`、`PortfolioStrategy`

版本语义：

- `QuantInvestorV8` / `--architecture v8`：legacy frozen，不指向 V9 实现
- `QuantInvestorV9` / `--architecture v9`：current architecture
- `QuantInvestorLatest` / `--architecture latest`：默认最新版本，当前等于 V9

---

## V9 架构图

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                              数据层 Data Layer                           │
│ UnifiedDataBundle: OHLCV / 基本面 / 事件 / 情绪 / 宏观 / provenance      │
│ + point-in-time snapshots / offline corporate semantic snapshots        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
      ┌─────────────────────────┼─────────────────────────┐
      │                         │                         │
┌─────▼─────┐             ┌─────▼─────┐             ┌─────▼─────┐
│ kline     │             │ quant     │             │ fundamental│
│ 趋势/波动 │             │ 因子/模型 │             │ 财务/预测/估值 │
└─────┬─────┘             └─────┬─────┘             └─────┬─────┘
      │                         │                         │
┌─────▼─────┐             ┌─────▼─────┐                  │
│intelligence│            │  macro     │                  │
│事件/情绪/流│            │宏观 overlay │                  │
└─────┬─────┘             └─────┬─────┘                  │
      └──────────────┬──────────┴──────────┬─────────────┘
                     │                     │
      raw model -> evidence packet -> branch-local debate -> branch fusion
                     │
┌────────────────────▼─────────────────────────────────────────────────────┐
│                              风控层 Risk Layer                           │
│ 风险指标 / 现金比例 / 个股上限 / 回撤约束 / 组合优化 / macro overlay     │
└────────────────────┬─────────────────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────────────────┐
│                          集成裁判层 Ensemble Layer                       │
│ final_score/final_confidence -> regime weights -> quorum -> strategy    │
│ -> PortfolioStrategy / TradeRecommendation                              │
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
  --architecture latest \
  --stocks 000001.SZ 600519.SH 000858.SZ \
  --market CN \
  --capital 1000000 \
  --risk 中等 \
  --debate-model gpt-5.4-mini \
  --output report_v9.md
```

常用开关：

- `--architecture latest|v9|v8`
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

兼容旧参数：

- `--no-kronos`
- `--no-llm-debate`

### Python API

```python
from quant_investor import QuantInvestorV9

analyzer = QuantInvestorV9(
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

冻结的 legacy 入口仍然可用：

```python
from quant_investor import QuantInvestorV8
```

默认最新入口：

```python
from quant_investor import QuantInvestorLatest
```

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
│   │   └── legacy_v8_pipeline.py
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

- `QuantInvestorV8` 仍可导入，但保持 legacy frozen，不再指向 V9
- `QuantInvestorLatest` 指向 `QuantInvestorV9`
- `enable_llm_debate` 在 V8 中保持 legacy 顶层分支语义，在 V9 中仅作为 `enable_branch_debate` 的兼容 alias
- `quant_investor/contracts.py` 仍可导入，但真实定义源迁移到了 `quant_investor/branch_contracts.py`
- 旧 `llm_debate` 历史统计只做归档保存，不再映射到 `fundamental`

---

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。量化模型、宏观规则和回测结果均不保证未来收益，投资有风险，入市需谨慎。
