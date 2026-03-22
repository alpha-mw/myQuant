# Quant-Investor V8.0

<div align="center">

**五路并行研究主线 · 可信度治理 · 风控层 · 组合级集成裁判**

[![Version](https://img.shields.io/badge/Version-8.0.0-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 项目简介

**Quant-Investor 当前官方主线是 V8.0。**

V8 主线采用 `数据层 -> 五路并行研究分支 -> 风控层 -> 集成裁判层` 的组合级研究架构，重点解决三件事：

1. 统一数据包和分支契约，保证五个研究分支可以并行运行。
2. 在分支失败、数据降级、模拟数据回退时，保留 `reliability`、`branch_mode`、`provenance` 等可信度语义。
3. 输出面向组合而不是单点评分的 `PortfolioStrategy`，把仓位、候选标的、仓位上限和交易建议统一到一条主链里。

旧版和实验性版本不再保留在当前运行主目录；如需追溯，请查看 git 历史。

---

## 当前官方入口

- CLI 主入口：`quant-investor`
- Python API 主入口：`from quant_investor import QuantInvestorV8`
- 对外稳定契约：`UnifiedDataBundle`、`BranchResult`、`ResearchPipelineResult`、`PortfolioStrategy`
- 当前包版本：`8.0.0`

---

## 系统架构

```text
┌─────────────────────────────────────────────────────────────────┐
│                        数据层 Data Layer                        │
│  UnifiedDataBundle: OHLCV / 基本面 / 事件 / 情绪 / 宏观 / 来源元数据 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│ Branch A        │  │ Branch B        │  │ Branch C        │
│ kline / kronos  │  │ quant           │  │ llm_debate      │
│ K线趋势与收益判断 │  │ Alpha挖掘优先     │  │ 结构化多空辩论     │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                     │                     │
┌────────▼────────┐  ┌────────▼────────┐
│ Branch D        │  │ Branch E        │
│ intelligence    │  │ macro           │
│ 财务/事件/情绪融合 │  │ 宏观风险状态      │
└────────┬────────┘  └────────┬────────┘
         └──────────────┬──────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                      风控层 Risk Layer                          │
│  风险指标 / 仓位约束 / 现金比例 / 个股上限 / 组合优化             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                   集成裁判层 Ensemble Layer                     │
│  SignalCalibrator -> 自适应分支权重 -> macro overlay -> quorum │
│  -> PortfolioStrategy / TradeRecommendation                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 系统投资逻辑

### 1. 数据层先构造统一数据包

主线先构造 `UnifiedDataBundle`，其中不仅包含价格、基本面、事件和情绪，还会记录每只股票的来源元数据，例如：

- `data_source_status`
- `is_synthetic`
- `degraded_reason`
- `branch_mode`
- `reliability`

这一步的目标不是“尽量假装数据完整”，而是把真实数据、降级数据和模拟数据严格区分开。

### 2. 五个研究分支并行运行

V8 当前五个分支为：

| 分支 | 标识 | 当前职责 |
|:---|:---|:---|
| K 线分支 | `kline` | 使用 `heuristic / kronos / chronos` 后端输出趋势与预期收益 |
| 传统量化分支 | `quant` | Alpha 挖掘优先，失败时回退经典动量/低波/技术/量能因子 |
| 多空辩论分支 | `llm_debate` | 基于基本面、事件和情绪生成结构化多空论据 |
| 多维智能分支 | `intelligence` | 财务质量、事件风险、情绪、资金流和广度融合 |
| 宏观分支 | `macro` | 多市场宏观风控终端，失败时回退统计型宏观估计 |

每个分支统一输出 `BranchResult`，并保留自己的 `score`、`confidence`、`signals`、`risks`、`metadata`。

### 3. 统一量纲与可信度校准

V8 不依赖 README 里的固定手工权重表。当前主线会先把各分支结果映射成统一的 `CalibratedBranchSignal`，再在组合层做聚合：

- 统一预期收益尺度
- 统一分支 horizon
- 基于 `branch_mode` 和 `reliability` 做分支可信度折减
- 基于标的 provenance 对模拟数据和降级数据做个股级可信度折减

### 4. 风控与组合约束优先于交易建议

组合生成前会先做风险约束：

- 风险层输出现金比例、仓位约束和组合优化结果
- 宏观信号以 `macro overlay` 的形式作用于总仓位和总预期收益
- 分支成功数不足时，触发 `quorum` 约束
- 真实候选标的不足时，进入 `research_only` 或 `degraded` 模式

当前主线不是“宏观分数低于某条线就直接全部清仓”的单一规则，而是通过 `macro overlay + quorum + provenance + risk_level` 共同决定最终仓位。

### 5. 最终输出是组合级策略

集成裁判层最终输出 `PortfolioStrategy`，核心字段包括：

- `target_exposure`
- `candidate_symbols`
- `position_limits`
- `branch_consensus`
- `risk_summary`
- `provenance_summary`
- `research_mode`
- `trade_recommendations`

也就是说，V8 的终点不是单个分支的“买入/卖出标签”，而是带可信度与降级语义的组合级策略。

---

## 可信度与降级治理

V8 主线要求以下语义在报告和策略中持续保留：

- `research_only`: 分支成功数严重不足或真实候选标的不存在时，只保留研究参考，不生成可执行交易
- `degraded`: 存在分支失败、数据降级或模拟数据时，允许输出策略，但必须标记为降级结果
- `synthetic_symbols`: 使用模拟数据回退的标的
- `branch_mode`: 分支当前运行模式，例如 `kline_heuristic`、`alpha_research`、`macro_terminal`
- `reliability`: 分支级或标的级可信度
- `provenance_summary`: 统一记录模拟标的、降级标的、分支模式、分支可靠度和宏观 overlay

这部分是 V8 主线的重要设计点，不应在文档里被弱化或隐藏。

---

## 核心契约

```python
@dataclass
class UnifiedDataBundle:
    market: str
    symbols: list[str]
    symbol_data: dict[str, pd.DataFrame]
    fundamentals: dict[str, dict[str, Any]]
    event_data: dict[str, list[dict[str, Any]]]
    sentiment_data: dict[str, dict[str, Any]]
    macro_data: dict[str, Any]


@dataclass
class BranchResult:
    branch_name: str
    score: float
    confidence: float
    signals: dict[str, Any]
    risks: list[str]
    explanation: str
    symbol_scores: dict[str, float]


@dataclass
class PortfolioStrategy:
    target_exposure: float
    style_bias: str
    candidate_symbols: list[str]
    position_limits: dict[str, float]
    stop_loss_policy: dict[str, Any]
```

完整字段请以 `quant_investor/contracts.py` 为准。

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

# 可选 LLM / 扩展能力
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="your-gemini-key"
```

### V8 五路并行分析

```bash
quant-investor research run \
    --stocks 000001.SZ 600519.SH 000858.SZ \
    --market CN \
    --capital 1000000 \
    --risk 中等 \
    --output report_v8.md
```

常用开关：

- `--no-macro`
- `--no-kline` / `--no-kronos`
- `--no-quant`
- `--no-intelligence`
- `--no-llm-debate`
- `--kline-backend heuristic|kronos|chronos`
- `--allow-synthetic-for-research`

### Python API

```python
from quant_investor import QuantInvestorV8

analyzer = QuantInvestorV8(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    risk_level="中等",
    enable_kronos=True,
    enable_quant=True,
    enable_intelligence=True,
    enable_llm_debate=True,
    enable_macro=True,
    verbose=True,
)

result = analyzer.run()
print(result.final_report)
```

### 定向测试

```bash
pytest tests/unit/test_parallel_research_pipeline.py -v
pytest tests/unit/test_risk_management.py -v
pytest tests/unit/test_backtest.py -v
```

---

## Web 应用架构

### 后端（FastAPI）

```
web/
├── main.py                # Uvicorn 入口（uvicorn web.main:app）
├── app.py                 # FastAPI 应用工厂，CORS/路由/静态文件
├── config.py              # 后端配置（从 .env 读取）
├── api/
│   ├── analysis.py        # POST /api/v1/analysis — 触发分析任务
│   ├── data.py            # GET  /api/v1/data    — 行情/财务数据
│   ├── portfolio.py       # GET/POST /api/v1/portfolio — 组合管理
│   └── settings.py        # GET/PUT  /api/v1/settings  — 用户配置
├── services/              # 业务逻辑层
│   ├── analysis_service.py
│   ├── data_service.py
│   ├── portfolio_service.py
│   └── settings_service.py
├── tasks/
│   └── run_analysis_job.py  # 异步后台分析任务
└── db/                    # SQLite 数据层
```

### 前端（React + TypeScript）

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Dashboard.tsx        # 主仪表盘
│   │   ├── AnalysisHub.tsx      # 分析中心（发起/管理分析）
│   │   ├── StockDetail.tsx      # 个股详情页
│   │   ├── Watchlists.tsx       # 自选股管理
│   │   ├── DataExplorer.tsx     # 数据探索器
│   │   ├── AnalysisHistory.tsx  # 历史分析记录
│   │   ├── MarketStatus.tsx     # 市场状态监控
│   │   ├── RegimeMonitor.tsx    # 市场状态（Regime）实时监控
│   │   └── SettingsPage.tsx     # 系统设置
│   ├── components/              # 可复用 UI 组件
│   ├── api/                     # 前端 API 客户端
│   └── types/                   # TypeScript 类型定义
├── vite.config.ts               # Vite 构建配置（代理 → 后端:8000）
└── package.json                 # 依赖：React 19 / TailwindCSS / Recharts
```

**前端技术栈**：React 19 · TypeScript · Vite · TailwindCSS · Zustand · React Query · Recharts · Lightweight Charts

---

## 代码结构

```text
myQuant/
├── README.md
├── pyproject.toml
├── requirements.txt
├── quant_investor/
│   ├── __init__.py
│   ├── contracts.py                         # 稳定契约定义
│   ├── pipeline/                           # V8 主线编排
│   ├── data/                               # 统一数据层
│   ├── market/                             # 下载 / 批量分析 / 回测统一入口
│   ├── cli/                                # 单一 CLI 子命令入口
│   ├── macro_terminal_tushare.py           # 宏观风控终端
│   └── MACRO_RISK_GUIDE.md                 # 宏观分支说明
├── tests/
│   └── unit/
├── web/                                      # 可选 Web/API 层
└── frontend/                                 # 可选前端层
```

`web/` 和 `frontend/` 存在于仓库中，但**不属于当前官方研究主线的事实基准**。当前主线以 CLI、Python API、分支契约和单元测试为准。

---

## 宏观模块说明

宏观分支的当前实现入口是：

- `quant_investor/macro_terminal_tushare.py`
- `quant_investor/MACRO_RISK_GUIDE.md`

当前工厂函数支持 `CN` 和 `US`。如果要扩展 `HK` 或其他市场，需要先扩展实现，再更新文档。

---

## 版本说明

| 状态 | 版本 | 说明 |
|:---|:---|:---|
| **当前官方主线** | **V8.0** | 五路并行研究、可信度治理、风控层、组合级集成裁判 |
| 历史稳定版本 | V7.x / V6.x | 旧版串行或分层架构，仅保留于 git 历史 |
| 历史/实验版本 | git 历史 | 已从当前主目录清理，不作为当前官方发布线 |

---

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。量化模型、宏观规则和回测结果均不保证未来收益，投资有风险，入市需谨慎。
