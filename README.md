<div align="center">

<br/>

<img src="assets/logo.svg" alt="Quant-Investor" width="600"/>

# Quant-Investor

**三层多智能体量化投研系统**

*A deterministic research core with an optional LLM debate layer —*
*no hallucination reaches your portfolio.*

<br/>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-v13.0.0-FF6B35?style=flat-square)](https://github.com/alpha-mw/myQuant/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)

<br/>

[**快速开始**](#-快速开始) · [**架构设计**](#-架构) · [**API 参考**](#-api) · [**文档**](#-文档)

<br/>

</div>

---

## 设计理念

量化研究系统的核心矛盾：**LLM 有推理能力但会幻觉，规则引擎有确定性但无弹性。**

Quant-Investor 的解法是严格分层：

```
确定性控制链（硬约束，永不可被覆盖）
        ↑
LLM 审阅层（advisory-only，提供观点，不做决策）
        ↑
数据快照 + DeterministicFunnel + v13 四分支 DAG
```

**RiskGuard 具有一票否决权。** LLM 的任何输出只能作为参考信号进入 ICCoordinator，永远无法绕过风控硬约束。

---

## ✨ 核心能力

| 能力 | 说明 |
|------|------|
| 🏗 **三层数据协议** | `GlobalContext` → `SymbolResearchPacket` → `PortfolioDecision`，全程 Pydantic 结构化，可追溯 |
| 🔬 **v13 四分支研究 DAG** | 本地快照 → quant-only `DeterministicFunnel` → 量化 · 基本面 · 情报 · 宏观 → Bayesian selection |
| 🛡 **确定性风控** | RiskGuard 硬否决 → ICCoordinator 一致性校验 → PortfolioConstructor 权重分配 |
| 🤖 **可选 LLM 审阅层** | 支持 OpenAI / Claude / DeepSeek / Gemini / 通义 / Kimi，无 API Key 自动降级 |
| 📈 **Parquet 全市场运行面** | strict canonical Parquet、canonical batch symbol 读取和含 exclusive/wall timing 的 stage runtime profile |
| 🌏 **双市场覆盖** | A 股（Tushare Pro）+ 美股（yfinance），统一 pipeline |
| 📊 **Web 工作台** | React 19 + FastAPI，研究任务调度、历史回顾、实时进度 |
| ⏰ **定时任务** | `daily_runner.py` 支持 cron 调度，每日自动执行全市场扫描 |

---

## 🏛 架构

### 执行流水线

```
┌─────────────────────────────────────────────────────────────┐
│             Stage 1: Snapshot + DeterministicFunnel          │
│                                                             │
│   MarketDataReader ──► 本地快照 ──► quant-only funnel        │
│                                      (candidate shortlist)  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  Stage 2: v13 Research DAG                   │
│                                                             │
│   QuantAgent · FundamentalAgent · IntelligenceAgent · Macro │
│        │                                                    │
│        ▼                                                    │
│   SymbolResearchPacket ──► Bayesian selection               │
│        │                                                    │
│        └── optional LLM review hints (advisory-only)         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                 Stage 3: Unified Control Chain               │
│                                                             │
│   RiskGuard → ICCoordinator → PortfolioConstructor          │
│   (硬否决)     (一致性校验)    (确定性权重)                   │
│                                    │                        │
│                                    ▼                        │
│                              NarratorAgent                  │
│                              (ReportBundle)                 │
└─────────────────────────────────────────────────────────────┘
```

### 分支权重

**分支权重**

| 分支 | Quant Factor | Intelligence | Fundamental | Macro |
|------|:------------:|:------------:|:-----------:|:-----:|
| 权重 | 36% | 26% | 19% | 19% |

```
Quant Factor    ████████████████████████████████████  36%
Intelligence    ██████████████████████████            26%
Fundamental     ███████████████████                   19%
Macro           ███████████████████                   19%
```

### 数据协议

```python
GlobalContext              # 市场全局快照（宏观 + 市场结构）
    └── SymbolResearchPacket[]   # 逐标的多分支研究包
            └── PortfolioDecision     # 组合决策 + 执行计划
                    └── ReportBundle       # 可读报告聚合
```

---

## 🚀 快速开始

### 安装

```bash
# 推荐：使用 uv
uv pip install -e ".[dev]"

# 或 pip
pip install -e ".[dev]"
```

### 配置

```bash
cp .env.example .env
```

最小配置（A 股，纯算法模式，无需 LLM）：

```ini
TUSHARE_TOKEN=your_tushare_token
```

启用 LLM 审阅层（任选其一或多个）：

```ini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=...
```

### 三种使用方式

<details open>
<summary><strong>🐍 Python API</strong></summary>

```python
from quant_investor import QuantInvestor

investor = QuantInvestor(
    stock_pool=["000001.SZ", "600519.SH", "300750.SZ"],
    market="CN",
    total_capital=1_000_000,
    risk_level="中等",
    enable_agent_layer=True,       # 启用 LLM 审阅层
    agent_model="claude-3-5-sonnet-20241022",
    verbose=True,
)

result = investor.run()

# 查看组合建议
for rec in result.strategy.recommendations:
    print(f"{rec.symbol}  {rec.action:8s}  权重 {rec.weight:.1%}  {rec.rationale}")

# 完整报告
print(result.report_bundle.markdown_report)
```

</details>

<details>
<summary><strong>💻 CLI</strong></summary>

```bash
# A 股分析
quant-investor research run \
  --stocks 000001.SZ 600519.SH 300750.SZ \
  --market CN \
  --capital 1000000 \
  --risk 中等

# 美股分析
quant-investor research run \
  --stocks AAPL MSFT NVDA \
  --market US \
  --capital 100000 \
  --risk low
```

</details>

<details>
<summary><strong>🌐 Web 工作台</strong></summary>

```bash
# 启动（后端 + 前端一体）
quant-investor web --reload

# 或分离启动（前端开发模式）
./run_web.sh
```

访问 `http://localhost:8000/research`

路由：`/research` · `/history` · `/history/:jobId` · `/settings`

FastAPI research workspace backend 与 React/Vite research workspace frontend 保持同源 API 边界。

</details>

<details>
<summary><strong>⏰ 每日定时分析</strong></summary>

```bash
# 编辑参数后运行
vim daily_config.py
python daily_runner.py
```

`daily_config.py` 支持：市场选择、股票池、资金规模、LLM 模型及 fallback、reasoning 强度、数据下载策略、cron 调度时间。

</details>

<details>
<summary><strong>🐳 Docker</strong></summary>

```bash
docker compose up -d
```

</details>

---

## 🧩 项目结构

```
myQuant/
├── quant_investor/              # 核心引擎
│   ├── pipeline/
│   │   ├── mainline.py          # QuantInvestor 单一主线入口
│   │   ├── result_builder.py
│   │   └── result_types.py
│   ├── agent_protocol.py        # 三层数据协议定义
│   ├── agent_orchestrator.py    # 统一控制链编排
│   ├── agents/
│   │   ├── quant_agent.py       # 量化因子
│   │   ├── fundamental_agent.py # 基本面
│   │   ├── intelligence_agent.py# 舆情情报
│   │   ├── macro_agent.py       # 宏观
│   │   ├── risk_guard.py        # 硬否决风控 ⛔
│   │   ├── ic_coordinator.py    # 一致性协调
│   │   ├── portfolio_constructor.py
│   │   ├── narrator_agent.py    # 报告生成
│   │   ├── master_agent.py      # LLM IC 主席
│   │   └── subagents/           # LLM 审阅子 agent
│   ├── market/                  # 市场数据、全市场 DAG 和报告持久化
│   │   ├── market_data_reader.py# Parquet/CSV 统一读取门面
│   │   ├── market_data_store.py # canonical/clean storage validation helpers
│   │   ├── shared_csv_reader.py # 批量 symbol 读取兼容 alias
│   │   ├── runtime_profile.py   # market run/analyze stage profile
│   │   ├── dag/                 # context/research/Bayesian/control/reporting
│   │   ├── dag_executor.py      # v13 全市场 DAG 执行器
│   │   └── download.py          # maintain/download 兼容入口
│   ├── llm_gateway.py           # 统一 LLM 网关
│   └── reporting/               # Markdown 报告渲染
├── web/                         # FastAPI 后端
├── frontend/                    # React 19 + Vite + TailwindCSS
├── daily_runner.py              # 定时任务入口
├── daily_config.py              # 每日分析配置
├── tests/                       # 单元 & 集成测试
└── docs/                        # 架构 & 模块文档
```

---

## 🔑 环境变量

| 变量 | 必需 | 说明 |
|------|:----:|------|
| `TUSHARE_TOKEN` | A 股必需 | [Tushare Pro](https://tushare.pro) 数据接口 |
| `TUSHARE_URL` | 可选 | Tushare 代理 URL，默认使用主线代理 |
| `TUSHARE_RATE_LIMIT_PER_MIN` | 可选 | Tushare 客户端本地限速预算，默认 500 |
| `ANTHROPIC_API_KEY` | 可选 | Claude 模型（推荐用于 Review Layer） |
| `OPENAI_API_KEY` | 可选 | GPT 系列 |
| `DEEPSEEK_API_KEY` | 可选 | DeepSeek（性价比高） |
| `GOOGLE_API_KEY` | 可选 | Gemini 系列 |
| `DASHSCOPE_API_KEY` | 可选 | 通义千问 |
| `KIMI_API_KEY` | 可选 | Moonshot Kimi |

> **注意：** 所有 LLM Key 均为可选。未配置时系统自动降级为纯算法模式，研究核心和风控链路正常运行。

---

## 🧪 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 单元测试
pytest tests/unit/ -v

# 集成测试
pytest tests/integration/ -v

# 代码格式化
black quant_investor/ && flake8 quant_investor/
```

---

## 📐 协议术语表

| 术语 | 含义 |
|------|------|
| `GlobalContext` | 市场全局快照（宏观指标 + 市场结构） |
| `SymbolResearchPacket` | 单标的多分支研究结果聚合包 |
| `PortfolioDecision` | 组合决策，含权重分配与执行计划 |
| `ReportBundle` | NarratorAgent 输出的可读报告聚合 |
| `BranchVerdict` | 单分支分析结论（方向 + 置信度 + 证据链） |
| `ICDecision` | Investment Committee 共识决策 |
| `RiskDecision` | 风控决策，含仓位上限与否决记录 |
| `BranchOverlayVerdict` | LLM 分支审阅叠加意见（advisory-only） |
| `MasterICHint` | MasterAgent 对 IC 的提示信号 |
| `NarratorAgent -> ReportBundle` | 报告生成协议 |
| `AgentStatus` | `SUCCESS` / `DEGRADED` / `VETOED` |
| `ActionLabel` | `buy` / `hold` / `sell` / `watch` / `avoid` |
| 已移除旧标签 | `reject` / `light_buy` / `strong_buy` 不再作为公开动作标签 |

---

## 📚 文档

- [架构概览与版本管理](docs/architecture/entrypoints_and_versioning.md)
- [研究管道与数据协议](docs/architecture/research_pipeline_and_protocols.md)
- [模块索引](docs/modules/module_map.md)
- [宏观风险参考](docs/modules/macro_risk_reference.md)

---

## 许可证

[MIT License](LICENSE) © 2024 alpha-mw
