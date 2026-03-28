# Quant-Investor

> 单一主线量化投研系统，支持 A 股与美股，集成多 LLM 辩论框架、因子分析、风险管理与回测。

- **Package**: `quant-investor` v12.0.0
- **Python API**: `from quant_investor import QuantInvestor`
- **CLI**: `quant-investor research run`
- **Web UI**: `./run_web.sh`

---

## 特性

- **多智能体辩论**：多个 LLM 对同一标的展开结构化辩论，输出 `DebateVerdict` 与置信度校准信号
- **全流程研究管道**：基本面 → 量化因子 → 风险评估 → 投资组合构建，结果统一封装为 `ResearchPipelineResult`
- **因子库**：Alpha158、技术指标、基本面因子、宏观替代因子，支持遗传算法挖矿
- **风险管理**：VaR / CVaR、压力测试、因子风险模型、市场冲击估算
- **回测引擎**：Walk-forward 验证，支持 A 股与美股历史数据
- **宏观终端**：实时拉取 Tushare / FRED / AkShare 宏观指标，输出风险雷达
- **研究工作台**：FastAPI 后端 + React/Vite 前端，可视化研报与持仓管理

---

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### Python API

```python
from quant_investor import QuantInvestor

investor = QuantInvestor(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    risk_level="中等",
    verbose=True,
)

result = investor.run()
print(result.report_bundle)
```

### CLI

```bash
quant-investor research run \
  --stocks 000001.SZ 600519.SH \
  --market CN \
  --capital 1000000 \
  --risk 中等
```

### Web 工作台

```bash
# 同时启动后端 API 与前端开发服务
./run_web.sh

# 仅启动后端
quant-investor web --reload
```

前端位于 `frontend/`，通过 Vite 将 `/api` 代理到后端。

---

## 环境变量

复制 `.env.example` 并填写：

```bash
cp .env.example .env
```

主要变量：

| 变量 | 说明 |
|------|------|
| `TUSHARE_TOKEN` | Tushare Pro Token（A 股数据） |
| `FRED_API_KEY` | FRED 宏观数据 API Key |
| `FINNHUB_API_KEY` | Finnhub 美股数据 API Key |
| `OPENAI_API_KEY` | OpenAI（LLM 辩论引擎） |
| `ANTHROPIC_API_KEY` | Anthropic Claude（LLM 辩论引擎） |

---

## 项目结构

```text
myQuant/
├── quant_investor/       # 单一主线包实现
│   ├── pipeline/         # QuantInvestor 主管道
│   ├── agents/           # 多智能体辩论框架
│   ├── data/             # 数据获取与管理
│   ├── market/           # A 股 / 美股市场适配
│   ├── reporting/        # NarratorAgent → ReportBundle
│   ├── monitoring/       # 系统监控与告警
│   └── branch_contracts.py  # 公开数据契约（Pydantic 模型）
├── web/                  # FastAPI 后端
├── frontend/             # React/Vite 前端
├── tests/                # 单元与集成测试
├── docs/                 # 架构与模块文档
├── data/                 # 本地数据目录（git 忽略）
└── results/              # 本地输出目录（git 忽略）
```

---

## 公开契约

| 术语 | 说明 |
|------|------|
| `branch review` | 当前公开研究流程规范名 |
| `NarratorAgent → ReportBundle` | 当前报告协议 |
| `buy / hold / sell / watch / avoid` | 当前稳定动作标签 |

---

## 开发

```bash
# 运行测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v
```

---

## 文档

- [Documentation Index](docs/README.md)
- [Entrypoints and Versioning](docs/architecture/entrypoints_and_versioning.md)
- [Research Pipeline and Protocols](docs/architecture/research_pipeline_and_protocols.md)
- [Module Map](docs/modules/module_map.md)

---

## License

MIT
