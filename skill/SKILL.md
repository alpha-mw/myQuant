---
name: quant-investor
description: "用于 myQuant V8.0 五路并行研究主线的 Codex skill。当用户要在当前仓库中分析、修改、调试或测试量化研究流水线、分支契约、风险控制、Alpha 挖掘、宏观模块或回测相关 Python 代码时使用。默认只覆盖 CLI、Python API、测试与研究主线，不处理 web、frontend、FastAPI 服务或页面代码，除非用户明确要求。"
---

# Quant-Investor V8 Skill

这个 skill 面向当前仓库里的 `myQuant/` 项目主线，目标是让 Codex 优先围绕 V8.0 五路并行研究架构工作，而不是回到早期 V6/V7 文档或 web 代码。

## 适用范围

默认处理以下内容：

- `scripts/unified/` 下的 V8 主线研究、风险、组合、回测与数据模块
- `tests/` 下的单元测试与回归验证
- `README.md`、`pyproject.toml`、`.env.example` 等项目级说明
- A 股 / 美股研究逻辑、分支契约、降级策略、可信度治理

默认不处理以下内容，除非用户明确要求：

- `web/`
- `frontend/`
- `run_web.sh`
- FastAPI / Uvicorn 服务与页面交互
- 仓库内旧版界面层、展示层或前后端联调任务

默认把以下内容视为历史资料，而不是当前主线事实来源：

- `archive/skill_legacy_20260317/` 中的旧 skill 产物
- `scripts/unified/archive/` 以及 `scripts/v*/` 历史目录

只有当用户明确要求处理旧版本、迁移兼容或历史对比时，才读取这些文件。

## 当前主线的事实来源

处理 V8 任务时，优先按下面顺序建立上下文：

1. `README.md`
2. `pyproject.toml`
3. `scripts/unified/quant_investor_v8.py`
4. `scripts/unified/parallel_research_pipeline.py`
5. `scripts/unified/branch_contracts.py`
6. 与当前任务直接相关的研究模块，例如：
   - `scripts/unified/enhanced_data_layer.py`
   - `scripts/unified/alpha_mining.py`
   - `scripts/unified/risk_management_layer.py`
   - `scripts/unified/macro_terminal_tushare.py`
   - `scripts/unified/portfolio_backtest.py`
7. 对应的单元测试，例如：
   - `tests/unit/test_parallel_research_pipeline.py`
   - `tests/unit/test_data_layer.py`
   - `tests/unit/test_risk_management.py`
   - `tests/unit/test_backtest.py`

如果任务涉及宏观风控，再读取：

- `scripts/unified/MACRO_RISK_GUIDE.md`

如果任务明确指向历史版本，再进入：

- `scripts/unified/archive/`
- `scripts/v6.0/`
- `scripts/v2.9/`
- `scripts/v2.7/`

## 核心架构认知

当前 V8 主线不是旧式串行七层，而是：

`数据层 -> 五路并行研究分支 -> 风控层 -> 集成裁判层`

五个研究分支为：

- `kline` / `kronos`：OHLCV 时序趋势与未来收益判断
- `quant`：Alpha 挖掘优先，失败时回退经典因子
- `llm_debate`：基本面、行业、事件驱动的多空辩论
- `intelligence`：财务质量、情绪、事件风险、资金流、广度
- `macro`：宏观 regime、流动性、政策和市场风险状态

必须维护的稳定契约：

- 数据层输出 `UnifiedDataBundle`
- 各研究分支输出 `BranchResult`
- 编排层输出 `ResearchPipelineResult`
- 最终组合层输出 `PortfolioStrategy`

处理相关改动时，优先保证以下语义不被破坏：

- `research_only`
- `degraded`
- `provenance_summary`
- `synthetic_symbols`
- `branch_mode`
- `reliability`

如果一个改动会影响这些字段，必须同步检查报告输出、聚合逻辑和测试。

## 执行工作流

### 1. 先判断任务落点

先区分任务属于哪一类：

- 当前 V8 主线功能开发或缺陷修复
- 风控 / 组合策略调参
- 数据获取、降级与 synthetic 处理
- Alpha 挖掘或因子分析
- 回测与评估
- 历史版本兼容

若用户只说“改 myQuant”，默认按 V8 主线处理，不主动进入 web 或 archive。

### 2. 只读最小必要文件

不要一次性加载整个仓库。先读主入口、契约和相关测试，再按需深入到具体模块。

### 3. 先检查本地数据是否最新

- 如果任务依赖最新行情、A 股日线或宏观数据，先确认本地库与缓存中的最新交易日是否已经覆盖到最新可用交易日。
- 如果本地数据不是最新的，优先使用 `myQuant` 里的数据下载模块把最新数据落到本地，再继续分析、回测或验证；A 股默认走 Tushare 链路。
- 优先入口是：
  - `scripts/unified/data_manager.py`
  - `scripts/unified/stock_database.py`
  - `scripts/unified/macro_terminal_tushare.py`
- 不要为了临时完成任务而跳过本地数据层，直接用远端返回值替代本地落盘结果。
- 如果 `TUSHARE_TOKEN`、网络或接口可用性不足导致无法更新本地数据，需要明确说明阻塞点和未验证风险。

### 4. 先找测试，再改代码

优先通过现有测试确认模块边界。若缺少覆盖，为改动补充最小必要测试，尤其是：

- 分支失败降级
- synthetic symbol 排除
- 风险暴露收缩
- 组合候选与仓位上限
- 报告中的可信度 / provenance 字段

### 5. 改动时遵守项目约定

- 注释和文档字符串默认使用中文
- 使用 `loguru` 体系和 `get_logger`
- 优先使用项目自定义异常，避免裸 `except Exception`
- 不要把 API key、token、路径密钥硬编码进仓库
- 沿用现有模块风格，不因为“更现代”就擅自改成 web service 或前后端架构

### 6. 验证只做与任务相关的最小闭环

优先运行定向测试，其次再跑更大的集合。不要为了一个小修复默认跑全量长耗时任务。

## 常用入口

### 命令行分析

```bash
cd myQuant
python scripts/unified/quant_investor_v8.py \
  --stocks 000001.SZ 600519.SH 000858.SZ \
  --market CN \
  --capital 1000000 \
  --risk 中等
```

### Python API

```python
from scripts.unified import QuantInvestorV8

analyzer = QuantInvestorV8(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    risk_level="中等",
    enable_kronos=True,
    enable_intelligence=True,
    enable_llm_debate=True,
    enable_macro=True,
    verbose=True,
)
result = analyzer.run()
```

### 定向测试

```bash
cd myQuant
pytest tests/unit/test_parallel_research_pipeline.py -v
pytest tests/unit/test_risk_management.py -v
pytest tests/unit/test_backtest.py -v
```

## 文件映射速查

按任务类型优先查看：

- 主入口与公共导出：
  - `scripts/unified/quant_investor_v8.py`
  - `scripts/unified/__init__.py`
- 并行编排与契约：
  - `scripts/unified/parallel_research_pipeline.py`
  - `scripts/unified/branch_contracts.py`
- 数据层：
  - `scripts/unified/enhanced_data_layer.py`
- 传统量化 / Alpha：
  - `scripts/unified/alpha_mining.py`
  - `scripts/unified/factor_analyzer.py`
  - `scripts/unified/factor_neutralizer.py`
- 风控与组合：
  - `scripts/unified/risk_management_layer.py`
  - `scripts/unified/advanced_risk_metrics.py`
  - `scripts/unified/var_calculator.py`
  - `scripts/unified/stress_tester.py`
- 宏观：
  - `scripts/unified/macro_terminal_tushare.py`
  - `scripts/unified/MACRO_RISK_GUIDE.md`
- 回测：
  - `scripts/unified/portfolio_backtest.py`
  - `scripts/unified/backtest_engine.py`
- 全市场批量分析：
  - `scripts/unified/cn_full_market_analysis.py`
  - `scripts/unified/cn_full_market_batch_analysis.py`
  - `scripts/unified/us_full_market_analysis.py`

## 边界提醒

- 本 skill 默认不接管 `web/`、`frontend/`、页面渲染、接口路由和部署脚本。
- 即使 `pyproject.toml` 里存在 `fastapi`、`uvicorn` 依赖，也不要默认把仓库理解为 web 项目。
- 当用户要求“做成 skill”时，默认理解为让 Codex 能更好地维护 `myQuant` 研究主线，而不是重做一个聊天命令系统或前端产品。

## 输出要求

完成任务时，优先给出：

- 改动影响的主线行为
- 验证是否执行以及结果
- 是否仍有未覆盖风险

如果因为本地环境、数据接口或凭据缺失导致无法完成验证，需要明确指出阻塞点。
