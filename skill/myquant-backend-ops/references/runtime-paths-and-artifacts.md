# myQuant Runtime Paths and Artifacts

这一页回答“东西落在哪里、谁负责写、排障先看哪”。

## Core Runtime Files

### `.env`

- 位置：repo root `.env`
- 用途：运行时环境变量与 API keys
- 典型键：
  - `TUSHARE_TOKEN`
  - `KIMI_API_KEY`
  - `DEEPSEEK_API_KEY`
  - `DASHSCOPE_API_KEY`
  - `FRED_API_KEY`
  - `FINNHUB_API_KEY`
  - `DB_PATH`
  - `LOG_LEVEL`

Primary owners:
- `web/config.py`
- `web/services/settings_service.py`
- `web/routers/settings.py`
- `quant_investor/config.py`

### `data/stock_database.db`

- 主本地市场数据库
- 保存股票列表、OHLCV、因子、profile、fundamentals、peer relationships
- `DB_PATH` 默认指向这里

Primary owners:
- `web/services/data_service.py`
- `quant_investor/stock_database.py`

### `data/app.db`

- 旧分析面和部分 portfolio 状态数据库
- 主要由 legacy 分析服务使用，不是 workspace 主历史库

Primary owners:
- `web/services/analysis_service.py`
- `web/services/portfolio_service.py`

### `data/web_runs.db`

- 当前 workspace 主数据库
- 保存：
  - `runs`
  - `presets`
  - `trade_records`

Primary owners:
- `web/services/run_history_store.py`
- `web/services/preset_store.py`
- `web/routers/research.py`
- `web/routers/presets.py`
- `web/routers/settings.py`

如果用户问 “history 为什么没写进去”，先看这里，不要先看 `data/app.db`。

### `results/web_analysis/`

- legacy web analysis 结果目录
- 常见内容：
  - `analysis_*.json`
  - `analysis_request_*.json`
  - `jobs/job_*.json`

Primary owners:
- `web/services/analysis_service.py`
- `web/tasks/run_analysis_job.py`

### `data/workspace_learning/`

- workspace 自动保存的待跟踪 trade case 文件
- 一次 run 可能按 symbol 生成多个 `jobid_symbol.json`

Primary owners:
- `web/services/research_runner.py`
- `web/services/run_history_store.py`

### `data/llm_usage.jsonl`

- 在线 LLM 使用记录
- 用于 session 级 token 与成本汇总

Primary owners:
- `quant_investor/llm_gateway.py`
- `quant_investor/pipeline/mainline.py`

## Useful Distinctions

### Workspace vs legacy analysis

- workspace 历史、preset、trade records：`data/web_runs.db`
- legacy analysis session / portfolio mutation：`data/app.db`
- legacy analysis 输出文件：`results/web_analysis/`

### Deterministic path vs review layer

- 主线可在无 LLM key 时安全降级
- 缺少 `KIMI_API_KEY` 或 `DEEPSEEK_API_KEY` 不等于整个后端不可用
- 先区分是：
  - 主线执行失败
  - review layer 降级
  - 结果持久化失败

## Diagnostic Starting Points

### “为什么 workspace history 里没有这次运行”

先查：
1. `web/routers/research.py`
2. `web/services/research_runner.py`
3. `web/services/run_history_store.py`
4. `data/web_runs.db`

### “为什么 analysis 页面结果文件没有生成”

先查：
1. `web/services/analysis_service.py`
2. `web/tasks/run_analysis_job.py`
3. `results/web_analysis/`
4. `data/app.db`

### “数据库到底应该看哪个”

- 市场数据：`data/stock_database.db`
- workspace 历史：`data/web_runs.db`
- legacy analysis / portfolio：`data/app.db`
