# Module Map

本页按功能和模块整理当前仓库事实。

## Core Modules

### `quant_investor/pipeline`

- 角色：单一主线入口与研究主线编排
- 关键文件：`mainline.py`、`result_builder.py`、`result_types.py`
- 说明：`mainline.py` 是公开 `QuantInvestor` 入口；内部通过 market DAG artifacts 构造 `QuantInvestorPipelineResult`，不再依赖 legacy batch pipeline。

### `quant_investor/agents`

- 角色：结构化 review layer、控制链和只读 Narrator
- 关键文件：`risk_guard.py`、`ic_coordinator.py`、`portfolio_constructor.py`、`narrator_agent.py`

### `quant_investor/reporting`

- 角色：报告渲染、三桶整理、summary 生成
- 关键文件：`conclusion_renderer.py`、`diagnostics_bucketizer.py`、`executive_summary.py`

### `quant_investor/market`

- 角色：全市场维护、数据读取、分析、回测和报告持久化入口
- 关键文件：`market_data_reader.py`、`market_data_store.py`、`read_result.py`、`name_map.py`、`runtime_profile.py`、`full_report.py`、`report_persistence.py`、`legacy_synthesis.py`、`legacy_batch_analysis.py`、`dag_executor.py`
- 子模块：`dag/context.py`、`dag/research.py`、`dag/shortlist.py`、`dag/decision.py`、`dag/reporting.py`
- 说明：`run_market_analysis()` / `execute_market_dag()` 是当前 v14 full-market 主线；生产读取以 strict Parquet canonical 和 JSON manifest 为准，CSV 只允许作为人工导出或显式一次性迁移输入；`full_report.py` 负责全市场报告渲染，`report_persistence.py` 负责报告落盘和 runtime profile 写入；`legacy_synthesis.py` 与 `legacy_batch_analysis.py` 仅承接旧 batch/sample 入口，但输入仍必须满足当前 v14 envelope，遇到退休或未知结构字段会 fail closed，不会过滤后重包装。

### `quant_investor/macro`

- 角色：v14 PIT Macro observations、离线回放和 measurement-only observer
- 关键文件：`contracts.py`、`store.py`、`snapshot.py`、`observer.py`、`replay.py`
- 说明：`macro_observations` 只从严格 `_latest.json` generation store 进入 DAG observer metadata；standalone 文件仅允许显式离线 CLI 使用。observer 固定为 `production_eligible=false`、`applied=false`，不产生 Macro likelihood，也不改变 Markov、RiskGuard 或组合权重。完整数据和运行契约见 [Macro v2 Observer Contract](macro_v2_observer.md)。

### `quant_investor/learning`

- 角色：recall / proposal / reflection 闭环
- 说明：当前闭环只生成结构化 recall 与 proposal，不直接写回 live decision

## Supporting Modules

### `quant_investor/cli`

- 角色：统一 CLI 入口
- 说明：`research run` 始终执行单一主线

### `quant_investor/monitoring`

- 角色：正式复盘、策略记录和离线监控辅助
- 关键文件：`cn_aggressive_portfolio_tracker.py`、`cn_aggressive_market_metrics.py`、`cn_aggressive_daily_review.py`
- 说明：CN aggressive tracker 保留复盘编排、交易纪律和报告落盘；`cn_aggressive_market_metrics.py` 独立负责 full-market metrics、breadth 和 prewarm cache，不直接执行换仓或写 strategy decision。

### `quant_investor/versioning.py`

- 角色：版本常量、schema 常量与公开命名

### `quant_investor/_sourceless.py`

- 角色：限定作用域的 sourceless 导入兜底
- 说明：主线测试要求当前公开路径的 `.pyc` 必须有 `.py` 源文件对应

## Workspace Modules

### `web/`

- 角色：研究工作台 FastAPI 后端
- 关键文件：`api.py`、`routers/research.py`、`routers/settings.py`、`routers/presets.py`、`routers/universe.py`
- 说明：负责工作台运行请求、SSE 日志流、历史持久化、预设 CRUD 和环境/模型可用性查询

### `frontend/`

- 角色：研究工作台 React/Vite 前端
- 关键文件：`src/pages/ResearchPage.tsx`、`src/pages/HistoryPage.tsx`、`src/pages/SettingsPage.tsx`
- 说明：负责研究参数配置、实时运行观察、历史结果查看和 Settings 展示
