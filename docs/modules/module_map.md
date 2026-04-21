# Module Map

本页按功能和模块整理当前仓库事实。

## Core Modules

### `quant_investor/pipeline`

- 角色：单一主线入口与研究主线编排
- 关键文件：`mainline.py`、`parallel_research_pipeline.py`

### `quant_investor/agents`

- 角色：结构化 review layer、控制链和只读 Narrator
- 关键文件：`risk_guard.py`、`ic_coordinator.py`、`portfolio_constructor.py`、`narrator_agent.py`

### `quant_investor/reporting`

- 角色：报告渲染、三桶整理、summary 生成
- 关键文件：`conclusion_renderer.py`、`diagnostics_bucketizer.py`、`executive_summary.py`

### `quant_investor/market`

- 角色：全市场下载、分析、回测入口
- 说明：`run_market_analysis()` 是当前 full-market 主线

### `quant_investor/learning`

- 角色：recall / proposal / reflection 闭环
- 说明：当前闭环只生成结构化 recall 与 proposal，不直接写回 live decision

## Supporting Modules

### `quant_investor/cli`

- 角色：统一 CLI 入口
- 说明：`research run` 始终执行单一主线

### `quant_investor/kline_backends`

- 角色：K 线后端选择与融合
- 说明：`heuristic` 用于快筛，`kronos` / `chronos` / `hybrid` 用于显式深度分析路径

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
