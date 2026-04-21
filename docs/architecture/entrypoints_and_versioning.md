# Entrypoints And Versioning

本文件是当前入口、版本语义和公开命名的唯一事实来源。

## Public Entrypoints

- Python 入口：`quant_investor.QuantInvestor`
- Pipeline 结果类型：`quant_investor.pipeline.QuantInvestorPipelineResult`
- CLI 入口：`quant-investor research run`
- 当前仓库只发布单一主线，不提供架构切换参数和兼容别名。

## Workspace Entrypoints

- 研究工作台完整启动入口：仓库根目录 `./run_web.sh`
- `quant-investor web` 启动 `web.main:app`，该入口转发到 `web.workspace_app:app`
- `web.app:app` 保留为独立 API 服务入口；workspace 与 API 入口职责分离
- 当前工作台前端位于 `frontend/`，开发态通过 Vite 代理 `/api` 到 FastAPI

## Runtime Versions

- package version：`12.0.0`
- `ARCHITECTURE_VERSION = "12.0.0-stable"`
- `BRANCH_SCHEMA_VERSION = "branch-schema.v12.unified-mainline"`
- `IC_PROTOCOL_VERSION = "ic-protocol.v12.mainline"`
- `REPORT_PROTOCOL_VERSION = "report-protocol.v12.mainline"`
- `CALIBRATION_SCHEMA_VERSION = "2026-03-22.calibration.v2"`
- `AGENT_SCHEMA_VERSION = "2026-03-23.agent.v1"`

## Public Protocol Names

- `branch review` 是当前公开规范名。
- `NarratorAgent -> ReportBundle` 是当前结构化报告协议名。
- 当前稳定动作标签是 `buy / hold / sell / watch / avoid`。
- `reject / light_buy / strong_buy` 只保留历史映射语义，不是当前稳定 schema 名。

## Learning Schema Naming

learning 模块的结构化对象统一带 `schema_version`，命名采用 `learning.<object>.2026-03-24.v1` 风格。
