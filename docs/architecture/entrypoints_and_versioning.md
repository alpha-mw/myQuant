# Entrypoints And Versioning

本文件是当前入口、版本语义和公开命名的唯一事实来源。

## Public Entrypoints

- Python 入口：`quant_investor.QuantInvestor`
- Pipeline 结果类型：`quant_investor.pipeline.QuantInvestorPipelineResult`
- CLI 入口：`quant-investor research run`
- 当前仓库只发布单一研究主线，不提供架构切换参数。
- CN bounded maintenance 入口：`quant-investor market maintain --market CN --staged --resume`；每次只处理配置的 batch，并在 `data/cn_market_full/_maintenance_runs/<run_id>/` 写入进度。
- `quant-investor market download` 仅作为 `market maintain` 的兼容 alias 保留；新流程应优先使用 `market maintain`。
- Storage 验证入口：`quant-investor market storage-validate --market CN` 校验 Parquet canonical；`quant-investor market storage-validate-clean --market CN` 只读校验 clean/readiness lineage，不触发补数、provider 或写入。

## Workspace Entrypoints

- 研究工作台完整启动入口：仓库根目录 `./run_web.sh`
- `quant-investor web` 启动 `web.main:app`，该入口转发到 `web.workspace_app:app`
- `web.app:app` 保留为独立 API 服务入口；workspace 与 API 入口职责分离
- 当前工作台前端位于 `frontend/`，开发态通过 Vite 代理 `/api` 到 FastAPI

## Runtime Versions

- package version：`13.0.0`
- `ARCHITECTURE_VERSION = "13.0.0-stable"`
- `BRANCH_SCHEMA_VERSION = "branch-schema.v13.four-branch"`
- `IC_PROTOCOL_VERSION = "ic-protocol.v13.four-branch"`
- `REPORT_PROTOCOL_VERSION = "report-protocol.v13.four-branch"`
- `CALIBRATION_SCHEMA_VERSION = "2026-03-22.calibration.v2"`
- `AGENT_SCHEMA_VERSION = "2026-03-23.agent.v1"`

## Public Protocol Names

- `branch review` 是当前公开规范名。
- `NarratorAgent -> ReportBundle` 是当前结构化报告协议名。
- 当前稳定动作标签是 `buy / hold / sell / watch / avoid`。
- `reject / light_buy / strong_buy` 只保留历史映射语义，不是当前稳定 schema 名。

## Learning Schema Naming

learning 模块的结构化对象统一带 `schema_version`，命名采用 `learning.<object>.2026-03-24.v1` 风格。
