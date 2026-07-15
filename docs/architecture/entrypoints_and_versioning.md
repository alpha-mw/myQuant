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
- CN Fundamental 隔离重建入口：`quant-investor market fundamental-maintain --market CN --allow-live --authoritative-full-rebuild ...`；该入口只在显式授权后调用 provider，并把 accepted raw、逐请求结果和派生表写入独立 staging/checkpoint。v3 权威 scope 同时绑定 canonical bar 文件集合 SHA、逐证券首末交易边界和固定审计策略；binding 变化必须使用新 checkpoint root。
- CN Fundamental 晋升入口：`quant-investor market fundamental-promote --staging-root <path> --expected-pointer-sha256 <sha>`；该入口从 v3 checkpoint 重放 raw→derived、财务成熟资格期与 canonical-bar 日频覆盖，验证 exact-byte provenance，并以 CAS 推进 `_fundamental_latest.json`。
- CN Macro 正式补数入口：`quant-investor market macro-refresh --market CN --run-id <new_id> --expected-catalog-sha256 <sha> --expected-market-pointer-sha256 <sha> --allow-live`；只接受 latest-complete session，在 72 小时 capture window 内绑定五个 Tushare endpoint、exact bar files 和版本化 transform，经共享 writer lock、双 CAS 与 journal 原子推进 strict catalog。周审 schedule 只能检查 help，不得运行该写入口。
- v14 Intelligence 残留迁移入口：`scripts/retire_event_score_catalog_residual.py`；只在显式 catalog/market-pointer/source-table 三 SHA、新 run id、`--apply` 和静态确认 token 齐全时，把 `event_daily_score` 切到不含 `intelligence_score` 的不可变 generation。旧 Parquet 保持原字节，迁移使用共享 catalog lock、durable WAL、CAS、fresh strict-reader readback 与回滚；日常 schedule 不运行该入口。

## Workspace Entrypoints

- 研究工作台完整启动入口：仓库根目录 `./run_web.sh`
- `quant-investor web` 启动 `web.main:app`，该入口转发到 `web.workspace_app:app`
- `web.app:app` 保留为独立 API 服务入口；workspace 与 API 入口职责分离
- 当前工作台前端位于 `frontend/`，开发态通过 Vite 代理 `/api` 到 FastAPI

## Runtime Versions

- package version：`14.0.0`
- `ARCHITECTURE_VERSION = "14.0.0-stable"`
- `BRANCH_SCHEMA_VERSION = "branch-schema.v14.three-branch"`
- `LIKELIHOOD_SCHEMA_VERSION = "likelihood-schema.v14.two-likelihood"`
- `IC_PROTOCOL_VERSION = "ic-protocol.v14.three-branch"`
- `REPORT_PROTOCOL_VERSION = "report-protocol.v14.three-branch"`
- `CALIBRATION_SCHEMA_VERSION = "2026-07-14.calibration.v14.three-branch"`
- `AGENT_SCHEMA_VERSION = "2026-07-14.agent.v14.three-branch"`

## Current Web Result Envelope

`AnalysisSessionDetail` 与 Web 持久化结果必须在顶层同时携带并精确匹配：

- `architecture_version = "14.0.0-stable"`
- `branch_schema_version = "branch-schema.v14.three-branch"`
- `likelihood_schema_version = "likelihood-schema.v14.two-likelihood"`
- `report_protocol_version = "report-protocol.v14.three-branch"`

当前 Web DTO 只接受按 `kline, quant, fundamental, llm_debate, macro` 排序的分支列表，其中三个 canonical 分支必须启用。缺失 schema、旧 v13 schema、Intelligence、未知分支及旧 CN/US legacy artifact 都会 fail closed，不会被补字段、过滤或包装成当前 DTO。

## Current Market Report Artifact Envelope

`market analyze`、DAG-to-report synthesis、batch report artifact 与 full-market report builder 必须同时携带并精确匹配 Web envelope 的四项版本以及：

- `ic_protocol_version = "ic-protocol.v14.three-branch"`

每个 batch 顶层及其 `analysis_meta` 都必须携带这五项版本，`branches` 必须恰好包含 `quant, fundamental, macro`，任意嵌套结构也不得出现以 Intelligence 命名的机器字段。逐批持久化、full-market 汇总和 US 模拟组合 recommendation loader 都在读写或动作生成前执行同一合同；缺失 canonical 分支、额外的 K-Line/Intelligence/未知分支、旧版本或无版本 artifact 会明确报错，当前链路不再静默过滤、升级或消费旧 artifact。

当前 market batch 只写入 `results/v14/cn_analysis_full` 与 `results/v14/us_analysis_full`；旧的无版本目录仅保留为历史快照，不参与当前报告、Web metadata 或模拟组合 recommendation 消费。

## Public Protocol Names

- `branch review` 是当前公开规范名。
- `NarratorAgent -> ReportBundle` 是当前结构化报告协议名。
- 当前稳定动作标签是 `buy / hold / sell / watch / avoid`。
- `reject / light_buy / strong_buy` 只保留历史映射语义，不是当前稳定 schema 名。

## Learning Schema Naming

learning 模块的结构化对象统一带 `schema_version`，命名采用 `learning.<object>.2026-03-24.v1` 风格。
