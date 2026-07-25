# Entrypoints And Versioning

本文件是当前入口、版本语义和公开命名的唯一事实来源。

## Public Entrypoints

- Python 入口：`quant_investor.QuantInvestor`
- Pipeline 结果类型：`quant_investor.pipeline.QuantInvestorPipelineResult`
- CLI 入口：`quant-investor research run`
- 当前包版本为 `17.0.0`，它是删除旧公共入口的 V16 retirement release，
  不是 V17 runtime 协议。`market analyze/run` 只接受并默认使用 `v15`；
  任何已退役协议 literal 都在解析阶段 exit 2，且不回退、不写文件。
- CN bounded maintenance 入口：`quant-investor market maintain --market CN --staged --resume`；每次只处理配置的 batch，并在 `data/cn_market_full/_maintenance_runs/<run_id>/` 写入进度。
- `quant-investor market download` 仅作为 `market maintain` 的兼容 alias 保留；新流程应优先使用 `market maintain`。
- Storage 验证入口：`quant-investor market storage-validate --market CN` 校验 Parquet canonical；`quant-investor market storage-validate-clean --market CN` 只读校验 clean/readiness lineage，不触发补数、provider 或写入。
- CN Fundamental 隔离重建入口：`quant-investor market fundamental-maintain --market CN --allow-live --authoritative-full-rebuild ...`；该入口只在显式授权后调用 provider，并把 accepted raw、逐请求结果和派生表写入独立 staging/checkpoint。v3 权威 scope 同时绑定 canonical bar 文件集合 SHA、逐证券首末交易边界和固定审计策略；binding 变化必须使用新 checkpoint root。
- CN Fundamental 晋升入口：`quant-investor market fundamental-promote --staging-root <path> --expected-pointer-sha256 <sha>`；该入口从 v3 checkpoint 重放 raw→derived、财务成熟资格期与 canonical-bar 日频覆盖，验证 exact-byte provenance，并以 CAS 推进 `_fundamental_latest.json`。
- CN Macro observation/release-calendar 入口：`macro-release-calendar-publish` 以 expected-pointer CAS 发布只允许 prefix extension 的 immutable 官方日历；`macro-observation-official-refresh` 以完整官方 bundle 重建 36 条官方行并保留最新三条连续 local 行；`macro-local-observation-roll` 对同一 exact 39-row/13-indicator/three-history scope 做 local catch-up、滚动或修正。两个 observation projection 命令都绑定显式 `market-open-days.v1` 路径、bytes SHA、有序 CN 开市日与父代 lineage。
- CN Macro 正式补数入口：`quant-investor market macro-maintain --authoritative-refresh ... --expected-macro-observations-pointer-sha256 <sha> --expected-macro-release-calendar-pointer-sha256 <sha> --allow-live` 只写隔离 staging，并把 catalog、market pointer、canonical observations、release calendar、ready MacroSnapshot、v15 control projection、exact bars 与 provider evidence 逐层 hash-bind；`quant-investor market macro-promote --staging-root <path> --expected-catalog-sha256 <sha>` 独立重验后按 market writer -> catalog writer -> Macro-observation writer -> release-calendar writer 顺序持锁，对四个 canonical identities 做 CAS/readback，再以 journal 推进 strict catalog。production schema 自报不构成证据；必须重算 official/local row scope、parent lineage、evidence roles 和逐行 mapping，并且同时满足 `period_end <= logical as_of` 与 `available_at <= decision_cutoff_at`。旧 `macro-refresh` 不再是 CLI 入口，legacy one-step live writer 固定 fail closed 为 `macro_authoritative_stage_promotion_required`，v14 mart 不能重标为 v15。`--allow-tushare-fallback` 默认关闭且只适用于已分类的 NBS 瞬态传输故障；schedule 只有在本身含精确数据维护授权、四 identity CAS 且无分析/交易副作用时才可运行该链。
- v14 Intelligence 残留迁移入口：`scripts/retire_event_score_catalog_residual.py`；只在显式 catalog/market-pointer/source-table 三 SHA、新 run id、`--apply` 和静态确认 token 齐全时，把 `event_daily_score` 切到不含 `intelligence_score` 的不可变 generation。旧 Parquet 保持原字节，迁移使用共享 catalog lock、durable WAL、CAS、fresh strict-reader readback 与回滚；日常 schedule 不运行该入口。

## Workspace Entrypoints

- 研究工作台完整启动入口：仓库根目录 `./run_web.sh`
- `quant-investor web` 启动 `web.main:app`，该入口转发到 `web.workspace_app:app`
- `web.app:app` 保留为独立 API 服务入口；workspace 与 API 入口职责分离
- 当前工作台前端位于 `frontend/`，开发态通过 Vite 代理 `/api` 到 FastAPI

## Package And Protocol Versions

- package release：`17.0.0`（V16 retirement release）
- V17 runtime：不存在、未接入、未授权
- Factor Governance：`v4`，使用版本中立的 `results/factor_governance/`
- Dashboard：Contract v3

当前 production/default 协议继续使用：

- `ARCHITECTURE_VERSION = "15.0.0-stable"`
- `BRANCH_SCHEMA_VERSION = "branch-schema.v15.three-branch"`
- `LIKELIHOOD_SCHEMA_VERSION = "likelihood-schema.v15.two-likelihood"`
- `IC_PROTOCOL_VERSION = "ic-protocol.v15.three-branch"`
- `REPORT_PROTOCOL_VERSION = "report-protocol.v15.three-branch"`
- `CALIBRATION_SCHEMA_VERSION = "2026-07-16.calibration.v15.three-branch"`
- `AGENT_SCHEMA_VERSION = "2026-07-16.agent.v15.three-branch"`

## Runtime Readiness Semantics

每个 run 只固定一次 immutable Macro release-calendar generation，并构造一个
hash-bound `MacroReadinessEvidence`；Macro assessment、三分支 readiness 与
`v15_run_readiness` 复用同一证据和 cutoff。Macro logical date 可在 pinned CN
open-day calendar 上落后目标 0–2 个 session，但任何 positive-lag gap 从
logical-date 上海收盘到 decision cutoff 内只要出现 critical official event 就
阻断；lag 0 仍要求 pinned evidence 与完整 resolution。超过两个 session 或任何
calendar identity 漂移也阻断。

`branch_data_ready` green 只表示 `quant`、`fundamental`、`macro` 三个 canonical
branch 通过。Factor governance 独立评估；hash-bound human authorization receipt
也是新风险的必要但不充分条件，不能覆盖任何 data、Factor、candidate 或 portfolio
blocker，也不会自动令 `new_risk_authorized=true`。

## Current Production Web Result Envelope

`AnalysisSessionDetail` 与 Web 持久化结果必须在顶层同时携带并精确匹配：

- `architecture_version = "15.0.0-stable"`
- `branch_schema_version = "branch-schema.v15.three-branch"`
- `likelihood_schema_version = "likelihood-schema.v15.two-likelihood"`
- `report_protocol_version = "report-protocol.v15.three-branch"`

当前 Web DTO 只接受按 `kline, quant, fundamental, llm_debate, macro` 排序的分支列表，其中三个 canonical 分支必须启用。缺失 schema、旧 v13 schema、Intelligence、未知分支及旧 CN/US legacy artifact 都会 fail closed，不会被补字段、过滤或包装成当前 DTO。

## Current Production Market Report Artifact Envelope

`market analyze`、DAG-to-report synthesis、batch report artifact 与 full-market report builder 必须同时携带并精确匹配 Web envelope 的四项版本以及：

- `ic_protocol_version = "ic-protocol.v15.three-branch"`

每个 batch 顶层及其 `analysis_meta` 都必须携带这五项版本，`branches` 必须恰好包含 `quant, fundamental, macro`，任意嵌套结构也不得出现以 Intelligence 命名的机器字段。逐批持久化、full-market 汇总和 US 模拟组合 recommendation loader 都在读写或动作生成前执行同一合同；缺失 canonical 分支、额外的 K-Line/Intelligence/未知分支、旧版本或无版本 artifact 会明确报错，当前链路不再静默过滤、升级或消费旧 artifact。

当前 market batch 只写入 `results/v15/cn_analysis_full` 与 `results/v15/us_analysis_full`；v14 与旧的无版本目录仅保留为历史快照，不参与当前报告、Dashboard metadata 或模拟组合 recommendation 消费。

## Public Protocol Names

- `branch review` 是当前公开规范名。
- `NarratorAgent -> ReportBundle` 是当前结构化报告协议名。
- 当前稳定动作标签是 `buy / hold / sell / watch / avoid`。
- `reject / light_buy / strong_buy` 只保留历史映射语义，不是当前稳定 schema 名。

## Learning Schema Naming

learning 模块的结构化对象统一带 `schema_version`，命名采用 `learning.<object>.2026-03-24.v1` 风格。
