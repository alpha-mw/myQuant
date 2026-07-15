# Macro v2 Observer Contract

本页定义 v14 Macro v2 observer 的数据边界、运行边界和不可变决策约束。
Macro v2 是 measurement-only sidecar，不是新的 Bayesian 证据通道，也不拥有生产控制权。

## 不可变决策边界

- Macro v2 的所有 snapshot、readiness、manifest、replay 和 forward-observation
  产物都必须保持 `observer_only=true`、`production_eligible=false`、
  `applied=false`。
- v14 Bayesian likelihood 只包含 `quant_likelihood` 和
  `fundamental_likelihood`。Macro 仍只提供 branch context / prior，不产生第三个
  likelihood，也不进入 likelihood correlation matrix。
- DAG 只能把 Macro v2 结果写入
  `global_context.metadata["macro_v2_observer"]`。observer 的启停、内容、失败或
  缺失不得改变 Markov regime、RiskGuard、ICCoordinator、PortfolioConstructor、
  target weight、exposure cap 或 turnover cap。
- observer 读取或构建失败时只返回诊断 blocker；不得回退到 standalone 文件、
  未绑定 hash 的数据或推断值，也不得阻断既有确定性控制链。

## 两个数据面

| 数据面 | 唯一指针 | 用途 | 生产决策资格 |
| --- | --- | --- | --- |
| `data/parquet/cn/macro_daily` | `data/parquet/cn/_catalog.json` 中的 `macro_daily` entry | 正式 Macro branch 的 strict-Parquet 输入 | 仅 catalog-bound generation 可被生产读取 |
| `data/parquet/cn/macro_observations` | `data/parquet/cn/macro_observations/_latest.json` | Macro v2 PIT observations 与 observer/replay 输入 | 永远只读测量，不直接应用到决策 |

两套指针不能互相替代。`macro_observations/_latest.json` 不是
`macro_daily` 的生产指针；`macro_daily` 目录内的局部 pointer、孤立 Parquet、
JSON 或 JSONL 也不是生产真相。

### `macro_daily`：市场级 catalog 是唯一生产真相

生产读取由 `quant_investor.market.macro_mart.read_macro_mart()` 执行，并且必须同时
满足以下条件：

1. `data/parquet/cn/_catalog.json` 使用 `strict-parquet-catalog.v1`；
2. catalog 含 `macro_daily` entry 和合法 `generation_id`；
3. Parquet 与 generation manifest 都位于
   `macro_daily/_generations/<generation_id>/`，路径不能逃逸或经过 symlink；
4. catalog 声明的 Parquet SHA-256 和 manifest SHA-256 均通过 readback；
5. generation manifest 使用 `cn-macro-mart.v14`、绑定同一 generation，并明确
   `production_eligible=true`；
6. path 每一层都不是 symlink，hash 与 Parquet/JSON 解析使用同一份稳定读取的
   bytes；读取期间发生 replace 或原地改写时立即阻断；
7. canonical frame 只含声明字段，数值范围、日期、`fetched_at`、PIT 状态以及
   manifest/row source lineage 全部一致。
8. generation 同目录必须包含 hash-bound `provider_bundle.json`，完整保留
   `cn_pmi`、`cn_cpi`、`cn_ppi`、`sf_month`、`cn_m` 五个 Tushare 响应和
   selected-input 证据；不能自报 official release timestamp；
9. `primary_provenance` 必须绑定 provider bundle、canonical market pointer、
   逐个行情分区、输出 frame 和 Parquet 的 SHA-256，并使用
   `cn-macro-market-confirmation.v1`；该快照只供 capture 后的下一次决策，固定
   `historical_replay_eligible=false`；
10. catalog 发布必须持有全局 writer lock，同时对 catalog 与 market pointer
    执行 exact-byte/文件身份 CAS。事务 journal 只允许
    `prepared -> switched -> committed`，并绑定 expected market-pointer SHA；
    崩溃恢复只能在该 SHA、完整 required-table 闭包和 generation 全量 readback
    仍一致时完成，否则原字节回滚。无 journal 的 orphan transaction 与已落盘但
    未发布的同 run-id generation 必须可确定性清理后安全重试。
11. provider `fetched_at` 必须取最后一个 endpoint 响应完成时间；I/O 后和 catalog
    switch 前都要重验 72 小时窗口。每个 endpoint 的最新月份不得早于
    `month_end + max_release_lag_days` 所推导的截止下界。
12. 市场公式横截面只包含 terminal bar 等于目标 session 的证券；历史已终止证券
    不得进入当前 `macro_score`、breadth 或 volatility 计算，且该公式 universe
    必须进入 hash-bound provenance。

任何一个条件缺失或不一致都必须 fail closed。读取器不接受目录内“看起来最新”的
文件，也不根据 mtime、文件名或局部 pointer 猜测 canonical generation。

### Offline compatibility：只进入候选区

`quant-investor market macro-maintain --input-json ...` 只把四字段本地兼容行写到：

```text
data/parquet/cn/macro_daily/_candidates/<run_id>/
```

该流程会把调用方自报的 provenance 覆盖为
`manual_offline_snapshot`，并固定写出 `production_eligible=false`、
`applied=false`。它不会写入或推进 `data/parquet/cn/_catalog.json`，不会写局部
latest pointer，也不会把候选 generation 暴露给生产读取器。

因此，offline compatibility 输入只能用于审计和后续人工治理；它不能通过改名、
复制文件或直接修改候选 manifest 获得生产资格。正式 Macro 数据只能由完整的
strict-Parquet catalog 发布流程原子绑定。

### Live primary refresh：显式、最新 session、不可由 schedule 自动执行

正式补数入口是：

```bash
quant-investor market macro-refresh \
  --market CN \
  --as-of <latest_complete_trade_date> \
  --data-root data/parquet/cn/macro_daily \
  --run-id <new_generation_id> \
  --expected-catalog-sha256 <current_catalog_sha256> \
  --expected-market-pointer-sha256 <current_market_pointer_sha256> \
  --allow-live
```

该命令只接受当前 `_latest.json` 的 latest-complete session，且 capture 必须在
上海收盘后 72 小时内。`--allow-live`、两个 expected SHA 或新 run id 缺一即
fail closed；任一 provider 月份晚于 capture 或早于声明的最晚可用月份也会阻断。
它从 exact bar files 中 terminal date 等于目标 session 的证券计算：逐证券最近
20 日收益的等权市场确认
分数、正收益 breadth、20 日等权市场年化波动率的 trailing-252 weak-rank；
`policy_signal` 只使用已保留的 M2 阈值（`>10` supportive、`<=8` restrictive、
其余 neutral），其余四个端点只作 provenance/context。

若同一 market pointer 已有有效同版本 generation，命令只返回
`already_current`，不再次调用 provider，也不创建 generation。readiness schedule
只能核对该入口的 `--help` 并给出人工补数建议，不得添加 `--allow-live`、不得
推进 catalog。

### `macro_observations`：严格 `_latest.json`

`quant_investor.macro.store.load_observations()` 的默认读取路径是
`data/parquet/cn/macro_observations`。运行时读取必须存在合法
`_latest.json`，并验证：

- pointer schema、状态与 generation id；
- pointer 与 generation manifest 都明确且严格保持
  `observer_only=true`、`production_eligible=false`、`applied=false`；
- generation Parquet 和 manifest 的相对路径、目录约束及 symlink 安全；
- pointer 声明的 Parquet/manifest SHA-256；
- generation manifest、行数、列 schema 和 content-set hash。

缺少 pointer、hash 不一致、路径越界、schema 不匹配或内容冲突都必须 fail
closed。显式 `macro-backfill-publish` 使用 CAS 和 expected hashes 推进
`macro_observations/_latest.json`；这只发布 observer observations，不会推进
`macro_daily` catalog，也不会获得生产控制权。

## Standalone 文件规则

JSON、JSONL 或 Parquet standalone observations 仅允许在用户显式调用的离线 CLI
路径中读取，例如：

```bash
quant-investor market macro-analyze \
  --market CN \
  --as-of 2024-05-10 \
  --observations private/macro/observations.parquet

quant-investor market macro-maintain \
  --market CN \
  --as-of 2024-05-10 \
  --input-observations private/macro/observations.json \
  --staging-root results/v14/macro_observation_staging \
  --run-id local_20240510
```

Standalone 读取必须由 CLI 显式 opt in；DAG runtime 只读取严格 generation
store，不把 JSON、JSONL 或孤立 Parquet 当作隐式 fallback。`--observations`
也可以指向 immutable observation-store root；observer 会固定并披露一个已校验
generation，且不会调用 provider 或推进 store。

`macro-maintain --input-observations` 只写 sanitized staging manifest，不保存
raw rows，也不保留调用方自报的 provider provenance。它不会调用
`publish_observations()`、不会创建或推进 `macro_observations/_latest.json`，并且
不能与 live provider flag 同时使用。canonical observations 只能来自显式注入且
校验通过的 provider，或独立的 hash/CAS-bound backfill-publish 流程。

Date-only `--as-of` values use the Asia/Shanghai 15:00 close as the published
cutoff. An observation is usable only when its timezone-aware `available_at`
is at or before that cutoff. `period_end` and `release_at` are not substitutes
for availability.

Generated, ignored artifacts are written atomically with mode `0600` under:

```text
results/v14/macro_observer/CN/<as_of>/<snapshot_hash>/<generation_id>/
```

For explicit standalone files without generation provenance, the final
`<generation_id>` component is omitted.

The directory contains `macro_snapshot.json`, `macro_readiness.json`,
`macro_report.md`, and `macro_observations_manifest.json`. The snapshot hash
excludes generated time and paths and includes the cutoff, registry/model
versions, and every selected PIT observation hash.

## Compatibility mart candidate maintenance

The four-field macro mart input remains available as an audit-only candidate
workflow:

```bash
quant-investor market macro-maintain \
  --market CN \
  --as-of 2024-05-10 \
  --input-json private/macro/compatibility_row.json
```

The input must contain `macro_score`, `liquidity_score`,
`volatility_percentile`, and `policy_signal`. Maintenance writes a non-production
candidate under `macro_daily/_candidates/<run_id>/`; it never advances the
market catalog or exposes the candidate to production reads. Empty input,
unimplemented live access, older snapshots, invalid schema, and interrupted
writes leave the catalog unchanged.

For observation maintenance, `--allow-live` remains blocked unless a reviewed
provider transport is injected by the caller. `--allow-tushare-fallback` is a
second explicit permission and applies only to indicators missing from the
official result. Official and fallback values are never averaged; official
provenance wins for the same indicator and period.

## PIT replay

Replay requires a local strict-Parquet trading calendar containing `cal_date`
and `is_open`:

```bash
quant-investor market macro-replay \
  --market CN \
  --start-date 2024-01-01 \
  --end-date 2024-05-10 \
  --observations-root data/parquet/cn/macro_observations \
  --calendar private/calendars/sse_trade_cal.parquet
```

The run reads and validates the latest observation generation once, then uses
that pinned in-memory set for every open date. Outputs are private mode `0600`
files under `results/v14/macro_replay/CN/<run_id>/`: `daily_snapshots.parquet`,
`replay_manifest.json`, and `replay_report.md`. The manifest binds the
observation generation/content set, pointer hash, calendar hash, registry/model
versions, date range, and output hash. Every row is `observer_only=true`,
`production_eligible=false`, and `applied=false`.

## Offline Tushare raw normalization

Phase 3 adds an offline evidence compiler for documented Tushare macro tables.
It does not import or call the Tushare client. The initial whitelist is:

| Endpoint | Raw field | Indicator | Unit |
| --- | --- | --- | --- |
| `cn_gdp` | `gdp_yoy` | `cn.gdp_yoy` | `%` |
| `cn_cpi` | `nt_yoy` | `cn.cpi_yoy` | `%` |
| `cn_ppi` | `ppi_yoy` | `cn.ppi_yoy` | `%` |
| `cn_m` | `m1_yoy` | `cn.m1_yoy` | `%` |
| `cn_m` | `m2_yoy` | `cn.m2_yoy` | `%` |
| `sf_month` | `inc_month` | `cn.social_financing_flow` | `CNY_100M` |
| `cn_pmi` | `pmi010000` / `PMI010000` | `cn.pmi_manufacturing` | `index` |

Normalization requires three separate local files: a raw Tushare response using
`macro-tushare-raw-bundle.v1`, an operator-approved scope using
`macro-backfill-plan.v1`, and an official-page capture using
`macro-release-evidence-capture.v1`. The plan is not accepted from the raw
bundle. All three source files are copied into the immutable normalization
bundle and byte-hash bound.

Availability evidence requires `time_precision=timestamp`, timezone-aware
`release_at` and `available_at`, an allowlisted official source, record ID, and
an issuer-bound HTTPS URL. GDP/CPI/PPI/PMI evidence is restricted to NBS;
money-supply and social-financing evidence is restricted to PBOC. Tushare
`cn_schedule` contains a publication date but no time. It is retained as
date-level evidence only and cannot make a row promotable. Missing, conflicting,
date-only, future, pre-period, malformed, unexpected, or incomplete scope is
quarantined and blocks publication.

Prepare a private evidence bundle without advancing canonical storage:

```bash
quant-investor market macro-normalize-tushare \
  --market CN \
  --input-json private/macro/tushare_raw_bundle.json \
  --plan-json private/macro/backfill_plan.json \
  --evidence-json private/macro/release_evidence_capture.json \
  --run-id backfill_2024q1
```

Outputs are staged atomically with directories mode `0700` and files mode
`0600` under `results/v14/macro_normalization/CN/<run_id>/`. The raw response,
backfill plan, evidence capture, observations, quarantine, receipts, and
manifest are byte-hash bound.
Inputs containing credential-like keys, URL userinfo, or credential query
parameters are rejected before any source bytes are persisted.

Only a zero-quarantine, exact-scope bundle can be explicitly published. The
caller must bind the current observation pointer with compare-and-swap:

```bash
quant-investor market macro-backfill-publish \
  --market CN \
  --manifest results/v14/macro_normalization/CN/backfill_2024q1/normalization_manifest.json \
  --observations-root data/parquet/cn/macro_observations \
  --run-id macro_backfill_2024q1 \
  --expected-pointer-sha256 EMPTY \
  --expected-manifest-sha256 <normalization-manifest-sha256> \
  --expected-plan-sha256 <backfill-plan-sha256>
```

Use `EMPTY` only for the first generation; otherwise pass the exact SHA-256 of
the current `_latest.json`. Artifact tampering, missing periods, receipt drift,
quarantine rows, or a pointer race leaves the canonical pointer unchanged. The
publisher recompiles the saved raw/plan/evidence inputs and recomputes scope,
counts, receipts, quarantine, and observations instead of trusting manifest
claims.

The remaining nine national registry indicators, official-site raw parsers,
and all twelve industry-chain raw mappings remain explicitly unsupported until
real redacted fixtures and timestamp-level publication evidence exist.

## Forward observation ledger

Phase 4 adds a forward-only evidence clock. It records the latest trading
session whose 15:00 Asia/Shanghai close has actually passed according to the
process clock and a strict local Parquet calendar:

```bash
quant-investor market macro-observe-forward \
  --market CN \
  --calendar private/calendars/sse_trade_cal.parquet \
  --observations-root data/parquet/cn/macro_observations \
  --state-root results/v14/macro_forward_observation \
  --expected-pointer-sha256 EMPTY
```

Use `EMPTY` only for enrollment. Every later run must provide the current
forward-ledger `_latest.json` SHA-256. A same-session run is idempotent only
when both the snapshot and observation generation are unchanged. A changed
same-session result, pointer race, stale calendar, skipped open session,
artifact corruption, or symlink blocks the append. There is no CLI parameter
for pretending an earlier capture time and no historical backfill path.

Each immutable generation contains a cumulative hash-chained `ledger.jsonl`,
`summary.json`, and `manifest.json`, with private mode `0600` files behind an
atomic pointer. Events bind the snapshot, selected observations, observation
generation/pointer, strict calendar, readiness, coverage, blockers, and actual
recording time.

Reaching 90 sequential entries sets only `forward_duration_reached=true`.
`measurement_maturity_reached`, `production_eligible`,
`activation_authorized`, and `applied` remain false because outcome/stability
evidence and an authoritative production score are not implemented. Degraded
sessions remain visible as readiness gaps rather than disappearing from the
clock.

## Coverage and acquisition audit

Phase 5 exposes the gap between locally cached values and genuinely usable PIT
evidence. It audits all 16 national indicators and all 96 industry components
(12 chains × 8 components):

```bash
quant-investor market macro-coverage-audit \
  --market CN \
  --as-of 2026-07-14 \
  --observations data/parquet/cn/macro_observations \
  --raw-root data/parquet/cn/dag_core_raw \
  --output-dir results/v14/macro_coverage_audit
```

The raw inventory currently recognizes only the reviewed Phase 3 endpoint
whitelist. A local raw table, value, source label, snapshot ID, or `fetched_at`
does not prove when the statistic became available. Such rows are reported as
`raw_present_pit_evidence_missing` and never counted as PIT coverage.

National indicators are classified as `pit_signal_ready`,
`observation_present_not_signal_ready`, `raw_present_pit_evidence_missing`,
`mapped_raw_not_usable_as_of`, `mapped_raw_missing`, or
`mapping_not_implemented`. Every industry chain is
expanded into output, orders, inventory, price/margin, profits, capacity
utilization, capex, and exports, so missing components cannot disappear inside
an aggregate chain score. Unconfirmed industry authorities are written as
`UNCONFIRMED` rather than inferred.

Private mode-`0600` outputs contain `coverage_audit.json`, detailed national
and industry CSVs, a Markdown report, and a hash-bound manifest under
`results/v14/macro_coverage_audit/CN/<as_of>/<audit_hash>/`. Identical reruns are
idempotent only after artifact hash readback. The audit is always observer-only
and cannot publish observations or activate production scoring.

## Offline acquisition plan

Phase 6 converts one complete, hash-valid coverage audit into a deterministic
official-data acquisition contract. It does not fetch data and never emits an
observation:

```bash
quant-investor market macro-acquisition-plan \
  --market CN \
  --coverage-audit results/v14/macro_coverage_audit/CN/<as_of>/<audit_hash>/coverage_audit.json \
  --output-dir results/v14/macro_acquisition_plan
```

The input must contain exactly all 16 national registry indicators and all
12x8 industry components. A partial but internally re-hashed audit is rejected.
The planner maps each coverage status to explicit work:

- existing raw without PIT evidence -> `bind_timestamp_release_evidence`;
- mapped but absent raw -> `acquire_raw_and_release_evidence`;
- missing official mapping -> `implement_official_mapping`;
- local market confirmation -> `build_local_strict_parquet_observation`;
- industry authority not established -> `confirm_authority_and_mapping`;
- PIT-ready observations -> `none`, retained as `satisfied`.

NBS, PBOC, Customs, MOF, and NDRC routes have explicit issuer domains. Industry
authority remains `UNCONFIRMED` until reviewed source ownership is established.
Every open official task requires an immutable raw-capture SHA-256, an
issuer-bound HTTPS URL, a source record ID, timezone-aware `release_at` and
`available_at`, capture no earlier than availability, exact
period/value/unit/frequency, and a zero-quarantine recompile.

Private mode-`0600` JSON, CSV, Markdown, and hash-manifest outputs are written
under `results/v14/macro_acquisition_plan/CN/<as_of>/<plan_hash>/`. Reruns are
idempotent and artifact drift fails closed. All outputs retain
`production_eligible=false`, `activation_authorized=false`, `applied=false`,
and `observation_count=0`.

## Deferred production gates

Production scoring remains blocked until a separate reviewed change establishes
an authoritative 0–100 score surface, 90 trading days of
forward observation, no-leakage and stability evidence, Architect/Critic
approval, and Maxwell's explicit merge/activation confirmation.

## 运行开关与默认产物

默认配置保持 observer 关闭且 kill-switched：

```text
MACRO_V2_OBSERVER_ENABLED=0
MACRO_V2_OBSERVER_KILL_SWITCH=1
MACRO_V2_PRODUCTION_ENABLED=0
MACRO_V2_PRODUCTION_KILL_SWITCH=1
MACRO_V2_OBSERVATIONS_PATH=data/parquet/cn/macro_observations
MACRO_V2_OBSERVER_OUTPUT_DIR=results/v14/macro_observer
```

即使显式开启 observer，运行结果仍固定为 non-production 且
not-applied。production 命名的开关只是可见诊断状态，不能把 observer
接入决策链。公开 CLI 默认输出隔离在 `results/v14/`：

| 命令 | 默认输出 |
| --- | --- |
| `market macro-maintain --input-observations` | `results/v14/macro_observation_staging` |
| `market macro-analyze` | `results/v14/macro_observer` |
| `market macro-replay` | `results/v14/macro_replay` |
| `market macro-normalize-tushare` | `results/v14/macro_normalization` |
| `market macro-observe-forward` | `results/v14/macro_forward_observation` |
| `market macro-coverage-audit` | `results/v14/macro_coverage_audit` |
| `market macro-acquisition-plan` | `results/v14/macro_acquisition_plan` |

## 验证清单

- 修改 offline compatibility payload 后，`data/parquet/cn/_catalog.json` 字节不变；
- 缺失或篡改 Macro catalog、generation manifest 或 SHA-256 时，生产读取
  fail closed；
- 缺失或篡改 `macro_observations/_latest.json`、generation 或 content-set
  hash 时，observer runtime 只报告 blocker；
- 启停 Macro v2 observer 前后，候选集、两项 likelihood、posterior、Markov、
  RiskGuard、IC decision、target weight 与 exposure 完全一致；
- observer artifacts、observation/forward pointers 与 manifests 始终包含
  `observer_only=true`、`production_eligible=false` 和 `applied=false`；
- Phase 6 acquisition plan 只生成 `observation_count=0` 的官方数据采集任务，
  不发起网络请求、不发布 observation、不改变决策；
- 本地验证不调用 live provider、LLM、broker 或 execution API。
