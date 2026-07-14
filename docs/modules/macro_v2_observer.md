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
  --as-of YYYY-MM-DD \
  --observations /path/to/local-observations.json

quant-investor market macro-maintain \
  --market CN \
  --as-of YYYY-MM-DD \
  --input-observations /path/to/local-observations.json \
  --staging-root results/v14/macro_observation_staging \
  --run-id <run_id>
```

这些命令的 standalone 读取是显式 offline opt-in；DAG runtime 不开启该选项，
只调用严格 generation store。standalone 文件不能作为隐式 fallback，也不能绕过
`_latest.json`、CAS、hash 或 PIT 校验。

其中 `macro-maintain --input-observations` 只写 sanitized staging manifest：不保存
raw rows，也不保留调用方自报的 `source_system`、URL、record id 或 provider
provenance；manifest 固定为 observer-only、non-production、not-applied。该命令
不会调用 `publish_observations()`，不会创建或推进
`macro_observations/_latest.json`，并且不能与任何 live provider flag 同时使用。
canonical observations 只能来自显式注入且校验通过的 provider，或独立的
hash/CAS-bound backfill-publish 流程。

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

即使显式开启 observer，运行结果仍固定为 non-production、not-applied。production
命名的两个开关只作为可见诊断状态保留，当前 v14 contract 不允许它们把 observer
接入决策链。

公开 CLI 的默认输出均隔离在 `results/v14/`：

| 命令 | 默认输出 |
| --- | --- |
| `market macro-maintain --input-observations` | `results/v14/macro_observation_staging` |
| `market macro-analyze` | `results/v14/macro_observer` |
| `market macro-replay` | `results/v14/macro_replay` |
| `market macro-normalize-tushare` | `results/v14/macro_normalization` |
| `market macro-observe-forward` | `results/v14/macro_forward_observation` |
| `market macro-coverage-audit` | `results/v14/macro_coverage_audit` |

这些目录是报告、证据和测量状态，不是 canonical market-data pointer。任何产物都不应
被 RiskGuard、Markov 或 portfolio construction 当作决策输入。

## 验证清单

- 修改 offline compatibility payload 后，`data/parquet/cn/_catalog.json` 字节不变；
- 缺失或篡改 Macro catalog、generation manifest 或 SHA-256 时，生产读取 fail
  closed；
- 缺失或篡改 `macro_observations/_latest.json`、generation 或 content-set hash
  时，observer runtime 只报告 blocker；
- 启停 Macro v2 observer 前后，候选集、两项 likelihood、posterior、Markov、
  RiskGuard、IC decision、target weight 与 exposure 完全一致；
- observer artifacts、observation/forward pointers 与 manifests 始终包含
  `observer_only=true`、`production_eligible=false` 和 `applied=false`；
- CLI 默认输出路径均位于 `results/v14/`，且本地验证不调用 live provider、LLM、
  broker 或 execution API。
