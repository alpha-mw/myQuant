# myQuant Runtime Paths and Artifacts

这一页回答“当前 lane 从哪里读、产物在哪里、排障先看什么”。

## V17 mainline authority

```text
results/v17_mainline/strategies/<strategy-id>/
  _active.json
  runs/<run-id>/run.json
```

- `_active.json` 是公开可见性的唯一 authority carrier。
- public reader 精确解析 pointer 的 run ref 和 transitive closure。
- 缺失 pointer 返回 unavailable；无效链返回 blocker；都不得写 bootstrap、
  fallback 或 cache。
- 当前仓库没有公开 mainline publisher/activation command。包内公开的低层
  exact-once/CAS storage primitives 是可写库能力，但不是受治理的生产操作入口
  或 production authority。

## V4 Forward and Shadow

显式 request 位于：

```text
data/private/v17_v4_runs/forward_requests/
```

Shadow 产物使用自己的 V4 Forward roots，并由 exact path、byte SHA、semantic
SHA 和 receipt 链约束。它们不是 `results/v17_mainline/` 的 authority，不能
替代 active pointer。

## I0 and R2.2

R2.2 请求与显式输入位于：

```text
data/private/research_intelligence/evaluation_requests/
data/private/research_intelligence/evaluation_inputs/
```

- I0 evidence、hypothesis、regime input/receipt 和 memory inventory 都是显式、
  content-addressed 输入。
- R2.2 输出只在 stdout envelope 中返回；仓库不为它写 results 或 memory。
- memory append proposal 的持久化与 CAS 属于 caller，不属于 evaluator。
- 排障先检查 exact request path/SHA、origin closure、cutoff、artifact refs 和
  blocker code，不扫描目录找“最近一次”。

## Investment Intelligence v2

I2-I6 不扫描 runtime 目录。准备发布的 immutable v2 artifact root 是：

```text
results/v17_intelligence_v2/
```

ACTIVATE permit 与 quarantine marker 使用从 exact pointer/run 派生的固定路径：

```text
results/v17_mainline/strategies/<strategy-id>/activation_permits/<pointer-sha>.json
results/v17_mainline/strategies/<strategy-id>/quarantine/<run-id>.json
```

这些路径是契约，不是执行授权。缺少真实 Factor v5 prospective evidence、
Industry/Theme catalog、owner policy、ZDR capability、签名或成熟 Paper outcome 时，
保持 `RESEARCH_MAINLINE_CANDIDATE_BLOCKED`，不得用 fixture 替代。

## Canonical market data

生产/评审路径使用严格 Parquet canonical pointer、manifest 和 readback。常见
本地数据库或服务层数据源包括：

```text
data/stock_database.db
```

具体位置可被仓库配置覆盖，必须以当前 config 和 pointer 为准。CSV 只能用于
明确的导出或迁移，不是 canonical fallback。

维护、`storage-validate`、serving materialization、feature materialization 和
decision-mainline read 是不同步骤。单独看到一个验证成功行不能证明整个维护
或 mainline run 已完成。

## Display surface

展示面只有 `portfolio_dashboard/` 静态页，读取 exporter 生成的
`portfolio_dashboard/private/generated/cn_aggressive_dashboard.v1.js`。
它是只读的，不写 strategy record、不调用 provider。

以下是已删除 Web 工作台留下的旧产物，不是当前 authority；不要读取：

- `data/web_runs.db`
- `data/app.db`
- `results/web_analysis/`
- `data/workspace_learning/`

## Configuration

- repo root `.env` 是本地配置来源；不要输出 token 或 credentials。
- 已退休的 Fundamental overlay、Markov production overlay、旧 pipeline/engine
  和 total-timeout key 会 fail loudly；迁移见
  `docs/runbooks/v17_legacy_configuration_cleanup.md`。
- 保留的 Funnel/Bayesian shortlist/agent timeout 配置被 standalone automation
  消费，但这不表示该 automation 已注册为 V17 runtime。

## Diagnostic order

1. 确认 lane：mainline read、maintenance、Forward、R2.2 或 legacy。
2. 确认 exact path、SHA、schema、strategy、cutoff 和 pointer。
3. 检查 manifest/readback、receipt 和 blocker。
4. 确认零写入边界；不要以缓存、旧 run 或可选输入降级掩盖错误。
