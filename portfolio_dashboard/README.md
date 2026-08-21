# CN aggressive 持仓与业绩 Dashboard

这是一个只读、fail-closed 的历史组合复盘界面。受治理的唯一历史持仓根是：

```text
results/strategy_records/CN/aggressive_tech_manufacturing/
```

`aggressive_tech_manufacturing` 是历史展示标签；owner-declared canonical strategy ID
是 `cn-aggressive-tech-manufacturing`。身份映射本身不创建 V17 active pointer、Factor
admission、risk/publisher permit，也不授予 broker、order、execution 或 trade authority。

## Registered Store v3 合同

生产 Dashboard 只通过 `_record_store/current.v1.json` 选择 immutable
`myquant.strategy_record_catalog.v3` generation。current、previous、lineage 和业绩均不得
按目录名、mtime、文件名中的 “latest”、旧报告或聊天记忆推断。

Catalog v3 必须同时闭合：

- pointer 与 catalog 的 active/previous record 完全一致；
- lineage 无环、无有效父节点分叉、父记录全部存在，active record 只有一条 ancestry；
- active 与 previous 都是 ONLINE record，且 effective ledger 精确为同记录目录的
  `ledger_after_manual_switch.parquet`；
- manifest、manual manifest、Parquet ledger、P&L 与 financial-state SHA 逐项闭合；
- `performance_history_ref` 精确绑定 immutable generation 下的
  `manifest.v1.json`、`series.parquet` 与 `owner_declaration.v1.json`；
- performance 最后一行的 record、valuation date、manual SHA、Parquet ledger SHA 和
  financial-state SHA 与 active record 完全一致。

Catalog v1/v2 仍可由 Store manager 做 immutable storage audit，但
`performance_contract_ready=false`。生产 Dashboard 遇到 v1/v2 固定返回
`CANONICAL_PERFORMANCE_CLOSURE_MISSING`，不会启动旧历史 scanner、解压 archive、读取
旧 bundle 或按目录回退。legacy Dashboard 代码只允许显式 synthetic fixture 使用。

## 为什么不读取 legacy `ledger.csv`

禁用原因不是 CSV 格式本身有问题，也不是断言旧文件一定损坏。该文件曾经是与
registered pointer/catalog/manual-manifest/financial-state closure 并行的 legacy
authority；继续让运行时咨询它会重新引入双 ledger 漂移、目录猜测和 fallback 风险。
事后补登记 SHA 也不能把过去未由控制面选择的运行路径追溯升级为当前 authority。

持续更新只能由授权的 record producer 创建 immutable financial-state record，Store
manager 验证 Parquet closure 后，用 expected-parent pointer CAS 原子推进 record、lineage、
performance generation 与 catalog。无交易但产生新正式估值或新 financial-state SHA 时
同样追加业绩点；纯 receipt 型 no-action、没有新 financial state 时继承原 performance
ref。Dashboard 和周度 automation 只有读取权，不能创建估值或推进 pointer。

一次性 v2 → v3 迁移只消费 registered catalog v2 内嵌
`dashboard_projection.historical_records` 的归一化经济字段。migration adapter 不解析
其中的 `source_refs`，不打开任何 record ledger，不扫描 raw tree，也不读取 archive 或
旧 Dashboard bundle。

## Canonical 业绩与资金流

唯一业绩闭包位于：

```text
_record_store/performance/<performance-generation-id>/
  manifest.v1.json
  series.parquet
  owner_declaration.v1.json
```

历史 seed 固定复现 owner-corrected 的 100 万初始资本路径：旧 funding correction 只反转
其明确指定的错误资金流；旧逐点值不会被另一套公式重算覆盖。当整条 canonical series
逐点满足 `adjusted total = raw total - excluded flow`、起点为 100 万且最终外部流为 0 时，
Dashboard 固定报告 `initial_capital_return_excluding_external_flows`，并将已撤销的错误资金
事件从 `history.funding_events` 折叠为空。该折叠只改变展示语义，不改 Store 的 immutable
provenance bytes。

只有未来真实外部资金流使上述逐点恒等式不再成立时，新 v3 financial state 才使用
flow-neutral unitization：

```text
units_delta = amount_cny / pre_flow_unit_nav
post_flow_units = pre_flow_units + units_delta
post_flow_unit_nav = post_flow_nav / post_flow_units
```

每笔入金或赎回必须有 owner-declared cash-flow artifact，并精确绑定前后 record、NAV、
financial-state SHA、matching manual manifest，以及“除现金外持仓股数不变”的证明。
资金流本身不得改变 unit NAV；缺任一闭包时只阻断 performance/benchmark 域，不把资金流
静默置零。

Dashboard 的累计收益、区间收益与回撤直接消费 canonical unit NAV。为了兼容 v1 前端，
`adjusted_total_value` 映射为 `unit_nav × initial_units`；累计外部资金流则直接读取 series
字段，禁止用 `raw NAV - adjusted NAV` 反推，因为那会把新增资金后来赚取的收益误当成
资金流。

## 沪深300与证据边界

正式主要基准固定为：

```text
portfolio_dashboard/inputs/cn_index_benchmark.csv
ts_code = 000300.SH
```

必须报告文件 SHA、source system、共同区间、coverage、value date，以及
`exact_close` / `previous_trading_day_ffill` 语义。网页指数报价、sample、mock、strategy
snapshot 和零值补齐都不能成为正式基准。若 benchmark 只缺 canonical performance 的
末尾日期，组合 canonical return 仍可单独存在，v1 保持 `PARTIAL`，v2 的持仓和绝对
业绩可独立表态；缺失尾部的 benchmark NAV、return、excess 与 drawdown 字段固定为
`null/unavailable`，同时披露最后 exact benchmark date 与 `missing_dates`。历史中段
缺口仍然 BLOCKED，禁止前值、零值或网页报价补成正式相对业绩。

记录中的 `current_price` 只是该 record 的闭合价格，页面必须同时显示 `price_date`；它
不是实时行情。费用毛净口径、累计已实现 P&L、单股已实现 P&L、行业或主题暴露缺少
exact evidence 时保持 `UNKNOWN`。

## 页面与公开边界

内部页面展示组合/沪深300同轴累计收益、相邻超额、回撤、持仓变化、现金与权益暴露、
集中度及 exact source evidence。资金流只作为事件标记，不作为收益；关键数字无需 hover
即可读取。图表使用本地原生 SVG，不加载 CDN 或网络数据，并保留键盘、触控和
screen-reader 摘要。

公开静态副本 `public.html` 只保留允许公开的业绩视图。public-site builder 会从 bundle
bytes 中删除内部持仓、变化、绝对资产、资金流金额、路径与 SHA；公开边界不依赖 CSS
隐藏。

## 状态

- `FRESH`：Store v3、current/previous、canonical performance、基准共同区间与 bundle
  checker 均闭合；仍保留 as-of、record ID 和证据边界。
- `PARTIAL`：可验证数据仍可展示，但存在时效、benchmark尾部、非核心口径或 QA warning；
  页面不得称为实时或正式 V17 组合。
- `BLOCKED`：Store/pointer/catalog/performance closure、active-row reconciliation 或
  bundle checker 失败。exporter 返回非零，不能把磁盘旧 bundle 当成本轮结果。

真实 bundle 只原子刷新 Git ignored 的 private artifact set：

```text
portfolio_dashboard/private/generated/cn_aggressive_dashboard.v1.json
portfolio_dashboard/private/generated/cn_aggressive_dashboard.v1.js
portfolio_dashboard/private/generated/cn_aggressive_dashboard.v2.json
portfolio_dashboard/private/generated/cn_aggressive_dashboard.v2.js
portfolio_dashboard/private/generated/cn_aggressive_dashboard_selector.v2.json
portfolio_dashboard/private/generated/cn_aggressive_dashboard_selector.v2.js
```

Exporter 不接受任意 output path：六个 serving artifact 必须使用这些固定文件名，
并且在 `portfolio_dashboard/private/generated/` 内。它会在任何写入前拒绝 public、
System、Store、Market/Data 和 symlink escape 路径。刷新失败不再改写 serving selector，
而是在 `private/generated/attempts/` 追加 immutable failure receipt；只有 v1/v2/checker
全部通过后才最后更新 selector。

## 离线导出与检查

```bash
./.venv/bin/python scripts/manage_cn_strategy_records.py verify \
  --record-root results/strategy_records/CN/aggressive_tech_manufacturing
./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py
./.venv/bin/python scripts/check_cn_dashboard_export.py
./.venv/bin/python -m pytest -q \
  tests/unit/test_cn_dashboard_export.py \
  tests/unit/test_cn_dashboard_export_v2.py \
  tests/unit/test_cn_dashboard_v2.py \
  tests/unit/test_cn_dashboard_v2_selector.py
node portfolio_dashboard/tests/cn_aggressive_dashboard_contract_v1.test.js
node portfolio_dashboard/tests/cn_aggressive_dashboard_contract_v2.test.js
node --check portfolio_dashboard/app.js
node --check portfolio_dashboard/js/cn_aggressive_input.js
node --check portfolio_dashboard/js/cn_aggressive_dashboard_contract_v1.js
node --check portfolio_dashboard/js/cn_aggressive_dashboard_contract_v2.js
```

`build_cn_dashboard_history_integrity.py` 和 legacy scanner 不属于 registered governed-root
生产路径；不得用它们为 v1/v2 恢复业绩 authority。

## I1 边界

只有当前 intended ref 实际包含 I1，并且 bundle 显式绑定 exact、已验证的 I1 artifact
时才可显示 I1。没有 exact artifact 时 `i1_research` 必须为 `null`。I1 的五态始终是
research-only，不能映射为持仓变化、组合准入、买卖、订单或交易。

本页面不把历史 Store、身份声明或 Dashboard 包装成 V17 public run。
