# CN aggressive 持仓与业绩 Dashboard

这是一个只读、fail-closed 的历史组合复盘界面。它只消费：

```text
results/strategy_records/CN/aggressive_tech_manufacturing/YYYYMMDD_HHMM/
```

`aggressive_tech_manufacturing` 在这里是历史展示标签，不是 V17 canonical
strategy ID。Dashboard 不读取或修改 V17 active pointer，也不选股、不生成候选、
不重算组合、不写策略记录、不连接 provider/broker、不创建订单、不执行交易。

## 两条输入闭包

exporter 只扫描名称严格匹配 `YYYYMMDD_HHMM` 的 regular directory，并把当前持仓
权威与历史业绩连续性分开验证。

### 当前持仓权威

用于当前持仓及相邻持仓变化的记录必须同时满足：

- `manifest.json` 的 `market`、`strategy`、`timestamp` 和 `source_record` 有效；
- manifest 中的 Dashboard 文件引用均指向同一记录目录；
- manifest 内嵌的 `manual_execution` 与独立
  `manual_execution_manifest.json` 完全一致；
- effective ledger 是同目录 regular non-symlink 文件，稳定双读，且声明 SHA-256
  与 exact bytes 一致；
- ledger 市值与 manual manifest、`pnl_summary.csv` 的 cash / market value /
  total value / P&L 会计口径一致；
- manifest、manual manifest、ledger、P&L 和本地 benchmark 都写入 bundle 的
  exact-byte source closure，checker 会重新读取并核对。

候选池、switch plan、analysis recommendation、orders、pending/rejected trade 不参与
持仓解析。`no_action` / `carry_forward` 只延续有效 ledger；只有 hash-bound effective
ledger 能改变展示持仓。

### 历史业绩连续性

业绩统计从 `20260317_1203` 的归档初始组合开始；第一条 P&L 是
`20260318_1146`。旧记录不会被提升为当前持仓权威，但可在以下条件全部通过后作为
`LEGACY_EXACT_BYTES_NO_DECLARED_SHA` 的 PARTIAL 业绩证据：

- manifest 的 market、strategy、timestamp、同目录文件引用和直接 source record
  均有效；
- manifest、初始 ledger、P&L 及资金补充文件均为 regular non-symlink、稳定双读，
  exporter 计算 exact-byte SHA 并纳入 bundle source closure；
- P&L 的 cash、market value、total value、capital 和 portfolio P&L 会计勾稽通过；
- 估值日期来自 P&L 明示 quote snapshot；只有满足当前 hash-bound closure 的
  no-action 记录才可使用其 manifest data date；
- 资本基数变化必须有 exact manual funding supplement。无法区分的入金/出金会让
  export BLOCKED，而不是被算入收益。

旧档没有在当时 manifest 中声明这些 SHA，因此 Dashboard 必须保持 `PARTIAL` 并明确
展示该证据等级。checker 仍会对 exporter 本次计算并写入的全部 source-ref SHA 做
exact-byte 回读。

## 收益与基准

- 跨期收益从 2026-03-17 归档基线开始，使用 funding-aware time-weighted unitization；发现 funding supplement 时，
  必须由同目录 exact artifact 与 manual manifest 内嵌对象共同闭合。
- 已验证的资金流事件会展示日期、金额、前后总资产和证据 SHA；它只调整单位份额，
  不计入投资收益，也不产生持仓或交易动作。
- 主要基准固定为本地正式沪深300 (`000300.SH`) 序列；默认输入为 Git ignored 的
  `portfolio_dashboard/inputs/cn_index_benchmark.csv`。
- 只接受 `tushare.index_daily` 或 `eastmoney.push2his.kline`；sample、mock、strategy
  snapshot 不能成为正式基准。
- 只有 source row 明示 `previous_trading_day_ffill` 且 `value_date` 早于展示日期时，
  才允许前值填充。
- 费用毛净口径、累计已实现 P&L、单股已实现 P&L 或行业/主题 exposure 没有
  hash-bound 分项证据时保持 `UNKNOWN`，不会补值。

## 状态

- `FRESH`：最新 closure、P&L/TWR、沪深300共同区间和时效均通过，没有缺口。
- `PARTIAL`：当前持仓与核心业绩可验证，但存在历史、费用、辅助维度或时效缺口。
- `BLOCKED`：有效持仓少于两期、ledger/P&L/SHA 不闭合、funding 无法区分或沪深300
  不能形成共同区间；exporter 返回非零且不刷新现有 bundle。

真实 bundle 只写 Git ignored 的：

```text
portfolio_dashboard/private/generated/cn_aggressive_dashboard.v1.json
portfolio_dashboard/private/generated/cn_aggressive_dashboard.v1.js
```

tracked schema、sample、fixture 和 tests 只含 synthetic 数据。

## 离线导出与检查

```bash
./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py
./.venv/bin/python scripts/check_cn_dashboard_export.py
./.venv/bin/python -m pytest -q tests/unit/test_cn_dashboard_export.py
node portfolio_dashboard/tests/cn_aggressive_dashboard_contract_v1.test.js
node --check portfolio_dashboard/app.js
node --check portfolio_dashboard/js/cn_aggressive_input.js
node --check portfolio_dashboard/js/cn_aggressive_dashboard_contract_v1.js
```

exporter 会先在临时目录生成 JSON/JS，运行 shape、content hash、source-ref SHA 和
readback gate，之后才替换 private bundle；发布后复核失败会恢复原 bytes。

## I1 边界

只有当前 intended ref 实际包含 I1，并且 bundle 显式绑定 exact、已验证的 I1 artifact
时才可显示 I1。没有 exact artifact 时 `i1_research` 必须为 `null`。I1 的五态始终是
research-only，不能映射为持仓变化、组合准入、买卖、订单或交易。

本页面不把这套历史组合包装成 V17 public run。
