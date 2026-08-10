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
  `manual_execution_manifest.json` 完全一致；对 2026-08-04 至 2026-08-06 的
  historical revaluation carry-forward，允许使用同目录
  `backfill_provenance.json` 代替内嵌副本，但必须逐项绑定 manifest、独立 manual、
  effective ledger、P&L 和外部 automation history 的 exact bytes 与 SHA-256；
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
- 资本基数变化必须有 exact manual funding supplement，或有 exact、hash-bound 的
  owner correction 明确撤销错误 funding。无法区分的资本变化会让 export BLOCKED，
  而不是被算入收益。

旧档没有在当时 manifest 中声明这些 SHA。运行
`build_cn_dashboard_history_integrity.py` 后，Dashboard 会生成独立的私有 SHA 登记表，
逐条绑定实际参与业绩计算的文件；exporter 和 checker 会重新双读文件并核对登记表。
该登记表是事后完整性声明，不改写旧 manifest，也不会把旧记录提升为当前持仓权威。

## 收益与基准

- 跨期收益从 2026-03-17 的 ¥1,000,000 唯一资本基线开始；没有后续注资或出金。
  每个估值点用 `组合总资产 / 1,000,000 - 1` 计算累计收益。
- 旧归档中的 2026-07-09 funding supplement 已被最新 hash-bound owner correction
  明确撤销。更正前的旧点仅依据这条 correction 重建 100 万经济路径；更正后的最新
  record 直接使用已纠正 cash / total value，禁止再次扣减。最终 `funding_events` 为空，
  `net_external_flow` 和 current `excluded_external_flow` 均为 0。
- 当前 `portfolio.cash`、`portfolio.total_value`、现金权重、股票仓位、持仓 NAV 权重、
  组合 P&L 和换手分母统一使用上述 100 万经济组合口径。原始归档账户数值只保留在
  hash-bound accounting evidence 与 performance point 的勾稽字段中，不进入投资展示。
- owner correction 必须验证 correction 类型、被撤销 record 与金额、100 万初始资本、
  no-trade proof、manual manifest SHA 和会计恒等式；同一错误 funding 的重复
  superseding correction 只能应用一次，且不产生持仓或交易动作。
- 基准固定为本地正式沪深300 (`000300.SH`)、科创50 (`000688.SH`) 和创业板指
  (`399006.SZ`) 序列；沪深300是超额收益的主要基准，后两项是科技风格辅助对照。
  默认输入为 Git ignored 的 `portfolio_dashboard/inputs/cn_index_benchmark.csv`。
- 只接受 `tushare.index_daily` 或 `eastmoney.push2his.kline`；sample、mock、strategy
  snapshot 不能成为正式基准。
- 只有 source row 明示 `previous_trading_day_ffill` 且 `value_date` 早于展示日期时，
  才允许前值填充。
- 费用毛净口径、累计已实现 P&L、单股已实现 P&L 或行业/主题 exposure 没有
  hash-bound 分项证据时保持 `UNKNOWN`，不会补值。

## 页面与历史业绩分析

页面以投资业绩比较为首屏，而不是以持仓长表为首屏：

- 主图把组合净值、沪深300、科创50、创业板指和对沪深300累计超额从
  2026-03-17 的共同起点归一化为 100；页面不显示资金事件竖线或图表标注；
- 水下回撤图分别从组合与三项基准各自历史高点计算回撤；
- 量化评价 scorecard 直接从同一组已验证估值点派生预计年化收益、年化波动率、
  Sharpe、Sortino、Calmar、沪深300 Beta/相关系数、跟踪误差、信息比率和正收益
  区间占比。预计年化收益按实际日历跨度做几何折算；其他风险指标按相邻验证区间
  的实际观测频率年化；Sharpe 使用财政部/中债发布的中国 1 年期国债收益率，
  每个收益区间按区间起点对应的正式收益率用 ACT/365 折算，非公布日仅使用上一
  已公布工作日；所有年化值均为历史折算，
  不是未来收益承诺；
- 月度业绩表直接从已验证的 `performance_points` 派生月度组合收益、三项基准收益、
  对沪深300超额和月内最大回撤，不读取新数据、不改变 bundle，也不重算持仓；
- 当前持仓使用 NAV 配置条和精确表格展示，最近变化单列；
- 完整历史估值明细、证据路径与 SHA 使用可展开的二级区域，避免遮挡首屏结论。

图表由本地原生 SVG 渲染，不加载 CDN 或网络数据。鼠标、触摸和键盘左右键都可按日
查看日期、组合净值、三项基准净值、对沪深300累计超额和回撤；关键数字仍保留在摘要与月度表中，
不依赖 hover 才能读懂。

公开静态副本使用 `public.html`，只含四个区块：品牌页眉、业绩总览、月度拆解、历史
复盘。它不包含内部持仓、变化、会计、集中度、风险、历史明细或证据节点。
`build_cn_dashboard_public_site.py` 还会从 bundle bytes 中删除股票身份与持仓细节、
绝对资产、资金流金额、路径和 SHA；公开边界不依赖 CSS 隐藏。

## 状态

- `FRESH`（页面显示 `UPDATED`）：最新 closure、P&L、100 万唯一资本收益、三项正式基准共同区间和时效均通过，且最新正式业绩日期必须等于 Dashboard 运行日期。费用、累计已实现盈亏、行业等非核心口径仍可作为 warning 明示，但不会把完整的核心数据错误降级为 `PARTIAL`。
- `PARTIAL`：当前持仓仍可展示，但核心链存在未闭合的旧历史 SHA、最新正式业绩日期早于 Dashboard 运行日期、当前持仓只有未完成的交易标记估值，或存在更新但不可用的记录。
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
./.venv/bin/python scripts/build_cn_dashboard_history_integrity.py
./.venv/bin/python scripts/build_cn_dashboard_risk_free_input.py
./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py
./.venv/bin/python scripts/check_cn_dashboard_export.py
./.venv/bin/python scripts/build_cn_dashboard_public_site.py /private/tmp/cn-dashboard-public
./.venv/bin/python -m pytest -q tests/unit/test_cn_dashboard_export.py
node portfolio_dashboard/tests/cn_aggressive_dashboard_contract_v1.test.js
node --check portfolio_dashboard/app.js
node --check portfolio_dashboard/js/cn_aggressive_input.js
node --check portfolio_dashboard/js/cn_aggressive_dashboard_contract_v1.js
node --check portfolio_dashboard/js/cn_aggressive_dashboard_analysis_v1.js
node --check portfolio_dashboard/js/cn_aggressive_public_mode.js
```

exporter 会先在临时目录生成 JSON/JS，运行 shape、content hash、source-ref SHA 和
readback gate，之后才替换 private bundle；发布后复核失败会恢复原 bytes。

## I1 边界

只有当前 intended ref 实际包含 I1，并且 bundle 显式绑定 exact、已验证的 I1 artifact
时才可显示 I1。没有 exact artifact 时 `i1_research` 必须为 `null`。I1 的五态始终是
research-only，不能映射为持仓变化、组合准入、买卖、订单或交易。

本页面不把这套历史组合包装成 V17 public run。
