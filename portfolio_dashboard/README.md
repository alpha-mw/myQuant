# myQuant Dashboard v3

本目录是浏览器本地静态组合看板，不包含后端、交易接口或自动执行。

## 数据边界

- 真实数据只写 Git ignored 的 `private/` 与 `generated/`，原子落盘且
  mode `0600`。
- 正式私有入口是 `private/dashboard_snapshot.v3.json` 和 `.js`。
- tracked `js/generated_records.js` 是无真实账户数据的 sanitized loader；
  `sample/dashboard_snapshot.v3.json` 是合成样例。
- 真实 ticker、资金、持仓、绝对路径和交易不得进入 Git、URL 或
  `localStorage`。

## Contract v3

必需表固定为：

- `nav`
- `positions`
- `trades`
- `industries`
- `factors`

行业仅来自 strict `stock_basic.industry`。缺失值保持 `null`，不得从名称、
旧记录字段或目录推断。执行基线只认有效 `manual_execution_manifest.json`
指向的 contained ledger；`ledger.csv` 不得作为 fallback。

Snapshot 同时绑定 schema/hash、run/status/blockers、as-of matrix、正式交易
日历、NAV/fee/TWR provenance、manual manifest/ledger SHA、Factor v3、每日
reconciliation 和 v15 readiness 引用。CN 交易日历与行业等权 bars 只通过
`MarketDataReader` 校验 active `data/parquet/cn/_latest.json` 和 clean gate 后
读取 pointer 指向的 immutable
`_snapshots/<snapshot_id>/table/bars`；snapshot 同时记录 pointer path/SHA、
snapshot id 和 table root。固定 `data/parquet/cn/bars`、serving root 与 CSV
fallback 均禁止；`--benchmark-source tushare` 也只读取指数 close，不再调用
Tushare `trade_cal` 绕过 canonical 日历。

## 生成与校验

```bash
PYTHONPATH=. ./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py \
  --benchmark-source local \
  --benchmark-file portfolio_dashboard/inputs/cn_index_benchmark.csv \
  --data-root data

node portfolio_dashboard/tests/dashboard_contract_v3.test.js
PYTHONPATH=. ./.venv/bin/python scripts/check_cn_dashboard_export.py \
  --dashboard-root portfolio_dashboard
```

导出后必须 readback 私有 JSON/JS、五张 Contract CSV
`generated/nav_records.csv`、`positions_records.csv`、`trades_records.csv`、
`industries_records.csv`、`factors_records.csv`，以及 `export_summary.json`。
`benchmark_records.csv` 是附加 benchmark audit 表。确认 tracked loader/sample
未变化；缺日期、价格、费用、provenance 或 reconciliation 时保持
`null/partial/blocked`。
