# CN / aggressive_tech_manufacturing Performance Dashboard

本项目是一个本地静态网页版投资组合业绩 Dashboard，用于复盘
`CN/aggressive_tech_manufacturing` 的组合业绩、归因、风险暴露、持仓主题和交易记录。

页面会优先加载已导出的本地策略记录数据；如果没有生成记录包，则回退到模拟的 sample data，并在界面中标注“示例数据”。示例数据不代表真实业绩。sample benchmark 为模拟数据，仅用于演示，不代表真实指数。

## 如何本地打开

1. 解压或复制整个 `portfolio_dashboard/` 目录。
2. 双击 `index.html`。
3. 浏览器打开后会自动显示本地记录数据；若尚未生成记录包，则显示 sample data。
4. 可通过顶部按钮上传自己的 `nav.csv`、`positions.csv`、`trades.csv`。

不需要后端、数据库、Node.js build、登录或联网。

## 页面交互

- 顶部导航可快速跳转到总览、Benchmark 对比、月度、归因、风险、持仓和交易模块。
- 主基准 dropdown 控制超额收益、rolling beta 和 KPI 中的 benchmark 口径。
- 多 benchmark 对比 multi-select 控制净值、回撤、风险收益散点、相关性和 comparison table。
- 业绩总览中的 Lens 按钮可在多 benchmark 净值、相对主基准超额收益、回撤和 20D 波动率之间切换主图。
- 持仓表和交易表支持本地即时搜索，不会上传查询内容。
- 上传的 CSV 会保存在浏览器 `localStorage` 中；刷新页面后优先恢复用户上传数据，点击“重置示例数据”会清除这份本地保存。
- 日期区间和 Benchmark 变化后，KPI、图表、归因、风险与表格会同步刷新。

## 从 strategy_records 自动更新

浏览器出于安全限制，不能在双击打开的 `file://` 页面中自动扫描本机目录。因此本项目使用一个离线导出脚本读取仓库内的策略记录，并生成静态页面可直接加载的数据包。

在仓库根目录运行。推荐使用仓库虚拟环境，这样可以读取本地 Parquet `stock_basic` 并补齐 `sector`：

```bash
./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py
```

如果只需要 NAV、持仓权重、theme 和交易复盘，也可以使用系统 Python：

```bash
python3 scripts/export_cn_aggressive_dashboard_data.py
```

脚本默认读取：

```text
results/strategy_records/CN/aggressive_tech_manufacturing/
```

并写入：

```text
portfolio_dashboard/generated/nav_records.csv
portfolio_dashboard/generated/positions_records.csv
portfolio_dashboard/generated/trades_records.csv
portfolio_dashboard/generated/benchmark_records.csv
portfolio_dashboard/generated/export_summary.json
portfolio_dashboard/js/generated_records.js
```

双击 `portfolio_dashboard/index.html` 时，页面会自动加载 `js/generated_records.js` 中的本地记录数据，并在顶部显示“当前使用本地记录数据”。每日更新时，在新的策略记录目录生成后重新运行上述脚本即可刷新 Dashboard。

导出后建议运行只读校验，确认 `generated_records.js` 能解析、不会回退到 sample data，并核对 `export_summary.json` 的行数和 benchmark provenance：

```bash
./.venv/bin/python scripts/check_cn_dashboard_export.py
```

如果本次数据包要作为正式投委会 benchmark 口径，还需要强制要求 production-grade benchmark：

```bash
./.venv/bin/python scripts/check_cn_dashboard_export.py --require-production-benchmark
```

当前本地真实指数 close 尚未补齐时，这个强制命令会失败；这表示 Dashboard 展示数据可用，但 benchmark 仍不是正式投委会口径。

当前导出逻辑采用保守映射：

- NAV 来自每个记录目录的 `pnl_summary.csv`；`portfolio_nav` 使用份额化的时间加权净值。外部入金/出金按资金证据中的流前单位净值申购/赎回份额，入金瞬间不改变单位净值；资金证据必须包含 `total_value_before` 和 `total_value_after`，否则导出 fail closed
- `portfolio_nav_raw = total_value_after / initial_capital` 仅保留为账户资金基准比值，不用于跨资金基准的累计收益；资金基准发生变化但缺少 `manual_execution_manifest.json` 的有效资金证据时，导出 fail closed
- Benchmark 默认使用 local-first `auto`：优先读取 `portfolio_dashboard/inputs/cn_index_benchmark.csv` 中的本地真实指数 close；本地文件不存在时才尝试 `tushare` source；当前支持 `000300.SH -> csi300_nav`、`000905.SH -> csi500_nav`、`000852.SH -> csi1000_nav`、`000688.SH -> star50_nav`、`399006.SZ -> chinext_nav`
- 本地 benchmark 文件中的非交易日记录需要显式写 `coverage=previous_trading_day_ffill` 和真实 `value_date`；只要真实交易日 close 覆盖完整，这类非交易日前向填充仍可通过 production-grade 校验；交易日缺 close 仍保持缺失并降级，不做静默填充
- `benchmark_main_nav` 由可用的 `star50_nav`、`csi300_nav`、`chinext_nav` 组合生成；若本地 benchmark 或 Tushare 不可用，或交易日记录日期没有指数 close，`export_summary.json` 会降级标记为非 production-grade，不会伪造数据
- `benchmark_records.csv` 保存真实 close、归一化 NAV、`value_date` 和 `coverage`，便于审计 exact close 与 previous-trading-day ffill；如需完全离线旧口径，可运行 `CN_DASHBOARD_BENCHMARK_SOURCE=snapshot ./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py`
- 如果需要显式在线读取 Tushare `index_daily`，可运行 `./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py --benchmark-source tushare`；自动化维护默认不需要这个路径。若 Tushare 对部分真实交易日缺指数 close，可用 `scripts/backfill_cn_dashboard_benchmark.py --source eastmoney --fill-only-missing` 只补缺口并保留 `source_system=eastmoney.push2his.kline`
- 如果已经有 Wind、Choice、iFinD、Bloomberg、Tushare 离线导出或内部数据库生成的真实指数 close CSV，可用本地 benchmark 输入：

```bash
./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py \
  --benchmark-source local \
  --benchmark-file portfolio_dashboard/inputs/cn_index_benchmark.csv
```

本地 benchmark CSV 必需字段：

```csv
date,ts_code,close,source_system
```

可选字段：

```csv
coverage,value_date
```

本地文件至少要覆盖 `000300.SH`、`000905.SH`、`000852.SH`、`000688.SH`、`399006.SZ`。`close` 必须为正数，`source_system` 必须是真实来源；`sample`、`mock`、`demo` 或 `strategy_record.market_snapshot.indices` 会被拒绝，不能标记为 production-grade。非交易日如使用上一交易日 close，需要显式写 `coverage=previous_trading_day_ffill` 和真实 `value_date`；交易日缺口或 snapshot 补齐仍会降级。
- 持仓只来自可读 `ledger_after_manual_switch.csv` 与对应 `manual_execution_manifest.json`；缺失时跳过该记录，绝不回退到停用的 `ledger.csv`
- 个股日收益优先来自同目录 `holdings_review.csv` 的 `today_change_pct`
- 旧版 `metric,value` 纵表格式 `pnl_summary.csv` 会自动转成宽表指标；`initial_capital` 缺失时仅从同目录 `market_snapshot.json` 的组合总值和 PnL 推导
- 持仓主题优先来自当日 `market_snapshot.json` 的 `theme_strength.symbols` 及 candidate/switch plan theme；当日缺失时可沿用截至该日最近正式 strategy record 的显式 theme（只前向继承，不使用未来记录），最后才使用 `行业: <sector>` 透明回退标签
- `sector` 来自本地 `data/parquet/cn/dag_core_raw/table=stock_basic` 的 `industry` 字段；如果运行环境没有 `pyarrow`，该字段会留空并给出 warning
- 交易仅导出 `manual_switch_and_take_profit_orders.csv` 中已填成交价格和数量的记录，不把建议订单强行当成成交
- 记录缺少个股主题且无法从行业映射回退时，`theme` 才写为 `UNSPECIFIED_RECORD_THEME`

## CSV 字段说明

### nav.csv

最低字段：

```csv
date,portfolio_nav
```

推荐多 benchmark 字段：

```csv
date,portfolio_nav,benchmark_main_nav,csi300_nav,csi500_nav,csi1000_nav,star50_nav,chinext_nav,semiconductor_nav,robotics_nav,high_end_manufacturing_nav
```

可选字段：

```csv
portfolio_nav_raw,portfolio_nav_rebased,portfolio_return,portfolio_units,initial_capital,total_value_after,external_funding_cash_flow,benchmark_return,cash_weight,gross_exposure,net_exposure
```

- `date`: `YYYY-MM-DD`
- `portfolio_nav`: 现金流调整后的组合单位净值，用于收益、回撤和图表
- `portfolio_nav_raw`: `total_value_after / initial_capital` 的账户资金基准比值；资金扩容后会重置，不用于长期收益
- `portfolio_nav_rebased`: `portfolio_nav` 相对首条展示记录归一化后的净值
- `portfolio_units`: 份额化时间加权口径下的期末份额；新增资金按流前单位净值申购份额
- `initial_capital` / `total_value_after`: 当期资金基准与账户总资产
- `external_funding_cash_flow`: 来自正式手工执行 manifest 的当日外部资金流，已从收益计算中剔除
- `benchmark_main_nav`: 默认主基准净值
- `benchmark_nav`: 旧格式基准净值，仍兼容
- 任意字段名只要以 `_nav` 结尾且不是 `portfolio_nav`，都会自动识别为 benchmark
- `portfolio_return`: 组合日收益率，缺失时按 `portfolio_nav / previous_portfolio_nav - 1` 计算
- `benchmark_return`: 旧 `benchmark_nav` 的日收益率，缺失时按 NAV 自动计算
- `cash_weight`: 现金比例
- `gross_exposure`: 总敞口
- `net_exposure`: 净敞口

正式使用时，请使用 Wind、Choice、iFinD、Bloomberg、Tushare 或内部数据库的真实指数 close，并归一化为 NAV：

```text
benchmark_nav = benchmark_close / benchmark_close[first_valid_date]
```

新增 benchmark 只需要在 `nav.csv` 中增加一列，例如 `satellite_nav`。字段名以 `_nav` 结尾后，页面会自动识别；显示名可在 `js/data.js` 的 `BENCHMARK_LABELS` 中自定义。主基准可以使用 `benchmark_main_nav`，也可以在页面 dropdown 中切换。

### positions.csv

最低字段：

```csv
date,ticker,name,weight,theme
```

可选字段：

```csv
theme_source,sector,sub_sector,daily_return,contribution,market_value,quantity,avg_cost,cost_basis,current_price
```

- `weight`: 持仓权重；裸数字按小数权重解析，`0.08` 表示 8%，`1.5` 表示 150%；百分比请写 `8%`
- `weight_unit`: 可选，支持 `decimal` 或 `percent`；当 `weight_unit=percent` 时，`8` 会解析为 8%
- `daily_return`: 个股当日收益率
- `theme_source`: 主题来源，区分当日正式记录、历史正式记录前向继承、`stock_basic.industry` 回退或不可用
- `contribution`: 当日收益贡献；缺失但有 `weight` 和 `daily_return` 时，用 `weight × daily_return` 估算
- `quantity` / `shares`: 持仓股数；用于生成 FIFO 期初持仓 lot
- `avg_cost` 或 `cost_basis`: 期初持仓成本基础；`cost_basis` 优先，缺失时使用 `avg_cost × quantity`
- `current_price`: 当前价，仅展示/导出辅助；不会被用来冒充 FIFO 成本

### trades.csv

可选字段文件。未上传时，交易复盘模块显示“未上传交易数据”。

```csv
trade_date,ticker,name,side,price,quantity,trade_amount,fee,reason,theme
```

- `side`: `buy` 或 `sell`
- 已执行交易必须完整记录 `trade_date/timestamp`、`ticker/symbol`、`side/action`、`price`、`quantity/shares`、`trade_amount/trade_value`
- 导出脚本会跳过任何缺少上述关键字段的已执行交易，并在 `export_summary.json` 的 `trade_record_completeness` 中标记；`scripts/check_cn_dashboard_export.py` 会将该状态视为失败
- `reason`: 交易原因，例如 `Earnings upgrade`、`Policy catalyst`、`Valuation reset`、`Stop loss`、`Position sizing`、`Theme rotation`、`Risk control`

交易胜率、盈亏比、平均盈利/亏损、单笔最大盈利/亏损和平均持仓周期按 ticker 内 FIFO 买卖配对计算。若 `positions.csv` 提供 `quantity/shares` 和 `avg_cost/cost_basis`，交易引擎会用筛选区间首笔交易前最近的持仓快照生成期初 lot；若快照日期等于首笔交易日期，则按当日买卖回推期初数量。卖出数量超过已配对买入和期初 lot 数量时，页面会保留可配对部分并显示警告，不会中断 Dashboard 加载。

## 指标计算方法

- `total_return = 期末 portfolio_nav / 期初 portfolio_nav - 1`
- `benchmark_total_return = 期末 benchmark_nav / 期初 benchmark_nav - 1`
- `excess_return = total_return - benchmark_total_return`
- `annualized_return = (期末 NAV / 期初 NAV) ^ (365 / days) - 1`
- `annualized_volatility = std(daily_return) × sqrt(252)`
- `sharpe_ratio = annualized_return / annualized_volatility`
- Sharpe 默认 `rf = 0`
- `win_rate = 日收益率为正的交易日数量 / 总交易日数量`
- `drawdown = portfolio_nav / running_max_nav - 1`
- `max_drawdown = min(drawdown)`
- `daily_excess_return = portfolio_return - benchmark_return`
- `cumulative_excess_nav` 从 `1` 开始累乘 `1 + daily_excess_return`
- `monthly_return = 月末 portfolio_nav / 月初 portfolio_nav - 1`
- `rolling 20D volatility = 过去 20 个交易日 daily_return 标准差 × sqrt(252)`
- `rolling 60D beta = cov(portfolio_return, benchmark_return) / var(benchmark_return)`
- `Herfindahl Index = sum(weight^2)`

多 benchmark 指标：

- `Excess Return = portfolio_total_return - benchmark_total_return`
- `Annualized Excess Return = portfolio_annualized_return - benchmark_annualized_return`
- `Tracking Error = std(portfolio_return - benchmark_return) × sqrt(252)`
- `Information Ratio = annualized_excess_return / tracking_error`
- `Beta = cov(portfolio_return, benchmark_return) / var(benchmark_return)`
- `Correlation = corr(portfolio_return, benchmark_return)`

不同 benchmark 日期不完全一致时，页面使用 `portfolio_nav` 的日期作为主轴；benchmark 缺失日期会前值填充，起始段没有有效 benchmark 数据时跳过该 benchmark 的 return 计算。

## 如何替换 sample data

方式一：打开页面后使用顶部上传按钮，分别上传自己的 CSV 文件。

方式二：替换 `sample/` 目录下的 CSV 文件，然后在页面中上传这些文件。默认加载仍使用浏览器内置的模拟 sample data，目的是保证双击 `index.html` 时不受 `file://` 读取限制影响。

方式三：运行 `python3 scripts/export_cn_aggressive_dashboard_data.py`，从 `results/strategy_records/CN/aggressive_tech_manufacturing/` 重新生成本地记录数据包。页面会优先使用该数据包。

## 数据隐私说明

所有 CSV 解析、指标计算和图表渲染都在浏览器本地完成。页面不会上传数据到云端，不调用后端接口，不写入数据库。

## 常见问题

### 为什么默认数据和 sample CSV 都是模拟数据？

为了避免编造真实业绩或误用真实市场数据，默认样例仅用于演示页面结构和计算流程。
sample benchmark 也是模拟数据，仅用于演示多基准比较，不代表沪深300、中证500、科创50、半导体、机器人或任何真实指数。

### 上传一个文件后，其他模块为什么仍有数据？

上传 `nav.csv` 只替换业绩数据，positions/trades 会保留当前数据源；点击 `Reset to Sample Data` 可恢复全部样例。

### trades.csv 没有价格序列，为什么不计算 post-trade return？

交易后收益需要可靠的逐 ticker 价格序列。当前静态版本不抓取行情，也不从其他文件推断价格，所以数据不足时显示“缺少价格序列，暂不计算 post-trade return”。

### CSV 中百分号怎么处理？

`8%` 会解析为 `0.08`。权重字段中的纯数字若大于 `1` 且不超过 `100`，也会按百分数处理。

## 已知限制

- 不包含真实行情抓取
- 不包含自动估值
- 不包含后端数据库
- Sharpe 默认 `rf = 0`
- contribution 依赖 `positions.csv` 的字段质量
- 默认 sample data 是模拟数据，不代表真实组合业绩
- 默认 sample benchmark 是模拟数据，不代表真实指数；真实指数需要用户自行导出并归一化
