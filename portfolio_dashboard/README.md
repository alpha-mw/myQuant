# myQuant Dashboard 2.0

`portfolio_dashboard/` 是 `CN/aggressive_tech_manufacturing` 的本地静态
分析面。它只使用 vanilla JavaScript、SVG 和一次性 full-snapshot replacement；
不需要 Web workspace、认证、SSE、API、数据库、Node build 或网络。

## 数据边界

- 正式账户与执行事实仍来自 `manual_execution_manifest.json` 的 contained
  `next_ledger_path`，允许 `ledger_after_manual_switch.csv` 或 `.parquet`；禁止回退到
  `ledger.csv`。有效 baseline 必须由
  manifest 的允许状态、显式记录时间和 `next_ledger_path` 选择，ledger 路径必须
  留在 record root 内，并做读取前后稳定 SHA readback。manifest 若声明 ledger
  SHA-256 则必须匹配；历史 manifest 未声明时保留 computed SHA/provenance，并以
  `manual_ledger_sha_not_declared` 将 Dashboard 标为 partial，而不是停用现有事实源。
  最新 baseline 按 manifest 时间排序，不按目录名；较晚目录中的 invalid/stale
  no-action 记录不能覆盖时间更晚的有效 manual baseline。
- `scripts/export_cn_aggressive_dashboard_data.py` 生成
  `portfolio_dashboard/private/dashboard_snapshot.v2.json` 和 `.js`。
  `private/` 被本目录 `.gitignore` 忽略，是真实账户数据的唯一页面加载目录。
  导出器对 snapshot JSON/JS、summary 及四张 generated CSV 全部使用原子替换并
  强制 `0600`；semantic checker 会拒绝权限放宽或 JSON/JS payload 不一致。
- tracked `js/generated_records.js` 只是无真实数据的兼容 loader；tracked
  `sample/dashboard_snapshot.v2.json` 是空的合成示例契约，不含 ticker、资金、
  私有路径或真实持仓。
- 用户上传的 CSV 只存在于当前页面会话的内存中。刷新或关闭页面即清除；
  不写 `localStorage`，原始数据也不会写入 URL。URL hash 只保存日期、benchmark、
  workspace 和 chart lens 等非原始筛选状态。

双击 `index.html` 即可使用。没有私有快照时，页面明确回退到模拟 sample；sample
及 sample benchmark 只演示 UI 和计算方法，不代表真实业绩或真实指数。

## Contract v2

正式快照遵循 tracked
`schema/dashboard_contract.v2.schema.json`，顶层包含：

- `schema_version/schema_sha256/protocol_hash/run_id/generated_at`；
- `fresh/stale/partial/blocked` 状态和 `blockers`；
- `as_of_matrix`：策略记录、分析交易日、quote、benchmark value date、Theme、
  factor registry SHA；
- `trading_calendar`：只读 strict Parquet `trade_date` 生成的开放日掩码和 SHA；
- `nav_return_provenance`：NAV 的 gross/net 与费用嵌入口径；
- ledger、manual manifest、Theme、factor protocol 的 sanitized path summary 与 SHA；
- `nav/positions/trades/themes/factors` 五张表、`theme_protocol/factor_protocol`
  readback summary 和每日 `reconciliation`。

数值规则：

- 收益一律为 decimal。`today_change_pct=-0.46` 在导出边界转换一次为
  `daily_return=-0.0046`。
- `nav_weight` 与 `equity_sleeve_weight` 分列；兼容字段 `weight` 等于
  `nav_weight`。
- `contribution = nav_weight × daily_return`；每个有效 NAV return 日必须有持仓贡献
  或明确 cash/fee residual。覆盖率和 `unexplained_residual` 由 checker 从原始表重算，
  贡献条数必须等于当日全部持仓、每行 lineage 完整且 sleeve 权重合计为 1；cash/fee
  residual 不能制造 coverage。只有 100% 日期覆盖且全部在 1bp 内才可标记
  `reconciled`。
  归因的 valid day 同样由 strict Parquet 开放日掩码决定；position contribution
  必须带明确 quote/effective date lineage（可来自 holdings review、ledger 或明确的
  market snapshot analysis date）。mask 外日期不计入 valid/covered，并在
  `reconciliation.diagnostics/blockers` 留痕。
- 缺失费用、价格或日期为 JSON `null` / CSV 空值，不得伪造为零。任一费用未知时，
  净实现盈亏、净胜率、净盈亏比、费用后收益和费用拖累保持 unavailable；毛盈亏可
  单独展示。NAV gross/net 或费用嵌入口径不明时禁止二次扣费。
- TWR 单位净值保留 manifest-backed 资金流处理。超额曲线使用相对财富比
  `(1 + portfolio return) / (1 + benchmark return)`；月度收益使用上月末锚点。
- NAV 与 benchmark 先压缩到 strict Parquet 开放日，再从压缩后的点重算收益；
  weekend/non-open snapshot 不参与 CAGR、超额、月度、rolling 或 drawdown。
  benchmark comparison、图表与筛选区间月度锚使用同一压缩序列；future-as-of blocker
  存在时，正式 NAV/benchmark 图截断至 `analysis_trading_date`。
- Contract v2 中所有动态 `*_nav` 字段统一导出为 JSON number/null，禁止数字字符串。
- `partial_fill/partially_filled` 只接受实际 fill quantity/price/value 字段；无 status
  的 order 和仅有 order 数量的部分成交都不算成交。
- 年化指标仅在有效日收益不少于 60 且 strict Parquet 开放日覆盖率不少于 95% 时
  生成；20D/60D 窗口也必须达到各自开放日覆盖率 95%。不存在 weekday/count
  proxy；交易日掩码缺失时显示 `formal_trading_calendar_missing`。

## 六个工作区

1. 今日总览：告警、六个核心指标、NAV/暴露主图、持仓纪律快照；
2. 持仓与纪律：NAV/sleeve 权重、动作、止损止盈、quote age、风险和暴露；
3. 交易与决策：recommendation、decision、order、fill、费用来源和 ledger delta；
4. Theme：protocol/formal/rollback summary、attention × industrial validation、
   5/20/60/120D trajectory、lane、lifecycle、供应链与 thesis review；
5. Factor：生产/shadow/blocked、slot、权重、health、challenger 和 transition；
6. 归因与数据审计：as-of、来源 SHA、每日勾稽、benchmark provenance 和月度表现。

workspace 按需渲染，单个 workspace 同时最多六张主图。390px 视图不产生页面级
横向滚动，首屏纪律表只保留关键列；桌面仍是主分析面。

## 导出与校验

在仓库根目录离线执行：

```bash
PYTHONPATH=. python3 scripts/export_cn_aggressive_dashboard_data.py \
  --benchmark-source local \
  --benchmark-file portfolio_dashboard/inputs/cn_index_benchmark.csv \
  --trading-calendar-root data/parquet/cn/bars

PYTHONPATH=. python3 scripts/check_cn_dashboard_export.py
```

若必须使用正式投委会 benchmark：

```bash
PYTHONPATH=. python3 scripts/check_cn_dashboard_export.py \
  --require-production-benchmark
```

checker 同时验证结构、schema/protocol hash、formal trading-day mask、decimal return、
贡献公式、完整归因覆盖、费用/NAV provenance、路径隐私、TWR funding lineage、
canonical ledger、benchmark provenance 和 CSV/readback 行数。任何 v2 summary 与
snapshot 不一致都会 fail closed；canonical Factor full-chain producer 不可用时，
任何声称 `applied` 的 artifact 都会显示为 blocked。

本地前端契约测试：

```bash
node portfolio_dashboard/tests/dashboard_contract_v2.test.js
PYTHONPATH=. pytest -q -p no:cacheprovider \
  tests/unit/test_dashboard_contract_v2.py \
  tests/unit/test_dashboard_export_check.py
```
