# CN aggressive Dashboard v2 daily continuity

The private Dashboard publishes two compatible views:

- `cn_aggressive_dashboard.v1` remains the immutable Store/performance view.
- `cn_aggressive_dashboard.v2` nests the exact v1 document and adds a
  fail-closed daily continuity and strict-close valuation view.

V2 never creates a Strategy Record, changes the active ledger, or appends a
canonical performance point. Its current view is derived from three exact
authorities:

1. the registered active closure supplies shares, cash, average cost, and cost
   basis;
2. the deterministic daily `NO_ACTION` receipt confirms that those facts
   remain effective through the current Shanghai date; and
3. the healthy strict CN snapshot supplies the latest complete close for every
   active holding.

For example, the current state is represented as:

```text
financial-state anchor       20260814_1824 / 2026-08-14
continuity receipt           automation-20260818-daily-review-v1
holdings valid through       2026-08-18
latest verified local close  2026-08-17
```

The internal UI therefore reports:

```text
持仓估值 UPDATED · 最新可用严格收盘 2026-08-17
组合绝对业绩 VIEW-UPDATED · 截至 2026-08-17
财务状态锚点 2026-08-14 · NO_ACTION 连续有效至 2026-08-18
基准相对业绩 · 截至 2026-08-14
```

The portfolio mark updates current prices, market values, NAV, cash/gross
weights, unrealized P&L, absolute cumulative return, continuity interval return,
and portfolio drawdown. Benchmark-relative metrics keep their own verified
date; they are never forward-filled to the later portfolio mark.

## Last-good selector and attempt receipts

The private page accepts v2 only when the Dashboard selector is `UPDATED`, its
attempt ID and content SHA match v2, the nested v1 equals the separately loaded
v1, and the view has not expired at Shanghai midnight. Refresh work is staged
and checked before the selector changes. A failed attempt preserves the last
good selector and bundle bytes and writes an immutable receipt under
`portfolio_dashboard/private/generated/attempts/`.

```text
refresh starts       -> last-good selector unchanged
all checks pass      -> selector UPDATED (published last)
any failure          -> last-good selector unchanged + BLOCKED attempt receipt
process interruption -> last-good selector unchanged
```

Freshness expiry still prevents an old successful bundle from presenting as
current after Shanghai midnight. The selector and v2 are Dashboard-only; they
have no Store, market, broker, order, Paper, or trade authority.

## Benchmark tail separation

The canonical portfolio series may extend beyond one or more benchmark inputs.
Only a trailing benchmark gap is allowed. Portfolio NAV and absolute return
remain exact through the canonical performance date, while missing benchmark
NAV, relative return and excess fields are `null`, coverage is `unavailable`,
and each benchmark reports its last exact `end_date` plus exact
`missing_dates`. A middle-of-history benchmark gap remains blocking. V2 may
therefore report current holdings/absolute performance separately from
`benchmark_relative=AS_OF_PRIOR_DATE`.

## Daily commands

The existing no-argument exporter invocation now publishes v1 JSON/JS and v2
JSON/JS transactionally, then advances the selector:

```bash
./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py \
  --project-root /Users/maxwell/mySpace/myQuant \
  --record-root \
    /Users/maxwell/mySpace/myQuant/results/strategy_records/CN/aggressive_tech_manufacturing

./.venv/bin/python scripts/check_cn_dashboard_export.py \
  --project-root /Users/maxwell/mySpace/myQuant
```

The exporter writes only the six fixed filenames inside
`portfolio_dashboard/private/generated/`.  It rejects custom paths, public
directories, System/Store/data paths, and any symlinked output-root or output
file before it writes even the initial `REFRESHING` selector.

The public page remains v1-only. It never loads, copies, or serves the private
v2 research mark or selector.

## Current phase boundary

Phase 1 is zero-network and uses only the latest complete strict CN close.
Credential-free Sina intraday evidence remains a separate future phase. It must
use an immutable private sidecar and may never enter Strategy Record Store,
the effective ledger, canonical performance, Paper fills, or execution state.
