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

## Fail-closed selector

The private page accepts v2 only when the Dashboard selector is `UPDATED`, its
attempt ID and content SHA match v2, the nested v1 equals the separately loaded
v1, and the view has not expired at Shanghai midnight.

```text
refresh starts     -> selector REFRESHING
all checks pass    -> selector UPDATED (published last)
any failure        -> selector BLOCKED
process interruption -> selector remains REFRESHING
```

This prevents an earlier same-day successful bundle from continuing to look
current after a later refresh fails. The selector and v2 are Dashboard-only;
they have no Store, market, broker, order, Paper, or trade authority.

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
