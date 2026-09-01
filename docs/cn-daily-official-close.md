# CN Daily Official Close

`close-through-latest` is the governed offline close step for
`aggressive_tech_manufacturing`. It closes the production gap between a complete
CN Market snapshot and the Strategy Record Store-v3 performance head.

## Production invariant

The public Dashboard may advance only when:

```text
latest official performance date
  == latest required close date
```

`latest required close date` is derived from the continuous exact-date prefix
shared by the registered exchange Calendar, strict CN Market history and the
immutable three-index benchmark generation. It is intentionally independent of
event readiness: a missing event closure blocks official close and public
publication rather than making stale output appear eligible.

## Authority split

- `cn-daily-data-update` owns post-close Market, Calendar and benchmark capture.
- `close-through-latest` is offline and consumes exact pointer/manifest/receipt
  SHAs. It never calls a provider, broker, order or execution API.
- The pointer-selected event generation explicitly closes executions, orders,
  fills, funding, cost-basis changes, corporate actions and manual changes for
  every date. Missing is not empty.
- `cn-dashboard` remains a read-only publisher while the new close owner proves
  two distinct real trading days. It preserves public last-good bytes when the
  official close is not caught up.

The standing owner policy is
`operations/policies/cn-daily-official-close-policy.v1.json`. It authorizes only
the daily event closure, official valuation/performance/catalog/Store CAS,
immutable benchmark input and redacted public publication. It grants no actual
holdings, cash-without-event, broker, order or trade authority.

## Batch transaction

The command computes all missing open dates through the required close date. A
missing middle date causes zero Store mutation for the whole batch.

Before mutation it writes one frozen plan containing all preimage SHAs, output
paths, IDs and effective timestamps. New catch-up records use the versioned ID
`YYYYMMDD_HHMMSS-bNN`; old minute IDs and the fixed 2026-08-21 late-publication
contract remain unchanged.

After every record, performance and catalog candidate is validated, all Store,
Market, benchmark, Calendar, event and policy preimages are reread. One
`publish_catalog` call performs the only Store pointer CAS. Pre-CAS record and
performance artifacts are immutable recoverable orphans. The post-CAS
completion receipt records the actual observed CAS time; the frozen plan's
`effective_at` is never described as the actual commit time.

Exact replay is a no-op. A late event at or before the official head returns
`OFFICIAL_CLOSE_RESTATEMENT_REQUIRED`; ordinary catch-up never rewrites history.

## Commands

Capture/publish benchmark inputs inside the registered `PROJECT_ENV` boundary:

```bash
./.venv/bin/python scripts/operations/run_cn_benchmark_close.py \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --start-date YYYY-MM-DD --end-date YYYY-MM-DD \
  --generation-id <id> --expected-pointer-sha256 <sha> --execute
```

Plan the Store close without Store mutation:

```bash
./.venv/bin/python scripts/manage_cn_strategy_records.py close-through-latest \
  --record-root results/strategy_records/CN/aggressive_tech_manufacturing \
  --project-root /Users/maxwell/mySpace/myQuant \
  --expected-pointer-sha <sha> \
  --expected-market-pointer-sha <sha> \
  --expected-benchmark-pointer-sha <sha> \
  --expected-event-pointer-sha <sha> \
  --calendar-receipt <path> --calendar-receipt-sha <sha> \
  --policy-path operations/policies/cn-daily-official-close-policy.v1.json \
  --policy-sha <sha>
```

Add `--execute` only after the dry-run plan and candidate inputs are approved.

## Deployment

The 20:20 owner and 21:00 fallback must use the same clean installed release and
the same Store operation lock. Deployment is fail-closed: outside their run
windows pause both jobs, update and exact-readback both, then resume both. Any
partial failure leaves both paused. The 09:45 MORNING_STRATEGY task is not part
of this writer and remains unchanged.
