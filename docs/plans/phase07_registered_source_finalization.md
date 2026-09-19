# Phase 7 registered-source official finalization

Status: Part A Architect amendments accepted; Critic APPROVE;
IMPLEMENTED_LOCAL_VALIDATION. Full integration remains open.

Part A verification: 106 focused performance, native Store, existing-close
adoption, plan and adapter tests passed in5.65s. Black/flake8 and source mypy pass.
Only `performance.py` and its new focused test module changed for this part.
No caller yet supplies the new mode; it becomes operational only after the
registered-source binder and batch v2 integration. Exact evidence is retained in
`.agent/acceptance/phase7-finalization-part-a-validation.json`.
Original requirements and source reachability are retained in
`daily_evidence_dag_chat_design.md` and `phase07_registered_transition_gap_audit.md`.

## End state and execution sequence

The full path must prove previous EOD -> already registered intraday financial
state -> official T close. Decision reads the first state; the existing
`build_record`/batch writer revalues the second state without applying any new
fill or changing its shares, cost basis or cash. Native Store lineage preserves
both financial transitions. Ordinary empty-event batch v1 remains intact.

Work proceeds in three concrete parts, without treating Part A as full Phase7:

1. Same-day official performance finalization in the existing performance owner.
2. Exact registered-source evidence and seven-domain declaration, with separate
   previous-EOD and writer-source pointer custody. Missing dimensions cannot be
   inferred empty. Modern registered trades, exact funding and isolated corporate
   postings require their respective owning evidence; unsupported or mixed inputs
   fail closed.
3. Versioned native batch/source/preparation/cutoff/corporate/portfolio/Store/replay
   integration, followed by native end-to-end scenarios. No deployment until the
   source route cannot publish a false empty closure for a registered change.

Parts B/C need their concrete schema/dispatch plan reviewed before implementation.
No real owner facts, records, policy, active pointer or scheduler changes occur in
these local implementation steps.

## Part A: exact same-day performance behavior

Keep `extend_performance_rows` as the calculation owner. Add an optional keyword
`official_close_source: Mapping[str, Any] | None = None`, defaulting to None.
All existing append/correction callers retain their behavior. A non-null source
requests only a same-day intraday-to-official finalization; it is incompatible
with correction mode, a supplied post-flow unit count or a nonzero new flow.

This argument is the owning native `validate_record` projection for the already
registered writer source. It is financial evidence, not a permission capability.
The function remains pure and cannot register, publish or activate anything.
The later batch v2 binder must independently read these exact source bytes,
native catalog/performance and the original pointers under existing write guards.

Before building a replacement, require:

- At least two validated performance rows, preserving a prior-day baseline.
- Source record id/date equal the last row; its `source_record` equals the
  preceding performance record, and the preceding date is strictly earlier.
- Source `official_valuation` is exactly false; its execution kind is
  `applied_effective_ledger`, and the last row is `REGISTERED_APPLIED_TRADES`.
  Do not accept a correction or an already official row as an intraday source.
- Source manual/ledger/financial-state SHAs equal the last row's corresponding
  SHAs, and its cash/equity/NAV/P&L accounting matches that row at native money
  precision. Reject absent/malformed values and a mismatched source.
- New strict record references the source id, has the same date, a distinct
  native batch-style record id, execution kind `carry_forward`, exact status
  `no_action_carry_forward_official_valuation`, `official_valuation=true`,
  `valuation_completeness_passed=true`,
  `valuation_status=OFFICIAL_STRICT_MARKET_CLOSE_COMPLETE` and
  `price_basis=strict_parquet_market_close_hash_bound`. Require exact
  `publication_class=BATCH_CATCH_UP_OFFICIAL_VALUATION` and explicitly present
  `funding=None` / `funding_correction=None`; even an offsetting new funding
  declaration cannot enter the final valuation record.
- The three supplied new artifact SHAs equal the new strict record's SHAs.
- Source and new position identities have exactly the same symbols, positive
  integral shares, average cost and cost basis. Compare finite decimal values
  without rounding away a quantity/cost change. Cash is unchanged. Market value
  and NAV can change through official valuation; the existing accounting checks
  still apply.

Replace only the final performance row, with the new official record id and
`REGISTERED_OFFICIAL_FINANCIAL_STATE`. Preserve the intraday row's unit count and
cumulative excluded external flow exactly; do not apply a funding event again.
Use the existing native close timestamp and existing return/drawdown recomputation
and row validation. Return a new sequence without mutating caller inputs. The
normal append/correction path and the stored Parquet schema stay unchanged.

## Part A acceptance

- Without the new exact source, same-day replacement still rejects.
- With a bound intraday source, one official row replaces it, date count stays
  unchanged, prior rows and inputs remain unchanged, and evidence is official
  close rather than correction. Strict close time is15:00 Shanghai.
- A source with an already applied contribution/redemption retains its exact
  units and cumulative external flow. Verify independent expected unit NAV and
  daily return; no second flow or artificial return from funding.
- Changed cash/shares/cost/symbols, duplicates, malformed/nonfinite values, bad
  refs/source/date, existing official/correction source, a skipped predecessor,
  ordinary record id, incomplete close, attempted new flow and mixed correction
  mode reject without mutation or any I/O. Wrong publication class and a new
  funding/funding-correction payload reject even if quantities, cash and SHAs match.
- Native deterministic Parquet write/read validates the resulting series.
- Existing performance append, owner correction, unitization and native no-trade
  close/adoption tests remain passing. Use focused checks, not another unrelated
  49-minute full suite before the remaining integration code exists.

Part A's passing tests prove only the performance transformation. They do not
prove source registration, seven-domain closure, Store CAS, end-to-end DAG,
real owner events, installed operation, OOS or unattended completion.
