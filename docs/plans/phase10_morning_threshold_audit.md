# Phase 10 initial audit — owner-threshold consumption

Initial audit: AUDIT_COMPLETE / IMPLEMENTATION_NOT_STARTED at the audit time.
Subsequent v3 implementation passed local validation; see
`.agent/acceptance/phase10-validation.json`. The initial findings below remain
historical context; the approved implementation plan is
`phase10_morning_threshold_consumption.md`.

The source design requires Morning to consume the previous completed EOD,
same-day captured quotes, and owner thresholds without performing maintenance.
The existing v2 consumer already reads and replays exact completed EOD and
next-session Calendar evidence, validates quote bytes/time/scope, and rejects
live consumption without prospective native provenance. Preserve those gates.

Two gaps remain:

- `morning_contract.expected_quote_symbols` accepts `morning-quote-policy.v1`.
  Its owner policy describes effective dates and additional quote symbols; it
  does not bind initial stops or trailing thresholds.
- `morning_report.validate_morning_report` checks authority and quote-timing
  declarations only. It does not prove that a report contains the corresponding
  deterministic owner-threshold review or source references.

Existing owner rules must be reused. The exact trailing-anchor policy dated
2026-09-01 and initial-stop policy dated 2026-08-28 both still match the hashes
pinned by `scripts/export_cn_research_risk.py` at this audit. Do not revise their
values, trigger semantics, validity, or authority. The pure
`strategy_records.research_risk.calculate_position_risk` already computes the
approved retention thresholds and keeps them non-executable.

The current exporter selects mutable Store, Calendar, Event and Market heads.
Calling that exporter directly from the Morning consumer would violate exact
prior-EOD consumption. The next implementation plan must define a frozen source
reader using the selected completion and retained native refs, an explicit owner
policy input/version, and a deterministic review bound to same-day quote bytes.
It must preserve lifecycle/corporate-action blockers, original policy clocks and
owner-confirmation boundaries. Phase 8 corporate reports are reconciliation
proofs with NON_EXECUTABLE threshold state, not independent authority to reset
or execute a threshold.

Next bounded work: inspect the exact stop/trailing nested contracts, specify the
smallest versioned Morning input and report extension, obtain Architect then
Critic review, and implement/test the pure consumer through existing CLI and
receipt/history paths. Include missing prior EOD, wrong quote day, missing or
changed owner policy, invalidated lifecycle/corporate action, stale threshold,
report mismatch, and no-maintenance/no-financial-write cases. Deployment and
scheduler migration remain separate later gates.
