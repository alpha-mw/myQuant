# Phase13–14 remaining acceptance audit

Status: READ_ONLY_AUDIT; no evaluator, factor policy or runtime gate changed.
The full Phase0–15 goal remains incomplete.

Current follow-up: the reviewed Phase13 research boundary extension is now locally
implemented. Its54 fixed reasons, safe native missing/SHA confirmation, wrapper/
cutoff/schema/day/ref checks and pre-probe rethrow are recorded in
`phase13_research_failure_mapping.md` and
`.agent/acceptance/phase13-research-diagnostics-validation.json`. Unknown projector,
security and race errors deliberately remain generic; this does not certify every
producer boundary or all of Phase13. The original audit below is retained as the
before-state. Phase14 default evaluator admission is now implemented after review:
`phase14_evaluator_ledger_design.md` and
`.agent/acceptance/phase14-evaluator-validation.json`.188 focused checks pass.
Its recorded-release path also exposed and repaired the current bridge's missing
5-module closure; the old archived release remains rejected without substitution.
Current-source installed/full-DAG and real prospective acceptance remain Phase15.

## Phase13

The source design requires the13 named failure categories plus retryable,
owner_action_required and recommended_next_node on every blocker.
`operations.daily_contract.FAILURES` includes all13 and `failure()` emits all
three fields with a validated node ID. This part exists.

The remaining producer-to-category audit is material: `daily_runner._node` only
preserves specific structured diagnosis for DependencyInputError, whose current
registry contains core-source and execution-context reasons. The outer run loop
maps other exceptions to VALIDATION_FAILED. Therefore a new exact cutoff/source
SHA, date or source-shape exception can still lose its useful category/retry
classification at the coordinator boundary. Do not mark this phase complete from
the presence of category constants alone. Audit owning producer adapters and add
only explicit code-owned mappings, never classification by loose exception-text
matching or permission assumptions. Existing unknown-writer safeguards remain.

## Phase14

Daily ledger generation/replay already records source and coordinator custody,
and Morning's selector uses that ledger. The missing requirement is evaluator
consumption: the source design explicitly reserves true OOS claims for
prospective=true evidence, excluding late/backfilled work.

`factors.production_outcomes.classify_seal` currently computes
prospective_eligible solely from proven seal_time_upper_bound <= signal close.
It validates registered_at grammar but does not require registration by that
deadline. A late registration can therefore keep that component-level flag true.
The diagnostic summary calls the count original_close_prospective_count.
There is no daily-ledger input/replay reference in production_outcomes or the
pure forward_evaluator path. Existing outcome evaluation is explicitly
NON_AUTHORIZING, which must be preserved; it is not evidence that the new whole
DAG prospective/OOS requirement is satisfied.

Next: define a versioned evaluator evidence projection using the owning daily
ledger reader, with exact refs and native replay before prospective admission.
Keep component seal diagnostics distinct from full-DAG prospective eligibility;
do not retroactively rewrite old immutable outcome/classification records or
silently call them true OOS. Architect then Critic review is required before
changing this evaluation contract. Acceptance must include late observation
registration, later Decision/Store completion, synthetic/history/recomputed data,
missing ledger, wrong ledger SHA and a genuinely prospective native receipt.
