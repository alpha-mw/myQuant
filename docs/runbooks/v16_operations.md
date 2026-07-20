# myQuant v16 Research-Candidate Operations

v16 is an isolated research-only candidate lane. v15 remains the production
and default authority until a separately reviewed activation changes that
state. These operations are offline by default and never call a broker or
create an order.

## Immutable boundaries

- Formal branches are exactly `quant`, `fundamental`, `macro`, `llm`, each at
  weight `0.25`.
- Retrieval notes annotate only `quant`, `fundamental`, or `macro`; they cannot
  carry scoring or weighting fields.
- All formal v16 outputs live below `results/v16/`.
- `results/v15/` is read-only. Do not copy or relabel a v15 payload as v16.
- `dashboard_contract.v16` is versioned alongside v16; Dashboard v3 and its
  existing app surfaces remain unchanged.
- RiskAdvisor is advisory-only. Deterministic eligibility, IC allocation,
  readiness, and human authorization remain authoritative.

## Required evidence before report generation

1. Exact v16 protocol envelope and sealed four-branch evidence.
2. Factor readiness using schema `factor-governance-readiness.v4`, protocol
   `v4`, at least 5 healthy factors, at least 3 families, a valid activation
   receipt (not a summary boolean), factor-set/runtime-contract hash readback,
   factor weight at most `0.20`, family weight at most `0.35`, and no blockers.
3. Calibration readiness using schema
   `calibration-readiness.v16.four-evidence` with, for each branch, at least 300
   samples and 8 non-overlapping cohorts.
4. Calibration metrics that independently pass all thresholds: Brier and
   log-loss bootstrap upper bounds below baseline, ECE at most `0.05`, interval
   coverage from `0.85` through `0.95`, alpha MAE below zero-alpha MAE,
   top-bucket edge lower bound above zero, and lambda fold range at most `0.20`.
5. One IC action for every sealed menu symbol, complete existing weights,
   positive target count at most 12, and target weights plus cash at 1.
6. A hash-bound handoff for the exact execution plan, Stage 2 response, and
   capital map.
7. A sealed `codex-review-human-authorization.v1` receipt bound to the run,
   Stage 2 response, capital map, human identity, authorization time, and
   expiry. An unsigned dictionary is never authorization.

Missing evidence is a blocker. Do not set a readiness boolean by hand to bypass
the underlying fields.

The v4.3 prior-diagnostic fifth-factor nomination and the v4.4 five-candidate
prospective preregistration contract do not satisfy this readiness gate. All
five v4.4 candidates start at weight zero; historical nomination statistics are
not inherited; formal measurement, family BH, maturity, Gates 1-8, dedup,
canonical replay, fresh health, and same-day activation receipts remain absent.
Even a successful future v4.4 private publication must remain
`PROSPECTIVE_PREREGISTRATION_ONLY` with every production/new-risk authority
false until those independent post-publication gates close.

A classification-only v4.1 A_quant catalog cannot satisfy Factor readiness
above. Its 267 schema-readable candidate definitions and 18 classification
primitives are validate-only evidence, not executable signals, screening,
statistics, qualification, an activation receipt, or new-risk authority. Its
protected-control readback covers build and precommit identity only and does
not lock concurrent external maintenance.

The separate v4.1 pinned-operator proof is also insufficient on its own. It may
set only `operator_runtime_equivalence_verified=true` after the exact 37
definitions and seven adversarial primitive probes match under the hash-bound
myQuant PIT envelope. It must keep `signal_computability_proven=false` and all
screening, BH, qualification, admission, registry, apply, production, and
new-risk fields false. Do not reinterpret this proof as evidence that all raw
fundamental inputs exist or implement the pinned A_quant semantics.

The follow-on exact-37 computability bundle may set
`signal_computability_proven=true` only for claim scope
`pinned_aquant_git_data_exact37_source_semantic_computability.v1`. The accepted
reference bundle is
`factor_v4_1_signal_computability_20260719T003554Z`, with proof byte SHA-256
`1bdb61112f70f4cf31b9435d8f027ba384e4575ee81168782d64d052749e73ab`
and independent readback byte SHA-256
`3f379493ce5804db4d4708319d389343e650afee83abe9453d0a936a3f375c3b`.
It keeps the known A_quant/myQuant calendar gaps as exact non-passing facts and
sets completeness, same-snapshot, screening, BH, maturity, qualification,
admission, registry, apply, production, portfolio, and new-risk authority
false. It cannot satisfy Factor readiness or be relabeled as current-snapshot
screening evidence.

## CLI state machine

`market analyze` and `market run` keep v15 as the default. An explicit
`--decision-protocol v16` runs Eligibility, formal full-market Quant, and the
deterministic Funnel, then stops at `S1_PREPARED`. It does not run the v15
Bayesian or portfolio path. The formal handoff commands are:

```bash
quant-investor market codex-review-export ...
quant-investor market codex-review-receive ...
quant-investor market codex-review-validate ...
quant-investor market codex-review-resume ...
quant-investor market codex-review-status ...
```

Each mutation requires the expected state SHA. `--no-agent-layer` emits only a
diagnostic report and cannot create a shortlist, target weights, or new risk.

## Build order

1. Seal the deterministic funnel and Codex Stage 1 union.
2. Seal one `quant/fundamental/macro/llm` evidence record per symbol.
3. Reconstruct the disconnected posterior runtime only from bound canonical
   model/training artifacts, then generate the full-union posterior menu with
   exact ordered Q/F/M/LLM evidence. The evidence-v2 producer requires the
   canonical eight-component cost model and fails closed if any cost artifact
   is missing. Existing non-evidence-v2 report surfaces continue to preserve
   `posterior_edge_after_costs=null` when cost evidence is unavailable.
4. Record RiskAdvisor `warnings` and `recommendations` without changing formal
   scores or allocation.
5. Validate Codex IC `BUY/HOLD/AVOID/SELL` actions, existing/target weights,
   `selected_for_portfolio`, rationales, and `cash_ratio`.
6. Canonicalize the execution plan and verify the handoff binds its SHA.
7. Build and write `results/v16/<run-id>/v16_run_readiness.json`.
8. Use its immutable reference to build and write
   `results/v16/<run-id>/v16_candidate_decision_report.json`.
9. Export a sanitized Dashboard v16 snapshot only after the Dashboard
   activation gate is independently verified.

Both JSON writers use canonical serialization, atomic replacement, owner-only
mode `0600`, and SHA-256 readback.

Evidence producers and recomputers must enter through the secure intake
factories and retain the returned bound artifact through parsing. Private and
trust-material plus governed-data roots use the built-in Darwin descriptor ACL
verifier; ancestors with any allow ACL are rejected. Do not import a private
low-level reader, inject an ACL assertion, decode an unbound path read, or
substitute a discovered file. ACL verification unsupported or inconclusive is
a blocker.

## Fail-closed outcomes

The readiness artifact must set `new_risk_authorized=false` and
`readiness_status=no_new_risk` for any unresolved blocker. Common exact
blockers include:

- `factor_count_below_minimum:actual=<n>:required=5`
- `factor_family_count_below_minimum:actual=<n>:required=3`
- `factor_readiness_schema_not_v4`
- `factor_activation_receipt_missing`
- `factor_weight_limit_or_normalization_invalid`
- `calibration_threshold_not_met`
- `calibration_gate_failed:<gate>`
- `handoff_missing`
- `handoff_not_ready`
- `new_risk_human_authorization_missing_or_invalid`
- `activation_codex_gate_not_ready`
- `activation_dashboard_gate_not_ready`
- `global_attempt_registry_authority_not_integrated`
- `evidence_v2_disconnected_from_authorizing_consumers`

`activation_candidate=false` must include non-empty
`activation_blockers`. Passing candidate activation does not itself move the
v15 production/default pointer.

The current readiness builder also preserves
`global_attempt_registry_authority_not_integrated` and
`evidence_v2_disconnected_from_authorizing_consumers` unconditionally. Legacy
calibration summaries or caller-supplied Codex/Dashboard booleans therefore
cannot produce `activation_candidate=true` or `new_risk_authorized=true` while
the reviewed evidence-v2 authority migration is absent.

The optional evidence-v2 provisional attempt journal is not that migration. It
has no default path and must not be initialized by an audit run. Its local hash
chain coordinates one process-visible A/B/C attempt but cannot prove deletion
or rollback by the same OS user. Do not use its state as Calibration, Factor,
Codex, Dashboard, human-authorization, or production-pointer evidence.

## Calendar, clock, and schedule-v3 preflight

The disconnected source compiler accepts only the fixed private inventory at
`private/v16/evidence_v2/calendar_sources`: 22 consumed files and two explicit
exclusions. It does not fetch or refresh sources. The combined acceptance
opens each consumed physical file once, validates all 28 semantic bindings,
recomputes the 242-session 2026 calendar and listed-equity auction clock, and
performs the nine-source local recheck. It writes nothing:

```bash
/Users/maxwell/mySpace/myQuant/.venv/bin/python -c \
  'from quant_investor.v16.evidence_v2.calendar_recheck import validate_private_calendar_recheck_acceptance; p = validate_private_calendar_recheck_acceptance("/Users/maxwell/mySpace/myQuant/private/v16/evidence_v2/calendar_sources"); print(p["semantic_sha256"], p["blockers"])'
```

Success is local semantic correspondence only. It must still report
`transport_freshness_status=not_independently_attested`, all three false
authority flags, and the exact capture-time, transport-freshness, and
disconnected-consumer blockers. Do not use it as a source refresh, activation
receipt, human authorization, Dashboard gate, or production/default switch.

Schedule v3 rejects any `s0` before `2026-07-06`, any weekend or closure,
skipped target session, overlapping slot, non-exact UTC clock boundary, or
target window beyond the 2026 calendar. Epoch A binds no models; B/C bind the
ordered `quant/fundamental/macro/llm` frozen bundles and an exact
schedule-specific calibration universe. The full evidence bundle, not a bare
schedule JSON, is required for RFC3161 anchor and lineage validation.

## Report and Dashboard checks

The report must expose:

- exact four-branch contributions at `0.25` each;
- retrieval audit annotations without score/confidence/likelihood/weight;
- posterior win rate, expected alpha, nullable edge after costs, and both 90%
  intervals;
- advisory-only RiskAdvisor warnings and recommendations;
- every IC action, existing/target weight, selection flag, rationales, selected
  symbols (maximum 12), and cash ratio;
- handoff, eligibility, execution, v16 readiness, and activation state.

Tracked schemas and tests must use synthetic symbols only and contain no real
account, broker, holding, or trade data.

## Local validation

Run the focused offline checks from the repository root:

```bash
/Users/maxwell/mySpace/myQuant/.venv/bin/python -m pytest \
  tests/unit/test_factor_governance_operator_runtime_equivalence_v4_1.py \
  tests/unit/test_build_factor_v4_1_operator_runtime_equivalence.py \
  tests/unit/test_factor_governance_signal_computability_v4_1.py \
  tests/unit/test_build_factor_v4_1_signal_computability.py \
  tests/unit/test_v16_evidence_v2_contracts.py \
  tests/unit/test_v16_evidence_v2_schedule_target.py \
  tests/unit/test_v16_evidence_v2_metrics_runtime_timestamp.py \
  tests/unit/test_v16_evidence_v2_posterior.py \
  tests/unit/test_v16_evidence_v2_calendar.py \
  tests/unit/test_v16_evidence_v2_session_clock.py \
  tests/unit/test_v16_evidence_v2_calendar_recheck.py \
  tests/unit/test_v16_evidence_v2_schedule_v3.py \
  tests/unit/test_v16_run_readiness.py \
  tests/unit/test_v16_candidate_decision_report.py -q

node portfolio_dashboard/tests/dashboard_contract_v16.test.js

/Users/maxwell/mySpace/myQuant/.venv/bin/python -m py_compile \
  quant_investor/monitoring/v16_run_readiness.py \
  quant_investor/reporting/v16_candidate_decision.py

/Users/maxwell/mySpace/myQuant/.venv/bin/python -m json.tool \
  portfolio_dashboard/schema/dashboard_contract.v16.schema.json >/dev/null
```

The temporary v16 worktree may not have `pytest` installed in its own venv; the
main repository venv above is the provisioned local test runtime. No live data,
LLM, or execution surface is invoked by these checks.
