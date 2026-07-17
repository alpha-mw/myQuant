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
3. Generate calibrated posterior menu fields and include exact ordered
   Q/F/M/LLM formal evidence for Stage 2. Preserve
   `posterior_edge_after_costs=null` when cost evidence is missing.
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

`activation_candidate=false` must include non-empty
`activation_blockers`. Passing candidate activation does not itself move the
v15 production/default pointer.

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
