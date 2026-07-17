# v16 Four-Branch Candidate Decision

Status: research-only candidate. The v15 production/default pointers remain
authoritative. This decision does not switch a runtime, registry, dashboard,
broker, or order path.

## Goal and scope

The v16 candidate lane evaluates a sealed candidate set with exactly four
formal branches and records enough evidence to audit the posterior, Codex IC
decision, handoff, and fail-closed readiness result. The versioned additions
are isolated below `results/v16/` and use `dashboard_contract.v16`.

The formal branch order is exactly:

1. `quant`
2. `fundamental`
3. `macro`
4. `llm`

Each formal contribution has weight `0.25`. Missing evidence is a blocker; it
is never replaced by a neutral score. `retrieval_evidence` is a separate audit
annotation for `quant`, `fundamental`, or `macro`. Its schema has no score,
confidence, likelihood, contribution, or weight field, so it cannot overwrite
formal evidence or become a fifth branch.

## Candidate and decision contract

The deterministic funnel may provide at most 500 symbols. Codex Stage 1 may
add at most 100 distinct symbols, and the sealed union may contain at most 600.
The posterior menu is bounded at 50. Negative net edge remains visible; missing
cost evidence produces `posterior_edge_after_costs=null`, not a cost-free
substitute.

Every Stage 2 menu entry carries exact ordered Q/F/M/LLM formal raw score,
confidence, calibrated probability, and evidence ids. Retrieval annotations
remain separate and cannot overwrite these fields.

Every report exposes these posterior fields:

- `posterior_win_rate`
- `posterior_expected_alpha`
- `posterior_edge_after_costs`
- `win_rate_interval_90`
- `expected_alpha_interval_90`

RiskAdvisor output contains only advisory warnings and recommendations. It
cannot change eligibility, action, target weight, branch evidence, posterior,
or readiness.

Codex IC must emit exactly one `BUY`, `HOLD`, `AVOID`, or explicit internal
`SELL` action for every sealed menu symbol. Every action records
`selected_for_portfolio`, `existing_weight`, `target_weight`, `rationale`, and
an optional `risk_acceptance_rationale`. `HOLD` preserves the existing weight;
`AVOID` and `SELL` have zero target weight. At most 12 symbols may have positive
target weight, and target weights plus `cash_ratio` must equal 1 within
`1e-6`. The report validator does not normalize or silently cap an invalid IC
allocation.

## Readiness and activation

Eligibility, execution, readiness, and activation are separate states.
Eligibility says whether the evidence may enter candidate construction.
Execution describes the reviewed plan and is always free of broker side
effects. Readiness aggregates deterministic gates. `activation_candidate`
only says the v16 candidate passed Factor, Calibration, Codex, and Dashboard
activation gates; it does not replace the current v15 production pointer.

Factor readiness is recomputed from a
`factor-governance-readiness.v4`/protocol `v4` payload. It requires:

- at least 5 healthy production factors and at least 3 families;
- a present full v4 activation receipt whose seal and as-of, protocol,
  registry, factor-set, and runtime-contract hashes all validate;
- a recomputed healthy-factor domain matching the factor-set and runtime hash;
- normalized per-factor absolute weight no greater than `0.20`;
- normalized per-family absolute weight no greater than `0.35`;
- no governance blockers.

v2 and v3 Factor payloads cannot be relabelled and passed as v4.

Calibration readiness is recomputed from
`calibration-readiness.v16.four-evidence`; a single trusted boolean is not
sufficient. Every branch requires at least 300 samples and 8 non-overlapping
cohorts. The aggregate validation requires both Brier and log-loss bootstrap
upper bounds to beat their baselines, ECE at most `0.05`, 90% interval coverage
within `[0.85, 0.95]`, alpha MAE below zero-alpha MAE, positive top-bucket edge
lower bound, and lambda fold range no greater than `0.20`.

Human authorization is the strict sealed
`codex-review-human-authorization.v1` receipt. It must be `AUTHORIZED`, identify
a human, be live at readiness generation time, and match the handoff's Stage 2
response and capital-map hashes. Free-form approval flags are rejected.

New risk remains unauthorized when any required branch, Factor, Calibration,
candidate, eligibility, execution-plan, handoff, activation, or hash-bound
human-authorization gate fails. In particular, Factor count below 5, missing
handoff, failed calibration, or missing human authorization always yields
`readiness_status=no_new_risk`.

## Version and artifact boundary

The only writable formal namespace for this lane is `results/v16/`. Canonical
files are `v16_run_readiness.json` and
`v16_candidate_decision_report.json`, written atomically with mode `0600` and
verified by canonical JSON SHA-256 readback. A path under `results/v15/` is
rejected. A payload carrying a v15 schema, architecture, branch, report, or
readiness envelope is rejected even if one outer field is changed to say v16.

v15 artifacts remain read-only historical/production evidence. They are not
copied, wrapped, or promoted into v16.

## Dashboard and data safety

`dashboard_contract.v16` references only v16 readiness and candidate-report
artifacts under `results/v16/`. It surfaces the four equal contributions,
retrieval audit annotations, posterior values and intervals, advisory
RiskAdvisor output, exact IC allocation, handoff, eligibility, execution, and
activation/readiness state.

Tracked schemas and tests contain synthetic identifiers only. Real account,
holding, trade, or broker data remains restricted to ignored owner-only private
outputs and must never enter a tracked sample or test fixture.

## Non-goals

- No global v15 default or pointer switch.
- No mutation of the existing v15 report, readiness, Factor, or Dashboard v3
  contracts.
- No live provider, LLM, broker, order, or execution call.
- No automatic Factor registry activation or human-authorization synthesis.
- No fallback that invents costs, scores, calibration evidence, or handoff
  state.

## Acceptance, risks, and stop condition

Acceptance requires exact version rejection, exact four-branch/equal-weight
validation, explicit posterior/IC semantics, all fail-closed readiness tests,
Dashboard schema checks, and owner-only atomic artifact round trips.

The main residual risk is integration drift between the candidate pipeline and
future exporters. Exporters must call the v16 validators instead of assembling
lookalike JSON. Work stops without promotion when any required evidence,
receipt, calibration metric, handoff, or human authorization is absent or
cannot be hash-verified.
