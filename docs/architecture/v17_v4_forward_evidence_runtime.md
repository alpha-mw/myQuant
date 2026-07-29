# V17 v4 Forward Evidence Runtime

## Scope

The forward-evidence lane is additive to `myquant.v17.v4`. It accumulates
immutable research observations without changing the production/default
protocol:

```text
default_protocol_state = V15_DEFAULT
global_activation_state = INACTIVE
run_state = FORWARD_EVIDENCE_ACTIVE
research_runtime_default = false
formal_activation_eligible = false
```

`V15_DEFAULT` identifies the unchanged default protocol.
`global_activation_state=INACTIVE` states that V17 is not globally activated.
`FORWARD_EVIDENCE_ACTIVE` appears only as the state of one successfully closed
research run. It never grants formal publication, canary, promotion, provider,
execution, broker, order, or trade authority.

The existing strict `shadow-run.v3` and `shadow-session-ref.v3` contracts are
unchanged. `RELEASE_CANDIDATE` delegates to that closed path. The additive
forward-evidence artifact family is not accepted by formal activation, canary,
or default-routing services.

## Immutable request

`run-forward` accepts one exact request reference:

```text
data/private/v17_v4_runs/forward_requests/{request_id}.json
```

The request binds profile, strategy, decision session, cutoff, source
snapshot, factor-set pointer and every provided stage input by canonical path
and byte SHA-256. `request_id` is
`forward-request-{sha256(canonical request body)}`; the body excludes only
`request_id` and `semantic_sha256`.

The CLI does not accept duplicate profile, strategy, or cutoff flags. This
prevents operator arguments from drifting from the sealed request.

## Profiles

| Profile | Required closure | Optional closure | Successful state |
| --- | --- | --- | --- |
| `EXPLORE` | source, allocation, Quant, full-universe factor observation | Fundamental, Fusion, strategy observation, Deep, holdings | `EXPLORE_COMPLETE` |
| `FORWARD_EVIDENCE` | source, Core/Challenger allocation, Quant, full-universe factor observation, Fusion, strategy observation, final session ref | Fundamental, Deep, holdings | `FORWARD_EVIDENCE_ACTIVE` |
| `RELEASE_CANDIDATE` | unchanged strict Shadow v3 closure | none | strict delegate result |

Each stage has two independent state axes:

- execution outcome: `SUCCEEDED`, `BLOCKED`, or `SKIPPED`;
- evidence completeness: `COMPLETE`, `PARTIAL`, or `UNAVAILABLE`.

Only an absent optional input may be `SKIPPED/UNAVAILABLE`. A supplied input
with invalid schema, SHA, PIT time, future data, replay, authority, or lineage
blocks the run. A required missing stage also blocks the run.

## Publication and recovery

Each stage writes an immutable output, reads and replays the exact bytes, then
writes its completion receipt. A later optional stage cannot discard earlier
valid outputs. Run artifacts use:

```text
results/v17_v4_shadow/forward_evidence/strategies/{strategy}/runs/{request_id}/
  outputs/{stage}.json
  receipts/{stage}.json
  run.json
results/v17_v4_shadow/forward_evidence/strategies/{strategy}/sessions/
  {decision_session}/{request_id}.json
```

The session ref is the only discoverable completion marker. It is written
last, after all required receipts replay, transitive references replay, and
the exact factor-set pointer is reread. A crash may leave immutable orphan
stage bytes, but it cannot create a complete session. Retrying the same
request is byte-idempotent; different bytes at an existing identity fail
closed.

Artifacts are capped at 64 MiB and publication requires at least 512 MiB of
free space.

## Factor allocation

Tier is evidence-derived and frozen in the run allocation:

- `CORE`: exact Factor Governance production active-set plus activation and
  health closure;
- `CHALLENGER`: a 20-session diagnostic receipt with at least 60 distinct
  mature origins, at least 100 symbols per origin, mean RankIC above 0.02,
  annualized RankICIR at least 0.5, positive cost-adjusted group return,
  stability at least 0.60, maximum absolute correlation to existing factors
  below 0.70, and freshness within five exact open sessions;
- `EXPERIMENTAL`: computable but without either complete evidence closure.

Daily `FORWARD_EVIDENCE` uses only Core and Challenger factors. `EXPLORE` may
run Experimental factors to start new evidence. Experimental factors never
become production factors automatically.

## Quant v3

For each decision session:

1. Type-7 1%/99% winsorization.
2. Robust z-score `(x - median) / (1.4826 * MAD)`.
3. Industry fixed-effect residual.
4. Log-market-cap residual.
5. Joint beta-252d and Amihud-20d residual.
6. Equal-weight aggregation inside each available family.
7. Equal-weight aggregation across available families.

`MAD=0` produces exposure zero with `ZERO_MAD`; missing observations are never
filled. Every neutralizer must be exact-reference bound and available no later
than the decision cutoff.

Per symbol:

```text
factor_coverage = available_factor_count / selected_factor_count
family_coverage = available_family_count / selected_family_count
coverage_ratio = min(factor_coverage, family_coverage)
confidence_penalty = 1 - coverage_ratio
effective_score = composite_score * coverage_ratio
```

At least two factors, two families, and 0.5 coverage are required for a
score. Otherwise that symbol is explicitly `UNAVAILABLE`; the run does not
invent a zero score.

## Industry, theme, Fundamental, and Fusion

Industry scoring uses bounded `[0,1]` inputs:

```text
0.15 demand
+ 0.10 supply
+ 0.10 inventory
+ 0.10 pricing power
+ 0.05 capex
+ 0.15 earnings revision
+ 0.10 policy
+ 0.10 market confirmation
+ 0.05 narrative
+ 0.10 * (1 - crowding risk)
```

The sum is multiplied by bounded confidence. Theme scoring permits positive
weight only for `DIRECT_BENEFICIARY` or a `SUPPLIER` with confidence at least
0.80. `SECOND_ORDER` and `CONCEPT_ONLY` receive zero theme weight.

Fundamental configured weights are:

```text
financial quality  25%
industry cycle     25%
earnings revision  20%
theme/narrative    10%
valuation          15%
governance          5%
```

Available components are reweighted for the raw score. Coverage is the sum of
their configured weights, and the effective score is `raw * coverage`.
Coverage 1 is `COMPLETE`, coverage from 0.25 to below 1 is `PARTIAL`, and
coverage below 0.25 is `UNAVAILABLE`.

Fusion retains the 50/50 configured Quant/Fundamental policy while making
availability explicit. Quant is mandatory. Fundamental's effective weight is
`0.5 * fundamental_coverage`. Branch percentiles use average ties and only
score-present symbols. Fusion publishes raw score, coverage, penalty, and
coverage-adjusted effective score; final ties use symbol ASCII order.

## Observations, labels, and evaluation

Full-universe factor observations are separate from strategy-pool Fusion
observations. This prevents Top-N preselection from contaminating Factor
Governance evidence.

Labels use the exact Shanghai open-session calendar at horizons 1, 5, 10, 20,
and 60. They are adjusted-close simple returns and also bind exact market,
industry, and flat 20 bps round-trip cost adjustments. A label matures only
after 15:00 Asia/Shanghai on its end session. No historical backfill,
cross-origin pooling, or inferred calendar is allowed.

The evaluation receipt binds the sorted immutable origin inventory and exact
existing-factor inventory. Its lineage key is factor name, definition SHA,
factor-set SHA, Quant-policy SHA, horizon, and source-lineage SHA. Duplicate
requests with the same observation SHA use the ASCII-smallest reference and
record the duplicates. Different byte or semantic SHAs for the same lineage
block the complete inventory as `DUPLICATE_ORIGIN_CONFLICT`; the evaluator
must not choose a preferred origin.

## Rollback

Rollback disables the new command or schedule only. Immutable forward
evidence is retained for audit. No forward artifact is deleted, no older
schema is rewritten, and V15 remains the default throughout.
