# V17 Sprint R2.2 Forward Research Evaluator

## Scope

Sprint R2.2 adds an offline, research-only evaluation layer above exact V4
Forward Observation evidence. It evaluates matured outcomes; it does not create
observations or labels, fetch data, choose a factor, change a factor weight,
update a Bayesian posterior, persist memory, select securities or authorize an
order.

The implementation is isolated under
`quant_investor/intelligence/evaluator/`. It does not replace the V4 observation
runtime or the current V17 mainline public contracts. The only supported
decision protocol remains `myquant.v17.v4`. The evaluator does not inspect,
infer or change any strategy active pointer, so code availability does not imply
operational activation.

## Exact input protocol

The public command accepts one request locator:

```bash
quant-investor-v17-v4 research-evaluate \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/research_intelligence/evaluation_requests/forward-evaluation-request-<sha256>.json \
  --request-sha256 <exact-byte-sha256>
```

The filename, `request_id`, semantic SHA and exact bytes are mutually bound.
The request is canonical JSON with version
`myquant.v17.research-intelligence.forward-evaluation-request.v1`. It contains:

- `as_of` and `evaluated_at`, which must be the same canonical UTC-second time;
- one sealed `forward-evaluation-policy.v1`;
- one to 64 canonical origins;
- one exact memory inventory ref;
- the closed no-authority and protocol-state fields.

Each origin binds:

- an exact V4 Session relative path and byte SHA;
- one Factor Observation binding for every declared factor;
- exactly one required `v17-quant-core` observation and optional Industry and
  Industry+Theme observations;
- one explicit universe-anchor Factor Observation;
- one explicit matured Label;
- zero or more explicit V4 Evaluation Receipts;
- the complete explicit recursive closure;
- an optional causal I0 Regime Input/Receipt binding.

Every sealed I0 Hypothesis, Evidence, Regime or memory document is referenced by
five fields: artifact id, artifact version, byte SHA, semantic SHA and exact
relative path. Its only allowed path is:

```text
data/private/research_intelligence/evaluation_inputs/<artifact-id>.json
```

The evaluator uses a single shared descriptor-relative reader. Paths that are
absolute, traversing, symlinked, hardlinked, case-ambiguous, unsafe, changed
during read or inconsistent with their SHA fail closed. No pointer, `latest`,
directory scan or stale fallback is permitted.

## Evaluation policy

The sealed policy fixes before evaluation:

- horizon: `1`, `5`, `10`, `20` or `60` Shanghai sessions;
- minimum subject count, matured-origin count, joint coverage and industry
  mapping coverage;
- factor identities, exact definitions and orientation;
- the three fixed variants and all seven comparison rules;
- hypothesis metric/window rules and exact falsification bindings;
- descriptive calibration mappings;
- regime minimum stratum size and `GLOBAL_BREADTH` scope.

Hypothesis rule and falsification windows use canonical `YYYY-MM-DD` origin
session dates. Timestamp boundaries are rejected because the evaluator's
atomic observation identity is a Shanghai trading session, not an intraday
instant. Exact factor/window recomputations are shared across hypotheses and a
request may contain at most 128 distinct factor-window combinations.

A policy created after the earliest evaluated origin is post-hoc. Its numeric
evidence remains visible, but variant and hypothesis conclusions are downgraded
to inconclusive/uncertain and the limitation is explicit.

## Factor evaluation

Each origin is evaluated cross-sectionally before equal-origin aggregation.
Decimal precision is 50; serialized values use 12 places and round-half-even.
Higher/lower orientation is applied before ranking. Ties receive average ranks,
and a tie is never split across quintiles.

The fixed 20-row metric inventory is:

1. `rank_ic`
2. `icir_base`
3. `icir`
4. `quantile_return_q1`
5. `quantile_return_q2`
6. `quantile_return_q3`
7. `quantile_return_q4`
8. `quantile_return_q5`
9. `long_short_spread`
10. `turnover`
11. `score_coverage`
12. `label_coverage`
13. `joint_coverage`
14. `industry_mapping_coverage`
15. `origin_maturity_coverage`
16. `neutralized_alpha`
17. `cost_adjusted_return`
18. `q5_long_only_cost_adjusted_return`
19. `drawdown`
20. `stability`

`rank_ic` is Spearman correlation of oriented score and total return within an
origin. `icir_base` is mean origin RankIC divided by its sample standard
deviation. `icir` is the deliberately naive annualization:

```text
icir = icir_base * sqrt(252 / horizon_sessions)
```

The annualization limitation states that serial correlation is not adjusted.
Gross spread is Q5 minus Q1. `cost_adjusted_return` deducts one flat 20 bp from
that spread; the supplied Q5 cost-adjusted label is not charged twice.
Turnover is one half of the absolute Q5-weight change and is available only
across explicit consecutive origins. Drawdown compounds the Q5 long-only
cost-adjusted return only when all forward windows are strictly non-overlapping.
Coverage is always calculated against the sealed universe anchor. Missing
labels are never imputed.

The receipt does not contain or mutate a factor weight, tier, lifecycle,
promotion or governance decision.

## Variant comparison

The only admitted variants are:

```text
v17-quant-core
v17-quant-plus-industry
v17-quant-plus-industry-theme
```

Industry and Industry+Theme are compared independently with Core on return,
RankIC, ICIR, turnover, drawdown, coverage and cost-adjusted return. Every rule
has a direction, tolerance, improvement threshold and degradation veto. A
positive incremental conclusion requires no veto and at least one thresholded
improvement. Missing optional observations stay `UNAVAILABLE`; paired-origin
mismatches remain explicit. The Industry+Theme comparison is cumulative versus
Core and does not claim to isolate Theme from Industry.

No comparison selects a production variant.

## Hypothesis and falsification

The I0 hypothesis and every supporting/contrary evidence item are replayed at
their own historical timestamp. Their exact evidence set must match the sealed
hypothesis, every source must belong to the authorized V4 closure, and each
falsification horizon must equal the evaluation horizon.

The result vocabulary is:

- `SUPPORTED`: preregistered, all support rules pass, no evaluated contrary
  rule passes, no falsification condition triggers, and maturity/coverage gates
  pass;
- `FAILED`: a preregistered numeric falsification condition triggers;
- `UNCERTAIN`: incomplete/conflicting evidence, unavailable metrics, or any
  post-hoc conclusion.

Falsification is reported independently as `TRIGGERED`, `NOT_TRIGGERED` or
`INCONCLUSIVE`. A failed hypothesis is not deleted.

## Bayesian calibration seam

Calibration groups historical evidence by source type, direction and strength,
then records mature count, success count and empirical success rate under an
explicit factor-metric mapping. This is `Calibration Evidence`, not a posterior
update. The evaluator imports no posterior writer, accepts no posterior target
and emits the limitation `BAYESIAN_CALIBRATION_DIAGNOSTIC_ONLY`.

## Regime-conditioned evaluation

Regime bindings replay an exact I0 Regime Input and Regime Receipt at the
receipt's historical timestamp. Only the selected Market, Industry and Theme
states from the one-step causal filter are consumed. Posterior maps are not
accepted by the aggregation helper; there is no backward smoothing, model
fitting or parameter update.

When the optional binding is supplied, the evaluator revalidates both artifacts
and requires exact replay of the receipt's input ref and evidence refs. Every
Regime Input source and every regime evidence source must be a member of the
origin's recursively authorized V4 source closure. Both binding scope fields
must equal `GLOBAL_BREADTH`. Temporal binding is closed as
`input available_at <= receipt timestamp <= origin run cutoff`. An
outside-closure, malformed or mistimed supplied binding blocks the whole
evaluation; it is never downgraded to an unavailable optional feature.

Each layer is evaluated independently, not as a joint state. The ten
regime-conditioned metrics are RankIC, raw and annualized ICIR, spread,
cost-adjusted return, joint coverage, neutralized alpha, stability, turnover
and drawdown. Insufficient state origins remain `UNAVAILABLE`.

## Memory integration

The request binds one immutable memory inventory and its hash-chain tip. The
evaluator returns a proposed suffix; it never writes the inventory. For every
hypothesis the suffix contains `EVALUATED`, followed by
`HYPOTHESIS_SUPPORTED` or `HYPOTHESIS_FALSIFIED` when resolved. A falsified
hypothesis also appends `FAILED_CASE` with status `FAILED`. The supplied chain
is not mutated, and the proposal binds both the expected prior tip and proposed
new tip.

Persistence is a separate caller-owned compare-and-swap action. R2.2 has no
filesystem writer for evaluation outputs or memory.

## Receipt topology

The stdout envelope embeds independently content-addressed artifacts:

```text
Universe Inventory
Factor Evaluation Receipts
Variant Factor Evaluation Receipts
Variant Comparison Receipt
Hypothesis Evaluation Receipts
Calibration Evidence Receipt
Regime Evaluation Receipt
Memory Append Proposal
Forward Evaluation Receipt
```

The primary `forward-evaluation-receipt.v1` includes at least:

```text
evaluation_id
observation_refs
label_refs
factor_refs
hypothesis_refs
universe_ref
evaluation_window
metrics
limitations
implementation_sha
```

It also binds the request, policy, variant comparison, calibration, regime,
hypothesis evaluation, memory proposal and embedded subreceipt inventory. The
implementation SHA comes from the verified I0 package manifest.

## Authority and exits

Every content-addressed R2.2 output carries:

```text
research_only = true
production = false
decision_protocol = myquant.v17.v4
mainline_authority = false
operational_activation_unchanged = true
broker = false
execution = false
order = false
trade = false
```

The shared I0 authority map also keeps provider, LLM, factor-governance write,
formal activation, portfolio, research-runtime default and selector false.

CLI exit codes are `0` for a complete canonical envelope, `2` for an expected
request/artifact/temporal/memory blocker and `3` for implementation-integrity or
unexpected internal failure. Blocked envelopes preserve the same closed
authority state and do not expose a traceback on stdout.
