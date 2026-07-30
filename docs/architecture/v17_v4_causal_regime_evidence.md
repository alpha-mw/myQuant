# V17 v4 Causal Regime Evidence

## Decision and boundary

`myquant.v17.v4.regime-evidence.v2` is an additive, filtered-causal research
artifact. It records one immutable regime classification produced from exact
V17 v4 source artifacts. It is artifact production only; it does not grant
production, default, formal, canary, promotion, portfolio, Factor Governance,
provider, selector, broker, order, execution, or trade authority.

The v2 contract remains valid for historical validation, status, and replay,
but new public v2 publication is disabled. The retained
`regime-evidence-build` command exits `2` with
`REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE`, creates no artifact, and does not
invoke either producer. `regime-evidence-v3-build` is the only command allowed
to publish new Regime Evidence. There is no automatic v2-to-v3 conversion and
no retrospective backfill.

The existing `myquant.v17.v4.regime-evidence.v1` schema, builder, validators,
bytes, and SHA identities remain unchanged. V1 is a portfolio-control
multiplier envelope. It is not a causal classification and is not accepted as
a v2 prior or conditioning input. V2 is not accepted by any existing v1
consumer, portfolio overlay, forward stage, formal-activation service,
eligibility service, canary service, Factor Governance surface, weight/tier
allocator, or runtime selector.

V5 adaptation and V5 predecessor-pin updates are deferred to Sprint 1D. Sprint
1C does not modify V5.

## Only permitted time mode

The only permitted inference mode is:

```text
PRIOR_SESSION_EFFECTIVE_NEXT_SESSION
```

For an observed Shanghai session `S`, the exact sealed calendar determines the
next open session `D`. The feature snapshot is complete only after the close of
`S`; the evidence is first eligible and effective on `D`. For example:

```text
observed_through_session = 2026-07-29
decision_session = 2026-07-30
effective_session = 2026-07-30
```

Same-session inference and execution are prohibited. A pre-close feature
family could support a different contract later, but it is not part of this
sprint.

`created_at`, `computed_at`, `available_at`, and `published_at` are the same
UTC-second timestamp. New publication requires all of the following:

- the timestamp is after `S` at `15:00 Asia/Shanghai`;
- the timestamp is no later than the declared Factor observation cutoff for
  `D`;
- the declared cutoff itself falls on `D` in `Asia/Shanghai`; a caller cannot
  move the cutoff to a later local date to publish evidence for a past
  decision session;
- `abs(captured UTC now - created_at) <= 300 seconds`.

The wall-clock freshness rule is checked only for a new publication. An exact
retry checks the occupied completion slot before the runtime clock, and
historical status/replay never applies a current wall-clock freshness test.
The monotonic clock is used only to enforce the 10-second closure-traversal
budget.

This same-local-session rule prevents producer-side historical backfill, but
it does not assert that a future Factor origin used the same declared cutoff.
The future read-only binding must still require
`regime.published_at <= factor_origin.cutoff` and exact session equality. Until
that origin exists, a v2 artifact is descriptive regime evidence, not proof
that any Factor origin consumed it.

The policy is `NO_RETROACTIVE_CAUSAL_BACKFILL`. A command run after the allowed
publication window cannot create a missing historical v2 artifact, even when
the historical inputs are internally consistent. Existing immutable evidence
may always be replayed by exact path and SHA.

## Scope and causal source closure

V2 accepts only `scope_kind=FULL_MARKET` backed by
`source_scope=FULL_PIT_MARKET`. The feature snapshot must prove exact symbol
set equality with the active PIT market membership for `S`, must declare
`sampled=false`, and must contain at least 30 symbols. `SUBSET`, sampled,
market-reference, inferred-full-market, or underfilled evidence is rejected.
A count alone never proves full-market coverage.

The CLI accepts only explicit input paths and expected byte SHA-256 values:

- inference policy;
- pinned no-training model snapshot;
- pinned native default transition matrix;
- full-market feature snapshot;
- optional immediately preceding v2 regime evidence.

There is no latest-pointer scan, directory discovery, glob, fallback, JSONL
history read, cache substitution, locator inference, or provider call. The
policy resource is:

```text
resources/regime_inference_policy.v1.json
```

The closure permits no hidden references. Every reference discovered in a
validated artifact must equal the declared reference set. Symlinks, multiple
hard links, absolute paths, `..`, ASCII-casefold aliases, cycles, forks, and
unregistered versions fail closed. Reference accounting counts unique exact
`(version, identity, path, byte SHA)` bindings, so replaying the same binding
does not consume the budget twice.

This producer accepts only registered, schema-validated canonical JSON
snapshots and terminals. It does not admit a direct Parquet or other opaque
raw-file reference. A future producer that derives the feature snapshot from
Parquet requires a separate contract and implementation; it cannot bypass
this closure. The registered market and PIT terminal schemas cap each sealed
symbol inventory at 10,000 entries.

Closure traversal is bounded by all of these limits:

| Limit | Value |
| --- | ---: |
| reference depth | 5 |
| artifact nodes | 16 |
| one JSON artifact | 2 MiB |
| total JSON bytes | 8 MiB |
| one raw/file artifact | 64 MiB |
| total raw bytes | 8 GiB |
| unique exact references | 128 |
| symbols in one registered market/PIT terminal | 10,000 |
| monotonic traversal budget | 10 seconds |

Limit exhaustion is an integrity failure. It is not converted to an empty
sample or input gap.

## Fixed model, transition, and native states

The model is fixed and has no training surface:

```text
model_id = v17-v4-native-regime-filtered-model
model_version = PINNED_RULE_BASED_NO_TRAINING_V1
model_kind = PINNED_RULE_BASED_NO_TRAINING
formula_version = NATIVE_HEURISTIC_LIKELIHOOD_DECIMAL_V1
model_training_end_session = null
training_source_refs = []
transition_source = PINNED_NATIVE_DEFAULT_V1
```

The native state order is authoritative for serialization and deterministic
ties:

1. `趋势上涨`
2. `震荡低波`
3. `震荡高波`
4. `趋势下跌`
5. `未知`

The fixed bootstrap prior in that order is:

```text
[0.25, 0.25, 0.25, 0.20, 0.05]
```

Bootstrap is permitted only for the exact policy bootstrap pair:
`decision_session=2026-07-30` and
`observed_through_session=2026-07-29`. It is not enough that no earlier v2
evidence happens to exist. Every later publication requires the unique prior
v2 artifact whose `effective_session` equals the current
`observed_through_session`. Missing predecessors, forks, gaps, late bootstrap,
and rebootstrap fail closed. A v1 artifact can never fill this role.

The pinned default transition rows are:

| From / to | 趋势上涨 | 震荡低波 | 震荡高波 | 趋势下跌 | 未知 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 趋势上涨 | 0.62 | 0.22 | 0.10 | 0.04 | 0.02 |
| 震荡低波 | 0.22 | 0.46 | 0.20 | 0.07 | 0.05 |
| 震荡高波 | 0.12 | 0.24 | 0.42 | 0.17 | 0.05 |
| 趋势下跌 | 0.07 | 0.18 | 0.25 | 0.46 | 0.04 |
| 未知 | 0.20 | 0.25 | 0.25 | 0.20 | 0.10 |

Historical transition estimation and mutable Markov JSONL are not part of the
v2 lineage.

## Deterministic inference

All arithmetic uses decimal strings parsed as `Decimal`; binary floating point
is not an inference authority. Intermediate and published probabilities use
precision 12 with `ROUND_HALF_EVEN`. `clamp(x)` means
`min(1, max(0, x))`.

The pinned no-training model computes:

```text
normalized_return = clamp((average_return + 0.015) / 0.030)
breadth_score = clamp(breadth)
momentum_score = clamp(momentum_share)
liquidity_score = clamp(average_liquidity)
macro_score_norm = clamp((macro_score + 1) / 2)
volatility_score = clamp(average_volatility / 0.035)

pressure_score = clamp(
    0.40 * median_drawdown / 0.20
  + 0.35 * fake_breakout_share
  + 0.25 * (1 - average_liquidity)
)

risk_on_score = clamp(
    0.30 * normalized_return
  + 0.25 * breadth_score
  + 0.20 * momentum_score
  + 0.15 * liquidity_score
  + 0.10 * macro_score_norm
  - 0.25 * volatility_score
  - 0.20 * pressure_score
)

neutral_score = 1 - min(abs(risk_on_score - 0.50) * 2, 1)
```

```text
L(未知) = 0.04

L(趋势上涨) =
    0.05 + 0.55*risk_on_score + 0.20*breadth
  + 0.15*(1-volatility_score) + 0.05*momentum_share

L(趋势下跌) =
    0.05 + 0.45*(1-risk_on_score) + 0.20*(1-breadth)
  + 0.20*pressure_score + 0.10*volatility_score

L(震荡低波) =
    0.05 + 0.35*(1-volatility_score) + 0.30*neutral_score
  + 0.20*(1-pressure_score) + 0.10*breadth

L(震荡高波) =
    0.05 + 0.35*volatility_score + 0.35*pressure_score
  + 0.15*neutral_score + 0.05*fake_breakout_share
```

After normalizing likelihood `L`, prior propagation and filtering are:

```text
predicted[j] = sum(previous[i] * transition[i,j] for i in native states)
unnormalized[j] = predicted[j] * L[j]
posterior[j] = unnormalized[j] / sum(unnormalized)
```

This is filtered causal inference only. No backward pass, smoothed state,
future observation, revised posterior, or reader-side reclassification is
allowed. The builder seals the hard state and full posterior. The hard state
is the argmax of the serialized posterior, with ties resolved by native state
order. The reader replays and verifies it; the reader cannot select a new
state.

Published posterior values use largest-remainder reconciliation so the five
serialized decimals sum exactly to `1.000000000000`. Equal remainders use
native state order. NaN, Infinity, out-of-range probabilities, missing or
extra states, and a non-unit posterior fail closed.

## Publication, exact-once behavior, and rollback

The publication behavior below describes the frozen Python v2 producer used
for legacy artifact verification and isolated contract/fixture tests. It is
not reachable from a public V4 publication command.

The caller prepositions and seals child inputs first. The builder validates
and replays the policy, model, transition, feature, and optional predecessor
closure, computes the causal filter, validates the candidate evidence, and
publishes the evidence last. The single completion slot is:

```text
data/private/v17_v4_sources/regime_evidence/
  {strategy_id}/{effective_session}/regime_evidence.v2.json
```

One strategy/effective-session slot may contain only one canonical byte
identity. An identical retry returns the existing artifact with
`reused=true`; a different candidate is a conflict. Concurrent identical
publishers converge on the same bytes. Concurrent conflicting publishers
produce one winner and one blocked conflict. A post-write readback failure
never overwrites the occupied slot.

The artifact seals descriptive state, posterior, exact source references,
causal sessions and timestamps, replay result, `blocker_codes`, and the all-false
authority attestation. It contains no weight, tier, allocation, action,
governance decision, portfolio instruction, or execution instruction.

The public v2 build path is disabled. Immutable v1 and v2 evidence is retained
for audit, status, and replay. This quarantine never deletes, rewrites,
relabels, truncates, converts, or reuses an occupied artifact.

## Missing input and failure classification

Missing genuine current policy/model/transition/feature/predecessor closure
returns:

```text
status = TRUE_CURRENT_CANONICAL_INPUT_GAP
exit_code = 2
```

No completion artifact is created. SHA drift, schema or semantic mismatch,
causality failure, malformed probability, security failure, conflict,
traversal-limit exhaustion, hidden reference, and replay divergence are
integrity failures. They return `status=BLOCKED`, one or more `blocker_codes`,
and exit 2. Integrity failures are never relabeled as input gaps.

Both success and blocked CLI responses include the canonical all-false
authority object with exactly `formal_research_publication`,
`research_runtime_default`, `execution`, `broker`, `order`, and `trade`.
Provider, Factor Governance, portfolio, selector, broker, order, execution,
and trade side effects are separately attested false. Canary, promotion,
performance, same-session execution, and formal eligibility remain false in
the artifact; weight, tier, and allocation fields are absent.

Those classifications remain part of the frozen Python builder and historical
reader/replay behavior. The public `regime-evidence-build` command does not
inspect input closure: it always returns
`REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE`, reports
`CONTRACT_VALIDATED_NOT_DEPLOYABLE`, points to
`regime-evidence-v3-build`, and attests `artifact_created=false`.

## Acceptance matrix

The 45 user cases are normative:

| # | Case | Expected result |
| ---: | --- | --- |
| 1 | legal filtered causal inference | publish and replay |
| 2 | hard state uses native enum | accept |
| 3 | posterior is complete and sums to 1 | accept |
| 4 | argmax tie is deterministic | native-order winner |
| 5 | native hard state and posterior policy are consistent | accept |
| 6 | smoothed inference | fail closed |
| 7 | future feature row | fail closed |
| 8 | feature `available_at` after cutoff | fail closed |
| 9 | model training end after session | fail closed |
| 10 | model SHA mismatch | fail closed |
| 11 | transition SHA mismatch | fail closed |
| 12 | feature SHA mismatch | fail closed |
| 13 | semantic SHA mismatch | fail closed |
| 14 | identity mismatch | fail closed |
| 15 | strategy mismatch | fail closed |
| 16 | session mismatch | fail closed |
| 17 | `SUBSET` scope | fail closed |
| 18 | sample below minimum | fail closed |
| 19 | posterior missing a state | fail closed |
| 20 | posterior contains an extra state | fail closed |
| 21 | NaN or Infinity | fail closed |
| 22 | probability outside `[0,1]` | fail closed |
| 23 | posterior sum is not exactly 1 | fail closed |
| 24 | hard state differs from sealed argmax | fail closed |
| 25 | publication after Factor cutoff | fail closed |
| 26 | effective session after decision session | fail closed |
| 27 | symlink in closure | fail closed |
| 28 | multiply linked file | fail closed |
| 29 | path escape | fail closed |
| 30 | hidden reference | fail closed |
| 31 | recursive cycle | fail closed |
| 32 | depth, node, byte, symbol, ref, or time limit | fail closed |
| 33 | exact-once identical reuse | return existing bytes |
| 34 | conflicting duplicate | fail closed |
| 35 | pre-publication candidate replay | validate successfully before write |
| 36 | post-publication exact readback replay | accept |
| 37 | canonical bytes are deterministic | byte-identical |
| 38 | emitted JSON contains no NaN or Infinity | accept |
| 39 | v1 bytes and SHAs are unchanged | zero drift |
| 40 | V15, V16, V17 v2/v3, and V5 are unmodified | zero diff |
| 41 | weight, tier, and allocation fields are absent | accept |
| 42 | every authority field is false | accept |
| 43 | no provider, broker, order, or trade side effect | accept |
| 44 | historical backfill attempt | `NO_RETROACTIVE_CAUSAL_BACKFILL` |
| 45 | no qualified current inputs | `TRUE_CURRENT_CANONICAL_INPUT_GAP`, no completion |

Additional Critic cases are also required:

- delayed identical retry bypasses new-publication clock freshness and returns
  the occupied identical slot;
- historical status/readback replays without a wall-clock freshness test;
- concurrent identical and concurrent conflicting publishers have the
  exact-once outcomes defined above;
- a forked predecessor, missing contiguous predecessor, or late bootstrap
  fails closed;
- new-publication freshness is tested at `-300`, `+300`, and just outside both
  boundaries;
- mutation of every source and training leaf is detected;
- inference-policy-to-source SHA closure succeeds when exact and fails on any
  drift;
- an injected `artifact_loader` is propagated through every recursive read.
- a direct Parquet or opaque raw-file reference is rejected rather than
  treated as a validated source node.

## Stop condition and deferred work

Sprint 1C stops when v2 schemas, semantic validators, fixed resources, runtime
producer/replay, explicit CLI, exact test matrix, manifests, and report-backed
real-input status are complete. Missing real inputs remain
`TRUE_CURRENT_CANONICAL_INPUT_GAP`; no fixture is promoted to current evidence.

Sprint 1E-0A.1 subsequently quarantines only new public v2 publication. It
does not change the frozen v1/v2 contracts, validator semantics, Python
producer, status/replay behavior, or any v3 contract or producer.

This sprint does not build a source snapshot or trainer, migrate Markov JSONL,
change v1 consumers, add overlay or forward-stage acceptance, add governance,
weights, tiers, or allocations, update the V5 adapter/pin, create a schedule,
or invoke any provider, selector, broker, order, execution, or trade surface.
