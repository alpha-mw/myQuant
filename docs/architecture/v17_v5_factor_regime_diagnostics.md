# V17 v5 Origin-Regime Factor Diagnostics

## Scope

Sprint 1D keeps the Sprint 1B descriptive statistics and changes only the
accepted regime evidence source. V5 now conditions already verified
origin-level Factor RankIC observations only on causal V4
`myquant.v17.v4.regime-evidence.v2` states that were sealed, published, and
available before the Factor origin cutoff. It is a descriptive research
surface only. It cannot create Factor weights, change a Factor tier, recommend
a lifecycle action, make an alpha-validity claim or grant production,
governance, portfolio or trading authority.

V5 reads V4 only through the Sprint-1A exact path/SHA closure reader. It does
not scan directories, select a latest artifact, write V4, rebuild regime from
raw market data, import or call the V4 runtime producer, or read V15 JSONL
history.

## V4 regime source discovery and pin

| Candidate | Version | Path | Causal at origin | Immutable | Replayable | Decision |
|---|---|---|---:|---:|---:|---|
| V4 causal Regime Evidence | `myquant.v17.v4.regime-evidence.v2` | explicit caller-supplied V4 artifact ref | Yes, if publication ordering and closure pass | Yes | Yes | Conditioning-eligible when policy v2 rules pass |
| Registered V4 regime evidence | `myquant.v17.v4.regime-evidence.v1` | explicit caller-supplied V4 artifact ref | No | Yes | Yes | Integrity-checkable only; conditioning-ineligible |
| V15 Markov history | mutable JSONL history | `results/regime/markov_regime_history.jsonl` | Unproven | No | No | Rejected |
| V17 v2 Markov overlay | `markov-overlay.v1` | legacy v2 artifact | Not admissible to v5 | Not under V4 contract | Not under V4 contract | Rejected |

The predecessor is pinned to V4 Sprint 1C commit
`1da7ffb636a3254940525d746549d15e827f06ba`. The current integration mode is
`WORKTREE_COLOCATED_PREDECESSOR`: the V5 branch keeps the V4 commit as an exact
merge parent, then V5 verifies the package/runtime manifests and schema bytes
from that integrated predecessor. V4 files are not hand-copied or rewritten by
V5.

The v1 artifact seals `strategy_id`, `cutoff`, `available_at`, identity,
byte/semantic SHA and `gross_multiplier`, but it has no causal hard state,
posterior, decision/effective session, publication timestamp or source refs.
`gross_multiplier`, `role`, `created_at` and other metadata are never used to
infer a state. V1 returns
`REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE`.

Policy v2 accepts only v2 evidence with:

```text
inference_kind = FILTERED_CAUSAL
smoothing_used = false
publication_phase = PRIOR_SESSION_EFFECTIVE_NEXT_SESSION
scope_kind = FULL_MARKET
hard_state_derivation = SEALED_ARGMAX_POLICY_V1
no_retroactive_causal_backfill = true
```

V5 must not recompute posterior probabilities, rerun argmax, alter the V4
tie-break rule, remap state names, infer a hard state from posterior-only
evidence, or convert v1 to v2. Conditioning states are exactly:

```text
趋势上涨
震荡低波
震荡高波
趋势下跌
```

`未知` is a valid V4 state but is conditioning-ineligible.

## Origin causality

One origin row binds exactly one:

1. V4 Factor evaluation receipt;
2. full-universe Factor observation;
3. naturally matured 20-Shanghai-session label;
4. observation run, request and source locator;
5. Factor implementation SHA-256;
6. V4 regime evidence artifact.

The binding requires:

```text
regime.version = myquant.v17.v4.regime-evidence.v2
regime.decision_session = factor_origin.decision_session
regime.effective_session = factor_origin.decision_session
regime.available_at <= factor_origin.cutoff
regime.published_at <= factor_origin.cutoff
regime.observed_through_session < factor_origin.decision_session
regime.observed_through_session = previous_open_session(factor_origin.decision_session)
label.origin_session = factor_origin.decision_session
label.horizon_sessions = 20
```

Only the origin regime is admissible. The label-end regime, any transition
inside the forward horizon, future path, hindsight classification,
full-sample-smoothed state, in-memory live state and V5-recomputed state are
forbidden. Stale earlier regimes and relaxed `effective_session <= origin`
fallbacks are forbidden. Duplicate origins, multiple conditioning-eligible v2
regimes for one origin, unsupported versions, strategy mismatches and
incomplete lineage fail closed.

V2 grouping uses only the hard state sealed by the V4 producer. Posterior
probabilities remain descriptive. If the sealed hard state is `未知`, the
origin is not included in any conditionable by-regime group and records
`REGIME_HARD_STATE_UNKNOWN` as a limitation.

## Diagnostic states

The only states are:

- `UNAVAILABLE`: a required Factor or regime evidence source is absent;
- `UNOBSERVED`: admissible sources exist but there are no naturally matured
  origins;
- `ACCUMULATING`: at least one valid origin binding exists.

No explicit real V4 v2 artifact means `UNAVAILABLE`, even when the v2 schema
and adapter are installed. A real v2 artifact with no mature Factor origin, or
only `未知` / unconditionable origins, remains `UNOBSERVED`.

The Sprint 1D CLI is validation-only for an explicit Factor receipt plus one
explicit regime artifact; it does not infer or assemble a multi-origin
inventory from that pair. An otherwise eligible pair therefore returns
`UNAVAILABLE`, `OBSERVED_FACTOR_REGIME_CLI_PATH_NOT_ENABLED`, and
`origin_binding_result=NOT_ENABLED` until a separately reviewed origin
assembly entrypoint exists.

Thresholds of 20 descriptive origins and 60 stability origins only annotate
sample maturity. They do not create a pass/fail, validity, promotion, weight or
tier conclusion. Sprint 1D adds no other state.

## Statistical definitions

The time-series unit is one naturally matured origin-level cross-sectional
RankIC. Stocks inside a cross-section are not treated as independent time
observations. RankIC is reused from the verified Sprint-1A origin evidence and
is never reconstructed from an aggregate receipt.

For origin values \(x_1,\ldots,x_n\):

```text
rank_ic_mean = sum(x_t) / n
rank_ic_std = sqrt(sum((x_t - mean)^2) / (n - 1))
rank_icir = rank_ic_mean / rank_ic_std
```

`rank_ic_std` and `rank_icir` are null for `n < 2`; ICIR is also null when the
sample standard deviation is zero.

Coverage is `comparable_symbol_count / eligible_symbol_count`. Means and
minimums use the exact bound origin rows. The p10 statistic uses deterministic
nearest-rank selection. Regime occupancy concentration is the Herfindahl sum
of squared origin shares.

Twenty-session forward labels overlap. Newey-West uses a fixed lag of 19:

```text
gamma_k = sum((x_t - mean) * (x_(t-k) - mean)) / n
w_k = 1 - k / 20
long_run_variance = gamma_0 + 2 * sum(w_k * gamma_k, k=1..19)
se(mean) = sqrt(long_run_variance / n)
t = mean / se(mean)
```

The HAC standard error and t-statistic are null for fewer than 20 observations,
nonpositive long-run variance or zero standard error. No p-value, FDR, PBO or
Deflated Sharpe is computed in Sprint 1D.

All numeric output is finite, deterministic, `ROUND_HALF_EVEN`, and rendered
at 12 decimal places. Undefined or zero-denominator statistics are null;
NaN/Infinity are forbidden. Regime rows and refs use explicit stable ordering.

These statistics are descriptive. Conditioning does not establish causality,
economic validity, online allocation weight or production eligibility.

## Authority and current real status

Every authority field is false. CLI state remains:

```text
default_protocol_state = V15_DEFAULT
global_activation_state = INACTIVE
run_state = INACTIVE
research_runtime_default = false
```

At Sprint-1D closure the expected real status remains `UNAVAILABLE` unless the
caller supplies both a real V4 Factor evaluation receipt and a real explicit
V4 regime-evidence v2 artifact with complete source closure. No fixture can be
wrapped as real evidence, no origin row is synthesized, and no
`ACCUMULATING` diagnostic is created from schema availability alone.

## Regime-chain deployability audit

Sprint 1D audits the V4 v2 chain without changing the V4 producer. The audit
checks whether each evidence recursively references the immediate predecessor,
whether closure depth, node count, byte count, replay time and memory grow with
session count, and whether a missed session can recover without historical
backfill.

Synthetic chain sizes are:

| Requested sessions | Estimated depth | Estimated nodes | Estimated bytes | Actual replay result |
|---:|---:|---:|---:|---|
| 20 | 40 | 160 | 81,920 | `BLOCKED` |
| 60 | 120 | 480 | 245,760 | `BLOCKED` |
| 260 | 520 | 2,080 | 1,064,960 | `BLOCKED` |
| 1,000 | 2,000 | 8,000 | 4,096,000 | `BLOCKED` |

The estimates describe the linear predecessor shape; they are not persisted
performance evidence. An actual V4 synthetic build/replay probe succeeds for
two contiguous sessions and first fails at session 3. The surfaced blocker is
`REGIME_EVIDENCE_V2_INPUT_TAMPER` with
`model_snapshot_ref readback failed`; the nested cause is the unchanged V4
closure-validation resource budget. Replay duration and peak memory for the
requested longer chains are therefore `NOT_MEASURED_AFTER_FIRST_FAILURE`.

The S0 bootstrap and S1 normal publications succeed. S2 is omitted. The S3
attempt is blocked before recovery by the same resource-budget failure.
An explicit S3 restart without a predecessor is also rejected with
`REGIME_EVIDENCE_V2_TEMPORAL_CAUSALITY` /
`NORMAL publication requires the contiguous prior v2`. The sealed V4
normal-publication rule requires the prior evidence effective session to equal
the current observed-through session, so the missed session has neither a
legal stale-predecessor fallback nor a legal restart path.

The audit records `V4_REGIME_CHAIN_SCALABILITY_GAP` if a 260-session chain
cannot replay under current safety limits. It records
`V4_REGIME_CHAIN_LIVENESS_GAP` if missing one required predecessor causes later
current sessions to remain permanently blocked. These gaps do not authorize V5
closure-limit increases or V4 producer edits in Sprint 1D.
