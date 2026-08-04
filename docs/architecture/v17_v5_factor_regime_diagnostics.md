# V17 v5 Origin-Regime Factor Diagnostics

## Scope

Sprint 1E-0B changes only the V5 predecessor binding and the read-only Regime
Evidence adapter. The diagnostic remains descriptive and retains the Sprint 1B
statistical definitions. It cannot create Factor weights, change a Factor tier,
recommend a lifecycle action, claim Factor or alpha validity, or grant
production, governance, portfolio, or trading authority.

V5 reads V4 only through the exact path/SHA bounded closure reader. It does not
scan for a latest artifact, call the V4 producer, reconstruct regime from raw
data, read V15 mutable history, write V4, or recursively traverse historical
Regime checkpoints.

## Pinned V4 predecessor

The active `WORKTREE_COLOCATED_PREDECESSOR` is the exact V4 commit:

```text
6a2fa23dec68d87eb686464a86d8ba8997416310
```

The earlier `73c5b6e...` bounded-regime binding remains available as immutable
v3 history. The additive v4 binding verifies the integrated V4 package manifest,
runtime manifest, V3 schema, V2 inference policy, V3 producer source, and the V2
publication block. No V4 source is copied or rewritten by V5.

| Source | Integrity | Conditioning |
|---|---:|---:|
| `myquant.v17.v4.regime-evidence.v1` | supported | rejected: no causal hard-state contract |
| `myquant.v17.v4.regime-evidence.v2` | supported | rejected: non-deployable publication chain |
| `myquant.v17.v4.regime-evidence.v3` | supported | eligible only after composite finality and continuity checks |
| V15 mutable JSONL or in-memory state | rejected | rejected |

There is no fallback to an older predecessor or Regime version.

## Composite finality

V3 evidence is finalized only when all of the following hold:

1. the evidence is read from an explicit workspace-relative path and expected
   byte SHA-256;
2. schema, identity, semantic SHA, strategy, cutoff, and pinned V4 closure
   verification pass;
3. `current_checkpoint_ref` resolves from an explicit path and byte SHA-256;
4. that checkpoint is a direct member of the evidence `source_refs` closure;
5. evidence and checkpoint agree on strategy, session binding, segment,
   continuity, posterior, hard state, record commitment, global accumulator,
   segment accumulator, chain identity, and all other duplicated sealed fields;
6. the current chain anchor, segment anchor, feature, model, transition, scope,
   and policy bindings form the bounded direct V4 closure.

A standalone checkpoint is not finalized evidence. Missing, orphaned, or
conflicting checkpoint state fails closed as
`REGIME_EVIDENCE_V3_NOT_FINALIZED`.

The checkpoint intentionally does not reverse-bind the evidence ID or evidence
byte/semantic SHA. Combining such a reverse binding with the evidence's
content-addressed checkpoint reference would create a hash cycle. Sprint 1E-0B
does not add a finality receipt and does not modify the V4 two-stage publication
protocol.

V5 never recomputes posterior, argmax, tie-breaks, transition commitments,
chain digests, segment continuity, or accumulators. It verifies the sealed
composite and the current bounded closure only.

## Conditioning eligibility

The V3 policy fixes:

```text
inference_kind = FILTERED_CAUSAL
smoothing_used = false
publication_phase = PRIOR_SESSION_EFFECTIVE_NEXT_SESSION
scope_kind = FULL_MARKET
hard_state_derivation = SEALED_ARGMAX_POLICY_V1
no_retroactive_causal_backfill = true
```

Current-evidence eligibility is deterministic:

| Current fact | Result |
|---|---|
| `CONTIGUOUS` and hard state is conditionable | eligible |
| `ROLLOVER` and hard state is conditionable | eligible |
| `GENESIS` | ineligible: `REGIME_CONTINUITY_GENESIS` |
| `RECOVERY` | ineligible: `REGIME_CONTINUITY_RECOVERY` |
| hard state `未知` | ineligible: `REGIME_HARD_STATE_UNKNOWN` |
| evidence not finalized | malformed/fail closed |

V5 does not infer missing evidence from segment sequence. Recovery does not
become conditionable merely because a later contiguous evidence exists.

## Origin causality

One conditionable origin uniquely binds:

1. a V4 Factor evaluation receipt and full-universe Factor observation;
2. a naturally matured 20-Shanghai-session label;
3. the exact Factor implementation SHA-256;
4. one finalized V3 evidence and its current checkpoint.

The binding requires:

```text
regime.decision_session = factor_origin.decision_session
regime.effective_session = factor_origin.decision_session
regime.available_at <= factor_origin.cutoff
regime.published_at <= factor_origin.cutoff
regime.observed_through_session =
    previous_open_session(factor_origin.decision_session)
label.origin_session = factor_origin.decision_session
label.horizon_sessions = 20
```

Stale-regime fallback, nearest-available fallback, label-end regime, transitions
inside the forward horizon, future or smoothed state, and duplicate V3 evidence
for one origin are forbidden.

The only diagnostic states are:

- `UNAVAILABLE`: a required real V4 Factor receipt or finalized V3 source is
  absent;
- `UNOBSERVED`: V3 exists but no mature conditionable origin exists, including
  all-GENESIS, all-RECOVERY, or all-`未知` samples;
- `ACCUMULATING`: at least one complete conditionable origin exists.

Schema installation alone never advances the state.

## Statistical definitions

The time-series unit is one naturally matured origin-level cross-sectional
RankIC. Stocks inside a cross-section are not independent time observations.
V5 reuses the Sprint 1A origin RankIC and never reconstructs it from an
aggregate receipt.

For origin observations \(x_1,\ldots,x_n\):

```text
rank_ic_mean = sum(x_t) / n
rank_ic_std = sqrt(sum((x_t - mean)^2) / (n - 1))
rank_icir = rank_ic_mean / rank_ic_std
```

Standard deviation and ICIR are null for `n < 2`; ICIR is also null when the
sample standard deviation is zero. Coverage p10 uses deterministic nearest-rank
selection. Regime concentration is the sum of squared regime shares.

Overlapping 20-session labels use Newey-West lag 19:

```text
gamma_k = sum((x_t - mean) * (x_(t-k) - mean)) / n
w_k = 1 - k / 20
long_run_variance = gamma_0 + 2 * sum(w_k * gamma_k, k=1..19)
se(mean) = sqrt(long_run_variance / n)
t = mean / se(mean)
```

The HAC metrics are null for insufficient samples, nonpositive long-run
variance, or zero standard error. No p-value, FDR, PBO, Deflated Sharpe,
validity verdict, allocation, or governance action is produced. All numerics
are finite deterministic decimal strings; JSON NaN and Infinity are forbidden.

## Current real status and authority

The current operational check remains:

```text
UNAVAILABLE
V4_FACTOR_EVIDENCE_UNAVAILABLE
V4_REGIME_EVIDENCE_V3_UNAVAILABLE
```

No fixture is a real receipt, no synthetic inventory is persisted, and no V4
producer is called. V15 remains default; global and run activation remain
inactive; every governance, promotion, execution, broker, order, trade,
selector, portfolio, provider, and LLM authority is false.
