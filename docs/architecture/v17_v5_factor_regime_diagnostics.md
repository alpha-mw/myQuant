# V17 v5 Origin-Regime Factor Diagnostics

## Scope

Sprint 1B conditions already verified origin-level Factor RankIC observations
on the regime state that was sealed and available at the same decision origin.
It is a descriptive research surface only. It cannot create Factor weights,
change a Factor tier, recommend a lifecycle action, make an alpha-validity
claim or grant production, governance, portfolio or trading authority.

V5 reads V4 only through the Sprint-1A exact path/SHA closure reader. It does
not scan directories, select a latest artifact, write V4, rebuild regime from
raw market data or import a V4 runtime producer.

## V4 regime source discovery

| Candidate | Version | Path | Causal at origin | Immutable | Replayable | Decision |
|---|---|---|---:|---:|---:|---|
| Registered V4 regime evidence | `myquant.v17.v4.regime-evidence.v1` | explicit caller-supplied V4 artifact ref | No | Yes | Yes | Registered provenance only; conditioning-ineligible |
| V15 Markov history | mutable JSONL history | `results/regime/markov_regime_history.jsonl` | Unproven | No | No | Rejected |
| V17 v2 Markov overlay | `markov-overlay.v1` | legacy v2 artifact | Not admissible to v5 | Not under V4 contract | Not under V4 contract | Rejected |

The registered V4 artifact seals `strategy_id`, `cutoff`, `available_at`,
identity, byte/semantic SHA and `gross_multiplier`, but has no hard regime
state, posterior, decision/effective session, publication timestamp or source
refs. `gross_multiplier`, `role`, `created_at` and other metadata are never
used to infer a state. There is no competing conditioning-eligible source, so
the current result is `REGIME_EVIDENCE_UNAVAILABLE`, not
`AMBIGUOUS_REGIME_EVIDENCE_SOURCE`.

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
regime.available_at <= factor_origin.cutoff
regime.published_at <= factor_origin.cutoff
regime.effective_session <= factor_origin.decision_session
regime.decision_session <= factor_origin.decision_session
label.origin_session = factor_origin.decision_session
label.horizon_sessions = 20
```

Only the origin regime is admissible. The label-end regime, any transition
inside the forward horizon, future path, hindsight classification,
full-sample-smoothed state, in-memory live state and V5-recomputed state are
forbidden. Duplicate origins, multiple regime states for one origin,
unsupported versions, strategy mismatches and incomplete lineage fail closed.

If a future V4 artifact contains only posterior probabilities, the adapter
preserves them but returns `REGIME_HARD_STATE_UNAVAILABLE`; it does not choose
an argmax. If it contains both a sealed hard state and posterior, grouping uses
only the sealed hard state and posterior remains descriptive.

## Diagnostic states

The only states are:

- `UNAVAILABLE`: a required Factor or regime evidence source is absent;
- `UNOBSERVED`: admissible sources exist but there are no naturally matured
  origins;
- `ACCUMULATING`: at least one valid origin binding exists.

Thresholds of 20 descriptive origins and 60 stability origins only annotate
sample maturity. They do not create a pass/fail, validity, promotion, weight or
tier conclusion. Sprint 1B adds no other state.

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
Deflated Sharpe is computed in Sprint 1B.

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

At Sprint-1B closure there is no conditioning-eligible real V4 hard-state
regime artifact and no real V4 Factor evaluation receipt. Therefore the real
status is `UNAVAILABLE`; no real origin row or `ACCUMULATING` diagnostic is
created.
