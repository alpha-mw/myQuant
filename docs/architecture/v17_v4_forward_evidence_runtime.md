# V17 v4 Forward-Evidence Runtime

## Purpose

`run-forward` is the explicit V17 v4 Shadow observation lane. It accumulates
future-only research evidence from a sealed request. It is not the decision
mainline and has no authority to publish or activate a mainline run.

## Invocation

Create the canonical request through the content-addressed library helper; do
not hand-edit its request ID or semantic hash. Run only by exact path and byte
SHA:

```bash
quant-investor-v17-v4 run-forward \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/v17_v4_runs/forward_requests/<request_id>.json \
  --request-sha256 <sha256>
```

The request binds the strategy, CN decision session, cutoff, source snapshot,
factor-set pointer, policies, profile, and supplied stage inputs.

## Profiles and closure

- `EXPLORE` requires source, allocation, Quant, and full-universe factor
  observation. Experimental factors are allowed.
- `FORWARD_EVIDENCE` requires source, Core/Challenger allocation, Quant,
  full-universe factor observation, Fusion, strategy-pool observation, and a
  final immutable session reference.
- `RELEASE_CANDIDATE` remains subject to its strict Shadow closure. Its name
  does not imply mainline eligibility.

A missing required stage or a supplied-invalid optional stage blocks the run.
Only an absent optional stage may be `SKIPPED/UNAVAILABLE`; valid incomplete
evidence may be `SUCCEEDED/PARTIAL`.

## Publication

Each completed stage writes immutable output, replays it, then writes its
receipt. The final session reference is written last and is the only completion
marker:

```text
results/v17_v4_shadow/forward_evidence/strategies/{strategy}/runs/{request_id}/
results/v17_v4_shadow/forward_evidence/strategies/{strategy}/sessions/
  {decision_session}/{request_id}.json
```

Orphan stage files are audit material, not completed observations. Retrying
the same request is byte-idempotent; conflicting bytes fail closed.

## Evidence rules

- Market and membership inputs are PIT, strict-Parquet, and available no later
  than the decision cutoff.
- Quant is mandatory. Missing observations are never filled with zero.
- A symbol requires at least two factors, two families, and 0.5 coverage.
- Fundamental coverage explicitly reduces its effective Fusion weight.
- Full-universe factor evidence remains separate from strategy-pool evidence.
- Labels use only future Shanghai closes and stay bound to factor, policy, and
  source lineage. Historical backfill and cross-lineage pooling are forbidden.
- Factor-set pointer drift between computation and publication blocks the run.

## Authority boundary

Shadow artifacts cannot be copied, renamed, or reinterpreted as
`myquant.v17.v4.mainline-run.v1`, cannot advance
`results/v17_mainline/strategies/{strategy_id}/_active.json`, and cannot be
returned as `myquant.v17.v4.mainline-public-run.v1`. Provider, LLM-control,
broker, order, execution,
and trade side effects remain forbidden unless a separate maintenance action
was explicitly authorized.
