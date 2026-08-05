# V17 v4 Mainline Contract

## Scope

`myquant.v17.v4` is the only supported decision mainline. It is a CN-only,
research and portfolio-decision runtime. It has no broker, order, execution, or
trade authority. Deterministic data, Factor, risk, portfolio, and readiness
gates remain authoritative; review-model output is advisory.

## Three v1 artifacts

The mainline authority chain has exactly three roles:

1. `myquant.v17.v4.mainline-run.v1` is an immutable, fully closed run at
   `results/v17_mainline/strategies/{strategy_id}/runs/{run_id}/run.json`.
2. `myquant.v17.v4.mainline-active-pointer.v1` is the single mutable authority object at
   `results/v17_mainline/strategies/{strategy_id}/_active.json`. It references
   one exact immutable run.
3. `myquant.v17.v4.mainline-public-run.v1` is a read-only DTO derived from the active pointer and its
   referenced run. It is never persisted as replacement authority.

Every artifact is canonical JSON and binds its protocol, market, strategy,
cutoff, identity, byte references, and semantic SHA. Unknown fields, wrong
versions, a non-CN market, unsafe paths, stale expected hashes, and incomplete
transitive references fail closed.

The immutable run has the exact fields `schema_id`, `protocol`,
`canonical_strategy_id`, `run_id`, `created_at`, `market`, `capabilities`,
`authority_source`, `formal_output_ref`, `portfolio_output_ref`,
`source_closure_ref`, and `semantic_sha256`. Its closed constants are
`market=CN_A_SHARE`, `capabilities=[RESEARCH_PORTFOLIO]`, and
`authority_source=FORMAL_V17_V4`.

The active pointer has exactly `schema_id`, `protocol`,
`canonical_strategy_id`, `run_id`, `updated_at`, `run_ref`, and
`semantic_sha256`. Its `run_ref.relative_path` must equal the canonical run
path implied by the same strategy and run IDs.

The public projection has exactly the V17 protocol and strategy identity,
`state=ACTIVE`, `market=CN_A_SHARE`, `capability=RESEARCH_PORTFOLIO`,
`authority_source=FORMAL_V17_V4`, all side-effect flags false,
`read_only=true`, `selector_used=false`, `fallback_used=false`, exact pointer,
run, formal, portfolio, and source references, cash/gross weights, targets, and
its semantic SHA.

## Authority and storage

```text
results/v17_mainline/
  strategies/{canonical_strategy_id}/
    _active.json
    runs/{run_id}/run.json
```

Only the exact active pointer grants public visibility. A directory entry, a
completed Shadow session, or a valid immutable run without pointer activation
has no public authority. Readers must never scan for the newest run.

The pointer is advanced only by the governed activation writer using an exact
expected prevalue, an atomic replacement, and exact post-write readback. The
immutable run must already exist and validate before the pointer write. A
failed precondition leaves the prior pointer byte-for-byte unchanged.

## Public resolution

All CLI, Web, Dashboard, and scheduled read surfaces resolve the same active
pointer. Resolution is read-only and deterministic:

```text
read exact pointer
  -> validate myquant.v17.v4.mainline-active-pointer.v1
  -> resolve exact run reference
  -> validate myquant.v17.v4.mainline-run.v1 and transitive closure
  -> project myquant.v17.v4.mainline-public-run.v1
```

If the pointer does not exist, the public state is exactly
`V17_MAINLINE_UNINITIALIZED`. The reader writes no bootstrap file, pointer,
run, cache, or fallback result. If the chain is present but invalid, it returns
`V17_MAINLINE_BLOCKED:<closed blocker>` and writes nothing.

## Code availability is not activation

Installing, merging, or deploying V17 v4 code makes the contract executable;
it does not activate any strategy. Operational activation is a separate,
auditable act that publishes a validated immutable run and then advances the
strategy's active pointer under CAS/readback.

Conversely, removing an entrypoint from a deployment does not rewrite or delete
governed artifacts. Operators must treat code rollout and active-pointer state
as separate evidence.

## Unsupported surfaces

The mainline currently supports CN only. A non-CN public request returns
`V17_MARKET_UNSUPPORTED` before result resolution or writes. Mainline backtesting is unsupported and returns
`V17_BACKTEST_UNAVAILABLE` with zero writes. There is no substitute replay,
cached result, inferred result, or alternate-protocol fallback.

## Shadow separation

`run-forward` publishes future-only Shadow observations under its own roots.
It may accumulate diagnostic factor evidence, but it cannot create or advance
the mainline active pointer, cannot publish a
`myquant.v17.v4.mainline-public-run.v1`, and is never a
source of mainline authority.
