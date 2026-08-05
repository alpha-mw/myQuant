# Entrypoints and Versioning

## One protocol

The only supported decision protocol is `myquant.v17.v4`. The package version,
artifact schema version, and operational activation state are separate facts:

- the package version identifies a software release;
- `myquant.v17.v4.*.v1` identifies an artifact contract;
- the strategy active pointer identifies the only public run.

There is no runtime protocol selector and no fallback to another protocol.

## Public entrypoints

All public result entrypoints are read-only projections of the same active
pointer and currently support `CN` only:

```bash
quant-investor research run \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>

quant-investor market analyze \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>

quant-investor market run \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>
```

The Web result surface is `GET /api/research/{strategy_id}`. It may receive
`expected_pointer_sha256` to require an exact pointer generation.

`market backtest` is not implemented for the V17 v4 mainline. It fails closed
with `V17_BACKTEST_UNAVAILABLE` and performs no writes.

## Fail-closed resolution

The only pointer is:

```text
results/v17_mainline/strategies/{strategy_id}/_active.json
```

Its run reference must resolve exactly to:

```text
results/v17_mainline/strategies/{strategy_id}/runs/{run_id}/run.json
```

Readers verify the canonical strategy ID, schema, byte and semantic hashes,
path containment, market, and transitive run closure. They do not scan `runs/`
for a latest file.

- Missing pointer: `V17_MAINLINE_UNINITIALIZED`, no writes.
- Invalid pointer or run: `V17_MAINLINE_BLOCKED:<blocker>`, no writes.
- Valid closed pointer: return a read-only
  `myquant.v17.v4.mainline-public-run.v1` projection.

## Research-only entrypoint

`quant-investor-v17-v4 run-forward` remains a Shadow observation command. Its
artifacts live outside `results/v17_mainline/`, are not read by public result
surfaces, and never grant mainline authority. See the
[forward-evidence contract](v17_v4_forward_evidence_runtime.md).
