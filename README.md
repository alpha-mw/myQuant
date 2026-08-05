# Quant-Investor

Quant-Investor is a fail-closed A-share research and portfolio-decision system.
V17 v4 is the only supported decision mainline. Public result surfaces resolve
one exact per-strategy active pointer; they do not choose a protocol or scan
result directories for a recent run.

## Core contract

```text
strict CN Parquet + PIT membership
  -> Quant cross-sectional preselection
  -> same-pool Quant + Fundamental evidence
  -> Macro / regime context
  -> deterministic risk and portfolio gates
  -> immutable mainline run
  -> governed active pointer
  -> read-only public run
```

The three authority artifacts are:

- `myquant.v17.v4.mainline-run.v1`: immutable closed run;
- `myquant.v17.v4.mainline-active-pointer.v1`: the sole public authority for a
  canonical strategy;
- `myquant.v17.v4.mainline-public-run.v1`: a read-only public projection.

Mainline state is stored under:

```text
results/v17_mainline/strategies/<strategy-id>/
  _active.json
  runs/<run-id>/run.json
```

The runtime is currently CN-only. It has no broker, order, execution, or trade
authority. Review-model output is advisory; deterministic data, Factor, risk,
portfolio, and readiness gates remain authoritative.

## Quick start

### Install

```bash
uv sync
cp .env.example .env
```

Fill only the credentials and paths needed for the explicitly authorized
workflow. Local verification should remain offline.

### Read the active strategy result

All three commands resolve the same active pointer and return the same public
authority chain:

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

If `_active.json` is absent, the result is
`V17_MAINLINE_UNINITIALIZED` and the command writes nothing. An invalid pointer
or run returns `V17_MAINLINE_BLOCKED:<blocker>` with no fallback and no writes.

Mainline backtesting is not supported. `quant-investor market backtest` fails
closed with `V17_BACKTEST_UNAVAILABLE` and writes nothing.

### Maintain CN data

Maintenance is an explicit workflow separate from public result resolution:

```bash
quant-investor market maintain \
  --market CN \
  --staged

quant-investor market storage-validate \
  --market CN
```

Strict Parquet canonical data and exact pointer/manifest readback are required.
The runtime does not silently substitute CSV, cached, mock, inferred, or stale
data.

## Forward Shadow evidence

`run-forward` is a separate V17 v4 Shadow research lane. It accumulates
future-only factor and model observations from one content-addressed sealed
request:

```bash
quant-investor-v17-v4 run-forward \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/v17_v4_runs/forward_requests/<request_id>.json \
  --request-sha256 <sha256>
```

Create requests with the canonical library helper; do not hand-edit IDs or
hashes. Only the final immutable session reference marks completion. Shadow
outputs under `results/v17_v4_shadow/` never grant mainline authority, never
advance `results/v17_mainline/.../_active.json`, and cannot be returned as a
public run.

## Code rollout versus operational activation

Merging, installing, or deploying V17 v4 code does not activate a strategy.
Activation separately requires a validated immutable mainline run followed by
an expected-prevalue pointer CAS and exact readback. Public readers never
bootstrap a missing pointer.

## Project map

```text
quant_investor/
  cli/                       public command routing
  market/                    CN maintenance and market workflows
  pipeline/                  research and portfolio pipeline
  v17_mainline/              active-pointer authority and public-run reader
  v17_v4_contract/           V17 v4 schemas and validators
  v17_v4_runtime/            research-only Shadow runtime services
  factors/                   Factor Governance
portfolio_dashboard/         read-only dashboard contracts
web/                         local research workspace API
docs/                        architecture and runbooks
results/v17_mainline/        governed active mainline results
results/v17_v4_shadow/       research-only forward evidence
```

## Development

Run the narrowest relevant tests first. For broad staged-upgrade work:

```bash
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

The repository is Python 3.13+. Do not call live data, LLM, broker, order,
execution, or trade APIs during local verification unless the task explicitly
authorizes that action.

## Documentation

- [Documentation index](docs/README.md)
- [V17 v4 mainline contract](docs/architecture/v17_v4_production_research_contract.md)
- [V17 v4 operations](docs/runbooks/v17_v4_operations.md)
- [Entrypoints and versioning](docs/architecture/entrypoints_and_versioning.md)
- [Forward-evidence runtime](docs/architecture/v17_v4_forward_evidence_runtime.md)
- [Factor Governance v4](docs/factor_governance_v4.md)

## License

MIT
