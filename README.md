<div align="center">

<img src="assets/logo.svg" alt="Quant-Investor" width="520"/>

**A fail-closed A-share research and portfolio-decision system.**

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-17.0.0-FF6B35?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

[Core contract](#core-contract) · [Quick start](#quick-start) · [Current state](#current-state) · [Project map](#project-map) · [Docs](#documentation)

</div>

---

V17 v4 is the only supported decision mainline. Public result surfaces resolve one
exact per-strategy active pointer; they do not choose a protocol and do not scan
result directories for a recent run.

The runtime is CN-only. It has **no broker, order, execution, or trade authority**.
Review-model output is advisory; deterministic data, Factor, risk, portfolio, and
readiness gates remain authoritative.

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

Three authority artifacts carry that chain:

| Artifact | Role |
|---|---|
| `myquant.v17.v4.mainline-run.v1` | immutable closed run |
| `myquant.v17.v4.mainline-active-pointer.v1` | sole public authority for a canonical strategy |
| `myquant.v17.v4.mainline-public-run.v1` | read-only public projection |

Mainline state lives under:

```text
results/v17_mainline/strategies/<strategy-id>/
  _active.json
  runs/<run-id>/run.json
```

### What fails closed

| Condition | Result |
|---|---|
| `_active.json` absent | `V17_MAINLINE_UNINITIALIZED`, nothing written |
| Invalid pointer or run | `V17_MAINLINE_BLOCKED:<blocker>`, no fallback, nothing written |
| `market backtest` invoked | `V17_BACKTEST_UNAVAILABLE`, nothing written |

There is no fallback path. Public readers never bootstrap a missing pointer, and
the runtime never silently substitutes CSV, cached, mock, inferred, or stale data.

## Quick start

```bash
uv sync
cp .env.example .env
```

Fill only the credentials and paths the explicitly authorized workflow needs.
Local verification should remain offline.

### Read the active strategy result

All three commands resolve the same active pointer and return the same public
authority chain:

```bash
quant-investor research run --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
```

```bash
quant-investor market analyze --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
```

```bash
quant-investor market run --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
```

### Maintain CN data

Maintenance is an explicit workflow, separate from public result resolution:

```bash
quant-investor market maintain --market CN --staged
```

```bash
quant-investor market storage-validate --market CN
```

Strict Parquet canonical data and exact pointer/manifest readback are required.

### Forward Shadow evidence

`run-forward` is a separate V17 v4 Shadow research lane. It accumulates
future-only factor and model observations from one content-addressed sealed
request:

```bash
quant-investor-v17-v4 run-forward --workspace-root /absolute/path/to/myQuant --request-path data/private/v17_v4_runs/forward_requests/<request_id>.json --request-sha256 <sha256>
```

Create requests with the canonical library helper; do not hand-edit IDs or hashes.
Only the final immutable session reference marks completion. Shadow outputs under
`results/v17_v4_shadow/` never grant mainline authority, never advance
`_active.json`, and cannot be returned as a public run.

Other subcommands: `verify`, `status`, `factor-set-status`, `deep-v3-compile`,
`forward-shadow-readiness`, `forward-shadow-status`.

## Current state

Merging, installing, or deploying V17 v4 code does not activate a strategy.
Activation separately requires a validated immutable mainline run followed by an
expected-prevalue pointer CAS and exact readback.

Two things are open, and both are deliberate rather than pending cleanup.

**Factor Governance v4 is not ready.** It requires five production factors, each
with `bh_q_value <= 0.10` under Benjamini-Hochberg within family. Measured over the
full open-session calendar, three of five candidates clear that bar. The two that
do not are the ones whose recorded gate evidence came from a short recent window;
they fail once RankIC significance is corrected for 30-day overlapping forward
returns. `factor_governance_ready` stays `false` until the factor pool can support
it. See [Factor Governance v4](docs/factor_governance_v4.md).

**There is no mainline publisher.** `quant_investor/v17_mainline/` is read-side
only, by design. Until a decision-output producer exists, public surfaces return
`V17_MAINLINE_UNINITIALIZED`.

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

Python 3.13+. Run the narrowest relevant tests first; for broad staged-upgrade work:

```bash
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

Do not call live data, LLM, broker, order, execution, or trade APIs during local
verification unless the task explicitly authorizes it.

## Documentation

- [Documentation index](docs/README.md)
- [V17 v4 mainline contract](docs/architecture/v17_v4_production_research_contract.md)
- [V17 v4 operations](docs/runbooks/v17_v4_operations.md)
- [Entrypoints and versioning](docs/architecture/entrypoints_and_versioning.md)
- [Research pipeline and protocols](docs/architecture/research_pipeline_and_protocols.md)
- [Forward-evidence runtime](docs/architecture/v17_v4_forward_evidence_runtime.md)
- [Factor Governance v4](docs/factor_governance_v4.md)
- [Module map](docs/modules/module_map.md)
- [Agent guide](AGENTS.md)

## License

[MIT](LICENSE) © 2024 alpha-mw
