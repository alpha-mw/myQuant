<div align="center">

<img src="assets/logo.svg" alt="Quant-Investor" width="520"/>

**A fail-closed A-share research and portfolio-decision system.**

*Deterministic control, advisory AI, governed results — nothing reaches a
decision without evidence that can be replayed.*

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-17.0.0-FF6B35?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

[Problem](#the-problem) · [Design stance](#design-stance) · [Architecture](#architecture) · [Factor governance](#factor-governance) · [Quick start](#quick-start) · [Current state](#current-state)

</div>

---

## The problem

Most quant systems do not fail loudly. They fail by quietly producing a number
that looks like the number you wanted. Quant-Investor is built around four
specific ways that happens.

**Evidence that does not survive its own window.** A factor mined on eight
months of data can show a rank IC of 0.05 and eight passing gates, then decay to
a quarter of that over five years. Daily rank IC against a 30-day forward return
is heavily overlapping, so a naive t-test inflates significance by roughly
√30 — enough to make an artifact look like an edge at p < 1e-4.

**Data that substitutes itself.** A missing Parquet partition becomes a CSV
fallback, a stale snapshot, or an inferred value. The pipeline still returns a
result, and nothing in the output says which bytes produced it.

**Definitions that move.** A factor defined as "residualized against the current
production composite" changes meaning every time the production set changes.
Its recorded evidence describes a baseline that no longer exists, and it cannot
be replayed from market data alone.

**Language models with decision authority.** A model that can adjust a weight
can hallucinate a weight. Once model output is mixed into the control path,
there is no longer a deterministic answer to "why this position".

## Design stance

The system's answers to those four, in order.

**Missing authority is an unavailable result, not permission to create one.**
Every public surface resolves exactly one pointer. If it is absent, the answer
is `V17_MAINLINE_UNINITIALIZED` and nothing is written. If it is invalid, the
answer is `V17_MAINLINE_BLOCKED:<blocker>` — no fallback, no scan for a recent
run, no bootstrap. Backtesting on the mainline is not degraded, it is refused:
`V17_BACKTEST_UNAVAILABLE`.

**Authority is content-addressed and moves by CAS.** Results travel a fixed
chain, and each link is bound to the bytes of the last:

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

| Artifact | Role |
|---|---|
| `myquant.v17.v4.mainline-run.v1` | immutable closed run |
| `myquant.v17.v4.mainline-active-pointer.v1` | sole public authority for a strategy |
| `myquant.v17.v4.mainline-public-run.v1` | read-only public projection |

Activation is a compare-and-swap against an expected prevalue, followed by exact
readback. Deploying code does not activate anything.

**A factor must be replayable from market data plus its own spec.** Definitions
that depend on mutable registry state are refused at the production gate — not
because they score badly, but because their value cannot be reproduced. Where
such a definition is genuinely wanted, its baseline is *pinned*: named
explicitly and bound by a content hash, so the arithmetic stays identical while
the input stops moving.

**The model narrates; it never decides.** `NarratorAgent` is read-only and
cannot alter candidates, risk limits, or weights. The optional LLM review layer
emits advisory hints only; with no key it degrades to a handoff and the
deterministic control chain is unchanged. Every weight comes from
`PortfolioConstructor`.

## Architecture

One deterministic DAG, three evidence branches, one control chain:

```text
snapshot -> DeterministicFunnel -> {quant, fundamental, macro}
  -> Bayesian selection -> RiskGuard -> ICCoordinator
  -> PortfolioConstructor -> NarratorAgent
```

The asymmetries in that line are deliberate:

- **Only `quant` and `fundamental` enter the Bayesian likelihood.** `macro` is
  prior and context — it shapes the risk budget, it does not vote on a symbol.
- **Fundamental evidence must prove its provenance.** Likelihood is admitted
  only when the canonical generation pointer verifies and lineage binds to
  `tushare_primary`. Otherwise the branch still emits diagnostics, but its
  likelihood is neutralized to exactly `0.50` — present, and provably
  uninformative.
- **The regime model may only reduce risk.** Markov output is applied as
  `min(baseline, suggested)` on target exposure and max single weight; turnover
  caps can tighten but never loosen. A regime signal cannot talk the book into a
  larger position.
- **Branches are not switchable.** There is no `enable_quant` flag. Requests
  carrying one fail rather than being silently ignored.
- **Notes are bucketed and load-bearing.** `investment_risks`,
  `coverage_notes`, and `diagnostic_notes` are separate by contract, and the
  latter two are barred from reaching allocation. A provider outage cannot leak
  into a position size.

Research is separated from production by construction, not by convention. The
Shadow lane (`quant-investor-v17-v4 run-forward`) accumulates future-only
evidence from content-addressed sealed requests. Its outputs never grant
mainline authority and can never be returned as a public run.

## Factor governance

No factor reaches production by being interesting. `FactorGovernanceProtocol v4`
holds a target set of **ten** healthy factors; five to nine is an underfilled
state that keeps mining in accelerated report-only mode, and below five the
system is in `no_new_risk`.

A factor is healthy only when all of these hold together:

| Requirement | Bar |
|---|---|
| Gates 1–8 | data safety, coverage/stability, IC-RankIC, group returns, cost/turnover, neutralization, out-of-sample robustness, portfolio increment |
| Maturity | 12 actual month-end RankIC sessions, or 8 non-overlapping 30-session cohorts, against a strict-Parquet calendar |
| Multiplicity | Benjamini-Hochberg **within family**, `q <= 0.10` |
| Identity | verified runtime contract and replay hash, unique name and slot |
| Health | fresh, not stale, not data-blocked |
| Allocation | ≤ 20% per factor, ≤ 35% per family |

Those last two caps interact in a way worth stating: at exactly five factors the
20% cap forces every factor to 20%, so two factors sharing a family would be
40% — over the 35% limit. **A five-factor set therefore mathematically requires
five distinct families.** Concentration is bounded by arithmetic, not by
judgment.

The calendar matters as much as the statistics. Sessions are derived from what
the active snapshot actually observed, and thin sessions are excluded — of 1,796
observed CN sessions, 858 carry a real cross-section. Maturity cut over the rest
would satisfy the arithmetic while resting on nothing.

## Quick start

```bash
uv sync
cp .env.example .env
```

Local verification stays offline. Fill only what an explicitly authorized
workflow needs.

Three commands resolve the same active pointer and return the same authority
chain:

```bash
quant-investor research run --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
quant-investor market analyze --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
quant-investor market run --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
```

Data maintenance is a separate, explicit workflow:

```bash
quant-investor market maintain --market CN --staged
quant-investor market storage-validate --market CN
```

The Shadow research lane, and its other subcommands (`verify`, `status`,
`factor-set-status`, `deep-v3-compile`, `forward-shadow-readiness`,
`forward-shadow-status`):

```bash
quant-investor-v17-v4 run-forward --workspace-root /absolute/path/to/myQuant --request-path <request.json> --request-sha256 <sha256>
```

## Current state

The runtime is CN-only and has **no broker, order, execution, or trade
authority**. Two things are open, and both are consequences of the design rather
than unfinished cleanup.

**Factor Governance v4 is not ready.** Measured over the full open-session
calendar, three of five candidates clear `q <= 0.10`. The two that fail are
exactly the two whose recorded gate evidence came from a short recent window;
they do not survive correction for 30-day overlapping forward returns. The bar
is doing its job, so `factor_governance_ready` stays `false` until the factor
pool can support it.

**There is no mainline publisher.** `quant_investor/v17_mainline/` is read-side
only and says so in its own module docstring. Until a decision-output producer
exists, public surfaces return `V17_MAINLINE_UNINITIALIZED` — which is the
correct answer, not a bug.

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

Python 3.13+. Run the narrowest relevant tests first; for broad staged-upgrade
work:

```bash
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

Do not call live data, LLM, broker, order, execution, or trade APIs during local
verification unless the task explicitly authorizes it.

## Documentation

- [Documentation index](docs/README.md)
- [V17 v4 mainline contract](docs/architecture/v17_v4_production_research_contract.md)
- [Research pipeline and protocols](docs/architecture/research_pipeline_and_protocols.md)
- [Factor Governance v4](docs/factor_governance_v4.md)
- [V17 v4 operations](docs/runbooks/v17_v4_operations.md)
- [Entrypoints and versioning](docs/architecture/entrypoints_and_versioning.md)
- [Forward-evidence runtime](docs/architecture/v17_v4_forward_evidence_runtime.md)
- [Module map](docs/modules/module_map.md)
- [Agent guide](AGENTS.md)

## License

[MIT](LICENSE) © 2024 alpha-mw
