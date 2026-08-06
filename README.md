<div align="center">

<img src="assets/logo.svg" alt="Quant-Investor" width="520"/>

**An A-share quantitative research and portfolio-decision system.**

*The hard part of quant is not finding a signal. It is proving the signal is
real — and refusing to act when the proof is missing.*

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-17.0.0-FF6B35?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**English** · [简体中文](README.zh-CN.md)

[Thesis](#the-thesis) · [Framework](#the-framework) · [How it runs](#how-it-runs) ·
[Problems it solves](#the-problems-it-solves) · [What is distinctive](#what-is-distinctive) ·
[Quick start](#quick-start) · [Current state](#current-state)

</div>

---

## What this is

A single-operator, CN-only quantitative research system: roughly 290k lines of
Python across 427 modules, 210 test files, no broker connection and no order
authority. It takes strict point-in-time market and fundamental data, produces
three independent evidence branches, fuses them through a deterministic control
chain, and publishes exactly one governed result per strategy.

Everything in it is organized around one idea, stated below.

## The thesis

Most quant systems do not fail loudly. They fail by quietly producing a number
that looks like the number you wanted. The interesting question is not "what is
my alpha", it is **"what would have to be true for this number to be wrong, and
would I have noticed?"**

Four things go wrong, and they go wrong silently:

| Failure | What it looks like from inside |
|---|---|
| **Evidence that does not survive its own window** | A factor mined on eight months shows rank IC 0.05 and 8/8 gates passed, then decays to a quarter of that over five years. Daily rank IC against a 30-day forward return is heavily overlapping, so a naive t-test inflates significance by ≈√30 — enough to make an artifact look like an edge at p < 1e-4. |
| **Data that substitutes itself** | A missing Parquet partition becomes a CSV fallback, a stale snapshot, or an inferred value. The pipeline still returns a result, and nothing in the output says which bytes produced it. |
| **Definitions that move** | A factor defined as "residual against the current production composite" changes meaning every time the production set changes. Its recorded evidence describes a baseline that no longer exists. |
| **Language models with decision authority** | A model that can adjust a weight can hallucinate a weight. Once model output enters the control path, "why this position" has no deterministic answer. |

The organizing principle of the whole repository is the response to those four:

> **Every number carries replayable evidence of how it was produced. Absence of
> evidence is a refusal, never a default.**

That is why the system is *fail-closed* rather than *best-effort*. A best-effort
pipeline optimizes for always returning something. This one optimizes for never
returning something it cannot defend. `V17_MAINLINE_UNINITIALIZED` is a correct
answer.

## The framework

One path, four stages, and no way to reach a later stage without proving where
the work came from.

```text
             strict CN Parquet + PIT membership
                              |
                        data snapshot                  ---- data authority
                              |
                     DeterministicFunnel
                              |
              +---------------+---------------+
              |               |               |
            Quant        Fundamental        Macro      ---- evidence
              |               |               |             production
              +---------------+---------------+
                              |
                     Bayesian posterior
                              |
                          RiskGuard  <--  Markov regime
                              |           (may only reduce)
                        ICCoordinator                  ---- deterministic
                              |                             control chain
                     PortfolioConstructor
                              |
                        NarratorAgent
                              |
                    mainline run (immutable)
                              |
                     active pointer (CAS)              ---- governance and
                              |                             publication
                    public run (read-only)
```

The factor pool feeding the quant branch runs its own governed loop off to the
side, under `FactorGovernanceProtocol v4`; it is the subject of
[Factor mining](#3-factor-mining-and-governance--weekly-report-only-month-end-proposals)
below.

**Data authority.** Canonical storage is Parquet with a hash-bound
manifest, promoted through `parquet_staging -> parquet_serving` only after
validation; implausible bytes are quarantined rather than repaired in place.
Universe membership is point-in-time (`results/pit_universe/`), so a backfilled
listing cannot retroactively join a 2021 cross-section. CSV exists only for
explicit export or migration, never as a read fallback.

**Evidence production.** `DeterministicFunnel` compresses the full
A-share market to a candidate set (default cap 500) using hard data-quality,
tradability and liquidity gates plus deterministic ranking — no model, no
randomness. Three branches then research the *same* pool: `quant`,
`fundamental`, `macro`. Separately, the factor pool that feeds the quant branch
is governed by its own eight-gate protocol (see below).

**The control chain.** Branch evidence becomes a posterior, the
posterior meets hard constraints, and constraints become weights. Each stage has
a typed contract (`BranchVerdict`, `RiskDecision`, `ICDecision`,
`PortfolioPlan`, `ReportBundle`) and each stage can only tighten what the
previous one allowed.

**Governance.** A completed pipeline result is not public. It becomes
an immutable `mainline-run.v1`, and only a compare-and-swap against an expected
prevalue — followed by exact readback — moves the active pointer. Deploying code
activates nothing.

The asymmetries in the middle of that path are deliberate, and each one closes a
specific way of lying to yourself:

- **Only `quant` and `fundamental` enter the Bayesian likelihood.** `macro` is
  prior and context; it shapes the risk budget, it never votes on a symbol.
- **Fundamental evidence must prove provenance.** Likelihood is admitted only
  when the canonical generation pointer verifies and lineage binds to
  `tushare_primary`. Otherwise the branch still emits diagnostics, but its
  likelihood is neutralized to exactly `0.50` — present, and provably
  uninformative.
- **The regime model may only reduce risk.** Markov output applies as
  `min(baseline, suggested)` on target exposure and max single weight; turnover
  caps can tighten, never loosen. A regime signal cannot talk the book into a
  bigger position.
- **Branches are not switchable.** There is no `enable_quant` flag. A request
  carrying one fails rather than being silently ignored.
- **Notes are bucketed and load-bearing.** `investment_risks`,
  `coverage_notes` and `diagnostic_notes` are separate by contract, and the
  latter two are barred from reaching allocation. A provider outage cannot leak
  into a position size.

## How it runs

Four loops on different clocks. They share storage and hashes; none of them can
short-circuit another.

### 1. Data maintenance — daily, explicit

```bash
quant-investor market maintain --market CN --staged
quant-investor market storage-validate --market CN
```

Download → clean → stage → validate → promote to serving → refresh manifests and
PIT membership. This is a separate, explicitly invoked workflow: no analysis
command ever triggers a network fetch to patch a hole it found.

### 2. Decision — the DAG

```text
data_snapshot (strict Parquet health, latest session)
  -> symbol universe / PIT membership
  -> batch read
  -> DeterministicFunnel            hard gates + deterministic ranking
  -> GlobalContext                  macro, regime, industry maps
  -> per-symbol research            quant | fundamental | macro
  -> Bayesian selection             prior + 2 likelihoods in log-odds
  -> RiskGuard                      hard veto, exposure & single-name caps
  -> ICCoordinator                  consensus, conflicts, structured actions
  -> PortfolioConstructor           deterministic target weights
  -> NarratorAgent                  read-only report bundle
  -> report persistence + stage timings
```

The Bayesian step is a two-likelihood log-odds update on a hierarchical prior,
with regime-aware action thresholds (buy at 0.52 in an uptrend, 0.60 in a
downtrend) and explicit penalties for coverage gaps and degraded backends.
`RiskGuard` applies a bilingual hard-veto keyword set — A-share risk text is
Chinese, so an English-only veto list silently never fires — plus a default 15%
single-name cap. Three or more distinct risk notes cap the action at HOLD on
their own.

The optional LLM review layer sits *beside* this chain, not inside it. It emits
advisory hints only; with no API key it degrades to a handoff and the
deterministic result is bit-identical. Every weight comes from
`PortfolioConstructor`.

### 3. Factor mining and governance — weekly report-only, month-end proposals

```text
candidate generation
  -> Gates 1-8         data safety, coverage/stability, IC-RankIC, group
                       returns, cost/turnover, sector x size neutralization,
                       purged CPCV out-of-sample, portfolio increment
  -> trial correction  deflated Sharpe, PBO, non-overlapping t > 3.0,
                       effective-trial clustering
  -> family BH         q <= 0.10 within family
  -> maturity          12 month-end RankIC sessions, or 8 non-overlapping
                       30-session cohorts, on the strict-Parquet calendar
  -> canonical replay  A/B/C/D arms through the real control chain, hash-bound
  -> month-end proposal (never an apply)
```

Weekly runs cannot change a weight. Month-end may emit at most two proposals,
and at the ten-factor target every addition must be one-in-one-out with a
strictly positive 95% lower bound on its paired incremental edge.

### 4. Publication and read

```text
pipeline result -> mainline-run.v1 (immutable)
               -> CAS on mainline-active-pointer.v1 (expected prevalue + readback)
               -> mainline-public-run.v1 (read-only projection)
```

Three commands resolve that same pointer and return the same authority chain:

```bash
quant-investor research run --workspace-root /abs/path/myQuant --strategy-id <id>
quant-investor market analyze --workspace-root /abs/path/myQuant --strategy-id <id>
quant-investor market run --workspace-root /abs/path/myQuant --strategy-id <id>
```

A separate **Shadow lane** accumulates future-only evidence from
content-addressed sealed requests. Its outputs never grant mainline authority
and can never be returned as a public run:

```bash
quant-investor-v17-v4 run-forward --workspace-root /abs/path/myQuant \
  --request-path <request.json> --request-sha256 <sha256>
```

Research is separated from production by construction, not by convention.

## The problems it solves

| Classic quant failure | What this system does about it |
|---|---|
| **Look-ahead and survivorship bias** | Point-in-time universe membership; per-row `availability_date` in the fundamental generation; regime features truncate dated frames at `as_of` before computing anything. |
| **Silent data substitution** | One canonical Parquet lane with hash-bound manifests; no CSV fallback; corruption is quarantined, not patched; missing input is an explicit blocker code. |
| **Overlapping-label significance inflation** | Non-overlapping 30-session cohort t-tests against a hurdle of **3.0**, not 2.0, with cohort width inferred from the series' own sampling gap. |
| **Multiple testing / the factor zoo** | Benjamini-Hochberg within family at `q <= 0.10`, plus deflated Sharpe (floor 0.95), PBO (ceiling 0.5), and an **effective**-trial count that collapses correlated candidates before computing the best-of-N null. |
| **Backtest overfitting** | Combinatorial purged cross-validation: 10 blocks, all C(10,2)=45 paths, 30-session purge and 30-session embargo counted in sessions, content-bound evidence hash. |
| **Factor redundancy** | The ranking objective is *incremental*: the production pool is projected out of each candidate cross-section by cross-section, and candidates are ordered on residualized ICIR. A clone of `low_dollar_volume` drops from RankIC 0.12 standalone to 0.011 residual and falls 95 places. |
| **Style exposure masquerading as alpha** | Gate 6 scores neutralized ICIR inside sector × size buckets, with size as cross-sectional terciles (fixed absolute thresholds let a rising market sweep everything into `large` and neutralize nothing). `style_exposure_only` flags factors that flip direction or lose half their ICIR under demeaning. |
| **Concentration by accident** | Caps of 20% per factor and 35% per family interact arithmetically: at exactly five factors the 20% cap forces every factor to 20%, so two sharing a family would be 40%. **A five-factor set therefore requires five distinct families** — bounded by arithmetic, not judgment. |
| **Definitions that cannot be replayed** | Definitions depending on mutable registry state are refused at the production gate. Where such a definition is genuinely wanted, its baseline is *pinned*: named explicitly and bound by content hash. |
| **Pro-cyclical risk-taking** | The regime overlay can only reduce. `min(baseline, suggested)`, always. |
| **LLM hallucination in the control path** | Advisory/decision separation is structural. `NarratorAgent` is read-only; the LLM layer cannot alter candidates, risk limits or weights; the system runs unchanged with no key at all. |
| **Unauditable results** | Immutable runs, CAS-only activation with readback, append-only WAL blueprints with before/after registry hashes, a same-day activation receipt, and a decision log where an advisory and a human action must be paired by ID before any fill is even considered. |
| **A thin calendar that flatters the statistics** | Sessions come from what the active snapshot actually observed, and thin sessions are excluded: of 1,796 observed CN sessions, 858 carry a real cross-section. Cutting maturity over the rest would satisfy the arithmetic while resting on nothing. |

## What is distinctive

**Fail-closed is the architecture, not a setting.** Every public surface
resolves exactly one pointer. Missing → `V17_MAINLINE_UNINITIALIZED`, nothing
written. Invalid → `V17_MAINLINE_BLOCKED:<blocker>`. No fallback, no scan for a
recent run, no bootstrap. Mainline backtesting is not degraded, it is refused:
`V17_BACKTEST_UNAVAILABLE`.

**Content-addressed authority.** Results travel a fixed chain and each link is
bound to the bytes of the last. Activation is compare-and-swap plus exact
readback — a race or a stale expectation aborts rather than overwrites.

**Advisory AI, deterministic control.** The system is genuinely multi-agent, and
none of the agents can decide anything. The LLM layer is a narrator and a critic
with a provider-priority chain behind it; remove it entirely and the weights do
not move.

**The factor bar is set by the size of the search, not by the best result.** The
deflated Sharpe asks what the best of N worthless trials would have scored. On
the current run that number is 1.067 against a best observed ICIR of 0.724 — so
the honest conclusion is that *enumerating more candidates makes admission
harder*, and the way forward is a smaller pre-registered search, not a bigger
one. That inversion is a load-bearing design conclusion, not an inconvenience.

**Redundancy is an objective, not a rejection.** Most pipelines check redundancy
last, as a veto. Here the search is *pointed at* incremental value: rank on what
a candidate adds to the pool, and a near-duplicate never reaches the top of the
list to be vetoed in the first place.

**The system reports its own shortfalls in its own vocabulary.** When mining
returns nothing, the output distinguishes `factor_exposure_evidence_not_ready`
(plumbing) from `no_qualified_positive_candidate` (a statement about the
candidates). Those are different problems and the run says which one it hit.

**Research and production cannot be confused.** The Shadow lane is a separate
runtime, a separate storage root and a separate schema family. There is no flag
that promotes one to the other.

## Quick start

```bash
uv sync
cp .env.example .env
```

Local verification stays offline. Fill only what an explicitly authorized
workflow needs. Then run any of the read commands in
[How it runs](#4-publication-and-read).

## Current state

CN-only, with **no broker, order, execution or trade authority**. Two things are
open, and both are consequences of the design rather than unfinished cleanup.

**Factor Governance v4 is not ready.** Measured over the full open-session
calendar, three of five candidates clear `q <= 0.10`. The two that fail are
exactly the two whose recorded gate evidence came from a short recent window;
they do not survive correction for 30-day overlapping forward returns. The bar
is doing its job, so `factor_governance_ready` stays `false`.

**Gates 1, 5 and 8 have no evidence producer yet**, so they fail closed and the
best any candidate reaches is 5/8. They need, respectively: a versioned
PIT/tradability audit per candidate, a participation-rate slippage model in
place of the current flat 1bp assumption, and the full A/B/C/D replay through
the real control chain.

**There is no mainline publisher.** `quant_investor/v17_mainline/` is read-side
only. Until a decision-output producer exists, public surfaces return
`V17_MAINLINE_UNINITIALIZED` — which is the correct answer, not a bug.

## Project map

```text
quant_investor/
  cli/                       public command routing
  data/                      sources, PIT universe, processing
  market/                    CN maintenance, canonical reads, DAG executor
  funnel/                    deterministic full-market compression
  agents/                    branches, RiskGuard, IC, PortfolioConstructor, Narrator
  bayesian/                  prior, likelihood, posterior, calibration
  macro/ regime/             context and risk-tightening overlays
  factors/                   Factor Governance v4, CPCV, trial correction
  pipeline/                  research and portfolio pipeline
  v17_mainline/              active-pointer authority and public-run reader
  v17_v4_contract/           V17 v4 schemas and validators
  v17_v4_runtime/            research-only Shadow runtime
  learning/                  post-trade reflection and memory promotion
portfolio_dashboard/         read-only dashboard contracts
web/                         local research workspace API
results/v17_mainline/        governed active mainline results
results/v17_v4_shadow/       research-only forward evidence
```

## Development

Python 3.13+. Run the narrowest relevant tests first; for broad staged-upgrade
work:

```bash
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

Do not call live data, LLM, broker, order, execution or trade APIs during local
verification unless the task explicitly authorizes it.

## Documentation

- [Documentation index](docs/README.md)
- [V17 v4 mainline contract](docs/architecture/v17_v4_production_research_contract.md)
- [Research pipeline and protocols](docs/architecture/research_pipeline_and_protocols.md)
- [Factor Governance v4](docs/factor_governance_v4.md)
- [Factor mining mechanism](docs/factor_mining_mechanism.md)
- [V17 v4 operations](docs/runbooks/v17_v4_operations.md)
- [Entrypoints and versioning](docs/architecture/entrypoints_and_versioning.md)
- [Forward-evidence runtime](docs/architecture/v17_v4_forward_evidence_runtime.md)
- [Module map](docs/modules/module_map.md)
- [Agent guide](AGENTS.md)

## License

[MIT](LICENSE) © 2024 alpha-mw
