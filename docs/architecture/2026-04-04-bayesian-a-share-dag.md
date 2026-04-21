# Bayesian A-Share DAG

## Active Execution Graph

All supported public entrypoints now converge on one primary internal graph:

1. `GlobalContext`
2. `Deterministic Funnel`
3. `Candidate Review Layer`
4. `Bayesian Decision Layer`
5. `Master Shortlist Discussion`
6. `Deterministic Portfolio Decision`
7. `Reporting / Persistence`

Public entrypoints kept stable where practical:

- `daily_runner.py`
- CLI in `quant_investor/cli/main.py`
- Web services in `web/services/research_runner.py`, `web/services/analysis_service.py`, and `web/tasks/run_analysis_job.py`

## Boundary Rules

- `full_a` remains a logical universe, not a physical directory contract.
- LLM layers do not scan the full market.
- Fundamental and intelligence evidence run only after deterministic compression.
- Master discussion sees shortlist evidence packs only.
- Final exposure and weights remain deterministic and are owned by portfolio/risk logic.
- `symbol` and `company_name` are canonical structured fields across run artifacts.

## Global Context

`GlobalContext` is built once per run and carries:

- universe and symbol-name mapping
- CN freshness targets and selected effective trade date
- data-quality diagnostics and quarantine summary
- macro regime and risk budget
- liquidity and tradability filters
- style exposure snapshot
- provider capability map for DeepSeek, OpenAI, Kronos, and Chronos

`GlobalContext` must not contain final recommendations or final weights.

## CN Freshness And Completeness

- Stable mode is the default via `CN_FRESHNESS_MODE=stable`.
- Strict same-day freshness only upgrades when current-scope coverage passes threshold.
- Early-stop logic aborts same-day loops when sampled symbols are mostly `stale_cached`.
- Completeness and download paths share one CN symbol-local-status evaluator and one CSV reader/quarantine path.

## Degradation Model

The system degrades explicitly instead of failing late inside symbol loops:

- provider health is probed before large execution
- Kline mode is reported exactly as `hybrid`, `kronos_only_degraded`, `chronos_only_degraded`, or `statistical_only_fallback`
- data-quality penalties propagate into Bayesian scoring
- portfolio decisions remain deterministic even if evidence layers are degraded

## Compatibility Notes

- `QuantInvestor.run()` is now a thin adapter over the market DAG.
- `run_market_analysis()` synthesizes compatibility payloads from DAG artifacts for CLI/Web consumers.
- Legacy batch-centric internals may still exist as compatibility shims, but internal callers should route through the DAG path.
