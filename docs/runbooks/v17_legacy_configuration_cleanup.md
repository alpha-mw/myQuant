# V17 Legacy Configuration Cleanup

This migration removes environment variables and Python attributes that looked
like V17 controls but had no current runtime consumer. The public V17 surfaces
resolve one exact active pointer; they do not build a pipeline, activate an
overlay, or apply a production Markov risk model from process-wide settings.

## Retired environment keys

The following keys are rejected explicitly. Remove them from `.env`, shell
profiles, launch agents, and automation wrappers before importing
`quant_investor.config`:

- `PIPELINE_MODE`
- `DECISION_ENGINE`
- `TOTAL_TIMEOUT_SECONDS`
- `FUNDAMENTAL_RESEARCH_OVERLAY_MODE`
- `FUNDAMENTAL_RESEARCH_ROOT`
- `FUNDAMENTAL_RESEARCH_ACTIVATION_PATH`
- `FUNDAMENTAL_RESEARCH_ACTIVATION_EXPECTED_SHA256`
- `RISK_GUARD_SINGLE_NAME_WEIGHT_CAP`
- `MARKOV_REGIME_ENABLED`
- `MARKOV_REGIME_EXECUTION_TARGET`
- `MARKOV_REGIME_HISTORY_PATH`
- `MARKOV_REGIME_PERSIST_ENABLED`
- `MARKOV_REGIME_MIN_MARKET_SAMPLE`
- `MARKOV_REGIME_MAX_REFERENCE_SYMBOLS`
- `MARKOV_REGIME_REFERENCE_UNIVERSE_CN`
- `MARKOV_REGIME_REFERENCE_UNIVERSE_US`

There is no replacement environment switch:

- Mainline authority comes only from the exact governed strategy pointer and
  its immutable transitive closure.
- Fundamental evidence is admitted through hash-bound V17 observation and
  generation artifacts, not a global overlay activation path.
- I0 regime inference receives an explicit content-addressed `RegimeInput`,
  performs one causal forward filter step, and has no production, persistence,
  risk, or portfolio authority.
- Risk and portfolio limits belong to a governed decision artifact produced by
  an upstream producer. This repository does not currently expose a mainline
  publisher or activation command.

## Preserved standalone automation settings

`FUNNEL_*`, `BAYESIAN_SHORTLIST_SIZE`, `DEFAULT_AGENT_TIMEOUT_SECONDS`, and
`DEFAULT_MASTER_TIMEOUT_SECONDS` remain because the restored standalone
automation modules consume them. That automation is an incomplete legacy lane:
it has no public V17 entrypoint, is not the I0/R2.2 research loop, and does not
grant mainline authority.

## Removed Python surface

`quant_investor.regime_detector` was an unreferenced single-layer heuristic that
mixed state detection with position caps, stop-loss settings, rebalance
frequency, and branch-weight adjustments. It is intentionally removed rather
than retained as a compatibility shim. Current research-only regime code lives
under `quant_investor.intelligence.regime` and has a different, no-authority
contract.
