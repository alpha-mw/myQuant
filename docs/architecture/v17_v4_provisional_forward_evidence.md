# V17 v4 Provisional Forward Evidence

`PROVISIONAL_FORWARD_EVIDENCE` is an explicit research-only runtime for
immutable, replayable inputs that do not yet satisfy V4 Source Truth. It does
not relax causal integrity: exact bytes, semantic identity, PIT availability,
session and factor implementation bindings, safe paths, and authority ceilings
remain hard gates.

The profile deliberately separates two facts:

- `research_runtime_eligible=true` and `research_evaluation_eligible=true`;
- `production_governance_eligible=false`, with formal activation, promotion,
  Factor Governance writes, selector, execution, broker, order, and trade all
  disabled.

Its security identity is run-scoped. The provisional key hashes exchange
namespace, PIT ticker, the exact listing-interval reference, and the immutable
source snapshot. It is not a permanent `security_id`, and separate listing
intervals or ticker reuse are never merged.

## Frozen source boundary

The immutable request explicitly names every input path and SHA. At run start,
the runtime freezes each input's verified bytes and seals a source snapshot containing the
market pointer and manifest, PIT membership pointer and manifest, research
universe, factor set, Quant input, and any optional context. All inputs must
have `available_at <= cutoff`. A mutable pointer that later changes produces
`CURRENT_POINTER_CHANGED_DURING_RUN`; it does not rewrite the frozen snapshot
or invalidate observations already produced from it.

Missing trusted authority root, population census, governed Security Master,
corporate-action closure, Regime, Fundamental, Industry, Theme, holdings, or
Deep evidence is recorded in the immutable limitation receipt. These gaps do
not block the Quant Core. No provider or LLM fallback is permitted.

## Observation and variants

Quant scoring reuses the sealed V4 forward scoring implementation: type-7
winsorization, robust normalization, and sequential neutralization. Each
eligible security remains in the observation even when an individual factor
is missing. Missing values remain typed missing evidence and are never filled
with zero. Available factor/family counts, coverage, renormalized weight,
coverage penalty, contribution, and exact implementation/source/universe refs
are retained.

The three variant artifacts are independent:

1. `v17-quant-core` is `COMPLETE` when Quant completes.
2. `v17-quant-plus-industry` is `COMPLETE` with exact Industry evidence and
   `PARTIAL` without it.
3. `v17-quant-plus-industry-theme` is `COMPLETE` with both contexts, `PARTIAL`
   with one, and `UNAVAILABLE` with neither.

Downstream failure cannot delete an already published Quant observation. The
run manifest is published last after exact readback replay.

## Labels and evaluation

The label contract fixes horizons 1/5/10/20/60. A new origin begins with
`PENDING` labels. Matured labels must bind an exact sealed calendar ref and the
complete ordered future-session sequence for the selected horizon. Every
return dimension is independently `AVAILABLE` or `UNAVAILABLE`; unavailable
Industry, execution-price, or cost inputs are not fabricated. Historical
causal backfill is permanently disabled.

Factor, branch, and variant-comparison evaluation receipts are eligible only
for research evaluation. Regime-conditioned evidence remains independently
`UNAVAILABLE`; ordinary exposures, labels, RankIC, ICIR, quantile spread,
coverage, turnover, neutralized alpha, and cost-adjusted research metrics can
accumulate without Regime evidence.
