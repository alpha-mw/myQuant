# Fundamental Research v14 Runbook

## Purpose

This workflow upgrades the CN fundamental branch with evidence-backed company
research performed by an external Codex session. The production DAG remains
offline: it exports a hash-bound request, imports an untrusted structured
response, validates it locally, and may apply at most one bounded fundamental
overlay in a later run.

The deterministic control chain remains authoritative:

`PIT facts -> fundamental base -> dossier overlay -> Bayesian -> RiskGuard -> IC -> PortfolioConstructor`

The active v14 branch contract remains exactly `quant`, `fundamental`, and
`macro`; the dossier changes only the Fundamental verdict. Counterfactual
artifacts reject missing or extra branches and any retired Intelligence key.

## Public workflow

```bash
quant-investor market fundamental-research-prepare \
  --market CN \
  --as-of 2026-07-14T15:00:00+08:00 \
  --analysis-run results/cn_analysis_full/<run> \
  --holdings-manifest results/strategy_records/CN/aggressive_tech_manufacturing/<run>/manual_execution_manifest.json

quant-investor market fundamental-research-import \
  --request <symbol>.request.v1.json \
  --response <symbol>.response.v1.json

quant-investor market fundamental-research-status --market CN

quant-investor market fundamental-research-target-weight-produce \
  --request <symbol>.request.v1.json \
  --dossier-id <dossier-id> \
  --actual-analysis <run>/analysis_run_manifest.v1.json \
  --counterfactual-analysis <run>/analysis_run_manifest.without_dossier.v1.json

quant-investor market fundamental-research-nav-produce \
  --target-weight-observation <observation.v1.json> \
  --attribution-date 2026-07-15 \
  --data-root data

quant-investor market fundamental-research-gate-evidence \
  --holdings-manifest results/strategy_records/CN/aggressive_tech_manufacturing/<run>/manual_execution_manifest.json
```

`prepare` selects the union of the explicit run's actual Top-K and the symbols
in the explicit manifest-selected ledger. It exports no capital, cost basis,
position weight, credential, other-account data, or absolute path. The external
Codex session may browse, but the local command never calls a provider or LLM.

## Research contract

Each request fixes six dimensions and their weights:

- financial and accounting quality: 25%
- business model and unit economics: 15%
- industry structure and value chain: 15%
- product competitiveness, moat, and pricing power: 20%
- management and capital allocation: 10%
- valuation and scenarios: 15%

Per-symbol research limits are 60 minutes, 20 searches, and 25 deduplicated
documents. These are external-worker budgets and telemetry, not evidence
quality gates.

Every non-zero dimension requires at least one eligible primary claim. Primary
sources are exchange/regulator filings, statutory disclosures, company IR,
government or industry-association publications, and competitor statutory
disclosures. Secondary research may corroborate, identify conflicts, or add a
risk flag, but cannot create a non-zero contribution by itself.

The local PIT core fixes industry classification, peer set, and valuation
price. The external worker cannot replace them with web-selected peers or web
prices. Unresolved primary-source conflict makes the dimension neutral and
keeps a visible risk flag.

## Time and application rules

- A scoring source must have `first_available_at <= decision_cutoff`.
- Every scoring claim must also be effective at the cutoff (`valid_from` /
  `valid_until`) and all of its supporting sources must be locally eligible and
  available by that cutoff. A historical neutral claim cannot make a separate
  future directional claim scoreable.
- A validated dossier can first be used by the next decision cutoff after its
  import; it never changes a completed run.
- The maximum TTL is 30 calendar days. A newer local disclosure identity/hash
  immediately supersedes the dossier.
- Historical replay can consume only a dossier that was already imported and
  available at that historical cutoff.
- One symbol consumes at most one dossier per run, and the delta is always
  computed from the request-bound deterministic base score.
- A dossier overlay replaces the generic fundamental LLM overlay. The two must
  never stack.

## Scoring and modes

Codex reports a discrete dimension signal only:

`strong_negative=-1`, `negative=-0.5`, `neutral=0`, `positive=0.5`,
`strong_positive=1`, or `unknown`.

The local importer computes the weighted signal and clamps its score effect to
the active cap. Positive effects require at least four qualified dimensions,
including financial quality, competitiveness/pricing power, and valuation.
Verified primary negative evidence may contribute without that positive
coverage gate. Fundamental confidence is unchanged in v1.

`FUNDAMENTAL_RESEARCH_OVERLAY_MODE` is `off`, `shadow`, `limited`, or
`production`; the default is `shadow`. External responses cannot change it.
Invalid, expired, superseded, replayed, or lineage-mismatched dossiers always
produce an exact zero delta.

`limited` and `production` additionally require a mode-matching activation
manifest whose bytes match
`FUNDAMENTAL_RESEARCH_ACTIVATION_EXPECTED_SHA256`. The manifest must record a
separate `approved_by=maxwell` confirmation and bind a contained, non-symlink,
mode-`0600` gate-evidence file by SHA-256. Limited admission requires at least
10 shadow trading days, 30 validated dossiers, 10 companies, three industries,
holdings coverage and zero critical errors. Production requires at least 20
shadow days, 60 dossiers, 20 companies, three industries, 10 limited days,
holdings coverage and zero critical errors. Self-asserting a pass flag without
matching evidence is rejected.
Gate evidence v2 is not a caller-supplied count sheet. The generator binds a
sanitized holdings-scope snapshot to repository-relative paths and hashes for
the canonical `manual_execution_manifest.json` and Parquet manual ledger,
then recomputes validated dossiers, companies, industries, shadow/limited
dates, target-weight counterfactual dates, NAV-attribution dates, recent import
success, and critical errors from the private job, application, and
longitudinal hash-chain ledgers. Every rebuild re-reads both canonical holdings
files, requires the exact manual-execution v2 schema and an eligible producer
status, requires a timezone-aware timestamp no later than the current gate
cutoff and no more than seven days old, and
recomputes positive-share symbols from Parquet. Source drift fails closed.
The separately approved activation manifest must repeat both the source manual
manifest SHA-256 and canonical Parquet ledger SHA-256, so approval is bound to
the same holdings scope used by the evidence generator.
Limited phase 1 is restricted to limited days 0--4 and an absolute cap of
`0.03`; phase 2 requires at least five completed limited days and permits at
most `0.05`. Production requires at least 10 completed limited days and is the
only phase that may declare the `0.10` cap.

Limited additionally requires at least 10 target-weight counterfactual dates.
Production requires at least 20 target-weight dates and 10 realized NAV
attribution dates. Each longitudinal event must match a real application by
request, dossier, run key and trading date. It binds separate mode-`0600`
actual and counterfactual outcomes plus original self-hashed analysis-run
manifests or canonical NAV snapshots by SHA-256, common generation, timestamp,
metric type and value. NAV attribution requires a limited/production applied
event; shadow/off/skipped events cannot satisfy it.
Every gate rebuild revalidates those files; deletion or drift removes the date
and records a critical error. Event-id collisions and duplicate logical
observations are rejected.

The DAG now runs the alternative dossier/no-dossier shortlist through the same
`RiskGuard -> IC -> PortfolioConstructor` chain and writes an immutable
companion analysis manifest beside the actual manifest. The replay rebuilds
the alternative fundamental verdicts, branch summaries and Bayesian records,
uses no actual-run IC or Portfolio Master hints, and persists the independently
computed RiskGuard, IC, portfolio-plan and portfolio-decision payloads. The
companion is emitted for CN only; US analysis never consumes an overlay or
emits a fundamental-research companion. The target-weight
producer accepts only that self-hashed pair and an eligible application event.
The NAV producer then rolls both portfolios through the next realized session
using strict canonical Parquet bars, writes self-hashed canonical NAV
snapshots, and appends the observation. Hand-authored values, missing bars,
same-day attribution, generation drift, or an ineligible application fail
closed.
If the latest 20 received jobs contain at least 10 samples and validation
success falls below 80%, a requested limited/production mode automatically
degrades to shadow; it does not apply a score delta.
Every limited/production consumption also rebuilds current readiness and
reapplies holdings coverage plus all mode thresholds. A later supersede,
artifact drift, coverage loss or threshold regression therefore fails closed
even when the originally approved evidence remains authentic.

Date-only CN runs use 15:00 Asia/Shanghai as the decision cutoff. A dossier is
usable only when its request cutoff and its `VALIDATED` job event are both
strictly earlier, its fundamental data generation still matches, and the same
request/dossier has not already been evaluated for that run key. Shadow writes
only a counterfactual application event. Limited/production replace the
fundamental verdict and suppress the generic fundamental LLM overlay so the two
deltas cannot stack.
Import first binds the request path, request SHA and task SHA to the immutable
run `manifest.v1.json`; a schema-valid hand-written request in a canonical-looking
directory is rejected before any `RECEIVED` ledger event is appended.

For every eligible symbol, the Bayesian stage also records the alternate
with-dossier/no-dossier posterior, rank, shortlist membership, and pre-control
suggested weight without mutating the authoritative branch result. Final target
weight and realized NAV attribution are longitudinal governance metrics: they
must be derived from these hash-bound run artifacts and the later canonical
portfolio/ledger snapshots, never estimated inside the current decision run.

## Private artifacts and incident response

Artifacts live under ignored `results/fundamental_research/`; the root is mode
`0700`. Request,
response, dossier, import report, overlay report, job ledger, and application
ledger are mode `0600`, atomic, symlink-rejecting, hash-bound, and read back
after writes. Each `VALIDATED` job event binds the canonical request, response,
dossier, and overlay bytes; runtime consumption rechecks all four hashes and
their lineage. Responses are untrusted JSON with strict schema and size/count
limits; they cannot contain executable commands, paths, HTML, NaN, or unknown
fields.

Any future-information acceptance, identity/hash mismatch, secondary-only
score, overlay stacking, cap breach, control-chain bypass, or private-data leak
requires `off`. Ordinary per-symbol failure produces zero delta for that symbol
without blocking other valid symbols.
