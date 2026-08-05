# Module Map

## Mainline authority

### `quant_investor/v17_mainline`

- `constants.py`: frozen V17 v4 schema IDs, storage root, supported CN market,
  authority source, state names, and closed blocker codes.
- `contracts.py`: canonical JSON, exact-field, reference, active-pointer, run,
  and transitive authority validation.
- Runtime/public-surface modules: immutable run publication, pointer CAS, and
  read-only public projection.

This package owns the only decision-mainline authority carrier. It rejects
Shadow and forward-evidence artifacts as authority.

### `quant_investor/cli`

Routes `research run`, `market analyze`, and `market run` to the same V17
active-pointer reader. The public result path is CN-only and read-only.

## Decision pipeline

### `quant_investor/pipeline`

Orchestrates research evidence, structured review, deterministic controls, and
portfolio output. A completed pipeline result does not become public until it
is closed as a mainline run and separately activated.

### `quant_investor/market`

Owns strict market maintenance, canonical reads, data-governance checks, and
supporting analysis helpers. CSV is limited to explicit export or migration;
mainline inputs use canonical Parquet plus hash-bound manifests and pointers.
Mainline backtesting is unsupported.

### `quant_investor/agents`

Contains structured review, RiskGuard, IC coordination, portfolio construction,
and narration. Advisory output cannot override deterministic control gates.

### `quant_investor/factors`

Contains Factor Governance and production-control evidence. Report, nomination,
and Shadow evidence do not automatically grant mainline authority.

### `quant_investor/macro` and `quant_investor/regime`

Produce hash-bound context and exposure-tightening overlays. Missing required
evidence is explicit and cannot be replaced by a silent no-op.

## V17 contracts and research

### `quant_investor/v17_v4_contract`

Provides V17 v4 schemas, canonicalization, validators, and packaged resources.

### `quant_investor/v17_v4_runtime`

Provides the governed V17 runtime and the separate `run-forward` Shadow
evidence workflow. Shadow outputs never advance the mainline pointer.

## Public workspace

### `web/`

The FastAPI workspace exposes the active V17 result through
`GET /api/research/{strategy_id}`. The optional expected pointer SHA pins an
exact generation.

### `portfolio_dashboard/`

Renders the read-only V17 mainline DTO. It is a consumer, not an activation or
result-selection authority.
