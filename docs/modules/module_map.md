# Module Map

## Mainline authority and public readers

### `quant_investor/v17_mainline`

- `constants.py` defines the frozen V17 v4 schema IDs, CN-only market boundary,
  authority source, storage root, states and blocker codes.
- `contracts.py` validates canonical JSON, exact references, active pointers,
  runs and transitive authority.
- `runtime.py` reads one exact strategy pointer and projects the public DTO.
- `storage.py` provides low-level, contract-tested storage primitives. It is not
  a public production publisher or activation workflow.

This package owns the only decision-mainline authority carrier. The repository
currently exposes read-side access only: there is no public command that
publishes or activates a production mainline run. Shadow and Investment
Intelligence artifacts are rejected as decision authority.

### `quant_investor/cli`

`research run`, `market analyze` and `market run` all read the same exact V17
active pointer. They require `--strategy-id`, are CN-only and write nothing.
`market backtest` is deliberately unavailable. Market maintenance commands are
a separate data-management surface; `market download` remains a compatibility
alias.

### `quant_investor/pipeline`

The public `QuantInvestor` object is a read-only facade over the V17 mainline
reader. It does not orchestrate research, construct a portfolio, publish a run
or activate a pointer.

## Market data and research evidence

### `quant_investor/market`

Owns strict market maintenance, canonical Parquet reads, data-governance checks
and supporting analysis helpers. CSV is limited to explicit export or migration
paths. Mainline backtesting is unsupported.

### `quant_investor/factors`

Contains Factor Governance and production-control evidence. Reports,
nominations and Shadow forward evidence do not grant mainline authority.

### `quant_investor/macro`

Contains source-bound macro observation helpers. These helpers do not own the
mainline pointer and cannot act as a silent risk or allocation override. The
separate `macro_terminal_tushare.py` module is a standalone historical/manual
reference surface, not the V17 mainline macro contract.

## V17 contracts and research runtimes

### `quant_investor/v17_v4_contract`

Provides V17 v4 schemas, canonicalization, validators and packaged resources.

### `quant_investor/v17_v4_runtime`

Provides the governed, research-only V4 Forward Observation workflow and the
separate `run-forward` Shadow evidence lane. Shadow outputs never advance the
mainline pointer.

### `quant_investor/intelligence`

Implements the I0 Investment Intelligence layer and the R2.2 Forward Research
Evaluator. The regime package performs exactly one causal Markov filter step
for Market, Industry and Theme. It has no backward smoothing, persistence,
portfolio, broker or execution authority. R2.2 is offline and stdout-only; any
memory update is a caller-owned proposal, not a write.

### `quant_investor/intelligence/decision`

Implements the I1 Investment Decision Intelligence library above exact I0 and
optional exact R2.2 replay. It builds content-addressed Decision Context, Risk
Assessment, five-state Decision, deterministic Memo, Decision Discipline and
external-review-only Paper Intake values. Validators rerun the complete supplied
closure and require byte-for-byte equality rather than trusting a resealed
document.

This package is library-only. It has no public CLI or Web registration,
scheduler, daemon, persistence or Memory writer, selector, portfolio builder,
paper adapter implementation, provider or LLM call, broker, order, execution or
trade authority. `PAPER_CANDIDATE` means only external paper-review eligibility;
it is not portfolio admission or a trading signal.

### `quant_investor/v17_v3_contract`

Retains explicit historical compatibility contracts used by governed V4
research tooling. It is not a competing public decision mainline.

## Standalone legacy automation

### `quant_investor/automation`

Contains a restored, standalone V15-style automation implementation. It has no
public V17 CLI or web registration, does not schedule the I0/R2.2 research loop,
and still has unresolved lazy runtime dependencies. Do not treat its presence
as an active V17 workflow or invoke it without an explicit legacy task and
dependency validation.

## Public workspace

### `web/`

The FastAPI workspace exposes the exact active V17 result through
`GET /api/research/{strategy_id}`. The optional
`expected_pointer_sha256` query pins one pointer generation. Settings, universe
and data endpoints are supporting surfaces; none publishes or activates a V17
mainline result.

### `portfolio_dashboard/`

Renders the read-only V17 mainline DTO. It is a consumer, not an activation or
result-selection authority.
