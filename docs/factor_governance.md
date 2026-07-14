# Factor Governance

## FactorGovernanceProtocol v2 (v13.1 freeze exception)

FactorGovernanceProtocol v2 freezes the research and mutation method, not the
factor identities. Weekly mining and health runs are report-only. The current
branch does **not** permit a new production transition: the available replay
normalizer does not read back the actual bytes of every v13 DAG artifact, so
`canonical_full_chain_replay_producer_unavailable` is a hard blocker. Once a
readback-bound producer exists, the protocol still permits at most one
one-for-one slot transition on the last valid trading day of a month. The
stable local policy hash is
available from `quant_investor.factors.governance_protocol_v2.protocol_hash()`.
Retrieve it without touching the registry with:

```bash
python -c 'from quant_investor.factors.governance_protocol_v2 import PROTOCOL_HASH; print(PROTOCOL_HASH)'
```

The versioned contracts are:

- `FactorSlot` (`factor-slot.v2`): `family + dominant_primitive_cluster`, one
  incumbent and one reserve.
- `FactorEvidenceWindow` (`factor-evidence-window.v2`): snapshot, data/code/cost
  hashes, actual non-overlapping forward-cohort intervals, purge/embargo
  settings, fold count and evaluation hash.
- `FactorTransitionPlan` (`factor-transition-plan.v2`): incumbent, challenger,
  the complete canonical A/B/C/D evidence artifact and its hash,
  before/after weights, blockers and rollback contract.
- `RegistryMutationPlan` (`registry-mutation-plan.v2`): exact target records,
  expected registry SHA, protected metadata, evidence/challenger payload,
  independent mutation-budget-ledger path, WAL path and required inverse patch.

Candidate maturity is at least 12 distinct month-end RankIC observations or
eight non-overlapping 30-day forward cohorts. Mining applies Benjamini-Hochberg
within each family at `q <= 0.10`. A walk-forward claim must be purged with a
30-day embargo and carry per-fold evidence hashes. `data_blocked` neither adds
nor clears an independent alpha-failure streak; two distinct mature failures
make an incumbent reduced/watch eligible, and three make it deprecation
eligible.

A C-arm replacement will be accepted only from a future readback-bound
canonical producer and a full `Quant -> Theme -> Bayesian -> RiskGuard ->
PortfolioConstructor` replay. The current JSON normalizer verifies structure,
self-hashes, strict-snapshot calendar/hash, and A/B/C/D arm hashes for research
reports, but caller-provided 64-hex hashes are not proof of actual DAG artifact
bytes and therefore cannot authorize mutation. Paired after-cost daily deltas,
coverage and drawdown
are recomputed from the incumbent/challenger return arrays; caller-supplied
pass booleans or delta scalars are ignored. The delta uses a deterministic moving-block
bootstrap (minimum 60 samples, fixed seed/block length/resample count recorded
and hashed). The 95% CI lower bound must be positive; computed annualized net
excess improvement must be at least one percentage point; drawdown worsening
must be at most two percentage points; challenger coverage must be at least 95%
of incumbent coverage; and turnover, slippage and tail-risk evidence must stay
inside the strategy limits.

Normalized absolute risk budgets are capped at 20% per factor and 35% per
family. Insufficient evidence retains the prior weights; it never writes a
fixed nominal fallback. The baseline registry is intentionally not restored:
the current selectable set remains one factor while historical records remain
non-live comparison evidence. The corrected metadata count/names/name-set hash
must exactly match the selectable set; this is a baseline consistency repair,
not a transition.

Production runtime uses the same fail-closed boundary. A non-empty selectable
set is not sufficient. `governance_runtime_status()` also requires the registry
protocol version/hash to equal the local v2 policy, exact production-set
count/names/hash metadata, a non-empty family and dominant-primitive cluster
for every live record, one incumbent per slot, valid 20%/35% normalized risk
budgets, and canonical readback-bound evidence marked production eligible.
Until the canonical producer exists, the current one-factor baseline therefore
reports `governance_blocked`, confidence `0`, and no legacy fallback. Explicit
report-only shadow scoring may still compute comparison ranks, but it carries
`factor_mode=historical_shadow_report_only`, confidence `0`, and
`production_eligible=false`; it cannot enter the production DAG as evidence.

Production scoring additionally requires an exact
`factor-production-runtime-contract.v1` entry for every selectable factor under
registry metadata `production_factor_runtime_contracts`. The contract binds the
factor definition and raw record hashes, the allowlisted implementation version
and local code-byte SHA, strict-Parquet column/data semantics and lookback, the
Gate 2 runtime coverage floor, minimum cross-section, and a locally read-back
evidence artifact SHA. Contract names must equal the production set exactly.
`FactorLibrary` name fallback, amount reconstructed from close times volume,
partial-factor renormalization, and neutral filling are forbidden in production;
one compute, output, coverage, lookback, or symbol-set failure blocks the whole
Quant branch. Report-only shadow scoring retains its historical compatibility
helpers and remains confidence zero.

Runtime admission does not trust injected registry objects or serialized
`strict_loader` claims. It reloads `registry_metadata.path` through the strict
snapshot parser and binds the exact registry bytes SHA, complete raw-record
name/SHA set, parsed selectable records, and runtime contracts to that readback.
Each ready score additionally attests the exact symbol count/set, full 100%
per-factor coverage, contract minimum cross-section, required-column frame
values, bounded symbol scores, registry/contracts/receipt identities, and the
result hash. The digest streams canonical, typed consumed values for each
factor's exact lookback without materializing a full-frame payload. The scorer
and DAG branch boundary independently recompute it from their actual frames;
the global boundary accepts only their sealed process-local validation token
and independently revalidates its identities and current governance readback.
Serialized metadata without that independent proof is never production-ready.

The independent production switch is fail-closed by default:

```bash
QUANT_PRODUCTION_KILL_SWITCH=true
QUANT_PRODUCTION_ACTIVATION_RECEIPT=
QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256=
```

Only the exact lowercase value `false` proceeds to receipt validation. The
`quant-production-activation-receipt.v1` file must have exact mode `0600`, be
read back without change, and match the exact-byte SHA supplied only through
`QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256`; registry metadata is not a SHA
fallback. Its payload binds the registry path and bytes SHA, production-set SHA,
runtime-contract aggregate, per-factor implementation-code SHAs,
FactorGovernanceProtocol version/hash, activation ID, approver, and its own
canonical payload hash. The repository does not create or ship an activation
receipt; the current canonical producer blocker remains authoritative even when
these environment variables are set.

Produce a deterministic, private **report-only** evidence artifact from a local
full-chain replay:

```bash
python scripts/build_factor_governance_replay_evidence.py \
  --full-chain-replay-json <private-full-chain-replay.json> \
  --output-json <private-canonical-evidence.json>
```

The normalizer output is content-addressed and mode `0600`, but it is marked
`production_apply_eligible=false`. Hand-written transition/mutation envelopes
are not accepted. Supplying all apply arguments below still exits blocked with
`canonical_full_chain_replay_producer_unavailable` until the real producer is
implemented:

```bash
python scripts/daily_factor_mining_automation.py \
  --apply-governed-transitions \
  --protocol-version v2 \
  --expected-protocol-hash <exact-local-policy-hash> \
  --governed-evidence-json <private-canonical-evidence.json> \
  --mutation-budget-ledger <private-monthly-budget-ledger.jsonl>
```

The CLI re-verifies normalized evidence and internally builds a report-only
transition/mutation plan. It cannot reserve the monthly budget, write a WAL, or
touch the registry while the producer blocker is active. Any apply request
exits non-zero. Inverse-WAL rollback remains available for an already-existing
valid mutation and never deletes or refunds its monthly reservation.

Rollback is dry-run by default. It requires the exact current registry SHA,
input WAL SHA, protocol/transition/mutation/evidence hashes, and the same
append-only budget ledger. Add `--apply-rollback --rollback-wal <new-wal>` only
after reviewing the dry run:

```bash
python scripts/rollback_factor_governance_transition.py \
  --registry-path quant_investor/factor_registry/mined_factors.json \
  --inverse-wal <inverse-wal.json> \
  --mutation-budget-ledger <private-monthly-budget-ledger.jsonl> \
  --protocol-version v2 \
  --expected-protocol-hash <protocol-sha256> \
  --expected-current-registry-sha256 <registry-sha256> \
  --expected-inverse-wal-sha256 <wal-sha256> \
  --expected-transition-hash <transition-sha256> \
  --expected-mutation-plan-hash <mutation-plan-sha256> \
  --expected-evidence-hash <evidence-sha256>
```

Both the legacy bulk "current champion equals the production pool" reconciler
and `mine_quant_branch_factors.py --write-production-candidates` are retired;
either direct-write request exits blocked without changing the registry. A zero-selectable-factor
registry produces `governance_blocked`, zero Quant confidence and no legacy
proxy fallback.

## Objective

> Legacy offline-library reference: the Pass 1-Pass 13 sections below document
> older manual admission and research-library contracts. They are not an
> activation authority for FactorGovernanceProtocol v2. The v2 freeze-exception
> section above, its protocol hash, canonical evidence gate, and month-end
> mutation budget take precedence.

Phase 9 Pass 1 adds an offline factor governance layer for defining factor
contracts, validating existing backtest summaries against explicit thresholds,
recording admission decisions, and materializing a production factor library.

This pass is schema, lifecycle, store, and admission contract only. It does not
calculate factor matrices, parse expressions, run backtests, fetch data, call
LLMs, or connect factors to stock selection or portfolio construction.

## Lifecycle States

The mined-factor runtime v2 path uses:

`research_candidate -> shadow -> mature_candidate -> production_candidate ->
production_factor -> watch/reduced/deprecated`.

The older offline factor-library schema still retains `draft`, `backtested`,
`validated_research`, `paper_trading`, `production`, `rejected` and `disabled`
for historical research artifacts. Those library labels do not authorize a v2
registry transition.

## Production Factor Hard Rules

- Production requires an explicit `approve_production` admission decision.
- Production requires a validation report ID, admission decision ID, and
  `production_since` timestamp.
- Production library entries are sorted deterministically by `factor_id` and
  `factor_version`.
- Duplicate `factor_id` + `factor_version` pairs are rejected.
- Non-production entries are filtered out before building a production library.
- Production stock selection must not consume draft, research, rejected,
  deprecated, disabled, or paper-trading factors.

## Governed Factor Health Automation

The scheduled governed-factor health review is weekly, strict-fresh, and
report-only. Its production-safe invocation is:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
MYQUANT_MARKET_DATA_BACKEND=parquet \
MYQUANT_MARKET_DATA_MODE_POLICY=strict \
uv run python scripts/factor_health_automation.py \
  --cadence weekly \
  --market CN \
  --universes full_a \
  --fresh-evaluation \
  --strict-fresh-evaluation
```

The fresh batch is atomic. Every selectable production factor must have local
fresh evidence, every decision must report `evaluation_source=fresh_evaluation`,
the factor-name set must match exactly, and the batch must have no blockers or
data-blocked factors. The full fresh-analysis context must also load at least
the configured symbol ratio. The strict runtime smoke is part of the same stop
condition: it must use Parquet/strict with no fallback, a healthy snapshot,
non-empty symbol readback, `factor_mode=governed_mined_factors`, and the exact
monitored factor count. A partial batch or failed runtime smoke returns blocked
and cannot fall back to registry evidence. Data-blocked observations do not
consume or increment an alpha-failure window and cannot reduce weight or
deprecate a factor.

Fresh evidence records two independent identities:

- `maturity_window_id` identifies the matured forward-return cohort and is the
  only identity used to count distinct alpha-failure observations. Recomputing
  the same cohort does not add another failure, even if older cohorts are
  revisited out of order. The monitor persists the active distinct failure IDs;
  a genuinely newer healthy window resets the active streak, while a
  data-blocked window neither increments nor resets it.
- `evaluation_hash` records the snapshot, universe, cost assumptions,
  implementation hash, and evaluation configuration for audit comparison.

Existing-factor incremental evidence is leave-one-out: the factor being tested
is removed from the comparison composite. This avoids self-inclusion in Gate 8
and correlation evidence.

The runtime smoke uses strict CN Parquet canonical data through
`MarketDataReader`: it reads `data/parquet/cn/_latest.json`, samples symbol
serving files under the snapshot manifest, and invokes the quant-branch
mined-factor runtime. It never scans legacy `data/clean/cn_daily` CSV
directories. If the pointer, manifest, table dataset, serving cache, PIT input,
or readback is unavailable, the run fails closed as
`parquet_canonical_unavailable` or a specific fresh-evaluation blocker.

`--apply-registry-actions` is a retired compatibility flag: it emits a blocked,
report-only health result and can never mutate the registry. Health evidence
feeds the v2 transition plan; only the v2 month-end apply path may change
production membership. Registry evidence alone is always report-only.

## Quant Factor Selection Shadow

`scripts/run_quant_factor_selection_shadow.py` is a measurement-only runtime
selection experiment. It reads the strict Parquet snapshot, promotes the
selected candidate in memory only, and recomputes exact `MinedFactorScorer`
components and Quant-score Top-20/Top-50 rankings. It writes no registry,
strategy, portfolio, order, or execution record.

The A arm no longer derives its identity from `selectable_factors()`. It
requires an explicit historical baseline manifest. The manifest lists every
factor and shadow weight, hashes the raw JSON content of each named registry
record, and self-hashes the complete list. Loading it builds a separate
in-memory report-only registry; it does not change record state, selectable
membership, weights, or bytes in the formal registry. Missing manifests,
duplicate names, count drift, record drift, or manifest-hash drift block the
run. This permits the current one incumbent plus thirteen historical records
to reconstruct the old-14 comparison without reviving those thirteen factors.

Build the private mode-`0600` manifest from an explicit reviewed list (repeat
`--factor` exactly 14 times), then pass it explicitly to the shadow runner:

```bash
python scripts/build_factor_historical_shadow_manifest.py \
  --registry-path quant_investor/factor_registry/mined_factors.json \
  --baseline-id old14-reviewed-20260712 \
  --factor <factor-1>=0.05 \
  --factor <factor-2>=0.05 \
  --output-json <private-old14-manifest.json>

python scripts/run_quant_factor_selection_shadow.py \
  --historical-baseline-manifest <private-old14-manifest.json> \
  --expected-production-factor-count 14
```

The retained `--expected-production-factor-count` flag name is legacy CLI
compatibility; in this runner it now means the manifest-bound historical
baseline count, not the current selectable production count.

The pre-registered arms are:

- A: all explicit manifest-bound historical factors at manifest shadow weights.
- B_i: A with one historical baseline factor removed.
- C_i: B_i plus the candidate at the removed factor's actual absolute weight.
- D: A plus a dynamically calculated candidate nominal weight that produces an
  exact 3% effective absolute-weight share.

The report includes runtime-parity deltas, rank Spearman, Top-N overlap and
membership changes, candidate coverage, and covered-versus-uncovered selection
rates. It does not substitute the mining Gate 8 linear return overlay for a
runtime rerank.

The v3 create-once preregistration file locks the arms, maturity rule, effective
3% weight, historical-baseline count, lookback, and all coverage thresholds.
Unregistered sensitivity overrides block the governed series. A separate v3
create-once baseline contract locks those experiment parameters plus the
historical manifest hash, factor identities, weights and source-record hashes;
candidate version,
implementation, expression and record hash; strict PIT policy; runtime code
hashes; and Top-N profile. Future market snapshots may advance, but contract
drift blocks the observation. The append-once v3 ledger key includes
`snapshot/as_of/candidate/baseline_contract_hash`; ledger rows are lineage and
deduplication evidence only and can never raise candidate maturity.

Candidate maturity requires at least 12 month-end RankIC observations or eight
non-overlapping 30-day cohorts. A 90-trading-day checkpoint is diagnostic only
and is not sufficient evidence by itself. The shadow covers the Quant-score
ranking only; it cannot claim complete production-screening impact until the
Theme Candidate Pool, liquidity/tradability gates, Fundamental conditional
increment, Bayesian/risk controls, and downstream portfolio funnel are also
evaluated.

## Replacement Readiness And Registry Mutation Safety

`scripts/review_quant_factor_replacement_readiness.py` consumes one or more
strict-fresh health reports and selection-shadow observations. It is
measurement-only and can emit only `keep`, `watchlist`,
`reduce_weight_proposal`, `deprecation_proposal`, or `blocked`. It has no
registry or promotion option.

Two distinct matured alpha-failure windows are required before a reduction can
even be proposed; three are required before a deprecation can be proposed.
Duplicate maturity windows and data-blocked observations do not count. Any
proposal also requires candidate maturity and coverage, leave-one-out removal
that is not worse, a real C-arm replacement better than A and B, redundancy
evidence, diversifier/tail-protection safety, positive cross-branch conditional
increment, an explicit covered-versus-uncovered selection-bias review with
`acceptable=true`, and complete downstream scope. The system does not invent a
bias threshold after observing a run; missing or rejected review evidence blocks
the proposal. Only a passed, measurement-only shadow with unchanged registry,
matched preregistration and baseline contracts, exact runtime parity, and an
empty fail-closed blocker list can supply proposal evidence. Old or forged
ledger rows cannot supply maturity, and regressing cohort chronology blocks the
review.

Registry mutations used by explicitly authorized workflows go through
`quant_investor.factors.registry_store`. The store performs strict JSON/schema
readback, file and record compare-and-swap checks, same-directory atomic replace
with fsync, and a durable before/after/inverse write-ahead journal. The journal
is atomically persisted as `prepared` before registry replacement and changed to
`applied` only after strict readback; a crash-surviving `prepared` journal can be
rolled back only when its after-record and metadata CAS preconditions match.
Mutation APIs default to dry-run. A record-scoped inverse rollback preserves
unrelated later changes and fails on CAS conflict instead of overwriting them.
The current writer accepts only `mined-factor-registry.v1` and rejects unknown
record fields, so a future schema or extension cannot be silently truncated by
a health or mining rewrite.

All work in these sections is a `v13-frozen-20260707` freeze exception. A
protocol-valid v2 month-end transition no longer needs case-by-case manual
approval, but the protocol, PIT, maturity, mutation budget and deterministic
risk gates remain mandatory. Merging this freeze-exception branch still
requires Maxwell's explicit confirmation. No factor workflow calls a broker or
creates orders/trades.

## Admission Gates

`evaluate_backtest_against_thresholds` evaluates an already-produced
`FactorBacktestResult` against `FactorValidationThresholds`.

Hard gates include:
- sample days
- coverage ratio
- rank IC mean
- ICIR
- IC t-stat
- after-cost Sharpe
- positive IC ratio
- positive after-cost top-bottom spread
- monotonicity when required
- point-in-time evidence when required

Warning gates include optional drawdown, turnover, and production-correlation
threshold checks. Missing hard-gate metrics fail the related gate.

By default:
- `pass` recommends `validated_research`.
- `warn` recommends `paper_trading`.
- `fail` recommends `rejected`.
- Admission proposals from a passing report approve paper trading only; production
  approval is intentionally manual in Pass 1.

## First-Pass Non-Goals

Phase 9 Pass 1 does not implement:
- factor matrix loading or persistence
- expression parsing or sandbox execution
- single-factor backtest execution
- correlation or redundancy analysis beyond recorded snapshots
- index-enhancement optimization
- live stock scoring
- `PortfolioConstructor` integration
- `RiskGuard` integration
- posterior, calibration, or overlay changes
- provider, market download, broker, LLM, or web/frontend behavior

## Matrix Data Contract And Expression Sandbox

Phase 9 Pass 2 adds an offline matrix contract and safe expression sandbox for
research-only factor primitives. Matrix data is shaped strictly as
`symbols x dates`, with rows aligned to symbols and columns aligned to ascending
ISO trade dates.

Standard fields include:
- `open`
- `high`
- `low`
- `close`
- `volume`
- `amount`
- `industry`
- `benchmark_close`
- `benchmark_weight`

The helper layer can derive:
- `vwap`: `amount / volume`, returning missing values for missing or zero volume.
- `ret1`: one-period close-to-close return per symbol.
- `benchmark_ret`: one-period benchmark return broadcast across symbols when
  `benchmark_close` is present.

Expressions are parsed through a Python AST whitelist and never through arbitrary
`eval` or `exec`. Allowed names are matrix fields, standard derived fields, and
explicit extra fixture fields supplied by the caller. Calls are limited to the
safe operator whitelist, including time-series operators such as `ts_delay`,
`ts_mean`, `ts_std`, and `ts_corr`; cross-sectional operators such as `cs_rank`,
`cs_zscore`, `cs_indneut`, and `cs_booksize`; and basic elementwise arithmetic.

This pass remains local-only and deterministic:
- no live provider calls
- no Tushare, yfinance, LLM, broker, or web/frontend access
- no wiring into stock selection, `PortfolioConstructor`, or `RiskGuard`

`ts_delay` exists only as a safe expression operator in Pass 2. Execution delay
for research PnL is handled by the Pass 3 backtester described below.

## Single-Factor Long-Short Backtester

Phase 9 Pass 3 adds an offline single-factor backtester on top of the matrix
contract and expression artifacts. It consumes a `FactorMatrix`, a
`MatrixDataBundle`, and a `FactorBacktestConfig`, then writes research artifacts:
weight matrices, daily records, aggregate `FactorBacktestResult` summaries, and
single-factor run envelopes.

Delay alignment is explicit:
- `signal_date`: the factor value date used to form the research book.
- `execution_start_date`: `signal_date + delay_days`.
- `execution_end_date`: `execution_start_date + holding_period_days`.

For the default one-day holding period and `delay_days=1`, a signal on T forms
weights at T, starts execution on T+1, and records the forward return ending on
T+2. Weights are not shifted inside the weight constructor; the daily record
builder applies the alignment.

Execution prices are local matrix fields:
- `open`: uses the bundle `open` field.
- `close`: uses the bundle `close` field.
- `vwap`: uses the bundle `vwap` field, or derives it locally from
  `amount / volume` when absent.

The initial weighting method is `equal_quantile_booksize`. Scores are
`factor_value * expected_direction`, where `expected_direction` comes from
factor matrix metadata and defaults to positive. Long books use
`long_quantile`; short books use `short_quantile` only in long-short mode.
Selected long names receive `+1 / long_count`, selected short names receive
`-1 / short_count`, and net weights are the cellwise sum.

Turnover is defined as:

```text
0.5 * sum(abs(next_weight - previous_weight))
```

across the union of symbols. The first tradable record compares the current net
book to an all-zero book. Costs deduct
`turnover * decimal(transaction_cost_bps + slippage_bps + market_impact_bps)`
from the daily long-short return.

Current Pass 3 limitations:
- no slicing or regime analysis yet
- no advanced transaction cost or capacity model yet
- no factor correlation or redundancy analysis yet
- no production admission yet
- no live provider calls
- no stock selection, `PortfolioConstructor`, or `RiskGuard` wiring

## Factor Backtest Alignment Audit

Phase 12 Pass 1 adds an offline audit layer for factor backtest alignment. It
does not change the Phase 9 backtester, official selection, candidate lists,
branch scores, posterior scores, `RiskGuard`, `PortfolioConstructor`, target
weights, orders, providers, LLMs, or execution.

The audit makes the date semantics explicit:
- `signal_date`: the factor matrix date used to form weights.
- `execution_start_date`: the first date where the signal is allowed to be
  traded.
- `execution_end_date`: the date where the holding-window return ends and the
  daily backtest record is dated.

`delay_days` is the number of matrix date steps between `signal_date` and
`execution_start_date`. A delay of `1` means signal T can execute at T+1. A
delay of `2` means signal T cannot execute until T+2. Same-day execution
(`delay_days=0`) is blocked by the audit and is not an admissible default.

`holding_period_days` is the number of matrix date steps from
`execution_start_date` to `execution_end_date`. With `holding_period_days=1`,
the return source starts on the execution date and ends one matrix date later.
The return window must start after the signal date and end after execution
starts.

Execution prices remain local matrix fields:
- `open`: requires the bundle `open` field.
- `close`: requires the bundle `close` field.
- `vwap`: uses the bundle `vwap` field, or derives it from `amount / volume`
  when `vwap` is absent.

The default T+1 VWAP example is:

```text
signal_date=T
execution_start_date=T+1
execution_end_date=T+2
execution_price=vwap
weight source index=T
return source index=T+1
```

The return matrix audit recomputes expected execution returns from the selected
price field as `price[t + holding_period_days] / price[t] - 1`. It compares the
observed execution return matrix against that forward execution window using a
fixed tolerance. A matrix that instead matches a shifted prior close-to-close
return such as `price[t] / price[t-1] - 1` is flagged as misaligned/lookahead
risk because it is not the expected execution-window return source.

Alignment audit artifacts are diagnostics only. They can be saved under
`data/factor_library/alignment_audit`, but this pass does not approve any
factor into production and does not wire the factor library into selection or
portfolio construction.

## A-share Tradability Mask and Execution Feasibility Audit

Phase 12 Pass 2 adds an offline A-share tradability mask and execution
feasibility audit for factor backtests. It reads local `MatrixDataBundle`,
`FactorWeightMatrix`, and optional `SingleFactorBacktestRun` artifacts only. It
does not call live providers, LLMs, brokers, market maintenance, market
analysis, market runs, or frontend/web code.

The tradability mask records five cell-level masks:
- `can_trade_mask`: basic market tradability for the symbol/date.
- `can_buy_mask`: whether a new long entry or weight increase is feasible.
- `can_sell_mask`: whether an exit, reduction, or synthetic short-leg sell is
  feasible.
- `can_hold_mask`: whether research accounting can continue carrying the
  position.
- `research_eligible_mask`: whether the symbol/date remains eligible for the
  research universe.

The A-share checks are deterministic and field-driven:
- suspension blocks trading, buys, and sells; with the suspension filter it also
  removes the cell from research eligibility.
- limit-up blocks buys and new long entries, but does not block sells.
- limit-down blocks sells and exits, but does not block buys.
- ST / risk-warning status removes research eligibility and blocks new buys when
  the ST filter is enabled.
- delisted status blocks trade, buy, sell, hold, and research eligibility.
- new listings below `min_listing_days` block new buys and research eligibility.
- valid price is taken from an explicit `valid_price` field, or inferred from a
  positive price field; VWAP can be derived from `amount / volume`.
- valid volume is taken from `valid_volume`, or inferred from positive `volume`.
- low liquidity uses either `low_liquidity` or `amount < min_amount`; it blocks
  new buys and research eligibility while allowing sells unless another blocker
  applies.

The execution feasibility audit compares factor target weights against the
tradability mask on the execution date. It computes each symbol transition as
`target_weight - previous_weight`, classifies it as buy, sell, or hold, and
flags blocked buy/sell transitions when the execution-date mask prevents the
required action. Long-short factor books remain research analytics: the short
leg is audited as synthetic sell/buy transitions for diagnostics, not as
broker-ready A-share cash-equity short execution.

Artifacts are diagnostics only and can be saved under
`data/factor_library/tradability_audit`:
- `tradability_masks.jsonl`
- `tradability_audit_reports.jsonl`
- `execution_feasibility_reports.jsonl`
- `tradability_audit_report.md`
- `execution_feasibility_report.md`

No-runtime-impact boundary: this pass does not alter official scoring, stock
selection, candidate lists, branch scores, posterior scores, `RiskGuard`,
`PortfolioConstructor`, target weights, orders, `action_taken_today`, providers,
LLMs, or execution.

Current limitations:
- no PnL adjustment yet
- no partial fill model
- no broker execution model
- no live provider calls

## Offline Execution Cost and Penalty Simulation

Phase 12 Pass 3 adds a separate offline simulation layer that applies execution
cost and penalty assumptions to factor backtest daily records. It consumes
existing `SingleFactorBacktestRun`, `FactorWeightMatrix`, local
`MatrixDataBundle`, and optional `AShareTradabilityMask` artifacts. The output
is a separate simulated return series and adjusted-run artifact; the original
backtest run, daily records, target weights, official stock selection,
posterior scores, `RiskGuard`, `PortfolioConstructor`, orders, providers, LLMs,
and execution paths are not mutated or replaced.

The cost model records these components independently:
- commission, exchange fees, slippage, and spread cost as absolute executed
  trade weight times the configured bps rate.
- stamp tax using the A-share sell-side convention by default. If
  `apply_stamp_tax_on_sell_only` is disabled, stamp tax applies to all executed
  trades in the simulation.
- market impact from a deterministic participation proxy:

```text
participation_rate = abs(executed_trade_weight) * portfolio_value / amount
```

The supported impact models are fixed bps, linear participation, and square-root
impact. Linear participation computes
`impact_bps = impact_coefficient * participation_rate`; square-root impact
computes `impact_bps = impact_coefficient * sqrt(participation_rate)`.

Tradability constraints are applied before costs. Blocked buys and blocked
sells keep the previous weight under the default `keep_previous_weight` policy.
The `block_to_cash` policy is recorded as the same conservative simplification
in this pass because cash mechanics are not modeled. The
`mark_unexecutable_only` policy can leave target weights intact while marking
the transition as blocked for diagnostics. Blocked-transition penalties use the
simple deterministic rule
`abs(target_weight - executable_weight) * abs(gross_return)`.

Participation above `max_participation_rate` creates partial-fill and
low-capacity diagnostics, but Pass 3 does not change executed trade size for
partial fills. Missing amount, volume, or price data are warning diagnostics.
Long-short factor books remain research analytics: any short leg is reported as
a caveat and is not treated as a broker-ready A-share cash-equity short.

Artifacts are simulation-only and can be saved under
`data/factor_library/execution_cost`:
- `execution_cost_reports.jsonl`
- `execution_adjusted_runs.jsonl`
- `execution_adjusted_daily_records.jsonl`
- `execution_cost_report.md`
- `execution_cost_dashboard.json`

No-runtime-impact boundary: this execution-cost simulation is offline-only and
does not alter official scoring, stock selection, candidate lists, branch
scores, posterior scores, `RiskGuard`, `PortfolioConstructor`, target weights,
orders, `action_taken_today`, providers, LLMs, brokers, or execution.

Current limitations:
- simple proxy model only
- no partial fill mechanics beyond diagnostics
- no broker execution model
- no live provider calls
- not wired into admission by default

## Metrics, Slicing, Cost/Capacity Validation

Phase 9 Pass 4 adds offline pre-admission validation helpers on top of
`SingleFactorBacktestRun`. These helpers summarize return/risk behavior, slice
the existing daily records, and record cost/capacity diagnostics without changing
admission defaults or production behavior.

Return metric summaries include:
- mean daily return
- annualized return
- annualized volatility
- Sharpe ratio
- maximum drawdown
- positive return ratio
- cumulative return

Slice validation supports:
- full-sample validation
- recent 1-year, 3-year, and 5-year trailing windows when enough local records
  exist
- regime slices supplied by the caller as a local `date -> regime_label` mapping

Each slice compares before-cost (`long_short_return`) and after-cost
(`after_cost_return`) series, optional excess return series, turnover metrics,
coverage/missing ratios, and average long/short book counts. Threshold breaches
produce deterministic warnings. Insufficient sample days fail the slice.

Cost and capacity diagnostics use supplied local matrix fields only. The first
capacity proxy uses the `amount` matrix as daily traded value, active weighted
symbols from the backtest weight matrix, and a configured maximum participation
rate. Daily capacity is approximated as:

```text
average active-symbol amount * max_participation_rate / max(turnover, epsilon)
```

Participation breaches compare requested daily turnover value
(`turnover * target_capital`) against allowed trade value. This is a simple
offline ADV/participation proxy, not a broker execution model.

The cost/capacity report also records:
- before-cost versus after-cost Sharpe
- average turnover
- configured total cost bps
- estimated average cost return
- cost drag ratio, computed as
  `max(0, before_cost_sharpe - after_cost_sharpe) / abs(before_cost_sharpe)`
  when both Sharpe values are positive
- average ADV from the local `amount` matrix
- participation breach ratio
- tradability ratio from the local `tradability_mask`
- coverage ratio from the aggregate backtest result

`build_enhanced_factor_validation_report` combines the existing aggregate
`FactorBacktestResult`, optional robustness report, and optional cost/capacity
report into an enhanced `FactorValidationReport`. It can recommend
`validated_research`, `paper_trading`, or `rejected`, but it never approves
production and does not replace the Pass 1 admission decision flow.

Current Pass 4 limitations:
- no correlation or redundancy analysis yet
- no portfolio contribution or index-enhancement validation yet
- no production admission
- no live provider calls
- no stock selection, `PortfolioConstructor`, or `RiskGuard` wiring
- the capacity model is a simplified offline proxy, not a broker execution model

## Correlation, Redundancy, And Portfolio Contribution

Phase 9 Pass 5 adds an offline incremental research layer for comparing a
candidate factor against existing production or research factor artifacts before
admission review. It remains a pure helper layer and does not approve factors or
alter any default runtime behavior.

The redundancy analysis can compare:
- after-cost return-series correlation against existing
  `SingleFactorBacktestRun` records
- cross-sectional matrix rank correlation when `FactorMatrix` artifacts are
  supplied for both candidate and reference factors
- IC-series correlation when daily record metadata contains `ic` or `rank_ic`
- simple residual mean return after neutralizing candidate after-cost returns
  against one reference factor return series

Correlation pair verdicts are:
- `distinct`
- `related`
- `redundant`
- `insufficient_data`

`build_factor_redundancy_report` aggregates pair-level results, records maximum
absolute return, matrix-rank, and IC correlations, and lists related or redundant
reference factor IDs for later research review. A redundant verdict is only a
pre-admission warning; it does not disable or approve anything by itself.

The contribution analysis builds local factor-return pools from existing
single-factor backtest runs. Baseline pool returns are equal-weighted by default
or use caller-supplied run weights. Missing source dates are handled by
renormalizing available source weights for that date. The candidate factor is
then combined with the baseline pool using configured baseline and candidate
weights.

Contribution reports include:
- incremental annualized return
- incremental Sharpe
- incremental maximum drawdown, where positive means drawdown got worse
- incremental turnover
- verdicts of `improves`, `neutral`, `degrades`, or `insufficient_data`

Current Pass 5 limitations:
- this is factor-return contribution analysis, not live portfolio construction
- there is no index-enhancement optimizer yet
- there is no production approval or admission replacement
- no live provider calls, Tushare/yfinance calls, LLM calls, broker calls, or web
  calls are made
- no stock selection, `PortfolioConstructor`, or `RiskGuard` wiring is added

## Production Factor Library And Audit

Phase 9 Pass 6 closes the offline governance loop with a production library
builder, audit report, guardrail helper, and dashboard payload.

Production library construction requires an explicit
`approve_production` `FactorAdmissionDecision` with `target_status=production`.
A passing validation report alone is not sufficient, and the helper does not
auto-approve production factors. By default, each production entry must have a
matching factor definition, a matching validation report with `pass` or `warn`,
and current validation evidence. Failed validation reports are excluded from the
library.

Validation currency is checked from `expires_at` when present. Otherwise the
audit uses `last_revalidation_at`, then `production_since`, plus the policy
`production_revalidation_days`. Missing dates are treated as expired.

Redundancy and contribution artifacts are pre-admission evidence. They can warn
on redundant factors, weak contribution, or missing incremental review, but they
do not approve or reject production by themselves. Non-production factors remain
blocked from formal stock selection and portfolio construction. The guardrail
helper can report `allowed`, `blocked`, or `shadow_only` for future integration,
but it is not wired into stock selection, `PortfolioConstructor`, or `RiskGuard`
by default.

Audit outputs are local artifacts:
- append-only JSONL audit reports
- a stable markdown audit report
- a JSON dashboard payload
- a JSON-serializable context patch for future runtime shadow reads

Current Pass 6 limitations:
- no runtime stock selection integration
- no `PortfolioConstructor` integration
- no `RiskGuard` integration
- no automatic production approval
- no live provider, Tushare/yfinance, LLM, broker, or web calls

## Shadow Scoring Comparison

Phase 11 Pass 1 adds a read-only comparison layer between local production
factor-library signals and already-produced official candidate/ranking outputs.
It is observability only: it does not alter official scoring, candidate
selection, posterior scores, `RiskGuard`, `PortfolioConstructor`, target
weights, orders, providers, LLMs, broker/execution, or web/frontend behavior.

The comparison reads the local production factor library when supplied or
available under `data/factor_library/production_factors.json`. It can also read
local factor matrices from `data/factor_library/matrix/factor_matrices.jsonl`
when a caller supplies those artifacts. Missing libraries, matrices, symbols,
dates, or values produce warnings in the report instead of runtime failures.

For each production factor, the scorer extracts the latest matrix value with a
matrix date less than or equal to `as_of`. Raw values are cross-sectionally
rank-normalized across the supplied official candidate symbols. The factor's
`expected_direction` comes from the matching `FactorDefinition` when available,
then matrix metadata, then defaults to positive direction. Higher adjusted
values receive better normalized scores.

The first shadow score is equal-weighted across covered production factors.
For each candidate, the report records:
- official score and rank, deriving rank from official score when needed
- shadow factor score and shadow factor rank
- `rank_delta = official_rank - shadow_factor_rank`, where a positive value
  means shadow factors rank the candidate higher than the official output
- raw `score_delta = shadow_factor_score - official_score` when both exist
- factor coverage ratio and warning codes

Report diagnostics include official Top-N symbols, shadow Top-N symbols, their
intersection, overlap ratio, largest positive and negative rank deltas, compact
candidate tables, JSON dashboard payloads, and append-only JSONL score/report
ledgers.

Current limitations:
- shadow scores are not official scores
- no stock-selection effect
- no portfolio-construction effect
- no factor weighting optimization beyond equal weighting
- local production library and factor matrices are required for meaningful
  coverage
- audit-blocked factors remain excluded unless explicitly requested by the
  caller for read-only diagnostics

## Artifact Locations

The default local store root is `data/factor_library`.

- `factor_definitions.jsonl`
- `factor_backtest_results.jsonl`
- `factor_validation_reports.jsonl`
- `factor_admission_decisions.jsonl`
- `production_factors.json`
- `deprecated_factors.json`

The JSONL ledgers are append-only and reject duplicate IDs on append.

The matrix store root is `data/factor_library/matrix`.

- `matrix_contracts.jsonl`
- `matrix_bundles.jsonl`
- `factor_matrices.jsonl`
- `expression_results.jsonl`

These JSONL ledgers are fixture/research oriented in Pass 2 and are not a
production parquet store.

The single-factor backtest store root is `data/factor_library/backtest`.

- `factor_weight_matrices.jsonl`
- `factor_backtest_runs.jsonl`
- `factor_daily_records.jsonl`

These ledgers are append-only research artifacts. They do not approve factors
for production and are not consumed by formal selection or portfolio construction.

The enhanced validation store root is `data/factor_library/validation`.

- `factor_robustness_reports.jsonl`
- `factor_cost_capacity_reports.jsonl`
- `enhanced_validation_reports.jsonl`

These ledgers are append-only research artifacts for later admission review.
They do not wire factors into live scoring, formal selection, or portfolio
construction.

The incremental correlation/contribution store root is
`data/factor_library/incremental`.

- `factor_redundancy_reports.jsonl`
- `factor_contribution_reports.jsonl`

These ledgers are append-only pre-admission research artifacts. They do not wire
factors into stock scoring, formal selection, portfolio construction,
`PortfolioConstructor`, or `RiskGuard`.

The production library audit store root is `data/factor_library/audit`.

- `factor_library_audit_reports.jsonl`
- `factor_library_audit_report.md`
- `factor_governance_dashboard.json`

These artifacts summarize production-library readiness and known blockers. They
do not alter runtime behavior.

The shadow scoring comparison store root is
`data/factor_library/shadow_scoring`.

- `shadow_factor_scores.jsonl`
- `shadow_candidate_scores.jsonl`
- `shadow_comparison_reports.jsonl`
- `shadow_comparison_report.md`
- `shadow_scoring_dashboard.json`

These artifacts are append-only or report/dashboard outputs for read-only
official-versus-shadow comparison. They do not alter runtime behavior.

## Multi-Date Shadow Evidence Collection

Phase 13 adds an offline evidence collection layer for local multi-date shadow
scoring review. It reads already-computed candidate snapshots, a local
production factor library, local factor matrices, and optional local audit
artifacts. It then summarizes whether official rankings and production-factor
shadow rankings look stable enough to justify a later paper-portfolio
comparison.

A real sample run uses a manifest of local artifacts:

```bash
PYTHON=./.venv/bin/python scripts/collect_factor_shadow_evidence.py \
  --input-manifest data/factor_library/evidence/sample_manifest.json \
  --output-dir data/factor_library/evidence \
  --generated-at 2026-04-27T00:00:00Z \
  --top-n 30 \
  --min-observation-days 20
```

The evidence report tracks official versus shadow rank stability with:

- top-N overlap trend: for each date, the official top-N symbols are compared
  with the shadow top-N symbols; the overlap ratio is
  `len(overlap_symbols) / min(top_n, candidate_count)`.
- factor coverage trend: the date-level average of per-candidate covered
  production factors divided by usable production factors.
- rank drift trend: the date-level average and maximum absolute difference
  between official rank and shadow factor rank for candidates that have both.
- audit blocker/fail day counts: dates with library blockers, alignment audit
  fail verdicts, tradability audit fail verdicts, and execution-cost warn/fail
  verdicts.

The default evidence store root is `data/factor_library/evidence`.

- `evidence_date_results.jsonl`
- `multi_date_evidence_reports.jsonl`
- `evidence_report.md`
- `evidence_dashboard.json`

No-runtime-impact boundary: this pass is evidence only. It does not alter
official scores, stock selection, posterior, `RiskGuard`,
`PortfolioConstructor`, target weights, orders, providers, LLMs, broker APIs,
or execution.

Current limitations:

- evidence only;
- no paper portfolio yet;
- no official scoring impact;
- requires local artifacts and candidate snapshots.

## Governed Mined Factor Runtime Admission

This section describes the older v1 registry-admission mechanics only. It is
not authority for v2 production activation. The v2 runtime contract and
automatic month-end transition protocol above supersede the former manual
promotion workflow.

Mined-factor lifecycle states are:

- `draft`
- `research_candidate`
- `paper_factor`
- `production_candidate`
- `production_factor`
- `deprecated`

Automated review decisions are limited to:

- `reject`
- `revise`
- `watchlist`
- `paper_factor`
- `production_candidate`

`production_factor` was not an automated v1 review decision. Under v2, only the
hash-bound month-end transition engine may create a production transition;
manual editing or the old direct writer is not an activation path.

The mined-factor evaluator records eight gate results:

1. Data safety.
2. Coverage and stability.
3. IC / RankIC.
4. Group returns.
5. Cost and turnover.
6. Neutralization and exposure.
7. Out-of-sample robustness.
8. Portfolio incremental validation.

Gate 1 failure is a hard reject. Passing Gates 1-4 without enough OOS or
portfolio-incremental evidence allows only `paper_factor`. All eight gates must
pass, and production-grade ICIR plus positive-IC-ratio thresholds must be met,
before the automated decision can reach `production_candidate`.

Runtime quant-branch admission is stricter than review admission. The runtime
scorer consumes only factors where:

- `state == production_factor`
- all eight gates passed
- `weight != 0`
- no deprecation reason is present

All `draft`, `research_candidate`, `paper_factor`, and `production_candidate`
records are skipped by design. If the registry has no selectable
`production_factor`, or if any v2 registry/protocol/slot/budget/evidence check
fails, the DAG quant branch and standalone `QuantAgent` enter
`governance_blocked` with zero confidence. They never fall back to the legacy
`short_term_return` / `volatility_penalty` deterministic proxy.

## A_quant Expression Retest Bridge

Use `scripts/retest_aquant_alpha_mix_8gate.py` to retest A_quant
`alpha_mix_vwap*_ocfprofit_*` candidates on the myQuant data layer. The runner
loads A_quant `audit_extended/ready_factors.json` and
`independent_ready_subset.json`, computes the expression runtime locally, and
feeds the resulting metrics into the existing `FactorGateEvaluator`. It writes
review evidence under `reports/factor_governance/aquant_alpha_mix_retest_*` and
does not modify `quant_investor/factor_registry/mined_factors.json`.

The runtime implementation string for this bridge is
`aquant_expression:<factor_name>`. It is only consumed by the normal mined-factor
runtime when the registry record is already a selectable `production_factor`
with all eight gates passed. `paper_factor`, `research_candidate`, and
`production_candidate` records remain skipped.

`fin_ocf_to_profit` is loaded from the source-backed PIT helper at
`quant_investor.factors.pit_fundamentals`. Production reads use the strict
Parquet fundamental generation selected by
`data/parquet/cn/_fundamental_latest.json`. A generation contains period,
daily, and quarantine tables plus hashes under
`data/parquet/cn/_fundamental_generations/`; the pointer is published only
after all three tables pass readback. Before the first generation is published,
the reader accepts the legacy `fundamental_daily/part.parquet` and
`fundamental_period/part.parquet` layout. Legacy metadata CSV inputs are not
production read ports.

Fundamental refreshes are merge-preserving. Missing or empty provider responses
do not delete prior PIT rows, and a less-complete exact-key row cannot replace a
more-complete prior row. A complete incoming row wins a quality tie. Provider
deletion requires a future explicit tombstone path; absence is never inferred
as deletion. Readiness reports include expected-symbol coverage so a partial
full-A fetch cannot be described as complete merely because its returned rows
have high field coverage. The operational `fundamental-maintain` path resolves
the requested universe from canonical components intersected with strict
Parquet serving before deriving the mart. A missing scope or symbol coverage
below 95% fails Gate 2 and does not publish a new
canonical pointer. Low-level fixture/recovery writers may still materialize a
blocked generation for audit, but must preserve `gate2_passed=false`.

Global mart readiness and candidate-level readiness are separate controls. A
full-A coverage failure is disclosed globally; it does not erase complete PIT
records for covered symbols. Candidate and holding review still blocks a symbol
with no record, warns on a partial record, and admits a complete record only
when its generation has an auditable provider priority. Offline reconstruction
from canonical raw data may claim `tushare_primary` only when its generation
manifest names local Tushare refresh evidence; otherwise it remains
`manual_offline_snapshot`.

An operator-directed rebuild from already-canonical local raw Parquet may call
`write_fundamental_mart(..., write_raw_snapshots=False)` to avoid duplicating
the same raw data as CSV. The provider manifest must identify the offline raw
scope and state that no network was used; normal provider maintenance keeps the
default raw-snapshot behavior.

```text
ts_code, report_period, availability_date, metric_name, value, source,
fetched_at, raw_table, raw_field
```

For Tushare `income` and `cashflow`, `availability_date` is
`f_ann_date || ann_date`. `fetched_at` is audit metadata only and is not treated
as the historical availability date when announcement dates exist. The derived
metric is `n_cashflow_act / n_income`; zero or missing denominators produce
`NaN`. Optional Tushare backfill is limited to this PIT financial coverage and
must not call broker, LLM, or production registry writes. The retest runner
applies per-request and total-elapsed provider timeouts; timeout or partial
coverage is reported as a blocker rather than converted into an alpha verdict.

The default registry is empty:

```json
{
  "schema_version": "mined-factor-registry.v1",
  "factors": []
}
```

## Mining Candidate Diversity Admission

The admission/diversity calculations below remain useful report evidence, but
their former direct registry-write and bulk production-reconciliation behavior
is retired. They may feed a v2 challenger/evidence plan only; they cannot
authorize production runtime or bypass the one-slot monthly mutation budget.

The weekly mining writer applies `candidate-diversity-policy.v1` after a
candidate has positive evidence and passes all eight gates. Registry admission
then requires three additional deterministic checks:

1. one alpha-first champion per exact factor family;
2. one champion per connected component that shares a dominant underlying
   primitive (normalized absolute contribution at least 50%); and
3. one champion per candidate-signal correlation component using median
   absolute monthly Spearman correlation at or above 0.70.

Pairwise correlation requires at least 20 common symbols on each date and at
least three valid common rebalance dates. A single survivor makes correlation
not applicable. Multiple survivors with incomplete required pair evidence fail
closed and cannot write the registry.

Champion order is ICIR, mean RankIC, cost-adjusted return, incremental master
return, lower turnover, lower existing-factor correlation, then factor name.
Research-only implementations remain visible in the report but are not
registry-write eligible. All non-champion variants stay in the mining report
with `skip_reason=same_family_redundant` and a stage of `family`,
`primitive_lineage`, or `signal_cluster`.

The writer independently validates the diversity policy hash and champion
status before its CAS/WAL mutation. `max_registry_candidates` is applied only
after diversity selection, and every skipped qualified candidate is enumerated
in the registry manifest. If no diverse champion remains, strict weekly mining
may carry forward a valid incumbent; without an incumbent it returns failure
and leaves the registry unchanged.

Existing zero-weight candidates are not deleted or rewritten. Each mining run
emits `legacy_candidate_redundancy_audit.json`; they remain non-live unless a
new current full-A run selects them as a champion.

Weekly mining defaults to the exact `full_a` universe and publishes fail-closed
challenger evidence, but it is report-only and cannot mutate the registry.
Production exposure evidence uses the strict
Parquet `dag_core_raw/stock_basic` reference. Size uses same-trade-date
`daily_basic.total_mv` where available, then a bounded reconstruction from
strict `dag_core_raw/daily_basic_ext.total_share` times same-day unadjusted
close. The catalog hash, row count, and status for all source tables must pass.
Symbol, evaluation-date, minimum cross-section, and combined size-pair coverage
must each be at least 95%; exact PIT size-pair coverage must be at least 60%;
reconstructed pairs may not exceed 35% of valid combined pairs. The share
reference date must cover the last evaluation date. Reports disclose exact and
reconstructed counts separately and do not label the hybrid result as fully
PIT. Exposure is recomputed from the restricted analysis context used for
candidate metrics; publishing the full-history context's exposure metadata is
invalid. Its first evaluation date must not precede the resolved analysis start.

The former scheduled-mining direct production reconciliation is retained only
as historical documentation. Current CLIs are report-only unless the explicit
FactorGovernanceProtocol v2 apply arguments are supplied, and even then the
canonical producer gate, one-slot plan, CAS/WAL and monthly budget must pass.
Strict evidence failure is `data_blocked`: it leaves the registry byte-identical
and does not advance or clear an incumbent failure streak. Factor health is also
report-only and cannot retire, carry forward, promote, or reweight production
records. The retired bulk/direct writers return a structured blocker instead of
silently applying their historical semantics.

The following record shape is a legacy schema example, not an activation
instruction. Do not manually promote it:

```json
{
  "name": "momentum_1m",
  "version": "v1",
  "state": "production_factor",
  "implementation": "alpha_mining.FactorLibrary:momentum_1m",
  "weight": 0.10,
  "gate_results": [
    {"gate_id": 1, "gate_key": "data_safety", "passed": true},
    {"gate_id": 2, "gate_key": "coverage_stability", "passed": true},
    {"gate_id": 3, "gate_key": "ic_rankic", "passed": true},
    {"gate_id": 4, "gate_key": "group_returns", "passed": true},
    {"gate_id": 5, "gate_key": "cost_turnover", "passed": true},
    {"gate_id": 6, "gate_key": "neutralization_exposure", "passed": true},
    {"gate_id": 7, "gate_key": "oos_robustness", "passed": true},
    {"gate_id": 8, "gate_key": "portfolio_incremental", "passed": true}
  ]
}
```

### Production Quant evaluation boundary

Production scoring additionally requires a readback-verified immutable
evaluation context. It binds the exact market and universe symbol-set hash,
one common `evaluation_as_of`, the canonical pointer and snapshot-manifest
bytes, per-symbol read provenance, the latest complete trade date, an
independent readback-verified market-calendar artifact proving that the date is
open, and (for CN) complete required PIT-membership artifacts whose canonical
Parquet rows are re-evaluated and matched exactly to the scoped statuses.
Non-CN markets record PIT as explicitly not applicable. Missing context,
artifact drift, a future/stale/duplicate/disordered/intraday date, symbol
identity drift, or unequal terminal dates blocks Quant with confidence zero.
The DAG quarantines an invalid frame with a stable per-symbol diagnostic before
building the context, so one stale or malformed symbol cannot poison otherwise
eligible symbols. Quarantined frames never enter Quant or cross-section inputs;
the scorer's direct API remains fail-closed if such a frame reaches it.

The context SHA is part of runtime metadata, output attestation, the
process-local branch validation token, and the global Quant identity check.
Serialized claims plus frames alone are insufficient to establish readiness;
the boundary also requires the verified context or its internal validation
token. This does not relax the activation receipt, canonical producer, kill
switch, or freeze-exception merge gates.

## Future Roadmap

1. Index-enhancement validation on top of the offline contribution layer.
2. Manual admission review on top of enhanced validation and incremental
   research artifacts.
3. Shadow-read production library outputs in reports after explicit admission.
4. Compare factor-driven selection against current selection before any runtime
   selection integration.
