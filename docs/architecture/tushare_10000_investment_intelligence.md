# Tushare 10,000 Investment Intelligence Flow

This document describes the governed source path added for the Tushare 10,000
points capability. It is a data and research capability, not a trading
activation. The code remains offline by default and does not add a public CLI,
Web route, scheduler, daemon, broker, order or execution surface.

## What the higher points level changes

The useful change is access to batch-oriented Fundamental, Industry and Theme
sources. It does not make every Tushare endpoint available and does not grant
separate real-time or minute-data permissions. Static point requirements and
observed capability are recorded separately. A lane becomes usable only after
an exact live probe returns the expected schema and a complete, non-truncated
partition.

```text
sealed endpoint policy and execution plan
  -> exact capability receipt
  -> source-bound request receipts
  -> Fundamental v4 / I2 Industry / I3 Theme compiler
  -> replayed I4 and Decision v2 inputs
  -> deterministic I6 research capability
```

Raw Tushare rows cannot directly change Factor admission, Quant score, Fusion,
Bayesian posterior, Decision state, portfolio weights or the V17 active
pointer.

## One transport and exact decimals

All new Tushare consumers reuse
`quant_investor.v17_v4_runtime.tushare_https.OfficialTushareHttpsClient`.
The v2 adapters explicitly select strict decimal decoding. JSON fractional
values are decoded as `Decimal`; NaN and infinity are rejected. The token is
read only by the transport and is excluded from receipts, logs, exceptions and
content hashes.

The capability probe is implemented by
`scripts/probe_tushare_10000_capabilities.py`. Its default mode is dry-run and
does not read a token. A live probe requires an exact policy path/SHA and a new
private output root.

## Fundamental VIP v4

The v3 per-symbol provider contract is unchanged. VIP data uses a separate v4
closure with six source tables:

| Canonical table | Endpoint | Partition |
|---|---|---|
| `income` | `income_vip` | quarter end |
| `balancesheet` | `balancesheet_vip` | quarter end |
| `cashflow` | `cashflow_vip` | quarter end |
| `fina_indicator` | `fina_indicator_vip` | quarter end |
| `forecast` | `forecast_vip` | quarter end |
| `daily_basic` | `daily_basic` | canonical open session |

For as-of `T`, the daily window begins at `T - DateOffset(years=5)`. The five
financial tables begin two additional calendar years earlier. All boundaries
are inclusive. Each expected partition has one terminal identity; retries
increase its attempt count without creating a new partition.

The performance gate compares physical network calls, including retries:

```text
vip_network_attempts * 10 <= baseline_provider_calls_attempted
```

The baseline count comes from the fully validated v3 provider manifest. It is
not inferred from the number of symbols or logical requests.

### Shadow and promotion are separate

`scripts/run_tushare_vip_fundamental_shadow.py` performs the pre-promotion
workflow:

1. replay the exact v4 execution closure and comparison policy;
2. load the exact v3 accepted-raw checkpoint;
3. execute or resume every VIP partition through the official strict-decimal
   client;
4. compare baseline and VIP canonical row multisets;
5. run the existing formal v3 derivation once for each lane;
6. write the complete immutable provider-evidence fileset;
7. materialize an isolated staging generation only when every gate passes.

The script is dry-run by default. It cannot promote a pointer. An incomplete
network run preserves its checkpoint and returns `ACQUISITION_BLOCKED`. A raw,
derived, coverage or performance mismatch preserves durable evidence and
returns `RECONCILIATION_BLOCKED` without building a promotion-ready manifest.

`scripts/run_tushare_vip_fundamental_upgrade.py` is the separate promotion
operator. It accepts only an already sealed v4 staging generation and delegates
the commit point to the existing `promote_staged_fundamental_generation()`
helper. Promotion requires `--execute`, live authority and the exact expected
Fundamental pointer SHA. The durable journal distinguishes pre-CAS failure,
successful promotion, rollback and uncertain recovery. `PROMOTION_UNCERTAIN`
stops the workflow without cleanup or retry.

The v4 generation permanently carries baseline and VIP raw tables, request and
logical-coverage receipts, comparison policy, all comparison outputs,
reconciliation receipt and a closed fileset manifest. No promoted generation
depends on `/private/tmp`.

## I2 Industry compiler

The Industry source compiler consumes `index_classify` for the SW2021 L1/L2/L3
taxonomy and `index_member_all` in exact `L3 x {Y,N}` partitions. The hierarchy
is preserved, while a decision-facing company projects exactly one active L1
exposure with weight `1.000000000000`. Missing membership is `UNMAPPED`;
conflicting active L1 membership is `AMBIGUOUS`. Both block admission.
`stock_basic.industry` is display metadata only and cannot enter the membership
builder.

## I3 Theme compiler

DC is the primary Theme source. The compiler closes one registry snapshot and
one membership receipt for every company. TDX is used only for the sealed
ASCII-sorted set of companies whose DC coverage is incomplete. Complete DC
`NO_MEMBERSHIP` cannot be overwritten by TDX.

DC and TDX codes use separate namespaces. Same-name themes are not merged.
When a selected provider returns multiple memberships, deterministic equal
weights are quantized to 12 decimal places and the final ASCII-sorted member
receives the exact residual. Snapshot membership is effective only for that
snapshot date; it is not extrapolated into a historical interval.

Industry or Theme compilation failure does not roll back an independently
validated Fundamental promotion. It does keep Decision v2 and I6 blocked.

## I6 Market risk projection

`MarketRiskProjection.v1` accepts only exact, same-session canonical daily,
daily-basic, suspension and price-limit refs. Missing, stale or incomplete core
inputs block the projection. The result can only tighten portfolio policy:

```text
gross cap     = min(base, projection)
cash floor    = max(base, projection)
security cap  = min(base, projection)
vetoes        = union(base, projection)
```

`cyq_perf`, `moneyflow` and other diagnostics are outside the v1 schema and
cannot be injected as extra keys. A future use requires a new versioned
projection rather than a fallback.

## Operational sequence and stop boundary

```text
offline contract and package gates
  -> guarded capability probe
  -> same-as-of full-scope v3 baseline
  -> sealed v4 execution closure
  -> VIP shadow acquisition and reconciliation
  -> isolated staging generation
  -> final pointer and protected-state recheck
  -> journaled Fundamental CAS
  -> I2 and I3 source compilation
  -> Decision v2 and I6 replay
```

The Fundamental pointer may change only after raw and derived equality,
coverage closure, the 10% attempt gate, staging readback and expected-current
CAS all pass. This workflow never writes the V17 active pointer, Factor
registry, Factor production set, actual holdings or a trading system.

Before a real run, the exact scope, calendar, baseline provider manifest,
comparison policy, membership file, execution closure and all relevant SHA-256
values must exist. Missing evidence is a blocker; fixtures and stale
generations are not substitutes.
