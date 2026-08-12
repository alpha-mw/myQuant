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

The original v1 partition lane retains its physical-attempt gate for backward
replay compatibility:

```text
vip_network_attempts * 10 <= baseline_provider_calls_attempted
```

The baseline count comes from the fully validated v3 provider manifest. It is
not inferred from the number of symbols or logical requests.

The owner-authorized v2 official-partition lane remains replayable as
`OWNER_AUTHORIZED_EXACT_ANN_DATE_NO_RATIO_CAP`, but it only proves the exact
`ann_date` keyset for `fina_indicator_vip`. V3 also remains replayable.
V3 removed the empirical hot-window split for `balancesheet_vip`,
`cashflow_vip`, and `income_vip`: each endpoint is requested once for every
calendar announcement date from `financial_start` through `as_of`, inclusive,
using the documented `start_date=end_date` parameters and no inferred `period`,
company type, or report type. `fina_indicator_vip` continues to use its exact
`ann_date` parameter. The sealed `announcement_date_keyset_proofs` bind all four
endpoints to the same start, end, date count, ordered-keyset SHA-256, PIT cutoff,
and report-period comparison window. A missing date, resealed gap, different
cutoff, incomplete leaf probe, or non-v3 plan fails replay. The v3 mode is:

```text
OWNER_AUTHORIZED_EXACT_ANN_DATE_FULL_STATEMENT_KEYSET_NO_RATIO_CAP
```

The 2026-04-30 high-volume leaf probe returned 1,730 balance-sheet rows, 2,594
cash-flow rows, and 2,143 income rows; every response matched its schema and
announcement-date scope and reported `has_more=false`. This observation proves
that the documented leaf shape is executable, not that every historical leaf
will succeed. The acquisition still blocks on any future leaf with
`has_more=true`, a scope mismatch, duplicate conflict, or an oversized local
response.

The authorization removes only the network-attempt ratio. It does not relax the
sealed request upper bound, terminal receipt keyset, `has_more=false`, row scope,
duplicate, PIT window, restatement-winner, or baseline/VIP equality requirements.
Legacy v1 reconciliation continues to enforce the old ratio, and v2 evidence
cannot be silently reinterpreted as v3 evidence.

The first v3 shadow exposed an unresolved projection collision. Official
statement contracts provide `report_type` and `comp_type` filters, but v3
requested neither and retained only the narrow comparison projection. The saved
v3 evidence therefore cannot prove whether its duplicate rows came from distinct
report classes. New execution plans use v4 to close those documented dimensions
rather than silently deduplicating v3 output. V4 combines the complete calendar
announcement-date keyset with the documented latest consolidated report and
company-type dimensions:

```text
each exact ann_date
  x report_type=1
  x comp_type in {1,2,3,4}
```

V4 keeps exact duplicates as blockers. It does not relax raw multiset equality
or restatement-winner comparison. To remain below the 8 MiB artifact limit, the
plan seals a compact deterministic schedule, the complete date-keyset SHA-256,
the ordered company-type keyset, and a streaming SHA-256 over every regenerated
request row. Every validation replay regenerates and hashes the complete
topology. A live v4 plan cannot be sealed until nonempty, `has_more=false`
dimension probes exist for all three statement endpoints and all four company
types.

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
derived, coverage or legacy-v1 performance mismatch preserves durable evidence
and returns `RECONCILIATION_BLOCKED` without building a promotion-ready
manifest. The official-partition v2 path currently ends at a replayable shadow
comparison. It does not disguise official request receipts as legacy physical
receipts and does not yet construct a promotion-ready v4 fileset.

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

Comparison policy v2 remains replayable with its original byte identity. New
reconciliation runs use comparison policy v3, which seals an inclusive date
window for every table and applies that same window to both baseline and VIP
rows. Statement-table winner selection may explicitly use
`update_flag -> declared non-null completeness -> canonical ASCII` ordering;
other tables retain canonical ASCII ordering. These rules only make the window
and selected restatement deterministic. They do not forgive raw row, duplicate,
coverage, or business-value differences, which continue to block promotion.

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

The member endpoints return memberships from more than one board/index domain;
they do not accept the registry's `idx_type` filter. The compiler therefore
projects each complete member response onto the exact, complete `概念板块`
registry captured for the same provider and trade date. Registry-external rows
are retained in the immutable provider capture but cannot become Theme
memberships. A mixed response uses only its registry-bound Theme IDs. A DC
response containing only registry-external IDs enters the sealed TDX fallback;
a TDX response containing only registry-external IDs remains `AMBIGUOUS`.

DC and TDX codes use separate namespaces. Same-name themes are not merged.
When a selected provider returns multiple memberships, deterministic equal
weights are quantized to 12 decimal places and the final ASCII-sorted member
receives the exact residual. Snapshot membership is effective only for that
snapshot date; it is not extrapolated into a historical interval.

Full-A membership is stored as deterministic contiguous company shards. Each
shard contains at most 9,000 membership rows and must independently satisfy the
global 8 MiB artifact limit. The source receipt binds the ordered, nonoverlapping
catalog ranges and proves that their company coverage exactly closes the sealed
full-A keyset. The v1 singleton receipt remains replayable; new full-A builds use
the sharded v2 receipt.

Industry or Theme compilation failure does not roll back an independently
validated Fundamental promotion. It does keep Decision v2 and I6 blocked.

### Current sealed source result (2026-08-12)

The current full-A source compilation remains fail-closed where required:

- Fundamental: the v5 execution plan sealed 11,471 terminal requests over the
  complete announcement-date, report-period and open-session keysets. The live
  shadow completed all 11,471 requests. A zero-network replay after removing
  only byte-identical physical projection duplicates matches `balancesheet`,
  `cashflow`, `daily_basic`, `forecast` and `income` exactly. `fina_indicator`
  remains different by two multiset rows: Tushare now reports `ann_date`
  `20260312` for `920188.BJ` period `20250630`, while the older baseline captured
  `20251205`; all compared business values are equal. A separate one-request
  standard `fina_indicator` probe also returned `20260312`, proving current
  per-symbol/VIP agreement but not same-capture-time equality with the archived
  baseline. Reconciliation therefore remains `RECONCILIATION_BLOCKED`; no
  fileset, staging generation or pointer promotion was produced. The replay
  summary is
  `/private/tmp/myquant-r7-v5-reconciliation-diff-20260812-r3/summary.json`
  with SHA-256
  `1a81e3e2ac2778735d767bcf74cedaa93ee6920e4b21c87b4fa493ad4752f837`.
- Industry: the three SW2021 taxonomy partitions and all 692 sealed
  `L3 x {Y,N}` membership partitions replay successfully. Of the 5,485
  target-session PIT-active companies, 5,473 have one active L1 and 12 are
  `UNMAPPED`; none is `AMBIGUOUS`. The compiler remains `PARTIAL_BLOCKED`, and
  no Decision v2 admission is possible for those 12 subjects.
- Theme: a new exact-`20260807` DC plan completed 5,486 of 5,486 partitions with
  no incomplete response. The sealed fallback keyset contains only
  `603159.SH` and `605055.SH`; one TDX registry request and those two membership
  requests also completed. The resulting source receipt replays across eight
  catalog shards, covers all 5,485 target-session PIT-active companies, and
  reports no `NO_MEMBERSHIP`, `AMBIGUOUS` or `UNMAPPED` subject. Its source
  receipt ID is
  `039b8508c4dec4a969758f54771e2f40523c5b79c9eff4b787e948be3daac743`.
- No Theme membership was admitted by inference, name matching or a stale
  catalog. Theme identity is not a current blocker.
- Factor v5: no prospective admitted-set artifact exists for this target
  session. Historical or diagnostic factor results cannot substitute for that
  closure, so B0 Quant v5 cannot produce an initial pool.

These are valid partial outcomes, not acquisition failures. Theme is available
for the exact target session. Fundamental reconciliation, the missing Factor v5
prospective admitted set and the 12 Industry identity blockers keep B0,
Decision v2 and I6 unavailable.

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
coverage closure, the applicable sealed performance mode, staging readback and
expected-current CAS all pass. For v2, the applicable mode is the exact
announcement-date proof described above, not the legacy 10% ratio. This
workflow never writes the V17 active pointer, Factor registry, Factor
production set, actual holdings or a trading system.

Before a real run, the exact scope, calendar, baseline provider manifest,
comparison policy, membership file, execution closure and all relevant SHA-256
values must exist. Missing evidence is a blocker; fixtures and stale
generations are not substitutes.
