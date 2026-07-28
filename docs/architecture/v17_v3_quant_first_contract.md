# V17 Protocol v3 Quant-First Contract

## Status and authority

`myquant.v17.v3` is an additive, isolated research protocol. It does not
replace or wrap the V15 production/default runtime and it does not reinterpret
the byte-sealed `myquant.v17.v2` evidence.

The only formal authority that v3 may acquire is publication of a
hash-bound research result:

```text
formal_research_publication_authority = true
execution_authority = false
production_default = false
broker_authority = false
order_authority = false
trade_authority = false
```

The normal `quant-investor` entrypoint remains V15. V3 is available only from
the explicit `quant-investor-v17-v3` entrypoint.

## Pipeline

The v3 pipeline is fixed:

```text
admitted PIT sources
  -> Quant-only full-A preselector
  -> sealed organic selection pool (maximum 500)
  -> independent Quant and Fundamental branches over the same pool
  -> calibrated deterministic fusion (maximum 24)
  -> Fundamental deep research reduction/veto
  -> permissions, pretrade checks and base-risk research portfolio
  -> Macro/Markov monotonic risk overlay
  -> final permissions/pretrade revalidation and portfolio research output
```

`review_only_holdings` is separate from `selection_pool_symbols`. A holding
outside the organic pool may be reviewed, locked, trimmed or exited by an
independent deterministic risk rule, but it cannot enter fusion or receive a
positive target delta.

Macro and Markov are not stock-selection branches. They may only tighten
exposure, cash, name, industry, cluster, turnover or stress constraints. LLM
content is sealed-evidence input to Fundamental deep research only; it cannot
change membership, rank, permissions, pointers or trades.

## Preselector and branch identity

The formal preselector consumes only:

- bitemporal `CN/full_a` membership valid at the decision cutoff;
- strict canonical Parquet market data;
- research eligibility, tradability and liquidity evidence;
- the FactorGovernanceProtocol v4 production inventory effective at the
  origin cutoff.

A current-cutoff shadow run may instead use the separately typed
`PROVISIONAL_RESEARCH` v17 baseline. That baseline is an exact packaged
inventory of price/volume definitions and is not FactorGovernanceProtocol v4
qualification, shadow observation, production readiness or activation
evidence. It is valid only with the closed `SHADOW_CURRENT_PRESELECT` and
`SHADOW_CURRENT_MODEL_PORTFOLIO` source profiles. The baseline reference and
mode are carried through the initial pool, portfolio and terminal shadow
artifacts. Formal analysis, calibration, activation and current-formal reads
reject it explicitly.

The history requirement is the maximum of 120 canonical sessions and every
active factor's sealed lookback or warmup. Each factor must meet its sealed
coverage and cross-section contract. Every READY security has a complete,
finite and non-constant active-factor matrix; missing values are explicit
exclusions and are never neutral-imputed.

The preselector and the downstream Quant branch must have disjoint definition
hashes, families and lineages. A formal run hard-stops if no eligible disjoint
inventory exists.

Both branch outputs contain exactly one ordered record for every organic pool
security, including UNAVAILABLE records. They bind the same locator, cutoff,
pool path, byte SHA, semantic SHA, count, symbol order and policy SHA. Missing,
duplicate, extra or reordered records stop fusion before publication.

Source closure construction is staged and acyclic. A sealed PRESELECT locator
and its RAW/input closure produce the immutable initial-pool artifact. A later
ANALYZE locator binds that exact pool and the two branch outputs. Both branches
bind the PRESELECT locator recorded by the pool, never the later ANALYZE
locator that contains them. Runtime replays preselection and requires the
reconstructed pool to be byte- and semantic-identical before consuming either
branch. The terminal output separately binds the current ANALYZE locator and
the resulting downstream artifacts. Forward references from an artifact to a
locator or manifest that already contains that artifact are forbidden.

The admitted source topology has two closed layers. A `RAW` manifest binds the
PIT market, membership, governance and Fundamental evidence. A
`DERIVED_CLOSURE` manifest binds its exact parent RAW manifest and only the
typed preselection, branch, calibration, Deep and portfolio artifacts derived
from it. Runtime analysis consumes only roles admitted through one of these
closed manifests; free-form or unregistered role names are rejected.

`RAW` closures have an explicit profile. `HISTORICAL_FORMAL` retains the full
corporate-action requirements, and calibration/promotion additionally require
benchmark total-return and official terminal-delisting cash. `SHADOW_CURRENT`
binds current calendar, bars, PIT fundamentals, PIT membership and the exact
Factor v4 readiness capture, but it cannot contain holdings. The two
shadow-current phases cannot be substituted into formal or calibration paths.

## Fusion and calibration

Fusion first forms the common READY domain. Both percentiles are computed over
that exact domain:

```text
quant_scalar = Quant three-factor composite_score
fundamental_scalar = Fundamental five-pillar total_score
percentile = (average_ascending_rank - 1) / (n - 1)
percentile(n=1) = 1
fusion_score = wq * quant_percentile + (1 - wq) * fundamental_percentile
```

Normative scores are Decimal values quantized to `1e-12` with
`ROUND_HALF_EVEN`. The final tie tuple is fusion score descending, then
security code ascending. A READY fusion selects exactly 24 names; fewer than
24 common READY names fail closed. Deep-research rejection never triggers
backfill.

The formal weight is selected from `0.25, 0.30, ..., 0.75`. Historical origins
are the last canonical Shanghai open session of each calendar month. The five
outer folds are the 60 consecutive mature month ends ending at the latest
origin whose 252-session label is available at the calibration cutoff,
partitioned into five consecutive 12-month blocks.

For each outer fold, weight selection uses only the latest 60 consecutive
month ends before the fold whose 252-session labels end strictly before the
fold begins. Every scheduled training, outer and active-refit month must
reconstruct the complete source closure, exact PRESELECT locator, exact
initial-pool artifact, two same-pool branches and labels. Quant and Fundamental
may not bind distinct pool artifacts even when their symbol arrays are
identical. Calibration origins require at least 24 common READY securities.
Every calibration receipt carries a nonempty, exact evidence closure. The
`QUANT_TIMING`, `FUNDAMENTAL_FORWARD` and `FUSION_PROMOTION` receipts bind
locators whose manifest phases are respectively
`QUANT_TIMING_CALIBRATION`, `FUNDAMENTAL_FORWARD_CALIBRATION` and
`FUSION_PROMOTION`; a PRESELECT locator cannot substitute for any calibration
gate. Activation and current-result reads resolve all 120 monthly
Quant/Fundamental branch, initial-pool and PRESELECT-locator refs again; a
missing, distinct or drifted historical byte fails closed.

The monthly statistics are:

- mean of the Top24 indicators that 60-session total-return excess return is
  positive;
- linear q25 of the Top24 252-session total-return excess returns.

Stock and `H00300.CSI` pre-tax total-return labels use identical start and end
sessions and official terminal cash for delisted securities.

Every 60-month series uses one shared 10,000-replicate circular moving-block
bootstrap index matrix: 12-month blocks, five starts per replicate, PCG64 seed
`170317`. The formal promotion gate uses only the concatenated outer results
and requires one-sided 95% lower bounds above `0.50` for 60-session hit rate
and above zero for mean monthly 252-session q25.

The receipt must disclose that the 60 overlapping monthly outcomes contain
only five effective 12-month blocks and therefore establish a
`research_screening_bound`, not stable production or execution evidence. A
50/50 result without a promotion receipt is shadow-only and cannot be consumed
by formal publication.

## Monotonic downstream rules

For an unheld candidate, incomplete Deep evidence or a severe red flag may
revoke BUY eligibility. For an existing holding, Deep may disable additional
buying or lock the current weight but cannot create a reduction or exit.

Every fusion Top24 name must have exactly one typed Deep row. An explicit
unavailable row for an unheld name produces `BUY_VETO` and a zero target.
Omitting the row or the Deep artifact is structural incompleteness and
hard-stops; names are never backfilled.

Deep adjustment is multiplicative in return magnitude, not an absolute
percentage-point change:

```text
penalty = clamp(0.10 * max(-weighted_signal, 0), 0, 0.10)
adjusted_q25 = base_q25 - abs(base_q25) * penalty
```

The Macro/Markov baseline is the deterministic target after Deep,
permissions, pretrade and the base risk policy, but before either overlay.
Every overlaid security weight must be no larger than its baseline weight,
gross exposure cannot increase, membership cannot expand and released weight
goes only to cash. Renormalization into another security is forbidden.

The current model-only allocation policy assigns
`min(0.03, floor8(0.72 / N))` before Deep, with no redistribution after a
veto or permission failure. A fully vetoed Top24 therefore reconciles to 24
zero targets and 100% cash. This is a complete model calculation, not an
account recommendation. `MODEL_ONLY_NO_PRIVATE_HOLDINGS` is shadow-only and
does not fabricate an empty holdings snapshot. The separate `HOLDINGS_AWARE`
path retains exact snapshot and session-freshness checks.

Macro and Markov are reported as `APPLIED` only when exact typed overlay
evidence is admitted and monotonic validation succeeds. Otherwise each stage
is labelled `UNAVAILABLE_NO_OP`; omission is never described as having run.

## State, activation and isolation

V3 owns only:

```text
data/private/v17_v3_sources/
data/private/v17_v3_runs/
results/v17_v3_shadow/
results/v17_v3_formal_research/
```

V2/V3 cross-version roots, imports, artifacts, latest pointers, casefold
aliases and symlink aliases are rejected before any write.

An immutable, typed formal-research candidate is finalized before activation
and has no publication authority by itself. Activation requires an accepted
FUSION_PROMOTION receipt for the same strategy and cutoff. The ACTIVE receipt
binds both exact artifacts, then the activation pointer advances from absent
to ACTIVE and formal latest publishes that exact candidate. The promotion's
package-manifest SHA transitively binds a packaged runtime-build manifest with
the byte SHA-256 of every v3 runtime and algorithm Python source. Activation
and current-result reads revalidate that build identity.

Each new cutoff reruns the complete promotion/health gate. A failed first
activation writes `ACTIVATION_REJECTED`, leaves that cutoff's activation
pointer absent and preserves any existing formal-latest pointer byte-for-byte.
REVOKED is valid only as a successor to an existing ACTIVE receipt; it advances
the activation pointer to a persistent REVOKED tombstone before formal latest
advances to its REVOKED tombstone. Consumers revalidate the activation pointer,
so a crash between those two CAS operations fails closed. The same strategy and
cutoff cannot be reactivated. If ACTIVE pointer publication succeeded but formal
latest is absent, revocation creates the REVOKED formal-latest tombstone from
the absent state; it never overwrites a newer-cutoff latest pointer.

Operations that touch activation or formal latest take the strategy-level
formal-latest lock first and the strategy-plus-cutoff activation lock second.
Formal latest advances monotonically by cutoff. A losing CAS may leave only
content-addressed `IMMUTABLE_UNPUBLISHED_EVIDENCE`; it cannot become current.
Consumers revalidate the exact ACTIVE pointer before presenting a formal
artifact as current.

## Delivery boundary

Phase A implements the additive contract, pure runtime and synthetic/offline
verification. Its terminal delivery status is
`NOT_ACTIVATED_DATA_BLOCKED`.

Phase B may begin only when true bitemporal history, origin-time factor
governance receipts, the machine-derived raw history span, Quant 1260-session
calibration, Fundamental 2520-session calibration and a fresh holdings
snapshot are all available. Provider acquisition is not part of Phase A.
