# V17 v5 Investment Intelligence Contract

## Goal

`myquant.v17.v5` is a research-only successor that may derive new, replayable
intelligence evidence from exact immutable V17 v4 Forward Evidence artifacts.
It does not replace, relabel, mutate, or activate V17 v4.

Phase 0 established:

- a separate `v17_v5_contract` / `v17_v5_runtime` boundary;
- closed package and runtime manifests;
- permanently false authority;
- a pinned V17 v4 predecessor policy;
- a bounded, no-follow, read-only compatibility reader;
- `status` and `verify`.

Sprint 1A adds one library-only surface:

- a sealed descriptive Factor diagnostic policy and schema;
- a pure in-memory maturity, cross-sectional RankIC, coverage and replay kernel;
- a recursively verified, allowlisted V17 v4 Forward Evidence read closure;
- a sealed V4-to-V5 Factor evidence adapter policy;
- a pure descriptive Factor lifecycle diagnostic;
- no writer, run command, governance action or empirical effectiveness
  conclusion.

Sprint 1B adds:

- a sealed origin-regime diagnostic policy;
- a V4 regime-evidence adapter that reuses the Sprint-1A bounded closure reader;
- immutable origin bindings across Factor observation, 20-session matured
  label, Factor implementation identity and origin-available regime evidence;
- unconditional and by-regime descriptive RankIC statistics;
- a stdout-only CLI diagnostic surface;
- no online weights, tier changes, lifecycle advice or authority.

Sprint 1E-0B adds:

- an exact V5 predecessor pin to V4 bounded-chain and V2-publication-block
  commit `73c5b6eea6c60d9a31865e176646687ffeee9d6a`;
- V4 `myquant.v17.v4.regime-evidence.v3` composite-finality validation through
  the existing bounded read-only adapter;
- a v3 diagnostic policy that accepts only finalized, filtered-causal,
  prior-session, full-market, sealed-argmax V4 regime evidence;
- stricter origin binding: regime decision/effective session must equal the
  Factor origin session and observed-through must be the immediately preceding
  Shanghai open session;
- deterministic continuity eligibility: `CONTIGUOUS` and `ROLLOVER` are
  conditionable; `GENESIS`, `RECOVERY`, and `未知` are not;
- no V4 producer modification, artifact creation, online weights, tier changes,
  lifecycle advice or authority.

## Scope and non-goals

Sprint 1E-0B has no run orchestrator, schedule, artifact writer, output root,
model call, provider call, or portfolio surface. It does
not modify V15, V16, V17 v2/v3, V17 v4 runtime, Factor Governance, formal
activation, canary, promotion, the default selector, execution, broker, order,
or trade behavior.

The only root-admissible predecessor artifacts are:

```text
myquant.v17.v4.forward-evaluation-receipt.v1
myquant.v17.v4.regime-evidence.v1
myquant.v17.v4.regime-evidence.v2
myquant.v17.v4.regime-evidence.v3
```

V1 and V2 remain valid for integrity checks but are not conditioning-eligible.
Only finalized V3 can enter origin-regime conditioning.

The evaluation receipt expands only through the sealed compatibility-policy
graph: origin and existing-factor inventories, 20-session labels, observation
run and request, factor observations and factor set, source locator, input
bundle and slice manifests, stage receipts/outputs and immutable Parquet
leaves. Raw upstream `source_refs` are terminal provenance bindings; the
adapter does not dereference them or treat them as statistical observations.
Unknown versions, partial refs, hidden refs and undeclared edges fail closed.

## Predecessor identity

The Sprint 1E-0B compatibility policy fixes:

```text
source commit:
  73c5b6eea6c60d9a31865e176646687ffeee9d6a
V17 v4 package manifest:
  270c863fdcc2b092265444db9cc2fac9e3e19e1ef5fb2a36ddde6b47e443a1ff
V17 v4 runtime manifest:
  7c7dc183a419623542fb1d8b95d092283c948c46a804eedd8424f931645f3a28
```

The predecessor mechanism is `WORKTREE_COLOCATED_PREDECESSOR`. V5 keeps the
exact V4 commit as a merge parent, verifies the colocated V4
package/runtime manifests, and does not hand-copy or reinterpret V4 producer
files.

V5 output can never present a v4 artifact as a v5 artifact. Future v5 receipts
must preserve the source protocol, source version, relative path, byte SHA-256,
semantic SHA-256 and adapter/policy identity.

## Read protocol

The compatibility reader accepts an absolute canonical workspace root, a
workspace-relative path, exact byte SHA-256, strategy identity and decision
cutoff. It:

1. loads the sealed compatibility policy and verifies the pinned v4 package
   and runtime manifests;
2. rejects absolute, escaping, symlinked, hard-linked, casefold-ambiguous and
   non-allowlisted artifact paths;
3. uses descriptor-relative no-follow traversal and checks
   `lstat/open/fstat` identity;
4. checks inode, size, link count and timestamps before and after the bounded
   read;
5. recomputes byte SHA-256;
6. executes the exact V17 v4 schema and semantic validator;
7. enforces strategy, cutoff, availability and authority closure;
8. recursively follows every declared exact artifact ref;
9. validates registered JSON, generic terminal audit and Parquet metadata
   according to the per-version policy;
10. rejects cycles, conflicting duplicate nodes, partial refs and undeclared
    transitive references.

Sprint-1A closure limits are 128 MiB per artifact, 512 MiB total, 32 levels and
512 nodes. Parquet leaves are additionally limited to 10,000,000 rows and
4,096 row groups. Limits are fail-closed and do not authorize source
discovery.

V5 runtime imports the V17 v4 contract only. Importing any
`quant_investor.v17_v4_runtime` module is forbidden and covered by an AST
boundary test.

The Sprint-1A diagnostic kernel also does not import the shared
`forward_evaluator`, Factor Governance registry, production-control or tier
allocation modules. Its runtime source is byte-bound by the V5 runtime
manifest.

The package manifest byte-binds packaged JSON and inventories the contract
Python filenames. It does not byte-bind the contract Python contents because
`resources.py` contains the package manifest's self-binding constant. Git
checkpoint review and tests remain the source binding for those contract
modules; `verify` must not be described as a complete byte seal of contract
Python.

## Sprint 1E-0B Regime evidence v3 adapter

The V5 adapter accepts only explicit path plus SHA-256 inputs. It never scans
for a latest artifact, calls the V4 producer, downloads market data, reads V15
mutable history, converts older evidence to V3, or reconstructs regime from raw
data.

The policy-eligible V4 V3 evidence must satisfy:

```text
version = myquant.v17.v4.regime-evidence.v3
inference_kind = FILTERED_CAUSAL
smoothing_used = false
publication_phase = PRIOR_SESSION_EFFECTIVE_NEXT_SESSION
scope_kind = FULL_MARKET
hard_state_derivation = SEALED_ARGMAX_POLICY_V1
no_retroactive_causal_backfill = true
```

State probabilities must contain exactly the sealed state order, use canonical
12-decimal strings, be finite and in `[0, 1]`, and sum to
`1.000000000000`. Evidence also needs its explicit current checkpoint in the
direct source closure, and the two artifacts must agree on all duplicated
session, segment, posterior, hard-state, commitment and accumulator fields.
V5 does not recompute posterior, argmax, commitments, chain digest, or
continuity. It does not issue a finality receipt or recursively traverse older
checkpoints. `CONTIGUOUS` and `ROLLOVER` may be conditionable; `GENESIS`,
`RECOVERY`, and `未知` are not.

For a Factor origin, the V3 evidence must bind exactly:

```text
regime.decision_session = factor_origin.decision_session
regime.effective_session = factor_origin.decision_session
regime.published_at <= factor_origin.cutoff
regime.available_at <= factor_origin.cutoff
regime.observed_through_session = previous_open_session(factor_origin.decision_session)
```

The adapter rejects orphan evidence, stale fallback, label-end regime, horizon
transition regime, multiple V3 evidences for one origin, future publication,
and any posterior-only or smoothed state. V1 remains integrity-checkable and
ineligible. V2 remains integrity-checkable but returns
`REGIME_EVIDENCE_V2_NON_DEPLOYABLE`.

## Authority

Every V17 v5 authority field remains permanently false:

```text
formal research publication
research runtime default
formal activation
canary
promotion
provider
LLM
Factor Governance write
portfolio
selector
execution
broker
order
trade
```

CLI state remains:

```text
default_protocol_state = V15_DEFAULT
global_activation_state = INACTIVE
run_state = INACTIVE
```

## Roadmap and evidence gates

Later work is split into independently reviewed Sprints:

1. Quant Intelligence diagnostics.
2. Industry Intelligence.
3. Fundamental Intelligence.
4. Theme Intelligence.
5. Integrated research evaluation and reporting.

The current v4 IndustryContext, ThemeExposure and Regime evidence are inputs,
not functionality to duplicate. No Sprint may automatically change a Factor
tier or weight, write Factor Governance, or produce a portfolio action.

Any empirical conclusion must be evaluated separately for each exact
`(strategy, factor-set, adapter-policy, source-lineage)` stratum. The baseline
requires at least 60 naturally matured 20-session origins and at least 100
comparable symbols per origin. Overlapping 20-session labels require HAC,
block-bootstrap, or a registered non-overlap sensitivity check. Regime and
interaction analyses require their own preregistered sample minima.

Before those gates pass, outputs may only be:

```text
UNOBSERVED
ACCUMULATING
UNAVAILABLE
```

They cannot claim factor, industry, fundamental, theme, regime or strategy
effectiveness.

The complete Sprint-1D origin-regime causality and statistical contract is
defined in
`docs/architecture/v17_v5_factor_regime_diagnostics.md`.

## Sprint 1A Factor diagnostic contract

`myquant.v17.v5.factor-diagnostic.v1` is a descriptive artifact, not a receipt.
It is returned only in memory and has no governed output path. The only states
are:

- `UNOBSERVED`: a complete exact stratum and sealed calendar are supplied, but
  there are zero naturally matured origins;
- `ACCUMULATING`: at least one naturally matured origin is supplied;
- `UNAVAILABLE`: an explicit prerequisite such as calendar, label contract or
  lineage is absent. This state has no stratum SHA, origins or statistics.

Malformed identity, SHA, date, cutoff, conflicting duplicate, mixed session
identity, noncanonical decimal, future label or resource-limit input raises
`FactorDiagnosticError` with exit code 2. It is never converted to
`UNAVAILABLE`.

Each observed diagnostic fixes one exact stratum:

```text
strategy_id
factor_name
factor_definition_sha256
factor_implementation_sha256
factor_set_sha256
quant_policy_sha256
adapter_policy_byte_sha256
source_lineage_series_sha256
market_calendar_sha256
horizon_sessions = 20
```

`source_lineage_series_sha256` is the stable series/policy identity.
`evidence_lineage_sha256` remains distinct for each origin. The runtime
computes maturity from the ordered Shanghai open-session calendar, exact
20-session end, label `available_at` and evaluation cutoff; it does not accept
a caller-provided maturity boolean.

Factor and forward-return values are canonical finite decimal strings. The
comparable domain is their exact symbol intersection. RankIC uses ASCII symbol
ordering, exact Decimal average ranks for ties, `ROUND_HALF_EVEN` and 12 output
decimal places. Constant factor or return vectors produce an origin-level
unavailable metric; nonfinite or noncanonical values fail closed.

`descriptive_coverage_minimum_met=true` only means at least 60 RankIC-available
naturally matured origins with at least 100 comparable symbols each. It always
coexists with:

```text
gate_scope = DESCRIPTIVE_ONLY
inference_gate_passed = false
inference_eligible = false
effectiveness_claimed = false
effectiveness_conclusion = null
```

Tier, weight, promotion, Factor Governance and all operational authority remain
false. Overlap-robust inference is explicitly deferred.

The in-memory limits are 4,096 origins, 10,000 symbols per origin and 2,000,000
total supplied symbol rows.

## V4 Factor evidence adapter

`adapt_v4_factor_evidence` accepts only `V4CompatibilityRead` values produced
by the compatibility reader, an exact evaluation cutoff, one factor identity
and a sealed Shanghai open-session sequence. It does not discover files.

For each usable origin it requires:

- a `factor_evaluation_receipt` with one matching 20-session origin;
- exact request/run/inventory/label/observation refs and one inactive
  `FORWARD_EVIDENCE_ACTIVE` run;
- one selected Factor row whose definition, implementation and factor-set
  identities agree with the receipt lineage;
- one current source locator and input bundle with exact required and
  neutralizer field identities;
- a complete Factor observation and complete, naturally matured 20-session
  label whose source-lineage SHA and return arithmetic replay exactly.

The series SHA is derived from the stable locator/bundle versions, required
field set, Factor-slice field set and neutralizer field set. Daily snapshot
identity is not used as the series identity; each origin evidence SHA instead
binds the exact request, run, Factor set, Factor observation, source locator
and label refs. Structurally valid missing evidence becomes `UNAVAILABLE` or
`UNOBSERVED`. Hash, identity, cutoff, calendar, arithmetic, stratum or
authority contradictions raise `V4FactorAdapterError` with exit code 2 and
produce no artifact.

No real persisted V4 evaluation receipt or finalized Regime Evidence V3 is
available for Sprint 1E-0B. Therefore the current evidence state remains
`UNAVAILABLE`; no Factor effectiveness claim is made.

## Factor lifecycle diagnostic

`myquant.v17.v5.factor-lifecycle-diagnostic.v1` aggregates only sealed V5
Factor diagnostics for the same exact factor and stratum. Its outputs are
limited to `UNOBSERVED`, `ACCUMULATING` and `UNAVAILABLE`, with origin-count
and first/last-session description. Empty input is malformed; an explicit
missing prerequisite must use the unavailable builder.

The lifecycle artifact contains no tier, weight, action, effectiveness or
promotion conclusion. `lifecycle_action` and `lifecycle_conclusion` are always
null, and every authority field remains false.

## Acceptance and stop conditions

Sprint 1E-0B is accepted only if package/runtime/predecessor verification,
semantic replay, V3 adapter positive and adversarial tests, origin-binding
tests, authority/import/no-write boundary tests, V15 public smoke, full V17 v4
regression, mypy, Black and `git diff --check` pass.

Stop before any operational writer or governance integration if any predecessor
manifest drifts, an unallowlisted reference is accepted, a resource limit is
bypassed, a v4 runtime writer is imported, a file is written, an existing CLI
entrypoint changes, any authority is true, or V15/V17 v4 regresses.
