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
- no V17 v4 artifact admission, reader expansion, writer, run command or
  empirical effectiveness conclusion.

## Scope and non-goals

Sprint 1A has no run orchestrator, schedule, artifact writer, output root,
model call, provider call, or portfolio surface. It does
not modify V15, V16, V17 v2/v3, V17 v4 runtime, Factor Governance, formal
activation, canary, promotion, the default selector, execution, broker, order,
or trade behavior.

The only initially allowed predecessor artifact is:

```text
myquant.v17.v4.regime-evidence.v1
```

It is self-contained and has no transitive artifact reference. Adding another
V17 v4 artifact requires a compatibility-policy revision that explicitly binds
its version, schema ID, identity field, path namespaces and every transitive
edge. Unknown versions and hidden references fail closed.

## Predecessor identity

The Phase-0 compatibility policy fixes:

```text
source commit:
  ec1370553fdf7ca0951ec4b03ea9fc426a872b4e
V17 v4 package manifest:
  fdc0aba035cdfff243df1a191431c84cfd7638fd0d94d877c7b37b29d5bc6875
V17 v4 runtime manifest:
  09700937c1fac82b2e3bbd405f1cbe7d31e71faea6a6c71e2d57d0c8c2b87b04
```

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
8. rejects unallowlisted transitive references.

Phase-0 closure limits are 64 MiB per JSON artifact, 256 MiB total, eight
levels and 64 nodes. The current self-contained allowlist uses one node.

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

## Acceptance and stop conditions

Sprint 1A is accepted only if package/runtime/predecessor verification, semantic
replay, reader positive and negative tests, authority/import/no-write boundary
tests, V15 public smoke, full V17 v4 regression, mypy, Black and
`git diff --check` pass.

Stop before any operational Factor adapter if any predecessor manifest drifts, an unallowlisted
reference is accepted, a resource limit is bypassed, a v4 runtime writer is
imported, a file is written, an existing CLI entrypoint changes, any authority
is true, or V15/V17 v4 regresses.
