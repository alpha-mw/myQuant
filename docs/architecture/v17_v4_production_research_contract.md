# V17 v4 Production-Research Contract

## Decision and scope

The production-capable successor to the frozen `myquant.v17.v3` research
protocol is:

```text
protocol_id = myquant.v17.v4
contract_package = quant_investor.v17_v4_contract
runtime_package = quant_investor.v17_v4_runtime
explicit_cli = quant-investor-v17-v4
```

`myquant.v17.v3` remains byte-stable and permanently non-default. V4 copies
forward the quant-first pipeline and may reuse pure implementation helpers, but
it never relabels a v3 artifact, schema, receipt, pointer, or authority field.
A v3 artifact may appear only as an explicitly typed migration or comparison
input to a v4 artifact.

Production in this contract means the default research and portfolio runtime.
It never means automated execution:

| State | formal research publication | research runtime default | execution | broker | order | trade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `V15_DEFAULT` | false | false | false | false | false | false |
| `FORMAL_ACTIVE` | true | false | false | false | false | false |
| `DEFAULT_ELIGIBLE` | true | false | false | false | false | false |
| `CANARY` | true | false | false | false | false | false |
| `RESEARCH_DEFAULT_ACTIVE` | true | true | false | false | false | false |
| `ROLLED_BACK_TO_V15` | true | false | false | false | false | false |

The four execution-related fields are JSON Schema `const: false` in every v4
artifact. `formal_research_publication` and `research_runtime_default` are
separate fields. Formal activation cannot modify the shared default protocol
selector.

## Frozen predecessor

The v4 branch starts from:

```text
v3 source commit = 3a5d69934b686087d4d734f5b5e8254de9dd4a6d
v3 receipt commit = 2ad0618
v3 package manifest sha256 =
  e8dc183437a631f60aa06e17dd59f10dccce21dd3c4d576adfa479dad6b2674c
v3 runtime manifest sha256 =
  c7b86c3dc0d04a3dbe95ebd2034d76772a2c3312ac51d0cfe119258ad69eebf5
```

The complete predecessor inventory is
`docs/reviews/v17_v3_phase_a_baseline_files.sha256`.

## Owned roots and exact paths

V4 owns these roots:

```text
data/private/v17_v4_sources/
data/private/v17_v4_runs/
results/v17_v4_shadow/
results/v17_v4_formal_research/
results/v17_v4_canary/
```

Cross-protocol routing owns one separate control root:

```text
results/research_runtime_control/
```

The control root is not a model-artifact root. Only selector-transition
services may write selector, intent, receipt, and immutable protocol-target
paths. The V15 active-run publisher may CAS-write only
`active_runs/v15/{strategy_id}.json` while holding
`active_runs/v15/.locks/{strategy_id}.lock`. It cannot write any other control
path. All files and directories below the governed roots are owner-only `0600`
and `0700`, use descriptor-relative no-follow traversal, reject hard links,
and reject ASCII-casefold aliases.

The mutable and immutable paths are:

```text
results/v17_v4_formal_research/strategies/{strategy_id}/_active.json
results/v17_v4_formal_research/strategies/{strategy_id}/.active.lock
results/v17_v4_formal_research/strategies/{strategy_id}/receipts/{receipt_id}.json

results/v17_v4_canary/strategies/{strategy_id}/_eligible.json
results/v17_v4_canary/strategies/{strategy_id}/_current.json
results/v17_v4_canary/strategies/{strategy_id}/runs/{run_id}/
results/v17_v4_canary/strategies/{strategy_id}/receipts/{receipt_id}.json

results/research_runtime_control/default_protocol_selector.v1.json
results/research_runtime_control/.default-protocol-selector.lock
results/research_runtime_control/protocol_targets/v15/{byte_sha256}.json
results/research_runtime_control/protocol_targets/myquant-v17-v4/{byte_sha256}.json
results/research_runtime_control/active_runs/v15/{strategy_id}.json
results/research_runtime_control/active_runs/v15/.locks/{strategy_id}.lock
results/research_runtime_control/v15_runs/{run_id}/run.json
results/research_runtime_control/bootstrap_intents/{intent_id}.json
results/research_runtime_control/bootstrap_receipts/{receipt_id}.json
results/research_runtime_control/cutover_intents/{intent_id}.json
results/research_runtime_control/cutover_receipts/{receipt_id}.json
results/research_runtime_control/rollback_intents/{intent_id}.json
results/research_runtime_control/rollback_receipts/{receipt_id}.json
```

`default_protocol_selector.v1.json` is the only object whose successful CAS
can change the default research protocol. It points to one immutable protocol
target, never to a run. The target identifies the protocol-specific,
per-strategy active-run pointer, which in turn identifies one immutable run.
No v4 formal or canary service imports the selector writer.

## Artifact namespaces

V4 packages the following new schemas:

```text
myquant.v17.v4.common.schema.v1
myquant.v17.v4.authority.schema.v1
myquant.v17.v4.formal-activation-receipt.schema.v1
myquant.v17.v4.formal-active-pointer.schema.v1
myquant.v17.v4.formal-output.schema.v1
myquant.v17.v4.default-eligibility-receipt.schema.v1
myquant.v17.v4.default-eligible-pointer.schema.v1
myquant.v17.v4.canary-receipt.schema.v1
myquant.v17.v4.canary-pointer.schema.v1
myquant.v17.v4.dual-run-comparison.schema.v1
myquant.v17.v4.historical-canary-policy.schema.v1
myquant.v17.v4.pit-generation-catalog.schema.v1
myquant.v17.v4.pit-catalog-pointer.schema.v1
myquant.research-runtime.protocol-target.schema.v1
myquant.research-runtime.active-run-pointer.schema.v1
myquant.research-runtime.default-protocol-selector.schema.v1
myquant.research-runtime.route-transition-intent.schema.v1
myquant.research-runtime.route-bootstrap-receipt.schema.v1
myquant.research-runtime.cutover-receipt.schema.v1
myquant.research-runtime.rollback-receipt.schema.v1
```

The `myquant.research-runtime.*` schemas and their writers belong to the
neutral `quant_investor.research_runtime_control` package. The v4 package
consumes them but cannot redefine or publish them.

Factor production control packages:

```text
factor-governance-production-control.transaction.schema.v1
factor-governance-production-control.pre-activation-eligibility.schema.v1
factor-governance-production-control.authorization-receipt.schema.v1
factor-governance-production-control.rollback-authorization-receipt.schema.v1
factor-governance-production-control.wal-record.schema.v1
factor-governance-production-control.active-set-pointer.schema.v1
factor-governance-production-control.activation-receipt.schema.v1
factor-governance-production-control.rollback-receipt.schema.v1
```

These identities belong to
`quant_investor.factors.production_control_v1`, not to the v4 scaffold or V17
contract package.

The control module also closes its internal registry and exact-readback inputs
under `factor-governance-production-control.registry.v1`,
`factor-governance-production-control.readiness-readback.v1`,
`factor-governance-production-control.runtime-contract-set.v1`,
`factor-governance-production-control.v4-evidence-set.v1`, and
`factor-governance-production-control.v4-replay-set.v1`. Their references are
revalidated against the supplied exact source bytes before any transaction
intent or WAL file is written. Per-factor v4 evidence is also revalidated
through canonical replay readback from its owner-only immutable local file;
the evidence and replay sets are derived only after complete-chain, context,
quantitative, factor-identity, runtime-contract, file-hash, and semantic-hash
bindings pass. Coherent caller-declared evidence and replay digests are not an
admissible source.

Calibration artifacts use the independent v4 identities
`myquant.v17.v4.calibration-origin-inventory.v1`,
`myquant.v17.v4.calibration-receipt.v1`, and
`myquant.v17.v4.fusion-promotion-receipt.v1`. They are package-manifest and
runtime-build-manifest bound; no v3 calibration identity is relabelled.

The native Deep and production-portfolio closure uses:

```text
myquant.v17.v4.fusion-top24.v1
myquant.v17.v4.official-evidence.v1
myquant.v17.v4.issuer-dossier.v1
myquant.v17.v4.event-scan.v1
myquant.v17.v4.deep-evidence-bundle.v1
myquant.v17.v4.holdings-snapshot.v1
myquant.v17.v4.portfolio-risk-policy.v1
myquant.v17.v4.pretrade-permissions.v1
myquant.v17.v4.regime-evidence.v1
myquant.v17.v4.portfolio-overlay.v1
myquant.v17.v4.portfolio-output.v1
```

Every fusion Top24 row has exactly one Deep row. `COMPLETE` requires an
official filing or announcement, an issuer dossier no older than 30 calendar
days, and a regulatory/corporate-event scan no older than seven calendar
days. `UNAVAILABLE` is retained as a zero-target `BUY_VETO`; omission,
backfill, caller-supplied replacement targets, evidence-byte drift, or
symbol-domain drift fails closed.

Formal portfolio construction is `HOLDINGS_AWARE` only. The snapshot
reconciles position market value plus cash exactly to positive NAV and is no
more than one canonical session old. The freshness calculation binds the
exact active PIT generation catalog and its exact
`myquant.v17.v4.dataset.cn_open_day_calendar.v1` bytes, requires the same
history start and decision session, and requires at least 2,520 canonical
sessions. Runtime revalidates each calendar row's exact field set, CN market,
open-day flag, canonical weekday date, availability cutoff, uniqueness and
ordering; adjacent open sessions cannot be more than 15 calendar days apart.
It recomputes the natural-key hashes, row-set hash, latest availability and
row count from the exact dataset bytes and requires exact equality with the
PIT catalog summary. Permissions cover the exact union of Top24 and outside-
pool holdings. Every downstream read recomputes held state, current target and
lane from the exact holdings bytes; an outside-pool holding is
`REVIEW_ONLY_HOLDING` and cannot receive a positive target delta. A held name
with `can_sell=false` cannot be reduced by base risk, Macro, Markov or final
portfolio construction; a conflicting mandatory risk reduction fails closed.

The risk policy is effective and unexpired at the decision cutoff and binds
the exact permission and allocation rule hashes. Single-name, industry,
cluster, gross, cash-floor and turnover controls only shrink targets and never
redistribute released weight. Both Macro and Markov require typed AVAILABLE
evidence with a multiplier in `[0, 1]`; each output must be `APPLIED`, can
only multiply its exact predecessor targets downward, and releases the
difference to cash. A missing overlay is not a production no-op and blocks
formal activation.

The quant-first model artifacts are copied forward under `myquant.v17.v4.*`
identities. Their v3 discriminators, semantic hashes, and authority envelopes
are not accepted as v4 identities. The v4 package and runtime manifests bind
every copied-forward or new byte.

Every artifact reference contains:

```text
artifact_id
artifact_version
relative_path
byte_sha256
semantic_sha256
strategy_id
cutoff
```

References are exact, same-strategy, no-later-than-cutoff, and resolved before
any authority or mutable-pointer write.

Every v4 state-transition receipt contains exactly:

```text
version
protocol_version
receipt_id
strategy_id
recorded_at
from_state
to_state
status
authority
evidence_refs
semantic_sha256
```

The formal receipt additionally requires `cutoff`, `formal_output_ref`,
`source_locator_ref`, `factor_control_active_set_ref`,
`factor_control_activation_receipt_ref`, `quant_calibration_receipt_ref`,
`fundamental_calibration_receipt_ref`, `fusion_promotion_receipt_ref`,
`deep_bundle_ref`, `holdings_snapshot_ref`, `risk_policy_ref`,
`macro_overlay_ref`, `markov_overlay_ref`, `expected_pointer_sha256`,
`observed_pointer_sha256`, `proposed_pointer_sha256`, and
`post_readback_sha256`.

The eligibility receipt additionally requires `formal_active_pointer_ref`,
`public_surface_receipt_refs`, `validation_receipt_refs`,
`selector_bootstrap_receipt_ref`, `rollback_drill_receipt_ref`,
`expected_pointer_sha256`, `observed_pointer_sha256`,
`proposed_pointer_sha256`, and `post_readback_sha256`.

The canary-start receipt additionally requires `eligibility_pointer_ref`,
`historical_canary_policy_ref`, `v15_protocol_target_ref`,
`v15_active_run_pointer_ref`, `session_window`, `paired_run_ids`,
`expected_pointer_sha256`, `observed_pointer_sha256`,
`proposed_pointer_sha256`, and `post_readback_sha256`. The canary-complete or
canary-failed receipt additionally requires `comparison_refs`,
`completed_sessions`, `threshold_results`, and `side_effect_counters`.

The neutral protocol target contains exactly `version`, `target_id`,
`protocol_id`, `strategy_scope`, `active_run_pointer_template`,
`authority_ceiling`, and `semantic_sha256`. The authority ceiling fixes
execution, broker, order, and trade to false.

The neutral default selector contains exactly `version`, `selector_id`,
`status`, `protocol_target_ref`, `transition_intent_ref`, `updated_at`, and
`semantic_sha256`. A protocol active-run pointer contains exactly `version`,
`protocol_id`, `strategy_id`, `run_ref`, `cutoff`, `updated_at`, and
`semantic_sha256`.

Each bootstrap, cutover, or rollback intent contains exactly `version`,
`intent_id`, `transition`, `created_at`, `expected_selector_sha256`,
`expected_protocol_target_ref`, `proposed_protocol_target_ref`,
`expected_target_active_pointer_sha256`, `expected_target_run_ref`,
`proposed_selector_bytes_sha256`, `required_evidence_refs`, and
`semantic_sha256`. Its success, recovered, aborted, or blocked receipt
contains the same identity and transition fields plus `receipt_id`,
`recorded_at`, `observed_prevalue_sha256`, `observed_target_active_pointer_sha256`,
`post_readback_sha256`, `outcome`, and `semantic_sha256`; it never contains a
claimed post-readback value before the readback occurs.

## State transitions and receipts

### `V15_DEFAULT -> FORMAL_ACTIVE`

`myquant.v17.v4.formal-activation-receipt.v1` requires exact references to:

- v4 formal research output and its full source locator;
- accepted Quant timing, Fundamental forward, fusion calibration, and fusion
  promotion receipts;
- Factor v4 production activation receipt and registry readback;
- Top24 Deep evidence bundle;
- fresh holdings snapshot, permissions, and sealed risk policy;
- applied Macro and Markov overlays;
- v4 package and runtime manifests.

The formal active pointer has a per-strategy lock and CAS prevalue of either
`EMPTY` or the exact previous active-pointer SHA. The receipt is written
exactly once before the pointer. Pointer post-readback must equal the proposed
bytes. The transition never opens, locks, or writes
`results/research_runtime_control/`.

Failure status is `FORMAL_ACTIVATION_REJECTED`. A rejection may write only an
immutable rejection receipt. It cannot write an active pointer, eligibility
pointer, canary pointer, or default protocol selector.

### `FORMAL_ACTIVE -> DEFAULT_ELIGIBLE`

`myquant.v17.v4.default-eligibility-receipt.v1` requires:

- the current exact formal active receipt and pointer;
- CLI opt-in compatibility receipt;
- versioned Web DTO/route receipt;
- Dashboard Contract v4 receipt;
- schedule opt-in receipt;
- V15 regression and v4 full-test receipts;
- package/schema/manifest verification receipt;
- secret-scan receipt;
- zero broker/order/trade/execution import-and-call receipt;
- successful V15 selector bootstrap receipt;
- successful rollback dry-run and fault-injection receipt.

The eligibility pointer is a v4-only CAS from `EMPTY` or its exact prior SHA.
It grants no default authority and cannot write the shared protocol selector.

Failure status is `DEFAULT_ELIGIBILITY_BLOCKED`; it permits only an immutable
blocker receipt.

### `DEFAULT_ELIGIBLE -> CANARY`

A `CANARY_STARTED` receipt binds the current eligibility pointer, the exact
V15 protocol target and active-run pointer, the historical canary policy, a
fixed session window, and the first paired-run ID. It CAS advances only the v4
canary pointer.

A `CANARY_COMPLETED` receipt binds every paired V15/v4 run and one immutable
`myquant.v17.v4.dual-run-comparison.v1` artifact. It is valid only when every
threshold below passes. A failed comparison writes `CANARY_FAILED`, leaves
the default selector byte-for-byte unchanged, and prevents cutover.

### `CANARY -> RESEARCH_DEFAULT_ACTIVE`

The cutover receipt requires:

- current formal, eligibility, and completed-canary pointer refs;
- the exact current v4 active-run ref used by all public surfaces;
- expected and observed SHA-256 of `default_protocol_selector.v1.json`;
- the exact immutable V15 and v4 protocol-target refs;
- proposed v4 selector bytes and SHA-256;
- lock identity, cutover timestamp, and post-write readback SHA-256.

The immutable cutover intent is written before locking and contains the
expected V15 selector SHA, expected V15 target ref, proposed v4 selector bytes,
the expected v4 formal active-pointer SHA, its immutable run ref, and all
eligibility refs. Lock order is the v4 strategy formal-active lock followed by
the shared selector lock. Inside both locks, the service revalidates the
formal, eligibility, canary, active-run, immutable-run, selector, and protocol
target refs. The observed values must equal the caller-supplied expectations.
The selector is CAS-replaced once and exact post-write readback is required.
Only after successful readback is the immutable successful cutover receipt
written.

If the process crashes after CAS but before the success receipt, recovery
revalidates the immutable intent and the live selector. An exact proposed
selector produces one `CUTOVER_RECOVERED` receipt; an exact old selector
produces `CUTOVER_ABORTED`; any third state is a hard stop. Any pre-CAS
mismatch is `CUTOVER_CAS_BLOCKED` with no selector write.

### `RESEARCH_DEFAULT_ACTIVE -> ROLLED_BACK_TO_V15`

Rollback requires the original successful cutover receipt and the immutable
V15 protocol target. The expected prevalue is the exact v4 selector SHA
recorded by cutover. An immutable rollback intent is written before locking.
Lock order is the V15 strategy active-run lock followed by the shared selector
lock. Inside both locks, the service:

1. revalidates both immutable protocol targets;
2. confirms the live selector still equals the expected v4 SHA;
3. confirms the V15 standby active-run pointer resolves a current healthy run;
4. CAS-restores selector bytes that point to the V15 protocol target;
5. performs exact post-write readback;
6. writes the successful rollback receipt.

Rollback never deletes or edits v4 model evidence. Public consumers resolve
the restored selector, the immutable V15 target, and the continuously updated
V15 standby active-run pointer on their next request or scheduled run. It does
not restore a frozen pre-cutover V15 run. Rollback across an unrelated selector
version is forbidden. Crash recovery uses the same exact-old/exact-proposed/
third-state rule as cutover.

## Shared selector bootstrap

Before any public consumer reads the shared selector, a one-time bootstrap:

1. writes an immutable V15 protocol target identifying
   `active_runs/v15/{strategy_id}.json`;
2. writes an immutable V15 run binding exact readiness and portfolio/report
   artifacts;
3. CAS-creates the V15 active-run pointer from `EMPTY`;
4. writes a bootstrap intent for the selector;
5. CAS-creates `default_protocol_selector.v1.json` from `EMPTY`;
6. performs exact readback and writes the success receipt.

V15 continues to CAS-advance only its standby active-run pointer before and
after v4 cutover. V4 advances only its formal active-run pointer. All CLI, Web,
Dashboard, and schedule surfaces resolve and record four exact identities per
request: selector, immutable protocol target, protocol/strategy active-run
pointer, and immutable run. There is no missing-pointer fallback and no
scan-for-latest behavior.

## HTTPS provider gate

The only live Tushare transport admitted by v4 has this immutable allowlist:

```text
scheme = https
host = api.tushare.pro
port = 443 or omitted
path = /
query = empty
fragment = empty
userinfo = absent
redirects = forbidden, including same-host redirects
```

The implementation uses Python's default certificate store through
`ssl.create_default_context()` and exposes no `verify`, certificate, proxy, or
redirect override. Any 3xx response is a hard failure. HTTP, TLS downgrade,
host aliases, IP literals, custom provider URLs, and cross-domain redirects
are rejected before a socket is opened.

The only credential source is the process environment variable
`TUSHARE_TOKEN`. `TUSHARE_FALLBACK_TOKEN` is not read automatically. If a
backup credential is needed, the operator must explicitly install it as
`TUSHARE_TOKEN` before starting a run. The token, its hash, prefix, suffix, and
length are forbidden from logs, exceptions, manifests, artifacts, filenames,
HTTP-response captures, and telemetry.

HTTP responses are held in memory until the complete expected-key set and
response envelope validate. Larger multi-call acquisitions may spill only
response data, never request headers or bodies, to a new owner-only OS
temporary directory. No governed v4 path is created before all calls pass.
Failure removes the temporary directory and produces zero governed writes and
zero pointer changes.

Redaction tests inject distinct sample credentials into an environment value,
request body, response body, exception, and nested object. Error output must be
one of the static provider codes and contain none of the samples.

## Data admission contracts

All source artifacts live below `data/private/v17_v4_sources/`. Every
generation has an immutable data file, an exact expected-key inventory, a
manifest, and a CAS pointer. A generation is admitted only after a fresh
readback independently recomputes file SHA-256, semantic SHA-256, schema,
natural-key uniqueness, expected-key equality, coverage, availability time,
and parent lineage.

### PIT generation catalog

```text
schema = myquant.v17.v4.pit-generation-catalog.v1
generation =
  data/private/v17_v4_sources/pit_catalog/generations/{generation_id}.json
pointer =
  data/private/v17_v4_sources/pit_catalog/_latest.json
```

The catalog binds exact refs for `cn_open_day_calendar`, `market_bars`,
`universe_membership`, `pit_fundamentals`, `corporate_actions`,
`benchmark_total_return`, and `official_delisting_cash`.

Natural keys are:

- bars: `(security_code, trade_date)`;
- membership:
  `(security_code, valid_from, valid_to, published_at, revision_id)`;
- PIT fundamentals:
  `(security_code, report_period, announce_date, revision_id, field_id)`;
- corporate actions:
  `(security_code, ex_date, action_type, announced_at, revision_id)`;
- official delisting cash:
  `(security_code, terminal_session, currency, official_source_id)`;
- calendar and benchmark: `(market_or_benchmark_id, session)`.

All `available_at` values must be no later than the decision cutoff. The
calendar and bars must cover the exact decision session and the
machine-derived historical start. Membership, fundamentals, and corporate
actions must cover that same interval. The delisting-cash expected keys are
derived from the admitted official delisting events intersecting any
calibration label window; the observed set must equal that inventory exactly.

Missing, extra, duplicate, future-available, or conflicting keys produce
`SOURCE_ADMISSION_BLOCKED` and no catalog or source pointer write.

The current security-directory acquisition calls the official HTTPS
`stock_basic` endpoint separately for `L`, `D`, and `P`. Standard A-share
codes are admitted as bitemporal membership rows. A source row outside the
closed six-digit `BJ/SH/SZ` code namespace is not silently discarded: it is
retained as a hash-bound `UNSUPPORTED_SECURITY_CODE` exclusion. The current
fetch timestamp is `available_at`; it is never backdated to create historical
PIT authority.

Catalog publication first revalidates the complete seven-role closure in
memory, then independently streams the SHA-256 readback of every dataset and
expected-key inventory. It writes the immutable generation before CAS
advancing only
`data/private/v17_v4_sources/pit_catalog/_latest.json` under its dedicated
source lock. It cannot import or write the neutral research-default selector.

### Factor v4

FactorGovernanceProtocol v4 remains a research scaffold. Its protocol hash and
its permanent `production_apply_enabled=false` and
`new_risk_authorized=false` fields are not changed.

Production-research activation is owned by a separate protocol:

```text
protocol_id = factor-governance-production-control.v1
module = quant_investor.factors.production_control_v1
root = data/private/factor_governance_production_control_v1/
registry = registry/current.json
registry_lock = registry/.lock
active_set_pointer = active_sets/_active.json
active_set_lock = active_sets/.lock
wal = wal/{transaction_id}.jsonl
v4_activation_receipt = receipts/v4_activations/{receipt_id}.json
production_control_receipt = receipts/control_activations/{receipt_id}.json
rollback_receipt = receipts/rollbacks/{receipt_id}.json
rollback_authorization = rollback_authorizations/{receipt_id}.json
registry_snapshot = registry/snapshots/{byte_sha256}.json
```

Before mutation, production control builds a
`factor-governance-production-control.pre-activation-eligibility.v1` artifact.
It references the original v4 evidence, replay, transaction plan, activation
request, proposed registry, and runtime contracts by exact bytes and SHA. It
recomputes every v4 health and set gate except the not-yet-existing activation
receipt and requires `activation_receipt_missing` to be the only remaining
readiness blocker.

The transaction also requires an independent, exact authorization receipt with
`authorization_scope=factor_v4_production_research_activation`, an expiry, the
transaction-plan SHA, proposed-registry SHA, proposed-factor-set SHA, runtime
contracts SHA, and authorized operator identity. This authorization receipt is
not a v4 activation receipt and cannot claim that activation was performed.

Production control then performs WAL prepare, registry CAS, and exact
readback. Only after registry readback does it issue the
`factor-governance-activation-receipt.v4` whose
`activation_performed=true` describes that completed registry activation. It
recomputes full v4 readiness with that receipt, then CAS-advances the active-set
pointer and writes the production-control success receipt. Its rollback
restores the exact before-registry bytes and predecessor active-set pointer
under inverse CAS. Rollback requires its own intent-before-lock authorization
receipt scoped to `factor_v4_production_research_rollback`; its validity
window, transaction, successful control receipt, current hashes, and exact
restore hashes must all match before either lock is acquired.

Lock order is registry lock then active-set lock. The immutable transaction
intent contains `transaction_id`, `as_of`, `authorized_by`,
`authorization_scope=factor_v4_production_research_activation`,
`v4_pre_activation_eligibility_ref`, `v4_replay_ref`,
`v4_transaction_plan_ref`, `v4_activation_request_ref`,
`production_control_authorization_receipt_ref`, `expected_registry_sha256`,
`proposed_registry_sha256`, `expected_active_set_sha256`,
`proposed_active_set_sha256`, `runtime_contracts_sha256`, and
`semantic_sha256`. WAL records `PREPARED`, `REGISTRY_COMMITTED`,
`V4_RECEIPT_ISSUED`, `READINESS_RECOMPUTED`, `ACTIVE_SET_COMMITTED`, or
`ROLLED_BACK` with exact before/after hashes. The production-control success
receipt is written only after both CAS readbacks. A crash after registry CAS
but before active-set CAS must either idempotently issue or revalidate the
exact v4 receipt, recompute readiness, and finish the exact transaction, or
apply the inverse registry CAS before any active-set pointer can advance.

The only authority it grants to V17 is
`active_for_production_research=true`. Its execution, broker, order, trade, and
account-new-risk fields are permanently false. V17 requires the exact
production-control active-set pointer and receipt; it never requires or
pretends that the v4 scaffold's permanent false fields became true.

The v4 source binding also points to the exact registry readback and a
recomputed readiness record. It rejects quality, shadow, provisional,
caller-declared readiness, and shadow activation receipts.

Passing requires:

- state `production_factor` for every factor;
- five through ten production factors and every factor healthy;
- at least three families, and five distinct families when there are exactly
  five factors;
- unique nonempty slots;
- nonzero finite weights, factor absolute weight at most 20%, and family
  absolute weight at most 35%;
- Gates 1-8, authoritative maturity, family BH `q <= 0.10`, verified runtime
  and replay identity, and fresh healthy observations;
- activation receipt and registry readback with exact matching hashes;
- production-control receipt with
  `active_for_production_research=true` and exact active-set readback;
- readiness `source_as_of` no more than three canonical open sessions behind
  the decision session and no more than eight calendar days behind the cutoff.

The original v4 `production_apply_enabled=false` and
`new_risk_authorized=false` values remain mandatory. Any attempt to flip them,
or any missing production-control receipt, shadow activation receipt, or
readback mismatch, is a hard blocker. A report-only factor set cannot be
relabelled. Historical calibration must reference the factor set actually
effective at each origin; current production-control activation cannot create
retroactive origin authority.

The current legacy registry contains one old `production_factor`, but that row
does not carry the v4 family, slot, Gates 1-8, BH, runtime-contract, canonical
replay, and fresh-health closure. It therefore contributes zero factors to the
v4 production set. Production control permits the reviewed underfilled floor
of five healthy factors and continues accelerated mining toward ten; it never
relabels legacy or `production_candidate` rows to satisfy that floor.
Freshness is recomputed from each factor's hash-bound strict open-session
calendar and `health.source_as_of`; a caller-supplied `fresh=true` flag is
insufficient.

### Calibration

```text
quant input span = at least 1260 canonical open sessions at every origin
fundamental input span = at least 2520 canonical open sessions at every origin
origin schedule = last Shanghai open session of 120 consecutive calendar months
label maturity = exact 60-session and 252-session total-return labels
common READY minimum = 24 names at every origin
```

The closure window is exactly the last 120 consecutive origins. Earlier,
equally governed antecedent origins may be supplied solely to make each
fold's 60-origin training window leakage-free after enforcing the 252-session
label maturity boundary. Antecedents do not replace, shorten, or create gaps
inside the 120-origin closure window.

Every origin reconstructs the exact PIT catalog, PRESELECT locator, initial
pool, same-pool Quant/Fundamental branches, benchmark, corporate actions, and
official terminal cash. Each referenced artifact is read back as canonical
bytes and must match its byte and semantic hashes. PRESELECT locators, initial
pools, Quant/Fundamental branch outputs, and total-return labels use closed
native v4 schemas; a metadata-only `calibration_binding` cannot stand in for
their payloads. Benchmark, corporate-action, and official-delisting-cash refs
must be byte-for-byte identical to the corresponding refs in the validated PIT
catalog. Pool order, scores, label rows, delisting cash, and the historical
Factor production-control active set are then recomputed from the native
payloads. No month may be skipped. The last 60 mature origins form five
consecutive 12-month outer folds; each fold trains only on the latest 60
earlier origins whose 252-session labels end before the fold starts.

All receipts bind the fixed 10,000-replicate circular 12-month block-bootstrap
matrix with PCG64 seed `170317`. Fusion promotion requires one-sided 95% lower
bounds strictly above `0.50` for 60-session hit rate and strictly above zero
for the 252-session q25 statistic.

Any origin, source, branch, pool, label, schedule, span, bootstrap, or receipt
drift is `CALIBRATION_CLOSURE_BLOCKED` and permits no promotion.

### Deep, holdings, risk, Macro, and Markov

Deep requires exactly one typed row for every fusion Top24 name. Each row binds
the admitted official filing/announcement evidence available by cutoff, an
issuer dossier no older than 30 calendar days, and a regulatory/corporate-event
scan no older than seven calendar days. An unavailable row remains a BUY veto;
an omitted row is structural failure.

Production-research portfolios require `HOLDINGS_AWARE`. The holdings snapshot
must be exact, owner-only, available by cutoff, and no more than one canonical
open session old. `MODEL_ONLY_NO_PRIVATE_HOLDINGS` is shadow-only.

The risk policy must be sealed, effective on the decision session, unexpired,
and hash-bound to permissions and the portfolio. Macro and Markov must both be
typed `APPLIED` overlays for default eligibility. Each overlay may only reduce
name weights or gross exposure and release weight to cash. `UNAVAILABLE_NO_OP`
is permitted in shadow, but blocks `FORMAL_ACTIVE` and `DEFAULT_ELIGIBLE`.

## Public surfaces

Before cutover:

- `quant-investor ... --decision-protocol v15` remains accepted and default;
- `quant-investor ... --decision-protocol v17-v4` is explicit opt-in only;
- Web uses a new `/api/v4/research-runs` route and v4 DTO namespace;
- Dashboard Contract v4 is a new schema and does not rename
  `v15_run_readiness`;
- the v4 schedule writes only canary artifacts;
- canary Web and Dashboard views are read-only and visibly labelled `CANARY`.

The V15 DTO, Dashboard Contract v3, V15 command spelling, and V15 schedule
remain byte-compatible. After selector bootstrap, all surfaces resolve the same
exact selector, immutable protocol target, per-strategy active-run pointer,
and immutable run. Cutover changes only the protocol selector, not four
separate defaults or a frozen run reference.

## Canary thresholds

Canary has two separately sealed stages.

### Historical dual replay policy

Before operational canary, 60 consecutive mature month-end origins are run
through V15 and v4 at matching decision cutoffs. A read-only comparison
sidecar binds each protocol's actual source closure and classifies the pair:

- `COMPARABLE` only when cutoff, canonical calendar, underlying market-bar
  bytes, benchmark bytes, and holdings snapshot are exact matches;
- `NON_COMPARABLE` otherwise, with exact differing refs and no inferred
  equivalence.

Any `NON_COMPARABLE` origin blocks policy sealing. V15 is not claimed to
consume the v4 PIT catalog.

The immutable `myquant.v17.v4.historical-canary-policy.v1` records the
empirical distributions and pass bands for:

- V15 Top12 recall inside v4 Top24 and rank overlap;
- gross and cash exposure difference;
- L1 portfolio distance and maximum common-name target difference;
- turnover;
- industry and cluster exposure differences;
- held-name positive increases;
- trim and exit disagreements;
- Deep, permission, Macro, and Markov veto invariants.

The policy uses the observed 5th percentile as the minimum overlap/recall band
and the observed 95th percentile as the maximum absolute-difference band.
Risk invariants are absolute: v4 gross may not exceed V15 gross by more than
`0.05`, a veto may not have a positive target delta, and Macro/Markov may not
increase a name or gross exposure. The policy version, all 60 pair refs,
quantile method, and resulting numeric bands are hash-bound before operational
canary starts.

### Five-session operational canary

The operational window is five consecutive Shanghai open sessions with five
successful paired runs. Each run writes a comparison sidecar and must be
`COMPARABLE` under the same lower-level rules used by historical replay.

All conditions must pass:

- five of five V15 and v4 paired runs complete without structural, data,
  contract, or timeout failure;
- every model-difference metric remains inside the sealed historical policy;
- every absolute risk invariant passes;
- each v4 run latency is no more than twice its paired V15 latency and no more
  than 30 minutes;
- analysis-time provider, LLM control, broker, execution, order, and trade
  side-effect counters are all zero;
- selector, target, active-run, formal, eligibility, canary, data, and Factor
  pointer CAS mismatch counters are zero;
- every public consumer records the same selector and protocol-target SHA for
  the session.

The five-session stage is an operational smoke test, not a statistical model
validation claim. Any failure is `CANARY_FAILED` and leaves V15 as the
default. The entire five-session window restarts after the blocker is
corrected; passing and failing sessions cannot be cherry-picked.

## Validation and stop conditions

The cutover gate requires:

- all v4 unit, integration, contract, schema, manifest, storage, CAS, repair,
  secret-redaction, and failure-injection tests;
- the entire V15 public/default regression and frozen v2/v3 boundary tests;
- CLI, Web, Dashboard v4, and schedule compatibility tests;
- Factor v4 exact-readback and no-relabel tests;
- 120-origin calibration replay;
- five-session canary;
- successful forward cutover and rollback rehearsal against a temporary
  control root;
- `scripts/staged_upgrade_quality_gate.sh`;
- a final code review and independent completion verification.

Any missing data evidence, unhealthy Factor v4 set, incomplete calibration,
stale holdings, missing Deep or applied overlays, public-consumer mismatch,
canary failure, untested rollback, dirty worktree, test failure, secret hit, or
non-false execution/broker/order/trade authority stops the process at V15
default.
