# Production future Calendar and Morning closure

Status: Architect APPROVE_WITH_CHANGES incorporated; Critic APPROVE. Implementation in progress. No implementation,
provider invocation, production install, or scheduler switch is authorized by this
file itself. The full Phase0–15 objective remains unchanged. Current frozen
synthetic acceptance d2b2398 continues independently and must not be hot-patched.

## Required end state and current evidence

Original design `daily_evidence_dag_chat_design.md` Phase10 requires Morning to
consume previous completed EOD plus same-day quote and owner thresholds, with no
maintenance. Phase12 requires the existing EOD/fallback/Morning tasks to use that
path. Synthetic success alone cannot satisfy the production path.

Current source:
- `market/next_session_acquisition.py` verifies the running installed release,
  invokes the fixed native Calendar operator and returns validated capture refs.
- `market/tushare_calendar_authority.py` uses fixed documentation HTTPS GET and
  OfficialTushareHttpsClient requests for SSE/SZSE/BSE. Its execution count of four
  is a contract constant, not independent transport provenance.
- `market/next_session_proof.py` deliberately publishes/reads synthetic v1 only.
- `market/future_calendar_context.py` supports DISABLED/SYNTHETIC_FIXTURE_ONLY;
  current configured producer carries only fixture provenance.
- `scripts/daily_morning_consumer.py` correctly refuses synthetic/live-ineligible
  EOD. `daily_morning_seal.py` invokes PREFLIGHT itself: the direct SEAL refusal
  in prepare_morning_consumer is not by itself a broken sealing route.

## Proposed minimal intended path

1. Add an explicitly versioned production provenance/proof contract, separate
   from unchanged synthetic v1, and native reader dispatch by exact schema.
   A caller-supplied `live=true`, transport enum, context mode, raw capture files
   or a counter cannot create live eligibility.
2. Establish provenance within the verified installed capture invocation. Add a
   private scoped transport recorder at the existing fixed documentation and
   official API boundaries. It records only successful exact requests, sanitized
   fixed destination/method/API/parameters, response SHA, actual UTC start/end,
   and TLS/redirect validation facts. No token, credential, request body containing
   token, arbitrary response headers or exception text is persisted.
   Expected set: one pinned docs GET and three trade_cal requests with exact
   exchanges, start and +21-day end. No caller callback/transport/clock argument.
   Guard against active fixture capability and substituted known adapter/clock
   entry points. The scope is ordinary verified-process integrity, not protection
   against a malicious process owner or arbitrary Python instrumentation.
3. Seal provenance durably in the installed operator invocation, binding exact
   install input, verified release, capture execution/success refs and raw hashes.
   Use native immutable publication and deterministic paths. Recovery may only
   read/replay retained provenance; it cannot infer or manufacture it after a
   capture. A crash before provenance publication leaves retained capture without
   admission, with typed PROVENANCE_UNAVAILABLE and zero recapture on recovery.
4. Production proof builder replays native capture, exact recorder receipt and
   chronology before publishing a schema-versioned proof. Preserve SSE/SZSE
   agreement, full horizon, EOD-open and earliest next-open selection, native root
   identity/release validation and false trading/portfolio authority. Publication
   confers Calendar source provenance only, never whole EOD or Morning eligibility.
5. Add exact v3 configured context/state for production mode rather than widening
   v2 fixture contracts. Keep v1/v2 byte semantics intact. Fresh Core invokes capture and publication; recovery invokes a separate retained-
   evidence reader/binder with no capture, recorder activation or transport
   construction reachable. State persists at each durable boundary.
   Production state uses transport_evidence_ref, never fixture_transport_evidence_ref.
   DISABLED remains explicit. Public synthetic mode still requires private fixture
   scope; production mode requires installed capture provenance and rejects that
   scope before writes/provider calls. Mode is routing, not admission authority.
6. Existing future_calendar_outputs/Core handoff/selected Calendar binds the exact
   production publication. No scan/latest fallback. No attachment/replacement
   after Calendar selection or EOD seal. Completed replay remains read-only.
7. Morning reader dispatches by proof schema, rechecks proof custody alongside
   other source refs, and requires production Calendar provenance for PREFLIGHT
   and SEAL in addition to existing native EOD/ledger/PIT/date/quote/owner-policy
   gates. REPLAY remains research-only. Use existing seal/history route; do not
   remove the direct-SEAL guard as a shortcut. No producer reachable from Morning.

## Required contract detail before implementation

Architect completed with APPROVE_WITH_CHANGES. Exact contract proposals and
required amendments are incorporated below. Critic approved consistency, implementation readiness, acceptance coverage and
stop conditions. Local implementation proceeds under the existing task authority;
provider calls, production installation and scheduler application remain separate.

## Recovery and immutability

| Interrupted boundary | Permitted recovery | Network calls |
| --- | --- | --- |
| Core persisted, no capture | Retain typed acquisition absence/failure | 0 |
| Native capture sealed, no transport receipt | PROVENANCE_UNAVAILABLE, retain capture | 0 |
| Transport receipt sealed, state not bound | Exact deterministic replay then bind | 0 |
| Proof sealed, state not bound | Exact retained proof adoption | 0 |
| FUTURE_BOUND before Core handoff | Existing native Core recovery with exact refs | 0 |
| Calendar selected or EOD completed | Read only; no new proof or failure attachment | 0 |

Do not downgrade security/identity/race/unknown contract failures into ordinary
missing provenance. Keep typed native acquisition failures and raw failure refs.

## Acceptance and stop conditions

- Focused tests: all existing synthetic proofs unchanged; caller labels/copies,
  wrong release, wrong endpoint/API/params/raw, partial call set, fixture scope,
  swapped adapters, regressed/future time and mutated selected proof rejected.
- Actual native offline transport-boundary fixture tests must remain explicitly
  tests. They exercise control flow but are never labelled real provider evidence.
- Installed native offline test of capture -> provenance -> proof -> configured
  Core -> EOD -> Morning uses explicit test evidence isolation. Its result cannot
  satisfy real unattended or live-source acceptance.
- Every recovery boundary above with network/producers forbidden and exact
  workspace inventory; no duplicate acquisition, Factor generation or publication.
- Preserve synthetic driver results and source versions separately. New runtime
  edits require new frozen final verification; do not restart current jobs.
- Production capability implementation/installation, actual governed source run,
  scheduler update, independent unattended receipt and observation are distinct
  gates. Actual source calls/deployment require their established exact authority;
  production config remains incomplete until real inputs/owner facts are supplied.
- Stop implementation if truthful provenance needs a new external service,
  undocumented trust claim, new credential destination, widened transport API,
  weaker native validation or false test-to-live promotion. Re-review the design.

## Integration inventory from current source

- DailyFactorLoop has explicit v2 branches at recovery preflight, `_save_state`,
  fresh Core preflight, state creation, handoff publication and `_recover_core`.
  A new context/state must cover all six, not just widen shared validation.
- `future_calendar_outputs` already checks proof publication before Calendar
  terminal, Factor deployed release and current-day native Calendar policy.
  Preserve these checks for each supported proof version.
- Morning currently folds `future["synthetic"]` into overall synthetic state,
  but does not independently consume `live_eligible`. Add explicit provenance
  admission and exact future-proof recheck; a false synthetic flag is insufficient.
- The actual seal route rewrites SEAL to PREFLIGHT for native preparation and
  validates report bytes/threshold bindings itself. Preserve this existing route
  and direct-SEAL refusal. History must replay the same versioned Calendar proof.

## Exact contract proposal for review (not yet approved)

All objects reject unknown keys. Workspace refs are exact `{path, sha256}`;
legacy native refs remain exact `{relative_path, byte_sha256}`. Days are YYYYMMDD,
SHA values lowercase 64 hex, commits lowercase 40 hex, UTC timestamps canonical
with seconds and Z. FALSE_AUTHORITY remains the existing exact all-false map.

### Production transport evidence v1

`cn-calendar-production-transport.v1` exact keys:
`schema_version`, `eod_trade_date`, `release_install_input_ref`, `release_ref`,
`execution_ref`, `success_ref`, `capture_root_name`, `recorder_identity`, `events`, `event_set_sha256`,
`recorded_at`, `authority`.

`events` has exactly four ordered entries. Each entry has exact keys:
`ordinal`, `kind`, `method`, `scheme`, `host`, `port`, `path`, `api_name`,
`parameters`, `expected_fields`, `request_started_at`, `response_completed_at`,
`response_bytes`, `response_sha256`, `http_status`, `tls_verified`, `redirect_count`.
Ordinals are exact integers 0–3 (bool rejected), scheme https, port443; host/path
are exactly the existing DOCS_URL or official API endpoint components.
response_bytes is a positive bounded integer matching retained raw length.
event_set_sha256 hashes canonical JSON of the exact events array.
recorder_identity is exactly `{release_ref, module_file_refs}`; module_file_refs
is a sorted unique array of `{path, sha256}` for the installed recorder, transport,
and Calendar authority modules. Paths are fixed package-relative names and SHAs
are checked against the verified immutable installed release manifest, not caller
provided identities. recorder_identity.release_ref equals the receipt release_ref.

Entry0 is DOCUMENTATION / GET / exact existing DOCS_URL, null api_name,
empty parameters, empty expected_fields, HTTP200, TLStrue, redirects0; its SHA
must equal native documentation.raw. Entries1–3 are PROVIDER / POST / exact
existing official API endpoint / trade_cal / native expected fields. Parameters
are exact native start_date/end_date/exchange for SSE,SZSE,BSE in native order.
Their SHAs and response_bytes must equal the corresponding native raw responses. Validation
replays all native provider contracts; merely matching these values is not proof
that an arbitrary caller used the installed operator.

Only the private capture-bound recorder can supply events to the production
writer in that same invocation. Neither context JSON nor a public proof-publisher
argument may supply an events list or assertion of observed transport. Runtime
verification and known adapter identity checks happen before acquisition. A fixed
count, posthoc raw-file inspection or absence of fixture flag is insufficient.
Recorder reset is guaranteed on every exception; nested scopes reject.

Deterministic evidence path:
`<day-root>/calendar-future/production-transport/<execution-byte-sha>.json`.
Its recorded_at is actual clock after native success; all events must fall inside
native acquisition bounds. The exact timestamp relation must account for native
second precision without relaxing future-time or reversed-order checks.

### Production provenance and proof v2

`cn-calendar-acquisition-provenance.v2` exact keys:
`schema_version`, `execution_ref`, `success_ref`, `release_ref`,
`release_install_input_ref`, `transport_evidence_ref`, `issuer_route`,
`transport_mode`, `observed_network_call_count`, `recorded_at`, `authority`.
Routes are code-owned INSTALLED_OFFICIAL_CALENDAR / OFFICIAL_HTTPS; count4 is
rederived from validated events. Provenance chronology is native success <=
transport receipt <= provenance <= proof publication <= reader current clock.

`cn-next-session-calendar-proof.v2` has the same exact field names as v1,
plus exact `transport_evidence_ref` and `source_classification` fields,
with schema_version v2, synthetic false and source_classification fixed to
INSTALLED_NATIVE_HTTPS_OBSERVED; acquisition_provenance_ref must point
to v2 provenance. Native refs/projection/release/time/policy/limitations must
replay exactly; public caller false flags never bypass the provenance reader.
`cn-next-session-calendar-proof-publication.v2` exact keys remain
schema_version, proof_ref, proof_sealed_at, authority, with schema v2.

Use separate `production-provenance/<execution-byte-sha>.json`; retain the existing
content-addressed `proofs/<proof-sha>.json` and `publications/<proof-sha>.json`
layout. Schema dispatch is exact and cross-version pairings reject. No v1 rewrite,
no conversion of v1 synthetic proof into v2, no proof publication from old capture
without same-invocation retained production transport evidence.

Reader output keeps existing keys; synthetic false/live_eligible true means only
Calendar provenance passed. consumer_admission remains false: the full EOD and
Morning validators retain admission authority. Morning explicitly checks both
flags plus its existing EOD gates and replays this exact future proof in recheck.

### Configured context and state v3

Context `cn-daily-factor-loop.v3` has exactly the v2 field names; accepted mode
values are DISABLED and PRODUCTION_INSTALLED_CAPTURE. v2 SYNTHETIC_FIXTURE_ONLY semantics
remain unchanged. The mode selects a route; it is never a provenance assertion.
All current release/input/path checks remain. Production entry rejects private
fixture scope before creating directories, loading credentials or making calls.

State `cn-daily-factor-state.v3` has exactly the v2 field names except
fixture_transport_evidence_ref is replaced by transport_evidence_ref and schema
is v3. Phases are CORE_READY, CAPTURE_SEALED, TRANSPORT_BOUND, FUTURE_BOUND,
CORE_PUBLISHED. CORE_READY has all future/handoff refs null; CAPTURE_SEALED
requires capture pair but evidence/proof/failure/handoff null; TRANSPORT_BOUND
requires capture pair and transport evidence, with proof/failure/handoff null.
FUTURE_BOUND requires
exactly one proof/failure for enabled mode; a proof always requires capture pair
and evidence. CORE_PUBLISHED additionally requires exact handoff. DISABLED has
no future refs. Context SHA and trade date bind every recovery read before writes.
Fresh and recovery handoff must preserve exact production refs.

### Validation errors and testing boundaries

Retain existing projection errors, native failure receipts and immutable/path/SHA
security errors. PROVENANCE_UNAVAILABLE means capture exists but same-invocation
transport evidence is absent; it does not authorize re-acquisition or overwriting.
Malformed/mismatched evidence, swapped adapters, fixture-in-production or release
identity failure must remain refusal, not be treated as ordinary source absence.
Additional error names are enumerated in the governing amendments below.

Offline tests can exercise native publication/readback and rejection boundaries
with explicit transport test seams, but must disclose the seam and cannot count
as real provider provenance. A live-source/unattended acceptance remains required.
No tests should confer production authority on copied synthetic objects.

## Architect amendments incorporated (governing details)

Architect APPROVE_WITH_CHANGES accepted direction and required the following
constraints, which govern any less precise wording above:

- The factual claim is: the verified installed process observed successful HTTPS
  responses from fixed Tushare hosts using existing CA/TLS policy. It is not
  exchange-official provenance, hostile-process protection, or elevated BSE
  authority. Existing source limitations and all-false authority persist.
- API response events are emitted inside `_fetch_raw` only after status200,
  redirect/size checks and exact raw bytes; documentation events are emitted in
  the corresponding fixed GET boundary. Safe API/params/expected-fields identity
  is staged privately before `_fetch_raw`, then associated with that successful
  event. Failed attempts cannot become successful events. Extra calls reject.
- Before acquiring, reject active fixture scope or substitution of production
  recorder/capture/transport entrypoints, API connection/TLS-context aliases,
  documentation connection/context, and clock aliases. Identity checks are
  ordinary verified-process guards; they do not create a Python security sandbox.
- Fresh production binder owns a private recorder scope across acquisition and
  transport publication. Native acquisition returns the capture; binder persists
  CAPTURE_SEALED before publishing the sibling transport receipt, then persists
  TRANSPORT_BOUND before proof publication. Opaque recorder data is consumed only
  by the private writer, never accepted as public caller events. Scope resets
  on all exits. No leaf is appended to the sealed native capture root.
- Recovery is a separate reader/binder that cannot activate recorder or transport.
  CORE_READY/CAPTURE_SEALED may discover only exact deterministic native refs. A
  complete retained transport receipt permits adoption; absent evidence yields
  typed PROVENANCE_UNAVAILABLE. Damaged/mismatched/extra-event evidence blocks
  with a contract/security error rather than being treated as ordinary absence.
- Proof v2 directly binds transport_evidence_ref as well as its provenance ref.
  Their identities must be equal; reader verifies all event/raw/release/module
  bindings before live_eligible true. Unknown and cross-version schema pairs
  reject with no fallback. Synthetic v1 publication bytes/readback stay intact.
- Production context mode is exactly PRODUCTION_INSTALLED_CAPTURE or DISABLED.
  v3 DISABLED bypasses CAPTURE_SEALED/TRANSPORT_BOUND; no future evidence is allowed.
- Morning PREFLIGHT requires native production v2 live_eligible true and false
  synthetic, exact release/Calendar bindings and a second exact proof readback.
  Existing EOD/ledger/quote/risk gates remain. SEAL continues through its existing
  PREFLIGHT route. REPLAY may accept either proof version but stays research-only.

Additional error names proposed for exact dispatch:
`NEXT_SESSION_TRANSPORT_ROUTE_UNAVAILABLE` (substitution/fixture),
`NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID` (missing/extra/order/type),
`NEXT_SESSION_TRANSPORT_BINDING_INVALID` (release/module/raw/request mismatch),
`NEXT_SESSION_TRANSPORT_CHRONOLOGY_INVALID`,
`NEXT_SESSION_PRODUCTION_SCHEMA_INVALID`,
`MORNING_PRODUCTION_CALENDAR_REQUIRED`.
These errors block; only an absent durable evidence receipt uses the existing
PROVENANCE_UNAVAILABLE typed capability failure.

Additional required cases: recorder inactive outside production scope; known
transport monkeypatch rejected before calls; failed attempts not counted; extra
request rejected; response length/hash equals capture; each of five state-boundary
crashes recovers without network; synthetic v1 unchanged; unknown/cross-version
schema rejected; production proof with a non-synthetic contemporaneous EOD reaches
PREFLIGHT, and existing SEAL invokes that route. Controlled offline seams must be
explicit and do not satisfy real provider or unattended-run acceptance.
