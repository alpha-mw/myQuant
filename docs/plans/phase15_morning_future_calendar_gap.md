# Configured daily path to next-session Morning evidence

The c3 installed synthetic run completed five daily stages and a complete read-only
aggregate replay. Its Morning consumer stopped because the sealed Calendar node
has no next_session_calendar_proof. Preserve those exact immutable results.

## Established cause

DailyFactorLoop.core_completed publishes Core through publish_core_pool without
next_session_calendar_proof_ref/failure_ref. Its recovery path also omits them.
CorePoolContext and native-input/materialization already support those optional
references, and the Morning consumer requires the proof. Optional absence is a
valid EOD-only state, but it does not fulfill this task's EOD-to-Morning goal.
The older direct native fixture explicitly produced and passed a future proof;
the configured path did not. Do not fix this by editing sealed terminals or
allowing Morning to infer dates or acquire provider data.

## Proposed implementation boundary

1. Inspect the existing installed next-session acquisition and proof publisher,
   their release/policy/time bindings, and the Factor loop context contract.
   Determine the exact existing source for each required input before edits.
2. Connect the existing producer capability before Calendar/Core publication on
   the configured fresh path. Retain explicit EOD-only operation when this
   capability is not configured. Any added configuration is versioned and
   validated; never silently add live calls to unrelated maintenance users.
3. Persist exact future-proof/failure refs in the existing recovery state before
   Core publication. Recovery reuses and verifies retained evidence, never
   recaptures under a finalized request or changes a sealed Calendar terminal.
4. Pass these refs through both fresh and recovery Core paths and verify they
   survive materialization into the sealed EOD. Missing/invalid proof continues
   to block Morning. Preserve all SHA, release, Calendar and time checks.
5. Update the configured offline fixture using only simulated external transport
   and disclosed clock; do not mock the producer, Core, proof verifier or Morning.
   Reuse the repaired read-only replay guard for Morning, whose old harness also
   forbids the native Factor read lock.

## Verification and stop conditions

Focused tests: opt-in and absent capability; future proof/failure propagation;
wrong date/release/SHA/time; interruption before/after proof publication; recovery
without provider calls; Morning rejection without proof and actual acceptance
with the bound proof. Assess full CI based on runtime/schema impact.
Do not overwrite original c3 evidence or claim its missing Morning proof exists.
A new installed version/scenario will be necessary if runtime contracts change;
state its scope and do not merge old and new runs into one uninterrupted claim.
No production cutover, real provider call, holdings or broker mutation is part of
this repair. Stop and reassess if the existing producer cannot supply a required
binding, rather than inventing a policy/date or weakening the consumer.

Architect review precedes Critic review before public configuration/schema or
security-sensitive implementation. This plan is pending those reviews.

## Producer lane constraint discovered during review

The existing public proof publisher is explicitly synthetic-only, and its reader
accepts only SYNTHETIC_FIXTURE provenance. capture_next_session_calendar acquires
through installed transport but returns no consumer proof/eligibility. The repair
must not label this as a live producer or silently grant LIVE provenance. Separate
synthetic configured-chain closure from any additional production-capable issuer
work, preserving the full goal and its uncompleted live gates.

## Revised implementation contract after Architect review

Architect verdict REVISE has been incorporated; this section governs the earlier
proposal where more specific. Implement configured SYNTHETIC REPLAY closure only;
the full objective still includes the separately uncompleted live issuer gate.

- Introduce exact cn-daily-factor-loop.v2 with the existing required release and
  capture fields and next_session_calendar_mode enum DISABLED or
  SYNTHETIC_FIXTURE_ONLY. Historical v1 retains its existing interpretation and
  means disabled. Shared validation is used by read_factor_loop_context,
  execution_controls and archived_handoff_context. No caller date/horizon/clock,
  policy bytes or transport callback in the public context.
- Use the maintenance validated target_date, existing release-install reference
  bytes/SHA, release_repository_root and release_commit. Native acquisition owns
  its 21-natural-day horizon and policy/capability artifacts. Disabled/v1 makes
  zero additional provider calls. SYNTHETIC_FIXTURE_ONLY runs only within the disclosed offline fixture
  with replaced external transport. Public/live launcher contexts reject this
  mode. If synthetic transport provenance cannot be established, fail closed
  with PROVENANCE_UNAVAILABLE; never publish synthetic provenance for real
  transport bytes. Real capture may be retained as non-admitted evidence only
  until a separately governed truthful issuer exists.
- Define cn-daily-factor-state.v2 with validated explicit nullable future proof
  and failure refs; mutual exclusion is required. In enabled mode an attempted
  lane produces one proof or controlled failure ref before fresh Core publication.
  Save this binding before publishing Core. Fresh and _recover_core both pass it.
- Recovery uses only the deterministic target capture root and exact native
  execution/success refs. Never scan/latest-select or reacquire. Before any
  capture, recovery must retain a controlled unavailable outcome without calling
  providers. After immutable capture and before proof, validate retained capture
  and finish synthetic publication idempotently only before Calendar selection
  and EOD sealing. After saved ref, verify and reuse it. After Calendar selection,
  reuse the original terminal; never attach missing or replace future evidence.
- Map attempted acquisition failures to next_session_failure's existing typed
  codes and exact native failure ref when available. No raw exceptions in public
  failure payloads. Disabled absence is distinct from attempted failure.
- No changes to daily request/recipe, CLI flags, graph, completion, Morning request
  or native-input schemas: existing optional reference plumbing is sufficient.
- Update Morning harness to use the tested readonly_replay_guard so native read
  synchronization is preserved while data writers/network remain forbidden.
- Tests cover disabled/v1 no-call behavior; real proof propagation; typed failure;
  crash before capture, after capture, after proof, after state and after Calendar
  selection; no recovery provider calls; invalid release/date/SHA/policy/time and
  conflicting refs; future proof sealed after Calendar terminal rejection.
- New installed scenario required. Original c3 results stay immutable. New
  aggregate read-only replay and Morning REPLAY must use its exact new EOD proof.
  Never count the old missing-proof chain as a successful Morning path. Production
  PREFLIGHT/SEAL remains blocked without a separately governed live issuer.

Implementation readiness: Architect requested the provenance correction above
and confirmed the remaining amendments are ready for Critic after that correction.
Critic review pending.

## Exact contracts and fixture boundary (Critic revision)

### Context

v2 has exactly these fields:
- schema_version: literal cn-daily-factor-loop.v2.
- release_install_input_ref: exact relative path/sha256 reference.
- release_repository_root: canonical absolute directory string.
- release_commit: lowercase 40-hex string matching verified install.
- calendar_capture_parent: canonical absolute directory string, existing producer
  path validation unchanged.
- initial_calendar_receipt_ref: nullable exact path/sha256 reference (required key;
  only required non-null by initial historical settlement when no state exists).
- next_session_calendar_mode: DISABLED or SYNTHETIC_FIXTURE_ONLY.
No other fields. v1 keeps existing permissive historical semantics and disabled
future lane; do not impose these new required fields retroactively.

### Non-public fixture transport capability

Use a private in-process ContextVar capability scoped by an offline acceptance
runner under tests/unit. No environment variable, JSON field, CLI option, request,
recipe or automation setting can construct it. A private context manager installs
both fixed offline provider and documentation adapters and holds their exact
object identities; it rejects nested/replaced adapters. Calls outside that scope
cannot enable fixture mode. Public entry paths (daily-maintain, configured dispatch,
automatic catch-up) validate this before network calls or filesystem writes.
The offline runner invokes those same paths inside the private scope; it does not
mock acquisition, proof validation, Core publication or Morning.

Before capture, the capability binds workspace, EOD date, release-install SHA and
an immutable fixture manifest containing exact documentation bytes SHA and each
expected SSE/SZSE provider-response bytes SHA. The manifest is generated from the
pinned fixture source, not caller configuration. After real native acquisition,
verify the actual immutable capture raw/documentation hashes against that manifest
and the exact native execution/success refs. Publish typed evidence with exact keys:
schema_version=cn-calendar-fixture-transport-evidence.v1, eod_trade_date,
release_install_input_sha256, fixture_manifest_ref, documentation_sha256,
provider_response_sha256 (exact SSE/SZSE mapping), execution_ref, success_ref,
recorded_at, authority (all false).

The new configured synthetic publication entry requires BOTH active private
capability and this evidence, replays capture verification, compares every expected
hash and ref, and only then calls the synthetic publisher. Files or synthetic=True
alone never authorize publication. Recovery runner re-establishes the private
capability from the same pinned fixture manifest and verifies durable evidence;
no provider calls occur. Missing capability/evidence, adapter substitution or hash
mismatch gives PROVENANCE_UNAVAILABLE. Existing direct synthetic publisher remains
an explicitly named research API; do not claim it is a live issuer. No real
transport may pass the configured synthetic publication entry.

### Recovery state

v2 has exactly schema_version, phase, trade_date, context_sha256,
calendar_receipt_ref, core_checkpoint_ref, core_observation_refs,
future_capture_refs, fixture_transport_evidence_ref,
next_session_calendar_proof_ref, next_session_calendar_failure_ref,
core_handoff_ref. All refs use their existing native reference shape; nullable keys
are always present. trade_date is YYYYMMDD, context_sha256 is lowercase 64-hex.
calendar_receipt_ref and core_checkpoint_ref are non-null; observations are null
or exact LOW/W80 mapping of path/sha256 refs. future_capture_refs is null or exact
execution_ref/success_ref native ref mapping; no directories/latest pointers.
Proof and failure refs are mutually exclusive. Validate on every save and read.

Phases:
- CORE_READY: observations non-null; future and handoff refs null.
- CAPTURE_BOUND: capture and fixture evidence refs non-null; proof/failure/handoff null.
- FUTURE_BOUND: enabled lane exactly one proof/failure ref; disabled lane both null;
  proof requires capture+fixture evidence; failure validates existing typed contract.
- CORE_PUBLISHED: FUTURE_BOUND constraints plus non-null handoff ref and replay of
  its selected Calendar binding. No future-ref replacement is permitted.
Existing v1 state remains readable on disabled/v1 contexts. Context/state date or
SHA mismatch, unknown fields/phases and invalid combinations fail closed.

### Crash/recovery table

| Crash point | Allowed recovery writes before Calendar selection | Outcome/ref source | Provider calls |
| --- | --- | --- | --- |
| Before capture | Typed failure then FUTURE_BOUND state | ACQUISITION_FAILED; no native capture refs | 0 |
| After capture, before durable fixture evidence | Typed failure then state | PROVENANCE_UNAVAILABLE; exact deterministic native refs when valid | 0 |
| After CAPTURE_BOUND/evidence | Idempotent proof publication then state | Exact saved refs and fixture manifest hashes; invalid provenance -> PROVENANCE_UNAVAILABLE | 0 |
| After proof, before FUTURE_BOUND save | State only after exact proof revalidation | Deterministic proof identity derived from saved capture/evidence; no directory scan | 0 |
| After FUTURE_BOUND save | Core publication/state if not yet selected | Verify/reuse exact saved proof/failure ref | 0 |
| After Calendar selection or EOD seal | No future-evidence writes or replacement | Reuse selected terminal; missing proof remains CALENDAR_NEXT_SESSION_UNAVAILABLE | 0 |

Native acquisition failure retains its exact native failure ref where available.
Post-capture native validation uses existing controlled failure codes and phase
constraints. A corrupt native ref is rejected rather than fabricated. The
pre-capture state must be saved before acquisition so missing evidence cannot be
misread as permission to start a fresh provider call during recovery.

Tests additionally prove capability cannot be enabled by caller JSON or a boolean,
public paths refuse before side effects, substituted/real adapters cannot receive
synthetic provenance, all recovery cases have zero provider calls, and original
c3 artifact bytes/mtimes are preserved. New installed acceptance remains synthetic
research replay only. Revised Critic confirmation required before implementation.

## Review outcome and implementation start

Critic APPROVE received after the exact-contract revision. Shared v2 context/state
validators implemented in future_calendar_context.py; historical v1 remains
permissive and disabled. 16 focused contract tests pass. Integration, private
transport capability/custody, crash recovery and new installed acceptance remain
to be completed; validator tests alone do not prove the whole path.

Implementation detail: native capture includes BSE response custody in addition to
SSE/SZSE projection. Pin all three transport raw response hashes in the fixture
manifest/evidence; derive next-open projection only from the existing SSE/SZSE
agreement rule.
