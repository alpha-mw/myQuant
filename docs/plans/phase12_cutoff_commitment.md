# Phase12 cutoff commitment and input profile

Status: PROFILE_IMPLEMENTED_VALIDATION_IN_PROGRESS. Architect amendments were
adopted and Critic APPROVED before timing/profile implementation. This implements the remaining timing/input
integration; it does not replace Phase12 bootstrap, installed release or scheduler
acceptance. Prior timing/source-reader and missing-only decisions are in
`phase12_input_preparation_design.md`.

## Exact versions and immutable paths

Execution recipe v5 uses v4 fields, replaces `corporate_action_context_ref` with
`corporate_action_template_ref`, and adds `research_timing` with exact fields
`mode`, `policy_ref` and `acquisition_deadline`. Existing research_sources fields stay exact.

- CURRENT_POST_ACQUISITION: sources.as_of=null; deadline is canonical UTC-second
  text inside the target Shanghai date. Acquired or exact pinned Theme allowed.
- HISTORICAL_RETAINED: sources.as_of is the exact historical cutoff on target
  date; deadline=null; theme_acquisition_ref=null and exact retained per-date
  Theme descriptor required. The existing historical native Calendar route still
  owns date eligibility; no present-day provider call substitutes for history.

Old v1-v4 remain byte-exact. New mode validation cannot silently convert or persist
an old recipe as v5. Its non-temporal fields reuse owning legacy validation.
Collection/binding v3 and automatic request v2 select v5 for newly prepared rows;
all normal old-profile replay stays supported.

The timing policy is code-owned and introduced with this installed source version.
`cn-daily-research-timing-policy.v1` has exactly schema_version, mode, timezone,
acquisition_deadline_local_time, cutoff_rule, alignment_max_seconds, authority.
timezone=Asia/Shanghai, alignment_max_seconds is integer1 and authority is exact
FALSE_AUTHORITY. Current mode requires local deadline23:59:59 and
cutoff_rule=AFTER_VERIFIED_SOURCE_CUSTODY; historical requires local deadline=null
and cutoff_rule=EXACT_RETAINED_AS_OF. No policy field enables providers or expands
the existing scheduled authorization. The recipe deadline must equal the exact
target-date UTC conversion of this installed policy, not a caller-selected time.
Both bundle and cutoff receipt bind the same exact policy_ref as the recipe.
Fresh expired current execution rejects before new maintenance/provider work;
receipt-directed readback/recovery is separately allowed as specified below.

Corporate template v1 has exact fields schema_version, strategy_id,
tracking_policy_ref, named_event_refs, anchor_reviews_ref. named_event_refs is
null (unknown) or a sorted unique exact physical-ref list. No event facts or
owner declarations are synthesized. After cutoff, derive existing native v1
event-list/context wrappers with that cutoff and those unchanged refs.

Theme policy/handoff v3 preserves v2 scope, claim, native fallback and capture
budgets. It denotes raw acquisition under the deadline. Source completeness is
validated through native readers; final projections are computed at the resolved
cutoff. No future-cutoff projection is published. Existing v2 semantics stay exact.

Let E be `journal.root/executions/<outer-request-sha256>`:

- source bundle: `E/inputs/acquired-sources.v1.json`;
- cutoff commitment: `E/research-cutoff.v1.json`;
- legacy-shaped native research payload: `E/inputs/research-<payload-sha>.json`;
- research wrapper v2: `E/inputs/research.v2.json`;
- corporate event list: `E/inputs/corporate-events-<sha>.json`, null only when the
  original named_event_refs is null;
- corporate context: `E/inputs/corporate-context-<sha>.json`;
- portfolio: unchanged `journal.root/inputs/portfolio/<plan-sha>/state.v1.json`;
- materialization v5: `E/materialization.v5.json`.

No path scan, alternate version fallback, regenerated cutoff or separate writer
authority exists. Conflicting fixed-path bytes reject.

## Source bundle and native proof

`cn-daily-acquired-sources.v1` has exact fields schema_version, trade_date,
request_ref, maintenance_handoff_ref, core_handoff_ref, pool_manifest_ref,
theme_source_handoff_ref, corporate_action_template_ref, timing_policy_ref,
auxiliary_stage_refs, event_pointer_ref, native_request_fields. Auxiliary refs are
the exact final Fundamental/Macro stage refs (null only for PINNED mode). The Event
pointer uses the existing native historical lookup and existing day-owned
`inputs/event-pointer-<sha>.json` custody path; no Event generation is written.
native_request_fields is the exact existing native research payload without
as_of: strategy_id, policy, expected_trade_date, expected_factor_pointer_sha256,
industry_source, theme_source, company_evidence, and the LOW/W80 observation
path/SHA pairs. company_evidence has exposure_rows, fundamental_source, macro_risk.
The bundle is derived by the existing materializer from the exact recipe and
native auxiliary stage refs. Fundamental pointer custody is retained as today.

Construct the bundle in memory and persist its fixed path only after final
auxiliary refs, replayable Theme source closure and mandatory native preview
validation succeed. Critical missing sources must not freeze a null source into
an immutable bundle; their producer may still finish on retry. Once written, the
bundle source choices cannot change under the same request.

The code-owned source reader validates bundle ancestry, native pool/rank/LOW/W80,
original descriptors and all transitive native source closures. Native validation
at the deadline may be used only as an unpublished pre-cutoff source read to
establish custody. It is followed by full native projection at the actual cutoff
before any timing commitment. Such provisional outputs never become artifacts,
captures, Decision inputs, or an availability assertion. Portfolio loading takes
no provisional as_of.

Native validation invokes the existing projectors and missing-only classifier.
All five ordinary source nodes must yield SUCCEEDED execution, with optional
missing-only Exposure allowed and its original business status preserved. Focus
is reconstructed with the exact existing two-company native/PIT inputs. Critical
missing or corrupt sources reject before a fresh cutoff commitment.

Macro first-seal admission uses the existing current closure guard. Commitment
readback may reconstruct its exact frozen closure/risk/freshness to prove original
source binding without selecting current heads. This does not replace the normal
fresh Decision compiler's current-Macro admission guard or the full EOD reader.
Extract deterministic projection construction from the existing owning
`verify_current_macro_readiness_closure` into a helper, retaining every current
guard. Frozen replay derives this same projection only after intrinsic
closure/target/cutoff validation; it does not assert a new current-head match.

## Cutoff receipt and ordering

`cn-daily-research-cutoff.v1` has exactly:

schema_version, trade_date, request_ref, maintenance_handoff_ref,
core_handoff_ref, source_bundle_ref, native_request_ref, store_plan_ref, portfolio_state_ref,
timing_policy_ref, mode, acquisition_deadline, as_of, portfolio_created_at,
physical_reads_completed_at, sealed_at,
source_refs, source_times, projection_sha256s, corporate_event_list_ref, corporate_context_ref,
portfolio_timing_status, prospective, authority.

Refs are exact path/SHA pairs. source_refs is a sorted unique union of retained
physical roots and leaves observed by owning readers; nested generation/table
closures remain under native validation. source_times is a sorted unique list of
exact objects with role, subject_id, source_ref, original_time, time_semantics.
The finite role/time mapping is the reviewed owning-reader table. No generic
timestamp lookup is allowed. Original fractional timestamps are preserved.

Exact role enum: INDUSTRY_TAXONOMY, INDUSTRY_MEMBERSHIP, THEME_POOL_DC,
THEME_POOL_TDX, THEME_FOCUS_DC, THEME_FOCUS_TDX, THEME_HANDOFF,
EXPOSURE_DECLARATION, FUNDAMENTAL_DECLARATION, FUNDAMENTAL_NATIVE_DERIVATION,
MACRO_CLOSURE, EVENT_GENERATION, CORPORATE_POLICY, CORPORATE_EVENT,
CORPORATE_REVIEW_DECLARATION, STORE_PLAN, PORTFOLIO_SOURCE_SEAL,
PORTFOLIO_POINTER_PUBLICATION. Exact time_semantics enum: PROVIDER_CAPTURE,
SOURCE_DECLARED, LOCAL_CLOSURE, LOCAL_PUBLICATION, OWNER_EFFECTIVE. Actual physical custody has its
own receipt field and cannot be supplied as a source's declared time. Industry
and Theme final captures use PROVIDER_CAPTURE; handoff/Macro/Event/Store-plan/
portfolio seals use LOCAL_CLOSURE; portfolio pointer publication uses
LOCAL_PUBLICATION and binds the retained pointer containing published_at.
The portfolio source-seal row identifies the native record and binds its catalog;
each Fundamental native derivation row binds the exact owning manifest or retained
pointer that actually contains that timestamp. The native availability owner
uses the maximum of both; attributing a pointer-only timestamp to a manifest is
forbidden.
Exposure/Fundamental declared
time and corporate announcement/review declarations use SOURCE_DECLARED;
Fundamental native derivation uses LOCAL_CLOSURE; policy effective time uses
OWNER_EFFECTIVE. subject_id is the company, native generation/record/event/
declaration ID, or exact `ALL` for unpartitioned roles. Rows sort uniquely by
(role, subject_id, source_ref.path, source_ref.sha256); an observed ref cannot
appear with conflicting timestamps under the same role/subject.

physical_reads_completed_at retains actual UTC microseconds. as_of, sealed_at
portfolio_created_at and current deadline retain native canonical UTC seconds.
portfolio_created_at is explicitly committed so historical state bytes remain
reconstructible after a crash even when created_at differs from historical as_of.
An owner-effective row
is descriptive; it never replaces the measured physical custody bound.

projection_sha256s has exactly industry, theme, exposure, fundamental, macro.
For industry/theme/exposure/fundamental, the formula is
sha256(canonical_json_bytes(ordered_native_artifact_list)), with output ordering
owned by existing artifact_output_names and native builders. Macro hashes exactly
the canonical object {native_admission: exact_macro_ready_projection,
artifacts: ordered_macro_risk_and_freshness_list}. Current mode computes its
admission at actual cutoff immediately before commitment; frozen/historical
replay reconstructs it from the intrinsic closure and original frozen refs.
Different output ordering must fail; never concatenate raw hashes. An absent
critical node cannot be represented by an empty successful hash.
corporate_event_list_ref and corporate_context_ref bind exact deterministic paths
and bytes reconstructed from template/cutoff/original refs. An empty supplied
event list is only a declaration of no named events, never issuer CLOSED_EMPTY.
prospective is false and authority is the existing exact FALSE_AUTHORITY.

Current-day transaction under the existing day lock:

1. Validate exact source bundle/native sources and prepare the owning Store plan;
   retain/read/recheck the native pre-close book and all source bytes. Record
   actual aware UTC physical-read completion.
2. At most one bounded <=1-second alignment is allowed to reach the next native
   whole-second boundary. Sample once, serialize canonical seconds, and require
   cutoff >= physical-read completion and original availability envelopes. Clock
   regression, target-date change or expiry blocks; no resample/wait loop.
3. Build exact portfolio bytes in memory with as_of=created_at=cutoff. Compute all
   native source projections at cutoff and the exact corporate context bytes.
   No provider or source selection occurs after cutoff. Recheck every source,
   sample actual high-resolution seal observation, require cutoff <= observation
   <= deadline, then derive native whole-second sealed_at. Persist immediately;
   expiry comparison uses the actual observation, not its floored serialization.
4. Persist the immutable cutoff commitment BEFORE portfolio state. The receipt
   commits source set/times/projection hashes AND expected portfolio state SHA.
5. Write only committed state bytes, then exact native payload/wrapper/context and
   materialization. Read back all bindings. No consumer is admitted until every
   required object and existing native gate passes.

Historical mode keeps supplied historical as_of and actual custody/state-created
and receipt times. Owning native research and issuer/review validators still
enforce their original as_of rules. Historical financial book/Event local seals
may be late, as already permitted by native portfolio/event replay; they remain
explicitly late under native portfolio timing and historical/prospective=false
classification. The descriptive time-row grammar cannot waive any native
research availability gate or invent an earlier local seal.

Crash before commitment leaves no new dynamic state; exact retained pointer and
source bundle may remain. Before deadline it can retry native validation and
select a fresh uncommitted cutoff. After deadline absent commitment blocks.
Crash after commitment loads exact source bundle and timing policy, reconstructs
portfolio/native payload/event list/context/wrapper, and compares every ref/SHA and
projection hash with the commitment before writing only missing committed bytes.
Conflicting existing bytes reject. Materialization v5 is last. This recovery may
occur after deadline; source-set/time/SHA differences block. State/payload/context
without the exact commitment cannot enter the new profile.
Existing complete commitment/state is read-only and never resampled. Old fixed
states keep their old path/bytes and cannot be relabeled.

## Research wrapper and downstream closure

`cn-daily-research-request.v2` has exact fields schema_version, trade_date,
native_request_ref, cutoff_ref, source_completion_policy. The selector is exactly
native-optional-exposure-missing.v1. The cutoff receipt binds native payload bytes;
the wrapper binds both payload and receipt, avoiding a SHA cycle.

An owning loader validates wrapper/receipt/payload/native ancestry and returns
the unchanged native payload plus the separately validated selector. Native
compile consumes only the exact native payload; source recipe derivation receives
the selector explicitly for Exposure. No unknown native fields are ignored.
Research capture identity still uses the wrapper ref. Legacy native payloads keep
existing behavior and hashes.

Decision recipe v2 stays exact but its reader loads the wrapper and verifies the
same as_of, native Store plan, commitment and expected portfolio state. Native
input v6 extends v5 with cutoff_ref; materialization v5 adds cutoff_ref and
corporate_action_template_ref while retaining final native context ref. Both must
bind the same wrapper/receipt/state and serving policy. Completed research,
Macro/Decision/corporate replay, ledger and Morning use the owning loader and
version-selected closure. No completed EOD bypass or new current-head fallback.

Exact version chain is recipe5 -> materialization5 -> native6. Extend sibling
materialization conflict checks through v5. Native6 fields are exactly native5
plus cutoff_ref. Existing recipe/materialization1-4 and native1-5 readers remain
unchanged. New automatic rows require collection/binding3 and automatic request2;
new acquired Theme requires policy/handoff3. Wrapper2 is selected only by exact
schema, never a path/version scan. Completed readers use that owning dispatch.

## Acceptance

Real native book/capture/projector checks run with explicit provider/core seams
where required; final installed/full-DAG evidence remains separate. Required
checks cover immediate acquisition-to-compile, current/historical chronology,
one bounded clock alignment, actual deadline/clock regression, missing-only versus
critical/corrupt sources, receipt-before-state crash recovery, changed source set,
wrong native projection/context/state hashes, immutable replay after deadline,
old-profile byte/semantic stability, selector propagation through completed replay,
and exact existing Decision admission. No production event/veto/holdings/automation
state is modified for local validation.

Additional acceptance: advance current Macro head after commitment but before
Decision; frozen commitment proof remains valid while fresh Decision keeps its
independent current guard. Change a future corporate wrapper path while preserving
semantic bytes; recovery must reject the exact-ref mismatch.

## Implemented checkpoint and remaining verification

Native portfolio loading now precedes cutoff selection. The cutoff writer commits
the entire expected object set before state/payload/context/wrapper writes,
prechecks all existing-object conflicts before repair, and supports exact receipt
recovery after expiry. The source bundle is frozen only after mandatory native
preview and corporate-source validation, with original auxiliary/Event custody.
Current Macro admission is hashed separately from frozen readback and still runs
independently during fresh native Decision compilation.

The wrapper, recipe5/materialization5/native6, and collection/binding3/automatic2
dispatch is implemented across producer, completed replay, ledger, Morning and
serving readers. Verified legacy completed prefixes remain reusable; only newly
declared v3 rows require native6. Existing public fields/authorities stay unchanged.

Focused transaction tests use actual native book/storage/clock paths but explicitly
control research-source and corporate proof builders. They do not certify all
five source domains together or a new native6 EOD. Separate native-source tests
now exercise SourceBundle/corporate projection, current/historical native6
materialization, compiler consumption, replay after expiry, and Macro head
advancement after commitment. `.agent/acceptance/phase12-native-source-validation.json`
records3 native integration cases PASS185.79s. Core/maintenance receipts, focus PIT,
Factor activation and cutoff-specific synthetic rank remain explicit fixture
boundaries; installed/full-Core/public-EOD/unattended proofs remain required.
No real source, financial state or automation changed.

Native integration found and fixed an incorrect raw-pointer field assumption:
Fundamental manifest SHA comes from native frozen-pointer inspection's
derivation_binding; raw pointers do not carry manifest_sha256. The returned native
manifest and retained bytes are compared. Industry replay now validates the common
plan once per batch while retaining every partition and complete capture check,
without cross-call caching. Actual retained696-ref Industry readback returned
READY in2.196s without source changes.

An isolated current-runtime installation now completes the actual native Factor,
Core,16-node daily chain and EODv2 over materialization5/native6. Repeat of the
exact request is NO_ACTION, with1439 protected files unchanged and producer/network/
journal writes forbidden. See `.agent/acceptance/phase12-installed-v6-validation.json`.
The data/transport and core/cutoff clocks are marked synthetic; technical release
verification and Factor/Core are not substituted.468 runtime/build files match
the tested snapshot. The request is assembled after native Core, so this does not
prove initial pre-Core provisioning or the live scheduler. Ledger is correctly
prospective=false and synthetic=true. Full goal and Phase12 remain incomplete.
