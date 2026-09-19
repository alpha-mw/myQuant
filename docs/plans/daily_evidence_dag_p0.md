# P0 Daily Evidence DAG Closure

Status: implementation in progress; full objective remains active.

## Objective and acceptance

Connect the existing governed Market/PIT/Calendar, Factor, LOW/W80 observations,
Top100, Theme/Industry/economic exposure, Fundamental/Macro, research Decision,
Store valuation/continuity, Dashboard, and next-session Morning consumer into one
deterministic, resumable daily production path. This is operational continuity,
not a change to factors, weights, selection policy, stops, or trading authority.

Required delivery: actual node inventory and source-backed break diagnosis;
explicit nine-state node model; strict dependency resolution; immutable exact
input/output refs and append-only attempts; deterministic replay and conflict
detection; crash reconciliation before re-execution; ordered Calendar-driven
catch-up without pointer rollback; source-backed prospective/late/recomputed
timestamps; production adapters reusing existing writers; a read-only Morning
completed-DAG gate; full-DAG integration tests including at least five synthetic
consecutive trading dates, independent failure and recovery cases; reviewed
automation migration preparation and exact deployment verification when authorized.
Synthetic success must never be described as five real unattended trading days.

## Current integration constraints

- Main checkout is 68df456 with concurrent Fundamental and weekly/Store reporting
  edits. Preserve those edits. Shared close coverage and research-risk modules
  are owned by the other task; consume their intended interfaces, do not replace.
- Deployed Factor producer is in immutable release 7b26ac2, outside main history.
  Read its source and use its stable interface; do not copy an older producer or
  restore the retired automation/daily_runner path. Integration/release provenance
  must explicitly account for this split before deployment.
- This morning's exact receipt identifies Factor and observations as of20260904,
  absent same-date Top100, Store and Dashboard valuation20260903. Refresh these
  exact refs before claiming the current baseline.
- Store accounting must remain independent of missing research alpha. Decision
  has an ordering/completion edge in the orchestration graph, but a missing
  research decision cannot invalidate otherwise valid official-close inputs.
  The entire investment DAG remains incomplete when a required research node is
  missing. Do not turn this distinction into an all-green success projection.
- Missing event/benchmark evidence blocks Store publication; never infer empty
  owner events, adjust actual shares/cash/costs, or fabricate historical receipts.
- A complete source-bound negative/insufficient-evidence Decision is different
  from an absent/invalid Decision artifact. Preserve the existing five states;
  action-like example names in the reference plan do not authorize a new policy.
- Fundamental can have an older financial cutoff if PIT-valid and known before
  decision cutoff; do not demand artificial daily financial releases. Missing
  critical bindings remain explicit blockers.

## Execution sequence

1. Freeze baseline source/config/artifact identities and inventory all17 nodes,
   including Corporate Action reconciliation and Morning consumer. Write
   docs/architecture/daily_evidence_dag.md and diagnostic-only
   results/system/daily_dag_baseline.json (never System activation).
2. Architect then Critic review this plan and exact contracts before implementing
   production orchestration/schema changes. While reviewing, continue read-only
   adapter inventory and deterministic baseline capture.
3. Implement state/ref/failure/timestamp contracts and durable day journal in
   quant_investor/operations, with exact explicit request validation, no arbitrary
   command/plugin callbacks from untrusted requests, fixed allowlisted adapters,
   safe paths, locking and post-crash discovery via known deterministic refs.
4. Wire the existing daily launcher/CLI path and repair automatic Top100 readiness
   after verified Factor/LOW/W80. Add exact per-date replay support without using
   today's head for historical dates or rewriting immutable historical outputs.
5. Connect research producer/binding adapters and existing official-close/coverage
   and Dashboard paths; preserve separate financial/research completeness.
6. Connect ordered catch-up, prospective ledger and Morning's prior-DAG-only gate.
7. Test real adapters with isolated synthetic evidence through five full trading
   days; cover all10 reference cases plus tampering, crash-after-write recovery,
   concurrency, false authority, stale artifacts and late timestamp classification.
8. Run required full CI-equivalent checks, inspect actual current production
   readiness read-only, prepare exact scheduler/release changes and validate them.
   Keep production/provider/deployment writes outside local synthetic verification.

## Completion audit

Track proof separately for implementation, installed entrypoints, deployment,
synthetic continuity, and real unattended receipts. Every required node, adapter,
test, contract and safety invariant needs direct evidence. Do not mark the goal
complete at a schema-only, generic-engine-only, plan-only or mock-handler milestone.

## Normative contracts after Architect review (2026-09-07)

Architect verdict REVISE was addressed by the following binding amendments. The
user's exact nine states override the review's suggested alternative spellings.

### Graph and lifecycle

Normative17 nodes: calendar, market, pit, factor, low_observation, w80_observation,
top100, theme, industry, exposure, fundamental, macro, corporate_action_recon,
decision, store, dashboard, morning. LOW/W80 are independently observable but
share the existing batch writer; one missing/invalid child does not count as two
successful observations. Morning is the next-day consumer, not a production writer.

States: NOT_STARTED, READY, RUNNING, SUCCEEDED, PARTIAL, BLOCKED, FAILED, SKIPPED,
STALE. NOT_STARTED/BLOCKED/FAILED(retryable) -> READY only after exact dependencies
and recovery validation; READY -> RUNNING -> SUCCEEDED/PARTIAL/BLOCKED/FAILED.
Unsatisfied dependencies -> SKIPPED (reason UPSTREAM_INCOMPLETE). A previously
successful node whose immutable input/output validation now fails is STALE with a
terminal conflict or validation failure; it cannot silently rerun or overwrite.
NO_ACTION is an adapter result mapped to SUCCEEDED only after native validation.
IN_DOUBT is recovery_state, not a tenth lifecycle state: a RUNNING attempt without
terminal receipt first reconciles deterministic native output before any retry.
Domain OPEN/AVAILABLE/UNVERIFIED/INSUFFICIENT_EVIDENCE and financial/factor/research
readiness remain independent of lifecycle. A source-complete negative Decision
can finish compilation successfully; missing critical source inputs cannot.

Each edge is one of requires (successful execution prerequisites), after (ordering
only, completed failure does not suppress independent branch), completeness_requires
(must succeed for the final day seal). Top100 requires factor,low_observation,
w80_observation and approved policy; no Fundamental/Macro edge. Decision requires
registered source bindings, allowing valid low-frequency Fundamental cutoff with
explicit lag warning. Store requires calendar,market,exact event/corporate-action
accounting, held closes and benchmark inputs; Decision is after only. An absent
threshold-anchor reconciliation can block moving thresholds without blocking
otherwise exact accounting when no unresolved financial adjustment remains.
Dashboard depends financially on Store and carries research incompleteness. Final
production completion requires all applicable financial/research nodes; missing
critical research inputs cannot disappear behind financially successful output.

### Execution integration and locks

Use the existing daily launcher and DailyFactorLoop core callback. Bounded core
handoff: verify sealed core -> Factor -> LOW/W80 -> Top100 -> immutable core-handoff
receipt, before auxiliary Fundamental/Macro resumes. Do not run full DAG inside
that callback. A downstream coordinator resumes after maintenance returns, or
from the already sealed handoff during registered recovery, without relaunching
maintenance. Status may be produced during auxiliary stalls from exact handoff.

Lock order is maintenance (when already required) -> per-day DAG -> subsystem.
Never acquire maintenance while holding DAG lock. Sequential day handling must
not hold two day locks or roll back any subsystem pointer. Own attempts are an
orchestration journal, not another Market/Factor/financial authority and not a
cross-store atomic transaction.

### Journal, recovery, failure taxonomy

Node business key hashes graph contract SHA, date, node, release/install ref,
policy refs, ordered upstream immutable refs and typed external inputs. Caller
requests may not contain shell commands, import paths or executable callbacks.
Fixed code-owned adapters alone select existing writers. Native mutation permits
remain explicit and scoped; graph eligibility is not authority.

Write owner-only immutable request/start/terminal attempt leaves at deterministic
per-date roots under results/operations/daily_production/CN/YYYYMMDD. Adapter code
identity/SHA, input/output refs, started/finished/recovered times, attempt number,
write set, stable failure and explicit false trading/System/Mainline authority
are required. dag-status.v1.json is atomically replaced projection of append-only
receipts; completion.v1.json is write-once and binds all node receipts. Reconstruct
projection after crash from known journal filenames, never mtime. Existing success
requires revalidation; same inputs -> NO_ACTION; different immutable outputs ->
IDEMPOTENCY_CONFLICT. Changed pending inputs create an explicitly linked new
attempt, preserving the prior attempt; completed old inputs cannot be relabeled.

Stable failures include INPUT_MISSING, INPUT_STALE, POINTER_MISMATCH, SHA_MISMATCH,
UPSTREAM_INCOMPLETE, PROVIDER_UNAVAILABLE, AUTHORIZATION_BLOCKED, SCHEMA_MISMATCH,
IDEMPOTENCY_CONFLICT, DATE_MISMATCH, CALENDAR_MISMATCH,
CORPORATE_ACTION_UNRECONCILED, POLICY_BLOCKED, IO_TRANSIENT, WRITER_FAILED,
POST_WRITE_IN_DOUBT, VALIDATION_FAILED, UNSUPPORTED_LINEAGE_GAP. Each records
retryable, owner_action_required, canonical_state_may_have_changed and
recommended_next_node. Unknown exceptions are terminal/unconfirmed, never blindly
retryable. Provider retries keep the existing logical-task budgets.

Crash adoption uses native validator plus reconstructed exact inputs at known
output refs. If original publication time cannot be proved, availability is no
earlier than recovery observation and evidence is UNKNOWN_LEGACY/recovered.
Conflicting leaves stop. Preserve Store and Dashboard last-good selectors.

### Catch-up and prospective time

Enumerate dates only from a validated Calendar and exact last-completion identity.
No weekend/holiday financial state. Advance genuine upstream head gaps in order.
For downstream gaps behind head, add an exact trade-date/pointer lineage reader
under Factor lock; recheck ancestry, never substitute current signals or roll back.
Missing historic generation behind advanced head is UNSUPPORTED_LINEAGE_GAP.
Store uses the existing close-through-latest atomic prefix and accepted coverage
analyzer; do not replace its event/benchmark/held-close validation.

Record effective_trade_date, source available_at, generation sealed_at,
observation registered_at, coordinator first_observed_at, completed_at, recovered_at,
Calendar-bound prediction deadline and actual publication evidence. Classifications:
CONTEMPORANEOUS, LATE_REGISTERED, RETROSPECTIVE_RECOMPUTE, UNKNOWN_LEGACY.
Only proven contemporaneous availability can be real-time OOS eligible. No supplied
clock, generation.created_at, successful replay or filesystem mtime can manufacture
availability. Synthetic fixtures carry synthetic=true and prospective=false in
production-facing projections regardless of simulated timestamp.

### Pool, Morning and source integration

The full Phase0–15 design supersedes the earlier optional projection proposal.
Phase3 now uses the reviewed native five-leaf tabular pool successor at the same
daily path; see `phase03_tabular_pool.md`. Preserve original four-leaf historical
closures without mutation. New manifests record actual first-publication time
and preserve the native rank's separate source-derived time.

Morning v2 request binds one exact prior-session completion ref and same-day quote
seal plus owner policy refs. Derive Factor/observations/Top100/Decision/Store/Dashboard
from completion and revalidate; incomplete prerequisite ->
MORNING_UPSTREAM_DAG_INCOMPLETE. Historical v1 remains explicit only, never silently
selected by the new automation. Morning imports no maintenance/writer adapter.

Implementation worktree is /private/tmp/myquant-p0-daily-evidence-dag at7b26ac2.
Main68df456 is its verified ancestor. Reuse native deployed handlers. Before final
integration, consume the other task's named accepted close-coverage/research-risk
commit and prove both histories are ancestors; resolve overlapping CLI/access and
Fundamental files, not copy uncommitted files. No deployment from dirty source.

### Enumerated integration acceptance

1. Five consecutive synthetic Calendar trading days through real production
   adapters/native validators and writers, all immutable completion refs checked.
2. Same-day identical rerun: zero repeat business writes and validated NO_ACTION.
3. Factor/LOW/W80 succeeded but pool absent: auto-publish; injected Top100 failure
   then retry resumes there without repeating Factor; pool precedes stalled aux.
4. Store prior date/input gap: exact Calendar prefix catch-up when inputs arrive;
   missing events/benchmarks block Store while research can progress.
5. Valid no-trade event closure: new strict-close valuation, identical shares/cash/
   cost, new financial state; absent event is not equivalent to no-trade.
6. Unreconciled corporate-action threshold anchor remains NON_EXECUTABLE; accounting
   may complete only if its distinct required adjustment/event inputs are exact.
7. Fundamental valid low-frequency lag warns and permits Decision; critical missing
   evidence blocks it; independent Store still progresses on valid inputs.
8. Market/PIT/Calendar/SHA/input-authority tampering and immutable conflict fail
   closed; concurrent invocation has one writer and no duplicate attempts.
9. Crash after native write before terminal receipt adopts only validated exact
   output, preserves original timestamps or classifies recovery conservatively.
10. Weekend/holiday NO_ACTION, ordered multi-day catch-up, late/recomputed/unknown
    provenance never eligible as real-time OOS; no hidden use of current head.

Additionally verify Morning v2 pure-consumer behavior, stale Dashboard refusal,
source/release identities, zero broker/order/trade/System/Mainline/actual-holdings
mutation and provider-free synthetic execution. Full unit and CI lint/type/format
checks follow focused integration. Do not count generic success stubs as full DAG.

## Critic readiness amendments

EOD has16 production nodes; Morning is the17th inventory node, a separate T+1
consumer. completion.v1.json binds exactly the16 EOD node terminal receipts and
never includes Morning. Morning's receipt binds the completed previous EOD seal;
there is no circular completion dependency. Day status is SUCCEEDED only when
all required EOD nodes succeeded and native outputs revalidate. Otherwise it is
RUNNING while an owned attempt runs, FAILED for terminal execution errors, BLOCKED
for unmet critical prerequisites, or PARTIAL for completed but incomplete branches.
No completion seal exists for BLOCKED/PARTIAL. CLOSED Calendar date returns
NON_TRADING_DAY_NO_ACTION without any financial node/write.

Single production surface: `quant-investor production daily-close --workspace-root
<root> --request <canonical-relative-json> --expected-request-sha256 <sha>`.
The versioned request is `cn-daily-production-request.v1`: exact market=CN,
strategy_id=aggressive_tech_manufacturing, action=PLAN/EXECUTE/RESUME/CATCH_UP,
target_trade_date, exact Calendar ref, previous_completion_ref (explicit bootstrap
null only with validated initial state), pinned release/install/graph/policy refs,
existing slot/logical-claim context, and typed per-date node input refs. It rejects
arbitrary node lists/commands/imports, supplied successful states and authority
flags. PLAN is read-only. RESUME cannot initiate maintenance and consumes existing
checkpoint/handoff. CATCH_UP enumerates Calendar from the previous completion to
target and requires exact per-date input sets; it cannot replay current sources as
historical inputs. Read-only surface `production daily-status --workspace-root
<root> --market CN --strategy aggressive_tech_manufacturing --trade-date YYYYMMDD`
replays the deterministic day journal and completion refs without starting writers.

Existing shell launcher remains credential/installation/slot bootstrap and calls
this single coordinator for20:20; maintenance-only early slots stay unchanged.
The coordinator calls existing maintenance exactly once outside DAG lock with the
bounded core hook. Recovery can use RESUME even if maintenance auxiliary stage is
unfinished, consuming sealed handoff only, without another maintenance invocation.
Provider scopes, native logical budgets and mutation authorizations remain governed
by existing installed context and explicit live opt-in, not DAG status/request flags.

Adapter fan-out is explicit: one native maintenance invocation owns Market/PIT
production and emits exact separate node refs via its core checkpoint; Calendar
capture remains the existing core callback's guarded provider operation. All three
project SUCCEEDED only after native source/receipt replay. Factor is existing
rollover. One observation batch writer creates LOW/W80; validate each child exactly
and reconcile an already written child without cloning/recreating the batch.
Top100 adapter calls registered research pool-publish after both children verify.
Historical research extends existing read_observation_history under _active_lock;
new public exact pointer/date resolver must recheck current ancestry after copying
immutable values. Historical replay revalidates lineage membership before pool
publication instead of demanding historical pointer remain current.

Prospective ledger path: results/prospective/CN/YYYYMMDD/evidence-ledger.v1.json,
write-once and bound by the EOD completion receipt. It binds actual Factor generation
seal, both observation registration times, Top100 publication, Decision publication,
Store close and Dashboard completion. The exact prediction deadline is an explicit
registered policy/calendar reference, never an invented timestamp default. Without
that evidence, classification=UNKNOWN_LEGACY, prospective=false. DAG eligibility is
the conjunction of all required artifacts' proven available-at <= that deadline;
late registration/recompute/recovered-unknown/synthetic always sets prospective=false.
Preserve existing classify_seal and immutable outcome cohort labels; distinguish
legacy signal-cohort provenance from the new full-DAG availability assertion.
Recovery unknown publication adopts no availability earlier than first recovery read.

Tests additionally enumerate every legal and illegal node-state transition, day
completion with missing/partial node, no Morning/EOD circularity, and strict readonly
status/PLAN/RESUME permissions. Adoption after crashes cannot grant a new source
publication timestamp. Five-day integration proof includes exact16-node receipt
sets, monotonic dates/pointers, read-only next-day Morning, and every native validator.

Deployment acceptance must preserve existing schedules and DUAL_RUN fallback while
binding primary and fallback to the same new entrypoint/release/graph. Prepare exact
preimages and migration payloads, validate offline, then perform only authorized
control-plane updates with readback. Prove one per-date owner and no duplicate native
writers under primary/fallback concurrency. If cutover readback fails, restore exact
prior scheduler configs through the same control plane, not source/pointer rollback.
Report implementation/install/deployment/synthetic continuity/real scheduled receipts
separately. No production deployment until source ancestry and required gates pass.

### Final readiness details

The normative EOD completion predicate is all16 EOD nodes have SUCCEEDED terminal
receipts, every native validator and bound ref replays, graph/request identity is
exact, and the prospective ledger validates (eligibility may be false). Morning is
excluded. The DAG seal indicates complete production evidence, not investment
admission. The following domain mappings are exhaustive for completion:

| Nodes | EOD required | Accepted lifecycle | Domain mapping |
|---|---|---|---|
| calendar,market,pit | yes | SUCCEEDED | Native complete/core-ready exact target+coverage+SHA; registered degraded Calendar authority is disclosed, invalid/missing blocks |
| factor | yes | SUCCEEDED | VERIFIED/ACTIVE/READY exact target or validated historical lineage; control W75 never selectable |
| low_observation,w80_observation | yes, separately | SUCCEEDED | OPEN/NON_AUTHORIZING exact child ref/date/signal/source; late remains non-prospective |
| top100 | yes | SUCCEEDED | PUBLISHED or exact native NO_ACTION, valid100-row canonical rank+selected-symbols closure |
| theme,industry,exposure | yes | SUCCEEDED | All critical registered source bindings valid; explicit critical MISSING/UNVERIFIED maps PARTIAL and prevents full EOD seal; no implicit exposure confidence |
| fundamental,macro | yes | SUCCEEDED | PIT-valid known-at binding; acceptable low-frequency lag is warning, not failure; critical missing/veto maps BLOCKED |
| corporate_action_recon | yes | SUCCEEDED | Required financial event/adjustment closure valid; distinct unsupported moving-threshold anchor stays warning/NON_EXECUTABLE, never invented adjustment |
| decision | yes | SUCCEEDED | Exact five-state compiled evidence; source-complete negative research outcome allowed, missing critical source closure PARTIAL/BLOCKED |
| store | yes | SUCCEEDED | Native official-close/performance/continuity exact requested date; no changed actual shares/cost/cash; missing exact input BLOCKED |
| dashboard | yes | SUCCEEDED | Native private checker valid, Store date and required research refs exact; stale BLOCKED/STALE |
| morning | no | separate receipt | Exact prior completed EOD + same-day quote, no maintenance/writer execution |

All other lifecycle states prevent the EOD seal. Aggregate priority: validated
completion->SUCCEEDED; terminal safety/fatal writer error->FAILED; live owned
attempt->RUNNING; missing critical inputs->BLOCKED; finished independent branches
without full closure->PARTIAL; otherwise NOT_STARTED. A stale/tampered receipt
never wins over a previously green projection.

CLI exit0 means valid PLAN/status or fully verified execution/no-action; exit2
means well-formed typed business BLOCKED/PARTIAL; exit1 means terminal execution/
validation failure. All return machine JSON separating execution and business state.
PLAN/status produce no files and do not read credentials. No caller-supplied
arbitrary execution hook is allowed. Unknown adapter exceptions become WRITER_FAILED
with retryable=false and state-change uncertainty preserved.

Prospective validator validates exact ledger schema/ref closure, immutable input
identity, UTC-aware canonical timestamps and Calendar/policy deadline ref. Equality
at the configured deadline counts eligible (available_at <= deadline); timezone-
naive values reject. Comparison occurs on parsed UTC instants, not lexical offsets.
Expose a read-only eligible-ledger selector which invokes that validator and admits
only prospective=true and synthetic=false, rejecting any unsupported legacy/recovery
provenance. Same exact publication evidence repeats identical ledger bytes; different
bytes at the same immutable identity conflict. Existing outcome artifacts are not
rewritten. Missing deadline -> UNKNOWN_LEGACY/false, never a guessed default.

Use a restricted wrapper over SecureSystemStorage for journal refs/locks/exact writes;
only daily_production/prospective governed roots are exposed, no activation methods.
Test command: `uv run pytest tests/unit/test_daily_evidence_dag_contract.py
tests/unit/test_daily_evidence_dag_integration.py -q`. Retain five-session exact refs
and independent replay result in
reports/operations/daily_evidence_dag/synthetic-five-day-proof.json with explicit
synthetic=true and real_unattended_proof=false; fixture source artifacts must remain
available for replay. Replaying the complete fixture must have identical financial/
research/completion/ledger bytes and zero additional native business writer calls.

Final deployment stops on dirty release source, missing accepted ancestor, wrong
installed import, failed unit/lint/type gates, scheduler readback drift, simultaneous
old prose/native orchestration, invalid completion replay, or failed first production-
equivalent receipt. Preserve21:00 fallback and21:30 Dashboard until separate cutover
criteria prove success. Source/task setup and synthetic work may continue while
production input/permission gates remain blocked; never label those future proofs
complete based on this plan.

Review status 2026-09-07T06:21:43.208578+00:00: Architect amendments incorporated; Critic APPROVE. Contract implementation19 focused tests pass; integration not yet implemented.

Progress 2026-09-07T06:42:54.652675+00:00: journal, native core Top100 adapter/hook and historical input reader implemented in isolated worktree;76 focused tests pass. Full coordinator/adapters/five-day proof/deployment remain outstanding.

Progress 2026-09-07T06:59:17.828543+00:00: linked input revisions and internal fixed-graph runner implemented;88 focused tests pass. Full native registry and EOD seal still pending; fixture-adapter tests are not five-day proof.

Progress 2026-09-07T07:21:18.023197+00:00: single-owner7-node native core handoff and exact observation-ref recovery implemented;95 focused tests and real readonly Calendar/LOW/W80 probes pass. Full16-node DAG/Store/complete seal/five-day acceptance remain pending.

Progress 2026-09-07T08:16:40.101299+00:00: native Store prepared plan, retained-pointer commit proof, metadata-only recovery and fixed adapter implemented;182 focused tests pass. Actual Store9/3 readonly proof verified. Full-v3 synthetic CAS and five-day full-native integration remain pending.

Progress 2026-09-07T09:06:32.953954+00:00: five-session native Store segment and current native Dashboard publication verified;193 combined tests/21post-version tests pass. Full16-node five-day DAG and historical Dashboard snapshots remain pending.

Progress 2026-09-07T10:20:57.358181+00:00: native research source/Decision adapters and actual-time immutable capture implemented;37research/legacy+28fast checks pass. Full native registry, Factor fixture and16-node proof still incomplete.
