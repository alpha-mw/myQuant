# Phase12 bootstrap launch and credential-free recovery

Status: IMPLEMENTED_LOCAL_VALIDATION. Architect amendments and Critic APPROVE preceded
implementation. This is the next required Part B integration,
not completion of input provisioning or the full Phase0–15 objective.

## Actual gap and intended route

The existing execute recipe v5, bootstrap.v1 declaration, execute_daily_recipe and
native cutoff materialization already support first-day execution. The installed
launcher rejects it because cli.daily_launch and scripts.daily_launch_inspection
only accept automatic CATCH_UP requests. Thus no correct initial EXECUTE request can
take the same credential-aware launcher path.

Extend that existing launcher path for exactly an explicit current
cn-daily-production-request.v1 EXECUTE whose validated recipe is v5, has
bootstrap_ref and has no previous_completion_ref. Preserve automatic request v1/v2
behavior and all other public actions. No activation, fake initial EOD, alternative
Factor path, schedule update or live call is part of local implementation.

## Existing inputs and remaining provisioning

The source/config audit found a retained complete Industry descriptor (692
membership partitions plus native plan/capture closure), code-owned Theme v3
acquisition policy, corporate template v1 and the official-close policy. Theme
policy and corporate template are date-independent; prediction deadline policy is
date-bound and has no native default. Benchmark/Event sources must exist through
their owning producers before recipe preimages are frozen; their later mutation
cannot be hidden by replacing the request SHA.

Actual automation instructions explicitly prohibit retrospective event declarations
without owner confirmation of all seven dimensions. The pending Sep4–11 factual
question is still unanswered. Current-date standing empty-inventory authority does
not authorize filling those historical gaps. This blocks only affected production
inputs; do not manufacture no-change days.

After this launcher repair, implement installed source configuration/daily
preparation through existing producers and immutable request registration. That
remains required, alongside genuine initial bootstrap and later automatic
request/seed transition. Do not call this bounded launcher work Phase12 complete.

## Request and inspection contracts

Add one pure bootstrap recipe predicate in operations/daily_launch_contract.py
(or the existing bootstrap contract owner): validate_production_request and
validate_execution_recipe, then require current request.v1 EXECUTE, recipe.v5,
bootstrap_ref not null and previous_completion_ref null. No arbitrary caller flag
can select this profile. Callers read only exact original request/recipe refs to
identify the profile; do not walk mutable Store preimages before completed checks.

Keep cn-daily-launch-inspection.v1 unchanged for automatic CATCH_UP. Add exact
cn-daily-launch-inspection.v2 for the bootstrap profile, with existing fields plus
target_trade_date. Validate that date against the original request at the CLI and
native composition boundaries. Existing modes retain 0/10/11:
- COMPLETE_READ_ONLY: a validated native production-result.v1 EXECUTE NO_ACTION,
  either COMPLETE with the exact target EOD row, or NON_TRADING_DAY with no rows;
- LOCAL_REPAIR: result null; exact retained proof permits provider-free recovery;
- PRODUCER_REQUIRED: result null; native initial/static controls verified, possible
  acquisition remains, but this mode is never itself execution permission.

The three fields retry/owner/next-node belong to existing failures; do not invent
attempt or safe-retry state in this observational inspection.

## Native inspection order

Use a responsibility-named scripts/daily_bootstrap_launch.py, added explicitly to
the fixed native bridge module map without new operation names. Existing
daily_launch_inspection dispatches by validated request kind.

1. Read original request and recipe by SHA and validate bootstrap profile. Check
   any existing automatic pending-run ownership; a foreign ACTIVE automatic run
   blocks this explicit first-day launch. Do not modify leases or heads.
2. If the exact target completion exists, use inspect_recorded_completion's minted
   snapshot to bind the original request SHA and retained recipe; replay all native
   completion nodes. Never compare a historical recipe's old Store pointer SHA
   against today's mutable heads. Preserve native synthetic-workspace checks.
   If publish_current_dashboard is true, use the existing read-only serving gate:
   matching recorded current publication => COMPLETE_READ_ONLY; valid sealed EOD
   with missing publication => LOCAL_REPAIR; expired/corrupt/mismatched proof blocks.
   If publication was explicitly false, full native EOD replay is sufficient.
3. For a known fixed logical slot claim without EOD, use the existing
   locked_finalized_maintenance_replay reader (existing O_RDONLY lock only,
   nonblocking, no creation) to recognize a fully verified CONFIRMED_CLOSED session.
   Recheck request/recipe and native evidence before returning NON_TRADING_DAY.
   Missing/unfinalized/unsafe/contended proof is never a completed claim. Do not
   parse error strings into success or ignore a security exception as absence.
4. Otherwise validate read_execution_controls, installed/research/timing policy and
   static Store controls. Before a fresh run, require the existing native active
   Factor bootstrap baseline. Do not repeat the initial-active-parent check after
   a proven core handoff, because Factor legitimately advanced during that attempt.
5. Read exactly executions/<request-SHA>/maintenance-handoff.v1.json if present and
   bind its original request/recipe/release/date. A matching recipe-v5 cutoff receipt
   at executions/<request-SHA>/research-cutoff.v1.json is the minimum evidence for
   LOCAL_REPAIR before EOD: replay through read_cutoff_inputs(repair=False).
   The only tolerated rejection is typed DependencyInputError with reason_code
   CUTOFF_COMMITTED_OBJECT_MISSING, raised after all committed byte conflicts are
   checked. This permits repairing missing committed objects after deadline, never
   a new cutoff/source choice. Other corruption/security/clock errors block.
6. If a materialization exists, additionally use its owning fixed version/path
   readback before classifying local recovery. If no committed cutoff is available,
   validate fresh acquisition/date rules and return PRODUCER_REQUIRED. Existing
   finalized core recovery remains through execute_daily_recipe; it is not a new
   maintenance producer path. No speculative partial-source inference.

## Provider-free execution guard

Extend the existing public --no-producers flag only to this validated bootstrap
EXECUTE profile. Keep all existing automatic constraints and rejection for PLAN,
RESUME, arbitrary EXECUTE and historical/internal generated requests.

Pass the flag from cli.daily_production through dispatch_daily_request to
execute_daily_recipe. At its entry, before loop construction, provider work or
materialization writes, require one of:
- a matching valid completed EOD (serving-only recovery allowed);
- exact confirmed non-trading session proof;
- matching original handoff and validated committed cutoff/local input proof.

Otherwise fail with an explicit bootstrap no-producer precondition. For retained
handoff recovery, call existing materialization with _execute_theme=False; any
missing source/handoff after inspection must fail without acquisition or a new
cutoff. Missing committed objects may be recreated only under existing exact-SHA
cutoff repair. Fresh normal EXECUTE keeps _execute_theme=True and its original
maintenance/Factor/Calendar gates. No exception grants retry authority.

Inspection is repeated from the same request and saved inspection SHA immediately
before the shell branch. Existing shell modes0/10/11, credential helper, emitted
machine output, STARTED/ENDED receipts and0/2/3 exit semantics remain. Completed
bootstrap mode emits the validated native result without invoking a writer or
loading credentials. Local mode uses --no-producers and no inherited token;
producer mode alone reads PROJECT_ENV. No new shell profile or command string
from serialized input.

## Acceptance and stop conditions

- The real shell/installed CLI classification route accepts a valid bootstrap
  profile and still accepts old automatic profiles; malformed/non-bootstrap/
  historical/mixed request profiles reject before credentials.
- Fresh bootstrap inspection uses native baseline/static guards and makes no
  provider/canonical writes. Execution still runs the existing initial bootstrap
  before first maintenance and native Calendar succession after it.
- Exact completed bootstrap inspection/readback works after Store/Factor current
  heads advance; zero providers/writers/credential reads and same completion SHA.
- Confirmed non-trading repeat is credential-free with native proof; unsafe,
  unfinalized or foreign-slot evidence cannot become NO_ACTION.
- Committed-cutoff/materialization partial recovery works without providers,
  including after deadline; wrong SHA, missing original handoff, conflicting bytes,
  disappearing proof or no cutoff rejects before provider/loop construction.
- Full native serving proof: valid publication, missing publication repair,
  corruption and expiry; preserve publication-after-EOD and no-backwards behavior.
- Saved inspection/request/release drift and foreign automatic ACTIVE lease block.
- Focused existing initial-execution/bootstrap/launcher/materialization/automatic
  tests plus direct no-producer negatives. Real native adapters/source readers
  where available; disclose controlled install/producer/Calendar/clock/EOD seams.
- Add the new module to complete real closed-import verification. Freeze runtime
  sources before long native checks. Final installed/full-DAG bootstrap remains
  Phase15 unless actually exercised here; source provisioning remains Part B.
- No main holdings/cash/events, Macro veto, policies, provider calls, scheduler,
  deployment, old archived release or pointer-SHA manipulation in local work.

## Accepted architecture amendments (authoritative)

The Architect's initial10 findings and retained-ref follow-up are accepted. This
section supersedes conflicting order/flag/alias text above.

Preserve the public production daily-close --no-producers contract unchanged.
Add the explicit internal daily_launch --mode recover. It re-reads/rederives the
saved inspection under the same verified native context and requires LOCAL_REPAIR.
Automatic v1 dispatches the existing daily_close(no_producers=True). Bootstrap v2
dispatches daily_close(committed_recovery_only=True), a private fixed restriction
not a new public CLI flag or a field in serialized requests. The shell's existing
mode10 branch calls this recover mode with no token. Modes inspect/validate/emit
stay read-only. Recover emits the native-validated result and public0/2/3 exit;
internal10/11 never escape as task outcomes.

Inspection v2 has exactly existing fields plus target_trade_date and recovery_scope.
Its exact matrix:
- COMPLETE_READ_ONLY / NONE / validated EXECUTE NO_ACTION result;
- LOCAL_REPAIR / SERVING_ONLY / null;
- LOCAL_REPAIR / COMMITTED_DAG_RECOVERY / null;
- PRODUCER_REQUIRED / NONE / null.
Missing committed objects use COMMITTED_DAG_RECOVERY, which explicitly permits
exact committed-object repair, materialization with _execute_theme=False and
run_materialized_native_input(resume=True), including governed Research, Decision,
corporate, Store CAS and Dashboard node writers. It never permits maintenance,
Calendar/provider/Theme acquisition, Factor rollover, a new cutoff/source choice
or a different Store plan. It is credential-free, not writer-free; no new authority
is granted by inspection or the private restriction.

Warm inspection order: completed EOD -> exact finalized closed-session proof ->
request-owned handoff -> committed cutoff and selected materialization -> only
then fresh acquisition/date, active Factor parent and mutable Store/Event/
benchmark checks when no already-proven warm state exists. Never call initial
parent/current-preimage guards after the attempt advanced those heads.

At actual explicit bootstrap execution, acquire existing AutomaticRunStorage.locked
nonblocking, reject foreign ACTIVE automatic lease, and hold the lock across
execute_daily_recipe. Do not create/update a bootstrap pending lease. Preserve
automatic lock -> day/maintenance lock -> Dashboard publication lock order. This
execution recheck closes the race after read-only inspection. Ordinary automatic
derived EXECUTE remains inside its existing controller capability, never nested.

Retained identity is content-addressed and distinct from transport provenance.
Do not add another launch-binding artifact. The shared pure binder derives:
E = results/operations/daily_production/CN/<target>/executions/<request-sha>,
request_ref=E/inputs/request-<request-sha>.json,
recipe_ref=E/inputs/recipe-<original-recipe-sha>.json,
handoff=E/maintenance-handoff.v1.json.
Require exact full retained refs, byte equality to original launcher request and
recipe bytes, the embedded original recipe_ref, target, release/install, bootstrap
declaration and v5 current profile. Original launcher refs remain exact in the
inspection/saved-file transport boundary. Identical original bytes at another
external path may address the same SHA-keyed execution only after this binder
passes; a saved inspection cannot be moved to that other caller ref. Missing
original input is never reconstructed from a retained copy.

For completed EOD use its minted snapshot and
read_recorded_maintenance_handoff(snapshot), with the handoff's exact retained
refs. The current snapshot's8 roles include recipe but not a separate request role;
its owning recorded-handoff reader already safely reads/rechecks the retained
request. Re-read those exact derived retained bytes for the shared binder; no
new snapshot field or locator scan. Partial recovery uses read_maintenance_handoff.
Recheck both original and retained bytes before return/dispatch and after replay.

The private committed restriction is enforced again inside execute_daily_recipe,
before loop/provider paths. Completed EOD performs only native serving recovery;
native confirmed closed returns only no-session result. Otherwise, under the
existing day lock, re-read exact original handoff/cutoff/materialization, accept
only successful cutoff replay or fixed typed CUTOFF_COMMITTED_OBJECT_MISSING,
repair only matching committed bytes, force _execute_theme=False and run the
materialized DAG. Every other state raises a committed-recovery precondition.
Immutable conflicts, other typed errors, security/clock/corrupt or disappearing
proof cannot enable recovery. Fresh normal EXECUTE remains through existing guards.

Finalized non-trading proof must have CONFIRMED_CLOSED for the exact target, no
core completion, fixed target-2020-execute claim/attempt ancestry and unchanged
request/recipe bytes. Existing O_RDONLY nonblocking maintenance lock is allowed;
no lock or proof is fabricated. Busy, unsafe, unfinalized and changed proof fails.

Acceptance additionally covers warm heads advancing, automatic run racing between
inspection/execution, committed missing-versus-conflicting objects, downstream
writers with upstream/provider/rollover calls forbidden, changed closed-session
proof, differing original/retained paths, corrupt retained bytes, wrong retained
recipe path, old direct EXECUTE recovery without a new binding, missing original
inputs and saved-inspection ref substitution. Full native EOD/serving validation
and public --no-producers rejection remain unchanged.
