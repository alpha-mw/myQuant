# Phase12 — existing launcher integration and three daily responsibilities

Status: Part A COMPLETED_LOCAL_VALIDATION after seven Architect amendments and
Critic APPROVE. Final330 checks PASS38.63s; nine-source mypy,13-file Black, scoped
flake8/public CLI complexity10 and zsh syntax pass. Exact source/test hashes and
limitations: `.agent/acceptance/phase12-launcher-validation.json`. Real current
interpreter/shell invalid-install rejection retained native STARTED/ENDED without
credential or native claims; positive installed/EOD boundaries in unit tests are
explicitly controlled. No scheduler or live-provider mutation occurred. This phase remains responsible
for both execution wiring and concrete daily input/scheduler cutover. Part A below
repairs the existing launcher; Part B input provisioning and task migration remain
required and cannot be claimed complete from launcher-only tests.

## Part A: same launcher, exact automatic request

Extend `scripts/operations/run_cn_daily_slot.sh` with one mutually exclusive DAG
profile. Required arguments are `--daily-production-request` and
`--expected-daily-production-request-sha256`, `--release-repository-root`,
`--release-install-input` and `--expected-release-install-input-sha256`.
Existing --python/--expected-import-root/--workspace-root/--run-root remain.
Require logical slot2020. Python/workspace/run/import/repository roots are absolute;
request and release-input paths are canonical ASCII workspace-relative refs. Both
hashes are lowercase64-character SHA256. For this DAG profile, run_root is exactly
workspace_root/data/private/cn_daily_maintenance, matching the existing controller.
Reject partial groups, invalid path types/SHAs,
Factor-context flags, scope-transition and retirement flags before running the
interpreter. Existing maintenance-only slots and legacy profile keep their exact
behavior until the explicit scheduler cutover.

Verify installed import origin and preserve the existing native launcher
STARTED/ENDED receipts. The DAG branch calls a code-owned installed helper through
`python -I`, never inserts arbitrary sys.path, and verifies the exact clean release
input through the same fixed native bridge as public daily-close. Accept only
automatic request v1/action CATCH_UP/collection v2 for this scheduled profile.
Bootstrap remains the existing explicit first-day setup, never a fabricated seed.

Add a fixed installed `daily_launch_inspection` operation in the existing
daily-production module. It reads exact request/release refs, resolves or replays
the existing automatic resolution and validates all native input/Calendar/EOD
evidence without creating a resolution, lock, claim or provider call. Output
schema `cn-daily-launch-inspection.v1` has exactly `schema_version, request_ref,
release_install_ref, mode, result, authority`; mode is one of:

- COMPLETE_READ_ONLY: an existing frozen resolution and derived files replay all
  its EOD rows successfully, with current-serving record/bytes valid when required.
  Its matching pending record must already be IDLE. A missing or ACTIVE matching
  lease needs LOCAL_REPAIR even when all financial evidence is complete.
  result is the exact existing automatic-result wrapper with NO_ACTION. Emit it
  without reading credentials or calling any executor/publisher.
- LOCAL_REPAIR: native EODs are complete but automatic metadata or serving remains
  incomplete and current pending publication is still within its original native
  valid_through; or the same pending current request is expired and needs the narrow
  existing EXPIRED_UNSTARTED recovery. result is null. Dispatch with no producers
  and no credential access. Recheck all constraints under the automatic lock.
- PRODUCER_REQUIRED: at least one required native EOD is absent and exact source
  ownership/preconditions are present. result is null. Use existing PROJECT_ENV
  credential preflight, then the normal public daily-close automatic dispatch.

Invalid source ownership, unavailable required input, corrupt/pending conflicting
request or release mismatch raises an expected typed blocker before credentials.
An ACTIVE lease for a different exact request is a conflict; do not relabel it a
new producer opportunity. The later input-preparation path must select the
appropriate pending/current request rather than make the launcher guess.
Still-current RECORDED_EOD_PUBLICATION_EXPIRED and
EVIDENCE_SEALED_PUBLICATION_EXPIRED are typed blockers, not repairable presentation.
Corrupt records/bytes are blockers. A later fresh request can consume the EOD as
historical evidence through Phase11; inspection does not retime that scope.
No arbitrary caller Boolean can mark work complete. Inspection grants no execution
authority and must not consume retries or infer a successful scheduler run.

Inspection is exposed only through a fixed CLI helper used by the launcher, with
validated result fields and installed origin. Its internal exit dispatch is
0=COMPLETE_READ_ONLY,10=LOCAL_REPAIR,11=PRODUCER_REQUIRED,2=expected blocked,
3=unexpected failure. Save inspection stdout/stderr under the exact current
launcher attempt; never locate results by latest, mtime or directory scan. The
regular public result contract retains0/2/3. Extract a completed inspection's
validated nested result through a fixed JSON reader, not shell evaluation.
The wire contract is exact: COMPLETE_READ_ONLY requires action CATCH_UP and exact
NO_ACTION outer/nested execution state with original request/release/resolution
bindings; other modes require result=null; authority is all-false and extra or
unknown fields fail. Both the inspection and its fixed reader use verified native
bridge operations; no host reconstruction fallback. The reader validates the exact
saved stdout file/SHA, request/release refs and schema, and repeats the native
inspection to reject state/byte drift before the shell branches. It does not parse
fragments with grep/jq, evaluate a command or trust an unverified mode string.

## Restrict execution for local repair

Add optional `--no-producers` to public `production daily-close`, accepted only
with an automatic CATCH_UP request. It may write the automatic resolution/lease
or repair publication of an already sealed EOD; it never starts maintenance,
research/Store/Dashboard node producers or provider acquisition. Pass a private
restriction through the existing dispatcher/controller, default unchanged.

The existing automatic engine handles pending/expired lease recovery first. Then
the no-producers route must reprove every required EOD complete before any new
producer call. Missing evidence raises a blocker without a producer callback,
even when initial inspection said LOCAL_REPAIR. An expired unstarted closure
remains immutable; the old request is not rewritten or immediately rerun.
The missing/invalid completion blocker is AUTO_PRODUCERS_REQUIRED before binding
persistence, maintenance or materialization. Validate flag applicability before
opening the native bridge. Successful EXPIRED_UNSTARTED closure of the supplied
request still returns its expected blocked result, never completed daily-close.

Add read-only support to the existing Part A controller for inspection, sharing
its full native completion/Calendar/ancestry and row/result validation. A missing
day stays incomplete with no binding/maintenance/materialization writes. Current
serving is checked through native `observed_serving_status`, matched to the exact
replayed completion, and requires RECORDED_EOD_PUBLICATION. That reader is never
substituted for native EOD admission. Historical rows keep the approved Part A
semantics. Read-only inspection cannot mint a receipt, publish or renew freshness.

No-producers actual dispatch keeps the normal Phase9 publisher, allowing an
already sealed current EOD to recover its valid presentation without Tushare.
Existing expiry rules continue to reject unproven or expired new publication.
If the day becomes incomplete between inspection and dispatch, the restriction
fails closed. It cannot become a new producer run just because credentials are
absent or an input file disappeared.

## Credentials and receipts

The DAG branch does not run the legacy separate Factor recovery handler: the
automatic/native EOD path owns its exact replay/recovery, preventing a second
producer entry. Preserve the old pre-credential recovery order for legacy callers.
The branch exits after its own inspection/dispatch and cannot fall through to
legacy Factor recovery, transient-veto recovery or direct daily-maintain blocks.
Only PRODUCER_REQUIRED reaches the existing `.env` token reader and credential
receipt. Use only the fixed official Tushare destination and preserve all existing
token secrecy/PROJECT_ENV checks. Do not read credentials in these local tests.
COMPLETE_READ_ONLY and LOCAL_REPAIR explicitly remove inherited TUSHARE_TOKEN
from the child environment and make no provider calls. All new dispatch remains
bound to the same request/release hashes in inspection and launcher output.

Save `daily-close.stdout.json` and `daily-close.stderr.log` in the exact launcher
attempt directory; propagate the real daily-close exit. Reuse native STARTED/ENDED
and do not call a partial run successful. Producer output and launcher receipt
remain separate from a genuine scheduler receipt and from financial effectiveness.
ENDED records the actual branch exit: complete0, expected blocker2, unexpected3,
or the normal public producer exit0/2/3. Internal inspection10/11 never escape as
the launcher/public result. Retain inspection logs on every path after STARTED.

## Acceptance for Part A

- Real shell rejects mixed/partial flags before interpreter/credential/claim use.
- Source/release/request mismatch blocks before credentials, no producer callback.
- Completed fallback emits verified NO_ACTION with no credential read or any
  provider/producer/publication call. Native serving or EOD corruption cannot pass.
- LOCAL_REPAIR uses no-producers and no credentials; publication can repair only
  a validated existing EOD. Missing-after-inspection never starts a producer.
- PRODUCER_REQUIRED performs PROJECT_ENV preflight and calls the same public
  automatic route once; failure/exit and exact STARTED/ENDED/output refs survive.
- Request/release drift between inspection and dispatch fails; no latest scan,
  mutable-head substitution, retired fallback or second Factor invocation.
- Existing launcher mode tests, automatic/Part A/public CLI/serving regressions,
  native closed-import checks and relevant static checks. Synthetic/fake transports
  and controlled outer EOD/installed seams must remain clearly labeled.

## Part B still required

Provide the exact automatic request to each scheduled invocation through a
code-owned input-preparation path. It must freeze native Calendar/raw evidence,
use explicit per-date source/recipe/native-input ownership and preserve pending
request identity across EOD/fallback. Missing event, owner, Industry or exposure
evidence remains a dated blocker; never create an empty event assertion or claim
unavailable inputs ready. Inspect existing native producers before adding a new
source abstraction. Initial completed seed/setup stays explicit.

Prepare current-source installed contexts, then update the existing EOD/fallback/
Morning tasks in place. Retain the authorized logical2020/full-producer and21:00
fallback timing until a reviewed concrete schedule change. Morning uses native
v3 prior EOD/quote/policy consumption and never maintenance. Address the21:30
read-only task as part of the exact cutover; preserve it until that boundary.
Keep weekly/A_quant tasks, observation heartbeat, models and notification choices.

Saved producer/fallback prompts and prior owner authorization already authorize
their defined scheduled live-provider/scoped writes; do not ask again for that
same boundary. No off-slot live run or live mutation is performed in Part A.
Scheduler/deployment claims require actual readback and a real independent run;
all-source installed and full-DAG gates remain Phase15. Part B needs its concrete
review before input/scheduler changes, without shrinking the full Phase0–15 goal.

Read-only fixed-path provisioning audit additionally confirmed that the main
workspace has neither a native daily_production/CN directory nor either completed
head file. The frozen7b26 Factor context exists, but is not a native EOD seed.
Record: `.agent/acceptance/phase12-provisioning-before.json`. Part B must handle
explicit initial bootstrap and real source provisioning before actual cutover.

The DAG profile also fails with exit3 if native ENDED persistence fails. It never
prints a successful launcher completion merely because the daily result was0;
legacy failure behavior remains unchanged. All subsequent replay/repair still
uses the immutable request and existing automatic controller.
