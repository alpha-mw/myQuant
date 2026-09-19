# Phase12 configured source preparation in the existing launcher

Status: IMPLEMENTED_LOCAL_VALIDATION. Architect revision approved, then Critic
approved before implementation. Previous work completed native Event/benchmark producers
and256 related tests. This change connects config -> Calendar/native sources ->
daily_prepare -> existing launcher. It does not deploy schedules or certify live EOD.

## Entry and authority

Extend scripts/operations/run_cn_daily_slot.sh with a mutually exclusive
--daily-source-config / --expected-daily-source-config-sha256 pair alongside its
existing exact-request pair. Keep all shared installed/release arguments,2020-only
DAG scope and legacy1620/1720/1820 behavior. Config profile requires existing config
v1 and publish_current_dashboard=true; one-off explicit preparation remains able
to use false. Existing launcher inspection, local recovery and daily-close dispatch
remain the sole DAG execution path after request selection.

Add installed cli/daily_sources.py and one fixed native bridge operation owned by
scripts/daily_source_inputs.py. Modes inspect, provision, emit and select never
accept executable config or caller-selected provider. Every call binds config and
the exact supplied install ref to the running verified native context. Only the
source acquisition mode can receive PROJECT_ENV credentials. Preparation itself
is not provider authorization or EOD admission.

## Registered source-request locator

Use one explicit locator, not a latest/mtime/directory scan:
results/operations/daily_production/CN/source-configs/<config-SHA>/request.v1.json.
Exact schema cn-daily-source-request.v1 fields: schema_version, config_ref,
trade_date, calendar_ref, raw_calendar_ref, preparation_commitment_ref, request_ref,
previous_locator_sha256 (null for initial), authority=FALSE_AUTHORITY,
content_sha256. Request/commitment paths must be the existing exact config/date
preparation paths and the existing commitment must reconstruct the same request.
It is only a transport locator, not an execution lease or completion claim.

All locator writes/CAS occur under the existing AutomaticRunStorage lock. Preserve
its pending lease. Refactor daily_prepare's existing locked publication just enough
to invoke a fixed internal locator publication hook after its immutable commitment
is written and before any generated objects/request. A failed hook prevents
request publication. Existing standalone daily_prepare behavior stays unchanged.
The hook is code-owned and not accepted from JSON/CLI; it writes only this fixed
locator and validates config/date/commitment/request binding and expected old bytes.
No new bootstrap execution binding or financial authority is introduced.

Selection priority, before current source/clock-dependent acquisition:

1. Existing ACTIVE automatic pending request: validate exact request/release and
   return that original ref. No new source selection or locator takeover.
2. Registered locator: validate config/ref/commitment derivation. Missing generated
   objects use the existing committed preparation repair, including across midnight.
   Corruption/missing commitment blocks. A same-day request is returned unchanged.
3. Older locator without a native completed target remains selected for the existing
   launcher to resume or report its exact blocker. Do not skip an unfinished
   bootstrap merely because the current date changed. Expired unstarted requests
   keep existing native deadline/missing-history blockers; do not manufacture
   historical Theme/Event inputs or silently retime/restart them.
4. To advance an older locator, prove the target's native completed EOD and bind its
   original request identity (original/retained aliases use the existing binders).
   Use historical native EOD replay, not current Dashboard freshness/serving repair;
   old page expiry must neither block a new day's inputs nor be renewed. Automatic
   runs must be IDLE/no ACTIVE lease before advancement. Required publication=true
   supplies the normal native head for subsequent automatic request generation;
   missing/corrupt head stays a blocker, never inferred from directory contents.
5. If no locator applies, reuse this date/config's exact existing preparation
   commitment before fresh source capture; never invent an alternate original ref.

The publication hook closes the first-bootstrap cross-midnight gap: no request
can be handed to the configured launcher before its locator exists. A crash
between preparation commitment and locator leaves no newly published request;
same-day recovery can adopt that exact commitment. Uncommitted old source captures
do not establish that a DAG request ran.

## Calendar and source preparation

Use fixed paths under the current date/config preparation root, sources/:
calendar-capture.v1.json, calendar.json and calendar.raw.json. Calendar capture
schema cn-daily-source-calendar.v1 has exactly schema_version, config_ref,
trade_date (observed Shanghai civil day), calendar (native receipt including the
fixed raw_response_path), raw_response_base64, authority=FALSE_AUTHORITY.
The embedded wire bytes preserve exact provider bytes. Native receipt/raw replay
must pass and bind fixed paths/sha/config/date. Persist this one immutable capture
before materializing the two files; missing files can be recreated only from the
committed bytes, conflicts reject. No source timestamp is rewritten on retry.

Fresh capture calls only acquire_close_session_authority with the fixed official
client. Before each call write one immutable request-start marker, using only
calendar-request-1.json and calendar-request-2.json (exact config/date/start time
and fixed API name). At most two calls per date/config; a marker without capture
proves no success, but consumes its request budget. No open-ended retry. Fresh
observed date must equal actual Shanghai day and observation must
not be future. CONFIRMED_CLOSED returns a source NON_TRADING_DAY result with null
request and no Event/benchmark/DAG producer call; it never bootstraps Friday from
a weekend observation. Existing nontrading capture can be replayed token-free.

For MATCHED_OPEN, before index requests or current Event writes, load native
Storev3/performance frontier, Calendar open prefix and current Event generation.
Require every historical missing OPEN day to have its explicit native closure;
return exact missing_event_dates otherwise. Only the current day can be supplied
by the standing-policy Event command. No retrospective declaration is produced.
Verify config static source/policy/install/loop controls before providers where
possible; known missing inputs block instead of consuming benchmark quota.

Capture/publish missing benchmark days through the installed Tushare producer,
using exact Calendar-derived required dates. Existing complete generation needs
no acquisition; stale/missing compatibility bytes use a small owning benchmark
repair helper under the existing publication scope, requiring exact current
pointer SHA and using its immutable rows without reserializing/replacing native
generation bytes. Use a deterministic generation ID from a full SHA of canonical
config ref/date/selected original preimage (within native ID length bounds).
Persist a source plan before invoking benchmark capture so the original preimage
and requested date set survive CAS/interruption. Exact schema
cn-daily-source-plan.v1 fields: schema_version, config_ref, trade_date,
calendar_capture_ref, store_pointer_ref, event_pointer_ref, benchmark_pointer_ref,
benchmark_start_date, benchmark_end_date, benchmark_required_dates,
benchmark_generation_id, authority=FALSE_AUTHORITY. Benchmark range/generation
fields are null when no acquisition is needed. Unknown/foreign advancement blocks
or is revalidated through the same committed native publication; no reselection.

Next call the repaired owning daily Event command with exact Calendar/raw inputs,
then release the source-stage automatic lock and invoke existing daily_prepare,
which reacquires that lock, validates final native preimages and registers its
request/locator. Recheck any competing ACTIVE work at each boundary. Source plan
refs are historical preimages; our own valid benchmark/Event CAS is expected and
must not be mistaken for source corruption. Native readback determines completion.
When a missing-date range contains already present OPEN days, required_dates is
the full Calendar OPEN set inside the requested start/end range; overlaps must
pass the producer's exact normalized equality check. It cannot omit provider rows
or relabel missing data as closed sessions.

## Source inspection/result and shell handoff

Exact cn-daily-source-result.v1 fields: schema_version, mode, config_ref,
release_install_ref, trade_date, request_ref, calendar_capture_ref,
preparation_commitment_ref, authority=FALSE_AUTHORITY. Mode mapping:
REQUEST_AVAILABLE=0 (request ref nonnull), NON_TRADING_DAY=0 (request null),
LOCAL_PREPARATION=10 (request may be null), ACQUISITION_REQUIRED=11 (request null).
Other missing/corrupt inputs use typed BLOCKED/exit2, unexpected failures exit3.
No source result claims EOD completion. Native launcher validates returned requests.

Inspect is read-only and provider-free. Provision rederives state under the lock;
--no-providers restricts it to committed repair and existing local source work.
It cannot acquire Calendar/index data if inspection changes. Provision returns
only REQUEST_AVAILABLE or NON_TRADING_DAY after source and preparation readback.
Source inspection is saved in the existing launcher attempt directory. emit/select
read the exact saved inspection path/SHA and rederive it; emit only prints the
validated result, select supplies the same JSON request ref for shell parsing.
Saved-path grammar is the existing slot-2020 timestamp/PID attempt root with fixed
source-inspection.stdout.json name. No eval or shell interpolation of JSON.

Shell reads PROJECT_ENV only for source mode11 or subsequent DAG mode11. Reuse the
same in-memory token for that invocation if both need it; do not create duplicate
credential-preflight identities. All inspection/local repair children explicitly
receive no token. Once request is selected, run the existing daily_launch
inspect/validate/emit/recover or production daily-close block unchanged. Source
failure cannot fall through into legacy maintenance/Factor recovery. Preserve
STARTED/ENDED and truthful0/2/3 exits. No automation prompt/config is edited here.

## Acceptance

Exercise real source composition with synthetic native Calendar/Store/Event/
benchmark fixtures, commitment interruptions and native readers. Cover first
bootstrap, saved request reuse, partial bootstrap across midnight, older completed
source locator advancement without refreshing an expired old page, automatic
pending precedence, same-day zero-provider repeat, nontrading day, missing historic
Event facts, missing/static/changed input, source-plan CAS recovery and foreign
advance, scope/locator tampering, no-provider enforcement and lock contention.
Use real shell tests for credential routing, saved-output substitution, early
validation, failures and no legacy fallthrough. Disclose controlled install/EOD
proof seams. Run relevant source/prepare/launcher/automatic/bootstrap/full-import
checks and static gates. Final installed full-DAG and actual unattended proof
remain the final acceptance work; do not claim them from these local tests.

## Architect revision (supersedes earlier locator timing)

The locator must exist before any benchmark/Event CAS, not merely before request
publication. Exact locator fields are schema_version, state, config_ref,
trade_date, calendar_capture_ref, calendar_ref, raw_calendar_ref, source_plan_ref,
preparation_commitment_ref, request_ref, previous_locator_sha256, authority and
content_sha256. States: PREPARING_SOURCES (last two output refs null) and
REQUEST_AVAILABLE (both exact nonnull refs). A complete source plan and replayed
MATCHED_OPEN Calendar are prerequisites for PREPARING_SOURCES. NON_TRADING_DAY
creates no locator/source plan because it performs no source CAS or DAG request.

Allowed transitions only: none -> PREPARING_SOURCES; same date/inputs
PREPARING_SOURCES -> REQUEST_AVAILABLE; proven old completed REQUEST_AVAILABLE ->
new-day PREPARING_SOURCES. Same-date config/Calendar/plan/request changes reject.
Cross-day PREPARING always reads its original plan/Calendar. Missing historical
event evidence is a factual blocker, never replaced with current-day evidence.

Add a narrow source-config storage owner, borrowing the active same-workspace
AutomaticRunStorage lock. It can only write the fixed config locator plus
source-configs/<config-SHA>/history/<old-SHA>.json. Initial write requires absence;
updates preserve original prior bytes in history before exact-SHA CAS using0600
temporary/fsync/atomic replace/directory fsync/readback. Replays verify the full
recorded history chain, legal transitions and bindings. Unknown/missing/conflicting
history blocks; identical bytes do not rewrite. No public generic projection writer.

The preparation hook CASes only a matching PREPARING locator to REQUEST_AVAILABLE
after the existing preparation commitment and before generated request objects.
It runs under daily_prepare's newly reacquired automatic lock, checks pending and
the prior locator again, and cannot be supplied via CLI/JSON. Source CAS and
preparation stages never nest a second automatic lock or reverse native lock order.

ACTIVE pending can be adopted only if this config has a valid REQUEST_AVAILABLE
locator with exactly that request/ref/release/commitment. Otherwise report the
foreign pending conflict before any provider/source operation. No takeover.

Add event_generation_id, event_operation (NO_ACTION_EXISTING/CREATE_CURRENT_EMPTY)
and event_source_mode (null/CALENDAR_ONLY) to the exact source plan. New generation
ID derives from full canonical config/date/old Event SHA/Calendar capture identity.
No-action plan requires the original current pointer and original valid date row.
Create plan permits only original preimage or its exact planned successor.

Event recovery uses the existing native immutable generation as its source-fact
commitment. A staged generations/<planned-ID>.v1.json can be resumed only after
native schema/seal checks, exact parent closure-set equality, the single planned
new date/policy/Calendar source, original15:30 cutoff and actual-original seal time
within that date, <=now. It never reconstructs a damaged/absent generation or
re-signs old facts. Add a bounded owning reader/recovery entry to the existing
Record manager so this validation and publication use Record operation -> Event
current lock. Existing planned successor is adopted only with original parent SHA
and exact candidate bytes. If neither staged nor published candidate exists after
the target day, CREATE_CURRENT_EMPTY is blocked; fresh historical creation remains
forbidden. Normal same-day missing candidate calls the repaired existing command.

Benchmark matrix: no v2+original pointer permits fresh capture only with provider
mode; v2 permits exact parent/raw candidate reconstruction without providers;
candidate current permits NO_ACTION/compatibility repair; foreign advance rejects.
No-acquisition repair uses current exact immutable generation under publication
scope and never reserializes native generation bytes.

No-provider mode may replay/materialize committed Calendar, inspect/resume a
PREPARING locator, finish a committed benchmark capture, repair compatibility,
create the planned Event only while still same-day/aftercutoff, adopt its existing
committed generation across days, and repair preparation/request objects. It may
not create Calendar request markers, invoke providers, acquire missing benchmark
capture, create historical Event, change dates/preimages/generation/request or
treat token absence as success.

Two Calendar markers are consumed before each one native acquisition call, with
no hidden retry. With no complete capture and two existing markers, block. If the
clock crosses date during acquisition, consume the marker but do not commit a
capture or locator under the wrong date. All present marker fields must bind the
same config/date/fixed endpoint and source path. Retry never reclaims markers.

Old locator advancement uses historical native EOD proof, never current serving
expiry. Bootstrap uses its original/retained byte binder and completed snapshot.
Automatic uses existing canonical completed-prefix adoption/replay semantics; no
old Calendar is passed to a fresh resolver and no clock is backdated. Missing or
corrupt required head still blocks next automatic preparation.

For shell transport, daily_sources select is a dedicated internal output mode
which emits exactly two validated single-line printable-ASCII values (request
path then lowercase64hex SHA), only after rederiving saved source output. Shell
reads quoted newline-separated array elements and rechecks existing path/SHA
grammar; no eval, jq or command construction. All other modes emit canonical JSON.
This explicit internal transport exception does not change other CLI contracts.

Additional acceptance: each source/locator/preparation boundary interrupted and
resumed across midnight; original staged Event generation retained/adopted without
new seal; absent late Event blocks; foreign ACTIVE config request blocks early;
locator history/CAS/same-date mismatches reject; old expired page is not touched
when historical native completion/head permits advancement; source-stage token
cannot leak into later inspection or credential-free DAG recovery.

## Implemented behavior and local evidence

The existing launcher now accepts the exact static source-config pair and calls
installed `quant_investor.cli.daily_sources`. The fixed bridge operation is
daily_source_inputs. Source results feed the unchanged daily-launch inspection,
recovery and public daily-close path. Select's internal exit10 represents a
verified nontrading result with no request text; shell emits its validated JSON
and returns public0. It never leaks internal10/11 as a task outcome.

The fixed preparation hook receives only in-process expected source preimages
and prior locator SHA from the native source composition. It compares them with
the existing preparation commitment, then upgrades the locator before request
files. This adds no JSON/CLI callback or financial authority.

Cross-day source recovery is deliberately narrower than permission to backdate
research. A committed Event generation may be adopted with its original seal.
An existing preparation commitment can repair request objects across midnight.
If that preparation commitment never existed before midnight, CURRENT research
preparation still raises its native freshness blocker; no time or policy is changed.

Local evidence:290 related tests passed37.41s, followed by52 focused source/CLI/
shell tests including actual competing source-run lock contention20.57s (overlap).
Black/flake8 passed12selected files; mypy passed8sources; zsh syntax and the two
new helpers in the existing manager passed separate checks. Receipt and exact
source hashes: `.agent/acceptance/phase12-configured-launcher-validation.json`.
These tests control install/Factor and old-EOD proof boundaries, and use synthetic
native data plus a controlled interpreter for real shell routing. Actual installed
source config, release deployment, live bootstrap, scheduler readback and an
independent unattended receipt remain required for Phase12/15 acceptance.

## Exposure input catalog

For current recipe v5/v6, `exposure_rows_ref` may reference an explicit
`{"schema_version":"cn-daily-exposure-catalog.v1","rows":[...]}` document.
Rows use the existing native declaration fields. The complete catalog is bound
by its original SHA; after Top100 and the modern focus Theme/PIT handoff are
verified, the collector selects Top100 plus the fixed focus companies. Frozen
source-bundle replay independently repeats that selection. Unselected physical
source files and declaration times are not claimed as verified evidence.

Legacy list inputs remain exact already-scoped declarations, including their
existing rejection of extra/duplicate companies. Historical-timing or non-focus
profiles cannot opt into the catalog. Missing facts retain the existing native
policy; selected facts still require valid physical sources, SHA, semantic and
source-time checks. The catalog envelope is administrative custody and supplies
no aggregate availability timestamp.
