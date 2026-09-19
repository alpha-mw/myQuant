# Phase12 daily request preparation

Status: IMPLEMENTED_LOCAL_VALIDATION. Architect amendments accepted and Critic
APPROVE before implementation. This closes date/source-to-request assembly. Provider
acquisition, scheduler wiring and unattended acceptance remain separate work.

## Bounded result

Add an installed package-only preparation command which reads an exact reusable
configuration and an exact native Calendar receipt/wire pair. It registers the
existing bootstrap EXECUTE v1 or automatic CATCH_UP v2 request and recipe v5.
The result is an input reference for the existing launcher, never execution or
investment authority. No new production request, recipe, Calendar, event,
benchmark, pending lease, completion or activation schema is introduced.

Configuration v1 contains fixed CN/strategy/graph scope, release/install and loop
refs, research/store/timing policy refs, Theme acquisition and corporate template
refs, Industry/exposure refs, risk-free ref, nullable explicit prediction deadline
local time, publication boolean and nullable initial completion seed ref. It has
no date placeholders, executable commands, provider endpoint or credentials.
Fundamental/Macro use the existing MAINTENANCE_STAGE source mode. Theme uses the
existing acquisition profile. A null prediction deadline means no prediction
policy; it never defaults to the research acquisition deadline.

## Preparation and reuse

1. Validate config bytes/SHA, exact fields and all referenced input bytes. Use the
   existing installed release verifier before command dispatch. Native static
   policy/loop readers remain authoritative. Source times are unchanged.
2. Replay the exact native Calendar pair and derive its observed Shanghai date.
   Look up only that date's exact registry path first. For fresh preparation,
   require an aware clock, the current Shanghai civil date and no future Calendar
   observation, then classify_requested_session. CONFIRMED_CLOSED returns
   NON_TRADING_DAY and no request
   writes; it never bootstraps the preceding Friday.
3. Registry is under the existing daily journal root, preparation/<config SHA>/
   <date>. A commitment records exact config and Calendar refs plus the canonical
   generated objects. Persist it before objects; register request last. Repeat
   reads the commitment, checks original config/Calendar bytes and generated
   objects, and repairs only missing committed objects. Conflicting bytes reject.
   Repeat can repair across midnight: it never selects new mutable heads,
   retimes inputs or acquires providers.
   An existing commitment for that date with another Calendar ref rejects.
4. Serialize fresh registration and committed repair using the existing
   AutomaticRunStorage lock. Do not write the automatic pending lease. ACTIVE
   pending work blocks new preparation with its exact request ref; an already
   committed request may be returned only if pending is absent/IDLE or matches.
5. Select a structurally valid current head pair or configured seed before native
   source preparation. This is a locator only; full native EOD replay remains the
   launcher's gate. If the locator already equals target, register an empty
   automatic collection without new Store/Factor/source preparation. Otherwise,
   fresh preparation reads the fixed Store/Event/benchmark pointer paths with
   native owning readers and exact SHA rechecks. Validate registered Store v3 and
   performance frontier. Require Calendar coverage from the official frontier
   through the target, event closure for every missing OPEN day, and all three
   native benchmark rows for those dates. Report missing dates explicitly. Do
   not create events, retrospective declarations, prices, or a compatibility
   benchmark authority. Check Dashboard benchmark compatibility bytes against
   that exact native generation. Later execution still performs its full gates.
6. If a valid current Dashboard head pair or configured seed exists, build the
   existing automatic request/one-day v3 collection. Head+seed follows the native
   resolver's exclusion rule; choose null seed once a head pair exists. Never
   scan for a latest completion. Missing historical recipes remain the existing
   automatic resolver's explicit blocker; no fabricated historical sources.
7. Otherwise require native ACTIVE Factor baseline on Calendar's immediate
   previous OPEN date and no previous/target EOD. Build the existing bootstrap
   declaration from that exact verified parent and frozen Store preimages.
8. Construct prediction policy only from the explicit configured local deadline,
   recipe v5 CURRENT_POST_ACQUISITION and existing request schemas. Run owning
   contract validators plus package-owned static control readers before commitment.
   Store policy exact bytes are bound here; its full native scope check remains
   the existing launcher/Store gate (the owning parser is script-side). Read every
   referenced source and recheck input bytes and native pointer identities before
   and after registration. Successful preparation is REGISTERED_INPUTS_ONLY;
   neither success nor failure implies execution admission or EOD completion.

## Files and acceptance

Own operations/daily_preparation_contract.py (pure config/assembly),
operations/daily_preparation.py (native source/read/register), and
cli/daily_prepare.py (installed entrypoint); reuse JournalStorage without widening
its writer root or the native bridge. No existing launcher/scheduler change yet.

Focused tests must cover native Calendar weekday/weekend/holiday/date mismatch,
explicit/null prediction deadlines, bootstrap parent mismatch, missing Event and
benchmark dates, native pointer drift, stale compatibility bytes, complete prefix,
automatic seed/head selection, pending conflicts, commitment crash recovery and
same-byte/mtime repeat, corruption/symlinks and package-only installed dispatch.
Use existing native Event/benchmark/Store fixtures where practical; disclose
controlled release and Factor seams. No live providers or actual book writes.
Run relevant preparation/Calendar/bootstrap/automatic/native import regressions,
Black, scoped flake8 and mypy. Stop this change at those criteria; full installed
end-to-end acceptance belongs to the final current-source validation.

## Exact contracts and accepted Architect amendments

The following exact definitions supersede earlier shorthand. No extra fields.

Configuration `cn-daily-preparation-config.v1`:

- schema_version, market="CN", strategy_id="aggressive_tech_manufacturing",
  graph_sha256=GRAPH_SHA256;
- required exact refs: release_ref, release_install_ref, factor_loop_context_ref,
  research_policy_ref, store_policy_ref, timing_policy_ref,
  theme_acquisition_ref, corporate_action_template_ref, industry_source_ref,
  risk_free_ref;
- nullable refs: exposure_rows_ref, seed_completion_ref (existing fixed EOD path);
- prediction_deadline_local_time: null or strict HH:MM:SS, Shanghai;
- publish_current_dashboard: exact bool.

Commitment `cn-daily-preparation-commitment.v1`:

- schema_version, trade_date (original Calendar observed date), config_ref,
  calendar_ref, raw_calendar_ref, prepared_at (canonical UTC seconds);
- locator: exact mode (HEAD/SEED/NONE), completion_ref (nullable), observed_head
  (null or exact document/json_sha256/mirror_sha256), configured_seed_ref
  (nullable), selected_seed_ref (nullable). Replay HEAD via existing head_bytes
  and head_js; configured seed must equal config, selected seed null under HEAD;
- construction: null for automatic empty collection; otherwise exact
  store_preimages (three existing pointer refs), benchmark_ref (fixed Dashboard
  compatibility source), previous_trade_date and factor_parent_pointer_sha256.
  Last two fields are nonnull only for bootstrap, null for automatic recipe;
- objects: ordered array of exact {ref, document}, using only the fixed paths
  below; reconstruct all documents from config/locator/construction/Calendar and
  compare canonical bytes and SHA, never trust an arbitrary objects array;
- store_policy_admission="DEFERRED_TO_NATIVE_STORE";
- authority=existing all-false FALSE_AUTHORITY.

Fixed root is results/operations/daily_production/CN/<observed-date>/
preparation/<config-sha>. Commitment is commitment.v1.json; generated request is
request.json. Other leaves are under objects/: prediction-policy.json (only with
explicit deadline and nonempty recipe), bootstrap.json (only bootstrap),
recipe.json (bootstrap or nonempty automatic), collection.json (only automatic).
Order is prediction policy if any, bootstrap if any, recipe if any, collection
if any, request last. No optional unknown object, mutable latest pointer or scan.

Result `cn-daily-preparation-result.v1` has exactly schema_version, status,
trade_date, config_ref, calendar_ref, raw_calendar_ref, commitment_ref,
request_ref, store_policy_admission, authority. Status is
REGISTERED_INPUTS_ONLY (both output refs nonnull) or NON_TRADING_DAY (both null).
Expected errors use the existing CLI BLOCKED/exit2 boundary and bounded typed
missing_event_dates, missing_benchmark_dates or pending_request_ref fields;
unexpected errors exit3. No completion/admission claim.

Fresh locator checks require Calendar coverage/open membership and locator<=target.
HEAD wins over configured seed, but both original observations are bound in
commitment. With no HEAD/SEED only bootstrap is possible. Same-day locator generates
only empty collection and request; no native pointers, Factor or date policy reads.
Config itself and release installation remain checked. Nonempty branches run the
package static readers and exact source binding; original source times unchanged.

Use one existing automatic lock, no pending lease writes. All generated object
conflicts are checked before any repair. Existing commitment replay occurs before
current date, source, head or Factor checks. Its original config/Calendar refs must
match exactly, including paths. Rebuild the exact allowed objects, compare the
committed documents and byte hashes, then write only missing objects with request
last. Recheck commitment, originals and generated bytes after writes. Return only
the same committed request if ACTIVE pending matches; foreign ACTIVE always blocks.

Fresh preparation retains and rechecks the exact head pair, all source bytes and
native pointer identities before commitment and after request publication. Drift
fails, leaving any already-written commitment intact for exact recovery. No fresh
reselection in the same call. Tests explicitly cover head drift at both boundaries,
same-day commitment/request crash followed by next-day exact recovery with all
native heads/Factor reads forbidden, and malformed commitment/extra object paths.

## Implemented entrypoint and evidence

The installed package entrypoint is `python -I -m quant_investor.cli.daily_prepare`.
It requires --workspace-root, --config, --expected-config-sha256, --calendar,
--expected-calendar-sha256, --raw-calendar, --expected-raw-calendar-sha256 and
--release-repository-root. The config's exact release_install_ref is verified
against the running installed interpreter before preparation. The command makes
no provider call and does not invoke the launcher. Its request_ref is the input
to the existing launcher's --daily-production-request and expected SHA arguments.

Source reads reuse System's descriptor-relative directory checks and the existing
CLI request-file identity/read/budget primitives. They permit only 0400/0600/0644
regular single-link, owner-owned files, capped at the existing 8 MiB request-file
limit per source. FIFO opens are nonblocking; aliases, executable/group-writable
files, changed inode/mtime/ctime or bytes reject. Existing policy and compatibility
file modes are preserved. No shared storage permission contract was widened.

Validation: 196 related tests passed, including 51 preparation/CLI cases and
existing Calendar/bootstrap/request/policy/automatic/launcher/full closed import
coverage. Native Calendar wire replay, Store v3, Event and benchmark fixture
readers are exercised; static installed-loop/Factor evidence is explicitly
controlled. CLI isolated package import passed without script loading. This is
not a freshly installed EOD, source provisioning, scheduler or unattended proof.
Exact final checks and source hashes are retained in
`.agent/acceptance/phase12-daily-preparation-validation.json`.
