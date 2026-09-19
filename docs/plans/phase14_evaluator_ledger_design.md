# Phase14 evaluator admission from the daily ledger

Status: IMPLEMENTED_LOCAL_VALIDATION. Initial plan, native route amendment and
the exact bridge-closure repair passed Architect then Critic.188 focused checks
pass. Receipt `.agent/acceptance/phase14-evaluator-validation.json` records exact
scope and limitations. Original Phase0–15 goal remains active.

## Concrete problem and existing route

Both cli/main.py factor production-settle and market/daily_factor_loop.py
DailyFactorLoop._settle call factors.production_outcomes.settle_production_observations.
That path publishes immutable raw-close outcomes and a diagnostics.v1 summary.
classify_seal.prospective_eligible describes signal seal timing only; late observation
registration can still return true. The summary's original_close_prospective_count
therefore cannot certify full-DAG OOS. Keep old immutable classifications unchanged.

Phase14 must be active by default in this existing path. No optional parallel
evaluator, new CLI flags, caller-provided prospective booleans or new source-input
provisioning mechanism will be introduced. The prior draft's optional evidence
request was superseded after verifying the fixed owning completion path.

## Exact selection and native authority

1. For each registered observation's exact signal_date, use DailyJournal and its
   existing JournalStorage reader to read exactly
   results/operations/daily_production/CN/YYYYMMDD/completion.v1.json. This is a
   fixed per-day immutable destination, not a current-head/latest/mtime scan.
   Pin the observed SHA on this first read. No path substitution or date fallback.
   If absent, return UNCONFIRMED with DAILY_COMPLETION_MISSING; no writer/repair.
2. Add one owning read-only availability inspection helper to
   scripts/daily_ledger_replay.py. It performs the existing full native
   replay_native_completion and validates exact date/ref/all16 nodes before
   exposing the existing ledger classification. Refactor
   select_eligible_daily_evidence to use this helper and retain its current
   exact positive return shape and DAILY_EVIDENCE_NOT_PROSPECTIVE rejection.
   Morning remains unchanged. No recursion from completion into outcomes.
3. The helper returns exact completion_ref, ledger_ref or null, classification,
   prospective, synthetic and recomputed plus validation scope. Extend the owning
   replay_completed_ledger result with its already reconstructed recomputed value.
   A completed legacy EOD with
   no ledger is UNKNOWN_LEGACY/ineligible. Invalid native replay, invalid ledger
   shape/flags/classification, mismatched refs or incomplete nodes raises, never
   turns into a positive or a diagnosed valid retrospective day. All declared
   negative classifications require successful full native replay.
4. A factor-side admission helper lazily imports that owning script reader.
   After successful replay it reads the exact ledger by SHA using native storage
   and binds: trade_date; core_timing.effective_trade_date; factor_pointer_ref SHA;
   core_timing.observation_registration[LOW/W80] exact observation ref and
   registered_at; the matching node_custody output ref. The observation has already
   passed native validate_factor_production_observation and _matches_inputs in
   production_outcomes, binding its factor ID/generation/Market/PIT/Calendar.
   This prevents a good daily receipt from admitting another signal or a copied
   observation. Do this binding for valid ineligible ledgers as well.
5. Require ledger prospective is True, synthetic is False, recomputed is False
   and classification CONTEMPORANEOUS for ELIGIBLE. Late/backfilled, historical,
   recomputed, synthetic, missing, legacy and invalid evidence cannot enter OOS.
   Do not add an artificial original-close deadline: the native registered
   prediction policy and custody comparisons own the valid prediction deadline.
   Registration/Decision/Store/policy custody after that deadline is excluded by
   native derivation.
6. Errors are observed at the admission boundary only and preserve raw diagnostic
   work. Use fixed states/reasons: ELIGIBLE (empty reasons), UNCONFIRMED
   (DAILY_COMPLETION_MISSING), INELIGIBLE
   (SYNTHETIC_EVIDENCE, LATE_REGISTERED, RETROSPECTIVE_RECOMPUTE,
   UNKNOWN_LEGACY), INVALID (DAILY_NATIVE_EVIDENCE_INVALID,
   DAILY_OBSERVATION_BINDING_MISMATCH or DAILY_EVIDENCE_CHANGED).
   No exception text, paths/tracebacks or invented cause are exposed as reasons.
   Unexpected native errors become INVALID, OOS count zero; raw diagnostics remain.
7. Perform daily replay after raw horizon evaluation, immediately before admission
   projection publication. At most one replay per exact completion-ref in this
   invocation, shared by LOW/W80. Recheck exact completion, ledger and observation
   bytes before publishing each admission and before publishing the summary.
   A changed source invalidates that admission and removes its rows from aggregate
   eligibility. Missing completion may be reconsidered on a later invocation;
   an exclusion is not a permanent cached judgment. No cross-call cache.

## Artifact and output contract

Add factors/production_outcome_admission.py to own the versioned projection,
fixed reason/state grammar, source selection/binding, immutable publication and
factor/horizon grouping. Reuse production_outcome_sources.persist_source and
FactorProductionStore native reads; no new writer/root/lock or activation lane.

Each observation gets a content-addressed JSON document in existing
results/factors/outcome_sources/<sha>.json:
- schema_version: factor-production-daily-admission.v1
- authority: NON_AUTHORIZING; factor_admission: false
- eligibility_scope: LOCAL_COORDINATOR_AVAILABILITY
- observation_ref, factor_id, signal_date, factor_pointer_sha256,
  factor_generation_sha256 (all from verified original observation)
- completion_ref (nullable), ledger_ref (nullable)
- state, reason_codes (fixed rules above), prospective (boolean)
- classification (native value or null), synthetic (boolean or null),
  recomputed (boolean or null)
- implementation: SHA map of owning admission source and the native ledger/replay
  entrypoint sources, preventing unchanged prices from pinning stale admission
- no invented observed-at/past-availability timestamp; this is a diagnostic
  projection requiring native replay on later consumption, not a new authority.
- native_replay_required: true (also explicit in the nested OOS summary).

The default summary becomes factor-production-diagnostics.v2. Preserve existing
raw horizon states, outcome refs, market/cursor fields, limitations and authority.
Rename original_close_prospective_count to component_sealed_by_close_count and add
component_classification_scope=SIGNAL_SEAL_TIMING_ONLY. classify_seal and old source
records keep their exact prior semantics/bytes; they are never used for new OOS.

Add oos_evidence with schema_version factor-production-oos-summary.v1, false factor
admission, same local scope, per-observation admission refs/state/reasons,
eligible_observation_count, eligible_signal_day_count and exclusions by reason.
Include by_factor -> horizon1/5/20/60 aggregates: eligible EVALUATED outcome refs,
unique origin count, metric sample counts and means for ic/rank_ic/raw-close
long_short_spread. Metrics come from native-read validated immutable outcome
artifacts, not arbitrary summary rows. Unavailable metrics stay unavailable; no
zero imputation. Do not mix LOW/W80, horizons, duplicate observations or unaudited
outcomes in means. WAITING/DATA_PENDING/INVALID raw outcomes are not OOS outcomes.
Zero admitted outcomes yields explicit unavailable metrics and zero sample counts.
Raw-price aggregates still do not prove economic/executable return or effectiveness.
effectiveness_state remains FACTOR_EFFECTIVENESS_INSUFFICIENT_EVIDENCE.

Do not add admission to the raw outcome identity/body or rewrite old outcomes.
The existing _implementation whole-file fingerprint naturally starts a new raw
implementation series when production_outcomes changes; this is expected and is
not bypassed with a pinned old SHA. Existing revisions remain immutable.
An exact same-source repeat deduplicates raw outcomes and admission documents.
A later valid ledger can create a new admission projection without changing raw
price evidence. Every new summary rebuilds admission using the owning native reader;
a stored positive projection alone is never a trust shortcut.

CLI flags and call signatures stay unchanged. In-repo consumer inventory found
only DailyFactorLoop/report, which reads horizon states/errors and retains the
summary; verify its tests for the v2 summary. Public v2 summary change is explicit
and documented; old saved v1 summaries remain readable as component diagnostics.

## Acceptance and stop conditions

- Focused existing outcomes/Factor-loop/CLI tests retain numerical metrics,
  maturity/denominator behavior, no financial authority and repeat idempotence.
- Default public settle entry and daily-loop call exercise admission with no flags.
  No ledger gives zero OOS while raw diagnostic settlement still succeeds.
- Pre-close signal seal + late observation registration cannot enter new OOS,
  while original immutable classify_seal record remains unchanged.
- Real native timing derivation excludes late observation, Decision, Store and
  policy custody; synthetic/history/recompute/legacy/missing/wrong-SHA/wrong-day/
  wrong-observation proofs are excluded or invalid. Never trust raw true flags.
- Positive native integration uses existing synthetic full-DAG fixture with
  explicitly controlled provenance/clock seams needed to exercise the prospective
  branch. Report that seam; it is not a real-time historical OOS claim. Native
  source/ledger/observation bindings and full replay execute normally wherever
  possible; no file flag is edited to manufacture a real admission.
- Read the existing installed retrospective EOD with the actual evaluator admission
  route and prove it is excluded. Do not restart its producer.
- Verify eligible-only grouped metrics, unknown/malformed values, duplicate/other
  factor/date refs, same-price admission changes, source drift before publication,
  protected old observation/outcome bytes/mtimes and provider/network/write guards.
- Run relevant native selector/Morning/ledger regressions after shared helper edits,
  scoped static checks, and a final-source receipt. Freeze implementation before
  long native tests; do not edit hashed sources during those tests.
- No holdings, paper book, Factor policy, Macro veto, active pointers, scheduler,
  deployment or live provider changes. Phase12/15 and historical fact gaps remain.

## Accepted Architect amendments and exact contract

Architect APPROVE_WITH_CHANGES; all10 amendments accepted. The following details
take precedence over abbreviated descriptions above.

State precedence is deterministic, one reason per excluded observation:
1. Missing fixed completion: UNCONFIRMED / DAILY_COMPLETION_MISSING.
2. Invalid completion/claimed-ledger/ref/schema/native replay: INVALID /
   DAILY_NATIVE_EVIDENCE_INVALID. A v2 completion with missing ledger is invalid.
3. Valid fully replayed legacy completion without ledger: INELIGIBLE / UNKNOWN_LEGACY.
4. Valid ledger with wrong observation/date/pointer/custody/output binding: INVALID /
   DAILY_OBSERVATION_BINDING_MISMATCH (takes precedence over valid-negative labels).
5. Valid synthetic ledger: INELIGIBLE / SYNTHETIC_EVIDENCE.
6. Valid nonsynthetic recomputed/retrospective ledger: INELIGIBLE /
   RETROSPECTIVE_RECOMPUTE.
7. Valid late ledger: INELIGIBLE / LATE_REGISTERED; remaining valid unknown legacy
   classification: INELIGIBLE / UNKNOWN_LEGACY.
8. Exact contemporary/prospective/nonsynthetic/non-recomputed ledger: ELIGIBLE / [].
Any byte drift after the invocation's first pin overrides its affected observation
or date with INVALID / DAILY_EVIDENCE_CHANGED. DAILY_LEDGER_UNAVAILABLE is removed.

Observation binding also compares the Core timing custody terminal_ref against the
full node_custody row for the alias, and both against the completion terminal ref.
The full node's LOW/W80 output ref must equal the exact observation ref. Native
observation validator supplies the original factor/generation/date/registration
fields; no caller-declared factor ID substitutes for them.

The exact OOS-summary fields are schema_version, authority, factor_admission,
eligibility_scope, native_replay_required, batch_scope, admissions,
eligible_observation_count, eligible_signal_day_count, exclusions_by_reason,
by_factor. batch_scope=CURRENT_CURSOR_BATCH. Admission rows have exactly
observation_ref, admission_ref, factor_id, signal_date, state, reason_codes.
exclusions_by_reason is a sorted fixed-reason -> integer count; each excluded
observation contributes once. Eligible signal days are distinct dates.
by_factor[factor_id][horizon] has exactly eligible_outcome_refs,
eligible_origin_count and metrics. metrics has exactly ic, rank_ic,
long_short_spread; each metric has state, sample_count, mean. Unavailable means
UNAVAILABLE/0/null. Factor IDs, numeric horizons1/5/20/60, observation rows and refs
are sorted deterministically. Include observed factors with zero eligible groups
so missing evidence remains visible. Summary scope is this bounded cursor batch,
not lifetime performance or a combined cross-factor portfolio.

Every aggregate reads an immutable outcome through _read_outcome and binds its
observation_ref, origin_session, horizon, factor identity and exact evaluation_ref
to one eligible admission and one current EVALUATED settlement row. A ref can
appear only once; duplicate/conflicting rows cannot silently add samples. Metrics
come only from artifact diagnostics. Parse native AVAILABLE finite values, one
sample per unique outcome, unweighted arithmetic means, binary64 .17g output.
Reject malformed/non-finite AVAILABLE metrics; never count them as zero or reuse
an unavailable spread. Raw diagnostic outcomes and all prior summary fields other
than the explicit component-count rename are retained.

One invocation remembers its first fixed-path observation for every date, including
absence, and caches replay by exact(path,SHA). LOW/W80 share that replay. A later
different SHA or absent->present transition invalidates all admissions for that
date; no new bytes are replayed in the same invocation. A later invocation may
reconsider. Recheck completion/ledger/observation bytes after replay, before each
admission publication, before summary construction, and after summary persistence.
Publication uses a bounded correction: if the post-summary check finds new drift,
publish deterministic invalid admission replacements and a corrected summary;
recheck once more. A further newly affected source aborts the summary return and
cursor update rather than looping. Old positive objects remain immutable and have
native_replay_required=true. Already invalid sources cannot regain eligibility in
the same invocation.

Add owning readback functions for admission projections and v2 OOS summaries.
They validate exact content refs/schema/implementation maps, rerun current native
admission for their exact original observations and reconstruct aggregate binding
from immutable outcome artifacts. Reject a stale projection/summary instead of
trusting its stored boolean. These readers are code-owned read-only functions,
not a new command, writer or alternate evaluator. Saved v1 is component-only and
cannot be admitted through this OOS reader.

The admission implementation map has exactly the named hashes of
factors/production_outcome_admission.py, scripts/daily_ledger_replay.py,
scripts/daily_completion_replay.py, operations/ledger_readback.py and
factors/production_observation.py, plus combined sha256 over the canonical named
hash map. Verify exact keys, individual lowercase64 hashes and combined SHA.
Source drift during the invocation cannot retain a positive admission. Raw
outcome implementation identity retains its existing independent contract.

The positive native branch test is a controlled prospective integration fixture,
not a genuine historical real-time observation. An existing installed retrospective
receipt may also fail current-source replay because earlier stored adapter hashes
differ; report the exact observed exclusion and do not mislabel a replay failure
as a successfully validated retrospective ledger. Final current-source installed
proof remains Phase15; do not restart old producers or modify immutable proof roots.

## Native route amendment after standalone failure (authoritative)

Architect reviewed the concrete failure and APPROVE_WITH_CHANGES; all8 route
amendments accepted. This section supersedes direct script imports and the earlier
five-file fingerprint. The wheel intentionally contains quant_investor only.
Native scripts must stay behind native_bridge.verified_native_context. Existing
combined tests happened to preload script paths; standalone tests exposed the
invalid assumption. Do not add scripts to the wheel, sys.path manipulation, or
dynamic import fallback. Keep these tests isolated during final validation.

Add package-only operations/outcome_native_replay.py. Given an exact fixed-day
completion ref, call inspect_recorded_completion. A missing completion is handled
before this helper. A recorded v2 completion must supply its minted
completed_handoff_snapshot. Use verify_archived_handoff_context(snapshot), which
already replays exact release/install/slot identity. Derive only from the verified
snapshot: install ref/raw, canonical release_repository_root, python_executable,
final_commit, final_tree and installed_code_manifest_sha256. Recheck snapshot after
native execution and during subsequent admission rechecks; do not independently
scan or guess a release location.

Add a private ContextVar to native_bridge, set only after verified preloading and
operation construction, reset before leaving the context. It contains exact
release-input SHA, canonical repository root, final commit, thread identity and
the existing operations mapping. Expose only invoke_active_completion_replay with
fixed completion arguments and expected release identity; no general operation
accessor and no mapping returned. A matching active context invokes its existing
completion_replay callable. No context/mismatched identity selects the recorded
child route; errors from a matched active invocation never trigger a retry via the
child. Context teardown/exceptions/other threads cannot retain this capability.

Otherwise invoke the verified recorded Python with -I -B and one fixed code-owned
-c runner. No shell and no input-derived code/operation. The exact canonical input
has only workspace, trade_date, completion_ref, release_install_ref and
repository_root. The child securely rereads install bytes/SHA, enters its own
verified_native_context and calls only completion_replay. It emits a minimal
canonical envelope containing exact ref/date/all16 nodes/native_replay_validated,
synthetic and the native ledger result; all native incidental stdout goes to
bounded stderr. No current-release retry, latest install lookup or unsupported
old-operation fallback is allowed.

Use an isolated temporary working directory and a minimal environment containing
PATH to system tools, LANG/LC_ALL, PYTHONHASHSEED, PYTHONPATH empty,
PYTHONDONTWRITEBYTECODE and TMPDIR. Do not inherit provider, model, broker, proxy,
or user credentials. Input is bounded to64KiB, stdout1MiB, stderr256KiB, and one
native call has a300-second timeout. Temporary file-backed I/O and bounded process
polling enforce output limits without buffering unbounded output. On timeout or
overflow terminate/reap only the owned process group. The child denies network
access and canonical-source writes; bridge/probe temporary runtime files remain
allowed. Source/financial paths are never a child output destination. Tests must
verify the actual spawned interpreter/arguments/environment and rejection paths.

The parent validates the native envelope itself (not child booleans alone): exact
ref/date/node set, true native replay, boolean synthetic, matching bound ledger
ref/classification/prospective/synthetic, and recomputed. Old completed replay
results may omit recomputed; only after full native replay under the exact archived
release may the value be taken from the SHA-bound ledger in the already validated
snapshot, and it must satisfy native classification consistency. This is not a raw
ledger fallback. The child's output shape and optional old ledger field profile
are exact and code-owned. Any other mismatch is INVALID / DAILY_NATIVE_EVIDENCE_INVALID.

An intrinsically valid recorded EODv1 has no minted snapshot/release binding.
Exclude it as INELIGIBLE / UNKNOWN_LEGACY with evidence_validation_scope=
RECORDED_EOD_WITHOUT_NATIVE_RELEASE. Do not claim full-native valid legacy proof.
Missing completion has evidence_validation_scope=NO_DAILY_COMPLETION. Valid v2
uses FULL_NATIVE_EOD_AVAILABILITY. Invalid cases use UNCONFIRMED_NATIVE_EVIDENCE.
Add evidence_validation_scope to the exact admission fields so the distinction
is visible. Native replay unavailability is never an admission path.

Replace implementation with exactly evaluator, runner_protocol_sha256,
native_replay and sha256. evaluator has named hashes for
factors/production_outcome_admission.py, factors/production_observation.py,
operations/outcome_native_replay.py, operations/native_bridge.py,
operations/completed_handoff_snapshot.py and operations/archived_handoff_context.py.
runner_protocol_sha256 hashes the literal fixed runner/protocol source.
native_replay is null before verified release binding (including missing, v1 and
early-invalid evidence), otherwise exactly release_install_ref, final_commit,
final_tree, installed_code_manifest_sha256, repository_root and
operation=completion_replay. Combined sha256 covers the other three exact fields.
Active and child routes have identical identity. No PID/time/temp-path/route label.
Different historic releases can coexist in one batch; each observation projection
uses its own verified native identity. Evaluator/package and snapshot source drift
invalidates affected admissions; missing scripts never prevent missing-completion
diagnostics. Saved readers validate the exact structure and recompute the full
identity by repeating native selection instead of demanding current script files.

The existing scripts/daily_ledger_replay helper remains only for bridge-owned
consumers. Factor code contains no scripts import. The default command and loop
signatures remain unchanged. Cache at most one call per pinned date/completion/
release identity per invocation. Acceptance adds standalone missing/v2, matching
active context, mismatch->recorded child, no fallback after active failure,
mixed release identity, v1 exclusion, changed interpreter/install/commit, malformed
or oversized child output, timeout/process cleanup, no inherited credentials,
protected source bytes and readback behavior. Architect then Critic approval of
this amendment was completed before these route changes.

The actual archived child reached full replay's ledger materialization readback,
then its own old bridge rejected scripts.daily_store_adoption. The bounded error
trace and negative result are retained; all14,862 source-workspace files stayed
unchanged. The current bridge's exact5-key closure is repaired and its full real
script preload passes with only a disclosed installed-runtime verifier seam.
Do not rerun or modify the old archived release to manufacture acceptance. Valid
prospective branch tests use native observations/timing with a controlled native
replay boundary; they are not real-time OOS or current installed full-DAG proof.
