# Phase 0–15 execution register

User objective: 根据《投资决策状态解释》的设计，逐步完成 Phase 0–15 全路径。
Source design is retained verbatim in `daily_evidence_dag_chat_design.md`.
This register supersedes treating the earlier short English objective as the
complete acceptance checklist. No whole-goal completion is claimed.

Existing implementation and tests are retained. New work proceeds in phase order;
ongoing native acceptance continues to validate the shared underlying pipeline.
Existing canonical contracts remain authoritative until a reviewed versioned
change is implemented; examples in the chat are not silently substituted for
current native research or financial authority.

## Current acceptance checkpoint (2026-09-17, Asia/Shanghai)

This checkpoint supersedes the historical status wording below. Evidence applies
only to frozen runtime `5060dee2d4b39b50a735393d056ebe5adb4efe3b` unless
explicitly marked historical. The full Phase0–15 objective is **incomplete**.

- All12 frozen CI gates passed, with5185 unit tests passed,3 skipped and87
  warnings; source drift was empty. Receipt:
  `.agent/acceptance/phase15-canonical-scan-final-ci-pass.json`.
- Current installed configured Aug27 first-day execution and independent
  full16-node native replay passed. Cases1/2/5/7 have current evidence: automatic
  Top100, no-trade Store and legitimate Fundamental warning behavior. Only1/5
  normal dates is verified complete. Receipt:
  `.agent/acceptance/phase15-5060-first-day-acceptance.json`.
- Case8 passed a fresh full positive replay and exact Market/PIT SHA rejection
  using physical fault files and an explicitly scoped native-read routing test
  seam. Selected298 original files and capture-root identity were preserved;
  this is not a whole-workspace physical-copy claim. Receipt:
  `.agent/acceptance/phase15-5060-case8-acceptance.json`.
- Existing successor driver completed same-day repeat and owns remaining four dates,
  aggregate replay, Morning, separate Top100 crash/unresolved Corporate cases,
  and closed-day acceptance. Their current-version completion remains unproven.
  Current root: `.agent/acceptance/phase15-canonical-scan-location.json`.
- Case4 retained recovery and Case9 same-day repeat now passed on this same
  runtime. Case4 proves full16-node replay, exactly one financial CAS and the
  missing date closed once with original upstream/cutoff/pool preserved; its two
  earlier harness refusals remain disclosed. Case9 returns NO_ACTION/COMPLETE
  with the exact original completion and unchanged workspace. Receipts:
  `.agent/acceptance/phase15-5060-case4-acceptance.json` and
  `.agent/acceptance/phase15-5060-case9-acceptance.json`. Do not rerun either.
- Production Calendar capture/proof integration and the canonical-control-
  character scan repair are included in this frozen CI, but synthetic installed
  proofs do not establish real-source execution, production deployment,
  unattended operation or real observation. Formal owner account events,
  prediction deadline, production bindings and cutover remain unresolved.
- `.agent/acceptance/phase15-case-evidence-index.json` is the detailed current
  case index. Older full-day proofs remain under their original versions and
  cannot fill current-version acceptance gaps. Do not rerun passed gates without
  new source changes, failures or unresolved concerns that require them.

## Historical acceptance checkpoint (2026-09-14)

This checkpoint supersedes the open-work wording in the historical checkpoints
below. The original whole Phase0–15 objective remains incomplete.

- Phase7 C2 source/cutoff/Corporate/Dashboard/origin/committed-metadata and
  automatic cross-day recovery are now locally implemented and tested. Their
  native versus controlled outer admission boundaries are explicit in
  `phase7-c2-native7-validation.json`, `phase7-c2-dashboard-validation.json`,
  `phase7-c2-recovery-validation.json` and `phase7-c2-crossday-validation.json`
  under `.agent/acceptance/`. Do not restart Part C implementation from the older
  paragraphs below. Current installed full-path evidence remains outstanding.
- The later full frozen unit run finished with4823 passed,6 failed,3 skipped;
  all four static gates passed and frozen inputs stayed unchanged. The6 known
  failures were repaired;162 focused tests passed in current committed isolated
  checkout f7968454dcf537b20369a762bdbb16f56b2ee019. Current Dashboard28 tests and
  both legacy suites also pass after correcting two stale test cache-version
  assertions. Preserve the original failed full-run result and the scoped
  post-repair proof separately; do not call it an all-green rerun.
- Current native installed acceptance uses a verified same-commit installation
  and coherent synthetic business clocks. The original configured request is
  prepared before Core. Earlier test runs with release-path alias or incomplete
  clock coverage are preserved failures/interruption evidence, not completed
  days. Current location: `.agent/acceptance/phase15-configured-clock-location.json`.
- Four same-config automatic successors, exact five-day native readback and the
  following Morning are prepared, not yet passed. Installed original-request
  repeat and closed-day financial no-op checks are also prepared. Their driver
  refuses entry until the original configured process has actually passed.
- `.agent/acceptance/phase15-case-evidence-index.json` tracks each original Case1–10
  separately. Component tests and older installed snapshots do not prove the
  current full-DAG crash, backlog, corporate, lag or SHA scenarios by themselves.
  Remaining production migration/configuration/owner-fact boundaries are unchanged.

| Phase | Design requirement | Verified existing evidence | Remaining work / disposition |
|---|---|---|---|
| 0 | Real baseline, node inventory and exact break diagnosis | Original diagnostic baseline dated 2026-09-07, 17-node inventory, ORCHESTRATOR_MISSING and Store INPUT_MISSING; architecture document exists | Historical baseline artifact verified. Five mutable source refs have since changed; do not present baseline as current readiness. Audit: `.agent/acceptance/phase0-baseline-audit.json` |
| 1 | Nine states plus consistent trigger/blocking/retry/ref/time/attempt fields | Common metadata implemented; 83 focused checks and real 16-node journal readback passed | COMPLETED_LOCAL_VALIDATION: original times/refs/attempts preserved, unknowns null, running projection visible before writer; final whole-path acceptance remains separate |
| 2 | Code-owned dependency resolution and explicit incompatible-input blocking | Typed date/ref/schema/Calendar/missing-source diagnosis, transitive root causes, producer-context admission; 135 focused tests passed | COMPLETED_LOCAL_VALIDATION: 16 execution-control checks repeated after import cleanup; Phase 1 compatibility and targeted static checks pass. Receipt: `.agent/acceptance/phase2-validation.json` |
| 3 | Automatic/idempotent Factor→LOW/W80→Top100 and requested tabular output/manifest | Installed native Aug27 full EOD passed; real pool publication and crash-adoption tests | COMPLETED_LOCAL_VALIDATION: native five-leaf Parquet publication, exact manifest, public CLI/core, binary/legacy replay, zero-write repeat, Morning and late-time classification pass. 121 focused checks and static checks passed; `.agent/acceptance/phase3-validation.json`. Pre-existing weekly maintenance fixture CLOSE_RECEIPT_REPLAY_MISMATCH remains separately disclosed. |
| 4 | DC primary/registered TDX fallback, distinct membership/exposure/confidence/source evidence, PCB/AI hardware coverage of 002463.SZ and 002384.SZ | Native Theme/Industry/Exposure adapters and source gates exist | COMPLETED_LOCAL_VALIDATION: v2 acquisition/PIT binding/handoff, two-company membership/exposure reports, named refs, input revisions and completed-source/ledger readers integrated. 142 focused regressions plus two separately passed slow native 100-company cases. `.agent/acceptance/phase4-validation.json`; final installed/current-source whole-path acceptance remains. |
| 5 | PIT-valid low-frequency Fundamental/Macro with FRESH/ACCEPTABLE_LAG/STALE_WARNING/MISSING and only critical missing evidence blocking | Native source admission and source-derived timestamps exist | COMPLETED_LOCAL_VALIDATION: PIT/cohort and native availability checks, four-state reports, exact frozen Macro decoder, named captures/status/revisions and versioned legacy/new replay are integrated. 151 tests passed, 2 legacy-only inapplicable cases skipped; a further 72 overlapping precision/ledger checks passed after fixing fractional source-time readback. 14-source mypy and 21-file formatting passed. Native retained Macro closure yields ACCEPTABLE_LAG across 112 registered subjects, zero critical missing and zero current observation head reads. `.agent/acceptance/phase5-validation.json`; final installed full-path acceptance remains |
| 6 | Standardized research Decision bound to all evidence and portfolio state, versioned output and confidence/blockers | Native compiler emits existing canonical five research states with evidence refs | COMPLETED_LOCAL_VALIDATION: native portfolio custody, v3 recipe/input dispatch, eight-domain Decision v2 report, immutable side capture, completed replay and retrospective ledger exclusion integrated. 175 focused tests passed; separately 5 current report/capture tests passed after ref guards/cache. 12-source mypy and 26-file formatting passed. Native Store/materialization and complete legacy Decision replay passed; outer report admission/collector seams and final installed v3 whole-DAG proof remain explicitly separate. `.agent/acceptance/phase6-validation.json` |
| 7 | Exact Store close-forward, including no-trade daily state | Native Store/backlog/CAS tests; frozen3c46 all five days plus cash/shares/cost/valuation/authority audit passed | Current-source native five-session/backlog/CAS recovery and event integrity/frozen provenance are verified. Native already-closed batch adoption preserves original plan/output refs and pre-close portfolio, with strict custody and no second CAS:58 focused tests pass;3 older mocked fixture failures reproduced unchanged on frozen3c46. `.agent/acceptance/phase7-existing-close-validation.json`. Nonempty financial-event and other manual-record adoption plus final whole-path acceptance remain |
| 8 | Corporate-action reconciliation and non-executable unresolved thresholds | Native full-window Decimal checks, five named source kinds, native account attribution, separate owner declaration and frozen replay | COMPLETED_LOCAL_VALIDATION: execute/materialization v3, native v4 and corporate v2 report integrated with completed replay and late-ledger exclusion;174 focused tests pass. Native synthetic dividend/split and same-day conflict blocking pass. Actual retained Zijin Aug21 change is detected retrospectively and remains unresolved without real source/account/owner evidence. `.agent/acceptance/phase8-validation.json`; installed full-DAG/source provisioning and scheduled adoption remain separate |
| 9 | Dashboard consumes DAG authorities, matching completed date and rejecting stale output | Native Store/Dashboard replay, five-source collector, guarded publication, expiry and HTTP UI | CORE_LOCAL_VALIDATION_PASSED: six review corrections fixed;160 Python checks,12 Node contract checks, legacy browser contracts and static checks pass. Current v5 Store/Dashboard native rebuild passes with explicit outer EOD admission seam, read-only inventory and serving-head independence. Actual HTTP expiry view verified. File-browser QA remains platform-policy blocked; current-source installed full-DAG proof remains final acceptance. `.agent/acceptance/phase9-validation.json` |
| 10 | Morning consumes prior completed EOD, same-day quotes and owner thresholds, without maintenance | Native frozen risk sources, quote parser, v3 consumer/report and receipt/history, plus preserved v2 tests | COMPLETED_LOCAL_VALIDATION: explicit v3 policy refs, unchanged native formulas, independent stop/trailing proof, quote warnings, source-bound deterministic report and immutable receipts/history implemented.192 focused tests and static checks pass. Native v4/v5 Store replay is independent of current heads. Component tests explicitly control full EOD/Calendar/Decision and live-provenance admission; actual installed/full-DAG/live Morning remains final acceptance. `.agent/acceptance/phase10-validation.json` |
| 11 | Ordered automatic gap detection/catch-up and safe resume | Native Calendar routing/maintenance guard, binding-v2 handoff, automatic selection/recovery and independent historical financial Dashboard proof | COMPLETED_LOCAL_VALIDATION: Part A and reviewed Part B are integrated into the same public daily-close path. Final union382 tests PASS33.54s; static checks pass. Automatic head-or-seed selection, contiguous prefix validation, exact-ref frozen resolution, nonblocking lease, guarded expired-unstarted closure, public0/2/3 results and unchanged completed repeats are covered. Historical native Dashboard handles a future benchmark tail without serving writes. Original frozen failure remains unchanged. Full EOD admission/producers in automatic tests are explicitly controlled; current-source installed/full-DAG/unattended acceptance remains Phase15. `.agent/acceptance/phase11-validation.json` |
| 12 | EOD producer, night fallback and Morning automation use the unified path | Configured source acquisition/preparation now routes into the same automatic/bootstrap launcher | CONFIGURED_LAUNCHER_LOCAL_VALIDATION:290 related tests and final52 source/CLI/shell/concurrency tests pass (overlap). Two-stage locator/history CAS, Calendar budget, native Event/benchmark recovery, exact request reuse, credential isolation and no legacy fallthrough. `.agent/acceptance/phase12-configured-launcher-validation.json`. Actual installed static config, genuine bootstrap/automatic seed transition, current-source release installation, scheduler cutover/readback and real unattended receipt remain |
| 13 | Fixed error taxonomy with retry/owner/next-node data | Required taxonomy plus reviewed54 research reason mappings and safe native file diagnosis | RESEARCH_BOUNDARIES_LOCALLY_VALIDATED:101 main regressions,75 native/source/CLI cases and final41 source/diagnostic checks pass (overlap; do not sum). Original writer custody and CLI JSON preserved. `.agent/acceptance/phase13-research-diagnostics-validation.json`. Broader producer coverage remains separate; unknown security/race/projector errors stay generic |
| 14 | Source-backed prospective ledger, exclusion of late/recomputed/synthetic data from true OOS | Default production-settle/Factor-loop admission, immutable OOS projection/readback, source-bound factor/horizon metrics and recorded-release replay implemented | IMPLEMENTED_LOCAL_VALIDATION:188 focused checks pass; full real bridge module preload verified with disclosed running-install seam. Actual old installed readback rejects its missing module closure, leaves14,862 workspace files unchanged and contributes zero OOS. Current5-key map repaired; no old-release substitution. `.agent/acceptance/phase14-evaluator-validation.json`. Final installed/full-DAG/prospective acceptance remainsPhase15 |
| 15 | Ten full-path/failure cases and consecutive-day acceptance | Frozen3c46 all five dates, full read-only replay and financial invariants passed; native Morning replay passed | IN PROGRESS. Frozen five-day proof does not cover later Phase1–5 edits. Installed historical Dashboard recovery, remaining phase gaps and final current-source full-unit/CI gates remain |

## 2026-09-13 current-source validation checkpoint

The source-frozen full unit suite completed with **4532 passed, 7 failed and
3 skipped** in2954.97s. All914 source/build/test file hashes remained unchanged
through the run. The7 failures were then repaired without weakening production
validation:5 obsolete Store fixtures now use the native fixture, the weekly
consumer uses a complete native synthetic Calendar pair, and the isolated offline
release test explicitly uses the installed test environment's dependency cache.
Focused verification passed15 Store/weekly cases plus1 real isolated build/install/
origin/operator case in236.67s. The original failed full-run receipt is preserved;
the entire suite has not been repeated after these focused changes.
`.agent/acceptance/phase15-current-unit-repairs.json` records that distinction.

The actual mixed ordinary/owner-revalidated legacy initial-stop policy now parses
through the current reader with its original bytes/mtime. Its fixed stop remains
independent of a missing trailing anchor; old quote/retired trailing metadata are
never thresholds.97 current-workspace Morning/policy tests and scoped static
checks pass. `.agent/acceptance/phase10-existing-owner-stop-validation.json`.

Five reusable static source/policy objects and704 source readbacks are prepared
in `.agent/acceptance/phase12-static-source-inputs-20260913/`; this is not an
executable final config or a production install. Actual release/loop refs,
historical owner facts, prediction-deadline choice and corporate evidence remain
unresolved. Phase7 reachability and intraday-to-official-close requirements are
recorded in `phase07_registered_transition_gap_audit.md`; Phase13 source/preparation
error metadata is still a design draft. Full goal/installed five-day/unattended
acceptance remains incomplete.

## Phase7 registered-source continuation

Part A same-day performance finalization is locally implemented: exact native
intraday source, separate official close, unchanged cash/positions, preserved
units/cumulative flow and official evidence rather than correction. Part B adds
an exact owner-declared BUY declaration under the existing Event owner, original
writer-pointer custody, native replay, a manager command and bidirectional guards
against false empty publication.221combined tests pass34.20s. Receipts:
`.agent/acceptance/phase7-finalization-part-a-validation.json` and
`.agent/acceptance/phase7-registered-event-part-b-validation.json`.

These components are not yet wired into source/preparation/batch-v2/cutoff/
portfolio/Store/Dashboard replay. Part C and supported SELL/funding/corporate
profiles remain necessary; the full Phase7 and Phase0–15 goal remain incomplete.
No actual owner facts, financial state, provider, scheduler or deployment was
changed by these local tests.

## Phase7 C1 native integration checkpoint

Native batch v2 now consumes the exact registered BUY declaration, retains both
writer and Decision-baseline pointers, and uses the existing official valuation
writer/CAS with same-day performance finalization. Store adapter, completed
adoption, frozen commit readback and portfolio binding dispatch by exact plan
path/schema version.192combined tests passed27.27s; final Calendar-target follow-up
passed22focused tests11.67s. `.agent/acceptance/phase7-c1-validation.json`.

C2 source-plan/preparation/recipe/cutoff/corporate/Dashboard integration remains
open, as do ordered historical registered-source catch-up and broader SELL,
funding and corporate profiles. This native component proof does not establish
a configured full daily run, installation cutover or unattended operation.

## Phase7 C2 local source/report checkpoint

The configured source entry now recognizes native registered BUY evidence,
requires the exact previous-EOD Store baseline, and generates source-plan v2,
preparation v2 and recipe v6 without publishing an empty current-day Event.
Corporate v3 emits a separate registered transition report, preserving the T-1
Decision report and distinguishing accounting admission from required risk-policy
revalidation. Existing-position add, new position, original-request repair and
frozen report replay are covered by240 passing combined tests in62.13s.
`.agent/acceptance/phase7-c2-source-corporate-validation.json` records the native
financial/Store evidence and controlled all-node EOD/install/Calendar transport
test seams. No current-source full-DAG or installed proof is claimed.

C2 subsequently connected bundle v2, cutoff v2, wrapper v3, materialization v6 and
native input v7 to Corporate v3, native Store close and Morning risk sources.
Two native add/new-position scenarios passed;176 regressions,3 native legacy
cutoff scenarios and34 final custody/grammar/preflight checks passed in separate
runs. Exact sources and fixture boundaries:
`.agent/acceptance/phase7-c2-native7-validation.json`.

Dashboard/serving v2 is now implemented and locally validated:84 Python checks,
28 browser checks and HTTP desktop visual QA pass. The same registered report is
forwarded as a fifth output, its final Store link is natively verified, and the
view distinguishes owner changes from zero new close-writer trades. Exact receipt:
`.agent/acceptance/phase7-c2-dashboard-validation.json`.

C2 automatic origin custody is now integrated: capability/ACTIVE ownership,
immutable forward origin, handoff v4, cutoff source rechecks and completed/
archived snapshot transport. 183 related regressions, final30 origin/handoff
checks and3 native cutoff/financial scenarios pass (overlapping scopes; do not
sum). `.agent/acceptance/phase7-c2-origin-validation.json` records source hashes
and explicit outer EOD/installation/Core seams.

The committed-DAG recovery guard is now integrated into EXECUTE/RESUME and the
existing native metadata owner. Actual synthetic CAS-interruption recovery
creates only the two missing metadata files, preserves original financial/input
bytes and repeats without a financial CAS. Original configured preparation and
distinct Calendar roles replay independently. Before-write source/native/time
faults reject without metadata writes. Current evidence is recorded in
`.agent/acceptance/phase7-c2-recovery-validation.json`; full EOD/Core/installation
remain controlled seams.

Registered automatic cross-day routing, launch inspection v3 and typed expired
publication receipts are now implemented. 264 related checks pass, including
CLI protected dispatch, same-day/cross-day original recovery, source-head removal,
ACTIVE preservation on incomplete/unknown work, verified expired-EOD retirement
and exact unstarted preparation absence proof. The automatic tests control
commit/EOD admission and publication observation; the native package proof is
separate. `.agent/acceptance/phase7-c2-crossday-validation.json` records the scope.

C2 remains open: complete EOD/Morning admission, full original-custody finalized
recovery and the actual configured daily path still require completion. Five-domain
and full EOD admission in Dashboard tests remain controlled seams; no final
installed or unattended result is claimed. Native file-browser QA remains separate
from the tested Node file-route contract and HTTP browser view.
The original full goal, broader supported financial profiles and historical
ordering, final release/scheduler and unattended acceptance remain incomplete.

## Phase12 current integration audit

The pre-Phase13 Python/runtime snapshot has an isolated installed native proof:
commit `eaa60b2d7522db887d387ea99c47a7961c1257a0`,16 native nodes,
materialization5/native6 and EODv2 COMPLETE. Same-request replay returned NO_ACTION
with1439 protected files' bytes/mtimes unchanged and producers/network/journal
writes forbidden.468 runtime/build files matched at that checkpoint. Later Phase13
source edits have local validation but are not covered by this installed snapshot;
final installed acceptance must use the final source after remaining phases.
`.agent/acceptance/phase12-installed-v6-validation.json` records the evidence.
Data/transport/core-cutoff clocks are explicitly synthetic; Factor/Core and release
verification are native. This is one downstream-assembled day, not initial
pre-Core provisioning, production deployment, current-source five-day or unattended
acceptance. UI worktree assets are outside the Python runtime snapshot comparison.

The reviewed cutoff/input profile is implemented and under validation: recipe5,
Theme handoff3, wrapper2, materialization5/native6 and collection/binding3/auto2.
Combined focused checks158PASS2SKIP10.83s; native book/storage clock tests prove
receipt-first recovery, but source/corporate proof builders in those transaction
tests are controlled. Subsequent native five-domain/corporate, current/historical
native6 materialization/compiler/replay and Macro head-change tests passed3 cases
in185.79s. Factor/Core/installation remain explicit outer fixture boundaries;
full public/EOD/unattended acceptance remains. Receipts:
`.agent/acceptance/phase12-cutoff-progress.json` and
`.agent/acceptance/phase12-native-source-validation.json`. No live, financial or
scheduler changes. The current book's September4 filename is a recording timestamp;
its actual no-action valuation date isSeptember3. September4-11 changes require
the pending owner fact confirmation before missing financial dates can be closed.

The reviewed missing-only Exposure classifier is implemented as an explicit
new-profile selector:59 focused checks PASS19.05s, native Decision comparison
proves qualified PAPER_CANDIDATE becomes INSUFFICIENT_EVIDENCE after removing
revenue evidence, and report bytes stay unchanged. Receipt
`.agent/acceptance/phase12-exposure-completion-validation.json`. Existing public
legacy requests retain their former behavior. Wrapper2 now propagates the selector;
installed/public whole-path acceptance remains.

The existing Theme acquisition binder now accepts supported execution recipes
v3/v4 through the owning version-aware validator. Native capture/handoff and
provider-free replay are covered:59 focused tests PASS21.63s, scoped static checks
PASS. Receipt `.agent/acceptance/phase12-theme-profile-validation.json`.
This is an existing-profile repair; the new timing/input/bootstrap profile remains
unfinished in `phase12_input_preparation_design.md`.

Exact Sep11 maintenance evidence is PARTIAL with MACRO_WRITE_VETO_ACTIVE;
Fundamental HEALTH_ONLY and blocked Macro stages supply no research source refs.
`.agent/acceptance/phase12-real-input-audit.json` retains the exact original
receipts and current veto. No live/source/financial/scheduler changes occurred.

A bounded broad-unit diagnostic was interrupted after329.80s:803 passed,
2 skipped,1 known weekly component-closure fixture failure. It is not a completed
full-unit/CI acceptance run; final Phase15 checks remain required.

## Current execution order

1. Preserve and verify the original Phase 0 baseline without rewriting its date or
   substituting current heads. This artifact audit is complete.
2. Phase 1 common status metadata is implemented after Architect/Critic review;
   focused checks and real frozen-journal readback passed. Phase 2 is also locally verified.
3. Phase 3 through Phase 6 local validation are complete; proceed to Phase 7 and
   through the remaining phases. Do not mark a phase complete from keyword matches.
4. Incorporate ongoing native receipts as evidence for the capabilities they
   actually cover. A passing frozen-source run does not certify later source edits.
5. Phase13 research diagnosis and Phase14 default ledger-consuming evaluator are
   locally validated. Preserve broader producer coverage audit and final installed
   acceptance. The legacy seal-only flag is retained as component timing and never
   admits new OOS. Resume Phase12 pre-Core provisioning/bootstrap and deployment
   prerequisites, then final current-source full-path acceptance. Historical audit
   `.agent/acceptance/phase13-14-audit.json` is a before-state, superseded by the
   Phase13 and Phase14 validation receipts for their stated scopes.

## Live acceptance at scope update

- Five-day native run: session `36956`, producer
  `3c46cb36918ca00d4fb61707bcfc16ca06b5c2a1`, root
  `/private/tmp/myquant-final-source-native-20260909T030031Z`.
  All five dates (Aug27, Aug28, Aug31, Sep01, Sep02), final read-only replay and financial invariants passed; session36956 terminal0. This frozen 3c46 run does not include later Phase 1 or Phase 2 code.
- Independent baseline at
  `/private/tmp/myquant-public-history-native-20260909T031046Z` passed (session
  `36966` terminal 0). Installed Morning session `4263` also returned terminal 0.
  Its proof is synthetic research replay, not a live Morning success receipt.

The design prohibits new factors/weights, strategy changes, brokers/trades, owner
stop changes, expanded paper authority, holdings mutations and weakened checks.
Continue within those boundaries. Deployment/automation changes must be concrete
and reviewable and use the applicable authorization boundary when reached.

## 2026-09-14T05:57:35.148084+00:00 — Current Phase15 reader repair and acceptance restart

The actual second-day configured catch-up exposed a Dashboard input mode mismatch after the prior native EOD replay passed. The bounded seven-file repair passed Architect/Critic, 249 focused tests, static/access checks and a newly verified installed read of the exact retained16 input refs without changing their bytes or metadata. Evidence: `.agent/acceptance/phase15-catchup-input-mode-repair.json`.

Candidate install `c3c0b38263f8cd045b9a7f18e1d5fff2b4c2848b` correctly refuses the original `bf078...` request at the exact running-install boundary. Old E proofs remain version-scoped and its interrupted harness is not called uninterrupted. A fresh finite installed five-day/Morning/boundary chain is running at `/private/tmp/myquant-catchup-input-mode-20260914T054810Z` (handle74723); no full-chain PASS yet. Independent original Case4 recovery6375 has committed Store but still awaits its final proof. Overall Phase0–15 remains incomplete, including production migration/owner evidence and observation gates.
