# Phase 11 — catch-up routing and automatic gap resolution

Status: Part A COMPLETED_LOCAL_VALIDATION after accepted Architect amendments
and Critic APPROVE. Receipt `.agent/acceptance/phase11-part-a-validation.json`:
265 focused tests passed; follow-up 56 passed includes two additional handoff
cases. Eight-source mypy, fourteen-file Black and scoped flake8 passed. A fresh
native synthetic Dashboard scenario reproduces the benchmark-tail rejection in
current mode and succeeds in historical mode with no serving writes or repeat
changes. Full EOD admission is outside that component proof. Part B subsequently
completed Architect/Critic review and local implementation; the final combined
382-check receipt is `.agent/acceptance/phase11-validation.json`, with its contract
in `phase11_automatic_resolution.md`. Installed/full-DAG acceptance remains separate.
This does not narrow the full Phase0–15 objective.

## Observed defects and scope

`.agent/acceptance/phase11-before.json` reproduces current Calendar rejection as
CATCHUP_SESSION_NOT_HISTORICAL and historical request.v2 derivation retaining
publish_current_dashboard=true. The current handler also applies current-serving
completion to every completed date and requires a caller-selected predecessor.

Repair the existing production daily-close -> catchup binding -> native
maintenance -> materialization -> native EOD -> serving path. No new factors,
weights, thresholds, trades, holdings/cash mutation, providers in local tests,
deployment or automation changes. Preserve all immutable old requests/SHAs and
dirty worktree changes. No latest-result directory scan or retired fallback.

## Part A — exact mixed-date routing

Add cn-daily-catchup-recipes.v2 with exactly schema_version, recipes,
publication_policy. Policy is HISTORICAL_ONLY or CURRENT_OBSERVED_CLOSE_ONLY.
Only execute-recipe.v4 templates may be newly executed under collection v2,
ensuring native v5 capture-before-EOD publication. Legacy collection/binding v1
and their already recorded bytes retain their existing interpretation.

The policy is an explicit instruction for unmaterialized templates. Derive each
day's Dashboard flag before any immutable request or native input exists:
HISTORICAL_ONLY -> false for every day; CURRENT_OBSERVED_CLOSE_ONLY -> true only
when day == root target == authorized close == Calendar observed local date and
the native current classifier proves MATCHED_OPEN.
All earlier days are historical captures. Never edit a supplied already
materialized native input or old binding to obtain those flags; reject an
incompatible unfinished supplied input. Completed earlier EODs may be consumed
as immutable historical evidence, independently of whether their original current
publication later expired. Do not claim their old EXECUTE serving action succeeded.

Add cn-daily-catchup-binding.v2, fixed binding.v2.json under the existing
root-request SHA/date directory. Keep all v1 fields and add maintenance_mode,
dashboard_mode, publication_policy. Enums are HISTORICAL / CURRENT and
HISTORICAL_CAPTURE / CURRENT_LATEST_EOD. Both modes derive from the same original
exact Calendar/raw bytes, not a filename or current wall-clock guess.

For day < Calendar observed local date, require native classify_catchup_session
and the existing historical maintenance fileset/custody path; derive the existing
production-request.v2. For day == observed local date, require the native current
requested-session classifier to prove MATCHED_OPEN, target equality and exact
adjacency to the actual previous completed EOD; derive ordinary request.v1
EXECUTE. No future/unclosed target, calendar gap or previous-date mismatch.
Current maintenance uses the existing fresh-current path, not a historical
exception or supplied historical Calendar. Its fresh Calendar must also agree
on the requested predecessor before the first business write. Add the smallest
internal expected-predecessor guard if needed, retaining native Calendar/raw
replay and truthful failure records. Do not infer holidays.

Current derived requests need no historical handoff: their existing immutable
request/recipe and verified predecessor are the normal native authority. The
controller's v2 binding retains derivation proof and must match the current
request bytes. Historical requests continue passing their exact binding through
historical maintenance/handoff and replay. Update fixed path/version readers
consistently; no fallback search between binding versions.

Readback reconstructs original modes from original Calendar bytes and stored
policy. Fresh execution guards may refuse stale/current-day work; completed
readback never retimes a historical receipt or rewrites publication intent.
An existing native maintenance core/handoff follows its existing exact resume
path. Unknown or incompatible input/claim ownership remains a clear conflict,
not permission to create a replacement claim or mutate an old request.

For v2 collections, a completed intermediate date is acknowledged only after full
native EOD replay and explicit predecessor/Calendar continuity. Skip its current
serving gate as historical consumption. For unfinished correctly routed
historical inputs, their native Dashboard flag is already false, so the existing
post-EOD gate naturally performs no current publication. Do not pass an unchecked
caller flag to disable serving. Latest CURRENT_LATEST_EOD still requires the
Phase9 publisher/readback. A valid recorded publication that is now expired may
be acknowledged as dated history; an expired latest EOD with no proven
publication remains EVIDENCE_SEALED_PUBLICATION_EXPIRED. No freshness renewal,
backdated receipt, new selector bytes after expiry, or fabricated publication.
A later new target may advance using the old native EOD as history.

The original frozen3c46 failed historical request remains unchanged and not
completed as authored. Its wrong mode/old release is not a transient provider
failure. Keep the original failure evidence, and validate the corrected routing
with a new independent current-source native scenario, clearly distinguished
from recovery of that old job. Do not restart or poll its terminal process.

## Part A acceptance

- Same-day target plus preceding missing historical dates dispatch in Calendar
  order with native historical/current classifiers, and no current-day historical
  override. Fresh Calendar predecessor disagreement blocks before business writes.
- Historical-only and intermediate bindings derive false Dashboard flags; only
  the authorized latest target may derive true. Source templates remain unchanged.
- Wrong mode, target, predecessor, policy, schema, bytes or materialized inputs fail.
- Binding persistence/replay is exact and idempotent, including crashes between
  its existing three writes; v1 readback remains unchanged.
- Already completed earlier dates with expired serving are consumed only as native
  EOD history and do not block a later target; current latest publication failure
  still stops success. No old publication receipt is fabricated.
- Current-source native historical Dashboard with a future benchmark tail succeeds
  through the proper historical path without publishing current UI; preserve old
  failure evidence. Full installed historical/full-DAG proof remains final gate.
- Focused public catchup/materialization/requested-session/maintenance/handoff,
  Dashboard/serving and regression tests plus applicable static checks.

## Part B — automatic operation (implemented in the separately reviewed plan)

Provide an explicit versioned automatic request through the same public command,
with code-owned expected close from fresh native Calendar and automatic anchor
selection. Reuse the Phase9 completed-head as a locator, validate its exact EOD,
and inspect only Calendar-derived fixed completion paths forward from it. Every
adopted prefix must pass full native EOD replay and exact predecessor continuity;
never select by directory recency, mtime or a loose latest-result scan. A missing
head requires an explicit validated initial completion seed (normal bootstrap
EXECUTE remains the initial setup path); a present corrupt head cannot fall back
to that seed. Missing/inconsistent ancestry or a completion beyond an unfilled
gap stops before writers.

Freeze the resolved anchor, target, ordered dates, Calendar refs, head preimage,
per-date scope and derived explicit catchup request before execution. Repeats must
read that same resolution rather than reselect a moving head. A nonblocking
strategy-level automatic-run lock and a narrowly scoped active-run record may be
needed so overlapping EOD/night invocations resume one exact run rather than
launch competing day claims. New input/release/policy conflicts cannot be ignored.
Do not add an independent EOD authority index or a scheduler implementation.

Automatic PLAN must remain read-only; execution/resume invokes Part A and the
existing native producers only. Source recipes/native inputs are still exact
prerequisites; missing evidence must be reported by date, never synthesized from
stale or different-day artifacts. The later automation phase will prepare those
inputs through the existing intended acquisition path. Automatic result metadata
must expose the original request/resolution and historical versus current-serving
scope so complete historical consumption cannot be mistaken for renewed current
publication. The exact new schema and concurrency/recovery rules require a
separate bounded review before Part B edits.


## Accepted Part A Architect amendments

Architect REVISE items are accepted; this section makes the execution contract
exact before Critic review. Part B remains open and is not covered by Part A.

| Calendar relationship / collection policy | Maintenance | Dashboard | Derived flag |
| --- | --- | --- | --- |
| day < observed local date, either policy | HISTORICAL | HISTORICAL_CAPTURE | false |
| day == observed local date, HISTORICAL_ONLY | CURRENT | HISTORICAL_CAPTURE | false |
| day == root target == authorized close == observed local date, CURRENT_OBSERVED_CLOSE_ONLY and MATCHED_OPEN | CURRENT | CURRENT_LATEST_EOD | true |

Reject every other combination. In particular Friday on a Saturday/Sunday
Calendar is historical, even if Friday is the latest authorized close. No new
current publication is inferred for it. Reconstruct modes from the immutable
original Calendar/policy on readback; do not retime them from today's clock.

Binding v2 is created only for recipe-derived dates. Its exact field set is the
existing v1 FIELDS plus maintenance_mode, dashboard_mode, publication_policy.
Collection v2's exact field set is schema_version, recipes, publication_policy.
Known version selects the fixed binding filename; no probing/fallback between
versions. New incomplete supplied native inputs receive no binding and must be
native v5, exact trade_date/previous_trade_date, fixed Phase9 policy, and the same
mode-derived publish_current_dashboard flag before any business writer. Never
rewrite their bytes. Completed input receipts are checked against their retained
native input ref rather than overwritten.

Current recipe-derived execution remains ordinary request.v1 and does NOT pass
_catchup_binding_ref into native maintenance/handoff. The controller still stores
binding.v2 and verifies the derived request SHA against completed handoff. Past
recipe-derived execution remains request.v2 and carries its exact historical
binding through existing native maintenance/handoff/replay. No new request or
handoff schema is needed solely for current batch membership.

Add a pure current-predecessor classifier alongside classify_requested_session.
It replays the same exact Calendar/raw bytes, requires MATCHED_OPEN, requires the
supplied predecessor to be present, and proves it is the immediately preceding
open session. Keep classify_catchup_session unchanged. Pass only the expected
predecessor scalar into current maintenance. After fresh Calendar capture, a
mismatch may retain claim/Calendar/failure evidence but seals
REQUESTED_SESSION_BLOCKED with zero canonical business writes and no callbacks.
Finalized current replay must prove the same original Calendar target/predecessor
before any core callback, handoff or materialization. The current handoff readback
also checks recipe.previous_completion_ref against its retained Calendar.

For every completed-prefix date in a v2 collection: fully replay its native EOD,
read its exact native_inputs_ref, compare previous_trade_date to the controller's
actual predecessor, and replay its own retained Calendar/raw files to prove
adjacency. Missing native inputs/Calendar/predecessor proof is a conflict, not a
historical-serving shortcut. A validated historical date bypasses current-serving
publication only as CATCH_UP historical consumption; the old EXECUTE action is
not relabeled successful. Only CURRENT_LATEST_EOD invokes Phase9 serving gates.
Failure in that latest current gate remains incomplete. Legacy collection v1
scope and records remain unchanged.
