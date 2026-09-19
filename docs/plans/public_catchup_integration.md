# Complete public catch-up integration

Implementation status (2026-09-09): Architect corrections and final Critic APPROVE
incorporated. Public dispatch, template derivation, private historical execution,
handoff/archive provenance, and finalized-maintenance recovery are now implemented
in the main checkout. Focused integration/regression: 197 passed in 8.46s, including
public CLI two-day interruption/retry/no-write completion, fault injection after
each generated-file write, source/binding bypass rejection, and real finalized
historical maintenance lock/core replay. Public test writers/install/EOD replay are
controlled seams; this is not the installed full-DAG/five-session acceptance.
Full installed acceptance and the final goal audit remain outstanding.

This is one integration deliverable in the main checkout, not an isolated schema
exercise. Preserve the user objective and stop adding disconnected versions.

## Public input and deterministic resolution

Keep the existing production daily-close/CATCH_UP surface and request.v1 field set.
Permit its existing recipe_ref to optionally name a canonical collection:
`{"schema_version":"cn-daily-catchup-recipes.v1","recipes":{YYYYMMDD:template}}`.
Without recipe_ref, preserve the existing materialized-input-only path unchanged.
Each template uses the existing execute-recipe.v1/v2 fields, with exactly three
runtime-derived entries required null: previous_completion_ref, bootstrap_ref and
store_preimages.store_pointer_ref. Event/benchmark refs and every other field are
explicit caller inputs. No callbacks, source scanner, clock, success or permission
flags. Date, market, strategy, graph and release/install must bind to root request.
Collection date keys must be within Calendar-required dates and cannot overlap
root day_input_refs. All template shapes and declared source refs are checked before
writers. Existing Calendar/raw validation remains authoritative. This upstream path
is historical-only: requested dates precede Calendar observation and actual runtime.
Today still uses the unchanged ordinary EXECUTE path; no Calendar rule is relaxed.

For each date in verified Calendar order, native-replay the fixed completed EOD if
present and return NO_ACTION. Otherwise use supplied native inputs or the declared
template. A missing/failed day blocks successors. For a template, fully replay the
immediate previous EOD, read its exact Store terminal/pointer ref, and use that SHA
as expected canonical Store preimage. Revalidate actual current preimage using the
existing native Store gate before fresh execution. Never move a pointer backwards.
Fill previous_completion_ref from the preceding actual completion, bootstrap=None,
and the Store preimage from that native evidence. Validate the fully resolved recipe.
No future hash is guessed and no original template is changed.

Persist generated recipe, ordinary EXECUTE request, and a deterministic binding
(root CATCH_UP request ref, template collection ref/date, previous EOD ref, generated
recipe/request refs) under the known day/catchup/root-request-SHA directory with
existing immutable JournalStorage under one short day lock. Release it before
maintenance. On retry, reread and reconstruct the binding from exact recorded refs;
never choose new preimages from mutable heads or change immutable input bytes.
The root request and binding provide the source/derivation trace for generated input.

## Existing execution path

Extend execute_daily_recipe with a private code-only historical Calendar input,
provided solely by the new fixed batch controller from the root request. No public
clock override. Existing handoff/completed EOD replay stays first. Native installed
context, policies, Store and previous-EOD gates stay authoritative for fresh work.
For a finalized historical maintenance claim without handoff, use existing descriptor
lock and native core replay before fresh Factor-preimage checks; verify target and
fixed core ref, and match recorded Calendar/raw to root inputs before callbacks.
No claim or budget renewal. Ordinary current-session path unchanged. Historical
fresh execution calls the already implemented native historical maintenance branch,
then existing handoff.v3/materialization/native DAG/seal paths. Each completion is
fully replayed before advancing the chain. Unknown/failed native outcomes cannot
produce COMPLETE. Preserve completed prefix and block later dates.

## Verification and delivery

Architect then Critic review before implementation. Reuse existing planner/result
contracts, completion replay, recipe validator, execution controls, Store gates and
native script bridge. Add only the collection/controller/derivation glue required.
No bypass of PIT/SHA/calendar/prospective/authority rules. Native script import map
and exact access-review entry must cover the controller; no wildcard relaxation.

Acceptance is public CLI -> real batch planning/resolution -> existing per-day
execution -> bound results, including two missing upstream dates and interruption
between them, repeat/no-write completed prefix, malformed source/date/previous refs,
source drift and authority rejection. Controlled native seams in fast tests are
identified as such. Then exercise installed native historical full-DAG inputs and
at least five synthetic consecutive sessions on final source; preserve original
failures, no relabeling of synthetic evidence as live. Main source and concurrent
user changes must remain intact. Full goal is not complete until these end-to-end
requirements and final-source checks pass. No live calls, deployment or trading.

## Accepted architecture correction

The derivation binding must be reachable from authoritative downstream evidence.
Amend unshipped handoff.v3 with mandatory catchup_binding_ref; publisher and both
readers validate/recheck it. It contains root request ref, collection ref and date,
original Calendar/raw refs, previous actual completion, previous Store terminal and
pointer refs, and generated recipe/request refs. Reconstruct generated bytes exactly.
Private historical execute requires this binding; a free Calendar mapping is not
accepted by execute_daily_recipe. Handoff compares root Calendar semantics/raw SHA to
its retained attempt proof. Missing root/binding/original sources block replay.

Completed-day resolution takes precedence: a pre-existing validated EOD can be adopted
without a new binding; declarations in the immutable original collection do not become
conflicts merely because a retry now sees completed work. Once this controller's binding
exists, however, completion must match its generated request SHA or fail conflict.
Materialized-input and template declarations are disjoint. Missing ownership for any
uncompleted required date blocks before writers. Every successful completion is fully
replayed before becoming the next predecessor. Source/derivation checks are pure reads;
full native previous-EOD replay is performed by the controller before maintenance, so
handoff publication under the maintenance lock does not recursively reacquire that lock.

## Exact binding and private execution contract

Binding schema is cn-daily-catchup-binding.v1. Exact fields:
schema_version, market, strategy_id, trade_date, graph_sha256, release_install_ref,
root_request_ref, collection_ref, calendar_ref, raw_calendar_ref,
previous_completion_ref, previous_store_terminal_ref, previous_store_pointer_ref,
recipe_ref, execution_request_ref, authority. Market/strategy/graph/install bind root;
authority is exactly the existing all-false declaration. Every ref is the existing
exact workspace-relative path/lowercase64SHA grammar. No optional extras or clocks.

Fixed directory: results/operations/daily_production/CN/<day>/catchup/<root-request-SHA>.
Files: recipe.json, request.json, binding.v1.json. Generated request is
cn-daily-production-request.v2, action EXECUTE, same13fields as request.v1, same
ordinary EXECUTE nullability; reserved for the controller. execute_daily_recipe
accepts only private _catchup_binding_ref for this request version. V2 without
binding, v1 with binding, free Calendar override, wrong root/collection/date/ref,
or different derived bytes must reject before claims/maintenance/handoff/provider.
Public direct EXECUTE v2 has no way to supply that private parameter and rejects.
The root CATCH_UP remains request.v1 and public surface is unchanged.

Canonical generated recipe/request bytes are reconstructed from exact binding inputs.
Write recipe, request, then binding once under the day lock. If interrupted after any
write, retry uses the same root, previous EOD and its immutable Store pointer to
regenerate the identical set. Existing differing bytes stop conflict; no alternate
path, newer mutable preimage, second binding, or orphan authority. No writer uses
the generated request until binding validation has completed. Inject failures after
each of the three writes and prove exact-byte recovery or conflict. A present invalid
binding blocks completed-day adoption; no-binding pre-existing completed EOD retains
priority. A binding-owned EOD must match the generated request SHA.
