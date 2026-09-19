# Case8 acceptance with path-bound future Calendar evidence

Status: Architect amendments incorporated and Critic APPROVE. Harness prepared; installed case NOT RUN. No business/runtime
source edits or changes to the running frozen candidate5060dee2.

Original requirement: Market/PIT SHA mismatch blocks the full EOD entry. Existing
helper `_native_installed_source_mismatch.py` physically copies a completed day,
then corrupts one copied output and asserts exact native status and entry errors.

New prerequisite conflict: Calendar now binds a next-session proof. That proof's
native capture-root receipt binds directory device/inode and absolute capture
parent. Copying the producer workspace cannot reproduce that physical custody;
merely copying nested proof leaves is insufficient. Do not rewrite capture IDs,
source SHA, inode values, Calendar terminal refs or EOD refs to make a copy pass.

## Proposed narrow read-only test harness

Keep the completed original producer workspace read-only and use its exact
installed interpreter and original completed-day references. Require the original
configured driver PASS and configured-native-proof.json with full16-node native
replay before starting. Native status and Core handoff must pass again with no
fault. This is a test harness, not a production reader or fallback.

Create two isolated physical fault files under a new private fixture root, each
at the same workspace-relative path as one selected immutable output:
market.market_input_ref and pit.market_pit_selection_ref. Copy exact original
bytes, verify SHA, then flip one byte without changing length. No original file
is written or resealed.

For each negative case, temporarily patch only
SecureSystemStorage.read_workspace_file_bytes in this separate test process.
The wrapper routes exactly the selected original-workspace/path pair to a real
SecureSystemStorage instance rooted at the physical fault fixture. It invokes
the saved original method for that read and every other read; it never fabricates
bytes, SHA, stat metadata, validator results, node status or exceptions. Every
other path/workspace continues through the unmodified original reader. Restore
the method in a finally/context-manager boundary. Count routed reads and require
that the intended faulty file was actually consumed.

Native read_daily_status must produce STALE with
DAILY_STATUS_OUTPUT_SHA_MISMATCH for precisely the intended node. Native
replay_native_completion must refuse exactly EOD_READBACK_TERMINAL_CHANGED:<node>.
Do not replace these validators or accept an arbitrary parse/missing-file error.
The injected routing seam must be reported as
ONE_PHYSICAL_OUTPUT_CORRUPTION_WITH_EXACT_READ_ROUTING_TEST_SEAM; do not claim a
seam-free physical clone of the entire workspace.

Forbid journal/financial publication and network in the test process. Preserve
exact bytes/modes/mtime of selected immutable original closure and capture-root
physical identities. Track selected Day1 immutable evidence rather than mutable
current heads, because the normal automatic successor driver may advance later
trade dates independently. No claim that a concurrent whole workspace stayed
unchanged. Full original proof/selected closure refs and fault-file refs/hashes,
read-hit counts, exact errors and harness SHA go into a separate acceptance receipt.

## Alternatives and decision needed in review

- Relabeling/resealing copied capture evidence is prohibited.
- Faulting/restoring the running primary workspace risks concurrent successor
  interference and modifies sealed evidence; reject this route.
- A separate full EOD producer dedicated to physical in-place faults would avoid
  the routing seam but repeats an expensive full producer. Use that alternative
  if the reviewer finds the scoped seam insufficient for the original requirement.
- Reusing old-version Case8 as current evidence is prohibited.

## Verification and stop conditions

- Test the routing wrapper with two roots and multiple paths: exactly one pair
  redirects, the returned bytes/SHA/stat come from the real faulty file, all other
  reads stay original, and exceptions restore the method.
- Require a positive baseline and exact error assertions; no optional bypass for
  unavailable future proof/custody. Original driver/code/install mismatch stops.
- If frozen source uses another reading API for these paths, review that exact
  boundary rather than broad monkeypatching. No production code changes.
- Actual installed Case8 remains NOT RUN until first-day full native proof exists.

## Architect refinements (governing)

- Run a fresh positive replay_native_completion in this harness process before
  faults, and require native_replay_validated plus exactly all16 validated nodes.
  Also require all16 current selected node statuses SUCCEEDED. Prior producer
  receipt is prerequisite only, not a substitute for this positive call.
- For each negative call, exactly the target node is STALE; every other EOD node
  remains SUCCEEDED. Exact reason/entry errors above remain mandatory.
- Fault files use canonical resolved private root/parents, owner-safe directories,
  mode0600, current owner and one hard link. Flip one bit, preserving byte length
  and requiring a different SHA. Use separate Market and PIT roots so only the
  currently selected source is corrupted/routed per case.
- Forward maximum_bytes unchanged. Count wrapper calls, exact target redirects,
  successful native physical reads and non-target redirects; require target hits,
  successful reads equal redirects, zero non-target redirects and no recursion.
- Reuse existing readonly_replay_guard for native consistency-lock behavior and
  publication/network prohibitions; do not veto necessary native read locks.
- Inventory explicitly enumerated selected immutable evidence: completion,
  release/native-input refs, all16 terminal/request/output refs and Core handoff,
  plus the exact bound future Calendar publication/core/provenance and full native
  capture root. Record the full inventory path list and its scope. Include capture
  parent/root path and root device/inode; do not claim whole concurrent workspace
  invariance or infer that mutable current heads cannot advance independently.
- Runtime source remains unchanged. Implement a separate pinned acceptance helper
  under .agent/acceptance, leaving old-version harness/proofs and frozen repository
  tests untouched. Its focused routing tests are separate harness verification,
  not additions retrospectively claimed in the already-passed full CI.
