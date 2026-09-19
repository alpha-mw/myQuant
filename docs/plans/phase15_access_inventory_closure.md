# Close the source inventory gap found by frozen-repository CI

Current source was committed only inside isolated CI checkout
`5fb93c405e4c821e1c1fa99249fd8c99af15d433`. The main checkout's scanner sees tracked
files plus existing reviewed rules, so newly untracked implementation files were
not all covered. The isolated checkout made them tracked and correctly reported
29 unexplained callers. Every scanner-classified target-operation set is empty;
this is not proof that a module has no other direct filesystem calls.

Initial proposal is the explicit29 exact-path `AllowRule(operations=())` entries
now drafted in `scripts/check_strategy_record_access.py`. Reasons distinguish pure
contracts/reports, native readers, DAG-only metadata writers, and delegation to
the existing manager/benchmark owners. No wildcard, operation allowance,
parser, source gate or financial behavior changes.

Architect review found a real ownership gap: `registered_daily_event_sources`
contained actual directory/immutable-file writes in `publish_prepared`, relying
only on its current caller holding the manager lock. Revised implementation:
move those exact operations into `command_publish_registered_event_declaration`
inside its existing operation-lock block. Directory creation is a nested helper
inside that locked entry, not an independently callable writer. Preserve all
path/alias/mode checks, exact native bytes, existing timestamp on repeat, native
readback and public result shape. Reader module retains pure candidate construction
and readback; its old writer name becomes an explicit rejecting tombstone, with
no I/O. No new lock/capability system or financial state mutation is added.

Also refine the recovery caller reason: after guarded metadata repair it resumes
only the already-bound native input, whose frozen Store proof precludes another
CAS. Tests must prove retired helper writes nothing and direct manager invocation
holds the operation lock before directory/file publication, alongside existing
registered declaration/repeat/tamper checks.

Review the exact new entries against source semantics, particularly
`scripts/daily_source_inputs.py` and `scripts/registered_daily_event_sources.py`,
and the journal-only writers. Reject any entry whose rationale hides a direct
financial/policy writer or ungoverned delegated publication. Existing native
writer/manager authority and locks must remain unchanged.

Acceptance: scanner on a committed view of all current source has zero unexplained
callers; scanner's unknown/direct-operation/wildcard/drift tests still pass; all29
new entries have empty operations and unique paths. Preserve original failed CI
output. Test fixture repairs (cutoff mock missing receipt profile and isolation
tests relying on a real canonical directory) are separate, without runtime gate
changes. This is inventory closure, not production deployment or full goal closure.

Implementation and local validation (2026-09-14): Architect and Critic approved.
The manager now owns directory creation, both immutable writes, fsync, readback
and source recheck under its original operation lock. The existing context manager
yields its descriptor (other callers ignore it); this entry checks the same file
identity and actual exclusion before/between/after writes and readback. Retired
reader helper raises REGISTERED_EVENT_MANAGER_PUBLICATION_REQUIRED without I/O.
The focused 162-test set passed, including real lock probes, replaced/released lock,
source drift, exact pointer-only retry and repeat timestamps. All stable static
gates pass. The freshly committed f7968454dcf537b20369a762bdbb16f56b2ee019 view
passes the access scanner with 65 reviewed callers; the 29 additions remain exact
empty-operation rules. Full CI and native end-to-end acceptance are tracked
separately; this repair does not complete the overall goal.
