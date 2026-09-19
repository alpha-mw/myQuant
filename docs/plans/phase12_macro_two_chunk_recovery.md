# Bounded Macro recovery through the actual Sep16 Market frontier

Purpose: remove the proven Macro admission blocker using existing native APIs.
No runtime code or validation-rule change. User approval for the exact live
acquisition/canonical writes is pending; preparation alone does not grant it.

## Exact prepared inputs

- Runtime: independently installed CI-passed 5060dee2d4b39b50a735393d056ebe5adb4efe3b.
- First contract: `.agent/acceptance/phase12-macro-recovery-transaction/transaction-first.v2.json`.
- Native current-Market validation: same directory, `current-market-validation.json`.
- First dates: Sep4,7,8,9,10. Second dates: Sep11,14,15,16.
- Original eight per-date reconstructions remain explicitly retrospective.
- Current Market source is Sep16; do not claim Sep18 readiness or historical OOS.

## Execution

1. Recheck exact contract/input hashes and four current pointer preimages, original
   veto, source/capture lineage and actual installed identity. Drift stops execution.
2. After authorization, call existing prepare_cn_macro_maintenance_transaction
   once for first chunk, authority_mode canonical, with exact v2 contract.
   allow_live permits only the two native requests pinned in
   `live-request-inventory.json` beside the first contract: GET over HTTPS,
   fixed hosts/paths, redirects disabled, environment proxies disabled,
   native response limit and parser/code identities. Send no credentials,
   account facts or portfolio symbols. Additional URLs/network errors stop.
3. Candidate preparation may write only its new private run root. Confirm its
   prepared receipt and native commit preflight, then invoke existing journaled
   commit once. Require returned status SUCCESS, terminal true, target Sep10,
   exact journal_path and a POSTCHECK_PASSED then TERMINAL/SUCCESS journal chain.
   Actual release/observation pointer hashes must equal both returned hashes
   and prepared component new_pointer_sha256. Preserve original veto evidence.
4. Derive second contract and lineage from actual first commit outputs. Never
   predict SHA values or reuse first preimages. Use exact Sep16 final canonical
   coverage: three retrospective dates Sep11/14/15 plus Sep16 final coverage.
5. Prepare/commit second chunk through the same entry. Require the same SUCCESS,
   terminal, journal and actual pointer checks, now target Sep16. The native
   commit _postcheck loads both canonical components and records POSTCHECK_PASSED
   while the veto is still present. Confirm lineage/coverage through Sep16 and
   unchanged Market/PIT. Then call clear_cn_daily_write_veto with lane macro,
   original expected veto SHA and exact terminal-journal reason; verify the
   archived bytes and clear receipt. That API does not itself prove recovery
   success. Do not unlink/rewrite veto or clear after the first partial chunk.
6. Build the exact second terminal's closure using build_macro_readiness_closure
   with default strict veto handling, then validate_macro_readiness_closure.
   Require intrinsic status READY, target_date 20260916, exact journal/pointers,
   and actual post-recovery available_at. This is historical closure only.
   Do not call verify_current_macro_readiness_closure with a fabricated Sep16
   decision time: the actual Sep18 recovery was not available on Sep16. A fresh
   current-date Macro/EOD run is still required for current operating readiness.
   Keep Market/PIT, Factor/System pointers, Store/Event/holdings and schedules
   unchanged. Any unresolved readiness error remains blocking.

The veto-clear reason format is fixed:
`phase12-macro-recovery:20260916:terminal-sha256=<actual second terminal SHA256>`.
Resolve the digest from the validated immutable terminal journal; record its
exact path/digest before clearing. Never invent the digest in advance.

Before clear, call build_macro_readiness_closure with the exact second terminal
path/SHA. Require precisely MACRO_READINESS_VETO_ARCHIVE_INVALID and confirm the
expected archive path is genuinely absent (an existing malformed archive is not
acceptable). The builder must have reached the veto-lifecycle stage after all
preceding journal/component/date checks. Separately require original veto SHA,
exact current Market/PIT/Release/Observation hashes, and all non-veto input refs.
Any other error or drift stops with the veto preserved. Do not simulate an
archive, patch a reader or backdate availability.

After clear, require intrinsic READY/target 20260916, veto_lifecycle CLEARED,
the original veto SHA, exact archive/clear-receipt refs and available_at equal to
the real clear time. Replay that exact closure and recheck all four pointers.
Failure stops as an anomaly; do not repeat commits/clear or invent a rollback.

## Stop conditions and limits

- No force/CAS override, pointer rollback shortcut, date spoofing or retry under
  a new run id after ambiguous commit. Inspect exact journal and use native
  recover_macro_transaction only for supported interrupted states.
- First commit followed by second failure is explicitly partial recovery;
  preserve actual first progress and leave veto/full readiness blocking.
- Do not rerun CI or the already passed per-date projections. No new framework.
- Review only this transaction sequence, especially intermediate-date support,
  retrospective availability, live destinations and exact veto lifecycle.
