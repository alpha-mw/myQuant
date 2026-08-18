# Unified Runtime Hard Cutover

This directory defines the repository-facing migration from version-named
runtime surfaces to one stable myQuant runtime. The cutover is intentionally
breaking: supported callers use the stable packages and the single
`quant-investor` executable after migration.

The cutover changes names and routing, not authority. It does not publish a
mainline result, admit or activate a factor, advance a pointer, create a paper
fill, connect a broker, or trade.

## Stable public surface

The supported runtime is organized by responsibility:

| Responsibility | Stable package or command group |
|---|---|
| Canonical contracts and sealed artifacts | `quant_investor.contracts` |
| System verification, status, suspension, and generation storage | `quant_investor.system` / `quant-investor system` |
| Factor bootstrap, prospective admission, and read-only health | `quant_investor.factors.governance` / `quant-investor factor` |
| Research compilation, forward observation, evaluation, and inspection | `quant_investor.intelligence` / `quant-investor research` |
| Exact active-result resolution and read-only projection | `quant_investor.mainline` |
| Public operator entrypoint | `quant-investor` |

The stable Python surface is responsibility-specific:

- contracts: `canonical_json_bytes`, `ContractDefinition`,
  `register_contract`, `seal_artifact`, `validate_artifact`, and
  `artifact_byte_sha256`;
- system: `SystemStore`, `SystemStore.pointer_history(newest_first=True)`,
  `SystemStore.build_validation_run_request`, `SystemStore.run_validation`,
  `build_suspended_generation`, and `suspend_system`;
- factor governance: `bootstrap_factor_definitions`,
  `build_bootstrap_exception_evidence`,
  `validate_bootstrap_exception_evidence`,
  `build_bootstrap_factor_set`, `validate_bootstrap_factor_set`,
  `compute_bootstrap_signals`, `FactorValidationStore`,
  `BootstrapClosure`, deterministic namespace helpers, and the stable artifact
  validators. Prospective and status writes are available only through
  `FactorValidationStore`; contextual replay is available only through the
  fixed System validation profiles;
- intelligence: `forward`, `evaluate`, `compile_evidence`,
  `assess_readiness`, and `inspect`;
- intelligence domains: `decision_context.build_decision_context`,
  `investment_decision.make_investment_decision`,
  `industry.assess_industry`, `theme.assess_theme`,
  `fundamental.assess_fundamental`, `advisory.review_advisory`,
  `advisory.replay_advisory`, `portfolio.construct_research_portfolio`,
  `portfolio.observe_paper_portfolio`, and `portfolio.assess_graduation`;
- mainline: `MainlineStore`, `MainlineError`, `read_public_run`,
  `build_mainline_candidate`, and `mainline_status`.

These names describe callable responsibility, not a compatibility layer. The
source modules and exact signatures in the intended checkout remain final
authority.

The unified system store owns the following authority-relevant layout:

```text
results/system/objects/<kind>/<byte_sha256>.json
results/system/generations/<generation_id>/manifest.json
results/system/pointer_history/<pointer_byte_sha256>.json
results/system/candidate_state/<namespace_hash>/...
results/system/validation_requests/<validation_request_id>.json
results/system/validation_runs/<namespace_hash>/...
results/system/validation_custody/<attestation_byte_sha256>/...
results/system/source_verification_cache/<attestation_byte_sha256>/snapshot.json
results/system/_active.json
results/system/_migration_complete.json
```

Candidate-state, validation-request, intent/prepared/completion, source-snapshot,
and validation-custody records are owner-only and non-authorizing. Their exact
readback is required by the OPERATIONAL generation verifier. The active pointer
is runtime authority only when the permanent initial-migration marker closes
the detached receipt, final cutover authorization, prepared pointer bytes,
initial generation, and deployed release. An operational pointer without that
exact marker never grants Factor `ACTIVE` authority.
Detached authority artifacts use content-addressed exact-once publication below
an owner-only external root. Every directory component is opened without
following symlinks; both fresh and idempotent leaves must be regular `0600`
files owned by the effective UID with one hard link, and publication completes
only after same-descriptor byte/inode readback and containing-directory fsync.

System status is one of `UNINITIALIZED`, `PARTIAL`, `ACTIVE`, `BLOCKED`, or
`SUSPENDED`. External routing is independently `ACTIVE`, `UNCONFIRMED`,
`SYSTEM_ACTIVE_AUTOMATION_DISABLED`, or `SYSTEM_EXTERNAL_ROUTING_DRIFT`.
The public status payload also reports `calendar_authority_route`,
`calendar_authority_confidence`, and exact `calendar_source_limitations` for
an active production bootstrap generation. An absent active pointer is
`UNINITIALIZED` with
readiness `EMPTY` and blocker `SYSTEM_ACTIVE_POINTER_ABSENT`.

There is no runtime version selector and no compatibility executable. A removed
import or command is an error; callers must migrate to the stable target in the
[CLI mapping](cli-mapping.md) and the repository-controlled source-to-target
rules under `operations/unified_cutover/`.

The machine-controlled rule, custody, and preflight set is exactly:

```text
operations/unified_cutover/rules.json
operations/unified_cutover/dynamic-import-allowlist.json
operations/unified_cutover/source-to-target.json
operations/unified_cutover/pre-cutover-dirty-inventory.json
docs/migrations/unified-cutover/legacy-seed-manifest.json
docs/migrations/unified-cutover/legacy-custody-scope.json
docs/migrations/unified-cutover/replacement-test-map.json
```

The comprehensive legacy seed manifest and legacy custody scope live only in
migration custody so active runtime rules contain no retired identifiers. The
scope owns the baseline custody facts, retired pointer-filename rules, and
legacy runtime roots. It also binds every retired source-only active path to its
baseline byte hash and a `REPLACED` or `BEHAVIOR_INTENTIONALLY_REMOVED`
disposition; active rules carry only the custody path and its expected byte
hash. Each file is compact canonical JSON identified by `kind` and
`contract_sha256`, with no numeric public version field. The replacement-test
map freezes the same approved baseline commit and tree and gives every deleted
legacy test exactly one stable replacement or an explicit intentionally-removed
disposition. Resolve an inventory with
`scripts/resolve_unified_cutover_inventory.py`; build the no-write migration
receipt with `scripts/build_unified_migration_receipt.py`. Neither script grants
external deployment or activation authority.

## Migration sequence

1. Freeze the pre-cutover source inventory and classify every legacy source as
   `replace`, `archive`, or `remove`.
2. Verify the stable contracts, system store, factor governance, intelligence,
   mainline resolver, and CLI without live providers or external writes.
3. Rewrite imports and invocations to stable targets. Do not add aliases for
   removed runtime routes, dynamic-import fallbacks, latest-by-mtime discovery,
   or stale-result substitution. The documented `market download` maintenance
   alias remains outside this cutover.
4. Apply the source-to-target rules once. A collision, ambiguous source,
   missing required target, or hash mismatch blocks the cutover.
5. Run the replacement test map and the full unit, lint, and type gates.
6. Confirm that the unified active pointer and its bound immutable generation,
   the independent market/PIT/Fundamental pointers, the Strategy Record Store
   pointer, the portfolio ledger, installed Codex files, and Dashboard files
   are byte-identical before and after repository verification.

The migration is complete only when every required mapping is resolved, every
removed surface fails explicitly, and every stable replacement passes its
positive and no-write tests. A partially migrated checkout is not a supported
runtime.

## Production cutover gates

Repository implementation and offline rehearsal do not authorize production
installation or activation. Before building the final release or operational
generation, run:

```bash
python operations/unified_cutover/verify_pre_cutover_dirty_inventory.py
```

The canonical pre-cutover manifest freezes the complete dirty set observed at
its captured main-checkout HEAD. It makes no fixed-count, file-category, or
ownership assumption: every row binds the exact two-column porcelain status,
canonical path, byte SHA-256, and size. While any row has
`user_disposition=UNCONFIRMED`, an exact capture readback is still a hard stop;
it is evidence that the user files remain untouched, not approval to absorb
them.

To satisfy this one gate, a user-confirmed clean integration commit that is a
descendant of the captured HEAD must explicitly disposition every captured row as either
`ABSORBED_IN_INTEGRATION_COMMIT` or `EXPLICITLY_DISPOSITIONED`, record a reason
for each row, and bind the integration commit plus confirmation identity and
time. Absorbed rows must have the captured bytes at their captured paths in the
integration commit. Any later HEAD, status, path, size, byte, missing-row, or
extra-row drift blocks before the final build or CAS. A successful report only
makes the checkout eligible for the next preflight gate; it sets both
`final_build_authorized` and `cas_authorized` to false.

Final build and CAS authority is derived only from the complete fixed
System-owned gate set. `run_cutover_gate` selects repository-defined argv and a
scrubbed environment from `gate_id`; callers cannot supply a command or convert
an arbitrary `exit 0` into evidence. Each result seals the exact commit/tree,
runner source bytes, runner specification, executable identity, stdout,
stderr, stdin SHA-256, exit status, timestamps, and subject reference. The
replacement-selector gate loads the canonical ledger, expands all 130 unique
selectors, and rejects any missing collection/execution, deselection, skip, or
xfail. The release-origin gate consumes an exact evidence/release pair on
stdin; it requires frozen Git package bytes, sdist bytes, wheel bytes, a locked
non-editable dependency installation, isolated import origin, installed code
manifest, and compiled contract catalog to agree. A zero-selected pytest run
cannot satisfy either gate. Long-running tests are
executed before the active lock. Under the lock the System revalidates their
immutable receipts and the current frozen tree; it does not rerun pytest.
The `clean_detached_clone` gate and release builder both require an exact clean
detached HEAD, double-read the commit/tree/status/root, and reject even a clean
checkout that remains attached to a branch.
The release builder creates a separate temporary build environment, resolves
the declared `dev` extra from the exact lock in offline mode, and then invokes
the build backend with isolation disabled inside that prepared environment.
The final runtime environment is separately synchronized from the same lock
before the exact wheel is installed non-editably; a missing cached locked
artifact is a hard failure, never a network fallback.

Final authorization validation has two distinct modes. `PRE_CAS_CURRENT`
requires current HEAD/tree and a clean checkout to equal the frozen release.
`HISTORICAL` replays the immutable initial Git objects and authority chain after
activation but deliberately does not require future HEAD to remain pinned to
the initial release. This preserves the permanent marker across legitimate
descendant releases without weakening the first-CAS gate.
The initial production-bootstrap receipt follows the same split:
`PRE_CAS_CURRENT` replays the current installed release, assembler bytes and
all strict source semantics, while `HISTORICAL` obtains the original assembler
blob from the final authorization's frozen Git commit and verifies the anchored
generation without comparing it to a later installed release. Historical
replay parses the initial stored catalog only as non-executable dispatch
metadata, validates each artifact envelope and object reference from exact
bytes, and checks the anchored emergency controller by its stored SHA and
historical metadata envelope. It neither compares the initial catalog with the
descendant compiled catalog nor regenerates the initial controller from the
descendant controller template. Historical source files remain owner/mode/link
checked and exact-hash checked; historical code is read as bytes and is never
executed. Emergency containment walks the secure raw pointer history to the
literal-`EMPTY` root, authenticates that initial manifest and stored catalog,
and statically resolves the one suspended generation and manifest sealed into
the initial controller. It deliberately does not read, repair, or require the
permanent marker, so an absent or malformed marker cannot disable containment.
Only that fixed target may receive the non-empty CAS; a second otherwise valid
`SYSTEM_SUSPENDED` generation is rejected and Factor authority remains blocked.

Fundamental safe-successor replay consumes an explicit sealed fileset resolver.
JSON and evidence inputs use bounded exact-byte reads; Parquet inputs remain
under a stable `O_NOFOLLOW` descriptor for decoding and exit-time path/inode
readback. Decoding uses at most 2,048 rows and 16 MiB per batch, rejects
oversized row groups before decoding, and enforces fixed total row, column,
cell, compressed-byte, and uncompressed-byte bounds. Pointer, manifest,
provider evidence, permanent support refs, sealed target refs, immutable target
refs, and all three Fundamental tables must be present in the declared fileset.
No pathname reopen or whole-row-group `to_pandas()` is authoritative in this
production validation path.

Fundamental freshness is an immutable disclosure, not a numerical admission
threshold. The initial generation must still include one fully valid
safe-successor source, but that source retains its own historical
market/PIT/scope bindings rather than being rewritten to current Factor inputs.
The receipt seals `system_as_of_date`, Fundamental snapshot cutoff,
calendar/open-session age, latest admitted availability date, code-owned Factor
dependencies, and a nullable VETO-only operator reference. A valid veto can
only reduce authority and stops before generation/final authorization/CAS.
Active status reports the sealed advisory values; it never recomputes age from
the current date or a descendant calendar.
Current VETO issuance is bound to the effective UID and to a source leaf owned
by that same UID; frozen historical replay preserves, but does not reinterpret,
the initial actor. Daily forecast admission is row-relative:
`forecast_ann_date` must not follow its own `trade_date`, even when both dates
are inside the overall Fundamental and System cutoffs.

Calendar admission is an exact tagged union. `EXCHANGE_OFFICIAL` remains the
full-authority route and requires the native issuer closure described below.
`TRUSTED_PROVIDER_DEGRADED` is the narrowly admitted daily-Factor route for a
fresh Tushare `trade_cal` transaction: retained official Tushare documentation,
one exact SSE response, one exact SZSE response, and one exact-empty BSE
capability probe. It is not exchange-official authority. SSE and SZSE must have
identical normalized daily rows and finite `pretrade_date` ancestry. BSE rows
are explicitly `POLICY_PROJECTED` from that common SSE/SZSE projection and are
never labeled provider-direct. The permanent limitations are
`BSE_CALENDAR_POLICY_PROJECTED_FROM_SSE_SZSE` and
`CALENDAR_AUTHORITY_DEGRADED`; status and historical verification retain both.
The route supplies only a code-owned 09:30-15:00 Asia/Shanghai daily Factor
processing envelope and grants no intraday, market-hours, portfolio, broker,
order, trade, or funds authority.

The installed-release capture command is `system calendar-capture`. It makes
exactly one documentation GET plus three `trade_cal` reads, retains no token or
credential-bearing request body, and publishes an owner-only capture root only
after the exact release-install input has replayed against a separately named
clean detached release checkout and all raw bytes, artifacts, directory fsyncs,
and no-follow readbacks pass. The complete thirteen-leaf success root, including
its success marker, is finalized and fsynced in staging before one atomic
no-replace rename makes that terminal outcome visible. The marker binds the
published directory device/inode. A lost rename acknowledgement or one-shot
post-rename readback fault is recovered only by replaying that exact pinned
inode and exact prepared bytes; an ambiguous identity fails closed. Production
assembly copies the exact byte snapshot returned by validation of that same
fixed root rather than reopening its leaves, so a loose, rehomed, or changed
execution/success document is inadmissible. Tushare responses are not signed,
so the receipt proves reviewed installed-code custody and exact bytes rather
than cryptographic remote-peer identity. Project-authored JSON,
weekday/holiday inference, bars, Macro data, and legacy V17 calendars remain
forbidden.

Deterministic request validation occurs before release/install closure: the
cutoff must be in range, capture-root names ending in `.failure` in any case are
reserved, and the requested success/failure pair must both be absent under the
pinned-parent publication lock. Invalid input or reuse therefore performs no
release replay or HTTPS call and consumes no capture name. Publication repeats
the absence check under the same lock to close the post-preflight race.

Every pre-terminal capture failure after release/install closure publishes only
a sanitized terminal artifact in the sibling `<capture-root>.failure`
directory. That
directory is a no-replace, owner-only `0700` root containing exactly one
owner-only `0600` `capture-failure.json`; the artifact binds its directory
device/inode, the requested success-root name, a controlled error code, the
failure time, and the fact that the success-root rename did not complete.
Success and failure publishers arbitrate under the same pinned-parent lock and
both reject an existing terminal peer, so exactly one outcome can ever become
visible for a capture name. Rename-boundary faults are reconciled by comparing
the pinned staging device/inode with the no-follow target entry; an indeterminate
identity fails closed without claiming either publication state. It never
retains exception text, tokens, headers, provider bodies, or credentials. Existence of a
pre-existing directory does not establish that the
success root was published. Failure publication never edits, deletes, repairs,
or completes the requested success root, and neither a failure artifact nor a
failure root is admissible where production requires the exact success marker.
The success/failure root pair is single-use; retry requires a fresh capture-root
name.

An official decoder admission must bind the exchange and evidence role to the
exact HTTPS endpoint/query contract (or, for a notice body, to the exact body
URL carried by a validated notice-index entry), redirect policy, real
status/content type and response headers, retained native body bytes and SHA, capture time,
decoder bytes, and normalized projection SHA. Decoder tests must replay that
retained capture rather than manufacture a body from the expected projection.
The runtime calendar is compiled from admitted official trading-week rules,
continuous-session rules, annual holiday notices, temporary-closure notices,
and session-change notices. Notice discovery is a separate authority: every
exchange has a code-owned exact required-category set, and every admitted index
page must bind its issuer category, page number, inclusive publication window,
and request query in both directions. Complete pagination, the exact category
union, all relevant body captures, a discovery start early enough to retain
prior-year annual notices, and the distinct calendar effective-coverage window
must all close. Per-role counts, individual object sizes, and aggregate replay
bytes are bounded before source bodies are materialized. Until this native
rule/notice closure covers every exchange in the sealed PIT cohort through the
exact cutoff, the `EXCHANGE_OFFICIAL` route remains blocked; it does not block
the separately authorized degraded provider route when every provider gate is
closed.

The currently admitted index contract is deliberately page-number-only and
requires the exact four semantic query values (category, page, publication
start, publication end). It does not claim cursor support. A real issuer index
that uses a cursor, fixed literal parameters, or no server-side date filter
requires a separately reviewed exact contract and retained native fixture.

Any unknown pointer, caller, writer, authority claim, resolver difference,
custody failure, source-closure mismatch, test failure,
deployment-projection mismatch, or expected-pointer mismatch is also a hard
stop before CAS.

Before CAS, a failed deployment step restores the exact previously installed
release and external routing bytes under expected-hash checks. Successful CAS
is the point of no return: no retired runtime, pointer, or release may be
restored afterward. Any post-CAS active verification failure must move only to
the prebuilt `SYSTEM_SUSPENDED` generation. Automation enablement failure keeps
the unified generation active and reports
`SYSTEM_ACTIVE_AUTOMATION_DISABLED`; it does not authorize fallback.

## Fail-closed behavior

- Missing, malformed, non-canonical, wrong-owner, wrong-mode, out-of-root, or
  hash-mismatched artifacts are rejected.
- A missing system active generation reports `UNINITIALIZED` with
  `SYSTEM_ACTIVE_POINTER_ABSENT`. Invalid system closure reports `BLOCKED`.
  Readers never scan for a nearby or newer-looking artifact.
- Mainline reads report `MAINLINE_UNINITIALIZED`, `MAINLINE_BLOCKED`, or
  `MAINLINE_ARGUMENTS_INVALID` as applicable. Unsupported backtesting reports
  `BACKTEST_UNAVAILABLE`. A blocked mainline result contains exactly `status`,
  `active_generation_id`, `mainline_state`, `investment_state`, `blockers`, and
  a null `result`.
- Intelligence-domain outputs are inactive and research-only. They do not call
  a provider, model, broker, order, or trade surface and do not grant
  publication, activation, portfolio, or execution authority.
- Research and factor status commands are diagnostic. They never supply
  publication, factor admission, activation, portfolio, broker, or trade
  authority.
- Historical version-named artifacts may remain immutable evidence, but they
  are not import roots, CLI routes, active pointers, or fallback state.
- Installed Codex skills and automations are external deployment targets. The
  repository copies under `operations/codex/` do not install or activate them.

Stable system failures are explicit: `SYSTEM_CONTRACT_INVALID`,
`SYSTEM_PRECONDITION_FAILED`, `SYSTEM_STORAGE_SECURITY`, `SYSTEM_NOT_FOUND`,
`SYSTEM_IMMUTABLE_CONFLICT`, and `SYSTEM_CAS_MISMATCH` exit with code 2;
`SYSTEM_STORAGE_ERROR` exits with code 3. A CAS failure exposes only expected
and observed pointer SHA values. Failures never authorize fallback or pointer
repair.

## Companion documents

- [Migration behavior matrix](behavior-matrix.md)
- [CLI mapping](cli-mapping.md)
- [Replacement test map](replacement-test-map.md)
- [Canonical replacement-test dispositions](replacement-test-map.json)
- [Stable factor governance](../../factor_governance.md)
- [Codex deployment copies](../../../operations/codex/README.md)
