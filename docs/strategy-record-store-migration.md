# Strategy-record store migration

Status: Phase 1 governance and compatibility cutover. This phase does **not**
move, delete, rewrite, compress, deduplicate, or garbage-collect any live
strategy-record payload.

## Goal and authority model

`results/strategy_records` is an audit store, not an application scratch
directory. The migration puts one registered API between that store and every
reader or writer while keeping the existing payload byte-for-byte available.

The authority chain is:

1. an immutable receipt describes one `market`, `strategy`, and `record_id`;
2. typed artifact references bind exact logical names, media types, byte sizes,
   and SHA-256 values (ledger authority is Parquet canonical);
3. an immutable catalog generation enumerates every registered record and its
   storage state;
4. a small active pointer selects one catalog generation by SHA/CAS;
5. registered projections (current holdings, recent history, Dashboard history)
   are derived from and hash-bound to that catalog;
6. business callers consume the registered store API, never directory order,
   mtime, an absolute legacy path, or a freshly constructed path.

The catalog and active pointer are routing authority. They do not create
investment, execution, broker, publication, or trading authority. Legacy
payload remains audit evidence. A Dashboard bundle, history-integrity registry,
report, spreadsheet, or decision-log event is a consumer or derivative, not an
alternative catalog.

The public store surface is deliberately small:

| API | Authority |
| --- | --- |
| `load_registered_catalog` | Read and verify the pointer-selected immutable catalog; legacy state with no pointer returns `None`, while a broken initialized store fails closed. |
| `catalog_online_record_dirs` | Resolve catalog-declared online legacy record directories; never discover by mtime. |
| `resolve_active_record_dirs` | Resolve the active/previous exact closure for registered holdings readers. |
| `catalog_history_entries` | Return registered recent/history DTOs without a recursive market-root scan. |
| `bootstrap_catalog` | Build the first immutable catalog/pointer from a verified legacy inventory. |
| `publish_catalog` | Publish one immutable catalog generation and CAS the pointer. |

`bootstrap_catalog` and `publish_catalog` are the only public library writer
APIs. Inventory, staging, sealing, no-action receipts, verification, and archive
rehearsal are privileged operations in `scripts/manage_cn_strategy_records.py`,
exposed as the explicit `inventory`, `stage-init`, `seal-publish`, `no-action`,
`verify`, and `archive-rehearsal` subcommands. They are not general library
write authority.

The authoritative registered strategy in Phase 1 is:

```text
market=CN
strategy=aggressive_tech_manufacturing
```

Registration is the single `_record_store/current.v1.json` pointer to an
immutable catalog, not a caller-controlled allowlist. No pointer means an
unbootstrapped legacy store; once `_record_store` exists, a missing or invalid
pointer fails closed rather than falling back. The catalog enumerates identity
exhaustively: an unregistered `(market, strategy)` pair fails closed. The one
compatibility exception is the existing
`US/simulated_portfolio_10000` HistoryLoader path: it may read immediate legacy
run directories only, may not write, archive, publish a catalog, or inherit CN
registration, and must remain covered by a named test. No other strategy gains
access merely because a matching directory exists.

## Live inventory and cutoff

The following inventory was revalidated locally on 2026-08-10 (Asia/Shanghai)
against:

```text
results/strategy_records/CN/aggressive_tech_manufacturing
```

| Class | Live count |
| --- | ---: |
| Immediate children | 192 |
| Strict `YYYYMMDD_HHMM` record directories | 182 |
| Nonstandard directories | 5 |
| Immediate files | 5 |
| Strict records with `record_id < 20260701_0000` | 118 |

The five nonstandard directories are
`20260324_1343_01`, `20260618_broad_factor_validation`,
`20260618_early_acceleration`, `20260618_factor_rethink`, and `_cache`. The five
files are `.DS_Store`, `backfill_audit_20260715_20260807.json`,
`daily_transaction_index_20260715_20260807.csv`, `latest_notes_payload.md`, and
`trading_discipline.md`. The last strict ID before the cutoff is
`20260630_1614`; the first at or after it is `20260701_0946`.

These are a checked baseline, not permission to act. Every bootstrap, verify,
or archive rehearsal must inventory the live root again and fail closed if the
set, classes, counts, file types, or cutoff result differ. The cutoff is an
exclusive lexical comparison of a validated `YYYYMMDD_HHMM` ID, interpreted in
Asia/Shanghai. A file, nonstandard directory, malformed ID, symlink, or other
special filesystem object is never silently included in the 118-record set.

## Exhaustive access matrix

`Legacy policy` is deliberately narrow. “Bootstrap only” means a named adapter
may read existing bytes to build and compare a projection, but normal registered
runtime may not fall back to that scan.

| Caller | Owner | Current read/write | Registered API | Legacy policy | Required test |
| --- | --- | --- | --- | --- | --- |
| `scripts/cn_dashboard_common.py` current/common reader | Dashboard data contract | Direct immediate-child scan; read only | `load_registered_catalog`, active pointer IDs, catalog Dashboard projection, typed source-ref verification | `scan_valid_records` only for unregistered fixtures/bootstrap | registered catalog parity; corrupt ref/pointer fail closed |
| Dashboard historical performance in `cn_dashboard_common.py` | Dashboard data contract | Direct immediate-child scan; read only | catalog Dashboard projection / registered performance history | `scan_historical_performance_records` only for unregistered fixtures/bootstrap | registered history parity; archive state remains readable |
| `scripts/export_cn_aggressive_dashboard_data.py` | Dashboard publisher | Indirect store read; writes only generated Dashboard bundle | common registered bundle API plus source-ref readback | no raw record-root fallback | pre/stage/post readback and old-bundle rollback |
| `scripts/check_cn_dashboard_export.py` | Dashboard verifier | Indirect bundle/source read; no record writes | bundle verifier backed by typed registered refs | none | SHA/path drift rejects bundle |
| `scripts/build_cn_dashboard_history_integrity.py` | Dashboard integrity operator | Direct historical bootstrap scan; writes registry outside live records | `load_dashboard_catalog_projection` with registered catalog validation | raw bootstrap only when unregistered; normal Dashboard never scans archive members | exact record set, catalog SHA, and projection SHA |
| `scripts/build_holdings_fundamental_sheet.py` | Holdings diligence | Direct latest-directory/manual-ledger read; spreadsheet write outside store | `resolve_active_record_dirs`, active closure, declared contained Parquet ref | no CSV fallback; no direct latest-by-name/mtime fallback | Parquet SHA, containment, regular-file, symlink, and active-pointer tests |
| `quant_investor.automation.history_loader.HistoryLoader` | Daily automation runtime | Recursive market-root read | `load_registered_catalog`, `catalog_history_entries` | read-only immediate-run compatibility solely for registered exception `US/simulated_portfolio_10000` | CN registered history identity and named US compatibility test |
| `quant_investor.automation.daily_runner` and top-level runner | Daily automation runtime | Supplies market-root/config and consumes history; no strategy-record writer | passes explicit `history_strategy`; consumes HistoryLoader DTOs | no implicit CN strategy or direct root construction | dry-run/report-only/full path use identical explicit strategy |
| `quant_investor.automation.report_builder` | Daily report owner | Reads loader DTO; writes report outside store | DTO fields `record_id`, `storage_state`, `evidence_status` | prose may name the store; no filesystem access | report provenance contains DTO identity, not reconstructed path |
| `daily_config.py` | Daily automation configuration owner | Configuration only | explicit `history_strategy=aggressive_tech_manufacturing` | missing/unknown registration blocks | config validation rejects missing or unknown strategy |
| `scripts/manage_cn_strategy_records.py` | Record-store operator | New bootstrap/stage/publish/verify/rehearsal CLI | privileged `command_*`/`build_inventory` operations; catalog mutation only through `bootstrap_catalog`/`publish_catalog` | bootstrap reads legacy; stage/archive temp I/O is explicit; no source-record mutation | explicit subcommand, CAS, budgets, malicious member/path rejection, zero source-payload mutation |
| `scripts/validate_cn_aggressive_factor_universe.py` | Factor research | Currently writes a nonstandard directory under the production root | writes under `results/research/CN/aggressive_tech_manufacturing/...` | existing legacy directory is classified only; never adopted as a run | default output is outside strategy-record root |
| `scripts/log_decision.py` metadata | Decision audit | Append-only decision log may contain absolute legacy `run_dir` metadata | new events store `record_id` plus typed `artifact_ref`; resolver handles location | never rewrite old JSONL; old absolute path is provenance only | new event has no authoritative absolute run path; old event remains byte-identical |
| automation `automation` | External automation owner | Legacy prompt/config may directly create record directories and files | only registered receipt transaction/write API | old config retained by SHA for rollback; direct writes forbidden after cutover | captured old/new config SHA; transaction and no-action budget tests |
| automation `cn-dashboard` | External automation owner | Reads live records and may run Dashboard builders | registered Dashboard/catalog API | bootstrap scanner only in explicit integrity build | registered projection parity and source-ref verification |
| automation `myquant-cn` | External automation owner | Legacy prompt/config may read/write CN records | registered catalog/history and receipt transaction APIs | old config retained by SHA for rollback | captured old/new config SHA; unknown registration blocks |
| automation `a-2` | External automation owner | Legacy prompt/config may read/write/report from records | registered catalog/history and receipt transaction APIs | old config retained by SHA for rollback | captured old/new config SHA; fault injection leaves old pointer active |

External automation configuration is outside repository source authority. The
four names above are the complete Phase 1 automation registration set. Cutover
must capture each old config SHA and proposed new config SHA; a missing,
renamed, unreadable, or concurrently changed config is a blocker. Repository
code must not infer that an automation was migrated from documentation alone.

## Writer contract and budgets

Callers must use the manager's explicit `stage-init` then `seal-publish`
transaction; no-action writers use its `no-action` subcommand. They must not
call `mkdir`,
`open`, `write_*`, `rename`, `replace`, or `copy*` against the strategy-record
root. A no-action/carry-forward outcome is still a first-class receipt: it binds
the prior effective financial state and reason codes without copying the prior
ledger or embedding a full evidence bundle.

For archive-aware catalogs, the pointer/catalog `generation_id` identifies the
current immutable publication and therefore advances for each no-action
receipt. The hash-bound history registry's `intended_generation_id` identifies
the generation where the registered historical projection last changed; it is
intentionally inherited across receipt-only publications. Dashboard readers
validate both identities separately and continue to require exact registry
bytes, embedded-body equality, projection SHA, record/ref/inventory closure,
and archive bindings.

New writes fail before publication if any budget is exceeded:

| Budget | Limit |
| --- | ---: |
| no-action receipt canonical bytes | 64 KiB |
| one new-run logical file | 8 MiB |
| all new-run logical files | 16 MiB |
| logical file count per new run | 128 |
| canonical catalog JSON | 32 MiB |

Legacy bootstrap is not retroactively subjected to new-run budgets; it only
classifies and hashes existing payload. Budget exemption never permits a legacy
payload rewrite or lets a new writer use the legacy format.

## Transaction, failure, orphan, and rollback rules

Publication is a same-filesystem staged transaction:

1. validate registration, record identity, typed refs, regular-file and path
   containment rules, budgets, and expected old pointer;
2. write a new private stage and fsync every file plus its directory;
3. re-read canonical bytes, file counts, sizes, schemas, hashes, and closure;
4. publish immutable objects/receipt/catalog generation without overwriting an
   existing identity;
5. compare-and-swap the small active pointer from the captured old SHA to the
   new catalog SHA;
6. re-read the pointer and full registered closure before reporting success.

Fault tests must stop after each boundary (partial stage write, completed stage,
immutable generation publish, before pointer CAS, and after pointer CAS/readback
failure). Before the CAS, readers keep the old catalog. After an uncertain CAS,
success is not reported until readback proves one complete closure. A collision,
short write, SHA drift, schema failure, budget failure, stale expected pointer,
or fsync/rename error fails closed.

Any unreferenced immutable stage, generation, or final payload is an orphan. It
is retained and listed by `verify`; it is never automatically garbage-collected,
rewritten, reused as authority, or deleted. Cleanup requires a separate,
explicit maintenance authorization.

Rollback changes routing, not evidence. It may only:

- CAS the active pointer back to the exact old catalog SHA;
- revert the scoped code hunk;
- restore the exact old config SHA for each of `automation`, `cn-dashboard`,
  `myquant-cn`, and `a-2`.

Rollback must not delete, move, edit, or hide live payload, stages, immutable
generations, final payload, or orphans.

## Archive rehearsal boundary

Phase 1 may produce a read-only archive plan/rehearsal for the currently
eligible 118 strict records. It may not move or delete them. The rehearsal must
reject path traversal, absolute members, duplicate normalized names, symlinks,
hardlinks, devices/FIFOs/sockets, escaping refs, case-fold collisions, and bytes
or hashes that differ from the registered source closure.

Per monthly rehearsal archive:

- at most 10,000 members;
- at most 2 GiB expanded logical bytes;
- free space before staging at least `2 * source logical bytes + 256 MiB`.

The rehearsal must restore into a fresh temporary root, verify every member,
receipt, typed ref, catalog binding, byte count, and SHA, and then prove reader
parity. Archive success is still not authority to delete the hot copy.

## Static access gate and stop condition

`scripts/check_strategy_record_access.py` scans Git-tracked Python, shell, and
JavaScript source for target literals and direct traversal/write operations.
Tests, docs, generated fixtures, vendored code, and live results are outside its
production-source scan. Every finding must match the reviewed allow table with
a narrow reason; there are zero unexplained findings and no wildcard permission
for arbitrary callers.

Stop the migration without pointer/config mutation when any of these occurs:

- live inventory differs from the revalidated baseline and has not received a
  new owner-approved baseline;
- a caller, strategy, automation, or direct access is unregistered;
- old/new config SHA cannot be captured for all four automations;
- Dashboard, holdings sheet, or HistoryLoader parity differs;
- catalog, receipt, object, projection, or pointer closure is incomplete;
- a budget, archive-safety, free-space, restore, or fault-injection test fails;
- any proposed step would move, delete, or rewrite a live payload.

## Phase 2 archive authority and quarantine cutover

Phase 2 is a separately gated location migration for the exact 118 strict
records with `record_id < 20260701_0000`. It does not change investment,
holdings, accounting, Factor, Mainline, broker, order, execution, or trading
authority. It does not delete a strategy record. The terminal physical state is
an archive-aware catalog plus a recoverable same-device quarantine.

The immutable authority graph is deliberately acyclic:

1. each final monthly archive manifest binds the frozen pre-cutover ONLINE
   catalog, the final archive bytes, and every record inventory and logical
   source reference;
2. its restore receipt binds that manifest and archive plus one fresh, bounded,
   full restore result;
3. history-integrity v2 binds the intended generation ID, Dashboard projection
   content SHA, exact logical references, and exact archive manifest/receipt
   references, but never the candidate catalog SHA;
4. the candidate catalog binds the history registry and archive objects;
5. the current pointer binds the candidate catalog;
6. the final transaction seal binds all already-created objects, move receipts,
   configuration hashes, and the source reconstruction receipt.

Final archive locators are project-relative and may resolve only below:

```text
results/strategy_record_archives/CN/aggressive_tech_manufacturing/
  monthly/v1/<YYYY-MM>/
```

An archived row retains its original `relative_path`, complete inventory,
`file_count`, `total_bytes`, logical evidence closure, and normalized Dashboard
projection. `state` and `storage_state` must be identical and one of exactly:

- `ONLINE`;
- `ARCHIVED`;
- `NONSTANDARD_RESEARCH_OUTPUT`;
- `AUXILIARY_ROOT_FILE`.

An ONLINE row may not carry an authoritative archive locator. An ARCHIVED row
must carry the complete archive, manifest, restore-receipt, and member-prefix
locator. Current and previous must be distinct ONLINE records. Ordinary readers
do not decompress archives: they consume the catalog's complete normalized
projection while independently validating the online archive storage closure.
Deep audit and restore alone extract into a fresh bounded temporary directory.

The quarantine namespace is outside the hot record root:

```text
results/strategy_record_quarantine/CN/aggressive_tech_manufacturing/
  <transaction_id>/records/<record_id>
```

The transaction journal is immutable under
`_record_store/quarantine_transactions/<transaction_id>/`. The record root and
quarantine parent must have the same device ID, and the transaction target must
be new and empty. Every manager mutation command acquires
`_record_store/.operation.v2.lock`. Each move is ordered as durable intent,
under-lock `lstat` inventory verification, `os.rename`, `fsync` of both parent
directories, exact target inventory verification, and durable completion
receipt. Source-directory `unlink`, `rmtree`, `shutil.move`, copy fallback, and
`os.replace` are forbidden.

The filesystem facts determine recovery:

| Source | Target | Meaning and action |
| --- | --- | --- |
| present | absent | verify and perform the planned rename |
| absent | present | verify and complete or resume the receipt |
| present | present | conflict; stop without another move |
| absent | absent | lost; stop without another move |

Before pointer CAS, candidate artifacts have no authority. After candidate CAS
but before the first move, pointer-only rollback is allowed only after exact
revalidation. After the first move, the archive-aware pointer remains
authoritative. An ONLINE rollback may be published only after every moved
directory has been atomically restored and all 118 original inventories have
been reverified. Partial migration is a resumable state, never a reason to hide
missing ONLINE evidence.

### Two independent gates

The implementation gate requires all focused archive/store/Dashboard/history
tests, malformed-member and locator security tests, crash/fault recovery tests,
full unit/flake8/mypy/Node checks, real-artifact normalized shadow parity, final
archive full restore, and a binary-safe source reconstruction rehearsal from
the recorded base HEAD. A skipped required check fails the gate.

The physical-move gate then revalidates the frozen predecessor pointer/catalog,
the exact 118-record set, all archive/manifest/receipt hashes, four automation
definition hashes, protected data pointers, the candidate CAS, post-CAS readers
while hot directories remain online, same-device placement, and the empty
quarantine target. Any strict `YYYYMMDD_HHMM` child absent from the registered
catalog blocks candidate generation, even when it is newer than the archive
cutoff; the owner must first classify it through the governed writer path.
Any drift stops before candidate publication and before the first rename.

Successful completion requires exactly `64 ONLINE + 118 ARCHIVED + 5
NONSTANDARD_RESEARCH_OUTPUT + 5 AUXILIARY_ROOT_FILE = 192`, with the 118 hot
paths absent and the 118 quarantine paths present exactly once. Dashboard
positions, current/previous, accounting, history point order and values,
funding-aware returns, drawdowns, rejected IDs/reasons, benchmark, risk-free
inputs, authority flags, and evidence status remain exact. Only enumerated
generation, pointer, registry, archive-storage, generated timestamp, and
top-level content-hash provenance may differ.

Quarantine deletion is not part of Phase 2. Because quarantine is on the same
filesystem, the cutover reduces the production directory footprint but does
not release disk space. Deletion may only be proposed in a future, separately
authorized maintenance after at least one real scheduled automation soak and a
new full restore verification.

### 2026-08-10 completed cutover

Transaction `cn-aggressive-archive-cutover-20260810-v2` published catalog
generation `cn-aggressive-archive-cutover-20260810-v2` and moved exactly 118
pre-July records into the recoverable quarantine. Its terminal verification is
`source_records=0`, `quarantine_records=118`, `dual_records=0`, and
`lost_records=0`. The catalog contains 118 ARCHIVED and 66 ONLINE records;
active/previous are `20260810_1948` / `20260810_1943`.

Catalog-v2 writers extend the existing normalized projection one validated
record at a time and publish a generation-bound history registry. They never
rescan missing archived hot directories. The legacy default history-registry
CLI path resolves to the immutable catalog binding; an explicitly different
path still fails closed. Rollback clears candidate registry inheritance when
returning to an all-ONLINE v1 catalog. Quarantine remains retained and must not
be deleted before the separately authorized soak and restore gate.
