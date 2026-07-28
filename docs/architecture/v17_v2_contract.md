# v17 Lifecycle Protocol v2 Contract

## Status

This document defines the Phase 0 contract package in
`quant_investor.v17_v2_contract`.

- Package version: `17.0.0`.
- Lifecycle protocol: `myquant.v17.v2`.
- Production/default protocol: v15, unchanged.
- Runtime authority: `false`.
- Broker, order, trade, schedule, Dashboard, release, cutover, and purge
  authority: none.

The sibling package is deliberately outside `quant_investor.v17`. Importing it
must not execute or import the legacy v17 prototype runtime.

Artifact and schema identifiers are separate namespaces. An artifact envelope
uses `myquant.v17.v2.<artifact>.v1`, while the JSON Schema that validates that
envelope uses `myquant.v17.v2.<artifact>.schema.v1`. Therefore a schema's
`$id` uses the latter form and its `properties.version.const` uses the former.
Schema registries and external schema references bind the schema identifier;
artifact/resource references bind the artifact version.

`package_manifest.v1.json` binds every other protocol-v2 JSON resource and
schema plus the independently frozen hashes of the 19 legacy prototypes. It
cannot hash itself without creating a circular document. The package's
`resources.py` therefore freezes the manifest's byte SHA-256 as well as every
asset listed by the manifest; tests require the two inventories to be exact
and source/sdist/wheel/non-editable-install bytes to agree.

`package_parity.py` is the offline verifier for already-built artifacts. Its
archive/path/tamper behavior is unit-tested with synthetic surfaces; the real
wheel, sdist, and fresh non-editable installation are checked as a separate
owner-private acceptance-evidence step so ordinary unit tests never build,
install, resolve, or download dependencies.

## Mandatory validation sequence

Every protocol-v2 acceptance path has one fixed order:

1. parse stored bytes with `load_canonical_resource`, which requires strict
   UTF-8, compact sorted-key JSON, and exactly one trailing newline;
2. execute the artifact's sole hash-bound packaged Draft 2020-12 schema;
3. execute the artifact-family cross-document validator;
4. compare the caller's expected ledger/pointer byte SHA under the storage
   layer's locked CAS;
5. write only the action-matrix paths enumerated for the accepted cell.

`schema_validation.py` implements the closed Draft 2020-12 keyword subset used
by all 15 schemas using only the standard library. Schema preflight rejects an
unknown keyword or format, so a future schema cannot silently exceed the
executor. All public cross-document validators call the packaged schema gate
before their relationship checks. `validate_canonical_contract_bytes` is the
bytes-first dispatcher for a stored document and its selected cross-document
validator.
The callback is mandatory: omitting it or passing `None` rejects acceptance.
`validate_canonical_schema_bytes` is a separately named inspection helper and
explicitly grants no acceptance or publication authority.

Phase 0 implements only steps 1–3. It has no CAS, storage, publication, or
runtime authority, so steps 4–5 remain prohibited until a later reviewed
runtime tranche.

## Physical namespace

Protocol v2 owns only these paths:

```text
data/private/v17_sources/protocol-v2/
  objects/
  manifests/
  locators/

results/v17_shadow/protocol-v2/
  runs/
  models/objects/
  outcomes/
  _latest/
    shadow.json
    .latest.lock
```

The current flat v17 paths and package resources are frozen legacy prototypes.
Protocol v2 cannot read them as active authority, write them, fall back to
them, or reuse their identifiers.

`.latest.lock` is a coordination sidecar only. It carries no business data or
authority and may be used solely to serialize checked publication or explicit
repair of `shadow.json`; no other `_latest/` sidecar is owned or writable.

Named-object creation must classify both v1 and v2 inventories before import,
directory creation, locking, input-file reads, receipts, or CAS. Exact and
ASCII-casefold-equivalent entries collide regardless of whether the entry is a
regular file, directory, symlink, broken symlink, or special node. An ancestor
symlink also rejects the operation. Content-addressed objects are the only
reuse exception, and only an exact single-link `0600` regular file with the
derived path, byte count, digest, and metadata may be reused.

Private content objects use only `.json`, `.parquet`, or `.blob`. The `.bin`
suffix is rejected consistently by derivation helpers, schemas, and
cross-document path validation.

## Action compatibility matrix

The canonical action matrix evaluates the full Cartesian domain:

- five detected versions: `ABSENT`, `v1`, `v2`, `unknown`, `malformed`;
- eight actions: source maintenance, risk-policy seal, prepare, receive,
  finalize, status read, artifact read, and latest repair;
- thirteen entry states: missing, unknown, malformed, five nonterminal states,
  and five immutable terminal states;
- three checkpoints: `PRE_IMPORT`, `ACCEPTED`, and `INITIALIZED`.

The 1,560 cells must each match exactly one ordered rule. A cell records command
commit, business acceptance, allowed write namespaces, allowed target outcomes,
exit behavior, retry/CAS behavior, and latest effect.

The checkpoints mean:

1. `PRE_IMPORT` permits only bounded protocol/state header detection and
   inventory classification.
2. `ACCEPTED` permits complete protocol-v2 validation but no writes.
3. `INITIALIZED` is reached only after input, source, binding, namespace, and
   locked-CAS checks pass. Only then may the decision's enumerated paths be
   written.

v1 terminal artifacts permit only status and artifact reads. Every v1
nonterminal, v1 mutation, unknown/malformed envelope, cross-protocol collision,
wrong-state action, and pre-initialization CAS failure exits 2 with zero writes.
A durably committed hard stop also exits 2 and reports
`durably_committed=true`; the three business terminals and idempotent successful
reads exit 0.

## Lifecycle and immutable terminals

The nonterminal states are:

```text
PREPARED
DETERMINISTIC_COMPLETE
DEEP_REQUEST_READY
DEEP_RESPONSE_RECEIVED
PORTFOLIO_COMPLETE
```

The immutable terminal states are:

```text
SHADOW_COMPLETE_AWAITING_HUMAN_DECISION
SHADOW_RANK_COMPLETE_NO_PORTFOLIO
SHADOW_PORTFOLIO_INFEASIBLE
HARD_STOP_SNAPSHOT_DRIFT
HARD_STOP_INVALID_EVIDENCE
```

Every ledger mutation requires the expected ledger byte SHA-256. CAS conflicts
are zero-write. Failed work cannot silently become a published artifact.
Terminal-ledger commit and latest publication are separate durable operations;
a terminal with an unpublished pointer requires explicit latest repair.

Every ledger snapshot binds the exact package-manifest SHA, the complete
manifest-derived resource/schema inventories, and the byte SHA of the exact
package-owned Python module inventory. Missing, extra, reordered, or altered
bindings are invalid.

`validate_shadow_ledger` validates one snapshot, including its packaged
bindings and internal history, but by itself makes no historical-integrity
claim. `validate_shadow_ledger_successor` hashes the supplied stored predecessor
bytes for one CAS transition. `validate_shadow_ledger_chain` requires the full
canonical byte sequence beginning at sequence zero, rehashes every predecessor,
and verifies immutable fields, history prefixes, artifact retention, and state
boundaries. Terminal publication must pass the full-chain validator. A latest
repair with a non-`EMPTY` predecessor also receives and hashes the previous
pointer bytes; a declared `EMPTY` predecessor forbids supplied bytes.

Terminal validation accepts the exact stored ledger and output bytes, never
caller mappings that it could reserialize on their behalf. It also consumes
the complete byte-exact admitted source DAG, requires the ledger locator and
nonempty input-binding inventory to equal the admission result, and rejects a
ledger artifact that points back to the ledger, output, or latest state
carriers. Output generation and latest publication must be nondecreasing in
absolute UTC time after the ledger and any predecessor pointer.

## Source authority and hash DAG

Phase 0 freezes the rank-stage roles that can be established without guessing:

```text
market_pointer
market_snapshot_manifest
market_bars_dataset
cn_open_day_calendar_dataset
pit_generation_catalog
fundamental_generation_catalog
fundamental_raw_tables_dataset
H00300_total_return_dataset
corporate_actions_dataset
official_delisting_cash_dataset
deep_evidence_dataset
```

`market_pointer` and `market_snapshot_manifest` have frozen role, kind, phase,
and failure disposition, but their payload schemas remain pending. The
portfolio role inventory, overlay payload contracts, risk-policy snapshot
schema, and dataset-record schema registry are also pending. Therefore
`source_role_matrix.v1.json` is intentionally:

```text
completeness = PARTIAL
runtime_usable = false
authority = false
```

Verification-receipt roles are forbidden as caller-provided authority.
Macro/Markov mappings are required only when the corresponding controller is
enabled. An enabled portfolio input that is unavailable must lead to
`SHADOW_RANK_COMPLETE_NO_PORTFOLIO`; an explicitly disabled controller does not
require a mapping.

The immutable hash DAG is:

```text
raw objects / dataset shards / availability
  -> dataset manifests
  -> source manifest
  -> catalogs and observation-disposition summaries
  -> source binding set
  -> source locator
```

The locator is written last and cannot be updated. Duplicate
`(stage, disposition_id)` entries are rejected. Disposition ordering includes
the summary SHA/path and dataset-manifest SHA/path, with canonical JSON bytes as
the final tie-break. A summary's embedded dataset reference must match the
bound reference field by field.
Every dataset shard binds the root logical schema with
`SHA256(canonical_json({"version":"myquant.v17.v2.dataset-schema-digest.v1",
"schema":<root schema>}))`; a merely well-formed but different digest is
rejected.
Dataset summaries use the hash-bound
`myquant.v17.v2.dataset-summary.schema.v1`; an untyped summary cannot enter
runtime admission. A summary's `row_count` must exactly equal its resolved
dataset manifest's `total_row_count`.

`validate_source_hash_dag` is the structural maintenance validator; it does not
authorize runtime use and may validate a deliberately partial closure.
Runtime admission is the separate
`admit_runtime_source_hash_dag -> SourceAdmissionOutcome` gate, not a boolean
mode on the structural validator. It first requires the exact hash-bound
approved role-matrix bytes, `COMPLETE`, `runtime_usable=true`, no pending
registry entries, and only frozen schemas. It then requires the exact canonical
stored bytes for every DAG document, a one-to-one required-role inventory,
dataset role equality across manifest/catalog/binding, declared phase equality,
and dispatch through each role's declared packaged schema. Every raw source
object, including dataset shards, must be supplied as its exact stored bytes;
runtime admission never canonicalizes a caller mapping on its behalf.
Generation catalogs are role-bearing `OBJECT` carriers with required root
`role` and `phase` fields, exactly one catalog per declared catalog role, and
an AVAILABLE provenance row in the source manifest. Other available `OBJECT`
documents must themselves carry the exact declared role and phase. Missing,
detached, shared, duplicate, cross-kind, role-substituted, phase-substituted,
or schema-substituted evidence is invalid.

Admission precedence is fixed: an unavailable required role with
`REJECT_BEFORE_INITIALIZED_ZERO_WRITE` raises a zero-write validation failure;
otherwise an unavailable required portfolio role returns the typed business
outcome `SHADOW_RANK_COMPLETE_NO_PORTFOLIO`; only a completely available
required closure returns `ADMITTED`. The currently packaged matrix is
`PARTIAL`, so the public runtime-admission and terminal-publication paths always
fail closed in Phase 0.

## Canonical JSON, identity, and limits

Stored JSON is compact, sorted-key UTF-8 plus one trailing newline. Semantic
SHA-256 removes only the root `semantic_sha256` field; nested digests remain
part of the parent hash. Duplicate keys, BOM, NaN, Infinity, silent Unicode
normalization, and noncanonical identifiers are rejected.
RFC3339 values are compared as timezone-aware UTC instants rather than as
strings. The source manifest cannot predate cutoff, catalogs cannot predate
their manifest, the locator cannot predate its DAG, and a ledger cannot predate
its admitted locator.

Opaque identifiers, path-component identifiers, and security codes use the
frozen validators in the contract package. Security codes have exactly the
wire form `NNNNNN.SZ`, `NNNNNN.SH`, or `NNNNNN.BJ`; exchange membership is
validated from sealed PIT evidence, never inferred from the first digit.

All numeric limits are inclusive. Integer limit inputs require
`type(value) is int`, so booleans, floats, and numeric strings are rejected.
Aggregate checks are performed before addition. The single machine-readable
limits resource controls:

- ordinary JSON 16 MiB and the fixed fundamental-generation-manifest profile
  128 MiB;
- depth 64, container size 65,536, total nodes 1,000,000, string size 1 MiB,
  key size 1,024 bytes, and integer-token length 64 digits;
- 256 sources; 1,024 candidates, deep reviews, and distinct evidence IDs;
- 10,000 universe symbols; 2,898 open days per symbol; 100,000 observations per
  calibration cell; 65,536 Parquet rows per batch;
- 128 ledger artifact roles and 64 history entries;
- 4,096 dataset shards, 8 GiB per shard, 512 GiB aggregate bytes, and
  100,000,000 aggregate rows.

The legacy dataset-manifest v1 envelope remains byte-identical and retains its
100,000-shard behavior. Only the independent protocol-v2 wrapper applies the
4,096-shard bound.

## Deep-research binding

A deep-research report binds the request reference, run, cutoff, security code,
template resource SHA, and every evidence identifier. It preserves the five
fact-to-judgment layers and the required coverage sections. The six signals are
complete and limited to `-1`, `-0.5`, `0`, `0.5`, or `1`. A triggered severe
red flag requires sealed evidence. Narrative arrays remain order-sensitive;
set-like reference arrays are canonicalized explicitly.

Codex remains Fundamental-only and advisory. It cannot add securities or data
sources, change prices or peers, alter base eligibility, override risk caps, or
set final weights.

## Historical Phase 0 v1 evidence design (superseded)

The remainder of this section records the pre-review v1 evidence design only.
It is not an executable or acceptance contract. The authoritative Phase 0
protocol is the single-session v2 design in the next section.

Phase 0 closes only when all of the following hold:

- the diff stays inside the approved contract-only allowlist;
- all 19 legacy resource/schema files retain their independently frozen byte
  SHA-256 values;
- every new resource and schema has a unique casefolded identifier and exact
  package hash;
- source, sdist, wheel, and a non-editable installation contain identical
  normalized package paths and bytes;
- focused, staged, and full offline test suites pass without new skips/xfails;
- a never-before-used CPython 3.13 environment completes the unchanged locked
  offline sync;
- the transcript, receipt, source-state seal, build/test logs, and package
  payload hashes are bound by an owner-private evidence index.

The Phase 0 evidence index must also contain a contract-only allowlist receipt:
base commit, complete worktree porcelain/diff seal, every allowed Phase 0 path,
every pre-existing non-Phase-0 path, and the final source-state SHA. This
distinguishes the sibling contract tranche from the worktree's earlier legacy
retirement/runtime edits without rewriting or claiming ownership of them.
The required gate manifest is a separate owner-private `0600` canonical JSON
input. It is accepted only when it binds exactly one fixed-role record for each
of `native_sync_receipt`, `native_sync_log`, `v2_evidence_tests`,
`recommended_core_tests`, `full_offline_suite`, `mypy`, `black`, `diff_check`,
`package_parity`, and `hash_freeze_readback`. Each record uses the canonical
role as its ID, the fixed artifact/log kind, and the same final source binding:
`base_commit`, `source_state_sha256`, `porcelain_sha256`,
`binary_diff_sha256`, and `untracked_inventory_sha256`. Missing, duplicate,
unknown, stale, or source-mismatched gates fail closed.

The index does not trust a generic pass/fail status. It reopens every raw gate
file and checks role-specific semantics. The native dependency receipt must be
the real canonical output from `scripts/v17_offline_dependency_evidence.py`,
with its complete exact top-level and nested shape: scope, inputs, runtime uv,
target venv, Python/platform, expected and installed reconciliation,
artifact records/counts, missing/invalid lists, acceptance flags, failure
reasons, and semantic SHA. Extra or omitted fields fail closed. The index
requires an accepted native dependency environment, exact installed
reconciliation, `PASSED` operator-asserted native uv sync, offline/no-network
flags, and CPython 3.13. It recomputes the producer's exact HEAD/diff source
receipt and binds the same base/head, porcelain, and untracked bytes as the
index source. The native log must separately
prove the exact `uv sync --python <base CPython 3.13> --locked --all-extras
--offline` exit-0 command and bind `UV_PROJECT_ENVIRONMENT` to the receipt's
fresh target venv. The native `--python` executable may be a distinct path, but
its resolved path must equal the receipt's target Python. The receipt target
Python must be `<fresh venv>/bin/python`; all later Python-based log receipts
must use that same fresh interpreter. The native log's resolved uv executable,
uv version, and cache path must equal `runtime.uv` and `inputs.uv_cache`; the
index rereads the actual uv executable and verifies its resolved path, byte
SHA-256, size, mode, and executable bit. Pytest, mypy, and Black command
versions must equal their corresponding versions in the receipt's exact
installed-distribution inventory, including the same CPython version.
Pytest gates must parse the final pytest summary and match the claims for
passed, skipped, failed, errors, xfailed, and xpassed; only the full offline
suite may carry explicit skips. The full offline suite command must include
`-rs`; its claims must bind `raw_output_sha256` and a structured
`skip_allowlist`, and every allowlist entry must match a raw pytest
`SKIPPED [n] path:line: reason` line with the summed counts equal to the final
skipped total. Changed skip paths or reasons fail closed even when the total
skip count is unchanged, and xfail/xpass must remain zero. The
`recommended_core_tests` role also carries the repository staged-upgrade gate
without adding an eleventh role: its raw log must contain the staged
focused-tests marker, staged focused-mypy marker, mypy success text, and exactly
one `staged_upgrade_exit_code=0` marker, with the claim
`staged_upgrade_exit_code=0`; the index JSON Schema also fixes that claim to
integer zero. Mypy must prove `Success: no issues found`;
Black must prove unchanged with no `would reformat`; diff-check raw output must
be empty. Package parity must be the sealed
`myquant.v17.v2.phase0-package-parity-evidence.v1` report from
`scripts/v17_phase0_package_evidence.py`, not the parity helper's shorter JSON.
The index binds its exact envelope, current/before/after source bindings,
source-binding input artifact, CPython 3.13 and uv binary readbacks, frozen uv
0.10.9 and pip 25.2 identities, the exact five-package Hatchling backend
inventory, and fresh build/install environments outside the repository. It
also checks all 13 commands in order with exact derived argv, environment,
repo cwd, zero integer exit, tool identity, sanitized host environment, bound
stdout/stderr bytes, and combined command SHA. The wheel must be built from the
bound sdist, and the exact bound wheel must be installed with
`--no-deps --no-index --no-compile`. Artifact paths, SHAs, sizes, four-surface
inventory parity, non-editable installed provenance, `quant-investor` `17.0.0`
metadata, RECORD counts, and dist-info hashes are cross-checked. The complete
package report is independently executable under the closed stdlib Draft
2020-12 schema at
`scripts/schemas/v17_phase0_package_evidence.v1.schema.json`; schema integer
fields reject booleans. Its `$id` is the separate
`myquant.v17.v2.phase0-package-parity-evidence.schema.v1` namespace while the
report `version` remains the artifact ID. The index loads, preflights, and
executes this checked-in schema before its cross-field semantic validator.
Hash-freeze readback must match the current
`validators.py`, `resources.py`, `generation_catalog.v1.schema.json`, and
`package_manifest.v1.json` bytes plus the same final source binding. The root
index reports `accepted=true` and `status=SEALED` only after all gates pass.
Every log role is a single-file command envelope: the first line is exactly
`MYQUANT_PHASE0_COMMAND_RECEIPT=<compact canonical JSON>`, and the remaining
bytes are the combined command output whose SHA-256 and size are bound by that
receipt. The command receipt version is
`myquant.v17.v2.phase0-command-receipt.v1`; it binds role, repo cwd, current
source binding, exact argv/env/exit/tool-version for each command, output
SHA/size, and its own semantic SHA. Every command-receipt or claim count, size,
and exit-code field must be a JSON integer; boolean `false` is not zero. Bare
success-shaped logs are invalid. The complete built report is preflighted and
validated against the index's Draft 2020-12 schema with the repository's
closed, stdlib-only schema executor.

The repository-owned index is an independent validator; it is not the command
producer for the seven log roles. Until a separately reviewed producer exists,
those logs may only be generated by one fixed-role, owner-private external
harness. Supplemental audit material must preserve the exact harness byte
SHA-256 and its invocation record. Handwritten receipt JSON and arbitrary-argv
runners are not acceptable. This supplemental producer provenance is not part
of command-receipt v1 and is not schema-bound by the index, so it remains an
explicit Phase 0 residual boundary; it does not add a gate or change the
receipt shape.

`scripts/v17_phase0_evidence_index.py` writes the index and its byte-SHA
sidecar exact-once into one owner-private 0700 directory. Publication holds a
non-following directory descriptor, brackets the link commit with source and
external-evidence readback, and rolls back only the exact staged inodes on
drift.

Any pending source-role registry item keeps runtime authority false. Phase 1
runtime/CLI work requires a new Architect review followed by a Critic review.
Phase 0 does not authorize runtime/CLI/schedule/Dashboard wiring, main or
release cutover, non-Phase-0 staging, or v16 result-root purge.

## Authoritative Phase 0 v2 evidence session

Phase 0 evidence is produced by one repository-owned, stdlib-only session
runner. It is invoked by the frozen CPython 3.13.7 binary with `-I -S -B`,
accepts no arbitrary command or environment override, and executes the complete
ten-role DAG once. It is deliberately not resumable. A failed session is
`UNPUBLISHED`; retry requires new bundle and work roots.

The bundle root and work root are distinct, non-nested, owner-owned `0700`
directories outside the repository and the protected v16 roots. The bundle
contains only these fixed evidence files, in this order:

```text
00_session.json
10_native_sync.log
20_native_dependency.json
30_v2_tests.log
31_recommended_core.log
32_full_suite.log
33_mypy.log
34_black.log
35_diff_check.log
40_package_parity.json
50_hash_freeze.json
60_gate_manifest.json
70_evidence_index.json
70_evidence_index.json.sha256
99_unpublished_failure.json  # failure only
```

The work root contains the fresh native environment, package build/install
environments, sdist and wheel work, the linked-worktree alternate Git index,
and bounded runtime state. Both roots are retained on success and failure.
Only a valid final index plus its byte-SHA sidecar means `SEALED`; partial
files never grant acceptance.

Breaking pre-seal evidence shapes use new identifiers and reject every old
shape without fallback:

| Artifact | Artifact version | Schema version |
|---|---|---|
| dirty classification | `myquant.v17.v2.phase0-pre-existing-classification.v2` | `myquant.v17.v2.phase0-pre-existing-classification.schema.v2` |
| session | `myquant.v17.v2.phase0-session.v1` | `myquant.v17.v2.phase0-session.schema.v1` |
| command receipt | `myquant.v17.v2.phase0-command-receipt.v2` | `myquant.v17.v2.phase0-command-receipt.schema.v2` |
| native dependency | `v17_third_party_dependency_environment_evidence.v2` | `myquant.v17.v2.third-party-dependency-environment-evidence.schema.v2` |
| skip baseline | `myquant.v17.v2.phase0-skip-baseline.v1` | `myquant.v17.v2.phase0-skip-baseline.schema.v1` |
| package parity | `myquant.v17.v2.phase0-package-parity-evidence.v2` | `myquant.v17.v2.phase0-package-parity-evidence.schema.v2` |
| hash freeze | `myquant.v17.v2.phase0-hash-freeze.v2` | `myquant.v17.v2.phase0-hash-freeze.schema.v2` |
| gate manifest | `myquant.v17.v2.phase0-gate-manifest.v2` | `myquant.v17.v2.phase0-gate-manifest.schema.v2` |
| evidence index | `myquant.v17.v2.phase0-evidence-index.v2` | `myquant.v17.v2.phase0-evidence-index.schema.v2` |
| failure receipt | `myquant.v17.v2.phase0-unpublished-failure.v1` | `myquant.v17.v2.phase0-unpublished-failure.schema.v1` |

Every schema is a self-contained closed Draft 2020-12 document. Artifact and
schema identifiers remain separate. The production path stable-reads each
schema without following symlinks, binds its raw SHA, preflights the supported
keyword set, runs instance validation, and then runs the exact-key semantic
validator. The index repeats those checks for every input and repeats its own
schema and semantic validation at build, immediately before publication, and
against linked readback.

Command logs begin with canonical command-receipt v2 JSON. The remainder is a
binary sequence of unsigned 64-bit big-endian stdout length, stdout bytes,
unsigned 64-bit big-endian stderr length, and stderr bytes for each command.
Every frame is bound by command ordinal, exact argv/cwd/closed environment,
integer exit/signal, tool identity, byte offsets, sizes, and SHA-256. Stream,
command, and file limits are respectively 128 MiB, 256 MiB, and less than
512 MiB. A nonzero exit, overflow, parser failure, or postcondition drift
produces immutable failure evidence and prevents the final index.

The native gate is exactly `uv sync --python <frozen-base-python> --locked
--all-extras --offline` into a never-before-used environment. CPython is
exactly 3.13.7 and uv is exactly 0.10.9 with their frozen binary identities.
The single global uv cache path and directory identity are cross-bound across
native and package evidence. Native and package-build environments must not
contain pip. Pip 25.2 is permitted only in the package-install environment,
using the isolated CPython `ensurepip` wheel whose name, size, and SHA are
frozen.

The session source seal includes Git tracked/untracked bytes and a stable
physical package-source superset, including ignored nodes. After the exact
five-package build backend is installed, package evidence runs the real
Hatchling selector before and after the build and binds canonical sdist/wheel
rows. Package, hash-freeze, and index evidence must agree with a live
revalidation through the retained bound build environment. Phase 0 validates
the `quant-investor` package lane; it does not claim that the separate
`app = web.main:app` entry-point target is importable.

### Reviewed dual-runtime execution matrix

Phase 0 deliberately uses six distinct interpreter or environment roles. The
main-bound lane is a reviewed compatibility exception for the fixed lexical and
runtime semantics of the historical Factor v4.3/v4.4 suite. It is formal gate
evidence, but it is neither fresh nor hermetic and cannot replace any fresh
release gate.

| Role | Bound interpreter/environment | Permitted responsibility | pip lifecycle |
|---|---|---|---|
| base session parent | frozen CPython 3.13.7, invoked with `-I -S -B` | orchestrate the single non-resumable session, validate policy/evidence, and launch fixed child roles | pip is not importable or loaded and no site path is visible; the Homebrew installation may contain out-of-scope global pip files; no install or sync |
| base skip parent | the same frozen CPython 3.13.7, invoked with `-I -S -B` | produce the immutable skip baseline and launch the policy-controlled main-bound collection | the same isolated no-site pip-visibility rule; no install or sync |
| fresh native | never-before-used venv from exact locked `uv sync --locked --all-extras --offline` | native dependency evidence, v2 tests, recommended-core gate, mypy, and Black | pip absent; no `ensurepip` |
| main full-suite | pre-existing, byte-bound main-worktree venv; installed `quant-investor` metadata remains main-bound while candidate imports resolve under the candidate root | full-suite compatibility gate preserving Factor v4.3/v4.4 historical runtime semantics | no install, sync, or `ensurepip`; the observed environment is not mutated |
| package build | fresh owner-private venv with the exact five-package Hatchling backend installed by offline uv | build the canonical sdist and wheel and revalidate Hatch selection | the `pip` distribution and wrappers remain absent |
| package install | separate fresh owner-private venv containing only the built project plus the frozen install tooling | install the exact built wheel with `--no-deps --no-index --no-compile` and prove source/sdist/wheel/installed parity | the only role allowed to run `ensurepip`; pip is exactly 25.2 from the frozen bundled wheel |

The main-bound wrapper must reconstruct the hash-bound normal main-venv startup
semantics, including `site.venv` processing and `execsitecustomize`, before
replacing only the repository import root with the candidate root. Candidate
`quant_investor` modules must then resolve under the candidate worktree, while
the installed-distribution metadata and the rest of the runtime remain
explicitly main-bound.

Pytest entry-point autoload remains disabled. The exact pytest argv fragment is
`-p pytest_cov -p asyncio -p anyio`. Pytest 9 performs standard entry-point
resolution while preserving these registration names; no custom bootstrap
registration substitutes different names or modules. The resulting
plugin-manager order and hook trace are verified:

| Order | Entry-point name | Module value | Distribution |
|---:|---|---|---|
| 1 | `pytest_cov` | `pytest_cov.plugin` | `pytest-cov==7.1.0` |
| 2 | `asyncio` | `pytest_asyncio.plugin` | `pytest-asyncio==1.3.0` |
| 3 | `anyio` | `anyio.pytest_plugin` | `anyio==4.13.0` |

`SEALED` requires both halves of the reviewed exception:

1. the fresh CPython 3.13 locked offline sync plus native dependency, v2,
   recommended-core, mypy, Black, and package build/install parity gates; and
2. the policy-bound main-interpreter compatibility/full-suite gate.

Neither half is sufficient alone. If a fresh sync or native gate fails, the
main environment cannot fill the gap. If the main wrapper, its frozen policy,
plugin topology, or full suite fails, fresh results cannot erase the historical
Factor-runtime failure. Package build or install drift also stops sealing.

`OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED` remains a normative limitation for
both lanes. Evidence may assert `offline_policy_enforced=true`, but must also
state `kernel_egress_attested=false` and `network_unreachability_proven=false`.
If a legacy record still carries `network_actions_performed=false`, that value
is runner-declared intent only. The fixed command plan does not explicitly
request a known live API, but it does not audit every test's socket or egress
behavior and therefore proves neither absence of network API use, network
unreachability, nor absence of sockets. Phase 0 must stop rather than silently
upgrading the claim if OS-level egress attestation becomes an acceptance
requirement.

The skip baseline is produced before the final session by a separate fixed
offline producer. Acceptance requires the sum of canonical
`(path, line, reason, count)` rows to equal 42, with zero fail, error, xfail,
or xpass. The final full-suite rows must equal the frozen rows exactly; an
unchanged total with changed paths, lines, reasons, or counts fails closed.

The following limitations are normative and must be carried by the session and
final index:

```text
PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET
NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED
OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED
OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE
PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT
```

The main-bound full-suite runtime remains pre-existing and non-hermetic, not a
fresh release environment. Its pre-collection policy binds only the approved
runtime/plugin trees, not the complete main runtime fileset. The main
distribution inventory also retains four known invalid dist-info stubs; those
stubs are disclosed and classified rather than silently normalized into valid
distributions.

Both authority v16 roots and both candidate-worktree v16 roots are sampled
before and after every stage as either `ABSENT` or `PRESENT_DIRECTORY`.
This proves only their stable top-level identity and absence from direct
argv/env/cwd references; it does not claim process-I/O surveillance.

All evidence files use exact-once publication through a pinned owner-private
directory descriptor, a `0600` staged inode, `fsync`, a hard link to an absent
final name, and stable readback. The staged link may be removed only after
successful same-inode proof. No failed, orphaned, or published evidence is
deleted, replaced, repaired, or reused automatically.

Phase 0 `SEALED` still grants no runtime, CLI, Dashboard, schedule, release,
cutover, staging, broker, order, trade, or purge authority. The next possible
state-changing gate is an owner-confirmed per-file/per-hunk DirtySplitGate
discard manifest.
