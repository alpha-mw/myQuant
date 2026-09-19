# Phase 3: native Top100 Parquet publication

The source design requires the existing daily strategy directory to contain a
real `top100.parquet` plus a manifest binding its 100 rows to the exact Factor,
LOW/W80 observations, Market/PIT and approved strategy policy. The current
`DailyResearchPoolStore` atomically publishes four JSON leaves. Add the required
table to that existing publisher; no sidecar exporter, alternate root, strategy
change or replacement CLI.

## Contracts and historical compatibility

Retain the registered `daily_research_pool_manifest` contract and its exact
four-leaf reconstruction for immutable historical readback. Introduce the
responsibility-named `daily_research_tabular_pool_manifest` compiled contract,
with the existing manifest fields plus the eleven explicitly requested fields:
`trade_date`, `generated_at`, `factor_generation_id`, `factor_pointer_sha`,
`low_observation_sha`, `w80_observation_sha`, `market_pointer_sha`,
`pit_pointer_sha`, `policy_sha`, `row_count`, `top100_sha`.
Contract identity uses the repository semantic envelope, without numeric payload
schema labels. New publications always use this new contract and exactly five
leaves: the four existing names and `top100.parquet`.

All SHA aliases map to validated source bytes, not string guesses: Factor and
policy match rank bindings; LOW/W80 are native validated observations matching
rank artifact refs and each other on date, generation, Market, PIT and Calendar;
Market/PIT pointer SHAs come from those observations. Generation ID comes from
the exact rank generation ref. `trade_date` is the ISO rendering of signal_date.
`row_count=100`; `top100_sha` hashes the actual published Parquet bytes.

New publication uses the full validated observation artifacts supplied by the
existing CLI/core adapter, or reads their exact canonical date/alias paths and
checks the rank's full artifact refs when the optional argument is omitted.
It never scans for a recent input or substitutes an archived date. Verification
dispatches only by a registered, validated manifest kind. The active DAG probe
requires TABULAR explicitly; an old four-leaf pool cannot satisfy new production.
Existing completed EODs may read their original legacy closure unchanged.

## Table and timestamp semantics

Parquet has an exact nonnullable Arrow schema: rank int32 (1..100), symbol UTF-8,
low_percentile, w80_percentile and combined_percentile decimal128(13,12).
Rows preserve the exact native rank order and decimals. No float conversion,
new ranking, weights, theme filtering or portfolio admission. Use the installed
PyArrow dependency, fixed writer settings and schema metadata.

`generated_at` records actual UTC time once when building a new publication;
it equals the new manifest envelope created_at and cannot precede the rank's
source-derived created_at. Historical source time remains in the unchanged rank.
On repeat/readback, validate and retain this original persisted timestamp, never
consult a new clock or require it to equal the current invocation time.

Readback verifies stable original file bytes and SHA before decoding, bounds the
Parquet input size, and checks exact schema, 100 rows, all values and ordering
against the validated rank. It does not require a later PyArrow release to
reproduce identical serialization bytes. The persisted manifest/receipt still
bind exact bytes, and all read sources are checked unchanged during validation.

## Publication and integration

Extend existing staging/write/fsync/atomic-no-replace operations to the exact
format-owned leaf inventory. Verify every staged leaf before exposing the day
directory. Recheck native inputs/pointer through the existing before_publish
callback. On a concurrent winner, validate its original timestamp/table against
the same inputs and adopt only an exact semantic match, without overwriting.

Preserve public command_status PUBLISHED/NO_ACTION. Add publication_state
SUCCEEDED/ALREADY_SUCCEEDED. A same-day different input or format yields explicit
RESEARCH_POOL_CONFLICT / publication_state CONFLICT and preserves all old bytes.
Corrupt or incomplete existing directories fail closed and are never filled in.
Crash after directory publication but before journal terminal remains adoptable
through native verification, with no second publisher invocation.

New Top100 terminals bind all five output refs, including the binary file.
Core dependency checking and completed-core replay must separate byte custody
from JSON decoding for this one code-owned binary leaf; native table validation
remains in the owning pool store. Do not make generic suffix-based parse bypasses.
Research source admission accepts only the two compiled manifest kinds and
verifies the appropriate complete pool. Existing JSON consumers retain rank and
selection files. Existing public pool-publish CLI and automatic Factor→LOW/W80→
Top100 invocation must both exercise the tabular publisher.

The exact manifest-kind scan also finds `intelligence/morning.py` (legacy-only
kind admission) and `scripts/cn_weekly_review_v2.py` (manual legacy document
reconstruction). Update these two existing consumers to admit only the two
registered formats and use the owning pool readback. Preserve Morning's purely
read-only behavior; a missing or corrupt binary cannot count as verified Top100.

## Verification and stop conditions

Architect then Critic review precedes this public artifact/storage change.
Required focused evidence:

1. Real native observations and ranked 100-row fixture through existing publisher;
   table values/types/order and every manifest source field verified.
2. Second publication ALREADY_SUCCEEDED with identical all-file bytes/mtime and
   no callback/writer call; simulated concurrent winner and crash adoption.
3. Same-day changed valid input or legacy-format destination conflicts without
   modifying any existing bytes. No implicit upgrade of old directories.
4. Table tamper, missing/extra leaf, row reorder/decimal/schema/null changes,
   forged manifest binding and invalid source observation rejected.
5. Original four-leaf manifest/receipt fixtures remain readable byte-for-byte;
   they fail an explicit TABULAR requirement.
6. Core DAG automatically publishes five refs, downstream byte checks accept the
   code-owned binary file, completed-core replay validates it without JSON parse,
   and public pool-publish executes the same implementation.
7. Existing relevant storage/core/research-source/completion/weekly consumer tests
   and changed-file static checks; inspect scope before broad final CI.

Do not deploy or rewrite existing production artifacts. Preserve live frozen
acceptance processes and unrelated concurrent changes. Stop this phase once the
explicit tabular contract and its two real entrypaths are locally exercised;
later Phase0–15 acceptance remains open.

## Accepted Architect precision

- Keep `_pool_documents` exactly legacy; add a separate tabular builder and
  manifest-driven full verifier. First validate the persisted manifest and its
  registered kind, then that kind's exact leaf inventory. Never infer the format
  from a filename suffix or whether a table happens to exist.
- For an existing destination, inspect its persisted manifest before reading a
  clock. New manifest.created_at and receipt.created_at both equal generated_at;
  generated_at must be no earlier than rank.created_at. Only a genuinely absent
  destination samples a new canonical UTC second. A concurrent loser rebuilds
  against the winner's original timestamp and bytes, not its discarded candidate.
- Every existing-directory mismatch (legacy format on new publication, different
  input, corruption, missing/extra leaf) raises a typed RESEARCH_POOL_CONFLICT with
  publication_state=CONFLICT. No filling, success-shaped error or overwrite.
- Fixed Arrow field order is rank, symbol, low_percentile, w80_percentile,
  combined_percentile; all fields nonnullable. Schema metadata is exactly
  `quant_investor.pool_format=top100`. Symbol grammar remains the native rank's
  canonical CN symbol rule. All percentiles retain decimal128(13,12).
- First-write Parquet settings: version=2.6, data_page_version=2.0,
  compression=NONE, use_dictionary=false, write_statistics=false,
  row_group_size=100, store_schema=true, write_page_index=false,
  write_page_checksum=true. Bounded input is at most 8 MiB; metadata must describe
  one group, five columns and 100 rows before table decoding. Replay compares
  exact schema including metadata, null counts and every expected value.
- `CoreContext._check_upstream` receives the code-owned upstream node identity.
  Top100 calls owning full pool verification and compares the complete returned
  output map with the terminal map; other upstreams retain JSON verification.
  Completed-core replay also delegates native table validation, and records the
  verified Parquet bytes for its unchanged-source check without JSON parsing.
- Source aliases are reconstructed from exact native artifacts. A valid envelope
  or a caller-provided set of SHA strings alone does not satisfy provenance.

Architect verdict: APPROVE_WITH_CHANGES; above amendments adopted before Critic.

## Local implementation acceptance

COMPLETED_LOCAL_VALIDATION. Current publisher exercised exact frozen native
synthetic rank/observations in an isolated workspace and produced the real five
files. Repeat publication preserved all bytes/mtime. Public CLI, automatic core
publisher, crash adoption, binary and legacy completed-core replay, active legacy
rejection, Morning readback and late-publication classification are covered by
121 passing focused tests (21.82s). Nine source files pass mypy; changed-file
flake8 and Black checks pass. Exact hashes: `.agent/acceptance/phase3-validation.json`.

The full weekly closure fixture still fails its maintenance receipt gate
CLOSE_RECEIPT_REPLAY_MISMATCH, while its Top100 portion verifies. The same failure
was reproduced on untouched frozen 3c46 source. Preserve this pre-existing failure
and the native guard; it is not a claim of whole weekly or whole-DAG acceptance.
Final installed release and complete Phase0–15 acceptance remain separate.
