# Phase 9: Dashboard publication bound to completed daily evidence

Status: Architect amendments accepted; Critic APPROVE; implementation and local validation in progress.

## Reproduced defect and intended outcome

`.agent/acceptance/phase9-publication-before.json` executes the actual current
Dashboard publisher against an isolated native Store backlog. Its selector is
UPDATED and all six serving files exist while completion.v1.json does not exist.
This is the real publisher-return/pre-EOD-seal boundary, not a fabricated full-day
receipt. The current composition seals EOD only after Dashboard publication.

New production must first validate and seal the complete daily evidence DAG, then
publish a current Dashboard that explicitly consumes Store, Factor, Top100, Theme
and Decision from that seal. Failure to finish current publication must be visible
in the same producer run and its public exit status. It cannot be reported as
COMPLETE merely because the evidence seal exists. No holding/trade/policy changes.

## Explicit version route, legacy replay preserved

Add execution recipe v4 = v3 plus exact code-owned
`dashboard_publication_policy="native-eod-first.v1"`; materialization v4 at fixed
materialization.v4.json with the same addition; native inputs v5 = v4 plus that
literal. All previous native/recipe/materialization versions remain readable and
replay their exact old outputs. Extend all fixed-path version conflict checks,
loaded-context equality, catchup templates, Corporate/Decision v4+ compatibility,
EOD exact adapter-type checks and native completion readers.

Only native inputs v5 selects the new Dashboard adapter. It runs the existing
native financial render and capture in the requested historical/current mode,
but never writes serving bundles or selector. A successful node means the exact
rendered evidence is retained and validated. Existing v1/v2 financial schemas and
financial formulas stay unchanged. Legacy CurrentDashboardAdapter retains its
historical replay profile; do not reinterpret previously sealed output sets.

## Bind all required DAG authorities

The new adapter is constructed from the fixed registry after the required native
nodes have succeeded. Collect exact journal request/terminal/output refs for
store, factor, top100, theme and decision from the same day/release and validate
request identity and selected terminal equality; no current/latest result search.
Each producer's original validators remain authoritative. The final sealer already
replays every native node and must verify these bindings again.

Add registered `daily_dashboard_evidence` as a fixed named Dashboard output
`daily_evidence`, alongside `capture`, `v1`, `v2`. Its source recipe records the
five named terminal refs and exact output refs. Its payload has trade_date,
strategy_id, source_bindings, source_refs, top100_count, decision_state_counts,
and research_state. Counts are derived from the native Top100 manifest and Phase6
Decision report, not recomputed selection logic. State counts use the existing
five canonical Decision states; no new confidence/stock-ranking semantics.
Research_state is EVIDENCE_BOUND after all five source domains validate. Native
Factor and Theme identities/dates remain available via exact source bindings.
No positive source state is inferred merely from a source-looking filename.

The private page adds a small dated evidence section showing the five source
states/identities and native counts. It reuses existing visual styles. It cannot
show a current research section if its date/refs disagree with the selected
financial bundle or EOD. Public v1-only pages and public export remain unaffected.

## Authoritative completed head and publication order

The new profile's EOD keeps the existing completion.v1.json path and v2 completion
shape: its Dashboard terminal now proves immutable capture/evidence, not serving
publication. No publication receipt is inserted into EOD, avoiding a hash cycle.
Native completion replay for v5 rebuilds the capture/evidence set without requiring
mutable serving files or a later publication receipt.

An authoritative private current-completed head is necessary to establish latest
completed date without scanning results or confusing advanced Store/Market heads
with completed DAGs. New-profile convergence writes
`portfolio_dashboard/private/generated/cn_daily_completed_head.v1.json` only after
native EOD validation. Exact fields: schema_version=cn-daily-completed-head.v1,
trade_date, completion_ref, previous_head_sha256, registered_at, authority,
content_sha256. authority uses the existing all-false operations shape. Its
semantic SHA excludes content_sha256. Retain original head bytes in fixed
content-addressed custody before replacing the head. Head registration is a
projection over existing validated EOD, not another financial authority.

A single governed private Dashboard publication lock serializes head registration
and current serving writes across days. Reuse SecureSystemStorage's current-owner
safe lock/path primitives. Hold day lock before Dashboard lock; never reverse the
order. Avoid nested flock acquisition: the native publication functions share a
code-owned lock context when called by the new publisher. No network or long
native full-DAG rebuild occurs while holding this global publication lock.

Head transitions are monotonic by date and exact refs. Same date/same completion
is read-only; same date/different completion conflicts. A newer candidate must
prove the exact predecessor chain through its declared completed input refs.
First registration requires the candidate's native-verified explicit predecessor
or bootstrap evidence. A crash after sealing but before head update can be
recovered by traversing only those declared predecessors, never a directory scan
or inferred missing-day completion. Older candidates cannot replace the head.

After head registration, publish only the exact head-selected native EOD's retained
financial pair and daily evidence. Refuse a stale candidate, mismatch, missing
seal, source integrity failure or expired native view. Revalidate exact head bytes
before final selector commit and final readback. Do not select mutable current
Store/Market heads after sealing; their later advancement alone must not substitute
a different source or invalidate a valid retained evidence set.

## Serving contract and existing writer boundary

Add private `cn_daily_dashboard_evidence.v1.json` and `.js`, derived from the sealed
Dashboard evidence plus exact completion/head refs. Existing financial filenames
remain unchanged. New selector v3 retains old selector fields and adds trade_date,
completion_ref and daily_evidence_sha256. All JSON/JS bytes are deterministic from
retained data plus a persisted actual publication intent time. Selector updated_at
is actual publication time, never backdated to generation time; it must be after
EOD validation and within the native financial view's valid-through boundary.

Reuse native bundle/selector writers with an explicit internal sealed-publication
context. Once the governed completed head exists, legacy direct current publication
must fail with a clear EOD-publication-required error instead of bypassing the new
boundary. Legacy read-only replay and historical rendering remain allowed. Both
native current writers participate in the same lock/guard, so an old standalone
export cannot race a new sealed publication or restore an old selector. Keep
private output confinement and ReplaySources' no-writer guard intact.

The private client validates the new selector's completion/date/evidence/financial
bindings and displays a dated complete view only when they agree. It checks the
authoritative private completed-head JSON when serving the new profile; missing,
malformed, unavailable or mismatched head blocks the current-view claim. Preserve
legacy v2 contract reading, with no automatic fallback from an invalid v3 to v2.
Resolve file/HTTP loading compatibility explicitly during implementation review;
do not claim a stale embedded JS projection proves the current authoritative head.

## Recovery, producer result and observational status

Persist a fixed publication intent before serving writes, then write all data
before the selector. Retain expected bytes/refs and actual clocks. A fixed
post-seal receipt `dashboard/serving-publication.v2.json` binds the exact EOD/head,
intent and all serving files. Publication failure never rewrites EOD or invokes
Store/Factor/research producers again. Recovery validates the retained intent and
selected head, completes only missing publication work, and adopts identical
already-published bytes without changing their times or replaying business writers.

Both fresh convergence and the existing-completion RESUME branch invoke the
post-seal publication boundary for current v5 inputs. Historical v5 inputs keep
dated capture/evidence and do not replace the current selector. A historical
request's frozen mode is never edited to make a failed current request pass.

Return the existing public production result shape. If current publication is
pending/failed, its day is INCOMPLETE with nonzero exit status and no claimed public
completion_ref in that result; the immutable evidence seal remains discoverable
through exact local status. The local status projection distinguishes
EVIDENCE_SEALED_PUBLICATION_PENDING from full current serving completion. An older
candidate superseded by a newer completed head is rejected for current publication;
it must not overwrite newer bytes or fabricate a success receipt. Use existing
typed failures with clear DASHBOARD_STALE/publication-pending reasons, rather than
claiming success or inventing a second financial run.

## Verification and acceptance

- Reproduce baseline native unsafe publication; preserve its failed-boundary proof.
- New current Dashboard node emits only dated immutable capture/evidence. No serving
  files/selector changes before a fully validated native EOD seal.
- Exact Store/Factor/Top100/Theme/Decision source/date/release/ref binding tests,
  native evidence-derived count checks, source corruption and legacy replay.
- Real native financial renderer/publisher exercised in isolation with precise
  outer-DAG admission seams disclosed; new composition/RESUME tests verify order
  and public incomplete exit on publication failure. Full installed DAG proof is
  still required in Phase15.
- Crash before EOD, after EOD before head, after head before data, after data before
  selector, and after selector before receipt; exact recovery, unchanged financial
  state and no second business writer. Do not backdate recovered availability.
- Two dates/concurrent publishers: monotonic completed head and selector; exact
  predecessor handling; stale caller and legacy writer cannot roll UI backward.
- Head missing/bad SHA/symlink/future timestamp, stale/expired view, unknown versions,
  conflicting same-date refs, missing evidence and unavailable source never fall
  back to another date or legacy view as current.
- Private client contract tests and browser visual/readback check for valid,
  publication-pending and mismatch states. Public page separation preserved.
- Record source hashes, focused tests, native receipts, residual production facts,
  and exact rollout requirements. Do not claim whole Phase0–15 complete.

## Accepted Architect decisions

The current-completed head means latest **registered completed EOD**, not current
holdings. New page/view designation is LATEST_COMPLETED_EOD and always displays the
exact EOD date/cutoff. It must not reuse labels asserting current-day holdings or
renew the original native v2 freshness. Later Store/Market movement does not alter
the dated snapshot. A new publication after native valid_through fails with
EVIDENCE_SEALED_PUBLICATION_EXPIRED; a previously served dated EOD may remain
readable as dated evidence, never as renewed current-holdings authority.

Head rules: preserve original bytes/time on same date/same ref; conflict on same
date/different ref; reject older candidate. For a newer candidate, traverse exact
previous_completion_ref links (maximum32 links), rejecting missing refs, cycles,
non-decreasing dates and overflow. Stop only at the selected head's exact EOD or
the first-registration candidate's validated explicit bootstrap/predecessor.
previous_head_sha256 is null only at first registration. Predecessor bytes are
retained at their exact SHA path before head replacement and reread after CAS.
Incomplete head proposals do not become authority. An exact valid mirror proposal
matching the candidate, prior head SHA and EOD chronology is adopted byte-for-byte
with its original registration clock; mismatched proposals fail closed. Already
committed head time is never rewritten.

Publish exact JSON and JS completed-head forms. Selector v3 adds trade_date,
completion_ref, completed_head_sha256, v1_byte_sha256, v2_byte_sha256 and
 daily_evidence_sha256 to the old selector fields. The two financial byte hashes
are hashes of the exact retained/rendered JSON bytes, not guessed from a browser's
floating-number reserialization. Head SHA is the exact authoritative head JSON
byte SHA; daily evidence SHA is its exact serving JSON byte SHA. Fixed shapes and
semantic/body hashes remain independently checked by native validators.

Head mirror order is deliberately mirror-first, then authoritative JSON CAS,
under the publication lock. Before a new head becomes authoritative the file
route already has a new mirror that cannot agree with the old selector, so a
crash cannot leave both old mirror and old selector falsely agreeing after the
JSON head advances. A failed first head commit may leave a mirror proposal; it
is a blocked/pending state, not authority to restore legacy publication. Either
head-form's existence activates the guarded publication boundary. Only a JSON
head matching the selected native EOD is authoritative for native writers.

Client route is explicit:
- HTTP/HTTPS: fetch the private head JSON with cache bypass, validate exact bytes,
  shape, content and bindings; compare with selector v3. Missing/unavailable head
  for v3 or a present head mirror blocks. A true404 with no head/mirror and a
  legacy selector remains pre-cutover legacy reading only.
- file://: read the explicitly registered head JS mirror, require its exact raw
  JSON SHA, date and completion to agree with selector v3. This is a declared
  offline mirror route, not fallback to v2, and the view is always dated EOD.
- New generated JS forms include the exact original JSON text alongside parsed
  data, so byte hashes can be checked without reserializing numbers differently.
  Legacy JS byte forms stay unchanged. Use browser WebCrypto/UTF8 for actual byte
  verification, and block v3 if the required crypto is unavailable. Node contract
  tests alone do not establish browser hashing/file behavior.
- Present but inconsistent head/selector/evidence/financial data yields a blocked
  or pending view. An invalid v3 must never silently show legacy v2 as current.
  Public v1-only rendering does not load these private sources.

The publication capability owns both the lock and authorization. It is constructed
internally after exact native EOD validation and head/preimage admission, confines
all writes to the fixed private file set, and expires when its context exits.
Requests cannot serialize it. Both existing bundle/selector functions enter this
shared context (or acquire only the Dashboard lock as legacy callers); they never
reacquire a held flock. Legacy calls are refused before any serving write when
head JSON or mirror exists. Rollback/recovery remains in the same context.
Retained-source publication is admitted only for the exact captured bytes bound
by the validated seal. It does not disable the general ReplaySources no-writer
guard or treat arbitrary retained-source contexts as publication authority.

Publication intent binds completion/head, Dashboard terminal/output refs, exact six
data-file refs, actual intent_created_at and native valid_through. These inputs
fully determine the selector template without assigning a publication timestamp.
The selector uses actual post-data-readback commit-attempt time; complete matching
JSON/JS pairs may preserve their already-written time while still unexpired.
A separate immutable selector-commit seals exact two selector refs, original time,
actual first_verified_at, intent/head/completion, valid_through and false authority.
Without that commit, expired recovery cannot promote file bytes into availability
proof. With a valid pre-expiry commit, a missing final receipt can be recorded as
historical metadata after expiry: preserve original selector/verification times
and add actual receipt_recorded_at. Existing receipts are immutable. Expired views
remain dated STALE and cannot satisfy fresh current publication completion.

The v5 Dashboard request identity includes all five authority terminal refs. Its
terminal outputs are exactly capture/v1/v2/daily_evidence. Source bindings include
request refs, terminal refs and exact output refs for each domain. Native Top100
manifest pool_size and selected-symbol set must match the Decision v2 company set;
all five canonical state counts are present and their sum equals that set. The
current source names are Factor generation; tabular Top100 manifest/rank/selected
symbols/Parquet; Theme artifact/capture; Decision native result plus Phase6
 decision.v2.json/capture; Store's exact native day outputs. No directory search.
Completion replay uses retained originals only, never serving files or completed
head. Current publication completion is a separate post-seal admission gate.

The existing public result shape remains unchanged: serving failure is INCOMPLETE,
public completion_ref=null and exit2; local status retains the precise sealed EOD
with EVIDENCE_SEALED_PUBLICATION_PENDING or EXPIRED. Historical v5 may return its
completed EOD without a head/serving write. Superseded older current candidates
remain historical evidence and must not replace current files or receive a
fabricated publication receipt.

## Implementation alignment corrections

A bounded implementation review found acceptance blockers. Address all six:
factory-sealed immutable verification proofs and proof-derived capability bytes;
selector commit clock after data readback with expiry checks at selector/receipt;
client STALE dated view after expiry; shared full read-only receipt verification;
exact adoption of interrupted head-JS proposals; sticky legacy rejection when a
selector-v3 commit remains after other serving markers are missing.

For selector timing, the immutable intent binds six data files and the exact
selector template inputs (EOD/head/data hashes), not a prepublication clock.
After data readback, either adopt an already matching complete JSON/JS selector
pair with its original timestamp, or construct a new selector using actual commit
attempt time and check expiry immediately before the write. Partial/uncommitted
selector bytes do not freeze a false publication time. Recheck expiry after
selector readback and before receipt sealing. Persist a selector-commit record
only after both selector forms have been verified, then the final receipt.
A crash after selector write but before that record is recovered by validating
and adopting the complete pair, without changing its timestamp. A partial pair
requires a fresh current commit attempt while still within valid_through.
Final receipt binds the exact selector-commit record and timestamp, and retains
separate actual first_verified_at; no inferred earlier availability. Existing
committed metadata is immutable, and corruption after verified commit fails.

Architect amendment accepted: without a sealed selector-commit, expired recovery
cannot mint commit/publication evidence from selector bytes. A sealed pre-expiry
selector-commit may support a missing final receipt after expiry as historical
metadata only. Preserve original selector_updated_at and first_verified_at; add
actual receipt_recorded_at. Chronology is EOD <= head <= intent <= selector <=
first_verified_at <= valid_through. The commit binds completion/head/intent refs,
head SHA, exact two selector file refs, times, valid_through, recovered_unknown,
false authority and content hash. Final public state after expiry is dated STALE.
