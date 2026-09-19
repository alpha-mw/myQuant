# Phase12 native source producers before request freezing

Status: IMPLEMENTED_LOCAL_VALIDATION. Main plan and parent-custody amendment both
completed Architect then Critic approval before implementation. This change repairs
the two native source dependencies of daily_prepare; it is not scheduler or
full Phase12 acceptance.

## Observed dependency and intended result

The existing publish-daily-event-closure requires a maintenance receipt, while
recipe v5 requires the Event pointer before maintenance. Its implementation checks
only a target-like JSON field and permits creating a new closure for any historical
date after cutoff. Repair that existing command with exact native Calendar input
and an explicit current-day-only creation gate. Do not invent retrospective owner
facts or a second Event store/publisher.

The existing scripts/operations/run_cn_benchmark_close.py contains its provider,
merge and native publish logic behind a script path insertion. Move that producer
into an installed package owner and retain the existing script CLI/arguments as a
thin wrapper. Add exact wire custody, missing-date completeness and replayable
capture so a retry need not call the provider again. No bridge changes this step.

## Event command repair

Own scripts/manage_cn_strategy_records.py::command_publish_daily_event_closure and
its existing CLI parser. Preserve existing maintenance-receipt and SHA flags as a
supported input form. Add an alternative exact Calendar receipt+SHA and raw
Calendar+SHA pair. The two input forms are mutually exclusive and complete; no
defaults, alternate paths, latest scan or manufactured maintenance receipt.

- Both routes resolve source bytes safely inside the exact project root, check
  paths/SHA/identity and replay_close_session_authority. The maintenance route
  requires native cn-daily-maintenance-attempt.v1, a matching target, its exact
  close_session_receipt_ref, and the Calendar's exact raw_response_path/bytes.
  The new route uses the four explicit Calendar flags directly. A Calendar-only
  source is never labelled a maintenance execution.
- Use the existing script-side official-close policy owner parser for full scope;
  also require the explicit standing empty-inventory policy, Shanghai timezone,
  15:30 cutoff and policy effective time. Do not change owner policy bytes.
- Replay current pointer-selected closure set before a new write. Existing exact
  date closure can return NO_ACTION after its original sources are checked; no
  closure is rewritten/re-signed. Current input arguments do not replace that
  original evidence. A conflict retains OFFICIAL_CLOSE_RESTATEMENT_REQUIRED.
- New closure requires requested day == actual Shanghai today == Calendar
  observed date == native authorized close date, classify_requested_session is
  MATCHED_OPEN, Calendar observation <= now, and now >= same-day15:30. New
  historical/future/closed-day creation fails. Current standing-policy empty
  closure is the same previously authorized owner policy operation; this does
  not create an event intake path or infer retrospective facts.
- Reuse build_empty_closure/publish_generation and preserve every existing row.
  New source_receipt_ref points to the exact selected Calendar receipt (native raw
  linkage retained) or original maintenance receipt for the legacy route. Policy
  ref remains owner_declaration_ref as in existing standing-policy closure.
- Existing command execution keeps its outer native _operation_lock and Event CAS.
  Programmatic callers must hold that same lock; no scheduler call is made here.
  Recheck policy/source bytes and pointer before publication, then native readback.

## Installed benchmark producer

Own new quant_investor/market/cn_benchmark_capture.py and reduce the existing script
to a compatibility launcher. Preserve CLI flag names, default source=tushare,
PLAN_ONLY behavior and existing initial-full-history guard. The future scheduled
composition will call the package producer with source=tushare; this step performs
no provider call in the real workspace.

The package callable receives explicit workspace/start/end/generation/preimage and
optional required ISO dates (exact Calendar-derived dates in daily composition).
No arbitrary URL/token payload/configured command. Tushare uses the fixed native
OfficialTushareHttpsClient and index_daily for the existing three REQUIRED_CODES;
21 seconds between requests, with monthly windows and bounded response parsing.
Never change the caller's existing TUSHARE_TOKEN environment. The legacy CLI may
load PROJECT_ENV exactly as before, with scoped restoration around its invocation.

Persist a capture commitment before native publication. Its exact v2 fields are:
schema_id=myquant.cn_benchmark_acquisition_receipt.v2, generation_id, captured_at,
start_date, end_date, required_dates (null or sorted unique ISO dates), codes,
expected_pointer_sha256, row_count, rows_sha256, source_system, credential_contract,
credential_material_recorded=false, broker_order_trade_authority=false,
requests (ordered exact request records), content_sha256.

Each Tushare request record has exactly api_name=index_daily, params (ts_code,
start_date,end_date), fields=[ts_code,trade_date,close], raw_ref={path,sha256}.
Raw responses are immutable byte files under the existing generation capture root,
named by request index and wire SHA; never persist the credential-bearing request
body. The ordered request set must equal the expected three-code/month-window
partition, with no missing/extra/duplicate requests. Re-read and replay every
raw_ref through replay_tushare_response_bytes, validate no pagination/truncation,
exact fields, code/date/window and finite positive prices. Recompute rows/count/SHA.
Eastmoney remains only the explicit legacy CLI route; preserve its existing v1
receipt behavior and do not claim native Tushare wire evidence for it.

- New rows must cover all required dates and all three codes; reject unexpected
  dates when an explicit required set is supplied. Absence never means a holiday.
- Merge native existing generation rows additively. Identical overlaps may replay;
  any differing existing value/metadata fails BENCHMARK_EXISTING_ROW_CONFLICT,
  requiring separate correction authority. Do not silently overwrite history.
- Existing exact v2 capture is re-read with original arguments/preimage and wire
  refs; same-generation mismatch or partial conflicting bytes block. It can
  resume native publish with no provider access. A crash before commitment may
  require a new capture; no previous success is claimed from uncommitted files.
- If current native generation/receipt equals the expected committed publication,
  return NO_ACTION and avoid rewriting compatibility bytes. If only compatibility
  bytes are missing/stale, regenerate the exact projection from that native
  generation; no new provider/generation is created. A foreign advanced pointer
  cannot be rolled back or substituted.
- Use existing native publish_generation/CAS and immutable generation readers.
  Compatibility CSV is always a projection, never financial benchmark authority.
  Capture has original provider acquisition time; replay does not retime it.

## Acceptance and stop conditions

Use native Calendar wire fixtures and native Event/benchmark stores in isolated
synthetic workspaces. Control transport and clock only; prohibit real providers,
actual book writes, scheduler changes, owner policy edits and old archived runs.
Event: both exact source forms, pre-cutoff, closed/future/historical creation,
forged target-only receipt, raw/SHA/path mismatch, full closure preservation,
existing historical NO_ACTION and original-ref readback, drift/conflict/CAS.
Benchmark: three native wire requests, completeness, wrong code/date/duplicate or
paginated/truncated response, finite prices, explicit missing dates, overlap
conflict, capture replay after write interruption, same generation no-provider
NO_ACTION, compatibility-only repair, foreign pointer rejection, unchanged token
environment and retained legacy PLAN_ONLY wrapper flags.
Run scoped tests/static checks plus daily preparation/Store/Event/benchmark and
full native import regressions. Then stop this producer change. Actual installed
source config, capture orchestration and launcher profile wiring are still the
next Phase12 integration; no source inputs are falsely marked ready in this step.

## Accepted Architect amendments (authoritative)

1. Add package `strategy_records/daily_event_source.py`, shared by daily command
   creation/NO_ACTION/post-publication readback, CorporateActionEvidence and the
   daily preparation/corporate cutoff consumers. It applies only to explicit
   standing-policy daily closures (owner_declaration_ref == policy_ref). Symbolic
   catalog receipts keep the existing resolver; separately declared retrospective
   sources keep their existing route. The generic Event store structural reader
   stays independent of workspace-level external source interpretation; the daily
   owner adds semantic validation after its native generation readback.
2. Source dispatch uses exact schema only: close-session-receipt.v1 or
   maintenance-attempt.v1. Calendar requires its raw_response_path and
   raw_response_sha256 to bind the explicitly selected raw file; native replay
   proves target/observed day and MATCHED_OPEN. Maintenance requires target_date
   exactly (no aliases), mode execute, native component-terminal field profile,
   exact stage fields/order/status/write booleans, write-count/unchanged coherence,
   exact state_ref with matching state fields/stage_states, and Calendar/raw link.
   Recognized optional fields are only the current producer's transport_retry,
   core_completion_ref, factor_loop, write_veto_ref, macro_write_veto_ref,
   logical_claim_ref, started_ref and the three provider-summary fields. No unknown
   field or target-only JSON is accepted. Modern started/logical/provider profiles
   must also bind the fixed ended.json terminal receipt. The known older complete
   component-terminal/state profile can be read without inventing an ended marker;
that establishes only Calendar/attempt source semantics, not Factor/EOD success.
   PARTIAL/BLOCKED attempt status cannot be reclassified COMPLETE by this reader.
3. Existing closure NO_ACTION checks expected pointer and the entire native
   generation, then original policy/owner/source/raw dependencies and their hashes
   before rechecking pointer. New supplied evidence never replaces originals.
   Invalid originals fail; no re-signing or historical recreation.
4. Give the daily command an owning operation-lock entry and a private locked
   helper. CLI excludes this one self-locking handler from its generic outer lock;
   programmatic command calls therefore take the same native operation lock too.
   Lock order stays Record operation -> Event current. No serialized lock flag.
5. Retain the exact four-way date equality/current cutoff/effective policy gate
   above, with actual seal/generation time and zero new historical closures.
6. Tushare capture path is exactly
   data/private/cn_benchmark_close/<generation-id>/raw/<index>-<wire-SHA>.json and
   capture.v2.json. Always inspect v2 before token access/client creation/sleep.
   Existing v1 blocks same-generation v2 creation; no upgrade. V2 replay binds the
   original arguments/preimage, partitions and every raw byte. Partial uncommitted
   raw files establish no success; a new response may coexist only at its exact
   content-addressed name and the eventual committed request set is unambiguous.
7. Request order is REQUIRED_CODES order times ascending monthly windows; params
   are compact dates, top-level range/required dates are ISO. Capture-complete
   timestamp is sampled after the final raw is stably written; replay preserves
   it. Required dates must be nonempty sorted unique/range-bound when supplied,
   and exact Cartesian code/date completeness is required. Null supplies no
   Calendar completeness claim. Reject duplicate rows as ambiguity.
8. Full normalized row equality governs overlap; no tolerance or source refresh
   override. All historical value or metadata conflicts fail.
9. Extend the existing native benchmark store owner with a publication scope using
   its existing .latest.lock, with a nonserializable/revoked scope identity. Keep
   existing publish_generation signature/behavior; internally reuse the scope.
   New governed publication helper installs/adopts exact candidate and writes the
   compatibility CSV inside this same scope, rechecks pointer and CSV before
   unlock. Existing publish_generation writers also take the same lock, so another
   publisher cannot advance between this candidate CAS and projection write.
   Compatibility repair requires current == committed candidate; a foreign
   successor blocks before projection writes. No second benchmark lock/store.
10. Package acquisition does not mutate the token environment; CLI PROJECT_ENV
    scope restores absent/original state in finally and is entered only for fresh
    transport. No token, token hash, credential-bearing request body or exception
    text is written to evidence.
11. Add tests for valid supplied/invalid original Event source; all source-reader
    entrypoints; programmatic Event lock; modern terminal coherence; and concurrent
    native benchmark advance blocked until matching compatibility publication.
    Preserve and disclose any legacy source that cannot satisfy its exact profile;
    do not repair facts or modify archived production evidence to pass.

### Parent custody amendment (Architect APPROVE)

Add exactly parent_pointer_ref to v2 capture. Existing benchmark store retains
generations but not old pointer bytes. For nonempty preimage, before any provider
call retain the exact safely read current pointer at fixed
data/private/cn_benchmark_close/<candidate-id>/parent-pointer.json; ref SHA must
equal expected_pointer_sha256 and parent generation must differ from candidate.
Use this pointer's exact native manifest/series refs and immutable generation
reader to validate parent rows. Build candidate only from these parent rows plus
captured rows, including strict full-row overlap equality. Never use current
candidate rows to reconstruct old history.

For EMPTY preimage, parent_pointer_ref=null and no parent-pointer.json may exist.
Without a capture commitment, an orphan parent file gives no recovery authority;
actual current preimage must still match before any acquisition. With commitment,
replay parent and raw sources first, then publication lock allows only expected
preimage or the fully reconstructed exact candidate pointer; any foreign pointer
fails. Candidate manifest's acquisition ref is the final receipt SHA, so there is
no SHA cycle. Parent/generation/raw bytes are rechecked on every readback.

### Historical diagnostic references

The actual Sep3 attempt records MACRO_WRITE_VETO.json SHA94ca6446..., while that
mutable diagnostic file has subsequently advanced to0730ed36.... The new daily
source reader validates the old veto reference syntax in the immutable attempt,
but does not dereference today's veto as historical Event authority. It still
replays original Calendar/raw, immutable attempt/state, original policy and modern
terminal/start/claim refs. Its output retains maintenance_status=PARTIAL and
eod_admission=false; it makes no Macro or Factor success claim. No original
receipt/veto bytes or SHA are changed. A regression test covers this separation.

## Local validation and remaining integration

Final related regression:256PASS202.87s, including native Record/Event/benchmark
stores, source producers, preparation, Corporate, cutoff, adoption and full closed
script imports. Focused producer suite:50PASS3.07s (overlaps). Black/flake8 passed
for11 selected files; mypy passed for6 sources; four changed functions in the two
large existing scripts passed separate formatting checks, preserving other edits.
Exact source hashes and test sessions:
`.agent/acceptance/phase12-native-source-producers-validation.json`.

The real Sep3 source readback passed only the declared Calendar/attempt scope and
retained PARTIAL with EOD admission false:
`.agent/acceptance/phase12-native-daily-event-readback.json`.
All provider and financial-write tests use isolated synthetic workspaces; no real
provider, portfolio, veto, owner policy or scheduler was changed. Next work remains
the installed static config, Calendar/source orchestration and existing launcher
integration, followed by genuine bootstrap and unattended acceptance.
