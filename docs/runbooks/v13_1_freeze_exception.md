# myQuant v13.1 Freeze-Exception Runbook (Retired)

> Status: historical only. The v13.1 freeze and its merge exception were
> retired on 2026-07-15. This file remains for reproducibility of archived
> evidence and does not govern current `main`, schedules, activation, or merges.
> Use `docs/runbooks/v14_operations.md` for the current operating contract.

This runbook historically operated the local Dashboard 2.0, Theme Protocol v2, and
FactorGovernanceProtocol v2 implementation. It is offline by default and does
not authorize a broker, Web API, LLM, network provider, registry transition, or
merge.

## Historical safety state

- The isolated `codex/myquant-governance-dashboard-v2` branch requirement is
  retained here only as historical evidence; it is not a current branch gate.
- Keep `THEME_V2_FORMAL_ENABLED=0` and
  `THEME_V2_FORMAL_KILL_SWITCH=1`, and keep
  `THEME_FORMAL_RECONCILIATION_PERSIST_ENABLED=0` until the joint gate is ready.
- Factor mining and health are report-only. PR4 forward apply is statically
  disabled even when governed arguments and local canonical replay evidence are
  present.
- Keep private snapshots, Theme membership/evidence/theses, replay evidence,
  threshold seals, WALs, and mutation ledgers under ignored `private/` or
  `results/` paths.
- A blocked surface stays blocked. Do not replace missing evidence with zero,
  a legacy proxy, a hand-written pass flag, or a same-day repeated snapshot.

The current one-factor registry is a baseline state, not proof that one factor
is methodologically correct. Historical factors remain shadow evidence; this
runbook never restores them in bulk.

## 1. Dashboard full-snapshot replacement

Export from the canonical strategy record and a strict local trading calendar.
The exporter writes the private contract beneath the dashboard's ignored
`private/` directory and leaves the tracked compatibility loader sanitized.

```bash
SOURCE_REPO=<read-only-source-repo>
PYTHONPATH="$PWD" python scripts/export_cn_aggressive_dashboard_data.py \
  --record-root "$SOURCE_REPO/results/strategy_records/CN/aggressive_tech_manufacturing" \
  --dashboard-root portfolio_dashboard \
  --benchmark-source local \
  --benchmark-file "$SOURCE_REPO/portfolio_dashboard/inputs/cn_index_benchmark.csv" \
  --trading-calendar-root "$SOURCE_REPO/data/parquet/cn/bars"

PYTHONPATH="$PWD" python scripts/check_cn_dashboard_export.py \
  --dashboard-root portfolio_dashboard
```

Use `--require-production-benchmark` only when the benchmark artifact declares
and passes production provenance. A missing explicit analysis date, quote time,
Theme date, fee provenance, trading-calendar mask, or attribution day is
`null/partial/blocked`; the exporter and checker must not infer it from the
latest filename or record date.

The static application can load the private snapshot or a user-selected file.
Uploaded data is session-only. It must not enter Git, URL parameters,
`localStorage`, or the tracked sample bundle. The JSON/JavaScript snapshot and
every real CSV/export summary are atomically written mode `0600`; the checker
rejects payload mismatches or relaxed permissions.

## 2. Theme membership and PE/VC knowledge

Theme membership is curated from a trusted local source. The migration tool
does not infer symbol mappings. Approval is hash-bound and requires a trusted
symbol master; missing coverage returns `coverage_blocked`.

```bash
PYTHONPATH="$PWD" python scripts/manage_theme_membership_v2.py build-draft \
  private/imports/theme_memberships.jsonl \
  --symbol-master private/reference/symbol_master.json

PYTHONPATH="$PWD" python scripts/manage_theme_membership_v2.py approve \
  private/theme_knowledge/membership_drafts/<draft>.json \
  --expected-draft-hash <reviewed-draft-sha256>

PYTHONPATH="$PWD" python scripts/manage_theme_membership_v2.py validate \
  --symbol-master private/reference/symbol_master.json \
  --as-of YYYY-MM-DD
```

Observer mode reads this canonical v2 store independently of the legacy
concept/industry mapping:

```bash
export THEME_MEMBERSHIP_V2_ENABLED=1
export THEME_MEMBERSHIP_V2_PATH=private/theme_knowledge/theme_membership.v2.jsonl
export THEME_MEMBERSHIP_V2_REQUIRED=0
export THEME_MEMBERSHIP_V2_EXPECTED_SHA256=
```

Before any future formal activation, set `REQUIRED=1` and pin
`EXPECTED_SHA256` to the reviewed canonical file. Formal mode rejects an
unpinned, non-0600, stale, inactive, or incomplete membership store. A latest
revocation/tombstone wins over an older active revision and can break a theme
immediately.

Word, Markdown, JSON, and local Notion exports enter PE/VC knowledge as drafts.
Approval sets formal availability to the later of source availability and
approval time, preventing backdated knowledge from entering a PIT replay.
Initialize the private approval key once, before the first approval, and back it
up securely; the command refuses to replace an existing key or initialize over
an existing canonical store.

```bash
PYTHONPATH="$PWD" python scripts/manage_pevc_knowledge.py init-key

PYTHONPATH="$PWD" python scripts/manage_pevc_knowledge.py import-draft \
  private/imports/thesis.docx --source-type word

PYTHONPATH="$PWD" python scripts/manage_pevc_knowledge.py approve \
  private/theme_knowledge/drafts/<draft>.json \
  --expected-draft-hash <reviewed-draft-sha256>

PYTHONPATH="$PWD" python scripts/manage_pevc_knowledge.py validate \
  --as-of YYYY-MM-DD
```

Approvals are bound to the reviewed draft archive by a chained HMAC ledger and
a recoverable mode-0600 transaction WAL. Approved `(thesis_id, version)` rows
are immutable; changed content requires a new version. Backdated migration is
not a normal approval path: it additionally requires a real mode-0600 evidence
file and expected byte hash, and it still cannot make the thesis available
before the transaction's recorded Asia/Shanghai business date.

Evaluate the observer from a local Theme snapshot. `formal_pool` remains empty
at pre-ranking time; only a post-Bayesian/RiskGuard/PortfolioConstructor
reconciliation artifact may produce it.

```bash
PYTHONPATH="$PWD" python scripts/run_theme_protocol_v2.py \
  --theme-snapshot private/replay/theme_snapshot.json \
  --as-of YYYY-MM-DD \
  --markov-regime 趋势上涨 \
  --output private/replay/theme_protocol_v2.json
```

Do not pass `--formal-enabled` during observer accumulation. Formal activation
also requires fresh PIT membership, industrial evidence, crowding and valuation
risk axes, positive after-cost edge/BUY, RiskGuard, PortfolioConstructor, 20
distinct live-shadow trading days, a complete verified joint manifest, and
PortfolioConstructor metadata proving that the Theme cap was actually applied.
The current branch intentionally has no canonical five-path replay producer,
so runtime returns `canonical_joint_replay_producer_not_implemented` before
reading any caller-provided manifest. A self-hashed hand-written manifest can
never lift that blocker.

When those gates are ready, the formal scheduler must enable reconciliation
persistence together with the formal switch. If the private `0600` artifact
cannot be written and read back, post-control reconciliation clears the formal
pool and reports blocked.

## 3. Weekly Factor report and month-end transition

Run weekly mining and health checks without apply flags. Both commands consume
strict local Parquet and produce review artifacts only.

```bash
SOURCE_REPO=<read-only-source-repo>
PYTHONPATH="$PWD" python scripts/daily_factor_mining_automation.py \
  --data-root "$SOURCE_REPO/data"

PYTHONPATH="$PWD" python scripts/factor_health_automation.py \
  --cadence weekly --data-root "$SOURCE_REPO/data" --fresh-evaluation
```

Production Quant runtime is independently fail-closed. It requires matching v2
protocol and production-set metadata, explicit slot/family identities, valid
20%/35% risk budgets, and readback-bound canonical evidence. The current
one-factor baseline is therefore `governance_blocked` with confidence zero; a
non-empty selectable count is not a readiness signal.

To reconstruct the historical old-14 comparison, first create a private
self-hashed manifest from an explicit reviewed list (repeat `--factor` for all
14 records), then pass it explicitly to the read-only selection shadow:

```bash
PYTHONPATH="$PWD" python scripts/build_factor_historical_shadow_manifest.py \
  --registry-path quant_investor/factor_registry/mined_factors.json \
  --baseline-id old14-reviewed-20260712 \
  --factor <factor-1>=0.05 \
  --factor <factor-2>=0.05 \
  --output-json private/factor/old14_manifest.json

MYQUANT_MARKET_DATA_BACKEND=parquet \
MYQUANT_MARKET_DATA_MODE_POLICY=strict \
CN_MARKET_DATA_DIR="$SOURCE_REPO/data/cn_market_full" \
PYTHONPATH="$PWD" python scripts/run_quant_factor_selection_shadow.py \
  --historical-baseline-manifest private/factor/old14_manifest.json \
  --expected-production-factor-count 14
```

The loader hashes every raw registry record and never changes the formal
registry. Omitting the manifest or changing any record/hash/count blocks the
run. This shadow is Quant-rank measurement only, has confidence zero, and is
not the missing canonical full-DAG producer.

The legacy Factor replay command is a report-only JSON normalizer. The PR4
canonical local producer separately binds exact artifact bytes and verifies
private immutable receipt/bundle readback. A successful local byte readback is
an integrity result only: it does not authenticate a canonical producer and it
does not authorize production apply. Both paths therefore remain
`production_apply_eligible=false`.

```bash
PYTHONPATH="$PWD" python scripts/build_factor_governance_replay_evidence.py \
  --full-chain-replay-json private/factor/full_chain_replay.json \
  --output-json private/factor/governance_evidence.json
```

For PR4 local byte readback, first create an owner-only `0700` private root.
Supply one exact draft graph to publish its immutable bundle and registry-SHA
receipt; omit `--draft-path` to verify only the exact current receipt. Neither
form grants production authority:

```bash
PYTHONPATH="$PWD" python scripts/build_factor_governance_canonical_replay.py \
  --private-root "$PWD/private/factor/canonical_replay" \
  --registry-path "$PWD/quant_investor/factor_registry/mined_factors.json" \
  --draft-path "$PWD/private/factor/canonical_replay_draft.json"

PYTHONPATH="$PWD" python scripts/build_factor_governance_canonical_replay.py \
  --private-root "$PWD/private/factor/canonical_replay" \
  --registry-path "$PWD/quant_investor/factor_registry/mined_factors.json"
```

Every `path` field inside the draft and every referenced manifest must likewise
be a normalized absolute path. Relative paths fail closed and are never
resolved from the current working directory.

This replay schema has one PIT membership snapshot, therefore `as_of` must
equal `window_end`; do not use a later review-date PIT set for an earlier
window. The existing private root must be `0700`. Before any named publish path
is created, unique file/directory probes must prove that the active umask
preserves exact `0600`/`0700` modes. A failed probe creates no `bundles/`,
`receipts/`, lock, temp, or final path; any random probe orphan from a crash is
unselected and must not be scanned.

The publisher serializes each exact destination with a persistent owner-only
lock file and uses a deterministic destination/content-hash temp name. This is
what makes a prepared temp and a post-link `nlink=2` crash state recoverable
without directory discovery. After an ambiguous failure, rerun only the exact
same draft or run the no-`--draft-path` exact receipt verification shown above.
Do not scan, glob, choose `latest`, or manually remove/replace a final bundle or
receipt. The failed call may already be a correct exact commit, or it may have
left a correct orphan in a renamed directory; exact verification is the only
supported reconciliation and still does not authorize production apply.
Receipt and bundle finals must remain exact compact, sorted canonical JSON with
one trailing newline; never reformat and rebind their hashes.

Forward apply is hard-blocked with
`forward_factor_apply_not_authorized_pr4`. The command below is an
adversarial/fail-closed check only: it must exit non-zero immediately after
argument parsing, before semantic validation, evidence/mining/report I/O, path
conversion, registry reads, WAL creation, or monthly-budget reservation. A
rollback of an already-existing valid inverse WAL remains available and never
refunds that month's budget.

```bash
PYTHONPATH="$PWD" python scripts/daily_factor_mining_automation.py \
  --data-root data \
  --apply-governed-transitions \
  --protocol-version v2 \
  --expected-protocol-hash <protocol-sha256> \
  --governed-evidence-json private/factor/governance_evidence.json \
  --mutation-budget-ledger private/factor/monthly_mutation_ledger.jsonl
```

Any apply request must return non-zero with the fixed PR4 blocker. Do not edit
the transition envelope, valid trading days, arm deltas, mutation plan, WAL, or
registry by hand. Do not enable producer control by configuration, environment
variable, or runtime monkeypatch: dynamic control is not an authorization
surface and is excluded from the protocol hash.

PR5 performance and fault evidence is also local-only. Run the full reference
performance audit explicitly; it is not part of ordinary daily automation and
may take roughly 20 minutes on the reference Apple Silicon host:

```bash
PYTHONPATH="$PWD" ./.venv/bin/python scripts/run_quant_runtime_performance_audit.py
```

Accept only one final JSON line with `status=pass`, `reference_profile=true`,
`reference_acceptance_eligible=true`, no blockers, exact
`production_runtime_input_sha256=2` / `validate_production_frames=1` observed
calls, zero calls on every enumerated guarded apply/socket/urllib/subprocess
surface, native current-RSS common-baseline peak
increment at or below 128 MiB, and unchanged canonical registry SHA/mode/link
count. The default profile is 5,520 symbols with 5/14-factor workloads, one
timed warmup per operation and three samples; an untimed native-memory preflight
runs one 14-factor digest and one validation first and is disclosed separately.
Custom smaller dimensions or relaxed budgets are smoke and diagnostic runs
only; they remain `reference_acceptance_eligible=false` even when
`status=pass`. Even a reference pass
is not an activation receipt, does not authenticate a producer, and cannot
lift `forward_factor_apply_not_authorized_pr4`.

The guarded-surface counters are deliberately non-exhaustive negative evidence;
they do not claim to intercept every possible Python networking or write path.

The deterministic PR5 fault matrix lives in
`tests/unit/test_pr4_replay_publisher_fault_matrix.py`. It exercises exact
idempotence, byte drift, unsafe filesystem objects, publisher crash/retry
boundaries, and all forward-apply entry points while guarding the production
registry SHA, `0644` mode and `nlink=1` in every scenario.

Rollback is dry-run by default and binds the current registry, inverse WAL,
transition, mutation, evidence, protocol, and append-only budget ledger hashes.
After reviewing the dry-run result, repeat with `--apply-rollback` and a new
rollback WAL path. The mutation-budget reservation remains consumed.

```bash
PYTHONPATH="$PWD" python scripts/rollback_factor_governance_transition.py \
  --registry-path quant_investor/factor_registry/mined_factors.json \
  --inverse-wal private/factor/inverse_wal.json \
  --mutation-budget-ledger private/factor/monthly_mutation_ledger.jsonl \
  --protocol-version v2 \
  --expected-protocol-hash <protocol-sha256> \
  --expected-current-registry-sha256 <registry-sha256> \
  --expected-inverse-wal-sha256 <inverse-wal-sha256> \
  --expected-transition-hash <transition-sha256> \
  --expected-mutation-plan-hash <mutation-plan-sha256> \
  --expected-evidence-hash <evidence-sha256>

# Only after the dry-run output is reviewed:
PYTHONPATH="$PWD" python scripts/rollback_factor_governance_transition.py \
  --registry-path quant_investor/factor_registry/mined_factors.json \
  --inverse-wal private/factor/inverse_wal.json \
  --mutation-budget-ledger private/factor/monthly_mutation_ledger.jsonl \
  --protocol-version v2 \
  --expected-protocol-hash <protocol-sha256> \
  --expected-current-registry-sha256 <registry-sha256> \
  --expected-inverse-wal-sha256 <inverse-wal-sha256> \
  --expected-transition-hash <transition-sha256> \
  --expected-mutation-plan-hash <mutation-plan-sha256> \
  --expected-evidence-hash <evidence-sha256> \
  --rollback-wal private/factor/rollback_wal.json \
  --apply-rollback
```

## 4. Holdout seal and joint activation gate

Create the threshold seal once, after validation and before opening holdout. The
seal writer derives the only accepted private path from the dataset SHA, records
it in a hash-chained `0600` canonical ledger, and refuses a second seal. An
operator cannot select an alternate seal path. The ledger is bound to the fixed
`myquant-v13.1-freeze-exception` cycle and permits exactly one dataset entry;
changing the caller-supplied dataset hash cannot create a fresh threshold seal.

```bash
PYTHONPATH="$PWD" python scripts/seal_v13_1_holdout_thresholds.py \
  --thresholds-json private/replay/thresholds.json \
  --dataset-sha256 <pit-dataset-sha256> \
  --validation-end-date YYYY-MM-DD
```

Each of the five scenario entries and each live-shadow observation is a JSON
artifact reference with an expected SHA-256. The verifier re-reads every
artifact, checks dataset/split/trading-date/protocol hashes, and recomputes
metric pass against the sealed thresholds. A self-declared `passed=true` is
ignored. These checks are still attestations, not a substitute for actually
executing the five historical DAG paths, so the current branch hard-blocks
Theme/Factor/joint production activation with
`canonical_joint_replay_producer_not_implemented` until that producer exists.

```bash
PYTHONPATH="$PWD" python scripts/run_v13_1_joint_replay_gate.py \
  --run-id <run-id> \
  --trade-dates-json private/replay/trade_dates.json \
  --theme-shadow-dates-json private/replay/theme_shadow_index.json \
  --thresholds-json private/replay/thresholds.json \
  --expected-threshold-seal-sha256 <seal-file-sha256> \
  --expected-threshold-seal-ledger-sha256 <seal-ledger-sha256> \
  --dataset-sha256 <pit-dataset-sha256> \
  --protocol-hashes-json private/replay/protocol_hashes.json \
  --scenario-results-json private/replay/scenario_index.json \
  --acceptance-json private/replay/acceptance_evidence.json \
  --expected-acceptance-sha256 <acceptance-file-sha256> \
  --open-holdout \
  --expected-threshold-hash <sealed-threshold-hash> \
  --output private/replay/joint_manifest.json
```

Required future producer scenarios are `industry_baseline`, `theme_v2_observer`,
`theme_v2_formal_gate`, `factor_protocol_v2`, and
`theme_v2_plus_factor_v2`. Dashboard validation remains independently usable;
Theme, Factor, and the joint path remain disabled until the canonical producer,
all evidence gates, and 20 live-shadow days exist. Merge still requires
Maxwell's explicit confirmation.

## 5. Verification and stop condition

```bash
CN_MARKET_DATA_DIR=<read-only-canonical-cn-root> \
PYTHON=<repo-python> \
bash scripts/v13_1_freeze_exception_quality_gate.sh
```

The quality gate runs focused Dashboard/Theme/Factor/joint tests, JavaScript
syntax and contract tests, full Theme and Factor sweeps, public CLI/package
smoke, staged-upgrade compatibility, and diff hygiene. It must use the isolated
branch's `PYTHONPATH`; an editable virtual environment may otherwise import the
original dirty worktree.

Stop without enabling the affected switch if any of these remain: Dashboard P0
or reconciliation blocker, Theme PIT/coverage/valuation/crowding/20-day shadow
blocker, Factor transition/rollback/idempotence blocker, strict-Parquet/DAG
replay failure, threshold/evidence/hash mismatch, or
`forward_factor_apply_not_authorized_pr4`,
`canonical_joint_replay_producer_not_implemented`, or a production Quant
runtime status other than `ready` (including the current `governance_blocked`).
