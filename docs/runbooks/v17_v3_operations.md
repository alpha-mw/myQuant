# V17 v3 Offline and Formal-Research Operations

This runbook covers only the explicit `quant-investor-v17-v3` entrypoint.
`quant-investor`, `market analyze`, `market run`, schedules and Dashboard
continue to use V15.

No command in this runbook may call a provider, LLM, broker, order or trade
surface.

## Phase A: verify the additive implementation

Run package, contract and namespace verification:

```bash
./.venv/bin/python scripts/run_v17_v3_phase_a_gate.py
./.venv/bin/quant-investor-v17-v3 verify
```

Run the focused offline tests:

```bash
./.venv/bin/python -m pytest \
  tests/unit/test_v17_v3_contract.py \
  tests/unit/test_v17_v3_schemas.py \
  tests/unit/test_v17_v3_algorithms.py \
  tests/unit/test_v17_v3_runtime.py \
  tests/unit/test_v17_v3_activation.py \
  tests/unit/test_v17_v3_cli.py \
  tests/unit/test_v17_v3_boundary.py \
  tests/integration/test_v17_v3_phase_a_gate.py -q
```

Then run the existing staged gate and V15 regression checks:

```bash
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
./.venv/bin/python -m pytest tests/unit/test_public_package_smoke.py -q
./.venv/bin/python -m pytest tests/unit/test_v15_run_readiness.py -q
./.venv/bin/quant-investor market --help
```

Phase A is complete only when v2 protected bytes and all V15 public behavior
remain unchanged. The expected delivery state is:

```text
NOT_ACTIVATED_DATA_BLOCKED
```

That state is not an activation failure and must not be relabelled as a formal
research result.

## Validate a pre-positioned sealed source closure

Phase A does not acquire, download or copy provider data. An operator must
pre-position the exact locator and its closure under the governed source root
in an owner-private workspace. The validation command accepts only a governed
workspace-relative locator path and its expected byte SHA:

```bash
./.venv/bin/quant-investor-v17-v3 admit-sources \
  --workspace-root /owner/private/v17_v3 \
  --source-locator data/private/v17_v3_sources/example/locator.json \
  --expected-source-locator-sha256 <sha256-of-stored-locator-bytes>
```

Analysis never accepts free-form source arrays. It reloads the locator by path
and expected SHA and derives every input from the admitted roles.

## Build the staged Quant initial pool

The source graph is deliberately acyclic. First admit a PRESELECT locator that
contains only the RAW closure and typed Quant preselection inputs, then persist
its deterministic pool:

```bash
./.venv/bin/quant-investor-v17-v3 build-initial-pool \
  --workspace-root /owner/private/v17_v3 \
  --source-locator \
    data/private/v17_v3_sources/locators/example/preselect.json \
  --expected-source-locator-sha256 <sha256-of-preselect-locator-bytes>
```

The command returns the immutable pool path and SHA. The later ANALYZE locator
must bind that exact pool and both branch outputs, and must identify the
PRESELECT locator through `preselection_locator_ref`. Branches bind the staged
PRESELECT locator and pool; they never bind the later locator that contains
them. Runtime replays preselection and rejects any byte or semantic drift.

## Shadow analysis

```bash
./.venv/bin/quant-investor-v17-v3 analyze \
  --mode shadow \
  --workspace-root /owner/private/v17_v3 \
  --source-locator data/private/v17_v3_sources/example/locator.json \
  --expected-source-locator-sha256 <sha256-of-stored-locator-bytes>
```

If no valid fusion-promotion receipt exists, shadow output may use only the
explicitly labelled `UNCALIBRATED_50_50` comparator. It cannot be copied,
wrapped or promoted into the formal-research root.

## Build a current model-only shadow closure

Provider acquisition and v3 compilation are separate operations. The compiler
below is offline: it consumes already captured evidence plus strict canonical
Parquet and never reads a token or calls a provider.

```bash
./.venv/bin/python scripts/build_v17_v3_current_shadow.py \
  --workspace-root /owner/private/v17_v3 \
  --repo-root /absolute/path/to/myQuant \
  --phase1-run-root /owner/private/v17-phase1-run \
  --market-pointer /absolute/path/to/data/parquet/cn/_latest.json \
  --expected-market-pointer-sha256 <sha256> \
  --factor-readiness /absolute/path/to/results/factor_governance/readiness.json \
  --expected-factor-readiness-sha256 <sha256> \
  --bak-basic-acquisition-manifest /owner/private/acquisition_manifest.json \
  --expected-bak-basic-manifest-sha256 <sha256> \
  --cutoff <current-second-precision-UTC-instant> \
  --strategy-id cn-full-a-v17-v3-model \
  --run-id <unique-run-id>
```

The compiler requires the canonical pointer's
`latest_complete_trade_date` to equal the last completed admitted CN open
session at the supplied cutoff. It validates source hashes, Parquet schemas,
primary-key uniqueness, source availability, PIT membership and READY
coverage. A stale or malformed closure returns `DATA_BLOCKED`; it never
substitutes cached, inferred or mock rows.

This route uses `factor_baseline_mode=PROVISIONAL_RESEARCH` and
`portfolio_basis=MODEL_ONLY_NO_PRIVATE_HOLDINGS`. Neither label is Factor v4
production readiness or account holdings. When Deep evidence is absent, the
compiler creates one explicit unavailable declaration for every Top24 name;
runtime converts each declaration to `BUY_VETO` with a zero target and no
backfill. A successful all-veto result is therefore `SHADOW_COMPLETE` with
100% model cash, not a buy list.

The terminal records Macro and Markov separately as `APPLIED` or
`UNAVAILABLE_NO_OP`. It must not claim an overlay ran when its exact typed
evidence is absent. Formal analysis and activation reject provisional-factor
or model-only artifacts before any formal write.

## Daily V15/V17 gray observation

The CN daily V15 review discovers completed current-shadow workspaces under
`data/private/v17_v3_workspaces` after the authoritative V15 record has been
written. Discovery is read-only. It does not invoke this compiler, a provider
or any V17 state transition.

Only an exact V15/V17 pair with the same decision session and the same strict
CN market-pointer bytes is classified `COMPARABLE`. A missing or stale V17
shadow is recorded as `GRAY_UNAVAILABLE` and leaves the V15 review successful
and unchanged. The per-run outputs are
`v15_v17_gray_comparison.{json,md}` inside the V15 strategy-record directory.

The comparison reports rank overlap, holding coverage and exposure differences.
Schema v2 also records every current V15 holding against the exact V17
preselection, Quant branch, Fundamental branch and Fusion artifacts. Scores are
shown on a 0-100 display scale together with same-pool percentiles, calibration
label, Top24/Deep state and explicit blockers. A holding that did not enter the
Quant Top500 remains `UNAVAILABLE_PRESELECTION_NOT_SELECTED` for both branches;
the gray report must not bypass the Quant-first architecture to invent scores.

It also matures prior rank-set returns at 1, 5 and 20 local sessions when exact
strict-Parquet closes exist. The current v2 reader retains v1 history when
building those cumulative labels. These are equal-weight rank diagnostics only.
`SHADOW_COMPLETE` with 24 `BUY_VETO` rows and 100% model cash is an unavailable
Deep-evidence state, not a defensive or performance win. V15 remains the only
production/default result and no comparison can authorize a broker, order or
trade.

## Fusion calibration

Calibration is offline and reads only the admitted PIT closure:

```bash
./.venv/bin/quant-investor-v17-v3 calibrate-fusion \
  --workspace-root /owner/private/v17_v3 \
  --source-locator data/private/v17_v3_sources/example/locator.json \
  --expected-source-locator-sha256 <sha256-of-stored-locator-bytes>
```

Every scheduled training, outer and active-refit month is mandatory. Missing
origin-time factor governance, fewer than 24 common READY securities, missing
labels, a source hash change, all-invalid weights or either failed lower bound
produces `PROMOTION_REJECTED`. It does not create an activation receipt or a
REVOKED receipt because no activation attempt or ACTIVE predecessor exists.

The generated typed receipt binds the exact cutoff, locator, code/resource
hashes, upstream Quant/Fundamental calibration receipts, factor inventories,
month inventories, bootstrap index matrix, promotion statistics and selected
`active_formal_research_weight`. Upstream receipts require nonempty exact
evidence refs and phase-specific locators:
`QUANT_TIMING_CALIBRATION` for Quant timing,
`FUNDAMENTAL_FORWARD_CALIBRATION` for Fundamental forward, and
`FUSION_PROMOTION` for fusion. A PRESELECT locator is not calibration
evidence. Every historical branch must resolve to the same exact initial pool
and PRESELECT locator as its peer branch. The bound package manifest includes
a runtime-build manifest containing every v3 runtime/algorithm Python byte
SHA-256. A rejected calibration produces `PROMOTION_REJECTED`; activation has
not yet occurred.

## Formal-research candidate and activation

Phase A delivery remains `NOT_ACTIVATED_DATA_BLOCKED`. Do not use activation as
a readiness probe because it is a governed state mutation; use `verify` and
the Phase A gate script instead.

After every Phase B stop condition below is cleared, first finalize an
immutable, authority-free formal candidate from the ANALYZE locator:

```bash
./.venv/bin/quant-investor-v17-v3 analyze \
  --mode formal-research \
  --workspace-root /owner/private/v17_v3 \
  --source-locator data/private/v17_v3_sources/example/locator.json \
  --expected-source-locator-sha256 <sha256-of-analyze-locator-bytes>
```

Then activate the exact accepted promotion and exact candidate. Activation is
valid for one strategy and cutoff only:

```bash
./.venv/bin/quant-investor-v17-v3 activate-formal-research \
  --workspace-root /owner/private/v17_v3 \
  --promotion-receipt \
    data/private/v17_v3_runs/<run-id>/fusion_promotion_receipt.json \
  --expected-promotion-receipt-sha256 <sha256-of-promotion-receipt-bytes> \
  --formal-output \
    results/v17_v3_formal_research/strategies/<strategy-id>/runs/<run-id>/formal_output.json \
  --expected-formal-output-sha256 <sha256-of-formal-candidate-bytes>
```

The activation transaction resolves and revalidates the complete transitive
closure before its first authority write, including every scheduled month's
Quant/Fundamental branch, initial-pool and PRESELECT-locator bytes, the three
phase-specific calibration evidence closures and the runtime-build manifest.
`status` repeats the same validation. A complete portfolio research result
still carries no execution, broker, order or trade authority.

## Revoke current formal-research publication

```bash
./.venv/bin/quant-investor-v17-v3 revoke-formal-research \
  --workspace-root /owner/private/v17_v3 \
  --strategy-id <strategy> \
  --cutoff <timezone-aware-cutoff> \
  --expected-active-receipt-sha256 <sha256> \
  --reason <reviewed-reason>
```

Revocation appends an immutable receipt and advances the activation pointer by
CAS. It never edits or deletes the ACTIVE receipt or prior research artifact.
The historical artifact remains directly readable, while `status` and current
consumers must return `NO_CURRENT_ACTIVE_FORMAL_RESULT`. If activation crashed
after writing ACTIVE pointer but before formal latest, revocation CAS-creates the
REVOKED latest tombstone from absent; a newer-cutoff latest is never overwritten.

## Phase B stop conditions

Do not activate when any of the following is true:

- bitemporal membership or origin-time governance receipts are incomplete;
- the machine-derived raw history span is unavailable;
- Quant 1260-session or Fundamental 2520-session calibration is unavailable;
- holdings are missing, from another strategy, or more than one canonical
  session stale;
- any locator, cutoff, pool, branch, policy, resource or package SHA drifts;
- a scheduled calibration month is skipped or has fewer than 24 common READY
  names;
- deterministic core replay bytes differ;
- v2 bytes or V15 public behavior change;
- Deep, holdings, Macro, Markov or LLM evidence influences organic selection;
- an owner-private, symlink, hard-link, mode or CAS check fails.

The current repository does not contain the required historical PIT and fresh
holdings evidence. Until those inputs exist, Phase B remains `DATA_BLOCKED`.
