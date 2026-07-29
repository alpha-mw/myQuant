# V17 v4 Shadow Operations

V17 v4 is available as an explicit, research-only Shadow lane. V15 remains
the production/default protocol. None of the commands below calls a provider,
LLM, broker, order, execution, or trade API.

## Current dynamic Shadow flow

```text
PIT/data gate
  -> Quant cross-sectional preselection
  -> monthly-audited research factor-set pointer
  -> same-pool dynamic Quant + Fundamental branches
  -> UNCALIBRATED_FORWARD_ACCUMULATING Fusion Top24
  -> owner-supplied Deep v3 official-evidence compiler
  -> immutable Shadow run + session ref
  -> explicit-ref V15/v4 daily gray comparison
```

The current research factor set is data, not a list frozen into the daily
automation or runtime. The monthly Factor audit owns a separate research-only
CAS pointer. Each immutable set binds its exact literature-incubator catalog
definitions, implementation/resource hashes, selection audit, previous set,
selection cutoff, and first effective Shanghai session. Rotation never writes
the Factor production registry, production active set, production-control
receipt, selector, or V15 route.

A set published before `15:00 Asia/Shanghai` on an open session may use that
session's close as its first forward decision. The immutable set and pointer
must bind `published_at`, and the selection cutoff must not be later than that
publication time. Same-session CAS publication also requires `published_at` to
be no more than five minutes behind the publisher process clock. A publication
at or after the close starts on the next open session. Same-session activation
never labels the interval before publication: the first 1/5/20-session labels
use only future Shanghai closes after the decision close. Legacy sets without
`published_at` remain readable but cannot claim same-session effectiveness.

The daily compiler reads the research pointer and set by exact path and SHA,
then rereads both before publishing either the prediction observation or the
Shadow session. A pointer race blocks publication. The selected set contains
one factor per catalog `slot`, at most eight factors, and may change only
through the monthly audit. A rotation starts a new diagnostic lineage; samples
from different set or policy hashes are never pooled.

Zero accumulated sessions do not block Fusion. The bootstrap policy uses
Quant/Fundamental weights `0.5/0.5` and is labelled exactly
`UNCALIBRATED_FORWARD_ACCUMULATING`. It is not a calibrated receipt and is
permanently ineligible for formal activation, canary evidence, performance
evidence, or policy promotion. One immutable prediction observation is written
per decision session. Separate 1/5/20-session matured-label receipts are
written only when their future Shanghai sessions close; there is no historical
backfill.

Deep v3 and the new Shadow artifacts are explicitly `shadow_only=true`,
`formal_activation_eligible=false`, and `canary_evidence_eligible=false`.
Formal activation, portfolio, eligibility, and canary continue to accept the
existing formal v1 closure only. Missing owner-prepositioned Deep assessment
bytes produce blocker readiness with `model_output_present=false`; they never
trigger an automatic assessment, LLM call, or fabricated `SHADOW_COMPLETE`.

The offline preparation compiler never invokes maintenance or a provider. A
separate preflight may classify `TRUE_CURRENT_CANONICAL_INPUT_GAP`; only that
classification may route to an already-authorized market, Fundamental, or
Macro maintenance workflow, followed by strict storage validation. Missing
forward history or Deep assessment bytes are not canonical-data gaps.

## Build the real Forward Evidence source snapshot

After the canonical close is fully sealed, bind the exact bytes that were
audited. Do not reuse a SHA after any pointer or strategy-universe file
changes:

```bash
quant-investor-v17-v4 build-source-snapshot \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id cn-aggressive-tech-manufacturing \
  --decision-session <YYYY-MM-DD> \
  --cutoff <UTC-second-timestamp> \
  --market-pointer-sha256 <sha256> \
  --fundamental-pointer-sha256 <sha256> \
  --factor-set-pointer-sha256 <sha256> \
  --strategy-universe-path <full_metrics.parquet> \
  --strategy-universe-sha256 <sha256> \
  --strategy-universe-manifest-path <breadth.json> \
  --strategy-universe-manifest-sha256 <sha256>
```

Exit `0` and `status=READY` mean that `source_locator.json` was validated,
written last, and read back. Exit `2` with
`status=TRUE_CURRENT_CANONICAL_INPUT_GAP` means no completion locator was
published. The command is offline and has no Factor Governance, activation,
selector, provider, broker, execution, order, or trade authority.

Inspect the current monthly-rotating set and the additive Deep/Shadow v3
surfaces with explicit paths and hashes:

```bash
quant-investor-v17-v4 factor-set-status \
  --workspace-root /absolute/path/to/myQuant

quant-investor-v17-v4 deep-v3-compile \
  --workspace-root /absolute/path/to/myQuant \
  --assessment-manifest-path <owner-prepositioned-manifest> \
  --assessment-manifest-sha256 <sha256> \
  --created-at <UTC-second-timestamp>

quant-investor-v17-v4 forward-shadow-status \
  --workspace-root /absolute/path/to/myQuant \
  --session-ref-path <exact-session-ref-path> \
  --session-ref-sha256 <sha256>
```

When an exact current run cannot proceed, `forward-shadow-readiness` may write
only blocker readiness. It requires explicit blocker codes and cannot create
model output or a session ref.

## Legacy fixed-trio compatibility lane

The v1 Quant and v2 Shadow contracts below remain byte-compatible for replay
of already-created artifacts. They are not the current monthly-rotating lane
and must not be used by the daily automation to freeze future factor choices.

## 1. Replay the legacy research Quant branch

The legacy v1 branch compiled exactly these three literature-incubator
factors:

- `cn_fip_continuous_direction_12m`
- `cn_low_market_adjusted_tail_asymmetry_252d`
- `cn_low_total_skewness_20d`

They are evaluated as full-universe PIT cross-sectional ranks and combined by
an arithmetic mean. The resulting branch seals every per-symbol factor value,
the composite score, the factor-definition hash, the frozen-policy hash and
the exact market-slice reference. `cn_low_left_tail_var1_250d` and the other
stopped candidates are excluded.

Prepare a canonical Parquet slice below `data/private/v17_v4_sources/` with
exactly `symbol`, `trade_date`, `adj_close`, and `available_at`. It must cover
the full ranking universe through the decision session; each `available_at`
must be no later than the decision cutoff.

```bash
quant-investor-v17-v4 quant-compile \
  --workspace-root /absolute/path/to/myQuant \
  --run-id <lower-case-hyphenated-run-id> \
  --output-id <quant-output-id> \
  --initial-pool-path data/private/v17_v4_runs/<run_id>/initial_pool.json \
  --initial-pool-sha256 <sha256> \
  --market-slice-path data/private/v17_v4_sources/<capture>/quant_market.parquet \
  --market-slice-sha256 <sha256>
```

The command writes
`data/private/v17_v4_runs/<run_id>/research_quant_branch.json`. Missing
history, stale pool coverage, a post-cutoff row, a missing factor value, or
non-deterministic replay blocks the entire Quant branch.

This artifact is `shadow_only=true`,
`formal_activation_eligible=false`, and `canary_evidence_eligible=false`.
The formal calibration path still accepts only
`myquant.v17.v4.branch-output.v1`; therefore these research factors cannot be
silently relabeled as Factor v4 production factors. Formal use requires their
own Factor v4 activation closure and a separate production-branch migration.

## 2. Replay legacy Deep v2

Preposition the canonical assessment manifest and every official raw byte
below `data/private/v17_v4_sources/deep_raw/`. Each assessment must cover the
Fusion Top24 in rank order and contain either the fixed ten modules or an
explicit blocker-only `UNAVAILABLE` row.

```bash
quant-investor-v17-v4 deep-compile \
  --workspace-root /absolute/path/to/myQuant \
  --assessment-manifest-path data/private/v17_v4_runs/<run_id>/deep_assessment_manifest.json \
  --assessment-manifest-sha256 <sha256> \
  --created-at <UTC-second-timestamp>
```

The compiler performs no collection and no model call. It binds raw bytes,
publisher/availability times, parser identity and parser SHA, compiles the ten
module dossier, applies the packaged scoring policy, and replays the complete
closure before exact-once publication.

## 3. Replay legacy Factor-gated Shadow

Run `shadow-publish` with the exact active-set and production-control receipt
from Factor Governance v4 plus every current model input. The strategy ID is
a single lower-case hyphenated path component, for example
`cn-aggressive-tech-manufacturing`.

If either Factor reference is absent or the full transaction closure does not
replay, the command writes only a blocker readiness artifact. It does not read
or publish model output. A successful run rereads the Quant market slice,
recomputes all three research factors, recomputes same-pool Fusion rankings
and Deep v2 results before writing immutable run and session-ref artifacts.
A plain `branch-output.v1` Quant file or a score-only substitute is rejected.

An operator may make a single explicit Shadow-only exception for the fixed
research trio by omitting all four formal Factor path/SHA arguments and
supplying
`--research-factor-shadow-only-override-id <lowercase-canonical-id>`.
The command then writes an immutable
`research-factor-shadow-assertion.v1`, `shadow-run.v2`, and
`shadow-session-ref.v2`. The assertion binds the override ID exact-once to the
strategy, cutoff, decision session, run ID, fixed trio, and factor-policy SHA.
Reusing the ID with different bytes fails. Supplying any formal Factor
argument together with the override also fails.

This exception does not create Factor production evidence. The v2 run has no
production-factor inventory, active-set reference, or production-control
receipt, and remains `shadow_only=true`,
`formal_activation_eligible=false`, and `canary_evidence_eligible=false`.
Formal activation, canary, V15/default routing, broker, order, execution, and
trade surfaces do not accept the assertion or either v2 Shadow artifact.

```bash
quant-investor-v17-v4 shadow-publish \
  --workspace-root /absolute/path/to/myQuant \
  --readiness-id <readiness_id> \
  --shadow-run-id <run_id> \
  --strategy-id cn-aggressive-tech-manufacturing \
  --cutoff <UTC-second-timestamp> \
  --decision-session <YYYY-MM-DD> \
  --created-at <UTC-second-timestamp> \
  --factor-active-set-path <path> \
  --factor-active-set-sha256 <sha256> \
  --factor-control-receipt-path <path> \
  --factor-control-receipt-sha256 <sha256> \
  --source-locator-path <path> \
  --source-locator-sha256 <sha256> \
  --initial-pool-path <path> \
  --initial-pool-sha256 <sha256> \
  --quant-branch-path <path> \
  --quant-branch-sha256 <sha256> \
  --fundamental-branch-path <path> \
  --fundamental-branch-sha256 <sha256> \
  --fusion-top24-path <path> \
  --fusion-top24-sha256 <sha256> \
  --deep-bundle-path <path> \
  --deep-bundle-sha256 <sha256> \
  --holdings-snapshot-path <path> \
  --holdings-snapshot-sha256 <sha256>
```

For the explicit research-only exception, replace the four
`--factor-*` arguments with:

```bash
  --research-factor-shadow-only-override-id <operator_assertion_id>
```

Read one exact session:

```bash
quant-investor-v17-v4 shadow-status \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id cn-aggressive-tech-manufacturing \
  --decision-session <YYYY-MM-DD> \
  --expected-sha256 <session-ref-sha256>
```

## 4. Daily V15/v4 gray comparison

The daily review no longer scans or discovers V17 v3 workspaces. Supply the
exact v4 session-ref path and SHA:

```bash
python -m quant_investor.monitoring.cn_aggressive_daily_review \
  --v17-v4-workspace-root /absolute/path/to/myQuant \
  --v17-v4-shadow-session-ref results/v17_v4_shadow/strategies/<strategy_id>/sessions/<YYYY-MM-DD>.json \
  --v17-v4-shadow-session-sha256 <sha256>
```

The comparison is `COMPARABLE` only when the V15 manifest contains
`v17_v4_comparison_inputs` with the same decision session and exact
`calendar_ref`, `market_bars_ref`, `holdings_ref`, and `source_closure_ref`
as the Shadow run. Missing or different bindings remain `NON_COMPARABLE`.

Mature 1/5/20-session results are separate immutable
`CLOSE_RETURN_DIAGNOSTIC_ONLY` labels under
`results/v17_v4_shadow/gray_labels/`. They bind the origin comparison,
current market pointer, snapshot manifest, and relevant monthly Parquet
bytes. They are not total-return labels and are never eligible for a
performance or production conclusion.

## 5. Run the additive forward-evidence lane

`run-forward` is the daily research entrypoint for staged, partial Shadow
closure. It does not replace `shadow-publish`, mutate the V15 selector, or
relax the strict Shadow v3 transaction.

Preposition one canonical sealed request at:

```text
data/private/v17_v4_runs/forward_requests/<request_id>.json
```

The request ID is content-derived. It binds the profile, strategy, decision
session, cutoff, source snapshot, factor-set pointer, policy references, and
every supplied stage input. Run it by exact path and byte SHA only:

```bash
quant-investor-v17-v4 run-forward \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/v17_v4_runs/forward_requests/<request_id>.json \
  --request-sha256 <sha256>
```

Available profiles are:

- `EXPLORE`: source, allocation, Quant, and full-universe factor observation
  are required. Fundamental, Fusion, strategy observation, Deep, and holdings
  may be absent.
- `FORWARD_EVIDENCE`: source, Core/Challenger allocation, Quant,
  full-universe factor observation, Fusion, strategy observation, and the
  final immutable session ref are required. Fundamental, Deep, and holdings
  may be absent or partial.
- `RELEASE_CANDIDATE`: delegates to the unchanged strict Shadow v3 closure and
  remains ineligible for formal activation, canary, promotion, or default
  routing.

The command exits `0` only after the selected profile's required receipts and
transitive references replay. A missing required stage, supplied-invalid
optional stage, PIT/future-data failure, SHA/schema mismatch, authority claim,
lineage drift, or factor-pointer race exits `2` and writes no final session
ref. An absent optional stage is recorded as `SKIPPED/UNAVAILABLE`; valid
partial evidence is `SUCCEEDED/PARTIAL`.

Successful daily evidence is discoverable only through:

```text
results/v17_v4_shadow/forward_evidence/strategies/<strategy_id>/sessions/
  <decision_session>/<request_id>.json
```

Stage outputs and receipts may survive a later failure for audit and exact
retry. Do not scan orphan run directories as completed observations.

The status has two independent axes:

```text
default_protocol_state = V15_DEFAULT
global_activation_state = INACTIVE
run_state = FORWARD_EVIDENCE_ACTIVE
research_runtime_default = false
formal_activation_eligible = false
```

`default_protocol_state` identifies V15 as the unchanged default protocol.
`global_activation_state` applies to V17 as a whole. `run_state` applies only
to one successfully closed `FORWARD_EVIDENCE` session and does not imply
production/default activation. The complete scoring, tier, label,
de-duplication, and rollback policy is documented in
`docs/architecture/v17_v4_forward_evidence_runtime.md`.
