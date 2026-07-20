# v16 Publication Source v2

## Status

This is a private, additive, nonauthorizing publication lane for the accepted
v16 evidence-v2 chain. It does not modify or reinterpret:

- `candidate_decision_report.v16`;
- `dashboard_contract.v16`;
- the Dashboard application or generated payload;
- v15 production/default authority;
- any production or activation pointer.

The legacy report and Dashboard contracts are not readiness-v4 authority.

## Artifact Order

`v16.publication-source-plan.v2` binds an immutable readiness-v4 reference and
predeclares its own path plus these outputs under one owner-private run root:

1. `v16.candidate-source-report.v2`
2. `dashboard_contract.v16.evidence-v2`
3. `v16.dashboard-source-status.v2`
4. `v16.publication-aggregate.v2`

The plan and all outputs must be unique direct children of the same canonical
private root. The plan fixes the output order before any output exists.

## Source Report

The report reopens the exact typed readiness-v4 evidence. That validation
transitively recomputes the full-union posterior from bound Stage1, runtime,
formal-branch, and cost artifacts and revalidates the IC, execution, and
handoff source statuses.

The report projects:

- Q/F/M/LLM at exact weight `0.25`;
- canonical `f64:` posterior, interval, branch, existing-weight, target-weight,
  and cash values;
- evidence IDs and detached source/model/cost references;
- retrieval advisory data without score, confidence, probability, or weight;
- exactly one IC action per menu symbol and rationale hashes;
- source-only execution and handoff references;
- every readiness-v4 blocker and blocker-source row.

RiskAdvisor remains advisory-only. Position, reference-price, RiskAdvisor,
Stage2 execution, execution-plan/market-state, human-signature, handoff,
anti-rollback, authorizing-consumer, and Dashboard activation contracts remain
explicit blockers.

## Dashboard Projection

The evidence-v2 Dashboard snapshot copies only the validated source report and
readiness fields. It uses a separate JSON schema and is not loaded by the
current Dashboard application. All readiness and activation fields are fixed
false, including new risk, production apply/pointer, Codex activation,
Dashboard activation, sealed live-human receipt, and broker side effects.

The Dashboard source status adds blockers for missing delivery attestation,
application integration, and the governed Dashboard activation receipt.

## Publication Aggregate

The aggregate is written last. It binds these five inputs by artifact ID,
absolute path, schema, byte SHA-256, semantic SHA-256, root policy, and byte
size:

1. publication plan;
2. readiness-v4;
3. candidate source report;
4. Dashboard evidence-v2 snapshot;
5. Dashboard source status.

`publication_artifact_set_complete=true` means only that this private artifact
set was validated, written, and read back coherently. It is not production
readiness, investment authority, human approval, Dashboard activation, or a
pointer-switch receipt. The aggregate remains `readiness_status=no_new_risk`
with all authority fields false.

## Private Publisher

`publication_bundle_io_v2` is the only writer for the bundle. It:

- requires an existing, canonical, owner-owned mode-0700 run root;
- pins the root directory descriptor and verifies its path/inode identity;
- preflights every target as absent;
- creates files with `O_EXCL | O_NOFOLLOW`, mode 0600;
- fsyncs and stably reads back each file;
- fsyncs the pinned directory after each creation and after the aggregate;
- never removes partial output after a failure.

An interrupted bundle is terminal for that run root. The missing aggregate is
the fail-closed indication of incompleteness, and an existing partial file
prevents same-run retry.

Production verification must not invoke this publisher against `results/v16`.
The implementation tests publish only under pytest temporary directories.

## Numeric Domain

Evidence-v2 canonical JSON forbids native floats. Numeric report and Dashboard
values use the existing canonical finite binary64 encoding. Non-finite values,
noncanonical negative zero, and lossy native-float JSON substitutions fail
closed.

## Side Effects

The builders and validators are pure. They do not call providers, an LLM,
candidate generation, portfolio construction, the legacy Codex workflow,
Dashboard activation, a broker, order creation, trading, canonical
maintenance, or production/default pointer operations.
