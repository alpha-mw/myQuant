# v16 Calibration Source v3

Status: **disconnected, source-incomplete, and permanently nonauthorizing**.

This contract adds a prospective source-recomputation lane without changing
the historical calibration metrics, schedule, target, readiness, report, or
Dashboard contracts. It does not authorize Factor activation, v16 activation,
a production/default pointer switch, portfolio construction, orders, or
trading.

## Additive contract chain

The source lane has one version-exact dependency chain:

```text
calibration-universe-plan.v3
  -> evidence-schedule-declaration.v4
  -> target source validation v4
  -> prospective-calibration-source-status.v3
  -> v16_run_readiness.v3
```

Earlier v1, v2, and v3 artifacts remain valid only for their existing
consumers. They cannot be relabeled or adapted into this chain. In particular:

- schedule-v4 binds a calibration plan v3, never a v2 calibration universe;
- target-v4 reopens the plan, schedule, stock source set, stock marks,
  H00300 manifest/table, and cost source status;
- readiness-v3 consumes only calibration source status v3 and schedule
  lineage v4;
- no existing readiness, report, Dashboard, or production consumer imports
  readiness-v3.

## Pre-s0 plan

`v16.calibration-universe-plan.v3` is fixed before observation. It binds:

- the hermetic runtime capsule and exact runtime evidence artifacts;
- exact resolver implementation manifests for `prior_probability` and
  `branch_probability`;
- the ordered frozen model bundle for each of `quant`, `fundamental`, `macro`,
  and `llm`;
- at least 300 prospective samples and eight non-overlapping cohorts for each
  branch, with one common `(slot_id, symbol)` universe;
- unique future paths and schemas for Stage1 request/response, branch status,
  prediction timestamp attempt/receipt, stock marks, cost status, and target
  status;
- exact stock-source-set, H00300 manifest, cost-source, and lambda-training
  references.

Epoch A cannot carry this plan. Epoch B/C may bind it, but the plan contains no
prediction, alpha, interval, lambda, cost, outcome, metric, readiness, or
authorization value.

The stock source-set manifest binds the four source identities once: strict
market Parquet, adjustment factors, PIT membership, and suspensions. Each
sample binds that manifest plus its H00300 manifest. This keeps the prospective
plan below the canonical JSON item bound while preserving transitive byte
identity. The market Parquet ref must use the governed-data root; the three
canonical supporting evidence refs must use the private-evidence root. An
arbitrary or caller-defined root policy is rejected.

## Implemented source recomputation

Only two values currently have a repository-defined source algorithm:

1. `prior_base_rate` is recomputed from the byte-bound base-rate training
   evidence in `PosteriorRuntimeBundle`.
2. `calibrated_probability` is recomputed by deterministic Stage1 formal-row
   replay plus the byte-bound likelihood calibration store.

Their resolver identity binds the manifest, raw module source, runtime capsule,
and source-tree component. The resolver execution itself is not yet connected
to an independently attested execution authority, so both values retain exact
`calibration_resolver_execution_binding_not_integrated` blockers.

Every branch prediction status is independently RFC3161-bound after `s0` close
and before `s1` open. Its artifact and timestamp paths must equal the paths
predeclared in the plan.

## Unsupported requirements

The repository currently contains no source-backed, branch-only algorithm for
the following requirements:

| Requirement ID | Missing governed decision |
| --- | --- |
| `branch_only_alpha_interval_model` | Formula and training contract for branch alpha and prediction interval |
| `fold_training_algorithm` | Prospective fold construction, fit rule, and lambda selection algorithm |
| `eight_component_cost_model` | Source resolver for the exact eight ordered CN cost components |

These are not implementation defaults. Until a separately reviewed policy and
source contract exists, the implementation must not infer them from posterior
metrics, accept caller-populated numbers, borrow v2 values, or synthesize a
formula. The exact blockers are:

```text
calibration_prediction_requirement_unsupported:branch={branch}:requirement=branch_only_alpha_interval_model
calibration_lambda_requirement_unsupported:branch={branch}:fold={fold_id}:requirement=fold_training_algorithm
calibration_cost_requirement_unsupported:sample={sample_id}:requirement=eight_component_cost_model
calibration_target_outcome_blocked:sample={sample_id}:dependency=eight_component_cost_model
calibration_source_recomputation_incomplete
```

Target-v4 may validate stock and benchmark boundary sources, but it must stop
before an outcome while the cost model is unsupported. A cost status carries
source refs and a blocker only; it contains no cost vector.

## Fail-closed readiness

`v16_run_readiness.v3` always emits:

```text
activation_candidate=false
new_risk_authorized=false
readiness_status=no_new_risk
broker_side_effects=false
```

It fixes the formal branches to `quant`, `fundamental`, `macro`, and `llm` at
`0.25` each. Retrieval is evidence-only with no score or weight, and
RiskAdvisor is advisory-only. Blockers are deduplicated for the summary while
`blocker_sources` remains an ordered list, so identical blocker text from
multiple samples or gates cannot overwrite provenance.

Even after source recomputation is complete, readiness-v3 cannot become an
authorizing schema. A later reviewed schema would still need external calendar
capture/transport attestation, Codex authority v2, Dashboard activation receipt
v2, global attempt registry and anti-rollback authority, complete Factor-v4
evidence, and an explicit production pointer-switch authorization.

## Purity and operations

All modules in this chain are pure builders or validators. They do not perform
path discovery, writes, network access, provider calls, LLM calls, candidate
generation, portfolio construction, broker access, orders, or trades. Evidence
loading remains an explicit caller responsibility through byte-bound artifact
objects.

Rollback for this additive slice means removing only the new v3/v4 source-lane
modules, tests, and this document after an explicit user decision. Existing
evidence, readiness, Factor, report, Dashboard, registry, and production files
must not be reset or relabeled.
