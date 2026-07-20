# v15/v16 Authority Compatibility Matrix

This matrix separates current production authority from versioned v16
research and migration contracts. A schema label never upgrades an artifact or
makes it authoritative.

| Contract | Canonical schema / filename | Current role | Accepted consumer |
| --- | --- | --- | --- |
| v15 run readiness | `v15_run_readiness.v2` / `v15_run_readiness.json` | Current production authorization summary; fail closed without every v15 gate and a live human receipt | Existing v15 reports, manifests, and Dashboard v3 |
| v16 readiness v1 | `v16_run_readiness.v1` / `v16_run_readiness.json` | Legacy v16 candidate diagnostic; permanently nonauthorizing while the evidence-v2 migration blockers are mandatory | Candidate report v16 and Dashboard Contract v16 only |
| v16 readiness v2 foundation | `v16_run_readiness.v2` / `v16_run_readiness_v2.json` | Structurally nonauthorizing evidence migration carrier; always `no_new_risk` in this schema | No production/report/Dashboard consumer in the foundation slice |
| evidence schedule v2 | `v16.evidence-schedule-declaration.v2` | Historical validation only | Schedule-v2 target helpers only |
| evidence schedule v3 | `v16.evidence-schedule-declaration.v3` | Required schedule lineage for readiness-v2 evidence | V3-specific target/calibration bindings only |
| candidate report v16 | `candidate_decision_report.v16` / `v16_candidate_decision_report.json` | Existing v1-linked diagnostic report | Existing report readers only |
| Dashboard Contract v16 | `dashboard.contract.v16` | Existing v1-linked diagnostic projection | Existing Dashboard v16 validator only |

The disconnected foundation implementations are
`quant_investor.v16.evidence_v2.factor_carrier` and
`quant_investor.v16.evidence_v2.readiness`. The schedule-v3 target entrypoints
live beside the legacy target helpers in
`quant_investor.v16.evidence_v2.target`; their names end in `_v3` and require
`ScheduleAnchorBindingV3`.

## Hard boundaries

- Readiness v1 must retain
  `evidence_v2_disconnected_from_authorizing_consumers` and
  `global_attempt_registry_authority_not_integrated`. It cannot produce
  `activation_candidate=true` or `new_risk_authorized=true`.
- Readiness v2 is a distinct schema and filename. Its validator requires
  `activation_candidate=false`, `new_risk_authorized=false`, and
  `readiness_status=no_new_risk` regardless of supplied evidence. Enabling
  authorization requires a later, separately reviewed schema.
- Readiness v2 never accepts schedule v2, a bare path, a standalone hash,
  caller-asserted health/readiness booleans, or a summary-to-authority adapter.
- Candidate report v16 and Dashboard Contract v16 continue to require readiness
  v1. They do not accept readiness v2 until versioned consumers are introduced.
- v15 remains the production/default authority until a separately authorized
  cutover. No v16 activation candidate changes the production pointer.

## Evidence DAG

The foundation DAG is acyclic:

```text
private raw evidence -> bound canonical artifacts -> schedule-v3 binding
legacy registry bytes + Factor-v4 evidence -> Factor production-set carrier
schedule-v3 binding + Factor carrier -> readiness-v2 no-new-risk projection
```

Readiness never feeds an artifact that it consumes. Future Codex human
authorization must bind the execution-plan and handoff hashes after those
artifacts exist. A future Dashboard activation receipt must bind code/schema
identity independently of both readiness and export.

Private evidence uses `v16.private-evidence-root.v2`; trust material uses
`v16.trust-material-root.v2`; governed immutable data uses
`v16.governed-data-root.v2`. Secure descriptor-bound readers enforce the
corresponding root policy. Cross-policy substitution is invalid.

## Side-effect boundary

Foundation validation may read explicitly bound local bytes. It must not write
the Factor registry, readiness/report/Dashboard production artifacts, canonical
market data, pointers, orders, or broker state, and must not invoke a network,
provider, LLM, candidate generator, portfolio constructor, subprocess, or
activation API.

## Readiness-v2 foundation blockers

The foundation builder emits blockers in sorted order and records one source
for each blocker. Its fixed blocker vocabulary is:

- `calendar_recheck_capture_time_not_independently_evidenced`
- `calendar_recheck_transport_freshness_not_independently_attested`
- `calibration_source_recomputation_not_integrated`
- `codex_authority_chain_v2_not_integrated`
- `dashboard_activation_receipt_v2_not_integrated`
- `evidence_v2_disconnected_from_authorizing_consumers`
- `global_attempt_registry_authority_not_integrated`
- `production_pointer_switch_not_authorized`
- `provisional_journal_head_not_bound_to_external_anti_rollback_authority`
- `readiness_v2_foundation_schema_nonauthorizing`

Missing schedule evidence adds `schedule_v3_lineage_missing`. Factor carrier
readback blockers are prefixed with `factor_v4:` and retain their exact source
text. A present schedule-v3 lineage contributes only blockers recomputed by its
bound lineage validator. The foundation exposes no writer or CLI command.

The Factor carrier derives production names, states, raw weights, normalized
absolute weights, the 20% per-factor cap, count bounds, record hashes, and set
hash only from the exact legacy registry bytes. It leaves family count and the
35% family cap unverified until separately bound v4 family/slot evidence
exists; it never infers either field from legacy tags or metadata.
