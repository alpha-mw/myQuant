# v16 Codex Authority Source v2

## Status

This lane is additive, disconnected, and structurally nonauthorizing. It does
not replace the v15 production/default authority and does not switch a
production pointer. An activation candidate is not a production activation.

The implemented DAG is:

```text
full-union posterior v2 + readiness v3
  -> Codex authority source plan v2
  -> Codex IC source status v2
  -> execution source status v2
  -> handoff source status v2
  -> v16 run readiness v4
```

## Artifacts

The source plan predeclares unique absolute paths, schemas, and the
`v16.private-evidence-root.v2` policy for these artifacts:

| Artifact | Schema |
| --- | --- |
| MenuSeal | `codex-review-menu.v1` |
| Stage2 request | `codex-review-stage2-request.v1` |
| Stage2 response | `codex-review-stage2-response.v1` |
| IC source status | `v16.codex-ic-source-status.v2` |
| Execution source status | `v16.execution-source-status.v2` |
| Handoff source status | `v16.handoff-source-status.v2` |
| Readiness | `v16_run_readiness.v4` |

Every consumed artifact is reopened from bound bytes. Review artifacts retain
their native-float v1 JSON contract and self-seal, while evidence-v2 status
artifacts encode numeric projections with canonical `f64:` values.

## IC Validation

`codex_ic_source_v2` reruns the full-union posterior validator from bound
Stage1, runtime, formal-branch, and cost artifacts. It then independently
checks:

- Menu symbols and order equal the recomputed posterior menu.
- Posterior values, Q/F/M/LLM branch values, evidence IDs, and retrieval
  advisory projections match exactly.
- Retrieval remains evidence-only and RiskAdvisor remains advisory-only.
- MenuSeal, Stage2 request, and Stage2 response bytes are strict and canonical;
  their self-seals and detached references agree.
- Stage2 request derives exactly from the MenuSeal and preserves the Stage1
  predecessor.
- Every menu symbol has exactly one `BUY`, `HOLD`, `AVOID`, or `SELL` decision.
- Existing weights are complete, positive target weights cover no more than 12
  symbols, and target weights plus cash equal 1.

The internal timestamp relation
`decision_cutoff_at <= sealed_at < expires_at` is checked. No independent
precommit chronology or model-execution attestation is inferred from it.

## Authority Boundary

The new modules do not directly import or call the legacy Codex workflow,
CapitalMap, HumanAuthorization, ExecutionGate, portfolio construction, broker,
or order surfaces. They do not accept capital, shares, prices, an execution
plan, market state, a human receipt, an authorization boolean, order
eligibility, or a handoff-ready claim.

In particular, none of these inputs is authority evidence for this lane:

- `codex-review-human-authorization.v1`, even when self-sealed;
- a caller-provided `human_authorized=true` value;
- a bare capital, execution, or handoff mapping;
- an existing v1 Codex review state transition to `AUTHORIZED`.

Execution and handoff statuses are source-status projections only. Readiness-v4
always emits `readiness_status=no_new_risk`,
`new_risk_authorized=false`, and false Codex, Dashboard, live-human, broker,
production-apply, and production-pointer authority fields.

## Open Contracts

The following requirements remain unsupported and are exact blockers:

- `menu_position_source_contract`
- `menu_reference_price_source_contract`
- `risk_advisory_source_contract`
- `stage2_model_execution_attestation`
- `execution_plan_source_contract`
- `execution_market_state_source_contract`
- `live_human_identity_signature_protocol`
- `handoff_delivery_attestation_protocol`

Production readiness additionally requires an external anti-rollback
authority, authorizing-consumer integration, a governed Dashboard activation
receipt, and a separately authorized production pointer switch. These are
external governance decisions; this source lane does not invent defaults for
them.

## Side Effects

All builders and validators are pure over supplied bound artifacts. They do not
write review state, call a provider or LLM, generate candidates, construct a
portfolio, activate Dashboard, access a broker, create orders, or trade.
