# Phase 8 reconciliation audit

Status: AUDIT_COMPLETE; implementation and phase acceptance remain open.

## Reproduced native gap

The read-only canonical risk exporter, observed on 2026-09-12, selects the existing
Store official date 2026-09-03. It returns PARTIAL with one blocker:
`601899.SH:CORPORATE_ACTION_OR_ADJUSTMENT_REVIEW_REQUIRED`. The stock's trailing
threshold is NON_EXECUTABLE and its calculated prices are null. This diagnosis is
retrospective and uses the currently retained Market snapshot 20260911T135213Z;
it is not an original September 3 PIT production receipt.

Exact source references and results are in
`.agent/acceptance/phase8-native-risk-baseline.json`. The bound Zijin frame changes
adj_factor from 2.2091 on 20260820 to 2.2365 on 20260821. The last daily pair,
20260902 and 20260903, both has 2.2365. Thus the existing corporate node's
previous-day-only comparison misses an unresolved action within the active
tracking window that began 20260819. A no-change result for the last pair cannot
establish reconciliation of the whole anchor window.

The exact existing owner policy is
`results/policies/risk/aggressive_tech_manufacturing/trailing-anchor.v1/owner-trailing-anchor-policy-20260901-v1.json`,
SHA b313aa91e1f7ca1e8922b2d22f7735ceee3190675c2e8dab3955c69f0f1d342a.
Its explicit invalidation condition includes corporate action. Numerical
back-adjustment alone cannot reinstate that owner's anchor or authorize a stop
change. The baseline native risk exporter correctly preserves the veto.

## Missing behavior, in dependency order

1. Bind the existing frozen pre-close portfolio and the exact owner tracking
   policy to corporate evidence. Validate the complete required Calendar/Market
   interval for every held symbol, not only T-1 and T. Missing anchor, missing
   daily evidence and position lifecycle changes remain explicit blockers.
2. Admit a reviewed, exact, immutable named-action source contract for split,
   dividend, rights, bonus issue and share conversion. Source-backed identity,
   effective date and availability are separate from an adjustment-factor
   observation. No action kind, subscription, entitlement or settlement may be
   inferred from the factor ratio.
3. Report cost_basis_adjustment, shares_adjustment and threshold_anchor_adjustment
   separately. Reconciliation must cite actual registered financial pre/post
   evidence and the applicable owner policy. An asserted after-value, issuer
   announcement or numeric transform alone is not an account posting or renewed
   owner authority. Missing evidence remains UNCONFIRMED/NON_EXECUTABLE.
4. Reuse the existing corporate DAG node, source capture and completed replay;
   version the changed recipe/report contract and preserve legacy exact replay.
   Bind every new input through the intended materialization path and ledger
   provenance. A same-day action conflicting with CLOSED_EMPTY must block the
   no-action Store close rather than allow an unchanged book.
5. Verify all named cases with isolated explicit fixtures, full-window historical
   factor changes, missing/duplicate/future/conflicting evidence, immutable replay,
   and no accounting/owner mutation. Keep the original canonical Zijin blocker
   until actual reconciliation evidence is available and validated.

This audit does not claim named events are identified, does not establish new
financial treatment, and does not modify policies, actual positions, cash,
thresholds or automations. Architect then Critic review is required for the
concrete contract/implementation plan before those changes, as required by AGENTS.
