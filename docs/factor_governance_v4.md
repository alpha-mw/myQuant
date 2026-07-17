# FactorGovernanceProtocol v4 research scaffold

FactorGovernanceProtocol v4 is a research-only shadow contract. It does not
replace the current v15 FactorGovernance v3 runtime, change an activation
pointer, write `mined_factors.json`, append a production WAL, or authorize new
risk. v2 and v3 evidence remain historical/current evidence in their own
contracts and are never auto-upgraded to v4.

## Readiness states

The target production set has exactly ten healthy factors.

- `ready_target_10`: exactly ten healthy factors and every set-level contract
  passes.
- `underfilled_accelerated_mining`: five through nine healthy factors pass the
  minimum baseline. Weekly mining continues in accelerated report-only mode.
- `no_new_risk`: fewer than five factors, fewer than three families, more than
  ten factors, an invalid weight/slot/evidence/runtime contract, stale or
  blocked health, or a missing/invalid activation receipt.

`new_risk_eligible` is a readiness result, not authority. This scaffold always
returns `new_risk_authorized=false` and
`production_apply_enabled=false`. Any v16 activation must fail closed unless a
separate production workflow verifies the same-day hash-bound receipt and all
other v16 gates. The v15 default remains authoritative.

For exactly five normalized factors, the 20% per-factor cap means every factor
must have exactly 20% absolute weight. The 35% family cap therefore forbids two
of those factors from sharing a family: the five-factor baseline mathematically
requires five distinct families. For larger sets the minimum family count is
three, subject to the same 35% cap.

## Healthy factor contract

A factor is healthy only when all of the following pass together:

1. State is exactly `production_factor` and weight has a finite, non-zero
   magnitude.
2. Governance gates 1 through 8 are present and pass.
3. Maturity is either 12 distinct month-end RankIC periods or 8 non-overlapping
   30-day forward cohorts. A standalone 90-day diagnostic is not authoritative.
4. Benjamini-Hochberg is applied by family and the factor q-value is at most
   0.10.
5. The factor has one non-empty slot and no other production factor occupies
   that slot.
6. Its runtime contract is verified and its canonical semantic SHA matches.
7. Its latest health observation is fresh and `healthy`.
8. The set has a valid same-day v4 activation receipt bound to the protocol,
   transaction plan, registry bytes, production factor set, and aggregate
   runtime contracts.

The normalized absolute-weight caps are 20% per factor and 35% per family.
`data_blocked` is not an alpha failure and must not increment failure streaks,
but it blocks factor health and set readiness until fresh data closes it.

Candidate admission is also machine-checkable before any registry proposal.
The candidate must carry a healthy strict-Parquet source receipt bound to the
full-A universe, `duplicate_primitive=false`, a passed high-correlation dedup
check, `initial_weight=0`, an authoritative maturity route, family BH q-value
at most 0.10, all eight gates, and verified v4 replay evidence. Failure returns
an explicit blocked assessment; the validator raises and no candidate registry
proposal is allowed.

## Cadence and proposals

Weekly mining and factor-health runs are report-only. They cannot create an
add/replace/watch/reduce/deprecate transaction or change weights.

Month-end may emit at most two proposals. `watch_proposal`,
`reduce_proposal`, and `deprecate_proposal` remain proposals; none is an apply
command. While the set has fewer than ten factors, a reviewed add proposal may
fill a missing slot. Once the set reaches ten, additions are forbidden and each
replacement proposal must be one-in-one-out. Its paired incremental-edge 95%
confidence lower bound must be strictly greater than zero.

Distinct matured alpha failures map to month-end proposals as follows: one is
`watch_proposal`, two is `reduce_proposal` with a suggested 50% weight, and
three or more is `deprecate_proposal`. Duplicate failure-window IDs count once.
Data-blocked windows never enter that count, but they identify the blocking
factor and block factor-set new risk. Every result remains `apply=false`.

## Canonical replay v4

Each A/B/C/D replay arm contains exactly this chain:

`Eligibility -> Quant -> Funnel -> CodexS1 -> Bayesian -> RiskAdvisor -> CodexIC -> PortfolioConstructor`

Every stage binds the exact v4 context, predecessor byte SHA, predecessor
semantic SHA, and its own output semantic SHA. Eligibility binds the admitted
symbol universe. Quant may score only that universe; Funnel may only narrow
it. Bayesian binds CodexS1, RiskAdvisor binds Bayesian, CodexIC binds semantic
hashes for every upstream stage, and PortfolioConstructor consumes exact
CodexIC decision hashes.

RiskAdvisor is advisory-only. Its output remains hash-bound in the replay, but
a positive PortfolioConstructor weight does not require RiskAdvisor approval.
A positive weight does require a hash-bound CodexIC `buy` decision.

The A/B/C/D factor sets prove a one-slot replacement:

- A: incumbent production set;
- B: incumbent removed;
- C: challenger inserted into the same family/slot;
- D: the same replacement set under the final replay arm.

At target ten, the replay rejects a challenger unless the incremental-edge 95%
lower bound is positive. Replay and evidence schemas are v4-only. Local
readback accepts one explicit absolute regular file, owned by the current user,
mode `0600`, with canonical JSON bytes and no scan/latest/fallback behavior.

## WAL, CAS, rollback, and receipt

`build_factor_v4_transaction_plan` produces an inert plan with four independent
bindings:

- an append-only WAL blueprint containing before/after registry byte hashes;
- a CAS blueprint comparing the exact expected registry SHA before any swap;
- an inverse rollback blueprint that compares the proposed SHA and restores
  the exact before-registry and factor-set hashes;
- a separately authorized activation-receipt requirement.

The plan carries `plan_only=true`, `registry_mutation_performed=false`, and
`production_apply_enabled=false`. The WAL is `planned_not_written`, CAS is
`planned_not_attempted`, and inverse rollback is `planned_not_applied`.
Building or validating the plan performs none of those operations.

`FactorV4ShadowTransactionStore` is the separately testable research store.
It has no default path and owns only fixed files beneath a caller-supplied
`shadow_root`; it cannot discover or open the current Factor registry or
activation pointer. Under an independent file lock it can initialize a v4
shadow registry, append and fsync a 0600 WAL intent, write a 0600 inverse
manifest containing the exact before bytes, CAS the exact expected shadow
registry SHA through an atomic replace, read back the result, append a commit
record, and write a 0600 shadow receipt. Rollback requires a separate scope,
CAS-compares the post-transaction SHA, restores the exact before bytes, appends
rollback WAL records, and replaces the shadow receipt with a revocation.

The store's receipt schema is
`factor-governance-shadow-activation-receipt.v4`; it explicitly carries
`production_activation_performed=false`. It cannot satisfy the production
activation-receipt validator used by readiness and therefore cannot unlock v16
new risk.

An activation request is not a receipt. A receipt validates only with the exact
`factor-governance-activation-receipt.v4` schema, an explicit
`factor_v4_production_activation` authorization scope, same-day timezone-aware
activation time, and canonical hashes for its complete activation context and
receipt. Receipt validation still does not perform activation.

## Offline scripts

All scripts require explicit input and output paths and write only the requested
research artifact as mode `0600`:

```bash
./.venv/bin/python scripts/build_factor_v4_readiness_plan.py \
  --input-json <readiness-input.json> \
  --output-json <research-readiness.json>

./.venv/bin/python scripts/build_factor_v4_replay_validation.py \
  --kind replay \
  --input-json <canonical-replay.v4.json> \
  --output-json <replay-validation.json>

./.venv/bin/python scripts/build_factor_v4_transaction_plan.py \
  --input-json <transaction-input.json> \
  --output-json <research-transaction-plan.json>
```

These commands never accept a registry path or activation pointer and do not
have an apply flag.
