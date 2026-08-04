# V17 v5 Sprint 1E-0B Operations

V17 v5 remains non-operational. V15 remains the default protocol and V17 v4
remains the Forward Evidence producer. Sprint 1E-0B only verifies the exact V4
predecessor and enables read-only V3 origin-regime diagnostics.

The available commands are:

```bash
./.venv/bin/quant-investor-v17-v5 status
./.venv/bin/quant-investor-v17-v5 verify
./.venv/bin/quant-investor-v17-v5 factor-regime-diagnostics --help
```

`status` must keep:

```text
protocol_version = myquant.v17.v5
default_protocol_state = V15_DEFAULT
global_activation_state = INACTIVE
run_state = INACTIVE
research_runtime_default = false
```

All authority fields are false.

## Verification

`verify` closes:

- the V5 package and runtime manifests;
- V5 Regime diagnostic policies v1, v2, and v3;
- the exact Release-RC-1 worktree-colocated V4 predecessor
  `6a2fa23dec68d87eb686464a86d8ba8997416310`;
- all V4 package assets and runtime sources;
- the V3 schema, V2 inference policy, V3 producer source, and V2 public
  publication block.

V5 does not copy or modify V4 files and has no fallback to an older pin. The
v1/v2/v3 compatibility policies and predecessor-binding schemas remain explicit
historical identities; v4 is additive and active for this release candidate.

## Current unavailable mode

Until real V4 evidence exists:

```bash
./.venv/bin/quant-investor-v17-v5 factor-regime-diagnostics \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --strategy-id <strategy-id> \
  --factor-name <factor-name> \
  --evaluation-cutoff <origin-cutoff-utc> \
  --created-at <created-at-utc> \
  --output-id <caller-output-id> \
  --factor-evidence-unavailable \
  --regime-evidence-unavailable
```

Expected status and reasons:

```text
UNAVAILABLE
V4_FACTOR_EVIDENCE_UNAVAILABLE
V4_REGIME_EVIDENCE_V3_UNAVAILABLE
```

Installing V3 schemas does not imply `UNOBSERVED` or `ACCUMULATING`.

## Explicit V3 validation mode

An explicit V3 read requires:

```text
--regime-evidence-path
--regime-evidence-sha256
--regime-checkpoint-path
--regime-checkpoint-sha256
```

Factor evidence likewise requires both path and SHA. Paths are
workspace-relative immutable V4 refs. The command never scans for a latest
artifact, calls a provider, downloads data, invokes the V4 producer, or writes
V4/V5 artifacts.

V1 and V2 can still be integrity-checked by the compatibility reader:

- V1 returns `REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE`;
- V2 returns `REGIME_EVIDENCE_V2_NON_DEPLOYABLE`;
- only finalized V3 can be conditioning eligible.

The CLI currently validates exact inputs and emits one stdout-only diagnostic.
It does not assemble a multi-origin inventory from a single Factor/Regime pair.
An otherwise complete pair therefore remains fail-closed with
`OBSERVED_FACTOR_REGIME_CLI_PATH_NOT_ENABLED` until a separately reviewed
origin assembly entrypoint exists.

## Composite finality

V3 evidence is finalized only when:

- evidence path/SHA, schema, identity, semantic SHA, strategy, cutoff, and
  pinned closure pass;
- `current_checkpoint_ref` resolves from the caller's explicit checkpoint
  path/SHA;
- checkpoint is in the evidence direct `source_refs` closure;
- evidence and checkpoint agree on session, segment, phase, posterior, hard
  state, record commitment, global accumulator, segment accumulator, and chain
  identity;
- the current chain/segment/feature/model/transition/scope/policy closure is
  bounded and complete.

A standalone checkpoint is insufficient. Missing or conflicting checkpoint
state exits 2 with `REGIME_EVIDENCE_V3_NOT_FINALIZED`.

The checkpoint does not reverse-bind evidence bytes because that would form a
hash cycle with the evidence's checkpoint content reference. V5 adds no
finality receipt and does not alter V4 publication.

V5 does not recompute posterior, argmax, commitments, digests, or continuity,
and it does not traverse historical checkpoints.

## Conditioning

```text
CONTIGUOUS -> eligible
ROLLOVER   -> eligible
GENESIS    -> ineligible
RECOVERY   -> ineligible
未知        -> ineligible
not finalized -> malformed/fail closed
```

Eligibility also requires `FILTERED_CAUSAL`, no smoothing,
`PRIOR_SESSION_EFFECTIVE_NEXT_SESSION`, `FULL_MARKET`,
`SEALED_ARGMAX_POLICY_V1`, the no-backfill flag, exact origin/effective session
binding, prior Shanghai open-session observation, and publication before the
Factor cutoff.

No online Factor weight, tier change, lifecycle action, validity conclusion,
promotion, production target, portfolio output, broker call, order, or trade is
produced.
