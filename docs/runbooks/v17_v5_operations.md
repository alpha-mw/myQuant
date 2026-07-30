# V17 v5 Sprint-1D Operations

V17 v5 remains non-operational. V15 remains the default and V17 v4 remains the
Forward Evidence runtime. Sprint 1D pins V5 to the exact V4 causal Regime
Evidence v2 predecessor and enables the read-only v2 adapter for descriptive
origin-regime diagnostics. It does not modify the V4 producer, create regime
artifacts, add online Factor weights, change tiers, recommend lifecycle
actions or grant operational authority.

The available commands are:

```bash
./.venv/bin/quant-investor-v17-v5 status
./.venv/bin/quant-investor-v17-v5 verify
./.venv/bin/quant-investor-v17-v5 factor-regime-diagnostics --help
```

`status` must report:

```text
protocol_version = myquant.v17.v5
default_protocol_state = V15_DEFAULT
global_activation_state = INACTIVE
run_state = INACTIVE
status = SPRINT1D_CAUSAL_REGIME_ADAPTER_AVAILABLE_NOT_OPERATIONAL
```

All authority fields must be false.

`verify` additionally validates:

- the closed V17 v5 package manifest;
- the closed V17 v5 runtime manifest;
- the compatibility policy;
- the pinned V17 v4 package and runtime manifests at
  `1da7ffb636a3254940525d746549d15e827f06ba`.

Sprint 1D uses a worktree-colocated predecessor integration. The local V5
branch keeps the exact V4 Sprint 1C commit as a merge parent instead of
hand-copying V4 files. V5 still treats V4 as read-only.

There is no V17 v5 operational run, schedule, output path or mutable pointer.
`factor-regime-diagnostics` returns one JSON object to stdout and must not
create files.

With the current real evidence gap, use explicit unavailable declarations:

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

The result must have `status=UNAVAILABLE` and the reason codes
`V4_FACTOR_EVIDENCE_UNAVAILABLE` and
`V4_REGIME_EVIDENCE_V2_UNAVAILABLE`. Installing the v2 schema is not enough to
return `UNOBSERVED` or `ACCUMULATING`; those states require explicit real v2
evidence and, for `ACCUMULATING`, at least one conditionable mature origin.
`output-id` identifies the stdout delivery envelope; the diagnostic retains its
own content-derived `diagnostic_id`.

The Sprint 1D CLI validates explicit Factor and regime inputs but does not
assemble multiple mature origins from a single receipt/regime pair. If both
exact inputs are otherwise eligible, it fails closed with
`OBSERVED_FACTOR_REGIME_CLI_PATH_NOT_ENABLED` and
`origin_binding_result=NOT_ENABLED`; it never fabricates an origin inventory.

Exact evidence mode requires both path and SHA-256 for each input:

```text
--factor-evidence-path + --factor-evidence-sha256
--regime-evidence-path + --regime-evidence-sha256
```

Paths are explicit workspace-relative V4 artifact refs. The command never
scans for a latest artifact. A hash, identity, closure, causality or policy
contradiction exits 2; malformed evidence is not converted to `UNAVAILABLE`.
The regime evidence path must point to
`myquant.v17.v4.regime-evidence.v2`. `myquant.v17.v4.regime-evidence.v1` may
still be integrity-checked by the compatibility reader, but it is not
conditioning-eligible and returns
`REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE`.

The library-only surface is:

```python
from quant_investor.v17_v5_runtime import (
    FactorOriginSample,
    FactorSampleStratum,
    adapt_v4_factor_evidence,
    build_factor_diagnostic,
    build_factor_diagnostic_from_v4,
    build_factor_lifecycle_diagnostic,
    build_unavailable_factor_diagnostic,
    build_unavailable_factor_lifecycle_diagnostic,
    validate_factor_diagnostic_replay,
    validate_factor_lifecycle_diagnostic_replay,
    adapt_v4_regime_evidence,
    build_factor_regime_origin_inventory,
    build_regime_conditioned_factor_diagnostic,
    build_unavailable_regime_conditioned_factor_diagnostic,
)
```

Callers may provide one exact stratum directly, or first use
`read_v4_artifact` on exact V4 forward-evaluation-receipt path/SHA inputs and
pass the returned closure to `build_factor_diagnostic_from_v4`. The runtime
derives maturity and comparable symbol intersections. It does not scan for
evidence or write the returned dictionary.

Do not persist the result as a receipt, treat
`descriptive_coverage_minimum_met` as a statistical or promotion gate, or infer
Factor effectiveness. Even when that field is true,
`inference_gate_passed=false` and every authority field remains false.

The Python compatibility reader is limited to exact caller-supplied V17 v4
references. It requires:

```text
workspace_root
relative_path
expected_byte_sha256
expected_strategy_id
decision_cutoff
```

It returns a read-only in-memory root plus its complete, sorted dependency
closure. A path escape, symlink, hard link, case alias, hash mismatch, future
cutoff/availability, unknown artifact, cycle, partial/hidden transitive
reference, schema failure or authority violation raises
`V4CompatibilityError` and writes nothing. Raw upstream source bindings remain
terminal provenance and are never followed.

The v2 regime adapter accepts only evidence that satisfies the pinned Sprint 1D
policy:

```text
version = myquant.v17.v4.regime-evidence.v2
inference_kind = FILTERED_CAUSAL
smoothing_used = false
publication_phase = PRIOR_SESSION_EFFECTIVE_NEXT_SESSION
scope_kind = FULL_MARKET
hard_state_derivation = SEALED_ARGMAX_POLICY_V1
no_retroactive_causal_backfill = true
```

V5 does not recompute the posterior, rerun argmax, change tie-breaks, map
states, call the V4 producer or read V15 JSONL history. The hard state is the
sealed V4 state. The `未知` state is a valid v2 artifact state but is
conditioning-ineligible and is not placed in a by-regime bucket.

When no compatible V4 evaluation receipt exists, use the empty adapter input.
The result is `UNAVAILABLE`; do not synthesize a receipt, origin, label or
market calendar.

`verify` byte-binds packaged JSON and runtime Python. Contract Python is
filename-inventoried rather than byte-bound inside the self-referential package
manifest; use the reviewed Git checkpoint as its source binding.

Regime-chain deployability remains a read-only audit result in Sprint 1D. If a
long chain exceeds the current closure limits or a missed session permanently
blocks later evidence, record the blocker as
`V4_REGIME_CHAIN_SCALABILITY_GAP` or `V4_REGIME_CHAIN_LIVENESS_GAP`. Do not
raise V5 closure limits and do not modify the V4 producer in this Sprint.

The Sprint 1D synthetic V4 probe first fails at contiguous session 3, before
the requested 20/60/260/1,000-session checkpoints. All four checkpoints are
therefore `BLOCKED`; timing and peak memory are not extrapolated. The direct
error is `REGIME_EVIDENCE_V2_INPUT_TAMPER` /
`model_snapshot_ref readback failed`, with the V4 closure resource budget as
the nested cause. In the missed-session S0/S1/S2/S3 scenario, S3 is likewise
blocked. An explicit S3 restart without a predecessor fails with
`REGIME_EVIDENCE_V2_TEMPORAL_CAUSALITY` /
`NORMAL publication requires the contiguous prior v2`; the sealed
contiguous-predecessor rule provides neither a stale fallback nor a restart
path.
