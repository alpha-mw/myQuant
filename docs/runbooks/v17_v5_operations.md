# V17 v5 Sprint-1B Operations

V17 v5 remains non-operational. V15 remains the default and V17 v4 remains the
Forward Evidence runtime. Sprint 1B adds only a read-only regime evidence
adapter and origin-regime descriptive diagnostics. It does not add online
Factor weights, tier changes, lifecycle actions or operational authority.

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
status = SPRINT1B_REGIME_CONDITIONED_FACTOR_DIAGNOSTICS_AVAILABLE_NOT_OPERATIONAL
```

All authority fields must be false.

`verify` additionally validates:

- the closed V17 v5 package manifest;
- the closed V17 v5 runtime manifest;
- the compatibility policy;
- the pinned V17 v4 package and runtime manifests.

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
`V4_FACTOR_EVIDENCE_UNAVAILABLE` and `V4_REGIME_EVIDENCE_UNAVAILABLE`.
`output-id` identifies the stdout delivery envelope; the diagnostic retains its
own content-derived `diagnostic_id`.

Exact evidence mode requires both path and SHA-256 for each input:

```text
--factor-evidence-path + --factor-evidence-sha256
--regime-evidence-path + --regime-evidence-sha256
```

Paths are explicit workspace-relative V4 artifact refs. The command never
scans for a latest artifact. A hash, identity, closure, causality or policy
contradiction exits 2; malformed evidence is not converted to `UNAVAILABLE`.

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

Do not add an artifact version to the compatibility policy until its complete
transitive closure and resource bounds have independent tests. The current
adapter is pure and descriptive: it cannot write Factor Governance, alter a
tier or weight, make a promotion decision, or create an operational run.

When no compatible V4 evaluation receipt exists, use the empty adapter input.
The result is `UNAVAILABLE`; do not synthesize a receipt, origin, label or
market calendar.

`verify` byte-binds packaged JSON and runtime Python. Contract Python is
filename-inventoried rather than byte-bound inside the self-referential package
manifest; use the reviewed Git checkpoint as its source binding.

The registered V4 `regime-evidence.v1` currently has no sealed hard state,
posterior, decision/effective session, publication timestamp or source refs.
The adapter therefore reports `REGIME_HARD_STATE_UNAVAILABLE`; it must not
derive a state from `gross_multiplier`, `role`, timestamps or raw market data.
No real `ACCUMULATING` regime-conditioned diagnostic can be produced until both
a causally complete V4 regime artifact and real V4 Factor evaluation receipts
exist.
