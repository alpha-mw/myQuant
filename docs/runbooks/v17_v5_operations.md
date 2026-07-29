# V17 v5 Sprint-1A Operations

V17 v5 remains non-operational. V15 remains the default and V17 v4 remains the
Forward Evidence runtime. Sprint 1A only adds a pure Python descriptive Factor
diagnostic library.

The only available commands are:

```bash
./.venv/bin/quant-investor-v17-v5 status
./.venv/bin/quant-investor-v17-v5 verify
```

`status` must report:

```text
protocol_version = myquant.v17.v5
default_protocol_state = V15_DEFAULT
global_activation_state = INACTIVE
run_state = INACTIVE
status = SPRINT1A_V4_FACTOR_EVIDENCE_DIAGNOSTICS_AVAILABLE_NOT_OPERATIONAL
```

All authority fields must be false.

`verify` additionally validates:

- the closed V17 v5 package manifest;
- the closed V17 v5 runtime manifest;
- the compatibility policy;
- the pinned V17 v4 package and runtime manifests.

There is no V17 v5 run command, schedule, output path or mutable pointer.
Verification must not create files.

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
