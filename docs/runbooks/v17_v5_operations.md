# V17 v5 Phase-0 Operations

V17 v5 is not operational. V15 remains the default and V17 v4 remains the
Forward Evidence runtime.

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
status = PHASE0_CONTRACT_AVAILABLE_NOT_OPERATIONAL
```

All authority fields must be false.

`verify` additionally validates:

- the closed V17 v5 package manifest;
- the closed V17 v5 runtime manifest;
- the compatibility policy;
- the pinned V17 v4 package and runtime manifests.

There is no V17 v5 run command, schedule, output path or mutable pointer.
Verification must not create files.

The Python compatibility reader is limited to exact caller-supplied V17 v4
references. It requires:

```text
workspace_root
relative_path
expected_byte_sha256
expected_strategy_id
decision_cutoff
```

It returns a read-only in-memory result. A path escape, symlink, hard link,
case alias, hash mismatch, future cutoff/availability, unknown artifact,
unexpected transitive reference, schema failure or authority violation raises
`V4CompatibilityError` and writes nothing.

Do not add an artifact version to the compatibility policy until its complete
transitive closure and resource bounds have independent tests. Do not start
Sprint 1 until the Phase-0 acceptance and review gates in
`docs/architecture/v17_v5_investment_intelligence.md` pass.
