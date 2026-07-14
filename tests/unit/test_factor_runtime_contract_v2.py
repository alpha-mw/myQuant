from __future__ import annotations

import copy
import hashlib
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.branch_contracts import BranchResult
from quant_investor.factors.governance import (
    GATE_SPECS,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import (
    MinedFactorRegistry,
    MinedFactorScorer,
    RuntimeFactorScore,
    _mint_production_evaluation_context,
    production_symbol_set_sha256,
    production_runtime_input_sha256,
    production_runtime_metadata_is_ready,
    production_runtime_score_is_ready,
)
from quant_investor.factors.governance_protocol_v2 import (
    governance_runtime_status,
    protocol_hash,
)
from quant_investor.factors.runtime_contract import (
    ACTIVATION_RECEIPT_SCHEMA_VERSION,
    RUNTIME_CONTRACT_SCHEMA_VERSION,
    activation_receipt_payload_sha256,
    factor_definition_sha256,
    factor_record_payload_sha256,
    implementation_code_sha256,
    production_implementation_spec,
    production_runtime_contracts_sha256,
    validate_production_runtime_contracts,
    validate_quant_production_activation,
)
from quant_investor.market.dag.packets import _build_global_quant_verdict
from quant_investor.market.pit_universe import PITUniverseRecord


PIT_OBSERVED_AT = "2026-01-06T00:00:00Z"


def _gates(*, coverage: float = 1.0) -> list[GateResult]:
    rows = [
        GateResult(
            gate_id=spec.gate_id,
            gate_key=spec.key,
            title=spec.title,
            passed=True,
            metrics={},
        )
        for spec in GATE_SPECS
    ]
    rows[1].metrics = {"coverage_rate": coverage}
    return rows


def _record(
    name: str = "pv_low_dollar_volume_5d",
    *,
    implementation: str | None = None,
    weight: float = 0.05,
) -> FactorRecord:
    return FactorRecord(
        name=name,
        version="v1",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        category="liquidity",
        implementation=implementation or f"price_volume:{name}",
        weight=weight,
        direction=1.0,
        gate_results=_gates(),
        metadata={
            "factor_family": "liquidity",
            "dominant_primitive_cluster": name,
        },
    )


def _contract(record: FactorRecord, evidence_path: Path) -> dict[str, object]:
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    spec = production_implementation_spec(record.implementation)
    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "factor_name": record.name,
        "factor_version": record.version,
        "implementation_id": record.implementation,
        "implementation_version": spec["implementation_version"],
        "implementation_code_sha256": implementation_code_sha256(
            record.implementation
        ),
        "required_columns": spec["required_columns"],
        "data_semantics": spec["data_semantics"],
        "lookback_rows": spec["lookback_rows"],
        "gate2_min_coverage_rate": 1.0,
        "min_cross_section": 20,
        "factor_definition_sha256": factor_definition_sha256(record),
        "factor_record_sha256": factor_record_payload_sha256(record),
        "factor_evidence_path": str(evidence_path),
        "factor_evidence_sha256": evidence_sha,
    }


def _contract_metadata(
    record: FactorRecord,
    contract: dict[str, object],
    registry_path: Path,
) -> dict[str, object]:
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "mined-factor-registry.v1",
                "metadata": {
                    "production_factor_runtime_contracts": {
                        record.name: contract,
                    },
                },
                "factors": [record.to_dict()],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    strict_registry = MinedFactorRegistry.load_production(registry_path)
    assert "strict_load_error" not in strict_registry.metadata
    return dict(strict_registry.metadata)


def test_runtime_contract_requires_exact_allowlisted_readback_bound_contract(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"verified"}\n', encoding="utf-8")
    record = _record()

    missing = validate_production_runtime_contracts([record], {})
    assert missing["status"] == "governance_blocked"
    assert "production_runtime_contracts_missing" in missing["blockers"]

    contract = _contract(record, evidence)
    metadata = _contract_metadata(record, contract, tmp_path / "registry.json")
    ready = validate_production_runtime_contracts([record], metadata)
    assert ready["status"] == "ready"
    assert ready["blockers"] == []

    contract["implementation_id"] = "alpha_mining.FactorLibrary:momentum_1m"
    blocked = validate_production_runtime_contracts(
        [record],
        _contract_metadata(record, contract, tmp_path / "registry.json"),
    )
    assert blocked["status"] == "governance_blocked"
    assert any("implementation_id_mismatch" in item for item in blocked["blockers"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gate2_min_coverage_rate", True),
        ("gate2_min_coverage_rate", "1.0"),
        ("min_cross_section", True),
        ("min_cross_section", "20"),
        ("lookback_rows", 0),
        ("required_columns", ["trade_date", "close"]),
    ],
)
def test_runtime_contract_rejects_coercion_and_semantic_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    record = _record()
    contract = _contract(record, evidence)
    contract[field] = value

    result = validate_production_runtime_contracts(
        [record],
        _contract_metadata(record, contract, tmp_path / "registry.json"),
    )

    assert result["status"] == "governance_blocked"
    assert result["blockers"]


def test_runtime_contract_rejects_hash_drift_and_unknown_fields(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    record = _record()
    contract = _contract(record, evidence)
    contract["future_extension"] = True
    contract["implementation_code_sha256"] = "0" * 64
    evidence.write_text('{"tampered":true}\n', encoding="utf-8")

    result = validate_production_runtime_contracts(
        [record],
        _contract_metadata(record, contract, tmp_path / "registry.json"),
    )

    assert result["status"] == "governance_blocked"
    assert any("unknown_fields" in item for item in result["blockers"])


@pytest.mark.parametrize("drift", ["code", "evidence", "record", "definition"])
def test_runtime_contract_rejects_every_bound_hash_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    record = _record()
    contract = _contract(record, evidence)
    if drift == "code":
        contract["implementation_code_sha256"] = "0" * 64
    elif drift == "evidence":
        evidence.write_text('{"tampered":true}\n', encoding="utf-8")
    elif drift == "record":
        contract["factor_record_sha256"] = "0" * 64
    else:
        contract["factor_definition_sha256"] = "0" * 64

    result = validate_production_runtime_contracts(
        [record],
        _contract_metadata(record, contract, tmp_path / "registry.json"),
    )

    assert result["status"] == "governance_blocked"
    assert any("sha256" in item for item in result["blockers"])


def test_runtime_contract_binds_snapshot_hash_to_current_factor_record(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    snapshot_record = _record(weight=0.05)
    contract = _contract(snapshot_record, evidence)
    metadata = _contract_metadata(
        snapshot_record,
        contract,
        tmp_path / "registry.json",
    )

    current_record = _record(weight=0.10)
    result = validate_production_runtime_contracts([current_record], metadata)

    assert result["status"] == "governance_blocked"
    assert any("factor_record_sha256" in item for item in result["blockers"])

    metadata["production_factor_runtime_contracts"] = {
        current_record.name: _contract(current_record, evidence)
    }
    recomputed_contract = validate_production_runtime_contracts(
        [current_record], metadata
    )
    assert recomputed_contract["status"] == "governance_blocked"
    assert any(
        "factor_record_sha256" in item or "registry_snapshot" in item
        for item in recomputed_contract["blockers"]
    )


def test_runtime_contract_reloads_disk_snapshot_and_rejects_forged_memory(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"verified"}\n', encoding="utf-8")
    disk_record = _record(weight=0.05)
    disk_contract = _contract(disk_record, evidence)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "mined-factor-registry.v1",
                "metadata": {
                    "production_factor_runtime_contracts": {
                        disk_record.name: disk_contract,
                    },
                },
                "factors": [disk_record.to_dict()],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    strict_registry = MinedFactorRegistry.load_production(registry_path)
    assert "strict_load_error" not in strict_registry.metadata

    forged_record = _record(weight=0.10)
    forged_contract = _contract(forged_record, evidence)
    forged_metadata = {
        **strict_registry.metadata,
        "production_factor_runtime_contracts": {
            forged_record.name: forged_contract,
        },
        "record_sha256s": {
            forged_record.name: factor_record_payload_sha256(forged_record),
        },
    }

    result = validate_production_runtime_contracts(
        [forged_record],
        forged_metadata,
    )

    assert result["status"] == "governance_blocked"
    assert any(
        "registry_snapshot" in blocker or "registry_readback" in blocker
        for blocker in result["blockers"]
    )


def test_runtime_contract_binds_complete_snapshot_record_sha_set(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"verified"}\n', encoding="utf-8")
    production_record = _record(weight=0.05)
    historical_record = _record("pv_high_dollar_volume_5d", weight=0.0)
    historical_record.state = FactorLifecycleState.DEPRECATED
    historical_record.deprecated_reason = "historical_only"
    contract = _contract(production_record, evidence)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "mined-factor-registry.v1",
                "metadata": {
                    "production_factor_runtime_contracts": {
                        production_record.name: contract,
                    },
                },
                "factors": [
                    production_record.to_dict(),
                    historical_record.to_dict(),
                ],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    strict_registry = MinedFactorRegistry.load_production(registry_path)
    assert validate_production_runtime_contracts(
        strict_registry.selectable_factors(),
        strict_registry.metadata,
    )["status"] == "ready"

    incomplete_sha_set = copy.deepcopy(strict_registry.metadata)
    incomplete_sha_set["record_sha256s"].pop(historical_record.name)
    result = validate_production_runtime_contracts(
        strict_registry.selectable_factors(),
        incomplete_sha_set,
    )

    assert result["status"] == "governance_blocked"
    assert (
        "production_runtime_registry_snapshot_record_sha256s_mismatch"
        in result["blockers"]
    )


def test_production_activation_defaults_blocked_and_false_requires_receipt(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    metadata = {
        "strict_loader": True,
        "path": str(registry_path),
        "registry_sha256": registry_sha,
    }
    manifest = {"production_factor_set_sha256": "b" * 64}
    contracts_sha = "c" * 64
    code_hashes = {"pv_low_dollar_volume_5d": "e" * 64}

    missing = validate_quant_production_activation(
        metadata,
        manifest,
        contracts_sha,
        implementation_code_sha256s=code_hashes,
        protocol_version="v2",
        protocol_hash_value="d" * 64,
        environ={},
    )
    assert "quant_production_kill_switch_missing" in missing["blockers"]

    no_receipt = validate_quant_production_activation(
        metadata,
        manifest,
        contracts_sha,
        implementation_code_sha256s=code_hashes,
        protocol_version="v2",
        protocol_hash_value="d" * 64,
        environ={"QUANT_PRODUCTION_KILL_SWITCH": "false"},
    )
    assert "quant_production_activation_receipt_missing" in no_receipt["blockers"]

    receipt_path = tmp_path / "activation_receipt.json"
    receipt = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "status": "authorized",
        "activation_id": "test-only-activation",
        "approved_by": "test-fixture",
        "issued_at": "2026-07-14T00:00:00Z",
        "kill_switch_value": "false",
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": registry_sha,
        "production_factor_set_sha256": "b" * 64,
        "production_runtime_contracts_sha256": contracts_sha,
        "implementation_code_sha256s": code_hashes,
        "factor_governance_protocol_version": "v2",
        "factor_governance_protocol_hash": "d" * 64,
    }
    receipt["receipt_sha256"] = activation_receipt_payload_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    receipt_file_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    missing_external_hash = validate_quant_production_activation(
        metadata,
        manifest,
        contracts_sha,
        implementation_code_sha256s=code_hashes,
        protocol_version="v2",
        protocol_hash_value="d" * 64,
        environ={
            "QUANT_PRODUCTION_KILL_SWITCH": "false",
            "QUANT_PRODUCTION_ACTIVATION_RECEIPT": str(receipt_path),
        },
    )
    assert (
        "quant_production_activation_receipt_expected_sha256_missing"
        in missing_external_hash["blockers"]
    )

    ready = validate_quant_production_activation(
        metadata,
        manifest,
        contracts_sha,
        implementation_code_sha256s=code_hashes,
        protocol_version="v2",
        protocol_hash_value="d" * 64,
        environ={
            "QUANT_PRODUCTION_KILL_SWITCH": "false",
            "QUANT_PRODUCTION_ACTIVATION_RECEIPT": str(receipt_path),
            "QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256": receipt_file_sha,
        },
    )
    assert ready["status"] == "ready"
    assert ready["blockers"] == []

    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
    tampered = validate_quant_production_activation(
        metadata,
        manifest,
        contracts_sha,
        implementation_code_sha256s=code_hashes,
        protocol_version="v2",
        protocol_hash_value="d" * 64,
        environ={
            "QUANT_PRODUCTION_KILL_SWITCH": "false",
            "QUANT_PRODUCTION_ACTIVATION_RECEIPT": str(receipt_path),
            "QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256": receipt_file_sha,
        },
    )
    assert (
        "quant_production_activation_receipt_exact_bytes_mismatch"
        in tampered["blockers"]
    )


def test_production_activation_sha_cannot_fall_back_to_registry_metadata(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    contracts_sha = "c" * 64
    code_hashes = {"pv_low_dollar_volume_5d": "e" * 64}
    receipt_path = tmp_path / "activation_receipt.json"
    receipt = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "status": "authorized",
        "activation_id": "test-only-activation",
        "approved_by": "test-fixture",
        "issued_at": "2026-07-14T00:00:00Z",
        "kill_switch_value": "false",
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": registry_sha,
        "production_factor_set_sha256": "b" * 64,
        "production_runtime_contracts_sha256": contracts_sha,
        "implementation_code_sha256s": code_hashes,
        "factor_governance_protocol_version": "v2",
        "factor_governance_protocol_hash": "d" * 64,
    }
    receipt["receipt_sha256"] = activation_receipt_payload_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    receipt_file_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    metadata = {
        "strict_loader": True,
        "path": str(registry_path),
        "registry_sha256": registry_sha,
        "quant_production_activation_receipt_sha256": receipt_file_sha,
    }

    result = validate_quant_production_activation(
        metadata,
        {"production_factor_set_sha256": "b" * 64},
        contracts_sha,
        implementation_code_sha256s=code_hashes,
        protocol_version="v2",
        protocol_hash_value="d" * 64,
        environ={
            "QUANT_PRODUCTION_KILL_SWITCH": "false",
            "QUANT_PRODUCTION_ACTIVATION_RECEIPT": str(receipt_path),
        },
    )

    assert result["status"] == "governance_blocked"
    assert (
        "quant_production_activation_receipt_expected_sha256_missing"
        in result["blockers"]
    )


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o644])
def test_production_activation_requires_exact_0600_permissions(
    tmp_path: Path,
    mode: int,
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    contracts_sha = "c" * 64
    code_hashes = {"pv_low_dollar_volume_5d": "e" * 64}
    receipt_path = tmp_path / "activation_receipt.json"
    receipt = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "status": "authorized",
        "activation_id": "test-only-activation",
        "approved_by": "test-fixture",
        "issued_at": "2026-07-14T00:00:00Z",
        "kill_switch_value": "false",
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": registry_sha,
        "production_factor_set_sha256": "b" * 64,
        "production_runtime_contracts_sha256": contracts_sha,
        "implementation_code_sha256s": code_hashes,
        "factor_governance_protocol_version": "v2",
        "factor_governance_protocol_hash": "d" * 64,
    }
    receipt["receipt_sha256"] = activation_receipt_payload_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(mode)
    receipt_file_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    result = validate_quant_production_activation(
        {
            "strict_loader": True,
            "path": str(registry_path),
            "registry_sha256": registry_sha,
        },
        {"production_factor_set_sha256": "b" * 64},
        contracts_sha,
        implementation_code_sha256s=code_hashes,
        protocol_version="v2",
        protocol_hash_value="d" * 64,
        environ={
            "QUANT_PRODUCTION_KILL_SWITCH": "false",
            "QUANT_PRODUCTION_ACTIVATION_RECEIPT": str(receipt_path),
            "QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256": receipt_file_sha,
        },
    )

    assert result["status"] == "governance_blocked"
    assert (
        "quant_production_activation_receipt_permissions_unsafe"
        in result["blockers"]
    )


@pytest.mark.parametrize("value", ["", "true", "TRUE", "0", "off", "invalid"])
def test_production_activation_rejects_noncanonical_or_active_switch(
    tmp_path: Path,
    value: str,
) -> None:
    result = validate_quant_production_activation(
        {},
        {"production_factor_set_sha256": "a" * 64},
        "b" * 64,
        implementation_code_sha256s={},
        protocol_version="v2",
        protocol_hash_value="c" * 64,
        environ={"QUANT_PRODUCTION_KILL_SWITCH": value},
    )
    assert result["status"] == "governance_blocked"


def test_registry_runtime_load_uses_strict_parser_without_forgiving_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    record = _record().to_dict()
    record["gate_results"][0]["passed"] = "true"
    path.write_text(
        json.dumps(
            {
                "schema_version": "mined-factor-registry.v1",
                "metadata": {},
                "factors": [record],
            }
        ),
        encoding="utf-8",
    )

    registry = MinedFactorRegistry.load_production(path)

    assert registry.factors == []
    assert registry.metadata["strict_loader"] is True
    assert "strict_load_error" in registry.metadata


def test_in_memory_production_registry_cannot_bypass_strict_provenance() -> None:
    status = governance_runtime_status(
        MinedFactorRegistry.from_records([_record()])
    )

    assert status["status"] == "governance_blocked"
    assert "registry_not_strictly_loaded" in status["blockers"]
    assert "production_runtime_contracts_missing" in status["blockers"]


def _frames(*, include_amount: bool = True) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for index in range(20):
        rows = 6
        symbol = f"S{index:02d}"
        frame = pd.DataFrame(
            {
                "ts_code": [symbol] * rows,
                "trade_date": pd.date_range("2026-01-01", periods=rows),
                "adj_close": np.linspace(10 + index, 11 + index, rows),
                "vol": np.array([100, 104, 108, 112, 116, 120 + index], dtype=float),
            }
        )
        if include_amount:
            frame["amount"] = np.linspace(1000 + index, 1200 + index, rows)
        result[symbol] = frame
    return result


def _evaluation_context(frames: dict[str, pd.DataFrame], tmp_path: Path):
    artifact_paths: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    payload = {
        "snapshot_id": "runtime-fixture",
        "latest_complete_trade_date": "20260106",
    }
    for name in ("snapshot_pointer", "snapshot_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifact_paths[name] = str(path.resolve())
        artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    calendar_path = tmp_path / "open_day_calendar.json"
    calendar_path.write_text(
        json.dumps(
            {
                "schema_version": "market-open-days.v1",
                "market": "CN",
                "open_dates": ["20260106"],
            }
        ),
        encoding="utf-8",
    )
    artifact_paths["open_day_calendar"] = str(calendar_path.resolve())
    artifact_hashes["open_day_calendar"] = hashlib.sha256(
        calendar_path.read_bytes()
    ).hexdigest()
    pit_canonical = tmp_path / "pit_canonical.parquet"
    pd.DataFrame(
        [
            PITUniverseRecord(
                symbol=symbol,
                source_list_status="L",
                list_date="20200101",
                effective_from="20200101",
                observed_at=PIT_OBSERVED_AT,
                source="tushare.stock_basic",
                source_run_id="pit-runtime-fixture",
                raw_payload_hash=f"fixture-{symbol}",
                membership_quality="ok",
            ).to_dict()
            for symbol in frames
        ]
    ).to_parquet(pit_canonical, index=False)
    pit_manifest = tmp_path / "pit_manifest.json"
    pit_manifest.write_text(
        json.dumps(
            {
                "schema_version": "cn_pit_universe_manifest.v1",
                "membership_schema_version": "cn_pit_universe.v1",
                "source": "tushare.stock_basic",
                "source_run_id": "pit-runtime-fixture",
                "observed_at": PIT_OBSERVED_AT,
                "row_count": len(frames),
                "canonical_path": str(pit_canonical.resolve()),
            }
        ),
        encoding="utf-8",
    )
    artifact_paths.update(
        {
            "pit_manifest": str(pit_manifest.resolve()),
            "pit_canonical": str(pit_canonical.resolve()),
        }
    )
    artifact_hashes.update(
        {
            "pit_manifest": hashlib.sha256(pit_manifest.read_bytes()).hexdigest(),
            "pit_canonical": hashlib.sha256(pit_canonical.read_bytes()).hexdigest(),
        }
    )
    return _mint_production_evaluation_context(
        evaluation_as_of="20260106",
        market="CN",
        universe_key="full_a",
        universe_sha256=production_symbol_set_sha256(list(frames)),
        snapshot_id="runtime-fixture",
        latest_complete_trade_date="20260106",
        pit_membership_status="verified",
        pit_membership_as_of="20260106",
        pit_membership_proof_sha256="a" * 64,
        pit_membership_not_applicable_reason="",
        open_day_proof_sha256=artifact_hashes["open_day_calendar"],
        read_result_provenance_sha256="c" * 64,
        verified_artifact_paths=artifact_paths,
        verified_artifact_sha256s=artifact_hashes,
    )


def _runtime_status_for(records: list[FactorRecord]) -> dict[str, object]:
    contracts: dict[str, dict[str, object]] = {}
    for record in records:
        required = (
            ["trade_date", "amount"]
            if "low_dollar_volume" in record.name
            else ["trade_date", "vol"]
        )
        contracts[record.name] = {
            "required_columns": required,
            "lookback_rows": 5,
            "gate2_min_coverage_rate": 1.0,
            "min_cross_section": 20,
        }
    return {
        "status": "ready",
        "factor_mode": "governed_mined_factors",
        "confidence_multiplier": 1.0,
        "production_eligible": True,
        "blockers": [],
        "factor_runtime_contracts": contracts,
    }


def test_any_factor_compute_failure_blocks_whole_branch_without_renormalizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [
        _record("pv_low_dollar_volume_5d", weight=0.05),
        _record("pv_volume_stability_5d", weight=0.05),
    ]
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records(records))
    monkeypatch.setattr(
        scorer,
        "_runtime_contract",
        lambda: (records, _runtime_status_for(records)),
    )
    original = scorer._price_volume_factor

    def fail_second(name, frames, **kwargs):
        if name == "pv_volume_stability_5d":
            raise RuntimeError("fixture compute failure")
        return original(name, frames, **kwargs)

    monkeypatch.setattr(scorer, "_price_volume_factor", fail_second)

    frames = _frames()
    result = scorer.score(
        frames,
        evaluation_context=_evaluation_context(frames, tmp_path),
    )

    assert result.factor_count == 0
    assert result.factors_used == []
    assert result.governance_status == "governance_blocked"
    assert any("factor_compute_error" in item for item in result.runtime_blockers)


def test_production_required_amount_cannot_fall_back_to_close_times_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _record()
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records([record]))
    monkeypatch.setattr(
        scorer,
        "_runtime_contract",
        lambda: ([record], _runtime_status_for([record])),
    )

    missing_amount_frames = _frames(include_amount=False)
    context = _evaluation_context(missing_amount_frames, tmp_path)
    result = scorer.score(
        missing_amount_frames,
        evaluation_context=context,
    )

    assert result.factor_count == 0
    assert result.governance_status == "governance_blocked"
    assert any("required_columns" in item for item in result.runtime_blockers)

    non_finite_frames = _frames()
    non_finite_frames["S00"].loc[
        non_finite_frames["S00"].index[-1], "amount"
    ] = np.inf
    non_finite = scorer.score(
        non_finite_frames,
        evaluation_context=_evaluation_context(non_finite_frames, tmp_path),
    )
    assert non_finite.governance_status == "governance_blocked"
    assert any("required_columns" in item for item in non_finite.runtime_blockers)


def test_production_unknown_implementation_has_no_name_based_fallback() -> None:
    record = _record(
        "momentum_1m",
        implementation="unregistered:momentum_1m",
    )
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records([record]))

    with pytest.raises(ValueError, match="not allowlisted"):
        scorer._compute_factor(record, _frames())


@pytest.mark.parametrize(
    "case",
    ["empty", "nonfinite", "constant", "foreign", "duplicate", "missing"],
)
def test_production_factor_output_must_be_finite_nonconstant_exact_symbol_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    record = _record()
    frames = _frames()
    symbols = list(frames)
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records([record]))
    monkeypatch.setattr(
        scorer,
        "_runtime_contract",
        lambda: ([record], _runtime_status_for([record])),
    )

    values = pd.Series(np.arange(20, dtype=float), index=symbols)
    if case == "empty":
        values = pd.Series(dtype=float)
    elif case == "nonfinite":
        values.iloc[-1] = np.inf
    elif case == "constant":
        values[:] = 1.0
    elif case == "foreign":
        values.index = [*symbols[:-1], "FOREIGN"]
    elif case == "duplicate":
        values.index = [*symbols[:-1], symbols[0]]
    elif case == "missing":
        values = values.iloc[:-1]
    monkeypatch.setattr(
        scorer,
        "_price_volume_factor",
        lambda *args, **kwargs: values,
    )

    result = scorer.score(
        frames,
        evaluation_context=_evaluation_context(frames, tmp_path),
    )

    assert result.factor_count == 0
    assert result.governance_status == "governance_blocked"
    assert result.runtime_blockers


def test_production_factor_requires_full_lookback_and_minimum_cross_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _record()
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records([record]))
    monkeypatch.setattr(
        scorer,
        "_runtime_contract",
        lambda: ([record], _runtime_status_for([record])),
    )

    short_history = {
        symbol: frame.tail(4)
        for symbol, frame in _frames().items()
    }
    short_result = scorer.score(
        short_history,
        evaluation_context=_evaluation_context(short_history, tmp_path),
    )
    assert short_result.governance_status == "governance_blocked"
    assert any("lookback" in item for item in short_result.runtime_blockers)

    small_cross_section = dict(list(_frames().items())[:19])
    small_result = scorer.score(
        small_cross_section,
        evaluation_context=_evaluation_context(small_cross_section, tmp_path),
    )
    assert small_result.governance_status == "governance_blocked"
    assert any("min_cross_section" in item for item in small_result.runtime_blockers)


def test_production_success_uses_exact_contract_factor_and_symbol_sets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [
        _record("pv_low_dollar_volume_5d"),
        _record("pv_volume_stability_5d"),
    ]
    frames = _frames()
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records(records))
    monkeypatch.setattr(
        scorer,
        "_runtime_contract",
        lambda: (records, _runtime_status_for(records)),
    )

    result = scorer.score(
        frames,
        evaluation_context=_evaluation_context(frames, tmp_path),
    )

    assert result.governance_status == "ready"
    assert result.factor_count == 2
    assert result.factors_used == [record.name for record in records]
    assert set(result.factor_weights) == set(result.factors_used)
    assert set(result.factor_coverages) == set(result.factors_used)
    assert set(result.symbol_scores) == set(frames)


def test_contract_set_hash_is_order_independent() -> None:
    first = {"b": {"value": 2}, "a": {"value": 1}}
    second = {"a": {"value": 1}, "b": {"value": 2}}
    assert production_runtime_contracts_sha256(first) == (
        production_runtime_contracts_sha256(second)
    )


def test_downstream_readiness_revalidates_real_strict_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import quant_investor.factors.governance_protocol_v2 as protocol_module

    monkeypatch.setattr(
        protocol_module,
        "canonical_replay_producer_control",
        lambda: {
            "producer_available": True,
            "artifact_bytes_readback_bound": True,
            "production_apply_eligible": True,
            "blocker": "",
        },
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"verified"}\n', encoding="utf-8")
    factor_names = [
        "pv_low_dollar_volume_5d",
        "pv_high_dollar_volume_5d",
        "pv_volume_stability_5d",
        "pv_momentum_5d",
        "pv_price_efficiency_5d",
    ]
    records = [_record(name, weight=0.20) for name in factor_names]
    for index, record in enumerate(records):
        record.metadata["factor_family"] = f"family-{index}"
        record.metadata["dominant_primitive_cluster"] = f"cluster-{index}"
    contracts = {record.name: _contract(record, evidence) for record in records}

    digest_contract = {
        records[0].name: contracts[records[0].name],
    }
    digest_frames = _frames()
    digest = production_runtime_input_sha256(digest_frames, digest_contract)
    assert digest == production_runtime_input_sha256(
        dict(reversed(list(digest_frames.items()))), digest_contract
    )
    reversed_rows = {
        symbol: frame.iloc[::-1].reset_index(drop=True)
        for symbol, frame in digest_frames.items()
    }
    assert digest == production_runtime_input_sha256(
        reversed_rows, digest_contract
    )
    outside_lookback = {
        symbol: pd.concat(
            [
                frame.iloc[[0]].assign(
                    trade_date=pd.Timestamp("2025-12-01"),
                    amount=1.0,
                ),
                frame,
            ],
            ignore_index=True,
        )
        for symbol, frame in digest_frames.items()
    }
    assert digest == production_runtime_input_sha256(
        outside_lookback, digest_contract
    )
    irrelevant_column = {symbol: frame.copy() for symbol, frame in digest_frames.items()}
    irrelevant_column[next(iter(irrelevant_column))].loc[:, "vol"] += 1.0
    assert digest == production_runtime_input_sha256(
        irrelevant_column, digest_contract
    )
    changed_value = {symbol: frame.copy() for symbol, frame in digest_frames.items()}
    changed_symbol = next(iter(changed_value))
    last_two = changed_value[changed_symbol].index[-2:]
    changed_value[changed_symbol].loc[last_two, "amount"] = (
        changed_value[changed_symbol].loc[last_two, "amount"].to_numpy()[::-1]
    )
    assert digest != production_runtime_input_sha256(changed_value, digest_contract)
    changed_date = {symbol: frame.copy() for symbol, frame in digest_frames.items()}
    changed_date[changed_symbol].loc[
        changed_date[changed_symbol].index[-1], "trade_date"
    ] += pd.Timedelta(days=1)
    assert digest != production_runtime_input_sha256(changed_date, digest_contract)
    two_contracts = {
        records[0].name: contracts[records[0].name],
        records[1].name: contracts[records[1].name],
    }
    assert production_runtime_input_sha256(digest_frames, two_contracts) == (
        production_runtime_input_sha256(
            digest_frames,
            dict(reversed(list(two_contracts.items()))),
        )
    )

    typed_contract = {
        "typed-factor": {
            "required_columns": ["trade_date", "value"],
            "lookback_rows": 2,
        }
    }
    typed_dates = pd.date_range("2026-01-01", periods=2)

    def typed_digest(values: list[object]) -> str:
        return production_runtime_input_sha256(
            {
                "TYPED": pd.DataFrame(
                    {
                        "trade_date": typed_dates,
                        "value": pd.Series(values, dtype=object),
                    }
                )
            },
            typed_contract,
        )

    assert typed_digest([True, False]) != typed_digest([1, 0])
    assert typed_digest([1, 0]) != typed_digest(["1", "0"])
    assert typed_digest([np.inf, 0.0]) != typed_digest([-np.inf, 0.0])
    assert typed_digest([None, pd.NA]) == typed_digest([np.nan, pd.NaT])

    benchmark_dates = pd.date_range("2026-01-01", periods=91)
    benchmark_frames = {
        f"B{index:04d}": pd.DataFrame(
            {
                "trade_date": benchmark_dates,
                "adj_close": np.linspace(10.0 + index, 11.0 + index, 91),
                "vol": np.linspace(1000.0, 1100.0, 91),
                "amount": np.linspace(10000.0, 11000.0, 91),
            }
        )
        for index in range(1000)
    }
    benchmark_contract = copy.deepcopy(digest_contract)
    benchmark_contract[records[0].name]["required_columns"] = [
        "trade_date",
        "adj_close",
        "vol",
        "amount",
    ]
    benchmark_contract[records[0].name]["lookback_rows"] = 91
    tracemalloc.start()
    production_runtime_input_sha256(benchmark_frames, benchmark_contract)
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_bytes < 25 * 1024 * 1024
    del benchmark_frames

    registry = MinedFactorRegistry.from_records(records)
    manifest = registry.selectable_manifest()
    registry.metadata = {
        **manifest,
        "factor_governance_protocol_version": "v2",
        "factor_governance_protocol_hash": protocol_hash(),
        "factor_governance_last_evidence_hash": "e" * 64,
        "factor_governance_last_evaluation_hash": "f" * 64,
        "factor_governance_evidence_schema": (
            "factor-governance-replay-evidence.v2"
        ),
        "factor_governance_production_apply_eligible": True,
        "factor_governance_production_apply_blocker": "",
        "production_factor_runtime_contracts": contracts,
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": registry.schema_version,
                "metadata": registry.metadata,
                "factors": [record.to_dict() for record in records],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    strict_registry = MinedFactorRegistry.load_production(registry_path)
    assert "strict_load_error" not in strict_registry.metadata
    contract_status = validate_production_runtime_contracts(
        strict_registry.selectable_factors(), strict_registry.metadata
    )
    assert contract_status["status"] == "ready"

    receipt_path = tmp_path / "activation_receipt.json"
    receipt = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "status": "authorized",
        "activation_id": "controlled-test-activation",
        "approved_by": "test-fixture",
        "issued_at": "2026-07-14T00:00:00Z",
        "kill_switch_value": "false",
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": strict_registry.metadata["registry_sha256"],
        "production_factor_set_sha256": manifest[
            "production_factor_set_sha256"
        ],
        "production_runtime_contracts_sha256": contract_status[
            "contracts_sha256"
        ],
        "implementation_code_sha256s": contract_status[
            "implementation_code_sha256s"
        ],
        "factor_governance_protocol_version": "v2",
        "factor_governance_protocol_hash": protocol_hash(),
    }
    receipt["receipt_sha256"] = activation_receipt_payload_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setenv("QUANT_PRODUCTION_KILL_SWITCH", "false")
    monkeypatch.setenv(
        "QUANT_PRODUCTION_ACTIVATION_RECEIPT", str(receipt_path)
    )
    monkeypatch.setenv(
        "QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256",
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )

    status = governance_runtime_status(strict_registry)
    assert status["status"] == "ready"
    factors_used = list(status["production_factor_names"])
    one_symbol_score = RuntimeFactorScore(
        symbol_scores={"TEST.SZ": 0.25},
        factor_count=len(factors_used),
        factors_used=factors_used,
        factor_weights={name: 0.20 for name in factors_used},
        factor_coverages={name: 1.0 for name in factors_used},
        registry_metadata={
            **strict_registry.metadata,
            "governance_runtime": {**status, "production_eligible": True},
        },
        governance_status="ready",
        factor_mode="governed_mined_factors",
        confidence_multiplier=1.0,
        production_eligible=True,
    )
    assert production_runtime_metadata_is_ready(
        one_symbol_score.to_metadata()
    ) is False

    frames = _frames()
    evaluation_context = _evaluation_context(frames, tmp_path)
    score = MinedFactorScorer(strict_registry).score(
        frames,
        evaluation_context=evaluation_context,
    )
    assert score.governance_status == "ready"

    ready_metadata = score.to_metadata()
    assert ready_metadata["symbol_count"] == 20
    assert len(ready_metadata["symbol_set_sha256"]) == 64
    assert len(ready_metadata["production_input_sha256"]) == 64
    assert len(ready_metadata["production_output_attestation_sha256"]) == 64

    def metadata_is_ready(metadata: dict[str, object]) -> bool:
        return production_runtime_metadata_is_ready(
            metadata,
            expected_symbols=list(frames),
            expected_symbol_scores=score.symbol_scores,
            expected_frames=frames,
            expected_evaluation_context=evaluation_context,
        )

    assert production_runtime_metadata_is_ready(ready_metadata) is False
    assert metadata_is_ready(ready_metadata) is True
    assert production_runtime_score_is_ready(
        score,
        expected_symbols=list(frames),
        expected_frames=frames,
        expected_evaluation_context=evaluation_context,
    ) is True
    reordered_frames = dict(reversed(list(frames.items())))
    assert production_runtime_score_is_ready(
        score,
        expected_symbols=list(reordered_frames),
        expected_frames=reordered_frames,
        expected_evaluation_context=evaluation_context,
    ) is True

    quant_result = BranchResult(
        branch_name="quant",
        final_score=float(np.mean(list(score.symbol_scores.values()))),
        final_confidence=0.72,
        symbol_scores=dict(score.symbol_scores),
        metadata={
            "governance_status": "ready",
            "factor_mode": "governed_mined_factors",
            "production_eligible": True,
            "mined_factor_runtime": ready_metadata,
        },
    )
    global_verdict = _build_global_quant_verdict(
        cross_section_quant={},
        symbol_count=len(frames),
        quant_result=quant_result,
        expected_frames=frames,
    )
    assert global_verdict.metadata["production_quant_evidence"] is False
    assert _build_global_quant_verdict(
        cross_section_quant={},
        symbol_count=len(frames),
        quant_result=quant_result,
    ).metadata["production_quant_evidence"] is False

    import quant_investor.factors.runtime as runtime_module
    import quant_investor.market.dag.packets as packets_module

    original_digest = runtime_module.production_runtime_input_sha256
    original_validate = runtime_module._validate_production_frames
    digest_call_count = 0
    validate_call_count = 0

    def counting_digest(runtime_frames, runtime_contracts):
        nonlocal digest_call_count
        digest_call_count += 1
        return original_digest(runtime_frames, runtime_contracts)

    def counting_validate(runtime_frames, *, symbols, context):
        nonlocal validate_call_count
        validate_call_count += 1
        return original_validate(
            runtime_frames,
            symbols=symbols,
            context=context,
        )

    monkeypatch.setattr(
        runtime_module,
        "production_runtime_input_sha256",
        counting_digest,
    )
    monkeypatch.setattr(
        runtime_module,
        "_validate_production_frames",
        counting_validate,
    )
    monkeypatch.setattr(
        packets_module,
        "production_runtime_input_sha256",
        counting_digest,
    )
    monkeypatch.setattr(
        packets_module,
        "score_with_mined_factors",
        lambda runtime_frames, *, evaluation_context=None: MinedFactorScorer(
            strict_registry
        ).score(
            runtime_frames,
            evaluation_context=evaluation_context,
        ),
    )
    validated_result, validation_token = (
        packets_module._build_quant_branch_result_with_validation(
            frames=frames,
            evaluation_context=evaluation_context,
        )
    )
    assert validation_token is not None
    with pytest.raises(TypeError):
        json.dumps(validation_token)
    validated_global = _build_global_quant_verdict(
        cross_section_quant={},
        symbol_count=len(frames),
        quant_result=validated_result,
        validation_token=validation_token,
    )
    assert validated_global.metadata["production_quant_evidence"] is True
    assert digest_call_count == 2
    assert validate_call_count == 1
    forged_validated_result = copy.deepcopy(validated_result)
    forged_validated_symbol = next(iter(forged_validated_result.symbol_scores))
    forged_validated_result.symbol_scores[forged_validated_symbol] *= -1.0
    forged_validated_result.final_score = float(
        np.mean(list(forged_validated_result.symbol_scores.values()))
    )
    assert _build_global_quant_verdict(
        cross_section_quant={},
        symbol_count=len(frames),
        quant_result=forged_validated_result,
        validation_token=validation_token,
    ).metadata["production_quant_evidence"] is False
    assert digest_call_count == 2
    assert validate_call_count == 1
    drifted_context_result = copy.deepcopy(validated_result)
    drifted_context_result.metadata["mined_factor_runtime"][
        "production_evaluation_context_sha256"
    ] = "d" * 64
    assert _build_global_quant_verdict(
        cross_section_quant={},
        symbol_count=len(frames),
        quant_result=drifted_context_result,
        validation_token=validation_token,
    ).metadata["production_quant_evidence"] is False

    coverage_below_exact = copy.deepcopy(ready_metadata)
    coverage_below_exact["factor_coverages"][factors_used[0]] = 0.99
    assert metadata_is_ready(coverage_below_exact) is False

    one_symbol_metadata = copy.deepcopy(ready_metadata)
    one_symbol_metadata["symbol_count"] = 1
    one_symbol_metadata["symbol_set_sha256"] = hashlib.sha256(
        json.dumps(
            ["TEST.SZ"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert metadata_is_ready(one_symbol_metadata) is False

    forged_symbol_count = copy.deepcopy(ready_metadata)
    forged_symbol_count["symbol_count"] = 21
    assert metadata_is_ready(forged_symbol_count) is False

    forged_symbol_set = copy.deepcopy(ready_metadata)
    forged_symbol_set["symbol_set_sha256"] = "9" * 64
    assert metadata_is_ready(forged_symbol_set) is False

    out_of_range = copy.deepcopy(score)
    out_of_range.symbol_scores[next(iter(out_of_range.symbol_scores))] = 1.1
    assert production_runtime_score_is_ready(
        out_of_range,
        expected_symbols=list(frames),
        expected_frames=frames,
        expected_evaluation_context=evaluation_context,
    ) is False

    forged_output = copy.deepcopy(score)
    first_symbol = next(iter(forged_output.symbol_scores))
    forged_output.symbol_scores[first_symbol] = (
        -0.5 if forged_output.symbol_scores[first_symbol] >= 0.0 else 0.5
    )
    assert production_runtime_score_is_ready(
        forged_output,
        expected_symbols=list(frames),
        expected_frames=frames,
        expected_evaluation_context=evaluation_context,
    ) is False
    forged_branch_result = copy.deepcopy(quant_result)
    forged_branch_result.symbol_scores[first_symbol] = (
        -0.5
        if forged_branch_result.symbol_scores[first_symbol] >= 0.0
        else 0.5
    )
    forged_branch_result.final_score = float(
        np.mean(list(forged_branch_result.symbol_scores.values()))
    )
    assert _build_global_quant_verdict(
        cross_section_quant={},
        symbol_count=len(frames),
        quant_result=forged_branch_result,
        expected_frames=frames,
    ).metadata["production_quant_evidence"] is False

    drifted_frames = {symbol: frame.copy() for symbol, frame in frames.items()}
    drifted_frames[first_symbol].loc[
        drifted_frames[first_symbol].index[-1], "amount"
    ] += 1.0
    assert production_runtime_score_is_ready(
        score,
        expected_symbols=list(frames),
        expected_frames=drifted_frames,
        expected_evaluation_context=evaluation_context,
    ) is False

    forged_contracts_sha = copy.deepcopy(ready_metadata)
    forged_contracts_sha["registry"]["governance_runtime"][
        "factor_runtime_contracts_sha256"
    ] = "a" * 64
    assert metadata_is_ready(forged_contracts_sha) is False

    forged_code_sha = copy.deepcopy(ready_metadata)
    forged_code_sha["registry"]["governance_runtime"][
        "factor_runtime_implementation_code_sha256s"
    ][factors_used[0]] = "b" * 64
    assert metadata_is_ready(forged_code_sha) is False

    forged_receipt_sha = copy.deepcopy(ready_metadata)
    forged_receipt_sha["registry"]["governance_runtime"][
        "quant_production_activation"
    ]["receipt_file_sha256"] = "c" * 64
    assert metadata_is_ready(forged_receipt_sha) is False
