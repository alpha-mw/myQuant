from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.governance import (
    GATE_SPECS,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import MinedFactorRegistry, MinedFactorScorer
from quant_investor.factors.governance_protocol_v2 import governance_runtime_status
from quant_investor.factors.runtime_contract import (
    ACTIVATION_RECEIPT_SCHEMA_VERSION,
    RUNTIME_CONTRACT_SCHEMA_VERSION,
    activation_receipt_payload_sha256,
    factor_definition_sha256,
    factor_record_payload_sha256,
    implementation_code_sha256,
    production_runtime_contracts_sha256,
    validate_production_runtime_contracts,
    validate_quant_production_activation,
)


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
    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "factor_name": record.name,
        "factor_version": record.version,
        "implementation_id": record.implementation,
        "implementation_version": "price-volume-runtime.v1",
        "implementation_code_sha256": implementation_code_sha256(
            record.implementation
        ),
        "required_columns": ["trade_date", "amount"],
        "data_semantics": "strict-parquet-cn-daily-adjusted.v1",
        "lookback_rows": 5,
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
) -> dict[str, object]:
    return {
        "production_factor_runtime_contracts": {record.name: contract},
        "record_sha256s": {
            record.name: factor_record_payload_sha256(record),
        },
    }


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
    metadata = _contract_metadata(record, contract)
    ready = validate_production_runtime_contracts([record], metadata)
    assert ready["status"] == "ready"
    assert ready["blockers"] == []

    contract["implementation_id"] = "alpha_mining.FactorLibrary:momentum_1m"
    blocked = validate_production_runtime_contracts([record], metadata)
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
        _contract_metadata(record, contract),
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
        _contract_metadata(record, contract),
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
        _contract_metadata(record, contract),
    )

    assert result["status"] == "governance_blocked"
    assert any("sha256" in item for item in result["blockers"])


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
        frame = pd.DataFrame(
            {
                "trade_date": pd.date_range("2026-01-01", periods=rows),
                "adj_close": np.linspace(10 + index, 11 + index, rows),
                "vol": np.array([100, 104, 108, 112, 116, 120 + index], dtype=float),
            }
        )
        if include_amount:
            frame["amount"] = np.linspace(1000 + index, 1200 + index, rows)
        result[f"S{index:02d}"] = frame
    return result


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

    result = scorer.score(_frames())

    assert result.factor_count == 0
    assert result.factors_used == []
    assert result.governance_status == "governance_blocked"
    assert any("factor_compute_error" in item for item in result.runtime_blockers)


def test_production_required_amount_cannot_fall_back_to_close_times_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records([record]))
    monkeypatch.setattr(
        scorer,
        "_runtime_contract",
        lambda: ([record], _runtime_status_for([record])),
    )

    result = scorer.score(_frames(include_amount=False))

    assert result.factor_count == 0
    assert result.governance_status == "governance_blocked"
    assert any("required_columns" in item for item in result.runtime_blockers)

    non_finite_frames = _frames()
    non_finite_frames["S00"].loc[
        non_finite_frames["S00"].index[-1], "amount"
    ] = np.inf
    non_finite = scorer.score(non_finite_frames)
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

    result = scorer.score(frames)

    assert result.factor_count == 0
    assert result.governance_status == "governance_blocked"
    assert result.runtime_blockers


def test_production_factor_requires_full_lookback_and_minimum_cross_section(
    monkeypatch: pytest.MonkeyPatch,
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
    short_result = scorer.score(short_history)
    assert short_result.governance_status == "governance_blocked"
    assert any("lookback" in item for item in short_result.runtime_blockers)

    small_cross_section = dict(list(_frames().items())[:19])
    small_result = scorer.score(small_cross_section)
    assert small_result.governance_status == "governance_blocked"
    assert any("min_cross_section" in item for item in small_result.runtime_blockers)


def test_production_success_uses_exact_contract_factor_and_symbol_sets(
    monkeypatch: pytest.MonkeyPatch,
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

    result = scorer.score(frames)

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
