from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath

import pytest

from quant_investor.contracts import canonical_json_bytes, get_contract
from quant_investor.factors.governance.errors import FactorGovernanceError
import quant_investor.factors.production_observation as observation_module
from quant_investor.factors.production_observation import (
    _register_factor_production_observations,
    build_factor_production_observation,
    validate_factor_production_observation,
)


def _ref(kind: str, token: str) -> dict[str, str]:
    return {
        "kind": kind,
        "contract_sha256": token * 64,
        "artifact_id": f"artifact-{token}",
        "semantic_sha256": token * 64,
        "byte_sha256": token * 64,
    }


def _inputs() -> dict:
    return {
        "signal_date": "20260820",
        "factor_generation_id": "factor-production-generation-" + "1" * 64,
        "factor_generation_sha256": "2" * 64,
        "factor_pointer_sha256": "3" * 64,
        "market_pointer_sha256": "4" * 64,
        "market_manifest_sha256": "5" * 64,
        "pit_pointer_sha256": "6" * 64,
        "pit_manifest_sha256": "7" * 64,
        "pit_membership_sha256": "8" * 64,
        "calendar_compilation_ref": _ref("system.trusted_provider_calendar_compilation", "9"),
        "calendar_capture_custody_attestation_ref": _ref(
            "factor.production_calendar_capture_custody_attestation", "a"
        ),
        "factor_rows": [
            {
                "factor_id": "pv_low_dollar_volume_5d",
                "factor_alias": "LOW",
                "signal_sha256": "b" * 64,
                "signal_symbol_set_sha256": "c" * 64,
                "symbol_count": 5217,
            },
            {
                "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
                "factor_alias": "W80",
                "signal_sha256": "d" * 64,
                "signal_symbol_set_sha256": "c" * 64,
                "symbol_count": 5217,
            },
        ],
    }


def test_production_observation_contract_is_distinct_from_admission_observation() -> None:
    production = get_contract("factor.production_observation")
    prospective = get_contract("factor.prospective_observation")
    assert production.identity_field == "factor_production_observation_id"
    assert prospective.identity_field == "observation_id"
    assert production.contract_sha256 != prospective.contract_sha256


def test_observation_is_exact_immutable_non_authorizing_and_deterministic() -> None:
    inputs = _inputs()
    first = build_factor_production_observation(
        inputs=inputs,
        factor_row=inputs["factor_rows"][0],
        registered_at="2026-08-20T13:00:00Z",
    )
    second = build_factor_production_observation(
        inputs=inputs,
        factor_row=inputs["factor_rows"][0],
        registered_at="2026-08-20T13:00:00Z",
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    payload = validate_factor_production_observation(first)["payload"]
    assert payload["state"] == "OPEN"
    assert payload["authority"] == "NON_AUTHORIZING"
    assert payload["planned_horizons"] == [1, 5, 20, 60]
    assert payload["return_formula"] == "close(T+h)/close(T)-1"
    assert all(
        payload[field] == "NONE"
        for field in (
            "system_authority",
            "mainline_authority",
            "investment_authority",
            "portfolio_authority",
            "strategy_record_authority",
            "broker_authority",
        )
    )
    tampered = {**first, "payload": {**payload, "signal_sha256": "e" * 64}}
    with pytest.raises(FactorGovernanceError, match="contract failed"):
        validate_factor_production_observation(tampered)


def test_observation_rejects_registration_before_signal_date() -> None:
    inputs = _inputs()
    with pytest.raises(FactorGovernanceError, match="precedes its signal date"):
        build_factor_production_observation(
            inputs=inputs,
            factor_row=inputs["factor_rows"][0],
            registered_at="2026-08-19T23:59:59Z",
        )


@dataclass(frozen=True)
class _Stored:
    data: bytes

    @property
    def byte_sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


class _FakeStore:
    files: dict[str, bytes] = {}
    inputs = _inputs()

    def __init__(self, _workspace_root: str) -> None:
        pass

    @contextmanager
    def _active_lock(self):
        yield

    def read_active_observation_inputs(self) -> dict:
        return self.inputs

    def read_optional(self, value: PurePosixPath):
        raw = self.files.get(str(value))
        return None if raw is None else _Stored(raw)

    def write_exact_once(self, value: PurePosixPath, raw: bytes) -> _Stored:
        path = str(value)
        existing = self.files.setdefault(path, raw)
        if existing != raw:
            raise FactorGovernanceError("Factor authority immutable artifact conflicts")
        return _Stored(existing)


def test_registry_writes_two_observations_then_replays_as_no_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeStore.files = {}
    _FakeStore.inputs = _inputs()
    monkeypatch.setattr(observation_module, "FactorProductionStore", _FakeStore)
    first = _register_factor_production_observations(
        "/workspace", registered_at="2026-08-20T13:00:00Z"
    )
    second = _register_factor_production_observations(
        "/workspace", registered_at="2026-08-20T13:00:00Z"
    )
    assert first["command_status"] == "REGISTERED"
    assert first["created_count"] == 2
    assert second["command_status"] == "NO_ACTION"
    assert second["created_count"] == 0
    assert first["observations"] == second["observations"]
    assert sorted(_FakeStore.files) == [
        "results/factors/observations/2026/08/20/LOW.json",
        "results/factors/observations/2026/08/20/W80.json",
    ]


def test_registry_fails_closed_when_same_date_path_binds_different_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeStore.files = {}
    _FakeStore.inputs = _inputs()
    monkeypatch.setattr(observation_module, "FactorProductionStore", _FakeStore)
    _register_factor_production_observations("/workspace", registered_at="2026-08-20T13:00:00Z")
    changed = _inputs()
    changed["factor_rows"][0]["signal_sha256"] = "e" * 64
    _FakeStore.inputs = changed
    with pytest.raises(FactorGovernanceError, match="immutable path conflicts"):
        _register_factor_production_observations("/workspace", registered_at="2026-08-20T13:00:01Z")
