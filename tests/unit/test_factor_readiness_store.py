from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_investor.factors.readiness_store import (
    FactorReadinessStoreError,
    install_factor_readiness_exact_once,
    read_factor_readiness,
    validate_factor_readiness,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "factor-governance-readiness.v4",
        "status": "no_new_risk",
        "factor_governance_ready": False,
        "new_risk_eligible": False,
        "new_risk_authorized": False,
        "production_apply_enabled": False,
        "activation_receipt": {
            "valid": False,
            "receipt": None,
            "blockers": ["activation_receipt_missing"],
        },
    }


def _write_private(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def test_exact_once_install_is_private_byte_identical_and_idempotent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "readiness.json"
    target = tmp_path / "neutral" / "readiness.json"
    _write_private(source, _payload())

    assert (
        install_factor_readiness_exact_once(
            source_path=source,
            target_path=target,
        )
        == "installed"
    )
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert read_factor_readiness(target)["status"] == "no_new_risk"

    assert (
        install_factor_readiness_exact_once(
            source_path=source,
            target_path=target,
        )
        == "already_installed_identical"
    )


def test_exact_once_install_never_overwrites_different_target(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    _write_private(source, _payload())
    changed = _payload()
    changed["blockers"] = ["different"]
    _write_private(target, changed)
    before = target.read_bytes()

    with pytest.raises(FactorReadinessStoreError, match="differs from source"):
        install_factor_readiness_exact_once(
            source_path=source,
            target_path=target,
        )

    assert target.read_bytes() == before


def test_invalid_source_fails_before_target_parent_or_file_is_created(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "neutral" / "readiness.json"
    invalid = _payload()
    invalid["new_risk_authorized"] = True
    _write_private(source, invalid)

    with pytest.raises(FactorReadinessStoreError, match="activation authority"):
        install_factor_readiness_exact_once(
            source_path=source,
            target_path=target,
        )

    assert not target.exists()
    assert not target.parent.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("factor_governance_ready", True),
        ("new_risk_eligible", True),
        ("new_risk_authorized", True),
        ("production_apply_enabled", True),
    ],
)
def test_readiness_authority_is_always_rejected(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(FactorReadinessStoreError, match="activation authority"):
        validate_factor_readiness(payload)


def test_non_null_or_hashed_activation_receipt_is_rejected() -> None:
    payload = _payload()
    payload["activation_receipt"] = {
        "valid": False,
        "receipt": {"authorization_scope": "unexpected"},
        "blockers": ["invalid"],
    }
    with pytest.raises(FactorReadinessStoreError, match="payload must be null"):
        validate_factor_readiness(payload)

    payload = _payload()
    payload["activation_receipt"] = {
        "valid": False,
        "receipt": None,
        "receipt_sha256": "0" * 64,
        "blockers": ["invalid"],
    }
    with pytest.raises(FactorReadinessStoreError, match="SHA must be absent"):
        validate_factor_readiness(payload)
