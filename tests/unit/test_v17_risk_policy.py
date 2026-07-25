from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from quant_investor.v17.contracts import V17ContractError
from quant_investor.v17.risk_policy import (
    OWNER_MANDATE_VERSION,
    build_available_risk_policy_snapshot,
    build_unavailable_risk_policy_snapshot,
    seal_risk_policy_from_owner_mandate,
    validate_portfolio_risk_policy_snapshot,
)
from quant_investor.v17.semantic import seal_semantic


def _source_refs() -> list[dict[str, str]]:
    return [
        {
            "source_id": "owner-policy",
            "path": "data/private/v17_sources/objects/policy.json",
            "byte_sha256": "a" * 64,
            "semantic_sha256": "b" * 64,
        }
    ]


def _available() -> dict[str, object]:
    return build_available_risk_policy_snapshot(
        policy_id="cn-shadow-risk-v1",
        strategy_id="cn-shadow",
        market="CN",
        pit_cutoff="2026-07-21",
        as_of="2026-07-21T00:00:00Z",
        expires_at="2026-07-24T00:00:00Z",
        gross_cap=0.8,
        cash_floor=0.2,
        single_name_cap=0.1,
        industry_cap=0.3,
        cluster_cap=0.2,
        beta_cap=1.2,
        stress_loss_cap=0.15,
        adv20_participation_cap=0.05,
        turnover_cap=0.3,
        stress_scenario="cn_equity_stress_v1",
        source_refs=_source_refs(),
    )


def test_available_risk_policy_is_exact_nonauthorizing_and_cutoff_bound() -> None:
    policy = _available()
    assert policy["availability"] == "AVAILABLE"
    assert policy["authority"] is False
    assert validate_portfolio_risk_policy_snapshot(policy, cutoff="2026-07-22T00:00:00Z") == policy

    expired = {k: v for k, v in policy.items() if k != "semantic_sha256"}
    expired["expires_at"] = "2026-07-21T12:00:00Z"
    with pytest.raises(V17ContractError, match="expired"):
        validate_portfolio_risk_policy_snapshot(
            seal_semantic(expired), cutoff="2026-07-22T00:00:00Z"
        )

    expires_exactly = {k: v for k, v in policy.items() if k != "semantic_sha256"}
    expires_exactly["expires_at"] = "2026-07-22T00:00:00Z"
    with pytest.raises(V17ContractError, match="expired"):
        validate_portfolio_risk_policy_snapshot(
            seal_semantic(expires_exactly), cutoff="2026-07-22T00:00:00Z"
        )


def test_unavailable_risk_policy_cannot_carry_caps_or_refs() -> None:
    unavailable = build_unavailable_risk_policy_snapshot(
        policy_id="cn-shadow-risk-v1",
        strategy_id="cn-shadow",
        market="CN",
        reason="owner_policy_not_provided",
    )
    assert set(unavailable) == {
        "version",
        "policy_id",
        "strategy_id",
        "market",
        "availability",
        "reason",
        "authority",
        "semantic_sha256",
    }
    injected = {k: v for k, v in unavailable.items() if k != "semantic_sha256"}
    injected["gross_cap"] = 0.5
    with pytest.raises(V17ContractError, match="shape mismatch"):
        validate_portfolio_risk_policy_snapshot(seal_semantic(injected), cutoff=None)


def test_authority_true_and_semantic_tampering_fail_closed() -> None:
    policy = _available()
    authority = {k: v for k, v in policy.items() if k != "semantic_sha256"}
    authority["authority"] = True
    with pytest.raises(V17ContractError, match="authority"):
        validate_portfolio_risk_policy_snapshot(seal_semantic(authority), cutoff=None)

    tampered = dict(policy)
    tampered["gross_cap"] = 0.1
    with pytest.raises(ValueError, match="semantic_sha256 mismatch"):
        validate_portfolio_risk_policy_snapshot(tampered, cutoff=None)


def test_owner_mandate_seal_validates_before_0600_atomic_write(tmp_path: Path) -> None:
    snapshot = {k: v for k, v in _available().items() if k != "semantic_sha256"}
    mandate = {"version": OWNER_MANDATE_VERSION, "risk_policy": snapshot}
    raw = json.dumps(mandate, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    source = tmp_path / "owner.json"
    source.write_bytes(raw)
    source.chmod(0o600)
    expected = hashlib.sha256(raw).hexdigest()
    output_root = tmp_path / "sealed"
    output = output_root / "risk.json"

    sealed, output_sha = seal_risk_policy_from_owner_mandate(
        source,
        output,
        expected_owner_mandate_sha256=expected,
        output_root=output_root,
        validation_cutoff="2026-07-22T00:00:00Z",
    )
    assert sealed["authority"] is False
    assert output_sha == hashlib.sha256(output.read_bytes()).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    repeated, repeated_sha = seal_risk_policy_from_owner_mandate(
        source,
        output,
        expected_owner_mandate_sha256=expected,
        output_root=output_root,
        validation_cutoff="2026-07-22T00:00:00Z",
    )
    assert repeated == sealed
    assert repeated_sha == output_sha

    changed_snapshot = dict(snapshot)
    changed_snapshot["gross_cap"] = 0.7
    changed_mandate = {
        "version": OWNER_MANDATE_VERSION,
        "risk_policy": changed_snapshot,
    }
    changed_raw = (
        json.dumps(changed_mandate, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    changed_source = tmp_path / "changed-owner.json"
    changed_source.write_bytes(changed_raw)
    with pytest.raises(V17ContractError, match="different bytes"):
        seal_risk_policy_from_owner_mandate(
            changed_source,
            output,
            expected_owner_mandate_sha256=hashlib.sha256(changed_raw).hexdigest(),
            output_root=output_root,
            validation_cutoff="2026-07-22T00:00:00Z",
        )

    rejected = tmp_path / "rejected" / "risk.json"
    with pytest.raises(V17ContractError, match="mismatch"):
        seal_risk_policy_from_owner_mandate(
            source,
            rejected,
            expected_owner_mandate_sha256="0" * 64,
            output_root=rejected.parent,
            validation_cutoff="2026-07-22T00:00:00Z",
        )
    assert not rejected.parent.exists()
