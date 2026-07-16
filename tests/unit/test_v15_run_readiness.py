from __future__ import annotations

import stat
from pathlib import Path

import pytest

from quant_investor.monitoring.v15_run_readiness import (
    build_v15_run_readiness,
    canonical_sha256,
    load_v15_run_readiness,
    write_v15_run_readiness,
)


def _branches(*, ready: bool = True):
    return {
        name: {"status": "pass" if ready else "block", "blockers": [] if ready else ["missing"]}
        for name in ("quant", "fundamental", "macro")
    }


def _payload(**overrides):
    portfolio = {"valid": True, "target_weights": {"000001.SZ": 0.1}}
    values = {
        "run_id": "run-1",
        "generated_at": "2026-07-16T10:00:00+08:00",
        "analysis_trade_date": "2026-07-15",
        "market_data_ready": True,
        "market_data_blockers": [],
        "branch_readiness": _branches(),
        "branch_objects": {name: True for name in ("quant", "fundamental", "macro")},
        "factor_governance": {
            "governance_status": "ready",
            "production_eligible": True,
            "registry_file_sha256": "a" * 64,
            "production_factor_set_sha256": "b" * 64,
            "blockers": [],
        },
        "candidate_decision": {"candidate_decision_status": "complete", "blocker": ""},
        "portfolio_constructor": portfolio,
        "human_authorization": {
            "authorized": True,
            "run_id": "run-1",
            "analysis_trade_date": "2026-07-15",
            "portfolio_sha256": canonical_sha256(portfolio),
        },
        "risk_reduction_quote_gate": {"authorized": False, "blockers": ["fresh_quote_missing"]},
        "material_warnings": [],
    }
    values.update(overrides)
    return build_v15_run_readiness(**values)


def test_missing_human_authorization_fails_closed() -> None:
    payload = _payload(human_authorization=None)
    assert payload["new_risk_authorized"] is False
    assert "new_risk_human_authorization_missing_or_invalid" in payload["blockers"]


def test_materialized_and_ready_are_independent() -> None:
    payload = _payload(branch_readiness=_branches(ready=False))
    assert all(payload["branch_objects_materialized"].values())
    assert not any(payload["branch_data_ready"].values())
    assert payload["new_risk_authorized"] is False


def test_empty_candidate_has_exact_portfolio_blocker() -> None:
    payload = _payload(
        candidate_decision={
            "candidate_decision_status": "empty",
            "blocker": "no_candidate_selected_by_portfolio_constructor",
        },
        portfolio_constructor={"valid": True, "target_weights": {}},
        human_authorization=None,
    )
    assert payload["candidate_decision_status"] == "empty"
    assert "no_candidate_selected_by_portfolio_constructor" in payload["blockers"]


def test_atomic_owner_only_roundtrip(tmp_path: Path) -> None:
    payload = _payload()
    path = tmp_path / "v15_run_readiness.json"
    reference = write_v15_run_readiness(path, payload)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_v15_run_readiness(path, expected_sha256=reference["sha256"]) == payload
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_v15_run_readiness(path, expected_sha256="0" * 64)
