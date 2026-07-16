from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import quant_investor.factors.governance_protocol_v3 as factor_protocol_v3
from scripts.check_cn_dashboard_export import (
    DASHBOARD_SCHEMA_VERSION,
    validate_dashboard_contract_v3,
)
from scripts.export_cn_aggressive_dashboard_data import (
    _factor_canonical_producer_control,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v3_sample_has_industries_and_no_theme_surface() -> None:
    payload = json.loads(
        (ROOT / "portfolio_dashboard/sample/dashboard_snapshot.v3.json").read_text()
    )
    assert payload["schema_version"] == "dashboard_contract.v3"
    assert "industries" in payload
    assert "themes" not in payload
    assert "theme_protocol" not in payload


def test_checker_rejects_v2_contract() -> None:
    errors, _ = validate_dashboard_contract_v3({"schema_version": "dashboard_contract.v2"})
    assert DASHBOARD_SCHEMA_VERSION == "dashboard_contract.v3"
    assert any("schema_version" in error for error in errors)


def test_dashboard_accepts_producer_implemented_compatibility_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factor_protocol_v3,
        "canonical_replay_producer_control",
        lambda: {
            "producer_implemented": True,
            "production_apply_eligible": False,
            "blocker": "factor_v3_canonical_producer_not_authenticated",
        },
    )

    control = _factor_canonical_producer_control()

    assert control["producer_available"] is True
    assert control["production_apply_eligible"] is False


def test_checker_requires_pointer_bound_immutable_cn_market_evidence() -> None:
    payload = json.loads(
        (ROOT / "portfolio_dashboard/sample/dashboard_snapshot.v3.json").read_text()
    )
    mask_payload = {
        "source_system": "strict_parquet.cn_bars.trade_date",
        "start_date": "2026-07-15",
        "end_date": "2026-07-15",
        "expected_open_dates": ["2026-07-15"],
    }
    evidence = {
        "latest_pointer_path_summary": "<data_root>/parquet/cn/_latest.json",
        "latest_pointer_sha256": "a" * 64,
        "snapshot_id": "snap-v4",
        "table_root_path_summary": "<data_root>/parquet/cn/bars",
        "latest_complete_trade_date": "20260715",
        "fallback_used": False,
    }
    payload["trading_calendar"] = {
        "status": "available",
        "source_system": "strict_parquet.cn_bars.trade_date",
        "path_summary": "<data_root>/parquet/cn/bars",
        "market_snapshot": evidence,
        **mask_payload,
        "expected_open_date_count": 1,
        "first_open_date": "2026-07-15",
        "last_open_date": "2026-07-15",
        "mask_sha256": hashlib.sha256(
            json.dumps(
                mask_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    payload.setdefault("sources", {})["cn_market_snapshot"] = dict(evidence)

    errors, _ = validate_dashboard_contract_v3(payload)
    assert any("pointer-bound immutable snapshot root" in error for error in errors)
    assert any("forbidden fixed/serving/CSV root" in error for error in errors)

    immutable_root = "<data_root>/parquet/cn/_snapshots/snap-v4/table/bars"
    payload["trading_calendar"]["market_snapshot"][
        "table_root_path_summary"
    ] = immutable_root
    payload["sources"]["cn_market_snapshot"][
        "table_root_path_summary"
    ] = immutable_root
    payload["trading_calendar"]["path_summary"] = immutable_root
    errors, _ = validate_dashboard_contract_v3(payload)
    assert not any("pointer-bound immutable snapshot root" in error for error in errors)
    assert not any("forbidden fixed/serving/CSV root" in error for error in errors)
