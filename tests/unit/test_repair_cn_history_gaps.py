from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from quant_investor.market.cn_nontrading_evidence import canonical_json_sha256


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "repair_cn_history_gaps",
    ROOT / "scripts" / "repair_cn_history_gaps.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _audit_payload() -> dict:
    payload = {
        "schema_version": "myquant-cn-history-audit.v4",
        "audit_method": "full_recompute_from_canonical",
        "full_window_recomputed": True,
        "prior_trade_dates_reused": 0,
        "audited_trade_dates_count": 100,
        "audited_trade_dates": [f"2026{index:04d}" for index in range(100)],
        "canonical": {
            "snapshot_id": "snapshot",
            "storage_validation": {"status": "passed"},
        },
        "canonical_window_evidence": {"table_serving_match": True},
        "pit_membership_evidence": {"sha256": "a" * 64},
    }
    payload["audit_sha256"] = canonical_json_sha256(payload)
    return payload


def test_source_audit_payload_rejects_rehashed_stale_mutation() -> None:
    payload = _audit_payload()
    MODULE._validate_source_audit_payload(payload)

    payload["prior_trade_dates_reused"] = 95
    payload_without_sha = dict(payload)
    payload_without_sha.pop("audit_sha256")
    payload["audit_sha256"] = canonical_json_sha256(payload_without_sha)
    with pytest.raises(SystemExit, match="reused prior"):
        MODULE._validate_source_audit_payload(payload)


def test_repair_refuses_symbol_date_already_present_in_canonical() -> None:
    assert MODULE._stale_target_keys(
        {("20260701", "000001.SZ"), ("20260702", "000002.SZ")},
        {
            "20260701": ["000001.SZ", "000003.SZ"],
            "20260703": ["000004.SZ"],
        },
    ) == [("20260701", "000001.SZ")]
