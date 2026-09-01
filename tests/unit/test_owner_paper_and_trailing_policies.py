from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAILING = (
    ROOT
    / "results/policies/risk/aggressive_tech_manufacturing/trailing-anchor.v1"
    / "owner-trailing-anchor-policy-20260901-v1.json"
)
PAPER = (
    ROOT
    / "results/policies/paper/aggressive_tech_manufacturing"
    / "owner-paper-risk-execution-policy-20260901-v1.json"
)


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert isinstance(value, dict)
    assert hashlib.sha256(raw).hexdigest()
    return value


def test_owner_trailing_anchor_policy_is_research_and_paper_only() -> None:
    policy = _load(TRAILING)
    assert policy["schema_version"] == "owner-trailing-anchor-policy.v1"
    assert policy["authority"] == {
        "research_threshold_calculation": True,
        "paper_risk_reduction_input": True,
        "store_mutation": False,
        "actual_holdings_mutation": False,
        "broker": False,
        "live_order": False,
        "live_execution": False,
        "trade": False,
    }
    anchors = {row["symbol"]: row for row in policy["anchors"]}
    assert set(anchors) == {
        "002008.SZ",
        "002384.SZ",
        "002463.SZ",
        "002916.SZ",
        "601899.SH",
        "605358.SH",
        "688183.SH",
    }
    for symbol in ("002008.SZ", "002384.SZ", "605358.SH"):
        assert anchors[symbol]["tracking_start_date"] == "20260901"
        assert anchors[symbol]["anchor_state"] == "OWNER_APPROVED_RESET_NO_STRUCTURED_BUY_FOUND"
        assert anchors[symbol]["exclude_pre_anchor_peaks"] is True
    assert anchors["002463.SZ"]["anchor_ref"]["shares"] == 700
    assert anchors["002463.SZ"]["anchor_ref"]["final_total_fee_cny"] == "9.21"


def test_owner_paper_policy_never_grants_real_trading_authority() -> None:
    policy = _load(PAPER)
    assert policy["schema_version"] == "owner-paper-risk-execution-policy.v1"
    assert policy["account_scope"] == "ALL_REGISTERED_PAPER_ACCOUNTS"
    assert policy["automatic_paper_execution"] is True
    assert policy["action_scope"] == "RISK_REDUCING_SELLS_ONLY"
    authority = policy["authority"]
    assert authority["paper_order"] is True
    assert authority["paper_fill"] is True
    assert authority["paper_ledger_mutation"] is True
    assert authority["broker"] is False
    assert authority["live_order"] is False
    assert authority["live_execution"] is False
    assert authority["actual_holdings_mutation"] is False
    assert authority["funds_transfer"] is False
    assert policy["real_trading_authority"] is False
    assert policy["execution_policy"]["adverse_slippage_fraction"] == "0.05"
    assert policy["execution_policy"]["price_type"] == "NEXT_VALID_TRADING_DAY_OPEN"
    assert policy["execution_policy"]["partial_fill"] == "DISABLED_FULL_VALID_LOT_OR_NO_FILL"
    assert policy["fees"] == {
        "broker_commission_rate": "0.0001",
        "broker_commission_minimum_cny": "5.00",
        "commission_includes_regulatory_and_handling": True,
        "transfer_fee_rate": "0.00001",
        "transfer_fee_sides": "BUY_AND_SELL",
        "stamp_duty_rate": "0.0005",
        "stamp_duty_side": "SELL_ONLY",
        "fee_rounding": "CNY_0.01_HALF_UP",
    }
    assert policy["writer_state"] == "POLICY_READY_REGISTERED_PAPER_WRITER_REQUIRED"
