"""Owner-declared executions: the owner names the price, everything else holds."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import importlib.util
from pathlib import Path

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.paper.contracts import PaperError, seal_document
from quant_investor.paper.execution import execute_owner_sell

SPEC = importlib.util.spec_from_file_location(
    "paper_entry_fixtures", Path(__file__).parent / "test_paper_entry.py"
)
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)


def _owner_intent(
    workspace: Path, position: dict, *, price: str, shares: int = 500
) -> tuple[dict, dict]:
    instruction = workspace / "inputs/owner-instruction.json"
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_bytes(b'{"instruction":"legacy exit"}')
    instruction.chmod(0o600)
    value = seal_document(
        {
            "schema_version": "paper-owner-intent.v1",
            "source_intent_id": "paper-605358-sh-20260930-owner-exit-100",
            "idempotency_key_sha256": "0" * 64,
            "economic_action_key_sha256": "1" * 64,
            "account_id": "paper-alpha",
            "strategy_id": "aggressive_tech_manufacturing",
            "signal_date": "20260930",
            "eligible_from_trade_date": "20260930",
            "symbol": position["symbol"],
            "action": "EXIT_100",
            "requested_ratio": "1.00",
            "requested_shares": shares,
            "price_cny": price,
            "price_basis": "OWNER_DECLARED_STRICT_CLOSE",
            "owner_instruction_ref": fixtures._relative(workspace, instruction),
            "reason_codes": ["OWNER_DIRECTED_LEGACY_EXIT"],
            "policy_ref": {
                "path": fixtures.contracts.POLICY_RELATIVE_PATH,
                "sha256": fixtures.contracts.POLICY_SHA256,
            },
            "expected_account_pointer_sha256": "EMPTY",
            "expected_position": {
                "shares": position["shares"],
                "settled_shares": position["settled_shares"],
                "avg_cost": f"{float(position['avg_cost']):.4f}",
            },
            "evidence_refs": [],
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
    )
    ref = {
        "path": "inputs/owner-intent.json",
        "sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    }
    return value, ref


def _position() -> dict:
    return {"symbol": "688183.SH", "shares": 500, "settled_shares": 500, "avg_cost": "65.2600"}


def _eligibility(workspace: Path, intent_ref: dict, *, limit_up="50.00", limit_down="40.00"):
    _path, value, ref = fixtures._entry_eligibility(
        workspace, intent_ref, open_price="45.00", limit_up=limit_up, limit_down=limit_down
    )
    return value, ref


def test_owner_sell_fills_at_the_named_price_without_slippage(tmp_path: Path) -> None:
    workspace = fixtures._workspace(tmp_path)
    position = _position()
    intent, intent_ref = _owner_intent(workspace, position, price="45.3600")
    eligibility, eligibility_ref = _eligibility(workspace, intent_ref)
    outcome = execute_owner_sell(
        intent=intent,
        intent_ref=intent_ref,
        eligibility=eligibility,
        eligibility_ref=eligibility_ref,
        position=position,
        cash_before=Decimal("1000000.0000"),
        evaluated_open_session_count=1,
    )
    assert outcome["outcome"] == "FILLED"
    assert outcome["order"]["simulated_price"] == "45.3600"
    assert outcome["order"]["price_type"] == "OWNER_DECLARED_STRICT_CLOSE"
    accounting = outcome["accounting"]
    assert accounting["shares_sold"] == 500
    # 500 * 45.36 = 22,680; stamp 11.34 + commission 5.00 + transfer 0.23
    assert outcome["fill"]["total_fees"] == "16.5700"
    assert accounting["cash_after"] == "1022663.4300"
    assert accounting["realized_pnl_delta"] == "-9966.5700"


def test_owner_price_outside_the_session_band_is_rejected(tmp_path: Path) -> None:
    workspace = fixtures._workspace(tmp_path)
    position = _position()
    intent, intent_ref = _owner_intent(workspace, position, price="60.0000")
    eligibility, eligibility_ref = _eligibility(workspace, intent_ref)
    with pytest.raises(PaperError, match="PAPER_OWNER_PRICE_OUTSIDE_LIMIT"):
        execute_owner_sell(
            intent=intent,
            intent_ref=intent_ref,
            eligibility=eligibility,
            eligibility_ref=eligibility_ref,
            position=position,
            cash_before=Decimal("1000000.0000"),
            evaluated_open_session_count=1,
        )


def test_owner_share_count_must_match_the_position(tmp_path: Path) -> None:
    workspace = fixtures._workspace(tmp_path)
    position = _position()
    intent, intent_ref = _owner_intent(workspace, position, price="45.3600", shares=300)
    eligibility, eligibility_ref = _eligibility(workspace, intent_ref)
    with pytest.raises(PaperError, match="PAPER_POSITION_MISMATCH"):
        execute_owner_sell(
            intent=intent,
            intent_ref=intent_ref,
            eligibility=eligibility,
            eligibility_ref=eligibility_ref,
            position=position,
            cash_before=Decimal("1000000.0000"),
            evaluated_open_session_count=1,
        )


def test_owner_sell_pends_while_the_session_has_not_arrived(tmp_path: Path) -> None:
    workspace = fixtures._workspace(tmp_path)
    position = _position()
    intent, intent_ref = _owner_intent(workspace, position, price="45.3600")
    intent = dict(intent, eligible_from_trade_date="20261009")
    eligibility, eligibility_ref = _eligibility(workspace, intent_ref)
    outcome = execute_owner_sell(
        intent=intent,
        intent_ref=intent_ref,
        eligibility=eligibility,
        eligibility_ref=eligibility_ref,
        position=position,
        cash_before=Decimal("1000000.0000"),
        evaluated_open_session_count=1,
    )
    assert outcome["outcome"] == "PENDING"
    assert outcome["pending"]["blocker_codes"] == ["PAPER_NEXT_SESSION_NOT_REACHED"]
