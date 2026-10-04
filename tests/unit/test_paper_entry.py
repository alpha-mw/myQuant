"""Policy-bounded Paper entries: buy math, T+1 lots and store application."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.paper import contracts
from quant_investor.paper.contracts import PaperError, seal_document
from quant_investor.paper.execution import (
    calculate_buy_fees,
    calculate_buy_shares,
    economic_action_key,
    execute_buy,
    execute_sell,
)
from quant_investor.paper.store import PaperStore

ROOT = Path(__file__).resolve().parents[2]


def _relative(workspace: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(workspace).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    policy_target = workspace / contracts.POLICY_RELATIVE_PATH
    policy_target.parent.mkdir(parents=True, mode=0o700)
    shutil.copyfile(ROOT / contracts.POLICY_RELATIVE_PATH, policy_target)
    policy_target.chmod(0o600)
    return workspace


def _registration(workspace: Path) -> dict:
    source = workspace / "inputs/paper-genesis.json"
    source.parent.mkdir(parents=True, mode=0o700)
    source.write_bytes(b'{"owner":"maxwell","paper":true}')
    source.chmod(0o600)
    return seal_document(
        {
            "schema_version": "paper-account-registration.v1",
            "account_id": "paper-alpha",
            "account_type": "PAPER",
            "strategy_id": "aggressive_tech_manufacturing",
            "currency": "CNY",
            "allowed_writer_id": contracts.WRITER_ID,
            "policy_ref": {
                "path": contracts.POLICY_RELATIVE_PATH,
                "sha256": contracts.POLICY_SHA256,
            },
            "genesis_source_ref": _relative(workspace, source),
            "initial_cash": "1000000.0000",
            "initial_positions": [
                {
                    "symbol": "002916.SZ",
                    "name": "深南电路",
                    "shares": 200,
                    "settled_shares": 200,
                    "avg_cost": "100.0000",
                    "cost_basis": "20000.0000",
                    "realized_pnl": "0.0000",
                    "cumulative_fees": "0.0000",
                    "acquisition_lots": [
                        {
                            "shares": 200,
                            "acquisition_date": "20260810",
                            "settlement_date": "20260811",
                        }
                    ],
                }
            ],
            "all_initial_shares_settled": True,
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
    )


def _entry_intent(
    workspace: Path, pointer_sha: str, *, nav: str = "1000000.0000"
) -> tuple[Path, dict, dict[str, str]]:
    economic = economic_action_key(
        account_id="paper-alpha",
        policy_id=contracts.POLICY_ID,
        signal_date="20260930",
        symbol="688183.SH",
        action="ENTRY",
        shares=0,
    )
    value = seal_document(
        {
            "schema_version": "paper-entry-intent.v1",
            "source_intent_id": "intent-shengyi-20260930-entry",
            "idempotency_key_sha256": economic,
            "economic_action_key_sha256": economic,
            "account_id": "paper-alpha",
            "strategy_id": "aggressive_tech_manufacturing",
            "signal_date": "20260930",
            "eligible_from_trade_date": "20261008",
            "symbol": "688183.SH",
            "name": "生益电子",
            "side": "BUY",
            "action": "ENTRY",
            "target_weight": "0.14",
            "minimum_cash_fraction": "0.05",
            "account_nav_cny": nav,
            "nav_evidence_ref": _relative(workspace, _seed(workspace, "nav")),
            "reason_codes": ["POOL_COMBINED_PERCENTILE_GE_0_90"],
            "policy_ref": {
                "path": contracts.POLICY_RELATIVE_PATH,
                "sha256": contracts.POLICY_SHA256,
            },
            "expected_account_pointer_sha256": pointer_sha,
            "expected_position": None,
            "evidence_refs": [],
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
    )
    path = workspace / "inputs/entry-intent.json"
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o600)
    return path, value, _relative(workspace, path)


def _seed(workspace: Path, name: str) -> Path:
    path = workspace / f"inputs/{name}.json"
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.write_bytes(canonical_json_bytes({"name": name}))
    path.chmod(0o600)
    return path


def _entry_eligibility(
    workspace: Path,
    intent_ref: dict[str, str],
    *,
    symbol: str = "688183.SH",
    signal_date: str = "20260930",
    open_price: str = "100.0000",
    limit_up: str = "120.0000",
    limit_down: str = "80.0000",
    suspended: bool = False,
    corporate: str = "CLEAR",
    trade_date: str = "20261008",
) -> tuple[Path, dict, dict[str, str]]:
    refs = {
        name: _relative(workspace, _seed(workspace, name))
        for name in ("calendar", "bar", "limit", "suspension", "corporate")
    }
    value = seal_document(
        {
            "schema_version": "paper-input-eligibility.v1",
            "account_id": "paper-alpha",
            "source_intent_ref": intent_ref,
            "symbol": symbol,
            "signal_date": signal_date,
            "eligible_trade_date": "20261008",
            "evaluated_trade_date": trade_date,
            "open_price": open_price,
            "previous_close": "100.0000",
            "limit_up": limit_up,
            "limit_down": limit_down,
            "suspended": suspended,
            "corporate_action_state": corporate,
            "open_session_ordinal": 1,
            "expiry_session_ordinal": 3,
            "calendar_ref": refs["calendar"],
            "raw_bar_ref": refs["bar"],
            "price_limit_ref": refs["limit"],
            "suspension_ref": refs["suspension"],
            "corporate_action_ref": refs["corporate"],
            "evidence_status": "READY",
        }
    )
    path = workspace / "inputs/entry-eligibility.json"
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o600)
    return path, value, _relative(workspace, path)


def _buy(
    workspace,
    intent_value,
    intent_ref,
    eligibility_value,
    eligibility_ref,
    *,
    position=None,
    cash="1000000.0000",
    nav=None,
):
    return execute_buy(
        intent=intent_value,
        intent_ref=intent_ref,
        eligibility=eligibility_value,
        eligibility_ref=eligibility_ref,
        position=position,
        cash_before=Decimal(cash),
        account_nav=Decimal(nav if nav is not None else intent_value["account_nav_cny"]),
        evaluated_open_session_count=1,
    )


def test_buy_fees_charge_commission_and_transfer_only() -> None:
    fees = calculate_buy_fees(Decimal("10000.00"))
    assert fees["commission"] == Decimal("5.00")
    assert fees["transfer_fee"] == Decimal("0.10")
    assert fees["stamp_duty"] == Decimal("0.00")
    assert fees["total_fees"] == Decimal("5.10")
    assert fees["total_cost"] == Decimal("10005.10")


def test_buy_shares_respect_the_lot_and_the_fee_inclusive_budget() -> None:
    assert calculate_buy_shares(budget=Decimal("10000"), price=Decimal("33.45")) == 200
    assert calculate_buy_shares(budget=Decimal("3345"), price=Decimal("33.45")) == 0
    assert calculate_buy_shares(budget=Decimal("3345"), price=Decimal("33.40")) == 0
    assert calculate_buy_shares(budget=Decimal("3350"), price=Decimal("33.40")) == 100
    # 300 lots of 33.45 fit the raw budget but not after the CNY 5 minimum fee.
    assert calculate_buy_shares(budget=Decimal("10040"), price=Decimal("33.45")) == 200


def test_entry_fills_at_the_slipped_open_within_the_position_cap(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _path, intent, intent_ref = _entry_intent(workspace, "EMPTY")
    _path, eligibility, eligibility_ref = _entry_eligibility(workspace, intent_ref)
    outcome = _buy(workspace, intent, intent_ref, eligibility, eligibility_ref)

    assert outcome["outcome"] == "FILLED"
    assert outcome["order"]["side"] == "BUY"
    # 100.00 * 1.005 rounds up to the cent, capped by limit_up above.
    assert outcome["order"]["simulated_price"] == "100.5000"
    accounting = outcome["accounting"]
    # 14% of NAV is 140,000; 100.50 per share keeps cost inside the cap.
    assert accounting["shares_bought"] == 1300
    assert accounting["avg_cost_after"] == "100.5111"
    assert accounting["realized_pnl_delta"] == "0.0000"
    assert Decimal(accounting["cash_after"]) < Decimal("1000000.0000")


def test_entry_never_breaches_the_cash_floor(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _path, intent, intent_ref = _entry_intent(workspace, "EMPTY")
    _path, eligibility, eligibility_ref = _entry_eligibility(workspace, intent_ref)
    # Cash equals the NAV floor, so there is no budget at all.
    outcome = _buy(workspace, intent, intent_ref, eligibility, eligibility_ref, cash="50000.0000")
    assert outcome["outcome"] == "SKIPPED"
    assert outcome["pending"]["status"] == "PAPER_ENTRY_SKIPPED_NO_BUDGET"


def test_entry_skips_below_one_lot(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _path, intent, intent_ref = _entry_intent(workspace, "EMPTY")
    _path, intent, intent_ref = _entry_intent(workspace, "EMPTY", nav="10000.0000")
    _path, eligibility, eligibility_ref = _entry_eligibility(
        workspace, intent_ref, open_price="300.0000", limit_up="400.0000"
    )
    outcome = _buy(workspace, intent, intent_ref, eligibility, eligibility_ref, cash="10000.0000")
    assert outcome["outcome"] == "SKIPPED"
    assert outcome["pending"]["status"] == "PAPER_ENTRY_SKIPPED_BELOW_MINIMUM_LOT"


def test_entry_pends_at_a_limit_up_open(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _path, intent, intent_ref = _entry_intent(workspace, "EMPTY")
    _path, eligibility, eligibility_ref = _entry_eligibility(
        workspace, intent_ref, open_price="120.0000", limit_up="120.0000"
    )
    outcome = _buy(workspace, intent, intent_ref, eligibility, eligibility_ref)
    assert outcome["outcome"] == "PENDING"
    assert outcome["pending"]["blocker_codes"] == ["PAPER_BUY_PRICE_AT_LIMIT_UP"]


def test_entry_pends_on_suspension_and_corporate_action(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _path, intent, intent_ref = _entry_intent(workspace, "EMPTY")
    _path, eligibility, eligibility_ref = _entry_eligibility(workspace, intent_ref, suspended=True)
    outcome = _buy(workspace, intent, intent_ref, eligibility, eligibility_ref)
    assert outcome["pending"]["status"] == "PENDING_SUSPENDED"

    _path, eligibility, eligibility_ref = _entry_eligibility(
        workspace, intent_ref, corporate="PENDING"
    )
    outcome = _buy(workspace, intent, intent_ref, eligibility, eligibility_ref)
    assert outcome["pending"]["status"] == "PENDING_CORPORATE_ACTION"


def test_entry_rejects_a_position_that_does_not_match_expectation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _path, intent, intent_ref = _entry_intent(workspace, "EMPTY")
    _path, eligibility, eligibility_ref = _entry_eligibility(workspace, intent_ref)
    with pytest.raises(PaperError, match="PAPER_POSITION_MISMATCH"):
        _buy(
            workspace,
            intent,
            intent_ref,
            eligibility,
            eligibility_ref,
            position={
                "symbol": "688183.SH",
                "shares": 500,
                "settled_shares": 500,
                "avg_cost": "1.0",
            },
        )


def test_store_creates_the_row_with_an_unsettled_t_plus_one_lot(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = PaperStore(workspace)
    registered = store.register(_registration(workspace))
    pointer_sha = registered["pointer_sha256"]
    _path, intent, intent_ref = _entry_intent(workspace, pointer_sha)
    _path, eligibility, eligibility_ref = _entry_eligibility(workspace, intent_ref)
    outcome = _buy(workspace, intent, intent_ref, eligibility, eligibility_ref)

    result = store.commit(
        account_id="paper-alpha",
        expected_pointer_sha256=pointer_sha,
        intent=intent,
        intent_ref=intent_ref,
        eligibility=eligibility,
        eligibility_ref=eligibility_ref,
        outcome=outcome,
    )
    assert result["command_status"] == "FILLED"
    loaded = store.load_account("paper-alpha")
    row = next(item for item in loaded["ledger"] if item["symbol"] == "688183.SH")
    assert row["shares"] == outcome["accounting"]["shares_bought"]
    # Bought today, so nothing is sellable until the next session (T+1).
    assert row["settled_shares"] == 0
    lots = json.loads(row["acquisition_lots_json"])
    assert lots == [
        {"shares": row["shares"], "acquisition_date": "20261008", "settlement_date": "20261009"}
    ]
    assert row["avg_cost"] == Decimal(outcome["accounting"]["avg_cost_after"])


def test_settlement_promotes_lots_and_a_sell_consumes_them_fifo(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = PaperStore(workspace)
    registered = store.register(_registration(workspace))
    pointer_sha = registered["pointer_sha256"]
    _path, intent, intent_ref = _entry_intent(workspace, pointer_sha)
    _path, eligibility, eligibility_ref = _entry_eligibility(workspace, intent_ref)
    outcome = _buy(workspace, intent, intent_ref, eligibility, eligibility_ref)
    store.commit(
        account_id="paper-alpha",
        expected_pointer_sha256=pointer_sha,
        intent=intent,
        intent_ref=intent_ref,
        eligibility=eligibility,
        eligibility_ref=eligibility_ref,
        outcome=outcome,
    )

    # Next session: the seeded 200 shares are older and settle first, then the
    # 2026-10-08 lot. Selling consumes the oldest lot first.
    loaded = store.load_account("paper-alpha")
    pointer_sha = loaded["pointer_sha256"]
    position = next(item for item in loaded["ledger"] if item["symbol"] == "002916.SZ")
    economic = economic_action_key(
        account_id="paper-alpha",
        policy_id=contracts.POLICY_ID,
        signal_date="20261008",
        symbol="002916.SZ",
        action="REDUCE_50",
        shares=100,
    )
    sell_intent = seal_document(
        {
            "schema_version": "paper-risk-intent.v1",
            "source_intent_id": "intent-shennan-20261009-reduce50",
            "idempotency_key_sha256": economic,
            "economic_action_key_sha256": economic,
            "account_id": "paper-alpha",
            "strategy_id": "aggressive_tech_manufacturing",
            "signal_date": "20261008",
            "eligible_from_trade_date": "20261009",
            "symbol": "002916.SZ",
            "action": "REDUCE_50",
            "requested_ratio": "0.50",
            "requested_shares": 100,
            "reason_codes": ["PROFIT_GIVEBACK_GE_35"],
            "policy_ref": {
                "path": contracts.POLICY_RELATIVE_PATH,
                "sha256": contracts.POLICY_SHA256,
            },
            "expected_account_pointer_sha256": pointer_sha,
            "expected_position": {
                "shares": int(position["shares"]),
                "settled_shares": 200,
                "avg_cost": str(position["avg_cost"]),
            },
            "evidence_refs": [],
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
    )
    sell_intent_ref = {"path": "inputs/sell-intent.json", "sha256": "0" * 64}
    sell_intent_ref["sha256"] = hashlib.sha256(canonical_json_bytes(sell_intent)).hexdigest()
    _path, sell_eligibility, sell_eligibility_ref = _entry_eligibility(
        workspace,
        sell_intent_ref,
        symbol="002916.SZ",
        signal_date="20261008",
        trade_date="20261009",
    )
    sell_outcome = execute_sell(
        intent=sell_intent,
        intent_ref=sell_intent_ref,
        eligibility=sell_eligibility,
        eligibility_ref=sell_eligibility_ref,
        position=position,
        cash_before=Decimal(loaded["state"]["cash"]),
        evaluated_open_session_count=1,
    )
    assert sell_outcome["outcome"] == "FILLED"
    assert sell_outcome["accounting"]["shares_sold"] == 100
    store.commit(
        account_id="paper-alpha",
        expected_pointer_sha256=pointer_sha,
        intent=sell_intent,
        intent_ref=sell_intent_ref,
        eligibility=sell_eligibility,
        eligibility_ref=sell_eligibility_ref,
        outcome=sell_outcome,
    )
    reloaded = store.load_account("paper-alpha")
    after = next(item for item in reloaded["ledger"] if item["symbol"] == "002916.SZ")
    assert after["shares"] == 100
    assert after["settled_shares"] == 100
    assert json.loads(after["acquisition_lots_json"]) == [
        {"shares": 100, "acquisition_date": "20260810", "settlement_date": "20260811"}
    ]
