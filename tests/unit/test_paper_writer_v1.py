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
    calculate_fees,
    calculate_sell_shares,
    economic_action_key,
    execute_sell,
)
from quant_investor.paper.runtime import account_status, risk_exit_preview, writer_status
from quant_investor.paper.store import PaperStore

ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: dict | bytes, *, mode: int = 0o600) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(mode)
    return (
        path.relative_to(
            path.parents[len(path.parts) - len(path.anchor.split("/")) - 1]
        ).as_posix(),
        hashlib.sha256(raw).hexdigest(),
    )


def _relative(workspace: Path, path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(workspace).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
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
            "initial_cash": "100000.0000",
            "initial_positions": [
                {
                    "symbol": "002916.SZ",
                    "name": "深南电路",
                    "shares": 200,
                    "settled_shares": 200,
                    "avg_cost": "10.0000",
                    "cost_basis": "2000.0000",
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


def _intent(workspace: Path, pointer_sha: str) -> tuple[Path, dict, dict[str, str]]:
    policy_id = "owner-paper-risk-execution-policy-20260901-v1"
    economic = economic_action_key(
        account_id="paper-alpha",
        policy_id=policy_id,
        signal_date="20260901",
        symbol="002916.SZ",
        action="REDUCE_50",
        shares=100,
    )
    value = seal_document(
        {
            "schema_version": "paper-risk-intent.v1",
            "source_intent_id": "intent-shennan-20260901-reduce50",
            "idempotency_key_sha256": economic,
            "economic_action_key_sha256": economic,
            "account_id": "paper-alpha",
            "strategy_id": "aggressive_tech_manufacturing",
            "signal_date": "20260901",
            "eligible_from_trade_date": "20260902",
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
            "expected_position": {"shares": 200, "settled_shares": 200, "avg_cost": "10.0000"},
            "evidence_refs": [],
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
    )
    path = workspace / "inputs/intent.json"
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o600)
    return path, value, _relative(workspace, path)


def _eligibility(
    workspace: Path,
    intent_ref: dict[str, str],
    *,
    ready: bool,
) -> tuple[Path, dict, dict[str, str]]:
    refs = {}
    if ready:
        for name in ("calendar", "bar", "limit", "suspension", "corporate"):
            path = workspace / f"inputs/{name}.json"
            path.write_bytes(
                canonical_json_bytes({"name": name, "symbol": "002916.SZ", "date": "20260902"})
            )
            path.chmod(0o600)
            refs[name] = _relative(workspace, path)
    value = seal_document(
        {
            "schema_version": "paper-input-eligibility.v1",
            "account_id": "paper-alpha",
            "source_intent_ref": intent_ref,
            "symbol": "002916.SZ",
            "signal_date": "20260901",
            "eligible_trade_date": "20260902",
            "evaluated_trade_date": "20260902" if ready else "20260901",
            "open_price": "12.0000" if ready else None,
            "previous_close": "11.0000" if ready else None,
            "limit_up": "13.0000" if ready else None,
            "limit_down": "9.0000" if ready else None,
            "suspended": False,
            "corporate_action_state": "CLEAR",
            "open_session_ordinal": 1 if ready else 0,
            "expiry_session_ordinal": 3,
            "calendar_ref": refs.get("calendar"),
            "raw_bar_ref": refs.get("bar"),
            "price_limit_ref": refs.get("limit"),
            "suspension_ref": refs.get("suspension"),
            "corporate_action_ref": refs.get("corporate"),
            "evidence_status": "READY" if ready else "NOT_YET_AVAILABLE",
        }
    )
    path = workspace / (
        "inputs/eligibility-ready.json" if ready else "inputs/eligibility-pending.json"
    )
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o600)
    return path, value, _relative(workspace, path)


def test_writer_status_is_read_only_when_no_account(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    before = sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*"))
    result = writer_status(workspace_root=str(workspace))
    assert result["writer_status"] == "PAPER_ACCOUNT_NOT_REGISTERED"
    assert result["broker"] is False
    assert not (workspace / contracts.PAPER_ROOT).exists()
    after = sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*"))
    assert after == before


def test_fee_and_lot_golden_vectors() -> None:
    assert calculate_sell_shares(action="REDUCE_25", settled_shares=200) == 0
    assert calculate_sell_shares(action="REDUCE_50", settled_shares=200) == 100
    assert calculate_sell_shares(action="EXIT_100", settled_shares=55) == 55
    low = calculate_fees(Decimal("1000.00"))
    assert low == {
        "commission": Decimal("5.00"),
        "transfer_fee": Decimal("0.01"),
        "stamp_duty": Decimal("0.50"),
        "total_fees": Decimal("5.51"),
        "net_cash_proceeds": Decimal("994.49"),
    }
    high = calculate_fees(Decimal("100000.00"))
    assert high["commission"] == Decimal("10.00")
    assert high["transfer_fee"] == Decimal("1.00")
    assert high["stamp_duty"] == Decimal("50.00")


def test_temp_account_pending_fill_and_idempotent_replay(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = PaperStore(workspace)
    registered = store.register(_registration(workspace))
    assert registered["sequence"] == 0
    pointer = registered["pointer_sha256"]
    intent_path, intent, intent_ref = _intent(workspace, pointer)
    pending_path, pending_eligibility, pending_ref = _eligibility(
        workspace, intent_ref, ready=False
    )
    preview = risk_exit_preview(
        workspace_root=str(workspace),
        account_id="paper-alpha",
        intent_path=intent_path.relative_to(workspace).as_posix(),
        expected_intent_sha256=intent_ref["sha256"],
        eligibility_path=pending_path.relative_to(workspace).as_posix(),
        expected_eligibility_sha256=pending_ref["sha256"],
    )
    assert preview["outcome"]["outcome"] == "PENDING"
    pending_commit = store.commit(
        account_id="paper-alpha",
        expected_pointer_sha256=pointer,
        intent=intent,
        intent_ref=intent_ref,
        eligibility=pending_eligibility,
        eligibility_ref=pending_ref,
        outcome=preview["outcome"],
    )
    assert pending_commit["command_status"] == "PENDING"

    loaded = store.load_account("paper-alpha")
    ready_path, ready_eligibility, ready_ref = _eligibility(workspace, intent_ref, ready=True)
    ready_preview = risk_exit_preview(
        workspace_root=str(workspace),
        account_id="paper-alpha",
        intent_path=intent_path.relative_to(workspace).as_posix(),
        expected_intent_sha256=intent_ref["sha256"],
        eligibility_path=ready_path.relative_to(workspace).as_posix(),
        expected_eligibility_sha256=ready_ref["sha256"],
    )
    assert ready_preview["outcome"]["outcome"] == "FILLED"
    filled = store.commit(
        account_id="paper-alpha",
        expected_pointer_sha256=loaded["pointer_sha256"],
        intent=intent,
        intent_ref=intent_ref,
        eligibility=ready_eligibility,
        eligibility_ref=ready_ref,
        outcome=ready_preview["outcome"],
    )
    assert filled["command_status"] == "FILLED"
    final = store.load_account("paper-alpha")
    assert final["ledger"][0]["shares"] == 100
    assert final["state"]["cash"] == "101134.4200"
    replay = store.commit(
        account_id="paper-alpha",
        expected_pointer_sha256=final["pointer_sha256"],
        intent=intent,
        intent_ref=intent_ref,
        eligibility=ready_eligibility,
        eligibility_ref=ready_ref,
        outcome=ready_preview["outcome"],
    )
    assert replay["command_status"] == "NO_ACTION_ALREADY_APPLIED"


def test_pointer_conflict_and_fault_preserve_old_pointer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = PaperStore(workspace)
    registered = store.register(_registration(workspace))
    intent_path, intent, intent_ref = _intent(workspace, registered["pointer_sha256"])
    eligibility_path, eligibility, eligibility_ref = _eligibility(
        workspace, intent_ref, ready=False
    )
    preview = risk_exit_preview(
        workspace_root=str(workspace),
        account_id="paper-alpha",
        intent_path=intent_path.relative_to(workspace).as_posix(),
        expected_intent_sha256=intent_ref["sha256"],
        eligibility_path=eligibility_path.relative_to(workspace).as_posix(),
        expected_eligibility_sha256=eligibility_ref["sha256"],
    )
    with pytest.raises(PaperError, match="PAPER_COMPARE_AND_SWAP_CONFLICT"):
        store.commit(
            account_id="paper-alpha",
            expected_pointer_sha256="0" * 64,
            intent=intent,
            intent_ref=intent_ref,
            eligibility=eligibility,
            eligibility_ref=eligibility_ref,
            outcome=preview["outcome"],
        )
    assert store.load_account("paper-alpha")["pointer_sha256"] == registered["pointer_sha256"]

    def fail(point: str) -> None:
        if point == "BEFORE_RECORD_RENAME":
            raise RuntimeError("fault")

    fault_store = PaperStore(workspace, fault_hook=fail)
    with pytest.raises(RuntimeError, match="fault"):
        fault_store.commit(
            account_id="paper-alpha",
            expected_pointer_sha256=registered["pointer_sha256"],
            intent=intent,
            intent_ref=intent_ref,
            eligibility=eligibility,
            eligibility_ref=eligibility_ref,
            outcome=preview["outcome"],
        )
    assert store.load_account("paper-alpha")["pointer_sha256"] == registered["pointer_sha256"]


def test_account_status_missing_is_stable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = account_status(workspace_root=str(workspace), account_id="paper-alpha")
    assert result["account_status"] == "PAPER_ACCOUNT_NOT_REGISTERED"
    assert result["actual_holdings_mutation"] is False


def test_limit_suspension_and_third_session_expiry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = PaperStore(workspace)
    registered = store.register(_registration(workspace))
    _intent_path, intent, intent_ref = _intent(workspace, registered["pointer_sha256"])
    _eligibility_path, eligibility, eligibility_ref = _eligibility(
        workspace, intent_ref, ready=True
    )
    position = store.load_account("paper-alpha")["ledger"][0]
    eligibility["suspended"] = True
    expired = execute_sell(
        intent=intent,
        intent_ref=intent_ref,
        eligibility=eligibility,
        eligibility_ref=eligibility_ref,
        position=position,
        cash_before=Decimal("100000.0000"),
        evaluated_open_session_count=3,
    )
    assert expired["outcome"] == "EXPIRED"
    assert expired["pending"]["status"] == "EXPIRED_REEVALUATION_REQUIRED"

    eligibility["suspended"] = False
    eligibility["open_price"] = eligibility["limit_down"]
    blocked = execute_sell(
        intent=intent,
        intent_ref=intent_ref,
        eligibility=eligibility,
        eligibility_ref=eligibility_ref,
        position=position,
        cash_before=Decimal("100000.0000"),
        evaluated_open_session_count=1,
    )
    assert blocked["outcome"] == "PENDING"
    assert blocked["pending"]["status"] == "PENDING_LIMIT_BLOCKED"


def test_symlinked_live_paper_root_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    root = workspace / "results/paper"
    root.mkdir(parents=True, mode=0o700)
    (root / "accounts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PaperError, match="PAPER_STORAGE_SECURITY"):
        writer_status(workspace_root=str(workspace))
