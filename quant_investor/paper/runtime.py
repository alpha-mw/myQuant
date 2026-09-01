"""Public Paper Writer v1 status, preview, registration, run, and verify surfaces."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from quant_investor.contracts import canonical_json_bytes

from .contracts import (
    POLICY_RELATIVE_PATH,
    POLICY_SHA256,
    PaperError,
    validate_eligibility,
    validate_intent,
    validate_registration,
    writer_registration,
)
from .execution import execute_sell
from .store import PaperStore


def _read_exact(
    workspace: Path,
    path_value: str,
    expected_sha: str,
    *,
    code: str,
    owner_only: bool = True,
    canonical_required: bool = True,
) -> tuple[bytes, dict[str, Any], str]:
    if type(path_value) is not str or not path_value or path_value.startswith("/"):
        raise PaperError(code, "path must be workspace-relative")
    relative = Path(path_value)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise PaperError(code, "path is not canonical")
    try:
        path = (workspace / relative).resolve(strict=True)
        path.relative_to(workspace)
        before = path.lstat()
    except (OSError, ValueError) as exc:
        raise PaperError(code, "path unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o022
        or (owner_only and stat.S_IMODE(before.st_mode) != 0o600)
    ):
        raise PaperError(code, "file security differs")
    raw = path.read_bytes()
    middle = path.lstat()
    second = path.read_bytes()
    after = path.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (middle.st_dev, middle.st_ino, middle.st_size, middle.st_mtime_ns)
        or (middle.st_dev, middle.st_ino, middle.st_size, middle.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or raw != second
    ):
        raise PaperError(code, "file changed during read")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha:
        raise PaperError(code, "SHA differs")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PaperError(code, "JSON invalid") from exc
    if type(value) is not dict or (canonical_required and canonical_json_bytes(value) != raw):
        raise PaperError(code, "JSON is not canonical")
    return raw, value, observed


def _policy(workspace: Path) -> tuple[dict[str, Any], dict[str, str]]:
    raw, value, observed = _read_exact(
        workspace,
        POLICY_RELATIVE_PATH,
        POLICY_SHA256,
        code="PAPER_POLICY_SHA_MISMATCH",
        owner_only=False,
        canonical_required=False,
    )
    if (
        value.get("schema_version") != "owner-paper-risk-execution-policy.v1"
        or value.get("policy_id") != "owner-paper-risk-execution-policy-20260901-v1"
        or value.get("account_scope") != "ALL_REGISTERED_PAPER_ACCOUNTS"
        or value.get("automatic_paper_execution") is not True
        or value.get("action_scope") != "RISK_REDUCING_SELLS_ONLY"
        or value.get("real_trading_authority") is not False
    ):
        raise PaperError("PAPER_POLICY_INVALID", "owner policy fields differ")
    return value, {"path": POLICY_RELATIVE_PATH, "sha256": observed}


def _release_ready(
    *,
    workspace: Path,
    release_install_input_path: str,
    expected_release_install_input_sha256: str,
    release_repository_root: str,
) -> dict[str, Any]:
    raw, _value, _sha = _read_exact(
        workspace,
        release_install_input_path,
        expected_release_install_input_sha256,
        code="PAPER_WRITER_RELEASE_NOT_READY",
    )
    try:
        from quant_investor.system.release_install import verify_running_release_install_input

        verification = verify_running_release_install_input(
            raw,
            repository_root=release_repository_root,
        )
    except Exception as exc:
        raise PaperError("PAPER_WRITER_RELEASE_NOT_READY", "release replay failed") from exc
    if verification.get("state") != "PASS":
        raise PaperError("PAPER_WRITER_RELEASE_NOT_READY", "verification did not pass")
    return verification


def writer_status(*, workspace_root: str) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve(strict=True)
    policy, policy_ref = _policy(workspace)
    store = PaperStore(workspace)
    accounts = store.account_ids()
    blockers = [] if accounts else ["PAPER_ACCOUNT_NOT_REGISTERED"]
    return {
        "writer_id": writer_registration()["writer_id"],
        "writer_version": "1",
        "writer_status": "READY" if accounts else "PAPER_ACCOUNT_NOT_REGISTERED",
        "registration_status": "REGISTERED_IN_CODE",
        "policy_status": "READY",
        "policy_ref": policy_ref,
        "policy_writer_state": policy["writer_state"],
        "account_count": len(accounts),
        "account_ids": accounts,
        "write_capability": "ACCOUNT_REQUIRED" if not accounts else "INSTALLED_RELEASE_REQUIRED",
        "broker": False,
        "real_order": False,
        "actual_holdings_mutation": False,
        "blockers": blockers,
    }


def account_status(*, workspace_root: str, account_id: str) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve(strict=True)
    store = PaperStore(workspace)
    if account_id not in store.account_ids():
        return {
            "account_id": account_id,
            "account_status": "PAPER_ACCOUNT_NOT_REGISTERED",
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
            "blockers": ["PAPER_ACCOUNT_NOT_REGISTERED"],
        }
    loaded = store.load_account(account_id)
    return {
        "account_id": account_id,
        "account_status": "READY",
        "sequence": loaded["pointer"]["sequence"],
        "pointer_sha256": loaded["pointer_sha256"],
        "cash": loaded["state"]["cash"],
        "position_count": len(loaded["ledger"]),
        "pending_count": len(loaded["state"].get("pending_intents") or {}),
        "broker": False,
        "real_order": False,
        "actual_holdings_mutation": False,
        "blockers": [],
    }


def account_register(
    *,
    workspace_root: str,
    registration_path: str,
    expected_registration_sha256: str,
    allow_write: bool,
    release_install_input_path: str,
    expected_release_install_input_sha256: str,
    release_repository_root: str,
) -> dict[str, Any]:
    if allow_write is not True:
        raise PaperError("PAPER_WRITE_NOT_AUTHORIZED", "--allow-write required")
    workspace = Path(workspace_root).resolve(strict=True)
    _release_ready(
        workspace=workspace,
        release_install_input_path=release_install_input_path,
        expected_release_install_input_sha256=expected_release_install_input_sha256,
        release_repository_root=release_repository_root,
    )
    _raw, value, _sha = _read_exact(
        workspace,
        registration_path,
        expected_registration_sha256,
        code="PAPER_ACCOUNT_REGISTRATION_INVALID",
    )
    registration = validate_registration(value)
    _policy_value, policy_ref = _policy(workspace)
    if registration["policy_ref"] != policy_ref:
        raise PaperError("PAPER_POLICY_SHA_MISMATCH", "registration policy differs")
    return PaperStore(workspace).register(registration)


def risk_exit_preview(
    *,
    workspace_root: str,
    account_id: str,
    intent_path: str,
    expected_intent_sha256: str,
    eligibility_path: str,
    expected_eligibility_sha256: str,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve(strict=True)
    _policy_value, policy_ref = _policy(workspace)
    store = PaperStore(workspace)
    if account_id not in store.account_ids():
        return {
            "command_status": "PAPER_ACCOUNT_NOT_REGISTERED",
            "account_id": account_id,
            "write_set": [],
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
            "blockers": ["PAPER_ACCOUNT_NOT_REGISTERED"],
        }
    loaded = store.load_account(account_id)
    _intent_raw, intent_value, intent_sha = _read_exact(
        workspace, intent_path, expected_intent_sha256, code="PAPER_INTENT_INVALID"
    )
    _eligibility_raw, eligibility_value, eligibility_sha = _read_exact(
        workspace,
        eligibility_path,
        expected_eligibility_sha256,
        code="PAPER_ELIGIBILITY_INVALID",
    )
    intent = validate_intent(intent_value)
    eligibility = validate_eligibility(eligibility_value)
    if intent["account_id"] != account_id or intent["policy_ref"] != policy_ref:
        raise PaperError("PAPER_POLICY_SHA_MISMATCH", "intent policy/account differs")
    applied = (loaded["state"].get("applied_source_intents") or {}).get(intent["source_intent_id"])
    if isinstance(applied, Mapping):
        if applied.get("intent_sha256") != intent_sha:
            raise PaperError("PAPER_IDEMPOTENCY_CONFLICT", intent["source_intent_id"])
        return {
            "command_status": "NO_ACTION_ALREADY_APPLIED",
            "account_id": account_id,
            "expected_current_pointer_sha256": loaded["pointer_sha256"],
            "outcome": None,
            "write_set": [],
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
            "blockers": [],
        }
    existing_pending = (loaded["state"].get("pending_intents") or {}).get(
        intent["source_intent_id"]
    )
    if intent["expected_account_pointer_sha256"] != loaded["pointer_sha256"] and not (
        isinstance(existing_pending, Mapping)
        and existing_pending.get("intent_sha256") == intent_sha
    ):
        raise PaperError("PAPER_COMPARE_AND_SWAP_CONFLICT", "intent pointer differs")
    position = next((row for row in loaded["ledger"] if row["symbol"] == intent["symbol"]), None)
    if position is None:
        raise PaperError("PAPER_POSITION_MISMATCH", intent["symbol"])
    intent_ref = {"path": intent_path, "sha256": intent_sha}
    eligibility_ref = {"path": eligibility_path, "sha256": eligibility_sha}
    count = int((existing_pending or {}).get("evaluated_open_session_count", 0))
    if eligibility["evidence_status"] == "READY":
        count = max(count + 1, int(eligibility["open_session_ordinal"]))
    outcome = execute_sell(
        intent=intent,
        intent_ref=intent_ref,
        eligibility=eligibility,
        eligibility_ref=eligibility_ref,
        position=position,
        cash_before=Decimal(loaded["state"]["cash"]),
        evaluated_open_session_count=count,
    )
    return {
        "command_status": "PREVIEW_COMPLETE",
        "account_id": account_id,
        "expected_current_pointer_sha256": loaded["pointer_sha256"],
        "outcome": outcome,
        "write_set": [],
        "broker": False,
        "real_order": False,
        "actual_holdings_mutation": False,
        "blockers": [],
    }


def risk_exit_run(
    *,
    workspace_root: str,
    account_id: str,
    intent_path: str,
    expected_intent_sha256: str,
    eligibility_path: str,
    expected_eligibility_sha256: str,
    expected_current_pointer_sha256: str,
    allow_write: bool,
    release_install_input_path: str,
    expected_release_install_input_sha256: str,
    release_repository_root: str,
) -> dict[str, Any]:
    if allow_write is not True:
        raise PaperError("PAPER_WRITE_NOT_AUTHORIZED", "--allow-write required")
    workspace = Path(workspace_root).resolve(strict=True)
    _release_ready(
        workspace=workspace,
        release_install_input_path=release_install_input_path,
        expected_release_install_input_sha256=expected_release_install_input_sha256,
        release_repository_root=release_repository_root,
    )
    preview = risk_exit_preview(
        workspace_root=workspace_root,
        account_id=account_id,
        intent_path=intent_path,
        expected_intent_sha256=expected_intent_sha256,
        eligibility_path=eligibility_path,
        expected_eligibility_sha256=expected_eligibility_sha256,
    )
    if preview["command_status"] == "PAPER_ACCOUNT_NOT_REGISTERED":
        return preview
    if preview["command_status"] == "NO_ACTION_ALREADY_APPLIED":
        return preview
    if preview["expected_current_pointer_sha256"] != expected_current_pointer_sha256:
        raise PaperError("PAPER_COMPARE_AND_SWAP_CONFLICT", "caller pointer differs")
    _intent_raw, intent_value, intent_sha = _read_exact(
        workspace, intent_path, expected_intent_sha256, code="PAPER_INTENT_INVALID"
    )
    _eligibility_raw, eligibility_value, eligibility_sha = _read_exact(
        workspace,
        eligibility_path,
        expected_eligibility_sha256,
        code="PAPER_ELIGIBILITY_INVALID",
    )
    return PaperStore(workspace).commit(
        account_id=account_id,
        expected_pointer_sha256=expected_current_pointer_sha256,
        intent=validate_intent(intent_value),
        intent_ref={"path": intent_path, "sha256": intent_sha},
        eligibility=validate_eligibility(eligibility_value),
        eligibility_ref={"path": eligibility_path, "sha256": eligibility_sha},
        outcome=preview["outcome"],
    )


def verify_account(*, workspace_root: str, account_id: str) -> dict[str, Any]:
    status = account_status(workspace_root=workspace_root, account_id=account_id)
    if status["account_status"] != "READY":
        return {"command_status": "PAPER_ACCOUNT_NOT_REGISTERED", **status}
    loaded = PaperStore(workspace_root).load_account(account_id)
    return {
        "command_status": "VERIFIED",
        "account_id": account_id,
        "sequence": loaded["pointer"]["sequence"],
        "pointer_sha256": loaded["pointer_sha256"],
        "position_count": len(loaded["ledger"]),
        "broker": False,
        "real_order": False,
        "actual_holdings_mutation": False,
        "blockers": [],
    }


__all__ = [
    "account_register",
    "account_status",
    "risk_exit_preview",
    "risk_exit_run",
    "verify_account",
    "writer_status",
]
