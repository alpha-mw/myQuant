"""Read the existing 20:20 maintenance claim without creating or renewing its budget."""

from pathlib import PurePosixPath
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.market.maintenance_journal import logical_task_claim
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, validate_ref
from .daily_journal import _validate_day


def inspect_handoff_slot_claim(*, workspace, claim_ref, handoff, started):
    from .maintenance_handoff_contract import handoff_task_date

    return inspect_daily_slot_claim(
        workspace=workspace, claim_ref=claim_ref, run_date=handoff_task_date(handoff, started)
    )


def inspect_daily_slot_claim(*, workspace: str, claim_ref: dict, run_date: str) -> dict:
    """Call after installed-runtime verification; this only checks the existing claim.

    The native maintenance lock and its attempt/close-request records still own
    actual budget enforcement. Reading a valid claim is not a fresh allowance.
    """
    _validate_day(run_date)
    validate_ref(claim_ref)
    key = run_date + "-2020-execute"
    path = PurePosixPath(claim_ref["path"])
    if len(path.parts) < 4 or path.parts[-3:] != ("logical_tasks", key, "claim.json"):
        raise ContractError("DAILY_SLOT_CLAIM_PATH_INVALID")
    storage = SecureSystemStorage(workspace)
    raw = storage.read_workspace_file_bytes(claim_ref["path"], maximum_bytes=1024 * 1024)
    if raw.byte_sha256 != claim_ref["sha256"]:
        raise ContractError("DAILY_SLOT_CLAIM_SHA_MISMATCH")
    value = parse_canonical_json_bytes(raw.data)
    expected = logical_task_claim(logical_key=key, slot="2020", mode="execute")
    if raw.data != canonical_json_bytes(expected):
        raise ContractError("DAILY_SLOT_CLAIM_INSTALL_OR_POLICY_DRIFT")
    if storage.read_workspace_file_bytes(claim_ref["path"], maximum_bytes=1024 * 1024) != raw:
        raise ContractError("DAILY_SLOT_CLAIM_CHANGED_DURING_READ")
    return {
        "claim_ref": dict(claim_ref),
        "logical_key": key,
        "run_date": run_date,
        "slot": "2020",
        "mode": "execute",
        "maintenance_run_root": str(path.parent.parent.parent),
        "attempt_budget": value["attempt_budget"],
        "close_request_budget": value["close_request_budget"],
        "new_budget_granted": False,
        "execution_authorized": False,
    }
