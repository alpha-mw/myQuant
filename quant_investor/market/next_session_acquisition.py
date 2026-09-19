"""Code-owned installed future Calendar acquisition; no caller transport or clock."""

from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path

from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.journal_revisions import selected_binding
from quant_investor.system.release_install import verify_running_release_install_input
from .tushare_calendar_authority import capture_trusted_provider_calendar_evidence
from .next_session_calendar import HORIZON_NATURAL_DAYS, inspect_next_session_capture


def capture_next_session_calendar(
    *,
    workspace: str,
    eod_trade_date: str,
    release_install_input_raw: bytes,
    expected_release_install_input_sha256: str,
    release_repository_root: str,
) -> dict:
    """Acquire through the fixed native operator; provenance publication is separate.

    This function may perform live native provider calls when explicitly invoked.
    It cannot be used by Morning, and returns no Morning eligibility or proof ref.
    """
    validate_ref({"path": "release-input.json", "sha256": expected_release_install_input_sha256})
    if (
        type(release_install_input_raw) is not bytes
        or hashlib.sha256(release_install_input_raw).hexdigest()
        != expected_release_install_input_sha256
    ):
        raise ContractError("NEXT_SESSION_RELEASE_INPUT_SHA_MISMATCH")
    journal = DailyJournal(workspace, eod_trade_date)
    verified = verify_running_release_install_input(
        release_install_input_raw, repository_root=release_repository_root
    )
    if verified["state"] != "PASS":
        raise ContractError("NEXT_SESSION_INSTALLED_RUNTIME_REQUIRED")
    cutoff = datetime.strptime(eod_trade_date, "%Y%m%d").date() + timedelta(
        days=HORIZON_NATURAL_DAYS
    )
    root_name = "future-" + eod_trade_date + "-" + expected_release_install_input_sha256[:16]
    parent_relative = journal.root / "calendar-future/captures"
    parent = Path(workspace).resolve(strict=True) / parent_relative
    with journal.locked():
        if journal.storage.read(str(journal.root / "completion.v1.json")) is not None:
            raise ContractError("NEXT_SESSION_EOD_ALREADY_COMPLETED")
        if selected_binding(journal.storage, str(journal.root / "nodes/calendar"))[0] is not None:
            raise ContractError("NEXT_SESSION_CALENDAR_ALREADY_SELECTED")
        fd, _, _ = journal.storage._parent(str(parent_relative / "unused"), create=True)
        os.close(fd)
        # Native preflight rejects existing success/failure roots. Do not adopt an
        # unknown capture as LIVE merely because it is at a deterministic path.
        captured = capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name=root_name,
            cutoff_date=cutoff.isoformat(),
            release_install_input_raw=release_install_input_raw,
            expected_release_install_input_sha256=expected_release_install_input_sha256,
            release_repository_root=release_repository_root,
        )
        replay = inspect_next_session_capture(
            workspace=workspace,
            eod_trade_date=eod_trade_date,
            execution=captured["capture_execution"],
            execution_ref=captured["capture_execution_file_ref"],
            success=captured["capture_success"],
            success_ref=captured["capture_success_file_ref"],
        )
        return {"capture": captured, "native_projection": replay, "consumer_admission": False}
