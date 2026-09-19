"""Bounded attempt/recovery records inside the existing daily maintenance lock."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

from .close_session_authority import CloseSessionAuthorityResult, replay_close_session_authority


def logical_task_claim(*, logical_key: str, slot: str, mode: str) -> dict[str, Any]:
    """Shared native claim bytes; budgets and installation identity are code-owned."""
    return {
        "schema_version": "cn-daily-logical-task.v1",
        "logical_key": logical_key,
        "mode": mode,
        "slot": slot,
        "installation": {
            "python": str(Path(sys.executable).absolute()),
            "module": str(Path(__file__).resolve()),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "close_request_budget": 2,
        "attempt_budget": 2,
    }


def _verify_finalized_attempt(
    *,
    workspace: Path,
    attempt: Path,
    claim_ref: dict,
    mode: str,
    read: Callable,
    ref: Callable,
    error: Callable,
) -> dict[str, Any] | None:
    """Shared native finalized-attempt validation; caller owns read safety/exclusion.

    No claim, attempt, lock, budget or provider is created by this function.
    The native coordinator and descriptor-bound recovery use the same validator.
    """
    final = attempt / "attempt.json"
    receipt = json.loads(read(final))
    if receipt.get("logical_claim_ref") != claim_ref:
        raise error("LOGICAL_TASK_RECEIPT_CLAIM_MISMATCH")
    if receipt.get("execution_disposition") == "NON_TRADING_DAY":
        from .requested_session import validate_closed_attempt
        from .daily_maintenance import _path_present

        veto = Path(claim_ref["path"]).parents[2] / "WRITE_VETO.json"
        if _path_present(veto):
            read(veto)
            raise error("WRITE_VETO_ACTIVE")
        if mode != "execute":
            raise error("REQUESTED_CLOSED_ATTEMPT_MODE_INVALID")
        day = Path(claim_ref["path"]).parent.name.split("-", 1)[0]
        projection = validate_closed_attempt(
            attempt=attempt, receipt=receipt, requested_trade_date=day, read=read
        )
        return {
            **receipt,
            "requested_session_result": projection,
            "attempt_receipt_ref": ref(final),
            "logical_replay": "VERIFIED_SAME_INPUT",
            "provider_calls_this_attempt": False,
        }
    if receipt.get("factor_input_readiness") == "NOT_APPLICABLE" or receipt.get(
        "execution_disposition"
    ) not in {None, "REQUESTED_SESSION_BLOCKED"}:
        raise error("REQUESTED_SESSION_ATTEMPT_DISPOSITION_INVALID")
    if receipt.get("factor_input_readiness") != "READY" or mode != "execute":
        return None
    from quant_investor.factors.production_rollover import validate_daily_maintenance_receipt

    checkpoint = attempt / "core-completion.json"
    selected = checkpoint if checkpoint.exists() else final
    validate_daily_maintenance_receipt(
        workspace_root=workspace,
        receipt_path=selected,
        expected_receipt_sha256=ref(selected)["sha256"],
    )
    return {
        **receipt,
        "attempt_receipt_ref": ref(final),
        "logical_replay": "VERIFIED_SAME_INPUT",
        "provider_calls_this_attempt": False,
    }


def _project_requested_replay(
    *,
    receipt: dict,
    requested_trade_date: str,
    read: Callable,
    error: Callable,
    expected_previous_trade_date=None,
) -> dict:
    """Add a current-call CLOSED projection without rewriting ordinary receipt fields."""
    from .requested_session import recorded_requested_session
    from .daily_maintenance import _path_present

    veto = Path(receipt["logical_claim_ref"]["path"]).parents[2] / "WRITE_VETO.json"
    if _path_present(veto):
        read(veto)
        raise error("WRITE_VETO_ACTIVE")
    attempt = Path(receipt["attempt_receipt_ref"]["path"]).parent
    projected = recorded_requested_session(
        attempt=attempt,
        receipt=receipt,
        requested_trade_date=requested_trade_date,
        read=read,
        expected_previous_trade_date=expected_previous_trade_date,
    )
    if projected["classification"] == "CONFIRMED_CLOSED":
        return {**receipt, "requested_session_result": projected}
    return receipt


class DailyOperationJournal:
    """A schedule-slot claim; random attempt directories cannot reset its budget."""

    def __init__(
        self,
        root: Path,
        workspace: Path,
        *,
        now: datetime,
        slot: str,
        mode: str,
        _historical_trade_date: str | None = None,
    ):
        from .daily_maintenance import (
            _child_directory,
            _write_once,
            _canonical_json_bytes,
            _read_owner_file,
            DailyMaintenanceError,
        )

        task_day = (
            now.strftime("%Y%m%d") if _historical_trade_date is None else _historical_trade_date
        )
        if _historical_trade_date is not None:
            from zoneinfo import ZoneInfo

            try:
                valid = (
                    type(task_day) is str
                    and datetime.strptime(task_day, "%Y%m%d").strftime("%Y%m%d") == task_day
                    and isinstance(now, datetime)
                    and now.tzinfo is not None
                    and now.utcoffset() is not None
                    and task_day < now.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
                    and mode == "execute"
                    and slot == "2020"
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise DailyMaintenanceError("HISTORICAL_TASK_IDENTITY_INVALID")
        self.write = _write_once
        self.encode = _canonical_json_bytes
        self.read = lambda p: _read_owner_file(p, code="OPERATION_JOURNAL_UNSAFE")
        self.error = DailyMaintenanceError
        self.root = root
        self.workspace = workspace
        self.mode = mode
        self.now = now
        self.slot = slot
        self.key = f"{task_day}-{slot}-{mode}"
        self.path = _child_directory(_child_directory(root, "logical_tasks"), self.key)
        expected = logical_task_claim(logical_key=self.key, slot=slot, mode=mode)
        self.identity = expected["installation"]
        claim = self.path / "claim.json"
        if claim.exists():
            if self.read(claim) != self.encode(expected):
                raise self.error("LOGICAL_TASK_INSTALL_OR_POLICY_DRIFT")
        else:
            self.write(claim, self.encode(expected))
        self.claim_ref = self.ref(claim)

    def ref(self, path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": hashlib.sha256(self.read(path)).hexdigest()}

    def attempts(self) -> list[Path]:
        rows = []
        for index in (1, 2):
            path = self.path / f"attempt-{index}.json"
            if path.exists():
                row = json.loads(self.read(path))
                attempt = Path(row["attempt_root"])
                attempt.relative_to(self.root / "attempts")
                if row["logical_key"] != self.key:
                    raise self.error("LOGICAL_TASK_CLAIM_MISMATCH")
                rows.append(attempt)
        return rows

    def recover_or_replay(self) -> dict[str, Any] | None:
        attempts = self.attempts()
        if not attempts:
            return None
        previous = attempts[-1]
        final = previous / "attempt.json"
        if final.exists():
            return _verify_finalized_attempt(
                workspace=self.workspace,
                attempt=previous,
                claim_ref=self.claim_ref,
                mode=self.mode,
                read=self.read,
                ref=self.ref,
                error=self.error,
            )
        # With the daily flock held, the predecessor process is no longer owner.
        completed, uncertain = [], []
        for stage in ("PIT", "MARKET", "HISTORY", "FUNDAMENTAL", "MACRO_RELEASE"):
            if (previous / f"stage-{stage}.json").exists():
                completed.append(self.ref(previous / f"stage-{stage}.json"))
            elif (previous / f"start-{stage}.json").exists():
                uncertain.append(stage)
        checkpoint = previous / "core-completion.json"
        recovery = {
            "state": "IN_DOUBT" if uncertain else "INTERRUPTED",
            "predecessor_started_ref": self.ref(previous / "started.json"),
            "completed_stage_refs": completed,
            "unknown_transaction_stages": uncertain,
            "core_completion_ref": self.ref(checkpoint) if checkpoint.exists() else None,
            "recovery_action": (
                "RECONCILE_COMPONENT_JOURNAL_BEFORE_WRITE"
                if uncertain
                else "BOUNDED_SUCCESSOR_FROM_EXACT_COMPONENT_REPLAY"
            ),
        }
        record = previous / "recovery.json"
        if not record.exists():
            self.write(record, self.encode(recovery))
        if uncertain:
            return {
                "schema_version": "cn-daily-maintenance-attempt.v1",
                "status": "IN_DOUBT",
                "maintenance_status": "IN_DOUBT",
                "mode": self.mode,
                "target_date": (
                    json.loads(self.read(checkpoint)).get("target_date")
                    if checkpoint.exists()
                    else None
                ),
                "blockers": ["INTERRUPTED_COMPONENT_WRITE_UNCONFIRMED"],
                "core_completion_ref": recovery["core_completion_ref"],
                "recovery_ref": self.ref(record),
                "logical_claim_ref": self.claim_ref,
                "provider_calls_this_attempt": False,
            }
        return None

    def bind(self, attempt: Path) -> None:
        count = len(self.attempts())
        if count >= 2:
            raise self.error("LOGICAL_TASK_ATTEMPT_BUDGET_EXHAUSTED")
        previous = self.attempts()
        row = {
            "logical_key": self.key,
            "attempt_root": str(attempt),
            "predecessor_started_ref": (
                self.ref(previous[-1] / "started.json") if previous else None
            ),
            "claim_ref": self.claim_ref,
        }
        self.write(self.path / f"attempt-{count + 1}.json", self.encode(row))

    def acquire(
        self, callback: Callable[..., CloseSessionAuthorityResult], *, now: datetime
    ) -> CloseSessionAuthorityResult:
        for attempt in reversed(self.attempts()[:-1]):
            path = attempt / "close-session-receipt.json"
            if path.exists():
                receipt = json.loads(self.read(path))
                raw = self.read(Path(receipt["raw_response_path"]))
                result = replay_close_session_authority(receipt, raw)
                return result
        for number in (1, 2):
            path = self.path / f"close-request-{number}.json"
            if not path.exists():
                self.write(
                    path,
                    self.encode(
                        {
                            "state": "REQUEST_STARTED",
                            "api_name": "trade_cal",
                            "sequence": number,
                            "logical_key": self.key,
                            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "claim_ref": self.claim_ref,
                        }
                    ),
                )
                return callback(now=now)
        raise self.error("LOGICAL_TASK_CLOSE_REQUEST_BUDGET_EXHAUSTED")
