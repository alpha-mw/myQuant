"""Fail-closed orchestration for same-day post-close CN data maintenance.

This module owns scheduling truth, local evidence, locking, and stage ordering.
It deliberately does not decode market providers or serialize canonical pointers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, time, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Final
from zoneinfo import ZoneInfo

from .close_session_authority import (
    CloseSessionAuthorityError,
    CloseSessionAuthorityResult,
    acquire_close_session_authority,
)
from .tushare_transport import TushareHttpsError

TIMEZONE: Final = "Asia/Shanghai"
ATTEMPT_SLOTS: Final = ("1620", "1720", "1820", "2020")
FINAL_SLOT: Final = "2020"
STAGES: Final = ("PIT", "MARKET", "HISTORY", "FUNDAMENTAL", "MACRO_RELEASE")
STAGE_STATUSES: Final = frozenset({"READY", "NO_ACTION", "RETRY_PENDING", "BLOCKED"})
TERMINAL_FAILURES: Final = frozenset({"BLOCKED", "SAME_DAY_SLA_MISSED", "WRITE_VETO_ACTIVE"})
_SLOT_STARTS: Final = (
    (time(16, 20), "1620"),
    (time(17, 20), "1720"),
    (time(18, 20), "1820"),
    (time(20, 20), "2020"),
)
_PROTECTED_SURFACES: Final = (
    "FACTOR_REGISTRY",
    "MAINLINE_ACTIVE_POINTER",
    "DASHBOARD",
    "PAPER_LEDGER",
    "HOLDINGS",
    "BROKER",
    "ORDERS",
    "TRADES",
)


class DailyMaintenanceError(RuntimeError):
    """One controlled coordinator error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class MaintenanceContext:
    """Immutable, pointer-opaque input supplied to one component adapter."""

    workspace_root: Path
    run_root: Path
    attempt_root: Path
    target_date: str
    attempt_slot: str
    mode: str
    close_session_receipt: Mapping[str, Any]
    close_session_receipt_path: Path
    close_session_receipt_sha256: str
    prior_stage_results: tuple[Mapping[str, Any], ...] = ()


StageCallback = Callable[[MaintenanceContext], Mapping[str, Any]]
StatusCallback = Callable[[MaintenanceContext], Mapping[str, Any]]


@dataclass(frozen=True)
class MaintenanceComponents:
    """Injectable adapters; provider and pointer semantics stay in their owners."""

    pit: StageCallback | None = None
    market: StageCallback | None = None
    history: StageCallback | None = None
    fundamental: StageCallback | None = None
    macro_release: StageCallback | None = None
    system_status: StatusCallback | None = None


class _RunLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> "_RunLock":
        flags = os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        created = False
        try:
            fd = os.open(self._path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise DailyMaintenanceError("RUN_LOCK_UNAVAILABLE") from exc
            try:
                fd = os.open(self._path, flags)
            except OSError as open_exc:
                raise DailyMaintenanceError("RUN_LOCK_UNAVAILABLE") from open_exc
        else:
            created = True
        try:
            if created:
                os.fchmod(fd, 0o600)
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise DailyMaintenanceError("RUN_LOCK_UNSAFE")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise DailyMaintenanceError("ALREADY_RUNNING") from None
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, *_args: object) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DailyMaintenanceError("ATTEMPT_EVIDENCE_NOT_CANONICAL") from exc


def _write_once(path: Path, raw: bytes) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise DailyMaintenanceError("ATTEMPT_EVIDENCE_WRITE_FAILED") from exc
    try:
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    except BaseException:
        try:
            os.close(fd)
        finally:
            raise
    else:
        os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def _read_owner_file(path: Path, *, code: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise DailyMaintenanceError(code) from exc
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise DailyMaintenanceError(code)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _path_present(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DailyMaintenanceError("EVIDENCE_PATH_UNREADABLE") from exc
    return True


def _owner_only_directory(path: Path, *, create: bool) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise DailyMaintenanceError("RUN_ROOT_NOT_ABSOLUTE")
    if create:
        try:
            candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise DailyMaintenanceError("RUN_ROOT_UNAVAILABLE") from exc
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise DailyMaintenanceError("RUN_ROOT_UNAVAILABLE") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise DailyMaintenanceError("RUN_ROOT_NOT_OWNER_ONLY")
    return candidate


def _child_directory(parent: Path, name: str) -> Path:
    path = parent / name
    try:
        path.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise DailyMaintenanceError("ATTEMPT_DIRECTORY_FAILED") from exc
    metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise DailyMaintenanceError("ATTEMPT_DIRECTORY_UNSAFE")
    return path


def _attempt_root(run_root: Path, *, now: datetime, slot: str) -> Path:
    attempts = _child_directory(run_root, "attempts")
    name = (
        f"{now.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{slot}-{secrets.token_hex(8)}"
    )
    path = attempts / name
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise DailyMaintenanceError("ATTEMPT_DIRECTORY_FAILED") from exc
    return path


def resolve_attempt_slot(*, now: datetime, requested: str) -> str:
    """Resolve one exact local slot bucket without guessing before 16:20."""

    if requested in ATTEMPT_SLOTS:
        return requested
    if requested != "auto":
        raise DailyMaintenanceError("ATTEMPT_SLOT_INVALID")
    if now.tzinfo is None or now.utcoffset() is None:
        raise DailyMaintenanceError("ATTEMPT_TIME_INVALID")
    local_time = now.astimezone(ZoneInfo(TIMEZONE)).time().replace(tzinfo=None)
    selected = ""
    for start, slot in _SLOT_STARTS:
        if local_time >= start:
            selected = slot
    if not selected:
        raise DailyMaintenanceError("OUTSIDE_ATTEMPT_WINDOW")
    return selected


def _validate_component_result(stage: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DailyMaintenanceError("COMPONENT_RESULT_INVALID")
    result = dict(value)
    status = result.get("status")
    if status not in STAGE_STATUSES or type(result.get("write_performed")) is not bool:
        raise DailyMaintenanceError("COMPONENT_RESULT_INVALID")
    evidence = result.get("evidence", {})
    if not isinstance(evidence, Mapping):
        raise DailyMaintenanceError("COMPONENT_RESULT_INVALID")
    normalized = {
        "stage": stage,
        "status": status,
        "write_performed": result["write_performed"],
        "blockers": sorted(
            set(result.get("blockers", []))
            if isinstance(result.get("blockers", []), list)
            and all(type(item) is str for item in result.get("blockers", []))
            else []
        ),
        "evidence": dict(evidence),
    }
    _canonical_json_bytes(normalized)
    return normalized


def _missing_component(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "BLOCKED",
        "write_performed": False,
        "blockers": [f"{stage}_COMPONENT_NOT_REGISTERED"],
        "evidence": {},
    }


def _run_component(
    *,
    stage: str,
    callback: StageCallback | None,
    context: MaintenanceContext,
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if callback is None:
        result = _missing_component(stage)
    else:
        try:
            stage_context = replace(
                context,
                prior_stage_results=tuple(dict(item) for item in prior_results),
            )
            result = _validate_component_result(stage, callback(stage_context))
        except DailyMaintenanceError:
            result = {
                "stage": stage,
                "status": "BLOCKED",
                "write_performed": False,
                "blockers": [f"{stage}_COMPONENT_RESULT_INVALID"],
                "evidence": {},
            }
        except Exception:
            result = {
                "stage": stage,
                "status": "BLOCKED",
                "write_performed": False,
                "blockers": [f"{stage}_COMPONENT_EXCEPTION"],
                "evidence": {},
            }
    if context.mode == "shadow" and result["write_performed"]:
        return {
            "stage": stage,
            "status": "BLOCKED",
            "write_performed": True,
            "blockers": ["SHADOW_WRITE_DETECTED"],
            "evidence": result["evidence"],
        }
    return result


def _fundamental_health(context: MaintenanceContext) -> Mapping[str, Any]:
    """Perform only the registered binding-aware Fundamental readback."""

    from .fundamental_generation import load_fundamental_binding

    try:
        binding = load_fundamental_binding(context.workspace_root / "data/parquet/cn")
    except Exception:
        return {
            "status": "BLOCKED",
            "write_performed": False,
            "blockers": ["FUNDAMENTAL_BINDING_READBACK_FAILED"],
            "evidence": {},
        }
    mixed = binding.get("mixed") is True
    ready = binding.get("binding_aware_research_ready") is True
    successor_states_match = (
        binding.get("legacy_direct_reader_provenance") == "limited"
        and binding.get("homogeneous_history_ready") is False
    )
    if mixed and not successor_states_match:
        ready = False
    return {
        "status": "READY" if ready else "BLOCKED",
        "write_performed": False,
        "blockers": [] if ready else ["FUNDAMENTAL_BINDING_NOT_READY"],
        "evidence": {
            "schema_version": binding.get("schema_version"),
            "generation_id": binding.get("generation_id"),
            "binding_sha256": binding.get("binding_sha256"),
            "target_cutoff": binding.get("target_cutoff"),
            "mixed": mixed,
            "binding_aware_research_ready": ready,
            "homogeneous_history_ready": binding.get("homogeneous_history_ready"),
            "legacy_direct_reader_provenance": binding.get("legacy_direct_reader_provenance"),
        },
    }


def _system_usability(context: MaintenanceContext, callback: StatusCallback | None) -> bool | str:
    try:
        if callback is None:
            from quant_investor.cli.unified import system_status

            observed = system_status(workspace_root=str(context.workspace_root))
        else:
            observed = dict(callback(context))
    except Exception:
        return "UNCONFIRMED"
    if type(observed.get("usable_for_investment_research")) is bool:
        return bool(observed["usable_for_investment_research"])
    capabilities = observed.get("capabilities")
    investment = capabilities.get("investment") if isinstance(capabilities, Mapping) else None
    if investment in {"READY", "ACTIVE", "AVAILABLE"}:
        return True
    if investment in {"BLOCKED", "SUSPENDED", "UNINITIALIZED", "UNAVAILABLE"}:
        return False
    return "UNCONFIRMED"


def _overall_status(*, stage_results: list[dict[str, Any]], slot: str, mode: str) -> str:
    statuses = {result["status"] for result in stage_results}
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "RETRY_PENDING" in statuses:
        return "SAME_DAY_SLA_MISSED" if slot == FINAL_SLOT else "RETRY_PENDING"
    if statuses == {"NO_ACTION"}:
        return "NO_ACTION"
    return "SHADOW_COMPLETE" if mode == "shadow" else "COMPLETE"


def _write_veto(
    run_root: Path, payload: Mapping[str, Any], *, filename: str = "WRITE_VETO.json"
) -> tuple[str, str]:
    path = run_root / filename
    raw = _canonical_json_bytes(payload)
    try:
        sha = _write_once(path, raw)
    except DailyMaintenanceError:
        try:
            existing = _read_owner_file(path, code="WRITE_VETO_UNSAFE")
        except DailyMaintenanceError:
            raise
        else:
            return str(path), hashlib.sha256(existing).hexdigest()
    return str(path), sha


def _seal_attempt(
    *, attempt_root: Path, payload: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    state_path = attempt_root / "state.json"
    receipt_path = attempt_root / "attempt.json"
    state_sha = _write_once(state_path, _canonical_json_bytes(state))
    final_payload = dict(payload)
    final_payload["state_ref"] = {
        "path": str(state_path),
        "sha256": state_sha,
    }
    receipt_sha = _write_once(receipt_path, _canonical_json_bytes(final_payload))
    final_payload["attempt_receipt_ref"] = {
        "path": str(receipt_path),
        "sha256": receipt_sha,
    }
    return final_payload


def run_cn_daily_maintenance(
    *,
    workspace_root: str | Path,
    run_root: str | Path,
    mode: str,
    attempt_slot: str = "auto",
    components: MaintenanceComponents | None = None,
    now: datetime | None = None,
    close_authority: Callable[..., CloseSessionAuthorityResult] = (acquire_close_session_authority),
) -> dict[str, Any]:
    """Run one locked orchestration attempt without owning component internals."""

    if mode not in {"shadow", "execute"}:
        raise DailyMaintenanceError("MAINTENANCE_MODE_INVALID")
    observed_now = now or datetime.now(tz=ZoneInfo(TIMEZONE))
    if observed_now.tzinfo is None or observed_now.utcoffset() is None:
        raise DailyMaintenanceError("ATTEMPT_TIME_INVALID")
    local_now = observed_now.astimezone(ZoneInfo(TIMEZONE))
    slot = resolve_attempt_slot(now=local_now, requested=attempt_slot)
    root = _owner_only_directory(Path(run_root), create=True)
    if components is None:
        from .daily_components import build_default_components

        selected_components = build_default_components(workspace_root=workspace_root)
    else:
        selected_components = components
    try:
        lock = _RunLock(root / ".daily-maintenance.lock")
        lock.__enter__()
    except DailyMaintenanceError as exc:
        if exc.code != "ALREADY_RUNNING":
            raise
        status = "SAME_DAY_SLA_MISSED" if slot == FINAL_SLOT else "ALREADY_RUNNING"
        return {
            "schema_version": "cn-daily-maintenance-attempt.v1",
            "status": status,
            "maintenance_status": status,
            "same_day_status": status,
            "fundamental_integrity_status": "UNCONFIRMED",
            "fundamental_refresh_status": "HEALTH_ONLY",
            "mode": mode,
            "attempt_slot": slot,
            "target_date": None,
            "canonical_unchanged": True,
            "usable_for_investment_research": "UNCONFIRMED",
            "blockers": ["ALREADY_RUNNING"],
        }
    try:
        attempt = _attempt_root(root, now=local_now, slot=slot)
        active_veto = root / "WRITE_VETO.json"
        if mode == "execute" and _path_present(active_veto):
            veto_raw = _read_owner_file(active_veto, code="WRITE_VETO_UNSAFE")
            payload = {
                "schema_version": "cn-daily-maintenance-attempt.v1",
                "status": "WRITE_VETO_ACTIVE",
                "maintenance_status": "WRITE_VETO_ACTIVE",
                "same_day_status": "WRITE_VETO_ACTIVE",
                "fundamental_integrity_status": "UNCONFIRMED",
                "fundamental_refresh_status": "HEALTH_ONLY",
                "mode": mode,
                "attempt_slot": slot,
                "target_date": None,
                "canonical_unchanged": True,
                "usable_for_investment_research": "UNCONFIRMED",
                "stage_results": [],
                "blockers": ["WRITE_VETO_ACTIVE"],
                "write_veto_ref": {
                    "path": str(active_veto),
                    "sha256": hashlib.sha256(veto_raw).hexdigest(),
                },
                "protected_surfaces": list(_PROTECTED_SURFACES),
            }
            return _seal_attempt(attempt_root=attempt, payload=payload, state=payload)
        try:
            close_result = close_authority(now=local_now)
            raw_response = bytes(close_result.raw_response_bytes)
            raw_sha = hashlib.sha256(raw_response).hexdigest()
            close_receipt = dict(close_result.receipt)
            if close_receipt.get("raw_response_sha256") != raw_sha:
                raise CloseSessionAuthorityError("CLOSE_AUTHORITY_RAW_SHA_MISMATCH")
            target_date = close_receipt.get("target_trade_date")
            if type(target_date) is not str or len(target_date) != 8 or not target_date.isdigit():
                raise CloseSessionAuthorityError("CLOSE_AUTHORITY_TARGET_INVALID")
        except (CloseSessionAuthorityError, TushareHttpsError) as exc:
            code = getattr(exc, "code", "CLOSE_AUTHORITY_FAILED")
            retryable = code in {
                "TUSHARE_API_ERROR",
                "TUSHARE_HTTP_STATUS_ERROR",
                "TUSHARE_TRANSPORT_ERROR",
                "CLOSE_SESSION_NOT_AVAILABLE",
                "CLOSE_CALENDAR_DATE_COVERAGE_INCOMPLETE",
                "CLOSE_SESSION_TARGET_NOT_TODAY",
                "CLOSE_CALENDAR_EMPTY",
            }
            status = (
                "SAME_DAY_SLA_MISSED"
                if retryable and slot == FINAL_SLOT
                else "RETRY_PENDING" if retryable else "BLOCKED"
            )
            payload = {
                "schema_version": "cn-daily-maintenance-attempt.v1",
                "status": status,
                "maintenance_status": status,
                "same_day_status": status,
                "fundamental_integrity_status": "UNCONFIRMED",
                "fundamental_refresh_status": "HEALTH_ONLY",
                "mode": mode,
                "attempt_slot": slot,
                "target_date": None,
                "canonical_unchanged": True,
                "usable_for_investment_research": "UNCONFIRMED",
                "stage_results": [],
                "blockers": [code],
                "protected_surfaces": list(_PROTECTED_SURFACES),
            }
            if mode == "execute" and status == "BLOCKED":
                veto_path, veto_sha = _write_veto(
                    root,
                    {
                        "schema_version": "cn-daily-maintenance-write-veto.v1",
                        "created_at": local_now.astimezone(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "attempt_slot": slot,
                        "blockers": [code],
                    },
                )
                payload["write_veto_ref"] = {"path": veto_path, "sha256": veto_sha}
            return _seal_attempt(attempt_root=attempt, payload=payload, state=payload)
        except Exception:
            payload = {
                "schema_version": "cn-daily-maintenance-attempt.v1",
                "status": "BLOCKED",
                "maintenance_status": "BLOCKED",
                "same_day_status": "BLOCKED",
                "fundamental_integrity_status": "UNCONFIRMED",
                "fundamental_refresh_status": "HEALTH_ONLY",
                "mode": mode,
                "attempt_slot": slot,
                "target_date": None,
                "canonical_unchanged": True,
                "usable_for_investment_research": "UNCONFIRMED",
                "stage_results": [],
                "blockers": ["CLOSE_AUTHORITY_EXCEPTION"],
                "protected_surfaces": list(_PROTECTED_SURFACES),
            }
            if mode == "execute":
                veto_path, veto_sha = _write_veto(
                    root,
                    {
                        "schema_version": "cn-daily-maintenance-write-veto.v1",
                        "created_at": local_now.astimezone(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "attempt_slot": slot,
                        "blockers": ["CLOSE_AUTHORITY_EXCEPTION"],
                    },
                )
                payload["write_veto_ref"] = {"path": veto_path, "sha256": veto_sha}
            return _seal_attempt(attempt_root=attempt, payload=payload, state=payload)
        raw_path = attempt / "close-session.raw.json"
        _write_once(raw_path, raw_response)
        close_receipt["attempt_slot"] = slot
        close_receipt["raw_response_path"] = str(raw_path)
        close_path = attempt / "close-session-receipt.json"
        close_sha = _write_once(close_path, _canonical_json_bytes(close_receipt))
        context = MaintenanceContext(
            workspace_root=Path(workspace_root).expanduser().resolve(),
            run_root=root,
            attempt_root=attempt,
            target_date=target_date,
            attempt_slot=slot,
            mode=mode,
            close_session_receipt=close_receipt,
            close_session_receipt_path=close_path,
            close_session_receipt_sha256=close_sha,
        )
        core_callbacks: tuple[tuple[str, StageCallback | None], ...] = (
            ("PIT", selected_components.pit),
            ("MARKET", selected_components.market),
            ("HISTORY", selected_components.history),
        )
        stage_results: list[dict[str, Any]] = []
        core_halted = False
        for stage, callback in core_callbacks:
            if core_halted:
                result = {
                    "stage": stage,
                    "status": "NO_ACTION",
                    "write_performed": False,
                    "blockers": ["UPSTREAM_STAGE_NOT_READY"],
                    "evidence": {"skipped": True},
                }
            else:
                result = _run_component(
                    stage=stage,
                    callback=callback,
                    context=context,
                    prior_results=stage_results,
                )
            stage_results.append(result)
            core_halted = core_halted or result["status"] in {"BLOCKED", "RETRY_PENDING"}
        core_results = list(stage_results)
        if core_halted:
            for stage in ("FUNDAMENTAL", "MACRO_RELEASE"):
                stage_results.append(
                    {
                        "stage": stage,
                        "status": "NO_ACTION",
                        "write_performed": False,
                        "blockers": ["UPSTREAM_STAGE_NOT_READY"],
                        "evidence": {"skipped": True},
                    }
                )
        else:
            fundamental_result = _run_component(
                stage="FUNDAMENTAL",
                callback=selected_components.fundamental or _fundamental_health,
                context=context,
                prior_results=core_results,
            )
            macro_veto = root / "MACRO_WRITE_VETO.json"
            if mode == "execute" and _path_present(macro_veto):
                veto_raw = _read_owner_file(macro_veto, code="MACRO_WRITE_VETO_UNSAFE")
                macro_result = {
                    "stage": "MACRO_RELEASE",
                    "status": "BLOCKED",
                    "write_performed": False,
                    "blockers": ["MACRO_WRITE_VETO_ACTIVE"],
                    "evidence": {
                        "macro_write_veto_ref": {
                            "path": str(macro_veto),
                            "sha256": hashlib.sha256(veto_raw).hexdigest(),
                        }
                    },
                }
            else:
                macro_result = _run_component(
                    stage="MACRO_RELEASE",
                    callback=selected_components.macro_release,
                    context=context,
                    prior_results=core_results,
                )
            stage_results.extend((fundamental_result, macro_result))
        fundamental_result = next(item for item in stage_results if item["stage"] == "FUNDAMENTAL")
        same_day_results = [item for item in stage_results if item["stage"] != "FUNDAMENTAL"]
        factor_core_results = [
            item for item in stage_results if item["stage"] in {"PIT", "MARKET", "HISTORY"}
        ]
        core_ready = (
            all(
                item["status"] in {"READY", "NO_ACTION"}
                and not item["blockers"]
                and item["evidence"].get("skipped") is not True
                for item in factor_core_results
            )
            and len(factor_core_results) == 3
        )
        core_hard_block = any(item["status"] == "BLOCKED" for item in factor_core_results)
        factor_input_readiness = "READY" if mode == "execute" and core_ready else "BLOCKED"
        factor_input_shadow_readiness = "READY" if mode == "shadow" and core_ready else "BLOCKED"
        core_blockers = sorted(
            {blocker for item in factor_core_results for blocker in item["blockers"]}
        )
        same_day_status = _overall_status(stage_results=same_day_results, slot=slot, mode=mode)
        if fundamental_result["evidence"].get("skipped") is True:
            fundamental_integrity_status = "UNCONFIRMED"
        elif fundamental_result["status"] in {"READY", "NO_ACTION"}:
            fundamental_integrity_status = "READY"
        elif fundamental_result["status"] == "BLOCKED":
            fundamental_integrity_status = "BLOCKED"
        else:
            fundamental_integrity_status = "UNCONFIRMED"
        macro_result = next(item for item in stage_results if item["stage"] == "MACRO_RELEASE")
        macro_status = macro_result["status"]
        raw_macro_blockers = macro_result.get("blockers")
        macro_blockers = (
            [str(value) for value in raw_macro_blockers]
            if isinstance(raw_macro_blockers, list)
            else []
        )
        if core_ready and macro_status == "BLOCKED":
            maintenance_status = "PARTIAL"
        elif same_day_status in {
            "BLOCKED",
            "RETRY_PENDING",
            "SAME_DAY_SLA_MISSED",
        }:
            maintenance_status = same_day_status
        elif fundamental_integrity_status == "READY":
            maintenance_status = same_day_status
        else:
            maintenance_status = "BLOCKED"
        blockers = sorted({blocker for result in stage_results for blocker in result["blockers"]})
        canonical_write_count = sum(1 for result in stage_results if result["write_performed"])
        payload = {
            "schema_version": "cn-daily-maintenance-attempt.v1",
            "status": maintenance_status,
            "maintenance_status": maintenance_status,
            "same_day_status": same_day_status,
            "factor_input_readiness": factor_input_readiness,
            "factor_input_shadow_readiness": factor_input_shadow_readiness,
            "factor_input_change": "UNKNOWN",
            "core_blockers": core_blockers,
            "macro_status": macro_status,
            "macro_blockers": macro_blockers,
            "macro_used_by_factor": False,
            "fundamental_used_by_factor": False,
            "factor_rollover_eligible": factor_input_readiness == "READY",
            "fundamental_integrity_status": fundamental_integrity_status,
            "fundamental_refresh_status": "HEALTH_ONLY",
            "mode": mode,
            "attempt_slot": slot,
            "target_date": context.target_date,
            "canonical_unchanged": canonical_write_count == 0,
            "canonical_write_count": canonical_write_count,
            "usable_for_investment_research": _system_usability(
                context, selected_components.system_status
            ),
            "close_session_receipt_ref": {
                "path": str(close_path),
                "sha256": close_sha,
            },
            "stage_results": stage_results,
            "blockers": blockers,
            "protected_surfaces": list(_PROTECTED_SURFACES),
        }
        if mode == "execute" and core_hard_block:
            veto_path, veto_sha = _write_veto(
                root,
                {
                    "schema_version": "cn-daily-maintenance-write-veto.v1",
                    "created_at": local_now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "attempt_slot": slot,
                    "target_date": context.target_date,
                    "blockers": blockers,
                    "attempt_root": str(attempt),
                },
                filename="WRITE_VETO.json",
            )
            payload["write_veto_ref"] = {"path": veto_path, "sha256": veto_sha}
        elif mode == "execute" and macro_status == "BLOCKED":
            veto_path, veto_sha = _write_veto(
                root,
                {
                    "schema_version": "cn-daily-maintenance-macro-write-veto.v1",
                    "created_at": local_now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "attempt_slot": slot,
                    "target_date": context.target_date,
                    "blockers": macro_blockers,
                    "attempt_root": str(attempt),
                },
                filename="MACRO_WRITE_VETO.json",
            )
            payload["macro_write_veto_ref"] = {"path": veto_path, "sha256": veto_sha}
        state = {
            "schema_version": "cn-daily-maintenance-state.v1",
            "status": maintenance_status,
            "maintenance_status": maintenance_status,
            "same_day_status": same_day_status,
            "factor_input_readiness": factor_input_readiness,
            "factor_input_shadow_readiness": factor_input_shadow_readiness,
            "factor_input_change": "UNKNOWN",
            "core_blockers": core_blockers,
            "macro_status": macro_status,
            "macro_blockers": macro_blockers,
            "macro_used_by_factor": False,
            "fundamental_used_by_factor": False,
            "factor_rollover_eligible": factor_input_readiness == "READY",
            "fundamental_integrity_status": fundamental_integrity_status,
            "fundamental_refresh_status": "HEALTH_ONLY",
            "mode": mode,
            "attempt_slot": slot,
            "target_date": context.target_date,
            "stage_states": {result["stage"]: result["status"] for result in stage_results},
            "blockers": blockers,
        }
        return _seal_attempt(attempt_root=attempt, payload=payload, state=state)
    finally:
        lock.__exit__(None, None, None)


def clear_cn_daily_write_veto(
    *,
    run_root: str | Path,
    expected_veto_sha256: str,
    reason: str,
    lane: str = "global",
) -> dict[str, Any]:
    """Archive one exact veto under the same lock; never delete its evidence."""

    if (
        len(expected_veto_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_veto_sha256)
        or not reason
        or reason.strip() != reason
        or len(reason) > 240
        or any(ord(character) < 0x20 for character in reason)
        or lane not in {"global", "macro"}
    ):
        raise DailyMaintenanceError("CLEAR_WRITE_VETO_ARGUMENTS_INVALID")
    root = _owner_only_directory(Path(run_root), create=False)
    with _RunLock(root / ".daily-maintenance.lock"):
        veto = root / ("WRITE_VETO.json" if lane == "global" else "MACRO_WRITE_VETO.json")
        if not _path_present(veto):
            return {
                "schema_version": "cn-daily-maintenance-veto-clear.v1",
                "status": "NO_ACTION",
                "cleared": False,
                "lane": lane,
            }
        raw = _read_owner_file(veto, code="WRITE_VETO_UNSAFE")
        observed_sha = hashlib.sha256(raw).hexdigest()
        if observed_sha != expected_veto_sha256:
            raise DailyMaintenanceError("WRITE_VETO_SHA_MISMATCH")
        archive = _child_directory(root, "veto_archive")
        archived_path = archive / f"{observed_sha}.json"
        if _path_present(archived_path):
            archived_raw = _read_owner_file(archived_path, code="WRITE_VETO_ARCHIVE_CONFLICT")
            if hashlib.sha256(archived_raw).hexdigest() != observed_sha:
                raise DailyMaintenanceError("WRITE_VETO_ARCHIVE_CONFLICT")
            veto.unlink()
        else:
            os.replace(veto, archived_path)
        cleared_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        receipt = {
            "schema_version": "cn-daily-maintenance-veto-clear.v1",
            "status": "CLEARED",
            "cleared": True,
            "cleared_at": cleared_at,
            "reason": reason,
            "lane": lane,
            "archived_veto_ref": {
                "path": str(archived_path),
                "sha256": observed_sha,
            },
        }
        receipt_path = archive / (
            f"{observed_sha}.{cleared_at.replace(':', '').replace('-', '')}."
            f"{secrets.token_hex(4)}.clear.json"
        )
        receipt_sha = _write_once(receipt_path, _canonical_json_bytes(receipt))
        receipt["clear_receipt_ref"] = {
            "path": str(receipt_path),
            "sha256": receipt_sha,
        }
        return receipt


def cli_exit_required(payload: Mapping[str, Any]) -> bool:
    return payload.get("status") in TERMINAL_FAILURES


__all__ = [
    "ATTEMPT_SLOTS",
    "DailyMaintenanceError",
    "MaintenanceComponents",
    "MaintenanceContext",
    "clear_cn_daily_write_veto",
    "cli_exit_required",
    "resolve_attempt_slot",
    "run_cn_daily_maintenance",
]
