"""Exact maintenance and canonical-input gates for Factor production rollover."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Final, Mapping

from .governance.errors import FactorGovernanceError

_MAX_RECEIPT_BYTES: Final = 16 * 1024 * 1024
_REQUIRED_STAGES: Final = ("PIT", "MARKET", "HISTORY", "FUNDAMENTAL", "MACRO_RELEASE")


def _read_owner_file(path: Path, *, root: Path, label: str) -> tuple[bytes, str]:
    try:
        resolved_root = root.resolve(strict=True)
        lexical = Path(os.path.abspath(path))
        cursor = resolved_root
        for part in lexical.relative_to(resolved_root).parts:
            cursor = cursor / part
            if stat.S_ISLNK(cursor.lstat().st_mode):
                raise FactorGovernanceError(f"{label} contains a symbolic link")
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        before = resolved.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_RECEIPT_BYTES
        ):
            raise FactorGovernanceError(f"{label} is not an owner-controlled regular file")
        first = resolved.read_bytes()
        middle = resolved.lstat()
        second = resolved.read_bytes()
        after = resolved.lstat()
    except (OSError, ValueError) as exc:
        raise FactorGovernanceError(f"{label} is unavailable") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )
    if (
        identity(before) != identity(middle)
        or identity(middle) != identity(after)
        or first != second
    ):
        raise FactorGovernanceError(f"{label} changed during stable read")
    return first, hashlib.sha256(first).hexdigest()


def _mapping(raw: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FactorGovernanceError(f"{label} is not strict JSON") from exc
    if type(value) is not dict:
        raise FactorGovernanceError(f"{label} is not a JSON object")
    return value


def _validate_ref(
    value: Any, *, attempt_root: Path, label: str
) -> tuple[dict[str, str], dict[str, Any]]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise FactorGovernanceError(f"{label} fields differ")
    path = Path(str(value["path"]))
    raw, observed = _read_owner_file(path, root=attempt_root, label=label)
    if observed != value["sha256"]:
        raise FactorGovernanceError(f"{label} SHA differs")
    return {"path": str(path), "sha256": observed}, _mapping(raw, label=label)


def validate_daily_maintenance_receipt(
    *,
    workspace_root: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate one exact execute-mode authoritative CN maintenance receipt."""

    workspace = Path(workspace_root).resolve(strict=True)
    registered_root = workspace / "data/private/cn_daily_maintenance/attempts"
    path = Path(receipt_path).absolute()
    try:
        attempt_root = path.parent.resolve(strict=True)
        attempt_root.relative_to(registered_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise FactorGovernanceError(
            "maintenance receipt is outside the registered attempt root"
        ) from exc
    checkpoint = path.name == "core-completion.json"
    if path.name not in {"attempt.json", "core-completion.json"}:
        raise FactorGovernanceError("maintenance receipt filename differs")
    raw, observed_sha = _read_owner_file(path, root=attempt_root, label="maintenance receipt")
    if observed_sha != expected_receipt_sha256:
        raise FactorGovernanceError("maintenance receipt SHA differs")
    receipt = _mapping(raw, label="maintenance receipt")
    from quant_investor.market.historical_session import CORE_SCHEMA, CORE_FIELDS

    historical = checkpoint and receipt.get("schema_version") == CORE_SCHEMA
    if historical:
        if set(receipt) != CORE_FIELDS or receipt.get("historical_session_ref") is None:
            raise FactorGovernanceError("historical core fields differ")
    elif "historical_session_ref" in receipt:
        raise FactorGovernanceError("historical evidence requires core v2")
    common = receipt.get("status")
    partial_factor_ready = (
        common == "PARTIAL"
        and receipt.get("maintenance_status") == "PARTIAL"
        and receipt.get("same_day_status") == "BLOCKED"
        and receipt.get("factor_input_readiness") == "READY"
        and receipt.get("core_blockers") == []
        and receipt.get("macro_status") == "BLOCKED"
        and type(receipt.get("macro_blockers")) is list
        and bool(receipt.get("macro_blockers"))
        and receipt.get("macro_used_by_factor") is False
        and receipt.get("fundamental_used_by_factor") is False
        and receipt.get("factor_rollover_eligible") is True
    )
    if checkpoint:
        if (
            receipt.get("schema_version")
            != (CORE_SCHEMA if historical else "cn-daily-maintenance-core.v1")
            or receipt.get("producer") != "quant_investor.market.daily_maintenance"
            or receipt.get("scope") != "FACTOR_INPUTS_ONLY"
            or common != "CORE_COMPLETE"
            or receipt.get("maintenance_status") != "IN_PROGRESS"
            or receipt.get("mode") != "execute"
            or receipt.get("blockers") != []
            or receipt.get("other_authority") != "NONE"
        ):
            raise FactorGovernanceError("core checkpoint is not an execute input closure")
        _, started = _validate_ref(
            receipt.get("started_ref"), attempt_root=attempt_root, label="attempt start"
        )
        if started.get("mode") != "execute" or started.get("state") != "STARTED":
            raise FactorGovernanceError("core checkpoint start binding differs")
        refs = receipt.get("stage_refs")
        if not isinstance(refs, list) or len(refs) != 3:
            raise FactorGovernanceError("core checkpoint stage refs differ")
        for item, row in zip(refs, receipt.get("stage_results", [])):
            _, record = _validate_ref(item, attempt_root=attempt_root, label="core stage")
            if record.get("result") != row:
                raise FactorGovernanceError("core checkpoint stage evidence differs")
    else:
        if (
            receipt.get("schema_version") != "cn-daily-maintenance-attempt.v1"
            or receipt.get("mode") != "execute"
            or (common not in {"COMPLETE", "NO_ACTION"} and not partial_factor_ready)
            or receipt.get("maintenance_status") != common
            or (
                not partial_factor_ready
                and (receipt.get("same_day_status") != common or receipt.get("blockers") != [])
            )
            or "write_veto_ref" in receipt
            or "core_write_veto_ref" in receipt
        ):
            raise FactorGovernanceError("maintenance receipt is not an authoritative success")
    target = receipt.get("target_date")
    if type(target) is not str or len(target) != 8 or not target.isdigit():
        raise FactorGovernanceError("maintenance receipt target date differs")
    if (workspace / "data/private/cn_daily_maintenance/WRITE_VETO.json").exists():
        raise FactorGovernanceError("maintenance write veto is active")
    state_ref = None
    if not checkpoint:
        state_ref, state = _validate_ref(
            receipt.get("state_ref"), attempt_root=attempt_root, label="maintenance state"
        )
        if (
            state.get("status") != common
            or state.get("target_date") != target
            or state.get("mode") != "execute"
            or (not partial_factor_ready and state.get("blockers") != [])
        ):
            raise FactorGovernanceError("maintenance state differs from receipt")
    close_ref, close = _validate_ref(
        receipt.get("close_session_receipt_ref"),
        attempt_root=attempt_root,
        label="close-session receipt",
    )
    if not historical and close.get("target_trade_date") != target:
        raise FactorGovernanceError("close-session target differs")
    raw_path = close.get("raw_response_path")
    raw_sha = close.get("raw_response_sha256")
    if type(raw_path) is not str or type(raw_sha) is not str:
        raise FactorGovernanceError("close-session raw response reference is absent")
    _raw, observed_raw_sha = _read_owner_file(
        Path(raw_path), root=attempt_root, label="close-session raw response"
    )
    if observed_raw_sha != raw_sha:
        raise FactorGovernanceError("close-session raw response SHA differs")
    from quant_investor.market.close_session_authority import replay_close_session_authority

    replay_close_session_authority(close, _raw)
    historical_evidence = None
    if historical:
        from datetime import datetime, timezone
        from quant_investor.market.historical_session import (
            read_historical_core,
            validate_historical_core_timing,
        )
        from quant_investor.market.close_session_authority import CloseSessionAuthorityError
        from quant_investor.market.tushare_transport import TushareHttpsError
        from quant_investor.operations.daily_contract import utc_stamp

        observed_now = datetime.now(timezone.utc)
        try:
            evidence = read_historical_core(
                attempt_root=attempt_root,
                proof_ref=receipt["historical_session_ref"],
                close_ref=receipt["close_session_receipt_ref"],
                started_ref=receipt["started_ref"],
                target=target,
                read=lambda p, label: _read_owner_file(p, root=attempt_root, label=label)[0],
            )
            historical_evidence = evidence["historical_session"]
            validate_historical_core_timing(
                observed_at=historical_evidence["observed_at"],
                started_at=evidence["started_at"],
                sealed_at=receipt["sealed_at"],
            )
            if utc_stamp(receipt["sealed_at"]) > observed_now:
                raise ValueError("historical_core_seal_in_future")
        except (OSError, ValueError, CloseSessionAuthorityError, TushareHttpsError) as exc:
            raise FactorGovernanceError("historical core evidence differs") from exc
    rows = receipt.get("stage_results")
    required_stages = _REQUIRED_STAGES[:3] if checkpoint else _REQUIRED_STAGES
    if type(rows) is not list or [row.get("stage") for row in rows] != list(required_stages):
        raise FactorGovernanceError("maintenance stage set or order differs")
    stage_states: dict[str, str] = {}
    stage_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if type(row) is not dict or type(row.get("blockers")) is not list:
            raise FactorGovernanceError("maintenance stage row differs")
        stage = row["stage"]
        status = row.get("status")
        if type(status) is not str:
            raise FactorGovernanceError("maintenance stage status differs")
        if stage in {"PIT", "MARKET", "HISTORY"}:
            if status not in {"READY", "NO_ACTION"} or row["blockers"]:
                raise FactorGovernanceError(f"maintenance core stage {stage} is not closed")
        elif not partial_factor_ready and (status not in {"READY", "NO_ACTION"} or row["blockers"]):
            raise FactorGovernanceError(f"maintenance stage {stage} is not closed")
        stage_states[stage] = status
        stage_rows[stage] = row
    closure = validate_factor_core_closure(
        workspace=workspace, target=target, stage_rows=stage_rows
    )
    result = {
        "receipt_path": str(path),
        "receipt_sha256": observed_sha,
        "state_ref": state_ref,
        "close_session_receipt_ref": close_ref,
        "target_date": target,
        "status": common,
        "stage_states": stage_states,
        "upstream_maintenance_status": "IN_PROGRESS" if checkpoint else common,
        "macro_status": receipt.get("macro_status", "NOT_STARTED"),
        "macro_blockers": list(receipt.get("macro_blockers") or []),
        "macro_used_by_factor": False,
        "core_closure": closure,
    }
    if historical:
        result["historical_session_ref"] = receipt["historical_session_ref"]
        result["historical_session"] = historical_evidence
        result["sealed_at"] = receipt["sealed_at"]
    return result


def validate_factor_core_closure(
    *, workspace: Path, target: str, stage_rows: Mapping[str, Any]
) -> dict[str, str]:
    """Shared exact PIT/Market/History verification for final and early receipts."""

    pit_evidence = stage_rows["PIT"].get("evidence")
    market_evidence = stage_rows["MARKET"].get("evidence")
    history_evidence = stage_rows["HISTORY"].get("evidence")
    if not all(
        isinstance(value, Mapping)
        for value in (
            pit_evidence,
            market_evidence,
            history_evidence,
        )
    ):
        raise FactorGovernanceError("maintenance core evidence is absent")
    pit_binding = dict(pit_evidence.get("pit_binding") or {})  # type: ignore[union-attr]
    required_pit = {
        "generation_id",
        "generation_manifest_path",
        "generation_manifest_sha256",
        "canonical_path",
        "canonical_sha256",
        "discovery_pointer_path",
        "discovery_pointer_sha256",
    }
    if not required_pit <= set(pit_binding):
        raise FactorGovernanceError("maintenance PIT binding is incomplete")
    for path_key, sha_key, label in (
        ("generation_manifest_path", "generation_manifest_sha256", "maintenance PIT manifest"),
        ("canonical_path", "canonical_sha256", "maintenance PIT membership"),
        ("discovery_pointer_path", "discovery_pointer_sha256", "maintenance PIT pointer"),
    ):
        _raw, observed = _read_owner_file(
            Path(str(pit_binding[path_key])), root=workspace, label=label
        )
        if observed != pit_binding[sha_key]:
            raise FactorGovernanceError(f"{label} SHA differs")
    market_pointer_path = Path(str(market_evidence.get("pointer_path") or ""))  # type: ignore[union-attr]
    market_pointer_sha = str(market_evidence.get("pointer_sha256") or "")  # type: ignore[union-attr]
    market_manifest_path = Path(
        str(market_evidence.get("snapshot_manifest_path") or "")  # type: ignore[union-attr]
    )
    market_manifest_sha = str(
        market_evidence.get("snapshot_manifest_sha256") or ""  # type: ignore[union-attr]
    )
    market_pointer_raw, observed_market_pointer_sha = _read_owner_file(
        market_pointer_path, root=workspace, label="maintenance Market pointer"
    )
    _market_manifest_raw, observed_market_manifest_sha = _read_owner_file(
        market_manifest_path, root=workspace, label="maintenance Market manifest"
    )
    if (
        observed_market_pointer_sha != market_pointer_sha
        or observed_market_manifest_sha != market_manifest_sha
    ):
        raise FactorGovernanceError("maintenance Market binding SHA differs")
    market_pointer = _mapping(market_pointer_raw, label="maintenance Market pointer")
    coverage = market_pointer.get("coverage")
    if (
        market_pointer.get("latest_complete_trade_date") != target
        or market_pointer.get("manifest_path") != str(market_manifest_path)
        or not isinstance(coverage, Mapping)
        or coverage.get("pit_generation_id") != pit_binding["generation_id"]
        or coverage.get("pit_generation_manifest_sha256")
        != pit_binding["generation_manifest_sha256"]
        or coverage.get("pit_membership_sha256") != pit_binding["canonical_sha256"]
    ):
        raise FactorGovernanceError("maintenance Market/PIT binding differs")
    history_path = Path(str(history_evidence.get("audit_path") or ""))  # type: ignore[union-attr]
    history_sha = str(history_evidence.get("audit_sha256") or "")  # type: ignore[union-attr]
    history_raw, observed_history_sha = _read_owner_file(
        history_path, root=workspace, label="maintenance History audit"
    )
    history = _mapping(history_raw, label="maintenance History audit")
    history_projection = history.get("canonical")
    if (
        observed_history_sha != history_sha
        or history_evidence.get("history_audit_status") != "passed"  # type: ignore[union-attr]
        or history.get("history_audit_status") != "passed"
        or history.get("target_trade_date") != target
        or history.get("effective_trade_date") != target
        or history.get("audited_trade_dates_count") != 100
        or not isinstance(history_projection, Mapping)
        or history_projection.get("latest_sha256") != market_pointer_sha
    ):
        raise FactorGovernanceError("maintenance History closure differs")
    return {
        "pit_generation_id": str(pit_binding["generation_id"]),
        "pit_pointer_path": str(pit_binding["discovery_pointer_path"]),
        "pit_pointer_sha256": str(pit_binding["discovery_pointer_sha256"]),
        "pit_manifest_path": str(pit_binding["generation_manifest_path"]),
        "pit_manifest_sha256": str(pit_binding["generation_manifest_sha256"]),
        "pit_membership_path": str(pit_binding["canonical_path"]),
        "pit_membership_sha256": str(pit_binding["canonical_sha256"]),
        "market_pointer_path": str(market_pointer_path),
        "market_pointer_sha256": market_pointer_sha,
        "market_manifest_path": str(market_manifest_path),
        "market_manifest_sha256": market_manifest_sha,
        "history_path": str(history_path),
        "history_sha256": history_sha,
    }


def canonical_input_closure(
    *, workspace_root: str | os.PathLike[str], market_data_root: str | os.PathLike[str]
) -> dict[str, str]:
    """Freeze exact current Market/PIT discovery and manifest bytes for a rollover CAS."""

    workspace = Path(workspace_root).resolve(strict=True)
    data_root = Path(market_data_root).resolve(strict=True)
    market_pointer_path = data_root / "parquet/cn/_latest.json"
    market_raw, market_pointer_sha = _read_owner_file(
        market_pointer_path, root=workspace, label="canonical Market pointer"
    )
    market_pointer = _mapping(market_raw, label="canonical Market pointer")
    manifest_value = market_pointer.get("manifest_path")
    if type(manifest_value) is not str:
        raise FactorGovernanceError("canonical Market manifest path is absent")
    market_manifest_path = Path(manifest_value)
    _market_manifest_raw, market_manifest_sha = _read_owner_file(
        market_manifest_path, root=workspace, label="canonical Market manifest"
    )
    pit_pointer_path = data_root / "parquet/cn/reference/stock_basic_membership_latest.json"
    pit_raw, pit_pointer_sha = _read_owner_file(
        pit_pointer_path, root=workspace, label="canonical PIT pointer"
    )
    pit_pointer = _mapping(pit_raw, label="canonical PIT pointer")
    pit_manifest_value = pit_pointer.get("generation_manifest_path")
    if type(pit_manifest_value) is not str:
        raise FactorGovernanceError("canonical PIT manifest path is absent")
    pit_manifest_path = Path(pit_manifest_value)
    _pit_manifest_raw, pit_manifest_sha = _read_owner_file(
        pit_manifest_path, root=workspace, label="canonical PIT manifest"
    )
    if pit_pointer.get("generation_manifest_sha256") != pit_manifest_sha:
        raise FactorGovernanceError("canonical PIT manifest SHA differs")
    return {
        "market_pointer_path": str(market_pointer_path),
        "market_pointer_sha256": market_pointer_sha,
        "market_manifest_path": str(market_manifest_path),
        "market_manifest_sha256": market_manifest_sha,
        "pit_pointer_path": str(pit_pointer_path),
        "pit_pointer_sha256": pit_pointer_sha,
        "pit_manifest_path": str(pit_manifest_path),
        "pit_manifest_sha256": pit_manifest_sha,
    }


__all__ = ["canonical_input_closure", "validate_daily_maintenance_receipt"]
