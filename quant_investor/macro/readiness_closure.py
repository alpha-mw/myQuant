"""Immutable, source-bound Macro pipeline readiness closure.

The closure proves only that the canonical Macro data pipeline and any prior
write-veto lifecycle closed.  It says nothing about the economic regime and
never grants System, portfolio, Paper, order, or trading authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Final, Mapping

from quant_investor.contracts import canonical_json_bytes

from .maintenance_transaction import (
    JOURNAL_SCHEMA,
    PHASES,
    PREPARED_SCHEMA,
    MacroMaintenanceTransactionError,
    generation_tree_sha256,
)
from .release_calendar import load_release_calendar
from .store import load_observations

CLOSURE_SCHEMA: Final = "cn-macro-readiness-closure.v1"
CLOSURE_ROOT: Final = Path("results/intelligence/macro_readiness")
TRANSACTION_ROOT: Final = Path("data/private/macro_recovery_transactions")
MARKET_POINTER: Final = Path("data/parquet/cn/_latest.json")
PIT_POINTER: Final = Path("data/parquet/cn/reference/stock_basic_membership_latest.json")
RELEASE_POINTER: Final = Path("data/parquet/cn/macro_release_calendar/_latest.json")
OBSERVATIONS_POINTER: Final = Path("data/parquet/cn/macro_observations/_latest.json")
VETO_PATH: Final = Path("data/private/cn_daily_maintenance/MACRO_WRITE_VETO.json")
VETO_ARCHIVE: Final = Path("data/private/cn_daily_maintenance/veto_archive")
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE: Final = re.compile(r"^[0-9]{8}$")
_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_MAX_BYTES: Final = 128 * 1024 * 1024


class MacroReadinessClosureError(ValueError):
    """Stable fail-closed error for Macro readiness evidence."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _workspace(value: str | os.PathLike[str]) -> Path:
    try:
        root = Path(value).resolve(strict=True)
        observed = os.lstat(root)
    except OSError as exc:
        raise MacroReadinessClosureError("MACRO_READINESS_WORKSPACE_INVALID") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise MacroReadinessClosureError("MACRO_READINESS_WORKSPACE_INVALID")
    return root


def _relative(value: Any, *, code: str) -> Path:
    text = str(value or "")
    candidate = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or candidate.is_absolute()
        or candidate.as_posix() != text
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise MacroReadinessClosureError(code)
    return Path(*candidate.parts)


def _safe_path(root: Path, relative: Path, *, code: str) -> Path:
    path = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise MacroReadinessClosureError(code) from exc
        if stat.S_ISLNK(observed.st_mode) or observed.st_uid != os.geteuid():
            raise MacroReadinessClosureError(code)
        if stat.S_IMODE(observed.st_mode) & 0o022:
            raise MacroReadinessClosureError(code)
    return path


def _read(
    root: Path,
    relative: Path,
    *,
    code: str,
    expected_sha256: str | None = None,
    max_bytes: int = _MAX_BYTES,
) -> bytes:
    path = _safe_path(root, relative, code=code)
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > max_bytes
        ):
            raise MacroReadinessClosureError(code)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        signature = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if signature != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise MacroReadinessClosureError(f"{code}_CHANGED")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if signature != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MacroReadinessClosureError(f"{code}_CHANGED")
    except MacroReadinessClosureError:
        raise
    except OSError as exc:
        raise MacroReadinessClosureError(code) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) != before.st_size:
        raise MacroReadinessClosureError(f"{code}_CHANGED")
    if expected_sha256 is not None and _sha(raw) != expected_sha256:
        raise MacroReadinessClosureError(f"{code}_SHA_MISMATCH")
    return raw


def _json(raw: bytes, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MacroReadinessClosureError(code) from exc
    if type(value) is not dict:
        raise MacroReadinessClosureError(code)
    return value


def _ref(value: Any, *, code: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise MacroReadinessClosureError(code)
    path = _relative(value["path"], code=code)
    digest = str(value["sha256"] or "")
    if _SHA_RE.fullmatch(digest) is None:
        raise MacroReadinessClosureError(code)
    return {"path": path.as_posix(), "sha256": digest}


def _instant(value: Any, *, code: str) -> datetime:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise MacroReadinessClosureError(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise MacroReadinessClosureError(code)
    return parsed.astimezone(timezone.utc)


def _instant_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _date(value: Any, *, code: str) -> str:
    text = str(value or "")
    if _DATE_RE.fullmatch(text) is None:
        raise MacroReadinessClosureError(code)
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise MacroReadinessClosureError(code) from exc
    return text


def _absolute_relative(root: Path, value: Any, *, code: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        raise MacroReadinessClosureError(code)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise MacroReadinessClosureError(code) from exc
    if root / relative != path:
        raise MacroReadinessClosureError(code)
    return relative


def _pointer_payload(raw: bytes, *, code: str) -> dict[str, Any]:
    value = _json(raw, code=code)
    generation = str(value.get("generation_id") or "")
    if _ID_RE.fullmatch(generation) is None:
        raise MacroReadinessClosureError(code)
    return value


def _journal(
    root: Path,
    terminal_ref: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Path, str]:
    terminal = _ref(terminal_ref, code="MACRO_READINESS_TERMINAL_REF_INVALID")
    terminal_path = Path(terminal["path"])
    parts = terminal_path.parts
    if (
        len(parts) != 7
        or Path(*parts[:3]) != TRANSACTION_ROOT
        or parts[4] != "journals"
        or parts[3] != parts[5]
        or parts[6] != "0007-terminal.json"
        or _ID_RE.fullmatch(parts[3]) is None
    ):
        raise MacroReadinessClosureError("MACRO_READINESS_TERMINAL_PATH_INVALID")
    journal_dir = terminal_path.parent
    expected_names = [
        f"{index:04d}-{phase.lower()}.json" for index, phase in enumerate(PHASES, start=1)
    ]
    directory = _safe_path(root, journal_dir, code="MACRO_READINESS_JOURNAL_UNSAFE")
    if sorted(path.name for path in directory.iterdir()) != expected_names:
        raise MacroReadinessClosureError("MACRO_READINESS_JOURNAL_SHAPE_INVALID")
    rows: list[dict[str, Any]] = []
    prepared_path = ""
    prepared_sha = ""
    previous_time: datetime | None = None
    for index, (phase, name) in enumerate(zip(PHASES, expected_names), start=1):
        relative = journal_dir / name
        raw = _read(root, relative, code="MACRO_READINESS_JOURNAL_RECORD_UNSAFE")
        value = _json(raw, code="MACRO_READINESS_JOURNAL_RECORD_INVALID")
        if set(value) != {
            "schema_version",
            "sequence",
            "phase",
            "recorded_at",
            "prepared_path",
            "prepared_sha256",
            "details",
        } or (
            value.get("schema_version") != JOURNAL_SCHEMA
            or value.get("sequence") != index
            or value.get("phase") != phase
            or type(value.get("details")) is not dict
        ):
            raise MacroReadinessClosureError("MACRO_READINESS_JOURNAL_RECORD_INVALID")
        recorded = _instant(value["recorded_at"], code="MACRO_READINESS_JOURNAL_TIME_INVALID")
        if previous_time is not None and recorded < previous_time:
            raise MacroReadinessClosureError("MACRO_READINESS_JOURNAL_TIME_INVALID")
        previous_time = recorded
        if index == 1:
            prepared_path = str(value["prepared_path"])
            prepared_sha = str(value["prepared_sha256"])
        elif (
            value.get("prepared_path") != prepared_path
            or value.get("prepared_sha256") != prepared_sha
        ):
            raise MacroReadinessClosureError("MACRO_READINESS_JOURNAL_PREPARED_DIFFERS")
        if _SHA_RE.fullmatch(prepared_sha) is None:
            raise MacroReadinessClosureError("MACRO_READINESS_PREPARED_SHA_INVALID")
        rows.append(
            {
                "phase": phase,
                "sequence": index,
                "recorded_at": _instant_text(recorded),
                "path": relative.as_posix(),
                "sha256": _sha(raw),
            }
        )
    if rows[-1]["sha256"] != terminal["sha256"]:
        raise MacroReadinessClosureError("MACRO_READINESS_TERMINAL_SHA_MISMATCH")
    terminal_value = _json(
        _read(
            root,
            terminal_path,
            code="MACRO_READINESS_TERMINAL_UNSAFE",
            expected_sha256=terminal["sha256"],
        ),
        code="MACRO_READINESS_TERMINAL_INVALID",
    )
    if terminal_value.get("details") != {"status": "SUCCESS"}:
        raise MacroReadinessClosureError("MACRO_READINESS_TERMINAL_NOT_SUCCESS")
    return (
        rows,
        _absolute_relative(root, prepared_path, code="MACRO_READINESS_PREPARED_PATH_INVALID"),
        prepared_sha,
    )


def _component(
    root: Path,
    prepared_dir: Path,
    prepared: Mapping[str, Any],
    *,
    name: str,
    canonical_root: Path,
) -> dict[str, Any]:
    value = prepared.get(name)
    if type(value) is not dict:
        raise MacroReadinessClosureError("MACRO_READINESS_COMPONENT_INVALID")
    expected_root = root / canonical_root.parent
    if Path(str(value.get("canonical_root") or "")) != expected_root:
        raise MacroReadinessClosureError("MACRO_READINESS_CANONICAL_ROOT_INVALID")
    generation_id = str(value.get("generation_id") or "")
    tree_sha = str(value.get("generation_tree_sha256") or "")
    pointer_sha = str(value.get("new_pointer_sha256") or "")
    if (
        _ID_RE.fullmatch(generation_id) is None
        or _SHA_RE.fullmatch(tree_sha) is None
        or _SHA_RE.fullmatch(pointer_sha) is None
    ):
        raise MacroReadinessClosureError("MACRO_READINESS_COMPONENT_INVALID")
    artifact_name = str(value.get("new_pointer_artifact") or "")
    artifact = _relative(artifact_name, code="MACRO_READINESS_POINTER_ARTIFACT_INVALID")
    frozen_path = prepared_dir / artifact
    frozen_raw = _read(
        root,
        frozen_path,
        code="MACRO_READINESS_POINTER_ARTIFACT_UNSAFE",
        expected_sha256=pointer_sha,
    )
    pointer = _pointer_payload(frozen_raw, code="MACRO_READINESS_POINTER_ARTIFACT_INVALID")
    if pointer["generation_id"] != generation_id:
        raise MacroReadinessClosureError("MACRO_READINESS_POINTER_GENERATION_DIFFERS")
    installed = canonical_root.parent / "_generations" / generation_id
    try:
        installed_tree = generation_tree_sha256(root / installed)
    except (OSError, MacroMaintenanceTransactionError) as exc:
        raise MacroReadinessClosureError("MACRO_READINESS_GENERATION_INVALID") from exc
    if installed_tree != tree_sha:
        raise MacroReadinessClosureError("MACRO_READINESS_GENERATION_TREE_DIFFERS")
    return {
        "current_path": canonical_root.as_posix(),
        "frozen_ref": {"path": frozen_path.as_posix(), "sha256": pointer_sha},
        "generation_id": generation_id,
        "generation_tree_sha256": tree_sha,
    }


def _veto_lifecycle(
    root: Path,
    prepared: Mapping[str, Any],
    *,
    terminal_time: datetime,
    require_current_absence: bool,
) -> dict[str, Any]:
    bindings = prepared.get("input_bindings")
    if type(bindings) is not dict:
        raise MacroReadinessClosureError("MACRO_READINESS_INPUT_BINDINGS_INVALID")
    veto = bindings.get("macro_veto")
    if veto is None:
        if require_current_absence and os.path.lexists(root / VETO_PATH):
            raise MacroReadinessClosureError("MACRO_READINESS_LIVE_VETO_PRESENT")
        return {"state": "NOT_PRESENT", "available_at": _instant_text(terminal_time)}
    if type(veto) is not dict or set(veto) != {"path", "sha256"}:
        raise MacroReadinessClosureError("MACRO_READINESS_VETO_BINDING_INVALID")
    if Path(str(veto["path"] or "")) != root / VETO_PATH:
        raise MacroReadinessClosureError("MACRO_READINESS_VETO_PATH_INVALID")
    veto_sha = str(veto["sha256"] or "")
    if _SHA_RE.fullmatch(veto_sha) is None:
        raise MacroReadinessClosureError("MACRO_READINESS_VETO_SHA_INVALID")
    archive = VETO_ARCHIVE / f"{veto_sha}.json"
    _read(
        root,
        archive,
        code="MACRO_READINESS_VETO_ARCHIVE_INVALID",
        expected_sha256=veto_sha,
    )
    archive_dir = _safe_path(root, VETO_ARCHIVE, code="MACRO_READINESS_VETO_ARCHIVE_INVALID")
    matches = sorted(archive_dir.glob(f"{veto_sha}.*.clear.json"))
    if len(matches) != 1:
        raise MacroReadinessClosureError("MACRO_READINESS_CLEAR_RECEIPT_NOT_UNIQUE")
    clear_relative = matches[0].relative_to(root)
    clear_raw = _read(root, clear_relative, code="MACRO_READINESS_CLEAR_RECEIPT_INVALID")
    clear = _json(clear_raw, code="MACRO_READINESS_CLEAR_RECEIPT_INVALID")
    if set(clear) != {
        "schema_version",
        "status",
        "cleared",
        "cleared_at",
        "reason",
        "lane",
        "archived_veto_ref",
    } or (
        clear.get("schema_version") != "cn-daily-maintenance-veto-clear.v1"
        or clear.get("status") != "CLEARED"
        or clear.get("cleared") is not True
        or clear.get("lane") != "macro"
        or clear.get("archived_veto_ref") != {"path": str(root / archive), "sha256": veto_sha}
    ):
        raise MacroReadinessClosureError("MACRO_READINESS_CLEAR_RECEIPT_INVALID")
    cleared = _instant(clear["cleared_at"], code="MACRO_READINESS_CLEAR_TIME_INVALID")
    if cleared < terminal_time:
        raise MacroReadinessClosureError("MACRO_READINESS_CLEAR_TIME_INVALID")
    if require_current_absence and os.path.lexists(root / VETO_PATH):
        raise MacroReadinessClosureError("MACRO_READINESS_LIVE_VETO_PRESENT")
    return {
        "state": "CLEARED",
        "available_at": _instant_text(cleared),
        "original_veto_sha256": veto_sha,
        "archive_ref": {"path": archive.as_posix(), "sha256": veto_sha},
        "clear_receipt_ref": {
            "path": clear_relative.as_posix(),
            "sha256": _sha(clear_raw),
        },
    }


def _input_refs(root: Path, prepared: Mapping[str, Any]) -> list[dict[str, str]]:
    bindings = prepared.get("input_bindings")
    if type(bindings) is not dict:
        raise MacroReadinessClosureError("MACRO_READINESS_INPUT_BINDINGS_INVALID")
    rows: list[dict[str, str]] = []
    skip = {"macro_veto", "market_pointer_authority", "pit_pointer_authority"}
    for name, value in sorted(bindings.items()):
        if name in skip:
            continue
        if type(value) is not dict or set(value) != {"path", "sha256"}:
            raise MacroReadinessClosureError("MACRO_READINESS_INPUT_BINDING_INVALID")
        relative = _absolute_relative(
            root,
            value["path"],
            code="MACRO_READINESS_INPUT_BINDING_PATH_INVALID",
        )
        digest = str(value["sha256"] or "")
        if _SHA_RE.fullmatch(digest) is None:
            raise MacroReadinessClosureError("MACRO_READINESS_INPUT_BINDING_INVALID")
        _read(
            root,
            relative,
            code="MACRO_READINESS_INPUT_BINDING_INVALID",
            expected_sha256=digest,
        )
        rows.append({"name": str(name), "path": relative.as_posix(), "sha256": digest})
    return rows


def build_macro_readiness_closure(
    *,
    workspace_root: str | os.PathLike[str],
    terminal_path: str,
    terminal_sha256: str,
    _require_current_veto_absence: bool = True,
) -> dict[str, Any]:
    """Build one deterministic closure projection without writing it."""

    root = _workspace(workspace_root)
    journal_rows, prepared_relative, prepared_sha = _journal(
        root,
        {"path": terminal_path, "sha256": terminal_sha256},
    )
    prepared_raw = _read(
        root,
        prepared_relative,
        code="MACRO_READINESS_PREPARED_INVALID",
        expected_sha256=prepared_sha,
    )
    prepared = _json(prepared_raw, code="MACRO_READINESS_PREPARED_INVALID")
    if set(prepared) != {
        "schema_version",
        "target_date",
        "prepared_at",
        "authority_mode",
        "release",
        "observations",
        "input_bindings",
        "authorities",
    } or (
        prepared.get("schema_version") != PREPARED_SCHEMA
        or prepared.get("authority_mode") != "canonical"
    ):
        raise MacroReadinessClosureError("MACRO_READINESS_PREPARED_INVALID")
    target = _date(prepared["target_date"], code="MACRO_READINESS_TARGET_DATE_INVALID")
    authorities = prepared.get("authorities")
    if type(authorities) is not dict or set(authorities) != {"market", "pit"}:
        raise MacroReadinessClosureError("MACRO_READINESS_AUTHORITIES_INVALID")
    prepared_dir = prepared_relative.parent
    frozen: dict[str, Any] = {}
    for name, relative in (("market", MARKET_POINTER), ("pit", PIT_POINTER)):
        authority = authorities.get(name)
        if (
            type(authority) is not dict
            or Path(str(authority.get("pointer_path") or "")) != root / relative
        ):
            raise MacroReadinessClosureError("MACRO_READINESS_AUTHORITY_PATH_INVALID")
        digest = str(authority.get("pointer_sha256") or "")
        artifact = _relative(
            authority.get("pointer_artifact"),
            code="MACRO_READINESS_AUTHORITY_ARTIFACT_INVALID",
        )
        artifact_relative = prepared_dir / artifact
        raw = _read(
            root,
            artifact_relative,
            code="MACRO_READINESS_AUTHORITY_ARTIFACT_INVALID",
            expected_sha256=digest,
        )
        frozen[name] = {
            "current_path": relative.as_posix(),
            "frozen_ref": {"path": artifact_relative.as_posix(), "sha256": digest},
            "snapshot": _json(raw, code="MACRO_READINESS_AUTHORITY_INVALID"),
        }
    release = _component(
        root,
        prepared_dir,
        prepared,
        name="release",
        canonical_root=RELEASE_POINTER,
    )
    observations = _component(
        root,
        prepared_dir,
        prepared,
        name="observations",
        canonical_root=OBSERVATIONS_POINTER,
    )
    observations_pointer = _pointer_payload(
        _read(
            root,
            Path(observations["frozen_ref"]["path"]),
            code="MACRO_READINESS_OBSERVATIONS_POINTER_INVALID",
            expected_sha256=observations["frozen_ref"]["sha256"],
        ),
        code="MACRO_READINESS_OBSERVATIONS_POINTER_INVALID",
    )
    metadata = observations_pointer.get("metadata")
    market_snapshot = frozen["market"]["snapshot"]
    pit_snapshot = frozen["pit"]["snapshot"]
    if market_snapshot.get("latest_complete_trade_date") != target or not str(
        pit_snapshot.get("generation_id") or ""
    ).startswith(f"pit-{target}-"):
        raise MacroReadinessClosureError("MACRO_READINESS_AUTHORITY_DATE_INVALID")
    if type(metadata) is not dict or (
        metadata.get("as_of") != target
        or metadata.get("local_target_trade_date") != target
        or metadata.get("latest_local_trade_date") != target
    ):
        raise MacroReadinessClosureError("MACRO_READINESS_OBSERVATIONS_DATE_INVALID")
    release_generation = root / RELEASE_POINTER.parent / "_generations" / release["generation_id"]
    market_days = _json(
        _read(
            root,
            release_generation.relative_to(root) / "market_open_days.json",
            code="MACRO_READINESS_RELEASE_DATE_INVALID",
        ),
        code="MACRO_READINESS_RELEASE_DATE_INVALID",
    )
    if target not in market_days.get("open_dates", []):
        raise MacroReadinessClosureError("MACRO_READINESS_RELEASE_DATE_INVALID")
    terminal_time = _instant(
        journal_rows[-1]["recorded_at"], code="MACRO_READINESS_TERMINAL_TIME_INVALID"
    )
    veto = _veto_lifecycle(
        root,
        prepared,
        terminal_time=terminal_time,
        require_current_absence=_require_current_veto_absence,
    )
    available = _instant(veto["available_at"], code="MACRO_READINESS_AVAILABLE_AT_INVALID")
    return {
        "schema_version": CLOSURE_SCHEMA,
        "status": "READY",
        "target_date": target,
        "available_at": _instant_text(available),
        "transaction_id": Path(terminal_path).parts[3],
        "journal_refs": journal_rows,
        "prepared_ref": {"path": prepared_relative.as_posix(), "sha256": prepared_sha},
        "frozen_pointers": {
            **frozen,
            "release": release,
            "observations": observations,
        },
        "input_refs": _input_refs(root, prepared),
        "veto_lifecycle": veto,
        "research_only": True,
        "system_authority": False,
        "portfolio_authority": False,
        "paper_authority": False,
        "broker": False,
        "order": False,
        "trade": False,
    }


def validate_macro_readiness_closure(
    *,
    workspace_root: str | os.PathLike[str],
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay intrinsic historical closure without requiring current heads."""

    root = _workspace(workspace_root)
    value = dict(closure)
    required = {
        "schema_version",
        "status",
        "target_date",
        "available_at",
        "transaction_id",
        "journal_refs",
        "prepared_ref",
        "frozen_pointers",
        "input_refs",
        "veto_lifecycle",
        "research_only",
        "system_authority",
        "portfolio_authority",
        "paper_authority",
        "broker",
        "order",
        "trade",
    }
    if set(value) != required or (
        value.get("schema_version") != CLOSURE_SCHEMA
        or value.get("status") != "READY"
        or value.get("research_only") is not True
        or any(
            value.get(field) is not False
            for field in (
                "system_authority",
                "portfolio_authority",
                "paper_authority",
                "broker",
                "order",
                "trade",
            )
        )
    ):
        raise MacroReadinessClosureError("MACRO_READINESS_CLOSURE_INVALID")
    journal = value.get("journal_refs")
    if type(journal) is not list or len(journal) != 7:
        raise MacroReadinessClosureError("MACRO_READINESS_CLOSURE_INVALID")
    terminal = journal[-1]
    if type(terminal) is not dict:
        raise MacroReadinessClosureError("MACRO_READINESS_CLOSURE_INVALID")
    rebuilt = build_macro_readiness_closure(
        workspace_root=root,
        terminal_path=str(terminal.get("path") or ""),
        terminal_sha256=str(terminal.get("sha256") or ""),
        _require_current_veto_absence=False,
    )
    if rebuilt != value:
        raise MacroReadinessClosureError("MACRO_READINESS_CLOSURE_DOES_NOT_REPLAY")
    return value


def verify_current_macro_readiness_closure(
    *,
    workspace_root: str | os.PathLike[str],
    closure: Mapping[str, Any],
    expected_target_date: str,
    decision_as_of: str,
) -> dict[str, Any]:
    """Verify intrinsic closure plus current heads and decision-time availability."""

    root = _workspace(workspace_root)
    value = validate_macro_readiness_closure(workspace_root=root, closure=closure)
    target = _date(expected_target_date, code="MACRO_READINESS_TARGET_DATE_INVALID")
    if value["target_date"] != target:
        raise MacroReadinessClosureError("MACRO_READINESS_TARGET_DATE_DIFFERS")
    as_of = _instant(decision_as_of, code="MACRO_READINESS_DECISION_TIME_INVALID")
    if (
        as_of.strftime("%Y%m%d") != target
        or _instant(value["available_at"], code="MACRO_READINESS_AVAILABLE_AT_INVALID") > as_of
    ):
        raise MacroReadinessClosureError("MACRO_READINESS_NOT_AVAILABLE_AT_DECISION")
    if os.path.lexists(root / VETO_PATH):
        raise MacroReadinessClosureError("MACRO_READINESS_LIVE_VETO_PRESENT")
    pointers = value["frozen_pointers"]
    for name in ("market", "pit", "release", "observations"):
        row = pointers[name]
        current = Path(row["current_path"])
        expected = row["frozen_ref"]["sha256"]
        _read(
            root, current, code="MACRO_READINESS_CURRENT_POINTER_INVALID", expected_sha256=expected
        )
    try:
        load_release_calendar(
            canonical_root=root / RELEASE_POINTER.parent,
            expected_pointer_sha256=pointers["release"]["frozen_ref"]["sha256"],
        )
        rows, observations = load_observations(root / OBSERVATIONS_POINTER.parent)
    except Exception as exc:
        raise MacroReadinessClosureError("MACRO_READINESS_SEMANTIC_POSTCHECK_FAILED") from exc
    if (
        not rows
        or observations.get("pointer_sha256") != pointers["observations"]["frozen_ref"]["sha256"]
        or observations.get("generation_id") != pointers["observations"]["generation_id"]
    ):
        raise MacroReadinessClosureError("MACRO_READINESS_SEMANTIC_POSTCHECK_FAILED")
    projection = {
        "closure_sha256": _sha(canonical_json_bytes(value)),
        "target_date": target,
        "available_at": value["available_at"],
        "pointer_sha256": {
            name: pointers[name]["frozen_ref"]["sha256"]
            for name in ("market", "pit", "release", "observations")
        },
        "veto_state": value["veto_lifecycle"]["state"],
    }
    return projection


def seal_macro_readiness_closure(
    *,
    workspace_root: str | os.PathLike[str],
    terminal_path: str,
    terminal_sha256: str,
) -> dict[str, Any]:
    """Write one deterministic content-addressed closure receipt."""

    root = _workspace(workspace_root)
    closure = build_macro_readiness_closure(
        workspace_root=root,
        terminal_path=terminal_path,
        terminal_sha256=terminal_sha256,
    )
    raw = canonical_json_bytes(closure)
    digest = _sha(raw)
    parent = root / CLOSURE_ROOT / closure["target_date"]
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root / CLOSURE_ROOT, 0o700)
    os.chmod(parent, 0o700)
    path = parent / f"{digest}.json"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        existing = _read(
            root,
            path.relative_to(root),
            code="MACRO_READINESS_CLOSURE_CONFLICT",
            expected_sha256=digest,
        )
        if existing != raw:
            raise MacroReadinessClosureError("MACRO_READINESS_CLOSURE_CONFLICT")
        return {
            "status": "NO_ACTION",
            "closure_path": path.relative_to(root).as_posix(),
            "closure_sha256": digest,
            "closure": closure,
        }
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _read(
        root,
        path.relative_to(root),
        code="MACRO_READINESS_CLOSURE_WRITE_FAILED",
        expected_sha256=digest,
    )
    return {
        "status": "SEALED",
        "closure_path": path.relative_to(root).as_posix(),
        "closure_sha256": digest,
        "closure": closure,
    }


__all__ = [
    "CLOSURE_SCHEMA",
    "MacroReadinessClosureError",
    "build_macro_readiness_closure",
    "seal_macro_readiness_closure",
    "validate_macro_readiness_closure",
    "verify_current_macro_readiness_closure",
]
