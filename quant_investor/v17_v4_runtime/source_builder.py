"""Offline, fail-closed materialization for V17 v4 Forward Evidence sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.v17_v4_contract import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import strict_json_loads

from .research_factor_set import (
    ResearchFactorSetError,
    ResearchFactorSetStore,
)
from .source_storage import (
    SourceExactOnceConflict,
    SourceStorageError,
    SourceStore,
)

PROTOCOL_VERSION: Final = "myquant.v17.v4"
SLICE_MANIFEST_VERSION: Final = "myquant.v17.v4.forward-source-slice-manifest.v1"
FACTOR_BUNDLE_VERSION: Final = "myquant.v17.v4.forward-factor-input-bundle.v1"
SOURCE_LOCATOR_VERSION: Final = "myquant.v17.v4.forward-source-locator.v1"
PARQUET_VERSION: Final = "myquant.v17.v4.forward-source-parquet.v1"
SNAPSHOT_ROOT: Final = PurePosixPath("data/private/v17_v4_sources/snapshots")
MARKET_POINTER: Final = PurePosixPath("data/parquet/cn/_latest.json")
FUNDAMENTAL_POINTER: Final = PurePosixPath("data/parquet/cn/_fundamental_latest.json")
FACTOR_POINTER: Final = PurePosixPath(
    "data/private/v17_v4_sources/research_factor_sets/_current.json"
)
FACTOR_FIELDS: Final = (
    "adj_close",
    "fin_debt_to_assets",
    "fin_ocf_to_profit",
    "fin_roe",
    "total_mv",
)
NEUTRALIZER_FIELDS: Final = (
    "beta_252d",
    "industry",
    "log_market_cap",
)
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
_IDENTIFIER_RE: Final = re.compile(
    r"^[a-z0-9][a-z0-9_.:-]{0,127}$",
    re.ASCII,
)
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_CUTOFF_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T" r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_MAX_EXTERNAL_FILE_BYTES: Final = 8 * 1024 * 1024 * 1024
_MAX_EXTERNAL_JSON_BYTES: Final = 256 * 1024 * 1024
_MARKET_LOOKBACK_CALENDAR_DAYS: Final = 550
_BETA_WINDOW: Final = 252
_BETA_MIN_OBSERVATIONS: Final = 126


class SourceSnapshotError(RuntimeError):
    """Base error for the bounded source materializer."""

    exit_code = 2


class SourceSnapshotGap(SourceSnapshotError):
    """An exact current canonical input is absent, stale, or inconsistent."""

    def __init__(self, *blockers: str) -> None:
        normalized = tuple(sorted(set(blockers)))
        super().__init__("TRUE_CURRENT_CANONICAL_INPUT_GAP: " + "; ".join(normalized))
        self.blockers = normalized


@dataclass(frozen=True)
class _BoundFile:
    path: Path
    relative_path: str
    byte_sha256: str
    size: int
    identity: tuple[int, int, int, int, int, int, int]
    raw: bytes | None = None


@dataclass(frozen=True)
class _MaterializedSlice:
    field_name: str
    parquet_path: str
    parquet_raw: bytes
    manifest_path: str
    manifest: dict[str, Any]
    manifest_raw: bytes
    manifest_ref: dict[str, str]
    row_count: int
    non_null_count: int


def _gap(code: str, detail: str) -> SourceSnapshotGap:
    return SourceSnapshotGap(f"{code}: {detail}")


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise _gap("INVALID_IDENTIFIER", label)
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise _gap("INVALID_SHA256", label)
    return value


def _day(value: Any, *, label: str) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        result = value.date().isoformat()
    elif isinstance(value, date):
        result = value.isoformat()
    elif type(value) is str:
        text = value.strip()
        if re.fullmatch(r"[0-9]{8}", text, re.ASCII):
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            result = date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise _gap("INVALID_DATE", label) from exc
    else:
        raise _gap("INVALID_DATE", label)
    return result


def _cutoff(value: Any, *, label: str) -> str:
    if type(value) is not str or _CUTOFF_RE.fullmatch(value) is None:
        raise _gap("INVALID_CUTOFF", label)
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise _gap("INVALID_CUTOFF", label) from exc
    return value


def _available_at(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise _gap("MISSING_AVAILABLE_AT", label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _gap("INVALID_AVAILABLE_AT", label) from exc
    if parsed.tzinfo is None:
        raise _gap("INVALID_AVAILABLE_AT", f"{label} is timezone-naive")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.microsecond:
        parsed = parsed + timedelta(seconds=1)
    return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _before_cutoff(value: str, cutoff: str, *, label: str) -> str:
    if value > cutoff:
        raise _gap(
            "SOURCE_AFTER_CUTOFF",
            f"{label} available_at={value} cutoff={cutoff}",
        )
    return value


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _workspace_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value)
    if not root.is_absolute() or any(part in {"", ".", ".."} for part in root.parts[1:]):
        raise _gap("INVALID_WORKSPACE_ROOT", "workspace root is not canonical")
    resolved = root.resolve(strict=True)
    if resolved != root or not root.is_dir():
        raise _gap("INVALID_WORKSPACE_ROOT", "workspace root resolves elsewhere")
    return root


def _source_path(
    root: Path,
    value: str | os.PathLike[str] | PurePosixPath,
) -> tuple[Path, str]:
    text = str(value)
    if not text or "\\" in text:
        raise _gap("INVALID_SOURCE_PATH", text or "<empty>")
    candidate_value = Path(text)
    if any(part in {"", ".", ".."} for part in candidate_value.parts):
        raise _gap("INVALID_SOURCE_PATH", text)
    candidate = candidate_value if candidate_value.is_absolute() else root / candidate_value
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise _gap("SOURCE_OUTSIDE_WORKSPACE", text) from exc
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            current_stat = current.lstat()
        except FileNotFoundError as exc:
            raise _gap("SOURCE_MISSING", relative.as_posix()) from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise _gap("SOURCE_SYMLINK_REJECTED", relative.as_posix())
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(current_stat.st_mode):
            raise _gap("SOURCE_PARENT_NOT_DIRECTORY", relative.as_posix())
    return candidate, relative.as_posix()


def _relative_to_pointer(
    pointer_path: PurePosixPath,
    value: Any,
) -> str:
    if type(value) is not str or not value:
        raise _gap("POINTER_PATH_MISSING", str(pointer_path))
    candidate = PurePosixPath(value)
    if candidate.is_absolute():
        return value
    return str(pointer_path.parent / candidate)


def _bind_file(
    root: Path,
    value: str | os.PathLike[str] | PurePosixPath,
    *,
    expected_sha256: str | None = None,
    include_raw: bool = False,
) -> _BoundFile:
    path, relative = _source_path(root, value)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise _gap("SOURCE_OPEN_FAILED", relative) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise _gap("SOURCE_NOT_REGULAR", relative)
        if before.st_size > _MAX_EXTERNAL_FILE_BYTES:
            raise _gap("SOURCE_TOO_LARGE", relative)
        if include_raw and before.st_size > _MAX_EXTERNAL_JSON_BYTES:
            raise _gap("SOURCE_JSON_TOO_LARGE", relative)
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if include_raw else None
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _identity(before) != _identity(after) or size != after.st_size:
        raise _gap("SOURCE_CHANGED_DURING_READ", relative)
    observed = digest.hexdigest()
    if expected_sha256 is not None and observed != _sha256(
        expected_sha256,
        label=f"{relative} expected SHA",
    ):
        raise _gap(
            "SOURCE_SHA256_MISMATCH",
            f"path={relative} expected={expected_sha256} observed={observed}",
        )
    return _BoundFile(
        path,
        relative,
        observed,
        size,
        _identity(after),
        None if chunks is None else b"".join(chunks),
    )


def _assert_unchanged(binding: _BoundFile) -> None:
    try:
        observed = binding.path.lstat()
    except FileNotFoundError as exc:
        raise _gap("SOURCE_DISAPPEARED", binding.relative_path) from exc
    if _identity(observed) != binding.identity:
        raise _gap("SOURCE_CHANGED_BEFORE_PUBLICATION", binding.relative_path)


def _read_json(
    root: Path,
    value: str | os.PathLike[str] | PurePosixPath,
    *,
    expected_sha256: str | None = None,
    label: str,
) -> tuple[dict[str, Any], _BoundFile]:
    binding = _bind_file(
        root,
        value,
        expected_sha256=expected_sha256,
        include_raw=True,
    )
    assert binding.raw is not None
    try:
        payload = strict_json_loads(binding.raw, label=label)
    except (TypeError, ValueError) as exc:
        raise _gap("SOURCE_JSON_INVALID", binding.relative_path) from exc
    if type(payload) is not dict:
        raise _gap("SOURCE_JSON_INVALID", f"{binding.relative_path} root")
    return payload, binding


def _large_external_json_projection(
    raw: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    """Parse one bounded upstream manifest without applying V17 artifact limits."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def parse_integer(token: str) -> int:
        digits = token[1:] if token.startswith("-") else token
        if len(digits) > 64:
            raise ValueError("integer digit limit")
        return int(token)

    def parse_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return value

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite constant: {token}")

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=parse_float,
            parse_int=parse_integer,
        )
    except (RecursionError, UnicodeError, TypeError, ValueError) as exc:
        raise _gap("SOURCE_JSON_INVALID", label) from exc
    if type(payload) is not dict:
        raise _gap("SOURCE_JSON_INVALID", f"{label} root")
    daily = (payload.get("tables") or {}).get("fundamental_daily")
    if type(daily) is not dict:
        raise _gap("FUNDAMENTAL_MANIFEST_SCHEMA_GAP", "fundamental_daily")
    return {
        "generation_id": payload.get("generation_id"),
        "status": payload.get("status"),
        "tables": {
            "fundamental_daily": {
                "sha256": daily.get("sha256"),
            }
        },
    }


def _read_fundamental_manifest(
    root: Path,
    value: str | os.PathLike[str] | PurePosixPath,
) -> tuple[dict[str, Any], _BoundFile]:
    binding = _bind_file(root, value, include_raw=True)
    assert binding.raw is not None
    payload = _large_external_json_projection(
        binding.raw,
        label=binding.relative_path,
    )
    return payload, binding


def _source_ref(
    binding: _BoundFile,
    *,
    role: str,
    as_of: str,
    available_at: str,
    media_type: str,
) -> dict[str, str]:
    return {
        "as_of": _day(as_of, label=f"{role}.as_of"),
        "available_at": _cutoff(available_at, label=f"{role}.available_at"),
        "byte_sha256": binding.byte_sha256,
        "media_type": media_type,
        "relative_path": binding.relative_path,
        "role": _identifier(role, label=f"{role}.role"),
        "status": "VERIFIED",
    }


def _sorted_source_refs(values: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = [dict(value) for value in values]
    rows.sort(key=lambda row: (row["role"], row["relative_path"]))
    if len({(row["role"], row["relative_path"]) for row in rows}) != len(rows):
        raise _gap("DUPLICATE_SOURCE_REF", "role/path collision")
    return rows


def _fraction(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0"
    value = f"{numerator / denominator:.8f}".rstrip("0").rstrip(".")
    return value or "0"


def _logical_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return format(numeric, ".17g")
    return str(value)


def _table_semantic_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        canonical_bytes(
            {
                "algorithm": "myquant.v17.v4.forward-source-table.v1",
                "columns": list(frame.columns),
            }
        )
    )
    for row in frame.itertuples(index=False, name=None):
        digest.update(b"\n")
        digest.update(canonical_bytes([_logical_scalar(value) for value in row]))
    return digest.hexdigest()


def _parquet_bytes(
    frame: pd.DataFrame,
    *,
    artifact_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    available_at: str,
) -> tuple[bytes, str, list[dict[str, Any]]]:
    semantic_sha = _table_semantic_sha256(frame)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = {
        b"artifact_id": artifact_id.encode("ascii"),
        b"artifact_version": PARQUET_VERSION.encode("ascii"),
        b"available_at": available_at.encode("ascii"),
        b"cutoff": cutoff.encode("ascii"),
        b"decision_session": decision_session.encode("ascii"),
        b"schema_version": PARQUET_VERSION.encode("ascii"),
        b"semantic_sha256": semantic_sha.encode("ascii"),
        b"strategy_id": strategy_id.encode("ascii"),
    }
    table = table.replace_schema_metadata(metadata)
    sink = io.BytesIO()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        data_page_version="1.0",
        use_dictionary=False,
        version="2.6",
        write_statistics=True,
    )
    columns: list[dict[str, Any]] = []
    for position, column in enumerate(frame.columns):
        dtype = table.schema.field(column).type
        if pa.types.is_floating(dtype):
            logical = "float64"
        elif pa.types.is_integer(dtype):
            logical = "int64"
        else:
            logical = "string"
        columns.append(
            {
                "logical_type": logical,
                "name": column,
                "nullable": bool(frame[column].isna().any()),
                "position": position,
            }
        )
    return sink.getvalue(), semantic_sha, columns


def _artifact_ref(
    document: Mapping[str, Any],
    raw: bytes,
    *,
    identity_field: str,
    relative_path: str,
) -> dict[str, str]:
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _materialize_slice(
    frame: pd.DataFrame,
    *,
    artifact_kind: str,
    field_name: str,
    value_fields: Sequence[str],
    relative_parquet_path: str,
    source_refs: Sequence[Mapping[str, str]],
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    available_at: str,
) -> _MaterializedSlice:
    if frame.empty:
        raise _gap("EMPTY_MATERIALIZED_SLICE", field_name)
    frame = frame.reset_index(drop=True)
    artifact_id = f"forward-source-{field_name.replace('_', '-')}-{decision_session}"
    parquet_raw, parquet_semantic, columns = _parquet_bytes(
        frame,
        artifact_id=f"{artifact_id}-parquet",
        strategy_id=strategy_id,
        decision_session=decision_session,
        cutoff=cutoff,
        available_at=available_at,
    )
    parquet_ref = {
        "artifact_id": f"{artifact_id}-parquet",
        "artifact_version": PARQUET_VERSION,
        "byte_sha256": hashlib.sha256(parquet_raw).hexdigest(),
        "cutoff": cutoff,
        "relative_path": relative_parquet_path,
        "semantic_sha256": parquet_semantic,
        "strategy_id": strategy_id,
    }
    non_null = int(frame[list(value_fields)].notna().all(axis=1).sum())
    row_count = len(frame)
    sessions = sorted(
        {
            _day(value, label=f"{field_name}.trade_date")
            for value in frame["trade_date"].dropna().tolist()
        }
    )
    if not sessions:
        raise _gap("SLICE_SESSION_MISSING", field_name)
    manifest_path = str(PurePosixPath(relative_parquet_path).with_suffix(".manifest.json"))
    manifest = seal_semantic(
        {
            "artifact_kind": artifact_kind,
            "authority": dict(NO_AUTHORITY),
            "available_at": available_at,
            "canary_evidence_eligible": False,
            "columns": columns,
            "coverage": {
                "coverage_ratio": _fraction(non_null, row_count),
                "non_null_count": non_null,
                "null_count": row_count - non_null,
            },
            "cutoff": cutoff,
            "decision_session": decision_session,
            "field_name": field_name,
            "first_session": sessions[0],
            "formal_activation_eligible": False,
            "last_session": sessions[-1],
            "manifest_id": artifact_id,
            "parquet_ref": parquet_ref,
            "parquet_schema_version": PARQUET_VERSION,
            "performance_evidence_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "row_count": row_count,
            "schema_validation_status": "VALIDATED",
            "shadow_only": True,
            "source_refs": _sorted_source_refs(source_refs),
            "strategy_id": strategy_id,
            "version": SLICE_MANIFEST_VERSION,
        }
    )
    validate_artifact(manifest)
    manifest_raw = canonical_resource_bytes(manifest)
    return _MaterializedSlice(
        field_name,
        relative_parquet_path,
        parquet_raw,
        manifest_path,
        manifest,
        manifest_raw,
        _artifact_ref(
            manifest,
            manifest_raw,
            identity_field="manifest_id",
            relative_path=manifest_path,
        ),
        row_count,
        non_null,
    )


def _month_parts(table_root: Path, start: date, end: date) -> list[Path]:
    parts = sorted(table_root.rglob("part.parquet"))
    if not parts:
        raise _gap("MARKET_PARQUET_MISSING", table_root.as_posix())
    selected: list[Path] = []
    for path in parts:
        year_match = next(
            (part.removeprefix("year=") for part in path.parts if part.startswith("year=")),
            None,
        )
        month_match = next(
            (part.removeprefix("month=") for part in path.parts if part.startswith("month=")),
            None,
        )
        if year_match is None or month_match is None:
            selected.append(path)
            continue
        try:
            month_start = date(int(year_match), int(month_match), 1)
        except ValueError as exc:
            raise _gap("MARKET_PARTITION_INVALID", path.as_posix()) from exc
        next_month = (
            date(month_start.year + 1, 1, 1)
            if month_start.month == 12
            else date(month_start.year, month_start.month + 1, 1)
        )
        if next_month > start and month_start <= end:
            selected.append(path)
    if not selected:
        raise _gap("MARKET_LOOKBACK_MISSING", f"{start}..{end}")
    return selected


def _read_market(
    root: Path,
    table_root_value: Any,
    *,
    symbols: set[str],
    decision_session: str,
) -> tuple[pd.DataFrame, list[_BoundFile]]:
    table_root, _ = _source_path(root, str(table_root_value))
    if not table_root.is_dir():
        raise _gap("MARKET_TABLE_ROOT_INVALID", str(table_root_value))
    end = date.fromisoformat(decision_session)
    start = end - timedelta(days=_MARKET_LOOKBACK_CALENDAR_DAYS)
    bindings: list[_BoundFile] = []
    frames: list[pd.DataFrame] = []
    for path in _month_parts(table_root, start, end):
        binding = _bind_file(root, path)
        bindings.append(binding)
        try:
            frame = pq.read_table(
                path,
                columns=[
                    "symbol",
                    "trade_date",
                    "adj_close",
                    "total_mv",
                ],
            ).to_pandas()
        except (OSError, ValueError, pa.ArrowException) as exc:
            raise _gap("MARKET_PARQUET_INVALID", binding.relative_path) from exc
        frame["trade_date"] = frame["trade_date"].map(
            lambda value: _day(value, label="market.trade_date")
        )
        frame = frame[
            frame["symbol"].astype(str).isin(symbols)
            & (frame["trade_date"] >= start.isoformat())
            & (frame["trade_date"] <= decision_session)
        ]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise _gap("MARKET_UNIVERSE_EMPTY", decision_session)
    market = pd.concat(frames, ignore_index=True)
    market["symbol"] = market["symbol"].astype(str)
    for field_name in ("adj_close", "total_mv"):
        market[field_name] = pd.to_numeric(
            market[field_name],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    market = market.sort_values(
        ["trade_date", "symbol"],
        kind="mergesort",
    ).drop_duplicates(["trade_date", "symbol"], keep="last")
    sessions = sorted(market["trade_date"].unique().tolist())
    if not sessions or sessions[-1] != decision_session:
        raise _gap(
            "MARKET_DECISION_SESSION_MISSING",
            f"expected={decision_session} observed={sessions[-1] if sessions else 'EMPTY'}",
        )
    if len(sessions) < _BETA_WINDOW + 1:
        raise _gap(
            "MARKET_LOOKBACK_INCOMPLETE",
            f"required={_BETA_WINDOW + 1} observed={len(sessions)}",
        )
    keep_sessions = set(sessions[-(_BETA_WINDOW + 1) :])
    market = market[market["trade_date"].isin(keep_sessions)].reset_index(drop=True)
    return market, bindings


def _read_pit_membership(
    path: Path,
    *,
    decision_session: str,
    cutoff: str,
) -> pd.DataFrame:
    try:
        frame = pq.read_table(path).to_pandas()
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise _gap("PIT_MEMBERSHIP_INVALID", path.as_posix()) from exc
    required = {
        "symbol",
        "industry",
        "source_list_status",
        "effective_from",
        "effective_to",
        "observed_at",
    }
    if not required.issubset(frame.columns):
        raise _gap(
            "PIT_MEMBERSHIP_SCHEMA_GAP",
            ",".join(sorted(required - set(frame.columns))),
        )
    frame["symbol"] = frame["symbol"].astype(str)
    frame["effective_from"] = frame["effective_from"].map(
        lambda value: _day(value, label="pit.effective_from")
    )
    effective_to = frame["effective_to"]
    active_to = effective_to.map(
        lambda value: bool(pd.isna(value)) or str(value).strip() in {"", "None", "NaT", "nan"}
    )
    normalized_to = effective_to.map(
        lambda value: (
            decision_session
            if bool(pd.isna(value)) or str(value).strip() in {"", "None", "NaT", "nan"}
            else _day(value, label="pit.effective_to")
        )
    )
    observed = (
        frame["observed_at"]
        .dropna()
        .map(lambda value: _available_at(str(value), label="pit.observed_at"))
    )
    if not observed.empty and observed.max() > cutoff:
        raise _gap(
            "PIT_OBSERVED_AFTER_CUTOFF",
            f"observed={observed.max()} cutoff={cutoff}",
        )
    active = frame[
        (frame["source_list_status"].astype(str) == "L")
        & (frame["effective_from"] <= decision_session)
        & (active_to | (normalized_to >= decision_session))
    ].copy()
    active = active.sort_values(
        ["symbol", "effective_from"],
        kind="mergesort",
    ).drop_duplicates("symbol", keep="last")
    if active.empty:
        raise _gap("PIT_ACTIVE_MEMBERSHIP_EMPTY", decision_session)
    return active[["symbol", "industry"]].reset_index(drop=True)


def _read_latest_fundamentals(
    path: Path,
    *,
    symbols: set[str],
    decision_session: str,
    cutoff: str,
) -> pd.DataFrame:
    columns = [
        "ts_code",
        "trade_date",
        "end_date",
        "availability_date",
        "fetched_at",
        "fin_roe",
        "fin_ocf_to_profit",
        "fin_debt_to_assets",
    ]
    try:
        parquet = pq.ParquetFile(path)
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise _gap("FUNDAMENTAL_PARQUET_INVALID", path.as_posix()) from exc
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise _gap(
            "FUNDAMENTAL_SCHEMA_GAP",
            ",".join(sorted(set(columns) - set(parquet.schema_arrow.names))),
        )
    latest = pd.DataFrame(columns=columns)
    try:
        batches = parquet.iter_batches(batch_size=250_000, columns=columns)
        for batch in batches:
            frame = batch.to_pandas()
            frame = frame[frame["ts_code"].astype(str).isin(symbols)].copy()
            if frame.empty:
                continue
            frame["trade_date"] = frame["trade_date"].map(
                lambda value: _day(value, label="fundamental.trade_date")
            )
            frame["availability_date"] = frame["availability_date"].map(
                lambda value: _day(value, label="fundamental.availability_date")
            )
            frame["fetched_at"] = frame["fetched_at"].map(
                lambda value: _available_at(
                    str(value),
                    label="fundamental.fetched_at",
                )
            )
            frame = frame[
                (frame["trade_date"] <= decision_session)
                & (frame["availability_date"] <= decision_session)
                & (frame["fetched_at"] <= cutoff)
            ]
            if frame.empty:
                continue
            frame["ts_code"] = frame["ts_code"].astype(str)
            frame = frame.sort_values(
                ["ts_code", "trade_date", "availability_date", "end_date"],
                kind="mergesort",
            ).drop_duplicates("ts_code", keep="last")
            latest = pd.concat([latest, frame], ignore_index=True)
            latest = latest.sort_values(
                ["ts_code", "trade_date", "availability_date", "end_date"],
                kind="mergesort",
            ).drop_duplicates("ts_code", keep="last")
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise _gap("FUNDAMENTAL_PARQUET_READ_FAILED", path.as_posix()) from exc
    if latest.empty:
        raise _gap("FUNDAMENTAL_PIT_INPUT_EMPTY", decision_session)
    for field_name in (
        "fin_roe",
        "fin_ocf_to_profit",
        "fin_debt_to_assets",
    ):
        latest[field_name] = pd.to_numeric(
            latest[field_name],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    return latest.reset_index(drop=True)


def _neutralizers(
    market: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    decision_session: str,
    available_at: str,
) -> pd.DataFrame:
    prices = market.pivot(
        index="trade_date",
        columns="symbol",
        values="adj_close",
    ).sort_index()
    returns = prices.pct_change(fill_method=None)
    market_return = returns.mean(axis=1, skipna=True)
    beta: dict[str, float | None] = {}
    observations: dict[str, int] = {}
    for symbol in universe["symbol"].tolist():
        if symbol not in returns.columns:
            beta[symbol] = None
            observations[symbol] = 0
            continue
        pair = (
            pd.concat(
                [returns[symbol], market_return],
                axis=1,
                keys=["asset", "market"],
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        pair = pair.tail(_BETA_WINDOW)
        count = len(pair)
        observations[symbol] = count
        if count < _BETA_MIN_OBSERVATIONS:
            beta[symbol] = None
            continue
        variance = float(pair["market"].var(ddof=1))
        if not math.isfinite(variance) or variance <= 1e-18:
            beta[symbol] = None
            continue
        covariance = float(pair["asset"].cov(pair["market"]))
        value = covariance / variance
        beta[symbol] = value if math.isfinite(value) else None
    latest_market = market.sort_values(
        ["symbol", "trade_date"],
        kind="mergesort",
    ).drop_duplicates("symbol", keep="last")
    latest_market = latest_market.set_index("symbol")
    rows: list[dict[str, Any]] = []
    for row in universe.itertuples(index=False):
        symbol = str(row.symbol)
        total_mv = (
            float(latest_market.at[symbol, "total_mv"])
            if symbol in latest_market.index and pd.notna(latest_market.at[symbol, "total_mv"])
            else math.nan
        )
        log_market_cap = (
            math.log(total_mv) if math.isfinite(total_mv) and total_mv > 0 else math.nan
        )
        source_market_session = (
            str(latest_market.at[symbol, "trade_date"]) if symbol in latest_market.index else None
        )
        rows.append(
            {
                "symbol": symbol,
                "trade_date": decision_session,
                "available_at": available_at,
                "industry": row.industry,
                "log_market_cap": log_market_cap,
                "beta_252d": beta[symbol],
                "beta_observations": observations[symbol],
                "source_market_session": source_market_session,
            }
        )
    result = pd.DataFrame(rows)
    result["beta_observations"] = result["beta_observations"].astype("int64")
    return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def _validate_source_inputs(
    root: Path,
    *,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    market_pointer_sha256: str,
    fundamental_pointer_sha256: str,
    factor_set_pointer_sha256: str,
    strategy_universe_path: str,
    strategy_universe_sha256: str,
    strategy_universe_manifest_path: str,
    strategy_universe_manifest_sha256: str,
) -> tuple[
    list[dict[str, str]],
    list[_BoundFile],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    str,
    str,
    int,
    int,
]:
    market_pointer, market_pointer_binding = _read_json(
        root,
        MARKET_POINTER,
        expected_sha256=market_pointer_sha256,
        label="canonical market pointer",
    )
    if (
        market_pointer.get("status") != "OK"
        or market_pointer.get("blockers") != []
        or not bool((market_pointer.get("coverage") or {}).get("complete"))
    ):
        raise _gap("MARKET_POINTER_NOT_READY", decision_session)
    market_session = _day(
        market_pointer.get("latest_complete_trade_date"),
        label="market.latest_complete_trade_date",
    )
    if market_session != decision_session:
        raise _gap(
            "LATEST_COMPLETE_TRADE_DATE_MISMATCH",
            f"expected={decision_session} observed={market_session}",
        )
    market_available = _before_cutoff(
        _available_at(
            market_pointer.get("updated_at"),
            label="market.updated_at",
        ),
        cutoff,
        label="market",
    )
    market_manifest, market_manifest_binding = _read_json(
        root,
        str(market_pointer.get("manifest_path")),
        label="canonical market manifest",
    )
    if (
        market_manifest.get("snapshot_id") != market_pointer.get("snapshot_id")
        or _day(
            market_manifest.get("latest_complete_trade_date"),
            label="market_manifest.latest_complete_trade_date",
        )
        != decision_session
        or market_manifest.get("status") != "OK"
        or market_manifest.get("blockers") != []
        or market_manifest.get("readback_validated") is not True
        or market_manifest.get("table_root") != market_pointer.get("table_root")
    ):
        raise _gap("MARKET_MANIFEST_BINDING_MISMATCH", decision_session)

    coverage = market_pointer["coverage"]
    pit_manifest, pit_manifest_binding = _read_json(
        root,
        str(coverage.get("pit_generation_manifest_path")),
        expected_sha256=str(coverage.get("pit_generation_manifest_sha256")),
        label="PIT membership manifest",
    )
    pit_available = _before_cutoff(
        _available_at(pit_manifest.get("written_at"), label="pit.written_at"),
        cutoff,
        label="PIT membership",
    )
    pit_binding = _bind_file(
        root,
        str(coverage.get("pit_membership_path")),
        expected_sha256=str(coverage.get("pit_membership_sha256")),
    )
    if (
        pit_manifest.get("canonical_sha256") != pit_binding.byte_sha256
        or Path(str(pit_manifest.get("canonical_path"))) != pit_binding.path
    ):
        raise _gap("PIT_MANIFEST_BINDING_MISMATCH", pit_binding.relative_path)
    pit = _read_pit_membership(
        pit_binding.path,
        decision_session=decision_session,
        cutoff=cutoff,
    )

    universe_manifest, universe_manifest_binding = _read_json(
        root,
        strategy_universe_manifest_path,
        expected_sha256=strategy_universe_manifest_sha256,
        label="strategy universe manifest",
    )
    strategy_available = _before_cutoff(
        _available_at(
            universe_manifest.get("generated_at"),
            label="strategy_universe.generated_at",
        ),
        cutoff,
        label="strategy universe",
    )
    universe_binding = _bind_file(
        root,
        strategy_universe_path,
        expected_sha256=strategy_universe_sha256,
    )
    if (
        _day(
            universe_manifest.get("analysis_trade_date"),
            label="strategy_universe.analysis_trade_date",
        )
        != decision_session
        or Path(str(universe_manifest.get("full_metrics_path"))) != universe_binding.path
        or not bool((universe_manifest.get("schema_validation") or {}).get("schema_valid"))
        or not bool((universe_manifest.get("data_coverage") or {}).get("data_coverage_valid"))
    ):
        raise _gap("STRATEGY_UNIVERSE_MANIFEST_MISMATCH", decision_session)
    try:
        strategy_universe = pq.read_table(universe_binding.path).to_pandas()
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise _gap(
            "STRATEGY_UNIVERSE_PARQUET_INVALID",
            universe_binding.relative_path,
        ) from exc
    if "symbol" not in strategy_universe.columns:
        raise _gap("STRATEGY_UNIVERSE_SCHEMA_GAP", "symbol")
    strategy_universe["symbol"] = strategy_universe["symbol"].astype(str)
    if strategy_universe["symbol"].duplicated().any():
        raise _gap("STRATEGY_UNIVERSE_DUPLICATE_SYMBOL", decision_session)
    strategy_symbol_count = len(strategy_universe)
    if strategy_symbol_count <= 0:
        raise _gap("STRATEGY_UNIVERSE_EMPTY", decision_session)

    fundamental_pointer, fundamental_pointer_binding = _read_json(
        root,
        FUNDAMENTAL_POINTER,
        expected_sha256=fundamental_pointer_sha256,
        label="canonical fundamental pointer",
    )
    if fundamental_pointer.get("status") != "OK":
        raise _gap("FUNDAMENTAL_POINTER_NOT_READY", decision_session)
    fundamental_manifest, fundamental_manifest_binding = _read_fundamental_manifest(
        root,
        _relative_to_pointer(
            FUNDAMENTAL_POINTER,
            fundamental_pointer.get("manifest_path"),
        ),
    )
    if (
        fundamental_manifest.get("generation_id") != fundamental_pointer.get("generation_id")
        or fundamental_manifest.get("status") != "OK"
    ):
        raise _gap("FUNDAMENTAL_MANIFEST_BINDING_MISMATCH", decision_session)
    daily_relative = _relative_to_pointer(
        FUNDAMENTAL_POINTER,
        (fundamental_pointer.get("tables") or {}).get("fundamental_daily"),
    )
    daily_row = (fundamental_manifest.get("tables") or {}).get("fundamental_daily")
    if type(daily_row) is not dict:
        raise _gap("FUNDAMENTAL_MANIFEST_SCHEMA_GAP", "fundamental_daily")
    declared_daily_sha = str(daily_row.get("sha256"))
    provenance_sha = str(
        (fundamental_pointer.get("primary_provenance") or {})
        .get("output_parquet_sha256", {})
        .get("fundamental_daily")
    )
    if declared_daily_sha != provenance_sha:
        raise _gap("FUNDAMENTAL_SHA_LINEAGE_MISMATCH", daily_relative)
    fundamental_binding = _bind_file(
        root,
        daily_relative,
        expected_sha256=declared_daily_sha,
    )

    factor_pointer_payload, factor_pointer_binding = _read_json(
        root,
        FACTOR_POINTER,
        expected_sha256=factor_set_pointer_sha256,
        label="research factor-set pointer",
    )
    try:
        factor_state = ResearchFactorSetStore(str(root)).read_current()
    except ResearchFactorSetError as exc:
        raise _gap("FACTOR_SET_READ_FAILED", str(exc)) from exc
    if factor_state.pointer_ref["byte_sha256"] != factor_pointer_binding.byte_sha256:
        raise _gap("FACTOR_SET_POINTER_REREAD_MISMATCH", decision_session)
    factor_set = factor_state.factor_set
    factor_available = _before_cutoff(
        _available_at(
            factor_pointer_payload.get("published_at"),
            label="factor_set.published_at",
        ),
        cutoff,
        label="factor set",
    )
    if (
        factor_set.get("strategy_id") != strategy_id
        or _day(
            factor_set.get("effective_from_session"),
            label="factor_set.effective_from_session",
        )
        > decision_session
    ):
        raise _gap("FACTOR_SET_NOT_EFFECTIVE", decision_session)
    factor_set_binding = _bind_file(
        root,
        factor_state.factor_set_ref["relative_path"],
        expected_sha256=factor_state.factor_set_ref["byte_sha256"],
    )

    strategy_symbols = set(strategy_universe["symbol"].tolist())
    pit_symbols = set(pit["symbol"].tolist())
    included_symbols = strategy_symbols & pit_symbols
    if not included_symbols:
        raise _gap("STRATEGY_PIT_INTERSECTION_EMPTY", decision_session)
    pit_lookup = pit.set_index("symbol")
    strategy_lookup = strategy_universe.set_index("symbol")
    universe_rows: list[dict[str, Any]] = []
    snapshot_available = max(
        market_available,
        pit_available,
        strategy_available,
        factor_available,
    )
    for symbol in sorted(included_symbols):
        source_row = strategy_lookup.loc[symbol]
        universe_rows.append(
            {
                "symbol": symbol,
                "trade_date": decision_session,
                "available_at": snapshot_available,
                "industry": pit_lookup.at[symbol, "industry"],
                "strategy_name": (
                    source_row.get("name") if isinstance(source_row, pd.Series) else None
                ),
                "strategy_category": (
                    source_row.get("category") if isinstance(source_row, pd.Series) else None
                ),
            }
        )
    universe = pd.DataFrame(universe_rows)
    market, market_bindings = _read_market(
        root,
        market_pointer.get("table_root"),
        symbols=included_symbols,
        decision_session=decision_session,
    )
    fundamentals = _read_latest_fundamentals(
        fundamental_binding.path,
        symbols=included_symbols,
        decision_session=decision_session,
        cutoff=cutoff,
    )
    fundamental_available = _before_cutoff(
        max(fundamentals["fetched_at"].astype(str).tolist()),
        cutoff,
        label="fundamental",
    )
    snapshot_available = max(snapshot_available, fundamental_available)
    universe["available_at"] = snapshot_available

    source_refs: list[dict[str, str]] = [
        _source_ref(
            factor_pointer_binding,
            role="factor_set_pointer",
            as_of=decision_session,
            available_at=factor_available,
            media_type="application/json",
        ),
        _source_ref(
            factor_set_binding,
            role="factor_set",
            as_of=str(factor_set["effective_from_session"]),
            available_at=factor_available,
            media_type="application/json",
        ),
        _source_ref(
            fundamental_manifest_binding,
            role="fundamental_manifest",
            as_of=max(fundamentals["trade_date"].astype(str).tolist()),
            available_at=fundamental_available,
            media_type="application/json",
        ),
        _source_ref(
            fundamental_pointer_binding,
            role="fundamental_pointer",
            as_of=max(fundamentals["trade_date"].astype(str).tolist()),
            available_at=fundamental_available,
            media_type="application/json",
        ),
        _source_ref(
            fundamental_binding,
            role="fundamental_daily",
            as_of=max(fundamentals["trade_date"].astype(str).tolist()),
            available_at=fundamental_available,
            media_type="application/vnd.apache.parquet",
        ),
        _source_ref(
            market_manifest_binding,
            role="market_manifest",
            as_of=decision_session,
            available_at=market_available,
            media_type="application/json",
        ),
        _source_ref(
            market_pointer_binding,
            role="market_pointer",
            as_of=decision_session,
            available_at=market_available,
            media_type="application/json",
        ),
        _source_ref(
            pit_manifest_binding,
            role="pit_membership_manifest",
            as_of=decision_session,
            available_at=pit_available,
            media_type="application/json",
        ),
        _source_ref(
            pit_binding,
            role="pit_membership",
            as_of=decision_session,
            available_at=pit_available,
            media_type="application/vnd.apache.parquet",
        ),
        _source_ref(
            universe_manifest_binding,
            role="strategy_universe_manifest",
            as_of=decision_session,
            available_at=strategy_available,
            media_type="application/json",
        ),
        _source_ref(
            universe_binding,
            role="strategy_universe",
            as_of=decision_session,
            available_at=strategy_available,
            media_type="application/vnd.apache.parquet",
        ),
    ]
    for index, binding in enumerate(market_bindings):
        source_refs.append(
            _source_ref(
                binding,
                role=f"market_bar_part_{index:03d}",
                as_of=decision_session,
                available_at=market_available,
                media_type="application/vnd.apache.parquet",
            )
        )
    all_bindings = [
        market_pointer_binding,
        market_manifest_binding,
        pit_manifest_binding,
        pit_binding,
        universe_manifest_binding,
        universe_binding,
        fundamental_pointer_binding,
        fundamental_manifest_binding,
        fundamental_binding,
        factor_pointer_binding,
        factor_set_binding,
        *market_bindings,
    ]
    return (
        _sorted_source_refs(source_refs),
        all_bindings,
        universe,
        market,
        fundamentals,
        dict(factor_state.factor_set_ref),
        snapshot_available,
        fundamental_available,
        strategy_symbol_count,
        len(pit),
    )


def build_source_snapshot(
    workspace_root: str,
    *,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    market_pointer_sha256: str,
    fundamental_pointer_sha256: str,
    factor_set_pointer_sha256: str,
    strategy_universe_path: str,
    strategy_universe_sha256: str,
    strategy_universe_manifest_path: str,
    strategy_universe_manifest_sha256: str,
) -> dict[str, Any]:
    """Build one immutable, schema-validated Forward Evidence source snapshot."""

    root = _workspace_root(workspace_root)
    strategy = _identifier(strategy_id, label="strategy_id")
    session = _day(decision_session, label="decision_session")
    exact_cutoff = _cutoff(cutoff, label="cutoff")
    if session > exact_cutoff[:10]:
        raise _gap(
            "DECISION_SESSION_AFTER_CUTOFF",
            f"session={session} cutoff={exact_cutoff}",
        )
    (
        source_refs,
        source_bindings,
        universe,
        market,
        fundamentals,
        factor_set_ref,
        available_at,
        fundamental_available,
        strategy_symbol_count,
        pit_active_symbol_count,
    ) = _validate_source_inputs(
        root,
        strategy_id=strategy,
        decision_session=session,
        cutoff=exact_cutoff,
        market_pointer_sha256=market_pointer_sha256,
        fundamental_pointer_sha256=fundamental_pointer_sha256,
        factor_set_pointer_sha256=factor_set_pointer_sha256,
        strategy_universe_path=strategy_universe_path,
        strategy_universe_sha256=strategy_universe_sha256,
        strategy_universe_manifest_path=strategy_universe_manifest_path,
        strategy_universe_manifest_sha256=strategy_universe_manifest_sha256,
    )
    source_set_sha = hashlib.sha256(canonical_bytes(source_refs)).hexdigest()
    snapshot_root = SNAPSHOT_ROOT / session

    universe_slice = _materialize_slice(
        universe,
        artifact_kind="UNIVERSE",
        field_name="universe",
        value_fields=("symbol",),
        relative_parquet_path=str(snapshot_root / "universe.parquet"),
        source_refs=source_refs,
        strategy_id=strategy,
        decision_session=session,
        cutoff=exact_cutoff,
        available_at=available_at,
    )
    market_with_available = market.copy()
    market_with_available.insert(2, "available_at", available_at)
    factor_slices: list[_MaterializedSlice] = []
    for field_name in ("adj_close", "total_mv"):
        factor_slices.append(
            _materialize_slice(
                market_with_available[
                    ["symbol", "trade_date", "available_at", field_name]
                ].reset_index(drop=True),
                artifact_kind="FACTOR_INPUT",
                field_name=field_name,
                value_fields=(field_name,),
                relative_parquet_path=str(
                    snapshot_root / "factor_inputs" / f"{field_name}.parquet"
                ),
                source_refs=source_refs,
                strategy_id=strategy,
                decision_session=session,
                cutoff=exact_cutoff,
                available_at=available_at,
            )
        )

    latest_lookup = fundamentals.set_index("ts_code")
    for field_name in (
        "fin_roe",
        "fin_ocf_to_profit",
        "fin_debt_to_assets",
    ):
        rows: list[dict[str, Any]] = []
        for symbol in universe["symbol"].tolist():
            if symbol in latest_lookup.index:
                source_row = latest_lookup.loc[symbol]
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": session,
                        "available_at": fundamental_available,
                        "source_trade_date": str(source_row["trade_date"]),
                        field_name: source_row[field_name],
                    }
                )
            else:
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": session,
                        "available_at": fundamental_available,
                        "source_trade_date": None,
                        field_name: math.nan,
                    }
                )
        frame = (
            pd.DataFrame(rows)
            .sort_values(
                "symbol",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        factor_slices.append(
            _materialize_slice(
                frame,
                artifact_kind="FACTOR_INPUT",
                field_name=field_name,
                value_fields=(field_name,),
                relative_parquet_path=str(
                    snapshot_root / "factor_inputs" / f"{field_name}.parquet"
                ),
                source_refs=source_refs,
                strategy_id=strategy,
                decision_session=session,
                cutoff=exact_cutoff,
                available_at=available_at,
            )
        )
    factor_slices.sort(key=lambda value: value.field_name)

    neutralizer_frame = _neutralizers(
        market,
        universe,
        decision_session=session,
        available_at=available_at,
    )
    neutralizer_slice = _materialize_slice(
        neutralizer_frame,
        artifact_kind="NEUTRALIZER",
        field_name="neutralizer",
        value_fields=NEUTRALIZER_FIELDS,
        relative_parquet_path=str(snapshot_root / "neutralizer.parquet"),
        source_refs=source_refs,
        strategy_id=strategy,
        decision_session=session,
        cutoff=exact_cutoff,
        available_at=available_at,
    )

    bundle = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": available_at,
            "bundle_id": f"forward-factor-inputs-{session}",
            "canary_evidence_eligible": False,
            "cutoff": exact_cutoff,
            "decision_session": session,
            "factor_set_ref": factor_set_ref,
            "factor_slices": [
                {
                    "available_at": value.manifest["available_at"],
                    "field_name": value.field_name,
                    "non_null_count": value.non_null_count,
                    "row_count": value.row_count,
                    "slice_ref": value.manifest_ref,
                }
                for value in factor_slices
            ],
            "formal_activation_eligible": False,
            "neutralizer_fields": list(NEUTRALIZER_FIELDS),
            "neutralizer_ref": neutralizer_slice.manifest_ref,
            "performance_evidence_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "required_fields": list(FACTOR_FIELDS),
            "schema_validation_status": "VALIDATED",
            "shadow_only": True,
            "source_set_sha256": source_set_sha,
            "strategy_id": strategy,
            "universe_ref": universe_slice.manifest_ref,
            "version": FACTOR_BUNDLE_VERSION,
        }
    )
    validate_artifact(bundle)
    bundle_path = str(snapshot_root / "factor_input_bundle.json")
    bundle_raw = canonical_resource_bytes(bundle)
    bundle_ref = _artifact_ref(
        bundle,
        bundle_raw,
        identity_field="bundle_id",
        relative_path=bundle_path,
    )

    limitations: list[str] = []
    excluded = strategy_symbol_count - len(universe)
    if excluded:
        limitations.append(f"{excluded} strategy symbols absent from active PIT membership")
    neutralizer_missing = len(universe) - neutralizer_slice.non_null_count
    if neutralizer_missing:
        limitations.append(f"{neutralizer_missing} universe symbols have incomplete neutralizers")
    locator = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": available_at,
            "canary_evidence_eligible": False,
            "counts": {
                "excluded_symbols": excluded,
                "pit_active_symbols": pit_active_symbol_count,
                "strategy_source_symbols": strategy_symbol_count,
                "universe_symbols": len(universe),
            },
            "cutoff": exact_cutoff,
            "decision_session": session,
            "factor_input_bundle_ref": bundle_ref,
            "factor_slice_refs": [value.manifest_ref for value in factor_slices],
            "formal_activation_eligible": False,
            "limitations": sorted(limitations),
            "locator_id": f"forward-source-locator-{session}",
            "neutralizer_ref": neutralizer_slice.manifest_ref,
            "performance_evidence_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "schema_validation_status": "VALIDATED",
            "shadow_only": True,
            "snapshot_id": f"forward-source-{session}",
            "source_refs": source_refs,
            "source_set_sha256": source_set_sha,
            "status": "READY",
            "strategy_id": strategy,
            "universe_ref": universe_slice.manifest_ref,
            "version": SOURCE_LOCATOR_VERSION,
        }
    )
    validate_artifact(locator)
    locator_path = str(snapshot_root / "source_locator.json")
    locator_raw = canonical_resource_bytes(locator)

    for binding in source_bindings:
        _assert_unchanged(binding)
    store = SourceStore(
        str(root),
        max_read_bytes=512 * 1024 * 1024,
    )
    publications: list[tuple[str, bytes]] = [
        (universe_slice.parquet_path, universe_slice.parquet_raw),
        (universe_slice.manifest_path, universe_slice.manifest_raw),
    ]
    for materialized in factor_slices:
        publications.extend(
            [
                (materialized.parquet_path, materialized.parquet_raw),
                (materialized.manifest_path, materialized.manifest_raw),
            ]
        )
    publications.extend(
        [
            (neutralizer_slice.parquet_path, neutralizer_slice.parquet_raw),
            (neutralizer_slice.manifest_path, neutralizer_slice.manifest_raw),
            (bundle_path, bundle_raw),
        ]
    )
    created = 0
    reused = 0
    try:
        for relative_path, raw in publications:
            result = store.write_exact_once(relative_path, raw)
            created += int(result.created)
            reused += int(not result.created)
            if store.read(relative_path, result.byte_sha256) != raw:
                raise SourceSnapshotError(f"immutable readback mismatch: {relative_path}")
        locator_result = store.write_exact_once(locator_path, locator_raw)
        created += int(locator_result.created)
        reused += int(not locator_result.created)
        readback = store.read(locator_path, locator_result.byte_sha256)
    except SourceExactOnceConflict as exc:
        raise _gap(
            "IMMUTABLE_SNAPSHOT_CONFLICT",
            str(snapshot_root),
        ) from exc
    except SourceStorageError as exc:
        raise SourceSnapshotError("source snapshot publication failed") from exc
    if readback != locator_raw:
        raise SourceSnapshotError("source locator readback mismatch")
    validate_artifact(strict_json_loads(readback, label="source locator readback"))
    for binding in source_bindings:
        _assert_unchanged(binding)
    return {
        "authority": dict(NO_AUTHORITY),
        "available_at": available_at,
        "created_artifacts": created,
        "cutoff": exact_cutoff,
        "decision_session": session,
        "factor_input_bundle_ref": bundle_ref,
        "formal_activation_eligible": False,
        "limitations": sorted(limitations),
        "locator_ref": _artifact_ref(
            locator,
            locator_raw,
            identity_field="locator_id",
            relative_path=locator_path,
        ),
        "research_runtime_default": False,
        "reused_artifacts": reused,
        "snapshot_root": str(snapshot_root),
        "status": "READY",
        "strategy_id": strategy,
    }


def gap_payload(exc: SourceSnapshotGap) -> dict[str, Any]:
    """Return the only fail-closed CLI state for current canonical gaps."""

    return {
        "authority": dict(NO_AUTHORITY),
        "blockers": list(exc.blockers),
        "formal_activation_eligible": False,
        "research_runtime_default": False,
        "status": "TRUE_CURRENT_CANONICAL_INPUT_GAP",
    }


__all__ = [
    "FACTOR_FIELDS",
    "NEUTRALIZER_FIELDS",
    "SourceSnapshotError",
    "SourceSnapshotGap",
    "build_source_snapshot",
    "gap_payload",
]
