"""Fail-closed CN Macro compatibility mart persistence.

Offline compatibility rows are staged as immutable, non-production candidates.
Production readers accept only a generation bound by the market-level strict
Parquet catalog, its generation manifest, and both declared SHA-256 digests.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar

import pandas as pd

from quant_investor.market.branch_readiness import SOURCE_OFFLINE
from quant_investor.market.branch_readiness import (
    SOURCE_PUBLIC_FALLBACK,
    SOURCE_TUSHARE,
)


DEFAULT_MACRO_ROOT = Path("data/parquet/cn/macro_daily")
DEFAULT_RAW_SNAPSHOT_ROOT = Path("data/cn_market_full/_snapshots/macro")
MACRO_FIELDS = (
    "macro_score",
    "liquidity_score",
    "volatility_percentile",
    "policy_signal",
)
CANDIDATE_MANIFEST_SCHEMA = "cn-macro-mart-candidate.v14"
CANONICAL_MANIFEST_SCHEMA = "cn-macro-mart.v14"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_INPUT_FIELDS = {
    "trade_date",
    *MACRO_FIELDS,
    "source",
    "source_priority",
    "pit_status",
    "fetched_at",
}
_Parsed = TypeVar("_Parsed")
_SOURCE_PRIORITY_BY_SOURCE = {
    SOURCE_TUSHARE: SOURCE_TUSHARE,
    SOURCE_PUBLIC_FALLBACK: SOURCE_PUBLIC_FALLBACK,
    SOURCE_OFFLINE: SOURCE_OFFLINE,
}


class MacroMartPromotionError(RuntimeError):
    """Raised when Macro candidate or canonical lineage is unsafe."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
    if pd.isna(parsed):
        raise MacroMartPromotionError("macro_as_of_missing_or_invalid")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_sha256(value: Any, *, blocker: str) -> str:
    text = str(value or "").strip().lower()
    if _HEX_SHA256.fullmatch(text) is None:
        raise MacroMartPromotionError(blocker)
    return text


def _assert_safe_write_root(root: Path) -> Path:
    if root.exists() and root.is_symlink():
        raise MacroMartPromotionError("macro_root_symlink_rejected")
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve()
    if not resolved.is_dir():
        raise MacroMartPromotionError("macro_root_not_directory")
    return resolved


def _strict_read_root(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise MacroMartPromotionError("macro_root_missing_or_unsafe")
    cursor = Path(raw.anchor) if raw.is_absolute() else Path.cwd()
    parts = raw.parts[1:] if raw.is_absolute() else raw.parts
    for part in parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise MacroMartPromotionError(
                "macro_root_missing_or_unsafe"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise MacroMartPromotionError("macro_root_symlink_rejected")
        if not stat.S_ISDIR(metadata.st_mode):
            raise MacroMartPromotionError("macro_root_missing_or_unsafe")
    return cursor.resolve(strict=True)


def _safe_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value or not _SAFE_RUN_ID.fullmatch(value) or value in {".", ".."}:
        raise MacroMartPromotionError("macro_run_id_unsafe")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise MacroMartPromotionError("macro_output_symlink_rejected")
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@contextmanager
def _staging_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".staging.lock"
    if lock_path.exists() and lock_path.is_symlink():
        raise MacroMartPromotionError("macro_lock_symlink_rejected")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validated_offline_frame(
    indicators: Mapping[str, Any],
    *,
    as_of: str,
) -> pd.DataFrame:
    row = dict(indicators)
    unknown = sorted(set(row) - _ALLOWED_INPUT_FIELDS)
    if unknown:
        raise MacroMartPromotionError(
            f"macro_unknown_input_fields:{','.join(unknown)}"
        )
    if not row:
        raise MacroMartPromotionError("macro_empty_candidate")
    missing = [
        field for field in MACRO_FIELDS if field not in row or pd.isna(row[field])
    ]
    if missing:
        raise MacroMartPromotionError(
            f"macro_required_fields_missing:{','.join(missing)}"
        )
    candidate_date = _date_text(as_of or row.get("trade_date"))
    if row.get("trade_date") and _date_text(row["trade_date"]) != candidate_date:
        raise MacroMartPromotionError("macro_trade_date_as_of_mismatch")
    row["trade_date"] = candidate_date
    # Local compatibility input is never provider evidence. Caller-reported
    # provenance is deliberately overwritten instead of trusted.
    row["source"] = SOURCE_OFFLINE
    row["source_priority"] = SOURCE_OFFLINE
    row["pit_status"] = "manual_offline_snapshot"
    row["fetched_at"] = _now_utc()
    frame = pd.DataFrame([row])
    for field in ("macro_score", "liquidity_score", "volatility_percentile"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if frame[list(MACRO_FIELDS)].isna().any().any():
        raise MacroMartPromotionError("macro_required_fields_invalid")
    policy_signal = str(frame.iloc[0]["policy_signal"] or "").strip()
    if not policy_signal:
        raise MacroMartPromotionError("macro_policy_signal_empty")
    frame.loc[:, "policy_signal"] = policy_signal
    if not frame["macro_score"].between(-1.0, 1.0).all():
        raise MacroMartPromotionError("macro_score_out_of_range")
    if not frame["liquidity_score"].between(-1.0, 1.0).all():
        raise MacroMartPromotionError("macro_liquidity_score_out_of_range")
    if not frame["volatility_percentile"].between(0.0, 100.0).all():
        raise MacroMartPromotionError("macro_volatility_percentile_out_of_range")
    return frame


def _resolve_catalog_member(
    market_root: Path,
    raw_path: Any,
    *,
    blocker: str,
) -> Path:
    relative = Path(str(raw_path or "").strip())
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise MacroMartPromotionError(blocker)
    cursor = market_root
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        try:
            status = os.lstat(cursor)
        except OSError as exc:
            raise MacroMartPromotionError(blocker) from exc
        if stat.S_ISLNK(status.st_mode):
            raise MacroMartPromotionError(blocker)
        is_final = index == len(relative.parts) - 1
        if is_final and not stat.S_ISREG(status.st_mode):
            raise MacroMartPromotionError(blocker)
        if not is_final and not stat.S_ISDIR(status.st_mode):
            raise MacroMartPromotionError(blocker)
    candidate = market_root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MacroMartPromotionError(blocker) from exc
    market_resolved = market_root.resolve(strict=True)
    if market_resolved not in resolved.parents:
        raise MacroMartPromotionError(blocker)
    return candidate


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_verified_member(
    path: Path,
    *,
    trust_root: Path,
    expected_sha256: str | None,
    hash_blocker: str,
    changed_blocker: str,
    unreadable_blocker: str,
    parser: Callable[[bytes], _Parsed],
) -> _Parsed:
    """Hash and parse the same bytes while pinning every path component."""

    try:
        relative = path.relative_to(trust_root)
    except ValueError as exc:
        raise MacroMartPromotionError(unreadable_blocker) from exc
    if not relative.parts:
        raise MacroMartPromotionError(unreadable_blocker)

    descriptors: list[int] = []
    component_states: list[tuple[Path, tuple[int, ...], int]] = []
    cursor = trust_root
    opened_any = False
    try:
        root_before = os.lstat(trust_root)
        if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(
            root_before.st_mode
        ):
            raise MacroMartPromotionError(unreadable_blocker)
        root_descriptor = os.open(
            trust_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        descriptors.append(root_descriptor)
        opened_any = True
        root_opened = os.fstat(root_descriptor)
        if _stat_signature(root_before) != _stat_signature(root_opened):
            raise MacroMartPromotionError(changed_blocker)
        component_states.append(
            (trust_root, _stat_signature(root_before), root_descriptor)
        )

        parent_descriptor = root_descriptor
        for index, part in enumerate(relative.parts):
            cursor = cursor / part
            before = os.lstat(cursor)
            is_final = index == len(relative.parts) - 1
            if stat.S_ISLNK(before.st_mode):
                raise MacroMartPromotionError(unreadable_blocker)
            if is_final and not stat.S_ISREG(before.st_mode):
                raise MacroMartPromotionError(unreadable_blocker)
            if not is_final and not stat.S_ISDIR(before.st_mode):
                raise MacroMartPromotionError(unreadable_blocker)
            flags = os.O_RDONLY | os.O_NOFOLLOW
            if not is_final:
                flags |= os.O_DIRECTORY
            descriptor = os.open(part, flags, dir_fd=parent_descriptor)
            descriptors.append(descriptor)
            opened = os.fstat(descriptor)
            if _stat_signature(before) != _stat_signature(opened):
                raise MacroMartPromotionError(changed_blocker)
            component_states.append(
                (cursor, _stat_signature(before), descriptor)
            )
            parent_descriptor = descriptor

        file_descriptor = descriptors[-1]
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)

        for component, before_signature, descriptor in component_states:
            if (
                _stat_signature(os.fstat(descriptor)) != before_signature
                or _stat_signature(os.lstat(component)) != before_signature
            ):
                raise MacroMartPromotionError(changed_blocker)
        if (
            expected_sha256 is not None
            and hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise MacroMartPromotionError(hash_blocker)
        try:
            parsed = parser(payload)
        except MacroMartPromotionError:
            raise
        except Exception as exc:
            raise MacroMartPromotionError(unreadable_blocker) from exc
        for component, before_signature, descriptor in component_states:
            if (
                _stat_signature(os.fstat(descriptor)) != before_signature
                or _stat_signature(os.lstat(component)) != before_signature
            ):
                raise MacroMartPromotionError(changed_blocker)
        return parsed
    except MacroMartPromotionError:
        raise
    except OSError as exc:
        blocker = changed_blocker if opened_any else unreadable_blocker
        raise MacroMartPromotionError(blocker) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _parse_json_object(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("json_object_required")
    return dict(value)


def _validate_canonical_frame(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    if frame.empty or set(frame.columns) != _ALLOWED_INPUT_FIELDS:
        raise MacroMartPromotionError("macro_catalog_table_contract_invalid")

    numeric = {
        field: pd.to_numeric(frame[field], errors="coerce")
        for field in (
            "macro_score",
            "liquidity_score",
            "volatility_percentile",
        )
    }
    if any(series.isna().any() for series in numeric.values()):
        raise MacroMartPromotionError("macro_required_fields_invalid")
    if not numeric["macro_score"].between(-1.0, 1.0).all():
        raise MacroMartPromotionError("macro_score_out_of_range")
    if not numeric["liquidity_score"].between(-1.0, 1.0).all():
        raise MacroMartPromotionError(
            "macro_liquidity_score_out_of_range"
        )
    if not numeric["volatility_percentile"].between(0.0, 100.0).all():
        raise MacroMartPromotionError(
            "macro_volatility_percentile_out_of_range"
        )

    policy = frame["policy_signal"].fillna("").astype(str).str.strip()
    if policy.eq("").any():
        raise MacroMartPromotionError("macro_policy_signal_empty")
    trade_dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    if trade_dates.isna().any():
        raise MacroMartPromotionError("macro_trade_date_invalid")
    normalized_dates = trade_dates.dt.strftime("%Y-%m-%d")
    if normalized_dates.duplicated().any():
        raise MacroMartPromotionError("macro_trade_date_duplicate")

    if (
        str(manifest.get("table") or "") != "macro_daily"
        or str(manifest.get("provider_status") or "")
        != "verified_provider_snapshot"
    ):
        raise MacroMartPromotionError(
            "macro_generation_manifest_lineage_invalid"
        )
    manifest_source = str(manifest.get("source") or "").strip()
    manifest_priority = str(
        manifest.get("source_priority") or ""
    ).strip()
    manifest_pit = str(manifest.get("pit_status") or "").strip()
    if not manifest_source or not manifest_priority:
        raise MacroMartPromotionError(
            "macro_generation_manifest_lineage_invalid"
        )
    expected_priority = _SOURCE_PRIORITY_BY_SOURCE.get(manifest_source)
    if expected_priority is None or manifest_priority != expected_priority:
        raise MacroMartPromotionError("macro_source_priority_mismatch")
    sources = frame["source"].fillna("").astype(str).str.strip()
    priorities = (
        frame["source_priority"].fillna("").astype(str).str.strip()
    )
    if (
        sources.eq("").any()
        or priorities.eq("").any()
        or not sources.eq(manifest_source).all()
        or not priorities.eq(manifest_priority).all()
    ):
        raise MacroMartPromotionError("macro_source_lineage_mismatch")
    pit_statuses = frame["pit_status"].fillna("").astype(str).str.strip()
    if (
        manifest_pit != "market_point_in_time"
        or not pit_statuses.eq(manifest_pit).all()
    ):
        raise MacroMartPromotionError("macro_pit_lineage_mismatch")
    fetched_at = pd.to_datetime(
        frame["fetched_at"],
        errors="coerce",
        utc=True,
    )
    if fetched_at.isna().any():
        raise MacroMartPromotionError("macro_fetched_at_invalid")
    try:
        manifest_as_of = _date_text(manifest.get("as_of"))
    except MacroMartPromotionError as exc:
        raise MacroMartPromotionError(
            "macro_generation_manifest_as_of_mismatch"
        ) from exc
    if manifest_as_of != str(normalized_dates.max()):
        raise MacroMartPromotionError(
            "macro_generation_manifest_as_of_mismatch"
        )
    if "row_count" in manifest:
        try:
            row_count = int(manifest["row_count"])
        except (TypeError, ValueError) as exc:
            raise MacroMartPromotionError(
                "macro_generation_manifest_row_count_mismatch"
            ) from exc
        if row_count != len(frame):
            raise MacroMartPromotionError(
                "macro_generation_manifest_row_count_mismatch"
            )


def read_macro_mart(
    *,
    data_root: str | Path = DEFAULT_MACRO_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the sole catalog-bound Macro generation and verify its lineage."""

    root = _strict_read_root(data_root)
    market_root = root.parent
    catalog_path = market_root / "_catalog.json"
    if not catalog_path.exists():
        raise MacroMartPromotionError("macro_catalog_missing")
    catalog = _read_verified_member(
        catalog_path,
        trust_root=market_root,
        expected_sha256=None,
        hash_blocker="macro_catalog_hash_mismatch",
        changed_blocker="macro_catalog_changed_during_read",
        unreadable_blocker="macro_catalog_invalid",
        parser=_parse_json_object,
    )
    if catalog.get("schema_version") != "strict-parquet-catalog.v1":
        raise MacroMartPromotionError("macro_catalog_schema_invalid")
    tables = catalog.get("tables")
    if not isinstance(tables, Mapping):
        raise MacroMartPromotionError("macro_catalog_tables_invalid")
    entry = tables.get("macro_daily")
    if not isinstance(entry, Mapping):
        raise MacroMartPromotionError("macro_catalog_entry_missing")
    entry = dict(entry)
    generation_id = str(entry.get("generation_id") or "").strip()
    if (
        not generation_id
        or generation_id in {".", ".."}
        or not _SAFE_RUN_ID.fullmatch(generation_id)
    ):
        raise MacroMartPromotionError("macro_catalog_generation_invalid")
    table_path = _resolve_catalog_member(
        market_root,
        entry.get("path"),
        blocker="macro_catalog_table_path_invalid",
    )
    generation_manifest_path = _resolve_catalog_member(
        market_root,
        entry.get("generation_manifest"),
        blocker="macro_catalog_manifest_path_invalid",
    )
    expected_generation_root = root / "_generations" / generation_id
    if (
        table_path.parent != expected_generation_root
        or generation_manifest_path.parent != expected_generation_root
    ):
        raise MacroMartPromotionError("macro_catalog_generation_path_mismatch")
    table_sha = _assert_sha256(
        entry.get("parquet_sha256"),
        blocker="macro_catalog_table_hash_invalid",
    )
    manifest_sha = _assert_sha256(
        entry.get("generation_manifest_sha256"),
        blocker="macro_catalog_manifest_hash_invalid",
    )
    generation_manifest = _read_verified_member(
        generation_manifest_path,
        trust_root=market_root,
        expected_sha256=manifest_sha,
        hash_blocker="macro_catalog_manifest_hash_mismatch",
        changed_blocker="macro_catalog_manifest_changed_during_read",
        unreadable_blocker="macro_generation_manifest_invalid",
        parser=_parse_json_object,
    )
    if generation_manifest.get("schema_version") != CANONICAL_MANIFEST_SCHEMA:
        raise MacroMartPromotionError("macro_generation_manifest_schema_invalid")
    if str(generation_manifest.get("generation_id") or "") != generation_id:
        raise MacroMartPromotionError("macro_generation_manifest_id_mismatch")
    if generation_manifest.get("production_eligible") is not True:
        raise MacroMartPromotionError("macro_generation_not_production_eligible")
    if str(generation_manifest.get("parquet_sha256") or "") != table_sha:
        raise MacroMartPromotionError("macro_generation_manifest_table_hash_mismatch")
    manifest_table_path = Path(
        str(generation_manifest.get("table_path") or "").strip()
    )
    if (
        manifest_table_path.is_absolute()
        or manifest_table_path.parts != (table_path.name,)
    ):
        raise MacroMartPromotionError("macro_generation_manifest_table_path_mismatch")
    frame = _read_verified_member(
        table_path,
        trust_root=market_root,
        expected_sha256=table_sha,
        hash_blocker="macro_catalog_table_hash_mismatch",
        changed_blocker="macro_catalog_table_changed_during_read",
        unreadable_blocker="macro_catalog_table_unreadable",
        parser=lambda payload: pd.read_parquet(io.BytesIO(payload)),
    )
    _validate_canonical_frame(frame, generation_manifest)
    manifest = {
        **generation_manifest,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_schema_version": str(catalog.get("schema_version")),
        "resolved_table_path": str(table_path),
        "resolved_generation_manifest": str(generation_manifest_path),
        "generation_manifest_sha256": manifest_sha,
    }
    return frame, manifest


def write_macro_mart(
    indicators: Mapping[str, Any] | None = None,
    *,
    as_of: str = "",
    data_root: str | Path = DEFAULT_MACRO_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    run_id: str = "",
    provider_status: str = "offline_input",
    source_priority: str = SOURCE_OFFLINE,
    provider_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage one immutable offline candidate without changing canonical state."""

    del provider_status, source_priority
    run_id = _safe_run_id(
        run_id or f"cn_macro_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    root = _assert_safe_write_root(Path(data_root).expanduser())
    raw_root = _assert_safe_write_root(Path(raw_snapshot_root).expanduser())
    frame = _validated_offline_frame(dict(indicators or {}), as_of=as_of)
    with _staging_lock(root):
        generation = root / "_candidates" / run_id
        if generation.exists():
            raise MacroMartPromotionError("macro_candidate_generation_exists")
        generation.mkdir(parents=True, mode=0o700)
        table = generation / "part.parquet"
        frame.to_parquet(table, index=False)
        os.chmod(table, 0o600)
        readback = pd.read_parquet(table)
        if len(readback) != 1 or any(
            field not in readback.columns for field in MACRO_FIELDS
        ):
            raise MacroMartPromotionError("macro_candidate_readback_failed")
        raw_path = raw_root / f"{run_id}.csv"
        if raw_path.exists():
            raise MacroMartPromotionError("macro_raw_snapshot_already_exists")
        _atomic_write_bytes(raw_path, frame.to_csv(index=False).encode("utf-8"))
        manifest = {
            "schema_version": CANDIDATE_MANIFEST_SCHEMA,
            "run_id": run_id,
            "generation_id": run_id,
            "table": "macro_daily",
            "table_path": str(table.relative_to(root)),
            "parquet_sha256": _sha256(table),
            "raw_snapshot": str(raw_path),
            "raw_snapshot_sha256": _sha256(raw_path),
            "as_of": str(frame.iloc[0]["trade_date"]),
            "source": SOURCE_OFFLINE,
            "source_priority": SOURCE_OFFLINE,
            "provider_status": "offline_input",
            "pit_status": "manual_offline_snapshot",
            "provider_manifest": dict(provider_manifest or {}),
            "production_eligible": False,
            "applied": False,
            "staged_at": _now_utc(),
        }
        manifest_path = generation / "manifest.json"
        _atomic_write_bytes(
            manifest_path,
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8"),
        )
        manifest["generation_manifest"] = str(manifest_path)
        manifest["generation_manifest_sha256"] = _sha256(manifest_path)
        _fsync_directory(generation)
    return manifest


def run_cn_macro_maintenance(
    *,
    indicators: Mapping[str, Any] | None = None,
    as_of: str = "",
    data_root: str | Path = DEFAULT_MACRO_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    allow_live: bool = False,
    allow_public_fallback: bool = False,
    run_id: str = "",
) -> dict[str, Any]:
    """Stage local input or fail closed; provider access is not implemented."""

    if not indicators:
        if allow_live:
            provider_status = "blocked_live_provider_not_implemented"
        elif allow_public_fallback:
            provider_status = "blocked_public_fallback_not_implemented"
        else:
            provider_status = "no_update_no_input"
        return {
            "run_id": run_id or "",
            "provider_status": provider_status,
            "status": "blocked",
            "promoted": False,
            "manifest": {},
        }
    manifest = write_macro_mart(
        indicators,
        as_of=as_of,
        data_root=data_root,
        raw_snapshot_root=raw_snapshot_root,
        run_id=run_id,
        provider_manifest={
            "allow_live": bool(allow_live),
            "allow_public_fallback": bool(allow_public_fallback),
        },
    )
    return {
        "run_id": manifest["run_id"],
        "provider_status": "offline_input",
        "status": "staged",
        "promoted": False,
        "manifest": manifest,
    }


__all__ = [
    "CANONICAL_MANIFEST_SCHEMA",
    "CANDIDATE_MANIFEST_SCHEMA",
    "DEFAULT_MACRO_ROOT",
    "DEFAULT_RAW_SNAPSHOT_ROOT",
    "MACRO_FIELDS",
    "MacroMartPromotionError",
    "read_macro_mart",
    "run_cn_macro_maintenance",
    "write_macro_mart",
]
