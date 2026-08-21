"""Strict-local compiler for one CN market-breadth observation.

One explicit immutable v4 market manifest supplies the actual Parquet rows.
One separately hash-pinned target-date manifest supplies top-level v3/v4
coverage, and one independently hash-pinned full-A artifact supplies the
expected symbol set.  The compiler is read-only and produces exactly one row.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Callable, Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_investor.macro.contracts import (
    MacroObservation,
    SHANGHAI,
    UTC,
    canonical_hash,
    published_cutoff,
)
from quant_investor.market.market_data_reader import (
    _complete_coverage_blockers,
)


LOCAL_MARKET_BREADTH_EVIDENCE_SCHEMA = (
    "cn-local-market-breadth-evidence.v2"
)
LOCAL_MARKET_BREADTH_FORMULA_CONTRACT = {
    "schema_version": "cn-market-breadth-formula.v2",
    "indicator_id": "market.breadth",
    "selection": "one_explicit_coverage_certified_trade_date",
    "universe": "explicit_sha256_bound_full_a_scope",
    "key_columns": ["ts_code", "trade_date"],
    "value_column": "pct_chg",
    "numerator": "count(pct_chg>0)",
    "denominator": "count(finite_observed_full_a_bars)",
    "scale": 100.0,
    "minimum_observed_bars": 100,
    "coverage_closure": (
        "observed_symbols_disjoint_union_non_blocking_absent_symbols_"
        "equals_expected_scope"
    ),
    "non_finite_policy": "reject",
    "duplicate_key_policy": "reject",
}
LOCAL_MARKET_BREADTH_FORMULA_SHA256 = canonical_hash(
    LOCAL_MARKET_BREADTH_FORMULA_CONTRACT
)
MIN_OBSERVED_BARS = 100
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_SCOPE_BYTES = 16 * 1024 * 1024
_MAX_PART_BYTES = 512 * 1024 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")


class LocalMarketObservationError(ValueError):
    """Raised when strict-local market evidence is not safely compilable."""


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _datetime_ns(value: datetime) -> int:
    utc_value = value.astimezone(UTC)
    return (
        int(utc_value.timestamp()) * 1_000_000_000
        + utc_value.microsecond * 1_000
    )


def _utc_from_ns(value: int) -> datetime:
    # MacroObservation timestamps are microsecond-resolution. Round upward so
    # the serialized timestamp never precedes exact filesystem availability.
    total_microseconds = (int(value) + 999) // 1_000
    seconds, microseconds = divmod(total_microseconds, 1_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(
        microsecond=microseconds
    )


def _absolute_path(value: str | Path, *, blocker: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.parts or ".." in raw.parts:
        raise LocalMarketObservationError(blocker)
    return Path(os.path.abspath(raw))


def _assert_no_symlink_components(path: Path, *, blocker: str) -> None:
    absolute = _absolute_path(path, blocker=blocker)
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise LocalMarketObservationError(blocker) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise LocalMarketObservationError(blocker)


def _stable_file_bytes(
    path: Path,
    *,
    blocker: str,
    changed_blocker: str,
    max_bytes: int,
) -> tuple[bytes, tuple[int, ...]]:
    resolved = _absolute_path(path, blocker=blocker)
    _assert_no_symlink_components(resolved, blocker=blocker)
    descriptor: int | None = None
    try:
        before = os.lstat(resolved)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise LocalMarketObservationError(blocker)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
        signature = _stat_signature(before)
        if _stat_signature(os.fstat(descriptor)) != signature:
            raise LocalMarketObservationError(changed_blocker)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise LocalMarketObservationError(blocker)
            chunks.append(chunk)
        if (
            _stat_signature(os.fstat(descriptor)) != signature
            or _stat_signature(os.lstat(resolved)) != signature
        ):
            raise LocalMarketObservationError(changed_blocker)
        return b"".join(chunks), signature
    except LocalMarketObservationError:
        raise
    except OSError as exc:
        raise LocalMarketObservationError(blocker) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _required_sha256(value: Any, *, blocker: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        character not in _SHA256_HEX for character in text
    ):
        raise LocalMarketObservationError(blocker)
    return text


def _json_object(raw: bytes, *, blocker: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalMarketObservationError(blocker) from exc
    if not isinstance(payload, dict):
        raise LocalMarketObservationError(blocker)
    return payload


def _compact_date(value: Any, *, blocker: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = (
            datetime.strptime(text, "%Y%m%d").date()
            if len(text) == 8 and text.isdigit()
            else date.fromisoformat(text)
        )
    except ValueError as exc:
        raise LocalMarketObservationError(blocker) from exc
    return parsed.strftime("%Y%m%d")


def _coverage_integer(value: Any, *, blocker: str) -> int:
    if isinstance(value, bool):
        raise LocalMarketObservationError(blocker)
    try:
        number = int(value)
        if float(value) != float(number):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise LocalMarketObservationError(blocker) from exc
    return number


def _symbol_list(
    raw: Any,
    *,
    blocker: str,
) -> list[str]:
    if not isinstance(raw, list):
        raise LocalMarketObservationError(blocker)
    normalized = [str(symbol or "").strip().upper() for symbol in raw]
    if (
        any(not symbol for symbol in normalized)
        or len(normalized) != len(set(normalized))
    ):
        raise LocalMarketObservationError(blocker)
    return sorted(normalized)


def _coverage_symbol_list(
    coverage: Mapping[str, Any],
    field_name: str,
) -> list[str]:
    return _symbol_list(
        coverage.get(field_name, []),
        blocker=(
            f"local_breadth_coverage_{field_name}_duplicates_or_invalid"
        ),
    )


def _coverage_summary(
    coverage: Mapping[str, Any],
    *,
    target_trade_date: str,
) -> tuple[dict[str, Any], str]:
    schema = str(coverage.get("coverage_schema_version") or "").strip()
    if schema not in {"cn-full-a-coverage.v3", "cn-full-a-coverage.v4"}:
        raise LocalMarketObservationError(
            "local_breadth_coverage_schema_invalid"
        )
    blockers = _complete_coverage_blockers(
        coverage,
        latest_complete_trade_date=target_trade_date,
    )
    if blockers:
        raise LocalMarketObservationError(
            "local_breadth_coverage_contract_invalid:" + ",".join(blockers)
        )
    if coverage.get("complete") is not True:
        raise LocalMarketObservationError(
            "local_breadth_coverage_not_complete"
        )
    try:
        coverage_ratio = float(coverage.get("coverage_ratio"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise LocalMarketObservationError(
            "local_breadth_coverage_ratio_invalid"
        ) from exc
    if (
        isinstance(coverage.get("coverage_ratio"), bool)
        or not math.isfinite(coverage_ratio)
        or coverage_ratio != 1.0
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_ratio_not_one"
        )
    expected_scope_count = _coverage_integer(
        coverage.get("expected_scope_count"),
        blocker="local_breadth_coverage_expected_scope_count_invalid",
    )
    coverage_complete_count = _coverage_integer(
        coverage.get("coverage_complete_count"),
        blocker="local_breadth_coverage_complete_count_invalid",
    )
    observed_bar_count = _coverage_integer(
        coverage.get("observed_bar_count"),
        blocker="local_breadth_coverage_observed_bar_count_invalid",
    )
    blocking_incomplete_count = _coverage_integer(
        coverage.get("blocking_incomplete_count"),
        blocker="local_breadth_coverage_blocking_count_invalid",
    )
    if (
        expected_scope_count <= 0
        or coverage_complete_count != expected_scope_count
        or blocking_incomplete_count != 0
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_counts_invalid"
        )
    categories_raw = coverage.get("categories_checked")
    if not isinstance(categories_raw, list):
        raise LocalMarketObservationError(
            "local_breadth_coverage_categories_invalid"
        )
    categories = [str(item or "").strip().lower() for item in categories_raw]
    if (
        any(not item for item in categories)
        or len(categories) != len(set(categories))
        or "full_a" not in categories
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_full_a_missing"
        )
    symbol_fields = (
        "suspended_symbols",
        "inactive_symbols",
        "verified_nontrading_bak_daily_zero_symbols",
        "verified_terminal_delisting_symbols",
        "allowed_stale_symbols",
        "non_blocking_absent_symbols",
        "true_missing_symbols",
    )
    symbol_sets = {
        field_name: _coverage_symbol_list(coverage, field_name)
        for field_name in symbol_fields
    }
    if symbol_sets["true_missing_symbols"]:
        raise LocalMarketObservationError(
            "local_breadth_coverage_true_missing_nonempty"
        )
    if (
        observed_bar_count
        + len(symbol_sets["non_blocking_absent_symbols"])
        != expected_scope_count
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_observed_absent_count_mismatch"
        )
    if coverage.get("classification_sets_disjoint") is not True:
        raise LocalMarketObservationError(
            "local_breadth_coverage_classification_sets_not_disjoint"
        )
    date_fields = (
        "latest_available_trade_date",
        "latest_complete_trade_date",
        "upsert_target_trade_date",
        "coverage_trade_date",
    )
    normalized_dates = {
        field_name: _compact_date(
            coverage.get(field_name),
            blocker=f"local_breadth_coverage_{field_name}_invalid",
        )
        for field_name in date_fields
    }
    if any(
        value != target_trade_date for value in normalized_dates.values()
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_trade_date_mismatch"
        )
    expected_scope_sha256 = _required_sha256(
        coverage.get("expected_scope_sha256"),
        blocker="local_breadth_coverage_expected_scope_sha256_invalid",
    )
    pit_generation_id = str(coverage.get("pit_generation_id") or "").strip()
    if pit_generation_id and Path(pit_generation_id).name != pit_generation_id:
        raise LocalMarketObservationError(
            "local_breadth_coverage_pit_generation_id_invalid"
        )
    pit_fields: dict[str, Any] = {
        "pit_generation_manifest_path": str(
            coverage.get("pit_generation_manifest_path") or ""
        ).strip(),
        "pit_generation_manifest_sha256": str(
            coverage.get("pit_generation_manifest_sha256") or ""
        ).strip().lower(),
        "pit_membership_path": str(
            coverage.get("pit_membership_path") or ""
        ).strip(),
        "pit_membership_sha256": str(
            coverage.get("pit_membership_sha256") or ""
        ).strip().lower(),
    }
    for field_name in (
        "pit_generation_manifest_sha256",
        "pit_membership_sha256",
    ):
        digest = str(pit_fields[field_name])
        if digest and (
            len(digest) != 64
            or any(character not in _SHA256_HEX for character in digest)
        ):
            raise LocalMarketObservationError(
                f"local_breadth_coverage_{field_name}_invalid"
            )
    if schema == "cn-full-a-coverage.v4" and (
        not pit_generation_id
        or not pit_fields["pit_generation_manifest_path"]
        or not pit_fields["pit_generation_manifest_sha256"]
        or not pit_fields["pit_membership_path"]
        or not pit_fields["pit_membership_sha256"]
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_v4_pit_binding_missing"
        )
    summary: dict[str, Any] = {
        "coverage_schema_version": schema,
        "complete": True,
        "coverage_ratio": 1.0,
        "categories_checked": sorted(categories),
        "expected_scope_count": expected_scope_count,
        "coverage_complete_count": coverage_complete_count,
        "observed_bar_count": observed_bar_count,
        "blocking_incomplete_count": blocking_incomplete_count,
        "expected_scope_sha256": expected_scope_sha256,
        "classification_sets_disjoint": True,
        "pit_generation_id": pit_generation_id,
        **pit_fields,
        **symbol_sets,
        **normalized_dates,
    }
    return summary, canonical_hash(summary)


def _canonical_mapping_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    try:
        return canonical_hash(dict(left)) == canonical_hash(dict(right))
    except (TypeError, ValueError):
        return False


def _coverage_from_manifest(
    manifest: Mapping[str, Any],
    *,
    target_trade_date: str,
) -> tuple[dict[str, Any], str]:
    top = manifest.get("coverage")
    metadata = manifest.get("metadata")
    metadata_coverage = (
        metadata.get("coverage") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(top, Mapping):
        raise LocalMarketObservationError(
            "local_breadth_coverage_top_level_invalid"
        )
    if not _canonical_mapping_equal(top, metadata_coverage):
        raise LocalMarketObservationError(
            "local_breadth_coverage_top_metadata_conflict"
        )
    summary, contract_sha = _coverage_summary(
        top,
        target_trade_date=target_trade_date,
    )
    return summary, contract_sha


def _snapshot_time(snapshot_id: str) -> datetime:
    try:
        return datetime.strptime(
            snapshot_id,
            "%Y%m%dT%H%M%SZ",
        ).replace(tzinfo=UTC)
    except ValueError as exc:
        raise LocalMarketObservationError(
            "local_breadth_snapshot_id_utc_invalid"
        ) from exc


def _resolve_manifest_path(
    raw_value: Any,
    *,
    manifest_path: Path,
    blocker: str,
) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise LocalMarketObservationError(blocker)
    raw = Path(text).expanduser()
    if ".." in raw.parts:
        raise LocalMarketObservationError(blocker)
    if raw.is_absolute():
        candidates = [_absolute_path(raw, blocker=blocker)]
    else:
        candidates = [
            _absolute_path(Path.cwd() / raw, blocker=blocker),
            _absolute_path(manifest_path.parent / raw, blocker=blocker),
        ]
    existing: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in existing:
            existing.append(candidate)
    if len(existing) != 1:
        raise LocalMarketObservationError(blocker)
    _assert_no_symlink_components(existing[0], blocker=blocker)
    return existing[0]


def _directory_signature(path: Path, *, blocker: str) -> tuple[int, ...]:
    _assert_no_symlink_components(path, blocker=blocker)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise LocalMarketObservationError(blocker) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise LocalMarketObservationError(blocker)
    return _stat_signature(metadata)


def _read_month_part(
    part_path: Path,
) -> tuple[pd.DataFrame, bytes, tuple[int, ...]]:
    raw, signature = _stable_file_bytes(
        part_path,
        blocker="local_breadth_part_unsafe_or_unreadable",
        changed_blocker="local_breadth_part_changed_during_read",
        max_bytes=_MAX_PART_BYTES,
    )
    try:
        frame = pd.read_parquet(
            io.BytesIO(raw),
            columns=["ts_code", "trade_date", "pct_chg"],
        )
    except Exception as exc:
        raise LocalMarketObservationError(
            "local_breadth_part_parquet_invalid"
        ) from exc
    if set(frame.columns) != {"ts_code", "trade_date", "pct_chg"}:
        raise LocalMarketObservationError("local_breadth_part_schema_invalid")
    if frame.empty:
        raise LocalMarketObservationError("local_breadth_part_empty")
    symbols = (
        frame["ts_code"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )
    raw_dates = frame["trade_date"].astype("string").fillna("").str.strip()
    if symbols.eq("").any() or not raw_dates.str.fullmatch(r"[0-9]{8}").all():
        raise LocalMarketObservationError("local_breadth_part_values_invalid")
    parsed_dates = pd.to_datetime(
        raw_dates,
        format="%Y%m%d",
        errors="coerce",
    )
    numeric = pd.to_numeric(frame["pct_chg"], errors="coerce")
    if parsed_dates.isna().any():
        raise LocalMarketObservationError("local_breadth_part_values_invalid")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise LocalMarketObservationError("local_breadth_pct_chg_non_finite")
    normalized = pd.DataFrame(
        {
            "ts_code": symbols.astype(str),
            "trade_date": raw_dates.astype(str),
            "pct_chg": numeric.astype(float),
        }
    )
    if normalized.duplicated(["ts_code", "trade_date"]).any():
        raise LocalMarketObservationError("local_breadth_duplicate_bar")
    return normalized, raw, signature


def _load_scope_artifact(
    *,
    path: Path,
    expected_sha256: str,
    coverage_summary: Mapping[str, Any],
) -> tuple[frozenset[str], dict[str, Any], bytes, tuple[int, ...]]:
    raw, signature = _stable_file_bytes(
        path,
        blocker="local_breadth_scope_artifact_unsafe_or_unreadable",
        changed_blocker="local_breadth_scope_artifact_changed_during_read",
        max_bytes=_MAX_SCOPE_BYTES,
    )
    actual_file_sha = hashlib.sha256(raw).hexdigest()
    if actual_file_sha != expected_sha256:
        raise LocalMarketObservationError(
            "local_breadth_scope_artifact_sha256_mismatch"
        )
    payload = _json_object(
        raw,
        blocker="local_breadth_scope_artifact_json_invalid",
    )
    symbols = _symbol_list(
        payload.get("full_a"),
        blocker="local_breadth_scope_artifact_full_a_invalid",
    )
    semantic_sha = hashlib.sha256(
        "\n".join(symbols).encode("utf-8")
    ).hexdigest()
    if len(symbols) != int(coverage_summary["expected_scope_count"]):
        raise LocalMarketObservationError(
            "local_breadth_scope_artifact_count_mismatch"
        )
    if semantic_sha != coverage_summary["expected_scope_sha256"]:
        raise LocalMarketObservationError(
            "local_breadth_scope_artifact_semantic_sha256_mismatch"
        )
    scope = frozenset(symbols)
    classification_fields = (
        "suspended_symbols",
        "inactive_symbols",
        "verified_nontrading_bak_daily_zero_symbols",
        "verified_terminal_delisting_symbols",
        "allowed_stale_symbols",
        "non_blocking_absent_symbols",
        "true_missing_symbols",
    )
    if any(
        not set(coverage_summary[field_name]).issubset(scope)
        for field_name in classification_fields
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_classification_outside_scope"
        )
    evidence = {
        "path": str(path),
        "file_sha256": actual_file_sha,
        "mtime_ns": int(signature[4]),
        "size_bytes": len(raw),
        "full_a_count": len(symbols),
        "full_a_semantic_sha256": semantic_sha,
    }
    return scope, evidence, raw, signature


def _aware_clock_value(clock: Callable[[], datetime] | None) -> datetime:
    value = clock() if clock is not None else datetime.now(UTC)
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise LocalMarketObservationError(
            "local_breadth_clock_timezone_required"
        )
    return value.astimezone(UTC)


def compile_local_market_breadth_observation(
    *,
    snapshot_manifest_path: str | Path,
    expected_snapshot_manifest_sha256: str,
    coverage_manifest_path: str | Path,
    expected_coverage_manifest_sha256: str,
    target_trade_date: str,
    scope_artifact_path: str | Path,
    expected_scope_artifact_sha256: str,
    as_of: str | datetime,
    clock: Callable[[], datetime] | None = None,
) -> tuple[MacroObservation, dict[str, Any]]:
    """Compile one hash-bound local ``market.breadth`` observation."""

    expected_snapshot_manifest_sha = _required_sha256(
        expected_snapshot_manifest_sha256,
        blocker="local_breadth_expected_snapshot_manifest_sha256_invalid",
    )
    expected_coverage_manifest_sha = _required_sha256(
        expected_coverage_manifest_sha256,
        blocker="local_breadth_expected_coverage_manifest_sha256_invalid",
    )
    expected_scope_file_sha = _required_sha256(
        expected_scope_artifact_sha256,
        blocker="local_breadth_expected_scope_artifact_sha256_invalid",
    )
    target_date = _compact_date(
        target_trade_date,
        blocker="local_breadth_target_trade_date_invalid",
    )
    manifest_path = _absolute_path(
        snapshot_manifest_path,
        blocker="local_breadth_manifest_path_invalid",
    )
    coverage_path = _absolute_path(
        coverage_manifest_path,
        blocker="local_breadth_coverage_manifest_path_invalid",
    )
    scope_path = _absolute_path(
        scope_artifact_path,
        blocker="local_breadth_scope_artifact_path_invalid",
    )
    manifest_raw, manifest_signature = _stable_file_bytes(
        manifest_path,
        blocker="local_breadth_manifest_unsafe_or_unreadable",
        changed_blocker="local_breadth_manifest_changed_during_read",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != expected_snapshot_manifest_sha:
        raise LocalMarketObservationError(
            "local_breadth_manifest_sha256_mismatch"
        )
    manifest = _json_object(
        manifest_raw,
        blocker="local_breadth_manifest_json_invalid",
    )
    snapshot_id = str(manifest.get("snapshot_id") or "").strip()
    snapshot_at = _snapshot_time(snapshot_id)
    if manifest_path.name != f"{snapshot_id}.json":
        raise LocalMarketObservationError(
            "local_breadth_manifest_snapshot_id_path_mismatch"
        )
    if manifest_path.parent.name != "_snapshots":
        raise LocalMarketObservationError(
            "local_breadth_v4_snapshot_manifest_root_invalid"
        )
    bound_manifest_path = _resolve_manifest_path(
        manifest.get("manifest_path"),
        manifest_path=manifest_path,
        blocker="local_breadth_manifest_self_path_invalid",
    )
    if bound_manifest_path != manifest_path:
        raise LocalMarketObservationError(
            "local_breadth_manifest_self_path_mismatch"
        )
    if (
        str(manifest.get("market") or "").strip().upper() != "CN"
        or str(manifest.get("status") or "").strip().upper() != "OK"
        or manifest.get("readback_validated") is not True
        or list(manifest.get("blockers") or [])
    ):
        raise LocalMarketObservationError(
            "local_breadth_manifest_contract_invalid"
        )
    try:
        cutoff = published_cutoff(as_of)
    except ValueError as exc:
        raise LocalMarketObservationError(
            "local_breadth_as_of_invalid"
        ) from exc
    latest_complete = _compact_date(
        manifest.get("latest_complete_trade_date"),
        blocker="local_breadth_latest_complete_trade_date_invalid",
    )
    data_coverage = manifest.get("coverage")
    if (
        not isinstance(data_coverage, Mapping)
        or str(data_coverage.get("coverage_schema_version") or "").strip()
        != "cn-full-a-coverage.v4"
    ):
        raise LocalMarketObservationError(
            "local_breadth_snapshot_manifest_v4_required"
        )
    if target_date > latest_complete:
        raise LocalMarketObservationError(
            "local_breadth_target_after_snapshot_latest_complete"
        )

    coverage_raw, coverage_signature = _stable_file_bytes(
        coverage_path,
        blocker="local_breadth_coverage_manifest_unsafe_or_unreadable",
        changed_blocker="local_breadth_coverage_manifest_changed_during_read",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    coverage_manifest_sha = hashlib.sha256(coverage_raw).hexdigest()
    if coverage_manifest_sha != expected_coverage_manifest_sha:
        raise LocalMarketObservationError(
            "local_breadth_coverage_manifest_sha256_mismatch"
        )
    coverage_manifest = _json_object(
        coverage_raw,
        blocker="local_breadth_coverage_manifest_json_invalid",
    )
    coverage_snapshot_id = str(
        coverage_manifest.get("snapshot_id") or ""
    ).strip()
    coverage_snapshot_at = _snapshot_time(coverage_snapshot_id)
    if coverage_path.name != f"{coverage_snapshot_id}.json":
        raise LocalMarketObservationError(
            "local_breadth_coverage_manifest_snapshot_id_path_mismatch"
        )
    bound_coverage_path = _resolve_manifest_path(
        coverage_manifest.get("manifest_path"),
        manifest_path=coverage_path,
        blocker="local_breadth_coverage_manifest_self_path_invalid",
    )
    if bound_coverage_path != coverage_path:
        raise LocalMarketObservationError(
            "local_breadth_coverage_manifest_self_path_mismatch"
        )
    if (
        str(coverage_manifest.get("market") or "").strip().upper() != "CN"
        or str(coverage_manifest.get("status") or "").strip().upper() != "OK"
        or coverage_manifest.get("readback_validated") is not True
        or list(coverage_manifest.get("blockers") or [])
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_manifest_contract_invalid"
        )
    coverage_latest_complete = _compact_date(
        coverage_manifest.get("latest_complete_trade_date"),
        blocker=(
            "local_breadth_coverage_manifest_"
            "latest_complete_trade_date_invalid"
        ),
    )
    if coverage_latest_complete != target_date:
        raise LocalMarketObservationError(
            "local_breadth_coverage_manifest_target_mismatch"
        )
    coverage_summary, coverage_contract_sha256 = _coverage_from_manifest(
        coverage_manifest,
        target_trade_date=target_date,
    )
    if snapshot_at > cutoff:
        raise LocalMarketObservationError(
            "local_breadth_snapshot_after_published_cutoff"
        )
    if coverage_snapshot_at > cutoff:
        raise LocalMarketObservationError(
            "local_breadth_coverage_snapshot_after_published_cutoff"
        )
    if published_cutoff(target_date) > cutoff:
        raise LocalMarketObservationError(
            "local_breadth_target_after_as_of"
        )
    fetched_at = _aware_clock_value(clock)

    table_root = _resolve_manifest_path(
        manifest.get("table_root"),
        manifest_path=manifest_path,
        blocker="local_breadth_table_root_invalid",
    )
    reconstruction = manifest.get("retrospective_reconstruction")
    if reconstruction is None:
        expected_table_root = _absolute_path(
            manifest_path.parent / snapshot_id / "table" / "bars",
            blocker="local_breadth_v4_table_root_invalid",
        )
    else:
        if (
            not isinstance(reconstruction, Mapping)
            or set(reconstruction)
            != {
                "classification",
                "reconstructed_at",
                "source_snapshot_manifest_path",
                "source_snapshot_manifest_sha256",
            }
            or reconstruction.get("classification")
            != "RETROSPECTIVE_RECONSTRUCTION"
            or reconstruction.get("reconstructed_at")
            != snapshot_at.isoformat()
        ):
            raise LocalMarketObservationError(
                "local_breadth_retrospective_reconstruction_invalid"
            )
        source_path = _absolute_path(
            reconstruction.get("source_snapshot_manifest_path"),
            blocker="local_breadth_reconstruction_source_path_invalid",
        )
        source_raw, _source_signature = _stable_file_bytes(
            source_path,
            blocker="local_breadth_reconstruction_source_unsafe",
            changed_blocker="local_breadth_reconstruction_source_changed",
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        if hashlib.sha256(source_raw).hexdigest() != _required_sha256(
            reconstruction.get("source_snapshot_manifest_sha256"),
            blocker="local_breadth_reconstruction_source_sha_invalid",
        ):
            raise LocalMarketObservationError(
                "local_breadth_reconstruction_source_sha_mismatch"
            )
        source_manifest = _json_object(
            source_raw,
            blocker="local_breadth_reconstruction_source_json_invalid",
        )
        expected_table_root = _resolve_manifest_path(
            source_manifest.get("table_root"),
            manifest_path=source_path,
            blocker="local_breadth_reconstruction_table_root_invalid",
        )
    if table_root != expected_table_root:
        raise LocalMarketObservationError(
            "local_breadth_v4_immutable_table_root_required"
        )
    table_signature = _directory_signature(
        table_root,
        blocker="local_breadth_table_root_invalid",
    )

    scope, scope_evidence, scope_raw, scope_signature = _load_scope_artifact(
        path=scope_path,
        expected_sha256=expected_scope_file_sha,
        coverage_summary=coverage_summary,
    )

    year = int(target_date[:4])
    month = int(target_date[4:6])
    relative_part = (
        Path(f"year={year:04d}")
        / f"month={month:02d}"
        / "part.parquet"
    )
    part_path = table_root / relative_part
    frame, part_raw, part_signature = _read_month_part(part_path)
    part_sha = hashlib.sha256(part_raw).hexdigest()
    target_rows_all = frame.loc[frame["trade_date"] == target_date].copy()
    if target_rows_all.empty:
        raise LocalMarketObservationError(
            "local_breadth_target_rows_missing"
        )
    all_target_symbols = set(target_rows_all["ts_code"].astype(str))
    rows = target_rows_all.loc[
        target_rows_all["ts_code"].isin(scope)
    ].copy()
    rows = rows.sort_values("ts_code", kind="mergesort").reset_index(drop=True)
    observed_symbols = set(rows["ts_code"].astype(str))
    absent_symbols = set(coverage_summary["non_blocking_absent_symbols"])
    if len(rows) != len(observed_symbols):
        raise LocalMarketObservationError(
            "local_breadth_target_unique_symbol_count_mismatch"
        )
    if len(observed_symbols) != int(coverage_summary["observed_bar_count"]):
        raise LocalMarketObservationError(
            "local_breadth_actual_observed_bar_count_mismatch"
        )
    if observed_symbols & absent_symbols:
        raise LocalMarketObservationError(
            "local_breadth_observed_absent_overlap"
        )
    if observed_symbols | absent_symbols != set(scope):
        raise LocalMarketObservationError(
            "local_breadth_expected_scope_closure_mismatch"
        )
    row_count = int(len(rows))
    if row_count < MIN_OBSERVED_BARS:
        raise LocalMarketObservationError(
            f"local_breadth_rows_insufficient:{target_date}:{row_count}"
        )

    manifest_mtime_ns = int(manifest_signature[4])
    coverage_manifest_mtime_ns = int(coverage_signature[4])
    effective_available_ns = max(
        _datetime_ns(snapshot_at),
        _datetime_ns(coverage_snapshot_at),
        manifest_mtime_ns,
        coverage_manifest_mtime_ns,
        int(scope_signature[4]),
        int(part_signature[4]),
    )
    effective_available_at = _utc_from_ns(effective_available_ns)
    if effective_available_ns > _datetime_ns(cutoff):
        raise LocalMarketObservationError(
            "local_breadth_effective_available_after_published_cutoff"
        )
    if _datetime_ns(fetched_at) < effective_available_ns:
        raise LocalMarketObservationError(
            "local_breadth_fetched_before_effective_available_at"
        )
    release_local = datetime.combine(
        datetime.strptime(target_date, "%Y%m%d").date(),
        time(15, 0),
        tzinfo=SHANGHAI,
    )
    if effective_available_at < release_local.astimezone(UTC):
        raise LocalMarketObservationError(
            "local_breadth_effective_available_before_session_release"
        )

    positive_count = int(rows["pct_chg"].gt(0.0).sum())
    value = round(positive_count / row_count * 100.0, 12)
    if not math.isfinite(value):
        raise LocalMarketObservationError(
            "local_breadth_value_non_finite"
        )
    row_set = [
        {
            "ts_code": str(row.ts_code),
            "trade_date": str(row.trade_date),
            "pct_chg": float(row.pct_chg),
        }
        for row in rows.itertuples(index=False)
    ]
    outside_scope_symbols = sorted(all_target_symbols - set(scope))
    scope_evidence_sha256 = canonical_hash(scope_evidence)
    binding_payload = {
        "data_snapshot_id": snapshot_id,
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_manifest_mtime_ns": manifest_mtime_ns,
        "coverage_snapshot_id": coverage_snapshot_id,
        "coverage_manifest_sha256": coverage_manifest_sha,
        "coverage_manifest_mtime_ns": coverage_manifest_mtime_ns,
        "target_trade_date": target_date,
        "coverage_source_location": "top",
        "coverage_summary": dict(coverage_summary),
        "coverage_contract_sha256": coverage_contract_sha256,
        "scope_evidence_sha256": scope_evidence_sha256,
        "part_relative_path": relative_part.as_posix(),
        "part_sha256": part_sha,
        "part_mtime_ns": int(part_signature[4]),
        "effective_available_at": effective_available_at.isoformat(),
        "formula_contract_sha256": LOCAL_MARKET_BREADTH_FORMULA_SHA256,
        "row_count": row_count,
        "positive_row_count": positive_count,
        "row_set_sha256": canonical_hash({"rows": row_set}),
        "outside_scope_symbol_count": len(outside_scope_symbols),
        "outside_scope_symbols_sha256": hashlib.sha256(
            "\n".join(outside_scope_symbols).encode("utf-8")
        ).hexdigest(),
        "value": value,
    }
    binding_sha = canonical_hash(binding_payload)
    source_url = (
        "local://strict-parquet/cn/snapshots/"
        f"{snapshot_id}/bars/{target_date}"
    )
    observation = MacroObservation.from_mapping(
        {
            "indicator_id": "market.breadth",
            "dimension_type": "market_confirmation",
            "period_end": target_date,
            "release_at": release_local.isoformat(),
            "available_at": effective_available_at.isoformat(),
            "vintage_id": f"local-strict-parquet.v2:{binding_sha}",
            "value": value,
            "unit": "%",
            "frequency": "daily",
            "source_system": "local_strict_parquet",
            "source_record_id": (
                f"market.breadth:{target_date}:{binding_sha}"
            ),
            "source_url": source_url,
            "fetched_at": fetched_at.isoformat(),
            "quality_status": "pass",
        }
    )
    binding = {
        **binding_payload,
        "binding_sha256": binding_sha,
        "vintage_id": observation.vintage_id,
        "content_hash": observation.content_hash,
    }

    manifest_readback, manifest_readback_signature = _stable_file_bytes(
        manifest_path,
        blocker="local_breadth_manifest_unsafe_or_unreadable",
        changed_blocker="local_breadth_manifest_changed_during_compile",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    if (
        manifest_readback != manifest_raw
        or manifest_readback_signature != manifest_signature
    ):
        raise LocalMarketObservationError(
            "local_breadth_manifest_changed_during_compile"
        )
    coverage_readback, coverage_readback_signature = _stable_file_bytes(
        coverage_path,
        blocker="local_breadth_coverage_manifest_unsafe_or_unreadable",
        changed_blocker=(
            "local_breadth_coverage_manifest_changed_during_compile"
        ),
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    if (
        coverage_readback != coverage_raw
        or coverage_readback_signature != coverage_signature
    ):
        raise LocalMarketObservationError(
            "local_breadth_coverage_manifest_changed_during_compile"
        )
    scope_readback, scope_readback_signature = _stable_file_bytes(
        scope_path,
        blocker="local_breadth_scope_artifact_unsafe_or_unreadable",
        changed_blocker=(
            "local_breadth_scope_artifact_changed_during_compile"
        ),
        max_bytes=_MAX_SCOPE_BYTES,
    )
    if (
        scope_readback != scope_raw
        or scope_readback_signature != scope_signature
    ):
        raise LocalMarketObservationError(
            "local_breadth_scope_artifact_changed_during_compile"
        )
    if _directory_signature(
        table_root,
        blocker="local_breadth_table_root_changed_during_compile",
    ) != table_signature:
        raise LocalMarketObservationError(
            "local_breadth_table_root_changed_during_compile"
        )
    part_readback, part_readback_signature = _stable_file_bytes(
        part_path,
        blocker="local_breadth_part_unsafe_or_unreadable",
        changed_blocker="local_breadth_part_changed_during_compile",
        max_bytes=_MAX_PART_BYTES,
    )
    if (
        hashlib.sha256(part_readback).hexdigest() != part_sha
        or part_readback_signature != part_signature
    ):
        raise LocalMarketObservationError(
            "local_breadth_part_changed_during_compile"
        )
    part_evidence = {
        "relative_path": relative_part.as_posix(),
        "sha256": part_sha,
        "mtime_ns": int(part_signature[4]),
        "size_bytes": len(part_raw),
        "month_row_count": int(len(frame)),
        "target_all_row_count": int(len(target_rows_all)),
        "target_scope_row_count": row_count,
        "target_outside_scope_row_count": int(
            len(target_rows_all) - row_count
        ),
    }
    evidence: dict[str, Any] = {
        "schema_version": LOCAL_MARKET_BREADTH_EVIDENCE_SCHEMA,
        "market": "CN",
        "indicator_id": "market.breadth",
        "as_of_cutoff": cutoff.isoformat(),
        "data_snapshot_id": snapshot_id,
        "data_snapshot_available_at": snapshot_at.isoformat(),
        "snapshot_manifest_mtime_ns": manifest_mtime_ns,
        "effective_available_at": effective_available_at.isoformat(),
        "snapshot_manifest_path": str(manifest_path),
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_latest_complete_trade_date": latest_complete,
        "coverage_snapshot_id": coverage_snapshot_id,
        "coverage_snapshot_available_at": coverage_snapshot_at.isoformat(),
        "coverage_manifest_mtime_ns": coverage_manifest_mtime_ns,
        "coverage_manifest_path": str(coverage_path),
        "coverage_manifest_sha256": coverage_manifest_sha,
        "target_trade_date": target_date,
        "coverage_source_location": "top",
        "coverage_summary": dict(coverage_summary),
        "coverage_contract_sha256": coverage_contract_sha256,
        "scope_artifact": scope_evidence,
        "scope_evidence_sha256": scope_evidence_sha256,
        "table_root": str(table_root),
        "part_file": part_evidence,
        "formula_contract": dict(LOCAL_MARKET_BREADTH_FORMULA_CONTRACT),
        "formula_contract_sha256": LOCAL_MARKET_BREADTH_FORMULA_SHA256,
        "observation_binding": binding,
        "observation_content_hash": observation.content_hash,
        "fetched_at": fetched_at.isoformat(),
        "local_read_only": True,
        "canonical_write": False,
        "canonical_published": False,
    }
    evidence["evidence_sha256"] = canonical_hash(evidence)
    return observation, evidence


__all__ = [
    "LOCAL_MARKET_BREADTH_EVIDENCE_SCHEMA",
    "LOCAL_MARKET_BREADTH_FORMULA_CONTRACT",
    "LOCAL_MARKET_BREADTH_FORMULA_SHA256",
    "LocalMarketObservationError",
    "compile_local_market_breadth_observation",
]
