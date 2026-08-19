"""Sealed, replayable CN daily market capture and publication.

The acquisition phase is deliberately separated from publication.  Provider
responses are written once to an immutable private fileset; publication reads
only that fileset and therefore cannot silently refetch or choose a different
winner after the preflight checks have passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import stat
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_investor.market.market_data_store import MarketDataStore

CAPTURE_SCHEMA = "cn-market-daily-capture.v1"
ENDPOINT_SCHEMA = "cn-market-provider-response.v1"
TARGET_AUTHORITY_SCHEMA = "cn-close-session-receipt.v1"
CLASSIFICATION_SCHEMA = "cn-market-target-classification.v1"
_ENDPOINTS = ("daily", "daily_basic", "adj_factor")
_CLASSIFICATION_REASONS = (
    "observed",
    "suspended",
    "non_trading",
    "delisted",
    "prelisting",
    "inactive",
    "true_missing",
)
_SYMBOL_RE = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_FSYNC_RESERVE_BYTES = 64 * 1024 * 1024

_FIELDS: dict[str, tuple[str, ...]] = {
    "daily": (
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ),
    "daily_basic": (
        "ts_code",
        "trade_date",
        "turnover_rate",
        "volume_ratio",
        "pe",
        "pb",
        "total_mv",
        "circ_mv",
    ),
    "adj_factor": ("ts_code", "trade_date", "adj_factor"),
}


class MarketDailyCaptureBlocked(RuntimeError):
    """Raised when capture or replay cannot safely authorize publication."""

    def __init__(self, blockers: Sequence[str], receipt: Mapping[str, Any] | None = None):
        self.blockers = list(dict.fromkeys(str(item) for item in blockers if str(item)))
        self.receipt = dict(receipt or {})
        super().__init__(";".join(self.blockers) or "market_daily_capture_blocked")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_sha256(value: Any, *, blocker: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise MarketDailyCaptureBlocked([blocker])
    return digest


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_trade_date(value: Any) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    if len(digits) < 8:
        return ""
    compact = digits[:8]
    try:
        date(int(compact[:4]), int(compact[4:6]), int(compact[6:8]))
    except ValueError:
        return ""
    return compact


def _stable_regular_bytes(path: str | Path, *, label: str) -> bytes:
    resolved = Path(path).expanduser()
    try:
        before = resolved.lstat()
    except OSError as exc:
        raise MarketDailyCaptureBlocked([f"{label}_missing:{type(exc).__name__}"]) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise MarketDailyCaptureBlocked([f"{label}_not_regular"])
    if before.st_size > _MAX_EVIDENCE_BYTES:
        raise MarketDailyCaptureBlocked([f"{label}_too_large"])
    raw = resolved.read_bytes()
    after = resolved.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != before.st_size:
        raise MarketDailyCaptureBlocked([f"{label}_changed_during_read"])
    return raw


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketDailyCaptureBlocked([f"{label}_json_invalid"]) from exc
    if not isinstance(value, dict):
        raise MarketDailyCaptureBlocked([f"{label}_not_object"])
    return value


def _write_new_file(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise MarketDailyCaptureBlocked([f"capture_path_exists:{path}"])
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _prepare_capture_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise MarketDailyCaptureBlocked(["capture_root_not_absolute"])
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise MarketDailyCaptureBlocked(["capture_root_not_directory"])
        if any(path.iterdir()):
            raise MarketDailyCaptureBlocked(["capture_root_not_empty"])
    else:
        path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)
    return path


def _prepare_market_candidate_root(
    root: str | Path,
    *,
    private_pit_generation_binding: Mapping[str, Any] | None,
) -> tuple[Path, dict[str, str] | None]:
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise MarketDailyCaptureBlocked(["candidate_root_not_absolute"])
    if path.is_symlink():
        raise MarketDailyCaptureBlocked(["candidate_root_symlink_rejected"])
    if not path.exists():
        path.mkdir(parents=True, mode=0o700)
    if not path.is_dir() or path.stat().st_uid != os.geteuid():
        raise MarketDailyCaptureBlocked(["candidate_root_not_private"])
    path.chmod(0o700)
    provided_pit = (
        _pit_reference(private_pit_generation_binding)
        if private_pit_generation_binding is not None
        else None
    )
    candidate_reference = path / "parquet" / "cn" / "reference"
    pit_is_candidate_local = False
    if provided_pit is not None:
        try:
            Path(provided_pit["generation_manifest_path"]).resolve(strict=True).relative_to(
                candidate_reference.resolve(strict=False)
            )
            Path(provided_pit["canonical_path"]).resolve(strict=True).relative_to(
                candidate_reference.resolve(strict=False)
            )
        except (OSError, RuntimeError, ValueError):
            pit_is_candidate_local = False
        else:
            pit_is_candidate_local = True
    entries = list(path.iterdir())
    if not entries:
        if pit_is_candidate_local:
            raise MarketDailyCaptureBlocked(["candidate_private_pit_store_missing"])
        return path, None

    parquet_root = path / "parquet"
    market_root = parquet_root / "cn"
    reference_root = market_root / "reference"
    if (
        {entry.name for entry in entries} != {"parquet"}
        or parquet_root.is_symlink()
        or not parquet_root.is_dir()
        or {entry.name for entry in parquet_root.iterdir()} != {"cn"}
        or market_root.is_symlink()
        or not market_root.is_dir()
        or {entry.name for entry in market_root.iterdir()} != {"reference"}
        or reference_root.is_symlink()
        or not reference_root.is_dir()
    ):
        raise MarketDailyCaptureBlocked(["candidate_root_contains_unexpected_preexisting_files"])
    if provided_pit is None:
        raise MarketDailyCaptureBlocked(["candidate_private_pit_binding_required"])
    if not pit_is_candidate_local:
        raise MarketDailyCaptureBlocked(["candidate_private_pit_binding_not_local"])
    from quant_investor.market.pit_universe import PITUniverseStore

    expected = provided_pit
    try:
        observed = PITUniverseStore(root_dir=reference_root).load_generation_binding()
    except RuntimeError as exc:
        raise MarketDailyCaptureBlocked([f"candidate_private_pit_store_invalid:{exc}"]) from exc
    observed_ref = {
        "generation_id": str(observed.get("generation_id") or ""),
        "generation_manifest_path": str(observed.get("generation_manifest_path") or ""),
        "generation_manifest_sha256": str(observed.get("generation_manifest_sha256") or ""),
        "canonical_path": str(observed.get("canonical_path") or ""),
        "canonical_sha256": str(observed.get("canonical_sha256") or ""),
    }
    if expected != observed_ref:
        raise MarketDailyCaptureBlocked(["candidate_private_pit_binding_mismatch"])
    for directory in (path, parquet_root, market_root):
        directory.chmod(0o700)
    return path, expected


def _tree_inventory(root: Path, *, label: str) -> tuple[list[Path], int]:
    if root.is_symlink() or not root.is_dir():
        raise MarketDailyCaptureBlocked([f"{label}_root_invalid"])
    files: list[Path] = []
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            if (current / name).is_symlink():
                raise MarketDailyCaptureBlocked([f"{label}_nested_symlink"])
        for name in file_names:
            path = current / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise MarketDailyCaptureBlocked([f"{label}_nonregular_file"])
            files.append(path)
            total += metadata.st_size
    if not files:
        raise MarketDailyCaptureBlocked([f"{label}_empty"])
    return sorted(files), total


def _tree_sha256(root: Path, *, label: str) -> tuple[str, int, int]:
    files, total = _tree_inventory(root, label=label)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        raw = _stable_regular_bytes(path, label=f"{label}_member")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest(), len(files), total


def _resource_preflight(*, candidate_root: Path, source_bytes: int) -> dict[str, Any]:
    disk = shutil.disk_usage(candidate_root)
    staging_bytes = source_bytes
    candidate_final_bytes = source_bytes
    rollback_or_orphan_bytes = source_bytes
    fsync_reserve = max(_FSYNC_RESERVE_BYTES, source_bytes // 20)
    before_margin = (
        source_bytes
        + staging_bytes
        + candidate_final_bytes
        + rollback_or_orphan_bytes
        + fsync_reserve
    )
    required = (before_margin * 5 + 3) // 4
    try:
        available_ram = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        physical_ram = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        available_ram = -1
        physical_ram = -1
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rlimit_as = resource.getrlimit(resource.RLIMIT_AS)
    rlimit_data = resource.getrlimit(resource.RLIMIT_DATA)
    receipt: dict[str, Any] = {
        "schema_version": "cn-market-shadow-resource-preflight.v1",
        "status": "PASSED" if disk.free >= required else "BLOCKED",
        "captured_at": _utc_now(),
        "source_snapshot_bytes": source_bytes,
        "source_capture_bytes": source_bytes,
        "staging_temp_bytes": staging_bytes,
        "candidate_final_bytes": candidate_final_bytes,
        "rollback_or_orphan_bytes": rollback_or_orphan_bytes,
        "fsync_reserve_bytes": fsync_reserve,
        "margin_numerator": 25,
        "margin_denominator": 100,
        "required_free_disk_bytes": required,
        "observed_free_disk_bytes": disk.free,
        "available_ram_bytes": available_ram,
        "physical_ram_bytes": physical_ram,
        "process_max_rss_raw": rss,
        "rlimit_as": list(rlimit_as),
        "rlimit_data": list(rlimit_data),
        "blockers": (
            [] if disk.free >= required else ["shadow_candidate_disk_headroom_insufficient"]
        ),
    }
    if receipt["status"] != "PASSED":
        raise MarketDailyCaptureBlocked(list(receipt.get("blockers") or []), receipt)
    return receipt


def _copy_snapshot_tree(source: Path, target: Path, *, label: str) -> None:
    _tree_inventory(source, label=label)
    if target.exists() or target.is_symlink():
        raise MarketDailyCaptureBlocked([f"{label}_candidate_target_exists"])
    shutil.copytree(source, target, symlinks=False, copy_function=shutil.copy2)


def _protected_refs_unchanged(refs: Sequence[Mapping[str, Any]]) -> None:
    blockers: list[str] = []
    for reference in refs:
        path = str(reference.get("path") or "")
        expected = str(reference.get("sha256") or "")
        if reference.get("kind") == "tree":
            actual, count, size = _tree_sha256(Path(path), label="protected_market_tree")
            changed = (
                actual != expected
                or count != int(reference.get("file_count", -1))
                or size != int(reference.get("size_bytes", -1))
            )
        else:
            actual = _sha256(_stable_regular_bytes(path, label="protected_market_file"))
            changed = actual != expected
        if changed:
            blockers.append(f"protected_market_bytes_changed:{Path(path).name}")
    if blockers:
        raise MarketDailyCaptureBlocked(blockers)


def _scope_symbols(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("full_a", payload.get("symbols"))
    if not isinstance(values, list):
        raise MarketDailyCaptureBlocked(["full_a_scope_missing"])
    symbols = [str(value or "").strip().upper() for value in values]
    if (
        not symbols
        or any(not _SYMBOL_RE.fullmatch(symbol) for symbol in symbols)
        or len(set(symbols)) != len(symbols)
    ):
        raise MarketDailyCaptureBlocked(["full_a_scope_invalid"])
    return sorted(symbols)


def _load_scope(scope_path: str | Path, expected_scope_sha256: str) -> tuple[list[str], bytes, str]:
    expected = _valid_sha256(expected_scope_sha256, blocker="expected_scope_sha256_invalid")
    raw = _stable_regular_bytes(scope_path, label="full_a_scope")
    actual = _sha256(raw)
    if actual != expected:
        raise MarketDailyCaptureBlocked([f"full_a_scope_sha256_mismatch:{actual}!={expected}"])
    return _scope_symbols(_json_object(raw, label="full_a_scope")), raw, actual


def _load_target_authority(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], bytes, str, str]:
    expected = _valid_sha256(expected_sha256, blocker="expected_target_authority_sha256_invalid")
    raw = _stable_regular_bytes(path, label="target_authority")
    actual = _sha256(raw)
    if actual != expected:
        raise MarketDailyCaptureBlocked(["target_authority_sha256_mismatch"])
    payload = _json_object(raw, label="target_authority")
    if payload.get("schema_version") != TARGET_AUTHORITY_SCHEMA:
        raise MarketDailyCaptureBlocked(["target_authority_schema_invalid"])
    status = str(payload.get("status") or "").strip().upper()
    if status not in {"AUTHORIZED", "PASSED", "TARGET_AUTHORIZED"}:
        raise MarketDailyCaptureBlocked(["target_authority_not_authorized"])
    target = _normalize_trade_date(
        payload.get("target_trade_date") or payload.get("target_date") or payload.get("target")
    )
    if not target:
        raise MarketDailyCaptureBlocked(["target_authority_trade_date_invalid"])
    raw_response_path = str(payload.get("raw_response_path") or "").strip()
    raw_response_sha = str(payload.get("raw_response_sha256") or "").strip().lower()
    if raw_response_path or raw_response_sha:
        if not raw_response_path or not _SHA256_RE.fullmatch(raw_response_sha):
            raise MarketDailyCaptureBlocked(["target_authority_raw_binding_invalid"])
        provider_raw = _stable_regular_bytes(
            raw_response_path, label="target_authority_raw_response"
        )
        if _sha256(provider_raw) != raw_response_sha:
            raise MarketDailyCaptureBlocked(["target_authority_raw_response_sha256_mismatch"])
    return payload, raw, actual, target


def _authority_open_trade_dates(payload: Mapping[str, Any]) -> list[str]:
    declared = payload.get("open_trade_dates")
    if isinstance(declared, list):
        values = [_normalize_trade_date(value) for value in declared]
        if any(not value for value in values) or values != sorted(set(values)):
            raise MarketDailyCaptureBlocked(["target_authority_open_dates_invalid"])
        return values
    raw_path = str(payload.get("raw_response_path") or "").strip()
    if not raw_path:
        return []
    raw = _stable_regular_bytes(raw_path, label="target_authority_raw_response")
    try:
        provider_payload = json.loads(raw.decode("utf-8"))
        data = provider_payload["data"]
        fields = data["fields"]
        items = data["items"]
        exchange_index = fields.index("exchange")
        date_index = fields.index("cal_date")
        open_index = fields.index("is_open")
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return []
    dates: list[str] = []
    for row in items:
        if (
            not isinstance(row, list)
            or len(row) != len(fields)
            or row[exchange_index] != "SSE"
            or row[open_index] not in {0, 1}
        ):
            raise MarketDailyCaptureBlocked(["target_authority_open_dates_invalid"])
        if row[open_index] == 1:
            trade_date = _normalize_trade_date(row[date_index])
            if not trade_date:
                raise MarketDailyCaptureBlocked(["target_authority_open_dates_invalid"])
            dates.append(trade_date)
    if dates != sorted(set(dates)):
        raise MarketDailyCaptureBlocked(["target_authority_open_dates_invalid"])
    return dates


def _authorized_target_window(
    *,
    authority: Mapping[str, Any],
    authority_target: str,
    requested_target_trade_dates: Sequence[str] | None,
    parent_latest_complete_trade_date: str,
    same_target_rebind: bool,
) -> tuple[list[str], str]:
    if requested_target_trade_dates is None:
        if same_target_rebind:
            raise MarketDailyCaptureBlocked(["same_target_rebind_window_required"])
        return [authority_target], ""
    requested = [_normalize_trade_date(value) for value in requested_target_trade_dates]
    if (
        not requested
        or any(not value for value in requested)
        or requested != sorted(set(requested))
        or requested[-1] != authority_target
    ):
        raise MarketDailyCaptureBlocked(["catch_up_target_window_invalid"])
    if len(requested) > 5:
        raise MarketDailyCaptureBlocked(["catch_up_window_too_large"])
    parent = _normalize_trade_date(parent_latest_complete_trade_date)
    if same_target_rebind:
        open_dates = _authority_open_trade_dates(authority)
        if (
            parent != authority_target
            or requested != [authority_target]
            or authority_target not in open_dates
        ):
            raise MarketDailyCaptureBlocked(["same_target_rebind_contract_invalid"])
        return requested, parent
    if not parent or parent >= authority_target:
        raise MarketDailyCaptureBlocked(["catch_up_parent_trade_date_invalid"])
    open_dates = _authority_open_trade_dates(authority)
    if not open_dates:
        raise MarketDailyCaptureBlocked(["target_authority_open_dates_missing"])
    expected = [trade_date for trade_date in open_dates if parent < trade_date <= authority_target]
    if len(expected) > 5:
        raise MarketDailyCaptureBlocked(["catch_up_window_too_large"])
    if requested != expected:
        raise MarketDailyCaptureBlocked(["catch_up_window_not_contiguous"])
    return requested, parent


def _pit_reference(binding: Mapping[str, Any]) -> dict[str, str]:
    manifest_path = str(binding.get("generation_manifest_path") or "").strip()
    manifest_sha = _valid_sha256(
        binding.get("generation_manifest_sha256"),
        blocker="pit_generation_manifest_sha256_invalid",
    )
    canonical_path = str(binding.get("canonical_path") or binding.get("path") or "").strip()
    canonical_sha = _valid_sha256(
        binding.get("canonical_sha256") or binding.get("sha256"),
        blocker="pit_membership_sha256_invalid",
    )
    if not manifest_path or not canonical_path:
        raise MarketDailyCaptureBlocked(["pit_generation_binding_incomplete"])
    manifest_raw = _stable_regular_bytes(manifest_path, label="pit_generation_manifest")
    canonical_raw = _stable_regular_bytes(canonical_path, label="pit_membership")
    if _sha256(manifest_raw) != manifest_sha:
        raise MarketDailyCaptureBlocked(["pit_generation_manifest_sha256_mismatch"])
    if _sha256(canonical_raw) != canonical_sha:
        raise MarketDailyCaptureBlocked(["pit_membership_sha256_mismatch"])
    return {
        "generation_id": str(binding.get("generation_id") or ""),
        "generation_manifest_path": str(Path(manifest_path).expanduser()),
        "generation_manifest_sha256": manifest_sha,
        "canonical_path": str(Path(canonical_path).expanduser()),
        "canonical_sha256": canonical_sha,
    }


def _provider_frame(provider: Any, endpoint: str, target: str) -> pd.DataFrame:
    func = getattr(provider, endpoint, None)
    if not callable(func):
        raise MarketDailyCaptureBlocked([f"{endpoint}_endpoint_unavailable"])
    fields = ",".join(_FIELDS[endpoint])
    try:
        frame = func(trade_date=target, fields=fields)
    except TypeError:
        try:
            frame = func(trade_date=target)
        except Exception as exc:
            raise MarketDailyCaptureBlocked(
                [f"{endpoint}_endpoint_error:{type(exc).__name__}"]
            ) from exc
    except Exception as exc:
        raise MarketDailyCaptureBlocked(
            [f"{endpoint}_endpoint_error:{type(exc).__name__}"]
        ) from exc
    if not isinstance(frame, pd.DataFrame):
        raise MarketDailyCaptureBlocked([f"{endpoint}_response_not_dataframe"])
    if frame.empty:
        raise MarketDailyCaptureBlocked([f"{endpoint}_response_empty"])
    return frame.copy()


def _endpoint_payload(endpoint: str, frame: pd.DataFrame, target: str) -> dict[str, Any]:
    required = list(_FIELDS[endpoint])
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise MarketDailyCaptureBlocked([f"{endpoint}_schema_missing:{','.join(missing_columns)}"])
    response_fields = [str(column) for column in frame.columns]
    raw_rows = [
        {str(column): _json_safe(value) for column, value in zip(response_fields, values)}
        for values in frame.itertuples(index=False, name=None)
    ]
    selected = frame.loc[:, required].copy()
    rows: list[dict[str, Any]] = []
    keys: list[str] = []
    blockers: list[str] = []
    for ordinal, values in enumerate(selected.itertuples(index=False, name=None)):
        row = {column: _json_safe(value) for column, value in zip(required, values)}
        symbol = str(row.get("ts_code") or "").strip().upper()
        trade_date = _normalize_trade_date(row.get("trade_date"))
        if not _SYMBOL_RE.fullmatch(symbol):
            blockers.append(f"{endpoint}_symbol_invalid:{ordinal}")
        if trade_date != target:
            blockers.append(f"{endpoint}_wrong_trade_date:{ordinal}:{trade_date}")
        row["ts_code"] = symbol
        row["trade_date"] = trade_date
        rows.append(row)
        keys.append(f"{symbol}@{trade_date}")
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicate_keys:
        blockers.append(f"{endpoint}_duplicate_keys:{len(duplicate_keys)}")
    if endpoint == "adj_factor":
        values = pd.to_numeric(selected["adj_factor"], errors="coerce")
        if values.isna().any() or values.le(0).any():
            blockers.append("adj_factor_invalid")
    if endpoint == "daily":
        for column in ("open", "high", "low", "close", "vol", "amount"):
            values = pd.to_numeric(selected[column], errors="coerce")
            invalid = values.isna() | values.lt(0)
            if column in {"open", "high", "low", "close"}:
                invalid |= values.le(0)
            if invalid.any():
                blockers.append(f"daily_numeric_invalid:{column}")
        numeric = selected[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        if (
            numeric["high"].lt(numeric[["open", "low", "close"]].max(axis=1)).any()
            or numeric["low"].gt(numeric[["open", "high", "close"]].min(axis=1)).any()
        ):
            blockers.append("daily_ohlc_relationship_invalid")
    if endpoint == "daily_basic":
        for column in ("total_mv", "circ_mv"):
            values = pd.to_numeric(selected[column], errors="coerce")
            if values.isna().any() or values.lt(0).any():
                blockers.append(f"daily_basic_numeric_invalid:{column}")
    if blockers:
        raise MarketDailyCaptureBlocked(blockers)
    return {
        "schema_version": ENDPOINT_SCHEMA,
        "endpoint": endpoint,
        "target_trade_date": target,
        "fields": required,
        "response_fields": response_fields,
        "raw_row_count": len(raw_rows),
        "raw_provider_order_sha256": _sha256(_canonical_json_bytes(raw_rows)),
        "raw_rows": raw_rows,
        "row_count": len(rows),
        "provider_order_sha256": _sha256(_canonical_json_bytes(rows)),
        "keyset_sha256": _sha256("\n".join(sorted(keys)).encode("utf-8")),
        "keys": keys,
        "rows": rows,
    }


def _reason_sets(value: Mapping[str, Sequence[str]] | None) -> dict[str, set[str]]:
    raw = dict(value or {})
    result: dict[str, set[str]] = {}
    for reason in _CLASSIFICATION_REASONS[1:-1]:
        result[reason] = {
            str(symbol or "").strip().upper()
            for symbol in raw.get(reason, []) or []
            if str(symbol or "").strip()
        }
    return result


def _classification_evidence_refs(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for reason, raw_reference in dict(value or {}).items():
        if reason not in _CLASSIFICATION_REASONS[1:-1] or not isinstance(raw_reference, Mapping):
            raise MarketDailyCaptureBlocked(["classification_evidence_invalid"])
        reference = dict(raw_reference)
        path = str(reference.get("path") or "").strip()
        expected = _valid_sha256(
            reference.get("sha256"),
            blocker=f"{reason}_evidence_sha256_invalid",
        )
        raw = _stable_regular_bytes(path, label=f"{reason}_evidence")
        if _sha256(raw) != expected:
            raise MarketDailyCaptureBlocked([f"{reason}_evidence_sha256_mismatch"])
        refs[reason] = {
            "path": str(Path(path).expanduser()),
            "sha256": expected,
        }
        if "payload_sha256" in reference:
            refs[reason]["payload_sha256"] = _valid_sha256(
                reference["payload_sha256"],
                blocker=f"{reason}_evidence_payload_sha256_invalid",
            )
        for field in ("inferred_dates", "terminal_provider_proof"):
            if field in reference:
                refs[reason][field] = reference[field]
    return refs


def classify_market_target(
    *,
    scope_symbols: Sequence[str],
    observed_symbols: Sequence[str],
    reason_sets: Mapping[str, Sequence[str]] | None,
    target_trade_date: str,
) -> dict[str, Any]:
    """Return a complete, mutually exclusive target-day classification."""

    scope = set(scope_symbols)
    observed = set(observed_symbols)
    reasons = _reason_sets(reason_sets)
    blockers: list[str] = []
    if observed - scope:
        blockers.append(f"daily_out_of_scope:{len(observed - scope)}")
    for reason, symbols in reasons.items():
        if symbols - scope:
            blockers.append(f"{reason}_out_of_scope:{len(symbols - scope)}")
        if symbols & observed:
            blockers.append(f"{reason}_observed_conflict:{len(symbols & observed)}")
    absent = scope - observed
    classified: set[str] = set()
    output: dict[str, list[str]] = {"observed": sorted(observed & scope)}
    for reason in _CLASSIFICATION_REASONS[1:-1]:
        members = reasons[reason] & absent
        overlap = members & classified
        if overlap:
            blockers.append(f"classification_overlap:{reason}:{len(overlap)}")
        output[reason] = sorted(members)
        classified |= members
    true_missing = absent - classified
    output["true_missing"] = sorted(true_missing)
    if true_missing:
        blockers.append(f"true_missing:{len(true_missing)}")
    union: set[str] = set()
    for reason in _CLASSIFICATION_REASONS:
        members = set(output[reason])
        if members & union:
            blockers.append("classification_sets_not_disjoint")
        union |= members
    if union != scope:
        blockers.append("classification_scope_not_closed")
    return {
        "schema_version": CLASSIFICATION_SCHEMA,
        "target_trade_date": target_trade_date,
        "status": "PASSED" if not blockers else "BLOCKED",
        "classification_sets_disjoint": not any(
            item.startswith("classification_") for item in blockers
        ),
        "expected_scope_count": len(scope),
        "coverage_complete_count": len(scope) - len(true_missing),
        "observed_bar_count": len(output["observed"]),
        "counts": {reason: len(output[reason]) for reason in _CLASSIFICATION_REASONS},
        "symbols": output,
        "blockers": list(dict.fromkeys(blockers)),
    }


def validate_market_endpoint_frames(
    *,
    daily: pd.DataFrame,
    daily_basic: pd.DataFrame,
    adj_factor: pd.DataFrame,
    target_trade_date: str,
    scope_symbols: Sequence[str],
    reason_sets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Validate raw endpoint frames without dropping duplicate observations."""

    target = _normalize_trade_date(target_trade_date)
    if not target:
        raise MarketDailyCaptureBlocked(["target_trade_date_invalid"])
    payloads = {
        "daily": _endpoint_payload("daily", daily, target),
        "daily_basic": _endpoint_payload("daily_basic", daily_basic, target),
        "adj_factor": _endpoint_payload("adj_factor", adj_factor, target),
    }
    keysets = {name: set(payload["keys"]) for name, payload in payloads.items()}
    blockers: list[str] = []
    for endpoint in ("daily_basic", "adj_factor"):
        missing = keysets["daily"] - keysets[endpoint]
        extra = keysets[endpoint] - keysets["daily"]
        if missing:
            blockers.append(f"{endpoint}_keyset_missing:{len(missing)}")
        if extra:
            blockers.append(f"{endpoint}_keyset_extra:{len(extra)}")
    observed = [key.split("@", 1)[0] for key in payloads["daily"]["keys"]]
    classification = classify_market_target(
        scope_symbols=scope_symbols,
        observed_symbols=observed,
        reason_sets=reason_sets,
        target_trade_date=target,
    )
    blockers.extend(classification["blockers"])
    if blockers:
        raise MarketDailyCaptureBlocked(blockers)
    return {"payloads": payloads, "classification": classification}


def _write_blocked_receipt(root: Path, blockers: Sequence[str], target: str = "") -> dict[str, Any]:
    receipt = {
        "schema_version": CAPTURE_SCHEMA,
        "status": "BLOCKED",
        "captured_at": _utc_now(),
        "target_trade_date": target,
        "blockers": list(dict.fromkeys(str(item) for item in blockers)),
        "canonical_write_authorized": False,
    }
    path = root / "blocked.json"
    if not path.exists():
        _write_new_file(path, _canonical_json_bytes(receipt))
    receipt["receipt_path"] = str(path)
    receipt["receipt_sha256"] = _sha256(path.read_bytes())
    return receipt


def _mapping_for_trade_date(
    value: Mapping[str, Any] | None,
    *,
    trade_date: str,
    target_trade_dates: Sequence[str],
) -> Mapping[str, Any]:
    raw = dict(value or {})
    if (
        raw
        and set(raw).issubset(set(target_trade_dates))
        and all(isinstance(item, Mapping) for item in raw.values())
    ):
        selected = raw.get(trade_date, {})
        return dict(selected) if isinstance(selected, Mapping) else {}
    return raw


def capture_market_daily(
    *,
    provider: Any,
    capture_root: str | Path,
    target_authority_path: str | Path,
    expected_target_authority_sha256: str,
    scope_path: str | Path,
    expected_scope_sha256: str,
    pit_generation_binding: Mapping[str, Any],
    expected_market_pointer_sha256: str,
    reason_sets: Mapping[str, Any] | None = None,
    classification_evidence: Mapping[str, Any] | None = None,
    target_trade_dates: Sequence[str] | None = None,
    parent_latest_complete_trade_date: str = "",
    same_target_rebind: bool = False,
) -> dict[str, Any]:
    """Acquire and seal one authorized 1-5 session CN market window."""

    root = _prepare_capture_root(capture_root)
    target = ""
    try:
        authority, authority_raw, authority_sha, target = _load_target_authority(
            target_authority_path, expected_target_authority_sha256
        )
        ordered_targets, parent_date = _authorized_target_window(
            authority=authority,
            authority_target=target,
            requested_target_trade_dates=target_trade_dates,
            parent_latest_complete_trade_date=(parent_latest_complete_trade_date),
            same_target_rebind=bool(same_target_rebind),
        )
        scope, scope_raw, scope_sha = _load_scope(scope_path, expected_scope_sha256)
        pit = _pit_reference(pit_generation_binding)
        market_sha = _valid_sha256(
            expected_market_pointer_sha256,
            blocker="expected_market_pointer_sha256_invalid",
        )
        sessions: list[dict[str, Any]] = []
        for trade_date in ordered_targets:
            frames = {
                endpoint: _provider_frame(provider, endpoint, trade_date) for endpoint in _ENDPOINTS
            }
            session_reasons = _mapping_for_trade_date(
                reason_sets,
                trade_date=trade_date,
                target_trade_dates=ordered_targets,
            )
            validated = validate_market_endpoint_frames(
                daily=frames["daily"],
                daily_basic=frames["daily_basic"],
                adj_factor=frames["adj_factor"],
                target_trade_date=trade_date,
                scope_symbols=scope,
                reason_sets=session_reasons,
            )
            session_evidence = _mapping_for_trade_date(
                classification_evidence,
                trade_date=trade_date,
                target_trade_dates=ordered_targets,
            )
            evidence_refs = _classification_evidence_refs(
                session_evidence  # type: ignore[arg-type]
            )
            if (
                validated["classification"]["counts"]["non_trading"]
                and "non_trading" not in evidence_refs
            ):
                raise MarketDailyCaptureBlocked(["non_trading_classification_evidence_required"])
            terminal_reference = dict(evidence_refs.get("delisted") or {})
            if terminal_reference.get("terminal_provider_proof") is True and (
                not terminal_reference.get("payload_sha256")
                or not isinstance(terminal_reference.get("inferred_dates"), Mapping)
            ):
                raise MarketDailyCaptureBlocked(["terminal_delisting_evidence_incomplete"])
            endpoint_refs: dict[str, dict[str, Any]] = {}
            endpoint_root = root if len(ordered_targets) == 1 else root / trade_date
            for endpoint in _ENDPOINTS:
                path = endpoint_root / f"{endpoint}.json"
                raw = _canonical_json_bytes(validated["payloads"][endpoint])
                _write_new_file(path, raw)
                endpoint_refs[endpoint] = {
                    "path": str(path),
                    "sha256": _sha256(raw),
                    "row_count": validated["payloads"][endpoint]["row_count"],
                    "provider_order_sha256": validated["payloads"][endpoint][
                        "provider_order_sha256"
                    ],
                    "keyset_sha256": validated["payloads"][endpoint]["keyset_sha256"],
                }
            sessions.append(
                {
                    "trade_date": trade_date,
                    "endpoints": endpoint_refs,
                    "classification": validated["classification"],
                    "classification_evidence": evidence_refs,
                }
            )
        latest_session = sessions[-1]
        manifest = {
            "schema_version": CAPTURE_SCHEMA,
            "status": "CAPTURED",
            "market": "CN",
            "captured_at": _utc_now(),
            "target_trade_date": target,
            "target_trade_dates": ordered_targets,
            "parent_latest_complete_trade_date": parent_date,
            "same_target_rebind": bool(same_target_rebind),
            "target_authority": {
                "path": str(Path(target_authority_path).expanduser()),
                "sha256": authority_sha,
                "captured_bytes_sha256": _sha256(authority_raw),
            },
            "scope": {
                "path": str(Path(scope_path).expanduser()),
                "sha256": scope_sha,
                "captured_bytes_sha256": _sha256(scope_raw),
                "symbol_count": len(scope),
                "symbol_set_sha256": _sha256("\n".join(scope).encode("utf-8")),
            },
            "pit_generation": pit,
            "expected_market_pointer_sha256": market_sha,
            "sessions": sessions,
            "endpoints": latest_session["endpoints"],
            "classification": latest_session["classification"],
            "classification_evidence": latest_session["classification_evidence"],
            "provider_accounting": {
                "calls": len(_ENDPOINTS) * len(ordered_targets),
                "failed": 0,
                "malformed": 0,
                "has_more": False,
            },
            "canonical_write_authorized": False,
            "blockers": [],
        }
        manifest_path = root / "manifest.json"
        manifest_raw = _canonical_json_bytes(manifest)
        _write_new_file(manifest_path, manifest_raw)
        return {
            "schema_version": CAPTURE_SCHEMA,
            "status": "CAPTURED",
            "target_trade_date": target,
            "target_trade_dates": ordered_targets,
            "same_target_rebind": bool(same_target_rebind),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_raw),
            "provider_accounting": manifest["provider_accounting"],
            "classification": manifest["classification"],
            "canonical_write_authorized": False,
            "blockers": [],
        }
    except MarketDailyCaptureBlocked as exc:
        receipt = _write_blocked_receipt(root, exc.blockers, target)
        raise MarketDailyCaptureBlocked(exc.blockers, receipt) from exc


def replay_market_daily_capture(
    *,
    capture_manifest_path: str | Path,
    expected_capture_manifest_sha256: str,
    scope_path: str | Path,
    expected_scope_sha256: str,
    pit_generation_binding: Mapping[str, Any],
    expected_market_pointer_sha256: str,
) -> dict[str, Any]:
    """Replay every captured byte and semantic gate without a provider call."""

    manifest_expected = _valid_sha256(
        expected_capture_manifest_sha256,
        blocker="expected_capture_manifest_sha256_invalid",
    )
    manifest_raw = _stable_regular_bytes(capture_manifest_path, label="capture_manifest")
    if _sha256(manifest_raw) != manifest_expected:
        raise MarketDailyCaptureBlocked(["capture_manifest_sha256_mismatch"])
    manifest = _json_object(manifest_raw, label="capture_manifest")
    if manifest.get("schema_version") != CAPTURE_SCHEMA or manifest.get("status") != "CAPTURED":
        raise MarketDailyCaptureBlocked(["capture_manifest_schema_or_status_invalid"])
    target = _normalize_trade_date(manifest.get("target_trade_date"))
    target_reference = dict(manifest.get("target_authority") or {})
    target_payload, _target_raw, target_sha, authority_target = _load_target_authority(
        target_reference.get("path") or "",
        target_reference.get("sha256") or "",
    )
    if target_sha != target_reference.get("captured_bytes_sha256") or (authority_target != target):
        raise MarketDailyCaptureBlocked(["capture_target_authority_mismatch"])
    raw_targets = manifest.get("target_trade_dates") or [target]
    if not isinstance(raw_targets, list):
        raise MarketDailyCaptureBlocked(["capture_target_window_schema_invalid"])
    ordered_targets = [_normalize_trade_date(value) for value in raw_targets]
    parent_date = _normalize_trade_date(manifest.get("parent_latest_complete_trade_date"))
    same_target_rebind = manifest.get("same_target_rebind", False)
    if type(same_target_rebind) is not bool:
        raise MarketDailyCaptureBlocked(["same_target_rebind_schema_invalid"])
    if parent_date:
        replayed_targets, replayed_parent = _authorized_target_window(
            authority=target_payload,
            authority_target=target,
            requested_target_trade_dates=ordered_targets,
            parent_latest_complete_trade_date=parent_date,
            same_target_rebind=same_target_rebind,
        )
        if replayed_targets != ordered_targets or replayed_parent != parent_date:
            raise MarketDailyCaptureBlocked(["capture_target_window_replay_mismatch"])
    elif ordered_targets != [target]:
        raise MarketDailyCaptureBlocked(["capture_target_window_parent_missing"])
    scope, _scope_raw, scope_sha = _load_scope(scope_path, expected_scope_sha256)
    manifest_scope = dict(manifest.get("scope") or {})
    if (
        str(Path(scope_path).expanduser()) != str(manifest_scope.get("path") or "")
        or scope_sha != str(manifest_scope.get("sha256") or "")
        or _sha256("\n".join(scope).encode("utf-8"))
        != str(manifest_scope.get("symbol_set_sha256") or "")
    ):
        raise MarketDailyCaptureBlocked(["capture_scope_binding_mismatch"])
    pit = _pit_reference(pit_generation_binding)
    if pit != dict(manifest.get("pit_generation") or {}):
        raise MarketDailyCaptureBlocked(["capture_pit_binding_mismatch"])
    market_sha = _valid_sha256(
        expected_market_pointer_sha256,
        blocker="expected_market_pointer_sha256_invalid",
    )
    if market_sha != str(manifest.get("expected_market_pointer_sha256") or ""):
        raise MarketDailyCaptureBlocked(["capture_market_pointer_binding_mismatch"])
    raw_sessions = manifest.get("sessions")
    if not isinstance(raw_sessions, list):
        raw_sessions = [
            {
                "trade_date": target,
                "endpoints": manifest.get("endpoints") or {},
                "classification": manifest.get("classification") or {},
                "classification_evidence": manifest.get("classification_evidence") or {},
            }
        ]
    if [
        _normalize_trade_date(item.get("trade_date"))
        for item in raw_sessions
        if isinstance(item, Mapping)
    ] != ordered_targets:
        raise MarketDailyCaptureBlocked(["capture_session_window_mismatch"])
    frames_by_endpoint: dict[str, list[pd.DataFrame]] = {endpoint: [] for endpoint in _ENDPOINTS}
    classifications: list[dict[str, Any]] = []
    for session in raw_sessions:
        if not isinstance(session, Mapping):
            raise MarketDailyCaptureBlocked(["capture_session_schema_invalid"])
        session_date = _normalize_trade_date(session.get("trade_date"))
        session_frames: dict[str, pd.DataFrame] = {}
        for endpoint in _ENDPOINTS:
            reference = dict((session.get("endpoints") or {}).get(endpoint) or {})
            raw = _stable_regular_bytes(
                reference.get("path") or "",
                label=f"capture_{session_date}_{endpoint}",
            )
            if _sha256(raw) != str(reference.get("sha256") or ""):
                raise MarketDailyCaptureBlocked([f"capture_{endpoint}_sha256_mismatch"])
            payload = _json_object(raw, label=f"capture_{session_date}_{endpoint}")
            if (
                payload.get("schema_version") != ENDPOINT_SCHEMA
                or payload.get("endpoint") != endpoint
                or payload.get("target_trade_date") != session_date
                or payload.get("fields") != list(_FIELDS[endpoint])
                or int(payload.get("raw_row_count", -1)) != len(payload.get("raw_rows") or [])
                or payload.get("raw_provider_order_sha256")
                != _sha256(_canonical_json_bytes(payload.get("raw_rows") or []))
                or int(payload.get("row_count", -1)) != len(payload.get("rows") or [])
                or payload.get("provider_order_sha256")
                != _sha256(_canonical_json_bytes(payload.get("rows") or []))
            ):
                raise MarketDailyCaptureBlocked([f"capture_{endpoint}_semantic_mismatch"])
            payload_keys = [
                f"{row.get('ts_code')}@{row.get('trade_date')}" for row in payload.get("rows") or []
            ]
            if (
                payload.get("keys") != payload_keys
                or payload.get("keyset_sha256")
                != _sha256("\n".join(sorted(payload_keys)).encode("utf-8"))
                or int(reference.get("row_count", -1)) != len(payload_keys)
                or reference.get("provider_order_sha256") != payload.get("provider_order_sha256")
                or reference.get("keyset_sha256") != payload.get("keyset_sha256")
            ):
                raise MarketDailyCaptureBlocked([f"capture_{endpoint}_keyset_or_order_mismatch"])
            session_frames[endpoint] = pd.DataFrame(payload.get("rows") or [])
            frames_by_endpoint[endpoint].append(session_frames[endpoint])
        session_classification = dict(session.get("classification") or {})
        session_reasons = dict(session_classification.get("symbols") or {})
        session_reasons.pop("observed", None)
        session_reasons.pop("true_missing", None)
        validated = validate_market_endpoint_frames(
            daily=session_frames["daily"],
            daily_basic=session_frames["daily_basic"],
            adj_factor=session_frames["adj_factor"],
            target_trade_date=session_date,
            scope_symbols=scope,
            reason_sets=session_reasons,
        )
        if validated["classification"] != session_classification:
            raise MarketDailyCaptureBlocked(["capture_classification_replay_mismatch"])
        evidence_refs = _classification_evidence_refs(session.get("classification_evidence") or {})
        if evidence_refs != dict(session.get("classification_evidence") or {}):
            raise MarketDailyCaptureBlocked(["capture_classification_evidence_replay_mismatch"])
        if (
            validated["classification"]["counts"]["non_trading"]
            and "non_trading" not in evidence_refs
        ):
            raise MarketDailyCaptureBlocked(["non_trading_classification_evidence_required"])
        classifications.append(validated["classification"])
    frames = {
        endpoint: pd.concat(parts, ignore_index=True)
        for endpoint, parts in frames_by_endpoint.items()
    }
    if (
        manifest.get("endpoints") != raw_sessions[-1].get("endpoints")
        or manifest.get("classification") != classifications[-1]
    ):
        raise MarketDailyCaptureBlocked(["capture_latest_session_projection_mismatch"])
    return {
        "schema_version": CAPTURE_SCHEMA,
        "status": "REPLAYED",
        "target_trade_date": target,
        "target_trade_dates": ordered_targets,
        "parent_latest_complete_trade_date": parent_date,
        "same_target_rebind": same_target_rebind,
        "manifest": manifest,
        "frames": frames,
        "scope_symbols": scope,
        "classification": classifications[-1],
        "session_classifications": classifications,
        "expected_market_pointer_sha256": market_sha,
    }


def _bars_frame(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    daily = frames["daily"].copy()
    adj = frames["adj_factor"].copy()
    basic = frames["daily_basic"].copy()
    bars = daily.merge(adj, on=["ts_code", "trade_date"], how="inner", validate="one_to_one")
    bars = bars.merge(basic, on=["ts_code", "trade_date"], how="inner", validate="one_to_one")
    for column in ("open", "high", "low", "close"):
        bars[f"adj_{column}"] = pd.to_numeric(bars[column], errors="coerce") * pd.to_numeric(
            bars["adj_factor"], errors="coerce"
        )
    return bars.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def build_private_market_candidate(
    *,
    production_data_root: str | Path,
    candidate_data_root: str | Path,
    expected_production_market_pointer_sha256: str,
    private_pit_generation_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Clone the active v4 snapshot behind a private, rewritten pointer.

    The source snapshot is copied, never linked, so a later candidate write
    cannot mutate protected canonical bytes through a shared inode.
    """

    production = Path(production_data_root).expanduser()
    candidate = Path(candidate_data_root).expanduser()
    if not production.is_absolute() or not candidate.is_absolute():
        raise MarketDailyCaptureBlocked(["candidate_and_production_roots_must_be_absolute"])
    production_resolved = production.resolve(strict=True)
    candidate_resolved = candidate.resolve(strict=False)
    if production_resolved == candidate_resolved:
        raise MarketDailyCaptureBlocked(["candidate_root_matches_production_root"])
    for child, parent, blocker in (
        (
            candidate_resolved,
            production_resolved,
            "candidate_root_nested_in_production_root",
        ),
        (
            production_resolved,
            candidate_resolved,
            "production_root_nested_in_candidate_root",
        ),
    ):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise MarketDailyCaptureBlocked([blocker])

    expected_pointer_sha = _valid_sha256(
        expected_production_market_pointer_sha256,
        blocker="expected_production_market_pointer_sha256_invalid",
    )
    source_store = MarketDataStore(market="CN", data_root=production)
    source_validation = source_store.validate_latest()
    if source_validation.get("status") != "passed":
        raise MarketDailyCaptureBlocked(["production_market_snapshot_not_healthy"])
    pointer_path = production / "parquet" / "cn" / "_latest.json"
    pointer_raw = _stable_regular_bytes(pointer_path, label="production_market_pointer")
    if _sha256(pointer_raw) != expected_pointer_sha:
        raise MarketDailyCaptureBlocked(["production_market_pointer_sha256_mismatch"])
    pointer = _json_object(pointer_raw, label="production_market_pointer")
    if (
        pointer.get("status") != "OK"
        or dict(pointer.get("coverage") or {}).get("coverage_schema_version")
        != "cn-full-a-coverage.v4"
    ):
        raise MarketDailyCaptureBlocked(["production_market_pointer_not_exact_v4"])
    source_snapshot = source_store.reader._snapshot_from_payload(pointer)
    manifest_path = source_snapshot.manifest_path
    manifest_raw = _stable_regular_bytes(manifest_path, label="production_market_manifest")
    manifest = _json_object(manifest_raw, label="production_market_manifest")
    if str(manifest.get("snapshot_id") or "") != source_snapshot.snapshot_id or dict(
        manifest.get("coverage") or {}
    ) != dict(pointer.get("coverage") or {}):
        raise MarketDailyCaptureBlocked(["production_market_pointer_manifest_mismatch"])
    table_sha, table_file_count, table_bytes = _tree_sha256(
        source_snapshot.table_root, label="production_market_table"
    )
    serving_sha, serving_file_count, serving_bytes = _tree_sha256(
        source_snapshot.serving_root, label="production_market_serving"
    )
    root, private_pit = _prepare_market_candidate_root(
        candidate,
        private_pit_generation_binding=private_pit_generation_binding,
    )
    resource_receipt = _resource_preflight(
        candidate_root=root,
        source_bytes=table_bytes + serving_bytes,
    )
    resource_path = root / "resource-preflight.json"
    resource_raw = _canonical_json_bytes(resource_receipt)
    _write_new_file(resource_path, resource_raw)

    snapshot_id = source_snapshot.snapshot_id
    candidate_market_root = root / "parquet" / "cn"
    candidate_snapshot_dir = candidate_market_root / "_snapshots" / snapshot_id
    candidate_table = candidate_snapshot_dir / "table" / "bars"
    candidate_serving = candidate_snapshot_dir / "serving" / "bars"
    candidate_manifest_path = candidate_market_root / "_snapshots" / f"{snapshot_id}.json"
    candidate_pointer_path = candidate_market_root / "_latest.json"
    _copy_snapshot_tree(
        source_snapshot.table_root,
        candidate_table,
        label="production_market_table",
    )
    _copy_snapshot_tree(
        source_snapshot.serving_root,
        candidate_serving,
        label="production_market_serving",
    )

    candidate_manifest = dict(manifest)
    candidate_manifest["manifest_path"] = str(candidate_manifest_path)
    candidate_manifest["table_root"] = str(candidate_table)
    candidate_manifest["derived_serving_root"] = str(candidate_serving)
    candidate_coverage = dict(candidate_manifest.get("coverage") or {})
    if private_pit is not None:
        candidate_coverage.update(
            {
                "pit_generation_id": private_pit["generation_id"],
                "pit_generation_manifest_path": private_pit["generation_manifest_path"],
                "pit_generation_manifest_sha256": private_pit["generation_manifest_sha256"],
                "pit_membership_path": private_pit["canonical_path"],
                "pit_membership_sha256": private_pit["canonical_sha256"],
            }
        )
    candidate_manifest["coverage"] = candidate_coverage
    candidate_manifest_raw = _canonical_json_bytes(candidate_manifest)
    _write_new_file(candidate_manifest_path, candidate_manifest_raw)

    candidate_pointer = dict(pointer)
    candidate_pointer["manifest_path"] = str(candidate_manifest_path)
    candidate_pointer["table_root"] = str(candidate_table)
    candidate_pointer["derived_serving_root"] = str(candidate_serving)
    candidate_pointer["coverage"] = candidate_coverage
    candidate_pointer_raw = _canonical_json_bytes(candidate_pointer)
    _write_new_file(candidate_pointer_path, candidate_pointer_raw)

    candidate_validation = MarketDataStore(market="CN", data_root=root).validate_latest()
    if candidate_validation.get("status") != "passed":
        raise MarketDailyCaptureBlocked(
            [
                "private_market_candidate_readback_failed:"
                + ",".join(candidate_validation.get("blockers") or [])
            ]
        )
    protected_refs: list[dict[str, Any]] = [
        {
            "path": str(pointer_path),
            "sha256": expected_pointer_sha,
            "role": "production_market_pointer",
        },
        {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_raw),
            "role": "production_market_manifest",
        },
        {
            "path": str(source_snapshot.table_root),
            "sha256": table_sha,
            "file_count": table_file_count,
            "size_bytes": table_bytes,
            "kind": "tree",
            "role": "production_market_table",
        },
        {
            "path": str(source_snapshot.serving_root),
            "sha256": serving_sha,
            "file_count": serving_file_count,
            "size_bytes": serving_bytes,
            "kind": "tree",
            "role": "production_market_serving",
        },
    ]
    health_path = production / "parquet" / "cn" / "_health_ledger.jsonl"
    if health_path.exists():
        health_raw = _stable_regular_bytes(health_path, label="production_market_health_ledger")
        protected_refs.append(
            {
                "path": str(health_path),
                "sha256": _sha256(health_raw),
                "role": "production_market_health_ledger",
            }
        )
    protected_private_pit_refs: list[dict[str, Any]] = []
    if private_pit is not None:
        for role, pit_path, expected_sha in (
            (
                "candidate_pit_manifest",
                private_pit["generation_manifest_path"],
                private_pit["generation_manifest_sha256"],
            ),
            (
                "candidate_pit_membership",
                private_pit["canonical_path"],
                private_pit["canonical_sha256"],
            ),
        ):
            protected_private_pit_refs.append(
                {
                    "path": pit_path,
                    "sha256": expected_sha,
                    "role": role,
                }
            )
        discovery_path = (
            root / "parquet" / "cn" / "reference" / "stock_basic_membership_latest.json"
        )
        discovery_raw = _stable_regular_bytes(
            discovery_path, label="candidate_pit_discovery_pointer"
        )
        protected_private_pit_refs.append(
            {
                "path": str(discovery_path),
                "sha256": _sha256(discovery_raw),
                "role": "candidate_pit_discovery_pointer",
            }
        )
    _protected_refs_unchanged(protected_refs)
    _protected_refs_unchanged(protected_private_pit_refs)
    return {
        "schema_version": "cn-market-private-candidate.v1",
        "status": "READY",
        "candidate_data_root": str(root),
        "candidate_pointer_path": str(candidate_pointer_path),
        "candidate_pointer_sha256": _sha256(candidate_pointer_raw),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": _sha256(candidate_manifest_raw),
        "source_snapshot_id": snapshot_id,
        "source_latest_complete_trade_date": source_snapshot.latest_complete_trade_date,
        "private_pit_generation": private_pit or {},
        "protected_production_refs": protected_refs,
        "protected_candidate_pit_refs": protected_private_pit_refs,
        "resource_preflight": {
            **resource_receipt,
            "path": str(resource_path),
            "sha256": _sha256(resource_raw),
        },
        "readback": candidate_validation,
        "canonical_write_performed": False,
        "blockers": [],
    }


def publish_market_daily_capture(
    *,
    capture_manifest_path: str | Path,
    expected_capture_manifest_sha256: str,
    data_root: str | Path,
    scope_path: str | Path,
    expected_scope_sha256: str,
    pit_generation_binding: Mapping[str, Any],
    expected_market_pointer_sha256: str,
    publication_expected_market_pointer_sha256: str = "",
    store: MarketDataStore | None = None,
    publication_mode: str = "execute",
) -> dict[str, Any]:
    """Publish a fully replayed capture through the existing locked Market CAS."""

    replay = replay_market_daily_capture(
        capture_manifest_path=capture_manifest_path,
        expected_capture_manifest_sha256=expected_capture_manifest_sha256,
        scope_path=scope_path,
        expected_scope_sha256=expected_scope_sha256,
        pit_generation_binding=pit_generation_binding,
        expected_market_pointer_sha256=expected_market_pointer_sha256,
    )
    target = replay["target_trade_date"]
    ordered_targets = list(replay["target_trade_dates"])
    if (
        not ordered_targets
        or len(ordered_targets) > 5
        or ordered_targets != sorted(set(ordered_targets))
        or ordered_targets[-1] != target
    ):
        raise MarketDailyCaptureBlocked(["publication_target_window_invalid"])
    same_target_rebind = replay.get("same_target_rebind") is True
    if same_target_rebind and (
        ordered_targets != [target] or replay.get("parent_latest_complete_trade_date") != target
    ):
        raise MarketDailyCaptureBlocked(["publication_same_target_rebind_invalid"])
    classification = replay["classification"]
    if classification["counts"]["true_missing"] != 0:
        raise MarketDailyCaptureBlocked(["true_missing_nonzero_precommit"])
    publication_parent_sha = (
        _valid_sha256(
            publication_expected_market_pointer_sha256,
            blocker="publication_expected_market_pointer_sha256_invalid",
        )
        if str(publication_expected_market_pointer_sha256 or "").strip()
        else replay["expected_market_pointer_sha256"]
    )
    resolved_store = store or MarketDataStore(market="CN", data_root=data_root)
    frozen_parent_date = str(replay.get("parent_latest_complete_trade_date") or "")
    if frozen_parent_date:
        parent_readback = resolved_store.validate_latest()
        observed_parent_date = _normalize_trade_date(
            parent_readback.get("latest_complete_trade_date")
            or parent_readback.get("latest_trade_date")
        )
        if parent_readback.get("status") != "passed" or observed_parent_date != frozen_parent_date:
            raise MarketDailyCaptureBlocked(["publication_parent_trade_date_mismatch"])
    coverage_symbols = dict(classification["symbols"])
    classification_evidence = dict(replay["manifest"].get("classification_evidence") or {})
    nontrading_evidence = dict(classification_evidence.get("non_trading") or {})
    delisted_evidence = dict(classification_evidence.get("delisted") or {})
    verified_terminal_delisting = (
        coverage_symbols["delisted"]
        if delisted_evidence.get("terminal_provider_proof") is True
        else []
    )
    inactive_symbols = sorted(
        set(coverage_symbols["inactive"])
        | set(coverage_symbols["delisted"])
        | set(coverage_symbols["prelisting"])
    )
    non_blocking_absent = sorted(
        set(coverage_symbols["suspended"])
        | set(inactive_symbols)
        | set(coverage_symbols["non_trading"])
    )
    metadata = {
        "schema_version": "cn-market-daily-publication.v1",
        "status": "OK",
        "storage_mode": "parquet-direct",
        "publication_mode": str(publication_mode),
        "target_trade_date": target,
        "target_trade_dates": ordered_targets,
        "same_target_rebind": same_target_rebind,
        "capture_manifest_path": str(Path(capture_manifest_path).expanduser()),
        "capture_manifest_sha256": expected_capture_manifest_sha256,
        "capture_expected_market_pointer_sha256": replay["expected_market_pointer_sha256"],
        "publication_expected_market_pointer_sha256": publication_parent_sha,
        "coverage": {
            "coverage_schema_version": "cn-full-a-coverage.v4",
            "complete": True,
            "coverage_ratio": 1.0,
            "coverage_complete_count": classification["coverage_complete_count"],
            "expected_scope_count": classification["expected_scope_count"],
            "observed_bar_count": classification["observed_bar_count"],
            "blocking_incomplete_count": 0,
            "categories_checked": ["full_a"],
            "latest_available_trade_date": target,
            "latest_complete_trade_date": target,
            "upsert_target_trade_date": target,
            "coverage_trade_date": target,
            "expected_scope_sha256": replay["manifest"]["scope"]["symbol_set_sha256"],
            "suspended_symbols": coverage_symbols["suspended"],
            "inactive_symbols": inactive_symbols,
            "verified_terminal_delisting_symbols": verified_terminal_delisting,
            "verified_terminal_delisting_evidence_path": str(delisted_evidence.get("path") or ""),
            "verified_terminal_delisting_evidence_sha256": str(
                delisted_evidence.get("sha256") or ""
            ),
            "verified_terminal_delisting_payload_sha256": str(
                delisted_evidence.get("payload_sha256") or ""
            ),
            "verified_terminal_delisting_inferred_dates": dict(
                delisted_evidence.get("inferred_dates") or {}
            ),
            "verified_nontrading_bak_daily_zero_symbols": coverage_symbols["non_trading"],
            "verified_nontrading_evidence_path": str(nontrading_evidence.get("path") or ""),
            "verified_nontrading_evidence_sha256": str(nontrading_evidence.get("sha256") or ""),
            "allowed_stale_symbols": [],
            "non_blocking_absent_symbols": non_blocking_absent,
            "non_trading_symbols": coverage_symbols["non_trading"],
            "delisted_symbols": coverage_symbols["delisted"],
            "prelisting_symbols": coverage_symbols["prelisting"],
            "true_missing_symbols": [],
            "classification_sets_disjoint": True,
            "pit_membership_path": replay["manifest"]["pit_generation"]["canonical_path"],
            "pit_membership_sha256": replay["manifest"]["pit_generation"]["canonical_sha256"],
            "pit_generation_id": replay["manifest"]["pit_generation"]["generation_id"],
            "pit_generation_manifest_path": replay["manifest"]["pit_generation"][
                "generation_manifest_path"
            ],
            "pit_generation_manifest_sha256": replay["manifest"]["pit_generation"][
                "generation_manifest_sha256"
            ],
            "daily_basic_coverage": {"status": "OK", "coverage_ratio": 1.0},
            "adj_factor_coverage": {"status": "OK", "coverage_ratio": 1.0},
        },
        "blockers": [],
    }
    commit = resolved_store.upsert_bars(
        _bars_frame(replay["frames"]),
        target_trade_date=target,
        target_trade_dates=ordered_targets,
        source="market_daily_sealed_capture",
        metadata=metadata,
        expected_latest_pointer_sha256=publication_parent_sha,
    )
    readback = resolved_store.validate_latest()
    if readback.get("status") != "passed":
        raise MarketDailyCaptureBlocked(["market_publication_readback_failed"])
    return {
        "schema_version": "cn-market-daily-publication.v1",
        "status": "PUBLISHED" if publication_mode == "execute" else "SHADOW_CANDIDATE",
        "target_trade_date": target,
        "target_trade_dates": ordered_targets,
        "same_target_rebind": same_target_rebind,
        "capture_manifest_path": str(capture_manifest_path),
        "capture_manifest_sha256": expected_capture_manifest_sha256,
        "provider_refetched": False,
        "capture_expected_market_pointer_sha256": replay["expected_market_pointer_sha256"],
        "publication_previous_market_pointer_sha256": publication_parent_sha,
        "parquet_commit": commit,
        "readback": readback,
        "classification": classification,
        "blockers": [],
    }


def shadow_market_daily_capture(
    *,
    shadow_data_root: str | Path,
    production_data_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Clone history and publish only to a complete private Market root."""

    if kwargs.get("store") is not None:
        raise MarketDailyCaptureBlocked(["shadow_store_injection_not_permitted"])
    expected_production_sha = str(kwargs.get("expected_market_pointer_sha256") or "")
    candidate = build_private_market_candidate(
        production_data_root=production_data_root,
        candidate_data_root=shadow_data_root,
        expected_production_market_pointer_sha256=expected_production_sha,
        private_pit_generation_binding=kwargs["pit_generation_binding"],
    )
    publication = publish_market_daily_capture(
        capture_manifest_path=kwargs["capture_manifest_path"],
        expected_capture_manifest_sha256=kwargs["expected_capture_manifest_sha256"],
        data_root=candidate["candidate_data_root"],
        scope_path=kwargs["scope_path"],
        expected_scope_sha256=kwargs["expected_scope_sha256"],
        pit_generation_binding=kwargs["pit_generation_binding"],
        expected_market_pointer_sha256=expected_production_sha,
        publication_expected_market_pointer_sha256=candidate["candidate_pointer_sha256"],
        publication_mode="shadow",
    )
    _protected_refs_unchanged(candidate["protected_production_refs"])
    _protected_refs_unchanged(candidate["protected_candidate_pit_refs"])
    candidate_pointer_path = Path(candidate["candidate_pointer_path"])
    candidate_pointer_raw = _stable_regular_bytes(
        candidate_pointer_path, label="published_shadow_candidate_pointer"
    )
    (Path(candidate["candidate_data_root"]) / "parquet" / "cn").chmod(0o500)
    return {
        "schema_version": "cn-market-daily-shadow-candidate.v1",
        "status": "SHADOW_CANDIDATE",
        "target_trade_date": publication["target_trade_date"],
        "target_trade_dates": publication["target_trade_dates"],
        "candidate_data_root": candidate["candidate_data_root"],
        "candidate_pointer_path": candidate["candidate_pointer_path"],
        "candidate_parent_pointer_sha256": candidate["candidate_pointer_sha256"],
        "candidate_pointer_sha256": _sha256(candidate_pointer_raw),
        "source_snapshot_id": candidate["source_snapshot_id"],
        "source_latest_complete_trade_date": candidate["source_latest_complete_trade_date"],
        "protected_production_refs": candidate["protected_production_refs"],
        "protected_candidate_pit_refs": candidate["protected_candidate_pit_refs"],
        "resource_preflight": candidate["resource_preflight"],
        "provider_refetched": False,
        "canonical_write_authorized": False,
        "canonical_write_performed": False,
        "publication": publication,
        "readback": publication["readback"],
        "blockers": [],
    }


validate_market_daily_capture = replay_market_daily_capture


__all__ = [
    "CAPTURE_SCHEMA",
    "CLASSIFICATION_SCHEMA",
    "ENDPOINT_SCHEMA",
    "MarketDailyCaptureBlocked",
    "build_private_market_candidate",
    "capture_market_daily",
    "classify_market_target",
    "publish_market_daily_capture",
    "replay_market_daily_capture",
    "shadow_market_daily_capture",
    "validate_market_daily_capture",
    "validate_market_endpoint_frames",
]
