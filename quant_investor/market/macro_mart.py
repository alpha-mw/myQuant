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
import math
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterator, Mapping, TypeVar
from zoneinfo import ZoneInfo

import numpy as np
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
PRIMARY_PROVENANCE_SCHEMA = "cn-macro-primary-provenance.v1"
PROVIDER_BUNDLE_SCHEMA = "cn-macro-provider-bundle.v1"
TRANSFORM_VERSION = "cn-macro-market-confirmation.v1"
MARKET_FORMULA_UNIVERSE_SCHEMA = "cn-macro-formula-universe.v1"
MARKET_FORMULA_SELECTION_RULE = (
    "symbol_terminal_trade_date_equals_target_trade_date"
)
STRICT_CATALOG_SCHEMA = "strict-parquet-catalog.v1"
LEGACY_CATALOG_SCHEMA = "myquant-cn-clean-catalog.v1"
CATALOG_WRITER_LOCK_FILENAME = (
    "._catalog.json.intelligence-retirement.lock"
)
TRANSACTION_JOURNAL_SCHEMA = "cn-macro-catalog-transaction.v1"
CAPTURE_WINDOW_HOURS = 72
MARKET_LOOKBACK_CALENDAR_DAYS = 450
VOLATILITY_WINDOW_SESSIONS = 20
VOLATILITY_PERCENTILE_LOOKBACK = 252
VOLATILITY_MIN_OBSERVATIONS = 60
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
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_PRIMARY_MACRO_CAPABILITY = object()
_ENDPOINT_SPECS: dict[str, dict[str, Any]] = {
    "cn_pmi": {
        "month_field": "month",
        "value_fields": ("PMI010000",),
        "max_release_lag_days": 15,
    },
    "cn_cpi": {
        "month_field": "month",
        "value_fields": ("nt_yoy",),
        "max_release_lag_days": 45,
    },
    "cn_ppi": {
        "month_field": "month",
        "value_fields": ("ppi_yoy",),
        "max_release_lag_days": 45,
    },
    "sf_month": {
        "month_field": "month",
        "value_fields": ("inc_month",),
        "max_release_lag_days": 45,
    },
    "cn_m": {
        "month_field": "month",
        "value_fields": ("m1_yoy", "m2_yoy"),
        "max_release_lag_days": 45,
    },
}


@dataclass(frozen=True)
class _PrimaryMacroAttestation:
    capability: object
    provider_bundle_sha256: str
    canonical_market_pointer_sha256: str
    market_input_files_sha256: str
    market_formula_universe_sha256: str
    output_frame_sha256: str
    transform_version: str


class MacroMartPromotionError(RuntimeError):
    """Raised when Macro candidate or canonical lineage is unsafe."""


def _now_utc() -> str:
    return _utc_now().replace(microsecond=0).isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MacroMartPromotionError(
            "macro_provenance_not_canonical_json"
        ) from exc


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.loc[:, sorted(str(column) for column in normalized.columns)]
    payload = normalized.to_json(
        orient="records",
        date_format="iso",
        double_precision=15,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    try:
        payload = json.loads(
            frame.to_json(
                orient="records",
                date_format="iso",
                double_precision=15,
            )
        )
    except (TypeError, ValueError) as exc:
        raise MacroMartPromotionError(
            "macro_provider_response_not_json_safe"
        ) from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, Mapping) for item in payload
    ):
        raise MacroMartPromotionError("macro_provider_response_invalid")
    return [dict(item) for item in payload]


def _read_verified_bytes_and_json(
    path: Path,
    *,
    trust_root: Path,
    expected_sha256: str | None,
    hash_blocker: str,
    changed_blocker: str,
    unreadable_blocker: str,
) -> tuple[bytes, dict[str, Any]]:
    return _read_verified_member(
        path,
        trust_root=trust_root,
        expected_sha256=expected_sha256,
        hash_blocker=hash_blocker,
        changed_blocker=changed_blocker,
        unreadable_blocker=unreadable_blocker,
        parser=lambda payload: (payload, _parse_json_object(payload)),
    )


def _read_stable_bytes(path: Path, *, blocker: str) -> bytes:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise MacroMartPromotionError(blocker)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except MacroMartPromotionError:
        raise
    except OSError as exc:
        raise MacroMartPromotionError(blocker) from exc
    try:
        signature = _stat_signature(before)
        if _stat_signature(os.fstat(descriptor)) != signature:
            raise MacroMartPromotionError(blocker)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if (
            _stat_signature(os.fstat(descriptor)) != signature
            or _stat_signature(os.lstat(path)) != signature
        ):
            raise MacroMartPromotionError(blocker)
        return b"".join(chunks)
    except MacroMartPromotionError:
        raise
    except OSError as exc:
        raise MacroMartPromotionError(blocker) from exc
    finally:
        os.close(descriptor)


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


@contextmanager
def _catalog_writer_lock(market_root: Path) -> Iterator[None]:
    """Share the sole production catalog-writer lock with retirement code."""

    lock_path = market_root / CATALOG_WRITER_LOCK_FILENAME
    if lock_path.exists() and lock_path.is_symlink():
        raise MacroMartPromotionError("macro_catalog_lock_symlink_rejected")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    locked = False
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise MacroMartPromotionError("macro_catalog_lock_invalid")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
    except MacroMartPromotionError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise MacroMartPromotionError(
            "macro_catalog_lock_unavailable"
        ) from exc
    try:
        yield
    finally:
        if descriptor is not None:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _safe_directory(path: Path, *, blocker: str) -> Path:
    if path.exists() and path.is_symlink():
        raise MacroMartPromotionError(blocker)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise MacroMartPromotionError(blocker)
    os.chmod(path, 0o700)
    return path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )


def _compact_trade_date(value: Any, *, blocker: str) -> str:
    text = str(value or "").strip().replace("-", "")
    try:
        parsed = datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise MacroMartPromotionError(blocker) from exc
    if parsed.strftime("%Y%m%d") != text:
        raise MacroMartPromotionError(blocker)
    return text


def _normalize_provider_month(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "")
    if re.fullmatch(r"\d{6}", text) is None:
        raise MacroMartPromotionError("macro_provider_month_invalid")
    try:
        datetime.strptime(text, "%Y%m")
    except ValueError as exc:
        raise MacroMartPromotionError("macro_provider_month_invalid") from exc
    return text


def _build_tushare_client() -> Any:
    try:
        import tushare as ts

        from quant_investor.config import config
        from quant_investor.credential_utils import create_tushare_pro
    except Exception as exc:
        raise MacroMartPromotionError(
            "macro_live_provider_dependency_unavailable"
        ) from exc
    client = create_tushare_pro(ts, config.TUSHARE_TOKEN, config.TUSHARE_URL)
    if client is None:
        raise MacroMartPromotionError("macro_live_provider_token_missing")
    return client


def _provider_query_window(captured_at: datetime) -> tuple[str, str]:
    end_month = captured_at.astimezone(_SHANGHAI).strftime("%Y%m")
    start = (
        pd.Timestamp(end_month + "01") - pd.DateOffset(months=36)
    ).strftime("%Y%m")
    return start, end_month


def _expected_latest_provider_month(
    cutoff_at: datetime,
    *,
    max_release_lag_days: int,
) -> str:
    """Return the latest month conservatively due by ``cutoff_at``.

    A month is due once its calendar month-end plus the endpoint's declared
    maximum release lag is on or before the Shanghai cutoff date.
    """

    if cutoff_at.tzinfo is None or cutoff_at.utcoffset() is None:
        raise MacroMartPromotionError("macro_provider_cutoff_clock_invalid")
    if max_release_lag_days < 0:
        raise MacroMartPromotionError("macro_provider_release_lag_invalid")
    local_date = cutoff_at.astimezone(_SHANGHAI).date()
    threshold = pd.Timestamp(local_date) - pd.Timedelta(
        days=max_release_lag_days
    )
    month_end = threshold + pd.offsets.MonthEnd(0)
    if threshold != month_end:
        month_end = threshold.replace(day=1) - pd.Timedelta(days=1)
    return month_end.strftime("%Y%m")


def _validate_provider_bundle_current_freshness(
    bundle: Mapping[str, Any],
    *,
    current_at: datetime,
) -> None:
    """Recheck release-lag freshness without rewriting stored PIT evidence."""

    if current_at.tzinfo is None or current_at.utcoffset() is None:
        raise MacroMartPromotionError(
            "macro_provider_current_freshness_clock_invalid"
        )
    current_utc = current_at.astimezone(timezone.utc)
    selected = bundle.get("selected_inputs")
    if not isinstance(selected, Mapping) or set(selected) != set(
        _ENDPOINT_SPECS
    ):
        raise MacroMartPromotionError(
            "macro_provider_bundle_selected_inputs_invalid"
        )
    for endpoint, spec in sorted(_ENDPOINT_SPECS.items()):
        chosen = selected.get(endpoint)
        if not isinstance(chosen, Mapping):
            raise MacroMartPromotionError(
                "macro_provider_bundle_selected_inputs_invalid"
            )
        selected_month = _normalize_provider_month(chosen.get("month"))
        expected_latest = _expected_latest_provider_month(
            current_utc,
            max_release_lag_days=int(spec["max_release_lag_days"]),
        )
        if selected_month < expected_latest:
            raise MacroMartPromotionError(
                "macro_provider_stale_new_run_id_required:"
                f"{endpoint}"
            )


def _fetch_provider_bundle(
    *,
    client: Any,
    trade_date: str,
    captured_at: datetime,
) -> dict[str, Any]:
    start_month, end_month = _provider_query_window(captured_at)
    endpoints: dict[str, Any] = {}
    selected_inputs: dict[str, Any] = {}
    for endpoint, spec in sorted(_ENDPOINT_SPECS.items()):
        method = getattr(client, endpoint, None)
        if not callable(method):
            raise MacroMartPromotionError(
                f"macro_provider_endpoint_unavailable:{endpoint}"
            )
        try:
            response = method(start_m=start_month, end_m=end_month)
        except Exception as exc:
            raise MacroMartPromotionError(
                f"macro_provider_request_failed:{endpoint}"
            ) from exc
        if not isinstance(response, pd.DataFrame) or response.empty:
            raise MacroMartPromotionError(
                f"macro_provider_response_empty:{endpoint}"
            )
        if len(response) < 12:
            raise MacroMartPromotionError(
                f"macro_provider_history_insufficient:{endpoint}"
            )
        column_lookup = {
            str(column).strip().lower(): str(column)
            for column in response.columns
        }
        month_field = column_lookup.get(
            str(spec["month_field"]).lower()
        )
        value_fields = {
            field: column_lookup.get(str(field).lower())
            for field in spec["value_fields"]
        }
        if month_field is None or any(
            value is None for value in value_fields.values()
        ):
            raise MacroMartPromotionError(
                f"macro_provider_schema_invalid:{endpoint}"
            )
        normalized = response.copy()
        normalized["__month"] = normalized[month_field].map(
            _normalize_provider_month
        )
        if normalized["__month"].duplicated().any():
            raise MacroMartPromotionError(
                f"macro_provider_month_duplicate:{endpoint}"
            )
        if normalized["__month"].gt(end_month).any():
            raise MacroMartPromotionError(
                f"macro_provider_future_month_rejected:{endpoint}"
            )
        for logical_field, actual_field in value_fields.items():
            assert actual_field is not None
            values = pd.to_numeric(normalized[actual_field], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise MacroMartPromotionError(
                    f"macro_provider_value_invalid:{endpoint}:{logical_field}"
                )
            normalized[actual_field] = values.astype(float)
        normalized = normalized.sort_values("__month", kind="mergesort")
        raw_frame = normalized.drop(columns=["__month"])
        raw_frame = raw_frame.loc[:, sorted(raw_frame.columns.astype(str))]
        records = _json_records(raw_frame)
        records_hash = hashlib.sha256(
            _canonical_json_bytes({"records": records})
        ).hexdigest()
        latest = normalized.iloc[-1]
        latest_month = str(latest["__month"])
        values = {
            logical_field: float(latest[str(actual_field)])
            for logical_field, actual_field in value_fields.items()
        }
        month_end = (
            pd.Timestamp(latest_month + "01")
            + pd.offsets.MonthEnd(0)
        )
        conservative_available_by = (
            month_end + pd.Timedelta(
                days=int(spec["max_release_lag_days"])
            )
        ).date().isoformat()
        endpoints[endpoint] = {
            "endpoint": endpoint,
            "query": {
                "start_m": start_month,
                "end_m": end_month,
            },
            "columns": sorted(raw_frame.columns.astype(str)),
            "row_count": int(len(records)),
            "records": records,
            "records_sha256": records_hash,
        }
        selected_inputs[endpoint] = {
            "month": latest_month,
            "values": values,
            "official_release_timestamp_known": False,
            "max_release_lag_days": int(spec["max_release_lag_days"]),
            "conservative_available_by": conservative_available_by,
            "transform_role": (
                "policy_signal" if endpoint == "cn_m" else "context_only"
            ),
        }
    completed_at = _utc_now()
    if (
        completed_at.tzinfo is None
        or completed_at.utcoffset() is None
        or captured_at.tzinfo is None
        or captured_at.utcoffset() is None
    ):
        raise MacroMartPromotionError("macro_provider_completion_clock_invalid")
    if completed_at < captured_at:
        raise MacroMartPromotionError("macro_provider_completion_before_start")
    fetched_at = completed_at.replace(microsecond=0).isoformat()
    for endpoint, spec in sorted(_ENDPOINT_SPECS.items()):
        expected_latest = _expected_latest_provider_month(
            completed_at,
            max_release_lag_days=int(spec["max_release_lag_days"]),
        )
        chosen = selected_inputs[endpoint]
        if str(chosen["month"]) < expected_latest:
            raise MacroMartPromotionError(
                f"macro_provider_latest_month_stale:{endpoint}"
            )
        chosen["observed_available_at"] = fetched_at
        chosen["expected_latest_month_lower_bound"] = expected_latest
    return {
        "schema_version": PROVIDER_BUNDLE_SCHEMA,
        "provider_id": "tushare_pro",
        "source": SOURCE_TUSHARE,
        "source_priority": SOURCE_TUSHARE,
        "trade_date": _date_text(trade_date),
        "fetched_at": fetched_at,
        "decision_cutoff_at": fetched_at,
        "live_requested": True,
        "historical_replay_eligible": False,
        "official_release_timestamps_claimed": False,
        "query_window": {
            "start_month": start_month,
            "end_month": end_month,
        },
        "endpoints": endpoints,
        "selected_inputs": selected_inputs,
    }


def _validate_live_target(
    pointer: Mapping[str, Any],
    *,
    requested_as_of: str,
    captured_at: datetime,
    enforce_capture_window: bool = True,
) -> str:
    if (
        str(pointer.get("status") or "").upper() != "OK"
        or pointer.get("blockers") not in ([], ())
    ):
        raise MacroMartPromotionError("macro_market_pointer_not_ready")
    latest_complete = _compact_trade_date(
        pointer.get("latest_complete_trade_date"),
        blocker="macro_market_pointer_latest_complete_invalid",
    )
    latest_available = _compact_trade_date(
        pointer.get("latest_available_trade_date")
        or pointer.get("latest_trade_date"),
        blocker="macro_market_pointer_latest_available_invalid",
    )
    if latest_complete != latest_available:
        raise MacroMartPromotionError("macro_market_pointer_tail_incomplete")
    if requested_as_of:
        requested = _compact_trade_date(
            requested_as_of,
            blocker="macro_requested_as_of_invalid",
        )
        if requested != latest_complete:
            raise MacroMartPromotionError(
                "macro_requested_as_of_not_latest_complete"
            )
    if not enforce_capture_window:
        return latest_complete
    local_capture = captured_at.astimezone(_SHANGHAI)
    session_date = datetime.strptime(latest_complete, "%Y%m%d").date()
    session_close = datetime.combine(
        session_date,
        time(hour=15),
        tzinfo=_SHANGHAI,
    )
    elapsed = local_capture - session_close
    if elapsed.total_seconds() < 0:
        raise MacroMartPromotionError("macro_capture_before_session_close")
    if elapsed > timedelta(hours=CAPTURE_WINDOW_HOURS):
        raise MacroMartPromotionError("macro_capture_window_expired")
    return latest_complete


def _resolve_bar_root(
    market_root: Path,
    pointer: Mapping[str, Any],
) -> Path:
    raw_text = str(pointer.get("table_root") or "").strip()
    if not raw_text:
        raise MacroMartPromotionError("macro_market_bar_root_missing")
    raw = Path(raw_text).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    root = _strict_read_root(raw)
    market_resolved = market_root.resolve(strict=True)
    if root != market_resolved and market_resolved not in root.parents:
        raise MacroMartPromotionError("macro_market_bar_root_escape")
    return root


def _selected_bar_paths(bar_root: Path, *, trade_date: str) -> list[Path]:
    target = datetime.strptime(trade_date, "%Y%m%d").date()
    start = target - timedelta(days=MARKET_LOOKBACK_CALENDAR_DAYS)
    selected: list[Path] = []
    seen_months: set[tuple[int, int]] = set()
    for path in sorted(bar_root.rglob("*.parquet")):
        try:
            relative = path.relative_to(bar_root)
        except ValueError as exc:
            raise MacroMartPromotionError(
                "macro_market_input_path_escape"
            ) from exc
        if len(relative.parts) != 3 or relative.name != "part.parquet":
            continue
        year_text, month_text, _ = relative.parts
        if (
            re.fullmatch(r"year=\d{4}", year_text) is None
            or re.fullmatch(r"month=\d{2}", month_text) is None
        ):
            continue
        year = int(year_text.split("=", 1)[1])
        month = int(month_text.split("=", 1)[1])
        try:
            month_start = date(year, month, 1)
        except ValueError as exc:
            raise MacroMartPromotionError(
                "macro_market_partition_invalid"
            ) from exc
        next_month = (
            date(year + 1, 1, 1)
            if month == 12
            else date(year, month + 1, 1)
        )
        month_end = next_month - timedelta(days=1)
        if month_end < start or month_start > target:
            continue
        key = (year, month)
        if key in seen_months:
            raise MacroMartPromotionError(
                "macro_market_partition_duplicate"
            )
        seen_months.add(key)
        selected.append(path)
    target_key = (target.year, target.month)
    if target_key not in seen_months or len(selected) < 12:
        raise MacroMartPromotionError("macro_market_history_insufficient")
    return selected


def _load_market_inputs(
    bar_root: Path,
    *,
    trade_date: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for path in _selected_bar_paths(bar_root, trade_date=trade_date):
        payload = _read_verified_member(
            path,
            trust_root=bar_root,
            expected_sha256=None,
            hash_blocker="macro_market_input_hash_mismatch",
            changed_blocker="macro_market_input_changed_during_read",
            unreadable_blocker="macro_market_input_unreadable",
            parser=lambda value: value,
        )
        try:
            frame = pd.read_parquet(
                io.BytesIO(payload),
                columns=["ts_code", "trade_date", "close"],
            )
        except Exception as exc:
            raise MacroMartPromotionError(
                "macro_market_input_parquet_invalid"
            ) from exc
        if set(frame.columns) != {"ts_code", "trade_date", "close"}:
            raise MacroMartPromotionError("macro_market_input_schema_invalid")
        frames.append(frame)
        evidence.append(
            {
                "path": path.relative_to(bar_root).as_posix(),
                "size_bytes": int(len(payload)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    symbol = combined["ts_code"].astype("string").fillna("").str.strip()
    raw_dates = (
        combined["trade_date"].astype("string").fillna("").str.strip()
    )
    valid_dates = raw_dates.str.fullmatch(r"\d{8}", na=False)
    parsed_dates = pd.to_datetime(
        raw_dates.where(valid_dates),
        format="%Y%m%d",
        errors="coerce",
    )
    close = pd.to_numeric(combined["close"], errors="coerce")
    if (
        symbol.eq("").any()
        or (~valid_dates).any()
        or parsed_dates.isna().any()
        or close.isna().any()
        or not np.isfinite(close.to_numpy()).all()
        or close.le(0.0).any()
    ):
        raise MacroMartPromotionError("macro_market_input_values_invalid")
    normalized = pd.DataFrame(
        {"ts_code": symbol, "trade_date": parsed_dates, "close": close}
    )
    target = pd.Timestamp(datetime.strptime(trade_date, "%Y%m%d"))
    start = target - pd.Timedelta(days=MARKET_LOOKBACK_CALENDAR_DAYS)
    normalized = normalized[
        normalized["trade_date"].between(start, target)
    ].copy()
    if normalized.empty or normalized["trade_date"].max() != target:
        raise MacroMartPromotionError("macro_market_target_session_missing")
    if normalized.duplicated(["ts_code", "trade_date"]).any():
        raise MacroMartPromotionError("macro_market_input_duplicate_key")
    normalized = normalized.sort_values(
        ["ts_code", "trade_date"],
        kind="mergesort",
    ).reset_index(drop=True)
    files_sha = _canonical_json_sha256({"files": evidence})
    return normalized, evidence, files_sha


def _verify_market_input_evidence(
    bar_root: Path,
    evidence: list[dict[str, Any]],
) -> None:
    for entry in evidence:
        relative = str(entry.get("path") or "")
        path = _resolve_catalog_member(
            bar_root,
            relative,
            blocker="macro_market_input_path_invalid",
        )
        expected_sha = _assert_sha256(
            entry.get("sha256"),
            blocker="macro_market_input_hash_invalid",
        )
        payload = _read_verified_member(
            path,
            trust_root=bar_root,
            expected_sha256=expected_sha,
            hash_blocker="macro_market_input_hash_mismatch",
            changed_blocker="macro_market_input_changed_during_read",
            unreadable_blocker="macro_market_input_unreadable",
            parser=lambda value: value,
        )
        if len(payload) != int(entry.get("size_bytes", -1)):
            raise MacroMartPromotionError("macro_market_input_size_mismatch")


def _formula_symbol_set_sha256(symbols: list[str]) -> str:
    return _canonical_json_sha256({"symbols": sorted(symbols)})


def _validate_market_formula_universe(
    raw: Mapping[str, Any],
    *,
    trade_date: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "selection_rule",
        "target_trade_date",
        "input_symbol_count",
        "target_terminal_symbol_count",
        "stale_symbol_count",
        "scored_symbol_count",
        "input_row_count",
        "target_terminal_row_count",
        "stale_row_count",
        "input_symbol_set_sha256",
        "target_terminal_symbol_set_sha256",
        "stale_symbol_set_sha256",
        "scored_symbol_set_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise MacroMartPromotionError(
            "macro_market_formula_universe_contract_invalid"
        )
    target = _compact_trade_date(
        raw.get("target_trade_date"),
        blocker="macro_market_formula_universe_target_invalid",
    )
    if (
        raw.get("schema_version") != MARKET_FORMULA_UNIVERSE_SCHEMA
        or raw.get("selection_rule") != MARKET_FORMULA_SELECTION_RULE
        or target
        != _compact_trade_date(
            trade_date,
            blocker="macro_market_formula_universe_target_invalid",
        )
    ):
        raise MacroMartPromotionError(
            "macro_market_formula_universe_contract_invalid"
        )

    counts: dict[str, int] = {}
    for field in (
        "input_symbol_count",
        "target_terminal_symbol_count",
        "stale_symbol_count",
        "scored_symbol_count",
        "input_row_count",
        "target_terminal_row_count",
        "stale_row_count",
    ):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MacroMartPromotionError(
                "macro_market_formula_universe_count_invalid"
            )
        counts[field] = value
    if (
        counts["input_symbol_count"]
        != counts["target_terminal_symbol_count"]
        + counts["stale_symbol_count"]
        or counts["input_row_count"]
        != counts["target_terminal_row_count"] + counts["stale_row_count"]
        or counts["scored_symbol_count"] < 100
        or counts["scored_symbol_count"]
        > counts["target_terminal_symbol_count"]
        or counts["target_terminal_symbol_count"]
        > counts["target_terminal_row_count"]
    ):
        raise MacroMartPromotionError(
            "macro_market_formula_universe_count_mismatch"
        )
    for field in (
        "input_symbol_set_sha256",
        "target_terminal_symbol_set_sha256",
        "stale_symbol_set_sha256",
        "scored_symbol_set_sha256",
    ):
        _assert_sha256(
            raw.get(field),
            blocker="macro_market_formula_universe_hash_invalid",
        )
    return dict(raw)


def _select_target_terminal_formula_universe(
    market: pd.DataFrame,
    *,
    trade_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"ts_code", "trade_date", "close"}
    if not isinstance(market, pd.DataFrame) or market.empty or not required.issubset(
        market.columns
    ):
        raise MacroMartPromotionError("macro_market_input_schema_invalid")
    target_text = _compact_trade_date(
        trade_date,
        blocker="macro_market_formula_universe_target_invalid",
    )
    target = pd.Timestamp(datetime.strptime(target_text, "%Y%m%d"))
    working = market.loc[:, ["ts_code", "trade_date", "close"]].copy()
    working["ts_code"] = (
        working["ts_code"].astype("string").fillna("").str.strip()
    )
    working["trade_date"] = pd.to_datetime(
        working["trade_date"],
        errors="coerce",
    )
    if working["ts_code"].eq("").any() or working["trade_date"].isna().any():
        raise MacroMartPromotionError("macro_market_input_values_invalid")
    if working["trade_date"].gt(target).any():
        raise MacroMartPromotionError("macro_market_future_session_rejected")
    if working.duplicated(["ts_code", "trade_date"]).any():
        raise MacroMartPromotionError("macro_market_input_duplicate_key")

    terminal = working.groupby("ts_code", sort=True)["trade_date"].max()
    input_symbols = [str(symbol) for symbol in terminal.index]
    selected_symbols = [
        str(symbol) for symbol in terminal.index[terminal.eq(target)]
    ]
    stale_symbols = [
        str(symbol) for symbol in terminal.index[terminal.lt(target)]
    ]
    if not selected_symbols:
        raise MacroMartPromotionError("macro_market_target_session_missing")
    selected = working.loc[working["ts_code"].isin(selected_symbols)].copy()
    selected = selected.sort_values(
        ["ts_code", "trade_date"],
        kind="mergesort",
    ).reset_index(drop=True)
    evidence = {
        "schema_version": MARKET_FORMULA_UNIVERSE_SCHEMA,
        "selection_rule": MARKET_FORMULA_SELECTION_RULE,
        "target_trade_date": target_text,
        "input_symbol_count": len(input_symbols),
        "target_terminal_symbol_count": len(selected_symbols),
        "stale_symbol_count": len(stale_symbols),
        "scored_symbol_count": 0,
        "input_row_count": int(len(working)),
        "target_terminal_row_count": int(len(selected)),
        "stale_row_count": int(len(working) - len(selected)),
        "input_symbol_set_sha256": _formula_symbol_set_sha256(input_symbols),
        "target_terminal_symbol_set_sha256": _formula_symbol_set_sha256(
            selected_symbols
        ),
        "stale_symbol_set_sha256": _formula_symbol_set_sha256(stale_symbols),
        "scored_symbol_set_sha256": _formula_symbol_set_sha256([]),
    }
    return selected, evidence


def _derive_macro_frame(
    market: pd.DataFrame,
    *,
    trade_date: str,
    provider_bundle: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered, universe_evidence = _select_target_terminal_formula_universe(
        market,
        trade_date=trade_date,
    )
    ordered["return"] = ordered.groupby("ts_code", sort=False)[
        "close"
    ].pct_change(fill_method=None)
    returns = ordered.dropna(subset=["return"]).copy()
    if (
        returns.empty
        or not np.isfinite(returns["return"].to_numpy()).all()
    ):
        raise MacroMartPromotionError("macro_market_returns_invalid")
    symbol_recent = returns.groupby("ts_code", sort=True)["return"].agg(
        lambda series: (
            float(series.tail(VOLATILITY_WINDOW_SESSIONS).mean())
            if len(series) >= VOLATILITY_WINDOW_SESSIONS
            else math.nan
        )
    )
    symbol_recent = symbol_recent.dropna()
    if len(symbol_recent) < 100:
        raise MacroMartPromotionError("macro_market_cross_section_insufficient")
    scored_symbols = [str(symbol) for symbol in symbol_recent.index]
    universe_evidence["scored_symbol_count"] = len(scored_symbols)
    universe_evidence["scored_symbol_set_sha256"] = _formula_symbol_set_sha256(
        scored_symbols
    )
    universe_evidence = _validate_market_formula_universe(
        universe_evidence,
        trade_date=trade_date,
    )
    macro_score = max(
        -1.0,
        min(1.0, float(fmean(symbol_recent.tolist())) * 20.0),
    )
    breadth = float(symbol_recent.gt(0.0).mean())
    market_daily = returns.groupby("trade_date", sort=True)["return"].mean()
    rolling_vol = (
        market_daily.rolling(
            VOLATILITY_WINDOW_SESSIONS,
            min_periods=VOLATILITY_WINDOW_SESSIONS,
        ).std(ddof=1)
        * math.sqrt(252.0)
    ).dropna()
    target = pd.Timestamp(datetime.strptime(trade_date, "%Y%m%d"))
    if target not in rolling_vol.index:
        raise MacroMartPromotionError("macro_market_target_volatility_missing")
    trailing = rolling_vol.loc[:target].tail(
        VOLATILITY_PERCENTILE_LOOKBACK
    )
    if len(trailing) < VOLATILITY_MIN_OBSERVATIONS:
        raise MacroMartPromotionError(
            "macro_market_volatility_history_insufficient"
        )
    current_vol = float(trailing.iloc[-1])
    if not math.isfinite(current_vol):
        raise MacroMartPromotionError("macro_market_volatility_invalid")
    volatility_percentile = float(trailing.le(current_vol).mean() * 100.0)
    selected = provider_bundle.get("selected_inputs")
    if not isinstance(selected, Mapping):
        raise MacroMartPromotionError("macro_provider_selected_inputs_missing")
    cn_m = selected.get("cn_m")
    values = cn_m.get("values") if isinstance(cn_m, Mapping) else None
    try:
        m2_yoy = float(values["m2_yoy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MacroMartPromotionError("macro_provider_m2_invalid") from exc
    if not math.isfinite(m2_yoy):
        raise MacroMartPromotionError("macro_provider_m2_invalid")
    policy_signal = (
        "supportive"
        if m2_yoy > 10.0
        else "restrictive" if m2_yoy <= 8.0 else "neutral"
    )
    row = {
        "trade_date": _date_text(trade_date),
        "macro_score": macro_score,
        "liquidity_score": breadth,
        "volatility_percentile": volatility_percentile,
        "policy_signal": policy_signal,
        "source": SOURCE_TUSHARE,
        "source_priority": SOURCE_TUSHARE,
        "pit_status": "market_point_in_time",
        "fetched_at": str(provider_bundle.get("fetched_at") or ""),
    }
    frame = pd.DataFrame([row], columns=sorted(_ALLOWED_INPUT_FIELDS))
    return frame, universe_evidence


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_primary_generation(
    *,
    root: Path,
    run_id: str,
    frame: pd.DataFrame,
    provider_bundle: Mapping[str, Any],
    market_pointer_sha256: str,
    market_input_evidence: list[dict[str, Any]],
    market_input_files_sha256: str,
    market_formula_universe: Mapping[str, Any],
) -> tuple[dict[str, Any], _PrimaryMacroAttestation]:
    formula_universe = _validate_market_formula_universe(
        market_formula_universe,
        trade_date=str(frame.iloc[0]["trade_date"]),
    )
    formula_universe_sha = _canonical_json_sha256(formula_universe)
    generations_root = _safe_directory(
        root / "_generations",
        blocker="macro_generations_root_invalid",
    )
    destination = generations_root / run_id
    if destination.exists() or destination.is_symlink():
        raise MacroMartPromotionError("macro_generation_exists")
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.", dir=generations_root)
    )
    os.chmod(temp_path, 0o700)
    try:
        provider_path = temp_path / "provider_bundle.json"
        provider_bytes = _canonical_json_bytes(provider_bundle) + b"\n"
        _atomic_write_bytes(provider_path, provider_bytes)
        provider_sha = hashlib.sha256(provider_bytes).hexdigest()

        table_path = temp_path / "part.parquet"
        frame.to_parquet(table_path, index=False)
        os.chmod(table_path, 0o600)
        _fsync_file(table_path)
        table_sha = _sha256(table_path)
        output_frame_sha = _frame_sha256(frame)
        provenance: dict[str, Any] = {
            "schema_version": PRIMARY_PROVENANCE_SCHEMA,
            "status": "verified_live_tushare",
            "source": SOURCE_TUSHARE,
            "source_priority": SOURCE_TUSHARE,
            "trade_date": str(frame.iloc[0]["trade_date"]),
            "fetched_at": str(frame.iloc[0]["fetched_at"]),
            "provider_bundle_sha256": provider_sha,
            "canonical_market_pointer_sha256": market_pointer_sha256,
            "market_input_files_sha256": market_input_files_sha256,
            "market_formula_universe_sha256": formula_universe_sha,
            "output_frame_sha256": output_frame_sha,
            "output_parquet_sha256": table_sha,
            "transform_version": TRANSFORM_VERSION,
            "historical_replay_eligible": False,
        }
        provenance["envelope_sha256"] = _canonical_json_sha256(provenance)
        manifest: dict[str, Any] = {
            "schema_version": CANONICAL_MANIFEST_SCHEMA,
            "generation_id": run_id,
            "run_id": run_id,
            "table": "macro_daily",
            "table_path": "part.parquet",
            "parquet_sha256": table_sha,
            "provider_bundle_path": "provider_bundle.json",
            "provider_bundle_sha256": provider_sha,
            "row_count": int(len(frame)),
            "as_of": str(frame.iloc[0]["trade_date"]),
            "source": SOURCE_TUSHARE,
            "source_priority": SOURCE_TUSHARE,
            "provider_status": "verified_provider_snapshot",
            "pit_status": "market_point_in_time",
            "decision_cutoff_at": str(frame.iloc[0]["fetched_at"]),
            "historical_replay_eligible": False,
            "transform_version": TRANSFORM_VERSION,
            "market_input_files": market_input_evidence,
            "market_input_files_sha256": market_input_files_sha256,
            "market_formula_universe": formula_universe,
            "market_formula_universe_sha256": formula_universe_sha,
            "primary_provenance": provenance,
            "production_eligible": True,
            "generated_at": _now_utc(),
        }
        _validate_provider_bundle(provider_bundle, manifest=manifest)
        _validate_canonical_frame(frame, manifest)
        _validate_primary_provenance(
            manifest,
            provider_bundle_sha256=provider_sha,
            table_sha256=table_sha,
            output_frame_sha256=output_frame_sha,
        )
        readback = pd.read_parquet(table_path)
        if _frame_sha256(readback) != output_frame_sha:
            raise MacroMartPromotionError(
                "macro_generation_parquet_readback_mismatch"
            )
        manifest_path = temp_path / "manifest.json"
        _atomic_json(manifest_path, manifest)
        manifest_sha = _sha256(manifest_path)
        _fsync_directory(temp_path)
        os.replace(temp_path, destination)
        _fsync_directory(generations_root)
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise
    manifest["generation_manifest"] = str(destination / "manifest.json")
    manifest["generation_manifest_sha256"] = manifest_sha
    manifest["resolved_table_path"] = str(destination / "part.parquet")
    manifest["resolved_provider_bundle"] = str(
        destination / "provider_bundle.json"
    )
    attestation = _PrimaryMacroAttestation(
        capability=_PRIMARY_MACRO_CAPABILITY,
        provider_bundle_sha256=provider_sha,
        canonical_market_pointer_sha256=market_pointer_sha256,
        market_input_files_sha256=market_input_files_sha256,
        market_formula_universe_sha256=formula_universe_sha,
        output_frame_sha256=output_frame_sha,
        transform_version=TRANSFORM_VERSION,
    )
    return manifest, attestation


def _load_primary_generation_for_retry(
    *,
    root: Path,
    run_id: str,
    trade_date: str,
    market_pointer_sha256: str,
) -> tuple[
    dict[str, Any],
    _PrimaryMacroAttestation,
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, Any],
] | None:
    generation_relative = Path("_generations") / run_id
    generation = root / generation_relative
    if not generation.exists():
        return None
    manifest_path = _resolve_catalog_member(
        root,
        (generation_relative / "manifest.json").as_posix(),
        blocker="macro_retry_generation_manifest_invalid",
    )
    manifest_bytes, manifest = _read_verified_bytes_and_json(
        manifest_path,
        trust_root=root,
        expected_sha256=None,
        hash_blocker="macro_retry_generation_manifest_hash_mismatch",
        changed_blocker="macro_retry_generation_manifest_changed",
        unreadable_blocker="macro_retry_generation_manifest_invalid",
    )
    if (
        manifest.get("schema_version") != CANONICAL_MANIFEST_SCHEMA
        or str(manifest.get("generation_id") or "") != run_id
        or str(manifest.get("run_id") or "") != run_id
        or manifest.get("production_eligible") is not True
        or str(manifest.get("transform_version") or "") != TRANSFORM_VERSION
        or _compact_trade_date(
            manifest.get("as_of"),
            blocker="macro_retry_generation_as_of_invalid",
        )
        != trade_date
        or str(manifest.get("table_path") or "") != "part.parquet"
        or str(manifest.get("provider_bundle_path") or "")
        != "provider_bundle.json"
    ):
        raise MacroMartPromotionError("macro_retry_generation_contract_invalid")
    table_sha = _assert_sha256(
        manifest.get("parquet_sha256"),
        blocker="macro_retry_generation_table_hash_invalid",
    )
    provider_sha = _assert_sha256(
        manifest.get("provider_bundle_sha256"),
        blocker="macro_retry_generation_provider_hash_invalid",
    )
    market_files_sha = _assert_sha256(
        manifest.get("market_input_files_sha256"),
        blocker="macro_retry_generation_market_hash_invalid",
    )
    formula_universe = manifest.get("market_formula_universe")
    if not isinstance(formula_universe, Mapping):
        raise MacroMartPromotionError(
            "macro_retry_generation_formula_universe_invalid"
        )
    validated_formula_universe = _validate_market_formula_universe(
        formula_universe,
        trade_date=trade_date,
    )
    formula_universe_sha = _assert_sha256(
        manifest.get("market_formula_universe_sha256"),
        blocker="macro_retry_generation_formula_universe_hash_invalid",
    )
    if formula_universe_sha != _canonical_json_sha256(
        validated_formula_universe
    ):
        raise MacroMartPromotionError(
            "macro_retry_generation_formula_universe_hash_mismatch"
        )
    table_path = _resolve_catalog_member(
        root,
        (generation_relative / "part.parquet").as_posix(),
        blocker="macro_retry_generation_table_invalid",
    )
    provider_path = _resolve_catalog_member(
        root,
        (generation_relative / "provider_bundle.json").as_posix(),
        blocker="macro_retry_generation_provider_invalid",
    )
    frame = _read_verified_member(
        table_path,
        trust_root=root,
        expected_sha256=table_sha,
        hash_blocker="macro_retry_generation_table_hash_mismatch",
        changed_blocker="macro_retry_generation_table_changed",
        unreadable_blocker="macro_retry_generation_table_unreadable",
        parser=lambda payload: pd.read_parquet(io.BytesIO(payload)),
    )
    provider_bundle = _read_verified_member(
        provider_path,
        trust_root=root,
        expected_sha256=provider_sha,
        hash_blocker="macro_retry_generation_provider_hash_mismatch",
        changed_blocker="macro_retry_generation_provider_changed",
        unreadable_blocker="macro_retry_generation_provider_invalid",
        parser=_parse_json_object,
    )
    _validate_provider_bundle(provider_bundle, manifest=manifest)
    _validate_provider_bundle_current_freshness(
        provider_bundle,
        current_at=_utc_now(),
    )
    _validate_canonical_frame(frame, manifest)
    output_frame_sha = _frame_sha256(frame)
    _validate_primary_provenance(
        manifest,
        provider_bundle_sha256=provider_sha,
        table_sha256=table_sha,
        output_frame_sha256=output_frame_sha,
    )
    provenance = manifest.get("primary_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("canonical_market_pointer_sha256")
        != market_pointer_sha256
    ):
        raise MacroMartPromotionError(
            "macro_retry_generation_market_pointer_mismatch"
        )
    market_input_evidence = manifest.get("market_input_files")
    if not isinstance(market_input_evidence, list):
        raise MacroMartPromotionError(
            "macro_retry_generation_market_evidence_invalid"
        )
    manifest["generation_manifest"] = str(manifest_path)
    manifest["generation_manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    manifest["resolved_table_path"] = str(table_path)
    manifest["resolved_provider_bundle"] = str(provider_path)
    attestation = _PrimaryMacroAttestation(
        capability=_PRIMARY_MACRO_CAPABILITY,
        provider_bundle_sha256=provider_sha,
        canonical_market_pointer_sha256=market_pointer_sha256,
        market_input_files_sha256=market_files_sha,
        market_formula_universe_sha256=formula_universe_sha,
        output_frame_sha256=output_frame_sha,
        transform_version=TRANSFORM_VERSION,
    )
    return (
        manifest,
        attestation,
        frame,
        [dict(item) for item in market_input_evidence],
        provider_bundle,
    )


def _existing_catalog_member_relative(
    market_root: Path,
    raw_value: Any,
    *,
    blocker: str,
) -> str:
    text = str(raw_value or "").strip()
    if not text:
        raise MacroMartPromotionError(blocker)
    raw = Path(text).expanduser()
    if ".." in raw.parts:
        raise MacroMartPromotionError(blocker)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, market_root / raw]
    candidate = next((item for item in candidates if item.exists()), None)
    if candidate is None:
        raise MacroMartPromotionError(blocker)
    try:
        absolute_candidate = Path(os.path.abspath(candidate))
        cursor = Path(absolute_candidate.anchor)
        parts = absolute_candidate.parts[1:]
        for part in parts:
            cursor = cursor / part
            metadata = os.lstat(cursor)
            if stat.S_ISLNK(metadata.st_mode):
                raise MacroMartPromotionError(blocker)
        resolved = absolute_candidate.resolve(strict=True)
        relative = resolved.relative_to(market_root.resolve(strict=True))
    except MacroMartPromotionError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise MacroMartPromotionError(blocker) from exc
    cursor = market_root.resolve(strict=True)
    for part in relative.parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise MacroMartPromotionError(blocker) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise MacroMartPromotionError(blocker)
    if not relative.parts:
        raise MacroMartPromotionError(blocker)
    return relative.as_posix()


def _contains_retired_intelligence(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "intelligence" in str(key).lower()
            or _contains_retired_intelligence(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_retired_intelligence(item) for item in value)
    return isinstance(value, str) and "intelligence" in value.lower()


def _validate_catalog_table_root(
    market_root: Path,
    raw_value: Any,
    *,
    blocker: str,
) -> None:
    relative = Path(str(raw_value or "").strip())
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise MacroMartPromotionError(blocker)
    cursor = market_root
    for part in relative.parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise MacroMartPromotionError(blocker) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MacroMartPromotionError(blocker)
    try:
        resolved = cursor.resolve(strict=True)
        market_resolved = market_root.resolve(strict=True)
    except OSError as exc:
        raise MacroMartPromotionError(blocker) from exc
    if market_resolved not in resolved.parents:
        raise MacroMartPromotionError(blocker)


def _validate_required_catalog_table(
    *,
    schema: str,
    market_root: Path,
    logical_table: str,
    entry: dict[str, Any],
) -> None:
    if entry.get("table_root"):
        _validate_catalog_table_root(
            market_root,
            entry["table_root"],
            blocker=f"macro_catalog_required_root_invalid:{logical_table}",
        )
    raw_path = entry.get("path")
    if not raw_path:
        raise MacroMartPromotionError(
            f"macro_catalog_required_path_missing:{logical_table}"
        )
    table_path = _resolve_catalog_member(
        market_root,
        raw_path,
        blocker=f"macro_catalog_required_path_invalid:{logical_table}",
    )
    declared_hashes = [
        value
        for value in (entry.get("sha256"), entry.get("parquet_sha256"))
        if value not in (None, "")
    ]
    if not declared_hashes:
        if schema == STRICT_CATALOG_SCHEMA:
            raise MacroMartPromotionError(
                f"macro_catalog_required_hash_missing:{logical_table}"
            )
        expected_sha = _sha256(table_path)
        entry["sha256"] = expected_sha
    else:
        normalized_hashes = {
            _assert_sha256(
                value,
                blocker=f"macro_catalog_required_hash_invalid:{logical_table}",
            )
            for value in declared_hashes
        }
        if len(normalized_hashes) != 1:
            raise MacroMartPromotionError(
                f"macro_catalog_required_hash_conflict:{logical_table}"
            )
        expected_sha = normalized_hashes.pop()
    _read_verified_member(
        table_path,
        trust_root=market_root,
        expected_sha256=expected_sha,
        hash_blocker=f"macro_catalog_required_hash_mismatch:{logical_table}",
        changed_blocker=f"macro_catalog_required_table_changed:{logical_table}",
        unreadable_blocker=f"macro_catalog_required_table_unreadable:{logical_table}",
        parser=lambda payload: pd.read_parquet(io.BytesIO(payload), columns=[]),
    )


def _validate_strict_catalog_required_table_closure(
    catalog: Mapping[str, Any],
    *,
    market_root: Path,
) -> None:
    if catalog.get("schema_version") != STRICT_CATALOG_SCHEMA:
        raise MacroMartPromotionError("macro_catalog_schema_invalid")
    if _contains_retired_intelligence(catalog):
        raise MacroMartPromotionError(
            "macro_catalog_retired_intelligence_present"
        )
    tables = catalog.get("tables")
    required_tables = catalog.get("required_tables")
    if not isinstance(tables, Mapping) or not tables:
        raise MacroMartPromotionError("macro_catalog_tables_invalid")
    if (
        not isinstance(required_tables, list)
        or not required_tables
        or not all(
            isinstance(name, str) and bool(name.strip())
            for name in required_tables
        )
        or len(required_tables) != len(set(required_tables))
    ):
        raise MacroMartPromotionError("macro_catalog_required_tables_invalid")
    if "macro_daily" not in required_tables:
        raise MacroMartPromotionError("macro_catalog_macro_not_required")
    missing_required = sorted(set(required_tables) - set(tables))
    if missing_required:
        raise MacroMartPromotionError(
            "macro_catalog_required_entry_missing:"
            + ",".join(missing_required)
        )
    for logical_table in sorted(required_tables):
        raw_entry = tables[logical_table]
        if not isinstance(raw_entry, dict):
            raise MacroMartPromotionError(
                f"macro_catalog_entry_invalid:{logical_table}"
            )
        _validate_required_catalog_table(
            schema=STRICT_CATALOG_SCHEMA,
            market_root=market_root,
            logical_table=logical_table,
            entry=raw_entry,
        )


def _strict_catalog_payload(
    *,
    old_catalog: Mapping[str, Any],
    market_root: Path,
    generation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    schema = str(old_catalog.get("schema_version") or "")
    if schema not in {LEGACY_CATALOG_SCHEMA, STRICT_CATALOG_SCHEMA}:
        raise MacroMartPromotionError("macro_catalog_schema_unsupported")
    try:
        payload = json.loads(
            json.dumps(old_catalog, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise MacroMartPromotionError("macro_catalog_not_json_safe") from exc
    if _contains_retired_intelligence(payload):
        raise MacroMartPromotionError(
            "macro_catalog_retired_intelligence_present"
        )
    tables = payload.get("tables")
    if not isinstance(tables, dict) or not tables:
        raise MacroMartPromotionError("macro_catalog_tables_invalid")
    required_tables = payload.get("required_tables")
    if (
        not isinstance(required_tables, list)
        or not required_tables
        or not all(
            isinstance(name, str) and bool(name.strip())
            for name in required_tables
        )
        or len(required_tables) != len(set(required_tables))
    ):
        raise MacroMartPromotionError("macro_catalog_required_tables_invalid")
    if "macro_daily" not in required_tables:
        raise MacroMartPromotionError("macro_catalog_macro_not_required")
    missing_required = sorted(set(required_tables) - set(tables))
    if missing_required:
        raise MacroMartPromotionError(
            "macro_catalog_required_entry_missing:"
            + ",".join(missing_required)
        )
    for logical_table, raw_entry in sorted(tables.items()):
        if not isinstance(raw_entry, dict):
            raise MacroMartPromotionError(
                f"macro_catalog_entry_invalid:{logical_table}"
            )
        if logical_table == "macro_daily":
            continue
        for field_name in ("path", "table_root"):
            if raw_entry.get(field_name):
                raw_entry[field_name] = _existing_catalog_member_relative(
                    market_root,
                    raw_entry[field_name],
                    blocker=f"macro_catalog_member_invalid:{logical_table}:{field_name}",
                )
        if logical_table in required_tables:
            _validate_required_catalog_table(
                schema=schema,
                market_root=market_root,
                logical_table=logical_table,
                entry=raw_entry,
            )
    generation_id = str(generation_manifest["generation_id"])
    generation_relative = Path("macro_daily") / "_generations" / generation_id
    table_relative = generation_relative / "part.parquet"
    manifest_relative = generation_relative / "manifest.json"
    provider_relative = generation_relative / "provider_bundle.json"
    for member in (table_relative, manifest_relative, provider_relative):
        _resolve_catalog_member(
            market_root,
            member.as_posix(),
            blocker="macro_generation_member_invalid",
        )
    table_path = market_root / table_relative
    manifest_path = market_root / manifest_relative
    provider_path = market_root / provider_relative
    expected_table_sha = _assert_sha256(
        generation_manifest.get("parquet_sha256"),
        blocker="macro_generation_table_hash_invalid",
    )
    expected_manifest_sha = _assert_sha256(
        generation_manifest.get("generation_manifest_sha256"),
        blocker="macro_generation_manifest_hash_invalid",
    )
    expected_provider_sha = _assert_sha256(
        generation_manifest.get("provider_bundle_sha256"),
        blocker="macro_generation_provider_hash_invalid",
    )
    table_size = _read_verified_member(
        table_path,
        trust_root=market_root,
        expected_sha256=expected_table_sha,
        hash_blocker="macro_generation_member_hash_mismatch",
        changed_blocker="macro_generation_member_changed",
        unreadable_blocker="macro_generation_member_unreadable",
        parser=lambda payload: (
            pd.read_parquet(io.BytesIO(payload), columns=[]),
            len(payload),
        )[1],
    )
    _read_verified_member(
        manifest_path,
        trust_root=market_root,
        expected_sha256=expected_manifest_sha,
        hash_blocker="macro_generation_member_hash_mismatch",
        changed_blocker="macro_generation_member_changed",
        unreadable_blocker="macro_generation_member_unreadable",
        parser=_parse_json_object,
    )
    _read_verified_member(
        provider_path,
        trust_root=market_root,
        expected_sha256=expected_provider_sha,
        hash_blocker="macro_generation_member_hash_mismatch",
        changed_blocker="macro_generation_member_changed",
        unreadable_blocker="macro_generation_member_unreadable",
        parser=_parse_json_object,
    )
    payload["schema_version"] = STRICT_CATALOG_SCHEMA
    tables["macro_daily"] = {
        "columns": sorted(_ALLOWED_INPUT_FIELDS),
        "date_column": "trade_date",
        "key_columns": ["trade_date"],
        "latest_date": _compact_trade_date(
            generation_manifest.get("as_of"),
            blocker="macro_generation_as_of_invalid",
        ),
        "logical_table": "macro_daily",
        "path": table_relative.as_posix(),
        "table_root": generation_relative.as_posix(),
        "generation_manifest": manifest_relative.as_posix(),
        "provider_bundle": provider_relative.as_posix(),
        "generation_id": generation_id,
        "parquet_sha256": expected_table_sha,
        "sha256": expected_table_sha,
        "generation_manifest_sha256": expected_manifest_sha,
        "provider_bundle_sha256": expected_provider_sha,
        "row_count": int(generation_manifest.get("row_count", 0)),
        "size_bytes": int(table_size),
        "snapshot_id": generation_id,
        "source": SOURCE_TUSHARE,
        "source_priority": SOURCE_TUSHARE,
        "status": "ok",
    }
    return payload


def _catalog_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _journal_payload(path: Path) -> dict[str, Any]:
    try:
        payload = _parse_json_object(_read_stable_bytes(path, blocker="macro_transaction_journal_invalid"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MacroMartPromotionError(
            "macro_transaction_journal_invalid"
        ) from exc
    if payload.get("schema_version") != TRANSACTION_JOURNAL_SCHEMA:
        raise MacroMartPromotionError("macro_transaction_journal_schema_invalid")
    return payload


def _set_journal_state(
    journal_path: Path,
    journal: Mapping[str, Any],
    *,
    state: str,
    detail: str = "",
) -> dict[str, Any]:
    updated = dict(journal)
    updated["state"] = state
    updated["updated_at"] = _now_utc()
    if detail:
        updated["detail"] = detail
    _atomic_json(journal_path, updated)
    return updated


def _recover_catalog_transactions(
    *,
    root: Path,
    catalog_path: Path,
) -> None:
    transactions_root = root / "_transactions"
    if not transactions_root.exists():
        return
    if transactions_root.is_symlink() or not transactions_root.is_dir():
        raise MacroMartPromotionError("macro_transactions_root_invalid")
    for transaction in sorted(transactions_root.iterdir()):
        if transaction.is_symlink() or not transaction.is_dir():
            raise MacroMartPromotionError("macro_transaction_path_invalid")
        journal_path = transaction / "journal.json"
        if not journal_path.exists():
            old_path = transaction / "old_catalog.json"
            new_path = transaction / "new_catalog.json"
            old_bytes = (
                _read_stable_bytes(
                    old_path,
                    blocker="macro_transaction_old_catalog_invalid",
                )
                if old_path.exists()
                else None
            )
            new_bytes = (
                _read_stable_bytes(
                    new_path,
                    blocker="macro_transaction_new_catalog_invalid",
                )
                if new_path.exists()
                else None
            )
            current = _read_stable_bytes(
                catalog_path,
                blocker="macro_catalog_invalid_during_recovery",
            )
            if new_bytes is not None and current == new_bytes:
                if old_bytes is None:
                    raise MacroMartPromotionError(
                        "macro_transaction_orphan_rollback_unavailable"
                    )
                _atomic_write_bytes(catalog_path, old_bytes)
                if _read_stable_bytes(
                    catalog_path,
                    blocker="macro_transaction_orphan_rollback_invalid",
                ) != old_bytes:
                    raise MacroMartPromotionError(
                        "macro_transaction_orphan_rollback_mismatch"
                    )
            elif old_bytes is not None and current != old_bytes:
                raise MacroMartPromotionError(
                    "macro_transaction_orphan_unknown_catalog_state"
                )
            elif old_bytes is None and new_bytes is not None:
                raise MacroMartPromotionError(
                    "macro_transaction_orphan_incomplete"
                )
            shutil.rmtree(transaction)
            _fsync_directory(transactions_root)
            continue
        journal = _journal_payload(journal_path)
        state = str(journal.get("state") or "")
        if state in {"committed", "rolled_back", "aborted"}:
            continue
        if state not in {"prepared", "switched"}:
            raise MacroMartPromotionError("macro_transaction_state_invalid")
        old_path = transaction / "old_catalog.json"
        new_path = transaction / "new_catalog.json"
        old_bytes = _read_stable_bytes(
            old_path,
            blocker="macro_transaction_old_catalog_invalid",
        )
        new_bytes = _read_stable_bytes(
            new_path,
            blocker="macro_transaction_new_catalog_invalid",
        )
        old_sha = _assert_sha256(
            journal.get("old_catalog_sha256"),
            blocker="macro_transaction_old_hash_invalid",
        )
        new_sha = _assert_sha256(
            journal.get("new_catalog_sha256"),
            blocker="macro_transaction_new_hash_invalid",
        )
        if (
            hashlib.sha256(old_bytes).hexdigest() != old_sha
            or hashlib.sha256(new_bytes).hexdigest() != new_sha
        ):
            raise MacroMartPromotionError("macro_transaction_catalog_hash_mismatch")
        current = _read_stable_bytes(
            catalog_path,
            blocker="macro_catalog_invalid_during_recovery",
        )
        current_sha = hashlib.sha256(current).hexdigest()
        if current_sha == old_sha:
            _set_journal_state(
                journal_path,
                journal,
                state="aborted",
                detail="catalog_remained_at_old_bytes",
            )
            continue
        if current_sha != new_sha:
            raise MacroMartPromotionError(
                "macro_transaction_unknown_catalog_state"
            )
        try:
            expected_pointer_sha = _assert_sha256(
                journal.get("expected_market_pointer_sha256"),
                blocker="macro_transaction_market_pointer_hash_invalid",
            )
            pointer_bytes = _read_stable_bytes(
                catalog_path.parent / "_latest.json",
                blocker="macro_transaction_market_pointer_unreadable",
            )
            if hashlib.sha256(pointer_bytes).hexdigest() != expected_pointer_sha:
                raise MacroMartPromotionError(
                    "macro_transaction_market_pointer_hash_mismatch"
                )
            recovered_catalog = _parse_json_object(new_bytes)
            _validate_strict_catalog_required_table_closure(
                recovered_catalog,
                market_root=catalog_path.parent,
            )
            read_macro_mart(data_root=root)
        except Exception as exc:
            current_check = _read_stable_bytes(
                catalog_path,
                blocker="macro_catalog_invalid_during_recovery",
            )
            if hashlib.sha256(current_check).hexdigest() != new_sha:
                raise MacroMartPromotionError(
                    "macro_transaction_rollback_unsafe"
                ) from exc
            _atomic_write_bytes(catalog_path, old_bytes)
            if _read_stable_bytes(
                catalog_path,
                blocker="macro_transaction_rollback_readback_invalid",
            ) != old_bytes:
                raise MacroMartPromotionError(
                    "macro_transaction_rollback_readback_mismatch"
                )
            _set_journal_state(
                journal_path,
                journal,
                state="rolled_back",
                detail=str(exc)[:300] or "new_catalog_failed_readback",
            )
            continue
        _set_journal_state(
            journal_path,
            journal,
            state="committed",
            detail="recovered_valid_switched_catalog",
        )


def _prepare_catalog_transaction(
    *,
    root: Path,
    run_id: str,
    old_catalog_bytes: bytes,
    new_catalog_bytes: bytes,
    generation_id: str,
    expected_market_pointer_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    transactions_root = _safe_directory(
        root / "_transactions",
        blocker="macro_transactions_root_invalid",
    )
    transaction = transactions_root / run_id
    if transaction.is_symlink():
        raise MacroMartPromotionError("macro_transaction_exists")
    if transaction.exists():
        journal_path = transaction / "journal.json"
        if not journal_path.exists():
            raise MacroMartPromotionError("macro_transaction_journal_missing")
        existing = _journal_payload(journal_path)
        if str(existing.get("state") or "") not in {"aborted", "rolled_back"}:
            raise MacroMartPromotionError("macro_transaction_exists")
        shutil.rmtree(transaction)
        _fsync_directory(transactions_root)
    transaction.mkdir(mode=0o700)
    os.chmod(transaction, 0o700)
    old_path = transaction / "old_catalog.json"
    new_path = transaction / "new_catalog.json"
    _atomic_write_bytes(old_path, old_catalog_bytes)
    _atomic_write_bytes(new_path, new_catalog_bytes)
    journal = {
        "schema_version": TRANSACTION_JOURNAL_SCHEMA,
        "run_id": run_id,
        "generation_id": generation_id,
        "state": "prepared",
        "old_catalog_sha256": hashlib.sha256(old_catalog_bytes).hexdigest(),
        "new_catalog_sha256": hashlib.sha256(new_catalog_bytes).hexdigest(),
        "expected_market_pointer_sha256": _assert_sha256(
            expected_market_pointer_sha256,
            blocker="macro_transaction_market_pointer_hash_invalid",
        ),
        "created_at": _now_utc(),
        "updated_at": _now_utc(),
    }
    journal_path = transaction / "journal.json"
    _atomic_json(journal_path, journal)
    _fsync_directory(transaction)
    _fsync_directory(transactions_root)
    return journal_path, journal


def _publish_catalog_generation(
    *,
    root: Path,
    run_id: str,
    old_catalog_bytes: bytes,
    new_catalog: Mapping[str, Any],
    market_pointer_path: Path,
    market_pointer_bytes: bytes,
    market_pointer_signature: tuple[int, ...],
    catalog_signature: tuple[int, ...],
    bar_root: Path,
    market_input_evidence: list[dict[str, Any]],
    generation_manifest: Mapping[str, Any],
    attestation: _PrimaryMacroAttestation,
) -> dict[str, Any]:
    if (
        attestation.capability is not _PRIMARY_MACRO_CAPABILITY
        or attestation.transform_version != TRANSFORM_VERSION
        or attestation.provider_bundle_sha256
        != generation_manifest.get("provider_bundle_sha256")
        or attestation.canonical_market_pointer_sha256
        != hashlib.sha256(market_pointer_bytes).hexdigest()
        or attestation.market_input_files_sha256
        != generation_manifest.get("market_input_files_sha256")
        or attestation.market_formula_universe_sha256
        != generation_manifest.get("market_formula_universe_sha256")
        or attestation.output_frame_sha256
        != generation_manifest.get("primary_provenance", {}).get(
            "output_frame_sha256"
        )
    ):
        raise MacroMartPromotionError("macro_primary_attestation_invalid")
    catalog_path = root.parent / "_catalog.json"
    current_catalog = _read_stable_bytes(
        catalog_path,
        blocker="macro_catalog_invalid_before_switch",
    )
    if (
        current_catalog != old_catalog_bytes
        or _stat_signature(os.lstat(catalog_path)) != catalog_signature
    ):
        raise MacroMartPromotionError("macro_catalog_cas_mismatch")
    current_pointer = _read_stable_bytes(
        market_pointer_path,
        blocker="macro_market_pointer_invalid_before_switch",
    )
    if (
        current_pointer != market_pointer_bytes
        or _stat_signature(os.lstat(market_pointer_path))
        != market_pointer_signature
    ):
        raise MacroMartPromotionError("macro_market_pointer_cas_mismatch")
    _validate_strict_catalog_required_table_closure(
        new_catalog,
        market_root=catalog_path.parent,
    )
    _verify_market_input_evidence(bar_root, market_input_evidence)
    new_catalog_bytes = _catalog_bytes(new_catalog)
    journal_path, journal = _prepare_catalog_transaction(
        root=root,
        run_id=run_id,
        old_catalog_bytes=old_catalog_bytes,
        new_catalog_bytes=new_catalog_bytes,
        generation_id=str(generation_manifest.get("generation_id") or ""),
        expected_market_pointer_sha256=hashlib.sha256(
            market_pointer_bytes
        ).hexdigest(),
    )
    new_sha = hashlib.sha256(new_catalog_bytes).hexdigest()
    try:
        _atomic_write_bytes(catalog_path, new_catalog_bytes)
        readback = _read_stable_bytes(
            catalog_path,
            blocker="macro_catalog_readback_invalid",
        )
        if readback != new_catalog_bytes:
            raise MacroMartPromotionError("macro_catalog_readback_mismatch")
        journal = _set_journal_state(
            journal_path,
            journal,
            state="switched",
        )
        switched_catalog = _parse_json_object(readback)
        _validate_strict_catalog_required_table_closure(
            switched_catalog,
            market_root=catalog_path.parent,
        )
        _, loaded_manifest = read_macro_mart(data_root=root)
        pointer_after = _read_stable_bytes(
            market_pointer_path,
            blocker="macro_market_pointer_invalid_after_switch",
        )
        if (
            pointer_after != market_pointer_bytes
            or _stat_signature(os.lstat(market_pointer_path))
            != market_pointer_signature
        ):
            raise MacroMartPromotionError(
                "macro_market_pointer_changed_during_switch"
            )
        _set_journal_state(
            journal_path,
            journal,
            state="committed",
        )
        return {
            "catalog_sha256": new_sha,
            "generation_manifest": loaded_manifest,
            "transaction_journal": str(journal_path),
        }
    except Exception as exc:
        current = _read_stable_bytes(
            catalog_path,
            blocker="macro_catalog_invalid_during_rollback",
        )
        if hashlib.sha256(current).hexdigest() != new_sha:
            raise MacroMartPromotionError("macro_catalog_rollback_unsafe") from exc
        _atomic_write_bytes(catalog_path, old_catalog_bytes)
        _set_journal_state(
            journal_path,
            journal,
            state="rolled_back",
            detail=str(exc)[:300],
        )
        raise


def _current_macro_is_equivalent(
    *,
    root: Path,
    trade_date: str,
    market_pointer_sha256: str,
    current_at: datetime,
) -> dict[str, Any] | None:
    try:
        frame, manifest = read_macro_mart(data_root=root)
    except MacroMartPromotionError:
        return None
    provenance = manifest.get("primary_provenance")
    if not isinstance(provenance, Mapping):
        return None
    formula_universe = manifest.get("market_formula_universe")
    if not isinstance(formula_universe, Mapping):
        return None
    try:
        validated_formula_universe = _validate_market_formula_universe(
            formula_universe,
            trade_date=trade_date,
        )
    except MacroMartPromotionError:
        return None
    formula_universe_sha = _canonical_json_sha256(
        validated_formula_universe
    )
    if (
        _compact_trade_date(
            manifest.get("as_of"),
            blocker="macro_generation_as_of_invalid",
        )
        != trade_date
        or provenance.get("canonical_market_pointer_sha256")
        != market_pointer_sha256
        or provenance.get("transform_version") != TRANSFORM_VERSION
        or manifest.get("market_formula_universe_sha256")
        != formula_universe_sha
        or provenance.get("market_formula_universe_sha256")
        != formula_universe_sha
        or "applied" in manifest
    ):
        return None
    provider_path = Path(str(manifest.get("resolved_provider_bundle") or ""))
    try:
        provider_sha = _assert_sha256(
            manifest.get("provider_bundle_sha256"),
            blocker="macro_current_provider_bundle_hash_invalid",
        )
        provider_bundle = _read_verified_member(
            provider_path,
            trust_root=root.parent,
            expected_sha256=provider_sha,
            hash_blocker="macro_current_provider_bundle_hash_mismatch",
            changed_blocker="macro_current_provider_bundle_changed",
            unreadable_blocker="macro_current_provider_bundle_invalid",
            parser=_parse_json_object,
        )
        _validate_provider_bundle_current_freshness(
            provider_bundle,
            current_at=current_at,
        )
    except MacroMartPromotionError as exc:
        if str(exc).startswith(
            "macro_provider_stale_new_run_id_required:"
        ):
            return None
        raise
    return {
        "status": "already_current",
        "promoted": False,
        "run_id": str(manifest.get("generation_id") or ""),
        "catalog_sha256": _sha256(root.parent / "_catalog.json"),
        "market_pointer_sha256": market_pointer_sha256,
        "manifest": manifest,
        "row": frame.iloc[-1].to_dict(),
    }


def refresh_cn_macro_mart(
    *,
    market: str = "CN",
    as_of: str = "",
    data_root: str | Path = DEFAULT_MACRO_ROOT,
    run_id: str,
    expected_catalog_sha256: str,
    expected_market_pointer_sha256: str,
    allow_live: bool = False,
) -> dict[str, Any]:
    """Build and atomically bind the latest-session live Macro mart."""

    if str(market).upper() != "CN":
        raise MacroMartPromotionError("macro_market_unsupported")
    if not allow_live:
        raise MacroMartPromotionError("macro_live_not_authorized")
    generation_id = _safe_run_id(run_id)
    expected_catalog_sha = _assert_sha256(
        expected_catalog_sha256,
        blocker="macro_expected_catalog_hash_invalid",
    )
    expected_pointer_sha = _assert_sha256(
        expected_market_pointer_sha256,
        blocker="macro_expected_market_pointer_hash_invalid",
    )
    root = _strict_read_root(Path(data_root).expanduser())
    market_root = root.parent
    catalog_path = market_root / "_catalog.json"
    pointer_path = market_root / "_latest.json"
    if not catalog_path.exists() or not pointer_path.exists():
        raise MacroMartPromotionError("macro_canonical_pointer_missing")

    with _catalog_writer_lock(market_root):
        _recover_catalog_transactions(root=root, catalog_path=catalog_path)
        catalog_bytes, catalog = _read_verified_bytes_and_json(
            catalog_path,
            trust_root=market_root,
            expected_sha256=expected_catalog_sha,
            hash_blocker="macro_expected_catalog_hash_mismatch",
            changed_blocker="macro_catalog_changed_during_read",
            unreadable_blocker="macro_catalog_invalid",
        )
        pointer_bytes, pointer = _read_verified_bytes_and_json(
            pointer_path,
            trust_root=market_root,
            expected_sha256=expected_pointer_sha,
            hash_blocker="macro_expected_market_pointer_hash_mismatch",
            changed_blocker="macro_market_pointer_changed_during_read",
            unreadable_blocker="macro_market_pointer_invalid",
        )
        catalog_signature = _stat_signature(os.lstat(catalog_path))
        pointer_signature = _stat_signature(os.lstat(pointer_path))

    captured_at = _utc_now()
    trade_date = _validate_live_target(
        pointer,
        requested_as_of=as_of,
        captured_at=captured_at,
        enforce_capture_window=False,
    )
    equivalent = _current_macro_is_equivalent(
        root=root,
        trade_date=trade_date,
        market_pointer_sha256=expected_pointer_sha,
        current_at=captured_at,
    )
    if equivalent is not None:
        return equivalent
    _validate_live_target(
        pointer,
        requested_as_of=as_of,
        captured_at=captured_at,
        enforce_capture_window=True,
    )
    bar_root = _resolve_bar_root(market_root, pointer)
    retry_generation = _load_primary_generation_for_retry(
        root=root,
        run_id=generation_id,
        trade_date=trade_date,
        market_pointer_sha256=expected_pointer_sha,
    )
    if retry_generation is not None:
        (
            generation_manifest,
            attestation,
            frame,
            market_evidence,
            provider_bundle,
        ) = retry_generation
        _verify_market_input_evidence(bar_root, market_evidence)
    else:
        client = _build_tushare_client()
        provider_bundle = _fetch_provider_bundle(
            client=client,
            trade_date=trade_date,
            captured_at=captured_at,
        )
        _validate_live_target(
            pointer,
            requested_as_of=as_of,
            captured_at=_utc_now(),
            enforce_capture_window=True,
        )
        market_frame, market_evidence, market_files_sha = (
            _load_market_inputs(
                bar_root,
                trade_date=trade_date,
            )
        )
        frame, market_formula_universe = _derive_macro_frame(
            market_frame,
            trade_date=trade_date,
            provider_bundle=provider_bundle,
        )
        generation_manifest, attestation = _write_primary_generation(
            root=root,
            run_id=generation_id,
            frame=frame,
            provider_bundle=provider_bundle,
            market_pointer_sha256=expected_pointer_sha,
            market_input_evidence=market_evidence,
            market_input_files_sha256=market_files_sha,
            market_formula_universe=market_formula_universe,
        )
    new_catalog = _strict_catalog_payload(
        old_catalog=catalog,
        market_root=market_root,
        generation_manifest=generation_manifest,
    )
    with _catalog_writer_lock(market_root):
        _recover_catalog_transactions(root=root, catalog_path=catalog_path)
        switch_at = _utc_now()
        _validate_live_target(
            pointer,
            requested_as_of=as_of,
            captured_at=switch_at,
            enforce_capture_window=True,
        )
        _validate_provider_bundle_current_freshness(
            provider_bundle,
            current_at=switch_at,
        )
        published = _publish_catalog_generation(
            root=root,
            run_id=generation_id,
            old_catalog_bytes=catalog_bytes,
            new_catalog=new_catalog,
            market_pointer_path=pointer_path,
            market_pointer_bytes=pointer_bytes,
            market_pointer_signature=pointer_signature,
            catalog_signature=catalog_signature,
            bar_root=bar_root,
            market_input_evidence=market_evidence,
            generation_manifest=generation_manifest,
            attestation=attestation,
        )
    return {
        "status": "promoted",
        "promoted": True,
        "run_id": generation_id,
        "catalog_sha256": published["catalog_sha256"],
        "previous_catalog_sha256": expected_catalog_sha,
        "market_pointer_sha256": expected_pointer_sha,
        "manifest": published["generation_manifest"],
        "transaction_journal": published["transaction_journal"],
        "row": frame.iloc[0].to_dict(),
    }


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


def _validate_provider_bundle(
    bundle: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> None:
    if (
        bundle.get("schema_version") != PROVIDER_BUNDLE_SCHEMA
        or bundle.get("provider_id") != "tushare_pro"
        or bundle.get("source") != SOURCE_TUSHARE
        or bundle.get("source_priority") != SOURCE_TUSHARE
        or bundle.get("live_requested") is not True
        or bundle.get("historical_replay_eligible") is not False
        or bundle.get("official_release_timestamps_claimed") is not False
    ):
        raise MacroMartPromotionError(
            "macro_provider_bundle_contract_invalid"
        )
    if _date_text(bundle.get("trade_date")) != _date_text(
        manifest.get("as_of")
    ):
        raise MacroMartPromotionError("macro_provider_bundle_as_of_mismatch")
    cutoff = pd.to_datetime(
        str(bundle.get("decision_cutoff_at") or ""),
        errors="coerce",
        utc=True,
    )
    fetched = pd.to_datetime(
        str(bundle.get("fetched_at") or ""),
        errors="coerce",
        utc=True,
    )
    if pd.isna(cutoff) or pd.isna(fetched) or cutoff != fetched:
        raise MacroMartPromotionError(
            "macro_provider_bundle_cutoff_invalid"
        )
    capture_month = fetched.tz_convert(_SHANGHAI).strftime("%Y%m")
    endpoints = bundle.get("endpoints")
    if not isinstance(endpoints, Mapping) or set(endpoints) != set(
        _ENDPOINT_SPECS
    ):
        raise MacroMartPromotionError(
            "macro_provider_bundle_endpoint_set_invalid"
        )
    selected = bundle.get("selected_inputs")
    if not isinstance(selected, Mapping) or set(selected) != set(
        _ENDPOINT_SPECS
    ):
        raise MacroMartPromotionError(
            "macro_provider_bundle_selected_inputs_invalid"
        )
    for endpoint in sorted(_ENDPOINT_SPECS):
        entry = endpoints[endpoint]
        chosen = selected[endpoint]
        if not isinstance(entry, Mapping) or not isinstance(chosen, Mapping):
            raise MacroMartPromotionError(
                "macro_provider_bundle_endpoint_invalid"
            )
        records = entry.get("records")
        if (
            not isinstance(records, list)
            or not records
            or not all(isinstance(item, Mapping) for item in records)
        ):
            raise MacroMartPromotionError(
                "macro_provider_bundle_records_invalid"
            )
        try:
            row_count = int(entry.get("row_count", -1))
        except (TypeError, ValueError) as exc:
            raise MacroMartPromotionError(
                "macro_provider_bundle_row_count_invalid"
            ) from exc
        if row_count != len(records):
            raise MacroMartPromotionError(
                "macro_provider_bundle_row_count_mismatch"
            )
        if str(entry.get("endpoint") or "") != endpoint:
            raise MacroMartPromotionError(
                "macro_provider_bundle_endpoint_identity_mismatch"
            )
        records_sha = _assert_sha256(
            entry.get("records_sha256"),
            blocker="macro_provider_bundle_records_hash_invalid",
        )
        if records_sha != hashlib.sha256(
            _canonical_json_bytes({"records": records})
        ).hexdigest():
            raise MacroMartPromotionError(
                "macro_provider_bundle_records_hash_mismatch"
            )
        spec = _ENDPOINT_SPECS[endpoint]
        normalized_rows: list[tuple[str, dict[str, float]]] = []
        for record in records:
            lookup = {
                str(key).strip().lower(): key for key in record
            }
            month_key = lookup.get(str(spec["month_field"]).lower())
            field_keys = {
                field: lookup.get(str(field).lower())
                for field in spec["value_fields"]
            }
            if month_key is None or any(
                key is None for key in field_keys.values()
            ):
                raise MacroMartPromotionError(
                    "macro_provider_bundle_record_schema_invalid"
                )
            month = _normalize_provider_month(record[month_key])
            if month > capture_month:
                raise MacroMartPromotionError(
                    "macro_provider_bundle_future_month_rejected"
                )
            values: dict[str, float] = {}
            for field, key in field_keys.items():
                try:
                    value = float(record[key])
                except (TypeError, ValueError) as exc:
                    raise MacroMartPromotionError(
                        "macro_provider_bundle_record_value_invalid"
                    ) from exc
                if not math.isfinite(value):
                    raise MacroMartPromotionError(
                        "macro_provider_bundle_record_value_invalid"
                    )
                values[field] = value
            normalized_rows.append((month, values))
        months = [item[0] for item in normalized_rows]
        if months != sorted(months) or len(months) != len(set(months)):
            raise MacroMartPromotionError(
                "macro_provider_bundle_record_months_invalid"
            )
        selected_month = _normalize_provider_month(chosen.get("month"))
        if selected_month != normalized_rows[-1][0]:
            raise MacroMartPromotionError(
                "macro_provider_bundle_selected_month_mismatch"
            )
        expected_latest = _expected_latest_provider_month(
            cutoff.to_pydatetime(),
            max_release_lag_days=int(spec["max_release_lag_days"]),
        )
        if selected_month < expected_latest:
            raise MacroMartPromotionError(
                f"macro_provider_latest_month_stale:{endpoint}"
            )
        declared_expected_latest = chosen.get(
            "expected_latest_month_lower_bound"
        )
        if (
            declared_expected_latest is not None
            and _normalize_provider_month(declared_expected_latest)
            != expected_latest
        ):
            raise MacroMartPromotionError(
                "macro_provider_bundle_selected_contract_invalid"
            )
        chosen_values = chosen.get("values")
        if not isinstance(chosen_values, Mapping):
            raise MacroMartPromotionError(
                "macro_provider_bundle_selected_values_invalid"
            )
        for field, expected_value in normalized_rows[-1][1].items():
            try:
                actual_value = float(chosen_values[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise MacroMartPromotionError(
                    "macro_provider_bundle_selected_values_invalid"
                ) from exc
            if not math.isfinite(actual_value) or actual_value != expected_value:
                raise MacroMartPromotionError(
                    "macro_provider_bundle_selected_values_mismatch"
                )
        observed_at = pd.to_datetime(
            str(chosen.get("observed_available_at") or ""),
            errors="coerce",
            utc=True,
        )
        if pd.isna(observed_at) or observed_at != cutoff:
            raise MacroMartPromotionError(
                "macro_provider_bundle_selected_input_after_cutoff"
            )
        try:
            chosen_lag = int(chosen.get("max_release_lag_days", -1))
        except (TypeError, ValueError) as exc:
            raise MacroMartPromotionError(
                "macro_provider_bundle_selected_contract_invalid"
            ) from exc
        if (
            chosen.get("official_release_timestamp_known") is not False
            or chosen_lag != int(spec["max_release_lag_days"])
            or str(chosen.get("transform_role") or "")
            != ("policy_signal" if endpoint == "cn_m" else "context_only")
        ):
            raise MacroMartPromotionError(
                "macro_provider_bundle_selected_contract_invalid"
            )


def _validate_primary_provenance(
    manifest: Mapping[str, Any],
    *,
    provider_bundle_sha256: str,
    table_sha256: str,
    output_frame_sha256: str,
) -> None:
    raw = manifest.get("primary_provenance")
    if not isinstance(raw, Mapping):
        raise MacroMartPromotionError(
            "macro_primary_provenance_missing"
        )
    envelope = dict(raw)
    declared = _assert_sha256(
        envelope.pop("envelope_sha256", ""),
        blocker="macro_primary_provenance_hash_invalid",
    )
    if _canonical_json_sha256(envelope) != declared:
        raise MacroMartPromotionError(
            "macro_primary_provenance_hash_mismatch"
        )
    required_hashes = (
        "provider_bundle_sha256",
        "canonical_market_pointer_sha256",
        "market_input_files_sha256",
        "output_frame_sha256",
        "output_parquet_sha256",
    )
    for field_name in required_hashes:
        _assert_sha256(
            envelope.get(field_name),
            blocker="macro_primary_provenance_contract_invalid",
        )
    market_files = manifest.get("market_input_files")
    if not isinstance(market_files, list) or not market_files:
        raise MacroMartPromotionError(
            "macro_primary_market_input_evidence_invalid"
        )
    normalized_paths: list[str] = []
    for entry in market_files:
        if not isinstance(entry, Mapping):
            raise MacroMartPromotionError(
                "macro_primary_market_input_evidence_invalid"
            )
        relative = Path(str(entry.get("path") or ""))
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
        ):
            raise MacroMartPromotionError(
                "macro_primary_market_input_evidence_invalid"
            )
        _assert_sha256(
            entry.get("sha256"),
            blocker="macro_primary_market_input_evidence_invalid",
        )
        try:
            size_bytes = int(entry.get("size_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise MacroMartPromotionError(
                "macro_primary_market_input_evidence_invalid"
            ) from exc
        if size_bytes <= 0:
            raise MacroMartPromotionError(
                "macro_primary_market_input_evidence_invalid"
            )
        normalized_paths.append(relative.as_posix())
    if (
        normalized_paths != sorted(normalized_paths)
        or len(normalized_paths) != len(set(normalized_paths))
        or _canonical_json_sha256({"files": market_files})
        != manifest.get("market_input_files_sha256")
    ):
        raise MacroMartPromotionError(
            "macro_primary_market_input_evidence_invalid"
        )
    formula_universe = manifest.get("market_formula_universe")
    formula_universe_sha_raw = manifest.get(
        "market_formula_universe_sha256"
    )
    provenance_formula_sha_raw = envelope.get(
        "market_formula_universe_sha256"
    )
    formula_fields_present = (
        formula_universe is not None,
        formula_universe_sha_raw is not None,
        provenance_formula_sha_raw is not None,
    )
    if not all(formula_fields_present) or not isinstance(
        formula_universe, Mapping
    ):
        raise MacroMartPromotionError(
            "macro_primary_formula_universe_evidence_invalid"
        )
    validated_formula_universe = _validate_market_formula_universe(
        formula_universe,
        trade_date=str(manifest.get("as_of") or ""),
    )
    formula_universe_sha = _assert_sha256(
        formula_universe_sha_raw,
        blocker="macro_primary_formula_universe_hash_invalid",
    )
    if (
        formula_universe_sha
        != _canonical_json_sha256(validated_formula_universe)
        or provenance_formula_sha_raw != formula_universe_sha
    ):
        raise MacroMartPromotionError(
            "macro_primary_formula_universe_hash_mismatch"
        )
    if (
        envelope.get("schema_version") != PRIMARY_PROVENANCE_SCHEMA
        or envelope.get("status") != "verified_live_tushare"
        or envelope.get("source") != SOURCE_TUSHARE
        or envelope.get("source_priority") != SOURCE_TUSHARE
        or envelope.get("transform_version") != TRANSFORM_VERSION
        or envelope.get("historical_replay_eligible") is not False
        or envelope.get("provider_bundle_sha256")
        != provider_bundle_sha256
        or envelope.get("output_parquet_sha256") != table_sha256
        or envelope.get("output_frame_sha256") != output_frame_sha256
        or envelope.get("market_input_files_sha256")
        != manifest.get("market_input_files_sha256")
        or _date_text(envelope.get("trade_date"))
        != _date_text(manifest.get("as_of"))
        or str(envelope.get("fetched_at") or "")
        != str(manifest.get("decision_cutoff_at") or "")
    ):
        raise MacroMartPromotionError(
            "macro_primary_provenance_contract_invalid"
        )


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
    catalog_provider_sha = _assert_sha256(
        entry.get("provider_bundle_sha256"),
        blocker="macro_catalog_provider_bundle_hash_invalid",
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
    provider_bundle_name = Path(
        str(generation_manifest.get("provider_bundle_path") or "").strip()
    )
    if (
        provider_bundle_name.is_absolute()
        or provider_bundle_name.parts != (provider_bundle_name.name,)
        or provider_bundle_name.name != "provider_bundle.json"
    ):
        raise MacroMartPromotionError(
            "macro_generation_provider_bundle_path_invalid"
        )
    provider_bundle_path = _resolve_catalog_member(
        market_root,
        (
            expected_generation_root.relative_to(market_root)
            / provider_bundle_name
        ).as_posix(),
        blocker="macro_generation_provider_bundle_path_invalid",
    )
    provider_bundle_sha = _assert_sha256(
        generation_manifest.get("provider_bundle_sha256"),
        blocker="macro_generation_provider_bundle_hash_invalid",
    )
    if provider_bundle_sha != catalog_provider_sha:
        raise MacroMartPromotionError(
            "macro_catalog_provider_bundle_hash_mismatch"
        )
    provider_bundle = _read_verified_member(
        provider_bundle_path,
        trust_root=market_root,
        expected_sha256=provider_bundle_sha,
        hash_blocker="macro_generation_provider_bundle_hash_mismatch",
        changed_blocker="macro_generation_provider_bundle_changed_during_read",
        unreadable_blocker="macro_generation_provider_bundle_invalid",
        parser=_parse_json_object,
    )
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
    _validate_provider_bundle(provider_bundle, manifest=generation_manifest)
    _validate_primary_provenance(
        generation_manifest,
        provider_bundle_sha256=provider_bundle_sha,
        table_sha256=table_sha,
        output_frame_sha256=_frame_sha256(frame),
    )
    manifest = {
        **generation_manifest,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_schema_version": str(catalog.get("schema_version")),
        "resolved_table_path": str(table_path),
        "resolved_generation_manifest": str(generation_manifest_path),
        "resolved_provider_bundle": str(provider_bundle_path),
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
    "refresh_cn_macro_mart",
    "run_cn_macro_maintenance",
    "write_macro_mart",
]
