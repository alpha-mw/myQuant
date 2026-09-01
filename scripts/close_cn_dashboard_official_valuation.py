#!/usr/bin/env python3
"""Build a hash-bound, no-trade CN Dashboard strict-close valuation record.

The normal build path is fully offline: it consumes one exact-byte evidence
input produced from the registered strict Market snapshot and benchmark input.
The historical Tushare capture helper is retained only as a separately
authorized compatibility surface and is never called by ``main``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quant_investor.config import Config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.strategy_records.store import (
    content_sha256,
    load_registered_catalog,
    resolve_active_record_dirs,
)

from cn_dashboard_common import canonical_json_bytes

# The producer must never use a fixed holdings list.  This compatibility tuple
# is retained only for the deprecated, separately-authorized capture helper
# below; the offline build path resolves its symbols from the pointer-selected
# Parquet predecessor ledger.
LEGACY_CAPTURE_STOCK_CODES = (
    "002008.SZ",
    "002384.SZ",
    "002463.SZ",
    "002916.SZ",
    "605358.SH",
    "688183.SH",
)
INDEX_CODES = ("000300.SH", "000688.SH", "399006.SZ")
CAPITAL_CNY = 1_000_000.0
EVIDENCE_SCHEMA = "cn_dashboard_strict_market_close_evidence.v1"
HISTORICAL_HOLDINGS_LANE = "REGISTERED_HISTORICAL_HOLDINGS_STORAGE_ONLY"
PROJECT_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SYMBOL_RE = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
EXPECTED_RECORD_FILES = {
    "manifest.json",
    "manual_execution_manifest.json",
    "ledger_after_manual_switch.parquet",
    "pnl_summary.csv",
    "strict_market_close_evidence.json",
}
ORDINARY_PUBLICATION_CLASS = "ORDINARY_SAME_DAY_OFFICIAL_VALUATION"
LATE_PUBLICATION_CLASS = "LATE_OFFICIAL_VALUATION_PUBLICATION"
BATCH_PUBLICATION_CLASS = "BATCH_CATCH_UP_OFFICIAL_VALUATION"
LATE_PUBLICATION_SCHEMA = "myquant.strategy_record_publication_delay.v1"
LATE_PUBLICATION_REASON = "SHARED_CHECKOUT_SAFETY_GATE_DELAY"
ORDINARY_PUBLICATION_REASON = "SAME_DAY_OFFICIAL_VALUATION"
BATCH_PUBLICATION_REASON = "OWNER_AUTHORIZED_CONTINUOUS_CLOSE_CATCH_UP"
LATE_VALUATION_DATE = date(2026, 8, 21)
LATE_PUBLICATION_DATE = date(2026, 8, 22)
LATE_SOURCE_RECORD = "20260820_1321"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_raw(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    before = path.stat()
    first = path.read_bytes()
    middle = path.stat()
    second = path.read_bytes()
    after = path.stat()
    identities = {
        (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        for item in (before, middle, after)
    }
    if len(identities) != 1 or first != second:
        raise ValueError(f"{label} was unstable during read")
    return first


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _exact_row(frame: Any, code: str, trade_date: str) -> dict[str, Any]:
    if frame is None or getattr(frame, "empty", True):
        raise RuntimeError(f"Tushare returned no exact-date row: {code}")
    required = {"ts_code", "trade_date", "close"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Tushare response columns incomplete: {code}")
    rows = frame.loc[:, ["ts_code", "trade_date", "close"]].to_dict("records")
    exact = [
        row for row in rows if str(row["ts_code"]) == code and str(row["trade_date"]) == trade_date
    ]
    if len(exact) != 1:
        raise RuntimeError(f"Tushare exact-date row count is not one: {code}")
    close = float(exact[0]["close"])
    if close <= 0:
        raise RuntimeError(f"Tushare close is not positive: {code}")
    return {"ts_code": code, "trade_date": trade_date, "close": close}


def fetch_tushare_evidence(trade_date: str) -> dict[str, Any]:
    import tushare as ts  # type: ignore[import-not-found]

    pro = create_tushare_pro(ts, Config.TUSHARE_TOKEN, Config.TUSHARE_URL)
    if pro is None:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    stocks = [
        _exact_row(
            pro.daily(ts_code=code, trade_date=trade_date),
            code,
            trade_date,
        )
        for code in LEGACY_CAPTURE_STOCK_CODES
    ]
    indices = [
        _exact_row(
            pro.index_daily(
                ts_code=code,
                start_date=trade_date,
                end_date=trade_date,
            ),
            code,
            trade_date,
        )
        for code in INDEX_CODES
    ]
    return {
        # This helper is intentionally not used by the production build path.
        # Keep its legacy capture shape isolated so an explicitly authorized
        # capture tool can still be migrated without making provider claims in
        # an official record.
        "schema_version": "cn_dashboard_tushare_close_evidence.v1",
        "provider": "tushare.pro",
        "stock_api": "daily",
        "index_api": "index_daily",
        "trade_date": trade_date,
        "coverage": "exact_close",
        "previous_trading_day_ffill": False,
        "stocks": stocks,
        "indices": indices,
    }


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or PROJECT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is missing or invalid")
    return value


def _project_path(
    project_root: Path,
    value: Any,
    *,
    label: str,
    must_exist: bool = True,
    allow_absolute: bool = False,
) -> Path:
    """Resolve one project-relative evidence path without symlink fallback."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is missing")
    relative = Path(value)
    if (relative.is_absolute() and not allow_absolute) or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{label} path must be project-relative")
    path = relative if relative.is_absolute() else project_root / relative
    try:
        resolved = path.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} path is unavailable") from exc
    if relative.is_absolute() and allow_absolute:
        temporary_roots = {
            Path(tempfile.gettempdir()).resolve(strict=True),
            Path("/private/tmp").resolve(strict=True),
        }
        if not any(
            resolved == temporary_root or temporary_root in resolved.parents
            for temporary_root in temporary_roots
        ):
            raise ValueError(f"{label} absolute path is outside private tmp")
    else:
        try:
            resolved.relative_to(project_root.resolve(strict=True))
        except ValueError as exc:
            raise ValueError(f"{label} path escapes project") from exc
    if resolved != path.absolute():
        raise ValueError(f"{label} path is not lexically exact")
    return path


def _json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_evidence_input(
    path: Path, *, expected_sha256: str, project_root: Path
) -> tuple[dict[str, Any], bytes]:
    """Read an offline evidence input through a stable exact-byte gate."""

    expected = _require_sha(expected_sha256, "expected evidence SHA")
    candidate = _project_path(
        project_root,
        path.as_posix(),
        label="evidence input",
        allow_absolute=True,
    )
    raw = _stable_raw(candidate, label="evidence input")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ValueError("evidence input SHA mismatch")
    return _json_bytes(raw, label="evidence input"), raw


def _row_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _normalize_symbol(value: Any, *, label: str) -> str:
    symbol = str(value or "").strip().upper()
    if SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError(f"{label} symbol is invalid")
    return symbol


def _finite_positive(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} is not positive and finite")
    return number


def _compact_date(value: Any) -> str:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y%m%d")
    text = str(value or "").strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.replace("-", "")


def _exact_date(value: Any, *, expected: str, label: str) -> str:
    text = _compact_date(value)
    if text != expected:
        raise ValueError(f"{label} date mismatch")
    return text


def _iso_date(value: Any, *, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} is missing or invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} is not canonical ISO date")
    return parsed


def _aware_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


def _publication_contract(
    *,
    publication_class: str,
    expected_valuation_date: str | None,
    expected_publication_date: str | None,
    publication_delay_reason: str | None,
    trade_date: str,
    evidence_trade_date: Any,
    source_record: str,
    record_id: str,
    recorded_at_iso: str,
    receipt_id: str,
    receipt_sha256: str,
    receipt_created_at: str,
    checkpoint_digest: str,
) -> tuple[datetime, dict[str, Any] | None]:
    """Validate publication timing and build the typed late contract."""

    if publication_class not in {
        ORDINARY_PUBLICATION_CLASS,
        LATE_PUBLICATION_CLASS,
        BATCH_PUBLICATION_CLASS,
    }:
        raise ValueError("publication class is invalid")
    valuation_date = _iso_date(expected_valuation_date, label="expected valuation date")
    publication_date = _iso_date(expected_publication_date, label="expected publication date")
    _exact_date(
        trade_date,
        expected=valuation_date.strftime("%Y%m%d"),
        label="valuation trade date",
    )
    _exact_date(
        evidence_trade_date,
        expected=valuation_date.strftime("%Y%m%d"),
        label="valuation evidence date",
    )
    recorded_at = _aware_timestamp(recorded_at_iso, label="recorded_at_iso")
    recorded_shanghai = recorded_at.astimezone(ZoneInfo("Asia/Shanghai"))
    if recorded_shanghai.date() != publication_date:
        raise ValueError("recorded_at publication date mismatch")
    if publication_class == BATCH_PUBLICATION_CLASS:
        prefix = recorded_shanghai.strftime("%Y%m%d_%H%M%S")
        if re.fullmatch(re.escape(prefix) + r"-b[0-9]{2}", record_id) is None:
            raise ValueError("batch record_id does not bind recorded_at second and ordinal")
    elif record_id != recorded_shanghai.strftime("%Y%m%d_%H%M"):
        raise ValueError("record_id does not match recorded_at Shanghai minute")
    receipt_at = _aware_timestamp(receipt_created_at, label="continuity receipt created_at")
    if receipt_at > recorded_at:
        raise ValueError("continuity receipt is later than recorded_at")

    delay_days = (publication_date - valuation_date).days
    if publication_class == BATCH_PUBLICATION_CLASS:
        if publication_date < valuation_date:
            raise ValueError("batch publication predates valuation")
        if publication_delay_reason != BATCH_PUBLICATION_REASON:
            raise ValueError("batch publication reason is invalid")
        return recorded_shanghai, None
    if publication_class == ORDINARY_PUBLICATION_CLASS:
        if delay_days != 0:
            raise ValueError("ordinary publication must be same-day")
        if publication_delay_reason not in {
            None,
            "",
            ORDINARY_PUBLICATION_REASON,
        }:
            raise ValueError("ordinary publication reason is invalid")
        return recorded_shanghai, None

    if (
        valuation_date != LATE_VALUATION_DATE
        or publication_date != LATE_PUBLICATION_DATE
        or source_record != LATE_SOURCE_RECORD
        or publication_delay_reason != LATE_PUBLICATION_REASON
    ):
        raise ValueError("late publication fixed identity mismatch")
    if delay_days != 1:
        raise ValueError("late publication must be exactly one day")
    delay = {
        "schema_id": LATE_PUBLICATION_SCHEMA,
        "publication_class": LATE_PUBLICATION_CLASS,
        "expected_valuation_date": valuation_date.isoformat(),
        "evidence_date": valuation_date.isoformat(),
        "expected_publication_date": publication_date.isoformat(),
        "source_record": source_record,
        "continuity_receipt_id": receipt_id,
        "continuity_receipt_sha256": receipt_sha256,
        "continuity_receipt_created_at": receipt_created_at,
        "continuity_checkpoint_digest": checkpoint_digest,
        "recorded_at_iso": recorded_at_iso,
        "publication_delay_reason": LATE_PUBLICATION_REASON,
        "delay_days": 1,
        "historical_holdings_storage_authority": True,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
    }
    return recorded_shanghai, delay


def _read_benchmark_rows(
    *,
    project_root: Path,
    path_value: Any,
    declared_sha: Any,
    expected_date: str,
    index_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    path = _project_path(project_root, path_value, label="benchmark input")
    expected = _require_sha(declared_sha, "benchmark input SHA")
    raw = _stable_raw(path, label="benchmark input")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ValueError("benchmark input SHA mismatch")
    try:
        rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    except UnicodeDecodeError as exc:
        raise ValueError("benchmark input is not valid UTF-8") from exc
    if not rows:
        raise ValueError("benchmark input is empty")
    wanted = {
        code: row
        for code, row in (
            (
                _normalize_symbol(_row_value(item, "ts_code", "symbol"), label="evidence index"),
                item,
            )
            for item in index_rows
        )
    }
    if set(wanted) != set(INDEX_CODES):
        raise ValueError("benchmark evidence index coverage is invalid")
    observed_rows: dict[str, list[dict[str, str]]] = {code: [] for code in INDEX_CODES}
    for row in rows:
        code = str(row.get("ts_code") or row.get("symbol") or "").strip().upper()
        date_value = _compact_date(row.get("date") or row.get("trade_date"))
        if code in observed_rows and date_value == expected_date:
            observed_rows[code].append(row)
    for code, evidence_row in wanted.items():
        matches = observed_rows[code]
        if len(matches) != 1:
            raise ValueError(f"benchmark input exact-date row count is not one: {code}")
        source_close = _finite_positive(matches[0].get("close"), label=f"benchmark close:{code}")
        evidence_close = _finite_positive(
            _row_value(evidence_row, "close", "index_close"),
            label=f"evidence index close:{code}",
        )
        if not math.isclose(source_close, evidence_close, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"benchmark close mismatch: {code}")
    return path_value, observed


def validate_strict_market_close_evidence(
    evidence: dict[str, Any],
    *,
    project_root: Path,
    expected_symbols: set[str],
    expected_trade_date: str,
    expected_market_pointer_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the local strict-Parquet close evidence contract.

    The contract is deliberately independent of provider APIs.  Every stock
    close is checked against its exact serving Parquet bytes, while the three
    benchmark closes are checked against the bound Dashboard benchmark input.
    """

    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("strict market close evidence schema mismatch")
    if evidence.get("market") != "CN":
        raise ValueError("strict market close evidence market mismatch")
    if evidence.get("trade_date") != expected_trade_date:
        raise ValueError("strict market close evidence trade date mismatch")
    if any(
        token in json.dumps(evidence, ensure_ascii=False, sort_keys=True).lower()
        for token in ("tushare", "provider", "stock_api", "index_api")
    ):
        raise ValueError("strict evidence must not contain provider claims")

    pointer_path_value = evidence.get("market_pointer_path")
    pointer_path = _project_path(project_root, pointer_path_value, label="market pointer")
    pointer_sha = _require_sha(evidence.get("market_pointer_sha256"), "market pointer SHA")
    pointer_raw = _stable_raw(pointer_path, label="market pointer")
    if hashlib.sha256(pointer_raw).hexdigest() != pointer_sha:
        raise ValueError("market pointer SHA mismatch")
    if expected_market_pointer_sha256 is not None and pointer_sha != _require_sha(
        expected_market_pointer_sha256, "expected market pointer SHA"
    ):
        raise ValueError("market pointer expected SHA mismatch")
    pointer = _json_bytes(pointer_raw, label="market pointer")
    if pointer.get("status") not in (None, "OK"):
        raise ValueError("market pointer status is not OK")
    snapshot_id = str(evidence.get("snapshot_id") or "")
    latest_complete = str(evidence.get("latest_complete_trade_date") or "").replace("-", "")
    if not snapshot_id or latest_complete != expected_trade_date:
        raise ValueError("market snapshot identity is incomplete")
    if (
        pointer.get("snapshot_id") != snapshot_id
        or str(pointer.get("latest_complete_trade_date") or "").replace("-", "")
        != expected_trade_date
    ):
        raise ValueError("market pointer snapshot binding mismatch")

    manifest_path_value = evidence.get("snapshot_manifest_path")
    manifest_path = _project_path(project_root, manifest_path_value, label="snapshot manifest")
    manifest_sha = _require_sha(evidence.get("snapshot_manifest_sha256"), "snapshot manifest SHA")
    manifest_raw = _stable_raw(manifest_path, label="snapshot manifest")
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha:
        raise ValueError("snapshot manifest SHA mismatch")
    manifest = _json_bytes(manifest_raw, label="snapshot manifest")
    pointer_manifest = pointer.get("manifest_path")
    if not isinstance(pointer_manifest, str) or not pointer_manifest.strip():
        raise ValueError("market pointer manifest binding is missing")
    pointer_manifest_path = _project_path(
        project_root,
        (
            pointer_manifest
            if not Path(pointer_manifest).is_absolute()
            else Path(pointer_manifest).resolve().relative_to(project_root.resolve()).as_posix()
        ),
        label="market pointer manifest",
    )
    if pointer_manifest_path.absolute() != manifest_path.absolute():
        raise ValueError("market pointer manifest binding mismatch")
    if (
        manifest.get("market", "CN") != "CN"
        or manifest.get("snapshot_id") != snapshot_id
        or str(manifest.get("latest_complete_trade_date") or "").replace("-", "")
        != expected_trade_date
    ):
        raise ValueError("snapshot manifest identity mismatch")
    serving_root_value = manifest.get("derived_serving_root")
    if not isinstance(serving_root_value, str) or not serving_root_value:
        raise ValueError("snapshot serving root is missing")
    serving_root = Path(serving_root_value)
    if not serving_root.is_absolute():
        serving_root = project_root / serving_root
    serving_root = serving_root.resolve(strict=True)

    stock_rows = evidence.get("stocks")
    if not isinstance(stock_rows, list) or len(stock_rows) != len(expected_symbols):
        raise ValueError("strict evidence stock row count is invalid")
    observed_stock_symbols: set[str] = set()
    stock_close_by_symbol: dict[str, float] = {}
    stock_refs: dict[str, dict[str, str]] = {}
    for row in stock_rows:
        if not isinstance(row, dict):
            raise ValueError("strict evidence stock row is invalid")
        symbol = _normalize_symbol(
            _row_value(row, "symbol", "ts_code"), label="strict evidence stock"
        )
        if symbol in observed_stock_symbols or symbol not in expected_symbols:
            raise ValueError("strict evidence stock symbol coverage is invalid")
        observed_stock_symbols.add(symbol)
        _exact_date(row.get("trade_date"), expected=expected_trade_date, label=f"stock:{symbol}")
        close = _finite_positive(
            _row_value(row, "close", "recorded_price"), label=f"stock close:{symbol}"
        )
        path_value = _row_value(row, "serving_parquet_path", "serving_path", "path")
        path = _project_path(project_root, path_value, label=f"stock serving:{symbol}")
        declared_sha = _require_sha(
            _row_value(row, "serving_parquet_sha256", "parquet_sha256", "sha256"),
            f"stock serving SHA:{symbol}",
        )
        raw = _stable_raw(path, label=f"stock serving:{symbol}")
        if hashlib.sha256(raw).hexdigest() != declared_sha:
            raise ValueError(f"stock serving SHA mismatch: {symbol}")
        try:
            path_resolved = path.resolve(strict=True)
            path_resolved.relative_to(serving_root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"stock serving path is outside snapshot: {symbol}") from exc
        if (
            path.name != "bars.parquet"
            or path.parent.name != f"symbol={symbol}"
            or path.parent.parent.resolve(strict=True) != serving_root
        ):
            raise ValueError(f"stock serving path identity mismatch: {symbol}")
        try:
            frame = pd.read_parquet(io.BytesIO(raw))
        except Exception as exc:
            raise ValueError(f"stock serving Parquet is unreadable: {symbol}") from exc
        if frame.empty:
            raise ValueError(f"stock serving Parquet is empty: {symbol}")
        date_column = next(
            (column for column in ("trade_date", "date", "Date") if column in frame.columns),
            None,
        )
        if date_column is None or "close" not in frame.columns:
            raise ValueError(f"stock serving schema is incomplete: {symbol}")
        dates = frame[date_column].map(_compact_date)
        exact_rows = frame.loc[dates == expected_trade_date]
        if len(exact_rows) != 1:
            raise ValueError(f"stock serving exact-date row count is not one: {symbol}")
        served_close = _finite_positive(exact_rows.iloc[0]["close"], label=f"served close:{symbol}")
        if not math.isclose(served_close, close, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"stock close mismatch: {symbol}")
        stock_close_by_symbol[symbol] = close
        stock_refs[symbol] = {
            "path": str(path_value),
            "sha256": declared_sha,
        }
    if observed_stock_symbols != expected_symbols:
        raise ValueError("strict evidence stock symbols do not match predecessor holdings")

    index_rows = evidence.get("indices")
    if not isinstance(index_rows, list) or len(index_rows) != len(INDEX_CODES):
        raise ValueError("strict evidence index row count is invalid")
    observed_index_codes: set[str] = set()
    benchmark_path_value: Any = evidence.get("benchmark_input_path")
    benchmark_sha_value: Any = evidence.get("benchmark_input_sha256")
    for row in index_rows:
        if not isinstance(row, dict):
            raise ValueError("strict evidence index row is invalid")
        code = _normalize_symbol(
            _row_value(row, "ts_code", "symbol"), label="strict evidence index"
        )
        if code in observed_index_codes or code not in INDEX_CODES:
            raise ValueError("strict evidence index coverage is invalid")
        observed_index_codes.add(code)
        _exact_date(row.get("trade_date"), expected=expected_trade_date, label=f"index:{code}")
        if benchmark_path_value is None:
            benchmark_path_value = _row_value(row, "benchmark_input_path", "benchmark_path")
        if benchmark_sha_value is None:
            benchmark_sha_value = _row_value(row, "benchmark_input_sha256", "benchmark_sha256")
        if row.get("benchmark_input_path", benchmark_path_value) != benchmark_path_value:
            raise ValueError("benchmark input path binding is inconsistent")
        if row.get("benchmark_input_sha256", benchmark_sha_value) != benchmark_sha_value:
            raise ValueError("benchmark input SHA binding is inconsistent")
    if observed_index_codes != set(INDEX_CODES):
        raise ValueError("strict evidence index symbols do not match contract")
    benchmark_path, benchmark_sha = _read_benchmark_rows(
        project_root=project_root,
        path_value=benchmark_path_value,
        declared_sha=benchmark_sha_value,
        expected_date=expected_trade_date,
        index_rows=index_rows,
    )
    return {
        "market_pointer_path": str(pointer_path_value),
        "market_pointer_sha256": pointer_sha,
        "snapshot_manifest_path": str(manifest_path_value),
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_id": snapshot_id,
        "latest_complete_trade_date": expected_trade_date,
        "stocks": stock_close_by_symbol,
        "stock_refs": stock_refs,
        "indices": {
            _normalize_symbol(
                _row_value(row, "ts_code", "symbol"), label="index"
            ): _finite_positive(_row_value(row, "close", "index_close"), label="index close")
            for row in index_rows
        },
        "benchmark_input_path": str(benchmark_path),
        "benchmark_input_sha256": benchmark_sha,
    }


def _bound_artifact(
    *,
    record_root: Path,
    source_dir: Path,
    closure: dict[str, Any],
    path_key: str,
    sha_key: str,
    label: str,
) -> tuple[Path, bytes]:
    relative_value = closure.get(path_key)
    declared_sha = closure.get(sha_key)
    if not isinstance(relative_value, str) or not isinstance(declared_sha, str):
        raise ValueError(f"registered active closure {label} binding is missing")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"registered active closure {label} path is invalid")
    candidate = record_root / relative
    raw = _stable_raw(candidate, label=f"source {label}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != source_dir.resolve(strict=True):
        raise ValueError(f"registered active closure {label} escapes source record")
    if hashlib.sha256(raw).hexdigest() != declared_sha:
        raise ValueError(f"registered active closure {label} SHA mismatch")
    return candidate, raw


def resolve_registered_source(
    *, record_root: Path, expected_pointer_sha: str
) -> tuple[Path, dict[str, Any]]:
    record_root = record_root.resolve(strict=True)
    pointer_path = record_root / "_record_store" / "current.v1.json"
    first_pointer_raw = _stable_raw(pointer_path, label="strategy-record pointer")
    observed_pointer_sha = hashlib.sha256(first_pointer_raw).hexdigest()
    if observed_pointer_sha != expected_pointer_sha:
        raise ValueError("strategy-record expected pointer SHA drift")
    registered = load_registered_catalog(record_root)
    if registered is None:
        raise ValueError("strategy-record store is unregistered")
    pointer, catalog = registered
    active_dirs = resolve_active_record_dirs(record_root, pointer=pointer, catalog=catalog)
    second_pointer_raw = _stable_raw(pointer_path, label="strategy-record pointer")
    if first_pointer_raw != second_pointer_raw:
        raise ValueError("strategy-record pointer drifted during source resolution")
    if not active_dirs:
        raise ValueError("registered active record directory is missing")
    source_dir = active_dirs[0].resolve(strict=True)
    active_record_id = pointer.get("active_record_id")
    closure = pointer.get("active_closure")
    if (
        not isinstance(active_record_id, str)
        or not isinstance(closure, dict)
        or closure.get("record_id") != active_record_id
        or closure.get("relative_path") != active_record_id
        or source_dir.name != active_record_id
        or source_dir.parent != record_root
    ):
        raise ValueError("registered active record/closure identity mismatch")
    return source_dir, dict(closure)


def _single_pnl_row(raw: bytes) -> dict[str, str]:
    try:
        rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    except UnicodeDecodeError as exc:
        raise ValueError("source P&L is not valid UTF-8") from exc
    if len(rows) != 1:
        raise ValueError("source P&L must contain exactly one row")
    return rows[0]


def _empty_csv_payload(path: Path, *, label: str) -> None:
    raw = _stable_raw(path, label=label)
    try:
        rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    if rows:
        raise ValueError(f"{label} contains order/trade rows")


def _money_close(left: float, right: float) -> bool:
    return abs(left - right) <= 0.01


def _source_closure(
    *, record_root: Path, source_dir: Path, closure: dict[str, Any]
) -> dict[str, Any]:
    _manifest_path, manifest_raw = _bound_artifact(
        record_root=record_root,
        source_dir=source_dir,
        closure=closure,
        path_key="manifest_path",
        sha_key="manifest_sha256",
        label="manifest",
    )
    _manual_path, manual_raw = _bound_artifact(
        record_root=record_root,
        source_dir=source_dir,
        closure=closure,
        path_key="manual_manifest_path",
        sha_key="manual_manifest_sha256",
        label="manual manifest",
    )
    ledger_path, ledger_raw = _bound_artifact(
        record_root=record_root,
        source_dir=source_dir,
        closure=closure,
        path_key="ledger_path",
        sha_key="ledger_sha256",
        label="ledger",
    )
    _pnl_path, pnl_raw = _bound_artifact(
        record_root=record_root,
        source_dir=source_dir,
        closure=closure,
        path_key="pnl_path",
        sha_key="pnl_sha256",
        label="P&L",
    )
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    manual = json.loads(manual_raw.decode("utf-8-sig"))
    ledger_name = str(manual.get("effective_manual_ledger_path") or "")
    if (
        ledger_name != "ledger_after_manual_switch.parquet"
        or ledger_path.name != "ledger_after_manual_switch.parquet"
    ):
        raise ValueError("source effective ledger must be canonical Parquet")
    ledger_sha = hashlib.sha256(ledger_raw).hexdigest()
    if ledger_sha != manual.get("next_ledger_sha256") or ledger_sha != manual.get(
        "ledger_after_manual_switch_parquet_sha256"
    ):
        raise ValueError("source effective ledger SHA mismatch")
    if (
        manifest.get("timestamp") != source_dir.name
        or manifest.get("market") != "CN"
        or manifest.get("strategy") != "aggressive_tech_manufacturing"
        or manifest.get("manual_execution") != manual
    ):
        raise ValueError("source record identity mismatch")
    if (
        manifest.get("action_taken_today") not in (False, True)
        or any(
            float(manifest.get(key, -1)) < 0 for key in ("trade_count", "order_count", "fill_count")
        )
        or not isinstance(manual.get("applied_local_trades", []), list)
        or not isinstance(manual.get("applied_owner_declared_trades", []), list)
        or manual.get("rejected_or_pending_trades") not in ([], None)
    ):
        raise ValueError("source record trade closure is invalid")
    # An applied effective ledger may be the predecessor authority.  Its
    # trades/fills are historical facts and must never be copied into the new
    # no-trade valuation record.
    if not (
        manual.get("no_trade_performed") is True
        or str(manual.get("execution_status") or manual.get("status") or "")
        in {"owner_declared_manual_execution_applied", "applied_effective_ledger"}
    ):
        raise ValueError("source record is not an admitted holdings closure")
    if (
        manual.get("funding_events") not in (None, [])
        or float(manual.get("net_external_flow", 0)) != 0
        or float(manual.get("excluded_external_flow", 0)) != 0
    ):
        raise ValueError("source record has a non-zero effective external flow")

    source_ledger = pd.read_parquet(ledger_path)
    required_columns = {"symbol", "shares", "avg_cost", "cost_basis", "current_value"}
    if not required_columns.issubset(source_ledger.columns):
        raise ValueError("source ledger accounting columns are incomplete")
    if source_ledger["symbol"].astype(str).duplicated().any():
        raise ValueError("source ledger contains duplicate symbols")
    capital = float(manual.get("capital_cny", 0))
    cash = float(manual.get("cash_after", "nan"))
    market_value = float(manual.get("market_value_after", "nan"))
    total_value = float(manual.get("total_value_after", "nan"))
    portfolio_pnl = float(manual.get("portfolio_pnl_after", "nan"))
    ledger_market_value = float(source_ledger["current_value"].sum())
    if (
        not _money_close(capital, CAPITAL_CNY)
        or not _money_close(ledger_market_value, market_value)
        or not _money_close(cash + market_value, total_value)
        or not _money_close(total_value - CAPITAL_CNY, portfolio_pnl)
    ):
        raise ValueError("source owner-corrected one-million accounting mismatch")
    financial_state = manual.get("financial_state")
    if not isinstance(financial_state, dict):
        raise ValueError("source financial state is missing")
    if (
        manual.get("financial_state_sha256") != closure.get("financial_state_sha256")
        or financial_state.get("ledger_sha256") != ledger_sha
    ):
        raise ValueError("source financial-state closure mismatch")
    pnl = _single_pnl_row(pnl_raw)
    for key, expected in (
        ("cash_after", cash),
        ("market_value_after", market_value),
        ("total_value_after", total_value),
        ("portfolio_pnl_after", portfolio_pnl),
    ):
        if not _money_close(float(pnl[key]), expected):
            raise ValueError("source P&L/accounting identity mismatch")
    return {
        "manifest": manifest,
        "manual": manual,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manual_sha256": hashlib.sha256(manual_raw).hexdigest(),
        "ledger_path": ledger_path,
        "ledger_sha256": ledger_sha,
        "source_ledger": source_ledger,
    }


def build_record(
    *,
    staging_dir: Path,
    record_root: Path,
    source_dir: Path,
    registered_closure: dict[str, Any],
    record_id: str,
    trade_date: str,
    recorded_at_iso: str,
    evidence: dict[str, Any],
    project_root: Path | None = None,
    expected_market_pointer_sha256: str | None = None,
    source_pointer_sha256: str | None = None,
    source_catalog_generation_id: str | None = None,
    source_catalog_sha256: str | None = None,
    continuity_receipt_id: str | None = None,
    continuity_receipt_sha256: str | None = None,
    continuity_receipt_created_at: str | None = None,
    continuity_checkpoint_digest: str | None = None,
    evidence_input_sha256: str | None = None,
    evidence_raw: bytes | None = None,
    publication_class: str = ORDINARY_PUBLICATION_CLASS,
    expected_valuation_date: str | None = None,
    expected_publication_date: str | None = None,
    publication_delay_reason: str | None = ORDINARY_PUBLICATION_REASON,
) -> dict[str, Any]:
    if staging_dir.name != record_id or not staging_dir.is_dir():
        raise ValueError("staging directory/record identity mismatch")
    if any(staging_dir.iterdir()):
        raise ValueError("staging directory must be empty before offline build")
    project = (project_root or Path.cwd()).resolve(strict=True)
    if evidence.get("trade_date") != trade_date:
        raise ValueError("valuation evidence trade date mismatch")
    source = _source_closure(
        record_root=record_root,
        source_dir=source_dir,
        closure=registered_closure,
    )
    source_manual = source["manual"]
    source_ledger = source["source_ledger"]
    symbols = {
        _normalize_symbol(symbol, label="source ledger")
        for symbol in source_ledger["symbol"].astype(str)
    }
    if not symbols or len(symbols) != len(source_ledger):
        raise ValueError("source holdings must be a non-empty unique symbol set")
    evidence_check = validate_strict_market_close_evidence(
        evidence,
        project_root=project,
        expected_symbols=symbols,
        expected_trade_date=trade_date,
        expected_market_pointer_sha256=expected_market_pointer_sha256,
    )
    close_by_code = evidence_check["stocks"]

    # These values are governance preimages, not inferred state.  The caller
    # must pass the exact pointer/catalog/continuity values used by the future
    # seal-publish CAS.  Keeping them in both manifests makes the candidate
    # self-auditing before the manager mutates the registered pointer.
    source_pointer_sha = _require_sha(
        source_pointer_sha256,
        "source Strategy Record Store pointer SHA",
    )
    catalog_generation = str(source_catalog_generation_id or "").strip()
    if not catalog_generation:
        raise ValueError("source catalog generation ID is missing")
    catalog_sha = _require_sha(source_catalog_sha256, "source catalog SHA")
    receipt_id = str(continuity_receipt_id or "").strip()
    if not receipt_id:
        raise ValueError("continuity receipt ID is missing")
    receipt_sha = _require_sha(continuity_receipt_sha256, "continuity receipt SHA")
    receipt_created_at = str(continuity_receipt_created_at or "").strip()
    if not receipt_created_at:
        raise ValueError("continuity receipt created_at is missing")
    checkpoint_digest = _require_sha(continuity_checkpoint_digest, "continuity checkpoint digest")
    expected_checkpoint_digest = content_sha256(registered_closure)
    if checkpoint_digest != expected_checkpoint_digest:
        raise ValueError("continuity checkpoint digest does not match active closure")
    if evidence_input_sha256 is not None:
        evidence_input_sha256 = _require_sha(evidence_input_sha256, "evidence input SHA")
    if publication_class == ORDINARY_PUBLICATION_CLASS:
        if expected_valuation_date is None:
            expected_valuation_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
        if expected_publication_date is None:
            expected_publication_date = (
                _aware_timestamp(recorded_at_iso, label="recorded_at_iso")
                .astimezone(ZoneInfo("Asia/Shanghai"))
                .date()
                .isoformat()
            )
    recorded_at, publication_delay = _publication_contract(
        publication_class=publication_class,
        expected_valuation_date=expected_valuation_date,
        expected_publication_date=expected_publication_date,
        publication_delay_reason=publication_delay_reason,
        trade_date=trade_date,
        evidence_trade_date=evidence.get("trade_date"),
        source_record=source_dir.name,
        record_id=record_id,
        recorded_at_iso=recorded_at_iso,
        receipt_id=receipt_id,
        receipt_sha256=receipt_sha,
        receipt_created_at=receipt_created_at,
        checkpoint_digest=checkpoint_digest,
    )

    ledger = source_ledger.copy()
    ledger["current_price"] = ledger["symbol"].map(close_by_code)
    ledger["current_value"] = ledger["shares"] * ledger["current_price"]
    ledger["unrealized_pnl"] = ledger["current_value"] - ledger["cost_basis"]
    ledger["unrealized_pnl_pct"] = ledger["unrealized_pnl"] / ledger["cost_basis"]
    market_value = float(ledger["current_value"].sum())
    cash = float(source_manual["cash_after"])
    total_value = cash + market_value
    ledger["equity_sleeve_weight"] = ledger["current_value"] / market_value
    if "market_weight" in ledger.columns:
        ledger["market_weight"] = ledger["equity_sleeve_weight"]
    ledger["nav_weight"] = ledger["current_value"] / total_value
    for immutable_column in ("shares", "avg_cost", "cost_basis"):
        if not ledger[immutable_column].equals(source_ledger[immutable_column]):
            raise ValueError(f"no-trade valuation mutated {immutable_column}")

    parquet_path = staging_dir / "ledger_after_manual_switch.parquet"
    ledger.to_parquet(parquet_path, index=False)
    parquet_sha = _sha(parquet_path)

    evidence_path = staging_dir / "strict_market_close_evidence.json"
    if evidence_raw is not None:
        if _json_bytes(evidence_raw, label="evidence input") != evidence:
            raise ValueError("evidence input object/readback mismatch")
        evidence_path.write_bytes(evidence_raw)
    else:
        _write_json(evidence_path, evidence)
    evidence_sha = _sha(evidence_path)

    pnl = {
        "record_time": recorded_at_iso,
        "quote_snapshot": f"{trade_date}_STRICT_PARQUET_EXACT_CLOSE",
        "initial_capital": f"{CAPITAL_CNY:.2f}",
        "cash_before": f"{cash:.2f}",
        "market_value_before": (f"{float(source_manual['market_value_after']):.2f}"),
        "total_value_before": (f"{float(source_manual['total_value_after']):.2f}"),
        "portfolio_pnl_before": (f"{float(source_manual['portfolio_pnl_after']):.2f}"),
        "portfolio_pnl_pct_before": (f"{float(source_manual['portfolio_return_after']):.8f}"),
        "realized_pnl_from_rebalance": "0.00",
        "cash_after": f"{cash:.2f}",
        "market_value_after": f"{market_value:.2f}",
        "total_value_after": f"{total_value:.2f}",
        "portfolio_pnl_after": f"{total_value - CAPITAL_CNY:.2f}",
        "portfolio_pnl_pct_after": f"{total_value / CAPITAL_CNY - 1.0:.8f}",
        "delta_vs_source_record": (
            f"{total_value - float(source_manual['total_value_after']):.2f}"
        ),
    }
    pnl_path = staging_dir / "pnl_summary.csv"
    _write_csv(pnl_path, [pnl])

    recorded_at_text = recorded_at.strftime("%Y-%m-%d %H:%M:%S CST")
    financial_state = {
        "capital_cny": CAPITAL_CNY,
        "cash_after": round(cash, 2),
        "market_value_after": round(market_value, 2),
        "total_value_after": round(total_value, 2),
        "portfolio_pnl_after": round(total_value - CAPITAL_CNY, 2),
        "portfolio_return_after": total_value / CAPITAL_CNY - 1.0,
        "ledger_sha256": parquet_sha,
    }
    manual = {
        "schema_version": "cn_aggressive_manual_execution.v3",
        "record_origin": "official_strict_market_close_revaluation",
        "historical_lane": HISTORICAL_HOLDINGS_LANE,
        "historical_holdings_storage_authority": True,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
        "not_v17_prediction_or_forward_observation": True,
        "capital_cny": CAPITAL_CNY,
        "status": "no_action_carry_forward_official_valuation",
        "execution_status": "no_action_carry_forward_official_valuation",
        "manual_execution_mode": "official_strict_market_close_no_broker_api",
        "advisory_only": False,
        "local_manual_fills_allowed": False,
        "owner_reported_external_fills": False,
        "no_trade_performed": True,
        "record_timestamp": record_id,
        "recorded_at": recorded_at_text,
        "recorded_at_iso": recorded_at_iso,
        "publication_class": publication_class,
        "trade_date": trade_date,
        "valuation_trade_date": trade_date,
        "valuation_status": "OFFICIAL_STRICT_MARKET_CLOSE_COMPLETE",
        "official_valuation": True,
        "price_basis": "strict_parquet_market_close_hash_bound",
        "quote_source": "local_strict_parquet_market_evidence",
        "decision_data_sufficient": True,
        "holdings_completeness_passed": True,
        "valuation_completeness_passed": True,
        "completeness_passed": True,
        "automation_run_status": "completed_offline_official_strict_close",
        "review_state": "OFFICIAL_STRICT_MARKET_CLOSE_NO_ACTION",
        "blockers": [],
        "source_record": source_dir.name,
        "supersedes_record": source_dir.name,
        "source_manifest_sha256": source["manifest_sha256"],
        "source_manual_manifest_sha256": source["manual_sha256"],
        "source_contained_ledger_sha256": source["ledger_sha256"],
        "applied_local_trades": [],
        "applied_owner_declared_trades": [],
        "rejected_or_pending_trades": [],
        "funding_events": [],
        "net_external_flow": 0.0,
        "excluded_external_flow": 0.0,
        "trade_count": 0,
        "order_count": 0,
        "fill_count": 0,
        "effective_manual_ledger_path": "ledger_after_manual_switch.parquet",
        "next_ledger_path": "ledger_after_manual_switch.parquet",
        "next_ledger_sha256": parquet_sha,
        "ledger_after_manual_switch_parquet": ("ledger_after_manual_switch.parquet"),
        "ledger_after_manual_switch_parquet_sha256": parquet_sha,
        "pnl_summary_path": "pnl_summary.csv",
        "pnl_summary_sha256": _sha(pnl_path),
        "valuation_evidence_path": "strict_market_close_evidence.json",
        "valuation_evidence_sha256": evidence_sha,
        "valuation_evidence_input_sha256": evidence_input_sha256,
        "market_pointer_path": evidence_check["market_pointer_path"],
        "market_pointer_sha256": evidence_check["market_pointer_sha256"],
        "snapshot_manifest_path": evidence_check["snapshot_manifest_path"],
        "snapshot_manifest_sha256": evidence_check["snapshot_manifest_sha256"],
        "snapshot_id": evidence_check["snapshot_id"],
        "latest_complete_trade_date": evidence_check["latest_complete_trade_date"],
        "benchmark_input_path": evidence_check["benchmark_input_path"],
        "benchmark_input_sha256": evidence_check["benchmark_input_sha256"],
        "source_pointer_sha256": source_pointer_sha,
        "source_catalog_generation_id": catalog_generation,
        "source_catalog_sha256": catalog_sha,
        "continuity_receipt_id": receipt_id,
        "continuity_receipt_sha256": receipt_sha,
        "continuity_receipt_created_at": receipt_created_at,
        "continuity_checkpoint_digest": checkpoint_digest,
        "ledger_provenance": {
            "declared_next_ledger_path": "ledger_after_manual_switch.parquet",
            "contained_in_run_directory": True,
            "regular_non_symlink_file": True,
            "stable_double_read": True,
            "declared_sha256": parquet_sha,
            "parquet_sha256": parquet_sha,
            "source_record": source_dir.name,
            "source_ledger_sha256": source["ledger_sha256"],
            "official_strict_market_close_only": True,
            "position_quantities_unchanged": True,
        },
        "effective_manual_holding_count": len(ledger),
        "source_manual_holding_count": len(source_ledger),
        "cash_before": cash,
        "gross_trade_value": 0.0,
        "fees_cny": 0.0,
        "cash_after": round(cash, 2),
        "market_value_after": round(market_value, 2),
        "total_value_after": round(total_value, 2),
        "portfolio_pnl_after": round(total_value - CAPITAL_CNY, 2),
        "portfolio_return_after": total_value / CAPITAL_CNY - 1.0,
        "realized_pnl_from_rebalance": 0.0,
        "provider_quote_called": False,
        "no_provider_quote_called": True,
        "no_llm_gateway_called": True,
        "no_broker_api_called": True,
        "no_order_created_by_codex": True,
        "no_execution_performed_by_codex": True,
        "financial_state": financial_state,
        "financial_state_sha256": hashlib.sha256(canonical_json_bytes(financial_state)).hexdigest(),
    }
    if publication_delay is not None:
        manual["publication_delay"] = dict(publication_delay)
    manual_path = staging_dir / "manual_execution_manifest.json"
    _write_json(manual_path, manual)
    files = {
        "pnl_summary": "pnl_summary.csv",
        "manual_execution_manifest": "manual_execution_manifest.json",
        "ledger_after_manual_switch": "ledger_after_manual_switch.parquet",
        "valuation_evidence": "strict_market_close_evidence.json",
    }
    file_sha = {name: _sha(staging_dir / name) for name in files.values()}
    manifest = {
        "schema_version": "cn_aggressive_daily_transaction_record.v1",
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "historical_holdings_storage_authority": True,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
        "timestamp": record_id,
        "recorded_at": recorded_at_text,
        "recorded_at_iso": recorded_at_iso,
        "publication_class": publication_class,
        "source_record": source_dir.name,
        "supersedes_record": source_dir.name,
        "source_manifest_sha256": source["manifest_sha256"],
        "source_pointer_sha256": source_pointer_sha,
        "source_catalog_generation_id": catalog_generation,
        "source_catalog_sha256": catalog_sha,
        "continuity_receipt_id": receipt_id,
        "continuity_receipt_sha256": receipt_sha,
        "continuity_receipt_created_at": receipt_created_at,
        "continuity_checkpoint_digest": checkpoint_digest,
        "formal_record": True,
        "completeness_passed": True,
        "record_origin": "official_strict_market_close_revaluation",
        "historical_lane": HISTORICAL_HOLDINGS_LANE,
        "not_v17_prediction_or_forward_observation": True,
        "action_taken_today": False,
        "trade_count": 0,
        "order_count": 0,
        "fill_count": 0,
        "review_state": "OFFICIAL_STRICT_MARKET_CLOSE_NO_ACTION",
        "automation_run_status": "completed_offline_official_strict_close",
        "blockers": [],
        "files": files,
        "file_sha256": file_sha,
        "data_snapshot": {
            "analysis_trade_date": trade_date,
            "valuation_trade_date": trade_date,
            "valuation_status": "OFFICIAL_STRICT_MARKET_CLOSE_COMPLETE",
            "freshness_mode": "strict_parquet_market_close_hash_bound",
            "valuation_evidence_path": "strict_market_close_evidence.json",
            "valuation_evidence_sha256": evidence_sha,
            "valuation_evidence_input_sha256": evidence_input_sha256,
            "market_pointer_path": evidence_check["market_pointer_path"],
            "market_pointer_sha256": evidence_check["market_pointer_sha256"],
            "snapshot_manifest_path": evidence_check["snapshot_manifest_path"],
            "snapshot_manifest_sha256": evidence_check["snapshot_manifest_sha256"],
            "snapshot_id": evidence_check["snapshot_id"],
            "latest_complete_trade_date": evidence_check["latest_complete_trade_date"],
            "benchmark_input_path": evidence_check["benchmark_input_path"],
            "benchmark_input_sha256": evidence_check["benchmark_input_sha256"],
            "source_pointer_sha256": source_pointer_sha,
            "source_catalog_generation_id": catalog_generation,
            "source_catalog_sha256": catalog_sha,
            "continuity_receipt_id": receipt_id,
            "continuity_receipt_sha256": receipt_sha,
            "continuity_receipt_created_at": receipt_created_at,
            "continuity_checkpoint_digest": checkpoint_digest,
            "source_record_transaction_marks_preserved": False,
            "new_quote_requested": True,
        },
        "manual_execution": manual,
        "side_effects": {
            "provider_quote_called": False,
            "broker": False,
            "live_order": False,
            "live_execution": False,
            "actual_position_quantity_mutation": False,
            "actual_cash_mutation": False,
            "v17_active_pointer_mutation": False,
            "strategy_record_store_pointer_cas_by_manager": True,
            "factor_registry_mutation": False,
            "production_rule_mutation": False,
        },
    }
    if publication_delay is not None:
        manifest["publication_delay"] = dict(publication_delay)
    _write_json(staging_dir / "manifest.json", manifest)
    if {path.name for path in staging_dir.iterdir()} != EXPECTED_RECORD_FILES:
        raise ValueError("new valuation record must contain exactly five expected files")
    return {
        "record_id": record_id,
        "source_record": source_dir.name,
        "cash": round(cash, 2),
        "market_value": round(market_value, 2),
        "total_value": round(total_value, 2),
        "portfolio_pnl": round(total_value - CAPITAL_CNY, 2),
        "ledger_sha256": parquet_sha,
        "manual_manifest_sha256": _sha(manual_path),
        "manifest_sha256": _sha(staging_dir / "manifest.json"),
        "valuation_evidence_sha256": evidence_sha,
        "publication_class": publication_class,
        "publication_delay": (dict(publication_delay) if publication_delay is not None else None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--expected-pointer-sha", required=True)
    parser.add_argument("--expected-catalog-generation-id", required=True)
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument("--continuity-receipt-id", required=True)
    parser.add_argument("--continuity-receipt-sha256", required=True)
    parser.add_argument("--continuity-receipt-created-at", required=True)
    parser.add_argument("--continuity-checkpoint-digest", required=True)
    parser.add_argument("--continuity-receipt-input", type=Path)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument(
        "--publication-class",
        choices=(LATE_PUBLICATION_CLASS,),
        help="Explicit exceptional late-publication class; omit for ordinary same-day",
    )
    parser.add_argument("--expected-valuation-date")
    parser.add_argument("--expected-publication-date")
    parser.add_argument(
        "--publication-delay-reason",
        choices=(LATE_PUBLICATION_REASON,),
    )
    parser.add_argument("--evidence-input", type=Path, required=True)
    parser.add_argument("--expected-evidence-sha256", required=True)
    parser.add_argument("--expected-market-pointer-sha256")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    # Accepted only for a short migration window.  The offline producer never
    # writes or updates benchmark inputs.
    parser.add_argument("--benchmark-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    project_root = args.project_root.resolve(strict=True)
    record_root = args.record_root.resolve(strict=True)
    source_dir, registered_closure = resolve_registered_source(
        record_root=record_root,
        expected_pointer_sha=args.expected_pointer_sha,
    )
    pointer_path = record_root / "_record_store" / "current.v1.json"
    pointer_raw = _stable_raw(pointer_path, label="strategy-record pointer")
    pointer = _json_bytes(pointer_raw, label="strategy-record pointer")
    if hashlib.sha256(pointer_raw).hexdigest() != args.expected_pointer_sha:
        raise ValueError("strategy-record pointer SHA drifted during binding")
    catalog_path_value = pointer.get("catalog_path")
    if not isinstance(catalog_path_value, str) or Path(catalog_path_value).is_absolute():
        raise ValueError("registered catalog path is invalid")
    catalog_path = record_root / catalog_path_value
    catalog_raw = _stable_raw(catalog_path, label="registered catalog")
    catalog_sha = hashlib.sha256(catalog_raw).hexdigest()
    expected_catalog_sha = _require_sha(args.expected_catalog_sha256, "expected catalog SHA")
    if catalog_sha != expected_catalog_sha or pointer.get("catalog_sha256") != catalog_sha:
        raise ValueError("registered catalog SHA preimage mismatch")
    if pointer.get("generation_id") != args.expected_catalog_generation_id:
        raise ValueError("registered catalog generation preimage mismatch")
    evidence, evidence_raw = load_evidence_input(
        args.evidence_input,
        expected_sha256=args.expected_evidence_sha256,
        project_root=project_root,
    )
    receipt_input = None
    if args.continuity_receipt_input is not None:
        receipt_path = _project_path(
            project_root,
            args.continuity_receipt_input.as_posix(),
            label="continuity receipt input",
            allow_absolute=True,
        )
        receipt_raw = _stable_raw(receipt_path, label="continuity receipt input")
        if hashlib.sha256(receipt_raw).hexdigest() != _require_sha(
            args.continuity_receipt_sha256, "continuity receipt SHA"
        ):
            raise ValueError("continuity receipt SHA preimage mismatch")
        receipt_input = _json_bytes(receipt_raw, label="continuity receipt input")
        if receipt_input.get("receipt_id", receipt_input.get("id")) != args.continuity_receipt_id:
            raise ValueError("continuity receipt ID preimage mismatch")
        if receipt_input.get("created_at") != args.continuity_receipt_created_at:
            raise ValueError("continuity receipt created_at preimage mismatch")
        observed_checkpoint = receipt_input.get(
            "checkpoint_digest", receipt_input.get("checkpoint_digest_sha256")
        )
        if observed_checkpoint != args.continuity_checkpoint_digest:
            raise ValueError("continuity checkpoint digest preimage mismatch")
    summary = build_record(
        staging_dir=args.staging_dir,
        record_root=record_root,
        source_dir=source_dir,
        registered_closure=registered_closure,
        record_id=args.record_id,
        trade_date=args.trade_date,
        recorded_at_iso=now.isoformat(),
        evidence=evidence,
        project_root=project_root,
        expected_market_pointer_sha256=args.expected_market_pointer_sha256,
        source_pointer_sha256=args.expected_pointer_sha,
        source_catalog_generation_id=args.expected_catalog_generation_id,
        source_catalog_sha256=args.expected_catalog_sha256,
        continuity_receipt_id=args.continuity_receipt_id,
        continuity_receipt_sha256=args.continuity_receipt_sha256,
        continuity_receipt_created_at=args.continuity_receipt_created_at,
        continuity_checkpoint_digest=args.continuity_checkpoint_digest,
        evidence_input_sha256=args.expected_evidence_sha256,
        evidence_raw=evidence_raw,
        publication_class=args.publication_class or ORDINARY_PUBLICATION_CLASS,
        expected_valuation_date=args.expected_valuation_date,
        expected_publication_date=args.expected_publication_date,
        publication_delay_reason=(
            args.publication_delay_reason
            if args.publication_class is not None
            else ORDINARY_PUBLICATION_REASON
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
