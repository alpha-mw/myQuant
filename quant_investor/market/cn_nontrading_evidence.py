"""Hash-bound evidence for CN symbol-days without a primary daily bar.

``bak_daily`` publishes zero-trading placeholder rows for some listed symbols
that did not trade on a given date.  Those rows are useful coverage evidence,
but they are not price bars and they are not equivalent to a regulatory
``suspend_d`` event.  This module keeps that distinction explicit and never
writes the placeholder rows into canonical OHLCV storage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


BAK_DAILY_NONTRADING_SCHEMA_VERSION = "cn-bak-daily-nontrading-evidence.v1"
BAK_DAILY_NONTRADING_CLASSIFICATION = "verified_nontrading_bak_daily_zero"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _compact_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    digits = "".join(character for character in text if character.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_symbols(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            _normalize_symbol(value)
            for value in values
            if _normalize_symbol(value)
        }
    )


def _json_scalar(value: Any) -> Any:
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
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return 0.0 if value == 0 else value
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def symbol_set_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256(
        "\n".join(_normalize_symbols(values)).encode("utf-8")
    ).hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a provider frame independent of row and column order."""

    if frame is None or not isinstance(frame, pd.DataFrame):
        return canonical_json_sha256({"columns": [], "rows": []})
    columns = sorted(str(column) for column in frame.columns)
    rows = [
        {column: _json_scalar(row.get(column)) for column in columns}
        for row in frame.to_dict(orient="records")
    ]
    rows.sort(
        key=lambda row: json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    return canonical_json_sha256({"columns": columns, "rows": rows})


def _numeric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_zero(value: Any, *, tolerance: float = 1e-12) -> bool:
    parsed = _numeric(value)
    return parsed is not None and abs(parsed) <= tolerance


def _equal_numeric(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    left_number = _numeric(left)
    right_number = _numeric(right)
    if left_number is None or right_number is None:
        return False
    return math.isclose(left_number, right_number, rel_tol=tolerance, abs_tol=tolerance)


def _validate_zero_row(row: Mapping[str, Any], *, trade_date: str) -> list[str]:
    reasons: list[str] = []
    if _compact_trade_date(row.get("trade_date")) != trade_date:
        reasons.append("trade_date_mismatch")
    for column in ("open", "high", "low", "vol", "amount"):
        if column not in row:
            reasons.append(f"{column}_missing")
        elif not _is_zero(row.get(column)):
            reasons.append(f"{column}_nonzero_or_invalid")
    close = _numeric(row.get("close"))
    pre_close = _numeric(row.get("pre_close"))
    if close is None or close <= 0:
        reasons.append("close_nonpositive_or_invalid")
    if pre_close is None or pre_close <= 0:
        reasons.append("pre_close_nonpositive_or_invalid")
    if not _equal_numeric(close, pre_close):
        reasons.append("close_pre_close_mismatch")
    for column in ("change", "pct_change", "pct_chg"):
        if column in row and _json_scalar(row.get(column)) is not None:
            if not _is_zero(row.get(column)):
                reasons.append(f"{column}_nonzero_or_invalid")
    return reasons


def build_bak_daily_nontrading_evidence(
    frame: pd.DataFrame,
    *,
    trade_date: str,
    primary_missing_symbols: Iterable[Any],
    query_params: Mapping[str, Any],
    pit_membership_path: str | Path,
    pit_membership_sha256: str,
    query_succeeded: bool = True,
    query_error: str = "",
) -> dict[str, Any]:
    """Build exact-date typed evidence for PIT-active primary-bar absences."""

    target_date = _compact_trade_date(trade_date)
    if not target_date:
        raise ValueError("trade_date must be YYYYMMDD-compatible")
    candidates = _normalize_symbols(primary_missing_symbols)
    candidate_set = set(candidates)
    work = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    raw_row_count = int(len(work))
    raw_rows_sha256 = dataframe_sha256(work)

    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    if not work.empty and "ts_code" in work.columns:
        for raw_row in work.to_dict(orient="records"):
            symbol = _normalize_symbol(raw_row.get("ts_code"))
            if symbol in candidate_set:
                rows_by_symbol.setdefault(symbol, []).append(raw_row)

    verified: list[str] = []
    matched_records: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    for symbol in candidates:
        rows = rows_by_symbol.get(symbol, [])
        if len(rows) != 1:
            rejected[symbol] = [
                "exact_row_missing" if not rows else "duplicate_exact_rows"
            ]
            continue
        row = rows[0]
        reasons = _validate_zero_row(row, trade_date=target_date)
        if reasons:
            rejected[symbol] = reasons
            continue
        verified.append(symbol)
        matched_records.append(
            {
                key: _json_scalar(row.get(key))
                for key in (
                    "ts_code",
                    "trade_date",
                    "name",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "change",
                    "pct_change",
                    "pct_chg",
                    "vol",
                    "amount",
                )
                if key in row
            }
        )

    verified = sorted(verified)
    matched_records.sort(key=lambda row: str(row.get("ts_code") or ""))
    payload: dict[str, Any] = {
        "schema_version": BAK_DAILY_NONTRADING_SCHEMA_VERSION,
        "classification": BAK_DAILY_NONTRADING_CLASSIFICATION,
        "trade_date": target_date,
        "source": "tushare.bak_daily",
        "query_params": dict(query_params),
        "query_succeeded": bool(query_succeeded),
        "query_error": str(query_error or ""),
        "raw_row_count": raw_row_count,
        "raw_rows_sha256": raw_rows_sha256,
        "primary_missing_symbols": candidates,
        "primary_missing_symbols_sha256": symbol_set_sha256(candidates),
        "verified_symbols": verified,
        "verified_symbol_count": len(verified),
        "verified_symbols_sha256": symbol_set_sha256(verified),
        "matched_records": matched_records,
        "matched_record_count": len(matched_records),
        "matched_records_sha256": canonical_json_sha256(matched_records),
        "rejected_symbols": dict(sorted(rejected.items())),
        "pit_membership_path": str(pit_membership_path),
        "pit_membership_sha256": str(pit_membership_sha256 or "").lower(),
        "writes_synthetic_bars": False,
        "regulatory_suspension_claimed": False,
        "generated_at": _utc_now_iso(),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_bak_daily_nontrading_evidence(
    payload: Mapping[str, Any],
    *,
    trade_date: str,
    primary_missing_symbols: Iterable[Any],
    pit_membership_sha256: str,
) -> list[str]:
    blockers: list[str] = []
    canonical_payload = dict(payload) if isinstance(payload, Mapping) else {}
    declared_payload_sha256 = str(
        canonical_payload.pop("payload_sha256", "") or ""
    ).strip().lower()
    if declared_payload_sha256 != canonical_json_sha256(canonical_payload):
        blockers.append("payload_sha256_mismatch")
    if payload.get("schema_version") != BAK_DAILY_NONTRADING_SCHEMA_VERSION:
        blockers.append("schema_version_mismatch")
    if payload.get("classification") != BAK_DAILY_NONTRADING_CLASSIFICATION:
        blockers.append("classification_mismatch")
    if _compact_trade_date(payload.get("trade_date")) != _compact_trade_date(
        trade_date
    ):
        blockers.append("trade_date_mismatch")
    if payload.get("query_succeeded") is not True:
        blockers.append("query_not_succeeded")
    if str(payload.get("source") or "") != "tushare.bak_daily":
        blockers.append("source_mismatch")
    if payload.get("query_params") != {
        "trade_date": _compact_trade_date(trade_date)
    }:
        blockers.append("query_params_mismatch")
    if str(payload.get("query_error") or ""):
        blockers.append("query_error_nonempty")
    if str(payload.get("pit_membership_sha256") or "").lower() != str(
        pit_membership_sha256 or ""
    ).lower():
        blockers.append("pit_membership_sha256_mismatch")
    expected_candidates = _normalize_symbols(primary_missing_symbols)
    declared_candidates = _normalize_symbols(
        payload.get("primary_missing_symbols", []) or []
    )
    if declared_candidates != expected_candidates:
        blockers.append("primary_missing_symbols_mismatch")
    if str(payload.get("primary_missing_symbols_sha256") or "").lower() != (
        symbol_set_sha256(expected_candidates)
    ):
        blockers.append("primary_missing_symbols_sha256_mismatch")
    verified = _normalize_symbols(payload.get("verified_symbols", []) or [])
    if not set(verified).issubset(set(expected_candidates)):
        blockers.append("verified_symbols_outside_primary_missing")
    if str(payload.get("verified_symbols_sha256") or "").lower() != (
        symbol_set_sha256(verified)
    ):
        blockers.append("verified_symbols_sha256_mismatch")
    if int(payload.get("verified_symbol_count") or 0) != len(verified):
        blockers.append("verified_symbol_count_mismatch")
    matched_records = payload.get("matched_records", []) or []
    if not isinstance(matched_records, list):
        blockers.append("matched_records_invalid")
        matched_records = []
    if int(payload.get("matched_record_count") or 0) != len(matched_records):
        blockers.append("matched_record_count_mismatch")
    if str(payload.get("matched_records_sha256") or "").lower() != (
        canonical_json_sha256(matched_records)
    ):
        blockers.append("matched_records_sha256_mismatch")
    matched_symbols: list[str] = []
    for record in matched_records:
        if not isinstance(record, Mapping):
            blockers.append("matched_record_invalid")
            continue
        symbol = _normalize_symbol(record.get("ts_code"))
        matched_symbols.append(symbol)
        blockers.extend(
            f"matched_record_contract:{symbol or 'unknown'}:{reason}"
            for reason in _validate_zero_row(
                record,
                trade_date=_compact_trade_date(trade_date),
            )
        )
    if sorted(matched_symbols) != verified:
        blockers.append("matched_record_symbols_mismatch")
    rejected = payload.get("rejected_symbols", {}) or {}
    if not isinstance(rejected, Mapping):
        blockers.append("rejected_symbols_invalid")
        rejected_symbols: set[str] = set()
    else:
        rejected_symbols = {
            _normalize_symbol(symbol)
            for symbol in rejected
            if _normalize_symbol(symbol)
        }
    if set(verified) & rejected_symbols:
        blockers.append("verified_rejected_overlap")
    if set(expected_candidates) != set(verified) | rejected_symbols:
        blockers.append("candidate_classification_union_mismatch")
    try:
        raw_row_count = int(payload.get("raw_row_count"))
    except (TypeError, ValueError, OverflowError):
        raw_row_count = -1
    if raw_row_count < len(matched_records):
        blockers.append("raw_row_count_invalid")
    raw_rows_sha256 = str(payload.get("raw_rows_sha256") or "").lower()
    if len(raw_rows_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in raw_rows_sha256
    ):
        blockers.append("raw_rows_sha256_invalid")
    pit_sha256 = str(payload.get("pit_membership_sha256") or "").lower()
    if len(pit_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in pit_sha256
    ):
        blockers.append("pit_membership_sha256_invalid")
    if not str(payload.get("pit_membership_path") or "").strip():
        blockers.append("pit_membership_path_missing")
    if payload.get("writes_synthetic_bars") is not False:
        blockers.append("synthetic_bar_contract_invalid")
    if payload.get("regulatory_suspension_claimed") is not False:
        blockers.append("regulatory_suspension_contract_invalid")
    return blockers


def evidence_cache_path(
    root: str | Path,
    *,
    trade_date: str,
    primary_missing_symbols: Iterable[Any],
) -> Path:
    target_date = _compact_trade_date(trade_date)
    digest = symbol_set_sha256(primary_missing_symbols)[:16]
    return (
        Path(root)
        / ".cache"
        / "nontrading_bak_daily"
        / target_date
        / f"primary_missing_{digest}.json"
    )


def read_evidence_cache(
    path: str | Path,
    *,
    trade_date: str,
    primary_missing_symbols: Iterable[Any],
    pit_membership_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    resolved = Path(path)
    if not resolved.exists():
        return {}, ["cache_missing"]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"cache_unreadable:{exc}"]
    if not isinstance(payload, dict):
        return {}, ["cache_invalid"]
    blockers = validate_bak_daily_nontrading_evidence(
        payload,
        trade_date=trade_date,
        primary_missing_symbols=primary_missing_symbols,
        pit_membership_sha256=pit_membership_sha256,
    )
    return dict(payload), blockers


def write_evidence_cache(path: str | Path, payload: Mapping[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, resolved)
    return resolved


__all__ = [
    "BAK_DAILY_NONTRADING_CLASSIFICATION",
    "BAK_DAILY_NONTRADING_SCHEMA_VERSION",
    "build_bak_daily_nontrading_evidence",
    "canonical_json_sha256",
    "dataframe_sha256",
    "evidence_cache_path",
    "file_sha256",
    "read_evidence_cache",
    "symbol_set_sha256",
    "validate_bak_daily_nontrading_evidence",
    "write_evidence_cache",
]
