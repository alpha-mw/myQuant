"""Explicit, hash-bound secondary exact-date CN daily-bar evidence.

This module is intentionally separate from the default Tushare maintenance
path.  It can probe Eastmoney for a missing exact-date bar only when the
caller explicitly enables that source.  Empty or malformed responses are
evidence of an unsuccessful probe, never evidence that a symbol did not
trade.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SECONDARY_DAILY_EVIDENCE_SCHEMA_VERSION = "cn-secondary-daily-evidence.v1"
SECONDARY_DAILY_CLASSIFICATION = "secondary_daily_exact_date_bar_probe"
EASTMONEY_SOURCE_SYSTEM = "eastmoney.push2his.kline"
EASTMONEY_ENDPOINT = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def _compact_trade_date(value: Any) -> str:
    text = str(value or "").strip()
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _eastmoney_secid(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    code, _, suffix = normalized.partition(".")
    if not code:
        return ""
    if suffix == "SH" or (not suffix and code.startswith(("5", "6", "9"))):
        return f"1.{code}"
    if suffix in {"SZ", "BJ"} or (
        not suffix and code.startswith(("0", "1", "2", "3", "4", "8"))
    ):
        return f"0.{code}"
    return ""


def _eastmoney_url(symbol: str, trade_date: str) -> str:
    secid = _eastmoney_secid(symbol)
    if not secid:
        raise ValueError(f"secondary_symbol_exchange_unsupported:{symbol}")
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "0",
        "beg": trade_date,
        "end": trade_date,
    }
    return f"{EASTMONEY_ENDPOINT}?{urlencode(params)}"


def _fetch_raw_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=12) as response:
        return response.read()


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_exact_row(
    item: Any,
    *,
    symbol: str,
    trade_date: str,
) -> tuple[dict[str, Any] | None, str]:
    parts = str(item or "").split(",")
    if len(parts) < 7:
        return None, "kline_row_too_short"
    row_date = _compact_trade_date(parts[0])
    if row_date != trade_date:
        return None, "kline_row_date_mismatch"
    open_price = _finite_float(parts[1])
    close = _finite_float(parts[2])
    high = _finite_float(parts[3])
    low = _finite_float(parts[4])
    volume = _finite_float(parts[5])
    amount_yuan = _finite_float(parts[6])
    change = _finite_float(parts[9]) if len(parts) > 9 else None
    pct_chg = _finite_float(parts[8]) if len(parts) > 8 else None
    if None in {open_price, close, high, low, volume, amount_yuan}:
        return None, "kline_numeric_field_invalid"
    assert open_price is not None
    assert close is not None
    assert high is not None
    assert low is not None
    assert volume is not None
    assert amount_yuan is not None
    if min(open_price, close, high, low) <= 0:
        return None, "kline_price_nonpositive"
    if volume < 0 or amount_yuan < 0:
        return None, "kline_volume_or_amount_negative"
    tolerance = 1e-8
    if high + tolerance < max(open_price, close, low):
        return None, "kline_high_inconsistent"
    if low - tolerance > min(open_price, close, high):
        return None, "kline_low_inconsistent"
    pre_close = close - change if change is not None else None
    if pre_close is not None and pre_close <= 0:
        return None, "kline_pre_close_nonpositive"
    return {
        "ts_code": symbol,
        "trade_date": trade_date,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "pre_close": pre_close,
        "change": change,
        "pct_chg": pct_chg,
        "vol": volume,
        # Eastmoney returns yuan; Tushare daily.amount is thousand yuan.
        "amount": amount_yuan / 1000.0,
    }, ""


def _extract_exact_row(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    trade_date: str,
) -> tuple[dict[str, Any] | None, int, str]:
    data = payload.get("data")
    klines = data.get("klines") if isinstance(data, Mapping) else []
    if klines is None:
        klines = []
    if not isinstance(klines, list):
        return None, 0, "kline_payload_invalid"
    exact = [item for item in klines if _compact_trade_date(str(item).split(",", 1)[0]) == trade_date]
    if len(exact) == 0:
        return None, len(klines), "exact_row_missing"
    if len(exact) != 1:
        return None, len(klines), "duplicate_exact_rows"
    row, reason = _parse_exact_row(exact[0], symbol=symbol, trade_date=trade_date)
    return row, len(klines), reason


def _raw_capture_path(root: Path, raw_sha256: str) -> Path:
    return root / "raw" / f"{raw_sha256}.json"


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def probe_eastmoney_daily_evidence(
    symbols: Iterable[Any],
    trade_date: str,
    *,
    output_root: str | Path,
    pit_membership_path: str | Path,
    pit_membership_sha256: str,
    query_run_id: str,
    fetch_raw: Callable[[str], bytes] = _fetch_raw_bytes,
) -> tuple[dict[str, Any], Path]:
    """Probe exact-date Eastmoney rows and persist a fully bound receipt."""

    target_date = _compact_trade_date(trade_date)
    if not target_date:
        raise ValueError("secondary_trade_date_invalid")
    normalized_symbols = _normalize_symbols(symbols)
    if not normalized_symbols:
        raise ValueError("secondary_symbol_scope_empty")
    pit_path = Path(pit_membership_path).expanduser()
    if not pit_path.exists():
        raise ValueError("secondary_pit_membership_missing")
    if _sha256_file(pit_path) != str(pit_membership_sha256).lower():
        raise ValueError("secondary_pit_membership_sha256_mismatch")
    run_id = str(query_run_id or "").strip()
    if not run_id:
        raise ValueError("secondary_query_run_id_required")

    evidence_root = (
        Path(output_root)
        / ".cache"
        / "secondary_daily"
        / target_date
        / f"pit_{str(pit_membership_sha256).lower()}"
    )
    entries: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for symbol in normalized_symbols:
        url = _eastmoney_url(symbol, target_date)
        entry: dict[str, Any] = {
            "symbol": symbol,
            "trade_date": target_date,
            "url": url,
            "query_params": {
                "symbol": symbol,
                "trade_date": target_date,
                "klt": "101",
                "fqt": "0",
            },
            "query_succeeded": False,
            "raw_capture_path": "",
            "raw_response_sha256": "",
            "raw_bytes": 0,
            "raw_kline_row_count": 0,
            "exact_row_count": 0,
            "row": None,
            "status": "error",
            "reason": "",
        }
        try:
            raw = bytes(fetch_raw(url))
            raw_sha256 = _sha256_bytes(raw)
            raw_path = _raw_capture_path(evidence_root, raw_sha256)
            if raw_path.exists() and _sha256_file(raw_path) != raw_sha256:
                raise ValueError("secondary_raw_capture_sha256_collision")
            _write_bytes_atomic(raw_path, raw)
            entry.update(
                {
                    "query_succeeded": True,
                    "raw_capture_path": str(raw_path),
                    "raw_response_sha256": raw_sha256,
                    "raw_bytes": len(raw),
                }
            )
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("secondary_json_payload_invalid")
            row, raw_row_count, reason = _extract_exact_row(
                payload,
                symbol=symbol,
                trade_date=target_date,
            )
            entry.update(
                {
                    "raw_kline_row_count": raw_row_count,
                    "exact_row_count": 1 if row is not None else 0,
                    "row": row,
                    "status": "observed" if row is not None else "empty",
                    "reason": reason,
                }
            )
            if row is not None:
                normalized_rows.append(row)
        except Exception as exc:
            entry["reason"] = f"{type(exc).__name__}:{exc}"
        entries.append(entry)

    normalized_rows.sort(key=lambda row: (row["trade_date"], row["ts_code"]))
    payload: dict[str, Any] = {
        "schema_version": SECONDARY_DAILY_EVIDENCE_SCHEMA_VERSION,
        "classification": SECONDARY_DAILY_CLASSIFICATION,
        "source": EASTMONEY_SOURCE_SYSTEM,
        "source_endpoint": EASTMONEY_ENDPOINT,
        "target_trade_date": target_date,
        "query_run_id": run_id,
        "pit_membership_path": str(pit_path),
        "pit_membership_sha256": str(pit_membership_sha256).lower(),
        "queried_symbols": normalized_symbols,
        "queried_symbols_sha256": _canonical_sha256(normalized_symbols),
        "observed_symbols": sorted(row["ts_code"] for row in normalized_rows),
        "observed_symbols_sha256": _canonical_sha256(
            sorted(row["ts_code"] for row in normalized_rows)
        ),
        "entries": entries,
        "normalized_rows": normalized_rows,
        "normalized_rows_sha256": _canonical_sha256(normalized_rows),
        "generated_at": _utc_now_iso(),
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    manifest_path = evidence_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(
        manifest_path,
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    readback = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = str(readback.pop("payload_sha256") or "")
    if declared != _canonical_sha256(readback):
        raise ValueError("secondary_evidence_payload_sha256_readback_mismatch")
    if _sha256_file(manifest_path) == "":
        raise ValueError("secondary_evidence_manifest_readback_failed")
    return payload, manifest_path


def validate_secondary_daily_evidence(
    path: str | Path,
    *,
    target_trade_date: str,
    expected_pit_membership_sha256: str,
) -> dict[str, Any]:
    """Read back a secondary receipt and return semantic blockers."""

    manifest_path = Path(path).expanduser()
    blockers: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "blocked", "blockers": [f"secondary_manifest_read:{exc}"]}
    if not isinstance(payload, Mapping):
        return {"status": "blocked", "blockers": ["secondary_manifest_not_object"]}
    work = dict(payload)
    declared = str(work.pop("payload_sha256") or "")
    if declared != _canonical_sha256(work):
        blockers.append("secondary_payload_sha256_mismatch")
    if payload.get("schema_version") != SECONDARY_DAILY_EVIDENCE_SCHEMA_VERSION:
        blockers.append("secondary_schema_version_mismatch")
    if payload.get("classification") != SECONDARY_DAILY_CLASSIFICATION:
        blockers.append("secondary_classification_mismatch")
    if payload.get("source") != EASTMONEY_SOURCE_SYSTEM:
        blockers.append("secondary_source_mismatch")
    if payload.get("source_endpoint") != EASTMONEY_ENDPOINT:
        blockers.append("secondary_source_endpoint_mismatch")
    if _compact_trade_date(payload.get("target_trade_date")) != _compact_trade_date(target_trade_date):
        blockers.append("secondary_trade_date_mismatch")
    if str(payload.get("pit_membership_sha256") or "").lower() != str(expected_pit_membership_sha256).lower():
        blockers.append("secondary_pit_membership_sha256_mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        blockers.append("secondary_entries_invalid")
        entries = []
    queried_symbols = sorted(
        _normalize_symbol(symbol)
        for symbol in payload.get("queried_symbols", []) or []
    )
    entry_symbols = sorted(
        _normalize_symbol(entry.get("symbol"))
        for entry in entries
        if isinstance(entry, Mapping)
    )
    if entry_symbols != queried_symbols:
        blockers.append("secondary_entries_scope_mismatch")
    for entry in entries:
        if not isinstance(entry, Mapping):
            blockers.append("secondary_entry_invalid")
            continue
        raw_path = Path(str(entry.get("raw_capture_path") or ""))
        raw_sha = str(entry.get("raw_response_sha256") or "").lower()
        if bool(entry.get("query_succeeded")):
            if not raw_path.exists():
                blockers.append("secondary_raw_capture_missing")
            elif _sha256_file(raw_path) != raw_sha:
                blockers.append("secondary_raw_capture_sha256_mismatch")
            else:
                try:
                    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    row, raw_row_count, reason = _extract_exact_row(
                        raw_payload,
                        symbol=_normalize_symbol(entry.get("symbol")),
                        trade_date=_compact_trade_date(
                            entry.get("trade_date")
                        ),
                    )
                    if raw_row_count != int(entry.get("raw_kline_row_count") or 0):
                        blockers.append("secondary_raw_kline_row_count_mismatch")
                    if reason != str(entry.get("reason") or ""):
                        blockers.append("secondary_raw_row_reason_mismatch")
                    if row != entry.get("row"):
                        blockers.append("secondary_normalized_row_mismatch")
                except Exception as exc:
                    blockers.append(
                        f"secondary_raw_payload_recompute_failed:{type(exc).__name__}"
                    )
    observed = sorted(
        _normalize_symbol(symbol) for symbol in payload.get("observed_symbols", []) or []
    )
    rows = payload.get("normalized_rows", []) or []
    row_symbols = sorted(
        _normalize_symbol(row.get("ts_code"))
        for row in rows
        if isinstance(row, Mapping)
    )
    if observed != row_symbols:
        blockers.append("secondary_observed_symbols_rows_mismatch")
    if str(payload.get("queried_symbols_sha256") or "") != _canonical_sha256(
        queried_symbols
    ):
        blockers.append("secondary_queried_symbols_sha256_mismatch")
    if str(payload.get("observed_symbols_sha256") or "") != _canonical_sha256(
        observed
    ):
        blockers.append("secondary_observed_symbols_sha256_mismatch")
    if str(payload.get("normalized_rows_sha256") or "") != _canonical_sha256(rows):
        blockers.append("secondary_normalized_rows_sha256_mismatch")
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "payload": dict(payload),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
    }


__all__ = [
    "EASTMONEY_ENDPOINT",
    "EASTMONEY_SOURCE_SYSTEM",
    "SECONDARY_DAILY_CLASSIFICATION",
    "SECONDARY_DAILY_EVIDENCE_SCHEMA_VERSION",
    "probe_eastmoney_daily_evidence",
    "validate_secondary_daily_evidence",
]
