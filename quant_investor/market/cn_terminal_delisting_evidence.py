"""Hash-bound evidence for a Shenzhen terminal-delisting boundary.

Tushare ``stock_basic`` can briefly retain ``list_status=L`` after the last
delisting-consolidation session.  This module does not rewrite PIT membership
or synthesize a bar.  It proves a narrow, date-bounded classification from the
terminal name-change row, the complete 15-session Shenzhen window, the next
open date, and exact-date non-trading readbacks.
"""

from __future__ import annotations

import json
import math
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_investor.market.cn_nontrading_evidence import (
    canonical_json_sha256,
    dataframe_sha256,
    file_sha256,
    symbol_set_sha256,
)
from quant_investor.market.pit_universe import (
    REASON_LISTED,
    evaluate_listing_status,
)

TERMINAL_DELISTING_SCHEMA_VERSION = "cn-terminal-delisting-evidence.v1"
TERMINAL_DELISTING_POLICY_VERSION = "szse-terminal-15-open-sessions.v1"
TERMINAL_DELISTING_CLASSIFICATION = "verified_terminal_delisting_absent"
TERMINAL_CHANGE_REASONS = frozenset({"终止上市", "退市整理期"})
STOCK_BASIC_FIELDS = "ts_code,name,list_status,list_date,delist_date"
NAMECHANGE_FIELDS = (
    "ts_code,name,start_date,end_date,ann_date,change_reason"
)
DAILY_FIELDS = "ts_code,trade_date"
TRADE_CAL_FIELDS = "cal_date,is_open"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compact_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    digits = "".join(character for character in text if character.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_symbols(values: Iterable[Any]) -> list[str]:
    return sorted({_normalize_symbol(value) for value in values if _normalize_symbol(value)})


def _normalize_identity(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return "".join(character for character in normalized if not character.isspace())


def _normalize_reason(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _validated_sha256(value: Any, *, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field_name} must be a complete 64-character SHA-256")
    return digest


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


def _frame_payload(frame: pd.DataFrame, query_params: Mapping[str, Any]) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    records = [
        {column: _json_scalar(row.get(column)) for column in columns}
        for row in frame.to_dict(orient="records")
    ]
    return {
        "query_params": dict(query_params),
        "query_succeeded": True,
        "columns": columns,
        "raw_row_count": int(len(frame)),
        "raw_rows_sha256": dataframe_sha256(frame),
        "raw_records": records,
    }


def _payload_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    columns = payload.get("columns", []) or []
    records = payload.get("raw_records", []) or []
    if not isinstance(columns, list) or not isinstance(records, list):
        return pd.DataFrame()
    return pd.DataFrame(records, columns=[str(column) for column in columns])


def _query_frame(
    provider: Any,
    endpoint: str,
    query_params: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    func = getattr(provider, endpoint, None)
    if func is None:
        return pd.DataFrame(), f"{endpoint}_provider_unavailable"
    try:
        frame = func(**dict(query_params))
    except Exception as exc:
        return pd.DataFrame(), f"{endpoint}_query_failed:{type(exc).__name__}:{exc}"
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame(), f"{endpoint}_response_not_dataframe"
    return frame.copy(), ""


def terminal_delisting_cache_path(
    root: str | Path,
    *,
    target_trade_date: str,
    candidate_symbols: Iterable[Any],
    pit_membership_sha256: str | None = None,
) -> Path:
    target = _compact_date(target_trade_date)
    digest = symbol_set_sha256(candidate_symbols)[:16]
    cache_root = Path(root) / ".cache" / "terminal_delisting" / target
    if pit_membership_sha256 is not None:
        pit_digest = _validated_sha256(
            pit_membership_sha256,
            field_name="pit_membership_sha256",
        )
        cache_root = cache_root / f"pit_{pit_digest}"
    return cache_root / f"candidates_{digest}.json"


def select_terminal_delisting_candidates(
    symbols: Iterable[Any],
    *,
    target_trade_date: str,
    pit_records_by_symbol: Mapping[str, Any],
) -> list[str]:
    """Select only PIT-listed Shenzhen rows with a terminal provider name."""

    selected: list[str] = []
    for symbol in _normalize_symbols(symbols):
        record = pit_records_by_symbol.get(symbol)
        if record is None or not symbol.endswith(".SZ"):
            continue
        status = evaluate_listing_status(
            record,
            symbol=symbol,
            as_of=target_trade_date,
        )
        if status.reason != REASON_LISTED or not status.tradable:
            continue
        if str(getattr(record, "source_list_status", "") or "").strip().upper() != "L":
            continue
        if _compact_date(getattr(record, "delist_date", "")):
            continue
        if not _normalize_identity(getattr(record, "name", "")).endswith("退"):
            continue
        selected.append(symbol)
    return selected


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _required_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    endpoint: str,
) -> list[str]:
    missing = sorted(required - set(str(column) for column in frame.columns))
    return [f"{endpoint}_required_columns_missing:{','.join(missing)}"] if missing else []


def _build_symbol_proof(
    provider: Any,
    *,
    symbol: str,
    target_trade_date: str,
    pit_record: Any,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    proof: dict[str, Any] = {"symbol": symbol, "verified": False, "queries": {}}
    if not symbol.endswith(".SZ"):
        return proof, ["symbol_not_szse"]
    pit_status = evaluate_listing_status(
        pit_record,
        symbol=symbol,
        as_of=target_trade_date,
    )
    if pit_status.reason != REASON_LISTED or not pit_status.tradable:
        return proof, [f"pit_not_listed_tradable:{pit_status.reason}"]

    stock_params = {
        "ts_code": symbol,
        "list_status": "L",
        "fields": STOCK_BASIC_FIELDS,
    }
    stock_frame, stock_error = _query_frame(provider, "stock_basic", stock_params)
    if stock_error:
        return proof, [stock_error]
    proof["queries"]["stock_basic"] = _frame_payload(stock_frame, stock_params)
    reasons.extend(
        _required_columns(
            stock_frame,
            {"ts_code", "name", "list_status", "list_date", "delist_date"},
            endpoint="stock_basic",
        )
    )
    if reasons:
        return proof, reasons
    stock_rows = stock_frame.loc[stock_frame["ts_code"].map(_normalize_symbol).eq(symbol)].to_dict(
        orient="records"
    )
    if len(stock_rows) != 1:
        return proof, ["stock_basic_exact_row_count_mismatch"]
    stock_row = stock_rows[0]
    stock_name = _normalize_identity(stock_row.get("name"))
    if _normalize_reason(stock_row.get("list_status")).upper() != "L":
        reasons.append("stock_basic_status_not_listed")
    if _compact_date(stock_row.get("delist_date")):
        reasons.append("stock_basic_delist_date_already_present")
    if not stock_name.endswith("退"):
        reasons.append("stock_basic_name_not_terminal")
    if reasons:
        return proof, reasons

    name_params = {"ts_code": symbol, "fields": NAMECHANGE_FIELDS}
    name_frame, name_error = _query_frame(provider, "namechange", name_params)
    if name_error:
        return proof, [name_error]
    proof["queries"]["namechange"] = _frame_payload(name_frame, name_params)
    reasons.extend(
        _required_columns(
            name_frame,
            {"ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"},
            endpoint="namechange",
        )
    )
    if reasons:
        return proof, reasons
    active_rows: list[dict[str, Any]] = []
    for row in name_frame.to_dict(orient="records"):
        start_date = _compact_date(row.get("start_date"))
        end_date = _compact_date(row.get("end_date"))
        if (
            _normalize_symbol(row.get("ts_code")) == symbol
            and start_date
            and start_date <= target_trade_date
            and (not end_date or target_trade_date <= end_date)
            and _normalize_identity(row.get("name")) == stock_name
            and _normalize_reason(row.get("change_reason")) in TERMINAL_CHANGE_REASONS
        ):
            active_rows.append(row)
    if len(active_rows) != 1:
        return proof, ["terminal_namechange_active_row_count_mismatch"]
    active_row = active_rows[0]
    start_date = _compact_date(active_row.get("start_date"))
    announcement_date = _compact_date(active_row.get("ann_date"))
    if not announcement_date or announcement_date > start_date:
        return proof, ["terminal_namechange_announcement_date_invalid"]

    daily_params = {
        "ts_code": symbol,
        "start_date": start_date,
        "end_date": target_trade_date,
        "fields": DAILY_FIELDS,
    }
    daily_frame, daily_error = _query_frame(provider, "daily", daily_params)
    if daily_error:
        return proof, [daily_error]
    proof["queries"]["daily"] = _frame_payload(daily_frame, daily_params)
    reasons.extend(
        _required_columns(
            daily_frame,
            {"ts_code", "trade_date"},
            endpoint="daily",
        )
    )
    if reasons:
        return proof, reasons
    daily_rows = daily_frame.loc[daily_frame["ts_code"].map(_normalize_symbol).eq(symbol)].copy()
    daily_dates = [_compact_date(value) for value in daily_rows["trade_date"].tolist()]
    if any(not value for value in daily_dates):
        reasons.append("daily_trade_date_invalid")
    if len(daily_dates) != len(set(daily_dates)):
        reasons.append("daily_trade_dates_duplicate")
    daily_dates = sorted(set(daily_dates))
    if len(daily_dates) != 15:
        reasons.append(f"terminal_daily_session_count_mismatch:{len(daily_dates)}")

    calendar_params = {
        "exchange": "SSE",
        "start_date": start_date,
        "end_date": target_trade_date,
        "is_open": "1",
        "fields": TRADE_CAL_FIELDS,
    }
    calendar_frame, calendar_error = _query_frame(
        provider,
        "trade_cal",
        calendar_params,
    )
    if calendar_error:
        return proof, [calendar_error]
    proof["queries"]["trade_cal"] = _frame_payload(
        calendar_frame,
        calendar_params,
    )
    reasons.extend(
        _required_columns(
            calendar_frame,
            {"cal_date", "is_open"},
            endpoint="trade_cal",
        )
    )
    if reasons:
        return proof, reasons
    open_rows = calendar_frame.loc[pd.to_numeric(calendar_frame["is_open"], errors="coerce").eq(1)]
    raw_open_dates = [_compact_date(value) for value in open_rows["cal_date"].tolist()]
    if any(not value for value in raw_open_dates):
        reasons.append("trade_cal_open_date_invalid")
    if len(raw_open_dates) != len(set(raw_open_dates)):
        reasons.append("trade_cal_open_dates_duplicate")
    open_dates = sorted(set(raw_open_dates))
    if len(open_dates) < 16:
        reasons.append(f"trade_cal_terminal_window_too_short:{len(open_dates)}")
    inferred_delist_date = open_dates[15] if len(open_dates) >= 16 else ""
    if open_dates and open_dates[0] != start_date:
        reasons.append("terminal_start_date_not_open_window_start")
    if len(open_dates) >= 15 and daily_dates != open_dates[:15]:
        reasons.append("terminal_daily_dates_not_first_15_open_sessions")
    if not inferred_delist_date or inferred_delist_date > target_trade_date:
        reasons.append("inferred_delist_date_after_target_or_missing")
    if daily_dates and any(value >= inferred_delist_date for value in daily_dates):
        reasons.append("daily_row_present_on_or_after_inferred_delist_date")
    if reasons:
        return proof, reasons

    suspend_params = {
        "ts_code": symbol,
        "trade_date": inferred_delist_date,
    }
    suspend_frame, suspend_error = _query_frame(
        provider,
        "suspend_d",
        suspend_params,
    )
    if suspend_error:
        return proof, [suspend_error]
    proof["queries"]["suspend_d"] = _frame_payload(
        suspend_frame,
        suspend_params,
    )
    if not suspend_frame.empty:
        reasons.append("inferred_delist_date_suspend_event_present")

    bak_params = {"ts_code": symbol, "trade_date": inferred_delist_date}
    bak_frame, bak_error = _query_frame(provider, "bak_daily", bak_params)
    if bak_error:
        return proof, [bak_error]
    proof["queries"]["bak_daily"] = _frame_payload(bak_frame, bak_params)
    if not bak_frame.empty:
        reasons.append("inferred_delist_date_bak_daily_row_present")
    if reasons:
        return proof, reasons

    proof.update(
        {
            "verified": True,
            "stock_basic_name": stock_name,
            "source_list_status": "L",
            "provider_delist_date": "",
            "terminal_change_reason": _normalize_reason(active_row.get("change_reason")),
            "terminal_start_date": start_date,
            "terminal_announcement_date": announcement_date,
            "terminal_daily_dates": daily_dates,
            "terminal_open_dates_through_target": open_dates,
            "last_terminal_trade_date": daily_dates[-1],
            "inferred_delist_date": inferred_delist_date,
        }
    )
    return proof, []


def build_terminal_delisting_evidence(
    provider: Any,
    *,
    target_trade_date: str,
    candidate_symbols: Iterable[Any],
    pit_records_by_symbol: Mapping[str, Any],
    pit_membership_path: str | Path,
    pit_membership_sha256: str,
) -> dict[str, Any]:
    target = _compact_date(target_trade_date)
    if not target:
        raise ValueError("target_trade_date must be YYYYMMDD-compatible")
    candidates = _normalize_symbols(candidate_symbols)
    proofs: dict[str, dict[str, Any]] = {}
    rejected: dict[str, list[str]] = {}
    verified: list[str] = []
    inferred_dates: dict[str, str] = {}
    for symbol in candidates:
        proof, reasons = _build_symbol_proof(
            provider,
            symbol=symbol,
            target_trade_date=target,
            pit_record=pit_records_by_symbol.get(symbol),
        )
        proofs[symbol] = proof
        if reasons:
            rejected[symbol] = reasons
            continue
        verified.append(symbol)
        inferred_dates[symbol] = str(proof["inferred_delist_date"])
    verified = sorted(verified)
    payload: dict[str, Any] = {
        "schema_version": TERMINAL_DELISTING_SCHEMA_VERSION,
        "resolver_policy_version": TERMINAL_DELISTING_POLICY_VERSION,
        "classification": TERMINAL_DELISTING_CLASSIFICATION,
        "target_trade_date": target,
        "source": ("tushare.stock_basic+namechange+daily+trade_cal+suspend_d+bak_daily"),
        "candidate_symbols": candidates,
        "candidate_symbol_count": len(candidates),
        "candidate_symbols_sha256": symbol_set_sha256(candidates),
        "verified_symbols": verified,
        "verified_symbol_count": len(verified),
        "verified_symbols_sha256": symbol_set_sha256(verified),
        "inferred_delist_dates": dict(sorted(inferred_dates.items())),
        "rejected_symbols": dict(sorted(rejected.items())),
        "symbol_proofs": dict(sorted(proofs.items())),
        "all_candidates_verified": bool(candidates) and verified == candidates,
        "pit_membership_path": str(pit_membership_path),
        "pit_membership_sha256": str(pit_membership_sha256 or "").lower(),
        "writes_synthetic_bars": False,
        "regulatory_exact_date_suspend_event_claimed": False,
        "generated_at": _utc_now_iso(),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_query_payload(
    payload: Mapping[str, Any],
    *,
    expected_params: Mapping[str, Any],
    endpoint: str,
) -> list[str]:
    blockers: list[str] = []
    if payload.get("query_succeeded") is not True:
        blockers.append(f"{endpoint}_query_not_succeeded")
    if payload.get("query_params") != dict(expected_params):
        blockers.append(f"{endpoint}_query_params_mismatch")
    frame = _payload_frame(payload)
    try:
        raw_row_count = int(payload.get("raw_row_count"))
    except (TypeError, ValueError, OverflowError):
        raw_row_count = -1
    if raw_row_count != len(frame):
        blockers.append(f"{endpoint}_raw_row_count_mismatch")
    if str(payload.get("raw_rows_sha256") or "").lower() != dataframe_sha256(frame):
        blockers.append(f"{endpoint}_raw_rows_sha256_mismatch")
    return blockers


def validate_terminal_delisting_evidence(
    payload: Mapping[str, Any],
    *,
    target_trade_date: str,
    candidate_symbols: Iterable[Any],
    pit_membership_path: str | Path,
    pit_membership_sha256: str,
) -> list[str]:
    blockers: list[str] = []
    work = dict(payload) if isinstance(payload, Mapping) else {}
    declared_payload_sha256 = str(work.pop("payload_sha256", "") or "").lower()
    if declared_payload_sha256 != canonical_json_sha256(work):
        blockers.append("payload_sha256_mismatch")
    if payload.get("schema_version") != TERMINAL_DELISTING_SCHEMA_VERSION:
        blockers.append("schema_version_mismatch")
    if payload.get("resolver_policy_version") != TERMINAL_DELISTING_POLICY_VERSION:
        blockers.append("resolver_policy_version_mismatch")
    if payload.get("classification") != TERMINAL_DELISTING_CLASSIFICATION:
        blockers.append("classification_mismatch")
    target = _compact_date(target_trade_date)
    if _compact_date(payload.get("target_trade_date")) != target:
        blockers.append("target_trade_date_mismatch")
    candidates = _normalize_symbols(candidate_symbols)
    if _normalize_symbols(payload.get("candidate_symbols", []) or []) != candidates:
        blockers.append("candidate_symbols_mismatch")
    if payload.get("candidate_symbol_count") != len(candidates):
        blockers.append("candidate_symbol_count_mismatch")
    if payload.get("candidate_symbols_sha256") != symbol_set_sha256(candidates):
        blockers.append("candidate_symbols_sha256_mismatch")
    verified = _normalize_symbols(payload.get("verified_symbols", []) or [])
    if verified != candidates or payload.get("all_candidates_verified") is not True:
        blockers.append("positive_cache_not_all_candidates_verified")
    if payload.get("verified_symbol_count") != len(verified):
        blockers.append("verified_symbol_count_mismatch")
    if payload.get("verified_symbols_sha256") != symbol_set_sha256(verified):
        blockers.append("verified_symbols_sha256_mismatch")
    if payload.get("rejected_symbols") not in ({}, None):
        blockers.append("positive_cache_rejected_symbols_nonempty")
    if str(payload.get("pit_membership_path") or "") != str(pit_membership_path):
        blockers.append("pit_membership_path_mismatch")
    if (
        str(payload.get("pit_membership_sha256") or "").lower()
        != str(pit_membership_sha256 or "").lower()
    ):
        blockers.append("pit_membership_sha256_mismatch")
    proofs = payload.get("symbol_proofs", {}) or {}
    inferred_dates = payload.get("inferred_delist_dates", {}) or {}
    if not isinstance(proofs, Mapping) or not isinstance(inferred_dates, Mapping):
        blockers.append("symbol_proofs_or_inferred_dates_invalid")
        proofs = {}
        inferred_dates = {}
    for symbol in candidates:
        proof = proofs.get(symbol, {}) or {}
        if not isinstance(proof, Mapping) or proof.get("verified") is not True:
            blockers.append(f"symbol_proof_invalid:{symbol}")
            continue
        if proof.get("source_list_status") != "L":
            blockers.append(f"symbol_source_status_mismatch:{symbol}")
        if _compact_date(proof.get("provider_delist_date")):
            blockers.append(f"symbol_provider_delist_date_nonempty:{symbol}")
        if not _normalize_identity(proof.get("stock_basic_name")).endswith("退"):
            blockers.append(f"symbol_terminal_name_mismatch:{symbol}")
        if _normalize_reason(proof.get("terminal_change_reason")) not in TERMINAL_CHANGE_REASONS:
            blockers.append(f"symbol_terminal_reason_mismatch:{symbol}")
        start_date = _compact_date(proof.get("terminal_start_date"))
        announcement_date = _compact_date(proof.get("terminal_announcement_date"))
        daily_dates = [
            _compact_date(value) for value in proof.get("terminal_daily_dates", []) or []
        ]
        open_dates = [
            _compact_date(value)
            for value in proof.get("terminal_open_dates_through_target", []) or []
        ]
        inferred = _compact_date(proof.get("inferred_delist_date"))
        if (
            not start_date
            or not announcement_date
            or announcement_date > start_date
            or len(daily_dates) != 15
            or len(set(daily_dates)) != 15
            or daily_dates != sorted(daily_dates)
            or len(open_dates) < 16
            or len(set(open_dates)) != len(open_dates)
            or open_dates != sorted(open_dates)
            or open_dates[0] != start_date
            or daily_dates != open_dates[:15]
            or inferred != open_dates[15]
            or inferred > target
            or any(value >= inferred for value in daily_dates)
            or _compact_date(inferred_dates.get(symbol)) != inferred
        ):
            blockers.append(f"symbol_terminal_window_mismatch:{symbol}")
        queries = proof.get("queries", {}) or {}
        if not isinstance(queries, Mapping):
            blockers.append(f"symbol_queries_invalid:{symbol}")
            queries = {}
        expected_queries = {
            "stock_basic": {
                "ts_code": symbol,
                "list_status": "L",
                "fields": STOCK_BASIC_FIELDS,
            },
            "namechange": {
                "ts_code": symbol,
                "fields": NAMECHANGE_FIELDS,
            },
            "daily": {
                "ts_code": symbol,
                "start_date": start_date,
                "end_date": target,
                "fields": DAILY_FIELDS,
            },
            "trade_cal": {
                "exchange": "SSE",
                "start_date": start_date,
                "end_date": target,
                "is_open": "1",
                "fields": TRADE_CAL_FIELDS,
            },
            "suspend_d": {"ts_code": symbol, "trade_date": inferred},
            "bak_daily": {"ts_code": symbol, "trade_date": inferred},
        }
        query_frames: dict[str, pd.DataFrame] = {}
        for endpoint, expected_params in expected_queries.items():
            query_payload = queries.get(endpoint, {}) or {}
            if not isinstance(query_payload, Mapping):
                blockers.append(f"symbol_query:{symbol}:{endpoint}_payload_invalid")
                query_payload = {}
            blockers.extend(
                f"symbol_query:{symbol}:{item}"
                for item in _validate_query_payload(
                    query_payload,
                    expected_params=expected_params,
                    endpoint=endpoint,
                )
            )
            query_frames[endpoint] = _payload_frame(query_payload)

        stock_frame = query_frames["stock_basic"]
        stock_missing = _required_columns(
            stock_frame,
            {"ts_code", "name", "list_status", "list_date", "delist_date"},
            endpoint="stock_basic",
        )
        blockers.extend(f"symbol_query:{symbol}:{item}" for item in stock_missing)
        if not stock_missing:
            stock_rows = stock_frame.loc[
                stock_frame["ts_code"].map(_normalize_symbol).eq(symbol)
            ].to_dict(orient="records")
            if len(stock_rows) != 1 or len(stock_frame) != 1:
                blockers.append(f"symbol_stock_basic_exact_row_count_mismatch:{symbol}")
            else:
                stock_row = stock_rows[0]
                stock_name = _normalize_identity(stock_row.get("name"))
                if (
                    stock_name != _normalize_identity(proof.get("stock_basic_name"))
                    or not stock_name.endswith("退")
                    or _normalize_reason(stock_row.get("list_status")).upper() != "L"
                    or _compact_date(stock_row.get("delist_date"))
                ):
                    blockers.append(f"symbol_stock_basic_semantic_mismatch:{symbol}")

        name_frame = query_frames["namechange"]
        name_missing = _required_columns(
            name_frame,
            {
                "ts_code",
                "name",
                "start_date",
                "end_date",
                "ann_date",
                "change_reason",
            },
            endpoint="namechange",
        )
        blockers.extend(f"symbol_query:{symbol}:{item}" for item in name_missing)
        if not name_missing:
            active_name_rows: list[dict[str, Any]] = []
            for row in name_frame.to_dict(orient="records"):
                row_start = _compact_date(row.get("start_date"))
                row_end = _compact_date(row.get("end_date"))
                if (
                    _normalize_symbol(row.get("ts_code")) == symbol
                    and row_start
                    and row_start <= target
                    and (not row_end or target <= row_end)
                    and _normalize_identity(row.get("name"))
                    == _normalize_identity(proof.get("stock_basic_name"))
                    and _normalize_reason(row.get("change_reason")) in TERMINAL_CHANGE_REASONS
                ):
                    active_name_rows.append(row)
            if len(active_name_rows) != 1:
                blockers.append(f"symbol_namechange_active_row_count_mismatch:{symbol}")
            else:
                active_name_row = active_name_rows[0]
                if (
                    _compact_date(active_name_row.get("start_date")) != start_date
                    or _compact_date(active_name_row.get("ann_date")) != announcement_date
                    or _normalize_reason(active_name_row.get("change_reason"))
                    != _normalize_reason(proof.get("terminal_change_reason"))
                ):
                    blockers.append(f"symbol_namechange_semantic_mismatch:{symbol}")

        daily_frame = query_frames["daily"]
        daily_missing = _required_columns(
            daily_frame,
            {"ts_code", "trade_date"},
            endpoint="daily",
        )
        blockers.extend(f"symbol_query:{symbol}:{item}" for item in daily_missing)
        if not daily_missing:
            raw_daily_symbols = {
                _normalize_symbol(value) for value in daily_frame["ts_code"].tolist()
            }
            raw_daily_dates = [_compact_date(value) for value in daily_frame["trade_date"].tolist()]
            if (
                raw_daily_symbols != {symbol}
                or any(not value for value in raw_daily_dates)
                or len(raw_daily_dates) != 15
                or len(set(raw_daily_dates)) != 15
                or sorted(raw_daily_dates) != daily_dates
                or _compact_date(proof.get("last_terminal_trade_date")) != daily_dates[-1]
            ):
                blockers.append(f"symbol_daily_semantic_mismatch:{symbol}")

        calendar_frame = query_frames["trade_cal"]
        calendar_missing = _required_columns(
            calendar_frame,
            {"cal_date", "is_open"},
            endpoint="trade_cal",
        )
        blockers.extend(f"symbol_query:{symbol}:{item}" for item in calendar_missing)
        if not calendar_missing:
            open_mask = pd.to_numeric(calendar_frame["is_open"], errors="coerce").eq(1)
            raw_calendar_dates = [
                _compact_date(value) for value in calendar_frame.loc[open_mask, "cal_date"].tolist()
            ]
            if (
                not open_mask.all()
                or any(not value for value in raw_calendar_dates)
                or len(raw_calendar_dates) != len(set(raw_calendar_dates))
                or sorted(raw_calendar_dates) != open_dates
            ):
                blockers.append(f"symbol_trade_cal_semantic_mismatch:{symbol}")

        if len(query_frames["suspend_d"]) != 0:
            blockers.append(f"symbol_suspend_rows_nonempty:{symbol}")
        if len(query_frames["bak_daily"]) != 0:
            blockers.append(f"symbol_bak_daily_rows_nonempty:{symbol}")
    if payload.get("writes_synthetic_bars") is not False:
        blockers.append("synthetic_bar_contract_invalid")
    if payload.get("regulatory_exact_date_suspend_event_claimed") is not False:
        blockers.append("suspension_claim_contract_invalid")
    return list(dict.fromkeys(blockers))


def read_terminal_delisting_evidence(
    path: str | Path,
    *,
    target_trade_date: str,
    candidate_symbols: Iterable[Any],
    pit_membership_path: str | Path,
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
    blockers = validate_terminal_delisting_evidence(
        payload,
        target_trade_date=target_trade_date,
        candidate_symbols=candidate_symbols,
        pit_membership_path=pit_membership_path,
        pit_membership_sha256=pit_membership_sha256,
    )
    return payload, blockers


def resolve_terminal_delisting_evidence(
    provider: Any,
    *,
    cache_root: str | Path,
    target_trade_date: str,
    candidate_symbols: Iterable[Any],
    pit_records_by_symbol: Mapping[str, Any],
    pit_membership_path: str | Path,
    pit_membership_sha256: str,
) -> dict[str, Any]:
    candidates = _normalize_symbols(candidate_symbols)
    if not candidates:
        return {
            "status": "not_needed",
            "verified_symbols": [],
            "inferred_delist_dates": {},
            "blockers": [],
            "evidence_path": "",
            "evidence_sha256": "",
            "payload_sha256": "",
        }
    path = terminal_delisting_cache_path(
        cache_root,
        target_trade_date=target_trade_date,
        candidate_symbols=candidates,
        pit_membership_sha256=pit_membership_sha256,
    )
    cached, cache_blockers = read_terminal_delisting_evidence(
        path,
        target_trade_date=target_trade_date,
        candidate_symbols=candidates,
        pit_membership_path=pit_membership_path,
        pit_membership_sha256=pit_membership_sha256,
    )
    if not cache_blockers:
        return {
            **cached,
            "status": "passed",
            "blockers": [],
            "cache_reused": True,
            "evidence_path": str(path),
            "evidence_sha256": file_sha256(path),
        }

    payload = build_terminal_delisting_evidence(
        provider,
        target_trade_date=target_trade_date,
        candidate_symbols=candidates,
        pit_records_by_symbol=pit_records_by_symbol,
        pit_membership_path=pit_membership_path,
        pit_membership_sha256=pit_membership_sha256,
    )
    if payload.get("all_candidates_verified") is not True:
        return {
            **payload,
            "status": "blocked",
            "blockers": ["terminal_delisting_candidates_not_fully_verified"],
            "cache_reused": False,
            "stale_cache_blockers": cache_blockers,
            "evidence_path": "",
            "evidence_sha256": "",
        }
    _atomic_json_write(path, payload)
    persisted, readback_blockers = read_terminal_delisting_evidence(
        path,
        target_trade_date=target_trade_date,
        candidate_symbols=candidates,
        pit_membership_path=pit_membership_path,
        pit_membership_sha256=pit_membership_sha256,
    )
    if readback_blockers:
        return {
            **payload,
            "status": "blocked",
            "verified_symbols": [],
            "blockers": [
                f"terminal_delisting_evidence_readback:{item}" for item in readback_blockers
            ],
            "cache_reused": False,
            "evidence_path": str(path),
            "evidence_sha256": file_sha256(path) if path.exists() else "",
        }
    return {
        **persisted,
        "status": "passed",
        "blockers": [],
        "cache_reused": False,
        "evidence_path": str(path),
        "evidence_sha256": file_sha256(path),
    }


def terminal_delist_dates(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("inferred_delist_dates", {}) or {}
    if not isinstance(raw, Mapping):
        return {}
    return {
        _normalize_symbol(symbol): _compact_date(value)
        for symbol, value in raw.items()
        if _normalize_symbol(symbol) and _compact_date(value)
    }


__all__ = [
    "TERMINAL_DELISTING_CLASSIFICATION",
    "TERMINAL_DELISTING_POLICY_VERSION",
    "TERMINAL_DELISTING_SCHEMA_VERSION",
    "build_terminal_delisting_evidence",
    "read_terminal_delisting_evidence",
    "resolve_terminal_delisting_evidence",
    "select_terminal_delisting_candidates",
    "terminal_delist_dates",
    "terminal_delisting_cache_path",
    "validate_terminal_delisting_evidence",
]
