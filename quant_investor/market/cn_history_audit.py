"""Full-window strict-Parquet history audit for the CN full-A universe."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from quant_investor.market.cn_nontrading_evidence import (
    build_bak_daily_nontrading_evidence,
    canonical_json_sha256,
    dataframe_sha256,
    evidence_cache_path,
    file_sha256,
    read_evidence_cache,
    symbol_set_sha256,
    write_evidence_cache,
)
from quant_investor.market.cn_terminal_delisting_evidence import (
    read_terminal_delisting_evidence,
    terminal_delist_dates,
)
from quant_investor.market.market_data_reader import (
    MarketDataReader,
    coverage_fingerprint,
)
from quant_investor.market.market_data_store import MarketDataStore
from quant_investor.market.pit_universe import (
    REASON_DELISTED,
    REASON_LISTED,
    REASON_PRE_LISTING,
    evaluate_listing_status,
)


CN_HISTORY_AUDIT_SCHEMA_VERSION = "myquant-cn-history-audit.v4"
SUSPENSION_CONTINUITY_SCHEMA_VERSION = (
    "cn-suspension-continuity-evidence.v1"
)
SUSPENSION_CONTINUITY_CLASSIFICATION = (
    "verified_suspension_continuity_absent"
)


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


def _normalize_symbols(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(value or "").strip().upper()
            for value in values
            if str(value or "").strip()
        }
    )


def _missing_pit_components(
    current_component_symbols: Iterable[str],
    pit_symbols: Iterable[str],
) -> list[str]:
    return sorted(
        set(_normalize_symbols(current_component_symbols))
        - set(_normalize_symbols(pit_symbols))
    )


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _membership_split(
    component_symbols: Iterable[str],
    records_by_symbol: Mapping[str, Any],
    trade_date: str,
    terminal_delist_dates_by_symbol: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    active: list[str] = []
    prelisting: list[str] = []
    delisted: list[str] = []
    unknown: list[str] = []
    unknown_reasons: dict[str, str] = {}
    delisted_on_target: list[str] = []
    terminal_dates = terminal_delist_dates_by_symbol or {}
    for symbol in _normalize_symbols(component_symbols):
        terminal_delist_date = _compact_trade_date(terminal_dates.get(symbol))
        if terminal_delist_date and trade_date >= terminal_delist_date:
            delisted.append(symbol)
            if trade_date == terminal_delist_date:
                delisted_on_target.append(symbol)
            continue
        status = evaluate_listing_status(
            records_by_symbol.get(symbol),
            symbol=symbol,
            as_of=trade_date,
        )
        if status.reason == REASON_PRE_LISTING:
            prelisting.append(symbol)
        elif status.reason == REASON_DELISTED:
            delisted.append(symbol)
            if trade_date == _compact_trade_date(status.delist_date):
                delisted_on_target.append(symbol)
        elif status.reason == REASON_LISTED and status.tradable:
            active.append(symbol)
        else:
            unknown.append(symbol)
            unknown_reasons[symbol] = str(status.reason or "unknown")
    return {
        "active": active,
        "prelisting": prelisting,
        "delisted": delisted,
        "unknown": unknown,
        "unknown_reasons": unknown_reasons,
        "delisted_on_target": delisted_on_target,
    }


def build_cn_history_audit(
    *,
    bars: pd.DataFrame,
    trade_dates: Iterable[str],
    component_symbols: Iterable[str],
    pit_records_by_symbol: Mapping[str, Any],
    suspended_evidence_by_date: Mapping[str, Iterable[str]],
    nontrading_evidence_by_date: Mapping[str, Mapping[str, Any]],
    suspension_continuity_by_date: Mapping[str, Iterable[str]] | None = None,
    evidence_references_by_date: Mapping[str, Mapping[str, Any]] | None = None,
    terminal_delist_dates_by_symbol: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Pure classification core used by the CLI and unit tests."""

    dates = [_compact_trade_date(value) for value in trade_dates]
    dates = [value for value in dates if value]
    if len(dates) != len(set(dates)):
        raise ValueError("trade_dates must be unique")
    components = _normalize_symbols(component_symbols)
    component_set = set(components)
    work = bars.copy() if isinstance(bars, pd.DataFrame) else pd.DataFrame()
    if not work.empty:
        if "ts_code" not in work.columns or "trade_date" not in work.columns:
            raise ValueError("bars must contain ts_code and trade_date")
        work["ts_code"] = work["ts_code"].map(
            lambda value: str(value or "").strip().upper()
        )
        work["trade_date"] = work["trade_date"].map(_compact_trade_date)

    per_date: list[dict[str, Any]] = []
    unresolved_dates: list[str] = []
    primary_absence_dates: list[str] = []
    all_true_missing: dict[str, list[str]] = {}
    all_adj_missing: dict[str, list[str]] = {}
    all_suspension_continuity: dict[str, list[str]] = {}
    evidence_refs = evidence_references_by_date or {}
    continuity_by_date = suspension_continuity_by_date or {}

    for trade_date in dates:
        references = dict(evidence_refs.get(trade_date, {}) or {})
        split = _membership_split(
            components,
            pit_records_by_symbol,
            trade_date,
            terminal_delist_dates_by_symbol,
        )
        active = set(split["active"])
        date_frame = work.loc[work.get("trade_date", pd.Series(dtype=str)).eq(trade_date)].copy()
        observed = {
            str(symbol).strip().upper()
            for symbol in date_frame.get("ts_code", pd.Series(dtype=str)).tolist()
            if str(symbol).strip().upper() in component_set
        }
        observed_active = observed & active
        observed_outside_active = observed - active
        inactive_or_prelisting = set(split["prelisting"]) | set(
            split["delisted"]
        )
        inactive_or_prelisting_absent = inactive_or_prelisting - observed
        terminal_delisting_absent = {
            symbol
            for symbol, delist_date in (
                terminal_delist_dates_by_symbol or {}
            ).items()
            if _compact_trade_date(delist_date)
            and trade_date >= _compact_trade_date(delist_date)
            and symbol in component_set
        } - observed
        primary_missing = active - observed_active
        exact_suspended = (
            set(_normalize_symbols(suspended_evidence_by_date.get(trade_date, [])))
            & primary_missing
        )
        suspension_continuity = (
            set(_normalize_symbols(continuity_by_date.get(trade_date, [])))
            & (primary_missing - exact_suspended)
        )
        suspended = exact_suspended | suspension_continuity
        after_suspension = primary_missing - suspended
        nontrading_payload = dict(
            nontrading_evidence_by_date.get(trade_date, {}) or {}
        )
        verified_nontrading = (
            set(
                _normalize_symbols(
                    nontrading_payload.get("verified_symbols", []) or []
                )
            )
            & after_suspension
        )
        true_missing = after_suspension - verified_nontrading

        adj_missing: set[str] = set()
        if "adj_factor" not in date_frame.columns:
            adj_missing = set(observed_active)
        elif not date_frame.empty:
            date_frame = date_frame.loc[
                date_frame["ts_code"].isin(observed_active)
            ].copy()
            adj_values = pd.to_numeric(
                date_frame["adj_factor"], errors="coerce"
            )
            adj_missing = set(
                date_frame.loc[
                    adj_values.isna() | adj_values.le(0), "ts_code"
                ].astype(str)
            )

        classification_sets = [
            observed,
            exact_suspended,
            suspension_continuity,
            verified_nontrading,
            inactive_or_prelisting_absent,
            true_missing,
        ]
        disjoint = not any(
            left & right
            for index, left in enumerate(classification_sets)
            for right in classification_sets[index + 1 :]
        )
        union_complete = set().union(*classification_sets) == component_set
        blockers: list[str] = []
        if split["unknown"]:
            blockers.append(f"pit_membership_unknown:{len(split['unknown'])}")
        if true_missing:
            blockers.append(f"true_missing:{len(true_missing)}")
        if adj_missing:
            blockers.append(f"adj_factor_missing:{len(adj_missing)}")
        if not disjoint:
            blockers.append("classification_sets_not_disjoint")
        if not union_complete:
            blockers.append("classification_union_incomplete")
        blockers.extend(
            str(item)
            for item in references.get("evidence_blockers", []) or []
            if str(item).strip()
        )
        if primary_missing:
            primary_absence_dates.append(trade_date)
        if blockers:
            unresolved_dates.append(trade_date)
        if true_missing:
            all_true_missing[trade_date] = sorted(true_missing)
        if adj_missing:
            all_adj_missing[trade_date] = sorted(adj_missing)
        if suspension_continuity:
            all_suspension_continuity[trade_date] = sorted(
                suspension_continuity
            )

        per_date.append(
            {
                "trade_date": trade_date,
                "expected_component_scope_count": len(components),
                "expected_active_scope_count": len(active),
                "expected_active_scope_sha256": symbol_set_sha256(active),
                "observed_bar_count": len(observed),
                "observed_bar_symbols_sha256": symbol_set_sha256(observed),
                "observed_active_count": len(observed_active),
                "observed_active_symbols_sha256": symbol_set_sha256(
                    observed_active
                ),
                "observed_outside_active_symbols": sorted(
                    observed_outside_active
                ),
                "excluded_prelisting_symbols": split["prelisting"],
                "excluded_delisted_symbols": split["delisted"],
                "excluded_delisted_on_target_symbols": split[
                    "delisted_on_target"
                ],
                "unknown_membership_symbols": split["unknown"],
                "unknown_membership_reasons": split["unknown_reasons"],
                "primary_daily_absent_count": len(primary_missing),
                "primary_daily_absent_symbols_sha256": symbol_set_sha256(
                    primary_missing
                ),
                "verified_suspended_absent": sorted(suspended),
                "verified_exact_suspended_absent": sorted(exact_suspended),
                "verified_suspension_continuity_absent": sorted(
                    suspension_continuity
                ),
                "verified_nontrading_bak_daily_zero": sorted(
                    verified_nontrading
                ),
                "verified_inactive_or_prelisting_absent": sorted(
                    inactive_or_prelisting_absent
                ),
                "verified_terminal_delisting_absent": sorted(
                    terminal_delisting_absent
                ),
                "true_missing_symbols": sorted(true_missing),
                "adj_factor_missing_symbols": sorted(adj_missing),
                "classification_sets_disjoint": disjoint,
                "classification_union_complete": union_complete,
                "suspend_evidence_path": str(
                    references.get("suspend_evidence_path") or ""
                ),
                "suspend_evidence_sha256": str(
                    references.get("suspend_evidence_sha256") or ""
                ),
                "bak_daily_evidence_path": str(
                    references.get("bak_daily_evidence_path") or ""
                ),
                "bak_daily_evidence_sha256": str(
                    references.get("bak_daily_evidence_sha256") or ""
                ),
                "suspension_continuity_evidence_path": str(
                    references.get("suspension_continuity_evidence_path")
                    or ""
                ),
                "suspension_continuity_evidence_sha256": str(
                    references.get("suspension_continuity_evidence_sha256")
                    or ""
                ),
                "terminal_delisting_evidence_path": str(
                    references.get("terminal_delisting_evidence_path") or ""
                ),
                "terminal_delisting_evidence_sha256": str(
                    references.get("terminal_delisting_evidence_sha256") or ""
                ),
                "blockers": blockers,
                "status": "passed" if not blockers else "blocked",
            }
        )

    return {
        "audit_method": "full_recompute_from_canonical",
        "prior_trade_dates_reused": 0,
        "full_window_recomputed": True,
        "audited_trade_dates_count": len(dates),
        "audited_trade_dates": dates,
        "per_date_count": len(per_date),
        "per_date": per_date,
        "history_primary_absence_dates": primary_absence_dates,
        "history_unresolved_gap_dates": unresolved_dates,
        "history_true_missing_symbols_by_date": all_true_missing,
        "history_adj_factor_missing_symbols_by_date": all_adj_missing,
        "history_suspension_continuity_symbols_by_date": (
            all_suspension_continuity
        ),
        "history_audit_status": "passed" if not unresolved_dates else "blocked",
    }


def _query_open_trade_dates(
    provider: Any,
    *,
    end_date: str,
    days: int,
) -> tuple[list[str], dict[str, Any]]:
    parsed_end = datetime.strptime(end_date, "%Y%m%d")
    start_date = (parsed_end - timedelta(days=max(days * 4, 400))).strftime(
        "%Y%m%d"
    )
    query_params = {
        "exchange": "SSE",
        "start_date": start_date,
        "end_date": end_date,
        "is_open": "1",
    }
    frame = provider.trade_cal(**query_params)
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("trade_cal returned no open-date evidence")
    if "cal_date" not in frame.columns:
        raise RuntimeError("trade_cal response lacks cal_date")
    if "is_open" not in frame.columns:
        raise RuntimeError("trade_cal response lacks is_open")
    open_rows = frame.loc[
        pd.to_numeric(frame["is_open"], errors="coerce").eq(1)
    ]
    open_dates = sorted(
        {
            _compact_trade_date(value)
            for value in open_rows["cal_date"].tolist()
            if _compact_trade_date(value)
            and _compact_trade_date(value) <= end_date
        }
    )
    if len(open_dates) < days:
        raise RuntimeError(
            f"trade_cal returned {len(open_dates)} open dates, fewer than {days}"
        )
    selected = open_dates[-days:]
    if len(selected) != days or len(set(selected)) != days:
        raise RuntimeError("trade_cal selected-date evidence is incomplete")
    return selected, {
        "source": "tushare.trade_cal",
        "query_params": query_params,
        "query_succeeded": True,
        "raw_row_count": int(len(frame)),
        "raw_rows_sha256": dataframe_sha256(frame),
        "selected_open_dates_sha256": symbol_set_sha256(selected),
        "ordered_open_dates_sha256": canonical_json_sha256(selected),
    }


def _read_suspend_evidence_cache(
    path: Path,
    *,
    trade_date: str,
) -> tuple[set[str], dict[str, Any], list[str]]:
    if not path.exists():
        return set(), {}, ["suspend_evidence_cache_missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return set(), {}, [f"suspend_evidence_cache_unreadable:{exc}"]
    if not isinstance(payload, dict):
        return set(), {}, ["suspend_evidence_cache_invalid"]
    blockers: list[str] = []
    canonical_payload = dict(payload)
    declared_sha256 = str(
        canonical_payload.pop("payload_sha256", "") or ""
    ).lower()
    if declared_sha256 != canonical_json_sha256(canonical_payload):
        blockers.append("suspend_evidence_payload_sha256_mismatch")
    if payload.get("version") != 5:
        blockers.append("suspend_evidence_version_mismatch")
    if _compact_trade_date(payload.get("trade_date")) != trade_date:
        blockers.append("suspend_evidence_trade_date_mismatch")
    if payload.get("query_succeeded") is not True:
        blockers.append("suspend_evidence_query_not_succeeded")
    if payload.get("source") != "tushare.suspend_d":
        blockers.append("suspend_evidence_source_mismatch")
    if payload.get("query_variant") != "trade_date":
        blockers.append("suspend_evidence_query_variant_mismatch")
    if payload.get("query_params") != {"trade_date": trade_date}:
        blockers.append("suspend_evidence_query_params_mismatch")
    if payload.get("exact_date_rows_validated") is not True:
        blockers.append("suspend_evidence_exact_date_not_validated")
    if payload.get("continuation_state_complete") is not False:
        blockers.append("suspend_evidence_continuation_contract_invalid")
    if not str(payload.get("query_run_id") or "").strip():
        blockers.append("suspend_evidence_query_run_id_missing")
    exact_event_records = payload.get("exact_event_records", []) or []
    if not isinstance(exact_event_records, list):
        blockers.append("suspend_evidence_exact_event_records_invalid")
        exact_event_records = []
    normalized_event_records: list[dict[str, str]] = []
    for record in exact_event_records:
        if not isinstance(record, Mapping):
            blockers.append("suspend_evidence_exact_event_record_invalid")
            continue
        symbol = str(record.get("ts_code") or "").strip().upper()
        event_date = _compact_trade_date(record.get("trade_date"))
        event_type = str(record.get("suspend_type") or "").strip().upper()
        if not symbol or event_date != trade_date or not event_type:
            blockers.append("suspend_evidence_exact_event_record_invalid")
            continue
        normalized_event_records.append(
            {
                "ts_code": symbol,
                "trade_date": event_date,
                "suspend_type": event_type,
            }
        )
    if normalized_event_records != exact_event_records:
        blockers.append("suspend_evidence_exact_event_records_not_canonical")
    if normalized_event_records != sorted(
        normalized_event_records,
        key=lambda item: (item["ts_code"], item["suspend_type"]),
    ):
        blockers.append("suspend_evidence_exact_event_records_not_sorted")
    if str(payload.get("exact_event_records_sha256") or "").lower() != (
        canonical_json_sha256(exact_event_records)
    ):
        blockers.append("suspend_evidence_exact_event_records_sha256_mismatch")
    try:
        exact_event_row_count = int(payload.get("exact_event_row_count"))
    except (TypeError, ValueError, OverflowError):
        exact_event_row_count = -1
    if exact_event_row_count != len(exact_event_records):
        blockers.append("suspend_evidence_exact_event_row_count_mismatch")
    derived_suspended = {
        record["ts_code"]
        for record in normalized_event_records
        if record["suspend_type"] == "S"
    }
    derived_resumed = {
        record["ts_code"]
        for record in normalized_event_records
        if record["suspend_type"] == "R"
    }
    derived_other = {
        record["ts_code"]
        for record in normalized_event_records
        if record["suspend_type"] not in {"S", "R"}
    }
    symbols = set(_normalize_symbols(payload.get("symbols", []) or []))
    if symbols != derived_suspended:
        blockers.append("suspend_evidence_suspend_records_mismatch")
    if str(payload.get("matched_symbols_sha256") or "").lower() != (
        symbol_set_sha256(symbols)
    ):
        blockers.append("suspend_evidence_symbols_sha256_mismatch")
    resumed = set(_normalize_symbols(payload.get("resume_symbols", []) or []))
    if resumed != derived_resumed:
        blockers.append("suspend_evidence_resume_records_mismatch")
    if str(payload.get("resume_symbols_sha256") or "").lower() != (
        symbol_set_sha256(resumed)
    ):
        blockers.append("suspend_evidence_resume_symbols_sha256_mismatch")
    other_events = set(
        _normalize_symbols(payload.get("other_event_symbols", []) or [])
    )
    if other_events != derived_other:
        blockers.append("suspend_evidence_other_records_mismatch")
    if str(payload.get("other_event_symbols_sha256") or "").lower() != (
        symbol_set_sha256(other_events)
    ):
        blockers.append("suspend_evidence_other_symbols_sha256_mismatch")
    try:
        matched_count = int(payload.get("matched_row_count"))
        raw_count = int(payload.get("raw_row_count"))
    except (TypeError, ValueError, OverflowError):
        matched_count = -1
        raw_count = -1
    if matched_count != len(symbols):
        blockers.append("suspend_evidence_matched_count_mismatch")
    if raw_count < exact_event_row_count:
        blockers.append("suspend_evidence_raw_count_invalid")
    raw_rows_sha256 = str(payload.get("raw_rows_sha256") or "").lower()
    if len(raw_rows_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in raw_rows_sha256
    ):
        blockers.append("suspend_evidence_raw_rows_sha256_invalid")
    return symbols, payload, blockers


def _select_suspension_continuity_symbols(
    *,
    unresolved_symbols: Iterable[str],
    bak_daily_payload: Mapping[str, Any],
    previous_suspended_symbols: Iterable[str],
    current_event_symbols: Iterable[str],
    next_suspended_symbols: Iterable[str],
) -> list[str]:
    """Select only exact-row-missing symbols bracketed by suspension rows."""

    rejected = bak_daily_payload.get("rejected_symbols", {}) or {}
    if not isinstance(rejected, Mapping):
        return []
    rejected_by_symbol = {
        str(symbol or "").strip().upper(): [
            str(reason or "").strip() for reason in reasons or []
        ]
        for symbol, reasons in rejected.items()
        if isinstance(reasons, list)
    }
    previous = set(_normalize_symbols(previous_suspended_symbols))
    current_events = set(_normalize_symbols(current_event_symbols))
    following = set(_normalize_symbols(next_suspended_symbols))
    return sorted(
        symbol
        for symbol in _normalize_symbols(unresolved_symbols)
        if rejected_by_symbol.get(symbol) == ["exact_row_missing"]
        and symbol in previous
        and symbol not in current_events
        and symbol in following
    )


def _suspension_continuity_cache_path(
    output_root: Path,
    *,
    trade_date: str,
    symbols: Iterable[str],
    pit_membership_sha256: str,
) -> Path:
    digest = symbol_set_sha256(symbols)[:16]
    pit_digest = str(pit_membership_sha256 or "").strip().lower()
    if len(pit_digest) != 64 or any(
        character not in "0123456789abcdef" for character in pit_digest
    ):
        raise ValueError(
            "pit_membership_sha256 must be a complete 64-character SHA-256"
        )
    return (
        output_root
        / ".cache"
        / "suspension_continuity"
        / trade_date
        / f"pit_{pit_digest}"
        / f"symbols_{digest}.json"
    )


def _build_suspension_continuity_evidence(
    *,
    trade_date: str,
    previous_open_trade_date: str,
    next_open_trade_date: str,
    selected_open_dates: Iterable[str],
    symbols: Iterable[str],
    previous_suspend_path: Path,
    current_suspend_path: Path,
    next_suspend_path: Path,
    previous_suspend_payload: Mapping[str, Any],
    current_suspend_payload: Mapping[str, Any],
    next_suspend_payload: Mapping[str, Any],
    bak_daily_path: Path,
    bak_daily_payload: Mapping[str, Any],
    pit_path: Path,
    pit_sha256: str,
    canonical_binding: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_symbols = _normalize_symbols(symbols)
    dates = [_compact_trade_date(value) for value in selected_open_dates]
    current_index = dates.index(trade_date)

    def _suspend_reference(
        path: Path,
        payload: Mapping[str, Any],
        expected_date: str,
    ) -> dict[str, Any]:
        return {
            "trade_date": expected_date,
            "path": str(path),
            "file_sha256": file_sha256(path),
            "payload_sha256": str(payload.get("payload_sha256") or ""),
            "query_run_id": str(payload.get("query_run_id") or ""),
            "matched_symbols_sha256": str(
                payload.get("matched_symbols_sha256") or ""
            ),
        }

    payload: dict[str, Any] = {
        "schema_version": SUSPENSION_CONTINUITY_SCHEMA_VERSION,
        "classification": SUSPENSION_CONTINUITY_CLASSIFICATION,
        "source": "hash_bound_suspend_d_open_day_sandwich",
        "trade_date": trade_date,
        "previous_open_trade_date": previous_open_trade_date,
        "next_open_trade_date": next_open_trade_date,
        "calendar_adjacency": {
            "selected_open_dates_sha256": symbol_set_sha256(dates),
            "ordered_open_dates_sha256": canonical_json_sha256(dates),
            "previous_index": current_index - 1,
            "current_index": current_index,
            "next_index": current_index + 1,
            "immediate_open_day_neighbors": True,
        },
        "symbols": normalized_symbols,
        "symbol_count": len(normalized_symbols),
        "symbols_sha256": symbol_set_sha256(normalized_symbols),
        "selection_contract": {
            "primary_bar_absent": True,
            "pit_active_on_target": True,
            "pit_active_on_previous_and_next_open_days": True,
            "primary_bar_absent_on_previous_target_and_next_open_days": True,
            "previous_open_day_exact_suspend_d_contains_symbol": True,
            "target_open_day_has_no_suspend_d_event": True,
            "next_open_day_exact_suspend_d_contains_symbol": True,
            "target_bak_daily_rejection_reason": "exact_row_missing",
        },
        "suspend_evidence": {
            "previous": _suspend_reference(
                previous_suspend_path,
                previous_suspend_payload,
                previous_open_trade_date,
            ),
            "current": _suspend_reference(
                current_suspend_path,
                current_suspend_payload,
                trade_date,
            ),
            "next": _suspend_reference(
                next_suspend_path,
                next_suspend_payload,
                next_open_trade_date,
            ),
        },
        "bak_daily_evidence": {
            "path": str(bak_daily_path),
            "file_sha256": file_sha256(bak_daily_path),
            "payload_sha256": str(
                bak_daily_payload.get("payload_sha256") or ""
            ),
            "primary_missing_symbols_sha256": str(
                bak_daily_payload.get("primary_missing_symbols_sha256") or ""
            ),
        },
        "pit_membership_evidence": {
            "path": str(pit_path),
            "file_sha256": pit_sha256,
            "active_open_dates": [
                previous_open_trade_date,
                trade_date,
                next_open_trade_date,
            ],
        },
        "canonical_evidence": dict(canonical_binding),
        "regulatory_exact_date_suspend_event_claimed": False,
        "writes_synthetic_bars": False,
        "generated_at": _utc_now_iso(),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_suspension_continuity_evidence(
    payload: Mapping[str, Any],
    *,
    trade_date: str,
    previous_open_trade_date: str,
    next_open_trade_date: str,
    selected_open_dates: Iterable[str],
    expected_symbols: Iterable[str],
    pit_sha256: str,
    canonical_binding: Mapping[str, Any],
) -> list[str]:
    """Validate both the derived payload and every referenced source file."""

    blockers: list[str] = []
    work = dict(payload) if isinstance(payload, Mapping) else {}
    declared_payload_sha256 = str(work.pop("payload_sha256", "") or "")
    if declared_payload_sha256 != canonical_json_sha256(work):
        blockers.append("continuity_payload_sha256_mismatch")
    if payload.get("schema_version") != SUSPENSION_CONTINUITY_SCHEMA_VERSION:
        blockers.append("continuity_schema_version_mismatch")
    if payload.get("classification") != SUSPENSION_CONTINUITY_CLASSIFICATION:
        blockers.append("continuity_classification_mismatch")
    if _compact_trade_date(payload.get("trade_date")) != trade_date:
        blockers.append("continuity_trade_date_mismatch")
    if payload.get("previous_open_trade_date") != previous_open_trade_date:
        blockers.append("continuity_previous_open_date_mismatch")
    if payload.get("next_open_trade_date") != next_open_trade_date:
        blockers.append("continuity_next_open_date_mismatch")
    symbols = _normalize_symbols(payload.get("symbols", []) or [])
    expected = _normalize_symbols(expected_symbols)
    if symbols != expected:
        blockers.append("continuity_symbols_mismatch")
    if payload.get("symbol_count") != len(expected):
        blockers.append("continuity_symbol_count_mismatch")
    if payload.get("symbols_sha256") != symbol_set_sha256(expected):
        blockers.append("continuity_symbols_sha256_mismatch")
    dates = [_compact_trade_date(value) for value in selected_open_dates]
    try:
        current_index = dates.index(trade_date)
    except ValueError:
        current_index = -1
    adjacency = payload.get("calendar_adjacency", {}) or {}
    if (
        current_index <= 0
        or current_index >= len(dates) - 1
        or dates[current_index - 1] != previous_open_trade_date
        or dates[current_index + 1] != next_open_trade_date
        or adjacency.get("previous_index") != current_index - 1
        or adjacency.get("current_index") != current_index
        or adjacency.get("next_index") != current_index + 1
        or adjacency.get("immediate_open_day_neighbors") is not True
        or adjacency.get("selected_open_dates_sha256")
        != symbol_set_sha256(dates)
        or adjacency.get("ordered_open_dates_sha256")
        != canonical_json_sha256(dates)
    ):
        blockers.append("continuity_calendar_adjacency_mismatch")
    selection_contract = payload.get("selection_contract", {}) or {}
    expected_contract = {
        "primary_bar_absent": True,
        "pit_active_on_target": True,
        "pit_active_on_previous_and_next_open_days": True,
        "primary_bar_absent_on_previous_target_and_next_open_days": True,
        "previous_open_day_exact_suspend_d_contains_symbol": True,
        "target_open_day_has_no_suspend_d_event": True,
        "next_open_day_exact_suspend_d_contains_symbol": True,
        "target_bak_daily_rejection_reason": "exact_row_missing",
    }
    if selection_contract != expected_contract:
        blockers.append("continuity_selection_contract_mismatch")
    if payload.get("regulatory_exact_date_suspend_event_claimed") is not False:
        blockers.append("continuity_exact_event_claim_invalid")
    if payload.get("writes_synthetic_bars") is not False:
        blockers.append("continuity_synthetic_bar_contract_invalid")

    suspend_evidence = payload.get("suspend_evidence", {}) or {}
    expected_suspend_dates = {
        "previous": previous_open_trade_date,
        "current": trade_date,
        "next": next_open_trade_date,
    }
    suspend_sets: dict[str, set[str]] = {}
    suspend_event_sets: dict[str, set[str]] = {}
    query_run_ids: set[str] = set()
    for position, expected_date in expected_suspend_dates.items():
        reference = suspend_evidence.get(position, {}) or {}
        path = Path(str(reference.get("path") or ""))
        if not path.exists():
            blockers.append(f"continuity_{position}_suspend_file_missing")
            continue
        if reference.get("file_sha256") != file_sha256(path):
            blockers.append(f"continuity_{position}_suspend_file_sha256_mismatch")
        cached_symbols, cached_payload, cached_blockers = (
            _read_suspend_evidence_cache(path, trade_date=expected_date)
        )
        blockers.extend(
            f"continuity_{position}_suspend:{item}"
            for item in cached_blockers
        )
        if reference.get("payload_sha256") != cached_payload.get(
            "payload_sha256"
        ):
            blockers.append(
                f"continuity_{position}_suspend_payload_sha256_mismatch"
            )
        reference_run_id = str(reference.get("query_run_id") or "")
        if reference_run_id != str(cached_payload.get("query_run_id") or ""):
            blockers.append(
                f"continuity_{position}_suspend_query_run_id_mismatch"
            )
        if reference_run_id:
            query_run_ids.add(reference_run_id)
        if reference.get("matched_symbols_sha256") != symbol_set_sha256(
            cached_symbols
        ):
            blockers.append(
                f"continuity_{position}_suspend_symbols_sha256_mismatch"
            )
        suspend_sets[position] = cached_symbols
        suspend_event_sets[position] = {
            str(record.get("ts_code") or "").strip().upper()
            for record in cached_payload.get("exact_event_records", []) or []
            if isinstance(record, Mapping)
            and str(record.get("ts_code") or "").strip()
        }
    if len(query_run_ids) != 1:
        blockers.append("continuity_suspend_query_run_id_not_shared")
    if suspend_sets:
        expected_set = set(expected)
        if not expected_set.issubset(suspend_sets.get("previous", set())):
            blockers.append("continuity_previous_suspend_symbols_missing")
        if expected_set & suspend_event_sets.get("current", set()):
            blockers.append("continuity_current_suspend_event_present")
        if not expected_set.issubset(suspend_sets.get("next", set())):
            blockers.append("continuity_next_suspend_symbols_missing")

    bak_reference = payload.get("bak_daily_evidence", {}) or {}
    bak_path = Path(str(bak_reference.get("path") or ""))
    if not bak_path.exists():
        blockers.append("continuity_bak_daily_file_missing")
    else:
        if bak_reference.get("file_sha256") != file_sha256(bak_path):
            blockers.append("continuity_bak_daily_file_sha256_mismatch")
        try:
            bak_payload = json.loads(bak_path.read_text(encoding="utf-8"))
        except Exception as exc:
            blockers.append(f"continuity_bak_daily_unreadable:{exc}")
            bak_payload = {}
        if bak_reference.get("payload_sha256") != bak_payload.get(
            "payload_sha256"
        ):
            blockers.append("continuity_bak_daily_payload_sha256_mismatch")
        rejected = bak_payload.get("rejected_symbols", {}) or {}
        if any(rejected.get(symbol) != ["exact_row_missing"] for symbol in expected):
            blockers.append("continuity_bak_daily_exact_row_missing_mismatch")
    pit_reference = payload.get("pit_membership_evidence", {}) or {}
    pit_path = Path(str(pit_reference.get("path") or ""))
    if not pit_path.exists():
        blockers.append("continuity_pit_file_missing")
    elif (
        pit_reference.get("file_sha256") != pit_sha256
        or file_sha256(pit_path) != pit_sha256
    ):
        blockers.append("continuity_pit_file_sha256_mismatch")
    if pit_reference.get("active_open_dates") != [
        previous_open_trade_date,
        trade_date,
        next_open_trade_date,
    ]:
        blockers.append("continuity_pit_active_dates_mismatch")
    if payload.get("canonical_evidence") != dict(canonical_binding):
        blockers.append("continuity_canonical_evidence_mismatch")
    return blockers


def _persist_suspension_continuity_evidence(
    path: Path,
    payload: Mapping[str, Any],
    **validation_context: Any,
) -> tuple[dict[str, Any], list[str]]:
    _atomic_json_write(path, payload)
    try:
        persisted = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"continuity_evidence_readback_failed:{exc}"]
    blockers = _validate_suspension_continuity_evidence(
        persisted,
        **validation_context,
    )
    return persisted, blockers


def _resolve_bak_daily_evidence(
    *,
    provider: Any,
    output_root: Path,
    trade_date: str,
    primary_missing_symbols: Iterable[str],
    pit_path: Path,
    pit_sha256: str,
    allow_online: bool,
) -> tuple[dict[str, Any], Path]:
    candidates = _normalize_symbols(primary_missing_symbols)
    path = evidence_cache_path(
        output_root,
        trade_date=trade_date,
        primary_missing_symbols=candidates,
        pit_membership_sha256=pit_sha256,
    )
    if not candidates:
        return {
            "status": "not_needed",
            "verified_symbols": [],
            "blockers": [],
        }, path
    cached, blockers = read_evidence_cache(
        path,
        trade_date=trade_date,
        primary_missing_symbols=candidates,
        pit_membership_sha256=pit_sha256,
    )
    if not blockers:
        return cached, path
    if not allow_online:
        return {
            "verified_symbols": [],
            "blockers": blockers + ["online_evidence_not_authorized"],
        }, path
    if provider is None or not hasattr(provider, "bak_daily"):
        return {
            "verified_symbols": [],
            "blockers": ["bak_daily_provider_unavailable"],
        }, path
    query_params = {"trade_date": trade_date}
    try:
        frame = provider.bak_daily(**query_params)
    except Exception as exc:
        return {
            "verified_symbols": [],
            "blockers": [f"bak_daily_query_failed:{exc}"],
        }, path
    if frame is None or not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame()
    payload = build_bak_daily_nontrading_evidence(
        frame,
        trade_date=trade_date,
        primary_missing_symbols=candidates,
        query_params=query_params,
        pit_membership_path=pit_path,
        pit_membership_sha256=pit_sha256,
    )
    write_evidence_cache(path, payload)
    persisted, readback_blockers = read_evidence_cache(
        path,
        trade_date=trade_date,
        primary_missing_symbols=candidates,
        pit_membership_sha256=pit_sha256,
    )
    if readback_blockers:
        return {
            "verified_symbols": [],
            "blockers": [
                f"bak_daily_evidence_readback:{item}"
                for item in readback_blockers
            ],
        }, path
    return persisted, path


def _read_canonical_window(
    reader: MarketDataReader,
    *,
    table_root: Path,
    serving_root: Path,
    start_date: str,
    end_date: str,
    selected_dates: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "adj_factor",
    ]
    selected = set(selected_dates)

    def _read(root: Path, label: str) -> pd.DataFrame:
        frame = reader._read_dataset(
            root,
            date_range=(start_date, end_date),
            columns=columns,
            derive_symbol_column=False,
        )
        missing_columns = sorted(set(columns) - set(frame.columns))
        if missing_columns:
            raise RuntimeError(
                f"{label} canonical window lacks columns: {missing_columns}"
            )
        work = frame.loc[:, columns].copy()
        work["ts_code"] = work["ts_code"].astype(str).str.strip().str.upper()
        work["trade_date"] = work["trade_date"].map(_compact_trade_date)
        work = work.loc[work["trade_date"].isin(selected)].copy()
        if work.duplicated(subset=["ts_code", "trade_date"]).any():
            raise RuntimeError(f"{label} canonical window has duplicate symbol-dates")
        if work["ts_code"].eq("").any() or work["trade_date"].eq("").any():
            raise RuntimeError(f"{label} canonical window has invalid identity fields")
        for column in columns[2:]:
            numeric = pd.to_numeric(work[column], errors="coerce")
            if numeric.isna().any():
                raise RuntimeError(
                    f"{label} canonical window has invalid {column} values"
                )
            work[column] = numeric
        if work["adj_factor"].le(0).any():
            raise RuntimeError(
                f"{label} canonical window has nonpositive adj_factor values"
            )
        return work.sort_values(["trade_date", "ts_code"]).reset_index(
            drop=True
        )

    table = _read(table_root, "table")
    serving = _read(serving_root, "serving")
    table_sha256 = dataframe_sha256(table)
    serving_sha256 = dataframe_sha256(serving)
    if table_sha256 != serving_sha256:
        table_keys = set(zip(table["trade_date"], table["ts_code"]))
        serving_keys = set(zip(serving["trade_date"], serving["ts_code"]))
        raise RuntimeError(
            "canonical table-serving window mismatch: "
            f"table_only={len(table_keys - serving_keys)},"
            f"serving_only={len(serving_keys - table_keys)}"
        )
    return table, {
        "start_trade_date": start_date,
        "end_trade_date": end_date,
        "selected_trade_date_count": len(selected),
        "table_row_count": int(len(table)),
        "serving_row_count": int(len(serving)),
        "table_sha256": table_sha256,
        "serving_sha256": serving_sha256,
        "table_serving_match": True,
        "duplicate_symbol_date_count": 0,
        "required_columns": columns,
    }


def run_cn_history_audit(
    *,
    data_root: str | Path = "data",
    output_root: str | Path = "data/cn_market_full",
    days: int = 100,
    end_date: str = "auto",
    allow_online: bool = False,
    provider: Any = None,
    suspended_loader: Callable[..., Iterable[str]] | None = None,
    trade_dates: Iterable[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Recompute the entire audit window without reading a prior audit."""

    if days <= 0:
        raise ValueError("days must be positive")
    root = Path(data_root)
    output = Path(output_root)
    store = MarketDataStore(market="CN", data_root=root)
    validation = store.validate_latest()
    if validation.get("status") != "passed":
        raise RuntimeError(
            "strict storage validation failed: "
            + "; ".join(validation.get("blockers", []) or [])
        )
    latest_path = root / "parquet" / "cn" / "_latest.json"
    reader = MarketDataReader(
        market="CN",
        data_root=root,
        mode_policy="strict",
    )
    latest_payload = reader._load_latest_payload(refresh=True)
    active_latest_sha256 = file_sha256(latest_path)
    effective_end = (
        _compact_trade_date(latest_payload.get("latest_complete_trade_date"))
        if str(end_date).strip().lower() == "auto"
        else _compact_trade_date(end_date)
    )
    if not effective_end:
        raise RuntimeError("history audit end date is unavailable")

    if trade_dates is None:
        if not allow_online:
            raise RuntimeError(
                "full history audit requires --allow-online for exact trade_cal evidence"
            )
        if provider is None:
            raise RuntimeError("configured provider is unavailable")
        selected_dates, calendar_evidence = _query_open_trade_dates(
            provider,
            end_date=effective_end,
            days=days,
        )
    else:
        selected_dates = [_compact_trade_date(value) for value in trade_dates]
        selected_dates = [value for value in selected_dates if value]
        if len(selected_dates) != days:
            raise RuntimeError(
                f"explicit trade_dates count {len(selected_dates)} does not equal {days}"
            )
        calendar_evidence = {
            "source": "explicit_test_input",
            "query_succeeded": False,
            "selected_open_dates_sha256": symbol_set_sha256(selected_dates),
            "ordered_open_dates_sha256": canonical_json_sha256(
                selected_dates
            ),
        }
    if (
        len(selected_dates) != days
        or len(set(selected_dates)) != days
        or selected_dates != sorted(selected_dates)
    ):
        raise RuntimeError(
            "history audit requires the exact ordered unique open-date window"
        )
    if selected_dates[-1] != effective_end:
        raise RuntimeError(
            f"history audit end {selected_dates[-1]} does not match {effective_end}"
        )

    components_path = root / "cn_universe" / "cn_index_components.json"
    components_payload = json.loads(components_path.read_text(encoding="utf-8"))
    latest_component_symbols = _normalize_symbols(
        components_payload.get("full_a", []) or []
    )
    if not latest_component_symbols:
        raise RuntimeError("full_a component scope is empty")

    coverage = latest_payload.get("coverage", {}) or {}
    if not isinstance(coverage, Mapping):
        coverage = {}
    pit_binding = reader.coverage_bound_pit()
    if pit_binding.get("status") != "passed":
        raise RuntimeError(
            "active market coverage PIT binding is invalid: "
            + ",".join(pit_binding.get("blockers", []) or [])
        )
    raw_pit_path = str(coverage.get("pit_membership_path") or "").strip()
    pit_path = Path(raw_pit_path)
    pit_sha256 = str(coverage.get("pit_membership_sha256") or "").lower()
    if (
        str(pit_binding.get("canonical_sha256") or "").lower()
        != pit_sha256
    ):
        raise RuntimeError(
            "active market coverage PIT SHA binding is inconsistent"
        )
    pit_records = dict(pit_binding.get("records", {}) or {})
    audit_symbols = _normalize_symbols(pit_records)
    if not audit_symbols:
        raise RuntimeError("PIT full-A membership scope is empty")
    pit_missing_current_components = _missing_pit_components(
        latest_component_symbols,
        audit_symbols,
    )
    if pit_missing_current_components:
        raise RuntimeError(
            "PIT membership omits current full-A components: "
            f"count={len(pit_missing_current_components)},"
            f"symbols={pit_missing_current_components[:20]}"
        )

    terminal_symbols = _normalize_symbols(
        coverage.get("verified_terminal_delisting_symbols", []) or []
    )
    terminal_evidence_payload: dict[str, Any] = {}
    terminal_evidence_path: Path | None = None
    terminal_evidence_file_sha256 = ""
    terminal_dates_by_symbol: dict[str, str] = {}
    if terminal_symbols:
        raw_terminal_path = str(
            coverage.get("verified_terminal_delisting_evidence_path") or ""
        ).strip()
        if not raw_terminal_path:
            raise RuntimeError("terminal delisting evidence path is missing")
        terminal_evidence_path = Path(raw_terminal_path)
        if not terminal_evidence_path.is_absolute():
            terminal_evidence_path = Path.cwd() / terminal_evidence_path
        expected_file_sha256 = str(
            coverage.get("verified_terminal_delisting_evidence_sha256") or ""
        ).lower()
        if not terminal_evidence_path.exists():
            raise RuntimeError(
                "terminal delisting evidence is missing: "
                f"{terminal_evidence_path}"
            )
        terminal_evidence_file_sha256 = file_sha256(
            terminal_evidence_path
        )
        if terminal_evidence_file_sha256 != expected_file_sha256:
            raise RuntimeError(
                "terminal delisting evidence file binding is stale"
            )
        terminal_evidence_payload, terminal_blockers = (
            read_terminal_delisting_evidence(
                terminal_evidence_path,
                target_trade_date=effective_end,
                candidate_symbols=terminal_symbols,
                pit_membership_path=str(pit_path),
                pit_membership_sha256=pit_sha256,
            )
        )
        if terminal_blockers:
            raise RuntimeError(
                "terminal delisting evidence is invalid: "
                + ",".join(terminal_blockers)
            )
        if str(
            terminal_evidence_payload.get("payload_sha256") or ""
        ).lower() != str(
            coverage.get("verified_terminal_delisting_payload_sha256") or ""
        ).lower():
            raise RuntimeError(
                "terminal delisting evidence payload binding is stale"
            )
        terminal_dates_by_symbol = terminal_delist_dates(
            terminal_evidence_payload
        )
        if terminal_dates_by_symbol != dict(
            coverage.get(
                "verified_terminal_delisting_inferred_dates", {}
            )
            or {}
        ):
            raise RuntimeError(
                "terminal delisting inferred-date binding is stale"
            )

    snapshot = reader._require_snapshot()
    bars, canonical_window_evidence = _read_canonical_window(
        reader,
        table_root=snapshot.table_root,
        serving_root=snapshot.serving_root,
        start_date=selected_dates[0],
        end_date=selected_dates[-1],
        selected_dates=selected_dates,
    )
    manifest_path = Path(str(latest_payload.get("manifest_path") or ""))
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    canonical_binding = {
        "snapshot_id": str(latest_payload.get("snapshot_id") or ""),
        "latest_path": str(latest_path),
        "latest_file_sha256": file_sha256(latest_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "table_window_sha256": canonical_window_evidence["table_sha256"],
        "serving_window_sha256": canonical_window_evidence[
            "serving_sha256"
        ],
        "table_serving_match": canonical_window_evidence[
            "table_serving_match"
        ],
    }

    suspended_by_date: dict[str, list[str]] = {}
    suspension_continuity_by_date: dict[str, list[str]] = {}
    nontrading_by_date: dict[str, dict[str, Any]] = {}
    references_by_date: dict[str, dict[str, Any]] = {}
    component_set = set(audit_symbols)

    suspend_paths_by_date: dict[str, Path] = {}
    suspend_payloads_by_date: dict[str, dict[str, Any]] = {}
    suspend_symbols_by_date: dict[str, set[str]] = {}
    suspend_blockers_by_date: dict[str, list[str]] = {}
    observed_symbols_by_date = {
        current_date: (
            set(
                bars.loc[
                    bars["trade_date"].eq(current_date), "ts_code"
                ].astype(str)
            )
            & component_set
        )
        for current_date in selected_dates
    }
    def _load_suspend_date(
        current_date: str,
        *,
        force_refresh: bool = False,
        query_run_id: str | None = None,
    ) -> None:
        evidence_blockers: list[str] = []
        try:
            if suspended_loader is None:
                loader_symbols: set[str] = set()
            elif force_refresh:
                loader_symbols = set(
                    _normalize_symbols(
                        suspended_loader(
                            current_date,
                            force_refresh=True,
                            query_run_id=query_run_id,
                        )
                    )
                )
            else:
                loader_symbols = set(
                    _normalize_symbols(suspended_loader(current_date))
                )
        except Exception as exc:
            loader_symbols = set()
            evidence_blockers.append(f"suspend_loader_failed:{exc}")
        suspend_path = output / ".cache" / f".suspend_{current_date}.json"
        cached_suspend_symbols, _suspend_payload, suspend_blockers = (
            _read_suspend_evidence_cache(
                suspend_path,
                trade_date=current_date,
            )
        )
        evidence_blockers.extend(suspend_blockers)
        if loader_symbols != cached_suspend_symbols:
            evidence_blockers.append("suspend_loader_cache_symbols_mismatch")
        suspend_paths_by_date[current_date] = suspend_path
        suspend_payloads_by_date[current_date] = dict(_suspend_payload)
        suspend_symbols_by_date[current_date] = (
            cached_suspend_symbols if not evidence_blockers else set()
        )
        suspend_blockers_by_date[current_date] = evidence_blockers

    for current_date in selected_dates:
        _load_suspend_date(current_date)

    def _continuity_candidates(
        current_index: int,
        unresolved_symbols: Iterable[str],
        bak_payload: Mapping[str, Any],
    ) -> list[str]:
        if current_index <= 0 or current_index >= len(selected_dates) - 1:
            return []
        current_date = selected_dates[current_index]
        previous_date = selected_dates[current_index - 1]
        next_date = selected_dates[current_index + 1]
        if any(
            suspend_blockers_by_date[date]
            for date in (previous_date, current_date, next_date)
        ):
            return []
        candidates = _select_suspension_continuity_symbols(
            unresolved_symbols=unresolved_symbols,
            bak_daily_payload=bak_payload,
            previous_suspended_symbols=suspend_symbols_by_date[previous_date],
            current_event_symbols={
                str(record.get("ts_code") or "").strip().upper()
                for record in suspend_payloads_by_date[current_date].get(
                    "exact_event_records", []
                )
                or []
                if isinstance(record, Mapping)
            },
            next_suspended_symbols=suspend_symbols_by_date[next_date],
        )
        return [
            symbol
            for symbol in candidates
            if all(
                symbol not in observed_symbols_by_date[date]
                and evaluate_listing_status(
                    pit_records.get(symbol),
                    symbol=symbol,
                    as_of=date,
                ).tradable
                for date in (previous_date, current_date, next_date)
            )
        ]

    continuity_refresh_run_id = (
        "history-audit-continuity-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    refreshed_triplets: set[tuple[str, str, str]] = set()
    for current_index, current_date in enumerate(selected_dates):
        if current_index <= 0 or current_index >= len(selected_dates) - 1:
            continue
        split = _membership_split(
            audit_symbols,
            pit_records,
            current_date,
            terminal_dates_by_symbol,
        )
        primary_missing = set(split["active"]) - observed_symbols_by_date[
            current_date
        ]
        exact_suspended = suspend_symbols_by_date[current_date] & primary_missing
        remaining = primary_missing - exact_suspended
        preliminary_bak, _preliminary_bak_path = _resolve_bak_daily_evidence(
            provider=provider,
            output_root=output,
            trade_date=current_date,
            primary_missing_symbols=remaining,
            pit_path=pit_path,
            pit_sha256=pit_sha256,
            allow_online=allow_online,
        )
        if preliminary_bak.get("blockers", []) or not remaining:
            continue
        verified = set(
            _normalize_symbols(
                preliminary_bak.get("verified_symbols", []) or []
            )
        ) & remaining
        candidates = _continuity_candidates(
            current_index,
            remaining - verified,
            preliminary_bak,
        )
        if not candidates:
            continue
        previous_date = selected_dates[current_index - 1]
        next_date = selected_dates[current_index + 1]
        triplet = (previous_date, current_date, next_date)
        query_run_ids = {
            str(suspend_payloads_by_date[date].get("query_run_id") or "")
            for date in triplet
        }
        if len(query_run_ids) == 1 and "" not in query_run_ids:
            continue
        if triplet in refreshed_triplets:
            continue
        refreshed_triplets.add(triplet)
        for refresh_date in triplet:
            _load_suspend_date(
                refresh_date,
                force_refresh=True,
                query_run_id=continuity_refresh_run_id,
            )

    for current_index, current_date in enumerate(selected_dates):
        split = _membership_split(
            audit_symbols,
            pit_records,
            current_date,
            terminal_dates_by_symbol,
        )
        active = set(split["active"])
        observed = observed_symbols_by_date[current_date]
        primary_missing = active - observed
        evidence_blockers: list[str] = []
        exact_suspended = (
            suspend_symbols_by_date[current_date] & primary_missing
        )
        suspended_by_date[current_date] = sorted(exact_suspended)
        remaining = primary_missing - exact_suspended
        bak_payload, bak_path = _resolve_bak_daily_evidence(
            provider=provider,
            output_root=output,
            trade_date=current_date,
            primary_missing_symbols=remaining,
            pit_path=pit_path,
            pit_sha256=pit_sha256,
            allow_online=allow_online,
        )
        nontrading_by_date[current_date] = bak_payload
        evidence_blockers.extend(
            f"bak_daily_evidence:{item}"
            for item in bak_payload.get("blockers", []) or []
            if str(item).strip()
        )
        verified_nontrading = set(
            _normalize_symbols(bak_payload.get("verified_symbols", []) or [])
        ) & remaining
        unresolved_after_bak = remaining - verified_nontrading

        continuity_symbols: list[str] = []
        continuity_path: Path | None = None
        if (
            unresolved_after_bak
            and current_index > 0
            and current_index < len(selected_dates) - 1
            and not (bak_payload.get("blockers", []) or [])
        ):
            previous_date = selected_dates[current_index - 1]
            next_date = selected_dates[current_index + 1]
            neighbor_blockers = {
                previous_date: suspend_blockers_by_date[previous_date],
                current_date: suspend_blockers_by_date[current_date],
                next_date: suspend_blockers_by_date[next_date],
            }
            invalid_neighbors = [
                date
                for date, blockers in neighbor_blockers.items()
                if blockers
            ]
            if invalid_neighbors:
                evidence_blockers.append(
                    "suspension_continuity_neighbor_evidence_invalid:"
                    + ",".join(invalid_neighbors)
                )
            else:
                candidates = _continuity_candidates(
                    current_index,
                    unresolved_after_bak,
                    bak_payload,
                )
                query_run_ids = {
                    str(
                        suspend_payloads_by_date[date].get("query_run_id")
                        or ""
                    )
                    for date in (
                        previous_date,
                        current_date,
                        next_date,
                    )
                }
                if candidates and (
                    len(query_run_ids) != 1 or "" in query_run_ids
                ):
                    evidence_blockers.append(
                        "suspension_continuity_query_run_not_shared"
                    )
                    candidates = []
                if candidates:
                    continuity_path = _suspension_continuity_cache_path(
                        output,
                        trade_date=current_date,
                        symbols=candidates,
                        pit_membership_sha256=pit_sha256,
                    )
                    continuity_payload = (
                        _build_suspension_continuity_evidence(
                            trade_date=current_date,
                            previous_open_trade_date=previous_date,
                            next_open_trade_date=next_date,
                            selected_open_dates=selected_dates,
                            symbols=candidates,
                            previous_suspend_path=suspend_paths_by_date[
                                previous_date
                            ],
                            current_suspend_path=suspend_paths_by_date[
                                current_date
                            ],
                            next_suspend_path=suspend_paths_by_date[next_date],
                            previous_suspend_payload=(
                                suspend_payloads_by_date[previous_date]
                            ),
                            current_suspend_payload=(
                                suspend_payloads_by_date[current_date]
                            ),
                            next_suspend_payload=(
                                suspend_payloads_by_date[next_date]
                            ),
                            bak_daily_path=bak_path,
                            bak_daily_payload=bak_payload,
                            pit_path=pit_path,
                            pit_sha256=pit_sha256,
                            canonical_binding=canonical_binding,
                        )
                    )
                    _persisted, continuity_blockers = (
                        _persist_suspension_continuity_evidence(
                            continuity_path,
                            continuity_payload,
                            trade_date=current_date,
                            previous_open_trade_date=previous_date,
                            next_open_trade_date=next_date,
                            selected_open_dates=selected_dates,
                            expected_symbols=candidates,
                            pit_sha256=pit_sha256,
                            canonical_binding=canonical_binding,
                        )
                    )
                    if continuity_blockers:
                        evidence_blockers.extend(
                            f"suspension_continuity_evidence:{item}"
                            for item in continuity_blockers
                        )
                    else:
                        continuity_symbols = candidates
        unresolved_after_continuity = unresolved_after_bak - set(
            continuity_symbols
        )
        if unresolved_after_continuity:
            evidence_blockers.extend(
                suspend_blockers_by_date[current_date]
            )
        suspension_continuity_by_date[current_date] = continuity_symbols
        suspend_path = suspend_paths_by_date[current_date]
        references_by_date[current_date] = {
            "suspend_evidence_path": str(suspend_path) if suspend_path.exists() else "",
            "suspend_evidence_sha256": (
                file_sha256(suspend_path) if suspend_path.exists() else ""
            ),
            "bak_daily_evidence_path": str(bak_path) if bak_path.exists() else "",
            "bak_daily_evidence_sha256": (
                file_sha256(bak_path) if bak_path.exists() else ""
            ),
            "suspension_continuity_evidence_path": (
                str(continuity_path)
                if continuity_path is not None and continuity_path.exists()
                else ""
            ),
            "suspension_continuity_evidence_sha256": (
                file_sha256(continuity_path)
                if continuity_path is not None and continuity_path.exists()
                else ""
            ),
            "terminal_delisting_evidence_path": (
                str(terminal_evidence_path)
                if terminal_evidence_path is not None
                else ""
            ),
            "terminal_delisting_evidence_sha256": (
                terminal_evidence_file_sha256
                if terminal_evidence_path is not None
                else ""
            ),
            "evidence_blockers": evidence_blockers,
        }

    audit = build_cn_history_audit(
        bars=bars,
        trade_dates=selected_dates,
        component_symbols=audit_symbols,
        pit_records_by_symbol=pit_records,
        suspended_evidence_by_date=suspended_by_date,
        nontrading_evidence_by_date=nontrading_by_date,
        suspension_continuity_by_date=suspension_continuity_by_date,
        evidence_references_by_date=references_by_date,
        terminal_delist_dates_by_symbol=terminal_dates_by_symbol,
    )
    if (
        audit.get("audited_trade_dates_count") != days
        or audit.get("per_date_count") != days
    ):
        raise RuntimeError(
            "history audit did not recompute the requested full window"
        )
    if file_sha256(latest_path) != active_latest_sha256:
        raise RuntimeError(
            "active market pointer changed during history audit"
        )
    report: dict[str, Any] = {
        "schema_version": CN_HISTORY_AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "market": "CN",
        "target_trade_date": effective_end,
        "effective_trade_date": effective_end,
        "stable_trade_date": effective_end,
        "history_audit_window": {
            "start": selected_dates[0],
            "end": selected_dates[-1],
        },
        **audit,
        "calendar_evidence": calendar_evidence,
        "suspension_continuity_refresh": {
            "query_run_id": (
                continuity_refresh_run_id if refreshed_triplets else ""
            ),
            "refreshed_triplets": [
                list(triplet) for triplet in sorted(refreshed_triplets)
            ],
        },
        "canonical_window_evidence": canonical_window_evidence,
        "canonical": {
            "snapshot_id": str(latest_payload.get("snapshot_id") or ""),
            "latest_path": str(latest_path),
            "latest_sha256": active_latest_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "coverage_fingerprint_sha256": coverage_fingerprint(
                latest_payload.get("coverage", {})
            ),
            "latest_complete_trade_date": effective_end,
            "storage_validation": validation,
        },
        "components_evidence": {
            "path": str(components_path),
            "sha256": file_sha256(components_path),
            "latest_component_symbol_count": len(latest_component_symbols),
            "latest_component_symbols_sha256": symbol_set_sha256(
                latest_component_symbols
            ),
            "audit_scope_source": "pit_stock_basic_membership_all_records",
            "audit_scope_symbol_count": len(audit_symbols),
            "audit_scope_symbols_sha256": symbol_set_sha256(audit_symbols),
        },
        "pit_membership_evidence": {
            "path": str(pit_path),
            "sha256": pit_sha256,
            "record_count": len(pit_records),
            "coverage_schema_version": str(
                coverage.get("coverage_schema_version") or ""
            ),
            "generation_id": str(coverage.get("pit_generation_id") or ""),
            "generation_manifest_path": str(
                coverage.get("pit_generation_manifest_path") or ""
            ),
            "generation_manifest_sha256": str(
                coverage.get("pit_generation_manifest_sha256") or ""
            ),
            "binding_source": "active_market_coverage",
        },
        "terminal_delisting_evidence": {
            "path": (
                str(terminal_evidence_path)
                if terminal_evidence_path is not None
                else ""
            ),
            "file_sha256": terminal_evidence_file_sha256,
            "payload_sha256": str(
                terminal_evidence_payload.get("payload_sha256") or ""
            ),
            "symbols": terminal_symbols,
            "inferred_delist_dates": terminal_dates_by_symbol,
        },
        "maintenance_status": (
            "complete" if audit["history_audit_status"] == "passed" else "partial"
        ),
        "latest_canonical_ready": True,
        "portfolio_data_ready": audit["history_audit_status"] == "passed",
        "historical_upsert_count": 0,
        "synthetic_bar_count": 0,
        "no_analysis_or_trading_side_effects": True,
    }
    report["audit_sha256"] = canonical_json_sha256(report)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output / f"history_audit_{timestamp}_full_recompute.json"
    _atomic_json_write(output_path, report)
    return report, output_path


__all__ = [
    "CN_HISTORY_AUDIT_SCHEMA_VERSION",
    "build_cn_history_audit",
    "run_cn_history_audit",
]
