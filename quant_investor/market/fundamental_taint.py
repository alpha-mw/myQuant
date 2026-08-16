"""Promotion-ineligible reachability proof for deferred Fundamental sources.

The module is diagnostic only.  It validates a deferred provider capture,
replays the same financial event kernel used by the safe-successor derivation,
and proves whether every unsupported observation is unreachable through one
bounded target cutoff.  It cannot build a generation or write a canonical
pointer.
"""

from __future__ import annotations

from itertools import groupby
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, NoReturn, Sequence

import pandas as pd
import pyarrow.parquet as pq

from .fundamental_incremental import (
    FINANCIAL_TABLES,
    SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256,
    replay_successor_event_trace,
    successor_financial_row_binding,
    successor_period_anchor_equal,
    successor_period_winner,
)
from .fundamental_provider_contract import frame_fingerprint
from .fundamental_successor_source import (
    iter_unsupported_observations,
    load_capture_symbol_rows,
    validate_successor_capture_fileset,
)


FUNDAMENTAL_TAINT_PROOF_SCHEMA = (
    "cn-fundamental-unsupported-observation-taint-proof.v1"
)
FUNDAMENTAL_TAINT_REPORT_SCHEMA = (
    "cn-fundamental-taint-non-reachability-report.v1"
)
FUNDAMENTAL_SOURCE_AUTHORITY_CLOSURE_SCHEMA = (
    "cn-fundamental-deferred-source-analysis-closure.v1"
)

_DATE_RE = re.compile(r"^[0-9]{8}$", re.ASCII)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_MAX_SYMBOL_ROWS = 100_000
_MAX_SYMBOL_BYTES = 64 * 1024 * 1024


class FundamentalTaintError(RuntimeError):
    """One deterministic taint-analysis blocker."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}" if message else self.code)


def _fail(code: str, message: str = "") -> NoReturn:
    raise FundamentalTaintError(code, message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FundamentalTaintError("TAINT_RECEIPT_NOT_CANONICAL") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sealed(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    body = dict(value)
    if field in body:
        _fail("TAINT_RECEIPT_FIELD_COLLISION", field)
    body[field] = _sha256(_canonical_bytes(body))
    return body


def _validate_seal(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    body = dict(value)
    digest = body.pop(field, None)
    if (
        type(digest) is not str
        or _SHA_RE.fullmatch(digest) is None
        or _sha256(_canonical_bytes(body)) != digest
    ):
        _fail("TAINT_RECEIPT_SEAL_INVALID", field)
    return dict(value)


def _date(value: Any, *, label: str) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.strftime("%Y%m%d")
    text = "".join(character for character in str(value or "") if character.isdigit())
    if _DATE_RE.fullmatch(text) is None:
        _fail("TAINT_DATE_INVALID", label)
    try:
        pd.Timestamp(text)
    except (TypeError, ValueError) as exc:
        raise FundamentalTaintError("TAINT_DATE_INVALID", label) from exc
    return text


def _symbol_rows(path: Path, symbol: str) -> pd.DataFrame:
    try:
        table = pq.read_table(path, filters=[("ts_code", "=", symbol)])
    except (OSError, ValueError) as exc:
        raise FundamentalTaintError(
            "TAINT_SUPPORT_READ_FAILED", str(path)
        ) from exc
    if table.num_rows > _MAX_SYMBOL_ROWS or table.nbytes > _MAX_SYMBOL_BYTES:
        _fail("TAINT_RESOURCE_BUDGET_EXCEEDED", symbol)
    frame = table.to_pandas()
    if not frame.empty and set(frame["ts_code"].astype(str)) != {symbol}:
        _fail("TAINT_SUBJECT_SCOPE_DRIFT", symbol)
    return frame


def _latest_parent_anchor(
    parent_daily_path: Path,
    *,
    symbol: str,
    parent_cutoff: str,
) -> tuple[dict[str, Any] | None, str]:
    frame = _symbol_rows(parent_daily_path, symbol)
    if frame.empty:
        return None, frame_fingerprint(frame)
    frame = frame.copy()
    frame["__trade_date"] = frame["trade_date"].map(
        lambda value: _date(value, label="parent trade_date")
    )
    frame = frame.loc[frame["__trade_date"] <= parent_cutoff].copy()
    if frame.empty:
        return None, frame_fingerprint(frame)
    latest_date = max(frame["__trade_date"])
    latest = frame.loc[frame["__trade_date"] == latest_date].drop(
        columns=["__trade_date"]
    )
    if len(latest) != 1:
        _fail("TAINT_PARENT_ANCHOR_NOT_UNIQUE", symbol)
    return latest.iloc[0].to_dict(), frame_fingerprint(latest.reset_index(drop=True))


def _membership_evidence(
    membership_path: Path,
    *,
    symbol: str,
) -> dict[str, Any]:
    try:
        table = pq.read_table(
            membership_path,
            filters=[("symbol", "=", symbol)],
            columns=[
                "symbol",
                "name",
                "list_date",
                "effective_from",
                "effective_to",
                "source_list_status",
            ],
        )
    except (OSError, ValueError) as exc:
        raise FundamentalTaintError(
            "TAINT_PIT_MEMBERSHIP_READ_FAILED", symbol
        ) from exc
    frame = table.to_pandas()
    if frame.empty or set(frame["symbol"].astype(str)) != {symbol}:
        _fail("TAINT_PIT_IDENTITY_UNCLOSED", symbol)
    list_dates = {
        _date(value, label="PIT list_date") for value in frame["list_date"]
    }
    if len(list_dates) != 1:
        _fail("TAINT_PIT_LIST_DATE_NOT_UNIQUE", symbol)
    names = sorted({str(value or "") for value in frame["name"]})
    body = {
        "symbol": symbol,
        "names": names,
        "list_date": next(iter(list_dates)),
        "membership_row_count": len(frame),
        "membership_frame_fingerprint": frame_fingerprint(frame),
    }
    body["identity_sha256"] = _sha256(_canonical_bytes(body))
    return body


def _parent_historical_presence(
    parent_period_path: Path,
    *,
    symbol: str,
    end_date: str,
    availability: str,
) -> tuple[bool, str]:
    frame = _symbol_rows(parent_period_path, symbol)
    if frame.empty:
        return False, frame_fingerprint(frame)
    selected = frame.loc[
        frame["end_date"].map(lambda value: _date(value, label="period end_date"))
        .eq(end_date)
        & frame["availability_date"].map(
            lambda value: _date(value, label="period availability")
        ).eq(availability)
    ].copy()
    return not selected.empty, frame_fingerprint(selected.reset_index(drop=True))


def _lineage_bindings(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        bindings = row.get("source_row_bindings", {})
        if isinstance(bindings, Mapping):
            values.update(str(value) for value in bindings.values() if value)
        previous = row.get("previous_year_income", {})
        if isinstance(previous, Mapping) and previous.get("row_binding"):
            values.add(str(previous["row_binding"]))
    return values


def _proof_for_observation(
    *,
    observation: Mapping[str, Any],
    financial_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    parent_anchor: Mapping[str, Any] | None,
    parent_anchor_fingerprint: str,
    historical_presence: bool,
    historical_fingerprint: str,
    membership: Mapping[str, Any],
    parent_cutoff: str,
    target_cutoff: str,
    support_start: str,
    authority_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(observation["row"])
    symbol, end_date, availability = tuple(observation["business_key"])
    row_binding = successor_financial_row_binding(
        table="balancesheet",
        row=row,
        target_cutoff=target_cutoff,
    )
    reasons: list[str] = []
    state = "TAINTED_NON_REACHABLE"
    list_date = str(membership["list_date"])
    if not (end_date < list_date and availability < list_date):
        state = "BLOCKING_UNKNOWN"
        reasons.append("PIT_PRELISTING_QUALIFICATION_FAILED")
    if parent_anchor is None:
        state = "BLOCKING_UNKNOWN"
        reasons.append("PARENT_DAILY_ANCHOR_MISSING")
    trace: dict[str, Any] | None = None
    try:
        trace = replay_successor_event_trace(
            financial_rows=financial_rows,
            symbol=symbol,
            parent_cutoff=parent_cutoff,
            target_cutoff=target_cutoff,
            support_start=support_start,
        )
    except Exception as exc:
        state = "BLOCKING_UNKNOWN"
        reasons.append(getattr(exc, "code", "DEPENDENCY_REPLAY_FAILED"))
    boundary_equal = False
    post_seam_same_key_events: list[str] = []
    target_winner_end_date = ""
    trace_receipt: Mapping[str, Any] = {}
    lineage_bindings: set[str] = set()
    if trace is not None:
        trace_receipt = trace["trace_receipt"]
        boundary = trace["boundary_winner"]
        if boundary is not None and parent_anchor is not None:
            boundary_equal = successor_period_anchor_equal(
                parent_anchor,
                boundary,
            )
        if not boundary_equal:
            state = "BLOCKING_UNKNOWN"
            reasons.append("SEAM_PERIOD_ANCHOR_MISMATCH")
        for table, rows in financial_rows.items():
            for candidate in rows:
                candidate_end = _date(
                    candidate.get("end_date"),
                    label=f"{table} end_date",
                )
                candidate_availability = _date(
                    candidate.get("availability_date")
                    or candidate.get("f_ann_date")
                    or candidate.get("ann_date"),
                    label=f"{table} availability",
                )
                if (
                    candidate_end == end_date
                    and parent_cutoff < candidate_availability <= target_cutoff
                ):
                    post_seam_same_key_events.append(
                        f"{table}|{candidate_availability}|{candidate_end}"
                    )
        if post_seam_same_key_events:
            state = "BLOCKING_UNKNOWN"
            reasons.append("TAINT_STATE_REACHABLE_POST_SEAM")
        lineage_bindings = _lineage_bindings(trace["delta_lineage"])
        if row_binding in lineage_bindings:
            state = "BLOCKING_UNKNOWN"
            reasons.append("TAINT_BINDING_PRESENT_IN_DELTA_LINEAGE")
        candidates = [
            value
            for value in [trace["boundary_winner"], *trace["delta_records"]]
            if value is not None
        ]
        try:
            winner = successor_period_winner(candidates)
        except Exception as exc:
            winner = None
            state = "BLOCKING_UNKNOWN"
            reasons.append(getattr(exc, "code", "TARGET_WINNER_UNCLOSED"))
        if winner is None:
            state = "BLOCKING_UNKNOWN"
            reasons.append("TARGET_PERIOD_WINNER_MISSING")
        else:
            target_winner_end_date = _date(
                winner.get("end_date"),
                label="target winner end_date",
            )
            if target_winner_end_date == end_date:
                state = "BLOCKING_UNKNOWN"
                reasons.append("TAINT_PERIOD_SELECTED_AT_TARGET")
    if not historical_presence:
        state = "BLOCKING_UNKNOWN"
        reasons.append("FROZEN_PREDECESSOR_HISTORY_PRESENCE_UNCONFIRMED")
    reasons = sorted(set(reasons))
    body = {
        "schema_version": FUNDAMENTAL_TAINT_PROOF_SCHEMA,
        "observation_state": state,
        "blocking_reasons": reasons,
        "symbol": symbol,
        "end_date": end_date,
        "availability_date": availability,
        "source_row_sha256": str(observation["row_sha256"]),
        "production_row_binding": row_binding,
        "source_record_path": str(observation["record_path"]),
        "source_request_ordinal": int(observation["ordinal"]),
        "pit_identity": dict(membership),
        "strict_prelisting_qualified": (
            end_date < list_date and availability < list_date
        ),
        "predecessor_prefix_preserved": True,
        "historical_derived_row_still_present": historical_presence,
        "historical_derived_row_fingerprint": historical_fingerprint,
        "current_provider_comp_type7_authority_accepted": False,
        "seam_period_anchor_equal": boundary_equal,
        "parent_daily_full_anchor_fingerprint": parent_anchor_fingerprint,
        "nonfinancial_daily_lanes": (
            "FORECAST_DAILY_BASIC_SIZE_INDEPENDENT_OF_FINANCIAL_TAINT"
        ),
        "post_seam_same_key_events": sorted(post_seam_same_key_events),
        "taint_binding_in_delta_lineage": row_binding in lineage_bindings,
        "target_period_winner_end_date": target_winner_end_date,
        "tainted_source_row_entered_suffix": False,
        "tainted_state_reachable_through_target": state != "TAINTED_NON_REACHABLE",
        "homogeneous_historical_source_reconstruction": False,
        "dependency_contract_sha256": (
            SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
        ),
        "event_trace_receipt": dict(trace_receipt),
        "authority_bindings": dict(authority_bindings),
    }
    return _sealed(body, field="proof_sha256")


def analyze_deferred_fundamental_taints(
    *,
    fileset_root: str | Path,
    parent_period_path: str | Path,
    parent_daily_path: str | Path,
    membership_path: str | Path,
    parent_cutoff: str,
    target_cutoff: str,
    support_start: str,
    authority_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one sealed, non-promotable analysis closure."""

    parent = _date(parent_cutoff, label="parent_cutoff")
    target = _date(target_cutoff, label="target_cutoff")
    if target <= parent:
        _fail("TAINT_TARGET_NOT_AFTER_PARENT")
    manifest = validate_successor_capture_fileset(fileset_root)
    for key, value in authority_bindings.items():
        if type(key) is not str or not isinstance(value, Mapping):
            _fail("TAINT_AUTHORITY_BINDING_INVALID")
        digest = value.get("sha256")
        if type(digest) is not str or _SHA_RE.fullmatch(digest) is None:
            _fail("TAINT_AUTHORITY_BINDING_INVALID", key)
    observations: Iterator[dict[str, Any]] = iter_unsupported_observations(
        fileset_root
    )
    proofs: list[dict[str, Any]] = []
    for symbol, grouped in groupby(
        observations,
        key=lambda value: str(value["business_key"][0]),
    ):
        financial_rows = {
            table: load_capture_symbol_rows(
                fileset_root,
                table=table,
                symbol=symbol,
                maximum_rows=_MAX_SYMBOL_ROWS,
                maximum_bytes=_MAX_SYMBOL_BYTES,
            )
            for table in FINANCIAL_TABLES
        }
        parent_anchor, parent_anchor_fingerprint = _latest_parent_anchor(
            Path(parent_daily_path),
            symbol=symbol,
            parent_cutoff=parent,
        )
        membership = _membership_evidence(
            Path(membership_path),
            symbol=symbol,
        )
        for observation in grouped:
            _symbol, end_date, availability = tuple(
                observation["business_key"]
            )
            historical_presence, historical_fingerprint = (
                _parent_historical_presence(
                    Path(parent_period_path),
                    symbol=symbol,
                    end_date=end_date,
                    availability=availability,
                )
            )
            proofs.append(
                _proof_for_observation(
                    observation=observation,
                    financial_rows=financial_rows,
                    parent_anchor=parent_anchor,
                    parent_anchor_fingerprint=parent_anchor_fingerprint,
                    historical_presence=historical_presence,
                    historical_fingerprint=historical_fingerprint,
                    membership=membership,
                    parent_cutoff=parent,
                    target_cutoff=target,
                    support_start=support_start,
                    authority_bindings=authority_bindings,
                )
            )
    proofs.sort(key=lambda value: value["proof_sha256"])
    passed = sum(
        proof["observation_state"] == "TAINTED_NON_REACHABLE"
        for proof in proofs
    )
    blocked = len(proofs) - passed
    status = "PASS" if blocked == 0 else "BLOCKED"
    report_body = {
        "schema_version": FUNDAMENTAL_TAINT_REPORT_SCHEMA,
        "taint_analysis_status": status,
        "parent_cutoff": parent,
        "target_cutoff": target,
        "dependency_contract_sha256": (
            SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
        ),
        "source_capture_manifest_sha256": manifest["manifest_sha256"],
        "deferred_observation_count": len(proofs),
        "tainted_non_reachable_count": passed,
        "blocking_unknown_count": blocked,
        "proofs": proofs,
        "authority_state": "DEFERRED_UNSUPPORTED_OBSERVATIONS",
        "authoritative_source_ready": False,
        "staging_eligible": False,
        "promotion_eligible": False,
        "canonical_write_authorized": False,
        "usable_for_investment_research": False,
    }
    report = _sealed(report_body, field="report_sha256")
    closure_body = {
        "schema_version": FUNDAMENTAL_SOURCE_AUTHORITY_CLOSURE_SCHEMA,
        "taint_analysis_status": status,
        "capture_manifest_sha256": manifest["manifest_sha256"],
        "taint_report_sha256": report["report_sha256"],
        "authority_state": "DEFERRED_UNSUPPORTED_OBSERVATIONS",
        "authoritative_source_ready": False,
        "staging_eligible": False,
        "promotion_eligible": False,
        "canonical_write_authorized": False,
        "usable_for_investment_research": False,
    }
    closure = _sealed(closure_body, field="closure_sha256")
    return validate_taint_analysis_result(
        {"report": report, "source_analysis_closure": closure}
    )


def validate_taint_analysis_result(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate proof seals and aggregate universal closure."""

    if set(value) != {"report", "source_analysis_closure"}:
        _fail("TAINT_ANALYSIS_RESULT_INVALID")
    report = _validate_seal(value["report"], field="report_sha256")
    closure = _validate_seal(
        value["source_analysis_closure"],
        field="closure_sha256",
    )
    proofs = report.get("proofs")
    if (
        report.get("schema_version") != FUNDAMENTAL_TAINT_REPORT_SCHEMA
        or type(proofs) is not list
        or report.get("authority_state")
        != "DEFERRED_UNSUPPORTED_OBSERVATIONS"
        or any(
            report.get(field) is not False
            for field in (
                "authoritative_source_ready",
                "staging_eligible",
                "promotion_eligible",
                "canonical_write_authorized",
                "usable_for_investment_research",
            )
        )
    ):
        _fail("TAINT_REPORT_CONTRACT_INVALID")
    validated_proofs: list[dict[str, Any]] = []
    for proof in proofs:
        validated = _validate_seal(proof, field="proof_sha256")
        if (
            validated.get("schema_version") != FUNDAMENTAL_TAINT_PROOF_SCHEMA
            or validated.get("observation_state")
            not in {"TAINTED_NON_REACHABLE", "BLOCKING_UNKNOWN"}
            or validated.get("current_provider_comp_type7_authority_accepted")
            is not False
            or validated.get("tainted_source_row_entered_suffix") is not False
        ):
            _fail("TAINT_PROOF_CONTRACT_INVALID")
        validated_proofs.append(validated)
    passed = sum(
        proof["observation_state"] == "TAINTED_NON_REACHABLE"
        for proof in validated_proofs
    )
    blocked = len(validated_proofs) - passed
    expected_status = "PASS" if blocked == 0 else "BLOCKED"
    if (
        report.get("deferred_observation_count") != len(validated_proofs)
        or report.get("tainted_non_reachable_count") != passed
        or report.get("blocking_unknown_count") != blocked
        or report.get("taint_analysis_status") != expected_status
        or closure.get("schema_version")
        != FUNDAMENTAL_SOURCE_AUTHORITY_CLOSURE_SCHEMA
        or closure.get("taint_analysis_status") != expected_status
        or closure.get("capture_manifest_sha256")
        != report.get("source_capture_manifest_sha256")
        or closure.get("taint_report_sha256") != report.get("report_sha256")
        or any(
            closure.get(field) is not False
            for field in (
                "authoritative_source_ready",
                "staging_eligible",
                "promotion_eligible",
                "canonical_write_authorized",
                "usable_for_investment_research",
            )
        )
    ):
        _fail("TAINT_ANALYSIS_CLOSURE_INVALID")
    return {"report": report, "source_analysis_closure": closure}


__all__ = [
    "FUNDAMENTAL_SOURCE_AUTHORITY_CLOSURE_SCHEMA",
    "FUNDAMENTAL_TAINT_PROOF_SCHEMA",
    "FUNDAMENTAL_TAINT_REPORT_SCHEMA",
    "FundamentalTaintError",
    "analyze_deferred_fundamental_taints",
    "validate_taint_analysis_result",
]
