"""Deterministic investment-data readiness receipts for B0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from ._core import (
    IntelligenceV2ContractError,
    common_fields,
    content_ref,
    exact_ref,
    require_exact_keys,
    require_no_future,
    seal,
    session_date,
    timestamp,
    validate_seal,
)

READINESS_POLICY_VERSION: Final = "myquant.v17.intelligence-v2.readiness-policy.v1"
READINESS_RECEIPT_VERSION: Final = (
    "myquant.v17.intelligence-v2.investment-data-readiness-receipt.v1"
)
READINESS_STATUSES: Final = {"AVAILABLE", "STALE", "BLOCKED"}
POLICY_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "fundamental_max_stale_sessions",
    "macro_block_after_seconds",
    "macro_stale_after_seconds",
    "policy_id",
    "production",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "version",
}
RECEIPT_FIELDS: Final = {
    "as_of",
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "market_data_cutoff",
    "overall_status",
    "policy_ref",
    "production",
    "quant_inputs_ready",
    "readiness_id",
    "research_only",
    "rows",
    "semantic_sha256",
    "target_trade_session",
    "timestamp",
    "version",
}


def build_readiness_policy(
    *,
    created_at: str,
    fundamental_max_stale_sessions: int,
    macro_stale_after_seconds: int,
    macro_block_after_seconds: int,
) -> dict[str, Any]:
    if (
        type(fundamental_max_stale_sessions) is not int
        or not 0 <= fundamental_max_stale_sessions <= 1
    ):
        raise IntelligenceV2ContractError("fundamental_max_stale_sessions must be zero or one")
    if (
        type(macro_stale_after_seconds) is not int
        or type(macro_block_after_seconds) is not int
        or not 0 <= macro_stale_after_seconds <= macro_block_after_seconds <= 31_536_000
    ):
        raise IntelligenceV2ContractError("macro age thresholds are invalid")
    return seal(
        {
            **common_fields(timestamp_value=created_at),
            "fundamental_max_stale_sessions": fundamental_max_stale_sessions,
            "macro_block_after_seconds": macro_block_after_seconds,
            "macro_stale_after_seconds": macro_stale_after_seconds,
            "version": READINESS_POLICY_VERSION,
        },
        identity_field="policy_id",
    )


def validate_readiness_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="policy_id")
    require_exact_keys(normalized, POLICY_FIELDS, label="readiness policy")
    expected = build_readiness_policy(
        created_at=normalized["timestamp"],
        fundamental_max_stale_sessions=normalized["fundamental_max_stale_sessions"],
        macro_stale_after_seconds=normalized["macro_stale_after_seconds"],
        macro_block_after_seconds=normalized["macro_block_after_seconds"],
    )
    if normalized != expected or normalized["version"] != READINESS_POLICY_VERSION:
        raise IntelligenceV2ContractError("readiness policy replay mismatch")
    return normalized


def _sessions(values: Sequence[Any], *, target: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceV2ContractError("open_sessions must be a sequence")
    rows = [
        session_date(value, label=f"open_sessions[{index}]") for index, value in enumerate(values)
    ]
    if rows != sorted(rows) or len(rows) != len(set(rows)) or target not in rows:
        raise IntelligenceV2ContractError("open_sessions must be sorted, unique and contain target")
    return rows


def _source_row(
    *,
    name: str,
    status: str,
    session: str | None,
    source_ref: Mapping[str, Any],
    blocker_codes: list[str],
) -> dict[str, Any]:
    if status not in READINESS_STATUSES:
        raise IntelligenceV2ContractError("readiness row status is invalid")
    return {
        "blocker_codes": blocker_codes,
        "name": name,
        "session": session,
        "source_ref": exact_ref(source_ref, label=f"{name}.source_ref"),
        "status": status,
    }


def _validated_source_refs(
    *,
    market_ref: Mapping[str, Any],
    pit_ref: Mapping[str, Any],
    fundamental_ref: Mapping[str, Any],
    macro_ref: Mapping[str, Any],
    macro_release_calendar_ref: Mapping[str, Any],
    cutoff: str,
    issued_at: str,
) -> dict[str, dict[str, str]]:
    references = {
        "MARKET": exact_ref(market_ref, label="market_ref"),
        "PIT_UNIVERSE": exact_ref(pit_ref, label="pit_ref"),
        "FUNDAMENTAL": exact_ref(fundamental_ref, label="fundamental_ref"),
        "MACRO": exact_ref(macro_ref, label="macro_ref"),
        "MACRO_RELEASE_CALENDAR": exact_ref(
            macro_release_calendar_ref,
            label="macro_release_calendar_ref",
        ),
    }
    for name, reference in references.items():
        require_no_future(
            available_at=reference["available_at"],
            as_of=issued_at,
            label=name,
        )
        if reference["cutoff"] > cutoff:
            raise IntelligenceV2ContractError(f"{name} source cutoff exceeds market cutoff")
    return references


def _fundamental_status(
    *,
    fundamental_session: str,
    target: str,
    sessions: Sequence[str],
    target_index: int,
    max_stale_sessions: int,
) -> tuple[str, list[str]]:
    if fundamental_session == target:
        return "AVAILABLE", []
    if target_index > 0 and fundamental_session == sessions[target_index - 1]:
        if max_stale_sessions == 1:
            return "STALE", ["FUNDAMENTAL_ONE_SESSION_STALE"]
        return "BLOCKED", ["FUNDAMENTAL_STALENESS_NOT_ALLOWED"]
    return "BLOCKED", ["FUNDAMENTAL_SESSION_LAG_EXCEEDED"]


def _macro_row(
    *,
    observed_at: str,
    latest_expected_release_at: str,
    cutoff: str,
    policy: Mapping[str, Any],
    macro_ref: Mapping[str, Any],
    calendar_ref: Mapping[str, Any],
) -> dict[str, Any]:
    observed = timestamp(observed_at, label="macro_observed_at")
    expected_release = timestamp(
        latest_expected_release_at,
        label="macro_latest_expected_release_at",
    )
    if observed > cutoff or expected_release > cutoff:
        raise IntelligenceV2ContractError("macro chronology exceeds market cutoff")
    cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    observed_dt = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    macro_age = int((cutoff_dt - observed_dt).total_seconds())
    if observed < expected_release:
        status, blockers = "BLOCKED", ["MACRO_EXPECTED_RELEASE_MISSING"]
    elif macro_age <= policy["macro_stale_after_seconds"]:
        status, blockers = "AVAILABLE", []
    elif macro_age <= policy["macro_block_after_seconds"]:
        status, blockers = "STALE", ["MACRO_STALE"]
    else:
        status, blockers = "BLOCKED", ["MACRO_AGE_EXCEEDED"]
    return {
        **_source_row(
            name="MACRO",
            status=status,
            session=None,
            source_ref=macro_ref,
            blocker_codes=blockers,
        ),
        "latest_expected_release_at": expected_release,
        "observed_at": observed,
        "release_calendar_ref": dict(calendar_ref),
    }


def build_investment_data_readiness(
    *,
    policy: Mapping[str, Any],
    target_trade_session: str,
    market_data_cutoff: str,
    open_sessions: Sequence[str],
    market_session: str,
    market_ref: Mapping[str, Any],
    pit_session: str,
    pit_ref: Mapping[str, Any],
    fundamental_session: str,
    fundamental_ref: Mapping[str, Any],
    macro_observed_at: str,
    macro_latest_expected_release_at: str,
    macro_ref: Mapping[str, Any],
    macro_release_calendar_ref: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    validated_policy = validate_readiness_policy(policy)
    target = session_date(target_trade_session, label="target_trade_session")
    cutoff = timestamp(market_data_cutoff, label="market_data_cutoff")
    issued_at = timestamp(as_of, label="as_of")
    if cutoff > issued_at:
        raise IntelligenceV2ContractError("market_data_cutoff exceeds as_of")
    sessions = _sessions(open_sessions, target=target)
    target_index = sessions.index(target)

    market_exact = session_date(market_session, label="market_session")
    pit_exact = session_date(pit_session, label="pit_session")
    fundamental_exact = session_date(fundamental_session, label="fundamental_session")
    references = _validated_source_refs(
        market_ref=market_ref,
        pit_ref=pit_ref,
        fundamental_ref=fundamental_ref,
        macro_ref=macro_ref,
        macro_release_calendar_ref=macro_release_calendar_ref,
        cutoff=cutoff,
        issued_at=issued_at,
    )

    rows: list[dict[str, Any]] = []
    rows.append(
        _source_row(
            name="MARKET",
            status="AVAILABLE" if market_exact == target else "BLOCKED",
            session=market_exact,
            source_ref=references["MARKET"],
            blocker_codes=[] if market_exact == target else ["MARKET_SESSION_MISMATCH"],
        )
    )
    rows.append(
        _source_row(
            name="PIT_UNIVERSE",
            status="AVAILABLE" if pit_exact == target else "BLOCKED",
            session=pit_exact,
            source_ref=references["PIT_UNIVERSE"],
            blocker_codes=[] if pit_exact == target else ["PIT_SESSION_MISMATCH"],
        )
    )

    fundamental_status, fundamental_blockers = _fundamental_status(
        fundamental_session=fundamental_exact,
        target=target,
        sessions=sessions,
        target_index=target_index,
        max_stale_sessions=validated_policy["fundamental_max_stale_sessions"],
    )
    rows.append(
        _source_row(
            name="FUNDAMENTAL",
            status=fundamental_status,
            session=fundamental_exact,
            source_ref=references["FUNDAMENTAL"],
            blocker_codes=fundamental_blockers,
        )
    )

    rows.append(
        _macro_row(
            observed_at=macro_observed_at,
            latest_expected_release_at=macro_latest_expected_release_at,
            cutoff=cutoff,
            policy=validated_policy,
            macro_ref=references["MACRO"],
            calendar_ref=references["MACRO_RELEASE_CALENDAR"],
        )
    )

    status_values = {row["status"] for row in rows}
    overall = (
        "BLOCKED"
        if "BLOCKED" in status_values
        else ("STALE" if "STALE" in status_values else "AVAILABLE")
    )
    quant_ready = all(
        row["status"] == "AVAILABLE" for row in rows if row["name"] in {"MARKET", "PIT_UNIVERSE"}
    )
    return seal(
        {
            **common_fields(timestamp_value=issued_at),
            "as_of": issued_at,
            "market_data_cutoff": cutoff,
            "overall_status": overall,
            "policy_ref": content_ref(validated_policy, identity_field="policy_id"),
            "quant_inputs_ready": quant_ready,
            "rows": rows,
            "target_trade_session": target,
            "version": READINESS_RECEIPT_VERSION,
        },
        identity_field="readiness_id",
    )


def validate_investment_data_readiness(
    document: Mapping[str, Any],
    **closure: Any,
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="readiness_id")
    require_exact_keys(normalized, RECEIPT_FIELDS, label="readiness receipt")
    expected = build_investment_data_readiness(**closure)
    if normalized != expected or normalized["version"] != READINESS_RECEIPT_VERSION:
        raise IntelligenceV2ContractError("readiness receipt replay mismatch")
    return normalized


__all__ = [
    "READINESS_POLICY_VERSION",
    "READINESS_RECEIPT_VERSION",
    "READINESS_STATUSES",
    "build_investment_data_readiness",
    "build_readiness_policy",
    "validate_investment_data_readiness",
    "validate_readiness_policy",
]
