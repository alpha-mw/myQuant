"""Source-bound, monotonic-only market risk projection for I6 research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .contracts import (
    PortfolioContractError,
    content_ref,
    decimal_in_unit,
    decimal_text,
    exact_source_ref,
    portfolio_common,
    require_exact_keys,
    seal,
    sorted_codes,
    timestamp,
    validate_seal,
)
from .policies import validate_portfolio_risk_policy

MARKET_RISK_PROJECTION_VERSION: Final = "myquant.v17.intelligence-v2.market-risk-projection.v1"
MARKET_INPUT_KINDS: Final = (
    "CANONICAL_DAILY",
    "CANONICAL_DAILY_BASIC",
    "CANONICAL_LIMIT",
    "CANONICAL_SUSPEND",
)
_INPUT_FIELDS: Final = {"freshness", "kind", "source_ref", "status", "target_session"}
_PROJECTION_FIELDS: Final = {
    "authority",
    "blocker_codes",
    "decision_protocol",
    "effective_cash_floor",
    "effective_gross_cap",
    "effective_security_cap",
    "effective_veto_codes",
    "input_rows",
    "policy_ref",
    "production",
    "projected_cash_floor",
    "projected_gross_cap",
    "projected_security_cap",
    "projected_veto_codes",
    "projection_id",
    "research_only",
    "semantic_sha256",
    "status",
    "target_session",
    "timestamp",
    "version",
}


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 8 or not value.isdigit():
        raise PortfolioContractError(f"{label} must be YYYYMMDD")
    return value


def _exact_keys(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    try:
        return require_exact_keys(value, fields, label=label)
    except PortfolioContractError:
        raise
    except Exception as exc:
        raise PortfolioContractError(f"{label} shape is invalid") from exc


def _input_state(
    row: Mapping[str, Any],
    *,
    kind: str,
    target_session: str,
    as_of: str,
) -> tuple[str, str, dict[str, str] | None, list[str]]:
    status = row["status"]
    freshness = row["freshness"]
    if status not in {"COMPLETE", "INCOMPLETE", "MISSING"}:
        raise PortfolioContractError("market input status is invalid")
    if freshness not in {"FRESH", "STALE", "UNAVAILABLE"}:
        raise PortfolioContractError("market input freshness is invalid")
    source = row["source_ref"]
    if status == "MISSING":
        if source is not None or freshness != "UNAVAILABLE":
            raise PortfolioContractError("missing market input must have no source ref")
        reference = None
    else:
        if type(source) is not dict:
            raise PortfolioContractError("market input requires an exact source ref")
        reference = exact_source_ref(source, label=f"{kind}.source_ref", as_of=as_of)
        if reference["cutoff"][:10].replace("-", "") != target_session:
            raise PortfolioContractError("market input source cutoff is from another session")
    blockers: list[str] = []
    if status != "COMPLETE":
        blockers.append(f"MARKET_RISK_{kind}_{status}")
    elif freshness != "FRESH":
        blockers.append(f"MARKET_RISK_{kind}_{freshness}")
    return status, freshness, reference, blockers


def _input_rows(
    values: Sequence[Mapping[str, Any]],
    *,
    target_session: str,
    as_of: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PortfolioContractError("market input rows must be a sequence")
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index, value in enumerate(values):
        row = _exact_keys(value, _INPUT_FIELDS, label=f"input_rows[{index}]")
        kind = row["kind"]
        if kind not in MARKET_INPUT_KINDS or row["target_session"] != target_session:
            raise PortfolioContractError("market input kind or target session is invalid")
        status, freshness, reference, row_blockers = _input_state(
            row,
            kind=kind,
            target_session=target_session,
            as_of=as_of,
        )
        blockers.extend(row_blockers)
        rows.append(
            {
                "freshness": freshness,
                "kind": kind,
                "source_ref": reference,
                "status": status,
                "target_session": target_session,
            }
        )
    kinds = [row["kind"] for row in rows]
    if tuple(kinds) != MARKET_INPUT_KINDS:
        raise PortfolioContractError("market input keyset must be exact and ASCII ordered")
    return rows, sorted(blockers, key=lambda item: item.encode("ascii"))


def _effective_projection(
    *,
    policy: Mapping[str, Any],
    gross_cap: Decimal,
    cash_floor: Decimal,
    security_cap: Decimal,
    veto_codes: Sequence[str],
) -> dict[str, Any]:
    effective_gross = min(Decimal(policy["target_gross"]), gross_cap)
    effective_cash = max(Decimal(policy["cash_floor"]), cash_floor)
    effective_security = min(Decimal(policy["per_security_cap"]), security_cap)
    effective_gross = min(effective_gross, Decimal("1") - effective_cash)
    effective_vetoes = sorted(
        set(policy["hard_veto_codes"]) | set(veto_codes),
        key=lambda item: item.encode("ascii"),
    )
    return {
        "effective_cash_floor": decimal_text(effective_cash),
        "effective_gross_cap": decimal_text(effective_gross),
        "effective_security_cap": decimal_text(effective_security),
        "effective_veto_codes": effective_vetoes,
    }


def build_market_risk_projection(
    *,
    portfolio_policy: Mapping[str, Any],
    target_session: str,
    input_rows: Sequence[Mapping[str, Any]],
    projected_gross_cap: Any,
    projected_cash_floor: Any,
    projected_security_cap: Any,
    projected_veto_codes: Sequence[Any],
    as_of: str,
) -> dict[str, Any]:
    policy = validate_portfolio_risk_policy(portfolio_policy)
    issued_at = timestamp(as_of, label="as_of")
    if policy["timestamp"] != issued_at:
        raise PortfolioContractError("market projection must share the portfolio policy cutoff")
    session = _session(target_session, label="target_session")
    if issued_at[:10].replace("-", "") != session:
        raise PortfolioContractError("market projection as_of is outside target session")
    inputs, blockers = _input_rows(input_rows, target_session=session, as_of=issued_at)
    gross = decimal_in_unit(projected_gross_cap, label="projected_gross_cap")
    cash = decimal_in_unit(projected_cash_floor, label="projected_cash_floor")
    security = decimal_in_unit(projected_security_cap, label="projected_security_cap")
    vetoes = sorted_codes(
        projected_veto_codes,
        label="projected_veto_codes",
        allow_empty=True,
    )
    effective = _effective_projection(
        policy=policy,
        gross_cap=gross,
        cash_floor=cash,
        security_cap=security,
        veto_codes=vetoes,
    )
    return seal(
        {
            **portfolio_common(at=issued_at),
            "blocker_codes": blockers,
            **effective,
            "input_rows": inputs,
            "policy_ref": content_ref(policy, identity_field="policy_id"),
            "projected_cash_floor": decimal_text(cash),
            "projected_gross_cap": decimal_text(gross),
            "projected_security_cap": decimal_text(security),
            "projected_veto_codes": vetoes,
            "status": "AVAILABLE" if not blockers else "BLOCKED",
            "target_session": session,
            "version": MARKET_RISK_PROJECTION_VERSION,
        },
        identity_field="projection_id",
    )


def validate_market_risk_projection(
    document: Mapping[str, Any],
    *,
    portfolio_policy: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="projection_id")
    _exact_keys(normalized, _PROJECTION_FIELDS, label="market risk projection")
    expected = build_market_risk_projection(
        portfolio_policy=portfolio_policy,
        target_session=normalized["target_session"],
        input_rows=normalized["input_rows"],
        projected_gross_cap=normalized["projected_gross_cap"],
        projected_cash_floor=normalized["projected_cash_floor"],
        projected_security_cap=normalized["projected_security_cap"],
        projected_veto_codes=normalized["projected_veto_codes"],
        as_of=normalized["timestamp"],
    )
    if normalized != expected or normalized["version"] != MARKET_RISK_PROJECTION_VERSION:
        raise PortfolioContractError("market risk projection replay mismatch")
    return normalized


def project_portfolio_limits(
    *,
    portfolio_policy: Mapping[str, Any],
    market_risk_projection: Mapping[str, Any],
) -> dict[str, Any]:
    policy = validate_portfolio_risk_policy(portfolio_policy)
    projection = validate_market_risk_projection(
        market_risk_projection,
        portfolio_policy=policy,
    )
    if projection["status"] != "AVAILABLE":
        raise PortfolioContractError("blocked market risk projection cannot constrain a portfolio")
    return {
        "cash_floor": projection["effective_cash_floor"],
        "gross_cap": projection["effective_gross_cap"],
        "industry_cap": policy["industry_cap"],
        "security_cap": projection["effective_security_cap"],
        "theme_cap": policy["theme_cap"],
        "veto_codes": projection["effective_veto_codes"],
    }


__all__ = [
    "MARKET_INPUT_KINDS",
    "MARKET_RISK_PROJECTION_VERSION",
    "build_market_risk_projection",
    "project_portfolio_limits",
    "validate_market_risk_projection",
]
