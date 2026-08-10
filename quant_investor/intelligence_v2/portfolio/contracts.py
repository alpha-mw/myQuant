"""Closed, research-only primitives for the I6 portfolio layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_FLOOR, localcontext
import re
from typing import Any, Final

from .._core import (
    IntelligenceV2ContractError,
    canonical_bytes,
    content_ref,
    decimal_text,
    decimal_value,
    exact_ref,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_content_ref,
    validate_seal,
)

PORTFOLIO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "factor_governance_write": False,
    "holdings_write": False,
    "llm": False,
    "mainline_authority": False,
    "order": False,
    "paper_only": True,
    "portfolio_mutation": False,
    "production": False,
    "provider": False,
    "research_only": True,
    "selector": False,
    "trade": False,
}

SUBJECT_FIELDS: Final = {
    "advisory_percentile",
    "adv_weight_capacity",
    "company_code",
    "decision_receipt",
    "decision_validation_closure",
    "deterministic_percentile",
    "drawdown",
    "fundamental_age_sessions",
    "hard_veto_codes",
    "industry_code",
    "industry_ref",
    "liquidity_ref",
    "risk_score",
    "security_ref",
    "theme_codes",
    "theme_refs",
}

CURRENT_POSITION_FIELDS: Final = {
    "adv_weight_capacity",
    "company_code",
    "current_weight",
    "industry_code",
    "industry_ref",
    "liquidity_ref",
    "security_ref",
    "theme_codes",
    "theme_refs",
}

TARGET_FIELDS: Final = {
    "company_code",
    "current_weight",
    "final_weight",
    "industry_code",
    "liquidity_cap",
    "theme_codes",
}

_COMPANY_RE: Final = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)


class PortfolioContractError(IntelligenceV2ContractError):
    """Stable fail-closed error for research portfolio contracts."""


def portfolio_common(*, at: str) -> dict[str, Any]:
    return {
        "authority": dict(PORTFOLIO_AUTHORITY),
        "decision_protocol": "myquant.v17.v4",
        "production": False,
        "research_only": True,
        "timestamp": timestamp(at, label="timestamp"),
    }


def company_code(value: Any, *, label: str) -> str:
    if type(value) is not str or _COMPANY_RE.fullmatch(value) is None:
        raise PortfolioContractError(f"{label} must be a canonical A-share code")
    return value


def decimal_in_unit(value: Any, *, label: str) -> Decimal:
    return decimal_value(value, label=label, minimum=Decimal("0"), maximum=Decimal("1"))


def positive_decimal(value: Any, *, label: str, maximum: Decimal | None = None) -> Decimal:
    result = decimal_value(value, label=label, minimum=Decimal("0"), maximum=maximum)
    if result <= 0:
        raise PortfolioContractError(f"{label} must be positive")
    return result


def quantum_floor(value: Decimal, quantum: Decimal) -> Decimal:
    if quantum <= 0:
        raise PortfolioContractError("weight quantum must be positive")
    with localcontext() as context:
        context.prec = 50
        units = (value / quantum).to_integral_value(rounding=ROUND_FLOOR)
        return units * quantum


def require_quantum_multiple(value: Decimal, quantum: Decimal, *, label: str) -> None:
    if quantum_floor(value, quantum) != value:
        raise PortfolioContractError(f"{label} is not a weight-quantum multiple")


def sorted_codes(
    values: Sequence[Any], *, label: str, allow_empty: bool, maximum: int = 64
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PortfolioContractError(f"{label} must be a sequence")
    result = [identifier(item, label=f"{label}[{index}]") for index, item in enumerate(values)]
    if (not allow_empty and not result) or len(result) > maximum or len(result) != len(set(result)):
        raise PortfolioContractError(f"{label} cardinality or uniqueness is invalid")
    expected = sorted(result, key=lambda item: item.encode("ascii"))
    if result != expected:
        raise PortfolioContractError(f"{label} must be ASCII sorted")
    return result


def sorted_content_refs(
    values: Sequence[Mapping[str, Any]], *, label: str, allow_empty: bool = False
) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PortfolioContractError(f"{label} must be a sequence")
    rows = [
        validate_content_ref(item, label=f"{label}[{index}]") for index, item in enumerate(values)
    ]
    keys = [
        (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        )
        for row in rows
    ]
    if (not allow_empty and not rows) or len(rows) > 256 or len(keys) != len(set(keys)):
        raise PortfolioContractError(f"{label} cardinality or uniqueness is invalid")
    if keys != sorted(keys, key=lambda item: tuple(value.encode("utf-8") for value in item)):
        raise PortfolioContractError(f"{label} must be canonically sorted")
    return rows


def exact_source_ref(value: Mapping[str, Any], *, label: str, as_of: str) -> dict[str, str]:
    row = exact_ref(value, label=label)
    if row["available_at"] > as_of or row["cutoff"] > as_of:
        raise PortfolioContractError(f"{label} contains future evidence")
    return row


def target_row(
    *,
    company: str,
    current: Decimal,
    final: Decimal,
    industry: str,
    themes: Sequence[str],
    liquidity_cap: Decimal,
) -> dict[str, Any]:
    return {
        "company_code": company_code(company, label="target.company_code"),
        "current_weight": decimal_text(current),
        "final_weight": decimal_text(final),
        "industry_code": identifier(industry, label="target.industry_code"),
        "liquidity_cap": decimal_text(liquidity_cap),
        "theme_codes": list(themes),
    }


__all__ = [
    "PORTFOLIO_AUTHORITY",
    "CURRENT_POSITION_FIELDS",
    "SUBJECT_FIELDS",
    "TARGET_FIELDS",
    "PortfolioContractError",
    "canonical_bytes",
    "company_code",
    "content_ref",
    "decimal_in_unit",
    "decimal_text",
    "decimal_value",
    "exact_source_ref",
    "portfolio_common",
    "positive_decimal",
    "quantum_floor",
    "require_exact_keys",
    "require_quantum_multiple",
    "seal",
    "sha256",
    "sorted_codes",
    "sorted_content_refs",
    "target_row",
    "timestamp",
    "validate_content_ref",
    "validate_seal",
]
