"""Second-stage paper capital-TV gate; pure and replay validated."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Final

from .contracts import (
    PortfolioContractError,
    company_code,
    decimal_in_unit,
    decimal_text,
    portfolio_common,
    require_exact_keys,
    seal,
    timestamp,
    validate_seal,
)

PAPER_CAPITAL_GATE_VERSION: Final = "myquant.v17.intelligence-v2.paper-capital-gate-receipt.v1"
CAPITAL_TV_LIMIT: Final = Decimal("0.10")
GATE_FIELDS: Final = {
    "authority",
    "baseline_targets",
    "capital_tv",
    "decision_protocol",
    "final_targets",
    "gate_id",
    "provisional_targets",
    "production",
    "reason_codes",
    "research_only",
    "semantic_sha256",
    "status",
    "timestamp",
    "version",
}


def _targets(values: Mapping[str, Any], *, label: str) -> list[dict[str, str]]:
    if type(values) is not dict or "CASH" not in values:
        raise PortfolioContractError(f"{label} must include CASH")
    rows = []
    total = Decimal("0")
    for key, raw_weight in values.items():
        code = "CASH" if key == "CASH" else company_code(key, label=f"{label}.company")
        weight = decimal_in_unit(raw_weight, label=f"{label}.{code}")
        total += weight
        rows.append({"asset_id": code, "weight": decimal_text(weight)})
    if total != Decimal("1"):
        raise PortfolioContractError(f"{label} weights must sum to one")
    rows.sort(key=lambda row: row["asset_id"].encode("ascii"))
    if len({row["asset_id"] for row in rows}) != len(rows):
        raise PortfolioContractError(f"{label} contains duplicate assets")
    return rows


def _weights(rows: list[dict[str, str]]) -> dict[str, Decimal]:
    return {row["asset_id"]: Decimal(row["weight"]) for row in rows}


def build_paper_capital_gate(
    *,
    baseline_targets: Mapping[str, Any],
    provisional_targets: Mapping[str, Any],
    evaluated_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(evaluated_at, label="evaluated_at")
    baseline = _targets(baseline_targets, label="baseline_targets")
    provisional = _targets(provisional_targets, label="provisional_targets")
    left = _weights(baseline)
    right = _weights(provisional)
    assets = set(left) | set(right)
    capital_tv = sum(
        (abs(left.get(asset, Decimal("0")) - right.get(asset, Decimal("0"))) for asset in assets),
        Decimal("0"),
    ) / Decimal("2")
    accepted = capital_tv <= CAPITAL_TV_LIMIT
    final = provisional if accepted else baseline
    return seal(
        {
            **portfolio_common(at=issued_at),
            "baseline_targets": baseline,
            "capital_tv": decimal_text(capital_tv),
            "final_targets": final,
            "provisional_targets": provisional,
            "reason_codes": [] if accepted else ["ADVISORY_CAPITAL_LIMIT_REJECTED"],
            "status": "ACCEPTED" if accepted else "REJECTED",
            "version": PAPER_CAPITAL_GATE_VERSION,
        },
        identity_field="gate_id",
    )


def validate_paper_capital_gate(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="gate_id")
    require_exact_keys(row, GATE_FIELDS, label="paper capital gate")
    expected = build_paper_capital_gate(**closure)
    if row != expected or row["version"] != PAPER_CAPITAL_GATE_VERSION:
        raise PortfolioContractError("paper capital gate replay mismatch")
    return row


__all__ = [
    "CAPITAL_TV_LIMIT",
    "PAPER_CAPITAL_GATE_VERSION",
    "build_paper_capital_gate",
    "validate_paper_capital_gate",
]
