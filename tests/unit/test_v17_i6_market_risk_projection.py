from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib

import pytest

from quant_investor.intelligence_v2._core import seal
from quant_investor.intelligence_v2.portfolio import (
    PortfolioContractError,
    build_market_risk_projection,
    build_portfolio_risk_policy,
    project_portfolio_limits,
    validate_market_risk_projection,
)

AS_OF = "2026-08-07T12:00:00Z"
SESSION = "20260807"


def content_ref(name: str) -> dict[str, str]:
    digest = hashlib.sha256(name.encode()).hexdigest()
    return {
        "artifact_id": name,
        "artifact_version": f"myquant.test.{name}.v1",
        "byte_sha256": digest,
        "semantic_sha256": digest,
    }


def source_ref(name: str, *, cutoff: str = "2026-08-07T10:00:00Z") -> dict[str, str]:
    return {
        **content_ref(name),
        "available_at": cutoff,
        "cutoff": cutoff,
        "relative_path": f"data/private/{name}.parquet",
    }


def policy() -> dict:
    return build_portfolio_risk_policy(
        created_at=AS_OF,
        target_positions=10,
        target_gross="0.60",
        cash_floor="0.40",
        per_security_cap="0.20",
        industry_cap="0.30",
        theme_cap="0.25",
        max_adv_participation="0.10",
        turnover_cap="0.50",
        weight_quantum="0.01",
        drawdown_threshold="0.20",
        risk_threshold="0.80",
        hard_veto_codes=["BASE_VETO"],
        macro_regime_rules=[
            {
                "cash_floor": "0.40",
                "gross_cap": "0.60",
                "regime": "NORMAL",
                "risk_multiplier": "1",
                "veto_codes": ["BASE_VETO"],
            }
        ],
        fundamental_staleness_allowance_sessions=1,
        paper_execution_policy_ref=content_ref("paper-policy"),
        graduation_policy_ref=content_ref("graduation-policy"),
    )


def inputs() -> list[dict]:
    return [
        {
            "freshness": "FRESH",
            "kind": kind,
            "source_ref": source_ref(kind.lower()),
            "status": "COMPLETE",
            "target_session": SESSION,
        }
        for kind in (
            "CANONICAL_DAILY",
            "CANONICAL_DAILY_BASIC",
            "CANONICAL_LIMIT",
            "CANONICAL_SUSPEND",
        )
    ]


def build_projection(**changes: object) -> dict:
    arguments = {
        "portfolio_policy": policy(),
        "target_session": SESSION,
        "input_rows": inputs(),
        "projected_gross_cap": "0.50",
        "projected_cash_floor": "0.50",
        "projected_security_cap": "0.15",
        "projected_veto_codes": ["MARKET_VETO"],
        "as_of": AS_OF,
    }
    arguments.update(changes)
    return build_market_risk_projection(**arguments)


def test_available_projection_is_replayable_and_only_tightens() -> None:
    projection = build_projection()
    assert projection["status"] == "AVAILABLE"
    assert projection["effective_gross_cap"] == "0.500000000000"
    assert projection["effective_cash_floor"] == "0.500000000000"
    assert projection["effective_security_cap"] == "0.150000000000"
    assert projection["effective_veto_codes"] == ["BASE_VETO", "MARKET_VETO"]
    assert validate_market_risk_projection(projection, portfolio_policy=policy()) == projection
    limits = project_portfolio_limits(
        portfolio_policy=policy(),
        market_risk_projection=projection,
    )
    assert limits == {
        "cash_floor": "0.500000000000",
        "gross_cap": "0.500000000000",
        "industry_cap": "0.300000000000",
        "security_cap": "0.150000000000",
        "theme_cap": "0.250000000000",
        "veto_codes": ["BASE_VETO", "MARKET_VETO"],
    }


@pytest.mark.parametrize(
    ("gross", "cash", "security"),
    [
        ("1", "0", "1"),
        ("0.60", "0.40", "0.20"),
        ("0.20", "0.70", "0.05"),
    ],
)
def test_min_max_union_formula_never_relaxes_owner_policy(
    gross: str, cash: str, security: str
) -> None:
    projection = build_projection(
        projected_gross_cap=gross,
        projected_cash_floor=cash,
        projected_security_cap=security,
        projected_veto_codes=[],
    )
    assert Decimal(projection["effective_gross_cap"]) <= Decimal(policy()["target_gross"])
    assert Decimal(projection["effective_cash_floor"]) >= Decimal(policy()["cash_floor"])
    assert Decimal(projection["effective_security_cap"]) <= Decimal(policy()["per_security_cap"])
    assert set(policy()["hard_veto_codes"]).issubset(projection["effective_veto_codes"])


def test_missing_stale_or_incomplete_core_input_is_blocked() -> None:
    missing = inputs()
    missing[1] = {
        "freshness": "UNAVAILABLE",
        "kind": "CANONICAL_DAILY_BASIC",
        "source_ref": None,
        "status": "MISSING",
        "target_session": SESSION,
    }
    stale = inputs()
    stale[2]["freshness"] = "STALE"
    incomplete = inputs()
    incomplete[3]["status"] = "INCOMPLETE"
    incomplete[3]["freshness"] = "UNAVAILABLE"
    for rows in (missing, stale, incomplete):
        projection = build_projection(input_rows=rows)
        assert projection["status"] == "BLOCKED"
        assert projection["blocker_codes"]
        with pytest.raises(PortfolioContractError, match="blocked"):
            project_portfolio_limits(
                portfolio_policy=policy(),
                market_risk_projection=projection,
            )


def test_diagnostic_or_extra_input_key_is_rejected_at_schema_boundary() -> None:
    diagnostic = inputs()
    diagnostic[-1]["kind"] = "CYQ_PERF"
    with pytest.raises(PortfolioContractError, match="kind"):
        build_projection(input_rows=diagnostic)
    extra = inputs()
    extra[0]["moneyflow"] = source_ref("moneyflow")
    with pytest.raises(PortfolioContractError, match="shape"):
        build_projection(input_rows=extra)


def test_cross_session_future_and_resealed_projection_forgery_are_rejected() -> None:
    wrong_session = inputs()
    wrong_session[0]["source_ref"] = source_ref(
        "wrong-session",
        cutoff="2026-08-06T10:00:00Z",
    )
    with pytest.raises(PortfolioContractError, match="another session"):
        build_projection(input_rows=wrong_session)

    future = inputs()
    future[0]["source_ref"] = source_ref("future", cutoff="2026-08-07T13:00:00Z")
    with pytest.raises(PortfolioContractError, match="future"):
        build_projection(input_rows=future)

    projection = build_projection()
    forged = deepcopy(projection)
    forged.pop("projection_id")
    forged.pop("semantic_sha256")
    forged["effective_gross_cap"] = "0.600000000000"
    forged = seal(forged, identity_field="projection_id")
    with pytest.raises(PortfolioContractError, match="replay mismatch"):
        validate_market_risk_projection(forged, portfolio_policy=policy())
