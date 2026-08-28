from __future__ import annotations

import hashlib

import pandas as pd

from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence._common import artifact_ref, build_artifact, business_identity
from quant_investor.intelligence.daily_evidence import (
    build_company_source_evidence,
    build_fundamental_assessments_from_frame,
    build_source_bound_economic_exposure_projection,
    theme_assessment_from_exposure,
)
from quant_investor.intelligence.storage import approved_theme_policy_v2

NOW = "2026-08-27T14:20:51Z"
COMPANY = "688092.SH"
OTHER = "600000.SH"
THEME = "TUSHARE_TDX:880948.TDX"


def _theme_projection() -> dict:
    policy = approved_theme_policy_v2()
    companies = [COMPANY, OTHER]
    return build_artifact(
        kind="theme_membership_projection",
        identity_field="projection_id",
        identity=business_identity(
            kind="theme_membership_projection",
            identity_inputs={"test": "source-bound-evidence"},
        ),
        created_at=NOW,
        fields={
            "as_of": NOW,
            "blocker_codes": [],
            "company_rows": [
                {
                    "company_code": COMPANY,
                    "provider": "TUSHARE_TDX",
                    "status": "MEMBERSHIP_ONLY",
                    "technology_theme_ids": [THEME],
                    "theme_ids": [THEME],
                },
                {
                    "company_code": OTHER,
                    "provider": "TUSHARE_TDX",
                    "status": "NO_MEMBERSHIP",
                    "technology_theme_ids": [],
                    "theme_ids": [],
                },
            ],
            "company_set_sha256": hashlib.sha256(
                canonical_json_bytes(sorted(companies))
            ).hexdigest(),
            "fallback_company_keyset": companies,
            "policy_ref": artifact_ref(policy),
            "source_refs": [],
            "status": "READY",
            "trade_date": "20260827",
        },
    )


def _exposure(share: str) -> dict:
    return build_company_source_evidence(
        company=COMPANY,
        source_type="COMPANY_ANNOUNCEMENT",
        source_path="data/private/company/688092.pdf",
        source_sha256="a" * 64,
        available_at="2025-12-24T00:00:00Z",
        metrics={"primary_theme_id": THEME, "theme_revenue_share": share},
        source_page=9,
        created_at=NOW,
    )


def test_revenue_share_policy_derives_low_medium_and_high() -> None:
    expected = (("0.017", "LOW"), ("0.10", "MEDIUM"), ("0.30", "HIGH"))
    for share, state in expected:
        projection, evidence = build_source_bound_economic_exposure_projection(
            as_of=NOW,
            daily_policy=approved_theme_policy_v2(),
            theme_projection=_theme_projection(),
            evidence=[_exposure(share)],
        )
        rows = {row["company_code"]: row for row in projection["payload"]["company_rows"]}
        assert rows[COMPANY]["economic_exposure_state"] == state
        assert len(rows[COMPANY]["evidence_refs"]) == 1
        assert rows[OTHER]["economic_exposure_state"] == "UNVERIFIED"
        assert evidence[COMPANY]["payload"]["source_page"] == 9


def test_low_exposure_builds_available_theme_with_hard_veto() -> None:
    projection, evidence = build_source_bound_economic_exposure_projection(
        as_of=NOW,
        daily_policy=approved_theme_policy_v2(),
        theme_projection=_theme_projection(),
        evidence=[_exposure("0.017")],
    )
    row = next(
        row for row in projection["payload"]["company_rows"] if row["company_code"] == COMPANY
    )
    assessment = theme_assessment_from_exposure(
        row=row,
        evidence=evidence[COMPANY],
        as_of=NOW,
    )

    assert assessment["payload"]["status"] == "AVAILABLE"
    assert assessment["payload"]["hard_veto_codes"] == ["LOW_ECONOMIC_EXPOSURE"]
    assert assessment["payload"]["component_score"] == "0.017000000000"


def test_fundamental_snapshot_builds_equal_weight_partial_assessment() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": COMPANY,
                "trade_date": "20260814",
                "fin_roe": 0.15,
                "fin_roa": 0.08,
                "fin_debt_to_assets": 0.30,
                "fin_net_profit_yoy": 0.20,
                "fin_ocf_to_profit": 1.10,
                "fin_fcf_to_profit": 0.90,
                "fcf_to_price": 0.04,
                "forecast_revision": 0.05,
            },
            {
                "ts_code": OTHER,
                "trade_date": "20260814",
                "fin_roe": 0.05,
                "fin_roa": 0.02,
                "fin_debt_to_assets": 0.60,
                "fin_net_profit_yoy": -0.10,
                "fin_ocf_to_profit": 0.50,
                "fin_fcf_to_profit": 0.20,
                "fcf_to_price": 0.01,
                "forecast_revision": -0.02,
            },
        ]
    )
    assessments, sources = build_fundamental_assessments_from_frame(
        frame=frame,
        companies=[COMPANY],
        source_path="data/parquet/cn/fundamental_daily.parquet",
        source_sha256="b" * 64,
        source_available_at="2026-08-14T15:00:00Z",
        as_of=NOW,
        industry_assessments={},
        theme_assessments={},
    )

    assessment = assessments[COMPANY]["payload"]
    assert assessment["status"] == "PARTIAL"
    assert assessment["coverage"] == "0.800000000000"
    assert assessment["minimum_coverage"] == "0.600000000000"
    assert len(assessment["source_refs"]) == 1
    assert sources[0]["payload"]["source_type"] == "FUNDAMENTAL_SNAPSHOT"
