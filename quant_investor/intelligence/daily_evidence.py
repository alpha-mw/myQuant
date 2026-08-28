"""Source-bound company evidence for daily Theme and Fundamental research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
import math
from pathlib import PurePosixPath
from typing import Any, Final

import pandas as pd

from quant_investor.contracts import canonical_json_bytes

from ._common import (
    IntelligenceError,
    artifact_payload,
    artifact_ref,
    build_artifact,
    business_identity,
    company_code,
    decimal_text,
    decimal_value,
    identifier,
    require_artifact_ref,
    require_no_future,
    sha256,
    timestamp,
)
from .daily import validate_daily_research_policy
from .fundamental import FUNDAMENTAL_COMPONENTS, assess_fundamental
from .theme import assess_theme
from .theme_governance import (
    EXPOSURE_PROJECTION_KIND,
    STRATEGY_ID,
    approved_theme_governance_policy,
)

SOURCE_KIND: Final = "company_source_evidence"
MARKET_RISK_KIND: Final = "market_risk_evidence"
EXPOSURE_SOURCE_TYPES: Final = frozenset(
    {
        "ANNUAL_REPORT",
        "INTERIM_REPORT",
        "COMPANY_ANNOUNCEMENT",
        "INVESTOR_RELATIONS_RECORD",
        "REVENUE_STRUCTURE",
    }
)
FUNDAMENTAL_SOURCE_TYPE: Final = "FUNDAMENTAL_SNAPSHOT"
FUNDAMENTAL_METRICS: Final = (
    "fin_roe",
    "fin_roa",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_fcf_to_profit",
    "fcf_to_price",
    "forecast_revision",
)
FUNDAMENTAL_WEIGHTS: Final = {component: "0.2" for component in FUNDAMENTAL_COMPONENTS}
FUNDAMENTAL_MINIMUM_COVERAGE: Final = "0.6"


def build_market_risk_evidence(
    *,
    source_path: str,
    source_sha256: str,
    blocker_codes: Sequence[str],
    classification: str,
    as_of: str,
) -> dict[str, Any]:
    """Seal approved Macro pipeline-closure state as research risk."""

    cutoff = timestamp(as_of, label="as_of")
    if classification not in {"PIPELINE_DATA_VETO", "CANONICAL_MACRO_READY"}:
        raise IntelligenceError("market risk classification is not approved")
    if type(source_path) is not str or not source_path or source_path.startswith("/"):
        raise IntelligenceError("market risk source path is invalid")
    digest = sha256(source_sha256, label="source_sha256")
    normalized = sorted(
        {identifier(code, label="market risk blocker") for code in blocker_codes},
        key=lambda value: value.encode("ascii"),
    )
    if classification == "PIPELINE_DATA_VETO":
        if normalized != ["MACRO_RELEASE_CONTRACT_BLOCKED"]:
            raise IntelligenceError("market risk blocker closure differs")
        hard_risk_codes = ["MACRO_DATA_VETO_ACTIVE"]
    else:
        if normalized:
            raise IntelligenceError("ready market risk cannot carry blockers")
        source = PurePosixPath(source_path)
        if (
            len(source.parts) != 5
            or source.parts[:3] != ("results", "intelligence", "macro_readiness")
            or source.parts[3].isdigit() is False
            or len(source.parts[3]) != 8
            or source.parts[4] != f"{digest}.json"
        ):
            raise IntelligenceError("ready market risk source path is invalid")
        hard_risk_codes = []
    return build_artifact(
        kind=MARKET_RISK_KIND,
        identity_field="evidence_id",
        identity=business_identity(
            kind=MARKET_RISK_KIND,
            identity_inputs={
                "classification": classification,
                "source_path": source_path,
                "source_sha256": digest,
            },
        ),
        created_at=cutoff,
        fields={
            "blocker_codes": normalized,
            "classification": classification,
            "hard_risk_codes": hard_risk_codes,
            "source_path": source_path,
            "source_sha256": digest,
            "status": "AVAILABLE",
        },
    )


def validate_market_risk_evidence(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact, payload = artifact_payload(value, expected_kind=MARKET_RISK_KIND)
    replay = build_market_risk_evidence(
        source_path=payload["source_path"],
        source_sha256=payload["source_sha256"],
        blocker_codes=payload["blocker_codes"],
        classification=payload["classification"],
        as_of=artifact["created_at"],
    )
    if replay != artifact:
        raise IntelligenceError("market risk evidence does not replay")
    return artifact


def _metric_value(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    parsed = decimal_value(value, label=label)
    return decimal_text(parsed)


def build_company_source_evidence(
    *,
    company: str,
    source_type: str,
    source_path: str,
    source_sha256: str,
    available_at: str,
    metrics: Mapping[str, Any],
    source_page: int | None,
    created_at: str,
) -> dict[str, Any]:
    """Seal one exact source file plus deterministic extracted metrics."""

    code = company_code(company)
    kind = identifier(source_type, label="source_type")
    if kind not in EXPOSURE_SOURCE_TYPES | {FUNDAMENTAL_SOURCE_TYPE}:
        raise IntelligenceError("company evidence source type is not approved")
    if type(source_path) is not str or not source_path or source_path.startswith("/"):
        raise IntelligenceError("company evidence source path must be workspace-relative")
    digest = sha256(source_sha256, label="source_sha256")
    available = timestamp(available_at, label="available_at")
    instant = timestamp(created_at, label="created_at")
    if available > instant:
        raise IntelligenceError("company evidence is future-dated")
    if source_page is not None and (type(source_page) is not int or source_page <= 0):
        raise IntelligenceError("company evidence source page is invalid")
    if type(metrics) is not dict or not metrics:
        raise IntelligenceError("company evidence metrics are empty")
    normalized_metrics: dict[str, str | None] = {}
    for key, value in metrics.items():
        metric = identifier(key, label="metric")
        if metric == "primary_theme_id":
            normalized_metrics[metric] = identifier(value, label="primary_theme_id")
        else:
            normalized_metrics[metric] = _metric_value(value, label=f"metrics.{metric}")
    return build_artifact(
        kind=SOURCE_KIND,
        identity_field="evidence_id",
        identity=business_identity(
            kind=SOURCE_KIND,
            identity_inputs={
                "company_code": code,
                "source_path": source_path,
                "source_sha256": digest,
                "source_type": kind,
            },
        ),
        created_at=instant,
        fields={
            "available_at": available,
            "company_code": code,
            "metrics": normalized_metrics,
            "source_page": source_page,
            "source_path": source_path,
            "source_sha256": digest,
            "source_type": kind,
        },
    )


def validate_company_source_evidence(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact, payload = artifact_payload(value, expected_kind=SOURCE_KIND)
    company_code(payload.get("company_code"))
    source_type = identifier(payload.get("source_type"), label="source_type")
    if source_type not in EXPOSURE_SOURCE_TYPES | {FUNDAMENTAL_SOURCE_TYPE}:
        raise IntelligenceError("company evidence source type is invalid")
    timestamp(payload.get("available_at"), label="available_at")
    sha256(payload.get("source_sha256"), label="source_sha256")
    if type(payload.get("source_path")) is not str or not payload["source_path"]:
        raise IntelligenceError("company evidence source path is invalid")
    metrics = payload.get("metrics")
    if type(metrics) is not dict or not metrics:
        raise IntelligenceError("company evidence metrics are invalid")
    replay = build_company_source_evidence(
        company=payload["company_code"],
        source_type=source_type,
        source_path=payload["source_path"],
        source_sha256=payload["source_sha256"],
        available_at=payload["available_at"],
        metrics=metrics,
        source_page=payload.get("source_page"),
        created_at=artifact["created_at"],
    )
    if replay != artifact:
        raise IntelligenceError("company evidence does not replay")
    return artifact


def _exposure_state(revenue_share: Decimal) -> tuple[str, str]:
    if revenue_share >= Decimal("0.30"):
        return "HIGH", "THEME_REVENUE_SHARE_AT_LEAST_30_PERCENT"
    if revenue_share >= Decimal("0.10"):
        return "MEDIUM", "THEME_REVENUE_SHARE_10_TO_30_PERCENT"
    if revenue_share > Decimal("0"):
        return "LOW", "THEME_REVENUE_SHARE_BELOW_10_PERCENT"
    return "UNVERIFIED", "THEME_REVENUE_SHARE_NOT_POSITIVE"


def build_source_bound_economic_exposure_projection(
    *,
    as_of: str,
    daily_policy: Mapping[str, Any] | bytes,
    theme_projection: Mapping[str, Any] | bytes,
    evidence: Sequence[Mapping[str, Any] | bytes],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Derive HIGH/MEDIUM/LOW only from approved quantitative revenue evidence."""

    cutoff = timestamp(as_of, label="as_of")
    governance = approved_theme_governance_policy()
    daily, daily_payload = artifact_payload(daily_policy, expected_kind="daily_research_policy")
    theme, theme_payload = artifact_payload(
        theme_projection, expected_kind="theme_membership_projection"
    )
    require_no_future(theme, as_of=cutoff, label="theme projection")
    require_artifact_ref(theme_payload["policy_ref"], daily, label="theme.policy_ref")
    if daily_payload["strategy_id"] != STRATEGY_ID:
        raise IntelligenceError("economic exposure strategy differs")
    validated: dict[str, dict[str, Any]] = {}
    for value in evidence:
        artifact = validate_company_source_evidence(value)
        payload = artifact["payload"]
        company = payload["company_code"]
        if company in validated:
            raise IntelligenceError("economic exposure evidence is duplicated")
        if payload["source_type"] not in EXPOSURE_SOURCE_TYPES:
            raise IntelligenceError("economic exposure source type differs")
        require_no_future(artifact, as_of=cutoff, label="economic exposure evidence")
        validated[company] = artifact

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for membership in theme_payload["company_rows"]:
        company = company_code(membership["company_code"])
        matched = membership["technology_theme_ids"]
        gate = (
            "UNAVAILABLE"
            if membership["status"] == "UNMAPPED"
            else "PASS" if matched else "REJECT_NON_TECH"
        )
        evidence_artifact = validated.get(company)
        state = "UNVERIFIED"
        refs: list[dict[str, str]] = []
        reasons = ["TECHNOLOGY_MEMBERSHIP_NOT_ADMITTED"]
        if gate == "PASS":
            reasons = ["ECONOMIC_EXPOSURE_SOURCE_REQUIRED"]
            if evidence_artifact is not None:
                metrics = evidence_artifact["payload"]["metrics"]
                primary_theme = metrics.get("primary_theme_id")
                if primary_theme not in matched:
                    raise IntelligenceError("economic exposure Theme binding differs")
                share = decimal_value(
                    metrics.get("theme_revenue_share"),
                    label="theme_revenue_share",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                )
                state, reason = _exposure_state(share)
                refs = [artifact_ref(evidence_artifact)]
                reasons = [reason]
            if state == "UNVERIFIED":
                blockers.append(f"ECONOMIC_EXPOSURE_UNVERIFIED:{company}")
        elif evidence_artifact is not None:
            raise IntelligenceError("economic exposure evidence targets a non-technology company")
        rows.append(
            {
                "company_code": company,
                "economic_exposure_state": state,
                "evidence_refs": refs,
                "membership_theme_ids": matched,
                "reason_codes": reasons,
                "technology_gate": gate,
            }
        )
    evidence_refs = sorted(
        (artifact_ref(value) for value in validated.values()),
        key=lambda row: row["artifact_id"].encode("utf-8"),
    )
    evidence_set_sha = hashlib.sha256(canonical_json_bytes(evidence_refs)).hexdigest()
    projection = build_artifact(
        kind=EXPOSURE_PROJECTION_KIND,
        identity_field="projection_id",
        identity=business_identity(
            kind=EXPOSURE_PROJECTION_KIND,
            identity_inputs={
                "company_set_sha256": theme_payload["company_set_sha256"],
                "evidence_set_sha256": evidence_set_sha,
                "policy_id": governance["artifact_id"],
                "trade_date": theme_payload["trade_date"],
            },
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "blocker_codes": sorted(blockers),
            "company_rows": rows,
            "company_set_sha256": theme_payload["company_set_sha256"],
            "daily_policy_ref": artifact_ref(daily),
            "governance_policy_ref": artifact_ref(governance),
            "status": "BLOCKED" if blockers else "READY",
            "strategy_id": STRATEGY_ID,
            "theme_projection_ref": artifact_ref(theme),
        },
    )
    return projection, validated


def theme_assessment_from_exposure(
    *,
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    """Build one Theme assessment; LOW exposure becomes a research hard veto."""

    artifact = validate_company_source_evidence(evidence)
    metrics = artifact["payload"]["metrics"]
    share = decimal_value(
        metrics["theme_revenue_share"],
        label="theme_revenue_share",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    state, _reason = _exposure_state(share)
    if state == "UNVERIFIED":
        raise IntelligenceError("Theme assessment requires verified economic exposure")
    theme_id = identifier(metrics["primary_theme_id"], label="primary_theme_id")
    risks = (
        [
            {
                "hard_veto_codes": ["LOW_ECONOMIC_EXPOSURE"],
                "severity": "0.7",
                "theme_id": theme_id,
            }
        ]
        if state == "LOW"
        else []
    )
    return assess_theme(
        company=row["company_code"],
        memberships=[
            {
                "available_at": artifact["payload"]["available_at"],
                "exposure": "1",
                "exposure_basis": "REVENUE_SHARE_EVIDENCE",
                "provider": "SOURCE_BOUND",
                "score": decimal_text(share),
                "status": "ACTIVE",
                "theme_id": theme_id,
            }
        ],
        provider_precedence=["SOURCE_BOUND"],
        catalog_complete=True,
        as_of=as_of,
        risk_rows=risks,
    )


def _percentile(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA)
    valid = values.dropna()
    result = pd.Series(index=series.index, dtype="float64")
    if valid.empty:
        return result
    if len(valid) == 1:
        result.loc[valid.index] = 0.5
        return result
    ranks = valid.rank(method="average", ascending=higher_is_better)
    if higher_is_better:
        ranks = valid.rank(method="average", ascending=True)
    else:
        ranks = valid.rank(method="average", ascending=False)
    result.loc[valid.index] = (ranks - 1) / (len(valid) - 1)
    return result


def build_fundamental_assessments_from_frame(
    *,
    frame: pd.DataFrame,
    companies: Sequence[str],
    source_path: str,
    source_sha256: str,
    source_available_at: str,
    as_of: str,
    industry_assessments: Mapping[str, Mapping[str, Any]],
    theme_assessments: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build approved equal-weight cross-sectional Fundamental assessments."""

    required = {"ts_code", "trade_date", *FUNDAMENTAL_METRICS}
    if not required <= set(frame.columns):
        raise IntelligenceError("Fundamental snapshot columns are incomplete")
    data = frame.copy()
    data["ts_code"] = data["ts_code"].map(company_code)
    data = (
        data.sort_values(["ts_code", "trade_date"], kind="mergesort")
        .groupby("ts_code", as_index=False)
        .tail(1)
    )
    data = data.set_index("ts_code", drop=False)
    percentiles = {
        "roe": _percentile(data["fin_roe"], higher_is_better=True),
        "roa": _percentile(data["fin_roa"], higher_is_better=True),
        "debt": _percentile(data["fin_debt_to_assets"], higher_is_better=False),
        "ocf": _percentile(data["fin_ocf_to_profit"], higher_is_better=True),
        "fcf_profit": _percentile(data["fin_fcf_to_profit"], higher_is_better=True),
        "growth": _percentile(data["fin_net_profit_yoy"], higher_is_better=True),
        "revision": _percentile(data["forecast_revision"], higher_is_better=True),
        "value": _percentile(data["fcf_to_price"], higher_is_better=True),
    }

    def average(values: Sequence[Any]) -> str | None:
        finite = [float(value) for value in values if value is not None and pd.notna(value)]
        return None if not finite else decimal_text(Decimal(str(sum(finite) / len(finite))))

    assessments: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for raw_company in companies:
        company = company_code(raw_company)
        if company not in data.index:
            continue
        row = data.loc[company]
        metrics = {
            metric: (
                None
                if row[metric] is None or pd.isna(row[metric])
                else Decimal(str(float(row[metric])))
            )
            for metric in FUNDAMENTAL_METRICS
        }
        metrics["snapshot_trade_date"] = Decimal(
            str(int(pd.Timestamp(row["trade_date"]).strftime("%Y%m%d")))
        )
        source = build_company_source_evidence(
            company=company,
            source_type=FUNDAMENTAL_SOURCE_TYPE,
            source_path=source_path,
            source_sha256=source_sha256,
            available_at=source_available_at,
            metrics=metrics,
            source_page=None,
            created_at=as_of,
        )
        sources.append(source)
        scores = {
            "business_quality": average(
                [
                    percentiles["roe"].get(company),
                    percentiles["roa"].get(company),
                    percentiles["debt"].get(company),
                ]
            ),
            "earnings_quality": average(
                [percentiles["ocf"].get(company), percentiles["fcf_profit"].get(company)]
            ),
            "growth_durability": average(
                [percentiles["growth"].get(company), percentiles["revision"].get(company)]
            ),
            "industry_cycle": None,
            "valuation": average([percentiles["value"].get(company)]),
        }
        assessments[company] = assess_fundamental(
            company=company,
            component_scores=scores,
            component_weights=FUNDAMENTAL_WEIGHTS,
            minimum_coverage=FUNDAMENTAL_MINIMUM_COVERAGE,
            as_of=as_of,
            source_refs=[artifact_ref(source)],
            industry_assessment=industry_assessments.get(company),
            theme_assessment=theme_assessments.get(company),
        )
    return assessments, sources


__all__ = [
    "EXPOSURE_SOURCE_TYPES",
    "FUNDAMENTAL_MINIMUM_COVERAGE",
    "FUNDAMENTAL_SOURCE_TYPE",
    "FUNDAMENTAL_WEIGHTS",
    "MARKET_RISK_KIND",
    "SOURCE_KIND",
    "build_company_source_evidence",
    "build_fundamental_assessments_from_frame",
    "build_market_risk_evidence",
    "build_source_bound_economic_exposure_projection",
    "theme_assessment_from_exposure",
    "validate_company_source_evidence",
    "validate_market_risk_evidence",
]
