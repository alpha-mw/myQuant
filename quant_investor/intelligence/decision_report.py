"""Native five-state Decision display with independent source-completeness evidence."""

from decimal import Decimal
import hashlib

from quant_investor.contracts import canonical_json_bytes, validate_artifact
from quant_investor.operations.daily_contract import validate_ref
from quant_investor.operations.research_capture import validate_compilation
from ._common import artifact_ref, build_artifact, business_identity, IntelligenceError, timestamp
from .investment_decision import validate_investment_decision
from .low_frequency import FRESHNESS_KIND
from .portfolio_state import REPORT_POLICY, STRATEGY_ID, validate_portfolio_state
from .pcb_ai_hardware import physical_refs
from .fundamental_time import availability_instant

REPORT_KIND = "daily_research_decision_report"
DOMAINS = ("factor", "top100", "theme", "industry", "exposure", "fundamental", "macro", "portfolio")


def _refs(artifacts, references):
    refs = {
        canonical_json_bytes(references[id(value)]): references[id(value)] for value in artifacts
    }
    return [refs[key] for key in sorted(refs)]


def _status(*, domain, complete, partial=False, reasons=()):
    codes = sorted(set(reasons))
    state = "PARTIAL" if partial else "COMPLETE" if complete else "MISSING"
    if state == "MISSING":
        codes = sorted({*codes, "DOMAIN_SOURCE_MISSING:" + domain})
    return state, codes


def _company_artifacts(artifacts, kind, company):
    values = [
        a for a in artifacts if a["kind"] == kind and a["payload"].get("company_code") == company
    ]
    if len(values) > 1:
        raise IntelligenceError("DECISION_REPORT_COMPANY_ARTIFACT_DUPLICATED")
    return values


def _projection_row(artifact, company):
    if artifact is None:
        return None
    rows = [row for row in artifact["payload"]["company_rows"] if row["company_code"] == company]
    if len(rows) > 1:
        raise IntelligenceError("DECISION_REPORT_COMPANY_ROW_DUPLICATED")
    return rows[0] if rows else None


def _freshness(report, *, domain, company=None):
    if report is None:
        return [], ["FRESHNESS_REPORT_MISSING:" + domain]
    body = validate_artifact(report, expected_kind=FRESHNESS_KIND)["payload"]
    if body["domain"] != domain.upper():
        raise IntelligenceError("DECISION_REPORT_FRESHNESS_DOMAIN_INVALID")
    if company is None:
        return body["warning_codes"], body["critical_missing_codes"]
    rows = [row for row in body["entries"] if row["subject_id"] == company]
    if len(rows) != 1:
        return [], ["FRESHNESS_COMPANY_MISSING:" + company]
    return rows[0]["warning_codes"], rows[0]["critical_missing_codes"]


def _fundamental_status(assessment, freshness, company):
    warnings, critical = _freshness(freshness, domain="fundamental", company=company)
    if assessment is None:
        return _status(domain="fundamental", complete=False, reasons=critical)
    body = assessment["payload"]
    valid = body["status"] in {"COMPLETE", "PARTIAL"} and not body["blocker_codes"] and not critical
    if body["status"] == "PARTIAL" and not body["blocker_codes"]:
        warnings = [*warnings, "FUNDAMENTAL_NATIVE_PARTIAL_COVERAGE"]
    return _status(
        domain="fundamental",
        complete=valid,
        partial=valid and bool(warnings),
        reasons=[*warnings, *critical, *body["blocker_codes"]],
    )


def _domain_statuses(company, *, projections, assessments, freshness, portfolio, physical):
    theme = _projection_row(projections["theme"], company)
    industry = _projection_row(projections["industry"], company)
    exposure = _projection_row(projections["exposure"], company)
    macro = projections["macro"]
    macro_warnings, macro_critical = _freshness(freshness.get("macro"), domain="macro")
    macro_ready = (
        macro is not None
        and macro["payload"]["classification"] == "CANONICAL_MACRO_READY"
        and not macro_critical
    )
    states = {
        "factor": _status(domain="factor", complete=bool(physical["factor"])),
        "top100": _status(domain="top100", complete=bool(physical["top100"])),
        "theme": _status(
            domain="theme",
            complete=theme is not None and theme["status"] in {"MEMBERSHIP_ONLY", "NO_MEMBERSHIP"},
        ),
        "industry": _status(
            domain="industry", complete=industry is not None and industry["status"] == "AVAILABLE"
        ),
        "exposure": _status(
            domain="exposure",
            complete=exposure is not None
            and exposure["economic_exposure_state"] in {"HIGH", "MEDIUM", "LOW"},
        ),
        "fundamental": _fundamental_status(
            assessments.get(company), freshness.get("fundamental"), company
        ),
        "macro": _status(
            domain="macro",
            complete=macro_ready,
            partial=macro_ready and bool(macro_warnings),
            reasons=[*macro_warnings, *macro_critical],
        ),
        "portfolio": _status(
            domain="portfolio",
            complete=True,
            partial=portfolio["timing_status"] == "LATE_RECORDED",
            reasons=portfolio["reason_codes"],
        ),
    }
    for domain in DOMAINS:
        if not physical[domain]:
            states[domain] = _status(domain=domain, complete=False, reasons=states[domain][1])
    return states


def _native_parts(result, references):
    artifacts = result["artifacts"]
    projections = {}
    for domain, kind in (
        ("rank", "factor_research_rank"),
        ("theme", "theme_membership_projection"),
        ("industry", "industry_source_projection"),
        ("exposure", "theme_economic_exposure_projection"),
        ("macro", "market_risk_evidence"),
    ):
        values = [value for value in artifacts if value["kind"] == kind]
        if len(values) > 1:
            raise IntelligenceError("DECISION_REPORT_NATIVE_PROJECTION_DUPLICATED")
        projections[domain] = values[0] if values else None
    rank = projections["rank"]
    if rank is None or rank["payload"]["status"] != "READY":
        raise IntelligenceError("DECISION_REPORT_RANK_UNAVAILABLE")
    companies = [row["symbol"] for row in rank["payload"]["pool_rows"]]
    if len(companies) != 100 or companies != [row["company_code"] for row in result["decisions"]]:
        raise IntelligenceError("DECISION_REPORT_TOP100_ORDER_MISMATCH")
    indexed = {canonical_json_bytes(references[id(value)]): value for value in artifacts}
    return artifacts, projections, companies, indexed


def _row_evidence(company, artifacts, projections, portfolio):
    selected = {domain: [] for domain in DOMAINS}
    selected["factor"] = selected["top100"] = [projections["rank"]]
    for domain in ("theme", "industry", "exposure", "macro"):
        if projections[domain] is not None:
            selected[domain].append(projections[domain])
    for domain in ("theme", "industry", "fundamental"):
        selected[domain].extend(_company_artifacts(artifacts, domain + "_assessment", company))
    for value in artifacts:
        body = value["payload"]
        if value["kind"] == "company_source_evidence" and body["company_code"] == company:
            domain = "fundamental" if body["source_type"] == "FUNDAMENTAL_SNAPSHOT" else "exposure"
            selected[domain].append(value)
    selected["portfolio"] = [portfolio]
    return selected


def build_decision_report(
    *,
    result,
    result_ref,
    portfolio_state,
    portfolio_state_ref,
    domain_physical_refs,
    freshness_reports,
    created_at,
):
    if set(domain_physical_refs) != set(DOMAINS) or set(freshness_reports) != {
        "fundamental",
        "macro",
    }:
        raise IntelligenceError("DECISION_REPORT_DOMAIN_SET_INVALID")
    custody = timestamp(created_at, label="Decision report actual creation")
    portfolio = validate_portfolio_state(portfolio_state)
    body = _validate_report_inputs(
        result, result_ref, portfolio, portfolio_state_ref, custody, freshness_reports
    )
    references = {id(value): artifact_ref(value) for value in [*result["artifacts"], portfolio]}
    artifacts, projections, companies, indexed = _native_parts(result, references)
    physical = {domain: physical_refs(domain_physical_refs[domain]) for domain in DOMAINS}
    if physical["portfolio"] != physical_refs([portfolio_state_ref, *body["source_refs"]]):
        raise IntelligenceError("DECISION_REPORT_PORTFOLIO_PHYSICAL_BINDING_MISMATCH")
    assessments = {}
    for company in companies:
        rows = _company_artifacts(artifacts, "fundamental_assessment", company)
        if rows:
            assessments[company] = rows[0]
    held = {row["symbol"] for row in body["positions"] if Decimal(row["shares"]) > 0}
    domain_artifacts = {domain: [] for domain in DOMAINS}
    domain_states = {domain: [] for domain in DOMAINS}
    company_rows = []
    for company, native_row in zip(companies, result["decisions"]):
        decision = indexed[canonical_json_bytes(native_row["decision_ref"])]
        context = indexed[canonical_json_bytes(decision["payload"]["context_ref"])]
        validate_investment_decision(decision, context=context)
        evidence = _row_evidence(company, artifacts, projections, portfolio)
        states = _domain_statuses(
            company,
            projections=projections,
            assessments=assessments,
            freshness=freshness_reports,
            portfolio=body,
            physical=physical,
        )
        for domain in DOMAINS:
            domain_artifacts[domain].extend(evidence[domain])
            domain_states[domain].append(states[domain])
        confidence = source_confidence(states)
        company_rows.append(
            {
                "symbol": company,
                "decision": decision["payload"]["state"],
                "native_decision_ref": references[id(decision)],
                "portfolio_membership": "HELD" if company in held else "NOT_HELD",
                "evidence_refs": _refs(
                    [value for values in evidence.values() for value in values], references
                ),
                "blockers": decision["payload"]["blocker_codes"],
                "reason_codes": decision["payload"]["reason_codes"],
                "confidence": confidence,
                "confidence_reason_codes": sorted(
                    {code for _, codes in states.values() for code in codes}
                ),
            }
        )
    bindings = _build_bindings(domain_artifacts, domain_states, physical, references)
    fields = {
        "as_of": result["as_of"],
        "trade_date": body["trade_date"],
        "strategy_id": STRATEGY_ID,
        "report_policy": REPORT_POLICY,
        "native_result_ref": validate_ref(result_ref),
        "portfolio_state_ref": validate_ref(portfolio_state_ref),
        "source_bindings": bindings,
        "company_rows": company_rows,
        "blocker_codes": sorted({code for row in company_rows for code in row["blockers"]}),
        "timing_status": body["timing_status"],
        "prospective": False,
    }
    return build_artifact(
        kind=REPORT_KIND,
        identity_field="report_id",
        identity=business_identity(
            kind=REPORT_KIND, identity_inputs={"created_at": custody, **fields}
        ),
        created_at=custody,
        fields=fields,
    )


def _build_bindings(domain_artifacts, domain_states, physical, references):
    bindings = {}
    for domain in DOMAINS:
        labels = {value[0] for value in domain_states[domain]}
        bindings[domain] = {
            "artifact_refs": _refs(domain_artifacts[domain], references),
            "physical_refs": physical[domain],
            "status": next(
                label for label in ("MISSING", "PARTIAL", "COMPLETE") if label in labels
            ),
            "reason_codes": sorted({code for _, codes in domain_states[domain] for code in codes}),
        }
    return bindings


def _validate_report_inputs(
    result, result_ref, portfolio, portfolio_state_ref, custody, freshness_reports
):
    body = portfolio["payload"]
    if (
        validate_ref(result_ref)["sha256"]
        != hashlib.sha256(canonical_json_bytes(result)).hexdigest()
        or validate_ref(portfolio_state_ref)["sha256"]
        != hashlib.sha256(canonical_json_bytes(portfolio)).hexdigest()
    ):
        raise IntelligenceError("DECISION_REPORT_INPUT_REF_BYTES_MISMATCH")
    if (
        body["strategy_id"] != STRATEGY_ID
        or result["strategy_id"] != STRATEGY_ID
        or body["as_of"] != result["as_of"]
        or availability_instant(portfolio["created_at"]) > availability_instant(custody)
    ):
        raise IntelligenceError("DECISION_REPORT_PORTFOLIO_CONTEXT_INVALID")
    for domain, report in freshness_reports.items():
        if report is not None:
            fresh = validate_artifact(report, expected_kind=FRESHNESS_KIND)
            if (
                fresh["payload"]["as_of"] != result["as_of"]
                or fresh["payload"]["trade_date"] != body["trade_date"]
            ):
                raise IntelligenceError("DECISION_REPORT_FRESHNESS_DATE_INVALID")
    validate_compilation(result, trade_date=body["trade_date"])
    return body


def source_confidence(states):
    labels = {value[0] for value in states.values()}
    return (
        "MISSING"
        if "MISSING" in labels
        else "PARTIAL_SOURCE_BOUND" if "PARTIAL" in labels else "COMPLETE_SOURCE_BOUND"
    )
