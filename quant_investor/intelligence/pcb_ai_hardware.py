"""Separate source-bound focus reports; never rank, strategy or trade authority."""

from copy import deepcopy
import hashlib

from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.production_pit import FOCUS_COMPANIES
from quant_investor.operations.daily_contract import validate_ref
from ._common import (
    artifact_payload,
    artifact_ref,
    build_artifact,
    business_identity,
    IntelligenceError,
    timestamp,
)

MEMBERSHIP_KIND = "pcb_ai_hardware_membership"
EVIDENCE_KIND = "pcb_ai_hardware_evidence"
FOCUS_TOPICS = ["PCB", "AI_HARDWARE"]
FOCUS_NAMES = {"002384.SZ": "东山精密", "002463.SZ": "沪电股份"}
FOCUS_SHA = hashlib.sha256(canonical_json_bytes(list(FOCUS_COMPANIES))).hexdigest()
MISSING_CODES = frozenset(
    {
        "FOCUS_SOURCE_MISSING",
        "FOCUS_PIT_NOT_ELIGIBLE",
        "FOCUS_MEMBERSHIP_UNAVAILABLE",
        "FOCUS_POOL_MEMBERSHIP_CONFLICT",
        "FOCUS_INDUSTRY_SOURCE_MISSING",
        "FOCUS_INDUSTRY_UNAVAILABLE",
        "FOCUS_EXPOSURE_SOURCE_MISSING",
        "FOCUS_EXPOSURE_NOT_QUALIFIED",
    }
)


def physical_refs(values):
    refs = {}
    for value in values:
        checked = validate_ref(value)
        if checked["path"] in refs and refs[checked["path"]] != checked["sha256"]:
            raise IntelligenceError("focus physical source refs conflict")
        refs[checked["path"]] = checked["sha256"]
    return [{"path": path, "sha256": digest} for path, digest in sorted(refs.items())]


def projection_rows(value, *, kind, as_of, scope=None):
    artifact, body = artifact_payload(value, expected_kind=kind)
    if body["as_of"] != as_of or artifact["created_at"] > as_of:
        raise IntelligenceError("focus projection cutoff differs")
    rows = body["company_rows"]
    codes = [row["company_code"] for row in rows]
    if codes != sorted(set(codes)) or (scope is not None and codes != list(scope)):
        raise IntelligenceError("focus projection company scope differs")
    if body["company_set_sha256"] != hashlib.sha256(canonical_json_bytes(codes)).hexdigest():
        raise IntelligenceError("focus projection company SHA differs")
    return {row["company_code"]: deepcopy(row) for row in rows}


def focus_artifact(kind, *, as_of, pool_manifest_ref, pit, source_refs, rows):
    stamp = timestamp(as_of, label="focus cutoff")
    if (
        pit["trade_date"] != stamp[:10].replace("-", "")
        or pit["company_keyset"] != list(FOCUS_COMPANIES)
        or pit["company_set_sha256"] != FOCUS_SHA
    ):
        raise IntelligenceError("focus PIT scope differs")
    if [row["company_code"] for row in rows] != list(FOCUS_COMPANIES):
        raise IntelligenceError("focus report must retain both companies")
    missing = []
    for row in rows:
        if (
            row["missing_codes"] != sorted(set(row["missing_codes"]))
            or set(row["missing_codes"]) - MISSING_CODES
        ):
            raise IntelligenceError("focus missing codes differ")
        missing.extend(code + ":" + row["company_code"] for code in row["missing_codes"])
    fields = {
        "as_of": stamp,
        "trade_date": pit["trade_date"],
        "focus_topics": list(FOCUS_TOPICS),
        "company_set_sha256": FOCUS_SHA,
        "pool_manifest_ref": validate_ref(pool_manifest_ref),
        **{
            key: validate_ref(pit[key])
            for key in (
                "pit_selection_ref",
                "pit_generation_manifest_ref",
                "pit_membership_ref",
            )
        },
        "source_refs": physical_refs([*pit["source_refs"], *source_refs]),
        "company_rows": rows,
        "missing_codes": sorted(missing),
        "completion_state": "PARTIAL_WITH_EXPLICIT_MISSING" if missing else "SUCCEEDED",
    }
    return build_artifact(
        kind=kind,
        identity_field="evidence_id",
        identity=business_identity(kind=kind, identity_inputs=fields),
        fields=fields,
        created_at=stamp,
    )


def build_focus_membership(
    *,
    as_of,
    pool_manifest_ref,
    pit,
    pool_theme,
    focus_theme,
    pool_source_refs,
    focus_source_refs,
):
    pool_rows = projection_rows(pool_theme, kind="theme_membership_projection", as_of=as_of)
    focus_rows = (
        {}
        if focus_theme is None
        else projection_rows(
            focus_theme,
            kind="theme_membership_projection",
            as_of=as_of,
            scope=FOCUS_COMPANIES,
        )
    )
    if (
        focus_theme is not None
        and pool_theme["payload"]["policy_ref"] != focus_theme["payload"]["policy_ref"]
    ):
        raise IntelligenceError("focus and pool Theme policies differ")
    if [row["symbol"] for row in pit["company_statuses"]] != list(FOCUS_COMPANIES):
        raise IntelligenceError("focus PIT statuses are not exact and ordered")
    statuses = {row["symbol"]: row for row in pit["company_statuses"]}
    if set(statuses) != set(FOCUS_COMPANIES):
        raise IntelligenceError("focus PIT statuses incomplete")
    focus_refs, pool_refs = physical_refs(focus_source_refs), physical_refs(pool_source_refs)
    rows = []
    for company in FOCUS_COMPANIES:
        native = focus_rows.get(company)
        overlap = pool_rows.get(company)
        missing = []
        if not statuses[company]["in_universe"] or not statuses[company]["research_eligible"]:
            missing.append("FOCUS_PIT_NOT_ELIGIBLE")
        if native is None:
            missing.append("FOCUS_SOURCE_MISSING")
        elif native["status"] == "UNMAPPED":
            missing.append("FOCUS_MEMBERSHIP_UNAVAILABLE")
        if native is not None and overlap is not None and native != overlap:
            missing.append("FOCUS_POOL_MEMBERSHIP_CONFLICT")
        rows.append(
            {
                "company_code": company,
                "company_name": FOCUS_NAMES[company],
                "in_top100": overlap is not None,
                "pit_status": deepcopy(statuses[company]),
                "membership": native if not missing else None,
                "source_evidence": {
                    "membership_rows": [native] if native is not None else [],
                    "membership_refs": focus_refs,
                    "overlap_membership_rows": [overlap] if overlap is not None else [],
                    "overlap_membership_refs": pool_refs if overlap is not None else [],
                },
                "missing_codes": sorted(missing),
            }
        )
    return focus_artifact(
        MEMBERSHIP_KIND,
        as_of=as_of,
        pool_manifest_ref=pool_manifest_ref,
        pit=pit,
        source_refs=[*focus_refs, *pool_refs],
        rows=rows,
    )


def partition_exposure_evidence(evidence, companies):
    """Validate the union before splitting facts into independent native scopes."""
    from .daily_evidence import validate_company_source_evidence

    allowed = set(companies) | set(FOCUS_COMPANIES)
    checked = {}
    for value in evidence:
        artifact = validate_company_source_evidence(value)
        company = artifact["payload"]["company_code"]
        if company not in allowed or company in checked:
            raise IntelligenceError("focus exposure union contains an unknown or duplicate company")
        checked[company] = artifact
    return (
        [value for company, value in checked.items() if company in set(companies)],
        [checked[company] for company in FOCUS_COMPANIES if company in checked],
    )


def append_focus_artifacts(ordinary, report, evidence):
    result = [*ordinary, report]
    identities = {(value["kind"], value["artifact_id"]): artifact_ref(value) for value in result}
    for value in sorted(
        evidence, key=lambda v: (v["kind"], v["artifact_id"], artifact_ref(v)["byte_sha256"])
    ):
        identity = (value["kind"], value["artifact_id"])
        ref = artifact_ref(value)
        if identity in identities:
            if identities[identity] != ref:
                raise IntelligenceError("focus artifact identity conflicts")
            continue
        identities[identity] = ref
        result.append(value)
    return result


def _focus_membership_context(membership, pit):
    _, body = artifact_payload(membership, expected_kind=MEMBERSHIP_KIND)
    as_of = body["as_of"]
    if (
        body["focus_topics"] != FOCUS_TOPICS
        or body["company_set_sha256"] != FOCUS_SHA
        or body["trade_date"] != pit["trade_date"]
        or any(
            body[key] != pit[key]
            for key in ("pit_selection_ref", "pit_generation_manifest_ref", "pit_membership_ref")
        )
    ):
        raise IntelligenceError("focus membership PIT or scope binding differs")
    members = {row["company_code"]: row for row in body["company_rows"]}
    if sorted(members) != list(FOCUS_COMPANIES):
        raise IntelligenceError("focus membership report scope differs")
    statuses = {row["symbol"]: row for row in pit["company_statuses"]}
    if set(statuses) != set(FOCUS_COMPANIES) or any(
        members[company]["pit_status"] != statuses[company] for company in FOCUS_COMPANIES
    ):
        raise IntelligenceError("focus membership listing status differs")
    return body, as_of, members


def build_focus_evidence(
    *,
    membership,
    pit,
    focus_theme,
    industry,
    evidence,
    industry_source_refs,
    daily_policy,
):
    """Preserve source-only facts; native revenue qualification alone assigns grades."""
    from .daily_evidence import build_source_bound_economic_exposure_projection

    body, as_of, members = _focus_membership_context(membership, pit)
    _, checked = partition_exposure_evidence(evidence, [])
    facts = {value["payload"]["company_code"]: value for value in checked}
    qualified = [
        value
        for company, value in facts.items()
        if (
            members[company]["membership"] is not None
            and members[company]["membership"]["technology_theme_ids"]
        )
    ]
    exposure_rows = {}
    if focus_theme is not None:
        projection, _ = build_source_bound_economic_exposure_projection(
            as_of=as_of,
            daily_policy=daily_policy,
            theme_projection=focus_theme,
            evidence=qualified,
        )
        exposure_rows = projection_rows(
            projection,
            kind=projection["kind"],
            as_of=as_of,
            scope=FOCUS_COMPANIES,
        )
    elif qualified:
        raise IntelligenceError("focus qualified evidence lacks native membership")
    industry_rows = (
        {}
        if industry is None
        else projection_rows(
            industry,
            kind="industry_source_projection",
            as_of=as_of,
            scope=FOCUS_COMPANIES,
        )
    )
    qualified_codes = {value["payload"]["company_code"] for value in qualified}
    industry_refs = physical_refs(industry_source_refs)
    rows = []
    for company in FOCUS_COMPANIES:
        member = members[company]
        missing = list(member["missing_codes"])
        native_industry = industry_rows.get(company)
        if native_industry is None:
            missing.append("FOCUS_INDUSTRY_SOURCE_MISSING")
        elif native_industry["status"] != "AVAILABLE":
            missing.append("FOCUS_INDUSTRY_UNAVAILABLE")
        native_exposure = exposure_rows.get(company) if member["membership"] is not None else None
        fact = facts.get(company)
        if fact is None:
            missing.append("FOCUS_EXPOSURE_SOURCE_MISSING")
        elif native_exposure is None or native_exposure["economic_exposure_state"] not in {
            "HIGH",
            "MEDIUM",
            "LOW",
        }:
            missing.append("FOCUS_EXPOSURE_NOT_QUALIFIED")
        refs = [] if fact is None else [artifact_ref(fact)]
        raw_members = member["source_evidence"]["membership_rows"]
        category = (
            "COMPLETE_SOURCE_BOUND"
            if not missing
            else (
                "MEMBERSHIP_ONLY"
                if member["membership"] is not None
                else (
                    "SOURCE_ONLY"
                    if raw_members or native_industry is not None or fact is not None
                    else "MISSING"
                )
            )
        )
        reasons = [*missing, member["pit_status"]["reason"]]
        if native_exposure is not None:
            reasons.extend(native_exposure["reason_codes"])
        rows.append(
            {
                "company_code": company,
                "company_name": FOCUS_NAMES[company],
                "in_top100": member["in_top100"],
                "pit_status": deepcopy(member["pit_status"]),
                "membership": deepcopy(member["membership"]),
                "industry": native_industry,
                "economic_exposure": native_exposure,
                "confidence": {"category": category, "reason_codes": sorted(set(reasons))},
                "source_evidence": {
                    "membership_refs": physical_refs(
                        [
                            *member["source_evidence"]["membership_refs"],
                            *member["source_evidence"]["overlap_membership_refs"],
                        ]
                    ),
                    "industry_refs": industry_refs,
                    "company_evidence_refs": refs if company in qualified_codes else [],
                    "unqualified_company_evidence_refs": [] if company in qualified_codes else refs,
                },
                "missing_codes": sorted(set(missing)),
            }
        )
    source_refs = [
        {"path": v["payload"]["source_path"], "sha256": v["payload"]["source_sha256"]}
        for v in checked
    ]
    return focus_artifact(
        EVIDENCE_KIND,
        as_of=as_of,
        pool_manifest_ref=body["pool_manifest_ref"],
        pit=pit,
        source_refs=[*body["source_refs"], *industry_refs, *source_refs],
        rows=rows,
    )
