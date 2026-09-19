"""Missing-only execution completion; native evidence and admission stay unchanged."""

import hashlib

from quant_investor.contracts import canonical_json_bytes, validate_artifact
from quant_investor.factors.production_pit import FOCUS_COMPANIES
from quant_investor.intelligence._common import company_code
from quant_investor.intelligence.daily_evidence import (
    build_source_bound_economic_exposure_projection,
    validate_company_source_evidence,
)
from quant_investor.intelligence.pcb_ai_hardware import EVIDENCE_KIND, build_focus_evidence
from quant_investor.intelligence.storage import approved_theme_policy_v2
from quant_investor.intelligence.theme_governance import (
    EXPOSURE_PROJECTION_KIND,
    build_unverified_economic_exposure_projection,
)
from .daily_contract import ContractError

POLICY = "native-optional-exposure-missing.v1"
FOCUS_MISSING = frozenset(
    {
        "FOCUS_SOURCE_MISSING",
        "FOCUS_PIT_NOT_ELIGIBLE",
        "FOCUS_MEMBERSHIP_UNAVAILABLE",
        "FOCUS_INDUSTRY_SOURCE_MISSING",
        "FOCUS_INDUSTRY_UNAVAILABLE",
        "FOCUS_EXPOSURE_SOURCE_MISSING",
        "FOCUS_EXPOSURE_NOT_QUALIFIED",
    }
)


def optional_exposure_completion(*, recipe):
    if "source_completion_policy" not in recipe:
        return False
    selected = recipe["source_completion_policy"]
    if type(selected) is not str or selected != POLICY:
        raise ContractError("EXPOSURE_COMPLETION_POLICY_INVALID")
    return True


def _company_rows(body, companies):
    rows = body["company_rows"]
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ContractError("EXPOSURE_COMPLETION_COMPANY_ROWS_INVALID")
    codes = [company_code(row.get("company_code")) for row in rows]
    expected_sha = hashlib.sha256(canonical_json_bytes(sorted(companies))).hexdigest()
    if (
        len(codes) != len(set(codes))
        or set(codes) != set(companies)
        or body["company_set_sha256"] != expected_sha
    ):
        raise ContractError("EXPOSURE_COMPLETION_COMPANY_SET_INVALID")
    return rows


def validate_optional_exposure_completion(*, companies, theme, projection, daily_policy, evidence):
    """Caller has read exact sources; this adds intrinsic native replay and classification."""
    if type(companies) is not list or not 0 < len(companies) <= 100:
        raise ContractError("EXPOSURE_COMPLETION_EXPECTED_COMPANIES_INVALID")
    expected = [company_code(company) for company in companies]
    if len(expected) != len(set(expected)) or daily_policy != approved_theme_policy_v2():
        raise ContractError("EXPOSURE_COMPLETION_CONTEXT_INVALID")
    native_theme = validate_artifact(theme, expected_kind="theme_membership_projection")
    body = native_theme["payload"]
    members = _company_rows(body, expected)
    if (
        body["status"] != "READY"
        or body["blocker_codes"]
        or any(row["status"] not in {"MEMBERSHIP_ONLY", "NO_MEMBERSHIP"} for row in members)
    ):
        raise ContractError("EXPOSURE_COMPLETION_THEME_INCOMPLETE")
    native = validate_artifact(projection, expected_kind=EXPOSURE_PROJECTION_KIND)
    rows = _company_rows(native["payload"], expected)
    checked = [validate_company_source_evidence(value) for value in evidence]
    codes = [value["payload"]["company_code"] for value in checked]
    if len(codes) != len(set(codes)) or not set(codes) <= set(expected):
        raise ContractError("EXPOSURE_COMPLETION_EVIDENCE_COMPANY_INVALID")
    arguments = dict(as_of=body["as_of"], daily_policy=daily_policy, theme_projection=native_theme)
    rebuilt = (
        build_source_bound_economic_exposure_projection(**arguments, evidence=checked)[0]
        if checked
        else build_unverified_economic_exposure_projection(**arguments)
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(native):
        raise ContractError("EXPOSURE_COMPLETION_NATIVE_REPLAY_MISMATCH")
    missing = []
    for row in rows:
        if row["technology_gate"] == "PASS" and row["economic_exposure_state"] == "UNVERIFIED":
            if row["evidence_refs"] or row["reason_codes"] != ["ECONOMIC_EXPOSURE_SOURCE_REQUIRED"]:
                raise ContractError("EXPOSURE_COMPLETION_NOT_OPTIONAL_MISSING")
            missing.append("ECONOMIC_EXPOSURE_UNVERIFIED:" + row["company_code"])
    missing = sorted(set(missing))
    if native["payload"]["blocker_codes"] != missing:
        raise ContractError("EXPOSURE_COMPLETION_BLOCKER_SET_INVALID")
    return tuple(missing)


def validate_optional_focus_completion(
    *, report, membership, pit, focus_theme, industry, evidence, industry_source_refs, daily_policy
):
    """Native focus replay precedes a finite missing-only policy; no ref admission by itself."""
    native = validate_artifact(report, expected_kind=EVIDENCE_KIND)
    rebuilt = build_focus_evidence(
        membership=membership,
        pit=pit,
        focus_theme=focus_theme,
        industry=industry,
        evidence=evidence,
        industry_source_refs=industry_source_refs,
        daily_policy=daily_policy,
    )
    if canonical_json_bytes(native) != canonical_json_bytes(rebuilt):
        raise ContractError("FOCUS_COMPLETION_NATIVE_REPLAY_MISMATCH")
    rows = _company_rows(native["payload"], list(FOCUS_COMPANIES))
    missing = []
    for row in rows:
        codes = row["missing_codes"]
        if codes != sorted(set(codes)) or not set(codes) <= FOCUS_MISSING:
            raise ContractError("FOCUS_COMPLETION_NOT_OPTIONAL_MISSING")
        exposure = row["economic_exposure"]
        if "FOCUS_EXPOSURE_NOT_QUALIFIED" in codes and exposure is not None:
            if (
                exposure["economic_exposure_state"] != "UNVERIFIED"
                or exposure["evidence_refs"]
                or exposure["reason_codes"]
                not in (
                    ["ECONOMIC_EXPOSURE_SOURCE_REQUIRED"],
                    ["TECHNOLOGY_MEMBERSHIP_NOT_ADMITTED"],
                )
            ):
                raise ContractError("FOCUS_COMPLETION_NOT_OPTIONAL_MISSING")
        missing.extend(code + ":" + row["company_code"] for code in codes)
    missing = sorted(set(missing))
    state = "PARTIAL_WITH_EXPLICIT_MISSING" if missing else "SUCCEEDED"
    if (
        native["payload"]["missing_codes"] != missing
        or native["payload"]["completion_state"] != state
    ):
        raise ContractError("FOCUS_COMPLETION_MISSING_SET_INVALID")
    return tuple(missing)
