"""Same-closure Evidence Graph v2 over B0, I2, I3, I4 and frozen I0/R2.2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from ...intelligence._core import (
    IntelligenceContractError,
    content_ref as v1_content_ref,
)
from ...intelligence.bayesian import validate_bayesian_receipt
from ...intelligence.evaluator.forward_evaluator import (
    run_forward_research_evaluation,
)
from ...intelligence.evaluator.receipts import (
    ENVELOPE_VERSION,
    HYPOTHESIS_RECEIPT_VERSION,
    MAIN_RECEIPT_VERSION,
)
from ...intelligence.evidence import validate_evidence_set
from ...intelligence.fusion import validate_branch, validate_fusion_receipt
from ...intelligence.hypothesis import validate_hypothesis
from ...intelligence.runtime import build_intelligence_runtime_receipt
from ..fundamental import validate_fundamental_profile
from ..industry import (
    validate_industry_component_receipt,
    validate_industry_evaluation_receipt,
)
from ..quant_producer import (
    validate_quant_branch_v5,
    validate_subject_branch_binding,
)
from ..readiness import (
    validate_investment_data_readiness,
)
from ..theme import (
    validate_theme_component_receipt,
    validate_theme_exposure_receipt,
)

from .._core import (
    canonical_bytes,
    common_fields,
    content_ref,
    decimal_text,
    exact_ref,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from .models import DecisionV2ContractError, decision_contract, normalize_risk_rows

EVIDENCE_GRAPH_V2_VERSION: Final = "myquant.v17.research-intelligence-v2.evidence-graph.v1"
I0_REPLAY_INPUT_FIELDS: Final = {
    "bayesian_receipts",
    "branches",
    "closure_refs",
    "evaluation_refs",
    "evidence",
    "expected_memory_tip",
    "fusion_receipt",
    "hypotheses",
    "label_refs",
    "memory_entries",
    "observation_bundle",
    "observation_refs",
    "regime_input",
    "regime_receipt",
    "session_byte_sha256",
    "session_relative_path",
    "workspace_root",
}
_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
}
_GRAPH_FIELDS: Final = _COMMON_FIELDS | {
    "as_of",
    "bayesian_ref",
    "bayesian_posterior",
    "blocker_codes",
    "company_code",
    "fundamental_profile_ref",
    "fundamental_readiness_status",
    "fundamental_stale_sessions",
    "fusion_ready",
    "graph_id",
    "hypothesis_ref",
    "i0_evidence_refs",
    "i0_fundamental_branch_ref",
    "i0_fusion_ref",
    "i0_quant_branch_ref",
    "i0_runtime_ref",
    "industry_component_ref",
    "industry_identity_ref",
    "industry_state",
    "overall_risk",
    "policy_independent_hard_veto_codes",
    "quant_branch_ref",
    "quant_percentile",
    "quant_pool_ref",
    "quant_score",
    "r22_envelope_ref",
    "r22_hypothesis_evaluation_ref",
    "r22_hypothesis_status",
    "r22_main_ref",
    "r22_preregistered",
    "readiness_ref",
    "risk_rows",
    "run_id",
    "semantic_sha256",
    "subject_binding_ref",
    "theme_component_ref",
    "theme_exposure_ref",
    "theme_state",
    "v2_manifest_ref",
    "version",
}
_R22_ENVELOPE_PAYLOAD_FIELDS: Final = {
    "calibration_evidence",
    "factor_evaluations",
    "hypothesis_evaluations",
    "main_receipt",
    "memory_proposal",
    "regime_evaluation",
    "request_ref",
    "universe_inventory",
    "variant_evaluation",
    "variant_factor_evaluations",
}
_R22_MAIN_PAYLOAD_FIELDS: Final = {
    "calibration_ref",
    "evaluation_artifact_refs",
    "evaluation_window",
    "factor_refs",
    "hypothesis_evaluation_refs",
    "hypothesis_refs",
    "implementation_sha",
    "label_refs",
    "limitations",
    "memory_proposal_ref",
    "metrics",
    "observation_refs",
    "policy_ref",
    "regime_ref",
    "request_ref",
    "source_evaluation_refs",
    "universe_ref",
    "variant_ref",
}


def _fail(message: str) -> None:
    raise DecisionV2ContractError(message)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact mapping")
    return dict(value)


def _optional_pair(
    document: Mapping[str, Any] | None,
    closure: Mapping[str, Any] | None,
    *,
    label: str,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if (document is None) != (closure is None):
        _fail(f"{label} document and closure must be provided together")
    return document, closure


def _same_content_identity(
    exact: Mapping[str, Any], content: Mapping[str, Any], *, label: str
) -> None:
    for field in ("artifact_id", "artifact_version", "byte_sha256", "semantic_sha256"):
        if exact.get(field) != content.get(field):
            _fail(f"{label} content identity mismatch")


def _validate_i0(
    inputs_value: Mapping[str, Any],
    *,
    company: str,
    hypothesis_id: str,
    as_of: str,
) -> dict[str, Any]:
    inputs = require_exact_keys(inputs_value, I0_REPLAY_INPUT_FIELDS, label="i0_replay_inputs")
    try:
        runtime = build_intelligence_runtime_receipt(**dict(inputs), as_of=as_of)
        evidence = validate_evidence_set(inputs["evidence"], as_of=as_of)
        branches = [
            validate_branch(row, evidence=evidence, as_of=as_of) for row in inputs["branches"]
        ]
        branch_types = [row["branch_type"] for row in branches]
        if sorted(branch_types) != ["FUNDAMENTAL", "QUANT"]:
            _fail("fresh I0 closure must contain only Quant and Fundamental branches")
        fusion = validate_fusion_receipt(inputs["fusion_receipt"], branches=branches, as_of=as_of)
        hypotheses = [
            validate_hypothesis(row, evidence=evidence, as_of=as_of) for row in inputs["hypotheses"]
        ]
        selected = [row for row in hypotheses if row["hypothesis_id"] == hypothesis_id]
        if len(selected) != 1 or company not in selected[0]["related_companies"]:
            _fail("selected I0 hypothesis is not uniquely bound to the company")
        bayesian = [
            validate_bayesian_receipt(row, evidence=evidence, as_of=as_of)
            for row in inputs["bayesian_receipts"]
        ]
        selected_bayesian = [row for row in bayesian if row["hypothesis_id"] == hypothesis_id]
        if len(selected_bayesian) != 1:
            _fail("selected hypothesis requires exactly one Bayesian receipt")
    except DecisionV2ContractError:
        raise
    except (IntelligenceContractError, TypeError, ValueError) as exc:
        raise DecisionV2ContractError(f"frozen I0 replay failed: {exc}") from exc
    by_type = {row["branch_type"]: row for row in branches}
    return {
        "bayesian": selected_bayesian[0],
        "evidence": evidence,
        "fundamental_branch": by_type["FUNDAMENTAL"],
        "fusion": fusion,
        "hypothesis": selected[0],
        "quant_branch": by_type["QUANT"],
        "runtime": runtime,
    }


def _closed_r22_receipts(
    *,
    workspace_root: str,
    request_path: str,
    request_sha256: str,
    as_of: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from ...intelligence.evaluator.receipts import validate_closed_receipt

    sha256(request_sha256, label="r22_request_sha256")
    if not workspace_root or not request_path:
        _fail("R2.2 exact request closure is required")
    try:
        envelope = run_forward_research_evaluation(
            workspace_root,
            request_path=request_path,
            request_sha256=request_sha256,
        )
        envelope = validate_closed_receipt(
            envelope,
            version=ENVELOPE_VERSION,
            identity_field="envelope_id",
            payload_fields=_R22_ENVELOPE_PAYLOAD_FIELDS,
        )
        if envelope["timestamp"] > as_of:
            _fail("R2.2 replay is future-known")
        request_ref = envelope["request_ref"]
        if (
            request_ref.get("relative_path") != request_path
            or request_ref.get("byte_sha256") != request_sha256
        ):
            _fail("R2.2 request ref does not bind the exact request")
        main = validate_closed_receipt(
            envelope["main_receipt"],
            version=MAIN_RECEIPT_VERSION,
            identity_field="evaluation_id",
            payload_fields=_R22_MAIN_PAYLOAD_FIELDS,
        )
        if main["request_ref"] != request_ref:
            _fail("R2.2 main receipt request binding mismatch")
    except DecisionV2ContractError:
        raise
    except (IntelligenceContractError, TypeError, ValueError) as exc:
        raise DecisionV2ContractError(f"R2.2 exact replay failed: {exc}") from exc
    return envelope, main


def _selected_r22_evaluation(
    *,
    envelope: Mapping[str, Any],
    main: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    from ...intelligence._core import validate_content_addressed

    hypothesis_ref = v1_content_ref(hypothesis, identity_field="hypothesis_id")
    if main["hypothesis_refs"].count(hypothesis_ref) != 1:
        _fail("R2.2 main receipt does not uniquely bind the I0 hypothesis")
    evaluations: list[dict[str, Any]] = []
    for value in envelope["hypothesis_evaluations"]:
        row = validate_content_addressed(value, identity_field="receipt_id")
        if row.get("version") != HYPOTHESIS_RECEIPT_VERSION:
            _fail("R2.2 hypothesis evaluation version mismatch")
        if row.get("timestamp") > as_of:
            _fail("R2.2 hypothesis evaluation is future-known")
        if row.get("hypothesis_ref") == hypothesis_ref:
            evaluations.append(row)
    if len(evaluations) != 1:
        _fail("R2.2 hypothesis evaluation is not unique")
    evaluation = evaluations[0]
    evaluation_ref = v1_content_ref(evaluation, identity_field="receipt_id")
    if main["hypothesis_evaluation_refs"].count(evaluation_ref) != 1:
        _fail("R2.2 selected evaluation is outside the main receipt")
    return evaluation


def _r22_result(evaluation: Mapping[str, Any]) -> tuple[str, bool]:
    status = evaluation.get("hypothesis_status")
    evidence_summary = evaluation.get("evidence_summary")
    if (
        status not in {"SUPPORTED", "FAILED", "UNCERTAIN"}
        or type(evidence_summary) is not dict
        or type(evidence_summary.get("preregistered")) is not bool
    ):
        _fail("R2.2 hypothesis result is malformed")
    return str(status), evidence_summary["preregistered"]


def _validate_r22(
    *,
    workspace_root: str,
    request_path: str,
    request_sha256: str,
    hypothesis: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    try:
        envelope, main = _closed_r22_receipts(
            workspace_root=workspace_root,
            request_path=request_path,
            request_sha256=request_sha256,
            as_of=as_of,
        )
        evaluation = _selected_r22_evaluation(
            envelope=envelope,
            main=main,
            hypothesis=hypothesis,
            as_of=as_of,
        )
        status, preregistered = _r22_result(evaluation)
    except DecisionV2ContractError:
        raise
    except (IntelligenceContractError, TypeError, ValueError) as exc:
        raise DecisionV2ContractError(f"R2.2 exact replay failed: {exc}") from exc
    return {
        "envelope": envelope,
        "evaluation": evaluation,
        "main": main,
        "preregistered": preregistered,
        "status": status,
    }


def _v1_ref(document: Mapping[str, Any], identity_field: str) -> dict[str, str]:
    return v1_content_ref(document, identity_field=identity_field)


def _evidence_source_bindings(
    evidence: Sequence[Mapping[str, Any]],
    *,
    targets: Mapping[str, Mapping[str, Any] | None],
) -> list[str]:
    blockers: list[str] = []
    for source_type, target in targets.items():
        matches = [row for row in evidence if row["source_type"] == source_type]
        if target is None or not matches:
            blockers.append(f"{source_type}_EVIDENCE_UNAVAILABLE")
            continue
        target_ref = content_ref(
            target,
            identity_field={
                "QUANT": "quant_branch_id",
                "FUNDAMENTAL": "profile_id",
                "INDUSTRY": (
                    "component_receipt_id" if "component_receipt_id" in target else "evaluation_id"
                ),
                "THEME": (
                    "component_receipt_id"
                    if "component_receipt_id" in target
                    else "exposure_receipt_id"
                ),
            }[source_type],
        )
        bound = False
        for row in matches:
            source_ref = row["source_ref"]
            if all(source_ref.get(field) == target_ref[field] for field in target_ref):
                bound = True
                break
        if not bound:
            blockers.append(f"{source_type}_EVIDENCE_REF_MISMATCH")
    return blockers


def _fundamental_readiness(
    receipt: Mapping[str, Any], closure: Mapping[str, Any], *, as_of: str
) -> tuple[dict[str, Any], str, int]:
    row = validate_investment_data_readiness(receipt, **dict(closure))
    if row["timestamp"] != as_of:
        _fail("B0 readiness must share the graph timestamp")
    matches = [item for item in row["rows"] if item["name"] == "FUNDAMENTAL"]
    if len(matches) != 1:
        _fail("B0 readiness Fundamental row is not unique")
    status = matches[0]["status"]
    stale = 1 if status == "STALE" else 0
    return row, status, stale


def _bind_same_run_scores(
    *,
    quant: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
    i0: Mapping[str, Any],
) -> None:
    if Decimal(i0["quant_branch"]["score"]) != Decimal(quant["percentile"]):
        _fail("frozen I0 Quant score does not project the B0 percentile")
    if profile is None or profile["score_present"] is not True:
        return
    if Decimal(i0["fundamental_branch"]["score"]) != Decimal(profile["effective_score"]):
        _fail("frozen I0 Fundamental score does not project I4 effective score")
    if Decimal(i0["fundamental_branch"]["availability"]) != Decimal(profile["coverage"]):
        _fail("frozen I0 Fundamental availability does not project I4 coverage")


def _validate_b0_layers(
    *,
    quant_branch: Mapping[str, Any],
    quant_closure: Mapping[str, Any],
    subject_binding: Mapping[str, Any],
    binding_closure: Mapping[str, Any],
    company: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    quant = validate_quant_branch_v5(
        quant_branch,
        **_mapping(quant_closure, label="quant closure"),
    )
    binding = validate_subject_branch_binding(
        subject_binding,
        **_mapping(binding_closure, label="subject binding closure"),
    )
    if quant["company_code"] != company or binding["company_code"] != company:
        _fail("B0 subject binding mismatch")
    _same_content_identity(
        binding["quant_branch_ref"],
        content_ref(quant, identity_field="quant_branch_id"),
        label="B0 Quant",
    )
    return quant, binding


def _validate_industry_layers(
    *,
    identity: Mapping[str, Any],
    identity_closure: Mapping[str, Any],
    component: Mapping[str, Any] | None,
    component_closure: Mapping[str, Any] | None,
    company: str,
    as_of: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    industry = validate_industry_evaluation_receipt(
        identity,
        **_mapping(identity_closure, label="industry closure"),
    )
    if industry["subject_id"] != company or industry["timestamp"] != as_of:
        _fail("I2 subject/time binding mismatch")
    component, component_closure = _optional_pair(
        component,
        component_closure,
        label="industry component",
    )
    if component is None:
        return industry, None
    validated = validate_industry_component_receipt(
        component,
        **_mapping(component_closure, label="industry component closure"),
    )
    if validated["evaluation_ref"] != content_ref(
        industry,
        identity_field="evaluation_id",
    ):
        _fail("I2 component does not bind the selected industry identity")
    return industry, validated


def _validate_theme_layers(
    *,
    exposure: Mapping[str, Any],
    exposure_closure: Mapping[str, Any],
    component: Mapping[str, Any] | None,
    component_closure: Mapping[str, Any] | None,
    company: str,
    as_of: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    theme = validate_theme_exposure_receipt(
        exposure,
        **_mapping(exposure_closure, label="theme closure"),
    )
    if theme["company_code"] != company or theme["timestamp"] != as_of:
        _fail("I3 subject/time binding mismatch")
    component, component_closure = _optional_pair(
        component,
        component_closure,
        label="theme component",
    )
    if component is None:
        return theme, None
    validated = validate_theme_component_receipt(
        component,
        **_mapping(component_closure, label="theme component closure"),
    )
    if validated["exposure_ref"] != content_ref(
        theme,
        identity_field="exposure_receipt_id",
    ):
        _fail("I3 component does not bind the selected theme exposure")
    return theme, validated


def _validate_profile(
    *,
    document: Mapping[str, Any] | None,
    closure: Mapping[str, Any] | None,
    company: str,
    as_of: str,
) -> dict[str, Any] | None:
    document, closure = _optional_pair(document, closure, label="fundamental profile")
    if document is None:
        return None
    profile = validate_fundamental_profile(
        document,
        **_mapping(closure, label="fundamental profile closure"),
    )
    if profile["timestamp"] != as_of or profile["company_code"] != company:
        _fail("I4 profile must share the graph subject and timestamp")
    return profile


def _bind_i0_branches(
    *,
    binding: Mapping[str, Any],
    fundamental_ref: Mapping[str, Any],
    quant: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
    i0: Mapping[str, Any],
    as_of: str,
) -> dict[str, str]:
    _same_content_identity(
        binding["frozen_v1_branch_ref"],
        _v1_ref(i0["quant_branch"], "branch_id"),
        label="frozen I0 Quant branch",
    )
    exact_fundamental_ref = exact_ref(
        fundamental_ref,
        label="frozen_v1_fundamental_branch_ref",
    )
    _same_content_identity(
        exact_fundamental_ref,
        _v1_ref(i0["fundamental_branch"], "branch_id"),
        label="frozen I0 Fundamental branch",
    )
    if binding["frozen_v1_branch_ref"]["cutoff"] > as_of or exact_fundamental_ref["cutoff"] > as_of:
        _fail("frozen I0 branch binding is future-known")
    _bind_same_run_scores(quant=quant, profile=profile, i0=i0)
    return exact_fundamental_ref


def _graph_blockers(
    *,
    evidence: Sequence[Mapping[str, Any]],
    quant: Mapping[str, Any],
    industry: Mapping[str, Any],
    industry_component: Mapping[str, Any] | None,
    theme: Mapping[str, Any],
    theme_component: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
    fundamental_status: str,
    risk_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers = _evidence_source_bindings(
        evidence,
        targets={
            "FUNDAMENTAL": profile,
            "INDUSTRY": industry_component or industry,
            "QUANT": quant,
            "THEME": theme_component or theme,
        },
    )
    if industry["state"] != "AVAILABLE":
        blockers.append(f"INDUSTRY_IDENTITY_{industry['state']}")
    if industry_component is None or industry_component["status"] != "AVAILABLE":
        blockers.append("INDUSTRY_COMPONENT_UNAVAILABLE")
    if theme["status"] not in {"AVAILABLE", "NO_MEMBERSHIP"}:
        blockers.append(f"THEME_IDENTITY_{theme['status']}")
    if theme["status"] == "AVAILABLE" and (
        theme_component is None or theme_component["status"] != "AVAILABLE"
    ):
        blockers.append("THEME_COMPONENT_UNAVAILABLE")
    if (
        profile is None
        or profile["status"] == "UNAVAILABLE"
        or profile["score_present"] is not True
    ):
        blockers.append("FUNDAMENTAL_PROFILE_UNAVAILABLE")
    if fundamental_status == "BLOCKED":
        blockers.append("FUNDAMENTAL_READINESS_BLOCKED")
    blockers.extend(
        f"RISK_{row['dimension']}_UNAVAILABLE"
        for row in risk_rows
        if row["status"] == "UNAVAILABLE"
    )
    return sorted(set(blockers), key=lambda value: value.encode("ascii"))


def _risk_summary(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[str]]:
    severities = [Decimal(row["severity"]) for row in rows if row["severity"] is not None]
    overall = None if not severities else decimal_text(max(severities))
    vetoes = sorted(
        {item for row in rows for item in row["hard_veto_codes"]},
        key=lambda value: value.encode("ascii"),
    )
    return overall, vetoes


@decision_contract
def build_evidence_graph_v2(
    *,
    run_id: str,
    company_code: str,
    selected_hypothesis_id: str,
    quant_branch: Mapping[str, Any],
    quant_branch_validation_closure: Mapping[str, Any],
    subject_binding: Mapping[str, Any],
    subject_binding_validation_closure: Mapping[str, Any],
    readiness_receipt: Mapping[str, Any],
    readiness_validation_closure: Mapping[str, Any],
    industry_identity: Mapping[str, Any],
    industry_identity_validation_closure: Mapping[str, Any],
    industry_component: Mapping[str, Any] | None,
    industry_component_validation_closure: Mapping[str, Any] | None,
    theme_exposure: Mapping[str, Any],
    theme_exposure_validation_closure: Mapping[str, Any],
    theme_component: Mapping[str, Any] | None,
    theme_component_validation_closure: Mapping[str, Any] | None,
    fundamental_profile: Mapping[str, Any] | None,
    fundamental_profile_validation_closure: Mapping[str, Any] | None,
    frozen_v1_fundamental_branch_ref: Mapping[str, Any],
    i0_replay_inputs: Mapping[str, Any],
    r22_request_path: str,
    r22_request_sha256: str,
    risk_rows: Sequence[Mapping[str, Any]],
    as_of: str,
) -> dict[str, Any]:
    """Build a policy-independent graph; contract failures never become states."""

    issued_at = timestamp(as_of, label="as_of")
    run = identifier(run_id, label="run_id")
    company = identifier(company_code, label="company_code")
    hypothesis_id = sha256(selected_hypothesis_id, label="selected_hypothesis_id")
    quant, binding = _validate_b0_layers(
        quant_branch=quant_branch,
        quant_closure=quant_branch_validation_closure,
        subject_binding=subject_binding,
        binding_closure=subject_binding_validation_closure,
        company=company,
    )
    _readiness, fundamental_status, stale_sessions = _fundamental_readiness(
        readiness_receipt, readiness_validation_closure, as_of=issued_at
    )
    industry, validated_industry_component = _validate_industry_layers(
        identity=industry_identity,
        identity_closure=industry_identity_validation_closure,
        component=industry_component,
        component_closure=industry_component_validation_closure,
        company=company,
        as_of=issued_at,
    )
    theme, validated_theme_component = _validate_theme_layers(
        exposure=theme_exposure,
        exposure_closure=theme_exposure_validation_closure,
        component=theme_component,
        component_closure=theme_component_validation_closure,
        company=company,
        as_of=issued_at,
    )
    profile = _validate_profile(
        document=fundamental_profile,
        closure=fundamental_profile_validation_closure,
        company=company,
        as_of=issued_at,
    )
    i0 = _validate_i0(
        i0_replay_inputs,
        company=company,
        hypothesis_id=hypothesis_id,
        as_of=issued_at,
    )
    _bind_i0_branches(
        binding=binding,
        fundamental_ref=frozen_v1_fundamental_branch_ref,
        quant=quant,
        profile=profile,
        i0=i0,
        as_of=issued_at,
    )
    r22 = _validate_r22(
        workspace_root=str(i0_replay_inputs["workspace_root"]),
        request_path=r22_request_path,
        request_sha256=r22_request_sha256,
        hypothesis=i0["hypothesis"],
        as_of=issued_at,
    )
    admitted_refs = [_v1_ref(row, "evidence_id") for row in i0["evidence"]]
    normalized_risk = normalize_risk_rows(
        risk_rows, admitted_evidence_refs=admitted_refs, as_of=issued_at
    )
    blockers = _graph_blockers(
        evidence=i0["evidence"],
        quant=quant,
        industry=industry,
        industry_component=validated_industry_component,
        theme=theme,
        theme_component=validated_theme_component,
        profile=profile,
        fundamental_status=fundamental_status,
        risk_rows=normalized_risk,
    )
    overall_risk, vetoes = _risk_summary(normalized_risk)
    fusion_blockers = {code for code in blockers if not code.startswith("RISK_")}
    return seal(
        {
            **common_fields(timestamp_value=issued_at),
            "as_of": issued_at,
            "bayesian_ref": _v1_ref(i0["bayesian"], "receipt_id"),
            "bayesian_posterior": i0["bayesian"]["posterior"],
            "blocker_codes": sorted(set(blockers), key=lambda value: value.encode("ascii")),
            "company_code": company,
            "fundamental_profile_ref": (
                None if profile is None else content_ref(profile, identity_field="profile_id")
            ),
            "fundamental_readiness_status": fundamental_status,
            "fundamental_stale_sessions": stale_sessions,
            "fusion_ready": not fusion_blockers,
            "hypothesis_ref": _v1_ref(i0["hypothesis"], "hypothesis_id"),
            "i0_evidence_refs": admitted_refs,
            "i0_fundamental_branch_ref": _v1_ref(i0["fundamental_branch"], "branch_id"),
            "i0_fusion_ref": _v1_ref(i0["fusion"], "receipt_id"),
            "i0_quant_branch_ref": _v1_ref(i0["quant_branch"], "branch_id"),
            "i0_runtime_ref": _v1_ref(i0["runtime"], "runtime_receipt_id"),
            "industry_component_ref": (
                None
                if validated_industry_component is None
                else content_ref(
                    validated_industry_component,
                    identity_field="component_receipt_id",
                )
            ),
            "industry_identity_ref": content_ref(industry, identity_field="evaluation_id"),
            "industry_state": industry["state"],
            "overall_risk": overall_risk,
            "policy_independent_hard_veto_codes": vetoes,
            "quant_branch_ref": content_ref(quant, identity_field="quant_branch_id"),
            "quant_percentile": quant["percentile"],
            "quant_pool_ref": quant["pool_ref"],
            "quant_score": quant["score"],
            "r22_envelope_ref": _v1_ref(r22["envelope"], "envelope_id"),
            "r22_hypothesis_evaluation_ref": _v1_ref(r22["evaluation"], "receipt_id"),
            "r22_hypothesis_status": r22["status"],
            "r22_main_ref": _v1_ref(r22["main"], "evaluation_id"),
            "r22_preregistered": r22["preregistered"],
            "readiness_ref": content_ref(_readiness, identity_field="readiness_id"),
            "risk_rows": normalized_risk,
            "run_id": run,
            "subject_binding_ref": content_ref(binding, identity_field="binding_id"),
            "theme_component_ref": (
                None
                if validated_theme_component is None
                else content_ref(
                    validated_theme_component,
                    identity_field="component_receipt_id",
                )
            ),
            "theme_exposure_ref": content_ref(theme, identity_field="exposure_receipt_id"),
            "theme_state": theme["status"],
            "v2_manifest_ref": binding["v2_manifest_ref"],
            "version": EVIDENCE_GRAPH_V2_VERSION,
        },
        identity_field="graph_id",
    )


@decision_contract
def validate_evidence_graph_v2(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="graph_id")
    require_exact_keys(row, _GRAPH_FIELDS, label="EvidenceGraphV2")
    expected = build_evidence_graph_v2(**closure)
    if row != expected or row["version"] != EVIDENCE_GRAPH_V2_VERSION:
        _fail("EvidenceGraphV2 replay mismatch")
    if canonical_bytes(row) != canonical_bytes(expected):
        _fail("EvidenceGraphV2 byte replay mismatch")
    return row


__all__ = [
    "EVIDENCE_GRAPH_V2_VERSION",
    "I0_REPLAY_INPUT_FIELDS",
    "build_evidence_graph_v2",
    "validate_evidence_graph_v2",
]
