"""Canonical, nonauthorizing candidate report projected from readiness-v4."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from .codex_ic_source_v2 import ValidatedCodexICSourceV2
from .contracts import (
    EvidenceV2Error,
    decode_f64,
    seal_semantic,
    validate_semantic_seal,
)
from .publication_plan_v2 import (
    CANDIDATE_REPORT_SCHEMA,
    PublicationPlanEvidenceBundleV2,
)
from .readiness_v4 import ValidatedReadinessSourceV4

REPORT_FOUNDATION_BLOCKERS = (
    "candidate_report_not_production_authority",
    "codex_requirement_unsupported:requirement="
    "candidate_report_publication_attestation_protocol",
)


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _f64(value: Any, *, label: str) -> str:
    decode_f64(value, label=label)
    return str(value)


@dataclass(frozen=True)
class ValidatedCandidateReportSourceV2:
    plan: dict[str, Any]
    readiness: dict[str, Any]
    readiness_sources: ValidatedReadinessSourceV4
    ic: ValidatedCodexICSourceV2


@dataclass(frozen=True)
class CandidateReportSourceEvidenceBundleV2:
    publication_plan: PublicationPlanEvidenceBundleV2

    def read(self) -> ValidatedCandidateReportSourceV2:
        if not isinstance(self.publication_plan, PublicationPlanEvidenceBundleV2):
            raise EvidenceV2Error("candidate report requires a publication plan bundle")
        plan, readiness = self.publication_plan.read()
        readiness_sources = self.publication_plan.readiness_evidence.read()
        ic = self.publication_plan.readiness_evidence.ic_evidence.read()
        if (
            readiness_sources.plan["protocol_attempt_id"]
            != plan["protocol_attempt_id"]
            or readiness_sources.plan["run_id"] != plan["run_id"]
            or ic.plan["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or ic.plan["run_id"] != plan["run_id"]
        ):
            raise EvidenceV2Error("candidate report source lineage drift")
        return ValidatedCandidateReportSourceV2(
            plan=plan,
            readiness=readiness,
            readiness_sources=readiness_sources,
            ic=ic,
        )


def _blocker_projection(
    readiness: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    source_rows = readiness.get("blocker_sources")
    if not isinstance(source_rows, list):
        raise EvidenceV2Error("readiness-v4 blocker sources must be a list")
    rows: list[dict[str, str]] = []
    for item in source_rows:
        if not isinstance(item, Mapping) or set(item) != {"blocker", "source"}:
            raise EvidenceV2Error("readiness-v4 blocker source row is invalid")
        rows.append(
            {
                "blocker": str(item["blocker"]),
                "source": f"readiness_v4:{item['source']}",
            }
        )
    rows.extend(
        {"blocker": blocker, "source": "candidate_report_source_v2"}
        for blocker in REPORT_FOUNDATION_BLOCKERS
    )
    if any(not item["blocker"] or not item["source"] for item in rows):
        raise EvidenceV2Error("candidate report blocker source row is empty")
    rows.sort(key=lambda item: (item["blocker"], item["source"]))
    blockers = sorted({item["blocker"] for item in rows})
    inherited = {str(item) for item in readiness.get("blockers", [])}
    if not inherited.issubset(blockers):
        raise EvidenceV2Error("candidate report blockers are not monotonic")
    return blockers, rows


def _menu_projection(validated: ValidatedCandidateReportSourceV2) -> list[dict[str, Any]]:
    posterior_rows = validated.ic.posterior["posteriors"]
    by_symbol = {item["symbol"]: item for item in posterior_rows}
    allocation_rows = validated.readiness_sources.ic_status["allocations"]
    allocation_by_symbol = {item["symbol"]: item for item in allocation_rows}
    symbols = validated.readiness_sources.ic_status["menu_symbols"]
    if (
        len(by_symbol) != len(posterior_rows)
        or len(allocation_by_symbol) != len(allocation_rows)
        or any(symbol not in by_symbol or symbol not in allocation_by_symbol for symbol in symbols)
    ):
        raise EvidenceV2Error("candidate report menu source coverage drift")
    result: list[dict[str, Any]] = []
    for symbol in symbols:
        posterior = by_symbol[symbol]
        allocation = allocation_by_symbol[symbol]
        branches = []
        for branch in posterior["branch_evidence"]:
            branches.append(
                {
                    "branch": str(branch["branch"]),
                    "raw_score": _f64(
                        branch["raw_score"],
                        label=f"{symbol}.raw_score",
                    ),
                    "confidence": _f64(
                        branch["confidence"],
                        label=f"{symbol}.confidence",
                    ),
                    "calibrated_probability": _f64(
                        branch["calibrated_probability"],
                        label=f"{symbol}.calibrated_probability",
                    ),
                    "evidence_ids": list(branch["evidence_ids"]),
                    "source_ref": dict(branch["source_ref"]),
                    "model_bundle_ref": dict(branch["model_bundle_ref"]),
                }
            )
        result.append(
            {
                "symbol": symbol,
                "posterior": {
                    "win_rate": _f64(
                        posterior["posterior_win_rate"],
                        label=f"{symbol}.posterior_win_rate",
                    ),
                    "expected_alpha": _f64(
                        posterior["posterior_expected_alpha"],
                        label=f"{symbol}.posterior_expected_alpha",
                    ),
                    "edge_after_costs": _f64(
                        posterior["posterior_edge_after_costs"],
                        label=f"{symbol}.posterior_edge_after_costs",
                    ),
                    "win_rate_interval_90": [
                        _f64(value, label=f"{symbol}.win_rate_interval_90")
                        for value in posterior["posterior_win_rate_interval_90"]
                    ],
                    "expected_alpha_interval_90": [
                        _f64(
                            value,
                            label=f"{symbol}.expected_alpha_interval_90",
                        )
                        for value in posterior["posterior_expected_alpha_interval_90"]
                    ],
                    "cost_input_ref": dict(posterior["cost_input_ref"]),
                },
                "branch_evidence": branches,
                "retrieval_advisory": [
                    {
                        "branch": str(item["branch"]),
                        "supporting_fact_ids": list(item["supporting_fact_ids"]),
                        "contradicting_fact_ids": list(
                            item["contradicting_fact_ids"]
                        ),
                        "conflict_note": item["conflict_note"],
                    }
                    for item in posterior["retrieval_advisory"]
                ],
                "allocation": {
                    "action": str(allocation["action"]),
                    "selected_for_portfolio": bool(
                        allocation["selected_for_portfolio"]
                    ),
                    "existing_weight": _f64(
                        allocation["existing_weight"],
                        label=f"{symbol}.existing_weight",
                    ),
                    "target_weight": _f64(
                        allocation["target_weight"],
                        label=f"{symbol}.target_weight",
                    ),
                    "rationale_sha256": str(allocation["rationale_sha256"]),
                    "severe_risk_count": int(allocation["severe_risk_count"]),
                    "risk_acceptance_rationale_sha256": allocation[
                        "risk_acceptance_rationale_sha256"
                    ],
                },
            }
        )
    return result


def build_candidate_report_source_v2(
    *,
    evidence: CandidateReportSourceEvidenceBundleV2,
) -> dict[str, Any]:
    if not isinstance(evidence, CandidateReportSourceEvidenceBundleV2):
        raise EvidenceV2Error("candidate report requires its typed evidence bundle")
    validated = evidence.read()
    blockers, blocker_sources = _blocker_projection(validated.readiness)
    ic_status = validated.readiness_sources.ic_status
    execution_status = validated.readiness_sources.execution_status
    handoff_status = validated.readiness_sources.handoff_status
    return seal_semantic(
        {
            "schema_version": CANDIDATE_REPORT_SCHEMA,
            "architecture_version": validated.readiness["architecture_version"],
            "protocol_attempt_id": validated.plan["protocol_attempt_id"],
            "run_id": validated.plan["run_id"],
            "generated_at": validated.readiness["generated_at"],
            "analysis_trade_date": validated.readiness["analysis_trade_date"],
            "publication_plan_ref": evidence.publication_plan.plan.reference.to_dict(),
            "readiness_v4_ref": evidence.publication_plan.readiness_v4.reference.to_dict(),
            "full_union_posterior_ref": ic_status["full_union_posterior_ref"],
            "codex_ic_status_ref": evidence.publication_plan.readiness_evidence.ic_status.reference.to_dict(),
            "execution_status_ref": evidence.publication_plan.readiness_evidence.execution_status.reference.to_dict(),
            "handoff_status_ref": evidence.publication_plan.readiness_evidence.handoff_status.reference.to_dict(),
            "formal_branches": validated.readiness["formal_branches"],
            "retrieval_role": validated.readiness["retrieval_role"],
            "risk_advisor_role": validated.readiness["risk_advisor_role"],
            "menu": _menu_projection(validated),
            "cash_ratio": _f64(ic_status["cash_ratio"], label="cash_ratio"),
            "positive_weight_count": int(ic_status["positive_weight_count"]),
            "target_plus_cash": _f64(
                ic_status["target_plus_cash"],
                label="target_plus_cash",
            ),
            "execution_artifact_role": execution_status["artifact_role"],
            "handoff_artifact_role": handoff_status["artifact_role"],
            "projection_validation_complete": True,
            "authority_source_complete": False,
            "readiness_status": "no_new_risk",
            "blockers": blockers,
            "blocker_sources": blocker_sources,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
            "production_pointer_switch_authorized": False,
            "dashboard_activation_authorized": False,
            "broker_side_effects": False,
        }
    )


def validate_candidate_report_source_v2(
    value: Mapping[str, Any],
    *,
    evidence: CandidateReportSourceEvidenceBundleV2,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    payload = _exact(
        payload,
        {
            "schema_version",
            "architecture_version",
            "protocol_attempt_id",
            "run_id",
            "generated_at",
            "analysis_trade_date",
            "publication_plan_ref",
            "readiness_v4_ref",
            "full_union_posterior_ref",
            "codex_ic_status_ref",
            "execution_status_ref",
            "handoff_status_ref",
            "formal_branches",
            "retrieval_role",
            "risk_advisor_role",
            "menu",
            "cash_ratio",
            "positive_weight_count",
            "target_plus_cash",
            "execution_artifact_role",
            "handoff_artifact_role",
            "projection_validation_complete",
            "authority_source_complete",
            "readiness_status",
            "blockers",
            "blocker_sources",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "production_pointer_switch_authorized",
            "dashboard_activation_authorized",
            "broker_side_effects",
            "semantic_sha256",
        },
        label="candidate report source v2",
    )
    if payload["schema_version"] != CANDIDATE_REPORT_SCHEMA:
        raise EvidenceV2Error("candidate report source v2 schema mismatch")
    rebuilt = build_candidate_report_source_v2(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("candidate report source v2 drifts from evidence")
    return payload


__all__ = [
    "REPORT_FOUNDATION_BLOCKERS",
    "CandidateReportSourceEvidenceBundleV2",
    "ValidatedCandidateReportSourceV2",
    "build_candidate_report_source_v2",
    "validate_candidate_report_source_v2",
]
