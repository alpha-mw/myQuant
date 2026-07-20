"""Recomputed, nonauthorizing v16 Codex IC source status.

This module validates sealed Menu/Stage2 artifacts against the recomputed
full-union posterior.  It deliberately does not consume legacy review state,
capital maps, caller authorization booleans, or human authorization receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import ValidationError

from quant_investor.codex_review.models import (
    MenuSeal,
    Stage2Request,
    Stage2Response,
    StrictModel,
)
from quant_investor.codex_review.storage import (
    CONTROL_MAX_BYTES,
    REQUEST_MAX_BYTES,
    RESPONSE_MAX_BYTES,
    canonical_json_bytes as review_canonical_json_bytes,
    parse_strict_json_bytes,
    sha256_bytes as review_sha256_bytes,
)
from quant_investor.v16.candidate_pipeline import (
    PosteriorMenuItem,
    Stage2Decision,
    validate_stage2_portfolio,
)

from .codex_authority_plan_v2 import (
    CODEX_AUTHORITY_PLAN_SCHEMA,
    CODEX_IC_STATUS_SCHEMA,
    MENU_REQUIREMENTS,
    MENU_SCHEMA,
    PRIVATE_ROOT_POLICY,
    STAGE2_REQUEST_SCHEMA,
    STAGE2_RESPONSE_SCHEMA,
    CodexAuthorityPlanEvidenceBundleV2,
)
from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    encode_f64,
    seal_semantic,
    validate_semantic_seal,
)
from .posterior import (
    FULL_UNION_POSTERIOR_SCHEMA,
    BoundReviewArtifact,
    Stage1ReviewBinding,
    validate_full_union_posterior_evidence,
)
from .posterior_runtime import PosteriorRuntimeArtifacts

CODEX_IC_SOURCE_BLOCKERS = (
    "codex_authority_v2_disconnected_from_authorizing_consumers",
    "codex_ic_source_recomputation_incomplete",
    "codex_plan_precommit_time_not_independently_attested",
)
_RESPONSE_BINDING_FIELDS = (
    "run_id",
    "stage",
    "git_sha",
    "config_path",
    "config_sha256",
    "prompt_path",
    "prompt_sha256",
    "model_id",
    "model_sha256",
    "pit_pointer_path",
    "pit_pointer_sha256",
    "predecessor_sha256",
    "decision_cutoff_at",
    "expires_at",
    "request_sha256",
)

_ReviewModelT = TypeVar("_ReviewModelT", bound=StrictModel)


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _matches_planned_reference(
    reference: EvidenceRef,
    planned: Mapping[str, Any],
) -> bool:
    return (
        reference.absolute_path == planned["absolute_path"]
        and reference.artifact_schema == planned["artifact_schema"]
        and reference.root_policy == planned["root_policy"]
    )


def _symbol_set_sha256(symbols: list[str]) -> str:
    normalized = sorted(set(str(item).strip() for item in symbols))
    encoded = review_canonical_json_bytes(normalized)
    if encoded.endswith(b"\n"):
        encoded = encoded[:-1]
    return review_sha256_bytes(encoded)


def _parse_review_artifact(
    artifact: BoundReviewArtifact,
    *,
    model_type: type[_ReviewModelT],
    max_bytes: int,
    digest_field: str,
) -> _ReviewModelT:
    if not isinstance(artifact, BoundReviewArtifact):
        raise EvidenceV2Error("Codex IC source requires a bound review artifact")
    expected_schema = model_type.model_fields["schema_version"].default
    if (
        artifact.reference.artifact_schema != expected_schema
        or artifact.reference.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("Codex IC review artifact reference is invalid")
    try:
        value = parse_strict_json_bytes(artifact.payload, max_bytes=max_bytes)
    except ValueError as exc:
        raise EvidenceV2Error(str(exc)) from exc
    if review_canonical_json_bytes(value) != artifact.payload:
        raise EvidenceV2Error("Codex IC review artifact bytes are not canonical")
    try:
        model = model_type.model_validate(value)
    except ValidationError as exc:
        raise EvidenceV2Error(str(exc)) from exc
    normalized = model.model_dump(mode="json")
    if review_canonical_json_bytes(normalized) != artifact.payload:
        raise EvidenceV2Error("Codex IC review model serialization drift")
    supplied = str(normalized.pop(digest_field, "")).lower()
    expected = review_sha256_bytes(review_canonical_json_bytes(normalized))
    if supplied != expected:
        raise EvidenceV2Error(f"Codex IC {digest_field} mismatch")
    if artifact.reference.semantic_sha256 != supplied:
        raise EvidenceV2Error("Codex IC semantic SHA does not bind the model seal")
    return model


@dataclass(frozen=True)
class FullUnionPosteriorEvidenceBundleV2:
    posterior: BoundCanonicalArtifact
    stage1_binding: Stage1ReviewBinding
    runtime_artifacts: PosteriorRuntimeArtifacts
    cost_model: BoundCanonicalArtifact
    formal_branch_artifacts: tuple[BoundCanonicalArtifact, ...]
    cost_artifacts: tuple[BoundCanonicalArtifact, ...]

    def read(self) -> dict[str, Any]:
        if (
            not isinstance(self.posterior, BoundCanonicalArtifact)
            or self.posterior.reference.artifact_schema
            != FULL_UNION_POSTERIOR_SCHEMA
            or self.posterior.reference.root_policy != PRIVATE_ROOT_POLICY
            or not isinstance(self.stage1_binding, Stage1ReviewBinding)
            or not isinstance(self.runtime_artifacts, PosteriorRuntimeArtifacts)
            or not isinstance(self.cost_model, BoundCanonicalArtifact)
            or type(self.formal_branch_artifacts) is not tuple
            or not self.formal_branch_artifacts
            or any(
                not isinstance(item, BoundCanonicalArtifact)
                for item in self.formal_branch_artifacts
            )
            or type(self.cost_artifacts) is not tuple
            or not self.cost_artifacts
            or any(
                not isinstance(item, BoundCanonicalArtifact)
                for item in self.cost_artifacts
            )
        ):
            raise EvidenceV2Error("full-union posterior source bundle is invalid")
        return validate_full_union_posterior_evidence(
            self.posterior.read(),
            stage1_binding=self.stage1_binding,
            runtime_artifacts=self.runtime_artifacts,
            cost_model=self.cost_model,
            formal_branch_artifacts=self.formal_branch_artifacts,
            cost_artifacts=self.cost_artifacts,
        )


@dataclass(frozen=True)
class ValidatedCodexICSourceV2:
    plan: dict[str, Any]
    posterior: dict[str, Any]
    menu: MenuSeal
    request: Stage2Request
    response: Stage2Response


@dataclass(frozen=True)
class CodexICSourceEvidenceBundleV2:
    plan: CodexAuthorityPlanEvidenceBundleV2
    posterior: FullUnionPosteriorEvidenceBundleV2
    menu: BoundReviewArtifact
    stage2_request: BoundReviewArtifact
    stage2_response: BoundReviewArtifact

    def read(self) -> ValidatedCodexICSourceV2:
        if not isinstance(self.plan, CodexAuthorityPlanEvidenceBundleV2) or not isinstance(
            self.posterior,
            FullUnionPosteriorEvidenceBundleV2,
        ):
            raise EvidenceV2Error("Codex IC plan/posterior bundle types are invalid")
        plan = self.plan.read()
        posterior = self.posterior.read()
        readiness_v3 = self.plan.readiness_v3.read()
        if (
            self.plan.plan.reference.artifact_schema != CODEX_AUTHORITY_PLAN_SCHEMA
            or self.plan.full_union_posterior.reference
            != self.posterior.posterior.reference
            or plan["full_union_posterior_ref"]
            != self.posterior.posterior.reference.to_dict()
            or plan["protocol_attempt_id"] != posterior["protocol_attempt_id"]
            or plan["run_id"] != posterior["run_id"]
            or readiness_v3.get("run_id") != plan["run_id"]
        ):
            raise EvidenceV2Error("Codex IC plan/posterior/readiness lineage drift")

        menu = _parse_review_artifact(
            self.menu,
            model_type=MenuSeal,
            max_bytes=CONTROL_MAX_BYTES,
            digest_field="menu_sha256",
        )
        request = _parse_review_artifact(
            self.stage2_request,
            model_type=Stage2Request,
            max_bytes=REQUEST_MAX_BYTES,
            digest_field="request_sha256",
        )
        response = _parse_review_artifact(
            self.stage2_response,
            model_type=Stage2Response,
            max_bytes=RESPONSE_MAX_BYTES,
            digest_field="response_sha256",
        )
        planned = plan["planned_artifacts"]
        for key, artifact in (
            ("menu", self.menu),
            ("stage2_request", self.stage2_request),
            ("stage2_response", self.stage2_response),
        ):
            if not _matches_planned_reference(artifact.reference, planned[key]):
                raise EvidenceV2Error(f"Codex IC artifact path drifts from plan: {key}")

        _validate_menu_against_posterior(menu=menu, posterior=posterior)
        _validate_stage2_chain(
            menu=menu,
            request=request,
            response=response,
            posterior=posterior,
        )
        return ValidatedCodexICSourceV2(
            plan=plan,
            posterior=posterior,
            menu=menu,
            request=request,
            response=response,
        )


def _validate_menu_against_posterior(
    *,
    menu: MenuSeal,
    posterior: Mapping[str, Any],
) -> None:
    if menu.run_id != posterior["run_id"] or menu.symbols != posterior["menu_symbols"]:
        raise EvidenceV2Error("Codex menu symbol/run binding drifts from posterior")
    stage1_response_ref = EvidenceRef.from_dict(posterior["stage1_response_ref"])
    if menu.stage1_response_sha256 != stage1_response_ref.semantic_sha256:
        raise EvidenceV2Error("Codex menu Stage1 predecessor drifts from posterior")
    rows = posterior["posteriors"]
    if not isinstance(rows, list):
        raise EvidenceV2Error("posterior rows must be a list")
    by_symbol = {str(item.get("symbol")): item for item in rows if isinstance(item, Mapping)}
    if len(by_symbol) != len(rows) or any(symbol not in by_symbol for symbol in menu.symbols):
        raise EvidenceV2Error("Codex menu posterior rows are missing or duplicated")
    for item in menu.items:
        source = by_symbol[item.symbol]
        for field in (
            "posterior_win_rate",
            "posterior_expected_alpha",
            "posterior_edge_after_costs",
        ):
            native = getattr(item, field)
            encoded = None if native is None else encode_f64(native)
            if source[field] != encoded:
                raise EvidenceV2Error(f"Codex menu posterior value drift: {item.symbol}:{field}")
        source_branches = source["branch_evidence"]
        if len(source_branches) != len(item.branch_evidence):
            raise EvidenceV2Error("Codex menu branch evidence count drift")
        for native, branch in zip(item.branch_evidence, source_branches, strict=True):
            if (
                native.branch != branch["branch"]
                or encode_f64(native.raw_score) != branch["raw_score"]
                or encode_f64(native.confidence) != branch["confidence"]
                or encode_f64(native.calibrated_probability)
                != branch["calibrated_probability"]
                or native.evidence_ids != branch["evidence_ids"]
            ):
                raise EvidenceV2Error(
                    f"Codex menu branch evidence drift: {item.symbol}:{native.branch}"
                )
        retrieval = [
            {
                "branch": advisory.branch,
                "supporting_fact_ids": advisory.supporting_fact_ids,
                "contradicting_fact_ids": advisory.contradicting_fact_ids,
                "conflict_note": advisory.conflict_note or None,
            }
            for advisory in item.retrieval_advisory
        ]
        if retrieval != source["retrieval_advisory"]:
            raise EvidenceV2Error(f"Codex menu retrieval drift: {item.symbol}")


def _validate_stage2_chain(
    *,
    menu: MenuSeal,
    request: Stage2Request,
    response: Stage2Response,
    posterior: Mapping[str, Any],
) -> None:
    if (
        request.run_id != menu.run_id
        or request.menu_sha256 != menu.menu_sha256
        or request.symbol_set != menu.symbols
        or request.existing_weights != menu.existing_weights
        or [item.model_dump(mode="json") for item in request.menu]
        != [item.model_dump(mode="json") for item in menu.items]
    ):
        raise EvidenceV2Error("Stage2 request drifts from the sealed menu")
    if request.predecessor_sha256 != posterior["stage1_response_ref"][
        "semantic_sha256"
    ]:
        raise EvidenceV2Error("Stage2 request predecessor drifts from posterior")
    if not request.decision_cutoff_at <= menu.sealed_at < request.expires_at:
        raise EvidenceV2Error("Codex menu timestamp is outside the internal request window")
    if request.model_sha256 != review_sha256_bytes(request.model_id.encode("utf-8")):
        raise EvidenceV2Error("Stage2 model SHA mismatch")
    if request.symbol_set_sha256 != _symbol_set_sha256(request.symbol_set):
        raise EvidenceV2Error("Stage2 symbol-set SHA mismatch")
    for field in _RESPONSE_BINDING_FIELDS:
        if getattr(request, field) != getattr(response, field):
            raise EvidenceV2Error(f"Stage2 response binding mismatch: {field}")
    if (
        response.symbol_set != request.symbol_set
        or response.symbol_set_sha256 != request.symbol_set_sha256
        or response.menu_sha256 != request.menu_sha256
    ):
        raise EvidenceV2Error("Stage2 response symbol/menu binding mismatch")
    menu_projection = [
        PosteriorMenuItem(
            symbol=item.symbol,
            posterior_win_rate=item.posterior_win_rate,
            posterior_expected_alpha=item.posterior_expected_alpha,
            posterior_edge_after_costs=item.posterior_edge_after_costs,
        )
        for item in request.menu
    ]
    decisions = [
        Stage2Decision(
            symbol=item.symbol,
            action=item.action.value,
            selected_for_portfolio=item.selected_for_portfolio,
            target_weight=item.target_weight,
            rationale=item.rationale,
            risk_acceptance_rationale=item.risk_acceptance_rationale or None,
        )
        for item in response.verdicts
    ]
    try:
        validate_stage2_portfolio(
            menu_projection,
            decisions,
            cash_ratio=response.cash_ratio,
            existing_weights=request.existing_weights,
            severe_risk_symbols={
                item.symbol
                for item in request.menu
                if item.risk_advisory.severity in {"high", "extreme"}
            },
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceV2Error(str(exc)) from exc


def _blocker_projection() -> tuple[list[str], list[dict[str, str]]]:
    rows = [
        {
            "blocker": f"codex_requirement_unsupported:requirement={requirement}",
            "source": f"codex_ic_source:requirement:{requirement}",
        }
        for requirement in MENU_REQUIREMENTS
    ]
    rows.extend(
        {"blocker": blocker, "source": "codex_ic_source_status"}
        for blocker in CODEX_IC_SOURCE_BLOCKERS
    )
    rows.sort(key=lambda item: (item["blocker"], item["source"]))
    return sorted({item["blocker"] for item in rows}), rows


def build_codex_ic_source_status_v2(
    *,
    evidence: CodexICSourceEvidenceBundleV2,
) -> dict[str, Any]:
    if not isinstance(evidence, CodexICSourceEvidenceBundleV2):
        raise EvidenceV2Error("Codex IC source status requires its typed evidence bundle")
    validated = evidence.read()
    verdict_by_symbol = {item.symbol: item for item in validated.response.verdicts}
    allocations = []
    for menu_item in validated.menu.items:
        verdict = verdict_by_symbol[menu_item.symbol]
        risk_rationale = verdict.risk_acceptance_rationale or None
        allocations.append(
            {
                "symbol": menu_item.symbol,
                "action": verdict.action.value,
                "selected_for_portfolio": verdict.selected_for_portfolio,
                "existing_weight": encode_f64(
                    validated.menu.existing_weights[menu_item.symbol]
                ),
                "target_weight": encode_f64(verdict.target_weight),
                "rationale_sha256": hashlib.sha256(
                    verdict.rationale.encode("utf-8")
                ).hexdigest(),
                "severe_risk_count": len(verdict.severe_risks),
                "risk_acceptance_rationale_sha256": (
                    hashlib.sha256(risk_rationale.encode("utf-8")).hexdigest()
                    if risk_rationale is not None
                    else None
                ),
            }
        )
    blockers, blocker_sources = _blocker_projection()
    total = validated.response.cash_ratio + sum(
        item.target_weight for item in validated.response.verdicts
    )
    return seal_semantic(
        {
            "schema_version": CODEX_IC_STATUS_SCHEMA,
            "protocol_attempt_id": validated.plan["protocol_attempt_id"],
            "run_id": validated.plan["run_id"],
            "source_plan_ref": evidence.plan.plan.reference.to_dict(),
            "readiness_v3_ref": evidence.plan.readiness_v3.reference.to_dict(),
            "full_union_posterior_ref": evidence.posterior.posterior.reference.to_dict(),
            "stage1_response_ref": validated.posterior["stage1_response_ref"],
            "menu_ref": evidence.menu.reference.to_dict(),
            "stage2_request_ref": evidence.stage2_request.reference.to_dict(),
            "stage2_response_ref": evidence.stage2_response.reference.to_dict(),
            "menu_symbols": list(validated.menu.symbols),
            "allocations": allocations,
            "cash_ratio": encode_f64(validated.response.cash_ratio),
            "positive_weight_count": sum(
                item.target_weight > 1e-6 for item in validated.response.verdicts
            ),
            "target_plus_cash": encode_f64(total),
            "posterior_recomputed_from_bound_sources": True,
            "menu_binding_validated": True,
            "stage2_allocation_validated": True,
            "risk_advisor_role": "advisory_only",
            "unsupported_requirement_ids": list(MENU_REQUIREMENTS),
            "source_recomputation_complete": False,
            "readiness_status": "no_new_risk",
            "blockers": blockers,
            "blocker_sources": blocker_sources,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_codex_ic_source_status_v2(
    value: Mapping[str, Any],
    *,
    evidence: CodexICSourceEvidenceBundleV2,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    payload = _exact(
        payload,
        {
            "schema_version",
            "protocol_attempt_id",
            "run_id",
            "source_plan_ref",
            "readiness_v3_ref",
            "full_union_posterior_ref",
            "stage1_response_ref",
            "menu_ref",
            "stage2_request_ref",
            "stage2_response_ref",
            "menu_symbols",
            "allocations",
            "cash_ratio",
            "positive_weight_count",
            "target_plus_cash",
            "posterior_recomputed_from_bound_sources",
            "menu_binding_validated",
            "stage2_allocation_validated",
            "risk_advisor_role",
            "unsupported_requirement_ids",
            "source_recomputation_complete",
            "readiness_status",
            "blockers",
            "blocker_sources",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "semantic_sha256",
        },
        label="Codex IC source status v2",
    )
    if payload["schema_version"] != CODEX_IC_STATUS_SCHEMA:
        raise EvidenceV2Error("Codex IC source status v2 schema mismatch")
    rebuilt = build_codex_ic_source_status_v2(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("Codex IC source status v2 drifts from evidence")
    return payload


__all__ = [
    "CODEX_IC_SOURCE_BLOCKERS",
    "CodexICSourceEvidenceBundleV2",
    "FullUnionPosteriorEvidenceBundleV2",
    "ValidatedCodexICSourceV2",
    "build_codex_ic_source_status_v2",
    "validate_codex_ic_source_status_v2",
]
