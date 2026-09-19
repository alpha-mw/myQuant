"""Shared native research projections with explicit inputs and read-only resolvers."""

from dataclasses import dataclass
from pathlib import Path
from quant_investor.cli import unified
from quant_investor.intelligence.daily_evidence import (
    build_source_bound_economic_exposure_projection,
    build_fundamental_assessments_from_frame,
)
from quant_investor.intelligence.theme_governance import (
    build_unverified_economic_exposure_projection,
)
from .dependency_diagnostics import DependencyInputError
from .daily_contract import NodeState, ContractError
from quant_investor.intelligence.low_frequency import freshness_profile, fundamental_freshness
from quant_investor.intelligence.theme_sources import split_theme_source, descriptor_refs
from quant_investor.intelligence.pcb_ai_hardware import (
    FOCUS_COMPANIES,
    build_focus_membership,
    build_focus_evidence,
    partition_exposure_evidence,
    append_focus_artifacts,
)
from .exposure_completion import (
    optional_exposure_completion,
    validate_optional_exposure_completion,
    validate_optional_focus_completion,
)


@dataclass(frozen=True)
class Projected:
    artifacts: list[dict]
    state: NodeState


def project_research_source(
    *,
    node,
    recipe,
    companies,
    source_document,
    source_file,
    workspace,
    theme=None,
    focus_membership=None,
):
    """Rebuild using native builders; callbacks may read only exact original refs."""
    if node in {"industry", "theme"}:
        field = "industry_source" if node == "industry" else "theme_source"
        values = {"as_of": recipe["as_of"], "policy": recipe["policy"], field: recipe[field]}
        native = (
            unified._daily_industry_projection
            if node == "industry"
            else unified._daily_theme_projection
        )
        artifact = native(values, companies, source_document)
        if artifact is None:
            return None
        artifacts = [artifact]
        if node == "theme" and recipe.get("focus_context") is not None:
            pool_source, focus_source, enabled = split_theme_source(recipe[field])
            if not enabled:
                raise ContractError("FOCUS_THEME_SOURCE_VERSION_MISMATCH")
            focus = unified._daily_theme_projection(
                {
                    "as_of": recipe["as_of"],
                    "policy": recipe["policy"],
                    "theme_source": focus_source,
                },
                list(FOCUS_COMPANIES),
                source_document,
            )
            artifacts.append(
                build_focus_membership(
                    as_of=recipe["as_of"],
                    pool_manifest_ref=recipe["pool_ref"],
                    pit=recipe["focus_context"],
                    pool_theme=artifact,
                    focus_theme=focus,
                    pool_source_refs=descriptor_refs(pool_source),
                    focus_source_refs=descriptor_refs(focus_source),
                )
            )
        return Projected(
            artifacts,
            NodeState.PARTIAL if artifact["payload"]["blocker_codes"] else NodeState.SUCCEEDED,
        )
    if node == "exposure":
        optional_completion = optional_exposure_completion(recipe=recipe)
        if theme is None:
            raise DependencyInputError("EXPOSURE_THEME_UPSTREAM_INCOMPLETE")
        rows = recipe["exposure_rows"]
        evidence = unified._daily_exposure_evidence(
            [] if rows is None else rows, recipe, source_file
        )
        focus_evidence = []
        if recipe.get("focus_context") is not None:
            evidence, focus_evidence = partition_exposure_evidence(evidence, companies)
        if evidence:
            projection, _ = build_source_bound_economic_exposure_projection(
                as_of=recipe["as_of"],
                daily_policy=recipe["policy"],
                theme_projection=theme,
                evidence=evidence,
            )
        else:
            projection = build_unverified_economic_exposure_projection(
                as_of=recipe["as_of"], daily_policy=recipe["policy"], theme_projection=theme
            )
        artifacts = [projection, *evidence]
        partial = bool(projection["payload"]["blocker_codes"])
        if optional_completion:
            validate_optional_exposure_completion(
                companies=companies,
                theme=theme,
                projection=projection,
                daily_policy=recipe["policy"],
                evidence=evidence,
            )
        if recipe.get("focus_context") is not None:
            if focus_membership is None:
                raise DependencyInputError("FOCUS_MEMBERSHIP_UPSTREAM_MISSING")
            pool_source, focus_source, enabled = split_theme_source(recipe["focus_theme_source"])
            if not enabled:
                raise ContractError("FOCUS_THEME_SOURCE_VERSION_MISMATCH")
            focus = unified._daily_theme_projection(
                {
                    "as_of": recipe["as_of"],
                    "policy": recipe["policy"],
                    "theme_source": focus_source,
                },
                list(FOCUS_COMPANIES),
                source_document,
            )
            rebuilt = build_focus_membership(
                as_of=recipe["as_of"],
                pool_manifest_ref=recipe["pool_ref"],
                pit=recipe["focus_context"],
                pool_theme=theme,
                focus_theme=focus,
                pool_source_refs=descriptor_refs(pool_source),
                focus_source_refs=descriptor_refs(focus_source),
            )
            if rebuilt != focus_membership:
                raise ContractError("FOCUS_MEMBERSHIP_UPSTREAM_REPLAY_MISMATCH")
            industry = unified._daily_industry_projection(
                {"as_of": recipe["as_of"], "industry_source": recipe["focus_industry_source"]},
                list(FOCUS_COMPANIES),
                source_document,
            )
            report = build_focus_evidence(
                membership=rebuilt,
                pit=recipe["focus_context"],
                focus_theme=focus,
                industry=industry,
                evidence=focus_evidence,
                industry_source_refs=descriptor_refs(recipe["focus_industry_source"]),
                daily_policy=recipe["policy"],
            )
            partial |= report["payload"]["completion_state"] != "SUCCEEDED"
            if optional_completion:
                validate_optional_focus_completion(
                    report=report,
                    membership=rebuilt,
                    pit=recipe["focus_context"],
                    focus_theme=focus,
                    industry=industry,
                    evidence=focus_evidence,
                    industry_source_refs=descriptor_refs(recipe["focus_industry_source"]),
                    daily_policy=recipe["policy"],
                )
            artifacts = append_focus_artifacts(artifacts, report, focus_evidence)
        return Projected(
            artifacts,
            NodeState.PARTIAL if partial and not optional_completion else NodeState.SUCCEEDED,
        )

    if node == "fundamental":
        return _project_fundamental(recipe, companies, source_file, workspace)
    raise ContractError("RESEARCH_PURE_SOURCE_NODE_INVALID")


def _project_fundamental(recipe, companies, source_file, workspace):
    profile = freshness_profile(recipe)
    descriptor = recipe["fundamental_source"]
    assessments, sources, known_at = {}, [], None
    if descriptor is not None:
        frame, source = unified._daily_fundamental_source(
            recipe,
            source_file,
            workspace=Path(workspace),
            decision_as_of=recipe["as_of"],
            include_native_time=profile is not None,
        )
        assessments, sources = build_fundamental_assessments_from_frame(
            frame=frame,
            companies=companies,
            source_path=source["path"],
            source_sha256=source["sha256"],
            source_available_at=source["available_at"],
            as_of=recipe["as_of"],
            industry_assessments={},
            theme_assessments={},
        )
        # Legacy fixtures cannot claim native provenance that they do not carry.
        known_at = source.get("native_known_at") or source["available_at"]
    missing = len(assessments) != len(companies) or any(
        assessment["payload"]["blocker_codes"] for assessment in assessments.values()
    )
    if profile is not None:
        report = fundamental_freshness(
            companies=companies,
            as_of=recipe["as_of"],
            descriptor=descriptor,
            sources=sources,
            assessments=assessments,
            known_at=known_at,
        )
        missing = bool(report["payload"]["critical_missing_codes"])
        sources = [*sources, report]
    if not sources:
        return None
    # Context-dependent assessment IDs are emitted only by native Decision.
    return Projected(sources, NodeState.PARTIAL if missing else NodeState.SUCCEEDED)
