"""Shared exact recipe and output-name reconstruction for production and replay."""

from quant_investor.factors.production_pit import read_bound_focus_pit
from quant_investor.intelligence.theme_sources import split_theme_source
from quant_investor.intelligence.pcb_ai_hardware import MEMBERSHIP_KIND, EVIDENCE_KIND
from .daily_contract import ContractError
from quant_investor.intelligence.low_frequency import FRESHNESS_KIND, FRESHNESS_CONTRACT


def derive_focus_context(*, workspace, request, rank, pool_store):
    _, _, enabled = split_theme_source(request["theme_source"])
    if not enabled:
        return None
    observations = pool_store._observations(rank, None)
    first = observations[0]["payload"]
    return read_bound_focus_pit(
        workspace=workspace,
        generation_ref=rank["payload"]["factor_generation_ref"],
        trade_date=rank["payload"]["signal_date"],
        expected_manifest_sha256=first["pit_manifest_sha256"],
        expected_membership_sha256=first["pit_membership_sha256"],
    )


def source_recipe(
    *,
    node,
    field,
    request,
    pool_ref,
    focus_context,
    freshness_contract=FRESHNESS_CONTRACT,
    completion_policy=None,
):
    recipe = {
        "as_of": request["as_of"],
        "strategy_id": request["strategy_id"],
        "pool_ref": pool_ref,
        "policy": request["policy"],
        field: (
            request[field]
            if node in {"industry", "theme"}
            else (request.get("company_evidence") or {}).get(field)
        ),
    }
    if completion_policy is not None:
        from .exposure_completion import POLICY

        if completion_policy != POLICY:
            raise ContractError("EXPOSURE_COMPLETION_POLICY_INVALID")
        if node == "exposure":
            recipe["source_completion_policy"] = completion_policy
    if focus_context is not None and node in {"theme", "exposure"}:
        recipe["focus_context"] = focus_context
        if node == "exposure":
            recipe["focus_theme_source"] = request["theme_source"]
            recipe["focus_industry_source"] = request["industry_source"]
    if node in {"fundamental", "macro"} and freshness_contract is not None:
        if freshness_contract != FRESHNESS_CONTRACT:
            raise ContractError("LOW_FREQUENCY_CONTRACT_INVALID")
        recipe["freshness_contract"] = freshness_contract
    return recipe


def artifact_output_names(artifacts):
    names = [
        (
            value["kind"]
            if value["kind"] in {MEMBERSHIP_KIND, EVIDENCE_KIND, FRESHNESS_KIND}
            else "artifact" if index == 0 else f"artifact_{index}"
        )
        for index, value in enumerate(artifacts)
    ]
    if len(set(names)) != len(names):
        raise ContractError("RESEARCH_SOURCE_OUTPUT_NAME_CONFLICT")
    return names
