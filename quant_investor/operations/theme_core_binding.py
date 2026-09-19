"""Derive Theme acquisition scope from original native core/rank verification."""

import hashlib
from pathlib import PurePosixPath
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.market.tushare.theme_capture import _company_keyset
from .core_handoff import inspect_core_handoff
from .core_pool import CoreContext
from .daily_contract import ContractError, validate_ref, utc_stamp
from .production_request import validate_production_request
from .execution_recipe import validate_execution_recipe
from .theme_acquisition import (
    validate_theme_acquisition_policy,
    validate_theme_policy_profile,
    POLICY_SCHEMA_V2,
    POLICY_SCHEMA_V3,
)
from quant_investor.factors.production_pit import read_bound_focus_pit


def bind_theme_acquisition(*, workspace: str, request_ref: dict, core_handoff_ref: dict) -> dict:
    """No supplied company list, plan creation, claim, lock, or provider operation."""
    reader = SecureSystemStorage(workspace)
    observed = {}

    def read(ref):
        checked = validate_ref(ref)
        stored = reader.read_workspace_file_bytes(checked["path"], maximum_bytes=16 * 1024 * 1024)
        if stored.byte_sha256 != checked["sha256"]:
            raise ContractError("THEME_CORE_SOURCE_SHA_MISMATCH")
        if checked["path"] in observed and observed[checked["path"]] != stored.data:
            raise ContractError("THEME_CORE_SOURCE_CHANGED")
        observed[checked["path"]] = stored.data
        return parse_canonical_json_bytes(stored.data)

    request = read(request_ref)
    validate_production_request(request, release_install_ref=request["release_install_ref"])
    if request["action"] != "EXECUTE":
        raise ContractError("THEME_ACQUISITION_EXECUTE_REQUEST_REQUIRED")
    recipe = validate_execution_recipe(read(request["recipe_ref"]), request=request)
    policy_ref = recipe.get("theme_acquisition_ref")
    if policy_ref is None:
        raise ContractError("THEME_ACQUISITION_MODE_REQUIRED")
    policy = validate_theme_acquisition_policy(read(policy_ref))
    validate_theme_policy_profile(recipe, policy)
    timing_fields = {}
    if policy["schema_version"] == POLICY_SCHEMA_V3:
        from .research_timing import validate_research_timing_policy, CURRENT

        timing = recipe["research_timing"]
        timing_policy = validate_research_timing_policy(read(timing["policy_ref"]))
        if timing["mode"] != CURRENT or timing_policy["mode"] != CURRENT:
            raise ContractError("THEME_ACQUISITION_CURRENT_TIMING_REQUIRED")
        timing_fields = {
            "acquisition_deadline": timing["acquisition_deadline"],
            "timing_policy_ref": timing["policy_ref"],
        }
    day = request["target_trade_date"]
    core = inspect_core_handoff(
        workspace=workspace,
        trade_date=day,
        handoff_ref=core_handoff_ref,
        release_ref=recipe["release_ref"],
    )
    terminal_ref = core["node_terminal_refs"]["top100"]
    completed = [read(ref)["finished_at"] for ref in core["node_terminal_refs"].values()]
    core_completed_at = max(completed, key=utc_stamp)
    terminal = read(terminal_ref)
    request_path = str(PurePosixPath(terminal_ref["path"]).parent.parent / "request.json")
    stored_request = reader.read_workspace_file_bytes(request_path, maximum_bytes=1024 * 1024)
    observed[request_path] = stored_request.data
    node_request = parse_canonical_json_bytes(stored_request.data)
    context = CoreContext(
        workspace, day, core["factor_pointer_ref"]["sha256"], recipe["release_ref"]
    )
    context.pointer_ref = dict(core["factor_pointer_ref"])
    arguments = context.pool_arguments(node_request)
    verified_refs = (
        context.pool.verify(**arguments, required_format="TABULAR")
        if policy["schema_version"] in {POLICY_SCHEMA_V2, POLICY_SCHEMA_V3}
        else context.pool.verify(**arguments)
    )
    pool_ref = terminal["output_refs"]["manifest.json"]
    if verified_refs["manifest.json"] != pool_ref:
        raise ContractError("THEME_CORE_POOL_REF_MISMATCH")
    companies = _company_keyset(
        sorted(row["symbol"] for row in arguments["rank"]["payload"]["pool_rows"])
    )
    if len(companies) > policy["maximum_companies"]:
        raise ContractError("THEME_CORE_COMPANY_LIMIT_EXCEEDED")
    focus_fields = {}
    if policy["schema_version"] in {POLICY_SCHEMA_V2, POLICY_SCHEMA_V3}:
        observation = arguments["observations"][0]["payload"]
        focus = read_bound_focus_pit(
            workspace=workspace,
            generation_ref=arguments["rank"]["payload"]["factor_generation_ref"],
            trade_date=day,
            expected_manifest_sha256=observation["pit_manifest_sha256"],
            expected_membership_sha256=observation["pit_membership_sha256"],
        )
        focus_fields = {
            "policy_schema": policy["schema_version"],
            "focus_context": focus,
            "special_company_keyset": focus["company_keyset"],
            "special_company_set_sha256": focus["company_set_sha256"],
            **{
                name: focus[name]
                for name in (
                    "pit_selection_ref",
                    "pit_generation_manifest_ref",
                    "pit_membership_ref",
                )
            },
        }
    context.recheck()
    if (
        inspect_core_handoff(
            workspace=workspace,
            trade_date=day,
            handoff_ref=core_handoff_ref,
            release_ref=recipe["release_ref"],
        )
        != core
    ):
        raise ContractError("THEME_CORE_HANDOFF_CHANGED")
    for path, raw in observed.items():
        if reader.read_workspace_file_bytes(path, maximum_bytes=16 * 1024 * 1024).data != raw:
            raise ContractError("THEME_CORE_SOURCE_CHANGED")
    return {
        **focus_fields,
        "trade_date": day,
        "company_keyset": companies,
        **(timing_fields or {"research_cutoff": recipe["research_sources"]["as_of"]}),
        "core_completed_at": core_completed_at,
        "identity": {
            "request_ref": dict(request_ref),
            "core_handoff_ref": dict(core_handoff_ref),
            "pool_manifest_ref": pool_ref,
            "rank_ref": verified_refs["factor_research_rank.json"],
            "company_set_sha256": hashlib.sha256(canonical_json_bytes(companies)).hexdigest(),
            "acquisition_policy_ref": policy_ref,
            "release_ref": recipe["release_ref"],
        },
    }
