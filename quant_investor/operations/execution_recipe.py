"""Stable pre-maintenance recipe; native leaf validators remain authoritative."""

from pathlib import PurePosixPath
from .daily_contract import ContractError, GRAPH_SHA256, utc_stamp, validate_ref
from .daily_journal import _validate_day

SCHEMA = "cn-daily-execute-recipe.v1"
SCHEMA_V2 = "cn-daily-execute-recipe.v2"
SCHEMA_V3 = "cn-daily-execute-recipe.v3"
SCHEMA_V4 = "cn-daily-execute-recipe.v4"
SCHEMA_V5 = "cn-daily-execute-recipe.v5"
SCHEMA_V6 = "cn-daily-execute-recipe.v6"
FIELDS = frozenset(
    {
        "schema_version",
        "market",
        "strategy_id",
        "target_trade_date",
        "graph_sha256",
        "release_ref",
        "release_install_ref",
        "previous_completion_ref",
        "bootstrap_ref",
        "factor_loop_context_ref",
        "policy_refs",
        "store_preimages",
        "retrospective_ref",
        "dashboard_sources",
        "research_sources",
        "publish_current_dashboard",
    }
)
FIELDS_V2 = FIELDS | {"theme_acquisition_ref"}
FIELDS_V3 = FIELDS_V2 | {"corporate_action_context_ref"}
FIELDS_V4 = FIELDS_V3 | {"dashboard_publication_policy"}
FIELDS_V5 = (FIELDS_V4 - {"corporate_action_context_ref"}) | {
    "corporate_action_template_ref",
    "research_timing",
}
FIELDS_V6 = FIELDS_V5 | {"registered_event_declaration_ref"}


def registered_validation_view(value):
    if type(value) is not dict or set(value) != FIELDS_V6 or value["schema_version"] != SCHEMA_V6:
        raise ContractError("EXECUTE_RECIPE_V6_FIELDS_INVALID")
    validate_ref(value["registered_event_declaration_ref"])
    if value["retrospective_ref"] is not None or value["bootstrap_ref"] is not None:
        raise ContractError("EXECUTE_RECIPE_REGISTERED_ANCHOR_UNSUPPORTED")
    return {
        **{k: v for k, v in value.items() if k != "registered_event_declaration_ref"},
        "schema_version": SCHEMA_V5,
    }


def _refs(value, keys, nullable=()):
    if type(value) is not dict or set(value) != set(keys):
        raise ContractError("EXECUTE_RECIPE_NESTED_FIELDS_INVALID")
    for name, ref in value.items():
        if name in nullable and ref is None:
            continue
        validate_ref(ref)


def validate_execution_recipe(value: dict, *, request: dict) -> dict:
    if type(value) is dict and value.get("schema_version") == SCHEMA_V6:
        validate_execution_recipe(registered_validation_view(value), request=request)
        return value
    if type(value) is dict and value.get("schema_version") == SCHEMA_V5:
        from .research_timing import validate_recipe_timing, recipe_v4_validation_view
        from .production_request import HISTORICAL_SCHEMA

        if set(value) != FIELDS_V5:
            raise ContractError("EXECUTE_RECIPE_V5_FIELDS_INVALID")
        validate_ref(value["corporate_action_template_ref"])
        validate_recipe_timing(
            value,
            historical_request=(
                request["schema_version"] == HISTORICAL_SCHEMA
                if request["action"] == "EXECUTE"
                else None
            ),
        )
        validate_execution_recipe(recipe_v4_validation_view(value), request=request)
        return value
    if type(value) is dict and value.get("schema_version") == SCHEMA_V4:
        from .dashboard_serving_contract import POLICY

        if set(value) != FIELDS_V4 or value["dashboard_publication_policy"] != POLICY:
            raise ContractError("EXECUTE_RECIPE_V4_FIELDS_INVALID")
        prior = {k: v for k, v in value.items() if k != "dashboard_publication_policy"}
        prior["schema_version"] = SCHEMA_V3
        validate_execution_recipe(prior, request=request)
        return value
    if type(value) is dict and value.get("schema_version") == SCHEMA_V3:
        if set(value) != FIELDS_V3:
            raise ContractError("EXECUTE_RECIPE_V3_FIELDS_INVALID")
        validate_ref(value["corporate_action_context_ref"])
        legacy = {k: v for k, v in value.items() if k != "corporate_action_context_ref"}
        legacy["schema_version"] = SCHEMA_V2
        validate_execution_recipe_v2(legacy, request=request)
        return value
    if type(value) is dict and value.get("schema_version") == SCHEMA_V2:
        return validate_execution_recipe_v2(value, request=request)
    if type(value) is not dict or set(value) != FIELDS or value["schema_version"] != SCHEMA:
        raise ContractError("EXECUTE_RECIPE_FIELDS_INVALID")
    return _validate_recipe_body(value, request=request)


def validate_execution_recipe_v2(value: dict, *, request: dict) -> dict:
    """Validate exact source-mode selection; acquisition policy is checked at preflight."""
    if type(value) is not dict or set(value) != FIELDS_V2 or value["schema_version"] != SCHEMA_V2:
        raise ContractError("EXECUTE_RECIPE_V2_FIELDS_INVALID")
    _validate_recipe_body(value, request=request)
    acquisition = value["theme_acquisition_ref"]
    pinned = value["research_sources"]["theme_source_ref"]
    if (acquisition is None) == (pinned is None):
        raise ContractError("EXECUTE_RECIPE_THEME_MODE_INVALID")
    if acquisition is not None:
        validate_ref(acquisition)
    return value


def validate_catchup_template(value: dict, *, request: dict) -> dict:
    """Validate a declared recipe before its actual predecessor exists."""
    if type(value) is dict and value.get("schema_version") == SCHEMA_V6:
        validate_catchup_template(registered_validation_view(value), request=request)
        return value
    if type(value) is dict and value.get("schema_version") == SCHEMA_V5:
        from .research_timing import recipe_v4_validation_view

        if set(value) != FIELDS_V5:
            raise ContractError("CATCHUP_TEMPLATE_V5_FIELDS_INVALID")
        validate_catchup_template(recipe_v4_validation_view(value), request=request)
        return value
    versions = {SCHEMA: FIELDS, SCHEMA_V2: FIELDS_V2, SCHEMA_V3: FIELDS_V3, SCHEMA_V4: FIELDS_V4}
    if (
        type(value) is not dict
        or type(value.get("schema_version")) is not str
        or value["schema_version"] not in versions
        or set(value) != versions[value["schema_version"]]
    ):
        raise ContractError("CATCHUP_TEMPLATE_FIELDS_INVALID")
    if (
        value["previous_completion_ref"] is not None
        or value["bootstrap_ref"] is not None
        or type(value["store_preimages"]) is not dict
        or value["store_preimages"].get("store_pointer_ref") is not None
    ):
        raise ContractError("CATCHUP_TEMPLATE_DERIVED_INPUT_FORBIDDEN")
    _validate_recipe_body(value, request=request, template=True)
    if value["schema_version"] in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4}:
        acquisition = value["theme_acquisition_ref"]
        if (acquisition is None) == (value["research_sources"]["theme_source_ref"] is None):
            raise ContractError("EXECUTE_RECIPE_THEME_MODE_INVALID")
        if acquisition is not None:
            validate_ref(acquisition)
    if value["schema_version"] in {SCHEMA_V3, SCHEMA_V4}:
        validate_ref(value["corporate_action_context_ref"])
    if value["schema_version"] == SCHEMA_V4:
        from .dashboard_serving_contract import POLICY

        if value["dashboard_publication_policy"] != POLICY:
            raise ContractError("EXECUTE_RECIPE_V4_POLICY_INVALID")
    return value


def _validate_recipe_body(value: dict, *, request: dict, template: bool = False) -> dict:
    for name in (
        "market",
        "strategy_id",
        "target_trade_date",
        "graph_sha256",
        "release_install_ref",
    ):
        if value[name] != request[name]:
            raise ContractError("EXECUTE_RECIPE_REQUEST_BINDING_INVALID")
    if (
        value["graph_sha256"] != GRAPH_SHA256
        or type(value["publish_current_dashboard"]) is not bool
    ):
        raise ContractError("EXECUTE_RECIPE_POLICY_INVALID")
    day = value["target_trade_date"]
    _validate_day(day)
    validate_ref(value["release_ref"])
    if not template and (value["previous_completion_ref"] is None) == (
        value["bootstrap_ref"] is None
    ):
        raise ContractError("EXECUTE_RECIPE_ANCHOR_INVALID")
    for key in (
        "previous_completion_ref",
        "bootstrap_ref",
        "factor_loop_context_ref",
        "retrospective_ref",
    ):
        if value[key] is not None:
            validate_ref(value[key])
    if request["action"] == "EXECUTE" and value["factor_loop_context_ref"] is None:
        raise ContractError("PRODUCTION_REQUEST_EXECUTE_CONTEXT_REQUIRED")
    if value["previous_completion_ref"] is not None:
        path = PurePosixPath(value["previous_completion_ref"]["path"])
        previous = path.parent.name
        _validate_day(previous)
        if (
            str(path) != f"results/operations/daily_production/CN/{previous}/completion.v1.json"
            or previous >= day
        ):
            raise ContractError("EXECUTE_RECIPE_ANCHOR_INVALID")
    _refs(value["policy_refs"], {"research", "store", "prospective"}, {"prospective"})
    _refs(
        value["store_preimages"],
        {"store_pointer_ref", "event_pointer_ref", "benchmark_pointer_ref"},
        {"store_pointer_ref"} if template else (),
    )
    _refs(value["dashboard_sources"], {"benchmark_ref", "risk_free_ref"})
    sources = value["research_sources"]
    if type(sources) is not dict or set(sources) != {
        "as_of",
        "industry_source_ref",
        "theme_source_ref",
        "exposure_rows_ref",
        "fundamental",
        "macro",
    }:
        raise ContractError("EXECUTE_RECIPE_SOURCE_FIELDS_INVALID")
    if utc_stamp(sources["as_of"]).strftime("%Y%m%d") != day:
        raise ContractError("EXECUTE_RECIPE_CUTOFF_DATE_INVALID")
    for key in ("industry_source_ref", "theme_source_ref", "exposure_rows_ref"):
        if sources[key] is not None:
            validate_ref(sources[key])
    for name in ("fundamental", "macro"):
        source = sources[name]
        if type(source) is not dict or set(source) != {"mode", "source_ref"}:
            raise ContractError("EXECUTE_RECIPE_SOURCE_MODE_INVALID")
        if source["mode"] == "PINNED":
            validate_ref(source["source_ref"])
        elif source["mode"] != "MAINTENANCE_STAGE" or source["source_ref"] is not None:
            raise ContractError("EXECUTE_RECIPE_SOURCE_MODE_INVALID")
    return value
