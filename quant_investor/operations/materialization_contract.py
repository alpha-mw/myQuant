"""Exact recipe-selected materialization versions; no latest-version fallback."""

from .daily_contract import ContractError, validate_ref
from .execution_recipe import (
    SCHEMA as RECIPE_V1,
    SCHEMA_V2 as RECIPE_V2,
    SCHEMA_V3 as RECIPE_V3,
    SCHEMA_V4 as RECIPE_V4,
    SCHEMA_V5 as RECIPE_V5,
    SCHEMA_V6 as RECIPE_V6,
)

SCHEMA_V1 = "cn-daily-materialization.v1"
SCHEMA_V2 = "cn-daily-materialization.v2"
SCHEMA_V3 = "cn-daily-materialization.v3"
SCHEMA_V4 = "cn-daily-materialization.v4"
SCHEMA_V5 = "cn-daily-materialization.v5"
SCHEMA_V6 = "cn-daily-materialization.v6"
FIELDS_V1 = frozenset(
    {
        "schema_version",
        "trade_date",
        "graph_sha256",
        "maintenance_handoff_ref",
        "research_request_ref",
        "store_plan_ref",
        "native_inputs_ref",
        "auxiliary_stage_refs",
        "sealed_at",
        "authority",
    }
)
FIELDS_V2 = FIELDS_V1 | {"theme_source_handoff_ref"}
FIELDS_V3 = FIELDS_V2 | {"corporate_action_context_ref"}
FIELDS_V4 = FIELDS_V3 | {"dashboard_publication_policy"}
FIELDS_V5 = FIELDS_V4 | {"cutoff_ref", "corporate_action_template_ref"}
FIELDS_V6 = FIELDS_V5 | {"registered_event_declaration_ref"}


def layout_for_recipe(recipe: dict) -> tuple[str, str, frozenset[str]]:
    version = recipe.get("schema_version")
    if version == RECIPE_V1:
        return SCHEMA_V1, "materialization.v1.json", FIELDS_V1
    if version == RECIPE_V2:
        return SCHEMA_V2, "materialization.v2.json", FIELDS_V2
    if version == RECIPE_V3:
        return SCHEMA_V3, "materialization.v3.json", FIELDS_V3
    if version == RECIPE_V4:
        return SCHEMA_V4, "materialization.v4.json", FIELDS_V4
    if version == RECIPE_V5:
        return SCHEMA_V5, "materialization.v5.json", FIELDS_V5
    if version == RECIPE_V6:
        return SCHEMA_V6, "materialization.v6.json", FIELDS_V6
    raise ContractError("MATERIALIZATION_RECIPE_VERSION_INVALID")


def validate_materialization_shape(value: dict) -> None:
    if type(value) is not dict:
        raise ContractError("MATERIALIZATION_CONTRACT_INVALID")
    fields = {
        SCHEMA_V1: FIELDS_V1,
        SCHEMA_V2: FIELDS_V2,
        SCHEMA_V3: FIELDS_V3,
        SCHEMA_V4: FIELDS_V4,
        SCHEMA_V5: FIELDS_V5,
        SCHEMA_V6: FIELDS_V6,
    }.get(value.get("schema_version"))
    if fields is None or set(value) != fields:
        raise ContractError("MATERIALIZATION_CONTRACT_INVALID")


def validate_materialization_version(
    value: dict,
    *,
    recipe: dict,
    execution,
    path: str,
    theme_policy: dict | None = None,
) -> None:
    validate_materialization_shape(value)
    schema, filename, _ = layout_for_recipe(recipe)
    if value["schema_version"] != schema or path != str(execution / filename):
        raise ContractError("MATERIALIZATION_VERSION_BINDING_INVALID")
    if schema in {SCHEMA_V3, SCHEMA_V4}:
        if (
            validate_ref(value["corporate_action_context_ref"])
            != recipe["corporate_action_context_ref"]
        ):
            raise ContractError("MATERIALIZATION_CORPORATE_CONTEXT_MISMATCH")
    if (
        schema == SCHEMA_V6
        and validate_ref(value["registered_event_declaration_ref"])
        != recipe["registered_event_declaration_ref"]
    ):
        raise ContractError("MATERIALIZATION_REGISTERED_DECLARATION_MISMATCH")
    if schema in {SCHEMA_V5, SCHEMA_V6}:
        if validate_ref(value["corporate_action_template_ref"]) != recipe[
            "corporate_action_template_ref"
        ] or validate_ref(value["cutoff_ref"])["path"] != str(
            execution
            / ("research-cutoff.v2.json" if schema == SCHEMA_V6 else "research-cutoff.v1.json")
        ):
            raise ContractError("MATERIALIZATION_CUTOFF_BINDING_INVALID")
        context_ref = validate_ref(value["corporate_action_context_ref"])
        if context_ref["path"] != str(
            execution / "inputs" / f"corporate-context-{context_ref['sha256']}.json"
        ):
            raise ContractError("MATERIALIZATION_CORPORATE_CONTEXT_PATH_INVALID")
    if schema in {SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}:
        from .dashboard_serving_contract import POLICY

        if (
            value["dashboard_publication_policy"] != recipe["dashboard_publication_policy"]
            or value["dashboard_publication_policy"] != POLICY
        ):
            raise ContractError("MATERIALIZATION_DASHBOARD_POLICY_INVALID")
    if schema in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}:
        if "theme_acquisition_ref" not in recipe:
            raise ContractError("MATERIALIZATION_THEME_MODE_MISSING")
        handoff = value["theme_source_handoff_ref"]
        if recipe["theme_acquisition_ref"] is None:
            if handoff is not None:
                raise ContractError("MATERIALIZATION_UNEXPECTED_THEME_HANDOFF")
        else:
            filename = "handoff.v1.json"
            if theme_policy is not None:
                from .theme_acquisition import (
                    validate_theme_acquisition_policy,
                    POLICY_SCHEMA_V2,
                    POLICY_SCHEMA_V3,
                    validate_theme_policy_profile,
                )

                policy = validate_theme_acquisition_policy(theme_policy)
                validate_theme_policy_profile(recipe, policy)
                if policy["schema_version"] == POLICY_SCHEMA_V3:
                    filename = "handoff.v3.json"
                elif policy["schema_version"] == POLICY_SCHEMA_V2:
                    filename = "handoff.v2.json"
            if validate_ref(handoff)["path"] != str(execution / "theme-source" / filename):
                raise ContractError("MATERIALIZATION_THEME_HANDOFF_PATH_INVALID")


def validate_bound_materialization(*, workspace, value, recipe, execution, path):
    """Native callers select the Theme version from exact policy bytes, never paths."""
    from quant_investor.contracts import parse_canonical_json_bytes
    from quant_investor.system.storage import SecureSystemStorage

    schema, _, _ = layout_for_recipe(recipe)
    validate_materialization_shape(value)
    policy = None
    source = None
    if (
        schema in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}
        and recipe.get("theme_acquisition_ref") is not None
    ):
        ref = validate_ref(recipe["theme_acquisition_ref"])
        reader = SecureSystemStorage(workspace)
        source = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024)
        if source.byte_sha256 != ref["sha256"]:
            raise ContractError("MATERIALIZATION_THEME_POLICY_SHA_MISMATCH")
        policy = parse_canonical_json_bytes(source.data)
    validate_materialization_version(
        value, recipe=recipe, execution=execution, path=path, theme_policy=policy
    )
    if (
        source is not None
        and reader.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024).data
        != source.data
    ):
        raise ContractError("MATERIALIZATION_THEME_POLICY_CHANGED")


def read_selected_materialization(*, journal, execution, recipe: dict):
    """One known path and one conflict check; never select by filesystem recency."""
    _, filename, _ = layout_for_recipe(recipe)
    for other in (
        "materialization.v1.json",
        "materialization.v2.json",
        "materialization.v3.json",
        "materialization.v4.json",
        "materialization.v5.json",
        "materialization.v6.json",
    ):
        if other != filename and journal.storage.read(str(execution / other)) is not None:
            raise ContractError("MATERIALIZATION_VERSION_CONFLICT")
    return journal.storage.read(str(execution / filename))
