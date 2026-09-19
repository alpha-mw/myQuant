"""Exact version dispatch shared by native input loading and historical replay."""

from .daily_contract import ContractError, validate_ref

FIELDS = frozenset(
    {
        "schema_version",
        "trade_date",
        "factor_pointer_sha256",
        "release_ref",
        "research_request_ref",
        "store_plan_ref",
        "store_policy_ref",
        "retrospective_ref",
        "calendar_ref",
        "market_snapshot_ref",
        "benchmark_ref",
        "risk_free_ref",
        "previous_trade_date",
        "adjustment_market_refs",
        "publish_current_dashboard",
    }
)
EXTRAS = frozenset({"next_session_calendar_proof_ref", "next_session_calendar_failure_ref"})


def validate_native_input_shape(value) -> None:
    if type(value) is not dict:
        raise ContractError("NATIVE_INPUT_DOCUMENT_SCHEMA_INVALID")
    schema = value.get("schema_version")
    if type(schema) is not str:
        raise ContractError("NATIVE_INPUT_DOCUMENT_SCHEMA_INVALID")
    shapes = {
        "cn-daily-native-inputs.v1": FIELDS,
        "cn-daily-native-inputs.v2": FIELDS | EXTRAS,
        "cn-daily-native-inputs.v3": FIELDS | EXTRAS | {"decision_recipe_ref"},
        "cn-daily-native-inputs.v4": FIELDS
        | EXTRAS
        | {"decision_recipe_ref", "corporate_action_context_ref"},
    }
    shapes["cn-daily-native-inputs.v5"] = shapes["cn-daily-native-inputs.v4"] | {
        "dashboard_publication_policy"
    }
    shapes["cn-daily-native-inputs.v6"] = shapes["cn-daily-native-inputs.v5"] | {"cutoff_ref"}
    shapes["cn-daily-native-inputs.v7"] = shapes["cn-daily-native-inputs.v6"] | {
        "registered_event_declaration_ref"
    }
    expected = shapes.get(schema)
    if expected is None or set(value) != expected:
        raise ContractError("NATIVE_INPUT_DOCUMENT_SCHEMA_INVALID")
    if schema in {
        "cn-daily-native-inputs.v3",
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        validate_ref(value["decision_recipe_ref"])
    if schema in {
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        validate_ref(value["corporate_action_context_ref"])

    if schema in {
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        from .dashboard_serving_contract import POLICY

        if value["dashboard_publication_policy"] != POLICY:
            raise ContractError("NATIVE_INPUT_DASHBOARD_POLICY_INVALID")
    if schema in {"cn-daily-native-inputs.v6", "cn-daily-native-inputs.v7"}:
        validate_ref(value["cutoff_ref"])
    if schema == "cn-daily-native-inputs.v7":
        validate_ref(value["registered_event_declaration_ref"])
        if value["retrospective_ref"] is not None:
            raise ContractError("NATIVE_INPUT_REGISTERED_RETROSPECTIVE_UNSUPPORTED")
