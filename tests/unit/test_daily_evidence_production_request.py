"""The initial EXECUTE request cannot require outputs that do not exist yet."""

from copy import deepcopy
import pytest
from quant_investor.operations.daily_contract import ContractError, GRAPH_SHA256
from quant_investor.operations.production_request import SCHEMA, validate_production_request
from quant_investor.operations.execution_recipe import validate_execution_recipe, SCHEMA as RECIPE


def request(action="EXECUTE"):
    ref = {"path": "input.json", "sha256": "a" * 64}
    value = {
        "schema_version": SCHEMA,
        "market": "CN",
        "strategy_id": "aggressive_tech_manufacturing",
        "action": action,
        "target_trade_date": "20260904",
        "graph_sha256": GRAPH_SHA256,
        "release_install_ref": ref,
        "recipe_ref": None,
        "maintenance_handoff_ref": None,
        "calendar_ref": None,
        "raw_calendar_ref": None,
        "previous_completion_ref": None,
        "day_input_refs": {},
    }
    if action in {"PLAN", "EXECUTE"}:
        value["recipe_ref"] = ref
    elif action == "RESUME":
        value["maintenance_handoff_ref"] = ref
    else:
        for key in ["calendar_ref", "raw_calendar_ref", "previous_completion_ref"]:
            value[key] = ref
        value["day_input_refs"] = {"20260904": ref}
    return value


def recipe():
    req = request()
    ref = deepcopy(req["release_install_ref"])
    return {
        "schema_version": RECIPE,
        **{
            k: req[k]
            for k in [
                "market",
                "strategy_id",
                "target_trade_date",
                "graph_sha256",
                "release_install_ref",
            ]
        },
        "release_ref": ref,
        "previous_completion_ref": {
            "path": "results/operations/daily_production/CN/20260903/completion.v1.json",
            "sha256": "b" * 64,
        },
        "bootstrap_ref": None,
        "factor_loop_context_ref": ref,
        "policy_refs": {"research": ref, "store": ref, "prospective": None},
        "store_preimages": {
            k: ref for k in ["store_pointer_ref", "event_pointer_ref", "benchmark_pointer_ref"]
        },
        "retrospective_ref": None,
        "dashboard_sources": {"benchmark_ref": ref, "risk_free_ref": ref},
        "research_sources": {
            "as_of": "2026-09-04T13:30:00Z",
            "industry_source_ref": None,
            "theme_source_ref": None,
            "exposure_rows_ref": None,
            "fundamental": {"mode": "MAINTENANCE_STAGE", "source_ref": None},
            "macro": {"mode": "MAINTENANCE_STAGE", "source_ref": None},
        },
        "publish_current_dashboard": True,
    }


def check(value):
    return validate_production_request(
        value, release_install_ref={"path": "input.json", "sha256": "a" * 64}
    )


@pytest.mark.parametrize("action", ["PLAN", "EXECUTE", "RESUME", "CATCH_UP"])
def test_action_specific_request(action):
    value = request(action)
    assert check(value) == value


@pytest.mark.parametrize(
    "field", ["command", "synthetic", "allow_live", "authority", "status", "slot_claim_ref"]
)
def test_injected_fields_reject(field):
    value = request()
    value[field] = True
    with pytest.raises(ContractError):
        check(value)


def test_execute_without_produced_refs_and_resume_without_recipe():
    assert check(request())["day_input_refs"] == {}
    value = request()
    value["calendar_ref"] = value["release_install_ref"]
    with pytest.raises(ContractError):
        check(value)
    value = request("RESUME")
    value["recipe_ref"] = value["release_install_ref"]
    with pytest.raises(ContractError):
        check(value)


def test_recipe_context_and_pinned_source_rules():
    value = recipe()
    assert validate_execution_recipe(value, request=request()) == value
    value["factor_loop_context_ref"] = None
    with pytest.raises(ContractError, match="EXECUTE_CONTEXT_REQUIRED"):
        validate_execution_recipe(value, request=request())
    assert validate_execution_recipe(value, request=request("PLAN")) == value
    value["research_sources"]["macro"]["mode"] = "PINNED"
    with pytest.raises(ContractError):
        validate_execution_recipe(value, request=request("PLAN"))
