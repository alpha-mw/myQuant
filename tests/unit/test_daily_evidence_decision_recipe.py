"""Exact new/legacy profile dispatch over a real native Store plan and preimage."""

import hashlib
import json

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.native_input_contract import (
    FIELDS,
    EXTRAS,
    validate_native_input_shape,
)
from quant_investor.operations.decision_recipe import (
    publish_decision_recipe,
    read_decision_recipe,
    native_portfolio_is_late,
)
from test_daily_evidence_portfolio_binding import setup
from test_daily_evidence_research_sources import put


@pytest.mark.parametrize("version", [1, 2, 3])
def test_exact_native_input_profiles(version):
    # Shape only: business inputs must separately pass their owning decoders.
    fields = FIELDS | (EXTRAS if version >= 2 else set())
    value = dict.fromkeys(fields)
    value["schema_version"] = f"cn-daily-native-inputs.v{version}"
    if version == 3:
        value["decision_recipe_ref"] = {"path": "recipe.json", "sha256": "a" * 64}
    validate_native_input_shape(value)
    changed = dict(value)
    changed["decision_recipe_ref"] = None
    with pytest.raises(ValueError):
        validate_native_input_shape(changed)
    if version == 3:
        value.pop("decision_recipe_ref")
        with pytest.raises(ValueError):
            validate_native_input_shape(value)


@pytest.mark.parametrize("schema", ["cn-daily-native-inputs.v4", None, [], True])
def test_unknown_profile_is_not_legacy(schema):
    with pytest.raises(ValueError):
        validate_native_input_shape({"schema_version": schema})


def test_recipe_publication_replays_exact_book_and_policy(tmp_path):
    _, _, _, plan_ref, journal = setup(tmp_path)
    request_ref = put(
        tmp_path,
        "research.json",
        {
            "as_of": "2026-08-24T13:30:00Z",
            "strategy_id": "aggressive_tech_manufacturing",
        },
    )
    with journal.locked():
        ref = publish_decision_recipe(
            journal=journal, research_request_ref=request_ref, store_plan_ref=plan_ref
        )
        assert (
            publish_decision_recipe(
                journal=journal, research_request_ref=request_ref, store_plan_ref=plan_ref
            )
            == ref
        )
    kwargs = dict(
        workspace=tmp_path,
        trade_date=journal.trade_date,
        recipe_ref=ref,
        research_request_ref=request_ref,
        store_plan_ref=plan_ref,
    )
    bound = read_decision_recipe(**kwargs)
    assert bound["portfolio"]["payload"]["timing_status"] == "LATE_RECORDED"
    native = dict.fromkeys(FIELDS | EXTRAS | {"decision_recipe_ref"})
    native.update(
        schema_version="cn-daily-native-inputs.v3",
        trade_date=journal.trade_date,
        research_request_ref=request_ref,
        store_plan_ref=plan_ref,
        decision_recipe_ref=ref,
    )
    assert native_portfolio_is_late(workspace=tmp_path, inputs=native)
    recipe = json.loads((tmp_path / ref["path"]).read_bytes())
    recipe["report_policy"] = {"new_weight": 0.5}
    raw = canonical_json_bytes(recipe)
    digest = hashlib.sha256(raw).hexdigest()
    bad = put(tmp_path, str(journal.root / "inputs" / f"decision-recipe-{digest}.json"), recipe)
    with pytest.raises(ValueError, match="RECIPE_BINDING_INVALID"):
        read_decision_recipe(**{**kwargs, "recipe_ref": bad})
