"""New timing-profile derivation over real synthetic Calendar and controlled EOD admission."""

import json

import pytest

from _public_catchup_fixture import routed_collection, completion, put
from quant_investor.operations import catchup_binding as binding
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.research_timing import (
    CURRENT,
    HISTORICAL,
    acquisition_deadline,
    approved_research_timing_policy,
)
from quant_investor.operations.automatic_catchup_contract import (
    REQUEST_SCHEMA_V2,
    validate_automatic_request,
)
from test_automatic_catchup_resolution import fixture as auto_fixture, resolve


def upgrade(root, request):
    value = json.loads((root / request["recipe_ref"]["path"]).read_bytes())
    value["schema_version"] = binding.COLLECTION_SCHEMA_V3
    for day, recipe in value["recipes"].items():
        mode = CURRENT if day == "20260828" else HISTORICAL
        recipe["schema_version"] = "cn-daily-execute-recipe.v5"
        recipe["corporate_action_template_ref"] = recipe.pop("corporate_action_context_ref")
        recipe["research_timing"] = {
            "mode": mode,
            "policy_ref": put(
                root, "timing-" + day + ".json", approved_research_timing_policy(mode)
            ),
            "acquisition_deadline": acquisition_deadline(day) if mode == CURRENT else None,
        }
        if mode == CURRENT:
            recipe["research_sources"]["as_of"] = None
    request["recipe_ref"] = put(root, "collection-v3.json", value)
    return value


def test_v3_binding_persists_exact_current_and_historical_modes(tmp_path):
    _, request = routed_collection(tmp_path)
    upgrade(tmp_path, request)
    root_ref = put(tmp_path, "request-v3.json", request)
    previous = request["previous_completion_ref"]
    for day in ("20260827", "20260828"):
        derived = binding.derive_catchup_binding(
            workspace=str(tmp_path), request_ref=root_ref, day=day, previous_completion_ref=previous
        )
        assert derived["binding"]["schema_version"] == binding.SCHEMA_V3
        assert derived["recipe"]["research_timing"]["mode"] == (
            CURRENT if day == "20260828" else HISTORICAL
        )
        journal = DailyJournal(str(tmp_path), day)
        with journal.locked():
            ref = binding.persist_catchup_binding(journal=journal, derived=derived)
        assert ref["path"].endswith("binding.v3.json")
        assert (
            binding.read_catchup_binding(workspace=str(tmp_path), binding_ref=ref)["binding"]
            == derived["binding"]
        )
        previous = completion(tmp_path, day, request)


def test_automatic_v2_requires_v3_collection_and_derives_new_profile(tmp_path, monkeypatch):
    request, _, _, _, _ = auto_fixture(tmp_path, monkeypatch)
    request["schema_version"] = REQUEST_SCHEMA_V2
    ref = put(tmp_path, "auto-v2.json", request)
    validate_automatic_request(request, release_install_ref=request["release_install_ref"])
    with pytest.raises(ContractError, match="COLLECTION_INVALID"):
        resolve(tmp_path, ref, request)
    upgrade(tmp_path, request)
    ref = put(tmp_path, "auto-v2-final.json", request)
    result = resolve(tmp_path, ref, request)
    assert (
        result["resolution"]["derived_collection"]["schema_version"] == binding.COLLECTION_SCHEMA_V3
    )
    assert all(
        recipe["schema_version"] == "cn-daily-execute-recipe.v5"
        for recipe in result["resolution"]["derived_collection"]["recipes"].values()
    )
