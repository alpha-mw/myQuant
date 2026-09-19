"""Handoff v4 integration; native origin chain has separate financial fixture coverage."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from _public_catchup_fixture import put
from _daily_preparation_fixture import snapshot
from test_daily_evidence_maintenance_handoff import context
from quant_investor.operations import maintenance_handoff as handoff
from quant_investor.operations import automatic_origin as origin
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.maintenance_handoff_contract import validate_handoff_shape
from quant_investor.operations.research_timing import CURRENT, acquisition_deadline
from quant_investor.operations.dashboard_serving_contract import POLICY


def fixture(root, monkeypatch):
    args, _ = context(root, monkeypatch)
    request = json.loads((root / args["request_ref"]["path"]).read_bytes())
    recipe = json.loads((root / request["recipe_ref"]["path"]).read_bytes())
    reference = request["release_install_ref"]
    recipe.update(
        schema_version="cn-daily-execute-recipe.v6",
        registered_event_declaration_ref=reference,
        corporate_action_template_ref=reference,
        theme_acquisition_ref=None,
        dashboard_publication_policy=POLICY,
        research_timing={
            "mode": CURRENT,
            "policy_ref": reference,
            "acquisition_deadline": acquisition_deadline("20260904"),
        },
    )
    recipe["research_sources"]["as_of"] = None
    recipe["research_sources"]["theme_source_ref"] = reference
    request["recipe_ref"] = put(root, request["recipe_ref"]["path"], recipe)
    args["request_ref"] = put(root, args["request_ref"]["path"], request)
    value = origin.origin_document(
        current={
            "auto_request_ref": reference,
            "resolution_ref": reference,
            "derived_request_ref": reference,
        },
        binding_ref=reference,
        execution_request_ref=args["request_ref"],
        day="20260904",
    )
    ref = put(root, origin.origin_path("20260904", args["request_ref"]), value)
    source = SimpleNamespace(recheck=lambda: None)
    verified = {
        "origin": value,
        "synthetic": True,
        "bound": {"request": request, "recipe": recipe, "sources": source},
        "resolution": {"resolution": {"resolved_at": "2026-09-04T06:59:00Z"}, "sources": source},
        "sources": source,
    }
    active = [True]

    def read(**kwargs):
        assert kwargs["reference"] == ref
        return verified

    def live(**kwargs):
        return (
            {k: value[k] for k in ("auto_request_ref", "resolution_ref", "derived_request_ref")}
            if active[0]
            else None
        )

    # This test exercises handoff/core retention, not automatic resolution or
    # native registered accounting, which test_registered_automatic_origin covers.
    monkeypatch.setattr(origin, "read_automatic_origin", read)
    monkeypatch.setattr(origin, "current_automatic_origin", live)
    return args, ref, verified, active


@pytest.mark.parametrize("automatic", [False, True])
def test_registered_direct_v2_and_automatic_v4_replay_original_bytes(
    tmp_path, monkeypatch, automatic
):
    args, ref, _, active = fixture(tmp_path, monkeypatch)
    if automatic:
        args["_automatic_origin_ref"] = ref
    result = handoff.publish_maintenance_handoff(**args)
    value = json.loads((tmp_path / result["path"]).read_bytes())
    assert value["schema_version"] == f"cn-daily-maintenance-handoff.v{4 if automatic else 2}"
    assert value.get("automatic_origin_ref") == (ref if automatic else None)
    before = snapshot(tmp_path)
    assert handoff.publish_maintenance_handoff(**args) == result
    active[0] = False
    assert (
        handoff.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=result)["handoff"]
        == value
    )
    assert snapshot(tmp_path) == before
    if automatic:
        with pytest.raises(ContractError, match="CAPABILITY_REQUIRED"):
            handoff.publish_maintenance_handoff(**args)
        assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "fault", ["execution_request", "recipe", "expired", "seal_before_resolution"]
)
def test_handoff_origin_mismatch_rejects_without_writing(tmp_path, monkeypatch, fault):
    args, ref, verified, active = fixture(tmp_path, monkeypatch)
    args["_automatic_origin_ref"] = ref
    if fault == "execution_request":
        verified["origin"]["execution_request_ref"] = {"path": "other.json", "sha256": "a" * 64}
    elif fault == "recipe":
        verified["bound"]["recipe"] = {"synthetic": "wrong recipe"}
    elif fault == "expired":
        active[0] = False
    else:
        result = handoff.publish_maintenance_handoff(**args)
        value = json.loads((tmp_path / result["path"]).read_bytes())
        value["sealed_at"] = "2026-09-04T06:58:00Z"
        result = put(tmp_path, result["path"], value)
    before = snapshot(tmp_path)
    with pytest.raises(ContractError):
        if fault == "seal_before_resolution":
            handoff.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=result)
        else:
            handoff.publish_maintenance_handoff(**args)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_only_handoff_v4_has_a_required_nonnull_origin(tmp_path, monkeypatch, version):
    args, ref, _, _ = fixture(tmp_path, monkeypatch)
    saved = handoff.publish_maintenance_handoff(**args)
    value = json.loads((tmp_path / saved["path"]).read_bytes())
    value["schema_version"] = f"cn-daily-maintenance-handoff.v{version}"
    if version == 1:
        del value["prospective_policy_ref"]
    elif version == 3:
        value.update(historical_session_ref=ref, catchup_binding_ref=ref)
    if version == 4:
        with pytest.raises(ContractError):
            validate_handoff_shape(value)
    value["automatic_origin_ref"] = ref
    if version == 4:
        assert validate_handoff_shape(value) == value
        value["automatic_origin_ref"] = None
    with pytest.raises(ContractError):
        validate_handoff_shape(value)


def test_completed_snapshot_retains_exact_origin_and_rechecks_it(tmp_path, monkeypatch):
    from test_daily_evidence_archived_context import fixture as archived_fixture
    from quant_investor.operations.completed_handoff_snapshot import _mint_snapshot

    old = archived_fixture(tmp_path, monkeypatch)[0]
    docs = {row[0]: row for row in old.documents}
    value = deepcopy(old.document("handoff"))
    ref = put(tmp_path, "automatic-origin.json", {"authority": FALSE_AUTHORITY})
    value.update(schema_version="cn-daily-maintenance-handoff.v4", automatic_origin_ref=ref)
    handoff_ref = put(tmp_path, docs["handoff"][1], value)
    docs["handoff"] = (
        "handoff",
        handoff_ref["path"],
        handoff_ref["sha256"],
        (tmp_path / handoff_ref["path"]).read_bytes(),
    )
    with pytest.raises(ContractError, match="ORIGIN_ROLE_INVALID"):
        _mint_snapshot(
            workspace=str(tmp_path), trade_date=old.trade_date, documents=tuple(docs.values())
        )
    docs["automatic_origin"] = (
        "automatic_origin",
        ref["path"],
        ref["sha256"],
        (tmp_path / ref["path"]).read_bytes(),
    )
    completion_ref = put(tmp_path, docs["completion"][1], {"synthetic": True})
    docs["completion"] = (
        "completion",
        completion_ref["path"],
        completion_ref["sha256"],
        (tmp_path / completion_ref["path"]).read_bytes(),
    )
    current = _mint_snapshot(
        workspace=str(tmp_path), trade_date=old.trade_date, documents=tuple(docs.values())
    )
    assert current.reference("automatic_origin") == ref
    current.recheck()
    from quant_investor.operations.archived_handoff_context import verify_archived_handoff_context

    source = SimpleNamespace(recheck=lambda: None)
    verified = {
        "origin": {"execution_request_ref": value["request_ref"]},
        "sources": source,
        "bound": {
            "request": {"synthetic": "request seam"},
            "recipe": current.document("recipe"),
            "sources": source,
        },
        "resolution": {"sources": source, "resolution": {"resolved_at": "2026-09-08T12:00:00Z"}},
    }
    monkeypatch.setattr(origin, "read_automatic_origin", lambda **kw: verified)
    assert verify_archived_handoff_context(current)["claim"]["run_date"] == current.trade_date
    verified["origin"]["execution_request_ref"] = {"path": "wrong.json", "sha256": "0" * 64}
    with pytest.raises(ContractError, match="ARCHIVED_AUTOMATIC_ORIGIN_MISMATCH"):
        verify_archived_handoff_context(current)
    put(tmp_path, ref["path"], {"tampered": True})
    with pytest.raises(ContractError, match="SNAPSHOT_CHANGED"):
        current.recheck()
