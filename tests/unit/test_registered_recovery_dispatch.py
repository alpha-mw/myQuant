"""Public EXECUTE/RESUME dispatch into the independently tested recovery owner."""

import json

import pytest

from _public_catchup_fixture import put
from test_registered_origin_handoff import fixture
from test_daily_evidence_store_materialization import context as _scripts_context  # noqa: F401
from scripts import daily_production, daily_materialization, daily_registered_recovery as recovery


def test_registered_committed_execute_does_not_use_bootstrap_profile(tmp_path, monkeypatch):
    args, _, _, _ = fixture(tmp_path, monkeypatch)
    request = json.loads((tmp_path / args["request_ref"]["path"]).read_bytes())
    calls = []
    monkeypatch.setattr(
        recovery,
        "recover_registered_committed",
        lambda **kw: calls.append(kw) or {"status": "PARTIAL", "completion_ref": None},
    )
    result = daily_production.dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=args["request_ref"],
        release_install_ref=request["release_install_ref"],
        synthetic=True,
        committed_recovery_only=True,
    )
    assert result["execution_state"] == "PARTIAL"
    assert calls == [
        {
            "workspace": str(tmp_path.resolve()),
            "request_ref": args["request_ref"],
            "release_install_ref": request["release_install_ref"],
            "synthetic": True,
            "automatic_origin_ref": None,
        }
    ]


@pytest.mark.parametrize("action", ["EXECUTE", "RESUME"])
def test_existing_registered_close_routes_to_guard_before_materialization(
    tmp_path, monkeypatch, action
):
    from quant_investor.operations.maintenance_handoff import (
        publish_maintenance_handoff,
        read_maintenance_handoff,
    )

    args, _, _, _ = fixture(tmp_path, monkeypatch)
    handoff_ref = publish_maintenance_handoff(**args)
    recovered = read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=handoff_ref)
    request = json.loads((tmp_path / args["request_ref"]["path"]).read_bytes())
    request_ref = args["request_ref"]
    if action == "RESUME":
        request.update(action="RESUME", recipe_ref=None, maintenance_handoff_ref=handoff_ref)
        request_ref = put(tmp_path, "resume.json", request)
    calls = []
    monkeypatch.setattr(recovery, "registered_commit_required", lambda **kw: True)
    monkeypatch.setattr(
        recovery,
        "recover_registered_committed",
        lambda **kw: calls.append(kw) or {"status": "PARTIAL", "completion_ref": None},
    )
    for module in (daily_production, daily_materialization):
        monkeypatch.setattr(module, "read_maintenance_handoff", lambda **kw: recovered)
        monkeypatch.setattr(
            module,
            "materialize_daily_inputs",
            lambda **kw: pytest.fail("fresh materialization before committed guard"),
        )
    result = daily_production.dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=request_ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
    )
    assert result["execution_state"] == "PARTIAL" and len(calls) == 1
    expected = args["request_ref"] if action == "EXECUTE" else recovered["handoff"]["request_ref"]
    assert calls[0]["request_ref"] == expected
    assert calls[0]["automatic_origin_ref"] is None
