"""Fixed installed inspection-reader wire; installed/EOD provenance seams are explicit."""

from contextlib import contextmanager
from copy import deepcopy
import json

import pytest
from _public_catchup_fixture import put
from test_automatic_catchup_execution import execution_fixture, run
from test_daily_launch_inspection import serving_state
from test_daily_evidence_public_catchup import snapshot
from quant_investor.cli import daily_launch as cli
from quant_investor.cli import daily_production as public
from quant_investor.cli.output import CommandError
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import EOD_NODE_IDS
from scripts import daily_launch_inspection as native
from scripts import daily_automatic_catchup as automatic
from scripts import daily_dashboard_publication as publication

INSPECTION = (
    "data/private/cn_daily_maintenance/launcher_attempts/"
    "slot-2020-20260828T140000Z-77/inspection.stdout.json"
)


def fixture(root, monkeypatch, mode):
    request, ref, _, calls, _ = execution_fixture(root, monkeypatch)
    monkeypatch.setattr(native, "verify_recipe_static_controls", lambda **kw: None)
    if mode != 11:
        run(root, request, ref)
        state = "RECORDED_EOD_PUBLICATION" if mode == 0 else "EVIDENCE_SEALED_PUBLICATION_PENDING"
        monkeypatch.setattr(
            publication,
            "observed_serving_status",
            lambda workspace, day: serving_state(root, day, state),
        )

    @contextmanager
    def installed(**kwargs):
        yield {
            "daily_launch_inspection": lambda **kw: native.inspect_daily_launch(
                **kw, synthetic=True
            ),
            "automatic_resolution": lambda **kw: automatic.inspect_automatic_resolution(
                **kw, synthetic=True
            ),
            "completion_replay": lambda **kw: {
                **kw,
                "native_replay_validated": True,
                "validated_nodes": sorted(EOD_NODE_IDS),
                "synthetic": False,
            },
        }

    monkeypatch.setattr(cli, "verified_native_context", installed)
    args = [
        "--workspace-root",
        str(root),
        "--request",
        ref["path"],
        "--expected-request-sha256",
        ref["sha256"],
        "--release-repository-root",
        str(root),
        "--release-install-input",
        request["release_install_ref"]["path"],
        "--expected-release-install-input-sha256",
        request["release_install_ref"]["sha256"],
    ]
    return request, ref, args, calls


@pytest.mark.parametrize("mode", [0, 10, 11])
def test_exact_inspection_file_revalidated_before_mode_or_completed_emit(
    tmp_path, monkeypatch, capsys, mode
):
    _, _, args, calls = fixture(tmp_path, monkeypatch, mode)
    before = snapshot(tmp_path)
    assert cli.main(args + ["--mode", "inspect"]) == mode
    raw = capsys.readouterr().out.encode()
    document = json.loads(raw)
    assert raw == canonical_json_bytes(document) and snapshot(tmp_path) == before
    selected = put(tmp_path, INSPECTION, raw)
    suffix = ["--inspection", selected["path"], "--expected-inspection-sha256", selected["sha256"]]
    before = snapshot(tmp_path)
    assert cli.main(args + ["--mode", "validate", *suffix]) == mode
    assert capsys.readouterr().out == "" and snapshot(tmp_path) == before
    if mode == 0:
        assert cli.main(args + ["--mode", "emit", *suffix]) == 0
        assert json.loads(capsys.readouterr().out) == document["result"]
    else:
        with pytest.raises(SystemExit) as stopped:
            cli.main(args + ["--mode", "emit", *suffix])
        assert stopped.value.code == 2
        capsys.readouterr()
    assert snapshot(tmp_path) == before and len(calls) == (0 if mode == 11 else 2)


@pytest.mark.parametrize("fault", ["sha", "mode", "request", "outside", "extra"])
def test_saved_inspection_tampering_cannot_authorize_shell_branch(
    tmp_path, monkeypatch, capsys, fault
):
    _, _, args, calls = fixture(tmp_path, monkeypatch, 11)
    cli.main(args + ["--mode", "inspect"])
    value = json.loads(capsys.readouterr().out)
    if fault == "mode":
        value["mode"] = "LOCAL_REPAIR"
    elif fault == "request":
        value["request_ref"] = {**value["request_ref"], "sha256": "f" * 64}
    elif fault == "extra":
        value["allow_live"] = True
    selected = put(tmp_path, INSPECTION if fault != "outside" else "elsewhere.json", value)
    suffix = [
        "--inspection",
        selected["path"],
        "--expected-inspection-sha256",
        "f" * 64 if fault == "sha" else selected["sha256"],
    ]
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        cli.main(args + ["--mode", "validate", *suffix])
    assert stopped.value.code == 2
    assert snapshot(tmp_path) == before and calls == []


def test_internal_auto_recover_keeps_the_existing_no_producers_restriction(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls, validations = [], []
    result = {
        "schema_version": "cn-daily-automatic-result.v1",
        "action": "CATCH_UP",
        "status": "NO_ACTION",
        "result": {"execution_state": "NO_ACTION"},
    }
    monkeypatch.setattr(cli, "_validate_result", lambda *a: validations.append(a))
    ref = {"path": "request.json", "sha256": "a" * 64}
    code = cli._recover(
        SimpleNamespace(workspace_root=str(tmp_path)),
        {"schema_version": "cn-daily-launch-inspection.v1", "mode": "LOCAL_REPAIR"},
        ref,
        {"action": "CATCH_UP"},
        ref,
        {"daily_close": lambda **kwargs: calls.append(kwargs) or result},
    )
    assert code == 0 and len(validations) == 1
    assert calls == [
        {
            "workspace": str(tmp_path),
            "request_ref": ref,
            "release_install_ref": ref,
            "no_producers": True,
        }
    ]


def test_no_producers_invalid_for_plan_before_installed_bridge(tmp_path, monkeypatch):
    request, ref, _, _ = fixture(tmp_path, monkeypatch, 11)
    request = {**deepcopy(request), "action": "PLAN"}
    ref = put(tmp_path, ref["path"], request)
    monkeypatch.setattr(
        public, "verified_native_context", lambda **kw: pytest.fail("entered bridge")
    )
    with pytest.raises(CommandError, match="AUTO_NO_PRODUCERS_FLAG_INVALID"):
        public.run_daily_close(
            workspace=str(tmp_path),
            request_path=ref["path"],
            expected_request_sha256=ref["sha256"],
            release_repository_root=str(tmp_path),
            release_install_input_path=request["release_install_ref"]["path"],
            expected_release_install_input_sha256=request["release_install_ref"]["sha256"],
            no_producers=True,
        )
