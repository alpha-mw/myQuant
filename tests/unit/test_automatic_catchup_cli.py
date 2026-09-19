"""Public schema and 0/2/3 exit contracts with explicit installed-operation seams."""

from contextlib import contextmanager
from copy import deepcopy
import json

import pytest
from _public_catchup_fixture import put
from test_automatic_catchup_resolution import fixture, resolve
from quant_investor.cli import daily_production as public
from quant_investor.cli.main import main
from scripts.daily_automatic_catchup import _wrapper


def args(root, request, ref):
    install = request["release_install_ref"]
    return [
        "production",
        "daily-close",
        "--workspace-root",
        str(root),
        "--request",
        ref["path"],
        "--expected-request-sha256",
        ref["sha256"],
        "--release-repository-root",
        str(root),
        "--release-install-input",
        install["path"],
        "--expected-release-install-input-sha256",
        install["sha256"],
    ]


@pytest.mark.parametrize(
    "fault,expected", [(None, 0), ("missing_inputs", 2), ("wrapper", 3), ("missing_reader", 3)]
)
def test_public_automatic_plan_schema_exit_and_readonly(
    tmp_path, monkeypatch, capsys, fault, expected
):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    if fault == "missing_inputs":
        collection = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
        del collection["recipes"]["20260828"]
        request["recipe_ref"] = put(tmp_path, request["recipe_ref"]["path"], collection)
        ref = put(tmp_path, ref["path"], request)
    context = resolve(tmp_path, ref, request)
    value = _wrapper(context, request_ref=ref)
    if fault == "wrapper":
        value = {**deepcopy(value), "target_trade_date": "20260827"}

    @contextmanager
    def installed(**kwargs):
        operations = {"daily_close": lambda **kw: value}
        if fault != "missing_reader":
            operations["automatic_resolution"] = lambda **kw: context["resolution"]
        yield operations

    monkeypatch.setattr(public, "verified_native_context", installed)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    if expected:
        with pytest.raises(SystemExit) as stopped:
            main(args(tmp_path, request, ref))
        assert stopped.value.code == expected
    else:
        main(args(tmp_path, request, ref))
    output = json.loads(capsys.readouterr().out)
    if expected != 3:
        assert output["status"] == ("BLOCKED" if expected == 2 else "PLANNED")
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_expected_installed_busy_error_keeps_code_without_host_class_identity(
    tmp_path, monkeypatch, capsys
):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    request["action"] = "CATCH_UP"
    ref = put(tmp_path, ref["path"], request)

    def dispatch(**kwargs):
        error = ValueError("AUTO_PENDING_REQUEST_CONFLICT")
        error.code = "AUTO_PENDING_REQUEST_CONFLICT"
        error.fields = {"pending_request_ref": ref}
        raise error

    @contextmanager
    def installed(**kwargs):
        yield {"daily_close": dispatch}

    monkeypatch.setattr(public, "verified_native_context", installed)
    with pytest.raises(SystemExit) as stopped:
        main(args(tmp_path, request, ref))
    assert stopped.value.code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["blocker_code"] == "AUTO_PENDING_REQUEST_CONFLICT"
    assert output["pending_request_ref"] == ref


@pytest.mark.parametrize("native_provenance", [False, True])
def test_public_current_incomplete_result_and_provenance_validation(
    tmp_path, monkeypatch, capsys, native_provenance
):
    from test_automatic_catchup_execution import execution_fixture, run
    from scripts.daily_automatic_catchup import inspect_automatic_resolution
    from quant_investor.operations.daily_contract import EOD_NODE_IDS

    request, ref, _, _, _ = execution_fixture(tmp_path, monkeypatch)
    result = run(tmp_path, request, ref)

    # Installed provenance/EOD admission are explicit seams; this is not a live run.
    @contextmanager
    def installed(**kwargs):
        yield {
            "daily_close": lambda **kw: result,
            "automatic_resolution": lambda **kw: inspect_automatic_resolution(**kw, synthetic=True),
            "completion_replay": lambda **kw: {
                **kw,
                "native_replay_validated": True,
                "validated_nodes": sorted(EOD_NODE_IDS),
                "synthetic": native_provenance,
            },
        }

    monkeypatch.setattr(public, "verified_native_context", installed)
    with pytest.raises(SystemExit) as stopped:
        main(args(tmp_path, request, ref))
    assert stopped.value.code == (3 if native_provenance else 2)
    output = json.loads(capsys.readouterr().out)
    if not native_provenance:
        assert output["status"] == "PARTIAL" and output["result"]["business_state"] == "INCOMPLETE"
