"""Installed launcher bootstrap wire with controlled native install/completion gates."""

from contextlib import contextmanager
import json

import pytest

from quant_investor.cli import daily_launch as cli
from quant_investor.operations.daily_contract import EOD_NODE_IDS
from test_daily_bootstrap_launch import fixture, put, completed, native
from scripts import daily_catchup

INSPECTION = (
    "data/private/cn_daily_maintenance/launcher_attempts/"
    "slot-2020-20260904T140000Z-77/inspection.stdout.json"
)


def setup(root, monkeypatch, mode):
    request, ref, _, paths, recovered = fixture(root, monkeypatch, handoff=mode != 11)
    calls = []
    if mode == 11:
        monkeypatch.setattr(native, "_fresh_controls", lambda *a, **kw: None)
    if mode == 0:
        completed(root, monkeypatch, request, paths, recovered)
        monkeypatch.setattr(daily_catchup, "_serving_gate", lambda *a, **kw: True)
    result = native._result(
        request["target_trade_date"],
        completion_ref={
            "path": "results/operations/daily_production/CN/20260904/completion.v1.json",
            "sha256": "e" * 64,
        },
    )

    def dispatch(**kwargs):
        calls.append(kwargs)
        return result

    @contextmanager
    def installed(**kwargs):
        yield {
            "daily_launch_inspection": lambda **kw: native.inspect_bootstrap_launch(
                **kw, synthetic=True
            ),
            "daily_close": dispatch,
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
    return ref, args, calls


@pytest.mark.parametrize("mode", [0, 10, 11])
def test_bootstrap_inspection_and_explicit_recover_use_exact_scope(
    tmp_path, monkeypatch, capsys, mode
):
    _, args, calls = setup(tmp_path, monkeypatch, mode)
    assert cli.main(args + ["--mode", "inspect"]) == mode
    raw = capsys.readouterr().out.encode()
    value = json.loads(raw)
    assert value["schema_version"] == "cn-daily-launch-inspection.v2"
    saved = put(tmp_path, INSPECTION, raw)
    suffix = ["--inspection", saved["path"], "--expected-inspection-sha256", saved["sha256"]]
    assert cli.main(args + ["--mode", "validate", *suffix]) == mode
    if mode == 10:
        assert cli.main(args + ["--mode", "recover", *suffix]) == 0
        assert len(calls) == 1 and calls[0]["committed_recovery_only"] is True
        assert "no_producers" not in calls[0]
    else:
        with pytest.raises(SystemExit) as error:
            cli.main(args + ["--mode", "recover", *suffix])
        assert error.value.code == 2 and calls == []


def test_saved_inspection_cannot_move_to_an_identical_request_alias(tmp_path, monkeypatch, capsys):
    ref, args, calls = setup(tmp_path, monkeypatch, 10)
    cli.main(args + ["--mode", "inspect"])
    saved = put(tmp_path, INSPECTION, capsys.readouterr().out.encode())
    alias = put(tmp_path, "alias/request.json", (tmp_path / ref["path"]).read_bytes())
    args[args.index("--request") + 1] = alias["path"]
    with pytest.raises(SystemExit) as error:
        cli.main(
            args
            + [
                "--mode",
                "recover",
                "--inspection",
                saved["path"],
                "--expected-inspection-sha256",
                saved["sha256"],
            ]
        )
    assert error.value.code == 2 and calls == []


def test_recovery_scope_drift_blocks_before_private_dispatch(tmp_path, monkeypatch, capsys):
    _, args, calls = setup(tmp_path, monkeypatch, 10)
    cli.main(args + ["--mode", "inspect"])
    value = json.loads(capsys.readouterr().out)
    value["recovery_scope"] = "SERVING_ONLY"
    saved = put(tmp_path, INSPECTION, value)
    with pytest.raises(SystemExit) as error:
        cli.main(
            args
            + [
                "--mode",
                "recover",
                "--inspection",
                saved["path"],
                "--expected-inspection-sha256",
                saved["sha256"],
            ]
        )
    assert error.value.code == 2 and calls == []
