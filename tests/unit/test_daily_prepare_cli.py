"""Installed command dispatch with an explicit release-verifier seam; no install claim."""

import json
import subprocess
import sys

from quant_investor.cli import daily_prepare as cli
from quant_investor.operations.daily_preparation_contract import PreparationError
from _daily_preparation_fixture import config, calendar


def arguments(root):
    _, ref = config(root)
    refs = calendar(root)
    return [
        "--workspace-root",
        str(root),
        "--config",
        ref["path"],
        "--expected-config-sha256",
        ref["sha256"],
        "--calendar",
        refs[0]["path"],
        "--expected-calendar-sha256",
        refs[0]["sha256"],
        "--raw-calendar",
        refs[1]["path"],
        "--expected-raw-calendar-sha256",
        refs[1]["sha256"],
        "--release-repository-root",
        str(root),
    ]


def test_cli_verifies_install_before_dispatch(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path)
    calls = []

    def verify(*args, **kwargs):
        calls.append("installed")
        return {"state": "PASS"}

    def prepare(**kwargs):
        assert calls == ["installed"]
        calls.append("prepare")
        return {"status": "REGISTERED_INPUTS_ONLY", "execution_authorized": False}

    monkeypatch.setattr(cli, "verify_running_release_install_input", verify)
    monkeypatch.setattr(cli, "prepare_daily_request", prepare)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "REGISTERED_INPUTS_ONLY"
    assert calls == ["installed", "prepare"]


def test_cli_reports_exact_bounded_missing_dates(tmp_path, monkeypatch, capsys):
    import pytest

    args = arguments(tmp_path)
    monkeypatch.setattr(cli, "verify_running_release_install_input", lambda *a, **k: {})

    def blocked(**kwargs):
        raise PreparationError(
            "PREPARATION_NATIVE_DATES_MISSING",
            missing_event_dates=["20260824"],
            missing_benchmark_dates=[],
        )

    monkeypatch.setattr(cli, "prepare_daily_request", blocked)
    with pytest.raises(SystemExit) as caught:
        cli.main(args)
    assert caught.value.code == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "BLOCKED",
        "blocker_code": "PREPARATION_NATIVE_DATES_MISSING",
        "missing_event_dates": ["20260824"],
        "missing_benchmark_dates": [],
    }


def test_package_only_import_in_isolated_process():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; import quant_investor.cli.daily_prepare; "
            "assert not any(x == 'scripts' or x.startswith('scripts.') for x in sys.modules); "
            "print('PACKAGE_ONLY_PASS')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PACKAGE_ONLY_PASS"


def test_install_rejection_stops_before_registration(tmp_path, monkeypatch, capsys):
    import pytest

    args = arguments(tmp_path)

    def reject(*args, **kwargs):
        raise ValueError("synthetic install rejection")

    def forbidden(**kwargs):
        raise AssertionError("preparation must not run")

    monkeypatch.setattr(cli, "verify_running_release_install_input", reject)
    monkeypatch.setattr(cli, "prepare_daily_request", forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main(args)
    assert caught.value.code == 2
    assert json.loads(capsys.readouterr().out)["blocker_code"] == "PREPARATION_INPUT_REJECTED"
