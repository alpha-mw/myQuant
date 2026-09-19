"""Installed source CLI contracts with an explicit native-context seam."""

from contextlib import contextmanager
import json
import subprocess
import sys

import pytest

from _daily_preparation_fixture import config, put
from quant_investor.cli import daily_sources as cli
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.source_slot_contract import RESULT_SCHEMA, SourceSlotError


def fixture(root, monkeypatch, mode="REQUEST_AVAILABLE"):
    cfg, ref = config(root)
    install = cfg["release_install_ref"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "mode": mode,
        "config_ref": ref,
        "release_install_ref": install,
        "trade_date": "20260825",
        "request_ref": (
            {"path": "requests/selected input.json", "sha256": "d" * 64}
            if mode == "REQUEST_AVAILABLE"
            else None
        ),
        "calendar_capture_ref": None,
        "preparation_commitment_ref": None,
        "authority": dict(FALSE_AUTHORITY),
    }
    calls = []

    @contextmanager
    def context(**kwargs):
        calls.append("verified_context")

        def operation(**kw):
            calls.append(kw)
            return dict(result)

        yield {"daily_source_inputs": operation}

    monkeypatch.setattr(cli, "verified_native_context", context)
    args = [
        "--workspace-root",
        str(root),
        "--config",
        ref["path"],
        "--expected-config-sha256",
        ref["sha256"],
        "--release-repository-root",
        str(root),
        "--release-install-input",
        install["path"],
        "--expected-release-install-input-sha256",
        install["sha256"],
    ]
    return args, result, calls


@pytest.mark.parametrize(
    "mode,code",
    [
        ("REQUEST_AVAILABLE", 0),
        ("NON_TRADING_DAY", 0),
        ("LOCAL_PREPARATION", 10),
        ("ACQUISITION_REQUIRED", 11),
    ],
)
def test_inspect_internal_modes_remain_exact_json(tmp_path, monkeypatch, capsys, mode, code):
    args, result, calls = fixture(tmp_path, monkeypatch, mode)
    assert cli.main(args + ["--mode", "inspect"]) == code
    assert json.loads(capsys.readouterr().out) == result
    assert calls[0] == "verified_context" and calls[1]["mode"] == "inspect"


def saved(root, result, name="source-inspection.stdout.json"):
    ref = put(
        root,
        "data/private/cn_daily_maintenance/launcher_attempts/slot-2020-20260825T122000Z-123/"
        + name,
        result,
    )
    return ["--inspection", ref["path"], "--expected-inspection-sha256", ref["sha256"]]


def test_select_is_only_two_validated_literal_fields(tmp_path, monkeypatch, capsys):
    args, result, _ = fixture(tmp_path, monkeypatch)
    assert cli.main(args + ["--mode", "select"] + saved(tmp_path, result)) == 0
    assert capsys.readouterr().out.splitlines() == [result["request_ref"]["path"], "d" * 64]


def test_nontrading_select_internal_code_has_no_request_text(tmp_path, monkeypatch, capsys):
    args, result, _ = fixture(tmp_path, monkeypatch, "NON_TRADING_DAY")
    assert cli.main(args + ["--mode", "select"] + saved(tmp_path, result)) == 10
    assert capsys.readouterr().out == ""


def test_saved_result_substitution_is_rejected(tmp_path, monkeypatch, capsys):
    args, result, _ = fixture(tmp_path, monkeypatch)
    old = {**result, "trade_date": "20260824"}
    with pytest.raises(SystemExit) as caught:
        cli.main(args + ["--mode", "emit"] + saved(tmp_path, old))
    assert caught.value.code == 2
    assert json.loads(capsys.readouterr().out)["blocker_code"] == "SOURCE_INSPECTION_CHANGED"


def test_no_provider_flag_reaches_native_provision(tmp_path, monkeypatch, capsys):
    args, _, calls = fixture(tmp_path, monkeypatch)
    assert cli.main(args + ["--mode", "provision", "--no-providers"]) == 0
    assert calls[1]["mode"] == "provision" and calls[1]["no_providers"] is True


@pytest.mark.parametrize("unexpected", [False, True])
def test_expected_blocker_and_unexpected_failure_have_distinct_exits(
    tmp_path, monkeypatch, capsys, unexpected
):
    args, _, _ = fixture(tmp_path, monkeypatch)

    @contextmanager
    def context(**kwargs):
        def fail(**kw):
            if unexpected:
                raise RuntimeError("private unexpected diagnostic")
            raise SourceSlotError(
                "SOURCE_HISTORICAL_EVENTS_MISSING", missing_event_dates=["20260824"]
            )

        yield {"daily_source_inputs": fail}

    monkeypatch.setattr(cli, "verified_native_context", context)
    with pytest.raises(SystemExit) as caught:
        cli.main(args + ["--mode", "inspect"])
    assert caught.value.code == (3 if unexpected else 2)
    out = capsys.readouterr()
    assert "private unexpected diagnostic" not in out.out + out.err
    if not unexpected:
        assert json.loads(out.out)["missing_event_dates"] == ["20260824"]


def test_source_cli_imports_no_unverified_scripts():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; import quant_investor.cli.daily_sources; "
            "assert not any(k == 'scripts' or k.startswith('scripts.') for k in sys.modules); "
            "print('PASS')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"
