"""Existing Factor context preflight remains read-only and precedes directory creation."""

import hashlib
from pathlib import Path
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.market import daily_factor_loop as loop
from quant_investor.operations.daily_contract import ContractError
from test_daily_evidence_production_request import request, recipe
from quant_investor.operations.execution_recipe import validate_execution_recipe


def context(root):
    release = canonical_json_bytes({})
    (root / "release.json").write_bytes(release)
    (root / "release.json").chmod(0o600)
    value = {
        "schema_version": "cn-daily-factor-loop.v1",
        "release_install_input_ref": {
            "path": "release.json",
            "sha256": hashlib.sha256(release).hexdigest(),
        },
        "release_repository_root": str(root),
    }
    raw = canonical_json_bytes(value)
    path = root / "context.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return {
        "workspace_root": str(root),
        "context_path": str(path),
        "context_sha256": hashlib.sha256(raw).hexdigest(),
    }, value


def test_context_preflight_reuses_native_installation_check_without_writes(tmp_path, monkeypatch):
    args, value = context(tmp_path)
    calls = []
    monkeypatch.setattr(
        loop,
        "verify_running_release_install_input",
        lambda *a, **k: (calls.append((a, k)) or {"state": "PASS"}),
    )
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    loaded, verified = loop.read_factor_loop_context(**args)
    assert loaded == value and verified["state"] == "PASS" and len(calls) == 1
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}


@pytest.mark.parametrize("failure", ["context_sha", "runtime"])
def test_invalid_context_does_not_create_run_directory(tmp_path, monkeypatch, failure):
    args, _ = context(tmp_path)
    if failure == "context_sha":
        args["context_sha256"] = "a" * 64
    monkeypatch.setattr(
        loop, "verify_running_release_install_input", lambda *a, **k: {"state": "BLOCKED"}
    )
    run_root = tmp_path / "new-run"
    with pytest.raises(ValueError):
        loop.DailyFactorLoop(**args, run_root=str(run_root))
    assert not run_root.exists()


def test_execute_requires_existing_loop_context_but_plan_does_not():
    value = recipe()
    value["factor_loop_context_ref"] = None
    with pytest.raises(ContractError, match="EXECUTE_CONTEXT_REQUIRED"):
        validate_execution_recipe(value, request=request("EXECUTE"))
    assert (
        validate_execution_recipe(value, request=request("PLAN"))["factor_loop_context_ref"] is None
    )


def test_v2_fixture_mode_is_rejected_before_install_or_directory_write(tmp_path, monkeypatch):
    args, value = context(tmp_path)
    value.update(
        schema_version="cn-daily-factor-loop.v2",
        release_commit="b" * 40,
        calendar_capture_parent=str(tmp_path / "capture"),
        initial_calendar_receipt_ref=None,
        next_session_calendar_mode="SYNTHETIC_FIXTURE_ONLY",
    )
    raw = canonical_json_bytes(value)
    Path(args["context_path"]).write_bytes(raw)
    args["context_sha256"] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(
        loop,
        "verify_running_release_install_input",
        lambda *a, **k: pytest.fail("installation reached before fixture preflight"),
    )
    run_root = tmp_path / "new-run"
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    with pytest.raises(ContractError, match="PROVENANCE_UNAVAILABLE"):
        loop.DailyFactorLoop(**args, run_root=str(run_root))
    assert not run_root.exists()
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
