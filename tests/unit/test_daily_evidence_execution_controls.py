"""Exact initial-control bytes are retained read-only, without native admission."""

import hashlib
import json
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.output import CommandError
from quant_investor.operations.execution_controls import read_execution_controls
from quant_investor.operations.execution_controls import (
    verify_execution_install_and_research_policies,
)
from quant_investor.operations.daily_contract import ContractError
from test_daily_evidence_production_request import request, recipe


def fixture(root):
    def put(path, value):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        parent = target.parent
        while parent != root:
            parent.chmod(0o700)
            parent = parent.parent
        raw = canonical_json_bytes(value)
        target.write_bytes(raw)
        target.chmod(0o600)
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}

    leaf = put("input.json", {"fixture": "native admission deliberately not asserted"})
    prior = put(
        "results/operations/daily_production/CN/20260903/completion.v1.json", {"fixture": "prior"}
    )
    value = recipe()

    def replace_refs(row):
        if type(row) is dict:
            if set(row) == {"path", "sha256"}:
                return prior if row["path"].endswith("completion.v1.json") else leaf
            return {key: replace_refs(child) for key, child in row.items()}
        return row

    value = replace_refs(value)
    recipe_ref = put("recipe.json", value)
    req = request()
    req["release_install_ref"] = leaf
    req["recipe_ref"] = recipe_ref
    return put("request.json", req), put


def test_initial_controls_are_readonly_and_documents_are_copies(tmp_path, monkeypatch):
    ref, _ = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("initial controls attempted maintenance or network")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr(
        "quant_investor.market.daily_maintenance.run_cn_daily_maintenance", forbidden
    )
    value = read_execution_controls(workspace=str(tmp_path), request_ref=ref)
    assert value.document(value.request_ref)["action"] == "EXECUTE"
    changed = value.document(value.recipe_ref)
    changed["target_trade_date"] = "19990101"
    assert value.document(value.recipe_ref)["target_trade_date"] == "20260904"
    value.recheck()
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_controls_reject_source_changed_after_capture(tmp_path):
    ref, _ = fixture(tmp_path)
    value = read_execution_controls(workspace=str(tmp_path), request_ref=ref)
    (tmp_path / "input.json").write_bytes(b"changed")
    with pytest.raises(CommandError, match="EXECUTION_CONTROL_CHANGED"):
        value.recheck()


def test_controls_reject_missing_declared_source_without_search(tmp_path):
    ref, _ = fixture(tmp_path)
    (tmp_path / "input.json").rename(tmp_path / "other.json")
    with pytest.raises(CommandError, match="EXECUTION_CONTROL_SHA_INVALID"):
        read_execution_controls(workspace=str(tmp_path), request_ref=ref)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "context_ref",
        "install",
        "release",
        "policy",
        "missing_commit",
        "wrong_commit",
        "missing_capture_parent",
    ],
)
def test_native_release_and_policy_binding_with_controlled_install_seam(
    tmp_path, monkeypatch, fault
):
    from test_unified_factor_manifest import _release
    from quant_investor.system.store import object_ref_for_artifact
    from quant_investor.intelligence.storage import approved_theme_policy_v2

    ref, put = fixture(tmp_path)
    req = json.loads((tmp_path / ref["path"]).read_bytes())
    rec = json.loads((tmp_path / req["recipe_ref"]["path"]).read_bytes())
    install_ref = put(
        "install.json", {"release_install_evidence": {"payload": {"final_commit": "c" * 40}}}
    )
    req["release_install_ref"] = rec["release_install_ref"] = install_ref
    context = {
        "schema_version": "cn-daily-factor-loop.v1",
        "release_install_input_ref": req["release_install_ref"],
        "release_repository_root": "/fixture/repository",
        "release_commit": "c" * 40,
        "calendar_capture_parent": str(tmp_path.resolve() / "calendar-captures"),
    }
    if fault == "missing_commit":
        context.pop("release_commit")
    elif fault == "wrong_commit":
        context["release_commit"] = "d" * 40
    elif fault == "missing_capture_parent":
        context.pop("calendar_capture_parent")
    if fault == "context_ref":
        context["release_install_input_ref"] = {"path": "other.json", "sha256": "a" * 64}
    rec["factor_loop_context_ref"] = put("context.json", context)
    release = _release()
    rec["release_ref"] = put("release.json", release)
    policy = approved_theme_policy_v2()
    if fault == "policy":
        policy = {"fixture": "unapproved"}
    rec["policy_refs"]["research"] = put("research-policy.json", policy)
    req["recipe_ref"] = put("recipe.json", rec)
    ref = put("request.json", req)
    installed = {"state": "PASS", "release_ref": object_ref_for_artifact(release)}
    if fault == "install":
        installed["state"] = "FAILED"
    elif fault == "release":
        installed["release_ref"] = {**installed["release_ref"], "artifact_id": "other"}
    calls = []

    def native_context(**kwargs):
        calls.append(kwargs)
        return context, installed

    monkeypatch.setattr(
        "quant_investor.market.daily_factor_loop.read_factor_loop_context", native_context
    )
    controls = read_execution_controls(workspace=str(tmp_path), request_ref=ref)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    if fault:
        with pytest.raises(ContractError):
            verify_execution_install_and_research_policies(controls)
    else:
        assert verify_execution_install_and_research_policies(controls)["installation"] == installed
        assert calls[0]["context_sha256"] == rec["factor_loop_context_ref"]["sha256"]
    if fault == "context_ref":
        assert calls == []
    if fault in {"missing_commit", "wrong_commit", "missing_capture_parent"}:
        import test_daily_evidence_store_materialization  # noqa: F401
        from scripts.daily_materialization import execute_daily_recipe
        from quant_investor.operations.dependency_diagnostics import DependencyInputError

        def forbidden(*args, **kwargs):
            pytest.fail("invalid producer dependency reached a writer or loop")

        monkeypatch.setattr("quant_investor.market.daily_factor_loop.DailyFactorLoop", forbidden)
        monkeypatch.setattr(
            "quant_investor.market.daily_maintenance.run_cn_daily_maintenance", forbidden
        )
        monkeypatch.setattr(
            "scripts.daily_store_materialization.verify_initial_store_controls", forbidden
        )
        with pytest.raises(DependencyInputError):
            execute_daily_recipe(workspace=str(tmp_path), request_ref=ref)
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert not (tmp_path / "calendar-captures").exists()


@pytest.mark.parametrize("fault", [None, "budget", "provider", "missing", "sha"])
def test_v2_policy_is_validated_before_maintenance(tmp_path, monkeypatch, fault):
    from quant_investor.operations.daily_contract import ContractError

    ref, put = fixture(tmp_path)
    req = json.loads((tmp_path / ref["path"]).read_bytes())
    rec = json.loads((tmp_path / req["recipe_ref"]["path"]).read_bytes())
    policy = {
        "schema_version": "cn-daily-theme-acquisition.v1",
        "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
        "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
        "maximum_companies": 100,
    }
    if fault == "budget":
        policy["maximum_companies"] = 101
    elif fault == "provider":
        policy["provider_priority"] = ["CALLER_PROVIDER"]
    rec["schema_version"] = "cn-daily-execute-recipe.v2"
    rec["theme_acquisition_ref"] = put("theme-policy.json", policy)
    rec["research_sources"]["theme_source_ref"] = None
    req["recipe_ref"] = put("recipe.json", rec)
    ref = put("request.json", req)
    if fault == "missing":
        (tmp_path / "theme-policy.json").unlink()
    elif fault == "sha":
        (tmp_path / "theme-policy.json").write_bytes(b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("preflight reached maintenance, loop construction or network")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr(
        "quant_investor.market.daily_maintenance.run_cn_daily_maintenance", forbidden
    )
    monkeypatch.setattr("quant_investor.market.daily_factor_loop.DailyFactorLoop", forbidden)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if fault:
        with pytest.raises((ContractError, CommandError)):
            read_execution_controls(workspace=str(tmp_path), request_ref=ref)
    else:
        controls = read_execution_controls(workspace=str(tmp_path), request_ref=ref)
        assert (
            controls.document(
                (rec["theme_acquisition_ref"]["path"], rec["theme_acquisition_ref"]["sha256"])
            )
            == policy
        )
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
