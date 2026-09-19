"""Pre-maintenance Store validation uses native inputs and cannot publish a plan."""

import pytest
from test_daily_evidence_store_materialization import context
from scripts.daily_store_materialization import verify_initial_store_controls
from quant_investor.operations.daily_contract import ContractError


def inventory(root):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_native_store_preflight_does_not_plan_or_lock(tmp_path, monkeypatch):
    _, recovered, _, _ = context(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Store preflight attempted plan/write/lock")

    monkeypatch.setattr("scripts.daily_store_materialization.prepare_store_plan", forbidden)
    monkeypatch.setattr("scripts.manage_cn_strategy_records._operation_lock", forbidden)
    before = inventory(tmp_path)
    result = verify_initial_store_controls(workspace=str(tmp_path), recipe=recovered["recipe"])
    assert result["execution_authorized"] is False
    assert result["official_date"] == "2026-08-21"
    assert result["store_preimages"] == recovered["recipe"]["store_preimages"]
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("fault", ["path", "sha", "policy"])
def test_store_preflight_rejects_wrong_identity(tmp_path, fault):
    _, recovered, _, _ = context(tmp_path)
    recipe = recovered["recipe"]
    if fault == "path":
        recipe["store_preimages"]["store_pointer_ref"]["path"] = "other.json"
    elif fault == "sha":
        recipe["store_preimages"]["benchmark_pointer_ref"]["sha256"] = "a" * 64
    else:
        recipe["policy_refs"]["store"]["sha256"] = "a" * 64
    before = inventory(tmp_path)
    with pytest.raises((ContractError, ValueError, RuntimeError)):
        verify_initial_store_controls(workspace=str(tmp_path), recipe=recipe)
    assert inventory(tmp_path) == before
