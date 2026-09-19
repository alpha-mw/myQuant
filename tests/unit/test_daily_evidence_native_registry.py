"""Fixed native composition tests; fixture core here is not full native DAG proof."""

from test_daily_evidence_dashboard_history import backlog
from test_daily_evidence_dashboard_adapter import ref
from scripts.daily_native_registry import NativeDailyInputs, NativeDailyRegistry
from quant_investor.operations.daily_contract import EOD_NODE_IDS
from quant_investor.operations.daily_runner import DayRunner
from test_daily_evidence_runner import FixtureAdapter
from test_daily_evidence_dag_journal import request


def test_deferred_construction_occurs_after_dependencies(tmp_path):
    runner = DayRunner(str(tmp_path), "20260904", {})
    calls, constructed = [], []

    def resolve(node, completed):
        from quant_investor.operations.daily_contract import GRAPH

        spec = next(s for s in GRAPH if s.node_id == node)
        assert all(completed[p]["state"] == "SUCCEEDED" for p in spec.requires)
        constructed.append(node)
        return {**request(), "node_id": node}, FixtureAdapter(tmp_path, node, calls)

    with runner.journal.locked():
        result = runner.run_locked({}, resolve=resolve)
    assert set(constructed) == EOD_NODE_IDS
    assert set(calls) == EOD_NODE_IDS
    assert result["completion_ref"] is None


def test_native_registry_constructs_dashboard_with_shared_day_lock(tmp_path):
    proof = backlog(tmp_path)
    plan = proof["sessions"][0]["plan"]
    # All downstream selected refs are real generated native fixture inputs.
    inputs = NativeDailyInputs(
        factor_pointer_sha256="a" * 64,
        release_ref=ref(tmp_path, "fixtures/release.json"),
        research_request_ref=ref(tmp_path, "fixtures/release.json"),
        store_arguments={},
        store_plan_ref={"path": plan["plan_path"], "sha256": plan["plan_sha256"]},
        event_pointer_sha256="b" * 64,
        market_snapshot_ref=ref(tmp_path, "data/parquet/cn/_snapshots/synthetic-20260824.json"),
        benchmark_ref=ref(tmp_path, "portfolio_dashboard/inputs/cn_index_benchmark.csv"),
        risk_free_ref=ref(tmp_path, "portfolio_dashboard/inputs/cn_govt_bond_yield.csv"),
        calendar_ref=ref(tmp_path, "fixtures/calendar-2026-08-24.json"),
    )
    registry = NativeDailyRegistry(str(tmp_path), "20260824", inputs)
    assert registry.node_ids == EOD_NODE_IDS
    with registry.runner.journal.locked():
        template, adapter = registry.resolve("dashboard", {})
        assert adapter.journal is registry.runner.journal
        adapter.execute(template)
        assert adapter.probe(template).outcome.state.value == "SUCCEEDED"
        assert registry.resolve("dashboard", {}) == (template, adapter)


def test_failed_core_prepare_is_not_cached(tmp_path, monkeypatch):
    import scripts.daily_native_registry as module
    from types import SimpleNamespace
    from quant_investor.operations.daily_journal import DailyJournal

    registry = object.__new__(NativeDailyRegistry)
    registry.workspace, registry.trade_date = str(tmp_path), "20260904"
    registry.runner = SimpleNamespace(journal=DailyJournal(str(tmp_path), "20260904"))
    registry.inputs = SimpleNamespace(
        factor_pointer_sha256="a" * 64,
        release_ref={"path": "release.json", "sha256": "b" * 64},
        next_session_calendar_proof_ref=None,
        next_session_calendar_failure_ref=None,
    )
    registry.adapters, registry.templates, registry.core = {}, {}, None

    class UnavailableCore:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, journal):
            raise ValueError("source unavailable")

    monkeypatch.setattr(module, "CoreContext", UnavailableCore)
    import pytest

    with registry.runner.journal.locked(), pytest.raises(ValueError, match="source unavailable"):
        registry.resolve("calendar", {})
    assert registry.core is None
    assert registry.adapters == {}
