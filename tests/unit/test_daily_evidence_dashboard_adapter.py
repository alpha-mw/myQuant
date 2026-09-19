"""Real historical Dashboard node execution, replay and interrupted-custody recovery."""

import hashlib
import pytest
from test_daily_evidence_dashboard_history import backlog
from _native_daily_store_fixture import DAYS
from scripts.daily_dashboard_adapter import HistoricalDashboardAdapter
from quant_investor.operations.daily_journal import DailyJournal


def ref(root, path):
    return {"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()}


def adapter(root, day, plan, adapter_type=HistoricalDashboardAdapter):
    journal = DailyJournal(str(root), day.replace("-", ""))
    value = adapter_type(
        workspace=str(root),
        journal=journal,
        release_ref=ref(root, "fixtures/release.json"),
        plan_ref={"path": plan["plan_path"], "sha256": plan["plan_sha256"]},
        market_ref=ref(
            root, "data/parquet/cn/_snapshots/synthetic-" + day.replace("-", "") + ".json"
        ),
        benchmark_ref=ref(root, "portfolio_dashboard/inputs/cn_index_benchmark.csv"),
        risk_free_ref=ref(root, "portfolio_dashboard/inputs/cn_govt_bond_yield.csv"),
    )
    with journal.locked():
        value.prepare()
    return value


def test_five_dashboard_nodes_on_one_native_store_commit(tmp_path, monkeypatch):
    proof = backlog(tmp_path)
    nodes = []
    for day in DAYS:
        node = adapter(tmp_path, day, proof["sessions"][0]["plan"])
        request = node.template()
        assert node.probe(request).safe_to_execute
        with node.journal.locked():
            node.execute(request)
        result = node.probe(request)
        assert result.outcome.state.value == "SUCCEEDED"
        nodes.append((node, request, result))
    snapshot = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    import scripts.daily_dashboard_adapter as module

    def unexpected(**kwargs):
        raise AssertionError("native renderer must not rerun sealed outputs")

    monkeypatch.setattr(module, "build_bundle", unexpected)
    for node, request, result in nodes:
        with node.journal.locked():
            node.execute(request)
        assert node.probe(request) == result
    assert snapshot == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    benchmark = tmp_path / "portfolio_dashboard/inputs/cn_index_benchmark.csv"
    benchmark.write_bytes(benchmark.read_bytes() + b"\n")
    for node, request, result in nodes:
        assert node.probe(request) == result


def test_restart_preserves_render_intent_after_unsealed_crash(tmp_path, monkeypatch):
    proof = backlog(tmp_path)
    plan = proof["sessions"][0]["plan"]
    node = adapter(tmp_path, DAYS[0], plan)
    request = node.template()

    def failed(**kwargs):
        raise OSError("synthetic failure before custody")

    monkeypatch.setattr(node.capture, "capture", failed)
    with node.journal.locked(), pytest.raises(OSError, match="synthetic failure"):
        node.execute(request)
    fresh = adapter(tmp_path, DAYS[0], plan)
    assert fresh.template() == request
    assert fresh.probe(request).safe_to_execute
    with fresh.journal.locked():
        fresh.execute(request)
    assert fresh.probe(request).outcome.state.value == "SUCCEEDED"


def test_running_attempt_adopts_native_capture_without_second_render(tmp_path, monkeypatch):
    from quant_investor.operations.daily_runner import DayRunner
    import scripts.daily_dashboard_adapter as module

    proof = backlog(tmp_path)
    node = adapter(tmp_path, DAYS[0], proof["sessions"][0]["plan"])
    request = node.template()
    runner = DayRunner(str(tmp_path), node.journal.trade_date, {"dashboard": node})
    runner.journal = node.journal
    with node.journal.locked():
        original = node.journal.begin(request, reconciled_no_write=True)
        node.execute(request)

        # Simulate death after native custody but before journal terminal.
        def unexpected(**kwargs):
            raise AssertionError("sealed native renderer rerun")

        monkeypatch.setattr(module, "build_bundle", unexpected)
        result = runner._node(request, node, resume=True)
        assert result["terminal"]["state"] == "SUCCEEDED"
        assert result["start_ref"] == original["start_ref"]
        assert result["command_status"] == "ADOPTED"
