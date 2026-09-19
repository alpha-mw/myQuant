"""Native current Dashboard publication and crash recovery, no fake publishers."""

import pytest
from test_daily_evidence_dashboard_history import backlog
from test_daily_evidence_dashboard_adapter import adapter
from _native_daily_store_fixture import DAYS
from scripts.daily_dashboard_adapter import CurrentDashboardAdapter
from export_cn_aggressive_dashboard_data import _expected_output_paths


def test_native_current_publication_and_retained_replay(tmp_path, monkeypatch):
    proof = backlog(tmp_path)
    node = adapter(tmp_path, DAYS[-1], proof["sessions"][0]["plan"], CurrentDashboardAdapter)
    request = node.template()
    assert node.probe(request).safe_to_execute
    with node.journal.locked():
        node.execute(request)
    result = node.probe(request)
    assert result.outcome.state.value == "SUCCEEDED"
    paths = _expected_output_paths(tmp_path)
    assert all(p.exists() for p in paths)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    import export_cn_aggressive_dashboard_data as exporter

    def unexpected(**kwargs):
        raise AssertionError("native publication must not repeat sealed output")

    monkeypatch.setattr(exporter, "publish_bundle_pair", unexpected)
    with node.journal.locked():
        node.execute(request)
    assert node.probe(request) == result
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    paths[0].write_bytes(b"later current UI")
    assert node.probe(request) == result


def test_crash_after_native_publication_recovers_exact_content(tmp_path, monkeypatch):
    proof = backlog(tmp_path)
    node = adapter(tmp_path, DAYS[-1], proof["sessions"][0]["plan"], CurrentDashboardAdapter)
    request = node.template()
    original = node.journal.storage.write

    def fail_receipt(path, raw):
        if path == node.publication_path:
            raise OSError("synthetic crash before publication receipt")
        return original(path, raw)

    with node.journal.locked():
        monkeypatch.setattr(node.journal.storage, "write", fail_receipt)
        with pytest.raises(OSError, match="synthetic crash"):
            node.execute(request)
    expected = {p: p.read_bytes() for p in _expected_output_paths(tmp_path)}
    assert node.probe(request).recovery_only
    monkeypatch.setattr(node.journal.storage, "write", original)
    benchmark = tmp_path / "portfolio_dashboard/inputs/cn_index_benchmark.csv"
    before_benchmark = benchmark.read_bytes()
    benchmark.write_bytes(before_benchmark + b"\n")
    from quant_investor.operations.daily_contract import ContractError

    with node.journal.locked(), pytest.raises(ContractError, match="SOURCE_SHA_MISMATCH"):
        node.execute(request)
    assert expected == {p: p.read_bytes() for p in expected}
    benchmark.write_bytes(before_benchmark)
    with node.journal.locked():
        node.execute(request)
    assert node.probe(request).outcome.state.value == "SUCCEEDED"
    assert expected == {p: p.read_bytes() for p in expected}


def test_five_native_current_dashboard_publications(tmp_path):
    from _native_daily_store_fixture import NativeStoreFixture, write
    from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter
    import json

    fixture = NativeStoreFixture(tmp_path)
    release = {
        "path": "fixtures/release.json",
        "sha256": write(tmp_path / "fixtures/release.json", {"synthetic": True}),
    }
    completed = []
    for day in DAYS:
        args = fixture.advance(day)
        plan = prepare_store_plan(args)
        store = StoreCloseAdapter(
            arguments=args,
            trade_date=day.replace("-", ""),
            plan_ref={"path": plan["plan_path"], "sha256": plan["plan_sha256"]},
            release_ref=release,
        )
        store.execute(store.template())
        node = adapter(tmp_path, day, plan, CurrentDashboardAdapter)
        request = node.template()
        with node.journal.locked():
            node.execute(request)
        result = node.probe(request)
        assert result.outcome.state.value == "SUCCEEDED"
        paths = _expected_output_paths(tmp_path)
        v2 = json.loads(paths[2].read_bytes())
        selector = json.loads(paths[4].read_bytes())
        assert v2["freshness"]["mark_as_of"] == day
        assert selector["v2_content_sha256"] == v2["content_sha256"]
        completed.append((node, request, result))
    for node, request, result in completed:
        assert node.probe(request) == result
