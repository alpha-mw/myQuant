"""Native Dashboard custody survives mutable heads without rereading substitutes."""

import hashlib
from datetime import datetime, timezone

import pytest
from test_daily_evidence_native_dashboard import fixture
from scripts.daily_dashboard_capture import DailyDashboardCapture
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import ContractError, GRAPH_SHA256
from quant_investor.contracts import canonical_json_bytes


def context(root):
    f, paths, v1, v2, stamp = fixture(root)
    journal = DailyJournal(str(root), "20260824")
    capture = DailyDashboardCapture(str(root), journal)
    raw = canonical_json_bytes(
        {
            "schema_version": "cn-daily-node-request.v1",
            "trade_date": "20260824",
            "node_id": "dashboard",
            "graph_sha256": GRAPH_SHA256,
            "release_ref": {
                "path": "fixtures/release.json",
                "sha256": hashlib.sha256((root / "fixtures/release.json").read_bytes()).hexdigest(),
            },
            "adapter_sha256": "a" * 64,
            "policy_refs": {},
            "input_refs": {},
        }
    )
    path = str(journal.root / "inputs/dashboard-request.json")
    with journal.locked():
        journal.storage.write(path, raw)
    request_ref = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    return f, journal, capture, request_ref, v1, v2


def test_native_dashboard_capture_replays_after_heads_advance(tmp_path):
    f, journal, capture, request_ref, v1, v2 = context(tmp_path)
    before = datetime.now(timezone.utc).replace(microsecond=0)
    with journal.locked():
        result = capture.capture(v1=v1, v2=v2, request_ref=request_ref)
        assert capture.capture(v1=v1, v2=v2, request_ref=request_ref) == result
    assert datetime.fromisoformat(result["captured_at"]) >= before
    assert result["captured_at"][:10] != "2026-08-24"
    f.advance("2026-08-25")
    snapshot = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert capture.read(request_ref=request_ref) == result
    assert snapshot == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    victim = tmp_path / result["sources"][0]["retained_ref"]["path"]
    victim.write_bytes(victim.read_bytes() + b" ")
    with pytest.raises(ContractError, match="OBJECT_MISSING_OR_CHANGED"):
        capture.read(request_ref=request_ref)


def test_capture_refuses_stale_native_day(tmp_path):
    f, journal, capture, request_ref, v1, v2 = context(tmp_path)
    f.advance("2026-08-25")
    with journal.locked(), pytest.raises(ContractError, match="NATIVE_VALIDATION_FAILED"):
        capture.capture(v1=v1, v2=v2, request_ref=request_ref)
    assert capture.read(request_ref=request_ref) is None


def test_five_native_store_dashboard_captures(tmp_path):
    from _native_daily_store_fixture import DAYS
    from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter
    from cn_dashboard_common import build_bundle
    from cn_dashboard_v2 import build_v2_bundle
    import export_cn_aggressive_dashboard_data as exporter
    from zoneinfo import ZoneInfo

    f, journal, capture, request_ref, v1, v2 = context(tmp_path)
    captures = []
    for day in DAYS:
        if day != DAYS[0]:
            args = f.advance(day)
            prepared = prepare_store_plan(args)
            adapter = StoreCloseAdapter(
                arguments=args,
                trade_date=day.replace("-", ""),
                plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
                release_ref={
                    "path": "fixtures/release.json",
                    "sha256": hashlib.sha256(
                        (tmp_path / "fixtures/release.json").read_bytes()
                    ).hexdigest(),
                },
            )
            adapter.execute(adapter.template())
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            stamp = now.isoformat(timespec="seconds")
            paths = exporter._expected_output_paths(tmp_path)
            v1 = build_bundle(
                project_root=tmp_path,
                record_root=f.root,
                benchmark_path=tmp_path / "portfolio_dashboard/inputs/cn_index_benchmark.csv",
                generated_at=stamp,
                today=now.date(),
                benchmark_gap_policy="strict",
            )
            v2 = build_v2_bundle(
                project_root=tmp_path,
                v1_bundle=v1,
                v1_json_path=paths[0],
                record_root=f.root,
                generation_local_date=now.date(),
                generated_at=stamp,
                publication_attempt_id="dashboard-v2-synthetic-" + day.replace("-", ""),
                v1_json_bytes_override=exporter._render_json(v1),
            )
            journal = DailyJournal(str(tmp_path), day.replace("-", ""))
            capture = DailyDashboardCapture(str(tmp_path), journal)
            original = captures[0][0]._request(captures[0][1])
            raw = canonical_json_bytes({**original, "trade_date": journal.trade_date})
            path = str(journal.root / "inputs/dashboard-request.json")
            with journal.locked():
                journal.storage.write(path, raw)
            request_ref = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
        with journal.locked():
            value = capture.capture(v1=v1, v2=v2, request_ref=request_ref)
        captures.append((capture, request_ref, value))
    snapshot = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    for capture, request_ref, value in captures:
        assert capture.read(request_ref=request_ref) == value
    assert snapshot == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert len(captures) == 5


def test_capture_retries_only_unsealed_custody(tmp_path, monkeypatch):
    f, journal, capture, request_ref, v1, v2 = context(tmp_path)
    original = journal.storage.write

    def fail_receipt(path, raw):
        if path == capture.receipt_path:
            raise OSError("synthetic crash before receipt")
        return original(path, raw)

    with journal.locked():
        monkeypatch.setattr(journal.storage, "write", fail_receipt)
        with pytest.raises(OSError, match="synthetic crash"):
            capture.capture(v1=v1, v2=v2, request_ref=request_ref)
        assert capture.read(request_ref=request_ref) is None
        monkeypatch.setattr(journal.storage, "write", original)
        value = capture.capture(v1=v1, v2=v2, request_ref=request_ref)
        assert capture.read(request_ref=request_ref) == value
    assert value["authority"]["actual_holdings_mutation"] is False
