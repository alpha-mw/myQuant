"""First-time native historical rendering follows verified Store commits."""

from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import pytest

from test_daily_evidence_native_store import run_sessions
from cn_dashboard_common import build_bundle, validate_bundle_shape, verify_source_refs
from cn_dashboard_v2 import build_v2_bundle, validate_v2_shape, verify_v2_source_refs
from quant_investor.market.market_data_reader import MarketDataReader
from export_cn_aggressive_dashboard_data import _render_json, _expected_output_paths
from quant_investor.strategy_records.store import StrategyRecordStoreError


@pytest.mark.parametrize("single_commit", [False, True])
def test_first_historical_v1_builds_after_five_store_days(tmp_path, single_commit):
    proof = backlog(tmp_path) if single_commit else run_sessions(tmp_path)
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    rendered = []
    for row in proof["sessions"]:
        plan_ref = {"path": row["plan"]["plan_path"], "sha256": row["plan"]["plan_sha256"]}
        bundle = build_bundle(
            project_root=tmp_path,
            record_root=tmp_path / "results/strategy_records/CN/aggressive_tech_manufacturing",
            benchmark_path=tmp_path / "portfolio_dashboard/inputs/cn_index_benchmark.csv",
            generated_at=now.isoformat(timespec="seconds"),
            today=now.date(),
            historical_close_plan_ref=plan_ref,
            historical_valuation_date=row["day"],
        )
        assert bundle["latest_data_date"] == row["day"]
        assert bundle["portfolio"]["performance_end_date"] == row["day"]
        assert validate_bundle_shape(bundle) == []
        assert verify_source_refs(bundle, tmp_path) == []
        paths = [ref["path"] for ref in bundle["source_refs"]]
        assert not any(p.endswith("_record_store/current.v1.json") for p in paths)
        assert any(p.endswith("committed-pointer.v1.json") for p in paths)
        assert plan_ref in bundle["source_refs"]
        compact = row["day"].replace("-", "")
        manifest = tmp_path / f"data/parquet/cn/_snapshots/synthetic-{compact}.json"
        reader = MarketDataReader(
            data_root=tmp_path / "data",
            frozen_snapshot_ref={
                "path": str(manifest.relative_to(tmp_path / "data")),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        )
        v2 = build_v2_bundle(
            project_root=tmp_path,
            v1_bundle=bundle,
            v1_json_path=_expected_output_paths(tmp_path)[0],
            record_root=tmp_path / "results/strategy_records/CN/aggressive_tech_manufacturing",
            generation_local_date=now.date(),
            generated_at=now.isoformat(timespec="seconds"),
            publication_attempt_id="dashboard-v2-historical-" + compact,
            market_reader=reader,
            v1_json_bytes_override=_render_json(bundle),
            historical_close_plan_ref=plan_ref,
            historical_valuation_date=row["day"],
        )
        assert v2["schema_version"] == "cn_aggressive_dashboard_history.v1"
        assert v2["evidence_timing"] == "RETROSPECTIVE_RECOMPUTE"
        assert v2["freshness"]["mark_as_of"] == row["day"]
        assert validate_v2_shape(v2) == []
        assert verify_v2_source_refs(v2, tmp_path, v1_bytes_override=_render_json(bundle)) == []
        rendered.append((compact, bundle, v2))

    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    with pytest.raises(StrategyRecordStoreError, match="SHA"):
        build_bundle(
            project_root=tmp_path,
            record_root=tmp_path / "results/strategy_records/CN/aggressive_tech_manufacturing",
            benchmark_path=tmp_path / "portfolio_dashboard/inputs/cn_index_benchmark.csv",
            generated_at=now.isoformat(timespec="seconds"),
            today=now.date(),
            historical_close_plan_ref={**plan_ref, "sha256": "0" * 64},
        )

    from quant_investor.operations.daily_contract import ContractError

    with pytest.raises(ContractError, match="DATE_NOT_IN_COMMIT"):
        build_bundle(
            project_root=tmp_path,
            record_root=tmp_path / "results/strategy_records/CN/aggressive_tech_manufacturing",
            benchmark_path=tmp_path / "portfolio_dashboard/inputs/cn_index_benchmark.csv",
            generated_at=now.isoformat(timespec="seconds"),
            today=now.date(),
            historical_close_plan_ref=plan_ref,
            historical_valuation_date="2026-08-21",
        )

    from scripts.daily_dashboard_capture import DailyDashboardCapture
    from quant_investor.operations.daily_journal import DailyJournal
    from quant_investor.operations.daily_contract import GRAPH_SHA256
    from quant_investor.contracts import canonical_json_bytes
    from export_cn_aggressive_dashboard_data import publish_bundle_pair
    from cn_dashboard_common import DashboardInputError

    for compact, bundle, v2 in rendered:
        journal = DailyJournal(str(tmp_path), compact)
        capture = DailyDashboardCapture(str(tmp_path), journal)
        raw = canonical_json_bytes(
            {
                "schema_version": "cn-daily-node-request.v1",
                "trade_date": compact,
                "node_id": "dashboard",
                "graph_sha256": GRAPH_SHA256,
                "release_ref": {
                    "path": "fixtures/release.json",
                    "sha256": hashlib.sha256(
                        (tmp_path / "fixtures/release.json").read_bytes()
                    ).hexdigest(),
                },
                "adapter_sha256": "a" * 64,
                "policy_refs": {},
                "input_refs": {},
            }
        )
        path = str(journal.root / "inputs/dashboard.json")
        ref = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
        with journal.locked():
            journal.storage.write(path, raw)
            value = capture.capture(v1=bundle, v2=v2, request_ref=ref)
        assert capture.read(request_ref=ref) == value
        paths = _expected_output_paths(tmp_path)
        with pytest.raises(DashboardInputError, match="historical_dashboard_cannot"):
            publish_bundle_pair(
                v1_bundle=bundle,
                v2_bundle=v2,
                v1_json_path=paths[0],
                v1_js_path=paths[1],
                v2_json_path=paths[2],
                v2_js_path=paths[3],
                project_root=tmp_path,
            )
        assert not any(path.exists() for path in paths)


def backlog(root):
    from _native_daily_store_fixture import NativeStoreFixture, DAYS, write
    from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter

    fixture = NativeStoreFixture(root)
    for day in DAYS:
        args = fixture.advance(day)
    plan = prepare_store_plan(args)
    release = {
        "path": "fixtures/release.json",
        "sha256": write(root / "fixtures/release.json", {"synthetic": True}),
    }
    adapter = StoreCloseAdapter(
        arguments=args,
        trade_date=DAYS[-1].replace("-", ""),
        plan_ref={"path": plan["plan_path"], "sha256": plan["plan_sha256"]},
        release_ref=release,
    )
    adapter.execute(adapter.template())
    assert adapter.probe(adapter.template()).outcome.state.value == "SUCCEEDED"
    return {"sessions": [{"day": day, "plan": plan} for day in DAYS]}
