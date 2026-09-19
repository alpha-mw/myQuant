"""Native Store-to-Dashboard segment. Synthetic data, real builders/checkers."""

from copy import deepcopy
from argparse import Namespace
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest

from _native_daily_store_fixture import NativeStoreFixture, write
from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter
import export_cn_aggressive_dashboard_data as exporter
from cn_dashboard_common import build_bundle, DashboardInputError
from cn_dashboard_v2 import build_v2_bundle, verify_v2_source_refs
from cn_dashboard_v2_selector import build_selector, publish_selector


def fixture(tmp_path):
    f = NativeStoreFixture(tmp_path)
    args = f.advance("2026-08-24")
    prepared = prepare_store_plan(args)
    adapter = StoreCloseAdapter(
        arguments=args,
        trade_date="20260824",
        plan_ref={"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]},
        release_ref={
            "path": "fixtures/release.json",
            "sha256": write(tmp_path / "fixtures/release.json", {"synthetic": True}),
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
        publication_attempt_id="dashboard-v2-synthetic-native",
        v1_json_bytes_override=exporter._render_json(v1),
    )
    return f, paths, v1, v2, stamp


def test_native_store_to_dashboard_pair_and_exact_close_gate(tmp_path):
    f, paths, v1, v2, stamp = fixture(tmp_path)
    exporter.require_expected_close_date(tmp_path, v1, v2, "20260824")
    exporter.publish_bundle_pair(
        v1_bundle=v1,
        v2_bundle=v2,
        v1_json_path=paths[0],
        v1_js_path=paths[1],
        v2_json_path=paths[2],
        v2_js_path=paths[3],
        project_root=tmp_path,
    )
    publish_selector(
        build_selector(
            attempt_id=v2["publication_attempt_id"],
            status="UPDATED",
            updated_at=stamp,
            reason="refresh_completed",
            v2_content_sha256=v2["content_sha256"],
        ),
        json_path=paths[4],
        js_path=paths[5],
        project_root=tmp_path,
        js_first=False,
    )
    assert verify_v2_source_refs(v2, tmp_path) == []
    assert v1["latest_data_date"] == v2["freshness"]["mark_as_of"] == "2026-08-24"
    assert v2["completeness"]["current_holdings"] == "COMPLETE"
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(DashboardInputError, match="DASHBOARD_STALE"):
        exporter.require_expected_close_date(tmp_path, v1, v2, "20260825")
    assert {p: p.read_bytes() for p in paths} == before


@pytest.mark.parametrize("domain", ["Store", "performance", "mark"])
def test_inconsistent_rendered_close_cannot_pass_expected_date(tmp_path, domain):
    f, paths, v1, v2, stamp = fixture(tmp_path)
    v1, v2 = deepcopy(v1), deepcopy(v2)
    if domain == "Store":
        v1["latest_data_date"] = "2026-08-21"
    elif domain == "performance":
        v1["portfolio"]["performance_end_date"] = "2026-08-21"
    else:
        v2["freshness"]["mark_as_of"] = "2026-08-21"
    with pytest.raises(DashboardInputError, match="DASHBOARD_STALE:" + domain):
        exporter.require_expected_close_date(tmp_path, v1, v2, "20260824")


def test_native_export_entrypoint_rejects_wrong_day_before_replacing_last_good(
    tmp_path, monkeypatch, capsys
):
    f, paths, v1, v2, stamp = fixture(tmp_path)
    exporter.publish_bundle_pair(
        v1_bundle=v1,
        v2_bundle=v2,
        v1_json_path=paths[0],
        v1_js_path=paths[1],
        v2_json_path=paths[2],
        v2_js_path=paths[3],
        project_root=tmp_path,
    )
    publish_selector(
        build_selector(
            attempt_id=v2["publication_attempt_id"],
            status="UPDATED",
            updated_at=stamp,
            reason="refresh_completed",
            v2_content_sha256=v2["content_sha256"],
        ),
        json_path=paths[4],
        js_path=paths[5],
        project_root=tmp_path,
        js_first=False,
    )
    before = {p: p.read_bytes() for p in paths}
    args = Namespace(
        project_root=tmp_path,
        record_root=f.root,
        benchmark=tmp_path / "portfolio_dashboard/inputs/cn_index_benchmark.csv",
        risk_free=tmp_path / "portfolio_dashboard/inputs/cn_govt_bond_yield.csv",
        generated_at=stamp,
        today=stamp[:10],
        attempt_id="dashboard-v2-synthetic-failed-date",
        history_integrity=exporter.DEFAULT_HISTORY_INTEGRITY,
        expected_trade_date="20260825",
        json_output=None,
        js_output=None,
        v2_json_output=None,
        v2_js_output=None,
        selector_json_output=None,
        selector_js_output=None,
    )
    monkeypatch.setattr(exporter, "parse_args", lambda: args)
    assert exporter.main() == 2
    assert "DASHBOARD_STALE" in capsys.readouterr().out
    assert {p: p.read_bytes() for p in paths} == before
