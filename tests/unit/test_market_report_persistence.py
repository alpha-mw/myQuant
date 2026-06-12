"""Market report persistence helper tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from quant_investor.market.report_persistence import (
    persist_market_analysis_outputs,
)
from quant_investor.market.runtime_profile import MarketRuntimeProfiler


def test_runtime_profile_records_exclusive_time_for_nested_stages(monkeypatch):
    import quant_investor.market.runtime_profile as runtime_profile

    clock = iter([1.0, 2.0, 5.0, 9.0])
    monkeypatch.setattr(runtime_profile, "perf_counter", lambda: next(clock))
    profiler = MarketRuntimeProfiler(market="CN", universe="hs300")

    with profiler.stage("outer"):
        with profiler.stage("inner"):
            pass

    inner, outer = profiler.stages
    assert inner["name"] == "inner"
    assert inner["seconds"] == 3.0
    assert inner["exclusive_seconds"] == 3.0
    assert inner["wall_seconds"] == 3.0

    assert outer["name"] == "outer"
    assert outer["seconds"] == 5.0
    assert outer["exclusive_seconds"] == 5.0
    assert outer["wall_seconds"] == 8.0
    assert outer["child_wall_seconds"] == 3.0
    assert "| Stage | Exclusive Seconds | Wall Seconds | Metadata |" in profiler.to_markdown()


def test_persist_outputs_writes_profile_next_to_report(tmp_path):
    profiler = MarketRuntimeProfiler(
        market="CN",
        universe="hs300",
        categories=["hs300", "zz500"],
        metadata={"mode": "sample"},
    )
    report_dir = tmp_path / "reports"

    def _generate_full_report(*args, **kwargs):
        return {
            "summary_report": str(report_dir / "summary.md"),
            "trade_report": str(report_dir / "trade.md"),
            "trade_data": str(report_dir / "trade.json"),
        }

    result = persist_market_analysis_outputs(
        all_results={
            "hs300": [{"symbol": "000001.SZ"}],
            "zz500": [],
        },
        market="CN",
        total_capital=1_000_000,
        top_k=3,
        analysis_output_dir=tmp_path,
        category_count=2,
        runtime_profiler=profiler,
        report_bundle=SimpleNamespace(markdown_report=""),
        generate_full_report=_generate_full_report,
    )

    runtime_profile_json = result.report_paths["runtime_profile_json"]
    runtime_profile_md = result.report_paths["runtime_profile_md"]
    profile_payload = json.loads(Path(runtime_profile_json).read_text(
        encoding="utf-8"
    ))

    assert result.runtime_profile["stages"][-1]["name"] == (
        "analysis_report_persistence"
    )
    assert result.runtime_profile["stages"][-1]["metadata"] == {
        "category_count": 2,
        "result_count": 1,
        "report_path_count": 3,
    }
    assert profile_payload["stages"][-1]["name"] == (
        "analysis_report_persistence"
    )
    assert runtime_profile_json.startswith(str(report_dir))
    assert runtime_profile_md.startswith(str(report_dir))
    assert result.report_paths["report_bundle"].markdown_report == ""
    assert Path(runtime_profile_md).read_text(encoding="utf-8").startswith(
        "# Market Runtime Profile"
    )
