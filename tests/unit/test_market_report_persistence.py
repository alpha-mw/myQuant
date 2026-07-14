"""Market report persistence helper tests."""

from __future__ import annotations

import json
import hashlib
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.market.report_persistence import (
    persist_market_analysis_outputs,
    write_analysis_run_manifest,
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
    profile_payload = json.loads(Path(runtime_profile_json).read_text(encoding="utf-8"))

    assert result.runtime_profile["stages"][-1]["name"] == ("analysis_report_persistence")
    assert result.runtime_profile["stages"][-1]["metadata"] == {
        "category_count": 2,
        "result_count": 1,
        "report_path_count": 3,
    }
    assert profile_payload["stages"][-1]["name"] == ("analysis_report_persistence")
    assert runtime_profile_json.startswith(str(report_dir))
    assert runtime_profile_md.startswith(str(report_dir))
    assert result.report_paths["report_bundle"].markdown_report == ""
    assert (
        Path(runtime_profile_md).read_text(encoding="utf-8").startswith("# Market Runtime Profile")
    )


def test_write_analysis_run_manifest_is_private_and_hash_bound(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    trade_report = report_dir / "trade.md"
    trade_report.write_text("report", encoding="utf-8")
    monkeypatch.setattr(
        "quant_investor.market.report_persistence._current_git_sha",
        lambda: "abc123",
    )

    path = Path(
        write_analysis_run_manifest(
            market="CN",
            analysis_output_dir=tmp_path,
            report_paths={"trade_report": str(trade_report)},
            analysis_meta={
                "market": "CN",
                "shortlist": [{"symbol": "000001.SZ"}],
                "symbol_research_packets": {"000001.SZ": {"branch_scores": {"fundamental": 0.2}}},
            },
        )
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "analysis-run-manifest.v1"
    assert payload["git_sha"] == "abc123"
    assert payload["analysis_meta_sha256"]
    assert payload["manifest_sha256"]
    assert os.stat(path).st_mode & 0o777 == 0o600
    original_bytes = path.read_bytes()

    second_path = Path(
        write_analysis_run_manifest(
            market="CN",
            analysis_output_dir=tmp_path,
            report_paths={"trade_report": str(trade_report)},
            analysis_meta={"market": "CN", "symbols": ["000001.SZ"]},
        )
    )
    assert second_path != path
    assert second_path.name == path.name == "analysis_run_manifest.v1.json"
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert path.read_bytes() == original_bytes


def test_write_analysis_run_manifest_emits_real_counterfactual_companion(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    trade_report = report_dir / "trade.md"
    trade_report.write_text("report", encoding="utf-8")
    monkeypatch.setattr(
        "quant_investor.market.report_persistence._current_git_sha",
        lambda: "abc123",
    )
    risk_decision = {
        "status": "success",
        "action_cap": "buy",
        "gross_exposure_cap": 0.5,
        "target_exposure_cap": 0.5,
        "max_weight": 0.2,
        "position_limits": {"000001.SZ": 0.2},
        "blocked_symbols": [],
    }
    alternative_portfolio = {
        "target_weights": {"000001.SZ": 0.16},
        "target_exposure": 0.16,
        "shortlist": [
            {
                "symbol": "000001.SZ",
                "rank_score": 0.4,
                "action": "buy",
                "confidence": 0.6,
                "expected_upside": 0.1,
            }
        ],
        "risk_constraints": {"risk_decision": risk_decision},
        "metadata": {"replay_variant": "with_dossier"},
    }
    replay = {
        "schema_version": "fundamental-control-chain-replay.v1",
        "measurement_only": True,
        "variant": "with_dossier",
        "branch_summaries": {"fundamental": {"final_score": 0.40, "final_confidence": 0.6}},
        "branch_verdicts_by_symbol": {
            "000001.SZ": {
                "fundamental": {
                    "final_score": 0.40,
                    "final_confidence": 0.6,
                    "metadata": {
                        "fundamental_research_variant": "with_dossier",
                        "fundamental_research_runtime": {
                            "request_id": "req-1",
                            "dossier_id": "dossier-1",
                            "measurement_only": True,
                            "applied": True,
                            "counterfactual": False,
                        },
                    },
                }
            }
        },
        "bayesian_records": [
            {
                "symbol": "000001.SZ",
                "posterior_action_score": 0.4,
                "posterior_expected_alpha": 0.1,
                "posterior_confidence": 0.6,
                "posterior_edge_after_costs": 0.08,
                "metadata": {
                    "fundamental_research_variant": "with_dossier",
                    "fundamental_score": 0.4,
                },
            }
        ],
        "shortlist": alternative_portfolio["shortlist"],
        "ic_hints_by_symbol": {},
        "risk_decision": risk_decision,
        "ic_decisions": [
            {
                "symbol": "000001.SZ",
                "final_score": 0.4,
                "final_confidence": 0.6,
                "action": "buy",
                "metadata": {"llm_hint_applied": False},
            }
        ],
        "portfolio_plan": {
            "target_weights": {"000001.SZ": 0.16},
            "target_exposure": 0.16,
            "position_limits": {"000001.SZ": 0.2},
            "blocked_symbols": [],
            "rejected_symbols": [],
        },
        "portfolio_decision": alternative_portfolio,
    }
    meta = {
        "portfolio_decision": {
            "target_weights": {"000001.SZ": 0.10},
            "metadata": {
                "fundamental_research_counterfactual_replay": replay,
            },
        },
        "data_snapshot": {
            "snapshot_id": "snapshot-1",
            "local_latest_trade_date": "20260714",
        },
        "global_context": {"universe_key": "full_a"},
    }

    path = Path(
        write_analysis_run_manifest(
            market="CN",
            analysis_output_dir=tmp_path,
            report_paths={"trade_report": str(trade_report)},
            analysis_meta=meta,
        )
    )
    companion = path.parent / "analysis_run_manifest.with_dossier.v1.json"
    actual_payload = json.loads(path.read_text(encoding="utf-8"))
    counter_payload = json.loads(companion.read_text(encoding="utf-8"))

    assert actual_payload["analysis_meta"]["fundamental_research_variant"] == "without_dossier"
    assert counter_payload["analysis_meta"]["fundamental_research_variant"] == "with_dossier"
    assert counter_payload["analysis_meta"]["portfolio_decision"] == alternative_portfolio
    assert counter_payload["analysis_meta"]["fundamental_research_control_chain"] == replay
    assert (
        counter_payload["analysis_meta"]["fundamental_research_source_manifest_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert companion.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "mutation",
    [
        "measurement_mode",
        "master_advisory",
        "branch_summary",
        "bayesian_shortlist",
        "ic_branch",
        "risk_cap",
    ],
)
def test_counterfactual_companion_rejects_semantically_fabricated_control_chain(
    tmp_path, monkeypatch, mutation
) -> None:
    report_dir = tmp_path / mutation
    report_dir.mkdir()
    trade_report = report_dir / "trade.md"
    trade_report.write_text("report", encoding="utf-8")
    monkeypatch.setattr(
        "quant_investor.market.report_persistence._current_git_sha",
        lambda: "abc123",
    )
    risk = {
        "action_cap": "buy",
        "gross_exposure_cap": 0.2,
        "target_exposure_cap": 0.2,
        "max_weight": 0.2,
        "position_limits": {"000001.SZ": 0.2},
        "blocked_symbols": [],
    }
    shortlist = [
        {
            "symbol": "000001.SZ",
            "rank_score": 0.4,
            "action": "buy",
            "confidence": 0.6,
            "expected_upside": 0.1,
        }
    ]
    replay = {
        "schema_version": "fundamental-control-chain-replay.v1",
        "measurement_only": True,
        "variant": "without_dossier",
        "branch_summaries": {"fundamental": {"final_score": 0.4, "final_confidence": 0.6}},
        "branch_verdicts_by_symbol": {
            "000001.SZ": {
                "fundamental": {
                    "final_score": 0.4,
                    "final_confidence": 0.6,
                    "metadata": {"fundamental_research_variant": "without_dossier"},
                }
            }
        },
        "bayesian_records": [
            {
                "symbol": "000001.SZ",
                "posterior_action_score": 0.4,
                "posterior_expected_alpha": 0.1,
                "posterior_confidence": 0.6,
                "posterior_edge_after_costs": 0.08,
                "metadata": {
                    "fundamental_research_variant": "without_dossier",
                    "fundamental_score": 0.4,
                },
            }
        ],
        "shortlist": shortlist,
        "ic_hints_by_symbol": {},
        "risk_decision": risk,
        "ic_decisions": [
            {
                "symbol": "000001.SZ",
                "final_score": 0.4,
                "final_confidence": 0.6,
                "action": "buy",
                "metadata": {"llm_hint_applied": False},
            }
        ],
        "portfolio_plan": {
            "target_weights": {"000001.SZ": 0.16},
            "target_exposure": 0.16,
            "position_limits": {"000001.SZ": 0.2},
            "blocked_symbols": [],
            "rejected_symbols": [],
        },
        "portfolio_decision": {
            "target_weights": {"000001.SZ": 0.16},
            "target_exposure": 0.16,
            "shortlist": shortlist,
            "risk_constraints": {"risk_decision": risk},
            "metadata": {},
        },
    }
    replay = deepcopy(replay)
    if mutation == "measurement_mode":
        replay["measurement_only"] = False
    elif mutation == "master_advisory":
        replay["portfolio_decision"]["master_hints"] = {
            "portfolio_master_output": {"status": "success", "score": 0.9}
        }
    elif mutation == "branch_summary":
        replay["branch_summaries"]["fundamental"]["final_score"] = -0.9
    elif mutation == "bayesian_shortlist":
        replay["shortlist"][0]["rank_score"] = -0.9
        replay["portfolio_decision"]["shortlist"][0]["rank_score"] = -0.9
    elif mutation == "ic_branch":
        replay["ic_decisions"][0]["final_score"] = -0.9
    else:
        replay["portfolio_plan"]["target_weights"]["000001.SZ"] = 0.9
        replay["portfolio_plan"]["target_exposure"] = 0.9
        replay["portfolio_decision"]["target_weights"]["000001.SZ"] = 0.9
        replay["portfolio_decision"]["target_exposure"] = 0.9
    meta = {
        "portfolio_decision": {"metadata": {"fundamental_research_counterfactual_replay": replay}}
    }
    with pytest.raises(ValueError, match="control-chain replay"):
        write_analysis_run_manifest(
            market="CN",
            analysis_output_dir=tmp_path,
            report_paths={"trade_report": str(trade_report)},
            analysis_meta=meta,
        )


def test_counterfactual_companion_rejects_inconsistent_control_chain(tmp_path, monkeypatch) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    trade_report = report_dir / "trade.md"
    trade_report.write_text("report", encoding="utf-8")
    monkeypatch.setattr(
        "quant_investor.market.report_persistence._current_git_sha",
        lambda: "abc123",
    )
    replay = {
        "schema_version": "fundamental-control-chain-replay.v1",
        "variant": "without_dossier",
        "branch_summaries": {"fundamental": {"final_score": 0.1}},
        "branch_verdicts_by_symbol": {
            "000001.SZ": {
                "fundamental": {
                    "final_score": 0.1,
                    "metadata": {"fundamental_research_variant": "without_dossier"},
                }
            }
        },
        "bayesian_records": [{"symbol": "000001.SZ"}],
        "shortlist": [{"symbol": "000001.SZ"}],
        "ic_hints_by_symbol": {},
        "risk_decision": {"status": "SUCCESS", "blocked_symbols": []},
        "ic_decisions": [{"symbol": "000001.SZ", "metadata": {"llm_hint_applied": False}}],
        "portfolio_plan": {
            "target_weights": {"000001.SZ": 0.05},
            "target_exposure": 0.05,
            "blocked_symbols": [],
            "rejected_symbols": [],
        },
        "portfolio_decision": {
            "target_weights": {"000001.SZ": 0.09},
            "target_exposure": 0.09,
            "shortlist": [{"symbol": "000001.SZ"}],
            "risk_constraints": {"risk_decision": {"status": "SUCCESS", "blocked_symbols": []}},
        },
    }
    meta = {
        "portfolio_decision": {"metadata": {"fundamental_research_counterfactual_replay": replay}}
    }

    with pytest.raises(ValueError, match="control-chain replay"):
        write_analysis_run_manifest(
            market="CN",
            analysis_output_dir=tmp_path,
            report_paths={"trade_report": str(trade_report)},
            analysis_meta=meta,
        )


def test_non_cn_analysis_never_emits_fundamental_companion(tmp_path, monkeypatch) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    trade_report = report_dir / "trade.md"
    trade_report.write_text("report", encoding="utf-8")
    monkeypatch.setattr(
        "quant_investor.market.report_persistence._current_git_sha",
        lambda: "abc123",
    )
    meta = {
        "portfolio_decision": {
            "metadata": {
                "fundamental_research_counterfactual_replay": {
                    "schema_version": "fundamental-control-chain-replay.v1",
                    "variant": "with_dossier",
                }
            }
        }
    }

    path = Path(
        write_analysis_run_manifest(
            market="US",
            analysis_output_dir=tmp_path,
            report_paths={"trade_report": str(trade_report)},
            analysis_meta=meta,
        )
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "fundamental_research_variant" not in payload["analysis_meta"]
    assert list(path.parent.glob("analysis_run_manifest.*_dossier.v1.json")) == []
