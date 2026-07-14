"""
公共入口与构建面烟测。
"""

from __future__ import annotations

from pathlib import Path

import quant_investor
import quant_investor.cli.main as cli_main


def test_public_package_exports():
    assert hasattr(quant_investor, "QuantInvestor")
    assert hasattr(quant_investor, "QuantInvestorPipelineResult")
    assert hasattr(quant_investor, "BranchResult")
    assert {
        name
        for name in dir(quant_investor)
        if name.startswith("QuantInvestor")
    } == {"QuantInvestor", "QuantInvestorPipelineResult"}


def test_cli_market_download_dispatches(monkeypatch):
    captured = {}

    def _fake_run_download(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_download", _fake_run_download)
    cli_main.main(["market", "download", "--market", "CN", "--category", "hs300"])

    assert captured["market"] == "CN"
    assert captured["categories"] == ["hs300"]


def test_cli_market_fundamental_maintain_dispatches(monkeypatch):
    captured = {}

    def _fake_run_fundamental_maintenance(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_fundamental_maintenance", _fake_run_fundamental_maintenance)
    cli_main.main(
        [
            "market",
            "fundamental-maintain",
            "--market",
            "CN",
            "--universes",
            "hs300,zz500,zz1000",
            "--years",
            "5",
            "--as-of",
            "20240510",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["universes"] == "hs300,zz500,zz1000"
    assert captured["years"] == 5
    assert captured["as_of"] == "20240510"
    assert captured["allow_live"] is False


def test_cli_market_data_governance_dispatches_local_read_only(monkeypatch):
    captured = {}

    def _fake_run_data_governance(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_data_governance", _fake_run_data_governance)
    cli_main.main(
        [
            "market",
            "data-governance",
            "--market",
            "CN",
            "--category",
            "full_a",
            "--as-of",
            "20240510",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["categories"] == ["full_a"]
    assert captured["as_of"] == "20240510"
    assert captured["allow_live"] is False
    assert captured["allow_public_fallback"] is False


def test_cli_market_macro_maintain_without_input_is_fail_closed(monkeypatch, capsys):
    captured = {}

    def _fake_run_macro_maintenance(**kwargs):
        captured.update(kwargs)
        return {"status": "blocked", "promoted": False}

    monkeypatch.setattr(cli_main, "run_macro_maintenance", _fake_run_macro_maintenance)
    cli_main.main(["market", "macro-maintain", "--market", "CN", "--as-of", "20240510"])

    assert captured["indicators"] is None
    assert captured["allow_live"] is False
    assert '"promoted": false' in capsys.readouterr().out


def test_cli_market_macro_analyze_is_explicit_local_observer(monkeypatch, tmp_path, capsys):
    observations = tmp_path / "observations.json"
    observations.write_text("[]", encoding="utf-8")
    captured = {}

    def _fake_run_macro_analysis(**kwargs):
        captured.update(kwargs)
        return {"active": True, "applied": False}

    monkeypatch.setattr(cli_main, "run_macro_analysis", _fake_run_macro_analysis)
    cli_main.main(
        [
            "market",
            "macro-analyze",
            "--market",
            "CN",
            "--as-of",
            "20240510",
            "--observations",
            str(observations),
        ]
    )

    assert captured["observations_path"] == str(observations)
    assert captured["market"] == "CN"
    assert '"applied": false' in capsys.readouterr().out


def test_cli_market_macro_observation_maintenance_is_explicit(monkeypatch, tmp_path, capsys):
    observations = tmp_path / "observations.json"
    observations.write_text('{"observations": []}', encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        cli_main,
        "run_macro_observation_maintenance",
        lambda **kwargs: captured.update(kwargs) or {"promoted": False},
    )
    cli_main.main(
        [
            "market",
            "macro-maintain",
            "--market",
            "CN",
            "--as-of",
            "20240510",
            "--input-observations",
            str(observations),
            "--run-id",
            "fixture",
        ]
    )

    assert captured["allow_live"] is False
    assert captured["allow_tushare_fallback"] is False
    assert captured["run_id"] == "fixture"
    assert '"promoted": false' in capsys.readouterr().out


def test_cli_market_macro_replay_dispatch(monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_replay",
        lambda **kwargs: captured.update(kwargs) or {"applied": False},
    )
    cli_main.main(
        [
            "market",
            "macro-replay",
            "--market",
            "CN",
            "--start-date",
            "2024-05-09",
            "--end-date",
            "2024-05-10",
            "--calendar",
            str(tmp_path / "calendar.parquet"),
        ]
    )

    assert captured["market"] == "CN"
    assert captured["calendar_path"].endswith("calendar.parquet")
    assert '"applied": false' in capsys.readouterr().out


def test_cli_market_macro_normalize_tushare_dispatch(monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_tushare_normalization",
        lambda **kwargs: captured.update(kwargs) or {"promoted": False},
    )
    cli_main.main(
        [
            "market",
            "macro-normalize-tushare",
            "--market",
            "CN",
            "--input-json",
            str(tmp_path / "raw.json"),
            "--plan-json",
            str(tmp_path / "plan.json"),
            "--evidence-json",
            str(tmp_path / "evidence.json"),
            "--run-id",
            "fixture",
        ]
    )
    assert captured["path"].endswith("raw.json")
    assert captured["plan_path"].endswith("plan.json")
    assert captured["evidence_path"].endswith("evidence.json")
    assert captured["run_id"] == "fixture"
    assert '"promoted": false' in capsys.readouterr().out


def test_cli_market_macro_backfill_publish_dispatch(monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_backfill_publish",
        lambda **kwargs: captured.update(kwargs) or {"promoted": True},
    )
    cli_main.main(
        [
            "market",
            "macro-backfill-publish",
            "--market",
            "CN",
            "--manifest",
            str(tmp_path / "normalization_manifest.json"),
            "--run-id",
            "g1",
            "--expected-pointer-sha256",
            "EMPTY",
            "--expected-manifest-sha256",
            "1" * 64,
            "--expected-plan-sha256",
            "2" * 64,
        ]
    )
    assert captured["expected_pointer_sha256"] == ""
    assert captured["expected_manifest_sha256"] == "1" * 64
    assert captured["expected_plan_sha256"] == "2" * 64
    assert captured["run_id"] == "g1"
    assert '"promoted": true' in capsys.readouterr().out


def test_cli_market_macro_forward_observation_dispatch(monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_forward_observation",
        lambda **kwargs: captured.update(kwargs) or {"promoted": True},
    )
    cli_main.main(
        [
            "market",
            "macro-observe-forward",
            "--market",
            "CN",
            "--calendar",
            str(tmp_path / "calendar.parquet"),
            "--state-root",
            str(tmp_path / "forward"),
            "--expected-pointer-sha256",
            "EMPTY",
        ]
    )
    assert captured["expected_pointer_sha256"] == ""
    assert captured["calendar_path"].endswith("calendar.parquet")
    assert captured["root"].endswith("forward")
    assert '"promoted": true' in capsys.readouterr().out


def test_cli_market_macro_coverage_audit_dispatch(monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_coverage",
        lambda **kwargs: captured.update(kwargs) or {"status": "blocked"},
    )
    cli_main.main(
        [
            "market",
            "macro-coverage-audit",
            "--market",
            "CN",
            "--as-of",
            "2024-05-10",
            "--observations",
            str(tmp_path / "observations"),
            "--raw-root",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "coverage"),
        ]
    )
    assert captured["as_of"] == "2024-05-10"
    assert captured["observations_path"].endswith("observations")
    assert captured["raw_root"].endswith("raw")
    assert captured["output_root"].endswith("coverage")
    assert '"status": "blocked"' in capsys.readouterr().out


def test_cli_market_macro_acquisition_plan_dispatch(monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_acquisition",
        lambda **kwargs: captured.update(kwargs) or {"status": "blocked"},
    )
    cli_main.main(
        [
            "market",
            "macro-acquisition-plan",
            "--market",
            "CN",
            "--coverage-audit",
            str(tmp_path / "coverage_audit.json"),
            "--output-dir",
            str(tmp_path / "plans"),
        ]
    )
    assert captured["market"] == "CN"
    assert captured["coverage_audit"].endswith("coverage_audit.json")
    assert captured["output_root"].endswith("plans")
    assert '"status": "blocked"' in capsys.readouterr().out


def test_datahub_public_sources_are_not_sourceless():
    required = [
        "quant_investor/data/hub.py",
        "quant_investor/data/models.py",
        "quant_investor/data/_registry.py",
        "quant_investor/data/_tushare_client.py",
        "quant_investor/data/sources/base.py",
        "quant_investor/data/sources/tushare_cn.py",
        "quant_investor/data/processing/cleaner.py",
        "quant_investor/data/universe/cn_universe.py",
    ]
    root = Path(__file__).resolve().parents[2]
    assert all((root / path).exists() for path in required)


def test_cli_market_analyze_dispatches(monkeypatch):
    captured = {}

    def _fake_run_market_analysis(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_market_analysis", _fake_run_market_analysis)
    cli_main.main(["market", "analyze", "--market", "US", "--mode", "sample"])

    assert captured["market"] == "US"
    assert captured["mode"] == "sample"


def test_cli_market_backtest_dispatches(monkeypatch):
    captured = {}

    def _fake_run_market_backtest(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_market_backtest", _fake_run_market_backtest)
    cli_main.main(["market", "backtest", "--market", "CN", "--category", "hs300"])

    assert captured["market"] == "CN"
    assert captured["categories"] == ["hs300"]


def test_pyproject_only_packages_quant_investor():
    pyproject_text = Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'packages = ["quant_investor"]' in pyproject_text
    assert 'quant-investor = "quant_investor.cli.main:main"' in pyproject_text


def test_cli_research_dispatches_single_mainline(monkeypatch):
    captured = {}

    class _FakeInvestor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

        def print_report(self):
            return None

    monkeypatch.setattr(cli_main, "QuantInvestor", _FakeInvestor)
    cli_main.main(["research", "run", "--stocks", "000001.SZ"])

    assert captured["stock_pool"] == ["000001.SZ"]


def test_single_mainline_one_symbol_mock_run_includes_version_fields(monkeypatch):
    from types import SimpleNamespace

    from quant_investor.agent_protocol import (
        ActionLabel,
        AgentStatus,
        BranchVerdict,
        ExecutionTrace,
        GlobalContext,
        PortfolioDecision,
        ShortlistItem,
        WhatIfPlan,
    )
    from quant_investor.branch_contracts import BranchResult
    import quant_investor.pipeline.mainline as mainline_module

    shortlist = [
        ShortlistItem(
            symbol="000001.SZ",
            company_name="平安银行",
            category="full_a",
            rank_score=0.91,
            action=ActionLabel.BUY,
            confidence=0.78,
            expected_upside=0.12,
            suggested_weight=0.25,
            rationale=["posterior top rank"],
        )
    ]
    portfolio_decision = PortfolioDecision(
        status=AgentStatus.SUCCESS,
        shortlist=shortlist,
        target_exposure=0.45,
        target_gross_exposure=0.45,
        target_net_exposure=0.45,
        cash_ratio=0.55,
        target_weights={"000001.SZ": 0.25},
        target_positions={"000001.SZ": 250000.0},
        metadata={"selected_count": 1},
    )
    report_bundle = SimpleNamespace(
        markdown_report="# DAG Report",
        headline="DAG headline",
        summary="DAG summary",
        executive_summary=["summary line"],
        market_view=["market line"],
        branch_verdicts={"macro": BranchVerdict(agent_name="macro", thesis="ok", final_score=0.2, final_confidence=0.7)},
        macro_verdict=BranchVerdict(agent_name="macro", thesis="ok", final_score=0.2, final_confidence=0.7),
        risk_decision=None,
        ic_decision=SimpleNamespace(action=ActionLabel.BUY, thesis="ic ok"),
        ic_decisions=[],
        model_role_metadata=SimpleNamespace(to_dict=lambda: {"branch_model": "deepseek-reasoner", "master_model": "moonshot-v1-128k"}),
        execution_trace=ExecutionTrace(),
        what_if_plan=WhatIfPlan(),
        portfolio_plan=SimpleNamespace(
            target_weights={"000001.SZ": 0.25},
            target_positions={"000001.SZ": 250000.0},
            position_limits={"000001.SZ": 0.25},
            blocked_symbols=[],
            rejected_symbols=[],
            target_exposure=0.45,
            target_gross_exposure=0.45,
            target_net_exposure=0.45,
            cash_ratio=0.55,
            execution_notes=["note"],
            construction_notes=[],
            status=AgentStatus.SUCCESS,
        ),
        coverage_summary=[],
        warnings=[],
        appendix_diagnostics=[],
        ic_hints_by_symbol={"000001.SZ": {"action": "buy", "score": 0.8}},
    )
    dag_artifacts = {
        "global_context": GlobalContext(
            market="CN",
            universe_key="full_a",
            universe_symbols=["000001.SZ"],
            universe_tiers={"total": ["000001.SZ"], "researchable": ["000001.SZ"], "shortlistable": ["000001.SZ"], "final_selected": ["000001.SZ"]},
        ),
        "symbol_research_packets": {},
        "branch_verdicts_by_symbol": {"000001.SZ": {"macro": BranchVerdict(agent_name="macro", thesis="ok", symbol="000001.SZ", final_score=0.2, final_confidence=0.7)}},
        "branch_summaries": {"macro": BranchVerdict(agent_name="macro", thesis="ok", final_score=0.2, final_confidence=0.7)},
        "macro_verdict": BranchVerdict(agent_name="macro", thesis="ok", final_score=0.2, final_confidence=0.7),
        "risk_decision": None,
        "ic_decisions": [],
        "shortlist": shortlist,
        "portfolio_plan": report_bundle.portfolio_plan,
        "portfolio_decision": portfolio_decision,
        "review_bundle": SimpleNamespace(branch_summaries={}, ic_hints_by_symbol={"000001.SZ": {"action": "buy"}}, fallback_reasons=[]),
        "model_role_metadata": report_bundle.model_role_metadata,
        "what_if_plan": report_bundle.what_if_plan,
        "execution_trace": report_bundle.execution_trace,
        "tradability_snapshot": {"000001.SZ": {"tradable": True}},
        "data_quality_issues": [],
        "data_quality_summary": {"researchable_count": 1},
        "resolver": {"resolution_strategy": "logical_full_a"},
        "report_bundle": report_bundle,
        "portfolio_master_output": SimpleNamespace(final_score=0.7, confidence=0.8),
        "portfolio_master_meta": {"confidence": 0.8},
        "branch_results": {"macro": BranchResult(branch_name="macro", final_score=0.2, final_confidence=0.7, symbol_scores={"000001.SZ": 0.2})},
        "bayesian_records": [],
        "funnel_output": SimpleNamespace(candidates=["000001.SZ"], excluded_symbols={}, funnel_metadata={}),
    }
    monkeypatch.setattr(mainline_module, "_execute_market_dag", lambda **kwargs: dag_artifacts)
    monkeypatch.setattr(
        mainline_module,
        "build_market_data_snapshot",
        lambda **_kwargs: {
            "market": "CN",
            "universe_key": "full_a",
            "summary_text": "isolated public-package fixture",
            "missing_requested_symbols": [],
            "unreadable_requested_symbols": [],
        },
    )

    result = quant_investor.QuantInvestor(
        stock_pool=["000001.SZ"],
        market="CN",
        verbose=False,
    ).run()

    assert result.architecture_version == "13.0.0-stable"
    assert result.branch_schema_version == "branch-schema.v13.four-branch"
    assert result.calibration_schema_version
    assert result.debate_template_version
    assert result.final_strategy.architecture_version == result.architecture_version
