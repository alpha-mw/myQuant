"""入口清理回归测试。"""

from __future__ import annotations

import importlib

import quant_investor
import quant_investor.pipeline as pipeline
import web.main as web_main
import web.workspace_app as workspace_app


def _mainline_exports(module) -> set[str]:
    return {
        name
        for name in getattr(module, "__all__", [])
        if name.startswith("QuantInvestor")
    }


def test_public_entrypoints_only_expose_single_mainline():
    assert quant_investor.QuantInvestor is pipeline.QuantInvestor
    assert _mainline_exports(quant_investor) == {
        "QuantInvestor",
        "QuantInvestorPipelineResult",
    }
    assert set(pipeline.__all__) == {
        "QuantInvestor",
        "QuantInvestorPipelineResult",
    }

    package_unexpected = [
        name
        for name in dir(quant_investor)
        if name.startswith("QuantInvestor")
        and name not in {"QuantInvestor", "QuantInvestorPipelineResult"}
    ]
    pipeline_unexpected = [
        name
        for name in dir(pipeline)
        if name.startswith("QuantInvestor")
        and name not in {"QuantInvestor", "QuantInvestorPipelineResult"}
    ]

    assert package_unexpected == []
    assert pipeline_unexpected == []


def test_cli_research_run_has_no_architecture_switch():
    cli_main = importlib.import_module("quant_investor.cli.main")
    parser = cli_main._build_parser()
    args = parser.parse_args(["research", "run", "--stocks", "000001.SZ"])
    research_action = next(
        action
        for action in parser._actions
        if getattr(action, "dest", "") == "command"
    )
    research_parser = research_action.choices["research"]
    research_sub_action = next(
        action
        for action in research_parser._actions
        if getattr(action, "dest", "") == "research_command"
    )
    run_parser = research_sub_action.choices["run"]
    route_flag = "--" + "architecture"

    assert not hasattr(args, "architecture")
    assert all(action.dest != "architecture" for action in run_parser._actions)
    assert all(
        route_flag not in action.option_strings
        for action in run_parser._actions
    )


def test_cli_web_help_describes_workspace_service():
    cli_main = importlib.import_module("quant_investor.cli.main")
    parser = cli_main._build_parser()

    help_text = parser.format_help()

    assert "启动研究工作台 Web 服务" in help_text


def test_workspace_entrypoint_is_the_workspace_app():
    assert web_main.app is workspace_app.app
