from __future__ import annotations

from pathlib import Path
import tomllib

from quant_investor.cli import main as cli_main
import quant_investor.versioning as versioning


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT_DOC = ROOT / "docs" / "architecture" / "entrypoints_and_versioning.md"


def test_single_mainline_package_and_runtime_versions_are_aligned():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    docs = ENTRYPOINT_DOC.read_text(encoding="utf-8")

    assert project["version"] == "12.0.0"
    assert "single mainline" in project["description"]

    assert versioning.ARCHITECTURE_VERSION == "12.0.0-stable"
    assert versioning.BRANCH_SCHEMA_VERSION == "branch-schema.v12.unified-mainline"
    assert versioning.IC_PROTOCOL_VERSION == "ic-protocol.v12.mainline"
    assert versioning.REPORT_PROTOCOL_VERSION == "report-protocol.v12.mainline"
    assert versioning.CALIBRATION_SCHEMA_VERSION == "2026-03-22.calibration.v2"
    assert versioning.AGENT_SCHEMA_VERSION == "2026-03-23.agent.v1"
    assert versioning.output_version_payload()["architecture_version"] == versioning.ARCHITECTURE_VERSION
    assert versioning.output_version_payload()["branch_schema_version"] == versioning.BRANCH_SCHEMA_VERSION
    assert versioning.output_version_payload()["ic_protocol_version"] == versioning.IC_PROTOCOL_VERSION
    assert versioning.output_version_payload()["report_protocol_version"] == versioning.REPORT_PROTOCOL_VERSION

    architecture_constants = [name for name in vars(versioning) if name.startswith("ARCHITECTURE_VERSION_")]
    branch_constants = [name for name in vars(versioning) if name.startswith("BRANCH_SCHEMA_VERSION_")]

    assert architecture_constants == []
    assert branch_constants == []
    assert "web.main:app" in docs
    assert 'IC_PROTOCOL_VERSION = "ic-protocol.v12.mainline"' in docs
    assert 'REPORT_PROTOCOL_VERSION = "report-protocol.v12.mainline"' in docs
    assert 'CALIBRATION_SCHEMA_VERSION = "2026-03-22.calibration.v2"' in docs
    assert 'AGENT_SCHEMA_VERSION = "2026-03-23.agent.v1"' in docs


def test_readme_and_cli_share_single_mainline_policy():
    parser = cli_main._build_parser()
    parsed = parser.parse_args(["research", "run", "--stocks", "000001.SZ"])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    parser = cli_main._build_parser()
    research_parser = parser._subparsers._group_actions[0].choices["research"]
    run_parser = research_parser._subparsers._group_actions[0].choices["run"]
    option_strings = [option for action in run_parser._actions for option in action.option_strings]
    route_flag = "--" + "architecture"

    assert "12.0.0" in readme
    assert route_flag not in readme
    assert "NarratorAgent -> ReportBundle" in readme
    assert "`buy` / `hold` / `sell` / `watch` / `avoid`" in readme
    assert "`reject` / `light_buy` / `strong_buy`" in readme
    assert "其余契约类型作为稳定数据模型导出供研究与测试复用" in readme

    assert "单一主线" in parser.description
    assert route_flag not in option_strings
    assert not hasattr(parsed, "architecture")


def test_versioning_module_exposes_only_single_mainline_payload():
    assert versioning.CURRENT_BRANCH_ORDER == ("kline", "quant", "fundamental", "intelligence", "macro")
    assert versioning.BRANCH_ORDER == versioning.CURRENT_BRANCH_ORDER
    assert versioning.output_version_payload() == {
        "architecture_version": versioning.ARCHITECTURE_VERSION,
        "branch_schema_version": versioning.BRANCH_SCHEMA_VERSION,
        "calibration_schema_version": versioning.CALIBRATION_SCHEMA_VERSION,
        "ic_protocol_version": versioning.IC_PROTOCOL_VERSION,
        "report_protocol_version": versioning.REPORT_PROTOCOL_VERSION,
    }
