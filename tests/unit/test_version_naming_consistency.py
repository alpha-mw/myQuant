from __future__ import annotations

from pathlib import Path
import tomllib

from quant_investor.cli import main as cli_main
import quant_investor.versioning as versioning


ROOT = Path(__file__).resolve().parents[2]


def test_single_mainline_package_and_runtime_versions_are_aligned():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock_packages = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    locked_project = next(item for item in lock_packages if item["name"] == project["name"])

    assert project["version"] == "15.0.0"
    assert locked_project["version"] == project["version"]
    assert "three-branch single mainline" in project["description"]

    assert versioning.ARCHITECTURE_VERSION == "15.0.0-stable"
    assert versioning.BRANCH_SCHEMA_VERSION == "branch-schema.v15.three-branch"
    assert versioning.LIKELIHOOD_SCHEMA_VERSION == "likelihood-schema.v15.two-likelihood"
    assert versioning.output_version_payload()["architecture_version"] == versioning.ARCHITECTURE_VERSION
    assert versioning.output_version_payload()["branch_schema_version"] == versioning.BRANCH_SCHEMA_VERSION

    architecture_constants = [name for name in vars(versioning) if name.startswith("ARCHITECTURE_VERSION_")]
    branch_constants = [name for name in vars(versioning) if name.startswith("BRANCH_SCHEMA_VERSION_")]

    assert architecture_constants == []
    assert branch_constants == []


def test_readme_and_cli_share_single_mainline_policy():
    parser = cli_main._build_parser()
    parsed = parser.parse_args(["research", "run", "--stocks", "000001.SZ"])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    parser = cli_main._build_parser()
    research_parser = parser._subparsers._group_actions[0].choices["research"]
    run_parser = research_parser._subparsers._group_actions[0].choices["run"]
    option_strings = [option for action in run_parser._actions for option in action.option_strings]
    route_flag = "--" + "architecture"

    assert "15.0.0" in readme
    assert route_flag not in readme
    assert "NarratorAgent -> ReportBundle" in readme
    assert "`buy` / `hold` / `sell` / `watch` / `avoid`" in readme
    assert "`reject` / `light_buy` / `strong_buy`" in readme

    assert "单一主线" in parser.description
    assert route_flag not in option_strings
    assert not hasattr(parsed, "architecture")


def test_versioning_module_exposes_only_single_mainline_payload():
    assert versioning.CURRENT_BRANCH_ORDER == ("quant", "fundamental", "macro")
    assert versioning.BRANCH_ORDER == versioning.CURRENT_BRANCH_ORDER
    assert versioning.output_version_payload() == {
        "architecture_version": versioning.ARCHITECTURE_VERSION,
        "branch_schema_version": versioning.BRANCH_SCHEMA_VERSION,
        "likelihood_schema_version": versioning.LIKELIHOOD_SCHEMA_VERSION,
        "calibration_schema_version": versioning.CALIBRATION_SCHEMA_VERSION,
        "ic_protocol_version": versioning.IC_PROTOCOL_VERSION,
        "report_protocol_version": versioning.REPORT_PROTOCOL_VERSION,
    }
