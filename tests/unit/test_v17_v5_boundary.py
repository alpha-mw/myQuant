from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import tomllib

import quant_investor.v17_v5_runtime.cli as cli_subject
from quant_investor.v17_v5_contract.resources import PackageResourceError
from quant_investor.v17_v5_runtime.authority import DELIVERY_STATUS, authority_envelope
from quant_investor.v17_v5_runtime.cli import main

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "quant_investor/v17_v5_runtime"


def _tree(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def test_v5_runtime_has_no_v4_runtime_or_writer_import() -> None:
    forbidden_modules = {
        "quant_investor.factors.forward_evaluator",
        "quant_investor.factors.production_control_v1",
        "quant_investor.factors.registry_store",
    }
    for path in sorted(RUNTIME.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("quant_investor.v17_v4_runtime"), path
                assert (node.module or "") not in forbidden_modules, path
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("quant_investor.v17_v4_runtime")
                    for alias in node.names
                ), path
                assert all(alias.name not in forbidden_modules for alias in node.names), path


def test_v5_authority_is_permanently_false() -> None:
    authority = authority_envelope()

    assert set(authority) == {
        "broker",
        "canary",
        "execution",
        "factor_governance_write",
        "formal_activation",
        "formal_research_publication",
        "llm",
        "order",
        "portfolio",
        "promotion",
        "provider",
        "research_runtime_default",
        "selector",
        "trade",
    }
    assert all(value is False for value in authority.values())
    assert DELIVERY_STATUS == "SPRINT1A_FACTOR_DIAGNOSTICS_AVAILABLE_NOT_OPERATIONAL"


def test_v5_cli_only_adds_status_and_verify_and_writes_nothing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["quant-investor"] == "quant_investor.cli.main:main"
    assert scripts["quant-investor-v17-v2"] == "quant_investor.v17_v2_runtime.cli:main"
    assert scripts["quant-investor-v17-v3"] == "quant_investor.v17_v3_runtime.cli:main"
    assert scripts["quant-investor-v17-v4"] == "quant_investor.v17_v4_runtime.cli:main"
    assert scripts["quant-investor-v17-v5"] == "quant_investor.v17_v5_runtime.cli:main"
    monkeypatch.chdir(tmp_path)
    before = _tree(tmp_path)

    assert main(["status"]) == 0
    assert main(["verify"]) == 0
    assert _tree(tmp_path) == before


def test_v5_cli_verify_fails_closed_with_exit_two(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = _tree(tmp_path)

    def fail_package_verify():
        raise PackageResourceError("sealed package drift")

    monkeypatch.setattr(cli_subject, "verify_package", fail_package_verify)

    assert main(["verify"]) == 2
    payload = capsys.readouterr().out
    assert '"verified":false' in payload
    assert '"error":"sealed package drift"' in payload
    assert _tree(tmp_path) == before


def test_v5_cli_help_remains_status_and_verify_only(capsys) -> None:
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "{status,verify}" in help_text
    assert "\n    run" not in help_text
