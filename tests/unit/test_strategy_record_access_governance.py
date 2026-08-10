from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from scripts.check_strategy_record_access import (
    ALLOW_RULES,
    AllowRule,
    audit_findings,
    scan_repository,
)


ROOT = Path(__file__).resolve().parents[2]


def _git_add(repo: Path, *paths: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)


def test_repository_has_zero_unexplained_strategy_record_access() -> None:
    findings = scan_repository(ROOT)
    assert audit_findings(findings, ALLOW_RULES) == ()


def test_gate_detects_target_bound_direct_operations(tmp_path: Path) -> None:
    production = tmp_path / "app.py"
    production.write_text(
        "from pathlib import Path\n"
        'record_root = Path("results/strategy_records")\n'
        "record_root.mkdir()\n"
        'record_root.open("wb")\n',
        encoding="utf-8",
    )
    _git_add(tmp_path, "app.py")

    findings = scan_repository(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == "app.py"
    assert findings[0].operations == ("mkdir", "open")
    problems = audit_findings(findings, ())
    assert len(problems) == 1
    assert problems[0].startswith("unexplained access: app.py")


def test_exact_allow_rule_rejects_operation_drift(tmp_path: Path) -> None:
    production = tmp_path / "reader.py"
    production.write_text(
        "from pathlib import Path\n"
        'record_root = Path("results/strategy_records")\n'
        "list(record_root.iterdir())\n",
        encoding="utf-8",
    )
    _git_add(tmp_path, "reader.py")
    findings = scan_repository(tmp_path)
    wrong_rule = AllowRule(
        path="reader.py",
        operations=(),
        reason="Reviewed literal-only caller.",
    )

    problems = audit_findings(findings, (wrong_rule,))

    assert len(problems) == 1
    assert problems[0].startswith("operation permission drift: reader.py")


def test_gate_scans_tracked_shell_and_javascript(tmp_path: Path) -> None:
    shell = tmp_path / "job.sh"
    shell.write_text(
        'record_root="results/strategy_records"\nmkdir "$record_root"\n',
        encoding="utf-8",
    )
    javascript = tmp_path / "dashboard.js"
    javascript.write_text(
        'const recordRoot = "results/strategy_records"; open(recordRoot);\n',
        encoding="utf-8",
    )
    _git_add(tmp_path, "job.sh", "dashboard.js")

    findings = scan_repository(tmp_path)

    assert [(item.path, item.operations) for item in findings] == [
        ("dashboard.js", ("open",)),
        ("job.sh", ("mkdir",)),
    ]


def test_gate_scans_canonical_backend_without_embedded_root_literal(
    tmp_path: Path,
) -> None:
    backend = tmp_path / "quant_investor" / "strategy_records" / "store.py"
    backend.parent.mkdir(parents=True)
    backend.write_text(
        "def publish(store_root):\n"
        "    store_root.mkdir()\n",
        encoding="utf-8",
    )
    _git_add(tmp_path, "quant_investor/strategy_records/store.py")

    findings = scan_repository(tmp_path)

    assert len(findings) == 1
    assert findings[0].target_lines == (0,)
    assert findings[0].operations == ("mkdir",)


def test_gate_ignores_tests_docs_fixtures_and_untracked_sources(tmp_path: Path) -> None:
    for relative in (
        "tests/test_fixture.py",
        "docs/example.js",
        "fixtures/record.sh",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            'target="results/strategy_records"\nopen(target)\n',
            encoding="utf-8",
        )
    untracked = tmp_path / "untracked.py"
    untracked.write_text(
        'open("results/strategy_records")\n',
        encoding="utf-8",
    )
    _git_add(
        tmp_path,
        "tests/test_fixture.py",
        "docs/example.js",
        "fixtures/record.sh",
    )

    assert scan_repository(tmp_path) == ()


def test_quarantine_rename_is_scoped_to_exact_manager_function() -> None:
    manager = ROOT / "scripts/manage_cn_strategy_records.py"
    tree = ast.parse(manager.read_text(encoding="utf-8"))
    rename_owners: list[str] = []
    forbidden_source_move_calls: list[str] = []
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        for call in (
            node for node in ast.walk(function) if isinstance(node, ast.Call)
        ):
            name = ast.unparse(call.func)
            if name == "os.rename":
                rename_owners.append(function.name)
            if name in {
                "shutil.move",
                "shutil.rmtree",
                "os.unlink",
                "os.remove",
            }:
                forbidden_source_move_calls.append(name)

    assert rename_owners == ["_move_records"]
    assert forbidden_source_move_calls == []
