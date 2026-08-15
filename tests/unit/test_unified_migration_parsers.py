from __future__ import annotations

import ast

import pytest

from quant_investor.migration.errors import (
    DYNAMIC_IMPORT_ALLOWLIST_MISMATCH,
    DYNAMIC_IMPORT_NOT_ALLOWLISTED,
    UNPARSEABLE_SHELL,
    UnifiedCutoverError,
)
from quant_investor.migration.parsers import (
    ast_node_sha256,
    parse_python_imports,
    parse_shell_config_edges,
    parse_toml_config_edges,
    parse_yaml_config_edges,
    shell_tokens,
)
from quant_investor.migration.rules import DynamicImportAllowance


def _only_call(source: str) -> ast.Call:
    return next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call))


def test_ast_literal_importlib_and_dunder_import_are_exact_edges() -> None:
    raw = b"import importlib as loader\nloader.import_module('pkg.alpha')\n__import__('pkg.beta')\n"
    parsed = parse_python_imports(
        raw,
        relative_path="src/main.py",
        module_name="src.main",
        is_package=False,
        allowlist={},
    )
    assert {(edge.module, edge.kind) for edge in parsed.imports} >= {
        ("importlib", "AST_IMPORT"),
        ("pkg.alpha", "AST_LITERAL_DYNAMIC"),
        ("pkg.beta", "AST_LITERAL_DYNAMIC"),
    }
    assert parsed.used_allowlist_keys == frozenset()


def test_nonliteral_dynamic_import_requires_path_line_ast_sha_and_finite_modules() -> None:
    source = "import importlib\nname = choose()\nimportlib.import_module(name)\n"
    call = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
    ][0]
    digest = ast_node_sha256(call)
    key = ("src/main.py", 3, digest)
    allowance = DynamicImportAllowance("src/main.py", 3, digest, ("pkg.alpha", "pkg.beta"))
    parsed = parse_python_imports(
        source.encode(),
        relative_path="src/main.py",
        module_name="src.main",
        is_package=False,
        allowlist={key: allowance},
    )
    assert parsed.used_allowlist_keys == frozenset({key})
    assert {edge.module for edge in parsed.imports if edge.kind == "AST_ALLOWLIST_DYNAMIC"} == {
        "pkg.alpha",
        "pkg.beta",
    }

    with pytest.raises(UnifiedCutoverError) as missing:
        parse_python_imports(
            source.encode(),
            relative_path="src/main.py",
            module_name="src.main",
            is_package=False,
            allowlist={},
        )
    assert missing.value.code == DYNAMIC_IMPORT_NOT_ALLOWLISTED

    bad = DynamicImportAllowance("src/other.py", 3, digest, ("pkg.alpha",))
    with pytest.raises(UnifiedCutoverError) as mismatch:
        parse_python_imports(
            source.encode(),
            relative_path="src/main.py",
            module_name="src.main",
            is_package=False,
            allowlist={key: bad},
        )
    assert mismatch.value.code == DYNAMIC_IMPORT_ALLOWLIST_MISMATCH


def test_structural_toml_yaml_and_shlex_routes_preserve_quoted_tokens() -> None:
    console = {"quant-investor": "quant_investor.cli.main:main"}
    toml_edges = parse_toml_config_edges(
        b'[project.scripts]\nquant-investor="quant_investor.cli.main:main"\n'
        b'[tool.runner]\ncommand="python -m quant_investor.cli.main system verify"\n',
        relative_path="pyproject.toml",
    )
    assert {edge.target for edge in toml_edges} == {"quant_investor.cli.main"}

    yaml_edges = parse_yaml_config_edges(
        b"jobs:\n  verify:\n    steps:\n      - run: 'python -m quant_investor.cli.main system verify'\n",
        relative_path=".github/workflows/verify.yml",
        console_scripts=console,
    )
    assert [(edge.target, edge.source_kind) for edge in yaml_edges] == [
        ("quant_investor.cli.main", "SHELL_PYTHON_M")
    ]

    workflow_edges = parse_yaml_config_edges(
        b"jobs:\n"
        b"  verify:\n"
        b"    steps:\n"
        b"      - name: Verify\n"
        b"        uses: actions/setup-python@v5\n"
        b"        with:\n"
        b'          python-version: "3.13"\n'
        b"      - run: quant-investor system verify\n",
        relative_path=".github/workflows/verify.yml",
        console_scripts=console,
    )
    assert [(edge.target, edge.source_kind) for edge in workflow_edges] == [
        ("quant_investor.cli.main", "CONSOLE_SCRIPT")
    ]

    shell_edges = parse_shell_config_edges(
        b'python "scripts/check file.py" && quant-investor system verify\n',
        relative_path="scripts/run.sh",
        console_scripts=console,
    )
    assert {(edge.target_kind, edge.target) for edge in shell_edges} == {
        ("path", "scripts/check file.py"),
        ("module", "quant_investor.cli.main"),
    }
    assert shell_tokens("python -m 'pkg.module'", label="quoted") == (
        "python",
        "-m",
        "pkg.module",
    )
    with pytest.raises(UnifiedCutoverError) as exc:
        shell_tokens("python 'unterminated", label="bad")
    assert exc.value.code == UNPARSEABLE_SHELL
