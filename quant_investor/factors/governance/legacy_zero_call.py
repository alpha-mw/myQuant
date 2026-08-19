"""Fixed release-tree scanner for retired Factor runtime reachability."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes
from quant_investor.system.store import validate_object_ref

from .errors import FactorGovernanceError

SCANNER_MODULE_PATH: Final = "quant_investor/factors/governance/legacy_zero_call.py"
SCANNER_COMMAND: Final = "quant-investor factor verify-legacy-zero-call"
_LEGACY_MODULE_PREFIXES: Final = (
    "quant_investor.v17_v4_runtime",
    "quant_investor.v17_v4_contract",
    "quant_investor.v17",
)
_LEGACY_TEXT_MARKERS: Final = (
    "quant_investor/v17_v4_runtime",
    "quant_investor/v17_v4_contract",
    "quant_investor.v17_v4_runtime",
    "quant_investor.v17_v4_contract",
)

ProcessRunner = Callable[..., Any]


def _run(
    runner: ProcessRunner,
    arguments: Sequence[str],
    *,
    repository_root: Path,
) -> bytes:
    try:
        result = runner(
            list(arguments),
            cwd=str(repository_root),
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        raise FactorGovernanceError("Factor legacy scanner process failed") from exc
    if (
        type(getattr(result, "returncode", None)) is not int
        or result.returncode != 0
        or type(getattr(result, "stdout", None)) is not bytes
        or type(getattr(result, "stderr", None)) is not bytes
    ):
        raise FactorGovernanceError("Factor legacy scanner process result differs")
    return result.stdout


def _dotted(node: ast.AST, aliases: Mapping[str, str] | None = None) -> str:
    if isinstance(node, ast.Name):
        value = node.id
    elif isinstance(node, ast.Attribute):
        parent = _dotted(node.value, aliases)
        value = f"{parent}.{node.attr}" if parent else node.attr
    else:
        value = ""
    if not value or not aliases:
        return value
    head, separator, tail = value.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{tail}" if separator else resolved


def _contains_legacy(value: str) -> bool:
    return any(marker in value for marker in (*_LEGACY_MODULE_PREFIXES, *_LEGACY_TEXT_MARKERS))


def _literal_strings(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _scan_python(raw: bytes) -> tuple[int, int, int]:  # noqa: C901
    try:
        tree = ast.parse(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, SyntaxError) as exc:
        raise FactorGovernanceError("Factor release Python source cannot be scanned") from exc
    aliases: dict[str, str] = {}
    imports = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                target = alias.name if alias.asname else local
                aliases[local] = target
                imports += int(
                    any(alias.name.startswith(prefix) for prefix in _LEGACY_MODULE_PREFIXES)
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                target = f"{module}.{alias.name}" if module else alias.name
                aliases[alias.asname or alias.name] = target
                imports += int(any(target.startswith(prefix) for prefix in _LEGACY_MODULE_PREFIXES))
    calls = 0
    path_hashes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = _dotted(node.func, aliases)
            calls += int(_contains_legacy(function))
            literals = [value for argument in node.args for value in _literal_strings(argument)]
            if function in {"importlib.import_module", "__import__", "builtins.__import__"}:
                imports += sum(_contains_legacy(value) for value in literals)
            if function in {"hash", "hashlib.sha256", "hashlib.md5", "hashlib.blake2b"}:
                path_hashes += sum(_contains_legacy(value) for value in literals)
            if function in {
                "open",
                "builtins.open",
                "os.open",
                "Path",
                "pathlib.Path",
                "importlib.resources.files",
                "importlib.resources.open_binary",
                "importlib.util.spec_from_file_location",
                "subprocess.run",
                "subprocess.call",
                "subprocess.check_call",
                "subprocess.check_output",
                "subprocess.Popen",
            }:
                path_hashes += sum(_contains_legacy(value) for value in literals)
    return imports, calls, path_hashes


def _scan_release_legacy_zero_call_with_runner(
    *,
    repository_root: str | Path,
    final_commit: str,
    final_tree: str,
    resolver_inventory_ref: Mapping[str, Any],
    process_runner: ProcessRunner = subprocess.run,
) -> dict[str, Any]:
    """Scan exact committed bytes; callers cannot supply count results."""

    root = Path(repository_root).resolve(strict=True)
    observed_tree = (
        _run(
            process_runner,
            ["git", "rev-parse", f"{final_commit}^{{tree}}"],
            repository_root=root,
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if observed_tree != final_tree:
        raise FactorGovernanceError("Factor legacy scanner release tree differs")
    paths = [
        value
        for value in _run(
            process_runner,
            ["git", "ls-tree", "-r", "--name-only", final_commit],
            repository_root=root,
        )
        .decode("utf-8", errors="strict")
        .splitlines()
        if value
    ]
    if paths != sorted(set(paths)):
        raise FactorGovernanceError("Factor legacy scanner release inventory differs")
    imports = calls = path_hashes = entrypoints = 0
    scanner_raw: bytes | None = None
    for relative in paths:
        if (
            not (relative.startswith("quant_investor/") and relative.endswith(".py"))
            and relative != "pyproject.toml"
        ):
            continue
        raw = _run(
            process_runner,
            ["git", "show", f"{final_commit}:{relative}"],
            repository_root=root,
        )
        if relative == SCANNER_MODULE_PATH:
            scanner_raw = raw
        if relative.endswith(".py") and not relative.startswith(
            ("quant_investor/v17_v4_runtime/", "quant_investor/v17_v4_contract/")
        ):
            observed = _scan_python(raw)
            imports += observed[0]
            calls += observed[1]
            path_hashes += observed[2]
        elif relative == "pyproject.toml":
            entrypoints += sum(
                _contains_legacy(line)
                for line in raw.decode("utf-8", errors="strict").splitlines()
                if "=" in line
            )
    if scanner_raw is None:
        raise FactorGovernanceError("Factor legacy scanner is absent from release")
    summary = {
        "active_legacy_import_count": imports,
        "active_legacy_call_count": calls,
        "active_legacy_path_hash_count": path_hashes,
        "legacy_entrypoint_count": entrypoints,
    }
    return {
        "final_commit": final_commit,
        "final_tree": final_tree,
        "resolver_inventory_ref": validate_object_ref(resolver_inventory_ref),
        **summary,
        "verification_module_path": SCANNER_MODULE_PATH,
        "verification_module_sha256": hashlib.sha256(scanner_raw).hexdigest(),
        "verification_command": SCANNER_COMMAND,
        "stdout": canonical_json_bytes(summary),
        "stderr": b"",
    }


def scan_release_legacy_zero_call(
    *,
    repository_root: str | Path,
    final_commit: str,
    final_tree: str,
    resolver_inventory_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Production scanner; the process implementation is fixed in code."""

    return _scan_release_legacy_zero_call_with_runner(
        repository_root=repository_root,
        final_commit=final_commit,
        final_tree=final_tree,
        resolver_inventory_ref=resolver_inventory_ref,
        process_runner=subprocess.run,
    )


__all__ = ["scan_release_legacy_zero_call"]
