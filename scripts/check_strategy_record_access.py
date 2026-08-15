#!/usr/bin/env python3
"""Fail closed on unreviewed strategy-record filesystem access.

The scan covers Git-tracked production Python, shell, and JavaScript plus the
exact reviewed runtime paths in ``ALLOW_RULES``.  The latter keeps a dirty
migration checkout fail-closed when a required backend is still untracked.
Tests, documentation, fixtures, generated samples, vendored code, and live
results are not production callers and are excluded.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SOURCE_SUFFIXES = frozenset({".py", ".sh", ".bash", ".zsh", ".js", ".mjs", ".cjs"})
EXCLUDED_PREFIXES = (
    "docs/",
    "results/",
    "tests/",
    "test/",
    "fixtures/",
    "portfolio_dashboard/sample/",
    "node_modules/",
    "vendor/",
    "dist/",
    "build/",
)
EXCLUDED_PARTS = frozenset({"__pycache__", "fixtures", "generated", "snapshots"})
EXCLUDED_FILES = frozenset({"scripts/check_strategy_record_access.py"})
TARGET_RE = re.compile(r"strategy[_-]records", re.IGNORECASE)
TARGETISH_NAME_RE = re.compile(
    r"(?:strategy_records?|strategy_record|record_store|record|store|catalog|"
    r"receipt|archive|object|stage)"
    r"(?:_(?:root|dir|path|file|store))?$",
    re.IGNORECASE,
)

METHOD_KINDS = {
    "glob": "glob",
    "rglob": "glob",
    "walk": "glob",
    "iterdir": "iterdir",
    "scandir": "iterdir",
    "mkdir": "mkdir",
    "makedirs": "mkdir",
    "open": "open",
    "fdopen": "open",
    "read": "open",
    "write": "open",
    "fsync": "open",
    "read_text": "open",
    "read_bytes": "open",
    "write_text": "open",
    "write_bytes": "open",
    "rename": "rename",
    "replace": "rename",
    "move": "rename",
    "copy": "copy",
    "copy2": "copy",
    "copyfile": "copy",
    "copytree": "copy",
    "link": "copy",
    "unlink": "delete",
    "rmdir": "delete",
    "rmtree": "delete",
    "remove": "delete",
}
TEXT_OPERATION_RE = re.compile(
    r"\b(glob|rglob|walk|iterdir|scandir|mkdir|makedirs|open|read_text|read_bytes|"
    r"write_text|write_bytes|rename|replace|move|copy|copy2|copyfile|copytree|"
    r"unlink|rmdir|rmtree|remove)\b"
)


@dataclass(frozen=True)
class Finding:
    path: str
    target_lines: tuple[int, ...]
    operations: tuple[str, ...]
    operation_lines: tuple[int, ...]


@dataclass(frozen=True)
class AllowRule:
    path: str
    operations: tuple[str, ...]
    reason: str


# Exact operation sets are deliberate: for every present reviewed caller, adding
# or removing direct access requires a reviewed allow-table change.
ALLOW_RULES: tuple[AllowRule, ...] = (
    AllowRule(
        path="quant_investor/automation/daily_runner.py",
        operations=(),
        reason="Reads the active unified generation and passes an explicit history strategy.",
    ),
    AllowRule(
        path="quant_investor/strategy_records/history.py",
        operations=(),
        reason=(
            "Reads only catalog-resolved registered history and an explicitly resolved "
            "report file; it never scans strategy directories or writes the store."
        ),
    ),
    AllowRule(
        path="quant_investor/automation/report_builder.py",
        operations=(),
        reason="Formats registered history DTOs and has no Strategy Record Store I/O.",
    ),
    AllowRule(
        path="quant_investor/strategy_records/store.py",
        operations=("copy", "delete", "iterdir", "mkdir", "open", "rename"),
        reason=(
            "The sole canonical store backend implements governed immutable publication, "
            "exact child-set validation, safe temporary cleanup, and registered reads."
        ),
    ),
    AllowRule(
        path="scripts/build_cn_dashboard_history_integrity.py",
        operations=(),
        reason="Invokes the named legacy bootstrap projection; it does not mutate records.",
    ),
    AllowRule(
        path="scripts/build_holdings_fundamental_sheet.py",
        operations=(),
        reason="Resolves the active closure through the store API and reads Parquet only.",
    ),
    AllowRule(
        path="scripts/close_cn_dashboard_official_valuation.py",
        operations=(),
        reason=(
            "Binds a Dashboard-only no-trade valuation to the registered active "
            "closure and expected pointer SHA through the store API; the manager "
            "alone owns record publication and pointer CAS."
        ),
    ),
    AllowRule(
        path="scripts/cn_dashboard_common.py",
        operations=("iterdir",),
        reason=(
            "Registered Dashboard reads use catalog projections; direct traversal remains "
            "only in the two named legacy/bootstrap scanners."
        ),
    ),
    AllowRule(
        path="scripts/export_cn_aggressive_dashboard_data.py",
        operations=(),
        reason="Passes the registered root to the common exporter and writes bundles elsewhere.",
    ),
    AllowRule(
        path="scripts/manage_cn_strategy_records.py",
        operations=("copy", "glob", "iterdir", "mkdir", "open", "rename"),
        reason=(
            "Privileged operator CLI owns governed stage/archive exact-byte copy I/O and the exact "
            "operation-lock-protected same-device quarantine rename state machine; it "
            "provides no source-record delete or copy fallback."
        ),
    ),
)


def _tracked_sources(repo_root: Path) -> list[Path]:
    command = ["git", "ls-files", "-z", "--"]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        relative_text = relative.as_posix()
        if relative.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        candidate = repo_root / relative
        if candidate.is_symlink():
            raise RuntimeError(f"tracked strategy-record caller must not be a symlink: {relative}")
        if not candidate.is_file():
            # A hard-cutover worktree may contain intentional tracked deletions
            # before the final integration commit.  Git-baseline custody is
            # validated separately by the migration resolver.
            continue
        if relative_text in EXCLUDED_FILES:
            continue
        if relative_text.startswith(EXCLUDED_PREFIXES):
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        paths.append(candidate)
    by_path = {path.resolve(): path for path in paths}
    for rule in ALLOW_RULES:
        candidate = repo_root / rule.path
        if candidate.is_file() and candidate.suffix.lower() in SOURCE_SUFFIXES:
            by_path[candidate.resolve()] = candidate
    return sorted(by_path.values())


def _target_lines(text: str) -> tuple[int, ...]:
    return tuple(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if TARGET_RE.search(line)
    )


def _source_segment(text: str, node: ast.AST) -> str:
    return ast.get_source_segment(text, node) or ""


def _python_operations(text: str, path: Path) -> tuple[tuple[str, ...], tuple[int, ...]]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"cannot parse tracked Python source {path}: {exc}") from exc

    tainted_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None or not TARGET_RE.search(_source_segment(text, value)):
            continue
        targets: Iterable[ast.expr]
        if isinstance(node, ast.Assign):
            targets = node.targets
        else:
            targets = (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                tainted_names.add(target.id)

    canonical_backend = path.as_posix().endswith(
        (
            "quant_investor/strategy_records/store.py",
            "scripts/manage_cn_strategy_records.py",
        )
    )

    def mentions_target(node: ast.AST) -> bool:
        segment = _source_segment(text, node)
        if TARGET_RE.search(segment):
            return True
        return any(
            isinstance(child, ast.Name)
            and (
                child.id in tainted_names
                or TARGETISH_NAME_RE.fullmatch(child.id) is not None
            )
            for child in ast.walk(node)
        )

    operations: set[str] = set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        method_name = ""
        receiver: ast.AST | None = None
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            receiver = node.func.value
        elif isinstance(node.func, ast.Name):
            method_name = node.func.id
        kind = METHOD_KINDS.get(method_name)
        if kind is None:
            continue
        relevant = canonical_backend
        if receiver is not None:
            relevant = relevant or mentions_target(receiver)
        else:
            relevant = relevant or any(mentions_target(argument) for argument in node.args)
        if not relevant:
            continue
        operations.add(kind)
        lines.add(int(getattr(node, "lineno", 0)))
    return tuple(sorted(operations)), tuple(sorted(line for line in lines if line))


def _text_operations(text: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    operations: set[str] = set()
    lines: set[int] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not (TARGET_RE.search(line) or re.search(r"\brecord[_-]?(?:root|dir|path)\b", line)):
            continue
        for match in TEXT_OPERATION_RE.finditer(line):
            operations.add(METHOD_KINDS[match.group(1)])
            lines.add(number)
    return tuple(sorted(operations)), tuple(sorted(lines))


def scan_repository(repo_root: Path) -> tuple[Finding, ...]:
    root = repo_root.resolve()
    findings: list[Finding] = []
    for path in _tracked_sources(root):
        text = path.read_text(encoding="utf-8")
        targets = _target_lines(text)
        if path.as_posix().endswith("quant_investor/strategy_records/store.py"):
            # The backend is target-bound by module ownership even though it accepts
            # an injected root and need not contain the repository literal.
            targets = targets or (0,)
        if not targets:
            continue
        if path.suffix.lower() == ".py":
            operations, operation_lines = _python_operations(text, path)
        else:
            operations, operation_lines = _text_operations(text)
        findings.append(
            Finding(
                path=path.relative_to(root).as_posix(),
                target_lines=targets,
                operations=operations,
                operation_lines=operation_lines,
            )
        )
    return tuple(findings)


def audit_findings(
    findings: Sequence[Finding],
    rules: Sequence[AllowRule] = ALLOW_RULES,
) -> tuple[str, ...]:
    by_path = {finding.path: finding for finding in findings}
    rule_by_path = {rule.path: rule for rule in rules}
    problems: list[str] = []
    duplicate_rules = sorted(
        path for path in rule_by_path if sum(rule.path == path for rule in rules) != 1
    )
    problems.extend(f"duplicate allow rule: {path}" for path in duplicate_rules)
    for path in sorted(set(by_path) - set(rule_by_path)):
        finding = by_path[path]
        problems.append(
            f"unexplained access: {path} operations={list(finding.operations)} "
            f"target_lines={list(finding.target_lines)}"
        )
    for path in sorted(set(by_path) & set(rule_by_path)):
        finding = by_path[path]
        rule = rule_by_path[path]
        if finding.operations != tuple(sorted(rule.operations)):
            problems.append(
                f"operation permission drift: {path} expected={list(rule.operations)} "
                f"actual={list(finding.operations)} lines={list(finding.operation_lines)}"
            )
        if not rule.reason.strip():
            problems.append(f"allow rule has no reason: {path}")
    return tuple(problems)


def _print_inventory(findings: Sequence[Finding]) -> None:
    print("path\toperations\ttarget_lines\toperation_lines")
    for finding in findings:
        print(
            f"{finding.path}\t{','.join(finding.operations) or '-'}\t"
            f"{','.join(map(str, finding.target_lines))}\t"
            f"{','.join(map(str, finding.operation_lines)) or '-'}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="print tracked findings without applying the allow table",
    )
    args = parser.parse_args(argv)
    findings = scan_repository(args.repo_root)
    if args.inventory:
        _print_inventory(findings)
        return 0
    problems = audit_findings(findings)
    if problems:
        print("strategy-record access governance failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"strategy-record access governance passed: {len(findings)} reviewed callers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
