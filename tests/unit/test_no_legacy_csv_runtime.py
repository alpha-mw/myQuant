from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = [
    ROOT / "quant_investor",
    ROOT / "daily_runner.py",
    ROOT / "daily_config.py",
    ROOT / "scripts",
]

# Every entry is a reviewed exemption from the legacy-CSV ban. An entry naming a
# path that no longer exists is therefore an exemption nobody decided to grant:
# recreate a file at that path and it inherits permission silently. The entries
# deleted by 389562a (2026-08-05) were pruned for that reason, and
# ``test_allowlist_has_no_dead_entries`` keeps the list from rotting again.
ALLOWLIST = {
    # Formal-review and audit readers consume strategy-record CSV artifacts,
    # never canonical market bars.
    # Dashboard export reads strategy-record CSV artifacts, not runtime market data.
    "scripts/backfill_cn_dashboard_benchmark.py",
    # Official valuation reads only the Dashboard benchmark series as CSV;
    # its governed holdings source remains the active canonical Parquet ledger.
    "scripts/close_cn_dashboard_official_valuation.py",
    "scripts/cn_dashboard_common.py",
    "scripts/merge_cn_dashboard_benchmark_fills.py",
}

FORBIDDEN_SNIPPETS = [
    "pd.read_csv(",
    "pandas.read_csv(",
    "read_csv(",
    "SharedCSVReader",
    "SharedCSVReadResult",
    "CSVStore",
    "USLocalCSVDataSource",
    "csv_reader",
    "csv_store",
    "shared_csv_reader",
    "csv.DictReader",
    "csv.reader",
]

HISTORICAL_AUDIT_PATH = "scripts/prepare_cn_strategy_accounting.py"
HISTORICAL_DECODER = "_historical_audit_rows"


def _historical_csv_lines(text: str, relative: str) -> tuple[set[int], list[str]]:
    """Review one audit-only decode, without exempting its production caller file."""
    tree = ast.parse(text)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    decoders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == HISTORICAL_DECODER
    ]
    references = [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == HISTORICAL_DECODER)
        or (isinstance(node, ast.Attribute) and node.attr == HISTORICAL_DECODER)
        or (isinstance(node, ast.alias) and node.name == HISTORICAL_DECODER)
        or (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").endswith("prepare_cn_strategy_accounting")
            and any(alias.name == "*" for alias in node.names)
        )
    ]
    if relative != HISTORICAL_AUDIT_PATH:
        return set(), (
            ["historical decoder escaped its audit module"] if references or decoders else []
        )
    if len(decoders) != 1:
        return set(), ["exact historical decoder is missing or duplicated"]
    decoder = decoders[0]
    calls = [
        node
        for node in ast.walk(decoder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "csv"
        and node.func.attr == "DictReader"
    ]
    if len(calls) != 1:
        return set(), ["historical decoder must contain one csv.DictReader call"]
    for reference in references:
        parent = parents.get(reference)
        owner = parent
        while owner is not None and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parents.get(owner)
        if not (
            isinstance(reference, ast.Name)
            and isinstance(parent, ast.Call)
            and parent.func is reference
            and isinstance(owner, ast.FunctionDef)
            and owner.name == "_extract_historical_audit"
        ):
            return set(), ["historical decoder may only be called directly by the gap audit"]
    return {calls[0].func.lineno}, []


def _csv_violations(text: str, relative: str) -> list[str]:
    allowed_lines, problems = _historical_csv_lines(text, relative)
    for line_number, line in enumerate(text.splitlines(), start=1):
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in line and not (
                snippet == "csv.DictReader" and line_number in allowed_lines
            ):
                problems.append(f"{relative}:{line_number}: {snippet}")
    return problems


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def test_allowlist_has_no_dead_entries() -> None:
    """A deleted file must not leave its CSV exemption behind.

    The allowlist grants permission by path, so an entry whose file is gone
    silently re-grants that permission to whatever is written at the path next.
    Pruning is part of deleting the file, not a later cleanup.
    """

    dead = sorted(entry for entry in ALLOWLIST if not (ROOT / entry).exists())
    assert dead == []
    assert HISTORICAL_AUDIT_PATH not in ALLOWLIST


def test_production_runtime_has_no_legacy_csv_read_ports() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        rel_path = path.relative_to(ROOT).as_posix()
        if rel_path in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        violations.extend(_csv_violations(text, rel_path))

    assert violations == []


@pytest.mark.parametrize(
    "escape",
    [
        "def prepare(raw):\n    return _historical_audit_rows(raw)\n",
        "def prepare(raw):\n    decode = _historical_audit_rows\n    return decode(raw)\n",
        "def _extract_historical_audit(raw):\n    decode = _historical_audit_rows\n",
        "def prepare(raw):\n    return csv.DictReader(raw)\n",
    ],
)
def test_historical_csv_exception_rejects_prospective_and_alias_escapes(escape):
    source = (
        "def _historical_audit_rows(raw):\n    return list(csv.DictReader(raw))\n"
        "def _extract_historical_audit(raw):\n    return _historical_audit_rows(raw)\n"
    )
    assert _csv_violations(source, HISTORICAL_AUDIT_PATH) == []
    assert _csv_violations(source + escape, HISTORICAL_AUDIT_PATH)


@pytest.mark.parametrize(
    "source",
    [
        "from scripts.prepare_cn_strategy_accounting import _historical_audit_rows as decode\n",
        "from scripts.prepare_cn_strategy_accounting import *\n",
        "rows = accounting._historical_audit_rows(raw)\n",
    ],
)
def test_historical_decoder_cannot_be_imported_by_production_callers(source):
    assert _csv_violations(source, "quant_investor/example.py")
