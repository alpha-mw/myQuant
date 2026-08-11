from __future__ import annotations

from pathlib import Path

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
    # Factor package integrity verification reads installed wheel RECORD
    # metadata, not production market bars.
    "scripts/build_factor_v4_3_prior_diagnostic_nomination.py",
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


def test_production_runtime_has_no_legacy_csv_read_ports() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        rel_path = path.relative_to(ROOT).as_posix()
        if rel_path in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                violations.append(f"{rel_path}: {snippet}")

    assert violations == []
