#!/usr/bin/env python3
"""Execute every exact replacement selector from the frozen cutover ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.migration.rules import (
    REPLACEMENT_TEST_MAP_RELATIVE_PATH,
    load_replacement_test_map,
)


@dataclass(eq=False)
class _SelectionAudit:
    requested: tuple[str, ...]
    collected: tuple[str, ...] = ()
    executed: set[str] = field(default_factory=set)
    deselected: int = 0
    skipped: int = 0
    xfailed: int = 0
    failed: int = 0

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = tuple(item.nodeid for item in session.items)

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected += len(items)

    def pytest_runtest_logreport(self, report: Any) -> None:
        if getattr(report, "wasxfail", None) is not None:
            self.xfailed += 1
        if report.skipped:
            self.skipped += 1
        if report.failed:
            self.failed += 1
        if report.when == "call" and report.passed:
            self.executed.add(report.nodeid)


def _matches(nodeid: str, selector: str) -> bool:
    return nodeid == selector or nodeid.startswith(selector + "[")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    loaded = load_replacement_test_map(root, REPLACEMENT_TEST_MAP_RELATIVE_PATH)
    selectors = tuple(
        sorted(
            {selector for entry in loaded.entries for selector in entry.replacement_test_selectors}
        )
    )
    audit = _SelectionAudit(selectors)
    exit_code = int(
        pytest.main(
            [*selectors, "-q", "--disable-warnings"],
            plugins=[audit],
        )
    )
    missing_collection = [
        selector
        for selector in selectors
        if not any(_matches(nodeid, selector) for nodeid in audit.collected)
    ]
    missing_execution = [
        selector
        for selector in selectors
        if not any(_matches(nodeid, selector) for nodeid in audit.executed)
    ]
    unexpected = [
        nodeid
        for nodeid in audit.collected
        if not any(_matches(nodeid, selector) for selector in selectors)
    ]
    passed = (
        len(selectors) == 130
        and exit_code == 0
        and not missing_collection
        and not missing_execution
        and not unexpected
        and audit.deselected == 0
        and audit.skipped == 0
        and audit.xfailed == 0
        and audit.failed == 0
    )
    summary = {
        "kind": "system.replacement-selector-execution",
        "state": "PASS" if passed else "FAIL",
        "replacement_map_sha256": loaded.sha256,
        "occurrence_count": sum(len(entry.replacement_test_selectors) for entry in loaded.entries),
        "unique_selector_count": len(selectors),
        "collected_item_count": len(audit.collected),
        "executed_item_count": len(audit.executed),
        "missing_collection_count": len(missing_collection),
        "missing_execution_count": len(missing_execution),
        "unexpected_item_count": len(unexpected),
        "deselected_count": audit.deselected,
        "skipped_count": audit.skipped,
        "xfailed_count": audit.xfailed,
        "failed_report_count": audit.failed,
        "pytest_exit_code": exit_code,
    }
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
