#!/usr/bin/env python3
"""Validate the static CN dashboard export bundle without changing files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_ROOT = PROJECT_ROOT / "portfolio_dashboard"
DEFAULT_SUMMARY_FILE = DEFAULT_DASHBOARD_ROOT / "generated" / "export_summary.json"
DEFAULT_GENERATED_JS = DEFAULT_DASHBOARD_ROOT / "js" / "generated_records.js"
REQUIRED_BENCHMARK_FIELDS = {
    "benchmark_main_nav",
    "benchmark_nav",
    "csi300_nav",
    "csi500_nav",
    "csi1000_nav",
    "star50_nav",
    "chinext_nav",
}
FORBIDDEN_SOURCE_TOKENS = ("sample", "mock", "demo")
SNAPSHOT_SOURCE_SYSTEM = "strategy_record.market_snapshot.indices"


class DashboardExportCheckError(ValueError):
    """Raised when generated_records.js cannot be parsed."""


@dataclass(frozen=True)
class ParsedGeneratedRecords:
    generated_at: str
    source_root: str
    latest_record: str
    record_count: int
    warnings: list[str]
    csv_bundle: dict[str, str]


def _find_matching(text: str, start: int, opener: str, closer: str) -> int:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    raise DashboardExportCheckError(f"unterminated {opener}{closer} block")


def _json_string_property(text: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*:\s*(\"(?:\\.|[^\"\\])*\")", text, flags=re.S)
    if not match:
        raise DashboardExportCheckError(f"generated_records.js missing {name}")
    return str(json.loads(match.group(1)))


def _int_property(text: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*:\s*(\d+)", text)
    if not match:
        raise DashboardExportCheckError(f"generated_records.js missing {name}")
    return int(match.group(1))


def _json_array_property(text: str, name: str) -> list[Any]:
    match = re.search(rf"\b{name}\s*:\s*\[", text)
    if not match:
        raise DashboardExportCheckError(f"generated_records.js missing {name}")
    start = match.end() - 1
    end = _find_matching(text, start, "[", "]")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise DashboardExportCheckError(f"generated_records.js {name} is not an array")
    return parsed


def parse_generated_records(path: Path) -> ParsedGeneratedRecords:
    if not path.exists():
        raise DashboardExportCheckError(f"generated_records.js not found: {path}")
    text = path.read_text(encoding="utf-8")
    if "window.DashboardGeneratedRecords" not in text:
        raise DashboardExportCheckError("generated_records.js missing DashboardGeneratedRecords assignment")
    warnings = _json_array_property(text, "warnings")
    csv_bundle = {
        "nav": _json_string_property(text, "nav"),
        "positions": _json_string_property(text, "positions"),
        "trades": _json_string_property(text, "trades"),
    }
    return ParsedGeneratedRecords(
        generated_at=_json_string_property(text, "generatedAt"),
        source_root=_json_string_property(text, "sourceRoot"),
        latest_record=_json_string_property(text, "latestRecord"),
        record_count=_int_property(text, "recordCount"),
        warnings=[str(item) for item in warnings],
        csv_bundle=csv_bundle,
    )


def parse_csv_rows(csv_text: str) -> list[dict[str, str]]:
    if not csv_text.strip():
        return []
    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        return []
    return list(reader)


def _csv_header(csv_text: str) -> list[str]:
    if not csv_text.strip():
        return []
    reader = csv.reader(StringIO(csv_text))
    try:
        return next(reader)
    except StopIteration:
        return []


def _has_forbidden_source(source_system: str) -> bool:
    normalized = source_system.strip().lower()
    return any(token in normalized for token in FORBIDDEN_SOURCE_TOKENS)


def check_dashboard_export(
    summary_file: Path = DEFAULT_SUMMARY_FILE,
    generated_js: Path = DEFAULT_GENERATED_JS,
    *,
    require_production_benchmark: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not summary_file.exists():
        errors.append(f"export_summary.json not found: {summary_file}")
        return {
            "ok": False,
            "summary_file": str(summary_file),
            "generated_js": str(generated_js),
            "errors": errors,
            "warnings": warnings,
        }
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"export_summary.json is not valid JSON: {exc}")
        summary = {}
    try:
        generated = parse_generated_records(generated_js)
    except DashboardExportCheckError as exc:
        errors.append(str(exc))
        generated = None

    nav_rows: list[dict[str, str]] = []
    positions_rows: list[dict[str, str]] = []
    trades_rows: list[dict[str, str]] = []
    nav_header: list[str] = []
    fallback_to_sample = True
    if generated is not None:
        nav_csv = generated.csv_bundle.get("nav", "")
        positions_csv = generated.csv_bundle.get("positions", "")
        trades_csv = generated.csv_bundle.get("trades", "")
        nav_rows = parse_csv_rows(nav_csv)
        positions_rows = parse_csv_rows(positions_csv)
        trades_rows = parse_csv_rows(trades_csv)
        nav_header = _csv_header(nav_csv)
        fallback_to_sample = not bool(nav_csv.strip() and positions_csv.strip())
        if fallback_to_sample:
            errors.append("generated_records.js does not contain records nav and positions CSV; dashboard would fall back to sample data.")
        if not nav_rows:
            errors.append("generated_records.js nav CSV has no data rows.")
        if not positions_rows:
            errors.append("generated_records.js positions CSV has no data rows.")

    if summary:
        if generated is not None:
            generated_pairs = [
                ("latest_record", summary.get("latest_record"), generated.latest_record),
                ("record_count", summary.get("record_count"), generated.record_count),
                ("generated_at", summary.get("generated_at"), generated.generated_at),
                ("source_root", summary.get("source_root"), generated.source_root),
            ]
            for label, expected, actual in generated_pairs:
                if expected != actual:
                    errors.append(f"{label} mismatch between export_summary.json and generated_records.js: {expected!r} != {actual!r}")
            row_pairs = [
                ("nav_rows", summary.get("nav_rows"), len(nav_rows)),
                ("positions_rows", summary.get("positions_rows"), len(positions_rows)),
                ("trade_rows", summary.get("trade_rows"), len(trades_rows)),
            ]
            for label, expected, actual in row_pairs:
                if expected != actual:
                    errors.append(f"{label} mismatch between export_summary.json and generated_records.js CSV: {expected!r} != {actual!r}")

        nav_source = summary.get("portfolio_nav_source") or {}
        funding_events = nav_source.get("funding_events") or []
        if funding_events:
            if nav_source.get("method") != "time_weighted_unitization":
                errors.append(
                    "portfolio NAV with external funding must use method='time_weighted_unitization'."
                )
            if nav_source.get("historical_return_preserved") is not True:
                errors.append("portfolio NAV funding lineage does not preserve historical return.")
            if "portfolio_units" not in nav_header:
                errors.append("generated_records.js nav CSV missing portfolio_units for funded unit NAV.")
            for index, event in enumerate(funding_events):
                if event.get("total_value_before") is None or event.get("total_value_after") is None:
                    errors.append(
                        f"portfolio funding event {index} missing total_value_before/total_value_after."
                    )
        capital_start = nav_source.get("capital_base_start")
        capital_end = nav_source.get("capital_base_end")
        if capital_start is not None and capital_end is not None:
            try:
                capital_changed = abs(float(capital_end) - float(capital_start)) > 0.01
            except (TypeError, ValueError):
                errors.append("portfolio NAV capital_base_start/capital_base_end is not numeric.")
            else:
                if capital_changed and not funding_events:
                    errors.append("portfolio capital base changed without a funding event in NAV lineage.")

        ledger_status = summary.get("effective_manual_ledger_status") or {}
        if ledger_status:
            if ledger_status.get("legacy_ledger_fallback_used") is not False:
                errors.append("effective manual positions must not use legacy ledger.csv fallback.")
            if ledger_status.get("status") == "valid":
                ledger_path = str(ledger_status.get("ledger_path") or "")
                manifest_path = str(ledger_status.get("manifest_path") or "")
                if not ledger_path.endswith("/ledger_after_manual_switch.csv"):
                    errors.append("effective manual ledger path is not ledger_after_manual_switch.csv.")
                if not manifest_path.endswith("/manual_execution_manifest.json"):
                    errors.append("effective manual ledger is missing its manual_execution_manifest.json lineage.")

        benchmark_source = summary.get("benchmark_source") or {}
        benchmark_fields = set(benchmark_source.get("benchmark_fields") or [])
        missing_fields = sorted(REQUIRED_BENCHMARK_FIELDS - benchmark_fields)
        missing_nav_fields = sorted(REQUIRED_BENCHMARK_FIELDS - set(nav_header))
        if missing_fields:
            errors.append(f"export_summary.json benchmark fields missing: {missing_fields}")
        if missing_nav_fields:
            errors.append(f"generated_records.js nav CSV benchmark fields missing: {missing_nav_fields}")

        source_system = str(benchmark_source.get("source_system") or "")
        source_status = str(benchmark_source.get("benchmark_source_status") or "")
        production_grade = bool(benchmark_source.get("production_grade"))
        if _has_forbidden_source(source_system):
            errors.append(f"benchmark source_system contains sample/mock/demo token: {source_system}")
        if SNAPSHOT_SOURCE_SYSTEM in source_system and production_grade:
            errors.append("strategy_record market_snapshot benchmark cannot be marked production_grade.")
        if production_grade and "partial_missing" in source_status:
            errors.append(
                "benchmark cannot be production_grade while benchmark_source_status="
                f"{source_status!r}."
            )
        if production_grade and source_status == "not_production_grade":
            errors.append("benchmark_source_status=not_production_grade cannot be production_grade.")
        if require_production_benchmark and not production_grade:
            errors.append(
                "benchmark is not production_grade; fill a verified continuous real index close source before using formal dashboard benchmark."
            )

        trade_completeness = summary.get("trade_record_completeness") or {}
        trade_status = str(trade_completeness.get("status") or "")
        skipped_trades = int(trade_completeness.get("skipped_incomplete_rows") or 0)
        if trade_status and trade_status != "complete":
            errors.append(
                "trade_record_completeness is not complete: "
                f"status={trade_status!r}, skipped_incomplete_rows={skipped_trades}."
            )
        if not production_grade:
            warnings.append(
                "Dashboard benchmark is not formal investment-committee grade: "
                f"status={source_status}, source_system={source_system}."
            )
        warnings.extend(str(item) for item in summary.get("warnings") or [])

    generated_mtime = generated_js.stat().st_mtime if generated_js.exists() else None
    summary_mtime = summary_file.stat().st_mtime if summary_file.exists() else None
    benchmark_source = summary.get("benchmark_source") if summary else {}
    result = {
        "ok": not errors,
        "summary_file": str(summary_file),
        "generated_js": str(generated_js),
        "generated_js_mtime": generated_mtime,
        "summary_mtime": summary_mtime,
        "latest_record": summary.get("latest_record") if summary else "",
        "record_count": summary.get("record_count") if summary else 0,
        "nav_rows": len(nav_rows),
        "positions_rows": len(positions_rows),
        "trade_rows": len(trades_rows),
        "fallback_to_sample": fallback_to_sample,
        "benchmark_source": benchmark_source,
        "require_production_benchmark": require_production_benchmark,
        "warnings": warnings,
        "errors": errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-root", type=Path, default=DEFAULT_DASHBOARD_ROOT)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--generated-js", type=Path)
    parser.add_argument(
        "--require-production-benchmark",
        action="store_true",
        help="Exit nonzero unless benchmark_source.production_grade is true.",
    )
    args = parser.parse_args()
    summary_file = args.summary_file or args.dashboard_root / "generated" / "export_summary.json"
    generated_js = args.generated_js or args.dashboard_root / "js" / "generated_records.js"

    result = check_dashboard_export(
        summary_file=summary_file,
        generated_js=generated_js,
        require_production_benchmark=args.require_production_benchmark,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
