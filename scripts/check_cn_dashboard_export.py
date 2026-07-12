#!/usr/bin/env python3
"""Validate the static CN dashboard export bundle without changing files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_ROOT = PROJECT_ROOT / "portfolio_dashboard"
DEFAULT_SUMMARY_FILE = DEFAULT_DASHBOARD_ROOT / "generated" / "export_summary.json"
DEFAULT_GENERATED_JS = DEFAULT_DASHBOARD_ROOT / "private" / "dashboard_snapshot.v2.js"
DASHBOARD_SCHEMA_VERSION = "dashboard_contract.v2"
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
    contract: dict[str, Any] | None = None


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


def _dashboard_contract(text: str) -> dict[str, Any] | None:
    match = re.search(r"window\.DashboardSnapshotV2\s*=\s*\{", text)
    if not match:
        return None
    start = match.end() - 1
    end = _find_matching(text, start, "{", "}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise DashboardExportCheckError("DashboardSnapshotV2 is not an object")
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
        contract=_dashboard_contract(text),
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


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_number_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _is_iso_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or "T" not in value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_dashboard_contract_v2(
    contract: dict[str, Any],
    *,
    schema_file: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Semantic validation beyond JSON Schema's structural checks."""

    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "schema_version",
        "schema_sha256",
        "protocol_hash",
        "run_id",
        "generated_at",
        "status",
        "blockers",
        "as_of_matrix",
        "sources",
        "trading_calendar",
        "nav_return_provenance",
        "nav",
        "positions",
        "trades",
        "themes",
        "theme_protocol",
        "factors",
        "factor_protocol",
        "reconciliation",
        "metric_policy",
    }
    missing = sorted(required - set(contract))
    if missing:
        errors.append(f"dashboard contract v2 missing required fields: {missing}")
    if contract.get("schema_version") != DASHBOARD_SCHEMA_VERSION:
        errors.append(
            f"dashboard schema_version must be {DASHBOARD_SCHEMA_VERSION!r}."
        )
    status = str(contract.get("status") or "")
    if status not in {"fresh", "stale", "partial", "blocked", "sample"}:
        errors.append(f"dashboard contract has invalid status={status!r}.")
    blockers = contract.get("blockers")
    if not isinstance(blockers, list):
        errors.append("dashboard contract blockers must be an array.")
    elif status == "fresh" and blockers:
        errors.append("dashboard contract status=fresh cannot contain blockers.")

    policy = contract.get("metric_policy") or {}
    if not isinstance(policy, dict):
        errors.append("dashboard metric_policy must be an object.")
        policy = {}
    protocol_payload = {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "metric_policy": policy,
        "required_tables": ["nav", "positions", "trades", "themes", "factors"],
    }
    protocol_hash = str(contract.get("protocol_hash") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", protocol_hash):
        errors.append("dashboard protocol_hash must be a lowercase SHA-256 hex digest.")
    elif protocol_hash != _canonical_json_sha256(protocol_payload):
        errors.append("dashboard protocol_hash does not match the canonical v2 metric policy.")

    schema_path = schema_file or (
        DEFAULT_DASHBOARD_ROOT / "schema" / "dashboard_contract.v2.schema.json"
    )
    schema_sha = contract.get("schema_sha256")
    if status != "sample":
        if not isinstance(schema_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", schema_sha):
            errors.append("dashboard schema_sha256 must be present for non-sample snapshots.")
        elif schema_path.is_file():
            actual_schema_sha = hashlib.sha256(schema_path.read_bytes()).hexdigest()
            if schema_sha != actual_schema_sha:
                errors.append("dashboard schema_sha256 does not match the tracked v2 schema.")

    serialized_sources = json.dumps(
        {
            "sources": contract.get("sources") or {},
            "trading_calendar_path": (contract.get("trading_calendar") or {}).get("path_summary")
            if isinstance(contract.get("trading_calendar") or {}, dict)
            else None,
        },
        ensure_ascii=False,
    )
    if re.search(r"(?:/Users/|file://|[A-Za-z]:\\\\)", serialized_sources):
        errors.append("dashboard sources contain a private absolute path.")

    for table in ("nav", "positions", "trades", "themes", "factors"):
        if not isinstance(contract.get(table), list):
            errors.append(f"dashboard contract {table} must be an array.")

    theme_protocol = contract.get("theme_protocol")
    if not isinstance(theme_protocol, dict):
        errors.append("dashboard theme_protocol must be an object.")
    else:
        if theme_protocol.get("schema_version") != "theme_protocol.v2":
            errors.append("dashboard theme_protocol schema_version must be v2.")
        if theme_protocol.get("status") == "blocked" and not theme_protocol.get(
            "blockers"
        ):
            errors.append("blocked theme_protocol must disclose blockers.")
        if (
            theme_protocol.get("status") != "blocked"
            and theme_protocol.get("readback_verified") is not True
        ):
            errors.append("non-blocked theme_protocol requires verified readback.")
        formal_pool = theme_protocol.get("formal_pool")
        if not isinstance(formal_pool, list):
            errors.append("dashboard theme_protocol.formal_pool must be an array.")
            formal_pool = []
        if (
            theme_protocol.get("formal_enabled") is False
            or theme_protocol.get("formal_kill_switch") is True
        ) and formal_pool:
            errors.append(
                "observer-only or killed theme_protocol cannot expose a formal pool."
            )
        if theme_protocol.get("formal_pool_count") != len(formal_pool):
            errors.append("dashboard theme_protocol formal_pool_count mismatch.")

    factor_protocol = contract.get("factor_protocol")
    if not isinstance(factor_protocol, dict):
        errors.append("dashboard factor_protocol must be an object.")
    else:
        if factor_protocol.get("schema_version") != "factor-governance-protocol.v2":
            errors.append("dashboard factor_protocol schema_version must be v2.")
        if factor_protocol.get("status") == "blocked" and not factor_protocol.get("blockers"):
            errors.append("blocked factor_protocol must disclose blockers.")
        if factor_protocol.get("status") != "blocked" and factor_protocol.get("readback_verified") is not True:
            errors.append("non-blocked factor_protocol requires verified readback.")
        producer_available = factor_protocol.get("canonical_producer_available")
        apply_eligible = factor_protocol.get(
            "canonical_production_apply_eligible"
        )
        if status != "sample" and not isinstance(producer_available, bool):
            errors.append(
                "factor_protocol must disclose canonical_producer_available."
            )
        if status != "sample" and not isinstance(apply_eligible, bool):
            errors.append(
                "factor_protocol must disclose canonical_production_apply_eligible."
            )
        if (
            producer_available is not True
            or apply_eligible is not True
        ) and (
            factor_protocol.get("status") == "applied"
            or factor_protocol.get("transition_applied") is True
        ):
            errors.append(
                "factor_protocol cannot display applied while the canonical "
                "full-chain producer is unavailable."
            )
        if (
            factor_protocol.get("status") == "blocked"
            and factor_protocol.get("canonical_producer_blocker")
            and factor_protocol.get("transition_applied") is True
        ):
            errors.append(
                "blocked factor_protocol must clear transition_applied."
            )

    for index, row in enumerate(contract.get("nav") or []):
        if not isinstance(row, dict):
            continue
        for field, value in row.items():
            if str(field).endswith("_nav") and not _is_number_or_none(value):
                errors.append(
                    f"nav[{index}].{field} must be numeric or null."
                )

    for index, row in enumerate(contract.get("positions") or []):
        if not isinstance(row, dict):
            errors.append(f"positions[{index}] is not an object.")
            continue
        daily_return = row.get("daily_return")
        nav_weight = row.get("nav_weight")
        sleeve_weight = row.get("equity_sleeve_weight")
        contribution = row.get("contribution")
        for field, value in (
            ("daily_return", daily_return),
            ("nav_weight", nav_weight),
            ("equity_sleeve_weight", sleeve_weight),
            ("contribution", contribution),
        ):
            if not _is_number_or_none(value):
                errors.append(f"positions[{index}].{field} must be numeric or null.")
        if isinstance(daily_return, (int, float)) and abs(float(daily_return)) > 0.30:
            errors.append(
                f"positions[{index}].daily_return={daily_return!r} is not a plausible decimal return; "
                "percentage points must be converted once at export."
            )
        if all(isinstance(value, (int, float)) for value in (nav_weight, daily_return, contribution)):
            expected = float(nav_weight) * float(daily_return)
            if not math.isclose(float(contribution), expected, abs_tol=1e-8):
                errors.append(
                    f"positions[{index}] contribution does not equal nav_weight * daily_return."
                )
        contribution_date = row.get("contribution_effective_date")
        contribution_date_source = row.get("contribution_date_source")
        if contribution_date is not None and not _is_iso_date(contribution_date):
            errors.append(
                f"positions[{index}].contribution_effective_date must be an explicit ISO date or null."
            )
        if (
            contribution_date is not None
            and (
                not isinstance(contribution_date_source, str)
                or not contribution_date_source
                or contribution_date_source == "unavailable"
            )
        ):
            errors.append(
                f"positions[{index}] contribution date requires explicit lineage source."
            )

    sleeve_by_date: dict[str, float] = {}
    for row in contract.get("positions") or []:
        if isinstance(row, dict) and isinstance(row.get("equity_sleeve_weight"), (int, float)):
            date = str(row.get("date") or "")
            sleeve_by_date[date] = sleeve_by_date.get(date, 0.0) + float(row["equity_sleeve_weight"])
    for date, total in sleeve_by_date.items():
        if not math.isclose(total, 1.0, abs_tol=0.001):
            warnings.append(
                f"equity_sleeve_weight sums to {total:.6f} on {date or '-'} instead of 1.0."
            )

    for index, row in enumerate(contract.get("trades") or []):
        if not isinstance(row, dict):
            continue
        fee = row.get("fee")
        fee_source = str(row.get("fee_source") or "")
        if not _is_number_or_none(fee):
            errors.append(f"trades[{index}].fee must be numeric or null.")
        if fee_source == "unknown" and fee is not None:
            errors.append(f"trades[{index}] unknown fee must be null, not zero.")
        if fee is None and fee_source != "unknown":
            warnings.append(f"trades[{index}] null fee should declare fee_source=unknown.")

    as_of = contract.get("as_of_matrix") or {}
    if not isinstance(as_of, dict):
        errors.append("dashboard as_of_matrix must be an object.")
        as_of = {}
    required_as_of_fields = {
        "strategy_record_date",
        "strategy_record_at",
        "analysis_trading_date",
        "quote_at",
        "benchmark_value_dates",
        "theme_date",
        "factor_registry_sha",
    }
    missing_as_of_fields = sorted(required_as_of_fields - set(as_of))
    if missing_as_of_fields:
        errors.append(f"dashboard as_of_matrix missing required fields: {missing_as_of_fields}.")
    for field in ("strategy_record_date", "analysis_trading_date", "theme_date"):
        value = as_of.get(field)
        if value is not None and not _is_iso_date(value):
            errors.append(f"dashboard as_of_matrix.{field} must be an explicit ISO date or null.")
    strategy_record_at = as_of.get("strategy_record_at")
    quote_at = as_of.get("quote_at")
    if strategy_record_at is not None and not _is_iso_timestamp(strategy_record_at):
        errors.append("dashboard as_of_matrix.strategy_record_at must be an explicit zoned ISO timestamp or null.")
    if (
        strategy_record_at is not None
        and _is_iso_timestamp(strategy_record_at)
        and as_of.get("strategy_record_date") != strategy_record_at[:10]
    ):
        errors.append("dashboard strategy_record_date must match explicit strategy_record_at.")
    if quote_at is not None and not _is_iso_timestamp(quote_at):
        errors.append("dashboard as_of_matrix.quote_at must be an explicit zoned ISO timestamp or null.")
    benchmark_value_dates = as_of.get("benchmark_value_dates")
    if not isinstance(benchmark_value_dates, dict):
        errors.append("dashboard as_of_matrix.benchmark_value_dates must be an object.")
        benchmark_value_dates = {}
    for field, value in benchmark_value_dates.items():
        if not _is_iso_date(value):
            errors.append(f"dashboard benchmark value date for {field!r} must be an explicit ISO date.")
    analysis_date = as_of.get("analysis_trading_date")
    if _is_iso_date(analysis_date):
        future_as_of_checks = {
            "as_of_theme_after_analysis_date": as_of.get("theme_date"),
            "as_of_quote_after_analysis_date": (
                quote_at[:10] if _is_iso_timestamp(quote_at) else None
            ),
        }
        for blocker, value in future_as_of_checks.items():
            if _is_iso_date(value) and value > analysis_date and blocker not in (
                blockers or []
            ):
                errors.append(
                    f"future as-of value must declare blocker={blocker}."
                )
        if any(
            _is_iso_date(value) and value > analysis_date
            for value in benchmark_value_dates.values()
        ) and "as_of_benchmark_after_analysis_date" not in (blockers or []):
            errors.append(
                "future benchmark value date must declare "
                "blocker=as_of_benchmark_after_analysis_date."
            )
    if status not in {"sample", "blocked"}:
        missing_as_of = {
            "strategy_record_date": "as_of_strategy_record_missing",
            "analysis_trading_date": "as_of_analysis_trade_date_missing",
            "quote_at": "as_of_quote_missing",
            "theme_date": "as_of_theme_missing",
        }
        for field, blocker in missing_as_of.items():
            if as_of.get(field) is None and blocker not in (blockers or []):
                errors.append(f"missing explicit {field} must declare blocker={blocker}.")
        if not benchmark_value_dates and "as_of_benchmark_value_dates_missing" not in (blockers or []):
            errors.append("missing explicit benchmark value dates must declare blocker=as_of_benchmark_value_dates_missing.")

    calendar = contract.get("trading_calendar") or {}
    if not isinstance(calendar, dict):
        errors.append("dashboard trading_calendar must be an object.")
        calendar = {}
    calendar_status = str(calendar.get("status") or "")
    if calendar_status not in {"available", "missing"}:
        errors.append("dashboard trading_calendar.status must be available or missing.")
    expected_open_dates = calendar.get("expected_open_dates") or []
    if not isinstance(expected_open_dates, list):
        errors.append("dashboard trading_calendar.expected_open_dates must be an array.")
        expected_open_dates = []
    normalized_dates = [str(value) for value in expected_open_dates]
    if normalized_dates != sorted(set(normalized_dates)):
        errors.append("dashboard trading calendar mask must be sorted and unique.")
    if any(not _is_iso_date(value) for value in normalized_dates):
        errors.append("dashboard trading calendar mask contains a non-ISO or invalid date.")
    if calendar_status == "available":
        if calendar.get("source_system") != "strict_parquet.cn_bars.trade_date":
            errors.append("dashboard trading calendar must come from strict Parquet trade_date.")
        try:
            declared_open_count = int(calendar.get("expected_open_date_count") or 0)
        except (TypeError, ValueError):
            declared_open_count = -1
        if declared_open_count != len(normalized_dates):
            errors.append("dashboard trading calendar expected_open_date_count mismatch.")
        expected_first = normalized_dates[0] if normalized_dates else None
        expected_last = normalized_dates[-1] if normalized_dates else None
        if calendar.get("first_open_date") != expected_first:
            errors.append("dashboard trading calendar first_open_date mismatch.")
        if calendar.get("last_open_date") != expected_last:
            errors.append("dashboard trading calendar last_open_date mismatch.")
        mask_payload = {
            "source_system": "strict_parquet.cn_bars.trade_date",
            "start_date": calendar.get("start_date"),
            "end_date": calendar.get("end_date"),
            "expected_open_dates": normalized_dates,
        }
        if calendar.get("mask_sha256") != _canonical_json_sha256(mask_payload):
            errors.append("dashboard trading calendar mask_sha256 mismatch.")
    elif normalized_dates:
        errors.append("unavailable trading calendar cannot contain expected open dates.")
    if policy.get("trading_calendar_required") is not True:
        errors.append("dashboard metric policy must require a formal trading calendar.")
    if calendar_status != "available" and status == "fresh":
        errors.append("dashboard status=fresh requires an available formal trading calendar.")

    provenance = contract.get("nav_return_provenance") or {}
    if not isinstance(provenance, dict):
        errors.append("dashboard nav_return_provenance must be an object.")
        provenance = {}
    if provenance.get("secondary_fee_adjustment_allowed") is not False:
        errors.append("dashboard NAV provenance must prohibit unverified secondary fee adjustment.")
    if provenance.get("gross_or_net") not in {"gross", "net", "unknown"}:
        errors.append("dashboard NAV provenance gross_or_net must be gross/net/unknown.")
    if not isinstance(provenance.get("source_field"), str) or not provenance.get("source_field"):
        errors.append("dashboard NAV provenance source_field must be a non-empty string.")
    if not isinstance(provenance.get("return_method"), str) or not provenance.get("return_method"):
        errors.append("dashboard NAV provenance return_method must be a non-empty string.")

    if provenance.get("trade_fee_inclusion") not in {"included", "excluded", "unknown"}:
        errors.append("dashboard NAV provenance trade_fee_inclusion must be included/excluded/unknown.")
    if (
        provenance.get("gross_or_net") == "unknown"
        or provenance.get("trade_fee_inclusion") == "unknown"
    ) and status not in {"sample", "blocked"} and "nav_fee_provenance_unknown" not in (blockers or []):
        errors.append("unknown NAV fee provenance must declare blocker=nav_fee_provenance_unknown.")

    reconciliation = contract.get("reconciliation") or {}
    if not isinstance(reconciliation, dict):
        errors.append("dashboard reconciliation must be an object.")
        reconciliation = {}
    if not isinstance(reconciliation.get("daily"), list):
        errors.append("dashboard reconciliation.daily must be an array.")
    else:
        if reconciliation.get("status") not in {"reconciled", "partial"}:
            errors.append("dashboard reconciliation.status must be reconciled or partial.")
        raw_daily_rows = [row for row in reconciliation.get("daily") or [] if isinstance(row, dict)]
        daily_dates = [str(row.get("date") or "") for row in raw_daily_rows]
        if len(daily_dates) != len(set(daily_dates)):
            errors.append("dashboard reconciliation.daily contains duplicate dates.")
        daily_rows = {str(row.get("date") or ""): row for row in raw_daily_rows}
        all_nav_return_dates = {
            str(row.get("date") or "")
            for row in contract.get("nav") or []
            if isinstance(row, dict)
            and _is_number_or_none(row.get("portfolio_return"))
            and row.get("portfolio_return") is not None
        }
        allowed_open_dates = set(normalized_dates)
        nav_by_date = {
            str(row.get("date") or ""): row
            for row in contract.get("nav") or []
            if isinstance(row, dict) and _is_number_or_none(row.get("portfolio_return"))
            and row.get("portfolio_return") is not None
            and calendar_status == "available"
            and str(row.get("date") or "") in allowed_open_dates
        }
        excluded_nav_return_dates = sorted(
            all_nav_return_dates - set(nav_by_date)
        )
        position_sums: dict[str, float] = {}
        position_counts: dict[str, int] = {}
        sleeve_sums: dict[str, float] = {}
        sleeve_counts: dict[str, int] = {}
        sleeve_missing: dict[str, int] = {}
        total_position_counts: dict[str, int] = {}
        invalid_position_lineage: dict[str, int] = {}
        missing_effective_date_count = 0
        excluded_position_effective_dates: set[str] = set()
        for row in contract.get("positions") or []:
            if not isinstance(row, dict):
                continue
            raw_effective_date = row.get("contribution_effective_date")
            date = str(raw_effective_date or "") if _is_iso_date(
                raw_effective_date
            ) else ""
            if not date:
                missing_effective_date_count += 1
            elif calendar_status != "available" or date not in allowed_open_dates:
                excluded_position_effective_dates.add(date)
            total_position_counts[date] = total_position_counts.get(date, 0) + 1
            numeric_lineage = all(
                isinstance(row.get(field), (int, float))
                and not isinstance(row.get(field), bool)
                and math.isfinite(float(row[field]))
                for field in ("nav_weight", "daily_return", "contribution")
            )
            contribution_matches = bool(
                numeric_lineage
                and math.isclose(
                    float(row["contribution"]),
                    float(row["nav_weight"]) * float(row["daily_return"]),
                    abs_tol=1e-8,
                )
            )
            if (
                not date
                or not str(row.get("ticker") or "").strip()
                or str(row.get("contribution_date_source") or "").strip()
                in {"", "unavailable"}
                or not numeric_lineage
                or not contribution_matches
            ):
                invalid_position_lineage[date] = (
                    invalid_position_lineage.get(date, 0) + 1
                )
            sleeve_weight = row.get("equity_sleeve_weight")
            if isinstance(sleeve_weight, (int, float)) and math.isfinite(
                float(sleeve_weight)
            ):
                sleeve_sums[date] = sleeve_sums.get(date, 0.0) + float(
                    sleeve_weight
                )
                sleeve_counts[date] = sleeve_counts.get(date, 0) + 1
            else:
                sleeve_missing[date] = sleeve_missing.get(date, 0) + 1
            contribution = row.get("contribution")
            if isinstance(contribution, (int, float)) and math.isfinite(
                float(contribution)
            ):
                position_sums[date] = position_sums.get(date, 0.0) + float(
                    contribution
                )
                position_counts[date] = position_counts.get(date, 0) + 1
        covered = 0
        reconciled_count = 0
        try:
            tolerance = float(reconciliation.get("tolerance") or 0.0001)
        except (TypeError, ValueError):
            errors.append("dashboard reconciliation tolerance must be numeric.")
            tolerance = 0.0001
        for date, nav_row in nav_by_date.items():
            audit = daily_rows.get(date)
            if audit is None:
                errors.append(f"attribution audit missing valid NAV return date {date}.")
                continue
            position_value = position_sums.get(date) if position_counts.get(date) else None
            aggregate_residual = nav_row.get("explicit_cash_fee_residual")
            if isinstance(aggregate_residual, (int, float)) and math.isfinite(float(aggregate_residual)):
                explicit_value = float(aggregate_residual)
            else:
                explicit_components = [
                    float(value)
                    for value in (
                        nav_row.get("cash_return_contribution"),
                        nav_row.get("fee_return_contribution"),
                    )
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                ]
                explicit_value = sum(explicit_components) if explicit_components else None
            sleeve_sum = sleeve_sums.get(date) if sleeve_counts.get(date) else None
            position_snapshot_complete = bool(
                position_value is not None
                and position_counts.get(date, 0)
                == total_position_counts.get(date, 0)
                and invalid_position_lineage.get(date, 0) == 0
                and sleeve_missing.get(date, 0) == 0
                and sleeve_sum is not None
                and math.isclose(sleeve_sum, 1.0, abs_tol=0.001)
            )
            is_covered = position_snapshot_complete
            if is_covered:
                covered += 1
                expected_residual = (
                    float(nav_row["portfolio_return"])
                    - (position_value or 0.0)
                    - (explicit_value or 0.0)
                )
                if abs(expected_residual) <= tolerance:
                    reconciled_count += 1
            else:
                expected_residual = None
            audit_return = audit.get("portfolio_return")
            if not isinstance(audit_return, (int, float)) or not math.isclose(
                float(audit_return), float(nav_row["portfolio_return"]), abs_tol=1e-12
            ):
                errors.append(f"attribution portfolio_return mismatch on {date}.")
            for label, actual, expected in (
                ("position_contribution", audit.get("position_contribution"), position_value),
                ("explicit_cash_fee_residual", audit.get("explicit_cash_fee_residual"), explicit_value),
                ("unexplained_residual", audit.get("unexplained_residual"), expected_residual),
            ):
                if actual is None and expected is None:
                    continue
                if not isinstance(actual, (int, float)) or expected is None or not math.isclose(
                    float(actual), float(expected), abs_tol=1e-10
                ):
                    errors.append(f"attribution {label} mismatch on {date}.")
            if audit.get("covered") is not is_covered:
                errors.append(f"attribution covered flag mismatch on {date}.")
            if audit.get("position_snapshot_complete") is not position_snapshot_complete:
                errors.append(
                    f"attribution position snapshot completeness mismatch on {date}."
                )
            for label, actual, expected in (
                (
                    "position_observation_count",
                    audit.get("position_observation_count"),
                    position_counts.get(date, 0),
                ),
                (
                    "total_position_count",
                    audit.get("total_position_count"),
                    total_position_counts.get(date, 0),
                ),
            ):
                if (
                    not isinstance(actual, int)
                    or isinstance(actual, bool)
                    or actual != expected
                ):
                    errors.append(f"attribution {label} mismatch on {date}.")
            audit_sleeve_sum = audit.get("equity_sleeve_weight_sum")
            if (
                audit_sleeve_sum is None
                and sleeve_sum is not None
            ) or (
                audit_sleeve_sum is not None
                and (
                    not isinstance(audit_sleeve_sum, (int, float))
                    or isinstance(audit_sleeve_sum, bool)
                    or sleeve_sum is None
                    or not math.isclose(
                        float(audit_sleeve_sum),
                        float(sleeve_sum),
                        abs_tol=1e-10,
                    )
                )
            ):
                errors.append(
                    f"attribution equity_sleeve_weight_sum mismatch on {date}."
                )
            expected_within = abs(expected_residual) <= tolerance if expected_residual is not None else None
            if audit.get("within_1bp") is not expected_within:
                errors.append(f"attribution within_1bp mismatch on {date}.")
        extra_daily_dates = sorted(set(daily_rows) - set(nav_by_date))
        if extra_daily_dates:
            errors.append(f"attribution audit contains dates without valid NAV returns: {extra_daily_dates}.")
        valid_count = len(nav_by_date)
        coverage_ratio = covered / valid_count if valid_count else 0.0
        if int(reconciliation.get("valid_nav_return_days") or 0) != valid_count:
            errors.append("attribution valid_nav_return_days mismatch.")
        if int(reconciliation.get("covered_days") or 0) != covered:
            errors.append("attribution covered_days mismatch.")
        try:
            declared_coverage = float(reconciliation.get("coverage_ratio") or 0.0)
        except (TypeError, ValueError):
            declared_coverage = -1.0
        if not math.isclose(declared_coverage, coverage_ratio, abs_tol=1e-12):
            errors.append("attribution coverage_ratio mismatch.")
        if int(reconciliation.get("reconciled_days") or 0) != reconciled_count:
            errors.append("attribution reconciled_days mismatch.")
        position_dates_without_nav_return = sorted(
            date
            for date in total_position_counts
            if date
            and date in allowed_open_dates
            and date not in nav_by_date
        )
        expected_reconciliation_blockers: list[str] = []
        if calendar_status != "available":
            expected_reconciliation_blockers.append(
                "attribution_formal_trading_calendar_missing"
            )
        if excluded_nav_return_dates or excluded_position_effective_dates:
            expected_reconciliation_blockers.append(
                "attribution_non_open_dates_excluded"
            )
        if missing_effective_date_count:
            expected_reconciliation_blockers.append(
                "attribution_position_effective_date_missing"
            )
        if position_dates_without_nav_return:
            expected_reconciliation_blockers.append(
                "attribution_position_date_without_nav_return"
            )
        if reconciliation.get("coverage_basis") != (
            "strict_parquet_trade_date_mask"
        ):
            errors.append(
                "attribution coverage_basis must use strict Parquet trade dates."
            )
        if reconciliation.get("calendar_status") != calendar_status:
            errors.append("attribution calendar_status mismatch.")
        if list(reconciliation.get("blockers") or []) != (
            expected_reconciliation_blockers
        ):
            errors.append("attribution blockers mismatch.")
        for blocker in expected_reconciliation_blockers:
            if blocker not in (blockers or []):
                errors.append(
                    f"attribution blocker must be promoted to contract blockers: {blocker}."
                )
        diagnostics = reconciliation.get("diagnostics")
        if not isinstance(diagnostics, dict):
            errors.append("attribution diagnostics must be an object.")
            diagnostics = {}
        expected_diagnostics = {
            "excluded_nav_return_dates": excluded_nav_return_dates,
            "excluded_position_effective_dates": sorted(
                excluded_position_effective_dates
            ),
            "positions_missing_effective_date_count": missing_effective_date_count,
            "position_dates_without_nav_return": position_dates_without_nav_return,
        }
        if diagnostics != expected_diagnostics:
            errors.append("attribution diagnostics mismatch.")
        fully_reconciled = valid_count > 0 and covered == valid_count and reconciled_count == valid_count
        if reconciliation.get("status") == "reconciled" and not fully_reconciled:
            errors.append(
                f"attribution coverage is {coverage_ratio:.2%}; low-coverage data cannot be reconciled."
            )
    if policy.get("returns_unit") != "decimal":
        errors.append("dashboard metric_policy.returns_unit must be decimal.")
    if policy.get("excess_curve") != "relative_wealth_ratio":
        errors.append("dashboard excess curve policy must use relative_wealth_ratio.")
    if policy.get("monthly_return") != "previous_month_end_anchor":
        errors.append("dashboard monthly return policy must use previous_month_end_anchor.")
    try:
        rolling_coverage = float(policy.get("rolling_window_min_open_day_coverage") or 0.0)
    except (TypeError, ValueError):
        rolling_coverage = -1.0
    if rolling_coverage < 0.95:
        errors.append("dashboard rolling windows require at least 95% formal open-day coverage.")
    return errors, warnings


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
    summary_is_v2 = summary.get("schema_version") == DASHBOARD_SCHEMA_VERSION
    real_private_bundle = (
        summary_is_v2 and summary.get("status") != "sample"
    ) or generated_js.parent.name == "private"
    if real_private_bundle:
        generated_private_paths = [
            summary_file,
            summary_file.with_name("nav_records.csv"),
            summary_file.with_name("positions_records.csv"),
            summary_file.with_name("trades_records.csv"),
            summary_file.with_name("benchmark_records.csv"),
        ]
        for artifact_path in generated_private_paths:
            if not artifact_path.exists():
                errors.append(
                    f"private dashboard generated artifact missing: {artifact_path.name}"
                )
                continue
            if artifact_path.is_symlink() or not artifact_path.is_file():
                errors.append(
                    f"private dashboard generated artifact must be regular: {artifact_path.name}"
                )
                continue
            permissions = artifact_path.stat().st_mode & 0o777
            if permissions != 0o600:
                errors.append(
                    "private dashboard generated artifact permissions must be 0600: "
                    f"{artifact_path.name} mode={permissions:04o}"
                )
    try:
        generated = parse_generated_records(generated_js)
    except DashboardExportCheckError as exc:
        errors.append(str(exc))
        generated = None
    private_json_contract: dict[str, Any] | None = None
    if generated_js.parent.name == "private":
        for private_path in (
            generated_js,
            generated_js.with_name("dashboard_snapshot.v2.json"),
        ):
            if not private_path.exists():
                errors.append(f"private dashboard snapshot missing: {private_path.name}")
                continue
            if private_path.is_symlink() or not private_path.is_file():
                errors.append(
                    f"private dashboard snapshot must be a regular file: {private_path.name}"
                )
                continue
            permissions = private_path.stat().st_mode & 0o777
            if permissions != 0o600:
                errors.append(
                    "private dashboard snapshot permissions must be 0600: "
                    f"{private_path.name} mode={permissions:04o}"
                )
        private_json_path = generated_js.with_name("dashboard_snapshot.v2.json")
        if private_json_path.is_file():
            try:
                loaded_private_json = json.loads(
                    private_json_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                errors.append(
                    f"private dashboard snapshot JSON is invalid: {exc}"
                )
            else:
                if isinstance(loaded_private_json, dict):
                    private_json_contract = loaded_private_json
                else:
                    errors.append(
                        "private dashboard snapshot JSON must be an object."
                    )
        if (
            generated is not None
            and generated.contract is not None
            and private_json_contract is not None
            and generated.contract != private_json_contract
        ):
            errors.append("private dashboard JSON/JS contract payload mismatch.")

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
        contract = generated.contract if generated is not None else None
        summary_is_v2 = summary.get("schema_version") == DASHBOARD_SCHEMA_VERSION
        if summary_is_v2 and contract is None:
            errors.append("export_summary.json declares dashboard_contract.v2 but snapshot JS has no contract.")
        if contract is not None:
            contract_errors, contract_warnings = validate_dashboard_contract_v2(contract)
            errors.extend(contract_errors)
            warnings.extend(contract_warnings)
            for field in (
                "schema_version",
                "schema_sha256",
                "protocol_hash",
                "run_id",
                "status",
                "blockers",
                "as_of_matrix",
                "trading_calendar",
                "nav_return_provenance",
                "reconciliation",
                "theme_protocol",
                "factor_protocol",
            ):
                if field in summary and summary.get(field) != contract.get(field):
                    errors.append(
                        f"{field} mismatch between export_summary.json and dashboard contract v2."
                    )
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
                manifest_status = str(
                    ledger_status.get("manifest_status") or ""
                ).strip()
                if (
                    not manifest_status
                    or manifest_status.lower() == "ok"
                    or "invalidated_price_basis_no_execution"
                    in manifest_status
                ):
                    errors.append(
                        "effective manual baseline has invalid manifest_status."
                    )
                ledger_path = str(ledger_status.get("ledger_path") or "")
                manifest_path = str(ledger_status.get("manifest_path") or "")
                if not ledger_path.endswith(
                    ("/ledger_after_manual_switch.csv", "/ledger_after_manual_switch.parquet")
                ):
                    errors.append(
                        "effective manual ledger path is not a canonical ledger_after_manual_switch sidecar."
                    )
                if not manifest_path.endswith("/manual_execution_manifest.json"):
                    errors.append("effective manual ledger is missing its manual_execution_manifest.json lineage.")
                if not _is_iso_timestamp(
                    ledger_status.get("manifest_recorded_at")
                ):
                    errors.append(
                        "effective manual baseline requires explicit manifest_recorded_at."
                    )
                ledger_sha = str(ledger_status.get("ledger_sha256") or "")
                if not re.fullmatch(r"[0-9a-f]{64}", ledger_sha):
                    errors.append(
                        "effective manual baseline requires ledger_sha256."
                    )
                contract_ledger_sha = str(
                    ((contract or {}).get("sources") or {})
                    .get("ledger", {})
                    .get("sha256")
                    or ""
                )
                if ledger_sha and ledger_sha != contract_ledger_sha:
                    errors.append(
                        "effective manual ledger SHA mismatch with contract source readback."
                    )
                if ledger_status.get("ledger_readback_verified") is not True:
                    errors.append(
                        "effective manual ledger readback must be verified."
                    )
                ledger_sha_declared = ledger_status.get(
                    "ledger_sha_declared"
                )
                if not isinstance(ledger_sha_declared, bool):
                    errors.append(
                        "effective manual baseline must disclose ledger_sha_declared."
                    )
                elif (
                    ledger_sha_declared is False
                    and "manual_ledger_sha_not_declared"
                    not in ((contract or {}).get("blockers") or [])
                ):
                    errors.append(
                        "undeclared manual ledger SHA must block the Dashboard contract."
                    )
                manifest_sha = str(
                    ledger_status.get("manifest_sha256") or ""
                )
                if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
                    errors.append(
                        "effective manual baseline requires manifest_sha256."
                    )
                contract_manifest_sha = str(
                    ((contract or {}).get("sources") or {})
                    .get("manual_manifest", {})
                    .get("sha256")
                    or ""
                )
                if manifest_sha and manifest_sha != contract_manifest_sha:
                    errors.append(
                        "effective manual manifest SHA mismatch with contract source readback."
                    )
                if ledger_status.get("manifest_readback_verified") is not True:
                    errors.append(
                        "effective manual manifest readback must be verified."
                    )

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
        "dashboard_contract": generated.contract if generated is not None else None,
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
    generated_js = args.generated_js or args.dashboard_root / "private" / "dashboard_snapshot.v2.js"
    if args.generated_js is None and not generated_js.exists():
        legacy_generated_js = args.dashboard_root / "js" / "generated_records.js"
        if legacy_generated_js.exists():
            generated_js = legacy_generated_js

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
