#!/usr/bin/env python3
"""Export checked-input evidence for the CN weekly portfolio review.

The exporter is offline and deterministic.  It writes only beneath an explicit
``/private/tmp/myquant-cn/<run-id>`` directory, never mutates the Strategy
Record Store, and never appends the decision log.  Thread, briefing, and web
research results arrive as bounded untrusted JSON inputs; they cannot provide
holdings, fills, NAV, benchmark, V17 authority, or executable actions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.portfolio_cycle.identity import resolve_strategy_identity  # noqa: E402
from quant_investor.strategy_records.performance import (  # noqa: E402
    assert_private_tmp,
    immutable_write,
    load_performance_history,
)
from quant_investor.strategy_records.store import (  # noqa: E402
    CATALOG_SCHEMA_V3,
    StrategyRecordStoreError,
    canonical_json_bytes,
    load_registered_catalog,
    regular_file_sha256,
)
from quant_investor.v17_mainline import derive_mainline_state  # noqa: E402
from quant_investor.v17_mainline.contracts import (  # noqa: E402
    parse_canonical as parse_v17_canonical,
    validate_formal_output,
    validate_ref as validate_v17_ref,
    validate_semantic as validate_v17_semantic,
)
from quant_investor.v17_mainline.storage import MainlineStore  # noqa: E402
from scripts.cn_dashboard_common import (  # noqa: E402
    DashboardInputError,
    _read_benchmark_rows,
    build_bundle as build_dashboard_bundle,
    stable_read as read_dashboard_artifact,
)
from scripts.log_decision import (  # noqa: E402
    DecisionLogError,
    make_event as make_decision_event,
    read_events as read_decision_events,
)


SCHEMA_ID = "cn_weekly_portfolio_evidence.v1"
HISTORICAL_LABEL = "aggressive_tech_manufacturing"
CANONICAL_STRATEGY_ID = "cn-aggressive-tech-manufacturing"
IDENTITY_PATH = (
    "results/portfolio_cycle/CN/cn-aggressive-tech-manufacturing/"
    "governance/strategy_identity.v1.json"
)
RECORD_ROOT = Path(
    "results/strategy_records/CN/aggressive_tech_manufacturing"
)
BENCHMARK_PATH = Path("portfolio_dashboard/inputs/cn_index_benchmark.csv")
RISK_FREE_PATH = Path("portfolio_dashboard/inputs/cn_govt_bond_yield.csv")
MARKET_CALENDAR_ROOT = Path("data/parquet/cn/macro_release_calendar")
MARKET_CALENDAR_POINTER = MARKET_CALENDAR_ROOT / "_latest.json"
DECISION_LOG_PATH = Path("results/decision_log/decision_log.jsonl")
FORMAL_WEEKLY_ADVISORY_SCHEMA = "myquant.v17.v4.weekly-advisory-input.v1"
FORMAL_GATE_NAMES = (
    "identity",
    "holdings",
    "factor",
    "market_pit",
    "fundamental",
    "risk",
    "portfolio_i6",
    "action_freshness",
)
DOMAIN_NAMES = (
    "STORE_HOLDINGS",
    "WEEKLY_OPERATIONS",
    "PERFORMANCE_BENCHMARK",
    "DAILY_REVIEW_COVERAGE",
    "MARKET_BRIEFING_COVERAGE",
    "PUBLIC_WEB_RESEARCH",
    "FORMAL_V17_ADVISORY",
    "DECISION_LOG",
    "QA",
)
DOMAIN_STATUSES = {
    "FRESH",
    "PARTIAL",
    "BLOCKED",
    "DEPENDENCY_BLOCKED",
    "NOT_APPLICABLE",
}
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_GENERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_CANONICAL_UTC = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
SHANGHAI = ZoneInfo("Asia/Shanghai")


class WeeklyEvidenceError(RuntimeError):
    """Stable fail-closed weekly evidence error."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _content_sha(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    return _sha(canonical_json_bytes(body))


def _parse_utc(value: str, *, label: str) -> datetime:
    if not isinstance(value, str) or _CANONICAL_UTC.fullmatch(value) is None:
        raise WeeklyEvidenceError(f"{label} is not a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise WeeklyEvidenceError(f"{label} is not a canonical UTC timestamp") from exc
    return parsed


def report_window(scheduled_at: str) -> dict[str, Any]:
    scheduled_utc = _parse_utc(scheduled_at, label="scheduled_at")
    local = scheduled_utc.astimezone(SHANGHAI)
    if (
        local.weekday() != 6
        or local.hour != 18
        or local.minute != 0
        or local.second != 0
        or local.microsecond != 0
    ):
        raise WeeklyEvidenceError("REPORT_WINDOW_UNBOUND")
    monday = local.date() - timedelta(days=local.weekday())
    start = datetime.combine(monday, time.min, tzinfo=SHANGHAI)
    outlook_start = datetime.combine(monday + timedelta(days=7), time.min, tzinfo=SHANGHAI)
    outlook_end = outlook_start + timedelta(days=7)
    iso = local.isocalendar()
    return {
        "report_week": f"{iso.year:04d}-W{iso.week:02d}",
        "scheduled_at": scheduled_at,
        "start_at": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_at": scheduled_at,
        "start_date": monday.isoformat(),
        "end_date": local.date().isoformat(),
        "outlook_start_at": outlook_start.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "outlook_end_at": outlook_end.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "timezone": "Asia/Shanghai",
        "interval_semantics": "start_inclusive_end_exclusive",
    }


def _domain(
    status: str,
    *,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in DOMAIN_STATUSES:
        raise WeeklyEvidenceError("domain status is invalid")
    return {
        "status": status,
        "blockers": blockers or [],
        "warnings": warnings or [],
        "evidence": evidence or {},
    }


def _safe_json_input(
    path: Path | None,
    *,
    expected_schema: str,
    report_week: str,
    label: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise WeeklyEvidenceError(f"{label} input is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > MAX_INPUT_BYTES
    ):
        raise WeeklyEvidenceError(f"{label} input storage is unsafe")
    digest, size = regular_file_sha256(path, label=f"{label} input")
    raw = path.read_bytes()
    if len(raw) != size or _sha(raw) != digest:
        raise WeeklyEvidenceError(f"{label} input changed during read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeeklyEvidenceError(f"{label} input is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_id") != expected_schema
        or value.get("report_week") != report_week
    ):
        raise WeeklyEvidenceError(f"{label} input contract mismatch")
    return value, {"path": str(path), "sha256": digest, "bytes": size}


def _benchmark_trade_dates(
    path: Path, *, start_date: str, end_date: str
) -> tuple[list[str], dict[str, Any]]:
    artifact = read_dashboard_artifact(path, PROJECT_ROOT)
    rows = _read_benchmark_rows(artifact, "000300.SH")
    dates = {
        value
        for value, row in rows.items()
        if start_date <= value <= end_date and row["coverage"] == "exact_close"
    }
    return sorted(dates), {
        "path": artifact.relative_path,
        "sha256": artifact.sha256,
        "bytes": len(artifact.data),
        "ts_code": "000300.SH",
    }


def _registered_cn_trade_dates(
    project_root: Path, *, start_date: str, end_date: str
) -> tuple[list[str], list[dict[str, Any]]]:
    pointer_path = project_root / MARKET_CALENDAR_POINTER
    pointer_sha, pointer_size = regular_file_sha256(
        pointer_path, label="registered market calendar pointer"
    )
    pointer = _read_exact_json(
        pointer_path, pointer_sha, label="registered market calendar pointer"
    )
    generation_id = pointer.get("generation_id")
    manifest_sha = pointer.get("manifest_sha256")
    if (
        pointer.get("schema_version") != "macro-release-calendar-pointer.v1"
        or not isinstance(generation_id, str)
        or _GENERATION_ID.fullmatch(generation_id) is None
        or not isinstance(manifest_sha, str)
        or _SHA256.fullmatch(manifest_sha) is None
    ):
        raise WeeklyEvidenceError("registered market calendar pointer is invalid")
    generation_root = project_root / MARKET_CALENDAR_ROOT / "_generations" / generation_id
    manifest_path = generation_root / "manifest.json"
    manifest = _read_exact_json(
        manifest_path, manifest_sha, label="registered market calendar manifest"
    )
    if (
        manifest.get("schema_version") != "macro-release-calendar-generation.v1"
        or manifest.get("generation_id") != generation_id
    ):
        raise WeeklyEvidenceError("registered market calendar manifest is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise WeeklyEvidenceError("registered market calendar artifacts are invalid")
    open_day_refs = [
        row
        for row in artifacts
        if isinstance(row, dict) and row.get("path") == "market_open_days.json"
    ]
    if len(open_day_refs) != 1:
        raise WeeklyEvidenceError("registered market open-days binding is missing")
    open_day_ref = open_day_refs[0]
    open_day_sha = open_day_ref.get("sha256")
    if not isinstance(open_day_sha, str) or _SHA256.fullmatch(open_day_sha) is None:
        raise WeeklyEvidenceError("registered market open-days SHA is invalid")
    open_day_path = generation_root / "market_open_days.json"
    open_days = _read_exact_json(
        open_day_path, open_day_sha, label="registered market open days"
    )
    raw_dates = open_days.get("open_dates")
    if (
        open_days.get("schema_version") != "market-open-days.v1"
        or open_days.get("market") != "CN"
        or not isinstance(raw_dates, list)
        or raw_dates != sorted(set(raw_dates))
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"\d{8}", value) is None
            for value in raw_dates
        )
    ):
        raise WeeklyEvidenceError("registered market open days are invalid")
    dates = [
        f"{value[:4]}-{value[4:6]}-{value[6:]}"
        for value in raw_dates
        if start_date <= f"{value[:4]}-{value[4:6]}-{value[6:]}" <= end_date
    ]
    readback_sha, readback_size = regular_file_sha256(
        pointer_path, label="registered market calendar pointer"
    )
    if readback_sha != pointer_sha or readback_size != pointer_size:
        raise WeeklyEvidenceError("registered market calendar pointer drifted")
    return dates, [
        {
            "path": MARKET_CALENDAR_POINTER.as_posix(),
            "sha256": pointer_sha,
            "bytes": pointer_size,
        },
        {
            "path": manifest_path.relative_to(project_root).as_posix(),
            "sha256": manifest_sha,
        },
        {
            "path": open_day_path.relative_to(project_root).as_posix(),
            "sha256": open_day_sha,
        },
    ]


def _daily_review_domain(
    value: dict[str, Any] | None,
    source_ref: dict[str, Any] | None,
    *,
    window: dict[str, Any],
    expected_trade_dates: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if value is None:
        return _domain(
            "PARTIAL", blockers=["DAILY_REVIEW_INPUT_MISSING"]
        ), []
    rows = value.get("items")
    if not isinstance(rows, list) or len(rows) > 10:
        raise WeeklyEvidenceError("daily review items are invalid")
    start = _parse_utc(window["start_at"], label="window start")
    end = _parse_utc(window["end_at"], label="window end")
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    covered_dates: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise WeeklyEvidenceError("daily review item is invalid")
        if (
            raw.get("title") != "A股量化投资与日度复盘"
            or raw.get("automation_id") != "automation"
        ):
            continue
        last_run = raw.get("last_run")
        if not isinstance(last_run, str):
            raise WeeklyEvidenceError("daily review last_run is invalid")
        observed = _parse_utc(last_run, label="daily review last_run")
        if not start <= observed < end:
            continue
        trade_date = raw.get("trade_date")
        if trade_date not in expected_trade_dates:
            continue
        key = ("automation", last_run)
        selected[key] = dict(raw)
        covered_dates.add(str(trade_date))
    missing = sorted(set(expected_trade_dates) - covered_dates)
    status = "FRESH" if not missing else "PARTIAL"
    return _domain(
        status,
        blockers=["DAILY_REVIEW_EXPECTED_TRADING_DAY_MISSING"] if missing else [],
        warnings=["missing_trade_dates:" + ",".join(missing)] if missing else [],
        evidence={"source_ref": source_ref, "covered_trade_dates": sorted(covered_dates)},
    ), [selected[key] for key in sorted(selected)]


def _briefing_domain(
    value: dict[str, Any] | None,
    source_ref: dict[str, Any] | None,
    *,
    expected_trade_dates: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if value is None:
        return _domain(
            "PARTIAL", blockers=["MARKET_BRIEFING_INPUT_MISSING"]
        ), []
    rows = value.get("items")
    if not isinstance(rows, list) or len(rows) > 20:
        raise WeeklyEvidenceError("market briefing items are invalid")
    by_date: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise WeeklyEvidenceError("market briefing item is invalid")
        briefing_date = raw.get("briefing_date")
        if briefing_date not in expected_trade_dates:
            continue
        # Input ordering is the explicit revision order; the final version for
        # a date wins.  Narrative content is retained only as untrusted text.
        by_date[str(briefing_date)] = dict(raw)
    missing = sorted(set(expected_trade_dates) - set(by_date))
    status = "FRESH" if not missing else "PARTIAL"
    return _domain(
        status,
        blockers=["MARKET_BRIEFING_EXPECTED_DATE_MISSING"] if missing else [],
        warnings=["missing_briefing_dates:" + ",".join(missing)] if missing else [],
        evidence={"source_ref": source_ref, "covered_dates": sorted(by_date)},
    ), [by_date[key] for key in sorted(by_date)]


def _web_domain(
    value: dict[str, Any] | None,
    source_ref: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if value is None:
        return _domain(
            "PARTIAL", blockers=["PUBLIC_WEB_RESEARCH_INPUT_MISSING"]
        ), None
    sources = value.get("sources")
    if (
        value.get("research_completed") is not True
        or not isinstance(sources, list)
        or not 1 <= len(sources) <= 16
    ):
        raise WeeklyEvidenceError("public web research contract is invalid")
    official_count = 0
    for source in sources:
        if not isinstance(source, dict):
            raise WeeklyEvidenceError("public web source is invalid")
        url = source.get("url")
        if not isinstance(url, str) or urlparse(url).scheme != "https":
            raise WeeklyEvidenceError("public web source URL is invalid")
        if not isinstance(source.get("published_or_event_date"), str):
            raise WeeklyEvidenceError("public web source date is missing")
        if source.get("source_class") in {
            "CHINA_OFFICIAL",
            "OVERSEAS_OFFICIAL",
            "EXCHANGE_OR_COMPANY_OFFICIAL",
        }:
            official_count += 1
    if official_count == 0:
        return _domain(
            "PARTIAL",
            blockers=["PUBLIC_WEB_OFFICIAL_SOURCE_MISSING"],
            evidence={"source_ref": source_ref, "source_count": len(sources)},
        ), value
    return _domain(
        "FRESH",
        evidence={
            "source_ref": source_ref,
            "source_count": len(sources),
            "official_source_count": official_count,
        },
    ), value


def _read_exact_json(path: Path, expected_sha: str, *, label: str) -> dict[str, Any]:
    digest, size = regular_file_sha256(path, label=label)
    if digest != expected_sha or size > MAX_INPUT_BYTES:
        raise WeeklyEvidenceError(f"{label} exact bytes mismatch")
    raw = path.read_bytes()
    if _sha(raw) != digest:
        raise WeeklyEvidenceError(f"{label} changed during read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeeklyEvidenceError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise WeeklyEvidenceError(f"{label} is not an object")
    return value


def _v17_ref_key(value: dict[str, str]) -> tuple[str, str, str]:
    return (
        value["schema_id"],
        value["relative_path"],
        value["byte_sha256"],
    )


def _bounded_text(value: Any, *, label: str, maximum: int = 4000) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise WeeklyEvidenceError(f"{label} is not bounded canonical text")
    return value


def _validate_formal_action(
    raw: Any,
    *,
    holdings_by_symbol: dict[str, dict[str, Any]],
    formal_ref_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    required = {
        "symbol",
        "company_name",
        "action",
        "shares_delta",
        "validity",
        "invalidation",
        "evidence_refs",
        "executable",
    }
    optional = {"target_weight", "risk_reduction_required"}
    if (
        not isinstance(raw, dict)
        or not required.issubset(raw)
        or not set(raw).issubset(required | optional)
    ):
        raise WeeklyEvidenceError("formal advisory action fields are invalid")
    action = dict(raw)
    symbol = action.get("symbol")
    if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
        raise WeeklyEvidenceError("formal advisory symbol is invalid")
    company_name = _bounded_text(
        action.get("company_name"), label="formal advisory company name", maximum=200
    )
    direction = action.get("action")
    if direction not in {"BUY", "ADD", "REDUCE", "EXIT", "HOLD", "WATCH"}:
        raise WeeklyEvidenceError("formal advisory action is invalid")
    shares = action.get("shares_delta")
    if not isinstance(shares, int) or isinstance(shares, bool):
        raise WeeklyEvidenceError("formal advisory shares_delta is not an integer")
    if direction in {"BUY", "ADD"} and shares <= 0:
        raise WeeklyEvidenceError("formal BUY/ADD shares_delta is not positive")
    if direction in {"REDUCE", "EXIT"} and shares >= 0:
        raise WeeklyEvidenceError("formal REDUCE/EXIT shares_delta is not negative")
    if direction in {"HOLD", "WATCH"} and shares != 0:
        raise WeeklyEvidenceError("formal HOLD/WATCH shares_delta is not zero")
    current = holdings_by_symbol.get(symbol)
    if direction == "BUY" and current is not None:
        raise WeeklyEvidenceError("formal BUY conflicts with an existing holding")
    if direction in {"ADD", "REDUCE", "EXIT", "HOLD"} and current is None:
        raise WeeklyEvidenceError("formal holding action has no current holding")
    if current is not None:
        current_shares = int(float(current["shares"]))
        if float(current_shares) != float(current["shares"]):
            raise WeeklyEvidenceError("current holding shares are not integral")
        if company_name != current.get("name"):
            raise WeeklyEvidenceError("formal action company name mismatches holdings")
        if direction == "EXIT" and -shares != current_shares:
            raise WeeklyEvidenceError("formal EXIT does not close the position")
        if direction == "REDUCE" and not 0 < -shares < current_shares:
            raise WeeklyEvidenceError("formal REDUCE quantity is invalid")
    if direction in {"REDUCE", "EXIT"} and action.get("risk_reduction_required") is not True:
        raise WeeklyEvidenceError("formal REDUCE/EXIT lacks an exact risk requirement")
    _bounded_text(action.get("validity"), label="formal advisory validity")
    _bounded_text(action.get("invalidation"), label="formal advisory invalidation")
    refs = action.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise WeeklyEvidenceError("formal advisory evidence refs are absent")
    for ref in refs:
        if (
            not isinstance(ref, dict)
            or set(ref) != {"path", "sha256"}
            or not isinstance(ref.get("path"), str)
            or not isinstance(ref.get("sha256"), str)
            or _SHA256.fullmatch(ref["sha256"]) is None
            or (ref["path"], ref["sha256"]) not in formal_ref_pairs
        ):
            raise WeeklyEvidenceError(
                "formal advisory evidence is not in the active V17 closure"
            )
    if action.get("executable") is not False:
        raise WeeklyEvidenceError("formal advisory action must be non-executable")
    target_weight = action.get("target_weight")
    if target_weight is not None:
        try:
            numeric_target = float(target_weight)
        except (TypeError, ValueError) as exc:
            raise WeeklyEvidenceError("formal target weight is invalid") from exc
        if not 0 <= numeric_target <= 1:
            raise WeeklyEvidenceError("formal target weight is outside [0,1]")
    return action


def _validate_formal_advisory_sidecar(
    value: dict[str, Any],
    *,
    public_run: dict[str, Any],
    formal_evidence_refs: list[dict[str, str]],
    store_evidence: dict[str, Any],
    holdings: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    validate_v17_semantic(value)
    required = {
        "schema_id",
        "protocol",
        "report_week",
        "scheduled_at",
        "canonical_strategy_id",
        "store_binding",
        "portfolio_output_ref",
        "source_closure_ref",
        "gates",
        "formal_outcome",
        "actions",
        "supersedes_event_id",
        "executable",
        "semantic_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_id") != FORMAL_WEEKLY_ADVISORY_SCHEMA
        or value.get("protocol") != "myquant.v17.v4"
        or value.get("report_week") != window["report_week"]
        or value.get("scheduled_at") != window["scheduled_at"]
        or value.get("canonical_strategy_id") != CANONICAL_STRATEGY_ID
        or value.get("executable") is not False
    ):
        raise WeeklyEvidenceError("formal weekly advisory contract mismatch")
    if value.get("portfolio_output_ref") != public_run.get("portfolio_output_ref") or value.get(
        "source_closure_ref"
    ) != public_run.get("source_closure_ref"):
        raise WeeklyEvidenceError("formal advisory V17 closure binding mismatch")
    expected_store = {
        "identity_sha256": store_evidence["identity_sha256"],
        "store_pointer_sha256": store_evidence["pointer_sha256"],
        "catalog_sha256": store_evidence["catalog_sha256"],
        "performance_manifest_sha256": store_evidence["performance_history_ref"][
            "manifest"
        ]["sha256"],
        "financial_state_sha256": store_evidence["active_closure"][
            "financial_state_sha256"
        ],
    }
    if value.get("store_binding") != expected_store:
        raise WeeklyEvidenceError("formal advisory Store binding mismatch")
    normalized_formal_refs = [
        validate_v17_ref(ref, label="formal evidence ref")
        for ref in formal_evidence_refs
    ]
    formal_keys = {_v17_ref_key(ref) for ref in normalized_formal_refs}
    formal_pairs = {(ref["relative_path"], ref["byte_sha256"]) for ref in normalized_formal_refs}
    gates = value.get("gates")
    if not isinstance(gates, dict) or tuple(sorted(gates)) != tuple(sorted(FORMAL_GATE_NAMES)):
        raise WeeklyEvidenceError("formal advisory gates are incomplete")
    for name in FORMAL_GATE_NAMES:
        gate = gates[name]
        if not isinstance(gate, dict) or set(gate) != {"verified", "ref"}:
            raise WeeklyEvidenceError(f"formal advisory gate {name} is invalid")
        if gate.get("verified") is not True:
            raise WeeklyEvidenceError(f"formal advisory gate {name} is not verified")
        normalized_ref = validate_v17_ref(gate.get("ref"), label=f"formal gate {name}")
        if _v17_ref_key(normalized_ref) not in formal_keys:
            raise WeeklyEvidenceError(
                f"formal advisory gate {name} is outside active V17 evidence"
            )
    positions = holdings.get("positions")
    if not isinstance(positions, list):
        raise WeeklyEvidenceError("formal advisory holdings are absent")
    holdings_by_symbol = {
        row["symbol"]: row
        for row in positions
        if isinstance(row, dict) and isinstance(row.get("symbol"), str)
    }
    actions_raw = value.get("actions")
    if not isinstance(actions_raw, list) or len(actions_raw) > 50:
        raise WeeklyEvidenceError("formal advisory actions exceed the bound")
    actions = [
        _validate_formal_action(
            row,
            holdings_by_symbol=holdings_by_symbol,
            formal_ref_pairs=formal_pairs,
        )
        for row in actions_raw
    ]
    ordered = sorted(actions, key=lambda row: (row["symbol"], row["action"]))
    if actions != ordered:
        raise WeeklyEvidenceError("formal advisory actions are not deterministic")
    outcome = value.get("formal_outcome")
    if outcome == "NO_ACTION" and actions:
        raise WeeklyEvidenceError("formal NO_ACTION carries actions")
    if outcome == "ADVISORY" and not actions:
        raise WeeklyEvidenceError("formal ADVISORY has no actions")
    if outcome not in {"ADVISORY", "NO_ACTION"}:
        raise WeeklyEvidenceError("formal advisory outcome is invalid")
    supersedes = value.get("supersedes_event_id")
    if supersedes is not None:
        _bounded_text(supersedes, label="formal supersedes_event_id", maximum=200)
    return {
        "status": outcome,
        "actions": actions,
        "supersedes_event_id": supersedes,
        "executable": False,
    }


def _load_active_formal_sidecar(
    public_run: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, str]]:
    store = MainlineStore(PROJECT_ROOT)
    formal_ref = validate_v17_ref(
        public_run.get("formal_output_ref"), label="active formal output ref"
    )
    formal_stored = store.read(formal_ref["relative_path"], formal_ref["byte_sha256"])
    formal = validate_formal_output(
        parse_v17_canonical(formal_stored.data), strategy_id=CANONICAL_STRATEGY_ID
    )
    evidence = formal.get("evidence_refs")
    if not isinstance(evidence, list):
        raise WeeklyEvidenceError("active formal evidence refs are absent")
    normalized = [validate_v17_ref(ref, label="active formal evidence ref") for ref in evidence]
    for ref in normalized:
        store.read(ref["relative_path"], ref["byte_sha256"])
    sidecars = [ref for ref in normalized if ref["schema_id"] == FORMAL_WEEKLY_ADVISORY_SCHEMA]
    if len(sidecars) != 1:
        raise WeeklyEvidenceError("FORMAL_ACTION_GATES_INCOMPLETE")
    sidecar_ref = sidecars[0]
    sidecar_stored = store.read(
        sidecar_ref["relative_path"], sidecar_ref["byte_sha256"]
    )
    sidecar = parse_v17_canonical(sidecar_stored.data)
    return sidecar, normalized, sidecar_ref


def _build_decision_envelope(
    *,
    formal: dict[str, Any],
    public_run: dict[str, Any],
    store_evidence: dict[str, Any],
    window: dict[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    run_sha = public_run["mainline_run_ref"]["byte_sha256"]
    return make_decision_event(
        {
            "schema_version": "decision_log.v2",
            "event_type": "advisory_envelope",
            "report_group_id": f"myquant-cn:{window['report_week']}",
            "idempotency_key": (
                f"myquant-cn:{window['report_week']}:{CANONICAL_STRATEGY_ID}:{run_sha}"
            ),
            "report_week": window["report_week"],
            "scheduled_at": window["scheduled_at"],
            "canonical_strategy_id": CANONICAL_STRATEGY_ID,
            "identity_sha256": store_evidence["identity_sha256"],
            "v17_active_run_sha256": run_sha,
            "v17_active_pointer_sha256": public_run["active_pointer_ref"][
                "byte_sha256"
            ],
            "store_pointer_sha256": store_evidence["pointer_sha256"],
            "catalog_sha256": store_evidence["catalog_sha256"],
            "performance_manifest_sha256": store_evidence[
                "performance_history_ref"
            ]["manifest"]["sha256"],
            "financial_state_sha256": store_evidence["active_closure"][
                "financial_state_sha256"
            ],
            "executable": False,
            "formal_outcome": formal["status"],
            "actions": formal["actions"],
            "supersedes_event_id": formal["supersedes_event_id"],
            "recorded_at": recorded_at,
        }
    )


def _decision_log_domain(
    event: dict[str, Any], *, path: Path
) -> tuple[dict[str, Any], bool]:
    try:
        events = read_decision_events(path)
    except (DecisionLogError, OSError) as exc:
        raise WeeklyEvidenceError(f"ADVISORY_LOG_BLOCKED:{exc}") from exc
    same_key = [
        row
        for row in events
        if row.get("schema_version") == "decision_log.v2"
        and row.get("idempotency_key") == event["idempotency_key"]
    ]
    if same_key:
        if same_key[0].get("semantic_sha256") != event["semantic_sha256"]:
            raise WeeklyEvidenceError("DECISION_LOG_IDEMPOTENCY_CONFLICT")
        return _domain(
            "FRESH",
            evidence={
                "event_id": same_key[0]["event_id"],
                "idempotency_key": event["idempotency_key"],
                "already_recorded": True,
            },
        ), True
    same_week = [
        row
        for row in events
        if row.get("schema_version") == "decision_log.v2"
        and row.get("report_week") == event["report_week"]
        and row.get("canonical_strategy_id") == CANONICAL_STRATEGY_ID
    ]
    if same_week and event.get("supersedes_event_id") != same_week[-1].get("event_id"):
        raise WeeklyEvidenceError("weekly V17 supersession is not explicit")
    return _domain(
        "PARTIAL",
        blockers=["ADVISORY_LOG_PENDING"],
        evidence={
            "event_id": event["event_id"],
            "idempotency_key": event["idempotency_key"],
            "already_recorded": False,
        },
    ), False


def _operations(
    *,
    root: Path,
    catalog: dict[str, Any],
    window: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    start_date = window["start_date"]
    end_date = window["end_date"]
    lineage = [
        row
        for row in catalog["lineage_index"]
        if start_date <= row["valuation_date"] <= end_date
    ]
    if any(row["storage_state"] == "ARCHIVED" for row in lineage):
        return _domain(
            "BLOCKED", blockers=["WEEKLY_LINEAGE_MEMBER_ARCHIVED"]
        ), [], []
    records = {row["record_id"]: row for row in catalog["records"]}
    events: list[dict[str, Any]] = []
    non_trade: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in lineage:
        record = records.get(row["record_id"])
        if not isinstance(record, dict):
            raise WeeklyEvidenceError("weekly lineage record is unregistered")
        manual_path = record.get("manual_manifest_path")
        manual_sha = record.get("manual_manifest_sha256")
        if not isinstance(manual_path, str) or not isinstance(manual_sha, str):
            if row["execution_class"] == "UNKNOWN_BLOCKED":
                warnings.append("manual_manifest_unavailable:" + row["record_id"])
                continue
            raise WeeklyEvidenceError("weekly manual manifest binding is absent")
        manual = _read_exact_json(
            root / manual_path, manual_sha, label="weekly manual manifest"
        )
        if row["execution_class"] == "APPLIED_TRADES":
            fills: list[Any] = []
            for key in ("applied_owner_declared_trades", "applied_local_trades"):
                value = manual.get(key, [])
                if not isinstance(value, list):
                    raise WeeklyEvidenceError("weekly fill list is invalid")
                fills.extend(value)
            if not fills:
                raise WeeklyEvidenceError("weekly applied-trade lineage has no exact fills")
            for fill in fills:
                if not isinstance(fill, dict):
                    raise WeeklyEvidenceError("weekly fill is invalid")
                symbol = fill.get("symbol")
                name = fill.get("name") or "UNKNOWN_NAME"
                side = fill.get("side") or fill.get("action") or "UNKNOWN"
                events.append(
                    {
                        "symbol": symbol,
                        "company_name": name,
                        "event_date": str(fill.get("trade_date") or row["valuation_date"]).replace(
                            "-", ""
                        ),
                        "actual_event": str(side).upper(),
                        "shares": fill.get("shares"),
                        "execution_price": fill.get("execution_price"),
                        "fees_cny": fill.get("fees_cny", "UNKNOWN"),
                        "source_record_id": row["record_id"],
                        "manual_manifest_sha256": manual_sha,
                        "decision_log_pairing": "UNPAIRED_OR_LEGACY_PAIRING_ONLY",
                    }
                )
        else:
            non_trade.append(
                {
                    "record_id": row["record_id"],
                    "valuation_date": row["valuation_date"],
                    "execution_class": row["execution_class"],
                    "publication_class": row["publication_class"],
                    "execution_status": manual.get("execution_status"),
                    "official_valuation": manual.get("official_valuation"),
                    "manual_manifest_sha256": manual_sha,
                    "described_as_trade": False,
                }
            )
    events.sort(
        key=lambda row: (
            str(row["event_date"]),
            str(row["source_record_id"]),
            str(row["symbol"]),
            str(row["actual_event"]),
        )
    )
    status = "PARTIAL" if warnings else "FRESH"
    return _domain(
        status,
        warnings=warnings,
        evidence={
            "lineage_index_sha256": catalog["lineage_index_sha256"],
            "active_lineage_record_count": len(lineage),
            "actual_fill_count": len(events),
        },
    ), events, non_trade


def _overall(domains: dict[str, dict[str, Any]]) -> str:
    statuses = {value["status"] for value in domains.values()}
    useful = any(
        domains[name]["status"] in {"FRESH", "PARTIAL"}
        and bool(domains[name].get("evidence"))
        for name in (
            "STORE_HOLDINGS",
            "PERFORMANCE_BENCHMARK",
            "DAILY_REVIEW_COVERAGE",
            "MARKET_BRIEFING_COVERAGE",
            "PUBLIC_WEB_RESEARCH",
        )
    )
    if not useful:
        return "BLOCKED"
    if statuses <= {"FRESH", "NOT_APPLICABLE"}:
        return "FRESH"
    return "PARTIAL"


def export(args: argparse.Namespace) -> dict[str, Any]:
    window = report_window(args.scheduled_at)
    generated_at = _parse_utc(args.generated_at, label="generated_at").strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    output_dir = assert_private_tmp(Path(args.output_dir))
    output_path = output_dir / "cn_weekly_portfolio_evidence.v1.json"
    if output_path.exists():
        raise WeeklyEvidenceError("weekly evidence output already exists")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    _, benchmark_input_ref = _benchmark_trade_dates(
        PROJECT_ROOT / BENCHMARK_PATH,
        start_date=window["start_date"],
        end_date=window["end_date"],
    )
    registered_trade_dates: list[str] = []
    market_calendar_refs: list[dict[str, Any]] = []
    market_calendar_error: str | None = None
    try:
        registered_trade_dates, market_calendar_refs = _registered_cn_trade_dates(
            PROJECT_ROOT,
            start_date=window["start_date"],
            end_date=window["end_date"],
        )
    except (StrategyRecordStoreError, WeeklyEvidenceError, KeyError) as exc:
        market_calendar_error = str(exc)
    daily_input, daily_ref = _safe_json_input(
        Path(args.daily_review_json) if args.daily_review_json else None,
        expected_schema="cn_weekly_daily_review_input.v1",
        report_week=window["report_week"],
        label="daily review",
    )
    briefing_input, briefing_ref = _safe_json_input(
        Path(args.market_briefing_json) if args.market_briefing_json else None,
        expected_schema="cn_weekly_market_briefing_input.v1",
        report_week=window["report_week"],
        label="market briefing",
    )
    web_input, web_ref = _safe_json_input(
        Path(args.public_web_json) if args.public_web_json else None,
        expected_schema="cn_weekly_public_web_research_input.v1",
        report_week=window["report_week"],
        label="public web research",
    )
    if market_calendar_error is None:
        daily_domain, daily_rows = _daily_review_domain(
            daily_input,
            daily_ref,
            window=window,
            expected_trade_dates=registered_trade_dates,
        )
        briefing_domain, briefing_rows = _briefing_domain(
            briefing_input,
            briefing_ref,
            expected_trade_dates=registered_trade_dates,
        )
    else:
        calendar_warning = "registered_calendar:" + market_calendar_error
        daily_domain = _domain(
            "PARTIAL",
            blockers=["REGISTERED_TRADING_CALENDAR_UNAVAILABLE"],
            warnings=[calendar_warning],
            evidence={"source_ref": daily_ref},
        )
        briefing_domain = _domain(
            "PARTIAL",
            blockers=["REGISTERED_TRADING_CALENDAR_UNAVAILABLE"],
            warnings=[calendar_warning],
            evidence={"source_ref": briefing_ref},
        )
        daily_rows = []
        briefing_rows = []
    web_domain, web_payload = _web_domain(web_input, web_ref)

    domains: dict[str, dict[str, Any]] = {
        "STORE_HOLDINGS": _domain("BLOCKED", blockers=["STORE_NOT_VERIFIED"]),
        "WEEKLY_OPERATIONS": _domain("DEPENDENCY_BLOCKED", blockers=["STORE_NOT_VERIFIED"]),
        "PERFORMANCE_BENCHMARK": _domain(
            "DEPENDENCY_BLOCKED", blockers=["STORE_NOT_VERIFIED"]
        ),
        "DAILY_REVIEW_COVERAGE": daily_domain,
        "MARKET_BRIEFING_COVERAGE": briefing_domain,
        "PUBLIC_WEB_RESEARCH": web_domain,
        "FORMAL_V17_ADVISORY": _domain(
            "BLOCKED", blockers=["FORMAL_ADVISORY_BLOCKED"]
        ),
        "DECISION_LOG": _domain(
            "DEPENDENCY_BLOCKED", blockers=["FORMAL_ADVISORY_BLOCKED"]
        ),
        "QA": _domain("FRESH"),
    }
    store_evidence: dict[str, Any] | None = None
    dashboard: dict[str, Any] | None = None
    operations: list[dict[str, Any]] = []
    non_trade_events: list[dict[str, Any]] = []
    source_refs: list[dict[str, Any]] = [benchmark_input_ref, *market_calendar_refs]
    pointer: dict[str, Any] | None = None
    catalog: dict[str, Any] | None = None
    try:
        loaded = load_registered_catalog(PROJECT_ROOT / RECORD_ROOT)
        if loaded is None:
            raise StrategyRecordStoreError("strategy-record store is unregistered")
        pointer, catalog = loaded
        if catalog.get("schema_id") != CATALOG_SCHEMA_V3:
            raise StrategyRecordStoreError("CANONICAL_PERFORMANCE_CLOSURE_MISSING")
        performance = load_performance_history(
            PROJECT_ROOT / RECORD_ROOT, catalog["performance_history_ref"]
        )
        pointer_path = PROJECT_ROOT / RECORD_ROOT / "_record_store/current.v1.json"
        pointer_sha, pointer_bytes = regular_file_sha256(
            pointer_path, label="strategy-record pointer"
        )
        if pointer_sha != _sha(pointer_path.read_bytes()):
            raise StrategyRecordStoreError("strategy-record pointer readback drift")
        identity_sha, identity_bytes = regular_file_sha256(
            PROJECT_ROOT / IDENTITY_PATH, label="strategy identity"
        )
        identity = resolve_strategy_identity(
            PROJECT_ROOT,
            declaration_path=IDENTITY_PATH,
            declaration_sha256=identity_sha,
            expected_historical_label=HISTORICAL_LABEL,
        )
        if identity.canonical_strategy_id != CANONICAL_STRATEGY_ID:
            raise StrategyRecordStoreError("strategy identity mismatch")
        dashboard = build_dashboard_bundle(
            project_root=PROJECT_ROOT,
            record_root=PROJECT_ROOT / RECORD_ROOT,
            benchmark_path=PROJECT_ROOT / BENCHMARK_PATH,
            risk_free_path=PROJECT_ROOT / RISK_FREE_PATH,
            generated_at=generated_at,
            today=_parse_utc(args.scheduled_at, label="scheduled_at")
            .astimezone(SHANGHAI)
            .date(),
        )
        domains["STORE_HOLDINGS"] = _domain(
            "FRESH",
            warnings=list(dashboard.get("warnings", [])),
            evidence={
                "latest_record_id": pointer["active_record_id"],
                "previous_record_id": pointer["previous_record_id"],
                "financial_state_sha256": pointer["active_closure"][
                    "financial_state_sha256"
                ],
            },
        )
        csi300 = next(
            row for row in dashboard["benchmarks"] if row["ts_code"] == "000300.SH"
        )
        domains["PERFORMANCE_BENCHMARK"] = _domain(
            "FRESH",
            warnings=[
                warning
                for warning in dashboard.get("warnings", [])
                if warning.startswith("latest_performance_stale")
            ],
            evidence={
                "performance_generation_id": catalog["performance_history_ref"][
                    "performance_generation_id"
                ],
                "performance_manifest_sha256": catalog["performance_history_ref"][
                    "manifest"
                ]["sha256"],
                "series_sha256": performance["series_sha256"],
                "benchmark_sha256": csi300["source_sha256"],
                "common_interval": [csi300["start_date"], csi300["end_date"]],
            },
        )
        store_evidence = {
            "pointer_path": RECORD_ROOT.joinpath("_record_store/current.v1.json").as_posix(),
            "pointer_sha256": pointer_sha,
            "pointer_bytes": pointer_bytes,
            "catalog_generation_id": pointer["generation_id"],
            "catalog_path": pointer["catalog_path"],
            "catalog_sha256": pointer["catalog_sha256"],
            "catalog_schema": catalog["schema_id"],
            "performance_contract_ready": True,
            "lineage_index_sha256": catalog["lineage_index_sha256"],
            "performance_history_ref": catalog["performance_history_ref"],
            "identity_path": IDENTITY_PATH,
            "identity_sha256": identity_sha,
            "identity_bytes": identity_bytes,
            "active_closure": pointer["active_closure"],
        }
        source_refs.extend(
            [
                {
                    "path": store_evidence["pointer_path"],
                    "sha256": pointer_sha,
                },
                {
                    "path": RECORD_ROOT.joinpath(pointer["catalog_path"]).as_posix(),
                    "sha256": pointer["catalog_sha256"],
                },
                {"path": IDENTITY_PATH, "sha256": identity_sha},
                *[
                    {
                        "path": RECORD_ROOT.joinpath(
                            catalog["performance_history_ref"][key]["path"]
                        ).as_posix(),
                        "sha256": catalog["performance_history_ref"][key]["sha256"],
                    }
                    for key in ("manifest", "series", "owner_declaration")
                ],
            ]
        )
    except (StrategyRecordStoreError, DashboardInputError, KeyError, StopIteration) as exc:
        blocker = str(exc)
        domains["STORE_HOLDINGS"] = _domain("BLOCKED", blockers=[blocker])
        domains["WEEKLY_OPERATIONS"] = _domain(
            "DEPENDENCY_BLOCKED", blockers=["STORE_HOLDINGS_BLOCKED"]
        )
        domains["PERFORMANCE_BENCHMARK"] = _domain(
            "DEPENDENCY_BLOCKED", blockers=["STORE_HOLDINGS_BLOCKED"]
        )

    if catalog is not None and store_evidence is not None:
        try:
            domains["WEEKLY_OPERATIONS"], operations, non_trade_events = _operations(
                root=PROJECT_ROOT / RECORD_ROOT,
                catalog=catalog,
                window=window,
            )
        except (StrategyRecordStoreError, WeeklyEvidenceError, KeyError) as exc:
            domains["WEEKLY_OPERATIONS"] = _domain(
                "BLOCKED", blockers=[str(exc)]
            )
            operations = []
            non_trade_events = []

    v17 = derive_mainline_state(PROJECT_ROOT, canonical_strategy_id=CANONICAL_STRATEGY_ID)
    v17_evidence: dict[str, Any] = {
        "derived_state": v17.derived_state,
        "blocker": v17.blocker.value if v17.blocker is not None else None,
        "public_run": v17.public_run,
    }
    formal_payload: dict[str, Any] = {
        "status": "FORMAL_ADVISORY_BLOCKED",
        "actions": [],
        "executable": False,
    }
    decision_payload: dict[str, Any] = {
        "status": "DEPENDENCY_BLOCKED",
        "write_performed": False,
        "path": DECISION_LOG_PATH.as_posix(),
        "envelope_path": None,
        "envelope_sha256": None,
        "already_recorded": False,
    }
    if v17.is_active:
        if store_evidence is None or dashboard is None or v17.public_run is None:
            domains["FORMAL_V17_ADVISORY"] = _domain(
                "DEPENDENCY_BLOCKED", blockers=["STORE_HOLDINGS_BLOCKED"]
            )
            domains["DECISION_LOG"] = _domain(
                "DEPENDENCY_BLOCKED", blockers=["STORE_HOLDINGS_BLOCKED"]
            )
        else:
            try:
                sidecar, formal_refs, sidecar_ref = _load_active_formal_sidecar(
                    v17.public_run
                )
                validated_formal = _validate_formal_advisory_sidecar(
                    sidecar,
                    public_run=v17.public_run,
                    formal_evidence_refs=formal_refs,
                    store_evidence=store_evidence,
                    holdings={"positions": dashboard["positions"]},
                    window=window,
                )
                envelope = _build_decision_envelope(
                    formal=validated_formal,
                    public_run=v17.public_run,
                    store_evidence=store_evidence,
                    window=window,
                    recorded_at=generated_at,
                )
                log_domain, already_recorded = _decision_log_domain(
                    envelope, path=PROJECT_ROOT / DECISION_LOG_PATH
                )
                envelope_path = output_dir / "decision_log_envelope.v2.json"
                envelope_raw = canonical_json_bytes(envelope)
                envelope_sha = immutable_write(
                    envelope_path, envelope_raw, max_bytes=128 * 1024
                )
                domains["FORMAL_V17_ADVISORY"] = _domain(
                    "FRESH",
                    evidence={
                        "weekly_advisory_ref": sidecar_ref,
                        "active_pointer_sha256": v17.public_run[
                            "active_pointer_ref"
                        ]["byte_sha256"],
                        "active_run_sha256": v17.public_run["mainline_run_ref"][
                            "byte_sha256"
                        ],
                    },
                )
                domains["DECISION_LOG"] = log_domain
                formal_payload = {
                    "status": "FORMAL_ADVISORY_READY",
                    "formal_outcome": validated_formal["status"],
                    "actions": validated_formal["actions"],
                    "executable": False,
                }
                decision_payload = {
                    "status": "RECORDED" if already_recorded else "PENDING_APPEND",
                    "write_performed": False,
                    "path": DECISION_LOG_PATH.as_posix(),
                    "envelope_path": str(envelope_path),
                    "envelope_sha256": envelope_sha,
                    "event_id": envelope["event_id"],
                    "idempotency_key": envelope["idempotency_key"],
                    "already_recorded": already_recorded,
                }
                source_refs.extend(
                    {
                        "path": ref["relative_path"],
                        "sha256": ref["byte_sha256"],
                    }
                    for ref in (
                        [
                            v17.public_run["active_pointer_ref"],
                            v17.public_run["mainline_run_ref"],
                            v17.public_run["formal_output_ref"],
                            v17.public_run["portfolio_output_ref"],
                            v17.public_run["source_closure_ref"],
                        ]
                        + formal_refs
                    )
                )
                source_refs.append(
                    {"path": str(envelope_path), "sha256": envelope_sha}
                )
            except (WeeklyEvidenceError, ValueError, RuntimeError, OSError, KeyError) as exc:
                blocker = str(exc) or "FORMAL_ACTION_GATES_INCOMPLETE"
                domains["FORMAL_V17_ADVISORY"] = _domain(
                    "BLOCKED", blockers=[blocker]
                )
                domains["DECISION_LOG"] = _domain(
                    "DEPENDENCY_BLOCKED", blockers=[blocker]
                )
    else:
        domains["FORMAL_V17_ADVISORY"] = _domain(
            "BLOCKED", blockers=["FORMAL_ADVISORY_BLOCKED", v17.derived_state]
        )
        domains["DECISION_LOG"] = _domain(
            "DEPENDENCY_BLOCKED", blockers=["FORMAL_ADVISORY_BLOCKED"]
        )

    bundle: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "generated_at": generated_at,
        "run_id": args.run_id,
        "report_window": window,
        "status": "BLOCKED",
        "domains": domains,
        "strategy": {
            "historical_label": HISTORICAL_LABEL,
            "canonical_strategy_id": CANONICAL_STRATEGY_ID,
            "historical_display_only": True,
        },
        "store": store_evidence,
        "holdings": (
            {
                "latest_record_id": dashboard["latest_valid_record"],
                "previous_record_id": dashboard["previous_valid_record"],
                "as_of": dashboard["latest_data_date"],
                "bundle_status": dashboard["status"],
                "data_age_calendar_days": dashboard["data_age_calendar_days"],
                "positions": dashboard["positions"],
                "changes": dashboard["changes"],
                "portfolio": dashboard["portfolio"],
                "concentration": dashboard["concentration"],
                "current_evidence": dashboard["current_evidence"],
                "previous_evidence": dashboard["previous_evidence"],
            }
            if dashboard is not None
            else None
        ),
        "weekly_operations": {
            "actual_fills": operations,
            "non_trade_events": non_trade_events,
            "actual_trade_count": len(operations),
        },
        "performance_benchmark": (
            {
                "portfolio": dashboard["portfolio"],
                "csi300": next(
                    row for row in dashboard["benchmarks"] if row["ts_code"] == "000300.SH"
                ),
                "funding_events": dashboard["history"]["funding_events"],
                "net_external_flow": dashboard["history"]["net_external_flow"],
                "benchmark_value_date_semantics": [
                    "exact_close",
                    "previous_trading_day_ffill",
                ],
            }
            if dashboard is not None
            else None
        ),
        "daily_reviews": {
            "trust": "UNTRUSTED_NARRATIVE_NOT_PORTFOLIO_AUTHORITY",
            "items": daily_rows,
        },
        "market_briefings": {
            "conversation_id": "6a394ef0-585c-83ec-863c-98e6bb6aec49",
            "trust": "UNTRUSTED_NARRATIVE_NOT_PORTFOLIO_AUTHORITY",
            "items": briefing_rows,
        },
        "public_web_research": {
            "trust": "RESEARCH_ONLY_NOT_LEDGER_BENCHMARK_OR_V17_AUTHORITY",
            "payload": web_payload,
        },
        "v17": v17_evidence,
        "formal_advisory": formal_payload,
        "decision_log": decision_payload,
        "source_refs": source_refs,
        "permissions": {
            "portfolio_state_read_only": True,
            "historical_holdings_store_authority_only": True,
            "repository_source_write": False,
            "strategy_record_write": False,
            "decision_log_write": False,
            "automation_memory_write": False,
            "public_web_research": True,
            "market_data_provider_api_calls": False,
            "v17_activation": False,
            "v17_mainline_authority": False,
            "new_risk_authorized": False,
            "broker_calls": False,
            "order_calls": False,
            "execution_calls": False,
            "trade_calls": False,
        },
        "warnings": sorted(
            {
                warning
                for domain in domains.values()
                for warning in domain.get("warnings", [])
            }
        ),
        "blockers": sorted(
            {
                blocker
                for domain in domains.values()
                for blocker in domain.get("blockers", [])
            }
        ),
    }
    bundle["status"] = _overall(domains)
    bundle["content_sha256"] = _content_sha(bundle)
    raw = canonical_json_bytes(bundle)
    if len(raw) > MAX_BUNDLE_BYTES:
        raise WeeklyEvidenceError("weekly evidence bundle exceeds byte budget")
    digest = immutable_write(output_path, raw, max_bytes=MAX_BUNDLE_BYTES)
    if digest != _sha(raw) or output_path.read_bytes() != raw:
        raise WeeklyEvidenceError("weekly evidence exact readback mismatch")
    return {
        "exported": True,
        "status": bundle["status"],
        "bundle_path": str(output_path),
        "content_sha256": bundle["content_sha256"],
        "byte_sha256": digest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduled-at", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--daily-review-json")
    parser.add_argument("--market-briefing-json")
    parser.add_argument("--public-web-json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export(args)
    except (WeeklyEvidenceError, StrategyRecordStoreError, OSError) as exc:
        print(
            json.dumps(
                {"exported": False, "status": "BLOCKED", "blocker": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
