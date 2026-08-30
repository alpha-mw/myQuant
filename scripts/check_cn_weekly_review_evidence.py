#!/usr/bin/env python3
"""Fail-closed checker for one CN weekly review evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.strategy_records.performance import load_performance_history  # noqa: E402
from quant_investor.strategy_records.store import (  # noqa: E402
    CATALOG_SCHEMA_V3,
    StrategyRecordStoreError,
    canonical_json_bytes,
    load_registered_catalog,
    regular_file_sha256,
)
from scripts.export_cn_weekly_review_evidence import (  # noqa: E402
    DOMAIN_NAMES,
    DOMAIN_STATUSES,
    MAX_BUNDLE_BYTES,
    RECORD_ROOT,
    SCHEMA_ID,
    WeeklyEvidenceError,
    _content_sha,
    _overall,
    report_window,
)


class WeeklyEvidenceCheckError(RuntimeError):
    """The weekly evidence bundle failed exact validation."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_bundle(path: Path) -> tuple[dict[str, Any], bytes, str]:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise WeeklyEvidenceCheckError("bundle is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size <= 0
        or metadata.st_size > MAX_BUNDLE_BYTES
    ):
        raise WeeklyEvidenceCheckError("bundle storage security is invalid")
    digest, size = regular_file_sha256(path, label="weekly evidence bundle")
    raw = path.read_bytes()
    if len(raw) != size or _sha(raw) != digest:
        raise WeeklyEvidenceCheckError("bundle changed during read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeeklyEvidenceCheckError("bundle is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise WeeklyEvidenceCheckError("bundle bytes are not canonical JSON")
    return value, raw, digest


def _check_domain_matrix(bundle: dict[str, Any]) -> None:
    domains = bundle.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(DOMAIN_NAMES):
        raise WeeklyEvidenceCheckError("evidence coverage matrix is incomplete")
    for name, domain in domains.items():
        if (
            not isinstance(domain, dict)
            or domain.get("status") not in DOMAIN_STATUSES
            or not isinstance(domain.get("blockers"), list)
            or not isinstance(domain.get("warnings"), list)
            or not isinstance(domain.get("evidence"), dict)
        ):
            raise WeeklyEvidenceCheckError(f"domain contract is invalid: {name}")
    if bundle.get("status") != _overall(domains):
        raise WeeklyEvidenceCheckError("overall status aggregation mismatch")
    if any(domain["status"] != "FRESH" for domain in domains.values()) and bundle.get(
        "status"
    ) == "FRESH":
        raise WeeklyEvidenceCheckError("non-fresh domain was hidden")


def _check_window(bundle: dict[str, Any]) -> None:
    window = bundle.get("report_window")
    if not isinstance(window, dict) or not isinstance(window.get("scheduled_at"), str):
        raise WeeklyEvidenceCheckError("report window is absent")
    if window != report_window(window["scheduled_at"]):
        raise WeeklyEvidenceCheckError("report window binding mismatch")


def _check_local_ref(ref: dict[str, Any], *, label: str) -> None:
    path = ref.get("path")
    digest = ref.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise WeeklyEvidenceCheckError(f"{label} ref is invalid")
    candidate = Path(path)
    if candidate.is_absolute():
        # Thread-produced narrative sidecars are the only allowed absolute
        # refs and must stay under /private/tmp.
        resolved = candidate.resolve(strict=True)
        if Path("/private/tmp").resolve(strict=True) not in resolved.parents:
            raise WeeklyEvidenceCheckError(f"{label} absolute ref escapes /private/tmp")
    else:
        resolved = (PROJECT_ROOT / candidate).resolve(strict=True)
        if PROJECT_ROOT.resolve(strict=True) not in resolved.parents:
            raise WeeklyEvidenceCheckError(f"{label} ref escapes project root")
    observed, _ = regular_file_sha256(resolved, label=label)
    if observed != digest:
        raise WeeklyEvidenceCheckError(f"{label} SHA-256 mismatch")


def _check_store(bundle: dict[str, Any]) -> None:
    holdings_status = bundle["domains"]["STORE_HOLDINGS"]["status"]
    if holdings_status not in {"FRESH", "PARTIAL"}:
        if bundle.get("store") is not None or bundle.get("holdings") is not None:
            raise WeeklyEvidenceCheckError("blocked holdings domain leaked stale values")
        return
    loaded = load_registered_catalog(PROJECT_ROOT / RECORD_ROOT)
    if loaded is None:
        raise WeeklyEvidenceCheckError("registered store disappeared")
    pointer, catalog = loaded
    if catalog.get("schema_id") != CATALOG_SCHEMA_V3:
        raise WeeklyEvidenceCheckError("weekly holdings did not consume catalog v3")
    store = bundle.get("store")
    holdings = bundle.get("holdings")
    if not isinstance(store, dict) or not isinstance(holdings, dict):
        raise WeeklyEvidenceCheckError("fresh holdings evidence is absent")
    pointer_path = PROJECT_ROOT / RECORD_ROOT / "_record_store/current.v1.json"
    pointer_sha, pointer_bytes = regular_file_sha256(
        pointer_path, label="strategy-record pointer"
    )
    expected = {
        "pointer_sha256": pointer_sha,
        "pointer_bytes": pointer_bytes,
        "catalog_generation_id": pointer["generation_id"],
        "catalog_path": pointer["catalog_path"],
        "catalog_sha256": pointer["catalog_sha256"],
        "catalog_schema": CATALOG_SCHEMA_V3,
        "performance_contract_ready": True,
        "lineage_index_sha256": catalog["lineage_index_sha256"],
        "performance_history_ref": catalog["performance_history_ref"],
        "active_closure": pointer["active_closure"],
    }
    for key, value in expected.items():
        if store.get(key) != value:
            raise WeeklyEvidenceCheckError(f"store evidence mismatch: {key}")
    if (
        holdings.get("latest_record_id") != pointer["active_record_id"]
        or holdings.get("previous_record_id") != pointer["previous_record_id"]
    ):
        raise WeeklyEvidenceCheckError("holdings pointer selection mismatch")
    performance = load_performance_history(
        PROJECT_ROOT / RECORD_ROOT, catalog["performance_history_ref"]
    )
    if performance["rows"][-1]["record_id"] != pointer["active_record_id"]:
        raise WeeklyEvidenceCheckError("performance active record mismatch")


def _check_operations(bundle: dict[str, Any]) -> None:
    operations = bundle.get("weekly_operations")
    if not isinstance(operations, dict):
        raise WeeklyEvidenceCheckError("weekly operations payload is absent")
    fills = operations.get("actual_fills")
    non_trade = operations.get("non_trade_events")
    if not isinstance(fills, list) or not isinstance(non_trade, list):
        raise WeeklyEvidenceCheckError("weekly operations tables are invalid")
    if operations.get("actual_trade_count") != len(fills):
        raise WeeklyEvidenceCheckError("weekly actual trade count mismatch")
    for row in fills:
        if not isinstance(row, dict) or row.get("decision_log_pairing") is None:
            raise WeeklyEvidenceCheckError("weekly fill evidence is incomplete")
    for row in non_trade:
        if not isinstance(row, dict) or row.get("described_as_trade") is not False:
            raise WeeklyEvidenceCheckError("non-trade event was misclassified")


def _check_performance_and_flows(bundle: dict[str, Any]) -> None:
    domain_status = bundle["domains"]["PERFORMANCE_BENCHMARK"]["status"]
    performance = bundle.get("performance_benchmark")
    if domain_status not in {"FRESH", "PARTIAL"}:
        if performance is not None:
            raise WeeklyEvidenceCheckError("blocked performance domain leaked values")
        return
    if not isinstance(performance, dict):
        raise WeeklyEvidenceCheckError("performance benchmark payload is absent")
    portfolio = performance.get("portfolio")
    benchmark = performance.get("csi300")
    funding_events = performance.get("funding_events")
    net_external_flow = performance.get("net_external_flow")
    if (
        not isinstance(portfolio, dict)
        or not isinstance(benchmark, dict)
        or not isinstance(funding_events, list)
        or not isinstance(net_external_flow, (int, float))
        or isinstance(net_external_flow, bool)
    ):
        raise WeeklyEvidenceCheckError("performance or funding evidence is invalid")
    running_flow = 0.0
    manifest_sha = bundle["store"]["performance_history_ref"]["manifest"]["sha256"]
    for event in funding_events:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("amount"), (int, float))
            or isinstance(event.get("amount"), bool)
            or event.get("binding_status") != "CANONICAL_PERFORMANCE_CLOSURE"
            or event.get("evidence_sha256") != manifest_sha
        ):
            raise WeeklyEvidenceCheckError("funding event is not canonically bound")
        running_flow += float(event["amount"])
    portfolio_flow = portfolio.get("excluded_external_flow")
    if (
        not isinstance(portfolio_flow, (int, float))
        or isinstance(portfolio_flow, bool)
    ):
        raise WeeklyEvidenceCheckError("portfolio external-flow total is invalid")
    if abs(running_flow - float(net_external_flow)) > 0.01:
        raise WeeklyEvidenceCheckError("net external flow does not reconcile")
    if abs(float(portfolio_flow) - float(net_external_flow)) > 0.01:
        raise WeeklyEvidenceCheckError("portfolio external-flow total does not reconcile")
    if performance.get("benchmark_value_date_semantics") != [
        "exact_close",
        "previous_trading_day_ffill",
    ]:
        raise WeeklyEvidenceCheckError("benchmark value-date semantics are invalid")


def _check_formal_boundaries(bundle: dict[str, Any]) -> None:
    formal = bundle.get("formal_advisory")
    decision_log = bundle.get("decision_log")
    if not isinstance(formal, dict) or not isinstance(decision_log, dict):
        raise WeeklyEvidenceCheckError("formal advisory boundary is absent")
    if bundle["domains"]["FORMAL_ADVISORY"]["status"] != "FRESH":
        if (
            formal.get("status") != "FORMAL_ADVISORY_BLOCKED"
            or formal.get("actions") != []
            or formal.get("executable") is not False
            or decision_log.get("write_performed") is not False
            or decision_log.get("status") != "NOT_APPLICABLE"
            or decision_log.get("reason") != "NO_FORMAL_ADVISORY_TO_LOG"
        ):
            raise WeeklyEvidenceCheckError("blocked formal advisory leaked actions or log writes")
    else:
        raise WeeklyEvidenceCheckError("weekly exporter does not produce formal advice")
    permissions = bundle.get("permissions")
    required_false = {
        "repository_source_write",
        "strategy_record_write",
        "decision_log_write",
        "automation_memory_write",
        "market_data_provider_api_calls",
        "system_activation",
        "mainline_authority",
        "new_risk_authorized",
        "broker_calls",
        "order_calls",
        "execution_calls",
        "trade_calls",
    }
    if not isinstance(permissions, dict) or any(
        permissions.get(key) is not False for key in required_false
    ):
        raise WeeklyEvidenceCheckError("scheduled-run permission receipt is invalid")
    if (
        permissions.get("portfolio_state_read_only") is not True
        or permissions.get("historical_holdings_store_authority_only") is not True
        or permissions.get("public_web_research") is not True
    ):
        raise WeeklyEvidenceCheckError("scheduled-run read boundary is invalid")


def check(path: Path) -> dict[str, Any]:
    bundle, raw, byte_sha = _read_bundle(path)
    if bundle.get("schema_id") != SCHEMA_ID:
        raise WeeklyEvidenceCheckError("weekly evidence schema is unsupported")
    observed_content = bundle.get("content_sha256")
    if not isinstance(observed_content, str) or observed_content != _content_sha(bundle):
        raise WeeklyEvidenceCheckError("weekly evidence content_sha256 mismatch")
    if b"ledger.csv" in raw.lower():
        raise WeeklyEvidenceCheckError("weekly evidence contains a disabled legacy ledger ref")
    _check_domain_matrix(bundle)
    _check_window(bundle)
    _check_store(bundle)
    _check_operations(bundle)
    _check_performance_and_flows(bundle)
    _check_formal_boundaries(bundle)
    source_refs = bundle.get("source_refs")
    if not isinstance(source_refs, list):
        raise WeeklyEvidenceCheckError("source_refs are absent")
    seen: set[tuple[str, str]] = set()
    for index, ref in enumerate(source_refs):
        if not isinstance(ref, dict):
            raise WeeklyEvidenceCheckError("source ref is invalid")
        identity = (str(ref.get("path")), str(ref.get("sha256")))
        if identity in seen:
            continue
        seen.add(identity)
        _check_local_ref(ref, label=f"weekly source ref {index}")
    for name in (
        "DAILY_REVIEW_COVERAGE",
        "MARKET_BRIEFING_COVERAGE",
        "PUBLIC_WEB_RESEARCH",
    ):
        ref = bundle["domains"][name]["evidence"].get("source_ref")
        if ref is not None:
            if not isinstance(ref, dict):
                raise WeeklyEvidenceCheckError(f"{name} source ref is invalid")
            _check_local_ref(ref, label=f"{name} input")
    return {
        "ok": True,
        "status": bundle["status"],
        "bundle_path": str(path),
        "content_sha256": observed_content,
        "byte_sha256": byte_sha,
        "bundle_bytes": len(raw),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = check(args.bundle)
    except (
        WeeklyEvidenceCheckError,
        WeeklyEvidenceError,
        StrategyRecordStoreError,
        OSError,
    ) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
