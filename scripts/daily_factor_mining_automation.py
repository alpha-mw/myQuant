#!/usr/bin/env python3
"""Weekly factor mining; forward apply is blocked pending a trusted producer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mine_quant_branch_factors import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    _production_market_evidence_blocker,
    parse_args as parse_mining_args,
    run_mining,
    write_outputs,
)
from scripts.retest_aquant_alpha_mix_8gate import _json_default  # noqa: E402
from quant_investor.factors.pit_fundamentals import (  # noqa: E402
    DEFAULT_FUNDAMENTAL_MART_ROOT,
)
from quant_investor.factors.governance_protocol_v3 import (  # noqa: E402
    FORWARD_PRODUCTION_APPLY_BLOCKER,
    PROTOCOL_HASH,
    PROTOCOL_VERSION,
    canonical_replay_producer_control,
    protocol_hash,
)
from quant_investor.factors.registry_store import (  # noqa: E402
    load_registry_snapshot_strict,
)

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def load_governance_replay_evidence(*_args: Any, **_kwargs: Any) -> None:
    """Retired v2 hook retained only so stale callers fail explicitly."""

    raise ValueError("factor-governance-replay-evidence.v2 is retired")


def build_registry_mutation_plan_from_evidence(
    *_args: Any, **_kwargs: Any
) -> None:
    """Retired v2 hook retained only so stale callers fail explicitly."""

    raise ValueError("legacy mutation plans are retired; v3 evidence cannot be auto-upgraded")


def apply_governed_transition(*_args: Any, **_kwargs: Any) -> None:
    """Registry mutation is unavailable until bootstrap is separately approved."""

    raise ValueError(FORWARD_PRODUCTION_APPLY_BLOCKER)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_shanghai() -> datetime:
    return datetime.now(tz=SHANGHAI_TZ)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": str(exc), "path": str(path)}
    if isinstance(payload, Mapping):
        return payload
    return {"payload_type": type(payload).__name__}


def latest_download_report(
    report_dir: Path,
) -> tuple[Path | None, Mapping[str, Any]]:
    paths = list(report_dir.glob("download_report_*.json"))
    if not paths:
        return None, {}
    latest = max(paths, key=lambda path: (path.stat().st_mtime, path.name))
    return latest, _load_json(latest)


def compact_download_status(
    report_path: Path | None,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(report.get("config", {}) or {})
    completeness = dict(report.get("completeness", {}) or {})
    same_day_probe = dict(
        report.get("same_day_close_probe")
        or config.get("same_day_close_probe")
        or {}
    )
    blocking_symbols = completeness.get("blocking_incomplete_symbols", [])
    blocking_stale = completeness.get("blocking_stale_symbols", [])
    suspended_stale = completeness.get("suspended_stale_symbols", [])
    return {
        "report_path": str(report_path) if report_path else "",
        "report_exists": bool(report_path),
        "freshness_mode": config.get("freshness_mode"),
        "strict_trade_date": config.get("strict_trade_date"),
        "effective_target_trade_date": (
            completeness.get("effective_target_trade_date")
            or config.get("effective_target_trade_date")
        ),
        "stable_trade_date": (
            completeness.get("stable_trade_date")
            or config.get("stable_trade_date")
        ),
        "latest_trade_date": (
            completeness.get("latest_trade_date")
            or report.get("latest_trade_date")
        ),
        "final_complete": completeness.get("final_complete"),
        "coverage_ratio": (
            completeness.get("coverage_ratio")
            or completeness.get("final_coverage_ratio")
        ),
        "blocking_incomplete_count": len(blocking_symbols)
        if isinstance(blocking_symbols, list)
        else completeness.get("blocking_incomplete_count"),
        "blocking_stale_count": len(blocking_stale)
        if isinstance(blocking_stale, list)
        else completeness.get("blocking_stale_count"),
        "suspended_stale_count": len(suspended_stale)
        if isinstance(suspended_stale, list)
        else completeness.get("suspended_stale_count"),
        "early_stop_reason": report.get("early_stop_reason", ""),
        "same_day_probe": {
            "source": same_day_probe.get("source"),
            "available": same_day_probe.get("available"),
            "available_count": same_day_probe.get("available_count"),
            "expected_count": same_day_probe.get("expected_count"),
            "coverage_ratio": same_day_probe.get("coverage_ratio"),
        },
    }


def strict_full_a_market_evidence(
    *,
    data_root: str | Path,
    universes: str,
    loaded_symbol_count: int,
) -> dict[str, Any]:
    """Read strict-Parquet lineage used to gate any future governed apply."""

    root = Path(data_root).expanduser()
    parquet_root = (
        root
        if root.name == "cn" and root.parent.name == "parquet"
        else root / "parquet" / "cn"
    )
    pointer_path = parquet_root / "_latest.json"
    pointer = _load_json(pointer_path) if pointer_path.exists() else {}
    coverage = dict(pointer.get("coverage", {}) or {})
    expected = int(
        coverage.get("expected_scope_count")
        or coverage.get("coverage_complete_count")
        or 0
    )
    table_root_text = str(pointer.get("table_root", "") or "").strip()
    serving_root_text = str(
        pointer.get("derived_serving_root", "") or ""
    ).strip()
    manifest_path_text = str(
        pointer.get("manifest_path", "") or ""
    ).strip()
    return {
        "backend": str(
            os.getenv("MYQUANT_MARKET_DATA_BACKEND", "parquet")
        ).lower(),
        "mode_policy": str(
            os.getenv("MYQUANT_MARKET_DATA_MODE_POLICY", "strict")
        ).lower(),
        "requested_universes": [
            item.strip().lower()
            for item in str(universes).split(",")
            if item.strip()
        ],
        "pointer_path": str(pointer_path),
        "pointer_status": pointer.get("status"),
        "snapshot_id": pointer.get("snapshot_id"),
        "latest_complete_trade_date": pointer.get(
            "latest_complete_trade_date"
        ),
        "coverage_complete": coverage.get("complete") is True,
        "coverage_ratio": coverage.get("coverage_ratio"),
        "expected_symbol_count": expected,
        "loaded_symbol_count": int(loaded_symbol_count),
        "table_root_exists": bool(table_root_text)
        and Path(table_root_text).exists(),
        "serving_root_exists": bool(serving_root_text)
        and Path(serving_root_text).exists(),
        "manifest_exists": bool(manifest_path_text)
        and Path(manifest_path_text).exists(),
    }


def _load_source_notes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        notes = payload.get("source_notes", payload.get("ideas", []))
        if isinstance(notes, list):
            return [dict(item) for item in notes if isinstance(item, Mapping)]
        return [dict(payload)]
    return []


def ensure_source_notes(
    *,
    requested_path: str,
    output_dir: Path,
    market_status: Mapping[str, Any],
    run_timestamp: str,
) -> tuple[Path, list[dict[str, Any]], str]:
    if str(requested_path or "").strip():
        path = Path(requested_path).expanduser()
        if path.exists():
            return path, _load_source_notes(path), "supplied"
        status = "supplied_missing_fallback_created"
    else:
        path = output_dir / "source_notes.json"
        status = "fallback_created"

    notes = [
        {
            "title": "local_market_data_status",
            "source_type": "local_download_report",
            "url": "",
            "observed_at": run_timestamp,
            "note": (
                "No external Chrome source-notes file was supplied to the "
                "wrapper. The automation prompt should populate this file "
                "from read-only Chrome research when Chrome is available."
            ),
            "market_status": dict(market_status),
        }
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"source_notes": notes},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path, notes, status


def _has_positive_evidence(result: Mapping[str, Any]) -> bool:
    metrics = dict(result.get("metrics", {}) or {})
    return any(
        _safe_float(metrics.get(key)) > 0.0
        for key in (
            "mean_rankic",
            "top_bottom_spread",
            "top_quantile_return",
            "cost_adjusted_return",
            "master_return_delta",
            "sharpe_delta",
        )
    )


def _factor_sets(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    positive_families: dict[str, int] = {}
    qualified_families: dict[str, int] = {}
    for result in results:
        family = str(result.get("family", "") or "unknown")
        families[family] = families.get(family, 0) + 1
        if _has_positive_evidence(result):
            positive_families[family] = (
                positive_families.get(family, 0) + 1
            )
        if result.get("decision") == "production_candidate":
            qualified_families[family] = (
                qualified_families.get(family, 0) + 1
            )
    return {
        "family_count": len(families),
        "positive_family_count": len(positive_families),
        "qualified_family_count": len(qualified_families),
        "families": sorted(families),
        "positive_families": sorted(positive_families),
        "qualified_families": sorted(qualified_families),
        "by_family": families,
    }


def evidence_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [result for result in results if _has_positive_evidence(result)]
    qualified = [
        result
        for result in results
        if result.get("decision") == "production_candidate"
    ]
    qualified_positive = [
        result for result in qualified if _has_positive_evidence(result)
    ]
    diverse_champions = [
        result
        for result in qualified_positive
        if dict(result.get("diversity_selection", {}) or {}).get(
            "final_registry_write_eligible"
        )
        is True
    ]
    return {
        "positive_evidence_count": len(positive),
        "qualified_count": len(qualified),
        "positive_candidate_count": len(qualified_positive),
        "diverse_positive_champion_count": len(diverse_champions),
        "positive_factors": [
            str(result.get("name", "")) for result in positive
        ],
        "qualified_factors": [
            str(result.get("name", "")) for result in qualified
        ],
        "positive_candidate_factors": [
            str(result.get("name", "")) for result in qualified_positive
        ],
        "diverse_positive_champions": [
            str(result.get("name", "")) for result in diverse_champions
        ],
        "family_coverage": _factor_sets(results),
    }


def _registry_manifest_for_failed_gate(
    *,
    registry_path: str,
    run_id: str,
    report_path: str,
    max_candidates: int,
    qualified_count: int,
    requested: bool,
    fail_closed_reason: str,
) -> dict[str, Any]:
    return {
        "requested": bool(requested),
        "registry_path": str(registry_path),
        "run_id": run_id,
        "source_report": report_path,
        "max_candidates": int(max_candidates),
        "qualified_count": int(qualified_count),
        "written_count": 0,
        "updated_count": 0,
        "skipped_count": 0,
        "written_factors": [],
        "updated_factors": [],
        "skipped_factors": [],
        "status": "success_gate_failed" if requested else "not_requested",
        "fail_closed_reason": fail_closed_reason,
    }


def render_summary_markdown(payload: Mapping[str, Any]) -> str:
    counts = dict(payload.get("evidence_counts", {}) or {})
    registry = dict(payload.get("registry_update_manifest", {}) or {})
    production_governance = dict(
        payload.get("production_family_governance_manifest", {}) or {}
    )
    lines = [
        "# myQuant Daily Factor Mining Automation",
        "",
        f"- Run timestamp: {payload.get('run_timestamp')}",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Mining output: `{payload.get('mining_output_dir')}`",
        f"- Source notes: `{payload.get('source_notes_path')}`",
        f"- Source notes status: {payload.get('source_notes_status')}",
        "- Data report: "
        f"`{payload.get('market_data_status', {}).get('report_path', '')}`",
        f"- Candidate count: {payload.get('candidate_count')}",
        f"- Positive evidence count: {counts.get('positive_evidence_count')}",
        "- Positive production-candidate count: "
        f"{counts.get('positive_candidate_count')}",
        "- Diverse positive champion count: "
        f"{counts.get('diverse_positive_champion_count')}",
        f"- Qualified count: {counts.get('qualified_count')}",
        "- Mining evidence gate passed (not a mutation): "
        f"{payload.get('success_gate_passed')}",
        f"- Fail-closed reason: {payload.get('fail_closed_reason') or '-'}",
        f"- Run mode: {payload.get('run_mode')}",
        f"- Registry update status: {registry.get('status')}",
        "- Factor protocol v2 transition status: "
        f"{production_governance.get('status')}",
        "- Production transition applied: "
        f"{payload.get('factor_protocol', {}).get('transition_applied')}",
        f"- Protocol hash: `{payload.get('factor_protocol', {}).get('protocol_hash', '')}`",
        "- Transition blockers: "
        f"{', '.join(production_governance.get('blockers', [])) or '-'}",
        "",
        (
            "Weekly mining is report-only. Registry mutation is available only "
            "for one protocol-v2 month-end targeted transition."
        ),
        (
            "No portfolio run, broker action, live provider call, or strategy "
            "record is performed by this wrapper."
        ),
        "",
        "## Positive Candidate Factors",
        "",
    ]
    factors = counts.get("positive_candidate_factors", [])
    if factors:
        for name in factors:
            lines.append(f"- `{name}`")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Registry Update",
            "",
            "```json",
            json.dumps(
                registry,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            "```",
            "",
            "## Market Data Status",
            "",
            "```json",
            json.dumps(
                payload.get("market_data_status", {}),
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--fundamental-mart-root",
        default=str(DEFAULT_FUNDAMENTAL_MART_ROOT),
    )
    parser.add_argument("--raw-report-dir", default="data/cn_market_full")
    parser.add_argument(
        "--output-root",
        default="reports/factor_governance/daily_mining",
    )
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--registry-owner",
        default="myQuant daily factor mining automation",
    )
    parser.add_argument("--source-notes-json", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--universes", default="full_a")
    parser.add_argument("--windows", default="5,10,15,20,25,30,40,60,90,120")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--warmup-days", type=int, default=260)
    parser.add_argument("--analysis-start-date", default="auto")
    parser.add_argument("--decision-cost-bps", type=float, default=1.0)
    parser.add_argument(
        "--incremental-sleeve-weight",
        type=float,
        default=0.03,
    )
    parser.add_argument("--max-registry-candidates", type=int, default=5)
    parser.add_argument("--min-positive-candidates", type=int, default=1)
    parser.add_argument(
        "--no-registry-write",
        action="store_true",
        help="Deprecated compatibility flag; report-only is already the default.",
    )
    parser.add_argument(
        "--apply-governed-transitions",
        action="store_true",
        help="Retired compatibility flag; v3 registry mutation remains blocked.",
    )
    parser.add_argument(
        "--protocol-version",
        default="",
        help="Legacy compatibility input; registry mutation remains blocked.",
    )
    parser.add_argument(
        "--expected-protocol-hash",
        default="",
        help="Required current protocol hash when applying a governed transition.",
    )
    parser.add_argument(
        "--governed-evidence-json",
        default="",
        help=(
            "Legacy v2 input is rejected; v3 bootstrap is plan-only."
        ),
    )
    parser.add_argument(
        "--mutation-budget-ledger",
        default=(
            "reports/factor_governance/state/"
            "monthly_mutation_budget.v1.jsonl"
        ),
        help="Independent append-only monthly mutation reservation ledger.",
    )
    parser.add_argument(
        "--strict-positive-evidence",
        action="store_true",
        help=(
            "Exit non-zero unless at least --min-positive-candidates "
            "production_candidate factors also show positive IC or return "
            "evidence."
        ),
    )
    args = parser.parse_args(argv)
    if args.apply_governed_transitions:
        return args
    elif any(
        str(value or "").strip()
        for value in (
            args.protocol_version,
            args.expected_protocol_hash,
            args.governed_evidence_json,
        )
    ):
        parser.error(
            "protocol/apply options require --apply-governed-transitions"
        )
    return args


def _forward_apply_blocked_payload() -> dict[str, Any]:
    producer_control = {
        "producer_implemented": True,
        "local_bytes_readback_verified": False,
        "canonical_producer_authenticated": False,
        "production_apply_authorized": False,
        "production_apply_eligible": False,
        "blocker": FORWARD_PRODUCTION_APPLY_BLOCKER,
    }
    manifest = {
        "schema_version": "factor-governance-protocol.v2",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": PROTOCOL_HASH,
        "apply_requested": True,
        "status": "blocked",
        "blockers": [FORWARD_PRODUCTION_APPLY_BLOCKER],
        "before_registry_sha256": "",
        "after_registry_sha256": "",
        "inverse_wal_path": "",
        "mutation_budget_ledger_path": "",
        "changed_record_names": [],
        "registry_mutation_manifest": None,
        "canonical_replay_producer_control": producer_control,
    }
    return {
        "summary_report_path": "",
        "success_gate_passed": False,
        "run_mode": "governed_apply_blocked",
        "fail_closed_reason": FORWARD_PRODUCTION_APPLY_BLOCKER,
        "candidate_count": 0,
        "evidence_counts": {
            "positive_evidence_count": 0,
            "positive_candidate_count": 0,
            "diverse_positive_champion_count": 0,
            "qualified_count": 0,
        },
        "registry_write_requested": True,
        "registry_write": False,
        "registry_update_manifest": manifest,
        "production_family_governance_manifest": manifest,
        "factor_protocol": {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": PROTOCOL_HASH,
            "apply_requested": True,
            "status": "blocked",
            "transition_applied": False,
            "blockers": [FORWARD_PRODUCTION_APPLY_BLOCKER],
            "canonical_replay_producer_control": producer_control,
        },
    }


def run_daily_automation(args: argparse.Namespace) -> dict[str, Any]:
    if args.apply_governed_transitions:
        return _forward_apply_blocked_payload()

    now = _now_shanghai()
    run_timestamp = now.isoformat(timespec="seconds")
    timestamp_slug = now.strftime("%Y%m%d_%H%M%S")
    run_id = (
        str(args.run_id or "").strip()
        or f"daily_factor_mining_{timestamp_slug}"
    )
    output_root = Path(args.output_root).expanduser()
    output_dir = output_root / run_id
    mining_output_dir = output_dir / "mining"
    report_path, report = latest_download_report(
        Path(args.raw_report_dir).expanduser()
    )
    market_status = compact_download_status(report_path, report)
    source_notes_path, source_notes, source_status = ensure_source_notes(
        requested_path=str(args.source_notes_json),
        output_dir=output_dir,
        market_status=market_status,
        run_timestamp=run_timestamp,
    )

    mining_argv = [
        "--data-root",
        str(args.data_root),
        "--fundamental-mart-root",
        str(args.fundamental_mart_root),
        "--universes",
        str(args.universes),
        "--windows",
        str(args.windows),
        "--horizon-days",
        str(args.horizon_days),
        "--warmup-days",
        str(args.warmup_days),
        "--analysis-start-date",
        str(args.analysis_start_date),
        "--decision-cost-bps",
        str(args.decision_cost_bps),
        "--incremental-sleeve-weight",
        str(args.incremental_sleeve_weight),
        "--output-dir",
        str(mining_output_dir),
        "--run-id",
        run_id,
        "--source-notes-json",
        str(source_notes_path),
        "--registry-path",
        str(args.registry_path),
        "--registry-owner",
        str(args.registry_owner),
        "--max-registry-candidates",
        str(args.max_registry_candidates),
    ]
    mining_payload = run_mining(parse_mining_args(mining_argv))
    results = [
        dict(item)
        for item in mining_payload.get("results", [])
        if isinstance(item, Mapping)
    ]
    counts = evidence_counts(results)
    market_evidence = strict_full_a_market_evidence(
        data_root=args.data_root,
        universes=str(args.universes),
        loaded_symbol_count=int(
            mining_payload.get("loaded_symbol_count", 0)
        ),
    )
    market_evidence["factor_exposure_evidence"] = dict(
        mining_payload.get("factor_exposure_evidence", {}) or {}
    )
    requested_universes = [
        item.strip()
        for item in str(args.universes).split(",")
        if item.strip()
    ]
    market_evidence_blocker = _production_market_evidence_blocker(
        universes=requested_universes,
        market_evidence=market_evidence,
    )
    diverse_positive_champion_count = int(
        counts["diverse_positive_champion_count"]
    )
    candidate_evidence_gate_passed = diverse_positive_champion_count >= int(
        args.min_positive_candidates
    )
    success_gate_passed = bool(
        candidate_evidence_gate_passed and not market_evidence_blocker
    )
    fail_closed_reason = ""
    if not candidate_evidence_gate_passed:
        fail_closed_reason = (
            "no_diverse_registry_champion"
            if int(counts["positive_candidate_count"]) > 0
            else (
                "no_qualified_positive_candidate"
                if int(counts["positive_evidence_count"]) > 0
                else "no_positive_ic_or_return_evidence"
            )
        )
    if market_evidence_blocker:
        fail_closed_reason = market_evidence_blocker

    registry_write_requested = bool(args.apply_governed_transitions)
    report_json_path = (
        Path(mining_payload["output_dir"])
        / "quant_branch_factor_mining_results.json"
    )
    if registry_write_requested and market_evidence_blocker:
        snapshot = load_registry_snapshot_strict(str(args.registry_path))
        production_governance_manifest = {
            "schema_version": "factor-governance-protocol.v3",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "apply_requested": True,
            "registry_path": str(args.registry_path),
            "run_id": run_id,
            "source_report": str(report_json_path),
            "status": "blocked",
            "blockers": [market_evidence_blocker],
            "before_registry_sha256": snapshot.registry_sha256,
            "after_registry_sha256": snapshot.registry_sha256,
            "inverse_wal_path": "",
            "changed_record_names": [],
            "canonical_replay_producer_control": (
                canonical_replay_producer_control()
            ),
        }
    elif registry_write_requested:
        snapshot = load_registry_snapshot_strict(str(args.registry_path))
        production_governance_manifest = {
            "schema_version": "factor-governance-protocol.v3",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "apply_requested": True,
            "registry_path": str(args.registry_path),
            "run_id": run_id,
            "source_report": str(report_json_path),
            "status": "blocked",
            "blockers": [FORWARD_PRODUCTION_APPLY_BLOCKER],
            "before_registry_sha256": snapshot.registry_sha256,
            "after_registry_sha256": snapshot.registry_sha256,
            "inverse_wal_path": "",
            "changed_record_names": [],
            "canonical_replay_producer_control": canonical_replay_producer_control(),
        }
    else:
        snapshot = load_registry_snapshot_strict(str(args.registry_path))
        producer_control = canonical_replay_producer_control()
        production_governance_manifest = {
            "schema_version": "factor-governance-protocol.v3",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "apply_requested": False,
            "registry_path": str(args.registry_path),
            "run_id": run_id,
            "source_report": str(report_json_path),
            "status": "report_only",
            "blockers": (
                [market_evidence_blocker]
                if market_evidence_blocker
                else []
            ),
            "before_registry_sha256": snapshot.registry_sha256,
            "after_registry_sha256": snapshot.registry_sha256,
            "inverse_wal_path": "",
            "changed_record_names": [],
            "canonical_replay_producer_control": producer_control,
        }
    registry_manifest = production_governance_manifest

    mining_payload["registry_write_requested"] = registry_write_requested
    mining_payload["registry_write"] = (
        registry_manifest.get("status") == "applied"
    )
    mining_payload["registry_update_manifest"] = registry_manifest
    mining_payload["production_family_governance_manifest"] = (
        production_governance_manifest
    )
    factor_protocol_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "apply_requested": registry_write_requested,
        "status": production_governance_manifest.get("status"),
        "transition_applied": (
            production_governance_manifest.get("status") == "applied"
        ),
        "canonical_replay_producer_control": (
            production_governance_manifest.get(
                "canonical_replay_producer_control",
                canonical_replay_producer_control(),
            )
        ),
        "transition_id": production_governance_manifest.get(
            "transition_id", ""
        ),
        "transition_hash": production_governance_manifest.get(
            "transition_hash", ""
        ),
        "mutation_plan_hash": production_governance_manifest.get(
            "mutation_plan_hash", ""
        ),
        "evidence_hash": production_governance_manifest.get(
            "evidence_hash", ""
        ),
        "governed_evidence_path": (
            str(args.governed_evidence_json)
            if registry_write_requested
            else ""
        ),
        "mutation_budget_ledger_path": (
            production_governance_manifest.get(
                "mutation_budget_ledger_path",
                str(args.mutation_budget_ledger),
            )
            if registry_write_requested
            else ""
        ),
        "mutation_budget_reservation": (
            production_governance_manifest.get(
                "mutation_budget_reservation"
            )
        ),
        "blockers": list(
            production_governance_manifest.get("blockers", []) or []
        ),
        "before_registry_sha256": production_governance_manifest.get(
            "before_registry_sha256", ""
        ),
        "after_registry_sha256": production_governance_manifest.get(
            "after_registry_sha256", ""
        ),
        "inverse_wal_path": production_governance_manifest.get(
            "inverse_wal_path", ""
        ),
    }
    mining_payload["factor_protocol"] = factor_protocol_summary
    write_outputs(Path(mining_payload["output_dir"]), mining_payload)

    payload: dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "mining_output_dir": str(mining_payload["output_dir"]),
        "summary_json_path": str(
            output_dir / "daily_factor_mining_summary.json"
        ),
        "summary_report_path": str(
            output_dir / "daily_factor_mining_summary.md"
        ),
        "source_notes_path": str(source_notes_path),
        "source_notes_status": source_status,
        "source_note_count": len(source_notes),
        "market_data_status": market_status,
        "production_market_evidence": market_evidence,
        "market_evidence_blocker": market_evidence_blocker,
        "candidate_count": int(mining_payload.get("candidate_count", 0)),
        "evidence_counts": counts,
        "success_gate_passed": success_gate_passed,
        "fail_closed_reason": fail_closed_reason,
        "registry_write_requested": registry_write_requested,
        "run_mode": (
            "governed_apply" if registry_write_requested else "report_only"
        ),
        "registry_update_manifest": registry_manifest,
        "production_family_governance_manifest": (
            production_governance_manifest
        ),
        "factor_protocol": factor_protocol_summary,
        "mining_conclusion": mining_payload.get("conclusion", ""),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "daily_factor_mining_summary.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "daily_factor_mining_summary.md").write_text(
        render_summary_markdown(payload),
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.apply_governed_transitions:
        print(FORWARD_PRODUCTION_APPLY_BLOCKER, file=sys.stderr)
        return 2
    try:
        payload = run_daily_automation(args)
    except (OSError, ValueError) as exc:
        print(f"factor_governance_apply_blocked={exc}", file=sys.stderr)
        return 2
    print(payload["summary_report_path"])
    print(
        json.dumps(
            {
                "success_gate_passed": payload["success_gate_passed"],
                "run_mode": payload["run_mode"],
                "fail_closed_reason": payload["fail_closed_reason"],
                "candidate_count": payload["candidate_count"],
                "positive_evidence_count": payload["evidence_counts"][
                    "positive_evidence_count"
                ],
                "positive_candidate_count": payload["evidence_counts"][
                    "positive_candidate_count"
                ],
                "diverse_positive_champion_count": payload[
                    "evidence_counts"
                ]["diverse_positive_champion_count"],
                "qualified_count": payload["evidence_counts"][
                    "qualified_count"
                ],
                "registry_update_status": payload["registry_update_manifest"][
                    "status"
                ],
                "production_family_governance_status": payload[
                    "production_family_governance_manifest"
                ]["status"],
                "factor_protocol_status": payload["factor_protocol"][
                    "status"
                ],
                "production_transition_applied": payload[
                    "factor_protocol"
                ]["transition_applied"],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    if args.strict_positive_evidence and not payload["success_gate_passed"]:
        return 2
    if args.apply_governed_transitions and payload["factor_protocol"][
        "status"
    ] != "applied":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
