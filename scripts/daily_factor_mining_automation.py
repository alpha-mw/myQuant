#!/usr/bin/env python3
"""Daily governed factor-mining wrapper for myQuant CN factor automation."""

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
    apply_production_family_governance,
    parse_args as parse_mining_args,
    run_mining,
    write_outputs,
)
from scripts.retest_aquant_alpha_mix_8gate import _json_default  # noqa: E402
from quant_investor.factors.pit_fundamentals import (  # noqa: E402
    DEFAULT_FUNDAMENTAL_MART_ROOT,
)
from quant_investor.factors.registry_store import (  # noqa: E402
    load_registry_snapshot_strict,
)

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


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
    manifest_path_text = str(pointer.get("manifest_path", "") or "").strip()
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
    production_champions = [
        *production_governance.get("promoted_factors", []),
        *production_governance.get("kept_factors", []),
    ]
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
        f"- Success gate passed: {payload.get('success_gate_passed')}",
        f"- Fail-closed reason: {payload.get('fail_closed_reason') or '-'}",
        f"- Registry update status: {registry.get('status')}",
        "- Production family governance status: "
        f"{production_governance.get('status')}",
        "- Production family champions: "
        f"{', '.join(production_champions) or '-'}",
        "",
        (
            "The scheduled path uses one CAS/WAL transaction to promote only "
            "current strict full-A 8-gate diversity champions."
        ),
        (
            "No portfolio run, broker action, order, or strategy record is "
            "performed."
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
            "## Production Market Evidence",
            "",
            "```json",
            json.dumps(
                payload.get("production_market_evidence", {}),
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
    parser.add_argument("--no-registry-write", action="store_true")
    parser.add_argument(
        "--strict-positive-evidence",
        action="store_true",
        help=(
            "Exit non-zero unless at least --min-positive-candidates "
            "production_candidate factors also show positive IC or return "
            "evidence."
        ),
    )
    return parser.parse_args(argv)


def run_daily_automation(args: argparse.Namespace) -> dict[str, Any]:
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
    diverse_positive_champion_count = int(
        counts["diverse_positive_champion_count"]
    )
    candidate_promotion_gate_passed = (
        diverse_positive_champion_count >= int(
            args.min_positive_candidates
        )
    )
    registry_snapshot = load_registry_snapshot_strict(args.registry_path)
    incumbent_factors = sorted(
        factor.name
        for factor in registry_snapshot.registry.selectable_factors()
    )

    registry_write_requested = not bool(args.no_registry_write)
    report_json_path = (
        Path(mining_payload["output_dir"])
        / "quant_branch_factor_mining_results.json"
    )
    market_evidence = strict_full_a_market_evidence(
        data_root=args.data_root,
        universes=str(args.universes),
        loaded_symbol_count=int(mining_payload.get("loaded_symbol_count", 0)),
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
    market_evidence_ready = not market_evidence_blocker
    incumbent_carry_forward = bool(
        market_evidence_ready
        and not candidate_promotion_gate_passed
        and incumbent_factors
    )
    success_gate_passed = bool(
        market_evidence_ready
        and (candidate_promotion_gate_passed or incumbent_carry_forward)
    )
    fail_closed_reason = ""
    if not success_gate_passed:
        fail_closed_reason = market_evidence_blocker or (
            "no_diverse_registry_champion"
            if int(counts["positive_candidate_count"]) > 0
            else (
                "no_qualified_positive_candidate"
                if int(counts["positive_evidence_count"]) > 0
                else "no_positive_ic_or_return_evidence"
            )
        )

    if (
        candidate_promotion_gate_passed
        and success_gate_passed
        and registry_write_requested
    ):
        production_governance_manifest = apply_production_family_governance(
            registry_path=str(args.registry_path),
            results=results,
            run_timestamp=run_timestamp,
            run_id=run_id,
            report_path=str(report_json_path),
            journal_path=mining_output_dir
            / f"production_family_governance_{timestamp_slug}.json",
            universes=requested_universes,
            market_evidence=market_evidence,
            owner=str(args.registry_owner),
            source_notes=source_notes,
            horizon_days=int(args.horizon_days),
            max_champions=int(args.max_registry_candidates),
            write=True,
        )
    elif incumbent_carry_forward:
        production_governance_manifest = {
            "requested": registry_write_requested,
            "registry_path": str(args.registry_path),
            "run_id": run_id,
            "source_report": str(report_json_path),
            "status": "no_registry_changes",
            "carry_forward_reason": "no_new_8gate_challenger",
            "kept_factors": incumbent_factors,
            "promoted_factors": [],
            "deprecated_factors": [],
            "before_registry_sha256": registry_snapshot.registry_sha256,
            "after_registry_sha256": registry_snapshot.registry_sha256,
            "fail_closed_reason": "",
        }
    else:
        production_governance_manifest = {
            "requested": registry_write_requested,
            "registry_path": str(args.registry_path),
            "run_id": run_id,
            "source_report": str(report_json_path),
            "status": (
                "success_gate_failed"
                if registry_write_requested
                else "not_requested"
            ),
            "fail_closed_reason": fail_closed_reason,
        }
    registry_manifest = production_governance_manifest

    mining_payload["registry_write_requested"] = registry_write_requested
    mining_payload["registry_write"] = (
        registry_manifest.get("status") in {"updated", "no_registry_changes"}
    )
    mining_payload["registry_update_manifest"] = registry_manifest
    mining_payload["production_family_governance_manifest"] = (
        production_governance_manifest
    )
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
        "candidate_promotion_gate_passed": candidate_promotion_gate_passed,
        "incumbent_carry_forward": incumbent_carry_forward,
        "incumbent_factors": incumbent_factors,
        "success_gate_passed": success_gate_passed,
        "fail_closed_reason": fail_closed_reason,
        "registry_write_requested": registry_write_requested,
        "registry_update_manifest": registry_manifest,
        "production_family_governance_manifest": (
            production_governance_manifest
        ),
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
    payload = run_daily_automation(args)
    print(payload["summary_report_path"])
    print(
        json.dumps(
            {
                "success_gate_passed": payload["success_gate_passed"],
                "candidate_promotion_gate_passed": payload[
                    "candidate_promotion_gate_passed"
                ],
                "incumbent_carry_forward": payload[
                    "incumbent_carry_forward"
                ],
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
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    registry_status = str(payload["registry_update_manifest"].get("status", ""))
    if args.strict_positive_evidence and (
        not payload["success_gate_passed"]
        or (
            payload["registry_write_requested"]
            and registry_status not in {"updated", "no_registry_changes"}
        )
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
