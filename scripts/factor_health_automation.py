#!/usr/bin/env python3
"""Periodic health checks for governed mined factors.

The run is offline and report-only. It reads the local mined-factor registry,
uses approved registry evidence for classification, and runs a strict Parquet
runtime smoke check. The retired ``--apply-registry-actions`` compatibility
flag returns blocked and cannot write; FactorGovernanceProtocol v2 is the only
production reconciliation authority.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.governance import (  # noqa: E402
    FactorAdmissionDecision,
    FactorLifecycleState,
    FactorRecord,
)
from quant_investor.factors.health import (  # noqa: E402
    active_failure_maturity_window_ids,
    classify_factor_health,
)
from quant_investor.factors.registry_store import (  # noqa: E402
    load_registry_snapshot_strict,
)
from quant_investor.factors.governance_protocol_v2 import (  # noqa: E402
    FORWARD_PRODUCTION_APPLY_BLOCKER,
)
from quant_investor.factors.runtime import MinedFactorRegistry  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cadence", choices=["daily", "weekly", "monthly", "review"], default="daily"
    )
    parser.add_argument(
        "--market",
        default="CN",
        help="Market used by the strict Parquet MarketDataReader runtime smoke.",
    )
    parser.add_argument(
        "--mode-policy",
        default=os.getenv("MYQUANT_MARKET_DATA_MODE_POLICY", "strict"),
        help="MarketDataReader mode policy for the runtime smoke.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Repository data root containing parquet/<market>/_latest.json.",
    )
    parser.add_argument("--universes", nargs="+", default=["full_a"])
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--warmup-days", type=int, default=260)
    parser.add_argument("--runtime-smoke-symbols", type=int, default=40)
    parser.add_argument("--output-dir", default="reports/factor_governance/health")
    parser.add_argument(
        "--registry-path",
        default="quant_investor/factor_registry/mined_factors.json",
    )
    parser.add_argument(
        "--apply-registry-actions",
        action="store_true",
        help=(
            "Retired compatibility flag. The run remains report-only and "
            "returns blocked; use daily_factor_mining_automation.py with the "
            "three explicit FactorGovernanceProtocol v2 apply arguments."
        ),
    )
    parser.add_argument(
        "--fresh-evaluation",
        action="store_true",
        help=(
            "Best-effort local re-evaluation of monitored price/volume factors. "
            "Report-only runs may fall back to registry evidence unless "
            "strict "
            "fresh evaluation is requested."
        ),
    )
    parser.add_argument(
        "--strict-fresh-evaluation",
        action="store_true",
        help="Return non-zero when --fresh-evaluation cannot evaluate every monitored factor.",
    )
    parser.add_argument("--analysis-start-date", default="auto")
    parser.add_argument("--min-analysis-price-coverage", type=float, default=0.95)
    parser.add_argument("--decision-cost-bps", type=float, default=1.0)
    parser.add_argument("--incremental-sleeve-weight", type=float, default=0.03)
    parser.add_argument(
        "--allow-production-promotion",
        action="store_true",
        help="Accepted for CLI compatibility. Scheduled runs should leave it disabled.",
    )
    parser.add_argument("--max-new-production", type=int, default=0)
    args = parser.parse_args(argv)
    if args.apply_registry_actions:
        return args
    if args.strict_fresh_evaluation and not args.fresh_evaluation:
        parser.error("--strict-fresh-evaluation requires --fresh-evaluation")
    if args.apply_registry_actions and not (
        args.fresh_evaluation and args.strict_fresh_evaluation
    ):
        parser.error(
            "--apply-registry-actions requires --fresh-evaluation and "
            "--strict-fresh-evaluation"
        )
    args.mode_policy = str(args.mode_policy or "").strip().lower()
    if args.strict_fresh_evaluation and args.mode_policy != "strict":
        parser.error(
            "--strict-fresh-evaluation requires --mode-policy strict"
        )
    if (
        not np.isfinite(args.min_analysis_price_coverage)
        or not 0.0 < args.min_analysis_price_coverage <= 1.0
    ):
        parser.error("--min-analysis-price-coverage must be in (0, 1]")
    if args.horizon_days <= 0:
        parser.error("--horizon-days must be positive")
    if args.warmup_days <= 0:
        parser.error("--warmup-days must be positive")
    if args.runtime_smoke_symbols <= 0:
        parser.error("--runtime-smoke-symbols must be positive")
    if not np.isfinite(args.decision_cost_bps) or args.decision_cost_bps < 0:
        parser.error("--decision-cost-bps must be finite and non-negative")
    if (
        not np.isfinite(args.incremental_sleeve_weight)
        or not 0.0 < args.incremental_sleeve_weight <= 1.0
    ):
        parser.error("--incremental-sleeve-weight must be in (0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.apply_registry_actions:
        print(FORWARD_PRODUCTION_APPLY_BLOCKER, file=sys.stderr)
        return 2
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"factor_health_{timestamp.replace(':', '')}.json"
    md_path = json_path.with_suffix(".md")

    registry_path = Path(args.registry_path)
    registry_snapshot = load_registry_snapshot_strict(registry_path)
    registry = registry_snapshot.registry
    monitored = _monitored_factors(registry)
    registry_evaluations = {
        factor.name: _evaluation_from_record(factor)
        for factor in monitored
        if _evaluation_from_record(factor) is not None
    }
    fresh_result = (
        _fresh_evaluations(args, monitored) if args.fresh_evaluation else {}
    )
    fresh_evaluations = dict(fresh_result.get("evaluations", {}) or {})
    fresh_blockers = list(fresh_result.get("blockers", []) or [])
    monitored_names = {factor.name for factor in monitored}
    fresh_names = set(fresh_evaluations)
    fresh_missing_factors = sorted(monitored_names - fresh_names)
    fresh_unexpected_factors = sorted(fresh_names - monitored_names)
    fresh_blockers.extend(
        f"fresh_evaluation_missing:{name}" for name in fresh_missing_factors
    )
    fresh_blockers.extend(
        f"fresh_evaluation_unexpected:{name}"
        for name in fresh_unexpected_factors
    )
    for name in sorted(monitored_names & fresh_names):
        fresh_blockers.extend(
            _fresh_evaluation_contract_blockers(
                name,
                fresh_evaluations[name],
            )
        )
    fresh_blockers = list(dict.fromkeys(str(item) for item in fresh_blockers))
    fresh_atomic_success = bool(
        args.fresh_evaluation
        and not fresh_blockers
        and fresh_names == monitored_names
    )
    decision_rows: list[dict[str, Any]] = []
    decisions = []

    for factor in monitored:
        monitor = dict((factor.metadata or {}).get("health_monitor", {}) or {})
        previous_failures = int(monitor.get("consecutive_failures", 0) or 0)
        if factor.name in fresh_evaluations:
            evaluation = fresh_evaluations[factor.name]
            source = "fresh_evaluation"
        elif args.strict_fresh_evaluation:
            evaluation = None
            source = "fresh_evaluation_missing"
        else:
            evaluation = registry_evaluations.get(factor.name)
            source = (
                "registry_evidence_fallback"
                if args.fresh_evaluation
                else "registry_evidence"
            )
            if evaluation is None:
                source = "missing"
        maturity_window_id = _maturity_window_id(evaluation)
        active_failure_windows = active_failure_maturity_window_ids(monitor)
        count_failure = maturity_window_id not in active_failure_windows
        current_evidence_end_date = _alpha_evidence_end_date(evaluation)
        last_evidence_end_date = str(
            monitor.get("last_alpha_evidence_end_date", "") or ""
        ).strip()
        chronology_blocker = _alpha_evidence_chronology_blocker(
            factor.name,
            current_evidence_end_date,
            last_evidence_end_date,
        )
        if chronology_blocker:
            fresh_blockers.append(chronology_blocker)
            count_failure = False
        decision = classify_factor_health(
            factor,
            evaluation,
            previous_failure_count=previous_failures,
            count_failure=count_failure,
        )
        decisions.append(decision)
        row = decision.to_dict()
        row["evaluation_source"] = source
        row.update(_evidence_age_fields(evaluation, timestamp))
        row["last_alpha_evidence_end_date"] = last_evidence_end_date
        row["evidence_chronology_status"] = (
            "blocked" if chronology_blocker else "ok"
        )
        decision_rows.append(row)

    fresh_blockers = list(dict.fromkeys(str(item) for item in fresh_blockers))
    if fresh_blockers:
        fresh_atomic_success = False

    data_blocked_factors = sorted(
        decision.factor_name
        for decision in decisions
        if decision.status.value == "data_blocked"
    )
    if args.strict_fresh_evaluation and data_blocked_factors:
        fresh_blockers.extend(
            f"fresh_evaluation_data_blocked:{name}"
            for name in data_blocked_factors
        )
        fresh_blockers = list(dict.fromkeys(fresh_blockers))
        fresh_atomic_success = False

    runtime_smoke = build_runtime_smoke(
        Path(args.data_root),
        args.universes,
        int(args.runtime_smoke_symbols),
        market=str(args.market),
        mode_policy=str(args.mode_policy),
    )
    runtime_smoke_blockers = _runtime_smoke_blockers(
        runtime_smoke,
        monitored_factor_count=len(monitored),
    )
    runtime_smoke_success = not runtime_smoke_blockers
    legacy_apply_requested = bool(args.apply_registry_actions)
    # FactorGovernanceProtocol v2 is the only production reconciliation
    # authority. Keep the legacy option observable but permanently incapable of
    # mutating records, even when its old fresh/runtime preconditions pass.
    registry_actions_eligible = False
    status_counts = Counter(decision.status.value for decision in decisions)
    action_counts = Counter(decision.action.value for decision in decisions)
    evaluation_source_counts = Counter(
        str(row.get("evaluation_source", "missing")) for row in decision_rows
    )
    evidence_age_summary = _evidence_age_summary(decision_rows)
    run_blocked = legacy_apply_requested or bool(
        args.strict_fresh_evaluation
        and (not fresh_atomic_success or not runtime_smoke_success)
    )
    registry_actions_applied = False
    registry_update_status = (
        "retired_protocol_v2_required"
        if legacy_apply_requested
        else "not_requested"
    )
    promotion_blockers = []
    if args.allow_production_promotion and int(args.max_new_production) > 0:
        promotion_blockers.append(
            "production promotion is intentionally disabled in health automation; "
            "production changes require the explicit FactorGovernanceProtocol v2 "
            "month-end targeted transition path"
        )

    payload = {
        "timestamp": timestamp,
        "cadence": args.cadence,
        "market": str(args.market).upper(),
        "mode_policy": str(args.mode_policy),
        "data_root": args.data_root,
        "universes": list(args.universes),
        "horizon_days": args.horizon_days,
        "warmup_days": args.warmup_days,
        "registry_path": str(registry_path),
        "apply_registry_actions": bool(args.apply_registry_actions),
        "registry_actions_applied": registry_actions_applied,
        "registry_actions_eligible": registry_actions_eligible,
        "registry_update_status": registry_update_status,
        "run_status": "blocked" if run_blocked else "ok",
        "allow_production_promotion": bool(args.allow_production_promotion),
        "registry_factor_count": len(registry.factors),
        "monitored_factor_count": len(monitored),
        "production_factor_count": len(monitored),
        "evaluated_factor_count": sum(
            1
            for row in decision_rows
            if row.get("evaluation_id") and row.get("evaluation_id") != "missing"
        ),
        "status_counts": dict(status_counts),
        "action_counts": dict(action_counts),
        "evaluation_source_counts": dict(evaluation_source_counts),
        "evidence_age_days": evidence_age_summary,
        "decisions": decision_rows,
        "qualified_new_candidates": [],
        "promoted_factors": [],
        "promotion_blockers": promotion_blockers,
        "fresh_evaluation": {
            "requested": bool(args.fresh_evaluation),
            "strict": bool(args.strict_fresh_evaluation),
            "atomic_success": fresh_atomic_success,
            "evaluated_factor_count": len(fresh_evaluations),
            "missing_factors": fresh_missing_factors,
            "unexpected_factors": fresh_unexpected_factors,
            "data_blocked_factors": data_blocked_factors,
            "blockers": fresh_blockers,
            "context": dict(fresh_result.get("context", {}) or {}),
        },
        "runtime_smoke": runtime_smoke,
        "runtime_smoke_blockers": runtime_smoke_blockers,
        "runtime_smoke_success": runtime_smoke_success,
        "registry_mutation_manifest": None,
        "registry_mutation_manifest_path": "",
        "registry_blockers": (
            [
                "legacy_apply_registry_actions_retired_use_"
                "factor_governance_protocol_v2"
            ]
            if legacy_apply_requested
            else []
        ),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    print(f"report_json={json_path}")
    print(f"report_markdown={md_path}")
    print(f"registry_factor_count={len(registry.factors)}")
    print(f"monitored_factor_count={len(monitored)}")
    print(f"status_counts={dict(status_counts)}")
    print(f"action_counts={dict(action_counts)}")
    smoke_mode = runtime_smoke.get("factor_mode", "unknown")
    smoke_count = runtime_smoke.get("factor_count", 0)
    smoke_coverage = runtime_smoke.get("coverage_rate", 0.0)
    print(
        "runtime_smoke="
        f"{smoke_mode} factor_count={smoke_count} coverage_rate={smoke_coverage}"
    )
    if run_blocked:
        print(f"fresh_evaluation_blockers={fresh_blockers}", file=sys.stderr)
        print(f"runtime_smoke_blockers={runtime_smoke_blockers}", file=sys.stderr)
        return 2
    return 0


def _monitored_factors(registry: MinedFactorRegistry) -> list[FactorRecord]:
    return [
        factor
        for factor in registry.factors
        if factor.state == FactorLifecycleState.PRODUCTION_FACTOR
        and not factor.deprecated_reason
    ]


def _evaluation_from_record(factor: FactorRecord) -> dict[str, Any] | None:
    if not factor.metrics and not factor.gate_results:
        return None
    decision = (
        factor.admission_decision.value
        if factor.admission_decision
        else FactorAdmissionDecision.PRODUCTION_CANDIDATE.value
        if factor.all_gates_passed()
        else ""
    )
    metrics = dict(factor.metrics or {})
    metrics.setdefault("horizon_days", int(factor.horizon_days))
    diagnostics = _diagnostics_from_record(factor, metrics)
    diagnostics["maturity_window_id"] = _build_maturity_window_id(
        metrics,
        diagnostics,
    )
    diagnostics["evaluation_hash"] = _build_evaluation_hash(
        factor.name,
        metrics,
        diagnostics,
    )
    diagnostics["evaluation_id"] = diagnostics["evaluation_hash"]
    return {
        "name": factor.name,
        "metrics": metrics,
        "review": {
            "decision": decision,
            "gate_results": [item.to_dict() for item in factor.gate_results],
        },
        "diagnostics": diagnostics,
    }


def _diagnostics_from_record(
    factor: FactorRecord,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(factor.metadata or {})
    evaluation_end_date = str(
        metrics.get("evaluation_end_date")
        or metadata.get("evaluation_end_date")
        or metadata.get("last_evaluation_end_date")
        or ""
    )
    rankic_count = (
        metrics.get("rankic_count")
        or metrics.get("rank_ic_count")
        or metadata.get("rankic_count")
        or ""
    )
    universes = (
        metadata.get("universes", [])
        or metadata.get("universe", [])
        or []
    )
    if isinstance(universes, str):
        universes = [universes]
    return {
        "evaluation_end_date": evaluation_end_date,
        "analysis_start_date": str(
            metrics.get("analysis_start_date")
            or metadata.get("analysis_start_date")
            or ""
        ),
        "rankic_count": rankic_count,
        "source_report": metadata.get("source_report", ""),
        "snapshot_id": str(
            metrics.get("snapshot_id") or metadata.get("snapshot_id") or ""
        ),
        "universes": list(universes),
        "decision_cost_bps": metrics.get(
            "decision_cost_bps", metadata.get("decision_cost_bps", "")
        ),
        "warmup_days": metrics.get(
            "warmup_days",
            metadata.get("warmup_days", ""),
        ),
        "implementation_hash": _implementation_hash(factor),
    }


def _fresh_evaluations(
    args: argparse.Namespace,
    factors: Sequence[FactorRecord],
) -> dict[str, Any]:
    blockers: list[str] = []
    evaluations: dict[str, dict[str, Any]] = {}
    try:
        from scripts.mine_quant_branch_factors import (  # type: ignore
            MiningCandidate,
            candidate_metrics,
            compute_price_volume_signal,
            evaluate_with_myquant_gate,
            restrict_context_to_analysis_window,
        )
    except Exception as exc:
        return {
            "evaluations": evaluations,
            "blockers": [f"fresh_evaluation_import_error:{exc}"],
        }

    context_result = _build_parquet_fresh_context(args)
    context = context_result.get("context")
    context_metadata = dict(context_result.get("metadata", {}) or {})
    blockers.extend(str(item) for item in context_result.get("blockers", []) or [])
    if context is None:
        return {
            "evaluations": evaluations,
            "blockers": blockers,
            "context": context_metadata,
        }

    try:
        context, _resolved_start = restrict_context_to_analysis_window(
            context,
            analysis_start_date=str(args.analysis_start_date),
            min_price_coverage=float(args.min_analysis_price_coverage),
        )
        context_metadata["analysis_start_date"] = _resolved_start
    except Exception as exc:
        return {
            "evaluations": evaluations,
            "blockers": [f"fresh_evaluation_context_error:{exc}"],
            "context": context_metadata,
        }

    context_metadata.update(
        {
            "analysis_start_date": _resolved_start,
            "evaluation_end_date": _latest_date(context.rebalance_dates),
            "horizon_days": int(args.horizon_days),
            "warmup_days": int(args.warmup_days),
            "decision_cost_bps": float(args.decision_cost_bps),
            "incremental_sleeve_weight": float(args.incremental_sleeve_weight),
            "existing_composite_mode": "leave_one_out_per_factor",
        }
    )
    active_registry = MinedFactorRegistry.from_records(list(factors))
    for factor in factors:
        try:
            candidate = _mining_candidate_from_record(factor, MiningCandidate)
            signal = compute_price_volume_signal(candidate, context)
            matured_cohort_dates = _mature_rankic_dates(
                signal,
                context.forward_return,
                context.rebalance_dates,
            )
            loo_existing, loo_blocker = _leave_one_out_existing_composite(
                context.existing_composite,
                signal,
                factor,
                active_registry,
            )
            metrics_context = replace(
                context,
                existing_composite=loo_existing,
                existing_blocker=loo_blocker or context.existing_blocker,
            )
            metrics = candidate_metrics(
                signal=signal,
                context=metrics_context,
                decision_cost_bps=float(args.decision_cost_bps),
                incremental_sleeve=float(args.incremental_sleeve_weight),
            )
            metrics["horizon_days"] = int(args.horizon_days)
            metrics["decision_cost_bps"] = float(args.decision_cost_bps)
            metrics["incremental_sleeve_weight"] = float(
                args.incremental_sleeve_weight
            )
            review = evaluate_with_myquant_gate(factor.name, metrics)
            evaluation_end_date = (
                matured_cohort_dates[-1] if matured_cohort_dates else ""
            )
            diagnostics = {
                **context_metadata,
                "evaluation_end_date": evaluation_end_date,
                "matured_cohort_dates": matured_cohort_dates,
                "rankic_count": metrics.get("rank_ic_count", ""),
                "excluded_factor": factor.name,
                "existing_composite_mode": "leave_one_out",
                "implementation_hash": _implementation_hash(
                    factor,
                    compute_price_volume_signal,
                    candidate_metrics,
                    evaluate_with_myquant_gate,
                    _mining_candidate_from_record,
                ),
            }
            diagnostics["maturity_window_id"] = _build_maturity_window_id(
                metrics,
                diagnostics,
            )
            diagnostics["evaluation_hash"] = _build_evaluation_hash(
                factor.name,
                metrics,
                diagnostics,
            )
            diagnostics["evaluation_id"] = diagnostics["evaluation_hash"]
            evaluations[factor.name] = {
                "name": factor.name,
                "metrics": metrics,
                "review": review.to_dict(),
                "diagnostics": diagnostics,
            }
            metric_blockers = [
                str(item)
                for item in metrics.get("blockers", []) or []
                if str(item)
            ]
            if bool(getattr(args, "strict_fresh_evaluation", False)):
                blockers.extend(
                    f"{factor.name}:fresh_evaluation_incomplete:{item}"
                    for item in metric_blockers
                )
        except Exception as exc:
            blockers.append(f"{factor.name}:{exc}")
    return {
        "evaluations": evaluations,
        "blockers": blockers,
        "context": context_metadata,
    }


def _build_parquet_fresh_context(args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "data_source": "parquet_canonical",
        "backend": "parquet",
        "market": str(getattr(args, "market", "CN") or "CN").strip().upper(),
        "mode_policy": str(
            getattr(args, "mode_policy", "strict") or "strict"
        ).strip().lower(),
        "data_root": str(Path(getattr(args, "data_root", "data")).expanduser()),
        "universes": list(getattr(args, "universes", []) or ["full_a"]),
        "symbols_requested": 0,
        "symbols_loaded": 0,
        "symbol_load_ratio": 0.0,
        "symbol_read_error_count": 0,
        "symbol_read_errors": [],
        "sample_symbols": [],
    }
    try:
        from quant_investor.market.market_data_reader import (
            MarketDataReader,
            MarketDataUnavailableError,
        )
        from scripts.retest_aquant_alpha_mix_8gate import (  # type: ignore
            RetestContext,
            build_price_matrices,
            forward_returns,
            rebalance_dates,
        )
        from scripts.mine_quant_branch_factors import (  # type: ignore
            MiningCandidate,
            compute_price_volume_signal,
        )
    except Exception as exc:
        return {
            "context": None,
            "metadata": metadata,
            "blockers": [f"parquet_fresh_context_import_error:{exc}"],
        }

    reader = MarketDataReader(
        market=metadata["market"],
        data_root=Path(metadata["data_root"]),
        mode_policy=metadata["mode_policy"],
    )
    try:
        snapshot = reader.snapshot()
    except MarketDataUnavailableError as exc:
        return {
            "context": None,
            "metadata": metadata,
            "blockers": [f"parquet_canonical_unavailable:{exc}"],
        }
    except Exception as exc:
        return {
            "context": None,
            "metadata": metadata,
            "blockers": [f"parquet_fresh_context_snapshot_error:{exc}"],
        }
    metadata.update(_snapshot_smoke_fields(snapshot))
    if not snapshot.get("healthy"):
        blockers = "; ".join(
            str(item) for item in snapshot.get("blockers", []) if str(item).strip()
        )
        return {
            "context": None,
            "metadata": metadata,
            "blockers": [
                f"parquet_canonical_unavailable:{blockers or 'strict Parquet snapshot is not healthy'}"
            ],
        }

    frames: dict[str, pd.DataFrame] = {}
    universe_by_symbol: dict[str, str] = {}
    requested_symbols: set[str] = set()
    symbol_read_errors: list[str] = []
    for universe in metadata["universes"]:
        try:
            symbols = reader.list_symbols(universe_key=str(universe or "full_a"))
        except Exception as exc:
            return {
                "context": None,
                "metadata": metadata,
                "blockers": [f"parquet_symbol_list_error:{universe}:{exc}"],
            }
        for symbol in symbols:
            normalized = str(symbol or "").strip().upper()
            if not normalized:
                continue
            requested_symbols.add(normalized)
            if normalized in frames:
                continue
            try:
                result = reader.read_symbol_frame(
                    normalized,
                    universe_key=str(universe or "full_a"),
                )
                frame = getattr(result, "frame", pd.DataFrame())
            except Exception as exc:
                symbol_read_errors.append(f"{normalized}:{exc}")
                continue
            if frame is None or frame.empty:
                symbol_read_errors.append(f"{normalized}:empty_frame")
                continue
            working = frame.copy()
            if "symbol" not in working.columns:
                working["symbol"] = normalized
            if "ts_code" not in working.columns:
                working["ts_code"] = normalized
            frames[normalized] = working
            universe_by_symbol[normalized] = str(universe or "full_a")

    metadata["symbols_requested"] = len(requested_symbols)
    metadata["symbols_loaded"] = len(frames)
    metadata["symbol_load_ratio"] = (
        len(frames) / len(requested_symbols)
        if requested_symbols
        else 0.0
    )
    metadata["symbol_read_error_count"] = len(symbol_read_errors)
    metadata["symbol_read_errors"] = symbol_read_errors[:20]
    metadata["sample_symbols"] = list(frames)[:5]
    if not frames:
        return {
            "context": None,
            "metadata": metadata,
            "blockers": ["parquet_fresh_context_no_frames"],
        }
    min_load_ratio = float(
        getattr(args, "min_analysis_price_coverage", 0.95) or 0.95
    )
    if float(metadata["symbol_load_ratio"]) < min_load_ratio:
        return {
            "context": None,
            "metadata": metadata,
            "blockers": [
                "parquet_fresh_context_symbol_load_ratio:"
                f"actual={metadata['symbol_load_ratio']:.6f}:"
                f"minimum={min_load_ratio:.6f}:"
                f"read_errors={len(symbol_read_errors)}"
            ],
        }

    try:
        adj_close, volume, amount = build_price_matrices(frames)
        forward = forward_returns(adj_close, int(args.horizon_days))
        monthly, biweekly = rebalance_dates(
            adj_close.index,
            int(args.warmup_days),
            int(args.horizon_days),
        )
    except Exception as exc:
        return {
            "context": None,
            "metadata": metadata,
            "blockers": [f"parquet_fresh_context_matrix_error:{exc}"],
        }

    existing = None
    existing_blocker = ""
    try:
        registry_path = Path(
            getattr(
                args,
                "registry_path",
                "quant_investor/factor_registry/mined_factors.json",
            )
        )
        registry = MinedFactorRegistry.load(registry_path)
        existing, existing_blocker = _compute_existing_price_volume_composite(
            registry,
            adj_close,
            volume,
            amount,
            candidate_type=MiningCandidate,
            signal_builder=compute_price_volume_signal,
        )
    except Exception as exc:
        existing_blocker = f"existing_composite_unavailable:{exc}"

    metadata["existing_composite_blocker"] = existing_blocker
    return {
        "context": RetestContext(
            frames=frames,
            universe_by_symbol=universe_by_symbol,
            adj_close=adj_close,
            volume=volume,
            amount=amount,
            forward_return=forward,
            rebalance_dates=monthly,
            biweekly_dates=biweekly,
            existing_composite=existing,
            existing_blocker=existing_blocker,
        ),
        "metadata": metadata,
        "blockers": [],
    }


def _compute_existing_price_volume_composite(
    registry: MinedFactorRegistry,
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
    *,
    candidate_type: Any,
    signal_builder: Any,
) -> tuple[pd.DataFrame | None, str]:
    active = registry.selectable_factors()
    if not active:
        return None, "no_selectable_production_factors"
    signal_context = SimpleNamespace(
        adj_close=adj_close,
        volume=volume,
        amount=amount,
    )
    composite = pd.DataFrame(0.0, index=adj_close.index, columns=adj_close.columns)
    total_weight = 0.0
    blockers: list[str] = []
    for factor in active:
        try:
            candidate = _mining_candidate_from_record(factor, candidate_type)
            raw = signal_builder(candidate, signal_context)
        except Exception as exc:
            blockers.append(f"{factor.name}:{exc}")
            continue
        weight = float(factor.weight) * (
            1.0 if float(getattr(factor, "direction", 1.0)) >= 0 else -1.0
        )
        ranked = raw.rank(axis=1, pct=True).mul(2.0).sub(1.0)
        composite = composite.add(ranked.fillna(0.0).mul(weight), fill_value=0.0)
        total_weight += abs(weight)
    if blockers:
        return None, "unsupported_existing_price_volume_factor:" + ";".join(blockers)
    if total_weight <= 1e-12:
        return None, "zero_existing_factor_weight"
    return composite.div(total_weight).clip(-1.0, 1.0), ""


def _leave_one_out_existing_composite(
    existing_composite: pd.DataFrame | None,
    candidate_signal: pd.DataFrame,
    factor: FactorRecord,
    registry: MinedFactorRegistry,
) -> tuple[pd.DataFrame | None, str]:
    """Remove ``factor`` from its Gate 8/correlation baseline."""

    active = {item.name: item for item in registry.selectable_factors()}
    selected = active.get(factor.name)
    if selected is None:
        return existing_composite, ""
    if existing_composite is None or existing_composite.empty:
        return None, "leave_one_out_existing_composite_unavailable"

    total_weight = sum(abs(float(item.weight)) for item in active.values())
    selected_weight = abs(float(selected.weight))
    remaining_weight = total_weight - selected_weight
    if remaining_weight <= 1e-12:
        return None, "leave_one_out_no_remaining_production_factors"

    signed_weight = float(selected.weight) * (
        1.0 if float(getattr(selected, "direction", 1.0)) >= 0 else -1.0
    )
    ranked = candidate_signal.rank(axis=1, pct=True).mul(2.0).sub(1.0)
    numerator = existing_composite.mul(total_weight).sub(
        ranked.fillna(0.0).mul(signed_weight),
        fill_value=0.0,
    )
    return numerator.div(remaining_weight).clip(-1.0, 1.0), ""


def _mature_rankic_dates(
    signal: pd.DataFrame,
    forward_return: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
) -> list[str]:
    matured: list[str] = []
    for date in dates:
        if date not in signal.index or date not in forward_return.index:
            continue
        pair = pd.concat(
            [
                signal.loc[date].rename("signal"),
                forward_return.loc[date].rename("return"),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if (
            len(pair) < 20
            or pair["signal"].nunique(dropna=True) <= 1
            or pair["return"].nunique(dropna=True) <= 1
        ):
            continue
        matured.append(pd.Timestamp(date).strftime("%Y-%m-%d"))
    return matured


def _mining_candidate_from_record(factor: FactorRecord, candidate_type: Any) -> Any:
    impl = str(factor.implementation or "").strip()
    if not impl.startswith("price_volume:"):
        raise ValueError(f"fresh evaluation supports price_volume factors only: {impl}")
    name = impl.split(":", 1)[1]
    if name.startswith("pv_short_reversal_"):
        family = "short_reversal"
    elif name.startswith("pv_volume_stability_smooth_"):
        family = "volume_stability_smooth"
    elif name.startswith("pv_volume_stability_"):
        family = "volume_stability"
    elif name.startswith("pv_low_dollar_volume_"):
        family = "low_dollar_volume"
    elif name.startswith("pv_amihud_illiquidity_"):
        family = "amihud_illiquidity"
    elif name.startswith("pv_blend_volstab19x2_mom90_amihud5_w"):
        family = "volstab_momentum_illiquidity_blend"
        weight_text = name.rsplit("_w", 1)[-1]
        outer_weight = float(weight_text) / 100.0
        return candidate_type(
            name=factor.name,
            family=family,
            category=factor.category,
            implementation=impl,
            description=factor.description,
            window=90,
            params={
                "volume_stability_base_window": 19,
                "volume_stability_smooth_window": 2,
                "momentum_window": 90,
                "amihud_window": 5,
                "outer_volume_stability_weight": outer_weight,
                "inner_momentum_weight": 0.60,
            },
        )
    else:
        raise ValueError(f"unsupported price_volume factor: {name}")
    return candidate_type(
        name=factor.name,
        family=family,
        category=factor.category,
        implementation=impl,
        description=factor.description,
        window=_first_window_from_name(name),
    )


def _first_window_from_name(name: str) -> int:
    for part in reversed(str(name).split("_")):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 20


def build_runtime_smoke(
    data_root: Path,
    universes: Sequence[str],
    sample_size: int,
    *,
    market: str = "CN",
    mode_policy: str = "strict",
) -> dict[str, Any]:
    base = {
        "data_source": "parquet_canonical",
        "backend": "parquet",
        "fallback_used": False,
        "market": str(market or "").strip().upper() or "CN",
        "mode_policy": str(mode_policy or "strict").strip().lower() or "strict",
        "data_root": str(data_root),
    }
    try:
        from quant_investor.market.dag.packets import _build_quant_branch_result
        from quant_investor.market.market_data_reader import (
            MarketDataReader,
            MarketDataUnavailableError,
        )
    except Exception as exc:
        return {
            **base,
            "factor_mode": "parquet_runtime_unavailable",
            "factor_count": 0,
            "coverage_rate": 0.0,
            "symbols": 0,
            "symbols_requested": 0,
            "symbols_loaded": 0,
            "error": f"strict Parquet runtime imports unavailable: {exc}",
        }

    reader = MarketDataReader(
        market=base["market"],
        data_root=data_root,
        mode_policy=base["mode_policy"],
    )
    try:
        snapshot = reader.snapshot()
        if not snapshot.get("healthy"):
            blockers = list(snapshot.get("blockers", []) or [])
            return {
                **base,
                **_snapshot_smoke_fields(snapshot),
                "factor_mode": "parquet_canonical_unavailable",
                "factor_count": 0,
                "coverage_rate": 0.0,
                "symbols": 0,
                "symbols_requested": 0,
                "symbols_loaded": 0,
                "error": "; ".join(str(item) for item in blockers)
                or "strict Parquet snapshot is not healthy",
            }
    except MarketDataUnavailableError as exc:
        return {
            **base,
            "factor_mode": "parquet_canonical_unavailable",
            "factor_count": 0,
            "coverage_rate": 0.0,
            "symbols": 0,
            "symbols_requested": 0,
            "symbols_loaded": 0,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            **base,
            "factor_mode": "error",
            "factor_count": 0,
            "coverage_rate": 0.0,
            "symbols": 0,
            "symbols_requested": 0,
            "symbols_loaded": 0,
            "error": str(exc),
        }

    frames: dict[str, pd.DataFrame] = {}
    symbols_requested = 0
    for universe in list(universes or ["full_a"]):
        try:
            symbols = reader.list_symbols(universe_key=str(universe or "full_a"))
        except MarketDataUnavailableError as exc:
            return {
                **base,
                **_snapshot_smoke_fields(snapshot),
                "factor_mode": "parquet_canonical_unavailable",
                "factor_count": 0,
                "coverage_rate": 0.0,
                "symbols": 0,
                "symbols_requested": symbols_requested,
                "symbols_loaded": len(frames),
                "error": str(exc),
            }
        symbols_requested += len(symbols)
        for symbol in symbols:
            if len(frames) >= max(int(sample_size), 1):
                break
            try:
                result = reader.read_symbol_frame(str(symbol))
                frame = getattr(result, "frame", pd.DataFrame())
            except Exception:
                continue
            if frame is None or frame.empty:
                continue
            frames[str(symbol)] = frame.tail(420)
        if len(frames) >= max(int(sample_size), 1):
            break
    if not frames:
        return {
            **base,
            **_snapshot_smoke_fields(snapshot),
            "factor_mode": "parquet_canonical_unavailable",
            "factor_count": 0,
            "coverage_rate": 0.0,
            "symbols": 0,
            "symbols_requested": symbols_requested,
            "symbols_loaded": 0,
            "error": "no Parquet serving frames loaded from strict MarketDataReader",
        }
    try:
        result = _build_quant_branch_result(frames=frames)
        runtime = result.metadata.get("mined_factor_runtime", {}) or {}
        return {
            **base,
            **_snapshot_smoke_fields(snapshot),
            "factor_mode": result.metadata.get("factor_mode", ""),
            "factor_count": runtime.get("factor_count", 0),
            "coverage_rate": runtime.get("coverage_rate", 0.0),
            "symbols": len(result.symbol_scores),
            "symbols_requested": symbols_requested,
            "symbols_loaded": len(frames),
        }
    except Exception as exc:
        return {
            **base,
            **_snapshot_smoke_fields(snapshot),
            "factor_mode": "error",
            "factor_count": 0,
            "coverage_rate": 0.0,
            "symbols": len(frames),
            "symbols_requested": symbols_requested,
            "symbols_loaded": len(frames),
            "error": str(exc),
        }


def _snapshot_smoke_fields(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.get("snapshot_id", ""),
        "latest_complete_trade_date": snapshot.get("latest_complete_trade_date", ""),
        "latest_trade_date": snapshot.get("latest_trade_date", ""),
        "latest_pointer_path": snapshot.get("latest_pointer_path", ""),
        "table_root": snapshot.get("table_root", ""),
        "serving_root": snapshot.get("serving_root", ""),
        "manifest_path": snapshot.get("manifest_path", ""),
        "snapshot_status": snapshot.get("status", ""),
        "snapshot_healthy": bool(snapshot.get("healthy", False)),
    }


def _runtime_smoke_blockers(
    runtime_smoke: Mapping[str, Any],
    *,
    monitored_factor_count: int,
) -> list[str]:
    blockers: list[str] = []
    backend = str(runtime_smoke.get("backend", "") or "").strip().lower()
    if backend != "parquet":
        blockers.append(f"runtime_smoke_backend:{backend or 'missing'}")
    mode_policy = str(
        runtime_smoke.get("mode_policy", "") or ""
    ).strip().lower()
    if mode_policy != "strict":
        blockers.append(
            f"runtime_smoke_mode_policy:{mode_policy or 'missing'}"
        )
    if runtime_smoke.get("fallback_used") is not False:
        blockers.append("runtime_smoke_fallback_used")
    error = str(runtime_smoke.get("error", "") or "").strip()
    if error:
        blockers.append(f"runtime_smoke_error:{error}")
    if not bool(runtime_smoke.get("snapshot_healthy", False)):
        blockers.append("runtime_smoke_snapshot_unhealthy")
    factor_mode = str(runtime_smoke.get("factor_mode", "") or "")
    if factor_mode != "governed_mined_factors":
        blockers.append(f"runtime_smoke_factor_mode:{factor_mode or 'missing'}")
    try:
        factor_count = int(runtime_smoke.get("factor_count", 0) or 0)
    except (TypeError, ValueError):
        factor_count = -1
    if factor_count != int(monitored_factor_count):
        blockers.append(
            "runtime_smoke_factor_count:"
            f"expected={int(monitored_factor_count)}:actual={factor_count}"
        )
    try:
        symbols_loaded = int(runtime_smoke.get("symbols_loaded", 0) or 0)
    except (TypeError, ValueError):
        symbols_loaded = 0
    if symbols_loaded <= 0:
        blockers.append("runtime_smoke_no_symbols_loaded")
    return blockers


def _fresh_evaluation_contract_blockers(
    factor_name: str,
    evaluation: Any,
) -> list[str]:
    prefix = f"{factor_name}:fresh_evaluation_contract"
    if not isinstance(evaluation, Mapping):
        return [f"{prefix}:evaluation_not_object"]

    blockers: list[str] = []
    evaluation_name = str(evaluation.get("name", "") or "").strip()
    if evaluation_name != factor_name:
        blockers.append(
            f"{prefix}:name_mismatch:{evaluation_name or 'missing'}"
        )

    metrics = evaluation.get("metrics")
    required_metrics = {
        "coverage_rate",
        "nan_rate",
        "icir",
        "positive_ic_ratio",
        "oos_positive_ratio",
        "neutralized_icir",
        "top_bottom_spread",
        "cost_adjusted_return",
        "turnover",
        "capacity_pressure",
    }
    if not isinstance(metrics, Mapping):
        blockers.append(f"{prefix}:metrics_not_object")
    else:
        missing_metrics = sorted(required_metrics - set(metrics))
        if missing_metrics:
            blockers.append(
                f"{prefix}:metrics_missing:{','.join(missing_metrics)}"
            )
        for key in sorted(required_metrics & set(metrics)):
            try:
                value = float(metrics[key])
            except (TypeError, ValueError):
                blockers.append(f"{prefix}:metric_not_numeric:{key}")
                continue
            if not np.isfinite(value):
                blockers.append(f"{prefix}:metric_not_finite:{key}")

    review = evaluation.get("review")
    if not isinstance(review, Mapping):
        blockers.append(f"{prefix}:review_not_object")
    else:
        if not str(review.get("decision", "") or "").strip():
            blockers.append(f"{prefix}:review_decision_missing")
        gate_results = review.get("gate_results")
        gate_ids: list[int] = []
        if not isinstance(gate_results, list):
            blockers.append(f"{prefix}:gate_results_not_list")
        else:
            for index, gate in enumerate(gate_results):
                if not isinstance(gate, Mapping):
                    blockers.append(
                        f"{prefix}:gate_result_not_object:{index}"
                    )
                    continue
                try:
                    gate_ids.append(int(gate.get("gate_id", 0) or 0))
                except (TypeError, ValueError):
                    blockers.append(f"{prefix}:gate_id_invalid:{index}")
                if not isinstance(gate.get("passed"), bool):
                    blockers.append(f"{prefix}:gate_passed_not_bool:{index}")
            if sorted(gate_ids) != list(range(1, 9)):
                blockers.append(
                    f"{prefix}:gate_ids_expected_1_to_8:actual="
                    f"{','.join(str(item) for item in sorted(gate_ids))}"
                )

    diagnostics = evaluation.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        blockers.append(f"{prefix}:diagnostics_not_object")
    else:
        if not str(diagnostics.get("maturity_window_id", "") or "").strip():
            blockers.append(f"{prefix}:maturity_window_id_missing")
        if not str(
            diagnostics.get("evaluation_hash")
            or diagnostics.get("evaluation_id")
            or ""
        ).strip():
            blockers.append(f"{prefix}:evaluation_hash_missing")
        evidence_end_date = _alpha_evidence_end_date(evaluation)
        if not evidence_end_date:
            blockers.append(f"{prefix}:evidence_end_date_missing")
        else:
            try:
                pd.Timestamp(evidence_end_date)
            except (TypeError, ValueError):
                blockers.append(f"{prefix}:evidence_end_date_invalid")
    return blockers


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Governed Factor Health Automation",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Cadence: {payload['cadence']}",
        f"- Registry factors: {payload['registry_factor_count']}",
        f"- Monitored production factors: {payload['monitored_factor_count']}",
        f"- Evaluated factors: {payload['evaluated_factor_count']}",
        f"- Run status: {payload.get('run_status', 'ok')}",
        f"- Registry actions requested: {payload['apply_registry_actions']}",
        (
            "- Registry actions applied: "
            f"{payload.get('registry_actions_applied', False)}"
        ),
        f"- Evaluation sources: {payload.get('evaluation_source_counts', {})}",
        f"- Evidence age days: {payload.get('evidence_age_days', {})}",
        f"- Runtime smoke: {payload['runtime_smoke']}",
        "",
        "## Production Factor Decisions",
        "",
        (
            "| Factor | Source | Status | Action | Failures | ICIR | "
            "Positive IC | OOS | Spread | Turnover | New Weight |"
        ),
        (
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |"
        ),
    ]
    for item in payload["decisions"]:
        metrics = item.get("health_metrics", {})
        lines.append(
            "| {name} | {source} | {status} | {action} | {failures} | {icir} | "
            "{pos} | {oos} | {spread} | {turnover} | {weight:.4f} |".format(
                name=item["factor_name"],
                source=item.get("evaluation_source", ""),
                status=item["status"],
                action=item["action"],
                failures=item["consecutive_failures"],
                icir=_fmt_float(metrics.get("icir")),
                pos=_fmt_pct(metrics.get("positive_ic_ratio")),
                oos=_fmt_pct(metrics.get("oos_positive_ratio")),
                spread=_fmt_pct(metrics.get("top_bottom_spread")),
                turnover=_fmt_float(metrics.get("turnover")),
                weight=float(item["new_weight"]),
            )
        )
    lines.extend(["", "## Qualified New Candidates", ""])
    candidates = payload.get("qualified_new_candidates", [])
    if not candidates:
        lines.append("- None.")
    else:
        for item in candidates:
            lines.append(f"- {item}")
    if payload.get("promotion_blockers"):
        lines.extend(["", "## Promotion Blockers", ""])
        for blocker in payload["promotion_blockers"]:
            lines.append(f"- {blocker}")
    fresh = payload.get("fresh_evaluation", {}) or {}
    if fresh.get("requested") or fresh.get("blockers"):
        lines.extend(["", "## Fresh Evaluation", ""])
        lines.append(f"- Requested: {fresh.get('requested')}")
        lines.append(f"- Strict: {fresh.get('strict')}")
        lines.append(f"- Atomic success: {fresh.get('atomic_success')}")
        lines.append(f"- Evaluated factors: {fresh.get('evaluated_factor_count')}")
        if fresh.get("missing_factors"):
            lines.append(f"- Missing factors: {fresh.get('missing_factors')}")
        if fresh.get("data_blocked_factors"):
            lines.append(
                f"- Data-blocked factors: {fresh.get('data_blocked_factors')}"
            )
        context = fresh.get("context", {}) or {}
        if context:
            lines.append(f"- Context: {context}")
        blockers = fresh.get("blockers", []) or []
        if blockers:
            lines.append("- Blockers:")
            for blocker in blockers:
                lines.append(f"  - {blocker}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Scheduled runs should normally omit `--allow-production-promotion`.",
            (
                "- Default health classification uses local registry evidence "
                "and a strict Parquet MarketDataReader runtime smoke."
            ),
            (
                "- The runtime smoke reads `_latest.json` and Parquet serving "
                "files; it does not scan legacy CSV daily directories."
            ),
            (
                "- Repeated runs over the same matured evaluation window are "
                "observed but not double-counted as new failures."
            ),
            (
                "- Registry actions require an atomic `--fresh-evaluation` "
                "plus `--strict-fresh-evaluation`; registry evidence is "
                "report-only."
            ),
            (
                "- Data-blocked observations do not increment alpha-failure "
                "streaks or change production weights."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluation_id(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return "missing"
    diagnostics = evaluation.get("diagnostics", {}) or {}
    explicit = str(
        diagnostics.get("evaluation_hash")
        or diagnostics.get("evaluation_id")
        or ""
    )
    if explicit:
        return explicit
    return _build_evaluation_hash(
        str(evaluation.get("name", "") or "unknown"),
        evaluation.get("metrics", {}) or {},
        diagnostics,
    )


def _maturity_window_id(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return "missing"
    diagnostics = evaluation.get("diagnostics", {}) or {}
    explicit = str(diagnostics.get("maturity_window_id", "") or "")
    if explicit:
        return explicit
    return _build_maturity_window_id(
        evaluation.get("metrics", {}) or {},
        diagnostics,
    )


def _build_maturity_window_id(
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    dates = [
        str(item)
        for item in diagnostics.get("matured_cohort_dates", []) or []
        if str(item)
    ]
    end_date = str(
        diagnostics.get("evaluation_end_date")
        or (dates[-1] if dates else "")
        or ""
    )
    horizon = _id_value(metrics.get("horizon_days", ""))
    rankic_count = _id_value(diagnostics.get("rankic_count", len(dates)))
    cohort_hash = hashlib.sha256(
        ",".join(dates).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"end={end_date}|h={horizon}|n={rankic_count}|cohorts={cohort_hash}"
    )


def _build_evaluation_hash(
    factor_name: str,
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    universes = diagnostics.get("universes", []) or []
    if isinstance(universes, str):
        universes = [universes]
    payload = {
        "factor_name": str(factor_name or "unknown"),
        "snapshot_id": str(diagnostics.get("snapshot_id", "") or ""),
        "universes": sorted({str(item) for item in universes if str(item)}),
        "analysis_start_date": str(
            diagnostics.get("analysis_start_date", "") or "full"
        ),
        "evaluation_end_date": str(
            diagnostics.get("evaluation_end_date", "") or ""
        ),
        "horizon_days": _id_value(metrics.get("horizon_days", "")),
        "warmup_days": _id_value(diagnostics.get("warmup_days", "")),
        "decision_cost_bps": _id_value(
            diagnostics.get("decision_cost_bps", "")
        ),
        "incremental_sleeve_weight": _id_value(
            diagnostics.get("incremental_sleeve_weight", "")
        ),
        "implementation_hash": str(
            diagnostics.get("implementation_hash", "") or ""
        ),
        "maturity_window_id": str(
            diagnostics.get("maturity_window_id")
            or _build_maturity_window_id(metrics, diagnostics)
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _implementation_hash(factor: FactorRecord, *callables: Any) -> str:
    metadata = dict(factor.metadata or {})
    identity = {
        "name": factor.name,
        "version": factor.version,
        "implementation": factor.implementation,
        "direction": float(factor.direction),
        "horizon_days": int(factor.horizon_days),
        "spec": metadata.get("spec", {}),
        "implementation_params": metadata.get("implementation_params", {}),
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    for callable_obj in callables:
        try:
            source = inspect.getsource(callable_obj)
        except (OSError, TypeError):
            source = (
                f"{getattr(callable_obj, '__module__', '')}:"
                f"{getattr(callable_obj, '__qualname__', repr(callable_obj))}"
            )
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
    return digest.hexdigest()[:16]


def _id_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value or "")


def _alpha_evidence_end_date(
    evaluation: Mapping[str, Any] | None,
) -> str:
    diagnostics = (evaluation or {}).get("diagnostics", {}) or {}
    explicit = str(diagnostics.get("evaluation_end_date", "") or "").strip()
    if explicit:
        return explicit
    cohort_dates = [
        str(item).strip()
        for item in diagnostics.get("matured_cohort_dates", []) or []
        if str(item).strip()
    ]
    return cohort_dates[-1] if cohort_dates else ""


def _alpha_evidence_chronology_blocker(
    factor_name: str,
    current_end_date: str,
    last_end_date: str,
) -> str:
    if not last_end_date:
        return ""
    if not current_end_date:
        return (
            f"fresh_evaluation_alpha_chronology:{factor_name}:"
            f"current_end=missing:last_end={last_end_date}"
        )
    try:
        current = pd.Timestamp(current_end_date).tz_localize(None).normalize()
        previous = pd.Timestamp(last_end_date).tz_localize(None).normalize()
    except (TypeError, ValueError):
        return (
            f"fresh_evaluation_alpha_chronology:{factor_name}:"
            f"current_end={current_end_date}:last_end={last_end_date}:"
            "unparseable"
        )
    if current < previous:
        return (
            f"fresh_evaluation_alpha_chronology:{factor_name}:"
            f"current_end={current_end_date}:last_end={last_end_date}:"
            "regressed"
        )
    return ""


def _evidence_age_fields(
    evaluation: Mapping[str, Any] | None,
    observed_at: str,
) -> dict[str, Any]:
    diagnostics = (evaluation or {}).get("diagnostics", {}) or {}
    end_date = _alpha_evidence_end_date(evaluation)
    age_days: int | None = None
    if end_date:
        try:
            observed = pd.Timestamp(observed_at).tz_localize(None).normalize()
            evidence_date = (
                pd.Timestamp(end_date).tz_localize(None).normalize()
            )
            age_days = int((observed - evidence_date).days)
        except (TypeError, ValueError):
            age_days = None
    return {
        "evidence_end_date": end_date,
        "evidence_age_days": age_days,
        "evidence_snapshot_id": str(diagnostics.get("snapshot_id", "") or ""),
    }


def _evidence_age_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known = [
        int(row["evidence_age_days"])
        for row in rows
        if row.get("evidence_age_days") is not None
    ]
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted(
        {str(row.get("evaluation_source", "missing")) for row in rows}
    ):
        source_rows = [
            row
            for row in rows
            if str(row.get("evaluation_source", "missing")) == source
        ]
        source_ages = [
            int(row["evidence_age_days"])
            for row in source_rows
            if row.get("evidence_age_days") is not None
        ]
        by_source[source] = {
            "count": len(source_rows),
            "known_count": len(source_ages),
            "unknown_count": len(source_rows) - len(source_ages),
            "min_age_days": min(source_ages) if source_ages else None,
            "max_age_days": max(source_ages) if source_ages else None,
        }
    return {
        "known_count": len(known),
        "unknown_count": len(rows) - len(known),
        "min_age_days": min(known) if known else None,
        "max_age_days": max(known) if known else None,
        "by_source": by_source,
    }


def _latest_date(dates: Sequence[pd.Timestamp]) -> str:
    if not dates:
        return ""
    latest = max(pd.Timestamp(item) for item in dates)
    return latest.strftime("%Y-%m-%d")


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return ""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
