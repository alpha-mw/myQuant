#!/usr/bin/env python3
"""Periodic health checks for governed mined factors.

The default run is offline and report-only.  It reads the local mined-factor
registry, uses the registry's approved 8-gate evidence for production-factor
health classification, and runs a local runtime smoke check.  Registry writes
only occur when ``--apply-registry-actions`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
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
    apply_health_decision,
    classify_factor_health,
)
from quant_investor.factors.runtime import MinedFactorRegistry  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cadence", choices=["daily", "weekly", "monthly", "review"], default="daily"
    )
    parser.add_argument("--data-root", default="data/clean/cn_daily")
    parser.add_argument("--universes", nargs="+", default=["hs300", "zz500", "zz1000"])
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
            "Apply conservative de-risking actions such as metadata updates, "
            "weight reduction, or deprecation."
        ),
    )
    parser.add_argument(
        "--fresh-evaluation",
        action="store_true",
        help=(
            "Best-effort local re-evaluation of monitored price/volume factors. "
            "If unavailable, the run falls back to registry evidence."
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"factor_health_{timestamp.replace(':', '')}.json"
    md_path = json_path.with_suffix(".md")

    registry_path = Path(args.registry_path)
    registry = MinedFactorRegistry.load(registry_path)
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
    decision_rows: list[dict[str, Any]] = []
    decisions = []

    for factor in monitored:
        monitor = dict((factor.metadata or {}).get("health_monitor", {}) or {})
        previous_failures = int(monitor.get("consecutive_failures", 0) or 0)
        evaluation = fresh_evaluations.get(factor.name) or registry_evaluations.get(
            factor.name
        )
        source = (
            "fresh_evaluation"
            if factor.name in fresh_evaluations
            else "registry_evidence"
        )
        count_failure = _evaluation_id(evaluation) != str(
            monitor.get("last_evaluation_id", "") or ""
        )
        decision = classify_factor_health(
            factor,
            evaluation,
            previous_failure_count=previous_failures,
            count_failure=count_failure,
        )
        decisions.append(decision)
        row = decision.to_dict()
        row["evaluation_source"] = source
        decision_rows.append(row)
        if args.apply_registry_actions:
            apply_health_decision(
                factor,
                decision,
                reviewed_at=timestamp,
                report_path=str(md_path),
            )

    runtime_smoke = build_runtime_smoke(
        Path(args.data_root),
        args.universes,
        int(args.runtime_smoke_symbols),
    )
    status_counts = Counter(decision.status.value for decision in decisions)
    action_counts = Counter(decision.action.value for decision in decisions)
    promotion_blockers = []
    if args.allow_production_promotion and int(args.max_new_production) > 0:
        promotion_blockers.append(
            "production promotion is intentionally disabled in health automation; "
            "promote production_factor records through manual registry review"
        )

    payload = {
        "timestamp": timestamp,
        "cadence": args.cadence,
        "data_root": args.data_root,
        "universes": list(args.universes),
        "horizon_days": args.horizon_days,
        "warmup_days": args.warmup_days,
        "registry_path": str(registry_path),
        "apply_registry_actions": bool(args.apply_registry_actions),
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
        "decisions": decision_rows,
        "qualified_new_candidates": [],
        "promoted_factors": [],
        "promotion_blockers": promotion_blockers,
        "fresh_evaluation": {
            "requested": bool(args.fresh_evaluation),
            "evaluated_factor_count": len(fresh_evaluations),
            "blockers": fresh_blockers,
        },
        "runtime_smoke": runtime_smoke,
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    if args.apply_registry_actions:
        registry.metadata.update(
            {
                "last_factor_health_review_at": timestamp,
                "last_factor_health_report_json": str(json_path),
                "last_factor_health_report_markdown": str(md_path),
                "last_factor_health_action_counts": dict(action_counts),
            }
        )
        write_registry(registry_path, registry)

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
    if args.strict_fresh_evaluation and args.fresh_evaluation and fresh_blockers:
        print(f"fresh_evaluation_blockers={fresh_blockers}", file=sys.stderr)
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
    return {
        "name": factor.name,
        "metrics": metrics,
        "review": {
            "decision": decision,
            "gate_results": [item.to_dict() for item in factor.gate_results],
        },
        "diagnostics": _diagnostics_from_record(factor, metrics),
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
    return {
        "evaluation_end_date": evaluation_end_date,
        "rankic_count": rankic_count,
        "source_report": metadata.get("source_report", ""),
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
            build_context,
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

    try:
        context = build_context(
            data_root=Path(args.data_root).expanduser(),
            universes=tuple(args.universes),
            horizon_days=int(args.horizon_days),
            warmup_days=int(args.warmup_days),
        )
        context, _resolved_start = restrict_context_to_analysis_window(
            context,
            analysis_start_date=str(args.analysis_start_date),
            min_price_coverage=float(args.min_analysis_price_coverage),
        )
    except Exception as exc:
        return {
            "evaluations": evaluations,
            "blockers": [f"fresh_evaluation_context_error:{exc}"],
        }

    for factor in factors:
        try:
            candidate = _mining_candidate_from_record(factor, MiningCandidate)
            signal = compute_price_volume_signal(candidate, context)
            metrics = candidate_metrics(
                signal=signal,
                context=context,
                decision_cost_bps=float(args.decision_cost_bps),
                incremental_sleeve=float(args.incremental_sleeve_weight),
            )
            review = evaluate_with_myquant_gate(factor.name, metrics)
            evaluations[factor.name] = {
                "name": factor.name,
                "metrics": metrics,
                "review": review.to_dict(),
                "diagnostics": {
                    "evaluation_end_date": _latest_date(context.rebalance_dates),
                    "rankic_count": metrics.get("rank_ic_count", ""),
                },
            }
        except Exception as exc:
            blockers.append(f"{factor.name}:{exc}")
    return {"evaluations": evaluations, "blockers": blockers}


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
) -> dict[str, Any]:
    frames: dict[str, pd.DataFrame] = {}
    for universe in universes:
        for path in sorted((data_root / str(universe)).glob("*.csv")):
            if len(frames) >= max(sample_size, 1):
                break
            try:
                frames[path.stem] = pd.read_csv(path).tail(420)
            except Exception:
                continue
        if len(frames) >= max(sample_size, 1):
            break
    if not frames:
        return {
            "factor_mode": "unavailable",
            "factor_count": 0,
            "coverage_rate": 0.0,
            "symbols": 0,
            "error": f"no local CSV frames found under {data_root}",
        }
    try:
        from quant_investor.market.dag.packets import _build_quant_branch_result

        result = _build_quant_branch_result(frames=frames)
        runtime = result.metadata.get("mined_factor_runtime", {}) or {}
        return {
            "factor_mode": result.metadata.get("factor_mode", ""),
            "factor_count": runtime.get("factor_count", 0),
            "coverage_rate": runtime.get("coverage_rate", 0.0),
            "symbols": len(result.symbol_scores),
        }
    except Exception as exc:
        return {
            "factor_mode": "error",
            "factor_count": 0,
            "coverage_rate": 0.0,
            "symbols": len(frames),
            "error": str(exc),
        }


def write_registry(path: Path, registry: MinedFactorRegistry) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": registry.schema_version,
                "metadata": registry.metadata,
                "factors": [record.to_dict() for record in registry.factors],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Governed Factor Health Automation",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Cadence: {payload['cadence']}",
        f"- Registry factors: {payload['registry_factor_count']}",
        f"- Monitored production factors: {payload['monitored_factor_count']}",
        f"- Evaluated factors: {payload['evaluated_factor_count']}",
        f"- Registry actions applied: {payload['apply_registry_actions']}",
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
        lines.append(f"- Evaluated factors: {fresh.get('evaluated_factor_count')}")
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
            "- Default health classification uses local registry evidence and runtime smoke.",
            (
                "- Repeated runs over the same matured evaluation window are "
                "observed but not double-counted as new failures."
            ),
            (
                "- The automation may reduce weight or deprecate weak "
                "production factors only when `--apply-registry-actions` is set."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluation_id(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return "missing"
    diagnostics = evaluation.get("diagnostics", {}) or {}
    end_date = str(diagnostics.get("evaluation_end_date", "") or "")
    horizon = str(
        (evaluation.get("metrics", {}) or {}).get("horizon_days", "") or ""
    )
    rankic_count = str(diagnostics.get("rankic_count", "") or "")
    if end_date:
        return f"end={end_date}|h={horizon}|n={rankic_count}"
    return str(evaluation.get("name", "") or "unknown")


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
