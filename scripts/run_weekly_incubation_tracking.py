#!/usr/bin/env python3
"""Build the local weekly incubation tracking record.

The runner is offline and deterministic for the same inputs. It writes only to
``results/incubation_tracking/`` and reads strategy/audit artifacts without
modifying them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from run_track_record_audit import DEFAULT_OUTPUT_ROOT, run_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "docs" / "runbooks" / "incubation_protocol.md"
DEFAULT_TRACKING_ROOT = PROJECT_ROOT / "results" / "incubation_tracking"
WARNING_TODO = "协议阈值未生效，本周报仅供参考，kill 规则不可执行"


@dataclass(frozen=True)
class IncubationThresholds:
    drawdown_tier_1_review: float = -0.12
    drawdown_tier_2_half: float = -0.20
    drawdown_tier_3_quarter: float = -0.30
    drawdown_tier_4_clear: float = -0.40
    kill_excess_window_weeks: int = 8
    kill_excess_cumulative_threshold: float = -0.10
    kill_phase_alpha_window_days: int = 60
    kill_phase_alpha_threshold: float = -0.05
    kill_slippage_p90_bps: float = 50.0
    kill_slippage_sustain_weeks: int = 2
    add_excess_window_weeks: int = 8
    add_excess_cumulative_threshold: float = 0.05
    add_phase_alpha_threshold: float = 0.05
    add_size_limit_pct_nav: float = 0.10
    markov_drawdown_advantage_pct: float = 5.0
    post_exit_positive_threshold: float = 0.30


DEFAULT_THRESHOLDS = IncubationThresholds()


def week_end_for(value: str | None = None) -> str:
    current = datetime.strptime(value, "%Y%m%d").date() if value else date.today()
    friday = current + timedelta(days=(4 - current.weekday()) % 7)
    return friday.strftime("%Y-%m-%d")


def protocol_warning(protocol_path: Path = DEFAULT_PROTOCOL) -> str:
    if protocol_path.exists() and "TODO(maxwell)" in protocol_path.read_text(encoding="utf-8"):
        return WARNING_TODO
    return ""


def load_existing_weekly_records(root: Path) -> list[dict[str, Any]]:
    jsonl_path = root / "weekly.jsonl"
    if not jsonl_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_decision_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def decision_log_completeness(events: list[dict[str, Any]]) -> dict[str, Any]:
    advisory = [row for row in events if row.get("event_type") == "advisory"]
    human = [row for row in events if row.get("event_type") == "human_action"]
    human_dates = {str(row.get("trade_date") or "") for row in human}
    paired = [
        row for row in advisory
        if str(row.get("trade_date") or "") in human_dates and str(row.get("trade_date") or "")
    ]
    denominator = max(len(advisory), len(human), 1)
    return {
        "advisory_count": len(advisory),
        "human_action_count": len(human),
        "paired_count": len(paired),
        "paired_ratio": len(paired) / denominator,
    }


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _return_between(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0:
        return None
    return end / start - 1.0


def _rolling_excess(star50: dict[str, Any], thresholds: IncubationThresholds) -> dict[str, Any]:
    series = [
        row for row in star50.get("aligned_series", [])
        if _safe_float(row.get("actual_nav")) is not None
        and _safe_float(row.get("benchmark_nav")) is not None
    ]
    window_days = thresholds.kill_excess_window_weeks * 5
    if len(series) <= window_days:
        return {
            "available": False,
            "value": None,
            "window_trading_days": window_days,
            "observations": len(series),
        }
    start = series[-window_days - 1]
    end = series[-1]
    actual = _return_between(_safe_float(start.get("actual_nav")), _safe_float(end.get("actual_nav")))
    bench = _return_between(_safe_float(start.get("benchmark_nav")), _safe_float(end.get("benchmark_nav")))
    value = actual - bench if actual is not None and bench is not None else None
    return {
        "available": value is not None,
        "value": value,
        "window_trading_days": window_days,
        "start_date": start.get("date"),
        "end_date": end.get("date"),
        "actual_return": actual,
        "benchmark_return": bench,
        "observations": len(series),
    }


def _phase_timing_alpha(metrics: dict[str, Any], thresholds: IncubationThresholds) -> dict[str, Any]:
    nav_rows = metrics.get("nav_rows", [])
    counterfactual = metrics.get("counterfactual1_nav", {})
    window_days = thresholds.kill_phase_alpha_window_days
    if len(nav_rows) <= window_days:
        return {
            "available": False,
            "value": None,
            "window_trading_days": window_days,
            "observations": len(nav_rows),
        }
    start = nav_rows[-window_days - 1]
    end = nav_rows[-1]
    start_date = str(start.get("date") or "")
    end_date = str(end.get("date") or "")
    actual = _return_between(_safe_float(start.get("nav")), _safe_float(end.get("nav")))
    cf = _return_between(
        _safe_float(counterfactual.get(start_date)),
        _safe_float(counterfactual.get(end_date)),
    )
    value = actual - cf if actual is not None and cf is not None else None
    return {
        "available": value is not None,
        "value": value,
        "window_trading_days": window_days,
        "start_date": start_date,
        "end_date": end_date,
        "actual_return": actual,
        "counterfactual1_return": cf,
        "observations": len(nav_rows),
    }


def _current_drawdown(nav_rows: list[dict[str, Any]]) -> dict[str, Any]:
    peak_nav = None
    peak_date = ""
    latest_nav = None
    latest_date = ""
    for row in nav_rows:
        nav = _safe_float(row.get("nav"))
        if nav is None:
            continue
        if peak_nav is None or nav > peak_nav:
            peak_nav = nav
            peak_date = str(row.get("date") or "")
        latest_nav = nav
        latest_date = str(row.get("date") or "")
    value = _return_between(peak_nav, latest_nav)
    return {
        "value": value,
        "peak_nav": peak_nav,
        "peak_date": peak_date,
        "latest_nav": latest_nav,
        "latest_date": latest_date,
    }


def _drawdown_tier(drawdown: float | None, thresholds: IncubationThresholds) -> str:
    if drawdown is None:
        return "unavailable"
    if drawdown <= thresholds.drawdown_tier_4_clear:
        return "tier_4_clear"
    if drawdown <= thresholds.drawdown_tier_3_quarter:
        return "tier_3_quarter"
    if drawdown <= thresholds.drawdown_tier_2_half:
        return "tier_2_half"
    if drawdown <= thresholds.drawdown_tier_1_review:
        return "tier_1_review"
    return "none"


def _slippage_p90(execution_quality: dict[str, Any]) -> float | None:
    values = [
        _safe_float(row.get("p90"))
        for row in execution_quality.get("slippage_bps", {}).values()
        if isinstance(row, dict)
    ]
    available = [value for value in values if value is not None]
    return max(available) if available else None


def _slippage_sustain_weeks(
    current_p90: float | None,
    previous_records: list[dict[str, Any]],
    thresholds: IncubationThresholds,
) -> int:
    if current_p90 is None or current_p90 <= thresholds.kill_slippage_p90_bps:
        return 0
    count = 1
    for row in reversed(previous_records):
        value = _safe_float(row.get("weekly_slippage_p90_bps"))
        if value is None or value <= thresholds.kill_slippage_p90_bps:
            break
        count += 1
    return count


def _post_exit_negative_share(metrics: dict[str, Any]) -> dict[str, Any]:
    sell = metrics.get("counterparty_quality", {}).get("sell", {})
    result: dict[str, Any] = {}
    for horizon in (5, 10, 20):
        result[f"ret_{horizon}d"] = sell.get(f"ret_{horizon}d", {}).get("negative_share")
    values = [_safe_float(value) for value in result.values()]
    valid = [value for value in values if value is not None]
    result["min_negative_share"] = min(valid) if valid else None
    return result


def _condition(name: str, value: float | int | None, threshold: float | int, triggered: bool) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "triggered": bool(triggered),
        "available": value is not None,
    }


def _evaluate_thresholds(
    metrics: dict[str, Any],
    previous_records: list[dict[str, Any]],
    thresholds: IncubationThresholds,
) -> dict[str, Any]:
    star50 = metrics.get("benchmarks", {}).get("star50_nav", {})
    rolling_excess = _rolling_excess(star50, thresholds)
    phase_alpha = _phase_timing_alpha(metrics, thresholds)
    drawdown = _current_drawdown(metrics.get("nav_rows", []))
    slippage = _slippage_p90(metrics.get("execution_quality", {}))
    sustain = _slippage_sustain_weeks(slippage, previous_records, thresholds)
    rolling_value = _safe_float(rolling_excess.get("value"))
    phase_value = _safe_float(phase_alpha.get("value"))
    kill_conditions = {
        "rolling_8w_excess": _condition(
            "rolling_8w_excess",
            rolling_value,
            thresholds.kill_excess_cumulative_threshold,
            rolling_value is not None and rolling_value < thresholds.kill_excess_cumulative_threshold,
        ),
        "phase_alpha_60d": _condition(
            "phase_alpha_60d",
            phase_value,
            thresholds.kill_phase_alpha_threshold,
            phase_value is not None and phase_value < thresholds.kill_phase_alpha_threshold,
        ),
        "slippage_p90_sustain": _condition(
            "slippage_p90_sustain",
            sustain,
            thresholds.kill_slippage_sustain_weeks,
            sustain >= thresholds.kill_slippage_sustain_weeks,
        ),
    }
    add_conditions = {
        "rolling_8w_excess": _condition(
            "rolling_8w_excess",
            rolling_value,
            thresholds.add_excess_cumulative_threshold,
            rolling_value is not None and rolling_value > thresholds.add_excess_cumulative_threshold,
        ),
        "phase_alpha_60d": _condition(
            "phase_alpha_60d",
            phase_value,
            thresholds.add_phase_alpha_threshold,
            phase_value is not None and phase_value > thresholds.add_phase_alpha_threshold,
        ),
    }
    return {
        "thresholds": asdict(thresholds),
        "rolling_8w_excess": rolling_excess,
        "phase_alpha_60d": phase_alpha,
        "current_drawdown": drawdown,
        "current_drawdown_tier": _drawdown_tier(_safe_float(drawdown.get("value")), thresholds),
        "weekly_slippage_p90_bps": slippage,
        "slippage_p90_sustain_weeks": sustain,
        "post_exit_negative_share": _post_exit_negative_share(metrics),
        "kill_conditions": kill_conditions,
        "add_conditions": add_conditions,
        "kill_review_triggered": any(row["triggered"] for row in kill_conditions.values()),
        "add_eligible": all(row["triggered"] for row in add_conditions.values()),
    }


def build_weekly_record(
    metrics: dict[str, Any],
    *,
    week_end: str,
    decision_events: list[dict[str, Any]] | None = None,
    warning: str = "",
    previous_records: list[dict[str, Any]] | None = None,
    thresholds: IncubationThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    benchmarks = metrics.get("benchmarks", {})
    star50 = benchmarks.get("star50_nav", {})
    beta = metrics.get("beta_adjusted_excess", {})
    exposure = metrics.get("regime_exposure_compliance", {})
    cost = metrics.get("estimated_execution_cost", {})
    shadow = metrics.get("shadow_ledgers", {})
    selection = metrics.get("selection_alpha", {})
    threshold_status = _evaluate_thresholds(metrics, previous_records or [], thresholds)
    nav_rows = metrics.get("nav_rows", [])
    basis_date = nav_rows[-1].get("date") if nav_rows else None
    return {
        "schema_version": "weekly_incubation_tracking.v2",
        "week_end": week_end,
        "date_basis": basis_date,
        "date_basis_status": "week_end_record" if basis_date == week_end else "latest_available_record",
        "warning": warning,
        "star50_full_window_excess": star50.get("full_window_excess"),
        "star50_rolling_8w_cumulative_excess": threshold_status["rolling_8w_excess"].get("value"),
        "phase_timing_alpha_60d": threshold_status["phase_alpha_60d"].get("value"),
        "beta_vs_star50": beta.get("beta"),
        "annualized_alpha_vs_star50": beta.get("alpha_annualized"),
        "ir_daily": beta.get("standard_ir_daily"),
        "exposure_compliance_rate": (
            1.0 - exposure.get("violation_ratio")
            if exposure.get("violation_ratio") is not None
            else None
        ),
        "gross_full_window_return": cost.get("gross_full_window_return"),
        "net_full_window_return": cost.get("net_full_window_return"),
        "shadow_cap050_difference_vs_actual": shadow.get("cap050_current_difference_vs_actual"),
        "shadow_machine_exit_difference_vs_actual": shadow.get("machine_exit_current_difference_vs_actual"),
        "selection_alpha": selection.get("selection_alpha"),
        "decision_log_completeness": decision_log_completeness(decision_events or []),
        **threshold_status,
    }


def write_weekly_outputs(root: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "weekly.jsonl"
    existing: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    week_end = record["week_end"]
    rows = [row for row in existing if row.get("week_end") != week_end]
    rows.append(record)
    rows.sort(key=lambda row: str(row.get("week_end") or ""))
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report_path = root / "weekly_report.md"
    report_path.write_text(render_weekly_report(rows[-1]), encoding="utf-8")
    return jsonl_path, report_path


def render_weekly_report(record: dict[str, Any]) -> str:
    lines = ["# Weekly Incubation Tracking", ""]
    if record.get("kill_review_triggered"):
        lines.append('<div style="color:#b00020;font-weight:700">KILL-REVIEW</div>')
        lines.append("")
    if record.get("add_eligible"):
        lines.append('<div style="color:#0b6b2b;font-weight:700">ADD-ELIGIBLE</div>')
        lines.append("")
    if record.get("warning"):
        lines.append(f"**{record['warning']}**")
        lines.append("")
    lines.extend(
        [
            f"- week_end: {record.get('week_end')}",
            f"- date_basis: {record.get('date_basis')} ({record.get('date_basis_status')})",
            f"- current_drawdown: {record.get('current_drawdown', {}).get('value')}",
            f"- current_drawdown_tier: {record.get('current_drawdown_tier')}",
            f"- star50_rolling_8w_cumulative_excess: {record.get('star50_rolling_8w_cumulative_excess')}",
            f"- phase_timing_alpha_60d: {record.get('phase_timing_alpha_60d')}",
            f"- weekly_slippage_p90_bps: {record.get('weekly_slippage_p90_bps')}",
            f"- slippage_p90_sustain_weeks: {record.get('slippage_p90_sustain_weeks')}",
            f"- beta_vs_star50: {record.get('beta_vs_star50')}",
            f"- annualized_alpha_vs_star50: {record.get('annualized_alpha_vs_star50')}",
            f"- ir_daily: {record.get('ir_daily')}",
            f"- exposure_compliance_rate: {record.get('exposure_compliance_rate')}",
            f"- selection_alpha: {record.get('selection_alpha')}",
            f"- gross_full_window_return: {record.get('gross_full_window_return')}",
            f"- net_full_window_return: {record.get('net_full_window_return')}",
            f"- shadow_cap050_difference_vs_actual: {record.get('shadow_cap050_difference_vs_actual')}",
            f"- shadow_machine_exit_difference_vs_actual: {record.get('shadow_machine_exit_difference_vs_actual')}",
            f"- decision_log_paired_ratio: {record.get('decision_log_completeness', {}).get('paired_ratio')}",
            f"- post_exit_negative_share: {json.dumps(record.get('post_exit_negative_share'), ensure_ascii=False, sort_keys=True)}",
        ]
    )
    lines.extend(["", "## Kill Conditions", ""])
    for name, condition in record.get("kill_conditions", {}).items():
        lines.append(
            f"- {name}: value={condition.get('value')} threshold={condition.get('threshold')} "
            f"triggered={condition.get('triggered')}"
        )
    lines.extend(["", "## Add Conditions", ""])
    for name, condition in record.get("add_conditions", {}).items():
        lines.append(
            f"- {name}: value={condition.get('value')} threshold={condition.get('threshold')} "
            f"triggered={condition.get('triggered')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--decision-log", type=Path, default=PROJECT_ROOT / "results" / "decision_log" / "decision_log.jsonl")
    parser.add_argument("--week-end")
    parser.add_argument("--audit-as-of")
    args = parser.parse_args()
    warning = protocol_warning(args.protocol)
    if warning:
        print(warning)
    audit_as_of = args.audit_as_of or date.today().strftime("%Y%m%d")
    metrics = run_audit(output_root=DEFAULT_OUTPUT_ROOT, as_of_date=audit_as_of, generate_plots=False)
    previous_records = [
        row for row in load_existing_weekly_records(args.tracking_root)
        if row.get("week_end") != (args.week_end or week_end_for(audit_as_of))
    ]
    record = build_weekly_record(
        metrics,
        week_end=args.week_end or week_end_for(audit_as_of),
        decision_events=_load_decision_events(args.decision_log),
        warning=warning,
        previous_records=previous_records,
    )
    jsonl_path, report_path = write_weekly_outputs(args.tracking_root, record)
    print(json.dumps({"weekly_jsonl": str(jsonl_path), "weekly_report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
