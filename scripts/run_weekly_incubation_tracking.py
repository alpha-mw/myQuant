#!/usr/bin/env python3
"""Build the local weekly incubation tracking record.

The runner is offline and deterministic for the same inputs. It writes only to
``results/incubation_tracking/`` and reads strategy/audit artifacts without
modifying them.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from run_track_record_audit import DEFAULT_OUTPUT_ROOT, run_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "docs" / "runbooks" / "incubation_protocol.md"
DEFAULT_TRACKING_ROOT = PROJECT_ROOT / "results" / "incubation_tracking"
WARNING_TODO = "协议阈值未生效，本周报仅供参考，kill 规则不可执行"


def week_end_for(value: str | None = None) -> str:
    current = datetime.strptime(value, "%Y%m%d").date() if value else date.today()
    friday = current + timedelta(days=(4 - current.weekday()) % 7)
    return friday.strftime("%Y-%m-%d")


def protocol_warning(protocol_path: Path = DEFAULT_PROTOCOL) -> str:
    if protocol_path.exists() and "TODO(maxwell)" in protocol_path.read_text(encoding="utf-8"):
        return WARNING_TODO
    return ""


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


def build_weekly_record(
    metrics: dict[str, Any],
    *,
    week_end: str,
    decision_events: list[dict[str, Any]] | None = None,
    warning: str = "",
) -> dict[str, Any]:
    benchmarks = metrics.get("benchmarks", {})
    star50 = benchmarks.get("star50_nav", {})
    beta = metrics.get("beta_adjusted_excess", {})
    exposure = metrics.get("regime_exposure_compliance", {})
    cost = metrics.get("estimated_execution_cost", {})
    shadow = metrics.get("shadow_ledgers", {})
    selection = metrics.get("selection_alpha", {})
    return {
        "schema_version": "weekly_incubation_tracking.v1",
        "week_end": week_end,
        "warning": warning,
        "star50_full_window_excess": star50.get("full_window_excess"),
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
    if record.get("warning"):
        lines.append(f"**{record['warning']}**")
        lines.append("")
    lines.extend(
        [
            f"- week_end: {record.get('week_end')}",
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
        ]
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
    record = build_weekly_record(
        metrics,
        week_end=args.week_end or week_end_for(audit_as_of),
        decision_events=_load_decision_events(args.decision_log),
        warning=warning,
    )
    jsonl_path, report_path = write_weekly_outputs(args.tracking_root, record)
    print(json.dumps({"weekly_jsonl": str(jsonl_path), "weekly_report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
