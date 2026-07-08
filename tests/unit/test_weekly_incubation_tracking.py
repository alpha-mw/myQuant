from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_weekly_incubation_tracking",
        ROOT / "scripts" / "run_weekly_incubation_tracking.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics():
    nav_rows = []
    aligned = []
    counterfactual = {}
    for index in range(70):
        date = f"2026-03-{index + 1:02d}"
        actual_nav = 1.0 + index * 0.001
        benchmark_nav = 1.0 + index * 0.0005
        nav_rows.append({"date": date, "nav": actual_nav})
        aligned.append({"date": date, "actual_nav": actual_nav, "benchmark_nav": benchmark_nav})
        counterfactual[date] = 1.0 + index * 0.0008
    return {
        "nav_rows": nav_rows,
        "counterfactual1_nav": counterfactual,
        "benchmarks": {"star50_nav": {"full_window_excess": 0.03, "aligned_series": aligned}},
        "beta_adjusted_excess": {
            "beta": 0.7,
            "alpha_annualized": 0.12,
            "standard_ir_daily": 0.05,
        },
        "regime_exposure_compliance": {"violation_ratio": 0.25},
        "estimated_execution_cost": {
            "gross_full_window_return": 0.20,
            "net_full_window_return": 0.19,
        },
        "shadow_ledgers": {
            "cap050_current_difference_vs_actual": 0.02,
            "machine_exit_current_difference_vs_actual": -0.01,
        },
        "selection_alpha": {"selection_alpha": 0.04},
        "execution_quality": {
            "slippage_bps": {
                "buy": {"p90": 12.0},
                "sell": {"p90": 20.0},
            }
        },
        "counterparty_quality": {
            "sell": {
                "ret_5d": {"negative_share": 0.6},
                "ret_10d": {"negative_share": 0.5},
                "ret_20d": {"negative_share": 0.4},
            }
        },
    }


def test_weekly_record_metrics_and_decision_pairing():
    mod = _load_module()
    record = mod.build_weekly_record(
        _metrics(),
        week_end="2026-07-10",
        decision_events=[
            {"event_type": "advisory", "trade_date": "2026-07-07"},
            {"event_type": "human_action", "trade_date": "2026-07-07"},
        ],
        warning="warn",
        fundamentals={
            "output_dir": "results/track_record_audit/20260710/fundamentals",
            "pending_disclosure_symbols": ["002008.SZ"],
            "high_scrutiny_symbols": ["002008.SZ", "002851.SZ"],
        },
    )

    assert record["warning"] == "warn"
    assert record["exposure_compliance_rate"] == 0.75
    assert record["decision_log_completeness"]["paired_ratio"] == 1.0
    assert record["selection_alpha"] == 0.04
    assert record["schema_version"] == "weekly_incubation_tracking.v2"
    assert record["current_drawdown_tier"] == "none"
    assert record["kill_review_triggered"] is False
    assert record["add_eligible"] is False
    assert record["post_exit_negative_share"]["ret_10d"] == 0.5
    assert record["fundamental_tracking"]["high_scrutiny_symbols"] == ["002008.SZ", "002851.SZ"]
    report = mod.render_weekly_report(record)
    assert "high_scrutiny_symbols: 002008.SZ, 002851.SZ" in report
    assert "pending_disclosure_symbols: 002008.SZ" in report


def test_weekly_write_is_idempotent_for_same_week(tmp_path):
    mod = _load_module()
    first = mod.build_weekly_record(_metrics(), week_end="2026-07-10", warning="")
    second = dict(first)
    second["selection_alpha"] = 0.05

    mod.write_weekly_outputs(tmp_path, first)
    jsonl_path, report_path = mod.write_weekly_outputs(tmp_path, second)

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["selection_alpha"] == 0.05
    assert "selection_alpha" in report_path.read_text(encoding="utf-8")


def test_threshold_constants_match_phase16_protocol():
    mod = _load_module()
    thresholds = mod.DEFAULT_THRESHOLDS
    assert thresholds.drawdown_tier_1_review == -0.12
    assert thresholds.drawdown_tier_2_half == -0.20
    assert thresholds.drawdown_tier_3_quarter == -0.30
    assert thresholds.drawdown_tier_4_clear == -0.40
    assert thresholds.kill_excess_window_weeks == 8
    assert thresholds.kill_excess_cumulative_threshold == -0.10
    assert thresholds.kill_phase_alpha_window_days == 60
    assert thresholds.kill_phase_alpha_threshold == -0.05
    assert thresholds.kill_slippage_p90_bps == 50
    assert thresholds.kill_slippage_sustain_weeks == 2
    assert thresholds.add_excess_window_weeks == 8
    assert thresholds.add_excess_cumulative_threshold == 0.05
    assert thresholds.add_phase_alpha_threshold == 0.05
    assert thresholds.add_size_limit_pct_nav == 0.10


def test_weekly_threshold_boundaries_and_banners():
    mod = _load_module()
    metrics = _metrics()
    metrics["execution_quality"]["slippage_bps"]["buy"]["p90"] = 55.0
    previous = [{"week_end": "2026-07-03", "weekly_slippage_p90_bps": 60.0}]
    record = mod.build_weekly_record(
        metrics,
        week_end="2026-07-10",
        previous_records=previous,
    )
    assert record["kill_conditions"]["slippage_p90_sustain"]["triggered"] is True
    assert record["kill_review_triggered"] is True
    assert "KILL-REVIEW" in mod.render_weekly_report(record)

    metrics = _metrics()
    first = metrics["benchmarks"]["star50_nav"]["aligned_series"][-41]
    last = metrics["benchmarks"]["star50_nav"]["aligned_series"][-1]
    first["actual_nav"] = 1.0
    first["benchmark_nav"] = 1.0
    last["actual_nav"] = 1.08
    last["benchmark_nav"] = 1.0
    start_date = metrics["nav_rows"][-61]["date"]
    end_date = metrics["nav_rows"][-1]["date"]
    metrics["nav_rows"][-61]["nav"] = 1.0
    metrics["nav_rows"][-1]["nav"] = 1.08
    metrics["counterfactual1_nav"][start_date] = 1.0
    metrics["counterfactual1_nav"][end_date] = 1.0
    record = mod.build_weekly_record(metrics, week_end="2026-07-10")
    assert record["add_conditions"]["rolling_8w_excess"]["triggered"] is True
    assert record["add_conditions"]["phase_alpha_60d"]["triggered"] is True
    assert record["add_eligible"] is True
    assert "ADD-ELIGIBLE" in mod.render_weekly_report(record)


def test_drawdown_tier_boundaries():
    mod = _load_module()
    thresholds = mod.DEFAULT_THRESHOLDS
    assert mod._drawdown_tier(-0.1199, thresholds) == "none"
    assert mod._drawdown_tier(-0.12, thresholds) == "tier_1_review"
    assert mod._drawdown_tier(-0.20, thresholds) == "tier_2_half"
    assert mod._drawdown_tier(-0.30, thresholds) == "tier_3_quarter"
    assert mod._drawdown_tier(-0.40, thresholds) == "tier_4_clear"


def test_protocol_warning_detects_todo(tmp_path):
    mod = _load_module()
    protocol = tmp_path / "incubation_protocol.md"
    protocol.write_text("Kill rule TODO(maxwell)\n", encoding="utf-8")
    assert mod.protocol_warning(protocol) == mod.WARNING_TODO
