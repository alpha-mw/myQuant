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
    return {
        "benchmarks": {"star50_nav": {"full_window_excess": 0.03}},
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
    )

    assert record["warning"] == "warn"
    assert record["exposure_compliance_rate"] == 0.75
    assert record["decision_log_completeness"]["paired_ratio"] == 1.0
    assert record["selection_alpha"] == 0.04


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


def test_protocol_warning_detects_todo(tmp_path):
    mod = _load_module()
    protocol = tmp_path / "incubation_protocol.md"
    protocol.write_text("Kill rule TODO(maxwell)\n", encoding="utf-8")
    assert mod.protocol_warning(protocol) == mod.WARNING_TODO
