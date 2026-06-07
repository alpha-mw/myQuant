from __future__ import annotations

import json
from pathlib import Path

from quant_investor.factors.governance import (
    FactorAdmissionDecision,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from scripts import factor_health_automation


def _production_record() -> FactorRecord:
    gates = [
        GateResult(gate_id=i, gate_key=f"gate_{i}", title=f"Gate {i}", passed=True)
        for i in range(1, 9)
    ]
    return FactorRecord(
        name="pv_short_reversal_5d",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="price_volume:pv_short_reversal_5d",
        weight=0.05,
        gate_results=gates,
        metrics={
            "coverage_rate": 0.90,
            "nan_rate": 0.10,
            "icir": 0.60,
            "mean_rankic": 0.05,
            "positive_ic_ratio": 0.70,
            "oos_positive_ratio": 0.70,
            "top_bottom_spread": 0.08,
            "cost_adjusted_return": 0.06,
            "turnover": 4.0,
            "capacity_pressure": 0.10,
            "neutralized_icir": 0.40,
            "horizon_days": 30,
        },
        admission_decision=FactorAdmissionDecision.PRODUCTION_CANDIDATE,
    )


def test_factor_health_automation_writes_report_without_registry_mutation(tmp_path):
    registry_path = tmp_path / "mined_factors.json"
    before = {
        "schema_version": "mined-factor-registry.v1",
        "metadata": {"fixture": True},
        "factors": [_production_record().to_dict()],
    }
    registry_path.write_text(json.dumps(before, indent=2), encoding="utf-8")

    output_dir = tmp_path / "reports"
    exit_code = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--data-root",
            str(tmp_path / "missing_data"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert json.loads(registry_path.read_text(encoding="utf-8")) == before
    reports = sorted(Path(output_dir).glob("factor_health_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["monitored_factor_count"] == 1
    assert payload["status_counts"] == {"healthy": 1}
    assert payload["action_counts"] == {"keep": 1}
    assert payload["runtime_smoke"]["factor_mode"] == "unavailable"
