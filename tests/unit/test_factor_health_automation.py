from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_investor.factors.governance import (
    GATE_SPECS,
    FactorAdmissionDecision,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.health import (
    FactorHealthAction,
    FactorHealthStatus,
    apply_health_decision,
    classify_factor_health,
)
from quant_investor.factors.runtime import MinedFactorRegistry
from scripts import factor_health_automation
from tests.fixtures.strict_cn_snapshot import coverage_v4, v4_snapshot_paths


def _production_record() -> FactorRecord:
    gates = [
        GateResult(
            gate_id=spec.gate_id,
            gate_key=spec.key,
            title=spec.title,
            passed=True,
        )
        for spec in GATE_SPECS
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


def _blend_record() -> FactorRecord:
    record = _production_record()
    return FactorRecord(
        name="pv_blend_volstab19x2_mom90_amihud5_w75",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="price_volume:pv_blend_volstab19x2_mom90_amihud5_w75",
        weight=0.05,
        gate_results=record.gate_results,
        metrics=record.metrics,
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
    assert payload["evaluation_source_counts"] == {"registry_evidence": 1}
    assert payload["evidence_age_days"]["unknown_count"] == 1
    assert payload["registry_update_status"] == "not_requested"
    assert payload["runtime_smoke"]["backend"] == "parquet"
    assert payload["runtime_smoke"]["factor_mode"] == "parquet_canonical_unavailable"
    assert "strict Parquet snapshot pointer unreadable" in payload["runtime_smoke"]["error"]


def test_runtime_smoke_reads_parquet_serving_without_csv_dependency(
    monkeypatch,
    tmp_path,
):
    _write_parquet_fixture(tmp_path)
    captured = {}

    def fake_build_quant_branch_result(*, frames):
        captured["symbols"] = sorted(frames)
        return SimpleNamespace(
            metadata={
                "factor_mode": "governed_mined_factors",
                "mined_factor_runtime": {
                    "factor_count": 14,
                    "coverage_rate": 1.0,
                },
            },
            symbol_scores={symbol: 0.1 for symbol in frames},
        )

    from quant_investor.market.dag import packets

    monkeypatch.setattr(
        packets,
        "_build_quant_branch_result",
        fake_build_quant_branch_result,
    )

    legacy_csv = tmp_path / "clean" / "cn_daily" / "hs300"
    legacy_csv.mkdir(parents=True)
    (legacy_csv / "999999.SZ.csv").write_text(
        "trade_date,close\n20260103,99\n",
        encoding="utf-8",
    )

    smoke = factor_health_automation.build_runtime_smoke(
        tmp_path,
        ["full_a"],
        2,
        market="CN",
        mode_policy="strict",
    )

    assert captured["symbols"] == ["000001.SZ", "000002.SZ"]
    assert smoke["data_source"] == "parquet_canonical"
    assert smoke["factor_mode"] == "governed_mined_factors"
    assert smoke["factor_count"] == 14
    assert smoke["coverage_rate"] == 1.0
    assert smoke["symbols_loaded"] == 2
    assert smoke["snapshot_id"] == "snap-001"
    assert smoke["latest_complete_trade_date"] == "20260103"


def test_fresh_evaluation_uses_parquet_context_not_legacy_csv(tmp_path):
    _write_parquet_fresh_fixture(tmp_path)
    legacy_csv = tmp_path / "hs300"
    legacy_csv.mkdir(parents=True)
    (legacy_csv / "999999.SZ.csv").write_text(
        "symbol,trade_date,close,vol\n999999.SZ,20260101,1,1\n",
        encoding="utf-8",
    )

    registry_path = tmp_path / "mined_factors.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "mined-factor-registry.v1",
                "metadata": {"fixture": True},
                "factors": [_production_record().to_dict()],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        data_root=str(tmp_path),
        universes=["hs300"],
        horizon_days=1,
        warmup_days=1,
        registry_path=str(registry_path),
        analysis_start_date="full",
        min_analysis_price_coverage=0.0,
        decision_cost_bps=1.0,
        incremental_sleeve_weight=0.03,
        market="CN",
        mode_policy="strict",
    )

    result = factor_health_automation._fresh_evaluations(
        args,
        [_production_record()],
    )

    assert result["blockers"] == []
    assert result["context"]["data_source"] == "parquet_canonical"
    assert result["context"]["symbols_loaded"] == 1
    assert result["context"]["sample_symbols"] == ["000001.SZ"]
    evaluation = result["evaluations"]["pv_short_reversal_5d"]
    assert evaluation["diagnostics"]["data_source"] == "parquet_canonical"
    assert evaluation["diagnostics"]["snapshot_id"] == "snap-fresh"
    assert evaluation["diagnostics"]["existing_composite_mode"] == "leave_one_out"
    assert evaluation["diagnostics"]["implementation_hash"]
    assert evaluation["diagnostics"]["maturity_window_id"]
    assert evaluation["diagnostics"]["evaluation_hash"].startswith("sha256:")


def test_fresh_evaluation_existing_composite_supports_blend_factors(tmp_path):
    _write_parquet_fresh_fixture(tmp_path)
    registry_path = tmp_path / "mined_factors.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "mined-factor-registry.v1",
                "metadata": {"fixture": True},
                "factors": [_production_record().to_dict(), _blend_record().to_dict()],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        data_root=str(tmp_path),
        universes=["full_a"],
        horizon_days=1,
        warmup_days=1,
        registry_path=str(registry_path),
        analysis_start_date="full",
        min_analysis_price_coverage=0.0,
        decision_cost_bps=1.0,
        incremental_sleeve_weight=0.03,
        market="CN",
        mode_policy="strict",
    )

    result = factor_health_automation._fresh_evaluations(
        args,
        [_production_record(), _blend_record()],
    )

    assert result["blockers"] == []
    assert result["context"]["existing_composite_blocker"] == ""
    assert set(result["evaluations"]) == {
        "pv_short_reversal_5d",
        "pv_blend_volstab19x2_mom90_amihud5_w75",
    }


def test_parquet_fresh_context_blocks_partial_symbol_readback(
    monkeypatch,
    tmp_path,
):
    from quant_investor.market import market_data_reader

    class FakeReader:
        def __init__(self, **_kwargs):
            pass

        def snapshot(self):
            return {"healthy": True, "snapshot_id": "snap-partial"}

        def list_symbols(self, *, universe_key):
            assert universe_key == "full_a"
            return ["000001.SZ", "000002.SZ"]

        def read_symbol_frame(self, symbol, *, universe_key):
            assert universe_key == "full_a"
            if symbol == "000002.SZ":
                raise ValueError("fixture readback failure")
            return SimpleNamespace(
                frame=pd.DataFrame(
                    {
                        "trade_date": ["2026-06-30"],
                        "close": [10.0],
                    }
                )
            )

    monkeypatch.setattr(market_data_reader, "MarketDataReader", FakeReader)
    result = factor_health_automation._build_parquet_fresh_context(
        SimpleNamespace(
            market="CN",
            mode_policy="strict",
            data_root=str(tmp_path),
            universes=["full_a"],
            min_analysis_price_coverage=0.95,
        )
    )

    assert result["context"] is None
    assert result["metadata"]["symbols_requested"] == 2
    assert result["metadata"]["symbols_loaded"] == 1
    assert result["metadata"]["symbol_load_ratio"] == 0.5
    assert result["metadata"]["symbol_read_error_count"] == 1
    assert any(
        blocker.startswith("parquet_fresh_context_symbol_load_ratio:")
        for blocker in result["blockers"]
    )


def test_strict_fresh_requires_fresh_and_apply_bypasses_old_semantic_checks():
    with pytest.raises(SystemExit, match="2"):
        factor_health_automation.parse_args(["--strict-fresh-evaluation"])

    apply_args = factor_health_automation.parse_args(["--apply-registry-actions"])
    assert apply_args.apply_registry_actions is True

    args = factor_health_automation.parse_args(
        [
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
            "--apply-registry-actions",
        ]
    )
    assert args.apply_registry_actions is True

    with pytest.raises(SystemExit, match="2"):
        factor_health_automation.parse_args(
            [
                "--fresh-evaluation",
                "--strict-fresh-evaluation",
                "--mode-policy",
                "permissive",
            ]
        )

    for option, value in [
        ("--min-analysis-price-coverage", "-1"),
        ("--min-analysis-price-coverage", "nan"),
        ("--horizon-days", "0"),
        ("--warmup-days", "0"),
        ("--runtime-smoke-symbols", "0"),
        ("--decision-cost-bps", "-1"),
        ("--incremental-sleeve-weight", "0"),
    ]:
        with pytest.raises(SystemExit, match="2"):
            factor_health_automation.parse_args([option, value])


def test_strict_fresh_incomplete_batch_blocks_without_registry_fallback(
    monkeypatch,
    tmp_path,
):
    record = _production_record()
    record.metadata = {"health_monitor": {"consecutive_failures": 2}}
    registry_path = tmp_path / "mined_factors.json"
    before = {
        "schema_version": "mined-factor-registry.v1",
        "metadata": {"fixture": True},
        "factors": [record.to_dict()],
    }
    registry_path.write_text(json.dumps(before, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        factor_health_automation,
        "_fresh_evaluations",
        lambda _args, _factors: {
            "evaluations": {},
            "blockers": ["fixture_missing_fresh_evidence"],
            "context": {"snapshot_id": "snap-fixture"},
        },
    )
    monkeypatch.setattr(
        factor_health_automation,
        "build_runtime_smoke",
        lambda *_args, **_kwargs: {
            "backend": "parquet",
            "mode_policy": "strict",
            "fallback_used": False,
            "factor_mode": "governed_mined_factors",
            "factor_count": 1,
            "coverage_rate": 1.0,
            "snapshot_healthy": True,
            "symbols_loaded": 1,
        },
    )
    output_dir = tmp_path / "reports"

    exit_code = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
        ]
    )

    assert exit_code == 2
    assert json.loads(registry_path.read_text(encoding="utf-8")) == before
    report = next(output_dir.glob("factor_health_*.json"))
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["run_status"] == "blocked"
    assert payload["registry_actions_applied"] is False
    assert payload["registry_update_status"] == "not_requested"
    assert payload["registry_blockers"] == []
    assert payload["fresh_evaluation"]["atomic_success"] is False
    assert payload["fresh_evaluation"]["missing_factors"] == [record.name]
    assert payload["fresh_evaluation"]["data_blocked_factors"] == [record.name]
    assert payload["evaluation_source_counts"] == {"fresh_evaluation_missing": 1}
    decision = payload["decisions"][0]
    assert decision["status"] == "data_blocked"
    assert decision["action"] == "observe"
    assert decision["consecutive_failures"] == 2


def test_data_blocked_does_not_increment_or_deprecate_alpha_failure():
    record = _production_record()

    missing = classify_factor_health(
        record,
        None,
        previous_failure_count=2,
        count_failure=True,
    )
    assert missing.status == FactorHealthStatus.DATA_BLOCKED
    assert missing.action == FactorHealthAction.OBSERVE
    assert missing.consecutive_failures == 2
    assert missing.new_weight == record.weight

    evaluation = factor_health_automation._evaluation_from_record(record)
    assert evaluation is not None
    evaluation["review"]["gate_results"][0]["passed"] = False
    first_gate_block = classify_factor_health(record, evaluation)
    assert first_gate_block.status == FactorHealthStatus.DATA_BLOCKED
    assert first_gate_block.action == FactorHealthAction.OBSERVE
    assert first_gate_block.consecutive_failures == 0
    assert first_gate_block.new_weight == record.weight

    record.metadata = {
        "health_monitor": {
            "active_failure_maturity_window_ids": ["w1", "w2"],
            "last_maturity_window_id": "last-alpha-window",
            "last_evaluation_hash": "sha256:last-alpha-evaluation",
        }
    }
    apply_health_decision(
        record,
        first_gate_block,
        reviewed_at="2026-07-11T00:00:00Z",
    )
    monitor = record.metadata["health_monitor"]
    assert monitor["last_maturity_window_id"] == "last-alpha-window"
    assert monitor["last_evaluation_hash"] == "sha256:last-alpha-evaluation"
    assert monitor["last_data_blocked_evaluation_hash"]
    assert monitor["active_failure_maturity_window_ids"] == ["w1", "w2"]


def test_low_weight_decay_never_increases_exposure():
    record = _production_record()
    record.weight = 0.005
    evaluation = factor_health_automation._evaluation_from_record(record)
    assert evaluation is not None
    evaluation["metrics"]["icir"] = 0.10

    decision = classify_factor_health(
        record,
        evaluation,
        previous_failure_count=1,
        count_failure=True,
    )

    assert decision.action == FactorHealthAction.REDUCE_WEIGHT
    assert decision.new_weight == pytest.approx(0.0025)
    assert abs(decision.new_weight) <= abs(record.weight)


def test_maturity_window_deduplicates_recomputation_while_hash_tracks_config():
    metrics = {"horizon_days": 30}
    base = {
        "evaluation_end_date": "2026-06-30",
        "matured_cohort_dates": ["2026-05-29", "2026-06-30"],
        "rankic_count": 2,
        "snapshot_id": "snap-a",
        "universes": ["full_a"],
        "analysis_start_date": "2024-01-01",
        "warmup_days": 260,
        "decision_cost_bps": 1.0,
        "incremental_sleeve_weight": 0.03,
        "implementation_hash": "impl-a",
    }
    maturity_a = factor_health_automation._build_maturity_window_id(
        metrics,
        base,
    )
    hash_a = factor_health_automation._build_evaluation_hash(
        "factor-a", metrics, {**base, "maturity_window_id": maturity_a}
    )
    recomputed = {
        **base,
        "snapshot_id": "snap-b",
        "decision_cost_bps": 20.0,
        "implementation_hash": "impl-b",
    }
    maturity_b = factor_health_automation._build_maturity_window_id(
        metrics,
        recomputed,
    )
    hash_b = factor_health_automation._build_evaluation_hash(
        "factor-a", metrics, {**recomputed, "maturity_window_id": maturity_b}
    )

    assert maturity_a == maturity_b
    assert hash_a != hash_b
    assert hash_a.startswith("sha256:")


def test_same_maturity_window_does_not_increment_failure_after_recompute(
    monkeypatch,
    tmp_path,
):
    record = _production_record()
    fresh = factor_health_automation._evaluation_from_record(record)
    assert fresh is not None
    fresh["metrics"]["icir"] = 0.10
    fresh["diagnostics"]["evaluation_end_date"] = "2026-06-30"
    fresh["diagnostics"]["maturity_window_id"] = "maturity-window-a"
    fresh["diagnostics"]["evaluation_hash"] = "sha256:new-evaluation"
    fresh["diagnostics"]["evaluation_id"] = "sha256:new-evaluation"
    record.metadata = {
        "health_monitor": {
            "consecutive_failures": 1,
            "last_maturity_window_id": "maturity-window-a",
            "last_evaluation_hash": "sha256:old-evaluation",
        }
    }
    registry_path = tmp_path / "mined_factors.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "mined-factor-registry.v1",
                "metadata": {"fixture": True},
                "factors": [record.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        factor_health_automation,
        "_fresh_evaluations",
        lambda _args, _factors: {
            "evaluations": {record.name: fresh},
            "blockers": [],
            "context": {"snapshot_id": "snap-recomputed"},
        },
    )
    monkeypatch.setattr(
        factor_health_automation,
        "build_runtime_smoke",
        lambda *_args, **_kwargs: {
            "backend": "parquet",
            "mode_policy": "strict",
            "fallback_used": False,
            "factor_mode": "governed_mined_factors",
            "factor_count": 1,
            "coverage_rate": 1.0,
            "snapshot_healthy": True,
            "symbols_loaded": 1,
        },
    )
    output_dir = tmp_path / "reports"

    exit_code = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
        ]
    )

    assert exit_code == 0
    report = next(output_dir.glob("factor_health_*.json"))
    decision = json.loads(report.read_text(encoding="utf-8"))["decisions"][0]
    assert decision["action"] == "observe"
    assert decision["consecutive_failures"] == 1
    assert decision["maturity_window_id"] == "maturity-window-a"
    assert decision["evaluation_hash"] == "sha256:new-evaluation"


def test_alternating_failure_windows_are_counted_only_once(
    monkeypatch,
    tmp_path,
):
    record = _production_record()
    registry_path = tmp_path / "mined_factors.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "mined-factor-registry.v1",
                "metadata": {"fixture": True},
                "factors": [record.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    current_window = {"value": "w1"}

    def fresh_evaluations(_args, factors):
        evaluation = factor_health_automation._evaluation_from_record(factors[0])
        assert evaluation is not None
        evaluation["metrics"]["icir"] = 0.10
        window = current_window["value"]
        evaluation["diagnostics"]["evaluation_end_date"] = "2026-06-30"
        evaluation["diagnostics"]["maturity_window_id"] = window
        evaluation["diagnostics"]["evaluation_hash"] = f"sha256:{window}"
        evaluation["diagnostics"]["evaluation_id"] = f"sha256:{window}"
        return {
            "evaluations": {record.name: evaluation},
            "blockers": [],
            "context": {"snapshot_id": f"snap-{window}"},
        }

    monkeypatch.setattr(
        factor_health_automation,
        "_fresh_evaluations",
        fresh_evaluations,
    )
    monkeypatch.setattr(
        factor_health_automation,
        "build_runtime_smoke",
        lambda *_args, **_kwargs: {
            "backend": "parquet",
            "mode_policy": "strict",
            "fallback_used": False,
            "factor_mode": "governed_mined_factors",
            "factor_count": 1,
            "coverage_rate": 1.0,
            "snapshot_healthy": True,
            "symbols_loaded": 1,
        },
    )

    observed_counts = []
    observed_actions = []
    for index, window in enumerate(["w1", "w2", "w1"], start=1):
        current_window["value"] = window
        output_dir = tmp_path / f"reports_{index}"
        exit_code = factor_health_automation.main(
            [
                "--registry-path",
                str(registry_path),
                "--output-dir",
                str(output_dir),
                "--fresh-evaluation",
                "--strict-fresh-evaluation",
            ]
        )
        assert exit_code == 0
        payload = json.loads(
            next(output_dir.glob("factor_health_*.json")).read_text(encoding="utf-8")
        )
        observed_counts.append(payload["decisions"][0]["consecutive_failures"])
        observed_actions.append(payload["decisions"][0]["action"])

    written = json.loads(registry_path.read_text(encoding="utf-8"))
    assert observed_counts == [1, 1, 1]
    assert observed_actions == ["watchlist", "watchlist", "watchlist"]
    assert "health_monitor" not in written["factors"][0]["metadata"]


def test_old_healthy_evidence_cannot_reset_newer_failure_streak(
    monkeypatch,
    tmp_path,
):
    record = _production_record()
    record.metadata = {
        "health_monitor": {
            "consecutive_failures": 1,
            "active_failure_maturity_window_ids": ["new-window"],
            "last_maturity_window_id": "new-window",
            "last_alpha_evidence_end_date": "2026-06-30",
        }
    }
    registry_path = tmp_path / "mined_factors.json"
    before_text = json.dumps(
        {
            "schema_version": "mined-factor-registry.v1",
            "metadata": {"fixture": True},
            "factors": [record.to_dict()],
        }
    )
    registry_path.write_text(before_text, encoding="utf-8")
    fresh = factor_health_automation._evaluation_from_record(record)
    assert fresh is not None
    fresh["diagnostics"].update(
        {
            "evaluation_end_date": "2026-05-31",
            "maturity_window_id": "old-healthy-window",
            "evaluation_hash": "sha256:old-healthy-window",
            "evaluation_id": "sha256:old-healthy-window",
        }
    )
    monkeypatch.setattr(
        factor_health_automation,
        "_fresh_evaluations",
        lambda _args, _factors: {
            "evaluations": {record.name: fresh},
            "blockers": [],
            "context": {"snapshot_id": "snap-old-healthy"},
        },
    )
    monkeypatch.setattr(
        factor_health_automation,
        "build_runtime_smoke",
        lambda *_args, **_kwargs: {
            "backend": "parquet",
            "mode_policy": "strict",
            "fallback_used": False,
            "factor_mode": "governed_mined_factors",
            "factor_count": 1,
            "coverage_rate": 1.0,
            "snapshot_healthy": True,
            "symbols_loaded": 1,
        },
    )
    output_dir = tmp_path / "reports"

    exit_code = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
        ]
    )

    assert exit_code == 2
    assert registry_path.read_text(encoding="utf-8") == before_text
    payload = json.loads(next(output_dir.glob("factor_health_*.json")).read_text(encoding="utf-8"))
    decision = payload["decisions"][0]
    assert decision["status"] == "healthy"
    assert decision["action"] == "observe"
    assert decision["consecutive_failures"] == 1
    assert decision["evidence_chronology_status"] == "blocked"
    assert any(blocker.endswith("regressed") for blocker in payload["fresh_evaluation"]["blockers"])


def test_strict_fresh_incomplete_gate_set_blocks_registry_write(
    monkeypatch,
    tmp_path,
):
    record = _production_record()
    registry_path = tmp_path / "mined_factors.json"
    before_text = json.dumps(
        {
            "schema_version": "mined-factor-registry.v1",
            "metadata": {"fixture": True},
            "factors": [record.to_dict()],
        }
    )
    registry_path.write_text(before_text, encoding="utf-8")
    fresh = factor_health_automation._evaluation_from_record(record)
    assert fresh is not None
    fresh["diagnostics"]["evaluation_end_date"] = "2026-06-30"
    fresh["review"]["gate_results"] = fresh["review"]["gate_results"][:7]
    monkeypatch.setattr(
        factor_health_automation,
        "_fresh_evaluations",
        lambda _args, _factors: {
            "evaluations": {record.name: fresh},
            "blockers": [],
            "context": {"snapshot_id": "snap-incomplete-gates"},
        },
    )
    monkeypatch.setattr(
        factor_health_automation,
        "build_runtime_smoke",
        lambda *_args, **_kwargs: {
            "backend": "parquet",
            "mode_policy": "strict",
            "fallback_used": False,
            "factor_mode": "governed_mined_factors",
            "factor_count": 1,
            "coverage_rate": 1.0,
            "snapshot_healthy": True,
            "symbols_loaded": 1,
        },
    )
    output_dir = tmp_path / "reports"

    exit_code = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
        ]
    )

    assert exit_code == 2
    assert registry_path.read_text(encoding="utf-8") == before_text
    payload = json.loads(next(output_dir.glob("factor_health_*.json")).read_text(encoding="utf-8"))
    assert payload["registry_update_status"] == "not_requested"
    assert any(
        "gate_ids_expected_1_to_8" in blocker for blocker in payload["fresh_evaluation"]["blockers"]
    )


def test_leave_one_out_composite_removes_candidate_self_inclusion():
    first = _production_record()
    second = _blend_record()
    registry = MinedFactorRegistry.from_records([first, second])
    index = pd.to_datetime(["2026-01-31"])
    columns = ["000001.SZ", "000002.SZ", "000003.SZ"]
    candidate_signal = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=index,
        columns=columns,
    )
    other_signal = pd.DataFrame(
        [[3.0, 1.0, 2.0]],
        index=index,
        columns=columns,
    )

    def signal_builder(candidate, _context):
        if candidate.name == first.name:
            return candidate_signal
        return other_signal

    existing, blocker = factor_health_automation._compute_existing_price_volume_composite(
        registry,
        candidate_signal,
        candidate_signal,
        candidate_signal,
        candidate_type=SimpleNamespace,
        signal_builder=signal_builder,
    )
    assert blocker == ""

    leave_one_out, blocker = factor_health_automation._leave_one_out_existing_composite(
        existing,
        candidate_signal,
        first,
        registry,
    )

    assert blocker == ""
    expected = other_signal.rank(axis=1, pct=True).mul(2.0).sub(1.0)
    pd.testing.assert_frame_equal(leave_one_out, expected)


def test_retired_health_apply_flag_blocks_before_report_or_manifest_write(
    monkeypatch,
    tmp_path,
):
    record = _production_record()
    registry_path = tmp_path / "mined_factors.json"
    before = {
        "schema_version": "mined-factor-registry.v1",
        "metadata": {"fixture": True},
        "factors": [record.to_dict()],
    }
    registry_path.write_text(json.dumps(before), encoding="utf-8")
    fresh = factor_health_automation._evaluation_from_record(record)
    assert fresh is not None
    fresh["diagnostics"]["evaluation_end_date"] = "2026-06-30"
    monkeypatch.setattr(
        factor_health_automation,
        "_fresh_evaluations",
        lambda _args, _factors: {
            "evaluations": {record.name: fresh},
            "blockers": [],
            "context": {"snapshot_id": "snap-atomic"},
        },
    )
    monkeypatch.setattr(
        factor_health_automation,
        "build_runtime_smoke",
        lambda *_args, **_kwargs: {
            "backend": "parquet",
            "mode_policy": "strict",
            "fallback_used": False,
            "factor_mode": "governed_mined_factors",
            "factor_count": 1,
            "coverage_rate": 1.0,
            "snapshot_healthy": True,
            "symbols_loaded": 1,
        },
    )
    output_dir = tmp_path / "reports"

    exit_code = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
            "--apply-registry-actions",
        ]
    )

    assert exit_code == 2
    assert not output_dir.exists()
    written = json.loads(registry_path.read_text(encoding="utf-8"))
    assert written == before


def test_retired_health_apply_never_reaches_legacy_cas_writer(
    monkeypatch,
    tmp_path,
):
    record = _production_record()
    registry_path = tmp_path / "mined_factors.json"
    before_text = json.dumps(
        {
            "schema_version": "mined-factor-registry.v1",
            "metadata": {"fixture": True},
            "factors": [record.to_dict()],
        }
    )
    registry_path.write_text(before_text, encoding="utf-8")
    fresh = factor_health_automation._evaluation_from_record(record)
    assert fresh is not None
    fresh["diagnostics"]["evaluation_end_date"] = "2026-06-30"
    monkeypatch.setattr(
        factor_health_automation,
        "_fresh_evaluations",
        lambda _args, _factors: {
            "evaluations": {record.name: fresh},
            "blockers": [],
            "context": {"snapshot_id": "snap-conflict"},
        },
    )
    monkeypatch.setattr(
        factor_health_automation,
        "build_runtime_smoke",
        lambda *_args, **_kwargs: {
            "backend": "parquet",
            "mode_policy": "strict",
            "fallback_used": False,
            "factor_mode": "governed_mined_factors",
            "factor_count": 1,
            "coverage_rate": 1.0,
            "snapshot_healthy": True,
            "symbols_loaded": 1,
        },
    )

    assert not hasattr(factor_health_automation, "apply_factor_record_patch")
    output_dir = tmp_path / "reports"

    exit_code = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
            "--apply-registry-actions",
        ]
    )

    assert exit_code == 2
    assert registry_path.read_text(encoding="utf-8") == before_text
    assert not output_dir.exists()


def test_strict_runtime_smoke_failure_blocks_apply_and_report_only(
    monkeypatch,
    tmp_path,
):
    record = _production_record()
    registry_path = tmp_path / "mined_factors.json"
    before_text = json.dumps(
        {
            "schema_version": "mined-factor-registry.v1",
            "metadata": {"fixture": True},
            "factors": [record.to_dict()],
        }
    )
    registry_path.write_text(before_text, encoding="utf-8")
    fresh = factor_health_automation._evaluation_from_record(record)
    assert fresh is not None
    fresh["diagnostics"]["evaluation_end_date"] = "2026-06-30"
    monkeypatch.setattr(
        factor_health_automation,
        "_fresh_evaluations",
        lambda _args, _factors: {
            "evaluations": {record.name: fresh},
            "blockers": [],
            "context": {"snapshot_id": "snap-runtime-broken"},
        },
    )
    monkeypatch.setattr(
        factor_health_automation,
        "build_runtime_smoke",
        lambda *_args, **_kwargs: {
            "backend": "csv",
            "mode_policy": "permissive",
            "fallback_used": True,
            "factor_mode": "error",
            "factor_count": 0,
            "coverage_rate": 0.0,
            "snapshot_healthy": False,
            "symbols_loaded": 0,
            "error": "fixture runtime failure",
        },
    )
    apply_output = tmp_path / "apply_reports"

    apply_exit = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(apply_output),
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
            "--apply-registry-actions",
        ]
    )

    assert apply_exit == 2
    assert registry_path.read_text(encoding="utf-8") == before_text
    assert not apply_output.exists()

    report_output = tmp_path / "report_only"
    report_exit = factor_health_automation.main(
        [
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(report_output),
            "--fresh-evaluation",
            "--strict-fresh-evaluation",
        ]
    )
    assert report_exit == 2
    assert registry_path.read_text(encoding="utf-8") == before_text
    report = next(report_output.glob("factor_health_*.json"))
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["registry_update_status"] == "not_requested"
    assert payload["run_status"] == "blocked"
    assert "runtime_smoke_snapshot_unhealthy" in payload["runtime_smoke_blockers"]
    assert "runtime_smoke_backend:csv" in payload["runtime_smoke_blockers"]
    assert "runtime_smoke_mode_policy:permissive" in payload["runtime_smoke_blockers"]
    assert "runtime_smoke_fallback_used" in payload["runtime_smoke_blockers"]
    assert any(
        item.startswith("runtime_smoke_error:") for item in payload["runtime_smoke_blockers"]
    )


def _write_parquet_fixture(root: Path) -> None:
    table_root, serving_root, manifest = v4_snapshot_paths(root, "snap-001")
    canonical = table_root / "year=2026" / "month=01"
    serving_1 = serving_root / "symbol=000001.SZ"
    serving_2 = serving_root / "symbol=000002.SZ"
    snapshot_dir = manifest.parent
    universe_dir = root / "cn_universe"
    canonical.mkdir(parents=True, exist_ok=True)
    serving_1.mkdir(parents=True, exist_ok=True)
    serving_2.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    universe_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260101",
                "open": 9.5,
                "high": 10.2,
                "low": 9.3,
                "close": 10.0,
                "vol": 1000,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260102",
                "open": 10.0,
                "high": 11.2,
                "low": 9.8,
                "close": 11.0,
                "vol": 1100,
            },
            {
                "ts_code": "000002.SZ",
                "trade_date": "20260101",
                "open": 19.5,
                "high": 20.2,
                "low": 19.3,
                "close": 20.0,
                "vol": 2000,
            },
            {
                "ts_code": "000002.SZ",
                "trade_date": "20260102",
                "open": 20.0,
                "high": 21.2,
                "low": 19.8,
                "close": 21.0,
                "vol": 2100,
            },
        ]
    )
    frame.to_parquet(canonical / "part.parquet", index=False)
    frame[frame["ts_code"].eq("000001.SZ")].to_parquet(
        serving_1 / "bars.parquet",
        index=False,
    )
    frame[frame["ts_code"].eq("000002.SZ")].to_parquet(
        serving_2 / "bars.parquet",
        index=False,
    )

    data_root = root
    coverage = coverage_v4(
        data_root,
        ["000001.SZ", "000002.SZ"],
        trade_date="20260103",
    )
    manifest.write_text(
        json.dumps({"snapshot_id": "snap-001", "coverage": coverage}, ensure_ascii=False),
        encoding="utf-8",
    )
    latest = {
        "status": "OK",
        "snapshot_id": "snap-001",
        "latest_complete_trade_date": "20260103",
        "latest_trade_date": "20260103",
        "table_root": str(table_root),
        "derived_serving_root": str(serving_root),
        "manifest_path": str(manifest),
        "coverage": coverage,
        "blockers": [],
    }
    latest_path = root / "parquet" / "cn" / "_latest.json"
    latest_path.write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")
    components = {
        "full_a": ["000001.SZ", "000002.SZ"],
        "hs300": ["000001.SZ"],
        "zz500": ["000002.SZ"],
    }
    (universe_dir / "cn_index_components.json").write_text(
        json.dumps(components, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_parquet_fresh_fixture(root: Path) -> None:
    table_root, serving_root, manifest = v4_snapshot_paths(root, "snap-fresh")
    canonical = table_root / "year=2026" / "month=01"
    serving_1 = serving_root / "symbol=000001.SZ"
    serving_2 = serving_root / "symbol=000002.SZ"
    snapshot_dir = manifest.parent
    universe_dir = root / "cn_universe"
    canonical.mkdir(parents=True, exist_ok=True)
    serving_1.mkdir(parents=True, exist_ok=True)
    serving_2.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    universe_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for day in range(1, 12):
        trade_date = f"202601{day:02d}"
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0 + day,
                "high": 10.5 + day,
                "low": 9.5 + day,
                "close": 10.0 + day,
                "vol": 1000 + day,
            }
        )
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": trade_date,
                "open": 20.0 + day,
                "high": 20.5 + day,
                "low": 19.5 + day,
                "close": 20.0 + day,
                "vol": 2000 + day,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(canonical / "part.parquet", index=False)
    frame[frame["ts_code"].eq("000001.SZ")].to_parquet(
        serving_1 / "bars.parquet",
        index=False,
    )
    frame[frame["ts_code"].eq("000002.SZ")].to_parquet(
        serving_2 / "bars.parquet",
        index=False,
    )

    data_root = root
    coverage = coverage_v4(
        data_root,
        ["000001.SZ", "000002.SZ"],
        trade_date="20260111",
    )
    manifest.write_text(
        json.dumps({"snapshot_id": "snap-fresh", "coverage": coverage}, ensure_ascii=False),
        encoding="utf-8",
    )
    latest = {
        "status": "OK",
        "snapshot_id": "snap-fresh",
        "latest_complete_trade_date": "20260111",
        "latest_trade_date": "20260111",
        "table_root": str(table_root),
        "derived_serving_root": str(serving_root),
        "manifest_path": str(manifest),
        "coverage": coverage,
        "blockers": [],
    }
    latest_path = root / "parquet" / "cn" / "_latest.json"
    latest_path.write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")
    components = {
        "full_a": ["000001.SZ", "000002.SZ"],
        "hs300": ["000001.SZ"],
        "zz500": ["000002.SZ"],
    }
    (universe_dir / "cn_index_components.json").write_text(
        json.dumps(components, ensure_ascii=False),
        encoding="utf-8",
    )
