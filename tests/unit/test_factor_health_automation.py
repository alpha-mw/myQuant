from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

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
    assert payload["runtime_smoke"]["backend"] == "parquet"
    assert payload["runtime_smoke"]["factor_mode"] == "parquet_canonical_unavailable"
    assert (
        "strict Parquet snapshot pointer missing"
        in payload["runtime_smoke"]["error"]
    )


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


def _write_parquet_fixture(root: Path) -> None:
    canonical = root / "parquet" / "cn" / "bars" / "year=2026" / "month=01"
    serving_1 = root / "parquet_serving" / "cn" / "bars" / "symbol=000001.SZ"
    serving_2 = root / "parquet_serving" / "cn" / "bars" / "symbol=000002.SZ"
    snapshot_dir = root / "parquet" / "cn" / "_snapshots"
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

    manifest = snapshot_dir / "snap-001.json"
    manifest.write_text(
        json.dumps({"snapshot_id": "snap-001"}, ensure_ascii=False),
        encoding="utf-8",
    )
    latest = {
        "status": "OK",
        "snapshot_id": "snap-001",
        "latest_complete_trade_date": "20260103",
        "latest_trade_date": "20260103",
        "table_root": str(root / "parquet" / "cn" / "bars"),
        "derived_serving_root": str(root / "parquet_serving" / "cn" / "bars"),
        "manifest_path": str(manifest),
        "coverage": {"row_count": int(len(frame)), "symbol_count": 2},
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
    canonical = root / "parquet" / "cn" / "bars" / "year=2026" / "month=01"
    serving_1 = root / "parquet_serving" / "cn" / "bars" / "symbol=000001.SZ"
    serving_2 = root / "parquet_serving" / "cn" / "bars" / "symbol=000002.SZ"
    snapshot_dir = root / "parquet" / "cn" / "_snapshots"
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

    manifest = snapshot_dir / "snap-fresh.json"
    manifest.write_text(
        json.dumps({"snapshot_id": "snap-fresh"}, ensure_ascii=False),
        encoding="utf-8",
    )
    latest = {
        "status": "OK",
        "snapshot_id": "snap-fresh",
        "latest_complete_trade_date": "20260111",
        "latest_trade_date": "20260111",
        "table_root": str(root / "parquet" / "cn" / "bars"),
        "derived_serving_root": str(root / "parquet_serving" / "cn" / "bars"),
        "manifest_path": str(manifest),
        "coverage": {"row_count": int(len(frame)), "symbol_count": 2},
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
