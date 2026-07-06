from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.themes.storage import ThemeSnapshotStore
from scripts.run_theme_threshold_sweep import (
    ThemeThresholdSweepConfig,
    run_threshold_sweep,
)


def _frame(closes: list[float], *, start: str = "2026-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range(start, periods=len(closes), freq="D"),
            "close": closes,
        }
    )


def _save_snapshot(root: Path, *, as_of: str = "20260101") -> None:
    ThemeSnapshotStore(root).save(
        {
            "status": "success",
            "market": "CN",
            "universe_key": "full_a",
            "as_of": as_of,
            "symbol_scores": {"000001.SZ": 0.78, "000002.SZ": 0.42},
            "symbol_primary_theme": {
                "000001.SZ": "industry::ai",
                "000002.SZ": "industry::banking",
            },
            "symbol_phase": {
                "000001.SZ": "confirmed_rotation",
                "000002.SZ": "distribution",
            },
            "symbol_risk_flags": {
                "000001.SZ": [],
                "000002.SZ": ["theme_distribution_risk"],
            },
            "theme_scores": {
                "industry::ai": {
                    "theme_name": "AI",
                    "score": 76.0,
                    "confidence": 0.82,
                    "member_count": 12,
                },
                "industry::banking": {
                    "theme_name": "Banking",
                    "score": 42.0,
                    "confidence": 0.51,
                    "member_count": 9,
                },
            },
        },
        market="CN",
        universe_key="full_a",
        as_of=as_of,
        run_id="synthetic",
    )


def test_theme_threshold_sweep_runner_outputs_deterministic_schema(tmp_path: Path):
    snapshot_dir = tmp_path / "snapshots"
    output_dir = tmp_path / "calibration"
    _save_snapshot(snapshot_dir)
    frames = {
        "000001.SZ": _frame([10.0, 10.5, 10.8, 11.0, 11.5, 12.0]),
        "000002.SZ": _frame([10.0, 9.8, 9.6, 9.4, 9.2, 9.0]),
    }
    config = ThemeThresholdSweepConfig(
        snapshot_dir=snapshot_dir,
        output_dir=output_dir,
        market="CN",
        universe_key="full_a",
        history_limit=10,
        min_sample=1,
    )

    first = run_threshold_sweep(config, frames=frames)
    first_json = Path(first["json_path"]).read_bytes()
    first_md = Path(first["markdown_path"]).read_bytes()
    second = run_threshold_sweep(config, frames=frames)

    assert Path(second["json_path"]).read_bytes() == first_json
    assert Path(second["markdown_path"]).read_bytes() == first_md

    payload = json.loads(first_json.decode("utf-8"))
    assert payload["schema_version"] == "theme_threshold_sweep.v1"
    assert payload["metadata"]["deterministic"] is True
    assert payload["metadata"]["offline_only"] is True
    assert payload["metadata"]["snapshot_count"] == 1
    assert payload["dataset"]["record_count"] == 2
    assert payload["grid_parameters"]["phase_score_gates"]["current"] == [35, 55, 70]
    assert payload["grid_parameters"]["crowding_weights"]["current"]["limitup_norm"] == 0.35
    assert payload["threshold_rows"]
    assert {
        "threshold_name",
        "threshold_value",
        "selected_count",
        "available_count",
        "avg_forward_alpha_5d",
        "avg_forward_alpha_5d_gross",
        "avg_forward_alpha_5d_net",
        "avg_forward_alpha_10d_gross",
        "avg_forward_alpha_10d_net",
        "avg_forward_alpha_20d_gross",
        "avg_forward_alpha_20d_net",
        "hit_rate_5d",
        "recommended_action",
    }.issubset(payload["threshold_rows"][0])
    assert payload["threshold_rows"][0]["avg_forward_alpha_5d_net"] == (
        payload["threshold_rows"][0]["avg_forward_alpha_5d_gross"]
    )
    assert "## Theme Threshold Sweep" in Path(first["markdown_path"]).read_text(encoding="utf-8")


def test_theme_threshold_sweep_runner_outputs_net_of_cost_alpha(tmp_path: Path):
    snapshot_dir = tmp_path / "snapshots"
    output_dir = tmp_path / "calibration"
    _save_snapshot(snapshot_dir)
    frames = {
        "000001.SZ": _frame([10.0, 10.5, 10.8, 11.0, 11.5, 12.0]),
        "000002.SZ": _frame([10.0, 9.8, 9.6, 9.4, 9.2, 9.0]),
    }
    config = ThemeThresholdSweepConfig(
        snapshot_dir=snapshot_dir,
        output_dir=output_dir,
        market="CN",
        universe_key="full_a",
        history_limit=10,
        min_sample=1,
        execution_cost_bps=25.0,
    )

    result = run_threshold_sweep(config, frames=frames)
    payload = result["payload"]
    row = next(
        item
        for item in payload["threshold_rows"]
        if item["threshold_name"] == "symbol_theme_score >= 0.70"
    )

    assert row["avg_forward_alpha_5d_net"] == pytest.approx(
        row["avg_forward_alpha_5d_gross"] - 0.0025
    )
    assert payload["metadata"]["execution_cost_bps"] == 25.0
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert "net_alpha5" in markdown
