from __future__ import annotations

import pandas as pd
import pytest

from quant_investor.v17 import runtime_pipeline
from quant_investor.v17.semantic import seal_semantic


def test_risk_model_rejects_zero_stress_shock(monkeypatch: pytest.MonkeyPatch) -> None:
    cutoff = "2026-07-22T07:00:00Z"
    source = seal_semantic(
        {
            "version": runtime_pipeline.RISK_MODEL_INPUT_SOURCE_VERSION,
            "market": "CN",
            "cutoff": cutoff,
            "market_snapshot_sha256": "a" * 64,
            "benchmark_symbol": "H00300.CSI",
            "algorithm": "OLS_BETA_V1",
            "beta_window": 20,
            "stress_scenario": "stress-v1",
            "benchmark_stress_shock": 0.0,
            "authority": False,
        }
    )
    monkeypatch.setattr(
        runtime_pipeline,
        "_read_bound_source",
        lambda _bundle, _role: source,
    )
    monkeypatch.setattr(
        runtime_pipeline,
        "_source_sha",
        lambda _bundle, _role: "a" * 64,
    )

    with pytest.raises(
        runtime_pipeline.V17PipelineError,
        match="benchmark_stress_shock must be negative",
    ):
        runtime_pipeline._derive_risk_model_attributes(
            object(),  # type: ignore[arg-type]
            cutoff,
            frames={},
            benchmark_frame=pd.DataFrame(),
            risk_policy={"stress_scenario": "stress-v1"},
        )
