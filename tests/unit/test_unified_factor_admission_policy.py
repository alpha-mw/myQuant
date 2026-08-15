from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from quant_investor.factors.governance import admission as admission_module


def _series(values: np.ndarray) -> pd.Series:
    sessions = pd.bdate_range("2025-01-02", periods=360).strftime("%Y-%m-%d")
    return pd.Series(values, index=sessions, dtype=float)


def test_trial_icir_uses_sample_standard_deviation_and_fails_closed() -> None:
    series_by_configuration = {
        "configuration-a": _series(np.linspace(-0.02, 0.04, 360)),
        "configuration-b": _series(np.linspace(-0.01, 0.03, 360) + np.sin(np.arange(360)) * 0.002),
    }
    (
        _,
        sharpe_by_configuration,
        trial_sharpe_std,
        _,
        complete,
        finite_count,
    ) = admission_module._trial_metrics(
        series_by_configuration,
        [("configuration-a",), ("configuration-b",)],
    )
    finite = [
        float(sharpe_by_configuration[configuration_id])
        for configuration_id in sorted(sharpe_by_configuration)
    ]
    assert complete is True
    assert finite_count == 2
    assert trial_sharpe_std == float(np.std(np.asarray(finite), ddof=1))

    incomplete_series = dict(series_by_configuration)
    incomplete_series["configuration-b"] = _series(np.ones(360))
    _, sharpes, trial_std, _, complete, finite_count = admission_module._trial_metrics(
        incomplete_series,
        [("configuration-a",), ("configuration-b",)],
    )
    assert sharpes["configuration-b"] is None
    assert complete is False
    assert finite_count == 1
    assert trial_std == 0.0


def test_shrunk_ic_uses_all_purged_cpcv_paths() -> None:
    open_sessions = list(pd.bdate_range("2025-01-02", periods=390).strftime("%Y-%m-%d"))
    signal_sessions = open_sessions[:360]
    rank_ic = pd.Series(
        np.linspace(0.005, 0.015, 360),
        index=signal_sessions,
        dtype=float,
    )
    preliminary, _, _ = admission_module._preliminary_candidate_metrics(
        [
            {
                "configuration_id": "configuration-a",
                "factor_id": "factor-a",
                "family": "liquidity",
            }
        ],
        {"configuration-a": rank_ic},
        open_sessions,
        {"configuration-a": None},
        trial_sharpe_std=0.0,
        effective_trials=1,
        trial_icir_complete=False,
    )
    metrics = preliminary["configuration-a"]
    path_means = admission_module._cpcv_path_means(rank_ic)
    expected_mean = float(np.mean(path_means))
    assert metrics["path_count"] == 45
    assert np.isclose(metrics["mean_path_ic"], expected_mean)
    assert np.isclose(
        metrics["shrunk_ic"],
        expected_mean * 45 / (45 + 10),
    )


def test_redundancy_representative_uses_dsr_path_mean_then_ascii() -> None:
    cluster_id = "cluster-a"
    components = [("configuration-a", "configuration-b", "configuration-c")]
    cluster_ids = {configuration_id: cluster_id for configuration_id in components[0]}
    eligible = {configuration_id: True for configuration_id in components[0]}

    preliminary = {
        "configuration-a": {"dsr": 0.96, "mean_path_ic": 0.03},
        "configuration-b": {"dsr": 0.97, "mean_path_ic": 0.01},
        "configuration-c": {"dsr": 0.97, "mean_path_ic": 0.02},
    }
    assert (
        admission_module._cluster_representatives(
            components,
            cluster_ids,
            eligible,
            preliminary,
        )[cluster_id]
        == "configuration-c"
    )

    preliminary["configuration-b"]["mean_path_ic"] = 0.02
    assert (
        admission_module._cluster_representatives(
            components,
            cluster_ids,
            eligible,
            preliminary,
        )[cluster_id]
        == "configuration-b"
    )


def test_turnover_is_annualized_to_252_open_sessions() -> None:
    annualized = admission_module._annualized_turnover(
        {
            "configuration-a": {
                "total_turnover": "10.000000000000",
                "annualized_turnover": "7.000000000000",
            }
        },
        ["configuration-a"],
    )
    assert annualized == {"configuration-a": Decimal("7.000000000000")}


def test_active_factor_limit_keeps_deterministic_top_ten() -> None:
    representatives = {f"cluster-{index:02d}": f"configuration-{index:02d}" for index in range(11)}
    preliminary = {
        f"configuration-{index:02d}": {
            "dsr": 1.0 - index / 100,
            "mean_path_ic": 0.02,
        }
        for index in range(11)
    }
    selected = admission_module._limited_representatives(representatives, preliminary)
    assert len(selected) == 10
    assert "configuration-10" not in selected
    assert selected == {f"configuration-{index:02d}" for index in range(10)}
