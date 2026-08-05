"""A mining run must report its own selection bias, not just its winner.

The run evaluates every candidate and then reports the best.  These metrics
record how many trials that took, how wide the trial Sharpes were, and whether
the leader still stands after both are accounted for.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_investor.factors.trial_correction import (
    DEFAULT_DSR_FLOOR,
    HARVEY_LIU_ZHU_T_HURDLE,
)
from scripts.mine_quant_branch_factors import (
    _set_trial_correction,
    build_trial_correction_evidence,
    qualified_candidates,
)

DATES = pd.bdate_range("2021-01-04", periods=900)


def _ic_series(mean: float, std: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, size=len(DATES)), index=DATES)


def _candidate(
    name: str,
    *,
    icir: float,
    ic_series: pd.Series,
    gates: int = 5,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": "test_family",
        "gates_passed": gates,
        "decision": "production_candidate",
        "metrics": {"icir": icir, "pool_residual_icir": icir},
        "_ic_series": ic_series,
    }


def _results() -> list[dict[str, Any]]:
    strong = _candidate(
        "strong", icir=3.0, ic_series=_ic_series(0.06, 0.02, seed=1)
    )
    weak = [
        _candidate(
            f"weak_{index}",
            icir=0.05,
            ic_series=_ic_series(0.0, 0.05, seed=100 + index),
        )
        for index in range(20)
    ]
    return [strong, *weak]


def test_reparameterisations_do_not_inflate_the_deflated_sharpe_bar() -> None:
    """Twenty variants of one idea must not cost twenty trials.

    The weak candidates here are near-copies of each other, so the effective
    trial count collapses and the bar the strong candidate has to clear drops
    with it.
    """

    base = _ic_series(0.0, 0.05, seed=500)
    rng = np.random.default_rng(501)
    clones = [
        _candidate(
            f"clone_{index}",
            icir=0.05,
            ic_series=base
            + pd.Series(
                rng.normal(0.0, 0.002, size=len(base)), index=base.index
            ),
        )
        for index in range(20)
    ]
    results = [
        _candidate("strong", icir=3.0, ic_series=_ic_series(0.06, 0.02, seed=1)),
        *clones,
    ]

    evidence = build_trial_correction_evidence(results)

    assert evidence["trial_count"] == 21
    assert evidence["effective_trial_count"] == 2
    assert evidence["largest_trial_cluster_size"] == 20


def test_the_effective_count_is_what_deflates_the_sharpe() -> None:
    results = _results()

    evidence = build_trial_correction_evidence(results)
    _set_trial_correction(results, evidence)
    strong = next(item for item in results if item["name"] == "strong")

    assert strong["metrics"]["effective_trial_count"] == (
        evidence["effective_trial_count"]
    )
    assert strong["metrics"]["trial_count"] == len(results)


def test_trial_correction_reports_the_size_of_the_search() -> None:
    results = _results()

    evidence = build_trial_correction_evidence(results)

    assert evidence["trial_count"] == len(results)
    assert evidence["trial_sharpe_std"] > 0.0
    assert evidence["dsr_floor"] == DEFAULT_DSR_FLOOR
    assert evidence["t_hurdle"] == HARVEY_LIU_ZHU_T_HURDLE


def test_a_consistent_edge_clears_both_corrections() -> None:
    results = _results()

    _set_trial_correction(results, build_trial_correction_evidence(results))
    strong = next(item for item in results if item["name"] == "strong")

    assert strong["metrics"]["nonoverlap_t_stat"] > HARVEY_LIU_ZHU_T_HURDLE
    assert strong["metrics"]["harvey_liu_zhu_passed"] is True
    assert strong["metrics"]["deflated_sharpe_ratio"] > DEFAULT_DSR_FLOOR
    assert strong["metrics"]["trial_corrected_passed"] is True


def test_noise_fails_both_corrections() -> None:
    results = _results()

    _set_trial_correction(results, build_trial_correction_evidence(results))
    weak = next(item for item in results if item["name"] == "weak_0")

    assert abs(weak["metrics"]["nonoverlap_t_stat"]) < HARVEY_LIU_ZHU_T_HURDLE
    assert weak["metrics"]["harvey_liu_zhu_passed"] is False
    assert weak["metrics"]["trial_corrected_passed"] is False


def test_trial_correction_records_the_trial_count_it_used() -> None:
    results = _results()

    _set_trial_correction(results, build_trial_correction_evidence(results))
    strong = next(item for item in results if item["name"] == "strong")

    assert strong["metrics"]["trial_count"] == len(results)


def test_a_candidate_without_an_ic_series_fails_closed() -> None:
    results = _results()
    results.append(
        {
            "name": "no_series",
            "family": "test_family",
            "gates_passed": 5,
            "decision": "production_candidate",
            "metrics": {"icir": 9.0, "pool_residual_icir": 9.0},
        }
    )

    _set_trial_correction(results, build_trial_correction_evidence(results))
    orphan = next(item for item in results if item["name"] == "no_series")

    assert orphan["metrics"]["trial_corrected_passed"] is False
    assert orphan["metrics"]["deflated_sharpe_ratio"] == 0.0


def test_pbo_is_computed_across_the_whole_candidate_set() -> None:
    results = _results()

    evidence = build_trial_correction_evidence(results)

    assert 0.0 <= evidence["pbo"] <= 1.0
    assert evidence["pbo_split_count"] > 0
    assert evidence["pbo_config_count"] == len(results)


def test_qualification_requires_both_the_gates_and_the_correction() -> None:
    results = _results()
    _set_trial_correction(results, build_trial_correction_evidence(results))

    names = {item["name"] for item in qualified_candidates(results)}

    assert "strong" in names
    assert not any(name.startswith("weak_") for name in names)


def test_the_correction_can_never_qualify_a_gate_rejection() -> None:
    results = _results()
    _set_trial_correction(results, build_trial_correction_evidence(results))
    strong = next(item for item in results if item["name"] == "strong")
    assert strong["metrics"]["trial_corrected_passed"] is True
    strong["decision"] = "reject"

    assert qualified_candidates(results) == []


def test_qualification_does_not_mutate_its_input() -> None:
    results = _results()
    _set_trial_correction(results, build_trial_correction_evidence(results))

    picked = qualified_candidates(results)
    picked[0]["decision"] = "tampered"

    assert all(item["decision"] != "tampered" for item in results)


def test_the_ic_series_is_not_left_in_the_serialised_result() -> None:
    results = _results()

    _set_trial_correction(results, build_trial_correction_evidence(results))

    assert all("_ic_series" not in item for item in results)
