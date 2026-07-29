from __future__ import annotations

import json
import math
from typing import Any

import pytest

from quant_investor.v17_v4_runtime.forward_scoring_v3 import (
    FUNDAMENTAL_COMPONENT_WEIGHTS,
    ForwardScoringV3Error,
    average_tie_percentiles_v3,
    fuse_forward_scores_v3,
    score_fundamental_forward_v3,
    score_quant_forward_v3,
    type7_quantile_v3,
    winsorize_type7_v3,
)

CUTOFF = "2026-07-29T07:00:00Z"
SYMBOLS = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")


def _neutralizers(
    symbols: tuple[str, ...] = SYMBOLS,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        symbol: {
            "industry": {"available_at": CUTOFF, "value": f"industry-{index % 2}"},
            "log_market_cap": {"available_at": CUTOFF, "value": 10 + index},
            "beta_252d": {"available_at": CUTOFF, "value": 0.8 + index * 0.1},
            "amihud_20d": {"available_at": CUTOFF, "value": 0.01 + index * 0.01},
        }
        for index, symbol in enumerate(symbols)
    }


def test_type7_winsor_and_average_tie_percentiles_are_exact() -> None:
    values = list(range(100))
    assert type7_quantile_v3(values, 0.01) == pytest.approx(0.99)
    assert type7_quantile_v3(values, 0.99) == pytest.approx(98.01)
    winsorized = winsorize_type7_v3(values)
    assert winsorized[0] == pytest.approx(0.99)
    assert winsorized[-1] == pytest.approx(98.01)
    assert average_tie_percentiles_v3({"a": 1, "b": 1, "c": 4}) == {
        "a": pytest.approx(0.5),
        "b": pytest.approx(0.5),
        "c": pytest.approx(1.0),
    }


def test_quant_zero_mad_is_available_zero_exposure_not_missing_fill() -> None:
    result = score_quant_forward_v3(
        symbols=SYMBOLS,
        selected_factors=(
            {"family": "quality", "name": "constant-quality"},
            {"family": "value", "name": "constant-value"},
        ),
        factor_values={
            "constant-quality": {symbol: 7 for symbol in SYMBOLS},
            "constant-value": {symbol: -2 for symbol in SYMBOLS},
        },
        neutralizer_inputs=_neutralizers(),
        cutoff=CUTOFF,
    )
    for row in result["records"]:
        assert row["status"] == "AVAILABLE"
        assert row["coverage"] == 1.0
        assert row["raw_composite_score"] == 0.0
        assert row["effective_score"] == 0.0
        assert {item["status"] for item in row["factor_evidence"]} == {"ZERO_MAD"}
        assert all(item["exposure"] == 0.0 for item in row["factor_evidence"])


def test_quant_coverage_boundary_and_missing_values_are_not_zero_filled() -> None:
    factors = tuple({"family": f"family-{index}", "name": f"factor-{index}"} for index in range(4))
    values = {
        f"factor-{index}": {
            symbol: float(symbol_index + index) for symbol_index, symbol in enumerate(SYMBOLS)
        }
        for index in range(4)
    }
    values["factor-2"].pop(SYMBOLS[0])
    values["factor-3"].pop(SYMBOLS[0])
    result = score_quant_forward_v3(
        symbols=SYMBOLS,
        selected_factors=factors,
        factor_values=values,
        neutralizer_inputs=_neutralizers(),
        cutoff=CUTOFF,
    )
    row = result["records"][0]
    assert row["available_factor_count"] == 2
    assert row["available_family_count"] == 2
    assert row["factor_coverage"] == 0.5
    assert row["family_coverage"] == 0.5
    assert row["coverage"] == 0.5
    assert row["status"] == "AVAILABLE"
    assert row["effective_score"] == pytest.approx(row["raw_composite_score"] * 0.5)
    missing = [
        evidence for evidence in row["factor_evidence"] if evidence["status"] == "MISSING_VALUE"
    ]
    assert [evidence["factor_name"] for evidence in missing] == [
        "factor-2",
        "factor-3",
    ]
    assert all(evidence["exposure"] is None for evidence in missing)


def test_quant_family_gate_and_neutralizer_cutoff_fail_closed() -> None:
    selected = (
        {"family": "same-family", "name": "factor-a"},
        {"family": "same-family", "name": "factor-b"},
        {"family": "other-1", "name": "factor-c"},
        {"family": "other-2", "name": "factor-d"},
    )
    values = {
        "factor-a": {symbol: index for index, symbol in enumerate(SYMBOLS)},
        "factor-b": {symbol: index + 1 for index, symbol in enumerate(SYMBOLS)},
        "factor-c": {symbol: index + 2 for index, symbol in enumerate(SYMBOLS[1:], start=1)},
        "factor-d": {symbol: index + 3 for index, symbol in enumerate(SYMBOLS[1:], start=1)},
    }
    result = score_quant_forward_v3(
        symbols=SYMBOLS,
        selected_factors=selected,
        factor_values=values,
        neutralizer_inputs=_neutralizers(),
        cutoff=CUTOFF,
    )
    first = result["records"][0]
    assert first["available_factor_count"] == 2
    assert first["available_family_count"] == 1
    assert first["status"] == "UNAVAILABLE"
    assert "AVAILABLE_FAMILY_COUNT_BELOW_2" in first["unavailability_reasons"]

    future = _neutralizers()
    future[SYMBOLS[0]]["beta_252d"]["available_at"] = "2026-07-29T07:00:01Z"
    with pytest.raises(ForwardScoringV3Error, match="beta_252d_after_cutoff"):
        score_quant_forward_v3(
            symbols=SYMBOLS,
            selected_factors=selected,
            factor_values=values,
            neutralizer_inputs=future,
            cutoff=CUTOFF,
        )


def test_quant_sequential_joint_residual_is_orthogonal_to_last_stage() -> None:
    symbols = tuple(f"S{index}" for index in range(8))
    neutralizers = {
        symbol: {
            "industry": {
                "available_at": CUTOFF,
                "value": "industry-a" if index < 4 else "industry-b",
            },
            "log_market_cap": {
                "available_at": CUTOFF,
                "value": [8.0, 9.5, 9.0, 11.0, 8.5, 10.5, 12.0, 9.2][index],
            },
            "beta_252d": {
                "available_at": CUTOFF,
                "value": [0.4, 1.2, 0.7, 1.5, 0.9, 0.3, 1.1, 1.7][index],
            },
            "amihud_20d": {
                "available_at": CUTOFF,
                "value": [0.08, 0.01, 0.12, 0.03, 0.07, 0.15, 0.02, 0.1][index],
            },
        }
        for index, symbol in enumerate(symbols)
    }
    result = score_quant_forward_v3(
        symbols=symbols,
        selected_factors=(
            {"family": "quality", "name": "signal-a"},
            {"family": "value", "name": "signal-b"},
        ),
        factor_values={
            "signal-a": {
                symbol: value
                for symbol, value in zip(
                    symbols,
                    (3.0, -1.0, 6.0, 2.0, 8.0, 0.5, 5.0, 4.0),
                    strict=True,
                )
            },
            "signal-b": {
                symbol: value
                for symbol, value in zip(
                    symbols,
                    (-2.0, 4.0, 1.0, 7.0, 0.0, 5.0, 9.0, 3.0),
                    strict=True,
                )
            },
        },
        neutralizer_inputs=neutralizers,
        cutoff=CUTOFF,
    )
    exposures = {row["symbol"]: row["factor_evidence"][0]["exposure"] for row in result["records"]}
    beta = {symbol: neutralizers[symbol]["beta_252d"]["value"] for symbol in symbols}
    amihud = {symbol: neutralizers[symbol]["amihud_20d"]["value"] for symbol in symbols}
    assert sum(exposures.values()) == pytest.approx(0.0, abs=1e-12)
    assert sum(exposures[symbol] * beta[symbol] for symbol in symbols) == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert sum(exposures[symbol] * amihud[symbol] for symbol in symbols) == pytest.approx(
        0.0, abs=1e-12
    )
    assert all(
        row["factor_evidence"][0]["industry_residual"] is not None
        and row["factor_evidence"][0]["log_market_cap_residual"] is not None
        for row in result["records"]
    )


def test_fundamental_uses_authoritative_component_names_weights_and_boundaries() -> None:
    assert {name: str(weight) for name, weight in FUNDAMENTAL_COMPONENT_WEIGHTS.items()} == {
        "financial_quality": "0.25",
        "industry_cycle": "0.25",
        "earnings_revision": "0.20",
        "theme_narrative": "0.10",
        "valuation": "0.15",
        "governance": "0.05",
    }
    result = score_fundamental_forward_v3(
        symbols=SYMBOLS[:3],
        financial_quality_values={},
        owner_component_scores={
            "industry_cycle": {SYMBOLS[0]: 0.8},
            "theme_narrative": {SYMBOLS[1]: 0.9},
            "governance": {SYMBOLS[1]: 0.3},
        },
        cutoff=CUTOFF,
    )
    first, second, third = result["records"]
    assert first["status"] == "PARTIAL"
    assert first["coverage"] == 0.25
    assert first["raw_score"] == pytest.approx(0.8)
    assert first["effective_score"] == pytest.approx(0.2)
    assert first["score_present"] is True
    assert second["status"] == "UNAVAILABLE"
    assert second["coverage"] == pytest.approx(0.15)
    assert second["raw_score"] == pytest.approx(0.7)
    assert second["effective_score"] == pytest.approx(0.105)
    assert second["score_present"] is False
    assert third["status"] == "UNAVAILABLE"
    assert third["coverage"] == 0.0
    assert third["raw_score"] is None
    assert third["effective_score"] is None


def test_fundamental_complete_coverage_preserves_raw_as_effective() -> None:
    result = score_fundamental_forward_v3(
        symbols=SYMBOLS[:3],
        financial_quality_values={
            "roe": {symbol: 0.1 + index * 0.1 for index, symbol in enumerate(SYMBOLS[:3])},
            "ocf_to_profit": {
                symbol: 0.8 + index * 0.2 for index, symbol in enumerate(SYMBOLS[:3])
            },
            "debt_to_assets": {
                symbol: 0.5 - index * 0.1 for index, symbol in enumerate(SYMBOLS[:3])
            },
        },
        owner_component_scores={
            component: {SYMBOLS[0]: score}
            for component, score in {
                "industry_cycle": 0.9,
                "earnings_revision": 0.8,
                "theme_narrative": 0.7,
                "valuation": 0.6,
                "governance": 0.5,
            }.items()
        },
        cutoff=CUTOFF,
    )
    first = result["records"][0]
    assert first["status"] == "COMPLETE"
    assert first["coverage"] == 1.0
    assert first["score_present"] is True
    assert first["effective_score"] == pytest.approx(first["raw_score"])


def test_financial_quality_ranks_ties_reverses_debt_and_accepts_two_metrics() -> None:
    result = score_fundamental_forward_v3(
        symbols=SYMBOLS[:3],
        financial_quality_values={
            "roe": {symbol: 0.2 for symbol in SYMBOLS[:3]},
            "ocf_to_profit": {symbol: 1.0 for symbol in SYMBOLS[:3]},
            "debt_to_assets": {
                SYMBOLS[0]: 0.1,
                SYMBOLS[1]: 0.5,
                SYMBOLS[2]: None,
            },
        },
        owner_component_scores={},
        cutoff=CUTOFF,
    )
    first, _, third = result["records"]
    assert first["status"] == "PARTIAL"
    assert first["coverage"] == 0.25
    financial = first["component_evidence"][0]["evidence"]
    by_metric = {row["metric"]: row for row in financial["metrics"]}
    assert by_metric["roe"]["percentile"] == pytest.approx(2 / 3)
    assert by_metric["debt_to_assets"]["percentile"] == pytest.approx(0.5)
    assert by_metric["debt_to_assets"]["component_score"] == pytest.approx(0.5)
    assert third["component_evidence"][0]["evidence"]["available_metric_count"] == 2
    assert third["component_evidence"][0]["status"] == "AVAILABLE"


@pytest.mark.parametrize(
    ("financial", "owner", "pattern"),
    [
        (
            {"debt_to_assets": {SYMBOLS[0]: 1.01}},
            {},
            "debt_to_assets.000001.SZ_range",
        ),
        (
            {},
            {"valuation": {SYMBOLS[0]: float("nan")}},
            "valuation.000001.SZ_nonfinite",
        ),
        (
            {},
            {"governance": {SYMBOLS[0]: 1.1}},
            "governance.000001.SZ_range",
        ),
    ],
)
def test_fundamental_nonfinite_and_out_of_range_inputs_block(
    financial: dict[str, dict[str, Any]],
    owner: dict[str, dict[str, Any]],
    pattern: str,
) -> None:
    with pytest.raises(ForwardScoringV3Error, match=pattern):
        score_fundamental_forward_v3(
            symbols=SYMBOLS,
            financial_quality_values=financial,
            owner_component_scores=owner,
            cutoff=CUTOFF,
        )


def test_owner_pit_score_after_cutoff_blocks_without_inference() -> None:
    with pytest.raises(ForwardScoringV3Error, match="industry_cycle.*after_cutoff"):
        score_fundamental_forward_v3(
            symbols=SYMBOLS,
            financial_quality_values={},
            owner_component_scores={
                "industry_cycle": {
                    SYMBOLS[0]: {
                        "available_at": "2026-07-29T07:00:01Z",
                        "score": 0.7,
                    }
                }
            },
            cutoff=CUTOFF,
        )


def test_fusion_attenuates_fundamental_weight_and_ranks_effective_score() -> None:
    symbols = ("A", "B", "C")
    result = fuse_forward_scores_v3(
        symbols=symbols,
        quant_scores={symbol: 1.0 for symbol in symbols},
        fundamental_scores={"B": 0.8, "C": 0.8},
        fundamental_coverages={"B": 1.0, "C": 0.5},
    )
    assert [row["symbol"] for row in result["records"]] == ["B", "C", "A"]
    rows = {row["symbol"]: row for row in result["records"]}
    assert rows["A"]["coverage"] == 0.5
    assert rows["A"]["confidence_penalty"] == 0.5
    assert rows["A"]["raw_score"] == pytest.approx(2 / 3)
    assert rows["A"]["effective_score"] == pytest.approx(1 / 3)
    assert rows["A"]["branch_evidence"][1]["score"] is None
    assert rows["A"]["branch_evidence"][1]["percentile"] is None
    assert rows["B"]["coverage"] == 1.0
    assert rows["B"]["effective_score"] == pytest.approx(17 / 24)
    assert rows["C"]["coverage"] == 0.75
    assert rows["C"]["effective_score"] == pytest.approx(25 / 48)
    assert json.loads(json.dumps(result)) == result


def test_fusion_ties_use_symbol_ascii_and_never_zero_fill_fundamental() -> None:
    result = fuse_forward_scores_v3(
        symbols=("B", "A", "C"),
        quant_scores={"A": 5, "B": 5, "C": 5},
        fundamental_scores={},
        fundamental_coverages={},
    )
    assert [row["symbol"] for row in result["records"]] == ["A", "B", "C"]
    for row in result["records"]:
        fundamental = row["branch_evidence"][1]
        assert fundamental["status"] == "UNAVAILABLE"
        assert fundamental["score"] is None
        assert fundamental["percentile"] is None
        assert row["effective_score"] == pytest.approx(1 / 3)


def test_fusion_blocks_missing_quant_and_invalid_fundamental_coverage() -> None:
    with pytest.raises(ForwardScoringV3Error, match="quant_scores_quant_always"):
        fuse_forward_scores_v3(
            symbols=("A", "B"),
            quant_scores={"A": 1},
            fundamental_scores={},
            fundamental_coverages={},
        )
    with pytest.raises(ForwardScoringV3Error, match="fundamental_coverages.B_range"):
        fuse_forward_scores_v3(
            symbols=("A", "B"),
            quant_scores={"A": 1, "B": 2},
            fundamental_scores={"B": 0.9},
            fundamental_coverages={"B": 1.1},
        )


def test_no_result_contains_nonfinite_numbers() -> None:
    result = fuse_forward_scores_v3(
        symbols=("A",),
        quant_scores={"A": 1},
        fundamental_scores={},
        fundamental_coverages={},
    )
    numeric = [
        result["records"][0][key]
        for key in (
            "available_weight",
            "confidence_penalty",
            "coverage",
            "effective_score",
            "raw_score",
        )
    ]
    assert all(math.isfinite(value) for value in numeric)
