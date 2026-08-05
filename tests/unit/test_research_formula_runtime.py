"""Run `research_formula:rank_blend` in production against a pinned baseline.

The registry's `formula_mom120_np_yoy_resid_*` are the only carriers of the
`formulaic_research` family, which v4 needs as its fifth. They were unreachable:
`research_formula:` was never in the production dispatcher, and their `right`
primitive residualizes against the live production set, which
`production_set_dependent_primitives` refuses outright.

Production admits them only in pinned form -- `residualize_right_against` names
the baseline factors and binds them with a content hash -- so the value is
reproducible from market data plus the spec. The unpinned `_resid_existing`
spelling stays refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.governance import (
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.residual_baseline import (
    RESIDUAL_BASELINE_SCHEMA_VERSION,
    baseline_sha256,
    build_baseline_composite,
    cross_sectional_residual,
    cs_rank_pct,
    validate_residual_baseline,
)
from quant_investor.factors.runtime import (
    PRODUCTION_IMPLEMENTATION_PREFIXES,
    PRODUCTION_RUNTIME_MODE,
    MinedFactorScorer,
)

_BASELINE_FACTORS = [
    {
        "name": "pv_low_dollar_volume_5d",
        "implementation": "price_volume:pv_low_dollar_volume_5d",
        "weight": 0.05,
        "direction": 1.0,
    }
]


def _pinned_baseline() -> dict:
    return {
        "schema_version": RESIDUAL_BASELINE_SCHEMA_VERSION,
        "factors": _BASELINE_FACTORS,
        "baseline_sha256": baseline_sha256(_BASELINE_FACTORS),
    }


def _frames(symbols: int = 40, rows: int = 200) -> dict[str, pd.DataFrame]:
    """Enough history for momentum_120 plus a distinct path per symbol."""

    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    rng = np.random.default_rng(11)
    frames = {}
    for index in range(symbols):
        drift = 0.0004 * (index - symbols / 2)
        steps = rng.standard_normal(rows) * 0.01 + drift
        close = 20.0 * np.exp(np.cumsum(steps))
        volume = rng.lognormal(12.0, 0.3, rows)
        symbol = f"{index:06d}.SZ"
        frames[symbol] = pd.DataFrame(
            {
                "symbol": [symbol] * rows,
                "trade_date": dates,
                "close": close,
                "adj_close": close,
                "volume": volume,
                "amount": close * volume,
            }
        )
    return frames


def _record(params: dict) -> FactorRecord:
    return FactorRecord(
        name="formula_probe",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="research_formula:rank_blend",
        weight=0.2,
        gate_results=[
            GateResult(gate_id=i, gate_key=f"gate{i}", title=f"Gate {i}", passed=True)
            for i in range(1, 9)
        ],
        metadata={"params": params},
    )


def _pinned_params(**overrides) -> dict:
    params = {
        "left": "momentum_120",
        "right": "momentum_60",  # price-only keeps this test hermetic
        "left_weight": 0.25,
        "residualize_right_against": _pinned_baseline(),
    }
    params.update(overrides)
    return params


def _scorer() -> MinedFactorScorer:
    return MinedFactorScorer(runtime_mode=PRODUCTION_RUNTIME_MODE)


# --- the factor now runs ----------------------------------------------------


def test_research_formula_is_allowlisted_in_production():
    assert "research_formula:" in PRODUCTION_IMPLEMENTATION_PREFIXES


def test_pinned_rank_blend_produces_a_real_cross_section():
    frames = _frames()

    values = _scorer()._compute_factor(_record(_pinned_params()), frames)

    assert set(values.index) == set(frames)
    assert values.notna().sum() >= 30
    assert float(values.std()) > 0.0


def test_pinned_rank_blend_matches_the_mining_arithmetic():
    """Recompute the blend from the published helpers and compare."""

    frames = _frames()
    scorer = _scorer()
    record = _record(_pinned_params())

    values = scorer._compute_factor(record, frames)

    left = scorer._price_volume_factor("pv_momentum_120d", frames)
    right = scorer._price_volume_factor("pv_momentum_60d", frames)
    baseline_raw = scorer._price_volume_factor("pv_low_dollar_volume_5d", frames)
    composite = build_baseline_composite(
        validate_residual_baseline(_pinned_baseline()),
        {"pv_low_dollar_volume_5d": baseline_raw},
    )
    residual = cross_sectional_residual(right, composite)
    expected = cs_rank_pct(left).mul(0.25) + cs_rank_pct(residual).mul(0.75)

    pd.testing.assert_series_equal(
        values.sort_index().astype(float),
        expected.sort_index().astype(float),
        check_names=False,
    )


def test_the_weight_actually_shifts_the_blend():
    frames = _frames()
    scorer = _scorer()

    low = scorer._compute_factor(_record(_pinned_params(left_weight=0.2)), frames)
    high = scorer._compute_factor(_record(_pinned_params(left_weight=0.8)), frames)

    assert not np.allclose(low.dropna().to_numpy(), high.dropna().to_numpy())


def test_residualization_actually_happened():
    """The blend must not equal the same blend against a raw right primitive."""

    frames = _frames()
    scorer = _scorer()

    values = scorer._compute_factor(_record(_pinned_params()), frames)
    left = scorer._price_volume_factor("pv_momentum_120d", frames)
    right = scorer._price_volume_factor("pv_momentum_60d", frames)
    unresidualized = cs_rank_pct(left).mul(0.25) + cs_rank_pct(right).mul(0.75)

    assert not np.allclose(
        values.dropna().to_numpy(),
        unresidualized.reindex(values.index).dropna().to_numpy(),
    )


# --- fail-closed paths ------------------------------------------------------


def test_an_unpinned_factor_is_still_refused():
    """The whole reason the guard exists."""

    record = _record(
        {"left": "momentum_120", "right": "fin_net_profit_yoy_resid_existing", "left_weight": 0.25}
    )

    with pytest.raises(ValueError, match="depends on the production set"):
        _scorer()._compute_factor(record, _frames())


def test_a_missing_baseline_is_refused_rather_than_silently_unresidualized():
    params = _pinned_params()
    del params["residualize_right_against"]

    with pytest.raises(ValueError, match="residual baseline"):
        _scorer()._compute_factor(_record(params), _frames())


def test_a_tampered_baseline_hash_is_refused():
    baseline = _pinned_baseline()
    baseline["baseline_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="baseline_sha256"):
        _scorer()._compute_factor(
            _record(_pinned_params(residualize_right_against=baseline)), _frames()
        )


def test_an_unknown_primitive_is_refused():
    with pytest.raises(ValueError, match="primitive"):
        _scorer()._compute_factor(
            _record(_pinned_params(left="not_a_primitive")), _frames()
        )


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_an_out_of_range_blend_weight_is_refused(weight):
    with pytest.raises(ValueError, match="weight"):
        _scorer()._compute_factor(
            _record(_pinned_params(left_weight=weight)), _frames()
        )


def test_an_unknown_research_formula_variant_is_refused():
    record = _record(_pinned_params())
    record.implementation = "research_formula:some_other_form"

    with pytest.raises(ValueError, match="research_formula"):
        _scorer()._compute_factor(record, _frames())
