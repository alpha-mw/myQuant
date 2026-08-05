"""Pin the residualization baseline so residual factors become replayable.

`formula_mom120_np_yoy_resid_*` residualize against `existing_composite`, which
`compute_existing_composite` derives from `registry.selectable_factors()`. That
makes the factor a function of the production set it is joining: promoting
anything rewrites it, and its recorded gate evidence stops describing what
production computes. `production_set_dependent_primitives` refuses that shape
outright.

A *pinned* baseline is the same arithmetic with the mutable input frozen: the
baseline factors are named explicitly and bound by a content hash, so the
residual is reproducible from market data plus the spec alone. Pinning the
baseline to what it actually was at mining time keeps the recorded 8-gate
evidence valid instead of requiring a re-gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.residual_baseline import (
    RESIDUAL_BASELINE_SCHEMA_VERSION,
    ResidualBaselineError,
    baseline_sha256,
    build_baseline_composite,
    cross_sectional_residual,
    cs_rank_pct,
    rank_normalize,
    validate_residual_baseline,
)

_MINING_TIME_BASELINE = [
    {
        "name": "pv_low_dollar_volume_5d",
        "implementation": "price_volume:pv_low_dollar_volume_5d",
        "weight": 0.05,
        "direction": 1.0,
    }
]


def _spec(factors=None, **overrides):
    factors = _MINING_TIME_BASELINE if factors is None else factors
    spec = {
        "schema_version": RESIDUAL_BASELINE_SCHEMA_VERSION,
        "factors": factors,
        "baseline_sha256": baseline_sha256(factors),
    }
    spec.update(overrides)
    return spec


# --- content hash -----------------------------------------------------------


def test_hash_is_stable_across_key_order_and_int_float_spelling():
    a = baseline_sha256([{"name": "f", "implementation": "price_volume:x", "weight": 1, "direction": 1}])
    b = baseline_sha256([{"direction": 1.0, "weight": 1.0, "implementation": "price_volume:x", "name": "f"}])

    assert a == b


def test_hash_changes_when_a_weight_changes():
    other = [{**_MINING_TIME_BASELINE[0], "weight": 0.06}]

    assert baseline_sha256(_MINING_TIME_BASELINE) != baseline_sha256(other)


def test_hash_changes_when_a_factor_is_added():
    extra = [*_MINING_TIME_BASELINE, {"name": "g", "implementation": "price_volume:y", "weight": 0.1, "direction": 1.0}]

    assert baseline_sha256(_MINING_TIME_BASELINE) != baseline_sha256(extra)


def test_hash_ignores_declaration_order():
    extra = {"name": "g", "implementation": "price_volume:y", "weight": 0.1, "direction": 1.0}

    assert baseline_sha256([_MINING_TIME_BASELINE[0], extra]) == baseline_sha256(
        [extra, _MINING_TIME_BASELINE[0]]
    )


# --- validation -------------------------------------------------------------


def test_accepts_the_mining_time_baseline():
    normalized = validate_residual_baseline(_spec())

    assert [item["name"] for item in normalized["factors"]] == ["pv_low_dollar_volume_5d"]
    assert normalized["baseline_sha256"] == baseline_sha256(_MINING_TIME_BASELINE)


def test_rejects_a_tampered_hash():
    with pytest.raises(ResidualBaselineError, match="baseline_sha256"):
        validate_residual_baseline(_spec(baseline_sha256="0" * 64))


def test_rejects_a_hash_that_no_longer_matches_edited_factors():
    """The whole point: editing the baseline must invalidate the pin."""

    spec = _spec()
    spec["factors"] = [{**_MINING_TIME_BASELINE[0], "weight": 0.99}]

    with pytest.raises(ResidualBaselineError, match="baseline_sha256"):
        validate_residual_baseline(spec)


def test_rejects_an_unknown_schema_version():
    with pytest.raises(ResidualBaselineError, match="schema_version"):
        validate_residual_baseline(_spec(schema_version="factor-residual-baseline.v99"))


def test_rejects_an_empty_baseline():
    """An empty baseline residualizes to NaN, which must not be silent."""

    with pytest.raises(ResidualBaselineError, match="at least one"):
        validate_residual_baseline(_spec(factors=[]))


@pytest.mark.parametrize(
    "implementation",
    [
        "builtin:whatever",
        # Allowlisted for production scoring, but not self-contained enough to
        # be a baseline: the v1 spec carries no expression binding.
        "aquant_expression:fund_fin_net_profit_yoy",
        # Would otherwise allow a baseline to depend on a factor that has its
        # own pinned baseline.
        "research_formula:rank_blend",
    ],
)
def test_rejects_a_baseline_factor_that_is_not_self_contained(implementation):
    factors = [{"name": "f", "implementation": implementation, "weight": 0.2, "direction": 1.0}]

    with pytest.raises(ResidualBaselineError, match="not allowlisted as a baseline"):
        validate_residual_baseline(_spec(factors=factors))


def test_baseline_validation_is_narrower_than_the_production_allowlist():
    """Executable in production does not imply usable as a pin."""

    from quant_investor.factors.runtime import is_production_allowlisted_implementation

    assert is_production_allowlisted_implementation("aquant_expression:x") is True
    with pytest.raises(ResidualBaselineError, match="not allowlisted as a baseline"):
        validate_residual_baseline(
            _spec(
                factors=[
                    {"name": "f", "implementation": "aquant_expression:x", "weight": 0.2, "direction": 1.0}
                ]
            )
        )


def test_rejects_a_zero_weight_baseline():
    factors = [{**_MINING_TIME_BASELINE[0], "weight": 0.0}]

    with pytest.raises(ResidualBaselineError, match="weight"):
        validate_residual_baseline(_spec(factors=factors))


def test_rejects_duplicate_baseline_factors():
    factors = [_MINING_TIME_BASELINE[0], dict(_MINING_TIME_BASELINE[0])]

    with pytest.raises(ResidualBaselineError, match="duplicate"):
        validate_residual_baseline(_spec(factors=factors))


def test_validation_does_not_mutate_the_caller_spec():
    spec = _spec()
    before = repr(spec)

    validate_residual_baseline(spec)

    assert repr(spec) == before


# --- rank normalization -----------------------------------------------------


def test_rank_normalize_matches_the_composite_convention():
    """`pct` rank mapped onto [-1, 1], as compute_existing_composite does."""

    values = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})

    normalized = rank_normalize(values)

    assert normalized.min() >= -1.0 and normalized.max() <= 1.0
    assert normalized.is_monotonic_increasing
    assert normalized["a"] < normalized["d"]


def test_rank_normalize_keeps_nan_as_nan():
    values = pd.Series({"a": 1.0, "b": np.nan, "c": 3.0})

    assert bool(pd.isna(rank_normalize(values)["b"]))


def test_cs_rank_pct_matches_the_mining_blend_convention():
    """`_cs_rank` is a bare pct rank in (0, 1], not the [-1, 1] mapping.

    The blend and the baseline composite deliberately use different rank
    conventions; conflating them would silently shift every blended value.
    """

    values = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})

    ranked = cs_rank_pct(values)

    assert ranked.min() > 0.0 and ranked.max() == pytest.approx(1.0)
    assert ranked["a"] == pytest.approx(0.25)
    assert ranked["d"] == pytest.approx(1.0)


# --- baseline composite -----------------------------------------------------


def test_single_factor_composite_is_that_factor_rank_normalized():
    """Our pinned case: one factor, so the weight divides straight back out."""

    baseline = validate_residual_baseline(_spec())
    raw = pd.Series({f"s{i}": float(i) for i in range(10)})

    composite = build_baseline_composite(baseline, {"pv_low_dollar_volume_5d": raw})

    assert np.allclose(composite.to_numpy(), rank_normalize(raw).to_numpy())


def test_composite_weights_and_normalizes_by_total_abs_weight():
    factors = [
        {"name": "a", "implementation": "price_volume:a", "weight": 0.3, "direction": 1.0},
        {"name": "b", "implementation": "price_volume:b", "weight": 0.1, "direction": 1.0},
    ]
    baseline = validate_residual_baseline(_spec(factors=factors))
    up = pd.Series({f"s{i}": float(i) for i in range(10)})
    down = pd.Series({f"s{i}": float(-i) for i in range(10)})

    composite = build_baseline_composite(baseline, {"a": up, "b": down})
    expected = (
        rank_normalize(up).mul(0.3) + rank_normalize(down).mul(0.1)
    ).div(0.4)

    assert np.allclose(composite.to_numpy(), expected.to_numpy())


def test_composite_applies_negative_direction():
    factors = [{**_MINING_TIME_BASELINE[0], "direction": -1.0}]
    baseline = validate_residual_baseline(_spec(factors=factors))
    raw = pd.Series({f"s{i}": float(i) for i in range(10)})

    composite = build_baseline_composite(baseline, {"pv_low_dollar_volume_5d": raw})

    assert np.allclose(composite.to_numpy(), -rank_normalize(raw).to_numpy())


def test_composite_is_clipped_to_the_unit_interval():
    baseline = validate_residual_baseline(_spec())
    raw = pd.Series({f"s{i}": float(i) for i in range(50)})

    composite = build_baseline_composite(baseline, {"pv_low_dollar_volume_5d": raw})

    assert composite.min() >= -1.0 and composite.max() <= 1.0


def test_composite_treats_missing_ranks_as_neutral_like_the_mining_path():
    """`compute_existing_composite` does `.fillna(0.0)` before weighting."""

    baseline = validate_residual_baseline(_spec())
    raw = pd.Series({"a": 1.0, "b": np.nan, "c": 3.0})

    composite = build_baseline_composite(baseline, {"pv_low_dollar_volume_5d": raw})

    assert float(composite["b"]) == pytest.approx(0.0)


def test_composite_fails_closed_when_a_baseline_factor_is_missing():
    baseline = validate_residual_baseline(_spec())

    with pytest.raises(ResidualBaselineError, match="pv_low_dollar_volume_5d"):
        build_baseline_composite(baseline, {})


# --- cross-sectional residual ----------------------------------------------


def test_residual_removes_the_baseline_component():
    baseline = pd.Series(np.linspace(-1.0, 1.0, 60))
    signal = baseline.mul(3.0).add(5.0)  # perfectly explained by the baseline

    residual = cross_sectional_residual(signal, baseline)

    assert np.allclose(residual.dropna().to_numpy(), 0.0, atol=1e-9)


def test_residual_keeps_the_orthogonal_component():
    rng = np.random.default_rng(0)
    baseline = pd.Series(np.linspace(-1.0, 1.0, 60))
    noise = pd.Series(rng.standard_normal(60))
    signal = baseline.mul(2.0).add(noise)

    residual = cross_sectional_residual(signal, baseline)

    assert float(residual.corr(baseline)) == pytest.approx(0.0, abs=1e-9)
    assert float(residual.std()) > 0.1


def test_residual_is_nan_when_the_cross_section_is_too_thin():
    """Matches the mining residualizer's 20-observation floor."""

    baseline = pd.Series(np.linspace(-1.0, 1.0, 19))
    signal = pd.Series(np.linspace(0.0, 1.0, 19))

    assert cross_sectional_residual(signal, baseline).isna().all()


def test_residual_falls_back_to_demeaning_when_the_baseline_is_constant():
    baseline = pd.Series([0.5] * 40)
    signal = pd.Series(np.linspace(0.0, 1.0, 40))

    residual = cross_sectional_residual(signal, baseline)

    assert float(residual.mean()) == pytest.approx(0.0, abs=1e-9)


def test_residual_only_uses_symbols_present_in_both():
    baseline = pd.Series({f"s{i}": float(i) for i in range(40)})
    signal = pd.Series({f"s{i}": float(i) * 2.0 for i in range(30)})

    residual = cross_sectional_residual(signal, baseline)

    assert set(residual.index) == set(signal.index)
    assert residual.notna().sum() == 30


def test_residual_propagates_nan_inputs_without_dropping_symbols():
    baseline = pd.Series({f"s{i}": float(i) for i in range(40)})
    signal = pd.Series({f"s{i}": float(i) for i in range(40)})
    signal["s5"] = np.nan

    residual = cross_sectional_residual(signal, baseline)

    assert bool(pd.isna(residual["s5"]))
    assert set(residual.index) == set(signal.index)
