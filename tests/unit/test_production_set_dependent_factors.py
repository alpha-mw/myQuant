"""Refuse to run factors whose definition depends on the production set.

`scripts/mine_quant_branch_factors.py` derives a `<primitive>_resid_existing`
variant of every formulaic primitive by regressing it on
`context.existing_composite`, and `compute_existing_composite` builds that
composite from `registry.selectable_factors()` — the *currently weighted
production factors*. A factor whose params reference such a primitive is
therefore defined in terms of the production set it is about to join.

Three things break as a result, and none of them announce themselves:

1. Promoting any factor silently rewrites the definition of every
   `_resid_existing` factor, so the 8-gate evidence on record describes a
   baseline that no longer exists.
2. `compute_existing_composite` returns `None` as soon as a non-`price_volume:`
   factor becomes selectable, and `_residualize_against_existing` maps a `None`
   baseline to `signal * nan`. The factor evaluates to all-NaN rather than
   erroring.
3. v4 requires a stable `runtime_contract_sha256` and a canonical replay. A
   value that depends on mutable registry state cannot be replayed from market
   data alone.

The implementation allowlist does not catch this: it asks whether the runtime
*can* execute an implementation, not whether the result is reproducible. So the
guard is separate, and it fails closed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant_investor.factors.governance import (
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import (
    PRODUCTION_RUNTIME_MODE,
    REPORT_ONLY_SHADOW_RUNTIME_MODE,
    MinedFactorScorer,
    is_production_allowlisted_implementation,
    production_set_dependent_primitives,
)


def _frames() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    return {
        "000001.SZ": pd.DataFrame(
            {
                "symbol": ["000001.SZ"] * len(dates),
                "trade_date": dates,
                "close": [10, 11, 12, 13, 14, 15, 16, 17],
                "adj_close": [10, 11, 12, 13, 14, 15, 16, 17],
                "volume": [100] * len(dates),
                "amount": [102, 113, 126, 139, 151, 160, 170, 180],
            }
        ),
    }


def _record(implementation: str, params: dict | None = None) -> FactorRecord:
    return FactorRecord(
        name="probe_factor",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation=implementation,
        weight=1.0,
        gate_results=[
            GateResult(
                gate_id=index,
                gate_key=f"gate{index}",
                title=f"Gate {index}",
                passed=True,
            )
            for index in range(1, 9)
        ],
        metadata={"params": params or {}},
    )


# --- the predicate ----------------------------------------------------------


def test_detects_the_registry_factor_that_actually_has_this_shape():
    """`formula_mom120_np_yoy_resid_w25`, verbatim from the registry."""

    record = _record(
        "research_formula:rank_blend",
        {
            "left": "momentum_120",
            "right": "fin_net_profit_yoy_resid_existing",
            "left_weight": 0.25,
        },
    )

    assert production_set_dependent_primitives(record) == (
        "fin_net_profit_yoy_resid_existing",
    )


def test_reports_every_offending_primitive_not_just_the_first():
    record = _record(
        "research_formula:rank_blend",
        {
            "left": "momentum_90_resid_existing",
            "right": "fin_ocf_to_profit_resid_existing",
            "left_weight": 0.5,
        },
    )

    assert production_set_dependent_primitives(record) == (
        "fin_ocf_to_profit_resid_existing",
        "momentum_90_resid_existing",
    )


def test_self_contained_factors_are_clean():
    record = _record(
        "research_formula:rank_blend",
        {"left": "momentum_120", "right": "fin_net_profit_yoy", "left_weight": 0.25},
    )

    assert production_set_dependent_primitives(record) == ()


@pytest.mark.parametrize(
    "implementation",
    [
        "price_volume:pv_low_dollar_volume_5d",
        "aquant_expression:fund_fin_net_profit_yoy",
    ],
)
def test_the_currently_promotable_factors_are_clean(implementation):
    assert production_set_dependent_primitives(_record(implementation)) == ()


def test_missing_or_malformed_params_are_not_treated_as_offending():
    assert production_set_dependent_primitives(_record("price_volume:x", {})) == ()
    assert (
        production_set_dependent_primitives(
            _record("research_formula:rank_blend", {"left": None, "left_weight": 0.5})
        )
        == ()
    )


# --- the gate ---------------------------------------------------------------


def test_production_refuses_a_production_set_dependent_factor():
    scorer = MinedFactorScorer(runtime_mode=PRODUCTION_RUNTIME_MODE)
    record = _record(
        "price_volume:pv_low_dollar_volume_5d",  # allowlisted prefix on purpose
        {"left": "momentum_120", "right": "fin_net_profit_yoy_resid_existing"},
    )

    assert is_production_allowlisted_implementation(record.implementation) is True
    with pytest.raises(ValueError, match="depends on the production set"):
        scorer._compute_factor(record, _frames())


def test_the_guard_is_independent_of_the_allowlist():
    """An allowlisted prefix must not launder a self-referential definition."""

    scorer = MinedFactorScorer(runtime_mode=PRODUCTION_RUNTIME_MODE)
    record = _record(
        "aquant_expression:whatever",
        {"right": "fin_net_profit_yoy_resid_existing"},
    )

    with pytest.raises(ValueError, match="depends on the production set"):
        scorer._compute_factor(record, _frames())


def test_shadow_mode_still_evaluates_them():
    """Research must keep mining and scoring these; only production refuses."""

    scorer = MinedFactorScorer(runtime_mode=REPORT_ONLY_SHADOW_RUNTIME_MODE)
    record = _record(
        "price_volume:pv_low_dollar_volume_5d",
        {"right": "fin_net_profit_yoy_resid_existing"},
    )

    try:
        scorer._compute_factor(record, _frames())
    except ValueError as exc:
        assert "depends on the production set" not in str(exc)
    except Exception:
        pass


def test_a_pinned_baseline_is_not_refused():
    """The escape hatch: `residual_baseline` freezes the mutable input.

    Same arithmetic, but the baseline is named and content-hashed instead of
    read from `registry.selectable_factors()`, so the value is reproducible
    from market data plus the spec. Only the unpinned `_resid_existing`
    reference is a blocker.
    """

    from quant_investor.factors.residual_baseline import (
        RESIDUAL_BASELINE_SCHEMA_VERSION,
        baseline_sha256,
    )

    factors = [
        {
            "name": "pv_low_dollar_volume_5d",
            "implementation": "price_volume:pv_low_dollar_volume_5d",
            "weight": 0.05,
            "direction": 1.0,
        }
    ]
    record = _record(
        "price_volume:pv_low_dollar_volume_5d",
        {
            "left": "momentum_120",
            "right": "fin_net_profit_yoy",
            "left_weight": 0.25,
            "residualize_right_against": {
                "schema_version": RESIDUAL_BASELINE_SCHEMA_VERSION,
                "factors": factors,
                "baseline_sha256": baseline_sha256(factors),
            },
        },
    )

    assert production_set_dependent_primitives(record) == ()


def test_clean_factors_still_pass_the_gate():
    scorer = MinedFactorScorer(runtime_mode=PRODUCTION_RUNTIME_MODE)

    try:
        scorer._compute_factor(_record("price_volume:pv_low_dollar_volume_5d"), _frames())
    except ValueError as exc:
        assert "depends on the production set" not in str(exc)
    except Exception:
        pass
