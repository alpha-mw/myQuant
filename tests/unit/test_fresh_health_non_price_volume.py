"""Let fresh health cover the factors v4 actually needs in production.

`assess_factor_record_v4` blocks any record whose `health` is not
`{status: healthy, fresh: True}`, and the only producer of fresh health is
`scripts/factor_health_automation`. That producer went through
`_mining_candidate_from_record`, which raised
"fresh evaluation supports price_volume factors only" for anything else.

The target v4 set needs five distinct families, and at five factors the 20%
per-factor and 35% per-family caps force exactly one factor per family. Two of
those five are not price_volume: the growth family is carried by
`aquant_expression:fund_fin_net_profit_yoy` and `formulaic_research` by
`research_formula:rank_blend`. Neither could ever obtain fresh health, so
`factor_governance_ready` was unreachable regardless of statistics.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.governance import (
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)

from scripts import factor_health_automation
from scripts.mine_quant_branch_factors import MiningCandidate

_BASELINE = {
    "schema_version": "factor-residual-baseline.v1",
    "factors": [
        {
            "name": "pv_low_dollar_volume_5d",
            "implementation": "price_volume:pv_low_dollar_volume_5d",
            "weight": 0.05,
            "direction": 1.0,
        }
    ],
    "baseline_sha256": "22b3bb6c7b10bccc0018a47000fee2854e5028957969d5faa002ca8f35b5dbe5",
}


def _record(name: str, implementation: str, **metadata) -> FactorRecord:
    return FactorRecord(
        name=name,
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        category="probe",
        implementation=implementation,
        weight=0.2,
        gate_results=[
            GateResult(gate_id=i, gate_key=f"gate{i}", title=f"Gate {i}", passed=True)
            for i in range(1, 9)
        ],
        metadata=dict(metadata),
    )


def _build(record: FactorRecord) -> MiningCandidate:
    return factor_health_automation._mining_candidate_from_record(record, MiningCandidate)


# --- price_volume is unchanged ---------------------------------------------


def test_price_volume_records_still_build_the_same_candidate():
    candidate = _build(_record("pv_short_reversal_5d", "price_volume:pv_short_reversal_5d"))

    assert candidate.family == "short_reversal"
    assert candidate.window == 5
    assert candidate.implementation == "price_volume:pv_short_reversal_5d"


def test_the_blend_record_still_carries_its_params():
    candidate = _build(
        _record(
            "pv_blend_volstab19x2_mom90_amihud5_w80",
            "price_volume:pv_blend_volstab19x2_mom90_amihud5_w80",
        )
    )

    assert candidate.family == "volstab_momentum_illiquidity_blend"
    assert candidate.params["outer_volume_stability_weight"] == pytest.approx(0.80)


# --- the growth family ------------------------------------------------------


def test_aquant_expression_records_now_build():
    """`fund_fin_net_profit_yoy` carries the growth family in the v4 set."""

    candidate = _build(
        _record(
            "fund_fin_net_profit_yoy",
            "aquant_expression:fund_fin_net_profit_yoy",
            expression="cs_rank(fin_net_profit_yoy)",
        )
    )

    assert candidate.implementation == "aquant_expression:fund_fin_net_profit_yoy"
    assert candidate.expression == "cs_rank(fin_net_profit_yoy)"


def test_an_aquant_record_without_an_expression_fails_closed():
    """An empty expression would evaluate to nothing rather than erroring."""

    with pytest.raises(ValueError, match="expression"):
        _build(_record("fund_x", "aquant_expression:fund_x"))


# --- the formulaic family ---------------------------------------------------


def test_pinned_research_formula_records_now_build():
    candidate = _build(
        _record(
            "formula_mom120_np_yoy_resid_w25",
            "research_formula:rank_blend",
            params={
                "left": "momentum_120",
                "right": "fin_net_profit_yoy",
                "left_weight": 0.25,
                "residualize_right_against": _BASELINE,
            },
        )
    )

    assert candidate.implementation == "research_formula:rank_blend"
    assert candidate.params["left"] == "momentum_120"
    assert candidate.params["residualize_right_against"]["baseline_sha256"].startswith(
        "22b3bb6c"
    )


def test_an_unpinned_research_formula_record_fails_closed():
    """Health must not measure a factor production would refuse to run."""

    record = _record(
        "formula_mom120_np_yoy_resid_w25",
        "research_formula:rank_blend",
        params={
            "left": "momentum_120",
            "right": "fin_net_profit_yoy_resid_existing",
            "left_weight": 0.25,
        },
    )

    with pytest.raises(ValueError, match="production set"):
        _build(record)


def test_a_research_formula_record_without_params_fails_closed():
    with pytest.raises(ValueError, match="params"):
        _build(_record("formula_x", "research_formula:rank_blend"))


def test_an_unknown_research_formula_variant_fails_closed():
    with pytest.raises(ValueError, match="research_formula"):
        _build(
            _record(
                "formula_x",
                "research_formula:something_else",
                params={"left": "momentum_120", "right": "fin_roe", "left_weight": 0.5},
            )
        )


# --- everything else still fails closed ------------------------------------


@pytest.mark.parametrize(
    "implementation",
    ["builtin:whatever", "alpha_mining.FactorLibrary:x", "", "nonsense"],
)
def test_unsupported_implementations_still_fail_closed(implementation):
    with pytest.raises(ValueError):
        _build(_record("probe", implementation))


# --- signal dispatch --------------------------------------------------------


def _panel_context():
    dates = pd.date_range("2024-01-01", periods=180, freq="B")
    symbols = [f"{i:06d}.SZ" for i in range(30)]
    rng = np.random.default_rng(3)
    adj_close = pd.DataFrame(
        20.0 * np.exp(np.cumsum(rng.standard_normal((len(dates), len(symbols))) * 0.01, axis=0)),
        index=dates,
        columns=symbols,
    )
    volume = pd.DataFrame(
        rng.lognormal(12.0, 0.3, (len(dates), len(symbols))), index=dates, columns=symbols
    )
    return SimpleNamespace(adj_close=adj_close, volume=volume, amount=adj_close * volume)


def _dispatch(candidate, context, *, expression_inputs=None, builder=None):
    return factor_health_automation._fresh_candidate_signal(
        candidate,
        context,
        expression_inputs=expression_inputs,
        candidate_type=MiningCandidate,
        price_volume_signal_builder=builder or (lambda c, ctx: ctx.adj_close * 0.0 + 1.0),
    )


def test_dispatch_routes_price_volume_to_the_price_volume_builder():
    context = _panel_context()
    seen = []

    def builder(candidate, ctx):
        seen.append(candidate.implementation)
        return ctx.adj_close * 0.0

    _dispatch(_build(_record("pv_short_reversal_5d", "price_volume:pv_short_reversal_5d")),
              context, builder=builder)

    assert seen == ["price_volume:pv_short_reversal_5d"]


def test_dispatch_computes_a_pinned_formulaic_signal_end_to_end():
    """momentum_120 blended with momentum_60 residualized against the pin."""

    context = _panel_context()
    candidate = _build(
        _record(
            "formula_probe",
            "research_formula:rank_blend",
            params={
                "left": "momentum_120",
                "right": "momentum_60",
                "left_weight": 0.25,
                "residualize_right_against": _BASELINE,
            },
        )
    )

    signal = _dispatch(candidate, context)

    assert signal.shape == context.adj_close.shape
    assert signal.notna().to_numpy().sum() > 0


def test_dispatch_fails_closed_when_aquant_inputs_are_unavailable():
    candidate = _build(
        _record("fund_x", "aquant_expression:fund_x", expression="cs_rank(fin_roe)")
    )

    with pytest.raises(ValueError, match="A_quant inputs"):
        _dispatch(candidate, _panel_context(), expression_inputs=None)


def test_dispatch_refuses_an_implementation_it_cannot_compute():
    candidate = MiningCandidate(
        name="x", family="", category="", implementation="builtin:x", description=""
    )

    with pytest.raises(ValueError, match="cannot compute implementation"):
        _dispatch(candidate, _panel_context())
