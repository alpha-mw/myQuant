"""Pin the production runtime's implementation allowlist.

The allowlist previously admitted only `price_volume:`, which capped the
production-executable universe at 3 of the registry's 8 production-level
factors — below the v4 minimum of 5, and spanning too few families to satisfy
the 20%/35% weight caps. `aquant_expression:` was already implemented in the
dispatcher and excluded from production alone, so admitting it is the cheapest
step toward a viable set.

The prefixes now live in one constant so the evaluation loop, the dispatcher,
and the mining pipeline's promotion gate cannot drift apart.
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
    PRODUCTION_IMPLEMENTATION_PREFIXES,
    PRODUCTION_RUNTIME_MODE,
    REPORT_ONLY_SHADOW_RUNTIME_MODE,
    MinedFactorScorer,
    is_production_allowlisted_implementation,
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


def _record(implementation: str) -> FactorRecord:
    return FactorRecord(
        name="probe_factor",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation=implementation,
        weight=1.0,
        gate_results=[
            GateResult(gate_id=index, gate_key=f"gate{index}", title=f"Gate {index}", passed=True)
            for index in range(1, 9)
        ],
        metadata={"expression": "cs_rank(fin_net_profit_yoy)"},
    )


@pytest.mark.parametrize(
    "implementation",
    ["price_volume:pv_low_dollar_volume_5d", "aquant_expression:fund_fin_net_profit_yoy"],
)
def test_allowlisted_prefixes(implementation):
    assert is_production_allowlisted_implementation(implementation) is True


@pytest.mark.parametrize(
    "implementation",
    [
        "research_formula:rank_blend",  # not in the dispatcher at all
        "builtin:something",
        "alpha158.FactorEngineer.cross_sectional_score",
        "alpha_mining.FactorLibrary:whatever",
        "",
        None,
    ],
)
def test_non_allowlisted_prefixes(implementation):
    assert is_production_allowlisted_implementation(implementation) is False


def test_allowlist_is_a_prefix_match_not_a_substring_match():
    assert is_production_allowlisted_implementation("evil_price_volume:x") is False


def test_constant_is_the_single_source_of_truth():
    assert "price_volume:" in PRODUCTION_IMPLEMENTATION_PREFIXES
    assert "aquant_expression:" in PRODUCTION_IMPLEMENTATION_PREFIXES
    assert "research_formula:" not in PRODUCTION_IMPLEMENTATION_PREFIXES


def test_aquant_expression_is_no_longer_refused_by_production_mode():
    """It used to raise "not allowlisted" before reaching its implementation."""

    scorer = MinedFactorScorer(runtime_mode=PRODUCTION_RUNTIME_MODE)

    try:
        scorer._compute_factor(_record("aquant_expression:probe_factor"), _frames())
    except ValueError as exc:
        assert "not allowlisted" not in str(exc)
    except Exception:
        pass  # any downstream data error is fine; the gate is what we assert


def test_research_formula_still_fails_closed_in_production():
    scorer = MinedFactorScorer(runtime_mode=PRODUCTION_RUNTIME_MODE)

    with pytest.raises(ValueError, match="not allowlisted"):
        scorer._compute_factor(_record("research_formula:rank_blend"), _frames())


def test_builtin_still_fails_closed_in_production():
    scorer = MinedFactorScorer(runtime_mode=PRODUCTION_RUNTIME_MODE)

    with pytest.raises(ValueError, match="not allowlisted"):
        scorer._compute_factor(_record("builtin:whatever"), _frames())


def test_shadow_mode_is_unchanged_and_still_broader_than_production():
    """Report-only shadow keeps evaluating implementations production refuses."""

    scorer = MinedFactorScorer(runtime_mode=REPORT_ONLY_SHADOW_RUNTIME_MODE)

    try:
        scorer._compute_factor(_record("builtin:whatever"), _frames())
    except ValueError as exc:
        assert "not allowlisted" not in str(exc)
    except Exception:
        pass
