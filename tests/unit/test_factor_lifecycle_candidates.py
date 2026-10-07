import json

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import lifecycle_candidates as lc
from quant_investor.factors.governance import implementations
from quant_investor.factors.governance.bootstrap import BLEND_W80, LOW_DOLLAR_VOLUME


def _frames(closes: dict[str, list[float]]) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-05", periods=len(next(iter(closes.values()))))
    return {
        symbol: pd.DataFrame({"trade_date": dates.strftime("%Y%m%d"), "adj_close": values})
        for symbol, values in closes.items()
    }


def test_definitions_are_canonical_unique_and_disjoint_from_bootstrap():
    rows = lc.candidate_definitions()
    ids = [row["factor_id"] for row in rows]
    assert ids == sorted(ids, key=lambda value: value.encode("utf-8"))
    assert len(ids) == len(set(ids)) == 6
    assert not set(ids) & {LOW_DOLLAR_VOLUME, BLEND_W80}
    for row in rows:
        expression = row["normalized_expression"]
        assert json.dumps(json.loads(expression), sort_keys=True, separators=(",", ":")) == (
            expression
        )
        assert row["authority"] == lc.CANDIDATE_AUTHORITY
        assert row["hypothesis"]


def test_bootstrap_and_prospective_registries_are_disjoint():
    assert implementations.BOOTSTRAP_FACTOR_IDS == {LOW_DOLLAR_VOLUME, BLEND_W80}
    assert implementations.PROSPECTIVE_FACTOR_IDS == set(lc.PRICE_VOLUME_CANDIDATES)
    assert not (implementations.BOOTSTRAP_FACTOR_IDS & implementations.PROSPECTIVE_FACTOR_IDS)
    assert implementations.INSTALLED_FACTOR_IDS == (
        implementations.BOOTSTRAP_FACTOR_IDS | implementations.PROSPECTIVE_FACTOR_IDS
    )
    assert set(implementations._BOOTSTRAP_ENTRYPOINTS) == {LOW_DOLLAR_VOLUME, BLEND_W80}
    assert lc.OCF_TO_PROFIT not in implementations.PROSPECTIVE_FACTOR_IDS


def test_trial_accounting_counts_families_and_nominal_trials():
    accounting = lc.batch_trial_accounting()
    assert accounting["nominal_trial_count"] == 6
    assert accounting["family_count"] == 4
    assert accounting["families"]["LOW_VOLATILITY"] == sorted(
        [lc.VOLATILITY_PENALTY_5D, lc.VOLATILITY_PENALTY_10D, lc.DOWNSIDE_VOLATILITY_20D]
    )


def test_price_candidates_match_hand_computed_values():
    rng = np.random.default_rng(7)
    closes = {
        "000001.SZ": list(10.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, 30))),
        "600000.SH": list(20.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 30))),
    }
    signals = lc.compute_price_volume_candidates(
        _frames(closes), source_format=lc.CANONICAL_PARQUET
    )
    for symbol, values in closes.items():
        close = pd.Series(values)
        returns = close.pct_change()
        assert signals[lc.VOLATILITY_PENALTY_5D][symbol] == pytest.approx(
            -returns.tail(5).std(ddof=1)
        )
        assert signals[lc.VOLATILITY_PENALTY_10D][symbol] == pytest.approx(
            -returns.tail(10).std(ddof=1)
        )
        assert signals[lc.DOWNSIDE_VOLATILITY_20D][symbol] == pytest.approx(
            -returns.tail(20).clip(upper=0.0).std(ddof=1)
        )
        assert signals[lc.SHORT_REVERSAL_5D][symbol] == pytest.approx(
            -(close.iloc[-1] / close.iloc[-6] - 1.0)
        )
        assert signals[lc.MAX_RETURN_20D][symbol] == pytest.approx(-returns.tail(20).max())


def test_candidates_are_deterministic_and_order_independent():
    values = [10, 10.5, 10.2, 10.8, 11.0, 10.9, 11.3]
    frames = _frames({"B.SZ": values, "A.SH": values[::-1]})
    reversed_frames = dict(reversed(list(frames.items())))
    first = lc.compute_price_volume_candidates(
        frames, source_format=lc.CANONICAL_PARQUET, factor_ids=[lc.VOLATILITY_PENALTY_5D]
    )
    second = lc.compute_price_volume_candidates(
        reversed_frames,
        source_format=lc.CANONICAL_PARQUET,
        factor_ids=[lc.VOLATILITY_PENALTY_5D],
    )
    pd.testing.assert_series_equal(
        first[lc.VOLATILITY_PENALTY_5D], second[lc.VOLATILITY_PENALTY_5D]
    )


def test_incomplete_window_yields_nan_not_a_shorter_window():
    values = [10.0, 10.1, np.nan, 10.3, 10.2, 10.4, 10.5]
    signals = lc.compute_price_volume_candidates(
        _frames({"X.SZ": values}),
        source_format=lc.CANONICAL_PARQUET,
        factor_ids=[lc.VOLATILITY_PENALTY_5D],
    )
    assert np.isnan(signals[lc.VOLATILITY_PENALTY_5D]["X.SZ"])


@pytest.mark.parametrize(
    "frames, kwargs, message",
    [
        ({}, {}, "in-memory"),
        (
            {"X.SZ": pd.DataFrame({"trade_date": ["20260105"], "close": [1.0]})},
            {},
            "missing",
        ),
        (
            {"X.SZ": pd.DataFrame({"trade_date": ["20260105"] * 2, "adj_close": [1.0, 1.1]})},
            {},
            "duplicated",
        ),
    ],
)
def test_strict_inputs_fail_closed(frames, kwargs, message):
    with pytest.raises(lc.LifecycleCandidateError, match=message):
        lc.compute_price_volume_candidates(frames, source_format=lc.CANONICAL_PARQUET, **kwargs)


def test_non_canonical_source_and_unknown_candidates_are_rejected():
    frames = _frames({"X.SZ": [1.0, 1.1, 1.2]})
    with pytest.raises(lc.LifecycleCandidateError, match="canonical Parquet"):
        lc.compute_price_volume_candidates(frames, source_format="csv")
    with pytest.raises(lc.LifecycleCandidateError, match="not a price-volume"):
        lc.compute_price_volume_candidates(
            frames, source_format=lc.CANONICAL_PARQUET, factor_ids=[lc.OCF_TO_PROFIT]
        )
    with pytest.raises(lc.LifecycleCandidateError, match="unknown"):
        lc.candidate_panel("pv_unknown", pd.DataFrame({"X": [1.0]}))


def _pit_rows(**overrides):
    rows = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "600000.SH"],
            "value": [0.8, 1.2, 0.5],
            "available_date": ["2026-04-20", "2026-08-25", "2026-08-30"],
            "source_classification": ["PIT", "PIT", "PIT"],
        }
    )
    for column, values in overrides.items():
        rows[column] = values
    return rows


def test_pit_candidate_takes_latest_available_value():
    values = lc.compute_pit_fundamental_candidate(_pit_rows(), as_of="20260831")
    assert values.to_dict() == {"000001.SZ": 1.2, "600000.SH": 0.5}


def test_pit_candidate_rejects_look_ahead_rows():
    with pytest.raises(lc.LifecycleCandidateError, match="after as_of"):
        lc.compute_pit_fundamental_candidate(_pit_rows(), as_of="20260826")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"source_classification": ["PIT", "RECONSTRUCTED", "PIT"]}, "not PIT"),
        ({"available_date": ["2026-04-20", None, "2026-08-30"]}, "available_date"),
    ],
)
def test_pit_candidate_rejects_non_pit_sources(overrides, message):
    with pytest.raises(lc.LifecycleCandidateError, match=message):
        lc.compute_pit_fundamental_candidate(_pit_rows(**overrides), as_of="20260831")


def test_pit_candidate_requires_all_pit_columns():
    rows = _pit_rows().drop(columns=["source_classification"])
    with pytest.raises(lc.LifecycleCandidateError, match="missing"):
        lc.compute_pit_fundamental_candidate(rows, as_of="20260831")
