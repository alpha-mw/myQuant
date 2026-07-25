from __future__ import annotations

import numpy as np
import pandas as pd

from quant_investor.v17.fundamental_scoring import score_fundamental_universe

CUTOFF = pd.Timestamp("2026-06-30T15:00:00+08:00")


def _snapshot(count: int = 30) -> pd.DataFrame:
    values = np.arange(count, dtype=float)
    return pd.DataFrame(
        {
            "symbol": [f"{index:06d}.SZ" for index in range(count)],
            "industry": ["synthetic-industry"] * count,
            "in_universe": [True] * count,
            "research_eligible": [True] * count,
            "membership_conflict": [False] * count,
            "membership_is_pit": [True] * count,
            "universe_id": ["CN/full_a"] * count,
            "availability": [CUTOFF - pd.Timedelta(days=1)] * count,
            "flow_basis": ["LATEST_TTM"] * count,
            "balance_sheet_basis": ["LATEST_REPORT_PERIOD"] * count,
            "capex_sign_convention": ["POSITIVE_OUTFLOW"] * count,
            "net_profit_ttm": 100.0 + values,
            "market_cap": 1000.0 + values,
            "cfo_ttm": 120.0 + values,
            "capex_ttm": 20.0 + values / 10.0,
            "fin_roe": 0.05 + values / 1000.0,
            "fin_ocf_to_profit": 0.7 + values / 100.0,
            "fin_net_profit_yoy": -0.1 + values / 100.0,
            "fin_debt_to_assets": 0.6 - values / 100.0,
        }
    )


def _history(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range(end=CUTOFF - pd.offsets.BDay(1), periods=252, freq="B")
    for _, item in snapshot.iterrows():
        fcf_yield = (item["cfo_ttm"] - item["capex_ttm"]) / item["market_cap"]
        metrics = {
            "fin_roe": item["fin_roe"],
            "fin_ocf_to_profit": item["fin_ocf_to_profit"],
            "fin_fcf_to_profit": (item["cfo_ttm"] - item["capex_ttm"]) / item["net_profit_ttm"],
            "fin_net_profit_yoy": item["fin_net_profit_yoy"],
            "fin_debt_to_assets": item["fin_debt_to_assets"],
            "fcf_to_price": fcf_yield,
        }
        for metric, current in metrics.items():
            for offset, date in enumerate(dates):
                rows.append(
                    {
                        "symbol": item["symbol"],
                        "trade_date": date,
                        "availability": date,
                        "is_open_day": True,
                        "metric": metric,
                        "value": float(current) + (offset - 126) / 10_000.0,
                    }
                )
    return pd.DataFrame(rows)


def test_scores_top24_and_appends_unavailable_holding_without_backfill() -> None:
    snapshot = _snapshot()
    snapshot.loc[29, "fin_roe"] = np.nan
    result = score_fundamental_universe(
        snapshot,
        _history(snapshot),
        cutoff=CUTOFF,
        holdings=("000029.SZ",),
    )

    assert len(result.ranked_symbols) == 24
    assert "000029.SZ" not in result.ranked_symbols
    assert result.appended_holdings == ("000029.SZ",)
    assert result.sealed_symbols[-1] == "000029.SZ"
    unavailable = result.scored.set_index("symbol").loc["000029.SZ"]
    assert unavailable["status"] == "UNAVAILABLE"
    assert "missing_main_metric:fin_roe" in unavailable["unavailable_reasons"]
    first = result.scored.set_index("symbol").loc["000000.SZ"]
    assert first["fcf_to_price"] == (120.0 - 20.0) / 1000.0
    # Profitability has no optional ROA evidence, so it uses ROE at 100%
    # instead of failing or zero-filling.
    assert np.isfinite(first["pillar_profitability"])


def test_small_industry_never_falls_back_to_market_rank() -> None:
    snapshot = _snapshot(19)
    empty_history = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "availability",
            "is_open_day",
            "metric",
            "value",
        ]
    )
    result = score_fundamental_universe(snapshot, empty_history, cutoff=CUTOFF)

    assert result.ranked_symbols == ()
    assert set(result.scored["status"]) == {"UNAVAILABLE"}
    assert all(
        "industry_sample_below_20" in reasons for reasons in result.scored["unavailable_reasons"]
    )


def test_membership_conflict_is_unavailable() -> None:
    snapshot = _snapshot()
    snapshot.loc[0, "membership_conflict"] = True
    result = score_fundamental_universe(snapshot, _history(snapshot), cutoff=CUTOFF)
    row = result.scored.set_index("symbol").loc["000000.SZ"]
    assert row["status"] == "UNAVAILABLE"
    assert "pit_membership_conflict" in row["unavailable_reasons"]


def test_final_industry_sample_is_rechecked_after_history_failures() -> None:
    snapshot = _snapshot(25)
    history = _history(snapshot)
    missing_symbols = {f"{index:06d}.SZ" for index in range(19, 25)}
    history = history.loc[
        ~(history["symbol"].isin(missing_symbols) & history["metric"].eq("fcf_to_price"))
    ]
    result = score_fundamental_universe(snapshot, history, cutoff=CUTOFF)
    assert result.ranked_symbols == ()
    survivors = result.scored.loc[~result.scored["symbol"].isin(missing_symbols)]
    assert all(
        "final_industry_available_sample_below_20" in reasons
        for reasons in survivors["unavailable_reasons"]
    )


def test_main_metric_readiness_is_preflighted_before_any_ranking() -> None:
    snapshot = _snapshot(30)
    history = _history(snapshot)
    # Different rows fail different main histories.  The final peer set is 20
    # and every metric must rank against that same set, independent of the
    # order MAIN_METRICS happens to be declared in.
    roe_missing = {f"{index:06d}.SZ" for index in range(20, 25)}
    cash_missing = {f"{index:06d}.SZ" for index in range(25, 30)}
    history = history.loc[
        ~(
            (history["symbol"].isin(roe_missing) & history["metric"].eq("fin_roe"))
            | (history["symbol"].isin(cash_missing) & history["metric"].eq("fin_ocf_to_profit"))
        )
    ]
    result = score_fundamental_universe(snapshot, history, cutoff=CUTOFF)
    available = result.scored.loc[result.scored["status"].eq("AVAILABLE")]
    assert len(available) == 20
    for metric in (
        "fin_roe",
        "fin_ocf_to_profit",
        "fin_net_profit_yoy",
        "fin_debt_to_assets",
        "fcf_to_price",
    ):
        assert available[f"{metric}_score"].notna().all()


def test_optional_fcf_to_profit_uses_cfo_minus_capex_definition() -> None:
    snapshot = _snapshot()
    snapshot["fin_fcf_to_profit"] = 999.0
    result = score_fundamental_universe(snapshot, _history(snapshot), cutoff=CUTOFF)
    row = result.scored.set_index("symbol").loc["000000.SZ"]
    assert row["fin_fcf_to_profit"] == (120.0 - 20.0) / 100.0


def test_provided_optional_metric_with_short_self_history_is_unavailable() -> None:
    snapshot = _snapshot()
    snapshot["fin_roa"] = 0.02 + np.arange(len(snapshot), dtype=float) / 10_000.0
    history = _history(snapshot)
    dates = pd.date_range(end=CUTOFF - pd.offsets.BDay(1), periods=252, freq="B")
    optional_rows: list[dict[str, object]] = []
    for _, item in snapshot.iterrows():
        symbol_dates = dates[1:] if item["symbol"] == "000000.SZ" else dates
        for offset, date in enumerate(symbol_dates):
            optional_rows.append(
                {
                    "symbol": item["symbol"],
                    "trade_date": date,
                    "availability": date,
                    "is_open_day": True,
                    "metric": "fin_roa",
                    "value": float(item["fin_roa"]) + (offset - 126) / 10_000.0,
                }
            )
    history = pd.concat([history, pd.DataFrame(optional_rows)], ignore_index=True)

    result = score_fundamental_universe(snapshot, history, cutoff=CUTOFF)

    failed = result.scored.set_index("symbol").loc["000000.SZ"]
    assert failed["status"] == "UNAVAILABLE"
    assert "self_history_below_252:fin_roa" in failed["unavailable_reasons"]


def test_provided_optional_metric_with_small_industry_sample_is_unavailable() -> None:
    snapshot = _snapshot()
    snapshot["forecast_revision"] = np.nan
    snapshot.loc[:18, "forecast_revision"] = np.linspace(0.01, 0.19, 19)
    history = _history(snapshot)
    dates = pd.date_range(end=CUTOFF - pd.offsets.BDay(1), periods=252, freq="B")
    optional_rows: list[dict[str, object]] = []
    for _, item in snapshot.loc[:18].iterrows():
        for offset, date in enumerate(dates):
            optional_rows.append(
                {
                    "symbol": item["symbol"],
                    "trade_date": date,
                    "availability": date,
                    "is_open_day": True,
                    "metric": "forecast_revision",
                    "value": float(item["forecast_revision"]) + (offset - 126) / 10_000.0,
                }
            )
    history = pd.concat([history, pd.DataFrame(optional_rows)], ignore_index=True)

    result = score_fundamental_universe(snapshot, history, cutoff=CUTOFF)

    provided = result.scored.iloc[:19]
    assert set(provided["status"]) == {"UNAVAILABLE"}
    assert all(
        "industry_optional_metric_sample_below_20:forecast_revision" in reasons
        for reasons in provided["unavailable_reasons"]
    )


def test_history_requires_unique_explicit_open_sessions() -> None:
    snapshot = _snapshot()
    history = _history(snapshot)
    duplicated = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    try:
        score_fundamental_universe(snapshot, duplicated, cutoff=CUTOFF)
    except ValueError as exc:
        assert "duplicate symbol/metric/open-session" in str(exc)
    else:
        raise AssertionError("duplicate open-session evidence was accepted")

    non_open = history.copy()
    non_open.loc[0, "is_open_day"] = False
    try:
        score_fundamental_universe(snapshot, non_open, cutoff=CUTOFF)
    except ValueError as exc:
        assert "canonical open-market sessions" in str(exc)
    else:
        raise AssertionError("non-open history row was accepted")
