"""Revised statements must not win normalization by incoming row order.

Tushare returns the statement as filed (``update_flag`` "0") and its later
revision ("1") under the same ``ann_date``, so nothing but the flag separates
them. ``_normalize_table`` used to sort on ``fetched_at`` — identical within a
run — and keep the last row, making the surviving figure depend on arrival
order. A_quant hit the same defect (its docs/assumptions.md A1d); the two
pipelines must agree that the filed figure is the point-in-time one.
"""

import pandas as pd

from quant_investor.market.fundamental_mart import _keep_first_disclosure, _normalize_table

FILED = 9_606_318.29
REVISED = 28_160_560.34


def _rows(order: tuple[str, ...]) -> pd.DataFrame:
    values = {"0": FILED, "1": REVISED}
    return pd.DataFrame(
        {
            "ts_code": ["600771.SH"] * len(order),
            "end_date": ["20230331"] * len(order),
            "ann_date": ["20230427"] * len(order),
            "f_ann_date": ["20230427"] * len(order),
            "update_flag": list(order),
            "n_income_attr_p": [values[flag] for flag in order],
        }
    )


def _normalize(order: tuple[str, ...]) -> pd.DataFrame:
    clean, _ = _normalize_table(
        _rows(order), table="income", run_id="test-run", source="test",
        derivation_timestamp="2026-08-30T00:00:00+00:00",
    )
    return clean


def test_filed_figure_survives_when_the_revision_arrives_last():
    clean = _normalize(("0", "1"))
    assert len(clean) == 1
    assert clean["n_income_attr_p"].iloc[0] == FILED


def test_filed_figure_survives_when_the_revision_arrives_first():
    """The row order must not decide the value."""
    clean = _normalize(("1", "0"))
    assert len(clean) == 1
    assert clean["n_income_attr_p"].iloc[0] == FILED


def test_a_revision_only_period_is_not_dropped():
    clean = _normalize(("1",))
    assert len(clean) == 1
    assert clean["n_income_attr_p"].iloc[0] == REVISED


def test_frame_without_update_flag_passes_through():
    frame = _rows(("0",)).drop(columns=["update_flag"])
    assert _keep_first_disclosure(frame).equals(frame)


def test_empty_frame_passes_through():
    frame = pd.DataFrame()
    assert _keep_first_disclosure(frame).empty


def test_other_periods_are_untouched():
    frame = pd.concat(
        [
            _rows(("0", "1")),
            _rows(("0",)).assign(end_date="20230630", ann_date="20230830", f_ann_date="20230830"),
        ],
        ignore_index=True,
    )
    clean, _ = _normalize_table(
        frame, table="income", run_id="test-run", source="test",
        derivation_timestamp="2026-08-30T00:00:00+00:00",
    )
    assert len(clean) == 2
    assert set(clean["end_date"].astype(str)) == {"20230331", "20230630"}


# ---------------------------------------------------------------------------
# A second disclosure that refiles only some statements
# ---------------------------------------------------------------------------

from quant_investor.market.fundamental_mart import _outer_period_frame  # noqa: E402


def _table(availability: list[str], **columns) -> pd.DataFrame:
    """A normalized raw table; fetched_at/source are added by _normalize_table."""
    return pd.DataFrame(
        {
            "ts_code": ["600771.SH"] * len(availability),
            "end_date": ["20230331"] * len(availability),
            "availability_date": availability,
            "fetched_at": ["2026-08-30T00:00:00+00:00"] * len(availability),
            "source": ["test"] * len(availability),
            **columns,
        }
    )


def test_unrefiled_statements_carry_into_the_later_vintage():
    """Only the income statement is refiled; equity must not go NULL."""
    period = _outer_period_frame(
        fina_indicator=pd.DataFrame(),
        income=_table(["2023-04-27", "2024-01-30"], n_income_attr_p=[FILED, REVISED]),
        balancesheet=_table(["2023-04-27"], total_assets=[5e9], total_liab=[3e9]),
        cashflow=_table(["2023-04-27"], n_cashflow_act=[1e8]),
    )
    assert len(period) == 2
    later = period.loc[period["availability_date"] == "2024-01-30"].iloc[0]
    assert later["inc_n_income_attr_p"] == REVISED
    assert later["bs_total_assets"] == 5e9      # carried, not NULL
    assert later["cf_n_cashflow_act"] == 1e8


def test_carry_never_moves_a_value_backwards():
    """The first vintage must not inherit anything disclosed later."""
    period = _outer_period_frame(
        fina_indicator=pd.DataFrame(),
        income=_table(["2023-04-27", "2024-01-30"], n_income_attr_p=[FILED, REVISED]),
        balancesheet=_table(["2024-01-30"], total_assets=[5e9], total_liab=[3e9]),
        cashflow=pd.DataFrame(),
    )
    first = period.loc[period["availability_date"] == "2023-04-27"].iloc[0]
    assert pd.isna(first["bs_total_assets"])


def test_carry_never_crosses_report_periods():
    income = pd.concat(
        [
            _table(["2023-04-27"], n_income_attr_p=[FILED]),
            _table(["2023-08-30"], n_income_attr_p=[REVISED]).assign(end_date="20230630"),
        ],
        ignore_index=True,
    )
    balancesheet = _table(["2023-04-27"], total_assets=[5e9], total_liab=[3e9])
    period = _outer_period_frame(pd.DataFrame(), income, balancesheet, pd.DataFrame())
    q2 = period.loc[period["end_date"] == "20230630"].iloc[0]
    assert pd.isna(q2["bs_total_assets"])


# ---------------------------------------------------------------------------
# Restated vintages fetched from report_type "4"
# ---------------------------------------------------------------------------

from quant_investor.market.fundamental_mart import (  # noqa: E402
    RESTATABLE_TABLES,
    RESTATED_REPORT_TYPE,
    _restatement_is_new,
)


def _vintage(end_date: str, ann_date: str, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600771.SH"],
            "end_date": [end_date],
            "ann_date": [ann_date],
            "n_income_attr_p": [value],
        }
    )


def test_only_the_three_statements_are_restatable():
    """fina_indicator, daily_basic and forecast carry no restated vintage."""
    assert RESTATABLE_TABLES == ("income", "balancesheet", "cashflow")
    assert RESTATED_REPORT_TYPE == "4"


def test_a_later_announcement_is_a_new_vintage():
    kept = _restatement_is_new(
        _vintage("20230331", "20240430", REVISED),
        _vintage("20230331", "20230427", FILED),
    )
    assert len(kept) == 1
    assert kept["n_income_attr_p"].iloc[0] == REVISED


def test_a_same_day_restatement_is_not_a_vintage():
    kept = _restatement_is_new(
        _vintage("20230331", "20230427", REVISED),
        _vintage("20230331", "20230427", FILED),
    )
    assert kept.empty


def test_an_earlier_announcement_is_not_a_vintage():
    kept = _restatement_is_new(
        _vintage("20230331", "20230101", REVISED),
        _vintage("20230331", "20230427", FILED),
    )
    assert kept.empty


def test_a_period_absent_from_the_primary_is_kept():
    kept = _restatement_is_new(
        _vintage("20221231", "20240430", REVISED),
        _vintage("20230331", "20230427", FILED),
    )
    assert len(kept) == 1


def test_an_undateable_primary_drops_the_restatement():
    """Without an ordering the second vintage would duplicate, not date."""
    primary = _vintage("20230331", "20230427", FILED)
    primary["ann_date"] = [None]
    kept = _restatement_is_new(_vintage("20230331", "20240430", REVISED), primary)
    assert kept.empty


def test_an_empty_primary_keeps_everything():
    kept = _restatement_is_new(_vintage("20230331", "20240430", REVISED), pd.DataFrame())
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# A transient network fault must not decide whether a symbol keeps its vintage
# ---------------------------------------------------------------------------

from quant_investor.market.fundamental_mart import _fetch_restated_rows  # noqa: E402



def _fetchable(end_date: str, ann_date: str, value: float) -> pd.DataFrame:
    """A restated row shaped as _strict_pit_cutoff requires it."""
    return pd.DataFrame(
        {
            "ts_code": ["600771.SH"],
            "end_date": [end_date],
            "ann_date": [ann_date],
            "f_ann_date": [ann_date],
            "update_flag": ["0"],
            "n_income": [value],
            "n_income_attr_p": [value],
        }
    )


class _NoWait:
    def wait(self) -> None:
        return None


def _restated_call(responses):
    """A provider whose successive calls yield the given results or raises."""
    calls = {"n": 0}

    def method(**kwargs):
        assert kwargs.get("report_type") == RESTATED_REPORT_TYPE
        index = calls["n"]
        calls["n"] += 1
        outcome = responses[min(index, len(responses) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return method, calls


def test_a_transient_failure_is_retried_and_recovers():
    """One TLS drop cost a symbol its restatement before this was retried."""
    good = _fetchable("20230331", "20240430", REVISED)
    method, calls = _restated_call([OSError("TLS drop"), good])
    kept, spent, _ = _fetch_restated_rows(
        method,
        symbol="600771.SH",
        table="income",
        table_start_text="20190101",
        end_text="20260828",
        primary=_fetchable("20230331", "20230427", FILED),
        limiter=_NoWait(),
        attempt_limit=3,
        initial_backoff=0.0,
        maximum_backoff=0.0,
    )
    assert calls["n"] == 2
    assert spent == 2
    assert len(kept) == 1
    assert kept["n_income_attr_p"].iloc[0] == REVISED


def test_exhausted_retries_leave_the_primary_rows_intact():
    """Giving up on the restatement must never fail the table."""
    method, calls = _restated_call([OSError("TLS drop")])
    kept, spent, stats = _fetch_restated_rows(
        method,
        symbol="600771.SH",
        table="income",
        table_start_text="20190101",
        end_text="20260828",
        primary=_fetchable("20230331", "20230427", FILED),
        limiter=_NoWait(),
        attempt_limit=3,
        initial_backoff=0.0,
        maximum_backoff=0.0,
    )
    assert calls["n"] == 3
    assert spent == 3
    assert kept.empty
    assert stats["rows"] == 0


def test_a_provider_without_report_type_is_not_retried():
    """TypeError means the endpoint has no such vintage, not a bad connection."""
    method, calls = _restated_call([TypeError("unexpected keyword")])
    kept, spent, _ = _fetch_restated_rows(
        method,
        symbol="600771.SH",
        table="income",
        table_start_text="20190101",
        end_text="20260828",
        primary=pd.DataFrame(),
        limiter=_NoWait(),
        attempt_limit=3,
        initial_backoff=0.0,
        maximum_backoff=0.0,
    )
    assert calls["n"] == 1
    assert spent == 1
    assert kept.empty
