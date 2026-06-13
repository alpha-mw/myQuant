from __future__ import annotations

from quant_investor.market.tushare_cleaning_types import (
    CLEANING_STATUS_PASS,
    TushareCleanProfile,
    TushareCleaningReport,
    make_cleaning_report_id,
)
from quant_investor.market.tushare_cleaning_profiles import (
    get_default_tushare_clean_profiles,
)
from quant_investor.market.tushare_data_cleaning import (
    get_default_tushare_clean_profiles as reexported_default_profiles,
    TushareCleanProfile as ReexportedTushareCleanProfile,
)


def test_tushare_cleaning_types_round_trip_and_legacy_reexport():
    profile = TushareCleanProfile(
        table_name="daily",
        primary_key=["trade_date", "ts_code", "ts_code"],
        required_columns=["ts_code", "trade_date"],
    )
    payload = profile.to_dict()

    assert payload["primary_key"] == ["trade_date", "ts_code"]
    assert TushareCleanProfile.from_dict(payload).to_dict() == payload
    assert ReexportedTushareCleanProfile is TushareCleanProfile

    report_id = make_cleaning_report_id(
        table_name="daily",
        source_path="raw.csv",
        generated_at="2026-01-01T00:00:00Z",
    )
    report = TushareCleaningReport(
        report_id=report_id,
        table_name="daily",
        generated_at="2026-01-01T00:00:00Z",
    )

    assert report.status == CLEANING_STATUS_PASS
    assert report.to_dict()["report_id"] == report_id


def test_tushare_cleaning_profiles_are_split_and_reexported():
    profiles = get_default_tushare_clean_profiles()

    assert reexported_default_profiles is get_default_tushare_clean_profiles
    assert sorted(profiles) == [
        "adj_factor",
        "daily",
        "daily_basic",
        "index_daily",
        "index_weight",
        "stk_limit",
        "stock_basic",
        "suspend",
        "suspend_d",
        "trade_cal",
    ]
    assert profiles["daily"].preferred_storage_format == "dual"
    assert profiles["daily"].factor_required_columns == [
        "adj_factor",
        "amount",
        "close",
        "high",
        "low",
        "open",
        "trade_date",
        "ts_code",
        "vol",
    ]
    assert "adj_close" in profiles["daily"].model_optional_columns
