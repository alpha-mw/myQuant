"""Contracts and static field sets for the CN fundamental mart."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_FUNDAMENTAL_ROOT = Path("data/clean/cn_fundamental")
DEFAULT_RAW_SNAPSHOT_ROOT = Path("data/cn_market_full/_snapshots/fundamental")
DEFAULT_READINESS_ROOT = Path("reports/fundamental_readiness")
DEFAULT_DAILY_ROOT = Path("data/clean/cn_daily")
DEFAULT_METADATA_ROOT = Path("data/metadata")
DEFAULT_UNIVERSES = ("hs300", "zz500", "zz1000")
FULL_A_UNIVERSE_KEYS = {"full_a", "full_market", "all_a", "all", "full"}
FULL_A_PHYSICAL_DIRECTORIES = ("hs300", "zz500", "zz1000", "other")

SOURCE_TABLES = ("fina_indicator", "income", "balancesheet", "cashflow", "daily_basic", "forecast")
FINANCIAL_SOURCE_TABLES = ("fina_indicator", "income", "balancesheet", "cashflow")
DERIVED_PERIOD_FIELDS = (
    "fin_roe",
    "fin_roa",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_fcf_to_profit",
    "free_cashflow",
)
DERIVED_DAILY_FIELDS = DERIVED_PERIOD_FIELDS + ("fcf_to_price", "forecast_revision")
FORECAST_DAILY_COLUMNS = (
    "ts_code",
    "forecast_end_date",
    "availability_date",
    "forecast_ann_date",
    "forecast_revision",
    "forecast_type",
    "forecast_summary",
    "forecast_change_reason",
    "forecast_source",
    "forecast_fetched_at",
    "forecast_ingest_run_id",
)


@dataclass(frozen=True)
class FundamentalMartArtifacts:
    run_id: str
    data_root: Path
    raw_snapshot_root: Path
    reports_root: Path
    fundamental_period_path: Path
    fundamental_daily_path: Path
    quarantine_path: Path
    readiness_json_path: Path
    readiness_md_path: Path
    readiness_csv_path: Path


__all__ = [
    "DEFAULT_DAILY_ROOT",
    "DEFAULT_FUNDAMENTAL_ROOT",
    "DEFAULT_METADATA_ROOT",
    "DEFAULT_RAW_SNAPSHOT_ROOT",
    "DEFAULT_READINESS_ROOT",
    "DEFAULT_UNIVERSES",
    "DERIVED_DAILY_FIELDS",
    "DERIVED_PERIOD_FIELDS",
    "FINANCIAL_SOURCE_TABLES",
    "FORECAST_DAILY_COLUMNS",
    "FULL_A_PHYSICAL_DIRECTORIES",
    "FULL_A_UNIVERSE_KEYS",
    "FundamentalMartArtifacts",
    "SOURCE_TABLES",
]
