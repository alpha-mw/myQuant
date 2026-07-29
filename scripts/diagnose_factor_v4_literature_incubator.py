#!/usr/bin/env python3
"""Run a no-label computability diagnostic for literature-backed v4 ideas."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_investor.factors import (  # noqa: E402
    governance_candidate_preregistration_v4_4 as exact_five_prereg,
)
from quant_investor.factors import (  # noqa: E402
    governance_exact_five_no_label_eval_v4_4 as exact_five_eval,
)
from quant_investor.factors import (  # noqa: E402
    governance_literature_incubator_v4 as incubator,
)
from quant_investor.factors import governance_screening_v4 as screening  # noqa: E402

SCHEMA_VERSION = "factor-governance-literature-incubator-diagnostic.v11"
DEFAULT_MARKET_POINTER = REPO_ROOT / "data/parquet/cn/_latest.json"
DEFAULT_FUNDAMENTAL_POINTER = REPO_ROOT / "data/parquet/cn/_fundamental_latest.json"
DEFAULT_COMPARISON_CATALOG = (
    REPO_ROOT
    / "reports/factor_governance/private/v4_1_formal_catalog"
    / "factor_v4_1_formal_catalog_20260718T191045Z"
    / "candidate_catalog.v4.json"
)
DEFAULT_COMPARISON_ONTOLOGY = (
    REPO_ROOT
    / "reports/factor_governance/private/v4_1_formal_catalog"
    / "factor_v4_1_formal_catalog_20260718T191045Z"
    / "primitive_ontology.v4.json"
)
DEFAULT_LOOKBACK_SESSIONS = incubator.SAME_MONTH_LOOKBACK_SESSIONS
STANDARD_DIAGNOSTIC_LOOKBACK_SESSIONS = 300
CORRELATION_MIN_CROSS_SECTION = 100
DEDUP_MIN_CROSS_SECTION = 20
DEDUP_MIN_MONTHS = 3
DEDUP_THRESHOLD = 0.70
SEASONALITY_CALENDAR_COVERAGE_FLOOR = 0.80
PROTECTED_EXACT_FIVE_NAMES = incubator.PROTECTED_EXACT_FIVE_CANDIDATE_NAMES

DEDUP_COMPARISON_ROUTES = {
    "cn_earnings_yield_ex_shell_30pct": ("fund_fcf_to_price",),
    "cn_low_beta_252d": (
        "pv_downside_volatility_60d",
        "pv_volatility_penalty_60d",
    ),
    "cn_52_week_high_momentum_12m": (
        "cn_low_total_skewness_20d",
        "pv_momentum_20d",
        "pv_momentum_120d",
        "pv_short_reversal_20d",
    ),
    "cn_high_price_delay_d1_52w": (
        "alpha_turnover_low_20d",
        "pv_amihud_illiquidity_20d",
        "pv_momentum_20d",
        "pv_price_efficiency_60d",
        "pv_short_reversal_20d",
    ),
    "cn_low_max_return_20d": (
        "pv_downside_volatility_60d",
        "pv_momentum_20d",
        "pv_volatility_penalty_60d",
    ),
    "cn_low_total_skewness_20d": (
        "cn_low_market_adjusted_tail_asymmetry_252d",
        "cn_low_max_return_20d",
        "pv_downside_volatility_60d",
        "pv_momentum_20d",
        "pv_volatility_penalty_60d",
    ),
    "cn_low_market_adjusted_tail_asymmetry_252d": (
        "cn_low_max_return_20d",
        "pv_downside_volatility_60d",
        "pv_momentum_20d",
        "pv_volatility_penalty_60d",
    ),
    "cn_quality_cash_low_leverage": (
        "formula_cash_growth_lowlev_w50",
        "fund_quality_cash_combo",
        "fund_quality_low_leverage_combo",
    ),
    "cn_same_month_seasonality_5y": (
        "pv_momentum_20d",
        "pv_momentum_120d",
    ),
    "cn_fip_continuous_direction_12m": (
        "cn_52_week_high_momentum_12m",
        "pv_momentum_20d",
        "pv_momentum_120d",
        "pv_price_efficiency_60d",
        "pv_short_reversal_20d",
    ),
    "cn_low_left_tail_var1_250d": (
        "cn_low_market_adjusted_tail_asymmetry_252d",
        "cn_low_max_return_20d",
        "cn_low_total_skewness_20d",
        "pv_downside_volatility_60d",
        "pv_momentum_20d",
        "pv_momentum_120d",
        "pv_short_reversal_20d",
        "pv_volatility_penalty_60d",
    ),
}


def _dedup_route_names(candidate_name: str) -> tuple[str, ...]:
    base = DEDUP_COMPARISON_ROUTES[candidate_name]
    return tuple(dict.fromkeys((*base, *PROTECTED_EXACT_FIVE_NAMES)))


class FactorV4LiteratureDiagnosticError(RuntimeError):
    """Raised when the offline diagnostic cannot bind its inputs."""


def _error(message: str) -> FactorV4LiteratureDiagnosticError:
    return FactorV4LiteratureDiagnosticError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise _error(f"JSON input must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise _error(f"cannot hash input {path}: {exc}") from exc
    return digest.hexdigest()


def _resolve_repo_path(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise _error(f"{label} path is missing")
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise _error(f"{label} must remain inside the repository") from exc
    if not resolved.is_file() and not resolved.is_dir():
        raise _error(f"{label} does not exist: {resolved}")
    return resolved


def _load_market_inputs(
    pointer_path: Path,
    *,
    lookback_sessions: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    pointer_sha = _sha256_file(pointer_path)
    pointer = _read_json(pointer_path)
    if _sha256_file(pointer_path) != pointer_sha:
        raise _error("market pointer changed while it was read")
    coverage = pointer.get("coverage")
    if (
        pointer.get("status") != "OK"
        or not isinstance(coverage, dict)
        or coverage.get("complete") is not True
        or coverage.get("coverage_ratio") != 1.0
        or pointer.get("blockers")
    ):
        raise _error("market pointer is not strict complete")
    snapshot_id = pointer.get("snapshot_id")
    latest_trade_date = pointer.get("latest_trade_date")
    if type(snapshot_id) is not str or type(latest_trade_date) is not str:
        raise _error("market pointer identity is incomplete")
    manifest_path = _resolve_repo_path(
        pointer.get("manifest_path"), label="market snapshot manifest"
    )
    manifest_sha = _sha256_file(manifest_path)
    manifest = _read_json(manifest_path)
    if _sha256_file(manifest_path) != manifest_sha:
        raise _error("market snapshot manifest changed while it was read")
    if (
        manifest.get("snapshot_id") != snapshot_id
        or manifest.get("latest_trade_date") != latest_trade_date
        or manifest.get("readback_validated") is not True
    ):
        raise _error("market snapshot manifest differs from the pointer")
    serving_root = _resolve_repo_path(
        pointer.get("derived_serving_root"), label="strict serving root"
    )
    dataset = ds.dataset(
        str(serving_root),
        format="parquet",
        exclude_invalid_files=True,
    )
    required_columns = {
        "ts_code",
        "trade_date",
        "open",
        "close",
        "vol",
        "amount",
        "adj_close",
        "pe",
        "total_mv",
        "turnover_rate",
    }
    if not required_columns.issubset(dataset.schema.names):
        raise _error("strict serving root lacks required incubator fields")
    dates_table = dataset.to_table(columns=["trade_date"])
    dates = pd.to_datetime(dates_table.column("trade_date").to_pandas(), format="%Y%m%d")
    sessions = pd.DatetimeIndex(sorted(pd.unique(dates)), name="trade_date")
    if len(sessions) < lookback_sessions:
        raise _error("strict serving history is shorter than requested lookback")
    sessions = sessions[-lookback_sessions:]
    start_text = sessions[0].strftime("%Y%m%d")
    end_text = sessions[-1].strftime("%Y%m%d")
    table = dataset.to_table(
        columns=[
            "ts_code",
            "trade_date",
            "open",
            "close",
            "vol",
            "amount",
            "adj_close",
            "pe",
            "total_mv",
            "turnover_rate",
        ],
        filter=(ds.field("trade_date") >= start_text) & (ds.field("trade_date") <= end_text),
    )
    raw = table.to_pandas()
    if raw.empty or raw.duplicated(["trade_date", "ts_code"]).any():
        raise _error("strict serving slice is empty or contains duplicate keys")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    raw["ts_code"] = raw["ts_code"].astype(str)
    serving_symbols = sorted(raw["ts_code"].unique())

    pit_path = _resolve_repo_path(coverage.get("pit_membership_path"), label="PIT membership")
    expected_pit_sha = coverage.get("pit_membership_sha256")
    observed_pit_sha = _sha256_file(pit_path)
    if expected_pit_sha != observed_pit_sha:
        raise _error("PIT membership SHA differs from the market pointer")
    pit = pd.read_parquet(
        pit_path,
        columns=["symbol", "effective_from", "effective_to"],
    )
    pit["symbol"] = pit["symbol"].astype(str)
    if pit["symbol"].duplicated().any():
        raise _error("PIT membership contains duplicate symbols")
    membership_symbols = set(pit["symbol"])
    unbound_serving_symbols = sorted(set(serving_symbols) - membership_symbols)
    symbols = pd.Index(
        sorted(set(serving_symbols) & membership_symbols),
        name="ts_code",
    )
    if symbols.empty:
        raise _error("strict serving and PIT membership have no common symbols")
    pit = pit.set_index("symbol").reindex(symbols)
    if pit["effective_from"].isna().any():
        raise _error("selected PIT membership contains missing effective dates")
    session_text = np.asarray(sessions.strftime("%Y%m%d"), dtype="U8")
    starts = pit["effective_from"].astype(str).str.replace("-", "", regex=False).to_numpy()
    ends = pit["effective_to"].fillna("").astype(str).str.replace("-", "", regex=False).to_numpy()
    mask_values = np.zeros((len(sessions), len(symbols)), dtype=bool)
    for column_index in range(len(symbols)):
        mask_values[:, column_index] = (session_text >= starts[column_index]) & (
            (ends[column_index] == "") | (session_text < ends[column_index])
        )
    pit_mask = pd.DataFrame(
        mask_values,
        index=sessions,
        columns=symbols,
        dtype=bool,
    )

    matrices: dict[str, pd.DataFrame] = {}
    for field in (
        "open",
        "close",
        "vol",
        "amount",
        "adj_close",
        "pe",
        "total_mv",
        "turnover_rate",
    ):
        matrices[field] = (
            raw.pivot(index="trade_date", columns="ts_code", values=field)
            .reindex(index=sessions, columns=symbols)
            .astype(float)
            .where(pit_mask)
        )
    binding = {
        "market_pointer_path": str(pointer_path.resolve()),
        "market_pointer_sha256": pointer_sha,
        "snapshot_id": snapshot_id,
        "latest_trade_date": latest_trade_date,
        "market_manifest_path": str(manifest_path),
        "market_manifest_sha256": manifest_sha,
        "strict_serving_root": str(serving_root),
        "pit_membership_path": str(pit_path),
        "pit_membership_sha256": observed_pit_sha,
        "session_count": len(sessions),
        "first_session": sessions[0].date().isoformat(),
        "last_session": sessions[-1].date().isoformat(),
        "symbol_count": len(symbols),
        "unbound_serving_symbol_count": len(unbound_serving_symbols),
        "unbound_serving_symbols_excluded": unbound_serving_symbols,
        "missing_pit_rows_failed_open": False,
    }
    return matrices, pit_mask, binding


def _load_fundamental_inputs(
    pointer_path: Path,
    *,
    sessions: pd.DatetimeIndex,
    symbols: pd.Index,
    pit_mask: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    pointer_sha = _sha256_file(pointer_path)
    pointer = _read_json(pointer_path)
    if _sha256_file(pointer_path) != pointer_sha:
        raise _error("Fundamental pointer changed while it was read")
    if pointer.get("status") != "OK" or pointer.get("metadata", {}).get("gate2_passed") is not True:
        raise _error("Fundamental pointer is not Gate2-ready")
    tables = pointer.get("tables")
    if not isinstance(tables, dict):
        raise _error("Fundamental pointer table map is missing")
    daily_path = _resolve_repo_path(
        str(Path("data/parquet/cn") / str(tables.get("fundamental_daily"))),
        label="Fundamental daily table",
    )
    dataset = ds.dataset(str(daily_path), format="parquet")
    fields = (
        "fcf_to_price",
        "fin_debt_to_assets",
        "fin_ocf_to_profit",
        "fin_roe",
    )
    required = {"ts_code", "trade_date", *fields}
    if not required.issubset(dataset.schema.names):
        raise _error("Fundamental daily table lacks required incubator fields")
    start_date = sessions[0].date()
    end_date = sessions[-1].date()
    table = dataset.to_table(
        columns=["ts_code", "trade_date", *fields],
        filter=(ds.field("trade_date") >= start_date) & (ds.field("trade_date") <= end_date),
    )
    raw = table.to_pandas()
    if raw.empty or raw.duplicated(["trade_date", "ts_code"]).any():
        raise _error("Fundamental daily slice is empty or contains duplicate keys")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["ts_code"] = raw["ts_code"].astype(str)
    matrices: dict[str, pd.DataFrame] = {}
    for field in fields:
        matrices[field] = (
            raw.pivot(index="trade_date", columns="ts_code", values=field)
            .reindex(index=sessions, columns=symbols)
            .astype(float)
            .where(pit_mask)
        )
    observed_dates = pd.DatetimeIndex(sorted(raw["trade_date"].unique()))
    binding = {
        "fundamental_pointer_path": str(pointer_path.resolve()),
        "fundamental_pointer_sha256": pointer_sha,
        "generation_id": pointer.get("generation_id"),
        "fundamental_daily_path": str(daily_path),
        "fundamental_daily_sha256": _sha256_file(daily_path),
        "first_observed_session": observed_dates[0].date().isoformat(),
        "last_observed_session": observed_dates[-1].date().isoformat(),
        "lag_calendar_days_vs_market": (sessions[-1].date() - observed_dates[-1].date()).days,
    }
    return matrices, binding


def _signal_summary(
    signal: pd.DataFrame,
    *,
    pit_mask: pd.DataFrame,
) -> dict[str, Any]:
    finite = (
        pd.DataFrame(
            np.isfinite(signal.to_numpy(dtype=float)),
            index=signal.index,
            columns=signal.columns,
        )
        & pit_mask
    )
    finite_by_date = finite.sum(axis=1)
    eligible_by_date = pit_mask.sum(axis=1)
    valid_dates = finite_by_date[finite_by_date > 0].index
    if valid_dates.empty:
        return {
            "status": "NOT_COMPUTABLE",
            "first_computable_session": None,
            "last_computable_session": None,
            "latest_finite_count": 0,
            "latest_eligible_count": 0,
            "latest_coverage_ratio": 0.0,
            "median_coverage_ratio": 0.0,
        }
    coverage = finite_by_date.div(eligible_by_date.replace(0, np.nan))
    last = valid_dates[-1]
    return {
        "status": "COMPUTABLE_RESEARCH_ONLY",
        "first_computable_session": valid_dates[0].date().isoformat(),
        "last_computable_session": last.date().isoformat(),
        "latest_finite_count": int(finite_by_date.loc[last]),
        "latest_eligible_count": int(eligible_by_date.loc[last]),
        "latest_coverage_ratio": float(coverage.loc[last]),
        "median_coverage_ratio": float(coverage.loc[valid_dates].median()),
    }


def _price_delay_input_history_diagnostic(
    *,
    market: dict[str, pd.DataFrame],
    pit_mask: pd.DataFrame,
) -> dict[str, Any]:
    wednesdays = pit_mask.index[pit_mask.index.weekday == incubator.PRICE_DELAY_WEEKDAY]
    close = market["adj_close"].where(market["adj_close"] > 0.0).reindex(wednesdays)
    market_cap = market["total_mv"].where(market["total_mv"] > 0.0).reindex(wednesdays)
    weekly_return = close.pct_change(fill_method=None)
    prior_cap = market_cap.shift(1)
    usable_weight = prior_cap.where(weekly_return.notna() & prior_cap.gt(0.0))
    market_return = (
        weekly_return.mul(usable_weight)
        .sum(axis=1, min_count=1)
        .div(usable_weight.sum(axis=1, min_count=1).replace(0.0, np.nan))
    )
    design_finite = (
        pd.concat(
            [market_return.shift(lag) for lag in range(incubator.PRICE_DELAY_MARKET_LAGS + 1)],
            axis=1,
        )
        .notna()
        .all(axis=1)
    )
    rolling_valid_design = (
        design_finite.astype(int)
        .rolling(
            incubator.PRICE_DELAY_WINDOW_WEEKS,
            min_periods=1,
        )
        .sum()
    )
    maximum_valid_design = int(rolling_valid_design.max()) if not rolling_valid_design.empty else 0
    minimum_cross_section = DEDUP_MIN_CROSS_SECTION
    result = {
        "status": (
            "INPUT_HISTORY_READY"
            if maximum_valid_design >= incubator.PRICE_DELAY_MIN_OBSERVATIONS
            else "BLOCKED_INSUFFICIENT_PIT_VALUE_WEIGHTED_MARKET_HISTORY"
        ),
        "exact_wednesday_anchor_count": len(wednesdays),
        "wednesdays_with_minimum_adj_close_cross_section": int(
            close.notna().sum(axis=1).ge(minimum_cross_section).sum()
        ),
        "wednesdays_with_minimum_total_mv_cross_section": int(
            market_cap.notna().sum(axis=1).ge(minimum_cross_section).sum()
        ),
        "finite_value_weighted_market_return_count": int(market_return.notna().sum()),
        "maximum_valid_design_observations_in_52_week_window": maximum_valid_design,
        "required_valid_design_observations": incubator.PRICE_DELAY_MIN_OBSERVATIONS,
        "minimum_cross_section_count": minimum_cross_section,
        "equal_weight_fallback_used": False,
        "labels_loaded": False,
        "forward_returns_loaded": False,
        "factor_v4_authority": False,
    }
    return result


def _pairwise_correlation_diagnostic(
    signals: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    names = sorted(signals)
    rows: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = signals[left_name]
            right = signals[right_name]
            values: list[float] = []
            for session in left.index.intersection(right.index):
                joined = pd.concat([left.loc[session], right.loc[session]], axis=1).dropna()
                if len(joined) < CORRELATION_MIN_CROSS_SECTION:
                    continue
                correlation = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")
                if np.isfinite(correlation):
                    values.append(float(correlation))
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "valid_session_count": len(values),
                    "mean_cross_sectional_spearman": (float(np.mean(values)) if values else None),
                    "median_cross_sectional_spearman": (
                        float(np.median(values)) if values else None
                    ),
                    "formal_dedup_evidence": False,
                }
            )
    return rows


def _cross_sectional_rank(values: pd.DataFrame) -> pd.DataFrame:
    return values.rank(
        axis=1,
        method="average",
        pct=True,
        na_option="keep",
    )


def _build_momentum_comparison_signals(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    close = adj_close.where(adj_close > 0.0)
    return {
        "pv_momentum_20d": close.div(close.shift(20)).sub(1.0).where(pit_mask),
        "pv_momentum_120d": close.div(close.shift(120)).sub(1.0).where(pit_mask),
    }


def _build_comparison_signals(
    *,
    market: dict[str, pd.DataFrame],
    fundamental: dict[str, pd.DataFrame],
    pit_mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    close = market["adj_close"].where(market["adj_close"] > 0.0)
    returns = close.pct_change(fill_method=None).where(pit_mask)
    amount = market["amount"].where(market["amount"] > 0.0)
    volatility = -returns.rolling(60, min_periods=5).std(ddof=1)
    downside = returns.where(returns < 0.0, 0.0)
    downside_volatility = -downside.rolling(
        60,
        min_periods=5,
    ).std(ddof=1)
    roe_rank = _cross_sectional_rank(fundamental["fin_roe"])
    cash_rank = _cross_sectional_rank(fundamental["fin_ocf_to_profit"])
    debt_rank = _cross_sectional_rank(fundamental["fin_debt_to_assets"])
    low_debt_rank = _cross_sectional_rank(-fundamental["fin_debt_to_assets"])
    path_length_60 = returns.abs().rolling(60, min_periods=5).sum()
    price_efficiency_60 = (
        close.div(close.shift(60)).sub(1.0).abs().div(path_length_60.replace(0.0, np.nan))
    )
    return {
        **_build_momentum_comparison_signals(
            adj_close=market["adj_close"],
            pit_mask=pit_mask,
        ),
        "fund_fcf_to_price": _cross_sectional_rank(fundamental["fcf_to_price"]).where(pit_mask),
        "pv_volatility_penalty_60d": volatility.where(pit_mask),
        "pv_downside_volatility_60d": downside_volatility.where(pit_mask),
        "alpha_turnover_low_20d": _cross_sectional_rank(
            -market["turnover_rate"].rolling(20, min_periods=5).mean()
        ).where(pit_mask),
        "pv_amihud_illiquidity_20d": (
            returns.abs().div(amount).rolling(20, min_periods=5).mean().where(pit_mask)
        ),
        "pv_price_efficiency_60d": price_efficiency_60.where(pit_mask),
        "pv_short_reversal_20d": close.div(close.shift(20)).sub(1.0).mul(-1.0).where(pit_mask),
        "fund_quality_cash_combo": (roe_rank + cash_rank).where(pit_mask),
        "fund_quality_low_leverage_combo": (roe_rank - debt_rank).where(pit_mask),
        "formula_cash_growth_lowlev_w50": (cash_rank.mul(0.5) + low_debt_rank.mul(0.5)).where(
            pit_mask
        ),
    }


def _build_protected_exact_five_comparison_signals(
    *,
    market: dict[str, pd.DataFrame],
    pit_mask: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    protected_rows = exact_five_prereg.EXPECTED_CANDIDATE_ROWS
    protected_names = tuple(row["name"] for row in protected_rows)
    if protected_names != PROTECTED_EXACT_FIVE_NAMES:
        raise _error("loaded exact-five candidate names differ from the protected set")
    program_names = tuple(row["name"] for row in exact_five_eval.SOURCE_PROGRAMS_V4_4)
    if program_names != protected_names:
        raise _error("loaded exact-five programs differ from the protected set")
    if tuple(exact_five_eval.CANDIDATE_DIRECTIONS) != protected_names:
        raise _error("loaded exact-five directions differ from the protected set")
    inputs = {
        "raw_close": market["close"],
        "raw_open": market["open"],
        "vol": market["vol"],
        "adj_close": market["adj_close"],
    }
    source = exact_five_eval.evaluate_source_dag_v4_4(
        inputs=inputs,
        pit_mask=pit_mask,
    )
    local = exact_five_eval.evaluate_local_formulas_v4_4(
        inputs=inputs,
        pit_mask=pit_mask,
    )
    adjusted: dict[str, pd.DataFrame] = {}
    for name in protected_names:
        try:
            pd.testing.assert_frame_equal(
                source[name],
                local[name],
                check_exact=True,
                check_dtype=True,
                check_names=True,
            )
        except AssertionError as exc:
            raise _error(f"exact-five source/local engines differ for {name}") from exc
        adjusted[name] = source[name].mul(exact_five_eval.CANDIDATE_DIRECTIONS[name])
    return (
        adjusted,
        {
            "status": "RESEARCH_DIAGNOSTIC_ONLY",
            "candidate_names": list(protected_names),
            "candidate_definition_identities": {
                row["name"]: row["definition_identity_sha256"] for row in protected_rows
            },
            "source_engine_id": exact_five_eval.SOURCE_ENGINE_ID,
            "local_engine_id": exact_five_eval.LOCAL_ENGINE_ID,
            "source_programs_semantic_sha256": (
                exact_five_eval.source_programs_semantic_sha256_v4_4()
            ),
            "source_local_engine_equivalence_proven": True,
            "directions_applied": copy.deepcopy(exact_five_eval.CANDIDATE_DIRECTIONS),
            "labels_loaded": False,
            "forward_returns_loaded": False,
            "formal_dedup_evidence": False,
            "factor_v4_authority": False,
        },
    )


def _closed_month_end_sessions(
    sessions: pd.DatetimeIndex,
) -> list[pd.Timestamp]:
    if sessions.empty:
        return []
    periods = sessions.to_period("M")
    latest_period = periods[-1]
    result: list[pd.Timestamp] = []
    for period in periods.unique():
        if period >= latest_period:
            continue
        positions = np.flatnonzero(periods == period)
        result.append(sessions[int(positions[-1])])
    return result


def _monthly_dedup_diagnostic(
    *,
    candidate_signals: dict[str, pd.DataFrame],
    comparison_signals: dict[str, pd.DataFrame],
    comparison_catalog: dict[str, Any],
    protected_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    incubator_catalog_by_name = {row["name"]: row for row in incubator.candidate_catalog_v4()}
    formal_catalog_by_name = {row["name"]: row for row in comparison_catalog["candidates"]}
    protected_catalog_by_name: dict[str, dict[str, Any]] = {}
    for row in protected_candidates:
        slot = row["slot"]
        if not slot.startswith("primitive:") or slot.count(":") != 1:
            raise _error("protected exact-five slot is not a primitive slot")
        protected_catalog_by_name[row["name"]] = {
            "definition_sha256": row["definition_identity_sha256"],
            "primitive_ids": [slot.removeprefix("primitive:")],
        }
    if tuple(protected_catalog_by_name) != PROTECTED_EXACT_FIVE_NAMES:
        raise _error("protected exact-five catalog differs from the frozen set")
    catalog_name_collisions = sorted(
        (set(incubator_catalog_by_name) & set(formal_catalog_by_name))
        | (set(incubator_catalog_by_name) & set(protected_catalog_by_name))
        | (set(formal_catalog_by_name) & set(protected_catalog_by_name))
    )
    if catalog_name_collisions:
        raise _error("comparison namespaces collide: " + ",".join(catalog_name_collisions))
    catalog_by_name = {
        **incubator_catalog_by_name,
        **formal_catalog_by_name,
        **protected_catalog_by_name,
    }
    expected_names = {
        name
        for candidate_name in DEDUP_COMPARISON_ROUTES
        for name in _dedup_route_names(candidate_name)
    }
    missing_catalog = sorted(expected_names - set(catalog_by_name))
    missing_signals = sorted(expected_names - set(comparison_signals))
    if missing_catalog:
        raise _error("comparison catalog lacks routed factors: " + ",".join(missing_catalog))
    if missing_signals:
        raise _error("comparison signal map lacks routed factors: " + ",".join(missing_signals))
    rows: list[dict[str, Any]] = []
    for candidate_name in sorted(DEDUP_COMPARISON_ROUTES):
        candidate = candidate_signals[candidate_name]
        for comparison_name in _dedup_route_names(candidate_name):
            comparison = comparison_signals[comparison_name]
            common_sessions = candidate.index.intersection(comparison.index)
            month_ends = _closed_month_end_sessions(common_sessions)
            correlations: list[float] = []
            monthly_rows: list[dict[str, Any]] = []
            for month_end in month_ends:
                joined = pd.concat(
                    [
                        candidate.loc[month_end],
                        comparison.loc[month_end],
                    ],
                    axis=1,
                ).dropna()
                if len(joined) < DEDUP_MIN_CROSS_SECTION:
                    continue
                value = joined.iloc[:, 0].corr(
                    joined.iloc[:, 1],
                    method="spearman",
                )
                if not np.isfinite(value):
                    continue
                absolute = float(abs(value))
                correlations.append(absolute)
                monthly_rows.append(
                    {
                        "month_end": month_end.date().isoformat(),
                        "abs_spearman": absolute,
                        "valid_common_symbol_count": len(joined),
                    }
                )
            median_abs = (
                float(np.median(correlations)) if len(correlations) >= DEDUP_MIN_MONTHS else None
            )
            breached = bool(median_abs is not None and median_abs >= DEDUP_THRESHOLD)
            if median_abs is None:
                status = "INSUFFICIENT_MONTHS_DIAGNOSTIC"
            elif breached:
                status = "THRESHOLD_BREACHED_DIAGNOSTIC"
            else:
                status = "BELOW_THRESHOLD_DIAGNOSTIC"
            existing = catalog_by_name[comparison_name]
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "existing_factor_name": comparison_name,
                    "existing_factor_source": (
                        "formal_comparison_catalog"
                        if comparison_name in formal_catalog_by_name
                        else (
                            "protected_v4_4_exact_five"
                            if comparison_name in protected_catalog_by_name
                            else "literature_incubator_candidate"
                        )
                    ),
                    "existing_definition_sha256": existing["definition_sha256"],
                    "existing_primitive_ids": existing["primitive_ids"],
                    "metric": ("median_monthly_cross_sectional_abs_spearman"),
                    "threshold": DEDUP_THRESHOLD,
                    "minimum_valid_month_count": DEDUP_MIN_MONTHS,
                    "minimum_common_symbol_count": (DEDUP_MIN_CROSS_SECTION),
                    "closed_month_end_rows": monthly_rows,
                    "valid_common_date_count": len(correlations),
                    "abs_correlation": median_abs,
                    "threshold_breached_diagnostic": breached,
                    "status": status,
                    "formal_dedup_evidence": False,
                }
            )
    return {
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "metric": "median_monthly_cross_sectional_abs_spearman",
        "threshold": DEDUP_THRESHOLD,
        "minimum_valid_month_count": DEDUP_MIN_MONTHS,
        "minimum_common_symbol_count": DEDUP_MIN_CROSS_SECTION,
        "closed_natural_months_only": True,
        "comparison_route_complete": True,
        "rows": rows,
        "formal_dedup_evidence": False,
    }


def _seasonality_calendar_coverage_diagnostic(
    *,
    signal: pd.DataFrame,
    pit_mask: pd.DataFrame,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stable_calendar_months: set[int] = set()
    for session in _closed_month_end_sessions(signal.index):
        eligible = pit_mask.loc[session]
        eligible_count = int(eligible.sum())
        finite = pd.Series(
            np.isfinite(signal.loc[session].to_numpy(dtype=float)),
            index=signal.columns,
        )
        finite_count = int((finite & eligible).sum())
        coverage_ratio = float(finite_count / eligible_count) if eligible_count > 0 else 0.0
        stable = coverage_ratio >= SEASONALITY_CALENDAR_COVERAGE_FLOOR
        if stable:
            stable_calendar_months.add(session.month)
        rows.append(
            {
                "month_end": session.date().isoformat(),
                "calendar_month": session.month,
                "finite_count": finite_count,
                "eligible_count": eligible_count,
                "coverage_ratio": coverage_ratio,
                "coverage_floor_passed": stable,
            }
        )
    missing = sorted(set(range(1, 13)) - stable_calendar_months)
    return {
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "coverage_floor": SEASONALITY_CALENDAR_COVERAGE_FLOOR,
        "required_calendar_month_count": 12,
        "stable_calendar_months": sorted(stable_calendar_months),
        "stable_calendar_month_count": len(stable_calendar_months),
        "missing_calendar_months": missing,
        "all_calendar_months_covered": not missing,
        "closed_month_rows": rows,
        "formal_coverage_evidence": False,
    }


def _candidate_routing_decisions(
    *,
    structural_audit: dict[str, Any],
    protected_exact_five_audit: dict[str, Any],
    monthly_dedup: dict[str, Any],
    seasonality_calendar_coverage: dict[str, Any],
    literature_assessments: list[dict[str, Any]],
    candidate_computability: dict[str, dict[str, Any]],
    candidate_input_history_diagnostics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    structural_by_name = {
        row["candidate_name"]: row for row in structural_audit["candidate_results"]
    }
    protected_by_name = {
        row["candidate_name"]: row for row in protected_exact_five_audit["candidate_results"]
    }
    monthly_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in monthly_dedup["rows"]:
        monthly_by_name.setdefault(row["candidate_name"], []).append(row)
    literature_by_name = {row["candidate_name"]: row for row in literature_assessments}
    if (
        set(literature_by_name) != set(structural_by_name)
        or set(protected_by_name) != set(structural_by_name)
        or set(candidate_computability) != set(structural_by_name)
        or not set(candidate_input_history_diagnostics).issubset(structural_by_name)
    ):
        raise _error(
            "literature/protected/computability audits differ from structural candidate set"
        )
    decisions: list[dict[str, Any]] = []
    for candidate_name in sorted(structural_by_name):
        structural = structural_by_name[candidate_name]
        protected = protected_by_name[candidate_name]
        monthly = monthly_by_name.get(candidate_name, [])
        literature = literature_by_name[candidate_name]
        if not structural["structural_collision_passed_diagnostic"]:
            status = "STOP_STRUCTURAL_COLLISION"
            reasons = ["exact_definition_or_primitive_collision"]
        elif not protected["protected_structural_collision_passed_diagnostic"]:
            status = "STOP_PROTECTED_EXACT_FIVE_COLLISION"
            reasons = [
                *[
                    "protected_definition_identity_collision:" + name
                    for name in protected["definition_identity_collision_names"]
                ],
                *["protected_slot_collision:" + name for name in protected["slot_collision_names"]],
            ]
        elif literature["future_preregistration_eligible"] is not True:
            status = "CONTROL_ONLY_LITERATURE_MECHANISM_CONFLICT"
            reasons = [
                "literature_status:" + literature["status"],
                *["adverse_source:" + source_id for source_id in literature["adverse_source_ids"]],
            ]
        elif candidate_computability[candidate_name]["status"] != "COMPUTABLE_RESEARCH_ONLY":
            status = "BLOCKED_NOT_COMPUTABLE"
            reasons = [
                "candidate_signal_status:" + candidate_computability[candidate_name]["status"]
            ]
            input_history = candidate_input_history_diagnostics.get(candidate_name)
            if input_history is not None:
                reasons.append("input_history_status:" + input_history["status"])
        elif any(row["status"] == "THRESHOLD_BREACHED_DIAGNOSTIC" for row in monthly):
            status = "STOP_HIGH_CORRELATION"
            reasons = sorted(
                "dedup_threshold_breached:" + row["existing_factor_name"]
                for row in monthly
                if row["status"] == "THRESHOLD_BREACHED_DIAGNOSTIC"
            )
        elif candidate_name == "cn_52_week_high_momentum_12m":
            input_history = candidate_input_history_diagnostics.get(candidate_name)
            if input_history is None or input_history["status"] != "PIT_CHINA_EPU_HISTORY_READY":
                status = "WAITING_FOR_PIT_CHINA_EPU_SERIES"
                reasons = [
                    "input_history_status:"
                    + ("MISSING_DIAGNOSTIC" if input_history is None else input_history["status"])
                ]
            elif any(row["status"] == "INSUFFICIENT_MONTHS_DIAGNOSTIC" for row in monthly):
                status = "WAITING_FOR_COMMON_CLOSED_MONTHS"
                reasons = sorted(
                    "insufficient_common_months:" + row["existing_factor_name"]
                    for row in monthly
                    if row["status"] == "INSUFFICIENT_MONTHS_DIAGNOSTIC"
                )
            elif monthly:
                status = "ELIGIBLE_FOR_FUTURE_PREREGISTRATION"
                reasons = ["diagnostic_routes_below_dedup_threshold"]
            else:
                status = "STOP_MISSING_DEDUP_ROUTE"
                reasons = ["no_monthly_dedup_route"]
        elif (
            candidate_name == "cn_same_month_seasonality_5y"
            and seasonality_calendar_coverage["all_calendar_months_covered"] is not True
        ):
            status = "WAITING_FOR_12_STABLE_CALENDAR_MONTHS"
            reasons = [
                "missing_stable_calendar_months:"
                + ",".join(
                    str(month) for month in seasonality_calendar_coverage["missing_calendar_months"]
                )
            ]
        elif any(row["status"] == "INSUFFICIENT_MONTHS_DIAGNOSTIC" for row in monthly):
            status = "WAITING_FOR_COMMON_CLOSED_MONTHS"
            reasons = sorted(
                "insufficient_common_months:" + row["existing_factor_name"]
                for row in monthly
                if row["status"] == "INSUFFICIENT_MONTHS_DIAGNOSTIC"
            )
        elif monthly:
            status = "ELIGIBLE_FOR_FUTURE_PREREGISTRATION"
            reasons = ["diagnostic_routes_below_dedup_threshold"]
        else:
            status = "STOP_MISSING_DEDUP_ROUTE"
            reasons = ["no_monthly_dedup_route"]
        decisions.append(
            {
                "candidate_name": candidate_name,
                "status": status,
                "reasons": reasons,
                "literature_status": literature["status"],
                "locked_falsification_tests": literature["locked_falsification_tests"],
                "formal_preregistration_created": False,
                "formal_dedup_evidence": False,
                "factor_v4_authority": False,
            }
        )
    return decisions


def _future_policy_applicability(
    routing_decisions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    routing_by_name = {row["candidate_name"]: row for row in routing_decisions}
    policy_candidates = (
        "cn_fip_continuous_direction_12m",
        "cn_low_max_return_20d",
        "cn_low_total_skewness_20d",
        "cn_low_market_adjusted_tail_asymmetry_252d",
        "cn_low_left_tail_var1_250d",
    )
    if not set(policy_candidates).issubset(routing_by_name):
        raise _error("future-policy candidates are missing routing decisions")
    result: dict[str, dict[str, Any]] = {}
    for candidate_name in policy_candidates:
        decision = routing_by_name[candidate_name]
        routing_status = decision["status"]
        if routing_status == "ELIGIBLE_FOR_FUTURE_PREREGISTRATION":
            applicability = "DIAGNOSTICALLY_ELIGIBLE_DRAFT_ONLY"
        else:
            applicability = "INAPPLICABLE_" + routing_status
        result[candidate_name] = {
            "applicability": applicability,
            "routing_status": routing_status,
            "routing_reasons": copy.deepcopy(decision["reasons"]),
            "formal_preregistration_created": False,
            "factor_v4_authority": False,
        }
    return result


def _load_comparison_artifacts(
    *,
    ontology_path: Path,
    catalog_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ontology_sha = _sha256_file(ontology_path)
    catalog_sha = _sha256_file(catalog_path)
    ontology = screening.validate_primitive_ontology_v4(_read_json(ontology_path))
    catalog = screening.validate_candidate_catalog_v4(
        _read_json(catalog_path),
        ontology=ontology,
    )
    if _sha256_file(ontology_path) != ontology_sha:
        raise _error("comparison ontology changed while it was read")
    if _sha256_file(catalog_path) != catalog_sha:
        raise _error("comparison catalog changed while it was read")
    return (
        ontology,
        catalog,
        {
            "comparison_ontology_path": str(ontology_path.resolve()),
            "comparison_ontology_file_sha256": ontology_sha,
            "comparison_ontology_semantic_sha256": ontology["semantic_sha256"],
            "comparison_catalog_path": str(catalog_path.resolve()),
            "comparison_catalog_file_sha256": catalog_sha,
            "comparison_catalog_semantic_sha256": catalog["semantic_sha256"],
            "comparison_factor_count": len(catalog["candidates"]),
        },
    )


def run_diagnostic(
    *,
    market_pointer: Path,
    fundamental_pointer: Path,
    comparison_ontology: Path = DEFAULT_COMPARISON_ONTOLOGY,
    comparison_catalog: Path = DEFAULT_COMPARISON_CATALOG,
    lookback_sessions: int,
) -> dict[str, Any]:
    module_path = REPO_ROOT / "quant_investor/factors/governance_literature_incubator_v4.py"
    runner_path = REPO_ROOT / "scripts/diagnose_factor_v4_literature_incubator.py"
    exact_five_eval_path = (
        REPO_ROOT / "quant_investor/factors/governance_exact_five_no_label_eval_v4_4.py"
    )
    exact_five_prereg_path = (
        REPO_ROOT / "quant_investor/factors/governance_candidate_preregistration_v4_4.py"
    )
    module_sha = _sha256_file(module_path)
    runner_sha = _sha256_file(runner_path)
    exact_five_eval_sha = _sha256_file(exact_five_eval_path)
    exact_five_prereg_sha = _sha256_file(exact_five_prereg_path)
    (
        comparison_ontology_payload,
        comparison_catalog_payload,
        comparison_binding,
    ) = _load_comparison_artifacts(
        ontology_path=comparison_ontology,
        catalog_path=comparison_catalog,
    )
    market, pit_mask, market_binding = _load_market_inputs(
        market_pointer, lookback_sessions=lookback_sessions
    )
    standard_mask = pit_mask.iloc[-STANDARD_DIAGNOSTIC_LOOKBACK_SESSIONS:].copy()
    standard_market = {
        name: frame.reindex(
            index=standard_mask.index,
            columns=standard_mask.columns,
        )
        for name, frame in market.items()
    }
    fundamental, fundamental_binding = _load_fundamental_inputs(
        fundamental_pointer,
        sessions=standard_mask.index,
        symbols=standard_mask.columns,
        pit_mask=standard_mask,
    )
    signals = {
        "cn_earnings_yield_ex_shell_30pct": (
            incubator.earnings_yield_ex_shell_v4(
                pe=standard_market["pe"],
                total_mv=standard_market["total_mv"],
                pit_mask=standard_mask,
            )
        ),
        "cn_low_beta_252d": incubator.low_beta_v4(
            adj_close=standard_market["adj_close"],
            pit_mask=standard_mask,
        ),
        "cn_52_week_high_momentum_12m": incubator.high_52_week_momentum_v4(
            adj_close=standard_market["adj_close"],
            pit_mask=standard_mask,
        ),
        "cn_high_price_delay_d1_52w": incubator.high_price_delay_d1_v4(
            adj_close=market["adj_close"],
            total_mv=market["total_mv"],
            pit_mask=pit_mask,
        ),
        "cn_low_max_return_20d": incubator.low_max_return_v4(
            adj_close=standard_market["adj_close"],
            pit_mask=standard_mask,
        ),
        "cn_low_total_skewness_20d": incubator.low_total_skewness_v4(
            adj_close=standard_market["adj_close"],
            pit_mask=standard_mask,
        ),
        "cn_low_market_adjusted_tail_asymmetry_252d": (
            incubator.low_market_adjusted_tail_asymmetry_v4(
                adj_close=standard_market["adj_close"],
                pit_mask=standard_mask,
            )
        ),
        "cn_quality_cash_low_leverage": (
            incubator.quality_cash_low_leverage_v4(
                fin_roe=fundamental["fin_roe"],
                fin_ocf_to_profit=fundamental["fin_ocf_to_profit"],
                fin_debt_to_assets=fundamental["fin_debt_to_assets"],
                pit_mask=standard_mask,
            )
        ),
        "cn_same_month_seasonality_5y": (
            incubator.same_month_seasonality_v4(
                adj_close=market["adj_close"],
                pit_mask=pit_mask,
            )
        ),
        "cn_fip_continuous_direction_12m": (
            incubator.fip_continuous_direction_v4(
                adj_close=market["adj_close"],
                pit_mask=pit_mask,
            )
        ),
        "cn_low_left_tail_var1_250d": incubator.low_left_tail_var1_v4(
            adj_close=standard_market["adj_close"],
            pit_mask=standard_mask,
        ),
    }
    comparison_signals = _build_comparison_signals(
        market=standard_market,
        fundamental=fundamental,
        pit_mask=standard_mask,
    )
    comparison_signals.update(
        _build_momentum_comparison_signals(
            adj_close=market["adj_close"],
            pit_mask=pit_mask,
        )
    )
    protected_signals, protected_signal_binding = _build_protected_exact_five_comparison_signals(
        market=standard_market,
        pit_mask=standard_mask,
    )
    signal_name_collisions = sorted(
        (set(comparison_signals) & set(signals))
        | (set(protected_signals) & set(signals))
        | (set(protected_signals) & set(comparison_signals))
    )
    if signal_name_collisions:
        raise _error(
            "comparison/protected signal namespaces collide: " + ",".join(signal_name_collisions)
        )
    comparison_signals.update(protected_signals)
    structural_audit = incubator.build_structural_audit_v4(
        comparison_ontology=comparison_ontology_payload,
        comparison_catalog=comparison_catalog_payload,
    )
    protected_exact_five_audit = incubator.build_protected_exact_five_audit_v4(
        protected_candidates=exact_five_prereg.EXPECTED_CANDIDATE_ROWS,
    )
    monthly_dedup = _monthly_dedup_diagnostic(
        candidate_signals=signals,
        comparison_signals={**comparison_signals, **signals},
        comparison_catalog=comparison_catalog_payload,
        protected_candidates=protected_exact_five_audit["protected_candidates"],
    )
    seasonality_calendar_coverage = _seasonality_calendar_coverage_diagnostic(
        signal=signals["cn_same_month_seasonality_5y"],
        pit_mask=pit_mask,
    )
    candidate_computability = {
        name: _signal_summary(
            signal,
            pit_mask=pit_mask.reindex(
                index=signal.index,
                columns=signal.columns,
            ),
        )
        for name, signal in signals.items()
    }
    candidate_input_history_diagnostics = {
        "cn_52_week_high_momentum_12m": {
            "status": "BLOCKED_MISSING_PIT_CHINA_EPU_SERIES",
            "required_regime_input": "Baker_Bloom_Davis_China_EPU_monthly",
            "required_publication_timestamp_or_frozen_release_lag": True,
            "same_month_EPU_lookahead_permitted": False,
            "labels_loaded": False,
            "forward_returns_loaded": False,
            "factor_v4_authority": False,
        },
        "cn_high_price_delay_d1_52w": _price_delay_input_history_diagnostic(
            market=market,
            pit_mask=pit_mask,
        ),
    }
    literature_assessments = incubator.candidate_literature_assessments_v4()
    routing_decisions = _candidate_routing_decisions(
        structural_audit=structural_audit,
        protected_exact_five_audit=protected_exact_five_audit,
        monthly_dedup=monthly_dedup,
        seasonality_calendar_coverage=seasonality_calendar_coverage,
        literature_assessments=literature_assessments,
        candidate_computability=candidate_computability,
        candidate_input_history_diagnostics=candidate_input_history_diagnostics,
    )
    future_policy_applicability = _future_policy_applicability(routing_decisions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": incubator.PROTOCOL_VERSION,
        "incubator_version": incubator.INCUBATOR_VERSION,
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "as_of": market_binding["latest_trade_date"],
        "input_bindings": {
            "market": market_binding,
            "fundamental": fundamental_binding,
            "comparison_catalog": comparison_binding,
            "protected_exact_five": protected_signal_binding,
            "code": {
                "signal_module_path": str(module_path.resolve()),
                "signal_module_sha256": module_sha,
                "diagnostic_runner_path": str(runner_path.resolve()),
                "diagnostic_runner_sha256": runner_sha,
                "exact_five_evaluator_path": str(exact_five_eval_path.resolve()),
                "exact_five_evaluator_sha256": exact_five_eval_sha,
                "exact_five_preregistration_path": str(exact_five_prereg_path.resolve()),
                "exact_five_preregistration_sha256": exact_five_prereg_sha,
            },
        },
        "candidate_catalog": incubator.candidate_catalog_v4(),
        "literature_idea_catalog": incubator.literature_idea_catalog_v4(),
        "candidate_literature_assessments": literature_assessments,
        "low_max_future_preregistration_policy": (
            incubator.low_max_future_preregistration_policy_v4()
        ),
        "low_total_skewness_future_preregistration_policy": (
            incubator.low_total_skewness_future_preregistration_policy_v4()
        ),
        "tail_asymmetry_future_preregistration_policy": (
            incubator.tail_asymmetry_future_preregistration_policy_v4()
        ),
        "fip_future_preregistration_policy": (incubator.fip_future_preregistration_policy_v4()),
        "left_tail_var1_future_preregistration_policy": (
            incubator.left_tail_var1_future_preregistration_policy_v4()
        ),
        "future_policy_applicability": future_policy_applicability,
        "candidate_computability": candidate_computability,
        "candidate_input_history_diagnostics": candidate_input_history_diagnostics,
        "protected_exact_five_computability": {
            name: _signal_summary(
                signal,
                pit_mask=standard_mask,
            )
            for name, signal in protected_signals.items()
        },
        "pairwise_correlation_diagnostic": (_pairwise_correlation_diagnostic(signals)),
        "structural_dedup_audit": structural_audit,
        "protected_exact_five_audit": protected_exact_five_audit,
        "monthly_dedup_diagnostic": monthly_dedup,
        "seasonality_calendar_coverage_diagnostic": (seasonality_calendar_coverage),
        "candidate_routing_decisions": routing_decisions,
        "measurement": {
            "labels_loaded": False,
            "forward_returns_loaded": False,
            "outcome_statistics_computed": False,
            "family_bh_run": False,
            "formal_gate_results_created": False,
        },
        "authority": incubator.AUTHORITY_FLAGS,
        "side_effects": incubator.SIDE_EFFECT_FLAGS,
        "formal_blockers": [
            "future_preregistration_missing",
            "post_publication_embargo_not_started",
            "240_post_embargo_open_sessions_missing",
            "12_closed_month_ends_missing",
            "factor_v4_gates_1_to_8_not_run",
            "family_multiple_testing_not_run",
            "formal_dedup_not_run_diagnostics_only",
            "protected_exact_five_formal_dedup_not_run_diagnostics_only",
            "raw_low_beta_china_mechanism_conflict_unresolved",
            (
                "low_max_future_preregistration_missing"
                if future_policy_applicability["cn_low_max_return_20d"]["routing_status"]
                == "ELIGIBLE_FOR_FUTURE_PREREGISTRATION"
                else "low_max_draft_inapplicable_"
                + future_policy_applicability["cn_low_max_return_20d"]["routing_status"].lower()
            ),
            "price_delay_future_preregistration_policy_missing",
            "52_week_high_PIT_China_EPU_history_missing",
            "low_total_skewness_future_preregistration_missing",
            "tail_asymmetry_future_preregistration_missing",
            "same_month_seasonality_12_calendar_month_coverage_missing",
            "fip_single_score_future_preregistration_policy_missing",
            (
                "left_tail_var1_future_preregistration_missing"
                if future_policy_applicability["cn_low_left_tail_var1_250d"]["routing_status"]
                == "ELIGIBLE_FOR_FUTURE_PREREGISTRATION"
                else "left_tail_var1_draft_inapplicable_"
                + future_policy_applicability["cn_low_left_tail_var1_250d"][
                    "routing_status"
                ].lower()
            ),
            "canonical_abcd_replay_not_run",
            "candidate_admission_missing",
            "activation_receipt_missing",
        ],
    }
    report["artifact_semantic_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    stable_inputs = (
        (market_pointer, market_binding["market_pointer_sha256"]),
        (fundamental_pointer, fundamental_binding["fundamental_pointer_sha256"]),
        (
            comparison_ontology,
            comparison_binding["comparison_ontology_file_sha256"],
        ),
        (
            comparison_catalog,
            comparison_binding["comparison_catalog_file_sha256"],
        ),
        (module_path, module_sha),
        (runner_path, runner_sha),
        (exact_five_eval_path, exact_five_eval_sha),
        (exact_five_prereg_path, exact_five_prereg_sha),
    )
    for path, expected_sha in stable_inputs:
        if _sha256_file(path) != expected_sha:
            raise _error(f"bound input changed during diagnostic: {path}")
    return report


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent_stat = path.parent.stat()
    except OSError as exc:
        raise _error(f"cannot inspect output parent: {exc}") from exc
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.getuid()
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise _error("output parent must be an owner-only 0700 directory")
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise _error(f"output already exists: {path}") from exc
    except OSError as exc:
        raise _error(f"cannot publish output {path}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-pointer",
        type=Path,
        default=DEFAULT_MARKET_POINTER,
    )
    parser.add_argument(
        "--fundamental-pointer",
        type=Path,
        default=DEFAULT_FUNDAMENTAL_POINTER,
    )
    parser.add_argument(
        "--comparison-ontology",
        type=Path,
        default=DEFAULT_COMPARISON_ONTOLOGY,
    )
    parser.add_argument(
        "--comparison-catalog",
        type=Path,
        default=DEFAULT_COMPARISON_CATALOG,
    )
    parser.add_argument(
        "--lookback-sessions",
        type=int,
        default=DEFAULT_LOOKBACK_SESSIONS,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.lookback_sessions < incubator.SAME_MONTH_LOOKBACK_SESSIONS:
        raise _error("lookback must cover the exact same-month seasonality history")
    report = run_diagnostic(
        market_pointer=args.market_pointer.resolve(),
        fundamental_pointer=args.fundamental_pointer.resolve(),
        comparison_ontology=args.comparison_ontology.resolve(),
        comparison_catalog=args.comparison_catalog.resolve(),
        lookback_sessions=args.lookback_sessions,
    )
    _write_exclusive(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "as_of": report["as_of"],
                "output": str(args.output.resolve()),
                "artifact_semantic_sha256": report["artifact_semantic_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
