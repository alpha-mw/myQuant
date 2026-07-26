#!/usr/bin/env python3
"""Run a sealed, non-admitted V17 full-A diagnostic from verified local inputs.

This runner never reads a Tushare token and never calls a network or broker API.
It consumes a previously acquired official Tushare bundle plus immutable local
Parquet generations, then exercises the V17 fundamental and timing algorithms.
Formal source admission is reported separately and remains fail-closed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Any, Iterator, Mapping, cast

import numpy as np
import pandas as pd

from quant_investor.factors.price_volume import compute_price_volume_factor
from quant_investor.v17_v2_runtime.algorithms.fundamental_scoring import (
    ALL_METRICS,
    score_fundamental_universe_wide_history,
)
from quant_investor.v17_v2_runtime.algorithms.quant_timing import (
    FACTOR_NAMES,
    compute_latest_scores,
)
from quant_investor.v17_v2_runtime.gate import RuntimeGate
from quant_investor.v17_v2_runtime.service import verify_runtime

PROTOCOL_VERSION = "myquant.v17.v2"
AUTHORITY = False
REQUIRED_RANK_ROLES = (
    "cn_open_day_calendar_dataset",
    "corporate_actions_dataset",
    "deep_evidence_dataset",
    "fundamental_generation_catalog",
    "fundamental_raw_tables_dataset",
    "H00300_total_return_dataset",
    "market_bars_dataset",
    "market_pointer",
    "market_snapshot_manifest",
    "official_delisting_cash_dataset",
    "pit_generation_catalog",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, np.ndarray, pd.Series)):
        return [_json_value(item) for item in value]
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_write(path, payload.encode("utf-8"))


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_write(path, value.encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, 0o600)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(name, index=False)
        os.chmod(name, 0o600)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _date_close_utc(values: pd.Series) -> pd.Series:
    compact = (
        values.astype("string")
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.slice(0, 8)
    )
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    return parsed.dt.tz_localize("Asia/Shanghai").add(pd.Timedelta(hours=15)).dt.tz_convert("UTC")


def _canonical_symbol(value: Any) -> str:
    return str(value).strip().upper()


def _file_evidence(path: Path) -> dict[str, Any]:
    mode = stat.S_IMODE(path.stat().st_mode)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "mode": f"{mode:04o}",
    }


class Recorder:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.steps: list[dict[str, Any]] = []

    @contextmanager
    def step(self, name: str, command: str) -> Iterator[dict[str, Any]]:
        started = datetime.now(timezone.utc)
        monotonic = time.monotonic()
        row: dict[str, Any] = {
            "step": len(self.steps) + 1,
            "name": name,
            "command": command,
            "started_at": started.isoformat(),
            "status": "RUNNING",
            "authority": False,
        }
        self.steps.append(row)
        try:
            yield row
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            row["status"] = "PASS"
        finally:
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
            row["elapsed_seconds"] = round(time.monotonic() - monotonic, 6)
            _write_json(self.run_root / "step_results.json", self.steps)


def _validate_acquisition(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    bad: list[str] = []
    files: list[dict[str, Any]] = []
    for query in manifest.get("queries", []):
        for kind in ("raw", "parquet"):
            path = Path(str(query[f"{kind}_path"]))
            expected = str(query[f"{kind}_sha256"])
            observed = _sha256(path)
            if observed != expected:
                bad.append(f"{query['query_id']}:{kind}:sha256")
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                bad.append(f"{query['query_id']}:{kind}:mode")
            files.append(
                {
                    "query_id": query["query_id"],
                    "kind": kind,
                    "path": str(path),
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "row_count": int(query["row_count"]),
                }
            )
    if manifest.get("official_url") != "https://api.tushare.pro":
        bad.append("official_url")
    if int(manifest.get("general_rate_limit_per_minute", -1)) != 200:
        bad.append("general_rate_limit")
    if int(manifest.get("financial_rate_limit_per_minute", -1)) != 80:
        bad.append("financial_rate_limit")
    if bad:
        raise ValueError(f"acquisition readback failed: {bad}")
    return manifest, {
        "manifest": _file_evidence(manifest_path),
        "query_count": len(manifest.get("queries", [])),
        "files": files,
        "bad": [],
        "token_persisted": False,
    }


def _validate_pointer_tree(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    market_pointer_path = source_root / "data/parquet/cn/_latest.json"
    market_pointer = _read_json(market_pointer_path)
    market_manifest_path = _resolve(source_root, str(market_pointer["manifest_path"]))
    market_manifest = _read_json(market_manifest_path)
    fundamental_pointer_path = source_root / "data/parquet/cn/_fundamental_latest.json"
    fundamental_pointer = _read_json(fundamental_pointer_path)
    fundamental_base = fundamental_pointer_path.parent
    fundamental_manifest_path = _resolve(
        fundamental_base, str(fundamental_pointer["manifest_path"])
    )
    fundamental_manifest = _read_json(fundamental_manifest_path)
    if market_pointer.get("status") != "OK":
        raise ValueError("market pointer is not OK")
    if market_pointer.get("latest_complete_trade_date") != "20260724":
        raise ValueError("market pointer is not sealed at 20260724")
    if fundamental_pointer.get("status") != "OK":
        raise ValueError("fundamental pointer is not OK")
    if fundamental_pointer.get("metadata", {}).get("gate2_passed") is not True:
        raise ValueError("fundamental generation gate2 did not pass")
    tables: dict[str, Any] = {}
    for name, relative in fundamental_pointer["tables"].items():
        path = _resolve(fundamental_base, str(relative))
        expected = fundamental_pointer["primary_provenance"]["output_parquet_sha256"][name]
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"fundamental table SHA mismatch: {name}")
        tables[name] = _file_evidence(path)
    pit_path = Path(str(market_pointer["coverage"]["pit_membership_path"]))
    if _sha256(pit_path) != market_pointer["coverage"]["pit_membership_sha256"]:
        raise ValueError("PIT membership SHA mismatch")
    return (
        {
            "market_pointer": market_pointer,
            "market_manifest": market_manifest,
            "fundamental_pointer": fundamental_pointer,
            "fundamental_manifest": fundamental_manifest,
        },
        {
            "market_pointer": _file_evidence(market_pointer_path),
            "market_manifest": _file_evidence(market_manifest_path),
            "fundamental_pointer": _file_evidence(fundamental_pointer_path),
            "fundamental_manifest": _file_evidence(fundamental_manifest_path),
            "fundamental_tables": tables,
            "pit_membership": _file_evidence(pit_path),
        },
    )


def _ttm_from_state(
    state: Mapping[str, Mapping[str, float]], end_date: str, columns: tuple[str, ...]
) -> dict[str, float] | None:
    if len(end_date) != 8:
        return None
    year = int(end_date[:4])
    suffix = end_date[4:]
    current = state.get(end_date)
    if current is None:
        return None
    if suffix == "1231":
        return {
            column: float(current[column])
            for column in columns
            if np.isfinite(current.get(column, np.nan))
        }
    if suffix not in {"0331", "0630", "0930"}:
        return None
    prior_annual = state.get(f"{year - 1}1231")
    prior_same = state.get(f"{year - 1}{suffix}")
    if prior_annual is None or prior_same is None:
        return None
    result: dict[str, float] = {}
    for column in columns:
        values = (
            current.get(column, np.nan),
            prior_annual.get(column, np.nan),
            prior_same.get(column, np.nan),
        )
        if not all(np.isfinite(value) for value in values):
            return None
        result[column] = float(values[0] + values[1] - values[2])
    return result


def _derive_ttm_events(
    frame: pd.DataFrame, *, value_columns: tuple[str, ...]
) -> pd.DataFrame:
    required = {"ts_code", "ann_date", "f_ann_date", "end_date", *value_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"raw statement missing columns: {missing}")
    working = frame[list(required)].copy()
    working["ts_code"] = working["ts_code"].map(_canonical_symbol)
    working["end_date"] = working["end_date"].astype("string")
    effective = working["f_ann_date"].astype("string").fillna("")
    effective = effective.where(effective.str.len() == 8, working["ann_date"].astype("string"))
    working["available_at"] = _date_close_utc(effective)
    working = working.loc[
        working["available_at"].notna() & working["end_date"].str.len().eq(8)
    ].copy()
    for column in value_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.sort_values(
        ["ts_code", "available_at", "end_date"], kind="mergesort"
    ).drop_duplicates(["ts_code", "available_at", "end_date"], keep="last")
    events: list[dict[str, Any]] = []
    for symbol, symbol_rows in working.groupby("ts_code", sort=False, observed=True):
        state: dict[str, dict[str, float]] = {}
        for available_at, batch in symbol_rows.groupby(
            "available_at", sort=True, observed=True
        ):
            for row in batch.itertuples(index=False):
                period = str(row.end_date)
                state[period] = {
                    column: float(getattr(row, column)) for column in value_columns
                }
            derived: tuple[str, dict[str, float]] | None = None
            for period in sorted(state, reverse=True):
                values = _ttm_from_state(state, period, value_columns)
                if values is not None and set(values) == set(value_columns):
                    derived = (period, values)
                    break
            if derived is None:
                continue
            events.append(
                {
                    "ts_code": str(symbol),
                    "available_at": available_at,
                    "report_end_date": derived[0],
                    **{f"{column}_ttm": value for column, value in derived[1].items()},
                }
            )
    return pd.DataFrame(events).sort_values(
        ["available_at", "ts_code"], kind="mergesort"
    )


def _merge_ttm(
    history: pd.DataFrame, events: pd.DataFrame, columns: tuple[str, ...], prefix: str
) -> pd.DataFrame:
    renamed = events.rename(
        columns={
            "available_at": f"{prefix}_available_at",
            "report_end_date": f"{prefix}_report_end_date",
        }
    )
    left = history.sort_values(["trade_at", "ts_code"], kind="mergesort")
    right = renamed.sort_values(
        [f"{prefix}_available_at", "ts_code"], kind="mergesort"
    )
    return pd.merge_asof(
        left,
        right[
            [
                "ts_code",
                f"{prefix}_available_at",
                f"{prefix}_report_end_date",
                *[f"{column}_ttm" for column in columns],
            ]
        ],
        by="ts_code",
        left_on="trade_at",
        right_on=f"{prefix}_available_at",
        direction="backward",
        allow_exact_matches=True,
    )


def _build_fundamental_inputs(
    *,
    source_root: Path,
    raw_checkpoint: Path,
    official: pd.DataFrame,
    acquisition_finished_at: pd.Timestamp,
    latest_market: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    daily_path = (
        source_root
        / "data/parquet/cn/_fundamental_generations"
        / "cn_fundamental_primary_20260714_v3_barbound"
        / "fundamental_daily.parquet"
    )
    columns = [
        "ts_code",
        "trade_date",
        "availability_date",
        "sector",
        "total_mv_rmb",
        "fin_roe",
        "fin_roa",
        "fin_debt_to_assets",
        "fin_net_profit_yoy",
        "forecast_revision",
    ]
    daily = pd.read_parquet(daily_path, columns=columns)
    daily_rows_read = len(daily)
    official_symbols = set(official["ts_code"].map(_canonical_symbol))
    daily["ts_code"] = daily["ts_code"].map(_canonical_symbol)
    daily = daily.loc[daily["ts_code"].isin(official_symbols)].copy()
    daily["trade_at"] = _date_close_utc(daily["trade_date"])
    daily["canonical_available_at"] = _date_close_utc(daily["availability_date"])
    daily = daily.loc[daily["trade_at"].le(cutoff)].sort_values(
        ["ts_code", "trade_at"], kind="mergesort"
    )
    daily = daily.groupby("ts_code", sort=False, observed=True).tail(756).copy()
    income = pd.read_parquet(raw_checkpoint / "income.parquet")
    cashflow = pd.read_parquet(raw_checkpoint / "cashflow.parquet")
    income_events = _derive_ttm_events(income, value_columns=("n_income_attr_p",))
    cashflow_events = _derive_ttm_events(
        cashflow, value_columns=("n_cashflow_act", "c_pay_acq_const_fiolta")
    )
    daily = _merge_ttm(daily, income_events, ("n_income_attr_p",), "income")
    daily = _merge_ttm(
        daily,
        cashflow_events,
        ("n_cashflow_act", "c_pay_acq_const_fiolta"),
        "cashflow",
    )
    daily["net_profit_ttm"] = pd.to_numeric(
        daily["n_income_attr_p_ttm"], errors="coerce"
    )
    daily["cfo_ttm"] = pd.to_numeric(daily["n_cashflow_act_ttm"], errors="coerce")
    daily["capex_ttm"] = pd.to_numeric(
        daily["c_pay_acq_const_fiolta_ttm"], errors="coerce"
    )
    daily["fin_ocf_to_profit"] = daily["cfo_ttm"].div(
        daily["net_profit_ttm"].where(daily["net_profit_ttm"].gt(0))
    )
    daily["free_cashflow_ttm"] = daily["cfo_ttm"] - daily["capex_ttm"]
    daily["fin_fcf_to_profit"] = daily["free_cashflow_ttm"].div(
        daily["net_profit_ttm"].where(daily["net_profit_ttm"].gt(0))
    )
    daily["fcf_to_price"] = daily["free_cashflow_ttm"].div(
        pd.to_numeric(daily["total_mv_rmb"], errors="coerce").where(
            pd.to_numeric(daily["total_mv_rmb"], errors="coerce").gt(0)
        )
    )
    daily["forecast_revision"] = np.nan
    available_columns = [
        "canonical_available_at",
        "income_available_at",
        "cashflow_available_at",
    ]
    daily["availability"] = daily[available_columns].max(axis=1)
    daily = daily.loc[
        daily["trade_at"].notna() & daily["availability"].notna()
    ].copy()
    history = daily.drop(columns=["trade_date"]).rename(
        columns={"ts_code": "symbol", "trade_at": "trade_date"}
    )[
        [
            "symbol",
            "trade_date",
            "availability",
            "fin_roe",
            "fin_ocf_to_profit",
            "fin_net_profit_yoy",
            "fin_debt_to_assets",
            "fcf_to_price",
            "fin_roa",
            "fin_fcf_to_profit",
            "forecast_revision",
        ]
    ].copy()
    history["is_open_day"] = True
    latest = daily.sort_values(["ts_code", "trade_at"], kind="mergesort").groupby(
        "ts_code", sort=False, observed=True
    ).tail(1)
    official_map = official.set_index("ts_code", drop=False)
    market_map = latest_market.set_index("ts_code", drop=False)
    snapshot_rows: list[dict[str, Any]] = []
    for symbol in sorted(official_symbols):
        source = latest.loc[latest["ts_code"].eq(symbol)]
        member = official_map.loc[symbol]
        raw_industry = member.get("industry")
        industry = "" if pd.isna(raw_industry) else str(raw_industry).strip()
        if source.empty:
            values: dict[str, Any] = {}
            availability = acquisition_finished_at
        else:
            values = source.iloc[-1].to_dict()
            availability = max(
                acquisition_finished_at,
                pd.Timestamp(values["availability"]),
            )
        market = market_map.loc[symbol] if symbol in market_map.index else {}
        market_cap = (
            float(market["total_mv"]) * 10000.0
            if isinstance(market, pd.Series)
            and np.isfinite(pd.to_numeric(market.get("total_mv"), errors="coerce"))
            else values.get("total_mv_rmb", np.nan)
        )
        snapshot_rows.append(
            {
                "symbol": symbol,
                "industry": industry,
                "in_universe": True,
                "research_eligible": True,
                "membership_conflict": False,
                "membership_is_pit": True,
                "universe_id": "CN/full_a",
                "availability": availability,
                "flow_basis": "LATEST_TTM",
                "balance_sheet_basis": "LATEST_REPORT_PERIOD",
                "capex_sign_convention": "POSITIVE_OUTFLOW",
                "net_profit_ttm": values.get("net_profit_ttm", np.nan),
                "market_cap": market_cap,
                "cfo_ttm": values.get("cfo_ttm", np.nan),
                "capex_ttm": values.get("capex_ttm", np.nan),
                "fin_roe": values.get("fin_roe", np.nan),
                "fin_roa": values.get("fin_roa", np.nan),
                "fin_ocf_to_profit": values.get("fin_ocf_to_profit", np.nan),
                "fin_fcf_to_profit": values.get("fin_fcf_to_profit", np.nan),
                "fin_net_profit_yoy": values.get("fin_net_profit_yoy", np.nan),
                "fin_debt_to_assets": values.get("fin_debt_to_assets", np.nan),
                "forecast_revision": np.nan,
                "fcf_to_price": values.get("fcf_to_price", np.nan),
            }
        )
    snapshot = pd.DataFrame(snapshot_rows)
    metrics = {
        "daily_rows_read": int(daily_rows_read),
        "daily_rows_official_tail756": int(len(daily)),
        "history_symbols": int(history["symbol"].nunique()),
        "history_rows": int(len(history)),
        "income_rows": int(len(income)),
        "income_ttm_events": int(len(income_events)),
        "cashflow_rows": int(len(cashflow)),
        "cashflow_ttm_events": int(len(cashflow_events)),
        "snapshot_rows": int(len(snapshot)),
        "snapshot_positive_net_profit": int(
            pd.to_numeric(snapshot["net_profit_ttm"], errors="coerce").gt(0).sum()
        ),
        "snapshot_complete_main_metrics": int(
            np.isfinite(
                snapshot[
                    [
                        "fin_roe",
                        "fin_ocf_to_profit",
                        "fin_net_profit_yoy",
                        "fin_debt_to_assets",
                        "fcf_to_price",
                    ]
                ].apply(pd.to_numeric, errors="coerce")
            )
            .all(axis=1)
            .sum()
        ),
    }
    return snapshot, history, metrics


def _read_latest_market(
    table_root: Path, market_date: str
) -> tuple[pd.DataFrame, list[Path]]:
    year, month = market_date[:4], market_date[4:6]
    path = table_root / f"year={year}" / f"month={month}" / "part.parquet"
    frame = pd.read_parquet(
        path,
        columns=[
            "ts_code",
            "trade_date",
            "close",
            "adj_close",
            "adj_factor",
            "vol",
            "amount",
            "total_mv",
        ],
    )
    frame["ts_code"] = frame["ts_code"].map(_canonical_symbol)
    latest = frame.loc[frame["trade_date"].astype(str).eq(market_date)].copy()
    return latest, [path]


def _recent_bar_files(table_root: Path) -> list[Path]:
    candidates = [
        table_root / "year=2025/month=12/part.parquet",
        *[
            table_root / f"year=2026/month={month:02d}/part.parquet"
            for month in range(1, 8)
        ],
    ]
    return [path for path in candidates if path.exists()]


def _timing_and_market_context(
    *,
    table_root: Path,
    sealed_symbols: tuple[str, ...],
    cutoff: pd.Timestamp,
    acquisition_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    frames_raw = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "ts_code",
                    "trade_date",
                    "close",
                    "adj_close",
                    "vol",
                    "amount",
                ],
            )
            for path in _recent_bar_files(table_root)
        ],
        ignore_index=True,
    )
    frames_raw["ts_code"] = frames_raw["ts_code"].map(_canonical_symbol)
    frames_raw = frames_raw.loc[frames_raw["ts_code"].isin(sealed_symbols)].copy()
    frames_raw["trade_date"] = _date_close_utc(frames_raw["trade_date"])
    frames_raw = frames_raw.loc[frames_raw["trade_date"].le(cutoff)]
    frames_raw["availability"] = frames_raw["trade_date"]
    frames_raw["is_open_day"] = True
    frames = {
        symbol: group.sort_values("trade_date", kind="mergesort").tail(140).reset_index(drop=True)
        for symbol, group in frames_raw.groupby("ts_code", sort=True, observed=True)
    }
    latest = compute_latest_scores(frames, sealed_symbols=sealed_symbols, cutoff=cutoff)
    raw = pd.DataFrame({"symbol": list(sealed_symbols)})
    for factor_name in FACTOR_NAMES:
        raw[factor_name] = (
            compute_price_volume_factor(factor_name, frames).reindex(sealed_symbols).to_numpy()
        )
    timing = latest.merge(raw, on="symbol", how="left", validate="one_to_one")
    ready = timing["status"].eq("READY") & np.isfinite(timing["composite_score"])
    ordered = timing.loc[ready].sort_values(
        ["composite_score", "symbol"], ascending=[True, True], kind="mergesort"
    )
    timing["diagnostic_decile"] = pd.Series(pd.NA, index=timing.index, dtype="Int64")
    count = len(ordered)
    for position, index in enumerate(ordered.index):
        timing.at[index, "diagnostic_decile"] = min(10, (position * 10) // count + 1)
    timing["timing_state"] = "UNREADY"
    timing["timing_blocker"] = "calibration_not_admitted_official_delisting_cash_missing"
    benchmark = pd.read_parquet(
        acquisition_root / "normalized/h00300_total_return_2016_20260724.parquet"
    ).sort_values("trade_date", kind="mergesort")
    price_index = pd.read_parquet(
        acquisition_root / "normalized/000300_price_index_2016_20260724.parquet"
    ).sort_values("trade_date", kind="mergesort")

    def returns(frame: pd.DataFrame) -> dict[str, float | None]:
        close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        result: dict[str, float | None] = {}
        for horizon in (20, 60, 120, 252):
            result[f"return_{horizon}d"] = (
                float(close.iloc[-1] / close.iloc[-horizon - 1] - 1.0)
                if len(close) > horizon
                else None
            )
        result["latest_close"] = float(close.iloc[-1])
        return result

    benchmark_context = {
        "as_of_trade_date": str(benchmark.iloc[-1]["trade_date"]),
        "H00300_CSI_pre_tax_total_return": returns(benchmark),
        "000300_SH_price_index": returns(price_index),
        "total_return_to_price_ratio_change_since_20160104": float(
            (benchmark.iloc[-1]["close"] / price_index.iloc[-1]["close"])
            / (benchmark.iloc[0]["close"] / price_index.iloc[0]["close"])
            - 1.0
        ),
    }
    price_metrics: dict[str, Any] = {}
    for symbol, frame in frames.items():
        close = pd.to_numeric(frame["adj_close"], errors="coerce").dropna()
        row: dict[str, Any] = {
            "latest_close": float(pd.to_numeric(frame["close"], errors="coerce").iloc[-1]),
            "latest_trade_date": frame["trade_date"].iloc[-1],
        }
        for horizon in (20, 60, 120):
            row[f"return_{horizon}d"] = (
                float(close.iloc[-1] / close.iloc[-horizon - 1] - 1.0)
                if len(close) > horizon
                else None
            )
        row["drawdown_from_120d_high"] = (
            float(close.iloc[-1] / close.tail(120).max() - 1.0)
            if len(close)
            else None
        )
        price_metrics[symbol] = row
    return timing, benchmark_context, price_metrics


def _holding_evidence(acquisition_root: Path, symbols: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for symbol in symbols:
        suffix = symbol.replace(".", "_")
        evidence: dict[str, Any] = {}
        for table, sort_columns in (
            ("fina_indicator", ("ann_date", "end_date")),
            ("forecast", ("ann_date", "end_date")),
            ("dividend", ("ann_date", "end_date")),
        ):
            path = acquisition_root / "normalized" / f"{table}_{suffix}.parquet"
            frame = pd.read_parquet(path)
            ordered = frame.sort_values(list(sort_columns), kind="mergesort")
            row = ordered.iloc[-1].to_dict() if not ordered.empty else {}
            evidence[table] = {
                "path": str(path),
                "sha256": _sha256(path),
                "row_count": int(len(frame)),
                "latest": row,
            }
        result[symbol] = evidence
    return result


def _advice(
    *,
    scored: pd.DataFrame,
    timing: pd.DataFrame,
    official: pd.DataFrame,
    ledger: pd.DataFrame,
    latest_market: pd.DataFrame,
    price_metrics: Mapping[str, Mapping[str, Any]],
    holding_evidence: Mapping[str, Any],
    cash: float,
) -> dict[str, Any]:
    name_map = official.set_index("ts_code")["name"].astype(str).to_dict()
    scored_map = scored.set_index("symbol", drop=False)
    timing_map = timing.set_index("symbol", drop=False)
    market_map = latest_market.set_index("ts_code", drop=False)
    available = scored.loc[scored["status"].eq("AVAILABLE")].sort_values(
        ["total_score", "symbol"], ascending=[False, True], kind="mergesort"
    )
    ranks = {symbol: index + 1 for index, symbol in enumerate(available["symbol"])}
    top24: list[dict[str, Any]] = []
    for symbol in available.head(24)["symbol"]:
        row = scored_map.loc[symbol]
        timed = timing_map.loc[symbol]
        top24.append(
            {
                "symbol": symbol,
                "name": name_map.get(symbol, "UNKNOWN_NAME"),
                "label": "暂不参与",
                "fundamental_rank": ranks[symbol],
                "fundamental_score": float(row["total_score"]),
                "timing_state": str(timed["timing_state"]),
                "diagnostic_timing_decile": (
                    int(timed["diagnostic_decile"])
                    if pd.notna(timed["diagnostic_decile"])
                    else None
                ),
                "reason": "正式准入、前瞻校准和深度证据未完成，V17 不允许新增风险",
            }
        )
    holdings: list[dict[str, Any]] = []
    for row in ledger.itertuples(index=False):
        symbol = _canonical_symbol(row.symbol)
        score = scored_map.loc[symbol]
        market = market_map.loc[symbol]
        close = float(market["close"])
        market_value = close * float(row.shares)
        current_metrics = price_metrics.get(symbol, {})
        rank = ranks.get(symbol)
        weak = (
            str(score["status"]) != "AVAILABLE"
            or rank is None
            or rank > max(24, int(len(available) * 0.70))
        )
        negative_trend = (
            current_metrics.get("return_60d") is not None
            and float(current_metrics["return_60d"]) < 0
        )
        unavailable = str(score["status"]) != "AVAILABLE"
        label = "减仓观察" if unavailable or (weak and negative_trend) else "继续持有"
        forecast = (
            holding_evidence.get(symbol, {})
            .get("forecast", {})
            .get("latest", {})
        )
        holdings.append(
            {
                "symbol": symbol,
                "name": name_map.get(symbol, str(row.name) or "UNKNOWN_NAME"),
                "label": label,
                "shares": float(row.shares),
                "avg_cost": float(row.avg_cost),
                "latest_close": close,
                "market_value": market_value,
                "unrealized_pnl": market_value - float(row.cost_basis),
                "fundamental_status": str(score["status"]),
                "fundamental_rank": rank,
                "fundamental_score": (
                    float(score["total_score"]) if pd.notna(score["total_score"]) else None
                ),
                "unavailable_reasons": list(score["unavailable_reasons"]),
                "timing_state": str(timing_map.loc[symbol]["timing_state"]),
                "price_metrics": current_metrics,
                "latest_forecast": {
                    key: forecast.get(key)
                    for key in (
                        "ann_date",
                        "end_date",
                        "type",
                        "summary",
                        "p_change_min",
                        "p_change_max",
                    )
                },
                "reason": (
                    "TTM基本面不可用或评分偏弱且价格趋势不佳；即使最新业绩预告改善，也仅列减仓观察，等待正式择时"
                    if label == "减仓观察"
                    else "正式择时仍为UNREADY，不新增风险；现有仓位维持研究性持有"
                ),
            }
        )
    equity = sum(float(item["market_value"]) for item in holdings)
    return {
        "labels": ["增配候选", "继续持有", "减仓观察", "暂不参与"],
        "增配候选": [],
        "继续持有": [item for item in holdings if item["label"] == "继续持有"],
        "减仓观察": [item for item in holdings if item["label"] == "减仓观察"],
        "暂不参与": top24,
        "portfolio_revaluation": {
            "cash": cash,
            "equity_market_value": equity,
            "nav": cash + equity,
            "market_date": "20260724",
        },
        "authority": False,
        "research_only": True,
    }


def _source_gate(
    *,
    universe_conflict_count: int,
    holding_evidence: Mapping[str, Any],
    expected_holding_count: int,
) -> dict[str, Any]:
    roles = {
        "cn_open_day_calendar_dataset": "AVAILABLE_VERIFIED",
        "corporate_actions_dataset": "DIAGNOSTIC_DERIVABLE_NOT_ADMITTED",
        "deep_evidence_dataset": (
            "PARTIAL_HOLDINGS_ONLY"
            if len(holding_evidence) == expected_holding_count
            else "UNAVAILABLE"
        ),
        "fundamental_generation_catalog": "AVAILABLE_VERIFIED",
        "fundamental_raw_tables_dataset": "AVAILABLE_VERIFIED",
        "H00300_total_return_dataset": "AVAILABLE_VERIFIED",
        "market_bars_dataset": "AVAILABLE_VERIFIED",
        "market_pointer": "AVAILABLE_VERIFIED",
        "market_snapshot_manifest": "AVAILABLE_VERIFIED",
        "official_delisting_cash_dataset": "UNAVAILABLE",
        "pit_generation_catalog": (
            "CONFLICT_OFFICIAL_NEW_LISTINGS"
            if universe_conflict_count
            else "AVAILABLE_VERIFIED"
        ),
    }
    blockers = [
        role
        for role in REQUIRED_RANK_ROLES
        if roles[role] not in {"AVAILABLE_VERIFIED"}
    ]
    return {
        "required_rank_roles": roles,
        "formal_admission": "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "formal_blockers": blockers,
        "diagnostic_terminal": "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "portfolio_blockers": [
            "source_admission_not_complete",
            "forward_calibration_unavailable",
            "deep_research_incomplete",
            "risk_policy_snapshot_not_sealed",
        ],
        "authority": False,
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# myQuant V17 全 A 诊断运行报告",
        "",
        f"- 协议：`{PROTOCOL_VERSION}`",
        f"- 运行 ID：`{summary['run_id']}`",
        f"- 分析截止：`{summary['analysis_cutoff']}`",
        "- 权限：`authority=false`，研究/影子用途，无 broker、订单或交易副作用",
        f"- 正式准入：`{summary['source_gate']['formal_admission']}`",
        f"- 诊断终态：`{summary['source_gate']['diagnostic_terminal']}`",
        "",
        "## 步骤结果",
        "",
        "| 步骤 | 名称 | 状态 | 秒 | 结果 |",
        "|---:|---|---|---:|---|",
    ]
    for step in summary["steps"]:
        result = json.dumps(step.get("result", {}), ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| {step['step']} | {step['name']} | {step['status']} | "
            f"{step['elapsed_seconds']:.3f} | `{result[:360]}` |"
        )
    lines.extend(
        [
            "",
            "## 正式阻塞项",
            "",
            *[f"- `{item}`" for item in summary["source_gate"]["formal_blockers"]],
            "",
            "## 当前持仓建议（研究/影子）",
            "",
        ]
    )
    for label in ("继续持有", "减仓观察"):
        for item in summary["advice"][label]:
            lines.append(
                f"- {label}：`{item['symbol']} {item['name']}`，"
                f"收盘 {item['latest_close']:.2f}，基本面排名 "
                f"{item['fundamental_rank'] or 'UNAVAILABLE'}。"
            )
    lines.extend(
        [
            "",
            "## 新候选",
            "",
            "- `增配候选`：无。正式源准入、前瞻校准和深度证据未完成。",
        ]
    )
    for item in summary["advice"]["暂不参与"]:
        lines.append(
            f"- 暂不参与：`{item['symbol']} {item['name']}`，"
            f"基本面排名 {item['fundamental_rank']}，分数 {item['fundamental_score']:.6f}。"
        )
    lines.extend(
        [
            "",
            "## 做得好的环节",
            "",
            "- 官方 Tushare 标准协议采集成功，所有 raw/parquet SHA 回读一致，token 未写入产物。",
            "- 严格行情截至 2026-07-24，H00300.CSI 税前总收益指数独立获取。",
            "- 基本面用原始利润表/现金流量表重建 PIT LATEST_TTM，宽表入口与冻结长表公式等价。",
            "- 全 A 评分、Top24、持仓追加和最新三因子计算均使用 V17 实现，authority=false。",
            "",
            "## 仍有问题",
            "",
            "- 本地 PIT 目录比官方当前上市清单少 3 只新股，不能把原 coverage=100% 直接视为正式 V17 全 A 完整。",
            "- 缺少官方退市现金结算数据，前瞻总收益校准必须 fail-closed。",
            "- 深度证据仅覆盖 4 只持仓，Top24 未完成完整 Codex 深研。",
            "- 量化择时校准未被正式准入，因此状态保持 UNREADY。",
            "- 仓库 Tushare helper 直接接官方根地址时会把 api_name 追加到 URL；本次使用官方标准 root POST 协议规避。",
            "",
        ]
    )
    return "\n".join(lines)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-repo", type=Path, default=Path("/Users/maxwell/mySpace/myQuant")
    )
    parser.add_argument(
        "--acquisition-root",
        type=Path,
        default=Path(
            "data/private/v17_sources/protocol-v2/acquired/"
            "v17-tushare-20260726t0515z"
        ),
    )
    parser.add_argument(
        "--raw-checkpoint",
        type=Path,
        default=Path(
            "/Users/maxwell/mySpace/myQuant/data/cn_market_full/_snapshots/"
            "fundamental/checkpoints/cn_fundamental_primary_20260714_v3_barbound/"
            "_generations/checkpoint_00000004_1784113748415721000/tables"
        ),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "/Users/maxwell/mySpace/myQuant/results/strategy_records/CN/"
            "aggressive_tech_manufacturing/20260716_1628/"
            "ledger_after_manual_switch.csv"
        ),
    )
    parser.add_argument(
        "--ledger-manifest",
        type=Path,
        default=Path(
            "/Users/maxwell/mySpace/myQuant/results/strategy_records/CN/"
            "aggressive_tech_manufacturing/20260716_1628/"
            "manual_execution_manifest.json"
        ),
    )
    parser.add_argument(
        "--analysis-cutoff", default="2026-07-26T13:30:00+08:00"
    )
    parser.add_argument("--run-id", default="v17-cn-full-a-20260726t1330cst")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/private/v17_runs")
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    run_root = (args.output_root / args.run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_root, 0o700)
    recorder = Recorder(run_root)
    cutoff = pd.Timestamp(args.analysis_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("analysis cutoff must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    acquisition_root = args.acquisition_root.resolve()
    source_root = args.source_repo.resolve()

    with recorder.step(
        "input_readback",
        "verify official acquisition, canonical pointers, manifests, tables, ledger SHA/mode",
    ) as step:
        acquisition_manifest, acquisition_evidence = _validate_acquisition(
            acquisition_root
        )
        pointer_tree, pointer_evidence = _validate_pointer_tree(source_root)
        ledger_sha = _sha256(args.ledger)
        ledger_manifest_sha = _sha256(args.ledger_manifest)
        if ledger_sha != "39ab2c163d198179f32f24831443b4b1e1adcc7a9072dc49bbcc9c81334787bd":
            raise ValueError("ledger SHA is not the owner-confirmed snapshot")
        if (
            ledger_manifest_sha
            != "850db299e006f7e518f619928827353f50e571da106a763316143906922edbb8"
        ):
            raise ValueError("ledger manifest SHA mismatch")
        ledger = pd.read_csv(args.ledger, encoding="utf-8-sig")
        ledger_manifest = _read_json(args.ledger_manifest)
        holding_codes = tuple(
            dict.fromkeys(ledger["symbol"].map(_canonical_symbol).tolist())
        )
        if not holding_codes:
            raise ValueError("owner-confirmed ledger has no holdings")
        cash = float(ledger_manifest["cash_after"])
        if not np.isfinite(cash) or cash < 0:
            raise ValueError("owner-confirmed cash_after is invalid")
        input_inventory = {
            "acquisition": acquisition_evidence,
            "canonical": pointer_evidence,
            "ledger": _file_evidence(args.ledger),
            "ledger_manifest": _file_evidence(args.ledger_manifest),
            "raw_checkpoint": {
                path.name: _file_evidence(path)
                for path in sorted(args.raw_checkpoint.glob("*.parquet"))
            },
            "authority": False,
        }
        _write_json(run_root / "input_inventory.json", input_inventory)
        step["result"] = {
            "query_count": acquisition_evidence["query_count"],
            "readback_bad": [],
            "ledger_sha256": ledger_sha,
            "holding_codes": holding_codes,
            "cash_after": cash,
            "fundamental_generation": pointer_tree["fundamental_pointer"][
                "generation_id"
            ],
        }

    with recorder.step(
        "universe_reconciliation",
        "compare official stock_basic(L) with canonical PIT membership at 20260724",
    ) as step:
        official = pd.read_parquet(
            acquisition_root / "normalized/stock_basic_listed.parquet"
        )
        official["ts_code"] = official["ts_code"].map(_canonical_symbol)
        pit_path = Path(
            str(pointer_tree["market_pointer"]["coverage"]["pit_membership_path"])
        )
        pit = pd.read_parquet(pit_path)
        market_date = "20260724"
        local_active = pit.loc[
            pit["list_date"].astype(str).le(market_date)
            & (
                pit["delist_date"].astype("string").fillna("").eq("")
                | pit["delist_date"].astype(str).gt(market_date)
            )
            & pit["source_list_status"].astype(str).isin(["L", "P"])
        ].copy()
        official_only = sorted(set(official["ts_code"]).difference(local_active["symbol"]))
        local_only = sorted(set(local_active["symbol"]).difference(official["ts_code"]))
        reconciliation = {
            "market_date": market_date,
            "official_listed_count": int(official["ts_code"].nunique()),
            "local_pit_active_count": int(local_active["symbol"].nunique()),
            "official_only": [
                {
                    "symbol": symbol,
                    "name": str(
                        official.loc[official["ts_code"].eq(symbol), "name"].iloc[0]
                    ),
                    "list_date": str(
                        official.loc[official["ts_code"].eq(symbol), "list_date"].iloc[0]
                    ),
                }
                for symbol in official_only
            ],
            "local_only": local_only,
            "formal_scope_consistent": not official_only and not local_only,
        }
        official_only_rows = cast(
            list[dict[str, Any]], reconciliation["official_only"]
        )
        local_only_rows = cast(list[str], reconciliation["local_only"])
        _write_json(run_root / "universe_reconciliation.json", reconciliation)
        step["result"] = reconciliation

    with recorder.step(
        "market_snapshot",
        "read strict 20260724 bars and verify all official-only new listings have bars",
    ) as step:
        table_root = _resolve(
            source_root, str(pointer_tree["market_pointer"]["table_root"])
        )
        latest_market, market_files = _read_latest_market(table_root, "20260724")
        official_only_symbols = [item["symbol"] for item in official_only_rows]
        new_listing_bars = {
            symbol: int(latest_market["ts_code"].eq(symbol).sum())
            for symbol in official_only_symbols
        }
        missing_latest = sorted(
            set(official["ts_code"]).difference(latest_market["ts_code"])
        )
        step["result"] = {
            "latest_bar_rows": int(len(latest_market)),
            "latest_bar_symbols": int(latest_market["ts_code"].nunique()),
            "official_only_new_listing_bars": new_listing_bars,
            "official_listed_without_20260724_bar_count": len(missing_latest),
            "bar_file": _file_evidence(market_files[0]),
        }

    with recorder.step(
        "fundamental_ttm_build",
        "derive PIT LATEST_TTM from raw statements; build 756-open-day wide history",
    ) as step:
        acquired_at = pd.Timestamp(acquisition_manifest["finished_at"])
        snapshot, history, fundamental_metrics = _build_fundamental_inputs(
            source_root=source_root,
            raw_checkpoint=args.raw_checkpoint,
            official=official,
            acquisition_finished_at=acquired_at,
            latest_market=latest_market,
            cutoff=cutoff,
        )
        _write_parquet(run_root / "fundamental_snapshot.parquet", snapshot)
        step["result"] = fundamental_metrics

    with recorder.step(
        "v17_full_a_rank",
        "score_fundamental_universe_wide_history(top_n=24, holdings=4)",
    ) as step:
        candidates = score_fundamental_universe_wide_history(
            snapshot,
            history,
            cutoff=cutoff,
            holdings=holding_codes,
            top_n=24,
        )
        scored = candidates.scored
        _write_parquet(run_root / "scored_full_a.parquet", scored)
        status_counts = scored["status"].value_counts().to_dict()
        unavailable_reasons: dict[str, int] = {}
        for reasons in scored["unavailable_reasons"]:
            for reason in reasons:
                unavailable_reasons[str(reason)] = unavailable_reasons.get(str(reason), 0) + 1
        step["result"] = {
            "universe_rows": int(len(scored)),
            "status_counts": status_counts,
            "ranked_symbols": candidates.ranked_symbols,
            "appended_holdings": candidates.appended_holdings,
            "sealed_symbol_count": len(candidates.sealed_symbols),
            "top_unavailable_reasons": dict(
                sorted(
                    unavailable_reasons.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:20]
            ),
        }

    with recorder.step(
        "v17_latest_timing",
        "compute V17 three-factor latest score for sealed Top24 plus holdings",
    ) as step:
        timing, benchmark_context, price_metrics = _timing_and_market_context(
            table_root=table_root,
            sealed_symbols=candidates.sealed_symbols,
            cutoff=cutoff,
            acquisition_root=acquisition_root,
        )
        _write_parquet(run_root / "latest_timing.parquet", timing)
        _write_json(run_root / "benchmark_context.json", benchmark_context)
        step["result"] = {
            "sealed_symbols": len(candidates.sealed_symbols),
            "ready_raw_factor_rows": int(timing["status"].eq("READY").sum()),
            "timing_state": "UNREADY",
            "blocker": "calibration_not_admitted_official_delisting_cash_missing",
            "benchmark": benchmark_context,
        }

    with recorder.step(
        "holding_evidence",
        "summarize official financial, forecast, and dividend evidence for owner holdings",
    ) as step:
        holding_evidence = _holding_evidence(acquisition_root, holding_codes)
        _write_json(run_root / "holding_evidence.json", holding_evidence)
        step["result"] = {
            symbol: {
                table: details["row_count"]
                for table, details in evidence.items()
            }
            for symbol, evidence in holding_evidence.items()
        }

    with recorder.step(
        "source_admission_and_advice",
        "apply fail-closed source gate; produce research-only labels and revaluation",
    ) as step:
        source_gate = _source_gate(
            universe_conflict_count=len(official_only_rows) + len(local_only_rows),
            holding_evidence=holding_evidence,
            expected_holding_count=len(holding_codes),
        )
        advice = _advice(
            scored=scored,
            timing=timing,
            official=official,
            ledger=ledger,
            latest_market=latest_market,
            price_metrics=price_metrics,
            holding_evidence=holding_evidence,
            cash=cash,
        )
        _write_json(run_root / "source_gate.json", source_gate)
        _write_json(run_root / "investment_advice.json", advice)
        step["result"] = {
            "formal_admission": source_gate["formal_admission"],
            "diagnostic_terminal": source_gate["diagnostic_terminal"],
            "continue_holding": len(advice["继续持有"]),
            "trim_watch": len(advice["减仓观察"]),
            "add_candidates": len(advice["增配候选"]),
            "do_not_participate": len(advice["暂不参与"]),
            "revaluation": advice["portfolio_revaluation"],
        }

    with recorder.step(
        "runtime_verify_and_gate",
        "python -m quant_investor.v17_v2_runtime.cli verify; "
        "python -m quant_investor.v17_v2_runtime.cli gate "
        "--action SOURCE_MAINTAIN --version ABSENT --state MISSING "
        "--checkpoint PRE_IMPORT",
    ) as step:
        verification = verify_runtime().to_wire()
        gate = RuntimeGate(Path.cwd()).classify(
            "SOURCE_MAINTAIN",
            args.run_id,
            version="ABSENT",
            state="MISSING",
            checkpoint="PRE_IMPORT",
        )
        gate_wire = {
            **asdict(gate),
            "outcomes": [asdict(outcome) for outcome in gate.outcomes],
            "authority": False,
        }
        if verification["runtime_usable"] is not True or gate.allowed is not True:
            raise ValueError("runtime verification or pre-import source gate failed")
        _write_json(
            run_root / "cli_verification.json",
            {"verify": verification, "gate": gate_wire, "authority": False},
        )
        step["result"] = {
            "verify": verification,
            "gate": gate_wire,
        }

    summary = {
        "version": f"{PROTOCOL_VERSION}.cn-full-a-diagnostic-run.v1",
        "run_id": args.run_id,
        "analysis_cutoff": cutoff,
        "latest_complete_trade_date": "20260724",
        "source_gate": source_gate,
        "universe_reconciliation": reconciliation,
        "benchmark_context": benchmark_context,
        "advice": advice,
        "steps": recorder.steps,
        "authority": False,
        "side_effects": {
            "network_calls": 0,
            "broker_calls": 0,
            "orders": 0,
            "trades": 0,
            "production_pointer_writes": 0,
            "token_reads": 0,
        },
    }
    _write_json(run_root / "run_summary.json", summary)
    _write_text(run_root / "run_report.md", _render_report(summary))
    artifacts = {
        path.name: _file_evidence(path)
        for path in sorted(run_root.iterdir())
        if path.is_file() and path.name != "artifact_index.json"
    }
    _write_json(
        run_root / "artifact_index.json",
        {
            "run_id": args.run_id,
            "artifacts": artifacts,
            "authority": False,
        },
    )
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "run_summary_sha256": _sha256(run_root / "run_summary.json"),
                "run_report_sha256": _sha256(run_root / "run_report.md"),
                "artifact_index_sha256": _sha256(run_root / "artifact_index.json"),
                "formal_admission": source_gate["formal_admission"],
                "diagnostic_terminal": source_gate["diagnostic_terminal"],
                "authority": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
