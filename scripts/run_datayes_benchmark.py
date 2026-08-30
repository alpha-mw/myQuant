#!/usr/bin/env python3
"""Run a research-only DataYes/Tushare benchmark on explicit dates and symbols."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant_investor.env_loading import load_env_file
from quant_investor.factors.governance.bootstrap import (
    BLEND_W80,
    CANONICAL_PARQUET,
    LOW_DOLLAR_VOLUME,
    compute_bootstrap_signals,
)
from quant_investor.factors.production_authority import FactorProductionStore
from quant_investor.market.datayes_provider import DataYesProvider
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.research.data_source_benchmark import (
    compare_candidates,
    compare_factors,
    compare_frames,
    compare_rankic,
    procurement_decision,
    rank_combined_signals,
    write_results,
)

DEFAULT_SYMBOLS = (
    "000001.SZ",
    "000002.SZ",
    "000333.SZ",
    "000651.SZ",
    "000858.SZ",
    "002415.SZ",
    "002594.SZ",
    "300750.SZ",
    "600000.SH",
    "600036.SH",
    "600519.SH",
    "600900.SH",
    "601318.SH",
    "601398.SH",
    "601899.SH",
    "601988.SH",
    "603288.SH",
    "688008.SH",
    "688111.SH",
    "688981.SH",
)


def _frames_from_combined(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(symbol): group.drop(columns=["symbol"], errors="ignore").reset_index(drop=True)
        for symbol, group in frame.groupby("ts_code", sort=True)
    }


def _domain_status(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "status": "ROWS_RETURNED" if not frame.empty else "NO_ROWS_RETURNED",
        "row_count": int(len(frame)),
        "fields": sorted(str(column) for column in frame.columns),
    }


def _compact_date(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .fillna("")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace("-", "", regex=False)
        .str[:8]
    )


def _tushare_indicator_from_raw(
    path: Path,
    *,
    symbols: tuple[str, ...],
    end_date: str,
) -> pd.DataFrame:
    raw = pd.read_parquet(
        path,
        columns=[
            "ts_code",
            "ann_date",
            "end_date",
            "roe_dt",
            "roe",
            "roa",
            "debt_to_assets",
            "netprofit_yoy",
        ],
    )
    raw = raw[raw["ts_code"].isin(symbols)].copy()
    raw["availability_date"] = _compact_date(raw["ann_date"])
    raw["end_date"] = _compact_date(raw["end_date"])
    raw = raw[raw["availability_date"].le(end_date)]
    roe = pd.to_numeric(raw["roe_dt"], errors="coerce").fillna(
        pd.to_numeric(raw["roe"], errors="coerce")
    )
    output = pd.DataFrame(
        {
            "ts_code": raw["ts_code"],
            "end_date": raw["end_date"],
            "availability_date": raw["availability_date"],
            "fin_roe": roe / 100.0,
            "fin_roa": pd.to_numeric(raw["roa"], errors="coerce") / 100.0,
            "fin_debt_to_assets": (pd.to_numeric(raw["debt_to_assets"], errors="coerce") / 100.0),
            "fin_net_profit_yoy": (pd.to_numeric(raw["netprofit_yoy"], errors="coerce") / 100.0),
        }
    )
    return (
        output.sort_values(["ts_code", "end_date", "availability_date"], kind="mergesort")
        .drop_duplicates(["ts_code", "end_date"], keep="last")
        .reset_index(drop=True)
    )


def _cache_request_sha256(
    *,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    factor_pointer_sha256: str | None,
) -> str:
    document = {
        "schema": "datayes-benchmark-cache-request.v1",
        "mapping": "datayes-canonical-v2",
        "symbols": list(symbols),
        "start_date": start_date,
        "end_date": end_date,
        "factor_pointer_sha256": factor_pointer_sha256,
    }
    raw = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cached_frame(
    cache_dir: Path | None,
    *,
    request_sha256: str,
    name: str,
    fetch: Callable[[], pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    if cache_dir is None:
        return fetch(), False
    if cache_dir.is_symlink():
        raise RuntimeError("benchmark cache directory cannot be a symlink")
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache_dir.chmod(0o700)
    root = cache_dir / request_sha256
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    path = root / f"{name}.parquet"
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("benchmark cache entry is not a regular file")
        return pd.read_parquet(path), True
    frame = fetch()
    temporary = root / f".{name}.{os.getpid()}.tmp.parquet"
    frame.to_parquet(temporary, index=False)
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return frame, False


def _active_head_cohort(workspace: Path, *, cohort_size: int) -> tuple[tuple[str, ...], str]:
    pointer_path = workspace / "results/factors/_active.json"
    pointer_sha = hashlib.sha256(pointer_path.read_bytes()).hexdigest()
    snapshot = FactorProductionStore(workspace).read_active_research_inputs(
        expected_pointer_sha256=pointer_sha
    )
    numeric = {
        factor_id: pd.Series(
            {symbol: float.fromhex(value) for symbol, value in values.items()},
            dtype=float,
        )
        for factor_id, values in snapshot["signal_values"].items()
    }
    ranked = rank_combined_signals(
        numeric,
        weights={LOW_DOLLAR_VOLUME: 0.5, BLEND_W80: 0.5},
    )
    return tuple(row["symbol"] for row in ranked[:cohort_size]), pointer_sha


def _anchor_dates(
    tushare_market: pd.DataFrame,
    datayes_market: pd.DataFrame,
    *,
    count: int,
    horizon: int,
) -> list[tuple[str, str]]:
    if count <= 0:
        return []
    left = set(tushare_market["trade_date"].astype(str))
    right = set(datayes_market["trade_date"].astype(str))
    dates = sorted(left & right)
    first = 100
    last = len(dates) - horizon - 1
    if last < first:
        return []
    indices = sorted(set(int(value) for value in np.linspace(first, last, count)))
    return [(dates[index], dates[index + horizon]) for index in indices]


def _signals_at(frames: dict[str, pd.DataFrame], *, anchor: str) -> dict[str, pd.Series]:
    truncated = {
        symbol: frame[frame["trade_date"].astype(str) <= anchor].copy()
        for symbol, frame in frames.items()
    }
    truncated = {symbol: frame for symbol, frame in truncated.items() if not frame.empty}
    return compute_bootstrap_signals(truncated, source_format=CANONICAL_PARQUET)


def _forward_returns(frames: dict[str, pd.DataFrame], *, anchor: str, target: str) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, frame in frames.items():
        keyed = frame.assign(_date=frame["trade_date"].astype(str)).set_index("_date")
        if anchor not in keyed.index or target not in keyed.index:
            continue
        start = pd.to_numeric(keyed.loc[anchor, "adj_close"], errors="coerce")
        end = pd.to_numeric(keyed.loc[target, "adj_close"], errors="coerce")
        if isinstance(start, pd.Series):
            start = start.iloc[-1]
        if isinstance(end, pd.Series):
            end = end.iloc[-1]
        if pd.notna(start) and pd.notna(end) and float(start) > 0.0:
            values[symbol] = float(end) / float(start) - 1.0
    return pd.Series(values, dtype=float)


def _rankic_by_anchor(
    frames: dict[str, pd.DataFrame],
    *,
    anchors: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for anchor, target in anchors:
        signals = _signals_at(frames, anchor=anchor)
        returns = _forward_returns(frames, anchor=anchor, target=target)
        for factor_id, signal in signals.items():
            pair = pd.concat([signal.rename("signal"), returns.rename("return")], axis=1).dropna()
            if len(pair) >= 3:
                result.setdefault(factor_id, {})[anchor] = float(
                    pair["signal"].corr(pair["return"], method="spearman")
                )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--cohort-from-active-head", action="store_true")
    parser.add_argument("--cohort-size", type=int, default=500)
    parser.add_argument("--skip-domain-probes", action="store_true")
    parser.add_argument("--rankic-anchor-count", type=int, default=0)
    parser.add_argument("--rankic-horizon", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    load_env_file(workspace / ".env")
    factor_pointer_sha = None
    if args.cohort_from_active_head:
        symbols, factor_pointer_sha = _active_head_cohort(workspace, cohort_size=args.cohort_size)
        cohort_kind = "CURRENT_TUSHARE_ACTIVE_HEAD_TOP_COHORT"
    else:
        symbols = tuple(value.strip().upper() for value in args.symbols.split(",") if value.strip())
        cohort_kind = "EXPLICIT_SYMBOLS"
    cache_request_sha256 = _cache_request_sha256(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        factor_pointer_sha256=factor_pointer_sha,
    )
    cache_dir = args.cache_dir.resolve() if args.cache_dir is not None else None
    cache_hits: dict[str, bool] = {}

    reader = MarketDataReader(data_root=workspace / "data")
    local_results = reader.read_symbol_frames(
        symbols, start_date=args.start_date, end_date=args.end_date
    )
    local_frames = {
        symbol: result.frame.copy()
        for symbol, result in local_results.items()
        if result.frame is not None and not result.frame.empty and not result.issues
    }
    tushare_market = pd.concat(local_frames.values(), ignore_index=True)

    datayes = DataYesProvider()
    datayes_market, cache_hits["market"] = _cached_frame(
        cache_dir,
        request_sha256=cache_request_sha256,
        name="market",
        fetch=lambda: datayes.daily(symbols, start_date=args.start_date, end_date=args.end_date),
    )
    market_domains = {"daily_ohlcv": _domain_status(datayes_market)}
    if not args.skip_domain_probes:
        market_domains.update(
            {
                "security_master": _domain_status(datayes.security_master(symbols)),
                "adjustment_factor": _domain_status(
                    datayes_market[["ts_code", "trade_date", "adj_factor"]]
                ),
                "st": _domain_status(
                    datayes.st(symbols, start_date=args.start_date, end_date=args.end_date)
                ),
                "suspension": _domain_status(
                    datayes.suspension(symbols, start_date=args.start_date, end_date=args.end_date)
                ),
                "price_limits": _domain_status(
                    datayes.price_limits(symbols, trade_date=args.end_date)
                ),
                "trade_calendar": _domain_status(
                    datayes.trade_calendar(start_date=args.start_date, end_date=args.end_date)
                ),
            }
        )
    market_metrics = compare_frames(
        tushare_market,
        datayes_market,
        keys=("ts_code", "trade_date"),
        fields=("open", "high", "low", "close", "vol", "amount", "turnover_rate", "pe", "pb"),
    )

    fundamental_path = workspace / "data/parquet/cn/fundamental_period/part.parquet"
    tushare_fundamental = pd.read_parquet(fundamental_path)
    tushare_fundamental = tushare_fundamental[
        tushare_fundamental["ts_code"].isin(symbols)
        & (
            tushare_fundamental["availability_date"].astype(str).str.replace("-", "")
            <= args.end_date
        )
    ]
    tushare_fundamental = tushare_fundamental.sort_values(
        ["ts_code", "end_date", "availability_date"], kind="mergesort"
    ).drop_duplicates(["ts_code", "end_date"], keep="last")
    datayes_fundamental, cache_hits["indicator_pit"] = _cached_frame(
        cache_dir,
        request_sha256=cache_request_sha256,
        name="indicator_pit",
        fetch=lambda: datayes.indicator_pit(symbols, start_date="20200101", end_date=args.end_date),
    )
    probe_symbols = symbols[:5]
    fundamental_domains = {}
    if not args.skip_domain_probes:
        fundamental_domains = {
            dataset: _domain_status(
                datayes.fundamental(
                    dataset,
                    probe_symbols,
                    start_date="20250101",
                    end_date=args.end_date,
                )
            )
            for dataset in (
                "raw_bs_pit",
                "raw_is_pit",
                "raw_cf_pit",
                "quarter_is_pit",
                "quarter_cf_pit",
                "ttm_is_pit",
                "ttm_cf_pit",
            )
        }
    fundamental_domains["indicator_pit"] = _domain_status(datayes_fundamental)
    datayes_fundamental = (
        datayes_fundamental[datayes_fundamental["availability_date"] <= args.end_date]
        .sort_values(["ts_code", "end_date", "availability_date"], kind="mergesort")
        .drop_duplicates(["ts_code", "end_date"], keep="last")
    )
    fundamental_current_canonical_metrics = compare_frames(
        tushare_fundamental,
        datayes_fundamental,
        keys=("ts_code", "end_date"),
        fields=("fin_roe", "fin_roa", "fin_debt_to_assets", "fin_net_profit_yoy"),
    )
    normalized_tushare_fundamental = _tushare_indicator_from_raw(
        workspace / "data/parquet/cn/fundamental_raw/table=fina_indicator/part.parquet",
        symbols=symbols,
        end_date=args.end_date,
    )
    fundamental_metrics = compare_frames(
        normalized_tushare_fundamental,
        datayes_fundamental,
        keys=("ts_code", "end_date"),
        fields=("fin_roe", "fin_roa", "fin_debt_to_assets", "fin_net_profit_yoy"),
    )

    datayes_frames = _frames_from_combined(datayes_market)
    tushare_signals = compute_bootstrap_signals(local_frames, source_format=CANONICAL_PARQUET)
    datayes_signals = compute_bootstrap_signals(datayes_frames, source_format=CANONICAL_PARQUET)
    factor_metrics = compare_factors(tushare_signals, datayes_signals, top_n=100)
    active_weights = {LOW_DOLLAR_VOLUME: 0.5, BLEND_W80: 0.5}
    tushare_candidates = rank_combined_signals(
        {factor_id: tushare_signals[factor_id] for factor_id in active_weights},
        weights=active_weights,
    )
    datayes_candidates = rank_combined_signals(
        {factor_id: datayes_signals[factor_id] for factor_id in active_weights},
        weights=active_weights,
    )
    candidate_comparison = compare_candidates(tushare_candidates, datayes_candidates, top_n=100)
    anchors = _anchor_dates(
        tushare_market,
        datayes_market,
        count=args.rankic_anchor_count,
        horizon=args.rankic_horizon,
    )
    rankic_metrics = compare_rankic(
        _rankic_by_anchor(local_frames, anchors=anchors),
        _rankic_by_anchor(datayes_frames, anchors=anchors),
    )
    limitations = [
        "The installed production factor engine exposes 2 active factors plus 1 control, not the requested ~20 core factors.",
        "The current canonical Fundamental comparison uses indicator PIT fields; raw/quarter/TTM PIT require separate field-level reconciliation.",
        "The published Tushare Fundamental mart contains legacy percent-to-ratio scale anomalies; this run uses an in-memory normalized projection for procurement metrics and does not mutate the published pointer.",
    ]
    if not anchors:
        limitations.append(
            "RankIC is unavailable because no explicit anchors fit inside the requested market window."
        )
    if cohort_kind == "CURRENT_TUSHARE_ACTIVE_HEAD_TOP_COHORT":
        limitations.append(
            f"The {len(symbols)}-symbol active-head challenge cohort is not a full-market replay; symbols below the cohort boundary cannot enter its Top100."
        )
    elif len(symbols) <= 100:
        limitations.append(
            f"A {len(symbols)}-symbol explicit cohort cannot establish a non-trivial full-market Top100 conclusion."
        )
    if args.skip_domain_probes:
        limitations.append(
            "Market and raw/quarter/TTM endpoint connectivity probes were skipped in this expanded run; the prior narrow run remains the evidence for those domains."
        )
    limitations.extend(
        f"DataYes provider limitation: {warning}" for warning in datayes.transport.warnings
    )
    procurement = procurement_decision(
        market=market_metrics,
        fundamental=fundamental_metrics,
        factors=factor_metrics,
        rankic=rankic_metrics,
        requested_factor_count=20,
    )
    current_installed_scope = procurement_decision(
        market=market_metrics,
        fundamental=fundamental_metrics,
        factors=factor_metrics,
        rankic=rankic_metrics,
        requested_factor_count=len(factor_metrics),
    )
    current_installed_scope.update(
        {
            "scope": "CURRENT_INSTALLED_TWO_ACTIVE_FACTORS_PLUS_W75_CONTROL",
            "cohort_kind": cohort_kind,
            "full_market_replay": False,
        }
    )
    payload = {
        "schema": "data-source-benchmark.v1",
        "status": "PARTIAL",
        "source_a": "TUSHARE_CANONICAL_PRODUCTION",
        "source_b": "DATAYES_TRIAL_RESEARCH_ONLY",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbol_count": len(symbols),
        "cohort_kind": cohort_kind,
        "factor_pointer_sha256": factor_pointer_sha,
        "cache_request_sha256": cache_request_sha256,
        "cache_hits": cache_hits,
        "market": market_metrics,
        "market_domains": market_domains,
        "fundamental": fundamental_metrics,
        "fundamental_current_canonical": fundamental_current_canonical_metrics,
        "fundamental_domains": fundamental_domains,
        "factors": factor_metrics,
        "candidate_comparison": candidate_comparison,
        "rankic": rankic_metrics,
        "rankic_anchors": [
            {"signal_date": anchor, "target_date": target} for anchor, target in anchors
        ],
        "limitations": limitations,
        "procurement": procurement,
        "current_installed_scope": current_installed_scope,
        "authority": {
            "canonical_mutated": False,
            "production_activated": False,
            "portfolio_mutated": False,
            "broker_order_trade": False,
        },
    }
    if factor_pointer_sha is not None:
        FactorProductionStore(workspace).assert_active_pointer(
            expected_pointer_sha256=factor_pointer_sha
        )
    paths = write_results(args.output_dir, payload)
    print(paths[0])
    print(paths[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
