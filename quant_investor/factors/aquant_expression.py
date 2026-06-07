"""Safe runtime evaluator for A_quant expression-backed factors."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_investor.factors.pit_fundamentals import (
    DEFAULT_FUNDAMENTAL_MART_ROOT,
    FUNDAMENTAL_METRICS,
    build_fundamental_metric_matrices,
    normalize_ts_code,
)

FUNDAMENTAL_FIELD_NAMES = set(FUNDAMENTAL_METRICS)
PRICE_FIELD_NAMES = {
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "vwap",
    "volume",
    "amount",
    "turnover_rate",
}
ALLOWED_NAMES = PRICE_FIELD_NAMES | FUNDAMENTAL_FIELD_NAMES
ALLOWED_FUNCTIONS = {"ts_mean", "cs_rank"}
ALLOWED_AST_NODES = (
    ast.Expression,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)


@dataclass
class AquantExpressionInputs:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adj_close: pd.DataFrame
    vwap: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    turnover_rate: pd.DataFrame
    fin_roe: pd.DataFrame
    fin_roa: pd.DataFrame
    fin_debt_to_assets: pd.DataFrame
    fin_net_profit_yoy: pd.DataFrame
    fin_ocf_to_profit: pd.DataFrame
    fin_fcf_to_profit: pd.DataFrame
    fcf_to_price: pd.DataFrame
    diagnostics: dict[str, Any]

    def context(self) -> dict[str, Any]:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "adj_close": self.adj_close,
            "vwap": self.vwap,
            "volume": self.volume,
            "amount": self.amount,
            "turnover_rate": self.turnover_rate,
            "fin_roe": self.fin_roe,
            "fin_roa": self.fin_roa,
            "fin_debt_to_assets": self.fin_debt_to_assets,
            "fin_net_profit_yoy": self.fin_net_profit_yoy,
            "fin_ocf_to_profit": self.fin_ocf_to_profit,
            "fin_fcf_to_profit": self.fin_fcf_to_profit,
            "fcf_to_price": self.fcf_to_price,
            "ts_mean": ts_mean,
            "cs_rank": cs_rank,
        }


def _validate_ast(node: ast.AST) -> None:
    for child in ast.walk(node):
        if not isinstance(child, ALLOWED_AST_NODES):
            raise ValueError(f"unsupported expression syntax: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id not in ALLOWED_NAMES | ALLOWED_FUNCTIONS:
            raise ValueError(f"unsupported expression name: {child.id}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError("unsupported expression function")
            if child.keywords:
                raise ValueError("keyword arguments are not supported")


def ts_mean(values: pd.DataFrame | pd.Series, window: object) -> pd.DataFrame | pd.Series:
    """Rolling time-series mean with a small minimum-period guard."""

    try:
        width = int(float(window))
    except Exception as exc:
        raise ValueError(f"invalid ts_mean window: {window}") from exc
    if width <= 0:
        raise ValueError("ts_mean window must be positive")
    min_periods = max(3, min(width, 5))
    return values.rolling(width, min_periods=min_periods).mean()


def cs_rank(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Cross-sectional percentile rank."""

    if isinstance(values, pd.Series):
        return values.rank(pct=True)
    return values.rank(axis=1, pct=True)


def evaluate_aquant_expression(expression: str, inputs: AquantExpressionInputs | Mapping[str, Any]) -> pd.DataFrame:
    """Evaluate a constrained A_quant expression into a date x symbol matrix."""

    text = str(expression or "").strip()
    if not text:
        raise ValueError("empty A_quant expression")
    tree = ast.parse(text, mode="eval")
    _validate_ast(tree)
    context = inputs.context() if isinstance(inputs, AquantExpressionInputs) else dict(inputs)
    result = eval(compile(tree, "<aquant_expression>", "eval"), {"__builtins__": {}}, context)
    if isinstance(result, pd.Series):
        result = result.to_frame().T
    if not isinstance(result, pd.DataFrame):
        raise ValueError("A_quant expression did not return a DataFrame")
    return result.replace([np.inf, -np.inf], np.nan).astype(float)


def _date_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    for column in ("trade_date", "date"):
        if column in frame.columns:
            return pd.DatetimeIndex(pd.to_datetime(frame[column], errors="coerce"))
    return pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))


def _numeric_column(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["__date__"] = _date_index(working)
    return working.dropna(subset=["__date__"]).sort_values("__date__").reset_index(drop=True)


def _matrix_from_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    value_builder: Any,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    for raw_symbol, frame in frames.items():
        symbol = normalize_ts_code(raw_symbol)
        if symbol not in matrix.columns:
            continue
        working = _ordered_frame(frame)
        if working.empty:
            continue
        values = pd.to_numeric(value_builder(working), errors="coerce")
        series = pd.Series(values.to_numpy(dtype=float), index=working["__date__"])
        series = series[~series.index.duplicated(keep="last")]
        matrix[symbol] = series.reindex(dates)
    return matrix


def build_aquant_expression_inputs(
    frames: Mapping[str, pd.DataFrame],
    *,
    metadata_dir: str | Path | None = None,
    pit_series_path: str | Path | None = None,
    fundamental_mart_root: str | Path | None = DEFAULT_FUNDAMENTAL_MART_ROOT,
    allow_legacy_fundamental_fallback: bool | None = None,
) -> AquantExpressionInputs:
    """Build price/volume and PIT financial matrices for A_quant expressions."""

    date_values: list[pd.Timestamp] = []
    normalized_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        normalized = normalize_ts_code(symbol)
        if not normalized or frame is None or frame.empty:
            continue
        working = _ordered_frame(frame)
        if working.empty:
            continue
        normalized_frames[normalized] = working
        date_values.extend(list(working["__date__"]))
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(date_values).unique()))
    symbols = list(normalized_frames)

    def column_builder(*names: str) -> Any:
        return lambda frame: _numeric_column(frame, names)

    def vwap_builder(frame: pd.DataFrame) -> pd.Series:
        direct = _numeric_column(frame, ("vwap", "VWAP"))
        if direct.notna().any():
            return direct
        amount = _numeric_column(frame, ("amount", "turnover", "dollar_volume"))
        volume = _numeric_column(frame, ("volume", "vol", "Volume")).replace(0.0, np.nan)
        return amount.mul(10.0).div(volume)

    open_ = _matrix_from_frames(normalized_frames, value_builder=column_builder("open", "Open"), dates=dates, symbols=symbols)
    high = _matrix_from_frames(normalized_frames, value_builder=column_builder("high", "High"), dates=dates, symbols=symbols)
    low = _matrix_from_frames(normalized_frames, value_builder=column_builder("low", "Low"), dates=dates, symbols=symbols)
    close = _matrix_from_frames(normalized_frames, value_builder=column_builder("close", "Close"), dates=dates, symbols=symbols)
    adj_close = _matrix_from_frames(normalized_frames, value_builder=column_builder("adj_close", "Adj Close", "close", "Close"), dates=dates, symbols=symbols)
    volume = _matrix_from_frames(normalized_frames, value_builder=column_builder("volume", "vol", "Volume"), dates=dates, symbols=symbols)
    amount = _matrix_from_frames(normalized_frames, value_builder=column_builder("amount", "turnover", "dollar_volume"), dates=dates, symbols=symbols)
    turnover_rate = _matrix_from_frames(normalized_frames, value_builder=column_builder("turnover_rate"), dates=dates, symbols=symbols)
    vwap = _matrix_from_frames(normalized_frames, value_builder=vwap_builder, dates=dates, symbols=symbols)
    allow_legacy = (
        bool(metadata_dir or pit_series_path)
        if allow_legacy_fundamental_fallback is None
        else bool(allow_legacy_fundamental_fallback)
    )
    fundamental_matrices, pit_diagnostics = build_fundamental_metric_matrices(
        dates,
        symbols,
        metrics=FUNDAMENTAL_METRICS,
        metadata_dir=metadata_dir,
        pit_series_path=pit_series_path,
        mart_root=fundamental_mart_root,
        allow_legacy_fallback=allow_legacy,
    )
    return AquantExpressionInputs(
        open=open_,
        high=high,
        low=low,
        close=close,
        adj_close=adj_close,
        vwap=vwap,
        volume=volume,
        amount=amount,
        turnover_rate=turnover_rate,
        fin_roe=fundamental_matrices["fin_roe"].reindex(index=dates, columns=symbols),
        fin_roa=fundamental_matrices["fin_roa"].reindex(index=dates, columns=symbols),
        fin_debt_to_assets=fundamental_matrices["fin_debt_to_assets"].reindex(index=dates, columns=symbols),
        fin_net_profit_yoy=fundamental_matrices["fin_net_profit_yoy"].reindex(index=dates, columns=symbols),
        fin_ocf_to_profit=fundamental_matrices["fin_ocf_to_profit"].reindex(index=dates, columns=symbols),
        fin_fcf_to_profit=fundamental_matrices["fin_fcf_to_profit"].reindex(index=dates, columns=symbols),
        fcf_to_price=fundamental_matrices["fcf_to_price"].reindex(index=dates, columns=symbols),
        diagnostics={"pit": pit_diagnostics},
    )


def compute_aquant_expression_factor(
    factor_name: str,
    frames: Mapping[str, pd.DataFrame],
    *,
    expression: str | None = None,
    metadata_dir: str | Path | None = None,
    pit_series_path: str | Path | None = None,
    fundamental_mart_root: str | Path | None = DEFAULT_FUNDAMENTAL_MART_ROOT,
    allow_legacy_fundamental_fallback: bool | None = None,
) -> pd.Series:
    """Compute the latest cross-sectional raw value for an A_quant expression factor."""

    if not expression:
        raise ValueError(f"missing expression for A_quant factor: {factor_name}")
    inputs = build_aquant_expression_inputs(
        frames,
        metadata_dir=metadata_dir,
        pit_series_path=pit_series_path,
        fundamental_mart_root=fundamental_mart_root,
        allow_legacy_fundamental_fallback=allow_legacy_fundamental_fallback,
    )
    values = evaluate_aquant_expression(expression, inputs)
    if values.empty:
        return pd.Series(dtype=float)
    latest = values.dropna(how="all").tail(1)
    if latest.empty:
        return pd.Series(index=values.columns, dtype=float)
    return latest.iloc[0].astype(float)


__all__ = [
    "AquantExpressionInputs",
    "build_aquant_expression_inputs",
    "compute_aquant_expression_factor",
    "cs_rank",
    "evaluate_aquant_expression",
    "ts_mean",
]
