"""Explicit v5 A_quant expression evaluator with daily-basic value/size fields."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_investor.factors.aquant_expression import cs_rank, ts_mean
from quant_investor.factors.pit_fundamentals import (
    DEFAULT_FUNDAMENTAL_MART_ROOT,
    FUNDAMENTAL_METRICS,
    build_fundamental_metric_matrices,
    normalize_ts_code,
)

FUNDAMENTAL_FIELD_NAMES = set(FUNDAMENTAL_METRICS)
DAILY_BASIC_FIELD_NAMES = {"turnover_rate", "pe", "pb", "total_mv", "circ_mv"}
PRICE_FIELD_NAMES = {
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "vwap",
    "volume",
    "amount",
} | DAILY_BASIC_FIELD_NAMES
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


@dataclass(frozen=True)
class AquantExpressionInputsV5:
    matrices: Mapping[str, pd.DataFrame]
    diagnostics: Mapping[str, Any]

    def context(self) -> dict[str, Any]:
        missing = ALLOWED_NAMES - set(self.matrices)
        extra = set(self.matrices) - ALLOWED_NAMES
        if missing or extra:
            raise ValueError(
                f"v5 expression matrix shape mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return {**dict(self.matrices), "ts_mean": ts_mean, "cs_rank": cs_rank}


def _validate_ast(node: ast.AST) -> None:
    for child in ast.walk(node):
        if not isinstance(child, ALLOWED_AST_NODES):
            raise ValueError(f"unsupported v5 expression syntax: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id not in ALLOWED_NAMES | ALLOWED_FUNCTIONS:
            raise ValueError(f"unsupported v5 expression name: {child.id}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError("unsupported v5 expression function")
            if child.keywords:
                raise ValueError("keyword arguments are not supported")


def evaluate_aquant_expression_v5(
    expression: str, inputs: AquantExpressionInputsV5 | Mapping[str, Any]
) -> pd.DataFrame:
    text = str(expression or "").strip()
    if not text:
        raise ValueError("empty v5 A_quant expression")
    tree = ast.parse(text, mode="eval")
    _validate_ast(tree)
    context = inputs.context() if isinstance(inputs, AquantExpressionInputsV5) else dict(inputs)
    result = eval(compile(tree, "<aquant_expression_v5>", "eval"), {"__builtins__": {}}, context)
    if isinstance(result, pd.Series):
        result = result.to_frame().T
    if not isinstance(result, pd.DataFrame):
        raise ValueError("v5 A_quant expression did not return a DataFrame")
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
    candidates: Sequence[str],
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
        values = _numeric_column(working, candidates)
        series = pd.Series(values.to_numpy(dtype=float), index=working["__date__"])
        series = series[~series.index.duplicated(keep="last")]
        matrix[symbol] = series.reindex(dates)
    return matrix


def build_aquant_expression_inputs_v5(
    frames: Mapping[str, pd.DataFrame],
    *,
    fundamental_mart_root: str | Path | None = DEFAULT_FUNDAMENTAL_MART_ROOT,
) -> AquantExpressionInputsV5:
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
    symbols = sorted(normalized_frames, key=lambda value: value.encode("ascii"))
    candidates_by_name = {
        "open": ("open", "Open"),
        "high": ("high", "High"),
        "low": ("low", "Low"),
        "close": ("close", "Close"),
        "adj_close": ("adj_close", "Adj Close", "close", "Close"),
        "volume": ("volume", "vol", "Volume"),
        "amount": ("amount", "turnover", "dollar_volume"),
        "turnover_rate": ("turnover_rate",),
        "pe": ("pe",),
        "pb": ("pb",),
        "total_mv": ("total_mv",),
        "circ_mv": ("circ_mv",),
    }
    matrices = {
        name: _matrix_from_frames(
            normalized_frames, candidates=columns, dates=dates, symbols=symbols
        )
        for name, columns in candidates_by_name.items()
    }
    volume = matrices["volume"].replace(0.0, np.nan)
    matrices["vwap"] = matrices["amount"].mul(10.0).div(volume)
    fundamental, diagnostics = build_fundamental_metric_matrices(
        dates,
        symbols,
        metrics=FUNDAMENTAL_METRICS,
        mart_root=fundamental_mart_root,
        allow_legacy_fallback=False,
    )
    for name in FUNDAMENTAL_FIELD_NAMES:
        matrices[name] = fundamental[name].reindex(index=dates, columns=symbols)
    return AquantExpressionInputsV5(matrices=matrices, diagnostics={"pit": diagnostics})


def compute_aquant_expression_factor_v5(
    factor_name: str,
    frames: Mapping[str, pd.DataFrame],
    *,
    expression: str,
    fundamental_mart_root: str | Path | None = DEFAULT_FUNDAMENTAL_MART_ROOT,
) -> pd.Series:
    if not expression:
        raise ValueError(f"missing v5 expression for factor: {factor_name}")
    inputs = build_aquant_expression_inputs_v5(frames, fundamental_mart_root=fundamental_mart_root)
    values = evaluate_aquant_expression_v5(expression, inputs)
    latest = values.dropna(how="all").tail(1)
    if latest.empty:
        return pd.Series(index=values.columns, dtype=float)
    return latest.iloc[0].astype(float)


__all__ = [
    "ALLOWED_NAMES",
    "AquantExpressionInputsV5",
    "build_aquant_expression_inputs_v5",
    "compute_aquant_expression_factor_v5",
    "evaluate_aquant_expression_v5",
]
