"""Offline single-factor backtest helpers.

This module consumes in-memory ``FactorMatrix`` and ``MatrixDataBundle``
artifacts. It does not fetch market data and does not connect research factors
to production stock selection or portfolio construction.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest_types import (
    BACKTEST_MODE_LONG_ONLY,
    BACKTEST_MODE_LONG_SHORT,
    DEFAULT_FACTOR_BACKTEST_DIR,
    DEFAULT_FACTOR_BACKTEST_RUNS_FILENAME,
    DEFAULT_FACTOR_DAILY_RECORDS_FILENAME,
    DEFAULT_FACTOR_WEIGHT_MATRICES_FILENAME,
    EXECUTION_PRICE_CLOSE,
    EXECUTION_PRICE_OPEN,
    EXECUTION_PRICE_VWAP,
    WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
    FactorBacktestAlignment,
    FactorDailyBacktestRecord,
    FactorWeightMatrix,
    SingleFactorBacktestRun,
    _EPSILON,
    _coerce_dates,
    _coerce_metadata,
    _ensure_json_serializable,
    _number_or_zero,
    _positive_int,
    _resolve_execution_price,
    _resolve_mode,
    _to_finite_float,
    _to_positive_price,
    make_daily_record_id,
    make_factor_backtest_run_id,
    make_factor_weights_id,
)
from quant_investor.factors.matrix import (
    FIELD_BENCHMARK_CLOSE,
    FIELD_BENCHMARK_RET,
    FIELD_CLOSE,
    FIELD_OPEN,
    FIELD_VWAP,
    FactorMatrix,
    MatrixDataBundle,
    add_standard_derived_fields,
)
from quant_investor.factors.schema import (
    FactorBacktestConfig,
    FactorBacktestResult,
    make_backtest_result_id,
)
from quant_investor.versioning import FACTOR_BACKTEST_SCHEMA_VERSION


def validate_finite_number(value: float, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number.")


def bps_to_decimal_return(value_bps: float) -> float:
    validate_finite_number(value_bps, field_name="value_bps")
    return float(value_bps) / 10000.0


def _usable_numbers(values: Sequence[float | None], *, field_name: str) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None:
            continue
        validate_finite_number(value, field_name=field_name)
        output.append(float(value))
    return output


def safe_mean(values: Sequence[float | None]) -> float | None:
    numbers = _usable_numbers(values, field_name="values")
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def safe_std(values: Sequence[float | None]) -> float | None:
    numbers = _usable_numbers(values, field_name="values")
    if not numbers:
        return None
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    return math.sqrt(variance)


def compound_returns(returns: Sequence[float | None]) -> float | None:
    numbers = _usable_numbers(returns, field_name="returns")
    if not numbers:
        return None
    total = 1.0
    for value in numbers:
        total *= 1.0 + value
    return total - 1.0


def max_drawdown_from_returns(returns: Sequence[float | None]) -> float | None:
    numbers = _usable_numbers(returns, field_name="returns")
    if not numbers:
        return None
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in numbers:
        equity *= 1.0 + value
        if equity > peak:
            peak = equity
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return max(0.0, max_drawdown)


def annualized_return_from_daily(
    returns: Sequence[float | None],
    *,
    trading_days: int = 252,
) -> float | None:
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    mean = safe_mean(returns)
    if mean is None:
        return None
    return mean * trading_days


def annualized_vol_from_daily(
    returns: Sequence[float | None],
    *,
    trading_days: int = 252,
) -> float | None:
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    std = safe_std(returns)
    if std is None:
        return None
    return std * math.sqrt(trading_days)


def sharpe_from_daily(
    returns: Sequence[float | None],
    *,
    trading_days: int = 252,
) -> float | None:
    ann_ret = annualized_return_from_daily(returns, trading_days=trading_days)
    ann_vol = annualized_vol_from_daily(returns, trading_days=trading_days)
    if ann_ret is None or ann_vol is None or abs(ann_vol) <= _EPSILON:
        return None
    return ann_ret / ann_vol


def estimate_turnover(prev_weights: Mapping[str, float], next_weights: Mapping[str, float]) -> float:
    symbols = set(prev_weights) | set(next_weights)
    total = 0.0
    for symbol in symbols:
        previous = float(prev_weights.get(symbol, 0.0))
        next_value = float(next_weights.get(symbol, 0.0))
        validate_finite_number(previous, field_name=f"prev_weights.{symbol}")
        validate_finite_number(next_value, field_name=f"next_weights.{symbol}")
        total += abs(next_value - previous)
    return 0.5 * total


def build_backtest_alignments(
    dates: Sequence[str],
    *,
    delay_days: int,
    holding_period_days: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    execution_price: str = EXECUTION_PRICE_VWAP,
) -> list[FactorBacktestAlignment]:
    resolved_dates = _coerce_dates(dates)
    resolved_delay = _positive_int(delay_days, "delay_days")
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    resolved_execution_price = _resolve_execution_price(execution_price)
    if start_date is not None:
        date.fromisoformat(start_date)
    if end_date is not None:
        date.fromisoformat(end_date)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be <= end_date.")

    alignments: list[FactorBacktestAlignment] = []
    for signal_index, signal_date in enumerate(resolved_dates):
        if start_date is not None and signal_date < start_date:
            continue
        if end_date is not None and signal_date > end_date:
            continue
        execution_start_index = signal_index + resolved_delay
        execution_end_index = execution_start_index + resolved_holding_period
        if execution_end_index >= len(resolved_dates):
            continue
        alignments.append(
            FactorBacktestAlignment(
                signal_date=signal_date,
                execution_start_date=resolved_dates[execution_start_index],
                execution_end_date=resolved_dates[execution_end_index],
                signal_index=signal_index,
                execution_start_index=execution_start_index,
                execution_end_index=execution_end_index,
                delay_days=resolved_delay,
                holding_period_days=resolved_holding_period,
                execution_price=resolved_execution_price,
                metadata={"alignment_policy": "signal_delay_execution_window"},
            )
        )
    return alignments


def build_execution_return_matrix(
    bundle: MatrixDataBundle,
    *,
    execution_price: str,
    holding_period_days: int = 1,
) -> list[list[float | None]]:
    resolved_execution_price = _resolve_execution_price(execution_price)
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    field_by_execution_price = {
        EXECUTION_PRICE_OPEN: FIELD_OPEN,
        EXECUTION_PRICE_VWAP: FIELD_VWAP,
        EXECUTION_PRICE_CLOSE: FIELD_CLOSE,
    }
    field_name = field_by_execution_price[resolved_execution_price]
    source_bundle = bundle
    if field_name == FIELD_VWAP and not source_bundle.has_field(FIELD_VWAP):
        source_bundle = add_standard_derived_fields(source_bundle)
    if not source_bundle.has_field(field_name):
        raise ValueError(f"bundle does not contain execution price field {field_name!r}.")
    prices = source_bundle.get_field(field_name)
    output: list[list[float | None]] = []
    for row in prices:
        result_row: list[float | None] = []
        for column_index, value in enumerate(row):
            future_index = column_index + resolved_holding_period
            if future_index >= len(row):
                result_row.append(None)
                continue
            start_price = _to_positive_price(value)
            end_price = _to_positive_price(row[future_index])
            if start_price is None or end_price is None:
                result_row.append(None)
                continue
            result_row.append(end_price / start_price - 1.0)
        output.append(result_row)
    return output


def _factor_bundle_symbols_dates_match(factor_matrix: FactorMatrix, bundle: MatrixDataBundle) -> bool:
    return (
        list(factor_matrix.symbols) == list(bundle.contract.symbols)
        and list(factor_matrix.dates) == list(bundle.contract.dates)
    )


def _expected_direction(factor_matrix: FactorMatrix) -> float:
    value = factor_matrix.metadata.get("expected_direction", 1.0)
    number = float(value)
    if not math.isfinite(number) or number == 0.0:
        raise ValueError("expected_direction must be finite and non-zero.")
    return 1.0 if number > 0.0 else -1.0


def _is_universe_eligible(bundle: MatrixDataBundle, row_index: int, date_index: int) -> bool:
    if bundle.universe_mask is None:
        return True
    return bool(bundle.universe_mask[row_index][date_index])


def _is_tradable(bundle: MatrixDataBundle, row_index: int, date_index: int) -> bool:
    if bundle.tradability_mask is None:
        return True
    if date_index >= len(bundle.contract.dates):
        return False
    return bool(bundle.tradability_mask[row_index][date_index])


def _make_empty_weight_rows(symbol_count: int, date_count: int) -> list[list[float | None]]:
    return [[0.0 for _ in range(date_count)] for _ in range(symbol_count)]


def _quantile_for_rank(rank: int, sample_count: int, quantile_count: int) -> int:
    return min(quantile_count, int(rank * quantile_count / sample_count) + 1)


def _config_snapshot(config: FactorBacktestConfig) -> dict[str, Any]:
    return dict(_ensure_json_serializable(config.to_dict(), "config"))


def build_quantile_weight_matrix(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    config: FactorBacktestConfig,
    *,
    mode: str = BACKTEST_MODE_LONG_SHORT,
    metadata: Mapping[str, Any] | None = None,
) -> FactorWeightMatrix:
    resolved_mode = _resolve_mode(mode)
    if not _factor_bundle_symbols_dates_match(factor_matrix, bundle):
        raise ValueError("factor_matrix symbols/dates must match bundle contract.")
    execution_price = _resolve_execution_price(config.execution_price)
    expected_direction = _expected_direction(factor_matrix)
    symbol_count = len(factor_matrix.symbols)
    date_count = len(factor_matrix.dates)
    long_weights = _make_empty_weight_rows(symbol_count, date_count)
    short_weights = _make_empty_weight_rows(symbol_count, date_count)
    net_weights = _make_empty_weight_rows(symbol_count, date_count)
    execution_returns = build_execution_return_matrix(
        bundle,
        execution_price=execution_price,
        holding_period_days=1,
    )

    for date_index in range(date_count):
        execution_start_index = date_index + config.delay_days
        eligible: list[tuple[str, int, float]] = []
        for row_index, symbol in enumerate(factor_matrix.symbols):
            factor_value = _to_finite_float(factor_matrix.values[row_index][date_index])
            if factor_value is None:
                continue
            if not _is_universe_eligible(bundle, row_index, date_index):
                continue
            if not _is_tradable(bundle, row_index, execution_start_index):
                continue
            if execution_start_index >= date_count:
                continue
            if execution_returns[row_index][execution_start_index] is None:
                continue
            score = factor_value * expected_direction
            eligible.append((symbol, row_index, score))

        if not eligible:
            continue
        eligible = sorted(eligible, key=lambda item: (item[2], item[0]))
        long_rows: list[int] = []
        short_rows: list[int] = []
        for rank, (_symbol, row_index, _score) in enumerate(eligible):
            quantile_index = _quantile_for_rank(rank, len(eligible), config.quantile_count)
            if quantile_index == config.long_quantile:
                long_rows.append(row_index)
            if (
                resolved_mode == BACKTEST_MODE_LONG_SHORT
                and config.long_short
                and config.short_quantile is not None
                and quantile_index == config.short_quantile
            ):
                short_rows.append(row_index)

        if long_rows:
            long_weight = 1.0 / len(long_rows)
            for row_index in long_rows:
                long_weights[row_index][date_index] = long_weight
        if short_rows:
            short_weight = -1.0 / len(short_rows)
            for row_index in short_rows:
                short_weights[row_index][date_index] = short_weight
        for row_index in range(symbol_count):
            net_weights[row_index][date_index] = (
                _number_or_zero(long_weights[row_index][date_index])
                + _number_or_zero(short_weights[row_index][date_index])
            )

    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "weighting_method": WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
            "mode": resolved_mode,
            "quantile_count": config.quantile_count,
            "long_quantile": config.long_quantile,
            "short_quantile": config.short_quantile,
            "expected_direction": expected_direction,
            "eligibility_policy": {
                "finite_factor_value": True,
                "universe_mask_signal_date": True,
                "tradability_mask_execution_start_date": True,
                "execution_return_required": True,
            },
            "execution_price": execution_price,
            "delay_days": config.delay_days,
            "config": _config_snapshot(config),
        }
    )

    return FactorWeightMatrix(
        weights_id=make_factor_weights_id(
            factor_matrix_id=factor_matrix.matrix_id,
            config_id=config.config_id,
            mode=resolved_mode,
        ),
        factor_matrix_id=factor_matrix.matrix_id,
        factor_id=factor_matrix.factor_id,
        factor_version=factor_matrix.factor_version,
        config_id=config.config_id,
        symbols=list(factor_matrix.symbols),
        dates=list(factor_matrix.dates),
        long_weights=long_weights,
        short_weights=short_weights,
        net_weights=net_weights,
        metadata=resolved_metadata,
    )


def _weights_for_column(
    symbols: Sequence[str],
    weights: Sequence[Sequence[float | None]],
    column_index: int,
) -> dict[str, float]:
    return {
        symbol: _number_or_zero(weights[row_index][column_index])
        for row_index, symbol in enumerate(symbols)
    }


def _nonzero_count(weights: Sequence[float | None]) -> int:
    return sum(1 for value in weights if abs(_number_or_zero(value)) > _EPSILON)


def _weighted_return(
    weights: Sequence[float | None],
    returns: Sequence[float | None],
) -> float | None:
    if _nonzero_count(weights) == 0:
        return None
    total = 0.0
    for weight, forward_return in zip(weights, returns):
        resolved_weight = _number_or_zero(weight)
        if abs(resolved_weight) <= _EPSILON:
            continue
        if forward_return is None:
            return None
        total += resolved_weight * forward_return
    return total


def _coverage_for_signal(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    signal_index: int,
) -> tuple[float, float]:
    denominator = 0
    usable = 0
    for row_index in range(len(factor_matrix.symbols)):
        if not _is_universe_eligible(bundle, row_index, signal_index):
            continue
        denominator += 1
        if _to_finite_float(factor_matrix.values[row_index][signal_index]) is not None:
            usable += 1
    if denominator == 0:
        return 0.0, 0.0
    coverage = usable / denominator
    return coverage, 1.0 - coverage


def _benchmark_return_at(
    bundle: MatrixDataBundle,
    execution_start_index: int,
    holding_period_days: int,
) -> float | None:
    if bundle.has_field(FIELD_BENCHMARK_RET):
        values = bundle.get_field(FIELD_BENCHMARK_RET)
        for row in values:
            if execution_start_index < len(row):
                value = _to_finite_float(row[execution_start_index])
                if value is not None:
                    return value
        return None
    if not bundle.has_field(FIELD_BENCHMARK_CLOSE):
        return None
    values = bundle.get_field(FIELD_BENCHMARK_CLOSE)
    if not values:
        return None
    row = values[0]
    end_index = execution_start_index + holding_period_days
    if execution_start_index >= len(row) or end_index >= len(row):
        return None
    start_price = _to_positive_price(row[execution_start_index])
    end_price = _to_positive_price(row[end_index])
    if start_price is None or end_price is None:
        return None
    return end_price / start_price - 1.0


def _rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    position = 0
    while position < len(indexed):
        end_position = position + 1
        while end_position < len(indexed) and indexed[end_position][1] == indexed[position][1]:
            end_position += 1
        average_rank = (position + end_position - 1) / 2.0 + 1.0
        for indexed_position in range(position, end_position):
            original_index = indexed[indexed_position][0]
            ranks[original_index] = average_rank
        position = end_position
    return ranks


def _pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    covariance = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    x_var = sum((x_value - x_mean) ** 2 for x_value in x_values)
    y_var = sum((y_value - y_mean) ** 2 for y_value in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator <= _EPSILON:
        return None
    return covariance / denominator


def _ic_for_alignment(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    returns: Sequence[Sequence[float | None]],
    alignment: FactorBacktestAlignment,
) -> tuple[float | None, float | None, int]:
    expected_direction = _expected_direction(factor_matrix)
    scores: list[float] = []
    forward_returns: list[float] = []
    for row_index in range(len(factor_matrix.symbols)):
        if not _is_universe_eligible(bundle, row_index, alignment.signal_index):
            continue
        factor_value = _to_finite_float(factor_matrix.values[row_index][alignment.signal_index])
        forward_return = returns[row_index][alignment.execution_start_index]
        if factor_value is None or forward_return is None:
            continue
        scores.append(factor_value * expected_direction)
        forward_returns.append(forward_return)
    pearson = _pearson_correlation(scores, forward_returns)
    rank = _pearson_correlation(_rank_values(scores), _rank_values(forward_returns)) if len(scores) >= 2 else None
    return pearson, rank, len(scores)


def compute_daily_backtest_records(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    config: FactorBacktestConfig,
    weight_matrix: FactorWeightMatrix,
    *,
    mode: str = BACKTEST_MODE_LONG_SHORT,
    holding_period_days: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> list[FactorDailyBacktestRecord]:
    resolved_mode = _resolve_mode(mode)
    if not _factor_bundle_symbols_dates_match(factor_matrix, bundle):
        raise ValueError("factor_matrix symbols/dates must match bundle contract.")
    if list(weight_matrix.symbols) != list(factor_matrix.symbols):
        raise ValueError("weight_matrix symbols must match factor_matrix symbols.")
    if list(weight_matrix.dates) != list(factor_matrix.dates):
        raise ValueError("weight_matrix dates must match factor_matrix dates.")
    execution_price = _resolve_execution_price(config.execution_price)
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    alignments = build_backtest_alignments(
        factor_matrix.dates,
        delay_days=config.delay_days,
        holding_period_days=resolved_holding_period,
        start_date=config.start_date or None,
        end_date=config.end_date or None,
        execution_price=execution_price,
    )
    execution_returns = build_execution_return_matrix(
        bundle,
        execution_price=execution_price,
        holding_period_days=resolved_holding_period,
    )
    daily_records: list[FactorDailyBacktestRecord] = []
    previous_net_weights = {symbol: 0.0 for symbol in factor_matrix.symbols}
    total_cost_bps = (
        config.transaction_cost_bps
        + config.slippage_bps
        + config.market_impact_bps
    )
    cost_per_turnover = bps_to_decimal_return(total_cost_bps)
    base_metadata = _coerce_metadata(metadata)

    for alignment in alignments:
        signal_index = alignment.signal_index
        execution_start_index = alignment.execution_start_index
        forward_returns = [
            execution_returns[row_index][execution_start_index]
            for row_index in range(len(factor_matrix.symbols))
        ]
        long_column = [
            weight_matrix.long_weights[row_index][signal_index]
            for row_index in range(len(factor_matrix.symbols))
        ]
        short_column = [
            weight_matrix.short_weights[row_index][signal_index]
            for row_index in range(len(factor_matrix.symbols))
        ]
        long_count = _nonzero_count(long_column)
        short_count = _nonzero_count(short_column)
        long_return = _weighted_return(long_column, forward_returns)
        short_return = _weighted_return(short_column, forward_returns)
        if resolved_mode == BACKTEST_MODE_LONG_ONLY:
            long_short_return = long_return
        elif long_count == 0 and short_count == 0:
            long_short_return = None
        elif (long_count > 0 and long_return is None) or (short_count > 0 and short_return is None):
            long_short_return = None
        else:
            long_short_return = (long_return or 0.0) + (short_return or 0.0)
        current_net_weights = _weights_for_column(
            factor_matrix.symbols,
            weight_matrix.net_weights,
            signal_index,
        )
        turnover = estimate_turnover(previous_net_weights, current_net_weights)
        previous_net_weights = current_net_weights
        after_cost_return = (
            None if long_short_return is None else long_short_return - turnover * cost_per_turnover
        )
        benchmark_return = _benchmark_return_at(bundle, execution_start_index, resolved_holding_period)
        excess_return = (
            None
            if after_cost_return is None or benchmark_return is None
            else after_cost_return - benchmark_return
        )
        coverage_ratio, missing_ratio = _coverage_for_signal(
            factor_matrix,
            bundle,
            signal_index,
        )
        pearson_ic, rank_ic, ic_sample_size = _ic_for_alignment(
            factor_matrix,
            bundle,
            execution_returns,
            alignment,
        )
        record_metadata = dict(base_metadata)
        record_metadata.update(
            {
                "alignment": alignment.to_dict(),
                "mode": resolved_mode,
                "holding_period_days": resolved_holding_period,
                "cost_bps": total_cost_bps,
                "ic": {
                    "pearson_ic": pearson_ic,
                    "rank_ic": rank_ic,
                    "sample_size": ic_sample_size,
                },
            }
        )
        daily_records.append(
            FactorDailyBacktestRecord(
                date=alignment.execution_end_date,
                signal_date=alignment.signal_date,
                execution_start_date=alignment.execution_start_date,
                execution_end_date=alignment.execution_end_date,
                long_return=long_return,
                short_return=short_return,
                long_short_return=long_short_return,
                after_cost_return=after_cost_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                turnover=turnover,
                long_count=long_count,
                short_count=short_count,
                coverage_ratio=coverage_ratio,
                missing_ratio=missing_ratio,
                metadata=record_metadata,
            )
        )
    return sorted(daily_records, key=lambda record: record.date)


def _factor_id_for_result(factor_matrix: FactorMatrix) -> str:
    return str(factor_matrix.factor_id or factor_matrix.matrix_id)


def _factor_version_for_result(factor_matrix: FactorMatrix) -> str:
    return str(factor_matrix.factor_version or "unversioned")


def _extract_ic_values(
    daily_records: Sequence[FactorDailyBacktestRecord],
    key: str,
) -> list[float | None]:
    values: list[float | None] = []
    for record in daily_records:
        ic_payload = record.metadata.get("ic", {})
        if isinstance(ic_payload, Mapping):
            values.append(ic_payload.get(key))  # type: ignore[arg-type]
    return values


def _average_int_field(
    daily_records: Sequence[FactorDailyBacktestRecord],
    field_name: str,
) -> float | None:
    values = [float(getattr(record, field_name)) for record in daily_records]
    return safe_mean(values)


def build_factor_backtest_result(
    *,
    factor_matrix: FactorMatrix,
    config: FactorBacktestConfig,
    daily_records: Sequence[FactorDailyBacktestRecord],
    metadata: Mapping[str, Any] | None = None,
) -> FactorBacktestResult:
    sorted_records = sorted(daily_records, key=lambda record: record.date)
    after_cost_returns = [record.after_cost_return for record in sorted_records]
    before_cost_returns = [record.long_short_return for record in sorted_records]
    pearson_ic_values = _extract_ic_values(sorted_records, "pearson_ic")
    rank_ic_values = _extract_ic_values(sorted_records, "rank_ic")
    usable_ic_values = _usable_numbers(pearson_ic_values, field_name="ic_values")
    ic_mean = safe_mean(pearson_ic_values)
    ic_std = safe_std(pearson_ic_values)
    ic_count = len(usable_ic_values)
    icir = None if ic_mean is None or ic_std is None or abs(ic_std) <= _EPSILON else ic_mean / ic_std
    ic_t_stat = None
    if ic_mean is not None and ic_std is not None and abs(ic_std) > _EPSILON and ic_count > 1:
        ic_t_stat = ic_mean / ic_std * math.sqrt(ic_count)
    positive_ic_ratio = None
    if ic_count > 0:
        positive_ic_ratio = sum(1 for value in usable_ic_values if value > 0.0) / ic_count
    sample_days = sum(1 for value in after_cost_returns if value is not None)
    coverage_ratio = safe_mean([record.coverage_ratio for record in sorted_records]) or 0.0
    missing_ratio = safe_mean([record.missing_ratio for record in sorted_records]) or 0.0
    resolved_metadata = _coerce_metadata(metadata)
    holding_period_days = int(resolved_metadata.get("holding_period_days", 1) or 1)
    execution_price = _resolve_execution_price(
        str(resolved_metadata.get("execution_price") or config.execution_price or EXECUTION_PRICE_VWAP)
    )
    resolved_metadata.update(
        {
            "factor_backtest_schema_version": FACTOR_BACKTEST_SCHEMA_VERSION,
            "alignment_policy": "signal_delay_execution_window",
            "execution_price": execution_price,
            "delay_days": config.delay_days,
            "holding_period_days": holding_period_days,
            "costing_policy": {
                "transaction_cost_bps": config.transaction_cost_bps,
                "slippage_bps": config.slippage_bps,
                "market_impact_bps": config.market_impact_bps,
                "deduction": "turnover * decimal_total_bps",
            },
            "pass": "phase9_pass3",
        }
    )
    factor_id = _factor_id_for_result(factor_matrix)
    factor_version = _factor_version_for_result(factor_matrix)
    start_date = (
        config.start_date
        or (sorted_records[0].date if sorted_records else factor_matrix.dates[0])
    )
    end_date = (
        config.end_date
        or (sorted_records[-1].date if sorted_records else factor_matrix.dates[-1])
    )
    return FactorBacktestResult(
        result_id=make_backtest_result_id(
            factor_id=factor_id,
            factor_version=factor_version,
            config_id=config.config_id,
        ),
        factor_id=factor_id,
        factor_version=factor_version,
        config_id=config.config_id,
        start_date=start_date,
        end_date=end_date,
        sample_days=sample_days,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        ann_ret=annualized_return_from_daily(after_cost_returns),
        ann_vol=annualized_vol_from_daily(after_cost_returns),
        sharpe=sharpe_from_daily(after_cost_returns),
        max_drawdown=max_drawdown_from_returns(after_cost_returns),
        turnover_avg=safe_mean([record.turnover for record in sorted_records]),
        long_num_avg=_average_int_field(sorted_records, "long_count"),
        short_num_avg=_average_int_field(sorted_records, "short_count"),
        rank_ic_mean=safe_mean(rank_ic_values),
        ic_mean=ic_mean,
        icir=icir,
        ic_t_stat=ic_t_stat,
        positive_ic_ratio=positive_ic_ratio,
        top_bottom_spread=safe_mean(before_cost_returns),
        after_cost_top_bottom_spread=safe_mean(after_cost_returns),
        before_cost_sharpe=sharpe_from_daily(before_cost_returns),
        after_cost_sharpe=sharpe_from_daily(after_cost_returns),
        monotonicity_score=None,
        capacity_estimate=None,
        slice_metrics={},
        metadata=resolved_metadata,
    )


def _determine_mode(config: FactorBacktestConfig, mode: str | None) -> str:
    if mode is not None:
        return _resolve_mode(mode)
    if config.long_short:
        return BACKTEST_MODE_LONG_SHORT
    if config.long_only:
        return BACKTEST_MODE_LONG_ONLY
    raise ValueError("config must enable long_short or long_only.")


def run_single_factor_backtest(
    factor_matrix: FactorMatrix,
    bundle: MatrixDataBundle,
    config: FactorBacktestConfig,
    *,
    mode: str | None = None,
    holding_period_days: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> SingleFactorBacktestRun:
    if not _factor_bundle_symbols_dates_match(factor_matrix, bundle):
        raise ValueError("factor_matrix symbols/dates must match bundle contract.")
    resolved_mode = _determine_mode(config, mode)
    execution_price = _resolve_execution_price(config.execution_price)
    resolved_holding_period = _positive_int(holding_period_days, "holding_period_days")
    base_metadata = _coerce_metadata(metadata)
    weight_matrix = build_quantile_weight_matrix(
        factor_matrix,
        bundle,
        config,
        mode=resolved_mode,
        metadata=base_metadata,
    )
    daily_records = compute_daily_backtest_records(
        factor_matrix,
        bundle,
        config,
        weight_matrix,
        mode=resolved_mode,
        holding_period_days=resolved_holding_period,
        metadata=base_metadata,
    )
    aggregate_result = build_factor_backtest_result(
        factor_matrix=factor_matrix,
        config=config,
        daily_records=daily_records,
        metadata={
            **base_metadata,
            "execution_price": execution_price,
            "holding_period_days": resolved_holding_period,
        },
    )
    run_id = make_factor_backtest_run_id(
        factor_matrix_id=factor_matrix.matrix_id,
        config_id=config.config_id,
        mode=resolved_mode,
        start_date=aggregate_result.start_date,
        end_date=aggregate_result.end_date,
    )
    weight_matrix.metadata = _coerce_metadata({
        **weight_matrix.metadata,
        "run_id": run_id,
    })
    for record in daily_records:
        record.metadata = _coerce_metadata({
            **record.metadata,
            "run_id": run_id,
            "daily_record_id": make_daily_record_id(run_id=run_id, date=record.date),
        })
    run_metadata = {
        **base_metadata,
        "factor_backtest_schema_version": FACTOR_BACKTEST_SCHEMA_VERSION,
        "offline_only": True,
        "pass": "phase9_pass3",
    }
    return SingleFactorBacktestRun(
        run_id=run_id,
        factor_matrix_id=factor_matrix.matrix_id,
        factor_id=factor_matrix.factor_id,
        factor_version=factor_matrix.factor_version,
        config_id=config.config_id,
        start_date=aggregate_result.start_date,
        end_date=aggregate_result.end_date,
        mode=resolved_mode,
        alignment_policy="signal_delay_execution_window",
        weighting_method=WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE,
        weight_matrix=weight_matrix,
        daily_records=daily_records,
        aggregate_result=aggregate_result,
        metadata=run_metadata,
    )


__all__ = [
    "EXECUTION_PRICE_OPEN",
    "EXECUTION_PRICE_VWAP",
    "EXECUTION_PRICE_CLOSE",
    "BACKTEST_MODE_LONG_SHORT",
    "BACKTEST_MODE_LONG_ONLY",
    "WEIGHTING_METHOD_EQUAL_QUANTILE_BOOKSIZE",
    "DEFAULT_FACTOR_BACKTEST_DIR",
    "DEFAULT_FACTOR_WEIGHT_MATRICES_FILENAME",
    "DEFAULT_FACTOR_BACKTEST_RUNS_FILENAME",
    "DEFAULT_FACTOR_DAILY_RECORDS_FILENAME",
    "FactorBacktestAlignment",
    "FactorWeightMatrix",
    "FactorDailyBacktestRecord",
    "SingleFactorBacktestRun",
    "make_factor_weights_id",
    "make_factor_backtest_run_id",
    "make_daily_record_id",
    "validate_finite_number",
    "bps_to_decimal_return",
    "safe_mean",
    "safe_std",
    "compound_returns",
    "max_drawdown_from_returns",
    "annualized_return_from_daily",
    "annualized_vol_from_daily",
    "sharpe_from_daily",
    "estimate_turnover",
    "build_backtest_alignments",
    "build_execution_return_matrix",
    "build_quantile_weight_matrix",
    "compute_daily_backtest_records",
    "build_factor_backtest_result",
    "run_single_factor_backtest",
]
