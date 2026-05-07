"""Safe matrix operators for offline factor-expression research."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


Matrix = list[list[float | None]]


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _is_matrix(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(
            isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray))
            for row in value
        )
    )


def _matrix(value: Sequence[Sequence[Any]], *, name: str = "matrix") -> Matrix:
    if not _is_matrix(value):
        raise ValueError(f"{name} must be a symbols x dates matrix.")
    rows: Matrix = []
    width: int | None = None
    for row_index, row in enumerate(value):
        row_values = [_to_float(item) for item in row]
        if width is None:
            width = len(row_values)
        elif len(row_values) != width:
            raise ValueError(f"{name} row {row_index} has inconsistent length.")
        rows.append(row_values)
    return rows


def _validate_same_shape(x: Matrix, y: Matrix, *, y_name: str = "y") -> None:
    if len(x) != len(y):
        raise ValueError(f"{y_name} must have the same number of rows as x.")
    for row_index, (x_row, y_row) in enumerate(zip(x, y)):
        if len(x_row) != len(y_row):
            raise ValueError(f"{y_name} row {row_index} must have the same length as x.")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    number = int(value)
    if number != value and not (isinstance(value, str) and str(number) == value):
        raise ValueError(f"{field_name} must be a positive integer.")
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return number


def _window_values(row: list[float | None], end_index: int, window: int) -> list[float] | None:
    start = end_index - window + 1
    if start < 0:
        return None
    values = row[start : end_index + 1]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values if value is not None]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rank_scaled(values: Sequence[float], current: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    positions = [index for index, value in enumerate(sorted_values) if value == current]
    if not positions:
        less_count = sum(1 for value in sorted_values if value < current)
        positions = [less_count]
    average_position = sum(positions) / len(positions)
    if len(sorted_values) == 1:
        return 0.0
    return average_position / (len(sorted_values) - 1)


def _quantile(values: Sequence[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value.")
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _blank_like(x: Matrix) -> Matrix:
    return [[None for _ in row] for row in x]


def _column_count(x: Matrix) -> int:
    return len(x[0]) if x else 0


def _as_matrix_or_scalar(
    value: Any,
    *,
    shape: tuple[int, int] | None = None,
) -> Matrix | float | None:
    if _is_matrix(value):
        matrix_value = _matrix(value)
        if shape is not None:
            rows, columns = shape
            if len(matrix_value) != rows or any(len(row) != columns for row in matrix_value):
                raise ValueError("matrix operands must have the same shape.")
        return matrix_value
    return _to_float(value)


def _elementwise_binary(
    x: Sequence[Sequence[Any]] | float | int,
    y: Sequence[Sequence[Any]] | float | int,
    fn: Callable[[float, float], float | None],
) -> Matrix:
    x_operand = _as_matrix_or_scalar(x)
    y_operand = _as_matrix_or_scalar(y)
    if not isinstance(x_operand, list) and not isinstance(y_operand, list):
        if x_operand is None or y_operand is None:
            return [[None]]
        return [[fn(x_operand, y_operand)]]
    base = x_operand if isinstance(x_operand, list) else y_operand
    assert isinstance(base, list)
    rows, columns = len(base), _column_count(base)
    if isinstance(x_operand, list):
        _validate_same_shape(base, x_operand, y_name="x")
    if isinstance(y_operand, list):
        _validate_same_shape(base, y_operand, y_name="y")

    output: Matrix = []
    for row_index in range(rows):
        row: list[float | None] = []
        for col_index in range(columns):
            left = x_operand[row_index][col_index] if isinstance(x_operand, list) else x_operand
            right = y_operand[row_index][col_index] if isinstance(y_operand, list) else y_operand
            if left is None or right is None:
                row.append(None)
            else:
                row.append(fn(left, right))
        output.append(row)
    return output


def _elementwise_unary(
    x: Sequence[Sequence[Any]],
    fn: Callable[[float], float | None],
) -> Matrix:
    matrix_x = _matrix(x, name="x")
    output: Matrix = []
    for x_row in matrix_x:
        row: list[float | None] = []
        for value in x_row:
            row.append(None if value is None else fn(value))
        output.append(row)
    return output


def ts_delay(x: Sequence[Sequence[Any]], periods: int) -> Matrix:
    lag = _positive_int(periods, "periods")
    matrix_x = _matrix(x, name="x")
    output: Matrix = []
    for row in matrix_x:
        if lag >= len(row):
            output.append([None for _ in row])
        else:
            output.append([None for _ in range(lag)] + list(row[:-lag]))
    return output


def ts_delta(x: Sequence[Sequence[Any]], periods: int) -> Matrix:
    return sub(x, ts_delay(x, periods))


def _rolling(
    x: Sequence[Sequence[Any]],
    window: int,
    fn: Callable[[list[float]], float | None],
) -> Matrix:
    lookback = _positive_int(window, "window")
    matrix_x = _matrix(x, name="x")
    output = _blank_like(matrix_x)
    for row_index, row in enumerate(matrix_x):
        for col_index in range(len(row)):
            values = _window_values(row, col_index, lookback)
            output[row_index][col_index] = None if values is None else fn(values)
    return output


def ts_mean(x: Sequence[Sequence[Any]], window: int) -> Matrix:
    return _rolling(x, window, _mean)


def ts_sum(x: Sequence[Sequence[Any]], window: int) -> Matrix:
    return _rolling(x, window, sum)


def ts_std(x: Sequence[Sequence[Any]], window: int) -> Matrix:
    return _rolling(x, window, _std)


def ts_min(x: Sequence[Sequence[Any]], window: int) -> Matrix:
    return _rolling(x, window, min)


def ts_max(x: Sequence[Sequence[Any]], window: int) -> Matrix:
    return _rolling(x, window, max)


def ts_rank(x: Sequence[Sequence[Any]], window: int) -> Matrix:
    def rank_current(values: list[float]) -> float:
        return _rank_scaled(values, values[-1])

    return _rolling(x, window, rank_current)


def ts_corr(x: Sequence[Sequence[Any]], y: Sequence[Sequence[Any]], window: int) -> Matrix:
    lookback = _positive_int(window, "window")
    matrix_x = _matrix(x, name="x")
    matrix_y = _matrix(y, name="y")
    _validate_same_shape(matrix_x, matrix_y, y_name="y")
    output = _blank_like(matrix_x)
    for row_index, (x_row, y_row) in enumerate(zip(matrix_x, matrix_y)):
        for col_index in range(len(x_row)):
            start = col_index - lookback + 1
            if start < 0:
                continue
            pairs = [
                (left, right)
                for left, right in zip(x_row[start : col_index + 1], y_row[start : col_index + 1])
                if left is not None and right is not None
            ]
            if len(pairs) < 2:
                continue
            xs = [left for left, _ in pairs]
            ys = [right for _, right in pairs]
            x_mean = _mean(xs)
            y_mean = _mean(ys)
            x_var = sum((value - x_mean) ** 2 for value in xs)
            y_var = sum((value - y_mean) ** 2 for value in ys)
            if x_var == 0.0 or y_var == 0.0:
                continue
            covariance = sum((left - x_mean) * (right - y_mean) for left, right in pairs)
            output[row_index][col_index] = covariance / math.sqrt(x_var * y_var)
    return output


def cs_rank(x: Sequence[Sequence[Any]]) -> Matrix:
    matrix_x = _matrix(x, name="x")
    output = _blank_like(matrix_x)
    for col_index in range(_column_count(matrix_x)):
        values = [
            matrix_x[row_index][col_index]
            for row_index in range(len(matrix_x))
            if matrix_x[row_index][col_index] is not None
        ]
        typed_values = [float(value) for value in values if value is not None]
        for row_index in range(len(matrix_x)):
            value = matrix_x[row_index][col_index]
            output[row_index][col_index] = (
                None if value is None else _rank_scaled(typed_values, value)
            )
    return output


def cs_zscore(x: Sequence[Sequence[Any]]) -> Matrix:
    matrix_x = _matrix(x, name="x")
    output = _blank_like(matrix_x)
    for col_index in range(_column_count(matrix_x)):
        values = [
            matrix_x[row_index][col_index]
            for row_index in range(len(matrix_x))
            if matrix_x[row_index][col_index] is not None
        ]
        typed_values = [float(value) for value in values if value is not None]
        if len(typed_values) < 2:
            continue
        mean = _mean(typed_values)
        std = _std(typed_values)
        if std == 0.0:
            continue
        for row_index in range(len(matrix_x)):
            value = matrix_x[row_index][col_index]
            output[row_index][col_index] = None if value is None else (value - mean) / std
    return output


def cs_winsorize(
    x: Sequence[Sequence[Any]],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> Matrix:
    lower = float(lower_quantile)
    upper = float(upper_quantile)
    if lower < 0.0 or upper > 1.0 or lower > upper:
        raise ValueError("lower_quantile and upper_quantile must satisfy 0 <= lower <= upper <= 1.")
    matrix_x = _matrix(x, name="x")
    output = _blank_like(matrix_x)
    for col_index in range(_column_count(matrix_x)):
        values = [
            matrix_x[row_index][col_index]
            for row_index in range(len(matrix_x))
            if matrix_x[row_index][col_index] is not None
        ]
        typed_values = [float(value) for value in values if value is not None]
        if not typed_values:
            continue
        lower_bound = _quantile(typed_values, lower)
        upper_bound = _quantile(typed_values, upper)
        for row_index in range(len(matrix_x)):
            value = matrix_x[row_index][col_index]
            if value is None:
                continue
            output[row_index][col_index] = min(max(value, lower_bound), upper_bound)
    return output


def cs_indneut(
    x: Sequence[Sequence[Any]],
    industry_by_symbol: Mapping[str, str],
    symbols: Sequence[str],
) -> Matrix:
    matrix_x = _matrix(x, name="x")
    if len(matrix_x) != len(symbols):
        raise ValueError("symbols length must match x rows.")
    output = _blank_like(matrix_x)
    for col_index in range(_column_count(matrix_x)):
        groups: dict[str, list[float]] = {}
        for row_index, symbol in enumerate(symbols):
            value = matrix_x[row_index][col_index]
            industry = industry_by_symbol.get(str(symbol))
            if value is None or industry is None:
                continue
            groups.setdefault(str(industry), []).append(value)
        means = {industry: _mean(values) for industry, values in groups.items() if values}
        for row_index, symbol in enumerate(symbols):
            value = matrix_x[row_index][col_index]
            industry = industry_by_symbol.get(str(symbol))
            if value is None or industry is None or industry not in means:
                continue
            output[row_index][col_index] = value - means[industry]
    return output


def cs_booksize(x: Sequence[Sequence[Any]]) -> Matrix:
    matrix_x = _matrix(x, name="x")
    output = _blank_like(matrix_x)
    for col_index in range(_column_count(matrix_x)):
        positives = [
            matrix_x[row_index][col_index]
            for row_index in range(len(matrix_x))
            if matrix_x[row_index][col_index] is not None and matrix_x[row_index][col_index] > 0
        ]
        negatives = [
            matrix_x[row_index][col_index]
            for row_index in range(len(matrix_x))
            if matrix_x[row_index][col_index] is not None and matrix_x[row_index][col_index] < 0
        ]
        positive_sum = sum(value for value in positives if value is not None)
        negative_abs_sum = sum(abs(value) for value in negatives if value is not None)
        for row_index in range(len(matrix_x)):
            value = matrix_x[row_index][col_index]
            if value is None:
                output[row_index][col_index] = None
            elif value > 0 and positive_sum != 0.0:
                output[row_index][col_index] = value / positive_sum
            elif value < 0 and negative_abs_sum != 0.0:
                output[row_index][col_index] = value / negative_abs_sum
            else:
                output[row_index][col_index] = 0.0
    return output


def add(
    x: Sequence[Sequence[Any]] | float | int,
    y: Sequence[Sequence[Any]] | float | int,
) -> Matrix:
    return _elementwise_binary(x, y, lambda left, right: left + right)


def sub(
    x: Sequence[Sequence[Any]] | float | int,
    y: Sequence[Sequence[Any]] | float | int,
) -> Matrix:
    return _elementwise_binary(x, y, lambda left, right: left - right)


def mul(
    x: Sequence[Sequence[Any]] | float | int,
    y: Sequence[Sequence[Any]] | float | int,
) -> Matrix:
    return _elementwise_binary(x, y, lambda left, right: left * right)


def div(
    x: Sequence[Sequence[Any]] | float | int,
    y: Sequence[Sequence[Any]] | float | int,
) -> Matrix:
    return _elementwise_binary(
        x,
        y,
        lambda left, right: None if right == 0.0 else left / right,
    )


def neg(x: Sequence[Sequence[Any]]) -> Matrix:
    return _elementwise_unary(x, lambda value: -value)


def abs_(x: Sequence[Sequence[Any]]) -> Matrix:
    return _elementwise_unary(x, abs)


def sign(x: Sequence[Sequence[Any]]) -> Matrix:
    def sign_value(value: float) -> float:
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0

    return _elementwise_unary(x, sign_value)


def log(x: Sequence[Sequence[Any]]) -> Matrix:
    return _elementwise_unary(x, lambda value: None if value <= 0.0 else math.log(value))


def sqrt(x: Sequence[Sequence[Any]]) -> Matrix:
    return _elementwise_unary(
        x,
        lambda value: None if value < 0.0 else math.sqrt(value),
    )


def maximum(
    x: Sequence[Sequence[Any]] | float | int,
    y: Sequence[Sequence[Any]] | float | int,
) -> Matrix:
    return _elementwise_binary(x, y, max)


def minimum(
    x: Sequence[Sequence[Any]] | float | int,
    y: Sequence[Sequence[Any]] | float | int,
) -> Matrix:
    return _elementwise_binary(x, y, min)


__all__ = [
    "Matrix",
    "ts_delay",
    "ts_delta",
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_rank",
    "ts_corr",
    "cs_rank",
    "cs_zscore",
    "cs_winsorize",
    "cs_indneut",
    "cs_booksize",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs_",
    "sign",
    "log",
    "sqrt",
    "maximum",
    "minimum",
]
