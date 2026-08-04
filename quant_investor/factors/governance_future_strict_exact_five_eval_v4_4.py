"""Future-only strict exact-five evaluator for FactorGovernance v4.4.

The module is intentionally filesystem-free.  It accepts only caller-owned,
in-memory arrays and exposes a deterministic 60-row historical halo / 128-row
output-block contract.  The contract-owned immutable operator program is
interpreted twice: once by pandas and once by an independent NumPy dispatcher.
Production evaluation is block-local: each manifest input block starts with a
fresh rolling state, contains exactly 60 historical halo sessions, and emits at
most 128 output sessions.  A monolithic pandas run is not a bit-equivalence
oracle for concatenated block-local results.

Labels, outcomes, statistics, providers, paths, and fallback data are outside
this module's API by design.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_investor.factors import (
    governance_future_strict_signal_computability_v4_4 as strict_contract,
)


PROTOCOL_VERSION = "v4"
EVIDENCE_CONTRACT_VERSION = "v4.4"
HALO = 60
OUTPUT_BLOCK = 128


def _fma_exact(x: float, y: float, z: float) -> float:
    """Portable ``math.fma``: ``x * y + z`` under a single rounding.

    ``math.fma`` only exists on CPython 3.13+, but this package supports
    3.10+.  The fused contract is reproduced bit-for-bit by evaluating the
    product-sum exactly in rational arithmetic and rounding once at the end.
    Rounding twice -- plain ``x * y + z`` -- would perturb the rolling
    variance recurrence below and therefore change published receipts, so
    the exact path is the only admissible fallback.
    """

    x = float(x)
    y = float(y)
    z = float(z)
    if math.isnan(x) or math.isnan(y) or math.isnan(z):
        return math.nan
    if math.isinf(x) or math.isinf(y):
        if x == 0.0 or y == 0.0:
            raise ValueError("invalid operation in fma")
        product_sign = math.copysign(1.0, x) * math.copysign(1.0, y)
        if math.isinf(z) and math.copysign(1.0, z) != product_sign:
            raise ValueError("invalid operation in fma")
        return math.copysign(math.inf, product_sign)
    if math.isinf(z):
        return z
    exact = Fraction(x) * Fraction(y) + Fraction(z)
    if exact == 0:
        # Fraction cannot represent signed zero.  IEEE round-to-nearest
        # yields -0.0 only when the product and the addend are both -0.0;
        # every genuine cancellation returns +0.0.
        product_sign = math.copysign(1.0, x) * math.copysign(1.0, y)
        product_is_zero = x == 0.0 or y == 0.0
        if (
            product_is_zero
            and z == 0.0
            and product_sign < 0.0
            and math.copysign(1.0, z) < 0.0
        ):
            return -0.0
        return 0.0
    # ``math.fma`` reports a non-representable finite result rather than
    # saturating to infinity; mirror that -- including the message -- so
    # callers cannot diverge by interpreter version.
    try:
        result = float(exact)
    except OverflowError as exc:
        raise OverflowError("overflow in fma") from exc
    if math.isinf(result):
        raise OverflowError("overflow in fma")
    return result


# Fast path where the interpreter provides the primitive; the exact fallback
# above is bit-identical and is what runs on 3.10-3.12.
_fma = getattr(math, "fma", _fma_exact)

INPUT_FIELDS = ("raw_close", "raw_open", "vol", "adj_close")
_GOLDEN_OPERATOR_PROGRAM_SET = strict_contract.validate_operator_program_set_v4_4(
    strict_contract.operator_program_set_v4_4()
)
FACTOR_DIRECTIONS = MappingProxyType(
    {
        program["name"]: program["direction"]
        for program in _GOLDEN_OPERATOR_PROGRAM_SET["candidates"]
    }
)
FACTOR_NAMES = tuple(FACTOR_DIRECTIONS)

PANDAS_ENGINE_ID = "closed_pandas_source_dag.future_strictexact.v4.4"
NUMPY_ENGINE_ID = "independent_numpy_local_formulas.future_strictexact.v4.4"
BLOCK_MANIFEST_SCHEMA_VERSION = (
    "factor-governance-future-strict-block-manifest.v4.4"
)
MATRIX_DESCRIPTOR_SCHEMA_VERSION = (
    "factor-governance-future-strict-matrix-descriptor.v4.4"
)
EXACT_COMPARISON_SCHEMA_VERSION = (
    "factor-governance-future-strict-exact-comparison.v4.4"
)
_BLOCK_ROW_FIELDS = frozenset(
    {
        "block_index",
        "input_start_offset",
        "input_end_offset",
        "output_start_offset",
        "output_end_offset",
        "local_output_start_offset",
        "local_output_end_offset",
        "input_row_count",
        "output_row_count",
        "input_first_date",
        "input_last_date",
        "output_first_date",
        "output_last_date",
        "symbol_axis",
        "future_halo_row_count",
    }
)
_BLOCK_ROW_INTEGER_FIELDS = frozenset(
    {
        "block_index",
        "input_start_offset",
        "input_end_offset",
        "output_start_offset",
        "output_end_offset",
        "local_output_start_offset",
        "local_output_end_offset",
        "input_row_count",
        "output_row_count",
        "future_halo_row_count",
    }
)
_CANONICAL_NAN_BITS = np.uint64(0x7FF8000000000000)
_SIGN_BITS = np.uint64(0x8000000000000000)
_MAGNITUDE_BITS = np.uint64(0x7FFFFFFFFFFFFFFF)
_SHA256_HEX = frozenset("0123456789abcdef")


class FactorGovernanceFutureStrictExactFiveEvalV4_4Error(ValueError):
    """Raised when the strict exact-five evaluator cannot prove its contract."""


def _error(message: str) -> FactorGovernanceFutureStrictExactFiveEvalV4_4Error:
    return FactorGovernanceFutureStrictExactFiveEvalV4_4Error(message)


def canonical_json_bytes_v4_4(value: Any) -> bytes:
    """Serialize a JSON value with the single canonical module encoding."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error(f"value is not canonical JSON: {exc}") from exc


def semantic_sha256_v4_4(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_4(value)).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise _error(f"{label} must be a lowercase SHA-256")
    return value


def _normalize_date(value: Any, label: str) -> str:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise _error(f"{label} cannot be NaT")
        timestamp = pd.Timestamp(value)
    elif isinstance(value, pd.Timestamp):
        timestamp = value
    elif isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
    elif isinstance(value, date):
        timestamp = pd.Timestamp(value)
    elif type(value) is str:
        if len(value) != 10 or value[4] != "-" or value[7] != "-":
            raise _error(f"{label} must use YYYY-MM-DD")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise _error(f"{label} is not a calendar date") from exc
    else:
        raise _error(f"{label} must be a date or YYYY-MM-DD string")
    if timestamp.tzinfo is not None:
        raise _error(f"{label} must be timezone-naive")
    if timestamp != timestamp.normalize():
        raise _error(f"{label} must not contain a time component")
    return timestamp.strftime("%Y-%m-%d")


def _normalize_dates(values: Sequence[Any], label: str = "dates") -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise _error(f"{label} must be a sequence")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise _error(f"{label} must be a sequence") from exc
    normalized = tuple(
        _normalize_date(value, f"{label}[{position}]")
        for position, value in enumerate(raw_values)
    )
    if not normalized:
        raise _error(f"{label} must not be empty")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise _error(f"{label} must be strictly increasing and unique")
    return normalized


def _normalize_symbols(
    values: Sequence[Any], label: str = "symbols"
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise _error(f"{label} must be a sequence")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise _error(f"{label} must be a sequence") from exc
    if not raw_values:
        raise _error(f"{label} must not be empty")
    normalized: list[str] = []
    for position, value in enumerate(raw_values):
        if type(value) is not str or not value or value != value.strip():
            raise _error(f"{label}[{position}] must be a non-empty trimmed string")
        if "\n" in value or "\r" in value:
            raise _error(f"{label}[{position}] cannot contain a newline")
        normalized.append(value)
    result = tuple(normalized)
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise _error(f"{label} must be sorted, unique, and complete")
    return result


def _axis_descriptor(items: Sequence[str]) -> dict[str, Any]:
    payload = b"".join(item.encode("utf-8") + b"\n" for item in items)
    return {
        "count": len(items),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first": items[0] if items else None,
        "last": items[-1] if items else None,
    }


def _strict_float64_matrix(
    value: Any, *, shape: tuple[int, int], label: str
) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
        raise _error(f"{label} must be an exact float64 NumPy matrix")
    if value.ndim != 2 or value.shape != shape:
        raise _error(f"{label} shape must be {shape}")
    if np.isinf(value).any():
        raise _error(f"{label} must not contain positive or negative infinity")
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _strict_bool_matrix(
    value: Any, *, shape: tuple[int, int], label: str
) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(bool):
        raise _error(f"{label} must be an exact bool NumPy matrix")
    if value.ndim != 2 or value.shape != shape:
        raise _error(f"{label} shape must be {shape}")
    result = np.array(value, dtype=bool, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class InputBlockV4_4:
    """Immutable, axis-bound input block with no implicit data resolution."""

    dates: tuple[str, ...]
    symbols: tuple[str, ...]
    raw_close: np.ndarray
    raw_open: np.ndarray
    vol: np.ndarray
    adj_close: np.ndarray
    pit_mask: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        *,
        dates: Sequence[Any],
        symbols: Sequence[Any],
        raw_close: np.ndarray,
        raw_open: np.ndarray,
        vol: np.ndarray,
        adj_close: np.ndarray,
        pit_mask: np.ndarray,
    ) -> "InputBlockV4_4":
        normalized_dates = _normalize_dates(dates)
        normalized_symbols = _normalize_symbols(symbols)
        shape = (len(normalized_dates), len(normalized_symbols))
        return cls(
            dates=normalized_dates,
            symbols=normalized_symbols,
            raw_close=_strict_float64_matrix(
                raw_close, shape=shape, label="raw_close"
            ),
            raw_open=_strict_float64_matrix(raw_open, shape=shape, label="raw_open"),
            vol=_strict_float64_matrix(vol, shape=shape, label="vol"),
            adj_close=_strict_float64_matrix(
                adj_close, shape=shape, label="adj_close"
            ),
            pit_mask=_strict_bool_matrix(pit_mask, shape=shape, label="pit_mask"),
        )


def build_input_block_v4_4(
    *,
    dates: Sequence[Any],
    symbols: Sequence[Any],
    raw_close: np.ndarray,
    raw_open: np.ndarray,
    vol: np.ndarray,
    adj_close: np.ndarray,
    pit_mask: np.ndarray,
) -> InputBlockV4_4:
    return InputBlockV4_4.from_arrays(
        dates=dates,
        symbols=symbols,
        raw_close=raw_close,
        raw_open=raw_open,
        vol=vol,
        adj_close=adj_close,
        pit_mask=pit_mask,
    )


def validate_input_block_v4_4(value: Any) -> InputBlockV4_4:
    if type(value) is not InputBlockV4_4:
        raise _error("input block must be an InputBlockV4_4")
    normalized_dates = _normalize_dates(value.dates)
    normalized_symbols = _normalize_symbols(value.symbols)
    if normalized_dates != value.dates or normalized_symbols != value.symbols:
        raise _error("input block axes are not canonical")
    shape = (len(value.dates), len(value.symbols))
    for field in INPUT_FIELDS:
        matrix = getattr(value, field)
        if (
            type(matrix) is not np.ndarray
            or matrix.dtype != np.dtype(np.float64)
            or matrix.ndim != 2
            or matrix.shape != shape
            or matrix.flags.writeable
            or not matrix.flags.c_contiguous
            or np.isinf(matrix).any()
        ):
            raise _error(f"input block {field} is not immutable strict float64")
    if (
        type(value.pit_mask) is not np.ndarray
        or value.pit_mask.dtype != np.dtype(bool)
        or value.pit_mask.ndim != 2
        or value.pit_mask.shape != shape
        or value.pit_mask.flags.writeable
        or not value.pit_mask.flags.c_contiguous
    ):
        raise _error("input block pit_mask is not immutable strict bool")
    return value


def build_block_manifest_v4_4(
    dates: Sequence[Any], symbols: Sequence[Any]
) -> dict[str, Any]:
    """Partition a calendar into independent 60-halo/128-output state scopes."""

    source_dates = _normalize_dates(dates, "source_calendar")
    full_symbols = _normalize_symbols(symbols, "historical_symbols")
    if len(source_dates) <= HALO:
        raise _error(f"source calendar must contain more than HALO={HALO} rows")
    blocks: list[dict[str, Any]] = []
    for block_index, output_start in enumerate(
        range(HALO, len(source_dates), OUTPUT_BLOCK)
    ):
        output_end = min(output_start + OUTPUT_BLOCK, len(source_dates))
        input_start = output_start - HALO
        input_end = output_end
        blocks.append(
            {
                "block_index": block_index,
                "input_start_offset": input_start,
                "input_end_offset": input_end,
                "output_start_offset": output_start,
                "output_end_offset": output_end,
                "local_output_start_offset": HALO,
                "local_output_end_offset": HALO + output_end - output_start,
                "input_row_count": input_end - input_start,
                "output_row_count": output_end - output_start,
                "input_first_date": source_dates[input_start],
                "input_last_date": source_dates[input_end - 1],
                "output_first_date": source_dates[output_start],
                "output_last_date": source_dates[output_end - 1],
                "symbol_axis": _axis_descriptor(full_symbols),
                "future_halo_row_count": 0,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": BLOCK_MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "halo": HALO,
        "output_block": OUTPUT_BLOCK,
        "source_row_count": len(source_dates),
        "proof_output_row_count": len(source_dates) - HALO,
        "source_calendar": list(source_dates),
        "proof_output_calendar": list(source_dates[HALO:]),
        "date_axis": _axis_descriptor(source_dates),
        "proof_output_date_axis": _axis_descriptor(source_dates[HALO:]),
        "symbol_axis": _axis_descriptor(full_symbols),
        "full_historical_symbols": list(full_symbols),
        "block_count": len(blocks),
        "blocks": blocks,
    }
    manifest["manifest_semantic_sha256"] = semantic_sha256_v4_4(manifest)
    return manifest


def _validate_block_row_structure_v4_4(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("block row must be an object")
    payload = copy.deepcopy(dict(value))
    if set(payload) != _BLOCK_ROW_FIELDS:
        raise _error("block row fields mismatch")
    for field in _BLOCK_ROW_INTEGER_FIELDS:
        if type(payload[field]) is not int:
            raise _error(f"block row {field} must be an exact integer")

    block_index = payload["block_index"]
    input_start = payload["input_start_offset"]
    input_end = payload["input_end_offset"]
    output_start = payload["output_start_offset"]
    output_end = payload["output_end_offset"]
    local_start = payload["local_output_start_offset"]
    local_end = payload["local_output_end_offset"]
    input_count = payload["input_row_count"]
    output_count = payload["output_row_count"]
    if (
        block_index < 0
        or input_start != block_index * OUTPUT_BLOCK
        or output_start != input_start + HALO
        or input_end != output_end
        or input_count != input_end - input_start
        or output_count != output_end - output_start
        or output_count <= 0
        or output_count > OUTPUT_BLOCK
        or input_count != HALO + output_count
        or local_start != HALO
        or local_end != input_count
        or local_end - local_start != output_count
        or payload["future_halo_row_count"] != 0
    ):
        raise _error("block row offsets/counts violate the deterministic partition")

    date_fields = (
        "input_first_date",
        "input_last_date",
        "output_first_date",
        "output_last_date",
    )
    for field in date_fields:
        value_date = payload[field]
        if (
            type(value_date) is not str
            or _normalize_date(value_date, field) != value_date
        ):
            raise _error(f"block row {field} must be an exact canonical date")
    if (
        payload["input_first_date"] >= payload["output_first_date"]
        or payload["output_first_date"] > payload["output_last_date"]
        or payload["input_last_date"] != payload["output_last_date"]
    ):
        raise _error("block row dates violate the deterministic partition")

    axis = payload["symbol_axis"]
    axis_fields = {"count", "sha256", "first", "last"}
    if type(axis) is not dict or set(axis) != axis_fields:
        raise _error("block row symbol axis fields mismatch")
    if type(axis["count"]) is not int or axis["count"] <= 0:
        raise _error("block row symbol axis count must be an exact positive integer")
    _sha256(axis["sha256"], "block row symbol axis SHA")
    if (
        type(axis["first"]) is not str
        or not axis["first"]
        or axis["first"] != axis["first"].strip()
        or type(axis["last"]) is not str
        or not axis["last"]
        or axis["last"] != axis["last"].strip()
        or axis["first"] > axis["last"]
    ):
        raise _error("block row symbol axis endpoints are invalid")
    return payload


def _validate_block_row_against_source_v4_4(
    value: Any,
    *,
    dates: Sequence[Any],
    symbols: Sequence[Any],
) -> dict[str, Any]:
    payload = _validate_block_row_structure_v4_4(value)
    expected_manifest = build_block_manifest_v4_4(dates, symbols)
    block_index = payload["block_index"]
    if block_index >= expected_manifest["block_count"]:
        raise _error("block row index is outside the deterministic partition")
    expected = expected_manifest["blocks"][block_index]
    if canonical_json_bytes_v4_4(payload) != canonical_json_bytes_v4_4(expected):
        raise _error("block row does not match the deterministic source partition")
    return payload


def _validate_block_row_against_input_block_v4_4(
    value: Any,
    source_block: InputBlockV4_4,
) -> tuple[dict[str, Any], InputBlockV4_4]:
    payload = _validate_block_row_structure_v4_4(value)
    block = validate_input_block_v4_4(source_block)
    local_start = payload["local_output_start_offset"]
    local_end = payload["local_output_end_offset"]
    if (
        len(block.dates) != payload["input_row_count"]
        or payload["input_first_date"] != block.dates[0]
        or payload["input_last_date"] != block.dates[-1]
        or payload["output_first_date"] != block.dates[local_start]
        or payload["output_last_date"] != block.dates[local_end - 1]
        or canonical_json_bytes_v4_4(payload["symbol_axis"])
        != canonical_json_bytes_v4_4(_axis_descriptor(block.symbols))
    ):
        raise _error("block row does not match the exact input block provenance")
    return payload, block


def validate_block_manifest_v4_4(
    value: Any,
    *,
    dates: Sequence[Any] | None = None,
    symbols: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("block manifest must be an object")
    payload = copy.deepcopy(dict(value))
    expected_fields = {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "halo",
        "output_block",
        "source_row_count",
        "proof_output_row_count",
        "source_calendar",
        "proof_output_calendar",
        "date_axis",
        "proof_output_date_axis",
        "symbol_axis",
        "full_historical_symbols",
        "block_count",
        "blocks",
        "manifest_semantic_sha256",
    }
    if set(payload) != expected_fields:
        raise _error("block manifest fields mismatch")
    if type(payload["blocks"]) is not list:
        raise _error("block manifest blocks must be an exact list")
    for block_row in payload["blocks"]:
        _validate_block_row_structure_v4_4(block_row)
    source_dates = payload["source_calendar"] if dates is None else dates
    full_symbols = payload["full_historical_symbols"] if symbols is None else symbols
    expected = build_block_manifest_v4_4(source_dates, full_symbols)
    if canonical_json_bytes_v4_4(payload) != canonical_json_bytes_v4_4(expected):
        raise _error("block manifest does not match the deterministic partition")
    return payload


def slice_input_block_v4_4(
    source: InputBlockV4_4, block_row: Mapping[str, Any]
) -> InputBlockV4_4:
    """Materialize one immutable halo-bearing block from an in-memory source."""

    source = validate_input_block_v4_4(source)
    row = _validate_block_row_against_source_v4_4(
        block_row, dates=source.dates, symbols=source.symbols
    )
    start = row["input_start_offset"]
    end = row["input_end_offset"]
    return build_input_block_v4_4(
        dates=source.dates[start:end],
        symbols=source.symbols,
        raw_close=np.array(source.raw_close[start:end], dtype=np.float64, copy=True),
        raw_open=np.array(source.raw_open[start:end], dtype=np.float64, copy=True),
        vol=np.array(source.vol[start:end], dtype=np.float64, copy=True),
        adj_close=np.array(source.adj_close[start:end], dtype=np.float64, copy=True),
        pit_mask=np.array(source.pit_mask[start:end], dtype=bool, copy=True),
    )


def _canonical_matrix_bit_views(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.array(matrix, dtype="<f8", order="C", copy=True)
    bits = values.view("<u8")
    bits[np.isnan(values)] = _CANONICAL_NAN_BITS
    magnitude = bits & _MAGNITUDE_BITS
    negated = bits.copy()
    negated[~np.isnan(values)] ^= _SIGN_BITS
    return bits, magnitude, negated


def matrix_hash_descriptor_v4_4(
    matrix: np.ndarray,
    *,
    dates: Sequence[Any],
    symbols: Sequence[Any],
) -> dict[str, Any]:
    normalized_dates = _normalize_dates(dates)
    normalized_symbols = _normalize_symbols(symbols)
    if type(matrix) is not np.ndarray or matrix.dtype != np.dtype(np.float64):
        raise _error("descriptor matrix must be an exact float64 NumPy matrix")
    expected_shape = (len(normalized_dates), len(normalized_symbols))
    if matrix.ndim != 2 or matrix.shape != expected_shape:
        raise _error(f"descriptor matrix shape must be {expected_shape}")
    values = np.array(matrix, dtype=np.float64, order="C", copy=False)
    bits, magnitude, negated = _canonical_matrix_bit_views(values)
    raw = bits.astype("<u8", copy=False).tobytes(order="C")
    magnitude_raw = magnitude.astype("<u8", copy=False).tobytes(order="C")
    negated_raw = negated.astype("<u8", copy=False).tobytes(order="C")
    finite = np.isfinite(values)
    descriptor = {
        "schema_version": MATRIX_DESCRIPTOR_SCHEMA_VERSION,
        "dtype": "float64-le",
        "layout": "row-major",
        "row_count": values.shape[0],
        "column_count": values.shape[1],
        "date_axis": _axis_descriptor(normalized_dates),
        "symbol_axis": _axis_descriptor(normalized_symbols),
        "matrix_sha256": hashlib.sha256(raw).hexdigest(),
        "bit_pattern_sha256": hashlib.sha256(raw).hexdigest(),
        "magnitude_bits_sha256": hashlib.sha256(magnitude_raw).hexdigest(),
        "elementwise_negated_sha256": hashlib.sha256(negated_raw).hexdigest(),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
        "positive_finite_count": int((finite & (values > 0.0)).sum()),
        "negative_finite_count": int((finite & (values < 0.0)).sum()),
        "positive_zero_count": int(
            ((values == 0.0) & ~np.signbit(values)).sum()
        ),
        "negative_zero_count": int(
            ((values == 0.0) & np.signbit(values)).sum()
        ),
        "byte_count": len(raw),
    }
    return descriptor


_MATRIX_DESCRIPTOR_FIELDS = {
    "schema_version",
    "dtype",
    "layout",
    "row_count",
    "column_count",
    "date_axis",
    "symbol_axis",
    "matrix_sha256",
    "bit_pattern_sha256",
    "magnitude_bits_sha256",
    "elementwise_negated_sha256",
    "finite_count",
    "nan_count",
    "positive_infinity_count",
    "negative_infinity_count",
    "positive_finite_count",
    "negative_finite_count",
    "positive_zero_count",
    "negative_zero_count",
    "byte_count",
}


def validate_global_descriptor_v4_4(
    value: Any,
    *,
    expected_dates: Sequence[Any] | None = None,
    expected_symbols: Sequence[Any] | None = None,
    matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("matrix descriptor must be an object")
    payload = copy.deepcopy(dict(value))
    if set(payload) != _MATRIX_DESCRIPTOR_FIELDS:
        raise _error("matrix descriptor fields mismatch")
    if payload["schema_version"] != MATRIX_DESCRIPTOR_SCHEMA_VERSION:
        raise _error("matrix descriptor schema mismatch")
    if payload["dtype"] != "float64-le" or payload["layout"] != "row-major":
        raise _error("matrix descriptor dtype/layout mismatch")
    integer_fields = (
        "row_count",
        "column_count",
        "finite_count",
        "nan_count",
        "positive_infinity_count",
        "negative_infinity_count",
        "positive_finite_count",
        "negative_finite_count",
        "positive_zero_count",
        "negative_zero_count",
        "byte_count",
    )
    for field in integer_fields:
        if type(payload[field]) is not int or payload[field] < 0:
            raise _error(f"matrix descriptor {field} must be a non-negative integer")
    total = payload["row_count"] * payload["column_count"]
    if payload["byte_count"] != total * 8:
        raise _error("matrix descriptor byte count mismatch")
    if (
        payload["finite_count"]
        + payload["nan_count"]
        + payload["positive_infinity_count"]
        + payload["negative_infinity_count"]
        != total
    ):
        raise _error("matrix descriptor value accounting mismatch")
    if (
        payload["positive_finite_count"]
        + payload["negative_finite_count"]
        + payload["positive_zero_count"]
        + payload["negative_zero_count"]
        != payload["finite_count"]
    ):
        raise _error("matrix descriptor finite sign accounting mismatch")
    for field in (
        "matrix_sha256",
        "bit_pattern_sha256",
        "magnitude_bits_sha256",
        "elementwise_negated_sha256",
    ):
        _sha256(payload[field], f"matrix descriptor {field}")
    if payload["matrix_sha256"] != payload["bit_pattern_sha256"]:
        raise _error("matrix and bit-pattern SHA must bind the same canonical bytes")
    for axis_name, count_field in (
        ("date_axis", "row_count"),
        ("symbol_axis", "column_count"),
    ):
        axis = payload[axis_name]
        if not isinstance(axis, Mapping) or set(axis) != {
            "count",
            "sha256",
            "first",
            "last",
        }:
            raise _error(f"matrix descriptor {axis_name} fields mismatch")
        if axis["count"] != payload[count_field]:
            raise _error(f"matrix descriptor {axis_name} count mismatch")
        _sha256(axis["sha256"], f"matrix descriptor {axis_name}.sha256")
        if axis["count"] == 0:
            if axis["first"] is not None or axis["last"] is not None:
                raise _error(f"empty {axis_name} must have null endpoints")
        elif type(axis["first"]) is not str or type(axis["last"]) is not str:
            raise _error(f"non-empty {axis_name} must have string endpoints")
    if expected_dates is not None:
        dates = _normalize_dates(expected_dates)
        if payload["date_axis"] != _axis_descriptor(dates):
            raise _error("matrix descriptor date axis mismatch")
    else:
        dates = None
    if expected_symbols is not None:
        symbols = _normalize_symbols(expected_symbols)
        if payload["symbol_axis"] != _axis_descriptor(symbols):
            raise _error("matrix descriptor symbol axis mismatch")
    else:
        symbols = None
    if matrix is not None:
        if dates is None or symbols is None:
            raise _error("matrix readback requires expected dates and symbols")
        if payload != matrix_hash_descriptor_v4_4(
            matrix, dates=dates, symbols=symbols
        ):
            raise _error("matrix descriptor readback mismatch")
    return payload


class StreamingMatrixDescriptorV4_4:
    """Incrementally hash chronological row blocks without concatenating them."""

    def __init__(self, symbols: Sequence[Any]) -> None:
        self._symbols = _normalize_symbols(symbols)
        self._matrix_hash = hashlib.sha256()
        self._magnitude_hash = hashlib.sha256()
        self._negated_hash = hashlib.sha256()
        self._date_hash = hashlib.sha256()
        self._row_count = 0
        self._first_date: str | None = None
        self._last_date: str | None = None
        self._finite_count = 0
        self._nan_count = 0
        self._positive_infinity_count = 0
        self._negative_infinity_count = 0
        self._positive_finite_count = 0
        self._negative_finite_count = 0
        self._positive_zero_count = 0
        self._negative_zero_count = 0
        self._finalized = False

    def update(self, dates: Sequence[Any], matrix: np.ndarray) -> None:
        if self._finalized:
            raise _error("streaming descriptor is already finalized")
        chunk_dates = _normalize_dates(dates, "chunk_dates")
        if self._last_date is not None and chunk_dates[0] <= self._last_date:
            raise _error("streaming descriptor chunks must be strictly chronological")
        if type(matrix) is not np.ndarray or matrix.dtype != np.dtype(np.float64):
            raise _error("streaming descriptor chunk must be exact float64")
        expected_shape = (len(chunk_dates), len(self._symbols))
        if matrix.ndim != 2 or matrix.shape != expected_shape:
            raise _error(f"streaming descriptor chunk shape must be {expected_shape}")
        values = np.array(matrix, dtype=np.float64, order="C", copy=False)
        bits, magnitude, negated = _canonical_matrix_bit_views(values)
        raw = bits.astype("<u8", copy=False).tobytes(order="C")
        self._matrix_hash.update(raw)
        self._magnitude_hash.update(
            magnitude.astype("<u8", copy=False).tobytes(order="C")
        )
        self._negated_hash.update(
            negated.astype("<u8", copy=False).tobytes(order="C")
        )
        for item in chunk_dates:
            self._date_hash.update(item.encode("utf-8") + b"\n")
        finite = np.isfinite(values)
        self._finite_count += int(finite.sum())
        self._nan_count += int(np.isnan(values).sum())
        self._positive_infinity_count += int(np.isposinf(values).sum())
        self._negative_infinity_count += int(np.isneginf(values).sum())
        self._positive_finite_count += int((finite & (values > 0.0)).sum())
        self._negative_finite_count += int((finite & (values < 0.0)).sum())
        self._positive_zero_count += int(
            ((values == 0.0) & ~np.signbit(values)).sum()
        )
        self._negative_zero_count += int(
            ((values == 0.0) & np.signbit(values)).sum()
        )
        if self._first_date is None:
            self._first_date = chunk_dates[0]
        self._last_date = chunk_dates[-1]
        self._row_count += len(chunk_dates)

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise _error("streaming descriptor is already finalized")
        if self._row_count <= 0:
            raise _error("streaming descriptor requires at least one row")
        self._finalized = True
        matrix_sha = self._matrix_hash.hexdigest()
        descriptor = {
            "schema_version": MATRIX_DESCRIPTOR_SCHEMA_VERSION,
            "dtype": "float64-le",
            "layout": "row-major",
            "row_count": self._row_count,
            "column_count": len(self._symbols),
            "date_axis": {
                "count": self._row_count,
                "sha256": self._date_hash.hexdigest(),
                "first": self._first_date,
                "last": self._last_date,
            },
            "symbol_axis": _axis_descriptor(self._symbols),
            "matrix_sha256": matrix_sha,
            "bit_pattern_sha256": matrix_sha,
            "magnitude_bits_sha256": self._magnitude_hash.hexdigest(),
            "elementwise_negated_sha256": self._negated_hash.hexdigest(),
            "finite_count": self._finite_count,
            "nan_count": self._nan_count,
            "positive_infinity_count": self._positive_infinity_count,
            "negative_infinity_count": self._negative_infinity_count,
            "positive_finite_count": self._positive_finite_count,
            "negative_finite_count": self._negative_finite_count,
            "positive_zero_count": self._positive_zero_count,
            "negative_zero_count": self._negative_zero_count,
            "byte_count": self._row_count * len(self._symbols) * 8,
        }
        return validate_global_descriptor_v4_4(
            descriptor,
            expected_symbols=self._symbols,
        )


def build_streaming_global_descriptors_v4_4(
    chunks: Iterable[tuple[Sequence[Any], Mapping[str, np.ndarray]]],
    *,
    symbols: Sequence[Any],
    expected_dates: Sequence[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build one global descriptor per factor from ordered non-halo chunks."""

    states = {name: StreamingMatrixDescriptorV4_4(symbols) for name in FACTOR_NAMES}
    observed_dates: list[str] = []
    chunk_count = 0
    for chunk_dates_raw, outputs in chunks:
        chunk_dates = _normalize_dates(chunk_dates_raw, "chunk_dates")
        if not isinstance(outputs, Mapping) or tuple(outputs) != FACTOR_NAMES:
            raise _error("streaming outputs must contain the exact ordered five factors")
        for name in FACTOR_NAMES:
            states[name].update(chunk_dates, outputs[name])
        observed_dates.extend(chunk_dates)
        chunk_count += 1
    if chunk_count == 0:
        raise _error("streaming descriptors require at least one chunk")
    if expected_dates is not None:
        expected = _normalize_dates(expected_dates, "expected_dates")
        if tuple(observed_dates) != expected:
            raise _error("streaming chunks do not cover the expected dates exactly once")
    return {name: states[name].finalize() for name in FACTOR_NAMES}


def _evaluate_pandas_validated_operator_program_v4_4(
    input_block: InputBlockV4_4,
    operator_program_set: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Interpret one input block with a fresh pandas rolling state."""

    block = validate_input_block_v4_4(input_block)
    index = pd.DatetimeIndex(block.dates, name="trade_date")
    columns = pd.Index(block.symbols, name="ts_code")
    pit = pd.DataFrame(
        np.array(block.pit_mask, dtype=bool, copy=True),
        index=index,
        columns=columns,
    )

    def pandas_node(frame: pd.DataFrame) -> pd.DataFrame:
        node = frame.astype(np.float64).where(pit)
        return node.mask(np.isinf(node))

    outputs: dict[str, np.ndarray] = {}
    for program in operator_program_set["candidates"]:
        nodes: dict[str, pd.DataFrame] = {}
        for instruction in program["nodes"]:
            node_id = instruction["node_id"]
            opcode = instruction["opcode"]
            input_ids = instruction["inputs"]
            parameters = instruction["parameters"]
            inputs = [nodes[input_id] for input_id in input_ids]

            if opcode == "source":
                value = pd.DataFrame(
                    np.array(
                        getattr(block, parameters["canonical_input"]),
                        dtype=np.float64,
                        copy=True,
                    ),
                    index=index,
                    columns=columns,
                )
            elif opcode == "constant":
                constant = struct.unpack(
                    ">d", bytes.fromhex(parameters["float64_be_hex"])
                )[0]
                value = pd.DataFrame(
                    np.full(pit.shape, constant, dtype=np.float64),
                    index=index,
                    columns=columns,
                )
            elif opcode == "shift":
                value = inputs[0].shift(periods=parameters["periods"])
            elif opcode == "add":
                value = inputs[0] + inputs[1]
            elif opcode == "subtract":
                value = inputs[0] - inputs[1]
            elif opcode == "multiply":
                value = inputs[0] * inputs[1]
            elif opcode == "divide":
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    value = inputs[0] / inputs[1]
            elif opcode == "absolute":
                value = inputs[0].abs()
            elif opcode == "sign":
                value = np.sign(inputs[0])
            elif opcode == "rolling_min":
                value = inputs[0].rolling(
                    window=parameters["window"],
                    min_periods=parameters["min_periods"],
                ).min()
            elif opcode == "rolling_max":
                value = inputs[0].rolling(
                    window=parameters["window"],
                    min_periods=parameters["min_periods"],
                ).max()
            elif opcode == "rolling_mean":
                value = inputs[0].rolling(
                    window=parameters["window"],
                    min_periods=parameters["min_periods"],
                ).mean()
            elif opcode == "rolling_std":
                value = inputs[0].rolling(
                    window=parameters["window"],
                    min_periods=parameters["min_periods"],
                ).std(ddof=parameters["ddof"])
            elif opcode == "cross_section_rank":
                if parameters["axis"] != "symbols":
                    raise _error("pandas rank axis must be symbols")
                value = inputs[0].rank(
                    axis=1,
                    method=parameters["method"],
                    na_option=parameters["na_option"],
                    pct=parameters["pct"],
                    ascending=parameters["ascending"],
                )
            else:
                raise _error(f"unsupported pandas opcode: {opcode!r}")
            nodes[node_id] = pandas_node(value)

        values = np.array(
            nodes[program["output_node_id"]].to_numpy(dtype=np.float64),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        values.setflags(write=False)
        outputs[program["name"]] = values
    return outputs


def _evaluate_numpy_validated_operator_program_v4_4(
    input_block: InputBlockV4_4,
    operator_program_set: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Interpret one input block with a fresh NumPy rolling state.

    The local rolling code reproduces pandas' compensated fixed-window mean
    and variance state transitions.  It shares no math helper with the pandas
    dispatcher.
    """

    block = validate_input_block_v4_4(input_block)
    row_count = len(block.dates)
    column_count = len(block.symbols)
    local_pit = np.array(block.pit_mask, dtype=bool, order="C", copy=True)

    # This helper is private to the NumPy engine and is independently applied
    # after every NumPy source/derived node.
    def numpy_node(value: np.ndarray) -> np.ndarray:
        node = np.array(value, dtype=np.float64, order="C", copy=True)
        node[~local_pit] = np.nan
        node[np.isinf(node)] = np.nan
        return node

    def numpy_rolling_extreme(
        source: np.ndarray,
        *,
        window: int,
        min_periods: int,
        choose_minimum: bool,
    ) -> np.ndarray:
        result = np.full(source.shape, np.nan, dtype=np.float64)
        for row in range(row_count):
            start = max(row - window + 1, 0)
            for column in range(column_count):
                selected = np.float64(np.nan)
                observation_count = 0
                for position in range(start, row + 1):
                    candidate = np.float64(source[position, column])
                    if candidate != candidate:
                        continue
                    observation_count += 1
                    if selected != selected:
                        selected = candidate
                    elif choose_minimum and candidate <= selected:
                        selected = candidate
                    elif not choose_minimum and candidate >= selected:
                        selected = candidate
                if observation_count >= min_periods:
                    result[row, column] = selected
        return result

    def numpy_rolling_mean(
        source: np.ndarray, window: int, min_periods: int
    ) -> np.ndarray:
        result = np.full(source.shape, np.nan, dtype=np.float64)
        for column in range(column_count):
            observation_count = 0
            negative_count = 0
            running_sum = np.float64(0.0)
            add_compensation = np.float64(0.0)
            remove_compensation = np.float64(0.0)
            previous_value = np.float64(source[0, column])
            consecutive_same = 0
            previous_start = 0
            previous_end = 0
            for row in range(row_count):
                start = max(row - window + 1, 0)
                end = row + 1
                if row == 0 or start >= previous_end:
                    observation_count = 0
                    negative_count = 0
                    running_sum = np.float64(0.0)
                    add_compensation = np.float64(0.0)
                    remove_compensation = np.float64(0.0)
                    previous_value = np.float64(source[start, column])
                    consecutive_same = 0
                    add_start = start
                else:
                    for position in range(previous_start, start):
                        removed = np.float64(source[position, column])
                        if removed == removed:
                            observation_count -= 1
                            y = np.float64(-removed - remove_compensation)
                            total = np.float64(running_sum + y)
                            remove_compensation = np.float64(
                                total - running_sum - y
                            )
                            running_sum = total
                            if np.signbit(removed):
                                negative_count -= 1
                    add_start = previous_end
                for position in range(add_start, end):
                    added = np.float64(source[position, column])
                    if added == added:
                        observation_count += 1
                        y = np.float64(added - add_compensation)
                        total = np.float64(running_sum + y)
                        add_compensation = np.float64(total - running_sum - y)
                        running_sum = total
                        if np.signbit(added):
                            negative_count += 1
                        if added == previous_value:
                            consecutive_same += 1
                        else:
                            consecutive_same = 1
                            previous_value = added
                if observation_count >= min_periods and observation_count > 0:
                    calculated = np.float64(
                        running_sum / np.float64(observation_count)
                    )
                    if consecutive_same >= observation_count:
                        calculated = previous_value
                    elif negative_count == 0 and calculated < 0.0:
                        calculated = np.float64(0.0)
                    elif negative_count == observation_count and calculated > 0.0:
                        calculated = np.float64(0.0)
                    result[row, column] = calculated
                previous_start = start
                previous_end = end
        return result

    def numpy_rolling_std(
        source: np.ndarray, window: int, min_periods: int, ddof: int
    ) -> np.ndarray:
        variance = np.full(source.shape, np.nan, dtype=np.float64)
        instability_tolerance = np.float64(np.finfo(np.float64).eps * 1e3)
        for column in range(column_count):
            observation_count = np.float64(0.0)
            running_mean = np.float64(0.0)
            squared_deviations = np.float64(0.0)
            add_compensation = np.float64(0.0)
            remove_compensation = np.float64(0.0)
            numerically_unstable = False
            previous_start = 0
            previous_end = 0
            for row in range(row_count):
                start = max(row - window + 1, 0)
                end = row + 1
                requires_recompute = row == 0 or start >= previous_end
                if not requires_recompute:
                    for position in range(previous_start, start):
                        removed = np.float64(source[position, column])
                        if removed == removed:
                            previous_m2 = squared_deviations
                            observation_count = np.float64(
                                observation_count - 1.0
                            )
                            if observation_count:
                                previous_mean = np.float64(
                                    running_mean - remove_compensation
                                )
                                y = np.float64(removed - remove_compensation)
                                total = np.float64(y - running_mean)
                                remove_compensation = np.float64(
                                    total + running_mean - y
                                )
                                running_mean = np.float64(
                                    running_mean - total / observation_count
                                )
                                squared_deviations = np.float64(
                                    _fma(
                                        -(removed - previous_mean),
                                        removed - running_mean,
                                        squared_deviations,
                                    )
                                )
                                if (
                                    previous_m2 * instability_tolerance
                                    > squared_deviations
                                ):
                                    numerically_unstable = True
                            else:
                                running_mean = np.float64(0.0)
                                squared_deviations = np.float64(0.0)
                                numerically_unstable = False
                    for position in range(previous_end, end):
                        added = np.float64(source[position, column])
                        if added != added:
                            continue
                        previous_m2 = squared_deviations
                        observation_count = np.float64(observation_count + 1.0)
                        previous_mean = np.float64(
                            running_mean - add_compensation
                        )
                        y = np.float64(added - add_compensation)
                        total = np.float64(y - running_mean)
                        add_compensation = np.float64(total + running_mean - y)
                        running_mean = np.float64(
                            running_mean + total / observation_count
                        )
                        squared_deviations = np.float64(
                            _fma(
                                added - previous_mean,
                                added - running_mean,
                                squared_deviations,
                            )
                        )
                        if previous_m2 * instability_tolerance > squared_deviations:
                            numerically_unstable = True
                if requires_recompute or numerically_unstable:
                    observation_count = np.float64(0.0)
                    running_mean = np.float64(0.0)
                    squared_deviations = np.float64(0.0)
                    add_compensation = np.float64(0.0)
                    remove_compensation = np.float64(0.0)
                    for position in range(start, end):
                        added = np.float64(source[position, column])
                        if added != added:
                            continue
                        previous_m2 = squared_deviations
                        observation_count = np.float64(observation_count + 1.0)
                        previous_mean = np.float64(
                            running_mean - add_compensation
                        )
                        y = np.float64(added - add_compensation)
                        total = np.float64(y - running_mean)
                        add_compensation = np.float64(total + running_mean - y)
                        running_mean = np.float64(
                            running_mean + total / observation_count
                        )
                        squared_deviations = np.float64(
                            _fma(
                                added - previous_mean,
                                added - running_mean,
                                squared_deviations,
                            )
                        )
                        if previous_m2 * instability_tolerance > squared_deviations:
                            numerically_unstable = True
                    numerically_unstable = False
                if (
                    observation_count >= min_periods
                    and observation_count > np.float64(ddof)
                ):
                    variance[row, column] = np.float64(
                        squared_deviations
                        / (observation_count - np.float64(ddof))
                    )
                previous_start = start
                previous_end = end
        with np.errstate(invalid="ignore"):
            result = np.sqrt(variance)
        result[variance < 0.0] = 0.0
        return np.array(result, dtype=np.float64, order="C", copy=False)

    outputs: dict[str, np.ndarray] = {}
    for program in operator_program_set["candidates"]:
        nodes: dict[str, np.ndarray] = {}
        for instruction in program["nodes"]:
            node_id = instruction["node_id"]
            opcode = instruction["opcode"]
            input_ids = instruction["inputs"]
            parameters = instruction["parameters"]
            inputs = [nodes[input_id] for input_id in input_ids]

            if opcode == "source":
                value = np.array(
                    getattr(block, parameters["canonical_input"]),
                    dtype=np.float64,
                    order="C",
                    copy=True,
                )
            elif opcode == "constant":
                constant = struct.unpack(
                    ">d", bytes.fromhex(parameters["float64_be_hex"])
                )[0]
                value = np.full(
                    (row_count, column_count), constant, dtype=np.float64
                )
            elif opcode == "shift":
                periods = parameters["periods"]
                value = np.full(
                    (row_count, column_count), np.nan, dtype=np.float64
                )
                if periods < row_count:
                    value[periods:] = inputs[0][:-periods]
            elif opcode == "add":
                value = inputs[0] + inputs[1]
            elif opcode == "subtract":
                value = inputs[0] - inputs[1]
            elif opcode == "multiply":
                value = inputs[0] * inputs[1]
            elif opcode == "divide":
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    value = inputs[0] / inputs[1]
            elif opcode == "absolute":
                value = np.abs(inputs[0])
            elif opcode == "sign":
                value = np.sign(inputs[0])
            elif opcode == "rolling_min":
                value = numpy_rolling_extreme(
                    inputs[0],
                    window=parameters["window"],
                    min_periods=parameters["min_periods"],
                    choose_minimum=True,
                )
            elif opcode == "rolling_max":
                value = numpy_rolling_extreme(
                    inputs[0],
                    window=parameters["window"],
                    min_periods=parameters["min_periods"],
                    choose_minimum=False,
                )
            elif opcode == "rolling_mean":
                value = numpy_rolling_mean(
                    inputs[0],
                    window=parameters["window"],
                    min_periods=parameters["min_periods"],
                )
            elif opcode == "rolling_std":
                value = numpy_rolling_std(
                    inputs[0],
                    window=parameters["window"],
                    min_periods=parameters["min_periods"],
                    ddof=parameters["ddof"],
                )
            elif opcode == "cross_section_rank":
                if (
                    parameters["axis"] != "symbols"
                    or parameters["method"] != "average"
                    or parameters["na_option"] != "keep"
                    or parameters["pct"] is not True
                ):
                    raise _error("NumPy rank parameters are unsupported")
                value = np.full(
                    (row_count, column_count), np.nan, dtype=np.float64
                )
                ascending = parameters["ascending"]
                for row in range(row_count):
                    valid_positions = np.flatnonzero(~np.isnan(inputs[0][row]))
                    if valid_positions.size == 0:
                        continue
                    sort_values = inputs[0][row, valid_positions]
                    if not ascending:
                        sort_values = -sort_values
                    ordered = valid_positions[
                        np.argsort(sort_values, kind="stable")
                    ]
                    start = 0
                    while start < ordered.size:
                        end = start + 1
                        tie_value = inputs[0][row, ordered[start]]
                        while (
                            end < ordered.size
                            and inputs[0][row, ordered[end]] == tie_value
                        ):
                            end += 1
                        average_rank = np.float64(
                            (np.float64(start + 1) + np.float64(end)) / 2.0
                        )
                        value[row, ordered[start:end]] = np.float64(
                            average_rank / np.float64(valid_positions.size)
                        )
                        start = end
            else:
                raise _error(f"unsupported NumPy opcode: {opcode!r}")
            nodes[node_id] = numpy_node(value)

        values = np.array(
            nodes[program["output_node_id"]],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        values.setflags(write=False)
        outputs[program["name"]] = values
    return outputs


def evaluate_pandas_engine_v4_4(
    input_block: InputBlockV4_4,
    *,
    operator_program_set: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate and execute the golden operator program on one input block."""

    validated_program = strict_contract.validate_operator_program_set_v4_4(
        operator_program_set
    )
    return _evaluate_pandas_validated_operator_program_v4_4(
        input_block, validated_program
    )


def evaluate_numpy_engine_v4_4(
    input_block: InputBlockV4_4,
    *,
    operator_program_set: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate and execute the golden operator program on one input block."""

    validated_program = strict_contract.validate_operator_program_set_v4_4(
        operator_program_set
    )
    return _evaluate_numpy_validated_operator_program_v4_4(
        input_block, validated_program
    )


def validate_factor_outputs_v4_4(
    outputs: Any,
    input_block: InputBlockV4_4,
    *,
    require_positive_proof: bool = True,
) -> dict[str, np.ndarray]:
    block = validate_input_block_v4_4(input_block)
    if not isinstance(outputs, Mapping) or tuple(outputs) != FACTOR_NAMES:
        raise _error("outputs must contain the exact ordered five factors")
    if type(require_positive_proof) is not bool:
        raise _error("require_positive_proof must be bool")
    shape = (len(block.dates), len(block.symbols))
    outside_pit = ~block.pit_mask
    inside_pit = block.pit_mask
    normalized: dict[str, np.ndarray] = {}
    for name in FACTOR_NAMES:
        value = outputs[name]
        if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
            raise _error(f"{name} output must be an exact float64 NumPy matrix")
        if value.ndim != 2 or value.shape != shape:
            raise _error(f"{name} output shape must be {shape}")
        if np.isinf(value).any():
            raise _error(f"{name} output contains positive or negative infinity")
        if np.any(~np.isnan(value[outside_pit])):
            raise _error(f"{name} output contains a value outside PIT")
        if require_positive_proof and not np.isfinite(value[inside_pit]).any():
            raise _error(f"{name} has no finite in-PIT value for a positive proof")
        normalized[name] = value
    return normalized


def compare_exact_matrices_v4_4(
    left: np.ndarray, right: np.ndarray, *, matrix_name: str = "matrix"
) -> dict[str, Any]:
    """Compare float64 matrices without tolerance or NaN-payload dependence."""

    if (
        type(left) is not np.ndarray
        or left.dtype != np.dtype(np.float64)
        or left.ndim != 2
        or type(right) is not np.ndarray
        or right.dtype != np.dtype(np.float64)
        or right.ndim != 2
    ):
        raise _error(
            f"{matrix_name} comparison requires two float64 NumPy matrices"
        )
    if left.shape != right.shape:
        raise _error(f"{matrix_name} comparison shape mismatch")
    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    left_positive_inf = np.isposinf(left)
    right_positive_inf = np.isposinf(right)
    left_negative_inf = np.isneginf(left)
    right_negative_inf = np.isneginf(right)
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    jointly_finite = left_finite & right_finite
    result = {
        "nan_mask_equal": bool(np.array_equal(left_nan, right_nan)),
        "positive_infinity_mask_equal": bool(
            np.array_equal(left_positive_inf, right_positive_inf)
        ),
        "negative_infinity_mask_equal": bool(
            np.array_equal(left_negative_inf, right_negative_inf)
        ),
        "finite_mask_equal": bool(np.array_equal(left_finite, right_finite)),
        "finite_bits_equal": bool(
            np.array_equal(
                left.view(np.uint64)[jointly_finite],
                right.view(np.uint64)[jointly_finite],
            )
        ),
        "tolerance_used": False,
        "rounding_used": False,
        "exact": False,
    }
    result["exact"] = all(
        result[field]
        for field in (
            "nan_mask_equal",
            "positive_infinity_mask_equal",
            "negative_infinity_mask_equal",
            "finite_mask_equal",
            "finite_bits_equal",
        )
    )
    if not result["exact"]:
        raise _error(f"exact matrix divergence for {matrix_name}")
    return result


def compare_exact_engine_outputs_v4_4(
    pandas_outputs: Any,
    numpy_outputs: Any,
    input_block: InputBlockV4_4,
    *,
    require_positive_proof: bool = True,
) -> dict[str, Any]:
    """Require exact finite bits, NaN/inf masks, and signed-zero identity."""

    block = validate_input_block_v4_4(input_block)
    source = validate_factor_outputs_v4_4(
        pandas_outputs,
        block,
        require_positive_proof=require_positive_proof,
    )
    local = validate_factor_outputs_v4_4(
        numpy_outputs,
        block,
        require_positive_proof=require_positive_proof,
    )
    rows: list[dict[str, Any]] = []
    for order, name in enumerate(FACTOR_NAMES, start=1):
        left = source[name]
        right = local[name]
        exact = compare_exact_matrices_v4_4(left, right, matrix_name=name)
        row = {
            "order": order,
            "name": name,
            "direction": FACTOR_DIRECTIONS[name],
            **exact,
        }
        rows.append(row)
    comparison: dict[str, Any] = {
        "schema_version": EXACT_COMPARISON_SCHEMA_VERSION,
        "pandas_engine_id": PANDAS_ENGINE_ID,
        "numpy_engine_id": NUMPY_ENGINE_ID,
        "factor_count": len(FACTOR_NAMES),
        "factors": rows,
        "tolerance_used": False,
        "rounding_used": False,
        "exact": True,
    }
    comparison["comparison_semantic_sha256"] = semantic_sha256_v4_4(comparison)
    return comparison


def slice_non_halo_outputs_v4_4(
    outputs: Mapping[str, np.ndarray],
    block_row: Mapping[str, Any],
    *,
    source_block: InputBlockV4_4,
) -> dict[str, np.ndarray]:
    """Slice outputs only after binding their row to the producing input block."""

    if not isinstance(outputs, Mapping) or tuple(outputs) != FACTOR_NAMES:
        raise _error("outputs must contain the exact ordered five factors")
    row, block = _validate_block_row_against_input_block_v4_4(
        block_row, source_block
    )
    local_start = row["local_output_start_offset"]
    local_end = row["local_output_end_offset"]
    expected_shape = (len(block.dates), len(block.symbols))
    result: dict[str, np.ndarray] = {}
    for name in FACTOR_NAMES:
        matrix = outputs[name]
        if (
            type(matrix) is not np.ndarray
            or matrix.dtype != np.dtype(np.float64)
            or matrix.ndim != 2
            or matrix.shape != expected_shape
        ):
            raise _error(f"{name} block output shape is invalid")
        selected = np.array(
            matrix[local_start:local_end], dtype=np.float64, order="C", copy=True
        )
        selected.setflags(write=False)
        result[name] = selected
    return result


__all__ = [
    "BLOCK_MANIFEST_SCHEMA_VERSION",
    "EVIDENCE_CONTRACT_VERSION",
    "EXACT_COMPARISON_SCHEMA_VERSION",
    "FACTOR_DIRECTIONS",
    "FACTOR_NAMES",
    "FactorGovernanceFutureStrictExactFiveEvalV4_4Error",
    "HALO",
    "INPUT_FIELDS",
    "InputBlockV4_4",
    "MATRIX_DESCRIPTOR_SCHEMA_VERSION",
    "NUMPY_ENGINE_ID",
    "OUTPUT_BLOCK",
    "PANDAS_ENGINE_ID",
    "PROTOCOL_VERSION",
    "StreamingMatrixDescriptorV4_4",
    "build_block_manifest_v4_4",
    "build_input_block_v4_4",
    "build_streaming_global_descriptors_v4_4",
    "canonical_json_bytes_v4_4",
    "compare_exact_matrices_v4_4",
    "compare_exact_engine_outputs_v4_4",
    "evaluate_numpy_engine_v4_4",
    "evaluate_pandas_engine_v4_4",
    "matrix_hash_descriptor_v4_4",
    "semantic_sha256_v4_4",
    "slice_input_block_v4_4",
    "slice_non_halo_outputs_v4_4",
    "validate_block_manifest_v4_4",
    "validate_factor_outputs_v4_4",
    "validate_global_descriptor_v4_4",
    "validate_input_block_v4_4",
]
