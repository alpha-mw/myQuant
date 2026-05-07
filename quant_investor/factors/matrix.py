"""Offline matrix data contracts for factor research.

The contracts in this module are deliberately in-memory and append-store
friendly. They do not load vendor data, run backtests, or connect factor values
to live selection or portfolio construction paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.versioning import (
    FACTOR_EXPRESSION_SCHEMA_VERSION,
    FACTOR_MATRIX_SCHEMA_VERSION,
)


FIELD_OPEN = "open"
FIELD_HIGH = "high"
FIELD_LOW = "low"
FIELD_CLOSE = "close"
FIELD_VOLUME = "volume"
FIELD_AMOUNT = "amount"
FIELD_INDUSTRY = "industry"
FIELD_BENCHMARK_CLOSE = "benchmark_close"
FIELD_BENCHMARK_RET = "benchmark_ret"
FIELD_BENCHMARK_WEIGHT = "benchmark_weight"
FIELD_VWAP = "vwap"
FIELD_RET1 = "ret1"

STANDARD_MATRIX_FIELDS = {
    FIELD_OPEN,
    FIELD_HIGH,
    FIELD_LOW,
    FIELD_CLOSE,
    FIELD_VOLUME,
    FIELD_AMOUNT,
    FIELD_INDUSTRY,
    FIELD_BENCHMARK_CLOSE,
    FIELD_BENCHMARK_RET,
    FIELD_BENCHMARK_WEIGHT,
    FIELD_VWAP,
    FIELD_RET1,
}

DEFAULT_FACTOR_MATRIX_DIR = Path("data/factor_library/matrix")
DEFAULT_MATRIX_CONTRACTS_FILENAME = "matrix_contracts.jsonl"
DEFAULT_MATRIX_BUNDLES_FILENAME = "matrix_bundles.jsonl"
DEFAULT_FACTOR_MATRICES_FILENAME = "factor_matrices.jsonl"
DEFAULT_EXPRESSION_RESULTS_FILENAME = "expression_results.jsonl"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ensure_json_serializable(value: Any, label: str) -> Any:
    safe = _json_safe(value)
    try:
        json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON-serializable values.") from exc
    return safe


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, "metadata"))


def _non_empty_str(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _ordered_unique(values: Sequence[Any], *, sort_values: bool = True) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return sorted(result) if sort_values else result


def _reject_duplicates(values: Sequence[Any], field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text in seen:
            raise ValueError(f"{field_name} must contain unique values; duplicate {text!r}.")
        seen.add(text)


def _coerce_symbols(values: Sequence[Any], *, metadata: Mapping[str, Any]) -> list[str]:
    raw = [str(value).strip() for value in values if str(value).strip()]
    if not raw:
        raise ValueError("symbols must be non-empty.")
    _reject_duplicates(raw, "symbols")
    preserve_order = bool(metadata.get("preserve_symbol_order", False))
    return raw if preserve_order else sorted(raw)


def _coerce_dates(values: Sequence[Any]) -> list[str]:
    dates = [str(value).strip() for value in values if str(value).strip()]
    if not dates:
        raise ValueError("dates must be non-empty.")
    parsed: list[date] = []
    for value in dates:
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"dates must be ISO dates; got {value!r}.") from exc
        if parsed_value.isoformat() != value:
            raise ValueError(f"dates must be canonical ISO dates; got {value!r}.")
        parsed.append(parsed_value)
    if any(current >= next_value for current, next_value in zip(parsed, parsed[1:])):
        raise ValueError("dates must be strictly ascending ISO dates.")
    return dates


def _coerce_unit_float(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _to_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _coerce_factor_value(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} values must be finite numeric values or None.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} values must be finite numeric values or None.")
    return number


def _matrix_dimensions(symbols: Sequence[str], dates: Sequence[str]) -> tuple[int, int]:
    return len(symbols), len(dates)


def _is_sequence_row(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_matrix(values: Sequence[Sequence[Any]], *, field_name: str) -> list[list[Any]]:
    if not _is_sequence_row(values):
        raise ValueError(f"{field_name} must be a matrix.")
    rows: list[list[Any]] = []
    for row in values:
        if not _is_sequence_row(row):
            raise ValueError(f"{field_name} must be a matrix.")
        rows.append([_json_safe(item) for item in row])
    return rows


def _coerce_numeric_matrix(
    values: Sequence[Sequence[Any]],
    *,
    field_name: str,
) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    if not _is_sequence_row(values):
        raise ValueError(f"{field_name} must be a matrix.")
    for row in values:
        if not _is_sequence_row(row):
            raise ValueError(f"{field_name} must be a matrix.")
        rows.append([_coerce_factor_value(item, field_name) for item in row])
    return rows


def _coerce_bool_matrix(values: Sequence[Sequence[Any]], *, field_name: str) -> list[list[bool]]:
    rows: list[list[bool]] = []
    if not _is_sequence_row(values):
        raise ValueError(f"{field_name} must be a matrix.")
    for row in values:
        if not _is_sequence_row(row):
            raise ValueError(f"{field_name} must be a matrix.")
        bool_row: list[bool] = []
        for item in row:
            if not isinstance(item, bool):
                raise ValueError(f"{field_name} values must be bool.")
            bool_row.append(item)
        rows.append(bool_row)
    return rows


def _validate_shape_for(
    values: Sequence[Sequence[Any]],
    *,
    rows: int,
    columns: int,
    field_name: str,
) -> None:
    if len(values) != rows:
        raise ValueError(f"{field_name} must have {rows} rows; got {len(values)}.")
    for row_index, row in enumerate(values):
        if not _is_sequence_row(row):
            raise ValueError(f"{field_name} row {row_index} must be a sequence.")
        if len(row) != columns:
            raise ValueError(
                f"{field_name} row {row_index} must have {columns} columns; got {len(row)}."
            )


def _slug(value: str | None) -> str:
    resolved = "none" if value is None else str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", resolved)
    return slug.strip("-") or "unknown"


def _short_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps(
        [_json_safe(part) for part in parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class MatrixDataContract:
    schema_version: str = FACTOR_MATRIX_SCHEMA_VERSION
    contract_id: str = ""
    universe: str = ""
    benchmark: str | None = None
    symbols: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    optional_fields: list[str] = field(default_factory=list)
    field_sources: dict[str, str] = field(default_factory=dict)
    point_in_time_flags: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_MATRIX_SCHEMA_VERSION)
        self.contract_id = _non_empty_str(self.contract_id, "contract_id")
        self.universe = _non_empty_str(self.universe, "universe")
        self.benchmark = _optional_str(self.benchmark)
        self.metadata = _coerce_metadata(self.metadata)
        self.symbols = _coerce_symbols(self.symbols, metadata=self.metadata)
        self.dates = _coerce_dates(self.dates)
        self.required_fields = _ordered_unique(self.required_fields)
        self.optional_fields = _ordered_unique(self.optional_fields)
        known_field_keys = (
            STANDARD_MATRIX_FIELDS
            | set(self.required_fields)
            | set(self.optional_fields)
        )
        self.field_sources = {
            str(key).strip(): str(value)
            for key, value in sorted(self.field_sources.items(), key=lambda item: str(item[0]))
            if str(key).strip()
        }
        for field_name in self.field_sources:
            if field_name not in known_field_keys:
                raise ValueError(f"field_sources contains unknown field {field_name!r}.")
        self.point_in_time_flags = {
            str(key).strip(): value
            for key, value in sorted(
                self.point_in_time_flags.items(),
                key=lambda item: str(item[0]),
            )
            if str(key).strip()
        }
        for field_name, value in self.point_in_time_flags.items():
            if field_name not in known_field_keys:
                raise ValueError(f"point_in_time_flags contains unknown field {field_name!r}.")
            if not isinstance(value, bool):
                raise ValueError(f"point_in_time_flags.{field_name} must be bool.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "universe": self.universe,
            "benchmark": self.benchmark,
            "symbols": list(self.symbols),
            "dates": list(self.dates),
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields),
            "field_sources": dict(_json_safe(self.field_sources)),
            "point_in_time_flags": dict(_json_safe(self.point_in_time_flags)),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MatrixDataContract":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_MATRIX_SCHEMA_VERSION)),
            contract_id=str(data.get("contract_id", "")),
            universe=str(data.get("universe", "")),
            benchmark=data.get("benchmark"),
            symbols=list(data.get("symbols", []) or []),
            dates=list(data.get("dates", []) or []),
            required_fields=list(data.get("required_fields", []) or []),
            optional_fields=list(data.get("optional_fields", []) or []),
            field_sources=dict(data.get("field_sources", {}) or {}),
            point_in_time_flags=dict(data.get("point_in_time_flags", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class MatrixDataBundle:
    schema_version: str = FACTOR_MATRIX_SCHEMA_VERSION
    bundle_id: str = ""
    contract: MatrixDataContract = field(default_factory=MatrixDataContract)
    fields: dict[str, list[list[float | int | str | None]]] = field(default_factory=dict)
    universe_mask: list[list[bool]] | None = None
    tradability_mask: list[list[bool]] | None = None
    industry_by_symbol: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_MATRIX_SCHEMA_VERSION)
        self.bundle_id = _non_empty_str(self.bundle_id, "bundle_id")
        if not isinstance(self.contract, MatrixDataContract):
            self.contract = MatrixDataContract.from_dict(self.contract)
        rows, columns = _matrix_dimensions(self.contract.symbols, self.contract.dates)
        self.metadata = _coerce_metadata(self.metadata)
        self.fields = {
            str(field_name): _coerce_matrix(values, field_name=str(field_name))
            for field_name, values in sorted(self.fields.items(), key=lambda item: str(item[0]))
        }
        for field_name, values in self.fields.items():
            self.validate_shape(values, field_name=field_name)
        self.industry_by_symbol = {
            str(symbol): str(industry)
            for symbol, industry in sorted(
                self.industry_by_symbol.items(),
                key=lambda item: str(item[0]),
            )
            if str(symbol)
        }
        for required_field in self.contract.required_fields:
            if required_field == FIELD_INDUSTRY and required_field not in self.fields:
                missing_symbols = [
                    symbol
                    for symbol in self.contract.symbols
                    if symbol not in self.industry_by_symbol
                ]
                if not missing_symbols:
                    continue
            if required_field not in self.fields:
                raise ValueError(f"required field {required_field!r} is missing from bundle.")
        if self.universe_mask is not None:
            self.universe_mask = _coerce_bool_matrix(self.universe_mask, field_name="universe_mask")
            _validate_shape_for(
                self.universe_mask,
                rows=rows,
                columns=columns,
                field_name="universe_mask",
            )
        if self.tradability_mask is not None:
            self.tradability_mask = _coerce_bool_matrix(
                self.tradability_mask,
                field_name="tradability_mask",
            )
            _validate_shape_for(
                self.tradability_mask,
                rows=rows,
                columns=columns,
                field_name="tradability_mask",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "contract": self.contract.to_dict(),
            "fields": dict(_json_safe(self.fields)),
            "universe_mask": _json_safe(self.universe_mask),
            "tradability_mask": _json_safe(self.tradability_mask),
            "industry_by_symbol": dict(_json_safe(self.industry_by_symbol)),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MatrixDataBundle":
        data = dict(payload)
        contract_payload = data.get("contract", {}) or {}
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_MATRIX_SCHEMA_VERSION)),
            bundle_id=str(data.get("bundle_id", "")),
            contract=MatrixDataContract.from_dict(contract_payload)
            if isinstance(contract_payload, Mapping)
            else contract_payload,
            fields=dict(data.get("fields", {}) or {}),
            universe_mask=data.get("universe_mask"),
            tradability_mask=data.get("tradability_mask"),
            industry_by_symbol=dict(data.get("industry_by_symbol", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def get_field(self, field_name: str) -> list[list[Any]]:
        if field_name not in self.fields:
            raise KeyError(field_name)
        return [list(row) for row in self.fields[field_name]]

    def has_field(self, field_name: str) -> bool:
        return field_name in self.fields

    def with_field(self, field_name: str, values: Sequence[Sequence[Any]]) -> "MatrixDataBundle":
        self.validate_shape(values, field_name=field_name)
        payload = self.to_dict()
        fields = dict(payload["fields"])
        fields[str(field_name)] = _json_safe(_coerce_matrix(values, field_name=field_name))
        payload["fields"] = fields
        return MatrixDataBundle.from_dict(payload)

    def validate_shape(self, values: Sequence[Sequence[Any]], *, field_name: str) -> None:
        rows, columns = _matrix_dimensions(self.contract.symbols, self.contract.dates)
        _validate_shape_for(values, rows=rows, columns=columns, field_name=field_name)


@dataclass
class FactorMatrix:
    schema_version: str = FACTOR_MATRIX_SCHEMA_VERSION
    matrix_id: str = ""
    factor_id: str | None = None
    factor_version: str | None = None
    expression: str = ""
    symbols: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    values: list[list[float | None]] = field(default_factory=list)
    coverage_ratio: float = 0.0
    missing_ratio: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_MATRIX_SCHEMA_VERSION)
        self.matrix_id = _non_empty_str(self.matrix_id, "matrix_id")
        self.factor_id = _optional_str(self.factor_id)
        self.factor_version = _optional_str(self.factor_version)
        self.expression = _non_empty_str(self.expression, "expression")
        self.symbols = [str(symbol).strip() for symbol in self.symbols if str(symbol).strip()]
        if not self.symbols:
            raise ValueError("symbols must be non-empty.")
        _reject_duplicates(self.symbols, "symbols")
        self.dates = _coerce_dates(self.dates)
        _validate_shape_for(
            self.values,
            rows=len(self.symbols),
            columns=len(self.dates),
            field_name="values",
        )
        self.values = _coerce_numeric_matrix(self.values, field_name="values")
        self.coverage_ratio = _coerce_unit_float(self.coverage_ratio, "coverage_ratio")
        self.missing_ratio = _coerce_unit_float(self.missing_ratio, "missing_ratio")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "matrix_id": self.matrix_id,
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "expression": self.expression,
            "symbols": list(self.symbols),
            "dates": list(self.dates),
            "values": _json_safe(self.values),
            "coverage_ratio": self.coverage_ratio,
            "missing_ratio": self.missing_ratio,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorMatrix":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_MATRIX_SCHEMA_VERSION)),
            matrix_id=str(data.get("matrix_id", "")),
            factor_id=data.get("factor_id"),
            factor_version=data.get("factor_version"),
            expression=str(data.get("expression", "")),
            symbols=list(data.get("symbols", []) or []),
            dates=list(data.get("dates", []) or []),
            values=list(data.get("values", []) or []),
            coverage_ratio=float(data.get("coverage_ratio", 0.0)),
            missing_ratio=float(data.get("missing_ratio", 0.0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ExpressionEvaluationResult:
    schema_version: str = FACTOR_EXPRESSION_SCHEMA_VERSION
    result_id: str = ""
    expression: str = ""
    factor_matrix: FactorMatrix = field(default_factory=FactorMatrix)
    used_fields: list[str] = field(default_factory=list)
    used_operators: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_EXPRESSION_SCHEMA_VERSION)
        self.result_id = _non_empty_str(self.result_id, "result_id")
        self.expression = _non_empty_str(self.expression, "expression")
        if not isinstance(self.factor_matrix, FactorMatrix):
            self.factor_matrix = FactorMatrix.from_dict(self.factor_matrix)
        self.used_fields = _ordered_unique(self.used_fields)
        self.used_operators = _ordered_unique(self.used_operators)
        self.warnings = _ordered_unique(self.warnings)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "expression": self.expression,
            "factor_matrix": self.factor_matrix.to_dict(),
            "used_fields": list(self.used_fields),
            "used_operators": list(self.used_operators),
            "warnings": list(self.warnings),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpressionEvaluationResult":
        data = dict(payload)
        matrix_payload = data.get("factor_matrix", {}) or {}
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_EXPRESSION_SCHEMA_VERSION)),
            result_id=str(data.get("result_id", "")),
            expression=str(data.get("expression", "")),
            factor_matrix=FactorMatrix.from_dict(matrix_payload)
            if isinstance(matrix_payload, Mapping)
            else matrix_payload,
            used_fields=list(data.get("used_fields", []) or []),
            used_operators=list(data.get("used_operators", []) or []),
            warnings=list(data.get("warnings", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def compute_coverage(values: Sequence[Sequence[Any]]) -> tuple[float, float]:
    total = 0
    covered = 0
    for row in values:
        if not _is_sequence_row(row):
            raise ValueError("values must be a matrix.")
        for item in row:
            total += 1
            if _to_finite_float(item) is not None:
                covered += 1
    if total == 0:
        return 0.0, 1.0
    coverage_ratio = covered / total
    return coverage_ratio, 1.0 - coverage_ratio


def make_matrix_contract_id(
    *,
    universe: str,
    benchmark: str | None,
    symbols: Sequence[str],
    dates: Sequence[str],
) -> str:
    ordered_symbols = sorted(str(symbol) for symbol in symbols)
    ordered_dates = [str(item) for item in dates]
    parts = [str(universe), benchmark, ordered_symbols, ordered_dates]
    return f"matrix-contract-{_slug(universe)}-{_slug(benchmark)}-{_short_hash(parts)}"


def make_matrix_bundle_id(*, contract_id: str, field_names: Sequence[str]) -> str:
    fields = sorted({str(field_name) for field_name in field_names})
    parts = [str(contract_id), fields]
    return f"matrix-bundle-{_slug(contract_id)}-{len(fields)}f-{_short_hash(parts)}"


def make_factor_matrix_id(
    *,
    expression: str,
    symbols: Sequence[str],
    dates: Sequence[str],
    factor_id: str | None = None,
) -> str:
    parts = [
        str(expression),
        [str(symbol) for symbol in symbols],
        [str(item) for item in dates],
        factor_id,
    ]
    prefix = _slug(factor_id) if factor_id else "adhoc"
    return f"factor-matrix-{prefix}-{_short_hash(parts)}"


def make_expression_result_id(*, expression: str, matrix_id: str) -> str:
    parts = [str(expression), str(matrix_id)]
    return f"expression-result-{_slug(matrix_id)}-{_short_hash(parts)}"


def _blank_float_matrix(symbols: Sequence[str], dates: Sequence[str]) -> list[list[float | None]]:
    return [[None for _ in dates] for _ in symbols]


def _compute_vwap(bundle: MatrixDataBundle) -> list[list[float | None]]:
    amount = bundle.get_field(FIELD_AMOUNT)
    volume = bundle.get_field(FIELD_VOLUME)
    output = _blank_float_matrix(bundle.contract.symbols, bundle.contract.dates)
    for row_index, (amount_row, volume_row) in enumerate(zip(amount, volume)):
        for col_index, (amount_value, volume_value) in enumerate(zip(amount_row, volume_row)):
            amount_number = _to_finite_float(amount_value)
            volume_number = _to_finite_float(volume_value)
            if amount_number is None or volume_number is None or volume_number == 0.0:
                continue
            output[row_index][col_index] = amount_number / volume_number
    return output


def _compute_ret1_from_close(close: Sequence[Sequence[Any]]) -> list[list[float | None]]:
    output: list[list[float | None]] = []
    for close_row in close:
        result_row: list[float | None] = []
        previous: float | None = None
        for col_index, value in enumerate(close_row):
            current = _to_finite_float(value)
            if col_index == 0 or current is None or previous is None or previous == 0.0:
                result_row.append(None)
            else:
                result_row.append(current / previous - 1.0)
            previous = current
        output.append(result_row)
    return output


def _compute_benchmark_ret(bundle: MatrixDataBundle) -> list[list[float | None]]:
    benchmark_close = bundle.get_field(FIELD_BENCHMARK_CLOSE)
    first_row_ret = _compute_ret1_from_close([benchmark_close[0]])[0]
    return [list(first_row_ret) for _ in bundle.contract.symbols]


def build_standard_derived_fields(bundle: MatrixDataBundle) -> dict[str, list[list[float | None]]]:
    derived: dict[str, list[list[float | None]]] = {}
    if bundle.has_field(FIELD_AMOUNT) and bundle.has_field(FIELD_VOLUME):
        derived[FIELD_VWAP] = _compute_vwap(bundle)
    if bundle.has_field(FIELD_CLOSE):
        derived[FIELD_RET1] = _compute_ret1_from_close(bundle.get_field(FIELD_CLOSE))
    allow_overwrite = bool(
        bundle.metadata.get("allow_derived_overwrite")
        or bundle.metadata.get("allow_standard_derived_overwrite")
    )
    if bundle.has_field(FIELD_BENCHMARK_CLOSE) and (
        allow_overwrite or not bundle.has_field(FIELD_BENCHMARK_RET)
    ):
        derived[FIELD_BENCHMARK_RET] = _compute_benchmark_ret(bundle)
    return derived


def add_standard_derived_fields(bundle: MatrixDataBundle) -> MatrixDataBundle:
    output = bundle
    allow_overwrite = bool(
        bundle.metadata.get("allow_derived_overwrite")
        or bundle.metadata.get("allow_standard_derived_overwrite")
    )
    for field_name, values in build_standard_derived_fields(bundle).items():
        if output.has_field(field_name) and not allow_overwrite:
            continue
        output = output.with_field(field_name, values)
    return output


__all__ = [
    "FIELD_OPEN",
    "FIELD_HIGH",
    "FIELD_LOW",
    "FIELD_CLOSE",
    "FIELD_VOLUME",
    "FIELD_AMOUNT",
    "FIELD_INDUSTRY",
    "FIELD_BENCHMARK_CLOSE",
    "FIELD_BENCHMARK_RET",
    "FIELD_BENCHMARK_WEIGHT",
    "FIELD_VWAP",
    "FIELD_RET1",
    "STANDARD_MATRIX_FIELDS",
    "DEFAULT_FACTOR_MATRIX_DIR",
    "DEFAULT_MATRIX_CONTRACTS_FILENAME",
    "DEFAULT_MATRIX_BUNDLES_FILENAME",
    "DEFAULT_FACTOR_MATRICES_FILENAME",
    "DEFAULT_EXPRESSION_RESULTS_FILENAME",
    "MatrixDataContract",
    "MatrixDataBundle",
    "FactorMatrix",
    "ExpressionEvaluationResult",
    "compute_coverage",
    "make_matrix_contract_id",
    "make_matrix_bundle_id",
    "make_factor_matrix_id",
    "make_expression_result_id",
    "build_standard_derived_fields",
    "add_standard_derived_fields",
]
