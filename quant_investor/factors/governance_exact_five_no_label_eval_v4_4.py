"""Filesystem-free exact-five no-label evaluator for FactorGovernance v4.4.

This module deliberately implements only the five definitions frozen by the
v4.4 preregistration oracle.  It exposes two independent calculation engines:
an allowlisted source-DAG interpreter and direct local formulas.  Neither
engine accepts labels, returns, statistics, providers, or mutable source code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quant_investor.factors import governance_candidate_preregistration_v4_4 as prereg


PROTOCOL_VERSION = "v4"
EVIDENCE_CONTRACT_VERSION = "v4.4"
ENGINE_RESULT_SCHEMA_VERSION = "factor-governance-exact-five-engine-result.v4.4"
SYNTHETIC_FIXTURE_SCHEMA_VERSION = (
    "factor-governance-exact-five-synthetic-fixture.v4.4"
)
SOURCE_ENGINE_ID = "restricted_source_dag.v4.4"
LOCAL_ENGINE_ID = "independent_local_formulas.v4.4"

INPUT_FIELDS = ("raw_close", "raw_open", "vol", "adj_close")
CANDIDATE_DIRECTIONS = {
    "alpha_range_position_momentum_20d": 1.0,
    "pv_low_overnight_gap_20d": -1.0,
    "pv_low_vol_ratio_10_60": -1.0,
    "pv_price_volume_consistency_20d": 1.0,
    "pv_low_vol_of_vol_20d": -1.0,
}

# This is a data-only, closed instruction set.  The interpreter below rejects
# every mutation rather than evaluating caller-supplied Python.
SOURCE_PROGRAMS_V4_4 = (
    {
        "name": "alpha_range_position_momentum_20d",
        "source": "A_quant",
        "program": (
            "cs_rank(div(sub(raw_close,ts_min(raw_close,20,1)),"
            "sub(ts_max(raw_close,20,1),ts_min(raw_close,20,1))))"
        ),
        "direction": 1.0,
    },
    {
        "name": "pv_low_overnight_gap_20d",
        "source": "myQuant.alpha158:OVERNIGHT_GAP_20D",
        "program": (
            "rolling_mean(abs(div(sub(raw_open,shift(raw_close,1)),"
            "add(shift(raw_close,1),1e-9))),20,20)"
        ),
        "direction": -1.0,
    },
    {
        "name": "pv_low_vol_ratio_10_60",
        "source": "myQuant.alpha158:VOL_RATIO_10_60",
        "program": (
            "div(rolling_std(pct_change(raw_close,1,None),10,10,1),"
            "add(rolling_std(pct_change(raw_close,1,None),60,60,1),1e-9))"
        ),
        "direction": -1.0,
    },
    {
        "name": "pv_price_volume_consistency_20d",
        "source": "myQuant.alpha158:PRICE_VOL_CONSISTENCY_20D",
        "program": (
            "rolling_mean(mul(sign(diff(raw_close,1)),sign(diff(vol,1))),20,20)"
        ),
        "direction": 1.0,
    },
    {
        "name": "pv_low_vol_of_vol_20d",
        "source": "myQuant.alpha158:VOL_OF_VOL_20D",
        "program": (
            "rolling_std(rolling_std(pct_change(adj_close,1,None),5,5,1),20,20,1)"
        ),
        "direction": -1.0,
    },
)


class FactorGovernanceExactFiveEvalV4_4Error(ValueError):
    """Raised when exact-five evaluation cannot prove its closed contract."""


def _error(message: str) -> FactorGovernanceExactFiveEvalV4_4Error:
    return FactorGovernanceExactFiveEvalV4_4Error(message)


def canonical_json_bytes_v4_4(value: Any) -> bytes:
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


def _axis_descriptor(index: pd.Index, *, label: str) -> dict[str, Any]:
    if label == "date":
        if not isinstance(index, pd.DatetimeIndex):
            raise _error("matrix date axis must be a DatetimeIndex")
        values = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in index]
    else:
        if any(type(value) is not str or not value for value in index):
            raise _error("matrix symbol axis must contain non-empty strings")
        values = list(index)
    if not index.is_unique or not index.is_monotonic_increasing:
        raise _error(f"matrix {label} axis must be strictly ordered and unique")
    if label == "symbol" and values != sorted(values):
        raise _error("matrix symbol axis must be sorted")
    return {
        "count": len(values),
        "sha256": hashlib.sha256(
            ("\n".join(values) + ("\n" if values else "")).encode("utf-8")
        ).hexdigest(),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
    }


def _normalized_float64_bytes(matrix: pd.DataFrame) -> bytes:
    values = matrix.to_numpy(dtype=np.float64, copy=True).astype("<f8", copy=False)
    bits = values.view("<u8")
    bits[np.isnan(values)] = np.uint64(0x7FF8000000000000)
    return bits.astype("<u8", copy=False).tobytes(order="C")


def matrix_hash_descriptor_v4_4(matrix: pd.DataFrame) -> dict[str, Any]:
    if type(matrix) is not pd.DataFrame or matrix.empty:
        raise _error("matrix must be a non-empty pandas DataFrame")
    dates = _axis_descriptor(matrix.index, label="date")
    symbols = _axis_descriptor(matrix.columns, label="symbol")
    values = matrix.to_numpy(dtype=np.float64, copy=False)
    raw = _normalized_float64_bytes(matrix)
    normalized_bits = np.frombuffer(raw, dtype="<u8").copy()
    magnitude_bits = normalized_bits & np.uint64(0x7FFFFFFFFFFFFFFF)
    elementwise_negated_bits = normalized_bits.copy()
    not_nan = ~np.isnan(values).reshape(-1)
    elementwise_negated_bits[not_nan] ^= np.uint64(0x8000000000000000)
    return {
        "dtype": "float64-le",
        "row_count": int(values.shape[0]),
        "column_count": int(values.shape[1]),
        "date_axis": dates,
        "symbol_axis": symbols,
        "matrix_sha256": hashlib.sha256(raw).hexdigest(),
        "finite_count": int(np.isfinite(values).sum()),
        "nan_count": int(np.isnan(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
        "positive_finite_count": int((np.isfinite(values) & (values > 0.0)).sum()),
        "negative_finite_count": int((np.isfinite(values) & (values < 0.0)).sum()),
        "positive_zero_count": int(
            ((values == 0.0) & ~np.signbit(values)).sum()
        ),
        "negative_zero_count": int(((values == 0.0) & np.signbit(values)).sum()),
        "byte_count": len(raw),
        "bit_pattern_sha256": hashlib.sha256(raw).hexdigest(),
        "magnitude_bits_sha256": hashlib.sha256(
            magnitude_bits.astype("<u8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "elementwise_negated_sha256": hashlib.sha256(
            elementwise_negated_bits.astype("<u8", copy=False).tobytes(order="C")
        ).hexdigest(),
    }


def build_synthetic_fixture_v4_4() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return a fresh deterministic, no-label fixture for contract validation."""

    index = pd.bdate_range(end="2026-07-20", periods=100, name="trade_date")
    columns = pd.Index(["000001.SZ", "000002.SZ", "600000.SH"], name="ts_code")
    step = np.arange(100, dtype=np.float64)[:, None]
    close = pd.DataFrame(
        12.0 + step * np.asarray([0.05, 0.08, 0.03])[None, :],
        index=index,
        columns=columns,
    )
    open_price = close * (1.0 + np.sin(step / 8.0) * 0.0025)
    volume = pd.DataFrame(
        90_000.0 + step * np.asarray([140.0, 90.0, 220.0])[None, :],
        index=index,
        columns=columns,
    )
    adjusted = close.copy()
    adjusted.iloc[45:, 0] *= 0.55
    pit = pd.DataFrame(True, index=index, columns=columns, dtype=bool)
    pit.iloc[10:15, 2] = False
    return {
        "raw_close": close,
        "raw_open": open_price,
        "vol": volume,
        "adj_close": adjusted,
    }, pit


def synthetic_fixture_binding_v4_4() -> dict[str, Any]:
    matrices, pit = build_synthetic_fixture_v4_4()
    binding = {
        "schema_version": SYNTHETIC_FIXTURE_SCHEMA_VERSION,
        "fixture_id": "exact_five_no_label_deterministic_20260720_v1",
        "input_fields": list(INPUT_FIELDS),
        "input_matrix_descriptors": {
            name: matrix_hash_descriptor_v4_4(matrices[name]) for name in INPUT_FIELDS
        },
        "pit_mask_descriptor": matrix_hash_descriptor_v4_4(pit.astype(np.float64)),
        "labels_present": False,
        "outcomes_present": False,
        "statistics_present": False,
    }
    binding["fixture_semantic_sha256"] = semantic_sha256_v4_4(binding)
    return binding


def _normalize_inputs(
    inputs: Mapping[str, pd.DataFrame], pit_mask: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if not isinstance(inputs, Mapping) or tuple(inputs) != INPUT_FIELDS:
        raise _error("inputs must contain exact ordered raw_close/raw_open/vol/adj_close")
    if type(pit_mask) is not pd.DataFrame or pit_mask.empty:
        raise _error("PIT mask must be a non-empty DataFrame")
    if any(dtype != bool for dtype in pit_mask.dtypes):
        raise _error("PIT mask must be strictly boolean")
    _axis_descriptor(pit_mask.index, label="date")
    _axis_descriptor(pit_mask.columns, label="symbol")
    normalized: dict[str, pd.DataFrame] = {}
    for field in INPUT_FIELDS:
        value = inputs[field]
        if type(value) is not pd.DataFrame or value.empty:
            raise _error(f"{field} must be a non-empty DataFrame")
        if not value.index.equals(pit_mask.index) or not value.columns.equals(
            pit_mask.columns
        ):
            raise _error(f"{field} axes differ from the PIT mask")
        try:
            numeric = value.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise _error(f"{field} must contain numeric values") from exc
        normalized[field] = numeric.where(pit_mask)
    return normalized, pit_mask.copy()


def validate_source_programs_v4_4(
    programs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(programs, Sequence) or isinstance(programs, (str, bytes)):
        raise _error("source programs must be a sequence")
    try:
        normalized = tuple(copy.deepcopy(dict(row)) for row in programs)
    except (TypeError, ValueError) as exc:
        raise _error("source programs must contain objects") from exc
    if normalized != SOURCE_PROGRAMS_V4_4:
        raise _error("source program inventory or literal definition mismatch")
    if tuple(row["name"] for row in normalized) != tuple(
        row["name"] for row in prereg.EXPECTED_CANDIDATE_ROWS
    ):
        raise _error("source program candidates differ from the v4.4 oracle")
    return normalized


def source_programs_semantic_sha256_v4_4() -> str:
    return semantic_sha256_v4_4(list(SOURCE_PROGRAMS_V4_4))


def _alpha_node(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    values = frame.astype(np.float64).replace([np.inf, -np.inf], np.nan)
    return values.where(mask)


def _range_node(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(np.float64).where(mask)


def evaluate_source_dag_v4_4(
    inputs: Mapping[str, pd.DataFrame],
    pit_mask: pd.DataFrame,
    *,
    programs: Sequence[Mapping[str, Any]] = SOURCE_PROGRAMS_V4_4,
) -> dict[str, pd.DataFrame]:
    """Interpret the exact closed source-DAG instruction set."""

    validate_source_programs_v4_4(programs)
    values, mask = _normalize_inputs(inputs, pit_mask)

    close = _range_node(values["raw_close"], mask)
    minimum = _range_node(close.rolling(20, min_periods=1).min(), mask)
    maximum = _range_node(close.rolling(20, min_periods=1).max(), mask)
    numerator = _range_node(close - minimum, mask)
    denominator = _range_node(maximum - minimum, mask)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = _range_node(numerator / denominator, mask)
    range_signal = _range_node(ratio.rank(axis=1, pct=True), mask)

    alpha_close = _alpha_node(values["raw_close"], mask)
    alpha_open = _alpha_node(values["raw_open"], mask)
    prior_close = _alpha_node(alpha_close.shift(1), mask)
    overnight_numerator = _alpha_node(alpha_open - prior_close, mask)
    overnight_denominator = _alpha_node(prior_close + 1e-9, mask)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        overnight_ratio = _alpha_node(overnight_numerator / overnight_denominator, mask)
    overnight_abs = _alpha_node(overnight_ratio.abs(), mask)
    overnight = _alpha_node(
        overnight_abs.rolling(20, min_periods=20).mean(), mask
    )

    returns = _alpha_node(alpha_close.pct_change(fill_method=None), mask)
    vol10 = _alpha_node(returns.rolling(10, min_periods=10).std(ddof=1), mask)
    vol60 = _alpha_node(returns.rolling(60, min_periods=60).std(ddof=1), mask)
    vol60_epsilon = _alpha_node(vol60 + 1e-9, mask)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vol_ratio = _alpha_node(vol10 / vol60_epsilon, mask)

    price_diff = _alpha_node(alpha_close.diff(), mask)
    volume = _alpha_node(values["vol"], mask)
    volume_diff = _alpha_node(volume.diff(), mask)
    signed_price = _alpha_node(np.sign(price_diff), mask)
    signed_volume = _alpha_node(np.sign(volume_diff), mask)
    signed_product = _alpha_node(signed_price * signed_volume, mask)
    consistency = _alpha_node(
        signed_product.rolling(20, min_periods=20).mean(), mask
    )

    adjusted = _alpha_node(values["adj_close"], mask)
    adjusted_returns = _alpha_node(adjusted.pct_change(fill_method=None), mask)
    adjusted_vol5 = _alpha_node(
        adjusted_returns.rolling(5, min_periods=5).std(ddof=1), mask
    )
    vol_of_vol = _alpha_node(
        adjusted_vol5.rolling(20, min_periods=20).std(ddof=1), mask
    )
    return {
        "alpha_range_position_momentum_20d": range_signal,
        "pv_low_overnight_gap_20d": overnight,
        "pv_low_vol_ratio_10_60": vol_ratio,
        "pv_price_volume_consistency_20d": consistency,
        "pv_low_vol_of_vol_20d": vol_of_vol,
    }


def evaluate_local_formulas_v4_4(
    inputs: Mapping[str, pd.DataFrame], pit_mask: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Compute the five formulas independently of the source-DAG engine."""

    values, mask = _normalize_inputs(inputs, pit_mask)

    raw_close = values["raw_close"].where(mask)
    low20 = raw_close.rolling(window=20, min_periods=1).min().where(mask)
    high20 = raw_close.rolling(window=20, min_periods=1).max().where(mask)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        position = ((raw_close - low20) / (high20 - low20)).where(mask)
    range_signal = position.rank(axis="columns", pct=True).where(mask)

    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.replace([np.inf, -np.inf], np.nan).where(mask)

    close = clean(values["raw_close"])
    open_price = clean(values["raw_open"])
    lagged = clean(close.shift(periods=1))
    gap = clean(open_price.sub(lagged))
    gap_base = clean(lagged.add(0.000000001))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gap_fraction = clean(gap.div(gap_base))
    overnight = clean(
        clean(gap_fraction.abs()).rolling(window=20, min_periods=20).mean()
    )

    one_day = clean(close.pct_change(periods=1, fill_method=None))
    short_sigma = clean(one_day.rolling(window=10, min_periods=10).std(ddof=1))
    long_sigma = clean(one_day.rolling(window=60, min_periods=60).std(ddof=1))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vol_ratio = clean(short_sigma.div(clean(long_sigma.add(0.000000001))))

    price_direction = clean(close.diff(periods=1).map(np.sign))
    canonical_volume = clean(values["vol"])
    volume_direction = clean(canonical_volume.diff(periods=1).map(np.sign))
    consistency = clean(
        clean(price_direction.multiply(volume_direction))
        .rolling(window=20, min_periods=20)
        .mean()
    )

    adjusted_close = clean(values["adj_close"])
    adjusted_change = clean(
        adjusted_close.pct_change(periods=1, fill_method=None)
    )
    five_day_sigma = clean(
        adjusted_change.rolling(window=5, min_periods=5).std(ddof=1)
    )
    vol_of_vol = clean(
        five_day_sigma.rolling(window=20, min_periods=20).std(ddof=1)
    )

    return {
        "alpha_range_position_momentum_20d": range_signal,
        "pv_low_overnight_gap_20d": overnight,
        "pv_low_vol_ratio_10_60": vol_ratio,
        "pv_price_volume_consistency_20d": consistency,
        "pv_low_vol_of_vol_20d": vol_of_vol,
    }


def _validate_outputs(
    outputs: Mapping[str, pd.DataFrame], pit_mask: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    expected = tuple(row["name"] for row in prereg.EXPECTED_CANDIDATE_ROWS)
    if not isinstance(outputs, Mapping) or tuple(outputs) != expected:
        raise _error("engine outputs must contain the exact ordered five candidates")
    normalized: dict[str, pd.DataFrame] = {}
    outside = ~pit_mask.to_numpy(dtype=bool, copy=False)
    for name in expected:
        value = outputs[name]
        if type(value) is not pd.DataFrame:
            raise _error(f"{name} output must be a DataFrame")
        if not value.index.equals(pit_mask.index) or not value.columns.equals(
            pit_mask.columns
        ):
            raise _error(f"{name} output axes differ from the PIT mask")
        numeric = value.astype(np.float64)
        raw = numeric.to_numpy(dtype=np.float64, copy=False)
        if np.any(~np.isnan(raw[outside])):
            raise _error(f"{name} contains a value outside the PIT mask")
        if int(np.isfinite(raw[~outside]).sum()) <= 0:
            raise _error(f"{name} has no finite in-PIT observation")
        normalized[name] = numeric
    return normalized


def build_engine_pass_result_v4_4(
    *,
    engine_id: str,
    pass_id: str,
    collection_sha256: str,
    outputs: Mapping[str, pd.DataFrame],
    pit_mask: pd.DataFrame,
) -> dict[str, Any]:
    if engine_id not in {SOURCE_ENGINE_ID, LOCAL_ENGINE_ID}:
        raise _error("engine_id is not an accepted independent engine")
    if pass_id not in {"fresh_pass_1", "fresh_pass_2"}:
        raise _error("pass_id must identify one of two fresh passes")
    if (
        type(collection_sha256) is not str
        or len(collection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in collection_sha256)
    ):
        raise _error("collection_sha256 must be lowercase SHA-256")
    normalized = _validate_outputs(outputs, pit_mask)
    rows: list[dict[str, Any]] = []
    for oracle in prereg.EXPECTED_CANDIDATE_ROWS:
        name = oracle["name"]
        direction = CANDIDATE_DIRECTIONS[name]
        raw = normalized[name]
        adjusted = raw * direction
        rows.append(
            {
                "order": oracle["order"],
                "name": name,
                "definition_identity_sha256": oracle[
                    "definition_identity_sha256"
                ],
                "direction": direction,
                "raw_matrix": matrix_hash_descriptor_v4_4(raw),
                "direction_adjusted_matrix": matrix_hash_descriptor_v4_4(adjusted),
            }
        )
    manifest = {
        "schema_version": ENGINE_RESULT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "engine_id": engine_id,
        "pass_id": pass_id,
        "collection_sha256": collection_sha256,
        "pit_mask": matrix_hash_descriptor_v4_4(pit_mask.astype(np.float64)),
        "candidates": rows,
        "signal_computability_proven": True,
        "statistics_run": False,
        "authority": False,
    }
    manifest["result_semantic_sha256"] = semantic_sha256_v4_4(manifest)
    return manifest


def _validate_matrix_descriptor_v4_4(
    value: Any, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = copy.deepcopy(dict(value))
    if set(payload) != {
        "dtype",
        "row_count",
        "column_count",
        "date_axis",
        "symbol_axis",
        "matrix_sha256",
        "finite_count",
        "nan_count",
        "positive_infinity_count",
        "negative_infinity_count",
        "positive_finite_count",
        "negative_finite_count",
        "positive_zero_count",
        "negative_zero_count",
        "byte_count",
        "bit_pattern_sha256",
        "magnitude_bits_sha256",
        "elementwise_negated_sha256",
    }:
        raise _error(f"{label} fields are not exact")
    if payload["dtype"] != "float64-le":
        raise _error(f"{label} dtype mismatch")
    for axis_name in ("date_axis", "symbol_axis"):
        axis = payload[axis_name]
        if not isinstance(axis, Mapping) or set(axis) != {
            "count",
            "sha256",
            "first",
            "last",
        }:
            raise _error(f"{label} {axis_name} fields are not exact")
        if (
            type(axis["count"]) is not int
            or axis["count"] <= 0
            or type(axis["sha256"]) is not str
            or len(axis["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in axis["sha256"])
            or type(axis["first"]) is not str
            or type(axis["last"]) is not str
        ):
            raise _error(f"{label} {axis_name} contract mismatch")
    if (
        type(payload["row_count"]) is not int
        or type(payload["column_count"]) is not int
        or payload["row_count"] != payload["date_axis"]["count"]
        or payload["column_count"] != payload["symbol_axis"]["count"]
    ):
        raise _error(f"{label} shape and axes mismatch")
    for field in (
        "matrix_sha256",
        "bit_pattern_sha256",
        "magnitude_bits_sha256",
        "elementwise_negated_sha256",
    ):
        digest = payload[field]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise _error(f"{label} {field} must be lowercase SHA-256")
    if payload["matrix_sha256"] != payload["bit_pattern_sha256"]:
        raise _error(f"{label} normalized bit-pattern SHA mismatch")
    count_fields = (
        "finite_count",
        "nan_count",
        "positive_infinity_count",
        "negative_infinity_count",
        "positive_finite_count",
        "negative_finite_count",
        "positive_zero_count",
        "negative_zero_count",
    )
    if any(type(payload[field]) is not int or payload[field] < 0 for field in count_fields):
        raise _error(f"{label} observation counts must be non-negative integers")
    cells = payload["row_count"] * payload["column_count"]
    if (
        payload["finite_count"]
        + payload["nan_count"]
        + payload["positive_infinity_count"]
        + payload["negative_infinity_count"]
        != cells
        or payload["negative_zero_count"] > payload["finite_count"]
        or payload["positive_finite_count"]
        + payload["negative_finite_count"]
        + payload["positive_zero_count"]
        + payload["negative_zero_count"]
        != payload["finite_count"]
        or payload["byte_count"] != cells * 8
    ):
        raise _error(f"{label} observation accounting mismatch")
    return payload


def validate_engine_pass_result_v4_4(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("engine pass result must be an object")
    payload = copy.deepcopy(dict(value))
    expected_fields = {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "engine_id",
        "pass_id",
        "collection_sha256",
        "pit_mask",
        "candidates",
        "signal_computability_proven",
        "statistics_run",
        "authority",
        "result_semantic_sha256",
    }
    if set(payload) != expected_fields:
        raise _error("engine pass result fields are not exact")
    if (
        payload["schema_version"] != ENGINE_RESULT_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
        or payload["engine_id"] not in {SOURCE_ENGINE_ID, LOCAL_ENGINE_ID}
        or payload["pass_id"] not in {"fresh_pass_1", "fresh_pass_2"}
        or payload["signal_computability_proven"] is not True
        or payload["statistics_run"] is not False
        or payload["authority"] is not False
    ):
        raise _error("engine pass result contract mismatch")
    collection = payload["collection_sha256"]
    if (
        type(collection) is not str
        or len(collection) != 64
        or any(character not in "0123456789abcdef" for character in collection)
    ):
        raise _error("engine collection SHA must be lowercase SHA-256")
    pit_descriptor = _validate_matrix_descriptor_v4_4(
        payload["pit_mask"], "engine PIT mask"
    )
    pit_cells = pit_descriptor["row_count"] * pit_descriptor["column_count"]
    if (
        pit_descriptor["finite_count"] != pit_cells
        or pit_descriptor["nan_count"] != 0
        or pit_descriptor["positive_infinity_count"] != 0
        or pit_descriptor["negative_infinity_count"] != 0
    ):
        raise _error("engine PIT mask descriptor must be finite and complete")
    supplied = payload.pop("result_semantic_sha256", None)
    if supplied != semantic_sha256_v4_4(payload):
        raise _error("engine pass result semantic SHA mismatch")
    payload["result_semantic_sha256"] = supplied
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise _error("engine pass result must contain exact five rows")
    for oracle, row in zip(prereg.EXPECTED_CANDIDATE_ROWS, candidates, strict=True):
        if not isinstance(row, Mapping):
            raise _error("engine candidate row must be an object")
        if set(row) != {
            "order",
            "name",
            "definition_identity_sha256",
            "direction",
            "raw_matrix",
            "direction_adjusted_matrix",
        }:
            raise _error("engine candidate row fields are not exact")
        if (
            row["order"] != oracle["order"]
            or row["name"] != oracle["name"]
            or row["definition_identity_sha256"]
            != oracle["definition_identity_sha256"]
            or row["direction"] != CANDIDATE_DIRECTIONS[oracle["name"]]
            or row["raw_matrix"].get("finite_count", 0) <= 0
        ):
            raise _error("engine candidate row differs from the exact-five oracle")
        raw_descriptor = _validate_matrix_descriptor_v4_4(
            row["raw_matrix"], f"{row['name']} raw matrix"
        )
        adjusted_descriptor = _validate_matrix_descriptor_v4_4(
            row["direction_adjusted_matrix"],
            f"{row['name']} direction-adjusted matrix",
        )
        for descriptor in (raw_descriptor, adjusted_descriptor):
            if (
                descriptor["row_count"] != pit_descriptor["row_count"]
                or descriptor["column_count"] != pit_descriptor["column_count"]
                or descriptor["date_axis"] != pit_descriptor["date_axis"]
                or descriptor["symbol_axis"] != pit_descriptor["symbol_axis"]
            ):
                raise _error("engine candidate matrix axes differ from the PIT mask")
        direction = row["direction"]
        if direction == 1.0:
            if raw_descriptor != adjusted_descriptor:
                raise _error(
                    "positive-direction adjusted descriptor must equal the raw descriptor"
                )
        elif (
            raw_descriptor["matrix_sha256"]
            == adjusted_descriptor["matrix_sha256"]
            or raw_descriptor["elementwise_negated_sha256"]
            != adjusted_descriptor["matrix_sha256"]
            or adjusted_descriptor["elementwise_negated_sha256"]
            != raw_descriptor["matrix_sha256"]
            or raw_descriptor["magnitude_bits_sha256"]
            != adjusted_descriptor["magnitude_bits_sha256"]
            or raw_descriptor["finite_count"] != adjusted_descriptor["finite_count"]
            or raw_descriptor["nan_count"] != adjusted_descriptor["nan_count"]
            or raw_descriptor["positive_infinity_count"]
            != adjusted_descriptor["negative_infinity_count"]
            or raw_descriptor["negative_infinity_count"]
            != adjusted_descriptor["positive_infinity_count"]
            or raw_descriptor["positive_finite_count"]
            != adjusted_descriptor["negative_finite_count"]
            or raw_descriptor["negative_finite_count"]
            != adjusted_descriptor["positive_finite_count"]
            or raw_descriptor["positive_zero_count"]
            != adjusted_descriptor["negative_zero_count"]
            or raw_descriptor["negative_zero_count"]
            != adjusted_descriptor["positive_zero_count"]
        ):
            raise _error("negative-direction adjusted descriptor transform mismatch")
    return copy.deepcopy(payload)


def engine_equivalence_payload_v4_4(value: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_engine_pass_result_v4_4(value)
    return {
        "pit_mask": validated["pit_mask"],
        "candidates": validated["candidates"],
        "signal_computability_proven": validated["signal_computability_proven"],
    }


__all__ = [
    "CANDIDATE_DIRECTIONS",
    "ENGINE_RESULT_SCHEMA_VERSION",
    "FactorGovernanceExactFiveEvalV4_4Error",
    "INPUT_FIELDS",
    "LOCAL_ENGINE_ID",
    "SOURCE_ENGINE_ID",
    "SOURCE_PROGRAMS_V4_4",
    "SYNTHETIC_FIXTURE_SCHEMA_VERSION",
    "build_engine_pass_result_v4_4",
    "build_synthetic_fixture_v4_4",
    "canonical_json_bytes_v4_4",
    "engine_equivalence_payload_v4_4",
    "evaluate_local_formulas_v4_4",
    "evaluate_source_dag_v4_4",
    "matrix_hash_descriptor_v4_4",
    "semantic_sha256_v4_4",
    "source_programs_semantic_sha256_v4_4",
    "synthetic_fixture_binding_v4_4",
    "validate_engine_pass_result_v4_4",
    "validate_source_programs_v4_4",
]
