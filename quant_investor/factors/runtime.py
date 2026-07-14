"""Runtime scoring adapter for governed mined factors.

Only production factors that passed all eight governance gates are allowed to
feed the quant branch. If the registry is empty or no factor is selectable,
the Quant branch is governance-blocked; callers must not manufacture a legacy
proxy score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import date, datetime
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from quant_investor.factors.governance import FactorLifecycleState, FactorRecord

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "factor_registry" / "mined_factors.json"
)
PRODUCTION_RUNTIME_MODE = "production"
REPORT_ONLY_SHADOW_RUNTIME_MODE = "report_only_shadow"
PRODUCTION_RUNTIME_INPUT_SCHEMA_VERSION = "quant-production-runtime-input.v1"
PRODUCTION_RUNTIME_OUTPUT_SCHEMA_VERSION = "quant-production-runtime-output.v1"
PRODUCTION_EVALUATION_CONTEXT_SCHEMA_VERSION = (
    "quant-production-evaluation-context.v1"
)
_PRODUCTION_EVALUATION_CONTEXT_SEAL = object()


@dataclass(frozen=True, slots=True)
class ProductionEvaluationContext:
    """Immutable as-of and provenance boundary for production Quant."""

    evaluation_as_of: str
    market: str
    universe_key: str
    universe_sha256: str
    snapshot_id: str
    latest_complete_trade_date: str
    pit_membership_status: str
    pit_membership_as_of: str
    pit_membership_proof_sha256: str
    pit_membership_not_applicable_reason: str
    open_day_proof_sha256: str
    read_result_provenance_sha256: str
    verified_artifact_paths: tuple[tuple[str, str], ...] = ()
    verified_artifact_sha256s: tuple[tuple[str, str], ...] = ()
    schema_version: str = PRODUCTION_EVALUATION_CONTEXT_SCHEMA_VERSION
    _sealed_payload_sha256: str = field(default="", repr=False, compare=False)
    _seal: object | None = field(default=None, repr=False, compare=False)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_as_of": self.evaluation_as_of,
            "market": self.market,
            "universe_key": self.universe_key,
            "universe_sha256": self.universe_sha256,
            "snapshot_id": self.snapshot_id,
            "latest_complete_trade_date": self.latest_complete_trade_date,
            "pit_membership_status": self.pit_membership_status,
            "pit_membership_as_of": self.pit_membership_as_of,
            "pit_membership_proof_sha256": self.pit_membership_proof_sha256,
            "pit_membership_not_applicable_reason": (
                self.pit_membership_not_applicable_reason
            ),
            "open_day_proof_sha256": self.open_day_proof_sha256,
            "read_result_provenance_sha256": (
                self.read_result_provenance_sha256
            ),
            "verified_artifact_sha256s": {
                name: digest for name, digest in self.verified_artifact_sha256s
            },
            "verified_artifact_paths": {
                name: path for name, path in self.verified_artifact_paths
            },
        }

    @property
    def context_sha256(self) -> str:
        return production_evaluation_context_sha256(self)

    def to_metadata(self) -> dict[str, Any]:
        return {**self.to_payload(), "context_sha256": self.context_sha256}


def production_evaluation_context_sha256(
    context: ProductionEvaluationContext,
) -> str:
    """Hash the complete immutable evaluation-context payload."""

    if not isinstance(context, ProductionEvaluationContext):
        raise TypeError("production evaluation context type invalid")
    raw = json.dumps(
        context.to_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _mint_production_evaluation_context(
    *,
    evaluation_as_of: str,
    market: str,
    universe_key: str,
    universe_sha256: str,
    snapshot_id: str,
    latest_complete_trade_date: str,
    pit_membership_status: str,
    pit_membership_as_of: str,
    pit_membership_proof_sha256: str,
    pit_membership_not_applicable_reason: str,
    open_day_proof_sha256: str,
    read_result_provenance_sha256: str,
    verified_artifact_paths: Mapping[str, str],
    verified_artifact_sha256s: Mapping[str, str],
) -> ProductionEvaluationContext:
    """Mint a process-local context after authoritative readback succeeds."""

    context = ProductionEvaluationContext(
        evaluation_as_of=evaluation_as_of,
        market=market,
        universe_key=universe_key,
        universe_sha256=universe_sha256,
        snapshot_id=snapshot_id,
        latest_complete_trade_date=latest_complete_trade_date,
        pit_membership_status=pit_membership_status,
        pit_membership_as_of=pit_membership_as_of,
        pit_membership_proof_sha256=pit_membership_proof_sha256,
        pit_membership_not_applicable_reason=(
            pit_membership_not_applicable_reason
        ),
        open_day_proof_sha256=open_day_proof_sha256,
        read_result_provenance_sha256=read_result_provenance_sha256,
        verified_artifact_paths=tuple(
            sorted((str(name), str(path)) for name, path in verified_artifact_paths.items())
        ),
        verified_artifact_sha256s=tuple(
            sorted((str(name), str(digest)) for name, digest in verified_artifact_sha256s.items())
        ),
        _seal=_PRODUCTION_EVALUATION_CONTEXT_SEAL,
    )
    object.__setattr__(
        context,
        "_sealed_payload_sha256",
        context.context_sha256,
    )
    return context


def production_factor_set_sha256(names: Sequence[str]) -> str:
    """Hash the sorted selectable factor-name set for metadata/readback checks."""

    normalized = sorted({str(name).strip() for name in names if str(name).strip()})
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def production_symbol_set_sha256(symbols: Sequence[str]) -> str:
    """Hash the exact normalized production symbol set."""

    return production_factor_set_sha256(symbols)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _context_from_metadata(
    metadata: Mapping[str, Any],
) -> ProductionEvaluationContext | None:
    try:
        artifact_hashes = dict(metadata["verified_artifact_sha256s"])
        artifact_paths = dict(metadata["verified_artifact_paths"])
        context = ProductionEvaluationContext(
            evaluation_as_of=str(metadata["evaluation_as_of"]),
            market=str(metadata["market"]),
            universe_key=str(metadata["universe_key"]),
            universe_sha256=str(metadata["universe_sha256"]),
            snapshot_id=str(metadata["snapshot_id"]),
            latest_complete_trade_date=str(
                metadata["latest_complete_trade_date"]
            ),
            pit_membership_status=str(metadata["pit_membership_status"]),
            pit_membership_as_of=str(metadata["pit_membership_as_of"]),
            pit_membership_proof_sha256=str(
                metadata["pit_membership_proof_sha256"]
            ),
            pit_membership_not_applicable_reason=str(
                metadata["pit_membership_not_applicable_reason"]
            ),
            open_day_proof_sha256=str(metadata["open_day_proof_sha256"]),
            read_result_provenance_sha256=str(
                metadata["read_result_provenance_sha256"]
            ),
            verified_artifact_paths=tuple(
                sorted(
                    (str(name), str(path))
                    for name, path in artifact_paths.items()
                )
            ),
            verified_artifact_sha256s=tuple(
                sorted(
                    (str(name), str(digest))
                    for name, digest in artifact_hashes.items()
                )
            ),
            schema_version=str(metadata["schema_version"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if metadata.get("context_sha256") != context.context_sha256:
        return None
    return context


def validate_production_evaluation_context(
    context: ProductionEvaluationContext | None,
    *,
    expected_symbols: Sequence[str],
    _require_readback_seal: bool = True,
) -> list[str]:
    """Return fail-closed evaluation-context blockers."""

    if context is None:
        return ["production_evaluation_context_missing"]
    if not isinstance(context, ProductionEvaluationContext):
        return ["production_evaluation_context_type_invalid"]
    blockers: list[str] = []
    if _require_readback_seal and (
        context._seal is not _PRODUCTION_EVALUATION_CONTEXT_SEAL
        or context._sealed_payload_sha256 != context.context_sha256
    ):
        blockers.append("production_evaluation_context_not_readback_verified")
    symbols = [str(symbol) for symbol in expected_symbols]
    if context.schema_version != PRODUCTION_EVALUATION_CONTEXT_SCHEMA_VERSION:
        blockers.append("production_evaluation_context_schema_invalid")
    if not re.fullmatch(r"\d{8}", context.evaluation_as_of):
        blockers.append("production_evaluation_as_of_invalid")
    else:
        parsed_as_of = pd.to_datetime(
            context.evaluation_as_of,
            format="%Y%m%d",
            errors="coerce",
        )
        if pd.isna(parsed_as_of):
            blockers.append("production_evaluation_as_of_invalid")
    if not context.market or context.market != context.market.upper():
        blockers.append("production_evaluation_market_invalid")
    if not context.universe_key.strip():
        blockers.append("production_evaluation_universe_key_missing")
    if (
        not symbols
        or any(not symbol for symbol in symbols)
        or len(symbols) != len(set(symbols))
        or context.universe_sha256 != production_symbol_set_sha256(symbols)
    ):
        blockers.append("production_evaluation_universe_sha256_mismatch")
    if not context.snapshot_id.strip():
        blockers.append("production_snapshot_id_missing")
    if context.latest_complete_trade_date != context.evaluation_as_of:
        blockers.append("production_latest_complete_trade_date_mismatch")
    if not _is_sha256(context.open_day_proof_sha256):
        blockers.append("production_open_day_proof_missing_or_invalid")
    if not _is_sha256(context.read_result_provenance_sha256):
        blockers.append("production_read_result_provenance_missing_or_invalid")
    artifact_hashes = dict(context.verified_artifact_sha256s)
    artifact_paths = dict(context.verified_artifact_paths)
    if (
        len(artifact_hashes) != len(context.verified_artifact_sha256s)
        or len(artifact_paths) != len(context.verified_artifact_paths)
        or set(artifact_paths) != set(artifact_hashes)
        or any(not name or not _is_sha256(digest) for name, digest in artifact_hashes.items())
        or "snapshot_pointer" not in artifact_hashes
        or "snapshot_manifest" not in artifact_hashes
    ):
        blockers.append("production_verified_artifact_set_invalid")
    else:
        for name, expected_sha in artifact_hashes.items():
            try:
                raw_path = Path(artifact_paths[name]).expanduser()
                if raw_path.is_symlink():
                    blockers.append(f"production_verified_artifact_symlink:{name}")
                    continue
                path = raw_path.resolve()
            except (OSError, RuntimeError, ValueError):
                blockers.append(f"production_verified_artifact_path_invalid:{name}")
                continue
            if not path.is_file():
                blockers.append(f"production_verified_artifact_missing:{name}")
                continue
            try:
                current_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                blockers.append(f"production_verified_artifact_unreadable:{name}")
                continue
            if current_sha != expected_sha:
                blockers.append(f"production_verified_artifact_bytes_drift:{name}")
    if context.market == "CN":
        if context.pit_membership_status != "verified":
            blockers.append("production_cn_pit_membership_not_verified")
        if context.pit_membership_as_of != context.evaluation_as_of:
            blockers.append("production_pit_membership_as_of_mismatch")
        if not _is_sha256(context.pit_membership_proof_sha256):
            blockers.append("production_pit_membership_proof_missing_or_invalid")
        if context.pit_membership_not_applicable_reason:
            blockers.append("production_cn_pit_not_applicable_forbidden")
        if (
            "pit_manifest" not in artifact_hashes
            or "pit_canonical" not in artifact_hashes
        ):
            blockers.append("production_cn_pit_artifact_readback_missing")
    else:
        if context.pit_membership_status != "not_applicable":
            blockers.append("production_non_cn_pit_status_invalid")
        if (
            context.pit_membership_as_of
            or context.pit_membership_proof_sha256
            or not context.pit_membership_not_applicable_reason.strip()
        ):
            blockers.append("production_non_cn_pit_not_applicable_invalid")
    return list(dict.fromkeys(blockers))


def _strict_daily_trade_dates(
    values: pd.Series,
) -> tuple[pd.DatetimeIndex | None, str | None]:
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return None, "production_frame_trade_date_timezone_aware"
    parsed: list[pd.Timestamp] = []
    for value in values:
        if value is None or value is pd.NaT or pd.isna(value):
            return None, "production_frame_trade_date_unparseable"
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"\d{8}", text):
                timestamp = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                timestamp = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
            else:
                return None, "production_frame_trade_date_unparseable"
        elif isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
            timestamp = pd.Timestamp(value)
        else:
            return None, "production_frame_trade_date_unparseable"
        if pd.isna(timestamp):
            return None, "production_frame_trade_date_unparseable"
        if timestamp.tzinfo is not None:
            return None, "production_frame_trade_date_timezone_aware"
        if timestamp != timestamp.normalize():
            return None, "production_frame_trade_date_not_daily"
        parsed.append(timestamp)
    return pd.DatetimeIndex(parsed), None


def _validate_production_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    symbols: Sequence[str],
    context: ProductionEvaluationContext,
) -> str | None:
    as_of = pd.to_datetime(
        context.evaluation_as_of,
        format="%Y%m%d",
        errors="coerce",
    )
    if pd.isna(as_of):
        return "production_evaluation_as_of_invalid"
    for symbol in symbols:
        frame = frames.get(symbol)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return f"production_frame_missing_or_empty:{symbol}"
        symbol_columns = [
            column for column in ("ts_code", "symbol") if column in frame.columns
        ]
        if not symbol_columns:
            return f"production_frame_symbol_column_missing:{symbol}"
        for column in symbol_columns:
            values = [str(value).strip() for value in frame[column]]
            if any(value != symbol for value in values):
                return f"production_frame_symbol_mismatch:{symbol}:{column}"
        if "trade_date" not in frame.columns:
            return f"production_frame_trade_date_missing:{symbol}"
        dates, error = _strict_daily_trade_dates(frame["trade_date"])
        if error:
            return f"{error}:{symbol}"
        assert dates is not None
        if dates.duplicated().any():
            return f"production_frame_duplicate_trade_date:{symbol}"
        if not dates.is_monotonic_increasing:
            return f"production_frame_date_order_invalid:{symbol}"
        if (dates > as_of).any():
            return f"production_frame_future_row:{symbol}"
        if dates[-1] != as_of:
            return f"production_frame_terminal_date_mismatch:{symbol}"
    return None


def _update_runtime_digest(
    digest: Any,
    label: str,
    value: str | bytes | bytearray,
) -> None:
    """Add one unambiguous length-prefixed field to a streaming digest."""

    label_bytes = label.encode("utf-8")
    value_bytes = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(label_bytes).to_bytes(4, "big"))
    digest.update(label_bytes)
    digest.update(len(value_bytes).to_bytes(8, "big"))
    digest.update(value_bytes)


def _canonical_runtime_scalar_bytes(value: Any) -> bytes:
    """Encode one consumed scalar without an intermediate lossy hash."""

    if value is None or value is pd.NA or value is pd.NaT:
        return b"N"
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return b"N"
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        return b"D" + pd.Timestamp(value).isoformat().encode("utf-8")
    if isinstance(value, (bool, np.bool_)):
        return b"B\x01" if bool(value) else b"B\x00"
    if isinstance(value, Integral):
        return b"I" + str(int(value)).encode("ascii")
    if isinstance(value, Real):
        numeric = float(value)
        if math.isnan(numeric):
            encoded = "nan"
        elif math.isinf(numeric):
            encoded = "+inf" if numeric > 0.0 else "-inf"
        else:
            encoded = numeric.hex()
        return b"F" + encoded.encode("ascii")
    if isinstance(value, str):
        return b"S" + value.encode("utf-8")
    raise TypeError(f"unsupported production runtime scalar: {type(value).__name__}")


def production_runtime_input_sha256(
    frames: Mapping[str, pd.DataFrame],
    contracts: Mapping[str, Any],
) -> str:
    """Hash the actual factor-required frame values consumed in production."""

    requirements: dict[str, tuple[list[str], int]] = {}
    for raw_factor_name, contract in contracts.items():
        factor_name = str(raw_factor_name)
        if not factor_name or factor_name in requirements:
            raise ValueError("runtime contract names must be unique and non-empty")
        if not isinstance(contract, Mapping):
            raise ValueError(f"runtime contract is not an object: {factor_name}")
        columns = contract.get("required_columns")
        lookback = contract.get("lookback_rows")
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(columns) != len(set(columns))
            or isinstance(lookback, bool)
            or not isinstance(lookback, int)
            or lookback <= 0
        ):
            raise ValueError(f"runtime contract input shape invalid: {factor_name}")
        requirements[factor_name] = (list(columns), lookback)

    normalized_frames: dict[str, pd.DataFrame] = {}
    for raw_symbol, frame in frames.items():
        symbol = str(raw_symbol)
        if not symbol or symbol in normalized_frames:
            raise ValueError("production runtime symbols must be unique and non-empty")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"production runtime frame invalid: {symbol}")
        normalized_frames[symbol] = frame

    from quant_investor.factors.price_volume import _ordered_frame

    digest = hashlib.sha256()
    _update_runtime_digest(
        digest, "schema_version", PRODUCTION_RUNTIME_INPUT_SCHEMA_VERSION
    )
    _update_runtime_digest(digest, "factor_count", str(len(requirements)))
    _update_runtime_digest(digest, "symbol_count", str(len(normalized_frames)))
    _update_runtime_digest(
        digest,
        "symbol_set_sha256",
        production_symbol_set_sha256(list(normalized_frames)),
    )
    for factor_name in sorted(requirements):
        columns, lookback = requirements[factor_name]
        _update_runtime_digest(digest, "factor", factor_name)
        _update_runtime_digest(digest, "lookback_rows", str(lookback))
        _update_runtime_digest(digest, "column_count", str(len(columns)))
        for column in columns:
            _update_runtime_digest(digest, "column", column)
        for symbol in sorted(normalized_frames):
            frame = normalized_frames[symbol]
            if any(column not in frame.columns for column in columns):
                raise ValueError(
                    f"production runtime required column missing: {symbol}"
                )
            consumed = _ordered_frame(frame, lookback_rows=lookback).loc[
                :, columns
            ]
            if len(consumed) != lookback:
                raise ValueError(
                    f"production runtime lookback missing: {factor_name}:{symbol}"
                )
            _update_runtime_digest(digest, "symbol", symbol)
            _update_runtime_digest(digest, "row_count", str(len(consumed)))
            for column, dtype in zip(columns, consumed.dtypes):
                _update_runtime_digest(digest, f"dtype:{column}", str(dtype))
                encoded_column = bytearray()
                for value in consumed[column]:
                    encoded_scalar = _canonical_runtime_scalar_bytes(value)
                    encoded_column.extend(
                        len(encoded_scalar).to_bytes(8, "big")
                    )
                    encoded_column.extend(encoded_scalar)
                _update_runtime_digest(
                    digest,
                    f"values:{column}",
                    encoded_column,
                )
    return digest.hexdigest()


def _symbol_scores_sha256(symbol_scores: Mapping[str, Any]) -> str:
    normalized: dict[str, str] = {}
    for raw_symbol, raw_score in symbol_scores.items():
        symbol = str(raw_symbol)
        if not symbol or symbol in normalized:
            raise ValueError("production score symbols must be unique and non-empty")
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            raise TypeError(f"production score is not numeric: {symbol}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"production score is not finite: {symbol}")
        normalized[symbol] = score.hex()
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _production_output_attestation_sha256(metadata: Mapping[str, Any]) -> str:
    registry = dict(metadata.get("registry", {}) or {})
    governance = dict(registry.get("governance_runtime", {}) or {})
    activation = dict(governance.get("quant_production_activation", {}) or {})
    payload = {
        "schema_version": PRODUCTION_RUNTIME_OUTPUT_SCHEMA_VERSION,
        "production_input_sha256": metadata.get("production_input_sha256"),
        "production_evaluation_context_sha256": metadata.get(
            "production_evaluation_context_sha256"
        ),
        "symbol_count": metadata.get("symbol_count"),
        "symbol_set_sha256": metadata.get("symbol_set_sha256"),
        "symbol_scores_sha256": metadata.get("symbol_scores_sha256"),
        "factor_count": metadata.get("factor_count"),
        "factors_used": list(metadata.get("factors_used", []) or []),
        "factor_weights": dict(metadata.get("factor_weights", {}) or {}),
        "factor_coverages": dict(metadata.get("factor_coverages", {}) or {}),
        "registry_sha256": registry.get("registry_sha256"),
        "production_factor_set_sha256": governance.get(
            "production_factor_set_sha256"
        ),
        "production_runtime_contracts_sha256": governance.get(
            "factor_runtime_contracts_sha256"
        ),
        "activation_receipt_file_sha256": activation.get("receipt_file_sha256"),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class MinedFactorRegistry:
    schema_version: str = "mined-factor-registry.v1"
    factors: list[FactorRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MinedFactorRegistry":
        return cls(
            schema_version=str(payload.get("schema_version", "mined-factor-registry.v1")),
            factors=[
                FactorRecord.from_dict(item)
                for item in payload.get("factors", [])
                if isinstance(item, Mapping) and str(item.get("name", "")).strip()
            ],
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def from_records(cls, records: Sequence[FactorRecord]) -> "MinedFactorRegistry":
        return cls(factors=list(records))

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "MinedFactorRegistry":
        """Load the research/report registry with historical forgiving semantics."""

        raw_path = path or os.getenv("MYQUANT_FACTOR_REGISTRY") or DEFAULT_REGISTRY_PATH
        registry_path = Path(raw_path).expanduser()
        if not registry_path.exists():
            return cls(metadata={"path": str(registry_path), "missing": True})
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            registry = cls.from_dict(payload if isinstance(payload, Mapping) else {})
            registry.metadata.setdefault("path", str(registry_path))
            return registry
        except Exception as exc:
            return cls(metadata={"path": str(registry_path), "load_error": str(exc)})

    @classmethod
    def load_production(
        cls,
        path: str | os.PathLike[str] | None = None,
    ) -> "MinedFactorRegistry":
        """Load production bytes through the strict raw-payload validator."""

        raw_path = path or os.getenv("MYQUANT_FACTOR_REGISTRY") or DEFAULT_REGISTRY_PATH
        registry_path = Path(raw_path).expanduser()
        try:
            from quant_investor.factors.registry_store import (
                load_registry_snapshot_strict,
            )

            snapshot = load_registry_snapshot_strict(registry_path)
        except Exception as exc:
            return cls(
                metadata={
                    "path": str(registry_path),
                    "strict_loader": True,
                    "strict_load_error": str(exc),
                    "load_error": str(exc),
                }
            )
        registry = snapshot.registry
        registry.metadata = {
            **dict(registry.metadata),
            "path": str(snapshot.path),
            "strict_loader": True,
            "registry_sha256": snapshot.registry_sha256,
            "record_sha256s": dict(snapshot.record_sha256s),
        }
        return registry

    def selectable_factors(self) -> list[FactorRecord]:
        return [factor for factor in self.factors if factor.selectable_in_quant_branch()]

    def selectable_manifest(self) -> dict[str, Any]:
        names = sorted(factor.name for factor in self.selectable_factors())
        return {
            "production_factor_count": len(names),
            "production_factor_names": names,
            "production_factor_set_sha256": production_factor_set_sha256(names),
        }

    def non_selectable_reasons(self) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for factor in self.factors:
            if factor.selectable_in_quant_branch():
                continue
            if factor.state != FactorLifecycleState.PRODUCTION_FACTOR:
                reasons[factor.name] = f"state={factor.state.value}"
            elif not factor.all_gates_passed():
                reasons[factor.name] = "not_all_gates_passed"
            elif not float(factor.weight):
                reasons[factor.name] = "zero_weight"
            elif factor.deprecated_reason:
                reasons[factor.name] = f"deprecated={factor.deprecated_reason}"
            else:
                reasons[factor.name] = "not_selectable"
        return reasons


@dataclass
class RuntimeFactorScore:
    symbol_scores: dict[str, float] = field(default_factory=dict)
    factor_count: int = 0
    factors_used: list[str] = field(default_factory=list)
    factor_weights: dict[str, float] = field(default_factory=dict)
    factor_coverages: dict[str, float] = field(default_factory=dict)
    skipped_factors: dict[str, str] = field(default_factory=dict)
    registry_metadata: dict[str, Any] = field(default_factory=dict)
    governance_status: str = "governance_blocked"
    factor_mode: str = "governance_blocked"
    confidence_multiplier: float = 0.0
    production_eligible: bool = False
    runtime_mode: str = PRODUCTION_RUNTIME_MODE
    runtime_blockers: list[str] = field(default_factory=list)
    production_input_sha256: str = ""
    production_evaluation_context: dict[str, Any] = field(default_factory=dict)
    production_evaluation_context_sha256: str = ""
    production_output_attestation_sha256: str = ""

    @property
    def coverage_rate(self) -> float:
        if self.factor_coverages:
            values = [
                max(0.0, min(1.0, float(value)))
                for value in self.factor_coverages.values()
            ]
            return float(sum(values) / max(len(values), 1))
        if not self.symbol_scores:
            return 0.0
        non_zero = sum(
            1
            for value in self.symbol_scores.values()
            if abs(float(value)) > 1e-12
        )
        return non_zero / max(len(self.symbol_scores), 1)

    def to_metadata(self) -> dict[str, Any]:
        applied_to_score = bool(self.factor_count > 0 and self.factors_used)
        symbols = [str(symbol) for symbol in self.symbol_scores]
        try:
            symbol_scores_sha256 = _symbol_scores_sha256(self.symbol_scores)
        except (TypeError, ValueError):
            symbol_scores_sha256 = ""
        return {
            "factor_count": self.factor_count,
            "factors_used": list(self.factors_used),
            "factor_weights": dict(self.factor_weights),
            "factor_coverages": dict(self.factor_coverages),
            "skipped_factors": dict(self.skipped_factors),
            "coverage_rate": self.coverage_rate,
            "applied_to_score": applied_to_score,
            "score_weight": float(
                sum(abs(float(weight)) for weight in self.factor_weights.values())
            )
            if applied_to_score
            else 0.0,
            "registry": dict(self.registry_metadata),
            "governance_status": self.governance_status,
            "factor_mode": self.factor_mode,
            "confidence_multiplier": float(self.confidence_multiplier),
            "production_eligible": bool(self.production_eligible),
            "runtime_mode": self.runtime_mode,
            "runtime_blockers": list(self.runtime_blockers),
            "symbol_count": len(symbols),
            "symbol_set_sha256": production_symbol_set_sha256(symbols),
            "symbol_scores_sha256": symbol_scores_sha256,
            "production_input_sha256": self.production_input_sha256,
            "production_evaluation_context": dict(
                self.production_evaluation_context
            ),
            "production_evaluation_context_sha256": (
                self.production_evaluation_context_sha256
            ),
            "production_output_attestation_sha256": (
                self.production_output_attestation_sha256
            ),
            "legacy_fallback_allowed": False,
        }


def _factor_window_from_name(name: str, default: int = 20) -> int:
    try:
        suffix = str(name).strip().rsplit("_", 1)[1]
        return max(int(suffix.removesuffix("d")), 1)
    except Exception:
        return int(default)


def _factor_window_pair_from_name(
    name: str,
    *,
    default: tuple[int, int] = (20, 5),
) -> tuple[int, int]:
    parts = str(name).strip().split("_")
    try:
        first = int(parts[-2].removesuffix("d"))
        second = int(parts[-1].removesuffix("d"))
        return max(first, 1), max(second, 1)
    except Exception:
        return default


def _price_volume_factor_lookback_rows(name: str) -> int:
    factor_name = str(name or "").strip()
    if not factor_name:
        return 0
    if factor_name.startswith("pv_blend_volstab19x2_mom90_amihud5_w"):
        return 91
    if factor_name.startswith("pv_volume_stability_smooth_"):
        base_window, smooth_window = _factor_window_pair_from_name(factor_name)
        return base_window + smooth_window
    if factor_name.startswith("pv_dollar_volume_growth_"):
        short_window, long_window = _factor_window_pair_from_name(
            factor_name,
            default=(20, 60),
        )
        return max(short_window, long_window)
    if factor_name.startswith(
        (
            "pv_momentum_",
            "pv_short_reversal_",
            "pv_volatility_penalty_",
            "pv_downside_volatility_",
            "pv_price_efficiency_",
            "pv_amihud_illiquidity_",
        )
    ):
        return _factor_window_from_name(factor_name) + 1
    if factor_name.startswith(
        (
            "pv_volume_stability_",
            "pv_low_dollar_volume_",
            "pv_high_dollar_volume_",
        )
    ):
        return _factor_window_from_name(factor_name)
    return 0


def _price_volume_required_lookback_rows(names: Sequence[str]) -> int:
    return max(
        (_price_volume_factor_lookback_rows(name) for name in names),
        default=0,
    )


class MinedFactorScorer:
    """Compute latest cross-sectional scores from governed production factors."""

    def __init__(
        self,
        registry: MinedFactorRegistry | None = None,
        *,
        runtime_mode: str = PRODUCTION_RUNTIME_MODE,
    ) -> None:
        normalized_mode = str(runtime_mode or "").strip()
        if normalized_mode not in {
            PRODUCTION_RUNTIME_MODE,
            REPORT_ONLY_SHADOW_RUNTIME_MODE,
        }:
            raise ValueError(f"unsupported factor runtime mode: {runtime_mode!r}")
        self.runtime_mode = normalized_mode
        if registry is not None:
            self.registry = registry
        elif self.runtime_mode == PRODUCTION_RUNTIME_MODE:
            self.registry = MinedFactorRegistry.load_production()
        else:
            self.registry = MinedFactorRegistry.load()

    def _runtime_contract(self) -> tuple[list[FactorRecord], dict[str, Any]]:
        if self.runtime_mode == REPORT_ONLY_SHADOW_RUNTIME_MODE:
            candidates = (
                self.registry.factors
                if self.registry.metadata.get("historical_shadow_only") is True
                else self.registry.selectable_factors()
            )
            active = [
                record
                for record in candidates
                if str(record.name).strip()
                and str(record.implementation).strip()
                and abs(float(record.weight)) > 1e-12
            ]
            blockers = [] if active else ["report_only_shadow_factor_set_empty"]
            return active, {
                "status": "report_only" if active else "governance_blocked",
                "factor_mode": (
                    "historical_shadow_report_only"
                    if active
                    else "governance_blocked"
                ),
                "confidence_multiplier": 0.0,
                "production_eligible": False,
                "legacy_fallback_allowed": False,
                "blockers": blockers,
            }

        # Local import avoids the module-load cycle: protocol v2 owns the
        # complete production readiness contract and imports this registry type.
        from quant_investor.factors.governance_protocol_v2 import (
            governance_runtime_status,
        )

        status = governance_runtime_status(self.registry)
        active = (
            sorted(
                self.registry.selectable_factors(),
                key=lambda record: record.name,
            )
            if status["status"] == "ready"
            else []
        )
        return active, {
            **status,
            "production_eligible": status["status"] == "ready",
        }

    def _empty_score(
        self,
        symbols: Sequence[str],
        *,
        skipped: Mapping[str, str],
        runtime_status: Mapping[str, Any],
    ) -> RuntimeFactorScore:
        return RuntimeFactorScore(
            symbol_scores={str(symbol): 0.0 for symbol in symbols},
            skipped_factors=dict(skipped),
            registry_metadata={
                **dict(self.registry.metadata),
                "governance_runtime": dict(runtime_status),
            },
            governance_status=str(
                runtime_status.get("status") or "governance_blocked"
            ),
            factor_mode=str(
                runtime_status.get("factor_mode") or "governance_blocked"
            ),
            confidence_multiplier=float(
                runtime_status.get("confidence_multiplier") or 0.0
            ),
            production_eligible=bool(
                runtime_status.get("production_eligible", False)
            ),
            runtime_mode=self.runtime_mode,
            runtime_blockers=[
                str(item)
                for item in runtime_status.get("blockers", []) or []
                if str(item)
            ],
        )

    def score(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        evaluation_context: ProductionEvaluationContext | None = None,
    ) -> RuntimeFactorScore:
        symbols = [str(symbol) for symbol in frames if str(symbol).strip()]
        active, runtime_status = self._runtime_contract()
        non_selectable = self.registry.non_selectable_reasons()
        active_names = {record.name for record in active}
        skipped = {
            name: reason
            for name, reason in non_selectable.items()
            if name not in active_names
        }
        if not symbols:
            return self._empty_score(
                [],
                skipped=skipped,
                runtime_status=runtime_status,
            )

        if self.runtime_mode == PRODUCTION_RUNTIME_MODE:
            context_blockers = validate_production_evaluation_context(
                evaluation_context,
                expected_symbols=symbols,
            )
            if context_blockers:
                blocked = dict(runtime_status)
                for blocker in context_blockers:
                    blocked = self._blocked_runtime_status(blocked, blocker)
                return self._empty_score(
                    symbols,
                    skipped=skipped,
                    runtime_status=blocked,
                )

        if not active:
            return self._empty_score(
                symbols,
                skipped=skipped,
                runtime_status=runtime_status,
            )

        if self.runtime_mode == PRODUCTION_RUNTIME_MODE:
            assert evaluation_context is not None
            return self._score_production(
                frames,
                symbols=symbols,
                active=active,
                skipped=skipped,
                runtime_status=runtime_status,
                evaluation_context=evaluation_context,
            )

        weighted_scores = pd.Series(0.0, index=symbols, dtype=float)
        total_weight = 0.0
        factors_used: list[str] = []
        factor_weights: dict[str, float] = {}
        factor_coverages: dict[str, float] = {}
        price_volume_prepared: Mapping[str, Any] | None = None
        price_volume_factor_cache: dict[str, Any] = {}
        price_volume_names = [
            str(factor.implementation or "").strip().split(":", 1)[1]
            for factor in active
            if str(factor.implementation or "").strip().startswith("price_volume:")
        ]
        price_volume_factor_cache["active_price_volume_names"] = tuple(price_volume_names)
        price_volume_lookback_rows = _price_volume_required_lookback_rows(
            price_volume_names
        )
        include_amihud_base = any(
            name.startswith("pv_amihud_illiquidity_")
            or name.startswith("pv_blend_volstab19x2_mom90_amihud5_w")
            for name in price_volume_names
        )

        for factor in active:
            try:
                impl = str(factor.implementation or "").strip()
                if impl.startswith("price_volume:"):
                    if price_volume_prepared is None:
                        from quant_investor.factors.price_volume import (
                            prepare_price_volume_frames,
                        )

                        price_volume_prepared = prepare_price_volume_frames(
                            frames,
                            include_amihud_base=include_amihud_base,
                            lookback_rows=price_volume_lookback_rows,
                        )
                    raw = self._price_volume_factor(
                        impl.split(":", 1)[1],
                        frames,
                        prepared_frames=price_volume_prepared,
                        factor_cache=price_volume_factor_cache,
                    )
                else:
                    raw = self._compute_factor(factor, frames)
            except Exception as exc:
                skipped[factor.name] = f"compute_error={exc}"
                continue
            valid = raw.replace([np.inf, -np.inf], np.nan).dropna()
            if valid.empty:
                skipped[factor.name] = "empty_factor_values"
                continue
            normalized = self._rank_normalize(raw.reindex(symbols))
            weight = float(factor.weight) * (1.0 if float(factor.direction) >= 0 else -1.0)
            if abs(weight) <= 1e-12:
                skipped[factor.name] = "zero_effective_weight"
                continue
            weighted_scores = weighted_scores.add(normalized.fillna(0.0) * weight, fill_value=0.0)
            total_weight += abs(weight)
            factors_used.append(factor.name)
            factor_weights[factor.name] = weight
            factor_coverages[factor.name] = float(
                valid.index.intersection(symbols).size / max(len(symbols), 1)
            )

        if total_weight <= 1e-12 or not factors_used:
            runtime_status = {
                **runtime_status,
                "status": "governance_blocked",
                "factor_mode": "governance_blocked",
                "confidence_multiplier": 0.0,
                "production_eligible": False,
                "blockers": [
                    *list(runtime_status.get("blockers", []) or []),
                    "no_runtime_factor_completed",
                ],
            }
            return self._empty_score(
                symbols,
                skipped=skipped,
                runtime_status=runtime_status,
            )

        symbol_scores = (weighted_scores / total_weight).clip(-1.0, 1.0).fillna(0.0)
        return RuntimeFactorScore(
            symbol_scores={symbol: float(symbol_scores.get(symbol, 0.0)) for symbol in symbols},
            factor_count=len(factors_used),
            factors_used=factors_used,
            factor_weights=factor_weights,
            factor_coverages=factor_coverages,
            skipped_factors=skipped,
            registry_metadata={
                **dict(self.registry.metadata),
                "governance_runtime": dict(runtime_status),
            },
            governance_status=str(runtime_status["status"]),
            factor_mode=str(runtime_status["factor_mode"]),
            confidence_multiplier=float(
                runtime_status.get("confidence_multiplier") or 0.0
            ),
            production_eligible=bool(
                runtime_status.get("production_eligible", False)
            ),
            runtime_mode=self.runtime_mode,
            runtime_blockers=[
                str(item)
                for item in runtime_status.get("blockers", []) or []
                if str(item)
            ],
        )

    @staticmethod
    def _blocked_runtime_status(
        runtime_status: Mapping[str, Any],
        blocker: str,
    ) -> dict[str, Any]:
        return {
            **dict(runtime_status),
            "status": "governance_blocked",
            "factor_mode": "governance_blocked",
            "confidence_multiplier": 0.0,
            "production_eligible": False,
            "blockers": list(
                dict.fromkeys(
                    [
                        *[
                            str(item)
                            for item in runtime_status.get("blockers", []) or []
                            if str(item)
                        ],
                        blocker,
                    ]
                )
            ),
        }

    def _score_production(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        symbols: Sequence[str],
        active: Sequence[FactorRecord],
        skipped: Mapping[str, str],
        runtime_status: Mapping[str, Any],
        evaluation_context: ProductionEvaluationContext,
    ) -> RuntimeFactorScore:
        """Execute the exact active set atomically without data substitution."""

        frame_blocker = _validate_production_frames(
            frames,
            symbols=symbols,
            context=evaluation_context,
        )
        if frame_blocker:
            blocked = self._blocked_runtime_status(runtime_status, frame_blocker)
            return self._empty_score(
                symbols,
                skipped=skipped,
                runtime_status=blocked,
            )

        contracts = runtime_status.get("factor_runtime_contracts")
        if not isinstance(contracts, Mapping) or set(contracts) != {
            factor.name for factor in active
        }:
            blocked = self._blocked_runtime_status(
                runtime_status,
                "production_runtime_contract_factor_set_mismatch",
            )
            return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)

        factor_series: dict[str, pd.Series] = {}
        factor_weights: dict[str, float] = {}
        factor_coverages: dict[str, float] = {}
        prepared: Mapping[str, Any] | None = None
        factor_cache: dict[str, Any] = {}
        price_volume_names = [
            str(factor.implementation).split(":", 1)[1]
            for factor in active
            if str(factor.implementation).startswith("price_volume:")
        ]
        factor_cache["active_price_volume_names"] = tuple(price_volume_names)
        include_amihud_base = any(
            name.startswith("pv_amihud_illiquidity_")
            or name.startswith("pv_blend_volstab19x2_mom90_amihud5_w")
            for name in price_volume_names
        )
        required_lookback = max(
            (
                int(dict(contracts[factor.name]).get("lookback_rows", 0) or 0)
                for factor in active
            ),
            default=0,
        )

        for factor in active:
            raw_contract = contracts.get(factor.name)
            if not isinstance(raw_contract, Mapping):
                blocker = f"factor_runtime_contract_missing:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            contract = dict(raw_contract)
            required_columns = contract.get("required_columns")
            lookback = contract.get("lookback_rows")
            minimum_coverage = contract.get("gate2_min_coverage_rate")
            min_cross_section = contract.get("min_cross_section")
            if (
                not isinstance(required_columns, list)
                or isinstance(lookback, bool)
                or not isinstance(lookback, int)
                or lookback <= 0
                or isinstance(minimum_coverage, bool)
                or not isinstance(minimum_coverage, (int, float))
                or not math.isfinite(float(minimum_coverage))
                or isinstance(min_cross_section, bool)
                or not isinstance(min_cross_section, int)
                or min_cross_section <= 0
            ):
                blocker = f"factor_runtime_contract_invalid:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            if len(symbols) < min_cross_section:
                blocker = f"factor_min_cross_section_not_met:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)

            valid_frames: dict[str, pd.DataFrame] = {}
            for symbol in symbols:
                frame = frames.get(symbol)
                if frame is None or frame.empty:
                    continue
                if any(column not in frame.columns for column in required_columns):
                    continue
                tail = frame.tail(lookback)
                if len(tail) < lookback:
                    continue
                if tail[required_columns].isna().any(axis=None):
                    continue
                columns_valid = True
                for column in required_columns:
                    values = tail[column]
                    if column == "trade_date":
                        normalized_dates = values.astype(str).str.strip()
                        if (
                            normalized_dates.eq("").any()
                            or normalized_dates.duplicated().any()
                            or not normalized_dates.is_monotonic_increasing
                        ):
                            columns_valid = False
                            break
                        continue
                    if (
                        pd.api.types.is_bool_dtype(values.dtype)
                        or not pd.api.types.is_numeric_dtype(values.dtype)
                    ):
                        columns_valid = False
                        break
                    numeric_values = values.to_numpy(dtype=float)
                    if (
                        not np.isfinite(numeric_values).all()
                        or (numeric_values <= 0.0).any()
                    ):
                        columns_valid = False
                        break
                if not columns_valid:
                    continue
                valid_frames[symbol] = frame
            coverage = len(valid_frames) / max(len(symbols), 1)
            if set(valid_frames) != set(symbols):
                blocker = (
                    f"factor_required_columns_or_lookback_missing:{factor.name}:"
                    f"coverage={coverage:.6f}"
                )
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            if coverage < float(minimum_coverage) - 1e-12:
                blocker = f"factor_gate2_runtime_coverage_below_contract:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            if len(valid_frames) < min_cross_section:
                blocker = f"factor_runtime_cross_section_below_contract:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            implementation = str(factor.implementation or "").strip()
            if not implementation.startswith("price_volume:"):
                blocker = f"factor_implementation_not_allowlisted:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            implementation_name = implementation.split(":", 1)[1]
            if implementation_name != factor.name:
                blocker = f"factor_implementation_name_mismatch:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            try:
                if prepared is None:
                    from quant_investor.factors.price_volume import (
                        prepare_price_volume_frames,
                    )

                    prepared = prepare_price_volume_frames(
                        valid_frames,
                        include_amihud_base=include_amihud_base,
                        lookback_rows=required_lookback,
                    )
                raw = self._price_volume_factor(
                    implementation_name,
                    valid_frames,
                    prepared_frames=prepared,
                    factor_cache=factor_cache,
                )
            except Exception as exc:
                blocker = f"factor_compute_error:{factor.name}:{exc}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            if not isinstance(raw, pd.Series) or raw.empty:
                blocker = f"factor_empty_output:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            normalized_index = [str(item) for item in raw.index]
            if len(normalized_index) != len(set(normalized_index)):
                blocker = f"factor_duplicate_output_index:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            if set(normalized_index) != set(symbols):
                blocker = f"factor_output_symbol_set_mismatch:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            raw = pd.Series(raw.to_numpy(), index=normalized_index)
            numeric = pd.to_numeric(raw.reindex(symbols), errors="coerce")
            if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
                blocker = f"factor_non_finite_or_missing_output:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            if numeric.nunique(dropna=False) <= 1:
                blocker = f"factor_constant_output:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            normalized = self._rank_normalize(numeric)
            if normalized.isna().any():
                blocker = f"factor_normalization_missing_output:{factor.name}"
                blocked = self._blocked_runtime_status(runtime_status, blocker)
                return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
            weight = float(factor.weight) * (
                1.0 if float(factor.direction) >= 0 else -1.0
            )
            factor_series[factor.name] = normalized
            factor_weights[factor.name] = weight
            factor_coverages[factor.name] = 1.0

        expected_names = [factor.name for factor in active]
        if list(factor_series) != expected_names:
            blocked = self._blocked_runtime_status(
                runtime_status,
                "production_factor_execution_set_mismatch",
            )
            return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
        total_abs_weight = sum(abs(value) for value in factor_weights.values())
        if not math.isfinite(total_abs_weight) or total_abs_weight <= 1e-12:
            blocked = self._blocked_runtime_status(
                runtime_status,
                "production_factor_total_abs_weight_invalid",
            )
            return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
        matrix = pd.DataFrame(factor_series, index=symbols)
        if matrix.isna().any(axis=None):
            blocked = self._blocked_runtime_status(
                runtime_status,
                "production_factor_matrix_missing_output",
            )
            return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
        weighted = sum(
            matrix[name] * factor_weights[name]
            for name in expected_names
        )
        symbol_scores = (weighted / total_abs_weight).clip(-1.0, 1.0)
        try:
            production_input_sha256 = production_runtime_input_sha256(
                frames,
                contracts,
            )
        except (TypeError, ValueError) as exc:
            blocked = self._blocked_runtime_status(
                runtime_status,
                f"production_runtime_input_attestation_failed:{exc}",
            )
            return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
        result = RuntimeFactorScore(
            symbol_scores={symbol: float(symbol_scores.loc[symbol]) for symbol in symbols},
            factor_count=len(expected_names),
            factors_used=expected_names,
            factor_weights=factor_weights,
            factor_coverages=factor_coverages,
            skipped_factors=dict(skipped),
            registry_metadata={
                **dict(self.registry.metadata),
                "governance_runtime": dict(runtime_status),
            },
            governance_status="ready",
            factor_mode="governed_mined_factors",
            confidence_multiplier=1.0,
            production_eligible=True,
            runtime_mode=self.runtime_mode,
            runtime_blockers=[],
            production_input_sha256=production_input_sha256,
            production_evaluation_context=evaluation_context.to_metadata(),
            production_evaluation_context_sha256=(
                evaluation_context.context_sha256
            ),
        )
        try:
            result.production_output_attestation_sha256 = (
                _production_output_attestation_sha256(result.to_metadata())
            )
        except (TypeError, ValueError):
            blocked = self._blocked_runtime_status(
                runtime_status,
                "production_runtime_output_attestation_failed",
            )
            return self._empty_score(symbols, skipped=skipped, runtime_status=blocked)
        return result

    def _compute_factor(
        self,
        factor: FactorRecord,
        frames: Mapping[str, pd.DataFrame],
    ) -> pd.Series:
        impl = str(factor.implementation or "").strip()
        if (
            self.runtime_mode == PRODUCTION_RUNTIME_MODE
            and not impl.startswith("price_volume:")
        ):
            raise ValueError(f"production implementation is not allowlisted: {impl}")
        if impl == "alpha158.FactorEngineer.cross_sectional_score":
            return self._alpha158_cross_sectional(frames)
        if impl.startswith("alpha_mining.FactorLibrary:"):
            return self._alpha_mining_factor(impl.split(":", 1)[1], frames)
        if impl.startswith("price_volume:"):
            return self._price_volume_factor(impl.split(":", 1)[1], frames)
        if impl.startswith("aquant_expression:"):
            return self._aquant_expression_factor(factor, impl.split(":", 1)[1], frames)
        if impl.startswith("builtin:"):
            if self.runtime_mode == REPORT_ONLY_SHADOW_RUNTIME_MODE:
                return self._builtin_factor(impl.split(":", 1)[1], frames)
            raise ValueError(f"production implementation is not allowlisted: {impl}")
        # Backward-compatible convention: a registry factor named like a
        # FactorLibrary method can be used without a verbose implementation path.
        if self.runtime_mode == REPORT_ONLY_SHADOW_RUNTIME_MODE:
            return self._alpha_mining_factor(factor.name, frames)
        raise ValueError(f"production implementation is not allowlisted: {impl}")

    @staticmethod
    def _alpha158_cross_sectional(frames: Mapping[str, pd.DataFrame]) -> pd.Series:
        from quant_investor.alpha158 import FactorEngineer

        engineer = FactorEngineer()
        scores = engineer.cross_sectional_score(dict(frames))
        return pd.Series(scores, dtype=float)

    def _alpha_mining_factor(self, name: str, frames: Mapping[str, pd.DataFrame]) -> pd.Series:
        from quant_investor.alpha_mining import FactorLibrary

        funcs = FactorLibrary.all_factor_funcs()
        func = funcs.get(str(name).strip())
        if func is None:
            raise ValueError(f"unknown FactorLibrary factor: {name}")
        combined = self._combined_frame(frames)
        if combined.empty:
            return pd.Series(dtype=float)
        values = func(combined)
        return self._latest_by_symbol(combined, values)

    @staticmethod
    def _price_volume_factor(
        name: str,
        frames: Mapping[str, pd.DataFrame],
        *,
        prepared_frames: Mapping[str, Any] | None = None,
        factor_cache: dict[str, Any] | None = None,
    ) -> pd.Series:
        from quant_investor.factors.price_volume import compute_price_volume_factor

        return compute_price_volume_factor(
            name,
            frames,
            prepared_frames=prepared_frames,
            factor_cache=factor_cache,
        )

    @staticmethod
    def _aquant_expression_factor(
        factor: FactorRecord,
        name: str,
        frames: Mapping[str, pd.DataFrame],
    ) -> pd.Series:
        from quant_investor.factors.aquant_expression import compute_aquant_expression_factor

        expression = str(factor.metadata.get("expression", "") or "").strip()
        metadata_dir = factor.metadata.get("metadata_dir")
        pit_series_path = factor.metadata.get("pit_series_path")
        fundamental_mart_root = factor.metadata.get("fundamental_mart_root")
        allow_legacy_fundamental_fallback = factor.metadata.get("allow_legacy_fundamental_fallback")
        return compute_aquant_expression_factor(
            str(name or factor.name),
            frames,
            expression=expression,
            metadata_dir=metadata_dir,
            pit_series_path=pit_series_path,
            fundamental_mart_root=fundamental_mart_root,
            allow_legacy_fundamental_fallback=allow_legacy_fundamental_fallback,
        )

    @staticmethod
    def _builtin_factor(name: str, frames: Mapping[str, pd.DataFrame]) -> pd.Series:
        values: dict[str, float] = {}
        for symbol, frame in frames.items():
            close = _close_series(frame)
            if close.empty:
                values[str(symbol)] = np.nan
                continue
            if name == "short_term_return":
                values[str(symbol)] = _window_return(close, 20)
            elif name == "volatility_penalty":
                returns = close.pct_change().dropna()
                values[str(symbol)] = (
                    -float(returns.tail(60).std()) if len(returns) >= 3 else np.nan
                )
            else:
                raise ValueError(f"unknown builtin factor: {name}")
        return pd.Series(values, dtype=float)

    @staticmethod
    def _combined_frame(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        for symbol, frame in frames.items():
            if frame is None or frame.empty:
                continue
            working = frame.copy()
            if "symbol" not in working.columns:
                working["symbol"] = str(symbol)
            if "date" not in working.columns:
                working["date"] = working.index
            chunks.append(working)
        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks, ignore_index=True)
        if "date" in combined.columns:
            combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
        return combined

    @staticmethod
    def _latest_by_symbol(combined: pd.DataFrame, values: pd.Series) -> pd.Series:
        working = combined[["symbol"]].copy()
        working["__factor__"] = pd.to_numeric(values.reindex(combined.index), errors="coerce")
        latest: dict[str, float] = {}
        for symbol, group in working.groupby("symbol", sort=False):
            series = group["__factor__"].replace([np.inf, -np.inf], np.nan).dropna()
            latest[str(symbol)] = float(series.iloc[-1]) if not series.empty else np.nan
        return pd.Series(latest, dtype=float)

    @staticmethod
    def _rank_normalize(values: pd.Series) -> pd.Series:
        clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = clean.dropna()
        result = pd.Series(0.0, index=clean.index, dtype=float)
        if len(valid) <= 1:
            return result
        ranks = pd.Series(stats.rankdata(valid, method="average"), index=valid.index, dtype=float)
        normalized = ((ranks - ranks.mean()) / (ranks.std(ddof=0) + 1e-9)).clip(-3.0, 3.0) / 3.0
        result.loc[normalized.index] = normalized
        return result.clip(-1.0, 1.0)


def score_with_mined_factors(
    frames: Mapping[str, pd.DataFrame],
    registry: MinedFactorRegistry | None = None,
    *,
    runtime_mode: str = PRODUCTION_RUNTIME_MODE,
    evaluation_context: ProductionEvaluationContext | None = None,
) -> RuntimeFactorScore:
    return MinedFactorScorer(
        registry=registry,
        runtime_mode=runtime_mode,
    ).score(frames, evaluation_context=evaluation_context)


def production_runtime_metadata_is_ready(
    metadata: Mapping[str, Any],
    *,
    expected_symbols: Sequence[str] | None = None,
    expected_symbol_scores: Mapping[str, Any] | None = None,
    expected_frames: Mapping[str, pd.DataFrame] | None = None,
    expected_input_digest: str | None = None,
    expected_evaluation_context: ProductionEvaluationContext | None = None,
    expected_evaluation_context_sha256: str | None = None,
) -> bool:
    """Independently revalidate serialized production claims at DAG boundaries."""

    if (
        expected_frames is None
        and expected_input_digest is None
    ) or (
        expected_evaluation_context is None
        and expected_evaluation_context_sha256 is None
    ):
        return False
    try:
        factor_count = metadata.get("factor_count")
        factors_used = list(metadata.get("factors_used", []) or [])
        factor_weights = dict(metadata.get("factor_weights", {}) or {})
        factor_coverages = dict(metadata.get("factor_coverages", {}) or {})
        runtime_blockers = list(metadata.get("runtime_blockers", []) or [])
        registry_metadata = dict(metadata.get("registry", {}) or {})
        governance = dict(registry_metadata.get("governance_runtime", {}) or {})
        expected_names = list(governance.get("production_factor_names", []) or [])
        runtime_contracts = dict(
            governance.get("factor_runtime_contracts", {}) or {}
        )
        code_hashes = dict(
            governance.get("factor_runtime_implementation_code_sha256s", {}) or {}
        )
        activation = dict(
            governance.get("quant_production_activation", {}) or {}
        )
        symbol_count = metadata.get("symbol_count")
        symbol_set_sha256 = metadata.get("symbol_set_sha256")
        symbol_scores_sha256 = metadata.get("symbol_scores_sha256")
        production_input_digest = metadata.get("production_input_sha256")
        evaluation_context_metadata = dict(
            metadata.get("production_evaluation_context", {}) or {}
        )
        evaluation_context_sha256 = metadata.get(
            "production_evaluation_context_sha256"
        )
        output_attestation = metadata.get(
            "production_output_attestation_sha256"
        )
    except (TypeError, ValueError):
        return False
    if isinstance(factor_count, bool) or not isinstance(factor_count, int):
        return False
    if factor_count <= 0 or len(factors_used) != factor_count:
        return False
    if len(factors_used) != len(set(factors_used)):
        return False
    if (
        isinstance(symbol_count, bool)
        or not isinstance(symbol_count, int)
        or symbol_count <= 0
        or not _is_sha256(symbol_set_sha256)
        or not _is_sha256(symbol_scores_sha256)
        or not _is_sha256(production_input_digest)
        or not _is_sha256(evaluation_context_sha256)
        or not _is_sha256(output_attestation)
    ):
        return False

    evaluation_context = _context_from_metadata(evaluation_context_metadata)
    context_symbols = (
        [str(symbol) for symbol in expected_symbols]
        if expected_symbols is not None
        else list(metadata.get("symbol_scores", {}) or {})
    )
    if evaluation_context is None:
        return False
    if evaluation_context.context_sha256 != evaluation_context_sha256:
        return False
    if expected_symbols is not None and validate_production_evaluation_context(
        evaluation_context,
        expected_symbols=context_symbols,
        _require_readback_seal=False,
    ):
        return False
    if expected_evaluation_context is not None and validate_production_evaluation_context(
        expected_evaluation_context,
        expected_symbols=context_symbols,
    ):
        return False
    if (
        expected_evaluation_context is not None
        and evaluation_context != expected_evaluation_context
    ):
        return False
    if expected_evaluation_context_sha256 is not None:
        if (
            not _is_sha256(expected_evaluation_context_sha256)
            or evaluation_context_sha256
            != expected_evaluation_context_sha256
        ):
            return False
    if (
        set(factor_weights) != set(factors_used)
        or set(factor_coverages) != set(factors_used)
        or expected_names != factors_used
        or set(runtime_contracts) != set(factors_used)
        or set(code_hashes) != set(factors_used)
    ):
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in factor_weights.values()
    ):
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) != 1.0
        for value in factor_coverages.values()
    ):
        return False
    try:
        if output_attestation != _production_output_attestation_sha256(metadata):
            return False
    except (TypeError, ValueError):
        return False

    if expected_symbols is not None:
        normalized_expected = [str(symbol) for symbol in expected_symbols]
        if (
            any(not symbol for symbol in normalized_expected)
            or len(normalized_expected) != len(set(normalized_expected))
            or symbol_count != len(normalized_expected)
            or symbol_set_sha256
            != production_symbol_set_sha256(normalized_expected)
        ):
            return False
    else:
        normalized_expected = None

    if expected_symbol_scores is not None:
        normalized_score_symbols = [
            str(symbol) for symbol in expected_symbol_scores
        ]
        if (
            any(not symbol for symbol in normalized_score_symbols)
            or len(normalized_score_symbols) != len(set(normalized_score_symbols))
            or symbol_count != len(normalized_score_symbols)
            or symbol_set_sha256
            != production_symbol_set_sha256(normalized_score_symbols)
            or (
                normalized_expected is not None
                and set(normalized_score_symbols) != set(normalized_expected)
            )
        ):
            return False
        try:
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
                for value in expected_symbol_scores.values()
            ):
                return False
            if symbol_scores_sha256 != _symbol_scores_sha256(
                expected_symbol_scores
            ):
                return False
        except (TypeError, ValueError):
            return False
    contracts_sha = governance.get("factor_runtime_contracts_sha256")
    registry_path_value = registry_metadata.get("path")
    if not isinstance(registry_path_value, str) or not registry_path_value.strip():
        return False
    if registry_metadata.get("strict_loader") is not True:
        return False

    # Recompute all executable identities instead of accepting well-shaped
    # nested claims.  Imports stay local to avoid the protocol/runtime cycle.
    try:
        from quant_investor.factors.governance_protocol_v2 import (
            governance_runtime_status,
        )
        from quant_investor.factors.runtime_contract import (
            implementation_code_sha256,
            production_runtime_contracts_sha256,
        )

        recomputed_contracts_sha = production_runtime_contracts_sha256(
            runtime_contracts
        )
        recomputed_code_hashes: dict[str, str] = {}
        for name in factors_used:
            contract = runtime_contracts.get(name)
            if not isinstance(contract, Mapping):
                return False
            min_cross_section = contract.get("min_cross_section")
            if (
                isinstance(min_cross_section, bool)
                or not isinstance(min_cross_section, int)
                or min_cross_section <= 0
                or symbol_count < min_cross_section
            ):
                return False
            implementation_id = contract.get("implementation_id")
            if not isinstance(implementation_id, str) or not implementation_id:
                return False
            code_sha = implementation_code_sha256(implementation_id)
            if contract.get("implementation_code_sha256") != code_sha:
                return False
            recomputed_code_hashes[name] = code_sha
        if contracts_sha != recomputed_contracts_sha:
            return False
        if code_hashes != recomputed_code_hashes:
            return False

        recomputed_input_digest: str | None = None
        if expected_frames is not None:
            frame_symbols = [str(symbol) for symbol in expected_frames]
            if (
                any(not symbol for symbol in frame_symbols)
                or len(frame_symbols) != len(set(frame_symbols))
                or symbol_count != len(frame_symbols)
                or symbol_set_sha256
                != production_symbol_set_sha256(frame_symbols)
                or (
                    normalized_expected is not None
                    and set(frame_symbols) != set(normalized_expected)
                )
            ):
                return False
            recomputed_input_digest = production_runtime_input_sha256(
                expected_frames,
                runtime_contracts,
            )
            if _validate_production_frames(
                expected_frames,
                symbols=frame_symbols,
                context=evaluation_context,
            ):
                return False
        if expected_input_digest is not None:
            if not _is_sha256(expected_input_digest):
                return False
            if (
                recomputed_input_digest is not None
                and expected_input_digest != recomputed_input_digest
            ):
                return False
            recomputed_input_digest = expected_input_digest
        if (
            recomputed_input_digest is not None
            and production_input_digest != recomputed_input_digest
        ):
            return False

        strict_registry = MinedFactorRegistry.load_production(registry_path_value)
        strict_metadata = dict(strict_registry.metadata or {})
        current_governance = governance_runtime_status(strict_registry)
        current_activation = dict(
            current_governance.get("quant_production_activation", {}) or {}
        )
    except (OSError, TypeError, ValueError, KeyError):
        return False

    if strict_metadata.get("strict_load_error") or strict_metadata.get("load_error"):
        return False
    try:
        claimed_path = Path(registry_path_value).expanduser().resolve()
        strict_path = Path(str(strict_metadata.get("path") or "")).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    if claimed_path != strict_path:
        return False
    if (
        registry_metadata.get("registry_sha256")
        != strict_metadata.get("registry_sha256")
        or registry_metadata.get("record_sha256s")
        != strict_metadata.get("record_sha256s")
        or registry_metadata.get("production_factor_runtime_contracts")
        != strict_metadata.get("production_factor_runtime_contracts")
    ):
        return False

    current_names = list(
        current_governance.get("production_factor_names", []) or []
    )
    current_contracts = dict(
        current_governance.get("factor_runtime_contracts", {}) or {}
    )
    current_code_hashes = dict(
        current_governance.get(
            "factor_runtime_implementation_code_sha256s", {}
        )
        or {}
    )
    if (
        current_governance.get("status") != "ready"
        or current_names != factors_used
        or current_contracts != runtime_contracts
        or current_governance.get("factor_runtime_contracts_sha256")
        != recomputed_contracts_sha
        or current_code_hashes != recomputed_code_hashes
        or governance.get("protocol_version")
        != current_governance.get("protocol_version")
        or governance.get("protocol_hash")
        != current_governance.get("protocol_hash")
        or governance.get("production_factor_count")
        != current_governance.get("production_factor_count")
        or governance.get("production_factor_set_sha256")
        != current_governance.get("production_factor_set_sha256")
    ):
        return False

    current_records = {
        record.name: record for record in strict_registry.selectable_factors()
    }
    try:
        expected_weights = {
            name: float(current_records[name].weight)
            * (1.0 if float(current_records[name].direction) >= 0.0 else -1.0)
            for name in factors_used
        }
        if any(
            not math.isclose(
                float(factor_weights[name]),
                expected_weights[name],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in factors_used
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False

    receipt_sha = activation.get("receipt_file_sha256")
    current_receipt_sha = current_activation.get("receipt_file_sha256")
    if (
        not isinstance(receipt_sha, str)
        or len(receipt_sha) != 64
        or any(char not in "0123456789abcdef" for char in receipt_sha)
        or receipt_sha != current_receipt_sha
        or activation.get("receipt_path") != current_activation.get("receipt_path")
    ):
        return False

    return bool(
        metadata.get("governance_status") == "ready"
        and metadata.get("factor_mode") == "governed_mined_factors"
        and metadata.get("production_eligible") is True
        and metadata.get("runtime_mode") == PRODUCTION_RUNTIME_MODE
        and metadata.get("applied_to_score") is True
        and not runtime_blockers
        and governance.get("status") == "ready"
        and governance.get("factor_mode") == "governed_mined_factors"
        and governance.get("production_eligible") is True
        and not list(governance.get("blockers", []) or [])
        and activation.get("status") == "ready"
        and not list(activation.get("blockers", []) or [])
        and current_activation.get("status") == "ready"
        and not list(current_activation.get("blockers", []) or [])
    )


def production_runtime_score_is_ready(
    score: RuntimeFactorScore,
    *,
    expected_symbols: Sequence[str],
    expected_frames: Mapping[str, pd.DataFrame] | None = None,
    expected_input_digest: str | None = None,
    expected_evaluation_context: ProductionEvaluationContext | None = None,
    expected_evaluation_context_sha256: str | None = None,
) -> bool:
    """Reject partial or internally inconsistent ready claims."""

    expected = [str(symbol) for symbol in expected_symbols]
    if (
        not expected
        or any(not symbol for symbol in expected)
        or len(expected) != len(set(expected))
        or (expected_frames is None and expected_input_digest is None)
    ):
        return False
    if set(score.symbol_scores) != set(expected):
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not -1.0 <= float(value) <= 1.0
        for value in score.symbol_scores.values()
    ):
        return False
    return production_runtime_metadata_is_ready(
        score.to_metadata(),
        expected_symbols=expected,
        expected_symbol_scores=score.symbol_scores,
        expected_frames=expected_frames,
        expected_input_digest=expected_input_digest,
        expected_evaluation_context=expected_evaluation_context,
        expected_evaluation_context_sha256=(
            expected_evaluation_context_sha256
        ),
    )


def _close_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    close_col = "close" if "close" in frame.columns else "Close" if "Close" in frame.columns else ""
    if not close_col:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[close_col], errors="coerce").dropna()


def _window_return(close: pd.Series, window: int) -> float:
    if window <= 0 or len(close) <= window:
        return 0.0
    base = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if abs(base) <= 1e-8:
        return 0.0
    return (latest / base) - 1.0


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "MinedFactorRegistry",
    "MinedFactorScorer",
    "PRODUCTION_RUNTIME_MODE",
    "PRODUCTION_EVALUATION_CONTEXT_SCHEMA_VERSION",
    "ProductionEvaluationContext",
    "REPORT_ONLY_SHADOW_RUNTIME_MODE",
    "RuntimeFactorScore",
    "production_runtime_metadata_is_ready",
    "production_evaluation_context_sha256",
    "production_runtime_input_sha256",
    "production_runtime_score_is_ready",
    "production_factor_set_sha256",
    "production_symbol_set_sha256",
    "validate_production_evaluation_context",
    "score_with_mined_factors",
]
