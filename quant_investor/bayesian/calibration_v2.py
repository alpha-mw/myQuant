"""Offline empirical calibration V2 for Bayesian prediction ledgers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.bayesian.outcome_ledger import (
    OUTCOME_STATUS_RESOLVED,
    OutcomeLedgerStore,
    OutcomeRecord,
    PredictionRecord,
)
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.versioning import CALIBRATION_V2_SCHEMA_VERSION


DEFAULT_CALIBRATION_V2_DIR = Path("data/bayesian_calibration_v2/v14")
DEFAULT_CALIBRATION_MODEL_FILENAME = "calibration_model_v2.json"
DEFAULT_CALIBRATION_REPORT_FILENAME = "calibration_report_v2.json"
TARGET_POSTERIOR_WIN_RATE = "posterior_win_rate"
GROUP_ALL_MARKETS = "ALL_MARKETS"
GROUP_ALL_HORIZONS = "ALL_HORIZONS"
GROUP_ALL_REGIMES = "ALL_REGIMES"
ALLOWED_CALIBRATION_TARGETS = frozenset(
    {
        TARGET_POSTERIOR_WIN_RATE,
        *(f"branch:{branch_name}" for branch_name in CANONICAL_BRANCH_ORDER),
    }
)


def _require_target_name(target_name: str) -> str:
    resolved = str(target_name).strip()
    if resolved not in ALLOWED_CALIBRATION_TARGETS:
        raise ValueError(f"Unsupported calibration target: {target_name!r}.")
    return resolved


def _require_schema(payload: Mapping[str, Any], *, artifact_type: str) -> str:
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != CALIBRATION_V2_SCHEMA_VERSION:
        raise ValueError(
            f"{artifact_type} schema mismatch: expected "
            f"{CALIBRATION_V2_SCHEMA_VERSION!r}, got {schema_version!r}."
        )
    return schema_version


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_json_safe(value))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _base_rate(positive_count: int, total_count: int) -> float | None:
    if total_count <= 0:
        return None
    return positive_count / total_count


def _validate_unit_interval(value: float, field_name: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return score


def _finite_non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative; got {value!r}.")
    return number


def _metric_key(
    *,
    target_name: str,
    market: str = GROUP_ALL_MARKETS,
    horizon_label: str = GROUP_ALL_HORIZONS,
    macro_regime: str = GROUP_ALL_REGIMES,
) -> tuple[str, str, str, str]:
    return (target_name, market, horizon_label, macro_regime)


def normalize_score_to_unit_interval(value: float) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"Calibration value must be finite; got {value!r}.")
    if 0.0 <= score <= 1.0:
        return score
    if -1.0 <= score < 0.0:
        return (score + 1.0) / 2.0
    if score > 1.0:
        return 1.0
    return 0.0


@dataclass
class CalibrationTrainingExample:
    prediction_id: str = ""
    run_id: str = ""
    symbol: str = ""
    market: str = ""
    horizon_days: int = 0
    horizon_label: str = ""
    macro_regime: str = ""
    target_name: str = TARGET_POSTERIOR_WIN_RATE
    raw_value: float = 0.0
    normalized_value: float = 0.0
    realized_label: int = 0
    realized_return: float | None = None
    benchmark_return: float | None = None
    excess_return: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_name = _require_target_name(self.target_name)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationTrainingExample":
        data = dict(payload)
        return cls(
            prediction_id=str(data.get("prediction_id", "")),
            run_id=str(data.get("run_id", "")),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            horizon_days=int(data.get("horizon_days", 0) or 0),
            horizon_label=str(data.get("horizon_label", "")),
            macro_regime=str(data.get("macro_regime", "")),
            target_name=str(data.get("target_name", TARGET_POSTERIOR_WIN_RATE)),
            raw_value=float(data.get("raw_value", 0.0) or 0.0),
            normalized_value=float(data.get("normalized_value", 0.0) or 0.0),
            realized_label=int(data.get("realized_label", 0) or 0),
            realized_return=_float_or_none(data.get("realized_return")),
            benchmark_return=_float_or_none(data.get("benchmark_return")),
            excess_return=_float_or_none(data.get("excess_return")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class CalibrationBucket:
    bucket_index: int = 0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    center: float = 0.0
    total_count: int = 0
    positive_count: int = 0
    raw_mean: float = 0.0
    empirical_rate: float | None = None
    prior_alpha: float = 0.0
    prior_beta: float = 0.0
    calibrated_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.bucket_index < 0:
            raise ValueError("Calibration bucket_index must be non-negative.")
        for field_name in ("lower_bound", "upper_bound", "center", "raw_mean"):
            setattr(
                self,
                field_name,
                _validate_unit_interval(getattr(self, field_name), field_name),
            )
        if self.lower_bound > self.upper_bound:
            raise ValueError("Calibration bucket lower_bound exceeds upper_bound.")
        if self.total_count < 0 or self.positive_count < 0:
            raise ValueError("Calibration bucket counts must be non-negative.")
        if self.positive_count > self.total_count:
            raise ValueError("Calibration positive_count exceeds total_count.")
        if self.empirical_rate is not None:
            self.empirical_rate = _validate_unit_interval(
                self.empirical_rate,
                "empirical_rate",
            )
        self.prior_alpha = _finite_non_negative(self.prior_alpha, "prior_alpha")
        self.prior_beta = _finite_non_negative(self.prior_beta, "prior_beta")
        self.calibrated_probability = _validate_unit_interval(
            self.calibrated_probability,
            "calibrated_probability",
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationBucket":
        data = dict(payload)
        return cls(
            bucket_index=int(data.get("bucket_index", 0) or 0),
            lower_bound=float(data.get("lower_bound", 0.0) or 0.0),
            upper_bound=float(data.get("upper_bound", 0.0) or 0.0),
            center=float(data.get("center", 0.0) or 0.0),
            total_count=int(data.get("total_count", 0) or 0),
            positive_count=int(data.get("positive_count", 0) or 0),
            raw_mean=float(data.get("raw_mean", 0.0) or 0.0),
            empirical_rate=_float_or_none(data.get("empirical_rate")),
            prior_alpha=float(data.get("prior_alpha", 0.0) or 0.0),
            prior_beta=float(data.get("prior_beta", 0.0) or 0.0),
            calibrated_probability=float(data.get("calibrated_probability", 0.0) or 0.0),
        )


@dataclass
class CalibrationCurveKey:
    target_name: str = TARGET_POSTERIOR_WIN_RATE
    market: str = GROUP_ALL_MARKETS
    horizon_label: str = GROUP_ALL_HORIZONS
    macro_regime: str = GROUP_ALL_REGIMES

    def __post_init__(self) -> None:
        self.target_name = _require_target_name(self.target_name)

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.target_name, self.market, self.horizon_label, self.macro_regime)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationCurveKey":
        data = dict(payload)
        return cls(
            target_name=str(data.get("target_name", TARGET_POSTERIOR_WIN_RATE)),
            market=str(data.get("market", GROUP_ALL_MARKETS)),
            horizon_label=str(data.get("horizon_label", GROUP_ALL_HORIZONS)),
            macro_regime=str(data.get("macro_regime", GROUP_ALL_REGIMES)),
        )


@dataclass
class CalibrationCurve:
    schema_version: str = CALIBRATION_V2_SCHEMA_VERSION
    key: CalibrationCurveKey = field(default_factory=CalibrationCurveKey)
    bucket_count: int = 10
    prior_strength: float = 20.0
    total_examples: int = 0
    positive_examples: int = 0
    base_rate: float | None = None
    buckets: list[CalibrationBucket] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_V2_SCHEMA_VERSION:
            raise ValueError(
                "Calibration curve schema mismatch: expected "
                f"{CALIBRATION_V2_SCHEMA_VERSION!r}, got {self.schema_version!r}."
            )
        if not isinstance(self.key, CalibrationCurveKey):
            raise ValueError("Calibration curve key must be a CalibrationCurveKey object.")
        self.key.__post_init__()
        if self.bucket_count <= 0:
            raise ValueError("Calibration curve bucket_count must be positive.")
        self.prior_strength = _finite_non_negative(
            self.prior_strength,
            "Calibration curve prior_strength",
        )
        if self.total_examples < 0 or self.positive_examples < 0:
            raise ValueError("Calibration curve totals must be non-negative.")
        if self.positive_examples > self.total_examples:
            raise ValueError(
                "Calibration curve positive_examples exceeds total_examples."
            )
        if self.base_rate is not None:
            self.base_rate = _validate_unit_interval(
                self.base_rate,
                "Calibration curve base_rate",
            )
        if len(self.buckets) != self.bucket_count:
            raise ValueError(
                "Calibration curve bucket length mismatch: "
                f"expected {self.bucket_count}, got {len(self.buckets)}."
            )
        for bucket in self.buckets:
            if not isinstance(bucket, CalibrationBucket):
                raise ValueError("Calibration curve buckets must be CalibrationBucket objects.")
            bucket.__post_init__()
        for expected_index, bucket in enumerate(self.buckets):
            if bucket.bucket_index != expected_index:
                raise ValueError(
                    "Calibration curve bucket indices must be contiguous and ordered."
                )
            expected_lower = expected_index / self.bucket_count
            expected_upper = (expected_index + 1) / self.bucket_count
            expected_center = (expected_lower + expected_upper) / 2.0
            if not math.isclose(bucket.lower_bound, expected_lower, abs_tol=1e-12):
                raise ValueError("Calibration curve bucket lower_bound is inconsistent.")
            if not math.isclose(bucket.upper_bound, expected_upper, abs_tol=1e-12):
                raise ValueError("Calibration curve bucket upper_bound is inconsistent.")
            if not math.isclose(bucket.center, expected_center, abs_tol=1e-12):
                raise ValueError("Calibration curve bucket center is inconsistent.")
            if not bucket.lower_bound <= bucket.center <= bucket.upper_bound:
                raise ValueError("Calibration curve bucket center is outside its bounds.")
        if sum(bucket.total_count for bucket in self.buckets) != self.total_examples:
            raise ValueError("Calibration curve total_examples does not match buckets.")
        if sum(bucket.positive_count for bucket in self.buckets) != self.positive_examples:
            raise ValueError("Calibration curve positive_examples does not match buckets.")
        expected_base_rate = _base_rate(self.positive_examples, self.total_examples)
        if expected_base_rate is None:
            if self.base_rate is not None:
                raise ValueError("Calibration curve base_rate must be null without examples.")
        elif self.base_rate is None or not math.isclose(
            self.base_rate,
            expected_base_rate,
            abs_tol=1e-12,
        ):
            raise ValueError("Calibration curve base_rate does not match totals.")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        payload = asdict(self)
        return dict(_json_safe(payload))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationCurve":
        data = dict(payload)
        schema_version = _require_schema(data, artifact_type="Calibration curve")
        raw_buckets = data.get("buckets", [])
        if not isinstance(raw_buckets, list) or any(
            not isinstance(bucket, Mapping) for bucket in raw_buckets
        ):
            raise ValueError("Calibration curve buckets must be objects.")
        return cls(
            schema_version=schema_version,
            key=CalibrationCurveKey.from_dict(dict(data.get("key", {}) or {})),
            bucket_count=int(10 if data.get("bucket_count") is None else data["bucket_count"]),
            prior_strength=float(20.0 if data.get("prior_strength") is None else data["prior_strength"]),
            total_examples=int(0 if data.get("total_examples") is None else data["total_examples"]),
            positive_examples=int(0 if data.get("positive_examples") is None else data["positive_examples"]),
            base_rate=_float_or_none(data.get("base_rate")),
            buckets=[
                CalibrationBucket.from_dict(bucket)
                for bucket in raw_buckets
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class CalibrationModelV2:
    schema_version: str = CALIBRATION_V2_SCHEMA_VERSION
    model_id: str = ""
    trained_at: str = ""
    bucket_count: int = 10
    prior_strength: float = 20.0
    min_examples_per_curve: int = 30
    curves: list[CalibrationCurve] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema_version != CALIBRATION_V2_SCHEMA_VERSION:
            raise ValueError(
                "Calibration model schema mismatch: expected "
                f"{CALIBRATION_V2_SCHEMA_VERSION!r}, got {self.schema_version!r}."
            )
        if self.bucket_count <= 0:
            raise ValueError("Calibration model bucket_count must be positive.")
        self.prior_strength = _finite_non_negative(
            self.prior_strength,
            "Calibration model prior_strength",
        )
        if self.min_examples_per_curve <= 0:
            raise ValueError("Calibration model min_examples_per_curve must be positive.")
        for curve in self.curves:
            if not isinstance(curve, CalibrationCurve):
                raise ValueError("Calibration model curves must be CalibrationCurve objects.")
            curve.__post_init__()
            if curve.bucket_count != self.bucket_count:
                raise ValueError("Calibration model and curve bucket_count mismatch.")
            if not math.isclose(
                curve.prior_strength,
                self.prior_strength,
                abs_tol=1e-12,
            ):
                raise ValueError("Calibration model and curve prior_strength mismatch.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationModelV2":
        data = dict(payload)
        schema_version = _require_schema(data, artifact_type="Calibration model")
        raw_curves = list(data.get("curves", []) or [])
        if any(not isinstance(curve, Mapping) for curve in raw_curves):
            raise ValueError("Calibration model curves must be objects.")
        return cls(
            schema_version=schema_version,
            model_id=str(data.get("model_id", "")),
            trained_at=str(data.get("trained_at", "")),
            bucket_count=int(10 if data.get("bucket_count") is None else data["bucket_count"]),
            prior_strength=float(20.0 if data.get("prior_strength") is None else data["prior_strength"]),
            min_examples_per_curve=int(
                30
                if data.get("min_examples_per_curve") is None
                else data["min_examples_per_curve"]
            ),
            curves=[
                CalibrationCurve.from_dict(curve)
                for curve in raw_curves
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def get_curve(self, key: CalibrationCurveKey) -> CalibrationCurve | None:
        target = key.as_tuple()
        for curve in self.curves:
            if curve.key.as_tuple() == target:
                return curve
        return None

    def _fallback_keys(
        self,
        target_name: str,
        *,
        market: str | None = None,
        horizon_label: str | None = None,
        macro_regime: str | None = None,
    ) -> list[CalibrationCurveKey]:
        target_name = _require_target_name(target_name)
        resolved_market = market or GROUP_ALL_MARKETS
        resolved_horizon = horizon_label or GROUP_ALL_HORIZONS
        resolved_regime = macro_regime or GROUP_ALL_REGIMES
        candidate_keys = [
            CalibrationCurveKey(target_name, resolved_market, resolved_horizon, resolved_regime),
            CalibrationCurveKey(target_name, resolved_market, resolved_horizon, GROUP_ALL_REGIMES),
            CalibrationCurveKey(target_name, resolved_market, GROUP_ALL_HORIZONS, GROUP_ALL_REGIMES),
            CalibrationCurveKey(target_name, GROUP_ALL_MARKETS, resolved_horizon, GROUP_ALL_REGIMES),
            CalibrationCurveKey(target_name, GROUP_ALL_MARKETS, GROUP_ALL_HORIZONS, GROUP_ALL_REGIMES),
        ]
        keys: list[CalibrationCurveKey] = []
        seen: set[tuple[str, str, str, str]] = set()
        for key in candidate_keys:
            if key.as_tuple() in seen:
                continue
            seen.add(key.as_tuple())
            keys.append(key)
        return keys

    def select_curve(
        self,
        target_name: str,
        *,
        market: str | None = None,
        horizon_label: str | None = None,
        macro_regime: str | None = None,
    ) -> CalibrationCurve | None:
        for key in self._fallback_keys(
            target_name,
            market=market,
            horizon_label=horizon_label,
            macro_regime=macro_regime,
        ):
            curve = self.get_curve(key)
            if curve is not None:
                return curve
        return None

    def calibrate(
        self,
        target_name: str,
        raw_value: float,
        *,
        market: str | None = None,
        horizon_label: str | None = None,
        macro_regime: str | None = None,
    ) -> float:
        normalized = normalize_score_to_unit_interval(raw_value)
        for key in self._fallback_keys(
            target_name,
            market=market,
            horizon_label=horizon_label,
            macro_regime=macro_regime,
        ):
            curve = self.get_curve(key)
            if curve is None:
                continue
            bucket_index = bucket_index_for_value(normalized, curve.bucket_count)
            if 0 <= bucket_index < len(curve.buckets):
                probability = curve.buckets[bucket_index].calibrated_probability
                return max(0.0, min(1.0, float(probability)))
        return normalized


@dataclass
class CalibrationMetricSummary:
    target_name: str = TARGET_POSTERIOR_WIN_RATE
    market: str = GROUP_ALL_MARKETS
    horizon_label: str = GROUP_ALL_HORIZONS
    macro_regime: str = GROUP_ALL_REGIMES
    example_count: int = 0
    positive_count: int = 0
    base_rate: float | None = None
    raw_brier_score: float | None = None
    calibrated_brier_score: float | None = None
    raw_log_loss: float | None = None
    calibrated_log_loss: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_name = _require_target_name(self.target_name)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationMetricSummary":
        data = dict(payload)
        return cls(
            target_name=str(data.get("target_name", TARGET_POSTERIOR_WIN_RATE)),
            market=str(data.get("market", GROUP_ALL_MARKETS)),
            horizon_label=str(data.get("horizon_label", GROUP_ALL_HORIZONS)),
            macro_regime=str(data.get("macro_regime", GROUP_ALL_REGIMES)),
            example_count=int(data.get("example_count", 0) or 0),
            positive_count=int(data.get("positive_count", 0) or 0),
            base_rate=_float_or_none(data.get("base_rate")),
            raw_brier_score=_float_or_none(data.get("raw_brier_score")),
            calibrated_brier_score=_float_or_none(data.get("calibrated_brier_score")),
            raw_log_loss=_float_or_none(data.get("raw_log_loss")),
            calibrated_log_loss=_float_or_none(data.get("calibrated_log_loss")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class CalibrationReport:
    schema_version: str = CALIBRATION_V2_SCHEMA_VERSION
    model_id: str = ""
    generated_at: str = ""
    total_examples: int = 0
    metric_summaries: list[CalibrationMetricSummary] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationReport":
        data = dict(payload)
        schema_version = _require_schema(data, artifact_type="Calibration report")
        return cls(
            schema_version=schema_version,
            model_id=str(data.get("model_id", "")),
            generated_at=str(data.get("generated_at", "")),
            total_examples=int(data.get("total_examples", 0) or 0),
            metric_summaries=[
                CalibrationMetricSummary.from_dict(summary)
                for summary in list(data.get("metric_summaries", []) or [])
                if isinstance(summary, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


def build_training_examples(
    predictions: Sequence[PredictionRecord],
    outcomes: Sequence[OutcomeRecord],
    *,
    include_posterior: bool = True,
    include_branches: bool = True,
    min_abs_return: float | None = None,
) -> list[CalibrationTrainingExample]:
    for prediction in predictions:
        PredictionRecord.from_dict(prediction.to_dict())
    for outcome in outcomes:
        OutcomeRecord.from_dict(outcome.to_dict())
    resolved_outcomes: dict[str, OutcomeRecord] = {
        outcome.prediction_id: outcome
        for outcome in outcomes
        if outcome.status == OUTCOME_STATUS_RESOLVED
    }
    examples: list[CalibrationTrainingExample] = []
    for prediction in predictions:
        outcome = resolved_outcomes.get(prediction.prediction_id)
        if outcome is None:
            continue
        selected_return = outcome.excess_return
        if selected_return is None:
            selected_return = outcome.realized_return
        if selected_return is None:
            continue
        selected_return = float(selected_return)
        if min_abs_return is not None and abs(selected_return) < min_abs_return:
            continue
        realized_label = 1 if selected_return > 0.0 else 0

        def _example(target_name: str, raw_value: float) -> CalibrationTrainingExample:
            return CalibrationTrainingExample(
                prediction_id=prediction.prediction_id,
                run_id=prediction.run_id,
                symbol=prediction.symbol,
                market=prediction.market,
                horizon_days=prediction.horizon_days,
                horizon_label=prediction.horizon_label,
                macro_regime=prediction.macro_regime,
                target_name=target_name,
                raw_value=float(raw_value),
                normalized_value=normalize_score_to_unit_interval(float(raw_value)),
                realized_label=realized_label,
                realized_return=outcome.realized_return,
                benchmark_return=outcome.benchmark_return,
                excess_return=outcome.excess_return,
                metadata={"outcome_id": outcome.outcome_id},
            )

        if include_posterior:
            examples.append(_example(TARGET_POSTERIOR_WIN_RATE, prediction.posterior_win_rate))
        if include_branches:
            for branch_name in CANONICAL_BRANCH_ORDER:
                raw_value = float((prediction.branch_scores or {}).get(branch_name, 0.0) or 0.0)
                examples.append(_example(f"branch:{branch_name}", raw_value))
    return examples


def bucket_index_for_value(value: float, bucket_count: int) -> int:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    normalized = normalize_score_to_unit_interval(value)
    if normalized >= 1.0:
        return bucket_count - 1
    return max(0, min(bucket_count - 1, int(normalized * bucket_count)))


def build_calibration_curve(
    examples: Sequence[CalibrationTrainingExample],
    key: CalibrationCurveKey,
    *,
    bucket_count: int = 10,
    prior_strength: float = 20.0,
) -> CalibrationCurve:
    _require_target_name(key.target_name)
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    grouped: dict[int, list[CalibrationTrainingExample]] = {index: [] for index in range(bucket_count)}
    for example in examples:
        normalized = _validate_unit_interval(example.normalized_value, "normalized_value")
        grouped[bucket_index_for_value(normalized, bucket_count)].append(example)

    total_examples = len(examples)
    positive_examples = sum(int(example.realized_label) for example in examples)
    buckets: list[CalibrationBucket] = []
    for index in range(bucket_count):
        lower_bound = index / bucket_count
        upper_bound = (index + 1) / bucket_count
        center = (lower_bound + upper_bound) / 2.0
        bucket_examples = grouped[index]
        total_count = len(bucket_examples)
        positive_count = sum(int(example.realized_label) for example in bucket_examples)
        raw_mean = (
            sum(example.normalized_value for example in bucket_examples) / total_count
            if total_count > 0
            else center
        )
        empirical_rate = _base_rate(positive_count, total_count)
        prior_alpha = center * prior_strength
        prior_beta = (1.0 - center) * prior_strength
        calibrated_probability = (
            positive_count + prior_alpha
        ) / max(total_count + prior_alpha + prior_beta, 1e-12)
        buckets.append(
            CalibrationBucket(
                bucket_index=index,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                center=center,
                total_count=total_count,
                positive_count=positive_count,
                raw_mean=raw_mean,
                empirical_rate=empirical_rate,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
                calibrated_probability=max(0.0, min(1.0, calibrated_probability)),
            )
        )

    return CalibrationCurve(
        schema_version=CALIBRATION_V2_SCHEMA_VERSION,
        key=key,
        bucket_count=bucket_count,
        prior_strength=prior_strength,
        total_examples=total_examples,
        positive_examples=positive_examples,
        base_rate=_base_rate(positive_examples, total_examples),
        buckets=buckets,
        metadata={},
    )


def _target_order(examples: Sequence[CalibrationTrainingExample]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for example in examples:
        if example.target_name not in seen:
            order.append(example.target_name)
            seen.add(example.target_name)
    return order


def _group_examples(
    examples: Sequence[CalibrationTrainingExample],
    key: CalibrationCurveKey,
) -> list[CalibrationTrainingExample]:
    return [
        example
        for example in examples
        if example.target_name == key.target_name
        and (key.market == GROUP_ALL_MARKETS or example.market == key.market)
        and (key.horizon_label == GROUP_ALL_HORIZONS or example.horizon_label == key.horizon_label)
        and (key.macro_regime == GROUP_ALL_REGIMES or example.macro_regime == key.macro_regime)
    ]


def train_calibration_model(
    examples: Sequence[CalibrationTrainingExample],
    *,
    bucket_count: int = 10,
    prior_strength: float = 20.0,
    min_examples_per_curve: int = 30,
    trained_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationModelV2:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    if min_examples_per_curve <= 0:
        raise ValueError("min_examples_per_curve must be positive.")
    resolved_trained_at = trained_at or _now_iso()
    for example in examples:
        _require_target_name(example.target_name)
    curves: list[CalibrationCurve] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for target_name in _target_order(examples):
        target_examples = [example for example in examples if example.target_name == target_name]
        markets = sorted({example.market for example in target_examples})
        horizons = sorted({example.horizon_label for example in target_examples})
        regimes = sorted({example.macro_regime for example in target_examples})
        candidate_keys: list[CalibrationCurveKey] = []
        for market in markets:
            for horizon in horizons:
                for regime in regimes:
                    candidate_keys.append(CalibrationCurveKey(target_name, market, horizon, regime))
                candidate_keys.append(CalibrationCurveKey(target_name, market, horizon, GROUP_ALL_REGIMES))
            candidate_keys.append(CalibrationCurveKey(target_name, market, GROUP_ALL_HORIZONS, GROUP_ALL_REGIMES))
        for horizon in horizons:
            candidate_keys.append(CalibrationCurveKey(target_name, GROUP_ALL_MARKETS, horizon, GROUP_ALL_REGIMES))
        candidate_keys.append(
            CalibrationCurveKey(target_name, GROUP_ALL_MARKETS, GROUP_ALL_HORIZONS, GROUP_ALL_REGIMES)
        )

        for key in candidate_keys:
            key_tuple = key.as_tuple()
            if key_tuple in seen_keys:
                continue
            group = _group_examples(target_examples, key)
            is_global = key_tuple == _metric_key(target_name=target_name)
            if len(group) >= min_examples_per_curve or (is_global and len(group) >= 1):
                seen_keys.add(key_tuple)
                curves.append(
                    build_calibration_curve(
                        group,
                        key,
                        bucket_count=bucket_count,
                        prior_strength=prior_strength,
                    )
                )

    model_id = (
        f"calibration-v2-{resolved_trained_at.replace(':', '').replace('-', '').replace('Z', 'z')}"
        f"-n{len(examples)}-c{len(curves)}"
    )
    return CalibrationModelV2(
        schema_version=CALIBRATION_V2_SCHEMA_VERSION,
        model_id=model_id,
        trained_at=resolved_trained_at,
        bucket_count=bucket_count,
        prior_strength=prior_strength,
        min_examples_per_curve=min_examples_per_curve,
        curves=curves,
        metadata=_coerce_metadata(metadata),
    )


def brier_score(predictions: Sequence[float], labels: Sequence[int]) -> float | None:
    if not predictions:
        return None
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length.")
    total = 0.0
    for prediction, label in zip(predictions, labels, strict=True):
        probability = normalize_score_to_unit_interval(prediction)
        total += (probability - int(label)) ** 2
    return total / len(predictions)


def log_loss(predictions: Sequence[float], labels: Sequence[int], eps: float = 1e-12) -> float | None:
    if not predictions:
        return None
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length.")
    total = 0.0
    for prediction, label in zip(predictions, labels, strict=True):
        probability = normalize_score_to_unit_interval(prediction)
        probability = max(eps, min(1.0 - eps, probability))
        y = int(label)
        total += -(y * math.log(probability) + (1 - y) * math.log(1.0 - probability))
    return total / len(predictions)


def build_calibration_report(
    model: CalibrationModelV2,
    examples: Sequence[CalibrationTrainingExample],
    *,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationReport:
    if not examples:
        return CalibrationReport(
            schema_version=CALIBRATION_V2_SCHEMA_VERSION,
            model_id=model.model_id,
            generated_at=generated_at or _now_iso(),
            total_examples=0,
            metric_summaries=[],
            metadata=_coerce_metadata(metadata),
        )
    summaries: list[CalibrationMetricSummary] = []
    for target_name in _target_order(examples):
        group = [example for example in examples if example.target_name == target_name]
        labels = [example.realized_label for example in group]
        raw_predictions = [example.normalized_value for example in group]
        calibrated_predictions = [
            model.calibrate(
                example.target_name,
                example.raw_value,
                market=example.market,
                horizon_label=example.horizon_label,
                macro_regime=example.macro_regime,
            )
            for example in group
        ]
        positive_count = sum(labels)
        summaries.append(
            CalibrationMetricSummary(
                target_name=target_name,
                market=GROUP_ALL_MARKETS,
                horizon_label=GROUP_ALL_HORIZONS,
                macro_regime=GROUP_ALL_REGIMES,
                example_count=len(group),
                positive_count=positive_count,
                base_rate=_base_rate(positive_count, len(group)),
                raw_brier_score=brier_score(raw_predictions, labels),
                calibrated_brier_score=brier_score(calibrated_predictions, labels),
                raw_log_loss=log_loss(raw_predictions, labels),
                calibrated_log_loss=log_loss(calibrated_predictions, labels),
                metadata={},
            )
        )
    return CalibrationReport(
        schema_version=CALIBRATION_V2_SCHEMA_VERSION,
        model_id=model.model_id,
        generated_at=generated_at or _now_iso(),
        total_examples=len(examples),
        metric_summaries=summaries,
        metadata=_coerce_metadata(metadata),
    )


class CalibrationV2Store:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_CALIBRATION_V2_DIR
        self.model_path = self.root_dir / DEFAULT_CALIBRATION_MODEL_FILENAME
        self.report_path = self.root_dir / DEFAULT_CALIBRATION_REPORT_FILENAME

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(_json_safe(payload)), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}.")
        return payload

    def save_model(self, model: CalibrationModelV2) -> Path:
        payload = model.to_dict()
        validated = CalibrationModelV2.from_dict(payload)
        return self._write_json(self.model_path, validated.to_dict())

    def load_model(self) -> CalibrationModelV2:
        return CalibrationModelV2.from_dict(self._read_json(self.model_path))

    def save_report(self, report: CalibrationReport) -> Path:
        _require_schema(report.to_dict(), artifact_type="Calibration report")
        return self._write_json(self.report_path, report.to_dict())

    def load_report(self) -> CalibrationReport:
        return CalibrationReport.from_dict(self._read_json(self.report_path))

    def train_from_ledger(
        self,
        ledger_store: OutcomeLedgerStore,
        *,
        bucket_count: int = 10,
        prior_strength: float = 20.0,
        min_examples_per_curve: int = 30,
        include_posterior: bool = True,
        include_branches: bool = True,
        min_abs_return: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[CalibrationModelV2, CalibrationReport]:
        examples = build_training_examples(
            ledger_store.read_predictions(),
            ledger_store.read_outcomes(),
            include_posterior=include_posterior,
            include_branches=include_branches,
            min_abs_return=min_abs_return,
        )
        model = train_calibration_model(
            examples,
            bucket_count=bucket_count,
            prior_strength=prior_strength,
            min_examples_per_curve=min_examples_per_curve,
            metadata=metadata,
        )
        report = build_calibration_report(model, examples, metadata=metadata)
        self.save_model(model)
        self.save_report(report)
        return model, report


__all__ = [
    "DEFAULT_CALIBRATION_V2_DIR",
    "DEFAULT_CALIBRATION_MODEL_FILENAME",
    "DEFAULT_CALIBRATION_REPORT_FILENAME",
    "TARGET_POSTERIOR_WIN_RATE",
    "GROUP_ALL_MARKETS",
    "GROUP_ALL_HORIZONS",
    "GROUP_ALL_REGIMES",
    "CalibrationTrainingExample",
    "CalibrationBucket",
    "CalibrationCurveKey",
    "CalibrationCurve",
    "CalibrationModelV2",
    "CalibrationMetricSummary",
    "CalibrationReport",
    "CalibrationV2Store",
    "normalize_score_to_unit_interval",
    "build_training_examples",
    "bucket_index_for_value",
    "build_calibration_curve",
    "train_calibration_model",
    "brier_score",
    "log_loss",
    "build_calibration_report",
]
