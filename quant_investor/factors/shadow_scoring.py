"""Read-only production factor shadow scoring comparison helpers.

This module compares local production factor matrix signals against already
computed official candidate rankings. It does not fetch data, call providers,
alter candidates, or connect factor scores to stock selection, posterior
scoring, RiskGuard, PortfolioConstructor, orders, or execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.matrix import FactorMatrix
from quant_investor.factors.schema import (
    FACTOR_STATUS_PRODUCTION,
    FactorDefinition,
    FactorLibraryEntry,
    ProductionFactorLibrary,
)
from quant_investor.versioning import (
    FACTOR_SHADOW_COMPARISON_SCHEMA_VERSION,
    FACTOR_SHADOW_SCORING_SCHEMA_VERSION,
)


SHADOW_SCORE_STATUS_OK = "ok"
SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX = "missing_factor_matrix"
SHADOW_SCORE_STATUS_MISSING_SYMBOL = "missing_symbol"
SHADOW_SCORE_STATUS_MISSING_DATE = "missing_date"
SHADOW_SCORE_STATUS_NON_PRODUCTION_FACTOR = "non_production_factor"
SHADOW_SCORE_STATUS_LIBRARY_MISSING = "library_missing"
SHADOW_SCORE_STATUS_AUDIT_BLOCKED = "audit_blocked"
SHADOW_SCORE_STATUS_INSUFFICIENT_DATA = "insufficient_data"

SHADOW_COMPARISON_STATUS_OK = "ok"
SHADOW_COMPARISON_STATUS_WARN = "warn"
SHADOW_COMPARISON_STATUS_FAIL = "fail"

DEFAULT_FACTOR_SHADOW_SCORING_DIR = Path("data/factor_library/shadow_scoring")
DEFAULT_SHADOW_FACTOR_SCORES_FILENAME = "shadow_factor_scores.jsonl"
DEFAULT_SHADOW_CANDIDATE_SCORES_FILENAME = "shadow_candidate_scores.jsonl"
DEFAULT_SHADOW_COMPARISON_REPORTS_FILENAME = "shadow_comparison_reports.jsonl"
DEFAULT_SHADOW_COMPARISON_MARKDOWN_FILENAME = "shadow_comparison_report.md"
DEFAULT_SHADOW_COMPARISON_DASHBOARD_FILENAME = "shadow_scoring_dashboard.json"

SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE = (
    "This shadow scoring comparison is read-only and does not alter official "
    "scores, stock selection, posterior, RiskGuard, PortfolioConstructor, "
    "target weights, orders, providers, LLMs, or execution."
)

SUPPORTED_SHADOW_SCORE_STATUSES = {
    SHADOW_SCORE_STATUS_OK,
    SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX,
    SHADOW_SCORE_STATUS_MISSING_SYMBOL,
    SHADOW_SCORE_STATUS_MISSING_DATE,
    SHADOW_SCORE_STATUS_NON_PRODUCTION_FACTOR,
    SHADOW_SCORE_STATUS_LIBRARY_MISSING,
    SHADOW_SCORE_STATUS_AUDIT_BLOCKED,
    SHADOW_SCORE_STATUS_INSUFFICIENT_DATA,
}
SUPPORTED_SHADOW_COMPARISON_STATUSES = {
    SHADOW_COMPARISON_STATUS_OK,
    SHADOW_COMPARISON_STATUS_WARN,
    SHADOW_COMPARISON_STATUS_FAIL,
}
SUPPORTED_FACTOR_WEIGHT_POLICIES = {"equal_weight"}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
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


def _coerce_json_dict(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, label))


def _coerce_json_list(values: Sequence[Mapping[str, Any]] | None, label: str) -> list[dict[str, Any]]:
    if values is None:
        return []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}[{index}] must be a JSON object.")
        rows.append(dict(_ensure_json_serializable(dict(item), f"{label}[{index}]")))
    return rows


def _non_empty_str(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric value; got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field_name)


def _unit_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _unit_float(value, field_name)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive integer; got {value!r}.")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive; got {value!r}.")
    return number


def _positive_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be non-negative integer; got {value!r}.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool; got {value!r}.")
    return value


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


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


def _validate_supported(value: str, field_name: str, supported: set[str]) -> None:
    if value not in supported:
        raise ValueError(f"{field_name} must be one of {sorted(supported)}; got {value!r}.")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _to_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_present(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _blocked_factor_ids(audit_report: Any | None) -> set[str]:
    if audit_report is None:
        return set()
    raw = None
    if isinstance(audit_report, Mapping):
        raw = audit_report.get("blocked_factor_ids")
    else:
        raw = getattr(audit_report, "blocked_factor_ids", None)
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _entry_key(entry: FactorLibraryEntry) -> tuple[str, str]:
    return (entry.factor_id, entry.factor_version)


@dataclass
class ShadowScoringConfig:
    schema_version: str = FACTOR_SHADOW_SCORING_SCHEMA_VERSION
    config_id: str = ""
    as_of: str = ""
    max_rank_delta_warning: int = 20
    top_n: int = 30
    min_factor_coverage_ratio: float = 0.50
    include_blocked_factors: bool = False
    normalize_factor_scores: bool = True
    factor_weight_policy: str = "equal_weight"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_SCORING_SCHEMA_VERSION)
        self.config_id = str(self.config_id)
        self.as_of = _non_empty_str(self.as_of, "as_of")
        self.max_rank_delta_warning = _non_negative_int(
            self.max_rank_delta_warning,
            "max_rank_delta_warning",
        )
        self.top_n = _positive_int(self.top_n, "top_n")
        self.min_factor_coverage_ratio = _unit_float(
            self.min_factor_coverage_ratio,
            "min_factor_coverage_ratio",
        )
        self.include_blocked_factors = _require_bool(
            self.include_blocked_factors,
            "include_blocked_factors",
        )
        self.normalize_factor_scores = _require_bool(
            self.normalize_factor_scores,
            "normalize_factor_scores",
        )
        self.factor_weight_policy = _non_empty_str(self.factor_weight_policy, "factor_weight_policy")
        _validate_supported(
            self.factor_weight_policy,
            "factor_weight_policy",
            SUPPORTED_FACTOR_WEIGHT_POLICIES,
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShadowScoringConfig":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_SCORING_SCHEMA_VERSION)),
            config_id=str(data.get("config_id", "")),
            as_of=str(data.get("as_of", "")),
            max_rank_delta_warning=int(data.get("max_rank_delta_warning", 20)),
            top_n=int(data.get("top_n", 30)),
            min_factor_coverage_ratio=float(data.get("min_factor_coverage_ratio", 0.50)),
            include_blocked_factors=data.get("include_blocked_factors", False),
            normalize_factor_scores=data.get("normalize_factor_scores", True),
            factor_weight_policy=str(data.get("factor_weight_policy", "equal_weight")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ShadowFactorScore:
    schema_version: str = FACTOR_SHADOW_SCORING_SCHEMA_VERSION
    factor_id: str = ""
    factor_version: str = ""
    symbol: str = ""
    as_of: str = ""
    raw_value: float | None = None
    normalized_score: float | None = None
    rank: int | None = None
    coverage_status: str = SHADOW_SCORE_STATUS_INSUFFICIENT_DATA
    warning_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_SCORING_SCHEMA_VERSION)
        self.factor_id = _non_empty_str(self.factor_id, "factor_id")
        self.factor_version = _non_empty_str(self.factor_version, "factor_version")
        self.symbol = _non_empty_str(self.symbol, "symbol")
        self.as_of = _non_empty_str(self.as_of, "as_of")
        self.raw_value = _optional_finite_float(self.raw_value, "raw_value")
        self.normalized_score = _unit_float_or_none(self.normalized_score, "normalized_score")
        self.rank = _positive_int_or_none(self.rank, "rank")
        self.coverage_status = _non_empty_str(self.coverage_status, "coverage_status")
        _validate_supported(
            self.coverage_status,
            "coverage_status",
            SUPPORTED_SHADOW_SCORE_STATUSES,
        )
        self.warning_codes = _ordered_unique(self.warning_codes)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShadowFactorScore":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_SCORING_SCHEMA_VERSION)),
            factor_id=str(data.get("factor_id", "")),
            factor_version=str(data.get("factor_version", "")),
            symbol=str(data.get("symbol", "")),
            as_of=str(data.get("as_of", "")),
            raw_value=data.get("raw_value"),
            normalized_score=data.get("normalized_score"),
            rank=data.get("rank"),
            coverage_status=str(data.get("coverage_status", SHADOW_SCORE_STATUS_INSUFFICIENT_DATA)),
            warning_codes=list(data.get("warning_codes", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ShadowCandidateScore:
    schema_version: str = FACTOR_SHADOW_SCORING_SCHEMA_VERSION
    symbol: str = ""
    name: str | None = None
    as_of: str = ""
    official_score: float | None = None
    official_rank: int | None = None
    shadow_factor_score: float | None = None
    shadow_factor_rank: int | None = None
    rank_delta: int | None = None
    score_delta: float | None = None
    factor_count: int = 0
    covered_factor_count: int = 0
    factor_coverage_ratio: float = 0.0
    warning_codes: list[str] = field(default_factory=list)
    factor_scores: list[ShadowFactorScore] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_SCORING_SCHEMA_VERSION)
        self.symbol = _non_empty_str(self.symbol, "symbol")
        self.name = _optional_str(self.name)
        self.as_of = _non_empty_str(self.as_of, "as_of")
        self.official_score = _optional_finite_float(self.official_score, "official_score")
        self.official_rank = _positive_int_or_none(self.official_rank, "official_rank")
        self.shadow_factor_score = _unit_float_or_none(
            self.shadow_factor_score,
            "shadow_factor_score",
        )
        self.shadow_factor_rank = _positive_int_or_none(
            self.shadow_factor_rank,
            "shadow_factor_rank",
        )
        self.rank_delta = None if self.rank_delta is None else int(self.rank_delta)
        self.score_delta = _optional_finite_float(self.score_delta, "score_delta")
        self.factor_count = _non_negative_int(self.factor_count, "factor_count")
        self.covered_factor_count = _non_negative_int(
            self.covered_factor_count,
            "covered_factor_count",
        )
        if self.covered_factor_count > self.factor_count:
            raise ValueError("covered_factor_count must be <= factor_count.")
        self.factor_coverage_ratio = _unit_float(
            self.factor_coverage_ratio,
            "factor_coverage_ratio",
        )
        self.warning_codes = _ordered_unique(self.warning_codes)
        self.factor_scores = [
            score if isinstance(score, ShadowFactorScore) else ShadowFactorScore.from_dict(score)
            for score in self.factor_scores
        ]
        self.factor_scores = sorted(
            self.factor_scores,
            key=lambda score: (score.factor_id, score.factor_version, score.symbol),
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "name": self.name,
            "as_of": self.as_of,
            "official_score": self.official_score,
            "official_rank": self.official_rank,
            "shadow_factor_score": self.shadow_factor_score,
            "shadow_factor_rank": self.shadow_factor_rank,
            "rank_delta": self.rank_delta,
            "score_delta": self.score_delta,
            "factor_count": self.factor_count,
            "covered_factor_count": self.covered_factor_count,
            "factor_coverage_ratio": self.factor_coverage_ratio,
            "warning_codes": list(self.warning_codes),
            "factor_scores": [score.to_dict() for score in self.factor_scores],
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShadowCandidateScore":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_SCORING_SCHEMA_VERSION)),
            symbol=str(data.get("symbol", "")),
            name=data.get("name"),
            as_of=str(data.get("as_of", "")),
            official_score=data.get("official_score"),
            official_rank=data.get("official_rank"),
            shadow_factor_score=data.get("shadow_factor_score"),
            shadow_factor_rank=data.get("shadow_factor_rank"),
            rank_delta=data.get("rank_delta"),
            score_delta=data.get("score_delta"),
            factor_count=int(data.get("factor_count", 0)),
            covered_factor_count=int(data.get("covered_factor_count", 0)),
            factor_coverage_ratio=float(data.get("factor_coverage_ratio", 0.0)),
            warning_codes=list(data.get("warning_codes", []) or []),
            factor_scores=[
                ShadowFactorScore.from_dict(score)
                for score in list(data.get("factor_scores", []) or [])
                if isinstance(score, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ShadowScoringComparisonReport:
    schema_version: str = FACTOR_SHADOW_COMPARISON_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    as_of: str = ""
    config: ShadowScoringConfig = field(
        default_factory=lambda: ShadowScoringConfig(as_of="1970-01-01")
    )
    production_factor_count: int = 0
    used_factor_count: int = 0
    candidate_count: int = 0
    scored_candidate_count: int = 0
    average_factor_coverage_ratio: float | None = None
    official_top_symbols: list[str] = field(default_factory=list)
    shadow_top_symbols: list[str] = field(default_factory=list)
    overlap_top_symbols: list[str] = field(default_factory=list)
    overlap_ratio: float | None = None
    largest_positive_rank_deltas: list[dict[str, Any]] = field(default_factory=list)
    largest_negative_rank_deltas: list[dict[str, Any]] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)
    status: str = SHADOW_COMPARISON_STATUS_WARN
    candidate_scores: list[ShadowCandidateScore] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_COMPARISON_SCHEMA_VERSION)
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        self.as_of = _non_empty_str(self.as_of, "as_of")
        if not isinstance(self.config, ShadowScoringConfig):
            self.config = ShadowScoringConfig.from_dict(self.config)
        for field_name in [
            "production_factor_count",
            "used_factor_count",
            "candidate_count",
            "scored_candidate_count",
        ]:
            setattr(self, field_name, _non_negative_int(getattr(self, field_name), field_name))
        self.average_factor_coverage_ratio = _unit_float_or_none(
            self.average_factor_coverage_ratio,
            "average_factor_coverage_ratio",
        )
        self.official_top_symbols = [str(symbol) for symbol in self.official_top_symbols]
        self.shadow_top_symbols = [str(symbol) for symbol in self.shadow_top_symbols]
        self.overlap_top_symbols = [str(symbol) for symbol in self.overlap_top_symbols]
        self.overlap_ratio = _unit_float_or_none(self.overlap_ratio, "overlap_ratio")
        self.largest_positive_rank_deltas = _coerce_json_list(
            self.largest_positive_rank_deltas,
            "largest_positive_rank_deltas",
        )
        self.largest_negative_rank_deltas = _coerce_json_list(
            self.largest_negative_rank_deltas,
            "largest_negative_rank_deltas",
        )
        self.warning_codes = _ordered_unique(self.warning_codes)
        self.status = _non_empty_str(self.status, "status")
        _validate_supported(self.status, "status", SUPPORTED_SHADOW_COMPARISON_STATUSES)
        self.candidate_scores = [
            score if isinstance(score, ShadowCandidateScore) else ShadowCandidateScore.from_dict(score)
            for score in self.candidate_scores
        ]
        self.candidate_scores = sorted(
            self.candidate_scores,
            key=lambda score: (
                score.official_rank is None,
                score.official_rank if score.official_rank is not None else 10**9,
                score.symbol,
            ),
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "as_of": self.as_of,
            "config": self.config.to_dict(),
            "production_factor_count": self.production_factor_count,
            "used_factor_count": self.used_factor_count,
            "candidate_count": self.candidate_count,
            "scored_candidate_count": self.scored_candidate_count,
            "average_factor_coverage_ratio": self.average_factor_coverage_ratio,
            "official_top_symbols": list(self.official_top_symbols),
            "shadow_top_symbols": list(self.shadow_top_symbols),
            "overlap_top_symbols": list(self.overlap_top_symbols),
            "overlap_ratio": self.overlap_ratio,
            "largest_positive_rank_deltas": _json_safe(self.largest_positive_rank_deltas),
            "largest_negative_rank_deltas": _json_safe(self.largest_negative_rank_deltas),
            "warning_codes": list(self.warning_codes),
            "status": self.status,
            "candidate_scores": [score.to_dict() for score in self.candidate_scores],
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShadowScoringComparisonReport":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_COMPARISON_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            generated_at=str(data.get("generated_at", "")),
            as_of=str(data.get("as_of", "")),
            config=ShadowScoringConfig.from_dict(dict(data.get("config", {}) or {})),
            production_factor_count=int(data.get("production_factor_count", 0)),
            used_factor_count=int(data.get("used_factor_count", 0)),
            candidate_count=int(data.get("candidate_count", 0)),
            scored_candidate_count=int(data.get("scored_candidate_count", 0)),
            average_factor_coverage_ratio=data.get("average_factor_coverage_ratio"),
            official_top_symbols=list(data.get("official_top_symbols", []) or []),
            shadow_top_symbols=list(data.get("shadow_top_symbols", []) or []),
            overlap_top_symbols=list(data.get("overlap_top_symbols", []) or []),
            overlap_ratio=data.get("overlap_ratio"),
            largest_positive_rank_deltas=[
                dict(row)
                for row in list(data.get("largest_positive_rank_deltas", []) or [])
                if isinstance(row, Mapping)
            ],
            largest_negative_rank_deltas=[
                dict(row)
                for row in list(data.get("largest_negative_rank_deltas", []) or [])
                if isinstance(row, Mapping)
            ],
            warning_codes=list(data.get("warning_codes", []) or []),
            status=str(data.get("status", SHADOW_COMPARISON_STATUS_WARN)),
            candidate_scores=[
                ShadowCandidateScore.from_dict(score)
                for score in list(data.get("candidate_scores", []) or [])
                if isinstance(score, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_shadow_scoring_config_id(config: ShadowScoringConfig) -> str:
    payload = config.to_dict()
    payload["config_id"] = ""
    parts = [
        payload.get("as_of", ""),
        payload.get("top_n", ""),
        payload.get("min_factor_coverage_ratio", ""),
        payload,
    ]
    return f"factor-shadow-scoring-config-{_slug(config.as_of)}-{_short_hash(parts)}"


def make_shadow_comparison_report_id(
    *,
    as_of: str,
    generated_at: str,
    candidate_symbols: Sequence[str],
) -> str:
    ordered_symbols = sorted({str(symbol).strip() for symbol in candidate_symbols if str(symbol).strip()})
    parts = [str(as_of), str(generated_at), ordered_symbols]
    return f"factor-shadow-comparison-{_slug(as_of)}-{_short_hash(parts)}"


def extract_factor_value_for_symbol(
    matrix: FactorMatrix,
    *,
    symbol: str,
    as_of: str,
) -> tuple[float | None, str]:
    resolved_symbol = str(symbol).strip()
    if resolved_symbol not in matrix.symbols:
        return None, SHADOW_SCORE_STATUS_MISSING_SYMBOL

    as_of_date = str(as_of).strip()[:10]
    eligible_dates = [
        (date_value, index)
        for index, date_value in enumerate(matrix.dates)
        if str(date_value) <= as_of_date
    ]
    if not eligible_dates:
        return None, SHADOW_SCORE_STATUS_MISSING_DATE

    _date_value, date_index = max(eligible_dates, key=lambda item: (item[0], item[1]))
    symbol_index = matrix.symbols.index(resolved_symbol)
    try:
        raw_value = matrix.values[symbol_index][date_index]
    except IndexError:
        return None, SHADOW_SCORE_STATUS_INSUFFICIENT_DATA
    if not _is_finite_number(raw_value):
        return None, SHADOW_SCORE_STATUS_INSUFFICIENT_DATA
    return float(raw_value), SHADOW_SCORE_STATUS_OK


def rank_normalize_factor_values(
    values_by_symbol: Mapping[str, float | None],
    *,
    expected_direction: float = 1.0,
) -> dict[str, tuple[float | None, int | None]]:
    direction = -1.0 if float(expected_direction) < 0 else 1.0
    valid_rows = [
        (str(symbol), float(value) * direction)
        for symbol, value in values_by_symbol.items()
        if _is_finite_number(value)
    ]
    valid_rows = sorted(valid_rows, key=lambda item: (-item[1], item[0]))

    output: dict[str, tuple[float | None, int | None]] = {
        str(symbol): (None, None)
        for symbol in values_by_symbol.keys()
    }
    count = len(valid_rows)
    for index, (symbol, _adjusted_value) in enumerate(valid_rows, start=1):
        normalized = 1.0 if count == 1 else 1.0 - ((index - 1) / (count - 1))
        output[symbol] = (float(normalized), index)
    return output


def resolve_factor_expected_direction(
    *,
    factor_id: str,
    factor_version: str,
    definitions: Sequence[FactorDefinition] | None = None,
    matrix: FactorMatrix | None = None,
) -> float:
    for definition in definitions or []:
        if definition.factor_id == factor_id and definition.version == factor_version:
            return -1.0 if float(definition.expected_direction) < 0 else 1.0

    if matrix is not None:
        value = matrix.metadata.get("expected_direction")
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 1.0
        return -1.0 if number < 0 else 1.0
    return 1.0


def build_factor_matrix_lookup(
    matrices: Sequence[FactorMatrix],
) -> dict[tuple[str, str], FactorMatrix]:
    lookup: dict[tuple[str, str], FactorMatrix] = {}
    for matrix in matrices:
        if not matrix.factor_id or not matrix.factor_version:
            continue
        key = (matrix.factor_id, matrix.factor_version)
        max_date = max(matrix.dates) if matrix.dates else ""
        current = lookup.get(key)
        if current is None:
            lookup[key] = matrix
            continue
        current_max_date = max(current.dates) if current.dates else ""
        if (max_date, _slug(current.matrix_id)) > (current_max_date, _slug(current.matrix_id)):
            lookup[key] = matrix
        elif max_date == current_max_date and matrix.matrix_id < current.matrix_id:
            lookup[key] = matrix
    return dict(sorted(lookup.items(), key=lambda item: item[0]))


def select_usable_production_factors(
    *,
    library: ProductionFactorLibrary | None,
    audit_report: Any | None = None,
    include_blocked_factors: bool = False,
) -> list[FactorLibraryEntry]:
    if library is None:
        return []
    blocked_ids = set() if include_blocked_factors else _blocked_factor_ids(audit_report)
    entries = [
        entry
        for entry in library.entries
        if entry.status == FACTOR_STATUS_PRODUCTION and entry.factor_id not in blocked_ids
    ]
    return sorted(entries, key=lambda entry: (entry.factor_id, entry.factor_version))


def _extract_candidate_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        payload = dict(candidate)
        symbol = _optional_str(_first_present(payload, ["symbol", "ts_code", "code"]))
        if symbol is None:
            continue
        name = _optional_str(_first_present(payload, ["name", "company_name", "stock_name"]))
        official_score = _to_optional_float(
            _first_present(
                payload,
                ["official_score", "final_score", "posterior_action_score", "score"],
            )
        )
        official_rank_raw = _first_present(payload, ["official_rank", "rank", "final_rank"])
        official_rank = None
        if official_rank_raw is not None:
            try:
                official_rank = _positive_int(official_rank_raw, "official_rank")
            except (TypeError, ValueError):
                official_rank = None
        rows.append(
            {
                "index": index,
                "symbol": symbol,
                "name": name,
                "official_score": official_score,
                "official_rank": official_rank,
            }
        )

    derived_order = sorted(
        rows,
        key=lambda row: (
            row["official_score"] is None,
            -float(row["official_score"] or 0.0),
            row["symbol"],
        ),
    )
    derived_ranks = {row["symbol"]: rank for rank, row in enumerate(derived_order, start=1)}
    for row in rows:
        if row["official_rank"] is None:
            row["official_rank"] = derived_ranks.get(row["symbol"])
    return rows


def _rank_shadow_scores(scores_by_symbol: Mapping[str, float | None]) -> dict[str, int | None]:
    valid_rows = [
        (str(symbol), float(score))
        for symbol, score in scores_by_symbol.items()
        if _is_finite_number(score)
    ]
    valid_rows = sorted(valid_rows, key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int | None] = {str(symbol): None for symbol in scores_by_symbol.keys()}
    for index, (symbol, _score) in enumerate(valid_rows, start=1):
        ranks[symbol] = index
    return ranks


def build_shadow_candidate_scores(
    *,
    candidates: Sequence[Mapping[str, Any]],
    library: ProductionFactorLibrary | None,
    factor_matrices: Sequence[FactorMatrix],
    definitions: Sequence[FactorDefinition] | None = None,
    audit_report: Any | None = None,
    config: ShadowScoringConfig,
    metadata: Mapping[str, Any] | None = None,
) -> list[ShadowCandidateScore]:
    candidate_rows = _extract_candidate_rows(candidates)
    candidate_symbols = [row["symbol"] for row in candidate_rows]
    matrix_lookup = build_factor_matrix_lookup(factor_matrices)
    usable_factors = select_usable_production_factors(
        library=library,
        audit_report=audit_report,
        include_blocked_factors=config.include_blocked_factors,
    )

    blocked_ids = _blocked_factor_ids(audit_report)
    excluded_blocked_ids = sorted(
        blocked_ids
        - {entry.factor_id for entry in usable_factors}
    )
    warnings_by_symbol: dict[str, set[str]] = {symbol: set() for symbol in candidate_symbols}
    factor_scores_by_symbol: dict[str, list[ShadowFactorScore]] = {
        symbol: []
        for symbol in candidate_symbols
    }
    normalized_by_symbol_factor: dict[tuple[str, str, str], float | None] = {}

    if library is None:
        for symbol in candidate_symbols:
            warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_LIBRARY_MISSING)
    if excluded_blocked_ids and not config.include_blocked_factors:
        for symbol in candidate_symbols:
            warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_AUDIT_BLOCKED)

    non_production_entries = [
        entry.factor_id
        for entry in (library.entries if library is not None else [])
        if entry.status != FACTOR_STATUS_PRODUCTION
    ]
    if non_production_entries:
        for symbol in candidate_symbols:
            warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_NON_PRODUCTION_FACTOR)

    for entry in usable_factors:
        key = _entry_key(entry)
        matrix = matrix_lookup.get(key)
        raw_values: dict[str, float | None] = {}
        statuses: dict[str, str] = {}
        if matrix is None:
            for symbol in candidate_symbols:
                raw_values[symbol] = None
                statuses[symbol] = SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX
                warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX)
        else:
            for symbol in candidate_symbols:
                value, status = extract_factor_value_for_symbol(
                    matrix,
                    symbol=symbol,
                    as_of=config.as_of,
                )
                raw_values[symbol] = value
                statuses[symbol] = status
                if status != SHADOW_SCORE_STATUS_OK:
                    warnings_by_symbol[symbol].add(status)

        direction = resolve_factor_expected_direction(
            factor_id=entry.factor_id,
            factor_version=entry.factor_version,
            definitions=definitions,
            matrix=matrix,
        )
        normalized = (
            rank_normalize_factor_values(raw_values, expected_direction=direction)
            if config.normalize_factor_scores
            else {
                symbol: (raw_values[symbol], None)
                for symbol in raw_values
            }
        )
        for symbol in candidate_symbols:
            normalized_score, rank = normalized.get(symbol, (None, None))
            if normalized_score is not None:
                normalized_by_symbol_factor[(symbol, entry.factor_id, entry.factor_version)] = normalized_score
            warning_codes = [] if statuses[symbol] == SHADOW_SCORE_STATUS_OK else [statuses[symbol]]
            factor_scores_by_symbol[symbol].append(
                ShadowFactorScore(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    symbol=symbol,
                    as_of=config.as_of,
                    raw_value=raw_values[symbol],
                    normalized_score=normalized_score,
                    rank=rank,
                    coverage_status=statuses[symbol],
                    warning_codes=warning_codes,
                    metadata={
                        **_coerce_json_dict(metadata, "metadata"),
                        "expected_direction": direction,
                        "matrix_id": matrix.matrix_id if matrix is not None else None,
                    },
                )
            )

    shadow_scores_by_symbol: dict[str, float | None] = {}
    for symbol in candidate_symbols:
        covered_scores = [
            score.normalized_score
            for score in factor_scores_by_symbol[symbol]
            if score.normalized_score is not None
        ]
        shadow_scores_by_symbol[symbol] = (
            float(sum(covered_scores) / len(covered_scores))
            if covered_scores
            else None
        )
        if not covered_scores:
            warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_INSUFFICIENT_DATA)

    shadow_ranks = _rank_shadow_scores(shadow_scores_by_symbol)
    official_scores_outside_unit = any(
        row["official_score"] is not None
        and not 0.0 <= float(row["official_score"]) <= 1.0
        for row in candidate_rows
    )

    output: list[ShadowCandidateScore] = []
    for row in candidate_rows:
        symbol = row["symbol"]
        factor_scores = factor_scores_by_symbol[symbol]
        covered_factor_count = sum(
            1
            for score in factor_scores
            if score.normalized_score is not None
        )
        factor_count = len(usable_factors)
        coverage_ratio = covered_factor_count / factor_count if factor_count else 0.0
        shadow_rank = shadow_ranks.get(symbol)
        official_rank = row["official_rank"]
        shadow_score = shadow_scores_by_symbol.get(symbol)
        rank_delta = (
            int(official_rank) - int(shadow_rank)
            if official_rank is not None and shadow_rank is not None
            else None
        )
        score_delta = (
            float(shadow_score) - float(row["official_score"])
            if shadow_score is not None and row["official_score"] is not None
            else None
        )
        candidate_metadata = {
            **_coerce_json_dict(metadata, "metadata"),
            "factor_weight_policy": config.factor_weight_policy,
        }
        if official_scores_outside_unit:
            candidate_metadata["official_score_scale_note"] = (
                "official_score is outside [0, 1]; score_delta is a raw difference"
            )
        output.append(
            ShadowCandidateScore(
                symbol=symbol,
                name=row["name"],
                as_of=config.as_of,
                official_score=row["official_score"],
                official_rank=official_rank,
                shadow_factor_score=shadow_score,
                shadow_factor_rank=shadow_rank,
                rank_delta=rank_delta,
                score_delta=score_delta,
                factor_count=factor_count,
                covered_factor_count=covered_factor_count,
                factor_coverage_ratio=coverage_ratio,
                warning_codes=list(warnings_by_symbol[symbol]),
                factor_scores=factor_scores,
                metadata=candidate_metadata,
            )
        )

    return sorted(
        output,
        key=lambda score: (
            score.official_rank is None,
            score.official_rank if score.official_rank is not None else 10**9,
            score.symbol,
        ),
    )


def _delta_row(candidate: ShadowCandidateScore) -> dict[str, Any]:
    return {
        "symbol": candidate.symbol,
        "name": candidate.name,
        "official_rank": candidate.official_rank,
        "shadow_factor_rank": candidate.shadow_factor_rank,
        "rank_delta": candidate.rank_delta,
        "official_score": candidate.official_score,
        "shadow_factor_score": candidate.shadow_factor_score,
        "factor_coverage_ratio": candidate.factor_coverage_ratio,
        "warning_codes": list(candidate.warning_codes),
    }


def build_shadow_scoring_comparison_report(
    *,
    candidates: Sequence[Mapping[str, Any]],
    library: ProductionFactorLibrary | None,
    factor_matrices: Sequence[FactorMatrix],
    definitions: Sequence[FactorDefinition] | None = None,
    audit_report: Any | None = None,
    config: ShadowScoringConfig,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> ShadowScoringComparisonReport:
    candidate_scores = build_shadow_candidate_scores(
        candidates=candidates,
        library=library,
        factor_matrices=factor_matrices,
        definitions=definitions,
        audit_report=audit_report,
        config=config,
        metadata=metadata,
    )
    candidate_count = len(candidate_scores)
    selected_factors = select_usable_production_factors(
        library=library,
        audit_report=audit_report,
        include_blocked_factors=config.include_blocked_factors,
    )
    matrix_lookup = build_factor_matrix_lookup(factor_matrices)
    production_factor_count = (
        len([entry for entry in library.entries if entry.status == FACTOR_STATUS_PRODUCTION])
        if library is not None
        else 0
    )
    used_factor_count = sum(1 for entry in selected_factors if _entry_key(entry) in matrix_lookup)
    scored_candidate_count = sum(
        1
        for candidate in candidate_scores
        if candidate.shadow_factor_score is not None
    )
    average_factor_coverage_ratio = (
        sum(candidate.factor_coverage_ratio for candidate in candidate_scores) / candidate_count
        if candidate_count
        else None
    )

    official_sorted = sorted(
        candidate_scores,
        key=lambda score: (
            score.official_rank is None,
            score.official_rank if score.official_rank is not None else 10**9,
            score.symbol,
        ),
    )
    shadow_sorted = sorted(
        [score for score in candidate_scores if score.shadow_factor_rank is not None],
        key=lambda score: (score.shadow_factor_rank or 10**9, score.symbol),
    )
    official_top_symbols = [score.symbol for score in official_sorted[: config.top_n]]
    shadow_top_symbols = [score.symbol for score in shadow_sorted[: config.top_n]]
    shadow_top_set = set(shadow_top_symbols)
    overlap_top_symbols = [
        score.symbol
        for score in official_sorted[: config.top_n]
        if score.symbol in shadow_top_set
    ]
    denominator = min(config.top_n, candidate_count)
    overlap_ratio = (
        len(overlap_top_symbols) / denominator
        if denominator
        else None
    )

    positive_deltas = sorted(
        [
            candidate
            for candidate in candidate_scores
            if candidate.rank_delta is not None and candidate.rank_delta > 0
        ],
        key=lambda score: (-(score.rank_delta or 0), score.symbol),
    )
    negative_deltas = sorted(
        [
            candidate
            for candidate in candidate_scores
            if candidate.rank_delta is not None and candidate.rank_delta < 0
        ],
        key=lambda score: ((score.rank_delta or 0), score.symbol),
    )
    warning_codes = set()
    for candidate in candidate_scores:
        warning_codes.update(candidate.warning_codes)
        if (
            candidate.rank_delta is not None
            and abs(candidate.rank_delta) > config.max_rank_delta_warning
        ):
            warning_codes.add("large_rank_delta")
    if library is None:
        warning_codes.add(SHADOW_SCORE_STATUS_LIBRARY_MISSING)
    if selected_factors and used_factor_count < len(selected_factors):
        warning_codes.add(SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX)
    if production_factor_count == 0 or not selected_factors:
        warning_codes.add("no_usable_production_factors")
    if (
        average_factor_coverage_ratio is not None
        and average_factor_coverage_ratio < config.min_factor_coverage_ratio
    ):
        warning_codes.add("low_factor_coverage")

    status = SHADOW_COMPARISON_STATUS_OK
    if warning_codes:
        status = SHADOW_COMPARISON_STATUS_WARN

    base_metadata = _coerce_json_dict(metadata, "metadata")
    report_metadata = {
        **base_metadata,
        "factor_shadow_scoring_schema_version": FACTOR_SHADOW_SCORING_SCHEMA_VERSION,
        "factor_shadow_comparison_schema_version": FACTOR_SHADOW_COMPARISON_SCHEMA_VERSION,
        "non_runtime_impact": True,
        "non_runtime_impact_note": SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
    }
    report_id = make_shadow_comparison_report_id(
        as_of=config.as_of,
        generated_at=generated_at,
        candidate_symbols=[candidate.symbol for candidate in candidate_scores],
    )
    return ShadowScoringComparisonReport(
        report_id=report_id,
        generated_at=generated_at,
        as_of=config.as_of,
        config=config,
        production_factor_count=production_factor_count,
        used_factor_count=used_factor_count,
        candidate_count=candidate_count,
        scored_candidate_count=scored_candidate_count,
        average_factor_coverage_ratio=average_factor_coverage_ratio,
        official_top_symbols=official_top_symbols,
        shadow_top_symbols=shadow_top_symbols,
        overlap_top_symbols=overlap_top_symbols,
        overlap_ratio=overlap_ratio,
        largest_positive_rank_deltas=[_delta_row(candidate) for candidate in positive_deltas[: config.top_n]],
        largest_negative_rank_deltas=[_delta_row(candidate) for candidate in negative_deltas[: config.top_n]],
        warning_codes=list(warning_codes),
        status=status,
        candidate_scores=candidate_scores,
        metadata=report_metadata,
    )


def _escape_pipe(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_optional_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _render_delta_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- None"]
    output = [
        "| Symbol | Name | Official Rank | Shadow Rank | Rank Delta | Shadow Score | Coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        output.append(
            "| "
            f"`{_escape_pipe(row.get('symbol', ''))}` | "
            f"{_escape_pipe(row.get('name') or '')} | "
            f"{row.get('official_rank') or ''} | "
            f"{row.get('shadow_factor_rank') or ''} | "
            f"{row.get('rank_delta') or ''} | "
            f"{_format_optional_float(row.get('shadow_factor_score'))} | "
            f"{_format_optional_float(row.get('factor_coverage_ratio'))} |"
        )
    return output


def render_shadow_scoring_comparison_markdown(
    report: ShadowScoringComparisonReport,
) -> str:
    lines = [
        f"# Factor Shadow Scoring Comparison: {report.report_id}",
        "",
        f"Generated at: `{_escape_pipe(report.generated_at)}`",
        f"As of: `{_escape_pipe(report.as_of)}`",
        "",
        "## Status",
        "",
        f"`{_escape_pipe(report.status)}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Production factor count | {report.production_factor_count} |",
        f"| Used factor count | {report.used_factor_count} |",
        f"| Candidate count | {report.candidate_count} |",
        f"| Scored candidate count | {report.scored_candidate_count} |",
        (
            "| Average factor coverage ratio | "
            f"{_format_optional_float(report.average_factor_coverage_ratio)} |"
        ),
        "",
        "## Top-N Overlap Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Top N | {report.config.top_n} |",
        f"| Overlap ratio | `{_format_optional_float(report.overlap_ratio)}` |",
        f"| Official top symbols | `{_escape_pipe(', '.join(report.official_top_symbols))}` |",
        f"| Shadow top symbols | `{_escape_pipe(', '.join(report.shadow_top_symbols))}` |",
        f"| Overlap symbols | `{_escape_pipe(', '.join(report.overlap_top_symbols))}` |",
        "",
        "## Largest Positive Rank Deltas",
        "",
    ]
    lines.extend(_render_delta_rows(report.largest_positive_rank_deltas))
    lines.extend(["", "## Largest Negative Rank Deltas", ""])
    lines.extend(_render_delta_rows(report.largest_negative_rank_deltas))

    lines.extend(
        [
            "",
            "## Candidate Score Table",
            "",
            "| Official Rank | Symbol | Name | Official Score | Shadow Rank | Shadow Score | Rank Delta | Coverage | Warnings |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if report.candidate_scores:
        for candidate in report.candidate_scores:
            lines.append(
                "| "
                f"{candidate.official_rank or ''} | "
                f"`{_escape_pipe(candidate.symbol)}` | "
                f"{_escape_pipe(candidate.name or '')} | "
                f"{_format_optional_float(candidate.official_score)} | "
                f"{candidate.shadow_factor_rank or ''} | "
                f"{_format_optional_float(candidate.shadow_factor_score)} | "
                f"{candidate.rank_delta if candidate.rank_delta is not None else ''} | "
                f"{_format_optional_float(candidate.factor_coverage_ratio)} | "
                f"`{_escape_pipe(', '.join(candidate.warning_codes))}` |"
            )
    else:
        lines.append("|  |  |  |  |  |  |  |  | No candidates. |")

    lines.extend(["", "## Warnings", ""])
    if report.warning_codes:
        lines.extend([f"- `{_escape_pipe(code)}`" for code in report.warning_codes])
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Runtime Impact",
            "",
            SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def build_shadow_scoring_dashboard_payload(
    report: ShadowScoringComparisonReport,
) -> dict[str, Any]:
    payload = {
        "status": report.status,
        "as_of": report.as_of,
        "production_factor_count": report.production_factor_count,
        "used_factor_count": report.used_factor_count,
        "candidate_count": report.candidate_count,
        "scored_candidate_count": report.scored_candidate_count,
        "overlap_ratio": report.overlap_ratio,
        "official_top_symbols": list(report.official_top_symbols),
        "shadow_top_symbols": list(report.shadow_top_symbols),
        "overlap_top_symbols": list(report.overlap_top_symbols),
        "warning_codes": list(report.warning_codes),
        "largest_positive_rank_deltas": _json_safe(report.largest_positive_rank_deltas),
        "largest_negative_rank_deltas": _json_safe(report.largest_negative_rank_deltas),
        "metadata": dict(_json_safe(report.metadata)),
    }
    json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return dict(_json_safe(payload))


__all__ = [
    "SHADOW_SCORE_STATUS_OK",
    "SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX",
    "SHADOW_SCORE_STATUS_MISSING_SYMBOL",
    "SHADOW_SCORE_STATUS_MISSING_DATE",
    "SHADOW_SCORE_STATUS_NON_PRODUCTION_FACTOR",
    "SHADOW_SCORE_STATUS_LIBRARY_MISSING",
    "SHADOW_SCORE_STATUS_AUDIT_BLOCKED",
    "SHADOW_SCORE_STATUS_INSUFFICIENT_DATA",
    "SHADOW_COMPARISON_STATUS_OK",
    "SHADOW_COMPARISON_STATUS_WARN",
    "SHADOW_COMPARISON_STATUS_FAIL",
    "DEFAULT_FACTOR_SHADOW_SCORING_DIR",
    "DEFAULT_SHADOW_FACTOR_SCORES_FILENAME",
    "DEFAULT_SHADOW_CANDIDATE_SCORES_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_REPORTS_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_MARKDOWN_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_DASHBOARD_FILENAME",
    "SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE",
    "ShadowScoringConfig",
    "ShadowFactorScore",
    "ShadowCandidateScore",
    "ShadowScoringComparisonReport",
    "make_shadow_scoring_config_id",
    "make_shadow_comparison_report_id",
    "extract_factor_value_for_symbol",
    "rank_normalize_factor_values",
    "resolve_factor_expected_direction",
    "build_factor_matrix_lookup",
    "select_usable_production_factors",
    "build_shadow_candidate_scores",
    "build_shadow_scoring_comparison_report",
    "render_shadow_scoring_comparison_markdown",
    "build_shadow_scoring_dashboard_payload",
]
