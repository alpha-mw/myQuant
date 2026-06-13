"""Contracts and serialization helpers for factor shadow scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.schema import FactorLibraryEntry
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
]
