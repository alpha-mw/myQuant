"""Offline multi-date factor shadow evidence collection.

This module aggregates read-only shadow scoring and audit artifacts across
local as-of dates. It does not fetch data, call providers, or wire factor
library signals into official selection, posterior scoring, risk, portfolio
construction, orders, or execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.matrix import FactorMatrix
from quant_investor.factors.schema import (
    FACTOR_STATUS_PRODUCTION,
    ProductionFactorLibrary,
)
from quant_investor.factors.shadow_scoring import (
    ShadowScoringConfig,
    build_shadow_scoring_comparison_report,
)
from quant_investor.versioning import (
    FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION,
    FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION,
)


EVIDENCE_STATUS_OK = "ok"
EVIDENCE_STATUS_WARN = "warn"
EVIDENCE_STATUS_FAIL = "fail"
EVIDENCE_STATUS_INSUFFICIENT_DATA = "insufficient_data"

EVIDENCE_ISSUE_MISSING_PRODUCTION_LIBRARY = "missing_production_library"
EVIDENCE_ISSUE_MISSING_FACTOR_MATRICES = "missing_factor_matrices"
EVIDENCE_ISSUE_MISSING_CANDIDATES = "missing_candidates"
EVIDENCE_ISSUE_LOW_FACTOR_COVERAGE = "low_factor_coverage"
EVIDENCE_ISSUE_LOW_TOP_N_OVERLAP = "low_top_n_overlap"
EVIDENCE_ISSUE_LARGE_RANK_DRIFT = "large_rank_drift"
EVIDENCE_ISSUE_AUDIT_BLOCKER = "audit_blocker"
EVIDENCE_ISSUE_ALIGNMENT_AUDIT_FAIL = "alignment_audit_fail"
EVIDENCE_ISSUE_TRADABILITY_AUDIT_FAIL = "tradability_audit_fail"
EVIDENCE_ISSUE_EXECUTION_COST_WARN = "execution_cost_warn"
EVIDENCE_ISSUE_INSUFFICIENT_OBSERVATION_DAYS = "insufficient_observation_days"

DEFAULT_FACTOR_EVIDENCE_DIR = Path("data/factor_library/evidence")
DEFAULT_EVIDENCE_DATE_RESULTS_FILENAME = "evidence_date_results.jsonl"
DEFAULT_MULTI_DATE_EVIDENCE_REPORTS_FILENAME = "multi_date_evidence_reports.jsonl"
DEFAULT_EVIDENCE_DASHBOARD_FILENAME = "evidence_dashboard.json"
DEFAULT_EVIDENCE_MARKDOWN_FILENAME = "evidence_report.md"

SUPPORTED_EVIDENCE_STATUSES = {
    EVIDENCE_STATUS_OK,
    EVIDENCE_STATUS_WARN,
    EVIDENCE_STATUS_FAIL,
    EVIDENCE_STATUS_INSUFFICIENT_DATA,
}

NON_RUNTIME_IMPACT_NOTE = (
    "This evidence report is offline-only and does not alter official scores, "
    "stock selection, posterior, RiskGuard, PortfolioConstructor, target "
    "weights, orders, providers, LLMs, or execution."
)


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


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(_ensure_json_serializable(value or {}, "metadata"))


def _json_dict(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    return dict(_ensure_json_serializable(value or {}, label))


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


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive integer; got {value!r}.")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive; got {value!r}.")
    return number


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be non-negative integer; got {value!r}.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric value; got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


def _unit_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]; got {value!r}.")
    return number


def _unit_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _unit_float(value, field_name)


def _non_negative_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool; got {value!r}.")
    return value


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _sorted_iso_dates(values: Sequence[Any]) -> list[str]:
    dates = _ordered_unique(values)
    for value in dates:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"as_of_dates must be ISO dates; got {value!r}.")
    return dates


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


def _validate_status(value: str) -> None:
    if value not in SUPPORTED_EVIDENCE_STATUSES:
        raise ValueError(f"status must be one of {sorted(SUPPORTED_EVIDENCE_STATUSES)}; got {value!r}.")


def _candidate_symbols(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    symbols: list[str] = []
    for row in candidates:
        for key in ("symbol", "ts_code", "code"):
            value = row.get(key)
            if value is not None and str(value).strip():
                symbols.append(str(value).strip())
                break
    return sorted(symbols)


@dataclass
class FactorEvidenceCollectionConfig:
    schema_version: str = FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION
    config_id: str = ""
    as_of_dates: list[str] = field(default_factory=list)
    top_n: int = 30
    min_observation_days: int = 20
    min_average_factor_coverage: float = 0.80
    min_top_n_overlap_ratio: float = 0.50
    max_average_abs_rank_delta: float | None = None
    require_library_audit_no_blocker: bool = True
    require_alignment_audit_pass: bool = False
    require_tradability_audit_pass: bool = False
    require_execution_cost_review: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)
        self.config_id = str(self.config_id)
        self.as_of_dates = _sorted_iso_dates(self.as_of_dates)
        self.top_n = _positive_int(self.top_n, "top_n")
        self.min_observation_days = _positive_int(self.min_observation_days, "min_observation_days")
        self.min_average_factor_coverage = _unit_float(
            self.min_average_factor_coverage,
            "min_average_factor_coverage",
        )
        self.min_top_n_overlap_ratio = _unit_float(
            self.min_top_n_overlap_ratio,
            "min_top_n_overlap_ratio",
        )
        self.max_average_abs_rank_delta = _non_negative_float_or_none(
            self.max_average_abs_rank_delta,
            "max_average_abs_rank_delta",
        )
        self.require_library_audit_no_blocker = _require_bool(
            self.require_library_audit_no_blocker,
            "require_library_audit_no_blocker",
        )
        self.require_alignment_audit_pass = _require_bool(
            self.require_alignment_audit_pass,
            "require_alignment_audit_pass",
        )
        self.require_tradability_audit_pass = _require_bool(
            self.require_tradability_audit_pass,
            "require_tradability_audit_pass",
        )
        self.require_execution_cost_review = _require_bool(
            self.require_execution_cost_review,
            "require_execution_cost_review",
        )
        self.metadata = _metadata(self.metadata)
        if not self.config_id:
            self.config_id = make_evidence_collection_config_id(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_id": self.config_id,
            "as_of_dates": list(self.as_of_dates),
            "top_n": self.top_n,
            "min_observation_days": self.min_observation_days,
            "min_average_factor_coverage": self.min_average_factor_coverage,
            "min_top_n_overlap_ratio": self.min_top_n_overlap_ratio,
            "max_average_abs_rank_delta": self.max_average_abs_rank_delta,
            "require_library_audit_no_blocker": self.require_library_audit_no_blocker,
            "require_alignment_audit_pass": self.require_alignment_audit_pass,
            "require_tradability_audit_pass": self.require_tradability_audit_pass,
            "require_execution_cost_review": self.require_execution_cost_review,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorEvidenceCollectionConfig":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)),
            config_id=str(data.get("config_id", "")),
            as_of_dates=list(data.get("as_of_dates", []) or []),
            top_n=int(data.get("top_n", 30)),
            min_observation_days=int(data.get("min_observation_days", 20)),
            min_average_factor_coverage=float(data.get("min_average_factor_coverage", 0.80)),
            min_top_n_overlap_ratio=float(data.get("min_top_n_overlap_ratio", 0.50)),
            max_average_abs_rank_delta=data.get("max_average_abs_rank_delta"),
            require_library_audit_no_blocker=data.get("require_library_audit_no_blocker", True),
            require_alignment_audit_pass=data.get("require_alignment_audit_pass", False),
            require_tradability_audit_pass=data.get("require_tradability_audit_pass", False),
            require_execution_cost_review=data.get("require_execution_cost_review", False),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorEvidenceDateInput:
    schema_version: str = FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION
    as_of: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    production_library_path: str | None = None
    factor_matrix_paths: list[str] = field(default_factory=list)
    library_audit_path: str | None = None
    alignment_audit_paths: list[str] = field(default_factory=list)
    tradability_audit_paths: list[str] = field(default_factory=list)
    execution_cost_report_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)
        self.as_of = _sorted_iso_dates([self.as_of])[0]
        self.candidates = [
            _json_dict(candidate, f"candidates[{index}]")
            for index, candidate in enumerate(self.candidates)
        ]
        self.production_library_path = _optional_str(self.production_library_path)
        self.factor_matrix_paths = _ordered_unique(self.factor_matrix_paths)
        self.library_audit_path = _optional_str(self.library_audit_path)
        self.alignment_audit_paths = _ordered_unique(self.alignment_audit_paths)
        self.tradability_audit_paths = _ordered_unique(self.tradability_audit_paths)
        self.execution_cost_report_paths = _ordered_unique(self.execution_cost_report_paths)
        self.metadata = _metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "candidates": [_json_safe(candidate) for candidate in self.candidates],
            "production_library_path": self.production_library_path,
            "factor_matrix_paths": list(self.factor_matrix_paths),
            "library_audit_path": self.library_audit_path,
            "alignment_audit_paths": list(self.alignment_audit_paths),
            "tradability_audit_paths": list(self.tradability_audit_paths),
            "execution_cost_report_paths": list(self.execution_cost_report_paths),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorEvidenceDateInput":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)),
            as_of=str(data.get("as_of", "")),
            candidates=[
                dict(candidate)
                for candidate in list(data.get("candidates", []) or [])
                if isinstance(candidate, Mapping)
            ],
            production_library_path=data.get("production_library_path"),
            factor_matrix_paths=list(data.get("factor_matrix_paths", []) or []),
            library_audit_path=data.get("library_audit_path"),
            alignment_audit_paths=list(data.get("alignment_audit_paths", []) or []),
            tradability_audit_paths=list(data.get("tradability_audit_paths", []) or []),
            execution_cost_report_paths=list(data.get("execution_cost_report_paths", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorAuditEvidenceSnapshot:
    schema_version: str = FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION
    as_of: str = ""
    library_exists: bool = False
    production_factor_count: int = 0
    library_audit_verdict: str | None = None
    library_blocker_count: int = 0
    library_warning_count: int = 0
    alignment_audit_verdicts: list[str] = field(default_factory=list)
    tradability_audit_verdicts: list[str] = field(default_factory=list)
    execution_cost_verdicts: list[str] = field(default_factory=list)
    audit_issue_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)
        self.as_of = _sorted_iso_dates([self.as_of])[0]
        self.library_exists = _require_bool(self.library_exists, "library_exists")
        self.production_factor_count = _non_negative_int(self.production_factor_count, "production_factor_count")
        self.library_audit_verdict = _optional_str(self.library_audit_verdict)
        self.library_blocker_count = _non_negative_int(self.library_blocker_count, "library_blocker_count")
        self.library_warning_count = _non_negative_int(self.library_warning_count, "library_warning_count")
        self.alignment_audit_verdicts = _ordered_unique(self.alignment_audit_verdicts)
        self.tradability_audit_verdicts = _ordered_unique(self.tradability_audit_verdicts)
        self.execution_cost_verdicts = _ordered_unique(self.execution_cost_verdicts)
        self.audit_issue_codes = _ordered_unique(self.audit_issue_codes)
        self.metadata = _metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "library_exists": self.library_exists,
            "production_factor_count": self.production_factor_count,
            "library_audit_verdict": self.library_audit_verdict,
            "library_blocker_count": self.library_blocker_count,
            "library_warning_count": self.library_warning_count,
            "alignment_audit_verdicts": list(self.alignment_audit_verdicts),
            "tradability_audit_verdicts": list(self.tradability_audit_verdicts),
            "execution_cost_verdicts": list(self.execution_cost_verdicts),
            "audit_issue_codes": list(self.audit_issue_codes),
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorAuditEvidenceSnapshot":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)),
            as_of=str(data.get("as_of", "")),
            library_exists=bool(data.get("library_exists", False)),
            production_factor_count=int(data.get("production_factor_count", 0)),
            library_audit_verdict=data.get("library_audit_verdict"),
            library_blocker_count=int(data.get("library_blocker_count", 0)),
            library_warning_count=int(data.get("library_warning_count", 0)),
            alignment_audit_verdicts=list(data.get("alignment_audit_verdicts", []) or []),
            tradability_audit_verdicts=list(data.get("tradability_audit_verdicts", []) or []),
            execution_cost_verdicts=list(data.get("execution_cost_verdicts", []) or []),
            audit_issue_codes=list(data.get("audit_issue_codes", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class FactorShadowEvidenceDateResult:
    schema_version: str = FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION
    result_id: str = ""
    as_of: str = ""
    candidate_count: int = 0
    production_factor_count: int = 0
    used_factor_count: int = 0
    scored_candidate_count: int = 0
    average_factor_coverage_ratio: float | None = None
    official_top_symbols: list[str] = field(default_factory=list)
    shadow_top_symbols: list[str] = field(default_factory=list)
    overlap_top_symbols: list[str] = field(default_factory=list)
    top_n_overlap_ratio: float | None = None
    average_abs_rank_delta: float | None = None
    max_abs_rank_delta: int | None = None
    large_rank_delta_symbols: list[str] = field(default_factory=list)
    shadow_report_id: str | None = None
    audit_snapshot: FactorAuditEvidenceSnapshot = field(
        default_factory=lambda: FactorAuditEvidenceSnapshot(as_of="1970-01-01")
    )
    warning_codes: list[str] = field(default_factory=list)
    status: str = EVIDENCE_STATUS_WARN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)
        self.result_id = _non_empty_str(self.result_id, "result_id")
        self.as_of = _sorted_iso_dates([self.as_of])[0]
        for field_name in [
            "candidate_count",
            "production_factor_count",
            "used_factor_count",
            "scored_candidate_count",
        ]:
            setattr(self, field_name, _non_negative_int(getattr(self, field_name), field_name))
        self.average_factor_coverage_ratio = _unit_float_or_none(
            self.average_factor_coverage_ratio,
            "average_factor_coverage_ratio",
        )
        self.official_top_symbols = [str(symbol) for symbol in self.official_top_symbols]
        self.shadow_top_symbols = [str(symbol) for symbol in self.shadow_top_symbols]
        self.overlap_top_symbols = sorted(str(symbol) for symbol in self.overlap_top_symbols)
        self.top_n_overlap_ratio = _unit_float_or_none(self.top_n_overlap_ratio, "top_n_overlap_ratio")
        self.average_abs_rank_delta = _non_negative_float_or_none(
            self.average_abs_rank_delta,
            "average_abs_rank_delta",
        )
        self.max_abs_rank_delta = None if self.max_abs_rank_delta is None else _non_negative_int(
            self.max_abs_rank_delta,
            "max_abs_rank_delta",
        )
        self.large_rank_delta_symbols = _ordered_unique(self.large_rank_delta_symbols)
        self.shadow_report_id = _optional_str(self.shadow_report_id)
        if not isinstance(self.audit_snapshot, FactorAuditEvidenceSnapshot):
            self.audit_snapshot = FactorAuditEvidenceSnapshot.from_dict(self.audit_snapshot)
        self.warning_codes = _ordered_unique(self.warning_codes)
        self.status = _non_empty_str(self.status, "status")
        _validate_status(self.status)
        self.metadata = _metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "as_of": self.as_of,
            "candidate_count": self.candidate_count,
            "production_factor_count": self.production_factor_count,
            "used_factor_count": self.used_factor_count,
            "scored_candidate_count": self.scored_candidate_count,
            "average_factor_coverage_ratio": self.average_factor_coverage_ratio,
            "official_top_symbols": list(self.official_top_symbols),
            "shadow_top_symbols": list(self.shadow_top_symbols),
            "overlap_top_symbols": list(self.overlap_top_symbols),
            "top_n_overlap_ratio": self.top_n_overlap_ratio,
            "average_abs_rank_delta": self.average_abs_rank_delta,
            "max_abs_rank_delta": self.max_abs_rank_delta,
            "large_rank_delta_symbols": list(self.large_rank_delta_symbols),
            "shadow_report_id": self.shadow_report_id,
            "audit_snapshot": self.audit_snapshot.to_dict(),
            "warning_codes": list(self.warning_codes),
            "status": self.status,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorShadowEvidenceDateResult":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)),
            result_id=str(data.get("result_id", "")),
            as_of=str(data.get("as_of", "")),
            candidate_count=int(data.get("candidate_count", 0)),
            production_factor_count=int(data.get("production_factor_count", 0)),
            used_factor_count=int(data.get("used_factor_count", 0)),
            scored_candidate_count=int(data.get("scored_candidate_count", 0)),
            average_factor_coverage_ratio=data.get("average_factor_coverage_ratio"),
            official_top_symbols=list(data.get("official_top_symbols", []) or []),
            shadow_top_symbols=list(data.get("shadow_top_symbols", []) or []),
            overlap_top_symbols=list(data.get("overlap_top_symbols", []) or []),
            top_n_overlap_ratio=data.get("top_n_overlap_ratio"),
            average_abs_rank_delta=data.get("average_abs_rank_delta"),
            max_abs_rank_delta=data.get("max_abs_rank_delta"),
            large_rank_delta_symbols=list(data.get("large_rank_delta_symbols", []) or []),
            shadow_report_id=data.get("shadow_report_id"),
            audit_snapshot=FactorAuditEvidenceSnapshot.from_dict(
                data.get("audit_snapshot", {"as_of": data.get("as_of", "1970-01-01")})
            ),
            warning_codes=list(data.get("warning_codes", []) or []),
            status=str(data.get("status", EVIDENCE_STATUS_WARN)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class MultiDateFactorEvidenceReport:
    schema_version: str = FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION
    report_id: str = ""
    generated_at: str = ""
    config: FactorEvidenceCollectionConfig = field(default_factory=FactorEvidenceCollectionConfig)
    observation_days: int = 0
    start_date: str | None = None
    end_date: str | None = None
    average_top_n_overlap_ratio: float | None = None
    min_top_n_overlap_ratio: float | None = None
    average_factor_coverage_ratio: float | None = None
    average_abs_rank_delta: float | None = None
    max_abs_rank_delta: int | None = None
    audit_blocker_days: int = 0
    alignment_fail_days: int = 0
    tradability_fail_days: int = 0
    execution_cost_warn_days: int = 0
    date_results: list[FactorShadowEvidenceDateResult] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)
    status: str = EVIDENCE_STATUS_WARN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)
        self.report_id = _non_empty_str(self.report_id, "report_id")
        self.generated_at = _non_empty_str(self.generated_at, "generated_at")
        if not isinstance(self.config, FactorEvidenceCollectionConfig):
            self.config = FactorEvidenceCollectionConfig.from_dict(self.config)
        self.observation_days = _non_negative_int(self.observation_days, "observation_days")
        self.start_date = _optional_str(self.start_date)
        self.end_date = _optional_str(self.end_date)
        for field_name in [
            "average_top_n_overlap_ratio",
            "min_top_n_overlap_ratio",
            "average_factor_coverage_ratio",
        ]:
            setattr(self, field_name, _unit_float_or_none(getattr(self, field_name), field_name))
        self.average_abs_rank_delta = _non_negative_float_or_none(
            self.average_abs_rank_delta,
            "average_abs_rank_delta",
        )
        self.max_abs_rank_delta = None if self.max_abs_rank_delta is None else _non_negative_int(
            self.max_abs_rank_delta,
            "max_abs_rank_delta",
        )
        for field_name in [
            "audit_blocker_days",
            "alignment_fail_days",
            "tradability_fail_days",
            "execution_cost_warn_days",
        ]:
            setattr(self, field_name, _non_negative_int(getattr(self, field_name), field_name))
        self.date_results = [
            result if isinstance(result, FactorShadowEvidenceDateResult)
            else FactorShadowEvidenceDateResult.from_dict(result)
            for result in self.date_results
        ]
        self.date_results = sorted(self.date_results, key=lambda result: result.as_of)
        self.warning_codes = _ordered_unique(self.warning_codes)
        self.status = _non_empty_str(self.status, "status")
        _validate_status(self.status)
        self.metadata = _metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "config": self.config.to_dict(),
            "observation_days": self.observation_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "average_top_n_overlap_ratio": self.average_top_n_overlap_ratio,
            "min_top_n_overlap_ratio": self.min_top_n_overlap_ratio,
            "average_factor_coverage_ratio": self.average_factor_coverage_ratio,
            "average_abs_rank_delta": self.average_abs_rank_delta,
            "max_abs_rank_delta": self.max_abs_rank_delta,
            "audit_blocker_days": self.audit_blocker_days,
            "alignment_fail_days": self.alignment_fail_days,
            "tradability_fail_days": self.tradability_fail_days,
            "execution_cost_warn_days": self.execution_cost_warn_days,
            "date_results": [result.to_dict() for result in self.date_results],
            "warning_codes": list(self.warning_codes),
            "status": self.status,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultiDateFactorEvidenceReport":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION)),
            report_id=str(data.get("report_id", "")),
            generated_at=str(data.get("generated_at", "")),
            config=FactorEvidenceCollectionConfig.from_dict(data.get("config", {})),
            observation_days=int(data.get("observation_days", 0)),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            average_top_n_overlap_ratio=data.get("average_top_n_overlap_ratio"),
            min_top_n_overlap_ratio=data.get("min_top_n_overlap_ratio"),
            average_factor_coverage_ratio=data.get("average_factor_coverage_ratio"),
            average_abs_rank_delta=data.get("average_abs_rank_delta"),
            max_abs_rank_delta=data.get("max_abs_rank_delta"),
            audit_blocker_days=int(data.get("audit_blocker_days", 0)),
            alignment_fail_days=int(data.get("alignment_fail_days", 0)),
            tradability_fail_days=int(data.get("tradability_fail_days", 0)),
            execution_cost_warn_days=int(data.get("execution_cost_warn_days", 0)),
            date_results=[
                FactorShadowEvidenceDateResult.from_dict(result)
                for result in list(data.get("date_results", []) or [])
                if isinstance(result, Mapping)
            ],
            warning_codes=list(data.get("warning_codes", []) or []),
            status=str(data.get("status", EVIDENCE_STATUS_WARN)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_evidence_collection_config_id(config: FactorEvidenceCollectionConfig) -> str:
    return f"factor-evidence-config-{_short_hash([config.as_of_dates, config.top_n, config.min_observation_days, config.min_average_factor_coverage, config.min_top_n_overlap_ratio, config.max_average_abs_rank_delta, config.require_library_audit_no_blocker, config.require_alignment_audit_pass, config.require_tradability_audit_pass, config.require_execution_cost_review])}"


def make_evidence_date_result_id(*, as_of: str, candidate_symbols: Sequence[str]) -> str:
    return f"factor-evidence-date-{_slug(as_of)}-{_short_hash([as_of, sorted(str(symbol) for symbol in candidate_symbols)])}"


def make_multi_date_evidence_report_id(*, generated_at: str, as_of_dates: Sequence[str]) -> str:
    return f"factor-evidence-report-{_slug(generated_at)}-{_short_hash([generated_at, sorted(str(date) for date in as_of_dates)])}"


def load_json_file_safe(path: str | Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None or not str(path).strip():
        return None, ["missing_json_path"]
    resolved = Path(path)
    if not resolved.exists():
        return None, [f"missing_json_file:{resolved}"]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"malformed_json_file:{resolved}:{exc}"]
    if not isinstance(payload, Mapping):
        return None, [f"json_file_not_object:{resolved}"]
    return dict(payload), []


def load_jsonl_file_safe(path: str | Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    if path is None or not str(path).strip():
        return [], ["missing_jsonl_path"]
    resolved = Path(path)
    if not resolved.exists():
        return [], [f"missing_jsonl_file:{resolved}"]
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"malformed_jsonl_file:{resolved}:{exc}"]
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            warnings.append(f"malformed_jsonl_line:{resolved}:{line_number}:{exc.msg}")
            continue
        if not isinstance(payload, Mapping):
            warnings.append(f"jsonl_line_not_object:{resolved}:{line_number}")
            continue
        rows.append(dict(payload))
    return rows, warnings


def load_factor_matrices_from_paths(paths: Sequence[str | Path]) -> tuple[list[FactorMatrix], list[str]]:
    matrices: list[FactorMatrix] = []
    warnings: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            warnings.append(f"missing_factor_matrix_file:{path}")
            continue
        payloads: list[dict[str, Any]]
        if path.suffix.lower() == ".jsonl":
            payloads, row_warnings = load_jsonl_file_safe(path)
            warnings.extend(row_warnings)
        else:
            payload, json_warnings = load_json_file_safe(path)
            warnings.extend(json_warnings)
            if payload is None:
                continue
            if isinstance(payload.get("factor_matrices"), list):
                payloads = [
                    dict(item)
                    for item in payload["factor_matrices"]
                    if isinstance(item, Mapping)
                ]
            else:
                payloads = [payload]
        for index, payload in enumerate(payloads):
            try:
                matrices.append(FactorMatrix.from_dict(payload))
            except (TypeError, ValueError) as exc:
                warnings.append(f"malformed_factor_matrix:{path}:{index}:{exc}")
    return sorted(matrices, key=lambda matrix: matrix.matrix_id), warnings


def load_production_library_safe(path: str | Path | None) -> tuple[ProductionFactorLibrary | None, list[str]]:
    payload, warnings = load_json_file_safe(path)
    if payload is None:
        return None, warnings
    try:
        return ProductionFactorLibrary.from_dict(payload), warnings
    except (TypeError, ValueError) as exc:
        return None, [*warnings, f"malformed_production_library:{path}:{exc}"]


def _payloads_from_paths(paths: Sequence[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in paths:
        resolved = Path(path)
        if resolved.suffix.lower() == ".jsonl":
            rows, row_warnings = load_jsonl_file_safe(resolved)
            payloads.extend(rows)
            warnings.extend(row_warnings)
        else:
            payload, json_warnings = load_json_file_safe(resolved)
            warnings.extend(json_warnings)
            if payload is not None:
                payloads.append(payload)
    return payloads, warnings


def _int_from_payload(payload: Mapping[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
            return len(value)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _verdict_from_payload(payload: Mapping[str, Any]) -> str | None:
    for key in ("verdict", "status", "audit_status", "result"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return None


def _issue_codes_from_payload(payload: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    raw_codes = payload.get("issue_codes")
    if isinstance(raw_codes, Sequence) and not isinstance(raw_codes, (str, bytes, bytearray)):
        codes.extend(str(code).strip() for code in raw_codes if str(code).strip())
    raw_issues = payload.get("issues")
    if isinstance(raw_issues, Sequence) and not isinstance(raw_issues, (str, bytes, bytearray)):
        for issue in raw_issues:
            if isinstance(issue, Mapping):
                code = issue.get("issue_code") or issue.get("code")
                if code is not None and str(code).strip():
                    codes.append(str(code).strip())
    return _ordered_unique(codes)


def build_audit_evidence_snapshot(
    *,
    as_of: str,
    production_library: ProductionFactorLibrary | None,
    library_audit_payload: Mapping[str, Any] | None = None,
    alignment_audit_payloads: Sequence[Mapping[str, Any]] | None = None,
    tradability_audit_payloads: Sequence[Mapping[str, Any]] | None = None,
    execution_cost_payloads: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorAuditEvidenceSnapshot:
    alignment_payloads = list(alignment_audit_payloads or [])
    tradability_payloads = list(tradability_audit_payloads or [])
    execution_payloads = list(execution_cost_payloads or [])
    issue_codes: list[str] = []
    for payload in [library_audit_payload, *alignment_payloads, *tradability_payloads, *execution_payloads]:
        if isinstance(payload, Mapping):
            issue_codes.extend(_issue_codes_from_payload(payload))

    return FactorAuditEvidenceSnapshot(
        as_of=as_of,
        library_exists=production_library is not None,
        production_factor_count=(
            sum(1 for entry in production_library.entries if entry.status == FACTOR_STATUS_PRODUCTION)
            if production_library is not None
            else 0
        ),
        library_audit_verdict=(
            _verdict_from_payload(library_audit_payload)
            if isinstance(library_audit_payload, Mapping)
            else None
        ),
        library_blocker_count=(
            _int_from_payload(
                library_audit_payload,
                ["blocker_count", "library_blocker_count", "blocked_factor_count", "blocked_factor_ids"],
            )
            if isinstance(library_audit_payload, Mapping)
            else 0
        ),
        library_warning_count=(
            _int_from_payload(library_audit_payload, ["warning_count", "warnings", "warning_codes"])
            if isinstance(library_audit_payload, Mapping)
            else 0
        ),
        alignment_audit_verdicts=[
            verdict for verdict in (_verdict_from_payload(payload) for payload in alignment_payloads) if verdict
        ],
        tradability_audit_verdicts=[
            verdict for verdict in (_verdict_from_payload(payload) for payload in tradability_payloads) if verdict
        ],
        execution_cost_verdicts=[
            verdict for verdict in (_verdict_from_payload(payload) for payload in execution_payloads) if verdict
        ],
        audit_issue_codes=issue_codes,
        metadata=_metadata(metadata),
    )


def _status_from_warnings(warnings: set[str], fail: bool, insufficient: bool) -> str:
    if insufficient:
        return EVIDENCE_STATUS_INSUFFICIENT_DATA
    if fail:
        return EVIDENCE_STATUS_FAIL
    if warnings:
        return EVIDENCE_STATUS_WARN
    return EVIDENCE_STATUS_OK


def _rank_delta_metrics(report: Any, threshold: float | None) -> tuple[float | None, int | None, list[str]]:
    deltas = [
        (score.symbol, abs(int(score.rank_delta)))
        for score in report.candidate_scores
        if score.rank_delta is not None
    ]
    if not deltas:
        return None, None, []
    average = sum(delta for _symbol, delta in deltas) / len(deltas)
    max_delta = max(delta for _symbol, delta in deltas)
    if threshold is not None:
        large = [symbol for symbol, delta in deltas if delta > threshold]
    else:
        large = [
            symbol
            for symbol, _delta in sorted(deltas, key=lambda item: (-item[1], item[0]))[:5]
        ]
    return average, max_delta, sorted(large)


def collect_shadow_evidence_for_date(
    *,
    date_input: FactorEvidenceDateInput,
    config: FactorEvidenceCollectionConfig,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorShadowEvidenceDateResult:
    candidates = [dict(candidate) for candidate in date_input.candidates]
    library, library_warnings = load_production_library_safe(date_input.production_library_path)
    matrices, matrix_warnings = load_factor_matrices_from_paths(date_input.factor_matrix_paths)
    library_audit_payload, library_audit_warnings = load_json_file_safe(date_input.library_audit_path)
    alignment_payloads, alignment_warnings = _payloads_from_paths(date_input.alignment_audit_paths)
    tradability_payloads, tradability_warnings = _payloads_from_paths(date_input.tradability_audit_paths)
    execution_payloads, execution_warnings = _payloads_from_paths(date_input.execution_cost_report_paths)

    audit_snapshot = build_audit_evidence_snapshot(
        as_of=date_input.as_of,
        production_library=library,
        library_audit_payload=library_audit_payload,
        alignment_audit_payloads=alignment_payloads,
        tradability_audit_payloads=tradability_payloads,
        execution_cost_payloads=execution_payloads,
        metadata={
            "loader_warnings": sorted(
                library_warnings
                + matrix_warnings
                + library_audit_warnings
                + alignment_warnings
                + tradability_warnings
                + execution_warnings
            )
        },
    )

    shadow_config = ShadowScoringConfig(
        as_of=date_input.as_of,
        top_n=config.top_n,
        min_factor_coverage_ratio=config.min_average_factor_coverage,
        metadata={"evidence_config_id": config.config_id},
    )
    shadow_report = build_shadow_scoring_comparison_report(
        candidates=candidates,
        library=library,
        factor_matrices=matrices,
        audit_report=library_audit_payload,
        config=shadow_config,
        generated_at=generated_at,
        metadata={"evidence_collection": True},
    )

    average_abs_rank_delta, max_abs_rank_delta, large_symbols = _rank_delta_metrics(
        shadow_report,
        config.max_average_abs_rank_delta,
    )
    warning_codes = set(shadow_report.warning_codes)
    fail = False
    insufficient = False

    if library is None:
        warning_codes.add(EVIDENCE_ISSUE_MISSING_PRODUCTION_LIBRARY)
    if not matrices:
        warning_codes.add(EVIDENCE_ISSUE_MISSING_FACTOR_MATRICES)
    if not candidates:
        warning_codes.add(EVIDENCE_ISSUE_MISSING_CANDIDATES)
        insufficient = True
    if (
        shadow_report.average_factor_coverage_ratio is not None
        and shadow_report.average_factor_coverage_ratio < config.min_average_factor_coverage
    ):
        warning_codes.add(EVIDENCE_ISSUE_LOW_FACTOR_COVERAGE)
    if (
        shadow_report.overlap_ratio is not None
        and shadow_report.overlap_ratio < config.min_top_n_overlap_ratio
    ):
        warning_codes.add(EVIDENCE_ISSUE_LOW_TOP_N_OVERLAP)
    if config.max_average_abs_rank_delta is not None and average_abs_rank_delta is not None:
        if average_abs_rank_delta > config.max_average_abs_rank_delta:
            warning_codes.add(EVIDENCE_ISSUE_LARGE_RANK_DRIFT)
    if audit_snapshot.library_blocker_count > 0:
        warning_codes.add(EVIDENCE_ISSUE_AUDIT_BLOCKER)
        fail = config.require_library_audit_no_blocker
    if "fail" in audit_snapshot.alignment_audit_verdicts:
        warning_codes.add(EVIDENCE_ISSUE_ALIGNMENT_AUDIT_FAIL)
        fail = fail or config.require_alignment_audit_pass
    if "fail" in audit_snapshot.tradability_audit_verdicts:
        warning_codes.add(EVIDENCE_ISSUE_TRADABILITY_AUDIT_FAIL)
        fail = fail or config.require_tradability_audit_pass
    if config.require_execution_cost_review and (
        "warn" in audit_snapshot.execution_cost_verdicts
        or "fail" in audit_snapshot.execution_cost_verdicts
    ):
        warning_codes.add(EVIDENCE_ISSUE_EXECUTION_COST_WARN)

    return FactorShadowEvidenceDateResult(
        result_id=make_evidence_date_result_id(
            as_of=date_input.as_of,
            candidate_symbols=_candidate_symbols(candidates),
        ),
        as_of=date_input.as_of,
        candidate_count=shadow_report.candidate_count,
        production_factor_count=shadow_report.production_factor_count,
        used_factor_count=shadow_report.used_factor_count,
        scored_candidate_count=shadow_report.scored_candidate_count,
        average_factor_coverage_ratio=shadow_report.average_factor_coverage_ratio,
        official_top_symbols=shadow_report.official_top_symbols,
        shadow_top_symbols=shadow_report.shadow_top_symbols,
        overlap_top_symbols=shadow_report.overlap_top_symbols,
        top_n_overlap_ratio=shadow_report.overlap_ratio,
        average_abs_rank_delta=average_abs_rank_delta,
        max_abs_rank_delta=max_abs_rank_delta,
        large_rank_delta_symbols=large_symbols,
        shadow_report_id=shadow_report.report_id,
        audit_snapshot=audit_snapshot,
        warning_codes=list(warning_codes),
        status=_status_from_warnings(warning_codes, fail=fail, insufficient=insufficient),
        metadata={
            **_metadata(metadata),
            "factor_shadow_evidence_schema_version": FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION,
            "factor_evidence_dashboard_schema_version": FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION,
            "non_runtime_impact": True,
            "no_official_score_change": True,
            "no_portfolio_change": True,
            "date_input_metadata": dict(_json_safe(date_input.metadata)),
        },
    )


def _average(values: Sequence[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def build_multi_date_factor_evidence_report(
    *,
    date_inputs: Sequence[FactorEvidenceDateInput],
    config: FactorEvidenceCollectionConfig,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> MultiDateFactorEvidenceReport:
    sorted_inputs = sorted(date_inputs, key=lambda item: item.as_of)
    date_results = [
        collect_shadow_evidence_for_date(
            date_input=date_input,
            config=config,
            generated_at=generated_at,
            metadata=metadata,
        )
        for date_input in sorted_inputs
    ]
    observation_days = len(date_results)
    overlap_values = [result.top_n_overlap_ratio for result in date_results]
    coverage_values = [result.average_factor_coverage_ratio for result in date_results]
    rank_values = [result.average_abs_rank_delta for result in date_results]
    max_rank_values = [result.max_abs_rank_delta for result in date_results if result.max_abs_rank_delta is not None]
    warning_codes: set[str] = set()
    for result in date_results:
        warning_codes.update(result.warning_codes)

    audit_blocker_days = sum(1 for result in date_results if result.audit_snapshot.library_blocker_count > 0)
    alignment_fail_days = sum(1 for result in date_results if "fail" in result.audit_snapshot.alignment_audit_verdicts)
    tradability_fail_days = sum(1 for result in date_results if "fail" in result.audit_snapshot.tradability_audit_verdicts)
    execution_cost_warn_days = sum(
        1
        for result in date_results
        if "warn" in result.audit_snapshot.execution_cost_verdicts
        or "fail" in result.audit_snapshot.execution_cost_verdicts
    )

    fail = any(result.status == EVIDENCE_STATUS_FAIL for result in date_results)
    insufficient = observation_days < config.min_observation_days
    if insufficient:
        warning_codes.add(EVIDENCE_ISSUE_INSUFFICIENT_OBSERVATION_DAYS)
    average_overlap = _average(overlap_values)
    average_coverage = _average(coverage_values)
    average_rank_delta = _average(rank_values)
    if average_overlap is not None and average_overlap < config.min_top_n_overlap_ratio:
        warning_codes.add(EVIDENCE_ISSUE_LOW_TOP_N_OVERLAP)
    if average_coverage is not None and average_coverage < config.min_average_factor_coverage:
        warning_codes.add(EVIDENCE_ISSUE_LOW_FACTOR_COVERAGE)
    if config.max_average_abs_rank_delta is not None and average_rank_delta is not None:
        if average_rank_delta > config.max_average_abs_rank_delta:
            warning_codes.add(EVIDENCE_ISSUE_LARGE_RANK_DRIFT)

    report_id = make_multi_date_evidence_report_id(
        generated_at=generated_at,
        as_of_dates=[result.as_of for result in date_results],
    )
    return MultiDateFactorEvidenceReport(
        report_id=report_id,
        generated_at=generated_at,
        config=config,
        observation_days=observation_days,
        start_date=date_results[0].as_of if date_results else None,
        end_date=date_results[-1].as_of if date_results else None,
        average_top_n_overlap_ratio=average_overlap,
        min_top_n_overlap_ratio=(
            min(value for value in overlap_values if value is not None)
            if any(value is not None for value in overlap_values)
            else None
        ),
        average_factor_coverage_ratio=average_coverage,
        average_abs_rank_delta=average_rank_delta,
        max_abs_rank_delta=max(max_rank_values) if max_rank_values else None,
        audit_blocker_days=audit_blocker_days,
        alignment_fail_days=alignment_fail_days,
        tradability_fail_days=tradability_fail_days,
        execution_cost_warn_days=execution_cost_warn_days,
        date_results=date_results,
        warning_codes=list(warning_codes),
        status=_status_from_warnings(warning_codes, fail=fail, insufficient=insufficient),
        metadata={
            **_metadata(metadata),
            "factor_shadow_evidence_schema_version": FACTOR_SHADOW_EVIDENCE_SCHEMA_VERSION,
            "factor_evidence_dashboard_schema_version": FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION,
            "non_runtime_impact": True,
            "no_official_score_change": True,
            "no_portfolio_change": True,
        },
    )


def _escape_pipe(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_optional_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def render_multi_date_evidence_markdown(report: MultiDateFactorEvidenceReport) -> str:
    lines = [
        f"# Multi-Date Factor Shadow Evidence: {report.report_id}",
        "",
        f"Generated at: `{_escape_pipe(report.generated_at)}`",
        f"Status: `{_escape_pipe(report.status)}`",
        "",
        "## Observation Window",
        "",
        f"- Start date: `{_escape_pipe(report.start_date or '')}`",
        f"- End date: `{_escape_pipe(report.end_date or '')}`",
        f"- Observation days: `{report.observation_days}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Average top-N overlap ratio | `{_format_optional_float(report.average_top_n_overlap_ratio)}` |",
        f"| Minimum top-N overlap ratio | `{_format_optional_float(report.min_top_n_overlap_ratio)}` |",
        f"| Average factor coverage ratio | `{_format_optional_float(report.average_factor_coverage_ratio)}` |",
        f"| Average absolute rank delta | `{_format_optional_float(report.average_abs_rank_delta)}` |",
        f"| Maximum absolute rank delta | `{report.max_abs_rank_delta if report.max_abs_rank_delta is not None else ''}` |",
        "",
        "## Date-Level Summary",
        "",
        "| As of | Status | Candidates | Used Factors | Top-N Overlap | Coverage | Avg Abs Rank Delta | Warnings |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report.date_results:
        lines.append(
            "| "
            f"`{_escape_pipe(result.as_of)}` | "
            f"`{_escape_pipe(result.status)}` | "
            f"{result.candidate_count} | "
            f"{result.used_factor_count} | "
            f"`{_format_optional_float(result.top_n_overlap_ratio)}` | "
            f"`{_format_optional_float(result.average_factor_coverage_ratio)}` | "
            f"`{_format_optional_float(result.average_abs_rank_delta)}` | "
            f"`{_escape_pipe(', '.join(result.warning_codes))}` |"
        )
    if not report.date_results:
        lines.append("|  |  |  |  |  |  |  | No date results. |")

    lines.extend(
        [
            "",
            "## Audit Blocker/Fail Summary",
            "",
            "| Field | Days |",
            "| --- | ---: |",
            f"| Library audit blocker days | {report.audit_blocker_days} |",
            f"| Alignment audit fail days | {report.alignment_fail_days} |",
            f"| Tradability audit fail days | {report.tradability_fail_days} |",
            f"| Execution cost warn/fail days | {report.execution_cost_warn_days} |",
            "",
            "## Large Rank Drift Summary",
            "",
            "| As of | Symbols |",
            "| --- | --- |",
        ]
    )
    drift_rows = [result for result in report.date_results if result.large_rank_delta_symbols]
    if drift_rows:
        for result in drift_rows:
            lines.append(
                f"| `{_escape_pipe(result.as_of)}` | `{_escape_pipe(', '.join(result.large_rank_delta_symbols))}` |"
            )
    else:
        lines.append("|  | None |")

    lines.extend(["", "## Warnings", ""])
    if report.warning_codes:
        lines.extend([f"- `{_escape_pipe(code)}`" for code in report.warning_codes])
    else:
        lines.append("- None")
    lines.extend(["", "## Non-Runtime Impact", "", NON_RUNTIME_IMPACT_NOTE, ""])
    return "\n".join(lines)


def build_factor_evidence_dashboard_payload(report: MultiDateFactorEvidenceReport) -> dict[str, Any]:
    payload = {
        "schema_version": FACTOR_EVIDENCE_DASHBOARD_SCHEMA_VERSION,
        "status": report.status,
        "generated_at": report.generated_at,
        "observation_days": report.observation_days,
        "start_date": report.start_date,
        "end_date": report.end_date,
        "average_top_n_overlap_ratio": report.average_top_n_overlap_ratio,
        "average_factor_coverage_ratio": report.average_factor_coverage_ratio,
        "average_abs_rank_delta": report.average_abs_rank_delta,
        "audit_blocker_days": report.audit_blocker_days,
        "alignment_fail_days": report.alignment_fail_days,
        "tradability_fail_days": report.tradability_fail_days,
        "execution_cost_warn_days": report.execution_cost_warn_days,
        "warning_codes": list(report.warning_codes),
        "date_summaries": [
            {
                "as_of": result.as_of,
                "status": result.status,
                "candidate_count": result.candidate_count,
                "production_factor_count": result.production_factor_count,
                "used_factor_count": result.used_factor_count,
                "scored_candidate_count": result.scored_candidate_count,
                "top_n_overlap_ratio": result.top_n_overlap_ratio,
                "average_factor_coverage_ratio": result.average_factor_coverage_ratio,
                "average_abs_rank_delta": result.average_abs_rank_delta,
                "max_abs_rank_delta": result.max_abs_rank_delta,
                "warning_codes": list(result.warning_codes),
            }
            for result in report.date_results
        ],
        "metadata": dict(_json_safe(report.metadata)),
    }
    json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return dict(_json_safe(payload))


__all__ = [
    "EVIDENCE_STATUS_OK",
    "EVIDENCE_STATUS_WARN",
    "EVIDENCE_STATUS_FAIL",
    "EVIDENCE_STATUS_INSUFFICIENT_DATA",
    "EVIDENCE_ISSUE_MISSING_PRODUCTION_LIBRARY",
    "EVIDENCE_ISSUE_MISSING_FACTOR_MATRICES",
    "EVIDENCE_ISSUE_MISSING_CANDIDATES",
    "EVIDENCE_ISSUE_LOW_FACTOR_COVERAGE",
    "EVIDENCE_ISSUE_LOW_TOP_N_OVERLAP",
    "EVIDENCE_ISSUE_LARGE_RANK_DRIFT",
    "EVIDENCE_ISSUE_AUDIT_BLOCKER",
    "EVIDENCE_ISSUE_ALIGNMENT_AUDIT_FAIL",
    "EVIDENCE_ISSUE_TRADABILITY_AUDIT_FAIL",
    "EVIDENCE_ISSUE_EXECUTION_COST_WARN",
    "EVIDENCE_ISSUE_INSUFFICIENT_OBSERVATION_DAYS",
    "DEFAULT_FACTOR_EVIDENCE_DIR",
    "DEFAULT_EVIDENCE_DATE_RESULTS_FILENAME",
    "DEFAULT_MULTI_DATE_EVIDENCE_REPORTS_FILENAME",
    "DEFAULT_EVIDENCE_DASHBOARD_FILENAME",
    "DEFAULT_EVIDENCE_MARKDOWN_FILENAME",
    "FactorEvidenceCollectionConfig",
    "FactorEvidenceDateInput",
    "FactorAuditEvidenceSnapshot",
    "FactorShadowEvidenceDateResult",
    "MultiDateFactorEvidenceReport",
    "make_evidence_collection_config_id",
    "make_evidence_date_result_id",
    "make_multi_date_evidence_report_id",
    "load_json_file_safe",
    "load_jsonl_file_safe",
    "load_factor_matrices_from_paths",
    "load_production_library_safe",
    "build_audit_evidence_snapshot",
    "collect_shadow_evidence_for_date",
    "build_multi_date_factor_evidence_report",
    "render_multi_date_evidence_markdown",
    "build_factor_evidence_dashboard_payload",
]
