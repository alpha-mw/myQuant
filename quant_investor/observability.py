"""Offline observability and audit bundle helpers for staged upgrade artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor import versioning
from quant_investor.versioning import AUDIT_BUNDLE_SCHEMA_VERSION, OBSERVABILITY_SCHEMA_VERSION


DEFAULT_OBSERVABILITY_DIR = Path("data/observability")
DEFAULT_AUDIT_BUNDLE_FILENAME = "audit_bundle.json"
DEFAULT_AUDIT_REPORT_FILENAME = "audit_report.md"
DEFAULT_DASHBOARD_PAYLOAD_FILENAME = "dashboard_payload.json"
DEFAULT_RUN_MANIFEST_FILENAME = "run_manifest.json"

ARTIFACT_TYPE_JSON = "json"
ARTIFACT_TYPE_JSONL = "jsonl"
ARTIFACT_TYPE_MARKDOWN = "markdown"
ARTIFACT_TYPE_DIRECTORY = "directory"
ARTIFACT_TYPE_UNKNOWN = "unknown"

HEALTH_STATUS_PASS = "pass"
HEALTH_STATUS_WARN = "warn"
HEALTH_STATUS_FAIL = "fail"
HEALTH_STATUS_UNKNOWN = "unknown"

_VALID_ARTIFACT_TYPES = {
    ARTIFACT_TYPE_JSON,
    ARTIFACT_TYPE_JSONL,
    ARTIFACT_TYPE_MARKDOWN,
    ARTIFACT_TYPE_DIRECTORY,
    ARTIFACT_TYPE_UNKNOWN,
}
_VALID_HEALTH_STATUSES = {
    HEALTH_STATUS_PASS,
    HEALTH_STATUS_WARN,
    HEALTH_STATUS_FAIL,
    HEALTH_STATUS_UNKNOWN,
}
_PHASE_SCRIPT_NAMES = [
    *(f"phase{phase}_quality_gate" for phase in range(1, 9)),
    "staged_upgrade_quality_gate",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return value


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    safe = _json_safe(value)
    try:
        json.dumps(safe, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise ValueError("metadata must contain only JSON-serializable values.") from exc
    return dict(safe)


def _short_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps([_json_safe(part) for part in parts], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _sorted_refs(refs: Sequence["ArtifactReference"]) -> list["ArtifactReference"]:
    return sorted(refs, key=lambda ref: (ref.name, ref.path))


def _status_from_counts(warning_count: int, failure_count: int) -> str:
    if failure_count > 0:
        return HEALTH_STATUS_FAIL
    if warning_count > 0:
        return HEALTH_STATUS_WARN
    return HEALTH_STATUS_PASS


def _non_negative_int_or_zero(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


def _derive_overall_status(module_summaries: Sequence["ModuleHealthSummary"]) -> str:
    if not module_summaries:
        return HEALTH_STATUS_UNKNOWN
    if any(summary.failure_count > 0 or summary.status == HEALTH_STATUS_FAIL for summary in module_summaries):
        return HEALTH_STATUS_FAIL
    if any(summary.warning_count > 0 or summary.status == HEALTH_STATUS_WARN for summary in module_summaries):
        return HEALTH_STATUS_WARN
    if all(summary.status == HEALTH_STATUS_PASS for summary in module_summaries):
        return HEALTH_STATUS_PASS
    return HEALTH_STATUS_UNKNOWN


def make_artifact_id(*, name: str, path: str) -> str:
    return f"artifact-{_short_hash([name, path])}"


def make_run_manifest_id(*, run_id: str, generated_at: str, artifact_ids: Sequence[str]) -> str:
    return f"run-manifest-{_short_hash([run_id, generated_at, sorted(str(value) for value in artifact_ids)])}"


def make_audit_bundle_id(*, manifest_id: str, generated_at: str) -> str:
    return f"audit-bundle-{_short_hash([manifest_id, generated_at])}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_records(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def read_json_file(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path)
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {resolved_path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed JSON in {resolved_path}: expected object.")
    return payload


def safe_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)


def _read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = Path(path)
    rows: list[dict[str, Any]] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL in {resolved_path} at line {line_number}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Malformed JSONL in {resolved_path} at line {line_number}: expected object.")
            rows.append(payload)
    return rows


def _artifact_type_for_path(path: Path) -> str:
    if path.is_dir():
        return ARTIFACT_TYPE_DIRECTORY
    suffix = path.suffix.lower()
    if suffix == ".json":
        return ARTIFACT_TYPE_JSON
    if suffix == ".jsonl":
        return ARTIFACT_TYPE_JSONL
    if suffix in {".md", ".markdown"}:
        return ARTIFACT_TYPE_MARKDOWN
    return ARTIFACT_TYPE_UNKNOWN


@dataclass
class ArtifactReference:
    schema_version: str = OBSERVABILITY_SCHEMA_VERSION
    artifact_id: str = ""
    name: str = ""
    artifact_type: str = ARTIFACT_TYPE_UNKNOWN
    path: str = ""
    exists: bool = False
    size_bytes: int | None = None
    sha256: str | None = None
    record_count: int | None = None
    schema_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact_type not in _VALID_ARTIFACT_TYPES:
            raise ValueError(f"artifact_type must be one of {sorted(_VALID_ARTIFACT_TYPES)}.")
        self.path = str(self.path)
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative when provided.")
        if self.record_count is not None and self.record_count < 0:
            raise ValueError("record_count must be non-negative when provided.")
        self.schema_hint = None if self.schema_hint is None else str(self.schema_hint)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactReference":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", OBSERVABILITY_SCHEMA_VERSION)),
            artifact_id=str(data.get("artifact_id", "")),
            name=str(data.get("name", "")),
            artifact_type=str(data.get("artifact_type", ARTIFACT_TYPE_UNKNOWN)),
            path=str(data.get("path", "")),
            exists=bool(data.get("exists", False)),
            size_bytes=data.get("size_bytes"),
            sha256=data.get("sha256"),
            record_count=data.get("record_count"),
            schema_hint=data.get("schema_hint"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class RunManifest:
    schema_version: str = OBSERVABILITY_SCHEMA_VERSION
    manifest_id: str = ""
    run_id: str = ""
    generated_at: str = ""
    as_of: str | None = None
    market: str | None = None
    universe_key: str | None = None
    universe_hash: str | None = None
    architecture_version: str | None = None
    schema_versions: dict[str, str] = field(default_factory=dict)
    artifact_refs: list[ArtifactReference] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.as_of = None if self.as_of is None else str(self.as_of)
        self.market = None if self.market is None else str(self.market)
        self.universe_key = None if self.universe_key is None else str(self.universe_key)
        self.universe_hash = None if self.universe_hash is None else str(self.universe_hash)
        self.architecture_version = None if self.architecture_version is None else str(self.architecture_version)
        self.schema_versions = {str(key): str(value) for key, value in sorted(self.schema_versions.items())}
        self.artifact_refs = _sorted_refs([
            ref if isinstance(ref, ArtifactReference) else ArtifactReference.from_dict(ref)
            for ref in self.artifact_refs
        ])
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(_json_safe(asdict(self)))
        payload["artifact_refs"] = [ref.to_dict() for ref in self.artifact_refs]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", OBSERVABILITY_SCHEMA_VERSION)),
            manifest_id=str(data.get("manifest_id", "")),
            run_id=str(data.get("run_id", "")),
            generated_at=str(data.get("generated_at", "")),
            as_of=data.get("as_of"),
            market=data.get("market"),
            universe_key=data.get("universe_key"),
            universe_hash=data.get("universe_hash"),
            architecture_version=data.get("architecture_version"),
            schema_versions=dict(data.get("schema_versions", {}) or {}),
            artifact_refs=[
                ArtifactReference.from_dict(ref)
                for ref in list(data.get("artifact_refs", []) or [])
                if isinstance(ref, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ModuleHealthSummary:
    schema_version: str = OBSERVABILITY_SCHEMA_VERSION
    module_name: str = ""
    status: str = HEALTH_STATUS_UNKNOWN
    artifact_count: int = 0
    record_count: int = 0
    warning_count: int = 0
    failure_count: int = 0
    key_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_HEALTH_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_HEALTH_STATUSES)}.")
        for field_name in ["artifact_count", "record_count", "warning_count", "failure_count"]:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        self.key_metrics = dict(_json_safe(self.key_metrics))
        self.warnings = sorted({str(warning) for warning in self.warnings})
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModuleHealthSummary":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", OBSERVABILITY_SCHEMA_VERSION)),
            module_name=str(data.get("module_name", "")),
            status=str(data.get("status", HEALTH_STATUS_UNKNOWN)),
            artifact_count=int(data.get("artifact_count", 0) or 0),
            record_count=int(data.get("record_count", 0) or 0),
            warning_count=int(data.get("warning_count", 0) or 0),
            failure_count=int(data.get("failure_count", 0) or 0),
            key_metrics=dict(data.get("key_metrics", {}) or {}),
            warnings=list(data.get("warnings", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class SystemObservabilitySummary:
    schema_version: str = OBSERVABILITY_SCHEMA_VERSION
    generated_at: str = ""
    overall_status: str = HEALTH_STATUS_UNKNOWN
    module_summaries: list[ModuleHealthSummary] = field(default_factory=list)
    total_artifacts: int = 0
    total_records: int = 0
    total_warnings: int = 0
    total_failures: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.overall_status not in _VALID_HEALTH_STATUSES:
            raise ValueError(f"overall_status must be one of {sorted(_VALID_HEALTH_STATUSES)}.")
        self.module_summaries = sorted(
            [
                summary if isinstance(summary, ModuleHealthSummary) else ModuleHealthSummary.from_dict(summary)
                for summary in self.module_summaries
            ],
            key=lambda summary: summary.module_name,
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "module_summaries": [summary.to_dict() for summary in self.module_summaries],
            "total_artifacts": self.total_artifacts,
            "total_records": self.total_records,
            "total_warnings": self.total_warnings,
            "total_failures": self.total_failures,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SystemObservabilitySummary":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", OBSERVABILITY_SCHEMA_VERSION)),
            generated_at=str(data.get("generated_at", "")),
            overall_status=str(data.get("overall_status", HEALTH_STATUS_UNKNOWN)),
            module_summaries=[
                ModuleHealthSummary.from_dict(summary)
                for summary in list(data.get("module_summaries", []) or [])
                if isinstance(summary, Mapping)
            ],
            total_artifacts=int(data.get("total_artifacts", 0) or 0),
            total_records=int(data.get("total_records", 0) or 0),
            total_warnings=int(data.get("total_warnings", 0) or 0),
            total_failures=int(data.get("total_failures", 0) or 0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class AuditBundle:
    schema_version: str = AUDIT_BUNDLE_SCHEMA_VERSION
    bundle_id: str = ""
    generated_at: str = ""
    run_manifest: RunManifest = field(default_factory=RunManifest)
    observability_summary: SystemObservabilitySummary = field(default_factory=SystemObservabilitySummary)
    dashboard_payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_manifest, RunManifest):
            self.run_manifest = RunManifest.from_dict(self.run_manifest)
        if not isinstance(self.observability_summary, SystemObservabilitySummary):
            self.observability_summary = SystemObservabilitySummary.from_dict(self.observability_summary)
        self.dashboard_payload = dict(_json_safe(self.dashboard_payload))
        json.dumps(self.dashboard_payload, ensure_ascii=False, sort_keys=True)
        self.warnings = sorted({str(warning) for warning in self.warnings})
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "generated_at": self.generated_at,
            "run_manifest": self.run_manifest.to_dict(),
            "observability_summary": self.observability_summary.to_dict(),
            "dashboard_payload": _json_safe(self.dashboard_payload),
            "warnings": list(self.warnings),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditBundle":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", AUDIT_BUNDLE_SCHEMA_VERSION)),
            bundle_id=str(data.get("bundle_id", "")),
            generated_at=str(data.get("generated_at", "")),
            run_manifest=RunManifest.from_dict(dict(data.get("run_manifest", {}) or {})),
            observability_summary=SystemObservabilitySummary.from_dict(
                dict(data.get("observability_summary", {}) or {})
            ),
            dashboard_payload=dict(data.get("dashboard_payload", {}) or {}),
            warnings=list(data.get("warnings", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def build_artifact_reference(
    *,
    name: str,
    path: str | Path,
    schema_hint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactReference:
    resolved_path = Path(path)
    artifact_type = _artifact_type_for_path(resolved_path)
    exists = resolved_path.exists()
    size_bytes = resolved_path.stat().st_size if exists and resolved_path.is_file() else None
    file_hash = sha256_file(resolved_path) if exists and resolved_path.is_file() else None
    record_count = None
    if exists and resolved_path.is_file():
        if artifact_type == ARTIFACT_TYPE_JSONL:
            record_count = count_jsonl_records(resolved_path)
        elif artifact_type == ARTIFACT_TYPE_JSON:
            record_count = 1
    path_text = str(resolved_path)
    return ArtifactReference(
        schema_version=OBSERVABILITY_SCHEMA_VERSION,
        artifact_id=make_artifact_id(name=name, path=path_text),
        name=name,
        artifact_type=artifact_type,
        path=path_text,
        exists=exists,
        size_bytes=size_bytes,
        sha256=file_hash,
        record_count=record_count,
        schema_hint=schema_hint,
        metadata=_coerce_metadata(metadata),
    )


def _add_expected_ref(
    refs: list[ArtifactReference],
    *,
    name: str,
    path: Path,
    schema_hint: str | None,
    module: str,
) -> None:
    refs.append(
        build_artifact_reference(
            name=name,
            path=path,
            schema_hint=schema_hint,
            metadata={"module": module},
        )
    )


def discover_phase_artifacts(
    *,
    outcome_ledger_dir: str | Path | None = None,
    calibration_v2_dir: str | Path | None = None,
    data_quality_dir: str | Path | None = None,
    risk_tensor_dir: str | Path | None = None,
    portfolio_optimizer_dir: str | Path | None = None,
    factor_library_dir: str | Path | None = None,
    docs_dir: str | Path | None = None,
    scripts_dir: str | Path | None = None,
) -> list[ArtifactReference]:
    refs: list[ArtifactReference] = []

    from quant_investor.bayesian.outcome_ledger import (
        DEFAULT_OUTCOME_LEDGER_DIR,
        DEFAULT_OUTCOMES_FILENAME,
        DEFAULT_PREDICTIONS_FILENAME,
    )
    outcome_root = Path(outcome_ledger_dir) if outcome_ledger_dir is not None else DEFAULT_OUTCOME_LEDGER_DIR
    _add_expected_ref(
        refs,
        name="outcome_ledger_predictions",
        path=outcome_root / DEFAULT_PREDICTIONS_FILENAME,
        schema_hint=getattr(versioning, "OUTCOME_LEDGER_SCHEMA_VERSION", None),
        module="outcome_ledger",
    )
    _add_expected_ref(
        refs,
        name="outcome_ledger_outcomes",
        path=outcome_root / DEFAULT_OUTCOMES_FILENAME,
        schema_hint=getattr(versioning, "OUTCOME_LEDGER_SCHEMA_VERSION", None),
        module="outcome_ledger",
    )

    from quant_investor.bayesian.calibration_v2 import (
        DEFAULT_CALIBRATION_MODEL_FILENAME,
        DEFAULT_CALIBRATION_REPORT_FILENAME,
        DEFAULT_CALIBRATION_V2_DIR,
    )
    calibration_root = Path(calibration_v2_dir) if calibration_v2_dir is not None else DEFAULT_CALIBRATION_V2_DIR
    _add_expected_ref(
        refs,
        name="calibration_v2_model",
        path=calibration_root / DEFAULT_CALIBRATION_MODEL_FILENAME,
        schema_hint=getattr(versioning, "CALIBRATION_V2_SCHEMA_VERSION", None),
        module="calibration_v2",
    )
    _add_expected_ref(
        refs,
        name="calibration_v2_report",
        path=calibration_root / DEFAULT_CALIBRATION_REPORT_FILENAME,
        schema_hint=getattr(versioning, "CALIBRATION_V2_SCHEMA_VERSION", None),
        module="calibration_v2",
    )

    try:
        from quant_investor.data_quality_contract import (
            DEFAULT_ASSESSMENTS_FILENAME,
            DEFAULT_DATA_QUALITY_CONTRACT_DIR,
            DEFAULT_SNAPSHOTS_FILENAME,
        )
    except ImportError:
        DEFAULT_DATA_QUALITY_CONTRACT_DIR = Path("data/data_quality_contract")
        DEFAULT_SNAPSHOTS_FILENAME = "snapshots.jsonl"
        DEFAULT_ASSESSMENTS_FILENAME = "assessments.jsonl"
    quality_root = Path(data_quality_dir) if data_quality_dir is not None else DEFAULT_DATA_QUALITY_CONTRACT_DIR
    _add_expected_ref(
        refs,
        name="data_quality_snapshots",
        path=quality_root / DEFAULT_SNAPSHOTS_FILENAME,
        schema_hint=getattr(versioning, "DATA_QUALITY_CONTRACT_SCHEMA_VERSION", None),
        module="data_quality",
    )
    _add_expected_ref(
        refs,
        name="data_quality_assessments",
        path=quality_root / DEFAULT_ASSESSMENTS_FILENAME,
        schema_hint=getattr(versioning, "DATA_QUALITY_CONTRACT_SCHEMA_VERSION", None),
        module="data_quality",
    )

    try:
        from quant_investor.risk_tensor import (
            DEFAULT_EXECUTION_REPORTS_FILENAME,
            DEFAULT_PORTFOLIO_TENSORS_FILENAME,
            DEFAULT_RISK_TENSOR_DIR,
            DEFAULT_SYMBOL_TENSORS_FILENAME,
        )
    except ImportError:
        DEFAULT_RISK_TENSOR_DIR = Path("data/risk_tensor")
        DEFAULT_SYMBOL_TENSORS_FILENAME = "symbol_tensors.jsonl"
        DEFAULT_PORTFOLIO_TENSORS_FILENAME = "portfolio_tensors.jsonl"
        DEFAULT_EXECUTION_REPORTS_FILENAME = "execution_reports.jsonl"
    risk_root = Path(risk_tensor_dir) if risk_tensor_dir is not None else DEFAULT_RISK_TENSOR_DIR
    _add_expected_ref(
        refs,
        name="risk_tensor_symbol_tensors",
        path=risk_root / DEFAULT_SYMBOL_TENSORS_FILENAME,
        schema_hint=getattr(versioning, "RISK_TENSOR_SCHEMA_VERSION", None),
        module="risk_tensor",
    )
    _add_expected_ref(
        refs,
        name="risk_tensor_portfolio_tensors",
        path=risk_root / DEFAULT_PORTFOLIO_TENSORS_FILENAME,
        schema_hint=getattr(versioning, "RISK_TENSOR_SCHEMA_VERSION", None),
        module="risk_tensor",
    )
    _add_expected_ref(
        refs,
        name="risk_tensor_execution_reports",
        path=risk_root / DEFAULT_EXECUTION_REPORTS_FILENAME,
        schema_hint=getattr(versioning, "RISK_TENSOR_SCHEMA_VERSION", None),
        module="risk_tensor",
    )

    try:
        from quant_investor.portfolio_optimizer import (
            DEFAULT_OPTIMIZED_PLANS_FILENAME,
            DEFAULT_PORTFOLIO_OPTIMIZER_DIR,
            DEFAULT_REBALANCE_RESULTS_FILENAME,
            DEFAULT_WALK_FORWARD_RESULTS_FILENAME,
        )
    except ImportError:
        DEFAULT_PORTFOLIO_OPTIMIZER_DIR = Path("data/portfolio_optimizer")
        DEFAULT_OPTIMIZED_PLANS_FILENAME = "optimized_plans.jsonl"
        DEFAULT_REBALANCE_RESULTS_FILENAME = "rebalance_results.jsonl"
        DEFAULT_WALK_FORWARD_RESULTS_FILENAME = "walk_forward_results.jsonl"
    optimizer_root = (
        Path(portfolio_optimizer_dir) if portfolio_optimizer_dir is not None else DEFAULT_PORTFOLIO_OPTIMIZER_DIR
    )
    _add_expected_ref(
        refs,
        name="portfolio_optimizer_plans",
        path=optimizer_root / DEFAULT_OPTIMIZED_PLANS_FILENAME,
        schema_hint=getattr(versioning, "PORTFOLIO_OPTIMIZER_SCHEMA_VERSION", None),
        module="portfolio_optimizer",
    )
    _add_expected_ref(
        refs,
        name="portfolio_optimizer_rebalance_results",
        path=optimizer_root / DEFAULT_REBALANCE_RESULTS_FILENAME,
        schema_hint=getattr(versioning, "PORTFOLIO_OPTIMIZER_SCHEMA_VERSION", None),
        module="portfolio_optimizer",
    )
    _add_expected_ref(
        refs,
        name="portfolio_optimizer_walk_forward_results",
        path=optimizer_root / DEFAULT_WALK_FORWARD_RESULTS_FILENAME,
        schema_hint=getattr(versioning, "PORTFOLIO_OPTIMIZER_SCHEMA_VERSION", None),
        module="portfolio_optimizer",
    )

    try:
        from quant_investor.factors.schema import (
            DEFAULT_FACTOR_ADMISSION_DECISIONS_FILENAME,
            DEFAULT_FACTOR_BACKTEST_RESULTS_FILENAME,
            DEFAULT_FACTOR_DEFINITIONS_FILENAME,
            DEFAULT_FACTOR_LIBRARY_DIR,
            DEFAULT_FACTOR_VALIDATION_REPORTS_FILENAME,
            DEFAULT_PRODUCTION_FACTORS_FILENAME,
        )
        from quant_investor.factors.store import (
            DEFAULT_FACTOR_GOVERNANCE_DASHBOARD_FILENAME,
            DEFAULT_FACTOR_CONTRIBUTION_REPORTS_FILENAME,
            DEFAULT_FACTOR_INCREMENTAL_DIR,
            DEFAULT_FACTOR_LIBRARY_AUDIT_DIR,
            DEFAULT_FACTOR_LIBRARY_AUDIT_REPORTS_FILENAME,
            DEFAULT_FACTOR_REDUNDANCY_REPORTS_FILENAME,
        )
    except ImportError:
        DEFAULT_FACTOR_LIBRARY_DIR = Path("data/factor_library")
        DEFAULT_FACTOR_DEFINITIONS_FILENAME = "factor_definitions.jsonl"
        DEFAULT_FACTOR_BACKTEST_RESULTS_FILENAME = "factor_backtest_results.jsonl"
        DEFAULT_FACTOR_VALIDATION_REPORTS_FILENAME = "factor_validation_reports.jsonl"
        DEFAULT_FACTOR_ADMISSION_DECISIONS_FILENAME = "factor_admission_decisions.jsonl"
        DEFAULT_PRODUCTION_FACTORS_FILENAME = "production_factors.json"
        DEFAULT_FACTOR_INCREMENTAL_DIR = Path("data/factor_library/incremental")
        DEFAULT_FACTOR_REDUNDANCY_REPORTS_FILENAME = "factor_redundancy_reports.jsonl"
        DEFAULT_FACTOR_CONTRIBUTION_REPORTS_FILENAME = "factor_contribution_reports.jsonl"
        DEFAULT_FACTOR_LIBRARY_AUDIT_DIR = Path("data/factor_library/audit")
        DEFAULT_FACTOR_LIBRARY_AUDIT_REPORTS_FILENAME = "factor_library_audit_reports.jsonl"
        DEFAULT_FACTOR_GOVERNANCE_DASHBOARD_FILENAME = "factor_governance_dashboard.json"
    factor_root = (
        Path(factor_library_dir) if factor_library_dir is not None else DEFAULT_FACTOR_LIBRARY_DIR
    )
    include_factor_refs = factor_library_dir is not None or factor_root.exists()
    if include_factor_refs:
        incremental_root = (
            factor_root / "incremental"
            if factor_library_dir is not None
            else DEFAULT_FACTOR_INCREMENTAL_DIR
        )
        audit_root = (
            factor_root / "audit"
            if factor_library_dir is not None
            else DEFAULT_FACTOR_LIBRARY_AUDIT_DIR
        )
        factor_schema = getattr(versioning, "FACTOR_GOVERNANCE_SCHEMA_VERSION", None)
        library_schema = getattr(versioning, "FACTOR_LIBRARY_SCHEMA_VERSION", None)
        incremental_schema = getattr(
            versioning,
            "FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION",
            None,
        )
        audit_schema = getattr(versioning, "FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION", None)
        for name, filename, schema_hint in [
            ("factor_definitions", DEFAULT_FACTOR_DEFINITIONS_FILENAME, factor_schema),
            ("factor_backtest_results", DEFAULT_FACTOR_BACKTEST_RESULTS_FILENAME, factor_schema),
            ("factor_validation_reports", DEFAULT_FACTOR_VALIDATION_REPORTS_FILENAME, factor_schema),
            ("factor_admission_decisions", DEFAULT_FACTOR_ADMISSION_DECISIONS_FILENAME, factor_schema),
            ("production_factors", DEFAULT_PRODUCTION_FACTORS_FILENAME, library_schema),
        ]:
            _add_expected_ref(
                refs,
                name=name,
                path=factor_root / filename,
                schema_hint=schema_hint,
                module="factor_governance",
            )
        _add_expected_ref(
            refs,
            name="factor_redundancy_reports",
            path=incremental_root / DEFAULT_FACTOR_REDUNDANCY_REPORTS_FILENAME,
            schema_hint=getattr(versioning, "FACTOR_CORRELATION_SCHEMA_VERSION", None),
            module="factor_governance",
        )
        _add_expected_ref(
            refs,
            name="factor_contribution_reports",
            path=incremental_root / DEFAULT_FACTOR_CONTRIBUTION_REPORTS_FILENAME,
            schema_hint=incremental_schema,
            module="factor_governance",
        )
        _add_expected_ref(
            refs,
            name="factor_library_audit_reports",
            path=audit_root / DEFAULT_FACTOR_LIBRARY_AUDIT_REPORTS_FILENAME,
            schema_hint=audit_schema,
            module="factor_governance",
        )
        _add_expected_ref(
            refs,
            name="factor_governance_dashboard",
            path=audit_root / DEFAULT_FACTOR_GOVERNANCE_DASHBOARD_FILENAME,
            schema_hint=audit_schema,
            module="factor_governance",
        )

    docs_root = Path(docs_dir) if docs_dir is not None else Path("docs")
    if docs_dir is not None or docs_root.exists():
        _add_expected_ref(
            refs,
            name="docs_system_upgrade_plan",
            path=docs_root / "system_upgrade_plan.md",
            schema_hint=None,
            module="docs_scripts",
        )

    scripts_root = Path(scripts_dir) if scripts_dir is not None else Path("scripts")
    if scripts_dir is not None or scripts_root.exists():
        for script_name in _PHASE_SCRIPT_NAMES:
            _add_expected_ref(
                refs,
                name=f"script_{script_name}",
                path=scripts_root / f"{script_name}.sh",
                schema_hint=None,
                module="docs_scripts",
            )

    return _sorted_refs(refs)


def _refs_for_module(artifact_refs: Sequence[ArtifactReference], module_name: str) -> list[ArtifactReference]:
    return [
        ref
        for ref in _sorted_refs(artifact_refs)
        if ref.metadata.get("module") == module_name or ref.name.startswith(module_name)
    ]


def _jsonl_count_or_failure(ref: ArtifactReference, warnings: list[str]) -> int:
    if not ref.exists:
        warnings.append(f"Missing optional artifact: {ref.name}")
        return 0
    if ref.artifact_type != ARTIFACT_TYPE_JSONL:
        return ref.record_count or 0
    try:
        return len(_read_jsonl_objects(ref.path))
    except ValueError as exc:
        warnings.append(str(exc))
        return -1


def _build_jsonl_summary(
    *,
    module_name: str,
    refs: Sequence[ArtifactReference],
    key_names: Mapping[str, str],
    extra_metrics: Mapping[str, Any] | None = None,
) -> ModuleHealthSummary:
    warnings: list[str] = []
    failure_count = 0
    metrics: dict[str, Any] = dict(extra_metrics or {})
    total_records = 0
    for ref in _sorted_refs(refs):
        count = _jsonl_count_or_failure(ref, warnings)
        if count < 0:
            failure_count += 1
            count = 0
        total_records += count
        metric_key = key_names.get(ref.name)
        if metric_key is not None:
            metrics[metric_key] = count
    warning_count = len(warnings)
    return ModuleHealthSummary(
        module_name=module_name,
        status=_status_from_counts(warning_count, failure_count),
        artifact_count=len(refs),
        record_count=total_records,
        warning_count=warning_count,
        failure_count=failure_count,
        key_metrics=metrics,
        warnings=warnings,
    )


def summarize_outcome_ledger_artifacts(artifact_refs: Sequence[ArtifactReference]) -> ModuleHealthSummary:
    refs = _refs_for_module(artifact_refs, "outcome_ledger")
    warnings: list[str] = []
    failure_count = 0
    prediction_records: list[dict[str, Any]] = []
    outcome_records: list[dict[str, Any]] = []
    for ref in refs:
        if not ref.exists:
            warnings.append(f"Missing optional artifact: {ref.name}")
            continue
        try:
            rows = _read_jsonl_objects(ref.path)
        except ValueError as exc:
            warnings.append(str(exc))
            failure_count += 1
            continue
        if ref.name == "outcome_ledger_predictions":
            prediction_records = rows
        elif ref.name == "outcome_ledger_outcomes":
            outcome_records = rows
    prediction_ids = {str(row.get("prediction_id")) for row in prediction_records if row.get("prediction_id")}
    resolved_prediction_ids = {str(row.get("prediction_id")) for row in outcome_records if row.get("prediction_id")}
    unresolved_estimate = len(prediction_ids - resolved_prediction_ids) if prediction_ids else None
    metrics = {
        "prediction_records": len(prediction_records),
        "outcome_records": len(outcome_records),
        "unresolved_estimate": unresolved_estimate,
    }
    warning_count = len(warnings)
    return ModuleHealthSummary(
        module_name="outcome_ledger",
        status=_status_from_counts(warning_count, failure_count),
        artifact_count=len(refs),
        record_count=len(prediction_records) + len(outcome_records),
        warning_count=warning_count,
        failure_count=failure_count,
        key_metrics=metrics,
        warnings=warnings,
    )


def summarize_calibration_artifacts(artifact_refs: Sequence[ArtifactReference]) -> ModuleHealthSummary:
    refs = _refs_for_module(artifact_refs, "calibration_v2")
    warnings: list[str] = []
    failure_count = 0
    model_exists = False
    report_exists = False
    curve_count: int | None = None
    total_records = 0
    for ref in refs:
        if not ref.exists:
            warnings.append(f"Missing optional artifact: {ref.name}")
            continue
        total_records += ref.record_count or 0
        try:
            payload = read_json_file(ref.path)
        except ValueError as exc:
            warnings.append(str(exc))
            failure_count += 1
            continue
        if ref.name == "calibration_v2_model":
            model_exists = True
            curves = payload.get("curves")
            curve_count = len(curves) if isinstance(curves, list) else 0
        elif ref.name == "calibration_v2_report":
            report_exists = True
    warning_count = len(warnings)
    return ModuleHealthSummary(
        module_name="calibration_v2",
        status=_status_from_counts(warning_count, failure_count),
        artifact_count=len(refs),
        record_count=total_records,
        warning_count=warning_count,
        failure_count=failure_count,
        key_metrics={
            "model_exists": model_exists,
            "report_exists": report_exists,
            "curve_count": curve_count,
        },
        warnings=warnings,
    )


def summarize_data_quality_artifacts(artifact_refs: Sequence[ArtifactReference]) -> ModuleHealthSummary:
    return _build_jsonl_summary(
        module_name="data_quality",
        refs=_refs_for_module(artifact_refs, "data_quality"),
        key_names={
            "data_quality_snapshots": "snapshot_records",
            "data_quality_assessments": "assessment_records",
        },
    )


def summarize_risk_tensor_artifacts(artifact_refs: Sequence[ArtifactReference]) -> ModuleHealthSummary:
    return _build_jsonl_summary(
        module_name="risk_tensor",
        refs=_refs_for_module(artifact_refs, "risk_tensor"),
        key_names={
            "risk_tensor_symbol_tensors": "symbol_tensor_records",
            "risk_tensor_portfolio_tensors": "portfolio_tensor_records",
            "risk_tensor_execution_reports": "execution_report_records",
        },
    )


def summarize_portfolio_optimizer_artifacts(artifact_refs: Sequence[ArtifactReference]) -> ModuleHealthSummary:
    return _build_jsonl_summary(
        module_name="portfolio_optimizer",
        refs=_refs_for_module(artifact_refs, "portfolio_optimizer"),
        key_names={
            "portfolio_optimizer_plans": "plan_records",
            "portfolio_optimizer_rebalance_results": "rebalance_result_records",
            "portfolio_optimizer_walk_forward_results": "walk_forward_result_records",
        },
    )


def summarize_factor_governance_artifacts(
    artifact_refs: Sequence[ArtifactReference],
) -> ModuleHealthSummary:
    refs = _refs_for_module(artifact_refs, "factor_governance")
    warnings: list[str] = []
    failure_count = 0
    metrics: dict[str, Any] = {
        "factor_definition_records": 0,
        "backtest_result_records": 0,
        "validation_report_records": 0,
        "admission_decision_records": 0,
        "production_factor_count": 0,
        "blocked_factor_count": 0,
        "shadow_only_factor_count": 0,
        "expired_factor_count": 0,
        "audit_verdict": HEALTH_STATUS_UNKNOWN,
        "audit_warning_count": 0,
        "audit_blocker_count": 0,
        "redundancy_report_records": 0,
        "contribution_report_records": 0,
        "audit_report_records": 0,
    }
    metric_names = {
        "factor_definitions": "factor_definition_records",
        "factor_backtest_results": "backtest_result_records",
        "factor_validation_reports": "validation_report_records",
        "factor_admission_decisions": "admission_decision_records",
        "factor_redundancy_reports": "redundancy_report_records",
        "factor_contribution_reports": "contribution_report_records",
        "factor_library_audit_reports": "audit_report_records",
    }
    total_records = 0
    for ref in refs:
        if not ref.exists:
            warnings.append(f"Missing optional factor artifact: {ref.name}")
            continue
        if ref.artifact_type == ARTIFACT_TYPE_JSONL:
            try:
                rows = _read_jsonl_objects(ref.path)
            except ValueError as exc:
                warnings.append(str(exc))
                failure_count += 1
                continue
            count = len(rows)
            total_records += count
            metric_name = metric_names.get(ref.name)
            if metric_name is not None:
                metrics[metric_name] = count
            if ref.name == "factor_library_audit_reports" and rows:
                latest_audit = rows[-1]
                blocked_ids = latest_audit.get("blocked_factor_ids", [])
                shadow_only_ids = latest_audit.get("shadow_only_factor_ids", [])
                metrics["audit_verdict"] = str(
                    latest_audit.get("verdict", HEALTH_STATUS_UNKNOWN) or HEALTH_STATUS_UNKNOWN
                )
                metrics["blocked_factor_count"] = (
                    len(blocked_ids) if isinstance(blocked_ids, list) else 0
                )
                metrics["shadow_only_factor_count"] = (
                    len(shadow_only_ids) if isinstance(shadow_only_ids, list) else 0
                )
                metrics["expired_factor_count"] = _non_negative_int_or_zero(
                    latest_audit.get("expired_factor_count", 0) or 0
                )
                metrics["audit_warning_count"] = _non_negative_int_or_zero(
                    latest_audit.get("warning_count", 0) or 0
                )
                metrics["audit_blocker_count"] = _non_negative_int_or_zero(
                    latest_audit.get("blocker_count", 0) or 0
                )
                if not metrics["production_factor_count"]:
                    metrics["production_factor_count"] = _non_negative_int_or_zero(
                        latest_audit.get("production_factor_count", 0) or 0
                    )
        elif ref.artifact_type == ARTIFACT_TYPE_JSON:
            try:
                payload = read_json_file(ref.path)
            except ValueError as exc:
                warnings.append(str(exc))
                failure_count += 1
                continue
            total_records += 1
            if ref.name == "production_factors":
                entries = payload.get("entries", [])
                metrics["production_factor_count"] = len(entries) if isinstance(entries, list) else 0
            elif ref.name == "factor_governance_dashboard":
                raw_counts = payload.get("counts", {})
                counts = dict(raw_counts) if isinstance(raw_counts, Mapping) else {}
                blocked_ids = payload.get("blocked_factor_ids", [])
                shadow_only_ids = payload.get("shadow_only_factor_ids", [])
                metrics["audit_verdict"] = str(
                    payload.get("verdict", HEALTH_STATUS_UNKNOWN) or HEALTH_STATUS_UNKNOWN
                )
                metrics["blocked_factor_count"] = _non_negative_int_or_zero(
                    counts.get(
                        "blocked_factor_count",
                        len(blocked_ids) if isinstance(blocked_ids, list) else 0,
                    )
                )
                metrics["shadow_only_factor_count"] = _non_negative_int_or_zero(
                    counts.get(
                        "shadow_only_factor_count",
                        len(shadow_only_ids) if isinstance(shadow_only_ids, list) else 0,
                    )
                )
                metrics["expired_factor_count"] = _non_negative_int_or_zero(
                    counts.get("expired_factor_count", metrics["expired_factor_count"])
                )
                metrics["audit_warning_count"] = _non_negative_int_or_zero(
                    counts.get("warning_count", metrics["audit_warning_count"])
                )
                metrics["audit_blocker_count"] = _non_negative_int_or_zero(
                    counts.get("blocker_count", metrics["audit_blocker_count"])
                )
                if not metrics["production_factor_count"]:
                    metrics["production_factor_count"] = _non_negative_int_or_zero(
                        counts.get("production_factor_count", 0)
                    )
        else:
            total_records += ref.record_count or 0
    warning_count = len(warnings)
    return ModuleHealthSummary(
        module_name="factor_governance",
        status=_status_from_counts(warning_count, failure_count),
        artifact_count=len(refs),
        record_count=total_records,
        warning_count=warning_count,
        failure_count=failure_count,
        key_metrics=metrics,
        warnings=warnings,
    )


def summarize_docs_and_scripts_artifacts(artifact_refs: Sequence[ArtifactReference]) -> ModuleHealthSummary:
    refs = _refs_for_module(artifact_refs, "docs_scripts")
    warnings: list[str] = []
    docs_count = 0
    script_count = 0
    missing_scripts: list[str] = []
    for ref in refs:
        if ref.name == "docs_system_upgrade_plan":
            if ref.exists:
                docs_count += 1
            else:
                warnings.append("Missing docs/system_upgrade_plan.md")
        elif ref.name.startswith("script_"):
            if ref.exists:
                script_count += 1
            else:
                missing_scripts.append(ref.name.replace("script_", ""))
    for script_name in sorted(missing_scripts):
        warnings.append(f"Missing phase quality gate script: {script_name}.sh")
    warning_count = len(warnings)
    return ModuleHealthSummary(
        module_name="docs_scripts",
        status=_status_from_counts(warning_count, 0),
        artifact_count=len(refs),
        record_count=docs_count + script_count,
        warning_count=warning_count,
        failure_count=0,
        key_metrics={
            "docs_count": docs_count,
            "script_count": script_count,
            "missing_scripts": sorted(missing_scripts),
        },
        warnings=warnings,
    )


def _gather_schema_versions() -> dict[str, str]:
    schema_versions: dict[str, str] = {}
    names = [
        "ARCHITECTURE_VERSION",
        "BRANCH_SCHEMA_VERSION",
        "LIKELIHOOD_SCHEMA_VERSION",
        "CALIBRATION_SCHEMA_VERSION",
        "OUTCOME_LEDGER_SCHEMA_VERSION",
        "CALIBRATION_V2_SCHEMA_VERSION",
        "POSTERIOR_OVERLAY_SCHEMA_VERSION",
        "DATA_QUALITY_CONTRACT_SCHEMA_VERSION",
        "RISK_TENSOR_SCHEMA_VERSION",
        "PORTFOLIO_OPTIMIZER_SCHEMA_VERSION",
        "OBSERVABILITY_SCHEMA_VERSION",
        "AUDIT_BUNDLE_SCHEMA_VERSION",
        "FACTOR_GOVERNANCE_SCHEMA_VERSION",
        "FACTOR_LIBRARY_SCHEMA_VERSION",
        "FACTOR_MATRIX_SCHEMA_VERSION",
        "FACTOR_EXPRESSION_SCHEMA_VERSION",
        "FACTOR_BACKTEST_SCHEMA_VERSION",
        "FACTOR_ROBUSTNESS_SCHEMA_VERSION",
        "FACTOR_COST_CAPACITY_SCHEMA_VERSION",
        "FACTOR_CORRELATION_SCHEMA_VERSION",
        "FACTOR_PORTFOLIO_CONTRIBUTION_SCHEMA_VERSION",
        "FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION",
        "FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION",
    ]
    for name in names:
        value = getattr(versioning, name, None)
        if value is not None:
            schema_versions[name] = str(value)
    from quant_investor.branch_config import BRANCH_WEIGHT_VERSION

    schema_versions["BRANCH_WEIGHT_VERSION"] = str(BRANCH_WEIGHT_VERSION)
    return dict(sorted(schema_versions.items()))


def build_run_manifest(
    *,
    run_id: str,
    artifact_refs: Sequence[ArtifactReference],
    generated_at: str | None = None,
    as_of: str | None = None,
    market: str | None = None,
    universe_key: str | None = None,
    universe_hash: str | None = None,
    architecture_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RunManifest:
    resolved_generated_at = generated_at or _utc_now_iso()
    sorted_refs = _sorted_refs(artifact_refs)
    manifest_id = make_run_manifest_id(
        run_id=run_id,
        generated_at=resolved_generated_at,
        artifact_ids=[ref.artifact_id for ref in sorted_refs],
    )
    return RunManifest(
        schema_version=OBSERVABILITY_SCHEMA_VERSION,
        manifest_id=manifest_id,
        run_id=run_id,
        generated_at=resolved_generated_at,
        as_of=as_of,
        market=market,
        universe_key=universe_key,
        universe_hash=universe_hash,
        architecture_version=architecture_version or getattr(versioning, "ARCHITECTURE_VERSION", None),
        schema_versions=_gather_schema_versions(),
        artifact_refs=sorted_refs,
        metadata=_coerce_metadata(metadata),
    )


def build_observability_summary(
    artifact_refs: Sequence[ArtifactReference],
    *,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SystemObservabilitySummary:
    resolved_generated_at = generated_at or _utc_now_iso()
    module_summaries = [
        summarize_outcome_ledger_artifacts(artifact_refs),
        summarize_calibration_artifacts(artifact_refs),
        summarize_data_quality_artifacts(artifact_refs),
        summarize_risk_tensor_artifacts(artifact_refs),
        summarize_portfolio_optimizer_artifacts(artifact_refs),
        summarize_docs_and_scripts_artifacts(artifact_refs),
    ]
    if _refs_for_module(artifact_refs, "factor_governance"):
        module_summaries.append(summarize_factor_governance_artifacts(artifact_refs))
    module_summaries = sorted(module_summaries, key=lambda summary: summary.module_name)
    return SystemObservabilitySummary(
        schema_version=OBSERVABILITY_SCHEMA_VERSION,
        generated_at=resolved_generated_at,
        overall_status=_derive_overall_status(module_summaries),
        module_summaries=module_summaries,
        total_artifacts=sum(summary.artifact_count for summary in module_summaries),
        total_records=sum(summary.record_count for summary in module_summaries),
        total_warnings=sum(summary.warning_count for summary in module_summaries),
        total_failures=sum(summary.failure_count for summary in module_summaries),
        metadata=_coerce_metadata(metadata),
    )


def build_dashboard_payload(
    manifest: RunManifest,
    summary: SystemObservabilitySummary,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    warnings = sorted(
        {
            warning
            for module_summary in summary.module_summaries
            for warning in module_summary.warnings
        }
    )
    payload = {
        "run_id": manifest.run_id,
        "generated_at": summary.generated_at,
        "overall_status": summary.overall_status,
        "schema_versions": dict(manifest.schema_versions),
        "modules": [
            {
                "module_name": module_summary.module_name,
                "status": module_summary.status,
                "key_metrics": dict(module_summary.key_metrics),
                "warnings": list(module_summary.warnings),
            }
            for module_summary in summary.module_summaries
        ],
        "artifacts": [
            {
                "name": ref.name,
                "artifact_type": ref.artifact_type,
                "path": ref.path,
                "exists": ref.exists,
                "record_count": ref.record_count,
                "schema_hint": ref.schema_hint,
            }
            for ref in manifest.artifact_refs
        ],
        "warnings": warnings,
        "metadata": _coerce_metadata(metadata),
    }
    json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)
    return dict(_json_safe(payload))


def build_audit_bundle(
    *,
    run_id: str,
    artifact_refs: Sequence[ArtifactReference],
    generated_at: str | None = None,
    as_of: str | None = None,
    market: str | None = None,
    universe_key: str | None = None,
    universe_hash: str | None = None,
    architecture_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AuditBundle:
    resolved_generated_at = generated_at or _utc_now_iso()
    manifest = build_run_manifest(
        run_id=run_id,
        artifact_refs=artifact_refs,
        generated_at=resolved_generated_at,
        as_of=as_of,
        market=market,
        universe_key=universe_key,
        universe_hash=universe_hash,
        architecture_version=architecture_version,
        metadata=metadata,
    )
    summary = build_observability_summary(
        artifact_refs,
        generated_at=resolved_generated_at,
        metadata=metadata,
    )
    dashboard_payload = build_dashboard_payload(manifest, summary, metadata=metadata)
    warnings = sorted(
        {
            warning
            for module_summary in summary.module_summaries
            for warning in module_summary.warnings
        }
    )
    return AuditBundle(
        schema_version=AUDIT_BUNDLE_SCHEMA_VERSION,
        bundle_id=make_audit_bundle_id(
            manifest_id=manifest.manifest_id,
            generated_at=resolved_generated_at,
        ),
        generated_at=resolved_generated_at,
        run_manifest=manifest,
        observability_summary=summary,
        dashboard_payload=dashboard_payload,
        warnings=warnings,
        metadata=_coerce_metadata(metadata),
    )


def render_audit_report_markdown(bundle: AuditBundle) -> str:
    manifest = bundle.run_manifest
    summary = bundle.observability_summary
    lines = [
        f"# Audit Bundle: {manifest.run_id}",
        "",
        f"Generated at: `{bundle.generated_at}`",
        "",
        "## Overall Status",
        "",
        f"`{summary.overall_status}`",
        "",
        "## Schema Versions",
        "",
        "| Name | Version |",
        "| --- | --- |",
    ]
    for name, value in manifest.schema_versions.items():
        lines.append(f"| {name} | `{value}` |")
    lines.extend([
        "",
        "## Module Health Summary",
        "",
        "| Module | Status | Artifacts | Records | Warnings | Failures |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for module_summary in summary.module_summaries:
        lines.append(
            "| "
            f"{module_summary.module_name} | `{module_summary.status}` | "
            f"{module_summary.artifact_count} | {module_summary.record_count} | "
            f"{module_summary.warning_count} | {module_summary.failure_count} |"
        )
    lines.extend(["", "## Key Metrics", ""])
    for module_summary in summary.module_summaries:
        lines.append(f"### {module_summary.module_name}")
        lines.append("")
        if module_summary.key_metrics:
            for key, value in sorted(module_summary.key_metrics.items()):
                lines.append(f"- `{key}`: `{value}`")
        else:
            lines.append("- None")
        lines.append("")
    lines.extend([
        "## Artifact Inventory",
        "",
        "| Name | Type | Exists | Records | Schema Hint | Path |",
        "| --- | --- | --- | ---: | --- | --- |",
    ])
    for ref in manifest.artifact_refs:
        exists_text = "yes" if ref.exists else "no"
        record_text = "" if ref.record_count is None else str(ref.record_count)
        schema_hint = ref.schema_hint or ""
        lines.append(
            f"| {ref.name} | {ref.artifact_type} | {exists_text} | {record_text} | "
            f"`{schema_hint}` | `{ref.path}` |"
        )
    lines.extend(["", "## Warnings", ""])
    if bundle.warnings:
        for warning in bundle.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Runtime Impact",
        "",
        "This audit bundle is generated offline and does not alter posterior, RiskGuard, "
        "PortfolioConstructor, provider, LLM, web, or execution behavior.",
        "",
    ])
    return "\n".join(lines)


class ObservabilityStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_OBSERVABILITY_DIR
        self.audit_bundle_path = self.root_dir / DEFAULT_AUDIT_BUNDLE_FILENAME
        self.audit_report_path = self.root_dir / DEFAULT_AUDIT_REPORT_FILENAME
        self.dashboard_payload_path = self.root_dir / DEFAULT_DASHBOARD_PAYLOAD_FILENAME
        self.run_manifest_path = self.root_dir / DEFAULT_RUN_MANIFEST_FILENAME

    def _write_text(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        return self._write_text(
            path,
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def save_audit_bundle(self, bundle: AuditBundle) -> Path:
        return self._write_json(self.audit_bundle_path, bundle.to_dict())

    def load_audit_bundle(self) -> AuditBundle:
        return AuditBundle.from_dict(read_json_file(self.audit_bundle_path))

    def save_audit_report(self, markdown: str) -> Path:
        return self._write_text(self.audit_report_path, markdown)

    def load_audit_report(self) -> str:
        return self.audit_report_path.read_text(encoding="utf-8")

    def save_dashboard_payload(self, payload: Mapping[str, Any]) -> Path:
        return self._write_json(self.dashboard_payload_path, payload)

    def load_dashboard_payload(self) -> dict[str, Any]:
        return read_json_file(self.dashboard_payload_path)

    def save_run_manifest(self, manifest: RunManifest) -> Path:
        return self._write_json(self.run_manifest_path, manifest.to_dict())

    def load_run_manifest(self) -> RunManifest:
        return RunManifest.from_dict(read_json_file(self.run_manifest_path))


__all__ = [
    "DEFAULT_OBSERVABILITY_DIR",
    "DEFAULT_AUDIT_BUNDLE_FILENAME",
    "DEFAULT_AUDIT_REPORT_FILENAME",
    "DEFAULT_DASHBOARD_PAYLOAD_FILENAME",
    "DEFAULT_RUN_MANIFEST_FILENAME",
    "ARTIFACT_TYPE_JSON",
    "ARTIFACT_TYPE_JSONL",
    "ARTIFACT_TYPE_MARKDOWN",
    "ARTIFACT_TYPE_DIRECTORY",
    "ARTIFACT_TYPE_UNKNOWN",
    "HEALTH_STATUS_PASS",
    "HEALTH_STATUS_WARN",
    "HEALTH_STATUS_FAIL",
    "HEALTH_STATUS_UNKNOWN",
    "ArtifactReference",
    "RunManifest",
    "ModuleHealthSummary",
    "SystemObservabilitySummary",
    "AuditBundle",
    "ObservabilityStore",
    "make_artifact_id",
    "make_run_manifest_id",
    "make_audit_bundle_id",
    "sha256_file",
    "count_jsonl_records",
    "read_json_file",
    "safe_json_dumps",
    "build_artifact_reference",
    "discover_phase_artifacts",
    "summarize_outcome_ledger_artifacts",
    "summarize_calibration_artifacts",
    "summarize_data_quality_artifacts",
    "summarize_risk_tensor_artifacts",
    "summarize_portfolio_optimizer_artifacts",
    "summarize_factor_governance_artifacts",
    "summarize_docs_and_scripts_artifacts",
    "build_run_manifest",
    "build_observability_summary",
    "build_dashboard_payload",
    "build_audit_bundle",
    "render_audit_report_markdown",
]
