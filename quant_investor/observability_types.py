"""Contracts for offline observability and audit bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    "AUDIT_BUNDLE_SCHEMA_VERSION",
    "OBSERVABILITY_SCHEMA_VERSION",
    "ArtifactReference",
    "RunManifest",
    "ModuleHealthSummary",
    "SystemObservabilitySummary",
    "AuditBundle",
    "make_artifact_id",
    "make_run_manifest_id",
    "make_audit_bundle_id",
]
