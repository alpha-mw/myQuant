"""Offline observability and audit bundle helpers for staged upgrade artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor import versioning
from quant_investor.observability_artifacts import (
    build_artifact_reference,
    count_jsonl_records,
    discover_phase_artifacts,
    read_json_file,
    safe_json_dumps,
    sha256_file,
    _read_jsonl_objects,
)
from quant_investor.observability_types import (
    ARTIFACT_TYPE_DIRECTORY,
    ARTIFACT_TYPE_JSON,
    ARTIFACT_TYPE_JSONL,
    ARTIFACT_TYPE_MARKDOWN,
    ARTIFACT_TYPE_UNKNOWN,
    AUDIT_BUNDLE_SCHEMA_VERSION,
    DEFAULT_AUDIT_BUNDLE_FILENAME,
    DEFAULT_AUDIT_REPORT_FILENAME,
    DEFAULT_DASHBOARD_PAYLOAD_FILENAME,
    DEFAULT_OBSERVABILITY_DIR,
    DEFAULT_RUN_MANIFEST_FILENAME,
    HEALTH_STATUS_FAIL,
    HEALTH_STATUS_PASS,
    HEALTH_STATUS_UNKNOWN,
    HEALTH_STATUS_WARN,
    OBSERVABILITY_SCHEMA_VERSION,
    ArtifactReference,
    AuditBundle,
    ModuleHealthSummary,
    RunManifest,
    SystemObservabilitySummary,
    _coerce_metadata,
    _derive_overall_status,
    _json_safe,
    _non_negative_int_or_zero,
    _sorted_refs,
    _status_from_counts,
    make_artifact_id,
    make_audit_bundle_id,
    make_run_manifest_id,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    try:
        from quant_investor.branch_config import BRANCH_WEIGHT_VERSION
    except ImportError:
        BRANCH_WEIGHT_VERSION = None
    if BRANCH_WEIGHT_VERSION is not None:
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
