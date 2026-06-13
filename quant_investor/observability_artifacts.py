"""Artifact discovery helpers for offline observability bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from quant_investor import versioning
from quant_investor.observability_types import (
    ARTIFACT_TYPE_DIRECTORY,
    ARTIFACT_TYPE_JSON,
    ARTIFACT_TYPE_JSONL,
    ARTIFACT_TYPE_MARKDOWN,
    ARTIFACT_TYPE_UNKNOWN,
    OBSERVABILITY_SCHEMA_VERSION,
    ArtifactReference,
    _PHASE_SCRIPT_NAMES,
    _coerce_metadata,
    _json_safe,
    _sorted_refs,
    make_artifact_id,
)


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

    try:
        from quant_investor.bayesian.outcome_ledger import (
            DEFAULT_OUTCOME_LEDGER_DIR,
            DEFAULT_OUTCOMES_FILENAME,
            DEFAULT_PREDICTIONS_FILENAME,
        )
    except ImportError:
        DEFAULT_OUTCOME_LEDGER_DIR = Path("data/bayesian_outcome_ledger")
        DEFAULT_PREDICTIONS_FILENAME = "predictions.jsonl"
        DEFAULT_OUTCOMES_FILENAME = "outcomes.jsonl"
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

    try:
        from quant_investor.bayesian.calibration_v2 import (
            DEFAULT_CALIBRATION_MODEL_FILENAME,
            DEFAULT_CALIBRATION_REPORT_FILENAME,
            DEFAULT_CALIBRATION_V2_DIR,
        )
    except ImportError:
        DEFAULT_CALIBRATION_V2_DIR = Path("data/bayesian_calibration_v2")
        DEFAULT_CALIBRATION_MODEL_FILENAME = "calibration_model_v2.json"
        DEFAULT_CALIBRATION_REPORT_FILENAME = "calibration_report_v2.json"
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

__all__ = [
    "sha256_file",
    "count_jsonl_records",
    "read_json_file",
    "safe_json_dumps",
    "build_artifact_reference",
    "discover_phase_artifacts",
]

