from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from quant_investor.observability import (
    ARTIFACT_TYPE_DIRECTORY,
    ARTIFACT_TYPE_JSON,
    ARTIFACT_TYPE_JSONL,
    AUDIT_BUNDLE_SCHEMA_VERSION,
    HEALTH_STATUS_FAIL,
    HEALTH_STATUS_PASS,
    HEALTH_STATUS_WARN,
    ArtifactReference,
    AuditBundle,
    ModuleHealthSummary,
    ObservabilityStore,
    RunManifest,
    SystemObservabilitySummary,
    build_artifact_reference,
    build_audit_bundle,
    build_dashboard_payload,
    build_observability_summary,
    build_run_manifest,
    count_jsonl_records,
    discover_phase_artifacts,
    make_artifact_id,
    make_audit_bundle_id,
    make_run_manifest_id,
    read_json_file,
    render_audit_report_markdown,
    safe_json_dumps,
    sha256_file,
    summarize_calibration_artifacts,
    summarize_docs_and_scripts_artifacts,
    summarize_outcome_ledger_artifacts,
)
from quant_investor.versioning import OBSERVABILITY_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[2]
FIXED_GENERATED_AT = "2026-04-26T00:00:00Z"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n\n", encoding="utf-8")


def _basic_ref(tmp_path: Path) -> ArtifactReference:
    payload = tmp_path / "payload.json"
    payload.write_text('{"ok": true}\n', encoding="utf-8")
    return build_artifact_reference(name="payload", path=payload, schema_hint="schema.v1")


def test_dataclass_round_trips(tmp_path) -> None:
    ref = _basic_ref(tmp_path)
    manifest = RunManifest(
        manifest_id="manifest",
        run_id="run",
        generated_at=FIXED_GENERATED_AT,
        as_of="2026-04-26",
        market="CN",
        universe_key="unit",
        universe_hash="hash",
        architecture_version="13.0.0-stable",
        schema_versions={"A": "v1"},
        artifact_refs=[ref],
    )
    module_summary = ModuleHealthSummary(
        module_name="unit",
        status=HEALTH_STATUS_WARN,
        artifact_count=1,
        record_count=1,
        warning_count=1,
        failure_count=0,
        key_metrics={"records": 1},
        warnings=["b", "a"],
    )
    system_summary = SystemObservabilitySummary(
        generated_at=FIXED_GENERATED_AT,
        overall_status=HEALTH_STATUS_WARN,
        module_summaries=[module_summary],
        total_artifacts=1,
        total_records=1,
        total_warnings=1,
        total_failures=0,
    )
    bundle = AuditBundle(
        bundle_id="bundle",
        generated_at=FIXED_GENERATED_AT,
        run_manifest=manifest,
        observability_summary=system_summary,
        dashboard_payload={"run_id": "run"},
        warnings=["z", "a"],
    )

    assert ArtifactReference.from_dict(ref.to_dict()).to_dict() == ref.to_dict()
    assert RunManifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()
    assert ModuleHealthSummary.from_dict(module_summary.to_dict()).warnings == ["a", "b"]
    assert SystemObservabilitySummary.from_dict(system_summary.to_dict()).to_dict() == system_summary.to_dict()
    assert AuditBundle.from_dict(bundle.to_dict()).warnings == ["a", "z"]


def test_deterministic_ids_and_file_helpers(tmp_path) -> None:
    assert make_artifact_id(name="a", path="/tmp/a") == make_artifact_id(name="a", path="/tmp/a")
    assert make_run_manifest_id(run_id="r", generated_at=FIXED_GENERATED_AT, artifact_ids=["b", "a"]) == (
        make_run_manifest_id(run_id="r", generated_at=FIXED_GENERATED_AT, artifact_ids=["a", "b"])
    )
    assert make_audit_bundle_id(manifest_id="m", generated_at=FIXED_GENERATED_AT) == make_audit_bundle_id(
        manifest_id="m",
        generated_at=FIXED_GENERATED_AT,
    )

    content = tmp_path / "content.txt"
    content.write_text("hello\n", encoding="utf-8")
    assert sha256_file(content) == hashlib.sha256(b"hello\n").hexdigest()

    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert count_jsonl_records(jsonl) == 2

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        read_json_file(bad_json)

    assert safe_json_dumps({"b": 1, "a": 2}) == safe_json_dumps({"a": 2, "b": 1})


def test_artifact_reference_file_types(tmp_path) -> None:
    json_file = tmp_path / "payload.json"
    json_file.write_text('{"ok": true}\n', encoding="utf-8")
    jsonl_file = tmp_path / "records.jsonl"
    jsonl_file.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    directory = tmp_path / "dir"
    directory.mkdir()
    missing = tmp_path / "missing.jsonl"

    json_ref = build_artifact_reference(name="json", path=json_file)
    jsonl_ref = build_artifact_reference(name="jsonl", path=jsonl_file)
    directory_ref = build_artifact_reference(name="dir", path=directory)
    missing_ref = build_artifact_reference(name="missing", path=missing)

    assert json_ref.artifact_type == ARTIFACT_TYPE_JSON
    assert json_ref.exists is True
    assert json_ref.size_bytes is not None
    assert json_ref.sha256 is not None
    assert json_ref.record_count == 1
    assert jsonl_ref.artifact_type == ARTIFACT_TYPE_JSONL
    assert jsonl_ref.record_count == 2
    assert missing_ref.exists is False
    assert missing_ref.sha256 is None
    assert directory_ref.artifact_type == ARTIFACT_TYPE_DIRECTORY
    assert directory_ref.size_bytes is None


def test_artifact_discovery_includes_expected_names_and_missing_refs(tmp_path) -> None:
    outcome_dir = tmp_path / "outcome"
    calibration_dir = tmp_path / "calibration"
    data_quality_dir = tmp_path / "quality"
    risk_dir = tmp_path / "risk"
    optimizer_dir = tmp_path / "optimizer"
    docs_dir = tmp_path / "docs"
    scripts_dir = tmp_path / "scripts"
    _write_jsonl(outcome_dir / "predictions.jsonl", [{"prediction_id": "p1"}])
    (docs_dir).mkdir()
    (docs_dir / "system_upgrade_plan.md").write_text("# Plan\n", encoding="utf-8")
    scripts_dir.mkdir()
    (scripts_dir / "phase1_quality_gate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    refs = discover_phase_artifacts(
        outcome_ledger_dir=outcome_dir,
        calibration_v2_dir=calibration_dir,
        data_quality_dir=data_quality_dir,
        risk_tensor_dir=risk_dir,
        portfolio_optimizer_dir=optimizer_dir,
        docs_dir=docs_dir,
        scripts_dir=scripts_dir,
    )
    names = [ref.name for ref in refs]

    assert names == sorted(names)
    assert "outcome_ledger_predictions" in names
    assert "outcome_ledger_outcomes" in names
    assert "calibration_v2_model" in names
    assert "docs_system_upgrade_plan" in names
    assert "script_phase8_quality_gate" in names
    assert any(ref.name == "outcome_ledger_outcomes" and ref.exists is False for ref in refs)


def test_artifact_discovery_defaults_to_v15_bayesian_namespaces(tmp_path) -> None:
    refs = discover_phase_artifacts(
        data_quality_dir=tmp_path / "quality",
        risk_tensor_dir=tmp_path / "risk",
        portfolio_optimizer_dir=tmp_path / "optimizer",
        factor_library_dir=tmp_path / "factor_library",
        docs_dir=tmp_path / "docs",
        scripts_dir=tmp_path / "scripts",
    )
    paths = {ref.name: ref.path for ref in refs}

    assert "/bayesian_outcome_ledger/v15/" in f"/{paths['outcome_ledger_predictions']}"
    assert "/bayesian_calibration_v2/v15/" in f"/{paths['calibration_v2_model']}"


def test_module_summaries_count_records_and_warn_or_fail(tmp_path) -> None:
    outcome_dir = tmp_path / "outcome"
    _write_jsonl(
        outcome_dir / "predictions.jsonl",
        [{"prediction_id": "p1"}, {"prediction_id": "p2"}],
    )
    _write_jsonl(outcome_dir / "outcomes.jsonl", [{"prediction_id": "p1"}])
    outcome_refs = discover_phase_artifacts(
        outcome_ledger_dir=outcome_dir,
        calibration_v2_dir=tmp_path / "missing_cal",
        data_quality_dir=tmp_path / "missing_dq",
        risk_tensor_dir=tmp_path / "missing_risk",
        portfolio_optimizer_dir=tmp_path / "missing_opt",
        docs_dir=tmp_path / "missing_docs",
        scripts_dir=tmp_path / "missing_scripts",
    )
    outcome_summary = summarize_outcome_ledger_artifacts(outcome_refs)
    assert outcome_summary.key_metrics["prediction_records"] == 2
    assert outcome_summary.key_metrics["outcome_records"] == 1
    assert outcome_summary.key_metrics["unresolved_estimate"] == 1

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "calibration_model_v2.json").write_text(
        json.dumps({"curves": [{"id": 1}, {"id": 2}]}),
        encoding="utf-8",
    )
    (calibration_dir / "calibration_report_v2.json").write_text('{"ok": true}', encoding="utf-8")
    calibration_refs = discover_phase_artifacts(calibration_v2_dir=calibration_dir)
    calibration_summary = summarize_calibration_artifacts(calibration_refs)
    assert calibration_summary.key_metrics["model_exists"] is True
    assert calibration_summary.key_metrics["report_exists"] is True
    assert calibration_summary.key_metrics["curve_count"] == 2

    docs_scripts_summary = summarize_docs_and_scripts_artifacts(outcome_refs)
    assert docs_scripts_summary.status == HEALTH_STATUS_WARN
    assert docs_scripts_summary.key_metrics["missing_scripts"]

    bad_dir = tmp_path / "bad_calibration"
    bad_dir.mkdir()
    (bad_dir / "calibration_model_v2.json").write_text("{bad json}", encoding="utf-8")
    bad_summary = summarize_calibration_artifacts(discover_phase_artifacts(calibration_v2_dir=bad_dir))
    assert bad_summary.status == HEALTH_STATUS_FAIL
    assert bad_summary.failure_count == 1


def test_audit_bundle_builders_are_serializable_and_deterministic(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    scripts_dir = tmp_path / "scripts"
    docs_dir.mkdir()
    scripts_dir.mkdir()
    (docs_dir / "system_upgrade_plan.md").write_text("# Plan\n", encoding="utf-8")
    for phase in range(1, 9):
        (scripts_dir / f"phase{phase}_quality_gate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (scripts_dir / "staged_upgrade_quality_gate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    refs = discover_phase_artifacts(
        outcome_ledger_dir=tmp_path / "outcome",
        calibration_v2_dir=tmp_path / "cal",
        data_quality_dir=tmp_path / "dq",
        risk_tensor_dir=tmp_path / "risk",
        portfolio_optimizer_dir=tmp_path / "opt",
        docs_dir=docs_dir,
        scripts_dir=scripts_dir,
    )

    manifest = build_run_manifest(run_id="run", artifact_refs=refs, generated_at=FIXED_GENERATED_AT)
    summary = build_observability_summary(refs, generated_at=FIXED_GENERATED_AT)
    dashboard = build_dashboard_payload(manifest, summary)
    bundle = build_audit_bundle(run_id="run", artifact_refs=refs, generated_at=FIXED_GENERATED_AT)
    bundle_again = build_audit_bundle(run_id="run", artifact_refs=refs, generated_at=FIXED_GENERATED_AT)

    assert "OBSERVABILITY_SCHEMA_VERSION" in manifest.schema_versions
    assert "AUDIT_BUNDLE_SCHEMA_VERSION" in manifest.schema_versions
    assert "BRANCH_SCHEMA_VERSION" in manifest.schema_versions
    assert "LIKELIHOOD_SCHEMA_VERSION" in manifest.schema_versions
    assert "BRANCH_WEIGHT_VERSION" in manifest.schema_versions
    assert summary.overall_status == HEALTH_STATUS_WARN
    json.dumps(dashboard, ensure_ascii=False, sort_keys=True)
    assert bundle.bundle_id == bundle_again.bundle_id
    assert bundle.warnings == sorted(bundle.warnings)


def test_markdown_report_contains_required_sections(tmp_path) -> None:
    bundle = build_audit_bundle(
        run_id="report-run",
        artifact_refs=[_basic_ref(tmp_path)],
        generated_at=FIXED_GENERATED_AT,
    )

    markdown = render_audit_report_markdown(bundle)

    assert "report-run" in markdown
    assert "Overall Status" in markdown
    assert "Module Health Summary" in markdown
    assert "This audit bundle is generated offline" in markdown


def test_observability_store_round_trips_and_rejects_bad_json(tmp_path) -> None:
    bundle = build_audit_bundle(
        run_id="store-run",
        artifact_refs=[_basic_ref(tmp_path)],
        generated_at=FIXED_GENERATED_AT,
    )
    markdown = render_audit_report_markdown(bundle)
    store = ObservabilityStore(tmp_path / "store")

    store.save_audit_bundle(bundle)
    assert store.load_audit_bundle().bundle_id == bundle.bundle_id
    store.save_audit_report(markdown)
    assert "store-run" in store.load_audit_report()
    store.save_dashboard_payload(bundle.dashboard_payload)
    assert store.load_dashboard_payload()["run_id"] == "store-run"
    store.save_run_manifest(bundle.run_manifest)
    assert store.load_run_manifest().manifest_id == bundle.run_manifest.manifest_id

    store.audit_bundle_path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        store.load_audit_bundle()


def test_cli_smoke_builds_audit_outputs(tmp_path) -> None:
    outcome_dir = tmp_path / "outcome"
    docs_dir = tmp_path / "docs"
    scripts_dir = tmp_path / "scripts"
    output_dir = tmp_path / "out"
    _write_jsonl(outcome_dir / "predictions.jsonl", [{"prediction_id": "p1"}])
    docs_dir.mkdir()
    docs_dir.joinpath("system_upgrade_plan.md").write_text("# Plan\n", encoding="utf-8")
    scripts_dir.mkdir()
    scripts_dir.joinpath("phase1_quality_gate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_audit_bundle.py"),
            "--run-id",
            "cli-run",
            "--output-dir",
            str(output_dir),
            "--outcome-ledger-dir",
            str(outcome_dir),
            "--calibration-v2-dir",
            str(tmp_path / "cal"),
            "--data-quality-dir",
            str(tmp_path / "dq"),
            "--risk-tensor-dir",
            str(tmp_path / "risk"),
            "--portfolio-optimizer-dir",
            str(tmp_path / "opt"),
            "--docs-dir",
            str(docs_dir),
            "--scripts-dir",
            str(scripts_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "run_id: cli-run" in result.stdout
    assert output_dir.joinpath("audit_bundle.json").exists()
    assert output_dir.joinpath("audit_report.md").exists()
    assert output_dir.joinpath("dashboard_payload.json").exists()
    assert output_dir.joinpath("run_manifest.json").exists()
