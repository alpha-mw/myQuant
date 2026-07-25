from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.retire_intelligence_catalog as retirement
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
)
from scripts.retire_intelligence_catalog import (
    ACTIVATION_RECEIPT_SCHEMA_VERSION,
    APPROVAL_ATTESTATION_SCHEMA_VERSION,
    CANONICAL_LIKELIHOOD_ORDER,
    CANONICAL_CATALOG_LOCK_PATH,
    CONFIRM_TOKEN,
    _build_parser,
    _tree_digest,
    retire_intelligence_catalog,
)
from tests.fixtures.strict_cn_snapshot import coverage_v4, v4_snapshot_paths


@pytest.fixture(autouse=True)
def _quiescent_process_scan(monkeypatch):
    monkeypatch.setattr(retirement, "_scan_active_runtime_processes", lambda: ([], []))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_fixture(tmp_path: Path) -> dict:
    repo_root = tmp_path / "repo"
    catalog_path = repo_root / "data" / "parquet" / "cn" / "_catalog.json"
    latest_path = repo_root / "data" / "parquet" / "cn" / "_latest.json"
    mart_path = repo_root / "data" / "parquet" / "cn" / "intelligence_daily" / "part.parquet"
    mart_path.parent.mkdir(parents=True)
    catalog = {
        "schema_version": "strict-parquet-catalog.v1",
        "required_tables": ["daily_basic", "intelligence_daily", "macro_daily"],
        "tables": {
            "daily_basic": {"path": "data/parquet/cn/daily_basic/part.parquet"},
            "intelligence_daily": {"path": str(mart_path.relative_to(repo_root))},
            "macro_daily": {"path": "data/parquet/cn/macro_daily/part.parquet"},
        },
    }
    _write_json(catalog_path, catalog)
    bars_root, serving_root, manifest_path = v4_snapshot_paths(
        repo_root / "data",
        "stable",
    )
    bars_root.mkdir(parents=True)
    (bars_root / "bars.parquet").write_bytes(b"probe only")
    serving_symbol_root = serving_root / "symbol=000001.SZ"
    serving_symbol_root.mkdir(parents=True)
    (serving_symbol_root / "bars.parquet").write_bytes(b"probe only")
    coverage = coverage_v4(
        repo_root / "data",
        ["000001.SZ"],
        trade_date="20260713",
    )
    _write_json(manifest_path, {"snapshot_id": "stable", "coverage": coverage})
    _write_json(
        latest_path,
        {
            "status": "OK",
            "snapshot_id": "stable",
            "latest_complete_trade_date": "20260713",
            "latest_trade_date": "20260713",
            "table_root": str(bars_root),
            "derived_serving_root": str(serving_root),
            "manifest_path": str(manifest_path),
            "coverage": coverage,
            "blockers": [],
        },
    )
    mart_path.write_bytes(b"immutable historical mart")

    report_labels = (
        "reports/daily",
        "reports/branch_readiness",
        "reports/branch_readiness_clean_parquet_smoke",
        "reports/holdings_dag_review",
    )
    report_tree_paths: dict[str, Path] = {}
    for index, label in enumerate(report_labels):
        tree = repo_root / label
        tree.mkdir(parents=True)
        (tree / f"fixture-{index}.txt").write_text(
            f"immutable report tree {index}\n", encoding="utf-8"
        )
        report_tree_paths[label] = tree
    expected_report_trees = {
        label: _tree_digest(repo_root, path) for label, path in report_tree_paths.items()
    }

    candidate_commit = "a" * 40
    activation_receipt_path = (
        repo_root / "reports" / "intelligence_retirement" / "v14_activation_receipt.json"
    )
    approval_attestation_path = tmp_path / "maxwell-approval.json"
    expected_files = {
        "catalog": _sha256(catalog_path),
        "latest": _sha256(latest_path),
        "mart": _sha256(mart_path),
    }
    receipt = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "canonical_branch_order": list(CANONICAL_BRANCH_ORDER),
        "canonical_likelihood_order": list(CANONICAL_LIKELIHOOD_ORDER),
        "candidate_commit": candidate_commit,
        "gates": {
            "code_gate_passed": True,
            "replay_gate_passed": True,
            "no_new_buy": True,
            "risk_guard_not_weakened": True,
            "parallel_work_integrated": True,
            "runtime_quiesced": True,
            "retired_pyc_removed": True,
        },
        "evidence": {
            "catalog": {
                "path": catalog_path.relative_to(repo_root).as_posix(),
                "sha256": expected_files["catalog"],
            },
            "latest": {
                "path": latest_path.relative_to(repo_root).as_posix(),
                "sha256": expected_files["latest"],
            },
            "mart": {
                "path": mart_path.relative_to(repo_root).as_posix(),
                "sha256": expected_files["mart"],
            },
            "report_trees": expected_report_trees,
        },
    }
    _write_json(activation_receipt_path, receipt)

    fixture = {
        "repo_root": repo_root,
        "catalog_path": catalog_path,
        "latest_path": latest_path,
        "mart_path": mart_path,
        "report_tree_paths": report_tree_paths,
        "activation_receipt_path": activation_receipt_path,
        "approval_attestation_path": approval_attestation_path,
        "candidate_commit": candidate_commit,
        "candidate_worktree_clean": True,
        "expected_catalog_sha256": expected_files["catalog"],
        "expected_latest_sha256": expected_files["latest"],
        "expected_mart_sha256": expected_files["mart"],
        "expected_report_trees": expected_report_trees,
        "apply": False,
        "confirm_token": None,
    }
    _write_approval(fixture)
    return fixture


def _write_approval(fixture: dict) -> None:
    receipt_sha = _sha256(fixture["activation_receipt_path"])
    approval = {
        "schema_version": APPROVAL_ATTESTATION_SCHEMA_VERSION,
        "approved_by": "Maxwell",
        "approval_scope": CONFIRM_TOKEN,
        "nonce": "b" * 64,
        "explicit_confirmation": True,
        "activation_receipt_sha256": receipt_sha,
        "candidate_commit": fixture["candidate_commit"],
        "catalog_sha256": fixture["expected_catalog_sha256"],
        "runtime_quiescence": {
            "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "max_age_seconds": 300,
            "candidate_commit": fixture["candidate_commit"],
            "catalog_sha256": fixture["expected_catalog_sha256"],
            "scope": [
                "daily_daemon",
                "web_runtime",
                "market_runtime",
                "research_runtime",
                "retired_source_and_pyc",
            ],
            "active_processes": [],
            "residue": [],
        },
    }
    _write_json(fixture["approval_attestation_path"], approval)
    fixture["approval_attestation_path"].chmod(0o600)


def _apply_fixture(fixture: dict, **overrides):
    kwargs = dict(fixture)
    kwargs.update({"apply": True, "confirm_token": CONFIRM_TOKEN})
    kwargs.update(overrides)
    return retire_intelligence_catalog(**kwargs)


def test_catalog_retirement_dry_run_is_read_only_without_activation_files(tmp_path):
    fixture = _write_fixture(tmp_path)
    before = fixture["catalog_path"].read_bytes()
    fixture["activation_receipt_path"].unlink()
    fixture["approval_attestation_path"].unlink()

    report = retire_intelligence_catalog(**fixture)

    assert report["status"] == "would_retire"
    assert report["apply_requested"] is False
    assert fixture["catalog_path"].read_bytes() == before
    assert report["removed_required_table"] is True
    assert report["removed_table_entry"] is True


def test_catalog_retirement_applies_only_catalog_entry_removal(tmp_path):
    fixture = _write_fixture(tmp_path)
    before = {
        "latest": fixture["latest_path"].read_bytes(),
        "mart": fixture["mart_path"].read_bytes(),
        "trees": {
            label: _tree_digest(fixture["repo_root"], path)
            for label, path in fixture["report_tree_paths"].items()
        },
    }

    report = _apply_fixture(fixture)

    catalog = json.loads(fixture["catalog_path"].read_text(encoding="utf-8"))
    assert report["status"] == "retired"
    assert "intelligence_daily" not in catalog["required_tables"]
    assert "intelligence_daily" not in catalog["tables"]
    assert fixture["latest_path"].read_bytes() == before["latest"]
    assert fixture["mart_path"].read_bytes() == before["mart"]
    assert {
        label: _tree_digest(fixture["repo_root"], path)
        for label, path in fixture["report_tree_paths"].items()
    } == before["trees"]


def test_catalog_retirement_requires_static_confirmation_token(tmp_path):
    fixture = _write_fixture(tmp_path)
    before = fixture["catalog_path"].read_bytes()

    report = _apply_fixture(fixture, confirm_token="WRONG")

    assert report["status"] == "blocked_confirmation_required"
    assert "static_confirmation_token_required" in report["blockers"]
    assert fixture["catalog_path"].read_bytes() == before


def test_catalog_retirement_requires_final_clean_candidate_commit(tmp_path):
    fixture = _write_fixture(tmp_path)

    report = _apply_fixture(fixture, candidate_worktree_clean=False)

    assert report["status"] == "blocked_candidate_state"
    assert "candidate_worktree_not_clean" in report["blockers"]


def test_catalog_retirement_requires_activation_receipt(tmp_path):
    fixture = _write_fixture(tmp_path)
    fixture["activation_receipt_path"].unlink()

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_activation_evidence"
    assert any(
        blocker.startswith("activation_receipt_unreadable:") for blocker in report["blockers"]
    )


@pytest.mark.parametrize(
    "gate",
    [
        "code_gate_passed",
        "replay_gate_passed",
        "no_new_buy",
        "risk_guard_not_weakened",
        "parallel_work_integrated",
        "runtime_quiesced",
        "retired_pyc_removed",
    ],
)
def test_catalog_retirement_requires_every_activation_gate(tmp_path, gate):
    fixture = _write_fixture(tmp_path)
    receipt = json.loads(fixture["activation_receipt_path"].read_text(encoding="utf-8"))
    receipt["gates"][gate] = False
    _write_json(fixture["activation_receipt_path"], receipt)
    _write_approval(fixture)

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_activation_evidence"
    assert f"activation_receipt_gate_not_passed:{gate}" in report["blockers"]


@pytest.mark.parametrize(
    ("field", "bad_value", "blocker"),
    [
        ("architecture_version", "13.1.0", "activation_receipt_architecture_version_mismatch"),
        (
            "branch_schema_version",
            "branch-schema.v13",
            "activation_receipt_branch_schema_version_mismatch",
        ),
        (
            "likelihood_schema_version",
            "likelihood-schema.v13",
            "activation_receipt_likelihood_schema_version_mismatch",
        ),
        (
            "report_protocol_version",
            "report-protocol.v13",
            "activation_receipt_report_protocol_version_mismatch",
        ),
        (
            "canonical_branch_order",
            ["quant", "fundamental"],
            "activation_receipt_branch_order_mismatch",
        ),
        ("canonical_likelihood_order", ["quant"], "activation_receipt_likelihood_order_mismatch"),
    ],
)
def test_catalog_retirement_requires_exact_v14_envelope(tmp_path, field, bad_value, blocker):
    fixture = _write_fixture(tmp_path)
    receipt = json.loads(fixture["activation_receipt_path"].read_text(encoding="utf-8"))
    receipt[field] = bad_value
    _write_json(fixture["activation_receipt_path"], receipt)
    _write_approval(fixture)

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_activation_evidence"
    assert blocker in report["blockers"]


def test_catalog_retirement_requires_repo_external_maxwell_approval(tmp_path):
    fixture = _write_fixture(tmp_path)
    inside_repo = fixture["repo_root"] / "approval.json"
    fixture["approval_attestation_path"].replace(inside_repo)

    report = _apply_fixture(fixture, approval_attestation_path=inside_repo)

    assert report["status"] == "blocked_path_or_configuration"
    assert "approval_attestation_must_be_repo_external" in report["blockers"]


def test_catalog_retirement_binds_approval_to_receipt_sha(tmp_path):
    fixture = _write_fixture(tmp_path)
    with fixture["activation_receipt_path"].open("a", encoding="utf-8") as handle:
        handle.write(" \n")

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_activation_evidence"
    assert "approval_attestation_activation_receipt_sha256_mismatch" in report["blockers"]


def test_catalog_retirement_rejects_stale_runtime_quiescence_attestation(tmp_path):
    fixture = _write_fixture(tmp_path)
    approval = json.loads(fixture["approval_attestation_path"].read_text(encoding="utf-8"))
    approval["runtime_quiescence"]["observed_at_utc"] = (
        (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    )
    _write_json(fixture["approval_attestation_path"], approval)
    fixture["approval_attestation_path"].chmod(0o600)

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_activation_evidence"
    assert "approval_attestation_runtime_observation_stale" in report["blockers"]


@pytest.mark.parametrize("residue_kind", ["source", "pyc"])
def test_catalog_retirement_scans_retired_source_and_pyc_residue(tmp_path, residue_kind):
    fixture = _write_fixture(tmp_path)
    if residue_kind == "source":
        residue = fixture["repo_root"] / "quant_investor" / "agents" / "intelligence_agent.py"
    else:
        residue = (
            fixture["repo_root"]
            / "quant_investor"
            / "agents"
            / "__pycache__"
            / "intelligence_agent.cpython-313.pyc"
        )
    residue.parent.mkdir(parents=True, exist_ok=True)
    residue.write_bytes(b"retired residue")
    before = fixture["catalog_path"].read_bytes()

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_initial_runtime_not_quiescent"
    assert "initial_retired_source_or_pyc_residue" in report["blockers"]
    assert fixture["catalog_path"].read_bytes() == before


def test_catalog_retirement_scans_active_runtime_processes(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    monkeypatch.setattr(
        retirement,
        "_scan_active_runtime_processes",
        lambda: (
            [
                {
                    "scope": "daily_daemon",
                    "pid": 123,
                    "parent_pid": 1,
                    "command": "python daily_runner.py --daemon",
                }
            ],
            [],
        ),
    )

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_initial_runtime_not_quiescent"
    assert "initial_runtime_processes_active" in report["blockers"]


@pytest.mark.parametrize("active_phase", ["pre_replace", "post_apply"])
def test_catalog_retirement_rechecks_runtime_quiescence_through_transaction(
    tmp_path, monkeypatch, active_phase
):
    fixture = _write_fixture(tmp_path)
    before = fixture["catalog_path"].read_bytes()
    calls = 0

    def capture_runtime_state(_repo_root):
        nonlocal calls
        calls += 1
        active_call = 2 if active_phase == "pre_replace" else 3
        processes = (
            [
                {
                    "scope": "research_runtime",
                    "pid": 456,
                    "parent_pid": 1,
                    "command": "quant-investor research run",
                }
            ]
            if calls == active_call
            else []
        )
        return {"active_processes": processes, "residue": []}, []

    monkeypatch.setattr(retirement, "_capture_runtime_state", capture_runtime_state)

    report = _apply_fixture(fixture)

    if active_phase == "pre_replace":
        assert report["status"] == "blocked_pre_replace_evidence_changed"
        assert "pre_replace_runtime_processes_active" in report["blockers"]
    else:
        assert report["status"] == "rolled_back_post_apply_readback_failed"
        assert "post_apply_runtime_processes_active" in report["blockers"]
        assert report["rollback_verified"] is True
    assert fixture["catalog_path"].read_bytes() == before


def test_catalog_retirement_holds_nonblocking_lock_for_full_transaction(tmp_path):
    fixture = _write_fixture(tmp_path)
    before = fixture["catalog_path"].read_bytes()
    lock_path = fixture["catalog_path"].parent / CANONICAL_CATALOG_LOCK_PATH.name
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        report = _apply_fixture(fixture)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert report["status"] == "blocked_catalog_lock_busy"
    assert report["catalog_lock"]["non_blocking"] is True
    assert fixture["catalog_path"].read_bytes() == before


def test_catalog_retirement_does_not_overwrite_immediate_concurrent_catalog_update(
    tmp_path, monkeypatch
):
    fixture = _write_fixture(tmp_path)
    concurrent_payload = b'{"concurrent":"newer"}\n'
    original_read = retirement._read_with_sha256
    catalog_reads = 0

    def read_with_concurrent_update(path):
        nonlocal catalog_reads
        if Path(path) == fixture["catalog_path"]:
            catalog_reads += 1
            if catalog_reads == 4:
                fixture["catalog_path"].write_bytes(concurrent_payload)
        return original_read(path)

    monkeypatch.setattr(retirement, "_read_with_sha256", read_with_concurrent_update)

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_catalog_cas_failed"
    assert "catalog_changed_immediately_before_replace" in report["blockers"]
    assert fixture["catalog_path"].read_bytes() == concurrent_payload


def test_catalog_retirement_stops_closed_on_initial_hash_drift(tmp_path):
    fixture = _write_fixture(tmp_path)
    before = fixture["catalog_path"].read_bytes()

    report = _apply_fixture(fixture, expected_catalog_sha256="0" * 64)

    assert report["status"] == "blocked_initial_evidence_mismatch"
    assert "initial_catalog_sha256_mismatch" in report["blockers"]
    assert fixture["catalog_path"].read_bytes() == before


@pytest.mark.parametrize("drift_target", ["latest", "mart", "report_tree"])
def test_catalog_retirement_revalidates_all_evidence_immediately_before_replace(
    tmp_path, monkeypatch, drift_target
):
    fixture = _write_fixture(tmp_path)
    before = fixture["catalog_path"].read_bytes()
    original_capture = retirement._capture_immutable_evidence
    calls = 0

    def capture_with_drift(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            if drift_target == "latest":
                fixture["latest_path"].write_bytes(b"drifted latest")
            elif drift_target == "mart":
                fixture["mart_path"].write_bytes(b"drifted mart")
            else:
                tree = fixture["report_tree_paths"]["reports/daily"]
                (tree / "fixture-0.txt").write_text("drifted tree\n", encoding="utf-8")
        return original_capture(**kwargs)

    monkeypatch.setattr(retirement, "_capture_immutable_evidence", capture_with_drift)

    report = _apply_fixture(fixture)

    assert report["status"] == "blocked_pre_replace_evidence_changed"
    assert fixture["catalog_path"].read_bytes() == before


def test_catalog_retirement_rolls_back_catalog_on_post_write_evidence_drift(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    before = fixture["catalog_path"].read_bytes()
    original_capture = retirement._capture_immutable_evidence
    calls = 0

    def capture_with_post_write_drift(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            tree = fixture["report_tree_paths"]["reports/holdings_dag_review"]
            (tree / "fixture-3.txt").write_text("post-write drift\n", encoding="utf-8")
        return original_capture(**kwargs)

    monkeypatch.setattr(retirement, "_capture_immutable_evidence", capture_with_post_write_drift)

    report = _apply_fixture(fixture)

    assert report["status"] == "rolled_back_post_apply_readback_failed"
    assert report["rollback_verified"] is True
    assert fixture["catalog_path"].read_bytes() == before


def test_production_parser_exposes_no_path_or_digest_overrides():
    parser = _build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert option_strings == {"-h", "--help", "--apply", "--confirm-token"}
