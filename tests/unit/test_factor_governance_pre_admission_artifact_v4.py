from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from quant_investor.codex_review.storage import (
    DifferentBytesError,
    ProtocolError,
    StateConflictError,
    canonical_json_bytes,
)
from quant_investor.factors.governance_pre_admission_artifact_v4 import (
    CODEX_IC_STATUS_SCHEMA_VERSION,
    CODEX_S1_STATUS_SCHEMA_VERSION,
    PRE_ADMISSION_REPORT_FILENAME,
    REPLAY_STATUS_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    FactorGovernancePreAdmissionV4Error,
    build_factor_governance_pre_admission_report_v4,
    canonical_semantic_sha256_v4,
    publish_factor_governance_pre_admission_report_v4,
    validate_factor_governance_pre_admission_report_v4,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _screening_summary() -> dict[str, object]:
    return {
        "schema_version": "factor-governance-screening-summary.v4",
        "evidence_class": "diagnostic_report_only",
        "screening_evidence_sha256": _digest("screening-evidence"),
        "candidate_count": 7,
        "evaluated_count": 5,
        "bh_pass_count": 2,
        "compute_failed_count": 2,
    }


def _codex_s1_status(artifact: str = "codex-s1") -> dict[str, object]:
    return {
        "schema_version": CODEX_S1_STATUS_SCHEMA_VERSION,
        "stage": "CodexS1",
        "status": "passed",
        "verified": True,
        "artifact_sha256": _digest(artifact),
    }


def _codex_ic_status() -> dict[str, object]:
    return {
        "schema_version": CODEX_IC_STATUS_SCHEMA_VERSION,
        "stage": "CodexIC",
        "status": "passed",
        "verified": True,
        "artifact_sha256": _digest("codex-ic"),
    }


def _replay_status() -> dict[str, object]:
    return {
        "schema_version": REPLAY_STATUS_SCHEMA_VERSION,
        "status": "passed",
        "verified": True,
        "artifact_sha256": _digest("canonical-replay"),
    }


def _build(
    *,
    run_id: str = "factor-v4-pre-admission-test",
    codex_s1_status: dict[str, object] | None = None,
    codex_ic_status: dict[str, object] | None = None,
    replay_status: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = _screening_summary()
    return build_factor_governance_pre_admission_report_v4(
        run_id=run_id,
        screening_summary=summary,
        screening_sha256=canonical_semantic_sha256_v4(summary),
        codex_s1_status=codex_s1_status,
        codex_ic_status=codex_ic_status,
        replay_status=replay_status,
    )


def _assert_always_inert(report: dict[str, object]) -> None:
    assert report["pre_admission_passed"] is False
    assert report["candidate_registry_proposal_allowed"] is False
    assert report["registry_write_enabled"] is False
    assert report["registry_mutation_performed"] is False
    assert report["production_apply_enabled"] is False
    assert report["proposals"] == []


def test_missing_real_codex_stages_and_replay_stays_pending_codex() -> None:
    report = _build()

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["status"] == "pending_codex"
    assert report["blockers"] == [
        "codex_s1_missing",
        "codex_ic_missing",
        "replay_missing",
    ]
    assert report["evidence_supplied"] == {
        "codex_s1": False,
        "codex_ic": False,
        "replay": False,
    }
    _assert_always_inert(report)
    assert validate_factor_governance_pre_admission_report_v4(report) == report


def test_all_verified_inputs_stop_at_pending_exact_admission() -> None:
    report = _build(
        codex_s1_status=_codex_s1_status(),
        codex_ic_status=_codex_ic_status(),
        replay_status=_replay_status(),
    )

    assert report["status"] == "pending_exact_admission"
    assert report["blockers"] == ["exact_admission_api_required"]
    assert report["screening_summary"] == _screening_summary()
    _assert_always_inert(report)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", "factor-governance-codex-s1-status.v2"),
        ("schema_version", "factor-governance-codex-s1-status.v3"),
        ("schema_version", "factor-governance-codex-s1-status.retired"),
        ("stage", "placeholder"),
        ("status", "placeholder"),
        ("verified", False),
        ("verified", 1),
        ("artifact_sha256", "0" * 64),
        ("artifact_sha256", "A" * 64),
    ],
)
def test_supplied_invalid_codex_stage_is_blocked_not_pending(
    field: str,
    bad_value: object,
) -> None:
    bad_s1 = _codex_s1_status()
    bad_s1[field] = bad_value

    report = _build(codex_s1_status=bad_s1)

    assert report["status"] == "blocked"
    assert report["codex_s1_status"] is None
    assert report["evidence_supplied"]["codex_s1"] is True
    assert "codex_s1_status_invalid" in report["blockers"]
    assert "codex_ic_missing" in report["blockers"]
    _assert_always_inert(report)


def test_supplied_v3_replay_and_invalid_codex_ic_are_blocked() -> None:
    bad_ic = _codex_ic_status()
    bad_ic["schema_version"] = "factor-governance-codex-ic-status.v3"
    bad_replay = _replay_status()
    bad_replay["schema_version"] = "factor-governance-canonical-replay.v3"

    report = _build(
        codex_s1_status=_codex_s1_status(),
        codex_ic_status=bad_ic,
        replay_status=bad_replay,
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        "codex_ic_status_invalid",
        "replay_status_invalid",
    ]
    _assert_always_inert(report)


@pytest.mark.parametrize("mutation", ["schema", "hash", "counts", "extra_field"])
def test_invalid_screening_summary_or_hash_is_blocked(mutation: str) -> None:
    summary = _screening_summary()
    screening_sha = canonical_semantic_sha256_v4(summary)
    if mutation == "schema":
        summary["schema_version"] = "factor-governance-screening-summary.v3"
        screening_sha = canonical_semantic_sha256_v4(summary)
    elif mutation == "hash":
        screening_sha = _digest("wrong-summary")
    elif mutation == "counts":
        summary["compute_failed_count"] = 3
        screening_sha = canonical_semantic_sha256_v4(summary)
    else:
        summary["verified"] = True
        screening_sha = canonical_semantic_sha256_v4(summary)

    report = build_factor_governance_pre_admission_report_v4(
        run_id="invalid-screening",
        screening_summary=summary,
        screening_sha256=screening_sha,
        codex_s1_status=_codex_s1_status(),
        codex_ic_status=_codex_ic_status(),
        replay_status=_replay_status(),
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == ["screening_summary_or_hash_invalid"]
    assert report["screening_summary"] == summary
    _assert_always_inert(report)


def test_validator_rejects_tampered_allow_flags_and_report_seal() -> None:
    report = _build()
    tampered = copy.deepcopy(report)
    tampered["candidate_registry_proposal_allowed"] = True
    tampered["report_sha256"] = canonical_semantic_sha256_v4(
        {key: value for key, value in tampered.items() if key != "report_sha256"}
    )

    with pytest.raises(
        FactorGovernancePreAdmissionV4Error,
        match="candidate_registry_proposal_allowed must remain false",
    ):
        validate_factor_governance_pre_admission_report_v4(tampered)

    bad_seal = copy.deepcopy(report)
    bad_seal["report_sha256"] = _digest("wrong-report")
    with pytest.raises(FactorGovernancePreAdmissionV4Error, match="report SHA mismatch"):
        validate_factor_governance_pre_admission_report_v4(bad_seal)


def test_private_publish_is_canonical_owner_only_and_same_bytes_idempotent(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    report = _build()

    first = publish_factor_governance_pre_admission_report_v4(
        private_root=private_root,
        run_id=report["run_id"],
        expected_report_sha256="empty",
        report=report,
    )
    path = Path(first["path"])
    assert first["created"] is True
    assert set(first) == {"path", "sha256", "created"}
    assert path.name == PRE_ADMISSION_REPORT_FILENAME
    assert path.read_bytes() == canonical_json_bytes(report)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first["sha256"]
    assert stat.S_IMODE(os.lstat(private_root).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(path.parent).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600
    assert os.lstat(path).st_uid == os.getuid()
    assert os.lstat(path).st_nlink == 1
    assert stat.S_IMODE(os.lstat(path.parent / ".lock").st_mode) == 0o600

    second = publish_factor_governance_pre_admission_report_v4(
        private_root=private_root,
        run_id=report["run_id"],
        expected_report_sha256=first["sha256"],
        report=report,
    )
    assert second == {"path": str(path), "sha256": first["sha256"], "created": False}


def test_publish_enforces_cas_and_rejects_different_bytes(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    report = _build()
    with pytest.raises(StateConflictError, match="CAS mismatch"):
        publish_factor_governance_pre_admission_report_v4(
            private_root=private_root,
            run_id=report["run_id"],
            expected_report_sha256=_digest("missing-state"),
            report=report,
        )

    first = publish_factor_governance_pre_admission_report_v4(
        private_root=private_root,
        run_id=report["run_id"],
        expected_report_sha256="empty",
        report=report,
    )
    with pytest.raises(StateConflictError, match="CAS mismatch"):
        publish_factor_governance_pre_admission_report_v4(
            private_root=private_root,
            run_id=report["run_id"],
            expected_report_sha256=_digest("wrong-existing-state"),
            report=report,
        )

    different = _build(codex_s1_status=_codex_s1_status("different-s1"))
    with pytest.raises(DifferentBytesError, match="different bytes"):
        publish_factor_governance_pre_admission_report_v4(
            private_root=private_root,
            run_id=different["run_id"],
            expected_report_sha256=first["sha256"],
            report=different,
        )


def test_publish_rejects_symlink_root_and_symlink_target_without_clobber(
    tmp_path: Path,
) -> None:
    report = _build()
    real_root = tmp_path / "real-private"
    real_root.mkdir(mode=0o700)
    linked_root = tmp_path / "linked-private"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ProtocolError, match="root cannot be a symlink"):
        publish_factor_governance_pre_admission_report_v4(
            private_root=linked_root,
            run_id=report["run_id"],
            expected_report_sha256="empty",
            report=report,
        )

    run_dir = real_root / report["run_id"]
    run_dir.mkdir(mode=0o700)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"do-not-touch\n")
    os.chmod(outside, 0o600)
    target = run_dir / PRE_ADMISSION_REPORT_FILENAME
    target.symlink_to(outside)
    with pytest.raises(ProtocolError, match="unsafe"):
        publish_factor_governance_pre_admission_report_v4(
            private_root=real_root,
            run_id=report["run_id"],
            expected_report_sha256="empty",
            report=report,
        )
    assert outside.read_bytes() == b"do-not-touch\n"


def test_publish_rejects_existing_hardlinked_report_after_exact_readback(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    report = _build()
    run_dir = private_root / report["run_id"]
    run_dir.mkdir(mode=0o700)
    source = tmp_path / "hardlink-source.json"
    source.write_bytes(canonical_json_bytes(report))
    os.chmod(source, 0o600)
    target = run_dir / PRE_ADMISSION_REPORT_FILENAME
    try:
        os.link(source, target)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(FactorGovernancePreAdmissionV4Error, match="one hard link"):
        publish_factor_governance_pre_admission_report_v4(
            private_root=private_root,
            run_id=report["run_id"],
            expected_report_sha256=expected,
            report=report,
        )
    assert os.lstat(source).st_nlink == 2
    assert source.read_bytes() == canonical_json_bytes(report)


def test_publish_preserves_preexisting_screening_and_ignores_latest_decoys(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    report = _build()
    run_dir = private_root / report["run_id"]
    run_dir.mkdir(mode=0o700)
    screening = run_dir / "screening_evidence.v4.json"
    screening.write_bytes(b'{"screening":"preserve"}\n')
    os.chmod(screening, 0o600)
    latest = private_root / "_latest.json"
    latest.write_bytes(b"not-json-and-never-read")
    os.chmod(latest, 0o600)
    before_screening = screening.read_bytes()
    before_latest = latest.read_bytes()

    result = publish_factor_governance_pre_admission_report_v4(
        private_root=private_root,
        run_id=report["run_id"],
        expected_report_sha256="empty",
        report=report,
    )

    assert screening.read_bytes() == before_screening
    assert latest.read_bytes() == before_latest
    assert {item.name for item in run_dir.iterdir()} == {
        ".lock",
        "screening_evidence.v4.json",
        PRE_ADMISSION_REPORT_FILENAME,
    }
    assert Path(result["path"]).parent == run_dir
