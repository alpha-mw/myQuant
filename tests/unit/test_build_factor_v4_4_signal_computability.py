from __future__ import annotations

import copy
from pathlib import Path

import pytest

from quant_investor.factors import governance_private_bundle_io as private_io
from quant_investor.factors import governance_signal_computability_v4_4 as contract
from scripts import build_factor_v4_4_signal_computability as subject


def _root(tmp_path: Path) -> Path:
    value = tmp_path.joinpath(*subject.ROOT_SUFFIX)
    value.mkdir(parents=True, mode=0o700)
    value.chmod(0o700)
    return value


def test_synthetic_builder_emits_exact_three_non_authorizing_artifacts() -> None:
    artifacts = subject.build_synthetic_artifacts_v4_4()
    assert tuple(artifacts) == subject.INPUT_FILENAMES
    field = subject._validate_field_receipt(artifacts[subject.FIELD_RECEIPT_FILENAME])
    operator = subject._validate_operator_receipt(
        artifacts[subject.OPERATOR_RECEIPT_FILENAME]
    )
    proof = contract.validate_signal_computability_proof_v4_4(
        artifacts[subject.PROOF_FILENAME]
    )
    assert field["evidence_scope"] == contract.SYNTHETIC_SCOPE
    assert operator["statistics_run"] is False
    assert proof["strict_snapshot_signal_computability_proven"] is False
    assert not any(proof["authority"].values())
    assert not any(proof["side_effects"].values())


def test_temp_private_publish_and_explicit_hash_readback(tmp_path: Path) -> None:
    root = _root(tmp_path)
    result = subject.run_synthetic_publish(
        private_root=root, run_id="synthetic_exact_five_001"
    )
    assert result["accepted"] is True
    assert result["evidence_scope"] == contract.SYNTHETIC_SCOPE
    assert result["strict_snapshot_signal_computability_proven"] is False
    assert not any(result["authority"].values())
    assert not any(result["side_effects"].values())
    bundle = Path(result["bundle_path"])
    assert bundle.parent == root
    assert bundle.stat().st_mode & 0o777 == 0o700
    for path in bundle.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.stat().st_nlink == 1
    reopened = subject.run_readback(
        bundle_path=bundle,
        expected_byte_sha256=result["readback_report_byte_sha256"],
        expected_semantic_sha256=result["readback_report_semantic_sha256"],
    )
    assert reopened["accepted"] is True
    assert reopened["bundle_path"] == str(bundle)


def test_no_clobber_rejects_duplicate_run_id(tmp_path: Path) -> None:
    root = _root(tmp_path)
    subject.run_synthetic_publish(private_root=root, run_id="same-run")
    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="already exists",
    ):
        subject.run_synthetic_publish(private_root=root, run_id="same-run")


def test_locked_revalidation_drift_leaves_no_final_or_staging_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    original = subject.build_synthetic_artifacts_v4_4
    calls = 0

    def drifting() -> dict:
        nonlocal calls
        calls += 1
        values = original()
        if calls > 1:
            values[subject.FIELD_RECEIPT_FILENAME] = copy.deepcopy(
                values[subject.FIELD_RECEIPT_FILENAME]
            )
            values[subject.FIELD_RECEIPT_FILENAME]["artifact_semantic_sha256"] = "0" * 64
        return values

    monkeypatch.setattr(subject, "build_synthetic_artifacts_v4_4", drifting)
    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="input revalidation failed",
    ):
        subject.run_synthetic_publish(private_root=root, run_id="drift-run")
    assert not (root / "drift-run").exists()
    assert not any(path.name.startswith(".staging") for path in root.iterdir())


def test_private_root_suffix_and_mode_are_enforced(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong"
    wrong.mkdir(mode=0o700)
    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="private suffix",
    ):
        subject.run_synthetic_publish(private_root=wrong, run_id="bad-root")

    root = _root(tmp_path)
    root.chmod(0o755)
    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="mode must be 0700",
    ):
        subject.run_synthetic_publish(private_root=root, run_id="bad-mode")


def test_wrong_readback_hash_and_authority_tamper_fail_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    result = subject.run_synthetic_publish(private_root=root, run_id="readback-hash")
    with pytest.raises(
        subject.FactorV4_4SignalComputabilityRunnerError,
        match="byte SHA mismatch",
    ):
        subject.run_readback(
            bundle_path=Path(result["bundle_path"]),
            expected_byte_sha256="0" * 64,
            expected_semantic_sha256=result["readback_report_semantic_sha256"],
        )

    artifacts = subject.build_synthetic_artifacts_v4_4()
    field = copy.deepcopy(artifacts[subject.FIELD_RECEIPT_FILENAME])
    field["authority"]["candidate_qualified"] = True
    field.pop("artifact_semantic_sha256")
    field = subject._seal(field)
    with pytest.raises(
        subject.FactorV4_4SignalComputabilityRunnerError,
        match="field-semantics receipt contract mismatch",
    ):
        subject._validate_field_receipt(field)


def test_cli_surface_is_synthetic_only_and_has_no_outcome_or_live_arguments() -> None:
    parser = subject.build_parser()
    help_text = parser.format_help()
    assert "publish-synthetic" in help_text
    assert "readback" in help_text
    for prohibited in (
        "--label",
        "--forward-return",
        "--ic",
        "--provider",
        "--network",
        "--registry",
        "--broker",
        "--order",
        "--trade",
        "--strict-publish",
    ):
        assert prohibited not in help_text
    assert subject.ROOT_SUFFIX[-1] == "v4_4_signal_computability_synthetic"
