from __future__ import annotations

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "build_factor_v4_1_operator_runtime_equivalence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_factor_v4_1_operator_runtime_equivalence_under_test",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

DISCOVERY_ROOT = (
    REPOSITORY_ROOT
    / "reports/factor_governance/private/v4_1_cycle"
    / "factor_v4_1_discovery_20260718T170345Z"
)
FORMAL_ROOT = (
    REPOSITORY_ROOT
    / "reports/factor_governance/private/v4_1_formal_catalog"
    / "factor_v4_1_formal_catalog_20260718T191045Z"
)
NO_LABEL_ROOT = (
    REPOSITORY_ROOT
    / "reports/factor_governance/private/v4_1_no_label_diagnostic"
    / "factor_v4_1_no_label_diagnostic_20260718T204202Z"
)

INPUT_PATHS = {
    "aquant_source_receipt": DISCOVERY_ROOT / "aquant_source_receipt.v4_1.json",
    "candidate_catalog": FORMAL_ROOT / "candidate_catalog.v4.json",
    "formal_catalog_readback": (
        FORMAL_ROOT / "formal_catalog_materialization_readback.v4_1.json"
    ),
    "no_label_operator_profile": (
        NO_LABEL_ROOT / "no_label_operator_profile.v4_1.json"
    ),
    "no_label_readback": NO_LABEL_ROOT / "no_label_diagnostic_readback.v4_1.json",
    "no_label_signal_diagnostic": (
        NO_LABEL_ROOT / "no_label_signal_diagnostic.v4_1.json"
    ),
    "primitive_mapping_proof": FORMAL_ROOT / "primitive_mapping_proof.v4_1.json",
    "source_idea_audit": DISCOVERY_ROOT / "source_idea_audit.v4_1.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_file(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _require_actual_inputs() -> None:
    if any(not path.is_file() for path in INPUT_PATHS.values()):
        pytest.skip("private v4.1 reference artifacts are unavailable")
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(WORKSPACE_ROOT),
            "cat-file",
            "-t",
            runner.evaluator.PINNED_COMMIT,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or result.stdout != b"commit\n":
        pytest.skip("pinned A_quant Git object is unavailable")


def _args(tmp_path: Path, *, run_id: str) -> Namespace:
    _require_actual_inputs()
    private_root = (
        tmp_path.resolve()
        / "reports/factor_governance/private/v4_1_operator_runtime_equivalence"
    )
    private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    args = Namespace(
        repository_root=str(REPOSITORY_ROOT),
        aquant_git_root=str(WORKSPACE_ROOT),
        aquant_pinned_commit=runner.evaluator.PINNED_COMMIT,
        private_root=str(private_root),
        run_id=run_id,
        cycle_id="cn_full_a_v4_1_20260717",
    )
    for binding_id, path in INPUT_PATHS.items():
        stem = str(runner.INPUT_SPECS[binding_id]["argument"])
        setattr(args, f"{stem}_path", str(path))
        setattr(args, f"expected_{stem}_sha256", _sha(path))
    code_paths = {
        expected_argument: REPOSITORY_ROOT / relative_path
        for relative_path, expected_argument in runner.CODE_SPECS.values()
    }
    for expected_argument, path in code_paths.items():
        setattr(args, expected_argument, _sha(path))
    return args


def test_actual_inputs_cross_bind_and_publish_nonauthorizing_bundle(
    tmp_path: Path,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("private no-clobber publication is Darwin-only")
    args = _args(tmp_path, run_id="operator_equivalence_test_20260718T010000Z")

    result = runner.run(args)

    assert result["accepted"] is True
    assert result["operator_runtime_equivalence_verified"] is True
    assert result["signal_computability_proven"] is False
    assert result["new_risk_authorized"] is False
    assert all(value is False for value in result["side_effects"].values())


def test_code_byte_drift_is_rejected_before_build(tmp_path: Path) -> None:
    args = _args(tmp_path, run_id="operator_equivalence_code_drift")
    args.expected_builder_sha256 = "0" * 64

    with pytest.raises(
        runner.FactorV4_1OperatorEquivalenceRunnerError,
        match="code binding SHA mismatch",
    ):
        runner._read_code_bindings(args, REPOSITORY_ROOT)


def test_code_binding_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    linked = tmp_path / "linked.py"
    linked.symlink_to(target)

    with pytest.raises(
        runner.FactorV4_1OperatorEquivalenceRunnerError,
        match="descriptor open failed",
    ):
        runner._read_code(linked, _sha(target))


def test_formal_readback_cannot_forge_catalog_byte_binding(tmp_path: Path) -> None:
    args = _args(tmp_path, run_id="operator_equivalence_formal_forgery")
    forged_parent = tmp_path / "forged_formal"
    forged_parent.mkdir(mode=0o700)
    forged_path = forged_parent / "formal_catalog_materialization_readback.v4_1.json"
    payload = json.loads(INPUT_PATHS["formal_catalog_readback"].read_text())
    for row in payload["artifact_bindings"]:
        if row["filename"] == "candidate_catalog.v4.json":
            row["byte_sha256"] = "0" * 64
    payload.pop("report_semantic_sha256")
    payload["report_semantic_sha256"] = runner.evaluator.semantic_sha256_v4_1(
        payload
    )
    raw = _canonical_file(payload)
    forged_path.write_bytes(raw)
    forged_path.chmod(0o600)
    args.formal_catalog_readback_path = str(forged_path)
    args.expected_formal_catalog_readback_sha256 = hashlib.sha256(raw).hexdigest()

    values, bindings = runner._read_inputs(args)
    with pytest.raises(
        runner.FactorV4_1OperatorEquivalenceRunnerError,
        match="does not bind exact candidate_catalog",
    ):
        runner._validate_cross_artifact_bindings(values, bindings)


def test_wrong_aquant_commit_is_rejected_without_git_fallback(tmp_path: Path) -> None:
    _require_actual_inputs()

    with pytest.raises(
        runner.FactorV4_1OperatorEquivalenceRunnerError,
        match="pinned commit differs",
    ):
        runner._read_pinned_sources(WORKSPACE_ROOT, "0" * 40)
