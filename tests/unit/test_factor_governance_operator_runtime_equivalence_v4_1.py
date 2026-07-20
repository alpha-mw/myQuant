from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator
from quant_investor.factors import governance_operator_runtime_equivalence_v4_1 as subject


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob(path: str) -> bytes:
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(WORKSPACE_ROOT),
            "cat-file",
            "blob",
            f"{evaluator.PINNED_COMMIT}:{path}",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("pinned A_quant Git object is unavailable")
    return result.stdout


def _semantic_field(binding_id: str) -> str:
    return {
        "aquant_source_receipt": "receipt_semantic_sha256",
        "source_idea_audit": "audit_semantic_sha256",
        "primitive_mapping_proof": "proof_semantic_sha256",
        "candidate_catalog": "semantic_sha256",
        "formal_catalog_readback": "report_semantic_sha256",
        "no_label_operator_profile": "operator_profile_semantic_sha256",
        "no_label_signal_diagnostic": "diagnostic_semantic_sha256",
        "no_label_readback": "report_semantic_sha256",
    }[binding_id]


def _bound_inputs() -> tuple[list[dict], list[dict], bytes, bytes, list[dict]]:
    paths = {
        "aquant_source_receipt": DISCOVERY_ROOT / "aquant_source_receipt.v4_1.json",
        "source_idea_audit": DISCOVERY_ROOT / "source_idea_audit.v4_1.json",
        "primitive_mapping_proof": FORMAL_ROOT / "primitive_mapping_proof.v4_1.json",
        "candidate_catalog": FORMAL_ROOT / "candidate_catalog.v4.json",
        "formal_catalog_readback": (
            FORMAL_ROOT / "formal_catalog_materialization_readback.v4_1.json"
        ),
        "no_label_operator_profile": NO_LABEL_ROOT / "no_label_operator_profile.v4_1.json",
        "no_label_signal_diagnostic": NO_LABEL_ROOT / "no_label_signal_diagnostic.v4_1.json",
        "no_label_readback": NO_LABEL_ROOT / "no_label_diagnostic_readback.v4_1.json",
    }
    if any(not path.is_file() for path in paths.values()):
        pytest.skip("private v4.1 reference bundles are unavailable")
    values = {binding_id: _read_json(path) for binding_id, path in paths.items()}
    ideas = evaluator.bind_pinned_source_ideas_v4_1(
        source_receipt=values["aquant_source_receipt"],
        source_idea_audit=values["source_idea_audit"],
        primitive_mapping_proof=values["primitive_mapping_proof"],
        formal_catalog=values["candidate_catalog"],
    )
    input_bindings = []
    for binding_id in subject.REQUIRED_INPUT_BINDING_IDS:
        path = paths[binding_id]
        raw = path.read_bytes()
        input_bindings.append(
            {
                "binding_id": binding_id,
                "absolute_path": str(path),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "semantic_sha256": values[binding_id][_semantic_field(binding_id)],
            }
        )
    code_paths = {
        "build_factor_v4_1_operator_runtime_equivalence.py": (
            REPOSITORY_ROOT / "scripts/build_factor_v4_1_operator_runtime_equivalence.py"
        ),
        "governance_aquant_no_label_eval_v4_1.py": Path(evaluator.__file__).resolve(),
        "governance_operator_runtime_equivalence_v4_1.py": Path(subject.__file__).resolve(),
        "governance_private_bundle_io.py": (
            REPOSITORY_ROOT
            / "quant_investor/factors/governance_private_bundle_io.py"
        ),
    }
    if any(not path.is_file() for path in code_paths.values()):
        pytest.skip("operator proof code inventory is incomplete")
    code_bindings = []
    for binding_id in subject.REQUIRED_CODE_BINDING_IDS:
        path = code_paths[binding_id]
        raw = path.read_bytes()
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
        code_bindings.append(
            {
                "binding_id": binding_id,
                "absolute_path": str(path),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "ast_sha256": hashlib.sha256(
                    ast.dump(tree, include_attributes=True).encode("utf-8")
                ).hexdigest(),
            }
        )
    return (
        ideas,
        input_bindings,
        _git_blob(subject.EXPRESSION_SOURCE_PATH),
        _git_blob(subject.OPERATORS_SOURCE_PATH),
        code_bindings,
    )


def _proof() -> dict:
    ideas, inputs, expression_source, operators_source, code = _bound_inputs()
    return subject.build_operator_runtime_equivalence_proof_v4_1(
        cycle_id=subject.EXPECTED_CYCLE_ID,
        bound_ideas=ideas,
        expression_source=expression_source,
        operators_source=operators_source,
        input_bindings=inputs,
        code_bindings=code,
    )


def _reseal(value: dict, field: str) -> dict:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    payload[field] = evaluator.semantic_sha256_v4_1(payload)
    return payload


def test_exact_37_pinned_runtime_differential_is_nonauthorizing() -> None:
    proof = _proof()

    assert proof["operator_runtime_equivalence_verified"] is True
    assert proof["raw_reference_divergence_count"] > 0
    assert len(proof["rows"]) == 37
    assert all(row["match"] is True for row in proof["rows"])
    assert all(row["local_outside_mask_non_nan_count"] == 0 for row in proof["rows"])
    assert proof["operator_probe_count"] == len(subject.OPERATOR_PROBE_EXPRESSIONS)
    assert proof["reference_inf_count"] > 0
    assert 200 in proof["ts_mean_windows"]
    assert all(row["match"] is True for row in proof["operator_probes"])
    assert proof["signal_computability_proven"] is False
    assert proof["screening_authority"] is False
    assert proof["registry_authority"] is False
    assert proof["new_risk_authorized"] is False
    assert proof["side_effects"] == subject.SIDE_EFFECT_FIELDS


def test_pinned_source_byte_drift_is_rejected() -> None:
    expression_source = _git_blob(subject.EXPRESSION_SOURCE_PATH)
    operators_source = _git_blob(subject.OPERATORS_SOURCE_PATH)

    with pytest.raises(
        subject.FactorGovernanceOperatorRuntimeEquivalenceV4_1Error,
        match="source SHA mismatch",
    ):
        subject.load_pinned_runtime_v4_1(
            expression_source=expression_source + b"\n",
            operators_source=operators_source,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value.update(signal_computability_proven=True),
            "authority mismatch",
        ),
        (
            lambda value: value.update(raw_reference_divergence_count=0),
            "divergence accounting mismatch",
        ),
        (
            lambda value: value["rows"].reverse(),
            "candidate order mismatch",
        ),
        (
            lambda value: value["source_bindings"][0].update(raw_sha256="0" * 64),
            "source binding identity mismatch",
        ),
        (
            lambda value: value["code_bindings"].pop(),
            "binding inventory mismatch",
        ),
        (
            lambda value: value["input_bindings"][0].update(
                absolute_path="/tmp/forged-input.json",
                byte_sha256="0" * 64,
                semantic_sha256="1" * 64,
            ),
            "binding identity mismatch",
        ),
        (
            lambda value: value["code_bindings"][0].update(
                absolute_path="/tmp/forged-code.py",
                byte_sha256="0" * 64,
                ast_sha256="1" * 64,
            ),
            "binding identity mismatch",
        ),
        (
            lambda value: value.update(pinned_commit="0" * 40),
            "identity mismatch",
        ),
        (
            lambda value: value["fixture"]["fields"][0].update(
                matrix_sha256="0" * 64
            ),
            "fixture differs from exact recomputation",
        ),
        (
            lambda value: value["operator_probes"][0].update(
                expression="close - open"
            ),
            "probe definitions mismatch",
        ),
        (
            lambda value: value["rows"][0].update(
                reference_masked_matrix_sha256="0" * 64,
                local_matrix_sha256="0" * 64,
            ),
            "differential result manifest mismatch",
        ),
    ],
)
def test_forged_or_incomplete_proof_is_rejected(mutator, message: str) -> None:
    proof = _proof()
    forged = copy.deepcopy(proof)
    forged.pop("proof_semantic_sha256")
    mutator(forged)
    forged["proof_semantic_sha256"] = evaluator.semantic_sha256_v4_1(forged)

    with pytest.raises(
        subject.FactorGovernanceOperatorRuntimeEquivalenceV4_1Error,
        match=message,
    ):
        subject.validate_operator_runtime_equivalence_proof_v4_1(forged)


def test_readback_recomputes_exact_proof_binding() -> None:
    proof = _proof()
    raw = subject.canonical_file_bytes_v4_1(proof)
    artifact_binding = [
        {
            "filename": subject.PROOF_FILENAME,
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "mode": 0o600,
            "uid": os.getuid(),
            "nlink": 1,
        }
    ]
    report = subject.build_readback_report_v4_1(
        run_id="factor_v4_1_operator_runtime_equivalence_20260719T000000Z",
        artifacts={subject.PROOF_FILENAME: proof},
        artifact_bindings=artifact_binding,
    )

    assert subject.validate_readback_report_v4_1(
        report,
        artifacts={subject.PROOF_FILENAME: proof},
        artifact_bindings=artifact_binding,
    ) == report
    assert report["operator_runtime_equivalence_verified"] is True
    assert report["signal_computability_proven"] is False


def test_forged_proof_file_binding_cannot_receive_readback() -> None:
    proof = _proof()
    raw = subject.canonical_file_bytes_v4_1(proof)
    forged_binding = [
        {
            "filename": subject.PROOF_FILENAME,
            "byte_sha256": "0" * 64,
            "size_bytes": len(raw),
            "mode": 0o600,
            "uid": os.getuid(),
            "nlink": 1,
        }
    ]

    with pytest.raises(
        subject.FactorGovernanceOperatorRuntimeEquivalenceV4_1Error,
        match="proof binding identity mismatch",
    ):
        subject.build_readback_report_v4_1(
            run_id="forged_operator_readback_binding",
            artifacts={subject.PROOF_FILENAME: proof},
            artifact_bindings=forged_binding,
        )


def test_forged_predecessor_binding_cannot_receive_readback() -> None:
    proof = _proof()
    forged = copy.deepcopy(proof)
    forged.pop("proof_semantic_sha256")
    forged["input_bindings"][0]["byte_sha256"] = "0" * 64
    forged["proof_semantic_sha256"] = evaluator.semantic_sha256_v4_1(forged)

    with pytest.raises(
        subject.FactorGovernanceOperatorRuntimeEquivalenceV4_1Error,
        match="binding identity mismatch",
    ):
        subject.build_readback_report_v4_1(
            run_id="forged_operator_equivalence",
            artifacts={subject.PROOF_FILENAME: forged},
            artifact_bindings=[],
        )
