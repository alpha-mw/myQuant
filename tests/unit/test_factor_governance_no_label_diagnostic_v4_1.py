from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator
from quant_investor.factors import governance_no_label_diagnostic_v4_1 as subject
from quant_investor.factors import governance_private_bundle_io as private_io


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ideas() -> list[dict]:
    rows: list[dict] = []
    statuses = (
        [(subject.STATUS_SIGNAL_DIAGNOSTIC, ["close"])] * 27
        + [(subject.STATUS_TURNOVER_BLOCKED, ["turnover_rate"])] * 2
        + [(subject.STATUS_FUNDAMENTAL_BLOCKED, ["fin_roe"])] * 8
    )
    for index, (_status, fields) in enumerate(statuses):
        rows.append(
            {
                "candidate_id": f"candidate:{index:02d}",
                "name": f"candidate_{index:02d}",
                "expression": "cs_rank(close)",
                "normalized_expression_ast": evaluator.normalize_expression_ast_v4_1(
                    "cs_rank(close)"
                ),
                "input_fields": fields,
                "source_definition_sha256": _sha(f"source:{index}"),
                "full_candidate_normalized_ast_sha256": _sha(f"ast:{index}"),
                "catalog_definition_sha256": _sha(f"catalog:{index}"),
                "mapping_semantic_sha256": _sha(f"mapping:{index}"),
                "initial_weight": 0.0,
            }
        )
    return rows


def _audit() -> dict:
    runner = Path("scripts/build_factor_v4_1_no_label_diagnostic.py").resolve()
    evaluator_path = Path(evaluator.__file__).resolve()
    return subject.build_structural_no_label_audit_v4_1(
        {
            "data_builder": (str(runner), runner.read_bytes()),
            "evaluator": (str(evaluator_path), evaluator_path.read_bytes()),
        }
    )


def _binding(identifier: str, path: Path) -> dict:
    return {
        "binding_id": identifier,
        "absolute_path": str(path.resolve()),
        "byte_sha256": _sha(identifier),
    }


def _profile(tmp_path: Path) -> dict:
    return subject.build_operator_profile_v4_1(
        cycle_id="cycle_test",
        bound_ideas=_ideas(),
        source_bindings=[_binding("source", tmp_path / "source.json")],
        code_bindings=[_binding("code", tmp_path / "code.py")],
        structural_audit=_audit(),
    )


def _artifacts(tmp_path: Path) -> tuple[dict, dict]:
    profile = _profile(tmp_path)
    dates = pd.date_range("2026-01-05", periods=2)
    columns = ["000001.SZ", "000002.SZ"]
    mask = pd.DataFrame(True, index=dates, columns=columns, dtype=bool)
    rows: list[dict] = []
    for index, idea in enumerate(_ideas()):
        status = subject.classify_idea_status_v4_1(idea)
        if status == subject.STATUS_SIGNAL_DIAGNOSTIC:
            values = [[1.0, 2.0], [3.0, 4.0]]
            if index == 0:
                values[0][0] = np.inf
                values[0][1] = -np.inf
            signal = pd.DataFrame(values, index=dates, columns=columns)
            rows.append(
                subject.build_diagnostic_row_v4_1(
                    idea=idea,
                    status=status,
                    signal=signal,
                    eligibility_mask=mask,
                )
            )
        else:
            rows.append(subject.build_diagnostic_row_v4_1(idea=idea, status=status))
    descriptor = evaluator.matrix_hash_descriptor_v4_1(mask.astype(float))
    signal = subject.build_signal_diagnostic_v4_1(
        cycle_id="cycle_test",
        operator_profile=profile,
        rows=rows,
        input_bindings=[_binding("input", tmp_path / "input.json")],
        protected_stability=[
            {
                "binding_id": "protected",
                "absolute_path": str((tmp_path / "protected.json").resolve()),
                "expected_sha256": _sha("protected"),
                "before_sha256": _sha("protected"),
                "after_sha256": _sha("protected"),
            }
        ],
        market_matrix_bindings=[_binding("matrix", tmp_path / "table")],
        session_scope_binding={
            "session_count": 2,
            "pit_record_count": 2,
            "component_count": 2,
            "descriptor_semantic_sha256": _sha("scope"),
            "eligibility_matrix": descriptor,
        },
        vwap_semantic_sha256=_sha("vwap"),
    )
    return profile, signal


def _reseal(value: dict, field: str) -> dict:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    payload[field] = subject.semantic_sha256_v4_1(payload)
    return payload


def _artifact_bindings(profile: dict, signal: dict) -> list[dict]:
    rows: list[dict] = []
    for filename, value in (
        (subject.OPERATOR_PROFILE_FILENAME, profile),
        (subject.DIAGNOSTIC_FILENAME, signal),
    ):
        raw = subject.canonical_file_bytes_v4_1(value)
        rows.append(
            {
                "filename": filename,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
        )
    return rows


def _bundle_values(profile: dict, signal: dict) -> tuple[dict, list[dict]]:
    artifacts = {
        subject.OPERATOR_PROFILE_FILENAME: profile,
        subject.DIAGNOSTIC_FILENAME: signal,
    }
    bindings = _artifact_bindings(profile, signal)
    report = subject.build_readback_report_v4_1(
        run_id="run_test",
        artifacts=artifacts,
        artifact_bindings=bindings,
    )
    return {**artifacts, subject.READBACK_FILENAME: report}, bindings


def test_exact_accounting_native_division_and_all_authority_false(tmp_path: Path) -> None:
    profile, signal = _artifacts(tmp_path)
    assert profile["status_counts"] == subject.EXACT_STATUS_COUNTS
    assert signal["status_counts"] == subject.EXACT_STATUS_COUNTS
    assert profile["operator_semantics"]["binary_divide"] == {
        "implementation": "native_pandas_divide",
        "nonfinite_rewrite": False,
    }
    first = signal["rows"][0]
    assert first["finite_count"] == 2
    assert first["positive_inf_count"] == 1
    assert first["negative_inf_count"] == 1
    assert first["nan_count"] == 0
    for field in subject.AUTHORITY_FIELDS:
        assert profile[field] is False
        assert signal[field] is False
    assert all(row["initial_weight"] == 0.0 for row in signal["rows"])


@pytest.mark.parametrize(
    "family",
    [
        "forward",
        "realized",
        "target",
        "label",
        "return",
        "backtest",
        "replay",
        "registry",
        "provider",
        "execution",
    ],
)
def test_structural_auditor_rejects_each_forbidden_identifier_family(
    tmp_path: Path, family: str
) -> None:
    evaluator_source = f"import ast\ndef {family}_signal():\n    pass\n".encode()
    builder_source = b"import argparse\nx = 1\n"
    with pytest.raises(subject.FactorGovernanceNoLabelDiagnosticV4_1Error):
        subject.build_structural_no_label_audit_v4_1(
            {
                "data_builder": (str((tmp_path / "builder.py").resolve()), builder_source),
                "evaluator": (
                    str((tmp_path / "evaluator.py").resolve()),
                    evaluator_source,
                ),
            }
        )


@pytest.mark.parametrize(
    "source_text",
    [
        "import os\nx = 1\n",
        "import ast\nx = eval('1')\n",
        "import ast\nx = exec('x=1')\n",
        "import ast\nx = compile('1', '<x>', 'eval')\n",
        "import ast\nx = __import__('os')\n",
        "import ast\nx = frame.pct_change()\n",
        "import ast\nx = frame.diff()\n",
        "import ast\nx = frame.shift(-1)\n",
        "import ast\nx = raw['undeclared_column']\n",
    ],
)
def test_structural_auditor_rejects_forbidden_import_calls_and_fields(
    tmp_path: Path, source_text: str
) -> None:
    with pytest.raises(subject.FactorGovernanceNoLabelDiagnosticV4_1Error):
        subject.build_structural_no_label_audit_v4_1(
            {
                "data_builder": (
                    str((tmp_path / "builder.py").resolve()),
                    b"import argparse\nx = 1\n",
                ),
                "evaluator": (
                    str((tmp_path / "evaluator.py").resolve()),
                    source_text.encode(),
                ),
            }
        )


def test_unknown_and_missing_fields_fail_at_every_public_layer(tmp_path: Path) -> None:
    profile, signal = _artifacts(tmp_path)
    profile_drift = copy.deepcopy(profile)
    profile_drift["alpha_authority"] = True
    profile_drift = _reseal(profile_drift, "operator_profile_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceNoLabelDiagnosticV4_1Error, match="fields"):
        subject.validate_operator_profile_v4_1(profile_drift)

    profile_nested = copy.deepcopy(profile)
    profile_nested["candidate_classifications"][0]["unknown"] = False
    profile_nested = _reseal(profile_nested, "operator_profile_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceNoLabelDiagnosticV4_1Error, match="fields"):
        subject.validate_operator_profile_v4_1(profile_nested)

    row = copy.deepcopy(signal["rows"][0])
    row.pop("nan_count")
    row = _reseal(row, "row_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceNoLabelDiagnosticV4_1Error, match="fields"):
        subject.validate_diagnostic_row_v4_1(row)

    for nested_key in (
        "input_bindings",
        "protected_stability",
        "session_scope_binding",
    ):
        drift = copy.deepcopy(signal)
        if nested_key == "session_scope_binding":
            drift[nested_key]["unknown"] = 1
        else:
            drift[nested_key][0]["unknown"] = 1
        drift = _reseal(drift, "diagnostic_semantic_sha256")
        with pytest.raises(subject.FactorGovernanceNoLabelDiagnosticV4_1Error):
            subject.validate_signal_diagnostic_v4_1(drift)


def test_signal_validator_rejects_duplicate_candidate_row_after_reseal(
    tmp_path: Path,
) -> None:
    _profile_value, signal = _artifacts(tmp_path)
    drift = copy.deepcopy(signal)
    drift["rows"][1] = copy.deepcopy(drift["rows"][0])
    drift = _reseal(drift, "diagnostic_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceNoLabelDiagnosticV4_1Error,
        match="37 distinct",
    ):
        subject.validate_signal_diagnostic_v4_1(drift)


def test_bundle_validator_rejects_reordered_rows_after_reseal(tmp_path: Path) -> None:
    profile, signal = _artifacts(tmp_path)
    values, _bindings = _bundle_values(profile, signal)
    drift = copy.deepcopy(signal)
    drift["rows"][0], drift["rows"][1] = drift["rows"][1], drift["rows"][0]
    drift = _reseal(drift, "diagnostic_semantic_sha256")
    values[subject.DIAGNOSTIC_FILENAME] = drift

    with pytest.raises(
        subject.FactorGovernanceNoLabelDiagnosticV4_1Error,
        match="row alignment mismatch",
    ):
        subject.validate_bundle_values_v4_1(values)


def test_readback_validator_rejects_row_hash_drift_after_reseal(
    tmp_path: Path,
) -> None:
    profile, signal = _artifacts(tmp_path)
    values, bindings = _bundle_values(profile, signal)
    drift = copy.deepcopy(signal)
    drift["rows"][0]["source_definition_sha256"] = _sha("drifted-source")
    drift["rows"][0] = _reseal(drift["rows"][0], "row_semantic_sha256")
    drift = _reseal(drift, "diagnostic_semantic_sha256")
    artifacts = {
        subject.OPERATOR_PROFILE_FILENAME: profile,
        subject.DIAGNOSTIC_FILENAME: drift,
    }

    with pytest.raises(
        subject.FactorGovernanceNoLabelDiagnosticV4_1Error,
        match="row alignment mismatch",
    ):
        subject.validate_readback_report_v4_1(
            values[subject.READBACK_FILENAME],
            artifacts=artifacts,
            artifact_bindings=bindings,
        )


def test_private_bundle_no_clobber_permissions_symlink_and_readback(
    tmp_path: Path,
) -> None:
    profile, signal = _artifacts(tmp_path)
    artifacts = {
        subject.OPERATOR_PROFILE_FILENAME: profile,
        subject.DIAGNOSTIC_FILENAME: signal,
    }
    contract = subject.build_private_bundle_contract_v4_1(
        expected_artifacts=artifacts
    )
    private_root = tmp_path.joinpath(*subject.PRIVATE_ROOT_SUFFIX)
    private_root.mkdir(parents=True, mode=0o700)
    os.chmod(private_root, 0o700)
    published = private_io.publish_private_bundle(
        private_root=private_root,
        run_id="run_test",
        artifacts=artifacts,
        contract=contract,
        revalidate_inputs=lambda: None,
    )
    bundle = Path(published["bundle_path"])
    assert stat_mode(bundle) == 0o700
    assert all(stat_mode(path) == 0o600 for path in bundle.iterdir())
    readback = private_io.readback_private_bundle(bundle, contract=contract)
    assert readback["accepted"] is True
    with pytest.raises(private_io.FactorGovernancePrivateBundleIOError):
        private_io.publish_private_bundle(
            private_root=private_root,
            run_id="run_test",
            artifacts=artifacts,
            contract=contract,
            revalidate_inputs=lambda: None,
        )
    link = private_root / "linked_run"
    link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(private_io.FactorGovernancePrivateBundleIOError):
        private_io.readback_private_bundle(link, contract=contract)


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777
