#!/usr/bin/env python3
"""Build synthetic-only exact-five v4.4 computability evidence.

This standalone entrypoint intentionally has no real-snapshot publication
mode.  It validates the additive contract and private publication mechanics
with deterministic synthetic data under an explicitly supplied private root.
A real strict-full-A proof requires a separately reviewed future entrypoint
after a cutoff strictly later than 2026-07-19 exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors import governance_exact_five_no_label_eval_v4_4 as evaluator  # noqa: E402
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402
from quant_investor.factors import governance_signal_computability_v4_4 as contract  # noqa: E402


FIELD_RECEIPT_FILENAME = "exact_five_field_semantics_receipt.v4_4.json"
OPERATOR_RECEIPT_FILENAME = "exact_five_operator_equivalence.v4_4.json"
PROOF_FILENAME = "exact_five_signal_computability.v4_4.json"
READBACK_FILENAME = "exact_five_signal_computability_readback.v4_4.json"
INPUT_FILENAMES = (
    FIELD_RECEIPT_FILENAME,
    OPERATOR_RECEIPT_FILENAME,
    PROOF_FILENAME,
)
ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_4_signal_computability_synthetic",
)
FIELD_RECEIPT_SCHEMA = "factor-governance-exact-five-field-semantics.v4.4"
OPERATOR_RECEIPT_SCHEMA = "factor-governance-exact-five-operator-equivalence.v4.4"
BUNDLE_READBACK_SCHEMA = "factor-governance-exact-five-private-readback.v4.4"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FactorV4_4SignalComputabilityRunnerError(ValueError):
    """Raised when the synthetic-only runner fails closed."""


def _error(message: str) -> FactorV4_4SignalComputabilityRunnerError:
    return FactorV4_4SignalComputabilityRunnerError(message)


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["artifact_semantic_sha256"] = evaluator.semantic_sha256_v4_4(payload)
    return payload


def _validate_self(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = copy.deepcopy(dict(value))
    supplied = _sha(payload.pop("artifact_semantic_sha256", None), f"{label} semantic SHA")
    if supplied != evaluator.semantic_sha256_v4_4(payload):
        raise _error(f"{label} semantic SHA mismatch")
    payload["artifact_semantic_sha256"] = supplied
    return payload


def _synthetic_matrices() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    return evaluator.build_synthetic_fixture_v4_4()


def _collection_sha(
    matrices: Mapping[str, pd.DataFrame], pit: pd.DataFrame
) -> str:
    fixture = evaluator.synthetic_fixture_binding_v4_4()
    if tuple(matrices) != evaluator.INPUT_FIELDS:
        raise _error("synthetic collection input inventory/order mismatch")
    descriptors = {
        name: evaluator.matrix_hash_descriptor_v4_4(matrices[name])
        for name in evaluator.INPUT_FIELDS
    }
    pit_descriptor = evaluator.matrix_hash_descriptor_v4_4(pit.astype(float))
    if (
        descriptors != fixture["input_matrix_descriptors"]
        or pit_descriptor != fixture["pit_mask_descriptor"]
    ):
        raise _error("synthetic collection differs from the deterministic fixture")
    return contract.synthetic_fixture_collection_sha256_v4_4()


def _snapshot_binding(pit: pd.DataFrame) -> dict[str, Any]:
    descriptor = evaluator.matrix_hash_descriptor_v4_4(pit.astype(float))
    return {
        "schema_version": contract.SNAPSHOT_BINDING_SCHEMA_VERSION,
        "source_kind": "synthetic_fixture",
        "market": "CN",
        "universe": "full_a",
        "snapshot_id": "20260720T000000Z",
        "analysis_start": descriptor["date_axis"]["first"],
        "cutoff": descriptor["date_axis"]["last"],
        "latest_trade_date": descriptor["date_axis"]["last"],
        "complete_trade_date": descriptor["date_axis"]["last"],
        "full_a_count": descriptor["column_count"],
        "covered_count": descriptor["column_count"],
        "coverage_ratio": 1.0,
        "full_a_semantic_sha256": hashlib.sha256(b"synthetic-full-a").hexdigest(),
        "snapshot_manifest_sha256": hashlib.sha256(b"synthetic-snapshot").hexdigest(),
        "table_inventory_sha256": hashlib.sha256(b"synthetic-table").hexdigest(),
        "pit_membership_sha256": hashlib.sha256(b"synthetic-pit").hexdigest(),
        "pit_manifest_sha256": hashlib.sha256(b"synthetic-pit-manifest").hexdigest(),
        "date_axis_sha256": descriptor["date_axis"]["sha256"],
        "symbol_axis_sha256": descriptor["symbol_axis"]["sha256"],
        "eligibility_matrix_sha256": descriptor["matrix_sha256"],
        "pit_mask_descriptor": descriptor,
        "fallbacks": {
            "csv": False,
            "mock": False,
            "serving": False,
            "stale_pointer": False,
        },
        "strict_full_a_proven": False,
    }


def _preregistration_binding() -> dict[str, Any]:
    return {
        "schema_version": contract.PREREG_BINDING_SCHEMA_VERSION,
        "binding_scope": "synthetic_fixture",
        "cycle_id": "cn_full_a_v4_4_20260720_20260720T000000Z",
        "bundle_path": (
            "/private/tmp/synthetic-only/"
            "cn_full_a_v4_4_20260720_20260720T000000Z"
        ),
        "artifact_count": 27,
        "readback_byte_sha256": hashlib.sha256(b"synthetic-prereg-readback").hexdigest(),
        "readback_semantic_sha256": hashlib.sha256(
            b"synthetic-prereg-readback-semantic"
        ).hexdigest(),
        "candidate_rows_semantic_sha256": (
            contract.candidate_rows_semantic_sha256_v4_4()
        ),
        "existing_signal_computability": "not_run",
        "existing_authority_false": True,
        "immutable_readback_accepted": False,
    }


def build_synthetic_artifacts_v4_4() -> dict[str, dict[str, Any]]:
    passes: list[dict[str, Any]] = []
    for pass_index in (1, 2):
        matrices, pit = _synthetic_matrices()
        collection = _collection_sha(matrices, pit)
        pass_id = f"fresh_pass_{pass_index}"
        source = evaluator.evaluate_source_dag_v4_4(matrices, pit)
        local = evaluator.evaluate_local_formulas_v4_4(matrices, pit)
        passes.append(
            {
                "pass_id": pass_id,
                "collection_sha256": collection,
                "engines": [
                    evaluator.build_engine_pass_result_v4_4(
                        engine_id=evaluator.SOURCE_ENGINE_ID,
                        pass_id=pass_id,
                        collection_sha256=collection,
                        outputs=source,
                        pit_mask=pit,
                    ),
                    evaluator.build_engine_pass_result_v4_4(
                        engine_id=evaluator.LOCAL_ENGINE_ID,
                        pass_id=pass_id,
                        collection_sha256=collection,
                        outputs=local,
                        pit_mask=pit,
                    ),
                ],
            }
        )
    _, pit = _synthetic_matrices()
    snapshot = _snapshot_binding(pit)
    preregistration = _preregistration_binding()
    proof = contract.build_signal_computability_proof_v4_4(
        evidence_scope=contract.SYNTHETIC_SCOPE,
        source_bindings=contract.SOURCE_BINDINGS_V4_4,
        field_adapters=contract.FIELD_ADAPTERS_V4_4,
        snapshot_binding=snapshot,
        preregistration_binding=preregistration,
        passes=passes,
    )
    field_receipt = _seal(
        {
            "schema_version": FIELD_RECEIPT_SCHEMA,
            "protocol_version": contract.PROTOCOL_VERSION,
            "evidence_contract_version": contract.EVIDENCE_CONTRACT_VERSION,
            "evidence_scope": contract.SYNTHETIC_SCOPE,
            "source_binding_verification": (
                "CONTRACT_CONSTANTS_ONLY_NOT_GIT_REOPENED"
            ),
            "source_bindings": copy.deepcopy(contract.SOURCE_BINDINGS_V4_4),
            "field_adapters": copy.deepcopy(list(contract.FIELD_ADAPTERS_V4_4)),
            "field_adapters_semantic_sha256": (
                contract.field_adapters_semantic_sha256_v4_4()
            ),
            "synthetic_fixture_binding": copy.deepcopy(
                proof["synthetic_fixture_binding"]
            ),
            "snapshot_binding": snapshot,
            "preregistration_binding": preregistration,
            "measurement": copy.deepcopy(contract.MEASUREMENT_FLAGS),
            "authority": copy.deepcopy(contract.AUTHORITY_FLAGS),
            "side_effects": copy.deepcopy(contract.SIDE_EFFECT_FLAGS),
        }
    )
    operator_receipt = _seal(
        {
            "schema_version": OPERATOR_RECEIPT_SCHEMA,
            "protocol_version": contract.PROTOCOL_VERSION,
            "evidence_contract_version": contract.EVIDENCE_CONTRACT_VERSION,
            "evidence_scope": contract.SYNTHETIC_SCOPE,
            "passes": copy.deepcopy(passes),
            "candidate_count": 5,
            "atomic_exact_five_passed": True,
            "independent_engine_equivalence_proven": True,
            "double_collection_reproducibility_proven": True,
            "statistics_run": False,
            "authority": copy.deepcopy(contract.AUTHORITY_FLAGS),
            "side_effects": copy.deepcopy(contract.SIDE_EFFECT_FLAGS),
        }
    )
    return {
        FIELD_RECEIPT_FILENAME: field_receipt,
        OPERATOR_RECEIPT_FILENAME: operator_receipt,
        PROOF_FILENAME: proof,
    }


def _validate_field_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_self(value, "field-semantics receipt")
    if set(payload) != {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "evidence_scope",
        "source_binding_verification",
        "source_bindings",
        "field_adapters",
        "field_adapters_semantic_sha256",
        "synthetic_fixture_binding",
        "snapshot_binding",
        "preregistration_binding",
        "measurement",
        "authority",
        "side_effects",
        "artifact_semantic_sha256",
    }:
        raise _error("field-semantics receipt fields are not exact")
    if (
        payload.get("schema_version") != FIELD_RECEIPT_SCHEMA
        or payload.get("protocol_version") != contract.PROTOCOL_VERSION
        or payload.get("evidence_contract_version")
        != contract.EVIDENCE_CONTRACT_VERSION
        or payload.get("evidence_scope") != contract.SYNTHETIC_SCOPE
        or payload.get("source_binding_verification")
        != "CONTRACT_CONSTANTS_ONLY_NOT_GIT_REOPENED"
        or payload.get("source_bindings") != contract.SOURCE_BINDINGS_V4_4
        or payload.get("field_adapters") != list(contract.FIELD_ADAPTERS_V4_4)
        or payload.get("field_adapters_semantic_sha256")
        != contract.field_adapters_semantic_sha256_v4_4()
        or payload.get("synthetic_fixture_binding")
        != evaluator.synthetic_fixture_binding_v4_4()
        or payload.get("measurement") != contract.MEASUREMENT_FLAGS
        or payload.get("authority") != contract.AUTHORITY_FLAGS
        or payload.get("side_effects") != contract.SIDE_EFFECT_FLAGS
    ):
        raise _error("field-semantics receipt contract mismatch")
    contract.validate_snapshot_binding_v4_4(
        payload.get("snapshot_binding"), evidence_scope=contract.SYNTHETIC_SCOPE
    )
    contract.validate_preregistration_binding_v4_4(
        payload.get("preregistration_binding")
    )
    return payload


def _validate_operator_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_self(value, "operator-equivalence receipt")
    if set(payload) != {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "evidence_scope",
        "passes",
        "candidate_count",
        "atomic_exact_five_passed",
        "independent_engine_equivalence_proven",
        "double_collection_reproducibility_proven",
        "statistics_run",
        "authority",
        "side_effects",
        "artifact_semantic_sha256",
    }:
        raise _error("operator-equivalence receipt fields are not exact")
    if (
        payload.get("schema_version") != OPERATOR_RECEIPT_SCHEMA
        or payload.get("protocol_version") != contract.PROTOCOL_VERSION
        or payload.get("evidence_contract_version")
        != contract.EVIDENCE_CONTRACT_VERSION
        or payload.get("evidence_scope") != contract.SYNTHETIC_SCOPE
        or payload.get("candidate_count") != 5
        or payload.get("atomic_exact_five_passed") is not True
        or payload.get("independent_engine_equivalence_proven") is not True
        or payload.get("double_collection_reproducibility_proven") is not True
        or payload.get("statistics_run") is not False
        or payload.get("authority") != contract.AUTHORITY_FLAGS
        or payload.get("side_effects") != contract.SIDE_EFFECT_FLAGS
        or not isinstance(payload.get("passes"), list)
        or len(payload["passes"]) != 2
    ):
        raise _error("operator-equivalence receipt contract mismatch")
    for index, pass_value in enumerate(payload["passes"], start=1):
        if pass_value.get("pass_id") != f"fresh_pass_{index}":
            raise _error("operator-equivalence pass identity mismatch")
        engines = pass_value.get("engines")
        if not isinstance(engines, list) or len(engines) != 2:
            raise _error("operator-equivalence engine inventory mismatch")
        for engine in engines:
            evaluator.validate_engine_pass_result_v4_4(engine)
    return payload


def _validate_bundle_readback(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_self(value, "private readback report")
    expected_fields = {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "run_id",
        "evidence_scope",
        "input_artifact_bindings",
        "proof_readback",
        "candidate_count",
        "strict_snapshot_signal_computability_proven",
        "readiness",
        "measurement",
        "authority",
        "side_effects",
        "artifact_semantic_sha256",
    }
    if set(payload) != expected_fields:
        raise _error("private readback report fields are not exact")
    if (
        payload["schema_version"] != BUNDLE_READBACK_SCHEMA
        or payload["protocol_version"] != contract.PROTOCOL_VERSION
        or payload["evidence_contract_version"] != contract.EVIDENCE_CONTRACT_VERSION
        or payload["evidence_scope"] != contract.SYNTHETIC_SCOPE
        or payload["candidate_count"] != 5
        or payload["strict_snapshot_signal_computability_proven"] is not False
        or payload["readiness"] != "NON_AUTHORIZING_COMPUTABILITY_ONLY"
        or payload["measurement"] != contract.MEASUREMENT_FLAGS
        or payload["authority"] != contract.AUTHORITY_FLAGS
        or payload["side_effects"] != contract.SIDE_EFFECT_FLAGS
    ):
        raise _error("private readback report contract mismatch")
    bindings = payload["input_artifact_bindings"]
    if not isinstance(bindings, list) or [row.get("filename") for row in bindings] != list(
        INPUT_FILENAMES
    ):
        raise _error("private readback input inventory/order mismatch")
    for row in bindings:
        if set(row) != {
            "filename",
            "byte_sha256",
            "semantic_sha256",
            "size_bytes",
            "mode",
            "uid",
            "nlink",
        }:
            raise _error("private readback binding fields are not exact")
        _sha(row["byte_sha256"], "private readback byte SHA")
        _sha(row["semantic_sha256"], "private readback semantic SHA")
        if (
            type(row["size_bytes"]) is not int
            or row["size_bytes"] <= 0
            or row["mode"] != 0o600
            or row["uid"] != os.getuid()
            or row["nlink"] != 1
        ):
            raise _error("private readback binding is not owner 0600/nlink1")
    if not isinstance(payload["proof_readback"], Mapping):
        raise _error("private readback proof binding is missing")
    return payload


def _validate_artifact(filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if filename == FIELD_RECEIPT_FILENAME:
        return _validate_field_receipt(value)
    if filename == OPERATOR_RECEIPT_FILENAME:
        return _validate_operator_receipt(value)
    if filename == PROOF_FILENAME:
        return contract.validate_signal_computability_proof_v4_4(value)
    if filename == READBACK_FILENAME:
        return _validate_bundle_readback(value)
    raise _error(f"unexpected exact-five artifact: {filename}")


def _build_readback_report(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(artifacts) != INPUT_FILENAMES:
        raise _error("readback builder input inventory/order mismatch")
    proof = contract.validate_signal_computability_proof_v4_4(
        artifacts[PROOF_FILENAME]
    )
    if len(artifact_bindings) != len(INPUT_FILENAMES):
        raise _error("readback builder binding inventory mismatch")
    bindings: list[dict[str, Any]] = []
    for filename, raw_binding in zip(INPUT_FILENAMES, artifact_bindings, strict=True):
        binding = dict(raw_binding)
        if binding.get("filename") != filename:
            raise _error("readback builder binding order mismatch")
        bindings.append(
            {
                "filename": filename,
                "byte_sha256": binding["byte_sha256"],
                "semantic_sha256": artifacts[filename]["artifact_semantic_sha256"],
                "size_bytes": binding["size_bytes"],
                "mode": binding["mode"],
                "uid": binding["uid"],
                "nlink": binding["nlink"],
            }
        )
    return _seal(
        {
            "schema_version": BUNDLE_READBACK_SCHEMA,
            "protocol_version": contract.PROTOCOL_VERSION,
            "evidence_contract_version": contract.EVIDENCE_CONTRACT_VERSION,
            "run_id": run_id,
            "evidence_scope": contract.SYNTHETIC_SCOPE,
            "input_artifact_bindings": bindings,
            "proof_readback": contract.build_signal_computability_readback_v4_4(
                proof=proof
            ),
            "candidate_count": 5,
            "strict_snapshot_signal_computability_proven": False,
            "readiness": "NON_AUTHORIZING_COMPUTABILITY_ONLY",
            "measurement": copy.deepcopy(contract.MEASUREMENT_FLAGS),
            "authority": copy.deepcopy(contract.AUTHORITY_FLAGS),
            "side_effects": copy.deepcopy(contract.SIDE_EFFECT_FLAGS),
        }
    )


def _validate_complete(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = (*INPUT_FILENAMES, READBACK_FILENAME)
    if not isinstance(values, Mapping) or tuple(values) != expected:
        raise _error("complete exact-five bundle inventory/order mismatch")
    normalized = {
        filename: _validate_artifact(filename, values[filename]) for filename in expected
    }
    field = normalized[FIELD_RECEIPT_FILENAME]
    operator = normalized[OPERATOR_RECEIPT_FILENAME]
    proof = normalized[PROOF_FILENAME]
    if (
        field["source_bindings"] != proof["source_bindings"]
        or field["field_adapters"] != proof["field_adapters"]
        or field["synthetic_fixture_binding"]
        != proof["synthetic_fixture_binding"]
        or field["snapshot_binding"] != proof["snapshot_binding"]
        or field["preregistration_binding"] != proof["preregistration_binding"]
        or operator["passes"] != proof["passes"]
    ):
        raise _error("complete exact-five bundle cross-artifact binding mismatch")
    proof_readback = normalized[READBACK_FILENAME]["proof_readback"]
    contract.validate_signal_computability_readback_v4_4(
        proof_readback, proof=proof
    )
    return normalized


BUNDLE_CONTRACT = private_io.PrivateBundleContract(
    root_suffix=ROOT_SUFFIX,
    input_filenames=INPUT_FILENAMES,
    readback_report_filename=READBACK_FILENAME,
    canonicalize=private_io.canonical_json_file_bytes,
    validate_artifact=_validate_artifact,
    validate_complete=_validate_complete,
    build_readback_report=_build_readback_report,
)


def run_synthetic_publish(
    *,
    private_root: Path,
    run_id: str,
    _test_revalidation_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    artifacts = build_synthetic_artifacts_v4_4()
    frozen = {
        name: private_io.canonical_json_file_bytes(value)
        for name, value in artifacts.items()
    }

    def revalidate_inputs() -> None:
        if _test_revalidation_hook is not None:
            _test_revalidation_hook()
        rebuilt = build_synthetic_artifacts_v4_4()
        current = {
            name: private_io.canonical_json_file_bytes(value)
            for name, value in rebuilt.items()
        }
        if current != frozen:
            raise _error("synthetic source or input drifted under publication lock")

    published = private_io.publish_private_bundle(
        private_root=private_root,
        run_id=run_id,
        artifacts=artifacts,
        contract=BUNDLE_CONTRACT,
        revalidate_inputs=revalidate_inputs,
    )
    independent = private_io.readback_private_bundle(
        published["bundle_path"], contract=BUNDLE_CONTRACT
    )
    if independent.get("accepted") is not True:
        raise _error("independent synthetic bundle readback failed")
    report = independent["readback_report"]
    report_path = Path(independent["bundle_path"]) / READBACK_FILENAME
    report_byte_sha = independent["artifact_descriptors"][READBACK_FILENAME][
        "byte_sha256"
    ]
    return {
        "mode": "publish-synthetic",
        "accepted": True,
        "bundle_path": independent["bundle_path"],
        "readback_report_path": str(report_path),
        "readback_report_byte_sha256": report_byte_sha,
        "readback_report_semantic_sha256": report["artifact_semantic_sha256"],
        "evidence_scope": contract.SYNTHETIC_SCOPE,
        "strict_snapshot_signal_computability_proven": False,
        "authority": copy.deepcopy(contract.AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(contract.SIDE_EFFECT_FLAGS),
    }


def run_readback(
    *, bundle_path: Path, expected_byte_sha256: str, expected_semantic_sha256: str
) -> dict[str, Any]:
    expected_byte = _sha(expected_byte_sha256, "expected readback byte SHA")
    expected_semantic = _sha(
        expected_semantic_sha256, "expected readback semantic SHA"
    )
    result = private_io.readback_private_bundle(bundle_path, contract=BUNDLE_CONTRACT)
    report = result["readback_report"]
    actual_byte = result["artifact_descriptors"][READBACK_FILENAME]["byte_sha256"]
    if actual_byte != expected_byte:
        raise _error("historical readback byte SHA mismatch")
    if report["artifact_semantic_sha256"] != expected_semantic:
        raise _error("historical readback semantic SHA mismatch")
    return {
        "mode": "readback",
        "accepted": True,
        "bundle_path": result["bundle_path"],
        "readback_report_byte_sha256": expected_byte,
        "readback_report_semantic_sha256": expected_semantic,
        "evidence_scope": contract.SYNTHETIC_SCOPE,
        "strict_snapshot_signal_computability_proven": False,
        "authority": copy.deepcopy(contract.AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(contract.SIDE_EFFECT_FLAGS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser(
        "publish-synthetic",
        help="publish deterministic synthetic evidence beneath an explicit private root",
    )
    publish.add_argument("--private-root", required=True)
    publish.add_argument("--run-id", required=True)
    readback = commands.add_parser(
        "readback", help="reopen one explicit immutable synthetic evidence bundle"
    )
    readback.add_argument("--bundle-path", required=True)
    readback.add_argument("--expected-readback-report-byte-sha256", required=True)
    readback.add_argument("--expected-readback-report-semantic-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "publish-synthetic":
            result = run_synthetic_publish(
                private_root=Path(args.private_root), run_id=args.run_id
            )
        else:
            result = run_readback(
                bundle_path=Path(args.bundle_path),
                expected_byte_sha256=args.expected_readback_report_byte_sha256,
                expected_semantic_sha256=args.expected_readback_report_semantic_sha256,
            )
    except Exception as exc:
        print(json.dumps({"accepted": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
