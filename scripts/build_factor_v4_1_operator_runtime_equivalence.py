#!/usr/bin/env python3
"""Build the non-authorizing Factor v4.1 operator-equivalence proof.

The runner reads only explicitly hash-bound private artifacts and exact A_quant
Git blobs.  It does not load market data, call providers, write registries, or
grant screening, qualification, admission, production, or new-risk authority.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator
from quant_investor.factors import governance_no_label_diagnostic_v4_1 as no_label
from quant_investor.factors import (
    governance_operator_runtime_equivalence_v4_1 as equivalence,
)
from quant_investor.factors import governance_private_bundle_io as private_io


INPUT_SPECS = {
    "aquant_source_receipt": {
        "argument": "aquant_source_receipt",
        "filename": "aquant_source_receipt.v4_1.json",
        "schema": "factor-governance-aquant-source-receipt.v4.1",
        "semantic_field": "receipt_semantic_sha256",
        "cycle_bound": False,
    },
    "candidate_catalog": {
        "argument": "candidate_catalog",
        "filename": "candidate_catalog.v4.json",
        "schema": "factor-candidate-catalog.v4",
        "semantic_field": "semantic_sha256",
        "cycle_bound": False,
    },
    "formal_catalog_readback": {
        "argument": "formal_catalog_readback",
        "filename": "formal_catalog_materialization_readback.v4_1.json",
        "schema": "factor-governance-formal-catalog-materialization-readback.v4.1",
        "semantic_field": "report_semantic_sha256",
        "cycle_bound": True,
    },
    "no_label_operator_profile": {
        "argument": "no_label_operator_profile",
        "filename": no_label.OPERATOR_PROFILE_FILENAME,
        "schema": no_label.OPERATOR_PROFILE_SCHEMA_VERSION,
        "semantic_field": "operator_profile_semantic_sha256",
        "cycle_bound": True,
    },
    "no_label_readback": {
        "argument": "no_label_readback",
        "filename": no_label.READBACK_FILENAME,
        "schema": no_label.READBACK_SCHEMA_VERSION,
        "semantic_field": "report_semantic_sha256",
        "cycle_bound": True,
    },
    "no_label_signal_diagnostic": {
        "argument": "no_label_signal_diagnostic",
        "filename": no_label.DIAGNOSTIC_FILENAME,
        "schema": no_label.DIAGNOSTIC_SCHEMA_VERSION,
        "semantic_field": "diagnostic_semantic_sha256",
        "cycle_bound": True,
    },
    "primitive_mapping_proof": {
        "argument": "primitive_mapping_proof",
        "filename": "primitive_mapping_proof.v4_1.json",
        "schema": "factor-governance-primitive-mapping-proof.v4.1",
        "semantic_field": "proof_semantic_sha256",
        "cycle_bound": True,
    },
    "source_idea_audit": {
        "argument": "source_idea_audit",
        "filename": "source_idea_audit.v4_1.json",
        "schema": "factor-governance-source-idea-audit.v4.1",
        "semantic_field": "audit_semantic_sha256",
        "cycle_bound": True,
    },
}

CODE_SPECS = {
    "build_factor_v4_1_operator_runtime_equivalence.py": (
        "scripts/build_factor_v4_1_operator_runtime_equivalence.py",
        "expected_builder_sha256",
    ),
    "governance_aquant_no_label_eval_v4_1.py": (
        "quant_investor/factors/governance_aquant_no_label_eval_v4_1.py",
        "expected_evaluator_sha256",
    ),
    "governance_operator_runtime_equivalence_v4_1.py": (
        "quant_investor/factors/governance_operator_runtime_equivalence_v4_1.py",
        "expected_equivalence_module_sha256",
    ),
    "governance_private_bundle_io.py": (
        "quant_investor/factors/governance_private_bundle_io.py",
        "expected_private_io_sha256",
    ),
}

FORMAL_FALSE_AUTHORITY_FIELDS = (
    "formal_admission_authority",
    "new_risk_authorized",
    "production_apply_enabled",
    "proposal_eligible",
    "qualification",
    "registry_entry_created",
    "runtime_equivalence_verified",
    "screening_eligible",
    "signal_computability_proven",
)

_SHA256_CHARS = frozenset("0123456789abcdef")
MAX_CODE_BYTES = 8 * 1024 * 1024


class FactorV4_1OperatorEquivalenceRunnerError(ValueError):
    """Raised when the operator-equivalence runner fails closed."""


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if type(value) is not str or not value:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"{label} must be an absolute normalized path"
        )
    path = Path(value)
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"{label} must be an absolute normalized path"
        )
    return path


def _semantic(value: Mapping[str, Any], field: str, label: str) -> str:
    stored = _sha(value.get(field), f"{label}.{field}")
    payload = {
        key: copy.deepcopy(item) for key, item in value.items() if key != field
    }
    if evaluator.semantic_sha256_v4_1(payload) != stored:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"{label} semantic self-hash mismatch"
        )
    return stored


def _input_validator(
    binding_id: str,
    *,
    cycle_id: str,
):
    spec = INPUT_SPECS[binding_id]

    def validate(value: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = copy.deepcopy(dict(value))
        if payload.get("schema_version") != spec["schema"]:
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"{binding_id} schema mismatch"
            )
        _semantic(payload, str(spec["semantic_field"]), binding_id)
        if spec["cycle_bound"] and payload.get("cycle_id") != cycle_id:
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"{binding_id} cycle mismatch"
            )
        if binding_id == "no_label_operator_profile":
            return no_label.validate_operator_profile_v4_1(payload)
        if binding_id == "no_label_signal_diagnostic":
            return no_label.validate_signal_diagnostic_v4_1(payload)
        if binding_id == "formal_catalog_readback":
            if payload.get("classification_only") is not True:
                raise FactorV4_1OperatorEquivalenceRunnerError(
                    "formal catalog readback is not classification-only"
                )
            for field in FORMAL_FALSE_AUTHORITY_FIELDS:
                if payload.get(field) is not False:
                    raise FactorV4_1OperatorEquivalenceRunnerError(
                        f"formal catalog authority field is not false: {field}"
                    )
        return payload

    return validate


def _input_argument(args: argparse.Namespace, binding_id: str, suffix: str) -> Any:
    stem = str(INPUT_SPECS[binding_id]["argument"])
    if suffix == "path":
        return getattr(args, f"{stem}_path")
    if suffix == "sha256":
        return getattr(args, f"expected_{stem}_sha256")
    raise FactorV4_1OperatorEquivalenceRunnerError(
        f"unsupported input argument suffix: {suffix}"
    )


def _read_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    values: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    for binding_id in equivalence.REQUIRED_INPUT_BINDING_IDS:
        spec = INPUT_SPECS[binding_id]
        path = _absolute_path(
            _input_argument(args, binding_id, "path"),
            f"{binding_id} path",
        )
        if path.name != spec["filename"]:
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"{binding_id} filename mismatch"
            )
        read = private_io.read_private_canonical_json(
            path,
            _sha(
                _input_argument(args, binding_id, "sha256"),
                f"expected {binding_id} SHA",
            ),
            _input_validator(binding_id, cycle_id=args.cycle_id),
            canonicalizer=private_io.canonical_json_file_bytes,
        )
        value = copy.deepcopy(dict(read["value"]))
        descriptor = copy.deepcopy(dict(read["descriptor"]))
        values[binding_id] = value
        bindings.append(
            {
                "binding_id": binding_id,
                "absolute_path": descriptor["absolute_path"],
                "byte_sha256": descriptor["byte_sha256"],
                "semantic_sha256": value[str(spec["semantic_field"])],
            }
        )
    return values, bindings


def _binding_by_filename(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"{label} artifact_bindings must be a list"
        )
    result: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, Mapping):
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"{label} artifact binding must be an object"
            )
        row = copy.deepcopy(dict(raw))
        filename = row.get("filename")
        if type(filename) is not str or filename in result:
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"{label} artifact binding inventory mismatch"
            )
        result[filename] = row
    return result


def _validate_cross_artifact_bindings(
    values: Mapping[str, Mapping[str, Any]],
    input_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["binding_id"]: row for row in input_bindings}
    ideas = evaluator.bind_pinned_source_ideas_v4_1(
        source_receipt=values["aquant_source_receipt"],
        source_idea_audit=values["source_idea_audit"],
        primitive_mapping_proof=values["primitive_mapping_proof"],
        formal_catalog=values["candidate_catalog"],
    )

    formal = values["formal_catalog_readback"]
    formal_bindings = _binding_by_filename(
        formal.get("artifact_bindings"), "formal catalog readback"
    )
    for binding_id in ("primitive_mapping_proof", "candidate_catalog"):
        filename = str(INPUT_SPECS[binding_id]["filename"])
        row = formal_bindings.get(filename)
        expected = by_id[binding_id]
        if (
            row is None
            or row.get("byte_sha256") != expected["byte_sha256"]
            or row.get("semantic_sha256") != expected["semantic_sha256"]
        ):
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"formal readback does not bind exact {binding_id}"
            )
    if formal.get("formal_catalog_sha256") != by_id["candidate_catalog"][
        "semantic_sha256"
    ]:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "formal readback catalog semantic binding mismatch"
        )
    if formal.get("mapping_proof_sha256") != by_id["primitive_mapping_proof"][
        "semantic_sha256"
    ]:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "formal readback mapping-proof semantic binding mismatch"
        )

    profile = values["no_label_operator_profile"]
    diagnostic = values["no_label_signal_diagnostic"]
    readback = values["no_label_readback"]
    no_label_bindings = readback.get("artifact_bindings")
    no_label.validate_readback_report_v4_1(
        readback,
        artifacts={
            no_label.OPERATOR_PROFILE_FILENAME: profile,
            no_label.DIAGNOSTIC_FILENAME: diagnostic,
        },
        artifact_bindings=no_label_bindings,
    )
    bound_no_label = _binding_by_filename(
        no_label_bindings, "no-label readback"
    )
    for binding_id in (
        "no_label_operator_profile",
        "no_label_signal_diagnostic",
    ):
        filename = str(INPUT_SPECS[binding_id]["filename"])
        row = bound_no_label.get(filename)
        expected = by_id[binding_id]
        if row is None or row.get("byte_sha256") != expected["byte_sha256"]:
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"no-label readback does not bind exact {binding_id}"
            )
    return ideas


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _read_code_once(path: Path) -> tuple[bytes, tuple[int, ...]]:
    required_flags = ("O_CLOEXEC", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "descriptor-safe code reads are unsupported on this platform"
        )
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"code binding descriptor open failed: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > MAX_CODE_BYTES
        ):
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"code binding size/type check failed: {path}"
            )
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise FactorV4_1OperatorEquivalenceRunnerError(
                    f"code binding was truncated during read: {path}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"code binding grew during read: {path}"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_metadata = os.lstat(path)
    except OSError as exc:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"code binding path disappeared: {path}"
        ) from exc
    signature = _signature(after)
    if (
        _signature(before) != signature
        or stat.S_ISLNK(path_metadata.st_mode)
        or _signature(path_metadata) != signature
    ):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"code binding identity changed during read: {path}"
        )
    return b"".join(chunks), signature


def _read_code(path: Path, expected_sha256: str) -> bytes:
    first, first_signature = _read_code_once(path)
    second, second_signature = _read_code_once(path)
    if first_signature != second_signature or first != second:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"code binding changed across stable reads: {path}"
        )
    if hashlib.sha256(first).hexdigest() != _sha(
        expected_sha256, f"expected code SHA for {path.name}"
    ):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"code binding SHA mismatch: {path}"
        )
    try:
        ast.parse(first.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"code binding is not valid UTF-8 Python: {path}"
        ) from exc
    return first


def _read_code_bindings(
    args: argparse.Namespace,
    repository_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    if Path(__file__).resolve() != (
        repository_root / CODE_SPECS[
            "build_factor_v4_1_operator_runtime_equivalence.py"
        ][0]
    ):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "builder is not executing from the bound repository root"
        )
    rows: list[dict[str, Any]] = []
    raw_by_id: dict[str, bytes] = {}
    for binding_id in equivalence.REQUIRED_CODE_BINDING_IDS:
        relative_path, expected_argument = CODE_SPECS[binding_id]
        path = repository_root / relative_path
        raw = _read_code(path, getattr(args, expected_argument))
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
        rows.append(
            {
                "binding_id": binding_id,
                "absolute_path": str(path),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "ast_sha256": hashlib.sha256(
                    ast.dump(tree, include_attributes=True).encode("utf-8")
                ).hexdigest(),
            }
        )
        raw_by_id[binding_id] = raw
    return rows, raw_by_id


def _git(
    git_root: Path,
    arguments: Sequence[str],
    *,
    context: str,
) -> bytes:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(git_root), *arguments],
        check=False,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=15,
    )
    if result.returncode != 0:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            f"Git object read failed: {context}"
        )
    return result.stdout


def _read_pinned_sources(
    git_root: Path,
    pinned_commit: str,
) -> dict[str, bytes]:
    if pinned_commit != evaluator.PINNED_COMMIT:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "A_quant pinned commit differs from the exact v4.1 contract"
        )
    if _git(
        git_root,
        ["cat-file", "-t", pinned_commit],
        context="pinned commit type",
    ) != b"commit\n":
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "A_quant pinned object is not a commit"
        )
    values: dict[str, bytes] = {}
    for path in sorted(equivalence.PINNED_RUNTIME_SOURCE_HASHES):
        raw = _git(
            git_root,
            ["cat-file", "blob", f"{pinned_commit}:{path}"],
            context=path,
        )
        if (
            hashlib.sha256(raw).hexdigest()
            != equivalence.PINNED_RUNTIME_SOURCE_HASHES[path]
            or len(raw) != equivalence.PINNED_RUNTIME_SOURCE_SIZES[path]
        ):
            raise FactorV4_1OperatorEquivalenceRunnerError(
                f"pinned A_quant source identity mismatch: {path}"
            )
        values[path] = raw
    return values


def _revalidate_inputs(
    *,
    args: argparse.Namespace,
    expected_values: Mapping[str, Mapping[str, Any]],
    expected_input_bindings: Sequence[Mapping[str, Any]],
    repository_root: Path,
    expected_code_bindings: Sequence[Mapping[str, Any]],
    expected_code_raw: Mapping[str, bytes],
    git_root: Path,
    expected_sources: Mapping[str, bytes],
) -> None:
    values, input_bindings = _read_inputs(args)
    _validate_cross_artifact_bindings(values, input_bindings)
    if equivalence.canonical_json_bytes_v4_1(values) != (
        equivalence.canonical_json_bytes_v4_1(expected_values)
    ) or equivalence.canonical_json_bytes_v4_1(input_bindings) != (
        equivalence.canonical_json_bytes_v4_1(expected_input_bindings)
    ):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "input artifacts changed before publication"
        )
    code_bindings, code_raw = _read_code_bindings(args, repository_root)
    if equivalence.canonical_json_bytes_v4_1(code_bindings) != (
        equivalence.canonical_json_bytes_v4_1(expected_code_bindings)
    ) or any(code_raw[key] != expected_code_raw[key] for key in expected_code_raw):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "proof code changed before publication"
        )
    sources = _read_pinned_sources(git_root, args.aquant_pinned_commit)
    if any(sources[key] != expected_sources[key] for key in expected_sources):
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "pinned A_quant Git blobs changed before publication"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = _absolute_path(args.repository_root, "repository root")
    git_root = _absolute_path(args.aquant_git_root, "A_quant Git root")
    private_root = _absolute_path(args.private_root, "private root")
    if not repository_root.is_dir() or not git_root.is_dir():
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "repository and Git roots must already exist"
        )

    values, input_bindings = _read_inputs(args)
    ideas = _validate_cross_artifact_bindings(values, input_bindings)
    code_bindings, code_raw = _read_code_bindings(args, repository_root)
    sources = _read_pinned_sources(git_root, args.aquant_pinned_commit)
    proof = equivalence.build_operator_runtime_equivalence_proof_v4_1(
        cycle_id=args.cycle_id,
        bound_ideas=ideas,
        expression_source=sources[equivalence.EXPRESSION_SOURCE_PATH],
        operators_source=sources[equivalence.OPERATORS_SOURCE_PATH],
        input_bindings=input_bindings,
        code_bindings=code_bindings,
    )
    contract = equivalence.build_private_bundle_contract_v4_1(
        expected_proof=proof
    )

    def revalidate() -> None:
        _revalidate_inputs(
            args=args,
            expected_values=values,
            expected_input_bindings=input_bindings,
            repository_root=repository_root,
            expected_code_bindings=code_bindings,
            expected_code_raw=code_raw,
            git_root=git_root,
            expected_sources=sources,
        )

    published = private_io.publish_private_bundle(
        private_root=private_root,
        run_id=args.run_id,
        artifacts={equivalence.PROOF_FILENAME: proof},
        contract=contract,
        revalidate_inputs=revalidate,
    )
    independent = private_io.readback_private_bundle(
        published["bundle_path"], contract=contract
    )
    if independent.get("accepted") is not True:
        raise FactorV4_1OperatorEquivalenceRunnerError(
            "independent operator-equivalence readback failed"
        )
    descriptors = independent["artifact_descriptors"]
    return {
        "accepted": True,
        "readiness": equivalence.READINESS,
        "bundle_path": independent["bundle_path"],
        "proof_byte_sha256": descriptors[equivalence.PROOF_FILENAME][
            "byte_sha256"
        ],
        "proof_semantic_sha256": proof["proof_semantic_sha256"],
        "readback_byte_sha256": descriptors[equivalence.READBACK_FILENAME][
            "byte_sha256"
        ],
        "readback_semantic_sha256": independent["readback_report"][
            "report_semantic_sha256"
        ],
        "operator_runtime_equivalence_verified": True,
        "signal_computability_proven": False,
        "new_risk_authorized": False,
        "side_effects": dict(equivalence.SIDE_EFFECT_FIELDS),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hash-bound, classification-only Factor v4.1 "
            "operator-runtime equivalence proof."
        )
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--aquant-git-root", required=True)
    parser.add_argument("--aquant-pinned-commit", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id", required=True)
    for spec in INPUT_SPECS.values():
        option = str(spec["argument"]).replace("_", "-")
        parser.add_argument(f"--{option}-path", required=True)
        parser.add_argument(f"--expected-{option}-sha256", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--expected-equivalence-module-sha256", required=True)
    parser.add_argument("--expected-private-io-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (
        FactorV4_1OperatorEquivalenceRunnerError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            json.dumps(
                {"accepted": False, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
