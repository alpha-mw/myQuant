"""Owner-private bundle for the Factor v4.3 prior diagnostic nomination.

This wrapper publishes exactly two already-validated, diagnostic-only inputs
and one generated readback report.  It grants no admission, activation, or
production authority.  Durable, exclusive publication is delegated to the
shared Factor-governance private bundle I/O implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import importlib
import os
import re
from typing import Any

from quant_investor.factors.governance_private_bundle_io import (
    PrivateBundleContract,
    publish_private_bundle,
    readback_private_bundle,
)


ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION = (
    "reports",
    "factor_governance",
    "private",
    "v4_3_prior_diagnostic_nomination",
)
# A concise alias is useful to callers that already live in this exact lane.
ROOT_SUFFIX_V4_3 = ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION

PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3 = (
    "prior_diagnostic_runtime_binding.v4_3.json"
)
PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3 = (
    "prior_diagnostic_nomination.v4_3.json"
)
PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3 = (
    "prior_diagnostic_nomination_readback.v4_3.json"
)
READBACK_REPORT_FILENAME_V4_3 = (
    PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3
)

INPUT_FILENAMES_V4_3 = (
    PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3,
    PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3,
)

READBACK_REPORT_SCHEMA_VERSION_V4_3 = (
    "factor-governance-prior-diagnostic-nomination-readback.v4.3"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_BASE_BINDING_FIELDS = frozenset(
    {"filename", "byte_sha256", "size_bytes", "mode", "uid", "nlink"}
)
_REPORT_BINDING_FIELDS = frozenset(
    {
        "filename",
        "byte_sha256",
        "semantic_sha256",
        "size_bytes",
        "mode",
        "uid",
        "nlink",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "filename",
        "run_id",
        "publication_evidence_scope",
        "intended_destination",
        "required_commit_primitive",
        "commit_success_claimed",
        "no_clobber_success_claimed",
        "fsync_success_claimed",
        "durability_success_claimed",
        "artifact_bindings",
        "authority",
        "side_effects",
        "artifact_semantic_sha256",
    }
)


class FactorGovernancePriorDiagnosticNominationBundleV4_3Error(ValueError):
    """Raised when the v4.3 diagnostic nomination bundle fails closed."""


def _error(message: str) -> FactorGovernancePriorDiagnosticNominationBundleV4_3Error:
    return FactorGovernancePriorDiagnosticNominationBundleV4_3Error(message)


def _diagnostic() -> Any:
    """Import the pure diagnostic schema lazily while its module is developed."""

    return importlib.import_module(
        "quant_investor.factors.governance_prior_diagnostic_nomination_v4_3"
    )


def _exact(
    value: Any,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise _error(f"{label} field names must be strings")
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        raise _error(
            f"{label} fields invalid: missing={','.join(missing) or '-'};"
            f"unknown={','.join(unknown) or '-'}"
        )
    _diagnostic().canonical_file_bytes_v4_3(payload)
    return payload


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _run_id(value: Any) -> str:
    if (
        type(value) is not str
        or _SAFE_RUN_ID.fullmatch(value) is None
        or ".." in value
        or value.startswith(".")
    ):
        raise _error("run_id must be one safe deterministic path segment")
    return value


def _all_false_mapping(value: Any, label: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise _error(f"{label} must be a nonempty all-false object")
    payload = dict(value)
    if any(type(key) is not str or not key for key in payload):
        raise _error(f"{label} field names must be nonempty strings")
    if any(item is not False for item in payload.values()):
        raise _error(f"{label} must remain exactly false")
    _diagnostic().canonical_file_bytes_v4_3(payload)
    return copy.deepcopy(payload)


def _self_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "artifact_semantic_sha256"
    }


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["artifact_semantic_sha256"] = _diagnostic().semantic_sha256_v4_3(
        _self_payload(payload)
    )
    return payload


def _canonical_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    canonicalize = _diagnostic().canonical_file_bytes_v4_3
    return canonicalize(left) == canonicalize(right)


def _validate_report_binding(
    value: Any,
    *,
    index: int,
    expected_filename: str,
) -> dict[str, Any]:
    row = _exact(
        value,
        _REPORT_BINDING_FIELDS,
        f"readback artifact binding[{index}]",
    )
    if row["filename"] != expected_filename:
        raise _error("readback artifact binding inventory/order mismatch")
    _sha256(row["byte_sha256"], "readback byte SHA")
    _sha256(row["semantic_sha256"], "readback semantic SHA")
    if type(row["size_bytes"]) is not int or row["size_bytes"] <= 0:
        raise _error("readback artifact size must be a positive integer")
    if (
        row["mode"] != 0o600
        or row["uid"] != os.getuid()
        or row["nlink"] != 1
    ):
        raise _error("readback artifact binding must remain owner 0600/nlink1")
    return copy.deepcopy(row)


def validate_prior_diagnostic_nomination_readback_v4_3(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the self-sealed generated readback report in isolation."""

    payload = _exact(value, _REPORT_FIELDS, "prior diagnostic readback report")
    if (
        payload["schema_version"] != READBACK_REPORT_SCHEMA_VERSION_V4_3
        or payload["protocol_version"] != "v4"
        or payload["filename"]
        != PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3
    ):
        raise _error("prior diagnostic readback schema/protocol/filename mismatch")
    run_id = _run_id(payload["run_id"])
    if payload["publication_evidence_scope"] != "PRECOMMIT_INTENT_ONLY":
        raise _error("prior diagnostic readback may claim precommit intent only")
    if payload["intended_destination"] != {
        "root_suffix": list(ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION),
        "directory_name": run_id,
    }:
        raise _error("prior diagnostic readback intended destination mismatch")
    if payload["required_commit_primitive"] != "renameatx_np(RENAME_EXCL)":
        raise _error("exclusive Darwin commit primitive is required")
    for field in (
        "commit_success_claimed",
        "no_clobber_success_claimed",
        "fsync_success_claimed",
        "durability_success_claimed",
    ):
        if payload[field] is not False:
            raise _error(f"{field} must remain false inside staged bytes")

    bindings = payload["artifact_bindings"]
    if not isinstance(bindings, list) or len(bindings) != len(INPUT_FILENAMES_V4_3):
        raise _error("readback artifact binding inventory/order mismatch")
    normalized_bindings = [
        _validate_report_binding(
            row,
            index=index,
            expected_filename=filename,
        )
        for index, (filename, row) in enumerate(
            zip(INPUT_FILENAMES_V4_3, bindings, strict=True)
        )
    ]
    authority = _all_false_mapping(payload["authority"], "readback authority")
    side_effects = _all_false_mapping(
        payload["side_effects"], "readback side effects"
    )
    diagnostic = _diagnostic()
    if authority != diagnostic.AUTHORITY_FLAGS:
        raise _error("readback authority flags differ from the exact false contract")
    if side_effects != diagnostic.SIDE_EFFECT_FLAGS:
        raise _error("readback side-effect flags differ from the exact false contract")
    supplied = _sha256(
        payload["artifact_semantic_sha256"],
        "prior diagnostic readback self SHA",
    )
    expected = diagnostic.semantic_sha256_v4_3(_self_payload(payload))
    if supplied != expected:
        raise _error("prior diagnostic readback artifact_semantic_sha256 mismatch")
    return {
        **copy.deepcopy(payload),
        "artifact_bindings": normalized_bindings,
        "authority": authority,
        "side_effects": side_effects,
    }


def _validate_artifact(filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic = _diagnostic()
    if filename == PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3:
        return diagnostic.validate_prior_diagnostic_runtime_binding_v4_3(value)
    if filename == PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3:
        return diagnostic.validate_prior_diagnostic_nomination_v4_3(value)
    if filename == PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3:
        return validate_prior_diagnostic_nomination_readback_v4_3(value)
    raise _error(f"unknown v4.3 prior diagnostic bundle artifact: {filename}")


def validate_prior_diagnostic_nomination_bundle_inputs_v4_3(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Purely validate the exact ordered two-input diagnostic bundle."""

    if not isinstance(values, Mapping) or set(values) != set(INPUT_FILENAMES_V4_3):
        raise _error("prior diagnostic bundle input inventory mismatch")
    normalized = {
        filename: _validate_artifact(filename, values[filename])
        for filename in INPUT_FILENAMES_V4_3
    }
    runtime = normalized[PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3]
    nomination = normalized[PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3]
    if runtime["run_id"] != nomination["run_id"]:
        raise _error("prior diagnostic input run_id crosslink mismatch")
    diagnostic = _diagnostic()
    cross_validator = getattr(
        diagnostic,
        "validate_prior_diagnostic_nomination_against_runtime_v4_3",
        None,
    )
    if not callable(cross_validator):
        raise _error("prior diagnostic runtime cross-validator is unavailable")
    cross_validator(nomination, runtime)
    return normalized


def _normalized_base_bindings(
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        not isinstance(artifact_bindings, Sequence)
        or isinstance(artifact_bindings, (str, bytes, bytearray))
        or len(artifact_bindings) != len(INPUT_FILENAMES_V4_3)
    ):
        raise _error("readback builder artifact binding inventory mismatch")
    normalized: list[dict[str, Any]] = []
    for index, (filename, value) in enumerate(
        zip(INPUT_FILENAMES_V4_3, artifact_bindings, strict=True)
    ):
        row = _exact(value, _BASE_BINDING_FIELDS, f"input binding[{index}]")
        if row["filename"] != filename:
            raise _error("readback builder artifact binding order mismatch")
        _sha256(row["byte_sha256"], "input binding byte SHA")
        if type(row["size_bytes"]) is not int or row["size_bytes"] <= 0:
            raise _error("input binding size must be a positive integer")
        if (
            row["mode"] != 0o600
            or row["uid"] != os.getuid()
            or row["nlink"] != 1
        ):
            raise _error("input binding must remain owner 0600/nlink1")
        normalized.append(copy.deepcopy(row))
    return normalized


def _build_readback_report(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = validate_prior_diagnostic_nomination_bundle_inputs_v4_3(artifacts)
    runtime = normalized[PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3]
    nomination = normalized[PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3]
    if _run_id(run_id) != nomination["run_id"] or runtime["run_id"] != run_id:
        raise _error("publication directory must equal deterministic nomination run_id")
    base_bindings = _normalized_base_bindings(artifact_bindings)
    report_bindings = [
        {
            **row,
            "semantic_sha256": normalized[row["filename"]][
                "artifact_semantic_sha256"
            ],
        }
        for row in base_bindings
    ]
    report = _seal(
        {
            "schema_version": READBACK_REPORT_SCHEMA_VERSION_V4_3,
            "protocol_version": "v4",
            "filename": PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3,
            "run_id": run_id,
            "publication_evidence_scope": "PRECOMMIT_INTENT_ONLY",
            "intended_destination": {
                "root_suffix": list(
                    ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION
                ),
                "directory_name": run_id,
            },
            "required_commit_primitive": "renameatx_np(RENAME_EXCL)",
            "commit_success_claimed": False,
            "no_clobber_success_claimed": False,
            "fsync_success_claimed": False,
            "durability_success_claimed": False,
            "artifact_bindings": report_bindings,
            "authority": copy.deepcopy(nomination["authority"]),
            "side_effects": copy.deepcopy(nomination["side_effects"]),
        }
    )
    return validate_prior_diagnostic_nomination_readback_v4_3(report)


def validate_prior_diagnostic_nomination_bundle_artifacts_v4_3(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate and exactly reconstruct the complete two-input/one-report set."""

    expected = {
        *INPUT_FILENAMES_V4_3,
        PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3,
    }
    if not isinstance(values, Mapping) or set(values) != expected:
        raise _error("complete prior diagnostic bundle inventory mismatch")
    normalized_inputs = validate_prior_diagnostic_nomination_bundle_inputs_v4_3(
        {filename: values[filename] for filename in INPUT_FILENAMES_V4_3}
    )
    report = validate_prior_diagnostic_nomination_readback_v4_3(
        values[PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3]
    )
    runtime = normalized_inputs[PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3]
    nomination = normalized_inputs[PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3]
    if report["run_id"] != runtime["run_id"] or report["run_id"] != nomination[
        "run_id"
    ]:
        raise _error("readback report run_id crosslink mismatch")
    if report["authority"] != nomination["authority"]:
        raise _error("readback report authority crosslink mismatch")
    if report["side_effects"] != nomination["side_effects"]:
        raise _error("readback report side-effect crosslink mismatch")

    by_name = {row["filename"]: row for row in report["artifact_bindings"]}
    for filename in INPUT_FILENAMES_V4_3:
        raw = _diagnostic().canonical_file_bytes_v4_3(normalized_inputs[filename])
        binding = by_name[filename]
        if (
            binding["byte_sha256"] != hashlib.sha256(raw).hexdigest()
            or binding["size_bytes"] != len(raw)
        ):
            raise _error(f"readback byte binding mismatch: {filename}")
        if binding["semantic_sha256"] != normalized_inputs[filename][
            "artifact_semantic_sha256"
        ]:
            raise _error(f"readback semantic binding mismatch: {filename}")

    base_bindings = [
        {key: row[key] for key in _BASE_BINDING_FIELDS}
        for row in report["artifact_bindings"]
    ]
    rebuilt = _build_readback_report(
        run_id=report["run_id"],
        artifacts=normalized_inputs,
        artifact_bindings=base_bindings,
    )
    if not _canonical_equal(report, rebuilt):
        raise _error("readback report deterministic reconstruction mismatch")
    return {
        **normalized_inputs,
        PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3: report,
    }


def prior_diagnostic_nomination_bundle_contract_v4_3() -> PrivateBundleContract:
    return PrivateBundleContract(
        root_suffix=ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION,
        input_filenames=INPUT_FILENAMES_V4_3,
        readback_report_filename=(
            PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3
        ),
        canonicalize=_diagnostic().canonical_file_bytes_v4_3,
        validate_artifact=_validate_artifact,
        validate_complete=validate_prior_diagnostic_nomination_bundle_artifacts_v4_3,
        build_readback_report=_build_readback_report,
    )


def publish_prior_diagnostic_nomination_bundle_v4_3(
    *,
    private_root: str | os.PathLike[str],
    artifacts: Mapping[str, Mapping[str, Any]],
    revalidate_inputs: Any,
    _test_fault_hook: Any = None,
    _test_race_hook: Any = None,
) -> dict[str, Any]:
    normalized = validate_prior_diagnostic_nomination_bundle_inputs_v4_3(artifacts)
    run_id = normalized[PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3]["run_id"]
    return publish_private_bundle(
        private_root=private_root,
        run_id=run_id,
        artifacts=normalized,
        contract=prior_diagnostic_nomination_bundle_contract_v4_3(),
        revalidate_inputs=revalidate_inputs,
        _test_fault_hook=_test_fault_hook,
        _test_race_hook=_test_race_hook,
    )


def readback_prior_diagnostic_nomination_bundle_files_v4_3(
    bundle_path: str | os.PathLike[str],
) -> dict[str, Any]:
    return readback_private_bundle(
        bundle_path,
        contract=prior_diagnostic_nomination_bundle_contract_v4_3(),
    )


def readback_prior_diagnostic_nomination_bundle_v4_3(
    bundle_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read the exact historical private bytes without mutable-source fallback."""

    return readback_prior_diagnostic_nomination_bundle_files_v4_3(bundle_path)


__all__ = [
    "FactorGovernancePriorDiagnosticNominationBundleV4_3Error",
    "INPUT_FILENAMES_V4_3",
    "PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3",
    "PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3",
    "PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3",
    "READBACK_REPORT_FILENAME_V4_3",
    "READBACK_REPORT_SCHEMA_VERSION_V4_3",
    "ROOT_SUFFIX_V4_3",
    "ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION",
    "prior_diagnostic_nomination_bundle_contract_v4_3",
    "publish_prior_diagnostic_nomination_bundle_v4_3",
    "readback_prior_diagnostic_nomination_bundle_files_v4_3",
    "readback_prior_diagnostic_nomination_bundle_v4_3",
    "validate_prior_diagnostic_nomination_bundle_artifacts_v4_3",
    "validate_prior_diagnostic_nomination_bundle_inputs_v4_3",
    "validate_prior_diagnostic_nomination_readback_v4_3",
]
