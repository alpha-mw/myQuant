"""Private report-only FactorGovernanceProtocol v4 pre-admission artifact.

This module is deliberately downstream of screening and explicit review status
summaries.  It never runs a screening/replay producer, discovers a registry,
or calls a transaction, receipt, or apply surface.  Even a complete set of
valid inputs stops at ``pending_exact_admission``; only a separate exact
admission API may assess or create a candidate registry proposal.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from collections.abc import Mapping
from typing import Any

from quant_investor.codex_review.storage import (
    CONTROL_MAX_BYTES,
    PRIVATE_DIR_MODE,
    PRIVATE_FILE_MODE,
    ProtocolError,
    assert_cas,
    canonical_json_bytes,
    read_private_bytes,
    run_lock,
    sha256_bytes,
    write_exact_once,
)

REPORT_SCHEMA_VERSION = "factor-governance-pre-admission-report.v4"
SCREENING_SUMMARY_SCHEMA_VERSION = "factor-governance-screening-summary.v4"
CODEX_S1_STATUS_SCHEMA_VERSION = "factor-governance-codex-s1-status.v4"
CODEX_IC_STATUS_SCHEMA_VERSION = "factor-governance-codex-ic-status.v4"
REPLAY_STATUS_SCHEMA_VERSION = "factor-governance-replay-status.v4"
PROTOCOL_VERSION = "v4"
PRE_ADMISSION_REPORT_FILENAME = "factor_governance_pre_admission_report.v4.json"

_SCREENING_SUMMARY_FIELDS = {
    "schema_version",
    "evidence_class",
    "screening_evidence_sha256",
    "candidate_count",
    "evaluated_count",
    "bh_pass_count",
    "compute_failed_count",
}
_CODEX_STATUS_FIELDS = {
    "schema_version",
    "stage",
    "status",
    "verified",
    "artifact_sha256",
}
_REPLAY_STATUS_FIELDS = {
    "schema_version",
    "status",
    "verified",
    "artifact_sha256",
}
_EVIDENCE_KEYS = ("codex_s1", "codex_ic", "replay")
_REPORT_FIELDS = {
    "schema_version",
    "protocol_version",
    "run_id",
    "status",
    "report_only",
    "screening_summary",
    "screening_sha256",
    "evidence_supplied",
    "codex_s1_status",
    "codex_ic_status",
    "replay_status",
    "blockers",
    "pre_admission_passed",
    "candidate_registry_proposal_allowed",
    "registry_write_enabled",
    "registry_mutation_performed",
    "production_apply_enabled",
    "proposals",
    "report_sha256",
}


class FactorGovernancePreAdmissionV4Error(ProtocolError):
    """Raised when a pre-admission report or private readback fails closed."""


def _canonical_semantic_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernancePreAdmissionV4Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_semantic_sha256_v4(value: Any) -> str:
    """Return the v4 semantic SHA (canonical JSON without a trailing newline)."""

    return hashlib.sha256(_canonical_semantic_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value != "0" * 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _run_id(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(token in value for token in ("/", "\\", ".."))
    ):
        raise FactorGovernancePreAdmissionV4Error(
            "run_id must be an exact safe non-empty path component"
        )
    return value


def _screening_is_valid(summary: Any, screening_sha256: Any) -> bool:
    if not isinstance(summary, dict) or set(summary) != _SCREENING_SUMMARY_FIELDS:
        return False
    if summary.get("schema_version") != SCREENING_SUMMARY_SCHEMA_VERSION:
        return False
    if summary.get("evidence_class") != "diagnostic_report_only":
        return False
    if not _is_sha256(summary.get("screening_evidence_sha256")):
        return False
    counts = (
        summary.get("candidate_count"),
        summary.get("evaluated_count"),
        summary.get("bh_pass_count"),
        summary.get("compute_failed_count"),
    )
    if any(type(value) is not int or value < 0 for value in counts):
        return False
    candidate_count, evaluated_count, bh_pass_count, compute_failed_count = counts
    if evaluated_count + compute_failed_count != candidate_count:
        return False
    if bh_pass_count > evaluated_count:
        return False
    if not _is_sha256(screening_sha256):
        return False
    return screening_sha256 == canonical_semantic_sha256_v4(summary)


def _normalize_codex_status(
    value: Any,
    *,
    expected_schema: str,
    expected_stage: str,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    if set(payload) != _CODEX_STATUS_FIELDS:
        return None
    if payload.get("schema_version") != expected_schema:
        return None
    if payload.get("stage") != expected_stage:
        return None
    if payload.get("status") != "passed":
        return None
    if payload.get("verified") is not True:
        return None
    if not _is_sha256(payload.get("artifact_sha256")):
        return None
    return copy.deepcopy(payload)


def _normalize_replay_status(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    if set(payload) != _REPLAY_STATUS_FIELDS:
        return None
    if payload.get("schema_version") != REPLAY_STATUS_SCHEMA_VERSION:
        return None
    if payload.get("status") != "passed":
        return None
    if payload.get("verified") is not True:
        return None
    if not _is_sha256(payload.get("artifact_sha256")):
        return None
    return copy.deepcopy(payload)


def _derive_state(
    *,
    screening_valid: bool,
    evidence_supplied: Mapping[str, bool],
    codex_s1_status: dict[str, Any] | None,
    codex_ic_status: dict[str, Any] | None,
    replay_status: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    normalized = {
        "codex_s1": codex_s1_status,
        "codex_ic": codex_ic_status,
        "replay": replay_status,
    }
    invalid: list[str] = []
    missing: list[str] = []
    if not screening_valid:
        invalid.append("screening_summary_or_hash_invalid")
    for name in _EVIDENCE_KEYS:
        if evidence_supplied[name]:
            if normalized[name] is None:
                invalid.append(f"{name}_status_invalid")
        else:
            missing.append(f"{name}_missing")
    if invalid:
        return "blocked", invalid + missing
    if missing:
        return "pending_codex", missing
    return "pending_exact_admission", ["exact_admission_api_required"]


def build_factor_governance_pre_admission_report_v4(
    *,
    run_id: str,
    screening_summary: Mapping[str, Any],
    screening_sha256: str,
    codex_s1_status: Mapping[str, Any] | None = None,
    codex_ic_status: Mapping[str, Any] | None = None,
    replay_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an inert pre-admission state report from explicit summaries only."""

    normalized_run_id = _run_id(run_id)
    if not isinstance(screening_summary, Mapping):
        summary: dict[str, Any] = {}
    else:
        summary = copy.deepcopy(dict(screening_summary))
    # Reject values that could not themselves be bound into canonical report bytes.
    _canonical_semantic_bytes(summary)
    normalized_screening_sha = screening_sha256 if type(screening_sha256) is str else None

    supplied = {
        "codex_s1": codex_s1_status is not None,
        "codex_ic": codex_ic_status is not None,
        "replay": replay_status is not None,
    }
    normalized_s1 = _normalize_codex_status(
        codex_s1_status,
        expected_schema=CODEX_S1_STATUS_SCHEMA_VERSION,
        expected_stage="CodexS1",
    )
    normalized_ic = _normalize_codex_status(
        codex_ic_status,
        expected_schema=CODEX_IC_STATUS_SCHEMA_VERSION,
        expected_stage="CodexIC",
    )
    normalized_replay = _normalize_replay_status(replay_status)
    screening_valid = _screening_is_valid(summary, normalized_screening_sha)
    status, blockers = _derive_state(
        screening_valid=screening_valid,
        evidence_supplied=supplied,
        codex_s1_status=normalized_s1,
        codex_ic_status=normalized_ic,
        replay_status=normalized_replay,
    )
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": normalized_run_id,
        "status": status,
        "report_only": True,
        "screening_summary": summary,
        "screening_sha256": normalized_screening_sha,
        "evidence_supplied": supplied,
        "codex_s1_status": normalized_s1,
        "codex_ic_status": normalized_ic,
        "replay_status": normalized_replay,
        "blockers": blockers,
        "pre_admission_passed": False,
        "candidate_registry_proposal_allowed": False,
        "registry_write_enabled": False,
        "registry_mutation_performed": False,
        "production_apply_enabled": False,
        "proposals": [],
    }
    payload["report_sha256"] = canonical_semantic_sha256_v4(payload)
    return validate_factor_governance_pre_admission_report_v4(payload)


def validate_factor_governance_pre_admission_report_v4(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact report schema and recompute its fail-closed state."""

    if not isinstance(value, Mapping):
        raise FactorGovernancePreAdmissionV4Error("pre-admission report must be an object")
    payload = copy.deepcopy(dict(value))
    if set(payload) != _REPORT_FIELDS:
        missing = sorted(_REPORT_FIELDS - set(payload))
        unknown = sorted(set(payload) - _REPORT_FIELDS)
        raise FactorGovernancePreAdmissionV4Error(
            f"pre-admission report fields invalid: missing={missing}; unknown={unknown}"
        )
    if payload["schema_version"] != REPORT_SCHEMA_VERSION:
        raise FactorGovernancePreAdmissionV4Error("unsupported pre-admission report schema")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernancePreAdmissionV4Error("pre-admission protocol must be v4")
    _run_id(payload["run_id"])
    if payload["report_only"] is not True:
        raise FactorGovernancePreAdmissionV4Error("pre-admission report must be report-only")
    for field in (
        "pre_admission_passed",
        "candidate_registry_proposal_allowed",
        "registry_write_enabled",
        "registry_mutation_performed",
        "production_apply_enabled",
    ):
        if payload[field] is not False:
            raise FactorGovernancePreAdmissionV4Error(f"{field} must remain false")
    if payload["proposals"] != []:
        raise FactorGovernancePreAdmissionV4Error("pre-admission proposals must remain empty")

    supplied = payload["evidence_supplied"]
    if (
        not isinstance(supplied, dict)
        or set(supplied) != set(_EVIDENCE_KEYS)
        or any(type(supplied[name]) is not bool for name in _EVIDENCE_KEYS)
    ):
        raise FactorGovernancePreAdmissionV4Error("evidence_supplied fields are invalid")
    normalized_s1 = _normalize_codex_status(
        payload["codex_s1_status"],
        expected_schema=CODEX_S1_STATUS_SCHEMA_VERSION,
        expected_stage="CodexS1",
    )
    normalized_ic = _normalize_codex_status(
        payload["codex_ic_status"],
        expected_schema=CODEX_IC_STATUS_SCHEMA_VERSION,
        expected_stage="CodexIC",
    )
    normalized_replay = _normalize_replay_status(payload["replay_status"])
    normalized_by_name = {
        "codex_s1": normalized_s1,
        "codex_ic": normalized_ic,
        "replay": normalized_replay,
    }
    for name in _EVIDENCE_KEYS:
        if payload[f"{name}_status"] is not None and normalized_by_name[name] is None:
            raise FactorGovernancePreAdmissionV4Error(f"stored {name} status is invalid")
        if not supplied[name] and payload[f"{name}_status"] is not None:
            raise FactorGovernancePreAdmissionV4Error(
                f"unsupplied {name} status must be null"
            )

    summary = payload["screening_summary"]
    screening_valid = _screening_is_valid(summary, payload["screening_sha256"])
    expected_status, expected_blockers = _derive_state(
        screening_valid=screening_valid,
        evidence_supplied=supplied,
        codex_s1_status=normalized_s1,
        codex_ic_status=normalized_ic,
        replay_status=normalized_replay,
    )
    if payload["status"] != expected_status:
        raise FactorGovernancePreAdmissionV4Error("pre-admission status is not recomputed")
    if payload["blockers"] != expected_blockers:
        raise FactorGovernancePreAdmissionV4Error("pre-admission blockers are not recomputed")
    report_sha = payload.pop("report_sha256")
    if not _is_sha256(report_sha) or report_sha != canonical_semantic_sha256_v4(payload):
        raise FactorGovernancePreAdmissionV4Error("pre-admission report SHA mismatch")
    payload["report_sha256"] = report_sha
    return payload


def _verify_private_report(
    path: Path,
    *,
    expected_bytes: bytes,
    expected_sha256: str,
) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise FactorGovernancePreAdmissionV4Error(
            "private pre-admission report lstat failed"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FactorGovernancePreAdmissionV4Error(
            "private pre-admission report must be a regular file"
        )
    if metadata.st_uid != os.getuid():
        raise FactorGovernancePreAdmissionV4Error(
            "private pre-admission report owner must be the current uid"
        )
    if metadata.st_nlink != 1:
        raise FactorGovernancePreAdmissionV4Error(
            "private pre-admission report must have exactly one hard link"
        )
    if stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE:
        raise FactorGovernancePreAdmissionV4Error(
            "private pre-admission report mode must be 0600"
        )
    readback = read_private_bytes(path, max_bytes=CONTROL_MAX_BYTES)
    if readback != expected_bytes or sha256_bytes(readback) != expected_sha256:
        raise FactorGovernancePreAdmissionV4Error(
            "private pre-admission report hash readback mismatch"
        )


def _verify_private_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise FactorGovernancePreAdmissionV4Error("private artifact path must be a directory")
    if metadata.st_uid != os.getuid():
        raise FactorGovernancePreAdmissionV4Error(
            "private artifact directory owner must be the current uid"
        )
    if stat.S_IMODE(metadata.st_mode) != PRIVATE_DIR_MODE:
        raise FactorGovernancePreAdmissionV4Error(
            "private artifact directories must be mode 0700"
        )


def publish_factor_governance_pre_admission_report_v4(
    *,
    private_root: str | Path,
    run_id: str,
    expected_report_sha256: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one fixed private report with CAS and exact-once semantics."""

    normalized_run_id = _run_id(run_id)
    validated = validate_factor_governance_pre_admission_report_v4(report)
    if validated["run_id"] != normalized_run_id:
        raise FactorGovernancePreAdmissionV4Error("report run_id does not match path run_id")
    if expected_report_sha256 != "empty" and not _is_sha256(expected_report_sha256):
        raise FactorGovernancePreAdmissionV4Error(
            "expected_report_sha256 must be 'empty' or lowercase SHA-256"
        )
    payload = canonical_json_bytes(validated)
    if len(payload) > CONTROL_MAX_BYTES:
        raise FactorGovernancePreAdmissionV4Error("pre-admission report exceeds size limit")

    with run_lock(private_root, normalized_run_id) as (root_path, run_dir):
        _verify_private_directory(root_path)
        _verify_private_directory(run_dir)
        report_path = run_dir / PRE_ADMISSION_REPORT_FILENAME
        assert_cas(report_path, expected_report_sha256)
        report_sha, created = write_exact_once(report_path, payload, root=root_path)
        _verify_private_report(
            report_path,
            expected_bytes=payload,
            expected_sha256=report_sha,
        )
    return {"path": str(report_path), "sha256": report_sha, "created": created}


__all__ = [
    "CODEX_IC_STATUS_SCHEMA_VERSION",
    "CODEX_S1_STATUS_SCHEMA_VERSION",
    "FactorGovernancePreAdmissionV4Error",
    "PRE_ADMISSION_REPORT_FILENAME",
    "PROTOCOL_VERSION",
    "REPLAY_STATUS_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SCREENING_SUMMARY_SCHEMA_VERSION",
    "build_factor_governance_pre_admission_report_v4",
    "canonical_semantic_sha256_v4",
    "publish_factor_governance_pre_admission_report_v4",
    "validate_factor_governance_pre_admission_report_v4",
]
