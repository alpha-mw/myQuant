"""Exact local screening and dedup evidence for v4 candidate admission."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from quant_investor.factors.governance_screening_v4 import (
    CANDIDATE_CATALOG_SCHEMA_VERSION,
    EVALUATED_STATUS,
    FDR_METHOD as SCREENING_FDR_METHOD,
    SCREENING_EVIDENCE_SCHEMA_VERSION,
    SOURCE_BINDING_FIELDS,
    canonical_json_bytes,
    canonical_semantic_sha256,
    validate_candidate_catalog_v4,
    validate_primitive_ontology_v4,
    validate_screening_evidence_v4,
)

SCREENING_READBACK_SCHEMA_VERSION = "factor-governance-screening-readback.v4"
DEDUP_READBACK_SCHEMA_VERSION = "factor-candidate-dedup-readback.v4"
DEDUP_EVIDENCE_SCHEMA_VERSION = "factor-candidate-dedup-evidence.v4"
DEDUP_METRIC = "median_monthly_cross_sectional_abs_spearman"
DEDUP_THRESHOLD = 0.70
MIN_VALID_COMMON_DATE_COUNT = 3
MAX_ADMISSION_EVIDENCE_BYTES = 8 * 1024 * 1024


class CandidateAdmissionEvidenceV4Error(ValueError):
    """Raised when exact v4 admission evidence fails closed."""


def canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


def file_sha256_for_payload(value: Any) -> str:
    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CandidateAdmissionEvidenceV4Error(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} must be an exact non-empty string"
        )
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAdmissionEvidenceV4Error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise CandidateAdmissionEvidenceV4Error(f"{label} field names must be strings")
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} fields invalid: {';'.join(details)}"
        )
    return payload


def _strict_json_loads(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidateAdmissionEvidenceV4Error(
                    f"{label} duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} bytes are not strict JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CandidateAdmissionEvidenceV4Error(f"{label} file must contain an object")
    return value


def read_exact_json_artifact_v4(
    *,
    path: str,
    expected_file_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Read one explicit owner-only canonical JSON file with descriptor SHA."""

    normalized_path = Path(_text(path, f"{label} path"))
    if "\x00" in str(normalized_path) or not normalized_path.is_absolute():
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} path must be absolute without NUL"
        )
    expected_sha = _sha(expected_file_sha256, f"{label} expected file SHA")
    try:
        before = normalized_path.lstat()
    except OSError as exc:
        raise CandidateAdmissionEvidenceV4Error(f"{label} lstat failed") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} must be a regular non-symlink file"
        )
    if before.st_uid != os.getuid():
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} owner must be the current uid"
        )
    if before.st_nlink != 1:
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} must have exactly one hard link"
        )
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise CandidateAdmissionEvidenceV4Error(f"{label} mode must be 0600")
    if before.st_size <= 0 or before.st_size > MAX_ADMISSION_EVIDENCE_BYTES:
        raise CandidateAdmissionEvidenceV4Error(f"{label} file size is invalid")

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            item.st_mode,
            item.st_nlink,
        )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(normalized_path, flags)
    except OSError as exc:
        raise CandidateAdmissionEvidenceV4Error(f"{label} open failed") from exc
    try:
        opened = os.fstat(fd)
        if identity(before) != identity(opened):
            raise CandidateAdmissionEvidenceV4Error(
                f"{label} path changed before readback"
            )
        chunks: list[bytes] = []
        remaining = MAX_ADMISSION_EVIDENCE_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        opened_after = os.fstat(fd)
    except OSError as exc:
        raise CandidateAdmissionEvidenceV4Error(f"{label} readback failed") from exc
    finally:
        os.close(fd)
    try:
        after = normalized_path.lstat()
    except OSError as exc:
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} disappeared after readback"
        ) from exc
    if len(raw) > MAX_ADMISSION_EVIDENCE_BYTES:
        raise CandidateAdmissionEvidenceV4Error(f"{label} file exceeds byte limit")
    if identity(before) != identity(after) or identity(opened) != identity(opened_after):
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} file identity changed during readback"
        )
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != expected_sha:
        raise CandidateAdmissionEvidenceV4Error(f"{label} file SHA mismatch")
    payload = _strict_json_loads(raw, label=label)
    if raw != canonical_file_bytes(payload):
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} bytes must be compact sorted canonical JSON"
        )
    return {"path": str(normalized_path), "file_sha256": file_sha, "payload": payload}


def _catalog_candidate(
    catalog: Mapping[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in catalog.get("candidates", [])
        if isinstance(row, Mapping) and row.get("name") == candidate_name
    ]
    if len(matches) != 1:
        raise CandidateAdmissionEvidenceV4Error(
            "candidate catalog must contain exactly one candidate row"
        )
    return matches[0]


def readback_screening_evidence_v4(
    descriptor: Mapping[str, Any],
    *,
    candidate_name: str,
) -> dict[str, Any]:
    """Read exact ontology/catalog/screening files and recompute candidate BH."""

    payload = _exact(
        descriptor,
        {
            "schema_version",
            "ontology_path",
            "ontology_file_sha256",
            "candidate_catalog_path",
            "candidate_catalog_file_sha256",
            "screening_evidence_path",
            "screening_evidence_file_sha256",
        },
        "screening readback descriptor",
    )
    if payload["schema_version"] != SCREENING_READBACK_SCHEMA_VERSION:
        raise CandidateAdmissionEvidenceV4Error("unsupported screening readback schema")
    ontology_file = read_exact_json_artifact_v4(
        path=payload["ontology_path"],
        expected_file_sha256=payload["ontology_file_sha256"],
        label="screening ontology",
    )
    catalog_file = read_exact_json_artifact_v4(
        path=payload["candidate_catalog_path"],
        expected_file_sha256=payload["candidate_catalog_file_sha256"],
        label="screening candidate catalog",
    )
    evidence_file = read_exact_json_artifact_v4(
        path=payload["screening_evidence_path"],
        expected_file_sha256=payload["screening_evidence_file_sha256"],
        label="screening evidence",
    )
    ontology = validate_primitive_ontology_v4(ontology_file["payload"])
    catalog = validate_candidate_catalog_v4(catalog_file["payload"], ontology=ontology)
    evidence = validate_screening_evidence_v4(
        evidence_file["payload"],
        ontology=ontology,
        catalog=catalog,
    )
    candidate = _catalog_candidate(catalog, candidate_name)
    rows = [row for row in evidence["rows"] if row["name"] == candidate_name]
    if len(rows) != 1:
        raise CandidateAdmissionEvidenceV4Error(
            "screening evidence must bind exactly one candidate row"
        )
    row = dict(rows[0])
    if row.get("evaluation_status") != EVALUATED_STATUS:
        raise CandidateAdmissionEvidenceV4Error(
            "screening candidate row must be evaluated"
        )
    return {
        "ontology": ontology,
        "candidate_catalog": catalog,
        "screening_evidence": evidence,
        "candidate_catalog_row": candidate,
        "screening_row": row,
        "candidate_catalog_sha256": catalog["semantic_sha256"],
        "screening_evidence_sha256": evidence["semantic_sha256"],
        "bh_q_value": row["bh_q_value"],
        "bh_pass": row["bh_pass"],
        "family": row["family"],
        "fdr_method": SCREENING_FDR_METHOD,
        "file_sha256s": {
            "ontology": ontology_file["file_sha256"],
            "candidate_catalog": catalog_file["file_sha256"],
            "screening_evidence": evidence_file["file_sha256"],
        },
        "exact_readback_verified": True,
    }


def _primitive_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CandidateAdmissionEvidenceV4Error(f"{label} must be a non-empty list")
    result = [_text(item, f"{label}[]") for item in value]
    if result != sorted(result) or len(result) != len(set(result)):
        raise CandidateAdmissionEvidenceV4Error(
            f"{label} must be sorted and unique"
        )
    return result


def _comparison_row(raw: Any, *, index: int) -> dict[str, Any]:
    row = _exact(
        raw,
        {
            "existing_factor_name",
            "existing_primitive_ids",
            "abs_correlation",
            "valid_common_date_count",
        },
        f"comparison_rows[{index}]",
    )
    name = _text(row["existing_factor_name"], f"comparison_rows[{index}].name")
    primitive_ids = _primitive_ids(
        row["existing_primitive_ids"],
        f"comparison_rows[{index}].existing_primitive_ids",
    )
    corr = row["abs_correlation"]
    if isinstance(corr, bool) or not isinstance(corr, (int, float)):
        raise CandidateAdmissionEvidenceV4Error(
            f"comparison_rows[{index}].abs_correlation must be numeric"
        )
    abs_corr = float(corr)
    if type(corr) is not float or not math.isfinite(abs_corr) or not 0.0 <= abs_corr <= 1.0:
        raise CandidateAdmissionEvidenceV4Error(
            f"comparison_rows[{index}].abs_correlation must be canonical finite float in [0,1]"
        )
    count = row["valid_common_date_count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise CandidateAdmissionEvidenceV4Error(
            f"comparison_rows[{index}].valid_common_date_count must be integer"
        )
    if count < MIN_VALID_COMMON_DATE_COUNT:
        raise CandidateAdmissionEvidenceV4Error(
            f"comparison_rows[{index}].valid_common_date_count must be >=3"
        )
    return {
        "existing_factor_name": name,
        "existing_primitive_ids": primitive_ids,
        "abs_correlation": abs_corr,
        "valid_common_date_count": count,
    }


def _source_bindings(value: Any, label: str) -> dict[str, str]:
    payload = _exact(value, set(SOURCE_BINDING_FIELDS), label)
    return {
        key: _sha(payload[key], f"{label}.{key}")
        for key in sorted(SOURCE_BINDING_FIELDS)
    }


def _dedup_semantic_sha(payload: Mapping[str, Any]) -> str:
    return canonical_semantic_sha256(payload, exclude_fields=("semantic_sha256",))


def build_candidate_dedup_evidence_v4(
    *,
    catalog: Mapping[str, Any],
    candidate_name: str,
    screening_evidence_sha256: str,
    source_bindings: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    evidence_complete: bool,
) -> dict[str, Any]:
    """Build canonical high-correlation/primitive dedup evidence."""

    normalized_name = _text(candidate_name, "candidate_name")
    if catalog.get("schema_version") != CANDIDATE_CATALOG_SCHEMA_VERSION:
        raise CandidateAdmissionEvidenceV4Error("dedup catalog must be v4")
    screening_sha = _sha(screening_evidence_sha256, "dedup screening evidence SHA")
    normalized_source_bindings = _source_bindings(source_bindings, "dedup source_bindings")
    candidate = _catalog_candidate(catalog, normalized_name)
    candidate_primitives = _primitive_ids(
        candidate["primitive_ids"],
        "candidate primitive_ids",
    )
    rows = [
        _comparison_row(row, index=index)
        for index, row in enumerate(comparison_rows)
    ]
    rows.sort(key=lambda item: item["existing_factor_name"])
    if len({row["existing_factor_name"] for row in rows}) != len(rows):
        raise CandidateAdmissionEvidenceV4Error(
            "dedup comparison rows must be unique by existing_factor_name"
        )
    comparison_factor_names = [row["existing_factor_name"] for row in rows]
    comparison_factor_set_sha256 = canonical_semantic_sha256(comparison_factor_names)
    duplicate = any(
        row["existing_primitive_ids"] == candidate_primitives for row in rows
    )
    high_corr_passed = bool(
        evidence_complete
        and not duplicate
        and all(row["abs_correlation"] < DEDUP_THRESHOLD for row in rows)
    )
    payload: dict[str, Any] = {
        "schema_version": DEDUP_EVIDENCE_SCHEMA_VERSION,
        "candidate_name": normalized_name,
        "candidate_definition_sha256": candidate["definition_sha256"],
        "candidate_catalog_sha256": catalog["semantic_sha256"],
        "screening_evidence_sha256": screening_sha,
        "source_bindings": normalized_source_bindings,
        "candidate_primitive_ids": candidate_primitives,
        "metric": DEDUP_METRIC,
        "threshold": DEDUP_THRESHOLD,
        "comparison_factor_names": comparison_factor_names,
        "comparison_factor_set_sha256": comparison_factor_set_sha256,
        "comparison_rows": rows,
        "evidence_complete": bool(evidence_complete),
        "duplicate_primitive": duplicate,
        "high_correlation_dedup_passed": high_corr_passed,
    }
    payload["semantic_sha256"] = _dedup_semantic_sha(payload)
    return payload


def validate_candidate_dedup_evidence_v4(
    evidence: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    candidate_name: str,
    screening_evidence_sha256: str,
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and recompute duplicate/high-correlation pass from rows."""

    payload = _exact(
        evidence,
        {
            "schema_version",
            "candidate_name",
            "candidate_definition_sha256",
            "candidate_catalog_sha256",
            "screening_evidence_sha256",
            "source_bindings",
            "candidate_primitive_ids",
            "metric",
            "threshold",
            "comparison_factor_names",
            "comparison_factor_set_sha256",
            "comparison_rows",
            "evidence_complete",
            "duplicate_primitive",
            "high_correlation_dedup_passed",
            "semantic_sha256",
        },
        "candidate dedup evidence",
    )
    if payload["schema_version"] != DEDUP_EVIDENCE_SCHEMA_VERSION:
        raise CandidateAdmissionEvidenceV4Error("unsupported dedup evidence schema")
    if payload["metric"] != DEDUP_METRIC:
        raise CandidateAdmissionEvidenceV4Error("dedup metric mismatch")
    if type(payload["threshold"]) is not float or payload["threshold"] != DEDUP_THRESHOLD:
        raise CandidateAdmissionEvidenceV4Error("dedup threshold must be canonical 0.70")
    if type(payload["evidence_complete"]) is not bool:
        raise CandidateAdmissionEvidenceV4Error("dedup evidence_complete must be boolean")
    normalized_name = _text(candidate_name, "candidate_name")
    if payload["candidate_name"] != normalized_name:
        raise CandidateAdmissionEvidenceV4Error("dedup candidate name mismatch")
    candidate = _catalog_candidate(catalog, normalized_name)
    if payload["candidate_definition_sha256"] != candidate["definition_sha256"]:
        raise CandidateAdmissionEvidenceV4Error("dedup candidate definition SHA mismatch")
    if payload["candidate_catalog_sha256"] != catalog["semantic_sha256"]:
        raise CandidateAdmissionEvidenceV4Error("dedup candidate catalog SHA mismatch")
    screening_sha = _sha(payload["screening_evidence_sha256"], "dedup screening evidence SHA")
    if screening_sha != _sha(screening_evidence_sha256, "expected screening evidence SHA"):
        raise CandidateAdmissionEvidenceV4Error("dedup screening evidence SHA mismatch")
    normalized_source_bindings = _source_bindings(payload["source_bindings"], "dedup source_bindings")
    if normalized_source_bindings != _source_bindings(source_bindings, "expected dedup source_bindings"):
        raise CandidateAdmissionEvidenceV4Error("dedup source bindings mismatch")
    candidate_primitives = _primitive_ids(
        payload["candidate_primitive_ids"],
        "dedup candidate_primitive_ids",
    )
    if candidate_primitives != candidate["primitive_ids"]:
        raise CandidateAdmissionEvidenceV4Error("dedup candidate primitive IDs mismatch")
    raw_rows = payload["comparison_rows"]
    if not isinstance(raw_rows, list):
        raise CandidateAdmissionEvidenceV4Error("dedup comparison_rows must be a list")
    rows = [
        _comparison_row(row, index=index)
        for index, row in enumerate(raw_rows)
    ]
    comparison_factor_names = _primitive_ids(
        payload["comparison_factor_names"],
        "dedup comparison_factor_names",
    ) if payload["comparison_factor_names"] else []
    if not isinstance(payload["comparison_factor_names"], list):
        raise CandidateAdmissionEvidenceV4Error(
            "dedup comparison_factor_names must be a list"
        )
    if rows != sorted(rows, key=lambda item: item["existing_factor_name"]):
        raise CandidateAdmissionEvidenceV4Error(
            "dedup comparison_rows must be sorted"
        )
    if len({row["existing_factor_name"] for row in rows}) != len(rows):
        raise CandidateAdmissionEvidenceV4Error(
            "dedup comparison_rows must be unique"
        )
    row_names = [row["existing_factor_name"] for row in rows]
    if comparison_factor_names != row_names:
        raise CandidateAdmissionEvidenceV4Error(
            "dedup comparison_factor_names must match comparison rows"
        )
    if payload["comparison_factor_set_sha256"] != canonical_semantic_sha256(
        comparison_factor_names
    ):
        raise CandidateAdmissionEvidenceV4Error(
            "dedup comparison factor set SHA mismatch"
        )
    recomputed = build_candidate_dedup_evidence_v4(
        catalog=catalog,
        candidate_name=normalized_name,
        screening_evidence_sha256=screening_sha,
        source_bindings=normalized_source_bindings,
        comparison_rows=rows,
        evidence_complete=payload["evidence_complete"],
    )
    if canonical_json_bytes({**payload, "semantic_sha256": recomputed["semantic_sha256"]}) != canonical_json_bytes(recomputed):
        raise CandidateAdmissionEvidenceV4Error("dedup evidence drifted from recomputation")
    if _sha(payload["semantic_sha256"], "dedup semantic SHA") != _dedup_semantic_sha(recomputed):
        raise CandidateAdmissionEvidenceV4Error("dedup semantic SHA mismatch")
    return recomputed


def readback_candidate_dedup_evidence_v4(
    descriptor: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    candidate_name: str,
    screening_evidence_sha256: str,
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Read exact dedup evidence and recompute pass/duplicate fields."""

    payload = _exact(
        descriptor,
        {
            "schema_version",
            "dedup_evidence_path",
            "dedup_evidence_file_sha256",
        },
        "dedup readback descriptor",
    )
    if payload["schema_version"] != DEDUP_READBACK_SCHEMA_VERSION:
        raise CandidateAdmissionEvidenceV4Error("unsupported dedup readback schema")
    artifact = read_exact_json_artifact_v4(
        path=payload["dedup_evidence_path"],
        expected_file_sha256=payload["dedup_evidence_file_sha256"],
        label="dedup evidence",
    )
    evidence = validate_candidate_dedup_evidence_v4(
        artifact["payload"],
        catalog=catalog,
        candidate_name=candidate_name,
        screening_evidence_sha256=screening_evidence_sha256,
        source_bindings=source_bindings,
    )
    return {
        "dedup_evidence": evidence,
        "dedup_evidence_sha256": evidence["semantic_sha256"],
        "dedup_evidence_file_sha256": artifact["file_sha256"],
        "comparison_factor_names": evidence["comparison_factor_names"],
        "comparison_factor_set_sha256": evidence["comparison_factor_set_sha256"],
        "duplicate_primitive": evidence["duplicate_primitive"],
        "high_correlation_dedup_passed": evidence["high_correlation_dedup_passed"],
        "evidence_complete": evidence["evidence_complete"],
        "exact_readback_verified": True,
    }


__all__ = [
    "DEDUP_EVIDENCE_SCHEMA_VERSION",
    "DEDUP_METRIC",
    "DEDUP_READBACK_SCHEMA_VERSION",
    "DEDUP_THRESHOLD",
    "SCREENING_READBACK_SCHEMA_VERSION",
    "CandidateAdmissionEvidenceV4Error",
    "build_candidate_dedup_evidence_v4",
    "canonical_file_bytes",
    "file_sha256_for_payload",
    "read_exact_json_artifact_v4",
    "readback_candidate_dedup_evidence_v4",
    "readback_screening_evidence_v4",
    "validate_candidate_dedup_evidence_v4",
]
