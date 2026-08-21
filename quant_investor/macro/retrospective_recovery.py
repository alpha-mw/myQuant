"""Prepare immutable retrospective Market projections for Macro catch-up."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class MacroRetrospectiveRecoveryError(ValueError):
    """Raised when captured bytes cannot prove a retrospective projection."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, expected_sha256: str, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise MacroRetrospectiveRecoveryError(f"{label}_unsafe")
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second or _sha(first) != expected_sha256:
        raise MacroRetrospectiveRecoveryError(f"{label}_sha_mismatch")
    try:
        value = json.loads(first.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MacroRetrospectiveRecoveryError(f"{label}_json_invalid") from exc
    if not isinstance(value, dict):
        raise MacroRetrospectiveRecoveryError(f"{label}_json_invalid")
    return value, first


def _coverage(
    *,
    base: Mapping[str, Any],
    session: Mapping[str, Any],
    target: str,
    evidence_path: Path,
    evidence_sha256: str,
) -> dict[str, Any]:
    classification = session.get("classification")
    if not isinstance(classification, Mapping) or classification.get("status") != "PASSED":
        raise MacroRetrospectiveRecoveryError("session_classification_invalid")
    symbols = classification.get("symbols")
    counts = classification.get("counts")
    if not isinstance(symbols, Mapping) or not isinstance(counts, Mapping):
        raise MacroRetrospectiveRecoveryError("session_classification_invalid")
    suspended = sorted(str(value) for value in symbols.get("suspended", []))
    inactive = sorted(
        {
            *[str(value) for value in symbols.get("inactive", [])],
            *[str(value) for value in symbols.get("delisted", [])],
            *[str(value) for value in symbols.get("prelisting", [])],
        }
    )
    non_trading = sorted(str(value) for value in symbols.get("non_trading", []))
    absent = sorted({*suspended, *inactive, *non_trading})
    observed = sorted(str(value) for value in symbols.get("observed", []))
    expected_count = int(classification.get("expected_scope_count") or 0)
    if (
        classification.get("classification_sets_disjoint") is not True
        or int(classification.get("coverage_complete_count") or 0) != expected_count
        or len(observed) != int(classification.get("observed_bar_count") or -1)
        or len(observed) + len(absent) != expected_count
        or set(observed) & set(absent)
    ):
        raise MacroRetrospectiveRecoveryError("session_scope_closure_invalid")
    result = dict(base)
    result.update(
        {
            "complete": True,
            "coverage_ratio": 1.0,
            "coverage_complete_count": expected_count,
            "expected_scope_count": expected_count,
            "observed_bar_count": len(observed),
            "blocking_incomplete_count": 0,
            "latest_available_trade_date": target,
            "latest_complete_trade_date": target,
            "upsert_target_trade_date": target,
            "coverage_trade_date": target,
            "suspended_symbols": suspended,
            "inactive_symbols": inactive,
            "non_trading_symbols": non_trading,
            "delisted_symbols": sorted(str(value) for value in symbols.get("delisted", [])),
            "prelisting_symbols": sorted(str(value) for value in symbols.get("prelisting", [])),
            "non_blocking_absent_symbols": absent,
            "true_missing_symbols": [],
            "classification_sets_disjoint": True,
            "verified_terminal_delisting_evidence_path": str(evidence_path),
            "verified_terminal_delisting_evidence_sha256": evidence_sha256,
        }
    )
    return result


def build_retrospective_market_projections(
    *,
    source_snapshot_manifest_path: Path,
    expected_source_snapshot_sha256: str,
    capture_manifest_path: Path,
    expected_capture_manifest_sha256: str,
    attempt_root: Path,
    reconstructed_at: str,
    output_root: Path,
) -> dict[str, Any]:
    """Build target-specific projections without changing any canonical pointer."""

    source, source_raw = _read_json(
        source_snapshot_manifest_path,
        expected_source_snapshot_sha256,
        label="source_snapshot",
    )
    capture, _capture_raw = _read_json(
        capture_manifest_path,
        expected_capture_manifest_sha256,
        label="capture_manifest",
    )
    try:
        base_stamp = datetime.fromisoformat(reconstructed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MacroRetrospectiveRecoveryError("reconstructed_at_invalid") from exc
    if base_stamp.tzinfo is None:
        raise MacroRetrospectiveRecoveryError("reconstructed_at_invalid")
    base_stamp = base_stamp.astimezone(timezone.utc).replace(microsecond=0)
    now = datetime.now(timezone.utc)
    if base_stamp > now or now - base_stamp > timedelta(hours=1):
        raise MacroRetrospectiveRecoveryError("reconstructed_at_not_current")
    if (
        source.get("metadata", {}).get("capture_manifest_sha256")
        != expected_capture_manifest_sha256
    ):
        raise MacroRetrospectiveRecoveryError("source_capture_binding_mismatch")
    sessions = capture.get("sessions")
    targets = capture.get("target_trade_dates")
    if not isinstance(sessions, list) or targets != ["20260818", "20260819", "20260820"]:
        raise MacroRetrospectiveRecoveryError("capture_target_set_invalid")
    by_target = {str(row.get("trade_date")): row for row in sessions}
    if set(by_target) != set(targets):
        raise MacroRetrospectiveRecoveryError("capture_session_set_invalid")
    base_coverage = source.get("coverage")
    if not isinstance(base_coverage, Mapping):
        raise MacroRetrospectiveRecoveryError("source_coverage_invalid")
    projection_inputs = {
        "classification": "RETROSPECTIVE_RECONSTRUCTION",
        "source_snapshot_manifest_path": str(source_snapshot_manifest_path),
        "source_snapshot_manifest_sha256": expected_source_snapshot_sha256,
        "capture_manifest_path": str(capture_manifest_path),
        "capture_manifest_sha256": expected_capture_manifest_sha256,
        "targets": targets,
        "reconstructed_at": base_stamp.isoformat(),
    }
    candidate_id = "macro-retrospective-market-" + _sha(canonical_json_bytes(projection_inputs))
    candidate_root = output_root / candidate_id
    snapshots_root = candidate_root / "_snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(candidate_root, 0o700)
    os.chmod(snapshots_root, 0o700)
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        evidence_path = attempt_root / f"pit-classification-evidence-{target}.json"
        evidence_sha = _sha(evidence_path.read_bytes())
        evidence, _ = _read_json(evidence_path, evidence_sha, label="classification_evidence")
        if evidence.get("target_trade_date") != target:
            raise MacroRetrospectiveRecoveryError("classification_target_mismatch")
        stamp = base_stamp + timedelta(seconds=index)
        snapshot_id = stamp.strftime("%Y%m%dT%H%M%SZ")
        path = snapshots_root / f"{snapshot_id}.json"
        coverage = _coverage(
            base=base_coverage,
            session=by_target[target],
            target=target,
            evidence_path=evidence_path,
            evidence_sha256=evidence_sha,
        )
        projection = dict(source)
        projection.update(
            {
                "snapshot_id": snapshot_id,
                "latest_available_trade_date": target,
                "latest_complete_trade_date": target,
                "latest_trade_date": target,
                "manifest_path": str(path),
                "coverage": coverage,
                "retrospective_reconstruction": {
                    "classification": "RETROSPECTIVE_RECONSTRUCTION",
                    "reconstructed_at": stamp.isoformat(),
                    "source_snapshot_manifest_path": str(source_snapshot_manifest_path),
                    "source_snapshot_manifest_sha256": expected_source_snapshot_sha256,
                },
            }
        )
        metadata = dict(projection.get("metadata") or {})
        metadata.update(
            {
                "target_trade_date": target,
                "coverage": coverage,
                "reconstruction_classification": "RETROSPECTIVE_RECONSTRUCTION",
                "reconstructed_at": stamp.isoformat(),
            }
        )
        projection["metadata"] = metadata
        raw = (
            json.dumps(projection, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        if path.exists() and path.read_bytes() != raw:
            raise MacroRetrospectiveRecoveryError("candidate_projection_conflict")
        if not path.exists():
            path.write_bytes(raw)
            os.chmod(path, 0o600)
        rows.append({"target_trade_date": target, "path": str(path), "sha256": _sha(raw)})
    manifest = {
        "schema_version": "macro-retrospective-market-reconstruction.v1",
        "candidate_id": candidate_id,
        "classification": "RETROSPECTIVE_RECONSTRUCTION",
        "reconstructed_at": base_stamp.isoformat(),
        "source_snapshot_manifest_path": str(source_snapshot_manifest_path),
        "source_snapshot_manifest_sha256": expected_source_snapshot_sha256,
        "capture_manifest_path": str(capture_manifest_path),
        "capture_manifest_sha256": expected_capture_manifest_sha256,
        "projections": rows,
        "canonical_pointer_write": False,
        "market_pointer_write": False,
        "pit_pointer_write": False,
    }
    manifest["content_sha256"] = _sha(canonical_json_bytes(manifest))
    manifest_raw = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    manifest_path = candidate_root / "manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_raw:
        raise MacroRetrospectiveRecoveryError("candidate_manifest_conflict")
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_raw)
        os.chmod(manifest_path, 0o600)
    return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_raw)}


__all__ = [
    "MacroRetrospectiveRecoveryError",
    "build_retrospective_market_projections",
    "canonical_json_bytes",
]
