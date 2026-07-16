"""Single-source v15 formal-run readiness contract.

The formal wrapper is the only producer.  Reports, manifests, and Dashboard
exports persist a hash reference to this artifact instead of recomputing gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "v15_run_readiness.v1"
REQUIRED_BRANCHES = ("quant", "fundamental", "macro")
VALID_CANDIDATE_STATUSES = frozenset({"blocked", "empty", "complete"})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_strings(values: Sequence[Any] | None) -> list[str]:
    return sorted({str(value).strip() for value in (values or ()) if str(value).strip()})


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def readiness_reference(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "path": path.name,
        "sha256": canonical_sha256(payload),
    }


def _branch_contract(
    branch_readiness: Mapping[str, Any] | None,
    branch_objects: Mapping[str, Any] | None,
) -> tuple[dict[str, bool], dict[str, bool], list[str]]:
    readiness = _mapping(branch_readiness)
    objects = _mapping(branch_objects)
    materialized: dict[str, bool] = {}
    ready: dict[str, bool] = {}
    blockers: list[str] = []
    for branch in REQUIRED_BRANCHES:
        branch_payload = _mapping(readiness.get(branch))
        materialized[branch] = bool(objects.get(branch, bool(branch_payload)))
        ready[branch] = bool(
            materialized[branch]
            and str(branch_payload.get("status") or "").lower() in {"pass", "ready"}
            and not list(branch_payload.get("blockers") or [])
        )
        if not materialized[branch]:
            blockers.append(f"branch_object_missing:{branch}")
        elif not ready[branch]:
            branch_blockers = _stable_strings(branch_payload.get("blockers") or [])
            blockers.extend(
                f"branch_data_not_ready:{branch}:{item}" for item in branch_blockers
            )
            if not branch_blockers:
                blockers.append(f"branch_data_not_ready:{branch}")
    return materialized, ready, blockers


def build_v15_run_readiness(
    *,
    run_id: str,
    generated_at: str,
    analysis_trade_date: str,
    market_data_ready: bool,
    market_data_blockers: Sequence[Any] | None,
    branch_readiness: Mapping[str, Any] | None,
    branch_objects: Mapping[str, Any] | None,
    factor_governance: Mapping[str, Any] | None,
    candidate_decision: Mapping[str, Any] | None,
    portfolio_constructor: Mapping[str, Any] | None,
    human_authorization: Mapping[str, Any] | None = None,
    risk_reduction_quote_gate: Mapping[str, Any] | None = None,
    material_warnings: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic, fail-closed v15 readiness payload."""

    objects, branches_ready, branch_blockers = _branch_contract(
        branch_readiness, branch_objects
    )
    factor = _mapping(factor_governance)
    factor_ready = bool(
        factor.get("production_eligible") is True
        and str(factor.get("governance_status") or "").lower() == "ready"
        and not list(factor.get("blockers") or [])
    )
    candidate = _mapping(candidate_decision)
    candidate_status = str(
        candidate.get("candidate_decision_status")
        or candidate.get("candidate_generation_status")
        or "blocked"
    ).lower()
    if candidate_status not in VALID_CANDIDATE_STATUSES:
        candidate_status = "blocked"
    candidate_blocker = str(candidate.get("blocker") or "").strip()
    portfolio = _mapping(portfolio_constructor)
    target_weights = _mapping(
        portfolio.get("target_weights") or portfolio.get("target_positions")
    )
    positive_targets = sorted(
        str(symbol)
        for symbol, raw_weight in target_weights.items()
        if _positive_number(raw_weight)
    )
    portfolio_valid = bool(
        portfolio
        and positive_targets
        and not list(portfolio.get("blockers") or [])
        and portfolio.get("valid", True) is True
    )
    authorization = _mapping(human_authorization)
    authorization_valid = bool(
        authorization.get("authorized") is True
        and authorization.get("run_id") == run_id
        and authorization.get("analysis_trade_date") == analysis_trade_date
        and authorization.get("portfolio_sha256") == canonical_sha256(portfolio)
        and authorization.get("expired") is not True
        and authorization.get("replayed") is not True
    )

    blockers = _stable_strings(market_data_blockers)
    if not market_data_ready:
        blockers.append("market_data_not_ready")
    blockers.extend(branch_blockers)
    if not factor_ready:
        factor_blockers = _stable_strings(factor.get("blockers") or [])
        blockers.extend(f"factor_governance_not_ready:{item}" for item in factor_blockers)
        if not factor_blockers:
            blockers.append("factor_governance_not_ready")
    if candidate_status == "blocked":
        blockers.append(candidate_blocker or "candidate_decision_blocked")
    elif candidate_status == "empty":
        blockers.append(candidate_blocker or "no_candidate_selected_by_portfolio_constructor")
    if not portfolio_valid:
        blockers.append("portfolio_constructor_result_not_valid")
    if not authorization_valid:
        blockers.append("new_risk_human_authorization_missing_or_invalid")

    new_risk_authorized = bool(
        market_data_ready
        and all(branches_ready.values())
        and factor_ready
        and candidate_status == "complete"
        and portfolio_valid
        and authorization_valid
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "generated_at": str(generated_at),
        "analysis_trade_date": str(analysis_trade_date),
        "market_data_ready": bool(market_data_ready),
        "branch_objects_materialized": objects,
        "branch_data_ready": branches_ready,
        "factor_governance_ready": factor_ready,
        "factor_governance": {
            "registry_file_sha256": str(
                factor.get("registry_file_sha256") or factor.get("registry_sha256") or ""
            ),
            "production_factor_set_sha256": str(
                factor.get("production_factor_set_sha256") or ""
            ),
            "blockers": _stable_strings(factor.get("blockers") or []),
        },
        "candidate_decision_status": candidate_status,
        "candidate_decision_blocker": candidate_blocker,
        "portfolio_constructor": {
            "valid": portfolio_valid,
            "sha256": canonical_sha256(portfolio),
            "positive_target_symbols": positive_targets,
        },
        "human_authorization": {
            "present": bool(authorization),
            "valid": authorization_valid,
            "receipt_sha256": canonical_sha256(authorization) if authorization else None,
        },
        "new_risk_authorized": new_risk_authorized,
        "risk_reduction_quote_gate": _mapping(risk_reduction_quote_gate),
        "blockers": _stable_strings(blockers),
        "material_warnings": _stable_strings(material_warnings),
    }


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def write_v15_run_readiness(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Atomically persist owner-only JSON and return its immutable reference."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid v15 run readiness schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    reference = readiness_reference(path, payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise RuntimeError("v15 run readiness readback hash mismatch")
    return reference


def load_v15_run_readiness(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("v15 run readiness must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid v15 run readiness artifact")
    if canonical_sha256(payload) != str(expected_sha256).lower():
        raise ValueError("v15 run readiness sha256 mismatch")
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "build_v15_run_readiness",
    "canonical_sha256",
    "load_v15_run_readiness",
    "readiness_reference",
    "write_v15_run_readiness",
]
