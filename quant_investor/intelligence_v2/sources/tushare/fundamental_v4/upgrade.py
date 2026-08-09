"""Recoverable orchestration around the existing Fundamental promotion CAS."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from quant_investor.market.fundamental_generation import (
    load_fundamental_pointer,
    pointer_sha256,
    preflight_staged_fundamental_promotion,
    promote_staged_fundamental_generation,
)

from ...._core import canonical_bytes, sha256
from .models import PROVIDER_MANIFEST_V4, FundamentalV4ContractError
from .promotion import (
    ZERO_SHA256,
    append_promotion_journal_event,
    build_promotion_event,
    classify_promotion_recovery,
    create_promotion_journal,
    read_promotion_journal,
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _attempt_arguments_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(dict(value))).hexdigest()


def _validated_v4_preflight(
    value: Mapping[str, Any],
    *,
    as_of: str,
) -> dict[str, Any]:
    preflight = dict(value)
    provider = preflight.get("provider_evidence")
    pointer = preflight.get("candidate_pointer")
    if not isinstance(provider, Mapping) or not isinstance(pointer, Mapping):
        raise FundamentalV4ContractError("VIP promotion requires v4 provider evidence")
    manifest = dict(pointer.get("metadata", {}) or {}).get("provider_manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != PROVIDER_MANIFEST_V4
        or manifest.get("as_of") != as_of
        or manifest.get("performance_gate_passed") is not True
    ):
        raise FundamentalV4ContractError("VIP promotion manifest authority is invalid")
    for key in (
        "candidate_pointer_sha256",
        "expected_pointer_sha256",
        "generation_aggregate_sha256",
        "manifest_sha256",
        "scope_sha256",
    ):
        preflight[key] = sha256(preflight.get(key), label=key)
    for key in ("implementation_sha256", "reconciliation_sha256"):
        sha256(provider.get(key), label=key)
    return preflight


def _pointer_validation(
    canonical_root: str | Path,
    *,
    observed_sha256: str,
    candidate_generation_id: str,
    candidate_pointer_sha256: str,
    old_pointer_sha256: str,
) -> tuple[bool, bool]:
    try:
        pointer = load_fundamental_pointer(canonical_root)
    except Exception:
        return False, False
    generation_id = "" if pointer is None else str(pointer.get("generation_id") or "")
    candidate_valid = (
        observed_sha256 == candidate_pointer_sha256 and generation_id == candidate_generation_id
    )
    old_valid = observed_sha256 == old_pointer_sha256 and pointer is not None
    return candidate_valid, old_valid


def _classify_after_interruption(
    *,
    journal_root: Path,
    canonical_root: str | Path,
    candidate_generation_id: str,
    candidate_pointer_sha256: str,
    old_pointer_sha256: str,
) -> tuple[str, str]:
    try:
        first = pointer_sha256(canonical_root)
        second = pointer_sha256(canonical_root)
    except Exception:
        return "PROMOTION_UNCERTAIN", "0" * 64
    candidate_valid, old_valid = _pointer_validation(
        canonical_root,
        observed_sha256=second,
        candidate_generation_id=candidate_generation_id,
        candidate_pointer_sha256=candidate_pointer_sha256,
        old_pointer_sha256=old_pointer_sha256,
    )
    state = classify_promotion_recovery(
        read_promotion_journal(journal_root),
        observed_pointer_sha256_first=first,
        observed_pointer_sha256_second=second,
        candidate_generation_valid=candidate_valid,
        old_generation_valid=old_valid,
    )
    return state, second


def run_staged_vip_promotion(
    *,
    staging_root: str | Path,
    canonical_root: str | Path,
    journal_root: str | Path,
    attempt_id: str,
    as_of: str,
    expected_pointer_sha256: str,
    package_sha256: str,
    authorized_arguments: Mapping[str, Any],
    clock: Callable[[], str] = _now,
) -> dict[str, Any]:
    """Seal INTENT, call the sole CAS helper, then evidence-classify outcome."""

    preflight = _validated_v4_preflight(
        preflight_staged_fundamental_promotion(
            staging_root=staging_root,
            canonical_root=canonical_root,
            expected_pointer_sha256=expected_pointer_sha256,
        ),
        as_of=as_of,
    )
    provider = dict(preflight["provider_evidence"])
    old_sha = preflight["expected_pointer_sha256"]
    candidate_sha = preflight["candidate_pointer_sha256"]
    candidate_generation_id = str(preflight["candidate_generation_id"])
    intent = build_promotion_event(
        attempt_id=attempt_id,
        event_type="INTENT",
        ordinal=1,
        previous_event_sha256=ZERO_SHA256,
        evidence={
            "as_of": as_of,
            "authorized_arguments_sha256": _attempt_arguments_sha256(authorized_arguments),
            "candidate_generation_id": candidate_generation_id,
            "candidate_pointer_sha256": candidate_sha,
            "expected_old_pointer_sha256": old_sha,
            "implementation_sha256": provider["implementation_sha256"],
            "manifest_sha256": preflight["manifest_sha256"],
            "package_sha256": sha256(package_sha256, label="package_sha256"),
            "reconciliation_sha256": provider["reconciliation_sha256"],
            "scope_sha256": preflight["scope_sha256"],
        },
        event_at=clock(),
    )
    journal = create_promotion_journal(journal_root, intent=intent)

    def record(event_type: str, evidence: Mapping[str, Any]) -> None:
        append_promotion_journal_event(
            journal,
            event_type=event_type,
            evidence=evidence,
            event_at=clock(),
        )

    try:
        result = promote_staged_fundamental_generation(
            staging_root=staging_root,
            canonical_root=canonical_root,
            expected_pointer_sha256=old_sha,
            phase_recorder=record,
        )
        if result.get("pointer_sha256") != candidate_sha:
            raise FundamentalV4ContractError("promotion target changed after INTENT")
        state = "PROMOTED"
        observed_sha = candidate_sha
    except Exception:
        state, observed_sha = _classify_after_interruption(
            journal_root=journal,
            canonical_root=canonical_root,
            candidate_generation_id=candidate_generation_id,
            candidate_pointer_sha256=candidate_sha,
            old_pointer_sha256=old_sha,
        )
    try:
        record(
            "TERMINAL",
            {"observed_pointer_sha256": observed_sha, "state": state},
        )
    except Exception:
        return {
            "attempt_id": attempt_id,
            "journal_root": str(journal),
            "state": "PROMOTION_UNCERTAIN",
            "terminal_recorded": False,
        }
    return {
        "attempt_id": attempt_id,
        "candidate_generation_id": candidate_generation_id,
        "candidate_pointer_sha256": candidate_sha,
        "journal_root": str(journal),
        "state": state,
        "terminal_recorded": True,
    }


__all__ = ["run_staged_vip_promotion"]
