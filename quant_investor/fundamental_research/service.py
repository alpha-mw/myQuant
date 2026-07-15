"""Offline response validation and import service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .models import (
    FundamentalOverlayV1,
    FundamentalResearchRequestV1,
    FundamentalResearchResponseV1,
    SourceEligibilityPolicyV1,
    compute_source_policy_sha256,
)
from .scoring import build_overlay
from .storage import atomic_write_json_model, load_json_model, model_sha256


class ResponseBindingError(ValueError):
    pass


def validate_response(
    request: FundamentalResearchRequestV1,
    response: FundamentalResearchResponseV1,
    *,
    imported_at: datetime,
    source_policy: SourceEligibilityPolicyV1 | None = None,
) -> FundamentalOverlayV1:
    effective_policy = source_policy or SourceEligibilityPolicyV1()
    if request.source_policy_sha256 != compute_source_policy_sha256(effective_policy):
        raise ResponseBindingError("request source policy hash does not match local policy")
    if (
        response.request_id != request.request_id
        or response.dossier.request_id != request.request_id
    ):
        raise ResponseBindingError("response request_id does not match request")
    expected_hash = model_sha256(request)
    if response.request_sha256 != expected_hash:
        raise ResponseBindingError("response request_sha256 does not match canonical request")
    overlay = build_overlay(
        request,
        response.dossier,
        imported_at=imported_at,
        source_policy=effective_policy,
    )
    if not overlay.eligible:
        raise ResponseBindingError(
            "response is not eligible for import: " + ",".join(overlay.blockers)
        )
    return overlay


def import_response_files(
    *,
    root: str | Path,
    request_path: str | Path,
    response_path: str | Path,
    dossier_path: str | Path,
    overlay_path: str | Path,
    imported_at: datetime,
    source_policy: SourceEligibilityPolicyV1 | None = None,
    validate_only: bool = False,
) -> FundamentalOverlayV1:
    """Validate untrusted JSON and optionally persist validated local artifacts."""
    request = load_json_model(root, request_path, FundamentalResearchRequestV1)
    response = load_json_model(root, response_path, FundamentalResearchResponseV1)
    overlay = validate_response(
        request,
        response,
        imported_at=imported_at,
        source_policy=source_policy,
    )
    if not validate_only:
        atomic_write_json_model(root, dossier_path, response.dossier)
        atomic_write_json_model(root, overlay_path, overlay)
    return overlay
