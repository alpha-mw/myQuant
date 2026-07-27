"""Read-only, explicitly versioned V17 v4 research-run route."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from quant_investor.v17_v4_runtime.public_surfaces import (
    PublicSurfaceError,
    resolve_public_run,
)
from web.config import PROJECT_ROOT
from web.models.v17_v4_models import V17V4ResearchRunResponse

router = APIRouter(
    prefix="/api/v4/research-runs",
    tags=["v17-v4-canary"],
)


@router.get("/{strategy_id}", response_model=V17V4ResearchRunResponse)
def get_v17_v4_research_run(
    strategy_id: str,
) -> V17V4ResearchRunResponse:
    try:
        payload = resolve_public_run(
            PROJECT_ROOT,
            strategy_id=strategy_id,
            surface="WEB",
        )
        return V17V4ResearchRunResponse.model_validate(payload)
    except (PublicSurfaceError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail="V17 v4 FORMAL_ACTIVE run is unavailable",
        ) from exc


__all__ = ["router"]
