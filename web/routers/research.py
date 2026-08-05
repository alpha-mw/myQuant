"""Read-only V17 mainline research endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from quant_investor.v17_mainline import V17MainlineError, read_public_run
from web.config import PROJECT_ROOT
from web.models.research_models import V17MainlinePublicRun

router = APIRouter(prefix="/api/research", tags=["research"])


@router.get("/{strategy_id}", response_model=V17MainlinePublicRun)
def get_active_research_run(
    strategy_id: str,
    expected_pointer_sha256: str | None = Query(default=None),
) -> V17MainlinePublicRun:
    """Read the exact active pointer; never scan history or migrate schemas."""

    try:
        payload = read_public_run(
            PROJECT_ROOT,
            strategy_id=strategy_id,
            expected_pointer_sha256=expected_pointer_sha256,
        )
        return V17MainlinePublicRun.model_validate(payload)
    except V17MainlineError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": exc.detail or exc.code},
        ) from exc


__all__ = ["router"]
