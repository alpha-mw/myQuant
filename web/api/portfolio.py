"""Portfolio API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from web.models.portfolio_models import (
    HoldingUpsertRequest,
    PortfolioMutationResponse,
    PortfolioStateResponse,
    WatchlistUpsertRequest,
)
from web.services import portfolio_service

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("", response_model=PortfolioStateResponse)
def get_portfolio_state():
    return PortfolioStateResponse(**portfolio_service.get_portfolio_state())


@router.post("/holdings", response_model=PortfolioMutationResponse)
def upsert_holding(req: HoldingUpsertRequest):
    return PortfolioMutationResponse(**portfolio_service.upsert_holding(req.model_dump()))


@router.delete("/holdings/{holding_id}", response_model=PortfolioMutationResponse)
def delete_holding(holding_id: int):
    result = portfolio_service.delete_holding(holding_id)
    if not result["ok"]:
        raise HTTPException(status_code=404, detail="Holding not found")
    return PortfolioMutationResponse(**result)


@router.post("/watchlist", response_model=PortfolioMutationResponse)
def upsert_watchlist(req: WatchlistUpsertRequest):
    return PortfolioMutationResponse(**portfolio_service.upsert_watchlist(req.model_dump()))


@router.delete("/watchlist/{symbol}", response_model=PortfolioMutationResponse)
def delete_watchlist(symbol: str):
    result = portfolio_service.delete_watchlist(symbol)
    if not result["ok"]:
        raise HTTPException(status_code=404, detail="Watchlist entry not found")
    return PortfolioMutationResponse(**result)
