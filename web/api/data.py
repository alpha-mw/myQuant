"""Data API endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from web.models.data_models import (
    CompetitorInfo,
    DatabaseStats,
    MarketOverviewResponse,
    OHLCVResponse,
    StockInfo,
    StockDossierResponse,
    StockListResponse,
    StockOverviewResponse,
)
from web.services import data_service

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/statistics", response_model=DatabaseStats)
def get_statistics():
    return data_service.get_statistics()


@router.get("/market/overview", response_model=MarketOverviewResponse)
def get_market_overview():
    return MarketOverviewResponse(**data_service.get_market_overview())


@router.get("/stocks", response_model=StockListResponse)
def list_stocks(
    market: Optional[str] = Query(None, description="Market filter: CN or US"),
    index: Optional[str] = Query(None, description="Index filter: hs300, zz500, zz1000"),
    search: Optional[str] = Query(None, description="Search by code or name"),
    industry: Optional[str] = Query(None, description="Industry filter"),
    completeness: Optional[str] = Query(None, description="Completeness filter"),
    recently_analyzed: Optional[bool] = Query(None, description="Recently analyzed filter"),
    has_fundamentals: Optional[bool] = Query(None, description="Has standardized fundamentals"),
    has_profile: Optional[bool] = Query(None, description="Has company profile"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    items, total = data_service.get_stocks(
        market=market,
        index_filter=index,
        search=search,
        industry=industry,
        completeness=completeness,
        recently_analyzed=recently_analyzed,
        has_fundamentals=has_fundamentals,
        has_profile=has_profile,
        offset=offset,
        limit=limit,
    )
    return StockListResponse(total=total, items=[StockInfo(**item) for item in items])


@router.get("/stocks/{ts_code}", response_model=StockInfo)
def get_stock(ts_code: str):
    detail = data_service.get_stock_detail(ts_code)
    if not detail:
        raise HTTPException(status_code=404, detail="Stock not found")
    return StockInfo(**detail)


@router.get("/stocks/{ts_code}/dossier", response_model=StockDossierResponse)
def get_stock_dossier(ts_code: str):
    detail = data_service.get_stock_dossier(ts_code)
    if not detail:
        raise HTTPException(status_code=404, detail="Stock not found")
    return StockDossierResponse(**detail)


@router.get("/stocks/{ts_code}/overview", response_model=StockOverviewResponse)
def get_stock_overview(ts_code: str):
    detail = data_service.get_stock_overview(ts_code)
    if not detail:
        raise HTTPException(status_code=404, detail="Stock not found")
    return StockOverviewResponse(**detail)


@router.get("/stocks/{ts_code}/ohlcv", response_model=OHLCVResponse)
def get_ohlcv(
    ts_code: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    records = data_service.get_ohlcv(ts_code, start_date, end_date)
    return OHLCVResponse(ts_code=ts_code, records=records, total=len(records))


@router.get("/stocks/{ts_code}/competitors", response_model=list[CompetitorInfo])
def get_competitors(ts_code: str, limit: int = Query(10, ge=1, le=50)):
    return [CompetitorInfo(**item) for item in data_service.get_competitors(ts_code, limit=limit)]


@router.post("/import")
def import_csv_data():
    """Import all CSV data from data/ directories into SQLite."""
    stats = data_service.import_csv_data()
    return stats
