"""Stock universe preset endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/universe", tags=["universe"])


class UniversePreset(BaseModel):
    key: str
    label: str
    description: str
    estimated_count: int


class UniversePresetsResponse(BaseModel):
    market: str
    presets: list[UniversePreset]


class UniverseSymbolsResponse(BaseModel):
    market: str
    key: str
    symbols: list[str]
    count: int


class UniverseResolveRequest(BaseModel):
    keys: list[str] = Field(min_length=1)
    operation: Literal["replace", "merge"] = "replace"
    existing_pool: list[str] = Field(default_factory=list)


class UniverseResolveResponse(BaseModel):
    market: str
    symbols: list[str]
    count: int
    resolved_keys: list[str]
    selection_meta: dict[str, Any]


_CN_PRESETS = [
    UniversePreset(key="hs300",   label="沪深300",     description="HS300 大盘蓝筹",           estimated_count=300),
    UniversePreset(key="zz500",   label="中证500",     description="ZZ500 中盘成长",           estimated_count=500),
    UniversePreset(key="zz1000",  label="中证1000",    description="ZZ1000 小盘活跃",          estimated_count=1000),
    UniversePreset(key="major",   label="主要指数全部", description="HS300+ZZ500+ZZ1000 去重",  estimated_count=1800),
    UniversePreset(key="all_a",   label="全部A股",     description="沪深两市全部A股",           estimated_count=5000),
]

_US_PRESETS = [
    UniversePreset(key="large_cap", label="Large Cap",  description="S&P 500 大盘股", estimated_count=500),
    UniversePreset(key="mid_cap",   label="Mid Cap",    description="中盘股",          estimated_count=400),
    UniversePreset(key="small_cap", label="Small Cap",  description="小盘股",          estimated_count=600),
    UniversePreset(key="all",       label="All US",     description="全部美股池",       estimated_count=1500),
]


@router.get("/{market}/presets", response_model=UniversePresetsResponse)
async def get_presets(market: str):
    market = market.upper()
    if market == "CN":
        return UniversePresetsResponse(market="CN", presets=_CN_PRESETS)
    if market == "US":
        return UniversePresetsResponse(market="US", presets=_US_PRESETS)
    raise HTTPException(status_code=400, detail="market must be CN or US")


@router.get("/{market}/{key}/symbols", response_model=UniverseSymbolsResponse)
async def get_symbols(market: str, key: str):
    market = market.upper()

    if market == "CN":
        symbols = await _fetch_cn_symbols(key)
    elif market == "US":
        symbols = await _fetch_us_symbols(key)
    else:
        raise HTTPException(status_code=400, detail="market must be CN or US")

    if symbols is None:
        raise HTTPException(status_code=404, detail=f"Unknown universe key: {key}")

    return UniverseSymbolsResponse(market=market, key=key, symbols=symbols, count=len(symbols))


@router.post("/{market}/resolve", response_model=UniverseResolveResponse)
async def resolve_universe(market: str, body: UniverseResolveRequest):
    """
    Resolve one or more universe keys into a deduplicated, lexicographically sorted
    symbol list.  If operation='merge', the existing_pool is unioned with the result.
    Returns symbols, count, resolved_keys, and selection_meta.
    """
    market = market.upper()
    if market not in ("CN", "US"):
        raise HTTPException(status_code=400, detail="market must be CN or US")

    per_key: dict[str, list[str]] = {}
    for key in body.keys:
        if market == "CN":
            syms = await _fetch_cn_symbols(key)
        else:
            syms = await _fetch_us_symbols(key)
        if syms is None:
            raise HTTPException(status_code=404, detail=f"Unknown universe key: {key}")
        per_key[key] = syms

    # Dedupe across keys, then merge/replace
    combined: set[str] = set()
    for syms in per_key.values():
        combined.update(syms)

    if body.operation == "merge" and body.existing_pool:
        combined.update(body.existing_pool)

    final = sorted(combined)

    selection_meta: dict[str, Any] = {
        "operation": body.operation,
        "keys": body.keys,
        "per_key_counts": {k: len(v) for k, v in per_key.items()},
        "total_before_dedupe": sum(len(v) for v in per_key.values()),
        "total_after_dedupe": len(final),
        "merged_from_existing": len(body.existing_pool) if body.operation == "merge" else 0,
    }

    return UniverseResolveResponse(
        market=market,
        symbols=final,
        count=len(final),
        resolved_keys=list(per_key.keys()),
        selection_meta=selection_meta,
    )


async def _fetch_cn_symbols(key: str) -> list[str] | None:
    import asyncio
    from functools import partial

    def _sync_fetch(k: str) -> list[str] | None:
        try:
            from quant_investor.data.universe.cn_universe import StockUniverse
            u = StockUniverse()
            if k == "hs300":
                return u.get_hs300()
            if k == "zz500":
                return u.get_zz500()
            if k == "zz1000":
                return u.get_zz1000()
            if k == "major":
                return u.get_major_indices()
            if k == "all_a":
                return u.get_all_stocks()
            return None
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, partial(_sync_fetch, key))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Tushare error: {exc}") from exc


async def _fetch_us_symbols(key: str) -> list[str] | None:
    from quant_investor.data.universe.us_universe import USStockUniverse
    u = USStockUniverse()
    if key == "all":
        return u.get_all_symbols()
    return u.get_by_category(key) or None
