"""Pydantic models for user holdings and watchlist endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class UserHolding(BaseModel):
    holding_id: int
    account_name: str = "默认账户"
    symbol: str
    name: Optional[str] = None
    market: str = "CN"
    quantity: float = 0.0
    cost_basis: Optional[float] = None
    notes: str = ""
    created_at: str
    updated_at: str


class WatchlistEntry(BaseModel):
    symbol: str
    name: Optional[str] = None
    market: str = "CN"
    priority: str = "normal"
    notes: str = ""
    created_at: str
    updated_at: str


class PortfolioSummary(BaseModel):
    account_count: int = 0
    accounts: list[str] = Field(default_factory=list)
    holdings_count: int = 0
    watchlist_count: int = 0
    holdings_by_account: dict[str, int] = Field(default_factory=dict)
    holding_symbols: list[str] = Field(default_factory=list)
    watchlist_symbols: list[str] = Field(default_factory=list)


class PortfolioStateResponse(BaseModel):
    holdings: list[UserHolding] = Field(default_factory=list)
    watchlist: list[WatchlistEntry] = Field(default_factory=list)
    summary: PortfolioSummary = Field(default_factory=PortfolioSummary)


class HoldingUpsertRequest(BaseModel):
    holding_id: Optional[int] = None
    account_name: str = "默认账户"
    symbol: str
    name: Optional[str] = None
    market: Optional[str] = None
    quantity: float = Field(..., gt=0)
    cost_basis: Optional[float] = Field(None, ge=0)
    notes: str = ""


class WatchlistUpsertRequest(BaseModel):
    symbol: str
    name: Optional[str] = None
    market: Optional[str] = None
    priority: str = "normal"
    notes: str = ""


class PortfolioMutationResponse(BaseModel):
    ok: bool = True
    message: str = ""
    state: PortfolioStateResponse = Field(default_factory=PortfolioStateResponse)
