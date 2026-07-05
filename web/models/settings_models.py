"""Pydantic models for settings API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CredentialStatus(BaseModel):
    name: str
    env_key: str
    is_set: bool
    masked_value: str = ""


class CredentialsStatusResponse(BaseModel):
    credentials: list[CredentialStatus]


class BacktestDefaults(BaseModel):
    initial_cash: float = 1000000.0
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.001
    slippage: float = 0.001


class DatabaseFileSummary(BaseModel):
    path: str
    exists: bool
    size_bytes: int | None = None
    modified_at: str | None = None


class WorkspaceDatabaseSummary(DatabaseFileSummary):
    run_count: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    preset_count: int = 0
    pending_trades: int = 0
    last_run_at: str | None = None


class SettingsResponse(BaseModel):
    credentials: list[CredentialStatus]
    backtest: BacktestDefaults
    db_path: str
    log_level: str
    stock_db: DatabaseFileSummary
    workspace_db: WorkspaceDatabaseSummary


class SettingsUpdateRequest(BaseModel):
    # API keys - only set if provided (non-None)
    tushare_token: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    dashscope_api_key: Optional[str] = None
    kimi_api_key: Optional[str] = None
    # Backtest defaults
    initial_cash: Optional[float] = None
    commission_rate: Optional[float] = None
    stamp_duty_rate: Optional[float] = None
    slippage: Optional[float] = None
    log_level: Optional[str] = None
