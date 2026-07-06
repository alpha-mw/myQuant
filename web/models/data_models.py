"""Pydantic models for data-related API endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DatabaseStats(BaseModel):
    total_stocks: int = 0
    cn_count: int = 0
    us_count: int = 0
    hs300_count: int = 0
    zz500_count: int = 0
    zz1000_count: int = 0
    total_records: int = 0
    stocks_with_data: int = 0
    date_range: str = "N/A"
    last_data_update: Optional[str] = None


class DataAvailability(BaseModel):
    ready: bool = False
    updated_at: Optional[str] = None
    source: Optional[str] = None
    note: Optional[str] = None


class DataCompleteness(BaseModel):
    technical: DataAvailability = Field(default_factory=DataAvailability)
    fundamentals: DataAvailability = Field(default_factory=DataAvailability)
    industry: DataAvailability = Field(default_factory=DataAvailability)
    competitors: DataAvailability = Field(default_factory=DataAvailability)
    business: DataAvailability = Field(default_factory=DataAvailability)
    profile: DataAvailability = Field(default_factory=DataAvailability)


class CompletenessCounts(BaseModel):
    technical_ready: int = 0
    fundamentals_ready: int = 0
    industry_ready: int = 0
    competitors_ready: int = 0
    business_ready: int = 0
    profile_ready: int = 0


class SectorDistributionItem(BaseModel):
    market: str = "CN"
    industry: str = "未分类"
    count: int = 0


class MarketPulse(BaseModel):
    sampled_stocks: int = 0
    rising_count_20d: int = 0
    positive_ratio_20d: float = 0.0
    avg_return_20d: float = 0.0
    avg_volatility_20d: float = 0.0
    risk_state: str = "neutral"
    breadth_label: str = "观望"
    last_trade_date: Optional[str] = None


class CandidateItem(BaseModel):
    symbol: str
    title: str = ""
    created_at: str = ""
    summary: str = ""


class MarketOverviewResponse(BaseModel):
    summary: DatabaseStats = Field(default_factory=DatabaseStats)
    completeness: CompletenessCounts = Field(default_factory=CompletenessCounts)
    market_pulse: MarketPulse = Field(default_factory=MarketPulse)
    sector_distribution: list[SectorDistributionItem] = Field(default_factory=list)
    candidate_symbols: list[str] = Field(default_factory=list)
    watch_candidates: list[CandidateItem] = Field(default_factory=list)


class StockInfo(BaseModel):
    ts_code: str
    name: Optional[str] = None
    industry: Optional[str] = None
    market: Optional[str] = None
    list_date: Optional[str] = None
    is_hs300: bool = False
    is_zz500: bool = False
    is_zz1000: bool = False
    last_update: Optional[str] = None
    record_count: int = 0
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    latest_close: Optional[float] = None
    change_pct: Optional[float] = None
    has_profile: bool = False
    has_fundamentals: bool = False
    recently_analyzed: bool = False
    completeness: DataCompleteness = Field(default_factory=DataCompleteness)


class StockListResponse(BaseModel):
    total: int
    items: list[StockInfo]


class OHLCVRecord(BaseModel):
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0


class OHLCVResponse(BaseModel):
    ts_code: str
    records: list[OHLCVRecord]
    total: int


class DownloadRequest(BaseModel):
    market: str = "CN"
    start_date: str = "20200101"
    end_date: Optional[str] = None
    batch_size: int = 100


class DownloadStatusResponse(BaseModel):
    total_stocks: int
    completed_stocks: int
    failed_stocks: list[str]
    progress_pct: float
    status: str


class CompetitorInfo(BaseModel):
    ts_code: str
    name: Optional[str] = None
    industry: Optional[str] = None
    latest_close: Optional[float] = None
    record_count: int = 0
    reason: Optional[str] = None
    similarity_score: Optional[float] = None


class StockMetric(BaseModel):
    label: str
    value: str
    tone: str = "neutral"


class StockFactorSignal(BaseModel):
    key: str
    label: str
    value: Optional[float] = None
    display_value: str = "-"
    signal: str = "neutral"
    description: str = ""


class StockAnalysisMention(BaseModel):
    analysis_id: str
    created_at: str
    source: str = "legacy"
    title: str = ""
    candidate: bool = False
    summary: str = ""


class QuoteOverview(BaseModel):
    latest_close: Optional[float] = None
    previous_close: Optional[float] = None
    change_pct: Optional[float] = None
    return_20d: Optional[float] = None
    return_60d: Optional[float] = None
    volatility_20d: Optional[float] = None
    avg_volume_20d: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


class FundamentalSnapshot(BaseModel):
    report_period: Optional[str] = None
    currency: Optional[str] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_asset: Optional[float] = None
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    market_cap: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    shareholder_equity: Optional[float] = None
    operating_cashflow: Optional[float] = None
    free_cashflow: Optional[float] = None
    source: Optional[str] = None
    fetched_at: Optional[str] = None


class FundamentalSeriesPoint(BaseModel):
    metric_name: str
    label: str
    period: str
    value: Optional[float] = None


class IndustryContext(BaseModel):
    market: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    industry_stock_count: int = 0
    peer_count: int = 0
    summary: str = ""
    notes: list[str] = Field(default_factory=list)


class BusinessProfile(BaseModel):
    summary: str = ""
    products: list[str] = Field(default_factory=list)
    business_lines: list[str] = Field(default_factory=list)
    website: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    employees: Optional[int] = None
    source: Optional[str] = None
    fetched_at: Optional[str] = None


class TechnicalOverview(BaseModel):
    key_metrics: list[StockMetric] = Field(default_factory=list)
    company_metrics: list[StockMetric] = Field(default_factory=list)
    factors: list[StockFactorSignal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StockOverviewResponse(BaseModel):
    stock: StockInfo
    display_name: str
    profile_summary: str
    tags: list[str]
    key_metrics: list[StockMetric]
    company_metrics: list[StockMetric]
    factors: list[StockFactorSignal]
    recent_analysis: list[StockAnalysisMention]


class StockDossierResponse(BaseModel):
    stock: StockInfo
    display_name: str
    profile_summary: str
    tags: list[str] = Field(default_factory=list)
    completeness: DataCompleteness = Field(default_factory=DataCompleteness)
    quote: QuoteOverview = Field(default_factory=QuoteOverview)
    technical: TechnicalOverview = Field(default_factory=TechnicalOverview)
    fundamentals: FundamentalSnapshot = Field(default_factory=FundamentalSnapshot)
    fundamental_series: list[FundamentalSeriesPoint] = Field(default_factory=list)
    industry_context: IndustryContext = Field(default_factory=IndustryContext)
    competitors: list[CompetitorInfo] = Field(default_factory=list)
    business_profile: BusinessProfile = Field(default_factory=BusinessProfile)
    analysis_history: list[StockAnalysisMention] = Field(default_factory=list)
