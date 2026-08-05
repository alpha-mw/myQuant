export interface DatabaseStats {
  total_stocks: number;
  cn_count: number;
  us_count: number;
  hs300_count: number;
  zz500_count: number;
  zz1000_count: number;
  total_records: number;
  stocks_with_data: number;
  date_range: string;
  last_data_update: string | null;
}

export interface DataAvailability {
  ready: boolean;
  updated_at: string | null;
  source: string | null;
  note: string | null;
}

export interface DataCompleteness {
  technical: DataAvailability;
  fundamentals: DataAvailability;
  industry: DataAvailability;
  competitors: DataAvailability;
  business: DataAvailability;
  profile: DataAvailability;
}

export interface CompletenessCounts {
  technical_ready: number;
  fundamentals_ready: number;
  industry_ready: number;
  competitors_ready: number;
  business_ready: number;
  profile_ready: number;
}

export interface SectorDistributionItem {
  market: string;
  industry: string;
  count: number;
}

export interface MarketPulse {
  sampled_stocks: number;
  rising_count_20d: number;
  positive_ratio_20d: number;
  avg_return_20d: number;
  avg_volatility_20d: number;
  risk_state: string;
  breadth_label: string;
  last_trade_date: string | null;
}

export interface CandidateItem {
  symbol: string;
  title: string;
  created_at: string;
  summary: string;
}

export interface MarketOverviewResponse {
  summary: DatabaseStats;
  completeness: CompletenessCounts;
  market_pulse: MarketPulse;
  sector_distribution: SectorDistributionItem[];
  candidate_symbols: string[];
  watch_candidates: CandidateItem[];
}

export interface StockInfo {
  ts_code: string;
  name: string | null;
  industry: string | null;
  market: string | null;
  list_date: string | null;
  is_hs300: boolean;
  is_zz500: boolean;
  is_zz1000: boolean;
  last_update: string | null;
  record_count: number;
  date_start: string | null;
  date_end: string | null;
  latest_close: number | null;
  change_pct: number | null;
  has_profile: boolean;
  has_fundamentals: boolean;
  recently_analyzed: boolean;
  completeness: DataCompleteness;
}

export interface StockListResponse {
  total: number;
  items: StockInfo[];
}

export interface OHLCVRecord {
  trade_date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount: number;
}

export interface OHLCVResponse {
  ts_code: string;
  records: OHLCVRecord[];
  total: number;
}

export interface CompetitorInfo {
  ts_code: string;
  name: string | null;
  industry: string | null;
  latest_close: number | null;
  record_count: number;
  reason?: string | null;
  similarity_score?: number | null;
}

export interface StockMetric {
  label: string;
  value: string;
  tone: string;
}

export interface StockFactorSignal {
  key: string;
  label: string;
  value: number | null;
  display_value: string;
  signal: string;
  description: string;
}

export interface StockAnalysisMention {
  analysis_id: string;
  created_at: string;
  source: string;
  title: string;
  candidate: boolean;
  summary: string;
}

export interface QuoteOverview {
  latest_close: number | null;
  previous_close: number | null;
  change_pct: number | null;
  return_20d: number | null;
  return_60d: number | null;
  volatility_20d: number | null;
  avg_volume_20d: number | null;
  high_52w: number | null;
  low_52w: number | null;
  support_level: number | null;
  resistance_level: number | null;
}

export interface FundamentalSnapshot {
  report_period: string | null;
  currency: string | null;
  revenue?: number | null;
  net_income?: number | null;
  gross_margin?: number | null;
  operating_margin?: number | null;
  roe?: number | null;
  roa?: number | null;
  debt_to_asset?: number | null;
  pe_ttm?: number | null;
  pb?: number | null;
  ps?: number | null;
  market_cap?: number | null;
  total_assets?: number | null;
  total_liabilities?: number | null;
  shareholder_equity?: number | null;
  operating_cashflow?: number | null;
  free_cashflow?: number | null;
  source?: string | null;
  fetched_at?: string | null;
}

export interface FundamentalSeriesPoint {
  metric_name: string;
  label: string;
  period: string;
  value: number | null;
}

export interface IndustryContext {
  market: string | null;
  sector: string | null;
  industry: string | null;
  industry_stock_count: number;
  peer_count: number;
  summary: string;
  notes: string[];
}

export interface BusinessProfile {
  summary: string;
  products: string[];
  business_lines: string[];
  website: string | null;
  city: string | null;
  region: string | null;
  country: string | null;
  employees: number | null;
  source: string | null;
  fetched_at: string | null;
}

export interface TechnicalOverview {
  key_metrics: StockMetric[];
  company_metrics: StockMetric[];
  factors: StockFactorSignal[];
  notes: string[];
}

export interface StockOverviewResponse {
  stock: StockInfo;
  display_name: string;
  profile_summary: string;
  tags: string[];
  key_metrics: StockMetric[];
  company_metrics: StockMetric[];
  factors: StockFactorSignal[];
  recent_analysis: StockAnalysisMention[];
}

export interface StockDossierResponse {
  stock: StockInfo;
  display_name: string;
  profile_summary: string;
  tags: string[];
  completeness: DataCompleteness;
  quote: QuoteOverview;
  technical: TechnicalOverview;
  fundamentals: FundamentalSnapshot;
  fundamental_series: FundamentalSeriesPoint[];
  industry_context: IndustryContext;
  competitors: CompetitorInfo[];
  business_profile: BusinessProfile;
  analysis_history: StockAnalysisMention[];
}

export interface CredentialStatus {
  name: string;
  env_key: string;
  is_set: boolean;
  masked_value: string;
}

export interface BacktestDefaults {
  initial_cash: number;
  commission_rate: number;
  stamp_duty_rate: number;
  slippage: number;
}

export interface SettingsResponse {
  credentials: CredentialStatus[];
  backtest: BacktestDefaults;
  db_path: string;
  log_level: string;
}

export interface UserHolding {
  holding_id: number;
  account_name: string;
  symbol: string;
  name: string | null;
  market: string;
  quantity: number;
  cost_basis: number | null;
  notes: string;
  created_at: string;
  updated_at: string;
}

export interface WatchlistEntry {
  symbol: string;
  name: string | null;
  market: string;
  priority: string;
  notes: string;
  created_at: string;
  updated_at: string;
}

export interface PortfolioSummary {
  account_count: number;
  accounts: string[];
  holdings_count: number;
  watchlist_count: number;
  holdings_by_account: Record<string, number>;
  holding_symbols: string[];
  watchlist_symbols: string[];
}

export interface PortfolioStateResponse {
  holdings: UserHolding[];
  watchlist: WatchlistEntry[];
  summary: PortfolioSummary;
}

export interface HoldingUpsertRequest {
  holding_id?: number | null;
  account_name?: string;
  symbol: string;
  name?: string | null;
  market?: string | null;
  quantity: number;
  cost_basis?: number | null;
  notes?: string;
}

export interface WatchlistUpsertRequest {
  symbol: string;
  name?: string | null;
  market?: string | null;
  priority?: string;
  notes?: string;
}

export interface PortfolioMutationResponse {
  ok: boolean;
  message: string;
  state: PortfolioStateResponse;
}
