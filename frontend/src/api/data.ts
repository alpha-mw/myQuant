import { api } from './client';
import type {
  CompetitorInfo,
  DatabaseStats,
  MarketOverviewResponse,
  OHLCVResponse,
  StockDossierResponse,
  StockInfo,
  StockListResponse,
  StockOverviewResponse,
} from '../types/api';

export function fetchStatistics() {
  return api.get<DatabaseStats>('/data/statistics');
}

export function fetchMarketOverview() {
  return api.get<MarketOverviewResponse>('/data/market/overview');
}

export function fetchStocks(params: {
  market?: string;
  index?: string;
  search?: string;
  industry?: string;
  completeness?: string;
  recently_analyzed?: boolean;
  has_fundamentals?: boolean;
  has_profile?: boolean;
  offset?: number;
  limit?: number;
}) {
  const query = new URLSearchParams();
  if (params.market) query.set('market', params.market);
  if (params.index) query.set('index', params.index);
  if (params.search) query.set('search', params.search);
  if (params.industry) query.set('industry', params.industry);
  if (params.completeness) query.set('completeness', params.completeness);
  if (params.recently_analyzed !== undefined) {
    query.set('recently_analyzed', String(params.recently_analyzed));
  }
  if (params.has_fundamentals !== undefined) {
    query.set('has_fundamentals', String(params.has_fundamentals));
  }
  if (params.has_profile !== undefined) {
    query.set('has_profile', String(params.has_profile));
  }
  if (params.offset !== undefined) query.set('offset', String(params.offset));
  if (params.limit !== undefined) query.set('limit', String(params.limit));
  return api.get<StockListResponse>(`/data/stocks?${query}`);
}

export function fetchStock(tsCode: string) {
  return api.get<StockInfo>(`/data/stocks/${tsCode}`);
}

export function fetchStockDossier(tsCode: string) {
  return api.get<StockDossierResponse>(`/data/stocks/${tsCode}/dossier`);
}

export function fetchStockOverview(tsCode: string) {
  return api.get<StockOverviewResponse>(`/data/stocks/${tsCode}/overview`);
}

export function fetchOHLCV(tsCode: string, startDate?: string, endDate?: string) {
  const query = new URLSearchParams();
  if (startDate) query.set('start_date', startDate);
  if (endDate) query.set('end_date', endDate);
  const qs = query.toString();
  return api.get<OHLCVResponse>(`/data/stocks/${tsCode}/ohlcv${qs ? `?${qs}` : ''}`);
}

export function fetchCompetitors(tsCode: string, limit = 8) {
  return api.get<CompetitorInfo[]>(`/data/stocks/${tsCode}/competitors?limit=${limit}`);
}
