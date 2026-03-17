import { api } from './client';
import type {
  HoldingUpsertRequest,
  PortfolioMutationResponse,
  PortfolioStateResponse,
  WatchlistUpsertRequest,
} from '../types/api';

export function fetchPortfolioState() {
  return api.get<PortfolioStateResponse>('/portfolio');
}

export function upsertHolding(payload: HoldingUpsertRequest) {
  return api.post<PortfolioMutationResponse>('/portfolio/holdings', payload);
}

export function deleteHolding(holdingId: number) {
  return api.delete<PortfolioMutationResponse>(`/portfolio/holdings/${holdingId}`);
}

export function upsertWatchlist(payload: WatchlistUpsertRequest) {
  return api.post<PortfolioMutationResponse>('/portfolio/watchlist', payload);
}

export function deleteWatchlist(symbol: string) {
  return api.delete<PortfolioMutationResponse>(`/portfolio/watchlist/${symbol}`);
}
