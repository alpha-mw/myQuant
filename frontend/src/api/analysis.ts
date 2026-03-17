import { api } from './client';
import type {
  AnalysisDeleteResponse,
  AnalysisHistoryResponse,
  AnalysisJobResponse,
  AnalysisOptionsResponse,
  AnalysisResult,
  AnalysisRunRequest,
  AnalysisRunResponse,
} from '../types/api';

export function fetchAnalysisHistory(limit = 10) {
  return api.get<AnalysisHistoryResponse>(`/analysis/history?limit=${limit}`);
}

export function fetchAnalysisOptions() {
  return api.get<AnalysisOptionsResponse>('/analysis/options');
}

export function fetchAnalysisResult(analysisId: string) {
  return api.get<AnalysisResult>(`/analysis/${analysisId}`);
}

export function fetchAnalysisJob(jobId: string) {
  return api.get<AnalysisJobResponse>(`/analysis/jobs/${jobId}`);
}

export function fetchRecentJobs(limit = 8) {
  return api.get<AnalysisJobResponse[]>(`/analysis/jobs?limit=${limit}`);
}

export function runAnalysis(payload: AnalysisRunRequest) {
  return api.post<AnalysisRunResponse>('/analysis/run', payload);
}

export function fetchAnalysisHistoryPaged(params: {
  limit?: number;
  offset?: number;
  search?: string;
  market?: string;
}) {
  const query = new URLSearchParams();
  if (params.limit) query.set('limit', String(params.limit));
  if (params.offset) query.set('offset', String(params.offset));
  if (params.search) query.set('search', params.search);
  if (params.market) query.set('market', params.market);
  return api.get<AnalysisHistoryResponse>(`/analysis/history?${query}`);
}

export function deleteAnalysis(analysisId: string) {
  return api.delete<AnalysisDeleteResponse>(`/analysis/${analysisId}`);
}

export function clearAnalysisHistory() {
  return api.delete<AnalysisDeleteResponse>('/analysis/history');
}
