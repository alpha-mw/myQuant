import { apiFetch } from './client'
import type {
  ResearchRunRequest,
  ResearchJobResponse,
  ResearchReportResponse,
  ResearchHistoryResponse,
  StartupContextResponse,
} from '../types/research'

export function submitResearchRun(req: ResearchRunRequest) {
  return apiFetch<ResearchJobResponse>('/api/research/run', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

export function getResearchJob(jobId: string) {
  return apiFetch<ResearchJobResponse>(`/api/research/${jobId}`)
}

export function getResearchReport(jobId: string) {
  return apiFetch<ResearchReportResponse>(`/api/research/${jobId}/report`)
}

export function getResearchHistory(page = 1, perPage = 20, market?: string) {
  const params = new URLSearchParams({ page: String(page), per_page: String(perPage) })
  if (market) params.set('market', market)
  return apiFetch<ResearchHistoryResponse>(`/api/research/history/list?${params}`)
}

export function deleteResearchRun(jobId: string) {
  return apiFetch<{ ok: boolean }>(`/api/research/${jobId}`, { method: 'DELETE' })
}

export function getStartupContext() {
  return apiFetch<StartupContextResponse>('/api/research/startup-context')
}
