import { apiFetch } from './client'
import type { V17MainlinePublicRun } from '../types/research'

export function getActiveResearchRun(
  strategyId: string,
  expectedPointerSha256?: string,
) {
  const strategy = encodeURIComponent(strategyId.trim())
  const params = new URLSearchParams()
  if (expectedPointerSha256?.trim()) {
    params.set('expected_pointer_sha256', expectedPointerSha256.trim())
  }
  const query = params.toString()
  return apiFetch<V17MainlinePublicRun>(
    `/api/research/${strategy}${query ? `?${query}` : ''}`,
  )
}
