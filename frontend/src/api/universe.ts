import { apiFetch } from './client'

export interface UniversePreset {
  key: string
  label: string
  description: string
  estimated_count: number
}

export interface UniversePresetsResponse {
  market: string
  presets: UniversePreset[]
}

export interface UniverseSymbolsResponse {
  market: string
  key: string
  symbols: string[]
  count: number
}

export interface UniverseResolveRequest {
  keys: string[]
  operation?: 'replace' | 'merge'
  existing_pool?: string[]
}

export interface UniverseResolveResponse {
  market: string
  symbols: string[]
  count: number
  resolved_keys: string[]
  selection_meta: Record<string, unknown>
}

export function getUniversePresets(market: string) {
  return apiFetch<UniversePresetsResponse>(`/api/universe/${market}/presets`)
}

export function getUniverseSymbols(market: string, key: string) {
  return apiFetch<UniverseSymbolsResponse>(`/api/universe/${market}/${key}/symbols`)
}

export function resolveUniverse(market: string, body: UniverseResolveRequest) {
  return apiFetch<UniverseResolveResponse>(`/api/universe/${market}/resolve`, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}
