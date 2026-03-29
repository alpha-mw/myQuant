import { apiFetch } from './client'
import type { Preset, PresetListResponse } from '../types/preset'
import type { ResearchRunRequest } from '../types/research'

export function listPresets() {
  return apiFetch<PresetListResponse>('/api/presets/')
}

export function createPreset(name: string, description: string, config: ResearchRunRequest) {
  return apiFetch<Preset>('/api/presets/', {
    method: 'POST',
    body: JSON.stringify({ name, description, config }),
  })
}

export function updatePreset(presetId: string, data: { name?: string; description?: string; config?: ResearchRunRequest }) {
  return apiFetch<Preset>(`/api/presets/${presetId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  })
}

export function deletePreset(presetId: string) {
  return apiFetch<{ ok: boolean }>(`/api/presets/${presetId}`, { method: 'DELETE' })
}
