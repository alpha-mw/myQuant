import type { ResearchRunRequest } from './research'

export interface Preset {
  preset_id: string
  name: string
  description: string
  config: ResearchRunRequest
  created_at: string
  updated_at: string
}

export interface PresetListResponse {
  presets: Preset[]
}
