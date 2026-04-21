import { create } from 'zustand'
import type { ResearchRunRequest } from '../types/research'

const DEFAULTS: ResearchRunRequest = {
  stock_pool: [],
  market: 'CN',
  capital: 1_000_000,
  risk_level: '中等',
  lookback_years: 1.0,
  kline_backend: 'hybrid',
  enable_macro: true,
  enable_quant: true,
  enable_kline: true,
  enable_fundamental: true,
  enable_intelligence: true,
  enable_agent_layer: true,
  review_model_priority: [],
  agent_model: '',
  agent_fallback_model: '',
  master_model: '',
  master_fallback_model: '',
  agent_timeout: 180,
  master_timeout: 900,
  stock_input_mode: 'custom',
  universe_keys: [],
  universe_operation: 'replace',
}

interface ResearchFormState extends ResearchRunRequest {
  setField: <K extends keyof ResearchRunRequest>(key: K, value: ResearchRunRequest[K]) => void
  loadPreset: (presetId: string, config: Partial<ResearchRunRequest>) => void
  reset: () => void
  toRequest: () => ResearchRunRequest
}

export const useResearchStore = create<ResearchFormState>((set, get) => ({
  ...DEFAULTS,
  setField: (key, value) =>
    set((state) => {
      if (key === 'preset_id') {
        return { preset_id: value as string | undefined }
      }

      if (key === 'market') {
        const nextMarket = value as ResearchRunRequest['market']
        if (state.market === nextMarket) {
          return {}
        }
        return {
          market: nextMarket,
          stock_pool: [],
          universe_keys: [],
          preset_id: undefined,
        }
      }

      return {
        [key]: value,
        preset_id: undefined,
      } as Partial<ResearchFormState>
    }),
  loadPreset: (presetId, config) => set({ ...DEFAULTS, ...config, preset_id: presetId }),
  reset: () => set(DEFAULTS),
  toRequest: () => {
    const state = get()
    return {
      stock_pool: state.stock_pool,
      market: state.market,
      capital: state.capital,
      risk_level: state.risk_level,
      lookback_years: state.lookback_years,
      kline_backend: state.kline_backend,
      enable_macro: state.enable_macro,
      enable_quant: state.enable_quant,
      enable_kline: state.enable_kline,
      enable_fundamental: state.enable_fundamental,
      enable_intelligence: state.enable_intelligence,
      enable_agent_layer: state.enable_agent_layer,
      review_model_priority: state.review_model_priority,
      agent_model: state.agent_model,
      agent_fallback_model: state.agent_fallback_model,
      master_model: state.master_model,
      master_fallback_model: state.master_fallback_model,
      agent_timeout: state.agent_timeout,
      master_timeout: state.master_timeout,
      preset_id: state.preset_id,
      stock_input_mode: state.stock_input_mode,
      universe_keys: state.universe_keys,
      universe_operation: state.universe_operation,
    }
  },
}))
