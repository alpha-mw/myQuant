export interface ResearchRunRequest {
  stock_pool: string[]
  market: 'CN' | 'US'
  capital: number
  risk_level: string
  lookback_years: number
  kline_backend: string
  enable_macro: boolean
  enable_quant: boolean
  enable_kline: boolean
  enable_fundamental: boolean
  enable_intelligence: boolean
  enable_agent_layer: boolean
  review_model_priority: string[]
  agent_model: string
  agent_fallback_model: string
  master_model: string
  master_fallback_model: string
  agent_timeout: number
  master_timeout: number
  preset_id?: string
  // Stock-pool selection metadata
  stock_input_mode?: 'custom' | 'universe' | 'multi'
  universe_keys?: string[]
  universe_operation?: 'replace' | 'merge'
}

export interface ResearchJobResponse {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  created_at: string
  progress_pct: number
  error?: string
  result_summary?: Record<string, unknown>
}

export interface ResearchHistoryItem {
  job_id: string
  created_at: string
  status: string
  market: string
  stock_pool: string[]
  total_time?: number
  risk_level: string
  preset_id?: string
}

export interface ResearchHistoryResponse {
  items: ResearchHistoryItem[]
  total: number
}

export interface ResearchReportResponse {
  markdown: string
}

export interface RecentRunSummary {
  job_id: string
  created_at: string
  market: string
  stock_pool: string[]
  status: string
  total_time?: number
  recall_context: Record<string, unknown>
  selection_meta: Record<string, unknown>
}

export interface StartupContextResponse {
  recent_runs: RecentRunSummary[]
  suggested_trades: Record<string, unknown>[]
  recall_summary: Record<string, unknown>
}
