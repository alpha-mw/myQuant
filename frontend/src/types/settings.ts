export interface CredentialStatus {
  name: string
  env_key: string
  is_set: boolean
  masked_value: string
}

export interface LLMModelOption {
  id: string
  provider: string
  label: string
  available: boolean
  prompt_price: number
  completion_price: number
}

export interface LLMModelsResponse {
  models: LLMModelOption[]
}

export interface BacktestDefaults {
  initial_cash: number
  commission_rate: number
  stamp_duty_rate: number
  slippage: number
}

export interface SettingsResponse {
  credentials: CredentialStatus[]
  backtest: BacktestDefaults
  db_path: string
  log_level: string
}

export interface SettingsUpdateRequest {
  tushare_token?: string
  openai_api_key?: string
  anthropic_api_key?: string
  deepseek_api_key?: string
  google_api_key?: string
  fred_api_key?: string
  finnhub_api_key?: string
  dashscope_api_key?: string
  kimi_api_key?: string
  initial_cash?: number
  commission_rate?: number
  stamp_duty_rate?: number
  slippage?: number
  log_level?: string
}
