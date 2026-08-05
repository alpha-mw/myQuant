export interface V17ArtifactRef {
  schema_id: string
  relative_path: string
  byte_sha256: string
}

export interface V17AuthorityFlags {
  broker_calls: false
  execution_calls: false
  llm_control_calls: false
  order_calls: false
  provider_calls: false
  selector_writes: false
  trade_calls: false
}

export interface V17Target {
  symbol: string
  current_target: string
  final_target: string
  lane: 'SELECTION_POOL' | 'REVIEW_ONLY_HOLDING'
}

export interface V17MainlinePublicRun {
  schema_id: 'myquant.v17.v4.mainline-public-run.v1'
  protocol: 'myquant.v17.v4'
  canonical_strategy_id: string
  run_id: string
  state: 'ACTIVE'
  market: 'CN_A_SHARE'
  capability: 'RESEARCH_PORTFOLIO'
  authority_source: 'FORMAL_V17_V4'
  authority_flags: V17AuthorityFlags
  read_only: true
  selector_used: false
  fallback_used: false
  active_pointer_ref: V17ArtifactRef
  mainline_run_ref: V17ArtifactRef
  formal_output_ref: V17ArtifactRef
  portfolio_output_ref: V17ArtifactRef
  source_closure_ref: V17ArtifactRef
  cash_weight: string
  gross_weight: string
  targets: V17Target[]
  semantic_sha256: string
}
