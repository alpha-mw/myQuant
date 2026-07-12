/*
 * Tracked compatibility loader only. Real portfolio snapshots are generated to
 * the ignored portfolio_dashboard/private/ directory and loaded first.
 */
window.DashboardSnapshotV2 = window.DashboardSnapshotV2 || {
  schema_version: "dashboard_contract.v2",
  schema_sha256: "07df194f84e33aa269cacbb707927ec708b458edc1c674cfb8a49a923ddad5a5",
  protocol_hash: "00d86cad4903a73f10b80a9055a38980df0fb61b639dc4320f6ec3cb23d648b0",
  run_id: "sample_dashboard_v2",
  generated_at: "2026-01-01T00:00:00+08:00",
  status: "sample",
  blockers: ["sample_data_not_for_investment_use"],
  as_of_matrix: {
    strategy_record_date: null,
    strategy_record_at: null,
    analysis_trading_date: null,
    quote_at: null,
    benchmark_value_dates: {},
    theme_date: null,
    factor_registry_sha: null
  },
  trading_calendar: {
    status: "missing",
    source_system: "strict_parquet.cn_bars.trade_date",
    path_summary: null,
    start_date: null,
    end_date: null,
    expected_open_dates: [],
    expected_open_date_count: 0,
    first_open_date: null,
    last_open_date: null,
    mask_sha256: null
  },
  nav_return_provenance: {
    source_field: "sample",
    return_method: "time_weighted_unitization",
    gross_or_net: "unknown",
    trade_fee_inclusion: "unknown",
    secondary_fee_adjustment_allowed: false
  },
  sources: {},
  nav: [],
  positions: [],
  trades: [],
  themes: [],
  theme_protocol: {
    schema_version: "theme_protocol.v2",
    protocol_hash: null,
    status: "blocked",
    blockers: ["sample_data_not_for_investment_use"],
    observer_enabled: null,
    formal_enabled: null,
    formal_kill_switch: null,
    formal_pool: [],
    formal_pool_count: 0,
    formal_producer: null,
    rollback_status: null,
    rollback_reason: null,
    lane_counts: null,
    readback_verified: false,
    artifact_sha256: null
  },
  factors: [],
  factor_protocol: {
    schema_version: "factor-governance-protocol.v2",
    protocol_version: "v2",
    expected_protocol_hash: null,
    protocol_hash: null,
    protocol_hash_match: false,
    status: "blocked",
    blockers: ["sample_data_not_for_investment_use"],
    evidence_hash: null,
    evidence_status: "missing",
    transition_id: null,
    transition_hash: null,
    transition_applied: false,
    rollback_status: "not_available",
    before_registry_sha256: null,
    after_registry_sha256: null,
    readback_verified: false,
    artifact_sha256: null,
    canonical_producer_available: false,
    canonical_production_apply_eligible: false,
    canonical_producer_blocker: "canonical_full_chain_replay_producer_unavailable"
  },
  reconciliation: {
    tolerance: 0.0001,
    daily: [],
    valid_nav_return_days: 0,
    covered_days: 0,
    coverage_ratio: 0,
    reconciled_days: 0,
    status: "partial",
    coverage_basis: "strict_parquet_trade_date_mask",
    calendar_status: "missing",
    blockers: ["attribution_formal_trading_calendar_missing"],
    diagnostics: {
      excluded_nav_return_dates: [],
      excluded_position_effective_dates: [],
      positions_missing_effective_date_count: 0,
      position_dates_without_nav_return: []
    }
  },
  metric_policy: {
    returns_unit: "decimal",
    contribution_formula: "nav_weight * daily_return",
    annualization_min_open_day_coverage: 0.95,
    annualization_min_valid_daily_returns: 60,
    annualization_insufficient_status: "insufficient_daily_history",
    rolling_window_min_open_day_coverage: 0.95,
    trading_calendar_required: true,
    excess_curve: "relative_wealth_ratio",
    monthly_return: "previous_month_end_anchor",
    unknown_numeric: null
  }
};
window.DashboardGeneratedRecords = window.DashboardGeneratedRecords || {
  generatedAt: "",
  sourceRoot: "",
  latestRecord: "",
  recordCount: 0,
  warnings: [],
  infos: ["未加载私有 Dashboard v2 快照，当前使用公开模拟 sample。"],
  contract: window.DashboardSnapshotV2,
  csv: { nav: "", positions: "", trades: "" }
};
