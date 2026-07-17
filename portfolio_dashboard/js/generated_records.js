/* Sanitized loader only. Real snapshots live under ignored private/. */
window.DashboardSnapshotV3 = window.DashboardSnapshotV3 || {
  schema_version: "dashboard_contract.v3",
  schema_sha256: "0".repeat(64),
  protocol_hash: "0".repeat(64),
  run_id: "synthetic-loader-v3",
  generated_at: null,
  status: "sample",
  blockers: [],
  v15_run_readiness: {
    schema_version: "v15_run_readiness.v1",
    path: "v15_run_readiness.json",
    sha256: "0".repeat(64)
  },
  as_of_matrix: {},
  trading_calendar: {},
  nav_return_provenance: {},
  sources: {},
  nav: [],
  positions: [],
  trades: [],
  industries: [],
  factors: [],
  factor_protocol: {
    schema_version: "factor-governance-protocol.v3",
    protocol_version: "v3",
    status: "blocked",
    blockers: ["synthetic_loader"],
    readback_verified: false
  },
  reconciliation: {},
  metric_policy: {}
};
window.DashboardGeneratedRecords = window.DashboardGeneratedRecords || {
  generatedAt: null,
  sourceRoot: null,
  latestRecord: null,
  recordCount: 0,
  warnings: [],
  infos: [],
  csv: { nav: "", positions: "", trades: "" },
  contract: window.DashboardSnapshotV3
};
