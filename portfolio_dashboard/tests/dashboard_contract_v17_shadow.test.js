"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const dashboard = require(path.join(root, "js", "v17_shadow.js"));
const schema = JSON.parse(
  fs.readFileSync(
    path.join(root, "schema", "dashboard_contract.v17-shadow.schema.json"),
    "utf8",
  ),
);
const html = fs.readFileSync(path.join(root, "v17_shadow.html"), "utf8");
const loaderSource = fs.readFileSync(path.join(root, "js", "v17_shadow.js"), "utf8");

const H = "a".repeat(64);
const G = "b".repeat(64);

function validContract() {
  const runId = "cn-v17-dashboard-test";
  const terminal = "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION";
  return {
    schema_version: "dashboard_contract.v17-shadow.v1",
    schema_sha256: H,
    availability: "AVAILABLE",
    generated_at: "2026-07-22T08:05:00Z",
    reason: null,
    source: {
      path: "results/v17_shadow/_latest/shadow.json",
      latest_pointer_sha256: G,
      ledger_sha256: H,
      output_sha256: G,
      readback_verified: true,
      fallback_used: false,
    },
    latest_pointer: {
      version: "myquant.v17.shadow-latest-pointer.v1",
      run_id: runId,
      terminal_state: terminal,
      ledger_path: `results/v17_shadow/runs/${runId}/ledger.json`,
      ledger_sha256: H,
      output_path: `results/v17_shadow/outcomes/${runId}-0005-test.json`,
      output_sha256: G,
      published_at: "2026-07-22T08:04:00Z",
      publication_mode: "NORMAL",
      authority: false,
      semantic_sha256: H,
    },
    terminal_output: {
      version: "myquant.v17.shadow-output.v1",
      run_id: runId,
      strategy_id: "cn-shadow",
      market: "CN",
      cutoff: "2026-07-22T07:00:00Z",
      terminal_state: terminal,
      rank_output: {
        initial_ranked_symbols: ["000001.SZ"],
        eligible_ranked_symbols: ["000001.SZ"],
        sealed_symbols: ["000001.SZ"],
        rows: [
          {
            symbol: "000001.SZ",
            deep_status: "DEEP_RESEARCH_COMPLETE",
            f_eligible: true,
            timing_state: "BUY_NOW",
          },
        ],
      },
      portfolio_output: { weights: [{ symbol: "000001.SZ", weight: 0.05 }] },
      blockers: [],
      source_manifest_sha256: G,
      ledger_predecessor_sha256: H,
      generated_at: "2026-07-22T08:03:00Z",
      authority: false,
      semantic_sha256: G,
    },
    authority: false,
  };
}

assert.strictEqual(schema.title, "myQuant Dashboard Contract v17 Shadow");
assert.strictEqual(
  schema.properties.schema_version.const,
  "dashboard_contract.v17-shadow.v1",
);
assert.strictEqual(schema.$defs.source.properties.fallback_used.const, false);
assert.strictEqual(
  schema.$defs.source.properties.path.const,
  "results/v17_shadow/_latest/shadow.json",
);

assert.ok(html.includes('src="generated/v17_shadow_latest.js"'));
assert.ok(html.includes('src="js/v17_shadow.js"'));
assert.ok(!html.includes("generated_records.js"));
assert.ok(!html.includes("dashboard_snapshot.v3"));
assert.ok(!loaderSource.includes("DashboardSnapshotV3"));
assert.ok(!loaderSource.includes("DashboardGeneratedRecords"));
assert.ok(!loaderSource.includes("sample/dashboard_snapshot"));

const available = dashboard.normalize(validContract());
assert.strictEqual(available.availability, "AVAILABLE");
assert.strictEqual(available.output.run_id, "cn-v17-dashboard-test");
assert.strictEqual(available.isBusinessTerminal, true);
assert.deepStrictEqual(dashboard.rankedRows(available.output.rank_output), [
  {
    rank: 1,
    symbol: "000001.SZ",
    name: "UNKNOWN_NAME",
    fundamental: "F_ELIGIBLE / DEEP_RESEARCH_COMPLETE",
    timing: "BUY_NOW",
  },
]);

assert.deepStrictEqual(
  dashboard.rankedRows({ ranked_symbols: ["000001.SZ"] }),
  [],
);

const missing = dashboard.normalize(undefined);
assert.strictEqual(missing.availability, "UNAVAILABLE");
assert.strictEqual(missing.reason, "v17_latest_loader_missing");

const unavailableContract = validContract();
unavailableContract.availability = "UNAVAILABLE";
unavailableContract.reason = "v17_latest_pointer_missing";
unavailableContract.source.latest_pointer_sha256 = null;
unavailableContract.source.ledger_sha256 = null;
unavailableContract.source.output_sha256 = null;
unavailableContract.source.readback_verified = false;
unavailableContract.latest_pointer = null;
unavailableContract.terminal_output = null;
const explicitlyUnavailable = dashboard.normalize(unavailableContract);
assert.strictEqual(explicitlyUnavailable.availability, "UNAVAILABLE");
assert.strictEqual(explicitlyUnavailable.reason, "v17_latest_pointer_missing");

const crossBoundDrift = validContract();
crossBoundDrift.terminal_output.terminal_state = "SHADOW_PORTFOLIO_INFEASIBLE";
crossBoundDrift.terminal_output.portfolio_output = null;
crossBoundDrift.terminal_output.blockers = ["portfolio_infeasible"];
const rejected = dashboard.normalize(crossBoundDrift);
assert.strictEqual(rejected.availability, "UNAVAILABLE");
assert.strictEqual(rejected.reason, "v17_dashboard_cross_binding_invalid");

const legacyGlobalsAreIgnored = dashboard.normalize({
  DashboardSnapshotV3: { status: "fresh" },
  DashboardGeneratedRecords: { recordCount: 1 },
});
assert.strictEqual(legacyGlobalsAreIgnored.availability, "UNAVAILABLE");

console.log("dashboard_contract_v17_shadow.test.js: PASS");
