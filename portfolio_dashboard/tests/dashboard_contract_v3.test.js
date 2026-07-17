"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const sample = JSON.parse(
  fs.readFileSync(path.join(root, "sample", "dashboard_snapshot.v3.json"), "utf8")
);

assert.strictEqual(sample.schema_version, "dashboard_contract.v3");
assert.ok(Array.isArray(sample.industries));
assert.strictEqual(Object.prototype.hasOwnProperty.call(sample, "themes"), false);
assert.strictEqual(Object.prototype.hasOwnProperty.call(sample, "theme_protocol"), false);

const loader = fs.readFileSync(path.join(root, "js", "generated_records.js"), "utf8");
const preloadedRecords = {
  generatedAt: "2026-01-01T00:00:00+08:00",
  sourceRoot: "private",
  latestRecord: "synthetic-private-record",
  recordCount: 1,
  warnings: [],
  infos: [],
  csv: {
    nav: "date,portfolio_nav\\n2026-01-01,1",
    positions: "date,ticker,name,weight\\n2026-01-01,SYNTHETIC,Sample,1",
    trades: ""
  },
  contract: { schema_version: "dashboard_contract.v3" }
};
const sandbox = {
  window: {
    DashboardSnapshotV3: preloadedRecords.contract,
    DashboardGeneratedRecords: preloadedRecords
  }
};
vm.runInNewContext(loader, sandbox);
assert.strictEqual(
  sandbox.window.DashboardGeneratedRecords,
  preloadedRecords,
  "sanitized loader must not overwrite a private snapshot loaded before it"
);
assert.strictEqual(
  sandbox.window.DashboardGeneratedRecords.csv.nav,
  preloadedRecords.csv.nav,
  "preloaded private NAV history must remain available to the dashboard"
);
console.log("dashboard_contract_v3.test.js: PASS");
