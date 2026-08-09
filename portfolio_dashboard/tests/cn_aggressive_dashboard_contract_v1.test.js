"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const Contract = require(path.join(root, "js", "cn_aggressive_dashboard_contract_v1.js"));
const schema = JSON.parse(
  fs.readFileSync(path.join(root, "schema", "cn_aggressive_dashboard.v1.schema.json"), "utf8")
);
const sample = JSON.parse(
  fs.readFileSync(path.join(root, "sample", "cn_aggressive_dashboard.v1.json"), "utf8")
);
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

assert.strictEqual(schema.properties.schema_version.const, "cn_aggressive_dashboard.v1");
assert.deepStrictEqual(schema.properties.status.enum, ["FRESH", "PARTIAL", "BLOCKED"]);
assert.strictEqual(schema.properties.strategy_label.const, "aggressive_tech_manufacturing");
assert.match(sample.positions[0].name, /^合成样例/);
assert.ok(sample.warnings.includes("synthetic_sample_only"));
assert.match(html, /id="historySummary"/);
assert.match(html, /id="performanceRows"/);
assert.match(html, /3月归档以来 TWR/);
assert.strictEqual(sample.history.archive_start_record, "20990102_1200");
assert.strictEqual(sample.history.funding_events.length, 1);
assert.strictEqual(sample.portfolio.performance_points[0].date, sample.history.archive_start_date);

assert.deepStrictEqual(Contract.validateBundle(sample), { valid: true, errors: [] });
const usable = Contract.deriveSnapshot(sample);
assert.strictEqual(usable.status, "PARTIAL");
assert.strictEqual(usable.bundle, sample);

const missing = Contract.deriveSnapshot(null);
assert.strictEqual(missing.status, "BLOCKED");
assert.strictEqual(missing.bundle, null);
assert.match(missing.blockers[0], /bundle_missing/);

const writeEnabled = structuredClone(sample);
writeEnabled.authority_flags.order_calls = true;
assert.strictEqual(Contract.deriveSnapshot(writeEnabled).status, "BLOCKED");

const unboundPosition = structuredClone(sample);
unboundPosition.positions[0].evidence_status = "UNVERIFIED";
assert.strictEqual(Contract.deriveSnapshot(unboundPosition).status, "BLOCKED");

const wrongReturnMethod = structuredClone(sample);
wrongReturnMethod.portfolio.return_method = "simple_return";
assert.strictEqual(Contract.deriveSnapshot(wrongReturnMethod).status, "BLOCKED");

const truncatedHistory = structuredClone(sample);
truncatedHistory.history.archive_start_record = "20981231_1200";
assert.strictEqual(Contract.deriveSnapshot(truncatedHistory).status, "BLOCKED");

const missingBenchmark = structuredClone(sample);
missingBenchmark.benchmarks = [];
assert.strictEqual(Contract.deriveSnapshot(missingBenchmark).status, "BLOCKED");

const fakeI1 = structuredClone(sample);
fakeI1.i1_display_status = "INFERRED_FROM_FREE_TEXT";
assert.strictEqual(Contract.deriveSnapshot(fakeI1).status, "BLOCKED");

const loader = fs.readFileSync(path.join(root, "js", "cn_aggressive_input.js"), "utf8");
const sandbox = { window: { MyQuantCNAggressiveDashboard: sample } };
vm.runInNewContext(loader, sandbox);
assert.strictEqual(sandbox.window.MyQuantCNAggressiveDashboard, sample);
const emptySandbox = { window: {} };
vm.runInNewContext(loader, emptySandbox);
assert.strictEqual(emptySandbox.window.MyQuantCNAggressiveDashboard, null);

console.log("cn_aggressive_dashboard_contract_v1.test.js: PASS");
