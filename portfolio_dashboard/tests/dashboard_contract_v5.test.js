"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const Contract = require(path.join(root, "js", "dashboard_contract_v5.js"));
const schema = JSON.parse(
  fs.readFileSync(path.join(root, "schema", "dashboard_contract.v5.schema.json"), "utf8")
);
const sample = JSON.parse(
  fs.readFileSync(path.join(root, "sample", "dashboard_snapshot.v5.json"), "utf8")
);

const ZEROS = "0".repeat(64);
const ONES = "1".repeat(64);

function ref(schemaId, relativePath, byteSha256 = ZEROS) {
  return { schema_id: schemaId, relative_path: relativePath, byte_sha256: byteSha256 };
}

function publicRun() {
  return {
    schema_id: "myquant.v17.v4.mainline-public-run.v1",
    protocol: "myquant.v17.v4",
    canonical_strategy_id: "cn-mainline",
    run_id: "run-20260804",
    state: "ACTIVE",
    market: "CN_A_SHARE",
    capability: "RESEARCH_PORTFOLIO",
    authority_source: "FORMAL_V17_V4",
    authority_flags: {
      broker_calls: false,
      execution_calls: false,
      llm_control_calls: false,
      order_calls: false,
      provider_calls: false,
      selector_writes: false,
      trade_calls: false
    },
    read_only: true,
    selector_used: false,
    fallback_used: false,
    active_pointer_ref: ref("myquant.v17.v4.mainline-active-pointer.v1", "results/v17_mainline/strategies/cn-mainline/_active.json"),
    mainline_run_ref: ref("myquant.v17.v4.mainline-run.v1", "results/v17_mainline/strategies/cn-mainline/runs/run-20260804/run.json", ONES),
    formal_output_ref: ref("myquant.v17.v4.formal-output.v1", "results/v17_v4_formal_research/formal/run-20260804.json"),
    portfolio_output_ref: ref("myquant.v17.v4.portfolio-output.v1", "results/v17_v4_formal_research/portfolio/run-20260804.json"),
    source_closure_ref: ref("myquant.v17.v4.pit-generation-catalog.v1", "data/private/v17_v4_sources/catalog.json"),
    cash_weight: "0.2",
    gross_weight: "0.8",
    targets: [
      { symbol: "000001.SZ", current_target: "0", final_target: "0.4", lane: "SELECTION_POOL" },
      { symbol: "600000.SH", current_target: "0.2", final_target: "0.4", lane: "REVIEW_ONLY_HOLDING" }
    ],
    semantic_sha256: ONES
  };
}

assert.strictEqual(schema.properties.schema_version.const, "dashboard_contract.v5");
assert.strictEqual(
  schema.properties.input_schema_id.const,
  "myquant.v17.v4.mainline-public-run.v1"
);
assert.deepStrictEqual(
  schema.properties.runtime_state.enum,
  ["ACTIVE", "V17_MAINLINE_UNINITIALIZED", "BLOCKED"]
);

assert.strictEqual(sample.schema_version, "dashboard_contract.v5");
assert.strictEqual(sample.runtime_state, "V17_MAINLINE_UNINITIALIZED");
assert.strictEqual(sample.public_run, null);
assert.strictEqual(sample.active_pointer_ref, null);
assert.ok(sample.blockers.length > 0);

const uninitialized = Contract.deriveSnapshot(null);
assert.strictEqual(uninitialized.runtime_state, "V17_MAINLINE_UNINITIALIZED");
assert.strictEqual(uninitialized.public_run, null);

const validRun = publicRun();
const validation = Contract.validatePublicRun(validRun);
assert.deepStrictEqual(validation, { valid: true, errors: [] });
const active = Contract.deriveSnapshot(validRun);
assert.strictEqual(active.runtime_state, "ACTIVE");
assert.strictEqual(active.strategy_id, "cn-mainline");
assert.strictEqual(active.public_run, validRun);
assert.strictEqual(active.active_pointer_ref, validRun.active_pointer_ref);
assert.deepStrictEqual(active.blockers, []);

const wrongSchema = publicRun();
wrongSchema.schema_id = "myquant.v17.v4.unknown-public-run.v1";
const schemaBlocked = Contract.deriveSnapshot(wrongSchema);
assert.strictEqual(schemaBlocked.runtime_state, "BLOCKED");
assert.strictEqual(schemaBlocked.public_run, null);
assert.match(schemaBlocked.blockers[0], /schema_id/);

const writeEnabled = publicRun();
writeEnabled.authority_flags.order_calls = true;
assert.strictEqual(Contract.deriveSnapshot(writeEnabled).runtime_state, "BLOCKED");

const indirectSelection = publicRun();
indirectSelection.selector_used = true;
assert.strictEqual(Contract.deriveSnapshot(indirectSelection).runtime_state, "BLOCKED");

const indirectSource = publicRun();
indirectSource.source_closure_ref.relative_path = "results/v17_mainline/source.json";
assert.strictEqual(Contract.deriveSnapshot(indirectSource).runtime_state, "BLOCKED");

const noTargets = publicRun();
noTargets.targets = [];
assert.strictEqual(Contract.deriveSnapshot(noTargets).runtime_state, "BLOCKED");

const extraField = publicRun();
extraField.previous_run_ref = extraField.mainline_run_ref;
assert.strictEqual(Contract.deriveSnapshot(extraField).runtime_state, "BLOCKED");

const unsorted = publicRun();
unsorted.targets.reverse();
assert.strictEqual(Contract.deriveSnapshot(unsorted).runtime_state, "BLOCKED");

const loader = fs.readFileSync(path.join(root, "js", "mainline_input.js"), "utf8");
const preloaded = publicRun();
const sandbox = { window: { MyQuantV17MainlinePublicRun: preloaded } };
vm.runInNewContext(loader, sandbox);
assert.strictEqual(
  sandbox.window.MyQuantV17MainlinePublicRun,
  preloaded,
  "sanitized loader must preserve an exact private public run loaded before it"
);
const invalidSandbox = { window: { MyQuantV17MainlinePublicRun: false } };
vm.runInNewContext(loader, invalidSandbox);
assert.strictEqual(
  invalidSandbox.window.MyQuantV17MainlinePublicRun,
  false,
  "sanitized loader must not mask a malformed private input as uninitialized"
);

console.log("dashboard_contract_v5.test.js: PASS");
