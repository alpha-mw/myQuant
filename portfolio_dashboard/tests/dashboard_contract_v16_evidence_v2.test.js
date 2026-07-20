"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const schema = JSON.parse(
  fs.readFileSync(
    path.join(root, "schema", "dashboard_contract.v16.evidence-v2.schema.json"),
    "utf8"
  )
);

const branches = ["quant", "fundamental", "macro", "llm"];
const forbiddenRetrievalFields = new Set([
  "score",
  "confidence",
  "probability",
  "likelihood",
  "weight",
]);
const f64Pattern = new RegExp(schema.$defs.f64.pattern);
const sha = "a".repeat(64);

function evidenceRef(artifactSchema, absolutePath) {
  return {
    schema_version: "v16.evidence-ref.v2",
    artifact_schema: artifactSchema,
    absolute_path: absolutePath,
    byte_sha256: sha,
    semantic_sha256: sha,
    root_policy: "v16.private-evidence-root.v2",
  };
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function decodeF64(value) {
  assert.match(value, f64Pattern);
  const token = value.slice(4);
  if (token === "0x0.0p+0") return 0;
  const match = token.match(/^(-?)0x([01])\.([0-9a-f]{13})p([+-][0-9]+)$/);
  assert.ok(match, `invalid f64 token ${value}`);
  const sign = match[1] === "-" ? -1 : 1;
  let mantissa = Number.parseInt(match[2], 16);
  [...match[3]].forEach((digit, index) => {
    mantissa += Number.parseInt(digit, 16) / 16 ** (index + 1);
  });
  return sign * mantissa * 2 ** Number.parseInt(match[4], 10);
}

function collectKeys(value, target = new Set()) {
  if (Array.isArray(value)) {
    value.forEach((item) => collectKeys(item, target));
  } else if (value && typeof value === "object") {
    Object.entries(value).forEach(([key, child]) => {
      target.add(key);
      collectKeys(child, target);
    });
  }
  return target;
}

function assertNoNativeFloats(value) {
  if (Array.isArray(value)) {
    value.forEach(assertNoNativeFloats);
  } else if (value && typeof value === "object") {
    Object.values(value).forEach(assertNoNativeFloats);
  } else if (typeof value === "number") {
    assert.ok(Number.isInteger(value), `native float leaked into snapshot: ${value}`);
  }
}

function validateSnapshot(payload) {
  assert.strictEqual(payload.schema_version, "dashboard_contract.v16.evidence-v2");
  assert.strictEqual(payload.architecture_version, "16.0.0");
  assert.deepStrictEqual(
    payload.formal_branches.map((item) => item.branch),
    branches
  );
  payload.formal_branches.forEach((item) => assert.strictEqual(item.weight, "0.25"));
  assert.strictEqual(payload.retrieval_role, "evidence_only_no_scoring_or_weight");
  assert.strictEqual(payload.risk_advisor_role, "advisory_only");
  assert.ok(payload.menu.length > 0 && payload.menu.length <= 50);
  payload.menu.forEach((item) => {
    assert.deepStrictEqual(
      item.branch_evidence.map((branch) => branch.branch),
      branches
    );
    item.branch_evidence.forEach((branch) => {
      decodeF64(branch.raw_score);
      decodeF64(branch.confidence);
      decodeF64(branch.calibrated_probability);
    });
    collectKeys(item.retrieval_advisory).forEach((key) => {
      assert.strictEqual(
        forbiddenRetrievalFields.has(key),
        false,
        `retrieval advisory contains forbidden field ${key}`
      );
    });
    assert.ok(["BUY", "HOLD", "AVOID", "SELL"].includes(item.allocation.action));
    decodeF64(item.allocation.existing_weight);
    decodeF64(item.allocation.target_weight);
  });
  const total =
    decodeF64(payload.cash_ratio) +
    payload.menu.reduce(
      (sum, item) => sum + decodeF64(item.allocation.target_weight),
      0
    );
  assert.ok(Math.abs(total - 1) <= 1e-12);
  assert.ok(Math.abs(decodeF64(payload.target_plus_cash) - 1) <= 1e-12);
  assert.ok(payload.positive_weight_count <= 12);
  assert.strictEqual(payload.projection_validation_complete, true);
  assert.strictEqual(payload.authority_source_complete, false);
  Object.entries(payload.readiness).forEach(([key, value]) => {
    if (key.endsWith("authorized") || key.endsWith("enabled") || key.endsWith("verified")) {
      assert.strictEqual(value, false, `${key} must remain false`);
    }
  });
  assert.strictEqual(payload.readiness.activation_candidate, false);
  assert.strictEqual(payload.readiness.broker_side_effects, false);
  assert.strictEqual(payload.readiness.status, "no_new_risk");
  assert.ok(payload.blockers.length > 0);
  assertNoNativeFloats(payload);
}

const branchEvidence = branches.map((branch) => ({
  branch,
  raw_score: "f64:0x1.999999999999ap-4",
  confidence: "f64:0x1.999999999999ap-1",
  calibrated_probability: "f64:0x1.3333333333333p-1",
  evidence_ids: [`AAA-${branch}`],
  source_ref: evidenceRef("v16.formal-branch-prediction.v2", `/private/source-${branch}.json`),
  model_bundle_ref: evidenceRef("v16.frozen-model-bundle.v2", `/private/model-${branch}.json`),
}));

const snapshot = {
  schema_version: "dashboard_contract.v16.evidence-v2",
  architecture_version: "16.0.0",
  protocol_attempt_id: "attempt-v16-publication-001",
  run_id: "run-v16-publication-001",
  generated_at: "2026-07-20T00:30:00Z",
  analysis_trade_date: "2026-07-17",
  source_refs: {
    publication_plan: evidenceRef("v16.publication-source-plan.v2", "/private/plan.json"),
    readiness_v4: evidenceRef("v16_run_readiness.v4", "/private/readiness.json"),
    candidate_report: evidenceRef("v16.candidate-source-report.v2", "/private/report.json"),
  },
  formal_branches: branches.map((branch) => ({ branch, weight: "0.25" })),
  retrieval_role: "evidence_only_no_scoring_or_weight",
  risk_advisor_role: "advisory_only",
  menu: [
    {
      symbol: "AAA",
      posterior: {
        win_rate: "f64:0x1.3333333333333p-1",
        expected_alpha: "f64:0x1.eb851eb851eb8p-6",
        edge_after_costs: "f64:0x1.47ae147ae147bp-6",
        win_rate_interval_90: [
          "f64:0x1.0000000000000p-1",
          "f64:0x1.6666666666666p-1",
        ],
        expected_alpha_interval_90: [
          "f64:0x1.47ae147ae147bp-7",
          "f64:0x1.47ae147ae147bp-5",
        ],
        cost_input_ref: evidenceRef("v16.posterior-cost-input.v2", "/private/cost-AAA.json"),
      },
      branch_evidence: branchEvidence,
      retrieval_advisory: [
        {
          branch: "quant",
          supporting_fact_ids: ["AAA-fact"],
          contradicting_fact_ids: [],
          conflict_note: null,
        },
      ],
      allocation: {
        action: "BUY",
        selected_for_portfolio: true,
        existing_weight: "f64:0x0.0p+0",
        target_weight: "f64:0x1.3333333333333p-1",
        rationale_sha256: sha,
        severe_risk_count: 0,
        risk_acceptance_rationale_sha256: null,
      },
    },
  ],
  cash_ratio: "f64:0x1.999999999999ap-2",
  positive_weight_count: 1,
  target_plus_cash: "f64:0x1.0000000000000p+0",
  readiness: {
    status: "no_new_risk",
    activation_candidate: false,
    new_risk_authorized: false,
    production_apply_enabled: false,
    production_pointer_switch_authorized: false,
    codex_activation_authorized: false,
    dashboard_activation_authorized: false,
    sealed_live_human_receipt_verified: false,
    broker_side_effects: false,
    source_readiness_status: "no_new_risk",
  },
  projection_validation_complete: true,
  authority_source_complete: false,
  blockers: ["dashboard_activation_receipt_v2_not_integrated"],
  blocker_sources: [
    {
      blocker: "dashboard_activation_receipt_v2_not_integrated",
      source: "readiness_v4:foundation",
    },
  ],
  semantic_sha256: sha,
};

assert.strictEqual(
  schema.properties.schema_version.const,
  "dashboard_contract.v16.evidence-v2"
);
assert.strictEqual(schema.properties.authority_source_complete.const, false);
assert.strictEqual(schema.$defs.readiness.properties.new_risk_authorized.const, false);
assert.strictEqual(schema.$defs.readiness.properties.dashboard_activation_authorized.const, false);
assert.strictEqual(schema.properties.positive_weight_count.maximum, 12);
assert.deepStrictEqual(schema.$defs.allocation.properties.action.enum, [
  "BUY",
  "HOLD",
  "AVOID",
  "SELL",
]);
assert.strictEqual(f64Pattern.test("f64:-0x0.0p+0"), false);
assert.strictEqual(f64Pattern.test("f64:nan"), false);
assert.strictEqual(f64Pattern.test("f64:0x1.0000000000000p+0"), true);

validateSnapshot(snapshot);

const authorized = deepClone(snapshot);
authorized.readiness.new_risk_authorized = true;
assert.throws(() => validateSnapshot(authorized));

const retrievalScoring = deepClone(snapshot);
retrievalScoring.menu[0].retrieval_advisory[0].weight = "f64:0x1.0p-2";
assert.throws(() => validateSnapshot(retrievalScoring));

console.log("dashboard_contract_v16_evidence_v2.test.js: PASS");
