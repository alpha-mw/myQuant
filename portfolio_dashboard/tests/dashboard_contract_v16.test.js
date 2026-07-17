"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const schema = JSON.parse(
  fs.readFileSync(
    path.join(root, "schema", "dashboard_contract.v16.schema.json"),
    "utf8"
  )
);

const branches = ["quant", "fundamental", "macro", "llm"];
const forbiddenRetrievalFields = new Set([
  "score",
  "confidence",
  "likelihood",
  "weight",
]);

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
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

function validateV16Snapshot(payload) {
  assert.strictEqual(payload.schema_version, "dashboard_contract.v16");
  assert.strictEqual(payload.architecture_version, "16.0.0");
  assert.strictEqual(payload.branch_schema_version, "v16.four-branch");
  assert.strictEqual(payload.results_namespace, "results/v16");
  assert.strictEqual(
    payload.v16_run_readiness.schema_version,
    "v16_run_readiness.v1"
  );
  assert.match(
    payload.v16_run_readiness.path,
    /^results\/v16(?:\/[A-Za-z0-9._-]+)*\/v16_run_readiness\.json$/
  );
  assert.strictEqual(
    payload.candidate_report.schema_version,
    "candidate_decision_report.v16"
  );
  assert.match(
    payload.candidate_report.path,
    /^results\/v16(?:\/[A-Za-z0-9._-]+)*\/v16_candidate_decision_report\.json$/
  );

  const decision = payload.candidate_decision;
  assert.deepStrictEqual(Object.keys(decision.branch_contributions), branches);
  branches.forEach((branch) => {
    assert.strictEqual(decision.branch_contributions[branch].weight, 0.25);
  });
  collectKeys(decision.retrieval_evidence).forEach((key) => {
    assert.strictEqual(
      forbiddenRetrievalFields.has(key),
      false,
      `retrieval evidence contains forbidden field ${key}`
    );
  });
  decision.retrieval_evidence.items.forEach((item) => {
    assert.ok(["quant", "fundamental", "macro"].includes(item.branch));
  });
  assert.ok(
    decision.posterior.posterior_edge_after_costs === null ||
      Number.isFinite(decision.posterior.posterior_edge_after_costs)
  );
  assert.strictEqual(decision.risk_advisor.advisory_only, true);
  assert.strictEqual(
    Object.prototype.hasOwnProperty.call(decision.risk_advisor, "blockers"),
    false
  );
  assert.ok(decision.ic.selected_symbols.length <= 12);
  assert.strictEqual(decision.ic.actions.length, decision.ic.menu_symbols.length);
  assert.deepStrictEqual(
    new Set(decision.ic.actions.map((item) => item.symbol)),
    new Set(decision.ic.menu_symbols)
  );
  decision.ic.actions.forEach((item) => {
    assert.ok(["BUY", "HOLD", "AVOID", "SELL"].includes(item.action));
    if (item.action === "HOLD") {
      assert.ok(Math.abs(item.target_weight - item.existing_weight) <= 1e-6);
    }
  });
  const total =
    decision.ic.cash_ratio +
    decision.ic.actions.reduce((sum, item) => sum + item.target_weight, 0);
  assert.ok(Math.abs(total - 1) <= 1e-6);
  assert.strictEqual(decision.execution.broker_side_effects, false);
  assert.strictEqual(
    decision.execution.new_risk_authorized,
    decision.readiness.new_risk_authorized
  );
}

assert.strictEqual(schema.properties.schema_version.const, "dashboard_contract.v16");
assert.strictEqual(schema.properties.architecture_version.const, "16.0.0");
assert.strictEqual(schema.properties.results_namespace.const, "results/v16");
assert.deepStrictEqual(
  schema.$defs.fourBranchContributions.required,
  branches
);
branches.forEach((branch) => {
  assert.ok(schema.$defs.fourBranchContributions.properties[branch]);
});
assert.strictEqual(schema.$defs.branchContribution.properties.weight.const, 0.25);
assert.strictEqual(schema.$defs.icDecision.properties.selected_symbols.maxItems, 12);
assert.deepStrictEqual(schema.$defs.icAction.properties.action.enum, [
  "BUY",
  "HOLD",
  "AVOID",
  "SELL",
]);
assert.strictEqual(schema.$defs.riskAdvisor.properties.advisory_only.const, true);
assert.strictEqual(
  Object.prototype.hasOwnProperty.call(schema.$defs.riskAdvisor.properties, "blockers"),
  false
);
collectKeys({
  retrievalEvidence: schema.$defs.retrievalEvidence,
  retrievalEvidenceItem: schema.$defs.retrievalEvidenceItem,
}).forEach((key) => {
  assert.strictEqual(
    forbiddenRetrievalFields.has(key),
    false,
    `retrieval schema contains forbidden field ${key}`
  );
});

const hash = "a".repeat(64);
const snapshot = {
  schema_version: "dashboard_contract.v16",
  schema_sha256: hash,
  architecture_version: "16.0.0",
  branch_schema_version: "v16.four-branch",
  results_namespace: "results/v16",
  run_id: "synthetic-v16-run",
  generated_at: "2026-07-17T10:00:00+08:00",
  status: "sample",
  blockers: [],
  v16_run_readiness: {
    schema_version: "v16_run_readiness.v1",
    path: "results/v16/synthetic-run/v16_run_readiness.json",
    sha256: hash,
    new_risk_authorized: false,
    blockers: ["activation_dashboard_gate_not_ready"],
    activation_candidate: false,
    activation_blockers: ["activation_dashboard_gate_not_ready"],
  },
  candidate_report: {
    schema_version: "candidate_decision_report.v16",
    path: "results/v16/synthetic-run/v16_candidate_decision_report.json",
    sha256: hash,
  },
  candidate_decision: {
    branch_contributions: Object.fromEntries(
      branches.map((branch) => [
        branch,
        {
          status: "ready",
          score: 0.2,
          weight: 0.25,
          contribution: 0.05,
          evidence_sha256: hash,
        },
      ])
    ),
    retrieval_evidence: {
      status: "verified",
      items: [
        {
          symbol: "SYNTH-0001",
          branch: "quant",
          supporting_fact_ids: ["synthetic-fact-1"],
          contradicting_fact_ids: [],
          conflict_note: null,
        },
      ],
      warnings: [],
    },
    posterior: {
      posterior_win_rate: 0.6,
      posterior_expected_alpha: 0.03,
      posterior_edge_after_costs: null,
      win_rate_interval_90: { lower: 0.5, upper: 0.7 },
      expected_alpha_interval_90: { lower: 0.01, upper: 0.05 },
    },
    risk_advisor: {
      advisory_only: true,
      warnings: ["synthetic warning"],
      recommendations: ["review manually"],
    },
    ic: {
      menu_symbols: ["SYNTH-0001"],
      actions: [
        {
          symbol: "SYNTH-0001",
          action: "AVOID",
          selected_for_portfolio: false,
          existing_weight: 0,
          target_weight: 0,
          rationale: "synthetic insufficient edge",
          risk_acceptance_rationale: null,
        },
      ],
      selected_symbols: [],
      cash_ratio: 1,
    },
    handoff: {
      status: "blocked",
      artifact_path: null,
      artifact_sha256: null,
      blockers: ["synthetic handoff blocker"],
    },
    eligibility: {
      eligible: false,
      blockers: ["synthetic eligibility blocker"],
    },
    execution: {
      status: "no_new_risk",
      new_risk_authorized: false,
      broker_side_effects: false,
      blockers: ["activation_dashboard_gate_not_ready"],
    },
    readiness: null,
  },
  as_of_matrix: {},
  trading_calendar: {},
  sources: {},
  nav: [],
  positions: [],
  trades: [],
  industries: [],
  factors: [],
  reconciliation: {},
  metric_policy: {},
};
snapshot.candidate_decision.readiness = snapshot.v16_run_readiness;

validateV16Snapshot(snapshot);

const v15 = deepClone(snapshot);
v15.schema_version = "dashboard_contract.v15";
assert.throws(() => validateV16Snapshot(v15));

const tooMany = deepClone(snapshot);
tooMany.candidate_decision.ic.menu_symbols = Array.from(
  { length: 13 },
  (_, index) => `SYNTH-${String(index).padStart(4, "0")}`
);
tooMany.candidate_decision.ic.actions = tooMany.candidate_decision.ic.menu_symbols.map(
  (symbol) => ({
    symbol,
    action: "BUY",
    selected_for_portfolio: true,
    existing_weight: 0,
    target_weight: 1 / 13,
    rationale: "synthetic selection",
    risk_acceptance_rationale: null,
  })
);
tooMany.candidate_decision.ic.selected_symbols = [
  ...tooMany.candidate_decision.ic.menu_symbols,
];
tooMany.candidate_decision.ic.cash_ratio = 0;
assert.throws(() => validateV16Snapshot(tooMany));

const weightedRetrieval = deepClone(snapshot);
weightedRetrieval.candidate_decision.retrieval_evidence.items[0].weight = 0.25;
assert.throws(() => validateV16Snapshot(weightedRetrieval));

const serialized = JSON.stringify(snapshot);
assert.strictEqual(serialized.includes("account_id"), false);
assert.strictEqual(serialized.includes("broker_account"), false);

console.log("dashboard_contract_v16.test.js: PASS");
