"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const Contract = require(path.join(root, "js", "cn_aggressive_dashboard_contract_v2.js"));
const V1Contract = require(path.join(root, "js", "cn_aggressive_dashboard_contract_v1.js"));
const indexHtml = fs.readFileSync(path.join(root, "index.html"), "utf8");
const publicHtml = fs.readFileSync(path.join(root, "public.html"), "utf8");
const appSource = fs.readFileSync(path.join(root, "app.js"), "utf8");
const sample = JSON.parse(
  fs.readFileSync(path.join(root, "sample", "cn_aggressive_dashboard.v1.json"), "utf8")
);

assert.match(indexHtml, /cn_aggressive_dashboard\.v2\.js\?cache=/);
assert.match(indexHtml, /cn_aggressive_dashboard_selector\.v2\.js\?cache=/);
assert.match(indexHtml, /cn_aggressive_dashboard_contract_v2\.js\?v=/);
assert.doesNotMatch(
  publicHtml,
  /cn_aggressive_dashboard\.v2|dashboard_selector\.v2|contract_v2/
);
assert.match(indexHtml, /__cnAggressivePrivateDashboardBundle/);
assert.match(indexHtml, /delete window\.MyQuantCNAggressiveDashboardV2/);
assert.match(appSource, /PrivateDashboardContract\.deriveSnapshot/);
assert.match(appSource, /function scheduleFreshnessRecheck/);
assert.match(appSource, /PrivateDashboardContract\.nextFreshnessRecheckDelay/);
assert.match(appSource, /v2Snapshot\.status\.freshness === "UPDATED"/);
assert.match(appSource, /freshnessTimer = window\.setTimeout/);
assert.doesNotMatch(appSource, /MyQuantCNAggressiveDashboardV2|MyQuantCNAggressiveDashboardSelectorV2|CNAggressiveDashboardContractV2/);
assert.match(appSource, /v2Snapshot\.holdings_label/);
assert.match(appSource, /v2Snapshot\.absolute_performance_label/);

const SHA = {
  bundle: "a".repeat(64),
  canonical: "b".repeat(64),
  financial: "c".repeat(64),
  ledger: "d".repeat(64),
  receipt: "e".repeat(64),
};

function makeCanonicalV1() {
  const value = structuredClone(sample);
  value.status = "PARTIAL";
  value.content_sha256 = "f".repeat(64);
  value.latest_valid_record = "20990103_1200";
  value.previous_valid_record = "20990102_1200";
  value.latest_data_date = "2099-01-03";
  value.positions[0].shares = 100;
  value.positions[0].avg_cost = 10;
  value.positions[0].cost_basis = 1000;
  value.positions[0].recorded_price = 11;
  value.positions[0].market_value = 1100;
  value.positions[0].unrealized_pnl = 100;
  value.positions[0].nav_weight = 0.0011;
  value.positions[0].equity_weight = 1;
  value.positions[0].price_date = "2099-01-03";
  value.current_evidence.financial_state_sha256 = SHA.financial;
  value.current_evidence.ledger_sha256 = SHA.ledger;
  value.portfolio.cash = 998900;
  value.portfolio.market_value = 1100;
  value.portfolio.total_value = 1000000;
  value.portfolio.cash_weight = 0.9989;
  value.portfolio.gross_exposure = 0.0011;
  value.portfolio.portfolio_pnl = 0;
  value.portfolio.current_unrealized_pnl = 100;
  value.portfolio.performance_initial_capital = 1000000;
  value.portfolio.excluded_external_flow = 0;
  value.portfolio.adjusted_total_value = 1000000;
  value.portfolio.cumulative_profit_excluding_external_flow = 0;
  value.portfolio.cumulative_return = 0;
  value.portfolio.performance_start_date = "2099-01-02";
  value.portfolio.performance_end_date = "2099-01-03";
  value.portfolio.performance_points = [
    {
      ...value.portfolio.performance_points[0],
      date: "2099-01-02",
      record: "20990102_1200",
      total_value: 1000000,
      excluded_external_flow: 0,
      adjusted_total_value: 1000000,
      portfolio_unit_nav: 1,
      portfolio_cumulative_return: 0,
    },
    {
      ...value.portfolio.performance_points[1],
      date: "2099-01-03",
      record: "20990103_1200",
      total_value: 1000000,
      excluded_external_flow: 0,
      adjusted_total_value: 1000000,
      portfolio_unit_nav: 1,
      portfolio_cumulative_return: 0,
    },
  ];
  value.history.archive_start_record = "20990102_1200";
  value.history.archive_start_date = "2099-01-02";
  value.history.latest_performance_date = "2099-01-03";
  value.history.funding_events = [];
  value.history.net_external_flow = 0;
  return value;
}

function makeV2(canonicalV1 = makeCanonicalV1()) {
  const price = 12;
  const marketValue = 1200;
  const cash = canonicalV1.portfolio.cash;
  const nav = cash + marketValue;
  return {
    schema_version: "cn_aggressive_dashboard.v2",
    publication_attempt_id: "dashboard-v2-20990104-154500",
    generated_at: "2099-01-04T15:45:00+08:00",
    generation_local_date: "2099-01-04",
    canonical_v1: canonicalV1,
    canonical_v1_ref: {
      path: "portfolio_dashboard/private/generated/cn_aggressive_dashboard.v1.json",
      sha256: SHA.canonical,
    },
    integrity: { status: "VERIFIED" },
    continuity_authority: {
      status: "NO_ACTION_BOUND",
      anchor_record_id: "20990103_1200",
      anchor_data_date: "2099-01-03",
      anchor_financial_state_sha256: SHA.financial,
      active_ledger_sha256: SHA.ledger,
      holdings_valid_through: "2099-01-04",
      financial_state_changed: false,
      receipt_id: "automation-20990104-daily-review-v1",
      receipt_content_sha256: SHA.receipt,
    },
    freshness: {
      status: "UPDATED",
      scope: "DAILY_SYNC_LATEST_VERIFIED_LOCAL_CLOSE",
      mark_as_of: "2099-01-04",
      generated_at: "2099-01-04T15:45:00+08:00",
      valid_through: "2099-01-04T23:59:59+08:00",
      source_kind: "STRICT_CN_EOD_CLOSE",
      reason: "CURRENT_DAILY_RECEIPT_AND_LATEST_LOCAL_CLOSE",
    },
    completeness: {
      current_holdings: "COMPLETE",
      current_absolute_performance: "COMPLETE",
      canonical_history: "COMPLETE",
      benchmark_relative: "AS_OF_PRIOR_DATE",
      benchmark_as_of: "2099-01-03",
      legacy_caveats: ["fee_basis_unknown"],
    },
    research_mark: {
      status: "AVAILABLE",
      authority: "VIEW_ONLY_NO_STORE_OR_PERFORMANCE_AUTHORITY",
      source_kind: "STRICT_CN_EOD_CLOSE",
      mark_date: "2099-01-04",
      anchor_record_id: "20990103_1200",
      base_ledger_sha256: SHA.ledger,
      base_financial_state_sha256: SHA.financial,
      positions: [{
        symbol: canonicalV1.positions[0].symbol,
        name: canonicalV1.positions[0].name,
        shares: 100,
        avg_cost: 10,
        cost_basis: 1000,
        price,
        price_date: "2099-01-04",
        price_evidence_status: "EXACT_CLOSE",
        market_value: marketValue,
        unrealized_pnl: 200,
        nav_weight: marketValue / nav,
        equity_weight: 1,
        source_ref: {
          path: "portfolio_dashboard/private/generated/market/000001.SZ.parquet",
          sha256: "1".repeat(64),
        },
      }],
      portfolio: {
        cash,
        market_value: marketValue,
        nav,
        unrealized_pnl: 200,
        cash_weight: cash / nav,
        gross_exposure: marketValue / nav,
      },
      current_absolute_performance: {
        point_date: "2099-01-04",
        anchor_date: "2099-01-03",
        marked_nav: nav,
        initial_capital: 1000000,
        cumulative_return: nav / 1000000 - 1,
        continuity_interval_return: nav / 1000000 - 1,
        max_drawdown: 0,
        evidence_status: "HASH_BOUND_CONTINUITY_MARK",
        authority: "VIEW_ONLY_NO_STORE_OR_PERFORMANCE_AUTHORITY",
      },
      canonical_effect: "NONE",
      ledger_effect: "NONE",
      performance_effect: "NONE",
      paper_effect: "NONE",
      trade_effect: "NONE",
    },
    source_refs: [
      {
        path: "portfolio_dashboard/private/generated/cn_aggressive_dashboard.v1.json",
        sha256: SHA.canonical,
      },
      {
        path: "portfolio_dashboard/private/generated/market/000001.SZ.parquet",
        sha256: "1".repeat(64),
      },
    ],
    content_sha256: SHA.bundle,
  };
}

function makeSelector(status = "UPDATED") {
  const value = {
    schema_version: "cn_aggressive_dashboard_selector.v2",
    attempt_id: "dashboard-v2-20990104-154500",
    status,
    updated_at: "2099-01-04T15:45:01+08:00",
    v2_content_sha256: status === "UPDATED" ? SHA.bundle : null,
    reason: status === "UPDATED" ? "publication_complete" : status.toLowerCase(),
    content_sha256: "0".repeat(64),
  };
  value.content_sha256 = Contract.nodeContentSha256(value);
  return value;
}

const canonicalV1 = makeCanonicalV1();
assert.deepStrictEqual(V1Contract.validateBundle(canonicalV1), { valid: true, errors: [] });

const bundle = makeV2(canonicalV1);
assert.deepStrictEqual(Contract.validateBundle(bundle), { valid: true, errors: [] });
const selector = makeSelector();
assert.deepStrictEqual(Contract.validateSelector(selector), { valid: true, errors: [] });
assert.strictEqual(
  Contract.nodeContentSha256({
    schema_version: "cn_aggressive_dashboard_selector.v2",
    attempt_id: "a",
    status: "UPDATED",
    updated_at: "2099-01-04T15:45:01+08:00",
    v2_content_sha256: "a".repeat(64),
    reason: "ok",
    content_sha256: "0".repeat(64),
  }),
  "938a1ce45d0419e21fd75af038fdd7dc5cf23975677b7af13d3ebcc02b6b4fd2"
);

const updated = Contract.deriveSnapshot(
  bundle,
  selector,
  canonicalV1,
  "2099-01-04T23:59:59+08:00"
);
assert.deepStrictEqual(updated.status, {
  integrity: "VERIFIED",
  freshness: "UPDATED",
  current_holdings: "COMPLETE",
  current_absolute_performance: "COMPLETE",
  canonical_history: "COMPLETE",
  benchmark_relative: "AS_OF_PRIOR_DATE",
});
assert.strictEqual(updated.bundle, bundle);
assert.strictEqual(updated.canonical.bundle, canonicalV1);
assert.deepStrictEqual(updated.blockers, []);
assert.strictEqual(updated.holdings_label, "持仓估值 UPDATED · 最新可用严格收盘 2099-01-04");
assert.strictEqual(updated.absolute_performance_label, "组合绝对业绩 VIEW-UPDATED · 截至 2099-01-04");
assert.strictEqual(updated.anchor_label, "财务状态锚点 2099-01-03 · NO_ACTION 连续有效至 2099-01-04");
assert.strictEqual(updated.benchmark_label, "基准相对业绩 · 截至 2099-01-03");
assert.strictEqual(
  Contract.nextFreshnessRecheckDelay(bundle, "2099-01-04T23:59:58+08:00"),
  1001
);
assert.strictEqual(
  Contract.nextFreshnessRecheckDelay(bundle, "2099-01-04T23:59:59+08:00"),
  1
);

const expired = Contract.deriveSnapshot(
  bundle,
  selector,
  canonicalV1,
  "2099-01-05T00:00:00+08:00"
);
assert.deepStrictEqual(expired.status, {
  integrity: "VERIFIED",
  freshness: "STALE",
  current_holdings: "STALE",
  current_absolute_performance: "STALE",
  canonical_history: "COMPLETE",
  benchmark_relative: "AS_OF_PRIOR_DATE",
});
assert.strictEqual(expired.bundle, null);
assert.strictEqual(expired.holdings_label, "持仓估值 STALE · 最新可用严格收盘 2099-01-04");
assert.strictEqual(expired.absolute_performance_label, "组合绝对业绩 STALE · 截至 2099-01-04");
assert.doesNotMatch(expired.absolute_performance_label, /UPDATED/);
assert.match(expired.blockers[0], /expired/);

const refreshing = Contract.deriveSnapshot(bundle, makeSelector("REFRESHING"), canonicalV1, "2099-01-04T16:00:00+08:00");
assert.strictEqual(refreshing.status.integrity, "VERIFIED");
assert.strictEqual(refreshing.status.current_holdings, "STALE");
assert.strictEqual(refreshing.bundle, null);
assert.match(refreshing.blockers[0], /selector_refreshing/);

const blocked = Contract.deriveSnapshot(bundle, makeSelector("BLOCKED"), canonicalV1, "2099-01-04T16:00:00+08:00");
assert.strictEqual(blocked.status.current_holdings, "BLOCKED");
assert.strictEqual(blocked.bundle, null);
assert.match(blocked.blockers[0], /selector_blocked/);

const missing = Contract.deriveSnapshot(null, null, canonicalV1, "2099-01-04T16:00:00+08:00");
assert.strictEqual(missing.status.current_holdings, "UNAVAILABLE");
assert.strictEqual(missing.bundle, null);
assert.strictEqual(missing.canonical.bundle, canonicalV1);

const legacyFresh = structuredClone(canonicalV1);
legacyFresh.status = "FRESH";
const legacyOnly = Contract.deriveSnapshot(null, null, legacyFresh, "2099-01-04T16:00:00+08:00");
assert.strictEqual(legacyOnly.canonical.status, "FRESH");
assert.strictEqual(legacyOnly.status.current_holdings, "UNAVAILABLE");
assert.notStrictEqual(legacyOnly.status.current_holdings, "UPDATED");

const attemptMismatch = makeSelector();
attemptMismatch.attempt_id = "different-attempt";
attemptMismatch.content_sha256 = Contract.nodeContentSha256(attemptMismatch);
assert.match(
  Contract.deriveSnapshot(bundle, attemptMismatch, canonicalV1, "2099-01-04T16:00:00+08:00").blockers[0],
  /attempt_id_mismatch/
);

const v2HashMismatch = makeSelector();
v2HashMismatch.v2_content_sha256 = "9".repeat(64);
v2HashMismatch.content_sha256 = Contract.nodeContentSha256(v2HashMismatch);
assert.match(
  Contract.deriveSnapshot(bundle, v2HashMismatch, canonicalV1, "2099-01-04T16:00:00+08:00").blockers[0],
  /content_sha256_mismatch/
);

const selectorSelfHashDrift = makeSelector();
selectorSelfHashDrift.reason = "tampered_after_seal";
assert.strictEqual(Contract.validateSelector(selectorSelfHashDrift).valid, false);
assert.match(
  Contract.deriveSnapshot(bundle, selectorSelfHashDrift, canonicalV1, "2099-01-04T16:00:00+08:00").blockers[0],
  /selector_invalid/
);

const differentActualV1 = structuredClone(canonicalV1);
differentActualV1.positions[0].name = "另一个仍符合 v1 的名称";
assert.deepStrictEqual(V1Contract.validateBundle(differentActualV1), { valid: true, errors: [] });
assert.match(
  Contract.deriveSnapshot(bundle, selector, differentActualV1, "2099-01-04T16:00:00+08:00").blockers[0],
  /nested_canonical_v1_mismatch/
);

const badAccounting = structuredClone(bundle);
badAccounting.research_mark.portfolio.market_value += 1;
assert.strictEqual(Contract.validateBundle(badAccounting).valid, false);
assert.match(Contract.validateBundle(badAccounting).errors.join("; "), /market value is inconsistent/);

const benchmarkDateDrift = structuredClone(bundle);
benchmarkDateDrift.completeness.benchmark_as_of = "2099-01-04";
benchmarkDateDrift.completeness.benchmark_relative = "COMPLETE";
benchmarkDateDrift.research_mark.current_absolute_performance.anchor_date = "2099-01-04";
assert.strictEqual(Contract.validateBundle(benchmarkDateDrift).valid, false);
assert.match(Contract.validateBundle(benchmarkDateDrift).errors.join("; "), /benchmark_as_of must remain canonical/);

const authorityEscalation = structuredClone(bundle);
authorityEscalation.research_mark.trade_effect = "ORDER_ALLOWED";
assert.strictEqual(Contract.validateBundle(authorityEscalation).valid, false);
assert.match(Contract.validateBundle(authorityEscalation).errors.join("; "), /must be NONE/);

const malformedNestedObject = structuredClone(bundle);
malformedNestedObject.continuity_authority = null;
assert.doesNotThrow(() => Contract.validateBundle(malformedNestedObject));
assert.strictEqual(Contract.validateBundle(malformedNestedObject).valid, false);

const staleBundle = structuredClone(bundle);
staleBundle.continuity_authority.status = "UNCONFIRMED";
staleBundle.continuity_authority.holdings_valid_through = "2099-01-03";
staleBundle.continuity_authority.receipt_id = null;
staleBundle.continuity_authority.receipt_content_sha256 = null;
staleBundle.freshness.status = "STALE";
staleBundle.freshness.reason = "DAILY_CONTINUITY_RECEIPT_MISSING";
staleBundle.completeness.current_holdings = "STALE";
staleBundle.completeness.current_absolute_performance = "STALE";
assert.deepStrictEqual(Contract.validateBundle(staleBundle), { valid: true, errors: [] });
const staleSnapshot = Contract.deriveSnapshot(staleBundle, selector, canonicalV1, "2099-01-04T16:00:00+08:00");
assert.strictEqual(staleSnapshot.status.freshness, "STALE");
assert.strictEqual(staleSnapshot.status.current_holdings, "STALE");
assert.strictEqual(staleSnapshot.bundle, staleBundle);
assert.doesNotMatch(staleSnapshot.holdings_label, /UPDATED/);
assert.doesNotMatch(staleSnapshot.absolute_performance_label, /UPDATED/);

const publishedCanonical = makeCanonicalV1();
publishedCanonical.latest_valid_record = "20990104_1200";
publishedCanonical.latest_data_date = "2099-01-04";
publishedCanonical.portfolio.performance_end_date = "2099-01-04";
publishedCanonical.portfolio.performance_points[1].date = "2099-01-04";
publishedCanonical.portfolio.performance_points[1].record = "20990104_1200";
publishedCanonical.history.latest_performance_date = "2099-01-04";
const financialPublication = makeV2(publishedCanonical);
financialPublication.continuity_authority.status = "FINANCIAL_STATE_PUBLICATION";
financialPublication.continuity_authority.anchor_record_id = "20990104_1200";
financialPublication.continuity_authority.anchor_data_date = "2099-01-04";
financialPublication.continuity_authority.financial_state_changed = true;
financialPublication.continuity_authority.receipt_id = null;
financialPublication.continuity_authority.receipt_content_sha256 = null;
financialPublication.freshness.reason = "CURRENT_FINANCIAL_PUBLICATION_AND_LATEST_LOCAL_CLOSE";
financialPublication.research_mark.anchor_record_id = "20990104_1200";
financialPublication.completeness.benchmark_as_of = "2099-01-04";
financialPublication.completeness.benchmark_relative = "COMPLETE";
financialPublication.research_mark.current_absolute_performance.anchor_date = "2099-01-04";
assert.deepStrictEqual(Contract.validateBundle(financialPublication), { valid: true, errors: [] });
assert.strictEqual(
  Contract.deriveSnapshot(financialPublication, selector, publishedCanonical, "2099-01-04T16:00:00+08:00").anchor_label,
  "财务状态锚点 2099-01-04 · FINANCIAL_STATE_PUBLICATION 连续有效至 2099-01-04"
);

console.log("cn_aggressive_dashboard_contract_v2.test.js: PASS");
