"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const Contract = require(path.join(root, "js", "cn_aggressive_dashboard_contract_v1.js"));
const Analysis = require(path.join(root, "js", "cn_aggressive_dashboard_analysis_v1.js"));
const schema = JSON.parse(
  fs.readFileSync(path.join(root, "schema", "cn_aggressive_dashboard.v1.schema.json"), "utf8")
);
const sample = JSON.parse(
  fs.readFileSync(path.join(root, "sample", "cn_aggressive_dashboard.v1.json"), "utf8")
);
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const publicHtml = fs.readFileSync(path.join(root, "public.html"), "utf8");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");

assert.strictEqual(schema.properties.schema_version.const, "cn_aggressive_dashboard.v1");
assert.deepStrictEqual(schema.properties.status.enum, ["FRESH", "PARTIAL", "BLOCKED"]);
assert.strictEqual(schema.properties.strategy_label.const, "aggressive_tech_manufacturing");
assert.match(sample.positions[0].name, /^合成样例/);
assert.ok(sample.warnings.includes("synthetic_sample_only"));
assert.match(html, /id="growthChart"/);
assert.match(html, /id="drawdownChart"/);
assert.match(html, /id="monthlyPerformanceRows"/);
assert.match(html, /id="historyInsightList"/);
assert.match(html, /id="quantMetricGrid"/);
assert.match(publicHtml, /id="quantMetricGrid"/);
assert.match(app, /预计年化收益/);
assert.match(publicHtml, /Sharpe 使用中国1年期国债收益率/);
assert.match(html, /id="historySummary"/);
assert.match(html, /id="performanceRows"/);
assert.match(html, /js\/cn_aggressive_dashboard_analysis_v1\.js/);
assert.match(html, /js\/cn_aggressive_public_mode\.js/);
assert.match(html, /id="evidenceDetails"/);
assert.match(html, /AlphaMx科技创新组合001号/);
assert.doesNotMatch(html, /Alpha-mw's科技创新组合001号/);
assert.match(html, /class="brand-mark"/);
assert.match(html, /id="performanceInsight">净值与回撤/);
assert.match(html, /public-section-brand/);
assert.match(html, /public-section-performance/);
assert.match(html, /public-section-monthly/);
assert.match(html, /public-section-readout/);
assert.match(html, /class="snapshot-strip internal-only"/);
assert.match(html, /class="panel holdings-panel internal-only"/);
assert.match(html, /role="group" aria-label="组合、沪深300、科创50、创业板指与累计超额的互动历史收益折线图"/);
assert.match(html, /移动鼠标或使用左右方向键查看具体日期与净值/);
assert.doesNotMatch(html, /7 月 9 日外部入金/);
assert.doesNotMatch(publicHtml, /7 月 9 日外部入金/);
assert.doesNotMatch(publicHtml, /剔除入金后净值/);
assert.doesNotMatch(app, /从此前剔除入金后净值高点计算/);
assert.doesNotMatch(app, /chart-funding-(?:line|label)/);
assert.strictEqual((publicHtml.match(/public-section-brand/g) || []).length, 1);
assert.strictEqual((publicHtml.match(/public-section-performance/g) || []).length, 1);
assert.strictEqual((publicHtml.match(/public-section-monthly/g) || []).length, 1);
assert.strictEqual((publicHtml.match(/public-section-readout/g) || []).length, 1);
assert.match(publicHtml, /AlphaMx科技创新组合001号/);
assert.doesNotMatch(publicHtml, /Alpha-mw's科技创新组合001号/);
assert.match(publicHtml, /组合从2026-03-17开始，以100 万为起始资金，移动鼠标或使用左右方向键查看具体日期与净值。/);
assert.doesNotMatch(publicHtml, /当前持仓与资产配置/);
assert.doesNotMatch(publicHtml, /最近一次持仓变化/);
assert.doesNotMatch(publicHtml, /收益与会计口径/);
assert.doesNotMatch(publicHtml, /证据路径与 SHA/);
assert.doesNotMatch(publicHtml, /月末剔除入金资产/);
assert.strictEqual(sample.history.archive_start_record, "20990102_1200");
assert.strictEqual(sample.history.funding_events.length, 1);
assert.strictEqual(sample.portfolio.performance_points[0].date, sample.history.archive_start_date);
assert.strictEqual(sample.portfolio.cash + sample.portfolio.market_value, sample.portfolio.total_value);
assert.strictEqual(sample.portfolio.total_value, sample.portfolio.adjusted_total_value);
assert.strictEqual(sample.portfolio.portfolio_pnl, sample.portfolio.total_value - sample.portfolio.performance_initial_capital);
assert.ok(Math.abs(sample.portfolio.cash_weight + sample.portfolio.gross_exposure - 1) < 1e-12);
assert.ok(Math.abs(sample.positions.reduce((total, position) => total + position.nav_weight, 0) - sample.portfolio.gross_exposure) < 1e-12);

const analysis = Analysis.buildAnalysis(sample);
assert.strictEqual(analysis.points.length, 2);
assert.strictEqual(analysis.monthly.length, 1);
assert.strictEqual(analysis.monthly[0].period, "2099-01");
assert.strictEqual(analysis.monthly[0].base_date, "2099-01-02");
assert.ok(Math.abs(analysis.monthly[0].portfolio_return - (10000 / 9900 - 1)) < 1e-12);
assert.ok(Math.abs(analysis.monthly[0].benchmark_return - 0.005) < 1e-12);
assert.ok(Math.abs(analysis.monthly[0].star50_return - 0.01) < 1e-12);
assert.ok(Math.abs(analysis.monthly[0].chinext_return + 0.005) < 1e-12);
assert.ok(Math.abs(analysis.monthly[0].excess_return - (10000 / 9900 - 1 - 0.005)) < 1e-12);
assert.strictEqual(analysis.deepest_portfolio_drawdown.value, 0);
assert.strictEqual(analysis.quantitative_metrics, null);

const metricFixture = [
  { date: "2025-01-01", portfolio_unit_nav: 1, csi300_nav: 1, risk_free_annual_yield: 0.02 },
  { date: "2025-04-01", portfolio_unit_nav: 1.1, csi300_nav: 1.02, risk_free_annual_yield: 0.021 },
  { date: "2025-07-01", portfolio_unit_nav: 1.045, csi300_nav: 1.01, risk_free_annual_yield: 0.019 },
  { date: "2026-01-01", portfolio_unit_nav: 1.1495, csi300_nav: 1.04, risk_free_annual_yield: 0.018 }
];
const riskMetrics = Analysis.quantitativeMetrics(metricFixture, -0.08);
assert.ok(riskMetrics.risk_free_rate > 0.019 && riskMetrics.risk_free_rate < 0.021);
assert.strictEqual(riskMetrics.risk_free_tenor, "1Y");
assert.strictEqual(riskMetrics.interval_count, 3);
assert.ok(riskMetrics.estimated_annualized_return > 0.14 && riskMetrics.estimated_annualized_return < 0.16);
assert.ok(riskMetrics.annualized_volatility > 0);
assert.ok(Number.isFinite(riskMetrics.sharpe_ratio));
assert.ok(Number.isFinite(riskMetrics.sortino_ratio));
assert.ok(Number.isFinite(riskMetrics.calmar_ratio));
assert.ok(Number.isFinite(riskMetrics.beta_csi300));
assert.ok(Number.isFinite(riskMetrics.correlation_csi300));
assert.ok(Number.isFinite(riskMetrics.tracking_error));
assert.ok(Number.isFinite(riskMetrics.information_ratio));
assert.ok(Math.abs(riskMetrics.positive_interval_ratio - 2 / 3) < 1e-12);

const registryBoundSample = structuredClone(sample);
registryBoundSample.portfolio.performance_points.forEach((point) => {
  point.evidence_status = "DASHBOARD_POST_HOC_SHA_REGISTRY_BOUND";
});
assert.strictEqual(
  Analysis.buildAnalysis(registryBoundSample).monthly[0].evidence_status,
  "HASH_BOUND"
);

const drawdownSample = structuredClone(sample);
drawdownSample.portfolio.performance_points.push({
  ...drawdownSample.portfolio.performance_points[1],
  date: "2099-01-04",
  record: "20990104_1200",
  total_value: 14000,
  excluded_external_flow: 5000,
  adjusted_total_value: 9000,
  portfolio_unit_nav: 0.9090909090909091,
  portfolio_cumulative_return: -0.09090909090909094,
  csi300_nav: 0.995,
  csi300_cumulative_return: -0.005,
  cumulative_excess_return: -0.08590909090909094
});
const drawdownAnalysis = Analysis.buildAnalysis(drawdownSample);
assert.ok(Math.abs(drawdownAnalysis.deepest_portfolio_drawdown.value + 0.1) < 1e-12);

assert.deepStrictEqual(Contract.validateBundle(sample), { valid: true, errors: [] });
const usable = Contract.deriveSnapshot(sample);
assert.strictEqual(usable.status, "PARTIAL");
assert.strictEqual(usable.bundle, sample);

const incompleteCurrentValuation = structuredClone(sample);
incompleteCurrentValuation.latest_valid_record = "20990104_1200";
incompleteCurrentValuation.latest_data_date = "2099-01-04";
incompleteCurrentValuation.current_evidence.official_valuation = false;
incompleteCurrentValuation.current_evidence.valuation_completeness_passed = false;
incompleteCurrentValuation.current_evidence.valuation_status = "BLOCKED_PENDING_STRICT_CLOSE";
incompleteCurrentValuation.current_evidence.price_basis = "synthetic_transaction_mark";
incompleteCurrentValuation.portfolio.current_valuation_status = "BLOCKED_PENDING_STRICT_CLOSE";
incompleteCurrentValuation.portfolio.cash += 100;
incompleteCurrentValuation.portfolio.total_value += 100;
incompleteCurrentValuation.portfolio.adjusted_total_value += 100;
incompleteCurrentValuation.portfolio.portfolio_pnl += 100;
incompleteCurrentValuation.portfolio.cumulative_profit_excluding_external_flow += 100;
incompleteCurrentValuation.portfolio.cash_weight = (
  incompleteCurrentValuation.portfolio.cash / incompleteCurrentValuation.portfolio.total_value
);
incompleteCurrentValuation.portfolio.gross_exposure = (
  incompleteCurrentValuation.portfolio.market_value / incompleteCurrentValuation.portfolio.total_value
);
incompleteCurrentValuation.positions.forEach((position) => {
  position.nav_weight = position.market_value / incompleteCurrentValuation.portfolio.total_value;
});
incompleteCurrentValuation.warnings.push(
  "latest_current_valuation_incomplete:BLOCKED_PENDING_STRICT_CLOSE"
);
assert.deepStrictEqual(
  Contract.validateBundle(incompleteCurrentValuation),
  { valid: true, errors: [] }
);
assert.strictEqual(Contract.deriveSnapshot(incompleteCurrentValuation).status, "PARTIAL");

const unjustifiedCurrentMismatch = structuredClone(incompleteCurrentValuation);
delete unjustifiedCurrentMismatch.current_evidence.official_valuation;
assert.strictEqual(Contract.deriveSnapshot(unjustifiedCurrentMismatch).status, "BLOCKED");

const freshIncompleteCurrentValuation = structuredClone(incompleteCurrentValuation);
freshIncompleteCurrentValuation.status = "FRESH";
assert.strictEqual(Contract.deriveSnapshot(freshIncompleteCurrentValuation).status, "BLOCKED");

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

const wrongEconomicCash = structuredClone(sample);
wrongEconomicCash.portfolio.cash += 5000;
assert.strictEqual(Contract.deriveSnapshot(wrongEconomicCash).status, "BLOCKED");

const wrongPositionWeight = structuredClone(sample);
wrongPositionWeight.positions[0].nav_weight = 0.01;
assert.strictEqual(Contract.deriveSnapshot(wrongPositionWeight).status, "BLOCKED");

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

const publicModeLoader = fs.readFileSync(path.join(root, "js", "cn_aggressive_public_mode.js"), "utf8");
const publicModeSandbox = { window: {} };
vm.runInNewContext(publicModeLoader, publicModeSandbox);
assert.strictEqual(publicModeSandbox.window.CNPublicDashboard, false);
const presetPublicModeSandbox = { window: { CNPublicDashboard: true } };
vm.runInNewContext(publicModeLoader, presetPublicModeSandbox);
assert.strictEqual(presetPublicModeSandbox.window.CNPublicDashboard, true);

console.log("cn_aggressive_dashboard_contract_v1.test.js: PASS");
