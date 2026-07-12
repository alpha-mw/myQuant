"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const context = {
  console,
  Date,
  Math,
  Number,
  Object,
  Array,
  String,
  window: { DashboardData: { MS_PER_DAY: 24 * 60 * 60 * 1000 } }
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, "js", "metrics.js"), "utf8"), context);
const Metrics = context.window.DashboardMetrics;

{
  class FakeElement {
    constructor(tagName) {
      this.tagName = tagName;
      this.children = [];
      this.className = "";
      this.textContent = "";
      this.visible = false;
      this.classList = {
        toggle: (name, enabled) => {
          if (name === "visible") this.visible = Boolean(enabled);
        }
      };
    }
    appendChild(child) {
      this.children.push(child);
      return child;
    }
    set innerHTML(value) {
      if (value === "") this.children = [];
    }
  }
  const panel = new FakeElement("div");
  const uiContext = {
    console,
    window: { DashboardCharts: {} },
    document: {
      getElementById: id => id === "messagePanel" ? panel : null,
      createElement: tagName => new FakeElement(tagName)
    }
  };
  vm.createContext(uiContext);
  vm.runInContext(fs.readFileSync(path.join(root, "js", "ui.js"), "utf8"), uiContext);
  uiContext.window.DashboardUI.renderMessages(
    ["error-1", "error-2"],
    ["warning-primary", "warning-detail"],
    ["info-primary", "info-detail"]
  );
  assert.equal(panel.visible, true);
  assert.equal(panel.children.length, 4);
  assert.equal(panel.children[0].className, "message error");
  assert.equal(panel.children[1].textContent, "warning-primary");
  assert.equal(panel.children[2].textContent, "info-primary");
  assert.equal(panel.children[3].tagName, "details");
  assert.equal(panel.children[3].children[1].children.length, 4);
}

function point(date, value) {
  return { date, dateObj: new Date(date + "T00:00:00"), portfolio_nav: value, benchmark_main_nav: value };
}

{
  const sparse = [point("2026-01-02", 1), point("2026-03-02", 1.02), point("2026-06-01", 1.03)];
  const result = Metrics.computePerformance(sparse, "benchmark_main_nav");
  assert.equal(result.kpis.annualized_return, null);
  assert.equal(result.kpis.sharpe_ratio, null);
  assert.equal(result.kpis.annualization_status, "formal_trading_calendar_missing");
}

{
  const dense = [];
  const cursor = new Date("2026-01-05T00:00:00");
  let index = 0;
  while (dense.length < 61) {
    if (cursor.getDay() !== 0 && cursor.getDay() !== 6) {
      const date = cursor.toISOString().slice(0, 10);
      dense.push(point(date, Math.pow(1.001, index)));
      index += 1;
    }
    cursor.setDate(cursor.getDate() + 1);
  }
  const calendar = {
    status: "available",
    expected_open_dates: dense.map(row => row.date)
  };
  const result = Metrics.computePerformance(dense, "benchmark_main_nav", null, calendar);
  assert.equal(result.kpis.annualization_status, "eligible");
  assert.ok(Number.isFinite(result.kpis.annualized_return));
}

{
  const expected = [];
  const cursor = new Date("2026-01-05T00:00:00");
  while (expected.length < 65) {
    if (cursor.getDay() !== 0 && cursor.getDay() !== 6) expected.push(cursor.toISOString().slice(0, 10));
    cursor.setDate(cursor.getDate() + 1);
  }
  const calendar = { status: "available", expected_open_dates: expected };
  const oneMissing = expected.map((date, index) => point(date, Math.pow(1.001, index)))
    .filter(row => row.date !== expected[50]);
  const twoMissing = oneMissing.filter(row => row.date !== expected[51]);
  const covered = Metrics.computePerformance(oneMissing, "benchmark_main_nav", null, calendar);
  const blocked = Metrics.computePerformance(twoMissing, "benchmark_main_nav", null, calendar);
  assert.equal(covered.rolling20Vol.at(-1).date, expected.at(-1));
  assert.notEqual(blocked.rolling20Vol.at(-1).date, expected.at(-1));
}

{
  const calendar = {
    status: "available",
    expected_open_dates: ["2026-01-30", "2026-02-02", "2026-02-03"]
  };
  const rows = [
    Object.assign(point("2026-01-30", 1.00), { benchmark_main_nav: 1.00, portfolio_return: 0 }),
    Object.assign(point("2026-01-31", 9.00), { benchmark_main_nav: 0.20, portfolio_return: 8.00 }),
    Object.assign(point("2026-02-01", 0.10), { benchmark_main_nav: 7.00, portfolio_return: -0.9888888889 }),
    Object.assign(point("2026-02-02", 1.10), { benchmark_main_nav: 1.05, portfolio_return: 10.00 }),
    Object.assign(point("2026-02-03", 1.21), { benchmark_main_nav: 1.1025, portfolio_return: 0.10 })
  ];
  const result = Metrics.computePerformance(rows, "benchmark_main_nav", null, calendar);
  assert.deepEqual(Array.from(result.navSeries, row => row.date), calendar.expected_open_dates);
  assert.ok(Math.abs(result.kpis.total_return - 0.21) < 1e-12);
  assert.ok(Math.abs(result.kpis.benchmark_total_return - 0.1025) < 1e-12);
  assert.ok(Math.abs(result.kpis.excess_return - (1.21 / 1.1025 - 1)) < 1e-12);
  assert.equal(result.kpis.max_drawdown, 0);
  assert.deepEqual(Array.from(result.monthly, row => row.month), ["2026-01", "2026-02"]);
  assert.ok(Math.abs(result.monthly[1].value - 0.21) < 1e-12);
  assert.ok(Math.abs(result.enrichedNav[1].portfolio_return_calc - 0.10) < 1e-12);
  assert.ok(Math.abs(result.enrichedNav[1].benchmark_return_calc - 0.05) < 1e-12);
}

{
  const calendar = {
    status: "available",
    expected_open_dates: ["2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04"]
  };
  const nav = [
    Object.assign(point("2026-01-30", 1.00), { benchmark_main_nav: 1.00 }),
    Object.assign(point("2026-01-31", 8.00), { benchmark_main_nav: 0.25 }),
    Object.assign(point("2026-02-02", 1.10), { benchmark_main_nav: 1.05 }),
    Object.assign(point("2026-02-03", 1.21), { benchmark_main_nav: 1.1025 }),
    Object.assign(point("2026-02-04", 9.00), { benchmark_main_nav: 0.10 })
  ];
  const dashboard = Metrics.computeDashboard({
    nav,
    positions: [],
    trades: [],
    benchmarks: [{ field: "benchmark_main_nav", label: "Main" }],
    tradingCalendar: calendar,
    navReturnProvenance: {},
    contract: {
      blockers: ["as_of_benchmark_after_analysis_date"],
      as_of_matrix: { analysis_trading_date: "2026-02-03" }
    }
  }, {
    startDate: "2026-02-01",
    endDate: "",
    benchmarkField: "benchmark_main_nav",
    selectedBenchmarkFields: ["benchmark_main_nav"]
  });
  const comparison = dashboard.benchmarkComparison;
  assert.deepEqual(Array.from(dashboard.navRows, row => row.date), ["2026-02-02", "2026-02-03"]);
  assert.deepEqual(Array.from(dashboard.performance.navSeries, row => row.date), ["2026-02-02", "2026-02-03"]);
  assert.deepEqual(Array.from(comparison.portfolioSeries, row => row.date), ["2026-02-02", "2026-02-03"]);
  assert.deepEqual(Array.from(comparison.portfolioDrawdown, row => row.date), ["2026-02-02", "2026-02-03"]);
  assert.deepEqual(Array.from(comparison.selectedBenchmarks[0].series, row => row.date), ["2026-02-02", "2026-02-03"]);
  assert.deepEqual(Array.from(comparison.selectedBenchmarks[0].drawdown, row => row.date), ["2026-02-02", "2026-02-03"]);
  assert.ok(Math.abs(dashboard.performance.monthly[0].value - 0.21) < 1e-12);
  assert.ok(Math.abs(dashboard.performance.kpis.total_return - comparison.portfolioRow.totalReturn) < 1e-12);
  assert.ok(dashboard.warnings.some(message => message.includes("capped at analysis_trading_date=2026-02-03")));
}

{
  const portfolioReturns = [
    { date: "2026-01-02", dateObj: new Date("2026-01-02"), daily_return: 0.10 }
  ];
  const benchmarkReturns = [
    { date: "2026-01-02", dateObj: new Date("2026-01-02"), daily_return: 0.05 }
  ];
  const excess = Metrics.calculateExcessNav(portfolioReturns, benchmarkReturns);
  assert.ok(Math.abs(excess[0].value - (1.10 / 1.05)) < 1e-12);
}

{
  const rows = [
    point("2026-01-30", 1.00),
    point("2026-02-27", 1.10),
    point("2026-03-31", 1.21)
  ];
  const result = Metrics.computePerformance(rows, "benchmark_main_nav");
  assert.equal(result.monthly[0].value, null);
  assert.ok(Math.abs(result.monthly[1].value - 0.10) < 1e-12);
  assert.ok(Math.abs(result.monthly[2].value - 0.10) < 1e-12);
  assert.equal(result.monthly[2].anchor, "previous_month_end");
}

{
  const trades = [{
    trade_date: "2026-01-02",
    dateObj: new Date("2026-01-02"),
    ticker: "TEST.SZ",
    name: "示例公司",
    side: "buy",
    price: 10,
    quantity: 10,
    trade_amount: 100,
    fee: null,
    theme: "示例主题"
  }];
  const result = Metrics.computeTrades(trades, { positions: [] });
  assert.equal(result.totals.fee, null);
  assert.equal(result.totals.fee_unknown_count, 1);
}

{
  const trades = [
    {
      trade_date: "2026-01-02", dateObj: new Date("2026-01-02"), ticker: "TEST.SZ",
      name: "示例公司", side: "buy", price: 10, quantity: 10, trade_amount: 100,
      fee: null, theme: "示例主题"
    },
    {
      trade_date: "2026-01-05", dateObj: new Date("2026-01-05"), ticker: "TEST.SZ",
      name: "示例公司", side: "sell", price: 12, quantity: 10, trade_amount: 120,
      fee: 1, theme: "示例主题"
    }
  ];
  const result = Metrics.computeTrades(trades, { positions: [] });
  assert.equal(result.closed[0].gross_realized_pnl, 20);
  assert.equal(result.closed[0].net_realized_pnl, null);
  assert.equal(result.totals.realized_pnl, null);
  assert.equal(result.totals.trade_win_rate, null);
  assert.equal(result.totals.profit_factor, null);
}

console.log("dashboard_contract_v2.test.js: ok");
