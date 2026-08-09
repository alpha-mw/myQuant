(function () {
  "use strict";

  var Contract = window.CNAggressiveDashboardContractV1;

  function byId(id) { return document.getElementById(id); }
  function setText(id, value) {
    var node = byId(id);
    if (node) node.textContent = value === null || value === undefined || value === "" ? "—" : String(value);
  }
  function money(value) {
    var number = Number(value);
    return Number.isFinite(number) ? "¥" + number.toLocaleString("zh-CN", { maximumFractionDigits: 2 }) : "UNKNOWN";
  }
  function number(value) {
    var parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toLocaleString("zh-CN", { maximumFractionDigits: 4 }) : "UNKNOWN";
  }
  function percent(value, signed) {
    var parsed = Number(value);
    if (!Number.isFinite(parsed)) return "UNKNOWN";
    var result = (parsed * 100).toFixed(2) + "%";
    return signed && parsed > 0 ? "+" + result : result;
  }
  function makeCell(row, value, className) {
    var cell = document.createElement("td");
    cell.textContent = value;
    if (className) cell.className = className;
    row.appendChild(cell);
  }
  function metric(list, label, value) {
    var wrapper = document.createElement("div");
    var term = document.createElement("dt");
    var detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value;
    wrapper.appendChild(term);
    wrapper.appendChild(detail);
    list.appendChild(wrapper);
  }
  function renderBlockers(blockers) {
    var panel = byId("blockerPanel");
    var list = byId("blockerList");
    list.replaceChildren();
    panel.hidden = blockers.length === 0;
    blockers.forEach(function (blocker) {
      var item = document.createElement("li");
      item.textContent = blocker;
      list.appendChild(item);
    });
  }
  function renderPositions(bundle) {
    var body = byId("positionRows");
    body.replaceChildren();
    bundle.positions.forEach(function (position) {
      var row = document.createElement("tr");
      makeCell(row, position.symbol + " " + position.name, "identity-cell");
      makeCell(row, number(position.shares), "numeric");
      makeCell(row, money(position.avg_cost), "numeric");
      makeCell(row, money(position.recorded_price) + " · " + position.price_date, "numeric");
      makeCell(row, money(position.market_value), "numeric");
      makeCell(row, percent(position.nav_weight), "numeric");
      makeCell(row, percent(position.equity_weight), "numeric");
      makeCell(row, money(position.unrealized_pnl), "numeric " + (position.unrealized_pnl < 0 ? "negative" : "positive"));
      makeCell(row, position.realized_pnl === null ? "UNKNOWN" : money(position.realized_pnl));
      makeCell(row, position.thesis_status + " · HASH-BOUND", "evidence-value");
      body.appendChild(row);
    });
  }
  function renderChanges(bundle) {
    setText("changeSubtitle", bundle.previous_valid_record + " → " + bundle.latest_valid_record);
    var body = byId("changeRows");
    body.replaceChildren();
    bundle.changes.forEach(function (change) {
      var row = document.createElement("tr");
      makeCell(row, change.symbol + " " + change.name, "identity-cell");
      makeCell(row, change.change_type, "mono");
      makeCell(row, number(change.previous_shares), "numeric");
      makeCell(row, number(change.current_shares), "numeric");
      makeCell(row, number(change.share_delta), "numeric");
      makeCell(row, percent(change.nav_weight_delta, true), "numeric");
      makeCell(row, percent(change.equity_weight_delta, true), "numeric");
      body.appendChild(row);
    });
  }
  function renderMetrics(bundle) {
    var portfolio = bundle.portfolio;
    var benchmark = bundle.benchmarks[0];
    var performance = byId("performanceList");
    performance.replaceChildren();
    metric(performance, "归档统计区间", portfolio.performance_start_date + " → " + portfolio.performance_end_date);
    metric(performance, "归档以来 TWR", percent(portfolio.cumulative_twr, true));
    metric(performance, "沪深300", percent(benchmark.return, true));
    metric(performance, "累计超额", percent(benchmark.excess_return, true));
    metric(performance, "组合最大回撤", percent(portfolio.max_drawdown));
    metric(performance, "沪深300最大回撤", percent(benchmark.max_drawdown));
    metric(performance, "最新记录区间收益", percent(portfolio.latest_record_interval_return, true));
    metric(performance, "最新区间换手", percent(portfolio.latest_interval_turnover));
    metric(performance, "当前未实现 P&L", money(portfolio.current_unrealized_pnl));
    metric(performance, "累计已实现 P&L", "UNKNOWN");
    metric(performance, "费用 / 毛净", portfolio.fee_basis + " / " + portfolio.gross_or_net);

    var concentration = byId("concentrationList");
    concentration.replaceChildren();
    metric(concentration, "持仓数", String(bundle.concentration.holding_count));
    metric(concentration, "Top 1 权益权重", percent(bundle.concentration.top1_equity_weight));
    metric(concentration, "Top 3 权益权重", percent(bundle.concentration.top3_equity_weight));
    metric(concentration, "权益 HHI", number(bundle.concentration.equity_hhi));
    metric(concentration, "现金", money(portfolio.cash) + " · " + percent(portfolio.cash_weight));
    metric(concentration, "权益市值 / 仓位", money(portfolio.market_value) + " · " + percent(portfolio.gross_exposure));
    metric(concentration, "行业 / 主题", "UNKNOWN · ledger 未 hash-bind 此口径");
    metric(concentration, "数据新鲜度", bundle.latest_data_date + " · " + bundle.data_age_calendar_days + " calendar days");
  }
  function historyEvidenceLabel(value) {
    if (value === "ARCHIVE_INCEPTION_EXACT_BYTES_NO_DECLARED_SHA") return "LEGACY BASELINE · exact bytes";
    if (value === "LEGACY_EXACT_BYTES_NO_DECLARED_SHA") return "LEGACY · exact bytes";
    if (value === "HASH_BOUND_CURRENT_CLOSURE") return "CURRENT · hash-bound";
    return value || "UNKNOWN";
  }
  function renderHistory(bundle) {
    var history = bundle.history;
    var summary = byId("historySummary");
    summary.replaceChildren();
    metric(summary, "归档起点", history.archive_start_date + " · " + history.archive_start_record);
    metric(summary, "首个 P&L", history.first_pnl_date + " · " + history.first_pnl_record);
    metric(summary, "纳入记录 / 估值点", history.included_record_count + " / " + history.performance_point_count);
    metric(summary, "历史证据", history.evidence_status);
    metric(summary, "旧档 exact-byte", String(history.legacy_exact_byte_record_count));
    metric(summary, "资金流事件 / 净额", history.funding_events.length + " / " + money(history.net_external_flow));
    metric(summary, "历史排除记录", String(history.rejected_record_count));

    var body = byId("performanceRows");
    body.replaceChildren();
    bundle.portfolio.performance_points.slice().reverse().forEach(function (point) {
      var row = document.createElement("tr");
      makeCell(row, point.date);
      makeCell(row, point.record, "mono");
      makeCell(row, money(point.total_value), "numeric");
      makeCell(row, number(point.portfolio_unit_nav), "numeric");
      makeCell(row, percent(point.portfolio_cumulative_return, true), "numeric");
      makeCell(row, percent(point.csi300_cumulative_return, true), "numeric");
      makeCell(row, percent(point.cumulative_excess_return, true), "numeric");
      makeCell(row, historyEvidenceLabel(point.evidence_status), "evidence-value");
      body.appendChild(row);
    });
  }
  function renderRisks(bundle) {
    var grid = byId("riskGrid");
    grid.replaceChildren();
    bundle.risks.forEach(function (risk) {
      var card = document.createElement("article");
      card.className = "risk-card " + risk.severity.toLowerCase();
      var label = document.createElement("span");
      var title = document.createElement("strong");
      var detail = document.createElement("p");
      label.textContent = risk.severity;
      title.textContent = risk.code;
      detail.textContent = risk.detail;
      card.appendChild(label);
      card.appendChild(title);
      card.appendChild(detail);
      grid.appendChild(card);
    });
  }
  function renderEvidence(bundle) {
    var body = byId("evidenceRows");
    body.replaceChildren();
    var entries = [
      ["Manifest", bundle.current_evidence.manifest_path, bundle.current_evidence.manifest_sha256],
      ["Manual execution manifest", bundle.current_evidence.manual_manifest_path, bundle.current_evidence.manual_manifest_sha256],
      ["Effective ledger", bundle.current_evidence.ledger_path, bundle.current_evidence.ledger_sha256],
      ["P&L", bundle.current_evidence.pnl_path, bundle.current_evidence.pnl_sha256],
      ["Archive baseline manifest", bundle.history.baseline_manifest_path, bundle.history.baseline_manifest_sha256],
      ["Archive baseline ledger", bundle.history.baseline_ledger_path, bundle.history.baseline_ledger_sha256],
      ["CSI300", bundle.benchmarks[0].source_path, bundle.benchmarks[0].source_sha256]
    ];
    bundle.history.funding_events.forEach(function (event) {
      entries.push(["Funding " + event.record, event.evidence_path, event.evidence_sha256]);
    });
    entries.forEach(function (entry) {
      var row = document.createElement("tr");
      makeCell(row, entry[0]);
      makeCell(row, entry[1], "mono path-cell");
      makeCell(row, entry[2], "mono hash-cell");
      body.appendChild(row);
    });
  }
  function renderWarnings(bundle) {
    setText("i1Status", bundle.i1_display_status + " · research-only states cannot change holdings");
    var list = byId("warningList");
    list.replaceChildren();
    bundle.warnings.forEach(function (warning) {
      var item = document.createElement("li");
      item.textContent = warning;
      list.appendChild(item);
    });
  }
  function renderBundle(bundle) {
    var benchmark = bundle.benchmarks[0];
    setText("latestRecord", bundle.latest_valid_record);
    setText("dataDate", bundle.latest_data_date);
    setText("totalValue", money(bundle.portfolio.total_value));
    setText("cashExposure", percent(bundle.portfolio.cash_weight) + " / " + percent(bundle.portfolio.gross_exposure));
    setText("twrValue", percent(bundle.portfolio.cumulative_twr, true));
    setText("benchmarkValue", percent(benchmark.return, true) + " / " + percent(benchmark.excess_return, true));
    setText("drawdownValue", percent(bundle.portfolio.max_drawdown));
    setText("pnlValue", money(bundle.portfolio.portfolio_pnl));
    renderPositions(bundle);
    renderChanges(bundle);
    renderMetrics(bundle);
    renderHistory(bundle);
    renderRisks(bundle);
    renderEvidence(bundle);
    renderWarnings(bundle);
    byId("dashboardContent").hidden = false;
  }
  function render() {
    var snapshot = Contract.deriveSnapshot(window.MyQuantCNAggressiveDashboard);
    var status = byId("runtimeStatus");
    status.textContent = snapshot.status;
    status.className = "status-pill " + snapshot.status.toLowerCase();
    renderBlockers(snapshot.blockers);
    if (snapshot.bundle) renderBundle(snapshot.bundle);
  }

  document.addEventListener("DOMContentLoaded", render);
})();
