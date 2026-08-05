(function () {
  "use strict";

  var Contract = window.DashboardContractV5;

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    var node = byId(id);
    if (node) node.textContent = value === null || value === undefined || value === "" ? "—" : String(value);
  }

  function formatWeight(value) {
    var number = Number(value);
    return Number.isFinite(number) ? (number * 100).toFixed(2) + "%" : "—";
  }

  function makeCell(row, value, className) {
    var cell = document.createElement("td");
    cell.textContent = value;
    if (className) cell.className = className;
    row.appendChild(cell);
  }

  function renderBlockers(snapshot) {
    var panel = byId("blockerPanel");
    var list = byId("blockerList");
    list.replaceChildren();
    panel.hidden = snapshot.blockers.length === 0;
    snapshot.blockers.forEach(function (blocker) {
      var item = document.createElement("li");
      item.textContent = blocker;
      list.appendChild(item);
    });
  }

  function renderTargets(publicRun) {
    var body = byId("targetRows");
    body.replaceChildren();
    publicRun.targets.forEach(function (target) {
      var row = document.createElement("tr");
      makeCell(row, target.symbol, "mono");
      makeCell(row, target.lane);
      makeCell(row, formatWeight(target.current_target), "numeric");
      makeCell(row, formatWeight(target.final_target), "numeric");
      body.appendChild(row);
    });
  }

  function renderReferences(publicRun) {
    var refs = [
      ["Active pointer", publicRun.active_pointer_ref],
      ["Mainline run", publicRun.mainline_run_ref],
      ["Formal output", publicRun.formal_output_ref],
      ["Portfolio output", publicRun.portfolio_output_ref],
      ["Source closure", publicRun.source_closure_ref]
    ];
    var body = byId("referenceRows");
    body.replaceChildren();
    refs.forEach(function (entry) {
      var row = document.createElement("tr");
      makeCell(row, entry[0]);
      makeCell(row, entry[1].schema_id, "mono");
      makeCell(row, entry[1].relative_path, "mono path-cell");
      makeCell(row, entry[1].byte_sha256, "mono hash-cell");
      body.appendChild(row);
    });
  }

  function renderAuthority(publicRun) {
    var body = byId("authorityRows");
    body.replaceChildren();
    Object.keys(publicRun.authority_flags).sort().forEach(function (key) {
      var row = document.createElement("tr");
      makeCell(row, key, "mono");
      makeCell(row, publicRun.authority_flags[key] ? "ENABLED" : "DISABLED", "disabled-value");
      body.appendChild(row);
    });
  }

  function renderUnavailable(snapshot) {
    setText("strategyId", "Unavailable");
    setText("runId", "Unavailable");
    setText("grossWeight", "—");
    setText("cashWeight", "—");
    setText("targetCount", "0");
    setText("semanticHash", "Unavailable");
    byId("activeContent").hidden = true;
    byId("emptyState").hidden = false;
    setText(
      "emptyStateText",
      snapshot.runtime_state === "V17_MAINLINE_UNINITIALIZED"
        ? "尚未发布单策略 active pointer。Dashboard 不会扫描历史运行，也不会显示替代数据。"
        : "主线 public run 未通过严格校验。Dashboard 已关闭组合详情。"
    );
  }

  function renderActive(snapshot) {
    var publicRun = snapshot.public_run;
    setText("strategyId", publicRun.canonical_strategy_id);
    setText("runId", publicRun.run_id);
    setText("grossWeight", formatWeight(publicRun.gross_weight));
    setText("cashWeight", formatWeight(publicRun.cash_weight));
    setText("targetCount", publicRun.targets.length);
    setText("semanticHash", publicRun.semantic_sha256);
    setText("protocolValue", publicRun.protocol);
    setText("marketValue", publicRun.market);
    setText("capabilityValue", publicRun.capability);
    setText("authoritySource", publicRun.authority_source);
    byId("activeContent").hidden = false;
    byId("emptyState").hidden = true;
    renderTargets(publicRun);
    renderReferences(publicRun);
    renderAuthority(publicRun);
  }

  function render() {
    var snapshot = Contract.deriveSnapshot(window.MyQuantV17MainlinePublicRun);
    var active = snapshot.runtime_state === "ACTIVE";
    var status = byId("runtimeStatus");
    status.textContent = snapshot.runtime_state;
    status.className = "status-pill " + (active ? "active" : "blocked");
    setText("inputSchema", snapshot.input_schema_id);
    renderBlockers(snapshot);
    if (active) renderActive(snapshot);
    else renderUnavailable(snapshot);
  }

  document.addEventListener("DOMContentLoaded", render);
})();
