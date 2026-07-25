(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.V17ShadowDashboard = api;
    if (root.document) {
      const start = function () {
        api.render(root.document, root.V17ShadowLatest);
      };
      if (root.document.readyState === "loading") {
        root.document.addEventListener("DOMContentLoaded", start, { once: true });
      } else {
        start();
      }
    }
  }
})(typeof window !== "undefined" ? window : undefined, function () {
  "use strict";

  const CONTRACT_VERSION = "dashboard_contract.v17-shadow.v1";
  const POINTER_VERSION = "myquant.v17.shadow-latest-pointer.v1";
  const OUTPUT_VERSION = "myquant.v17.shadow-output.v1";
  const LATEST_PATH = "results/v17_shadow/_latest/shadow.json";
  const TERMINALS = new Set([
    "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
    "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
    "SHADOW_PORTFOLIO_INFEASIBLE",
    "HARD_STOP_SNAPSHOT_DRIFT",
    "HARD_STOP_INVALID_EVIDENCE",
  ]);
  const BUSINESS_TERMINALS = new Set([
    "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
    "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
    "SHADOW_PORTFOLIO_INFEASIBLE",
  ]);
  const SHA256 = /^[0-9a-f]{64}$/;

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function hasExactKeys(value, keys) {
    return isObject(value)
      && Object.keys(value).sort().join("\u0000") === [...keys].sort().join("\u0000");
  }

  function unavailable(reason) {
    return {
      availability: "UNAVAILABLE",
      reason,
      contract: null,
      pointer: null,
      output: null,
      isBusinessTerminal: false,
    };
  }

  function validSource(source, availability) {
    const keys = [
      "path",
      "latest_pointer_sha256",
      "ledger_sha256",
      "output_sha256",
      "readback_verified",
      "fallback_used",
    ];
    if (!hasExactKeys(source, keys)
      || source.path !== LATEST_PATH
      || source.fallback_used !== false) {
      return false;
    }
    if (availability === "UNAVAILABLE") {
      return source.latest_pointer_sha256 === null
        && source.ledger_sha256 === null
        && source.output_sha256 === null
        && source.readback_verified === false;
    }
    return SHA256.test(source.latest_pointer_sha256)
      && SHA256.test(source.ledger_sha256)
      && SHA256.test(source.output_sha256)
      && source.readback_verified === true;
  }

  function validPointer(pointer) {
    const keys = [
      "version",
      "run_id",
      "terminal_state",
      "ledger_path",
      "ledger_sha256",
      "output_path",
      "output_sha256",
      "published_at",
      "publication_mode",
      "authority",
      "semantic_sha256",
    ];
    return hasExactKeys(pointer, keys)
      && pointer.version === POINTER_VERSION
      && typeof pointer.run_id === "string"
      && pointer.run_id.length > 0
      && TERMINALS.has(pointer.terminal_state)
      && pointer.ledger_path === `results/v17_shadow/runs/${pointer.run_id}/ledger.json`
      && typeof pointer.output_path === "string"
      && pointer.output_path.startsWith("results/v17_shadow/outcomes/")
      && pointer.output_path.endsWith(".json")
      && SHA256.test(pointer.ledger_sha256)
      && SHA256.test(pointer.output_sha256)
      && typeof pointer.published_at === "string"
      && pointer.published_at.length >= 20
      && (pointer.publication_mode === "NORMAL" || pointer.publication_mode === "REPAIR")
      && pointer.authority === false
      && SHA256.test(pointer.semantic_sha256);
  }

  function validTerminalShape(output) {
    const rankIsObject = isObject(output.rank_output);
    const portfolioIsObject = isObject(output.portfolio_output);
    if (output.terminal_state === "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION") {
      return rankIsObject && portfolioIsObject && output.blockers.length === 0;
    }
    if (output.terminal_state === "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"
      || output.terminal_state === "SHADOW_PORTFOLIO_INFEASIBLE") {
      return rankIsObject && output.portfolio_output === null && output.blockers.length > 0;
    }
    return output.rank_output === null
      && output.portfolio_output === null
      && output.blockers.length > 0;
  }

  function validOutput(output) {
    const keys = [
      "version",
      "run_id",
      "strategy_id",
      "market",
      "cutoff",
      "terminal_state",
      "rank_output",
      "portfolio_output",
      "blockers",
      "source_manifest_sha256",
      "ledger_predecessor_sha256",
      "generated_at",
      "authority",
      "semantic_sha256",
    ];
    return hasExactKeys(output, keys)
      && output.version === OUTPUT_VERSION
      && typeof output.run_id === "string"
      && output.run_id.length > 0
      && typeof output.strategy_id === "string"
      && output.strategy_id.length > 0
      && output.market === "CN"
      && typeof output.cutoff === "string"
      && output.cutoff.length >= 20
      && TERMINALS.has(output.terminal_state)
      && Array.isArray(output.blockers)
      && new Set(output.blockers).size === output.blockers.length
      && output.blockers.every((item) => typeof item === "string" && item.length > 0)
      && SHA256.test(output.source_manifest_sha256)
      && SHA256.test(output.ledger_predecessor_sha256)
      && typeof output.generated_at === "string"
      && output.generated_at.length >= 20
      && output.authority === false
      && SHA256.test(output.semantic_sha256)
      && validTerminalShape(output);
  }

  function normalize(contract) {
    const keys = [
      "schema_version",
      "schema_sha256",
      "availability",
      "generated_at",
      "reason",
      "source",
      "latest_pointer",
      "terminal_output",
      "authority",
    ];
    if (!hasExactKeys(contract, keys)
      || contract.schema_version !== CONTRACT_VERSION
      || !SHA256.test(contract.schema_sha256)
      || typeof contract.generated_at !== "string"
      || contract.generated_at.length < 20
      || contract.authority !== false) {
      return unavailable(contract ? "v17_dashboard_contract_invalid" : "v17_latest_loader_missing");
    }
    if (contract.availability === "UNAVAILABLE") {
      if (typeof contract.reason !== "string"
        || contract.reason.length === 0
        || contract.latest_pointer !== null
        || contract.terminal_output !== null
        || !validSource(contract.source, "UNAVAILABLE")) {
        return unavailable("v17_dashboard_contract_invalid");
      }
      return unavailable(contract.reason);
    }
    if (contract.availability !== "AVAILABLE"
      || contract.reason !== null
      || !validSource(contract.source, "AVAILABLE")
      || !validPointer(contract.latest_pointer)
      || !validOutput(contract.terminal_output)) {
      return unavailable("v17_dashboard_contract_invalid");
    }
    const pointer = contract.latest_pointer;
    const output = contract.terminal_output;
    if (pointer.run_id !== output.run_id
      || pointer.terminal_state !== output.terminal_state
      || contract.source.ledger_sha256 !== pointer.ledger_sha256
      || contract.source.output_sha256 !== pointer.output_sha256) {
      return unavailable("v17_dashboard_cross_binding_invalid");
    }
    return {
      availability: "AVAILABLE",
      reason: null,
      contract,
      pointer,
      output,
      isBusinessTerminal: BUSINESS_TERMINALS.has(output.terminal_state),
    };
  }

  function text(target, value) {
    if (target) {
      target.textContent = value == null || value === "" ? "—" : String(value);
    }
  }

  function appendRows(target, rows) {
    if (!target) {
      return;
    }
    target.replaceChildren();
    rows.forEach(function (row) {
      const item = target.ownerDocument.createElement("div");
      item.className = "v17-row";
      const label = target.ownerDocument.createElement("span");
      label.className = "v17-row-label";
      label.textContent = row[0];
      const value = target.ownerDocument.createElement("code");
      value.textContent = row[1] == null || row[1] === "" ? "—" : String(row[1]);
      item.append(label, value);
      target.append(item);
    });
  }

  function rankedRows(rankOutput) {
    if (!isObject(rankOutput)) {
      return [];
    }
    if (!Array.isArray(rankOutput.initial_ranked_symbols)
      || !Array.isArray(rankOutput.rows)) {
      return [];
    }
    const rows = new Map();
    for (const value of rankOutput.rows) {
      if (!isObject(value) || typeof value.symbol !== "string" || rows.has(value.symbol)) {
        return [];
      }
      rows.set(value.symbol, value);
    }
    return rankOutput.initial_ranked_symbols.map(function (symbol, index) {
      const value = rows.get(symbol);
      if (!value) {
        return null;
      }
      const eligibility = value.f_eligible === true ? "F_ELIGIBLE" : "F_INELIGIBLE";
      return {
        rank: index + 1,
        symbol,
        name: value.company_name || "UNKNOWN_NAME",
        fundamental: `${eligibility} / ${value.deep_status || "UNAVAILABLE"}`,
        timing: value.timing_state || "UNREADY",
      };
    }).filter(function (value) {
      return value !== null;
    });
  }

  function renderRanks(document, rankOutput) {
    const body = document.getElementById("rankTableBody");
    if (!body) {
      return;
    }
    body.replaceChildren();
    const rows = rankedRows(rankOutput);
    if (rows.length === 0) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.textContent = "当前 terminal 未提供可渲染的候选列表。";
      row.append(cell);
      body.append(row);
      return;
    }
    rows.forEach(function (candidate) {
      const row = document.createElement("tr");
      [
        candidate.rank,
        `${candidate.symbol} ${candidate.name}`,
        candidate.fundamental,
        candidate.timing,
        "SHADOW ONLY",
      ].forEach(function (value) {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.append(cell);
      });
      body.append(row);
    });
  }

  function render(document, rawContract) {
    const view = normalize(rawContract);
    const status = document.getElementById("v17Status");
    if (status) {
      status.className = `v17-status ${view.availability.toLowerCase()}`;
    }
    text(status, view.availability);
    text(document.getElementById("v17Reason"), view.reason);
    if (view.availability !== "AVAILABLE") {
      text(document.getElementById("v17RunId"), "UNAVAILABLE");
      text(document.getElementById("v17Terminal"), "UNAVAILABLE");
      text(document.getElementById("v17Cutoff"), "UNAVAILABLE");
      text(document.getElementById("v17PortfolioState"), "UNAVAILABLE");
      appendRows(document.getElementById("v17Evidence"), [
        ["source", LATEST_PATH],
        ["readback", "unavailable"],
        ["fallback", "false"],
      ]);
      renderRanks(document, null);
      text(document.getElementById("v17RankJson"), "UNAVAILABLE");
      text(document.getElementById("v17PortfolioJson"), "UNAVAILABLE");
      return view;
    }

    const output = view.output;
    const source = view.contract.source;
    text(document.getElementById("v17RunId"), output.run_id);
    text(document.getElementById("v17Terminal"), output.terminal_state);
    text(document.getElementById("v17Cutoff"), output.cutoff);
    text(
      document.getElementById("v17PortfolioState"),
      output.portfolio_output === null ? "NO PORTFOLIO" : "SHADOW PORTFOLIO",
    );
    appendRows(document.getElementById("v17Evidence"), [
      ["latest pointer", source.latest_pointer_sha256],
      ["ledger", source.ledger_sha256],
      ["terminal output", source.output_sha256],
      ["source manifest", output.source_manifest_sha256],
      ["schema", view.contract.schema_sha256],
      ["fallback", "false"],
      ["authority", "false"],
    ]);
    const blockerList = document.getElementById("v17Blockers");
    if (blockerList) {
      blockerList.replaceChildren();
      const blockers = output.blockers.length > 0 ? output.blockers : ["无阻断项"];
      blockers.forEach(function (blocker) {
        const item = document.createElement("li");
        item.textContent = blocker;
        blockerList.append(item);
      });
    }
    renderRanks(document, output.rank_output);
    text(
      document.getElementById("v17RankJson"),
      JSON.stringify(output.rank_output, null, 2),
    );
    text(
      document.getElementById("v17PortfolioJson"),
      JSON.stringify(output.portfolio_output, null, 2),
    );
    return view;
  }

  return {
    CONTRACT_VERSION,
    LATEST_PATH,
    normalize,
    rankedRows,
    render,
  };
});
