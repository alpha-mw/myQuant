(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CNAggressiveDashboardContractV1 = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  var SCHEMA_VERSION = "cn_aggressive_dashboard.v1";
  var SHA256_RE = /^[0-9a-f]{64}$/;
  var RECORD_RE = /^[0-9]{8}_[0-9]{4}$/;
  var SYMBOL_RE = /^[0-9]{6}\.(?:SH|SZ|BJ)$/;
  var FLAG_KEYS = [
    "benchmark_provider_calls",
    "broker_calls",
    "candidate_generation",
    "holdings_writes",
    "order_calls",
    "portfolio_recomputation",
    "provider_calls",
    "strategy_record_writes",
    "trade_calls",
    "v17_pointer_mutation"
  ];

  function isObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function validateBundle(value) {
    var errors = [];
    if (!isObject(value)) return { valid: false, errors: ["bundle must be an object"] };
    if (value.schema_version !== SCHEMA_VERSION) errors.push("schema_version is invalid");
    if (["FRESH", "PARTIAL", "BLOCKED"].indexOf(value.status) < 0) errors.push("status is invalid");
    if (value.market !== "CN") errors.push("market must be CN");
    if (value.strategy_label !== "aggressive_tech_manufacturing") errors.push("strategy_label is invalid");
    if (value.strategy_id_kind !== "HISTORICAL_DISPLAY_LABEL_NOT_V17_CANONICAL_ID") {
      errors.push("strategy_id_kind is invalid");
    }
    if (value.read_only !== true) errors.push("read_only must be true");
    if (!isObject(value.authority_flags)) {
      errors.push("authority_flags is invalid");
    } else {
      var keys = Object.keys(value.authority_flags).sort();
      if (keys.join("|") !== FLAG_KEYS.slice().sort().join("|")) errors.push("authority_flags shape is invalid");
      FLAG_KEYS.forEach(function (key) {
        if (value.authority_flags[key] !== false) errors.push("authority_flags." + key + " must be false");
      });
    }
    if (!SHA256_RE.test(value.content_sha256 || "")) errors.push("content_sha256 is invalid");
    if (value.status !== "BLOCKED") {
      if (!RECORD_RE.test(value.latest_valid_record || "")) errors.push("latest_valid_record is invalid");
      if (!RECORD_RE.test(value.previous_valid_record || "")) errors.push("previous_valid_record is invalid");
      if (!Array.isArray(value.positions) || value.positions.length < 1) {
        errors.push("positions are missing");
      } else {
        var seen = {};
        value.positions.forEach(function (position, index) {
          if (!isObject(position) || !SYMBOL_RE.test(position.symbol || "") || !position.name) {
            errors.push("positions[" + index + "] identity is invalid");
            return;
          }
          if (seen[position.symbol]) errors.push("positions contain duplicate symbols");
          seen[position.symbol] = true;
          ["shares", "avg_cost", "recorded_price", "market_value", "nav_weight", "equity_weight"].forEach(function (key) {
            if (!finite(position[key])) errors.push("positions[" + index + "]." + key + " is invalid");
          });
          if (position.evidence_status !== "HASH_BOUND_EFFECTIVE_LEDGER") {
            errors.push("positions[" + index + "] is not hash-bound");
          }
        });
      }
      if (!isObject(value.portfolio) || value.portfolio.return_method !== "initial_capital_return_excluding_external_flows") {
        errors.push("portfolio return method is invalid");
      } else {
        ["performance_initial_capital", "excluded_external_flow", "adjusted_total_value", "cumulative_profit_excluding_external_flow", "cumulative_return"].forEach(function (key) {
          if (!finite(value.portfolio[key])) errors.push("portfolio." + key + " is invalid");
        });
      }
      if (!isObject(value.history) || !RECORD_RE.test(value.history.archive_start_record || "")) {
        errors.push("historical performance summary is missing");
      } else if (!Array.isArray(value.history.funding_events) || !finite(value.history.net_external_flow)) {
        errors.push("historical funding summary is invalid");
      } else if (!isObject(value.portfolio) || !Array.isArray(value.portfolio.performance_points) || value.portfolio.performance_points.length < 2) {
        errors.push("historical performance points are missing");
      } else {
        var firstPoint = value.portfolio.performance_points[0];
        if (firstPoint.record !== value.history.archive_start_record || firstPoint.date !== value.history.archive_start_date) {
          errors.push("historical performance start is inconsistent");
        }
        value.portfolio.performance_points.forEach(function (point, index) {
          ["total_value", "excluded_external_flow", "adjusted_total_value", "portfolio_unit_nav", "portfolio_cumulative_return", "csi300_nav", "csi300_cumulative_return", "star50_nav", "star50_cumulative_return", "chinext_nav", "chinext_cumulative_return", "cumulative_excess_return"].forEach(function (key) {
            if (!finite(point[key])) errors.push("performance_points[" + index + "]." + key + " is invalid");
          });
        });
        if (finite(value.portfolio.performance_initial_capital) && (
          Math.abs(firstPoint.adjusted_total_value - value.portfolio.performance_initial_capital) > 0.01 ||
          Math.abs(firstPoint.portfolio_unit_nav - 1) > 1e-9
        )) errors.push("performance initial capital baseline is inconsistent");
        var lastPoint = value.portfolio.performance_points[value.portfolio.performance_points.length - 1];
        if (finite(lastPoint.adjusted_total_value) && finite(lastPoint.total_value) && finite(lastPoint.excluded_external_flow) &&
            Math.abs(lastPoint.adjusted_total_value - (lastPoint.total_value - lastPoint.excluded_external_flow)) > 0.01) {
          errors.push("performance external flow exclusion is inconsistent");
        }
        value.history.funding_events.forEach(function (event, index) {
          ["amount", "total_value_before", "total_value_after"].forEach(function (key) {
            if (!finite(event[key])) errors.push("funding_events[" + index + "]." + key + " is invalid");
          });
          if (!SHA256_RE.test(event.evidence_sha256 || "")) errors.push("funding_events[" + index + "] evidence SHA is invalid");
        });
      }
      var expectedBenchmarks = {
        CSI300: "沪深300|000300.SH",
        STAR50: "科创50|000688.SH",
        CHINEXT: "创业板指|399006.SZ"
      };
      if (!Array.isArray(value.benchmarks) || value.benchmarks.length !== 3) {
        errors.push("verified benchmark set is missing");
      } else {
        var actualBenchmarks = {};
        value.benchmarks.forEach(function (row) {
          if (!isObject(row) || !Array.isArray(row.missing_dates) || row.missing_dates.length !== 0) return;
          actualBenchmarks[row.id] = row.name + "|" + row.ts_code;
        });
        var benchmarkSetValid = Object.keys(expectedBenchmarks).every(function (id) {
          return actualBenchmarks[id] === expectedBenchmarks[id];
        }) && Object.keys(actualBenchmarks).length === Object.keys(expectedBenchmarks).length;
        if (!benchmarkSetValid) {
          errors.push("verified benchmark set is invalid");
        }
      }
      if (value.i1_research === null && value.i1_display_status !== "NOT_DISPLAYED_NO_EXACT_HASH_BOUND_I1_ARTIFACT") {
        errors.push("I1 absence status is invalid");
      }
      if (!Array.isArray(value.blockers) || value.blockers.length !== 0) errors.push("usable bundle must have no blockers");
    }
    return { valid: errors.length === 0, errors: errors };
  }

  function deriveSnapshot(value) {
    if (value === null || value === undefined) {
      return {
        schema_version: SCHEMA_VERSION,
        status: "BLOCKED",
        blockers: ["private_generated_dashboard_bundle_missing"],
        bundle: null
      };
    }
    var validation = validateBundle(value);
    if (!validation.valid) {
      return {
        schema_version: SCHEMA_VERSION,
        status: "BLOCKED",
        blockers: ["dashboard_bundle_invalid: " + validation.errors.join("; ")],
        bundle: null
      };
    }
    return {
      schema_version: SCHEMA_VERSION,
      status: value.status,
      blockers: value.blockers,
      bundle: value
    };
  }

  return {
    SCHEMA_VERSION: SCHEMA_VERSION,
    deriveSnapshot: deriveSnapshot,
    validateBundle: validateBundle
  };
});
