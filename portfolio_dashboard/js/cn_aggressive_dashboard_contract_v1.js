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
  var LEGACY_RETURN_METHOD = "initial_capital_return_excluding_external_flows";
  var CANONICAL_RETURN_METHOD = "flow_neutral_unitization_v1";
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
    if (value.public_redacted && (
      !Array.isArray(value.positions) || value.positions.length !== 0 ||
      !Array.isArray(value.changes) || value.changes.length !== 0 ||
      !Array.isArray(value.source_refs) || value.source_refs.length !== 0
    )) {
      errors.push("public redaction arrays must be empty");
    }
    if (value.status !== "BLOCKED") {
      if (!RECORD_RE.test(value.latest_valid_record || "")) errors.push("latest_valid_record is invalid");
      if (!RECORD_RE.test(value.previous_valid_record || "")) errors.push("previous_valid_record is invalid");
      if (!Array.isArray(value.positions) ||
          (!value.public_redacted && value.positions.length < 1)) {
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
      if (!Array.isArray(value.changes) ||
          (!value.public_redacted && value.changes.length < 1)) {
        errors.push("changes are missing");
      } else if (!value.public_redacted) {
        var seenChanges = {};
        value.changes.forEach(function (change, index) {
          if (!isObject(change) || !SYMBOL_RE.test(change.symbol || "") || !change.name) {
            errors.push("changes[" + index + "] identity is invalid");
            return;
          }
          if (seenChanges[change.symbol]) {
            errors.push("changes contain duplicate symbols");
          }
          seenChanges[change.symbol] = true;
          if (["NEW", "INCREASED", "REDUCED", "CLOSED", "UNCHANGED"].indexOf(change.change_type) < 0) {
            errors.push("changes[" + index + "].change_type is invalid");
          }
          var numberKeys = [
            "previous_shares",
            "current_shares",
            "share_delta",
            "nav_weight_delta",
            "equity_weight_delta"
          ];
          var marketValueKeys = [
            "previous_market_value",
            "current_market_value",
            "market_value_delta"
          ];
          var marketValuePresent = marketValueKeys.map(function (key) {
            return Object.prototype.hasOwnProperty.call(change, key);
          });
          if (marketValuePresent.some(Boolean) && !marketValuePresent.every(Boolean)) {
            errors.push("changes[" + index + "] market value group is incomplete");
            return;
          }
          if (numberKeys.some(function (key) { return !finite(change[key]); })) {
            errors.push("changes[" + index + "] values are invalid");
            return;
          }
          if (marketValuePresent.every(Boolean) && marketValueKeys.some(function (key) {
            return !finite(change[key]);
          })) {
            errors.push("changes[" + index + "] values are invalid");
            return;
          }
          if (Math.abs(change.share_delta - (change.current_shares - change.previous_shares)) > 1e-9) {
            errors.push("changes[" + index + "] share delta is inconsistent");
          }
          var expectedChangeType;
          if (change.previous_shares === 0 && change.current_shares > 0) {
            expectedChangeType = "NEW";
          } else if (change.current_shares === 0 && change.previous_shares > 0) {
            expectedChangeType = "CLOSED";
          } else if (change.current_shares > change.previous_shares) {
            expectedChangeType = "INCREASED";
          } else if (change.current_shares < change.previous_shares) {
            expectedChangeType = "REDUCED";
          } else {
            expectedChangeType = "UNCHANGED";
          }
          if (change.change_type !== expectedChangeType) {
            errors.push("changes[" + index + "] change type is inconsistent");
          }
          if (marketValuePresent.every(Boolean) && Math.abs(change.market_value_delta - (
            change.current_market_value - change.previous_market_value
          )) > 0.01) {
            errors.push("changes[" + index + "] market value delta is inconsistent");
          }
        });
      }
      var returnMethod = isObject(value.portfolio) ? value.portfolio.return_method : "";
      var canonicalReturn = returnMethod === CANONICAL_RETURN_METHOD;
      if (!isObject(value.portfolio) ||
          [LEGACY_RETURN_METHOD, CANONICAL_RETURN_METHOD].indexOf(returnMethod) < 0) {
        errors.push("portfolio return method is invalid");
      } else {
        ["cash", "market_value", "total_value", "cash_weight", "gross_exposure", "portfolio_pnl", "performance_initial_capital", "excluded_external_flow", "adjusted_total_value", "cumulative_profit_excluding_external_flow", "cumulative_return"].forEach(function (key) {
          if (!finite(value.portfolio[key])) errors.push("portfolio." + key + " is invalid");
        });
        if (finite(value.portfolio.cash) && finite(value.portfolio.market_value) && finite(value.portfolio.total_value) &&
            Math.abs(value.portfolio.cash + value.portfolio.market_value - value.portfolio.total_value) > 0.01) {
          errors.push("economic portfolio accounting is inconsistent");
        }
        if (!canonicalReturn && finite(value.portfolio.total_value) && finite(value.portfolio.adjusted_total_value) &&
            Math.abs(value.portfolio.total_value - value.portfolio.adjusted_total_value) > 0.01) {
          errors.push("economic portfolio total is inconsistent");
        }
        var pnlBase = canonicalReturn ? value.portfolio.adjusted_total_value : value.portfolio.total_value;
        if (finite(value.portfolio.portfolio_pnl) && finite(pnlBase) && finite(value.portfolio.performance_initial_capital) &&
            Math.abs(value.portfolio.portfolio_pnl - (pnlBase - value.portfolio.performance_initial_capital)) > 0.01) {
          errors.push("economic portfolio P&L is inconsistent");
        }
        if (finite(value.portfolio.cash_weight) && finite(value.portfolio.gross_exposure) &&
            Math.abs(value.portfolio.cash_weight + value.portfolio.gross_exposure - (value.public_redacted ? 0 : 1)) > 1e-9) {
          errors.push("economic portfolio weights are inconsistent");
        }
        var navWeightTotal = Array.isArray(value.positions) ? value.positions.reduce(function (total, position) {
          return total + (finite(position.nav_weight) ? position.nav_weight : 0);
        }, 0) : NaN;
        if (finite(navWeightTotal) && finite(value.portfolio.gross_exposure) &&
            Math.abs(navWeightTotal - value.portfolio.gross_exposure) > 1e-9) {
          errors.push("position NAV weights are inconsistent");
        }
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
          ["total_value", "excluded_external_flow", "adjusted_total_value", "portfolio_unit_nav", "portfolio_cumulative_return", "csi300_nav", "csi300_cumulative_return", "star50_nav", "star50_cumulative_return", "chinext_nav", "chinext_cumulative_return", "cumulative_excess_return", "risk_free_annual_yield"].forEach(function (key) {
            if (!finite(point[key])) errors.push("performance_points[" + index + "]." + key + " is invalid");
          });
          var expectedAdjusted = canonicalReturn
            ? point.portfolio_unit_nav * value.portfolio.performance_initial_capital
            : point.total_value - point.excluded_external_flow;
          if (finite(point.adjusted_total_value) && finite(expectedAdjusted) &&
              Math.abs(point.adjusted_total_value - expectedAdjusted) > 0.01) {
            errors.push("performance_points[" + index + "] external flow exclusion is inconsistent");
          }
        });
        if (finite(value.portfolio.performance_initial_capital) && (
          Math.abs(firstPoint.adjusted_total_value - value.portfolio.performance_initial_capital) > 0.01 ||
          Math.abs(firstPoint.portfolio_unit_nav - 1) > 1e-9
        )) errors.push("performance initial capital baseline is inconsistent");
        var lastPoint = value.portfolio.performance_points[value.portfolio.performance_points.length - 1];
        if (lastPoint.date !== value.portfolio.performance_end_date ||
            (value.history.latest_performance_date && lastPoint.date !== value.history.latest_performance_date)) {
          errors.push("latest performance date is inconsistent");
        }
        if (finite(value.portfolio.cumulative_return) && finite(lastPoint.portfolio_unit_nav) &&
            Math.abs(value.portfolio.cumulative_return - (lastPoint.portfolio_unit_nav - 1)) > 1e-9) {
          errors.push("latest performance return is inconsistent");
        }
        var currentValuationStatus = String(value.portfolio.current_valuation_status || "");
        var currentValuationIsExplicitlyIncomplete = value.status === "PARTIAL" &&
          currentValuationStatus.indexOf("BLOCKED") === 0 &&
          isObject(value.current_evidence) &&
          value.current_evidence.official_valuation === false &&
          value.current_evidence.valuation_completeness_passed === false &&
          value.current_evidence.valuation_status === currentValuationStatus &&
          Array.isArray(value.warnings) &&
          value.warnings.indexOf("latest_current_valuation_incomplete:" + currentValuationStatus) >= 0;
        var latestPerformanceTotal = canonicalReturn
          ? value.portfolio.adjusted_total_value
          : value.portfolio.total_value;
        if (finite(lastPoint.adjusted_total_value) && finite(latestPerformanceTotal) &&
            Math.abs(lastPoint.adjusted_total_value - latestPerformanceTotal) > 0.01 &&
            !currentValuationIsExplicitlyIncomplete) {
          errors.push("latest performance total is inconsistent");
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
      if (!isObject(value.risk_free) || value.risk_free.tenor !== "1Y" ||
          value.risk_free.source_system !== "chinabond.mof_govt_yield_curve" ||
          value.risk_free.day_count !== "ACT/365" ||
          value.risk_free.alignment !== "interval_start_previous_published_workday" ||
          !finite(value.risk_free.latest_annual_yield) ||
          !Array.isArray(value.risk_free.missing_dates) || value.risk_free.missing_dates.length !== 0) {
        errors.push("verified risk-free series is invalid");
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
