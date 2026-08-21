(function (root, factory) {
  "use strict";
  var v1Contract = null;
  if (typeof module === "object" && module.exports) {
    v1Contract = require("./cn_aggressive_dashboard_contract_v1.js");
  } else if (root) {
    v1Contract = root.CNAggressiveDashboardContractV1;
  }
  var api = factory(v1Contract);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CNAggressiveDashboardContractV2 = api;
})(typeof window !== "undefined" ? window : globalThis, function (V1Contract) {
  "use strict";

  var SCHEMA_VERSION = "cn_aggressive_dashboard.v2";
  var SELECTOR_SCHEMA_VERSION = "cn_aggressive_dashboard_selector.v2";
  var SHA256_RE = /^[0-9a-f]{64}$/;
  var DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
  var RECORD_RE = /^\d{8}_\d{4}$/;
  var SYMBOL_RE = /^[0-9]{6}\.(?:SH|SZ|BJ)$/;
  var ATTEMPT_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
  var VIEW_AUTHORITY = "VIEW_ONLY_NO_STORE_OR_PERFORMANCE_AUTHORITY";
  var FRESHNESS_SCOPE = "DAILY_SYNC_LATEST_VERIFIED_LOCAL_CLOSE";
  var STRICT_CLOSE = "STRICT_CN_EOD_CLOSE";
  var CONTINUITY_STATUSES = [
    "NO_ACTION_BOUND",
    "FINANCIAL_STATE_PUBLICATION",
    "UNCONFIRMED"
  ];
  var FRESHNESS_STATUSES = ["UPDATED", "STALE"];
  var FRESHNESS_REASONS = [
    "CURRENT_DAILY_RECEIPT_AND_LATEST_LOCAL_CLOSE",
    "CURRENT_FINANCIAL_PUBLICATION_AND_LATEST_LOCAL_CLOSE",
    "DAILY_CONTINUITY_RECEIPT_MISSING"
  ];
  var POSITION_PRICE_EVIDENCE = [
    "EXACT_CLOSE",
    "BOUND_SUSPENSION_CARRY_FORWARD"
  ];
  var EFFECT_KEYS = [
    "canonical_effect",
    "ledger_effect",
    "performance_effect",
    "paper_effect",
    "trade_effect"
  ];

  function isObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function approximately(left, right, tolerance) {
    return finite(left) && finite(right) && Math.abs(left - right) <= tolerance;
  }

  function hasExactKeys(value, keys) {
    if (!isObject(value)) return false;
    return Object.keys(value).sort().join("|") === keys.slice().sort().join("|");
  }

  function validDate(value) {
    if (!DATE_RE.test(value || "")) return false;
    var parts = value.split("-").map(Number);
    var date = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
    return date.getUTCFullYear() === parts[0] &&
      date.getUTCMonth() === parts[1] - 1 && date.getUTCDate() === parts[2];
  }

  function validShanghaiDateTime(value) {
    return typeof value === "string" &&
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+08:00$/.test(value) &&
      Number.isFinite(Date.parse(value));
  }

  function shanghaiDate(value) {
    var milliseconds = Date.parse(value);
    if (!Number.isFinite(milliseconds)) return null;
    return new Date(milliseconds + 8 * 60 * 60 * 1000).toISOString().slice(0, 10);
  }

  function validPath(value) {
    if (typeof value !== "string" || !value || value[0] === "/" || value.indexOf("\\") >= 0) return false;
    return value.split("/").every(function (part) {
      return part && part !== "." && part !== "..";
    });
  }

  function stableValue(value) {
    if (Array.isArray(value)) return value.map(stableValue);
    if (!isObject(value)) return value;
    var result = {};
    Object.keys(value).sort().forEach(function (key) {
      result[key] = stableValue(value[key]);
    });
    return result;
  }

  function semanticEqual(left, right) {
    try {
      return JSON.stringify(stableValue(left)) === JSON.stringify(stableValue(right));
    } catch (_error) {
      return false;
    }
  }

  function nodeContentSha256(value) {
    if (typeof require !== "function") return null;
    try {
      var crypto = require("crypto");
      var body = {};
      Object.keys(value).forEach(function (key) {
        if (key !== "content_sha256") body[key] = value[key];
      });
      return crypto.createHash("sha256")
        .update(JSON.stringify(stableValue(body)), "utf8")
        .digest("hex");
    } catch (_error) {
      return null;
    }
  }

  function validateSourceRef(value, label, errors) {
    if (!hasExactKeys(value, ["path", "sha256"])) {
      errors.push(label + " shape is invalid");
      return;
    }
    if (!validPath(value.path)) errors.push(label + ".path is invalid");
    if (!SHA256_RE.test(value.sha256 || "")) errors.push(label + ".sha256 is invalid");
  }

  function validateContinuity(value, generationLocalDate, errors) {
    var keys = [
      "status", "anchor_record_id", "anchor_data_date",
      "anchor_financial_state_sha256", "active_ledger_sha256",
      "holdings_valid_through", "financial_state_changed", "receipt_id",
      "receipt_content_sha256"
    ];
    if (!hasExactKeys(value, keys)) {
      errors.push("continuity_authority shape is invalid");
      return;
    }
    if (CONTINUITY_STATUSES.indexOf(value.status) < 0) errors.push("continuity_authority.status is invalid");
    if (!RECORD_RE.test(value.anchor_record_id || "")) errors.push("continuity anchor record is invalid");
    if (!validDate(value.anchor_data_date)) errors.push("continuity anchor date is invalid");
    if (!SHA256_RE.test(value.anchor_financial_state_sha256 || "")) errors.push("continuity financial-state SHA is invalid");
    if (!SHA256_RE.test(value.active_ledger_sha256 || "")) errors.push("continuity ledger SHA is invalid");
    if (!validDate(value.holdings_valid_through)) errors.push("holdings_valid_through is invalid");
    if (validDate(value.anchor_data_date) && validDate(value.holdings_valid_through) &&
        value.anchor_data_date > value.holdings_valid_through) {
      errors.push("holdings continuity cannot precede its anchor");
    }
    if (validDate(value.holdings_valid_through) && value.holdings_valid_through > generationLocalDate) {
      errors.push("holdings continuity cannot extend beyond the generation date");
    }
    if (value.status === "NO_ACTION_BOUND") {
      if (value.financial_state_changed !== false) errors.push("no-action continuity cannot change financial state");
      if (typeof value.receipt_id !== "string" || !value.receipt_id) errors.push("no-action receipt_id is invalid");
      if (!SHA256_RE.test(value.receipt_content_sha256 || "")) errors.push("no-action receipt SHA is invalid");
    } else if (value.status === "FINANCIAL_STATE_PUBLICATION") {
      if (value.financial_state_changed !== true) errors.push("financial publication must change financial state");
      if (value.receipt_id !== null || value.receipt_content_sha256 !== null) {
        errors.push("financial publication cannot use a no-action receipt");
      }
    } else if (value.status === "UNCONFIRMED") {
      if (value.financial_state_changed !== false || value.receipt_id !== null ||
          value.receipt_content_sha256 !== null ||
          value.holdings_valid_through !== value.anchor_data_date) {
        errors.push("unconfirmed continuity must stop at its unmodified anchor");
      }
    }
  }

  function validateFreshness(value, generatedAt, generationLocalDate, continuity, errors) {
    var keys = [
      "status", "scope", "mark_as_of", "generated_at", "valid_through",
      "source_kind", "reason"
    ];
    if (!hasExactKeys(value, keys)) {
      errors.push("freshness shape is invalid");
      return;
    }
    if (!isObject(continuity)) {
      errors.push("freshness continuity binding is unavailable");
      return;
    }
    if (FRESHNESS_STATUSES.indexOf(value.status) < 0) errors.push("freshness.status is invalid");
    if (value.scope !== FRESHNESS_SCOPE) errors.push("freshness.scope is invalid");
    if (!validDate(value.mark_as_of)) errors.push("freshness.mark_as_of is invalid");
    if (value.generated_at !== generatedAt) errors.push("freshness.generated_at is inconsistent");
    if (!validShanghaiDateTime(value.generated_at)) errors.push("freshness.generated_at is invalid");
    if (!validShanghaiDateTime(value.valid_through)) errors.push("freshness.valid_through is invalid");
    if (validShanghaiDateTime(value.generated_at) && validShanghaiDateTime(value.valid_through) &&
        Date.parse(value.valid_through) <= Date.parse(value.generated_at)) {
      errors.push("freshness.valid_through must follow generated_at");
    }
    if (value.source_kind !== STRICT_CLOSE) errors.push("freshness.source_kind is invalid");
    if (FRESHNESS_REASONS.indexOf(value.reason) < 0) errors.push("freshness.reason is invalid");
    if (validDate(value.mark_as_of) && value.mark_as_of > generationLocalDate) {
      errors.push("freshness mark cannot be in the future");
    }
    if (value.valid_through !== generationLocalDate + "T23:59:59+08:00") {
      errors.push("freshness.valid_through is not the local end of day");
    }
    if (value.status === "UPDATED") {
      if (continuity.holdings_valid_through !== generationLocalDate) {
        errors.push("UPDATED holdings continuity must reach the generation date");
      }
      var expectedReason = continuity.status === "NO_ACTION_BOUND"
        ? "CURRENT_DAILY_RECEIPT_AND_LATEST_LOCAL_CLOSE"
        : "CURRENT_FINANCIAL_PUBLICATION_AND_LATEST_LOCAL_CLOSE";
      if (value.reason !== expectedReason) errors.push("UPDATED freshness reason is inconsistent");
      if (continuity.status === "UNCONFIRMED") errors.push("unconfirmed continuity cannot be UPDATED");
    } else if (value.reason !== "DAILY_CONTINUITY_RECEIPT_MISSING") {
      errors.push("STALE freshness reason is inconsistent");
    }
  }

  function validateCompleteness(value, freshness, canonical, errors) {
    var keys = [
      "current_holdings", "current_absolute_performance", "canonical_history",
      "benchmark_relative", "benchmark_as_of", "legacy_caveats"
    ];
    if (!hasExactKeys(value, keys)) {
      errors.push("completeness shape is invalid");
      return;
    }
    if (!isObject(freshness)) {
      errors.push("completeness freshness binding is unavailable");
      return;
    }
    if (["COMPLETE", "STALE"].indexOf(value.current_holdings) < 0) errors.push("current holdings completeness is invalid");
    if (["COMPLETE", "STALE"].indexOf(value.current_absolute_performance) < 0) errors.push("absolute performance completeness is invalid");
    if (["COMPLETE", "PARTIAL"].indexOf(value.canonical_history) < 0) errors.push("canonical history completeness is invalid");
    if (["COMPLETE", "AS_OF_PRIOR_DATE"].indexOf(value.benchmark_relative) < 0) errors.push("benchmark completeness is invalid");
    if (!validDate(value.benchmark_as_of)) errors.push("benchmark_as_of is invalid");
    if (!Array.isArray(value.legacy_caveats) || value.legacy_caveats.some(function (item) {
      return typeof item !== "string" || !item;
    })) errors.push("legacy_caveats is invalid");
    if (Array.isArray(value.legacy_caveats) &&
        new Set(value.legacy_caveats).size !== value.legacy_caveats.length) {
      errors.push("legacy_caveats contain duplicates");
    }
    var expectedCurrent = freshness.status === "UPDATED" ? "COMPLETE" : "STALE";
    if (value.current_holdings !== expectedCurrent || value.current_absolute_performance !== expectedCurrent) {
      errors.push("current completeness is inconsistent with freshness");
    }
    var canonicalBenchmarkDates = new Set((canonical && canonical.benchmarks || []).map(function (row) {
      return row && row.end_date;
    }));
    if (canonicalBenchmarkDates.size !== 1 || !canonicalBenchmarkDates.has(value.benchmark_as_of)) {
      errors.push("benchmark_as_of must match the canonical benchmark closure");
    }
    if (validDate(value.benchmark_as_of) && validDate(freshness.mark_as_of)) {
      if (value.benchmark_as_of > freshness.mark_as_of) errors.push("benchmark_as_of cannot follow the holdings mark");
      var expectedBenchmark = value.benchmark_as_of < freshness.mark_as_of
        ? "AS_OF_PRIOR_DATE" : "COMPLETE";
      if (value.benchmark_relative !== expectedBenchmark) {
        errors.push("benchmark-relative completeness is inconsistent");
      }
    }
  }

  function maximumDrawdown(points, currentMarkedNav) {
    var peak = points.length && finite(points[0].adjusted_total_value)
      ? points[0].adjusted_total_value : 0;
    var deepest = 0;
    points.forEach(function (point) {
      peak = Math.max(peak, point.adjusted_total_value);
      deepest = Math.min(deepest, point.adjusted_total_value / peak - 1);
    });
    peak = Math.max(peak, currentMarkedNav);
    return Math.min(deepest, currentMarkedNav / peak - 1);
  }

  function validateResearchMark(value, canonical, continuity, freshness, completeness, errors) {
    var keys = [
      "status", "authority", "source_kind", "mark_date", "anchor_record_id",
      "base_ledger_sha256", "base_financial_state_sha256", "positions",
      "portfolio", "current_absolute_performance", "canonical_effect",
      "ledger_effect", "performance_effect", "paper_effect", "trade_effect"
    ];
    if (!hasExactKeys(value, keys)) {
      errors.push("research_mark shape is invalid");
      return;
    }
    if (!isObject(canonical) || !isObject(canonical.portfolio) ||
        !Array.isArray(canonical.portfolio.performance_points) ||
        !isObject(continuity) || !isObject(freshness) || !isObject(completeness)) {
      errors.push("research mark dependencies are invalid");
      return;
    }
    if (value.status !== "AVAILABLE") errors.push("research_mark.status is invalid");
    if (value.authority !== VIEW_AUTHORITY) errors.push("research_mark.authority is invalid");
    if (value.source_kind !== STRICT_CLOSE || value.source_kind !== freshness.source_kind) {
      errors.push("research_mark.source_kind is invalid");
    }
    if (value.mark_date !== freshness.mark_as_of) errors.push("research_mark.mark_date is inconsistent");
    if (value.anchor_record_id !== continuity.anchor_record_id) errors.push("research mark anchor is inconsistent");
    if (value.base_ledger_sha256 !== continuity.active_ledger_sha256) errors.push("research mark ledger binding is inconsistent");
    if (value.base_financial_state_sha256 !== continuity.anchor_financial_state_sha256) {
      errors.push("research mark financial-state binding is inconsistent");
    }
    EFFECT_KEYS.forEach(function (key) {
      if (value[key] !== "NONE") errors.push("research_mark." + key + " must be NONE");
    });

    var positionKeys = [
      "symbol", "name", "shares", "avg_cost", "cost_basis", "price",
      "price_date", "price_evidence_status", "market_value", "unrealized_pnl",
      "nav_weight", "equity_weight", "source_ref"
    ];
    var canonicalPositions = Array.isArray(canonical.positions) ? canonical.positions : [];
    if (!Array.isArray(value.positions) || value.positions.length !== canonicalPositions.length || !value.positions.length) {
      errors.push("research mark positions do not cover the canonical holdings");
      return;
    }
    var canonicalBySymbol = {};
    canonicalPositions.forEach(function (position) { canonicalBySymbol[position.symbol] = position; });
    var seen = {};
    var marketValue = 0;
    var unrealizedPnl = 0;
    value.positions.forEach(function (position, index) {
      var label = "research_mark.positions[" + index + "]";
      if (!hasExactKeys(position, positionKeys)) {
        errors.push(label + " shape is invalid");
        return;
      }
      if (!SYMBOL_RE.test(position.symbol || "") || seen[position.symbol]) {
        errors.push(label + " identity is invalid");
        return;
      }
      seen[position.symbol] = true;
      var base = canonicalBySymbol[position.symbol];
      if (!base || position.name !== base.name || !approximately(position.shares, base.shares, 0) ||
          !approximately(position.avg_cost, base.avg_cost, 1e-9) ||
          !approximately(position.cost_basis, base.cost_basis, 0.01)) {
        errors.push(label + " does not preserve the canonical holding");
      }
      ["shares", "avg_cost", "cost_basis", "price", "market_value", "unrealized_pnl", "nav_weight", "equity_weight"].forEach(function (key) {
        if (!finite(position[key])) errors.push(label + "." + key + " is invalid");
      });
      if ((finite(position.shares) && position.shares <= 0) ||
          (finite(position.price) && position.price <= 0)) {
        errors.push(label + " has invalid quantity or price");
      }
      if (!validDate(position.price_date) || position.price_date > value.mark_date) errors.push(label + ".price_date is invalid");
      if (POSITION_PRICE_EVIDENCE.indexOf(position.price_evidence_status) < 0) errors.push(label + ".price_evidence_status is invalid");
      if (position.price_evidence_status === "EXACT_CLOSE" && position.price_date !== value.mark_date) {
        errors.push(label + " strict-close date is inconsistent");
      }
      validateSourceRef(position.source_ref, label + ".source_ref", errors);
      if (!approximately(position.market_value, position.shares * position.price, 0.01)) {
        errors.push(label + " market value is inconsistent");
      }
      if (!approximately(position.cost_basis, position.shares * position.avg_cost, 0.01)) {
        errors.push(label + " cost basis is inconsistent");
      }
      if (!approximately(position.unrealized_pnl, position.market_value - position.cost_basis, 0.01)) {
        errors.push(label + " unrealized P&L is inconsistent");
      }
      marketValue += position.market_value;
      unrealizedPnl += position.unrealized_pnl;
    });

    var portfolioKeys = [
      "cash", "market_value", "nav", "unrealized_pnl", "cash_weight",
      "gross_exposure"
    ];
    var portfolio = value.portfolio;
    if (!hasExactKeys(portfolio, portfolioKeys)) {
      errors.push("research_mark.portfolio shape is invalid");
      return;
    }
    portfolioKeys.forEach(function (key) {
      if (!finite(portfolio[key])) errors.push("research_mark.portfolio." + key + " is invalid");
    });
    if (!approximately(portfolio.cash, canonical.portfolio.cash, 0.01)) errors.push("research mark cash changed");
    if (!approximately(portfolio.market_value, marketValue, 0.01)) errors.push("research mark market value is inconsistent");
    if (!approximately(portfolio.nav, portfolio.cash + portfolio.market_value, 0.01)) errors.push("research mark NAV is inconsistent");
    if (!approximately(portfolio.unrealized_pnl, unrealizedPnl, 0.01)) errors.push("research mark unrealized P&L is inconsistent");
    if (finite(portfolio.nav) && portfolio.nav > 0) {
      if (!approximately(portfolio.cash_weight, portfolio.cash / portfolio.nav, 1e-9) ||
          !approximately(portfolio.gross_exposure, portfolio.market_value / portfolio.nav, 1e-9)) {
        errors.push("research mark portfolio weights are inconsistent");
      }
      value.positions.forEach(function (position, index) {
        if (!approximately(position.nav_weight, position.market_value / portfolio.nav, 1e-9)) {
          errors.push("research_mark.positions[" + index + "].nav_weight is inconsistent");
        }
        var expectedEquityWeight = portfolio.market_value > 0
          ? position.market_value / portfolio.market_value : 0;
        if (!approximately(position.equity_weight, expectedEquityWeight, 1e-9)) {
          errors.push("research_mark.positions[" + index + "].equity_weight is inconsistent");
        }
      });
    } else {
      errors.push("research mark NAV must be positive");
    }

    var performanceKeys = [
      "point_date", "anchor_date", "marked_nav", "initial_capital",
      "cumulative_return", "continuity_interval_return", "max_drawdown",
      "evidence_status", "authority"
    ];
    var performance = value.current_absolute_performance;
    if (!hasExactKeys(performance, performanceKeys)) {
      errors.push("current_absolute_performance shape is invalid");
      return;
    }
    var points = canonical.portfolio.performance_points;
    var lastPoint = points[points.length - 1];
    if (canonical.portfolio.return_method !== "initial_capital_return_excluding_external_flows" ||
        !approximately(canonical.portfolio.performance_initial_capital, 1000000, 0.01) ||
        !approximately(canonical.portfolio.excluded_external_flow, 0, 0.01) ||
        !approximately(performance.initial_capital, 1000000, 0.01)) {
      errors.push("view-only continuity requires the fixed initial-capital path");
    }
    if (performance.point_date !== value.mark_date ||
        performance.anchor_date !== lastPoint.date) {
      errors.push("current absolute performance dates are inconsistent");
    }
    if (!approximately(performance.marked_nav, portfolio.nav, 0.01)) errors.push("current absolute performance NAV is inconsistent");
    if (!approximately(performance.cumulative_return, portfolio.nav / 1000000 - 1, 1e-9)) {
      errors.push("current absolute cumulative return is inconsistent");
    }
    if (!approximately(performance.continuity_interval_return, portfolio.nav / lastPoint.adjusted_total_value - 1, 1e-9)) {
      errors.push("current absolute continuity return is inconsistent");
    }
    var expectedDrawdown = maximumDrawdown(points, portfolio.nav);
    if (!approximately(performance.max_drawdown, expectedDrawdown, 1e-9)) errors.push("current absolute max drawdown is inconsistent");
    if (performance.evidence_status !== "HASH_BOUND_CONTINUITY_MARK") errors.push("current absolute evidence status is invalid");
    if (performance.authority !== VIEW_AUTHORITY) errors.push("current absolute authority is invalid");
  }

  function validateBundle(value) {
    var errors = [];
    var topKeys = [
      "schema_version", "publication_attempt_id", "generated_at",
      "generation_local_date", "canonical_v1", "canonical_v1_ref", "integrity",
      "continuity_authority", "freshness", "completeness", "research_mark",
      "source_refs", "content_sha256"
    ];
    if (!hasExactKeys(value, topKeys)) return { valid: false, errors: ["v2 bundle shape is invalid"] };
    if (value.schema_version !== SCHEMA_VERSION) errors.push("schema_version is invalid");
    if (!ATTEMPT_RE.test(value.publication_attempt_id || "")) errors.push("publication_attempt_id is invalid");
    if (!validShanghaiDateTime(value.generated_at)) errors.push("generated_at is invalid");
    if (!validDate(value.generation_local_date) || shanghaiDate(value.generated_at) !== value.generation_local_date) {
      errors.push("generation_local_date is inconsistent");
    }
    if (!V1Contract || typeof V1Contract.validateBundle !== "function") {
      errors.push("v1 contract is unavailable");
    } else {
      var v1Validation = V1Contract.validateBundle(value.canonical_v1);
      if (!v1Validation.valid) errors.push("canonical_v1 is invalid: " + v1Validation.errors.join("; "));
    }
    validateSourceRef(value.canonical_v1_ref, "canonical_v1_ref", errors);
    if (!hasExactKeys(value.integrity, ["status"]) || value.integrity.status !== "VERIFIED") {
      errors.push("integrity status is invalid");
    }
    validateContinuity(value.continuity_authority, value.generation_local_date, errors);
    if (isObject(value.canonical_v1) && isObject(value.continuity_authority)) {
      if (value.continuity_authority.anchor_record_id !== value.canonical_v1.latest_valid_record ||
          value.continuity_authority.anchor_data_date !== value.canonical_v1.latest_data_date) {
        errors.push("continuity anchor does not match canonical_v1");
      }
      if (!isObject(value.canonical_v1.current_evidence) ||
          value.continuity_authority.anchor_financial_state_sha256 !== value.canonical_v1.current_evidence.financial_state_sha256 ||
          value.continuity_authority.active_ledger_sha256 !== value.canonical_v1.current_evidence.ledger_sha256) {
        errors.push("continuity closure does not match canonical_v1");
      }
    }
    validateFreshness(value.freshness, value.generated_at, value.generation_local_date, value.continuity_authority, errors);
    validateCompleteness(value.completeness, value.freshness, value.canonical_v1, errors);
    validateResearchMark(
      value.research_mark,
      value.canonical_v1,
      value.continuity_authority,
      value.freshness,
      value.completeness,
      errors
    );
    if (!Array.isArray(value.source_refs) || !value.source_refs.length) {
      errors.push("source_refs are missing");
    } else {
      var seen = {};
      value.source_refs.forEach(function (sourceRef, index) {
        validateSourceRef(sourceRef, "source_refs[" + index + "]", errors);
        if (isObject(sourceRef)) {
          var identity = sourceRef.path;
          if (seen[identity]) errors.push("source_refs contain a duplicate");
          seen[identity] = true;
        }
      });
      if (!value.source_refs.some(function (sourceRef) {
        return semanticEqual(sourceRef, value.canonical_v1_ref);
      })) errors.push("canonical_v1_ref is not source-bound");
      if (isObject(value.research_mark) && Array.isArray(value.research_mark.positions)) {
        value.research_mark.positions.forEach(function (position, index) {
          if (isObject(position) && !value.source_refs.some(function (sourceRef) {
            return semanticEqual(sourceRef, position.source_ref);
          })) errors.push("research_mark.positions[" + index + "].source_ref is not source-bound");
        });
      }
      var sortedPaths = value.source_refs.map(function (sourceRef) {
        return isObject(sourceRef) ? sourceRef.path : "";
      }).slice().sort();
      var actualPaths = value.source_refs.map(function (sourceRef) {
        return isObject(sourceRef) ? sourceRef.path : "";
      });
      if (!semanticEqual(actualPaths, sortedPaths)) errors.push("source_refs are not sorted");
    }
    if (!SHA256_RE.test(value.content_sha256 || "")) errors.push("content_sha256 is invalid");
    return { valid: errors.length === 0, errors: errors };
  }

  function validateSelector(value) {
    var errors = [];
    var keys = [
      "schema_version", "attempt_id", "status", "updated_at",
      "v2_content_sha256", "reason", "content_sha256"
    ];
    if (!hasExactKeys(value, keys)) return { valid: false, errors: ["selector shape is invalid"] };
    if (value.schema_version !== SELECTOR_SCHEMA_VERSION) errors.push("selector schema_version is invalid");
    if (!ATTEMPT_RE.test(value.attempt_id || "")) errors.push("selector attempt_id is invalid");
    if (["REFRESHING", "UPDATED", "BLOCKED"].indexOf(value.status) < 0) errors.push("selector status is invalid");
    if (!validShanghaiDateTime(value.updated_at)) errors.push("selector updated_at is invalid");
    if (typeof value.reason !== "string" || !value.reason) errors.push("selector reason is invalid");
    if (value.status === "UPDATED") {
      if (!SHA256_RE.test(value.v2_content_sha256 || "")) errors.push("UPDATED selector v2 hash is invalid");
    } else if (value.v2_content_sha256 !== null) {
      errors.push("non-UPDATED selector cannot select a v2 hash");
    }
    if (!SHA256_RE.test(value.content_sha256 || "")) {
      errors.push("selector content_sha256 is invalid");
    } else {
      var observed = nodeContentSha256(value);
      if (observed !== null && observed !== value.content_sha256) errors.push("selector content_sha256 mismatch");
    }
    return { valid: errors.length === 0, errors: errors };
  }

  function canonicalSnapshot(actualV1) {
    if (!V1Contract || typeof V1Contract.deriveSnapshot !== "function") return null;
    var snapshot = V1Contract.deriveSnapshot(actualV1);
    return snapshot.status === "BLOCKED" ? null : snapshot;
  }

  function emptyLabels(kind) {
    return {
      holdings_label: "持仓估值 " + kind + " · 无可用严格收盘",
      absolute_performance_label: "组合绝对业绩 " + kind + " · 无可用视图",
      anchor_label: "财务状态锚点 " + kind,
      benchmark_label: "基准相对业绩 " + kind
    };
  }

  function statusObject(kind) {
    return {
      integrity: kind,
      freshness: kind,
      current_holdings: kind,
      current_absolute_performance: kind,
      canonical_history: kind,
      benchmark_relative: kind
    };
  }

  function failedSnapshot(kind, blocker, actualV1, labels, status) {
    var base = labels || emptyLabels(kind);
    return {
      schema_version: SCHEMA_VERSION,
      status: status || statusObject(kind),
      bundle: null,
      canonical: canonicalSnapshot(actualV1),
      blockers: [blocker],
      holdings_label: base.holdings_label,
      absolute_performance_label: base.absolute_performance_label,
      anchor_label: base.anchor_label,
      benchmark_label: base.benchmark_label
    };
  }

  function labelsFor(value, stale) {
    var mark = value.freshness.mark_as_of;
    var anchor = value.continuity_authority.anchor_data_date;
    var valid = value.continuity_authority.holdings_valid_through;
    var benchmark = value.completeness.benchmark_as_of;
    var continuityToken = value.continuity_authority.status === "NO_ACTION_BOUND"
      ? "NO_ACTION" : value.continuity_authority.status;
    return {
      holdings_label: "持仓估值 " + (stale ? "STALE" : "UPDATED") + " · 最新可用严格收盘 " + mark,
      absolute_performance_label: "组合绝对业绩 " + (stale ? "STALE" : "VIEW-UPDATED") + " · 截至 " + mark,
      anchor_label: "财务状态锚点 " + anchor + " · " + continuityToken + " 连续有效至 " + valid,
      benchmark_label: "基准相对业绩 · 截至 " + benchmark
    };
  }

  function nextFreshnessRecheckDelay(value, now) {
    if (!value || !value.freshness) return null;
    var nowMilliseconds = now === undefined || now === null ? Date.now() :
      (now instanceof Date ? now.getTime() : Date.parse(now));
    var validThroughMilliseconds = Date.parse(value.freshness.valid_through);
    if (!Number.isFinite(nowMilliseconds) || !Number.isFinite(validThroughMilliseconds)) {
      return null;
    }
    return Math.max(1, validThroughMilliseconds - nowMilliseconds + 1);
  }

  function deriveSnapshot(value, selector, actualV1, now) {
    var canonical = canonicalSnapshot(actualV1);
    if (!canonical) return failedSnapshot("BLOCKED", "canonical_v1_invalid_or_missing", actualV1);
    if (value === null || value === undefined || selector === null || selector === undefined) {
      return failedSnapshot("UNAVAILABLE", "dashboard_v2_or_selector_missing", actualV1);
    }
    var selectorValidation = validateSelector(selector);
    if (!selectorValidation.valid) {
      return failedSnapshot("BLOCKED", "dashboard_v2_selector_invalid: " + selectorValidation.errors.join("; "), actualV1);
    }
    if (selector.status === "BLOCKED") {
      return failedSnapshot("BLOCKED", "dashboard_v2_selector_blocked: " + selector.reason, actualV1);
    }
    if (selector.status === "REFRESHING") {
      var refreshingStatus = statusObject("STALE");
      refreshingStatus.integrity = "VERIFIED";
      return failedSnapshot("STALE", "dashboard_v2_selector_refreshing: " + selector.reason, actualV1, null, refreshingStatus);
    }
    var validation = validateBundle(value);
    if (!validation.valid) {
      return failedSnapshot("BLOCKED", "dashboard_v2_invalid: " + validation.errors.join("; "), actualV1);
    }
    if (selector.attempt_id !== value.publication_attempt_id) {
      return failedSnapshot("BLOCKED", "dashboard_v2_attempt_id_mismatch", actualV1);
    }
    if (selector.v2_content_sha256 !== value.content_sha256) {
      return failedSnapshot("BLOCKED", "dashboard_v2_content_sha256_mismatch", actualV1);
    }
    if (Date.parse(selector.updated_at) < Date.parse(value.generated_at) ||
        shanghaiDate(selector.updated_at) !== value.generation_local_date) {
      return failedSnapshot("BLOCKED", "dashboard_v2_selector_time_mismatch", actualV1);
    }
    if (!semanticEqual(value.canonical_v1, actualV1)) {
      return failedSnapshot("BLOCKED", "dashboard_v2_nested_canonical_v1_mismatch", actualV1);
    }
    var nowMilliseconds = now === undefined || now === null ? Date.now() :
      (now instanceof Date ? now.getTime() : Date.parse(now));
    if (!Number.isFinite(nowMilliseconds)) {
      return failedSnapshot("BLOCKED", "dashboard_v2_now_invalid", actualV1);
    }
    if (nowMilliseconds < Date.parse(value.generated_at)) {
      return failedSnapshot("BLOCKED", "dashboard_v2_now_precedes_generation", actualV1);
    }
    var expired = nowMilliseconds > Date.parse(value.freshness.valid_through);
    var stale = expired || value.freshness.status === "STALE";
    var labels = labelsFor(value, stale);
    if (expired) {
      var expiredStatus = {
        integrity: "VERIFIED",
        freshness: "STALE",
        current_holdings: "STALE",
        current_absolute_performance: "STALE",
        canonical_history: value.completeness.canonical_history,
        benchmark_relative: value.completeness.benchmark_relative
      };
      return failedSnapshot("STALE", "dashboard_v2_expired", actualV1, labels, expiredStatus);
    }
    return {
      schema_version: SCHEMA_VERSION,
      status: {
        integrity: "VERIFIED",
        freshness: value.freshness.status,
        current_holdings: value.completeness.current_holdings,
        current_absolute_performance: value.completeness.current_absolute_performance,
        canonical_history: value.completeness.canonical_history,
        benchmark_relative: value.completeness.benchmark_relative
      },
      bundle: value,
      canonical: canonical,
      blockers: stale ? ["dashboard_v2_freshness_stale: " + value.freshness.reason] : [],
      holdings_label: labels.holdings_label,
      absolute_performance_label: labels.absolute_performance_label,
      anchor_label: labels.anchor_label,
      benchmark_label: labels.benchmark_label
    };
  }

  return {
    SCHEMA_VERSION: SCHEMA_VERSION,
    SELECTOR_SCHEMA_VERSION: SELECTOR_SCHEMA_VERSION,
    VIEW_AUTHORITY: VIEW_AUTHORITY,
    deriveSnapshot: deriveSnapshot,
    nextFreshnessRecheckDelay: nextFreshnessRecheckDelay,
    nodeContentSha256: nodeContentSha256,
    semanticEqual: semanticEqual,
    validateBundle: validateBundle,
    validateSelector: validateSelector
  };
});
