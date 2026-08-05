(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.DashboardContractV5 = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  var INPUT_SCHEMA_ID = "myquant.v17.v4.mainline-public-run.v1";
  var DASHBOARD_SCHEMA_VERSION = "dashboard_contract.v5";
  var REF_KEYS = ["byte_sha256", "relative_path", "schema_id"];
  var AUTHORITY_KEYS = [
    "broker_calls",
    "execution_calls",
    "llm_control_calls",
    "order_calls",
    "provider_calls",
    "selector_writes",
    "trade_calls"
  ];
  var TARGET_KEYS = ["current_target", "final_target", "lane", "symbol"];
  var RUN_KEYS = [
    "active_pointer_ref",
    "authority_flags",
    "authority_source",
    "canonical_strategy_id",
    "capability",
    "cash_weight",
    "fallback_used",
    "formal_output_ref",
    "gross_weight",
    "mainline_run_ref",
    "market",
    "portfolio_output_ref",
    "protocol",
    "read_only",
    "run_id",
    "schema_id",
    "selector_used",
    "semantic_sha256",
    "source_closure_ref",
    "state",
    "targets"
  ];
  var SHA256_RE = /^[0-9a-f]{64}$/;
  var IDENTIFIER_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
  var SYMBOL_RE = /^[0-9]{6}\.(?:SH|SZ|BJ)$/;
  var REF_SCHEMA_RE = /^myquant\.v17\.v4\.[a-z0-9-]+\.v[0-9]+$/;
  var RELATIVE_PATH_RE = /^[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*$/;
  var UNIT_DECIMAL_RE = /^(?:0(?:\.[0-9]+)?|1(?:\.0+)?)$/;

  function isObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function sameKeys(value, expected) {
    if (!isObject(value)) return false;
    var actual = Object.keys(value).sort();
    var wanted = expected.slice().sort();
    return actual.length === wanted.length && actual.every(function (key, index) {
      return key === wanted[index];
    });
  }

  function validateRef(value, label, expectedSchemaId, requiredPrefix, errors) {
    if (!sameKeys(value, REF_KEYS)) {
      errors.push(label + " must contain only schema_id, relative_path and byte_sha256");
      return;
    }
    if (!REF_SCHEMA_RE.test(value.schema_id) || value.schema_id !== expectedSchemaId) {
      errors.push(label + ".schema_id is invalid");
    }
    if (!RELATIVE_PATH_RE.test(value.relative_path)) errors.push(label + ".relative_path is invalid");
    if (requiredPrefix && value.relative_path.indexOf(requiredPrefix) !== 0) {
      errors.push(label + ".relative_path is outside its governed root");
    }
    if (!SHA256_RE.test(value.byte_sha256)) errors.push(label + ".byte_sha256 is invalid");
  }

  function validateTarget(value, index, previousSymbol, errors) {
    var label = "targets[" + index + "]";
    if (!sameKeys(value, TARGET_KEYS)) {
      errors.push(label + " has an invalid shape");
      return previousSymbol;
    }
    if (!SYMBOL_RE.test(value.symbol)) errors.push(label + ".symbol is invalid");
    if (previousSymbol !== null && value.symbol <= previousSymbol) {
      errors.push("targets must be unique and sorted by symbol");
    }
    if (["SELECTION_POOL", "REVIEW_ONLY_HOLDING"].indexOf(value.lane) < 0) {
      errors.push(label + ".lane is invalid");
    }
    if (!UNIT_DECIMAL_RE.test(value.current_target)) errors.push(label + ".current_target is invalid");
    if (!UNIT_DECIMAL_RE.test(value.final_target)) errors.push(label + ".final_target is invalid");
    return value.symbol;
  }

  function validatePublicRun(value) {
    var errors = [];
    if (!sameKeys(value, RUN_KEYS)) {
      return { valid: false, errors: ["public run does not match the exact mainline DTO"] };
    }
    if (value.schema_id !== INPUT_SCHEMA_ID) errors.push("schema_id must be " + INPUT_SCHEMA_ID);
    if (value.protocol !== "myquant.v17.v4") errors.push("protocol is invalid");
    if (!IDENTIFIER_RE.test(value.canonical_strategy_id) || value.canonical_strategy_id.length > 80) {
      errors.push("canonical_strategy_id is invalid");
    }
    if (!IDENTIFIER_RE.test(value.run_id) || value.run_id.length > 80) errors.push("run_id is invalid");
    if (value.state !== "ACTIVE") errors.push("state must be ACTIVE");
    if (value.market !== "CN_A_SHARE") errors.push("market is invalid");
    if (value.capability !== "RESEARCH_PORTFOLIO") errors.push("capability is invalid");
    if (value.authority_source !== "FORMAL_V17_V4") errors.push("authority_source is invalid");
    if (value.read_only !== true) errors.push("read_only must be true");
    if (value.selector_used !== false) errors.push("selector_used must be false");
    if (value.fallback_used !== false) errors.push("fallback_used must be false");
    if (!sameKeys(value.authority_flags, AUTHORITY_KEYS)) {
      errors.push("authority_flags has an invalid shape");
    } else {
      AUTHORITY_KEYS.forEach(function (key) {
        if (value.authority_flags[key] !== false) errors.push("authority_flags." + key + " must be false");
      });
    }
    [
      ["active_pointer_ref", "myquant.v17.v4.mainline-active-pointer.v1", "results/v17_mainline/strategies/"],
      ["mainline_run_ref", "myquant.v17.v4.mainline-run.v1", "results/v17_mainline/strategies/"],
      ["formal_output_ref", "myquant.v17.v4.formal-output.v1", "results/v17_v4_formal_research/"],
      ["portfolio_output_ref", "myquant.v17.v4.portfolio-output.v1", "results/v17_v4_formal_research/"],
      ["source_closure_ref", "myquant.v17.v4.pit-generation-catalog.v1", "data/private/v17_v4_sources/"]
    ].forEach(function (item) {
      validateRef(value[item[0]], item[0], item[1], item[2], errors);
    });
    if (isObject(value.active_pointer_ref)) {
      var expectedPointerPath = "results/v17_mainline/strategies/" + value.canonical_strategy_id + "/_active.json";
      if (value.active_pointer_ref.relative_path !== expectedPointerPath) {
        errors.push("active_pointer_ref.relative_path does not match canonical_strategy_id");
      }
    }
    if (isObject(value.mainline_run_ref)) {
      var expectedRunPath = "results/v17_mainline/strategies/" + value.canonical_strategy_id +
        "/runs/" + value.run_id + "/run.json";
      if (value.mainline_run_ref.relative_path !== expectedRunPath) {
        errors.push("mainline_run_ref.relative_path does not match run identity");
      }
    }
    if (!UNIT_DECIMAL_RE.test(value.cash_weight)) errors.push("cash_weight is invalid");
    if (!UNIT_DECIMAL_RE.test(value.gross_weight)) errors.push("gross_weight is invalid");
    if (!Array.isArray(value.targets) || value.targets.length < 1 || value.targets.length > 524) {
      errors.push("targets must contain between 1 and 524 rows");
    } else {
      var previousSymbol = null;
      value.targets.forEach(function (target, index) {
        previousSymbol = validateTarget(target, index, previousSymbol, errors);
      });
    }
    if (!SHA256_RE.test(value.semantic_sha256)) errors.push("semantic_sha256 is invalid");
    return { valid: errors.length === 0, errors: errors };
  }

  function unavailableSnapshot(runtimeState, blocker) {
    return {
      schema_version: DASHBOARD_SCHEMA_VERSION,
      input_schema_id: INPUT_SCHEMA_ID,
      runtime_state: runtimeState,
      strategy_id: null,
      blockers: [blocker],
      active_pointer_ref: null,
      public_run: null
    };
  }

  function deriveSnapshot(publicRun) {
    if (publicRun === undefined || publicRun === null) {
      return unavailableSnapshot("V17_MAINLINE_UNINITIALIZED", "mainline_active_pointer_missing");
    }
    var validation = validatePublicRun(publicRun);
    if (!validation.valid) {
      return unavailableSnapshot("BLOCKED", "mainline_public_run_invalid: " + validation.errors.join("; "));
    }
    return {
      schema_version: DASHBOARD_SCHEMA_VERSION,
      input_schema_id: INPUT_SCHEMA_ID,
      runtime_state: "ACTIVE",
      strategy_id: publicRun.canonical_strategy_id,
      blockers: [],
      active_pointer_ref: publicRun.active_pointer_ref,
      public_run: publicRun
    };
  }

  return {
    INPUT_SCHEMA_ID: INPUT_SCHEMA_ID,
    DASHBOARD_SCHEMA_VERSION: DASHBOARD_SCHEMA_VERSION,
    deriveSnapshot: deriveSnapshot,
    validatePublicRun: validatePublicRun
  };
});
