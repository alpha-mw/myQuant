(function (root, factory) {
  "use strict";
  var v2 = typeof module === "object" && module.exports
    ? require("./cn_aggressive_dashboard_contract_v2.js") : root.CNAggressiveDashboardContractV2;
  var api = factory(v2);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CNDailyDashboardContract = api;
})(typeof window !== "undefined" ? window : globalThis, function (V2) {
  "use strict";
  var SHA = /^[0-9a-f]{64}$/;
  var DOMAINS = ["decision", "factor", "store", "theme", "top100"];
  var STATES = ["THESIS_INVALIDATED", "INSUFFICIENT_EVIDENCE", "WATCHLIST", "RESEARCH_APPROVED", "PAPER_CANDIDATE"];
  var HEAD = ["schema_version", "trade_date", "completion_ref", "previous_head_sha256", "registered_at", "authority", "content_sha256"];
  var SELECTOR = ["schema_version", "attempt_id", "status", "updated_at", "v2_content_sha256", "reason", "content_sha256", "trade_date", "completion_ref", "completed_head_sha256", "v1_byte_sha256", "v2_byte_sha256", "daily_evidence_sha256"];
  var VIEW = ["schema_version", "view_designation", "trade_date", "research_cutoff", "completion_ref", "completed_head_ref", "financial_refs", "daily_evidence_ref", "evidence", "native_valid_through", "authority", "content_sha256"];
  var REGISTERED_VIEW = ["registered_transition_ref", "registered_transition", "registered_close_summary"];
  var CLOSE_SUMMARY = ["source_profile", "store_plan_ref", "decision_baseline_pointer_ref", "writer_pointer_ref", "final_pointer_ref", "writer_record_id", "final_record_id", "official_valuation", "close_writer_trade_count", "close_writer_order_count", "close_writer_fill_count"];
  var POSITION_ROW = ["symbol", "baseline_position_state", "writer_position_state", "shares_before", "shares_after", "shares_delta", "avg_cost_before", "avg_cost_after", "cost_basis_before", "cost_basis_after", "cost_basis_delta", "change_kind", "fact_refs", "policy_revalidation_required", "risk_execution_state", "blocker_codes"];

  function fail(code) { var error = new Error(code); error.code = code; throw error; }
  function keys(value, expected) {
    if (!value || Array.isArray(value) || typeof value !== "object" ||
        Object.keys(value).sort().join("|") !== expected.slice().sort().join("|")) fail("DASHBOARD_SHAPE_INVALID");
  }
  function stable(value) {
    if (Array.isArray(value)) return value.map(stable);
    if (value && typeof value === "object") {
      var result = {};
      Object.keys(value).sort().forEach(function (key) { result[key] = stable(value[key]); });
      return result;
    }
    return value;
  }
  function equal(a, b) { return JSON.stringify(stable(a)) === JSON.stringify(stable(b)); }
  function ref(value) {
    keys(value, ["path", "sha256"]);
    if (typeof value.path !== "string" || !value.path || value.path[0] === "/" ||
        value.path.split("/").some(function (x) { return !x || x === "." || x === ".."; }) ||
        !SHA.test(value.sha256)) fail("DASHBOARD_REF_INVALID");
  }
  function authority(value) {
    keys(value, ["broker", "order", "trade", "actual_holdings_mutation", "system", "mainline"]);
    if (Object.keys(value).some(function (key) { return value[key] !== false; })) fail("DASHBOARD_AUTHORITY_INVALID");
  }
  async function sha(raw) {
    if (typeof raw !== "string") fail("DASHBOARD_RAW_BYTES_MISSING");
    var crypto = typeof globalThis !== "undefined" ? globalThis.crypto : null;
    if ((!crypto || !crypto.subtle) && typeof require === "function") crypto = require("crypto").webcrypto;
    if (!crypto || !crypto.subtle || typeof TextEncoder !== "function") fail("DASHBOARD_CRYPTO_UNAVAILABLE");
    var result = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(raw));
    return Array.from(new Uint8Array(result)).map(function (v) { return v.toString(16).padStart(2, "0"); }).join("");
  }
  async function document(raw, expectedKeys, schema) {
    var value;
    try { value = JSON.parse(raw); } catch (_) { fail("DASHBOARD_JSON_INVALID"); }
    keys(value, expectedKeys);
    if (value.schema_version !== schema || JSON.stringify(stable(value)) !== raw) fail("DASHBOARD_CANONICAL_BYTES_INVALID");
    var body = Object.assign({}, value); delete body.content_sha256;
    if (!SHA.test(value.content_sha256) || await sha(JSON.stringify(stable(body))) !== value.content_sha256) fail("DASHBOARD_CONTENT_SHA_MISMATCH");
    return value;
  }
  function validDay(value) {
    if (typeof value !== "string" || !/^\d{8}$/.test(value)) return false;
    var iso=value.slice(0,4)+"-"+value.slice(4,6)+"-"+value.slice(6,8);
    var date=new Date(iso+"T00:00:00Z");
    return Number.isFinite(date.getTime()) && date.toISOString().slice(0,10) === iso;
  }
  function validTime(value) { return typeof value === "string" && /(?:Z|[+-]\d\d:\d\d)$/.test(value) && Number.isFinite(Date.parse(value)); }

  function decimal(value) {
    if (typeof value !== "string" || !/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(value) || !Number.isFinite(Number(value))) fail("DASHBOARD_REGISTERED_NUMBER_INVALID");
    return Number(value);
  }

  async function registeredView(view, v1, evidence) {
    var artifact=view.registered_transition, summary=view.registered_close_summary;
    ref(view.registered_transition_ref); keys(summary,CLOSE_SUMMARY);
    if (!artifact || artifact.kind !== "registered_financial_transition_reconciliation" || await sha(JSON.stringify(stable(artifact))) !== view.registered_transition_ref.sha256) fail("DASHBOARD_REGISTERED_REPORT_SHA_INVALID");
    var body=artifact.payload;
    if (!body || body.trade_date !== view.trade_date || body.as_of !== view.research_cutoff || body.source_profile !== "OWNER_DECLARED_BUYS_V1" || body.evidence_level !== "OWNER_DECLARED" || body.broker_statement_verified !== false || body.financial_admission_state !== "VALIDATED_REGISTERED_TRANSITION" || body.risk_readiness_state !== "OWNER_POLICY_REVALIDATION_REQUIRED" || body.prospective !== false || body.research_only !== true || body.production !== false || body.run_state !== "INACTIVE") fail("DASHBOARD_REGISTERED_REPORT_SCOPE_INVALID");
    keys(body.authority,["broker","execution","factor_governance_write","llm_control","mainline_activation","order","portfolio_activation","provider","selector_write","trade"]);
    if (Object.keys(body.authority).some(function(k){return body.authority[k]!==false;})) fail("DASHBOARD_REGISTERED_AUTHORITY_INVALID");
    ["store_plan_ref","decision_baseline_pointer_ref","writer_pointer_ref","final_pointer_ref"].forEach(function(k){ref(summary[k]);});
    ["decision_baseline_pointer_ref","writer_pointer_ref","registered_event_declaration_ref","owner_fact_ref"].forEach(function(k){ref(body[k]);});
    if (summary.source_profile !== body.source_profile || !equal(summary.decision_baseline_pointer_ref,body.decision_baseline_pointer_ref) || !equal(summary.writer_pointer_ref,body.writer_pointer_ref) || !equal(summary.final_pointer_ref,evidence.source_bindings.store.output_refs.pointer) || summary.writer_record_id !== body.writer_record_id || summary.final_record_id === summary.writer_record_id || summary.final_record_id !== v1.latest_valid_record || summary.official_valuation !== true || summary.close_writer_trade_count !== 0 || summary.close_writer_order_count !== 0 || summary.close_writer_fill_count !== 0) fail("DASHBOARD_REGISTERED_CLOSE_BINDING_INVALID");
    if (!/\/plan\.v2\.json$/.test(summary.store_plan_ref.path) || summary.store_plan_ref.path.replace(/plan\.v2\.json$/, "committed-pointer.v1.json") !== summary.final_pointer_ref.path) fail("DASHBOARD_REGISTERED_PLAN_PATH_INVALID");
    if (!Array.isArray(body.source_refs) || !body.source_refs.some(function(r){return equal(r,summary.store_plan_ref);})) fail("DASHBOARD_REGISTERED_PLAN_SOURCE_MISSING");
    body.source_refs.forEach(ref);
    ["owner_declared_at","registered_at","custody_at"].forEach(function(k){if(!validTime(body[k]))fail("DASHBOARD_REGISTERED_TIME_INVALID");});
    if (Date.parse(body.owner_declared_at)>Date.parse(body.registered_at) || Date.parse(body.registered_at)>Date.parse(body.custody_at) || artifact.created_at!==body.custody_at) fail("DASHBOARD_REGISTERED_CUSTODY_INVALID");
    if (decimal(body.cash_after_cny)!==v1.portfolio.cash || decimal(body.cash_delta_cny)>=0) fail("DASHBOARD_REGISTERED_CASH_MISMATCH");
    decimal(body.cash_before_cny);
    if (!Array.isArray(body.position_rows) || !Array.isArray(v1.positions)) fail("DASHBOARD_REGISTERED_POSITIONS_INVALID");
    var positions={},seen={},changed=[];
    v1.positions.forEach(function(p){positions[p.symbol]=p;});
    body.position_rows.forEach(function(row){
      keys(row,POSITION_ROW);
      var current=positions[row.symbol];
      if (!current || seen[row.symbol] || row.writer_position_state!=="PRESENT") fail("DASHBOARD_REGISTERED_POSITION_UNION_MISMATCH");
      seen[row.symbol]=true;
      ["shares","avg_cost","cost_basis"].forEach(function(k){if(decimal(row[k+"_after"])!==current[k])fail("DASHBOARD_REGISTERED_FINAL_POSITION_MISMATCH");});
      var delta=decimal(row.shares_delta),qty=decimal(row.shares_before),cost=decimal(row.cost_basis_before);
      decimal(row.cost_basis_delta);
      if(!Number.isSafeInteger(qty) || !Number.isSafeInteger(Number(row.shares_after)) || !Number.isSafeInteger(delta))fail("DASHBOARD_REGISTERED_QUANTITY_INVALID");
      if (!Array.isArray(row.fact_refs) || !Array.isArray(row.blocker_codes)) fail("DASHBOARD_REGISTERED_POSITION_EVIDENCE_INVALID");
      row.fact_refs.forEach(function(r){ref(r);if(!body.source_refs.some(function(s){return equal(r,s);}))fail("DASHBOARD_REGISTERED_FACT_REF_UNBOUND");});
      if (row.change_kind==="UNCHANGED") {
        if(delta!==0 || qty!==current.shares || cost!==current.cost_basis || decimal(row.avg_cost_before)!==current.avg_cost || row.baseline_position_state!=="PRESENT" || row.policy_revalidation_required!==false || row.risk_execution_state!=="NOT_EVALUATED" || row.fact_refs.length || row.blocker_codes.length)fail("DASHBOARD_REGISTERED_UNCHANGED_ROW_INVALID");
      } else {
        if(delta<=0 || qty+delta!==current.shares || row.policy_revalidation_required!==true || row.risk_execution_state!=="NON_EXECUTABLE" || !equal(row.blocker_codes,["OWNER_POLICY_REVALIDATION_REQUIRED"]) || !row.fact_refs.length)fail("DASHBOARD_REGISTERED_CHANGED_ROW_INVALID");
        if(row.change_kind==="NEW_POSITION") {
          if(row.baseline_position_state!=="ABSENT" || qty!==0 || cost!==0 || row.avg_cost_before!==null)fail("DASHBOARD_REGISTERED_NEW_POSITION_INVALID");
        } else if(row.change_kind!=="EXISTING_POSITION_ADD" || row.baseline_position_state!=="PRESENT" || qty<=0 || decimal(row.avg_cost_before)<=0)fail("DASHBOARD_REGISTERED_CHANGE_KIND_INVALID");
        changed.push(row);
      }
    });
    if(Object.keys(seen).length!==Object.keys(positions).length || !changed.length)fail("DASHBOARD_REGISTERED_POSITION_UNION_MISMATCH");
    return {report:body,close:summary,changes:changed};
  }

  async function validate(raw, now) {
    var head = await document(raw.head, HEAD, "cn-daily-completed-head.v1");
    var selector = await document(raw.selector, SELECTOR, "cn_aggressive_dashboard_selector.v3");
    var proposed;
    try { proposed=JSON.parse(raw.evidence); } catch (_) { fail("DASHBOARD_JSON_INVALID"); }
    var registered=proposed.schema_version==="cn-daily-dashboard-serving.v2";
    var view = await document(raw.evidence, registered?VIEW.concat(REGISTERED_VIEW):VIEW, registered?"cn-daily-dashboard-serving.v2":"cn-daily-dashboard-serving.v1");
    var nowMs = now === undefined ? Date.now() : Date.parse(now);
    if (!Number.isFinite(nowMs)) fail("DASHBOARD_CLOCK_INVALID");
    if (!validDay(head.trade_date) || !validTime(head.registered_at) || Date.parse(head.registered_at) > nowMs) fail("DASHBOARD_HEAD_TIME_INVALID");
    ref(head.completion_ref); authority(head.authority); authority(view.authority);
    if (head.completion_ref.path !== "results/operations/daily_production/CN/" + head.trade_date + "/completion.v1.json") fail("DASHBOARD_COMPLETION_PATH_INVALID");
    if (head.previous_head_sha256 !== null && !SHA.test(head.previous_head_sha256)) fail("DASHBOARD_HEAD_PREIMAGE_INVALID");
    if (selector.status !== "UPDATED" || selector.reason !== "native_eod_completed" ||
        selector.trade_date !== head.trade_date || !equal(selector.completion_ref, head.completion_ref) ||
        selector.completed_head_sha256 !== await sha(raw.head)) fail("DASHBOARD_HEAD_SELECTOR_MISMATCH");
    if (!validTime(selector.updated_at) || Date.parse(selector.updated_at) > nowMs || Date.parse(selector.updated_at) < Date.parse(head.registered_at)) fail("DASHBOARD_SELECTOR_TIME_INVALID");
    if (selector.v1_byte_sha256 !== await sha(raw.v1) || selector.v2_byte_sha256 !== await sha(raw.v2) ||
        selector.daily_evidence_sha256 !== await sha(raw.evidence)) fail("DASHBOARD_SELECTED_BYTES_MISMATCH");
    var v1 = JSON.parse(raw.v1), v2 = JSON.parse(raw.v2);
    var validation = V2.validateBundle(v2);
    if (!validation.valid || !equal(v2.canonical_v1, v1) || selector.v2_content_sha256 !== v2.content_sha256 || selector.attempt_id !== v2.publication_attempt_id) fail("DASHBOARD_FINANCIAL_BINDING_INVALID");
    var iso = head.trade_date.slice(0,4)+"-"+head.trade_date.slice(4,6)+"-"+head.trade_date.slice(6,8);
    if (v1.latest_data_date !== iso || v2.freshness.mark_as_of !== iso || v1.portfolio.performance_end_date !== iso) fail("DASHBOARD_FINANCIAL_DATE_MISMATCH");
    if (view.view_designation !== "LATEST_COMPLETED_EOD" || view.trade_date !== head.trade_date ||
        !equal(view.completion_ref, head.completion_ref) || view.completed_head_ref.sha256 !== selector.completed_head_sha256 ||
        view.native_valid_through !== v2.freshness.valid_through || !validTime(view.research_cutoff)) fail("DASHBOARD_EVIDENCE_HEAD_MISMATCH");
    if (!validTime(view.native_valid_through)) fail("DASHBOARD_VALID_THROUGH_INVALID");
    if (Date.parse(selector.updated_at) > Date.parse(view.native_valid_through)) fail("DASHBOARD_EXPIRED_PUBLICATION_INTENT");
    ref(view.completed_head_ref); ref(view.daily_evidence_ref);
    keys(view.financial_refs, ["v1", "v2"]); ref(view.financial_refs.v1); ref(view.financial_refs.v2);
    if (view.financial_refs.v1.sha256 !== selector.v1_byte_sha256 || view.financial_refs.v2.sha256 !== selector.v2_byte_sha256) fail("DASHBOARD_EVIDENCE_FINANCIAL_REFS_INVALID");
    if (await sha(JSON.stringify(stable(view.evidence))) !== view.daily_evidence_ref.sha256 || view.evidence.kind !== "daily_dashboard_evidence") fail("DASHBOARD_DAILY_EVIDENCE_SHA_INVALID");
    var evidence = view.evidence.payload;
    if (evidence.trade_date !== head.trade_date || evidence.research_state !== "EVIDENCE_BOUND" || !Number.isInteger(evidence.top100_count) || evidence.top100_count !== 100) fail("DASHBOARD_RESEARCH_DATE_OR_COUNT_INVALID");
    keys(evidence.source_bindings, DOMAINS); keys(evidence.decision_state_counts, STATES);
    var total = 0;
    STATES.forEach(function (state) { var n=evidence.decision_state_counts[state]; if (!Number.isInteger(n) || n<0) fail("DASHBOARD_DECISION_COUNT_INVALID"); total+=n; });
    if (total !== evidence.top100_count) fail("DASHBOARD_DECISION_COMPANY_SET_INVALID");
    DOMAINS.forEach(function (name) {
      var item=evidence.source_bindings[name];
      keys(item,["node_id","request_ref","terminal_ref","output_refs","trade_date","state"]);
      if (item.node_id !== name || item.trade_date !== head.trade_date || item.state !== "SUCCEEDED") fail("DASHBOARD_SOURCE_DOMAIN_MISMATCH");
      ref(item.request_ref); ref(item.terminal_ref);
      Object.keys(item.output_refs).forEach(function (key) { ref(item.output_refs[key]); });
    });
    var changes = registered ? await registeredView(view,v1,evidence) : null;
    var expired = nowMs > Date.parse(view.native_valid_through);
    return {mode:"SEALED", valid:true, head:head, evidence:evidence, view:view, v1:v1,registered:changes,
      snapshot:{schema_version:"cn_daily_dashboard_view.v1",view_designation:"LATEST_COMPLETED_EOD",
        age_calendar_days:Math.max(0,Math.floor((Date.parse(new Date(nowMs+8*3600000).toISOString().slice(0,10)+"T00:00:00Z")-Date.parse(iso+"T00:00:00Z"))/86400000)),
        status:{integrity:"VERIFIED",freshness:expired?"STALE":"UPDATED",current_holdings:expired?"STALE":"AS_OF_COMPLETED_EOD",current_absolute_performance:expired?"STALE":"AS_OF_COMPLETED_EOD",canonical_history:v2.completeness.canonical_history,benchmark_relative:v2.completeness.benchmark_relative},
        bundle:v2,blockers:expired?["已超过快照有效期，仅供查看该日历史记录。"]:[],holdings_label:(expired?"已过期日终快照 · ":"最新已完成日终 · ")+iso,
        absolute_performance_label:"已完成日终业绩 · 截至 "+iso,
        anchor_label:"日终研究截止 "+view.research_cutoff,
        benchmark_label:"基准截至 "+v2.completeness.benchmark_as_of}};
  }

  async function fromPage(page) {
    var protocol = page.location.protocol;
    var selector = page.__cnAggressivePrivateDashboardSelector;
    var marker = "MyQuantCNDailyCompletedHeadRaw" in page || "MyQuantCNDailyDashboardEvidenceRaw" in page || "MyQuantCNDailyDashboardSelectorRaw" in page || (selector && selector.schema_version === "cn_aggressive_dashboard_selector.v3");
    var raw = {head:page.MyQuantCNDailyCompletedHeadRaw,selector:page.MyQuantCNDailyDashboardSelectorRaw,
      evidence:page.MyQuantCNDailyDashboardEvidenceRaw,v1:page.MyQuantCNDailyFinancialV1Raw,v2:page.MyQuantCNDailyFinancialV2Raw};
    if (protocol === "http:" || protocol === "https:") {
      async function fetchRaw(name, optional) {
        var controller=new AbortController();
        var timer=setTimeout(function () { controller.abort(); },10000);
        try {
          var response=await page.fetch("private/generated/"+name+"?eod="+Date.now(),{cache:"no-store",signal:controller.signal});
          if (optional && response.status===404) return null;
          if (!response.ok) fail("DASHBOARD_HTTP_SOURCE_UNAVAILABLE");
          return await response.text();
        } finally { clearTimeout(timer); }
      }
      raw.head=await fetchRaw("cn_daily_completed_head.v1.json",true);
      if (!raw.head && !marker) return {mode:"LEGACY",valid:true};
      if (!raw.head) fail("DASHBOARD_HEAD_MISSING");
      var values=await Promise.all(["cn_aggressive_dashboard_selector.v2.json","cn_daily_dashboard_evidence.v1.json","cn_aggressive_dashboard.v1.json","cn_aggressive_dashboard.v2.json"].map(function (name) { return fetchRaw(name,false); }));
      raw.selector=values[0];raw.evidence=values[1];raw.v1=values[2];raw.v2=values[3];
    } else if (protocol === "file:") {
      if (!marker) return {mode:"LEGACY",valid:true};
    } else fail("DASHBOARD_SERVING_ROUTE_UNSUPPORTED");
    return validate(raw);
  }
  return {validate:validate,fromPage:fromPage,sha256:sha};
});
