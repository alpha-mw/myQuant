"use strict";
// Unit-level route and byte-contract checks. These do not prove browser file access.
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const Contract = require("../js/cn_daily_dashboard_contract_v1.js");
const fixture = require("./fixtures/daily_sealed_synthetic_raw.json");
const raw = fixture.raw;
const registeredFixture = require("./fixtures/daily_registered_synthetic_raw.json");
const until = JSON.parse(raw.evidence).native_valid_through;
const hash = value => crypto.createHash("sha256").update(value).digest("hex");
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}
function seal(value) {
  delete value.content_sha256;
  value.content_sha256 = hash(JSON.stringify(canonical(value)));
  return JSON.stringify(canonical(value));
}
function page() {
  return {
    location: { protocol: "file:" },
    MyQuantCNDailyCompletedHeadRaw: raw.head,
    MyQuantCNDailyDashboardSelectorRaw: raw.selector,
    MyQuantCNDailyDashboardEvidenceRaw: raw.evidence,
    MyQuantCNDailyFinancialV1Raw: raw.v1,
    MyQuantCNDailyFinancialV2Raw: raw.v2,
    __cnAggressivePrivateDashboardSelector: JSON.parse(raw.selector),
  };
}

test("exact bytes bind dated EOD, five domains, and all 100 companies", async () => {
  const result = await Contract.validate(raw, until);
  assert.equal(result.snapshot.status.freshness, "UPDATED");
  assert.equal(result.snapshot.view_designation, "LATEST_COMPLETED_EOD");
  assert.equal(result.snapshot.status.current_holdings, "AS_OF_COMPLETED_EOD");
  assert.equal(result.evidence.top100_count, 100);
  assert.equal(Object.keys(result.evidence.source_bindings).length, 5);
  assert.equal(Object.values(result.evidence.decision_state_counts).reduce((a,b) => a+b,0), 100);
});

test("one millisecond after valid-through remains dated but becomes STALE", async () => {
  const result = await Contract.validate(raw, new Date(Date.parse(until)+1).toISOString());
  assert.equal(result.valid, true);
  assert.equal(result.snapshot.status.freshness, "STALE");
  assert.equal(result.snapshot.status.current_holdings, "STALE");
  assert.equal(result.snapshot.status.current_absolute_performance, "STALE");
  assert.match(result.snapshot.holdings_label, /已过期/);
  assert.equal(result.snapshot.age_calendar_days, 15);
  const nextDay = await Contract.validate(raw, "2026-09-13T00:00:00+08:00");
  assert.equal(nextDay.snapshot.age_calendar_days, 16);
  assert.deepEqual(result.v1, JSON.parse(raw.v1));
});

for (const name of ["head", "selector", "evidence", "v1", "v2"]) {
  test(`changed ${name} bytes cannot fall back to another view`, async () => {
    await assert.rejects(Contract.validate({...raw, [name]: raw[name]+" "}, until), /DASHBOARD_/);
  });
}

test("a validly hashed different head still fails selector binding", async () => {
  const head = JSON.parse(raw.head);
  head.completion_ref.sha256 = "f".repeat(64);
  await assert.rejects(Contract.validate({...raw, head:seal(head)}, until), /HEAD_SELECTOR_MISMATCH/);
});

test("a selector dated after expiry is rejected even when otherwise sealed", async () => {
  const selector = JSON.parse(raw.selector);
  selector.updated_at = new Date(Date.parse(until)+1000).toISOString();
  await assert.rejects(Contract.validate({...raw, selector:seal(selector)}, selector.updated_at), /EXPIRED_PUBLICATION/);
});

test("registered raw-string route uses byte proof and rejects missing head", async () => {
  const input = page();
  assert.equal((await Contract.fromPage(input)).mode, "SEALED");
  delete input.MyQuantCNDailyCompletedHeadRaw;
  await assert.rejects(Contract.fromPage(input), /DASHBOARD_/);
  assert.equal((await Contract.fromPage({location:{protocol:"file:"}})).mode, "LEGACY");
});

test("HTTP reads authoritative fixed JSON files without cache", async () => {
  const input = page(); input.location.protocol = "http:";
  const names = {
    "cn_daily_completed_head.v1.json": "head", "cn_aggressive_dashboard_selector.v2.json":"selector",
    "cn_daily_dashboard_evidence.v1.json":"evidence", "cn_aggressive_dashboard.v1.json":"v1",
    "cn_aggressive_dashboard.v2.json":"v2",
  };
  const calls = [];
  input.fetch = async (url, options) => {
    calls.push(url); assert.equal(options.cache,"no-store");
    const key = names[url.split("/").pop().split("?")[0]];
    assert.ok(key);
    return {ok:true,status:200,text:async()=>raw[key]};
  };
  assert.equal((await Contract.fromPage(input)).mode,"SEALED");
  assert.equal(calls.length,5);
  input.fetch = async () => ({ok:false,status:404});
  await assert.rejects(Contract.fromPage(input), /HEAD_MISSING/);
});

test("public page does not load private completed-EOD sources", () => {
  const html = fs.readFileSync(path.join(__dirname,"../public.html"),"utf8");
  assert.doesNotMatch(html,/cn_daily_completed_head|cn_daily_dashboard_evidence|cn_daily_dashboard_contract/);
});

test("registered serving v2 separates owner changes from the close writer", async () => {
  const input=registeredFixture.raw;
  const result=await Contract.validate(input,JSON.parse(input.evidence).native_valid_through);
  assert.equal(result.registered.changes.length,1);
  assert.equal(result.registered.changes[0].change_kind,"NEW_POSITION");
  assert.equal(result.registered.close.close_writer_trade_count,0);
  assert.equal(result.registered.report.broker_statement_verified,false);
  assert.equal(result.registered.report.risk_readiness_state,"OWNER_POLICY_REVALIDATION_REQUIRED");
  assert.equal(Object.keys(result.evidence.source_bindings).length,5);
});

function changedRegistered(mutate) {
  const input={...registeredFixture.raw};
  const view=JSON.parse(input.evidence);
  mutate(view);
  view.registered_transition_ref.sha256=hash(JSON.stringify(canonical(view.registered_transition)));
  input.evidence=seal(view);
  const selector=JSON.parse(input.selector);
  selector.daily_evidence_sha256=hash(input.evidence);
  input.selector=seal(selector);
  return input;
}

for(const [label,mutate] of [
  ["close trades",v=>v.registered_close_summary.close_writer_trade_count=1],
  ["close orders",v=>v.registered_close_summary.close_writer_order_count=1],
  ["close fills",v=>v.registered_close_summary.close_writer_fill_count=1],
  ["official state",v=>v.registered_close_summary.official_valuation=false],
  ["final pointer",v=>v.registered_close_summary.final_pointer_ref.sha256="f".repeat(64)],
  ["writer pointer",v=>v.registered_close_summary.writer_pointer_ref.sha256="f".repeat(64)],
  ["baseline",v=>v.registered_close_summary.decision_baseline_pointer_ref.sha256="f".repeat(64)],
  ["record identity",v=>v.registered_close_summary.final_record_id=v.registered_close_summary.writer_record_id],
  ["broker claim",v=>v.registered_transition.payload.broker_statement_verified=true],
  ["risk claim",v=>v.registered_transition.payload.risk_readiness_state="READY"],
  ["cash",v=>v.registered_transition.payload.cash_after_cny="1.00"],
  ["holding quantity",v=>v.registered_transition.payload.position_rows[0].shares_after="1"],
  ["missing holding",v=>v.registered_transition.payload.position_rows.pop()],
  ["declaration clock",v=>v.registered_transition.payload.registered_at="2099-01-01T00:00:00Z"],
  ["old version",v=>v.schema_version="cn-daily-dashboard-serving.v1"],
]) {
  test(`registered ${label} tampering fails with refreshed outer hashes`,async()=>{
    const input=changedRegistered(mutate);
    await assert.rejects(Contract.validate(input,JSON.parse(input.evidence).native_valid_through),/DASHBOARD_/);
  });
}
