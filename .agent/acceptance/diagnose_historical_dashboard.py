"""Read-only reproduction of the exact frozen Dashboard rendering inputs."""
import hashlib
import json
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

import quant_investor

case = Path(sys.argv[1]).resolve(strict=True)
receipt = json.loads((case / "fixture-receipt.json").read_text())
assert Path(quant_investor.__file__).resolve() == Path(receipt["runtime_verification"]["import_origin"]).resolve()
repo = case / "repository"
sys.path.extend([str(repo), str(repo / "scripts"), "/Users/maxwell/mySpace/myQuant/.venv/lib/python3.13/site-packages"])
from scripts import daily_dashboard_adapter as dashboard
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import utc_stamp

w = case / "factor-workspace"
base = w / "results/operations/daily_production/CN/20260828"
status = json.loads((base / "dag-status.v1.json").read_text())
key = status["nodes"]["dashboard"]["request_key"]
request = json.loads((base / "nodes/dashboard" / key / "request.json").read_text())
assert hashlib.sha256(Path(dashboard.__file__).read_bytes()).hexdigest() == request["adapter_sha256"]
intent_ref = request["input_refs"]["render_intent"]
raw = (w / intent_ref["path"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == intent_ref["sha256"]
intent = json.loads(raw)
refs = request["input_refs"]
adapter_type = dashboard.HistoricalDashboardAdapter if intent["historical_mode"] else dashboard.CurrentDashboardAdapter
adapter = adapter_type(workspace=str(w), journal=DailyJournal(str(w), "20260828"),
    release_ref=request["release_ref"], plan_ref=refs["store_plan"], market_ref=refs["market"],
    benchmark_ref=refs["benchmark"], risk_free_ref=refs["risk_free"])
adapter.recipe, adapter.recipe_ref = intent, intent_ref

def inventory():
    return {str(p.relative_to(w)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in w.rglob("*") if p.is_file()}

before = inventory()
output = {"producer_commit": receipt["commit"], "historical_mode": intent["historical_mode"],
          "request_key": key, "write_methods_called": False, "scope": "original native render prerequisites only"}
step = "request and source byte checks"
try:
    adapter._request(request)
    for ref in adapter.refs.values():
        adapter._source(ref)
    step = "committed Store readback"
    records = w / dashboard.RECORD_ROOT
    selected, _ = dashboard.committed_dashboard_store(project_root=w, record_root=records,
        plan_ref=refs["store_plan"], valuation_date="2026-08-28")
    step = "Market reader"
    reader = dashboard.MarketDataReader(data_root=w / "data", frozen_snapshot_ref={
        "path": Path(refs["market"]["path"]).relative_to("data").as_posix(), "sha256": refs["market"]["sha256"]})
    if not adapter.historical_mode:
        from quant_investor.strategy_records.store import load_registered_catalog
        assert load_registered_catalog(records) == selected, "DASHBOARD_CURRENT_STORE_NOT_SELECTED_COMMIT"
        current = dashboard.MarketDataReader(data_root=w / "data")
        assert current.snapshot().get("snapshot_id") == reader.snapshot().get("snapshot_id"), "DASHBOARD_CURRENT_MARKET_NOT_SELECTED_SNAPSHOT"
        reader = current
    now = utc_stamp(intent["generated_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
    step = "v1 native bundle"
    v1 = dashboard.build_bundle(project_root=w, record_root=records,
        benchmark_path=w/refs["benchmark"]["path"], risk_free_path=w/refs["risk_free"]["path"],
        generated_at=now.isoformat(timespec="seconds"), today=now.date(),
        historical_close_plan_ref=refs["store_plan"] if adapter.historical_mode else None,
        historical_valuation_date="2026-08-28" if adapter.historical_mode else None)
    step = "v2 native bundle"
    v2 = dashboard.build_v2_bundle(project_root=w, v1_bundle=v1,
        v1_json_path=dashboard._expected_output_paths(w)[0], record_root=records,
        generation_local_date=now.date(), generated_at=now.isoformat(timespec="seconds"),
        publication_attempt_id="dashboard-v2-dag-20260828-"+key[:16], market_reader=reader,
        v1_json_bytes_override=dashboard._render_json(v1),
        historical_close_plan_ref=refs["store_plan"] if adapter.historical_mode else None,
        historical_valuation_date="2026-08-28" if adapter.historical_mode else None)
    step = "native capture validation before any retain/write"
    from scripts.daily_dashboard_capture import (
        validate_bundle_shape, validate_v2_shape, verify_source_refs, verify_v2_source_refs,
    )
    errors = (validate_bundle_shape(v1) + validate_v2_shape(v2) + verify_source_refs(v1, w)
              + verify_v2_source_refs(v2, w, v1_bytes_override=dashboard._render_json(v1)))
    if errors:
        raise ValueError("DASHBOARD_NATIVE_VALIDATION_FAILED:" + ";".join(errors))
    step = "native capture date gate before any retain/write"
    adapter.capture._date_gate(v1, v2)
    output.update(status="RENDER_AND_PREWRITE_VALIDATION_PASSED", v1_date=v1.get("latest_data_date"), v2_schema=v2.get("schema_version"))
except Exception as exc:
    output.update(status="REPRODUCED_FAILURE", failed_step=step, error_type=type(exc).__name__,
        error=str(exc), traceback=traceback.format_exc())
output["workspace_sha_mtime_unchanged"] = before == inventory()
assert output["workspace_sha_mtime_unchanged"]
(case / "historical-dashboard-readonly-diagnosis.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps(output,indent=2))
