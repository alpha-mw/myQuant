"""Fresh current-source native financial/collector proof; full EOD admission excluded."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

from _public_catchup_fixture import routed_collection
from quant_investor.operations.catchup_binding import read_catchup_collection
from cn_dashboard_common import DashboardInputError
from export_cn_aggressive_dashboard_data import _expected_output_paths

spec = importlib.util.spec_from_file_location(
    "phase9_source_fixture", ".agent/acceptance/build_phase9_source_fixture.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

routing_root = Path(tempfile.mkdtemp(prefix="phase11-native-routing-")).resolve()
request_ref, _ = routed_collection(
    routing_root, now=datetime(2026, 9, 1, 13, tzinfo=timezone.utc)
)
context = read_catchup_collection(workspace=str(routing_root), request_ref=request_ref)
route = context["routing"]["20260828"]
assert route["maintenance_mode"] == "HISTORICAL"
assert route["dashboard_mode"] == "HISTORICAL_CAPTURE"

try:
    module.build(future_benchmark_day="2026-08-31", prefix="phase11-wrong-mode-")
except DashboardInputError as exc:
    assert "DASHBOARD_STALE:benchmark" in str(exc), str(exc)
    wrong_mode_error = str(exc)
else:
    raise AssertionError("current mode accepted a future benchmark tail")

root, node, request, outcome = module.build(
    publish_current_dashboard=route["dashboard_mode"] == "CURRENT_LATEST_EOD",
    future_benchmark_day="2026-08-31", prefix="phase11-historical-dashboard-",
)
assert outcome.state.value == "SUCCEEDED"
v2 = json.loads((root / outcome.output_refs["v2"]["path"]).read_bytes())
assert v2["schema_version"] == "cn_aggressive_dashboard_history.v1"
assert v2["freshness"]["mark_as_of"] == "2026-08-28"
assert v2["evidence_timing"] == "RETROSPECTIVE_RECOMPUTE"
assert "2026-08-31" in (root / "portfolio_dashboard/inputs/cn_index_benchmark.csv").read_text()
assert not any(path.exists() for path in _expected_output_paths(root))
before = {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
          for p in root.rglob("*") if p.is_file()}
with node.journal.locked():
    node.execute(request)
assert node.probe(request).outcome == outcome
assert before == {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                  for p in root.rglob("*") if p.is_file()}
record = {
    "observed_at": datetime.now(timezone.utc).isoformat(), "result": "PASS",
    "root": str(root), "routing_root": str(routing_root), "routing": route,
    "original_wrong_mode_reproduced": wrong_mode_error,
    "benchmark_tail": "2026-08-31", "historical_mark_date": "2026-08-28",
    "synthetic": True, "full_native_eod_admission": False,
    "native_financial_and_five_source_collector": True,
    "outputs": outcome.output_refs, "serving_files_written": False,
    "repeat_unchanged": True, "old_failed_job_recovered": False,
    "limitations": ["Native Store/Market/financial renderer and source collector exercised",
                    "Retained synthetic Factor/Theme/Decision artifacts and explicit journal wrappers",
                    "Not an installed full DAG or live financial run"],
}
Path(".agent/acceptance/phase11-native-historical-dashboard.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print(json.dumps(record, indent=2))
