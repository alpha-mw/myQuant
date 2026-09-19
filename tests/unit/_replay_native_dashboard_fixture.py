"""Read-only native Dashboard rebuild diagnostic over a real synthetic EOD."""

from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from daily_dashboard_capture import DailyDashboardCapture
from quant_investor.operations.dashboard_replay_sources import (
    RetainedDashboardSources,
    retained_dashboard_sources,
    retained_store_inventory,
    RetainedDashboardMarket,
)
from cn_dashboard_common import build_bundle
from cn_dashboard_v2 import build_v2_bundle
from export_cn_aggressive_dashboard_data import _render_json
from scripts.daily_completion_store import replay_completed_store
from quant_investor.operations.daily_journal import DailyJournal


def run(root: Path):
    workspace = root / "factor-workspace"
    records = workspace / "results/strategy_records/CN/aggressive_tech_manufacturing"
    ref = json.loads((root / "full-completion-ref.json").read_text())
    store = replay_completed_store(
        workspace=str(workspace), trade_date="20260827", completion_ref=ref
    )
    cap = DailyDashboardCapture(str(workspace), DailyJournal(str(workspace), "20260827"))
    seed = json.loads((workspace / cap.receipt_path).read_text())
    capture = cap.read(request_ref=seed["request_ref"])
    old_v1, old_v2 = [json.loads(cap._read(capture[k])) for k in ("v1_ref", "v2_ref")]
    sources = {
        row["source_ref"]["path"]: cap._read(row["retained_ref"]) for row in capture["sources"]
    }
    for path, raw in retained_store_inventory(
        project_root=workspace, record_root=records, pointer_ref=store["output_refs"]["pointer"]
    ).items():
        if path in sources and sources[path] != raw:
            raise ValueError("retained binding conflict")
        sources[path] = raw
    request = json.loads((workspace / capture["request_ref"]["path"]).read_text())
    refs = request["input_refs"]
    now = datetime.fromisoformat(old_v1["generated_at"])
    with retained_dashboard_sources(
        RetainedDashboardSources(workspace, sources, store["output_refs"]["pointer"])
    ):
        v1 = build_bundle(
            project_root=workspace,
            record_root=records,
            benchmark_path=workspace / refs["benchmark"]["path"],
            risk_free_path=workspace / refs["risk_free"]["path"],
            generated_at=old_v1["generated_at"],
            today=now.date(),
        )
        if v1 != old_v1:
            raise ValueError("native v1 rebuild differs")
        v2 = build_v2_bundle(
            project_root=workspace,
            v1_bundle=v1,
            v1_json_path=workspace / old_v2["canonical_v1_ref"]["path"],
            record_root=records,
            generation_local_date=now.date(),
            generated_at=old_v2["generated_at"],
            publication_attempt_id=old_v2["publication_attempt_id"],
            market_reader=RetainedDashboardMarket(
                project_root=workspace, manifest_ref=refs["market"]
            ),
            v1_json_bytes_override=_render_json(v1),
        )
    result = {
        "v1_equal": v1 == old_v1,
        "v2_equal": v2 == old_v2,
        "v2_different_keys": [k for k in v2 if v2[k] != old_v2.get(k)],
        "v2_rebuilt": v2,
    }
    (root / "dashboard-pair-replay-diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "v2_rebuilt"})


if __name__ == "__main__":
    run(Path(sys.argv[1]))
