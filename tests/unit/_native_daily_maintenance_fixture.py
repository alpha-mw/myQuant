"""Native maintenance orchestration over verified synthetic no-action core inputs.

Only scheduler logical time and external close-calendar response are simulated.
Stage adapters verify actual generated Market/PIT and native history audit files.
"""

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
from quant_investor.market.daily_maintenance import run_cn_daily_maintenance, MaintenanceComponents
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.close_session_authority import acquire_close_session_authority
from quant_investor.market.tushare_transport import replay_tushare_response_bytes
from quant_investor.factors.production_rollover import validate_daily_maintenance_receipt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maintenance(
    root,
    day: str | None = None,
    *,
    core_completed=None,
    auxiliary_callback=None,
    expected_target_trade_date=None,
    core_replay_completed=None,
    macro_callback=None,
    now=None,
):
    target = day or "2026-08-25"
    compact = target.replace("-", "")
    suffix = "" if day is None else "-" + day
    workspace = root / "factor-workspace"
    data = workspace / "data"
    reader = MarketDataReader(data_root=data)
    assert reader.snapshot()["healthy"]
    binding = reader.coverage_bound_pit()
    assert binding["status"] == "passed"
    pit_pointer = data / "parquet/cn/reference/stock_basic_membership_latest.json"
    pit_document = json.loads(pit_pointer.read_bytes())
    market_pointer = data / "parquet/cn/_latest.json"
    market = json.loads(market_pointer.read_bytes())
    audit_result = json.loads((root / ("successor-market-audit" + suffix + ".json")).read_text())
    audit = Path(audit_result["audit_path"])
    history = json.loads(audit.read_bytes())
    assert history["history_audit_status"] == "passed"
    assert history["audited_trade_dates_count"] == 100
    pit_evidence = {
        "pit_binding": {
            key: pit_document[key]
            for key in (
                "generation_id",
                "generation_manifest_path",
                "generation_manifest_sha256",
                "canonical_path",
                "canonical_sha256",
            )
        }
    }
    pit_evidence["pit_binding"].update(
        discovery_pointer_path=str(pit_pointer), discovery_pointer_sha256=sha(pit_pointer)
    )
    if macro_callback is not None:
        scope = workspace / "data/cn_universe/cn_index_components.json"
        pit_evidence.update(scope_path=str(scope), scope_sha256=sha(scope))
    market_evidence = {
        "pointer_path": str(market_pointer),
        "pointer_sha256": sha(market_pointer),
        "snapshot_manifest_path": market["manifest_path"],
        "snapshot_manifest_sha256": sha(Path(market["manifest_path"])),
    }
    history_evidence = {
        "audit_path": str(audit),
        "audit_sha256": sha(audit),
        "history_audit_status": "passed",
    }

    def component(evidence):
        def verified(context):
            assert context.target_date == compact
            return {
                "status": "NO_ACTION",
                "write_performed": False,
                "blockers": [],
                "evidence": evidence,
            }

        return verified

    fields = ["exchange", "cal_date", "is_open", "pretrade_date"]

    class SyntheticClient:
        def request(self, *, params, **kwargs):
            day = datetime.strptime(params["start_date"], "%Y%m%d")
            end = datetime.strptime(params["end_date"], "%Y%m%d")
            prior = day - timedelta(days=1)
            while prior.weekday() >= 5:
                prior -= timedelta(days=1)
            previous = prior.strftime("%Y%m%d")
            items = []
            while day <= end:
                opened = day.weekday() < 5
                items.append(
                    [params.get("exchange", "SSE"), day.strftime("%Y%m%d"), int(opened), previous]
                )
                if opened:
                    previous = day.strftime("%Y%m%d")
                day += timedelta(days=1)
            raw = json.dumps(
                {
                    "code": 0,
                    "msg": "",
                    "detail": "",
                    "request_id": "synthetic-close",
                    "data": {"fields": fields, "items": items, "has_more": False, "count": 0},
                }
            ).encode()
            return replay_tushare_response_bytes(raw, api_name="trade_cal", expected_fields=fields)

    def close(**kwargs):
        return acquire_close_session_authority(**kwargs, client=SyntheticClient())

    result = run_cn_daily_maintenance(
        workspace_root=workspace,
        run_root=data / "private/cn_daily_maintenance",
        mode="execute",
        attempt_slot="2020",
        now=now or datetime.fromisoformat(target + "T13:00:00+00:00"),
        close_authority=close,
        core_completed=core_completed,
        _expected_target_trade_date=expected_target_trade_date,
        _core_replay_completed=core_replay_completed,
        components=MaintenanceComponents(
            pit=component(pit_evidence),
            market=component(market_evidence),
            history=component(history_evidence),
            fundamental=auxiliary_callback,
            macro_release=macro_callback,
        ),
    )
    (root / ("successor-maintenance-result" + suffix + ".json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    ref = result["core_completion_ref"]
    verified = validate_daily_maintenance_receipt(
        workspace_root=workspace, receipt_path=ref["path"], expected_receipt_sha256=ref["sha256"]
    )
    output = {
        "synthetic": True,
        "logical_scheduler_time_simulated": True,
        "native_core_checkpoint": ref,
        "verification": verified,
        "full_dag_proof": False,
    }
    if macro_callback is not None:
        output["maintenance_result"] = result
    (root / ("successor-maintenance-verification" + suffix + ".json")).write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return output


if __name__ == "__main__":
    print(json.dumps(maintenance(Path(sys.argv[1])), indent=2))
