"""Bounded native successor case with explicit date/preimage and dated receipts."""

import hashlib
import json
from pathlib import Path
import sys
from quant_investor.cli.unified import factor_production_rollover, factor_production_observe
from quant_investor.operations.core_pool import publish_core_pool
from quant_investor.intelligence.storage import approved_theme_policy_v2


def run(root: Path, day: str, expected_pointer_sha: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _native_daily_rollover_fixture import prepare_market
    from _native_daily_maintenance_fixture import maintenance
    from _native_daily_calendar_fixture import capture_synthetic_calendar

    workspace = root / "factor-workspace"
    current = workspace / "results/factors/_active.json"
    if hashlib.sha256(current.read_bytes()).hexdigest() != expected_pointer_sha:
        raise ValueError("explicit successor preimage mismatch; inspect native recovery first")
    result_path = root / ("successor-case-" + day + ".json")
    if result_path.exists():
        raise ValueError("case receipt already exists; no blind restart")

    def save(name, value):
        path = root / (name + "-" + day + ".json")
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")
        print("PASS", day, name, flush=True)

    print("START", day, "market", flush=True)
    market = prepare_market(root, day)
    assert market["history"]["history_audit_status"] == "passed"
    save("market-case", market)
    print("START", day, "calendar", flush=True)
    calendar = capture_synthetic_calendar(
        release_root=root,
        fixture_source=Path(__file__).parent / "test_tushare_calendar_authority.py",
        cutoff=day,
    )
    save("calendar-case", calendar)
    prepared = maintenance(root, day)
    save("maintenance-case", prepared)
    c = calendar["capture"]
    m = prepared["native_core_checkpoint"]
    print("START", day, "rollover", flush=True)
    rolled = factor_production_rollover(
        workspace_root=str(workspace),
        market_data_root=str(workspace / "data"),
        calendar_capture_root=c["capture_root"],
        expected_calendar_success_sha256=c["capture_success_file_ref"]["byte_sha256"],
        maintenance_receipt=m["path"],
        expected_maintenance_receipt_sha256=m["sha256"],
        expected_current_pointer_sha256=expected_pointer_sha,
    )
    save("rollover-case", rolled)
    observed = factor_production_observe(workspace_root=str(workspace))
    save("observations-case", observed)
    compact = day.replace("-", "")
    policy = approved_theme_policy_v2()["payload"]
    core = None
    if (
        compact >= policy["effective_signal_date"]
        and day + "T07:00:00Z" >= policy["effective_from"]
    ):
        print("START", day, "core", flush=True)
        release = workspace / "fixtures/release.json"
        core = publish_core_pool(
            workspace=str(workspace),
            trade_date=compact,
            factor_pointer_sha256=rolled["factor_pointer_byte_sha256"],
            release_ref={
                "path": "fixtures/release.json",
                "sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
            },
        )
        save("core-case", core)
    result = {
        "synthetic": True,
        "day": day,
        "rollover": rolled,
        "observations": observed,
        "core": core,
        "top100_policy_eligible": core is not None,
        "full_dag_proof": False,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), sys.argv[2], sys.argv[3]), indent=2))
