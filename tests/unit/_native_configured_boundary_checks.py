"""Installed original-request idempotence and closed-day source-entry checks."""

from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def inventory(root):
    return {
        p.relative_to(root).as_posix(): (
            p.lstat().st_mode,
            p.lstat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        )
        for p in root.rglob("*")
    }


def run(root, dependencies, mode):
    import quant_investor

    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(receipt["runtime_verification"]["import_origin"]).resolve()
    ):
        raise ValueError("verified installed runtime required")
    repository = root / "repository"
    sys.path.extend(
        [
            str(repository),
            str(repository / "scripts"),
            str(repository / "tests/unit"),
            str(dependencies),
        ]
    )
    workspace = root / "factor-workspace"
    initial = json.loads((root / "configured-native-proof.json").read_bytes())

    def read(ref):
        raw = (workspace / ref["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("retained proof source changed")
        return json.loads(raw)

    config = read(initial["config_ref"])
    from _native_synthetic_clock import synthetic_clock
    from scripts.daily_production import dispatch_daily_request
    from scripts.daily_source_inputs import configured_source_inputs
    from quant_investor.cli import unified
    from quant_investor.intelligence.storage import DailyResearchPoolStore
    from quant_investor.strategy_records import store, event_store
    from quant_investor.market import cn_benchmark_store

    def forbidden(*args, **kwargs):
        raise AssertionError("repeat/closed-day entry invoked a producer or financial writer")

    def prohibit(stack):
        stack.enter_context(
            patch("quant_investor.market.daily_maintenance.run_cn_daily_maintenance", forbidden)
        )
        stack.enter_context(patch.object(unified, "factor_production_activate", forbidden))
        stack.enter_context(patch.object(unified, "factor_production_rollover", forbidden))
        if mode == "closed":
            stack.enter_context(patch.object(DailyResearchPoolStore, "publish", forbidden))
        stack.enter_context(patch.object(store, "_cas_pointer", forbidden))
        stack.enter_context(patch.object(event_store, "publish_generation", forbidden))
        stack.enter_context(patch.object(cn_benchmark_store, "publish_generation", forbidden))

    if mode == "repeat":
        before = inventory(workspace)
        with (
            synthetic_clock(datetime.fromisoformat("2026-08-27T13:20:00+00:00")),
            ExitStack() as stack,
        ):
            prohibit(stack)
            stack.enter_context(patch("socket.socket.connect", forbidden))
            result = dispatch_daily_request(
                workspace=str(workspace),
                request_ref=initial["request_ref"],
                release_install_ref=config["release_install_ref"],
                synthetic=True,
            )
        if (
            result["business_state"] != "COMPLETE"
            or result["days"][0]["completion_ref"] != initial["completion_ref"]
        ):
            raise ValueError("original request did not retain its exact completed result")
        if inventory(workspace) != before:
            raise ValueError("completed original-request repeat changed workspace files")
        proof = {
            "case": 9,
            "synthetic": True,
            "request_ref": initial["request_ref"],
            "completion_ref": initial["completion_ref"],
            "result": result,
            "maintenance_factor_and_financial_calls_forbidden": True,
            "workspace_unchanged": True,
            "production_deployed": False,
        }
    elif mode == "closed":
        from test_daily_evidence_requested_session import capture
        from quant_investor.operations.source_slot_contract import paths

        prefixes = (
            "results/strategy_records",
            "results/factors",
            "results/operations/daily_production",
            "portfolio_dashboard/private/generated",
            "data/parquet",
        )
        before = {p: inventory(workspace / p) for p in prefixes}
        day = "20260905"
        with (
            synthetic_clock(datetime.fromisoformat("2026-09-05T13:20:00+00:00")),
            ExitStack() as stack,
        ):
            prohibit(stack)
            stack.enter_context(
                patch(
                    "quant_investor.operations.source_slot_inputs.acquire_close_session_authority",
                    lambda **kw: capture(kw["now"].isoformat()),
                )
            )
            arguments = dict(
                workspace=str(workspace),
                config_ref=initial["config_ref"],
                release_install_ref=config["release_install_ref"],
                synthetic=True,
                mode="provision",
            )
            result = configured_source_inputs(**arguments)
            if result["mode"] != "NON_TRADING_DAY" or result["request_ref"] is not None:
                raise ValueError("native closed day produced an executable request")
            after_capture = inventory(workspace)
            if configured_source_inputs(**arguments, no_providers=True) != result:
                raise ValueError("closed-day no-provider repeat changed its result")
            if inventory(workspace) != after_capture:
                raise ValueError("closed-day repeat wrote new evidence")
        if {p: inventory(workspace / p) for p in prefixes} != before:
            raise ValueError("closed day changed a financial/DAG/serving state")
        if (workspace / paths(initial["config_ref"], day)["plan"]).exists():
            raise ValueError("closed day created a source plan")
        if (workspace / "results/operations/daily_production/CN" / day).exists():
            raise ValueError("closed day created a financial DAG")
        proof = {
            "case": 10,
            "synthetic": True,
            "trade_date": day,
            "result": result,
            "producer_and_financial_writers_forbidden": True,
            "financial_dag_and_serving_unchanged": True,
            "repeat_without_provider_or_writes": True,
            "production_deployed": False,
        }
    else:
        raise ValueError("unknown configured boundary case")
    (root / ("configured-" + mode + "-proof.json")).write_text(json.dumps(proof, indent=2) + "\n")
    print("PASS installed configured", mode, flush=True)
    return proof
