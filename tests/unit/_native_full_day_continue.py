"""Explicit Aug28 continuation of the retained native full scenario."""

import hashlib
import json
from pathlib import Path
import sys
import quant_investor  # establish the installed package before adding test helpers


def run(root: Path, dependency_path: Path, helper_path: Path) -> dict:
    expected_origin = json.loads((root / "fixture-receipt.json").read_text())[
        "runtime_verification"
    ]["import_origin"]
    if Path(quant_investor.__file__).resolve() != Path(expected_origin).resolve():
        raise ValueError("full native scenario must use its verified installed package")
    repository = root / "repository"
    sys.path.insert(0, str(helper_path))
    sys.path.extend(
        [
            str(repository),
            str(repository / "scripts"),
            str(repository / "tests/unit"),
            str(dependency_path),
        ]
    )
    from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
    from _native_daily_calendar_fixture import capture_synthetic_calendar
    from _native_daily_maintenance_fixture import maintenance
    from _native_shared_macro_fixture import build_shared_macro
    from _native_daily_store_fixture import NativeStoreFixture, DAYS
    from test_daily_evidence_research_sources import context, put
    from _native_daily_research_inputs import augment
    from quant_investor.cli.unified import (
        factor_production_rollover,
        factor_production_observe,
    )
    from quant_investor.market.cn_history_audit import run_cn_history_audit
    from quant_investor.intelligence.storage import publish_theme_policy_v2
    from quant_investor.operations.core_pool import publish_core_pool
    from scripts.daily_production_store_adapter import prepare_store_plan
    from scripts.daily_native_inputs import load_native_inputs
    from scripts.daily_native_registry import NativeDailyRegistry
    import pandas as pd

    def save(name, value):
        path = root / (name + ("-20260828" if name.startswith("full-") else "") + ".json")
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")
        print("PASS", name, flush=True)

    def ref(path):
        return {
            "path": str(path.relative_to(workspace)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    workspace = root / "factor-workspace"
    prior = json.loads((root / "full-rollover.json").read_text())
    pointer = workspace / "results/factors/_active.json"
    if (
        prior["as_of"] != "20260827"
        or hashlib.sha256(pointer.read_bytes()).hexdigest() != prior["factor_pointer_byte_sha256"]
    ):
        raise ValueError("explicit Aug27 preimage required; inspect native recovery before retry")
    if (
        workspace
        / "results/strategy_records/CN/aggressive_tech_manufacturing/_record_store/current.v1.json"
    ).exists():
        raise ValueError("Store already initialized; do not rerun setup")
    fixture = NativeFactorInputs(workspace / "synthetic-inputs", extra_future_sessions=4)
    args = fixture.day(4, extra_history=9)
    reader, _ = strict_market_from_factor_inputs(
        workspace,
        args,
        macro_ready_layout=True,
        pit_observed_at="2026-08-28T00:00:00Z",
        simulated_available_at="2026-08-28T07:30:00Z",
    )
    dates = pd.read_parquet(args["exchange_calendar_path"])["open_session"].tolist()
    audit, path = run_cn_history_audit(
        data_root=workspace / "data",
        output_root=workspace / "data/private/synthetic-history",
        days=100,
        end_date="20260828",
        allow_online=False,
        trade_dates=[d.strftime("%Y%m%d") for d in dates[-100:]],
    )
    save("successor-market-audit-2026-08-28", {"history": audit, "audit_path": str(path)})
    calendar = capture_synthetic_calendar(
        release_root=root,
        fixture_source=repository / "tests/unit/test_tushare_calendar_authority.py",
        cutoff="2026-08-28",
    )["capture"]
    checkpoint = maintenance(root, "2026-08-28")["native_core_checkpoint"]
    print("START successor rollover", flush=True)
    rolled = factor_production_rollover(
        workspace_root=str(workspace),
        market_data_root=str(workspace / "data"),
        calendar_capture_root=calendar["capture_root"],
        expected_calendar_success_sha256=calendar["capture_success_file_ref"]["byte_sha256"],
        maintenance_receipt=checkpoint["path"],
        expected_maintenance_receipt_sha256=checkpoint["sha256"],
        expected_current_pointer_sha256=prior["factor_pointer_byte_sha256"],
    )
    save("full-rollover", rolled)
    save("full-observations", factor_production_observe(workspace_root=str(workspace)))
    release = json.loads((root / "release-input.json").read_text())["deployed_release"]
    release_ref = put(workspace, "fixtures/release.json", release)
    publish_theme_policy_v2(workspace)
    print("START native core", flush=True)
    core = publish_core_pool(
        workspace=str(workspace),
        trade_date="20260828",
        factor_pointer_sha256=rolled["factor_pointer_byte_sha256"],
        release_ref=release_ref,
    )
    save("full-core", core)
    ctx, _ = context(
        workspace,
        pool_manifest_ref=core["terminal"]["output_refs"]["manifest.json"],
        release_ref=release_ref,
    )
    save(
        "research-inputs-20260828",
        {
            "synthetic": True,
            "request_ref": ctx.request_ref,
            "pool_ref": ctx.pool_ref,
            "release_ref": release_ref,
            "companies": ctx.companies,
            "top100_terminal_ref": core["terminal_ref"],
        },
    )
    bound = augment(root, day="20260828")
    print("START shared Macro", flush=True)
    macro = build_shared_macro(workspace, "2026-08-28")
    save("full-macro", macro)
    request = json.loads((workspace / bound["request_ref"]["path"]).read_text())
    macro_ref = macro["closure_ref"]
    request["company_evidence"]["macro_risk"] = {
        "classification": "CANONICAL_MACRO_READY",
        "source": macro_ref,
    }
    request["expected_trade_date"] = "20260828"
    request_ref = put(workspace, "synthetic-research/20260828/full-request.json", request)
    book = NativeStoreFixture(workspace, stock_symbols=fixture.symbols[:7], preserve_market=True)
    for day in DAYS[:5]:
        store_args = book.advance(day)
    plan = prepare_store_plan(store_args)
    inputs = {
        "schema_version": "cn-daily-native-inputs.v1",
        "trade_date": "20260828",
        "factor_pointer_sha256": rolled["factor_pointer_byte_sha256"],
        "release_ref": release_ref,
        "research_request_ref": request_ref,
        "store_plan_ref": {"path": plan["plan_path"], "sha256": plan["plan_sha256"]},
        "store_policy_ref": {"path": store_args["policy_path"], "sha256": store_args["policy_sha"]},
        "retrospective_ref": None,
        "calendar_ref": ref(book.calendar_path),
        "market_snapshot_ref": ref(Path(reader.snapshot()["manifest_path"])),
        "benchmark_ref": ref(workspace / "portfolio_dashboard/inputs/cn_index_benchmark.csv"),
        "risk_free_ref": ref(workspace / "portfolio_dashboard/inputs/cn_govt_bond_yield.csv"),
        "previous_trade_date": "20260827",
        "adjustment_market_refs": {},
        "publish_current_dashboard": True,
    }
    input_ref = put(workspace, "fixtures/full-native-inputs-20260828.json", inputs)
    day, native_inputs = load_native_inputs(workspace=str(workspace), input_ref=input_ref)
    registry = NativeDailyRegistry(str(workspace), day, native_inputs)
    print("START full native registry", flush=True)
    with registry.runner.journal.locked():
        status = registry.runner.run_locked({}, resume=True, resolve=registry.resolve)
        save("full-registry-status", status)
        if any(row["state"] != "SUCCEEDED" for row in status["nodes"].values()):
            raise RuntimeError("full native registry incomplete; inspect full-registry-status.json")
        completion = registry.seal_completion(native_inputs_ref=input_ref, synthetic=True)
    save("full-completion-ref", completion)
    return {
        "synthetic": True,
        "completed_native_days": 1,
        "completion_ref": completion,
        "full_five_day_proof": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
