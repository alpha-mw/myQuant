"""Explicit Aug28 continuation of the retained native full scenario."""

import hashlib
import json
from pathlib import Path
import sys
import quant_investor  # establish the installed package before adding test helpers


def run(
    root: Path, dependency_path: Path, helper_path: Path, day: str, *, future_calendar: bool = False
) -> dict:
    sequence = ["20260827", "20260828", "20260831", "20260901", "20260902"]
    if day not in sequence[1:]:
        raise ValueError("explicit synthetic successor date required")
    previous = sequence[sequence.index(day) - 1]
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    prior_suffix = "" if previous == "20260827" else "-" + previous
    expected_origin = json.loads((root / "fixture-receipt.json").read_text())[
        "runtime_verification"
    ]["import_origin"]
    if Path(quant_investor.__file__).resolve() != Path(expected_origin).resolve():
        raise ValueError("full native scenario must use its verified installed package")
    repository = root / "repository"
    if helper_path.resolve(strict=True) != (repository / "tests/unit").resolve(strict=True):
        raise ValueError("successor helpers must come from the pinned repository")
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
    from _native_daily_store_fixture import NativeStoreFixture
    from test_daily_evidence_research_sources import context, put
    from _native_daily_research_inputs import augment
    from quant_investor.cli.unified import (
        factor_production_rollover,
        factor_production_observe,
    )
    from quant_investor.market.cn_history_audit import run_cn_history_audit
    from quant_investor.intelligence.storage import publish_theme_policy_v2
    from quant_investor.operations.core_pool import publish_core_pool
    from scripts.daily_completion_replay import replay_native_completion
    import pandas as pd

    def save(name, value):
        path = root / (name + ("-" + day if name.startswith("full-") else "") + ".json")
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")
        print("PASS", name, flush=True)

    def ref(path):
        return {
            "path": str(path.relative_to(workspace)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    workspace = root / "factor-workspace"
    prior = json.loads((root / ("full-rollover" + prior_suffix + ".json")).read_text())
    pointer = workspace / "results/factors/_active.json"
    if (
        prior["as_of"] != previous
        or hashlib.sha256(pointer.read_bytes()).hexdigest() != prior["factor_pointer_byte_sha256"]
    ):
        raise ValueError("explicit Aug27 preimage required; inspect native recovery before retry")
    from quant_investor.operations.completion_readback import inspect_recorded_completion

    completion_ref = json.loads(
        (root / ("full-completion-ref" + prior_suffix + ".json")).read_text()
    )
    prior_completion = inspect_recorded_completion(
        workspace=str(workspace), trade_date=previous, completion_ref=completion_ref
    )["recorded_completion"]
    if (
        prior_completion["synthetic"] is not True
        or prior_completion["schema_version"] != "cn-daily-eod-completion.v2"
    ):
        raise ValueError("synthetic completion required")
    checked = replay_native_completion(
        workspace=str(workspace), trade_date=previous, completion_ref=completion_ref
    )
    if checked.get("native_replay_validated") is not True:
        raise ValueError("previous native v2 completion did not replay")

    def read_bound(reference):
        raw = (workspace / reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("successor retained control ref changed")
        return json.loads(raw)

    prior_materialization = read_bound(prior_completion["materialization_ref"])
    prior_handoff = read_bound(prior_materialization["maintenance_handoff_ref"])
    prior_recipe = read_bound(prior_handoff["recipe_ref"])
    previous_inputs = json.loads(
        (workspace / prior_completion["native_inputs_ref"]["path"]).read_bytes()
    )
    if (root / ("full-rollover-" + day + ".json")).exists():
        raise ValueError("successor already attempted; inspect exact stage recovery")
    # Attach to retained native fixture state; never call the initializing writer.
    from quant_investor.strategy_records import event_store, store
    from quant_investor.market import cn_benchmark_store

    book = object.__new__(NativeStoreFixture)
    book.project = workspace
    book.root = workspace / "results/strategy_records/CN/aggressive_tech_manufacturing"
    if store.load_registered_catalog(book.root) is None:
        raise ValueError("previous native Store unavailable")
    book.preserve_market = True
    book.policy_path = previous_inputs["store_policy_ref"]["path"]
    book.policy_sha = previous_inputs["store_policy_ref"]["sha256"]
    if hashlib.sha256((workspace / book.policy_path).read_bytes()).hexdigest() != book.policy_sha:
        raise ValueError("fixture owner policy drift")
    events = event_store.load_generation(book.root / "_event_store")
    book.closures = events["closures"]
    book.event_pointer = events["pointer_sha256"]
    book.days = [row["trade_date"] for row in book.closures]
    if not book.days or book.days[-1].replace("-", "") != previous:
        raise ValueError("event continuity does not end at previous completed day")
    book.benchmark_pointer = cn_benchmark_store.load_generation(
        workspace / "data/parquet/cn/benchmarks"
    )["pointer_sha256"]
    fixture = NativeFactorInputs(workspace / "synthetic-inputs", extra_future_sessions=3)
    book.stocks = tuple(fixture.symbols[:7])
    args = fixture.day(3 + sequence.index(day), extra_history=9)
    reader, _ = strict_market_from_factor_inputs(
        workspace,
        args,
        macro_ready_layout=True,
        pit_observed_at=iso + "T00:00:00Z",
        simulated_available_at=iso + "T07:30:00Z",
    )
    dates = pd.read_parquet(args["exchange_calendar_path"])["open_session"].tolist()
    audit, path = run_cn_history_audit(
        data_root=workspace / "data",
        output_root=workspace / "data/private/synthetic-history",
        days=100,
        end_date=day,
        allow_online=False,
        trade_dates=[d.strftime("%Y%m%d") for d in dates[-100:]],
    )
    save("successor-market-audit-" + iso, {"history": audit, "audit_path": str(path)})
    calendar = capture_synthetic_calendar(
        release_root=root,
        fixture_source=repository / "tests/unit/test_tushare_calendar_authority.py",
        cutoff=iso,
    )["capture"]
    checkpoint = maintenance(root, iso)["native_core_checkpoint"]
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
    release_ref = prior_recipe["release_ref"]
    read_bound(release_ref)
    publish_theme_policy_v2(workspace)
    future_ref = None
    if future_calendar:
        from _native_daily_calendar_fixture import publish_daily_future_proof

        future_ref = publish_daily_future_proof(
            release_root=root, workspace=workspace, trade_date=day
        )
        save("full-future-calendar-proof", future_ref)
    print("START native core", flush=True)
    core = publish_core_pool(
        workspace=str(workspace),
        trade_date=day,
        factor_pointer_sha256=rolled["factor_pointer_byte_sha256"],
        release_ref=release_ref,
        next_session_calendar_proof_ref=future_ref,
    )
    save("full-core", core)
    ctx, _ = context(
        workspace,
        pool_manifest_ref=core["terminal"]["output_refs"]["manifest.json"],
        release_ref=release_ref,
    )
    save(
        "research-inputs-" + day,
        {
            "synthetic": True,
            "request_ref": ctx.request_ref,
            "pool_ref": ctx.pool_ref,
            "release_ref": release_ref,
            "companies": ctx.companies,
            "top100_terminal_ref": core["terminal_ref"],
        },
    )
    bound = augment(root, day=day, native_fundamental=True)
    print("START shared Macro", flush=True)
    macro = build_shared_macro(workspace, iso)
    save("full-macro", macro)
    request = json.loads((workspace / bound["request_ref"]["path"]).read_text())
    macro_ref = macro["closure_ref"]
    request["company_evidence"]["macro_risk"] = {
        "classification": "CANONICAL_MACRO_READY",
        "source": macro_ref,
    }
    request["expected_trade_date"] = day
    request_ref = put(workspace, f"synthetic-research/{day}/full-request.json", request)
    store_args = book.advance(iso)
    from _native_v2_materialization_fixture import close_materialized_day

    print("START native successor handoff/materialization/v2 closure", flush=True)
    closure = close_materialized_day(
        workspace=workspace,
        request=request,
        store_args=store_args,
        core=core,
        checkpoint=checkpoint,
        loop_context=prior_recipe["factor_loop_context_ref"],
        installed_ref=prior_recipe["release_install_ref"],
        release_ref=release_ref,
        parent_pointer_sha256=prior["factor_pointer_byte_sha256"],
        previous_completion_ref=completion_ref,
    )
    save("full-v2-materialization", closure)
    status = closure["status"]
    save("full-registry-status", status)
    if status["status"] != "COMPLETE" or status["completion_ref"] is None:
        raise RuntimeError("native v2 successor incomplete")
    completion = status["completion_ref"]
    save("full-completion-ref", completion)
    return {
        "synthetic": True,
        "completed_native_days": 1,
        "completion_ref": completion,
        "full_five_day_proof": False,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            run(
                Path(sys.argv[1]),
                Path(sys.argv[2]),
                Path(sys.argv[3]),
                sys.argv[4],
                future_calendar=len(sys.argv) > 5 and sys.argv[5] == "future-calendar",
            ),
            indent=2,
        )
    )
