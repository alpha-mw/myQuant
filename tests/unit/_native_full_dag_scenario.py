"""Fresh full native synthetic case, run by its isolated installed interpreter."""

import hashlib
import json
from pathlib import Path
import sys
import quant_investor  # establish the installed package before adding test helpers


def run(
    root: Path,
    dependency_path: Path,
    *,
    future_calendar: bool = False,
    initial_execute: bool = False,
    initial_theme: bool = False,
    initial_full: bool = False,
    cutoff_profile: bool = False,
    switch_to_actual_clock=None,
    pre_core_entry=None,
    before_current_market=None,
) -> dict:
    expected_origin = json.loads((root / "fixture-receipt.json").read_text())[
        "runtime_verification"
    ]["import_origin"]
    if Path(quant_investor.__file__).resolve() != Path(expected_origin).resolve():
        raise ValueError("full native scenario must use its verified installed package")
    repository = root / "repository"
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
        factor_production_activate,
        factor_production_rollover,
        factor_production_observe,
    )
    from quant_investor.market.cn_history_audit import run_cn_history_audit
    from quant_investor.intelligence.storage import publish_theme_policy_v2
    from quant_investor.operations.core_pool import publish_core_pool
    import pandas as pd

    def save(name, value):
        path = root / (name + ".json")
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")
        print("PASS", name, flush=True)

    def ref(path):
        return {
            "path": str(path.relative_to(workspace)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    workspace = root / "factor-workspace"
    if workspace.exists():
        raise ValueError("fresh scenario required; inspect stage recovery rather than restart")
    fixture = NativeFactorInputs(workspace / "synthetic-inputs", extra_future_sessions=3)
    for offset in range(3):
        args = fixture.day(offset, extra_history=9)
        day = args["as_of"]
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        strict_market_from_factor_inputs(
            workspace,
            args,
            macro_ready_layout=True,
            pit_observed_at=iso + "T00:00:00Z",
            simulated_available_at=iso + "T07:30:00Z",
        )
    print("START baseline Calendar/activation", flush=True)
    calendar = capture_synthetic_calendar(
        release_root=root,
        fixture_source=repository / "tests/unit/test_tushare_calendar_authority.py",
        cutoff="2026-08-26",
    )["capture"]
    from _native_initial_execution_fixture import synthetic_factor_clock

    with synthetic_factor_clock(initial_theme or initial_full):
        activated = factor_production_activate(
            workspace_root=str(workspace),
            market_data_root=str(workspace / "data"),
            calendar_capture_root=calendar["capture_root"],
            expected_calendar_success_sha256=calendar["capture_success_file_ref"]["byte_sha256"],
            expected_empty=True,
        )
    save("full-baseline-activation", activated)
    if before_current_market is not None:
        before_current_market(root, fixture, activated)
    args = fixture.day(3, extra_history=9)
    reader, _ = strict_market_from_factor_inputs(
        workspace,
        args,
        macro_ready_layout=True,
        pit_observed_at="2026-08-27T00:00:00Z",
        simulated_available_at="2026-08-27T07:30:00Z",
    )
    dates = pd.read_parquet(args["exchange_calendar_path"])["open_session"].tolist()
    audit, path = run_cn_history_audit(
        data_root=workspace / "data",
        output_root=workspace / "data/private/synthetic-history",
        days=100,
        end_date="20260827",
        allow_online=False,
        trade_dates=[d.strftime("%Y%m%d") for d in dates[-100:]],
    )
    save("successor-market-audit-2026-08-27", {"history": audit, "audit_path": str(path)})
    if pre_core_entry is not None:
        return pre_core_entry(root, fixture, activated)
    if initial_execute or initial_theme or initial_full:
        from _native_initial_execution_fixture import run_initial_boundary

        return run_initial_boundary(
            root,
            fixture,
            activated,
            theme_probe=initial_theme or initial_full,
            source_complete=initial_full,
        )
    calendar = capture_synthetic_calendar(
        release_root=root,
        fixture_source=repository / "tests/unit/test_tushare_calendar_authority.py",
        cutoff="2026-08-27",
    )["capture"]
    checkpoint = maintenance(root, "2026-08-27")["native_core_checkpoint"]
    print("START successor rollover", flush=True)
    rolled = factor_production_rollover(
        workspace_root=str(workspace),
        market_data_root=str(workspace / "data"),
        calendar_capture_root=calendar["capture_root"],
        expected_calendar_success_sha256=calendar["capture_success_file_ref"]["byte_sha256"],
        maintenance_receipt=checkpoint["path"],
        expected_maintenance_receipt_sha256=checkpoint["sha256"],
        expected_current_pointer_sha256=activated["factor_pointer_byte_sha256"],
    )
    save("full-rollover", rolled)
    observed = factor_production_observe(workspace_root=str(workspace))
    save("full-observations", observed)
    from quant_investor.market.daily_factor_loop import DailyFactorLoop

    installed_ref = put(
        workspace,
        "fixtures/loop-installation.json",
        json.loads((root / "release-input.json").read_text()),
    )
    loop_context = put(
        workspace,
        "fixtures/loop-context.json",
        {
            "schema_version": "cn-daily-factor-loop.v1",
            "release_install_input_ref": installed_ref,
            "release_repository_root": str(repository),
            "release_commit": json.loads((root / "fixture-receipt.json").read_bytes())["commit"],
            "calendar_capture_parent": str(
                workspace / "data/private/native-loop-calendar-captures"
            ),
        },
    )
    loop = DailyFactorLoop(
        workspace_root=str(workspace),
        run_root=str(workspace / "data/private/native-loop-fixture"),
        context_path=str(workspace / loop_context["path"]),
        context_sha256=loop_context["sha256"],
    )
    release_ref = loop._core_release_ref()
    publish_theme_policy_v2(workspace)
    future_ref = None
    if future_calendar:
        from _native_daily_calendar_fixture import publish_daily_future_proof

        future_ref = publish_daily_future_proof(
            release_root=root, workspace=workspace, trade_date="20260827"
        )
        save("full-future-calendar-proof", future_ref)
    print("START native core", flush=True)
    if future_calendar:
        core = publish_core_pool(
            workspace=str(workspace),
            trade_date="20260827",
            factor_pointer_sha256=rolled["factor_pointer_byte_sha256"],
            release_ref=release_ref,
            next_session_calendar_proof_ref=future_ref,
        )
    else:
        core = loop._recover_core(
            {
                "core_observation_refs": {
                    row["factor_alias"]: {
                        "path": row["observation_path"],
                        "sha256": row["observation_sha256"],
                    }
                    for row in observed["observations"]
                }
            }
        )
    save("full-core", core)
    ctx, _ = context(
        workspace,
        pool_manifest_ref=core["terminal"]["output_refs"]["manifest.json"],
        release_ref=release_ref,
    )
    save(
        "research-inputs-20260827",
        {
            "synthetic": True,
            "request_ref": ctx.request_ref,
            "pool_ref": ctx.pool_ref,
            "release_ref": release_ref,
            "companies": ctx.companies,
            "top100_terminal_ref": core["terminal_ref"],
        },
    )
    bound = augment(root, native_fundamental=True)
    print("START shared Macro", flush=True)
    macro = build_shared_macro(workspace, "2026-08-27")
    save("full-macro", macro)
    request = json.loads((workspace / bound["request_ref"]["path"]).read_text())
    macro_ref = macro["closure_ref"]
    request["company_evidence"]["macro_risk"] = {
        "classification": "CANONICAL_MACRO_READY",
        "source": macro_ref,
    }
    request["expected_trade_date"] = "20260827"
    request_ref = put(workspace, "synthetic-research/20260827/full-request.json", request)
    book = NativeStoreFixture(workspace, stock_symbols=fixture.symbols[:7], preserve_market=True)
    for day in DAYS[:4]:
        store_args = book.advance(day)
    from _native_v2_materialization_fixture import close_materialized_day

    print("START native handoff/materialization/v2 closure", flush=True)
    closure = close_materialized_day(
        workspace=workspace,
        request=request,
        store_args=store_args,
        core=core,
        checkpoint=checkpoint,
        loop_context=loop_context,
        installed_ref=installed_ref,
        release_ref=release_ref,
        parent_pointer_sha256=activated["factor_pointer_byte_sha256"],
        cutoff_profile=cutoff_profile,
        book=book,
        switch_to_actual_clock=switch_to_actual_clock,
    )
    save("full-v2-materialization", closure)
    status = closure["status"]
    save("full-registry-status", status)
    if status["status"] != "COMPLETE" or status["completion_ref"] is None:
        raise RuntimeError("full native registry incomplete; inspect full-registry-status.json")
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
                future_calendar=len(sys.argv) > 3 and sys.argv[3] == "future-calendar",
                initial_execute=len(sys.argv) > 3 and sys.argv[3] == "initial-execute",
                initial_theme=len(sys.argv) > 3 and sys.argv[3] == "initial-theme",
                initial_full=len(sys.argv) > 3 and sys.argv[3] == "initial-full",
            ),
            indent=2,
        )
    )
