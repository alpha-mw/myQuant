"""Configured source orchestration through existing native producers and launcher refs."""

from argparse import Namespace
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes
from quant_investor.market import cn_benchmark_store as benchmark, cn_benchmark_capture as capture
from quant_investor.strategy_records import event_store
from quant_investor.strategy_records.daily_event_source import validate_daily_closure_source
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
from quant_investor.operations.automatic_catchup_contract import validate_automatic_request
from quant_investor.operations.automatic_catchup_resolution import _replay
from quant_investor.operations.bootstrap_launch_contract import BootstrapLaunchInputs
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.maintenance_handoff import read_recorded_maintenance_handoff
from quant_investor.operations.daily_contract import EOD_NODE_IDS
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.daily_preparation import prepare_daily_request, Sources
from quant_investor.operations.daily_preparation_contract import RECORD_ROOT, PREIMAGES
from quant_investor.operations.source_slot_contract import (
    LOCATOR_SCHEMA,
    RESULT_SCHEMA,
    SourceSlotError,
    PLAN_SCHEMA_V2,
    paths,
    seal,
    validate_result,
)
from quant_investor.operations.source_slot_storage import SourceLocatorStorage
from quant_investor.operations import source_slot_inputs as inputs
from quant_investor.operations.dashboard_serving_contract import (
    PREFIX,
    HEAD_JSON,
    HEAD_JS,
    head_bytes,
    head_js,
)
from scripts import manage_cn_strategy_records as manager
from scripts.daily_completion_replay import replay_native_completion


def _result(ctx, mode, day, *, locator=None, cal=None):
    value = {
        "schema_version": RESULT_SCHEMA,
        "mode": mode,
        "config_ref": ctx.config_ref,
        "release_install_ref": ctx.install_ref,
        "trade_date": day,
        "request_ref": (
            locator["request_ref"] if locator is not None and mode == "REQUEST_AVAILABLE" else None
        ),
        "calendar_capture_ref": (
            locator["calendar_capture_ref"]
            if locator is not None
            else None if cal is None else cal["capture_ref"]
        ),
        "preparation_commitment_ref": (
            None if locator is None else locator["preparation_commitment_ref"]
        ),
        "authority": dict(FALSE_AUTHORITY),
    }
    return validate_result(value, config_ref=ctx.config_ref, release_install_ref=ctx.install_ref)


def _locator_contexts(ctx):
    storage = SourceLocatorStorage(ctx.workspace, ctx.config_ref)
    chain = storage.chain()
    selected = None
    for index, (locator, stored) in enumerate(chain):
        cal = inputs.read_calendar(ctx, locator["trade_date"])
        if cal is None or any(
            locator[k] != cal[v]
            for k, v in (
                ("calendar_capture_ref", "capture_ref"),
                ("calendar_ref", "calendar_ref"),
                ("raw_calendar_ref", "raw_calendar_ref"),
            )
        ):
            raise SourceSlotError("SOURCE_LOCATOR_CALENDAR_MISMATCH")
        plan = inputs.read_plan(ctx, cal)
        if plan is None or locator["source_plan_ref"] != plan[1]:
            raise SourceSlotError("SOURCE_LOCATOR_PLAN_MISMATCH")
        missing = (
            []
            if locator["state"] == "PREPARING_SOURCES"
            else inputs.preparation_objects(ctx, locator, cal)
        )
        if index == 0:
            selected = {
                "locator": locator,
                "locator_sha": stored.byte_sha256,
                "cal": cal,
                "plan": plan[0],
                "plan_ref": plan[1],
                "missing": missing,
            }
    return storage, selected


def _pending(ctx, selected, storage=None):
    pending = (storage or AutomaticRunStorage(ctx.workspace)).pending()
    if pending is None or pending["state"] != "ACTIVE":
        return False
    if (
        selected is None
        or selected["locator"]["state"] != "REQUEST_AVAILABLE"
        or selected["locator"]["request_ref"] != pending["auto_request_ref"]
    ):
        raise SourceSlotError(
            "SOURCE_FOREIGN_ACTIVE_REQUEST", pending_request_ref=pending["auto_request_ref"]
        )
    if selected["missing"]:
        committed = ctx.source.document(selected["locator"]["preparation_commitment_ref"])
        request = committed["objects"][-1]["document"]
    else:
        request = ctx.source.document(pending["auto_request_ref"])
    validate_automatic_request(request, release_install_ref=ctx.install_ref)
    return True


def _completed_target(ctx, selected, *, synthetic):
    loc = selected["locator"]
    stored = ctx.storage.read(
        f"results/operations/daily_production/CN/{loc['trade_date']}/completion.v1.json"
    )
    if stored is None:
        return False
    ref = {"path": str(stored.relative_path), "sha256": stored.byte_sha256}
    request = ctx.source.document(loc["request_ref"])
    if request["schema_version"] == "cn-daily-production-request.v1":
        bound = BootstrapLaunchInputs(
            workspace=ctx.workspace,
            request_ref=loc["request_ref"],
            release_install_ref=ctx.install_ref,
        )
        inspected = inspect_recorded_completion(
            workspace=ctx.workspace, trade_date=loc["trade_date"], completion_ref=ref
        )
        snapshot = inspected.get("completed_handoff_snapshot")
        if snapshot is None:
            raise SourceSlotError("SOURCE_COMPLETED_SNAPSHOT_REQUIRED")
        bound.bind(read_recorded_maintenance_handoff(snapshot))
        proof = replay_native_completion(
            workspace=ctx.workspace, trade_date=loc["trade_date"], completion_ref=ref
        )
        if (
            proof.get("native_replay_validated") is not True
            or proof.get("completion_ref") != ref
            or proof.get("trade_date") != loc["trade_date"]
            or proof.get("synthetic") is not synthetic
            or proof.get("validated_nodes") != sorted(EOD_NODE_IDS)
        ):
            raise SourceSlotError("SOURCE_COMPLETED_NATIVE_PROOF_INVALID")
        bound.recheck()
        snapshot.recheck()
    else:
        validate_automatic_request(request, release_install_ref=ctx.install_ref)
        _replay(ctx.source, ref, request, synthetic)
    # Historical EOD admission deliberately does not invoke serving freshness.
    raw, mirror = ctx.source.optional(f"{PREFIX}/{HEAD_JSON}"), ctx.source.optional(
        f"{PREFIX}/{HEAD_JS}"
    )
    if raw is None or mirror is None or head_js(raw) != mirror:
        raise SourceSlotError("SOURCE_COMPLETED_HEAD_REQUIRED")
    head_bytes(raw)
    ctx.recheck()
    return True


def _event_args(ctx, cal, plan):
    policy = ctx.config["store_policy_ref"]
    day = cal["day"]
    return Namespace(
        project_root=ctx.workspace,
        record_root=str(ctx.root / RECORD_ROOT),
        trade_date=f"{day[:4]}-{day[4:6]}-{day[6:]}",
        policy_path=policy["path"],
        policy_sha256=policy["sha256"],
        maintenance_receipt=None,
        maintenance_receipt_sha256=None,
        calendar_receipt=str(ctx.root / cal["calendar_ref"]["path"]),
        calendar_receipt_sha256=cal["calendar_ref"]["sha256"],
        raw_calendar=str(ctx.root / cal["raw_calendar_ref"]["path"]),
        raw_calendar_sha256=cal["raw_calendar_ref"]["sha256"],
        expected_event_pointer_sha256=plan["event_pointer_ref"]["sha256"],
        generation_id=plan["event_generation_id"],
    )


def _event_state(ctx, cal, plan):
    root = ctx.root / RECORD_ROOT / "_event_store"
    if plan["event_operation"] == "USE_REGISTERED_DECLARATION":
        from scripts.registered_daily_event_sources import select_daily_source

        day = cal["day"]
        selected = select_daily_source(
            workspace=ctx.workspace,
            store_pointer_ref=plan["store_pointer_ref"],
            trade_date=f"{day[:4]}-{day[4:6]}-{day[6:]}",
        )
        if selected["state"] != "REGISTERED_INTRADAY" or any(
            selected[k] != plan[k]
            for k in (
                "registered_event_declaration_ref",
                "writer_pointer_ref",
                "decision_baseline_pointer_ref",
            )
        ):
            raise SourceSlotError("SOURCE_REGISTERED_DECLARATION_CHANGED")
        actual = Sources(ctx.workspace).pin(PREIMAGES["event_pointer_ref"])
        if actual != plan["event_pointer_ref"]:
            raise SourceSlotError("SOURCE_EVENT_FOREIGN_POINTER")
        inputs.registered_predecessor(ctx, cal, plan)
        return {"state": "USE_REGISTERED_DECLARATION", "pointer_sha256": actual["sha256"]}
    if plan["event_operation"] == "NO_ACTION_EXISTING":
        loaded = event_store.load_generation(root)
        if loaded["pointer_sha256"] != plan["event_pointer_ref"]["sha256"]:
            raise SourceSlotError("SOURCE_EVENT_FOREIGN_POINTER")
        rows = [r for r in loaded["closures"] if r["trade_date"].replace("-", "") == cal["day"]]
        if len(rows) != 1:
            raise SourceSlotError("SOURCE_EVENT_DATE_MISSING")
        validate_daily_closure_source(workspace=ctx.root, closure=rows[0])
        return {"state": "NO_ACTION_EXISTING", "pointer_sha256": loaded["pointer_sha256"]}
    result = manager.inspect_planned_daily_event(_event_args(ctx, cal, plan))
    if result["state"] == "ABSENT":
        current = inputs.utc_now().astimezone(ZoneInfo("Asia/Shanghai"))
        if current.strftime("%Y%m%d") != cal["day"]:
            raise SourceSlotError(
                "SOURCE_HISTORICAL_EVENT_GENERATION_MISSING", missing_event_dates=[cal["day"]]
            )
        if (current.hour, current.minute) < (15, 30):
            raise SourceSlotError("SOURCE_EVENT_OWNER_CUTOFF_NOT_REACHED")
    return result


def _benchmark_args(ctx, plan):
    return {
        "workspace_root": ctx.root,
        "start_date": plan["benchmark_start_date"],
        "end_date": plan["benchmark_end_date"],
        "generation_id": plan["benchmark_generation_id"],
        "expected_pointer_sha256": plan["benchmark_pointer_ref"]["sha256"],
        "required_dates": plan["benchmark_required_dates"],
    }


def _benchmark_requires_provider(ctx, plan):
    current = benchmark.load_generation(ctx.root / "data/parquet/cn/benchmarks")
    if plan["benchmark_generation_id"] is None:
        if current["pointer_sha256"] != plan["benchmark_pointer_ref"]["sha256"]:
            raise SourceSlotError("SOURCE_BENCHMARK_FOREIGN_POINTER")
        return False
    args = _benchmark_args(ctx, plan)
    storage = capture.CaptureStorage(ctx.workspace)
    base = capture.PREFIX + "/" + args["generation_id"]
    if storage.read(base + "/capture.v1.json") is not None:
        raise SourceSlotError("SOURCE_BENCHMARK_CAPTURE_VERSION_CONFLICT")
    stored = storage.read(base + "/capture.v2.json")
    if stored is None:
        if current["pointer_sha256"] != args["expected_pointer_sha256"]:
            raise SourceSlotError("SOURCE_BENCHMARK_FOREIGN_POINTER")
        return True
    reader = Sources(ctx.workspace)
    capture._read_capture(
        reader,
        stored,
        start_date=args["start_date"],
        end_date=args["end_date"],
        generation_id=args["generation_id"],
        expected=args["expected_pointer_sha256"],
        required=args["required_dates"],
        partitions=capture.request_partition(args["start_date"], args["end_date"]),
    )
    reader.recheck()
    if current["pointer_sha256"] != args["expected_pointer_sha256"] and (
        current["pointer"]["generation_id"] != args["generation_id"]
        or current["pointer"]["previous_pointer_sha256"] != args["expected_pointer_sha256"]
    ):
        raise SourceSlotError("SOURCE_BENCHMARK_FOREIGN_POINTER")
    return False


def _source_mode(ctx, cal, plan):
    actual = Sources(ctx.workspace).pin(PREIMAGES["store_pointer_ref"])
    if actual != plan["store_pointer_ref"]:
        raise SourceSlotError("SOURCE_STORE_PREIMAGE_CHANGED")
    if cal["missing"]:
        return "LOCAL_PREPARATION"
    _event_state(ctx, cal, plan)
    return (
        "ACQUISITION_REQUIRED" if _benchmark_requires_provider(ctx, plan) else "LOCAL_PREPARATION"
    )


def _inspect(ctx, *, synthetic):
    _, selected = _locator_contexts(ctx)
    pending = _pending(ctx, selected)
    today = inputs.utc_now().astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    if selected is not None:
        loc, cal = selected["locator"], selected["cal"]
        if loc["state"] == "REQUEST_AVAILABLE":
            if selected["missing"] or cal["missing"]:
                return _result(ctx, "LOCAL_PREPARATION", loc["trade_date"], locator=loc), selected
            if (
                pending
                or loc["trade_date"] == today
                or not _completed_target(ctx, selected, synthetic=synthetic)
            ):
                return _result(ctx, "REQUEST_AVAILABLE", loc["trade_date"], locator=loc), selected
        else:
            return (
                _result(
                    ctx, _source_mode(ctx, cal, selected["plan"]), loc["trade_date"], locator=loc
                ),
                selected,
            )
    inputs.verify_static(ctx, today)
    cal = inputs.read_calendar(ctx, today)
    if cal is None:
        inputs.calendar_budget(ctx, today)
        return _result(ctx, "ACQUISITION_REQUIRED", today), None
    if cal["classification"] == "CONFIRMED_CLOSED":
        return (
            _result(
                ctx, "LOCAL_PREPARATION" if cal["missing"] else "NON_TRADING_DAY", today, cal=cal
            ),
            None,
        )
    existing = inputs.read_plan(ctx, cal)
    plan = inputs.build_plan(ctx, cal) if existing is None else existing[0]
    return _result(ctx, _source_mode(ctx, cal, plan), today, cal=cal), None


def _new_locator(ctx, cal, plan_ref, storage, old):
    old_sha = None if old is None else old["locator_sha"]
    value = seal(
        {
            "schema_version": LOCATOR_SCHEMA,
            "state": "PREPARING_SOURCES",
            "config_ref": ctx.config_ref,
            "trade_date": cal["day"],
            "calendar_capture_ref": cal["capture_ref"],
            "calendar_ref": cal["calendar_ref"],
            "raw_calendar_ref": cal["raw_calendar_ref"],
            "source_plan_ref": plan_ref,
            "preparation_commitment_ref": None,
            "request_ref": None,
            "previous_locator_sha256": old_sha,
            "authority": dict(FALSE_AUTHORITY),
        }
    )
    return value, old_sha


def _produce_sources(ctx, cal, plan, *, no_providers):
    # Revalidate the registered financial source before even a benchmark producer.
    if plan["event_operation"] == "USE_REGISTERED_DECLARATION":
        _event_state(ctx, cal, plan)
    requires = _benchmark_requires_provider(ctx, plan)
    if requires and no_providers:
        raise SourceSlotError("SOURCE_PROVIDER_EVIDENCE_REQUIRED")
    if plan["benchmark_generation_id"] is None:
        benchmark.repair_current_compatibility(
            workspace_root=ctx.root, expected_pointer_sha256=plan["benchmark_pointer_ref"]["sha256"]
        )
        bench_sha = plan["benchmark_pointer_ref"]["sha256"]
    else:
        bench_sha = capture.capture_benchmark_close(**_benchmark_args(ctx, plan))["pointer_sha256"]
    state = _event_state(ctx, cal, plan)
    if state["state"] == "ABSENT":
        event_sha = manager.command_publish_daily_event_closure(_event_args(ctx, cal, plan))[
            "pointer_sha256"
        ]
    elif state["state"] in {"CURRENT_CANDIDATE", "STAGED_CANDIDATE"}:
        event_sha = manager.recover_planned_daily_event(_event_args(ctx, cal, plan))[
            "pointer_sha256"
        ]
    else:
        event_sha = state["pointer_sha256"]
    refs = {
        "store_pointer_ref": plan["store_pointer_ref"],
        "event_pointer_ref": {"path": PREIMAGES["event_pointer_ref"], "sha256": event_sha},
        "benchmark_pointer_ref": {"path": PREIMAGES["benchmark_pointer_ref"], "sha256": bench_sha},
    }
    reader = Sources(ctx.workspace)
    for ref in refs.values():
        reader.raw(ref)
    reader.recheck()
    return refs


def configured_source_inputs(
    *,
    workspace,
    config_ref,
    release_install_ref,
    mode="inspect",
    no_providers=False,
    synthetic=False,
):
    ctx = inputs.SourceContext(workspace, config_ref, release_install_ref)
    result, selected = _inspect(ctx, synthetic=synthetic)
    ctx.recheck()
    if mode == "inspect":
        return result
    if mode != "provision":
        raise SourceSlotError("SOURCE_OPERATION_MODE_INVALID")
    if result["mode"] in {"REQUEST_AVAILABLE", "NON_TRADING_DAY"}:
        return result
    binding = None
    with AutomaticRunStorage(ctx.workspace).locked() as lock:
        # Fresh context avoids carrying observations across our own source writes.
        ctx = inputs.SourceContext(workspace, config_ref, release_install_ref)
        result, selected = _inspect(ctx, synthetic=synthetic)
        _pending(ctx, selected, lock)
        if result["mode"] in {"REQUEST_AVAILABLE", "NON_TRADING_DAY"}:
            return result
        day = result["trade_date"]
        if selected is not None and selected["locator"]["state"] == "REQUEST_AVAILABLE":
            cal = inputs.read_calendar(ctx, day, materialize=True)
            binding = {"locator_sha256": selected["locator_sha"], "expected_preimages": None}
        else:
            cal = inputs.read_calendar(ctx, day, materialize=True)
            if cal is None:
                if no_providers:
                    raise SourceSlotError("SOURCE_PROVIDER_EVIDENCE_REQUIRED")
                cal = inputs.capture_calendar(ctx, day)
            if cal["classification"] == "CONFIRMED_CLOSED":
                return _result(ctx, "NON_TRADING_DAY", day, cal=cal)
            existing = inputs.read_plan(ctx, cal)
            if existing is None:
                plan = inputs.build_plan(ctx, cal)
                path = paths(config_ref, day, 2 if plan["schema_version"] == PLAN_SCHEMA_V2 else 1)[
                    "plan"
                ]
                ctx.storage.write(path, canonical_json_bytes(plan))
                existing = inputs.read_plan(ctx, cal)
            plan, plan_ref = existing
            storage, active = _locator_contexts(ctx)
            if active is None or active["locator"]["trade_date"] != day:
                value, old_sha = _new_locator(ctx, cal, plan_ref, storage, active)
                published = storage.publish(value, expected_sha256=old_sha, lock=lock)
                locator_sha = published.byte_sha256
            else:
                locator_sha = active["locator_sha"]
            _source_mode(ctx, cal, plan)
            expected = _produce_sources(ctx, cal, plan, no_providers=no_providers)
            binding = {"locator_sha256": locator_sha, "expected_preimages": expected}
        lock.require_lock()
        ctx.recheck()
    prepared = prepare_daily_request(
        workspace=workspace,
        config_ref=config_ref,
        calendar_ref=cal["calendar_ref"],
        raw_calendar_ref=cal["raw_calendar_ref"],
        now=inputs.utc_now(),
        _source_binding=binding,
    )
    if prepared["status"] != "REGISTERED_INPUTS_ONLY":
        raise SourceSlotError("SOURCE_PREPARATION_RESULT_INVALID")
    final_ctx = inputs.SourceContext(workspace, config_ref, release_install_ref)
    final, _ = _inspect(final_ctx, synthetic=synthetic)
    if final["mode"] != "REQUEST_AVAILABLE" or final["request_ref"] != prepared["request_ref"]:
        raise SourceSlotError("SOURCE_REQUEST_READBACK_INVALID")
    final_ctx.recheck()
    return final
