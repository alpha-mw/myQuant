"""Calendar custody and exact source plans for the configured launcher."""

import base64
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.market.close_session_authority import (
    acquire_close_session_authority,
    replay_close_session_authority,
)
from quant_investor.market.requested_session import classify_requested_session
from quant_investor.market import cn_benchmark_store as benchmark
from quant_investor.strategy_records import store, event_store, performance
from .daily_preparation import Sources, _validate_commitment, _walk_refs
from .daily_preparation_contract import validate_config, REF_FIELDS, PREIMAGES, RECORD_ROOT
from .daily_journal import FALSE_AUTHORITY
from .daily_contract import validate_ref, utc_stamp
from .execution_controls import verify_recipe_static_controls
from .journal_storage import JournalStorage
from .research_corporate_inputs import validate_corporate_template
from .theme_acquisition import validate_theme_acquisition_policy, POLICY_SCHEMA_V3
from .source_slot_contract import (
    CALENDAR_SCHEMA,
    PLAN_SCHEMA,
    PLAN_SCHEMA_V2,
    PLAN_FIELDS,
    REGISTERED_PLAN_FIELDS,
    SourceSlotError,
    paths,
    digest,
)


def utc_now():
    return datetime.now(timezone.utc)


class SourceContext:
    def __init__(self, workspace, config_ref, install_ref):
        self.root = Path(workspace).resolve(strict=True)
        self.workspace = str(self.root)
        self.source = Sources(self.workspace)
        self.config_ref = validate_ref(config_ref)
        self.config = validate_config(self.source.document(config_ref))
        if (
            self.config["release_install_ref"] != install_ref
            or self.config["publish_current_dashboard"] is not True
        ):
            raise SourceSlotError("SOURCE_CONFIG_LAUNCH_PROFILE_INVALID")
        self.install_ref = install_ref
        self.storage = JournalStorage(self.workspace)

    def recheck(self):
        self.source.recheck()


def verify_static(ctx, day):
    cfg, source = ctx.config, ctx.source
    from quant_investor.strategy_records.daily_event_source import validate_standing_policy

    cutoff = datetime.strptime(day + "153000", "%Y%m%d%H%M%S").replace(
        tzinfo=ZoneInfo("Asia/Shanghai")
    )
    validate_standing_policy(source.document(cfg["store_policy_ref"]), at=cutoff)
    for key in REF_FIELDS - {"seed_completion_ref"}:
        if cfg[key] is not None:
            source.raw(cfg[key])
    _walk_refs(source, source.document(cfg["industry_source_ref"]))
    template = validate_corporate_template(source.document(cfg["corporate_action_template_ref"]))
    _walk_refs(source, template)
    policy = validate_theme_acquisition_policy(source.document(cfg["theme_acquisition_ref"]))
    if policy["schema_version"] != POLICY_SCHEMA_V3:
        raise SourceSlotError("SOURCE_THEME_PROFILE_INVALID")
    # A view of real static controls, never a persisted or executable recipe.
    controls = {
        "schema_version": "cn-daily-execute-recipe.v5",
        "target_trade_date": day,
        "research_timing": {
            "mode": "CURRENT_POST_ACQUISITION",
            "policy_ref": cfg["timing_policy_ref"],
        },
        **{
            key: cfg[key]
            for key in ("factor_loop_context_ref", "release_ref", "release_install_ref")
        },
        "policy_refs": {
            "research": cfg["research_policy_ref"],
            "store": cfg["store_policy_ref"],
            "prospective": None,
        },
    }
    verify_recipe_static_controls(
        workspace=ctx.workspace, recipe=controls, document=source.document
    )
    ctx.recheck()


def read_calendar(ctx, day, *, materialize=False):
    selected = paths(ctx.config_ref, day)
    stored = ctx.storage.read(selected["capture"])
    if stored is None:
        return None
    value = parse_canonical_json_bytes(stored.data)
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema_version",
            "config_ref",
            "trade_date",
            "calendar",
            "raw_response_base64",
            "authority",
        }
        or value["schema_version"] != CALENDAR_SCHEMA
        or value["config_ref"] != ctx.config_ref
        or value["trade_date"] != day
        or value["authority"] != FALSE_AUTHORITY
    ):
        raise SourceSlotError("SOURCE_CALENDAR_CAPTURE_INVALID")
    raw = base64.b64decode(value["raw_response_base64"], validate=True)
    if (
        len(raw) > 1024 * 1024
        or base64.b64encode(raw).decode("ascii") != value["raw_response_base64"]
    ):
        raise SourceSlotError("SOURCE_CALENDAR_WIRE_INVALID")
    receipt = value["calendar"]
    if receipt.get("raw_response_path") != str(ctx.root / selected["raw"]) or receipt.get(
        "raw_response_sha256"
    ) != digest(raw):
        raise SourceSlotError("SOURCE_CALENDAR_LINK_INVALID")
    calendar = replay_close_session_authority(receipt, raw).receipt
    observed = datetime.strptime(calendar["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
    if observed.strftime("%Y%m%d") != day or observed > utc_now():
        raise SourceSlotError("SOURCE_CALENDAR_OBSERVATION_INVALID")
    result = classify_requested_session(requested_trade_date=day, receipt=calendar, raw=raw)
    calendar_raw = canonical_json_bytes(receipt)
    files = [(selected["calendar"], calendar_raw), (selected["raw"], raw)]
    missing = []
    for path, expected in files:
        actual = ctx.storage.read(path)
        if actual is None:
            missing.append(path)
        elif actual.data != expected:
            raise SourceSlotError("SOURCE_CALENDAR_OBJECT_CONFLICT")
    if materialize:
        for path, expected in files:
            ctx.storage.write(path, expected)
        missing = []
    ctx.source.raw({"path": selected["capture"], "sha256": stored.byte_sha256})
    return {
        "day": day,
        "calendar": calendar,
        "raw": raw,
        "classification": result["classification"],
        "capture_ref": {"path": selected["capture"], "sha256": stored.byte_sha256},
        "calendar_ref": {"path": selected["calendar"], "sha256": digest(calendar_raw)},
        "raw_calendar_ref": {"path": selected["raw"], "sha256": digest(raw)},
        "missing": missing,
    }


def calendar_budget(ctx, day):
    selected = paths(ctx.config_ref, day)
    marker = None
    for index in (1, 2):
        path = selected[f"marker{index}"]
        stored = ctx.storage.read(path)
        if stored is None:
            if marker is None:
                marker = path
            continue
        if index == 2 and marker == selected["marker1"]:
            raise SourceSlotError("SOURCE_CALENDAR_REQUEST_HISTORY_MISSING")
        value = parse_canonical_json_bytes(stored.data)
        if (
            type(value) is not dict
            or set(value)
            != {"schema_version", "config_ref", "trade_date", "sequence", "started_at", "api_name"}
            or value["schema_version"] != "cn-daily-source-calendar-request.v1"
            or value["config_ref"] != ctx.config_ref
            or value["trade_date"] != day
            or value["sequence"] != index
            or value["api_name"] != "trade_cal"
            or utc_stamp(value["started_at"]) > utc_now()
        ):
            raise SourceSlotError("SOURCE_CALENDAR_REQUEST_MARKER_INVALID")
    if marker is None:
        raise SourceSlotError("SOURCE_CALENDAR_REQUEST_BUDGET_EXHAUSTED")
    return marker


def capture_calendar(ctx, day):
    selected = paths(ctx.config_ref, day)
    marker = calendar_budget(ctx, day)
    before = utc_now()
    if before.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != day:
        raise SourceSlotError("SOURCE_CAPTURE_DATE_CHANGED")
    sequence = 1 if marker == selected["marker1"] else 2
    ctx.storage.write(
        marker,
        canonical_json_bytes(
            {
                "schema_version": "cn-daily-source-calendar-request.v1",
                "config_ref": ctx.config_ref,
                "trade_date": day,
                "sequence": sequence,
                "started_at": before.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "api_name": "trade_cal",
            }
        ),
    )
    result = acquire_close_session_authority(now=before)
    after = utc_now()
    if after.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != day:
        raise SourceSlotError("SOURCE_CAPTURE_DATE_CHANGED")
    receipt = {**result.receipt, "raw_response_path": str(ctx.root / selected["raw"])}
    replay_close_session_authority(receipt, result.raw_response_bytes)
    observed = datetime.strptime(receipt["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
    if observed.strftime("%Y%m%d") != day or observed > after:
        raise SourceSlotError("SOURCE_CAPTURE_OBSERVATION_INVALID")
    value = {
        "schema_version": CALENDAR_SCHEMA,
        "config_ref": ctx.config_ref,
        "trade_date": day,
        "calendar": receipt,
        "raw_response_base64": base64.b64encode(result.raw_response_bytes).decode("ascii"),
        "authority": dict(FALSE_AUTHORITY),
    }
    ctx.recheck()
    ctx.storage.write(selected["capture"], canonical_json_bytes(value))
    return read_calendar(ctx, day, materialize=True)


def generation_id(kind, config_ref, day, preimage, calendar_ref):
    return f"source-{kind}-{day}-" + digest(
        canonical_json_bytes(
            {
                "config_ref": config_ref,
                "day": day,
                "preimage": preimage,
                "calendar_ref": calendar_ref,
            }
        )
    )


def read_plan(ctx, cal):
    available = [
        (v, ctx.storage.read(paths(ctx.config_ref, cal["day"], v)["plan"])) for v in (1, 2)
    ]
    existing = [(v, stored) for v, stored in available if stored is not None]
    if not existing:
        return None
    if len(existing) != 1:
        raise SourceSlotError("SOURCE_PLAN_VERSION_CONFLICT")
    version, stored = existing[0]
    value = parse_canonical_json_bytes(stored.data)
    fields = PLAN_FIELDS | (REGISTERED_PLAN_FIELDS if version == 2 else set())
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != (PLAN_SCHEMA_V2 if version == 2 else PLAN_SCHEMA)
        or value["config_ref"] != ctx.config_ref
        or value["trade_date"] != cal["day"]
        or value["calendar_capture_ref"] != cal["capture_ref"]
        or value["authority"] != FALSE_AUTHORITY
    ):
        raise SourceSlotError("SOURCE_PLAN_INVALID")
    for key, path in PREIMAGES.items():
        if validate_ref(value[key])["path"] != path:
            raise SourceSlotError("SOURCE_PLAN_PREIMAGE_PATH_INVALID")
    if version == 2:
        if (
            value["event_operation"] != "USE_REGISTERED_DECLARATION"
            or value["event_source_mode"] != "REGISTERED_FINANCIAL_TRANSITION"
            or value["event_generation_id"] is not None
            or value["registered_source_state"] != "REGISTERED_INTRADAY"
            or value["registered_store_plan_ref"] is not None
            or validate_ref(value["writer_pointer_ref"])["sha256"]
            != value["store_pointer_ref"]["sha256"]
        ):
            raise SourceSlotError("SOURCE_PLAN_REGISTERED_PROFILE_INVALID")
        for name in ("registered_event_declaration_ref", "decision_baseline_pointer_ref"):
            validate_ref(value[name])
        from scripts.registered_daily_event_sources import read_declaration

        proof = read_declaration(
            workspace=ctx.workspace, declaration_ref=value["registered_event_declaration_ref"]
        )
        declaration = proof["declaration"]
        if (
            value["decision_baseline_pointer_ref"] != declaration["baseline_store_pointer_ref"]
            or value["writer_pointer_ref"] != declaration["writer_store_pointer_ref"]
            or declaration["trade_date"].replace("-", "") != cal["day"]
        ):
            raise SourceSlotError("SOURCE_PLAN_REGISTERED_BINDING_INVALID")
        for reference in proof["source_refs"]:
            ctx.source.raw(reference)
    elif value["event_operation"] == "NO_ACTION_EXISTING":
        if value["event_generation_id"] is not None or value["event_source_mode"] is not None:
            raise SourceSlotError("SOURCE_PLAN_EVENT_INVALID")
    elif value["event_operation"] == "CREATE_CURRENT_EMPTY":
        expected = generation_id(
            "event",
            ctx.config_ref,
            cal["day"],
            value["event_pointer_ref"]["sha256"],
            cal["capture_ref"],
        )
        if (
            value["event_source_mode"] != "CALENDAR_ONLY"
            or value["event_generation_id"] != expected
        ):
            raise SourceSlotError("SOURCE_PLAN_EVENT_INVALID")
    else:
        raise SourceSlotError("SOURCE_PLAN_EVENT_INVALID")
    names = (
        "benchmark_start_date",
        "benchmark_end_date",
        "benchmark_required_dates",
        "benchmark_generation_id",
    )
    if value["benchmark_generation_id"] is None:
        if any(value[key] is not None for key in names):
            raise SourceSlotError("SOURCE_PLAN_BENCHMARK_INVALID")
    else:
        from quant_investor.market.cn_benchmark_capture import _arguments

        _arguments(
            value[names[0]],
            value[names[1]],
            value[names[3]],
            value["benchmark_pointer_ref"]["sha256"],
            value[names[2]],
        )
        expected_days = [
            f"{d[:4]}-{d[4:6]}-{d[6:]}"
            for d in cal["calendar"]["ordered_open_dates"]
            if value[names[0]].replace("-", "") <= d <= value[names[1]].replace("-", "")
        ]
        if value[names[2]] != expected_days or value[names[3]] != generation_id(
            "benchmark",
            ctx.config_ref,
            cal["day"],
            value["benchmark_pointer_ref"]["sha256"],
            cal["capture_ref"],
        ):
            raise SourceSlotError("SOURCE_PLAN_BENCHMARK_CALENDAR_MISMATCH")
    ctx.source.raw({"path": str(stored.relative_path), "sha256": stored.byte_sha256})
    return value, {"path": str(stored.relative_path), "sha256": stored.byte_sha256}


def build_plan(ctx, cal):
    from scripts.registered_daily_event_sources import select_daily_source

    source = Sources(ctx.workspace)
    refs = {key: source.pin(path) for key, path in PREIMAGES.items() if key != "event_pointer_ref"}
    event_raw = source.optional(PREIMAGES["event_pointer_ref"])
    refs["event_pointer_ref"] = {
        "path": PREIMAGES["event_pointer_ref"],
        "sha256": event_store.EMPTY_POINTER_SHA256 if event_raw is None else digest(event_raw),
    }
    day = cal["day"]
    registered = select_daily_source(
        workspace=ctx.workspace,
        store_pointer_ref=refs["store_pointer_ref"],
        trade_date=f"{day[:4]}-{day[4:6]}-{day[6:]}",
    )
    if registered["state"] == "BLOCKED_FINALIZED_V2_INPUT_CUSTODY_MISSING":
        raise SourceSlotError(registered["state"])
    for reference in registered.get("source_refs", []):
        source.raw(reference)
    loaded = store.load_registered_catalog(ctx.root / RECORD_ROOT)
    if loaded is None or loaded[1]["schema_id"] != store.CATALOG_SCHEMA_V3:
        raise SourceSlotError("SOURCE_STORE_V3_REQUIRED")
    history = performance.load_performance_history(
        ctx.root / RECORD_ROOT, loaded[1]["performance_history_ref"]
    )
    frontier = str(history["rows"][-1]["valuation_date"]).replace("-", "")
    dates, day = cal["calendar"]["ordered_open_dates"], cal["day"]
    if frontier not in dates or frontier > day:
        raise SourceSlotError("SOURCE_STORE_CALENDAR_COVERAGE_INVALID")
    needed = sorted(set([d for d in dates if frontier < d <= day] + [day]))
    events = (
        []
        if event_raw is None
        else event_store.load_generation(ctx.root / RECORD_ROOT / "_event_store")["closures"]
    )
    event_dates = {r["trade_date"].replace("-", "") for r in events}
    missing = [d for d in needed if d < day and d not in event_dates]
    if missing:
        raise SourceSlotError("SOURCE_HISTORICAL_EVENTS_MISSING", missing_event_dates=missing)
    from quant_investor.strategy_records.daily_event_source import validate_daily_closure_source

    for closure in events:
        validate_daily_closure_source(workspace=ctx.root, closure=closure)
    indices = benchmark.load_generation(ctx.root / "data/parquet/cn/benchmarks")
    keys = {(r["date"].strftime("%Y%m%d"), r["ts_code"]) for r in indices["rows"]}
    missing_bench = [
        d for d in needed if any((d, code) not in keys for code in benchmark.REQUIRED_CODES)
    ]
    value = {
        "schema_version": PLAN_SCHEMA,
        "config_ref": ctx.config_ref,
        "trade_date": day,
        "calendar_capture_ref": cal["capture_ref"],
        **refs,
        "benchmark_start_date": None,
        "benchmark_end_date": None,
        "benchmark_required_dates": None,
        "benchmark_generation_id": None,
        "event_generation_id": None,
        "event_operation": "NO_ACTION_EXISTING",
        "event_source_mode": None,
        "authority": dict(FALSE_AUTHORITY),
    }
    if registered["state"] == "REGISTERED_INTRADAY":
        value.update(
            schema_version=PLAN_SCHEMA_V2,
            registered_source_state=registered["state"],
            **{k: registered[k] for k in REGISTERED_PLAN_FIELDS - {"registered_source_state"}},
            event_operation="USE_REGISTERED_DECLARATION",
            event_source_mode="REGISTERED_FINANCIAL_TRANSITION",
        )
    elif day not in event_dates:
        value.update(
            event_operation="CREATE_CURRENT_EMPTY",
            event_source_mode="CALENDAR_ONLY",
            event_generation_id=generation_id(
                "event",
                ctx.config_ref,
                day,
                refs["event_pointer_ref"]["sha256"],
                cal["capture_ref"],
            ),
        )
    if missing_bench:
        first, last = missing_bench[0], missing_bench[-1]

        def iso(d):
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"

        value.update(
            benchmark_start_date=iso(first),
            benchmark_end_date=iso(last),
            benchmark_required_dates=[iso(d) for d in dates if first <= d <= last],
            benchmark_generation_id=generation_id(
                "benchmark",
                ctx.config_ref,
                day,
                refs["benchmark_pointer_ref"]["sha256"],
                cal["capture_ref"],
            ),
        )
    source.recheck()
    return value


def preparation_objects(ctx, locator, cal):
    from .daily_preparation import _existing_commitment, _registered_predecessor

    selected_path, stored = _existing_commitment(
        ctx.storage, paths(ctx.config_ref, cal["day"])["root"]
    )
    if stored is None or selected_path != locator["preparation_commitment_ref"]["path"]:
        raise SourceSlotError("SOURCE_COMMITMENT_PATH_MISMATCH")
    commitment = ctx.source.document(locator["preparation_commitment_ref"])
    plan, _ = read_plan(ctx, cal)
    registered = plan["schema_version"] == PLAN_SCHEMA_V2
    selected = paths(ctx.config_ref, cal["day"], 2 if registered else 1)
    if (
        locator["preparation_commitment_ref"]["path"] != selected["commitment"]
        or commitment["schema_version"]
        != f"cn-daily-preparation-commitment.v{2 if registered else 1}"
    ):
        raise SourceSlotError("SOURCE_COMMITMENT_VERSION_MISMATCH")
    if registered and (
        type(commitment.get("construction")) is not dict
        or any(
            commitment["construction"].get(k) != plan[k]
            for k in (
                "registered_event_declaration_ref",
                "decision_baseline_pointer_ref",
                "writer_pointer_ref",
            )
        )
    ):
        raise SourceSlotError("SOURCE_COMMITMENT_REGISTERED_BINDING_INVALID")
    objects = _validate_commitment(
        commitment,
        config=ctx.config,
        config_ref=ctx.config_ref,
        calendar=cal["calendar"],
        calendar_ref=cal["calendar_ref"],
        raw_calendar_ref=cal["raw_calendar_ref"],
    )
    _registered_predecessor(
        ctx.source,
        commitment["construction"],
        commitment["locator"],
        cal["calendar"],
        prepared_at=commitment["prepared_at"],
    )
    if objects[-1]["ref"] != locator["request_ref"]:
        raise SourceSlotError("SOURCE_REQUEST_COMMITMENT_MISMATCH")
    missing = []
    for item in objects:
        stored = ctx.storage.read(item["ref"]["path"])
        if stored is None:
            missing.append(item["ref"]["path"])
        elif stored.data != canonical_json_bytes(item["document"]):
            raise SourceSlotError("SOURCE_PREPARATION_OBJECT_CONFLICT")
    return missing


def registered_predecessor(ctx, cal, plan):
    from .daily_preparation import _locator, _registered_predecessor
    from .daily_preparation_contract import CONSTRUCTION_SCHEMA_V2

    construction = {
        "schema_version": CONSTRUCTION_SCHEMA_V2,
        **{
            k: plan[k]
            for k in (
                "registered_event_declaration_ref",
                "decision_baseline_pointer_ref",
                "writer_pointer_ref",
            )
        },
    }
    _registered_predecessor(
        ctx.source,
        construction,
        _locator(ctx.source, ctx.config, cal["calendar"]),
        cal["calendar"],
        prepared_at=utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
