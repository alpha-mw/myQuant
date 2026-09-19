"""Read-only Calendar-derived catch-up selection and immutable resolution replay."""

from copy import deepcopy
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes
from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.market.close_session_authority import (
    replay_close_session_authority,
    CloseSessionAuthorityError,
)
from quant_investor.market.tushare_transport import TushareHttpsError
from quant_investor.system.errors import SystemNotFound, SystemStorageError
from .automatic_catchup_contract import (
    RESOLUTION_SCHEMA,
    RESOLUTION_FIELDS,
    completion_day,
    digest,
    document_ref,
    run_path,
    validate_automatic_request,
    REQUEST_SCHEMA_V2,
)
from .catchup_binding import (
    BindingSources,
    COLLECTION_SCHEMA_V2,
    COLLECTION_SCHEMA_V3,
    PUBLICATION_POLICIES,
    check_template_sources,
    route_day,
    validate_routed_native_input,
    verify_completed_edge,
)
from .daily_contract import ContractError, EOD_NODE_IDS, utc_stamp
from .daily_journal import FALSE_AUTHORITY
from .dashboard_serving_contract import PREFIX, HEAD_JSON, HEAD_JS, head_bytes, head_js
from .execution_recipe import validate_catchup_template
from .journal_storage import JournalStorage
from .production_request import SCHEMA as EXPLICIT_SCHEMA, validate_production_request


def optional_bytes(source, path):
    try:
        return source.storage.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data
    except (FileNotFoundError, SystemNotFound):
        return None
    except SystemStorageError as exc:
        if type(exc) is SystemStorageError and isinstance(exc.__cause__, FileNotFoundError):
            return None
        raise


def _head_pair(source):
    return tuple(optional_bytes(source, f"{PREFIX}/{name}") for name in (HEAD_JSON, HEAD_JS))


def _head_record(pair, seed):
    raw, mirror = pair
    if raw is None and mirror is None:
        if seed is None:
            raise ContractError("AUTO_INITIAL_COMPLETION_SEED_REQUIRED")
        return None
    if seed is not None:
        raise ContractError("AUTO_HEAD_AND_SEED_CONFLICT")
    if raw is None or mirror is None or head_js(raw) != mirror:
        raise ContractError("AUTO_HEAD_PAIR_CONFLICT")
    return {
        "document": head_bytes(raw),
        "json_sha256": digest(raw),
        "mirror_sha256": digest(mirror),
    }


def _retained_head(value, seed):
    if value is None:
        if seed is None:
            raise ContractError("AUTO_INITIAL_COMPLETION_SEED_REQUIRED")
        return None
    if type(value) is not dict or set(value) != {"document", "json_sha256", "mirror_sha256"}:
        raise ContractError("AUTO_RETAINED_HEAD_INVALID")
    raw = canonical_json_bytes(value["document"])
    if value != _head_record((raw, head_js(raw)), seed):
        raise ContractError("AUTO_RETAINED_HEAD_INVALID")
    return value


def _replay(source, ref, request, synthetic, previous_ref=None):
    from scripts.daily_completion_replay import replay_native_completion

    day = completion_day(ref)
    replay = replay_native_completion(
        workspace=source.workspace, trade_date=day, completion_ref=ref
    )
    if (
        replay.get("native_replay_validated") is not True
        or replay.get("completion_ref") != ref
        or replay.get("trade_date") != day
        or replay.get("synthetic") is not synthetic
        or replay.get("validated_nodes") != sorted(EOD_NODE_IDS)
    ):
        raise ContractError("AUTO_NATIVE_COMPLETION_INVALID")
    value = source.document(ref)
    if any(value.get(k) != request[k] for k in ("market", "strategy_id", "graph_sha256")):
        raise ContractError("AUTO_NATIVE_COMPLETION_SCOPE_INVALID")
    inputs = source.document(value["native_inputs_ref"])
    previous = (
        inputs["previous_trade_date"] if previous_ref is None else completion_day(previous_ref)
    )
    verify_completed_edge(
        workspace=source.workspace,
        day=day,
        completion_ref=ref,
        previous=previous,
        previous_ref=previous_ref,
    )
    validated = utc_stamp(value["native_validation_completed_at"])
    return inputs, validated


def _declarations(source, request, days):
    value = source.document(request["recipe_ref"])
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "recipes", "publication_policy"}
        or value["schema_version"]
        != (
            COLLECTION_SCHEMA_V3
            if request["schema_version"] == REQUEST_SCHEMA_V2
            else COLLECTION_SCHEMA_V2
        )
        or type(value["recipes"]) is not dict
        or type(value["publication_policy"]) is not str
        or value["publication_policy"] not in PUBLICATION_POLICIES
    ):
        raise ContractError("AUTO_COLLECTION_INVALID")
    recipes, inputs = value["recipes"], request["day_input_refs"]
    if (set(recipes) | set(inputs)) - set(days) or set(recipes) & set(inputs):
        raise ContractError("AUTO_INPUT_DATE_OWNERSHIP_INVALID")
    return value


def _prefix(source, request, locator, days, synthetic, frozen):
    refs, times, anchor = [], [], locator
    storage = JournalStorage(source.workspace)
    if frozen is not None:
        if type(frozen) is not list or len(frozen) > len(days):
            raise ContractError("AUTO_PREFIX_INVALID")
        for day, ref in zip(days, frozen):
            if completion_day(ref) != day:
                raise ContractError("AUTO_PREFIX_DATE_INVALID")
            _, stamp = _replay(source, ref, request, synthetic, anchor)
            refs.append(ref)
            times.append(stamp)
            anchor = ref
        return refs, anchor, times
    missing = False
    for day in days:
        path = f"results/operations/daily_production/CN/{day}/completion.v1.json"
        stored = storage.read(path)
        if stored is None:
            missing = True
            continue
        if missing:
            raise ContractError("AUTO_COMPLETION_BEYOND_GAP")
        ref = {"path": path, "sha256": stored.byte_sha256}
        _, stamp = _replay(source, ref, request, synthetic, anchor)
        refs.append(ref)
        times.append(stamp)
        anchor = ref
    return refs, anchor, times


def _derived(
    source, request_ref, request, collection, calendar, raw, locator, anchor, days, *, frozen=False
):
    target, anchor_day = calendar["target_trade_date"], completion_day(anchor)
    remaining = [day for day in days if day > anchor_day]
    selected = {
        **deepcopy(collection),
        "recipes": {
            day: deepcopy(collection["recipes"][day])
            for day in remaining
            if day in collection["recipes"]
        },
    }
    selected_ref = document_ref(run_path(request_ref, "collection.json"), selected)
    explicit = {
        "schema_version": EXPLICIT_SCHEMA,
        **{
            key: request[key]
            for key in ("market", "strategy_id", "graph_sha256", "release_install_ref")
        },
        "action": "CATCH_UP",
        "target_trade_date": target,
        "recipe_ref": selected_ref,
        "maintenance_handoff_ref": None,
        "calendar_ref": request["calendar_ref"],
        "raw_calendar_ref": request["raw_calendar_ref"],
        "previous_completion_ref": anchor,
        "day_input_refs": {
            day: request["day_input_refs"][day]
            for day in remaining
            if day in request["day_input_refs"]
        },
    }
    validate_production_request(explicit, release_install_ref=request["release_install_ref"])
    scopes, predecessor = {}, completion_day(locator)
    for day in days:
        scopes[day] = route_day(
            day=day,
            previous=predecessor,
            request=explicit,
            collection=collection,
            calendar=calendar,
            raw=raw,
        )
        if day in remaining:
            if day in selected["recipes"]:
                template = selected["recipes"][day]
                validate_catchup_template(
                    template, request={**explicit, "action": "EXECUTE", "target_trade_date": day}
                )
                if request["schema_version"] == REQUEST_SCHEMA_V2:
                    from .research_timing import CURRENT, HISTORICAL

                    mode = CURRENT if scopes[day]["maintenance_mode"] == "CURRENT" else HISTORICAL
                    if (
                        template["schema_version"]
                        not in {"cn-daily-execute-recipe.v5", "cn-daily-execute-recipe.v6"}
                        or template["research_timing"]["mode"] != mode
                    ):
                        raise ContractError("AUTO_RECIPE_V5_TIMING_REQUIRED")
                elif template["schema_version"] != "cn-daily-execute-recipe.v4":
                    raise ContractError("AUTO_RECIPE_V4_REQUIRED")
                check_template_sources(
                    source,
                    template,
                    profile=(
                        "FROZEN_REGISTERED_RECOVERY"
                        if frozen and template["schema_version"] == "cn-daily-execute-recipe.v6"
                        else "FRESH"
                    ),
                )
            elif day in explicit["day_input_refs"]:
                native = source.document(explicit["day_input_refs"][day])
                validate_routed_native_input(
                    native,
                    day=day,
                    previous=predecessor,
                    routing=scopes[day],
                    version="v3" if request["schema_version"] == REQUEST_SCHEMA_V2 else "v2",
                )
        predecessor = day
    if target == anchor_day and target not in scopes:
        native = source.document(source.document(anchor)["native_inputs_ref"])
        scopes[target] = route_day(
            day=target,
            previous=native["previous_trade_date"],
            request=explicit,
            collection=collection,
            calendar=calendar,
            raw=raw,
        )
    missing = [
        day
        for day in remaining
        if day not in selected["recipes"] and day not in explicit["day_input_refs"]
    ]
    return selected, selected_ref, explicit, scopes, missing


def resolve_automatic_request(
    *, workspace, request_ref, release_install_ref, synthetic, frozen=None, now=None
):
    """Build a read-only candidate, or reconstruct one exact saved resolution."""
    if type(synthetic) is not bool:
        raise ContractError("AUTO_PROVENANCE_INVALID")
    source = BindingSources(workspace)
    request = validate_automatic_request(
        source.document(request_ref), release_install_ref=release_install_ref
    )
    receipt = parse_json_bytes(
        source.raw(request["calendar_ref"]), label="automatic Calendar", require_canonical=False
    )
    raw = source.raw(request["raw_calendar_ref"])
    try:
        calendar = replay_close_session_authority(receipt, raw).receipt
    except (CloseSessionAuthorityError, TushareHttpsError) as exc:
        raise ContractError("AUTO_CALENDAR_REPLAY_REJECTED") from exc
    observed = datetime.strptime(calendar["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
    initial_pair = None
    if frozen is None:
        initial_pair = _head_pair(source)
        head = _head_record(initial_pair, request["seed_completion_ref"])
    else:
        if type(frozen) is not dict or set(frozen) != RESOLUTION_FIELDS:
            raise ContractError("AUTO_RESOLUTION_FIELDS_INVALID")
        head = _retained_head(frozen["observed_head"], request["seed_completion_ref"])
    locator = request["seed_completion_ref"] if head is None else head["document"]["completion_ref"]
    first, target = completion_day(locator), calendar["target_trade_date"]
    if not (calendar["calendar_start_date"] <= first <= target <= calendar["calendar_end_date"]):
        raise ContractError("AUTO_CALENDAR_COVERAGE_INVALID")
    open_days = calendar["ordered_open_dates"]
    if first not in open_days or target not in open_days:
        raise ContractError("AUTO_LOCATOR_OR_TARGET_NOT_OPEN")
    days = [day for day in open_days if first < day <= target]
    collection = _declarations(source, request, days)
    _, first_stamp = _replay(source, locator, request, synthetic)
    adopted, anchor, times = _prefix(
        source,
        request,
        locator,
        days,
        synthetic,
        None if frozen is None else frozen["adopted_completion_refs"],
    )
    selected, selected_ref, explicit, scopes, missing = _derived(
        source,
        request_ref,
        request,
        collection,
        calendar,
        raw,
        locator,
        anchor,
        days,
        frozen=frozen is not None,
    )
    actual = now or datetime.now(timezone.utc)
    if actual.tzinfo is None or actual.utcoffset() is None:
        raise ContractError("AUTO_CLOCK_INVALID")
    if frozen is None and observed.date() != actual.astimezone(ZoneInfo("Asia/Shanghai")).date():
        raise ContractError("AUTO_FRESH_CALENDAR_REQUIRED")
    stamp = actual if frozen is None else utc_stamp(frozen["resolved_at"])
    minimum = [observed, first_stamp, *times]
    if head is not None:
        minimum.append(utc_stamp(head["document"]["registered_at"]))
    if not max(minimum) <= stamp <= actual:
        raise ContractError("AUTO_RESOLUTION_TIME_INVALID")
    resolution = {
        "schema_version": RESOLUTION_SCHEMA,
        "auto_request_ref": request_ref,
        **{
            k: request[k]
            for k in (
                "market",
                "strategy_id",
                "graph_sha256",
                "release_install_ref",
                "calendar_ref",
                "raw_calendar_ref",
            )
        },
        "observed_head": head,
        "locator_ref": locator,
        "anchor_ref": anchor,
        "adopted_completion_refs": adopted,
        "target_trade_date": target,
        "ordered_trade_dates": days,
        "day_scopes": scopes,
        "derived_collection": selected,
        "derived_collection_ref": selected_ref,
        "derived_request": explicit,
        "derived_request_ref": document_ref(run_path(request_ref, "request.json"), explicit),
        "resolved_at": stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": dict(FALSE_AUTHORITY),
    }
    if utc_stamp(resolution["resolved_at"]) < max(minimum):
        raise ContractError("AUTO_RESOLUTION_SUBSECOND_CHRONOLOGY_INVALID")
    if frozen is not None and resolution != frozen:
        raise ContractError("AUTO_RESOLUTION_DERIVATION_MISMATCH")
    source.recheck()
    if initial_pair is not None and _head_pair(source) != initial_pair:
        raise ContractError("AUTO_HEAD_CHANGED_DURING_RESOLUTION")
    return {
        "request": request,
        "resolution": resolution,
        "missing_input_dates": missing,
        "sources": source,
    }


def read_automatic_resolution(*, workspace, resolution_ref, release_install_ref, synthetic):
    source = BindingSources(workspace)
    value = source.document(resolution_ref)
    if type(value) is not dict or set(value) != RESOLUTION_FIELDS:
        raise ContractError("AUTO_RESOLUTION_FIELDS_INVALID")
    if resolution_ref["path"] != run_path(value["auto_request_ref"], "resolution.v1.json"):
        raise ContractError("AUTO_RESOLUTION_PATH_INVALID")
    result = resolve_automatic_request(
        workspace=workspace,
        request_ref=value["auto_request_ref"],
        release_install_ref=release_install_ref,
        synthetic=synthetic,
        frozen=value,
    )
    if result["missing_input_dates"]:
        raise ContractError("AUTO_RESOLUTION_INPUTS_INCOMPLETE")
    for ref_key, body_key in (
        ("derived_collection_ref", "derived_collection"),
        ("derived_request_ref", "derived_request"),
    ):
        stored = JournalStorage(workspace).read(value[ref_key]["path"])
        if stored is not None and stored.data != canonical_json_bytes(value[body_key]):
            raise ContractError("AUTO_DERIVED_FILE_CONFLICT")
    source.recheck()
    return result
