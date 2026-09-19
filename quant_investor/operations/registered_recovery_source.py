"""Original configured preparation custody for registered automatic recovery."""

from pathlib import PurePosixPath

from .daily_contract import ContractError, utc_stamp
from .daily_preparation import _existing_commitment, _validate_commitment
from .daily_preparation_contract import preparation_root
from .journal_storage import JournalStorage
from .source_slot_inputs import SourceContext, read_calendar, read_plan
from .source_slot_storage import SourceLocatorStorage


def verify_configured_origin(*, source, origin, native_plan):
    """Follow code-owned preparation paths and linked locator history; no scan."""
    if origin is None:
        return None
    auto_ref = origin["origin"]["auto_request_ref"]
    parts = PurePosixPath(auto_ref["path"]).parts
    if (
        len(parts) < 6
        or parts[:4] != ("results", "operations", "daily_production", "CN")
        or parts[5] != "preparation"
    ):
        return None
    if len(parts) != 8 or parts[-1] != "request.json":
        raise ContractError("REGISTERED_RECOVERY_PREPARATION_PATH_INVALID")
    root = str(PurePosixPath(auto_ref["path"]).parent)
    path, stored = _existing_commitment(JournalStorage(source.workspace), root)
    if stored is None or not path.endswith("/commitment.v2.json"):
        raise ContractError("REGISTERED_RECOVERY_PREPARATION_MISSING")
    commitment_ref = {"path": path, "sha256": stored.byte_sha256}
    commitment = source.document(commitment_ref)
    context = SourceContext(
        source.workspace,
        commitment["config_ref"],
        origin["resolution"]["resolution"]["release_install_ref"],
    )
    day = origin["origin"]["trade_date"]
    if preparation_root(context.config_ref, day) != root:
        raise ContractError("REGISTERED_RECOVERY_PREPARATION_IDENTITY_INVALID")
    cal = read_calendar(context, day, materialize=False)
    if cal is None or cal["missing"] or cal["calendar"]["target_trade_date"] != day:
        raise ContractError("REGISTERED_RECOVERY_SOURCE_CALENDAR_MISSING")
    selected = read_plan(context, cal)
    if selected is None:
        raise ContractError("REGISTERED_RECOVERY_SOURCE_PLAN_MISSING")
    plan, plan_ref = selected
    if plan["schema_version"] != "cn-daily-source-plan.v2":
        raise ContractError("REGISTERED_RECOVERY_SOURCE_PLAN_PROFILE_INVALID")
    resolution = origin["resolution"]["resolution"]
    for name in ("calendar_ref", "raw_calendar_ref"):
        if commitment[name] != cal[name] or resolution[name] != cal[name]:
            raise ContractError("REGISTERED_RECOVERY_SOURCE_CALENDAR_MISMATCH")
        source.raw(cal[name])
    objects = _validate_commitment(
        commitment,
        config=context.config,
        config_ref=context.config_ref,
        calendar=cal["calendar"],
        calendar_ref=cal["calendar_ref"],
        raw_calendar_ref=cal["raw_calendar_ref"],
    )
    if objects[-1]["ref"] != auto_ref:
        raise ContractError("REGISTERED_RECOVERY_PREPARED_REQUEST_MISMATCH")
    for item in objects:
        if source.document(item["ref"]) != item["document"]:
            raise ContractError("REGISTERED_RECOVERY_PREPARED_OBJECT_CHANGED")
    chain = SourceLocatorStorage(source.workspace, context.config_ref).chain()
    matches = []
    for locator, retained in chain:
        source.raw({"path": retained.relative_path, "sha256": retained.byte_sha256})
        if locator["state"] == "REQUEST_AVAILABLE" and locator["request_ref"] == auto_ref:
            matches.append(locator)
    if len(matches) != 1:
        raise ContractError("REGISTERED_RECOVERY_REQUEST_LOCATOR_UNCONFIRMED")
    locator = matches[0]
    expected = {
        "config_ref": context.config_ref,
        "trade_date": day,
        "calendar_capture_ref": cal["capture_ref"],
        "calendar_ref": cal["calendar_ref"],
        "raw_calendar_ref": cal["raw_calendar_ref"],
        "source_plan_ref": plan_ref,
        "preparation_commitment_ref": commitment_ref,
    }
    construction = commitment["construction"]
    if (
        any(locator[key] != value for key, value in expected.items())
        or plan["registered_event_declaration_ref"]
        != native_plan["registered_event_declaration_ref"]
        or construction["registered_event_declaration_ref"]
        != native_plan["registered_event_declaration_ref"]
        or plan["decision_baseline_pointer_ref"] != native_plan["decision_baseline_pointer_ref"]
        or construction["decision_baseline_pointer_ref"]
        != native_plan["decision_baseline_pointer_ref"]
        or construction["writer_pointer_ref"] != plan["writer_pointer_ref"]
        or native_plan["requested_target"].replace("-", "") != day
    ):
        raise ContractError("REGISTERED_RECOVERY_SOURCE_CONSTRUCTION_MISMATCH")
    for key, field in (
        ("store_pointer_ref", "store_pointer_sha256"),
        ("event_pointer_ref", "event_pointer_sha256"),
        ("benchmark_pointer_ref", "benchmark_pointer_sha256"),
    ):
        if (
            construction["store_preimages"][key]["sha256"] != native_plan["preimages"][field]
            or plan[key]["sha256"] != native_plan["preimages"][field]
        ):
            raise ContractError("REGISTERED_RECOVERY_SOURCE_PREIMAGE_MISMATCH")
    prepared = utc_stamp(commitment["prepared_at"])
    if prepared > utc_stamp(native_plan["transaction_planned_at"]) or prepared > utc_stamp(
        resolution["resolved_at"]
    ):
        raise ContractError("REGISTERED_RECOVERY_PREPARATION_TIME_INVALID")
    # Source Calendar is verified above. Its SHA is intentionally not compared
    # to the distinct Core Calendar SHA consumed by native Store planning.
    context.recheck()
    source.recheck()
    return {
        "preparation_commitment_ref": commitment_ref,
        "prepared_at": commitment["prepared_at"],
        "source_context": context,
    }
