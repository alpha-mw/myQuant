"""Prove the exact pre-maintenance preparation files before expiring an auto run."""

from pathlib import PurePosixPath

from quant_investor.contracts import parse_canonical_json_bytes

from .daily_contract import ContractError, utc_stamp
from .daily_preparation import _existing_commitment
from .source_slot_inputs import SourceContext, read_calendar, read_plan, preparation_objects
from .source_slot_storage import SourceLocatorStorage
from .source_slot_contract import paths


def prepared_only(context, day, journal, directory_names):
    auto_ref = context["resolution"]["auto_request_ref"]
    parts = PurePosixPath(auto_ref["path"]).parts
    if (
        len(parts) != 8
        or parts[:6] != ("results", "operations", "daily_production", "CN", day, "preparation")
        or parts[-1] != "request.json"
    ):
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    root = str(PurePosixPath(auto_ref["path"]).parent)
    commitment_path, stored = _existing_commitment(journal, root)
    if stored is None:
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    commitment = parse_canonical_json_bytes(stored.data)
    ctx = SourceContext(
        context["sources"].workspace,
        commitment["config_ref"],
        context["resolution"]["release_install_ref"],
    )
    cal = read_calendar(ctx, day, materialize=False)
    if cal is None or cal["missing"]:
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    selected_plan = read_plan(ctx, cal)
    if selected_plan is None:
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    plan, plan_ref = selected_plan
    selected = paths(
        ctx.config_ref, day, 2 if plan["schema_version"] == "cn-daily-source-plan.v2" else 1
    )
    if selected["root"] != root:
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    chain = SourceLocatorStorage(ctx.workspace, ctx.config_ref).chain()
    matches = [
        value
        for value, _ in chain
        if value["state"] == "REQUEST_AVAILABLE" and value["request_ref"] == auto_ref
    ]
    if len(matches) != 1:
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    locator = matches[0]
    expected = {
        "calendar_capture_ref": cal["capture_ref"],
        "calendar_ref": cal["calendar_ref"],
        "raw_calendar_ref": cal["raw_calendar_ref"],
        "source_plan_ref": plan_ref,
        "preparation_commitment_ref": {"path": commitment_path, "sha256": stored.byte_sha256},
    }
    if any(locator[key] != value for key, value in expected.items()) or preparation_objects(
        ctx, locator, cal
    ):
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    allowed = {
        commitment_path,
        selected["capture"],
        selected["calendar"],
        selected["raw"],
        selected["plan"],
        *[item["ref"]["path"] for item in commitment["objects"]],
    }
    for index in (1, 2):
        path = selected[f"marker{index}"]
        marker = journal.read(path)
        if marker is None:
            if index == 1:
                raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
            continue
        value = parse_canonical_json_bytes(marker.data)
        if (
            type(value) is not dict
            or set(value)
            != {"schema_version", "config_ref", "trade_date", "sequence", "started_at", "api_name"}
            or value["schema_version"] != "cn-daily-source-calendar-request.v1"
            or value["config_ref"] != ctx.config_ref
            or value["trade_date"] != day
            or type(value["sequence"]) is not int
            or value["sequence"] != index
            or value["api_name"] != "trade_cal"
            or utc_stamp(value["started_at"]) > utc_stamp(context["resolution"]["resolved_at"])
        ):
            raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
        allowed.add(path)
    prefix = ("preparation", parts[6])
    if set(directory_names(("preparation",))) != {parts[6]}:
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    relative = {PurePosixPath(path).relative_to(root) for path in allowed}
    if any(
        len(path.parts) > 2
        or (len(path.parts) == 2 and path.parts[0] not in {"sources", "objects"})
        for path in relative
    ):
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    if set(directory_names(prefix)) != {path.parts[0] for path in relative}:
        raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    for directory in {path.parts[0] for path in relative if len(path.parts) == 2}:
        if set(directory_names((*prefix, directory))) != {
            path.name for path in relative if len(path.parts) == 2 and path.parts[0] == directory
        }:
            raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
    for path in allowed:
        current = journal.read(path)
        if current is None:
            raise ContractError("AUTO_EXPIRY_PREPARATION_UNCONFIRMED")
        ctx.source.raw({"path": path, "sha256": current.byte_sha256})
    ctx.recheck()
    return ctx.recheck
