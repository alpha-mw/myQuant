"""Immutable forward provenance for registered CURRENT automatic execution."""

from pathlib import Path

from quant_investor.contracts import canonical_json_bytes
from .automatic_catchup_contract import document_ref, RESOLUTION_FIELDS, RESOLUTION_SCHEMA
from .automatic_catchup_resolution import read_automatic_resolution
from .automatic_catchup_storage import current_automatic_origin
from .catchup_binding import BindingSources, read_catchup_binding
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority, _validate_day

SCHEMA = "cn-daily-automatic-origin.v1"
FIELDS = {
    "schema_version",
    "auto_request_ref",
    "resolution_ref",
    "derived_request_ref",
    "catchup_binding_ref",
    "execution_request_ref",
    "trade_date",
    "authority",
}


def origin_path(day, request_ref):
    _validate_day(day)
    validate_ref(request_ref)
    return (
        f"results/operations/daily_production/CN/{day}/executions/"
        f"{request_ref['sha256']}/inputs/automatic-origin.v1.json"
    )


def validate_origin(value):
    if (
        type(value) is not dict
        or set(value) != FIELDS
        or value.get("schema_version") != SCHEMA
        or not _false_authority(value["authority"])
    ):
        raise ContractError("AUTO_ORIGIN_FIELDS_INVALID")
    _validate_day(value["trade_date"])
    for name in FIELDS:
        if name.endswith("_ref"):
            validate_ref(value[name])
    return value


def _read_chain(source, value, *, synthetic):
    resolution_value = source.document(value["resolution_ref"])
    if (
        type(resolution_value) is not dict
        or set(resolution_value) != RESOLUTION_FIELDS
        or resolution_value["schema_version"] != RESOLUTION_SCHEMA
    ):
        raise ContractError("AUTO_ORIGIN_RESOLUTION_FIELDS_INVALID")
    # Handoff readers have no runtime provenance flag. Resolve it from the exact
    # prior completion, which the native resolution replay independently proves.
    prior = source.document(resolution_value["locator_ref"])
    recorded_synthetic = prior.get("synthetic") if type(prior) is dict else None
    if type(recorded_synthetic) is not bool or (
        synthetic is not None and (type(synthetic) is not bool or synthetic != recorded_synthetic)
    ):
        raise ContractError("AUTO_ORIGIN_PROVENANCE_MISMATCH")
    synthetic = recorded_synthetic
    context = read_automatic_resolution(
        workspace=source.workspace,
        resolution_ref=value["resolution_ref"],
        release_install_ref=resolution_value["release_install_ref"],
        synthetic=synthetic,
    )
    resolution = context["resolution"]
    bound = read_catchup_binding(
        workspace=source.workspace, binding_ref=value["catchup_binding_ref"]
    )
    binding, recipe = bound["binding"], bound["recipe"]
    if (
        resolution["auto_request_ref"] != value["auto_request_ref"]
        or resolution["derived_request_ref"] != value["derived_request_ref"]
        or binding["root_request_ref"] != value["derived_request_ref"]
        or binding["execution_request_ref"] != value["execution_request_ref"]
        or binding["trade_date"] != value["trade_date"]
        or binding["schema_version"] != "cn-daily-catchup-binding.v3"
        or binding["maintenance_mode"] != "CURRENT"
        or recipe["schema_version"] != "cn-daily-execute-recipe.v6"
        or bound["request"]["schema_version"] != "cn-daily-production-request.v1"
    ):
        raise ContractError("AUTO_ORIGIN_DERIVATION_MISMATCH")
    result = {
        "origin": value,
        "resolution": context,
        "bound": bound,
        "sources": source,
        "synthetic": synthetic,
    }
    recheck_origin(result)
    return result


def recheck_origin(context):
    for source in (
        context["sources"],
        context["resolution"]["sources"],
        context["bound"]["sources"],
    ):
        source.recheck()


def read_automatic_origin(*, workspace, reference, synthetic=None):
    source = BindingSources(str(Path(workspace).resolve(strict=True)))
    value = validate_origin(source.document(reference))
    if reference["path"] != origin_path(value["trade_date"], value["execution_request_ref"]):
        raise ContractError("AUTO_ORIGIN_PATH_INVALID")
    return {**_read_chain(source, value, synthetic=synthetic), "origin_ref": dict(reference)}


def require_origin_execution(
    context, *, request_ref, request, recipe=None, retained=False, sealed_at=None
):
    expected = context["origin"]["execution_request_ref"]
    if (
        (expected["sha256"] != request_ref["sha256"] if retained else expected != request_ref)
        or context["bound"]["request"] != request
        or (recipe is not None and context["bound"]["recipe"] != recipe)
    ):
        raise ContractError("AUTO_ORIGIN_EXECUTION_MISMATCH")
    if sealed_at is not None and utc_stamp(
        context["resolution"]["resolution"]["resolved_at"]
    ) > utc_stamp(sealed_at):
        raise ContractError("AUTO_ORIGIN_HANDOFF_TIME_INVALID")
    recheck_origin(context)


def require_live_origin(context, *, workspace):
    value = context["origin"]
    current = current_automatic_origin(
        workspace=workspace,
        derived_request_ref=value["derived_request_ref"],
        synthetic=context["synthetic"],
    )
    if current is None or any(value[k] != v for k, v in current.items()):
        raise ContractError("AUTO_ORIGIN_CAPABILITY_REQUIRED")
    recheck_origin(context)
    return current


def publish_automatic_origin(*, workspace, origin, synthetic=False):
    value = validate_origin(origin)
    source = BindingSources(str(Path(workspace).resolve(strict=True)))
    context = _read_chain(source, value, synthetic=synthetic)
    current = require_live_origin(context, workspace=workspace)
    journal = DailyJournal(workspace, value["trade_date"])
    path = origin_path(value["trade_date"], value["execution_request_ref"])
    reference = document_ref(path, value)
    with journal.locked():
        again = require_live_origin(context, workspace=workspace)
        if again != current:
            raise ContractError("AUTO_ORIGIN_CAPABILITY_CHANGED")
        if journal.storage.read(path) is None and (
            journal.storage.read(str(Path(path).parent.parent / "maintenance-handoff.v1.json"))
            is not None
            or journal.storage.read(str(journal.root / "completion.v1.json")) is not None
        ):
            raise ContractError("AUTO_ORIGIN_ORIGINAL_CUSTODY_MISSING")
        journal.storage.write(path, canonical_json_bytes(value))
    read_automatic_origin(workspace=workspace, reference=reference, synthetic=synthetic)
    return reference


def origin_document(*, current, binding_ref, execution_request_ref, day):
    return validate_origin(
        {
            "schema_version": SCHEMA,
            **current,
            "catchup_binding_ref": binding_ref,
            "execution_request_ref": execution_request_ref,
            "trade_date": day,
            "authority": dict(FALSE_AUTHORITY),
        }
    )
