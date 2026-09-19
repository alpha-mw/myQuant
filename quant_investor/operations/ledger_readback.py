"""Recorded EOD-v2 ledger bindings; native semantic replay remains mandatory."""

from pathlib import PurePosixPath
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, utc_stamp, validate_ref
from .daily_journal import _false_authority
from .prospective_storage import ledger_path
from .materialization_contract import validate_bound_materialization

FIELDS = frozenset(
    {
        "schema_version",
        "trade_date",
        "graph_sha256",
        "release_ref",
        "materialization_ref",
        "native_inputs_ref",
        "calendar_ref",
        "policy_ref",
        "policy_custody_at",
        "node_terminal_refs",
        "source_times",
        "core_timing",
        "node_custody",
        "synthetic",
        "recomputed",
        "classification",
        "prospective",
        "authority",
        "published_at",
    }
)


def _inspect_eod_ledger_bindings(
    *, workspace: str, completion: dict, custody: dict
) -> tuple[dict, dict]:
    reader = SecureSystemStorage(workspace)
    observed = {}

    def read(ref):
        validate_ref(ref)
        stored = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if stored.byte_sha256 != ref["sha256"]:
            raise ContractError("EOD_LEDGER_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = stored.data
        return parse_canonical_json_bytes(stored.data)

    ref = validate_ref(completion["prospective_ledger_ref"])
    if ref["path"] != ledger_path(completion["trade_date"]):
        raise ContractError("EOD_LEDGER_PATH_INVALID")
    ledger = read(ref)
    if (
        type(ledger) is not dict
        or set(ledger) != FIELDS
        or ledger["schema_version"] != "cn-daily-evidence-ledger.v1"
        or not _false_authority(ledger["authority"])
        or type(ledger["synthetic"]) is not bool
        or type(ledger["recomputed"]) is not bool
        or type(ledger["prospective"]) is not bool
        or ledger["classification"]
        not in {"CONTEMPORANEOUS", "LATE_REGISTERED", "UNKNOWN_LEGACY", "RETROSPECTIVE_RECOMPUTE"}
    ):
        raise ContractError("EOD_LEDGER_CONTRACT_INVALID")
    for field in (
        "trade_date",
        "graph_sha256",
        "release_ref",
        "native_inputs_ref",
        "materialization_ref",
        "node_terminal_refs",
        "synthetic",
    ):
        if canonical_json_bytes(ledger[field]) != canonical_json_bytes(completion[field]):
            raise ContractError("EOD_LEDGER_BINDING_INVALID:" + field)
    eligible = ledger["classification"] == "CONTEMPORANEOUS"
    if ledger["prospective"] is not eligible or (
        eligible and (ledger["synthetic"] or ledger["recomputed"])
    ):
        raise ContractError("EOD_LEDGER_CLASSIFICATION_INVALID")
    expected_state = "LEDGER_ELIGIBLE" if eligible else "LEDGER_INELIGIBLE"
    if completion["prospective_admission_state"] != expected_state:
        raise ContractError("EOD_LEDGER_ADMISSION_STATE_MISMATCH")
    if canonical_json_bytes(ledger["node_custody"]) != canonical_json_bytes(custody):
        raise ContractError("EOD_LEDGER_CUSTODY_MISMATCH")
    materialization = read(completion["materialization_ref"])
    if (
        materialization["native_inputs_ref"] != completion["native_inputs_ref"]
        or materialization["trade_date"] != completion["trade_date"]
        or materialization["graph_sha256"] != completion["graph_sha256"]
    ):
        raise ContractError("EOD_LEDGER_MATERIALIZATION_MISMATCH")
    handoff = read(materialization["maintenance_handoff_ref"])
    execution = PurePosixPath(materialization["maintenance_handoff_ref"]["path"]).parent
    recipe = read(handoff["recipe_ref"])
    validate_bound_materialization(
        workspace=workspace,
        value=materialization,
        recipe=recipe,
        execution=execution,
        path=completion["materialization_ref"]["path"],
    )
    if ledger["calendar_ref"] != handoff["calendar_ref"] or ledger["policy_ref"] != handoff.get(
        "prospective_policy_ref"
    ):
        raise ContractError("EOD_LEDGER_POLICY_CALENDAR_MISMATCH")
    if ledger["policy_custody_at"] != (
        handoff["sealed_at"] if ledger["policy_ref"] is not None else None
    ):
        raise ContractError("EOD_LEDGER_POLICY_CUSTODY_MISMATCH")
    if ledger["policy_ref"] is not None:
        read(ledger["policy_ref"])
    published = utc_stamp(ledger["published_at"])
    if (
        not utc_stamp(handoff["sealed_at"])
        <= utc_stamp(materialization["sealed_at"])
        <= published
        <= utc_stamp(completion["native_validation_completed_at"])
    ):
        raise ContractError("EOD_LEDGER_CHRONOLOGY_INVALID")
    if any(utc_stamp(row["completed_at"]) > published for row in custody["nodes"].values()):
        raise ContractError("EOD_LEDGER_PREMATURE_PUBLICATION")
    for path, raw in observed.items():
        if reader.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != raw:
            raise ContractError("EOD_LEDGER_CHANGED_DURING_READ")
    return ledger, observed


def inspect_eod_ledger_bindings(*, workspace: str, completion: dict, custody: dict) -> dict:
    return _inspect_eod_ledger_bindings(
        workspace=workspace, completion=completion, custody=custody
    )[0]
