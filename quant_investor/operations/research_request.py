"""Version-selected research wrapper loader; legacy payloads retain exact semantics."""

from quant_investor.system.storage import SecureSystemStorage
from .dependency_diagnostics import DependencyInputError
from .daily_contract import ContractError, validate_ref
from .daily_journal import DailyJournal
from .exposure_completion import POLICY
from .research_file_readback import read_research_bytes, parse_research_json

WRAPPER_SCHEMA = "cn-daily-research-request.v2"
WRAPPER_SCHEMA_V3 = "cn-daily-research-request.v3"
WRAPPER_FIELDS = frozenset(
    {"schema_version", "trade_date", "native_request_ref", "cutoff_ref", "source_completion_policy"}
)


def load_research_request(*, workspace, reference):
    reference = validate_ref(reference)
    storage = SecureSystemStorage(workspace)
    raw = read_research_bytes(storage, reference, maximum_bytes=16 * 1024 * 1024)
    if raw.byte_sha256 != reference["sha256"]:
        raise DependencyInputError("RESEARCH_REQUEST_SHA_MISMATCH")
    value = parse_research_json(raw.data)
    if type(value) is not dict:
        raise DependencyInputError("RESEARCH_REQUEST_DOCUMENT_INVALID")
    if "schema_version" not in value:
        return {
            "document": value,
            "native_request_ref": reference,
            "source_completion_policy": None,
            "cutoff": None,
        }
    if (
        set(value) != WRAPPER_FIELDS
        or value["schema_version"] not in {WRAPPER_SCHEMA, WRAPPER_SCHEMA_V3}
        or value["source_completion_policy"] != POLICY
    ):
        raise DependencyInputError("RESEARCH_REQUEST_WRAPPER_INVALID")
    from .research_cutoff import read_cutoff_inputs

    journal = DailyJournal(str(workspace), value["trade_date"])
    bound = read_cutoff_inputs(journal=journal, cutoff_ref=validate_ref(value["cutoff_ref"]))
    if (
        bound["research_request_ref"] != reference
        or bound["native_request_ref"] != value["native_request_ref"]
        or bound["receipt"]["schema_version"]
        != (
            "cn-daily-research-cutoff.v2"
            if value["schema_version"] == WRAPPER_SCHEMA_V3
            else "cn-daily-research-cutoff.v1"
        )
    ):
        raise DependencyInputError("RESEARCH_REQUEST_CUTOFF_BINDING_INVALID")
    payload = read_research_bytes(
        storage, value["native_request_ref"], maximum_bytes=16 * 1024 * 1024
    )
    if payload.byte_sha256 != value["native_request_ref"]["sha256"]:
        raise DependencyInputError("RESEARCH_REQUEST_PAYLOAD_SHA_MISMATCH")
    if (
        storage.read_workspace_file_bytes(reference["path"], maximum_bytes=16 * 1024 * 1024).data
        != raw.data
    ):
        raise ContractError("RESEARCH_REQUEST_CHANGED")
    return {
        "document": parse_research_json(payload.data),
        "native_request_ref": value["native_request_ref"],
        "source_completion_policy": POLICY,
        "cutoff": bound,
    }
