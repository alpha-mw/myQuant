"""Read-only core handoff integrity before the native resume runner revalidates inputs."""

from .core_pool import CORE_NODES
from .daily_contract import ContractError, GRAPH_SHA256, validate_ref
from .daily_journal import DailyJournal, _false_authority
from .daily_status import read_daily_status
from quant_investor.contracts import parse_canonical_json_bytes

FIELDS = frozenset(
    {"schema_version", "trade_date", "graph_sha256", "release_ref", "node_refs", "authority"}
)


def inspect_core_handoff(
    *, workspace: str, trade_date: str, handoff_ref: dict, release_ref: dict
) -> dict:
    validate_ref(handoff_ref)
    validate_ref(release_ref)
    journal = DailyJournal(workspace, trade_date)
    path = str(journal.root / "core-handoff.v1.json")
    if handoff_ref["path"] != path:
        raise ContractError("CORE_HANDOFF_PATH_INVALID")
    stored = journal.storage.read(path)
    if stored is None or stored.byte_sha256 != handoff_ref["sha256"]:
        raise ContractError("CORE_HANDOFF_SHA_MISMATCH")
    value = parse_canonical_json_bytes(stored.data)
    if (
        type(value) is not dict
        or set(value) != FIELDS
        or value["schema_version"] != "cn-daily-core-handoff.v1"
        or value["trade_date"] != trade_date
        or value["graph_sha256"] != GRAPH_SHA256
        or value["release_ref"] != release_ref
        or not _false_authority(value["authority"])
        or type(value["node_refs"]) is not dict
        or set(value["node_refs"]) != set(CORE_NODES)
    ):
        raise ContractError("CORE_HANDOFF_CONTRACT_INVALID")
    status = read_daily_status(workspace, trade_date)
    pointer_ref = None
    for node in CORE_NODES:
        validate_ref(value["node_refs"][node])
        row = status["nodes"][node]
        if row["state"] != "SUCCEEDED" or row.get("terminal_ref") != value["node_refs"][node]:
            raise ContractError("CORE_HANDOFF_TERMINAL_NOT_SELECTED")
        request_path = str(journal.root / "nodes" / node / row["request_key"] / "request.json")
        request_bytes = journal.storage.read(request_path)
        if request_bytes is None:
            raise ContractError("CORE_HANDOFF_REQUEST_MISSING")
        request = parse_canonical_json_bytes(request_bytes.data)
        if request["release_ref"] != release_ref:
            raise ContractError("CORE_HANDOFF_RELEASE_MISMATCH")
        selected = request["input_refs"]["factor_pointer"]
        validate_ref(selected)
        if pointer_ref is not None and pointer_ref != selected:
            raise ContractError("CORE_HANDOFF_POINTER_BINDING_MISMATCH")
        pointer_ref = selected
    after = read_daily_status(workspace, trade_date)
    if any(after["nodes"][node] != status["nodes"][node] for node in CORE_NODES):
        raise ContractError("CORE_HANDOFF_SELECTION_CHANGED_DURING_READ")
    if journal.storage.read(path) != stored:
        raise ContractError("CORE_HANDOFF_CHANGED_DURING_READ")
    return {
        "handoff_ref": dict(handoff_ref),
        "trade_date": trade_date,
        "release_ref": dict(release_ref),
        "factor_pointer_ref": pointer_ref,
        "node_terminal_refs": value["node_refs"],
        "validation_scope": "SELECTED_CORE_JOURNAL_AND_OUTPUT_BYTES",
        "native_replay_required": True,
        "execution_authorized": False,
    }
