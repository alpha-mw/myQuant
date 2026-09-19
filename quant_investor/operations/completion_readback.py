"""Read-only recorded EOD integrity, before native replay or any consumer admission."""

from datetime import datetime, timezone
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import (
    ContractError,
    EOD_NODE_IDS,
    GRAPH,
    GRAPH_SHA256,
    utc_stamp,
    validate_ref,
)
from .daily_journal import DailyJournal, _false_authority
from .daily_status import read_daily_status

FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "market",
        "strategy_id",
        "trade_date",
        "graph_sha256",
        "release_ref",
        "native_inputs_ref",
        "node_terminal_refs",
        "synthetic",
        "prospective_admission_state",
        "authority",
        "native_validation_completed_at",
    }
)
V2_FIELDS = FIELDS | {"materialization_ref", "prospective_ledger_ref"}


def inspect_recorded_completion(*, workspace: str, trade_date: str, completion_ref: dict) -> dict:
    """No locks/writes; a valid recorded closure does not itself prove native replay."""
    ref = validate_ref(completion_ref)
    journal = DailyJournal(workspace, trade_date)
    if ref["path"] != str(journal.root / "completion.v1.json"):
        raise ContractError("EOD_READBACK_PATH_INVALID")
    stored = journal.storage.read(ref["path"])
    if stored is None or stored.byte_sha256 != ref["sha256"]:
        raise ContractError("EOD_READBACK_SHA_MISMATCH")
    value = parse_canonical_json_bytes(stored.data, label="recorded EOD")
    if (
        type(value) is not dict
        or value.get("schema_version")
        not in {"cn-daily-eod-completion.v1", "cn-daily-eod-completion.v2"}
        or set(value)
        != (V2_FIELDS if value.get("schema_version") == "cn-daily-eod-completion.v2" else FIELDS)
        or value["status"] != "SUCCEEDED"
        or value["market"] != "CN"
        or value["strategy_id"] != "aggressive_tech_manufacturing"
        or value["trade_date"] != trade_date
        or value["graph_sha256"] != GRAPH_SHA256
        or type(value["synthetic"]) is not bool
        or not _false_authority(value["authority"])
        or value["prospective_admission_state"]
        not in (
            {"LEDGER_ELIGIBLE", "LEDGER_INELIGIBLE"}
            if value["schema_version"] == "cn-daily-eod-completion.v2"
            else {"NOT_CLAIMED"}
        )
        or type(value["node_terminal_refs"]) is not dict
        or set(value["node_terminal_refs"]) != EOD_NODE_IDS
    ):
        raise ContractError("EOD_READBACK_CONTRACT_INVALID")
    completed = utc_stamp(value["native_validation_completed_at"])
    if completed > datetime.now(timezone.utc):
        raise ContractError("EOD_READBACK_FUTURE_TIME")
    source = SecureSystemStorage(workspace)
    for key in ("release_ref", "native_inputs_ref"):
        item = validate_ref(value[key])
        raw = source.read_workspace_file_bytes(item["path"], maximum_bytes=4 * 1024 * 1024)
        if raw.byte_sha256 != item["sha256"]:
            raise ContractError("EOD_READBACK_INPUT_SHA_MISMATCH")
    status = read_daily_status(workspace, trade_date)
    for node, terminal_ref in value["node_terminal_refs"].items():
        validate_ref(terminal_ref)
        row = status["nodes"][node]
        if row["state"] != "SUCCEEDED" or row.get("terminal_ref") != terminal_ref:
            raise ContractError("EOD_READBACK_TERMINAL_CHANGED:" + node)
        request_path = str(journal.root / "nodes" / node / row["request_key"] / "request.json")
        request_stored = journal.storage.read(request_path)
        if request_stored is None:
            raise ContractError("EOD_READBACK_REQUEST_MISSING")
        request = parse_canonical_json_bytes(request_stored.data, label="EOD node request")
        if (
            request["release_ref"] != value["release_ref"]
            or request["trade_date"] != trade_date
            or request["node_id"] != node
            or request["graph_sha256"] != GRAPH_SHA256
        ):
            raise ContractError("EOD_READBACK_REQUEST_BINDING_INVALID")
        spec = next(item for item in GRAPH if item.node_id == node)
        upstream = {
            key: item for key, item in request["input_refs"].items() if key.startswith("upstream.")
        }
        expected_upstream = {
            "upstream." + parent: value["node_terminal_refs"][parent] for parent in spec.requires
        }
        if upstream != expected_upstream:
            raise ContractError("EOD_READBACK_UPSTREAM_BINDING_INVALID")
        if utc_stamp(row["terminal"]["finished_at"]) > completed:
            raise ContractError("EOD_READBACK_PREMATURE_SEAL")
    if journal.storage.read(ref["path"]) != stored:
        raise ContractError("EOD_READBACK_CHANGED_DURING_READ")
    from .daily_timing import recorded_daily_timing

    timing = recorded_daily_timing(
        trade_date=trade_date,
        nodes=status["nodes"],
        terminal_refs=value["node_terminal_refs"],
        verified_at=value["native_validation_completed_at"],
    )
    snapshot = None
    if value["schema_version"] == "cn-daily-eod-completion.v2":
        from .ledger_readback import _inspect_eod_ledger_bindings

        ledger, observed = _inspect_eod_ledger_bindings(
            workspace=workspace, completion=value, custody=timing
        )
        snapshot = _completed_handoff_snapshot(
            workspace=workspace,
            completion_ref=ref,
            completion_raw=stored.data,
            completion=value,
            ledger=ledger,
            observed=observed,
        )
        if journal.storage.read(ref["path"]) != stored:
            raise ContractError("EOD_READBACK_CHANGED_DURING_READ")
    result = {
        "completion_ref": ref,
        "recorded_completion": value,
        "recorded_custody_timing": timing,
        "validation_scope": "RECORDED_JOURNAL_AND_OUTPUT_BYTES",
        "native_replay_required": True,
        "consumer_admission": False,
    }
    if snapshot is not None:
        result["completed_handoff_snapshot"] = snapshot
    return result


def _completed_handoff_snapshot(
    *, workspace, completion_ref, completion_raw, completion, ledger, observed
):
    """Called only after this module validates the complete immutable v2 binding."""
    import hashlib
    from .completed_handoff_snapshot import _mint_snapshot

    reader = SecureSystemStorage(workspace)
    docs = {}

    def capture(role, ref):
        checked = validate_ref(ref)
        raw = observed.get(checked["path"])
        if raw is None:
            raw = reader.read_workspace_file_bytes(
                checked["path"], maximum_bytes=32 * 1024 * 1024
            ).data
        if hashlib.sha256(raw).hexdigest() != checked["sha256"]:
            raise ContractError("COMPLETED_SNAPSHOT_SOURCE_SHA_MISMATCH")
        docs[role] = (role, checked["path"], checked["sha256"], raw)
        return parse_canonical_json_bytes(raw)

    docs["completion"] = (
        "completion",
        completion_ref["path"],
        completion_ref["sha256"],
        completion_raw,
    )
    capture("ledger", completion["prospective_ledger_ref"])
    materialization = capture("materialization", completion["materialization_ref"])
    handoff = capture("handoff", materialization["maintenance_handoff_ref"])
    from .maintenance_handoff_contract import validate_handoff_shape, AUTOMATIC_SCHEMA

    validate_handoff_shape(handoff)
    if handoff["schema_version"] == AUTOMATIC_SCHEMA:
        capture("automatic_origin", handoff["automatic_origin_ref"])
    recipe = capture("recipe", handoff["recipe_ref"])
    context = capture("loop_context", recipe["factor_loop_context_ref"])
    capture("release_install_input", handoff["release_install_ref"])
    capture("logical_claim", handoff["logical_claim_ref"])
    if (
        handoff["trade_date"] != completion["trade_date"]
        or recipe["target_trade_date"] != completion["trade_date"]
        or handoff["graph_sha256"] != completion["graph_sha256"]
        or recipe["graph_sha256"] != completion["graph_sha256"]
        or handoff["release_ref"] != completion["release_ref"]
        or recipe["release_ref"] != completion["release_ref"]
        or recipe["release_install_ref"] != handoff["release_install_ref"]
        or context["release_install_input_ref"] != handoff["release_install_ref"]
        or ledger["core_timing"]["factor_pointer_ref"]["sha256"]
        != handoff["factor_pointer_ref"]["sha256"]
    ):
        raise ContractError("COMPLETED_SNAPSHOT_CROSS_BINDING_INVALID")
    snapshot = _mint_snapshot(
        workspace=workspace,
        trade_date=completion["trade_date"],
        documents=tuple(docs[k] for k in sorted(docs)),
    )
    snapshot.recheck()
    return snapshot
