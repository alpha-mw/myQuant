"""Read-only journal/output integrity view; not a substitute EOD completion seal."""

from pathlib import Path
import os
import stat

from quant_investor.contracts import parse_canonical_json_bytes, validate_artifact
from quant_investor.intelligence.low_frequency import FRESHNESS_KIND
from quant_investor.system.storage import SecureSystemStorage
from .daily_journal import DailyJournal, FALSE_AUTHORITY
from .journal_revisions import selected_binding
from .daily_contract import EOD_NODE_IDS, ContractError
from .status_projection import project_node_status


def read_daily_status(workspace: str, trade_date: str) -> dict:
    journal = DailyJournal(workspace, trade_date)
    sources = SecureSystemStorage(workspace)
    nodes = {}
    for node in sorted(EOD_NODE_IDS):
        root = str(journal.root / "nodes" / node)
        try:
            binding = selected_binding(journal.storage, root)
            key = binding[0]
            if key is None:
                nodes[node] = project_node_status({"state": "NOT_STARTED"})
                continue
            request = journal.storage.read(root + "/" + key + "/request.json")
            if request is None:
                raise ContractError("JOURNAL_SELECTED_REQUEST_MISSING")
            document = parse_canonical_json_bytes(request.data)
            value = journal.readonly_inspect(document)
            freshness = None
            for name, ref in value.get("terminal", {}).get("output_refs", {}).items():
                if node == "store":
                    from quant_investor.strategy_records.store import regular_file_sha256

                    prefix = "results/strategy_records/CN/aggressive_tech_manufacturing/"
                    path = Path(workspace).resolve() / ref["path"]
                    if not ref["path"].startswith(prefix) or path.resolve(strict=True) != path:
                        raise ContractError("DAILY_STATUS_STORE_PATH_INVALID")
                    metadata = path.stat()
                    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) not in {
                        0o600,
                        0o644,
                    }:
                        raise ContractError("DAILY_STATUS_STORE_MODE_INVALID")
                    digest, _ = regular_file_sha256(path, label="daily Store output")
                else:
                    stored = sources.read_workspace_file_bytes(
                        ref["path"], maximum_bytes=64 * 1024 * 1024
                    )
                    digest = stored.byte_sha256
                if digest != ref["sha256"]:
                    raise ContractError("DAILY_STATUS_OUTPUT_SHA_MISMATCH")
                if name == FRESHNESS_KIND:
                    if node not in {"fundamental", "macro"}:
                        raise ContractError("DAILY_STATUS_FRESHNESS_NODE_INVALID")
                    body = validate_artifact(stored.data, expected_kind=FRESHNESS_KIND)["payload"]
                    if body["domain"] != node.upper() or body["trade_date"] != trade_date:
                        raise ContractError("DAILY_STATUS_FRESHNESS_DOMAIN_INVALID")
                    freshness = {
                        key: body[key]
                        for key in (
                            "as_of",
                            "domain",
                            "freshness_state",
                            "warning_codes",
                            "critical_missing_codes",
                        )
                    }
                    freshness["source_ref"] = ref
            if journal.readonly_inspect(document) != value:
                raise ContractError("JOURNAL_CHANGED_DURING_READ")
            if selected_binding(journal.storage, root) != binding:
                raise ContractError("JOURNAL_CHANGED_DURING_READ")
            nodes[node] = project_node_status(value, recorded_request=document)
            if freshness is not None:
                nodes[node][FRESHNESS_KIND] = freshness
        except Exception as exc:
            nodes[node] = project_node_status(
                {"state": "STALE", "reason": type(exc).__name__ + ":" + str(exc)}
            )
    states = {row["state"] for row in nodes.values()}
    status = (
        "NOT_STARTED"
        if states == {"NOT_STARTED"}
        else "FAILED" if states & {"STALE", "FAILED"} else "PARTIAL"
    )
    result = {
        "schema_version": "cn-daily-status-readback.v1",
        "trade_date": trade_date,
        "status": status,
        "nodes": nodes,
        "authority": FALSE_AUTHORITY,
        "validation_scope": "JOURNAL_AND_OUTPUT_BYTES",
        "completion_ref": None,
        "completion_status": "AWAITING_NATIVE_COMPLETION_VALIDATION",
    }

    try:
        from scripts.daily_dashboard_publication import observed_serving_status

        serving = observed_serving_status(workspace, trade_date)
        if serving is not None:
            result["serving"] = serving
            result["completion_status"] = serving["publication_state"]
    except Exception as exc:
        result["serving"] = {
            "publication_state": "UNCONFIRMED",
            "detail": type(exc).__name__ + ":" + str(exc),
        }
    return result
