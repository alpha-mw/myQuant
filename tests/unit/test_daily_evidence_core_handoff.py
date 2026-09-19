"""Handoff binding and read races with a controlled status-reader seam."""

from copy import deepcopy
import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations import core_handoff as handoff
from quant_investor.operations.core_pool import CORE_NODES
from quant_investor.operations.daily_contract import GRAPH_SHA256, ContractError
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY


def context(root, monkeypatch):
    journal = DailyJournal(str(root), "20260904")
    source = {"path": "release.json", "sha256": "a" * 64}
    pointer = {"path": "pointer.json", "sha256": "b" * 64}
    rows = {}
    for node in CORE_NODES:
        ref = {"path": str(journal.root / "nodes" / node / "terminal.json"), "sha256": "c" * 64}
        rows[node] = {"state": "SUCCEEDED", "terminal_ref": ref, "request_key": "key"}
        journal.storage.write(
            str(journal.root / "nodes" / node / "key/request.json"),
            canonical_json_bytes(
                {"release_ref": source, "input_refs": {"factor_pointer": pointer}}
            ),
        )
    value = {
        "schema_version": "cn-daily-core-handoff.v1",
        "trade_date": "20260904",
        "graph_sha256": GRAPH_SHA256,
        "release_ref": source,
        "node_refs": {n: rows[n]["terminal_ref"] for n in CORE_NODES},
        "authority": FALSE_AUTHORITY,
    }
    path = str(journal.root / "core-handoff.v1.json")
    raw = canonical_json_bytes(value)
    journal.storage.write(path, raw)
    monkeypatch.setattr(handoff, "read_daily_status", lambda *a: {"nodes": deepcopy(rows)})
    return (
        dict(
            workspace=str(root),
            trade_date="20260904",
            handoff_ref={"path": path, "sha256": hashlib.sha256(raw).hexdigest()},
            release_ref=source,
        ),
        rows,
    )


def test_exact_selected_core_handoff_has_no_execution_authority(tmp_path, monkeypatch):
    args, _ = context(tmp_path, monkeypatch)
    result = handoff.inspect_core_handoff(**args)
    assert set(result["node_terminal_refs"]) == set(CORE_NODES)
    assert result["native_replay_required"] is True and result["execution_authorized"] is False


@pytest.mark.parametrize("fault", ["sha", "date", "release", "selected"])
def test_invalid_or_unselected_handoff_rejected(tmp_path, monkeypatch, fault):
    args, rows = context(tmp_path, monkeypatch)
    if fault == "sha":
        args["handoff_ref"]["sha256"] = "0" * 64
    elif fault == "date":
        args["trade_date"] = "20260903"
    elif fault == "release":
        args["release_ref"] = {"path": "other.json", "sha256": "a" * 64}
    else:
        rows["top100"]["state"] = "STALE"
    with pytest.raises(ContractError):
        handoff.inspect_core_handoff(**args)


def test_selection_change_during_read_is_rejected(tmp_path, monkeypatch):
    args, rows = context(tmp_path, monkeypatch)
    count = 0

    def read(*args):
        nonlocal count
        count += 1
        value = deepcopy(rows)
        if count == 2:
            value["factor"]["state"] = "STALE"
        return {"nodes": value}

    monkeypatch.setattr(handoff, "read_daily_status", read)
    with pytest.raises(ContractError, match="SELECTION_CHANGED"):
        handoff.inspect_core_handoff(**args)
