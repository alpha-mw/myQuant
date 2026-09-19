"""Real native claim producer and read-only daily-close binding; no provider calls."""

from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.maintenance_journal import DailyOperationJournal
from quant_investor.operations.slot_claim import inspect_daily_slot_claim
from quant_investor.operations.slot_claim import inspect_handoff_slot_claim
from quant_investor.operations.maintenance_handoff_contract import (
    HISTORICAL_SCHEMA,
    HANDOFF_V3_FIELDS,
)
from quant_investor.operations.daily_contract import GRAPH_SHA256
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.daily_contract import ContractError


def context(root):
    run = root / "maintenance"
    run.mkdir(mode=0o700)
    journal = DailyOperationJournal(
        run,
        root,
        now=datetime(2026, 9, 8, 20, 20, tzinfo=ZoneInfo("Asia/Shanghai")),
        slot="2020",
        mode="execute",
    )
    path = journal.path / "claim.json"
    return (
        journal,
        path,
        {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )


def inventory(root):
    return {
        str(p.relative_to(root)): (
            p.lstat().st_mode,
            p.lstat().st_mtime_ns,
            p.read_bytes() if p.is_file() else None,
        )
        for p in root.rglob("*")
    }


def test_native_claim_reader_does_not_create_or_reset_budget(tmp_path):
    journal, path, ref = context(tmp_path)
    before = inventory(tmp_path)
    value = inspect_daily_slot_claim(workspace=str(tmp_path), claim_ref=ref, run_date="20260908")
    assert value["attempt_budget"] == value["close_request_budget"] == 2
    assert value["execution_authorized"] is False and value["new_budget_granted"] is False
    assert inventory(tmp_path) == before
    assert journal.claim_ref["sha256"] == ref["sha256"]


@pytest.mark.parametrize("fault", ["date", "sha", "budget", "budget_type", "installation", "slot"])
def test_wrong_claim_identity_or_budget_rejected(tmp_path, fault):
    _, path, ref = context(tmp_path)
    day = "20260908"
    if fault == "date":
        day = "20260909"
    elif fault == "sha":
        ref["sha256"] = "a" * 64
    else:
        value = json.loads(path.read_text())
        if fault == "budget":
            value["attempt_budget"] = 3
        elif fault == "budget_type":
            value["attempt_budget"] = 2.0
        elif fault == "installation":
            value["installation"]["module"] = "/different/runtime.py"
        else:
            value["slot"] = "2100"
        raw = canonical_json_bytes(value)
        path.write_bytes(raw)
        ref["sha256"] = hashlib.sha256(raw).hexdigest()
    before = inventory(tmp_path)
    with pytest.raises(ContractError):
        inspect_daily_slot_claim(workspace=str(tmp_path), claim_ref=ref, run_date=day)
    assert inventory(tmp_path) == before


def test_native_claim_reopen_rejects_numeric_type_drift_without_rewriting(tmp_path):
    journal, path, _ = context(tmp_path)
    value = json.loads(path.read_text())
    value["attempt_budget"] = 2.0
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    from quant_investor.market.daily_maintenance import DailyMaintenanceError

    with pytest.raises(DailyMaintenanceError, match="INSTALL_OR_POLICY_DRIFT"):
        DailyOperationJournal(journal.root, tmp_path, now=journal.now, slot="2020", mode="execute")
    assert path.read_bytes() == raw


def test_handoff_claim_reader_uses_original_target_task_without_new_budget(tmp_path):
    _, _, ref = context(tmp_path)
    handoff = {key: dict(ref) for key in HANDOFF_V3_FIELDS if key.endswith("_ref")}
    handoff.update(
        schema_version=HISTORICAL_SCHEMA,
        trade_date="20260908",
        graph_sha256=GRAPH_SHA256,
        authority=FALSE_AUTHORITY,
        sealed_at="2026-09-09T00:01:00Z",
        prospective_policy_ref=None,
    )
    before = inventory(tmp_path)
    result = inspect_handoff_slot_claim(
        workspace=str(tmp_path),
        claim_ref=ref,
        handoff=handoff,
        started={"started_at": "2026-09-09T00:00:00Z"},
    )
    assert result["run_date"] == "20260908"
    assert result["new_budget_granted"] is False and result["execution_authorized"] is False
    assert inventory(tmp_path) == before
    handoff["schema_version"] = "unknown"
    with pytest.raises(ContractError, match="SCHEMA_INVALID"):
        inspect_handoff_slot_claim(
            workspace=str(tmp_path),
            claim_ref=ref,
            handoff=handoff,
            started={"started_at": "2026-09-09T00:00:00Z"},
        )
    assert inventory(tmp_path) == before
