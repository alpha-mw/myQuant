"""Proof custody tests; native capture replay has separate installed-fixture proof."""

import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.market import next_session_proof as proof
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import ContractError


def setup(tmp_path, monkeypatch):
    journal = DailyJournal(str(tmp_path), "20250610")
    base = journal.root / "calendar-future/captures/cap"

    def leaf(name, value):
        raw = canonical_json_bytes(value)
        journal.storage.write(str(base / name), raw)
        return {"relative_path": "cap/" + name, "byte_sha256": hashlib.sha256(raw).hexdigest()}

    execution = {
        "payload": {
            "deployed_release_ref": {"byte_sha256": "a" * 64},
            "network_call_count": 4,
            "observed_started_at": "2025-06-10T07:00:00Z",
            "observed_completed_at": "2025-06-10T07:00:01Z",
        }
    }
    success = {"payload": {"observed_completed_at": "2025-06-10T07:00:02Z"}}
    er, sr = leaf("capture-execution.json", execution), leaf("capture-success.json", success)
    info = {
        "eod_trade_date": "20250610",
        "observed_through_date": "20250701",
        "next_open_session": "20250611",
        "capture_root_ref": {},
        "transaction_ref": {},
        "execution_ref": er,
        "success_ref": sr,
        "provider_capture_refs": [],
        "raw_refs": [],
        "projection_sha256": "b" * 64,
        "policy_ref": {},
        "capability_ref": {},
        "source_limitations": ["synthetic fixture"],
        "execution": execution,
        "success": success,
        "projection": [],
        "calendar_policy": {},
    }
    monkeypatch.setattr(proof, "inspect_next_session_capture", lambda **kw: info)
    return journal, {
        "workspace": str(tmp_path),
        "eod_trade_date": "20250610",
        "execution": execution,
        "execution_ref": er,
        "success": success,
        "success_ref": sr,
    }


def inventory(root):
    return {
        str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
        for p in root.rglob("*")
    }


def test_publication_is_idempotent_and_readback_cannot_grant_live(tmp_path, monkeypatch):
    journal, args = setup(tmp_path, monkeypatch)
    ref = proof.publish_synthetic_next_session_proof(**args)
    before = inventory(tmp_path)
    assert proof.publish_synthetic_next_session_proof(**args) == ref
    read = proof.read_next_session_proof(
        workspace=str(tmp_path), eod_trade_date="20250610", publication_ref=ref
    )
    assert read["synthetic"] is True and read["live_eligible"] is False
    assert inventory(tmp_path) == before
    with pytest.raises(TypeError):
        proof.publish_synthetic_next_session_proof(**args, synthetic=False)


def test_completed_eod_cannot_receive_a_late_proof(tmp_path, monkeypatch):
    journal, args = setup(tmp_path, monkeypatch)
    journal.storage.write(str(journal.root / "completion.v1.json"), b"{}")
    with pytest.raises(ContractError, match="EOD_ALREADY_COMPLETED"):
        proof.publish_synthetic_next_session_proof(**args)
    assert not (tmp_path / journal.root / "calendar-future/publications").exists()


def test_receipt_tamper_is_rejected_without_writes(tmp_path, monkeypatch):
    journal, args = setup(tmp_path, monkeypatch)
    ref = proof.publish_synthetic_next_session_proof(**args)
    (tmp_path / ref["path"]).write_bytes(b"{}")
    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="SHA_MISMATCH"):
        proof.read_next_session_proof(
            workspace=str(tmp_path), eod_trade_date="20250610", publication_ref=ref
        )
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("fault", ["future_time", "numeric_authority"])
def test_invalid_existing_provenance_cannot_publish_proof_files(tmp_path, monkeypatch, fault):
    journal, args = setup(tmp_path, monkeypatch)
    info = proof.inspect_next_session_capture()
    value = {**proof._provenance_identity(info), "recorded_at": "2025-06-10T07:00:03Z"}
    if fault == "future_time":
        value["recorded_at"] = "2099-01-01T00:00:00Z"
    else:
        value["authority"] = dict.fromkeys(value["authority"], 0)
    path = (
        journal.root
        / "calendar-future/provenance"
        / (args["execution_ref"]["byte_sha256"] + ".json")
    )
    journal.storage.write(str(path), canonical_json_bytes(value))
    with journal.locked():
        pass
    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="PROVENANCE"):
        proof.publish_synthetic_next_session_proof(**args)
    assert inventory(tmp_path) == before
