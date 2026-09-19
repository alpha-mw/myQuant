import hashlib
import pytest
from quant_investor.market.next_session_failure import (
    publish_next_session_failure,
    read_next_session_failure,
)
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal


def test_acquisition_failure_is_typed_readonly_and_non_authorizing(tmp_path):
    ref = publish_next_session_failure(
        workspace=str(tmp_path),
        eod_trade_date="20260901",
        phase="ACQUISITION",
        failure_code="ACQUISITION_FAILED",
    )
    before = {
        str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
        for p in tmp_path.rglob("*")
    }
    value = read_next_session_failure(
        workspace=str(tmp_path), eod_trade_date="20260901", failure_ref=ref
    )
    assert value["consumer_admission"] is False
    assert not any(value["failure"]["authority"].values())
    assert {
        str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
        for p in tmp_path.rglob("*")
    } == before


def test_post_capture_failure_requires_exact_available_refs(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260901")

    def put(leaf):
        raw = b"{}"
        journal.storage.write(str(journal.root / "calendar-future/captures/cap" / leaf), raw)
        return {"relative_path": "cap/" + leaf, "byte_sha256": hashlib.sha256(raw).hexdigest()}

    execution, success = put("capture-execution.json"), put("capture-success.json")
    ref = publish_next_session_failure(
        workspace=str(tmp_path),
        eod_trade_date="20260901",
        phase="POST_CAPTURE_VALIDATION",
        failure_code="NO_LATER_OPEN",
        execution_ref=execution,
        success_ref=success,
    )
    assert (
        read_next_session_failure(
            workspace=str(tmp_path), eod_trade_date="20260901", failure_ref=ref
        )["failure"]["failure_code"]
        == "NO_LATER_OPEN"
    )
    with pytest.raises(ContractError, match="PHASE_BINDING"):
        publish_next_session_failure(
            workspace=str(tmp_path),
            eod_trade_date="20260901",
            phase="POST_CAPTURE_VALIDATION",
            failure_code="NO_LATER_OPEN",
        )


def test_uncontrolled_exception_text_cannot_be_persisted(tmp_path):
    with pytest.raises(ContractError, match="CONTRACT_INVALID"):
        publish_next_session_failure(
            workspace=str(tmp_path),
            eod_trade_date="20260901",
            phase="ACQUISITION",
            failure_code="error containing arbitrary payload",
        )
    assert not list(tmp_path.iterdir())


def test_source_change_before_lock_prevents_failure_publication(tmp_path, monkeypatch):
    from contextlib import contextmanager

    journal = DailyJournal(str(tmp_path), "20260901")
    base = journal.root / "calendar-future/captures/cap"
    raw = b"{}"
    refs = []
    for name in ("capture-execution.json", "capture-success.json"):
        journal.storage.write(str(base / name), raw)
        refs.append(
            {"relative_path": "cap/" + name, "byte_sha256": hashlib.sha256(raw).hexdigest()}
        )
    original = DailyJournal.locked

    @contextmanager
    def changed(self):
        with original(self):
            (tmp_path / base / "capture-success.json").write_bytes(b"changed")
            yield

    monkeypatch.setattr(DailyJournal, "locked", changed)
    with pytest.raises(ContractError, match="SOURCE_SHA_MISMATCH"):
        publish_next_session_failure(
            workspace=str(tmp_path),
            eod_trade_date="20260901",
            phase="POST_CAPTURE_VALIDATION",
            failure_code="NO_LATER_OPEN",
            execution_ref=refs[0],
            success_ref=refs[1],
        )
    assert not (tmp_path / journal.root / "calendar-future/failures").exists()


def test_post_capture_failure_cannot_mix_capture_roots(tmp_path):
    execution = {"relative_path": "one/capture-execution.json", "byte_sha256": "a" * 64}
    success = {"relative_path": "two/capture-success.json", "byte_sha256": "b" * 64}
    with pytest.raises(ContractError, match="CAPTURE_ROOT_MISMATCH"):
        publish_next_session_failure(
            workspace=str(tmp_path),
            eod_trade_date="20260901",
            phase="POST_CAPTURE_VALIDATION",
            failure_code="NO_LATER_OPEN",
            execution_ref=execution,
            success_ref=success,
        )
    assert not list(tmp_path.iterdir())


def test_native_failure_requires_native_contract_not_only_matching_hash(tmp_path):
    from quant_investor.system.errors import SystemContractError

    journal = DailyJournal(str(tmp_path), "20260901")
    path = "cap-failure/capture-failure.json"
    journal.storage.write(str(journal.root / "calendar-future/captures" / path), b"{}")
    ref = {"relative_path": path, "byte_sha256": hashlib.sha256(b"{}").hexdigest()}
    with pytest.raises(SystemContractError):
        publish_next_session_failure(
            workspace=str(tmp_path),
            eod_trade_date="20260901",
            phase="ACQUISITION",
            failure_code="ACQUISITION_FAILED",
            native_failure_ref=ref,
        )
    assert not (tmp_path / journal.root / "calendar-future/failures").exists()
