"""Use the real native maintenance lock; blocked readers cannot touch stage files."""

import hashlib
from pathlib import Path
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.daily_maintenance import _RunLock
from quant_investor.operations import maintenance_readback as reader
from quant_investor.operations.daily_contract import ContractError


def context(root):
    run = root / "maintenance"
    run.mkdir(mode=0o700)
    attempt = run / "attempts" / "one"
    attempt.mkdir(parents=True, mode=0o700)
    attempt.parent.chmod(0o700)
    core = attempt / "core-completion.json"
    raw = canonical_json_bytes({"fixture": "core"})
    core.write_bytes(raw)
    core.chmod(0o600)
    stage = attempt / "stage-FUNDAMENTAL.json"
    stage.write_bytes(
        canonical_json_bytes({"state": "STAGE_COMPLETED", "result": {"status": "SUCCESS"}})
    )
    stage.chmod(0o600)
    return run, dict(
        workspace=str(root),
        run_root="maintenance",
        core_ref={"path": str(core.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()},
    )


def test_live_native_lock_blocks_before_any_receipt_read(tmp_path, monkeypatch):
    run, args = context(tmp_path)
    monkeypatch.setattr(reader, "_read", lambda *a: pytest.fail("read while producer owns lock"))
    with _RunLock(run / ".daily-maintenance.lock"):
        with pytest.raises(ContractError, match="MAINTENANCE_AUXILIARY_RUNNING"):
            reader.read_auxiliary_stage_records(**args)


def test_existing_released_lock_reads_only_recorded_stages_without_writes(tmp_path):
    run, args = context(tmp_path)
    with _RunLock(run / ".daily-maintenance.lock"):
        pass
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    result = reader.read_auxiliary_stage_records(**args)
    assert result["stages"]["fundamental"]["state"] == "RECORDED"
    assert result["stages"]["macro"] == {"state": "MISSING", "ref": None, "document": None}
    assert result["native_replay_required"] is True and result["execution_authorized"] is False
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    with _RunLock(run / ".daily-maintenance.lock"):
        pass


def test_missing_lock_is_not_created(tmp_path):
    run, args = context(tmp_path)
    with pytest.raises(ContractError, match="LOCK_MISSING"):
        reader.read_auxiliary_stage_records(**args)
    assert not (run / ".daily-maintenance.lock").exists()


@pytest.mark.parametrize("fault", ["mode", "symlink", "hardlink", "fifo"])
def test_unsafe_lock_rejected(tmp_path, fault):
    run, args = context(tmp_path)
    lock = run / ".daily-maintenance.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    if fault == "mode":
        lock.chmod(0o644)
    elif fault == "symlink":
        other = run / "other"
        lock.rename(other)
        lock.symlink_to(other)
    elif fault == "fifo":
        import os

        lock.unlink()
        os.mkfifo(lock, 0o600)
    else:
        import os

        os.link(lock, run / "other")
    with pytest.raises(ContractError):
        reader.read_auxiliary_stage_records(**args)
