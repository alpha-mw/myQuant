"""Pre-network guards for the fixed installed acquisition route."""

import hashlib
import inspect
import pytest
from quant_investor.market import next_session_acquisition as acquisition
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal


def test_no_transport_clock_or_synthetic_override_parameter():
    names = set(inspect.signature(acquisition.capture_next_session_calendar).parameters)
    assert names == {
        "workspace",
        "eod_trade_date",
        "release_install_input_raw",
        "expected_release_install_input_sha256",
        "release_repository_root",
    }


def test_wrong_release_sha_rejected_before_any_runtime_or_provider_call(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before call")

    monkeypatch.setattr(acquisition, "verify_running_release_install_input", forbidden)
    monkeypatch.setattr(acquisition, "capture_trusted_provider_calendar_evidence", forbidden)
    with pytest.raises(ContractError, match="SHA_MISMATCH"):
        acquisition.capture_next_session_calendar(
            workspace=str(tmp_path),
            eod_trade_date="20260901",
            release_install_input_raw=b"{}",
            expected_release_install_input_sha256="a" * 64,
            release_repository_root=str(tmp_path),
        )
    assert not list(tmp_path.iterdir())


def test_completed_eod_rejects_acquisition_before_network(tmp_path, monkeypatch):
    journal = DailyJournal(str(tmp_path), "20260901")
    journal.storage.write(str(journal.root / "completion.v1.json"), b"{}")
    monkeypatch.setattr(
        acquisition, "verify_running_release_install_input", lambda *a, **k: {"state": "PASS"}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no provider call after completion")

    monkeypatch.setattr(acquisition, "capture_trusted_provider_calendar_evidence", forbidden)
    with pytest.raises(ContractError, match="EOD_ALREADY_COMPLETED"):
        acquisition.capture_next_session_calendar(
            workspace=str(tmp_path),
            eod_trade_date="20260901",
            release_install_input_raw=b"{}",
            expected_release_install_input_sha256=hashlib.sha256(b"{}").hexdigest(),
            release_repository_root=str(tmp_path),
        )
