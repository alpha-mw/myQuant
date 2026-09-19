import hashlib
import pytest
from quant_investor.market import _calendar_fixture_capability as fixture
from quant_investor.market import next_session_calendar, next_session_proof
from quant_investor.operations.daily_contract import ContractError


def scope(tmp_path):
    return dict(
        workspace=str(tmp_path),
        trade_date="20260902",
        install_sha="a" * 64,
        fixture_source_sha="b" * 64,
        documentation_raw=b"docs",
        provider_raw={k: k.encode() for k in fixture.EXCHANGES},
    )


def info():
    return {
        "execution": {
            "payload": {
                "release_install_input_file_ref": {"byte_sha256": "a" * 64},
                "documentation_raw_file_ref": {"byte_sha256": hashlib.sha256(b"docs").hexdigest()},
            }
        },
        "success": {"payload": {"observed_completed_at": "2020-01-01T00:00:00Z"}},
        "raw_refs": [
            {
                "relative_path": "cap/response-" + k.lower() + ".raw",
                "byte_sha256": hashlib.sha256(k.encode()).hexdigest(),
            }
            for k in fixture.EXCHANGES
        ],
        "execution_ref": {"relative_path": "cap/capture-execution.json", "byte_sha256": "c" * 64},
        "success_ref": {"relative_path": "cap/capture-success.json", "byte_sha256": "d" * 64},
    }


def captured():
    return {
        "capture_execution": {},
        "capture_execution_file_ref": info()["execution_ref"],
        "capture_success": {},
        "capture_success_file_ref": info()["success_ref"],
    }


def test_no_serialized_mode_or_boolean_grants_capability(tmp_path):
    with pytest.raises(ContractError, match="PROVENANCE_UNAVAILABLE"):
        fixture.require_fixture_capability(workspace=tmp_path)
    assert not list(tmp_path.iterdir())


def test_scope_binds_workspace_day_install_and_restores_adapters(tmp_path):
    original = fixture.native.OfficialTushareHttpsClient
    with fixture._offline_calendar_fixture(**scope(tmp_path)):
        fixture.require_fixture_capability(
            workspace=tmp_path, trade_date="20260902", install_sha="a" * 64
        )
        for kw in [dict(trade_date="20260903"), dict(install_sha="b" * 64)]:
            with pytest.raises(ContractError):
                fixture.require_fixture_capability(workspace=tmp_path, **kw)
        with pytest.raises(ContractError):
            with fixture._offline_calendar_fixture(**scope(tmp_path)):
                pass
    assert fixture.native.OfficialTushareHttpsClient is original
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("adapter", ["OfficialTushareHttpsClient", "_official_documentation_fetch"])
def test_substituted_or_real_adapter_cannot_issue_proof(tmp_path, monkeypatch, adapter):
    original = getattr(fixture.native, adapter)
    with fixture._offline_calendar_fixture(**scope(tmp_path)):
        with monkeypatch.context() as patch:
            patch.setattr(fixture.native, adapter, original)
            with pytest.raises(ContractError, match="PROVENANCE_UNAVAILABLE"):
                fixture.publish_fixture_evidence(
                    workspace=tmp_path,
                    trade_date="20260902",
                    install_sha="a" * 64,
                    captured=captured(),
                )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ["docs", "response", "install"])
def test_actual_capture_bytes_must_match_pinned_manifest(tmp_path, monkeypatch, fault):
    value = info()
    if fault == "docs":
        value["execution"]["payload"]["documentation_raw_file_ref"]["byte_sha256"] = "e" * 64
    elif fault == "install":
        value["execution"]["payload"]["release_install_input_file_ref"]["byte_sha256"] = "e" * 64
    else:
        value["raw_refs"][0]["byte_sha256"] = "e" * 64
    monkeypatch.setattr(next_session_calendar, "inspect_next_session_capture", lambda **kw: value)
    with fixture._offline_calendar_fixture(**scope(tmp_path)):
        with pytest.raises(ContractError, match="PROVENANCE_UNAVAILABLE"):
            fixture.publish_fixture_evidence(
                workspace=tmp_path, trade_date="20260902", install_sha="a" * 64, captured=captured()
            )
    assert not list(tmp_path.iterdir())


def test_evidence_is_exact_idempotent_and_required_before_publisher(tmp_path, monkeypatch):
    monkeypatch.setattr(next_session_calendar, "inspect_next_session_capture", lambda **kw: info())
    calls = []
    monkeypatch.setattr(
        next_session_proof,
        "publish_synthetic_next_session_proof",
        lambda **kw: calls.append(kw) or {"proof": True},
    )
    args = dict(
        workspace=tmp_path, trade_date="20260902", install_sha="a" * 64, captured=captured()
    )
    with fixture._offline_calendar_fixture(**scope(tmp_path)):
        with pytest.raises(ContractError):
            fixture.publish_bound_fixture_proof(
                **args, evidence_ref={"path": "missing", "sha256": "a" * 64}
            )
        assert calls == []
        ref = fixture.publish_fixture_evidence(**args)
        assert fixture.publish_fixture_evidence(**args) == ref
        assert fixture.publish_bound_fixture_proof(**args, evidence_ref=ref) == {"proof": True}
        with pytest.raises(ContractError):
            fixture.publish_bound_fixture_proof(**args, evidence_ref={**ref, "sha256": "e" * 64})
        assert len(calls) == 1
    with pytest.raises(ContractError):
        fixture.publish_bound_fixture_proof(**args, evidence_ref=ref)


@pytest.mark.parametrize("stamp", ["1900-01-01T00:00:00Z", "2999-01-01T00:00:00Z", "invalid"])
def test_fixture_evidence_chronology_is_not_trusted(stamp):
    identity = {"schema_version": "fixture-test"}
    with pytest.raises(ContractError, match="PROVENANCE_UNAVAILABLE"):
        fixture._validate_evidence({**identity, "recorded_at": stamp}, identity, info())
