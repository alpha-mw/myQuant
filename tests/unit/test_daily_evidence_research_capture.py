"""Native compiler output closure and actual-custody timestamp tests."""

from copy import deepcopy
import hashlib
from quant_investor.contracts import canonical_json_bytes
import pytest

from quant_investor.intelligence import compile_daily_intelligence
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.research_capture import ResearchCapture, validate_compilation
from test_unified_daily_intelligence import policy, projections, rank_artifact, NOW, STRATEGY


def result():
    p = policy()
    industry, theme = projections(p)
    return compile_daily_intelligence(
        as_of=NOW,
        strategy_id=STRATEGY,
        rank=rank_artifact(p),
        policy=p,
        industry_projection=industry,
        theme_projection=theme,
    )


def request_ref(root):
    path = root / "request.json"
    raw = canonical_json_bytes({"as_of": NOW, "strategy_id": STRATEGY})
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": "request.json", "sha256": hashlib.sha256(raw).hexdigest()}


def test_native_partial_capture_is_immutable_and_never_backdates_publication(tmp_path):
    value = result()
    journal = DailyJournal(str(tmp_path), "20260821")
    capture = ResearchCapture(journal)
    request = request_ref(tmp_path)
    with journal.locked():
        first = capture.publish(request, value)
        second = capture.publish(request, value)
    assert first == second
    assert first["research_status"] == "PARTIAL"
    assert first["research_cutoff"] == NOW
    assert first["captured_at"] != NOW
    assert first["native_timestamps_are_publication_proof"] is False
    assert capture.read(request)[1] == value


@pytest.mark.parametrize(
    "mutation", ["authority", "status", "missing_artifact", "missing_decision", "decision_state"]
)
def test_compilation_tampering_fails_closed(mutation):
    value = deepcopy(result())
    if mutation == "authority":
        value["authority"]["broker"] = True
    elif mutation == "status":
        value["status"] = "COMPLETE"
    elif mutation == "missing_artifact":
        value["artifacts"].remove(value["evaluation"])
    elif mutation == "missing_decision":
        value["decisions"].pop()
    else:
        value["decisions"][0]["state"] = "RESEARCH_APPROVED"
    with pytest.raises(ContractError):
        validate_compilation(value, trade_date="20260821")


def test_capture_detects_leaf_drift(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260821")
    capture = ResearchCapture(journal)
    request = request_ref(tmp_path)
    with journal.locked():
        manifest = capture.publish(request, result())
    (tmp_path / manifest["artifact_refs"][0]["path"]).write_bytes(b"{}")
    with pytest.raises(ContractError, match="ARTIFACT_DRIFT"):
        capture.read(request)


def test_interrupted_capture_adopts_bytes_with_current_custody_time(tmp_path, monkeypatch):
    journal = DailyJournal(str(tmp_path), "20260821")
    capture = ResearchCapture(journal)
    ref = request_ref(tmp_path)
    original = journal.storage.write

    def crash(path, raw, **kwargs):
        if path.endswith("/capture.v1.json"):
            raise RuntimeError("before custody seal")
        return original(path, raw, **kwargs)

    with journal.locked():
        with monkeypatch.context() as patch:
            patch.setattr(journal.storage, "write", crash)
            with pytest.raises(RuntimeError, match="custody seal"):
                capture.publish(ref, result())
        recovered = capture.publish(ref, result())
    assert recovered["recovered_custody"] is True
    assert recovered["captured_at"] != NOW


def test_changed_input_request_is_not_accepted_by_capture_reader(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260821")
    capture = ResearchCapture(journal)
    ref = request_ref(tmp_path)
    with journal.locked():
        capture.publish(ref, result())
    (tmp_path / "request.json").write_bytes(b"{}")
    with pytest.raises(ContractError, match="REQUEST_SHA_MISMATCH"):
        capture.read(ref)
