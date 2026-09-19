"""Failure boundaries with native rendering and explicitly controlled EOD admission."""

from dataclasses import fields
from datetime import datetime, timedelta, timezone
import json
import pickle

import pytest

from scripts import daily_dashboard_publication as pub
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.dashboard_publication_guard import (
    sealed_publication_scope,
    publication_scope,
)
from quant_investor.operations.dashboard_serving_contract import (
    PREFIX,
    HEAD_JSON,
    HEAD_JS,
    EVIDENCE_JSON,
    EVIDENCE_JS,
    SERVING_NAMES,
    SELECTOR_JSON,
    SELECTOR_JS,
    build_head,
    head_js,
)
from quant_investor.operations.journal_storage import JournalStorage
from test_daily_dashboard_sealed_publication import fixture, publish, inventory


def clock_at(monkeypatch, value):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr(pub, "datetime", Clock)


def cutoff(proof):
    return pub.instant(json.loads(proof.v2)["freshness"]["valid_through"])


def interrupt_write(monkeypatch, suffix):
    original = JournalStorage.write

    def write(self, path, raw, *args, **kwargs):
        if path.endswith(suffix):
            raise OSError("injected commit interruption")
        return original(self, path, raw, *args, **kwargs)

    monkeypatch.setattr(JournalStorage, "write", write)


def test_proof_cannot_be_forged_serialized_or_mutated(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    values = {f.name: getattr(proof, f.name) for f in fields(proof) if not f.name.startswith("_")}
    with pytest.raises(ContractError, match="FACTORY_REQUIRED"):
        pub.VerifiedDashboard(**values)
    with pytest.raises(TypeError):
        pickle.dumps(proof)
    with node.journal.locked(), sealed_publication_scope(tmp_path, proof) as cap:
        pub._register_head(proof, cap, node.journal)
        assert len(cap.expected) == 8
        with pytest.raises(ContractError, match="SEALED_BYTES_DIFFER"):
            cap.validate_bytes(tmp_path / PREFIX / EVIDENCE_JSON, b"arbitrary payload")
        proof.evidence["injected"] = True
        with pytest.raises(ContractError, match="PROOF_CHANGED"):
            cap.require()
    with pytest.raises(ContractError, match="PROOF_CHANGED"):
        with sealed_publication_scope(tmp_path, proof):
            pass


@pytest.mark.parametrize("remnant", ["valid_json", "corrupt_json", "corrupt_js"])
def test_selector_remnant_keeps_legacy_guard_closed(tmp_path, monkeypatch, remnant):
    node, proof = fixture(tmp_path, monkeypatch)
    publish(node, proof)
    for name in (HEAD_JSON, HEAD_JS, EVIDENCE_JSON, EVIDENCE_JS):
        (tmp_path / PREFIX / name).unlink()
    if remnant == "corrupt_json":
        (tmp_path / PREFIX / SELECTOR_JSON).write_bytes(
            b'{"schema_version":"cn_aggressive_dashboard_selector.v3"'
        )
    if remnant == "corrupt_js":
        (tmp_path / PREFIX / SELECTOR_JSON).unlink()
        (tmp_path / PREFIX / SELECTOR_JS).write_bytes(b"broken commit mirror")
    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="EOD_PUBLICATION_REQUIRED"):
        with publication_scope(tmp_path):
            pytest.fail("legacy write became available")
    assert inventory(tmp_path) == before


def test_interrupted_head_proposal_preserves_original_clock(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    stamp = proof.completion["native_validation_completed_at"]
    proposal = build_head(
        day=proof.day, ref=proof.completion_ref, previous_sha=None, registered_at=stamp
    )
    raw = canonical_json_bytes(proposal)
    (tmp_path / PREFIX).mkdir(parents=True, exist_ok=True)
    (tmp_path / PREFIX / HEAD_JS).write_bytes(head_js(raw))
    (tmp_path / PREFIX / HEAD_JS).chmod(0o600)
    with node.journal.locked(), sealed_publication_scope(tmp_path, proof) as cap:
        monkeypatch.setattr(pub, "_now", lambda: pytest.fail("valid proposal was restamped"))
        _, committed, _ = pub._register_head(proof, cap, node.journal)
    assert committed == raw
    assert (tmp_path / PREFIX / HEAD_JS).read_bytes() == head_js(raw)


@pytest.mark.parametrize("boundary", ["data", "selector", "commit"])
def test_crossing_expiry_cannot_mint_commit_or_receipt(tmp_path, monkeypatch, boundary):
    node, proof = fixture(tmp_path, monkeypatch)
    expired = cutoff(proof) + timedelta(seconds=1)
    if boundary == "data":
        import scripts.export_cn_aggressive_dashboard_data as native

        original = native.publish_bundle_pair

        def data(*args, **kwargs):
            result = original(*args, **kwargs)
            clock_at(monkeypatch, expired)
            return result

        monkeypatch.setattr(native, "publish_bundle_pair", data)
    elif boundary == "selector":
        import scripts.cn_dashboard_v2_selector as native

        original = native.publish_selector

        def selector(*args, **kwargs):
            result = original(*args, **kwargs)
            clock_at(monkeypatch, expired)
            return result

        monkeypatch.setattr(native, "publish_selector", selector)
    else:
        original = pub._validate_commit

        def commit(*args, **kwargs):
            result = original(*args, **kwargs)
            clock_at(monkeypatch, expired)
            return result

        monkeypatch.setattr(pub, "_validate_commit", commit)
    with pytest.raises(ContractError, match="PUBLICATION_EXPIRED"):
        publish(node, proof)
    folder = tmp_path / node.journal.root / "dashboard"
    assert not (folder / "selector-commit.v1.json").exists()
    assert not (folder / "serving-publication.v2.json").exists()
    if boundary == "data":
        assert not (tmp_path / PREFIX / SELECTOR_JSON).exists()


def test_selector_time_is_after_data_readback(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    import scripts.export_cn_aggressive_dashboard_data as native

    original = native.publish_bundle_pair
    later = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=10)

    def data(*args, **kwargs):
        result = original(*args, **kwargs)
        clock_at(monkeypatch, later)
        return result

    monkeypatch.setattr(native, "publish_bundle_pair", data)
    publish(node, proof)
    folder = tmp_path / node.journal.root / "dashboard"
    intent = json.loads((folder / "serving-intent.v1.json").read_bytes())
    commit = json.loads((folder / "selector-commit.v1.json").read_bytes())
    assert (
        pub.utc_stamp(intent["intent_created_at"])
        < pub.utc_stamp(commit["selector_updated_at"])
        == later
    )
    assert set(intent["files"]) == set(pub.FINANCIAL_NAMES) | {EVIDENCE_JSON, EVIDENCE_JS}


@pytest.mark.parametrize("expired", [False, True])
def test_complete_pair_without_commit_needs_unexpired_recovery(tmp_path, monkeypatch, expired):
    node, proof = fixture(tmp_path, monkeypatch)
    with monkeypatch.context() as fault:
        interrupt_write(fault, "selector-commit.v1.json")
        with pytest.raises(OSError):
            publish(node, proof)
    selector_before = (tmp_path / PREFIX / SELECTOR_JSON).read_bytes()
    before = inventory(tmp_path)
    clock_at(
        monkeypatch,
        (
            cutoff(proof) + timedelta(seconds=1)
            if expired
            else datetime.now(timezone.utc) + timedelta(seconds=2)
        ),
    )
    if expired:
        with pytest.raises(ContractError, match="PUBLICATION_EXPIRED"):
            publish(node, proof)
        assert inventory(tmp_path) == before
    else:
        result = publish(node, proof)
        receipt = json.loads((tmp_path / result["publication_ref"]["path"]).read_bytes())
        assert receipt["recovered_unknown"] is True
        assert pub.instant(json.loads(selector_before)["updated_at"]) == pub.utc_stamp(
            receipt["selector_updated_at"]
        )
    assert (tmp_path / PREFIX / SELECTOR_JSON).read_bytes() == selector_before


def test_partial_selector_gets_actual_recovery_clock(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    import scripts.cn_dashboard_v2_selector as native

    original = native._atomic_replace
    with monkeypatch.context() as fault:

        def write(path, raw):
            if path.name == SELECTOR_JS:
                raise OSError("partial selector")
            return original(path, raw)

        fault.setattr(native, "_atomic_replace", write)
        with pytest.raises(OSError):
            publish(node, proof)
    old = json.loads((tmp_path / PREFIX / SELECTOR_JSON).read_bytes())
    later = pub.instant(old["updated_at"]) + timedelta(seconds=5)
    clock_at(monkeypatch, later)
    publish(node, proof)
    selector = json.loads((tmp_path / PREFIX / SELECTOR_JSON).read_bytes())
    assert pub.instant(selector["updated_at"]) == later
    assert selector["updated_at"] != old["updated_at"]


def test_sealed_commit_can_get_historical_receipt_after_expiry(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    with monkeypatch.context() as fault:
        interrupt_write(fault, "serving-publication.v2.json")
        with pytest.raises(OSError):
            publish(node, proof)
    folder = tmp_path / node.journal.root / "dashboard"
    raw = (folder / "selector-commit.v1.json").read_bytes()
    commit = json.loads(raw)
    serving = {name: (tmp_path / PREFIX / name).read_bytes() for name in SERVING_NAMES}
    later = cutoff(proof) + timedelta(seconds=1)
    clock_at(monkeypatch, later)
    result = publish(node, proof)
    receipt = json.loads((tmp_path / result["publication_ref"]["path"]).read_bytes())
    assert result["status"] == "PUBLISHED_HISTORICAL_EXPIRED"
    monkeypatch.setattr(pub, "_recorded_serving_sources", lambda *args: proof)
    observed = pub.observed_serving_status(str(tmp_path), proof.day)
    assert observed["publication_state"] == "RECORDED_EOD_PUBLICATION_EXPIRED"
    assert receipt["first_verified_at"] == commit["first_verified_at"]
    assert receipt["selector_updated_at"] == commit["selector_updated_at"]
    assert pub.utc_stamp(receipt["receipt_recorded_at"]) == later
    assert (folder / "selector-commit.v1.json").read_bytes() == raw
    assert all((tmp_path / PREFIX / name).read_bytes() == value for name, value in serving.items())
    before = inventory(tmp_path)
    assert publish(node, proof) == result
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "fault", ["schema", "intent", "clock", "head", "files", "commit", "authority"]
)
def test_receipt_closure_rejects_tampering_without_writes(tmp_path, monkeypatch, fault):
    node, proof = fixture(tmp_path, monkeypatch)
    result = publish(node, proof)
    path = tmp_path / result["publication_ref"]["path"]
    receipt = json.loads(path.read_bytes())
    if fault == "schema":
        receipt["schema_version"] = "forged"
    elif fault == "clock":
        receipt["receipt_recorded_at"] = "2000-01-01T00:00:00Z"
    elif fault == "files":
        receipt["files"].pop(SELECTOR_JSON)
    elif fault == "authority":
        receipt["authority"]["trade"] = True
    else:
        key = {
            "intent": "intent_ref",
            "head": "completed_head_ref",
            "commit": "selector_commit_ref",
        }[fault]
        receipt[key]["sha256"] = "f" * 64
    path.write_bytes(canonical_json_bytes(receipt))
    before = inventory(tmp_path)
    with pytest.raises((ContractError, ValueError)):
        publish(node, proof)
    # Same full validator through read-only status; source admission remains the explicit seam.
    monkeypatch.setattr(pub, "_recorded_serving_sources", lambda *args: proof)
    with pytest.raises((ContractError, ValueError)):
        pub.observed_serving_status(str(tmp_path), proof.day)
    assert inventory(tmp_path) == before
