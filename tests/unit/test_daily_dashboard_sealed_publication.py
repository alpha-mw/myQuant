"""Real native financial publisher with an explicit outer EOD-admission seam."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json

import pytest

from test_daily_evidence_dashboard_history import backlog
from test_daily_evidence_dashboard_adapter import adapter
from _native_daily_store_fixture import DAYS
from scripts.daily_dashboard_adapter import HistoricalDashboardAdapter
from scripts import daily_dashboard_publication as publication
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.dashboard_serving_contract import (
    PREFIX,
    HEAD_JSON,
    SERVING_NAMES,
    SELECTOR_JSON,
    head_bytes,
)
from quant_investor.contracts import canonical_json_bytes


class CaptureCurrent(HistoricalDashboardAdapter):
    historical_mode = False


def fixture(root, monkeypatch):
    native = backlog(root)
    node = adapter(root, DAYS[-1], native["sessions"][0]["plan"], CaptureCurrent)
    request = node.template()
    with node.journal.locked():
        node.execute(request)
    outputs = node.probe(request).outcome.output_refs
    v1 = (root / outputs["v1"]["path"]).read_bytes()
    v2 = (root / outputs["v2"]["path"]).read_bytes()
    path = str(node.journal.root / "completion.v1.json")
    inputs = {"schema_version": "cn-daily-native-inputs.v5", "publish_current_dashboard": True}
    input_raw = canonical_json_bytes(inputs)
    input_path = str(node.journal.root / "dashboard/fixture-native-inputs.json")
    with node.journal.locked():
        node.journal.storage.write(input_path, input_raw)
    # This marks the controlled admission seam, not a fabricated valid EOD claim.
    completion = {
        "synthetic_fixture_admission": True,
        "native_inputs_ref": {"path": input_path, "sha256": hashlib.sha256(input_raw).hexdigest()},
        "native_validation_completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    raw = canonical_json_bytes(completion)
    with node.journal.locked():
        node.journal.storage.write(path, raw)
    ref = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    evidence = {"synthetic_fixture_source_summary": True, "source_admission_exercised": False}
    report_ref = {"path": "fixture-evidence.json", "sha256": "a" * 64}
    proof = publication.VerifiedDashboard(
        root,
        node.journal.trade_date,
        ref,
        completion,
        inputs,
        report_ref,
        {**outputs, "daily_evidence": report_ref},
        v1,
        v2,
        evidence,
        "2026-08-28T13:30:00Z",
        None,
        (),
        report_ref,
        _key=publication._PROOF_KEY,  # Explicit controlled EOD-admission seam.
    )

    def verified(*args):
        return replace(
            proof,
            _key=publication._PROOF_KEY,
            head_preimage=publication._optional(root, PREFIX + "/" + HEAD_JSON),
        )

    monkeypatch.setattr(publication, "verify_dashboard_for_publication", verified)
    return node, proof


def inventory(root):
    return {
        str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def publish(node, proof):
    return publication.publish_completed_dashboard(
        workspace=str(node.workspace), completion_ref=proof.completion_ref
    )


def test_exact_serving_head_and_idempotent_receipt(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    assert not any((tmp_path / PREFIX / name).exists() for name in SERVING_NAMES)
    result = publish(node, proof)
    assert result["status"] == "PUBLISHED"
    head = head_bytes((tmp_path / PREFIX / HEAD_JSON).read_bytes())
    selector = json.loads((tmp_path / PREFIX / SELECTOR_JSON).read_bytes())
    assert selector["schema_version"] == "cn_aggressive_dashboard_selector.v3"
    assert selector["completion_ref"] == head["completion_ref"] == proof.completion_ref
    assert (
        selector["completed_head_sha256"]
        == hashlib.sha256((tmp_path / PREFIX / HEAD_JSON).read_bytes()).hexdigest()
    )
    before = inventory(tmp_path)
    assert publish(node, proof) == result
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("boundary", ["head", "selector", "receipt"])
def test_interrupted_serving_recovers_without_financial_rerender(tmp_path, monkeypatch, boundary):
    node, proof = fixture(tmp_path, monkeypatch)
    with monkeypatch.context() as fault:

        def crash(*args, **kwargs):
            raise OSError("injected publication interruption")

        if boundary == "head":
            fault.setattr(publication, "_intent", crash)
        elif boundary == "selector":
            import scripts.cn_dashboard_v2_selector as selector

            fault.setattr(selector, "publish_selector", crash)
        else:
            from quant_investor.operations.journal_storage import JournalStorage

            original = JournalStorage.write

            def fail_receipt(self, path, raw, *args, **kwargs):
                if path.endswith("serving-publication.v2.json"):
                    crash()
                return original(self, path, raw, *args, **kwargs)

            fault.setattr(JournalStorage, "write", fail_receipt)
        with pytest.raises(OSError, match="injected"):
            publish(node, proof)
    original_head = (tmp_path / PREFIX / HEAD_JSON).read_bytes()
    import scripts.daily_dashboard_adapter as renderer

    monkeypatch.setattr(
        renderer, "build_bundle", lambda **kwargs: pytest.fail("financial rerender")
    )
    intent_path = tmp_path / node.journal.root / "dashboard/serving-intent.v1.json"
    intent_before = intent_path.read_bytes() if intent_path.exists() else None
    result = publish(node, proof)
    assert (tmp_path / PREFIX / HEAD_JSON).read_bytes() == original_head
    if intent_before is not None:
        assert intent_path.read_bytes() == intent_before
    receipt = json.loads((tmp_path / result["publication_ref"]["path"]).read_bytes())
    assert receipt["recovered_unknown"] is (boundary == "selector")


def test_legacy_writers_are_blocked_after_head_registration(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    publish(node, proof)
    from scripts.export_cn_aggressive_dashboard_data import publish_bundle
    from scripts.cn_dashboard_v2_selector import publish_selector

    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="EOD_PUBLICATION_REQUIRED"):
        publish_bundle(
            json.loads(proof.v1),
            tmp_path / PREFIX / "cn_aggressive_dashboard.v1.json",
            tmp_path / PREFIX / "cn_aggressive_dashboard.v1.js",
            tmp_path,
        )
    with pytest.raises(ContractError, match="EOD_PUBLICATION_REQUIRED"):
        publish_selector(
            {},
            json_path=tmp_path / PREFIX / SELECTOR_JSON,
            js_path=tmp_path / PREFIX / "cn_aggressive_dashboard_selector.v2.js",
            project_root=tmp_path,
            js_first=False,
        )
    assert inventory(tmp_path) == before


def test_older_verified_candidate_cannot_replace_completed_head(tmp_path, monkeypatch):
    node, proof = fixture(tmp_path, monkeypatch)
    publish(node, proof)
    old_ref = {
        **proof.completion_ref,
        "path": proof.completion_ref["path"].replace("20260828", "20260827"),
    }
    older = replace(
        proof,
        _key=publication._PROOF_KEY,
        day="20260827",
        completion_ref=old_ref,
        head_preimage=(tmp_path / PREFIX / HEAD_JSON).read_bytes(),
    )
    monkeypatch.setattr(publication, "verify_dashboard_for_publication", lambda *args: older)
    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="STALE_COMPLETED_CANDIDATE"):
        publish(node, older)
    # The other date's day lock may be created; serving and business files do not move.
    assert all((tmp_path / PREFIX / name).read_bytes() for name in SERVING_NAMES)
    for name in SERVING_NAMES:
        p = tmp_path / PREFIX / name
        assert (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) == before[str(p)]
