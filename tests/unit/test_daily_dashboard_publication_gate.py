"""Real final publication gate with explicitly controlled native run/seal admission."""

from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import hashlib

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.operations.daily_contract import EOD_NODE_IDS
from quant_investor.operations.native_input_contract import FIELDS, EXTRAS
from quant_investor.operations.dashboard_serving_contract import POLICY
from quant_investor.operations.production_result import (
    validate_production_result,
    production_result_exit_code,
)
from scripts import daily_completion as completion
from scripts import daily_native_inputs
from scripts import daily_dashboard_publication as publication


def prepare(root):
    journal = DailyJournal(str(root), "20260904")
    reference = {"path": "source.json", "sha256": "a" * 64}
    inputs = dict.fromkeys(FIELDS | EXTRAS)
    inputs.update(
        schema_version="cn-daily-native-inputs.v5",
        trade_date="20260904",
        decision_recipe_ref=reference,
        corporate_action_context_ref=reference,
        dashboard_publication_policy=POLICY,
        publish_current_dashboard=True,
    )
    path = root / "native.json"
    raw = canonical_json_bytes(inputs)
    path.write_bytes(raw)
    path.chmod(0o600)
    return journal, {"path": "native.json", "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize("resume", [False, True])
def test_current_completion_cannot_report_success_when_publication_fails(
    tmp_path, monkeypatch, resume
):
    journal, input_ref = prepare(tmp_path)
    calls = []
    completion_ref = {}

    def seal(*args, **kwargs):
        calls.append("seal")
        value = {"schema_version": "fixture-admission-only", "native_inputs_ref": input_ref}
        raw = canonical_json_bytes(value)
        path = str(journal.root / "completion.v1.json")
        journal.storage.write(path, raw)
        completion_ref.update(path=path, sha256=hashlib.sha256(raw).hexdigest())
        return dict(completion_ref)

    def run(*args, **kwargs):
        calls.append("run")
        return {"status": "PARTIAL", "nodes": {n: {"state": "SUCCEEDED"} for n in EOD_NODE_IDS}}

    def failed(**kwargs):
        calls.append("publish")
        assert (tmp_path / kwargs["completion_ref"]["path"]).exists()
        raise OSError("controlled serving failure after seal")

    monkeypatch.setattr(completion, "seal_materialized_completion", seal)
    monkeypatch.setattr(daily_native_inputs, "run_loaded_native_input", run)
    monkeypatch.setattr(publication, "publish_completed_dashboard", failed)
    registry = SimpleNamespace(
        workspace=str(tmp_path),
        trade_date=journal.trade_date,
        runner=SimpleNamespace(journal=journal),
    )
    with journal.locked():
        if resume:
            seal()
            calls.clear()
        result = completion.run_and_seal_materialized_input(
            registry,
            materialized=SimpleNamespace(native_inputs_ref=input_ref),
            resume=resume,
            synthetic=True,
        )
    assert calls == (["seal", "publish"] if resume else ["run", "seal", "publish"])
    assert result["status"] == "PARTIAL" and result["completion_ref"] is None
    assert result["sealed_evidence_ref"] == completion_ref
    assert result["completion_status"] == "EVIDENCE_SEALED_PUBLICATION_PENDING"
    assert (tmp_path / completion_ref["path"]).exists()
    public = {
        "schema_version": "cn-daily-production-result.v1",
        "action": "RESUME" if resume else "EXECUTE",
        "target_trade_date": journal.trade_date,
        "execution_state": "PARTIAL",
        "business_state": "INCOMPLETE",
        "days": [
            {
                "trade_date": journal.trade_date,
                "execution_state": "PARTIAL",
                "business_state": "INCOMPLETE",
                "completion_ref": None,
            }
        ],
        "authority": FALSE_AUTHORITY,
    }
    validate_production_result(
        public,
        action=public["action"],
        target_trade_date=journal.trade_date,
        expected_dates=[journal.trade_date],
    )
    assert production_result_exit_code(public) == 2
