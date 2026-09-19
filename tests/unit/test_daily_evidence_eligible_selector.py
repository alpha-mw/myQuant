"""Selector gates over a controlled full-native reply, not real eligible evidence."""

from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from scripts import daily_completion_replay as native
from scripts.daily_ledger_replay import select_eligible_daily_evidence
from quant_investor.operations.daily_contract import EOD_NODE_IDS, ContractError


def reply():
    return dict(
        native_replay_validated=True,
        trade_date="20260908",
        completion_ref={
            "path": "results/operations/daily_production/CN/20260908/completion.v1.json",
            "sha256": "a" * 64,
        },
        validated_nodes=sorted(EOD_NODE_IDS),
        synthetic=False,
        ledger={
            "ledger_ref": {
                "path": "results/prospective/CN/20260908/evidence-ledger.v1.json",
                "sha256": "b" * 64,
            },
            "validation_scope": "NATIVE_LEDGER_DERIVATION",
            "classification": "CONTEMPORANEOUS",
            "prospective": True,
            "synthetic": False,
            "recomputed": False,
        },
    )


def test_selector_requires_full_replay_and_grants_no_other_authority(tmp_path, monkeypatch):
    value = reply()
    calls = []

    def replay(**kw):
        calls.append(kw)
        return deepcopy(value)

    monkeypatch.setattr(native, "replay_native_completion", replay)
    result = select_eligible_daily_evidence(
        workspace=str(tmp_path), trade_date="20260908", completion_ref=value["completion_ref"]
    )
    assert len(calls) == 1 and calls[0]["completion_ref"] == value["completion_ref"]
    assert result["ledger_ref"] == value["ledger"]["ledger_ref"]
    assert result["factor_admission"] is False and not any(result["authority"].values())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "wrong_ref",
        "missing_node",
        "legacy",
        "late",
        "recovery",
        "synthetic",
        "ledger_synthetic",
        "int_boolean",
    ],
)
def test_unproven_or_noncontemporaneous_reply_rejected(tmp_path, monkeypatch, fault):
    value = reply()
    ref = deepcopy(value["completion_ref"])
    if fault == "partial":
        value["native_replay_validated"] = False
    elif fault == "wrong_ref":
        value["completion_ref"]["sha256"] = "c" * 64
    elif fault == "missing_node":
        value["validated_nodes"].pop()
    elif fault == "legacy":
        value.pop("ledger")
    elif fault == "late":
        value["ledger"]["classification"] = "LATE_REGISTERED"
    elif fault == "recovery":
        value["ledger"]["classification"] = "UNKNOWN_LEGACY"
    elif fault == "synthetic":
        value["synthetic"] = True
    elif fault == "ledger_synthetic":
        value["ledger"]["synthetic"] = True
    else:
        value["ledger"]["prospective"] = 1
    monkeypatch.setattr(native, "replay_native_completion", lambda **kw: value)
    with pytest.raises(ContractError):
        select_eligible_daily_evidence(
            workspace=str(tmp_path), trade_date="20260908", completion_ref=ref
        )
    assert list(tmp_path.iterdir()) == []
