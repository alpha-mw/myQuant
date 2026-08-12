from __future__ import annotations

import copy
import hashlib

import pytest

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2._core import seal
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.forensic_v5_1 import (
    REQUIRED_EPOCH_BINDINGS,
    build_fundamental_forensic_receipt_v5_1,
    build_inert_same_epoch_plan_v1,
    validate_fundamental_forensic_receipt_v5_1,
    validate_inert_same_epoch_plan_v1,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.models import (
    FundamentalV4ContractError,
)

TABLES = ("balancesheet", "cashflow", "daily_basic", "fina_indicator", "forecast", "income")
NOW = "2026-08-13T02:00:00Z"
BINDING_SHA = "9" * 64


def incident() -> dict[str, dict]:
    row_diff = {table: [] for table in TABLES}
    row_diff["fina_indicator"] = [
        {"baseline_count": 1, "row_sha256": "a" * 64, "vip_count": 0},
        {"baseline_count": 0, "row_sha256": "b" * 64, "vip_count": 1},
    ]
    value_diff = {table: [] for table in TABLES}
    value_diff["fina_indicator"] = [
        {
            "baseline_winner_sha256": "a" * 64,
            "key_sha256": "c" * 64,
            "vip_winner_sha256": None,
        },
        {
            "baseline_winner_sha256": None,
            "key_sha256": "d" * 64,
            "vip_winner_sha256": "b" * 64,
        },
    ]
    duplicate_diff = {
        table: {"baseline_duplicate_row_count": 0, "vip_duplicate_row_count": 0} for table in TABLES
    }
    table_evidence = {table: {"evidence": table} for table in TABLES}
    files = {
        "duplicate_diff.json": duplicate_diff,
        "raw_row_diff.json": row_diff,
        "raw_value_diff.json": value_diff,
        "table_evidence.json": table_evidence,
    }
    summary = {
        "checkpoint_execution_bundle_sha256": "e" * 64,
        "diff_counts": {
            table: {
                "row": 2 if table == "fina_indicator" else 0,
                "value": 2 if table == "fina_indicator" else 0,
            }
            for table in TABLES
        },
        "file_sha256": {
            name: hashlib.sha256(canonical_bytes(value)).hexdigest()
            for name, value in files.items()
        },
        "implementation_commit": "f" * 40,
        "package_sha256": "1" * 64,
        "passed": False,
        "physical_receipt_count": 11471,
        "status": "RECONCILIATION_BLOCKED",
        "transport_calls": 0,
        "version": "myquant.r7.fundamental-v5-reconciliation-forensic.v1",
    }
    return {"summary": summary, **files}


def receipt() -> dict:
    evidence = incident()
    return build_fundamental_forensic_receipt_v5_1(
        produced_at=NOW,
        subject_id="920188.BJ",
        period="20250630",
        baseline_ann_date="20251205",
        vip_ann_date="20260312",
        subject_binding_source_sha256=BINDING_SHA,
        expected_row_sha256=["a" * 64, "b" * 64],
        expected_key_sha256=["c" * 64, "d" * 64],
        summary=evidence["summary"],
        raw_row_diff=evidence["raw_row_diff.json"],
        raw_value_diff=evidence["raw_value_diff.json"],
        duplicate_diff=evidence["duplicate_diff.json"],
        table_evidence=evidence["table_evidence.json"],
    )


def test_forensic_is_content_addressed_inconclusive_and_zero_network() -> None:
    document = receipt()
    assert (
        validate_fundamental_forensic_receipt_v5_1(
            document, subject_binding_source_sha256=BINDING_SHA
        )
        == document
    )
    assert document["classification"] == "INSUFFICIENT_TO_DISAMBIGUATE"
    assert document["next_required_evidence"] == "SAME_SEALED_ACQUISITION_EPOCH_REQUIRED"
    assert document["transport_calls"] == 0
    assert document["promotion_authorized"] is False
    assert document["tolerance_applied"] is False
    assert set(document["possible_causes"]) == {
        "ACQUISITION_EPOCH_MISMATCH",
        "PROVIDER_REVISION_OR_RESTATEMENT_DRIFT",
    }


def test_forensic_fails_closed_on_sha_drift_and_receipt_tampering() -> None:
    evidence = incident()
    evidence["summary"]["file_sha256"]["raw_row_diff.json"] = "0" * 64
    with pytest.raises(FundamentalV4ContractError, match="source SHA mismatch"):
        build_fundamental_forensic_receipt_v5_1(
            produced_at=NOW,
            subject_id="920188.BJ",
            period="20250630",
            baseline_ann_date="20251205",
            vip_ann_date="20260312",
            subject_binding_source_sha256=BINDING_SHA,
            expected_row_sha256=["a" * 64, "b" * 64],
            expected_key_sha256=["c" * 64, "d" * 64],
            summary=evidence["summary"],
            raw_row_diff=evidence["raw_row_diff.json"],
            raw_value_diff=evidence["raw_value_diff.json"],
            duplicate_diff=evidence["duplicate_diff.json"],
            table_evidence=evidence["table_evidence.json"],
        )
    tampered = copy.deepcopy(receipt())
    tampered["classification"] = "PROVIDER_REVISION_PROVEN"
    with pytest.raises(FundamentalV4ContractError):
        validate_fundamental_forensic_receipt_v5_1(
            tampered, subject_binding_source_sha256=BINDING_SHA
        )


def test_same_epoch_plan_is_inert_and_has_no_implicit_bindings() -> None:
    forensic = receipt()
    plan = build_inert_same_epoch_plan_v1(produced_at=NOW, forensic_receipt=forensic)
    assert validate_inert_same_epoch_plan_v1(plan, forensic_receipt=forensic) == plan
    assert plan["campaign_state"] == "BLOCKED_PENDING_BOUND_INPUTS"
    assert plan["missing_binding_fields"] == list(REQUIRED_EPOCH_BINDINGS)
    assert plan["required_binding_fields"] == list(REQUIRED_EPOCH_BINDINGS)
    assert plan["execution_authorized"] is False
    assert plan["network_attempts_executed"] == 0
    assert plan["pointer_mutation_authorized"] is False
    assert plan["promotion_authorized"] is False
    assert plan["reuse_archived_baseline"] is False
    assert plan["tolerance_permitted"] is False


def test_same_epoch_plan_rejects_a_self_consistent_but_invalid_forensic_seal() -> None:
    forensic = receipt()
    body = {
        key: value
        for key, value in forensic.items()
        if key not in {"forensic_receipt_id", "semantic_sha256"}
    }
    body["promotion_authorized"] = True
    body["transport_calls"] = 99
    forged = seal(body, identity_field="forensic_receipt_id")

    with pytest.raises(FundamentalV4ContractError):
        build_inert_same_epoch_plan_v1(produced_at=NOW, forensic_receipt=forged)
