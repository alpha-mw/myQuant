from __future__ import annotations

import copy
from decimal import Decimal
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_investor.intelligence_v2._core import (
    canonical_bytes,
    common_fields,
    content_ref,
    seal,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    FundamentalV4ContractError,
    REQUIRED_EVIDENCE_PATHS,
    build_fundamental_comparison_policy,
    build_fundamental_provider_manifest_v4,
    build_fundamental_reconciliation_receipt,
    build_provider_evidence_fileset_manifest,
    capture_provider_evidence_directory,
    compare_fundamental_raw_tables,
    validate_fundamental_comparison_policy,
    validate_fundamental_provider_manifest_v4,
    validate_fundamental_reconciliation_receipt,
    validate_provider_evidence_fileset_manifest,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.models import (
    SOURCE_TABLES,
)

NOW = "2026-08-09T08:00:00Z"


def policy() -> dict[str, Any]:
    return build_fundamental_comparison_policy(
        table_policies={
            table: {
                "canonical_key_columns": ["ts_code", "period"],
                "column_rows": [
                    {"column": "ts_code", "kind": "TEXT"},
                    {"column": "period", "kind": "DATE"},
                    {"column": "amount", "kind": "DECIMAL"},
                    {"column": "ordinal", "kind": "INTEGER"},
                    {"column": "note", "kind": "TEXT"},
                ],
                "table": table,
                "winner_implementation_sha256": "a" * 64,
                "winner_order_columns": ["ordinal", "amount"],
                "winner_rule": "ASCII_CANONICAL_LAST",
            }
            for table in SOURCE_TABLES
        },
        created_at=NOW,
    )


def frame(
    *,
    amount: Any = Decimal("0.1"),
    ordinal: Any = 1,
    note: Any = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [["000001.SZ", "20260630", amount, ordinal, note]],
        columns=["ts_code", "period", "amount", "ordinal", "note"],
    )


def tables(value: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    return {table: (frame() if value is None else value.copy(deep=True)) for table in SOURCE_TABLES}


def test_policy_is_sealed_replayable_and_owner_order_is_semantic() -> None:
    document = policy()
    assert validate_fundamental_comparison_policy(document) == document

    forged = copy.deepcopy(document)
    forged["table_policies"][0]["winner_order_columns"].reverse()
    with pytest.raises(FundamentalV4ContractError):
        validate_fundamental_comparison_policy(forged)


def test_equivalent_order_dtype_null_and_decimal_forms_compare_equal() -> None:
    baseline = tables()
    vip = tables()
    baseline["income"] = pd.DataFrame(
        [
            ["000002.SZ", "20260630", Decimal("1000000000000000000"), 2, None],
            ["000001.SZ", "20260630", Decimal("0.1000000000004"), 1, None],
        ],
        columns=frame().columns,
    )
    vip["income"] = pd.DataFrame(
        [
            ["000001.SZ", "20260630", "1e-1", Decimal("1"), np.nan],
            ["000002.SZ", "20260630", 10**18, np.int64(2), pd.NA],
        ],
        columns=frame().columns,
    )

    result = compare_fundamental_raw_tables(
        baseline_tables=baseline,
        vip_tables=vip,
        policy=policy(),
    )
    assert result["passed"] is True
    assert not any(result["raw_row_diff"].values())
    assert not any(result["raw_value_diff"].values())


@pytest.mark.parametrize(
    ("baseline_value", "vip_value"),
    [
        (Decimal("-0"), Decimal("0.000000000000")),
        (Decimal("1.2345678901234"), Decimal("1.234567890123")),
        (Decimal("1.2345678901235"), Decimal("1.234567890124")),
    ],
)
def test_decimal_projection_uses_12_place_half_even(
    baseline_value: Decimal,
    vip_value: Decimal,
) -> None:
    result = compare_fundamental_raw_tables(
        baseline_tables=tables(frame(amount=baseline_value)),
        vip_tables=tables(frame(amount=vip_value)),
        policy=policy(),
    )
    assert result["passed"] is True


def test_empty_string_is_not_null() -> None:
    result = compare_fundamental_raw_tables(
        baseline_tables=tables(frame(note="")),
        vip_tables=tables(frame(note=None)),
        policy=policy(),
    )
    assert result["passed"] is False
    assert result["raw_row_diff"]["income"]


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), Decimal("NaN")])
def test_nonfinite_numeric_values_fail_closed(value: Any) -> None:
    with pytest.raises(FundamentalV4ContractError):
        compare_fundamental_raw_tables(
            baseline_tables=tables(frame(amount=value)),
            vip_tables=tables(),
            policy=policy(),
        )


def test_one_business_value_or_coverage_row_blocks() -> None:
    changed = tables()
    changed["income"] = frame(amount=Decimal("0.2"))
    value_result = compare_fundamental_raw_tables(
        baseline_tables=tables(),
        vip_tables=changed,
        policy=policy(),
    )
    assert value_result["passed"] is False
    assert value_result["raw_value_diff"]["income"]

    extra = tables()
    extra["income"] = pd.concat([extra["income"], frame()], ignore_index=True)
    coverage_result = compare_fundamental_raw_tables(
        baseline_tables=tables(),
        vip_tables=extra,
        policy=policy(),
    )
    assert coverage_result["passed"] is False
    assert coverage_result["duplicate_diff"]["income"] == {
        "baseline_duplicate_row_count": 0,
        "vip_duplicate_row_count": 1,
    }


def test_restatement_winner_change_is_reported() -> None:
    baseline = tables()
    vip = tables()
    baseline["income"] = pd.concat(
        [frame(amount=Decimal("0.1"), ordinal=1), frame(amount=Decimal("0.2"), ordinal=2)],
        ignore_index=True,
    )
    vip["income"] = pd.concat(
        [frame(amount=Decimal("0.1"), ordinal=1), frame(amount=Decimal("0.3"), ordinal=3)],
        ignore_index=True,
    )
    result = compare_fundamental_raw_tables(
        baseline_tables=baseline,
        vip_tables=vip,
        policy=policy(),
    )
    assert result["passed"] is False
    assert len(result["raw_value_diff"]["income"]) == 1


def test_table_set_and_column_order_are_exact() -> None:
    missing = tables()
    missing.pop("income")
    with pytest.raises(FundamentalV4ContractError):
        compare_fundamental_raw_tables(
            baseline_tables=missing,
            vip_tables=tables(),
            policy=policy(),
        )

    reordered = tables()
    reordered["income"] = reordered["income"][list(reversed(frame().columns))]
    with pytest.raises(FundamentalV4ContractError):
        compare_fundamental_raw_tables(
            baseline_tables=reordered,
            vip_tables=tables(),
            policy=policy(),
        )


def sealed_fixture(version: str, identity_field: str, **values: Any) -> dict[str, Any]:
    return seal(
        {
            **common_fields(timestamp_value=NOW),
            "version": version,
            **values,
        },
        identity_field=identity_field,
    )


def exact_output_ref(name: str, digest: str) -> dict[str, str]:
    return {
        "artifact_id": name,
        "artifact_version": f"{name}.v1",
        "available_at": NOW,
        "byte_sha256": digest,
        "cutoff": NOW,
        "relative_path": f"provider_evidence/comparison_outputs/{name}.json",
        "semantic_sha256": digest,
    }


def reconciliation_closure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline_attempts: int = 100,
) -> dict[str, Any]:
    from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
        reconciliation as module,
    )

    partition_rows = [
        {
            "partition_id": f"partition={index}",
            "table": table,
        }
        for index, table in enumerate(SOURCE_TABLES)
    ]
    plan = sealed_fixture(
        "request-plan.fixture.v1",
        "plan_id",
        baseline_planned_network_attempts=baseline_attempts,
        created_at=NOW,
        partition_rows=partition_rows,
        symbols=["000001.SZ"],
    )
    physical = [
        sealed_fixture(
            "physical.fixture.v1",
            "receipt_id",
            attempts=1,
            partition_id=row["partition_id"],
            table=row["table"],
        )
        for row in partition_rows
    ]
    coverages = [
        sealed_fixture(
            "coverage.fixture.v1",
            "coverage_id",
            company_code="000001.SZ",
            status="COMPLETE",
            table=table,
        )
        for table in SOURCE_TABLES
    ]
    baseline_evidence = [
        sealed_fixture(
            "raw-evidence.fixture.v1",
            "evidence_id",
            lane="BASELINE",
            table=table,
        )
        for table in SOURCE_TABLES
    ]
    vip_evidence = [
        sealed_fixture(
            "raw-evidence.fixture.v1",
            "evidence_id",
            lane="VIP",
            table=table,
        )
        for table in SOURCE_TABLES
    ]
    monkeypatch.setattr(module, "validate_fundamental_request_plan_v4", lambda *a, **k: plan)
    monkeypatch.setattr(
        module,
        "validate_provider_physical_request_receipt_v4",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        module,
        "validate_logical_symbol_table_coverage_v4",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        module,
        "validate_raw_table_evidence_v4",
        lambda value, **kwargs: value,
    )
    comparison_policy = policy()
    raw_tables = tables()
    comparison = compare_fundamental_raw_tables(
        baseline_tables=raw_tables,
        vip_tables=raw_tables,
        policy=comparison_policy,
    )
    derived = {
        name: {"baseline_sha256": "c" * 64, "vip_sha256": "c" * 64}
        for name in ("coverage", "fundamental_daily", "fundamental_period", "quarantine")
    }
    outputs = {
        "coverage_diff": [],
        "duplicate_diff": comparison["duplicate_diff"],
        "raw_row_diff": comparison["raw_row_diff"],
        "raw_value_diff": comparison["raw_value_diff"],
        "derived_fingerprints": derived,
    }
    output_refs = {
        name: exact_output_ref(name, hashlib.sha256(canonical_bytes(value)).hexdigest())
        for name, value in outputs.items()
    }
    return {
        "plan": plan,
        "endpoint_plans": {},
        "physical_receipts": physical,
        "logical_coverages": coverages,
        "baseline_raw_evidence": baseline_evidence,
        "vip_raw_evidence": vip_evidence,
        "baseline_tables": raw_tables,
        "vip_tables": raw_tables,
        "comparison_policy": comparison_policy,
        "comparison_output_refs": output_refs,
        "derived_fingerprints": derived,
    }


def test_reconciliation_receipt_closes_performance_and_full_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = reconciliation_closure(monkeypatch)
    receipt = build_fundamental_reconciliation_receipt(
        **closure,
        reconciled_at=NOW,
    )
    assert receipt["status"] == "PASSED"
    assert receipt["vip_network_attempts"] == 6
    assert receipt["performance_gate_passed"] is True
    assert validate_fundamental_reconciliation_receipt(receipt, **closure) == receipt


def test_reconciliation_performance_failure_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = reconciliation_closure(monkeypatch, baseline_attempts=10)
    receipt = build_fundamental_reconciliation_receipt(
        **closure,
        reconciled_at=NOW,
    )
    assert receipt["status"] == "BLOCKED"
    assert receipt["blocker_codes"] == ["VIP_NETWORK_ATTEMPT_LIMIT_EXCEEDED"]


def test_reconciliation_resealed_or_output_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = reconciliation_closure(monkeypatch)
    receipt = build_fundamental_reconciliation_receipt(
        **closure,
        reconciled_at=NOW,
    )
    forged = copy.deepcopy(receipt)
    forged["status"] = "BLOCKED"
    with pytest.raises(FundamentalV4ContractError):
        validate_fundamental_reconciliation_receipt(forged, **closure)

    mismatched = copy.deepcopy(closure)
    mismatched["comparison_output_refs"] = copy.deepcopy(closure["comparison_output_refs"])
    mismatched["comparison_output_refs"]["raw_row_diff"]["byte_sha256"] = "f" * 64
    with pytest.raises(FundamentalV4ContractError):
        build_fundamental_reconciliation_receipt(
            **mismatched,
            reconciled_at=NOW,
        )


def inventory_rows(overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    digests = dict(overrides or {})
    return [
        {
            "byte_sha256": digests.get(path, "d" * 64),
            "mode": "0600",
            "relative_path": path,
            "semantic_sha256": digests.get(path, "d" * 64),
            "size_bytes": 1,
        }
        for path in REQUIRED_EVIDENCE_PATHS
    ]


def test_provider_fileset_is_exact_sorted_and_excludes_itself() -> None:
    fileset = build_provider_evidence_fileset_manifest(
        inventory=inventory_rows(),
        created_at=NOW,
    )
    assert validate_provider_evidence_fileset_manifest(fileset) == fileset
    assert "fileset_manifest.json" not in {row["relative_path"] for row in fileset["inventory"]}
    with pytest.raises(FundamentalV4ContractError):
        build_provider_evidence_fileset_manifest(
            inventory=inventory_rows()[:-1],
            created_at=NOW,
        )


def write_evidence_fixture(root: Path) -> dict[str, bytes]:
    root.mkdir(mode=0o700)
    payloads = {path: f"payload:{path}".encode() for path in REQUIRED_EVIDENCE_PATHS}
    rows = []
    for relative_path, payload in sorted(payloads.items()):
        path = root / relative_path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        path.write_bytes(payload)
        os.chmod(path, 0o600)
        digest = hashlib.sha256(payload).hexdigest()
        rows.append(
            {
                "byte_sha256": digest,
                "mode": "0600",
                "relative_path": relative_path,
                "semantic_sha256": digest,
                "size_bytes": len(payload),
            }
        )
    fileset = build_provider_evidence_fileset_manifest(
        inventory=rows,
        created_at=NOW,
    )
    manifest_bytes = canonical_bytes(fileset)
    (root / "fileset_manifest.json").write_bytes(manifest_bytes)
    os.chmod(root / "fileset_manifest.json", 0o600)
    return {**payloads, "fileset_manifest.json": manifest_bytes}


def test_provider_evidence_storage_captures_exact_private_fileset(tmp_path: Path) -> None:
    root = tmp_path / "provider_evidence"
    expected = write_evidence_fixture(root)
    assert capture_provider_evidence_directory(root) == expected

    os.chmod(root / "request_receipts.jsonl", 0o644)
    with pytest.raises(FundamentalV4ContractError):
        capture_provider_evidence_directory(root)


def test_provider_evidence_storage_rejects_extra_and_hardlinked_files(tmp_path: Path) -> None:
    root = tmp_path / "provider_evidence"
    write_evidence_fixture(root)
    extra = root / "extra.json"
    extra.write_bytes(b"extra")
    os.chmod(extra, 0o600)
    with pytest.raises(FundamentalV4ContractError):
        capture_provider_evidence_directory(root)
    extra.unlink()

    target = root / "request_receipts.jsonl"
    target.unlink()
    os.link(root / "execution_plan.json", target)
    with pytest.raises(FundamentalV4ContractError):
        capture_provider_evidence_directory(root)


def manifest_closure(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
        manifest as module,
    )

    plan = sealed_fixture(
        "request-plan.fixture.v1",
        "plan_id",
        as_of="20260807",
        pit_cutoff=NOW,
        symbol_set_sha256="e" * 64,
        symbols=["000001.SZ"],
    )
    execution = sealed_fixture(
        "execution-closure.fixture.v1",
        "closure_id",
        request_plan=plan,
    )
    comparison_policy = policy()
    output_names = (
        "coverage_diff",
        "derived_fingerprints",
        "duplicate_diff",
        "raw_row_diff",
        "raw_value_diff",
    )
    output_refs = {
        name: exact_output_ref(name, str(index) * 64)
        for index, name in enumerate(output_names, start=1)
    }
    receipt = sealed_fixture(
        "reconciliation.fixture.v1",
        "receipt_id",
        baseline_network_attempts=100,
        comparison_output_refs=output_refs,
        performance_gate_passed=True,
        plan_ref=content_ref(plan, identity_field="plan_id"),
        status="PASSED",
        vip_network_attempts=6,
    )
    baseline_raw: list[dict[str, Any]] = []
    vip_raw: list[dict[str, Any]] = []
    digest_overrides = {
        "execution_plan.json": hashlib.sha256(canonical_bytes(execution)).hexdigest(),
        "reconciliation.json": hashlib.sha256(canonical_bytes(receipt)).hexdigest(),
        "comparison_policy.json": hashlib.sha256(canonical_bytes(comparison_policy)).hexdigest(),
        "request_receipts.jsonl": "a" * 64,
        "logical_coverage.parquet": "b" * 64,
    }
    for name, ref in output_refs.items():
        digest_overrides[f"comparison_outputs/{name}.json"] = ref["byte_sha256"]
    for lane, collection in (("baseline_raw", baseline_raw), ("vip_raw", vip_raw)):
        for index, table in enumerate(SOURCE_TABLES):
            digest = ("a" if lane == "baseline_raw" else "b") + f"{index:063x}"
            relative_path = f"{lane}/{table}.parquet"
            digest_overrides[relative_path] = digest
            collection.append(
                sealed_fixture(
                    "raw.fixture.v1",
                    "evidence_id",
                    canonical_multiset_sha256=("c" + f"{index:063x}"),
                    file_ref={
                        "artifact_id": f"{lane}-{table}",
                        "artifact_version": "raw.v1",
                        "available_at": NOW,
                        "byte_sha256": digest,
                        "cutoff": NOW,
                        "relative_path": f"provider_evidence/{relative_path}",
                        "semantic_sha256": digest,
                    },
                    lane="BASELINE" if lane == "baseline_raw" else "VIP",
                    table=table,
                )
            )
    fileset = build_provider_evidence_fileset_manifest(
        inventory=inventory_rows(digest_overrides),
        created_at=NOW,
    )
    monkeypatch.setattr(
        module,
        "validate_fundamental_execution_closure_v4",
        lambda *a, **k: execution,
    )
    monkeypatch.setattr(
        module,
        "validate_fundamental_reconciliation_receipt",
        lambda *a, **k: receipt,
    )
    return {
        "execution_closure": execution,
        "reconciliation": receipt,
        "reconciliation_closure": {
            "baseline_raw_evidence": baseline_raw,
            "comparison_policy": comparison_policy,
            "vip_raw_evidence": vip_raw,
        },
        "fileset": fileset,
        "request_receipts_ref": {
            "artifact_id": "request-receipts",
            "artifact_version": "jsonl.v1",
            "available_at": NOW,
            "byte_sha256": "a" * 64,
            "cutoff": NOW,
            "relative_path": "provider_evidence/request_receipts.jsonl",
            "semantic_sha256": "a" * 64,
        },
        "logical_coverage_ref": {
            "artifact_id": "logical-coverage",
            "artifact_version": "parquet.v1",
            "available_at": NOW,
            "byte_sha256": "b" * 64,
            "cutoff": NOW,
            "relative_path": "provider_evidence/logical_coverage.parquet",
            "semantic_sha256": "b" * 64,
        },
    }


def test_provider_manifest_requires_passed_reconciliation_and_durable_fileset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = manifest_closure(monkeypatch)
    manifest = build_fundamental_provider_manifest_v4(
        **closure,
        created_at=NOW,
    )
    assert manifest["schema_version"] == "cn-fundamental-provider-manifest.v4"
    assert manifest["authoritative_full_rebuild"] is True
    assert validate_fundamental_provider_manifest_v4(manifest, **closure) == manifest

    forged = copy.deepcopy(manifest)
    forged["vip_network_attempts"] = 7
    with pytest.raises(FundamentalV4ContractError):
        validate_fundamental_provider_manifest_v4(forged, **closure)
