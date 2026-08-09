"""Assemble and persist the exact Fundamental VIP shadow evidence fileset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
import os
from pathlib import Path
import stat
from typing import Any

import pandas as pd

from ...._core import canonical_bytes, content_ref
from .comparison import (
    compare_fundamental_raw_tables,
    validate_fundamental_comparison_policy,
)
from .contracts import (
    _build_logical_coverage_validated,
    _plan_and_endpoints,
    _partition_ids,
    _validate_physical_request_receipt_validated,
    build_raw_table_evidence_v4,
)
from .fileset import (
    REQUIRED_EVIDENCE_PATHS,
    build_provider_evidence_fileset_manifest,
)
from .manifest import build_fundamental_provider_manifest_v4
from .manifest import validate_fundamental_provider_manifest_v4
from .models import SOURCE_TABLES, FundamentalV4ContractError, fundamental_v4_contract
from .reconciliation import build_fundamental_reconciliation_receipt
from .reconciliation import _logical_coverages as _replay_logical_coverages
from .schedule import validate_fundamental_execution_closure_v4
from .storage import capture_provider_evidence_directory


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _exact_file_ref(
    *,
    artifact_id: str,
    relative_path: str,
    payload: bytes,
    semantic_sha256: str,
    cutoff: str,
    available_at: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": "myquant.v17.provider-evidence-file.v1",
        "available_at": available_at,
        "byte_sha256": hashlib.sha256(payload).hexdigest(),
        "cutoff": cutoff,
        "relative_path": f"provider_evidence/{relative_path}",
        "semantic_sha256": semantic_sha256,
    }


def _validated_receipts(
    values: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validated_plan, endpoints = _plan_and_endpoints(
        plan=plan,
        endpoint_plans=endpoint_plans,
    )
    plan_ref = content_ref(validated_plan, identity_field="plan_id")
    rows = [
        _validate_physical_request_receipt_validated(
            value,
            validated_plan=validated_plan,
            validated_plan_ref=plan_ref,
            endpoints=endpoints,
        )
        for value in values
    ]
    expected = {(row["table"], row["partition_id"]) for row in plan["partition_rows"]}
    actual = {(row["table"], row["partition_id"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise FundamentalV4ContractError("shadow physical receipt keyset is incomplete")
    return sorted(rows, key=lambda row: row["receipt_id"].encode("ascii"))


def _validated_coverages(
    values: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    physical_receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _replay_logical_coverages(
        values,
        plan=plan,
        endpoint_plans=endpoint_plans,
        physical_receipts=physical_receipts,
    )


@fundamental_v4_contract
def build_logical_coverages_from_shadow_v4(
    *,
    execution_closure: Mapping[str, Any],
    physical_receipts: Sequence[Mapping[str, Any]],
    vip_tables: Mapping[str, pd.DataFrame],
    assessed_at: str,
) -> tuple[dict[str, Any], ...]:
    """Project the batch VIP result into the exact full-scope logical keyset."""

    execution = validate_fundamental_execution_closure_v4(execution_closure)
    plan = execution["request_plan"]
    endpoint_plans = execution["endpoint_plans"]
    physical = _validated_receipts(
        physical_receipts,
        plan=plan,
        endpoint_plans=endpoint_plans,
    )
    tables = _raw_tables(vip_tables, label="VIP")
    expected_symbols = set(plan["symbols"])
    physical_by_table = {
        table: [row for row in physical if row["table"] == table] for table in SOURCE_TABLES
    }
    coverages: list[dict[str, Any]] = []
    for table in SOURCE_TABLES:
        frame = tables[table]
        date_column = "trade_date" if table == "daily_basic" else "end_date"
        if "ts_code" not in frame.columns or date_column not in frame.columns:
            raise FundamentalV4ContractError("shadow raw identity columns are missing")
        observed_symbols = set(frame["ts_code"].dropna().tolist())
        if any(
            type(symbol) is not str for symbol in observed_symbols
        ) or not observed_symbols.issubset(expected_symbols):
            raise FundamentalV4ContractError("shadow raw table contains an unknown symbol")
        if table == "daily_basic" and observed_symbols != expected_symbols:
            raise FundamentalV4ContractError("VIP daily full-A symbol scope is incomplete")
        start = plan["daily_start"] if table == "daily_basic" else plan["financial_start"]
        end = plan["as_of"]
        receipts = physical_by_table[table]
        partition_ids = _partition_ids(
            plan,
            table=table,
            expected_start=start,
            expected_end=end,
        )
        status = (
            "COMPLETE"
            if all(row["status"] in {"AVAILABLE", "EMPTY"} for row in receipts)
            else "INCOMPLETE"
        )
        grouped = {
            symbol: sorted(frame.loc[frame["ts_code"] == symbol, date_column].dropna().tolist())
            for symbol in plan["symbols"]
        }
        for symbol in plan["symbols"]:
            dates = grouped[symbol]
            coverages.append(
                _build_logical_coverage_validated(
                    validated_plan=plan,
                    receipts=receipts,
                    expected_partition_ids=partition_ids,
                    company_code=symbol,
                    table=table,
                    expected_start=start,
                    expected_end=end,
                    observed_start=None if not dates else dates[0],
                    observed_end=None if not dates else dates[-1],
                    row_count=len(frame.loc[frame["ts_code"] == symbol]),
                    missing_reason_codes=[],
                    duplicate_reason_codes=[],
                    restatement_reason_codes=[],
                    status=status,
                    assessed_at=assessed_at,
                )
            )
    return tuple(sorted(coverages, key=lambda row: row["coverage_id"].encode("ascii")))


def _raw_tables(
    value: Mapping[str, pd.DataFrame],
    *,
    label: str,
) -> dict[str, pd.DataFrame]:
    if type(value) is not dict or set(value) != set(SOURCE_TABLES):
        raise FundamentalV4ContractError(f"{label} raw table set is invalid")
    result: dict[str, pd.DataFrame] = {}
    for table in SOURCE_TABLES:
        frame = value[table]
        if not isinstance(frame, pd.DataFrame):
            raise FundamentalV4ContractError(f"{label} raw table is not a DataFrame")
        result[table] = frame.copy(deep=True)
    return result


def _derived_fingerprints(
    value: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    expected = {"coverage", "fundamental_daily", "fundamental_period", "quarantine"}
    if type(value) is not dict or set(value) != expected:
        raise FundamentalV4ContractError("shadow derived fingerprint set is invalid")
    result: dict[str, dict[str, str]] = {}
    for table in sorted(expected):
        row = value[table]
        if type(row) is not dict or set(row) != {"baseline_sha256", "vip_sha256"}:
            raise FundamentalV4ContractError("shadow derived fingerprint row is invalid")
        normalized: dict[str, str] = {}
        for key in ("baseline_sha256", "vip_sha256"):
            digest = row[key]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise FundamentalV4ContractError("shadow derived fingerprint is invalid")
            normalized[key] = digest
        result[table] = normalized
    return result


def _stable_regular_bytes(path: Path, *, expected_sha256: str) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise FundamentalV4ContractError("shadow membership evidence path is invalid")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise FundamentalV4ContractError("shadow membership evidence file is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise FundamentalV4ContractError("shadow membership evidence changed")
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise FundamentalV4ContractError("shadow membership evidence SHA mismatch")
    return payload


def _derived_projection_fingerprints(
    *,
    baseline_derived: Mapping[str, pd.DataFrame],
    vip_derived: Mapping[str, pd.DataFrame],
    baseline_tables: Mapping[str, pd.DataFrame],
    vip_tables: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, str]]:
    from quant_investor.market.fundamental_generation import (
        _deterministic_replay_projection,
    )
    from quant_investor.market.fundamental_provider_contract import frame_fingerprint

    result: dict[str, dict[str, str]] = {}
    names = {
        "fundamental_daily": "fundamental_daily",
        "fundamental_period": "fundamental_period",
        "quarantine": "fundamental_quarantine",
    }
    for output_name, table_name in names.items():
        result[output_name] = {
            "baseline_sha256": frame_fingerprint(
                _deterministic_replay_projection(
                    baseline_derived[table_name],
                    table_name=table_name,
                )
            ),
            "vip_sha256": frame_fingerprint(
                _deterministic_replay_projection(
                    vip_derived[table_name],
                    table_name=table_name,
                )
            ),
        }
    coverage_rows = {
        lane: {
            table: {
                "row_count": len(tables[table]),
                "symbols": sorted(set(tables[table]["ts_code"].dropna().tolist())),
            }
            for table in SOURCE_TABLES
        }
        for lane, tables in (("baseline", baseline_tables), ("vip", vip_tables))
    }
    result["coverage"] = {
        "baseline_sha256": hashlib.sha256(canonical_bytes(coverage_rows["baseline"])).hexdigest(),
        "vip_sha256": hashlib.sha256(canonical_bytes(coverage_rows["vip"])).hexdigest(),
    }
    return result


@fundamental_v4_contract
def derive_fundamental_shadow_v4(
    *,
    execution_closure: Mapping[str, Any],
    baseline_tables: Mapping[str, pd.DataFrame],
    vip_tables: Mapping[str, pd.DataFrame],
    membership_path: str | Path,
    membership_sha256: str,
    run_id: str,
    derivation_timestamp: str,
    non_blocking_absent_symbols: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the existing formal v3 derivation once per lane for exact comparison."""

    from quant_investor.market.fundamental_mart import rederive_fundamental_tables_v3

    execution = validate_fundamental_execution_closure_v4(execution_closure)
    plan = execution["request_plan"]
    baseline = _raw_tables(baseline_tables, label="baseline")
    vip = _raw_tables(vip_tables, label="VIP")
    membership = _stable_regular_bytes(
        Path(membership_path),
        expected_sha256=membership_sha256,
    )
    closure = {
        "membership_bytes": membership,
        "membership_sha256": membership_sha256,
        "as_of": plan["as_of"],
        "symbols": plan["symbols"],
        "non_blocking_absent_symbols": list(non_blocking_absent_symbols),
        "run_id": run_id,
        "source": "live_tushare_vip",
        "derivation_timestamp": derivation_timestamp,
    }
    baseline_derived, baseline_evidence = rederive_fundamental_tables_v3(
        baseline,
        **closure,
    )
    vip_derived, vip_evidence = rederive_fundamental_tables_v3(
        vip,
        **closure,
    )
    return {
        "baseline_derivation_evidence": baseline_evidence,
        "baseline_derived_tables": baseline_derived,
        "derived_fingerprints": _derived_projection_fingerprints(
            baseline_derived=baseline_derived,
            vip_derived=vip_derived,
            baseline_tables=baseline,
            vip_tables=vip,
        ),
        "vip_derivation_evidence": vip_evidence,
        "vip_derived_tables": vip_derived,
    }


def _raw_payloads(
    *,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    baseline_tables: Mapping[str, pd.DataFrame],
    vip_tables: Mapping[str, pd.DataFrame],
    comparison: Mapping[str, Any],
    comparison_policy: Mapping[str, Any],
    evidenced_at: str,
) -> tuple[dict[str, bytes], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    payloads: dict[str, bytes] = {}
    semantic: dict[str, str] = {}
    baseline_evidence: list[dict[str, Any]] = []
    vip_evidence: list[dict[str, Any]] = []
    policies = {row["table"]: row for row in comparison_policy["table_policies"]}
    for lane, tables, evidence_rows in (
        ("BASELINE", baseline_tables, baseline_evidence),
        ("VIP", vip_tables, vip_evidence),
    ):
        directory = f"{lane.lower()}_raw"
        for table in SOURCE_TABLES:
            relative_path = f"{directory}/{table}.parquet"
            payload = _parquet_bytes(tables[table])
            table_evidence = comparison["table_evidence"][table]
            multiset_key = (
                "baseline_multiset_sha256" if lane == "BASELINE" else "vip_multiset_sha256"
            )
            multiset_sha = table_evidence[multiset_key]
            payloads[relative_path] = payload
            semantic[relative_path] = multiset_sha
            evidence_rows.append(
                build_raw_table_evidence_v4(
                    plan=plan,
                    endpoint_plans=endpoint_plans,
                    lane=lane,
                    table=table,
                    file_ref=_exact_file_ref(
                        artifact_id=f"fundamental-{lane.lower()}-{table}",
                        relative_path=relative_path,
                        payload=payload,
                        semantic_sha256=multiset_sha,
                        cutoff=plan["pit_cutoff"],
                        available_at=evidenced_at,
                    ),
                    row_count=len(tables[table]),
                    column_order=list(tables[table].columns),
                    canonical_multiset_sha256=multiset_sha,
                    duplicate_row_count=comparison["duplicate_diff"][table][
                        f"{lane.lower()}_duplicate_row_count"
                    ],
                    winner_implementation_sha256=policies[table]["winner_implementation_sha256"],
                    evidenced_at=evidenced_at,
                )
            )
    return payloads, baseline_evidence, vip_evidence, semantic


def _comparison_payloads(
    *,
    comparison: Mapping[str, Any],
    coverages: Sequence[Mapping[str, Any]],
    derived: Mapping[str, Mapping[str, str]],
) -> dict[str, bytes]:
    coverage_diff = [
        {"coverage_id": row["coverage_id"], "status": row["status"]}
        for row in coverages
        if row["status"] != "COMPLETE"
    ]
    return {
        "comparison_outputs/coverage_diff.json": canonical_bytes(coverage_diff),
        "comparison_outputs/derived_fingerprints.json": canonical_bytes(dict(derived)),
        "comparison_outputs/duplicate_diff.json": canonical_bytes(comparison["duplicate_diff"]),
        "comparison_outputs/raw_row_diff.json": canonical_bytes(comparison["raw_row_diff"]),
        "comparison_outputs/raw_value_diff.json": canonical_bytes(comparison["raw_value_diff"]),
    }


def _document_parquet_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    frame = pd.DataFrame(
        {"document_json": [canonical_bytes(dict(value)).decode("utf-8") for value in values]}
    )
    return _parquet_bytes(frame)


def _inventory(
    payloads: Mapping[str, bytes],
    *,
    semantic_sha256: Mapping[str, str],
) -> list[dict[str, Any]]:
    if set(payloads) != set(REQUIRED_EVIDENCE_PATHS):
        raise FundamentalV4ContractError("shadow evidence payload set is incomplete")
    return [
        {
            "byte_sha256": hashlib.sha256(payloads[path]).hexdigest(),
            "mode": "0600",
            "relative_path": path,
            "semantic_sha256": semantic_sha256.get(
                path,
                hashlib.sha256(payloads[path]).hexdigest(),
            ),
            "size_bytes": len(payloads[path]),
        }
        for path in sorted(payloads)
    ]


@fundamental_v4_contract
def build_fundamental_shadow_bundle_v4(
    *,
    execution_closure: Mapping[str, Any],
    physical_receipts: Sequence[Mapping[str, Any]],
    logical_coverages: Sequence[Mapping[str, Any]],
    baseline_tables: Mapping[str, pd.DataFrame],
    vip_tables: Mapping[str, pd.DataFrame],
    comparison_policy: Mapping[str, Any],
    derived_fingerprints: Mapping[str, Mapping[str, Any]],
    assembled_at: str,
) -> dict[str, Any]:
    """Build the complete durable v4 evidence bundle without writing a pointer."""

    execution = validate_fundamental_execution_closure_v4(execution_closure)
    plan = execution["request_plan"]
    endpoint_plans = execution["endpoint_plans"]
    physical = _validated_receipts(
        physical_receipts,
        plan=plan,
        endpoint_plans=endpoint_plans,
    )
    coverages = _validated_coverages(
        logical_coverages,
        plan=plan,
        endpoint_plans=endpoint_plans,
        physical_receipts=physical,
    )
    baseline = _raw_tables(baseline_tables, label="baseline")
    vip = _raw_tables(vip_tables, label="VIP")
    policy = validate_fundamental_comparison_policy(comparison_policy)
    comparison = compare_fundamental_raw_tables(
        baseline_tables=baseline,
        vip_tables=vip,
        policy=policy,
    )
    derived = _derived_fingerprints(derived_fingerprints)
    raw_payloads, baseline_evidence, vip_evidence, semantic = _raw_payloads(
        plan=plan,
        endpoint_plans=endpoint_plans,
        baseline_tables=baseline,
        vip_tables=vip,
        comparison=comparison,
        comparison_policy=policy,
        evidenced_at=assembled_at,
    )
    payloads = {
        **raw_payloads,
        **_comparison_payloads(
            comparison=comparison,
            coverages=coverages,
            derived=derived,
        ),
        "comparison_policy.json": canonical_bytes(policy),
        "execution_plan.json": canonical_bytes(execution),
        "logical_coverage.parquet": _document_parquet_bytes(coverages),
        "request_receipts.jsonl": b"".join(canonical_bytes(row) + b"\n" for row in physical),
    }
    semantic.update(
        {
            "logical_coverage.parquet": hashlib.sha256(canonical_bytes(coverages)).hexdigest(),
        }
    )
    output_refs = {
        name: _exact_file_ref(
            artifact_id=f"fundamental-comparison-{name}",
            relative_path=f"comparison_outputs/{name}.json",
            payload=payloads[f"comparison_outputs/{name}.json"],
            semantic_sha256=hashlib.sha256(payloads[f"comparison_outputs/{name}.json"]).hexdigest(),
            cutoff=plan["pit_cutoff"],
            available_at=assembled_at,
        )
        for name in (
            "coverage_diff",
            "derived_fingerprints",
            "duplicate_diff",
            "raw_row_diff",
            "raw_value_diff",
        )
    }
    reconciliation_closure = {
        "baseline_raw_evidence": baseline_evidence,
        "baseline_tables": baseline,
        "comparison_output_refs": output_refs,
        "comparison_policy": policy,
        "derived_fingerprints": derived,
        "endpoint_plans": endpoint_plans,
        "logical_coverages": coverages,
        "physical_receipts": physical,
        "plan": plan,
        "vip_raw_evidence": vip_evidence,
        "vip_tables": vip,
    }
    reconciliation = build_fundamental_reconciliation_receipt(
        **reconciliation_closure,
        reconciled_at=assembled_at,
    )
    payloads["reconciliation.json"] = canonical_bytes(reconciliation)
    fileset = build_provider_evidence_fileset_manifest(
        inventory=_inventory(payloads, semantic_sha256=semantic),
        created_at=assembled_at,
    )
    request_payload = payloads["request_receipts.jsonl"]
    coverage_payload = payloads["logical_coverage.parquet"]
    manifest = None
    if reconciliation["status"] == "PASSED":
        manifest = build_fundamental_provider_manifest_v4(
            execution_closure=execution,
            reconciliation=reconciliation,
            reconciliation_closure=reconciliation_closure,
            fileset=fileset,
            request_receipts_ref=_exact_file_ref(
                artifact_id="fundamental-request-receipts",
                relative_path="request_receipts.jsonl",
                payload=request_payload,
                semantic_sha256=hashlib.sha256(request_payload).hexdigest(),
                cutoff=plan["pit_cutoff"],
                available_at=assembled_at,
            ),
            logical_coverage_ref=_exact_file_ref(
                artifact_id="fundamental-logical-coverage",
                relative_path="logical_coverage.parquet",
                payload=coverage_payload,
                semantic_sha256=semantic["logical_coverage.parquet"],
                cutoff=plan["pit_cutoff"],
                available_at=assembled_at,
            ),
            created_at=assembled_at,
        )
    return {
        "fileset": fileset,
        "payloads": {**payloads, "fileset_manifest.json": canonical_bytes(fileset)},
        "provider_manifest": manifest,
        "reconciliation": reconciliation,
        "reconciliation_closure": reconciliation_closure,
        "status": reconciliation["status"],
    }


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FundamentalV4ContractError("shadow evidence write stalled")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bundle_payloads(root: Path, payloads: Mapping[str, bytes]) -> None:
    for relative_path in sorted(payloads):
        path = root / relative_path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        _write_private(path, payloads[relative_path])
    directory_paths = sorted(
        {path.parent for path in root.rglob("*") if path.parent != root},
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory_path in directory_paths:
        _fsync_directory(directory_path)
    _fsync_directory(root)


@fundamental_v4_contract
def materialize_fundamental_v4_staging_generation(
    *,
    execution_closure: Mapping[str, Any],
    bundle: Mapping[str, Any],
    vip_tables: Mapping[str, pd.DataFrame],
    vip_derived_tables: Mapping[str, pd.DataFrame],
    data_root: str | Path,
    raw_snapshot_root: str | Path,
    reports_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Create one isolated, promotion-ready generation without touching canonical data."""

    from quant_investor.market.fundamental_mart import (
        _issue_live_tushare_v4_attestation,
        write_fundamental_mart,
    )

    execution = validate_fundamental_execution_closure_v4(execution_closure)
    provider_manifest = bundle.get("provider_manifest")
    if bundle.get("status") != "PASSED" or type(provider_manifest) is not dict:
        raise FundamentalV4ContractError("blocked shadow bundle cannot materialize staging")
    reconciliation = bundle.get("reconciliation")
    reconciliation_closure = bundle.get("reconciliation_closure")
    fileset = bundle.get("fileset")
    payloads = bundle.get("payloads")
    if (
        type(reconciliation) is not dict
        or type(reconciliation_closure) is not dict
        or type(fileset) is not dict
        or type(payloads) is not dict
    ):
        raise FundamentalV4ContractError("shadow staging closure is incomplete")
    validated_manifest = validate_fundamental_provider_manifest_v4(
        provider_manifest,
        execution_closure=execution,
        reconciliation=reconciliation,
        reconciliation_closure=reconciliation_closure,
        fileset=fileset,
        request_receipts_ref=provider_manifest["request_receipts_ref"],
        logical_coverage_ref=provider_manifest["logical_coverage_ref"],
    )
    tables = _raw_tables(vip_tables, label="VIP")
    roots = [Path(value) for value in (data_root, raw_snapshot_root, reports_root)]
    if any(
        not root.is_absolute() or ".." in root.parts or root.exists() or root.is_symlink()
        for root in roots
    ):
        raise FundamentalV4ContractError("shadow staging roots must be new absolute paths")
    attestation = _issue_live_tushare_v4_attestation(
        "live_tushare_vip",
        validated_manifest,
        tables,
    )
    published: dict[str, Any] = {}
    artifacts, readiness = write_fundamental_mart(
        tables,
        data_root=roots[0],
        raw_snapshot_root=roots[1],
        reports_root=roots[2],
        run_id=run_id,
        source="live_tushare_vip",
        provider_manifest=validated_manifest,
        write_raw_snapshots=False,
        require_expected_symbol_scope=True,
        publish_on_gate_failure=False,
        _live_tushare_attestation=attestation,
        _derived_tables_v3=vip_derived_tables,
        _provider_evidence_bytes=payloads,
        _published_pointer_out=published,
    )
    if (
        readiness.get("gate2_passed") is not True
        or published.get("primary_provenance_verified") is not True
        or published.get("generation_id") != run_id
    ):
        raise FundamentalV4ContractError("shadow staging generation readback failed")
    evidence_root = roots[0] / Path(str(published["manifest_path"])).parent / "provider_evidence"
    if capture_provider_evidence_directory(evidence_root) != payloads:
        raise FundamentalV4ContractError("shadow staging evidence changed")
    return {
        "artifacts": {
            key: str(value) for key, value in artifacts.__dict__.items() if key.endswith("_path")
        },
        "generation_id": run_id,
        "pointer": published,
        "provider_manifest_sha256": validated_manifest["semantic_sha256"],
        "readiness": readiness,
        "status": "STAGING_READY",
    }


@fundamental_v4_contract
def write_fundamental_shadow_bundle_v4(
    *,
    bundle: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Exact-write a built bundle to a new private provider_evidence root."""

    root = Path(output_root)
    if not root.is_absolute() or ".." in root.parts or root.exists() or root.is_symlink():
        raise FundamentalV4ContractError("shadow evidence output root is unsafe")
    payloads = bundle.get("payloads")
    if type(payloads) is not dict or set(payloads) != {
        *REQUIRED_EVIDENCE_PATHS,
        "fileset_manifest.json",
    }:
        raise FundamentalV4ContractError("shadow evidence bundle payload set is invalid")
    root.mkdir(mode=0o700)
    try:
        _write_bundle_payloads(root, payloads)
        captured = capture_provider_evidence_directory(root)
        if captured != payloads:
            raise FundamentalV4ContractError("shadow evidence readback mismatch")
    except Exception:
        # Preserve the incomplete root as forensic evidence; never auto-clean it.
        raise
    metadata = os.lstat(root)
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise FundamentalV4ContractError("shadow evidence root mode changed")
    provider_manifest = bundle["provider_manifest"]
    return {
        "fileset_sha256": bundle["fileset"]["fileset_sha256"],
        "output_root": str(root),
        "provider_manifest_sha256": (
            None if provider_manifest is None else provider_manifest["semantic_sha256"]
        ),
        "status": bundle["status"],
    }


__all__ = [
    "build_fundamental_shadow_bundle_v4",
    "build_logical_coverages_from_shadow_v4",
    "derive_fundamental_shadow_v4",
    "materialize_fundamental_v4_staging_generation",
    "write_fundamental_shadow_bundle_v4",
]
