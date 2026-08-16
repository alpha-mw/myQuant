"""Auditable isolation of pre-successor Fundamental source ambiguities.

This module does not select a winner for an ambiguous provider response.  It
proves only that an independently replayable, pre-cutoff conflict is already
behind an immutable predecessor boundary and that the exact successor delta
does not touch the poisoned ``(symbol, report_period)`` key.  Any conflict in
the successor window, any predecessor mismatch, or any same-period delta event
remains a hard blocker.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn, Sequence

import pandas as pd
import pyarrow.parquet as pq

from quant_investor.market.fundamental_provider_contract import (
    canonical_json_sha256,
)
from quant_investor.market.fundamental_successor_source import (
    open_support_tables,
    validate_successor_failure_evidence,
    validate_successor_support_fileset,
)
from quant_investor.market.tushare_transport import (
    replay_tushare_response_bytes,
)


HISTORICAL_TAINT_REGISTRY_SCHEMA = (
    "cn-fundamental-historical-taint-registry.v1"
)
HISTORICAL_TAINT_CLASSIFICATION = (
    "HISTORICAL_AMBIGUITY_NON_CURRENT_DELTA"
)
HISTORICAL_TAINT_STATUS = "VALID_WITH_HISTORICAL_TAINT"
_FINANCIAL_TABLES = (
    "fina_indicator",
    "income",
    "balancesheet",
    "cashflow",
)
_PERIOD_COLUMNS = (
    "ts_code",
    "end_date",
    "availability_date",
    "source_version",
    "source",
    "fetched_at",
    "fin_roe",
    "fin_roa",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_fcf_to_profit",
    "free_cashflow",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


class HistoricalTaintError(RuntimeError):
    """One fail-closed append-first historical-taint blocker."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}" if message else self.code)


def _fail(code: str, message: str = "") -> NoReturn:
    raise HistoricalTaintError(code, message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _date(value: Any, *, label: str) -> str:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        resolved = value.strftime("%Y%m%d")
    else:
        resolved = str(value or "").replace("-", "")
    try:
        datetime.strptime(resolved, "%Y%m%d")
    except ValueError:
        _fail("HISTORICAL_TAINT_DATE_INVALID", label)
    return resolved


def _next_date(value: str) -> str:
    resolved = datetime.strptime(value, "%Y%m%d").date() + timedelta(days=1)
    return resolved.strftime("%Y%m%d")


def _scalar(value: Any) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, Decimal):
        return {"kind": "decimal", "value": str(value)}
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return {"kind": "date", "value": value.strftime("%Y%m%d")}
    if type(value) in {str, int, bool}:
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            _fail("HISTORICAL_TAINT_NONFINITE_VALUE")
        return {"kind": "float", "value": repr(value)}
    return {"kind": type(value).__name__, "value": str(value)}


def _row_sha256(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    return canonical_json_sha256(
        {field: _scalar(row.get(field)) for field in fields}
    )


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        _fail("HISTORICAL_TAINT_EVIDENCE_INVALID", str(path))
    if type(value) is not dict:
        _fail("HISTORICAL_TAINT_EVIDENCE_INVALID", str(path))
    return value, payload


def _failure_layout(
    failure_root: str | Path,
    *,
    ordinal: int,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    root = Path(failure_root).expanduser().resolve(strict=True)
    failure = validate_successor_failure_evidence(root, ordinal=ordinal)
    if failure.get("error_code") != "SUCCESSOR_MATERIAL_DUPLICATE_CONFLICT":
        _fail("HISTORICAL_TAINT_ERROR_NOT_ELIGIBLE")
    request = dict(failure.get("request", {}) or {})
    if request.get("table") not in _FINANCIAL_TABLES:
        _fail("HISTORICAL_TAINT_TABLE_NOT_ELIGIBLE")
    if not root.name.endswith("-failures"):
        _fail("HISTORICAL_TAINT_FAILURE_LAYOUT_INVALID")
    source_root = root.parent / root.name.removesuffix("-failures")
    binding_path = source_root / "binding.json"
    failure_path = root / f"{ordinal:06d}.failure.json"
    raw_path = root / str(dict(failure["raw_response_ref"])["path"])
    for path in (binding_path, failure_path, raw_path):
        if not path.is_file() or path.is_symlink():
            _fail("HISTORICAL_TAINT_EVIDENCE_INVALID", str(path))
    return root, binding_path, failure_path, raw_path, failure


def _material_conflicts(
    *,
    failure: Mapping[str, Any],
    raw_bytes: bytes,
) -> list[dict[str, Any]]:
    request = dict(failure["request"])
    fields = tuple(str(value) for value in request["expected_fields"])
    response = replay_tushare_response_bytes(
        raw_bytes,
        api_name=str(request["endpoint"]),
        expected_fields=fields,
        strict_decimal_decode=True,
        max_response_items=int(request["row_ceiling"]),
    )
    rows = [dict(zip(fields, values, strict=True)) for values in response.rows]
    logical: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    table = str(request["table"])
    for row in rows:
        symbol = str(row.get("ts_code") or "")
        end_date = _date(row.get("end_date"), label="failure end_date")
        availability = _date(
            row.get("f_ann_date") or row.get("ann_date"),
            label="failure availability",
        )
        identity: tuple[str, ...]
        if table in {"income", "balancesheet", "cashflow"}:
            identity = (
                str(row.get("report_type") or ""),
                str(row.get("comp_type") or ""),
            )
        else:
            identity = (str(row.get("update_flag") or "UNVERSIONED"),)
        row["__physical_identity"] = identity
        logical.setdefault((symbol, end_date, availability), []).append(row)
    conflicts: list[dict[str, Any]] = []
    projection_fields = tuple(
        field
        for field in fields
        if field not in {"report_type", "comp_type", "update_flag"}
    )
    for key in sorted(logical):
        physical: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in logical[key]:
            physical.setdefault(tuple(row["__physical_identity"]), []).append(row)
        survivors: list[dict[str, Any]] = []
        for candidates in physical.values():
            highest = max(int(row.get("update_flag") or -1) for row in candidates)
            survivors.extend(
                row
                for row in candidates
                if int(row.get("update_flag") or -1) == highest
            )
        projections: dict[str, list[dict[str, Any]]] = {}
        for row in survivors:
            digest = _row_sha256(row, projection_fields)
            projections.setdefault(digest, []).append(row)
        if len(projections) < 2:
            continue
        candidates = [
            {
                "comp_type": str(row.get("comp_type") or ""),
                "projection_sha256": _row_sha256(row, projection_fields),
                "report_type": str(row.get("report_type") or ""),
                "row_sha256": _row_sha256(row, fields),
                "update_flag": str(row.get("update_flag") or ""),
            }
            for row in survivors
        ]
        candidates.sort(key=_canonical_bytes)
        conflicts.append(
            {
                "business_key": list(key),
                "candidate_count": len(candidates),
                "candidate_projection_count": len(projections),
                "candidates": candidates,
                "table": table,
            }
        )
    if not conflicts:
        _fail("HISTORICAL_TAINT_CONFLICT_NOT_REPRODUCED")
    return conflicts


def _parent_period_rows(
    path: Path,
    *,
    symbol: str,
    end_date: str,
    availability: str,
) -> list[dict[str, Any]]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [column for column in _PERIOD_COLUMNS if column in available]
    required = {"ts_code", "end_date", "availability_date"}
    if not required.issubset(columns):
        _fail("HISTORICAL_TAINT_PARENT_PERIOD_SCHEMA_INVALID")
    frame = pq.read_table(
        path,
        columns=columns,
        filters=[
            ("ts_code", "=", symbol),
            ("end_date", "=", end_date),
        ],
    ).to_pandas()
    return [
        row
        for row in frame.to_dict("records")
        if _date(row["availability_date"], label="parent availability")
        == availability
    ]


def _latest_parent_daily(path: Path, *, symbol: str, cutoff: str) -> dict[str, Any]:
    required = ["ts_code", "trade_date", "end_date", "availability_date"]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    if not set(required).issubset(available):
        _fail("HISTORICAL_TAINT_PARENT_DAILY_SCHEMA_INVALID")
    frame = pq.read_table(
        path,
        columns=required,
        filters=[("ts_code", "=", symbol)],
    ).to_pandas()
    rows = [
        row
        for row in frame.to_dict("records")
        if _date(row["trade_date"], label="parent trade_date") <= cutoff
    ]
    if not rows:
        _fail("HISTORICAL_TAINT_PARENT_DAILY_SUBJECT_MISSING", symbol)
    return max(
        rows,
        key=lambda row: _date(row["trade_date"], label="parent trade_date"),
    )


def _parent_table_binding(
    predecessor: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Path, str]:
    table_sha = str(dict(predecessor.get("table_sha256", {}) or {}).get(name) or "")
    refs = dict(predecessor.get("immutable_refs", {}) or {})
    matches: list[tuple[Path, str]] = []
    for value in refs.values():
        reference = dict(value) if isinstance(value, Mapping) else {}
        path = Path(str(reference.get("path") or ""))
        digest = str(reference.get("sha256") or "").lower()
        if path.name == f"{name}.parquet" and digest == table_sha:
            matches.append((path, digest))
    if len(matches) != 1 or _SHA256_RE.fullmatch(table_sha) is None:
        _fail("HISTORICAL_TAINT_PARENT_TABLE_REF_INVALID", name)
    path, digest = matches[0]
    resolved = path.expanduser().resolve(strict=True)
    if _sha256_file(resolved) != digest:
        _fail("HISTORICAL_TAINT_PARENT_TABLE_TAMPER", name)
    return resolved, digest


def _build_registry(
    *,
    failure_evidence: Sequence[Mapping[str, Any]],
    predecessor: Mapping[str, Any],
    parent_cutoff: str,
    target_cutoff: str,
    delta_fileset_root: str | Path,
    evidence_layout: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    parent = _date(parent_cutoff, label="parent cutoff")
    target = _date(target_cutoff, label="target cutoff")
    if target <= parent:
        _fail("HISTORICAL_TAINT_WINDOW_INVALID")
    delta_manifest = validate_successor_support_fileset(delta_fileset_root)
    plan = dict(dict(delta_manifest["binding"])["plan"])
    if plan.get("support_start") != _next_date(parent) or plan.get("target_date") != target:
        _fail("HISTORICAL_TAINT_DELTA_WINDOW_MISMATCH")
    period_path, period_sha = _parent_table_binding(
        predecessor,
        name="fundamental_period",
    )
    daily_path, daily_sha = _parent_table_binding(
        predecessor,
        name="fundamental_daily",
    )
    delta_tables = open_support_tables(delta_fileset_root)
    delta_keys: set[tuple[str, str]] = set()
    delta_rows = 0
    declared_financial_support = {
        (str(value["table"]), str(value["ts_code"]), str(value["end_date"]))
        for value in list(plan.get("financial_support_dependencies", []) or [])
    }
    empty_financial_support = {
        (
            str(request.get("table") or ""),
            str(dict(request.get("params", {}) or {}).get("ts_code") or ""),
            str(dict(request.get("params", {}) or {}).get("period") or ""),
        )
        for request, receipt in zip(
            list(plan.get("requests", []) or []),
            list(delta_manifest.get("request_receipts", []) or []),
            strict=True,
        )
        if request.get("partition_type") == "EXACT_SYMBOL_REPORT_PERIOD_SUPPORT"
        and receipt.get("status") == "EMPTY"
        and receipt.get("accepted_count") == 0
    }
    observed_financial_support: set[tuple[str, str, str]] = set()
    for table in _FINANCIAL_TABLES:
        for row in delta_tables.iter_rows(table):  # type: ignore[attr-defined]
            availability = _date(
                row.get("availability_date"),
                label="delta availability",
            )
            if availability <= parent:
                support_key = (
                    table,
                    str(row["ts_code"]),
                    _date(row["end_date"], label="financial support end"),
                )
                if support_key not in declared_financial_support:
                    _fail("HISTORICAL_TAINT_DELTA_ROW_OUT_OF_WINDOW")
                observed_financial_support.add(support_key)
                continue
            if availability > target:
                _fail("HISTORICAL_TAINT_DELTA_ROW_OUT_OF_WINDOW")
            delta_rows += 1
            delta_keys.add((str(row["ts_code"]), _date(row["end_date"], label="delta end")))
    if (
        observed_financial_support.union(empty_financial_support)
        != declared_financial_support
    ):
        _fail("HISTORICAL_TAINT_FINANCIAL_SUPPORT_MISMATCH")
    entries: list[dict[str, Any]] = []
    for evidence, layout in zip(failure_evidence, evidence_layout, strict=True):
        failure_root = str(evidence.get("failure_root") or "")
        ordinal = evidence.get("ordinal")
        if type(ordinal) is not int:
            _fail("HISTORICAL_TAINT_FAILURE_ORDINAL_INVALID")
        _root, binding_path, failure_path, raw_path, failure = _failure_layout(
            failure_root,
            ordinal=ordinal,
        )
        raw_bytes = raw_path.read_bytes()
        conflicts = _material_conflicts(failure=failure, raw_bytes=raw_bytes)
        expected_refs = {
            "binding_path": binding_path,
            "failure_path": failure_path,
            "raw_path": raw_path,
        }
        for field, source_path in expected_refs.items():
            relative = str(layout.get(field) or "")
            if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                _fail("HISTORICAL_TAINT_EVIDENCE_LAYOUT_INVALID")
            digest_field = field.replace("_path", "_sha256")
            if str(layout.get(digest_field) or "").lower() != _sha256_file(source_path):
                _fail("HISTORICAL_TAINT_EVIDENCE_LAYOUT_INVALID")
        for conflict in conflicts:
            symbol, end_date, availability = conflict["business_key"]
            if availability > parent:
                _fail("HISTORICAL_TAINT_CURRENT_WINDOW_CONFLICT", symbol)
            if (symbol, end_date) in delta_keys:
                _fail("HISTORICAL_TAINT_SAME_PERIOD_DELTA", f"{symbol}|{end_date}")
            parent_rows = _parent_period_rows(
                period_path,
                symbol=symbol,
                end_date=end_date,
                availability=availability,
            )
            if len(parent_rows) != 1:
                _fail("HISTORICAL_TAINT_PARENT_PERIOD_NOT_EXACT", f"{symbol}|{end_date}")
            latest = _latest_parent_daily(daily_path, symbol=symbol, cutoff=parent)
            latest_end = _date(latest["end_date"], label="latest parent end_date")
            latest_availability = _date(
                latest["availability_date"],
                label="latest parent availability",
            )
            if (latest_availability, latest_end) <= (availability, end_date):
                _fail("HISTORICAL_TAINT_REACHES_PARENT_WINNER", symbol)
            entries.append(
                {
                    **conflict,
                    "classification": HISTORICAL_TAINT_CLASSIFICATION,
                    "failure_evidence": dict(layout),
                    "failure_evidence_sha256": str(failure["failure_sha256"]),
                    "parent_period_row_sha256": _row_sha256(
                        parent_rows[0],
                        [field for field in _PERIOD_COLUMNS if field in parent_rows[0]],
                    ),
                    "parent_current_winner": {
                        "availability_date": latest_availability,
                        "end_date": latest_end,
                        "trade_date": _date(
                            latest["trade_date"],
                            label="latest parent trade_date",
                        ),
                    },
                    "same_period_delta_row_count": 0,
                }
            )
    if not entries:
        _fail("HISTORICAL_TAINT_REGISTRY_EMPTY")
    entries.sort(key=_canonical_bytes)
    poisoned = sorted(
        {f"{entry['business_key'][0]}|{entry['business_key'][1]}" for entry in entries}
    )
    body: dict[str, Any] = {
        "schema_version": HISTORICAL_TAINT_REGISTRY_SCHEMA,
        "status": HISTORICAL_TAINT_STATUS,
        "classification": HISTORICAL_TAINT_CLASSIFICATION,
        "predecessor_generation_id": str(predecessor.get("generation_id") or ""),
        "predecessor_reference_sha256": str(predecessor.get("reference_sha256") or ""),
        "parent_cutoff": parent,
        "target_cutoff": target,
        "delta_support_start": _next_date(parent),
        "delta_source_manifest_sha256": str(delta_manifest["manifest_sha256"]),
        "delta_financial_row_count": delta_rows,
        "bounded_financial_support_key_count": len(declared_financial_support),
        "bounded_financial_support_empty_count": len(empty_financial_support),
        "historical_conflict_count": len(entries),
        "poisoned_key_count": len(poisoned),
        "poisoned_keyset": poisoned,
        "poisoned_keyset_sha256": canonical_json_sha256(poisoned),
        "parent_table_sha256": {
            "fundamental_daily": daily_sha,
            "fundamental_period": period_sha,
        },
        "current_window_material_conflict_count": 0,
        "same_period_delta_row_count": 0,
        "winner_selection_applied": False,
        "entries": entries,
    }
    body["registry_sha256"] = canonical_json_sha256(body)
    return body


def build_historical_taint_registry(
    *,
    failure_evidence: Sequence[Mapping[str, Any]],
    predecessor: Mapping[str, Any],
    parent_cutoff: str,
    target_cutoff: str,
    delta_fileset_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Build a sealed registry and the exact files that must be preserved."""

    if not failure_evidence:
        _fail("HISTORICAL_TAINT_EVIDENCE_REQUIRED")
    layouts: list[dict[str, str]] = []
    evidence_files: dict[str, Path] = {}
    for index, evidence in enumerate(failure_evidence):
        ordinal = evidence.get("ordinal")
        if type(ordinal) is not int:
            _fail("HISTORICAL_TAINT_FAILURE_ORDINAL_INVALID")
        _root, binding, failure_path, raw_path, _failure = _failure_layout(
            str(evidence.get("failure_root") or ""),
            ordinal=ordinal,
        )
        prefix = f"historical_taint/evidence_{index:03d}"
        source_name = f"{prefix}/legacy_capture/binding.json"
        failure_name = (
            f"{prefix}/legacy_capture-failures/{ordinal:06d}.failure.json"
        )
        raw_name = f"{prefix}/legacy_capture-failures/{ordinal:06d}.raw.json"
        evidence_files[source_name] = binding
        evidence_files[failure_name] = failure_path
        evidence_files[raw_name] = raw_path
        layouts.append(
            {
                "binding_path": source_name,
                "binding_sha256": _sha256_file(binding),
                "failure_path": failure_name,
                "failure_sha256": _sha256_file(failure_path),
                "raw_path": raw_name,
                "raw_sha256": _sha256_file(raw_path),
            }
        )
    registry = _build_registry(
        failure_evidence=failure_evidence,
        predecessor=predecessor,
        parent_cutoff=parent_cutoff,
        target_cutoff=target_cutoff,
        delta_fileset_root=delta_fileset_root,
        evidence_layout=layouts,
    )
    return registry, evidence_files


def validate_historical_taint_registry(
    registry_path: str | Path,
    *,
    evidence_root: str | Path,
    predecessor: Mapping[str, Any],
    delta_fileset_root: str | Path,
) -> dict[str, Any]:
    """Rebuild a staged registry from sealed evidence and require exact bytes."""

    path = Path(registry_path).expanduser().resolve(strict=True)
    root = Path(evidence_root).expanduser().resolve(strict=True)
    registry, registry_bytes = _read_json(path)
    claimed = str(registry.get("registry_sha256") or "")
    body = dict(registry)
    body.pop("registry_sha256", None)
    if (
        registry.get("schema_version") != HISTORICAL_TAINT_REGISTRY_SCHEMA
        or registry.get("status") != HISTORICAL_TAINT_STATUS
        or registry.get("classification") != HISTORICAL_TAINT_CLASSIFICATION
        or _SHA256_RE.fullmatch(claimed) is None
        or canonical_json_sha256(body) != claimed
        or registry_bytes != _canonical_bytes(registry)
    ):
        _fail("HISTORICAL_TAINT_REGISTRY_INVALID")
    failure_evidence: list[dict[str, Any]] = []
    layouts: list[dict[str, str]] = []
    seen_failures: set[str] = set()
    for entry in list(registry.get("entries", []) or []):
        layout = dict(dict(entry).get("failure_evidence", {}) or {})
        failure_relative = Path(str(layout.get("failure_path") or ""))
        if failure_relative.is_absolute() or ".." in failure_relative.parts:
            _fail("HISTORICAL_TAINT_EVIDENCE_LAYOUT_INVALID")
        failure_root = (root / failure_relative).parent
        ordinal_text = failure_relative.name.removesuffix(".failure.json")
        if not ordinal_text.isdigit():
            _fail("HISTORICAL_TAINT_EVIDENCE_LAYOUT_INVALID")
        failure_identity = str(failure_relative)
        if failure_identity in seen_failures:
            continue
        seen_failures.add(failure_identity)
        layouts.append(layout)
        failure_evidence.append(
            {"failure_root": str(failure_root), "ordinal": int(ordinal_text)}
        )
    rebuilt = _build_registry(
        failure_evidence=failure_evidence,
        predecessor=predecessor,
        parent_cutoff=str(registry.get("parent_cutoff") or ""),
        target_cutoff=str(registry.get("target_cutoff") or ""),
        delta_fileset_root=delta_fileset_root,
        evidence_layout=layouts,
    )
    if _canonical_bytes(rebuilt) != registry_bytes:
        _fail("HISTORICAL_TAINT_REGISTRY_REPLAY_MISMATCH")
    return rebuilt


__all__ = [
    "HISTORICAL_TAINT_CLASSIFICATION",
    "HISTORICAL_TAINT_REGISTRY_SCHEMA",
    "HISTORICAL_TAINT_STATUS",
    "HistoricalTaintError",
    "build_historical_taint_registry",
    "validate_historical_taint_registry",
]
