"""Immutable CN benchmark-close generations used by official portfolio close.

This store is deliberately separate from the Strategy Record Store: it is a
Market input authority, not a portfolio/performance authority.  The mutable
``_latest.json`` pointer is advanced by one exact CAS after an immutable
manifest and Parquet series have been written and read back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

BENCHMARK_POINTER_SCHEMA: Final = "myquant.cn_benchmark_pointer.v1"
BENCHMARK_MANIFEST_SCHEMA: Final = "myquant.cn_benchmark_generation.v1"
BENCHMARK_SERIES_SCHEMA: Final = "myquant.cn_benchmark_series.v1"
REQUIRED_CODES: Final = ("000300.SH", "000688.SH", "399006.SZ")
EMPTY_POINTER_SHA256: Final = hashlib.sha256(b"").hexdigest()

_SHA = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ts_code", pa.string(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("source_system", pa.string(), nullable=False),
        pa.field("coverage", pa.string(), nullable=False),
        pa.field("value_date", pa.date32(), nullable=False),
    ],
    metadata={b"schema_id": BENCHMARK_SERIES_SCHEMA.encode("ascii")},
)


class CNBenchmarkStoreError(RuntimeError):
    """Benchmark store contract failure."""


class CNBenchmarkCASMismatch(CNBenchmarkStoreError):
    """The current benchmark pointer did not match the frozen preimage."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = _sha256(canonical_json_bytes(result))
    return result


def _validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    observed = value.get("content_sha256")
    if not isinstance(observed, str) or _SHA.fullmatch(observed) is None:
        raise CNBenchmarkStoreError(f"{label} content SHA is invalid")
    body = dict(value)
    del body["content_sha256"]
    if observed != _sha256(canonical_json_bytes(body)):
        raise CNBenchmarkStoreError(f"{label} content SHA mismatch")


def _read_regular(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise CNBenchmarkStoreError(f"{label} is not a regular file")
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise CNBenchmarkStoreError(f"{label} was unstable")
    return first


def pointer_sha256(root: Path) -> str:
    path = root / "_latest.json"
    return (
        _sha256(_read_regular(path, label="benchmark pointer"))
        if path.exists()
        else EMPTY_POINTER_SHA256
    )


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        day = date.fromisoformat(str(raw.get("date") or raw.get("trade_date")))
        code = str(raw.get("ts_code") or "")
        if code not in REQUIRED_CODES:
            raise CNBenchmarkStoreError("benchmark code is outside the governed set")
        key = (day.isoformat(), code)
        if key in seen:
            raise CNBenchmarkStoreError("benchmark date/code is duplicated")
        seen.add(key)
        try:
            close = float(raw.get("close"))
        except (TypeError, ValueError) as exc:
            raise CNBenchmarkStoreError("benchmark close is invalid") from exc
        if not close > 0:
            raise CNBenchmarkStoreError("benchmark close is not positive")
        coverage = str(raw.get("coverage") or "exact_close")
        value_date = date.fromisoformat(str(raw.get("value_date") or day.isoformat()))
        source_system = str(raw.get("source_system") or "")
        if coverage != "exact_close" or value_date != day or not source_system:
            raise CNBenchmarkStoreError("benchmark row is not an exact same-date close")
        normalized.append(
            {
                "date": day,
                "ts_code": code,
                "close": close,
                "source_system": source_system,
                "coverage": coverage,
                "value_date": value_date,
            }
        )
    normalized.sort(key=lambda row: (row["date"], row["ts_code"]))
    by_day: dict[date, set[str]] = {}
    for row in normalized:
        by_day.setdefault(row["date"], set()).add(row["ts_code"])
    if not normalized or any(codes != set(REQUIRED_CODES) for codes in by_day.values()):
        raise CNBenchmarkStoreError("benchmark generation lacks a complete three-index day")
    return normalized


def _write_exact_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular(path, label="benchmark immutable artifact") != raw:
            raise CNBenchmarkStoreError("benchmark immutable identity collision")
        return
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        if _read_regular(path, label="benchmark immutable artifact") != raw:
            raise CNBenchmarkStoreError("benchmark immutable identity collision") from None
    finally:
        temporary.unlink(missing_ok=True)


def _series_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    table = pa.Table.from_pydict(
        {field.name: [row[field.name] for row in rows] for field in _SCHEMA},
        schema=_SCHEMA,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        version="2.6",
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="2.0",
        row_group_size=max(1, len(rows)),
        store_schema=True,
    )
    return sink.getvalue().to_pybytes()


def publish_generation(
    root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    generation_id: str,
    captured_at: str,
    expected_pointer_sha256: str,
    acquisition_receipt_ref: Mapping[str, Any],
) -> dict[str, Any]:
    root = root.resolve()
    if _GENERATION.fullmatch(generation_id) is None:
        raise CNBenchmarkStoreError("benchmark generation ID is invalid")
    if (
        not isinstance(expected_pointer_sha256, str)
        or _SHA.fullmatch(expected_pointer_sha256) is None
    ):
        raise CNBenchmarkStoreError("expected benchmark pointer SHA is invalid")
    try:
        captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CNBenchmarkStoreError("benchmark captured_at is invalid") from exc
    if captured.tzinfo is None:
        raise CNBenchmarkStoreError("benchmark captured_at timezone is missing")
    normalized = _normalize_rows(rows)
    series_raw = _series_bytes(normalized)
    series_sha = _sha256(series_raw)
    relative_prefix = f"_generations/{generation_id}"
    series_relative = f"{relative_prefix}/series.parquet"
    manifest_relative = f"{relative_prefix}/manifest.v1.json"
    days = sorted({row["date"].isoformat() for row in normalized})
    receipt_ref = dict(acquisition_receipt_ref)
    if (
        set(receipt_ref) != {"path", "sha256"}
        or _SHA.fullmatch(str(receipt_ref.get("sha256"))) is None
    ):
        raise CNBenchmarkStoreError("benchmark acquisition receipt ref is invalid")
    manifest = _seal(
        {
            "schema_id": BENCHMARK_MANIFEST_SCHEMA,
            "generation_id": generation_id,
            "captured_at": captured.astimezone(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "codes": list(REQUIRED_CODES),
            "start_date": days[0],
            "end_date": days[-1],
            "trade_dates": days,
            "row_count": len(normalized),
            "series": {"path": series_relative, "sha256": series_sha, "bytes": len(series_raw)},
            "acquisition_receipt_ref": receipt_ref,
            "provider_authority": "MARKET_INPUT_ONLY",
            "portfolio_authority": False,
            "broker_order_trade_authority": False,
        }
    )
    manifest_raw = canonical_json_bytes(manifest)
    _write_exact_once(root / series_relative, series_raw)
    _write_exact_once(root / manifest_relative, manifest_raw)
    pointer = _seal(
        {
            "schema_id": BENCHMARK_POINTER_SCHEMA,
            "generation_id": generation_id,
            "manifest": {"path": manifest_relative, "sha256": _sha256(manifest_raw)},
            "series": manifest["series"],
            "start_date": days[0],
            "end_date": days[-1],
            "trade_dates": days,
            "previous_pointer_sha256": (
                None if expected_pointer_sha256 == EMPTY_POINTER_SHA256 else expected_pointer_sha256
            ),
            "portfolio_authority": False,
            "broker_order_trade_authority": False,
        }
    )
    pointer_raw = canonical_json_bytes(pointer)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".latest.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        observed = pointer_sha256(root)
        if observed != expected_pointer_sha256:
            raise CNBenchmarkCASMismatch(
                f"benchmark pointer CAS mismatch: expected {expected_pointer_sha256}, observed {observed}"
            )
        temporary = root / f"._latest.tmp-{os.getpid()}-{secrets.token_hex(4)}"
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        try:
            os.write(fd, pointer_raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, root / "_latest.json")
    finally:
        os.close(descriptor)
    loaded = load_generation(root)
    if loaded["pointer"] != pointer:
        raise CNBenchmarkStoreError("benchmark pointer readback mismatch")
    return {**loaded, "pointer_sha256": _sha256(pointer_raw)}


def load_generation(root: Path) -> dict[str, Any]:
    pointer_raw = _read_regular(root / "_latest.json", label="benchmark pointer")
    try:
        pointer = json.loads(pointer_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CNBenchmarkStoreError("benchmark pointer is invalid JSON") from exc
    _validate_seal(pointer, label="benchmark pointer")
    if pointer.get("schema_id") != BENCHMARK_POINTER_SCHEMA:
        raise CNBenchmarkStoreError("benchmark pointer schema mismatch")
    manifest_ref = pointer.get("manifest")
    if not isinstance(manifest_ref, dict):
        raise CNBenchmarkStoreError("benchmark manifest ref is absent")
    manifest_raw = _read_regular(root / str(manifest_ref.get("path")), label="benchmark manifest")
    if _sha256(manifest_raw) != manifest_ref.get("sha256"):
        raise CNBenchmarkStoreError("benchmark manifest SHA mismatch")
    manifest = json.loads(manifest_raw)
    _validate_seal(manifest, label="benchmark manifest")
    if manifest.get("schema_id") != BENCHMARK_MANIFEST_SCHEMA or manifest.get(
        "generation_id"
    ) != pointer.get("generation_id"):
        raise CNBenchmarkStoreError("benchmark manifest closure mismatch")
    series_ref = pointer.get("series")
    if not isinstance(series_ref, dict) or series_ref != manifest.get("series"):
        raise CNBenchmarkStoreError("benchmark series ref mismatch")
    series_raw = _read_regular(root / str(series_ref.get("path")), label="benchmark series")
    if _sha256(series_raw) != series_ref.get("sha256") or len(series_raw) != series_ref.get(
        "bytes"
    ):
        raise CNBenchmarkStoreError("benchmark series closure mismatch")
    table = pq.read_table(pa.BufferReader(series_raw))
    if table.schema != _SCHEMA:
        raise CNBenchmarkStoreError("benchmark series schema mismatch")
    rows = table.to_pylist()
    normalized = _normalize_rows(rows)
    if normalized != rows:
        raise CNBenchmarkStoreError("benchmark series order/values are non-canonical")
    return {
        "pointer": pointer,
        "manifest": manifest,
        "rows": rows,
        "pointer_sha256": _sha256(pointer_raw),
        "manifest_sha256": _sha256(manifest_raw),
        "series_sha256": _sha256(series_raw),
    }


__all__ = [
    "BENCHMARK_MANIFEST_SCHEMA",
    "BENCHMARK_POINTER_SCHEMA",
    "BENCHMARK_SERIES_SCHEMA",
    "CNBenchmarkCASMismatch",
    "CNBenchmarkStoreError",
    "EMPTY_POINTER_SHA256",
    "REQUIRED_CODES",
    "load_generation",
    "pointer_sha256",
    "publish_generation",
]
