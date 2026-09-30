"""Resolve strict Dashboard close evidence against the canonical Market table.

Strict close evidence names a frozen Market snapshot by manifest path and SHA.
The per-symbol ``serving/`` projection that v1 evidence cited is a derived copy
of ``table/`` and is not retained, so v1 verification reads the snapshot's
canonical ``table/bars`` partition that holds the valuation date instead. v2
evidence carries only ``{symbol, trade_date, close}`` rows plus an exact
``table_partition_ref``. Both require every evidenced close to match exactly
one table row.

Path resolution is lexical: no directory listing, so the same candidates work
under a retained-bytes replay context.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

CLOSE_TOLERANCE = 1e-9
STRICT_CLOSE_EVIDENCE_V1 = "cn_dashboard_strict_market_close_evidence.v1"
STRICT_CLOSE_EVIDENCE_V2 = "cn_dashboard_strict_market_close_evidence.v2"
STRICT_CLOSE_EVIDENCE_SCHEMAS = (STRICT_CLOSE_EVIDENCE_V1, STRICT_CLOSE_EVIDENCE_V2)
_V2_STOCK_KEYS = {"symbol", "trade_date", "close"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StrictCloseTableSourceError(ValueError):
    pass


def _snapshot_id(evidence: Mapping[str, Any]) -> str:
    snapshot_id = evidence.get("snapshot_id")
    if (
        type(snapshot_id) is not str
        or not snapshot_id
        or snapshot_id in {".", ".."}
        or PurePosixPath(snapshot_id).name != snapshot_id
        or "\\" in snapshot_id
    ):
        raise StrictCloseTableSourceError("STRICT_CLOSE_SNAPSHOT_ID_INVALID")
    return snapshot_id


def _compact_date(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) < 8:
        raise StrictCloseTableSourceError("STRICT_CLOSE_TRADE_DATE_INVALID")
    return digits[:8]


def snapshot_manifest_relative(evidence: Mapping[str, Any]) -> str:
    """Return the project-relative ``.../_snapshots/<id>.json`` manifest path."""

    snapshot_id = _snapshot_id(evidence)
    declared = evidence.get("snapshot_manifest_path")
    if type(declared) is not str or "\\" in declared or "\x00" in declared:
        raise StrictCloseTableSourceError("STRICT_CLOSE_MANIFEST_PATH_INVALID")
    path = PurePosixPath(declared)
    if (
        path.is_absolute()
        or path.as_posix() != declared
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.name != f"{snapshot_id}.json"
        or path.parent.name != "_snapshots"
    ):
        raise StrictCloseTableSourceError("STRICT_CLOSE_MANIFEST_PATH_INVALID")
    return declared


def table_partition_candidates(
    project_root: Path,
    evidence: Mapping[str, Any],
    manifest_raw: bytes,
) -> list[Path]:
    """Validate the manifest binding and return ordered table-file candidates.

    Hive month partitions (``year=YYYY/month=MM/part.parquet``) come first; a
    flat ``table/bars/part.parquet`` is accepted for unpartitioned snapshots.
    """

    snapshot_id = _snapshot_id(evidence)
    manifest_relative = PurePosixPath(snapshot_manifest_relative(evidence))
    try:
        manifest = json.loads(manifest_raw)
    except (TypeError, ValueError) as exc:
        raise StrictCloseTableSourceError("STRICT_CLOSE_MANIFEST_INVALID") from exc
    if not isinstance(manifest, dict) or manifest.get("snapshot_id") != snapshot_id:
        raise StrictCloseTableSourceError("STRICT_CLOSE_MANIFEST_SNAPSHOT_MISMATCH")
    declared = manifest.get("table_root")
    expected_tail = ("_snapshots", snapshot_id, "table", "bars")
    if type(declared) is not str or Path(declared).parts[-4:] != expected_tail:
        raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_ROOT_INVALID")
    trade_date = _compact_date(evidence.get("trade_date"))
    table_root = project_root / manifest_relative.parent / snapshot_id / "table" / "bars"
    return [
        table_root / f"year={trade_date[:4]}" / f"month={trade_date[4:6]}" / "part.parquet",
        table_root / "part.parquet",
    ]


def declared_table_partition(
    project_root: Path,
    evidence: Mapping[str, Any],
    manifest_raw: bytes,
) -> tuple[list[Path], str | None]:
    """Return the table-file candidates evidence may bind and its pinned SHA.

    v1 pins no table bytes, so every lexical candidate is allowed and the SHA is
    ``None``. v2 must name exactly one candidate with its SHA-256.
    """

    candidates = table_partition_candidates(project_root, evidence, manifest_raw)
    schema = evidence.get("schema_version")
    if schema == STRICT_CLOSE_EVIDENCE_V1:
        return candidates, None
    if schema != STRICT_CLOSE_EVIDENCE_V2:
        raise StrictCloseTableSourceError("STRICT_CLOSE_EVIDENCE_SCHEMA_INVALID")
    for row in evidence.get("stocks") or []:
        if not isinstance(row, Mapping) or set(row) != _V2_STOCK_KEYS:
            raise StrictCloseTableSourceError("STRICT_CLOSE_V2_STOCK_ROW_INVALID")
    ref = evidence.get("table_partition_ref")
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_PARTITION_REF_INVALID")
    sha = ref["sha256"]
    allowed = {path.relative_to(project_root).as_posix(): path for path in candidates}
    if ref["path"] not in allowed or type(sha) is not str or not _SHA256.fullmatch(sha):
        raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_PARTITION_REF_INVALID")
    return [allowed[ref["path"]]], sha


def build_table_close_rows(
    project_root: Path,
    *,
    snapshot_id: str,
    snapshot_manifest_path: str,
    manifest_raw: bytes,
    trade_date: Any,
    symbols: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read held-symbol closes from the snapshot table for v2 evidence."""

    compact = _compact_date(trade_date)
    binding = {
        "snapshot_id": snapshot_id,
        "snapshot_manifest_path": snapshot_manifest_path,
        "trade_date": compact,
    }
    for candidate in table_partition_candidates(project_root, binding, manifest_raw):
        if candidate.is_file() and not candidate.is_symlink():
            raw = candidate.read_bytes()
            break
    else:
        raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_PARTITION_MISSING:" + compact)
    closes = closes_from_table_partition(raw, symbols=symbols, trade_date=compact)
    rows = [
        {"symbol": symbol, "trade_date": compact, "close": closes[symbol]} for symbol in symbols
    ]
    ref = {
        "path": candidate.relative_to(project_root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return rows, ref


def closes_from_table_partition(
    raw: bytes,
    *,
    symbols: Iterable[str],
    trade_date: Any,
) -> dict[str, float]:
    """Return the single table close per symbol for ``trade_date``."""

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = sorted({str(symbol) for symbol in symbols})
    compact = _compact_date(trade_date)
    try:
        table = pq.read_table(io.BytesIO(raw), columns=["ts_code", "trade_date", "close"])
    except Exception as exc:  # pyarrow raises several unrelated types
        raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_UNREADABLE") from exc
    table = table.filter(pc.is_in(table["ts_code"].cast("string"), value_set=_string_array(wanted)))
    closes: dict[str, float] = {}
    for symbol, day, close in zip(
        table["ts_code"].to_pylist(),
        table["trade_date"].to_pylist(),
        table["close"].to_pylist(),
    ):
        if _compact_date(day) != compact:
            continue
        if symbol in closes:
            raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_ROW_DUPLICATE:" + symbol)
        if close is None or not math.isfinite(float(close)) or float(close) <= 0:
            raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_CLOSE_INVALID:" + symbol)
        closes[symbol] = float(close)
    missing = [symbol for symbol in wanted if symbol not in closes]
    if missing:
        raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_ROW_MISSING:" + ",".join(missing))
    return closes


def verify_evidence_closes(
    evidence: Mapping[str, Any],
    table_closes: Mapping[str, float],
) -> None:
    """Require each evidenced stock close to equal its canonical table close."""

    for row in evidence.get("stocks") or []:
        symbol = str(row.get("symbol") or row.get("ts_code"))
        try:
            recorded = float(row.get("close"))
        except (TypeError, ValueError) as exc:
            raise StrictCloseTableSourceError(
                "STRICT_CLOSE_EVIDENCE_CLOSE_INVALID:" + symbol
            ) from exc
        if symbol not in table_closes or abs(recorded - table_closes[symbol]) > CLOSE_TOLERANCE:
            raise StrictCloseTableSourceError("STRICT_CLOSE_TABLE_VALUE_MISMATCH:" + symbol)


def _string_array(values: list[str]):
    import pyarrow as pa

    return pa.array(values, type=pa.string())
