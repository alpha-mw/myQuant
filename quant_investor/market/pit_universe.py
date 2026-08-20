"""Point-in-time CN listing membership helpers.

The module is offline by default: pure status evaluation and local store
read/write do not call a provider. Online refresh is exposed as an explicit
function so scripts can gate it behind the approved maintenance window.
"""

from __future__ import annotations

import hashlib
import fcntl
import io
import json
import os
import re
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd

PIT_UNIVERSE_SCHEMA_VERSION = "cn_pit_universe.v1"
PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION = "cn_pit_universe_manifest.v1"
PIT_UNIVERSE_DISCOVERY_SCHEMA_VERSION = "cn_pit_universe_latest.v1"
PIT_UNIVERSE_REFRESH_SCHEMA_VERSION = "cn_pit_universe_refresh.v1"
PIT_UNIVERSE_LINEAGE_SCHEMA_VERSION = "cn_pit_universe_lineage.v1"
PIT_UNIVERSE_CAPTURE_SCHEMA_VERSION = "cn_pit_universe_capture.v2"
PIT_UNIVERSE_CAPTURE_PARTITION_SCHEMA_VERSION = "cn_pit_universe_capture_partition.v1"
PIT_UNIVERSE_CAPTURE_VALIDATION_SCHEMA_VERSION = "cn_pit_universe_capture_validation.v1"
PIT_UNIVERSE_EXTERNAL_EXCLUSION_SCHEMA_VERSION = "cn_pit_external_exclusions.v1"
PIT_UNIVERSE_SHADOW_CANDIDATE_SCHEMA_VERSION = "cn_pit_universe_shadow_candidate.v1"
PIT_UNIVERSE_EMPTY_PARENT_POINTER = "EMPTY"
PIT_UNIVERSE_MAX_LINEAGE_DEPTH = 64

PIT_UNIVERSE_GENERATIONS_DIRNAME = "_generations"
PIT_UNIVERSE_GENERATION_MANIFEST_FILENAME = "manifest.json"
PIT_UNIVERSE_GENERATION_CANONICAL_FILENAME = "stock_basic_membership.parquet"

LIST_STATUS_LISTED = "L"
LIST_STATUS_DELISTED = "D"
LIST_STATUS_PENDING = "P"
SUPPORTED_LIST_STATUSES = (
    LIST_STATUS_LISTED,
    LIST_STATUS_DELISTED,
    LIST_STATUS_PENDING,
)

STOCK_BASIC_FIELDS = (
    "ts_code",
    "name",
    "area",
    "industry",
    "market",
    "list_date",
    "delist_date",
    "list_status",
)
_CANONICAL_A_SYMBOL = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$")
_PROVIDER_EXTERNAL_LEGACY_DELISTED = re.compile(r"^T[0-9]{6}\.SH$")
MAX_EXTERNAL_LEGACY_DELISTED_EXCLUSIONS = 8
PIT_EXTERNAL_EXCLUSION_FILENAME = "excluded_provider_observations.json"

REASON_LISTED = "listed"
REASON_PRE_LISTING = "pre_listing"
REASON_PENDING = "pending"
REASON_DELISTED = "delisted"
REASON_MISSING_PIT_RECORD = "missing_pit_record"
REASON_CONFLICTING_STATUS_ROWS = "conflicting_status_rows"
REASON_MISSING_LIST_DATE = "missing_list_date"
REASON_MISSING_DELIST_DATE = "missing_delist_date"
REASON_OUTSIDE_FROZEN_SCOPE_PENDING = "outside_frozen_scope_pending"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _short_hash(parts: Sequence[Any], length: int = 12) -> str:
    payload = json.dumps(_json_safe(list(parts)), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_yyyymmdd(value: Any) -> str | None:
    if type(value) is not str or len(value) != 8 or not value.isdigit():
        return None
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None
    return value if parsed.strftime("%Y%m%d") == value else None


def _external_legacy_delisted_row(
    row: Mapping[str, Any], *, partition_status: str, effective_date: str
) -> bool:
    raw_identity = row.get("ts_code")
    if (
        type(raw_identity) is not str
        or _PROVIDER_EXTERNAL_LEGACY_DELISTED.fullmatch(raw_identity) is None
    ):
        return False
    list_date = _canonical_yyyymmdd(row.get("list_date"))
    delist_date = _canonical_yyyymmdd(row.get("delist_date"))
    if (
        partition_status != LIST_STATUS_DELISTED
        or type(row.get("list_status")) is not str
        or row.get("list_status") != LIST_STATUS_DELISTED
        or list_date is None
        or delist_date is None
        or not (list_date <= delist_date <= effective_date)
    ):
        raise RuntimeError("pit_external_legacy_delisted_context_invalid")
    for value in row.values():
        if isinstance(value, (Mapping, list, tuple, set)):
            raise RuntimeError("pit_external_legacy_delisted_row_invalid")
        if isinstance(value, float) and not pd.notna(value):
            raise RuntimeError("pit_external_legacy_delisted_row_invalid")
    return True


def _external_exclusion_entry(
    *, row: Mapping[str, Any], partition_status: str, partition_sha256: str, ordinal: int
) -> dict[str, Any]:
    normalized = dict(row)
    return {
        "classification": "PROVIDER_EXTERNAL_LEGACY_DELISTED",
        "identity": normalized["ts_code"],
        "partition_status": partition_status,
        "partition_sha256": partition_sha256,
        "row_ordinal": ordinal,
        "row": normalized,
        "row_sha256": _sha256_bytes(_json_bytes(normalized)),
        "exclusion_reason": "NONCANONICAL_PROVIDER_LEGACY_DELISTED_IDENTITY",
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: Any, *, blocker: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError(blocker)
    return digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _stable_read_bytes(path: Path, *, blocker: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError) as exc:
        raise RuntimeError(blocker) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(blocker)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if signature_before != signature_after:
            raise RuntimeError(blocker)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_json_mapping_bytes(payload: bytes, *, blocker: str) -> dict[str, Any]:
    try:
        loaded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(blocker) from exc
    if not isinstance(loaded, Mapping):
        raise RuntimeError(blocker)
    return dict(loaded)


def _require_absolute_private_root(path: str | Path, *, blocker: str) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        raise RuntimeError(blocker)
    if root.is_symlink():
        raise RuntimeError(blocker)
    if root.exists():
        resolved = root.resolve(strict=True)
        metadata = resolved.stat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise RuntimeError(blocker)
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeError(blocker)
        return resolved
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    resolved = root.resolve(strict=True)
    metadata = resolved.stat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError(blocker)
    return resolved


def _capture_member_path(
    capture_root: Path,
    declared: Any,
    *,
    expected_name: str,
    blocker: str,
) -> Path:
    raw = str(declared or "").strip()
    candidate = Path(raw)
    if not candidate.is_absolute() or candidate.name != expected_name:
        raise RuntimeError(blocker)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(capture_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(blocker) from exc
    if resolved.parent != capture_root or resolved.is_symlink():
        raise RuntimeError(blocker)
    metadata = resolved.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise RuntimeError(blocker)
    return resolved


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def normalize_symbol(value: Any) -> str:
    return _clean_text(value).upper()


def compact_date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return ""
        return value.strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = _clean_text(value)
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _row_payload_hash(row: Mapping[str, Any]) -> str:
    return _short_hash([dict(sorted((str(k), _json_safe(v)) for k, v in row.items()))], length=16)


def _source_run_id(observed_at: str) -> str:
    return f"pit-universe-{compact_date(observed_at) or 'unknown'}-{_short_hash([observed_at])}"


@dataclass
class PITUniverseRecord:
    schema_version: str = PIT_UNIVERSE_SCHEMA_VERSION
    symbol: str = ""
    name: str = ""
    area: str = ""
    industry: str = ""
    board_market: str = ""
    source_list_status: str = ""
    list_date: str = ""
    delist_date: str = ""
    effective_from: str = ""
    effective_to: str = ""
    observed_at: str = ""
    source: str = "tushare.stock_basic"
    source_run_id: str = ""
    raw_payload_hash: str = ""
    membership_quality: str = "ok"

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or PIT_UNIVERSE_SCHEMA_VERSION)
        self.symbol = normalize_symbol(self.symbol)
        self.source_list_status = _clean_text(self.source_list_status).upper()
        self.list_date = compact_date(self.list_date)
        self.delist_date = compact_date(self.delist_date)
        self.effective_from = compact_date(self.effective_from) or self.list_date
        self.effective_to = compact_date(self.effective_to) or self.delist_date
        self.name = _clean_text(self.name)
        self.area = _clean_text(self.area)
        self.industry = _clean_text(self.industry)
        self.board_market = _clean_text(self.board_market)
        self.observed_at = _clean_text(self.observed_at)
        self.source = _clean_text(self.source) or "tushare.stock_basic"
        self.source_run_id = _clean_text(self.source_run_id)
        self.raw_payload_hash = _clean_text(self.raw_payload_hash)
        self.membership_quality = _clean_text(self.membership_quality) or "ok"

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PITUniverseRecord":
        return cls(**dict(payload))


@dataclass
class PITListingStatus:
    symbol: str = ""
    date: str = ""
    in_universe: bool = False
    research_eligible: bool = False
    tradable: bool = False
    reason: str = REASON_MISSING_PIT_RECORD
    list_date: str = ""
    delist_date: str = ""
    source_list_status: str = ""
    observed_at: str = ""
    membership_quality: str = ""
    provider_listed: bool = False
    authority_membership: bool = False

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.date = compact_date(self.date)
        self.reason = _clean_text(self.reason)
        self.list_date = compact_date(self.list_date)
        self.delist_date = compact_date(self.delist_date)
        self.source_list_status = _clean_text(self.source_list_status).upper()
        self.observed_at = _clean_text(self.observed_at)
        self.membership_quality = _clean_text(self.membership_quality)
        if self.source_list_status:
            self.provider_listed = self.source_list_status == LIST_STATUS_LISTED
            self.authority_membership = (
                self.membership_quality != REASON_OUTSIDE_FROZEN_SCOPE_PENDING
            )

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))


def record_from_stock_basic_row(
    row: Mapping[str, Any],
    *,
    list_status: str,
    observed_at: str,
    source_run_id: str,
) -> PITUniverseRecord | None:
    symbol = normalize_symbol(row.get("ts_code") or row.get("symbol"))
    if not symbol:
        return None
    source_status = _clean_text(row.get("list_status") or list_status).upper()
    raw_hash = _row_payload_hash(row)
    return PITUniverseRecord(
        symbol=symbol,
        name=row.get("name", ""),
        area=row.get("area", ""),
        industry=row.get("industry", ""),
        board_market=row.get("market", ""),
        source_list_status=source_status,
        list_date=row.get("list_date", ""),
        delist_date=row.get("delist_date", ""),
        observed_at=observed_at,
        source_run_id=source_run_id,
        raw_payload_hash=raw_hash,
    )


def records_from_stock_basic_frames(
    frames_by_status: Mapping[str, pd.DataFrame],
    *,
    observed_at: str | None = None,
    source_run_id: str | None = None,
) -> list[PITUniverseRecord]:
    resolved_observed_at = observed_at or _utc_now_iso()
    resolved_run_id = source_run_id or _source_run_id(resolved_observed_at)
    records: list[PITUniverseRecord] = []
    for list_status in SUPPORTED_LIST_STATUSES:
        frame = frames_by_status.get(list_status)
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            record = record_from_stock_basic_row(
                row,
                list_status=list_status,
                observed_at=resolved_observed_at,
                source_run_id=resolved_run_id,
            )
            if record is not None:
                records.append(record)
    return records


def _quality_for_group(records: Sequence[PITUniverseRecord]) -> str:
    list_dates = {record.list_date for record in records if record.list_date}
    delist_dates = {record.delist_date for record in records if record.delist_date}
    if len(list_dates) > 1 or len(delist_dates) > 1:
        return REASON_CONFLICTING_STATUS_ROWS
    if (
        any(record.source_list_status == LIST_STATUS_DELISTED for record in records)
        and not delist_dates
    ):
        return REASON_MISSING_DELIST_DATE
    if all(record.source_list_status == LIST_STATUS_LISTED for record in records) and any(
        record.membership_quality == REASON_OUTSIDE_FROZEN_SCOPE_PENDING for record in records
    ):
        return REASON_OUTSIDE_FROZEN_SCOPE_PENDING
    return "ok"


def _record_rank(record: PITUniverseRecord) -> tuple[int, int, str, str]:
    status_rank = {
        LIST_STATUS_DELISTED: 3,
        LIST_STATUS_PENDING: 2,
        LIST_STATUS_LISTED: 1,
    }.get(record.source_list_status, 0)
    has_delist_date = 1 if record.delist_date else 0
    return (has_delist_date, status_rank, record.delist_date, record.observed_at)


def dedupe_latest_records(records: Sequence[PITUniverseRecord]) -> list[PITUniverseRecord]:
    by_symbol: dict[str, list[PITUniverseRecord]] = {}
    for record in records:
        if record.symbol:
            by_symbol.setdefault(record.symbol, []).append(record)

    latest: list[PITUniverseRecord] = []
    for symbol in sorted(by_symbol):
        group = by_symbol[symbol]
        chosen = PITUniverseRecord.from_dict(max(group, key=_record_rank).to_dict())
        chosen.membership_quality = _quality_for_group(group)
        latest.append(chosen)
    return latest


def carry_forward_historical_records(
    fresh_records: Sequence[PITUniverseRecord],
    parent_records: Sequence[PITUniverseRecord],
    *,
    required_symbols: Iterable[str],
    observed_at: str,
) -> tuple[list[PITUniverseRecord], list[str]]:
    """Prefix-extend a refresh with complete historical delisted parent rows.

    Fresh provider rows are always authoritative.  A missing parent row is
    eligible only when it is a closed, internally consistent historical
    delisting outside the required current component scope.  Any other shrink
    remains a hard blocker.
    """

    fresh = records_by_symbol(fresh_records)
    parent = records_by_symbol(parent_records)
    required = {normalize_symbol(symbol) for symbol in required_symbols if normalize_symbol(symbol)}
    cutoff_date = compact_date(observed_at)
    if not cutoff_date:
        raise RuntimeError("pit_refresh_observed_at_invalid")
    carried: list[str] = []
    rejected: list[str] = []
    for symbol in sorted(set(parent) - set(fresh)):
        record = parent[symbol]
        eligible = (
            symbol not in required
            and record.source_list_status == LIST_STATUS_DELISTED
            and record.membership_quality == "ok"
            and bool(record.list_date)
            and bool(record.delist_date)
            and record.effective_from == record.list_date
            and record.effective_to == record.delist_date
            and record.list_date < record.delist_date <= cutoff_date
        )
        if not eligible:
            rejected.append(symbol)
            continue
        fresh[symbol] = PITUniverseRecord.from_dict(record.to_dict())
        carried.append(symbol)
    if rejected:
        raise RuntimeError(
            "stock_basic refresh would shrink canonical PIT membership with "
            "non-carry-forward-eligible rows: "
            f"count={len(rejected)},symbols={rejected[:20]}"
        )
    return [fresh[symbol] for symbol in sorted(fresh)], carried


def records_by_symbol(records: Sequence[PITUniverseRecord]) -> dict[str, PITUniverseRecord]:
    return {record.symbol: record for record in dedupe_latest_records(records) if record.symbol}


def evaluate_listing_status(
    record: PITUniverseRecord | None,
    *,
    symbol: str,
    as_of: str | date | datetime,
) -> PITListingStatus:
    normalized_symbol = normalize_symbol(symbol)
    target_date = compact_date(as_of)
    if record is None:
        return PITListingStatus(
            symbol=normalized_symbol,
            date=target_date,
            reason=REASON_MISSING_PIT_RECORD,
        )

    allowed_qualities = {
        "ok",
        REASON_CONFLICTING_STATUS_ROWS,
        REASON_MISSING_LIST_DATE,
        REASON_MISSING_DELIST_DATE,
        REASON_OUTSIDE_FROZEN_SCOPE_PENDING,
    }
    if record.membership_quality not in allowed_qualities:
        return PITListingStatus(
            symbol=record.symbol,
            date=target_date,
            reason=record.membership_quality or "unsupported_membership_quality",
            list_date=record.list_date,
            delist_date=record.delist_date,
            source_list_status=record.source_list_status,
            observed_at=record.observed_at,
            membership_quality=record.membership_quality,
        )

    if record.membership_quality == REASON_CONFLICTING_STATUS_ROWS:
        return PITListingStatus(
            symbol=record.symbol,
            date=target_date,
            reason=REASON_CONFLICTING_STATUS_ROWS,
            list_date=record.list_date,
            delist_date=record.delist_date,
            source_list_status=record.source_list_status,
            observed_at=record.observed_at,
            membership_quality=record.membership_quality,
        )

    if record.membership_quality == REASON_OUTSIDE_FROZEN_SCOPE_PENDING:
        return PITListingStatus(
            symbol=record.symbol,
            date=target_date,
            in_universe=False,
            research_eligible=False,
            tradable=False,
            reason=REASON_OUTSIDE_FROZEN_SCOPE_PENDING,
            list_date=record.list_date,
            delist_date=record.delist_date,
            source_list_status=record.source_list_status,
            observed_at=record.observed_at,
            membership_quality=record.membership_quality,
        )

    if not record.list_date:
        return PITListingStatus(
            symbol=record.symbol,
            date=target_date,
            reason=REASON_MISSING_LIST_DATE,
            source_list_status=record.source_list_status,
            observed_at=record.observed_at,
            membership_quality=record.membership_quality,
        )

    if target_date and target_date < record.list_date:
        return PITListingStatus(
            symbol=record.symbol,
            date=target_date,
            reason=REASON_PRE_LISTING,
            list_date=record.list_date,
            delist_date=record.delist_date,
            source_list_status=record.source_list_status,
            observed_at=record.observed_at,
            membership_quality=record.membership_quality,
        )

    if record.delist_date and target_date and target_date >= record.delist_date:
        return PITListingStatus(
            symbol=record.symbol,
            date=target_date,
            reason=REASON_DELISTED,
            list_date=record.list_date,
            delist_date=record.delist_date,
            source_list_status=record.source_list_status,
            observed_at=record.observed_at,
            membership_quality=record.membership_quality,
        )

    if record.source_list_status == LIST_STATUS_DELISTED and not record.delist_date:
        return PITListingStatus(
            symbol=record.symbol,
            date=target_date,
            in_universe=False,
            research_eligible=False,
            tradable=False,
            reason=REASON_MISSING_DELIST_DATE,
            list_date=record.list_date,
            source_list_status=record.source_list_status,
            observed_at=record.observed_at,
            membership_quality=record.membership_quality,
        )

    if record.source_list_status == LIST_STATUS_PENDING:
        return PITListingStatus(
            symbol=record.symbol,
            date=target_date,
            in_universe=True,
            research_eligible=True,
            tradable=False,
            reason=REASON_PENDING,
            list_date=record.list_date,
            delist_date=record.delist_date,
            source_list_status=record.source_list_status,
            observed_at=record.observed_at,
            membership_quality=record.membership_quality,
        )

    return PITListingStatus(
        symbol=record.symbol,
        date=target_date,
        in_universe=True,
        research_eligible=True,
        tradable=True,
        reason=REASON_LISTED,
        list_date=record.list_date,
        delist_date=record.delist_date,
        source_list_status=record.source_list_status,
        observed_at=record.observed_at,
        membership_quality=record.membership_quality,
    )


def is_listed(
    symbol: str,
    as_of: str | date | datetime,
    records: Sequence[PITUniverseRecord] | Mapping[str, PITUniverseRecord],
) -> bool:
    by_symbol = records if isinstance(records, Mapping) else records_by_symbol(records)
    status = evaluate_listing_status(
        by_symbol.get(normalize_symbol(symbol)), symbol=symbol, as_of=as_of
    )
    return bool(status.in_universe and status.research_eligible)


def build_pit_universe_mask(
    symbols: Sequence[str],
    dates: Sequence[str | date | datetime],
    records: Sequence[PITUniverseRecord] | Mapping[str, PITUniverseRecord],
    *,
    required: bool = False,
) -> list[list[bool]]:
    """Build a MatrixDataBundle-compatible point-in-time universe mask.

    Missing PIT rows fail open unless the caller explicitly requires complete
    PIT coverage. Explicit pre-listing, delisted, and malformed rows fail
    closed because those rows contain enough local evidence to reject the cell.
    """
    by_symbol = records if isinstance(records, Mapping) else records_by_symbol(records)
    output: list[list[bool]] = []
    for symbol in symbols:
        normalized_symbol = normalize_symbol(symbol)
        row: list[bool] = []
        for current_date in dates:
            status = evaluate_listing_status(
                by_symbol.get(normalized_symbol),
                symbol=normalized_symbol,
                as_of=current_date,
            )
            if status.reason == REASON_MISSING_PIT_RECORD and not required:
                row.append(True)
            else:
                row.append(bool(status.in_universe and status.research_eligible))
        output.append(row)
    return output


def build_pit_delisted_field(
    symbols: Sequence[str],
    dates: Sequence[str | date | datetime],
    records: Sequence[PITUniverseRecord] | Mapping[str, PITUniverseRecord],
) -> list[list[bool]]:
    """Build a tradability ``delisted`` field from PIT listing membership."""
    by_symbol = records if isinstance(records, Mapping) else records_by_symbol(records)
    output: list[list[bool]] = []
    for symbol in symbols:
        normalized_symbol = normalize_symbol(symbol)
        row: list[bool] = []
        for current_date in dates:
            status = evaluate_listing_status(
                by_symbol.get(normalized_symbol),
                symbol=normalized_symbol,
                as_of=current_date,
            )
            row.append(status.reason in {REASON_DELISTED, REASON_MISSING_DELIST_DATE})
        output.append(row)
    return output


@dataclass
class PITUniverseFilterResult:
    symbols: list[str]
    metadata: dict[str, Any]
    quarantine_symbols: list[str]
    untradable_symbols: list[str]


def filter_symbols_by_pit_status(
    symbols: Sequence[str],
    *,
    as_of: str | date | datetime,
    records: Sequence[PITUniverseRecord] | Mapping[str, PITUniverseRecord],
    required: bool = False,
) -> PITUniverseFilterResult:
    by_symbol = records if isinstance(records, Mapping) else records_by_symbol(records)
    normalized_symbols = [
        normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)
    ]
    kept: list[str] = []
    quarantine: list[str] = []
    untradable: list[str] = []
    reasons: dict[str, str] = {}
    statuses: dict[str, dict[str, Any]] = {}

    for symbol in normalized_symbols:
        status = evaluate_listing_status(by_symbol.get(symbol), symbol=symbol, as_of=as_of)
        statuses[symbol] = status.to_dict()
        if status.in_universe and status.research_eligible:
            kept.append(symbol)
            if not status.tradable:
                untradable.append(symbol)
                reasons[symbol] = status.reason
            continue
        if status.reason == REASON_MISSING_PIT_RECORD and not required:
            kept.append(symbol)
            reasons[symbol] = status.reason
            continue
        if required or status.reason in {
            REASON_PRE_LISTING,
            REASON_DELISTED,
            REASON_CONFLICTING_STATUS_ROWS,
            REASON_MISSING_DELIST_DATE,
            REASON_MISSING_LIST_DATE,
        }:
            quarantine.append(symbol)
        reasons[symbol] = status.reason

    coverage_denominator = max(len(normalized_symbols), 1)
    missing_count = sum(
        1 for status in statuses.values() if status.get("reason") == REASON_MISSING_PIT_RECORD
    )
    metadata = {
        "schema_version": PIT_UNIVERSE_SCHEMA_VERSION,
        "enabled": True,
        "required": bool(required),
        "as_of": compact_date(as_of),
        "input_count": len(normalized_symbols),
        "kept_count": len(kept),
        "excluded_count": len(normalized_symbols) - len(kept),
        "missing_count": missing_count,
        "coverage_ratio": (len(normalized_symbols) - missing_count) / coverage_denominator,
        "reasons": reasons,
        "statuses": statuses,
    }
    return PITUniverseFilterResult(
        symbols=kept,
        metadata=metadata,
        quarantine_symbols=quarantine,
        untradable_symbols=untradable,
    )


class PITUniverseStore:
    def __init__(
        self,
        root_dir: str | Path = "data/parquet/cn/reference",
        *,
        raw_root: str | Path | None = None,
        compatibility_path: str | Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        production_root = Path("data/parquet/cn/reference")
        custom_root = self.root_dir != production_root
        self.raw_root = Path(
            raw_root
            if raw_root is not None
            else (self.root_dir.parent / "raw" if custom_root else "data/cn_universe/raw")
        )
        self.compatibility_path = Path(
            compatibility_path
            if compatibility_path is not None
            else (
                self.root_dir.parent / "stock_basic_membership_latest.json"
                if custom_root
                else "data/cn_universe/stock_basic_membership_latest.json"
            )
        )

    @classmethod
    def from_config(cls) -> "PITUniverseStore":
        from quant_investor.config import config

        return cls(
            root_dir=getattr(
                config,
                "PIT_UNIVERSE_SOURCE_ROOT",
                "data/parquet/cn/reference",
            ),
            raw_root="data/cn_universe/raw",
            compatibility_path="data/cn_universe/stock_basic_membership_latest.json",
        )

    @property
    def canonical_path(self) -> Path:
        """Return the frozen legacy canonical path.

        New refreshes never overwrite this file.  Callers that need the
        currently published generation must use ``latest_canonical_path`` or
        ``load_generation_binding`` so the manifest and Parquet hashes remain
        bound together.
        """
        return self.root_dir / "stock_basic_membership.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / "stock_basic_membership_latest.json"

    @property
    def generations_root(self) -> Path:
        return self.root_dir / PIT_UNIVERSE_GENERATIONS_DIRNAME

    @property
    def latest_canonical_path(self) -> Path:
        if self.manifest_path.exists():
            try:
                return Path(self.load_generation_binding()["canonical_path"])
            except RuntimeError as exc:
                if str(exc) != "pit_latest_generation_binding_missing":
                    raise
        return self.canonical_path

    def raw_snapshot_path(self, observed_at: str) -> Path:
        compact = compact_date(observed_at) or datetime.now(timezone.utc).strftime("%Y%m%d")
        suffix = _short_hash([observed_at])
        return self.raw_root / f"stock_basic_pit_snapshot_{compact}_{suffix}.jsonl"

    def _generation_member_path(self, declared: Any, *, blocker: str) -> Path:
        raw = str(declared or "").strip()
        if not raw:
            raise RuntimeError(blocker)
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root_dir / candidate
        lexical_root = Path(os.path.abspath(self.generations_root))
        lexical_candidate = Path(os.path.abspath(candidate))
        try:
            relative = lexical_candidate.relative_to(lexical_root)
        except ValueError as exc:
            raise RuntimeError(blocker) from exc
        current = lexical_root
        if current.is_symlink():
            raise RuntimeError(blocker)
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise RuntimeError(blocker)
        try:
            resolved_root = lexical_root.resolve(strict=True)
            resolved = lexical_candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(blocker) from exc
        return resolved

    @staticmethod
    def _records_sha256(records: Sequence[PITUniverseRecord]) -> str:
        return _sha256_bytes(_json_bytes({"records": [record.to_dict() for record in records]}))

    def load_generation_binding(
        self,
        manifest_path: str | Path | None = None,
        expected_manifest_sha256: str = "",
        *,
        _lineage_depth: int = 0,
        _visited_manifests: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """Load a hash-bound immutable PIT generation.

        With no explicit path, the atomic latest discovery manifest selects
        the generation.  An explicit path is accepted only with the expected
        manifest SHA-256, which prevents a mutable-path lookup from silently
        selecting different bytes.
        """
        if _lineage_depth > PIT_UNIVERSE_MAX_LINEAGE_DEPTH:
            raise RuntimeError("pit_generation_lineage_depth_exceeded")
        discovery_payload: dict[str, Any] = {}
        discovery_pointer_path = ""
        discovery_pointer_sha256 = ""
        if manifest_path is None:
            if not self.manifest_path.exists():
                raise RuntimeError("pit_latest_generation_binding_missing")
            if self.manifest_path.is_symlink():
                raise RuntimeError("pit_latest_pointer_symlink_invalid")
            discovery_bytes = _stable_read_bytes(
                self.manifest_path,
                blocker="pit_latest_pointer_readback_invalid",
            )
            try:
                loaded_pointer = json.loads(discovery_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("pit_latest_pointer_invalid") from exc
            if not isinstance(loaded_pointer, Mapping):
                raise RuntimeError("pit_latest_pointer_invalid")
            discovery_payload = dict(loaded_pointer)
            if discovery_payload.get(
                "discovery_schema_version"
            ) != PIT_UNIVERSE_DISCOVERY_SCHEMA_VERSION or not discovery_payload.get(
                "generation_manifest_path"
            ):
                raise RuntimeError("pit_latest_generation_binding_missing")
            manifest_path = str(discovery_payload["generation_manifest_path"])
            expected_manifest_sha256 = _validate_sha256(
                discovery_payload.get("generation_manifest_sha256"),
                blocker="pit_latest_generation_manifest_sha256_invalid",
            )
            discovery_pointer_path = str(self.manifest_path.resolve(strict=True))
            discovery_pointer_sha256 = _sha256_bytes(discovery_bytes)
        elif not expected_manifest_sha256:
            raise RuntimeError("pit_generation_manifest_expected_sha256_required")

        expected_manifest_sha256 = _validate_sha256(
            expected_manifest_sha256,
            blocker="pit_generation_manifest_expected_sha256_invalid",
        )
        resolved_manifest = self._generation_member_path(
            manifest_path,
            blocker="pit_generation_manifest_path_invalid",
        )
        resolved_manifest_text = str(resolved_manifest)
        if resolved_manifest_text in _visited_manifests:
            raise RuntimeError("pit_generation_lineage_cycle")
        visited = _visited_manifests | {resolved_manifest_text}
        if resolved_manifest.name != PIT_UNIVERSE_GENERATION_MANIFEST_FILENAME:
            raise RuntimeError("pit_generation_manifest_path_invalid")
        manifest_bytes = _stable_read_bytes(
            resolved_manifest,
            blocker="pit_generation_manifest_readback_invalid",
        )
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        if manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError("pit_generation_manifest_sha256_mismatch")
        try:
            loaded_manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("pit_generation_manifest_invalid") from exc
        if not isinstance(loaded_manifest, Mapping):
            raise RuntimeError("pit_generation_manifest_invalid")
        manifest = dict(loaded_manifest)
        generation_id = str(manifest.get("generation_id") or "").strip()
        if (
            manifest.get("schema_version") != PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION
            or manifest.get("membership_schema_version") != PIT_UNIVERSE_SCHEMA_VERSION
            or not generation_id
            or generation_id in {".", ".."}
            or any(
                ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for ch in generation_id
            )
            or resolved_manifest.parent.name != generation_id
            or resolved_manifest.parent.parent != self.generations_root.resolve(strict=True)
        ):
            raise RuntimeError("pit_generation_manifest_invalid")

        canonical_path = self._generation_member_path(
            manifest.get("canonical_path"),
            blocker="pit_generation_canonical_path_invalid",
        )
        expected_canonical_path = (
            resolved_manifest.parent / PIT_UNIVERSE_GENERATION_CANONICAL_FILENAME
        )
        if canonical_path != expected_canonical_path:
            raise RuntimeError("pit_generation_canonical_path_invalid")
        expected_canonical_sha256 = _validate_sha256(
            manifest.get("canonical_sha256"),
            blocker="pit_generation_canonical_sha256_invalid",
        )
        canonical_bytes = _stable_read_bytes(
            canonical_path,
            blocker="pit_generation_canonical_readback_invalid",
        )
        canonical_sha256 = _sha256_bytes(canonical_bytes)
        if canonical_sha256 != expected_canonical_sha256:
            raise RuntimeError("pit_generation_canonical_sha256_mismatch")
        try:
            frame = pd.read_parquet(io.BytesIO(canonical_bytes))
        except Exception as exc:
            raise RuntimeError("pit_generation_canonical_parquet_invalid") from exc
        expected_columns = list(PITUniverseRecord.__dataclass_fields__)
        if list(frame.columns) != expected_columns:
            raise RuntimeError("pit_generation_canonical_schema_invalid")
        records = [PITUniverseRecord.from_dict(row) for row in frame.to_dict(orient="records")]
        if len(records) != manifest.get("row_count"):
            raise RuntimeError("pit_generation_canonical_row_count_mismatch")
        records_sha256 = self._records_sha256(records)
        if records_sha256 != str(manifest.get("records_sha256") or ""):
            raise RuntimeError("pit_generation_records_sha256_mismatch")

        lineage = manifest.get("lineage")
        if lineage is not None:
            if not isinstance(lineage, Mapping):
                raise RuntimeError("pit_generation_lineage_invalid")
            required_lineage_keys = {
                "schema_version",
                "parent_generation_id",
                "parent_discovery_pointer_path",
                "parent_discovery_pointer_sha256",
                "parent_discovery_pointer_artifact_path",
                "parent_generation_manifest_path",
                "parent_generation_manifest_sha256",
                "parent_canonical_path",
                "parent_canonical_sha256",
                "parent_records_sha256",
                "carried_forward_symbols",
                "carried_forward_symbol_count",
                "carried_forward_records_sha256",
            }
            if (
                set(lineage) != required_lineage_keys
                or lineage.get("schema_version") != PIT_UNIVERSE_LINEAGE_SCHEMA_VERSION
            ):
                raise RuntimeError("pit_generation_lineage_invalid")
            parent_pointer_artifact = self._generation_member_path(
                lineage.get("parent_discovery_pointer_artifact_path"),
                blocker="pit_generation_parent_pointer_artifact_invalid",
            )
            if parent_pointer_artifact.parent != resolved_manifest.parent:
                raise RuntimeError("pit_generation_parent_pointer_artifact_invalid")
            parent_pointer_bytes = _stable_read_bytes(
                parent_pointer_artifact,
                blocker="pit_generation_parent_pointer_artifact_invalid",
            )
            if _sha256_bytes(parent_pointer_bytes) != _validate_sha256(
                lineage.get("parent_discovery_pointer_sha256"),
                blocker="pit_generation_parent_pointer_sha256_invalid",
            ):
                raise RuntimeError("pit_generation_parent_pointer_sha256_mismatch")
            try:
                parent_pointer_payload = json.loads(parent_pointer_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("pit_generation_parent_pointer_payload_invalid") from exc
            if not isinstance(parent_pointer_payload, Mapping) or any(
                parent_pointer_payload.get(field) != lineage.get(lineage_field)
                for field, lineage_field in (
                    ("generation_id", "parent_generation_id"),
                    (
                        "generation_manifest_path",
                        "parent_generation_manifest_path",
                    ),
                    (
                        "generation_manifest_sha256",
                        "parent_generation_manifest_sha256",
                    ),
                    ("canonical_path", "parent_canonical_path"),
                    ("canonical_sha256", "parent_canonical_sha256"),
                    ("records_sha256", "parent_records_sha256"),
                )
            ):
                raise RuntimeError("pit_generation_parent_pointer_binding_mismatch")
            parent = self.load_generation_binding(
                lineage.get("parent_generation_manifest_path"),
                str(lineage.get("parent_generation_manifest_sha256") or ""),
                _lineage_depth=_lineage_depth + 1,
                _visited_manifests=visited,
            )
            carried = sorted(
                {normalize_symbol(item) for item in lineage.get("carried_forward_symbols", [])}
            )
            if (
                not carried
                or lineage.get("carried_forward_symbol_count") != len(carried)
                or parent["generation_id"] != lineage.get("parent_generation_id")
                or parent["canonical_path"] != lineage.get("parent_canonical_path")
                or parent["canonical_sha256"] != lineage.get("parent_canonical_sha256")
                or parent["manifest"].get("records_sha256") != lineage.get("parent_records_sha256")
            ):
                raise RuntimeError("pit_generation_lineage_binding_mismatch")
            child_by_symbol = {record.symbol: record for record in records}
            parent_by_symbol = {record.symbol: record for record in parent["records"]}
            if any(
                symbol not in child_by_symbol
                or symbol not in parent_by_symbol
                or child_by_symbol[symbol].to_dict() != parent_by_symbol[symbol].to_dict()
                for symbol in carried
            ):
                raise RuntimeError("pit_generation_carried_record_mismatch")
            if self._records_sha256([child_by_symbol[symbol] for symbol in carried]) != str(
                lineage.get("carried_forward_records_sha256") or ""
            ):
                raise RuntimeError("pit_generation_carried_records_sha256_mismatch")

        if discovery_payload:
            pointer_generation_id = str(discovery_payload.get("generation_id") or "")
            if (
                pointer_generation_id != generation_id
                or str(discovery_payload.get("canonical_path") or "") != str(canonical_path)
                or str(discovery_payload.get("canonical_sha256") or "") != canonical_sha256
            ):
                raise RuntimeError("pit_latest_generation_binding_mismatch")

        return {
            "generation_id": generation_id,
            "generation_manifest_path": str(resolved_manifest),
            "generation_manifest_sha256": manifest_sha256,
            "canonical_path": str(canonical_path),
            "canonical_sha256": canonical_sha256,
            "records": records,
            "manifest": manifest,
            "discovery_pointer_path": discovery_pointer_path,
            "discovery_pointer_sha256": discovery_pointer_sha256,
        }

    def validate_generation_binding(
        self,
        manifest_path: str | Path | None = None,
        expected_manifest_sha256: str = "",
    ) -> dict[str, Any]:
        return self.load_generation_binding(
            manifest_path=manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
        )

    @contextmanager
    def _writer_lock(self) -> Iterator[None]:
        if self.root_dir.is_symlink():
            raise RuntimeError("pit_store_root_symlink_invalid")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.root_dir / ".pit_writer.lock"
        if lock_path.is_symlink():
            raise RuntimeError("pit_writer_lock_symlink_invalid")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def write_snapshot(
        self,
        *,
        raw_records: Sequence[PITUniverseRecord],
        latest_records: Sequence[PITUniverseRecord] | None = None,
        observed_at: str,
        source_run_id: str,
        expected_parent_pointer_sha256: str = "",
        parent_binding: Mapping[str, Any] | None = None,
        carried_forward_symbols: Sequence[str] = (),
        source_bindings: Mapping[str, Any] | None = None,
        write_compatibility_export: bool = True,
    ) -> dict[str, Any]:
        with self._writer_lock():
            parent_pointer_bytes: bytes | None = None
            expected_parent = str(expected_parent_pointer_sha256 or "").strip()
            pointer_present = self.manifest_path.exists() or self.manifest_path.is_symlink()
            if expected_parent == PIT_UNIVERSE_EMPTY_PARENT_POINTER:
                if pointer_present:
                    raise RuntimeError("pit_parent_pointer_cas_mismatch")
            elif expected_parent:
                expected_parent_sha = _validate_sha256(
                    expected_parent,
                    blocker="pit_expected_parent_pointer_sha256_invalid",
                )
                parent_pointer_bytes = _stable_read_bytes(
                    self.manifest_path,
                    blocker="pit_parent_pointer_readback_invalid",
                )
                if _sha256_bytes(parent_pointer_bytes) != expected_parent_sha:
                    raise RuntimeError("pit_parent_pointer_cas_mismatch")
            elif pointer_present:
                # Backward-compatible callers that do not provide an external
                # expectation are still bound to the exact pointer observed
                # inside the writer lock.  Empty never means an unchecked CAS.
                parent_pointer_bytes = _stable_read_bytes(
                    self.manifest_path,
                    blocker="pit_parent_pointer_readback_invalid",
                )
            return self._write_snapshot_locked(
                raw_records=raw_records,
                latest_records=latest_records,
                observed_at=observed_at,
                source_run_id=source_run_id,
                parent_binding=parent_binding,
                carried_forward_symbols=carried_forward_symbols,
                parent_pointer_bytes=parent_pointer_bytes,
                source_bindings=source_bindings,
                write_compatibility_export=write_compatibility_export,
            )

    def _write_snapshot_locked(
        self,
        *,
        raw_records: Sequence[PITUniverseRecord],
        latest_records: Sequence[PITUniverseRecord] | None = None,
        observed_at: str,
        source_run_id: str,
        parent_binding: Mapping[str, Any] | None = None,
        carried_forward_symbols: Sequence[str] = (),
        parent_pointer_bytes: bytes | None = None,
        source_bindings: Mapping[str, Any] | None = None,
        write_compatibility_export: bool = True,
    ) -> dict[str, Any]:
        latest = list(latest_records or dedupe_latest_records(raw_records))
        if self.root_dir.is_symlink():
            raise RuntimeError("pit_store_root_symlink_invalid")
        if self.generations_root.is_symlink():
            raise RuntimeError("pit_generations_root_symlink_invalid")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.generations_root.mkdir(parents=True, exist_ok=True)
        if self.generations_root.is_symlink():
            raise RuntimeError("pit_generations_root_symlink_invalid")
        self.raw_root.mkdir(parents=True, exist_ok=True)

        raw_path = self.raw_snapshot_path(observed_at)
        raw_bytes = b"".join(
            (json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            for record in raw_records
        )
        if raw_path.exists():
            if (
                _stable_read_bytes(raw_path, blocker="pit_raw_snapshot_readback_invalid")
                != raw_bytes
            ):
                raise RuntimeError("pit_raw_snapshot_no_clobber_conflict")
        else:
            _atomic_write_bytes(raw_path, raw_bytes)

        frame = pd.DataFrame([record.to_dict() for record in latest])
        parquet_buffer = io.BytesIO()
        frame.to_parquet(parquet_buffer, index=False)
        canonical_bytes = parquet_buffer.getvalue()
        canonical_sha256 = _sha256_bytes(canonical_bytes)
        records_sha256 = self._records_sha256(latest)
        frozen_source_bindings = dict(_json_safe(source_bindings or {}))
        generation_id = (
            f"pit-{compact_date(observed_at) or 'unknown'}-"
            f"{_short_hash([source_run_id, observed_at, records_sha256, frozen_source_bindings], length=16)}"
        )
        generation_root = (self.generations_root / generation_id).resolve()
        generation_manifest_path = generation_root / PIT_UNIVERSE_GENERATION_MANIFEST_FILENAME
        generation_canonical_path = generation_root / PIT_UNIVERSE_GENERATION_CANONICAL_FILENAME
        status_counts: dict[str, int] = {}
        quality_counts: dict[str, int] = {}
        for record in latest:
            status_counts[record.source_list_status] = (
                status_counts.get(record.source_list_status, 0) + 1
            )
            quality_counts[record.membership_quality] = (
                quality_counts.get(record.membership_quality, 0) + 1
            )

        manifest = {
            "schema_version": PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION,
            "membership_schema_version": PIT_UNIVERSE_SCHEMA_VERSION,
            "source": "tushare.stock_basic",
            "generation_id": generation_id,
            "source_run_id": source_run_id,
            "observed_at": observed_at,
            "written_at": _utc_now_iso(),
            "canonical_path": str(generation_canonical_path.resolve()),
            "canonical_sha256": canonical_sha256,
            "records_sha256": records_sha256,
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": _sha256_bytes(raw_bytes),
            "compatibility_path": str(self.compatibility_path.resolve()),
            "raw_row_count": len(raw_records),
            "row_count": len(latest),
            "status_counts": dict(sorted(status_counts.items())),
            "membership_quality_counts": dict(sorted(quality_counts.items())),
        }
        if frozen_source_bindings:
            manifest["source_bindings"] = frozen_source_bindings
        carried = sorted({normalize_symbol(item) for item in carried_forward_symbols})
        if carried:
            parent = dict(parent_binding or {})
            manifest["lineage"] = {
                "schema_version": PIT_UNIVERSE_LINEAGE_SCHEMA_VERSION,
                "parent_generation_id": str(parent.get("generation_id") or ""),
                "parent_discovery_pointer_path": str(parent.get("discovery_pointer_path") or ""),
                "parent_discovery_pointer_sha256": str(
                    parent.get("discovery_pointer_sha256") or ""
                ),
                "parent_discovery_pointer_artifact_path": str(
                    generation_root / "parent_pointer.json"
                ),
                "parent_generation_manifest_path": str(
                    parent.get("generation_manifest_path") or ""
                ),
                "parent_generation_manifest_sha256": str(
                    parent.get("generation_manifest_sha256") or ""
                ),
                "parent_canonical_path": str(parent.get("canonical_path") or ""),
                "parent_canonical_sha256": str(parent.get("canonical_sha256") or ""),
                "parent_records_sha256": str(
                    (parent.get("manifest") or {}).get("records_sha256") or ""
                ),
                "carried_forward_symbols": carried,
                "carried_forward_symbol_count": len(carried),
                "carried_forward_records_sha256": self._records_sha256(
                    [record for record in latest if record.symbol in set(carried)]
                ),
            }
        manifest_bytes = _json_bytes(manifest)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        if generation_root.exists():
            existing_manifest_bytes = _stable_read_bytes(
                generation_manifest_path,
                blocker="pit_generation_manifest_readback_invalid",
            )
            existing_manifest_sha256 = _sha256_bytes(existing_manifest_bytes)
            existing = self.load_generation_binding(
                generation_manifest_path,
                existing_manifest_sha256,
            )
            if (
                existing["manifest"].get("records_sha256") != records_sha256
                or existing["manifest"].get("source_run_id") != source_run_id
                or existing["manifest"].get("observed_at") != observed_at
                or existing["manifest"].get("source_bindings", {}) != frozen_source_bindings
                or existing["canonical_sha256"] != canonical_sha256
            ):
                raise RuntimeError("pit_generation_no_clobber_conflict")
            manifest = dict(existing["manifest"])
            manifest_bytes = existing_manifest_bytes
            manifest_sha256 = existing_manifest_sha256
        else:
            temporary_root = self.generations_root / (f".{generation_id}.{uuid.uuid4().hex}.tmp")
            temporary_root.mkdir(mode=0o700)
            try:
                _write_bytes_exclusive(
                    temporary_root / PIT_UNIVERSE_GENERATION_CANONICAL_FILENAME,
                    canonical_bytes,
                )
                _write_bytes_exclusive(
                    temporary_root / PIT_UNIVERSE_GENERATION_MANIFEST_FILENAME,
                    manifest_bytes,
                )
                if carried:
                    if not parent_pointer_bytes:
                        raise RuntimeError("pit_parent_pointer_artifact_missing")
                    _write_bytes_exclusive(
                        temporary_root / "parent_pointer.json",
                        parent_pointer_bytes,
                    )
                canonical_readback = _stable_read_bytes(
                    temporary_root / PIT_UNIVERSE_GENERATION_CANONICAL_FILENAME,
                    blocker="pit_generation_prepare_canonical_readback_invalid",
                )
                manifest_readback = _stable_read_bytes(
                    temporary_root / PIT_UNIVERSE_GENERATION_MANIFEST_FILENAME,
                    blocker="pit_generation_prepare_manifest_readback_invalid",
                )
                if _sha256_bytes(canonical_readback) != canonical_sha256:
                    raise RuntimeError("pit_generation_prepare_canonical_sha256_mismatch")
                if _sha256_bytes(manifest_readback) != manifest_sha256:
                    raise RuntimeError("pit_generation_prepare_manifest_sha256_mismatch")
                pd.read_parquet(io.BytesIO(canonical_readback))
                _fsync_directory(temporary_root)
                os.rename(temporary_root, generation_root)
                _fsync_directory(self.generations_root)
            finally:
                if temporary_root.exists():
                    shutil.rmtree(temporary_root)

        binding = self.load_generation_binding(
            generation_manifest_path,
            manifest_sha256,
        )
        discovery = {
            **binding["manifest"],
            "discovery_schema_version": PIT_UNIVERSE_DISCOVERY_SCHEMA_VERSION,
            "generation_manifest_path": binding["generation_manifest_path"],
            "generation_manifest_sha256": binding["generation_manifest_sha256"],
        }
        _atomic_write_bytes(self.manifest_path, _json_bytes(discovery))
        published = self.load_generation_binding()
        compatibility_export_status = "skipped"
        warnings: list[str] = []
        compatibility_payload = {
            "manifest": published["manifest"],
            "records": [record.to_dict() for record in published["records"]],
        }
        if write_compatibility_export:
            compatibility_export_status = "written"
            try:
                self.compatibility_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_bytes(
                    self.compatibility_path,
                    _json_bytes(compatibility_payload),
                )
            except (OSError, RuntimeError) as exc:
                # The discovery pointer and immutable generation are authoritative.
                # This legacy export is intentionally post-CAS and retryable; its
                # failure must never invalidate or roll back canonical state.
                compatibility_export_status = "retryable_warning"
                warnings.append(
                    "pit_compatibility_export_failed:" f"{type(exc).__name__}:{str(exc)}"
                )
        return {
            **published["manifest"],
            "generation_manifest_path": published["generation_manifest_path"],
            "generation_manifest_sha256": published["generation_manifest_sha256"],
            "discovery_pointer_path": published["discovery_pointer_path"],
            "discovery_pointer_sha256": published["discovery_pointer_sha256"],
            "compatibility_export_status": compatibility_export_status,
            "warnings": warnings,
        }

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        payload = dict(json.loads(self.manifest_path.read_text(encoding="utf-8")))
        if payload.get("generation_manifest_path"):
            return dict(self.load_generation_binding()["manifest"])
        return payload

    def load_latest_records(self) -> list[PITUniverseRecord]:
        if self.manifest_path.exists():
            try:
                return list(self.load_generation_binding()["records"])
            except RuntimeError as exc:
                if str(exc) != "pit_latest_generation_binding_missing":
                    raise
        if not self.canonical_path.exists():
            return []
        frame = pd.read_parquet(self.canonical_path)
        if frame is None or frame.empty:
            return []
        return [PITUniverseRecord.from_dict(row) for row in frame.to_dict(orient="records")]

    def records_by_symbol(self) -> dict[str, PITUniverseRecord]:
        return records_by_symbol(self.load_latest_records())

    def listing_status(self, symbol: str, as_of: str | date | datetime) -> PITListingStatus:
        by_symbol = self.records_by_symbol()
        return evaluate_listing_status(
            by_symbol.get(normalize_symbol(symbol)), symbol=symbol, as_of=as_of
        )

    def is_listed(self, symbol: str, as_of: str | date | datetime) -> bool:
        return is_listed(symbol, as_of, self.records_by_symbol())

    def listed_symbols(self, as_of: str | date | datetime) -> list[str]:
        by_symbol = self.records_by_symbol()
        result: list[str] = []
        for symbol in sorted(by_symbol):
            status = evaluate_listing_status(by_symbol[symbol], symbol=symbol, as_of=as_of)
            if status.in_universe and status.research_eligible:
                result.append(symbol)
        return result


def acquire_pit_universe_capture(
    pro: Any,
    *,
    capture_root: str | Path,
    observed_at: str | None = None,
    source_run_id: str | None = None,
    effective_date: str | None = None,
) -> dict[str, Any]:
    """Acquire the three registered ``stock_basic`` partitions exactly once.

    The capture is owner-private, path-backed, immutable after completion, and
    contains no canonical store writes.  Publication must replay this receipt;
    it never receives a provider object and therefore cannot refetch.
    """

    resolved_observed_at = observed_at or _utc_now_iso()
    resolved_run_id = source_run_id or _source_run_id(resolved_observed_at)
    resolved_effective_date = _canonical_yyyymmdd(effective_date)
    if resolved_effective_date is None:
        try:
            resolved_effective_date = datetime.fromisoformat(
                resolved_observed_at.replace("Z", "+00:00")
            ).strftime("%Y%m%d")
        except ValueError as exc:
            raise RuntimeError("pit_capture_effective_date_invalid") from exc
    root = _require_absolute_private_root(
        capture_root,
        blocker="pit_capture_root_not_absolute_private",
    )
    if any(root.iterdir()):
        raise RuntimeError("pit_capture_root_not_empty")

    frames = fetch_stock_basic_frames(pro)
    partition_receipts: list[dict[str, Any]] = []
    total_rows = 0
    fields = list(STOCK_BASIC_FIELDS)
    captured_symbols: set[str] = set()
    external_identities: set[str] = set()
    pending_exclusions: list[tuple[str, int, dict[str, Any]]] = []
    malformed_count = 0
    for list_status in SUPPORTED_LIST_STATUSES:
        frame = frames.get(list_status)
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError("pit_capture_partition_frame_invalid")
        if not frame.empty and list(frame.columns) != fields:
            raise RuntimeError(f"pit_capture_partition_schema_invalid:{list_status}")
        items: list[dict[str, Any]] = []
        for ordinal, source_row in enumerate(frame.to_dict(orient="records")):
            if set(source_row) != set(fields):
                raise RuntimeError(f"pit_capture_partition_schema_invalid:{list_status}")
            row = {field: _json_safe(source_row.get(field)) for field in fields}
            raw_identity = row.get("ts_code")
            try:
                excluded = _external_legacy_delisted_row(
                    row,
                    partition_status=list_status,
                    effective_date=resolved_effective_date,
                )
            except RuntimeError:
                excluded = False
                malformed_count += 1
            if excluded:
                assert isinstance(raw_identity, str)
                if raw_identity in external_identities:
                    malformed_count += 1
                external_identities.add(raw_identity)
                pending_exclusions.append((list_status, ordinal, row))
            else:
                symbol = normalize_symbol(raw_identity)
                if (
                    not _CANONICAL_A_SYMBOL.fullmatch(symbol)
                    or _clean_text(row.get("list_status")).upper() != list_status
                    or symbol in captured_symbols
                ):
                    malformed_count += 1
                captured_symbols.add(symbol)
            items.append(row)
        partition_payload = {
            "schema_version": PIT_UNIVERSE_CAPTURE_PARTITION_SCHEMA_VERSION,
            "source": "tushare.stock_basic",
            "request": {
                "exchange": "",
                "list_status": list_status,
                "fields": fields,
            },
            "response_fields": list(frame.columns),
            "row_count": len(items),
            "items": items,
        }
        partition_bytes = _json_bytes(partition_payload)
        partition_path = root / f"stock_basic_{list_status}.json"
        _write_bytes_exclusive(partition_path, partition_bytes)
        os.chmod(partition_path, 0o400)
        partition_receipts.append(
            {
                "list_status": list_status,
                "path": str(partition_path),
                "sha256": _sha256_bytes(partition_bytes),
                "row_count": len(items),
            }
        )
        total_rows += len(items)

    partition_sha_by_status = {row["list_status"]: row["sha256"] for row in partition_receipts}
    exclusions = [
        _external_exclusion_entry(
            row=row,
            partition_status=status,
            partition_sha256=partition_sha_by_status[status],
            ordinal=ordinal,
        )
        for status, ordinal, row in pending_exclusions
    ]
    exclusion_payload = {
        "schema_version": PIT_UNIVERSE_EXTERNAL_EXCLUSION_SCHEMA_VERSION,
        "effective_date": resolved_effective_date,
        "exclusion_count": len(exclusions),
        "items": exclusions,
    }
    exclusion_bytes = _json_bytes(exclusion_payload)
    exclusion_path = root / PIT_EXTERNAL_EXCLUSION_FILENAME
    _write_bytes_exclusive(exclusion_path, exclusion_bytes)
    os.chmod(exclusion_path, 0o400)
    exclusion_ref = {
        "path": str(exclusion_path),
        "sha256": _sha256_bytes(exclusion_bytes),
        "row_count": len(exclusions),
    }

    receipt = {
        "schema_version": PIT_UNIVERSE_CAPTURE_SCHEMA_VERSION,
        "source": "tushare.stock_basic",
        "source_run_id": resolved_run_id,
        "effective_date": resolved_effective_date,
        "observed_at": resolved_observed_at,
        "captured_at": _utc_now_iso(),
        "provider_call_count": len(SUPPORTED_LIST_STATUSES),
        "provider_accounting": {
            "failed": 0,
            "malformed": malformed_count,
            "canonical_row_count": len(captured_symbols),
            "excluded_provider_external": len(exclusions),
            "has_more": False,
            "partition_count": len(SUPPORTED_LIST_STATUSES),
            "provider_count": total_rows,
            "item_count": total_rows,
        },
        "raw_row_count": total_rows,
        "partitions": partition_receipts,
        "exclusion_inventory": exclusion_ref,
    }
    receipt_bytes = _json_bytes(receipt)
    receipt_path = root / "capture_receipt.json"
    _write_bytes_exclusive(receipt_path, receipt_bytes)
    os.chmod(receipt_path, 0o400)
    _fsync_directory(root)
    os.chmod(root, 0o500)
    return {
        **receipt,
        "capture_root": str(root),
        "capture_receipt_path": str(receipt_path),
        "capture_receipt_sha256": _sha256_bytes(receipt_bytes),
    }


def _load_capture_receipt(
    capture_receipt_path: str | Path,
    expected_capture_sha256: str,
) -> tuple[dict[str, Any], Path]:
    receipt_path = Path(capture_receipt_path).expanduser()
    if not receipt_path.is_absolute() or receipt_path.name != "capture_receipt.json":
        raise RuntimeError("pit_capture_receipt_path_invalid")
    capture_root = receipt_path.parent.resolve(strict=True)
    _require_absolute_private_root(
        capture_root,
        blocker="pit_capture_root_not_absolute_private",
    )
    receipt_path = _capture_member_path(
        capture_root,
        receipt_path,
        expected_name="capture_receipt.json",
        blocker="pit_capture_receipt_path_invalid",
    )
    receipt_bytes = _stable_read_bytes(
        receipt_path,
        blocker="pit_capture_receipt_readback_invalid",
    )
    expected_sha = _validate_sha256(
        expected_capture_sha256,
        blocker="pit_capture_receipt_expected_sha256_invalid",
    )
    if _sha256_bytes(receipt_bytes) != expected_sha:
        raise RuntimeError("pit_capture_receipt_sha256_mismatch")
    receipt = _load_json_mapping_bytes(
        receipt_bytes,
        blocker="pit_capture_receipt_invalid",
    )
    expected_files = {
        "capture_receipt.json",
        PIT_EXTERNAL_EXCLUSION_FILENAME,
        *(f"stock_basic_{status}.json" for status in SUPPORTED_LIST_STATUSES),
    }
    if {path.name for path in capture_root.iterdir()} != expected_files:
        raise RuntimeError("pit_capture_fileset_invalid")
    required_keys = {
        "schema_version",
        "source",
        "source_run_id",
        "effective_date",
        "observed_at",
        "captured_at",
        "provider_call_count",
        "provider_accounting",
        "raw_row_count",
        "partitions",
        "exclusion_inventory",
    }
    if (
        set(receipt) != required_keys
        or receipt.get("schema_version") != PIT_UNIVERSE_CAPTURE_SCHEMA_VERSION
        or receipt.get("source") != "tushare.stock_basic"
        or not str(receipt.get("source_run_id") or "")
        or not str(receipt.get("observed_at") or "")
        or _canonical_yyyymmdd(receipt.get("effective_date")) is None
        or receipt.get("provider_call_count") != len(SUPPORTED_LIST_STATUSES)
    ):
        raise RuntimeError("pit_capture_receipt_invalid")
    accounting = receipt.get("provider_accounting")
    exclusion_ref = receipt.get("exclusion_inventory")
    if not isinstance(exclusion_ref, Mapping) or set(exclusion_ref) != {
        "path",
        "sha256",
        "row_count",
    }:
        raise RuntimeError("pit_capture_exclusion_inventory_ref_invalid")
    excluded_count = exclusion_ref.get("row_count")
    raw_count = receipt.get("raw_row_count")
    canonical_count = (
        raw_count - excluded_count
        if isinstance(raw_count, int) and isinstance(excluded_count, int)
        else None
    )
    if not isinstance(accounting, Mapping) or dict(accounting) != {
        "failed": 0,
        "has_more": False,
        "item_count": receipt.get("raw_row_count"),
        "malformed": 0,
        "canonical_row_count": canonical_count,
        "excluded_provider_external": excluded_count,
        "partition_count": len(SUPPORTED_LIST_STATUSES),
        "provider_count": receipt.get("raw_row_count"),
    }:
        raise RuntimeError("pit_capture_provider_accounting_invalid")
    if (
        not isinstance(excluded_count, int)
        or excluded_count < 0
        or excluded_count > MAX_EXTERNAL_LEGACY_DELISTED_EXCLUSIONS
    ):
        raise RuntimeError("PIT_EXTERNAL_LEGACY_DELISTED_EXCLUSION_LIMIT_EXCEEDED")
    return receipt, capture_root


def _load_frozen_full_a_scope(
    canonical_scope_path: str | Path,
    expected_scope_sha256: str,
) -> tuple[set[str], str, str]:
    scope_path = Path(canonical_scope_path).expanduser()
    if not scope_path.is_absolute() or scope_path.is_symlink():
        raise RuntimeError("pit_full_a_scope_path_invalid")
    scope_bytes = _stable_read_bytes(
        scope_path,
        blocker="pit_full_a_scope_readback_invalid",
    )
    expected_sha = _validate_sha256(
        expected_scope_sha256,
        blocker="pit_full_a_scope_expected_sha256_invalid",
    )
    if _sha256_bytes(scope_bytes) != expected_sha:
        raise RuntimeError("pit_full_a_scope_sha256_mismatch")
    scope = _load_json_mapping_bytes(scope_bytes, blocker="pit_full_a_scope_invalid")
    declared = scope.get("full_a")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError("pit_full_a_scope_invalid")
    symbols = {normalize_symbol(item) for item in declared}
    if (
        len(symbols) != len(declared)
        or "" in symbols
        or any(not _CANONICAL_A_SYMBOL.fullmatch(symbol) for symbol in symbols)
    ):
        raise RuntimeError("pit_full_a_scope_invalid")
    return symbols, str(scope_path.resolve(strict=True)), expected_sha


def _empty_parent_binding() -> dict[str, Any]:
    return {
        "records": [],
        "generation_id": "",
        "generation_manifest_path": "",
        "generation_manifest_sha256": "",
        "canonical_path": "",
        "canonical_sha256": "",
        "discovery_pointer_path": "",
        "discovery_pointer_sha256": "",
        "manifest": {},
    }


def _replay_and_validate_pit_capture(
    capture_receipt_path: str | Path,
    expected_capture_sha256: str,
    *,
    store: PITUniverseStore,
    canonical_scope_path: str | Path,
    expected_scope_sha256: str,
    expected_parent_pointer_sha256: str = "",
) -> tuple[
    dict[str, Any],
    list[PITUniverseRecord],
    list[PITUniverseRecord],
    dict[str, Any],
]:
    receipt, capture_root = _load_capture_receipt(
        capture_receipt_path,
        expected_capture_sha256,
    )
    scope_symbols, scope_path, scope_sha = _load_frozen_full_a_scope(
        canonical_scope_path,
        expected_scope_sha256,
    )
    exclusion_ref = receipt["exclusion_inventory"]
    exclusion_path = _capture_member_path(
        capture_root,
        exclusion_ref["path"],
        expected_name=PIT_EXTERNAL_EXCLUSION_FILENAME,
        blocker="pit_capture_exclusion_inventory_path_invalid",
    )
    exclusion_bytes = _stable_read_bytes(
        exclusion_path, blocker="pit_capture_exclusion_inventory_readback_invalid"
    )
    if _sha256_bytes(exclusion_bytes) != _validate_sha256(
        exclusion_ref["sha256"], blocker="pit_capture_exclusion_inventory_sha256_invalid"
    ):
        raise RuntimeError("pit_capture_exclusion_inventory_sha256_mismatch")
    exclusion_inventory = _load_json_mapping_bytes(
        exclusion_bytes, blocker="pit_capture_exclusion_inventory_invalid"
    )
    if (
        set(exclusion_inventory) != {"schema_version", "effective_date", "exclusion_count", "items"}
        or exclusion_inventory.get("schema_version")
        != PIT_UNIVERSE_EXTERNAL_EXCLUSION_SCHEMA_VERSION
        or exclusion_inventory.get("effective_date") != receipt["effective_date"]
        or not isinstance(exclusion_inventory.get("items"), list)
        or exclusion_inventory.get("exclusion_count") != len(exclusion_inventory["items"])
        or exclusion_inventory.get("exclusion_count") != exclusion_ref["row_count"]
    ):
        raise RuntimeError("pit_capture_exclusion_inventory_invalid")
    expected_exclusions = list(exclusion_inventory["items"])
    observed_exclusions: list[dict[str, Any]] = []
    descriptors = receipt.get("partitions")
    if not isinstance(descriptors, list) or len(descriptors) != len(SUPPORTED_LIST_STATUSES):
        raise RuntimeError("pit_capture_partitions_invalid")
    raw_records: list[PITUniverseRecord] = []
    status_counts: dict[str, int] = {}
    seen_partition_symbols: set[str] = set()
    provider_listed_symbols: set[str] = set()
    pending_scope_rows: list[dict[str, Any]] = []
    total_rows = 0
    fields = list(STOCK_BASIC_FIELDS)
    for expected_status, descriptor in zip(SUPPORTED_LIST_STATUSES, descriptors):
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "list_status",
            "path",
            "sha256",
            "row_count",
        }:
            raise RuntimeError("pit_capture_partition_descriptor_invalid")
        if descriptor.get("list_status") != expected_status:
            raise RuntimeError("pit_capture_partition_order_invalid")
        partition_path = _capture_member_path(
            capture_root,
            descriptor.get("path"),
            expected_name=f"stock_basic_{expected_status}.json",
            blocker="pit_capture_partition_path_invalid",
        )
        partition_bytes = _stable_read_bytes(
            partition_path,
            blocker="pit_capture_partition_readback_invalid",
        )
        partition_sha = _validate_sha256(
            descriptor.get("sha256"),
            blocker="pit_capture_partition_sha256_invalid",
        )
        if _sha256_bytes(partition_bytes) != partition_sha:
            raise RuntimeError("pit_capture_partition_sha256_mismatch")
        partition = _load_json_mapping_bytes(
            partition_bytes,
            blocker="pit_capture_partition_invalid",
        )
        if set(partition) != {
            "schema_version",
            "source",
            "request",
            "response_fields",
            "row_count",
            "items",
        }:
            raise RuntimeError("pit_capture_partition_invalid")
        request = partition.get("request")
        items = partition.get("items")
        if (
            partition.get("schema_version") != PIT_UNIVERSE_CAPTURE_PARTITION_SCHEMA_VERSION
            or partition.get("source") != "tushare.stock_basic"
            or not isinstance(request, Mapping)
            or dict(request) != {"exchange": "", "fields": fields, "list_status": expected_status}
            or not isinstance(items, list)
            or partition.get("row_count") != len(items)
            or descriptor.get("row_count") != len(items)
            or partition.get("response_fields") not in (fields, [])
            or (items and partition.get("response_fields") != fields)
        ):
            raise RuntimeError("pit_capture_partition_invalid")
        partition_symbols: set[str] = set()
        for ordinal, row in enumerate(items):
            if not isinstance(row, Mapping) or set(row) != set(fields):
                raise RuntimeError("pit_capture_partition_row_schema_invalid")
            if _external_legacy_delisted_row(
                row,
                partition_status=expected_status,
                effective_date=receipt["effective_date"],
            ):
                entry = _external_exclusion_entry(
                    row=row,
                    partition_status=expected_status,
                    partition_sha256=partition_sha,
                    ordinal=ordinal,
                )
                if entry in observed_exclusions:
                    raise RuntimeError("pit_capture_external_identity_duplicate")
                observed_exclusions.append(entry)
                total_rows += 1
                continue
            raw_identity = row.get("ts_code")
            if type(raw_identity) is not str:
                raise RuntimeError("pit_capture_partition_row_identity_invalid")
            symbol = normalize_symbol(raw_identity)
            if (
                not _CANONICAL_A_SYMBOL.fullmatch(symbol)
                or _clean_text(row.get("list_status")).upper() != expected_status
                or symbol in partition_symbols
                or symbol in seen_partition_symbols
            ):
                raise RuntimeError("pit_capture_partition_row_identity_invalid")
            if expected_status == LIST_STATUS_LISTED:
                provider_listed_symbols.add(symbol)
                if symbol not in scope_symbols:
                    pending_scope_rows.append(
                        {
                            "identity": raw_identity,
                            "partition_status": expected_status,
                            "partition_sha256": partition_sha,
                            "row_ordinal": ordinal,
                            "row_sha256": _sha256_bytes(_json_bytes(dict(row))),
                        }
                    )
            record = record_from_stock_basic_row(
                row,
                list_status=expected_status,
                observed_at=str(receipt["observed_at"]),
                source_run_id=str(receipt["source_run_id"]),
            )
            if record is None:
                raise RuntimeError("pit_capture_partition_row_identity_invalid")
            if expected_status == LIST_STATUS_LISTED and symbol not in scope_symbols:
                record.membership_quality = REASON_OUTSIDE_FROZEN_SCOPE_PENDING
            partition_symbols.add(symbol)
            raw_records.append(record)
        seen_partition_symbols.update(partition_symbols)
        status_counts[expected_status] = len(partition_symbols)
        total_rows += len(partition_symbols)
    if observed_exclusions != expected_exclusions:
        raise RuntimeError("pit_capture_exclusion_inventory_replay_mismatch")
    pending_scope_rows.sort(key=lambda row: row["identity"])
    pending_scope_sha = _sha256_bytes(_json_bytes({"items": pending_scope_rows}))
    if total_rows != receipt.get("raw_row_count"):
        raise RuntimeError("pit_capture_row_count_mismatch")
    if status_counts.get(LIST_STATUS_LISTED, 0) == 0:
        raise RuntimeError("stock_basic listed status returned no rows")

    fresh_records = dedupe_latest_records(raw_records)
    for record in fresh_records:
        if (
            record.membership_quality not in {"ok", REASON_OUTSIDE_FROZEN_SCOPE_PENDING}
            or not record.list_date
            or record.effective_from != record.list_date
            or (
                record.source_list_status == LIST_STATUS_DELISTED
                and (not record.delist_date or record.effective_to != record.delist_date)
            )
            or (record.delist_date and record.list_date and record.delist_date < record.list_date)
        ):
            raise RuntimeError("pit_capture_effective_interval_invalid")

    if any(entry["identity"] in scope_symbols for entry in observed_exclusions):
        raise RuntimeError("pit_capture_external_identity_scope_collision")
    admitted_listed_symbols = {
        record.symbol
        for record in fresh_records
        if record.source_list_status == LIST_STATUS_LISTED and record.membership_quality == "ok"
    }
    observed_canonical_symbols = {record.symbol for record in raw_records}
    missing_scope = sorted(scope_symbols - observed_canonical_symbols)
    if missing_scope:
        raise RuntimeError(
            "PIT_FULL_A_SCOPE_INCOMPLETE:"
            f"count={len(missing_scope)},symbols={missing_scope[:20]}"
        )
    if admitted_listed_symbols != (scope_symbols & provider_listed_symbols) or len(
        provider_listed_symbols
    ) != len(admitted_listed_symbols) + len(pending_scope_rows):
        raise RuntimeError("pit_scope_expansion_pending_accounting_invalid")

    try:
        parent_binding = store.load_generation_binding()
    except RuntimeError as exc:
        if str(exc) != "pit_latest_generation_binding_missing":
            raise
        parent_binding = _empty_parent_binding()
    parent_scope = dict(
        dict(parent_binding.get("manifest") or {}).get("source_bindings") or {}
    ).get("full_a_scope")
    if parent_binding.get("generation_id") and not isinstance(parent_scope, Mapping):
        legacy_market_path = store.root_dir.parent / "_latest.json"
        legacy_market_raw = _stable_read_bytes(
            legacy_market_path, blocker="pit_legacy_scope_market_pointer_invalid"
        )
        legacy_market = _load_json_mapping_bytes(
            legacy_market_raw, blocker="pit_legacy_scope_market_pointer_invalid"
        )
        legacy_coverage = legacy_market.get("coverage")
        cursor_manifest = dict(parent_binding.get("manifest") or {})
        cursor_manifest_sha = parent_binding.get("generation_manifest_sha256")
        lineage_match = False
        for _depth in range(PIT_UNIVERSE_MAX_LINEAGE_DEPTH):
            if not isinstance(legacy_coverage, Mapping):
                break
            if (
                cursor_manifest.get("generation_id") == legacy_coverage.get("pit_generation_id")
                and cursor_manifest.get("canonical_sha256")
                == legacy_coverage.get("pit_membership_sha256")
                and cursor_manifest_sha == legacy_coverage.get("pit_generation_manifest_sha256")
            ):
                lineage_match = True
                break
            lineage = cursor_manifest.get("lineage")
            if not isinstance(lineage, Mapping):
                break
            parent_manifest_path = lineage.get("parent_generation_manifest_path")
            parent_manifest_sha = lineage.get("parent_generation_manifest_sha256")
            if type(parent_manifest_path) is not str or type(parent_manifest_sha) is not str:
                break
            parent_manifest_raw = _stable_read_bytes(
                Path(parent_manifest_path), blocker="pit_legacy_scope_lineage_invalid"
            )
            if _sha256_bytes(parent_manifest_raw) != parent_manifest_sha:
                raise RuntimeError("pit_legacy_scope_lineage_invalid")
            cursor_manifest = _load_json_mapping_bytes(
                parent_manifest_raw, blocker="pit_legacy_scope_lineage_invalid"
            )
            cursor_manifest_sha = parent_manifest_sha
        if (
            not isinstance(legacy_coverage, Mapping)
            or not lineage_match
            or legacy_coverage.get("expected_scope_sha256")
            != _sha256_bytes("\n".join(sorted(scope_symbols)).encode("utf-8"))
            or legacy_coverage.get("expected_scope_count") != len(scope_symbols)
        ):
            raise RuntimeError("pit_frozen_scope_predecessor_binding_changed")
        parent_scope = {"path": scope_path, "sha256": scope_sha}
    if parent_binding.get("generation_id") and (
        parent_scope.get("path") != scope_path or parent_scope.get("sha256") != scope_sha
    ):
        raise RuntimeError("pit_frozen_scope_predecessor_binding_changed")
    parent_pending = dict(
        dict(parent_binding.get("manifest") or {}).get("source_bindings") or {}
    ).get("scope_expansion_pending")
    if isinstance(parent_pending, Mapping):
        prior_identities = set(parent_pending.get("identities") or [])
        current_identities = {row["identity"] for row in pending_scope_rows}
        transitioned = {
            record.symbol
            for record in raw_records
            if record.source_list_status in {LIST_STATUS_DELISTED, LIST_STATUS_PENDING}
        }
        silently_removed = prior_identities - current_identities - transitioned
        if silently_removed:
            raise RuntimeError("pit_scope_expansion_pending_continuity_invalid")
        if prior_identities & scope_symbols:
            raise RuntimeError("pit_scope_expansion_admission_not_configured")
    parent_records_by_symbol = records_by_symbol(parent_binding.get("records") or [])
    fresh_records_by_symbol = records_by_symbol(fresh_records)
    transition_rows: list[dict[str, Any]] = []
    for identity in sorted(row["identity"] for row in pending_scope_rows):
        current_record = fresh_records_by_symbol[identity]
        predecessor_record = parent_records_by_symbol.get(identity)
        if predecessor_record is None:
            transition = "NEW_PENDING"
            predecessor_ref = None
        elif predecessor_record.membership_quality == "ok":
            transition = "AUTHORITY_REPAIR_TO_PENDING"
            predecessor_ref = {
                "record_sha256": _sha256_bytes(_json_bytes(predecessor_record.to_dict())),
                "source_list_status": predecessor_record.source_list_status,
                "list_date": predecessor_record.list_date,
            }
        elif predecessor_record.membership_quality == REASON_OUTSIDE_FROZEN_SCOPE_PENDING:
            transition = "PENDING_CONTINUITY"
            predecessor_ref = {
                "record_sha256": _sha256_bytes(_json_bytes(predecessor_record.to_dict())),
                "source_list_status": predecessor_record.source_list_status,
                "list_date": predecessor_record.list_date,
            }
        else:
            raise RuntimeError("pit_scope_expansion_pending_transition_invalid")
        if predecessor_record is not None and (
            predecessor_record.symbol != current_record.symbol
            or predecessor_record.source_list_status != current_record.source_list_status
            or predecessor_record.list_date != current_record.list_date
        ):
            raise RuntimeError("pit_scope_expansion_pending_identity_drift")
        transition_rows.append(
            {
                "identity": identity,
                "transition": transition,
                "predecessor": predecessor_ref,
                "successor": {
                    "record_sha256": _sha256_bytes(_json_bytes(current_record.to_dict())),
                    "source_list_status": current_record.source_list_status,
                    "list_date": current_record.list_date,
                    "membership_quality": current_record.membership_quality,
                },
            }
        )
    transition_sha = _sha256_bytes(_json_bytes({"items": transition_rows}))
    requested_parent = (
        str(expected_parent_pointer_sha256 or "").strip() or PIT_UNIVERSE_EMPTY_PARENT_POINTER
    )
    if parent_binding["discovery_pointer_sha256"]:
        if requested_parent == PIT_UNIVERSE_EMPTY_PARENT_POINTER:
            raise RuntimeError("pit_parent_pointer_cas_mismatch")
        expected_parent_sha = _validate_sha256(
            requested_parent,
            blocker="pit_expected_parent_pointer_sha256_invalid",
        )
        if parent_binding["discovery_pointer_sha256"] != expected_parent_sha:
            raise RuntimeError("pit_parent_pointer_cas_mismatch")
        effective_expected_parent = expected_parent_sha
    else:
        if requested_parent not in {"", PIT_UNIVERSE_EMPTY_PARENT_POINTER}:
            raise RuntimeError("pit_parent_pointer_cas_mismatch")
        effective_expected_parent = PIT_UNIVERSE_EMPTY_PARENT_POINTER

    latest_records, carried_forward_symbols = carry_forward_historical_records(
        fresh_records,
        parent_binding["records"],
        required_symbols=scope_symbols,
        observed_at=str(receipt["observed_at"]),
    )
    existing_symbols = {record.symbol for record in parent_binding["records"] if record.symbol}
    latest_symbols = {record.symbol for record in latest_records if record.symbol}
    missing_existing = sorted(existing_symbols - latest_symbols)
    if missing_existing:
        raise RuntimeError(
            "stock_basic refresh would shrink canonical PIT membership: "
            f"count={len(missing_existing)},symbols={missing_existing[:20]}"
        )
    report = {
        "schema_version": PIT_UNIVERSE_CAPTURE_VALIDATION_SCHEMA_VERSION,
        "capture_receipt_path": str(Path(capture_receipt_path).resolve(strict=True)),
        "capture_receipt_sha256": _validate_sha256(
            expected_capture_sha256,
            blocker="pit_capture_receipt_expected_sha256_invalid",
        ),
        "source_run_id": str(receipt["source_run_id"]),
        "observed_at": str(receipt["observed_at"]),
        "effective_date": str(receipt["effective_date"]),
        "provider_call_count": receipt["provider_call_count"],
        "provider_accounting": dict(receipt["provider_accounting"]),
        "exclusion_inventory": dict(exclusion_ref),
        "excluded_provider_external_count": len(observed_exclusions),
        "authority_scope": "FROZEN_FULL_A",
        "authority_scope_complete": True,
        "dynamic_whole_market_complete": not pending_scope_rows,
        "provider_listed_count": len(provider_listed_symbols),
        "authority_scope_count": len(scope_symbols),
        "scope_expansion_pending": bool(pending_scope_rows),
        "scope_expansion_pending_count": len(pending_scope_rows),
        "scope_expansion_pending_sha256": pending_scope_sha,
        "scope_expansion_pending_rows": pending_scope_rows,
        "scope_expansion_transition_count": len(transition_rows),
        "scope_expansion_transition_sha256": transition_sha,
        "scope_expansion_transition_rows": transition_rows,
        "status_counts": status_counts,
        "raw_row_count": receipt["raw_row_count"],
        "canonical_row_count": len(raw_records),
        "row_count": len(latest_records),
        "full_a_scope_path": scope_path,
        "full_a_scope_sha256": scope_sha,
        "full_a_scope_count": len(scope_symbols),
        "membership_nonshrinking": True,
        "carried_forward_symbols": carried_forward_symbols,
        "carried_forward_symbol_count": len(carried_forward_symbols),
        "parent_generation_id": parent_binding["generation_id"],
        "parent_generation_manifest_sha256": parent_binding["generation_manifest_sha256"],
        "parent_canonical_sha256": parent_binding["canonical_sha256"],
        "parent_discovery_pointer_sha256": parent_binding["discovery_pointer_sha256"],
        "expected_parent_pointer_sha256": effective_expected_parent,
    }
    return report, raw_records, latest_records, parent_binding


def validate_pit_universe_capture(
    capture_receipt_path: str | Path,
    expected_capture_sha256: str,
    *,
    store: PITUniverseStore,
    canonical_scope_path: str | Path,
    expected_scope_sha256: str,
    expected_parent_pointer_sha256: str = "",
) -> dict[str, Any]:
    report, _, _, _ = _replay_and_validate_pit_capture(
        capture_receipt_path,
        expected_capture_sha256,
        store=store,
        canonical_scope_path=canonical_scope_path,
        expected_scope_sha256=expected_scope_sha256,
        expected_parent_pointer_sha256=expected_parent_pointer_sha256,
    )
    return report


def _write_shadow_pit_candidate(
    *,
    shadow_root: str | Path,
    validation: Mapping[str, Any],
    raw_records: Sequence[PITUniverseRecord],
    latest_records: Sequence[PITUniverseRecord],
    parent_binding: Mapping[str, Any],
) -> dict[str, Any]:
    root = _require_absolute_private_root(
        shadow_root,
        blocker="pit_shadow_root_not_absolute_private",
    )
    if any(root.iterdir()):
        raise RuntimeError("pit_shadow_root_not_empty")
    candidate_store = PITUniverseStore(
        root_dir=root / "reference",
        raw_root=root / "raw",
        compatibility_path=root / "compatibility-export-disabled.json",
    )
    source_bindings = {
        "capture": {
            "schema_version": PIT_UNIVERSE_CAPTURE_SCHEMA_VERSION,
            "path": validation["capture_receipt_path"],
            "sha256": validation["capture_receipt_sha256"],
        },
        "external_exclusion_inventory": dict(validation["exclusion_inventory"]),
        "scope_expansion_pending": {
            "schema_version": "cn_pit_scope_expansion_pending.v1",
            "authority_scope": "FROZEN_FULL_A",
            "admission_status": "NOT_CONFIGURED",
            "count": validation["scope_expansion_pending_count"],
            "sha256": validation["scope_expansion_pending_sha256"],
            "identities": [row["identity"] for row in validation["scope_expansion_pending_rows"]],
            "rows": list(validation["scope_expansion_pending_rows"]),
            "transition_count": validation["scope_expansion_transition_count"],
            "transition_sha256": validation["scope_expansion_transition_sha256"],
            "transitions": list(validation["scope_expansion_transition_rows"]),
        },
        "full_a_scope": {
            "path": validation["full_a_scope_path"],
            "sha256": validation["full_a_scope_sha256"],
        },
        "canonical_parent": {
            "generation_id": parent_binding.get("generation_id", ""),
            "generation_manifest_path": parent_binding.get("generation_manifest_path", ""),
            "generation_manifest_sha256": parent_binding.get("generation_manifest_sha256", ""),
            "canonical_path": parent_binding.get("canonical_path", ""),
            "canonical_sha256": parent_binding.get("canonical_sha256", ""),
            "discovery_pointer_path": parent_binding.get("discovery_pointer_path", ""),
            "discovery_pointer_sha256": parent_binding.get("discovery_pointer_sha256", ""),
        },
        "shadow_authority": {
            "schema_version": PIT_UNIVERSE_SHADOW_CANDIDATE_SCHEMA_VERSION,
            "canonical_write_authorized": False,
            "promotion_eligible": False,
        },
    }
    published = candidate_store.write_snapshot(
        raw_records=raw_records,
        latest_records=latest_records,
        observed_at=str(validation["observed_at"]),
        source_run_id=str(validation["source_run_id"]),
        source_bindings=source_bindings,
        write_compatibility_export=False,
    )
    binding = candidate_store.load_generation_binding()
    if (
        binding["generation_id"] != published["generation_id"]
        or binding["generation_manifest_sha256"] != published["generation_manifest_sha256"]
        or binding["canonical_sha256"] != published["canonical_sha256"]
    ):
        raise RuntimeError("pit_shadow_generation_readback_mismatch")
    os.chmod(root, 0o500)
    return {
        **published,
        "canonical_write_authorized": False,
        "promotion_eligible": False,
        "pit_store_root": str(candidate_store.root_dir.resolve(strict=True)),
        "compatibility_export_status": "skipped",
    }


def publish_pit_universe_capture(
    capture_receipt_path: str | Path,
    expected_capture_sha256: str,
    *,
    store: PITUniverseStore,
    canonical_scope_path: str | Path,
    expected_scope_sha256: str,
    expected_parent_pointer_sha256: str = "",
    canonical: bool = True,
    shadow_root: str | Path | None = None,
) -> dict[str, Any]:
    """Publish or shadow-materialize a validated sealed PIT capture.

    No provider handle is accepted.  The canonical path performs an
    expected-parent reread under the store lock before pointer CAS.
    """

    validation, raw_records, latest_records, parent_binding = _replay_and_validate_pit_capture(
        capture_receipt_path,
        expected_capture_sha256,
        store=store,
        canonical_scope_path=canonical_scope_path,
        expected_scope_sha256=expected_scope_sha256,
        expected_parent_pointer_sha256=expected_parent_pointer_sha256,
    )
    if not canonical:
        if shadow_root is None:
            raise RuntimeError("pit_shadow_root_required")
        candidate = _write_shadow_pit_candidate(
            shadow_root=shadow_root,
            validation=validation,
            raw_records=raw_records,
            latest_records=latest_records,
            parent_binding=parent_binding,
        )
        return {
            **validation,
            "execute": False,
            "manifest": candidate,
            **{
                key: candidate[key]
                for key in (
                    "generation_id",
                    "generation_manifest_path",
                    "generation_manifest_sha256",
                    "canonical_path",
                    "canonical_sha256",
                    "discovery_pointer_path",
                    "discovery_pointer_sha256",
                )
            },
            "shadow_candidate": candidate,
            "compatibility_export_status": "skipped",
            "warnings": [],
        }
    if shadow_root is not None:
        raise RuntimeError("pit_shadow_root_forbidden_for_canonical_publish")
    published = store.write_snapshot(
        raw_records=raw_records,
        latest_records=latest_records,
        observed_at=str(validation["observed_at"]),
        source_run_id=str(validation["source_run_id"]),
        expected_parent_pointer_sha256=str(validation["expected_parent_pointer_sha256"]),
        parent_binding=parent_binding,
        carried_forward_symbols=validation["carried_forward_symbols"],
        source_bindings={
            "capture": {
                "schema_version": PIT_UNIVERSE_CAPTURE_SCHEMA_VERSION,
                "path": validation["capture_receipt_path"],
                "sha256": validation["capture_receipt_sha256"],
            },
            "external_exclusion_inventory": dict(validation["exclusion_inventory"]),
            "scope_expansion_pending": {
                "schema_version": "cn_pit_scope_expansion_pending.v1",
                "authority_scope": "FROZEN_FULL_A",
                "admission_status": "NOT_CONFIGURED",
                "count": validation["scope_expansion_pending_count"],
                "sha256": validation["scope_expansion_pending_sha256"],
                "identities": [
                    row["identity"] for row in validation["scope_expansion_pending_rows"]
                ],
                "rows": list(validation["scope_expansion_pending_rows"]),
                "transition_count": validation["scope_expansion_transition_count"],
                "transition_sha256": validation["scope_expansion_transition_sha256"],
                "transitions": list(validation["scope_expansion_transition_rows"]),
            },
            "full_a_scope": {
                "path": validation["full_a_scope_path"],
                "sha256": validation["full_a_scope_sha256"],
            },
        },
    )
    return {
        **validation,
        "execute": True,
        "manifest": published,
        **{
            key: published[key]
            for key in (
                "generation_id",
                "generation_manifest_path",
                "generation_manifest_sha256",
                "canonical_path",
                "canonical_sha256",
                "discovery_pointer_path",
                "discovery_pointer_sha256",
            )
        },
        "compatibility_export_status": published.get("compatibility_export_status", "written"),
        "warnings": list(published.get("warnings", [])),
    }


def fetch_stock_basic_frames(pro: Any) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    fields = ",".join(STOCK_BASIC_FIELDS)
    for list_status in SUPPORTED_LIST_STATUSES:
        frame = pro.stock_basic(exchange="", list_status=list_status, fields=fields)
        if frame is None:
            frame = pd.DataFrame()
        frames[list_status] = frame
    return frames


def refresh_pit_universe_from_tushare(
    pro: Any,
    *,
    store: PITUniverseStore,
    execute: bool = False,
    observed_at: str | None = None,
    source_run_id: str | None = None,
    required_symbols: Iterable[str] | None = None,
    expected_parent_pointer_sha256: str = "",
) -> dict[str, Any]:
    resolved_observed_at = observed_at or _utc_now_iso()
    resolved_run_id = source_run_id or _source_run_id(resolved_observed_at)
    frames = fetch_stock_basic_frames(pro)
    listed_frame = frames.get(LIST_STATUS_LISTED, pd.DataFrame())
    if not isinstance(listed_frame, pd.DataFrame) or listed_frame.empty:
        raise RuntimeError("stock_basic listed status returned no rows")
    raw_records = records_from_stock_basic_frames(
        frames,
        observed_at=resolved_observed_at,
        source_run_id=resolved_run_id,
    )
    fresh_records = dedupe_latest_records(raw_records)
    fresh_symbols = {record.symbol for record in fresh_records if record.symbol}
    required = {
        normalize_symbol(symbol) for symbol in required_symbols or [] if normalize_symbol(symbol)
    }
    missing_required = sorted(required - fresh_symbols)
    if missing_required:
        raise RuntimeError(
            "stock_basic refresh omits required current components: "
            f"count={len(missing_required)},symbols={missing_required[:20]}"
        )
    try:
        parent_binding = store.load_generation_binding()
    except RuntimeError as exc:
        if str(exc) != "pit_latest_generation_binding_missing":
            raise
        parent_binding = {
            "records": [],
            "generation_id": "",
            "generation_manifest_path": "",
            "generation_manifest_sha256": "",
            "canonical_path": "",
            "canonical_sha256": "",
            "discovery_pointer_path": "",
            "discovery_pointer_sha256": "",
            "manifest": {},
        }
    if expected_parent_pointer_sha256:
        expected_parent_sha = _validate_sha256(
            expected_parent_pointer_sha256,
            blocker="pit_expected_parent_pointer_sha256_invalid",
        )
        if parent_binding["discovery_pointer_sha256"] != expected_parent_sha:
            raise RuntimeError("pit_parent_pointer_cas_mismatch")
    elif execute and parent_binding["discovery_pointer_sha256"]:
        raise RuntimeError("pit_expected_parent_pointer_sha256_required")
    latest_records, carried_forward_symbols = carry_forward_historical_records(
        fresh_records,
        parent_binding["records"],
        required_symbols=required_symbols or (),
        observed_at=resolved_observed_at,
    )
    latest_symbols = {record.symbol for record in latest_records if record.symbol}
    existing_symbols = {record.symbol for record in parent_binding["records"] if record.symbol}
    missing_existing = sorted(existing_symbols - latest_symbols)
    if missing_existing:
        raise RuntimeError(
            "stock_basic refresh would shrink canonical PIT membership: "
            f"count={len(missing_existing)},symbols={missing_existing[:20]}"
        )
    status_counts = {
        status: int(len(frames.get(status, pd.DataFrame()))) for status in SUPPORTED_LIST_STATUSES
    }
    report: dict[str, Any] = {
        "schema_version": PIT_UNIVERSE_REFRESH_SCHEMA_VERSION,
        "source": "tushare.stock_basic",
        "source_run_id": resolved_run_id,
        "observed_at": resolved_observed_at,
        "execute": bool(execute),
        "provider_call_count": len(SUPPORTED_LIST_STATUSES),
        "raw_row_count": len(raw_records),
        "row_count": len(latest_records),
        "status_counts": status_counts,
        "required_symbol_count": len(required),
        "required_symbols_missing": missing_required,
        "existing_symbol_count": len(existing_symbols),
        "existing_symbols_missing": missing_existing,
        "membership_nonshrinking": True,
        "carried_forward_symbols": carried_forward_symbols,
        "carried_forward_symbol_count": len(carried_forward_symbols),
        "parent_generation_id": parent_binding["generation_id"],
        "parent_generation_manifest_sha256": parent_binding["generation_manifest_sha256"],
        "parent_canonical_sha256": parent_binding["canonical_sha256"],
        "parent_discovery_pointer_sha256": parent_binding["discovery_pointer_sha256"],
        "manifest": {},
    }
    if execute:
        published = store.write_snapshot(
            raw_records=raw_records,
            latest_records=latest_records,
            observed_at=resolved_observed_at,
            source_run_id=resolved_run_id,
            expected_parent_pointer_sha256=expected_parent_pointer_sha256,
            parent_binding=parent_binding,
            carried_forward_symbols=carried_forward_symbols,
        )
        report["manifest"] = published
        for evidence_key in (
            "generation_id",
            "generation_manifest_path",
            "generation_manifest_sha256",
            "canonical_path",
            "canonical_sha256",
            "discovery_pointer_path",
            "discovery_pointer_sha256",
        ):
            report[evidence_key] = published[evidence_key]
    return report


def estimate_historical_bar_backfill_cost(
    *,
    missing_trade_dates: int,
    endpoints_per_date: int = 3,
    unresolved_symbol_dates: int = 0,
    calls_per_symbol_date: int = 2,
) -> dict[str, Any]:
    date_scoped_calls = max(int(missing_trade_dates), 0) * max(int(endpoints_per_date), 0)
    symbol_tail_calls = max(int(unresolved_symbol_dates), 0) * max(int(calls_per_symbol_date), 0)
    return {
        "schema_version": "cn_pit_universe_backfill_cost.v1",
        "stock_basic_refresh_calls": len(SUPPORTED_LIST_STATUSES),
        "missing_trade_dates": max(int(missing_trade_dates), 0),
        "endpoints_per_date": max(int(endpoints_per_date), 0),
        "date_scoped_bar_calls": date_scoped_calls,
        "unresolved_symbol_dates": max(int(unresolved_symbol_dates), 0),
        "calls_per_symbol_date": max(int(calls_per_symbol_date), 0),
        "symbol_tail_calls": symbol_tail_calls,
        "total_estimated_calls": len(SUPPORTED_LIST_STATUSES)
        + date_scoped_calls
        + symbol_tail_calls,
    }


__all__ = [
    "PIT_UNIVERSE_SCHEMA_VERSION",
    "PIT_UNIVERSE_CAPTURE_SCHEMA_VERSION",
    "PIT_UNIVERSE_EMPTY_PARENT_POINTER",
    "PITUniverseRecord",
    "PITListingStatus",
    "PITUniverseFilterResult",
    "PITUniverseStore",
    "SUPPORTED_LIST_STATUSES",
    "acquire_pit_universe_capture",
    "build_pit_delisted_field",
    "build_pit_universe_mask",
    "compact_date",
    "carry_forward_historical_records",
    "dedupe_latest_records",
    "estimate_historical_bar_backfill_cost",
    "evaluate_listing_status",
    "fetch_stock_basic_frames",
    "filter_symbols_by_pit_status",
    "is_listed",
    "normalize_symbol",
    "publish_pit_universe_capture",
    "records_by_symbol",
    "records_from_stock_basic_frames",
    "refresh_pit_universe_from_tushare",
    "validate_pit_universe_capture",
]
