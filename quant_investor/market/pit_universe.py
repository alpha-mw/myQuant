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

REASON_LISTED = "listed"
REASON_PRE_LISTING = "pre_listing"
REASON_PENDING = "pending"
REASON_DELISTED = "delisted"
REASON_MISSING_PIT_RECORD = "missing_pit_record"
REASON_CONFLICTING_STATUS_ROWS = "conflicting_status_rows"
REASON_MISSING_LIST_DATE = "missing_list_date"
REASON_MISSING_DELIST_DATE = "missing_delist_date"


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
        if not stat.S_ISREG(before.st_mode):
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

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.date = compact_date(self.date)
        self.reason = _clean_text(self.reason)
        self.list_date = compact_date(self.list_date)
        self.delist_date = compact_date(self.delist_date)
        self.source_list_status = _clean_text(self.source_list_status).upper()
        self.observed_at = _clean_text(self.observed_at)
        self.membership_quality = _clean_text(self.membership_quality)

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
    if any(record.source_list_status == LIST_STATUS_DELISTED for record in records) and not delist_dates:
        return REASON_MISSING_DELIST_DATE
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
    status = evaluate_listing_status(by_symbol.get(normalize_symbol(symbol)), symbol=symbol, as_of=as_of)
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
    normalized_symbols = [normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)]
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
    missing_count = sum(1 for status in statuses.values() if status.get("reason") == REASON_MISSING_PIT_RECORD)
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
            else (
                self.root_dir.parent / "raw"
                if custom_root
                else "data/cn_universe/raw"
            )
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
        return _sha256_bytes(
            _json_bytes({"records": [record.to_dict() for record in records]})
        )

    def load_generation_binding(
        self,
        manifest_path: str | Path | None = None,
        expected_manifest_sha256: str = "",
    ) -> dict[str, Any]:
        """Load a hash-bound immutable PIT generation.

        With no explicit path, the atomic latest discovery manifest selects
        the generation.  An explicit path is accepted only with the expected
        manifest SHA-256, which prevents a mutable-path lookup from silently
        selecting different bytes.
        """
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
            if (
                discovery_payload.get("discovery_schema_version")
                != PIT_UNIVERSE_DISCOVERY_SCHEMA_VERSION
                or not discovery_payload.get("generation_manifest_path")
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
            or manifest.get("membership_schema_version")
            != PIT_UNIVERSE_SCHEMA_VERSION
            or not generation_id
            or generation_id in {".", ".."}
            or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in generation_id)
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
        records = [
            PITUniverseRecord.from_dict(row)
            for row in frame.to_dict(orient="records")
        ]
        if len(records) != manifest.get("row_count"):
            raise RuntimeError("pit_generation_canonical_row_count_mismatch")
        records_sha256 = self._records_sha256(records)
        if records_sha256 != str(manifest.get("records_sha256") or ""):
            raise RuntimeError("pit_generation_records_sha256_mismatch")

        if discovery_payload:
            pointer_generation_id = str(
                discovery_payload.get("generation_id") or ""
            )
            if (
                pointer_generation_id != generation_id
                or str(discovery_payload.get("canonical_path") or "")
                != str(canonical_path)
                or str(discovery_payload.get("canonical_sha256") or "")
                != canonical_sha256
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
    ) -> dict[str, Any]:
        with self._writer_lock():
            return self._write_snapshot_locked(
                raw_records=raw_records,
                latest_records=latest_records,
                observed_at=observed_at,
                source_run_id=source_run_id,
            )

    def _write_snapshot_locked(
        self,
        *,
        raw_records: Sequence[PITUniverseRecord],
        latest_records: Sequence[PITUniverseRecord] | None = None,
        observed_at: str,
        source_run_id: str,
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
        self.compatibility_path.parent.mkdir(parents=True, exist_ok=True)

        raw_path = self.raw_snapshot_path(observed_at)
        raw_bytes = b"".join(
            (
                json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            for record in raw_records
        )
        if raw_path.exists():
            if _stable_read_bytes(raw_path, blocker="pit_raw_snapshot_readback_invalid") != raw_bytes:
                raise RuntimeError("pit_raw_snapshot_no_clobber_conflict")
        else:
            _atomic_write_bytes(raw_path, raw_bytes)

        frame = pd.DataFrame([record.to_dict() for record in latest])
        parquet_buffer = io.BytesIO()
        frame.to_parquet(parquet_buffer, index=False)
        canonical_bytes = parquet_buffer.getvalue()
        canonical_sha256 = _sha256_bytes(canonical_bytes)
        records_sha256 = self._records_sha256(latest)
        generation_id = (
            f"pit-{compact_date(observed_at) or 'unknown'}-"
            f"{_short_hash([source_run_id, observed_at, records_sha256], length=16)}"
        )
        generation_root = (self.generations_root / generation_id).resolve()
        generation_manifest_path = (
            generation_root / PIT_UNIVERSE_GENERATION_MANIFEST_FILENAME
        )
        generation_canonical_path = (
            generation_root / PIT_UNIVERSE_GENERATION_CANONICAL_FILENAME
        )
        status_counts: dict[str, int] = {}
        quality_counts: dict[str, int] = {}
        for record in latest:
            status_counts[record.source_list_status] = status_counts.get(record.source_list_status, 0) + 1
            quality_counts[record.membership_quality] = quality_counts.get(record.membership_quality, 0) + 1

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
                or existing["canonical_sha256"] != canonical_sha256
            ):
                raise RuntimeError("pit_generation_no_clobber_conflict")
            manifest = dict(existing["manifest"])
            manifest_bytes = existing_manifest_bytes
            manifest_sha256 = existing_manifest_sha256
        else:
            temporary_root = self.generations_root / (
                f".{generation_id}.{uuid.uuid4().hex}.tmp"
            )
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
        compatibility_payload = {
            "manifest": binding["manifest"],
            "records": [record.to_dict() for record in binding["records"]],
        }
        _atomic_write_bytes(
            self.compatibility_path,
            _json_bytes(compatibility_payload),
        )
        discovery = {
            **binding["manifest"],
            "discovery_schema_version": PIT_UNIVERSE_DISCOVERY_SCHEMA_VERSION,
            "generation_manifest_path": binding["generation_manifest_path"],
            "generation_manifest_sha256": binding["generation_manifest_sha256"],
        }
        _atomic_write_bytes(self.manifest_path, _json_bytes(discovery))
        published = self.load_generation_binding()
        return {
            **published["manifest"],
            "generation_manifest_path": published["generation_manifest_path"],
            "generation_manifest_sha256": published["generation_manifest_sha256"],
            "discovery_pointer_path": published["discovery_pointer_path"],
            "discovery_pointer_sha256": published["discovery_pointer_sha256"],
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
        return evaluate_listing_status(by_symbol.get(normalize_symbol(symbol)), symbol=symbol, as_of=as_of)

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


def fetch_stock_basic_frames(pro: Any) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    fields = "ts_code,name,area,industry,market,list_date,delist_date,list_status"
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
    latest_records = dedupe_latest_records(raw_records)
    latest_symbols = {record.symbol for record in latest_records if record.symbol}
    required = {
        normalize_symbol(symbol)
        for symbol in required_symbols or []
        if normalize_symbol(symbol)
    }
    missing_required = sorted(required - latest_symbols)
    if missing_required:
        raise RuntimeError(
            "stock_basic refresh omits required current components: "
            f"count={len(missing_required)},symbols={missing_required[:20]}"
        )
    existing_symbols = {
        record.symbol
        for record in store.load_latest_records()
        if record.symbol
    }
    missing_existing = sorted(existing_symbols - latest_symbols)
    if missing_existing:
        raise RuntimeError(
            "stock_basic refresh would shrink canonical PIT membership: "
            f"count={len(missing_existing)},symbols={missing_existing[:20]}"
        )
    status_counts = {status: int(len(frames.get(status, pd.DataFrame()))) for status in SUPPORTED_LIST_STATUSES}
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
        "manifest": {},
    }
    if execute:
        published = store.write_snapshot(
            raw_records=raw_records,
            latest_records=latest_records,
            observed_at=resolved_observed_at,
            source_run_id=resolved_run_id,
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
        "total_estimated_calls": len(SUPPORTED_LIST_STATUSES) + date_scoped_calls + symbol_tail_calls,
    }


__all__ = [
    "PIT_UNIVERSE_SCHEMA_VERSION",
    "PITUniverseRecord",
    "PITListingStatus",
    "PITUniverseFilterResult",
    "PITUniverseStore",
    "SUPPORTED_LIST_STATUSES",
    "build_pit_delisted_field",
    "build_pit_universe_mask",
    "compact_date",
    "dedupe_latest_records",
    "estimate_historical_bar_backfill_cost",
    "evaluate_listing_status",
    "fetch_stock_basic_frames",
    "filter_symbols_by_pit_status",
    "is_listed",
    "normalize_symbol",
    "records_by_symbol",
    "records_from_stock_basic_frames",
    "refresh_pit_universe_from_tushare",
]
