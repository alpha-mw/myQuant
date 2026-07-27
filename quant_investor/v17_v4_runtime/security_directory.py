"""Strict current security-directory acquisition for V17 v4."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import re
from typing import Any, Final, NoReturn, Protocol

from quant_investor.v17_v4_contract.canonical import canonical_bytes

from .tushare_https import TushareResponse

STOCK_BASIC_FIELDS: Final = (
    "ts_code",
    "name",
    "area",
    "industry",
    "market",
    "list_date",
    "delist_date",
    "list_status",
)
LIST_STATUSES: Final = ("L", "D", "P")
SOURCE_ID: Final = "tushare.stock_basic"

_SECURITY_CODE_RE = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)
_COMPACT_DATE_RE = re.compile(r"^[0-9]{8}$", re.ASCII)
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)


class SourceAdmissionError(ValueError):
    """A source row or PIT lookup failed closed with one static code."""

    exit_code = 2

    def __init__(self, code: str = "SOURCE_ADMISSION_BLOCKED") -> None:
        self.code = code
        super().__init__(code)


class TushareClient(Protocol):
    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: tuple[str, ...],
    ) -> TushareResponse: ...


@dataclass(frozen=True)
class SecurityDirectoryEntry:
    security_code: str
    name: str
    area: str
    industry: str
    board_market: str
    source_list_status: str
    valid_from: str
    valid_to: str
    published_at: str
    revision_id: str
    available_at: str
    source_id: str = SOURCE_ID

    @property
    def natural_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.security_code,
            self.valid_from,
            self.valid_to,
            self.published_at,
            self.revision_id,
        )

    def as_row(self) -> dict[str, str]:
        return {
            "security_code": self.security_code,
            "name": self.name,
            "area": self.area,
            "industry": self.industry,
            "board_market": self.board_market,
            "source_list_status": self.source_list_status,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "published_at": self.published_at,
            "revision_id": self.revision_id,
            "available_at": self.available_at,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class MembershipDecision:
    security_code: str
    session: str
    in_universe: bool
    research_eligible: bool
    tradable: bool
    reason: str
    entry: SecurityDirectoryEntry


@dataclass(frozen=True)
class DirectoryExclusion:
    source_list_status: str
    reason: str
    source_row_sha256: str


@dataclass(frozen=True)
class SecurityDirectorySnapshot:
    observed_at: str
    entries: tuple[SecurityDirectoryEntry, ...]
    exclusions: tuple[DirectoryExclusion, ...]
    request_ids: tuple[str, ...]


def _blocked() -> NoReturn:
    raise SourceAdmissionError() from None


def _utc(value: Any) -> str:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        _blocked()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _blocked()
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _blocked()
    return value


def _iso_date(value: Any, *, required: bool) -> str:
    if value is None or value == "":
        if required:
            _blocked()
        return ""
    if type(value) is not str or _COMPACT_DATE_RE.fullmatch(value) is None:
        _blocked()
    try:
        parsed = date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError:
        _blocked()
    return parsed.isoformat()


def _text(value: Any, *, required: bool, maximum: int = 256) -> str:
    if value is None:
        if required:
            _blocked()
        return ""
    if type(value) is not str:
        _blocked()
    normalized = value.strip()
    if normalized != value or len(value.encode("utf-8")) > maximum:
        _blocked()
    if required and not value:
        _blocked()
    return value


def _entry_from_row(
    values: tuple[Any, ...],
    *,
    expected_status: str,
    observed_at: str,
) -> SecurityDirectoryEntry | DirectoryExclusion:
    if len(values) != len(STOCK_BASIC_FIELDS):
        _blocked()
    row = dict(zip(STOCK_BASIC_FIELDS, values, strict=True))
    status = _text(row["list_status"], required=True, maximum=1).upper()
    if status != expected_status:
        _blocked()
    valid_from = _iso_date(row["list_date"], required=True)
    valid_to = _iso_date(row["delist_date"], required=False)
    if valid_to and valid_to < valid_from:
        _blocked()
    if status == "D" and not valid_to:
        _blocked()
    name = _text(row["name"], required=True)
    area = _text(row["area"], required=False)
    industry = _text(row["industry"], required=False)
    board_market = _text(row["market"], required=False)
    raw_code = _text(row["ts_code"], required=True, maximum=32)
    security_code = raw_code.upper()
    if _SECURITY_CODE_RE.fullmatch(security_code) is None:
        return DirectoryExclusion(
            source_list_status=expected_status,
            reason="UNSUPPORTED_SECURITY_CODE",
            source_row_sha256=hashlib.sha256(
                canonical_bytes(list(values))
            ).hexdigest(),
        )
    revision_payload = {
        "security_code": security_code,
        "source_list_status": status,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "published_at": observed_at,
        "source_id": SOURCE_ID,
    }
    revision_id = "stock-basic-" + hashlib.sha256(
        canonical_bytes(revision_payload)
    ).hexdigest()
    return SecurityDirectoryEntry(
        security_code=security_code,
        name=name,
        area=area,
        industry=industry,
        board_market=board_market,
        source_list_status=status,
        valid_from=valid_from,
        valid_to=valid_to,
        published_at=observed_at,
        revision_id=revision_id,
        available_at=observed_at,
    )


def fetch_current_security_directory(
    client: TushareClient,
    *,
    observed_at: str,
) -> SecurityDirectorySnapshot:
    """Fetch L/D/P snapshots without writing or inferring historical availability."""

    cutoff = _utc(observed_at)
    entries: list[SecurityDirectoryEntry] = []
    exclusions: list[DirectoryExclusion] = []
    request_ids: list[str] = []
    seen_snapshot_keys: set[tuple[str, str]] = set()
    for status in LIST_STATUSES:
        response = client.request(
            api_name="stock_basic",
            params={"exchange": "", "list_status": status},
            expected_fields=STOCK_BASIC_FIELDS,
        )
        if response.api_name != "stock_basic" or response.has_more:
            _blocked()
        request_ids.append(response.request_id)
        for values in response.rows:
            parsed = _entry_from_row(
                values,
                expected_status=status,
                observed_at=cutoff,
            )
            if isinstance(parsed, DirectoryExclusion):
                exclusions.append(parsed)
                continue
            entry = parsed
            snapshot_key = (entry.security_code, entry.source_list_status)
            if snapshot_key in seen_snapshot_keys:
                _blocked()
            seen_snapshot_keys.add(snapshot_key)
            entries.append(entry)
    if not entries:
        _blocked()
    entries.sort(key=lambda entry: entry.natural_key)
    natural_keys = tuple(entry.natural_key for entry in entries)
    if len(natural_keys) != len(set(natural_keys)):
        _blocked()
    exclusions.sort(
        key=lambda exclusion: (
            exclusion.source_list_status,
            exclusion.reason,
            exclusion.source_row_sha256,
        )
    )
    if len(exclusions) != len(
        {
            (
                exclusion.source_list_status,
                exclusion.reason,
                exclusion.source_row_sha256,
            )
            for exclusion in exclusions
        }
    ):
        _blocked()
    return SecurityDirectorySnapshot(
        observed_at=cutoff,
        entries=tuple(entries),
        exclusions=tuple(exclusions),
        request_ids=tuple(request_ids),
    )


def _latest_entries(
    entries: Iterable[SecurityDirectoryEntry],
    *,
    security_code: str,
    decision_cutoff: str,
) -> tuple[SecurityDirectoryEntry, ...]:
    cutoff = _utc(decision_cutoff)
    code = security_code.upper()
    if _SECURITY_CODE_RE.fullmatch(code) is None:
        _blocked()
    candidates = tuple(
        entry
        for entry in entries
        if entry.security_code == code and entry.available_at <= cutoff
    )
    if not candidates:
        _blocked()
    latest_available_at = max(entry.available_at for entry in candidates)
    latest = tuple(
        entry
        for entry in candidates
        if entry.available_at == latest_available_at
    )
    identities = {
        (
            entry.source_list_status,
            entry.valid_from,
            entry.valid_to,
            entry.revision_id,
        )
        for entry in latest
    }
    if len(identities) != 1:
        _blocked()
    return latest


def evaluate_membership(
    entries: Iterable[SecurityDirectoryEntry],
    *,
    security_code: str,
    session: str,
    decision_cutoff: str,
) -> MembershipDecision:
    """Evaluate one security without current-listed or missing-row fallback."""

    try:
        session_date = date.fromisoformat(session)
    except (TypeError, ValueError):
        _blocked()
    latest = _latest_entries(
        entries,
        security_code=security_code,
        decision_cutoff=decision_cutoff,
    )
    entry = latest[0]
    if session_date < date.fromisoformat(entry.valid_from):
        return MembershipDecision(
            entry.security_code,
            session,
            False,
            False,
            False,
            "PRE_LISTING",
            entry,
        )
    if entry.valid_to and session_date >= date.fromisoformat(entry.valid_to):
        return MembershipDecision(
            entry.security_code,
            session,
            False,
            False,
            False,
            "DELISTED",
            entry,
        )
    if entry.source_list_status == "P":
        return MembershipDecision(
            entry.security_code,
            session,
            True,
            True,
            False,
            "PENDING",
            entry,
        )
    return MembershipDecision(
        entry.security_code,
        session,
        True,
        True,
        True,
        "LISTED",
        entry,
    )


__all__ = [
    "LIST_STATUSES",
    "DirectoryExclusion",
    "MembershipDecision",
    "SOURCE_ID",
    "STOCK_BASIC_FIELDS",
    "SecurityDirectoryEntry",
    "SecurityDirectorySnapshot",
    "SourceAdmissionError",
    "evaluate_membership",
    "fetch_current_security_directory",
]
