"""Build the strict-Parquet-derived CN open-session calendar for Factor v4.

`governance_protocol_v4.validate_open_session_calendar_v4` has always been able
to *check* this artifact, and `assess_candidate_maturity` requires one before it
will grant either maturity route. Nothing ever built one from real data: the
only `latest_pointer_sha256` values in the tree are test digests. Factor v4
therefore had a validator with no producer, which is one reason no factor record
has ever reached the readiness assessment.

The calendar is derived, not declared. Sessions come from the trade dates
actually observed in the active strict Parquet snapshot, and the payload binds
the pointer and manifest bytes so a later reader can prove which snapshot the
calendar was cut from.
"""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from quant_investor.factors.governance_protocol_v4 import (
    OPEN_SESSION_CALENDAR_SCHEMA_VERSION,
    OPEN_SESSION_CALENDAR_SOURCE,
    validate_open_session_calendar_v4,
)
from quant_investor.market.market_data_store import MarketDataStore


class OpenSessionCalendarError(ValueError):
    """Raised when the active snapshot cannot yield a valid v4 calendar."""


@dataclass(frozen=True)
class OpenSessionCalendar:
    """One immutable calendar plus the provenance needed to re-derive it."""

    payload: dict[str, Any]
    calendar_sha256: str
    snapshot_id: str
    excluded_sessions: tuple[str, ...]

    @property
    def open_session_dates(self) -> list[str]:
        return list(self.payload["open_session_dates"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(
    manifest_rel: str, *, data_root: Path, market_dir: Path
) -> Path:
    """Resolve `manifest_path` under either pointer convention.

    Published pointers record it repo-relative (`data/parquet/cn/_snapshots/x.json`),
    but the field is also written market-relative (`_snapshots/x.json`). Try the
    documented forms in order rather than guessing with `Path.name`, which would
    silently drop the `_snapshots/` component.
    """

    raw = Path(manifest_rel)
    candidates = [raw] if raw.is_absolute() else []
    parts = raw.parts
    # Repo-relative form: re-root everything below `parquet/<market>/` onto the
    # caller's market directory, so a relocated data_root still resolves.
    if market_dir.name in parts:
        tail = parts[parts.index(market_dir.name) + 1 :]
        if tail:
            candidates.append(market_dir.joinpath(*tail))
    # Leading data-root component, by literal name or by the caller's own root.
    for head in {"data", data_root.name}:
        if parts and parts[0] == head and len(parts) > 1:
            candidates.append(data_root.joinpath(*parts[1:]))
    candidates.extend([market_dir / raw, data_root / raw, Path.cwd() / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise OpenSessionCalendarError(f"missing snapshot manifest: {manifest_rel}")


def _iso_session(raw: Any) -> str | None:
    """Normalize one observed trade_date to ISO, or None when unusable."""

    text = str(raw or "").strip()[:10].replace("-", "")
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[0:4]), int(text[4:6]), int(text[6:8])).isoformat()
    except ValueError:
        return None


def build_open_session_calendar_v4(
    *,
    market: str = "CN",
    data_root: str | Path = "data",
) -> OpenSessionCalendar:
    """Cut a v4 open-session calendar from the active strict Parquet snapshot."""

    store = MarketDataStore(market=market, data_root=data_root)
    root = Path(data_root)
    pointer_path = root / "parquet" / market.lower() / "_latest.json"
    if not pointer_path.exists():
        raise OpenSessionCalendarError(f"missing strict Parquet pointer: {pointer_path}")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenSessionCalendarError("strict Parquet pointer is not valid JSON") from exc

    snapshot_id = str(pointer.get("snapshot_id") or "").strip()
    manifest_rel = str(pointer.get("manifest_path") or "").strip()
    if not snapshot_id or not manifest_rel:
        raise OpenSessionCalendarError("pointer must declare snapshot_id and manifest_path")
    market_dir = root / "parquet" / market.lower()
    manifest_path = _resolve_manifest_path(
        manifest_rel, data_root=root, market_dir=market_dir
    )

    table_root = market_dir / "_snapshots" / snapshot_id / "table" / "bars"
    if not table_root.exists():
        raise OpenSessionCalendarError(f"missing snapshot table root: {table_root}")

    observed: set[str] = set()
    unparsable = 0
    for part in glob.glob(str(table_root / "**" / "*.parquet"), recursive=True):
        column = pq.read_table(part, columns=["trade_date"])["trade_date"]
        for raw in pc.unique(column).to_pylist():
            session = _iso_session(raw)
            if session is None:
                unparsable += 1
            else:
                observed.add(session)
    if unparsable:
        raise OpenSessionCalendarError(
            f"{unparsable} observed trade_date value(s) are not calendar dates"
        )
    if not observed:
        raise OpenSessionCalendarError("snapshot yielded no observed trade dates")

    # The same corruption class C1 guards against on write is still present in
    # already published snapshots, so a calendar cut today must exclude it
    # rather than inherit a 1970 session.
    floor = store.first_plausible_session()
    floor_iso = f"{floor[0:4]}-{floor[4:6]}-{floor[6:8]}"
    excluded = tuple(sorted(item for item in observed if item < floor_iso))
    sessions = sorted(item for item in observed if item >= floor_iso)
    weekend = [item for item in sessions if date.fromisoformat(item).weekday() >= 5]
    if weekend:
        raise OpenSessionCalendarError(
            f"snapshot reports {len(weekend)} weekend session(s), "
            f"which the v4 calendar contract forbids: {weekend[:5]}"
        )
    if not sessions:
        raise OpenSessionCalendarError("no plausible sessions remain after exclusion")

    payload = {
        "schema_version": OPEN_SESSION_CALENDAR_SCHEMA_VERSION,
        "market": market.upper(),
        "source": OPEN_SESSION_CALENDAR_SOURCE,
        "latest_pointer_sha256": _file_sha256(pointer_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "open_session_dates": sessions,
    }
    validated = validate_open_session_calendar_v4(payload)
    return OpenSessionCalendar(
        payload=payload,
        calendar_sha256=validated["calendar_sha256"],
        snapshot_id=snapshot_id,
        excluded_sessions=excluded,
    )


def month_end_sessions(calendar: OpenSessionCalendar) -> list[str]:
    """Return the last open session of each calendar month, ascending.

    v4 maturity counts *actual* month-end sessions, so this is derived from the
    calendar rather than from month arithmetic.
    """

    last_by_month: dict[str, str] = {}
    for session in calendar.open_session_dates:
        last_by_month[session[:7]] = session
    return [last_by_month[month] for month in sorted(last_by_month)]


def nonoverlapping_cohorts(
    calendar: OpenSessionCalendar, *, size: int = 30
) -> list[dict[str, Any]]:
    """Slice the calendar into consecutive non-overlapping session cohorts.

    Cohorts are cut from the most recent session backwards so the newest
    evidence is always a whole cohort, matching how forward evaluation accrues.
    """

    if size < 1:
        raise ValueError("size must be at least 1")
    sessions = calendar.open_session_dates
    cohorts: list[dict[str, Any]] = []
    end = len(sessions)
    while end - size >= 0:
        window = sessions[end - size : end]
        cohorts.append(
            {
                "cohort_id": f"cohort-{window[0]}-{window[-1]}",
                "start": window[0],
                "end": window[-1],
                "horizon_days": size,
                "calendar_sha256": calendar.calendar_sha256,
                "open_session_dates": window,
            }
        )
        end -= size
    cohorts.reverse()
    return cohorts
