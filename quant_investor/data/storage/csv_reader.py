"""Local CSV reader utilities used by market data DAGs.

The helpers in this module are intentionally offline-only.  They centralize
lightweight diagnostics for resolver-backed market CSV reads without invoking
downloaders, providers, or model services.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_investor.agent_protocol import DataQualityIssue

_DATE_COLUMN_CANDIDATES = ("trade_date", "date")


@dataclass
class CSVReadResult:
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    path: str = ""
    symbol: str = ""
    category: str = ""
    universe_key: str = ""
    issues: list[DataQualityIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame"] = (
            self.frame.to_dict(orient="records")
            if isinstance(self.frame, pd.DataFrame)
            else []
        )
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


def _normalize_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if "." in text and text.replace(".", "", 1).isdigit():
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return ""


def _find_trade_date_column(columns: Iterable[Any]) -> str:
    by_lower = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in _DATE_COLUMN_CANDIDATES:
        column = by_lower.get(candidate)
        if column:
            return column
    return ""


def _make_issue(
    *,
    path: str | Path,
    symbol: str = "",
    category: str = "",
    universe_key: str = "",
    issue_type: str,
    severity: str = "warning",
    message: str,
    resolver_strategy: str = "",
    metadata: dict[str, Any] | None = None,
) -> DataQualityIssue:
    return DataQualityIssue(
        path=str(path),
        symbol=str(symbol or ""),
        category=str(category or ""),
        universe_key=str(universe_key or ""),
        issue_type=issue_type,
        severity=severity,
        message=message,
        resolver_strategy=str(resolver_strategy or ""),
        metadata=dict(metadata or {}),
    )


def peek_latest_date(path: str | Path, *, tail_bytes: int = 8192) -> str:
    """Return the last available date in a CSV without a full read."""
    csv_path = Path(path)
    if not csv_path.exists() or not csv_path.is_file():
        return ""
    try:
        with csv_path.open("rb") as handle:
            header_bytes = handle.readline()
            if not header_bytes:
                return ""
            header = (
                header_bytes.decode("utf-8-sig", errors="replace")
                .strip()
                .split(",")
            )
            date_column = _find_trade_date_column(header)
            if not date_column:
                return ""
            date_idx = [item.strip() for item in header].index(date_column)

            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max(1024, int(tail_bytes))))
            tail = handle.read().decode("utf-8-sig", errors="replace")
    except Exception:
        return ""

    tail_lines = [item.strip() for item in tail.splitlines() if item.strip()]
    for line in reversed(tail_lines):
        if line == ",".join(header):
            continue
        cells = line.split(",")
        if len(cells) <= date_idx:
            continue
        latest = _normalize_trade_date(cells[date_idx])
        if latest:
            return latest
    return ""


def _filter_by_trade_date(
    frame: pd.DataFrame,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    date_column = _find_trade_date_column(frame.columns)
    if frame.empty or not date_column:
        return frame
    normalized = frame[date_column].map(_normalize_trade_date)
    filtered = frame.copy()
    filtered["trade_date"] = normalized
    mask = normalized.str.len().eq(8)
    start = _normalize_trade_date(start_date)
    end = _normalize_trade_date(end_date)
    if start:
        mask &= normalized >= start
    if end:
        mask &= normalized <= end
    filtered = filtered.loc[mask].copy()
    if not filtered.empty:
        filtered = filtered.sort_values("trade_date").reset_index(drop=True)
    return filtered


def read_csv_with_diagnostics(
    path: str | Path,
    *,
    symbol: str = "",
    category: str = "",
    universe_key: str = "",
    resolver_strategy: str = "",
    start_date: str = "",
    end_date: str = "",
) -> CSVReadResult:
    csv_path = Path(path)
    metadata: dict[str, Any] = {
        "exists": csv_path.exists(),
        "start_date": _normalize_trade_date(start_date),
        "end_date": _normalize_trade_date(end_date),
    }
    issues: list[DataQualityIssue] = []

    if not csv_path.exists() or not csv_path.is_file():
        issues.append(
            _make_issue(
                path=csv_path,
                symbol=symbol,
                category=category,
                universe_key=universe_key,
                issue_type="missing_file",
                severity="error",
                message="CSV file is missing",
                resolver_strategy=resolver_strategy,
            )
        )
        return CSVReadResult(
            path=str(csv_path),
            symbol=symbol,
            category=category,
            universe_key=universe_key,
            issues=issues,
            metadata=metadata,
        )

    try:
        frame = pd.read_csv(csv_path, dtype={"trade_date": str})
    except Exception as exc:
        issues.append(
            _make_issue(
                path=csv_path,
                symbol=symbol,
                category=category,
                universe_key=universe_key,
                issue_type="read_error",
                severity="error",
                message=f"CSV read failed: {exc}",
                resolver_strategy=resolver_strategy,
            )
        )
        return CSVReadResult(
            path=str(csv_path),
            symbol=symbol,
            category=category,
            universe_key=universe_key,
            issues=issues,
            metadata=metadata,
        )

    metadata["row_count_raw"] = int(len(frame))
    if frame.empty:
        issues.append(
            _make_issue(
                path=csv_path,
                symbol=symbol,
                category=category,
                universe_key=universe_key,
                issue_type="empty_file",
                severity="warning",
                message="CSV file contains no rows",
                resolver_strategy=resolver_strategy,
            )
        )
    elif not _find_trade_date_column(frame.columns):
        issues.append(
            _make_issue(
                path=csv_path,
                symbol=symbol,
                category=category,
                universe_key=universe_key,
                issue_type="missing_trade_date",
                severity="error",
                message="CSV file has no trade_date column",
                resolver_strategy=resolver_strategy,
            )
        )
    else:
        frame = _filter_by_trade_date(
            frame,
            start_date=start_date,
            end_date=end_date,
        )

    metadata["row_count"] = int(len(frame))
    metadata["latest_trade_date"] = infer_latest_date_from_frames([frame])
    return CSVReadResult(
        frame=frame,
        path=str(csv_path),
        symbol=symbol,
        category=category,
        universe_key=universe_key,
        issues=issues,
        metadata=metadata,
    )


def infer_latest_date_from_frames(frames: Iterable[pd.DataFrame]) -> str:
    latest = ""
    for frame in frames:
        if frame is None or frame.empty:
            continue
        date_column = _find_trade_date_column(frame.columns)
        if not date_column:
            continue
        for value in frame[date_column].tail(256):
            normalized = _normalize_trade_date(value)
            if normalized and normalized > latest:
                latest = normalized
    return latest
