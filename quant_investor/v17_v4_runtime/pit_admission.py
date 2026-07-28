"""No-write, exact-key admission for the V17 v4 PIT source closure."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import re
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract.canonical import canonical_bytes

from .security_directory import (
    SecurityDirectoryEntry,
    SourceAdmissionError,
    evaluate_membership,
)

REQUIRED_ROLES: Final = (
    "benchmark_total_return",
    "cn_open_day_calendar",
    "corporate_actions",
    "market_bars",
    "official_delisting_cash",
    "pit_fundamentals",
    "universe_membership",
)

ROLE_FIELDS: Final = {
    "benchmark_total_return": (
        "benchmark_id",
        "session",
        "total_return_index",
        "available_at",
    ),
    "cn_open_day_calendar": (
        "market_id",
        "session",
        "is_open",
        "available_at",
    ),
    "corporate_actions": (
        "security_code",
        "ex_date",
        "action_type",
        "announced_at",
        "revision_id",
        "cash_amount_per_share",
        "split_ratio",
        "currency",
        "official_source_id",
        "available_at",
    ),
    "market_bars": (
        "security_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_factor",
        "available_at",
    ),
    "official_delisting_cash": (
        "security_code",
        "terminal_session",
        "currency",
        "official_source_id",
        "cash_amount_per_share",
        "settlement_date",
        "available_at",
    ),
    "pit_fundamentals": (
        "security_code",
        "report_period",
        "announce_date",
        "revision_id",
        "field_id",
        "value",
        "unit",
        "available_at",
    ),
    "universe_membership": (
        "security_code",
        "name",
        "area",
        "industry",
        "board_market",
        "source_list_status",
        "valid_from",
        "valid_to",
        "published_at",
        "revision_id",
        "available_at",
        "source_id",
    ),
}

NATURAL_KEYS: Final = {
    "benchmark_total_return": ("benchmark_id", "session"),
    "cn_open_day_calendar": ("market_id", "session"),
    "corporate_actions": (
        "security_code",
        "ex_date",
        "action_type",
        "announced_at",
        "revision_id",
    ),
    "market_bars": ("security_code", "trade_date"),
    "official_delisting_cash": (
        "security_code",
        "terminal_session",
        "currency",
        "official_source_id",
    ),
    "pit_fundamentals": (
        "security_code",
        "report_period",
        "announce_date",
        "revision_id",
        "field_id",
    ),
    "universe_membership": (
        "security_code",
        "valid_from",
        "valid_to",
        "published_at",
        "revision_id",
    ),
}

_SECURITY_CODE_RE = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$", re.ASCII)
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$", re.ASCII)
_DECIMAL_RE = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
    re.ASCII,
)
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_ACTION_TYPES = frozenset(
    {
        "CASH_DIVIDEND",
        "RIGHTS_ISSUE",
        "STOCK_SPLIT",
        "TERMINAL_DELISTING",
    }
)


@dataclass(frozen=True)
class DatasetInput:
    role: str
    rows: Sequence[Mapping[str, Any]]
    expected_keys: Sequence[Sequence[str]]


@dataclass(frozen=True)
class AdmittedDataset:
    role: str
    row_count: int
    natural_key_fields: tuple[str, ...]
    observed_keys_sha256: str
    expected_keys_sha256: str
    row_set_sha256: str
    latest_available_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "row_count": self.row_count,
            "natural_key_fields": list(self.natural_key_fields),
            "observed_keys_sha256": self.observed_keys_sha256,
            "expected_keys_sha256": self.expected_keys_sha256,
            "row_set_sha256": self.row_set_sha256,
            "latest_available_at": self.latest_available_at,
        }


@dataclass(frozen=True)
class AdmittedPitClosure:
    history_start: str
    decision_session: str
    decision_cutoff: str
    datasets: tuple[AdmittedDataset, ...]
    closure_sha256: str

    def for_role(self, role: str) -> AdmittedDataset:
        for dataset in self.datasets:
            if dataset.role == role:
                return dataset
        _blocked()

    def as_dict(self) -> dict[str, Any]:
        return {
            "history_start": self.history_start,
            "decision_session": self.decision_session,
            "decision_cutoff": self.decision_cutoff,
            "datasets": [dataset.as_dict() for dataset in self.datasets],
            "closure_sha256": self.closure_sha256,
        }


def _blocked() -> NoReturn:
    raise SourceAdmissionError() from None


def _date(value: Any) -> str:
    if type(value) is not str:
        _blocked()
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _blocked()
    if parsed.isoformat() != value:
        _blocked()
    return value


def _instant(value: Any) -> str:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        _blocked()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _blocked()
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _blocked()
    return value


def _text(
    value: Any,
    *,
    required: bool = True,
    maximum: int = 256,
) -> str:
    if type(value) is not str:
        _blocked()
    if value.strip() != value or len(value.encode("utf-8")) > maximum:
        _blocked()
    if required and not value:
        _blocked()
    return value


def _identifier(value: Any) -> str:
    text = _text(value, maximum=128)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _blocked()
    return text


def _security_code(value: Any) -> str:
    text = _text(value, maximum=9)
    if _SECURITY_CODE_RE.fullmatch(text) is None:
        _blocked()
    return text


def _decimal(
    value: Any,
    *,
    allow_negative: bool = False,
    positive: bool = False,
) -> Decimal:
    if type(value) is not str or _DECIMAL_RE.fullmatch(value) is None:
        _blocked()
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        _blocked()
    if (
        not parsed.is_finite()
        or (positive and parsed <= 0)
        or (not positive and not allow_negative and parsed < 0)
    ):
        _blocked()
    return parsed


def _validate_common(
    row: Mapping[str, Any],
    *,
    role: str,
    decision_cutoff: str,
) -> None:
    if type(row) is not dict or set(row) != set(ROLE_FIELDS[role]):
        _blocked()
    available_at = _instant(row["available_at"])
    if available_at > decision_cutoff:
        _blocked()


def _validate_calendar(row: Mapping[str, Any]) -> None:
    _identifier(row["market_id"])
    _date(row["session"])
    if row["is_open"] is not True:
        _blocked()


def _validate_bar(row: Mapping[str, Any]) -> None:
    _security_code(row["security_code"])
    _date(row["trade_date"])
    open_price = _decimal(row["open"], positive=True)
    high = _decimal(row["high"], positive=True)
    low = _decimal(row["low"], positive=True)
    close = _decimal(row["close"], positive=True)
    _decimal(row["volume"])
    _decimal(row["amount"])
    _decimal(row["adj_factor"], positive=True)
    if row["available_at"][:10] < row["trade_date"]:
        _blocked()
    if high < max(open_price, low, close) or low > min(open_price, high, close):
        _blocked()


def _membership_entry(row: Mapping[str, Any]) -> SecurityDirectoryEntry:
    status = _text(row["source_list_status"], maximum=1)
    if status not in {"D", "L", "P"}:
        _blocked()
    valid_from = _date(row["valid_from"])
    valid_to = row["valid_to"]
    if valid_to:
        valid_to = _date(valid_to)
        if valid_to < valid_from:
            _blocked()
    elif status == "D":
        _blocked()
    published_at = _instant(row["published_at"])
    available_at = _instant(row["available_at"])
    if published_at > available_at:
        _blocked()
    return SecurityDirectoryEntry(
        security_code=_security_code(row["security_code"]),
        name=_text(row["name"]),
        area=_text(row["area"], required=False),
        industry=_text(row["industry"], required=False),
        board_market=_text(row["board_market"], required=False),
        source_list_status=status,
        valid_from=valid_from,
        valid_to=valid_to,
        published_at=published_at,
        revision_id=_identifier(row["revision_id"]),
        available_at=available_at,
        source_id=_identifier(row["source_id"]),
    )


def _validate_fundamental(row: Mapping[str, Any]) -> None:
    _security_code(row["security_code"])
    report_period = _date(row["report_period"])
    announce_date = _date(row["announce_date"])
    if report_period > announce_date:
        _blocked()
    _identifier(row["revision_id"])
    _identifier(row["field_id"])
    _decimal(row["value"], allow_negative=True)
    _text(row["unit"])
    if row["available_at"][:10] < announce_date:
        _blocked()


def _validate_action(row: Mapping[str, Any]) -> None:
    _security_code(row["security_code"])
    _date(row["ex_date"])
    action_type = _text(row["action_type"], maximum=32)
    if action_type not in _ACTION_TYPES:
        _blocked()
    announced_at = _instant(row["announced_at"])
    if announced_at > row["available_at"]:
        _blocked()
    _identifier(row["revision_id"])
    cash = _decimal(row["cash_amount_per_share"])
    split = _decimal(row["split_ratio"])
    currency = _text(row["currency"], maximum=3)
    if _CURRENCY_RE.fullmatch(currency) is None:
        _blocked()
    _identifier(row["official_source_id"])
    if action_type == "CASH_DIVIDEND" and (cash <= 0 or split != 0):
        _blocked()
    if action_type == "STOCK_SPLIT" and (split <= 0 or cash != 0):
        _blocked()


def _validate_benchmark(row: Mapping[str, Any]) -> None:
    _identifier(row["benchmark_id"])
    _date(row["session"])
    _decimal(row["total_return_index"], positive=True)
    if row["available_at"][:10] < row["session"]:
        _blocked()


def _validate_delisting_cash(row: Mapping[str, Any]) -> None:
    _security_code(row["security_code"])
    terminal = _date(row["terminal_session"])
    currency = _text(row["currency"], maximum=3)
    if _CURRENCY_RE.fullmatch(currency) is None:
        _blocked()
    _identifier(row["official_source_id"])
    _decimal(row["cash_amount_per_share"])
    if _date(row["settlement_date"]) < terminal:
        _blocked()


def _validate_role_row(
    row: Mapping[str, Any],
    *,
    role: str,
    decision_cutoff: str,
) -> None:
    _validate_common(row, role=role, decision_cutoff=decision_cutoff)
    validators = {
        "benchmark_total_return": _validate_benchmark,
        "cn_open_day_calendar": _validate_calendar,
        "corporate_actions": _validate_action,
        "market_bars": _validate_bar,
        "official_delisting_cash": _validate_delisting_cash,
        "pit_fundamentals": _validate_fundamental,
        "universe_membership": _membership_entry,
    }
    validators[role](row)


def _ordered_hash(values: Iterable[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for value in sorted(tuple(tuple(item) for item in values)):
        digest.update(canonical_bytes(list(value)))
        digest.update(b"\n")
    return digest.hexdigest()


def _row_set_hash(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[tuple[str, ...]],
) -> str:
    ordered = sorted(
        zip(keys, rows, strict=True),
        key=lambda item: item[0],
    )
    digest = hashlib.sha256()
    for _, row in ordered:
        digest.update(canonical_bytes(dict(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def _admit_dataset(
    source: DatasetInput,
    *,
    decision_cutoff: str,
) -> tuple[AdmittedDataset, tuple[Mapping[str, Any], ...]]:
    if source.role not in REQUIRED_ROLES:
        _blocked()
    rows = tuple(source.rows)
    natural_key_fields = NATURAL_KEYS[source.role]
    observed_keys: list[tuple[str, ...]] = []
    latest_available_at = ""
    for row in rows:
        _validate_role_row(
            row,
            role=source.role,
            decision_cutoff=decision_cutoff,
        )
        key = tuple(
            _text(row[field], required=False)
            for field in natural_key_fields
        )
        observed_keys.append(key)
        latest_available_at = max(latest_available_at, row["available_at"])
    if len(observed_keys) != len(set(observed_keys)):
        _blocked()
    expected_keys = tuple(tuple(key) for key in source.expected_keys)
    if any(
        len(key) != len(natural_key_fields)
        or any(type(value) is not str for value in key)
        for key in expected_keys
    ):
        _blocked()
    if len(expected_keys) != len(set(expected_keys)):
        _blocked()
    if set(observed_keys) != set(expected_keys):
        _blocked()
    observed_sha = _ordered_hash(observed_keys)
    expected_sha = _ordered_hash(expected_keys)
    if observed_sha != expected_sha:
        _blocked()
    return (
        AdmittedDataset(
            role=source.role,
            row_count=len(rows),
            natural_key_fields=natural_key_fields,
            observed_keys_sha256=observed_sha,
            expected_keys_sha256=expected_sha,
            row_set_sha256=_row_set_hash(rows, observed_keys),
            latest_available_at=latest_available_at,
        ),
        rows,
    )


def admit_pit_closure(
    sources: Sequence[DatasetInput],
    *,
    history_start: str,
    decision_session: str,
    decision_cutoff: str,
    delisting_label_start: str | None = None,
    delisting_label_end: str | None = None,
) -> AdmittedPitClosure:
    """Validate all seven roles and return only immutable admission summaries."""

    start = _date(history_start)
    end = _date(decision_session)
    cutoff = _instant(decision_cutoff)
    label_start = _date(delisting_label_start or start)
    label_end = _date(delisting_label_end or end)
    if start > end or label_start > label_end:
        _blocked()
    roles = tuple(source.role for source in sources)
    if roles != REQUIRED_ROLES:
        _blocked()
    admitted: list[AdmittedDataset] = []
    rows_by_role: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for source in sources:
        dataset, rows = _admit_dataset(source, decision_cutoff=cutoff)
        admitted.append(dataset)
        rows_by_role[source.role] = rows

    calendar_sessions = {
        row["session"] for row in rows_by_role["cn_open_day_calendar"]
    }
    if (
        start not in calendar_sessions
        or end not in calendar_sessions
        or any(session < start or session > end for session in calendar_sessions)
    ):
        _blocked()
    for role, field in (
        ("market_bars", "trade_date"),
        ("benchmark_total_return", "session"),
    ):
        sessions = {row[field] for row in rows_by_role[role]}
        if sessions != calendar_sessions:
            _blocked()

    membership_rows = rows_by_role["universe_membership"]
    entries = tuple(_membership_entry(row) for row in membership_rows)
    member_codes = {entry.security_code for entry in entries}
    for role in (
        "market_bars",
        "pit_fundamentals",
        "corporate_actions",
        "official_delisting_cash",
    ):
        if any(row["security_code"] not in member_codes for row in rows_by_role[role]):
            _blocked()
    for row in rows_by_role["market_bars"]:
        decision = evaluate_membership(
            entries,
            security_code=row["security_code"],
            session=row["trade_date"],
            decision_cutoff=cutoff,
        )
        if not decision.in_universe or not decision.research_eligible:
            _blocked()

    expected_terminal_keys = {
        (
            row["security_code"],
            row["ex_date"],
            row["currency"],
            row["official_source_id"],
        )
        for row in rows_by_role["corporate_actions"]
        if (
            row["action_type"] == "TERMINAL_DELISTING"
            and label_start <= row["ex_date"] <= label_end
        )
    }
    observed_terminal_keys = {
        tuple(row[field] for field in NATURAL_KEYS["official_delisting_cash"])
        for row in rows_by_role["official_delisting_cash"]
    }
    if observed_terminal_keys != expected_terminal_keys:
        _blocked()

    summary = {
        "history_start": start,
        "decision_session": end,
        "decision_cutoff": cutoff,
        "datasets": [dataset.as_dict() for dataset in admitted],
    }
    return AdmittedPitClosure(
        history_start=start,
        decision_session=end,
        decision_cutoff=cutoff,
        datasets=tuple(admitted),
        closure_sha256=hashlib.sha256(canonical_bytes(summary)).hexdigest(),
    )


__all__ = [
    "AdmittedDataset",
    "AdmittedPitClosure",
    "DatasetInput",
    "NATURAL_KEYS",
    "REQUIRED_ROLES",
    "ROLE_FIELDS",
    "admit_pit_closure",
]
