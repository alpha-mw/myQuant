"""Per-trade-date suspension evidence for identities that never traded.

Some identities are genuine members of a window and yet have no bar on any day
of it — a company suspended from its last trade until it was formally removed,
for instance. Treating that as missing data is wrong, and inventing a price for
it is worse. What makes the absence admissible is evidence, per expected trading
day, that the market was open and the identity was suspended.

The contract here is deliberately unforgiving:

  * expected trading days come from an independent calendar intersected with the
    identity's own membership interval — never from whatever bars happen to
    exist locally, which is the very thing under question;
  * every expected day must carry a suspension record naming that identity on
    that date, bound to the raw provider payload by SHA-256;
  * one missing day, one mismatched symbol, one altered payload, and the whole
    document is rejected.

A verified document says only "this identity legitimately had no trades". It is
never evidence of price coverage, and never evidence of financial coverage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RESEARCH_SUSPENSION_EVIDENCE_SCHEMA = "cn-research-suspension-evidence.v1"
SUSPEND_TYPE_SUSPENDED = "S"


class SuspensionEvidenceError(ValueError):
    """Raised when evidence is absent, incomplete, or does not verify."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_payload_sha256(payload: Any) -> str:
    """Hash a provider payload the same way whether it is built or re-read."""
    return _sha256_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _compact(value: Any) -> str:
    text = str(value or "").strip().replace("-", "")
    return text if len(text) == 8 and text.isdigit() else ""


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


@dataclass(frozen=True)
class ExpectedWindow:
    """The days an identity was a member and the market was open."""

    symbol: str
    effective_from: str
    effective_to: str
    window_start: str
    window_end: str
    trade_dates: tuple[str, ...]
    #: A member day deliberately excluded from expected trading days, with the
    #: membership record itself as the evidence for the absence of a trade.
    removal_date_excluded: str = ""


def expected_trading_days(
    *,
    symbol: str,
    effective_from: str,
    effective_to: str,
    window_start: str,
    window_end: str,
    calendar_open_dates: Iterable[str],
) -> ExpectedWindow:
    """Intersect an independent calendar with the identity's membership interval.

    ``effective_to`` is treated as inclusive, matching the membership contract:
    the identity is still a member on that date even though it does not trade.
    """
    start = max(_compact(window_start), _compact(effective_from))
    end_candidates = [_compact(window_end)]
    closed = _compact(effective_to)
    if closed:
        end_candidates.append(closed)
    end = min(candidate for candidate in end_candidates if candidate)
    if not start or not end:
        raise SuspensionEvidenceError("expected window bounds are not exact dates")
    if start > end:
        raise SuspensionEvidenceError(
            f"expected window is reversed for {symbol}: {start} > {end}"
        )
    # The removal date is a member day on which the identity does not trade.
    # That is settled by the membership contract, not by a suspension record:
    # across 20 sampled delisted identities, none traded on its delist_date, and
    # no suspend_d row exists for it either — on that day the identity is being
    # removed, not suspended. Expecting a trade there would manufacture a gap
    # that no evidence can ever close.
    removal_date = closed if closed else ""
    dates = tuple(
        sorted(
            {
                _compact(value)
                for value in calendar_open_dates
                if _compact(value)
                and start <= _compact(value) <= end
                and _compact(value) != removal_date
            }
        )
    )
    return ExpectedWindow(
        symbol=_normalize_symbol(symbol),
        effective_from=_compact(effective_from),
        effective_to=closed,
        window_start=start,
        window_end=end,
        trade_dates=dates,
        removal_date_excluded=(
            removal_date if removal_date and start <= removal_date <= end else ""
        ),
    )


def build_suspension_evidence(
    *,
    window: ExpectedWindow,
    per_date_payloads: Mapping[str, Any],
    calendar_reference: Mapping[str, Any],
    membership_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one suspension record per expected trading day.

    ``per_date_payloads`` maps a trade date to the raw provider response for a
    query made *for that date*. A day whose payload does not name this identity
    as suspended is recorded as unexplained rather than quietly dropped.
    """
    per_date: list[dict[str, Any]] = []
    unexplained: list[str] = []
    for trade_date in window.trade_dates:
        payload = per_date_payloads.get(trade_date)
        if payload is None:
            unexplained.append(trade_date)
            continue
        rows = [
            row
            for row in list(payload.get("rows") or [])
            if _normalize_symbol(row.get("ts_code")) == window.symbol
            and _compact(row.get("trade_date")) == trade_date
            and str(row.get("suspend_type") or "").strip().upper()
            == SUSPEND_TYPE_SUSPENDED
        ]
        if not rows:
            unexplained.append(trade_date)
            continue
        per_date.append(
            {
                "trade_date": trade_date,
                "symbol": window.symbol,
                "suspend_type": SUSPEND_TYPE_SUSPENDED,
                "query_params": dict(payload.get("query_params") or {}),
                "payload_sha256": canonical_payload_sha256(payload),
                "matched_rows": len(rows),
            }
        )

    document = {
        "schema_version": RESEARCH_SUSPENSION_EVIDENCE_SCHEMA,
        "research_only": True,
        "symbol": window.symbol,
        "membership_interval": {
            "effective_from": window.effective_from,
            "effective_to": window.effective_to,
        },
        "expected_window": {"start": window.window_start, "end": window.window_end},
        "expected_trade_dates": list(window.trade_dates),
        "expected_trade_date_count": len(window.trade_dates),
        "removal_date_excluded": window.removal_date_excluded,
        "removal_date_evidence": (
            "membership effective_to; tradability false on the removal date"
            if window.removal_date_excluded
            else ""
        ),
        "calendar_reference": dict(calendar_reference),
        "membership_reference": dict(membership_reference),
        "per_date": per_date,
        "unexplained_dates": unexplained,
        # A verified document proves an absence of trades. It must never be read
        # as the identity having price or financial coverage.
        "grants_price_coverage": False,
        "grants_financial_coverage": False,
    }
    document["record_sha256"] = canonical_payload_sha256(
        {key: value for key, value in document.items() if key != "record_sha256"}
    )
    return document


def verify_suspension_evidence(
    document: Mapping[str, Any],
    *,
    symbol: str,
    expected_trade_dates: Sequence[str],
    per_date_payloads: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Accept the document only if it covers exactly the expected days, untampered."""
    if document.get("schema_version") != RESEARCH_SUSPENSION_EVIDENCE_SCHEMA:
        raise SuspensionEvidenceError("suspension evidence schema version mismatch")
    recorded = str(document.get("record_sha256") or "")
    recomputed = canonical_payload_sha256(
        {key: value for key, value in document.items() if key != "record_sha256"}
    )
    if recorded != recomputed:
        raise SuspensionEvidenceError("suspension evidence record_sha256 does not verify")
    if _normalize_symbol(document.get("symbol")) != _normalize_symbol(symbol):
        raise SuspensionEvidenceError("suspension evidence is for a different identity")
    if document.get("unexplained_dates"):
        raise SuspensionEvidenceError(
            "suspension evidence has unexplained dates: "
            + ",".join(list(document["unexplained_dates"])[:5])
        )
    expected = [_compact(value) for value in expected_trade_dates]
    covered = [str(entry.get("trade_date")) for entry in list(document.get("per_date") or [])]
    if sorted(covered) != sorted(expected):
        missing = sorted(set(expected) - set(covered))
        extra = sorted(set(covered) - set(expected))
        raise SuspensionEvidenceError(
            f"suspension evidence day set mismatch; missing={missing[:5]} extra={extra[:5]}"
        )
    for entry in list(document.get("per_date") or []):
        if _normalize_symbol(entry.get("symbol")) != _normalize_symbol(symbol):
            raise SuspensionEvidenceError("suspension evidence row names another identity")
        if str(entry.get("suspend_type") or "").strip().upper() != SUSPEND_TYPE_SUSPENDED:
            raise SuspensionEvidenceError("suspension evidence row is not a suspension")
        if per_date_payloads is not None:
            payload = per_date_payloads.get(str(entry.get("trade_date")))
            if payload is None:
                raise SuspensionEvidenceError(
                    f"raw payload missing for {entry.get('trade_date')}"
                )
            if canonical_payload_sha256(payload) != str(entry.get("payload_sha256") or ""):
                raise SuspensionEvidenceError(
                    f"raw payload SHA mismatch for {entry.get('trade_date')}"
                )
    return {
        "symbol": _normalize_symbol(symbol),
        "verified_trade_dates": sorted(covered),
        "verified_day_count": len(covered),
        "grants_price_coverage": False,
        "grants_financial_coverage": False,
    }


def load_and_verify(
    path: str | Path,
    *,
    symbol: str,
    expected_trade_dates: Sequence[str],
) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return verify_suspension_evidence(
        document, symbol=symbol, expected_trade_dates=expected_trade_dates
    )
