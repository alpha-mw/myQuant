"""Hash-bound web-notice evidence for an exact CN suspension date.

This is a separate evidence class from the Tushare ``suspend_d`` v5 cache.
It can only classify a missing bar when a captured notice explicitly binds the
symbol and a suspension window containing the target open day.  It never
creates an OHLCV row and never rewrites a Tushare cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_investor.market.cn_nontrading_evidence import (
    canonical_json_sha256,
    file_sha256,
    symbol_set_sha256,
)


OFFICIAL_SUSPENSION_EVIDENCE_SCHEMA_VERSION = (
    "cn-official-suspension-evidence.v1"
)
OFFICIAL_SUSPENSION_CLASSIFICATION = "verified_official_web_suspension_absent"


def _compact_date(value: Any) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _normalize_symbols(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(value or "").strip().upper()
            for value in values
            if str(value or "").strip()
        }
    )


def _valid_sha(value: Any) -> bool:
    digest = str(value or "").strip().lower()
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def official_suspension_evidence_path(
    output_root: str | Path,
    *,
    trade_date: str,
    pit_membership_sha256: str,
) -> Path:
    digest = str(pit_membership_sha256 or "").strip().lower()
    if not _valid_sha(digest):
        raise ValueError("official_suspension_pit_sha256_invalid")
    return (
        Path(output_root)
        / ".cache"
        / "official_suspension"
        / _compact_date(trade_date)
        / f"pit_{digest}"
        / "evidence.json"
    )


def _raw_contains(raw: bytes, fragment: str) -> bool:
    needle = str(fragment or "")
    if not needle:
        return False
    for encoding in ("utf-8", "gb18030", "gbk", "latin-1"):
        try:
            if needle in raw.decode(encoding, errors="strict"):
                return True
        except (UnicodeDecodeError, LookupError):
            continue
    return False


def build_official_web_suspension_evidence(
    *,
    output_root: str | Path,
    trade_date: str,
    symbols: Iterable[Any],
    pit_membership_path: str | Path,
    pit_membership_sha256: str,
    query_run_id: str,
    notices: Iterable[Mapping[str, Any]],
) -> Path:
    """Capture notice bytes and write a replayable, hash-bound receipt.

    ``notices`` is deliberately explicit: discovery and interpretation happen
    outside the canonical writer, while this function enforces the evidence
    contract before the receipt becomes reusable.
    """

    target_date = _compact_date(trade_date)
    expected_symbols = _normalize_symbols(symbols)
    pit_path = Path(pit_membership_path).expanduser()
    pit_sha = str(pit_membership_sha256 or "").strip().lower()
    run_id = str(query_run_id or "").strip()
    if not target_date or not expected_symbols:
        raise ValueError("official_suspension_scope_invalid")
    if not pit_path.exists() or not _valid_sha(pit_sha):
        raise ValueError("official_suspension_pit_binding_invalid")
    if file_sha256(pit_path) != pit_sha:
        raise ValueError("official_suspension_pit_sha256_mismatch")
    if not run_id:
        raise ValueError("official_suspension_query_run_id_required")

    evidence_path = official_suspension_evidence_path(
        output_root,
        trade_date=target_date,
        pit_membership_sha256=pit_sha,
    )
    raw_root = evidence_path.parent / "raw"
    normalized_notices: list[dict[str, Any]] = []
    for raw_spec in notices:
        spec = dict(raw_spec)
        symbol = str(spec.get("ts_code") or "").strip().upper()
        start_date = _compact_date(spec.get("suspension_start_date"))
        end_date_exclusive = _compact_date(spec.get("suspension_end_date_exclusive"))
        raw_source_path = Path(str(spec.get("raw_source_path") or "")).expanduser()
        raw = raw_source_path.read_bytes()
        raw_sha = hashlib.sha256(raw).hexdigest()
        if (
            symbol not in expected_symbols
            or not start_date
            or not end_date_exclusive
            or not raw
            or not start_date <= target_date < end_date_exclusive
        ):
            raise ValueError(f"official_suspension_notice_scope_invalid:{symbol}")
        required_fragments = [
            str(fragment)
            for fragment in list(spec.get("required_text_fragments") or [])
            if str(fragment)
        ]
        if not required_fragments or any(
            not _raw_contains(raw, fragment) for fragment in required_fragments
        ):
            raise ValueError(f"official_suspension_notice_text_not_verified:{symbol}")
        suffix = ".pdf" if raw_source_path.suffix.lower() == ".pdf" else ".html"
        durable_raw_path = raw_root / f"{raw_sha}{suffix}"
        if not durable_raw_path.exists():
            _atomic_write(durable_raw_path, raw)
        normalized_notices.append(
            {
                "ts_code": symbol,
                "notice_title": str(spec.get("notice_title") or "").strip(),
                "issuer_name": str(spec.get("issuer_name") or "").strip(),
                "issuer_host": str(spec.get("issuer_host") or "").strip().lower(),
                "source_class": str(spec.get("source_class") or "web_disclosure_mirror").strip(),
                "source_url": str(spec.get("source_url") or "").strip(),
                "linked_official_url": str(spec.get("linked_official_url") or "").strip(),
                "publication_date": _compact_date(spec.get("publication_date")),
                "suspension_start_date": start_date,
                "suspension_end_date_exclusive": end_date_exclusive,
                "target_date_covered": target_date,
                "required_text_fragments": required_fragments,
                "raw_capture_path": str(durable_raw_path),
                "raw_bytes_count": len(raw),
                "raw_bytes_sha256": raw_sha,
                "target_date_suspension_explicit": True,
            }
        )
    normalized_notices.sort(key=lambda item: item["ts_code"])
    notice_symbols = [item["ts_code"] for item in normalized_notices]
    if notice_symbols != expected_symbols:
        raise ValueError("official_suspension_notice_symbols_mismatch")
    payload: dict[str, Any] = {
        "schema_version": OFFICIAL_SUSPENSION_EVIDENCE_SCHEMA_VERSION,
        "classification": OFFICIAL_SUSPENSION_CLASSIFICATION,
        "source": "web_disclosure_notice",
        "trade_date": target_date,
        "query_run_id": run_id,
        "query_succeeded": True,
        "query_semantics": "explicit_notice_capture_for_exact_target_date",
        "queried_symbols": expected_symbols,
        "queried_symbols_sha256": symbol_set_sha256(expected_symbols),
        "matched_symbols": notice_symbols,
        "matched_symbols_sha256": symbol_set_sha256(notice_symbols),
        "unmatched_symbols": [],
        "pit_membership_path": str(pit_path),
        "pit_membership_sha256": pit_sha,
        "notices": normalized_notices,
        "notice_count": len(normalized_notices),
        "regulatory_exact_date_suspend_event_claimed": True,
        "writes_synthetic_bars": False,
        "generated_at": _utc_now_iso(),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    _atomic_write(
        evidence_path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"),
    )
    return evidence_path


def read_official_web_suspension_evidence(
    path: str | Path,
    *,
    trade_date: str,
    expected_pit_membership_path: str | Path,
    expected_pit_membership_sha256: str,
    expected_symbols: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Read and fully revalidate a captured notice bundle."""

    evidence_path = Path(path).expanduser()
    target_date = _compact_date(trade_date)
    blockers: list[str] = []
    payload: dict[str, Any] = {}
    if not evidence_path.exists():
        blockers.append("official_suspension_evidence_missing")
    else:
        try:
            loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
            else:
                blockers.append("official_suspension_evidence_invalid")
        except Exception as exc:
            blockers.append(f"official_suspension_evidence_unreadable:{exc}")
    declared_payload_sha = str(payload.get("payload_sha256") or "").lower()
    payload_without_sha = dict(payload)
    payload_without_sha.pop("payload_sha256", None)
    if not _valid_sha(declared_payload_sha) or declared_payload_sha != canonical_json_sha256(payload_without_sha):
        blockers.append("official_suspension_payload_sha256_mismatch")
    if payload.get("schema_version") != OFFICIAL_SUSPENSION_EVIDENCE_SCHEMA_VERSION:
        blockers.append("official_suspension_schema_version_mismatch")
    if payload.get("classification") != OFFICIAL_SUSPENSION_CLASSIFICATION:
        blockers.append("official_suspension_classification_mismatch")
    if _compact_date(payload.get("trade_date")) != target_date:
        blockers.append("official_suspension_trade_date_mismatch")
    expected_pit_path = str(Path(expected_pit_membership_path).expanduser())
    expected_pit_sha = str(expected_pit_membership_sha256 or "").lower()
    if str(payload.get("pit_membership_path") or "") != expected_pit_path:
        blockers.append("official_suspension_pit_path_mismatch")
    if str(payload.get("pit_membership_sha256") or "").lower() != expected_pit_sha:
        blockers.append("official_suspension_pit_sha256_mismatch")
    expected_set = set(_normalize_symbols(expected_symbols or []))
    queried = set(_normalize_symbols(payload.get("queried_symbols", []) or []))
    matched = set(_normalize_symbols(payload.get("matched_symbols", []) or []))
    if expected_set and not matched.issubset(expected_set):
        blockers.append("official_suspension_symbols_outside_expected_scope")
    if queried != matched:
        blockers.append("official_suspension_query_match_set_mismatch")
    if payload.get("unmatched_symbols") not in ([], None):
        blockers.append("official_suspension_unmatched_symbols_nonempty")
    if payload.get("regulatory_exact_date_suspend_event_claimed") is not True:
        blockers.append("official_suspension_exact_date_claim_missing")
    if payload.get("writes_synthetic_bars") is not False:
        blockers.append("official_suspension_synthetic_bar_contract_invalid")

    notice_symbols: set[str] = set()
    notices = payload.get("notices", []) or []
    if not isinstance(notices, list):
        blockers.append("official_suspension_notices_invalid")
        notices = []
    for notice in notices:
        if not isinstance(notice, Mapping):
            blockers.append("official_suspension_notice_invalid")
            continue
        item = dict(notice)
        symbol = str(item.get("ts_code") or "").strip().upper()
        notice_symbols.add(symbol)
        start_date = _compact_date(item.get("suspension_start_date"))
        end_date = _compact_date(item.get("suspension_end_date_exclusive"))
        if not symbol or not start_date <= target_date < end_date:
            blockers.append(f"official_suspension_notice_window_invalid:{symbol}")
        raw_path = Path(str(item.get("raw_capture_path") or "")).expanduser()
        raw_sha = str(item.get("raw_bytes_sha256") or "").lower()
        if not raw_path.exists():
            blockers.append(f"official_suspension_raw_missing:{symbol}")
            continue
        try:
            raw = raw_path.read_bytes()
        except Exception as exc:
            blockers.append(f"official_suspension_raw_unreadable:{symbol}:{exc}")
            continue
        if hashlib.sha256(raw).hexdigest() != raw_sha:
            blockers.append(f"official_suspension_raw_sha256_mismatch:{symbol}")
        if int(item.get("raw_bytes_count") or -1) != len(raw):
            blockers.append(f"official_suspension_raw_bytes_count_mismatch:{symbol}")
        for fragment in list(item.get("required_text_fragments") or []):
            if not _raw_contains(raw, str(fragment)):
                blockers.append(f"official_suspension_notice_text_missing:{symbol}")
    if notice_symbols != matched:
        blockers.append("official_suspension_notice_match_set_mismatch")
    if int(payload.get("notice_count") or -1) != len(notices):
        blockers.append("official_suspension_notice_count_mismatch")
    return {
        "status": "passed" if not blockers else "blocked",
        "verified_symbols": sorted(matched) if not blockers else [],
        "evidence_path": str(evidence_path),
        "evidence_sha256": file_sha256(evidence_path) if evidence_path.exists() else "",
        "payload_sha256": declared_payload_sha,
        "source": str(payload.get("source") or ""),
        "query_run_id": str(payload.get("query_run_id") or ""),
        "blockers": list(dict.fromkeys(blockers)),
    }
