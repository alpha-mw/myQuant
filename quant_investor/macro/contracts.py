"""Stable wire contracts for point-in-time macro observations and snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_FREQUENCIES = frozenset({"daily", "weekly", "monthly", "quarterly", "annual"})
SUPPORTED_QUALITY_STATUSES = frozenset({"pass", "fail", "degraded", "blocked"})
OFFICIAL_SOURCE_SYSTEMS = frozenset(
    {
        "nbs_official",
        "pbc_official",
        "pboc_official",
        "customs_official",
        "mof_official",
        "ndrc_official",
        "stats.gov.cn",
        "pbc.gov.cn",
        "customs.gov.cn",
        "mof.gov.cn",
        "ndrc.gov.cn",
    }
)
TUSHARE_SOURCE_SYSTEMS = frozenset({"tushare", "tushare_fallback", "tushare_primary"})
SUPPORTED_SOURCE_SYSTEMS = OFFICIAL_SOURCE_SYSTEMS | TUSHARE_SOURCE_SYSTEMS
_SOURCE_HOSTS = {
    "nbs_official": ("stats.gov.cn",),
    "stats.gov.cn": ("stats.gov.cn",),
    "pbc_official": ("pbc.gov.cn",),
    "pboc_official": ("pbc.gov.cn",),
    "pbc.gov.cn": ("pbc.gov.cn",),
    "customs_official": ("customs.gov.cn",),
    "customs.gov.cn": ("customs.gov.cn",),
    "mof_official": ("mof.gov.cn",),
    "mof.gov.cn": ("mof.gov.cn",),
    "ndrc_official": ("ndrc.gov.cn",),
    "ndrc.gov.cn": ("ndrc.gov.cn",),
    "tushare": ("tushare.pro",),
    "tushare_fallback": ("tushare.pro",),
    "tushare_primary": ("tushare.pro",),
}
_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "password",
        "secret",
        "token",
    }
)


def is_official_source(value: Any) -> bool:
    return str(value or "").strip().lower() in OFFICIAL_SOURCE_SYSTEMS


def is_tushare_source(value: Any) -> bool:
    return str(value or "").strip().lower() in TUSHARE_SOURCE_SYSTEMS


def normalize_source_url(value: Any, *, source_system: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("source_url_missing")
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("source_url_invalid") from exc
    if parsed.scheme.lower() != "https":
        raise ValueError("source_url_https_required")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source_url_userinfo_rejected")
    if port not in {None, 443}:
        raise ValueError("source_url_port_rejected")
    hostname = str(parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError("source_url_host_missing")
    system = str(source_system or "").strip().lower()
    allowed_hosts = _SOURCE_HOSTS.get(system, ())
    if not any(
        hostname == allowed or hostname.endswith(f".{allowed}")
        for allowed in allowed_hosts
    ):
        raise ValueError("source_url_issuer_mismatch")
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key.strip().lower() in _SENSITIVE_QUERY_KEYS for key, _ in query_items):
        raise ValueError("source_url_sensitive_query_rejected")
    if parsed.fragment:
        raise ValueError("source_url_fragment_rejected")
    normalized_query = urlencode(sorted(query_items), doseq=True)
    path = parsed.path or "/"
    return urlunsplit(("https", hostname, path, normalized_query, ""))


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def parse_timestamp(value: Any, *, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name}_missing")
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name}_invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name}_timezone_required")
    return parsed.astimezone(UTC)


def published_cutoff(value: Any) -> datetime:
    """Map date-only as-of values to the CN cash-market close cutoff."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("as_of_timezone_required")
        return value.astimezone(UTC)
    text = str(value or "").strip()
    if not text:
        raise ValueError("as_of_missing")
    if "T" in text or ":" in text:
        return parse_timestamp(text, field_name="as_of")
    try:
        parsed_date = datetime.strptime(text, "%Y%m%d").date() if text.isdigit() and len(text) == 8 else date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("as_of_invalid") from exc
    return datetime.combine(parsed_date, time(15, 0), tzinfo=SHANGHAI).astimezone(UTC)


@dataclass(frozen=True)
class MacroObservation:
    indicator_id: str
    dimension_type: str
    period_end: str
    release_at: str
    available_at: str
    vintage_id: str
    value: float
    unit: str
    frequency: str
    source_system: str
    fetched_at: str
    content_hash: str
    industry_chain: str = ""
    source_record_id: str = ""
    source_url: str = ""
    quality_status: str = "pass"

    @staticmethod
    def semantic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return the stable PIT identity; fetch time is deliberately excluded."""

        available_at = parse_timestamp(payload.get("available_at"), field_name="available_at")
        release_at = parse_timestamp(payload.get("release_at"), field_name="release_at")
        period_text = str(payload.get("period_end") or "").strip()
        period_end = (
            datetime.strptime(period_text, "%Y%m%d").date().isoformat()
            if period_text.isdigit() and len(period_text) == 8
            else date.fromisoformat(period_text).isoformat()
        )
        source_system = str(payload.get("source_system") or "").strip().lower()
        source_url = normalize_source_url(
            payload.get("source_url"),
            source_system=source_system,
        )
        return {
            "indicator_id": str(payload.get("indicator_id") or "").strip(),
            "dimension_type": str(payload.get("dimension_type") or "national").strip().lower(),
            "industry_chain": str(payload.get("industry_chain") or "").strip(),
            "period_end": period_end,
            "release_at": release_at.isoformat(),
            "available_at": available_at.isoformat(),
            "vintage_id": str(payload.get("vintage_id") or "initial").strip(),
            "value": float(payload.get("value")),
            "unit": str(payload.get("unit") or "").strip(),
            "frequency": str(payload.get("frequency") or "monthly").strip().lower(),
            "source_system": source_system,
            "source_record_id": str(payload.get("source_record_id") or "").strip(),
            "source_url": source_url,
            "quality_status": str(payload.get("quality_status") or "pass").strip().lower(),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MacroObservation":
        indicator_id = str(payload.get("indicator_id") or "").strip()
        if not indicator_id:
            raise ValueError("indicator_id_missing")
        dimension_type = str(payload.get("dimension_type") or "national").strip().lower()
        if dimension_type not in {"national", "industry", "market_confirmation"}:
            raise ValueError("dimension_type_invalid")
        period_end = str(payload.get("period_end") or "").strip()
        try:
            normalized_period_end = (
                datetime.strptime(period_end, "%Y%m%d").date().isoformat()
                if period_end.isdigit() and len(period_end) == 8
                else date.fromisoformat(period_end).isoformat()
            )
        except ValueError as exc:
            raise ValueError("period_end_invalid") from exc
        value = float(payload.get("value"))
        if not math.isfinite(value):
            raise ValueError("observation_value_non_finite")
        available_at = parse_timestamp(payload.get("available_at"), field_name="available_at")
        release_at = parse_timestamp(payload.get("release_at"), field_name="release_at")
        fetched_at = parse_timestamp(payload.get("fetched_at"), field_name="fetched_at")
        if available_at < release_at:
            raise ValueError("available_at_before_release_at")
        if fetched_at < available_at:
            raise ValueError("fetched_at_before_available_at")
        vintage_id = str(payload.get("vintage_id") or "initial").strip()
        unit = str(payload.get("unit") or "").strip()
        frequency = str(payload.get("frequency") or "monthly").strip().lower()
        source_system = str(payload.get("source_system") or "").strip().lower()
        quality_status = str(payload.get("quality_status") or "pass").strip().lower()
        if not vintage_id:
            raise ValueError("vintage_id_missing")
        if not unit:
            raise ValueError("unit_missing")
        if frequency not in SUPPORTED_FREQUENCIES:
            raise ValueError("frequency_unsupported")
        if not source_system:
            raise ValueError("source_system_missing")
        if source_system not in SUPPORTED_SOURCE_SYSTEMS:
            raise ValueError("source_system_unsupported")
        if quality_status not in SUPPORTED_QUALITY_STATUSES:
            raise ValueError("quality_status_unsupported")
        industry_chain = str(payload.get("industry_chain") or "").strip()
        parts = indicator_id.split(".")
        if dimension_type == "industry":
            if len(parts) != 3 or parts[0] != "industry" or not industry_chain or parts[1] != industry_chain:
                raise ValueError("industry_chain_indicator_mismatch")
        elif industry_chain:
            raise ValueError("industry_chain_for_non_industry_observation")
        from quant_investor.macro.registry import definition_for

        definition = definition_for(indicator_id, frequency)
        if definition is None:
            raise ValueError("indicator_id_unregistered")
        if not indicator_id.startswith("industry.") and definition.frequency != frequency:
            raise ValueError("indicator_frequency_mismatch")
        if definition.unit and definition.unit != unit:
            raise ValueError("indicator_unit_mismatch")
        source_record_id = str(payload.get("source_record_id") or "").strip()
        if not source_record_id:
            raise ValueError("source_record_id_missing")
        if any(ord(character) < 32 for character in source_record_id):
            raise ValueError("source_record_id_invalid")
        source_url = normalize_source_url(
            payload.get("source_url"),
            source_system=source_system,
        )
        content_hash = str(payload.get("content_hash") or "").strip()
        if content_hash and not _SHA256_RE.fullmatch(content_hash.lower()):
            raise ValueError("content_hash_invalid")
        stable = cls.semantic_payload(payload)
        expected_content_hash = canonical_hash(stable)
        if content_hash and content_hash.lower() != expected_content_hash:
            raise ValueError("content_hash_mismatch")
        content_hash = expected_content_hash
        return cls(
            indicator_id=indicator_id,
            dimension_type=dimension_type,
            industry_chain=industry_chain,
            period_end=normalized_period_end,
            release_at=release_at.isoformat(),
            available_at=available_at.isoformat(),
            vintage_id=vintage_id,
            value=value,
            unit=unit,
            frequency=frequency,
            source_system=source_system,
            source_record_id=source_record_id,
            source_url=source_url,
            fetched_at=fetched_at.isoformat(),
            content_hash=content_hash,
            quality_status=quality_status,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MacroSnapshot:
    market: str
    as_of: str
    published_cutoff: str
    registry_version: str
    score_model_version: str
    selected_observation_hashes: tuple[str, ...]
    national_states: Mapping[str, float]
    industry_chain_states: Mapping[str, float]
    market_confirmation: Mapping[str, float]
    coverage: Mapping[str, Any]
    freshness: Mapping[str, Any]
    source_lineage: Mapping[str, Any]
    readiness_status: str
    blockers: tuple[str, ...]
    macro_score: float
    confidence: float
    shadow_overlays: Mapping[str, Any] = field(default_factory=dict)
    snapshot_hash: str = ""

    def hash_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("snapshot_hash", None)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "macro-snapshot.v2",
            "market": self.market,
            "as_of": self.as_of,
            "published_cutoff": self.published_cutoff,
            "registry_version": self.registry_version,
            "score_model_version": self.score_model_version,
            "selected_observation_hashes": list(self.selected_observation_hashes),
            "national_states": dict(self.national_states),
            "industry_chain_states": dict(self.industry_chain_states),
            "market_confirmation": dict(self.market_confirmation),
            "coverage": dict(self.coverage),
            "freshness": dict(self.freshness),
            "source_lineage": dict(self.source_lineage),
            "readiness_status": self.readiness_status,
            "blockers": list(self.blockers),
            "macro_score": self.macro_score,
            "confidence": self.confidence,
            "shadow_overlays": dict(self.shadow_overlays),
            "snapshot_hash": self.snapshot_hash,
        }
