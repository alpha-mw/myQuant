from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.themes.types import clamp


_DATE_FORMATS = ("%Y-%m-%d", "%Y%m%d")
_CENTRAL_KEYWORDS = (
    "central",
    "state council",
    "national",
    "ministry",
    "ministerial",
    "国务院",
    "中央",
    "国家",
    "部委",
    "发展改革委",
    "发改委",
    "工信部",
    "财政部",
    "科技部",
)
_LOCAL_KEYWORDS = (
    "local",
    "provincial",
    "province",
    "municipal",
    "city",
    "地方",
    "省",
    "市",
)
_ASSOCIATION_KEYWORDS = (
    "association",
    "industry association",
    "协会",
    "联盟",
    "学会",
)
_PILOT_KEYWORDS = ("pilot", "试点", "示范")
_STANDARD_KEYWORDS = ("standard", "标准", "规范")
_PROCUREMENT_KEYWORDS = ("procurement", "采购", "招标")
_FUNDING_KEYWORDS = ("funding", "fund", "subsidy", "grant", "资金", "基金", "补贴", "专项")


@dataclass(frozen=True)
class PolicyEvent:
    event_id: str = ""
    title: str = ""
    issuer: str = ""
    publish_date: str = ""
    effective_date: str = ""
    policy_level: str = ""
    policy_type: str = ""
    theme_tags: list[str] = field(default_factory=list)
    industry_tags: list[str] = field(default_factory=list)
    symbol_tags: list[str] = field(default_factory=list)
    evidence_text: str = ""
    source_url: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyEvent":
        return cls(
            event_id=_text(value.get("event_id")),
            title=_text(value.get("title")),
            issuer=_text(value.get("issuer")),
            publish_date=_text(value.get("publish_date")),
            effective_date=_text(value.get("effective_date")),
            policy_level=_text(value.get("policy_level")),
            policy_type=_text(value.get("policy_type")),
            theme_tags=_text_list(value.get("theme_tags")),
            industry_tags=_text_list(value.get("industry_tags")),
            symbol_tags=_text_list(value.get("symbol_tags")),
            evidence_text=_text(value.get("evidence_text")),
            source_url=_text(value.get("source_url")),
        )

    def event_date(self) -> date | None:
        return _parse_date(self.effective_date) or _parse_date(self.publish_date)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "issuer": self.issuer,
            "publish_date": self.publish_date,
            "effective_date": self.effective_date,
            "policy_level": self.policy_level,
            "policy_type": self.policy_type,
            "theme_tags": list(self.theme_tags),
            "industry_tags": list(self.industry_tags),
            "symbol_tags": list(self.symbol_tags),
            "evidence_text": self.evidence_text,
            "source_url": self.source_url,
        }


@dataclass
class PolicyCatalystScore:
    policy_score: float = 0.0
    confidence: float = 0.0
    recency_score: float = 0.0
    authority_score: float = 0.0
    specificity_score: float = 0.0
    implementation_score: float = 0.0
    funding_score: float = 0.0
    beneficiary_clarity: float = 0.0
    evidence: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    policy_stage: str = "no_match"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_score": _finite(self.policy_score),
            "confidence": _finite(self.confidence),
            "recency_score": _finite(self.recency_score),
            "authority_score": _finite(self.authority_score),
            "specificity_score": _finite(self.specificity_score),
            "implementation_score": _finite(self.implementation_score),
            "funding_score": _finite(self.funding_score),
            "beneficiary_clarity": _finite(self.beneficiary_clarity),
            "evidence": list(self.evidence),
            "risk_flags": list(self.risk_flags),
            "policy_stage": str(self.policy_stage or "no_match"),
        }


class PolicyCatalystScanner:
    def __init__(self, *, event_path: str | Path, lookback_days: int = 30) -> None:
        self.event_path = Path(event_path)
        self.lookback_days = max(int(lookback_days or 30), 1)
        self.status = "unavailable"
        self.diagnostic_notes: list[str] = []

    def load_events(self) -> list[PolicyEvent]:
        self.status = "unavailable"
        self.diagnostic_notes = []
        if not self.event_path.exists():
            self.diagnostic_notes.append("policy_event_file_missing")
            return []

        events: list[PolicyEvent] = []
        try:
            text = self.event_path.read_text(encoding="utf-8")
        except OSError as exc:
            self.diagnostic_notes.append(f"policy_event_file_read_error: {exc}")
            return []

        try:
            for line_number, raw_line in enumerate(text.splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"line {line_number} is not an object")
                event = PolicyEvent.from_mapping(payload)
                if not event.event_id and not event.title:
                    raise ValueError(f"line {line_number} has no event identity")
                events.append(event)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            self.diagnostic_notes.append(f"policy_event_file_format_error: {exc}")
            return []

        self.status = "success"
        self.diagnostic_notes.append(f"policy_event_count={len(events)}")
        return events

    def score_theme(
        self,
        *,
        theme_id: str,
        theme_name: str,
        member_symbols: Sequence[str],
        as_of: str | date | datetime,
        events: Sequence[PolicyEvent] | None = None,
    ) -> PolicyCatalystScore:
        if events is None:
            policy_events = list(self.load_events())
            status = self.status
        else:
            policy_events = list(events)
            status = "success"
        if status != "success":
            return PolicyCatalystScore(policy_stage="unavailable")
        if not policy_events:
            return PolicyCatalystScore(policy_stage="no_match")

        as_of_date = _parse_date(as_of) or date.today()
        best: PolicyCatalystScore | None = None
        for event in policy_events:
            match = _match_event_to_theme(event, theme_id, theme_name, member_symbols)
            if not match["matched"]:
                continue
            score = _score_event(event, match, as_of_date, self.lookback_days)
            if best is None or (score.policy_score, score.confidence, score.evidence) > (
                best.policy_score,
                best.confidence,
                best.evidence,
            ):
                best = score

        return best or PolicyCatalystScore(policy_stage="no_match")


def _score_event(
    event: PolicyEvent,
    match: Mapping[str, Any],
    as_of: date,
    lookback_days: int,
) -> PolicyCatalystScore:
    recency_score = _recency_score(event.event_date(), as_of, lookback_days)
    authority_score = _authority_score(event)
    specificity_score = _specificity_score(match)
    implementation_score = _implementation_score(event.policy_type)
    funding_score = _funding_score(event.policy_type)
    beneficiary_clarity = _beneficiary_clarity(match)
    evidence_quality = clamp(len(event.evidence_text.strip()) / 120.0)
    policy_score = clamp(
        0.20 * recency_score
        + 0.18 * authority_score
        + 0.18 * specificity_score
        + 0.16 * beneficiary_clarity
        + 0.16 * implementation_score
        + 0.12 * funding_score
    )
    confidence = clamp(
        0.25 * authority_score
        + 0.25 * specificity_score
        + 0.20 * beneficiary_clarity
        + 0.20 * recency_score
        + 0.10 * evidence_quality
    )
    policy_stage = "active_catalyst" if policy_score >= 0.55 and confidence >= 0.50 else "watch"
    if recency_score <= 0.0:
        policy_stage = "stale"

    risk_flags: list[str] = []
    if recency_score <= 0.0:
        risk_flags.append("policy_stale")
    if authority_score < 0.50:
        risk_flags.append("policy_weak_authority")
    if beneficiary_clarity < 0.55:
        risk_flags.append("policy_generic_beneficiary")
    if confidence < 0.45:
        risk_flags.append("policy_low_confidence")

    evidence = [
        f"event_id={event.event_id or 'unknown'}",
        f"title={_truncate(event.title, 80)}",
        f"issuer={_truncate(event.issuer, 40)}",
        f"policy_type={_truncate(event.policy_type, 40)}",
    ]
    overlap = list(match.get("symbol_overlap", []) or [])
    if overlap:
        evidence.append(f"symbol_overlap={','.join(overlap[:5])}")

    return PolicyCatalystScore(
        policy_score=policy_score,
        confidence=confidence,
        recency_score=recency_score,
        authority_score=authority_score,
        specificity_score=specificity_score,
        implementation_score=implementation_score,
        funding_score=funding_score,
        beneficiary_clarity=beneficiary_clarity,
        evidence=evidence,
        risk_flags=risk_flags,
        policy_stage=policy_stage,
    )


def _match_event_to_theme(
    event: PolicyEvent,
    theme_id: str,
    theme_name: str,
    member_symbols: Sequence[str],
) -> dict[str, Any]:
    theme_tokens = {
        _normalize_tag(theme_id),
        _normalize_tag(theme_name),
        _normalize_tag(str(theme_id).split("::")[-1]),
    }
    theme_tokens.discard("")
    theme_tags = {_normalize_tag(tag) for tag in event.theme_tags}
    industry_tags = {_normalize_tag(tag) for tag in event.industry_tags}
    symbol_tags = {_normalize_symbol(tag) for tag in event.symbol_tags}
    symbols = {_normalize_symbol(symbol) for symbol in member_symbols}
    symbol_overlap = sorted(symbol for symbol in symbol_tags & symbols if symbol)
    theme_match = bool(theme_tokens & theme_tags)
    industry_match = bool(theme_tokens & industry_tags)
    return {
        "matched": bool(theme_match or industry_match or symbol_overlap),
        "theme_match": theme_match,
        "industry_match": industry_match,
        "symbol_overlap": symbol_overlap,
    }


def _recency_score(event_date: date | None, as_of: date, lookback_days: int) -> float:
    if event_date is None:
        return 0.0
    delta_days = (as_of - event_date).days
    if delta_days <= 0:
        return 1.0
    return clamp(1.0 - (delta_days / max(int(lookback_days), 1)))


def _authority_score(event: PolicyEvent) -> float:
    text = f"{event.policy_level} {event.issuer}".lower()
    if _contains_any(text, _CENTRAL_KEYWORDS):
        return 1.0
    if _contains_any(text, _LOCAL_KEYWORDS):
        return 0.62
    if _contains_any(text, _ASSOCIATION_KEYWORDS):
        return 0.35
    return 0.50


def _specificity_score(match: Mapping[str, Any]) -> float:
    score = 0.20
    if bool(match.get("industry_match")):
        score += 0.30
    if bool(match.get("theme_match")):
        score += 0.35
    if list(match.get("symbol_overlap", []) or []):
        score += 0.25
    return clamp(score)


def _beneficiary_clarity(match: Mapping[str, Any]) -> float:
    score = 0.20
    if bool(match.get("industry_match")):
        score += 0.20
    if bool(match.get("theme_match")):
        score += 0.35
    if list(match.get("symbol_overlap", []) or []):
        score += 0.30
    return clamp(score)


def _implementation_score(policy_type: str) -> float:
    text = policy_type.lower()
    score = 0.20
    if _contains_any(text, _PILOT_KEYWORDS):
        score += 0.30
    if _contains_any(text, _STANDARD_KEYWORDS):
        score += 0.25
    if _contains_any(text, _PROCUREMENT_KEYWORDS):
        score += 0.25
    if _contains_any(text, _FUNDING_KEYWORDS):
        score += 0.10
    return clamp(score)


def _funding_score(policy_type: str) -> float:
    text = policy_type.lower()
    score = 0.0
    if _contains_any(text, _FUNDING_KEYWORDS):
        score += 0.75
    if _contains_any(text, _PROCUREMENT_KEYWORDS):
        score += 0.20
    if _contains_any(text, _PILOT_KEYWORDS):
        score += 0.10
    return clamp(score)


def _parse_date(value: str | date | datetime | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _text(value)
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text[:8], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[,;，；|]+", value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        raw_items = [value]
    return [text for text in (_text(item) for item in raw_items) if text]


def _normalize_tag(value: Any) -> str:
    text = _text(value).lower()
    if "::" in text:
        text = text.split("::", maxsplit=1)[1]
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9a-z_\-\u4e00-\u9fff]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def _normalize_symbol(value: Any) -> str:
    return _text(value).upper()


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _truncate(value: str, limit: int) -> str:
    text = _text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _finite(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0
