from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


PolicyIssueSeverity = Literal["error", "warning"]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{8}$")
_ALLOWED_POLICY_LEVELS = {
    "central",
    "ministry",
    "local",
    "association",
    "exchange",
    "other",
}
_ALLOWED_POLICY_TYPES = {
    "plan",
    "notice",
    "subsidy",
    "standard",
    "pilot",
    "procurement",
    "consultation",
    "project_list",
    "funding",
    "tax",
    "other",
}


@dataclass(frozen=True)
class PolicyEventValidationIssue:
    line_no: int
    severity: PolicyIssueSeverity
    code: str
    message: str
    event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_no": self.line_no,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "event_id": self.event_id,
        }


def validate_policy_event_payload(
    payload: Mapping[str, Any],
    line_no: int = 0,
) -> list[PolicyEventValidationIssue]:
    """Validate one local policy event payload without fetching any sources."""

    event_id = _text(payload.get("event_id"))
    issues: list[PolicyEventValidationIssue] = []

    def add(
        severity: PolicyIssueSeverity,
        code: str,
        message: str,
        *,
        event_id_override: str | None = None,
    ) -> None:
        issues.append(
            PolicyEventValidationIssue(
                line_no=int(line_no or 0),
                severity=severity,
                code=code,
                message=message,
                event_id=event_id if event_id_override is None else event_id_override,
            )
        )

    if not event_id:
        add("error", "missing_event_id", "event_id is required and must be non-empty")
    if not _text(payload.get("title")):
        add("error", "missing_title", "title is required")
    if not _text(payload.get("issuer")):
        add("error", "missing_issuer", "issuer is required")

    publish_date = _text(payload.get("publish_date"))
    if not publish_date:
        add("error", "missing_publish_date", "publish_date is required")
    elif not _valid_date_text(publish_date):
        add(
            "error",
            "invalid_publish_date",
            "publish_date must use YYYY-MM-DD or YYYYMMDD",
        )

    if not (
        _has_non_empty_tag(payload.get("theme_tags"))
        or _has_non_empty_tag(payload.get("industry_tags"))
        or _has_non_empty_tag(payload.get("symbol_tags"))
    ):
        add(
            "error",
            "no_theme_mapping",
            "at least one of theme_tags, industry_tags, or symbol_tags must be non-empty",
        )

    policy_level = _text(payload.get("policy_level")).lower()
    if policy_level and policy_level not in _ALLOWED_POLICY_LEVELS:
        add(
            "warning",
            "unknown_policy_level",
            f"policy_level is not in the allowed enum: {policy_level}",
        )

    policy_type = _text(payload.get("policy_type")).lower()
    if policy_type and policy_type not in _ALLOWED_POLICY_TYPES:
        add(
            "warning",
            "unknown_policy_type",
            f"policy_type is not in the allowed enum: {policy_type}",
        )

    if not _text(payload.get("source_url")):
        add("warning", "missing_source_url", "source_url is empty; no network validation is attempted")
    if not _text(payload.get("evidence_text")):
        add("warning", "missing_evidence_text", "evidence_text is empty")

    return issues


def validate_policy_event_jsonl(path: str | Path) -> list[PolicyEventValidationIssue]:
    """Validate a local policy event JSONL file without resolving source URLs."""

    event_path = Path(path)
    if not event_path.exists():
        return [
            PolicyEventValidationIssue(
                line_no=0,
                severity="error",
                code="file_missing",
                message=f"policy event JSONL file does not exist: {event_path}",
                event_id="",
            )
        ]

    try:
        lines = event_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [
            PolicyEventValidationIssue(
                line_no=0,
                severity="error",
                code="file_read_error",
                message=str(exc),
                event_id="",
            )
        ]

    issues: list[PolicyEventValidationIssue] = []
    seen_event_lines: dict[str, int] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                PolicyEventValidationIssue(
                    line_no=line_no,
                    severity="error",
                    code="json_parse_error",
                    message=f"invalid JSON: {exc.msg}",
                    event_id="",
                )
            )
            continue
        if not isinstance(payload, Mapping):
            issues.append(
                PolicyEventValidationIssue(
                    line_no=line_no,
                    severity="error",
                    code="invalid_json_object",
                    message="each JSONL line must be a JSON object",
                    event_id="",
                )
            )
            continue

        event_id = _text(payload.get("event_id"))
        issues.extend(validate_policy_event_payload(payload, line_no=line_no))
        if event_id:
            previous_line = seen_event_lines.get(event_id)
            if previous_line is not None:
                issues.append(
                    PolicyEventValidationIssue(
                        line_no=line_no,
                        severity="error",
                        code="duplicate_event_id",
                        message=(
                            f"event_id duplicates line {previous_line}: {event_id}"
                        ),
                        event_id=event_id,
                    )
                )
            else:
                seen_event_lines[event_id] = line_no

    return issues


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _valid_date_text(value: str) -> bool:
    return bool(_DATE_RE.match(value.strip()))


def _has_non_empty_tag(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return any(bool(_text(item)) for item in value)
    except TypeError:
        return False


__all__ = [
    "PolicyEventValidationIssue",
    "validate_policy_event_jsonl",
    "validate_policy_event_payload",
]
