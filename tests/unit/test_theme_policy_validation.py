from __future__ import annotations

import json
from pathlib import Path

from quant_investor.themes.policy_validation import (
    PolicyEventValidationIssue,
    validate_policy_event_jsonl,
    validate_policy_event_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": "policy-1",
        "title": "半导体设备政策",
        "issuer": "工业和信息化部",
        "publish_date": "2026-06-01",
        "effective_date": "2026-06-02",
        "policy_level": "ministry",
        "policy_type": "notice",
        "theme_tags": ["半导体设备"],
        "industry_tags": [],
        "symbol_tags": [],
        "evidence_text": "本地维护缓存中的政策摘要。",
        "source_url": "local://policy-cache/policy-1",
    }
    payload.update(overrides)
    return payload


def _codes(issues: list[PolicyEventValidationIssue]) -> set[str]:
    return {issue.code for issue in issues}


def _write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "\n".join(
            row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def test_valid_payload_has_no_issues() -> None:
    assert validate_policy_event_payload(_valid_payload()) == []
    assert validate_policy_event_payload(_valid_payload(publish_date="20260601")) == []


def test_missing_required_fields_are_errors() -> None:
    issues = validate_policy_event_payload(
        _valid_payload(event_id="", title="", issuer="", publish_date=""),
        line_no=7,
    )

    assert {
        "missing_event_id",
        "missing_title",
        "missing_issuer",
        "missing_publish_date",
    } <= _codes(issues)
    assert all(issue.severity == "error" for issue in issues)
    assert {issue.line_no for issue in issues} == {7}


def test_no_theme_mapping_is_error() -> None:
    issues = validate_policy_event_payload(
        _valid_payload(theme_tags=[], industry_tags=[], symbol_tags=[]),
        line_no=2,
    )

    assert _codes(issues) == {"no_theme_mapping"}
    assert issues[0].severity == "error"
    assert issues[0].event_id == "policy-1"


def test_unknown_policy_level_and_type_are_warnings() -> None:
    issues = validate_policy_event_payload(
        _valid_payload(policy_level="national", policy_type="guidance"),
        line_no=3,
    )

    assert _codes(issues) == {"unknown_policy_level", "unknown_policy_type"}
    assert all(issue.severity == "warning" for issue in issues)


def test_duplicate_event_id_is_file_level_error(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "policy.jsonl",
        [
            _valid_payload(event_id="dup"),
            _valid_payload(event_id="dup", title="重复政策"),
        ],
    )

    issues = validate_policy_event_jsonl(path)

    assert "duplicate_event_id" in _codes(issues)
    duplicate = [issue for issue in issues if issue.code == "duplicate_event_id"][0]
    assert duplicate.line_no == 2
    assert duplicate.severity == "error"
    assert duplicate.event_id == "dup"


def test_malformed_json_line_is_error(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "policy.jsonl",
        [
            _valid_payload(event_id="ok"),
            "{bad json",
        ],
    )

    issues = validate_policy_event_jsonl(path)

    assert "json_parse_error" in _codes(issues)
    parse_issue = [issue for issue in issues if issue.code == "json_parse_error"][0]
    assert parse_issue.line_no == 2
    assert parse_issue.severity == "error"


def test_example_file_validates_without_errors_or_unexpected_warnings() -> None:
    issues = validate_policy_event_jsonl(
        REPO_ROOT / "quant_investor" / "themes" / "data" / "theme_policy_events.example.jsonl"
    )
    allowed_warnings = {"missing_source_url"}

    assert not [issue for issue in issues if issue.severity == "error"]
    assert _codes(issues) <= allowed_warnings
