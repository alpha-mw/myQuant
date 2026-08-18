from __future__ import annotations

import json

import pytest

from scripts.cn_dashboard_v2_selector import (
    build_selector,
    expected_private_dashboard_output_path,
    publish_selector,
    read_selector,
    validate_selector,
)


def test_selector_status_contract_and_ordered_publication(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    json_path = expected_private_dashboard_output_path(
        project_root, "cn_aggressive_dashboard_selector.v2.json"
    )
    js_path = expected_private_dashboard_output_path(
        project_root, "cn_aggressive_dashboard_selector.v2.js"
    )
    refreshing = build_selector(
        attempt_id="attempt-1",
        status="REFRESHING",
        updated_at="2026-08-18T09:30:00+08:00",
        reason="refresh_started",
    )
    publish_selector(
        refreshing,
        json_path=json_path,
        js_path=js_path,
        project_root=project_root,
        js_first=True,
    )
    assert read_selector(json_path) == refreshing
    assert "REFRESHING" in js_path.read_text(encoding="utf-8")

    updated = build_selector(
        attempt_id="attempt-1",
        status="UPDATED",
        updated_at="2026-08-18T09:31:00+08:00",
        reason="refresh_completed",
        v2_content_sha256="a" * 64,
    )
    publish_selector(
        updated,
        json_path=json_path,
        js_path=js_path,
        project_root=project_root,
        js_first=False,
    )
    assert read_selector(json_path) == updated
    assert "UPDATED" in js_path.read_text(encoding="utf-8")


def test_nonupdated_selector_rejects_v2_hash() -> None:
    selector = build_selector(
        attempt_id="attempt-2",
        status="BLOCKED",
        updated_at="2026-08-18T09:31:00+08:00",
        reason="refresh_failed",
    )
    selector["v2_content_sha256"] = "b" * 64
    selector["content_sha256"] = "0" * 64
    errors = validate_selector(selector)
    assert "selector_nonupdated_v2_sha_present" in errors


def test_selector_self_hash_and_stable_read_are_enforced(tmp_path) -> None:
    path = tmp_path / "selector.json"
    selector = build_selector(
        attempt_id="attempt-3",
        status="REFRESHING",
        updated_at="2026-08-18T09:30:00+08:00",
        reason="refresh_started",
    )
    selector["reason"] = "tampered"
    path.write_text(json.dumps(selector), encoding="utf-8")
    with pytest.raises(ValueError, match="selector_content_sha_invalid"):
        read_selector(path)


@pytest.mark.parametrize(
    ("attempt_id", "updated_at", "message"),
    [
        ("bad id", "2026-08-18T09:30:00+08:00", "attempt_id"),
        ("attempt-4", "2026-08-18T01:30:00Z", "updated_at"),
    ],
)
def test_selector_python_grammar_matches_browser_contract(
    attempt_id: str, updated_at: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_selector(
            attempt_id=attempt_id,
            status="REFRESHING",
            updated_at=updated_at,
            reason="refresh_started",
        )
