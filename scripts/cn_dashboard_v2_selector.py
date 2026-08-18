"""Fail-closed serving selector for the private CN aggressive Dashboard v2.

The selector is deliberately separate from Strategy Record Store state.  Its
only job is to stop an earlier successful Dashboard refresh from continuing to
look current after a later same-day refresh starts or fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "cn_aggressive_dashboard_selector.v2"
SELECTOR_STATUSES = {"REFRESHING", "UPDATED", "BLOCKED"}
MAX_SELECTOR_BYTES = 16 * 1024
_ATTEMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHANGHAI_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+08:00$")
PRIVATE_DASHBOARD_OUTPUT_PARTS = ("portfolio_dashboard", "private", "generated")
SELECTOR_JSON_FILENAME = "cn_aggressive_dashboard_selector.v2.json"
SELECTOR_JS_FILENAME = "cn_aggressive_dashboard_selector.v2.js"


def private_dashboard_output_directory(project_root: Path) -> Path:
    """Return the sole writable Dashboard artifact directory.

    Dashboard output is deliberately confined to the private generated tree.
    In particular, a symlinked ``private`` or ``generated`` directory must not
    silently redirect an otherwise read-only Dashboard workflow into System,
    Store, data, or public-deployment state.
    """

    resolved_root = project_root.resolve()
    output_directory = resolved_root
    for part in PRIVATE_DASHBOARD_OUTPUT_PARTS:
        output_directory = output_directory / part
        if output_directory.is_symlink():
            raise ValueError("dashboard_private_output_root_symlink_forbidden")
    return output_directory


def expected_private_dashboard_output_path(project_root: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("dashboard_output_filename_invalid")
    return private_dashboard_output_directory(project_root) / filename


def require_exact_private_dashboard_output_path(
    *, project_root: Path, path: Path, filename: str
) -> Path:
    """Reject every Dashboard output alias except its private canonical path."""

    expected = expected_private_dashboard_output_path(project_root, filename)
    actual = Path(os.path.abspath(os.fspath(path)))
    if actual != expected or actual.is_symlink():
        raise ValueError("dashboard_output_path_forbidden")
    return expected


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_sha256(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def build_selector(
    *,
    attempt_id: str,
    status: str,
    updated_at: str,
    reason: str,
    v2_content_sha256: str | None = None,
) -> dict[str, Any]:
    selector = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": str(attempt_id),
        "status": str(status),
        "updated_at": str(updated_at),
        "v2_content_sha256": v2_content_sha256,
        "reason": str(reason),
    }
    selector["content_sha256"] = _content_sha256(selector)
    errors = validate_selector(selector)
    if errors:
        raise ValueError("selector_invalid:" + ";".join(errors))
    return selector


def validate_selector(selector: Any) -> list[str]:
    if not isinstance(selector, dict):
        return ["selector_not_object"]
    required = {
        "schema_version",
        "attempt_id",
        "status",
        "updated_at",
        "v2_content_sha256",
        "reason",
        "content_sha256",
    }
    errors: list[str] = []
    if set(selector) != required:
        errors.append("selector_keys_invalid")
    if selector.get("schema_version") != SCHEMA_VERSION:
        errors.append("selector_schema_invalid")
    if (
        not isinstance(selector.get("attempt_id"), str)
        or _ATTEMPT_RE.fullmatch(selector["attempt_id"]) is None
    ):
        errors.append("selector_attempt_id_invalid")
    status = selector.get("status")
    if status not in SELECTOR_STATUSES:
        errors.append("selector_status_invalid")
    updated_at = selector.get("updated_at")
    if not isinstance(updated_at, str) or _SHANGHAI_TIMESTAMP_RE.fullmatch(updated_at) is None:
        errors.append("selector_updated_at_invalid")
    else:
        try:
            parsed_updated_at = datetime.fromisoformat(updated_at)
        except ValueError:
            errors.append("selector_updated_at_invalid")
        else:
            if parsed_updated_at.utcoffset() != timedelta(hours=8):
                errors.append("selector_updated_at_invalid")
    if not str(selector.get("reason") or "").strip():
        errors.append("selector_reason_invalid")
    v2_sha = selector.get("v2_content_sha256")
    if status == "UPDATED":
        if not isinstance(v2_sha, str) or len(v2_sha) != 64:
            errors.append("selector_v2_sha_invalid")
        elif any(character not in "0123456789abcdef" for character in v2_sha):
            errors.append("selector_v2_sha_invalid")
    elif v2_sha is not None:
        errors.append("selector_nonupdated_v2_sha_present")
    content_sha = selector.get("content_sha256")
    if not isinstance(content_sha, str) or content_sha != _content_sha256(selector):
        errors.append("selector_content_sha_invalid")
    if len(canonical_json_bytes(selector)) > MAX_SELECTOR_BYTES:
        errors.append("selector_too_large")
    return errors


def render_json(selector: dict[str, Any]) -> bytes:
    errors = validate_selector(selector)
    if errors:
        raise ValueError("selector_invalid:" + ";".join(errors))
    return (json.dumps(selector, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def render_js(selector: dict[str, Any]) -> bytes:
    errors = validate_selector(selector)
    if errors:
        raise ValueError("selector_invalid:" + ";".join(errors))
    payload = canonical_json_bytes(selector).decode("utf-8")
    return ("window.MyQuantCNAggressiveDashboardSelectorV2 = " + payload + ";\n").encode("utf-8")


def _atomic_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if path.read_bytes() != raw:
            raise OSError("selector_readback_mismatch")
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_selector(
    selector: dict[str, Any],
    *,
    json_path: Path,
    js_path: Path,
    project_root: Path,
    js_first: bool,
) -> None:
    json_path = require_exact_private_dashboard_output_path(
        project_root=project_root,
        path=json_path,
        filename=SELECTOR_JSON_FILENAME,
    )
    js_path = require_exact_private_dashboard_output_path(
        project_root=project_root,
        path=js_path,
        filename=SELECTOR_JS_FILENAME,
    )
    json_raw = render_json(selector)
    js_raw = render_js(selector)
    ordered = (
        ((js_path, js_raw), (json_path, json_raw))
        if js_first
        else ((json_path, json_raw), (js_path, js_raw))
    )
    for path, raw in ordered:
        _atomic_replace(path, raw)


def read_selector(path: Path) -> dict[str, Any]:
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise ValueError("selector_unstable_double_read")
    selector = json.loads(first.decode("utf-8"))
    errors = validate_selector(selector)
    if errors:
        raise ValueError("selector_invalid:" + ";".join(errors))
    return selector
