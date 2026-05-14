"""Markdown and JSON payload renderers for offline factor governance audits."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from quant_investor.factors.library import FactorLibraryAuditReport
from quant_investor.factors.schema import (
    DEFAULT_FACTOR_LIBRARY_DIR,
    DEFAULT_PRODUCTION_FACTORS_FILENAME,
    ProductionFactorLibrary,
)
from quant_investor.factors.store import (
    DEFAULT_FACTOR_GOVERNANCE_DASHBOARD_FILENAME,
    DEFAULT_FACTOR_LIBRARY_AUDIT_DIR,
    DEFAULT_FACTOR_LIBRARY_AUDIT_REPORTS_FILENAME,
)
from quant_investor.versioning import (
    FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
    FACTOR_LIBRARY_SCHEMA_VERSION,
    FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION,
)


NON_RUNTIME_IMPACT_NOTE = (
    "This factor library audit is generated offline and does not alter stock "
    "selection, PortfolioConstructor, RiskGuard, providers, LLMs, web, or "
    "broker/execution behavior."
)
FACTOR_LIBRARY_SHADOW_NON_RUNTIME_IMPACT_NOTE = (
    "This factor library status is read-only and does not alter stock selection, "
    "portfolio construction, RiskGuard, providers, LLMs, or execution."
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    return value


def _escape_pipe(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed JSON in {path}: expected object.")
    return dict(payload)


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"Malformed JSONL in {path} at line {line_number}: expected object."
                )
            rows.append(dict(payload))
    return rows


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _coerce_non_negative_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


def _parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _expired_entry_count(entries: list[Mapping[str, Any]], as_of: str | None) -> int:
    as_of_date = _parse_iso_date(as_of)
    if as_of_date is None:
        return 0
    expired = 0
    for entry in entries:
        expires_at = _parse_iso_date(entry.get("expires_at"))
        if expires_at is not None and expires_at < as_of_date:
            expired += 1
    return expired


def _empty_shadow_status(*, root_dir: Path, as_of: str | None) -> dict[str, Any]:
    return {
        "library_exists": False,
        "audit_exists": False,
        "library_id": None,
        "audit_report_id": None,
        "verdict": "unknown",
        "production_factor_count": 0,
        "blocked_factor_count": 0,
        "shadow_only_factor_count": 0,
        "expired_factor_count": 0,
        "warning_count": 0,
        "blocker_count": 0,
        "production_factor_ids": [],
        "blocked_factor_ids": [],
        "shadow_only_factor_ids": [],
        "warnings": [],
        "metadata": {
            "root_dir": str(root_dir),
            "as_of": as_of,
            "shadow_read_only": True,
            "not_runtime_wired": True,
        },
    }


def _apply_shadow_warning(status: dict[str, Any], message: str) -> None:
    warnings = list(status.get("warnings", []) or [])
    warnings.append(message)
    status["warnings"] = sorted({str(item) for item in warnings if str(item).strip()})
    status["warning_count"] = max(
        _coerce_non_negative_int(status.get("warning_count")),
        len(status["warnings"]),
    )
    if status.get("verdict") not in {"fail", "warn"}:
        status["verdict"] = "warn"


def _apply_shadow_failure(status: dict[str, Any], message: str) -> None:
    _apply_shadow_warning(status, message)
    status["verdict"] = "fail"


def _load_latest_audit_payload(
    *,
    audit_reports_path: Path,
    dashboard_path: Path,
    status: dict[str, Any],
) -> dict[str, Any] | None:
    if audit_reports_path.exists():
        status["audit_exists"] = True
        status["metadata"]["audit_reports_path"] = str(audit_reports_path)
        rows = _read_jsonl_objects(audit_reports_path)
        if rows:
            return rows[-1]
        _apply_shadow_warning(status, f"No audit reports found in {audit_reports_path}.")
        return None
    if dashboard_path.exists():
        status["audit_exists"] = True
        status["metadata"]["audit_dashboard_path"] = str(dashboard_path)
        return _read_json_object(dashboard_path)
    return None


def _apply_audit_payload(status: dict[str, Any], payload: Mapping[str, Any]) -> None:
    try:
        audit_report = FactorLibraryAuditReport.from_dict(payload)
    except (TypeError, ValueError):
        audit_report = None

    if audit_report is not None:
        status["audit_report_id"] = audit_report.report_id
        status["library_id"] = status.get("library_id") or audit_report.library_id
        status["verdict"] = audit_report.verdict
        if not status.get("production_factor_count"):
            status["production_factor_count"] = audit_report.production_factor_count
        status["blocked_factor_ids"] = list(audit_report.blocked_factor_ids)
        status["shadow_only_factor_ids"] = list(audit_report.shadow_only_factor_ids)
        status["blocked_factor_count"] = len(audit_report.blocked_factor_ids)
        status["shadow_only_factor_count"] = len(audit_report.shadow_only_factor_ids)
        status["expired_factor_count"] = audit_report.expired_factor_count
        status["warning_count"] = audit_report.warning_count
        status["blocker_count"] = audit_report.blocker_count
        issue_messages = [
            f"{issue.severity}: {issue.message}"
            for issue in audit_report.issues
            if issue.severity in {"warning", "blocker"}
        ]
        if issue_messages:
            status["warnings"] = sorted(
                {
                    *list(status.get("warnings", []) or []),
                    *issue_messages,
                }
            )
        status["metadata"]["audit_generated_at"] = audit_report.generated_at
        return

    status["audit_report_id"] = str(
        payload.get("report_id")
        or dict(payload.get("metadata", {}) or {}).get("audit_report_id")
        or ""
    ) or None
    status["library_id"] = status.get("library_id") or payload.get("library_id")
    status["verdict"] = str(payload.get("verdict", "unknown") or "unknown")
    counts = dict(payload.get("counts", {}) or {})
    blocked_ids = _coerce_string_list(payload.get("blocked_factor_ids"))
    shadow_ids = _coerce_string_list(payload.get("shadow_only_factor_ids"))
    status["blocked_factor_ids"] = blocked_ids
    status["shadow_only_factor_ids"] = shadow_ids
    status["blocked_factor_count"] = len(blocked_ids)
    status["shadow_only_factor_count"] = len(shadow_ids)
    status["expired_factor_count"] = _coerce_non_negative_int(
        payload.get("expired_factor_count", counts.get("expired_factor_count"))
    )
    status["warning_count"] = _coerce_non_negative_int(
        payload.get("warning_count", counts.get("warning_count"))
    )
    status["blocker_count"] = _coerce_non_negative_int(
        payload.get("blocker_count", counts.get("blocker_count"))
    )
    if not status.get("production_factor_count"):
        status["production_factor_count"] = _coerce_non_negative_int(
            payload.get("production_factor_count", counts.get("production_factor_count"))
        )


def load_factor_library_shadow_status(
    *,
    root_dir: str | Path | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Read production factor library/audit status without runtime impact.

    The helper is intentionally best-effort and read-only. Missing or malformed
    local artifacts are represented in the returned status instead of escaping
    into report generation.
    """

    factor_root = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_LIBRARY_DIR
    status = _empty_shadow_status(root_dir=factor_root, as_of=as_of)
    production_path = factor_root / DEFAULT_PRODUCTION_FACTORS_FILENAME
    audit_root = (
        factor_root / "audit"
        if root_dir is not None
        else DEFAULT_FACTOR_LIBRARY_AUDIT_DIR
    )
    audit_reports_path = audit_root / DEFAULT_FACTOR_LIBRARY_AUDIT_REPORTS_FILENAME
    dashboard_path = audit_root / DEFAULT_FACTOR_GOVERNANCE_DASHBOARD_FILENAME
    status["metadata"].update(
        {
            "production_library_path": str(production_path),
            "audit_reports_path": str(audit_reports_path),
            "audit_dashboard_path": str(dashboard_path),
        }
    )

    library_payload: dict[str, Any] | None = None
    if production_path.exists():
        status["library_exists"] = True
        try:
            library_payload = _read_json_object(production_path)
            library = ProductionFactorLibrary.from_dict(library_payload)
            status["library_id"] = library.library_id
            status["production_factor_ids"] = [
                entry.factor_id for entry in library.entries
            ]
            status["production_factor_count"] = len(library.entries)
            status["expired_factor_count"] = _expired_entry_count(
                [entry.to_dict() for entry in library.entries],
                as_of,
            )
            status["metadata"]["library_generated_at"] = library.generated_at
        except (TypeError, ValueError) as exc:
            _apply_shadow_failure(status, str(exc))
    else:
        _apply_shadow_warning(status, f"Missing production factor library: {production_path}.")

    try:
        audit_payload = _load_latest_audit_payload(
            audit_reports_path=audit_reports_path,
            dashboard_path=dashboard_path,
            status=status,
        )
        if audit_payload is not None:
            _apply_audit_payload(status, audit_payload)
        elif not status.get("audit_exists"):
            _apply_shadow_warning(status, f"Missing factor library audit artifact: {audit_reports_path}.")
    except (TypeError, ValueError) as exc:
        _apply_shadow_failure(status, str(exc))

    if library_payload is not None and not status.get("production_factor_ids"):
        entries = library_payload.get("entries", [])
        if isinstance(entries, list):
            production_ids = [
                str(entry.get("factor_id")).strip()
                for entry in entries
                if isinstance(entry, Mapping) and str(entry.get("factor_id", "")).strip()
            ]
            status["production_factor_ids"] = sorted(set(production_ids))
            status["production_factor_count"] = len(status["production_factor_ids"])

    status["blocked_factor_count"] = len(status.get("blocked_factor_ids", []) or [])
    status["shadow_only_factor_count"] = len(status.get("shadow_only_factor_ids", []) or [])
    if status["verdict"] == "pass" and status.get("warnings"):
        status["verdict"] = "warn"
    json.dumps(_json_safe(status), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return dict(_json_safe(status))


def _count_payload(report: FactorLibraryAuditReport) -> dict[str, int]:
    return {
        "production_factor_count": report.production_factor_count,
        "paper_trading_factor_count": report.paper_trading_factor_count,
        "rejected_factor_count": report.rejected_factor_count,
        "deprecated_factor_count": report.deprecated_factor_count,
        "disabled_factor_count": report.disabled_factor_count,
        "expired_factor_count": report.expired_factor_count,
        "issue_count": report.issue_count,
        "blocker_count": report.blocker_count,
        "warning_count": report.warning_count,
        "info_count": report.info_count,
    }


def _render_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- `{value}`" for value in values]


def _render_shadow_list(values: Any) -> list[str]:
    coerced = _coerce_string_list(values)
    if not coerced:
        return ["- None"]
    return [f"- `{_escape_pipe(value)}`" for value in coerced]


def render_factor_library_shadow_markdown(status: Mapping[str, Any]) -> str:
    lines = [
        "## Factor Library Status (Read-only Shadow)",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Library exists | `{bool(status.get('library_exists', False))}` |",
        f"| Audit exists | `{bool(status.get('audit_exists', False))}` |",
        f"| Library ID | `{_escape_pipe(status.get('library_id') or 'unknown')}` |",
        f"| Audit report ID | `{_escape_pipe(status.get('audit_report_id') or 'unknown')}` |",
        f"| Verdict | `{_escape_pipe(status.get('verdict') or 'unknown')}` |",
        "",
        "| Count | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "production_factor_count",
        "blocked_factor_count",
        "shadow_only_factor_count",
        "expired_factor_count",
        "warning_count",
        "blocker_count",
    ]:
        lines.append(f"| `{_escape_pipe(key)}` | {_coerce_non_negative_int(status.get(key))} |")

    lines.extend(["", "### Production Factor IDs", ""])
    lines.extend(_render_shadow_list(status.get("production_factor_ids")))
    lines.extend(["", "### Blocked Factor IDs", ""])
    lines.extend(_render_shadow_list(status.get("blocked_factor_ids")))
    lines.extend(["", "### Warnings", ""])
    lines.extend(_render_shadow_list(status.get("warnings")))
    lines.extend(
        [
            "",
            "### Runtime Impact",
            "",
            FACTOR_LIBRARY_SHADOW_NON_RUNTIME_IMPACT_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def render_factor_library_audit_markdown(report: FactorLibraryAuditReport) -> str:
    lines = [
        f"# Factor Library Audit: {report.report_id}",
        "",
        f"Generated at: `{report.generated_at}`",
        "",
        "## Verdict",
        "",
        f"`{report.verdict}`",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key, value in _count_payload(report).items():
        lines.append(f"| `{_escape_pipe(key)}` | {value} |")

    lines.extend(["", "## Allowed Factors", ""])
    lines.extend(_render_list(report.allowed_factor_ids))
    lines.extend(["", "## Blocked Factors", ""])
    lines.extend(_render_list(report.blocked_factor_ids))
    lines.extend(["", "## Shadow-only Factors", ""])
    lines.extend(_render_list(report.shadow_only_factor_ids))

    lines.extend(
        [
            "",
            "## Issue Table",
            "",
            "| Severity | Code | Factor | Version | Message |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    if report.issues:
        for issue in report.issues:
            lines.append(
                "| "
                f"`{_escape_pipe(issue.severity)}` | "
                f"`{_escape_pipe(issue.issue_code)}` | "
                f"{_escape_pipe(issue.factor_id or '')} | "
                f"{_escape_pipe(issue.factor_version or '')} | "
                f"{_escape_pipe(issue.message)} |"
            )
    else:
        lines.append("|  |  |  |  | No issues. |")

    lines.extend(
        [
            "",
            "## Policy Snapshot",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )
    for key, value in report.policy.to_dict().items():
        if key == "metadata":
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = value
        lines.append(f"| `{_escape_pipe(key)}` | `{_escape_pipe(rendered)}` |")

    lines.extend(
        [
            "",
            "## Runtime Impact",
            "",
            NON_RUNTIME_IMPACT_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def build_factor_governance_dashboard_payload(
    *,
    library: ProductionFactorLibrary | None,
    audit_report: FactorLibraryAuditReport,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "verdict": audit_report.verdict,
        "counts": _count_payload(audit_report),
        "allowed_factor_ids": list(audit_report.allowed_factor_ids),
        "blocked_factor_ids": list(audit_report.blocked_factor_ids),
        "shadow_only_factor_ids": list(audit_report.shadow_only_factor_ids),
        "issues": [issue.to_dict() for issue in audit_report.issues],
        "policy": audit_report.policy.to_dict(),
        "library_id": library.library_id if library is not None else audit_report.library_id,
        "schema_versions": {
            "FACTOR_LIBRARY_SCHEMA_VERSION": FACTOR_LIBRARY_SCHEMA_VERSION,
            "FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION": FACTOR_LIBRARY_AUDIT_SCHEMA_VERSION,
            "FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION": (
                FACTOR_PRODUCTION_GUARDRAIL_SCHEMA_VERSION
            ),
        },
        "metadata": {
            **dict(metadata or {}),
            "audit_report_id": audit_report.report_id,
            "offline_only": True,
            "not_runtime_wired": True,
        },
    }
    json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return dict(_json_safe(payload))


__all__ = [
    "FACTOR_LIBRARY_SHADOW_NON_RUNTIME_IMPACT_NOTE",
    "NON_RUNTIME_IMPACT_NOTE",
    "load_factor_library_shadow_status",
    "render_factor_library_audit_markdown",
    "render_factor_library_shadow_markdown",
    "build_factor_governance_dashboard_payload",
]
