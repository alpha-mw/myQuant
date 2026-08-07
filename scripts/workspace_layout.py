"""Workspace organization rules for the myQuant repository."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENVIRONMENT_ROLES = {
    "venv": "Current main-flow compatible environment.",
    ".venv": "Current script-side environment.",
    ".venv-managed": "Candidate managed environment kept for migration only.",
}

EXPLICIT_CLEANUP_DIRS = (
    Path(".cache"),
    Path(".mypy_cache"),
    Path(".pytest_cache"),
    Path(".uv-cache"),
    Path("results") / "htmlcov",
)

DERIVED_ARTIFACT_DIRS = {
    Path("results") / "htmlcov",
}


def _retired_module_path(stem: str) -> Path:
    return Path("quant_investor") / f"{stem}.py"


PROTECTED_INVENTORY_PATHS: tuple[tuple[Path, str, str], ...] = (
    (
        Path("data") / "parquet" / "cn" / "_latest.json",
        "active_runtime_source",
        "current canonical Parquet pointer; never delete in cleanup dry-run",
    ),
    (
        Path("data") / "parquet" / "cn" / "_catalog.json",
        "active_runtime_source",
        "current canonical Parquet table catalog; never delete in cleanup dry-run",
    ),
    (
        Path("data") / "parquet" / "cn" / "bars",
        "active_runtime_source",
        "canonical Parquet bars dataset governed by latest pointer",
    ),
    (
        Path("data") / "parquet_serving" / "cn" / "bars",
        "active_runtime_source",
        "symbol serving layer derived from the active canonical Parquet snapshot",
    ),
    (
        Path("data") / "factor_readiness" / "tushare",
        "duplicate_restore_source",
        "factor readiness lineage; audit before deleting duplicate copies",
    ),
    (
        Path("data") / "cleaning_reports" / "tushare",
        "duplicate_restore_source",
        "Tushare cleaning reports and storage audit lineage",
    ),
    (
        Path("data") / "raw_backups" / "tushare",
        "duplicate_restore_source",
        "raw restore source; delete only after hash/manifest proof",
    ),
    (
        Path("reports") / "storage" / "csv_quarantine",
        "duplicate_restore_source",
        "quarantine mirror; delete only after restore path and manifest proof",
    ),
    (
        Path("results") / "strategy_records",
        "strategy_evidence",
        "current and historical strategy evidence; never cache-clean directly",
    ),
    (
        Path("quant_investor") / "kline_backends",
        "code_retirement_candidate",
        "retired compatibility backend; isolate behind public CLI compatibility",
    ),
    (
        Path("quant_investor") / "kronos_predictor.py",
        "code_retirement_candidate",
        "retired Kronos predictor runtime; keep only for explicit legacy diagnostics until removal",
    ),
    (
        Path("quant_investor") / "intelligence.py",
        "code_retirement_candidate",
        "retired Kronos-era intelligence layer; v14 has no Intelligence runtime branch",
    ),
    (
        Path("quant_investor") / "_vendor" / "chronos",
        "code_retirement_candidate",
        "retired vendored Chronos runtime; audit imports and tests before removal",
    ),
    (
        Path("quant_investor") / "_vendor" / "chronos_loader.py",
        "code_retirement_candidate",
        "retired Chronos loader shim; audit imports and tests before removal",
    ),
    (
        Path("quant_investor") / "_vendor" / "kronos_model",
        "code_retirement_candidate",
        "retired vendored Kronos runtime; audit imports and tests before removal",
    ),
    (
        Path("quant_investor") / "agents" / "kline_agent.py",
        "code_retirement_candidate",
        "retired internal kline branch surface; audit references before removal",
    ),
    (
        Path("quant_investor") / "agents" / "subagents" / "kline_agent.py",
        "code_retirement_candidate",
        "retired kline subagent surface; explicit legacy imports only until removal",
    ),
    (
        _retired_module_path("advanced_risk_metrics"),
        "code_retirement_candidate",
        "retired orphaned advanced risk helper; single mainline uses canonical risk metadata",
    ),
    (
        _retired_module_path("factor_analyzer"),
        "code_retirement_candidate",
        "retired orphaned standalone factor analyzer; governed factor pipelines use package modules",
    ),
    (
        _retired_module_path("news_analysis"),
        "code_retirement_candidate",
        "retired orphaned news analyzer; v14 runtime must not call external news APIs here",
    ),
    (
        _retired_module_path("sentiment_analysis"),
        "code_retirement_candidate",
        "retired orphaned sentiment analyzer; v14 has no sentiment research branch",
    ),
    (
        _retired_module_path("signal_calibration"),
        "code_retirement_candidate",
        "retired orphaned signal calibration helper; calibration lives in governed local modules",
    ),
    (
        _retired_module_path("stress_tester"),
        "code_retirement_candidate",
        "retired orphaned stochastic stress helper; avoid random-noise stress tests outside governed risk paths",
    ),
    (
        _retired_module_path("var_calculator"),
        "code_retirement_candidate",
        "retired orphaned VaR helper; restore only after rewrite with seeded rng, ddof=1, and minimum-sample guards",
    ),
    (
        _retired_module_path("financial_analysis"),
        "code_retirement_candidate",
        "retired orphaned financial analysis helper; canonical fundamental branch owns financial metadata",
    ),
    (
        _retired_module_path("risk_management_layer"),
        "code_retirement_candidate",
        "retired orphaned risk management layer; canonical RiskGuard/risk tensor paths own runtime risk control",
    ),
)
_PATH_AUDIT_FILES = (
    Path(".venv/bin/activate"),
    Path(".venv/bin/activate.bat"),
    Path(".venv/bin/activate.csh"),
    Path(".venv/bin/activate.fish"),
    Path(".venv/bin/activate.nu"),
)

_PATH_AUDIT_GLOBS = (
    ".claude/worktrees/*/.git",
)

_SCAN_EXCLUDE_ROOTS = {
    ".git",
    ".venv",
    ".venv-managed",
    ".uv-python",
    "venv",
    "data",
    "results",
    "reports",
}


@dataclass(frozen=True)
class CleanupInventoryItem:
    relative_path: str
    classification: str
    delete_allowed: bool
    reason: str
    exists: bool
    is_dir: bool
    size_bytes: int | None = None


def _coerce_legacy_workspace_roots(
    legacy_roots: tuple[Path, ...] | list[Path] | tuple[str, ...] | list[str] | None,
) -> tuple[Path, ...]:
    if legacy_roots is None:
        return ()
    return tuple(Path(root).expanduser() for root in legacy_roots)


def get_repo_root(root: Path | None = None) -> Path:
    """Return the workspace root used by cleanup helpers."""
    if root is not None:
        return root.resolve()
    return Path(__file__).resolve().parents[1]


def get_runtime_tmp_dirs(root: Path | None = None) -> tuple[Path, Path]:
    """Return the runtime tmp directories reserved for local transient output."""
    repo_root = get_repo_root(root)
    return repo_root / "results" / "tmp", repo_root / "reports" / "tmp"


def ensure_runtime_tmp_dirs(root: Path | None = None) -> tuple[Path, Path]:
    """Create runtime tmp directories when they do not already exist."""
    tmp_dirs = get_runtime_tmp_dirs(root)
    for path in tmp_dirs:
        path.mkdir(parents=True, exist_ok=True)
    return tmp_dirs


def describe_environment_roles(root: Path | None = None) -> list[dict[str, object]]:
    """Describe the current Python environment directories kept in the repo."""
    repo_root = get_repo_root(root)
    descriptions: list[dict[str, object]] = []
    for relative_name, role in ENVIRONMENT_ROLES.items():
        path = repo_root / relative_name
        descriptions.append(
            {
                "relative_path": relative_name,
                "path": path,
                "role": role,
                "exists": path.exists(),
            }
        )
    return descriptions


def _should_skip_tree(repo_root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    if not relative.parts:
        return False
    if relative.parts[0] in _SCAN_EXCLUDE_ROOTS:
        return True
    return "node_modules" in relative.parts


def iter_cleanup_targets(root: Path | None = None) -> list[Path]:
    """Collect safe-to-delete cache directories inside the workspace."""
    repo_root = get_repo_root(root)
    targets: dict[str, Path] = {}

    for relative_dir in EXPLICIT_CLEANUP_DIRS:
        path = repo_root / relative_dir
        if path.exists():
            targets[str(path)] = path

    for current_root, dir_names, _file_names in os.walk(repo_root):
        current_path = Path(current_root)
        if _should_skip_tree(repo_root, current_path):
            dir_names[:] = []
            continue

        if current_path.name == "__pycache__":
            targets[str(current_path)] = current_path
            dir_names[:] = []
            continue

        dir_names[:] = [
            directory
            for directory in dir_names
            if not _should_skip_tree(repo_root, current_path / directory)
        ]

    return sorted(targets.values(), key=lambda path: path.relative_to(repo_root).as_posix())


def _directory_size_bytes(path: Path, *, max_entries: int = 2000) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    seen = 0
    for child in path.rglob("*"):
        seen += 1
        if seen > max_entries:
            return None
        if child.is_file():
            total += child.stat().st_size
    return total


def _cleanup_classification(path: Path) -> str:
    return "derived_artifact" if path in DERIVED_ARTIFACT_DIRS else "safe_cache"


def _cleanup_reason(path: Path) -> str:
    if path in DERIVED_ARTIFACT_DIRS:
        return "derived local artifact that can be regenerated"
    return "safe local cache directory"


def _inventory_item(
    repo_root: Path,
    relative_path: Path,
    *,
    classification: str,
    delete_allowed: bool,
    reason: str,
) -> CleanupInventoryItem:
    path = repo_root / relative_path
    return CleanupInventoryItem(
        relative_path=relative_path.as_posix(),
        classification=classification,
        delete_allowed=delete_allowed,
        reason=reason,
        exists=path.exists(),
        is_dir=path.is_dir(),
        size_bytes=_directory_size_bytes(path) if path.exists() else None,
    )


def build_cleanup_inventory(root: Path | None = None) -> dict[str, Any]:
    """Build a conservative dry-run cleanup inventory manifest."""
    repo_root = get_repo_root(root)
    items: dict[str, CleanupInventoryItem] = {}

    for path in iter_cleanup_targets(repo_root):
        relative_path = path.relative_to(repo_root)
        items[relative_path.as_posix()] = _inventory_item(
            repo_root,
            relative_path,
            classification=_cleanup_classification(relative_path),
            delete_allowed=True,
            reason=_cleanup_reason(relative_path),
        )

    for relative_path, classification, reason in PROTECTED_INVENTORY_PATHS:
        items[relative_path.as_posix()] = _inventory_item(
            repo_root,
            relative_path,
            classification=classification,
            delete_allowed=False,
            reason=reason,
        )

    ordered = [
        asdict(item)
        for item in sorted(
            items.values(),
            key=lambda item: (item.classification, item.relative_path),
        )
    ]
    summary: dict[str, int] = {}
    for item in ordered:
        classification = str(item["classification"])
        summary[classification] = summary.get(classification, 0) + 1

    return {
        "schema_version": "myquant.cleanup_inventory.v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "root": str(repo_root),
        "delete_candidate_count": sum(1 for item in ordered if item["delete_allowed"]),
        "protected_count": sum(1 for item in ordered if not item["delete_allowed"]),
        "summary": summary,
        "items": ordered,
    }


def render_cleanup_inventory_markdown(manifest: dict[str, Any]) -> str:
    """Render a compact human-readable cleanup inventory report."""
    lines = [
        "# Workspace Cleanup Inventory",
        "",
        f"- Schema: `{manifest.get('schema_version', '')}`",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Root: `{manifest.get('root', '')}`",
        f"- Delete candidates: {manifest.get('delete_candidate_count', 0)}",
        f"- Protected entries: {manifest.get('protected_count', 0)}",
        "",
        "| Classification | Delete | Path | Reason |",
        "| --- | ---: | --- | --- |",
    ]
    for item in manifest.get("items", []):
        lines.append(
            "| {classification} | {delete_allowed} | `{path}` | {reason} |".format(
                classification=item.get("classification", ""),
                delete_allowed="yes" if item.get("delete_allowed") else "no",
                path=item.get("relative_path", ""),
                reason=item.get("reason", ""),
            )
        )
    return "\n".join(lines) + "\n"


def write_cleanup_inventory_manifest(
    root: Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write JSON and Markdown cleanup inventory reports."""
    repo_root = get_repo_root(root)
    manifest = build_cleanup_inventory(repo_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root / "reports" / "project_cleanup" / f"cleanup_inventory_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cleanup_inventory.json"
    md_path = out_dir / "cleanup_inventory.md"
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_cleanup_inventory_markdown(manifest), encoding="utf-8")
    return {
        "json": str(json_path),
        "md": str(md_path),
    }


def remove_cleanup_targets(paths: list[Path]) -> list[Path]:
    """Delete collected cleanup targets and return the removed paths."""
    removed: list[Path] = []
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def iter_workspace_path_audit_targets(root: Path | None = None) -> list[Path]:
    """Collect local text files that may retain a moved workspace root."""
    repo_root = get_repo_root(root)
    targets: dict[str, Path] = {}

    for relative_path in _PATH_AUDIT_FILES:
        path = repo_root / relative_path
        if path.exists():
            targets[str(path)] = path

    for pattern in _PATH_AUDIT_GLOBS:
        for path in repo_root.glob(pattern):
            if path.is_file():
                targets[str(path)] = path

    return sorted(targets.values(), key=lambda path: path.relative_to(repo_root).as_posix())


def find_legacy_workspace_root_references(
    root: Path | None = None,
    *,
    legacy_roots: tuple[Path, ...] | list[Path] | tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, object]]:
    """Report local operational files that still point at the legacy workspace root."""
    repo_root = get_repo_root(root)
    resolved_legacy_roots = _coerce_legacy_workspace_roots(legacy_roots)
    findings: list[dict[str, object]] = []

    for path in iter_workspace_path_audit_targets(repo_root):
        text = path.read_text(encoding="utf-8")
        for legacy_root in resolved_legacy_roots:
            legacy_text = str(legacy_root)
            if legacy_text not in text:
                continue
            findings.append(
                {
                    "relative_path": path.relative_to(repo_root).as_posix(),
                    "path": path,
                    "legacy_root": legacy_root,
                }
            )
            break

    return findings


def replace_legacy_workspace_root_references(
    root: Path | None = None,
    *,
    new_root: Path | None = None,
    legacy_roots: tuple[Path, ...] | list[Path] | tuple[str, ...] | list[str] | None = None,
) -> list[Path]:
    """Rewrite legacy workspace roots inside local operational text files."""
    repo_root = get_repo_root(root)
    target_root = get_repo_root(new_root) if new_root is not None else repo_root
    resolved_legacy_roots = _coerce_legacy_workspace_roots(legacy_roots)
    updated_paths: list[Path] = []

    for path in iter_workspace_path_audit_targets(repo_root):
        original_text = path.read_text(encoding="utf-8")
        updated_text = original_text

        for legacy_root in resolved_legacy_roots:
            updated_text = updated_text.replace(str(legacy_root), str(target_root))

        if updated_text == original_text:
            continue

        path.write_text(updated_text, encoding="utf-8")
        updated_paths.append(path)

    return updated_paths
