"""Workspace organization rules for the myQuant repository."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT_DIRS = (
    "quant_investor",
    "portfolio_dashboard",
    "tests",
    "docs",
    "data",
    "results",
    "reports",
)

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

_RETIREMENT_REFERENCE_SCAN_ROOTS = (
    Path("quant_investor"),
    Path("tests"),
    Path("scripts"),
    Path("docs"),
)

_RETIREMENT_REFERENCE_SCAN_FILES = (
    Path("README.md"),
    Path("pyproject.toml"),
)

_RETIREMENT_REFERENCE_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
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


def _iter_code_retirement_candidates(repo_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for relative_path, classification, reason in PROTECTED_INVENTORY_PATHS:
        if classification != "code_retirement_candidate":
            continue
        path = repo_root / relative_path
        module_name = _retirement_module_name(relative_path)
        candidates.append(
            {
                "relative_path": relative_path.as_posix(),
                "path": path,
                "module_name": module_name,
                "tokens": _retirement_reference_tokens(relative_path, module_name),
                "reason": reason,
                "exists": path.exists(),
                "is_dir": path.is_dir(),
            }
        )
    return sorted(candidates, key=lambda item: str(item["relative_path"]))


def _retirement_module_name(relative_path: Path) -> str:
    if not relative_path.parts or relative_path.parts[0] != "quant_investor":
        return ""
    parts = list(relative_path.parts)
    if relative_path.suffix == ".py":
        parts[-1] = relative_path.stem
    return ".".join(parts)


def _retirement_reference_tokens(relative_path: Path, module_name: str) -> list[str]:
    tokens = {relative_path.as_posix()}
    if relative_path.suffix:
        tokens.add(relative_path.name)
    short_name = relative_path.stem if relative_path.suffix else relative_path.name
    if not relative_path.suffix and "_" in short_name:
        tokens.add(short_name)
    if module_name:
        tokens.add(module_name)
        tokens.add(module_name.replace(".", "/"))
    return sorted(token for token in tokens if token)


def _is_relative_to_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_retirement_candidate_source(source: Path, candidate_paths: list[Path]) -> bool:
    for candidate_path in candidate_paths:
        if source == candidate_path:
            return True
        if candidate_path.suffix:
            continue
        if _is_relative_to_path(source, candidate_path):
            return True
    return False


def _iter_retirement_reference_files(repo_root: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for relative_root in _RETIREMENT_REFERENCE_SCAN_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in _RETIREMENT_REFERENCE_SUFFIXES:
                continue
            files[path.relative_to(repo_root).as_posix()] = path
    for relative_path in _RETIREMENT_REFERENCE_SCAN_FILES:
        path = repo_root / relative_path
        if path.exists() and path.is_file():
            files[path.relative_to(repo_root).as_posix()] = path
    return [files[key] for key in sorted(files)]


def _classify_retirement_reference(
    source_relative_path: Path,
    *,
    candidate_paths: list[Path],
) -> str:
    if source_relative_path in {
        Path("scripts") / "workspace_layout.py",
        Path("tests") / "unit" / "test_workspace_cleanup.py",
    }:
        return "cleanup_manifest_reference"
    if _is_retirement_candidate_source(source_relative_path, candidate_paths):
        return "candidate_internal_reference"
    if source_relative_path.parts and source_relative_path.parts[0] == "tests":
        return "test_reference"
    if source_relative_path.parts and source_relative_path.parts[0] == "docs":
        return "docs_reference"
    if source_relative_path == Path("README.md"):
        return "docs_reference"
    if source_relative_path.parts and source_relative_path.parts[0] == "quant_investor":
        return "production_reference"
    if source_relative_path.parts and source_relative_path.parts[0] == "scripts":
        return "tooling_reference"
    return "other_reference"


def build_code_retirement_reference_audit(root: Path | None = None) -> dict[str, Any]:
    """Build a static reference audit for protected code-retirement candidates."""
    repo_root = get_repo_root(root)
    candidates = _iter_code_retirement_candidates(repo_root)
    candidate_paths = [Path(str(candidate["relative_path"])) for candidate in candidates]
    scan_files = _iter_retirement_reference_files(repo_root)
    candidate_payloads: list[dict[str, Any]] = []

    for candidate in candidates:
        relative_path = Path(str(candidate["relative_path"]))
        references: list[dict[str, Any]] = []
        for source_path in scan_files:
            source_relative_path = source_path.relative_to(repo_root)
            if source_relative_path == relative_path:
                continue
            try:
                lines = source_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(lines, start=1):
                if not any(token in line for token in candidate["tokens"]):
                    continue
                references.append(
                    {
                        "relative_path": source_relative_path.as_posix(),
                        "line": lineno,
                        "classification": _classify_retirement_reference(
                            source_relative_path,
                            candidate_paths=candidate_paths,
                        ),
                        "text": line.strip()[:240],
                    }
                )

        reference_summary: dict[str, int] = {}
        for reference in references:
            classification = str(reference["classification"])
            reference_summary[classification] = reference_summary.get(classification, 0) + 1

        candidate_payloads.append(
            {
                "relative_path": candidate["relative_path"],
                "module_name": candidate["module_name"],
                "reason": candidate["reason"],
                "exists": candidate["exists"],
                "is_dir": candidate["is_dir"],
                "tokens": candidate["tokens"],
                "reference_count": len(references),
                "production_reference_count": reference_summary.get("production_reference", 0),
                "reference_summary": dict(sorted(reference_summary.items())),
                "references": sorted(
                    references,
                    key=lambda item: (str(item["relative_path"]), int(item["line"])),
                ),
            }
        )

    return {
        "schema_version": "myquant.code_retirement_reference_audit.v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "root": str(repo_root),
        "candidate_count": len(candidate_payloads),
        "production_reference_count": sum(
            int(candidate["production_reference_count"])
            for candidate in candidate_payloads
        ),
        "candidates": candidate_payloads,
    }


def render_code_retirement_reference_audit_markdown(manifest: dict[str, Any]) -> str:
    """Render a compact Markdown report for code-retirement references."""
    lines = [
        "# Code Retirement Reference Audit",
        "",
        f"- Schema: `{manifest.get('schema_version', '')}`",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Root: `{manifest.get('root', '')}`",
        f"- Candidates: {manifest.get('candidate_count', 0)}",
        f"- Production references: {manifest.get('production_reference_count', 0)}",
        "",
        "| Candidate | Module | References | Production refs | Summary |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for candidate in manifest.get("candidates", []):
        summary = ", ".join(
            f"{key}={value}"
            for key, value in candidate.get("reference_summary", {}).items()
        )
        lines.append(
            "| `{path}` | `{module}` | {refs} | {prod_refs} | {summary} |".format(
                path=candidate.get("relative_path", ""),
                module=candidate.get("module_name", ""),
                refs=candidate.get("reference_count", 0),
                prod_refs=candidate.get("production_reference_count", 0),
                summary=summary or "-",
            )
        )
    lines.append("")
    for candidate in manifest.get("candidates", []):
        lines.append(f"## `{candidate.get('relative_path', '')}`")
        lines.append("")
        lines.append(f"- Reason: {candidate.get('reason', '')}")
        lines.append(f"- Module: `{candidate.get('module_name', '')}`")
        lines.append(f"- Production references: {candidate.get('production_reference_count', 0)}")
        lines.append("")
        for reference in candidate.get("references", [])[:20]:
            text = str(reference.get("text", "")).replace("|", "\\|")
            lines.append(
                "- `{path}:{line}` [{classification}] {text}".format(
                    path=reference.get("relative_path", ""),
                    line=reference.get("line", ""),
                    classification=reference.get("classification", ""),
                    text=text,
                )
            )
        if candidate.get("reference_count", 0) > 20:
            lines.append("- ... truncated after 20 references")
        lines.append("")
    return "\n".join(lines)


def write_code_retirement_reference_audit(
    root: Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write JSON and Markdown code-retirement reference audit reports."""
    repo_root = get_repo_root(root)
    manifest = build_code_retirement_reference_audit(repo_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root / "reports" / "project_cleanup" / f"code_retirement_reference_audit_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "code_retirement_reference_audit.json"
    md_path = out_dir / "code_retirement_reference_audit.md"
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(
        render_code_retirement_reference_audit_markdown(manifest),
        encoding="utf-8",
    )
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
