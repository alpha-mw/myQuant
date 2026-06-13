"""Write a current architecture/performance rebaseline audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.project_cleanup_status import (
    _batch_read_profile_proves_batch_path,
    _current_large_modules,
    _load_json,
    _market_report_split_evidence,
    _reader_cache_evidence,
    _runtime_profile_proves_stage_profile,
    _runtime_profile_summary,
    _slowest_stages,
    _strategy_profile_split_evidence,
)
from scripts.workspace_layout import get_repo_root

SCHEMA_VERSION = "myquant_architecture_performance_audit.v1"
RUNTIME_PROFILE_PATTERN = "results/cn_analysis_full/CN_Runtime_Profile_*.json"


def _latest_runtime_profile_path(repo_root: Path) -> Path | None:
    candidates = sorted(repo_root.glob(RUNTIME_PROFILE_PATTERN))
    return candidates[-1] if candidates else None


def _finding(
    *,
    finding_id: str,
    severity: str,
    title: str,
    evidence: list[str],
    recommended_patch: str,
) -> dict[str, Any]:
    return {
        "id": finding_id,
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "recommended_patch": recommended_patch,
    }


def _architecture_findings(
    repo_root: Path,
    runtime_profile: dict[str, Any] | None,
    current_large_modules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if current_large_modules:
        findings.append(
            _finding(
                finding_id="arch-large-modules",
                severity="medium",
                title="Current source still has modules above the large-module threshold",
                evidence=[
                    f"{item['path']}:{item['lines']}"
                    for item in current_large_modules[:10]
                ],
                recommended_patch=(
                    "Split current large modules along tested ownership boundaries before "
                    "marking the architecture rebaseline clear."
                ),
            )
        )
    if not _runtime_profile_proves_stage_profile(runtime_profile):
        findings.append(
            _finding(
                finding_id="perf-stage-profile-incomplete",
                severity="high",
                title="Latest market runtime profile does not prove complete DAG stage coverage",
                evidence=[
                    "required stages: dag_symbol_list, dag_batch_read, dag_funnel, "
                    "dag_candidate_research, dag_bayesian_selection, dag_control_chain, "
                    "dag_reporting_artifacts, analysis_report_persistence"
                ],
                recommended_patch=(
                    "Run or repair market analyze/run profiling until the persisted "
                    "market_runtime_profile.v1 report contains every required stage."
                ),
            )
        )
    if not _batch_read_profile_proves_batch_path(runtime_profile):
        findings.append(
            _finding(
                finding_id="perf-batch-read-not-proven",
                severity="high",
                title="Latest market runtime profile does not prove strict batch Parquet reads",
                evidence=[
                    "dag_batch_read must include batch_result_count>0, projected columns, "
                    "runtime lookback, and per_symbol_fallback_count=0"
                ],
                recommended_patch=(
                    "Keep the DAG on read_symbol_frames() with projected columns and "
                    "fail closed when strict Parquet cannot batch read."
                ),
            )
        )
    if not _reader_cache_evidence(repo_root):
        findings.append(
            _finding(
                finding_id="perf-reader-cache-missing",
                severity="medium",
                title="MarketDataReader cache evidence is missing",
                evidence=["market_data_reader.py should cache latest pointer, snapshot gate, serving symbols, and components payload"],
                recommended_patch=(
                    "Restore reader instance caches before accepting the performance "
                    "rebaseline."
                ),
            )
        )
    if not _market_report_split_evidence(repo_root):
        findings.append(
            _finding(
                finding_id="arch-report-persistence-split-missing",
                severity="medium",
                title="Market report rendering/persistence split evidence is missing",
                evidence=[
                    "required modules: runtime_profile.py, report_persistence.py, "
                    "full_report_helpers.py, full_report_sections.py"
                ],
                recommended_patch=(
                    "Keep report persistence and rendering outside the main analysis "
                    "orchestration boundary."
                ),
            )
        )
    if not _strategy_profile_split_evidence(repo_root):
        findings.append(
            _finding(
                finding_id="arch-strategy-profiler-split-missing",
                severity="medium",
                title="Strategy profiler/report split evidence is missing",
                evidence=[
                    "required CN tracker modules: review_layer, review_runtime, "
                    "rebalance, report_renderer"
                ],
                recommended_patch=(
                    "Keep reusable runtime profiling in the market layer and CN tracker "
                    "specific reporting in focused monitoring modules."
                ),
            )
        )
    return findings


def build_architecture_rebaseline_audit(root: Path | None = None) -> dict[str, Any]:
    """Build a read-only current architecture and performance audit."""
    repo_root = get_repo_root(root)
    runtime_profile_path = _latest_runtime_profile_path(repo_root)
    runtime_profile = _load_json(runtime_profile_path)
    current_large_modules = _current_large_modules(repo_root)
    findings = _architecture_findings(repo_root, runtime_profile, current_large_modules)

    stage_profile_available = _runtime_profile_proves_stage_profile(runtime_profile)
    batch_read_profile_proven = _batch_read_profile_proves_batch_path(runtime_profile)
    reader_cache = _reader_cache_evidence(repo_root)
    market_report_split = _market_report_split_evidence(repo_root)
    strategy_profile_split = _strategy_profile_split_evidence(repo_root)

    return {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": "current_rebaseline",
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "root": str(repo_root),
        "scope": (
            "read-only architecture/performance rebaseline for quant-investor "
            "market run/analyze"
        ),
        "mutation_status": {
            "source_edits": False,
            "data_deletions": False,
            "read_only_summary": True,
        },
        "summary": {
            "primary_finding_count": len(findings),
            "large_module_candidate_count": len(current_large_modules),
            "stage_profile_available": stage_profile_available,
            "batch_read_profile_proven": batch_read_profile_proven,
            "reader_cache_evidence": reader_cache,
            "market_report_split_evidence": market_report_split,
            "strategy_profile_split_evidence": strategy_profile_split,
        },
        "primary_findings": findings,
        "large_module_candidates": current_large_modules,
        "performance_evidence": {
            "latest_profile": _runtime_profile_summary(
                runtime_profile,
                str(runtime_profile_path) if runtime_profile_path else None,
            ),
            "slowest_stages": _slowest_stages(runtime_profile or {}),
        },
        "architecture_evidence": {
            "reader_cache_evidence": reader_cache,
            "market_report_split_evidence": market_report_split,
            "strategy_profile_split_evidence": strategy_profile_split,
        },
    }


def render_architecture_rebaseline_markdown(audit: dict[str, Any]) -> str:
    """Render a compact Markdown architecture rebaseline report."""
    summary = audit.get("summary", {})
    latest_profile = (audit.get("performance_evidence") or {}).get("latest_profile") or {}
    lines = [
        "# myQuant Current Architecture Rebaseline",
        "",
        f"- Schema: `{audit.get('schema_version', '')}`",
        f"- Generated at: `{audit.get('generated_at', '')}`",
        f"- Root: `{audit.get('root', '')}`",
        "- Mode: read-only summary; no source edits or data deletions",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Performance Evidence",
            "",
            f"- Latest profile: `{latest_profile.get('source_json')}`",
            f"- Total seconds: `{latest_profile.get('total_seconds')}`",
            f"- Stage count: `{latest_profile.get('stage_count')}`",
            (
                "- Per-symbol fallback count: "
                f"`{latest_profile.get('per_symbol_fallback_count')}`"
            ),
            "",
            "## Findings",
            "",
        ]
    )
    findings = audit.get("primary_findings", [])
    if not findings:
        lines.append("- No current architecture/performance findings under this audit.")
    for finding in findings:
        lines.append(
            "- `{id}` [{severity}] {title}".format(
                id=finding.get("id", ""),
                severity=finding.get("severity", ""),
                title=finding.get("title", ""),
            )
        )
    lines.extend(["", "## Large Modules", ""])
    candidates = audit.get("large_module_candidates", [])
    if not candidates:
        lines.append("- No current modules above the large-module threshold.")
    for candidate in candidates:
        lines.append(
            "- `{path}`: `{lines}` lines".format(
                path=candidate.get("path", ""),
                lines=candidate.get("lines", ""),
            )
        )
    return "\n".join(lines) + "\n"


def write_architecture_rebaseline_audit(
    root: Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write JSON and Markdown architecture rebaseline reports."""
    repo_root = get_repo_root(root)
    audit = build_architecture_rebaseline_audit(repo_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir) if output_dir is not None else (
        repo_root
        / "reports"
        / "project_cleanup"
        / f"architecture_rebaseline_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "architecture_performance_audit.json"
    md_path = out_dir / "architecture_performance_audit.md"
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        render_architecture_rebaseline_markdown(audit),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "md": str(md_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a read-only current architecture rebaseline report."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown architecture rebaseline reports.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = get_repo_root(args.root)
    paths = write_architecture_rebaseline_audit(
        repo_root,
        output_dir=args.output_dir,
    )
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    print("architecture rebaseline status:")
    print(f"workspace root: {payload['root']}")
    print(f"primary_findings: {payload['summary']['primary_finding_count']}")
    print(f"large_module_candidates: {payload['summary']['large_module_candidate_count']}")
    print("architecture rebaseline manifest:")
    print(f"  - json: {paths['json']}")
    print(f"  - md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
