#!/usr/bin/env python3
"""Collect offline multi-date factor shadow evidence from local artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_investor.factors.evidence import (  # noqa: E402
    FactorEvidenceCollectionConfig,
    FactorEvidenceDateInput,
    build_factor_evidence_dashboard_payload,
    build_multi_date_factor_evidence_report,
    render_multi_date_evidence_markdown,
)
from quant_investor.factors.store import FactorEvidenceStore  # noqa: E402


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed input manifest {path}: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Input manifest must be a JSON object: {path}")
    return dict(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-observation-days", type=int, default=20)
    parser.add_argument("--min-average-factor-coverage", type=float, default=0.80)
    parser.add_argument("--min-top-n-overlap-ratio", type=float, default=0.50)
    parser.add_argument(
        "--require-library-audit-no-blocker",
        dest="require_library_audit_no_blocker",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-require-library-audit-no-blocker",
        dest="require_library_audit_no_blocker",
        action="store_false",
    )
    parser.add_argument("--require-alignment-audit-pass", action="store_true")
    parser.add_argument("--require-tradability-audit-pass", action="store_true")
    parser.add_argument("--require-execution-cost-review", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = _load_manifest(args.input_manifest)
    date_inputs = [
        FactorEvidenceDateInput.from_dict(item)
        for item in manifest.get("date_inputs", [])
        if isinstance(item, Mapping)
    ]
    as_of_dates = list(manifest.get("as_of_dates", []) or [item.as_of for item in date_inputs])
    config = FactorEvidenceCollectionConfig(
        as_of_dates=as_of_dates,
        top_n=args.top_n,
        min_observation_days=args.min_observation_days,
        min_average_factor_coverage=args.min_average_factor_coverage,
        min_top_n_overlap_ratio=args.min_top_n_overlap_ratio,
        require_library_audit_no_blocker=args.require_library_audit_no_blocker,
        require_alignment_audit_pass=args.require_alignment_audit_pass,
        require_tradability_audit_pass=args.require_tradability_audit_pass,
        require_execution_cost_review=args.require_execution_cost_review,
        metadata=dict(manifest.get("metadata", {}) or {}),
    )
    report = build_multi_date_factor_evidence_report(
        date_inputs=date_inputs,
        config=config,
        generated_at=args.generated_at,
        metadata={"input_manifest": str(args.input_manifest)},
    )
    store = FactorEvidenceStore(args.output_dir)
    for result in report.date_results:
        store.append_date_result(result)
    store.append_multi_date_report(report)
    markdown_path = store.save_evidence_markdown(render_multi_date_evidence_markdown(report))
    dashboard_path = store.save_evidence_dashboard(build_factor_evidence_dashboard_payload(report))

    print(f"status: {report.status}")
    print(f"observation_days: {report.observation_days}")
    print(f"start/end: {report.start_date or ''}/{report.end_date or ''}")
    print(f"average_overlap: {report.average_top_n_overlap_ratio}")
    print(f"average_coverage: {report.average_factor_coverage_ratio}")
    print(f"warning_codes: {', '.join(report.warning_codes)}")
    print(f"date_results_path: {store.date_results_path}")
    print(f"multi_date_reports_path: {store.multi_date_reports_path}")
    print(f"markdown_path: {markdown_path}")
    print(f"dashboard_path: {dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
