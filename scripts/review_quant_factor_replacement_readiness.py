#!/usr/bin/env python3
"""Write an ignored, measurement-only Quant factor replacement review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_investor.factors.replacement_readiness import (  # noqa: E402
    assess_replacement_readiness,
)

JSON_FILENAME = "quant_factor_replacement_readiness.json"
MARKDOWN_FILENAME = "quant_factor_replacement_readiness.md"


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("replacement_readiness_%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read strict-fresh health and Quant selection-shadow evidence and emit "
            "proposal-only replacement readiness. This command cannot write a registry."
        )
    )
    parser.add_argument(
        "--health-json",
        action="append",
        required=True,
        help="Strict-fresh factor-health JSON; repeat for distinct matured windows.",
    )
    parser.add_argument(
        "--selection-shadow-json",
        action="append",
        default=[],
        help="Quant selection-shadow JSON; repeat when needed.",
    )
    parser.add_argument(
        "--observation-ledger",
        action="append",
        default=[],
        help="Selection-shadow JSONL observation ledger; repeat when needed.",
    )
    parser.add_argument("--candidate", default=None)
    parser.add_argument(
        "--factor",
        action="append",
        default=[],
        help="Limit review to one old production factor; repeat when needed.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Ignored report directory. Defaults under "
            "reports/factor_governance/replacement_readiness/."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.selection_shadow_json and not args.observation_ledger:
        parser.error("at least one --selection-shadow-json or --observation-ledger is required")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON input must be an object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw_line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(dict(payload))
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_markdown(payload: Mapping[str, Any]) -> str:
    candidate = payload.get("candidate", {})
    if not isinstance(candidate, Mapping):
        candidate = {}
    maturity = candidate.get("maturity", {})
    coverage = candidate.get("coverage", {})
    if not isinstance(maturity, Mapping):
        maturity = {}
    if not isinstance(coverage, Mapping):
        coverage = {}
    lines = [
        "# Quant Factor Replacement Readiness",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Measurement only: `{payload.get('measurement_only')}`",
        f"- Freeze policy: `{payload.get('freeze', {}).get('policy')}`",
        f"- Registry update: `{payload.get('registry_update_status')}`",
        f"- Candidate: `{candidate.get('name', '')}`",
        (
            "- Candidate maturity: "
            f"month-end=`{maturity.get('month_end_rankic_count')}`, "
            f"non-overlap 30d=`{maturity.get('nonoverlap_30d_cohort_count')}`, "
            f"passed=`{maturity.get('passed')}`"
        ),
        (
            f"- Candidate coverage: `{coverage.get('coverage_rate')}` "
            f"(passed=`{coverage.get('passed')}`)"
        ),
        (
            "- Covered/uncovered selection-bias review acceptable: "
            f"`{candidate.get('covered_uncovered_selection_bias_acceptable')}`"
        ),
        "",
        "## Factor decisions",
        "",
        "| Factor | Outcome | Distinct failures | Data-blocked ignored | Proposal blockers |",
        "|---|---|---:|---:|---|",
    ]
    for decision in payload.get("factor_decisions", []):
        blockers = ", ".join(decision.get("proposal_blockers", [])) or "-"
        lines.append(
            "| {factor} | `{outcome}` | {failures} | {blocked} | {blockers} |".format(
                factor=decision.get("factor_name", ""),
                outcome=decision.get("outcome", ""),
                failures=decision.get("distinct_matured_alpha_failure_count", 0),
                blocked=decision.get("data_blocked_window_count", 0),
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Fail-closed blockers",
            "",
        ]
    )
    lines.extend([f"- `{item}`" for item in payload.get("fail_closed_blockers", [])] or ["- None"])
    lines.extend(
        [
            "",
            "## Governance boundary",
            "",
            (
                "This report is measurement-only and freeze-bound. Outcomes ending in "
                "`_proposal` are review proposals, never registry actions. The runner has "
                "no registry path, write, promotion, or deprecation option."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    health_paths = [Path(value).expanduser() for value in args.health_json]
    shadow_paths = [Path(value).expanduser() for value in args.selection_shadow_json]
    ledger_paths = [Path(value).expanduser() for value in args.observation_ledger]
    health_reports = [_read_json(path) for path in health_paths]
    shadow_reports = [_read_json(path) for path in shadow_paths]
    observation_rows = [row for path in ledger_paths for row in _read_jsonl(path)]
    payload = assess_replacement_readiness(
        health_reports,
        shadow_reports,
        observation_rows,
        candidate_name=args.candidate,
        factor_names=args.factor or None,
    )
    payload["input_artifacts"] = {
        "health_json": [{"path": str(path), "sha256": _sha256(path)} for path in health_paths],
        "selection_shadow_json": [
            {"path": str(path), "sha256": _sha256(path)} for path in shadow_paths
        ],
        "observation_ledgers": [
            {"path": str(path), "sha256": _sha256(path)} for path in ledger_paths
        ],
    }
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path("reports/factor_governance/replacement_readiness") / _run_id()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_FILENAME
    markdown_path = output_dir / MARKDOWN_FILENAME
    payload["artifacts"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "gitignored_reports_root": True,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return payload, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload, output_dir = run(args)
    print(f"status={payload['status']}")
    print(f"measurement_only={payload['measurement_only']}")
    print(f"registry_update_status={payload['registry_update_status']}")
    print(f"output_dir={output_dir}")
    return 2 if payload["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
