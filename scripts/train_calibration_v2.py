#!/usr/bin/env python3
"""Train offline Bayesian calibration V2 artifacts from a local outcome ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.bayesian.calibration_v2 import CalibrationV2Store  # noqa: E402
from quant_investor.bayesian.outcome_ledger import OutcomeLedgerStore  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train offline Bayesian calibration V2 artifacts.")
    parser.add_argument("--ledger-dir", default=None, help="Directory containing predictions.jsonl and outcomes.jsonl.")
    parser.add_argument("--output-dir", default=None, help="Directory where model/report JSON files are written.")
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--min-examples-per-curve", type=int, default=30)
    parser.add_argument("--min-abs-return", type=float, default=None)
    parser.add_argument("--include-branches", dest="include_branches", action="store_true", default=True)
    parser.add_argument("--no-include-branches", dest="include_branches", action="store_false")
    parser.add_argument("--include-posterior", dest="include_posterior", action="store_true", default=True)
    parser.add_argument("--no-include-posterior", dest="include_posterior", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ledger_store = OutcomeLedgerStore(args.ledger_dir)
    calibration_store = CalibrationV2Store(args.output_dir)
    model, report = calibration_store.train_from_ledger(
        ledger_store,
        bucket_count=args.bucket_count,
        prior_strength=args.prior_strength,
        min_examples_per_curve=args.min_examples_per_curve,
        include_posterior=args.include_posterior,
        include_branches=args.include_branches,
        min_abs_return=args.min_abs_return,
        metadata={"source": "scripts/train_calibration_v2.py"},
    )

    print(f"total_examples: {report.total_examples}")
    print(f"curves: {len(model.curves)}")
    print(f"model_path: {calibration_store.model_path}")
    print(f"report_path: {calibration_store.report_path}")
    for summary in report.metric_summaries:
        print(
            "metric: "
            f"target={summary.target_name} "
            f"n={summary.example_count} "
            f"base_rate={summary.base_rate} "
            f"raw_brier={summary.raw_brier_score} "
            f"calibrated_brier={summary.calibrated_brier_score}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
