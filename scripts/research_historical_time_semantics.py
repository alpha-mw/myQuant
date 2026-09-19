"""Does a signal-day target weight depend on data that did not exist yet?

The check is a truncation invariance: hold every input up to a signal date
fixed, remove only the prices *after* it, and recompute. A weight for a date on
or before the cut must not move. Ex-post evaluability may legitimately become
unavailable — that is a different statement from the weight changing, and the
two must not be conflated.

Cuts are placed around delisting and around the last traded day, not only at the
end of the series, because that is where a forward-looking filter does its
damage: the identity stops having a future, and anything that consults that
future silently rewrites the past.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_historical_consumption import build_sample_inputs  # noqa: E402

from quant_investor.factors.backtest import build_quantile_weight_matrix  # noqa: E402
from quant_investor.factors.matrix import (  # noqa: E402
    FIELD_AMOUNT,
    FIELD_CLOSE,
    FIELD_VOLUME,
    MatrixDataBundle,
)
from quant_investor.factors.schema import FactorBacktestConfig  # noqa: E402


def _truncated_bundle(inputs: dict, cut_index: int) -> MatrixDataBundle:
    """Same bundle, with every price strictly after ``cut_index`` removed."""
    base = inputs["bundle"]
    fields = {}
    for name in (FIELD_CLOSE, FIELD_VOLUME, FIELD_AMOUNT):
        grid = base.fields[name]
        fields[name] = [
            [value if column <= cut_index else None for column, value in enumerate(row)]
            for row in grid
        ]
    return MatrixDataBundle(
        bundle_id=f"{base.bundle_id}-cut{cut_index}",
        contract=base.contract,
        fields=fields,
        universe_mask=base.universe_mask,
        metadata=dict(base.metadata),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--selection-requires-future-return",
        default="true",
        choices=["true", "false"],
        help="legacy behaviour filters candidates on a forward return being present",
    )
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    inputs = build_sample_inputs(root)
    symbols, iso_dates = inputs["symbols"], inputs["iso_dates"]
    factor = inputs["factor"]
    legacy = args.selection_requires_future_return == "true"

    def _weights(bundle):
        config = FactorBacktestConfig(
            config_id="research-time-semantics",
            quantile_count=2,
            long_quantile=1,
            long_short=False,
            selection_requires_future_return=legacy,
        )
        return build_quantile_weight_matrix(factor, bundle, config)

    baseline = _weights(inputs["bundle"])

    # Cut points chosen around the events, not at the tail.
    interesting = {
        "600311.SH_last_trade_2023-03-02": "2023-03-02",
        "600311.SH_pre_delist_2023-03-24": "2023-03-24",
        "000004.SZ_last_trade_2026-07-13": "2026-07-13",
        "000004.SZ_mid_window_2024-06-28": "2024-06-28",
        "series_tail_2026-09-04": "2026-09-04",
    }
    results = []
    for label, cut_date in interesting.items():
        if cut_date not in iso_dates:
            results.append({"cut": label, "status": "date_absent_from_axis"})
            continue
        cut_index = iso_dates.index(cut_date)
        truncated = _weights(_truncated_bundle(inputs, cut_index))
        changed = []
        for row_index, symbol in enumerate(symbols):
            for column in range(cut_index + 1):
                before = baseline.long_weights[row_index][column] or 0.0
                after = truncated.long_weights[row_index][column] or 0.0
                if abs(before - after) > 1e-12:
                    changed.append(
                        {
                            "symbol": symbol,
                            "date": iso_dates[column],
                            "baseline_weight": before,
                            "truncated_weight": after,
                            "days_before_cut": cut_index - column,
                        }
                    )
        results.append(
            {
                "cut": label,
                "cut_date": cut_date,
                "dates_at_or_before_cut": cut_index + 1,
                "changed_cells": len(changed),
                "first_changes": changed[:6],
                "verdict": "INVARIANT" if not changed else "SIGNAL_DAY_WEIGHT_MOVED",
            }
        )

    report = {
        "schema_version": "cn-research-validation.v1",
        "research_only": True,
        "selection_requires_future_return": legacy,
        "claim_under_test": (
            "target weights on a signal date must not change when only later "
            "prices change"
        ),
        "cuts": results,
        "overall": (
            "INVARIANT"
            if all(r.get("changed_cells", 0) == 0 for r in results)
            else "LOOKAHEAD_PRESENT"
        ),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
