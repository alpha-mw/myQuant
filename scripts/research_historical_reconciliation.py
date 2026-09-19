"""Per-day reconciliation for the historical sample, and evaluability accounting.

Two questions are answered separately because they are separate questions:

  * on which dates was an identity a member, and why not on the others;
  * on which of those dates could a return actually be evaluated afterwards.

A day whose return cannot be evaluated is reported as such. It is never recorded
as a zero return and never dropped from the series — either would turn "we do
not know" into "nothing happened".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_historical_consumption import build_sample_inputs  # noqa: E402

from quant_investor.factors.backtest import (  # noqa: E402
    build_execution_return_matrix,
    build_quantile_weight_matrix,
    compute_daily_backtest_records,
)
from quant_investor.factors.matrix import FIELD_CLOSE  # noqa: E402
from quant_investor.factors.schema import FactorBacktestConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()

    inputs = build_sample_inputs(root)
    symbols, iso_dates = inputs["symbols"], inputs["iso_dates"]
    bundle, factor = inputs["bundle"], inputs["factor"]
    mask, reasons = inputs["universe_mask"], inputs["reasons"]
    panel = inputs["panel"]
    panel_dates = {
        symbol: set(group["trade_date"])
        for symbol, group in panel.groupby("ts_code")
    }

    config = FactorBacktestConfig(
        config_id="research-historical-reconciliation",
        quantile_count=2,
        long_quantile=1,
        long_short=False,
        selection_requires_future_return=False,
    )
    weights = build_quantile_weight_matrix(factor, bundle, config)
    returns = build_execution_return_matrix(
        bundle, execution_price=config.execution_price, holding_period_days=1
    )
    closes = bundle.fields[FIELD_CLOSE]

    per_symbol = {}
    for row, symbol in enumerate(symbols):
        buckets = {
            "research_eligible_days": 0,
            "not_research_eligible": 0,
            "research_eligible_no_bar_that_day": 0,
            "research_eligible_priced_no_fundamental_row": 0,
            "research_eligible_priced_with_fundamental_row": 0,
            "research_eligible_priced_return_not_evaluable": 0,
            "weighted_days": 0,
            "weighted_but_return_not_evaluable": 0,
        }
        not_member_reasons: dict[str, int] = {}
        examples: list[dict] = []
        for column, iso in enumerate(iso_dates):
            # The mart panel stores ISO dates; comparing against a compact
            # YYYYMMDD key here matches nothing and silently reports every
            # priced day as lacking a fundamental row.
            panel_key = iso
            if not mask[row][column]:
                buckets["not_research_eligible"] += 1
                key = reasons[row][column]
                not_member_reasons[key] = not_member_reasons.get(key, 0) + 1
                continue
            buckets["research_eligible_days"] += 1
            priced = closes[row][column] is not None
            has_panel_row = panel_key in panel_dates.get(symbol, set())
            if not priced:
                buckets["research_eligible_no_bar_that_day"] += 1
                if len(examples) < 4:
                    examples.append(
                        {
                            "date": iso,
                            "classification": "research_eligible_no_bar_that_day",
                            "has_fundamental_row": has_panel_row,
                        }
                    )
                continue
            if has_panel_row:
                buckets["research_eligible_priced_with_fundamental_row"] += 1
            else:
                buckets["research_eligible_priced_no_fundamental_row"] += 1
            execution_index = column + config.delay_days
            evaluable = (
                execution_index < len(iso_dates)
                and returns[row][execution_index] is not None
            )
            if not evaluable:
                buckets["research_eligible_priced_return_not_evaluable"] += 1
            if (weights.long_weights[row][column] or 0.0) != 0.0:
                buckets["weighted_days"] += 1
                if not evaluable:
                    buckets["weighted_but_return_not_evaluable"] += 1
        buckets["not_research_eligible_reasons"] = not_member_reasons
        buckets["panel_rows"] = int((panel["ts_code"] == symbol).sum())
        buckets["examples"] = examples
        # The identity that the reconciliation has to close.
        buckets["research_eligible_days_minus_panel_rows"] = (
            buckets["research_eligible_days"] - buckets["panel_rows"]
        )
        buckets["difference_accounted_by"] = (
            buckets["research_eligible_no_bar_that_day"]
            + buckets["research_eligible_priced_no_fundamental_row"]
        )
        per_symbol[symbol] = buckets

    # Alignment consumes the tail: a signal date needs delay_days + holding
    # ahead of it, so the final dates cannot produce a record at all. That is a
    # window boundary, not a dropped observation, and is reported as such.
    expected_missing = config.delay_days + 1
    records = compute_daily_backtest_records(
        factor, bundle, config, weights, mode="long_only", holding_period_days=1
    )
    none_return_days = sum(1 for r in records if r.long_return is None)
    zero_return_days = sum(1 for r in records if r.long_return == 0.0)
    # Performance conclusions are withheld rather than computed over whatever
    # survives. quant_investor/factors/metrics.py drops null returns
    # (filter_none_finite) and then annualises with a fixed 252 factor, so an
    # incomplete series yields a complete-looking Sharpe as if the missing days
    # had earned the mean. With unevaluable days present and exit accounting
    # absent, any NAV, cumulative return or Sharpe from this sample would be
    # stating more than the data supports.
    evaluable_days = len(records) - none_return_days
    performance = {
        "evaluable_days": evaluable_days,
        "unevaluable_days": none_return_days,
        "records_emitted": len(records),
        "nav_reported": False,
        "cumulative_return_reported": False,
        "sharpe_reported": False,
        "withheld_because": [
            "unevaluable days present: aggregating over surviving days only "
            "would annualise a gapped series as if it were complete "
            "(metrics.annualized_return_from_daily = mean(non-null) * 252)",
            "exit accounting absent: a holding that loses eligibility is never "
            "sold, so realised delisting loss is missing from any return series",
        ],
    }
    report = {
        "schema_version": "cn-research-validation.v1",
        "performance_conclusions": performance,
        "research_only": True,
        "selection_requires_future_return": config.selection_requires_future_return,
        "per_symbol": per_symbol,
        "daily_records": {
            "dates_in_axis": len(iso_dates),
            "records_emitted": len(records),
            "days_with_unevaluable_return": none_return_days,
            "days_recorded_as_exactly_zero_return": zero_return_days,
            "dates_without_a_record": len(iso_dates) - len(records),
            "dates_without_a_record_expected_from_alignment": expected_missing,
            "alignment_accounts_for_all_missing_records": (
                len(iso_dates) - len(records) == expected_missing
            ),
            "note": (
                "an unevaluable return is emitted as null and the day is kept; "
                "it is not recorded as zero and not dropped"
            ),
        },
        "not_demonstrated": [
            "exit accounting: no exit price is derivable from these fields, so a "
            "holding that loses eligibility is not sold and no delisting loss is "
            "realised",
        ],
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
