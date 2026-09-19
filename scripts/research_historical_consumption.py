"""Consume a research historical generation through the factor backtest path.

Deliberately minimal and research-only: it wires one isolated fundamental
generation into a `MatrixDataBundle` and runs the existing quantile weighting,
so that a historical identity can be shown to take part in the computation on
the dates it was actually a member — and to drop out afterwards. It is not a
general backtest platform and must not grow into one.

What it is meant to demonstrate, and what it deliberately does not claim:
computing weights is not the same as accounting for an exit. A holding that
becomes ineligible here simply stops being selected; no exit price is booked,
because none can be derived from the fields this bundle carries.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from quant_investor.factors.backtest import build_quantile_weight_matrix
from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_CLOSE,
    FIELD_VOLUME,
    FactorMatrix,
    MatrixDataBundle,
    MatrixDataContract,
)
from quant_investor.factors.schema import FactorBacktestConfig
from quant_investor.market.pit_universe import (
    PITUniverseRecord,
    build_pit_universe_mask_with_reasons,
)


def _provider():
    import tushare as ts

    from quant_investor import config as C
    from quant_investor.config import TUSHARE_OFFICIAL_URL

    token = os.environ.get("TUSHARE_TOKEN") or getattr(C, "TUSHARE_TOKEN", "")
    pro = ts.pro_api(token, timeout=60)
    pro._DataApi__http_url = TUSHARE_OFFICIAL_URL
    return pro


def _iso(compact: str) -> str:
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"


def build_sample_inputs(root: Path) -> dict:
    """Assemble the bundle inputs once and cache the provider prices.

    The cache exists so that repeated analyses of the same sample compare the
    same numbers instead of re-fetching and quietly drifting.
    """
    generation = sorted((root / "staging" / "_fundamental_generations").iterdir())[0]
    panel = pd.read_parquet(generation / "fundamental_daily.parquet")
    panel["trade_date"] = panel["trade_date"].astype(str)

    membership = pd.read_parquet(
        json.loads((root / "parquet" / "cn" / "_latest.json").read_text())["coverage"][
            "pit_membership_path"
        ]
    )
    records = [
        PITUniverseRecord(
            symbol=str(row["symbol"]),
            source_list_status=str(row["source_list_status"]),
            list_date=str(row["list_date"]),
            delist_date=str(row["delist_date"] or ""),
            effective_from=str(row["effective_from"]),
            effective_to=str(row["effective_to"] or ""),
        )
        for _, row in membership.iterrows()
    ]

    # The contract requires sorted symbols and strictly ascending ISO dates; the
    # mart speaks YYYYMMDD, so both axes are normalised here rather than assumed.
    # The symbol axis comes from the membership, not from the panel: an
    # identity that was a member and never traded has no panel row, and taking
    # the axis from the panel would drop it silently — the exact omission this
    # sample exists to make visible.
    symbols = sorted(str(value) for value in membership["symbol"])
    dates = sorted(panel["trade_date"].unique())
    iso_dates = [_iso(d.replace("-", "")) for d in dates]

    cache_path = root / "provider_price_cache.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["trade_date"] = cached["trade_date"].astype(str)
        prices = {
            symbol: group.set_index("trade_date")
            for symbol, group in cached.groupby("ts_code")
        }
    else:
        pro = _provider()
        collected = []
        for symbol in symbols:
            frame = pro.daily(
                ts_code=symbol,
                start_date=dates[0].replace("-", ""),
                end_date=dates[-1].replace("-", ""),
            )
            frame["trade_date"] = frame["trade_date"].astype(str)
            if len(frame):
                collected.append(
                    frame[["ts_code", "trade_date", "close", "vol", "amount"]]
                )
        cached = (
            pd.concat(collected, ignore_index=True)
            if collected
            else pd.DataFrame(
                columns=["ts_code", "trade_date", "close", "vol", "amount"]
            )
        )
        cached.to_parquet(cache_path, index=False)
        prices = {
            symbol: group.set_index("trade_date")
            for symbol, group in cached.groupby("ts_code")
        }

    def _grid(column: str) -> list[list[float | None]]:
        rows: list[list[float | None]] = []
        for symbol in symbols:
            frame = prices.get(symbol)
            if frame is None:
                # A member with no trades at all: every cell stays empty. A zero
                # here would be a fabricated price.
                rows.append([None] * len(dates))
                continue
            row: list[float | None] = []
            for compact in (d.replace("-", "") for d in dates):
                if compact in frame.index:
                    value = frame.at[compact, column]
                    row.append(None if pd.isna(value) else float(value))
                else:
                    # No bar means no price. Leaving it None is the point: a
                    # zero here would be a fabricated trade at zero.
                    row.append(None)
            rows.append(row)
        return rows

    factor_grid: list[list[float | None]] = []
    pivot = panel.pivot_table(
        index="ts_code", columns="trade_date", values="fin_roe", aggfunc="last"
    )
    for symbol in symbols:
        factor_grid.append(
            [
                None
                if symbol not in pivot.index or pd.isna(pivot.at[symbol, d])
                else float(pivot.at[symbol, d])
                for d in dates
            ]
        )

    universe_mask, reasons = build_pit_universe_mask_with_reasons(
        symbols, iso_dates, records, required=True
    )

    contract = MatrixDataContract(
        contract_id="research-historical-sample",
        universe="full_a_hist",
        symbols=list(symbols),
        dates=list(iso_dates),
        required_fields=[FIELD_CLOSE, FIELD_VOLUME, FIELD_AMOUNT],
        metadata={"research_only": True, "tradable_universe": False},
    )
    bundle = MatrixDataBundle(
        bundle_id="research-historical-sample-bundle",
        contract=contract,
        fields={
            FIELD_CLOSE: _grid("close"),
            FIELD_VOLUME: _grid("vol"),
            FIELD_AMOUNT: _grid("amount"),
        },
        universe_mask=universe_mask,
        # Reasons cannot ride inside the bool mask, so they are carried here.
        metadata={"universe_mask_reasons": reasons, "research_only": True},
    )
    factor = FactorMatrix(
        matrix_id="research-historical-fin-roe",
        expression="fin_roe",
        symbols=list(symbols),
        dates=list(iso_dates),
        values=factor_grid,
    )
    return {
        "generation": generation,
        "panel": panel,
        "records": records,
        "symbols": symbols,
        "dates": dates,
        "iso_dates": iso_dates,
        "prices": prices,
        "grid": _grid,
        "bundle": bundle,
        "factor": factor,
        "universe_mask": universe_mask,
        "reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    inputs = build_sample_inputs(root)
    generation = inputs["generation"]
    panel = inputs["panel"]
    symbols = inputs["symbols"]
    iso_dates = inputs["iso_dates"]
    bundle = inputs["bundle"]
    factor = inputs["factor"]
    universe_mask = inputs["universe_mask"]
    reasons = inputs["reasons"]

    # quantile_count must not exceed the number of eligible symbols on a date, or
    # the top quantile is unreachable and every weight is silently zero. With a
    # sample this small the only safe setting is 2, and long_short is off because
    # a short quantile would be required otherwise.
    config = FactorBacktestConfig(
        config_id="research-historical-sample-config",
        quantile_count=2,
        long_quantile=2,
        long_short=False,
    )
    weights = build_quantile_weight_matrix(factor, bundle, config)

    # A second configuration selecting the *bottom* quantile, on identical data.
    # With fin_roe the two banks dominate the top half on almost every date, so
    # the top-quantile run alone cannot show a delisted identity receiving
    # weight — only that it was eligible. Ranking is participation either way;
    # this makes the participation visible rather than argued.
    bottom_config = FactorBacktestConfig(
        config_id="research-historical-sample-config-bottom",
        quantile_count=2,
        long_quantile=1,
        long_short=False,
    )
    bottom_weights = build_quantile_weight_matrix(factor, bundle, bottom_config)

    eligible_per_date = [
        sum(universe_mask[i][j] for i in range(len(symbols))) for j in range(len(iso_dates))
    ]
    nonzero_per_date = [
        sum(1 for i in range(len(symbols)) if (weights.long_weights[i][j] or 0.0) != 0.0)
        for j in range(len(iso_dates))
    ]
    violations = [
        {"date": iso_dates[j], "eligible": eligible_per_date[j]}
        for j in range(len(iso_dates))
        if eligible_per_date[j] >= 2 and nonzero_per_date[j] == 0
    ]

    per_symbol = {}
    for i, symbol in enumerate(symbols):
        eligible_dates = [iso_dates[j] for j in range(len(iso_dates)) if universe_mask[i][j]]
        weighted_dates = [
            iso_dates[j]
            for j in range(len(iso_dates))
            if (weights.long_weights[i][j] or 0.0) != 0.0
        ]
        per_symbol[symbol] = {
            "panel_rows": int((panel["ts_code"] == symbol).sum()),
            "research_eligible_days": len(eligible_dates),
            "research_eligible_first": eligible_dates[0] if eligible_dates else "",
            "research_eligible_last": eligible_dates[-1] if eligible_dates else "",
            "days_with_nonzero_long_weight": len(weighted_dates),
            "weighted_first": weighted_dates[0] if weighted_dates else "",
            "weighted_last": weighted_dates[-1] if weighted_dates else "",
            "distinct_exclusion_reasons": sorted({r for r in reasons[i]}),
        }
        bottom_dates = [
            iso_dates[j]
            for j in range(len(iso_dates))
            if (bottom_weights.long_weights[i][j] or 0.0) != 0.0
        ]
        per_symbol[symbol]["bottom_quantile_weighted_days"] = len(bottom_dates)
        per_symbol[symbol]["bottom_quantile_first"] = bottom_dates[0] if bottom_dates else ""
        per_symbol[symbol]["bottom_quantile_last"] = bottom_dates[-1] if bottom_dates else ""
        # Nothing may be weighted on a date the identity was not a member.
        per_symbol[symbol]["weighted_while_not_research_eligible"] = sum(
            1
            for j in range(len(iso_dates))
            if not universe_mask[i][j]
            and (
                (weights.long_weights[i][j] or 0.0) != 0.0
                or (bottom_weights.long_weights[i][j] or 0.0) != 0.0
            )
        )

    report = {
        "schema_version": "cn-research-validation.v1",
        "research_only": True,
        "promotable": False,
        "generation": generation.name,
        "symbols": symbols,
        "date_range": [iso_dates[0], iso_dates[-1]],
        "date_count": len(iso_dates),
        "backtest_config": {
            "quantile_count": config.quantile_count,
            "long_quantile": config.long_quantile,
            "long_short": config.long_short,
        },
        "eligible_count_distribution": {
            str(k): eligible_per_date.count(k) for k in sorted(set(eligible_per_date))
        },
        "dates_with_eligible_ge_2_but_no_weight": violations,
        "dates_with_eligible_ge_2_but_no_weight_detail": [
            {
                "date": iso_dates[j],
                "eligible_symbols": [
                    symbols[i] for i in range(len(symbols)) if universe_mask[i][j]
                ],
                "eligible_with_price": [
                    symbols[i]
                    for i in range(len(symbols))
                    if universe_mask[i][j]
                    and bundle.fields[FIELD_CLOSE][i][j] is not None
                ],
            }
            for j in range(len(iso_dates))
            if eligible_per_date[j] >= 2 and nonzero_per_date[j] == 0
        ],
        "per_symbol": per_symbol,
        "not_demonstrated": [
            "exit accounting: no exit price is derivable from these fields, so a "
            "holding that becomes ineligible is not sold at a price and no "
            "delisting loss is realised",
        ],
    }
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
