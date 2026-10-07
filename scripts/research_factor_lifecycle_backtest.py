#!/usr/bin/env python3
"""Choose the IC estimation half-life and replay lifecycle rules on history.

Reads the canonical table partitions of one frozen CN Market snapshot (bound by
manifest path and SHA-256), builds month-end RankIC series for the candidate
batch, the two bootstrap factors and a few reference families, then:

* scores half-lives by one-step-ahead IC prediction;
* replays a grid of lifecycle rules against a static equal-weight set and
  deflates the best rule by the grid size;
* attributes IC to size, liquidity and market regimes, named event windows,
  and a value-spread crowding proxy.

Writes write-once JSON and Markdown under ``reports/factor_lifecycle/`` and a
priors file for ``research_factor_lifecycle_monitor.py``.  This is research
evidence only: no pointer, weight, admission or activation is touched.

    uv run python scripts/research_factor_lifecycle_backtest.py --workspace-root .
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors import lifecycle_backtest as lb  # noqa: E402
from quant_investor.factors.governance.bootstrap import BLEND_W80, LOW_DOLLAR_VOLUME  # noqa: E402
from quant_investor.factors.governance.statistics import effective_trial_count  # noqa: E402
from quant_investor.factors.lifecycle_candidates import (  # noqa: E402
    CANDIDATE_BATCH_ID,
    PRICE_VOLUME_CANDIDATES,
    candidate_panel,
)

COLUMNS = ["trade_date", "ts_code", "adj_close", "amount", "vol", "total_mv", "pb"]
HALF_LIVES = (1.0, 2.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0, math.inf)
EVENTS = {
    "2015_crash": ("2015-06-01", "2015-09-30"),
    "2017_large_cap_rotation": ("2017-01-01", "2017-12-31"),
    "2021_small_cap_rally": ("2021-01-01", "2021-12-31"),
    "2024_microcap_crash": ("2024-01-01", "2024-02-29"),
    "2024_new_nine_articles": ("2024-04-01", "2024-06-30"),
}
REFERENCE_FAMILIES = ("pv_momentum_60d", "pv_size", "pv_amihud_20d")
MAX_FACTOR_WEIGHT = 0.35


def _capped_robustness(ic: pd.DataFrame, warmup: int, *, cap: float) -> dict[str, Any]:
    """Same grid and fair baseline with a per-factor weight cap.

    Separates rule value that comes from timing decay from value that comes
    from concentrating into the single strongest factor.
    """

    rules = lb.rule_grid(max_factor_weight=(cap,))
    fair = lb.expanding_positive_weighted(ic, warmup_periods=warmup, max_factor_weight=cap)
    fair_summary = lb.summarize_ic_series(fair)
    rows = []
    for rule in rules:
        series = lb.simulate_lifecycle(ic, rule)["portfolio_ic"].copy()
        series.iloc[:warmup] = math.nan
        rows.append(
            {
                "rule": asdict_rule(rule),
                "summary": lb.summarize_ic_series(series),
                "excess_vs_fair": lb.summarize_ic_series(series - fair),
            }
        )
    by_half_life: dict[str, list[float]] = {}
    for row in rows:
        by_half_life.setdefault(str(row["rule"]["half_life_periods"]), []).append(
            row["excess_vs_fair"]["mean"]
        )
    best = max(rows, key=lambda row: row["excess_vs_fair"]["ir"] or -math.inf)
    return {
        "max_factor_weight": cap,
        "trial_count": len(rules),
        "fair_capped": fair_summary,
        "best_excess_rule": best,
        "excess_deflation": lb.deflated_best_rule([row["excess_vs_fair"] for row in rows]),
        "rules_beating_fair_capped_ir": sum(
            1 for row in rows if (row["summary"]["ir"] or -math.inf) > (fair_summary["ir"] or 0)
        ),
        "mean_excess_by_half_life": {
            key: float(np.mean(items)) for key, items in sorted(by_half_life.items())
        },
    }


def asdict_rule(rule: lb.LifecycleRule) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(rule)


def _frozen_snapshot_ref(workspace: Path) -> dict[str, str]:
    pointer = json.loads((workspace / "data/parquet/cn/_latest.json").read_text("utf-8"))
    manifest = Path(pointer["manifest_path"]).resolve(strict=True)
    return {
        "path": manifest.relative_to((workspace / "data").resolve()).as_posix(),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def _load_panels(
    workspace: Path, snapshot_ref: dict[str, str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    from quant_investor.market.market_data_reader import MarketDataReader

    reader = MarketDataReader(
        market="CN", data_root=workspace / "data", frozen_snapshot_ref=snapshot_ref
    )
    frames = [
        pd.read_parquet(path, columns=COLUMNS) for path in reader.table_partition_paths(start, end)
    ]
    bars = pd.concat(frames, ignore_index=True)
    bars = bars[(bars["trade_date"] >= start) & (bars["trade_date"] <= end)]
    bars["trade_date"] = pd.to_datetime(bars["trade_date"], format="%Y%m%d")
    if bars.duplicated(["trade_date", "ts_code"]).any():
        raise SystemExit("duplicate (trade_date, ts_code) rows in canonical table")
    panels = {}
    for column in COLUMNS[2:]:
        panels[column] = bars.pivot(index="trade_date", columns="ts_code", values=column).astype(
            "float32"
        )
    return panels


def _w80_panel(panels: dict[str, pd.DataFrame], origins: list[pd.Timestamp]) -> pd.DataFrame:
    volume = panels["vol"].astype(float)
    close = panels["adj_close"].astype(float)
    amount = panels["amount"].astype(float)
    mean = volume.rolling(19, min_periods=5).mean()
    std = volume.rolling(19, min_periods=5).std(ddof=0)
    stability = (-(std / mean.where(mean > 0.0))).rolling(2, min_periods=2).mean()
    momentum = close / close.shift(90) - 1.0
    returns = close.pct_change(fill_method=None).abs()
    amihud = (returns / amount.where(amount > 0.0)).rolling(5, min_periods=5).mean()
    stability_rank = stability.loc[origins].rank(axis=1, pct=True)
    inner = (
        momentum.loc[origins].rank(axis=1, pct=True) * 0.60
        + amihud.loc[origins].rank(axis=1, pct=True) * 0.40
    ).rank(axis=1, pct=True)
    return stability_rank * 0.80 + inner * 0.20


def _signals(
    panels: dict[str, pd.DataFrame], origins: list[pd.Timestamp]
) -> dict[str, pd.DataFrame]:
    close = panels["adj_close"].astype(float)
    amount = panels["amount"].astype(float)
    signals: dict[str, pd.DataFrame] = {}
    for factor_id in PRICE_VOLUME_CANDIDATES:
        signals[factor_id] = candidate_panel(factor_id, close).loc[origins]
    signals[LOW_DOLLAR_VOLUME] = -np.log(
        amount.where(amount > 0.0).rolling(5, min_periods=5).mean()
    ).loc[origins]
    signals[BLEND_W80] = _w80_panel(panels, origins)
    signals["pv_momentum_60d"] = (close / close.shift(60) - 1.0).loc[origins]
    size = panels["total_mv"].astype(float)
    signals["pv_size"] = -np.log(size.where(size > 0.0)).loc[origins]
    returns = close.pct_change(fill_method=None).abs()
    signals["pv_amihud_20d"] = (
        (returns / amount.where(amount > 0.0)).rolling(20, min_periods=20).mean().loc[origins]
    )
    return signals


def _universe(panels: dict[str, pd.DataFrame], origins: list[pd.Timestamp]) -> pd.DataFrame:
    close = panels["adj_close"]
    seasoned = close.notna().rolling(120, min_periods=120).sum().loc[origins] >= 120
    traded = (panels["amount"].loc[origins] > 0) & (panels["total_mv"].loc[origins] > 0)
    return seasoned & traded


def _regimes(
    panels: dict[str, pd.DataFrame], labels: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    size = panels["total_mv"].loc[labels.index].where(universe)
    rows = {}
    previous_liquidity = None
    for origin in labels.index:
        label = labels.loc[origin]
        sizes = size.loc[origin].dropna()
        joined = pd.concat([sizes, label], axis=1, join="inner").dropna()
        if len(joined) < 90:
            continue
        terciles = pd.qcut(joined.iloc[:, 0].rank(method="first"), 3, labels=False)
        small = joined.iloc[:, 1][terciles == 0].mean()
        large = joined.iloc[:, 1][terciles == 2].mean()
        liquidity = float(np.log(panels["amount"].loc[origin].where(universe.loc[origin]).median()))
        rows[origin] = {
            "size_spread_small_minus_large": float(small - large),
            "market_equal_weight_return": float(joined.iloc[:, 1].mean()),
            "liquidity_log_median_amount_change": (
                liquidity - previous_liquidity if previous_liquidity is not None else math.nan
            ),
        }
        previous_liquidity = liquidity
    return pd.DataFrame.from_dict(rows, orient="index")


def _crowding(
    signals: dict[str, pd.DataFrame], panels: dict[str, pd.DataFrame], ic: pd.DataFrame
) -> dict[str, Any]:
    """Value spread (top minus bottom quintile median log P/B) and its link to the next IC."""

    pb = panels["pb"].astype(float)
    result: dict[str, Any] = {}
    for factor, panel in signals.items():
        spreads = {}
        autocorr = []
        previous = None
        for origin in ic.index:
            row = panel.loc[origin].dropna()
            values = np.log(pb.loc[origin].where(pb.loc[origin] > 0.0)).reindex(row.index)
            joined = pd.concat([row, values], axis=1).dropna()
            if len(joined) >= 100:
                quintile = pd.qcut(joined.iloc[:, 0].rank(method="first"), 5, labels=False)
                spreads[origin] = float(
                    joined.iloc[:, 1][quintile == 4].median()
                    - joined.iloc[:, 1][quintile == 0].median()
                )
            if previous is not None:
                pair = pd.concat([previous, row], axis=1, join="inner").dropna()
                if len(pair) >= 100:
                    autocorr.append(float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())))
            previous = row
        spread = pd.Series(spreads, dtype=float)
        z = (spread - spread.expanding(24).mean()) / spread.expanding(24).std()
        pair = pd.concat([z, ic[factor]], axis=1, sort=True).dropna()
        result[factor] = {
            "value_spread_last": float(spread.iloc[-1]) if len(spread) else None,
            "value_spread_z_last": float(z.dropna().iloc[-1]) if z.notna().any() else None,
            "corr_value_spread_z_with_ic": (
                float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) >= 24 else None
            ),
            "mean_signal_rank_autocorrelation": float(np.mean(autocorr)) if autocorr else None,
        }
    return result


def _priors(ic: pd.DataFrame, factors: list[str]) -> dict[str, dict[str, float]]:
    priors = {}
    for factor in factors:
        data = ic[factor].dropna()
        if len(data) < 24:
            continue
        mean = float(data.mean())
        standard_error = float(data.std(ddof=1) / math.sqrt(len(data)))
        haircut = 0.5 * mean
        priors[factor] = {
            "prior_mean": haircut,
            "prior_sd": max(abs(haircut), standard_error, 0.01),
            "reference_ic": haircut,
            "historical_mean": mean,
            "historical_months": float(len(data)),
        }
    return priors


def _write_once(path: Path, raw: bytes) -> None:
    if path.exists() and path.read_bytes() != raw:
        raise SystemExit(f"refusing to replace a different file at {path}")
    path.write_bytes(raw)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Factor lifecycle backtest `{report['report_id']}`",
        "",
        "Research evidence only; grants no admission, weight or activation.",
        "",
        f"- Snapshot: `{report['market_snapshot_ref']['path']}`",
        f"- Origins: {report['ic_panel']['first_origin']} .. "
        f"{report['ic_panel']['last_origin']} ({report['ic_panel']['origin_count']} month-ends)",
        f"- Label: enter next close, hold {report['label_horizon_sessions']} sessions",
        "",
        "## Half-life selection (one-step-ahead IC prediction, months)",
        "",
        "| half-life | MSE | MSE first half | MSE second half | rank corr |",
        "|---|---|---|---|---|",
    ]
    for row in report["half_life_selection"]["results"]:
        lines.append(
            f"| {row['half_life_periods']} | {row['mse']:.6f} | {row['mse_first_half']:.6f} | "
            f"{row['mse_second_half']:.6f} | {row['mean_cross_sectional_rank_corr']:.3f} |"
        )
    selection = report["half_life_selection"]
    lines += [
        "",
        f"Best: {selection['best_half_life_periods']} "
        f"(first half {selection.get('best_first_half')}, "
        f"second half {selection.get('best_second_half')}).",
        "",
        "## Factor IC summary (monthly)",
        "",
        "| factor | mean IC | IR | t | hit rate |",
        "|---|---|---|---|---|",
    ]
    for factor, row in report["factor_summaries"].items():
        if row.get("mean") is None:
            continue
        lines.append(
            f"| `{factor}` | {row['mean']:.4f} | {row['ir']:.3f} | {row['t']:.2f} | "
            f"{row['hit_rate']:.2f} |"
        )
    lifecycle = report["lifecycle"]
    best = lifecycle["best_rule"]
    best_excess = lifecycle["best_excess_rule"]
    static = lifecycle["static_equal_weight"]
    fair = lifecycle["fair_expanding_positive_weighted"]

    def _line(label: str, row: dict[str, Any]) -> str:
        return (
            f"- {label}: mean {row['mean']:.4f}, IR {row['ir']:.3f}, "
            f"worst 12m {row['worst_12_period_sum']:.3f}, "
            f"max drawdown {row['max_drawdown_cumulative_ic']:.3f}"
        )

    lines += [
        "",
        "## Lifecycle rules vs baselines",
        "",
        f"- Rules tried: {lifecycle['trial_count']}",
        _line("Naive static equal weight (includes negative-IC factors)", static),
        _line("Fair baseline: weight by max(expanding mean IC, 0), no states", fair),
        _line("Best rule by IR", best["summary"]) + f", turnover {best['turnover']:.2f}",
        f"- Best rule parameters: `{json.dumps(best['rule'], sort_keys=True)}`",
        f"- Rules beating naive static IR: {lifecycle['rules_beating_static_ir']}; "
        f"beating fair baseline IR: {lifecycle['rules_beating_fair_ir']}",
        f"- Best excess over fair baseline: mean {best_excess['excess_vs_fair']['mean']:.4f}, "
        f"IR {best_excess['excess_vs_fair']['ir']:.3f}",
        f"- DSR of best excess over {lifecycle['excess_deflation']['trial_count']} trials: "
        f"{lifecycle['excess_deflation']['dsr']:.3f} (the comparable statistic)",
        "",
        "Mean excess over the fair baseline, averaged over the grid, by parameter:",
        "",
    ]
    for axis, groups in lifecycle["mean_excess_by_axis"].items():
        rendered = ", ".join(f"{key}: {value:+.4f}" for key, value in groups.items())
        lines.append(f"- `{axis}`: {rendered}")
    capped = lifecycle["capped_robustness"]
    capped_best = capped["best_excess_rule"]
    rendered = ", ".join(
        f"{key}: {value:+.4f}" for key, value in capped["mean_excess_by_half_life"].items()
    )
    lines += [
        "",
        f"### Robustness: per-factor weight cap {capped['max_factor_weight']}",
        "",
        _line("Fair baseline, capped", capped["fair_capped"]),
        f"- Rules beating capped fair IR: {capped['rules_beating_fair_capped_ir']} of "
        f"{capped['trial_count']}",
        f"- Best capped excess: mean {capped_best['excess_vs_fair']['mean']:.4f}, "
        f"IR {capped_best['excess_vs_fair']['ir']:.3f}; DSR over {capped['trial_count']} "
        f"trials {capped['excess_deflation']['dsr']:.3f}",
        f"- Mean capped excess by half-life: {rendered}",
    ]
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--start", default="20110601")
    parser.add_argument("--first-origin", default="2012-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/factor_lifecycle")
    args = parser.parse_args()

    workspace = Path(args.workspace_root).resolve(strict=True)
    snapshot_ref = _frozen_snapshot_ref(workspace)
    manifest = json.loads((workspace / "data" / snapshot_ref["path"]).read_text("utf-8"))
    end = args.end or str(manifest["latest_complete_trade_date"])
    panels = _load_panels(workspace, snapshot_ref, args.start, end)
    dates = list(panels["adj_close"].index)
    origins = [
        origin
        for origin in lb.month_end_origins(dates)
        if origin >= pd.Timestamp(args.first_origin)
        and dates.index(origin) + args.horizon + 1 < len(dates)
    ]
    universe = _universe(panels, origins)
    labels = lb.forward_label(panels["adj_close"].astype(float), horizon=args.horizon)
    labels = labels.loc[origins].where(universe)
    signals = {name: panel.where(universe) for name, panel in _signals(panels, origins).items()}
    ic = lb.ic_panel(signals, labels, origins)
    factors = list(ic.columns)

    selection = lb.select_half_life(ic, half_lives=HALF_LIVES, min_history=12)
    summaries = {factor: lb.summarize_ic_series(ic[factor]) for factor in factors}
    effective = effective_trial_count({factor: ic[factor] for factor in factors})

    rules = lb.rule_grid()
    warmup = max(rule.probation_periods for rule in rules)
    static_series = lb.static_equal_weight(ic, warmup_periods=warmup)
    static_summary = lb.summarize_ic_series(static_series)
    fair_series = lb.expanding_positive_weighted(ic, warmup_periods=warmup)
    fair_summary = lb.summarize_ic_series(fair_series)
    results = []
    for rule in rules:
        run = lb.simulate_lifecycle(ic, rule)
        series = run["portfolio_ic"].copy()
        series.iloc[:warmup] = math.nan
        results.append(
            {
                "rule": run["rule"],
                "summary": lb.summarize_ic_series(series),
                "excess_vs_fair": lb.summarize_ic_series(series - fair_series),
                "turnover": run["turnover"],
                "final_states": run["final_states"],
                "retirements": run["retirements"],
                "reentries": run["reentries"],
            }
        )
    deflation = lb.deflated_best_rule([row["summary"] for row in results])
    excess_deflation = lb.deflated_best_rule([row["excess_vs_fair"] for row in results])
    best = max(results, key=lambda row: row["summary"]["ir"] or -math.inf)
    best_excess = max(results, key=lambda row: row["excess_vs_fair"]["ir"] or -math.inf)
    beating = sum(
        1 for row in results if (row["summary"]["ir"] or -math.inf) > (static_summary["ir"] or 0)
    )
    beating_fair = sum(
        1 for row in results if (row["summary"]["ir"] or -math.inf) > (fair_summary["ir"] or 0)
    )
    by_axis: dict[str, dict[str, float]] = {}
    for axis in (
        "half_life_periods",
        "probation_periods",
        "watch_p",
        "retire_p",
        "reentry_cooldown",
    ):
        groups: dict[str, list[float]] = {}
        for row in results:
            groups.setdefault(str(row["rule"][axis]), []).append(row["excess_vs_fair"]["mean"])
        by_axis[axis] = {key: float(np.mean(items)) for key, items in sorted(groups.items())}
    capped = _capped_robustness(ic, warmup, cap=MAX_FACTOR_WEIGHT)

    regimes = _regimes(panels, labels, universe)
    attribution = lb.regime_attribution(ic, regimes, events=EVENTS)
    crowding = _crowding(signals, panels, ic)
    priors = _priors(ic, [LOW_DOLLAR_VOLUME, BLEND_W80])

    report: dict[str, Any] = {
        "schema_version": "factor-lifecycle-backtest.v1",
        "authority": "NON_AUTHORIZING_RESEARCH",
        "candidate_batch_id": CANDIDATE_BATCH_ID,
        "market_snapshot_ref": snapshot_ref,
        "label_horizon_sessions": args.horizon,
        "label_formula": "adj_close[t+1+h]/adj_close[t+1]-1",
        "universe": "seasoned_120_sessions_and_traded_with_positive_total_mv_at_origin",
        "ic_panel": {
            "first_origin": str(ic.index[0].date()),
            "last_origin": str(ic.index[-1].date()),
            "origin_count": int(len(ic)),
            "factors": factors,
            "monthly_ic": {
                factor: {
                    str(k.date()): (None if math.isnan(v) else v) for k, v in ic[factor].items()
                }
                for factor in factors
            },
        },
        "effective_trials": effective,
        "half_life_selection": selection,
        "factor_summaries": summaries,
        "lifecycle": {
            "trial_count": len(rules),
            "warmup_periods": warmup,
            "static_equal_weight": static_summary,
            "fair_expanding_positive_weighted": fair_summary,
            "rules": results,
            "best_rule": best,
            "best_excess_rule": best_excess,
            "deflation": deflation,
            "excess_deflation": excess_deflation,
            "rules_beating_static_ir": beating,
            "rules_beating_fair_ir": beating_fair,
            "mean_excess_by_axis": by_axis,
            "capped_robustness": capped,
        },
        "attribution": attribution,
        "crowding": crowding,
        "priors": priors,
        "limitations": [
            "Raw research replay; not prospective evidence and not an admission input.",
            "No ST, limit-up/down or suspension-at-entry filter beyond a traded-at-origin check.",
            "Universe is the snapshot's canonical bar inventory; PIT membership is not rebound.",
            "No transaction cost; IC-weighted portfolio IC measures signal quality, not return.",
            "Bootstrap formulas are replayed as research panels, not production signals.",
            "Rule-grid parameters are trials; only the deflated statistic is comparable.",
        ],
    }
    raw = json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    report_id = f"backtest-{snapshot_ref['path'].split('/')[-1][:-5]}-{digest[:12]}"
    report["report_id"] = report_id
    output_dir = (workspace / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_raw = json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    _write_once(output_dir / f"{report_id}.json", (final_raw + "\n").encode())
    _write_once(output_dir / f"{report_id}.md", _markdown(report).encode())
    priors_raw = json.dumps(
        {
            "schema_version": "factor-lifecycle-priors.v1",
            "source_report": f"{report_id}.json",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon_note": "monthly 20-session RankIC level; a 50% publication haircut",
            "priors": priors,
        },
        indent=1,
        sort_keys=True,
    )
    _write_once(output_dir / f"priors-{report_id}.json", (priors_raw + "\n").encode())
    print(
        json.dumps(
            {
                "report": str(output_dir / f"{report_id}.json"),
                "best_half_life_months": selection["best_half_life_periods"],
                "static_ir": static_summary["ir"],
                "fair_ir": fair_summary["ir"],
                "best_rule_ir": best["summary"]["ir"],
                "excess_dsr": excess_deflation["dsr"],
                "rules_beating_static": beating,
                "rules_beating_fair": beating_fair,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
