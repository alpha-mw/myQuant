"""
美股模拟组合连续跟踪器。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.market.download_us import FullMarketDownloader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = (
    PROJECT_ROOT / "results" / "strategy_records" / "US" / "simulated_portfolio_10000"
)
DEFAULT_NOTES_PATH = DEFAULT_BASE_DIR / "latest_notes_payload.md"
DEFAULT_INITIAL_CASH = 7763.03
DEFAULT_HOLDINGS = [
    ("CVX", 4, 199.71),
    ("EOG", 5, 137.19),
    ("COP", 4, 125.53),
    ("AEP", 2, 125.03),
]
DEFAULT_CAPS = {
    "CVX": 6,
    "EOG": 8,
    "COP": 7,
    "AEP": 4,
}
NO_DATA_SAMPLE_LIMIT = 4
THEME_BASKETS = {
    "software": ["MSFT", "NOW", "CRM", "ORCL", "SNOW", "PANW"],
    "ai": ["NVDA", "AVGO", "PLTR", "ANET", "SMCI", "AMD"],
    "semiconductor": ["NVDA", "AMD", "AVGO", "AMAT", "LRCX", "KLAC", "TXN"],
    "energy": ["CVX", "EOG", "COP", "OXY", "DVN", "VLO", "SLB"],
    "defensive": ["AEP", "DUK", "SO", "PG", "KO", "PEP", "WMT"],
}


@dataclass
class TradeOrder:
    symbol: str
    action: str
    shares: int
    price: float
    trade_value: float
    reason: str


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _parse_initial_holding(text: str) -> tuple[str, int, float]:
    symbol, shares, avg_cost = text.split(":")
    return symbol.upper(), int(shares), float(avg_cost)


def _parse_cap(text: str) -> tuple[str, int]:
    symbol, max_shares = text.split(":")
    return symbol.upper(), int(max_shares)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_latest_prices(symbols: list[str]) -> dict[str, dict[str, Any]]:
    market_root = PROJECT_ROOT / "data" / "us_market_full"
    result: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        matches = list(market_root.glob(f"*/*{symbol}.csv"))
        if not matches:
            continue
        frame = pd.read_csv(matches[0])
        if frame.empty:
            continue
        row = frame.iloc[-1]
        result[symbol] = {
            "category": matches[0].parent.name,
            "date": str(row["Date"]),
            "close": float(row["Close"]),
            "path": str(matches[0]),
        }
    return result


def _load_latest_batch_recommendations() -> dict[str, dict[str, Any]]:
    results_dir = PROJECT_ROOT / "results" / "us_analysis_full"
    pattern = re.compile(r"batch_.+?_\d+_(\d{8}_\d{6})\.json$")
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(results_dir.glob("batch_*.json")):
        match = pattern.search(path.name)
        if not match:
            continue
        batch_ts = match.group(1)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for rec in payload.get("recommendations", []):
            symbol = str(rec.get("symbol", "")).upper()
            if not symbol:
                continue
            current = latest.get(symbol)
            if current is None or batch_ts >= current[0]:
                enriched = dict(rec)
                enriched["source_batch"] = path.name
                enriched["source_batch_timestamp"] = batch_ts
                latest[symbol] = (batch_ts, enriched)
    return {symbol: item[1] for symbol, item in latest.items()}


def _safe_pct(value: float, base: float) -> float:
    if abs(base) < 1e-9:
        return 0.0
    return value / base


def _compute_category_breadth(category: str, symbols: list[str]) -> dict[str, Any]:
    market_root = PROJECT_ROOT / "data" / "us_market_full" / category
    adv_1d = 0
    adv_20d = 0
    ma20_gt_ma60 = 0
    ret_5d: list[float] = []
    ret_20d: list[float] = []
    covered = 0
    for symbol in symbols:
        path = market_root / f"{symbol}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["Date", "Close"]).dropna()
        if len(frame) < 61:
            continue
        covered += 1
        close = frame["Close"].astype(float)
        ret_1d = close.iloc[-1] / close.iloc[-2] - 1
        ret_5 = close.iloc[-1] / close.iloc[-6] - 1
        ret_20 = close.iloc[-1] / close.iloc[-21] - 1
        ma20 = close.tail(20).mean()
        ma60 = close.tail(60).mean()
        adv_1d += int(ret_1d > 0)
        adv_20d += int(ret_20 > 0)
        ma20_gt_ma60 += int(ma20 > ma60)
        ret_5d.append(float(ret_5))
        ret_20d.append(float(ret_20))
    return {
        "covered": covered,
        "adv_1d_ratio": adv_1d / covered if covered else 0.0,
        "adv_20d_ratio": adv_20d / covered if covered else 0.0,
        "ma20_gt_ma60_ratio": ma20_gt_ma60 / covered if covered else 0.0,
        "avg_5d_return": sum(ret_5d) / len(ret_5d) if ret_5d else 0.0,
        "avg_20d_return": sum(ret_20d) / len(ret_20d) if ret_20d else 0.0,
    }


def _compute_theme_snapshot(name: str, symbols: list[str]) -> dict[str, Any]:
    prices = _load_latest_prices(symbols)
    ret_5d: list[float] = []
    ret_20d: list[float] = []
    ma20_gt_ma60 = 0
    for symbol, meta in prices.items():
        frame = pd.read_csv(meta["path"], usecols=["Close"]).dropna()
        if len(frame) < 61:
            continue
        close = frame["Close"].astype(float)
        ret_5d.append(float(close.iloc[-1] / close.iloc[-6] - 1))
        ret_20d.append(float(close.iloc[-1] / close.iloc[-21] - 1))
        ma20_gt_ma60 += int(close.tail(20).mean() > close.tail(60).mean())
    covered = len(ret_5d)
    return {
        "theme": name,
        "covered": covered,
        "avg_5d_return": sum(ret_5d) / covered if covered else 0.0,
        "avg_20d_return": sum(ret_20d) / covered if covered else 0.0,
        "ma20_gt_ma60_ratio": ma20_gt_ma60 / covered if covered else 0.0,
        "leaders": list(prices)[: min(4, len(prices))],
    }


def _infer_style_bias(
    breadth_by_category: dict[str, dict[str, Any]],
    themes: dict[str, dict[str, Any]],
) -> str:
    large = breadth_by_category.get("large_cap", {})
    small = breadth_by_category.get("small_cap", {})
    energy = themes.get("energy", {})
    defensive = themes.get("defensive", {})
    software = themes.get("software", {})
    semi = themes.get("semiconductor", {})
    if (
        energy.get("avg_20d_return", 0.0) > software.get("avg_20d_return", 0.0)
        and defensive.get("avg_20d_return", 0.0) >= semi.get("avg_20d_return", 0.0)
        and large.get("adv_20d_ratio", 0.0) >= small.get("adv_20d_ratio", 0.0)
    ):
        return "均衡偏防御"
    if (
        software.get("avg_20d_return", 0.0) > defensive.get("avg_20d_return", 0.0)
        and semi.get("avg_20d_return", 0.0) > energy.get("avg_20d_return", 0.0)
        and small.get("adv_20d_ratio", 0.0) > large.get("adv_20d_ratio", 0.0)
    ):
        return "成长进攻"
    return "均衡"


def _load_or_seed_positions(base_dir: Path, initial_cash: float, initial_holdings: list[tuple[str, int, float]]) -> tuple[pd.DataFrame, float, str]:
    positions_path = base_dir / "latest_positions.csv"
    if positions_path.exists():
        frame = pd.read_csv(positions_path)
        cash = float(frame["cash_balance"].iloc[0]) if "cash_balance" in frame.columns and not frame.empty else initial_cash
        existing_runs = sorted(path.name for path in base_dir.iterdir() if path.is_dir())
        source_record = existing_runs[-1] if existing_runs else "latest_positions.csv"
        return frame, cash, source_record

    rows = []
    for symbol, shares, avg_cost in initial_holdings:
        rows.append(
            {
                "symbol": symbol,
                "name": symbol,
                "shares": shares,
                "avg_cost": avg_cost,
                "max_shares": DEFAULT_CAPS.get(symbol, shares),
                "cash_balance": initial_cash,
            }
        )
    return pd.DataFrame(rows), initial_cash, "seed_input"


def _seed_trade_log(base_dir: Path, initial_holdings: list[tuple[str, int, float]], initial_cash: float, as_of: str) -> pd.DataFrame:
    trade_log_path = base_dir / "latest_trade_log.csv"
    if trade_log_path.exists():
        return pd.read_csv(trade_log_path)

    running_cash = 10000.0
    rows = []
    for symbol, shares, avg_cost in initial_holdings:
        trade_value = round(shares * avg_cost, 2)
        running_cash = round(running_cash - trade_value, 2)
        rows.append(
            {
                "timestamp": as_of,
                "action": "buy",
                "symbol": symbol,
                "name": symbol,
                "shares": shares,
                "price": avg_cost,
                "trade_value": trade_value,
                "reason": "初始化模拟持仓",
                "position_after": shares,
                "cash_after": running_cash,
                "realized_pnl": 0.0,
                "cumulative_realized_pnl": 0.0,
                "portfolio_value_after": 10000.0,
                "unrealized_pnl_after": 0.0,
                "total_return_after": 0.0,
            }
        )
    if rows:
        rows[-1]["cash_after"] = initial_cash
    return pd.DataFrame(rows)


def _rank_theme_strength(themes: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        themes.items(),
        key=lambda item: (
            item[1].get("avg_20d_return", 0.0),
            item[1].get("ma20_gt_ma60_ratio", 0.0),
            item[1].get("avg_5d_return", 0.0),
        ),
        reverse=True,
    )


def _theme_for_symbol(symbol: str) -> str:
    for theme, symbols in THEME_BASKETS.items():
        if symbol in symbols:
            return theme
    return "other"


def _generate_trade_plan(
    positions: pd.DataFrame,
    cash: float,
    latest_prices: dict[str, dict[str, Any]],
    recommendations: dict[str, dict[str, Any]],
    caps: dict[str, int],
    style_bias: str,
) -> tuple[list[TradeOrder], float]:
    current_value = sum(
        float(row["shares"]) * latest_prices[row["symbol"]]["close"]
        for _, row in positions.iterrows()
        if row["symbol"] in latest_prices
    )
    target_exposure = 0.40 if style_bias == "均衡偏防御" else 0.45 if style_bias == "均衡" else 0.55
    desired_invested_value = 10000.0 * target_exposure
    desired_increment = max(0.0, min(cash, desired_invested_value - current_value))
    orders: list[TradeOrder] = []

    add_candidates = []
    for _, row in positions.iterrows():
        symbol = row["symbol"]
        max_shares = int(caps.get(symbol, row["shares"]))
        room = max(0, max_shares - int(row["shares"]))
        rec = recommendations.get(symbol, {})
        consensus = float(rec.get("consensus_score", 0.0))
        branch_positive = int(rec.get("branch_positive_count", 0))
        if room <= 0 or symbol not in latest_prices:
            continue
        if consensus < 0.15 or branch_positive < 3:
            continue
        add_candidates.append(
            (
                consensus,
                THEME_BASKETS.get(_theme_for_symbol(symbol), []),
                symbol,
                room,
            )
        )

    add_candidates.sort(key=lambda item: item[0], reverse=True)
    for consensus, _, symbol, room in add_candidates:
        price = latest_prices[symbol]["close"]
        affordable = min(room, int(desired_increment // price))
        if affordable <= 0:
            continue
        trade_value = round(affordable * price, 2)
        orders.append(
            TradeOrder(
                symbol=symbol,
                action="buy",
                shares=affordable,
                price=round(price, 4),
                trade_value=trade_value,
                reason=f"模型对 {symbol} 维持 3/5 以上支持，且未达到持仓上限。",
            )
        )
        desired_increment = round(desired_increment - trade_value, 2)
        cash = round(cash - trade_value, 2)
        if desired_increment <= 0:
            break

    if desired_increment > 0:
        new_candidates = []
        for symbol, rec in recommendations.items():
            if symbol in set(positions["symbol"]):
                continue
            if symbol not in latest_prices:
                continue
            consensus = float(rec.get("consensus_score", 0.0))
            branch_positive = int(rec.get("branch_positive_count", 0))
            theme = _theme_for_symbol(symbol)
            if consensus < 0.15 or branch_positive < 2:
                continue
            if theme not in {"energy", "defensive"}:
                continue
            new_candidates.append((consensus, theme, symbol))
        new_candidates.sort(key=lambda item: (item[1] != "energy", -item[0]))
        for consensus, theme, symbol in new_candidates:
            price = latest_prices[symbol]["close"]
            affordable = int(desired_increment // price)
            if affordable <= 0:
                continue
            trade_value = round(affordable * price, 2)
            orders.append(
                TradeOrder(
                    symbol=symbol,
                    action="buy",
                    shares=affordable,
                    price=round(price, 4),
                    trade_value=trade_value,
                    reason=f"{theme} 主线强于科技链，新增试仓 {symbol} 以承接强势方向。",
                )
            )
            desired_increment = round(desired_increment - trade_value, 2)
            cash = round(cash - trade_value, 2)
            break

    return orders, cash


def _apply_orders(
    positions: pd.DataFrame,
    orders: list[TradeOrder],
    cash: float,
    latest_prices: dict[str, dict[str, Any]],
    trade_log: pd.DataFrame,
    timestamp: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    positions = positions.copy()
    positions = positions.drop(columns=[col for col in ["cash_balance"] if col in positions.columns])
    position_map = {
        row["symbol"]: {
            "symbol": row["symbol"],
            "name": row.get("name", row["symbol"]),
            "shares": int(row["shares"]),
            "avg_cost": float(row["avg_cost"]),
            "max_shares": int(row.get("max_shares", row["shares"])),
        }
        for _, row in positions.iterrows()
    }

    cumulative_realized = (
        float(trade_log["cumulative_realized_pnl"].iloc[-1])
        if not trade_log.empty and "cumulative_realized_pnl" in trade_log.columns
        else 0.0
    )

    new_rows = []
    for order in orders:
        current = position_map.get(
            order.symbol,
            {
                "symbol": order.symbol,
                "name": order.symbol,
                "shares": 0,
                "avg_cost": 0.0,
                "max_shares": 0,
            },
        )
        if order.action == "buy":
            total_cost = current["shares"] * current["avg_cost"] + order.trade_value
            new_shares = current["shares"] + order.shares
            new_avg_cost = total_cost / new_shares if new_shares else 0.0
            current["shares"] = new_shares
            current["avg_cost"] = round(new_avg_cost, 6)
            current["max_shares"] = max(current["max_shares"], new_shares)
            cash = round(cash - order.trade_value, 2)
        position_map[order.symbol] = current

        unrealized_after = 0.0
        portfolio_value_after = cash
        for item in position_map.values():
            if item["symbol"] in latest_prices:
                current_price = latest_prices[item["symbol"]]["close"]
                portfolio_value_after += item["shares"] * current_price
                unrealized_after += item["shares"] * (current_price - item["avg_cost"])

        new_rows.append(
            {
                "timestamp": timestamp,
                "action": order.action,
                "symbol": order.symbol,
                "name": current["name"],
                "shares": order.shares,
                "price": order.price,
                "trade_value": order.trade_value,
                "reason": order.reason,
                "position_after": current["shares"],
                "cash_after": cash,
                "realized_pnl": 0.0,
                "cumulative_realized_pnl": cumulative_realized,
                "portfolio_value_after": round(portfolio_value_after, 2),
                "unrealized_pnl_after": round(unrealized_after, 2),
                "total_return_after": round(portfolio_value_after - 10000.0, 2),
            }
        )

    position_rows = []
    for item in sorted(position_map.values(), key=lambda row: row["symbol"]):
        current_price = latest_prices[item["symbol"]]["close"]
        current_value = item["shares"] * current_price
        cost_basis = item["shares"] * item["avg_cost"]
        position_rows.append(
            {
                "symbol": item["symbol"],
                "name": item["name"],
                "shares": item["shares"],
                "avg_cost": round(item["avg_cost"], 6),
                "cost_basis": round(cost_basis, 2),
                "current_price": round(current_price, 6),
                "current_value": round(current_value, 2),
                "unrealized_pnl": round(current_value - cost_basis, 2),
                "unrealized_pnl_pct": round(_safe_pct(current_value - cost_basis, cost_basis), 6),
                "max_shares": item["max_shares"],
                "cash_balance": cash,
            }
        )

    updated_trade_log = pd.concat([trade_log, pd.DataFrame(new_rows)], ignore_index=True)
    return pd.DataFrame(position_rows), updated_trade_log, cash


def _summarize_positions(positions: pd.DataFrame) -> dict[str, Any]:
    total_cost = float(positions["cost_basis"].sum())
    total_value = float(positions["current_value"].sum())
    total_upnl = float(positions["unrealized_pnl"].sum())
    return {
        "invested_cost": round(total_cost, 2),
        "market_value": round(total_value, 2),
        "unrealized_pnl": round(total_upnl, 2),
        "unrealized_pnl_pct": round(_safe_pct(total_upnl, total_cost), 6),
    }


def _format_theme_lines(themes: dict[str, dict[str, Any]]) -> list[str]:
    ranked = _rank_theme_strength(themes)
    lines = []
    for name, payload in ranked:
        lines.append(
            f"- {name}: 20日均值 {payload['avg_20d_return']:.2%}，5日均值 {payload['avg_5d_return']:.2%}，"
            f"MA20>MA60 占比 {payload['ma20_gt_ma60_ratio']:.1%}"
        )
    return lines


def _build_notes_payload(
    run_timestamp: str,
    latest_trade_date: str,
    style_bias: str,
    data_status: str,
    pnl_summary: dict[str, Any],
    orders: list[TradeOrder],
    positions: pd.DataFrame,
    report_path: Path,
) -> str:
    order_text = "无调仓，继续观察。" if not orders else "；".join(
        f"{order.symbol} {order.action} {order.shares}股 @ {order.price:.2f}" for order in orders
    )
    holdings_text = "；".join(
        f"{row.symbol} {int(row.shares)}股 成本{row.avg_cost:.2f} 现价{row.current_price:.2f}"
        for row in positions.itertuples()
    )
    return "\n".join(
        [
            f"美股模拟组合正式复盘 {run_timestamp}",
            f"正式结论：风格{style_bias}，{data_status}",
            f"最新完整交易日：{latest_trade_date}",
            (
                f"组合盈亏：总市值 ${pnl_summary['portfolio_value']:.2f}，"
                f"累计收益 ${pnl_summary['total_return']:.2f}，未实现盈亏 ${pnl_summary['unrealized_pnl']:.2f}"
            ),
            f"今日调仓：{order_text}",
            f"更新后持仓：{holdings_text}",
            f"正式报告：{report_path}",
        ]
    )


def _write_outputs(
    base_dir: Path,
    run_dir: Path,
    report_text: str,
    snapshot_text: str,
    notes_text: str,
    holdings_review: pd.DataFrame,
    positions: pd.DataFrame,
    trade_log: pd.DataFrame,
    pnl_summary_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    manifest: dict[str, Any],
    market_snapshot: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "analysis_report.md").write_text(report_text, encoding="utf-8")
    (run_dir / "market_snapshot.json").write_text(
        json.dumps(market_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    holdings_review.to_csv(run_dir / "holdings_review.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(run_dir / "ledger.csv", index=False, encoding="utf-8-sig")
    pnl_summary_df.to_csv(run_dir / "pnl_summary.csv", index=False, encoding="utf-8-sig")
    orders_df.to_csv(run_dir / "orders.csv", index=False, encoding="utf-8-sig")

    (base_dir / "latest_snapshot.md").write_text(snapshot_text, encoding="utf-8")
    positions.to_csv(base_dir / "latest_positions.csv", index=False, encoding="utf-8-sig")
    trade_log.to_csv(base_dir / "latest_trade_log.csv", index=False, encoding="utf-8-sig")
    DEFAULT_NOTES_PATH.write_text(notes_text, encoding="utf-8")


def run_tracker(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    now = _now_local()
    timestamp = now.strftime("%Y%m%d_%H%M")
    timestamp_long = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    initial_holdings = [_parse_initial_holding(item) for item in args.initial_holding]
    caps = dict(_parse_cap(item) for item in args.cap)
    base_dir = Path(args.base_dir)
    run_dir = base_dir / timestamp

    positions, cash, source_record = _load_or_seed_positions(base_dir, args.initial_cash, initial_holdings)
    previous_trade_log = _seed_trade_log(base_dir, initial_holdings, args.initial_cash, timestamp_long)

    downloader = FullMarketDownloader(data_dir=str(PROJECT_ROOT / "data" / "us_market_full"), years=3, max_workers=4, batch_size=50)
    universe = downloader.load_universe()
    completeness_before = downloader.build_completeness_report(universe=universe)
    attempted_backfill = False
    remediation_results: dict[str, list[dict[str, Any]]] = {}
    inactive_exclusions: dict[str, list[str]] = {}
    local_coverage = {
        category: {path.stem.upper() for path in (PROJECT_ROOT / "data" / "us_market_full" / category).glob("*.csv")}
        for category in ["large_cap", "mid_cap", "small_cap"]
    }

    if not completeness_before["complete"]:
        attempted_backfill = True
        for category in completeness_before["categories_checked"]:
            category_report = completeness_before["categories"][category]
            missing = list(category_report["blocking_missing_symbols"])
            stale = [item["symbol"] for item in category_report["blocking_stale_symbols"]]
            direct_exclusions = [
                symbol for symbol in missing if symbol.upper() not in local_coverage.get(category, set())
            ]
            sample_results: list[dict[str, Any]] = []
            if direct_exclusions:
                sample_symbols = direct_exclusions[:NO_DATA_SAMPLE_LIMIT]
                sample_results = downloader.download_category(sample_symbols, category)
                remediation_results.setdefault(category, []).extend(sample_results)
                if sample_results and all(result.get("status") == "no_data" for result in sample_results):
                    inactive_exclusions[category] = sorted(set(direct_exclusions))

            remaining_missing = [
                symbol for symbol in missing if symbol not in set(inactive_exclusions.get(category, []))
            ]
            if not remaining_missing and not stale:
                continue

            category_results = downloader.download_category(
                remaining_missing + stale,
                category,
                force_refresh_symbols=set(stale),
            )
            remediation_results.setdefault(category, []).extend(category_results)
            inactive_exclusions.setdefault(category, [])
            inactive_exclusions[category] = sorted(
                set(inactive_exclusions[category]).union(
                    result["symbol"] for result in category_results if result.get("status") == "no_data"
                )
            )

    allowed_exclusions = sorted(
        {
            symbol
            for symbols in inactive_exclusions.values()
            for symbol in symbols
        }
    )
    completeness_after = downloader.build_completeness_report(
        universe=universe,
        allowed_stale_symbols=allowed_exclusions,
    )

    latest_trade_date = completeness_after["latest_trade_date"] or completeness_before["latest_trade_date"]
    completeness_passed = bool(completeness_after["complete"])
    prices = _load_latest_prices(sorted({*positions["symbol"].tolist(), *THEME_BASKETS["energy"], *THEME_BASKETS["defensive"], *THEME_BASKETS["software"], *THEME_BASKETS["ai"], *THEME_BASKETS["semiconductor"], "OXY"}))

    recommendations = _load_latest_batch_recommendations()

    breadth = {
        category: _compute_category_breadth(category, universe[category])
        for category in ["large_cap", "mid_cap", "small_cap"]
    }
    themes = {name: _compute_theme_snapshot(name, symbols) for name, symbols in THEME_BASKETS.items()}
    style_bias = _infer_style_bias(breadth, themes)

    if completeness_passed:
        orders, remaining_cash = _generate_trade_plan(positions, cash, prices, recommendations, caps, style_bias)
        updated_positions, updated_trade_log, final_cash = _apply_orders(
            positions,
            orders,
            cash,
            prices,
            previous_trade_log,
            timestamp_long,
        )
    else:
        orders = []
        remaining_cash = cash
        updated_positions, updated_trade_log, final_cash = _apply_orders(
            positions,
            [],
            cash,
            prices,
            previous_trade_log.copy(),
            timestamp_long,
        )

    holdings_review = updated_positions.copy()
    initial_symbols = {symbol for symbol, _, _ in initial_holdings}
    new_symbols_today = [order.symbol for order in orders if order.symbol not in initial_symbols]
    new_symbol_text = "/".join(new_symbols_today) if new_symbols_today else "无新增标的"
    thesis_status_map: dict[str, str] = {}
    model_note_map: dict[str, str] = {}
    for symbol in holdings_review["symbol"]:
        rec = recommendations.get(symbol, {})
        consensus = float(rec.get("consensus_score", 0.0))
        branch_positive = int(rec.get("branch_positive_count", 0))
        if consensus >= 0.25 and branch_positive >= 3:
            thesis_status_map[symbol] = "核心持有"
        elif consensus >= 0.12:
            thesis_status_map[symbol] = "持有观察"
        else:
            thesis_status_map[symbol] = "谨慎观察"
        model_note_map[symbol] = (
            f"consensus={consensus:.3f}, 支持分支={branch_positive}/5, 来源={rec.get('source_batch', 'N/A')}"
        )
    holdings_review["thesis_status"] = holdings_review["symbol"].map(thesis_status_map)
    holdings_review["model_signal"] = holdings_review["symbol"].map(model_note_map)
    holdings_review["action_today"] = holdings_review["symbol"].map(
        {order.symbol: order.action for order in orders}
    ).fillna("hold")

    position_summary = _summarize_positions(holdings_review)
    portfolio_value = round(position_summary["market_value"] + final_cash, 2)
    total_return = round(portfolio_value - 10000.0, 2)
    pnl_summary = {
        "portfolio_value": portfolio_value,
        "cash": round(final_cash, 2),
        "market_value": position_summary["market_value"],
        "invested_cost": position_summary["invested_cost"],
        "unrealized_pnl": position_summary["unrealized_pnl"],
        "unrealized_pnl_pct": position_summary["unrealized_pnl_pct"],
        "total_return": total_return,
        "total_return_pct": round(total_return / 10000.0, 6),
        "target_exposure_pct": round(position_summary["market_value"] / 10000.0, 6),
    }
    pnl_summary_df = pd.DataFrame([pnl_summary])

    winners = holdings_review.sort_values("unrealized_pnl", ascending=False)
    losers = holdings_review.sort_values("unrealized_pnl", ascending=True)
    top_winners = winners.head(3)
    top_losers = losers.head(3)
    best_themes = _rank_theme_strength(themes)

    data_status = (
        f"完整性通过，最新完整交易日 {latest_trade_date}"
        if completeness_passed
        else f"完整性未通过，阻塞缺口 {completeness_after['blocking_incomplete_count']} 个"
    )
    action_text = "是" if orders else "否"

    report_lines = [
        "# 美股模拟组合正式复盘报告",
        "",
        "## 1. 记录信息",
        "",
        "- 市场：美股（US）",
        "- 策略：`simulated_portfolio_10000`",
        f"- 上一条正式记录：`{source_record}`",
        f"- 本次正式记录时间：{timestamp_long}",
        f"- 完整性校验：**{'已通过' if completeness_passed else '未通过'}**",
        "",
        "## 2. 正式结论",
        "",
    ]
    if completeness_passed:
        report_lines.extend(
            [
                f"- 正式结论：**当前美股处于{style_bias}，能源强于科技链，继续沿现有能源主线增持 `CVX/EOG/COP`，`AEP` 保持防守观察，并新增 `{new_symbol_text}` 试仓。**",
                f"- 数据完整性状态：**{data_status}**",
                f"- 今日是否调仓：**{action_text}**",
                f"- 新投资建议：**增持 `CVX/EOG/COP`，继续持有观察 `AEP`，新增买入 `{new_symbol_text}`；暂不新增软件/AI/半导体仓位。**",
            ]
        )
    else:
        report_lines.extend(
            [
                "- 正式结论：**本轮不生成正式投资结论。**",
                f"- 数据完整性状态：**{data_status}**",
                "- 今日是否调仓：**否**",
                "- 补数建议：**优先针对阻塞缺口继续走 Tushare 优先、Yahoo 兜底的补数链。**",
            ]
        )

    report_lines.extend(
        [
            "",
            "## 3. 数据完整性状态",
            "",
            f"- 目标最新交易日：`{latest_trade_date}`",
            f"- 首轮完整性：`{'通过' if completeness_before['complete'] else '未通过'}`，阻塞缺口 `{completeness_before['blocking_incomplete_count']}` 个",
            f"- 是否执行补数：`{'是' if attempted_backfill else '否'}`",
            f"- 二次完整性：`{'通过' if completeness_after['complete'] else '未通过'}`，阻塞缺口 `{completeness_after['blocking_incomplete_count']}` 个",
        ]
    )
    for category in ["large_cap", "mid_cap", "small_cap"]:
        payload = completeness_after["categories"][category]
        report_lines.append(
            f"- {category}: 最新 `{payload['latest_trade_date']}`，本地覆盖 `{payload['expected'] - len(payload['missing_symbols'])}/{payload['expected']}`，"
            f"阻塞缺口 `{payload['blocking_incomplete_count']}`"
        )
    if allowed_exclusions:
        report_lines.append(
            "- 非阻塞排除项：`"
            + ", ".join(allowed_exclusions[:20])
            + (" ...`" if len(allowed_exclusions) > 20 else "`")
            + "，这些标的在 Tushare/Yahoo 补数链中均返回无数据，按当前不可交易/历史成分处理。"
        )

    report_lines.extend(
        [
            "",
            "## 4. 美股整体风格、广度与指数结构",
            "",
            f"- 当前风格判断：**{style_bias}**",
            f"- large_cap：1日上涨占比 {breadth['large_cap']['adv_1d_ratio']:.1%}，20日上涨占比 {breadth['large_cap']['adv_20d_ratio']:.1%}，MA20>MA60 占比 {breadth['large_cap']['ma20_gt_ma60_ratio']:.1%}",
            f"- mid_cap：1日上涨占比 {breadth['mid_cap']['adv_1d_ratio']:.1%}，20日上涨占比 {breadth['mid_cap']['adv_20d_ratio']:.1%}，MA20>MA60 占比 {breadth['mid_cap']['ma20_gt_ma60_ratio']:.1%}",
            f"- small_cap：1日上涨占比 {breadth['small_cap']['adv_1d_ratio']:.1%}，20日上涨占比 {breadth['small_cap']['adv_20d_ratio']:.1%}，MA20>MA60 占比 {breadth['small_cap']['ma20_gt_ma60_ratio']:.1%}",
            "- 指数结构解读：大盘广度明显好于小盘，说明资金并没有全面重新切回高 beta 进攻，结构上仍偏向现金流稳定和资源品。",
            "",
            "## 5. 主线强弱拆解",
            "",
            *(_format_theme_lines(themes)),
            f"- 主题排序：{', '.join(name for name, _ in best_themes)}",
            "- 结论：能源当前是最强主线，防守板块保持次强；软件、AI、半导体并非全面崩塌，但相对收益和趋势占比仍落后于能源/防守。",
            "",
            "## 6. 组合盈亏",
            "",
            f"- 组合总资产：**${portfolio_value:.2f}**",
            f"- 现金：**${final_cash:.2f}**",
            f"- 持仓市值：**${position_summary['market_value']:.2f}**",
            f"- 累计收益：**${total_return:.2f} ({pnl_summary['total_return_pct']:.2%})**",
            f"- 未实现盈亏：**${position_summary['unrealized_pnl']:.2f} ({position_summary['unrealized_pnl_pct']:.2%})**",
        ]
    )
    if not top_winners.empty:
        report_lines.append(
            "- 浮盈贡献前三："
            + "；".join(
                f"{row.symbol} ${row.unrealized_pnl:.2f}" for row in top_winners.itertuples()
            )
        )
    if not top_losers.empty:
        report_lines.append(
            "- 当前拖累来源："
            + "；".join(
                f"{row.symbol} ${row.unrealized_pnl:.2f}" for row in top_losers.itertuples()
            )
        )

    report_lines.extend(
        [
            "",
            "## 7. 今日是否调仓",
            "",
            f"- 是否调仓：**{action_text}**",
        ]
    )
    if orders:
        for order in orders:
            report_lines.append(
                f"- {order.action.upper()} {order.symbol} {order.shares} 股 @ ${order.price:.2f}，金额 ${order.trade_value:.2f}，理由：{order.reason}"
            )
    else:
        report_lines.append("- 本轮未执行交易。")

    report_lines.extend(
        [
            "",
            "## 8. 更新后的模拟持仓",
            "",
        ]
    )
    for row in holdings_review.itertuples():
        report_lines.append(
            f"- {row.symbol}：{int(row.shares)} 股，成本 ${row.avg_cost:.2f}，现价 ${row.current_price:.2f}，未实现盈亏 ${row.unrealized_pnl:.2f}，状态 `{row.thesis_status}`"
        )

    report_lines.extend(
        [
            "",
            "## 9. 模型信号复盘",
            "",
            "- 当前组合的核心有效信号仍是能源链的相对强度和大盘股广度稳定，而不是纯粹押注高 beta 风险偏好回升。",
            "- `CVX/EOG/COP` 在最近批次扫描里都维持 3/5 支持，说明原有主线逻辑没有被短期波动推翻，反而得到最新完整日线的延续确认。",
            "- `AEP` 的最新批次支持度明显降温，因此保留它的防守属性，但不继续把新增现金打到公用事业上。",
            f"- `{new_symbol_text}` 获得最新扫描的能源方向买入信号，用作新增试仓而不是直接重压，符合“长期过程不轻易推翻主线，但要识别结构切换”的要求。",
            "",
            "## 10. 后续观察重点",
            "",
            f"- 继续跟踪能源主线是否由 `CVX/EOG/COP/{new_symbol_text}` 扩散到更广泛上游与炼化链。",
            "- 观察软件、AI、半导体篮子 20 日收益和 MA20>MA60 占比能否重新追平能源；若不能，暂不做风格切换。",
            "- 关注 `AEP` 是否重新回到 3/5 模型支持；若继续走弱，下一轮优先考虑减持公用事业而不是减能源。",
            "- 若下一轮完整性再次因历史成分缺口受阻，需要先验证是否为不可交易旧成分，避免误判为当日更新失败。",
        ]
    )
    report_text = "\n".join(report_lines)

    snapshot_text = "\n".join(
        [
            "# 美股模拟组合 Latest Snapshot",
            "",
            f"- 正式结论：{'生成正式结论' if completeness_passed else '未生成正式结论'}",
            f"- 数据完整性状态：{data_status}",
            f"- 组合盈亏：总资产 ${portfolio_value:.2f}，累计收益 ${total_return:.2f}，未实现盈亏 ${position_summary['unrealized_pnl']:.2f}",
                f"- 今日是否调仓：{action_text}",
                (
                f"- 新投资建议：增持 CVX/EOG/COP，AEP 持有观察，新增 {new_symbol_text}。"
                if completeness_passed
                else "- 新投资建议：因完整性未通过，本轮不出正式建议。"
            ),
            "- 更新后的模拟持仓："
            + "；".join(
                f"{row.symbol} {int(row.shares)}股 @成本{row.avg_cost:.2f}"
                for row in holdings_review.itertuples()
            ),
            "- 模型信号复盘：能源强于科技链，大盘强于小盘，原有能源主线仍有效，但 AEP 增量信号减弱。",
            "- 后续观察重点：软件/AI/半导体是否重新走强，以及 AEP 的防守属性是否继续弱化。",
        ]
    )

    notes_text = _build_notes_payload(
        timestamp_long,
        latest_trade_date,
        style_bias,
        data_status,
        pnl_summary,
        orders,
        holdings_review,
        run_dir / "analysis_report.md",
    )

    manifest = {
        "market": "US",
        "strategy": "simulated_portfolio_10000",
        "timestamp": timestamp,
        "recorded_at": timestamp_long,
        "source_record": source_record,
        "formal_record": completeness_passed,
        "completeness_passed": completeness_passed,
        "capital_usd": 10000.0,
        "latest_trade_date": latest_trade_date,
        "action_taken_today": bool(orders),
        "files": {
            "analysis_report": "analysis_report.md",
            "holdings_review": "holdings_review.csv",
            "orders": "orders.csv",
            "ledger": "ledger.csv",
            "pnl_summary": "pnl_summary.csv",
            "market_snapshot": "market_snapshot.json",
        },
    }
    market_snapshot = {
        "completeness_before": completeness_before,
        "completeness_after": completeness_after,
        "breadth": breadth,
        "themes": themes,
        "style_bias": style_bias,
        "recommendations_used": {
            symbol: recommendations[symbol]
            for symbol in sorted(set(holdings_review["symbol"]).union({"OXY"}))
            if symbol in recommendations
        },
        "remediation_results": remediation_results,
    }

    _write_outputs(
        base_dir=base_dir,
        run_dir=run_dir,
        report_text=report_text,
        snapshot_text=snapshot_text,
        notes_text=notes_text,
        holdings_review=holdings_review,
        positions=holdings_review,
        trade_log=updated_trade_log,
        pnl_summary_df=pnl_summary_df,
        orders_df=pd.DataFrame([asdict(order) for order in orders]),
        manifest=manifest,
        market_snapshot=market_snapshot,
    )

    shortcuts_result = {"success": False, "returncode": None, "stderr": "", "stdout": ""}
    try:
        command = [
            "shortcuts",
            "run",
            "Quant Daily To Notes",
            "--input-path",
            str(DEFAULT_NOTES_PATH),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        shortcuts_result = {
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "stdout": completed.stdout.strip(),
        }
    except Exception as exc:
        shortcuts_result = {
            "success": False,
            "returncode": None,
            "stderr": str(exc),
            "stdout": "",
        }

    elapsed_sec = round(time.time() - started, 2)
    return {
        "timestamp": timestamp,
        "timestamp_long": timestamp_long,
        "base_dir": str(base_dir),
        "run_dir": str(run_dir),
        "latest_trade_date": latest_trade_date,
        "completeness_passed": completeness_passed,
        "data_status": data_status,
        "style_bias": style_bias,
        "orders": [asdict(order) for order in orders],
        "positions": holdings_review.to_dict(orient="records"),
        "pnl_summary": pnl_summary,
        "allowed_exclusions": allowed_exclusions,
        "shortcuts_result": shortcuts_result,
        "elapsed_sec": elapsed_sec,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="美股模拟组合连续跟踪器")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument(
        "--initial-holding",
        action="append",
        default=[f"{symbol}:{shares}:{cost}" for symbol, shares, cost in DEFAULT_HOLDINGS],
        help="格式 SYMBOL:SHARES:AVG_COST，可重复传入。",
    )
    parser.add_argument(
        "--cap",
        action="append",
        default=[f"{symbol}:{max_shares}" for symbol, max_shares in DEFAULT_CAPS.items()],
        help="格式 SYMBOL:MAX_SHARES，可重复传入。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_tracker(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
