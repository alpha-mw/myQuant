"""
A股激进科技制造策略正式复盘跟踪器。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_investor import QuantInvestor
from quant_investor.market.analyze import load_cn_stock_names
from quant_investor.market.download_cn import CNFullMarketDownloader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_NOTES_PATH = DEFAULT_BASE_DIR / "latest_notes_payload.md"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_SHORTCUT_NAME = "Quant Daily To Notes"
QUOTE_TIMEOUT = 20
DEFAULT_REVIEW_AGENT_MODEL = "gpt-4.1-mini"
DEFAULT_REVIEW_MASTER_MODEL = "gpt-4.1-mini"
DEFAULT_REVIEW_AGENT_TIMEOUT = 20.0
DEFAULT_REVIEW_MASTER_TIMEOUT = 45.0
DEFAULT_REVIEW_TOTAL_TIMEOUT = 180.0
INDEX_QUOTES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
    "sz399905": "中证500",
    "sz399852": "中证1000",
    "sh000688": "科创50",
    "sz399673": "创业板50",
}
THEME_BASKETS = {
    "先进材料": ["688519.SH", "688295.SH"],
    "AI存储": ["688525.SH"],
    "电子制造": ["002384.SZ", "002008.SZ"],
    "电力设备": ["601179.SH", "600487.SH"],
    "光通信": ["601869.SH"],
}


@dataclass
class ProposedOrder:
    symbol: str
    action: str
    shares: int
    price: float
    trade_value: float
    realized_pnl: float
    reason: str


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_pct(value: float, base: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return value / base


def _map_symbol_to_quote_code(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if text.startswith(("SH", "SZ")) and "." not in text:
        return text.lower()
    code, market = text.split(".")
    prefix = "sh" if market == "SH" else "sz"
    return f"{prefix}{code}"


def _decode_quote_payload(content: bytes) -> str:
    for encoding in ("gbk", "gb18030", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _parse_quote_payload(line: str) -> dict[str, Any] | None:
    text = str(line).strip()
    if not text or "=" not in text or "~" not in text:
        return None

    prefix, payload = text.split("=", 1)
    quote_code = prefix.replace("v_", "").strip()
    payload = payload.strip().strip(";").strip('"')
    parts = payload.split("~")
    if len(parts) < 6:
        return None

    current = _safe_float(parts[3])
    prev_close = _safe_float(parts[4])
    change = _safe_float(parts[31], current - prev_close) if len(parts) > 31 else current - prev_close
    change_pct = (
        _safe_float(parts[32], _safe_pct(change, prev_close) * 100.0)
        if len(parts) > 32
        else _safe_pct(change, prev_close) * 100.0
    )
    return {
        "quote_code": quote_code,
        "name": parts[1].strip() or quote_code,
        "current": current,
        "prev_close": prev_close,
        "open": _safe_float(parts[5]),
        "high": _safe_float(parts[33]) if len(parts) > 33 else 0.0,
        "low": _safe_float(parts[34]) if len(parts) > 34 else 0.0,
        "time": parts[30].strip() if len(parts) > 30 else "",
        "change": change,
        "change_pct": change_pct,
    }


def _fetch_tencent_quotes(quote_codes: list[str]) -> dict[str, dict[str, Any]]:
    if not quote_codes:
        return {}

    chunks = [quote_codes[idx : idx + 60] for idx in range(0, len(quote_codes), 60)]
    result: dict[str, dict[str, Any]] = {}
    session = requests.Session()

    for chunk in chunks:
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        response = session.get(url, timeout=QUOTE_TIMEOUT)
        response.raise_for_status()
        payload = _decode_quote_payload(response.content)
        for raw_line in payload.split(";"):
            parsed = _parse_quote_payload(raw_line)
            if parsed is None:
                continue
            result[parsed["quote_code"]] = parsed
    return result


def _price_series(frame: pd.DataFrame) -> pd.Series:
    return frame["close"].astype(float)


def _load_history_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    return frame


def _metric_return(close: pd.Series, periods: int) -> float:
    if close.empty:
        return 0.0
    if len(close) <= periods:
        base = float(close.iloc[0])
    else:
        base = float(close.iloc[-(periods + 1)])
    current = float(close.iloc[-1])
    return _safe_pct(current - base, base)


def _derive_stage_levels(frame: pd.DataFrame, current_price: float) -> tuple[float, float]:
    if frame.empty or current_price <= 0:
        return round(current_price * 1.08, 2), round(current_price * 0.94, 2)

    recent = frame.tail(60).copy()
    high = recent["high"].astype(float) if "high" in recent.columns else recent["close"].astype(float)
    low = recent["low"].astype(float) if "low" in recent.columns else recent["close"].astype(float)
    close = recent["close"].astype(float)
    prev_close = close.shift(1).fillna(close)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.tail(14).mean()) if len(true_range) >= 2 else current_price * 0.02
    atr = max(atr, current_price * 0.005)

    ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
    low20 = float(low.tail(20).min()) if len(low) >= 5 else current_price * 0.96
    high20 = float(high.tail(20).max()) if len(high) >= 5 else current_price * 1.06

    support = min(current_price, max(low20, ma20 - 0.75 * atr))
    resistance = max(high20, current_price + 1.5 * atr)

    stop_price = max(current_price * 0.75, min(support * 0.99, current_price * 0.985))
    target_price = max(current_price * 1.05, resistance)
    return round(target_price, 2), round(stop_price, 2)


def _score_full_market_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    scored = metrics.copy()
    rank_weights = {
        "ret1": 0.08,
        "ret5": 0.14,
        "ret20": 0.24,
        "ret60": 0.22,
        "close_vs_ma20": 0.12,
        "ma20_vs_ma60": 0.10,
        "ma60_vs_ma120": 0.06,
        "dd20": 0.04,
    }
    for column in rank_weights:
        scored[f"{column}_pct"] = scored[column].rank(method="average", pct=True)

    scored["score_full_market"] = 0.0
    for column, weight in rank_weights.items():
        scored["score_full_market"] += scored[f"{column}_pct"] * weight

    scored["score_full_market"] = scored["score_full_market"].round(6)
    scored = scored.sort_values(
        by=["score_full_market", "ret20", "ret60", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    scored["rank_full_market"] = range(1, len(scored) + 1)
    return scored


def _compute_category_breadth(
    category: str,
    symbols: list[str],
    data_root: Path,
    latest_trade_date: str,
    completeness_report: dict[str, Any],
) -> dict[str, Any]:
    covered = 0
    adv_1d = 0
    adv_20d = 0
    ma20_gt_ma60 = 0
    ret_1d_values: list[float] = []
    ret_20d_values: list[float] = []
    ret_60d_values: list[float] = []

    for symbol in symbols:
        path = data_root / category / f"{symbol}.csv"
        if not path.exists():
            continue
        frame = _load_history_frame(path)
        if frame.empty:
            continue
        latest_local_date = str(frame["trade_date"].iloc[-1]).replace("-", "")
        if latest_local_date != latest_trade_date:
            continue

        close = _price_series(frame).dropna().astype(float)
        if len(close) < 2:
            continue

        ret1 = _metric_return(close, 1)
        ret20 = _metric_return(close, 20)
        ret60 = _metric_return(close, 60)
        ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
        ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())

        covered += 1
        adv_1d += int(ret1 > 0)
        adv_20d += int(ret20 > 0)
        ma20_gt_ma60 += int(ma20 > ma60)
        ret_1d_values.append(ret1)
        ret_20d_values.append(ret20)
        ret_60d_values.append(ret60)

    payload = completeness_report["categories"][category]
    return {
        "ret1_positive_ratio": adv_1d / covered if covered else 0.0,
        "ret20_positive_ratio": adv_20d / covered if covered else 0.0,
        "ma20_gt_ma60_ratio": ma20_gt_ma60 / covered if covered else 0.0,
        "avg_ret1": sum(ret_1d_values) / len(ret_1d_values) if ret_1d_values else 0.0,
        "avg_ret20": sum(ret_20d_values) / len(ret_20d_values) if ret_20d_values else 0.0,
        "avg_ret60": sum(ret_60d_values) / len(ret_60d_values) if ret_60d_values else 0.0,
        "latest_count": covered,
        "expected": int(payload.get("expected", len(symbols))),
        "suspended_stale_count": len(payload.get("suspended_stale_symbols", [])),
    }


def _compute_full_market_metrics(
    components: dict[str, Any],
    data_root: Path,
    latest_trade_date: str,
) -> pd.DataFrame:
    stock_names = load_cn_stock_names()
    rows: list[dict[str, Any]] = []
    for category in ("hs300", "zz500", "zz1000"):
        for symbol in components.get(category, []):
            path = data_root / category / f"{symbol}.csv"
            if not path.exists():
                continue
            frame = _load_history_frame(path)
            if frame.empty or "trade_date" not in frame.columns or "close" not in frame.columns:
                continue

            latest_local_date = str(frame["trade_date"].iloc[-1]).replace("-", "")
            if latest_local_date != latest_trade_date:
                continue

            close = _price_series(frame).dropna().astype(float)
            if len(close) < 20:
                continue

            ma20 = float(close.tail(20).mean())
            ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())
            ma120 = float(close.tail(120).mean()) if len(close) >= 120 else ma60
            latest_close = float(close.iloc[-1])
            target_price, stop_price = _derive_stage_levels(frame, latest_close)

            rows.append(
                {
                    "symbol": symbol,
                    "name": stock_names.get(symbol, symbol),
                    "category": category,
                    "ret1": _metric_return(close, 1),
                    "ret5": _metric_return(close, 5),
                    "ret20": _metric_return(close, 20),
                    "ret60": _metric_return(close, 60),
                    "close_vs_ma20": _safe_pct(latest_close - ma20, ma20),
                    "ma20_vs_ma60": _safe_pct(ma20 - ma60, ma60),
                    "ma60_vs_ma120": _safe_pct(ma60 - ma120, ma120),
                    "dd20": _safe_pct(latest_close - float(close.tail(20).max()), float(close.tail(20).max())),
                    "latest_close": latest_close,
                    "stage_target_price": target_price,
                    "stage_stop_price": stop_price,
                }
            )

    metrics = pd.DataFrame(rows)
    return _score_full_market_metrics(metrics)


def _summarize_theme_strength(review: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theme, symbols in THEME_BASKETS.items():
        subset = review[review["symbol"].isin(symbols)].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "theme": theme,
                "symbols": subset["symbol"].tolist(),
                "avg_today_change_pct": float(subset["today_change_pct"].mean()),
                "avg_score": float(subset["score_full_market"].mean()),
                "avg_rank": float(subset["rank_full_market"].mean()),
                "avg_ret20": float(subset["ret20"].mean()),
            }
        )
    rows.sort(
        key=lambda item: (
            item["avg_score"],
            item["avg_today_change_pct"],
            item["avg_ret20"],
            -item["avg_rank"],
            item["theme"],
        ),
        reverse=True,
    )
    return rows


def _market_style_conclusion(indices: dict[str, dict[str, Any]], breadth: dict[str, dict[str, Any]]) -> str:
    hs300 = breadth.get("hs300", {})
    zz500 = breadth.get("zz500", {})
    zz1000 = breadth.get("zz1000", {})
    kc50 = indices.get("sh000688", {})
    hs300_idx = indices.get("sh000300", {})
    cyb = indices.get("sz399006", {})

    if (
        kc50.get("change_pct", 0.0) > hs300_idx.get("change_pct", 0.0)
        and zz1000.get("ret20_positive_ratio", 0.0) >= hs300.get("ret20_positive_ratio", 0.0)
    ):
        return "成长修复偏强，科技高弹性重新获得承接。"
    if (
        hs300.get("ret20_positive_ratio", 0.0) >= zz1000.get("ret20_positive_ratio", 0.0)
        and hs300_idx.get("change_pct", 0.0) >= cyb.get("change_pct", 0.0)
    ):
        return "风格仍偏大盘稳健，成长方向只是局部修复。"
    if kc50.get("change_pct", 0.0) > 0 and cyb.get("change_pct", 0.0) < 0:
        return "市场处于结构性修复，科创强于泛成长，资金更偏向有景气验证的硬科技。"
    return "市场仍是结构性分化，未回到全面进攻。"


def _tech_mainline_conclusion(theme_strength: list[dict[str, Any]]) -> tuple[str, str]:
    if not theme_strength:
        return "暂无足够主题样本。", "暂无足够主题样本。"

    strongest = theme_strength[0]
    weakest = theme_strength[-1]
    strong_text = (
        f"{strongest['theme']} 最强，平均盘中涨跌 {strongest['avg_today_change_pct']:+.2f}% ，"
        f"平均全市场评分 {strongest['avg_score']:.3f}。"
    )
    weak_text = (
        f"{weakest['theme']} 最弱，平均盘中涨跌 {weakest['avg_today_change_pct']:+.2f}% ，"
        f"平均全市场评分 {weakest['avg_score']:.3f}。"
    )
    return strong_text, weak_text


def _position_role(row: pd.Series) -> str:
    rank = int(row["rank_full_market"])
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])
    if rank <= 30 and price >= stop_price:
        return "核心持有"
    if rank <= 120 and price >= stop_price:
        return "稳定核心"
    if rank <= 260 and price >= stop_price * 0.995:
        return "观察持有"
    return "降级观察"


def _position_action(row: pd.Series) -> str:
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])
    score = float(row["score_full_market"])
    if price < stop_price and score < 0.72:
        return "减仓待确认"
    if price < stop_price:
        return "继续观察"
    return "继续持有"


def _position_reason(row: pd.Series) -> str:
    rank = int(row["rank_full_market"])
    score = float(row["score_full_market"])
    today = float(row["today_change_pct"])
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])

    if price < stop_price and score < 0.72:
        return (
            f"盘中 {today:+.2f}% 且仍低于阶段止损位 {stop_price:.2f}，"
            f"完整日线强度只在全市场第 {rank} 位，需把减仓判断留在下一次确认。"
        )
    if rank <= 20:
        return (
            f"完整日线仍处第一梯队，全市场排名第 {rank} 位，"
            f"盘中 {today:+.2f}% 说明主线承接仍在。"
        )
    if price < stop_price:
        return (
            f"盘中 {today:+.2f}% ，尚未重新站回阶段止损位 {stop_price:.2f}，"
            f"但全市场评分 {score:.3f} 仍未完全失真，先观察修复延续性。"
        )
    return (
        f"完整日线评分 {score:.3f}，全市场排名第 {rank} 位，"
        f"盘中 {today:+.2f}% ，继续按主线内部分化而非失效处理。"
    )


def _build_rebalance_plan(
    review: pd.DataFrame,
    review_recommendations: dict[str, dict[str, Any]],
) -> list[ProposedOrder]:
    weak = review[
        (review["current_price"] < review["stage_stop_price"])
        & (review["score_full_market"] < 0.72)
        & (review["today_change_pct"] <= 0.5)
    ].sort_values(["market_weight", "score_full_market"], ascending=[False, True])

    if weak.empty:
        return []

    orders: list[ProposedOrder] = []
    for row in weak.head(1).itertuples():
        review_payload = review_recommendations.get(str(row.symbol).upper(), {})
        review_action = str(review_payload.get("action", "hold")).strip().lower()
        if review_action != "sell":
            continue
        lot_size = 100
        shares = int(int(row.shares_before) * 0.2 // lot_size) * lot_size
        if shares <= 0:
            continue
        trade_value = round(shares * float(row.current_price), 2)
        realized = round(shares * (float(row.current_price) - float(row.buy_price)), 2)
        rationale = str(review_payload.get("rationale", "")).strip()
        orders.append(
            ProposedOrder(
                symbol=row.symbol,
                action="sell",
                shares=shares,
                price=round(float(row.current_price), 2),
                trade_value=trade_value,
                realized_pnl=realized,
                reason=(
                    f"{row.symbol} 仍低于阶段止损位 {float(row.stage_stop_price):.2f}，"
                    "且完整日线已明显落后于组合主线，"
                    + (f"Review Layer 亦给出 sell，理由：{rationale}" if rationale else "Review Layer 亦给出 sell。")
                ),
            )
        )
    return orders


def _apply_orders(
    source_ledger: pd.DataFrame,
    orders: list[ProposedOrder],
    cash_before: float,
    quote_prices: dict[str, float],
) -> tuple[pd.DataFrame, float, float]:
    ledger = source_ledger.copy()
    ledger = ledger.drop(columns=[column for column in ["market_weight"] if column in ledger.columns])
    ledger["shares"] = ledger["shares"].astype(int)
    cash_after = round(cash_before, 2)
    realized_total = 0.0
    order_map = {order.symbol: order for order in orders}

    updated_rows: list[dict[str, Any]] = []
    for row in ledger.itertuples():
        shares = int(row.shares)
        avg_cost = float(row.avg_cost)
        order = order_map.get(row.symbol)
        if order and order.action == "sell":
            shares = max(0, shares - order.shares)
            cash_after = round(cash_after + order.trade_value, 2)
            realized_total += order.realized_pnl

        price = float(quote_prices.get(row.symbol, getattr(row, "current_price", 0.0)))
        cost_basis = round(shares * avg_cost, 2)
        current_value = round(shares * price, 2)
        unrealized = round(current_value - cost_basis, 2)
        updated_rows.append(
            {
                "symbol": row.symbol,
                "name": row.name,
                "shares": shares,
                "avg_cost": round(avg_cost, 6),
                "cost_basis": cost_basis,
                "current_price": round(price, 2),
                "current_value": current_value,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": round(_safe_pct(unrealized, cost_basis), 6),
            }
        )

    updated = pd.DataFrame(updated_rows)
    invested = float(updated["current_value"].sum()) if not updated.empty else 0.0
    updated["market_weight"] = (
        updated["current_value"] / invested if invested > 0 else 0.0
    ).round(6)
    return updated, cash_after, round(realized_total, 2)


def _load_previous_record(
    base_dir: Path,
    source_record: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    run_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise RuntimeError("策略目录下不存在上一条正式记录，无法做连续复盘。")

    if source_record:
        latest_dir = base_dir / source_record
        if not latest_dir.exists():
            raise RuntimeError(f"指定的 source_record 不存在: {source_record}")
    else:
        latest_dir = sorted(run_dirs, key=lambda path: path.name)[-1]
    manifest_path = latest_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = pd.read_csv(latest_dir / "ledger.csv", encoding="utf-8-sig")
    pnl_summary = pd.read_csv(latest_dir / "pnl_summary.csv", encoding="utf-8-sig")
    return ledger, manifest, pnl_summary


def _allocate_run_timestamp(base_dir: Path, now: datetime) -> str:
    base = now.strftime("%Y%m%d_%H%M")
    candidate = base
    counter = 1
    while (base_dir / candidate).exists():
        candidate = f"{base}_{counter:02d}"
        counter += 1
    return candidate


def _format_symbol_set(symbols: list[str]) -> str:
    return " / ".join(symbols)


def _collect_suspension_exceptions(completeness_report: dict[str, Any]) -> list[dict[str, str]]:
    exceptions: list[dict[str, str]] = []
    for category in completeness_report.get("categories_checked", []):
        payload = completeness_report.get("categories", {}).get(category, {})
        for item in payload.get("suspended_stale_symbols", []):
            exceptions.append(
                {
                    "symbol": str(item.get("symbol", "")).upper(),
                    "latest_local_date": str(item.get("latest_local_date", "")),
                    "category": str(category).upper(),
                }
            )
    exceptions.sort(key=lambda row: (row["symbol"], row["category"], row["latest_local_date"]))
    return exceptions


def _format_suspension_exception_lines(completeness_report: dict[str, Any]) -> list[str]:
    exceptions = _collect_suspension_exceptions(completeness_report)
    if not exceptions:
        return []

    return [
        f"`{item['symbol']}`（{item['category']}，本地最新 `{item['latest_local_date']}`）"
        for item in exceptions
    ]


def _build_blocking_report(
    timestamp_long: str,
    source_record: str,
    completeness_before: dict[str, Any],
    completeness_after: dict[str, Any],
    attempted_backfill: bool,
) -> str:
    suspension_lines = _format_suspension_exception_lines(completeness_after)
    lines = [
        "# A股激进科技制造策略正式复盘报告",
        "",
        "## 1. 记录信息",
        "",
        "- 市场：A股（CN）",
        "- 策略：`aggressive_tech_manufacturing`",
        f"- 上一条正式记录：`{source_record}`",
        f"- 本次正式记录时间：{timestamp_long}",
        "- 完整性校验：**未通过**",
        "",
        "## 0. 正式结果速览",
        "",
        "- 正式结论：**不生成正式投资结论。**",
        (
            f"- 数据完整性状态：**未通过（目标最新交易日 `{completeness_after['latest_trade_date']}`，"
            f"阻塞缺口 `{completeness_after['blocking_incomplete_count']}` 个）**"
        ),
        "- 今日是否执行调仓：**否**",
        "- 明日准备事项：先补齐阻塞缺口，再重跑正式流程。",
        "",
        "## 2. 阻塞原因",
        "",
        (
            f"- 首轮完整性：`{'通过' if completeness_before['complete'] else '未通过'}`，"
            f"阻塞缺口 `{completeness_before['blocking_incomplete_count']}` 个"
        ),
        f"- 是否执行补数：`{'是' if attempted_backfill else '否'}`",
        (
            f"- 最终完整性：`{'通过' if completeness_after['complete'] else '未通过'}`，"
            f"阻塞缺口 `{completeness_after['blocking_incomplete_count']}` 个"
        ),
    ]
    for category in completeness_after["categories_checked"]:
        payload = completeness_after["categories"][category]
        missing_sample = payload.get("blocking_missing_symbols", [])[:10]
        stale_sample = [item["symbol"] for item in payload.get("blocking_stale_symbols", [])[:10]]
        lines.append(
            f"- {category}: 缺失 `{len(payload.get('blocking_missing_symbols', []))}` 个，"
            f"陈旧 `{len(payload.get('blocking_stale_symbols', []))}` 个，"
            f"示例缺口 `{', '.join(missing_sample + stale_sample) if (missing_sample or stale_sample) else '无'}`"
        )
    if suspension_lines:
        lines.append(
            "- 已识别为停牌非阻塞例外："
            + "；".join(suspension_lines)
        )
    lines.extend(
        [
            "",
            "## 3. 下一步补数建议",
            "",
            "- 继续使用 `quant_investor.market.download_cn` 对阻塞样本做单点重试。",
            "- 若仅剩停牌样本，必须核验 `suspend_d` 口径后再决定是否转为非阻塞例外。",
            "- 在完整数据校验通过前，不生成正式投资结论，也不更新调仓建议。",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    base_dir: Path,
    run_dir: Path,
    report_text: str,
    holdings_review: pd.DataFrame,
    ledger: pd.DataFrame,
    orders_df: pd.DataFrame,
    pnl_summary_df: pd.DataFrame,
    manifest: dict[str, Any],
    market_snapshot: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "analysis_report.md"
    holdings_path = run_dir / "holdings_review.csv"
    ledger_path = run_dir / "ledger.csv"
    orders_path = run_dir / "orders.csv"
    pnl_path = run_dir / "pnl_summary.csv"
    snapshot_path = run_dir / "market_snapshot.json"
    manifest_path = run_dir / "manifest.json"

    report_path.write_text(report_text, encoding="utf-8")
    holdings_review.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    orders_df.to_csv(orders_path, index=False, encoding="utf-8-sig")
    pnl_summary_df.to_csv(pnl_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(market_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    prefix = f"aggressive_portfolio_{manifest['timestamp']}_formal"
    shutil.copy2(report_path, raw_dir / f"{prefix}_report.md")
    shutil.copy2(holdings_path, raw_dir / f"{prefix}_holdings_review.csv")
    shutil.copy2(ledger_path, raw_dir / f"{prefix}_ledger.csv")
    shutil.copy2(orders_path, raw_dir / f"{prefix}_orders.csv")
    shutil.copy2(pnl_path, raw_dir / f"{prefix}_pnl_summary.csv")


def _build_notes_payload(
    trade_date: str,
    data_status: str,
    market_core_view: str,
    pnl_summary: dict[str, Any],
    orders: list[ProposedOrder],
    tomorrow_focus: list[str],
) -> str:
    if orders:
        order_text = "；".join(
            f"{order.symbol} {order.action} {order.shares}股 @ {order.price:.2f}" for order in orders
        )
    else:
        order_text = "无，本日维持现有结构。"

    return "\n".join(
        [
            f"# A股日度复盘 {trade_date}",
            "",
            f"- 数据完整性：{data_status}",
            f"- 市场核心判断：{market_core_view}",
            (
                f"- 组合盈亏：截至 `{pnl_summary['quote_snapshot']}`，总资产 "
                f"`{pnl_summary['total_value_after']:,.2f} 元`，较初始资金 "
                f"`{pnl_summary['portfolio_pnl_after']:,.2f} 元` "
                f"（{pnl_summary['portfolio_pnl_pct_after']:.2%}），较上一条正式记录变动 "
                f"`{pnl_summary['delta_vs_source_record']:,.2f} 元`。"
            ),
            f"- 是否调仓：{'是' if orders else '否'}",
            f"- 调仓内容：{order_text}",
            f"- 明日观察重点：{'；'.join(tomorrow_focus)}",
        ]
    )


def _to_plain_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_plain_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_payload(item) for item in value]
    return str(value)


def _build_review_recall_context(
    source_record: str,
    source_manifest: dict[str, Any],
    source_pnl: pd.DataFrame,
    source_ledger: pd.DataFrame,
) -> dict[str, Any]:
    latest_total_value = _safe_float(
        source_pnl["total_value_after"].iloc[-1] if "total_value_after" in source_pnl.columns else 0.0
    )
    latest_cash = _safe_float(
        source_pnl["cash_after"].iloc[-1] if "cash_after" in source_pnl.columns else 0.0
    )
    return {
        "source_record": source_record,
        "prior_quote_snapshot": str(source_manifest.get("quote_snapshot", "")),
        "prior_action_taken": bool(source_manifest.get("action_taken_today", False)),
        "prior_total_value": latest_total_value,
        "prior_cash_after": latest_cash,
        "existing_symbols": [str(symbol) for symbol in source_ledger["symbol"].tolist()],
        "existing_weights": {
            str(row.symbol): float(getattr(row, "market_weight", 0.0))
            for row in source_ledger.itertuples()
        },
    }


def _run_quantinvestor_review(
    symbols: list[str],
    source_record: str,
    source_manifest: dict[str, Any],
    source_pnl: pd.DataFrame,
    source_ledger: pd.DataFrame,
    total_capital: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    recall_context = _build_review_recall_context(
        source_record=source_record,
        source_manifest=source_manifest,
        source_pnl=source_pnl,
        source_ledger=source_ledger,
    )
    investor = QuantInvestor(
        stock_pool=list(symbols),
        market="CN",
        total_capital=total_capital,
        risk_level="中等",
        verbose=bool(args.review_verbose),
        enable_agent_layer=True,
        agent_model=str(args.review_agent_model),
        master_model=str(args.review_master_model),
        agent_timeout=float(args.review_agent_timeout),
        master_timeout=float(args.review_master_timeout),
        agent_total_timeout=float(args.review_total_timeout),
        recall_context=recall_context,
    )
    result = investor.run()

    master_output = result.master_review_output
    risk_output = result.risk_review_output
    trade_recommendations = [
        _to_plain_payload(item) for item in list(getattr(result.final_strategy, "trade_recommendations", []) or [])
    ]
    top_picks = [
        _to_plain_payload(item) for item in list(getattr(master_output, "top_picks", []) or [])
    ]
    branch_review_outputs = {
        name: (
            None
            if output is None
            else {
                "conviction": str(output.conviction),
                "conviction_score": float(output.conviction_score),
                "confidence": float(output.confidence),
                "key_insights": list(output.key_insights),
                "risk_flags": list(output.risk_flags),
                "disagreements_with_algo": list(output.disagreements_with_algo),
                "symbol_views": dict(output.symbol_views),
            }
        )
        for name, output in result.branch_review_outputs.items()
    }

    summary = {
        "agent_layer_enabled": bool(result.agent_layer_enabled),
        "master_review_ready": bool(master_output is not None),
        "risk_review_ready": bool(risk_output is not None),
        "agent_model": str(args.review_agent_model),
        "master_model": str(args.review_master_model),
        "branch_review_outputs": branch_review_outputs,
        "branch_review_success": {
            name: bool(output is not None)
            for name, output in result.branch_review_outputs.items()
        },
        "master_conviction": str(getattr(master_output, "final_conviction", "neutral")),
        "master_score": float(getattr(master_output, "final_score", 0.0) or 0.0),
        "master_confidence": float(getattr(master_output, "confidence", 0.0) or 0.0),
        "consensus_areas": list(getattr(master_output, "consensus_areas", []) or []),
        "disagreement_areas": list(getattr(master_output, "disagreement_areas", []) or []),
        "conviction_drivers": list(getattr(master_output, "conviction_drivers", []) or []),
        "debate_resolution": list(getattr(master_output, "debate_resolution", []) or []),
        "dissenting_views": list(getattr(master_output, "dissenting_views", []) or []),
        "portfolio_narrative": str(getattr(master_output, "portfolio_narrative", "") or ""),
        "top_picks": top_picks,
        "risk_assessment": str(getattr(risk_output, "risk_assessment", "")),
        "max_recommended_exposure": float(getattr(risk_output, "max_recommended_exposure", 0.0) or 0.0),
        "risk_warnings": list(getattr(risk_output, "risk_warnings", []) or []),
        "hedging_suggestions": list(getattr(risk_output, "hedging_suggestions", []) or []),
        "target_exposure": float(getattr(result.final_strategy, "target_exposure", 0.0) or 0.0),
        "candidate_symbols": list(getattr(result.final_strategy, "candidate_symbols", []) or []),
        "target_weights": {
            str(symbol): float(weight)
            for symbol, weight in dict(getattr(result.final_strategy, "target_weights", {}) or {}).items()
        },
        "trade_recommendations": trade_recommendations,
        "final_report": str(result.final_report or ""),
        "execution_log": list(getattr(result, "execution_log", []) or []),
    }
    summary["formal_ready"] = bool(
        summary["agent_layer_enabled"]
        and master_output is not None
        and risk_output is not None
    )
    return summary


def _review_recommendation_map(review_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    combined_items = list(review_summary.get("trade_recommendations", []) or []) + list(
        review_summary.get("top_picks", []) or []
    )
    for item in combined_items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper()
        if not symbol:
            continue
        result[symbol] = item
    return result


def _build_review_layer_blocking_report(
    timestamp_long: str,
    source_record: str,
    latest_trade_date: str,
    review_summary: dict[str, Any],
) -> str:
    lines = [
        "# A股激进科技制造策略正式复盘报告",
        "",
        "## 1. 记录信息",
        "",
        "- 市场：A股（CN）",
        "- 策略：`aggressive_tech_manufacturing`",
        f"- 上一条正式记录：`{source_record}`",
        f"- 本次正式记录时间：{timestamp_long}",
        "- 数据完整性：**已通过**",
        "- Review Layer：**未通过正式门槛**",
        "",
        "## 0. 正式结果速览",
        "",
        "- 正式结论：**不生成正式投资结论。**",
        f"- 数据完整性状态：**完整数据校验通过（目标最新交易日 `{latest_trade_date}`）**",
        "- 今日是否执行调仓：**否**",
        "- 明日准备事项：先修复主线 review layer，再重跑正式流程。",
        "",
        "## 2. 阻塞原因",
        "",
        (
            f"- agent_layer_enabled：`{review_summary.get('agent_layer_enabled', False)}`；"
            f"master review 就绪：`{review_summary.get('master_review_ready', False)}`；"
            f"risk review 就绪：`{review_summary.get('risk_review_ready', False)}`"
        ),
        f"- 使用模型：branch=`{review_summary.get('agent_model', '')}`，master=`{review_summary.get('master_model', '')}`",
        "- 说明：本轮要求必须走 `QuantInvestor` 主线并完成结构化 review layer；当前未达到正式输出门槛。",
    ]
    warnings = list(review_summary.get("risk_warnings", []) or [])
    if warnings:
        lines.append("- review layer 风险提示：" + "；".join(warnings[:3]))
    return "\n".join(lines)


def _write_review_layer_outputs(
    run_dir: Path,
    timestamp: str,
    review_summary: dict[str, Any],
) -> None:
    raw_dir = run_dir / "raw_exports"
    report_path = run_dir / "quantinvestor_review_report.md"
    summary_path = run_dir / "quantinvestor_review_summary.json"
    report_path.write_text(str(review_summary.get("final_report", "")), encoding="utf-8")
    summary_path.write_text(json.dumps(review_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(report_path, raw_dir / f"aggressive_portfolio_{timestamp}_quantinvestor_review_report.md")
    shutil.copy2(summary_path, raw_dir / f"aggressive_portfolio_{timestamp}_quantinvestor_review_summary.json")


def run_tracker(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    now = _now_local()
    timestamp = _allocate_run_timestamp(Path(args.base_dir), now)
    timestamp_long = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    base_dir = Path(args.base_dir)
    run_dir = base_dir / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    source_ledger, source_manifest, source_pnl = _load_previous_record(
        base_dir,
        source_record=args.source_record,
    )
    source_record = str(source_manifest.get("timestamp") or source_ledger.iloc[0].get("source_record", "unknown"))
    cash_before = _safe_float(source_pnl["cash_after"].iloc[-1] if "cash_after" in source_pnl.columns else 0.0)
    initial_capital = _safe_float(source_manifest.get("capital_cny"), DEFAULT_INITIAL_CAPITAL)
    source_total_value = _safe_float(
        source_pnl["total_value_after"].iloc[-1] if "total_value_after" in source_pnl.columns else initial_capital
    )

    downloader = CNFullMarketDownloader(
        data_dir=str(PROJECT_ROOT / "data" / "cn_market_full"),
        years=args.years,
        max_workers=4,
    )
    components = downloader.load_components()
    completeness_before = downloader.build_completeness_report(components=components)
    attempted_backfill = False
    download_report_path = None
    download_results: dict[str, Any] | None = None

    if not completeness_before["complete"]:
        attempted_backfill = True
        download_results = downloader.download_all(
            components=components,
            max_rounds=args.max_rounds,
            fail_on_incomplete=False,
        )
        completeness_after = download_results["completeness"]
        download_report_path = (
            PROJECT_ROOT
            / "data"
            / "cn_market_full"
            / f"download_report_{download_results['timestamp']}.json"
        )
    else:
        completeness_after = completeness_before

    latest_trade_date = completeness_after["latest_trade_date"]
    completeness_passed = bool(completeness_after["complete"])

    if not completeness_passed:
        report_text = _build_blocking_report(
            timestamp_long=timestamp_long,
            source_record=source_record,
            completeness_before=completeness_before,
            completeness_after=completeness_after,
            attempted_backfill=attempted_backfill,
        )
        empty_orders = pd.DataFrame(
            columns=["timestamp", "action", "symbol", "name", "shares", "price", "trade_value", "realized_pnl", "reason"]
        )
        pnl_summary = {
            "record_time": timestamp_long,
            "quote_snapshot": "",
            "initial_capital": initial_capital,
            "cash_before": cash_before,
            "market_value_before": 0.0,
            "total_value_before": source_total_value,
            "portfolio_pnl_before": round(source_total_value - initial_capital, 2),
            "portfolio_pnl_pct_before": round(_safe_pct(source_total_value - initial_capital, initial_capital), 6),
            "realized_pnl_from_rebalance": 0.0,
            "cash_after": cash_before,
            "market_value_after": 0.0,
            "total_value_after": source_total_value,
            "portfolio_pnl_after": round(source_total_value - initial_capital, 2),
            "portfolio_pnl_pct_after": round(_safe_pct(source_total_value - initial_capital, initial_capital), 6),
            "delta_vs_source_record": 0.0,
        }
        pnl_summary_df = pd.DataFrame([pnl_summary])
        notes_text = _build_notes_payload(
            trade_date=now.strftime("%Y-%m-%d"),
            data_status=(
                f"完整性未通过（目标最新交易日 `{latest_trade_date}`，"
                f"阻塞缺口 `{completeness_after['blocking_incomplete_count']}` 个）"
            ),
            market_core_view="本轮先停在补数与校验阶段，不生成正式投资判断。",
            pnl_summary=pnl_summary,
            orders=[],
            tomorrow_focus=["优先补齐阻塞缺口", "确认停牌样本是否可转为非阻塞例外"],
        )
        DEFAULT_NOTES_PATH.write_text(notes_text, encoding="utf-8")
        market_snapshot = {
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "quote_snapshot": "",
            "completeness_before": completeness_before,
            "completeness_after": completeness_after,
            "download_report": str(download_report_path) if download_report_path else None,
        }
        manifest = {
            "market": "CN",
            "strategy": "aggressive_tech_manufacturing",
            "timestamp": timestamp,
            "recorded_at": timestamp_long,
            "source_record": source_record,
            "formal_record": False,
            "completeness_passed": False,
            "capital_cny": initial_capital,
            "quote_snapshot": "",
            "action_taken_today": False,
            "files": {
                "analysis_report": "analysis_report.md",
                "holdings_review": "holdings_review.csv",
                "orders": "orders.csv",
                "ledger": "ledger.csv",
                "pnl_summary": "pnl_summary.csv",
                "market_snapshot": "market_snapshot.json",
            },
            "raw_exports": {
                "report": f"raw_exports/aggressive_portfolio_{timestamp}_formal_report.md",
                "orders": f"raw_exports/aggressive_portfolio_{timestamp}_formal_orders.csv",
                "ledger": f"raw_exports/aggressive_portfolio_{timestamp}_formal_ledger.csv",
                "pnl_summary": f"raw_exports/aggressive_portfolio_{timestamp}_formal_pnl_summary.csv",
                "holdings_review": f"raw_exports/aggressive_portfolio_{timestamp}_formal_holdings_review.csv",
            },
            "data_snapshot": {
                "latest_trade_date": latest_trade_date,
                "completeness": completeness_after,
                "download_report": str(download_report_path) if download_report_path else None,
            },
        }
        _write_outputs(
            base_dir=base_dir,
            run_dir=run_dir,
            report_text=report_text,
            holdings_review=source_ledger,
            ledger=source_ledger,
            orders_df=empty_orders,
            pnl_summary_df=pnl_summary_df,
            manifest=manifest,
            market_snapshot=market_snapshot,
        )
        shortcuts_result = {"success": False, "returncode": None, "stdout": "", "stderr": "数据完整性未通过，未执行 Shortcuts。"}
        return {
            "timestamp": timestamp,
            "timestamp_long": timestamp_long,
            "run_dir": str(run_dir),
            "completeness_passed": False,
            "latest_trade_date": latest_trade_date,
            "action_taken_today": False,
            "shortcuts_result": shortcuts_result,
            "elapsed_sec": round(time.time() - started, 2),
        }

    review_summary = _run_quantinvestor_review(
        symbols=[str(symbol) for symbol in source_ledger["symbol"].tolist()],
        source_record=source_record,
        source_manifest=source_manifest,
        source_pnl=source_pnl,
        source_ledger=source_ledger,
        total_capital=max(source_total_value, 0.0) or initial_capital,
        args=args,
    )
    review_ready = bool(review_summary.get("formal_ready"))

    if not review_ready:
        report_text = _build_review_layer_blocking_report(
            timestamp_long=timestamp_long,
            source_record=source_record,
            latest_trade_date=latest_trade_date,
            review_summary=review_summary,
        )
        empty_orders = pd.DataFrame(
            columns=["timestamp", "action", "symbol", "name", "shares", "price", "trade_value", "realized_pnl", "reason"]
        )
        pnl_summary = {
            "record_time": timestamp_long,
            "quote_snapshot": "",
            "initial_capital": initial_capital,
            "cash_before": cash_before,
            "market_value_before": 0.0,
            "total_value_before": source_total_value,
            "portfolio_pnl_before": round(source_total_value - initial_capital, 2),
            "portfolio_pnl_pct_before": round(_safe_pct(source_total_value - initial_capital, initial_capital), 6),
            "realized_pnl_from_rebalance": 0.0,
            "cash_after": cash_before,
            "market_value_after": 0.0,
            "total_value_after": source_total_value,
            "portfolio_pnl_after": round(source_total_value - initial_capital, 2),
            "portfolio_pnl_pct_after": round(_safe_pct(source_total_value - initial_capital, initial_capital), 6),
            "delta_vs_source_record": 0.0,
        }
        pnl_summary_df = pd.DataFrame([pnl_summary])
        notes_text = _build_notes_payload(
            trade_date=now.strftime("%Y-%m-%d"),
            data_status=f"完整数据校验通过（目标最新交易日 `{latest_trade_date}`）",
            market_core_view="数据完整，但 `QuantInvestor` 主线 review layer 未完成正式门槛，本轮不生成投资结论。",
            pnl_summary=pnl_summary,
            orders=[],
            tomorrow_focus=["优先修复 review layer 稳定性", "确认 master review 与 risk review 均成功后再重跑"],
        )
        DEFAULT_NOTES_PATH.write_text(notes_text, encoding="utf-8")
        market_snapshot = {
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "quote_snapshot": "",
            "completeness_before": completeness_before,
            "completeness_after": completeness_after,
            "download_report": str(download_report_path) if download_report_path else None,
            "review_layer": review_summary,
        }
        manifest = {
            "market": "CN",
            "strategy": "aggressive_tech_manufacturing",
            "timestamp": timestamp,
            "recorded_at": timestamp_long,
            "source_record": source_record,
            "formal_record": False,
            "completeness_passed": True,
            "review_layer_passed": False,
            "capital_cny": initial_capital,
            "quote_snapshot": "",
            "action_taken_today": False,
            "files": {
                "analysis_report": "analysis_report.md",
                "holdings_review": "holdings_review.csv",
                "orders": "orders.csv",
                "ledger": "ledger.csv",
                "pnl_summary": "pnl_summary.csv",
                "market_snapshot": "market_snapshot.json",
                "quantinvestor_review_report": "quantinvestor_review_report.md",
                "quantinvestor_review_summary": "quantinvestor_review_summary.json",
            },
            "raw_exports": {
                "report": f"raw_exports/aggressive_portfolio_{timestamp}_formal_report.md",
                "orders": f"raw_exports/aggressive_portfolio_{timestamp}_formal_orders.csv",
                "ledger": f"raw_exports/aggressive_portfolio_{timestamp}_formal_ledger.csv",
                "pnl_summary": f"raw_exports/aggressive_portfolio_{timestamp}_formal_pnl_summary.csv",
                "holdings_review": f"raw_exports/aggressive_portfolio_{timestamp}_formal_holdings_review.csv",
                "quantinvestor_review_report": f"raw_exports/aggressive_portfolio_{timestamp}_quantinvestor_review_report.md",
                "quantinvestor_review_summary": f"raw_exports/aggressive_portfolio_{timestamp}_quantinvestor_review_summary.json",
            },
            "data_snapshot": {
                "latest_trade_date": latest_trade_date,
                "completeness": completeness_after,
                "download_report": str(download_report_path) if download_report_path else None,
            },
            "review_layer": review_summary,
        }
        _write_outputs(
            base_dir=base_dir,
            run_dir=run_dir,
            report_text=report_text,
            holdings_review=source_ledger,
            ledger=source_ledger,
            orders_df=empty_orders,
            pnl_summary_df=pnl_summary_df,
            manifest=manifest,
            market_snapshot=market_snapshot,
        )
        _write_review_layer_outputs(
            run_dir=run_dir,
            timestamp=timestamp,
            review_summary=review_summary,
        )
        shortcuts_result = {"success": False, "returncode": None, "stdout": "", "stderr": "Review Layer 未通过正式门槛，未执行 Shortcuts。"}
        return {
            "timestamp": timestamp,
            "timestamp_long": timestamp_long,
            "run_dir": str(run_dir),
            "completeness_passed": True,
            "review_layer_passed": False,
            "latest_trade_date": latest_trade_date,
            "action_taken_today": False,
            "shortcuts_result": shortcuts_result,
            "elapsed_sec": round(time.time() - started, 2),
        }

    full_metrics = _compute_full_market_metrics(
        components=components,
        data_root=PROJECT_ROOT / "data" / "cn_market_full",
        latest_trade_date=latest_trade_date,
    )
    metrics_map = {
        row.symbol: row._asdict()
        for row in full_metrics.itertuples(index=False)
    }

    holding_quote_codes = [_map_symbol_to_quote_code(symbol) for symbol in source_ledger["symbol"]]
    index_quote_codes = list(INDEX_QUOTES.keys())
    quote_error = ""
    try:
        quote_payload = _fetch_tencent_quotes(index_quote_codes + holding_quote_codes)
    except Exception as exc:
        quote_payload = {}
        quote_error = str(exc)
    quote_snapshot = max(
        [quote_payload.get(code, {}).get("time", "") for code in index_quote_codes + holding_quote_codes],
        default="",
    )

    indices = {
        code: {
            **quote_payload[code],
            "name": INDEX_QUOTES[code],
        }
        for code in index_quote_codes
        if code in quote_payload
    }

    breadth = {
        category: _compute_category_breadth(
            category=category,
            symbols=components.get(category, []),
            data_root=PROJECT_ROOT / "data" / "cn_market_full",
            latest_trade_date=latest_trade_date,
            completeness_report=completeness_after,
        )
        for category in ("hs300", "zz500", "zz1000")
    }

    current_rows: list[dict[str, Any]] = []
    previous_value_map = {
        row.symbol: float(getattr(row, "current_value", 0.0))
        for row in source_ledger.itertuples()
    }
    for row in source_ledger.itertuples():
        symbol = row.symbol
        metric = metrics_map.get(symbol, {})
        quote = quote_payload.get(_map_symbol_to_quote_code(symbol), {})
        current_price = _safe_float(quote.get("current"), _safe_float(metric.get("latest_close"), getattr(row, "current_price", 0.0)))
        current_value = round(int(row.shares) * current_price, 2)
        buy_value = round(float(row.cost_basis), 2)
        unrealized = round(current_value - buy_value, 2)
        today_change_pct = round(_safe_float(quote.get("change_pct")), 2)
        staged_target = round(_safe_float(metric.get("stage_target_price"), getattr(row, "stage_target_price", current_price * 1.1)), 2)
        staged_stop = round(_safe_float(metric.get("stage_stop_price"), getattr(row, "stage_stop_price", current_price * 0.94)), 2)
        current_rows.append(
            {
                "symbol": symbol,
                "name": row.name,
                "category": metric.get("category", ""),
                "shares_before": int(row.shares),
                "buy_price": round(float(row.avg_cost), 6),
                "buy_value": buy_value,
                "current_price": round(current_price, 2),
                "current_value": current_value,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": round(_safe_pct(unrealized, buy_value), 6),
                "today_change_pct": today_change_pct,
                "ret5": round(_safe_float(metric.get("ret5")), 6),
                "ret20": round(_safe_float(metric.get("ret20")), 6),
                "ret60": round(_safe_float(metric.get("ret60")), 6),
                "close_vs_ma20": round(_safe_float(metric.get("close_vs_ma20")), 6),
                "ma20_vs_ma60": round(_safe_float(metric.get("ma20_vs_ma60")), 6),
                "ma60_vs_ma120": round(_safe_float(metric.get("ma60_vs_ma120")), 6),
                "dd20": round(_safe_float(metric.get("dd20")), 6),
                "rank_full_market": int(metric["rank_full_market"]) if metric and "rank_full_market" in metric else 9999,
                "score_full_market": round(_safe_float(metric.get("score_full_market")), 6),
                "stage_target_price": staged_target,
                "stage_stop_price": staged_stop,
                "delta_vs_source_record": round(current_value - previous_value_map.get(symbol, 0.0), 2),
            }
        )

    holdings_review = pd.DataFrame(current_rows)
    total_market_value_before = round(float(holdings_review["current_value"].sum()), 2)
    total_value_before = round(total_market_value_before + cash_before, 2)
    holdings_review["market_weight"] = (
        holdings_review["current_value"] / total_market_value_before if total_market_value_before > 0 else 0.0
    ).round(6)
    holdings_review["position_role"] = holdings_review.apply(_position_role, axis=1)
    holdings_review["recommended_action"] = holdings_review.apply(_position_action, axis=1)
    holdings_review["reason"] = holdings_review.apply(_position_reason, axis=1)
    review_recommendations = _review_recommendation_map(review_summary)
    holdings_review["review_layer_action"] = holdings_review["symbol"].map(
        lambda symbol: str(review_recommendations.get(str(symbol).upper(), {}).get("action", "hold"))
    )
    holdings_review["review_layer_target_weight"] = holdings_review["symbol"].map(
        lambda symbol: round(
            _safe_float(review_recommendations.get(str(symbol).upper(), {}).get("weight")),
            6,
        )
    )
    holdings_review["review_layer_confidence"] = holdings_review["symbol"].map(
        lambda symbol: round(
            _safe_float(review_recommendations.get(str(symbol).upper(), {}).get("confidence")),
            6,
        )
    )
    holdings_review["review_layer_rationale"] = holdings_review["symbol"].map(
        lambda symbol: str(review_recommendations.get(str(symbol).upper(), {}).get("rationale", ""))
    )
    holdings_review = holdings_review.sort_values(
        by=["score_full_market", "today_change_pct", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    theme_strength = _summarize_theme_strength(holdings_review)
    style_text = _market_style_conclusion(indices=indices, breadth=breadth)
    strongest_theme_text, weakest_theme_text = _tech_mainline_conclusion(theme_strength)
    orders = _build_rebalance_plan(holdings_review, review_recommendations)
    # 当前策略仍以主线内部修复为主，只在出现明确弱化共振时执行减仓。
    if len(orders) == 1 and theme_strength and theme_strength[0]["avg_score"] >= 0.8:
        orders = []

    order_rows = []
    for order in orders:
        name = source_ledger[source_ledger["symbol"] == order.symbol]["name"].iloc[0]
        order_rows.append(
            {
                "timestamp": timestamp_long,
                "action": order.action,
                "symbol": order.symbol,
                "name": name,
                "shares": order.shares,
                "price": order.price,
                "trade_value": order.trade_value,
                "realized_pnl": order.realized_pnl,
                "reason": order.reason,
            }
        )
    orders_df = pd.DataFrame(
        order_rows,
        columns=["timestamp", "action", "symbol", "name", "shares", "price", "trade_value", "realized_pnl", "reason"],
    )

    quote_prices = {row["symbol"]: float(row["current_price"]) for row in current_rows}
    updated_ledger, cash_after, realized_pnl = _apply_orders(
        source_ledger=source_ledger,
        orders=orders,
        cash_before=cash_before,
        quote_prices=quote_prices,
    )
    ledger_meta = holdings_review[
        ["symbol", "stage_target_price", "stage_stop_price", "position_role"]
    ].rename(columns={"position_role": "thesis_status"})
    updated_ledger = updated_ledger.merge(ledger_meta, on="symbol", how="left")

    total_market_value_after = round(float(updated_ledger["current_value"].sum()), 2)
    total_value_after = round(total_market_value_after + cash_after, 2)
    portfolio_pnl_after = round(total_value_after - initial_capital, 2)
    portfolio_pnl_before = round(total_value_before - initial_capital, 2)

    contribution = holdings_review.sort_values("delta_vs_source_record", ascending=False)
    detractors = holdings_review.sort_values("delta_vs_source_record", ascending=True)
    float_winners = holdings_review.sort_values("unrealized_pnl", ascending=False)
    float_losers = holdings_review.sort_values("unrealized_pnl", ascending=True)
    held_symbols = {str(symbol).upper() for symbol in holdings_review["symbol"].tolist()}
    held_review_sell_symbols = [
        symbol
        for symbol, payload in review_recommendations.items()
        if symbol in held_symbols and str(payload.get("action", "")).strip().lower() == "sell"
    ]

    master_conviction = str(review_summary.get("master_conviction", "neutral"))
    risk_assessment = str(review_summary.get("risk_assessment", ""))
    max_recommended_exposure = _safe_float(review_summary.get("max_recommended_exposure"))
    review_narrative = str(review_summary.get("portfolio_narrative", "")).strip()
    rebalance_reason = "今天不执行调仓，继续把强修复与弱滞涨的分化再观察一个交易日。"
    if orders:
        rebalance_reason = "执行温和调仓，结构化 Review Layer 与持仓弱化信号同时指向减仓。"
    elif held_review_sell_symbols:
        rebalance_reason = (
            "今天不执行调仓，但 Review Layer 已点名弱势持仓进入减仓观察，"
            "先等待一次进一步确认。"
        )
    elif master_conviction in {"sell", "strong_sell"}:
        rebalance_reason = "今天先不追着改单，但 Review Layer 已明显转谨慎，弱势仓位进入下一次确认窗口。"
    elif risk_assessment in {"high", "extreme"} and max_recommended_exposure <= 0.35:
        rebalance_reason = "今天不新增风险暴露，先按高风险环境处理，保留组合观察位。"
    elif master_conviction in {"buy", "strong_buy"} and review_narrative:
        rebalance_reason = "今天不执行调仓，Review Layer 仍认可主线，但先观察结构修复的持续性。"
    suspension_lines = _format_suspension_exception_lines(completeness_after)
    suspension_status_text = (
        "；停牌例外 "
        + "、".join(suspension_lines)
        + " 已识别为非阻塞"
        if suspension_lines
        else ""
    )
    data_status = (
        f"完整数据校验通过（目标最新交易日 `{latest_trade_date}`"
        + suspension_status_text
        + "）"
    )
    tomorrow_focus = [
        "确认先进材料与光通信的修复能否延续，不让单日反弹误判为全面重启",
        "继续观察 `大族激光 / 中国西电` 是否能重新站回阶段止损位",
        "跟踪科创50 相对沪深300 的强弱差，判断资金是否继续偏向硬科技",
    ]
    review_top_pick_symbols = [
        str(item.get("symbol", "")).upper()
        for item in review_summary.get("top_picks", [])
        if isinstance(item, dict) and str(item.get("symbol", "")).strip()
    ]
    held_review_sells = holdings_review[
        holdings_review["review_layer_action"].astype(str).str.lower() == "sell"
    ]

    report_lines = [
        "# A股激进科技制造策略正式复盘报告",
        "",
        "## 1. 记录信息",
        "",
        "- 市场：A股（CN）",
        "- 策略：`aggressive_tech_manufacturing`",
        f"- 上一条正式记录：`{source_record}`",
        f"- 本次正式记录时间：{timestamp_long}",
        f"- 盘中快照：{quote_snapshot or 'N/A'}",
        "- 完整性校验：**已通过**",
        "",
        "## 0. 正式结果速览",
        "",
        f"- 正式结论：**{rebalance_reason}**",
        f"- 数据完整性状态：**{data_status}**",
        f"- 今日是否执行调仓：**{'是' if orders else '否'}**",
        f"- 明日准备事项：{'；'.join(tomorrow_focus)}",
        "",
        "## 2. 数据完整性结论",
        "",
        f"- 当前运行时间：`{timestamp_long}`",
        f"- 首轮完整性：`{'通过' if completeness_before['complete'] else '未通过'}`，阻塞缺口 `{completeness_before['blocking_incomplete_count']}` 个",
        f"- 是否执行补数：`{'是' if attempted_backfill else '否'}`",
        f"- 最终完整性：`{'通过' if completeness_after['complete'] else '未通过'}`，阻塞缺口 `{completeness_after['blocking_incomplete_count']}` 个",
        (
            f"- HS300：最新 `{breadth['hs300']['latest_count']}/{breadth['hs300']['expected']}`"
            + (
                f"，停牌非阻塞例外 `{breadth['hs300']['suspended_stale_count']}` 个"
                if breadth["hs300"]["suspended_stale_count"]
                else ""
            )
        ),
        (
            f"- ZZ500：最新 `{breadth['zz500']['latest_count']}/{breadth['zz500']['expected']}`"
            + (
                f"，停牌非阻塞例外 `{breadth['zz500']['suspended_stale_count']}` 个"
                if breadth["zz500"]["suspended_stale_count"]
                else ""
            )
        ),
        (
            f"- ZZ1000：最新 `{breadth['zz1000']['latest_count']}/{breadth['zz1000']['expected']}`"
            + (
                f"，停牌非阻塞例外 `{breadth['zz1000']['suspended_stale_count']}` 个"
                if breadth["zz1000"]["suspended_stale_count"]
                else ""
            )
        ),
        (
            "- 停牌非阻塞例外："
            + "；".join(suspension_lines)
            if suspension_lines
            else "- 本轮无停牌非阻塞例外。"
        ),
        (
            f"- 本轮下载报告：`{download_report_path}`"
            if download_report_path
            else "- 本轮未触发补数，直接沿用本地完整数据。"
        ),
        "",
        "## 3. QuantInvestor 主线 / Review Layer 结构化结论",
        "",
        f"- Review Layer 已启用：`{review_summary['agent_layer_enabled']}`，branch 成功 `{sum(1 for ok in review_summary['branch_review_success'].values() if ok)}/{len(review_summary['branch_review_success'])}`。",
        f"- 使用模型：branch=`{review_summary['agent_model']}`，master=`{review_summary['master_model']}`。",
        (
            f"- Master conviction：`{review_summary['master_conviction']}`，分数 `{review_summary['master_score']:.2f}`，"
            f"置信度 `{review_summary['master_confidence']:.2f}`。"
        ),
        (
            f"- Risk review：`{review_summary['risk_assessment']}`，建议最大总暴露 "
            f"`{review_summary['max_recommended_exposure']:.0%}`。"
        ),
        (
            "- Conviction drivers："
            + "；".join(review_summary["conviction_drivers"][:3])
            if review_summary.get("conviction_drivers")
            else "- Conviction drivers：暂无明确额外驱动。"
        ),
        (
            "- Risk warnings："
            + "；".join(review_summary["risk_warnings"][:3])
            if review_summary.get("risk_warnings")
            else "- Risk warnings：暂无新增结构化风险提示。"
        ),
        (
            "- Review Layer top picks："
            + (" / ".join(review_top_pick_symbols) if review_top_pick_symbols else "本轮无新增 top picks。")
        ),
        (
            f"- 组合叙事：{review_narrative}"
            if review_narrative
            else "- 组合叙事：本轮未给出新增组合叙事，维持结构化中性判断。"
        ),
        "",
        "## 4. 当前组合盈亏判断",
        "",
        f"- 截至 `quote_snapshot={quote_snapshot or 'N/A'}`，组合总资产 **{total_value_after:,.2f} 元**，较初始资金 **{portfolio_pnl_after:,.2f} 元**，收益率 **{portfolio_pnl_after / initial_capital:.2%}**。",
        f"- 相对上一条正式记录 `{source_record}` 的 **{source_total_value:,.2f} 元**，当前盘中净值变动 **{total_value_after - source_total_value:,.2f} 元**。",
        "- 当前浮盈仓位："
        + (
            "；".join(
                f"{row.symbol}({row.name}) {row.unrealized_pnl:,.2f} 元"
                for row in float_winners[float_winners["unrealized_pnl"] > 0].head(3).itertuples()
            )
            if not float_winners[float_winners["unrealized_pnl"] > 0].empty
            else "无"
        ),
        "- 当前浮亏前三："
        + "；".join(
            f"{row.symbol}({row.name}) {row.unrealized_pnl:,.2f} 元"
            for row in float_losers.head(3).itertuples()
        ),
        "- 相对上一条正式记录的正向收益贡献："
        + "；".join(
            f"{row.symbol} {row.delta_vs_source_record:,.2f} 元"
            for row in contribution.head(3).itertuples()
        ),
        "- 相对上一条正式记录的拖累来源前三："
        + "；".join(
            f"{row.symbol} {row.delta_vs_source_record:,.2f} 元"
            for row in detractors.head(3).itertuples()
        ),
        "",
        "## 5. A股整体市场风格与指数结构",
        "",
        "### 5.1 指数结构（盘中快照）",
        "",
    ]

    for code in index_quote_codes:
        payload = indices.get(code)
        if not payload:
            continue
        report_lines.append(f"- {payload['name']}：{payload['change_pct']:+.2f}%")

    report_lines.extend(
        [
            "",
            f"结论：{style_text}",
            "",
            "### 5.2 广度与市场内部状态（基于最新完整日线）",
            "",
            (
                f"- HS300：1日上涨占比 {breadth['hs300']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['hs300']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['hs300']['ma20_gt_ma60_ratio']:.1%}"
            ),
            (
                f"- ZZ500：1日上涨占比 {breadth['zz500']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['zz500']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['zz500']['ma20_gt_ma60_ratio']:.1%}"
            ),
            (
                f"- ZZ1000：1日上涨占比 {breadth['zz1000']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['zz1000']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['zz1000']['ma20_gt_ma60_ratio']:.1%}"
            ),
            "",
            "### 5.3 科技 / 高端制造主线强弱",
            "",
            f"- {strongest_theme_text}",
            f"- {weakest_theme_text}",
            (
                f"- 主题强弱排序：{', '.join(item['theme'] for item in theme_strength)}"
                if theme_strength
                else "- 主题强弱排序：暂无"
            ),
            "",
            "## 6. 当前策略持仓复盘",
            "",
            "### 6.1 持仓相对强弱",
            "",
            "- 今日相对最强："
            + _format_symbol_set(holdings_review.sort_values(
                ["today_change_pct", "score_full_market"], ascending=[False, False]
            ).head(3)["symbol"].tolist()),
            "- 今日相对最弱："
            + _format_symbol_set(holdings_review.sort_values(
                ["today_change_pct", "score_full_market"], ascending=[True, True]
            ).head(3)["symbol"].tolist()),
            "",
            "### 6.2 当前弱点与结构预警",
            "",
        ]
    )

    weak_rows = holdings_review[holdings_review["current_price"] < holdings_review["stage_stop_price"]]
    if weak_rows.empty:
        report_lines.append("- 目前持仓都在阶段止损位上方，尚未触发新的硬性减仓信号。")
    else:
        for row in weak_rows.itertuples():
            report_lines.append(
                f"- `{row.symbol}` 当前价 {row.current_price:.2f} 仍低于阶段止损位 {row.stage_stop_price:.2f}，"
                f"状态 `{row.position_role}`。"
            )

    report_lines.extend(
        [
            "",
            "### 6.3 Review Layer 对当前持仓的结构化态度",
            "",
            (
                "- Review Layer 明确给出 `sell` 的持仓："
                + "；".join(
                    f"{row.symbol}（目标权重 {row.review_layer_target_weight:.1%}，置信度 {row.review_layer_confidence:.2f}）"
                    for row in held_review_sells.itertuples()
                )
                if not held_review_sells.empty
                else "- Review Layer 本轮没有对现有持仓给出 `sell` 指令。"
            ),
            "",
            "### 6.4 是否需要调仓",
            "",
            f"- 正式建议：**{'执行温和调仓' if orders else '本次不执行调仓'}**",
            (
                (
                    "- 原因：虽然 Review Layer 已对个别弱势持仓给出 `sell`，但当前主线最强分支仍在修复，先等待下一次确认再执行。"
                    if held_review_sells is not None and not held_review_sells.empty
                    else "- 原因：结构化 Review Layer 没有对现有持仓确认 `sell`，因此仍按长期主线观察，不因单日波动推翻。"
                )
                if not orders
                else "- 原因：弱支线在完整日线与盘中都继续落后，且 Review Layer 已给出 `sell`，满足温和减仓条件。"
            ),
            "",
            "## 7. 面向明日的观察重点和准备事项",
            "",
        ]
    )
    for idx, item in enumerate(tomorrow_focus, start=1):
        report_lines.append(f"{idx}. {item}")

    report_text = "\n".join(report_lines)

    pnl_summary = {
        "record_time": timestamp_long,
        "quote_snapshot": quote_snapshot,
        "initial_capital": initial_capital,
        "cash_before": round(cash_before, 2),
        "market_value_before": total_market_value_before,
        "total_value_before": total_value_before,
        "portfolio_pnl_before": portfolio_pnl_before,
        "portfolio_pnl_pct_before": round(_safe_pct(portfolio_pnl_before, initial_capital), 6),
        "realized_pnl_from_rebalance": realized_pnl,
        "cash_after": round(cash_after, 2),
        "market_value_after": total_market_value_after,
        "total_value_after": total_value_after,
        "portfolio_pnl_after": portfolio_pnl_after,
        "portfolio_pnl_pct_after": round(_safe_pct(portfolio_pnl_after, initial_capital), 6),
        "delta_vs_source_record": round(total_value_after - source_total_value, 2),
    }
    pnl_summary_df = pd.DataFrame([pnl_summary])

    notes_text = _build_notes_payload(
        trade_date=now.strftime("%Y-%m-%d"),
        data_status=data_status,
        market_core_view=(
            f"{style_text}{strongest_theme_text}{weakest_theme_text}"
            + (f"Review Layer：{review_narrative}" if review_narrative else "")
        ),
        pnl_summary=pnl_summary,
        orders=orders,
        tomorrow_focus=tomorrow_focus,
    )
    DEFAULT_NOTES_PATH.write_text(notes_text, encoding="utf-8")

    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": timestamp,
        "recorded_at": timestamp_long,
        "source_record": source_record,
        "formal_record": True,
        "completeness_passed": True,
        "capital_cny": initial_capital,
        "quote_snapshot": quote_snapshot,
        "action_taken_today": bool(orders),
        "files": {
            "analysis_report": "analysis_report.md",
            "holdings_review": "holdings_review.csv",
            "orders": "orders.csv",
            "ledger": "ledger.csv",
            "pnl_summary": "pnl_summary.csv",
            "market_snapshot": "market_snapshot.json",
            "quantinvestor_review_report": "quantinvestor_review_report.md",
            "quantinvestor_review_summary": "quantinvestor_review_summary.json",
        },
        "raw_exports": {
            "report": f"raw_exports/aggressive_portfolio_{timestamp}_formal_report.md",
            "orders": f"raw_exports/aggressive_portfolio_{timestamp}_formal_orders.csv",
            "ledger": f"raw_exports/aggressive_portfolio_{timestamp}_formal_ledger.csv",
            "pnl_summary": f"raw_exports/aggressive_portfolio_{timestamp}_formal_pnl_summary.csv",
            "holdings_review": f"raw_exports/aggressive_portfolio_{timestamp}_formal_holdings_review.csv",
            "quantinvestor_review_report": f"raw_exports/aggressive_portfolio_{timestamp}_quantinvestor_review_report.md",
            "quantinvestor_review_summary": f"raw_exports/aggressive_portfolio_{timestamp}_quantinvestor_review_summary.json",
        },
        "data_snapshot": {
            "latest_trade_date": latest_trade_date,
            "completeness": completeness_after,
            "download_report": str(download_report_path) if download_report_path else None,
        },
        "review_layer": review_summary,
    }
    market_snapshot = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "quote_snapshot": quote_snapshot,
        "indices": indices,
        "breadth": breadth,
        "completeness": completeness_after,
        "portfolio": {
            "total_value": total_value_after,
            "portfolio_pnl": portfolio_pnl_after,
            "portfolio_pnl_pct": round(_safe_pct(portfolio_pnl_after, initial_capital), 6),
            "delta_vs_source_record": round(total_value_after - source_total_value, 2),
        },
        "theme_strength": theme_strength,
        "review_layer": review_summary,
        "download_report": str(download_report_path) if download_report_path else None,
        "quote_fetch_error": quote_error or None,
    }

    _write_outputs(
        base_dir=base_dir,
        run_dir=run_dir,
        report_text=report_text,
        holdings_review=holdings_review,
        ledger=updated_ledger,
        orders_df=orders_df,
        pnl_summary_df=pnl_summary_df,
        manifest=manifest,
        market_snapshot=market_snapshot,
    )
    _write_review_layer_outputs(
        run_dir=run_dir,
        timestamp=timestamp,
        review_summary=review_summary,
    )

    shortcuts_result = {"success": False, "returncode": None, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(
            [
                "shortcuts",
                "run",
                args.shortcut_name,
                "--input-path",
                str(DEFAULT_NOTES_PATH),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        shortcuts_result = {
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        shortcuts_result = {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }

    return {
        "timestamp": timestamp,
        "timestamp_long": timestamp_long,
        "run_dir": str(run_dir),
        "latest_trade_date": latest_trade_date,
        "completeness_passed": True,
        "review_layer_passed": True,
        "action_taken_today": bool(orders),
        "data_status": data_status,
        "style_view": style_text,
        "tech_mainline": {
            "strongest": strongest_theme_text,
            "weakest": weakest_theme_text,
        },
        "review_layer": {
            "master_conviction": review_summary["master_conviction"],
            "risk_assessment": review_summary["risk_assessment"],
            "max_recommended_exposure": review_summary["max_recommended_exposure"],
        },
        "pnl_summary": pnl_summary,
        "shortcuts_result": shortcuts_result,
        "elapsed_sec": round(time.time() - started, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A股激进科技制造策略正式复盘跟踪器")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--years", type=int, default=7)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--shortcut-name", default=DEFAULT_SHORTCUT_NAME)
    parser.add_argument("--source-record", default=None)
    parser.add_argument("--review-agent-model", default=DEFAULT_REVIEW_AGENT_MODEL)
    parser.add_argument("--review-master-model", default=DEFAULT_REVIEW_MASTER_MODEL)
    parser.add_argument("--review-agent-timeout", type=float, default=DEFAULT_REVIEW_AGENT_TIMEOUT)
    parser.add_argument("--review-master-timeout", type=float, default=DEFAULT_REVIEW_MASTER_TIMEOUT)
    parser.add_argument("--review-total-timeout", type=float, default=DEFAULT_REVIEW_TOTAL_TIMEOUT)
    parser.add_argument("--review-verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_tracker(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
