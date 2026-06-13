"""Quote, ranking, candidate, and rebalance helpers for the CN tracker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_investor.monitoring.cn_aggressive_utils import _safe_float, _safe_pct


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUOTE_TIMEOUT = 20
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
CANDIDATE_POOL_PATH = PROJECT_ROOT / "results" / "cn_analysis_full" / "all_candidates.json"
CATEGORY_THEME_LABELS = {
    "hs300": "大盘核心资产",
    "zz500": "中盘制造主线",
    "zz1000": "小盘成长弹性",
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


# Full-market metrics/cache helpers live in cn_aggressive_market_metrics.py.

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
        f"{strongest['theme']} 的完整日线强度最强，平均全市场评分 {strongest['avg_score']:.3f}，"
        f"盘中涨跌 {strongest['avg_today_change_pct']:+.2f}%。"
    )
    weak_text = (
        f"{weakest['theme']} 的完整日线强度最弱，平均全市场评分 {weakest['avg_score']:.3f}，"
        f"盘中涨跌 {weakest['avg_today_change_pct']:+.2f}%。"
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


def _build_rebalance_plan(review: pd.DataFrame) -> list[ProposedOrder]:
    weak = review[
        (review["current_price"] < review["stage_stop_price"])
        & (review["score_full_market"] < 0.72)
        & (review["today_change_pct"] <= 0.5)
    ].sort_values(["market_weight", "score_full_market"], ascending=[False, True])

    if weak.empty:
        return []

    orders: list[ProposedOrder] = []
    for row in weak.head(1).itertuples():
        lot_size = 100
        shares = int(int(row.shares_before) * 0.2 // lot_size) * lot_size
        if shares <= 0:
            continue
        trade_value = round(shares * float(row.current_price), 2)
        realized = round(shares * (float(row.current_price) - float(row.buy_price)), 2)
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
                    "且完整日线已明显落后于组合主线，执行温和减仓。"
                ),
            )
        )
    return orders


def _load_local_candidate_symbols() -> list[str]:
    if not CANDIDATE_POOL_PATH.exists():
        return []
    try:
        payload = json.loads(CANDIDATE_POOL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    values = payload.get("full_a")
    if not isinstance(values, list):
        return []
    symbols: list[str] = []
    for item in values:
        symbol = str(item or "").strip().upper()
        if symbol:
            symbols.append(symbol)
    return symbols


def _theme_label_for_symbol(symbol: str, category: str) -> str:
    for theme, symbols in THEME_BASKETS.items():
        if symbol in symbols:
            return theme
    return CATEGORY_THEME_LABELS.get(category, "正式筛选候选")


def _evidence_quality_label(
    *,
    completeness_passed: bool,
    decision_data_sufficient: bool,
    analysis_trade_date: str,
    strict_trade_date: str,
    source: str,
) -> str:
    if completeness_passed:
        return "高"
    if decision_data_sufficient:
        return "中"
    if source == "latest_formal_screening" and analysis_trade_date and analysis_trade_date == strict_trade_date:
        return "中"
    return "中等偏弱"


def _build_candidate_pool(
    *,
    full_metrics: pd.DataFrame,
    held_symbols: list[str],
    completeness_passed: bool,
    decision_data_sufficient: bool,
    analysis_trade_date: str,
    strict_trade_date: str,
) -> pd.DataFrame:
    if full_metrics.empty:
        return pd.DataFrame()

    candidate_symbols = _load_local_candidate_symbols()
    candidate_priority = {symbol: idx for idx, symbol in enumerate(candidate_symbols)}
    held_set = {str(symbol).strip().upper() for symbol in held_symbols}
    metrics = full_metrics.copy()
    metrics["symbol"] = metrics["symbol"].astype(str).str.upper()
    metrics = metrics[~metrics["symbol"].isin(held_set)].copy()
    if metrics.empty:
        return metrics

    metrics["candidate_source"] = metrics["symbol"].map(
        lambda symbol: "latest_formal_screening" if symbol in candidate_priority else "full_market_strength"
    )
    metrics["candidate_priority"] = metrics["symbol"].map(lambda symbol: candidate_priority.get(symbol, 999999))
    metrics["theme_label"] = metrics.apply(
        lambda row: _theme_label_for_symbol(str(row["symbol"]), str(row.get("category", ""))),
        axis=1,
    )
    metrics["evidence_quality"] = metrics["candidate_source"].map(
        lambda source: _evidence_quality_label(
            completeness_passed=completeness_passed,
            decision_data_sufficient=decision_data_sufficient,
            analysis_trade_date=analysis_trade_date,
            strict_trade_date=strict_trade_date,
            source=str(source),
        )
    )
    metrics = metrics.sort_values(
        by=["candidate_priority", "score_full_market", "ret20", "ret60", "symbol"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    metrics["candidate_rank"] = range(1, len(metrics) + 1)
    return metrics.head(12).reset_index(drop=True)


def _build_switch_plan(
    *,
    holdings_review: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    completeness_passed: bool,
    decision_data_sufficient: bool,
) -> pd.DataFrame:
    if holdings_review.empty or candidate_pool.empty:
        return pd.DataFrame()

    weak_holdings = holdings_review.sort_values(
        by=["rank_full_market", "score_full_market", "today_change_pct"],
        ascending=[False, True, True],
    ).head(3)
    best_candidates = candidate_pool.sort_values(
        by=["candidate_priority", "score_full_market", "ret20", "symbol"],
        ascending=[True, False, False, True],
    ).head(3)

    rows: list[dict[str, Any]] = []
    for weak_row, candidate_row in zip(weak_holdings.itertuples(), best_candidates.itertuples()):
        score_gap = round(float(candidate_row.score_full_market) - float(weak_row.score_full_market), 6)
        ret20_gap = round(float(candidate_row.ret20) - float(weak_row.ret20), 6)
        superior = (
            score_gap >= 0.12
            and int(candidate_row.rank_full_market) + 120 <= int(weak_row.rank_full_market)
            and ret20_gap >= 0.08
        )
        actionable = superior and (completeness_passed or decision_data_sufficient)
        rows.append(
            {
                "sell_symbol": weak_row.symbol,
                "sell_name": weak_row.name,
                "sell_role": weak_row.position_role,
                "sell_rank_full_market": int(weak_row.rank_full_market),
                "sell_score_full_market": round(float(weak_row.score_full_market), 6),
                "buy_symbol": candidate_row.symbol,
                "buy_name": candidate_row.name,
                "buy_theme": candidate_row.theme_label,
                "buy_rank_full_market": int(candidate_row.rank_full_market),
                "buy_score_full_market": round(float(candidate_row.score_full_market), 6),
                "score_gap": score_gap,
                "ret20_gap": ret20_gap,
                "candidate_source": candidate_row.candidate_source,
                "evidence_quality": candidate_row.evidence_quality,
                "priority": "high" if superior else "watch",
                "action": "switch_now" if actionable else ("prepare_switch" if superior else "watch_only"),
                "switch_ratio_hint": "20%" if actionable else "先观察，不执行",
                "trigger_threshold": (
                    "候选继续留在本地强度前120，且现持仓未收复阶段止损位"
                    if not actionable
                    else "按20%试探性换仓，若候选继续强于卖出对象再递增"
                ),
                "no_switch_condition": (
                    "当前前日线+实时行情口径仍未满足决策数据要求，先不把结构优势直接转成实单"
                    if superior and not (completeness_passed or decision_data_sufficient)
                    else "若现持仓重新回到前250名且候选优势收敛，则继续持有原仓"
                ),
            }
        )
    return pd.DataFrame(rows)


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
    if invested > 0:
        updated["market_weight"] = (updated["current_value"] / invested).round(6)
    else:
        updated["market_weight"] = 0.0
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
