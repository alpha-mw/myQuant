"""
统一的 CN/US 市场样本分析、批量分析与组合级报告。
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.market.config import get_market_settings, normalize_categories
from quant_investor.pipeline import QuantInvestorV8

_STOCK_NAME_CACHE: dict[str, dict[str, str]] = {"CN": {}, "US": {}}


def load_stock_names(market: str, refresh: bool = False) -> dict[str, str]:
    settings = get_market_settings(market)
    cache = _STOCK_NAME_CACHE[settings.market]
    if cache and not refresh:
        return cache

    cache_path = Path(settings.name_cache_file)
    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as file:
            _STOCK_NAME_CACHE[settings.market] = json.load(file)
        return _STOCK_NAME_CACHE[settings.market]

    if settings.market == "US":
        return cache

    try:
        import tushare as ts

        pro = create_tushare_pro(ts, config.TUSHARE_TOKEN, config.TUSHARE_URL)
        if pro is None:
            return cache
        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name")
        if df is None or df.empty:
            return cache
        payload = dict(zip(df["ts_code"], df["name"]))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        _STOCK_NAME_CACHE[settings.market] = payload
        return payload
    except Exception:
        return cache


def load_cn_stock_names(refresh: bool = False) -> dict[str, str]:
    return load_stock_names("CN", refresh=refresh)


def load_us_stock_names(refresh: bool = False) -> dict[str, str]:
    return load_stock_names("US", refresh=refresh)


def get_stock_name(symbol: str, market: str = "CN") -> str:
    settings = get_market_settings(market)
    if not _STOCK_NAME_CACHE[settings.market]:
        load_stock_names(settings.market)
    fallback = "未知" if settings.market == "CN" else "N/A"
    return _STOCK_NAME_CACHE[settings.market].get(symbol, fallback)


def get_us_stock_name(symbol: str) -> str:
    return get_stock_name(symbol, market="US")


def category_name(category: str, market: str = "CN") -> str:
    settings = get_market_settings(market)
    return settings.category_labels.get(category, category)


def get_all_local_symbols(category: str, market: str = "CN", data_dir: str | None = None) -> list[str]:
    settings = get_market_settings(market)
    base_dir = Path(data_dir or settings.data_dir)
    category_dir = base_dir / category
    if not category_dir.exists():
        return []
    return sorted(path.stem for path in category_dir.glob("*.csv"))


def analyze_batch(
    symbols: list[str],
    category: str,
    batch_id: int,
    market: str = "CN",
    total_capital: float = 1_000_000,
    risk_level: str = "中等",
    verbose: bool = True,
) -> Optional[dict[str, Any]]:
    settings = get_market_settings(market)
    scoped_category_name = category_name(category, settings.market)

    print(f"\n{'=' * 80}")
    print(f"📊 分析 {scoped_category_name} - 批次 {batch_id}")
    print(f"{'=' * 80}")
    print(f"本批股票数: {len(symbols)}")
    print(f"前10只: {symbols[:10]}")

    try:
        analyzer = QuantInvestorV8(
            stock_pool=symbols,
            market=settings.market,
            total_capital=total_capital,
            risk_level=risk_level,
            enable_macro=True,
            enable_kronos=True,
            enable_intelligence=True,
            enable_llm_debate=True,
            verbose=verbose,
        )
        result = analyzer.run()

        recommendations = []
        for recommendation in result.final_strategy.trade_recommendations:
            payload = asdict(recommendation)
            payload["category"] = category
            payload["category_name"] = scoped_category_name
            recommendations.append(payload)

        analysis = {
            "market": settings.market,
            "category": category,
            "category_name": scoped_category_name,
            "batch_id": batch_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "stocks": symbols,
            "stock_count": len(symbols),
            "branches": {},
            "strategy": {
                "target_exposure": result.final_strategy.target_exposure,
                "style_bias": result.final_strategy.style_bias,
                "candidate_symbols": result.final_strategy.candidate_symbols,
                "position_limits": result.final_strategy.position_limits,
                "branch_consensus": result.final_strategy.branch_consensus,
                "risk_summary": result.final_strategy.risk_summary,
                "execution_notes": result.final_strategy.execution_notes,
                "research_mode": result.final_strategy.research_mode,
            },
            "recommendations": recommendations,
        }

        for name, branch in result.branch_results.items():
            analysis["branches"][name] = {
                "score": branch.score,
                "confidence": branch.confidence,
                "top_symbols": [
                    {"symbol": symbol, "score": score}
                    for symbol, score in sorted(
                        branch.symbol_scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:5]
                ],
            }

        print(f"✅ 批次 {batch_id} 分析完成")
        print(f"   目标仓位: {analysis['strategy']['target_exposure']:.0%}")
        print(f"   候选标的: {len(analysis['strategy']['candidate_symbols'])} 只")
        return analysis
    except Exception as exc:
        print(f"❌ 批次 {batch_id} 分析失败: {exc}")
        import traceback

        traceback.print_exc()
        return None


def save_batch_result(
    result: dict[str, Any],
    market: str = "CN",
    output_dir: str | None = None,
) -> str:
    settings = get_market_settings(market)
    target_dir = Path(output_dir or settings.analysis_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / (
        f"batch_{result['category']}_{int(result['batch_id']):03d}_{result['timestamp']}.json"
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"💾 批次结果已保存: {output_path}")
    return str(output_path)


def analyze_category_full(
    category: str,
    market: str = "CN",
    batch_size: Optional[int] = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    total_capital: float = 1_000_000,
    risk_level: str = "中等",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    settings = get_market_settings(market)
    scoped_batch_size = batch_size or settings.default_batch_size
    symbols = get_all_local_symbols(category, market=settings.market, data_dir=data_dir)
    total = len(symbols)

    print(f"\n{'=' * 80}")
    print(f"🚀 开始全量分析 {category_name(category, settings.market)}")
    print(f"{'=' * 80}")

    if total == 0:
        print(f"❌ 没有找到 {category} 的数据")
        return []

    print(f"总计 {total} 只股票需要分析")
    print(f"批次大小: {scoped_batch_size} 只")
    print(f"预计批次: {(total + scoped_batch_size - 1) // scoped_batch_size} 批")
    print(f"预计时间: {total * 2 / 60:.1f} 分钟")
    print(f"{'=' * 80}")

    all_results: list[dict[str, Any]] = []
    num_batches = (total + scoped_batch_size - 1) // scoped_batch_size
    for index in range(num_batches):
        start_idx = index * scoped_batch_size
        end_idx = min(start_idx + scoped_batch_size, total)
        batch_symbols = symbols[start_idx:end_idx]
        print(f"\n⏳ 进度: 批次 {index + 1}/{num_batches} ({start_idx + 1}-{end_idx}/{total})")
        result = analyze_batch(
            batch_symbols,
            category,
            index + 1,
            market=settings.market,
            total_capital=total_capital,
            risk_level=risk_level,
            verbose=verbose,
        )
        if result:
            all_results.append(result)
            save_batch_result(result, market=settings.market, output_dir=output_dir)
    return all_results


def _safe_average(values: list[float], default: float = 0.0) -> float:
    normalized = [float(value) for value in values if value is not None]
    return sum(normalized) / len(normalized) if normalized else default


def _normalize_with_cap(
    raw_scores: dict[str, float],
    total_target_exposure: float,
    max_single_weight: float,
) -> dict[str, float]:
    positive_scores = {symbol: score for symbol, score in raw_scores.items() if score > 0}
    if not positive_scores or total_target_exposure <= 0:
        return {}

    remaining = dict(positive_scores)
    weights = {symbol: 0.0 for symbol in positive_scores}
    remaining_exposure = total_target_exposure

    while remaining and remaining_exposure > 1e-8:
        total_score = sum(remaining.values())
        if total_score <= 0:
            break

        overflow_symbols = []
        for symbol, score in list(remaining.items()):
            proposed = remaining_exposure * score / total_score
            if proposed > max_single_weight + 1e-8:
                weights[symbol] = max_single_weight
                remaining_exposure -= max_single_weight
                overflow_symbols.append(symbol)

        if overflow_symbols:
            for symbol in overflow_symbols:
                remaining.pop(symbol, None)
            continue

        for symbol, score in remaining.items():
            weights[symbol] = remaining_exposure * score / total_score
        break

    return {symbol: weight for symbol, weight in weights.items() if weight > 0}


def _build_market_summary(all_results: dict[str, list[dict[str, Any]]], market: str = "CN") -> dict[str, Any]:
    settings = get_market_settings(market)
    summary: dict[str, Any] = {
        "market": settings.market,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": 0,
        "total_batches": 0,
        "categories": {},
    }

    for category, results in all_results.items():
        if not results:
            continue
        category_stocks = sum(item.get("stock_count", 0) for item in results)
        summary["total_stocks"] += category_stocks
        summary["total_batches"] += len(results)

        branch_scores: dict[str, list[float]] = {}
        candidate_count = 0
        for item in results:
            candidate_count += len(item.get("strategy", {}).get("candidate_symbols", []))
            for name, branch in item.get("branches", {}).items():
                branch_scores.setdefault(name, []).append(float(branch.get("score", 0.0)))

        summary["categories"][category] = {
            "category_name": category_name(category, settings.market),
            "batch_count": len(results),
            "stock_count": category_stocks,
            "candidate_count": candidate_count,
            "avg_target_exposure": _safe_average(
                [item.get("strategy", {}).get("target_exposure", 0.0) for item in results]
            ),
            "avg_branch_scores": {
                name: _safe_average(scores) for name, scores in branch_scores.items()
            },
        }
    return summary


def build_full_market_trade_plan(
    all_results: dict[str, list[dict[str, Any]]],
    market: str = "CN",
    total_capital: float = 1_000_000,
    top_k: int = 12,
) -> dict[str, Any]:
    settings = get_market_settings(market)
    summary = _build_market_summary(all_results, market=settings.market)
    collected: list[dict[str, Any]] = []

    for category, batches in all_results.items():
        for batch in batches:
            batch_target_exposure = float(batch.get("strategy", {}).get("target_exposure", 0.0))
            batch_style_bias = batch.get("strategy", {}).get("style_bias", "均衡")
            batch_risk_summary = batch.get("strategy", {}).get("risk_summary", {})
            for recommendation in batch.get("recommendations", []):
                if recommendation.get("action") != "buy":
                    continue
                if recommendation.get("data_source_status") != "real":
                    continue
                payload = dict(recommendation)
                payload["category"] = category
                payload["category_name"] = category_name(category, settings.market)
                payload["batch_target_exposure"] = batch_target_exposure
                payload["style_bias"] = batch_style_bias
                payload["risk_level"] = batch_risk_summary.get("risk_level", "normal")
                payload["rank_score"] = (
                    max(float(payload.get("suggested_weight", 0.0)), 0.001)
                    * (1 + max(float(payload.get("consensus_score", 0.0)), 0.0))
                    * (1 + max(float(payload.get("model_expected_return", 0.0)), 0.0))
                    * (0.8 + float(payload.get("confidence", 0.0)))
                    * (1 + float(payload.get("branch_positive_count", 0)) / 5)
                )
                collected.append(payload)

    deduped: dict[str, dict[str, Any]] = {}
    for item in sorted(collected, key=lambda entry: entry["rank_score"], reverse=True):
        deduped.setdefault(item["symbol"], item)

    ranked = list(deduped.values())[:top_k]
    if not ranked:
        return {
            "market_summary": summary,
            "portfolio_plan": {
                "total_capital": total_capital,
                "target_exposure": 0.0,
                "planned_investment": 0.0,
                "cash_reserve": total_capital,
                "selected_count": 0,
                "style_bias": "防御",
                "max_single_weight": 0.0,
                "category_exposure": {},
                "execution_notes": ["当前没有满足真实数据与买入条件的候选标的。"],
            },
            "recommendations": [],
        }

    weighted_exposure_values = []
    for batches in all_results.values():
        for batch in batches:
            stock_count = max(int(batch.get("stock_count", 0)), 1)
            weighted_exposure_values.extend(
                [float(batch.get("strategy", {}).get("target_exposure", 0.0))] * stock_count
            )

    target_exposure = min(max(_safe_average(weighted_exposure_values, default=0.35), 0.15), 0.80)
    max_single_weight = min(0.12, max(0.05, target_exposure / max(len(ranked), 1) * 2.2))

    active = ranked
    for _ in range(3):
        weight_map = _normalize_with_cap(
            {item["symbol"]: float(item["rank_score"]) for item in active},
            total_target_exposure=target_exposure,
            max_single_weight=max_single_weight,
        )
        filtered_active = []
        for item in active:
            weight = weight_map.get(item["symbol"], 0.0)
            entry_price = float(item.get("recommended_entry_price") or item.get("current_price") or 0.0)
            lot_size = int(item.get("lot_size", settings.lot_size))
            if entry_price <= 0:
                continue
            minimum_ticket = entry_price * lot_size
            if total_capital * weight + 1e-8 < minimum_ticket:
                continue
            filtered_active.append(item)
        if len(filtered_active) == len(active):
            break
        active = filtered_active

    weight_map = _normalize_with_cap(
        {item["symbol"]: float(item["rank_score"]) for item in active},
        total_target_exposure=target_exposure,
        max_single_weight=max_single_weight,
    )

    final_recommendations = []
    category_exposure: dict[str, float] = {}
    style_counter = Counter()
    planned_investment = 0.0

    for rank, item in enumerate(active, start=1):
        weight = weight_map.get(item["symbol"], 0.0)
        entry_price = float(item.get("recommended_entry_price") or item.get("current_price") or 0.0)
        lot_size = int(item.get("lot_size", settings.lot_size))
        shares = int((total_capital * weight) // max(entry_price, 0.01) // lot_size) * lot_size
        amount = shares * entry_price
        actual_weight = amount / total_capital if total_capital > 0 else 0.0
        if shares <= 0 or amount <= 0:
            continue

        final_item = dict(item)
        final_item["rank"] = rank
        final_item["portfolio_weight"] = round(actual_weight, 4)
        final_item["portfolio_amount"] = round(amount, 2)
        final_item["portfolio_shares"] = shares
        final_item["cash_buffer"] = round(total_capital * weight - amount, 2)
        final_recommendations.append(final_item)
        planned_investment += amount
        category_exposure[item["category"]] = category_exposure.get(item["category"], 0.0) + actual_weight
        style_counter[item.get("style_bias", "均衡")] += 1

    cash_reserve = max(total_capital - planned_investment, 0.0)
    portfolio_style_bias = style_counter.most_common(1)[0][0] if style_counter else "均衡"
    execution_notes = [
        f"全市场共扫描 {summary['total_stocks']} 只股票，最终入选 {len(final_recommendations)} 只。",
        f"组合计划投入约 {settings.currency_symbol}{planned_investment:,.0f}，保留现金约 {settings.currency_symbol}{cash_reserve:,.0f}。",
        f"单票上限 {max_single_weight:.1%}，优先采用分批建仓与纪律止损。",
    ]

    return {
        "market_summary": summary,
        "portfolio_plan": {
            "total_capital": total_capital,
            "target_exposure": round(sum(item["portfolio_weight"] for item in final_recommendations), 4),
            "planned_investment": round(planned_investment, 2),
            "cash_reserve": round(cash_reserve, 2),
            "selected_count": len(final_recommendations),
            "style_bias": portfolio_style_bias,
            "max_single_weight": round(max_single_weight, 4),
            "category_exposure": {key: round(value, 4) for key, value in category_exposure.items()},
            "execution_notes": execution_notes,
        },
        "recommendations": final_recommendations,
    }


def save_candidate_index(
    all_results: dict[str, list[dict[str, Any]]],
    market: str = "CN",
    output_dir: str | None = None,
) -> str:
    settings = get_market_settings(market)
    target_dir = Path(output_dir or settings.analysis_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[str]] = {}
    for category, batches in all_results.items():
        items: list[dict[str, Any]] = []
        for batch in batches:
            items.extend(batch.get("recommendations", []))
        ranked_symbols = [
            item["symbol"]
            for item in sorted(
                items,
                key=lambda rec: (
                    float(rec.get("consensus_score", 0.0)),
                    float(rec.get("suggested_weight", 0.0)),
                ),
                reverse=True,
            )
            if item.get("data_source_status") == "real"
        ]
        payload[category] = list(dict.fromkeys(ranked_symbols))

    output_file = target_dir / "all_candidates.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return str(output_file)


def generate_full_report(
    all_results: dict[str, list[dict[str, Any]]],
    market: str = "CN",
    output_dir: str | None = None,
    total_capital: float = 1_000_000,
    top_k: int = 12,
) -> dict[str, str]:
    settings = get_market_settings(market)
    target_dir = Path(output_dir or settings.analysis_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 80}")
    print(f"📊 生成{settings.market_name}全市场综合分析报告")
    print(f"{'=' * 80}")

    load_stock_names(settings.market)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = build_full_market_trade_plan(
        all_results,
        market=settings.market,
        total_capital=total_capital,
        top_k=top_k,
    )
    summary = plan["market_summary"]
    portfolio_plan = plan["portfolio_plan"]
    recommendations = plan["recommendations"]

    report_lines = [
        f"# {settings.report_flag} {settings.market_name}全市场组合级交易建议报告\n",
        f"**生成时间**: {summary['generated_at']}\n",
        "**分析架构**: Quant-Investor V8.0 五路并行研究\n",
        f"**分析覆盖**: {summary['total_stocks']} 只股票，{summary['total_batches']} 个批次\n",
        "\n---\n",
        "\n## 📊 市场全量扫描摘要\n",
    ]
    for category, info in summary["categories"].items():
        report_lines.append(f"\n### {info['category_name']}\n")
        report_lines.append(f"- **分析批次**: {info['batch_count']} 批\n")
        report_lines.append(f"- **股票总数**: {info['stock_count']} 只\n")
        report_lines.append(f"- **候选股数**: {info['candidate_count']} 只\n")
        report_lines.append(f"- **平均目标仓位**: {info['avg_target_exposure']:.1%}\n")
        report_lines.append("- **平均分支得分**:\n")
        for branch_name in ["kline", "quant", "llm_debate", "intelligence", "macro"]:
            score = info["avg_branch_scores"].get(branch_name)
            if score is not None:
                report_lines.append(f"  - {branch_name}: {score:+.3f}\n")

    report_lines.extend(
        [
            "\n## 💰 组合级执行计划\n",
            f"- **总资金**: {settings.currency_symbol}{portfolio_plan['total_capital']:,.0f}\n",
            f"- **计划总仓位**: {portfolio_plan['target_exposure']:.1%}\n",
            f"- **计划投入资金**: {settings.currency_symbol}{portfolio_plan['planned_investment']:,.0f}\n",
            f"- **预留现金**: {settings.currency_symbol}{portfolio_plan['cash_reserve']:,.2f}\n",
            f"- **组合风格**: {portfolio_plan['style_bias']}\n",
            f"- **单票上限**: {portfolio_plan['max_single_weight']:.1%}\n",
            f"- **类别暴露**: "
            + (
                " / ".join(
                    f"{category_name(category, settings.market)} {weight:.1%}"
                    for category, weight in portfolio_plan["category_exposure"].items()
                )
                if portfolio_plan["category_exposure"]
                else "暂无"
            )
            + "\n",
        ]
    )

    if recommendations:
        report_lines.extend(
            [
                "\n## 🎯 最终入选标的\n",
                "| 排名 | 代码 | 名称 | 类别 | 现价 | 推荐买入价 | 目标卖出价 | 止损价 | 推荐仓位 | 金额 | 预期空间 | 五路支持 |\n",
                "|:---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
            ]
        )
        for item in recommendations:
            stock_name = get_stock_name(item["symbol"], market=settings.market)
            current_price = float(item.get("current_price", 0))
            entry_low = float(item.get("entry_price_range", {}).get("low", current_price * 0.99))
            entry_high = float(item.get("entry_price_range", {}).get("high", current_price * 1.01))
            display_entry_price = float(
                item.get("recommended_entry_price") or (entry_low + entry_high) / 2 or current_price
            )
            report_lines.append(
                f"| {item['rank']} | {item['symbol']} | {stock_name} | {item['category_name']} | "
                f"{settings.currency_symbol}{current_price:.2f} | {settings.currency_symbol}{display_entry_price:.2f} | "
                f"{settings.currency_symbol}{item['target_price']:.2f} | {settings.currency_symbol}{item['stop_loss_price']:.2f} | "
                f"{item['portfolio_weight']:.1%} | {settings.currency_symbol}{item['portfolio_amount']:,.0f} | "
                f"{float(item['expected_upside']):.1%} | {item['branch_positive_count']}/5 |\n"
            )

        report_lines.append("\n## 🧭 个股执行说明\n")
        for item in recommendations[:12]:
            stock_name = get_stock_name(item["symbol"], market=settings.market)
            current_price = float(item.get("current_price", 0))
            entry_low = float(item.get("entry_price_range", {}).get("low", current_price * 0.99))
            entry_high = float(item.get("entry_price_range", {}).get("high", current_price * 1.01))
            display_entry_price = float(
                item.get("recommended_entry_price") or (entry_low + entry_high) / 2 or current_price
            )
            report_lines.append(f"\n### {item['rank']}. {item['symbol']} {stock_name} ({item['category_name']})\n")
            report_lines.append("**📊 交易参数**\n")
            report_lines.append(f"- **当前价格**: {settings.currency_symbol}{current_price:.2f}\n")
            report_lines.append(
                f"- **推荐买入价**: {settings.currency_symbol}{display_entry_price:.2f} "
                f"(可成交区间: {settings.currency_symbol}{entry_low:.2f} - {settings.currency_symbol}{entry_high:.2f})\n"
            )
            report_lines.append(
                f"- **目标卖出价**: {settings.currency_symbol}{item['target_price']:.2f} "
                f"(预期收益: {float(item['expected_upside']):.1%})\n"
            )
            report_lines.append(
                f"- **止损价格**: {settings.currency_symbol}{item['stop_loss_price']:.2f} "
                f"(最大亏损: {(item['stop_loss_price'] / max(display_entry_price, 0.01) - 1) * 100:.1f}%)\n"
            )
            report_lines.append(
                f"- **推荐仓位**: {item['portfolio_weight']:.1%} "
                f"({settings.currency_symbol}{item['portfolio_amount']:,.0f} / {item['portfolio_shares']:,}股)\n"
            )
            report_lines.append("\n**🔬 五路研究分析**\n")
            report_lines.append(f"- **综合评分**: {float(item['consensus_score']):+.2f}\n")
            report_lines.append(f"- **支持分支**: {item['branch_positive_count']}/5\n")
            report_lines.append(f"- **置信度**: {float(item['confidence']):.0%}\n")
            if item.get("risk_flags") or item.get("position_management"):
                report_lines.append("\n**⚠️ 风险提示与仓位管理**\n")
                for flag in item.get("risk_flags", [])[:3]:
                    report_lines.append(f"- {flag}\n")
                for note in item.get("position_management", [])[:2]:
                    report_lines.append(f"- {note}\n")
    else:
        report_lines.append("\n## 🎯 最终入选标的\n\n当前没有满足条件的最终买入标的，建议继续以现金观望。\n")

    report_lines.append("\n## ✅ 执行提醒\n")
    for note in portfolio_plan["execution_notes"]:
        report_lines.append(f"- {note}\n")

    summary_lines = [
        f"# {settings.report_flag} {settings.market_name}全市场分析摘要\n",
        f"**生成时间**: {summary['generated_at']}\n",
        f"**分析覆盖**: {summary['total_stocks']} 只股票，{summary['total_batches']} 个批次\n",
        f"**计划总仓位**: {portfolio_plan['target_exposure']:.1%}\n",
        f"**计划投入资金**: {settings.currency_symbol}{portfolio_plan['planned_investment']:,.0f}\n",
        f"**预留现金**: {settings.currency_symbol}{portfolio_plan['cash_reserve']:,.0f}\n",
        f"**最终入选标的数**: {portfolio_plan['selected_count']} 只\n",
        "\n## 类别摘要\n",
    ]
    for category, info in summary["categories"].items():
        summary_lines.append(
            f"- {info['category_name']}: {info['stock_count']} 只，"
            f"候选 {info['candidate_count']} 只，平均目标仓位 {info['avg_target_exposure']:.1%}\n"
        )
    summary_lines.append("\n## 执行提醒\n")
    for note in portfolio_plan["execution_notes"]:
        summary_lines.append(f"- {note}\n")

    summary_file = target_dir / f"{settings.market}_Full_Report_{timestamp}.md"
    data_file = target_dir / f"{settings.market}_Trade_Data_{timestamp}.json"
    report_file = target_dir / f"{settings.market}_Trade_Report_{timestamp}.md"
    with open(summary_file, "w", encoding="utf-8") as file:
        file.writelines(summary_lines)
    with open(report_file, "w", encoding="utf-8") as file:
        file.writelines(report_lines)
    with open(data_file, "w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2, ensure_ascii=False)

    candidate_file = save_candidate_index(all_results, market=settings.market, output_dir=str(target_dir))
    return {
        "summary_report": str(summary_file),
        "trade_report": str(report_file),
        "trade_data": str(data_file),
        "candidate_index": candidate_file,
    }


def run_market_analysis(
    market: str,
    mode: str = "batch",
    categories: list[str] | None = None,
    batch_size: int | None = None,
    total_capital: float = 1_000_000,
    top_k: int = 12,
    verbose: bool = True,
) -> dict[str, Any]:
    settings = get_market_settings(market)
    selected_categories = normalize_categories(settings.market, categories)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for category in selected_categories:
        if mode == "sample":
            symbols = get_all_local_symbols(category, market=settings.market)[: settings.default_batch_size]
            result = analyze_batch(
                symbols,
                category,
                1,
                market=settings.market,
                total_capital=total_capital,
                verbose=verbose,
            )
            all_results[category] = [result] if result else []
        else:
            all_results[category] = analyze_category_full(
                category,
                market=settings.market,
                batch_size=batch_size,
                total_capital=total_capital,
                verbose=verbose,
            )
    report_paths = generate_full_report(
        all_results,
        market=settings.market,
        total_capital=total_capital,
        top_k=top_k,
    )
    return {"results": all_results, "reports": report_paths}
