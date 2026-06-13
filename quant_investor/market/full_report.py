"""Full-market report rendering for market analysis artifacts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.market.config import get_market_settings
from quant_investor.market.full_report_helpers import (
    BRANCH_LABELS as BRANCH_LABELS,
    BRANCH_SUPPORT_DENOMINATOR,
    _STOCK_NAME_CACHE as _STOCK_NAME_CACHE,
    _branch_label as _branch_label,
    _build_analysis_meta,
    _build_market_summary,
    _canonical_branch_map as _canonical_branch_map,
    _confidence_label,
    _dedupe_text,
    _default_branch_conclusion as _default_branch_conclusion,
    _derive_stock_conclusion as _derive_stock_conclusion,
    _derive_stock_drag_drivers as _derive_stock_drag_drivers,
    _derive_stock_support_drivers as _derive_stock_support_drivers,
    _is_unknown_stock_name,
    _normalize_with_cap,
    _safe_average,
    _sanitize_text,
    _to_mapping,
    category_name,
    get_stock_name,
    load_stock_names,
)
from quant_investor.market.full_report_sections import (
    ActionConsistencyGuard,
    ConclusionRenderer,
    DiagnosticsBucketizer,
    ExecutiveSummaryBuilder,
    _aggregate_branch_summary,
)

# Keep legacy private aliases as module attributes for analyze.py and
# legacy_batch_analysis.py while the implementation lives in helper modules.
_LEGACY_HELPER_REEXPORTS = (
    _STOCK_NAME_CACHE,
    _branch_label,
    _canonical_branch_map,
    _default_branch_conclusion,
    _derive_stock_conclusion,
    _derive_stock_drag_drivers,
    _derive_stock_support_drivers,
)


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
            batch_target_exposure = float(
                batch.get("strategy", {}).get("target_exposure", 0.0)
            )
            batch_style_bias = batch.get("strategy", {}).get(
                "style_bias", "均衡"
            )
            batch_risk_summary = batch.get("strategy", {}).get(
                "risk_summary", {}
            )
            for recommendation in batch.get("recommendations", []):
                if recommendation.get("action") != "buy":
                    continue
                if recommendation.get("data_source_status") != "real":
                    continue
                payload = dict(recommendation)
                payload["category"] = category
                payload["category_name"] = category_name(
                    category, settings.market
                )
                raw_company_name = str(
                    payload.get("company_name", "") or ""
                ).strip()
                company_name = (
                    get_stock_name(
                        payload.get("symbol", ""), market=settings.market
                    )
                    if _is_unknown_stock_name(raw_company_name)
                    else raw_company_name
                )
                payload["company_name"] = company_name
                raw_name = str(payload.get("name", "") or "").strip()
                payload["name"] = (
                    company_name
                    if _is_unknown_stock_name(raw_name)
                    else raw_name
                )
                payload["batch_target_exposure"] = batch_target_exposure
                payload["style_bias"] = batch_style_bias
                payload["risk_level"] = batch_risk_summary.get(
                    "risk_level", "normal"
                )
                payload["rank_score"] = (
                    max(float(payload.get("suggested_weight", 0.0)), 0.001)
                    * (
                        1
                        + max(float(payload.get("consensus_score", 0.0)), 0.0)
                    )
                    * (
                        1
                        + max(
                            float(payload.get("model_expected_return", 0.0)),
                            0.0,
                        )
                    )
                    * (0.8 + float(payload.get("confidence", 0.0)))
                    * (
                        1
                        + float(payload.get("branch_positive_count", 0))
                        / BRANCH_SUPPORT_DENOMINATOR
                    )
                )
                collected.append(payload)

    deduped: dict[str, dict[str, Any]] = {}
    for item in sorted(
        collected, key=lambda entry: entry["rank_score"], reverse=True
    ):
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
                "execution_notes": [
                    "当前没有满足真实数据与买入条件的候选标的。"
                ],
            },
            "recommendations": [],
        }

    weighted_exposure_values = []
    for batches in all_results.values():
        for batch in batches:
            stock_count = max(int(batch.get("stock_count", 0)), 1)
            weighted_exposure_values.extend(
                [float(batch.get("strategy", {}).get("target_exposure", 0.0))]
                * stock_count
            )

    target_exposure = min(
        max(_safe_average(weighted_exposure_values, default=0.35), 0.15), 0.80
    )
    max_single_weight = min(
        0.12, max(0.05, target_exposure / max(len(ranked), 1) * 2.2)
    )

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
            entry_price = float(
                item.get("recommended_entry_price")
                or item.get("current_price")
                or 0.0
            )
            lot_size = int(
                item.get("lot_size", getattr(settings, "lot_size", 100))
            )
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
        entry_price = float(
            item.get("recommended_entry_price")
            or item.get("current_price")
            or 0.0
        )
        lot_size = int(
            item.get("lot_size", getattr(settings, "lot_size", 100))
        )
        shares = (
            int((total_capital * weight) // max(entry_price, 0.01) // lot_size)
            * lot_size
        )
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
        final_recommendations.append(ActionConsistencyGuard.apply(final_item))
        planned_investment += amount
        category_exposure[item["category"]] = (
            category_exposure.get(item["category"], 0.0) + actual_weight
        )
        style_counter[item.get("style_bias", "均衡")] += 1

    cash_reserve = max(total_capital - planned_investment, 0.0)
    portfolio_style_bias = (
        style_counter.most_common(1)[0][0] if style_counter else "均衡"
    )
    reliability = _safe_average(
        [float(item.get("confidence", 0.0)) for item in final_recommendations],
        default=0.0,
    )
    execution_notes = [
        (
            f"全市场共扫描 {summary['total_stocks']} 只股票，"
            f"最终入选 {len(final_recommendations)} 只。"
        ),
        (
            f"组合计划投入约 {settings.currency_symbol}"
            f"{planned_investment:,.0f}，保留现金约 "
            f"{settings.currency_symbol}{cash_reserve:,.0f}。"
        ),
        f"单票上限 {max_single_weight:.1%}，优先采用分批建仓与纪律止损。",
    ]

    return {
        "market_summary": summary,
        "portfolio_plan": {
            "total_capital": total_capital,
            "target_exposure": round(
                sum(
                    item["portfolio_weight"] for item in final_recommendations
                ),
                4,
            ),
            "planned_investment": round(planned_investment, 2),
            "cash_reserve": round(cash_reserve, 2),
            "selected_count": len(final_recommendations),
            "style_bias": portfolio_style_bias,
            "max_single_weight": round(max_single_weight, 4),
            "category_exposure": {
                key: round(value, 4)
                for key, value in category_exposure.items()
            },
            "execution_notes": execution_notes,
            "reliability": round(reliability, 4),
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


def _build_full_market_report_bundle(
    all_results: dict[str, list[dict[str, Any]]],
    *,
    market: str,
    total_capital: float,
    top_k: int,
) -> tuple[dict[str, Any], Any | None]:
    plan = build_full_market_trade_plan(
        all_results,
        market=market,
        total_capital=total_capital,
        top_k=top_k,
    )
    return plan, None


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
    plan, report_bundle = _build_full_market_report_bundle(
        all_results,
        market=settings.market,
        total_capital=total_capital,
        top_k=top_k,
    )
    summary = plan["market_summary"]
    portfolio_plan = _to_mapping(
        getattr(report_bundle, "portfolio_plan", None)
    ) or _to_mapping(plan.get("portfolio_plan"))
    target_weights = dict(portfolio_plan.get("target_weights", {}) or {})
    position_limits = dict(portfolio_plan.get("position_limits", {}) or {})
    category_exposure = dict(portfolio_plan.get("category_exposure", {}) or {})
    execution_notes = list(portfolio_plan.get("execution_notes", []) or [])
    if not portfolio_plan:
        portfolio_plan = {}
    portfolio_plan.setdefault("target_weights", target_weights)
    portfolio_plan.setdefault("position_limits", position_limits)
    portfolio_plan.setdefault("category_exposure", category_exposure)
    portfolio_plan.setdefault("execution_notes", execution_notes)
    portfolio_plan.setdefault(
        "target_exposure",
        float(
            portfolio_plan.get(
                "target_exposure",
                sum(target_weights.values()) if target_weights else 0.0,
            )
        ),
    )
    portfolio_plan.setdefault(
        "planned_investment",
        float(portfolio_plan["target_exposure"]) * float(total_capital),
    )
    portfolio_plan.setdefault(
        "cash_reserve",
        float(total_capital)
        * float(
            portfolio_plan.get(
                "cash_ratio", 1.0 - float(portfolio_plan["target_exposure"])
            )
        ),
    )
    portfolio_plan.setdefault(
        "style_bias", portfolio_plan.get("style_bias", "均衡")
    )
    portfolio_plan.setdefault(
        "max_single_weight",
        max(
            position_limits.values(),
            default=max(target_weights.values(), default=0.0),
        ),
    )
    portfolio_plan.setdefault("selected_count", len(target_weights))
    raw_recommendations = list(plan.get("recommendations", []) or [])
    recommendations = [
        ActionConsistencyGuard.apply(item) for item in raw_recommendations
    ]
    plan["recommendations"] = recommendations
    plan["portfolio_plan"] = portfolio_plan
    branch_summary = _aggregate_branch_summary(all_results)
    diagnostics = DiagnosticsBucketizer(all_results, branch_summary).bucket()
    executive_summary = ExecutiveSummaryBuilder(
        portfolio_plan, branch_summary
    ).build()
    analysis_meta = _build_analysis_meta(all_results)

    report_lines = [
        (
            f"# {settings.report_flag} "
            f"{settings.market_name}全市场组合级交易建议报告\n"
        ),
        f"**生成时间**: {summary['generated_at']}\n",
        "**分析架构**: Quant-Investor V13 四分支研究契约\n",
        (
            f"**分析覆盖**: {summary['total_stocks']} 只股票，"
            f"{summary['total_batches']} 个批次\n"
        ),
        f"**分析 universe**: {analysis_meta.get('universe', 'full_a')}\n",
        "\n## 三句话执行摘要\n",
    ]
    data_snapshot_summary = str(
        (analysis_meta.get("data_snapshot", {}) or {}).get("summary_text", "")
    ).strip()
    if data_snapshot_summary:
        report_lines.insert(5, f"**本地数据快照**: {data_snapshot_summary}\n")
    for line in executive_summary:
        report_lines.append(f"- {line}\n")

    cash_reserve_text = (
        f"{settings.currency_symbol}{portfolio_plan['cash_reserve']:,.0f}"
    )
    reliability_label = _confidence_label(
        float(portfolio_plan.get("reliability", 0.0))
    )
    report_lines.extend(
        [
            "\n## 为什么当前总仓位是这个水平\n",
            (
                f"- 当前计划总仓位为 {portfolio_plan['target_exposure']:.1%}，"
                f"计划投入 {settings.currency_symbol}"
                f"{portfolio_plan['planned_investment']:,.0f}，预留现金 "
                f"{cash_reserve_text}。\n"
            ),
            (
                f"- 组合风格偏 {portfolio_plan['style_bias']}，"
                f"单票上限控制在 {portfolio_plan['max_single_weight']:.1%}，"
                f"本轮最终纳入 {portfolio_plan['selected_count']} 只标的。\n"
            ),
            "- 类别暴露为 "
            + (
                " / ".join(
                    f"{category_name(category, settings.market)} {weight:.1%}"
                    for category, weight in portfolio_plan[
                        "category_exposure"
                    ].items()
                )
                if portfolio_plan["category_exposure"]
                else "暂无"
            )
            + "。\n",
        ]
    )

    report_lines.extend(
        [
            "\n## 数据覆盖与可信度摘要\n",
            (
                f"- 本次汇总 {summary['total_batches']}/"
                f"{summary['total_batches']} 批次，覆盖 "
                f"{summary['total_stocks']}/{summary['total_stocks']} 标的。\n"
            ),
            (
                f"- 组合层平均可信度为 {reliability_label}。\n"
            ),
        ]
    )
    for note in diagnostics["coverage_notes"][:5]:
        report_lines.append(f"- {_sanitize_text(note)}\n")
    if diagnostics["investment_risks"]:
        report_lines.append(
            f"- 需要前置注意的投资风险: {'；'.join(diagnostics['investment_risks'][:3])}\n"
        )
    if analysis_meta.get("model_role_metadata"):
        report_lines.extend(["\n## 模型角色与执行轨迹\n"])
        render_run_context = getattr(
            ConclusionRenderer, "render_run_context", None
        )
        if callable(render_run_context):
            report_lines.extend(
                render_run_context(
                    analysis_meta.get("model_role_metadata"),
                    analysis_meta.get("execution_trace"),
                    analysis_meta.get("what_if_plan"),
                )
            )
        else:
            report_lines.extend(
                ConclusionRenderer.render_model_role_metadata(
                    analysis_meta.get("model_role_metadata")
                )
            )
            report_lines.extend(
                ConclusionRenderer.render_execution_trace(
                    analysis_meta.get("execution_trace")
                )
            )
            report_lines.extend(
                ConclusionRenderer.render_what_if_plan(
                    analysis_meta.get("what_if_plan")
                )
            )

    if recommendations:
        report_lines.extend(
            [
                "\n## 最终推荐标的\n",
                (
                    "| 排名 | 代码 | 名称 | 类别 | 现价 | 推荐买入价 | "
                    "目标卖出价 | 止损价 | 推荐仓位 | 金额 | "
                    "预期空间 | v13分支支持 |\n"
                ),
                (
                    "|:---:|:---|:---|:---|---:|---:|---:|---:|"
                    "---:|---:|---:|---:|\n"
                ),
            ]
        )
        for item in recommendations:
            stock_name = str(
                item.get("company_name")
                or item.get("name")
                or get_stock_name(item["symbol"], market=settings.market)
            ).strip()
            if _is_unknown_stock_name(stock_name):
                stock_name = get_stock_name(
                    item["symbol"], market=settings.market
                )
            current_price = float(item.get("current_price", 0))
            entry_low = float(
                item.get("entry_price_range", {}).get(
                    "low", current_price * 0.99
                )
            )
            entry_high = float(
                item.get("entry_price_range", {}).get(
                    "high", current_price * 1.01
                )
            )
            display_entry_price = float(
                item.get("recommended_entry_price")
                or (entry_low + entry_high) / 2
                or current_price
            )
            report_lines.append(
                f"| {item['rank']} | {item['symbol']} | {stock_name} | "
                f"{item['category_name']} | {settings.currency_symbol}"
                f"{current_price:.2f} | {settings.currency_symbol}"
                f"{display_entry_price:.2f} | {settings.currency_symbol}"
                f"{item['target_price']:.2f} | {settings.currency_symbol}"
                f"{item['stop_loss_price']:.2f} | "
                f"{item['portfolio_weight']:.1%} | {settings.currency_symbol}"
                f"{item['portfolio_amount']:,.0f} | "
                f"{float(item['expected_upside']):.1%} | "
                f"{item['branch_positive_count']}/"
                f"{BRANCH_SUPPORT_DENOMINATOR} |\n"
            )

        for item in recommendations[:12]:
            report_lines.extend(
                ConclusionRenderer.render_stock(item, settings.market)
            )
            display_entry_price = float(
                item.get("recommended_entry_price")
                or item.get("current_price")
                or 0.01
            )
            max_loss = (
                float(item.get("stop_loss_price", 0.0))
                / max(display_entry_price, 0.01)
                - 1
            ) * 100
            report_lines.append(f"- 最大亏损: {max_loss:.1f}%\n")
            risk_observation = (
                "；".join(
                    _dedupe_text(
                        [
                            _sanitize_text(flag)
                            for flag in item.get("risk_flags", [])
                        ]
                    )[:3]
                )
                or "无"
            )
            report_lines.append(
                f"- 风险观察: {risk_observation}\n"
            )
            report_lines.append("")
    else:
        report_lines.append("\n## 最终推荐标的\n")
        report_lines.append(
            "- 当前没有满足条件的最终候选，建议继续以现金和观察仓位为主。\n"
        )

    report_lines.append("\n## v13 四分支结论\n")
    for branch_name in CANONICAL_BRANCH_ORDER:
        if branch_name in branch_summary:
            report_lines.extend(
                line + "\n"
                for line in ConclusionRenderer.render_branch(
                    branch_name, branch_summary[branch_name]
                )
            )

    report_lines.append("## 附录：工程诊断与详细运行日志\n")
    if diagnostics["diagnostic_notes"]:
        for note in diagnostics["diagnostic_notes"]:
            report_lines.append(f"- {note}\n")
    else:
        report_lines.append(
            f"- 当前未记录新增工程诊断（{summary['total_batches']}/"
            f"{summary['total_batches']} 批次）。\n"
        )

    for category, batches in all_results.items():
        for batch in batches:
            if not batch.get("execution_log"):
                continue
            log_count = len(batch.get("execution_log", []))
            report_lines.append(
                f"- {category_name(category, settings.market)} 批次 "
                f"{batch.get('batch_id', '-')}: "
                f"附带 {log_count}/{log_count} 条运行日志。\n"
            )

    summary_lines = [
        f"# {settings.report_flag} {settings.market_name}全市场分析摘要\n",
        f"**生成时间**: {summary['generated_at']}\n",
        (
            f"**分析覆盖**: {summary['total_stocks']} 只股票，"
            f"{summary['total_batches']} 个批次\n"
        ),
        f"**分析 universe**: {analysis_meta.get('universe', 'full_a')}\n",
        "\n## 三句话执行摘要\n",
    ]
    if data_snapshot_summary:
        summary_lines.insert(4, f"**本地数据快照**: {data_snapshot_summary}\n")
    for line in executive_summary:
        summary_lines.append(f"- {line}\n")
    summary_lines.append("\n## 执行提醒\n")
    for note in portfolio_plan["execution_notes"]:
        summary_lines.append(f"- {_sanitize_text(note)}\n")

    summary_file = target_dir / f"{settings.market}_Full_Report_{timestamp}.md"
    data_file = target_dir / f"{settings.market}_Trade_Data_{timestamp}.json"
    report_file = target_dir / f"{settings.market}_Trade_Report_{timestamp}.md"
    with open(summary_file, "w", encoding="utf-8") as file:
        file.writelines(summary_lines)
    with open(report_file, "w", encoding="utf-8") as file:
        file.writelines(report_lines)
    with open(data_file, "w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2, ensure_ascii=False)

    candidate_file = save_candidate_index(
        all_results, market=settings.market, output_dir=str(target_dir)
    )
    return {
        "summary_report": str(summary_file),
        "trade_report": str(report_file),
        "trade_data": str(data_file),
        "candidate_index": candidate_file,
    }


__all__ = [
    "ActionConsistencyGuard",
    "BRANCH_LABELS",
    "BRANCH_SUPPORT_DENOMINATOR",
    "ConclusionRenderer",
    "DiagnosticsBucketizer",
    "ExecutiveSummaryBuilder",
    "build_full_market_trade_plan",
    "category_name",
    "generate_full_report",
    "save_candidate_index",
]
