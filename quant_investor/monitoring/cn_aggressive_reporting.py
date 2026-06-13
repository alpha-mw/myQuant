"""Report formatting and output persistence helpers for the CN tracker."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.monitoring.cn_aggressive_rebalance import ProposedOrder
from quant_investor.monitoring.cn_aggressive_utils import _safe_float, _safe_pct


def _format_symbol_set(symbols: list[str]) -> str:
    return " / ".join(symbols)


def _format_holding_snapshot_set(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无"
    return " / ".join(_format_holding_snapshot(row) for row in frame.itertuples())


def _format_holding_snapshot(row: Any) -> str:
    buy_price = _safe_float(
        getattr(row, "buy_price", getattr(row, "avg_cost", 0.0)),
        0.0,
    )
    unrealized_pnl = _safe_float(getattr(row, "unrealized_pnl", 0.0), 0.0)
    unrealized_pnl_pct = _safe_float(getattr(row, "unrealized_pnl_pct", 0.0), 0.0)
    return (
        f"{row.symbol}({row.name}) 持有成本 `{buy_price:.2f}`，"
        f"PNL `{_format_signed_money(unrealized_pnl)}`（{unrealized_pnl_pct:+.2%}）"
    )


def _format_top_holdings_by_unrealized_pnl(frame: pd.DataFrame, *, positive: bool) -> str:
    if frame.empty or "unrealized_pnl" not in frame.columns:
        return "无"

    filtered = frame[frame["unrealized_pnl"] > 0] if positive else frame[frame["unrealized_pnl"] < 0]
    if filtered.empty:
        return "无"

    ordered = filtered.sort_values("unrealized_pnl", ascending=not positive)
    return "；".join(
        f"{row.symbol}({row.name}) {row.unrealized_pnl:,.2f} 元"
        for row in ordered.head(3).itertuples()
    )


def _format_top_delta_vs_source_record(frame: pd.DataFrame, *, positive: bool) -> str:
    if frame.empty or "delta_vs_source_record" not in frame.columns:
        return "无"

    filtered = (
        frame[frame["delta_vs_source_record"] > 0]
        if positive
        else frame[frame["delta_vs_source_record"] < 0]
    )
    if filtered.empty:
        return "无"

    ordered = filtered.sort_values("delta_vs_source_record", ascending=not positive)
    return "；".join(
        f"{row.symbol} {row.delta_vs_source_record:,.2f} 元"
        for row in ordered.head(3).itertuples()
    )


def _format_signed_money(value: float) -> str:
    return f"{value:+,.2f} 元"


def _format_holding_advice_line(row: Any) -> str:
    price = _safe_float(getattr(row, "current_price", 0.0), 0.0)
    buy_price = _safe_float(getattr(row, "buy_price", 0.0), 0.0)
    buy_value = _safe_float(getattr(row, "buy_value", 0.0), 0.0)
    unrealized_pnl = _safe_float(getattr(row, "unrealized_pnl", 0.0), 0.0)
    unrealized_pnl_pct = _safe_float(getattr(row, "unrealized_pnl_pct", 0.0), 0.0)
    stop_price = _safe_float(getattr(row, "stage_stop_price", 0.0), 0.0)
    target_price = _safe_float(getattr(row, "stage_target_price", 0.0), 0.0)
    stop_buffer = _safe_pct(price - stop_price, stop_price)
    hard_signal = "未触发阶段止损" if price >= stop_price else "低于阶段止损，需跟踪减仓确认"
    return (
        f"- `{row.symbol}`（{row.name}）：建议 `{row.recommended_action}`，"
        f"持仓角色 `{row.position_role}`；当前价 `{price:.2f}`，"
        f"持有成本 `{buy_price:.2f}`（成本金额 `{buy_value:,.2f} 元`），"
        f"阶段止损 `{stop_price:.2f}`（缓冲 {stop_buffer:+.2%}），"
        f"阶段目标 `{target_price:.2f}`；浮动 PNL `{_format_signed_money(unrealized_pnl)}`"
        f"（{unrealized_pnl_pct:+.2%}），"
        f"较上一条记录 `{_format_signed_money(float(row.delta_vs_source_record))}`；"
        f"全市场强度排名 `{int(row.rank_full_market)}`，今日涨跌 `{float(row.today_change_pct):+.2f}%`；"
        f"{hard_signal}。"
    )


def _format_candidate_advice_line(row: Any, switch_row: dict[str, Any] | None) -> str:
    source_label = "最新正式筛选结果" if str(row.candidate_source) == "latest_formal_screening" else "本地全市场强度"
    relative_advantage = (
        f"相对 `{switch_row['sell_symbol']}` 更优，强度排名前移 `{int(switch_row['sell_rank_full_market']) - int(switch_row['buy_rank_full_market'])}` 位"
        f"，20日动量高出 `{float(switch_row['ret20_gap']):+.2%}`；"
        if switch_row
        else f"当前在本地候选中位列前 `{int(row.candidate_rank)}`，20日收益 `{float(row.ret20):+.2%}`；"
    )
    if str(row.evidence_quality) == "高":
        major_risk = "若回撤跌破阶段止损位，短线强度可能失真；"
    elif str(row.evidence_quality) == "中":
        major_risk = "盘中采用前一交易日稳定日线结合实时行情，收盘后需用当日日线复核；"
    else:
        major_risk = "本地 strict 快照仍有缺口，结论依赖主导本地快照延续性；"
    trigger = (
        str(switch_row["trigger_threshold"])
        if switch_row
        else "若连续两次正式复盘仍在候选前列，可升级为优先观察对象"
    )
    return (
        f"- `{row.symbol}`（{row.name}）：主线 `{row.theme_label}`，来源 `{source_label}`；"
        f"{relative_advantage}主要风险：{major_risk}"
        f"触发条件：{trigger}；证据质量：`{row.evidence_quality}`。"
    )


def _format_warning_count_summary(warnings: list[Any]) -> str:
    if not warnings:
        return "无"
    counts = Counter(str(warning.code) for warning in warnings)
    return "，".join(f"{code}={count}" for code, count in sorted(counts.items()))


def _format_warning_messages(warnings: list[Any], *, severity: str | None = None, limit: int = 6) -> str:
    selected = [
        warning
        for warning in warnings
        if severity is None or str(getattr(warning, "severity", "")) == severity
    ]
    if not selected:
        return "无"
    messages = [str(warning.human_message).strip() for warning in selected if str(warning.human_message).strip()]
    return "；".join(messages[:limit]) + (" ..." if len(messages) > limit else "")


def _format_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text or "未知"


def _dominant_trade_date(date_counts: dict[str, Any]) -> str:
    normalized = {
        str(date).strip(): int(count or 0)
        for date, count in (date_counts or {}).items()
        if str(date).strip()
    }
    if not normalized:
        return ""
    return max(normalized.items(), key=lambda item: (item[1], item[0]))[0]


def _resolve_analysis_trade_date(completeness_report: dict[str, Any]) -> str:
    categories = dict(completeness_report.get("categories", {}) or {})
    full_a = dict(categories.get("full_a", {}) or {})
    dominant = _dominant_trade_date(full_a.get("date_counts", {}) or {})
    if dominant:
        return dominant

    aggregate_counts: dict[str, int] = {}
    for payload in categories.values():
        for date, count in (payload.get("date_counts", {}) or {}).items():
            key = str(date).strip()
            if not key:
                continue
            aggregate_counts[key] = aggregate_counts.get(key, 0) + int(count or 0)
    dominant = _dominant_trade_date(aggregate_counts)
    if dominant:
        return dominant
    return str(
        completeness_report.get("effective_target_trade_date")
        or completeness_report.get("latest_trade_date")
        or ""
    )


def _build_data_status_summary(
    completeness_report: dict[str, Any],
    analysis_trade_date: str,
    decision_data_sufficient: bool = False,
) -> str:
    target_trade_date = _format_trade_date(completeness_report.get("latest_trade_date"))
    effective_target_trade_date = _format_trade_date(
        completeness_report.get("effective_target_trade_date")
        or completeness_report.get("latest_trade_date")
    )
    strict_trade_date = _format_trade_date(
        completeness_report.get("strict_trade_date")
        or completeness_report.get("latest_trade_date")
    )
    freshness_mode = str(completeness_report.get("freshness_mode") or "strict")
    blocking_count = int(completeness_report.get("blocking_incomplete_count", 0) or 0)
    coverage_ratio = float(completeness_report.get("coverage_ratio", 0.0) or 0.0)
    pre_listing_count = len(completeness_report.get("pre_listing_symbols", []) or [])
    full_a = (
        completeness_report.get("categories", {})
        .get("full_a", {})
        or {}
    )
    dominant_count = int((full_a.get("date_counts", {}) or {}).get(analysis_trade_date, 0) or 0)
    dominant_expected = int(full_a.get("expected", 0) or 0)
    suspended_stale_count = len(
        (
            completeness_report.get("categories", {})
            .get("full_a", {})
            .get("suspended_stale_symbols", [])
            or []
        )
    )

    base = f"本地主导A股日线快照位于 `{_format_trade_date(analysis_trade_date)}`"
    if _format_trade_date(analysis_trade_date) != effective_target_trade_date:
        base += (
            f"，而当前完整性目标交易日为 `{effective_target_trade_date}`"
            f"（strict 最新 `{strict_trade_date}` / 报告 latest `{target_trade_date}`）"
        )
    else:
        base += f"，当前完整性目标交易日同样为 `{effective_target_trade_date}`"
    extras = [
        f"主导快照覆盖 `{dominant_count}/{dominant_expected}`",
        f"覆盖率 `{coverage_ratio:.1%}`",
        f"停牌/长期停牌例外 `{suspended_stale_count}` 个",
        f"预上市样本 `{pre_listing_count}` 个",
        f"完整性模式 `{freshness_mode}`",
    ]
    if blocking_count > 0:
        extras.insert(0, f"阻塞缺口 `{blocking_count}` 个")
    else:
        extras.insert(0, "阻塞缺口 `0` 个")
    suffix = "（" + "；".join(extras) + "）"
    if decision_data_sufficient and _format_trade_date(analysis_trade_date) != effective_target_trade_date:
        suffix += "；盘中决策口径接受前一交易日稳定日线结合实时行情，当日 strict 日线未出不视为决策阻断。"
    return base + suffix


def _build_data_snapshot_lines(
    completeness_report: dict[str, Any],
    quote_snapshot: str,
    analysis_trade_date: str,
    decision_data_sufficient: bool = False,
) -> list[str]:
    categories = dict(completeness_report.get("categories", {}) or {})
    full_a = dict(categories.get("full_a", {}) or {})
    analysis_count = int((full_a.get("date_counts", {}) or {}).get(analysis_trade_date, 0) or 0)
    lines = [
        (
            f"- 分析采用的主导本地快照：`{_format_trade_date(analysis_trade_date)}`，"
            f"`full_a` 覆盖 `{analysis_count}/{int(full_a.get('expected', 0) or 0)}`"
        ),
        f"- 本地最新交易日：`{_format_trade_date(completeness_report.get('latest_trade_date'))}`",
        f"- strict 最新交易日：`{_format_trade_date(completeness_report.get('strict_trade_date'))}`",
        f"- stable 最新交易日：`{_format_trade_date(completeness_report.get('stable_trade_date'))}`",
        f"- 当前采用目标交易日：`{_format_trade_date(completeness_report.get('effective_target_trade_date'))}`",
        f"- 新鲜度模式：`{completeness_report.get('freshness_mode') or 'strict'}`",
        (
            f"- 覆盖完成度：`{int(completeness_report.get('coverage_complete_count', 0) or 0)}/"
            f"{int(completeness_report.get('expected_scope_count', 0) or 0)}` "
            f"（{float(completeness_report.get('coverage_ratio', 0.0) or 0.0):.1%}）"
        ),
        (
            f"- 完整性状态：`{'通过' if completeness_report.get('complete') else '未通过'}`，"
            f"阻塞缺口 `{int(completeness_report.get('blocking_incomplete_count', 0) or 0)}` 个"
        ),
        (
            f"- 指数/持仓盘中快照：`{quote_snapshot}`；指数与持仓现价使用盘中行情，"
            "广度、主题与全市场强弱仍以本地最新日线快照为准。"
            if quote_snapshot
            else "- 指数/持仓盘中快照：`N/A`，本轮全部结论基于本地日线快照。"
        ),
    ]
    if decision_data_sufficient:
        lines.append(
            "- 决策数据口径：`盘中前日线+实时行情可用`；当日 strict 日线缺口仅作数据披露，"
            "不自动降级正式投资结论。"
        )

    pre_listing = list(completeness_report.get("pre_listing_symbols", []) or [])
    if pre_listing:
        lines.append(
            "- 预上市样本：`"
            + "，".join(
                f"{item.get('symbol')}@{_format_trade_date(item.get('list_date'))}"
                for item in pre_listing[:8]
            )
            + (" ...`" if len(pre_listing) > 8 else "`")
        )

    for category in completeness_report.get("categories_checked", []):
        payload = categories.get(category, {})
        if not payload:
            continue
        category_dominant_trade_date = _dominant_trade_date(payload.get("date_counts", {}) or {})
        category_dominant_count = int((payload.get("date_counts", {}) or {}).get(category_dominant_trade_date, 0) or 0)
        lines.append(
            f"- {category}：目标 `{_format_trade_date(payload.get('latest_trade_date'))}`，"
            f"主导本地快照 `{_format_trade_date(category_dominant_trade_date)}` "
            f"（`{category_dominant_count}/{int(payload.get('expected', 0) or 0)}`），"
            f"目标覆盖 `{int(payload.get('coverage_complete_count', 0) or 0)}/"
            f"{int(payload.get('expected', 0) or 0)}`，"
            f"阻塞缺口 `{int(payload.get('blocking_incomplete_count', 0) or 0)}`，"
            f"停牌例外 `{len(payload.get('suspended_stale_symbols', []) or [])}`"
        )
        blocking_missing = list(payload.get("blocking_missing_symbols", []) or [])
        if blocking_missing:
            lines.append(
                "- "
                + category
                + " 阻塞缺失：`"
                + "，".join(blocking_missing[:10])
                + (" ...`" if len(blocking_missing) > 10 else "`")
            )
        blocking_stale = list(payload.get("blocking_stale_symbols", []) or [])
        if blocking_stale:
            lines.append(
                "- "
                + category
                + " 阻塞旧档：`"
                + "，".join(
                    f"{item.get('symbol')}@{_format_trade_date(item.get('latest_local_date'))}"
                    for item in blocking_stale[:10]
                )
                + (" ...`" if len(blocking_stale) > 10 else "`")
            )
        suspended = list(payload.get("suspended_stale_symbols", []) or [])
        if suspended:
            lines.append(
                "- "
                + category
                + " 停牌/长期停牌例外：`"
                + "，".join(
                    f"{item.get('symbol')}@{_format_trade_date(item.get('latest_local_date'))}"
                    for item in suspended[:10]
                )
                + (" ...`" if len(suspended) > 10 else "`")
            )
    return lines


def _write_outputs(
    base_dir: Path,
    run_dir: Path,
    report_text: str,
    holdings_review: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    switch_plan_df: pd.DataFrame,
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
    candidate_path = run_dir / "candidate_pool.csv"
    switch_path = run_dir / "switch_plan.csv"
    ledger_path = run_dir / "ledger.csv"
    orders_path = run_dir / "orders.csv"
    pnl_path = run_dir / "pnl_summary.csv"
    snapshot_path = run_dir / "market_snapshot.json"
    manifest_path = run_dir / "manifest.json"
    runtime_profile_path = run_dir / "runtime_profile.json"

    report_path.write_text(report_text, encoding="utf-8")
    holdings_review.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    candidate_pool.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    switch_plan_df.to_csv(switch_path, index=False, encoding="utf-8-sig")
    ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    orders_df.to_csv(orders_path, index=False, encoding="utf-8-sig")
    pnl_summary_df.to_csv(pnl_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(market_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime_profile = manifest.get("runtime_profile") or market_snapshot.get("runtime_profile") or {}
    if runtime_profile:
        runtime_profile_path.write_text(
            json.dumps(runtime_profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    prefix = f"aggressive_portfolio_{manifest['timestamp']}_formal"
    shutil.copy2(report_path, raw_dir / f"{prefix}_report.md")
    shutil.copy2(holdings_path, raw_dir / f"{prefix}_holdings_review.csv")
    shutil.copy2(candidate_path, raw_dir / f"{prefix}_candidate_pool.csv")
    shutil.copy2(switch_path, raw_dir / f"{prefix}_switch_plan.csv")
    shutil.copy2(ledger_path, raw_dir / f"{prefix}_ledger.csv")
    shutil.copy2(orders_path, raw_dir / f"{prefix}_orders.csv")
    shutil.copy2(pnl_path, raw_dir / f"{prefix}_pnl_summary.csv")
    if runtime_profile_path.exists():
        shutil.copy2(runtime_profile_path, raw_dir / "runtime_profile.json")


def _build_notes_payload(
    trade_date: str,
    data_status: str,
    market_core_view: str,
    pnl_summary: dict[str, Any],
    orders: list[ProposedOrder],
    switch_plan_df: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    tomorrow_focus: list[str],
) -> str:
    if orders:
        order_text = "；".join(
            f"{order.symbol} {order.action} {order.shares}股 @ {order.price:.2f}" for order in orders
        )
    else:
        order_text = "无，本日维持现有结构。"

    if switch_plan_df.empty:
        switch_text = "否，暂无明显优于现持仓的可执行换仓对象。"
        switch_detail_text = "维持原组合，继续跟踪候选观察池。"
    else:
        top_switch = switch_plan_df.iloc[0]
        action = str(top_switch["action"])
        if action == "switch_now":
            switch_text = "是，已形成明确换仓优先级。"
        elif action == "prepare_switch":
            switch_text = "否，但已形成预备换仓对象。"
        else:
            switch_text = "否，先保留观察池。"
        sell_label = f"{top_switch['sell_symbol']} {top_switch['sell_name']}"
        buy_label = f"{top_switch['buy_symbol']} {top_switch['buy_name']}"
        switch_detail_text = (
            f"{sell_label} -> {buy_label}，优先级 `{top_switch['priority']}`，"
            f"触发条件：{top_switch['trigger_threshold']}"
        )

    if candidate_pool.empty:
        candidate_text = "暂无可用备选。"
    else:
        top_rows = candidate_pool.head(3)
        candidate_text = "；".join(
            f"{row.symbol}({row.name})/{row.theme_label}/证据{row.evidence_quality}"
            for row in top_rows.itertuples()
        )

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
            f"- 备选投资建议：{candidate_text}",
            f"- 是否调仓：{'是' if orders else '否'}",
            f"- 调仓内容：{order_text}",
            f"- 是否换仓：{switch_text}",
            f"- 换仓内容：{switch_detail_text}",
            f"- 明日观察重点：{'；'.join(tomorrow_focus)}",
        ]
    )
