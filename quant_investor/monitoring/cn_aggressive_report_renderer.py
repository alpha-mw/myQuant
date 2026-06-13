"""Markdown report renderer for the CN aggressive tracker."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_investor.monitoring.cn_aggressive_reporting import (
    _format_holding_advice_line,
    _format_holding_snapshot_set,
    _format_top_delta_vs_source_record,
    _format_top_holdings_by_unrealized_pnl,
    _format_warning_count_summary,
    _format_warning_messages,
)
from quant_investor.monitoring.cn_aggressive_review_layer import _render_dag_compliance_markdown


def build_formal_report_text(
    *,
    source_record: str,
    timestamp_long: str,
    quote_snapshot: str,
    completeness_passed: bool,
    decision_data_sufficient: bool,
    dag_four_branch_compliance: dict[str, Any],
    rebalance_reason: str,
    decision_guardrail: Any,
    data_status: str,
    orders: list[Any],
    switch_plan_df: pd.DataFrame,
    typed_warning_codes: list[str],
    codex_handoff_active: bool,
    review_attempt_summary: dict[str, Any],
    review_effective_summary: dict[str, Any],
    degraded_symbols: dict[str, str],
    candidate_pool: pd.DataFrame,
    tomorrow_focus: list[str],
    completeness_before: dict[str, Any],
    attempted_backfill: bool,
    data_snapshot_lines: list[str],
    total_value_after: float,
    portfolio_pnl_after: float,
    initial_capital: float,
    source_total_value: float,
    float_winners: pd.DataFrame,
    float_losers: pd.DataFrame,
    holdings_review: pd.DataFrame,
    index_quote_codes: list[str],
    indices: dict[str, dict[str, Any]],
    style_text: str,
    breadth: dict[str, dict[str, Any]],
    strongest_theme_text: str,
    weakest_theme_text: str,
    theme_strength: list[dict[str, Any]],
    review_model_role_metadata: dict[str, Any],
    review_layer: dict[str, Any],
    diagnostic_table: str,
    formal_warnings: list[Any],
    candidate_lines: list[str],
    switch_now: bool,
    switch_prepare: bool,
    factor_shadow_lines: list[str],
) -> str:
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
        f"- 完整性校验：**{'已通过' if completeness_passed else '未通过'}**",
        f"- 决策数据口径：**{'盘中前日线+实时行情可用' if decision_data_sufficient else 'strict 日线完整性优先'}**",
        "- 分析口径：**直接基于本地已有数据，不自动补数，不把完整性校验作为正式结论前置阻断。**",
        "- 分析链路：**统一 DAG / review-layer（逐持仓主线复核）**",
        (
            f"- DAG 四分支执行状态：**{dag_four_branch_compliance['status']}**；"
            f"`complete={str(bool(dag_four_branch_compliance['complete'])).lower()}`"
        ),
        "",
        "## 0. 正式结果速览",
        "",
        f"- 正式结论：**{rebalance_reason}**",
        f"- 报告展示标签：**`{decision_guardrail.display_label}`**",
        f"- 数据完整性状态：**{data_status}**",
        f"- 今日是否执行调仓：**{'是' if orders else '否'}**",
        (
            f"- 今日是否执行换仓：**{'是' if switch_now else '否'}**"
            if switch_plan_df is not None
            else "- 今日是否执行换仓：**否**"
        ),
        (
            f"- Typed diagnostics：`{', '.join(typed_warning_codes)}`"
            if typed_warning_codes
            else "- Typed diagnostics：无"
        ),
        (
            f"- 决策护栏说明：{decision_guardrail.arbitration_note}"
            if str(decision_guardrail.arbitration_note).strip()
            else "- 决策护栏说明：无"
        ),
        (
            "- LLM 复核状态：**本地 LLM 未执行；Codex 接管解释**，本地调用 `0` 次"
            if codex_handoff_active
            else (
                f"- LLM 复核状态：**已执行**，原始尝试 `{review_attempt_summary.get('call_count', 0)}` 次"
                f"（成功 `{review_attempt_summary.get('success_count', 0)}` / 失败 `{review_attempt_summary.get('failed_count', 0)}` / "
                f"fallback `{review_attempt_summary.get('fallback_count', 0)}`），"
                f"有效输出 `{review_effective_summary.get('call_count', 0)}` 次，"
                f"`{review_attempt_summary.get('total_tokens', 0)}` tokens，"
                f"估算成本 `${float(review_attempt_summary.get('estimated_cost_usd', 0.0)):.6f}`"
            )
        ),
        (
            "- Review-layer 降级：`"
            + "，".join(sorted(degraded_symbols.keys()))
            + "`"
            if degraded_symbols
            else "- Review-layer 降级：无"
        ),
        (
            "- 备选投资建议："
            + "；".join(
                f"{row.symbol}({row.name})/{row.theme_label}"
                for row in candidate_pool.head(3).itertuples()
            )
            if not candidate_pool.empty
            else "- 备选投资建议：暂无"
        ),
        f"- 明日准备事项：{'；'.join(tomorrow_focus)}",
        "",
        "## 2. 数据状态与本地快照",
        "",
        f"- 当前运行时间：`{timestamp_long}`",
        f"- 首轮完整性：`{'通过' if completeness_before['complete'] else '未通过'}`，阻塞缺口 `{int(completeness_before['blocking_incomplete_count'])}` 个",
        f"- 是否执行补数：`{'是' if attempted_backfill else '否'}`",
        "- 本轮策略：即使存在数据滞后或局部缺口，也继续给出正式分析与正式建议，但会明确披露数据限制。",
        *data_snapshot_lines,
        "",
        "## 3. 当前组合盈亏判断",
        "",
        f"- 截至 `quote_snapshot={quote_snapshot or 'N/A'}`，组合总资产 **{total_value_after:,.2f} 元**，较初始资金 **{portfolio_pnl_after:,.2f} 元**，收益率 **{portfolio_pnl_after / initial_capital:.2%}**。",
        f"- 相对上一条正式记录 `{source_record}` 的 **{source_total_value:,.2f} 元**，当前盘中净值变动 **{total_value_after - source_total_value:,.2f} 元**。",
        "- 当前浮盈仓位：" + _format_top_holdings_by_unrealized_pnl(float_winners, positive=True),
        "- 当前浮亏前三：" + _format_top_holdings_by_unrealized_pnl(float_losers, positive=False),
        "- 相对上一条正式记录的正向收益贡献："
        + _format_top_delta_vs_source_record(holdings_review, positive=True),
        "- 相对上一条正式记录的拖累来源前三："
        + _format_top_delta_vs_source_record(holdings_review, positive=False),
        "",
        "## 4. A股整体市场风格与指数结构",
        "",
        "### 4.1 指数结构（盘中快照）",
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
            "### 4.2 广度与市场内部状态（基于最新完整日线）",
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
            "### 4.3 科技 / 高端制造主线强弱",
            "",
            f"- {strongest_theme_text}",
            f"- {weakest_theme_text}",
            (
                f"- 主题强弱排序：{', '.join(item['theme'] for item in theme_strength)}"
                if theme_strength
                else "- 主题强弱排序：暂无"
            ),
            "",
            "## 5. 当前策略持仓复盘",
            "",
            "### 5.1 持仓相对强弱",
            "",
            "- 今日相对最强："
            + _format_holding_snapshot_set(holdings_review.sort_values(
                ["today_change_pct", "score_full_market"], ascending=[False, False]
            ).head(3)),
            "- 今日相对最弱："
            + _format_holding_snapshot_set(holdings_review.sort_values(
                ["today_change_pct", "score_full_market"], ascending=[True, True]
            ).head(3)),
            "",
            "### 5.2 当前弱点与结构预警",
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
            "### 5.3 统一 DAG / Review Layer 复核",
            "",
            (
                "- 本地 LLM 调用：`已禁用`；LLM 解释由 Codex 在运行后读取正式产物接管。"
                if codex_handoff_active
                else (
                    f"- 分支模型：`{review_model_role_metadata.get('resolved_branch_model') or review_model_role_metadata.get('branch_model') or 'N/A'}`；"
                    f"主模型：`{review_model_role_metadata.get('resolved_master_model') or review_model_role_metadata.get('master_model') or 'N/A'}`"
                )
            ),
            (
                f"- 原始尝试汇总：`{review_attempt_summary.get('call_count', 0)}` 次，"
                f"成功 `{review_attempt_summary.get('success_count', 0)}` 次，"
                f"失败 `{review_attempt_summary.get('failed_count', 0)}` 次，"
                f"fallback `{review_attempt_summary.get('fallback_count', 0)}` 次，"
                f"tokens `{review_attempt_summary.get('total_tokens', 0)}`，"
                f"成本 `${float(review_attempt_summary.get('estimated_cost_usd', 0.0)):.6f}`"
            ),
            (
                f"- 有效输出汇总：`{review_effective_summary.get('call_count', 0)}` 次，"
                f"成功 `{review_effective_summary.get('success_count', 0)}` 次，"
                f"tokens `{review_effective_summary.get('total_tokens', 0)}`，"
                f"成本 `${float(review_effective_summary.get('estimated_cost_usd', 0.0)):.6f}`"
            ),
            (
                "- review-layer fallback："
                + "；".join(review_layer.get("fallback_reasons", [])[:6])
                if review_layer.get("fallback_reasons")
                else "- review-layer fallback：无"
            ),
            (
                "- review-layer 降级标的："
                + "；".join(f"{symbol}: {reason}" for symbol, reason in list(degraded_symbols.items())[:6])
                if degraded_symbols
                else "- review-layer 降级标的：无"
            ),
            "",
            *_render_dag_compliance_markdown(dag_four_branch_compliance),
            "",
            "### 5.4 是否需要调仓",
            "",
            f"- 报告展示标签：`{decision_guardrail.display_label}`",
            (
                f"- 护栏仲裁说明：{decision_guardrail.arbitration_note}"
                if str(decision_guardrail.arbitration_note).strip()
                else "- 护栏仲裁说明：无"
            ),
            (
                f"- 触发 warning codes：`{', '.join(typed_warning_codes)}`"
                if typed_warning_codes
                else "- 触发 warning codes：无"
            ),
            "",
            "#### 5.4.1 决策诊断",
            "",
            diagnostic_table,
            "",
            "#### 5.4.2 正式建议",
            "",
            f"- 正式建议：**{'执行温和调仓' if orders else '本次不执行调仓'}**",
            (
                "- 原因：主线最强分支重新走强，今天更像内部修复而不是主线失效；弱支线虽然没有完全修复，但还不足以在单日里推翻长期主线。"
                if not orders
                else "- 原因：弱支线在完整日线与盘中都继续落后，已满足温和减仓条件。"
            ),
        ]
    )
    for row in holdings_review.itertuples():
        report_lines.append(_format_holding_advice_line(row))
    report_lines.extend(
        [
            "",
            "#### 5.4.3 证据质量与工程诊断",
            "",
            f"- 诊断汇总：{_format_warning_count_summary(formal_warnings)}",
            f"- Material 级证据缺口：{_format_warning_messages(formal_warnings, severity='material', limit=3)}",
            (
                f"- LLM / review-layer 状态：原始尝试 `{review_attempt_summary.get('call_count', 0)}` 次，"
                f"有效输出 `{review_effective_summary.get('call_count', 0)}` 次；"
                + (
                    "降级标的 `"
                    + "，".join(sorted(degraded_symbols.keys()))
                    + "`。"
                    if degraded_symbols
                    else "无逐标的 review-layer 降级。"
                )
            ),
            "- 工程诊断说明：provider、snapshot、旧 intelligence batch 等诊断仅用于说明证据等级，不直接作为逐票投资建议正文。",
            "",
            "##### 持仓诊断明细",
            "",
            diagnostic_table,
        ]
    )
    report_lines.extend(
        [
            "",
            "### 5.5 备选投资建议",
            "",
            (
                "- 本轮备选池来自本地 `results/cn_analysis_full/all_candidates.json` 与最新主导本地快照强度交叉筛选。"
                if not candidate_pool.empty
                else "- 本轮未提取到有效备选池。"
            ),
            *candidate_lines,
            "",
            "### 5.6 现持仓与备选标的换仓比较",
            "",
            (
                "- 正式换仓结论：**执行换仓**，优先按 20% 试探性替换最弱持仓。"
                if switch_now
                else (
                    "- 正式换仓结论：**暂不换仓，但已形成预备换仓对象**。"
                    if switch_prepare
                    else "- 正式换仓结论：**暂不换仓**，现阶段以观察池跟踪为主。"
                )
            ),
            (
                "- 不换仓条件：当前本地 strict 快照仍不完整，先不把结构优势直接转成实单。"
                if not completeness_passed
                else "- 不换仓条件：若弱持仓重新收复阶段止损位且候选优势收敛，则维持原组合。"
            ),
            "",
        ]
    )
    if switch_plan_df.empty:
        report_lines.append("- 当前没有形成明确的一对一换仓比较。")
    else:
        for row in switch_plan_df.itertuples(index=False):
            report_lines.append(
                f"- `{row.sell_symbol}`（{row.sell_name}） vs `{row.buy_symbol}`（{row.buy_name}）："
                f"候选主线 `{row.buy_theme}`，强度分差 `{float(row.score_gap):+.3f}`，"
                f"20日动量差 `{float(row.ret20_gap):+.2%}`；建议 `{row.action}`，"
                f"优先级 `{row.priority}`，比例提示 `{row.switch_ratio_hint}`；"
                f"触发阈值：{row.trigger_threshold}；不换仓条件：{row.no_switch_condition}"
            )

    report_lines.extend(
        [
            "",
            "### 5.7 因子库状态（只读影子观察）",
            "",
            *factor_shadow_lines,
            "",
            "## 6. 面向明日的观察重点和准备事项",
            "",
        ]
    )
    for idx, item in enumerate(tomorrow_focus, start=1):
        report_lines.append(f"{idx}. {item}")

    return "\n".join(report_lines)
