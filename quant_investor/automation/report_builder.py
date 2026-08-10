from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from quant_investor.automation.history_loader import HistoryLoader
from quant_investor.config import config as runtime_config


log = logging.getLogger("daily_runner")

_BRANCH_LABEL_MAP = {
    "quant": "量化因子",
    "fundamental": "基本面",
    "macro": "宏观",
}

_ACTION_EMOJI = {
    "买入": "🟢",
    "轻仓试错": "🟡",
    "观察": "⚪",
    "持有": "🔵",
    "减仓": "🟠",
    "清仓": "🔴",
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_average(values: list[Any], default: float = 0.0) -> float:
    nums = [_safe_float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else default


def _confidence_label(c: float) -> str:
    if c >= 0.70:
        return "高"
    if c >= 0.45:
        return "中"
    return "低"


class ReportBuilder:
    """从 pipeline 结果构建 8 章节 Markdown 决策报告。"""

    @staticmethod
    def _display_name(item: dict[str, Any]) -> str:
        symbol = str(item.get("symbol", "")).strip()
        company_name = str(item.get("company_name") or item.get("name") or "").strip()
        return f"{symbol} {company_name}".strip() if company_name else symbol

    def build(
        self,
        pipeline_result: dict[str, Any],
        config: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        all_results: dict[str, list[dict[str, Any]]] = pipeline_result.get("analysis", {})
        reports: dict[str, Any] = pipeline_result.get("reports", {})
        timing: dict[str, Any] = pipeline_result.get("timing", {})
        download: dict[str, Any] = pipeline_result.get("download", {})
        categories: list[str] = pipeline_result.get("categories", [])
        analysis_meta: dict[str, Any] = pipeline_result.get("analysis_meta", {})

        # 聚合分支数据
        branch_summary = self._aggregate_branches(all_results)

        # 构建组合计划
        plan = self._build_plan(all_results, config)
        portfolio_plan: dict[str, Any] = plan.get("portfolio_plan", {})
        recommendations: list[dict[str, Any]] = plan.get("recommendations", [])
        market_summary: dict[str, Any] = plan.get("market_summary", {})

        # 获取 NarratorAgent 生成的报告（如有）
        report_bundle = reports.get("report_bundle")
        narrator_md: str = ""
        executive_summary: list[str] = []
        market_view: str = ""
        if report_bundle is not None:
            narrator_md = getattr(report_bundle, "markdown_report", "") or ""
            executive_summary = list(getattr(report_bundle, "executive_summary", []) or [])
            market_view = str(getattr(report_bundle, "market_view", "") or "")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_stocks = int(market_summary.get("total_stocks", 0))
        selected_count = int(portfolio_plan.get("selected_count", len(recommendations)))

        sections = [
            self._header(config, now_str, total_stocks, selected_count),
            self._history_context(history),
            self._section_data_overview(market_summary, download, categories, config),
            self._section_market_overview(branch_summary, executive_summary, market_view, portfolio_plan),
            self._section_analysis_process(timing, config),
            self._section_bayesian_decision(analysis_meta, config),
            self._section_run_context(analysis_meta, report_bundle),
            self._section_subagent_decisions(branch_summary),
            self._section_master_decisions(executive_summary, market_view, portfolio_plan, narrator_md),
            self._section_investment_recommendations(recommendations, config["market"]),
            self._section_positions_orders(recommendations, portfolio_plan, config),
            self._section_next_steps(history, config, recommendations, timing),
        ]
        return "\n\n---\n\n".join(s for s in sections if s.strip())

    # ── 聚合工具 ──────────────────────────────────────────────────────────────

    def _aggregate_branches(
        self, all_results: dict[str, list[dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        """跨批次聚合各分支得分与结论。"""
        try:
            from quant_investor.market.analyze import _aggregate_branch_summary
            return _aggregate_branch_summary(all_results)
        except Exception as exc:
            log.debug("branch 聚合回退: %s", exc)
            return {}

    def _build_plan(
        self, all_results: dict[str, list[dict[str, Any]]], config: dict[str, Any]
    ) -> dict[str, Any]:
        """构建全市场交易计划。"""
        try:
            from quant_investor.market.analyze import build_full_market_trade_plan
            return build_full_market_trade_plan(
                all_results,
                market=config["market"],
                total_capital=config["total_capital"],
                top_k=config["top_k"],
            )
        except Exception as exc:
            log.debug("trade plan 构建回退: %s", exc)
            return {"portfolio_plan": {}, "recommendations": [], "market_summary": {}}

    # ── 各章节 ───────────────────────────────────────────────────────────────

    def _header(
        self, config: dict, now_str: str, total_stocks: int, selected_count: int
    ) -> str:
        capital_str = f"{config['total_capital']:,.0f}"
        review_chain = " -> ".join(config.get("review_model_priority", []) or ["(系统默认)"])
        return (
            f"# 📊 myQuant 每日 A 股分析报告\n\n"
            f"**生成时间**: {now_str}  \n"
            f"**市场**: {config['market']}  \n"
            f"**执行主线**: 统一 DAG + Bayesian Pipeline  \n"
            f"**风险偏好**: {config['risk_level']}  \n"
            f"**总资金**: ¥{capital_str}  \n"
            f"**分析股票数**: {total_stocks}  \n"
            f"**精选标的数**: {selected_count}  \n"
            f"**Review 模型优先级**: {review_chain}  \n"
            f"**Master Agent reasoning**: {config.get('master_reasoning_effort', '') or '(系统默认)'}"
        )

    def _history_context(self, history: list[dict[str, Any]]) -> str:
        loader = HistoryLoader()
        context = loader.format_context_section(history)
        return f"## 📚 历史分析上下文（最近 5 个日期的策略记录）\n\n{context}"

    def _section_data_overview(
        self,
        market_summary: dict,
        download: dict,
        categories: list[str],
        config: dict,
    ) -> str:
        total_stocks = int(market_summary.get("total_stocks", 0))
        total_batches = int(market_summary.get("total_batches", 0))
        generated_at = market_summary.get("generated_at", "N/A")
        download_status = download.get("status", "unknown")
        download_reason = download.get("reason", "")

        cat_lines = []
        for cat_name, cat_data in market_summary.get("categories", {}).items():
            label = cat_data.get("category_name", cat_name)
            count = cat_data.get("stock_count", 0)
            candidate = cat_data.get("candidate_count", 0)
            avg_exp = _safe_float(cat_data.get("avg_target_exposure", 0))
            cat_lines.append(
                f"| {label} | {count} | {candidate} | {avg_exp:.1%} |"
            )

        cat_table = ""
        if cat_lines:
            cat_table = (
                "\n| 板块 | 分析股票数 | 候选标的数 | 平均目标仓位 |\n"
                "|------|-----------|-----------|------------|\n"
                + "\n".join(cat_lines)
            )

        completeness_note = ""
        if isinstance(download.get("completeness_after"), dict):
            blocking = download["completeness_after"].get("blocking_incomplete_count", 0)
            if blocking:
                completeness_note = f"\n\n> ⚠️ 数据完整性：{blocking} 只股票存在阻塞性缺口，已跳过。"

        return (
            f"## § 1 数据概览\n\n"
            f"- **数据生成时间**: {generated_at}\n"
            f"- **数据下载状态**: {download_status}（{download_reason}）\n"
            f"- **分析覆盖**: {total_stocks} 只股票，共 {total_batches} 个批次\n"
            f"- **分析板块**: {', '.join(categories) if categories else '全量'}\n"
            f"{cat_table}{completeness_note}"
        )

    def _section_market_overview(
        self,
        branch_summary: dict,
        executive_summary: list[str],
        market_view: str,
        portfolio_plan: dict,
    ) -> str:
        macro = branch_summary.get("macro", {})
        macro_score = _safe_float(macro.get("score", 0))
        macro_confidence = _safe_float(macro.get("confidence", 0))
        macro_conclusion = macro.get("conclusion", "宏观数据暂未汇总。")

        exec_lines = ""
        if executive_summary:
            exec_lines = "\n**执行摘要（NarratorAgent 三句话）：**\n" + "\n".join(
                f"> {line}" for line in executive_summary
            )

        mv_line = f"\n**市场判断**: {market_view}" if market_view else ""

        style_bias = portfolio_plan.get("style_bias", "均衡")
        target_exp = _safe_float(portfolio_plan.get("target_exposure", 0))
        reliability = _safe_float(portfolio_plan.get("reliability", 0))

        branch_scores_lines = []
        for branch_key, blabel in _BRANCH_LABEL_MAP.items():
            b = branch_summary.get(branch_key, {})
            score = _safe_float(b.get("score", 0))
            conf = _safe_float(b.get("confidence", 0))
            if b:
                bar = "█" * int(abs(score) * 10) if abs(score) > 0.05 else "─"
                sign = "+" if score >= 0 else ""
                branch_scores_lines.append(
                    f"| {blabel} | {sign}{score:.3f} | {bar} | {_confidence_label(conf)}({conf:.2f}) |"
                )

        branch_table = ""
        if branch_scores_lines:
            branch_table = (
                "\n**各分支得分概览：**\n\n"
                "| 分支 | 得分 | 方向 | 可信度 |\n"
                "|------|------|------|-------|\n"
                + "\n".join(branch_scores_lines)
            )

        return (
            f"## § 2 市场概览\n\n"
            f"- **宏观评分**: {macro_score:+.3f}（可信度: {_confidence_label(macro_confidence)}）\n"
            f"- **宏观结论**: {macro_conclusion}\n"
            f"- **组合风格偏向**: {style_bias}\n"
            f"- **建议总仓位**: {target_exp:.1%}\n"
            f"- **整体可信度**: {_confidence_label(reliability)}（{reliability:.2f}）\n"
            f"{branch_table}{exec_lines}{mv_line}"
        )

    def _section_analysis_process(self, timing: dict, config: dict) -> str:
        dl_secs = _safe_float(timing.get("download_seconds", 0))
        an_secs = _safe_float(timing.get("analysis_seconds", 0))
        total_secs = _safe_float(timing.get("total_seconds", 0))
        review_chain = " -> ".join(config.get("review_model_priority", []) or ["(系统默认)"])

        return (
            f"## § 3 分析过程\n\n"
            f"**时间消耗：**\n"
            f"- 数据下载/检查: {dl_secs:.1f}s\n"
            f"- 分析与报告生成: {an_secs:.1f}s（{an_secs/60:.1f} 分钟）\n"
            f"- 总耗时: {total_secs:.1f}s（{total_secs/60:.1f} 分钟）\n\n"
            f"**分析配置：**\n"
            f"- K线后端: `{config['kline_backend']}`\n"
            f"- Review 模型优先级: `{review_chain}`\n"
            f"- Master Agent reasoning: `{config.get('master_reasoning_effort', '') or '(系统默认)'}`\n"
            f"- Subagent 超时: {config['agent_timeout']}s\n"
            f"- Master Agent 超时: {config['master_timeout']}s\n"
            f"- Agent Layer 启用: {'是' if config['enable_agent_layer'] else '否'}\n\n"
            f"**分析层级（统一 DAG）:** GlobalContext → 确定性漏斗压缩"
            f"（{config.get('funnel_max_candidates', runtime_config.FUNNEL_MAX_CANDIDATES)} 候选） → "
            f"三分支（量化+基本面+宏观） → "
            f"Bayesian 后验决策 → Master Discussion（Top {config.get('bayesian_shortlist_size', 50)}） → "
            f"确定性控制链 → 组合构建 → 报告生成"
        )

    def _section_run_context(self, analysis_meta: dict[str, Any], report_bundle: Any) -> str:
        model_role_metadata = analysis_meta.get("model_role_metadata")
        execution_trace = analysis_meta.get("execution_trace")
        what_if_plan = analysis_meta.get("what_if_plan")
        if not any([model_role_metadata, execution_trace, what_if_plan]) and report_bundle is not None:
            model_role_metadata = getattr(report_bundle, "model_role_metadata", None)
            execution_trace = getattr(report_bundle, "execution_trace", None)
            what_if_plan = getattr(report_bundle, "what_if_plan", None)
        if not any([model_role_metadata, execution_trace, what_if_plan]):
            return "## § 4 模型角色与执行轨迹\n\n_本次运行未记录结构化角色元数据或执行轨迹。_"

        from quant_investor.reporting.conclusion_renderer import ConclusionRenderer

        rendered = ConclusionRenderer.render_run_context(
            model_role_metadata,
            execution_trace,
            what_if_plan,
        )
        return "## § 4 模型角色与执行轨迹\n\n" + "\n".join(rendered).strip()

    def _section_bayesian_decision(self, analysis_meta: dict[str, Any], config: dict[str, Any]) -> str:
        """§ 4.5 Bayesian 决策层摘要。"""
        record_count = int(analysis_meta.get("bayesian_record_count", 0))
        funnel_candidates = int(analysis_meta.get("funnel_candidates_count", 0))
        funnel_excluded = int(analysis_meta.get("funnel_excluded_count", 0))
        shortlist_symbols = list(analysis_meta.get("bayesian_shortlist_symbols", []))
        if not any([record_count, funnel_candidates, funnel_excluded, shortlist_symbols]):
            return ""

        lines = [
            "## § 4.5 Bayesian 决策层",
            "",
            f"- **漏斗压缩**: 全市场 → {funnel_candidates} 候选（排除 {funnel_excluded} 只）",
            f"- **后验排名**: {record_count} 只候选完成 Bayesian 后验计算",
            f"- **Master Discussion 入选**: {len(shortlist_symbols)} 只",
        ]
        if shortlist_symbols:
            display_limit = max(1, int(config.get("bayesian_shortlist_size", 50) or 50))
            visible_symbols = shortlist_symbols[:display_limit]
            suffix = "" if len(shortlist_symbols) <= display_limit else f"（仅展示前 {display_limit} 只）"
            lines.append(f"- **精选标的**{suffix}: {', '.join(visible_symbols)}")
        lines.append("")
        lines.append(
            "> Bayesian 后验 = 分层先验（市场/宏观/行业/交易性/数据质量）"
            " × 多分支似然（log-odds 更新）× 相关性折扣 × 降级惩罚 × 覆盖折扣"
        )
        return "\n".join(lines)

    def _section_subagent_decisions(self, branch_summary: dict) -> str:
        if not branch_summary:
            return "## § 5 Subagent 决策过程\n\n_本次运行未启用 Agent Layer 或数据不可用。_"

        branch_blocks = []
        for branch_key, blabel in _BRANCH_LABEL_MAP.items():
            b = branch_summary.get(branch_key)
            if not b:
                continue
            score = _safe_float(b.get("score", 0))
            conf = _safe_float(b.get("confidence", 0))
            conclusion = b.get("conclusion", "暂无结论。")
            support = b.get("support_drivers", [])
            drag = b.get("drag_drivers", [])
            risks = b.get("investment_risks", [])
            coverage = b.get("coverage_notes", [])

            sign = "+" if score >= 0 else ""
            support_text = "；".join(str(s) for s in support[:3]) if support else "暂无明显支撑项。"
            drag_text = "；".join(str(d) for d in drag[:3]) if drag else "暂无明显拖累项。"
            risks_text = "；".join(str(r) for r in risks[:3]) if risks else "无"
            coverage_text = "；".join(str(c) for c in coverage[:2]) if coverage else "覆盖完整。"

            branch_blocks.append(
                f"### {blabel}分支（Subagent 分析）\n\n"
                f"| 项目 | 值 |\n|------|----|\n"
                f"| 综合得分 | `{sign}{score:.4f}` |\n"
                f"| 可信度 | {_confidence_label(conf)}（{conf:.2f}） |\n"
                f"| 分支结论 | {conclusion} |\n"
                f"| 支撑因素 | {support_text} |\n"
                f"| 拖累因素 | {drag_text} |\n"
                f"| 投资风险提示 | {risks_text} |\n"
                f"| 数据覆盖说明 | {coverage_text} |"
            )

        body = "\n\n".join(branch_blocks) if branch_blocks else "_无分支数据。_"
        return f"## § 5 Subagent 决策过程、逻辑和依据\n\n{body}"

    def _section_master_decisions(
        self,
        executive_summary: list[str],
        market_view: str,
        portfolio_plan: dict,
        narrator_md: str,
    ) -> str:
        exec_block = ""
        if executive_summary:
            exec_block = "**IC 综合三句话结论：**\n\n" + "\n".join(
                f"> {i+1}. {line}" for i, line in enumerate(executive_summary)
            )

        mv_block = f"\n\n**市场综合判断：**\n\n> {market_view}" if market_view else ""

        plan_notes = portfolio_plan.get("execution_notes", [])
        notes_block = ""
        if plan_notes:
            notes_block = "\n\n**组合执行备注：**\n" + "\n".join(
                f"- {note}" for note in plan_notes
            )

        target_exp = _safe_float(portfolio_plan.get("target_exposure", 0))
        style = portfolio_plan.get("style_bias", "均衡")
        selected = int(portfolio_plan.get("selected_count", 0))
        planned = _safe_float(portfolio_plan.get("planned_investment", 0))
        cash = _safe_float(portfolio_plan.get("cash_reserve", 0))

        summary_block = (
            f"\n\n**Master Agent 组合决策：**\n\n"
            f"| 决策项 | 结果 |\n|--------|------|\n"
            f"| 建议总仓位 | {target_exp:.1%} |\n"
            f"| 组合风格 | {style} |\n"
            f"| 精选标的数 | {selected} 只 |\n"
            f"| 计划投入 | ¥{planned:,.0f} |\n"
            f"| 保留现金 | ¥{cash:,.0f} |"
        )

        narrator_section = ""
        if narrator_md:
            # 从 narrator 报告中提取关键段落（避免完整复制导致报告过长）
            lines = narrator_md.split("\n")
            relevant = []
            capture = False
            for line in lines:
                if any(kw in line for kw in ["## 市场", "## 执行", "## 组合", "## 宏观", "## 决策"]):
                    capture = True
                if capture:
                    relevant.append(line)
                if len(relevant) > 30:
                    break
            if relevant:
                narrator_section = (
                    "\n\n<details>\n<summary>📋 NarratorAgent 完整报告摘要（展开）</summary>\n\n"
                    + "\n".join(relevant[:30])
                    + "\n\n</details>"
                )

        body = (
            f"{exec_block}{mv_block}{summary_block}{notes_block}{narrator_section}"
        ).strip()

        return f"## § 6 Master Agent 决策过程、逻辑和依据\n\n{body}"

    def _section_investment_recommendations(
        self, recommendations: list[dict], market: str
    ) -> str:
        if not recommendations:
            return (
                "## § 7 最终投资建议\n\n"
                "_本次分析未产生满足买入条件的候选标的。_\n\n"
                "> 可能原因：市场整体偏弱、宏观压制、数据覆盖不足。建议维持观望，等待信号改善。"
            )

        rows = []
        for item in recommendations:
            symbol = item.get("symbol", "")
            name = str(item.get("company_name") or item.get("name") or "").strip()
            action = item.get("action", "观察")
            emoji = _ACTION_EMOJI.get(action, "⚪")
            conf = _safe_float(item.get("confidence", 0))
            cur_price = _safe_float(item.get("current_price", 0))
            entry_price = _safe_float(item.get("recommended_entry_price", cur_price))
            target_price = _safe_float(item.get("target_price", 0))
            stop_loss = _safe_float(item.get("stop_loss_price", 0))
            pos_count = int(item.get("branch_positive_count", 0))
            rank = item.get("rank", "-")

            support = item.get("support_drivers", [])
            drag = item.get("drag_drivers", [])
            support_str = "；".join(str(s) for s in support[:2]) if support else "-"
            drag_str = "；".join(str(d) for d in drag[:2]) if drag else "-"
            one_line = item.get("one_line_conclusion", "")

            rows.append(
                f"\n### {rank}. {emoji} {symbol} {name}  |  {action}\n\n"
                f"- **一句话结论**: {one_line or '详见驱动因素。'}\n"
                f"- **分支支持**: {pos_count}/3 路正向  |  可信度: {_confidence_label(conf)}（{conf:.2f}）\n"
                f"- **价格参数**: 现价 ¥{cur_price:.2f}  →  参考买点 ¥{entry_price:.2f}  |  目标价 ¥{target_price:.2f}  |  止损价 ¥{stop_loss:.2f}\n"
                f"- **支撑因素**: {support_str}\n"
                f"- **拖累/风险**: {drag_str}"
            )

        body = "\n".join(rows)
        return f"## § 7 最终投资建议\n\n共精选 **{len(recommendations)}** 只标的：\n{body}"

    def _section_positions_orders(
        self,
        recommendations: list[dict],
        portfolio_plan: dict,
        config: dict,
    ) -> str:
        total_capital = config["total_capital"]
        target_exp = _safe_float(portfolio_plan.get("target_exposure", 0))
        planned = _safe_float(portfolio_plan.get("planned_investment", 0))
        cash = _safe_float(portfolio_plan.get("cash_reserve", total_capital - planned))
        max_weight = _safe_float(portfolio_plan.get("max_single_weight", 0))

        header = (
            f"**资金分配概览：**\n\n"
            f"| 项目 | 金额 | 比例 |\n|------|------|------|\n"
            f"| 总资金 | ¥{total_capital:,.0f} | 100% |\n"
            f"| 计划投入 | ¥{planned:,.0f} | {target_exp:.1%} |\n"
            f"| 保留现金 | ¥{cash:,.0f} | {1-target_exp:.1%} |\n"
            f"| 单票上限 | - | {max_weight:.1%} |\n"
        )

        if not recommendations:
            return (
                f"## § 8 仓位和买卖指令\n\n{header}\n"
                "_当前无买入指令，建议全仓现金等待机会。_"
            )

        order_rows = []
        buy_orders = [r for r in recommendations if r.get("action") in ("买入", "轻仓试错")]
        watch_orders = [r for r in recommendations if r.get("action") not in ("买入", "轻仓试错")]

        if buy_orders:
            order_rows.append(
                "\n**📥 买入/建仓指令：**\n\n"
                "| 序 | 股票 | 操作 | 参考买点 | 数量(股) | 金额 | 权重 | 止损价 |\n"
                "|---|------|------|---------|---------|------|------|-------|\n"
            )
            for item in buy_orders:
                display_name = self._display_name(item)
                action = item.get("action", "观察")
                entry = _safe_float(item.get("recommended_entry_price") or item.get("current_price", 0))
                shares = int(item.get("portfolio_shares", 0))
                amount = _safe_float(item.get("portfolio_amount", 0))
                weight = _safe_float(item.get("portfolio_weight", 0))
                stop_loss = _safe_float(item.get("stop_loss_price", 0))
                rank = item.get("rank", "-")
                order_rows.append(
                    f"| {rank} | {display_name} | {action} | "
                    f"¥{entry:.2f} | {shares:,} | ¥{amount:,.0f} | {weight:.2%} | ¥{stop_loss:.2f} |\n"
                )

        if watch_orders:
            order_rows.append(
                "\n**👁️ 观察/持续跟踪（暂不执行）：**\n\n"
            )
            for item in watch_orders:
                display_name = self._display_name(item)
                cur_price = _safe_float(item.get("current_price", 0))
                target_price = _safe_float(item.get("target_price", 0))
                one_line = item.get("one_line_conclusion", "信号不足，继续观察。")
                order_rows.append(
                    f"- **{display_name}** — 现价 ¥{cur_price:.2f} | 目标 ¥{target_price:.2f} | {one_line}\n"
                )

        body = header + "".join(order_rows)
        return f"## § 8 仓位和买卖指令\n\n{body}"

    def _section_next_steps(
        self,
        history: list[dict],
        config: dict,
        recommendations: list[dict],
        timing: dict,
    ) -> str:
        schedule_time = config.get("schedule_time", "17:30")
        next_run = f"下次分析时间: 明日 {schedule_time}"

        prev_note = ""
        if history:
            last = history[0]
            last_date = str(last.get("date", "") or "")
            last_strategy = str(last.get("strategy", "") or "")
            prev_note = f"- 最近参考记录: {last_date} / {last_strategy}"

        buy_list = [
            f"`{self._display_name(r)}` ¥{_safe_float(r.get('recommended_entry_price') or r.get('current_price', 0)):.2f}"
            for r in recommendations
            if r.get("action") in ("买入", "轻仓试错")
        ]
        watch_list = [
            f"`{self._display_name(r)}`"
            for r in recommendations
            if r.get("action") not in ("买入", "轻仓试错")
        ]

        buy_text = "、".join(buy_list) if buy_list else "无"
        watch_text = "、".join(watch_list[:5]) if watch_list else "无"

        timing_note = ""
        total_secs = _safe_float(timing.get("total_seconds", 0))
        if total_secs > 0:
            timing_note = f"- 本次分析耗时: {total_secs/60:.1f} 分钟"

        return (
            f"## § 9 下一步计划\n\n"
            f"**执行待办：**\n"
            f"- 待建仓标的: {buy_text}\n"
            f"- 待观察标的: {watch_text}\n"
            f"- 所有买入订单请参考 § 7 的参考买点和止损价执行\n"
            f"- 建议分批建仓，单次不超过目标仓位的 50%\n\n"
            f"**数据与系统：**\n"
            f"- {next_run}\n"
            f"{prev_note}\n"
            f"{timing_note}\n"
            f"- 历史上下文来自 `results/strategy_records/{config['market']}` 最近 5 个日期的正式记录\n\n"
            f"**风险提示：**\n"
            f"- 本报告为系统自动生成，仅供参考，不构成投资建议\n"
            f"- 请结合实际市场情况和个人风险承受能力做出决策\n"
            f"- 严格执行止损纪律，控制单次亏损"
        )
