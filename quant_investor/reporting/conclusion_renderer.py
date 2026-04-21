"""
报告层 markdown 渲染器。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_investor.agent_protocol import (
    BranchOverlayVerdict,
    BranchVerdict,
    ExecutionTrace,
    ICDecision,
    ModelRoleMetadata,
    MasterICHint,
    PortfolioPlan,
    WhatIfPlan,
    StockReviewBundle,
)
from quant_investor.reporting.action_consistency_guard import ActionConsistencyGuard
from quant_investor.reporting.diagnostics_bucketizer import dedupe_texts, sanitize_report_text
from quant_investor.reporting.executive_summary import confidence_label

BRANCH_LABELS = {
    "kline": "K线",
    "quant": "量化",
    "fundamental": "基本面",
    "intelligence": "智能融合",
    "macro": "宏观",
}


class ConclusionRenderer:
    """渲染市场观点、分支结论、股票卡片和 markdown。"""

    @classmethod
    def render_market_view(
        cls,
        macro_verdict: BranchVerdict,
        ic_decisions: Sequence[ICDecision],
        portfolio_plan: PortfolioPlan,
    ) -> str:
        regime = str(macro_verdict.metadata.get("regime", "neutral"))
        style_bias = str(macro_verdict.metadata.get("style_bias", "balanced"))
        target_gross = float(portfolio_plan.target_gross_exposure)
        ic_actions = [decision.action for decision in ic_decisions]
        guard = ActionConsistencyGuard.guard(
            action=ic_decisions[0].action if ic_decisions else "watch",
            conclusion=(
                f"当前宏观环境处于 {regime}，组合应在 {target_gross:.1%} 总暴露附近保持"
                f" {style_bias} 风格。"
            ),
            ic_actions=ic_actions,
            risk_action_cap=portfolio_plan.metadata.get("action_cap")
            or portfolio_plan.metadata.get("risk_action_cap"),
            subject="当前市场",
        )
        return guard["conclusion"]

    @classmethod
    def render_branch_conclusions(
        cls,
        branch_summaries: Mapping[str, BranchVerdict | Mapping[str, Any]],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for branch_name in sorted(branch_summaries):
            branch = branch_summaries[branch_name]
            if isinstance(branch, BranchVerdict):
                thesis = branch.thesis
                confidence = float(branch.final_confidence)
                risk_candidates = dedupe_texts(
                    sanitize_report_text(item) for item in branch.investment_risks
                )
                coverage_candidates = dedupe_texts(
                    sanitize_report_text(item) for item in branch.coverage_notes
                )
            else:
                thesis = str(branch.get("thesis") or branch.get("conclusion") or "").strip()
                confidence = float(branch.get("final_confidence", branch.get("confidence", 0.0)))
                risk_candidates = dedupe_texts(
                    sanitize_report_text(item)
                    for item in branch.get("investment_risks", branch.get("risks", []))
                )
                coverage_candidates = dedupe_texts(
                    sanitize_report_text(item)
                    for item in branch.get("coverage_notes", [])
                )
            label = BRANCH_LABELS.get(branch_name, branch_name)
            risks = [
                item for item in risk_candidates
                if "数据覆盖摘要" not in item and "缺少覆盖" not in item and "数据接口本轮不可用" not in item
            ][:2]
            coverage_hints = [
                item for item in (risk_candidates + coverage_candidates)
                if "数据覆盖摘要" in item or "缺少覆盖" in item or "数据接口本轮不可用" in item
            ][:2]
            risk_tail = f"；投资风险包括 {'；'.join(risks)}" if risks else ""
            coverage_tail = f"；数据覆盖提示包括 {'；'.join(coverage_hints)}" if coverage_hints else ""
            result[branch_name] = (
                f"{label}分支结论：{sanitize_report_text(thesis)}"
                f"；可信度为{confidence_label(confidence)}{risk_tail}{coverage_tail}。"
            )
        return result

    @classmethod
    def render_stock_cards(
        cls,
        ic_decisions: Sequence[ICDecision],
        portfolio_plan: PortfolioPlan,
    ) -> list[dict[str, Any]]:
        symbol_meta = cls._collect_symbol_meta(ic_decisions)
        cards: list[dict[str, Any]] = []
        for symbol, weight in sorted(
            portfolio_plan.target_positions.items(),
            key=lambda item: (-float(item[1]), item[0]),
        ):
            meta = symbol_meta.get(symbol, {})
            company_name = str(meta.get("company_name", "")).strip()
            display_symbol = f"{symbol} {company_name}".strip() if company_name else symbol
            ic_actions = meta.get("ic_actions", [meta.get("action", "buy")])
            raw_action = meta.get("action", "buy" if weight > 0 else "watch")
            default_conclusion = (
                f"{symbol} 当前进入目标仓位 {float(weight):.1%}，按纪律分批执行。"
                if float(weight) >= 0.08
                else f"{symbol} 当前进入观察后的轻仓配置区间，需继续验证。"
            )
            guard = ActionConsistencyGuard.guard(
                action=raw_action,
                conclusion=str(meta.get("one_line_conclusion") or default_conclusion),
                ic_actions=ic_actions,
                risk_action_cap=portfolio_plan.metadata.get("action_cap")
                or portfolio_plan.metadata.get("risk_action_cap"),
                subject=symbol,
            )
            cards.append(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "display_symbol": display_symbol,
                    "target_weight": float(weight),
                    "display_action": guard["display_action"],
                    "one_line_conclusion": guard["conclusion"],
                    "confidence": float(meta.get("confidence", 0.0)),
                }
            )
        return cards

    @classmethod
    def render_markdown(
        cls,
        executive_summary: Sequence[str],
        market_view: str,
        branch_conclusions: Mapping[str, str],
        stock_cards: Sequence[Mapping[str, Any]],
        coverage_summary: Sequence[str],
        appendix_diagnostics: Sequence[str],
    ) -> str:
        lines = [
            "# 投资研究执行报告",
            "",
            "## 三句话执行摘要",
        ]
        lines.extend(f"- {line}" for line in list(executive_summary)[:3])
        lines.extend(
            [
                "",
                "## 市场观点",
                f"- {market_view}",
                "",
                "## 分支结论",
            ]
        )

        for branch_name in ["kline", "quant", "fundamental", "intelligence", "macro"]:
            if branch_name not in branch_conclusions:
                continue
            lines.extend(
                [
                    f"### {BRANCH_LABELS.get(branch_name, branch_name)}分支",
                    f"- 结论: {branch_conclusions[branch_name]}",
                    "",
                ]
            )

        lines.append("## 推荐标的卡片")
        if stock_cards:
            for card in stock_cards:
                display_symbol = str(card.get("display_symbol") or card.get("symbol") or "").strip()
                lines.extend(
                    [
                        f"### {display_symbol}",
                        f"- 一句话结论: {card['one_line_conclusion']}",
                        f"- 展示动作: {card['display_action']}",
                        f"- 目标权重: {float(card['target_weight']):.1%}",
                        "",
                    ]
                )
        else:
            lines.extend(["- 当前没有进入目标仓位的标的。", ""])

        lines.append("## 数据覆盖摘要")
        lines.extend(f"- {line}" for line in coverage_summary)
        lines.extend(["", "## 附录：工程诊断"])
        lines.extend(f"- {line}" for line in appendix_diagnostics)
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def render_review_sections(
        cls,
        review_bundle: StockReviewBundle | None,
    ) -> list[str]:
        if review_bundle is None:
            return []

        lines = ["## LLM 复核层"]
        if review_bundle.fallback_reasons:
            lines.extend(f"- 降级原因: {reason}" for reason in review_bundle.fallback_reasons[:5])
        if review_bundle.macro_verdict is not None:
            lines.append(
                f"- 宏观参考: {sanitize_report_text(review_bundle.macro_verdict.thesis)}"
            )
        if review_bundle.risk_decision is not None:
            lines.append(
                f"- 风控摘要: {sanitize_report_text('；'.join(review_bundle.risk_decision.reasons[:3]) or '无')}"
            )

        branch_overlay_lines = cls._render_branch_overlay_sections(
            review_bundle.branch_overlay_verdicts_by_symbol
        )
        if branch_overlay_lines:
            lines.extend(branch_overlay_lines)

        master_hint_lines = cls._render_master_hint_sections(review_bundle.master_hints_by_symbol)
        if master_hint_lines:
            lines.extend(master_hint_lines)

        if review_bundle.ic_hints_by_symbol:
            lines.append("### IC Hints")
            for symbol in sorted(review_bundle.ic_hints_by_symbol):
                hint = review_bundle.ic_hints_by_symbol[symbol]
                lines.append(
                    f"- {symbol}: action={hint.get('action', 'hold')}, "
                    f"score={float(hint.get('score', 0.0)):.2f}, "
                    f"confidence={float(hint.get('confidence', 0.0)):.2f}"
                )
        lines.append("")
        return lines

    @classmethod
    def render_bayesian_section(
        cls,
        bayesian_records: list[Any] | None,
        funnel_summary: Mapping[str, Any] | None = None,
        symbol_name_map: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Render Bayesian decision breakdown for the report."""
        if not bayesian_records:
            return []
        name_map = dict(symbol_name_map or {})
        lines = ["## Bayesian 决策分解"]
        if funnel_summary:
            lines.append(
                f"- 漏斗压缩: {funnel_summary.get('compression_ratio', 'n/a')}"
            )
        lines.append(f"- 排名标的数: {len(bayesian_records)}")
        lines.append("")
        lines.append("| # | 标的 | 公司 | Action Score | Win Rate | Confidence | Edge After Costs | Capacity Penalty |")
        lines.append("|---|------|------|-------------|----------|------------|------------------|------------------|")
        for record in bayesian_records[:20]:
            symbol = getattr(record, "symbol", str(record.get("symbol", ""))) if isinstance(record, dict) else getattr(record, "symbol", "")
            company = getattr(record, "company_name", "") or name_map.get(symbol, "")
            action_score = float(getattr(record, "posterior_action_score", 0.0)) if not isinstance(record, dict) else float(record.get("posterior_action_score", 0.0))
            win_rate = float(getattr(record, "posterior_win_rate", 0.0)) if not isinstance(record, dict) else float(record.get("posterior_win_rate", 0.0))
            confidence = float(getattr(record, "posterior_confidence", 0.0)) if not isinstance(record, dict) else float(record.get("posterior_confidence", 0.0))
            edge_after_costs = float(getattr(record, "posterior_edge_after_costs", 0.0)) if not isinstance(record, dict) else float(record.get("posterior_edge_after_costs", 0.0))
            capacity_penalty = float(getattr(record, "posterior_capacity_penalty", 0.0)) if not isinstance(record, dict) else float(record.get("posterior_capacity_penalty", 0.0))
            rank = getattr(record, "rank", 0) if not isinstance(record, dict) else record.get("rank", 0)
            lines.append(
                f"| {rank} | {symbol} | {company} | {action_score:.3f} | {win_rate:.3f} | {confidence:.3f} | {edge_after_costs:.3f} | {capacity_penalty:.3f} |"
            )
        lines.append("")
        return lines

    @classmethod
    def render_model_role_metadata(
        cls,
        model_role_metadata: ModelRoleMetadata | Mapping[str, Any] | None,
    ) -> list[str]:
        if model_role_metadata is None:
            return []
        payload = cls._coerce_mapping(model_role_metadata)
        if not payload:
            return []
        lines = ["## 模型角色元数据"]
        lines.append(f"- 分支模型: {payload.get('branch_model', 'n/a')}")
        if payload.get("resolved_branch_model"):
            lines.append(f"- 分支模型（实际运行）: {payload.get('resolved_branch_model')}")
        if payload.get("agent_fallback_model"):
            lines.append(f"- 分支 fallback: {payload.get('agent_fallback_model')}")
        if payload.get("branch_fallback_used"):
            lines.append(f"- 分支 fallback 已启用: {payload.get('branch_fallback_reason', 'n/a')}")
        lines.append(f"- Master 模型: {payload.get('master_model', 'n/a')}")
        if payload.get("resolved_master_model"):
            lines.append(f"- Master 模型（实际运行）: {payload.get('resolved_master_model')}")
        if payload.get("master_fallback_model"):
            lines.append(f"- Master fallback: {payload.get('master_fallback_model')}")
        if payload.get("master_fallback_used"):
            lines.append(f"- Master fallback 已启用: {payload.get('master_fallback_reason', 'n/a')}")
        lines.append(
            f"- Master reasoning: {payload.get('master_reasoning_effort', 'n/a')}"
        )
        lines.append(
            f"- 分支角色: {payload.get('branch_role', 'per-stock analysis')}"
        )
        lines.append(
            f"- Master 角色: {payload.get('master_role', 'master synthesis / portfolio-level judgment before deterministic risk and sizing')}"
        )
        lines.append(
            f"- 分支 Provider: {payload.get('branch_provider', 'n/a')} | 超时: {float(payload.get('branch_timeout', 0.0)):.1f}s"
        )
        lines.append(
            f"- Master Provider: {payload.get('master_provider', 'n/a')} | 超时: {float(payload.get('master_timeout', 0.0)):.1f}s"
        )
        lines.append(
            f"- Agent Layer: {'启用' if payload.get('agent_layer_enabled') else '禁用'}"
        )
        if payload.get("universe_key"):
            lines.append(
                f"- Universe: {payload.get('universe_key')} | 规模: {int(payload.get('universe_size', 0))} | hash: {payload.get('universe_hash', 'n/a')}"
            )
        lines.append("")
        return lines

    @classmethod
    def render_execution_trace(
        cls,
        execution_trace: ExecutionTrace | Mapping[str, Any] | None,
    ) -> list[str]:
        if execution_trace is None:
            return []
        payload = cls._coerce_mapping(execution_trace)
        if not payload:
            return []
        steps = list(payload.get("steps", []) or [])
        lines = ["## 执行 Trace"]
        if payload.get("key_parameters"):
            key_params = cls._coerce_mapping(payload.get("key_parameters"))
            if key_params:
                lines.append(
                    "- 关键参数: "
                    + ", ".join(f"{key}={value}" for key, value in key_params.items())
                )
        if payload.get("resolver_directory_priority"):
            lines.append(
                "- Resolver priority: "
                + " > ".join(str(item) for item in payload.get("resolver_directory_priority", []) if str(item).strip())
            )
        if payload.get("physical_directories_used_for_full_a"):
            lines.append(
                "- full_a 物理目录: "
                + ", ".join(
                    str(item)
                    for item in payload.get("physical_directories_used_for_full_a", [])
                    if str(item).strip()
                )
            )
        if payload.get("resolution_strategy"):
            lines.append(f"- Resolver strategy: {payload.get('resolution_strategy')}")
        if payload.get("local_union_fallback_used") is not None:
            lines.append(
                f"- Local union fallback: {'是' if payload.get('local_union_fallback_used') else '否'}"
            )
        metadata = cls._coerce_mapping(payload.get("metadata"))
        if metadata.get("data_quality_issue_count") is not None:
            lines.append(
                f"- Data quality issues: {int(metadata.get('data_quality_issue_count', 0))}"
            )
        for step in steps:
            step_payload = cls._coerce_mapping(step)
            if not step_payload:
                continue
            fallback = str(step_payload.get("fallback_reason", "")).strip()
            timeout = float(step_payload.get("timeout_seconds", 0.0) or 0.0)
            conclusion = sanitize_report_text(str(step_payload.get("conclusion", "") or ""))
            lines.append(
                f"- {step_payload.get('stage', 'step')}: "
                f"{step_payload.get('role', 'n/a')} / {step_payload.get('model', 'n/a')} | "
                f"{'成功' if step_payload.get('success', True) else '降级'} | "
                f"{conclusion}"
            )
            if timeout > 0:
                lines.append(f"  - 超时预算: {timeout:.1f}s")
            if fallback:
                lines.append(f"  - 回退原因: {sanitize_report_text(fallback)}")
        final_outcome = cls._coerce_mapping(payload.get("final_deterministic_outcome"))
        if final_outcome:
            lines.append(
                "- 最终 deterministic outcome: "
                + ", ".join(f"{key}={value}" for key, value in final_outcome.items())
            )
        resolved_paths = cls._coerce_mapping(payload.get("resolved_file_paths_by_symbol"))
        if resolved_paths:
            preview = list(resolved_paths.items())[:5]
            lines.append(
                "- Resolved file paths: "
                + ", ".join(f"{symbol}=>{path}" for symbol, path in preview if str(path).strip())
                + (f" ... 共{len(resolved_paths)}只" if len(resolved_paths) > len(preview) else "")
            )
        lines.append("")
        return lines

    @classmethod
    def render_what_if_plan(
        cls,
        what_if_plan: WhatIfPlan | Mapping[str, Any] | None,
    ) -> list[str]:
        if what_if_plan is None:
            return []
        payload = cls._coerce_mapping(what_if_plan)
        if not payload:
            return []
        scenarios = list(payload.get("scenarios", []) or [])
        lines = ["## What-If 响应计划"]
        if payload.get("metadata"):
            meta = cls._coerce_mapping(payload.get("metadata"))
            if meta:
                lines.append(
                    "- 结构化元数据: "
                    + ", ".join(f"{key}={value}" for key, value in meta.items())
                )
        for scenario in scenarios:
            scenario_payload = cls._coerce_mapping(scenario)
            if not scenario_payload:
                continue
            indicators = scenario_payload.get("monitoring_indicators", [])
            indicator_text = "；".join(str(item) for item in indicators[:4]) if indicators else "无"
            lines.extend(
                [
                    f"### {scenario_payload.get('scenario_name', 'scenario')}",
                    f"- Trigger: {scenario_payload.get('trigger', '')}",
                    f"- Monitoring indicators: {indicator_text}",
                    f"- Action: {scenario_payload.get('action', '')}",
                    f"- Position adjustment rule: {scenario_payload.get('position_adjustment_rule', '')}",
                    f"- Rerun full-market daily path: {'是' if scenario_payload.get('rerun_full_market_daily_path') else '否'}",
                ]
            )
            if scenario_payload.get("metadata"):
                meta = cls._coerce_mapping(scenario_payload.get("metadata"))
                if meta:
                    lines.append(
                        "- Scenario metadata: "
                        + ", ".join(f"{key}={value}" for key, value in meta.items())
                    )
            lines.append("")
        return lines

    @classmethod
    def render_run_context(
        cls,
        model_role_metadata: ModelRoleMetadata | Mapping[str, Any] | None,
        execution_trace: ExecutionTrace | Mapping[str, Any] | None,
        what_if_plan: WhatIfPlan | Mapping[str, Any] | None,
    ) -> list[str]:
        lines: list[str] = []
        lines.extend(cls.render_model_role_metadata(model_role_metadata))
        lines.extend(cls.render_execution_trace(execution_trace))
        lines.extend(cls.render_what_if_plan(what_if_plan))
        return lines

    @staticmethod
    def _collect_symbol_meta(ic_decisions: Sequence[ICDecision]) -> dict[str, dict[str, Any]]:
        metadata_by_symbol: dict[str, dict[str, Any]] = {}
        for decision in ic_decisions:
            metadata = decision.metadata if isinstance(decision.metadata, Mapping) else {}
            candidates = metadata.get("symbol_candidates", [])
            if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        continue
                    symbol = str(candidate.get("symbol", "")).strip()
                    if not symbol:
                        continue
                    bucket = metadata_by_symbol.setdefault(symbol, {"ic_actions": []})
                    action = candidate.get("action", decision.action.value)
                    bucket["action"] = action
                    company_name = str(candidate.get("company_name", "") or metadata.get("company_name", "")).strip()
                    if company_name:
                        bucket["company_name"] = company_name
                    bucket["confidence"] = float(candidate.get("confidence", decision.final_confidence))
                    bucket["one_line_conclusion"] = str(
                        candidate.get("one_line_conclusion")
                        or candidate.get("thesis")
                        or candidate.get("conclusion")
                        or ""
                    ).strip()
                    bucket["ic_actions"].append(action)
            symbol_actions = metadata.get("symbol_actions", {})
            symbol_confidences = metadata.get("symbol_confidences", {})
            symbol_conclusions = metadata.get("symbol_conclusions", {})
            if isinstance(symbol_actions, Mapping):
                for symbol, action in symbol_actions.items():
                    text = str(symbol).strip()
                    if not text:
                        continue
                    bucket = metadata_by_symbol.setdefault(text, {"ic_actions": []})
                    bucket["action"] = action
                    company_name = str(metadata.get("company_name", "") or "").strip()
                    if company_name:
                        bucket["company_name"] = company_name
                    if isinstance(symbol_confidences, Mapping):
                        bucket["confidence"] = float(
                            symbol_confidences.get(text, decision.final_confidence)
                        )
                    if isinstance(symbol_conclusions, Mapping):
                        bucket["one_line_conclusion"] = str(symbol_conclusions.get(text, "")).strip()
                    bucket["ic_actions"].append(action)
            for symbol in decision.selected_symbols:
                text = str(symbol).strip()
                if not text:
                    continue
                bucket = metadata_by_symbol.setdefault(text, {"ic_actions": []})
                bucket.setdefault("action", decision.action.value)
                bucket.setdefault("confidence", float(decision.final_confidence))
                bucket.setdefault("one_line_conclusion", "")
                company_name = str(metadata.get("company_name", "") or "").strip()
                if company_name:
                    bucket.setdefault("company_name", company_name)
                bucket["ic_actions"].append(bucket["action"])
        for symbol in metadata_by_symbol:
            metadata_by_symbol[symbol]["ic_actions"] = list(
                dedupe_texts(str(item) for item in metadata_by_symbol[symbol].get("ic_actions", []))
            )
        return metadata_by_symbol

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
        if hasattr(value, "to_dict"):
            payload = value.to_dict()
            if isinstance(payload, Mapping):
                return {str(key): item for key, item in payload.items()}
        if hasattr(value, "__dict__"):
            return {
                str(key): item
                for key, item in value.__dict__.items()
                if not str(key).startswith("_")
            }
        return {}

    @staticmethod
    def _render_branch_overlay_sections(
        branch_overlay_verdicts_by_symbol: Mapping[str, Mapping[str, BranchOverlayVerdict | Mapping[str, Any]]],
    ) -> list[str]:
        if not branch_overlay_verdicts_by_symbol:
            return []
        lines = ["### 分支叠加复核"]
        for symbol in sorted(branch_overlay_verdicts_by_symbol):
            overlays = branch_overlay_verdicts_by_symbol[symbol]
            lines.append(f"- {symbol}")
            for branch_name in sorted(overlays):
                overlay = overlays[branch_name]
                if isinstance(overlay, BranchOverlayVerdict):
                    thesis = overlay.thesis
                    score = overlay.adjusted_score
                    confidence = overlay.adjusted_confidence
                    fallback_reason = overlay.telemetry.fallback_reason
                else:
                    thesis = str(overlay.get("thesis") or "")
                    score = float(overlay.get("adjusted_score", overlay.get("base_score", 0.0)))
                    confidence = float(
                        overlay.get("adjusted_confidence", overlay.get("base_confidence", 0.0))
                    )
                    fallback_reason = str(overlay.get("telemetry", {}).get("fallback_reason", ""))
                line = (
                    f"  - {branch_name}: {sanitize_report_text(thesis)} "
                    f"(score={score:.2f}, confidence={confidence:.2f})"
                )
                if fallback_reason:
                    line += f" [fallback={sanitize_report_text(fallback_reason)}]"
                lines.append(line)
        return lines

    @staticmethod
    def _render_master_hint_sections(
        master_hints_by_symbol: Mapping[str, MasterICHint | Mapping[str, Any]],
    ) -> list[str]:
        if not master_hints_by_symbol:
            return []
        lines = ["### 单股 Master 判断"]
        for symbol in sorted(master_hints_by_symbol):
            hint = master_hints_by_symbol[symbol]
            if isinstance(hint, MasterICHint):
                thesis = hint.thesis
                score = hint.score_hint
                confidence = hint.confidence_hint
                risk_flags = hint.risk_flags
            else:
                thesis = str(hint.get("thesis") or "")
                score = float(hint.get("score_hint", hint.get("score", 0.0)))
                confidence = float(hint.get("confidence_hint", hint.get("confidence", 0.0)))
                risk_flags = list(hint.get("risk_flags", []))
            lines.append(
                f"- {symbol}: {sanitize_report_text(thesis)} "
                f"(score={score:.2f}, confidence={confidence:.2f})"
            )
            if risk_flags:
                lines.extend(f"  - 风险: {sanitize_report_text(item)}" for item in risk_flags[:3])
        return lines
