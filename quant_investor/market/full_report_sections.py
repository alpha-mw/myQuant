"""Report section renderers for full-market report output."""

from __future__ import annotations

from typing import Any

from quant_investor.market.full_report_helpers import (
    BRANCH_SUPPORT_DENOMINATOR,
    _branch_label,
    _canonical_branch_map,
    _confidence_label,
    _dedupe_text,
    _default_branch_conclusion,
    _derive_stock_drag_drivers,
    _derive_stock_support_drivers,
    _is_unknown_stock_name,
    _safe_average,
    _sanitize_text,
    get_stock_name,
)


class ExecutiveSummaryBuilder:
    """生成面向投资决策的三句话执行摘要。"""

    def __init__(
        self,
        portfolio_plan: dict[str, Any],
        branch_summary: dict[str, dict[str, Any]],
    ) -> None:
        self.portfolio_plan = portfolio_plan
        self.branch_summary = branch_summary

    def _macro_score(self) -> float:
        return float(self.branch_summary.get("macro", {}).get("score", 0.0))

    def _reliability(self) -> float:
        values = [
            float(branch.get("confidence", 0.0))
            for branch in self.branch_summary.values()
            if branch is not None
        ]
        return _safe_average(values, default=0.0)

    def build(self) -> list[str]:
        exposure = float(self.portfolio_plan.get("target_exposure", 0.0))
        style = str(self.portfolio_plan.get("style_bias", "均衡"))
        selected = int(self.portfolio_plan.get("selected_count", 0))
        macro_score = self._macro_score()
        reliability = self._reliability()
        return [
            f"当前建议总仓位维持在 {exposure:.1%}，组合风格偏{style}。",
            f"宏观评分 {macro_score:+.2f}，本轮最终纳入 {selected} 只标的进入执行清单。",
            f"整体可信度为{_confidence_label(reliability)}，当前更适合纪律化分批执行。",
        ]


class ActionConsistencyGuard:
    """统一校验动作、分支支持度和风险文案的一致性。"""

    MIN_CONFIDENCE = 0.42
    MACRO_PRESSURE_THRESHOLD = -0.25

    @classmethod
    def apply(cls, recommendation: dict[str, Any]) -> dict[str, Any]:
        payload = dict(recommendation)
        positive_count = int(payload.get("branch_positive_count", 0))
        confidence = float(payload.get("confidence", 0.0))
        macro_score = float(payload.get("macro_score", 0.0))
        target_exposure = float(
            payload.get(
                "batch_target_exposure", payload.get("portfolio_weight", 0.0)
            )
        )
        weak_support = positive_count <= 2
        low_confidence = confidence < cls.MIN_CONFIDENCE
        macro_pressure = (
            macro_score <= cls.MACRO_PRESSURE_THRESHOLD
            or target_exposure <= 0.20
        )

        if macro_pressure or (weak_support and low_confidence):
            action = "观察"
        elif weak_support or low_confidence:
            action = "轻仓试错"
        else:
            action = "买入"

        reasons = list(payload.get("weight_cap_reasons", []))
        if weak_support:
            reasons.append(
                f"v13 四分支支持仅 {positive_count}/"
                f"{BRANCH_SUPPORT_DENOMINATOR}，不宜激进。"
            )
        if low_confidence:
            reasons.append(f"综合可信度仅 {_confidence_label(confidence)}。")
        if macro_pressure:
            reasons.append("宏观分支当前显著压仓，动作已自动下调。")

        payload["raw_action"] = payload.get("action", "")
        payload["action"] = action
        payload["weight_cap_reasons"] = _dedupe_text(
            [_sanitize_text(item) for item in reasons if item]
        )
        return payload


def _aggregate_branch_summary(
    all_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    total_batches = sum(len(batches) for batches in all_results.values())
    for batches in all_results.values():
        for batch in batches:
            for name, branch in _canonical_branch_map(
                dict(batch.get("branches", {}))
            ).items():
                bucket = aggregated.setdefault(
                    name,
                    {
                        "score_values": [],
                        "confidence_values": [],
                        "conclusions": [],
                        "support_drivers": [],
                        "drag_drivers": [],
                        "investment_risks": [],
                        "coverage_notes": [],
                        "diagnostic_notes": [],
                        "module_coverage": {},
                        "debate_statuses": [],
                    },
                )
                bucket["score_values"].append(float(branch.get("score", 0.0)))
                bucket["confidence_values"].append(
                    float(branch.get("confidence", 0.0))
                )
                bucket["conclusions"].append(str(branch.get("conclusion", "")))
                bucket["support_drivers"].extend(
                    str(item) for item in branch.get("support_drivers", [])
                )
                bucket["drag_drivers"].extend(
                    str(item) for item in branch.get("drag_drivers", [])
                )
                bucket["investment_risks"].extend(
                    str(item)
                    for item in branch.get(
                        "investment_risks", branch.get("risks", [])
                    )
                )
                bucket["coverage_notes"].extend(
                    str(item) for item in branch.get("coverage_notes", [])
                )
                bucket["diagnostic_notes"].extend(
                    str(item) for item in branch.get("diagnostic_notes", [])
                )
                bucket["debate_statuses"].append(
                    str(branch.get("debate_status", "skipped"))
                )
                for module_name, info in branch.get(
                    "module_coverage", {}
                ).items():
                    module_bucket = bucket["module_coverage"].setdefault(
                        module_name,
                        {
                            "label": info.get("label", module_name),
                            "available_symbols": 0,
                            "total_symbols": 0,
                            "disabled_batches": 0,
                            "status": info.get("status", "active"),
                        },
                    )
                    module_bucket["available_symbols"] += int(
                        info.get("available_symbols", 0)
                    )
                    module_bucket["total_symbols"] += int(
                        info.get("total_symbols", 0)
                    )
                    if info.get("status") == "disabled_global":
                        module_bucket["disabled_batches"] += 1
                        module_bucket["status"] = "disabled_global"

    finalized: dict[str, dict[str, Any]] = {}
    for name, bucket in aggregated.items():
        module_notes = []
        for module_name, info in bucket["module_coverage"].items():
            label = str(info.get("label", module_name))
            available = int(info.get("available_symbols", 0))
            total = int(info.get("total_symbols", 0))
            if info.get("status") == "disabled_global":
                module_notes.append(
                    f"{label}: 0/{max(total, 1)} 标的可用，"
                    f"{int(info.get('disabled_batches', 0))}/"
                    f"{max(total_batches, 1)} 批次全局剔除。"
                )
            elif total > 0 and available < total:
                module_notes.append(
                    f"{label}: {available}/{total} 标的已覆盖。"
                )
        finalized[name] = {
            "score": _safe_average(bucket["score_values"], default=0.0),
            "confidence": _safe_average(
                bucket["confidence_values"], default=0.0
            ),
            "conclusion": next(
                (text for text in bucket["conclusions"] if str(text).strip()),
                _default_branch_conclusion(
                    name, _safe_average(bucket["score_values"], default=0.0)
                ),
            ),
            "support_drivers": _dedupe_text(
                [_sanitize_text(item) for item in bucket["support_drivers"]]
            )[:3],
            "drag_drivers": _dedupe_text(
                [_sanitize_text(item) for item in bucket["drag_drivers"]]
            )[:3],
            "investment_risks": _dedupe_text(
                [_sanitize_text(item) for item in bucket["investment_risks"]]
            )[:5],
            "coverage_notes": _dedupe_text(
                [_sanitize_text(item) for item in bucket["coverage_notes"]]
                + module_notes
            )[:6],
            "diagnostic_notes": _dedupe_text(
                [_sanitize_text(item) for item in bucket["diagnostic_notes"]]
            )[:6],
            "module_coverage": bucket["module_coverage"],
            "debate_statuses": [
                status
                for status in _dedupe_text(bucket["debate_statuses"])
                if status and status != "unknown"
            ],
        }
    return finalized


class DiagnosticsBucketizer:
    """把投资风险、覆盖信息和工程诊断拆分到不同报告区域。"""

    def __init__(
        self,
        all_results: dict[str, list[dict[str, Any]]],
        branch_summary: dict[str, dict[str, Any]],
    ) -> None:
        self.all_results = all_results
        self.branch_summary = branch_summary

    def bucket(self) -> dict[str, list[str]]:
        total_batches = max(
            sum(len(batches) for batches in self.all_results.values()), 1
        )
        investment_risks: list[str] = []
        coverage_notes: list[str] = []
        diagnostic_notes: list[str] = []
        for branch_name, branch in self.branch_summary.items():
            investment_risks.extend(branch.get("investment_risks", []))
            coverage_notes.extend(branch.get("coverage_notes", []))
            diagnostic_notes.extend(
                f"{_sanitize_text(note)}（{total_batches}/{total_batches} 批次）"
                for note in branch.get("diagnostic_notes", [])
            )
        for batches in self.all_results.values():
            for batch in batches:
                execution_log = batch.get("execution_log", [])
                for line in execution_log[-5:]:
                    sanitized = _sanitize_text(str(line))
                    if sanitized and sanitized not in diagnostic_notes:
                        diagnostic_notes.append(
                            f"{sanitized}（1/{total_batches} 批次）"
                        )
        return {
            "investment_risks": _dedupe_text(investment_risks)[:8],
            "coverage_notes": _dedupe_text(coverage_notes)[:8],
            "diagnostic_notes": _dedupe_text(diagnostic_notes)[:12],
        }


class ConclusionRenderer:
    """渲染分支与个股结论。"""

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_dict"):
            payload = value.to_dict()
            if isinstance(payload, dict):
                return dict(payload)
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def render_branch(branch_name: str, branch: dict[str, Any]) -> list[str]:
        label = _branch_label(branch_name)
        conclusion = str(
            branch.get("conclusion")
            or _default_branch_conclusion(
                branch_name, float(branch.get("score", 0.0))
            )
        )
        support = _dedupe_text(branch.get("support_drivers", [])) or [
            "当前未观察到明显增量支撑。"
        ]
        drag = _dedupe_text(branch.get("drag_drivers", [])) or [
            "当前未观察到明显拖累项。"
        ]
        coverage = _dedupe_text(branch.get("coverage_notes", [])) or [
            "当前未发现显著覆盖缺口。"
        ]
        return [
            f"### {label}分支",
            f"- 平均得分: {branch_name}: {float(branch.get('score', 0.0)):+.3f}",
            f"- 结论: {conclusion}",
            f"- 主要驱动: {'；'.join(support[:2])}",
            f"- 主要拖累: {'；'.join(drag[:2])}",
            f"- 数据覆盖情况: {'；'.join(coverage[:2])}",
            (
                "- 可信度标签: "
                f"{_confidence_label(float(branch.get('confidence', 0.0)))}"
            ),
            "",
        ]

    @staticmethod
    def render_model_role_metadata(model_role_metadata: Any) -> list[str]:
        payload = ConclusionRenderer._coerce_mapping(model_role_metadata)
        if not payload:
            return ["- 当前未记录模型角色元数据。"]
        lines = ["- 模型角色元数据:"]
        for key in [
            "agent_model",
            "agent_fallback_model",
            "master_model",
            "master_fallback_model",
            "master_reasoning_effort",
        ]:
            if key in payload:
                lines.append(f"  - {key}: {payload[key]}")
        if payload.get("resolver_directory_priority"):
            lines.append(
                "  - resolver_directory_priority: "
                f"{payload['resolver_directory_priority']}"
            )
        if payload.get("physical_directories_used_for_full_a"):
            directories = ", ".join(
                str(item)
                for item in payload["physical_directories_used_for_full_a"]
            )
            lines.append(
                f"  - physical_directories_used_for_full_a: {directories}"
            )
        if payload.get("fallback_used") is not None:
            lines.append(f"  - fallback_used: {payload['fallback_used']}")
        return lines

    @staticmethod
    def render_execution_trace(execution_trace: Any) -> list[str]:
        payload = ConclusionRenderer._coerce_mapping(execution_trace)
        if not payload:
            return ["- 当前未记录执行轨迹。"]
        lines = ["- 执行轨迹:"]
        if payload.get("stage_summaries"):
            lines.append("  - stages:")
            for stage in payload.get("stage_summaries", [])[:8]:
                stage_payload = ConclusionRenderer._coerce_mapping(stage)
                if not stage_payload:
                    continue
                stage_name = stage_payload.get("stage_name", "stage")
                status = stage_payload.get("status", "")
                lines.append(f"    - {stage_name}: {status}")
        for key in [
            "resolver_directory_priority",
            "physical_directories_used_for_full_a",
            "local_union_fallback_used",
            "final_deterministic_outcome",
        ]:
            if key in payload:
                value = payload[key]
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                lines.append(f"  - {key}: {value}")
        return lines

    @staticmethod
    def render_what_if_plan(what_if_plan: Any) -> list[str]:
        payload = ConclusionRenderer._coerce_mapping(what_if_plan)
        scenarios = (
            payload.get("scenarios", [])
            if isinstance(payload.get("scenarios", []), list)
            else []
        )
        if not payload and not scenarios:
            return ["- 当前未记录 what-if 计划。"]
        lines = ["- what-if 计划:"]
        for scenario in scenarios[:6]:
            scenario_payload = ConclusionRenderer._coerce_mapping(scenario)
            if not scenario_payload:
                continue
            scenario_name = scenario_payload.get("scenario_name", "scenario")
            rerun_daily = (
                "是"
                if scenario_payload.get("rerun_full_market_daily_path")
                else "否"
            )
            position_rule = scenario_payload.get(
                "position_adjustment_rule", ""
            )
            lines.extend(
                [
                    f"  - 场景: {scenario_name}",
                    f"    - 触发: {scenario_payload.get('trigger', '')}",
                    f"    - 动作: {scenario_payload.get('action', '')}",
                    f"    - 仓位调整: {position_rule}",
                    f"    - 重新跑全市场: {rerun_daily}",
                ]
            )
        return lines

    @staticmethod
    def render_run_context(
        model_role_metadata: Any,
        execution_trace: Any,
        what_if_plan: Any,
    ) -> list[str]:
        lines: list[str] = []
        lines.extend(
            ConclusionRenderer.render_model_role_metadata(model_role_metadata)
        )
        lines.extend(
            ConclusionRenderer.render_execution_trace(execution_trace)
        )
        lines.extend(ConclusionRenderer.render_what_if_plan(what_if_plan))
        return lines

    @staticmethod
    def render_stock(item: dict[str, Any], market: str) -> list[str]:
        action = str(item.get("action", "观察"))
        stock_name = str(
            item.get("company_name")
            or item.get("name")
            or get_stock_name(item["symbol"], market=market)
        ).strip()
        if _is_unknown_stock_name(stock_name):
            stock_name = get_stock_name(item["symbol"], market=market)
        support = _dedupe_text(
            item.get("support_drivers", [])
            or _derive_stock_support_drivers(item)
        )
        drag = _dedupe_text(
            item.get("drag_drivers", []) or _derive_stock_drag_drivers(item)
        )
        weight_caps = _dedupe_text(item.get("weight_cap_reasons", []))
        if action == "买入":
            conclusion = (
                f"{item['symbol']} {stock_name} 当前获得 "
                f"{int(item.get('branch_positive_count', 0))}/"
                f"{BRANCH_SUPPORT_DENOMINATOR} "
                "个 v13 分支支持，可按计划分批执行。"
            )
        elif action == "轻仓试错":
            conclusion = (
                f"{item['symbol']} {stock_name} 当前仍有正向依据，"
                "但更适合轻仓试错。"
            )
        else:
            conclusion = (
                f"{item['symbol']} {stock_name} 当前信号不足以支撑激进执行，"
                "建议继续观察。"
            )
        one_line = str(item.get("one_line_conclusion", "") or "").strip()
        if action != "买入" and ("买" in one_line or "执行" in one_line):
            one_line = ""
        pressure_text = (
            "；".join((drag + weight_caps)[:3])
            if (drag or weight_caps)
            else "当前未见额外压仓理由。"
        )
        return [
            (
                f"### {item.get('rank', '-')}. {item['symbol']} "
                f"{stock_name} ({item['category_name']})"
            ),
            f"- 一句话结论: {one_line or conclusion}",
            f"- 入选原因: {'；'.join(support[:3])}",
            f"- 压仓原因: {pressure_text}",
            f"- 执行动作: {action}",
            (
                f"- 交易参数: 现价 {item['current_price']:.2f}，"
                f"参考买点 {item['recommended_entry_price']:.2f}，"
                f"目标价 {item['target_price']:.2f}，"
                f"止损价 {item['stop_loss_price']:.2f}"
            ),
            "",
        ]

__all__ = [
    "ActionConsistencyGuard",
    "ConclusionRenderer",
    "DiagnosticsBucketizer",
    "ExecutiveSummaryBuilder",
    "_aggregate_branch_summary",
]
