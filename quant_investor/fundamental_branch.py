#!/usr/bin/env python3
"""
V9 Fundamental Branch。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from quant_investor.branch_contracts import (
    BranchResult,
    CorporateDocumentSnapshot,
    ForecastSnapshot,
    FundamentalSnapshot,
    ManagementSnapshot,
    OwnershipSnapshot,
    UnifiedDataBundle,
)
from quant_investor.fundamental_components import (
    document_semantic_analyzer,
    financial_quality_analyzer,
    forecast_revision_analyzer,
    management_governance_analyzer,
    ownership_analyzer,
    valuation_analyzer,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class FundamentalBranch:
    """公司基本面研究分支。"""

    COMPONENT_WEIGHTS = {
        "financial_quality": 0.28,
        "forecast_revision": 0.18,
        "valuation": 0.18,
        "management_governance": 0.14,
        "ownership": 0.12,
        "document_semantics": 0.10,
    }
    MODULE_LABELS = {
        "financial_quality": "财务质量",
        "forecast_revision": "盈利预测",
        "valuation": "估值",
        "management_governance": "管理层治理",
        "ownership": "股东结构",
        "document_semantics": "文档语义",
    }

    def __init__(
        self,
        data_layer: Any,
        stock_pool: list[str],
        enable_document_semantics: bool = True,
    ) -> None:
        self.data_layer = data_layer
        self.stock_pool = stock_pool
        self.enable_document_semantics = enable_document_semantics

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        result: list[str] = []
        for item in items:
            if item and item not in result:
                result.append(item)
        return result

    def _module_status(self, module_name: str, snapshot: Any) -> str:
        if module_name == "document_semantics" and not self.enable_document_semantics:
            return "disabled_global"
        data_quality = dict(getattr(snapshot, "data_quality", {}) or {})
        missing_scope = str(data_quality.get("missing_scope", "")).strip().lower()
        if bool(data_quality.get("provider_missing")) or missing_scope == "global":
            return "disabled_global"
        if getattr(snapshot, "source", "") == "disabled":
            return "disabled_global"
        if bool(getattr(snapshot, "available", False)):
            return "available"
        return "missing_symbol"

    @staticmethod
    def _filter_investment_risks(component: Any, module_status: str) -> list[str]:
        if module_status != "available":
            return []
        return [str(item) for item in getattr(component, "risks", []) if str(item)]

    def _build_symbol_conclusion(
        self,
        symbol: str,
        available_modules: list[str],
        missing_modules: list[str],
        support_points: list[str],
        drag_points: list[str],
    ) -> str:
        available_text = "、".join(self.MODULE_LABELS[name] for name in available_modules[:3]) or "可用模块有限"
        missing_text = "、".join(self.MODULE_LABELS[name] for name in missing_modules[:2]) or "无"
        driver_text = support_points[0] if support_points else "当前主要依据已覆盖模块的中性结果。"
        if drag_points:
            return (
                f"{symbol} 的基本面判断主要由{available_text}驱动，"
                f"{driver_text}；未充分纳入评分的模块包括{missing_text}。"
            )
        return (
            f"{symbol} 的基本面判断主要由{available_text}驱动，"
            f"{driver_text}；未充分纳入评分的模块包括{missing_text}。"
        )

    def run(self, data_bundle: UnifiedDataBundle) -> BranchResult:
        symbol_scores: dict[str, float] = {}
        component_scores: dict[str, dict[str, float]] = {}
        expected_returns: dict[str, float] = {}
        symbol_quality: dict[str, dict[str, Any]] = {}
        symbol_conclusions: dict[str, str] = {}
        coverage_by_symbol: dict[str, float] = {}
        branch_risks: list[str] = []
        coverage_notes: list[str] = []
        diagnostic_notes: list[str] = []
        bull_points: dict[str, list[str]] = defaultdict(list)
        bear_points: dict[str, list[str]] = defaultdict(list)
        module_coverage: dict[str, dict[str, Any]] = {
            name: {
                "label": self.MODULE_LABELS[name],
                "weight": weight,
                "status": "active",
                "available_symbols": 0,
                "total_symbols": len(self.stock_pool),
                "missing_symbols": [],
                "excluded_from_denominator": False,
            }
            for name, weight in self.COMPONENT_WEIGHTS.items()
        }
        module_average_scores: dict[str, list[float]] = defaultdict(list)
        missing_modules_by_symbol: dict[str, list[str]] = {}

        for symbol in self.stock_pool:
            as_of = self._resolve_as_of(data_bundle, symbol)
            fundamental = self.data_layer.get_point_in_time_fundamental_snapshot(symbol, as_of)
            forecast = self.data_layer.get_earnings_forecast_snapshot(symbol, as_of)
            management = self.data_layer.get_management_snapshot(symbol, as_of)
            ownership = self.data_layer.get_ownership_snapshot(symbol, as_of)
            document = self._load_document_snapshot(symbol, as_of)
            snapshots = {
                "financial_quality": fundamental,
                "forecast_revision": forecast,
                "valuation": fundamental,
                "management_governance": management,
                "ownership": ownership,
                "document_semantics": document,
            }

            components = {
                "financial_quality": financial_quality_analyzer(fundamental),
                "forecast_revision": forecast_revision_analyzer(forecast),
                "valuation": valuation_analyzer(fundamental),
                "management_governance": management_governance_analyzer(management),
                "ownership": ownership_analyzer(ownership),
                "document_semantics": document_semantic_analyzer(document),
            }

            weighted_score = 0.0
            coverage_weight = 0.0
            total_weight = 0.0
            component_scores[symbol] = {}
            available_modules: list[str] = []
            missing_modules: list[str] = []
            support_points: list[str] = []
            drag_points: list[str] = []
            symbol_module_coverage: dict[str, Any] = {}

            for name, component in components.items():
                weight = self.COMPONENT_WEIGHTS[name]
                component_scores[symbol][name] = round(component.score, 4)
                status = self._module_status(name, snapshots[name])
                label = self.MODULE_LABELS[name]
                symbol_module_coverage[name] = {
                    "status": status,
                    "available": bool(component.available),
                    "weight": weight,
                }
                if status == "disabled_global":
                    module_coverage[name]["status"] = "disabled_global"
                    module_coverage[name]["excluded_from_denominator"] = True
                    module_coverage[name]["missing_symbols"].append(symbol)
                    missing_modules.append(name)
                    continue

                total_weight += weight
                if component.available:
                    coverage_weight += weight
                    weighted_score += component.score * weight
                    available_modules.append(name)
                    module_coverage[name]["available_symbols"] += 1
                    module_average_scores[name].append(float(component.score))
                    bull_points[symbol].extend(component.evidence[:2])
                    support_points.extend(
                        f"{label}: {str(item)}" for item in component.evidence[:2]
                    )
                    filtered_risks = self._filter_investment_risks(component, status)
                    bear_points[symbol].extend(filtered_risks[:2])
                    drag_points.extend(f"{label}: {str(item)}" for item in filtered_risks[:2])
                    branch_risks.extend(filtered_risks[:1])
                else:
                    module_coverage[name]["missing_symbols"].append(symbol)
                    missing_modules.append(name)

            normalized_score = _clamp(weighted_score / max(total_weight, 1e-8), -1.0, 1.0)
            coverage_ratio = coverage_weight / max(total_weight, 1e-8)
            coverage_by_symbol[symbol] = coverage_ratio
            missing_modules_by_symbol[symbol] = missing_modules

            symbol_scores[symbol] = normalized_score
            expected_returns[symbol] = _clamp(normalized_score * 0.09, -0.18, 0.18)
            support_points = self._dedupe(support_points)[:3]
            drag_points = self._dedupe(drag_points)[:3]
            conclusion = self._build_symbol_conclusion(
                symbol=symbol,
                available_modules=available_modules,
                missing_modules=missing_modules,
                support_points=support_points,
                drag_points=drag_points,
            )
            symbol_conclusions[symbol] = conclusion
            symbol_quality[symbol] = {
                "as_of": as_of,
                "coverage_ratio": round(coverage_ratio, 4),
                "available_modules": available_modules,
                "missing_modules": missing_modules,
                "documents_available": bool(document.available),
                "forecast_available": bool(forecast.available),
                "module_coverage": symbol_module_coverage,
                "conclusion": conclusion,
            }

        explanation = (
            "Fundamental 分支聚合财务质量、盈利预测、估值、管理层、股东结构和离线文档语义。"
        )
        if not self.enable_document_semantics:
            explanation += " 文档语义当前被关闭，自动回退为中性组件。"

        avg_coverage = _safe_mean(list(coverage_by_symbol.values()))
        confidence = _clamp(0.38 + avg_coverage * 0.42, 0.35, 0.82)
        reliability = _clamp(0.40 + avg_coverage * 0.45, 0.35, 0.85)
        deduped_risks = list(dict.fromkeys(branch_risks))
        for name, info in module_coverage.items():
            label = str(info["label"])
            available_count = int(info["available_symbols"])
            total_symbols = int(info["total_symbols"])
            info["coverage_ratio"] = round(available_count / max(total_symbols, 1), 4)
            if info["status"] == "disabled_global":
                coverage_notes.append(
                    f"{label} 全局不可用，已从评分分母剔除（0/{total_symbols} 标的）。"
                )
            elif available_count < total_symbols:
                coverage_notes.append(
                    f"{label} 当前覆盖 {available_count}/{total_symbols} 标的，缺失部分仅计入覆盖说明。"
                )

        module_score_rank = sorted(
            (
                (
                    name,
                    _safe_mean(scores),
                    module_coverage[name]["status"],
                )
                for name, scores in module_average_scores.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        support_drivers = [
            f"{self.MODULE_LABELS[name]}对本轮判断形成主要支撑。"
            for name, score, status in module_score_rank
            if status == "active" and score > 0.05
        ][:3]
        drag_drivers = [
            f"{self.MODULE_LABELS[name]}对本轮判断形成主要拖累。"
            for name, score, status in sorted(module_score_rank, key=lambda item: item[1])
            if status == "active" and score < -0.05
        ][:3]
        active_modules = [
            self.MODULE_LABELS[name]
            for name, info in module_coverage.items()
            if info["status"] == "active" and info["available_symbols"] > 0
        ]
        excluded_modules = [
            self.MODULE_LABELS[name]
            for name, info in module_coverage.items()
            if info["excluded_from_denominator"]
        ]
        conclusion = (
            "本次基本面结论主要由"
            + ("、".join(active_modules[:3]) if active_modules else "可用模块有限")
            + "驱动，未纳入评分的模块包括"
            + ("、".join(excluded_modules) if excluded_modules else "无")
            + "。"
        )

        return BranchResult(
            branch_name="fundamental",
            score=_safe_mean(list(symbol_scores.values())),
            confidence=confidence,
            signals={
                "component_scores": component_scores,
                "quality_breakdown": symbol_quality,
                "bull_case": dict(bull_points),
                "bear_case": dict(bear_points),
                "expected_return": expected_returns,
                "symbol_conclusions": symbol_conclusions,
            },
            risks=deduped_risks[:8],
            explanation=explanation,
            symbol_scores=symbol_scores,
            metadata={
                "branch_mode": "fundamental_snapshot_fusion",
                "reliability": reliability,
                "horizon_days": 30,
                "documents_enabled": self.enable_document_semantics,
            },
            data_quality={
                "coverage_ratio": round(avg_coverage, 4),
                "documents_enabled": self.enable_document_semantics,
                "documents_missing_symbols": [
                    symbol
                    for symbol, quality in symbol_quality.items()
                    if not quality["documents_available"]
                ],
                "missing_modules": missing_modules_by_symbol,
            },
            conclusion=conclusion,
            thesis_points=self._dedupe(list(symbol_conclusions.values()))[:3],
            investment_risks=deduped_risks[:8],
            coverage_notes=self._dedupe(coverage_notes),
            diagnostic_notes=self._dedupe(diagnostic_notes),
            support_drivers=self._dedupe(support_drivers)[:3],
            drag_drivers=self._dedupe(drag_drivers)[:3],
            module_coverage=module_coverage,
        )

    @staticmethod
    def _resolve_as_of(data_bundle: UnifiedDataBundle, symbol: str) -> str:
        df = data_bundle.symbol_data.get(symbol, pd.DataFrame())
        if df is not None and not df.empty and "date" in df.columns:
            return pd.to_datetime(df["date"].max()).strftime("%Y-%m-%d")
        return str(data_bundle.metadata.get("end_date", ""))

    def _load_document_snapshot(self, symbol: str, as_of: str) -> CorporateDocumentSnapshot:
        if not self.enable_document_semantics:
            return CorporateDocumentSnapshot(
                symbol=symbol,
                as_of=as_of,
                available=False,
                source="disabled",
                notes=["document_semantics_disabled"],
            )
        return self.data_layer.get_document_semantic_snapshot(symbol, as_of)
