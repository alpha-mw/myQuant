"""Factor lifecycle governance for mined alpha factors.

This module is intentionally conservative: a factor can be mined and reviewed
without being eligible for stock selection.  The quant branch only consumes
``production_factor`` records that have passed all eight admission gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class FactorLifecycleState(str, Enum):
    """Lifecycle states for every mined factor."""

    DRAFT = "draft"
    RESEARCH_CANDIDATE = "research_candidate"
    SHADOW = "shadow"
    MATURE_CANDIDATE = "mature_candidate"
    PAPER_FACTOR = "paper_factor"
    PRODUCTION_CANDIDATE = "production_candidate"
    PRODUCTION_FACTOR = "production_factor"
    WATCH = "watch"
    REDUCED = "reduced"
    DEPRECATED = "deprecated"


class FactorAdmissionDecision(str, Enum):
    """Allowed factor-review decisions.

    The evaluator only creates candidates.  A production transition is owned
    exclusively by FactorGovernanceProtocol v3 and may run automatically only
    through its explicit, hash-bound month-end apply path.
    """

    REJECT = "reject"
    REVISE = "revise"
    WATCHLIST = "watchlist"
    PAPER_FACTOR = "paper_factor"
    PRODUCTION_CANDIDATE = "production_candidate"


@dataclass(frozen=True)
class GateSpec:
    gate_id: int
    key: str
    title: str


GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec(1, "data_safety", "Gate 1: 数据安全"),
    GateSpec(2, "coverage_stability", "Gate 2: 覆盖率和稳定性"),
    GateSpec(3, "ic_rankic", "Gate 3: IC / RankIC"),
    GateSpec(4, "group_returns", "Gate 4: 分组收益"),
    GateSpec(5, "cost_turnover", "Gate 5: 交易成本和换手"),
    GateSpec(6, "neutralization_exposure", "Gate 6: 中性化和暴露分析"),
    GateSpec(7, "oos_robustness", "Gate 7: 样本外和稳健性"),
    GateSpec(8, "portfolio_incremental", "Gate 8: 组合增量验证"),
)
GATE_BY_ID = {spec.gate_id: spec for spec in GATE_SPECS}
GATE_BY_KEY = {spec.key: spec for spec in GATE_SPECS}


@dataclass
class GateResult:
    gate_id: int
    gate_key: str
    title: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "gate_key": self.gate_key,
            "title": self.title,
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GateResult":
        gate_id = int(payload.get("gate_id", 0) or 0)
        spec = GATE_BY_ID.get(gate_id) or GATE_BY_KEY.get(str(payload.get("gate_key", "")))
        if spec is None:
            spec = GateSpec(
                gate_id,
                str(payload.get("gate_key", f"gate_{gate_id}")),
                str(payload.get("title", "")),
            )
        return cls(
            gate_id=spec.gate_id,
            gate_key=spec.key,
            title=str(payload.get("title") or spec.title),
            passed=bool(payload.get("passed", False)),
            reasons=[str(item) for item in payload.get("reasons", []) if str(item)],
            metrics=dict(payload.get("metrics", {}) or {}),
            severity=str(payload.get("severity", "info") or "info"),
        )


@dataclass
class FactorRecord:
    """Governed factor registry record.

    The runtime scorer treats missing gate evidence as non-selectable.  This
    prevents mined or paper factors from accidentally becoming live signals.
    """

    name: str
    version: str = "v1"
    state: FactorLifecycleState = FactorLifecycleState.DRAFT
    category: str = "custom"
    implementation: str = ""
    weight: float = 0.0
    direction: float = 1.0
    horizon_days: int = 5
    owner: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    gate_results: list[GateResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    admission_decision: FactorAdmissionDecision | None = None
    thematic: bool = False
    narrow_coverage: bool = False
    approved_by: str = ""
    approved_at: str = ""
    deprecated_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorRecord":
        state = FactorLifecycleState(str(payload.get("state", FactorLifecycleState.DRAFT.value)))
        decision_raw = payload.get("admission_decision")
        decision = FactorAdmissionDecision(str(decision_raw)) if decision_raw else None
        return cls(
            name=str(payload.get("name", "")).strip(),
            version=str(payload.get("version", "v1") or "v1"),
            state=state,
            category=str(payload.get("category", "custom") or "custom"),
            implementation=str(payload.get("implementation", "") or ""),
            weight=float(payload.get("weight", 0.0) or 0.0),
            direction=float(payload.get("direction", 1.0) or 1.0),
            horizon_days=int(payload.get("horizon_days", 5) or 5),
            owner=str(payload.get("owner", "") or ""),
            description=str(payload.get("description", "") or ""),
            tags=[str(item) for item in payload.get("tags", []) if str(item)],
            gate_results=[
                GateResult.from_dict(item)
                for item in payload.get("gate_results", [])
                if isinstance(item, Mapping)
            ],
            metrics=dict(payload.get("metrics", {}) or {}),
            admission_decision=decision,
            thematic=bool(payload.get("thematic", False)),
            narrow_coverage=bool(payload.get("narrow_coverage", False)),
            approved_by=str(payload.get("approved_by", "") or ""),
            approved_at=str(payload.get("approved_at", "") or ""),
            deprecated_reason=str(payload.get("deprecated_reason", "") or ""),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "state": self.state.value,
            "category": self.category,
            "implementation": self.implementation,
            "weight": float(self.weight),
            "direction": float(self.direction),
            "horizon_days": int(self.horizon_days),
            "owner": self.owner,
            "description": self.description,
            "tags": list(self.tags),
            "gate_results": [item.to_dict() for item in self.gate_results],
            "metrics": dict(self.metrics),
            "admission_decision": (
                self.admission_decision.value if self.admission_decision else None
            ),
            "thematic": bool(self.thematic),
            "narrow_coverage": bool(self.narrow_coverage),
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "deprecated_reason": self.deprecated_reason,
            "metadata": dict(self.metadata),
        }

    def gate_map(self) -> dict[int, GateResult]:
        return {int(item.gate_id): item for item in self.gate_results}

    def all_gates_passed(self) -> bool:
        gate_map = self.gate_map()
        return all(gate_id in gate_map and gate_map[gate_id].passed for gate_id in range(1, 9))

    def selectable_in_quant_branch(self) -> bool:
        return (
            self.state == FactorLifecycleState.PRODUCTION_FACTOR
            and self.all_gates_passed()
            and float(self.weight) != 0.0
            and not self.deprecated_reason
        )


@dataclass
class FactorReview:
    factor_name: str
    current_state: FactorLifecycleState
    target_state: FactorLifecycleState
    decision: FactorAdmissionDecision
    gate_results: list[GateResult]
    summary: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "current_state": self.current_state.value,
            "target_state": self.target_state.value,
            "decision": self.decision.value,
            "gate_results": [item.to_dict() for item in self.gate_results],
            "summary": self.summary,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


class FactorGovernancePolicy:
    """Default thresholds for factor admission.

    Thresholds encode the requested 8-gate workflow.  They are deliberately
    minimum viable production rules; stricter asset-class specific policies can
    subclass or mutate this object.
    """

    min_coverage_rate = 0.60
    max_nan_rate = 0.40
    thematic_min_coverage_rate = 0.30
    thematic_max_nan_rate = 0.70
    max_extreme_value_ratio = 0.03
    max_coverage_bucket_share = 0.80

    min_watch_icir = 0.30
    min_production_candidate_icir = 0.50
    min_positive_ic_ratio = 0.52
    min_production_positive_ic_ratio = 0.55
    max_single_year_ic_contribution = 0.60

    min_monotonicity = 0.35
    max_turnover = 12.0
    max_capacity_pressure = 0.75
    min_neutralized_icir = 0.20
    max_existing_factor_corr = 0.70
    min_oos_positive_ratio = 0.55
    max_turnover_delta = 0.30
    max_drawdown_delta = 0.02


class FactorGateEvaluator:
    """Evaluate one factor against the eight admission gates."""

    def __init__(self, policy: FactorGovernancePolicy | None = None) -> None:
        self.policy = policy or FactorGovernancePolicy()

    def evaluate(
        self,
        *,
        factor_name: str,
        metrics: Mapping[str, Any],
        current_state: FactorLifecycleState | str = FactorLifecycleState.DRAFT,
    ) -> FactorReview:
        state = (
            current_state
            if isinstance(current_state, FactorLifecycleState)
            else FactorLifecycleState(str(current_state))
        )
        m = dict(metrics or {})
        gate_results = [
            self._gate1_data_safety(m),
            self._gate2_coverage(m),
            self._gate3_ic(m),
            self._gate4_group_returns(m),
            self._gate5_costs(m),
            self._gate6_exposure(m),
            self._gate7_oos(m),
            self._gate8_incremental(m),
        ]
        decision, target_state, reasons = self._decide(state, gate_results, m)
        summary = self._summary(decision, target_state, gate_results, reasons)
        return FactorReview(
            factor_name=factor_name,
            current_state=state,
            target_state=target_state,
            decision=decision,
            gate_results=gate_results,
            summary=summary,
            reasons=reasons,
            metrics=m,
        )

    def _result(
        self,
        gate_id: int,
        passed: bool,
        reasons: list[str],
        metrics: Mapping[str, Any] | None = None,
        severity: str | None = None,
    ) -> GateResult:
        spec = GATE_BY_ID[gate_id]
        return GateResult(
            gate_id=spec.gate_id,
            gate_key=spec.key,
            title=spec.title,
            passed=bool(passed),
            reasons=reasons,
            metrics=dict(metrics or {}),
            severity=severity or ("info" if passed else "error"),
        )

    @staticmethod
    def _f(metrics: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
        value = metrics.get(key, default)
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except Exception:
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _strict_bool(metrics: Mapping[str, Any], key: str) -> bool | None:
        value = metrics.get(key)
        return value if isinstance(value, bool) else None

    @staticmethod
    def _missing(metrics: Mapping[str, Any], keys: list[str]) -> list[str]:
        return [key for key in keys if key not in metrics]

    @staticmethod
    def _is_sha256(value: Any) -> bool:
        text = str(value or "").strip()
        return len(text) == 64 and all(
            char in "0123456789abcdef" for char in text
        )

    def _gate1_data_safety(self, m: Mapping[str, Any]) -> GateResult:
        required = [
            "no_future_leakage",
            "uses_availability_date",
            "point_in_time_rebalance",
            "adjusted_price_consistent",
            "tradability_rules_defined",
            "missingness_explained",
        ]
        missing = self._missing(m, required)
        failed = [key for key in required if self._strict_bool(m, key) is not True]
        reasons: list[str] = []
        if missing:
            reasons.append("缺少数据安全证据: " + ", ".join(missing))
        if failed:
            reasons.append("数据安全项未通过: " + ", ".join(failed))
        if not reasons:
            reasons.append("已确认 point-in-time、availability_date、复权与可交易性口径。")
        return self._result(
            1, not missing and not failed, reasons, {key: m.get(key) for key in required}
        )

    def _gate2_coverage(self, m: Mapping[str, Any]) -> GateResult:
        p = self.policy
        coverage = self._f(m, "coverage_rate")
        nan_rate = self._f(m, "nan_rate")
        monthly_min = self._f(m, "monthly_coverage_min")
        sector_share = self._f(m, "max_sector_coverage_share")
        if sector_share is None:
            sector_share = self._f(m, "sector_concentration")
        size_share = self._f(m, "max_size_bucket_coverage_share")
        if size_share is None:
            size_share = self._f(m, "market_cap_concentration")
        extreme_ratio = self._f(m, "extreme_value_ratio")
        thematic_flag = self._strict_bool(m, "thematic")
        narrow_flag = self._strict_bool(m, "narrow_coverage")
        thematic = thematic_flag is True or narrow_flag is True
        reasons: list[str] = []
        if ("thematic" in m and thematic_flag is None) or (
            "narrow_coverage" in m and narrow_flag is None
        ):
            reasons.append("thematic/narrow_coverage 必须是布尔值。")
        if any(
            value is None
            for value in (
                coverage,
                nan_rate,
                monthly_min,
                sector_share,
                size_share,
                extreme_ratio,
            )
        ):
            reasons.append(
                "缺少 coverage/nan/monthly/sector/size/extreme 完整证据。"
            )
            return self._result(2, False, reasons, dict(m), "error")
        assert coverage is not None
        assert nan_rate is not None
        assert monthly_min is not None
        assert sector_share is not None
        assert size_share is not None
        assert extreme_ratio is not None
        min_cov = p.thematic_min_coverage_rate if thematic else p.min_coverage_rate
        max_nan = p.thematic_max_nan_rate if thematic else p.max_nan_rate
        for label, value in (
            ("coverage_rate", coverage),
            ("nan_rate", nan_rate),
            ("monthly_coverage_min", monthly_min),
            ("max_sector_coverage_share", sector_share),
            ("max_size_bucket_coverage_share", size_share),
            ("extreme_value_ratio", extreme_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                reasons.append(f"{label} 必须位于 [0, 1]。")
        if coverage < min_cov:
            reasons.append(f"coverage_rate={coverage:.2%} 低于门槛 {min_cov:.0%}。")
        if nan_rate > max_nan:
            reasons.append(f"nan_rate={nan_rate:.2%} 高于上限 {max_nan:.0%}。")
        if monthly_min is not None and monthly_min < min_cov * 0.65:
            reasons.append(f"monthly_coverage_min={monthly_min:.2%} 不稳定。")
        if sector_share is not None and sector_share > p.max_coverage_bucket_share and not thematic:
            reasons.append("覆盖率长期集中于少数行业，且未标注 thematic/narrow coverage。")
        if size_share is not None and size_share > p.max_coverage_bucket_share and not thematic:
            reasons.append("覆盖率长期集中于单一市值桶，且未标注 thematic/narrow coverage。")
        if extreme_ratio is not None and extreme_ratio > p.max_extreme_value_ratio:
            reasons.append(f"extreme_value_ratio={extreme_ratio:.2%} 偏高。")
        passed = not reasons
        if passed:
            label = (
                "thematic/narrow coverage 已标注。"
                if thematic
                else "覆盖率、缺失率、行业与市值分布通过。"
            )
            reasons.append(label)
        return self._result(
            2,
            passed,
            reasons,
            {
                "coverage_rate": coverage,
                "nan_rate": nan_rate,
                "monthly_coverage_min": monthly_min,
                "max_sector_coverage_share": sector_share,
                "max_size_bucket_coverage_share": size_share,
                "extreme_value_ratio": extreme_ratio,
                "thematic": thematic,
            },
        )

    def _gate3_ic(self, m: Mapping[str, Any]) -> GateResult:
        p = self.policy
        icir = self._f(m, "icir", self._f(m, "ir"))
        rankic = self._f(m, "mean_rankic")
        positive = self._f(m, "positive_ic_ratio", self._f(m, "ic_positive_rate"))
        single_year = self._f(m, "max_single_year_ic_contribution")
        stable_direction = self._strict_bool(m, "rankic_direction_stable")
        fdr_passed = self._strict_bool(m, "family_fdr_passed")
        fdr_method = str(m.get("family_fdr_method", "") or "")
        fdr_q_value = self._f(m, "family_fdr_q_value")
        reasons: list[str] = []
        if icir is None:
            reasons.append("缺少 ICIR。")
        elif abs(icir) < p.min_watch_icir:
            reasons.append(f"ICIR={icir:.3f} 低于最低观察门槛 {p.min_watch_icir:.2f}。")
        if rankic is None:
            reasons.append("缺少 mean_rankic。")
        if positive is None:
            reasons.append("缺少 positive_ic_ratio。")
        elif not 0.0 <= positive <= 1.0:
            reasons.append("positive_ic_ratio 必须位于 [0, 1]。")
        elif positive < p.min_positive_ic_ratio:
            reasons.append(
                f"positive_ic_ratio={positive:.1%} 低于观察门槛 {p.min_positive_ic_ratio:.0%}。"
            )
        if stable_direction is not True:
            reasons.append("RankIC 方向不稳定。")
        if single_year is None:
            reasons.append("缺少单年份 IC 贡献证据。")
        elif not 0.0 <= single_year <= 1.0:
            reasons.append("单年份 IC 贡献必须位于 [0, 1]。")
        elif single_year > p.max_single_year_ic_contribution:
            reasons.append("IC 过度依赖单一年份。")
        if fdr_method != "benjamini_hochberg_by_family":
            reasons.append("缺少 family-level Benjamini-Hochberg 校正证据。")
        if (
            fdr_passed is not True
            or fdr_q_value is None
            or not 0.0 <= fdr_q_value <= 0.10
        ):
            reasons.append("family FDR q-value 未通过 0.10 门槛。")
        passed = not reasons
        if passed:
            reasons.append("IC/RankIC 达到观察门槛，且方向未显示单一年份依赖。")
        return self._result(
            3,
            passed,
            reasons,
            {
                "icir": icir,
                "mean_rankic": rankic,
                "positive_ic_ratio": positive,
                "rankic_direction_stable": stable_direction,
                "max_single_year_ic_contribution": single_year,
                "family_fdr_method": fdr_method,
                "family_fdr_q_value": fdr_q_value,
                "family_fdr_passed": fdr_passed,
            },
        )

    def _gate4_group_returns(self, m: Mapping[str, Any]) -> GateResult:
        p = self.policy
        spread = self._f(m, "top_bottom_spread", self._f(m, "long_short_return"))
        top_ret = self._f(m, "top_quantile_return")
        mono = self._f(m, "monotonicity", self._f(m, "monotonicity_score"))
        from_long_side = self._strict_bool(m, "long_short_from_long_side")
        reasons: list[str] = []
        if spread is None or spread <= 0.0:
            reasons.append("top-bottom spread 未证明为正。")
        if top_ret is None or top_ret <= 0.0:
            reasons.append("top quantile 没有实际 long-only 可买收益。")
        if mono is None or mono < p.min_monotonicity:
            reasons.append(f"分组收益单调性不足，monotonicity={mono}。")
        elif mono > 1.0:
            reasons.append("monotonicity 必须位于 [0, 1]。")
        if from_long_side is not True:
            reasons.append("long-short 收益主要来自低分组暴跌，而非高分组可买收益。")
        passed = not reasons
        if passed:
            reasons.append("分组收益、top quantile 与 long-short spread 通过。")
        return self._result(
            4,
            passed,
            reasons,
            {
                "top_bottom_spread": spread,
                "top_quantile_return": top_ret,
                "monotonicity": mono,
                "long_short_from_long_side": from_long_side,
            },
        )

    def _gate5_costs(self, m: Mapping[str, Any]) -> GateResult:
        p = self.policy
        turnover = self._f(m, "turnover", self._f(m, "annual_turnover"))
        cost_adj = self._f(m, "cost_adjusted_return")
        slippage_ok = self._strict_bool(m, "slippage_sensitivity_ok")
        execution_ok = self._strict_bool(m, "execution_realism")
        capacity = self._f(m, "capacity_pressure")
        reasons: list[str] = []
        if turnover is None:
            reasons.append("缺少 turnover。")
        elif turnover < 0.0:
            reasons.append("turnover 不能为负。")
        elif turnover > p.max_turnover:
            reasons.append(f"turnover={turnover:.2f}x 过高。")
        if cost_adj is None or cost_adj <= 0.0:
            reasons.append("扣成本后没有正超额。")
        if slippage_ok is not True:
            reasons.append("滑点敏感性过高。")
        if execution_ok is not True:
            reasons.append("缺少真实执行约束验证。")
        if capacity is None:
            reasons.append("缺少容量压力证据。")
        elif not 0.0 <= capacity <= 1.0:
            reasons.append("capacity_pressure 必须位于 [0, 1]。")
        elif capacity > p.max_capacity_pressure:
            reasons.append("容量压力过高，疑似依赖小市值/低流动性。")
        passed = not reasons
        if passed:
            reasons.append("交易成本、滑点、换手与容量通过。")
        return self._result(
            5,
            passed,
            reasons,
            {
                "turnover": turnover,
                "cost_adjusted_return": cost_adj,
                "slippage_sensitivity_ok": slippage_ok,
                "execution_realism": execution_ok,
                "capacity_pressure": capacity,
            },
        )

    def _gate6_exposure(self, m: Mapping[str, Any]) -> GateResult:
        p = self.policy
        neutral_icir = self._f(m, "neutralized_icir", self._f(m, "industry_size_neutral_icir"))
        corr = self._f(m, "correlation_with_existing")
        if corr is None:
            corr = self._f(m, "existing_factor_corr")
        style_only = self._strict_bool(m, "style_exposure_only")
        reasons: list[str] = []
        if neutral_icir is None or abs(neutral_icir) < p.min_neutralized_icir:
            reasons.append("行业+市值中性化后 alpha 不足，可能只是风格暴露。")
        if corr is None:
            reasons.append("缺少与现有因子的相关性证据。")
        elif abs(corr) > p.max_existing_factor_corr:
            reasons.append("与现有因子相关性过高，缺少独立增量。")
        if style_only is None:
            reasons.append("缺少 style_exposure_only 诊断。")
        elif style_only:
            reasons.append("该因子被标注为纯风格暴露，不能作为独立选股因子。")
        passed = not reasons
        if passed:
            reasons.append("中性化后仍有 alpha，且与现有因子不过度重合。")
        return self._result(
            6,
            passed,
            reasons,
            {
                "neutralized_icir": neutral_icir,
                "existing_factor_corr": corr,
                "style_exposure_only": style_only,
            },
        )

    def _gate7_oos(self, m: Mapping[str, Any]) -> GateResult:
        p = self.policy
        oos = self._f(m, "oos_positive_ratio", self._f(m, "walk_forward_positive_ratio"))
        param_ok = self._strict_bool(m, "parameter_stability")
        dates_ok = self._strict_bool(m, "date_range_robustness")
        freq_ok = self._strict_bool(m, "rebalance_frequency_robustness")
        universe_ok = self._strict_bool(m, "universe_robustness")
        regime_ok = self._strict_bool(m, "regime_robustness")
        purged = self._strict_bool(m, "walk_forward_purged")
        purge_days = self._f(m, "walk_forward_purge_days")
        embargo_days = self._f(m, "walk_forward_embargo_days")
        fold_count = self._f(m, "walk_forward_fold_count")
        evidence_hash = str(m.get("walk_forward_evidence_hash", "") or "").strip()
        reasons: list[str] = []
        if oos is None or oos < p.min_oos_positive_ratio:
            reasons.append(f"样本外 positive ratio 未达到 {p.min_oos_positive_ratio:.0%}。")
        elif oos > 1.0:
            reasons.append("样本外 positive ratio 必须位于 [0, 1]。")
        if param_ok is not True:
            reasons.append("参数敏感性未通过，疑似单参数有效。")
        if dates_ok is not True:
            reasons.append("不同起止日期稳健性不足。")
        if freq_ok is not True:
            reasons.append("不同调仓频率稳健性不足。")
        if universe_ok is not True:
            reasons.append("不同 universe 稳健性不足。")
        if regime_ok is not True:
            reasons.append("不同市场阶段稳健性不足。")
        if (
            purged is not True
            or purge_days is None
            or not purge_days.is_integer()
            or purge_days < 30
        ):
            reasons.append("walk-forward 未证明执行至少 30 日 purge。")
        if embargo_days != 30 or not embargo_days.is_integer():
            reasons.append("walk-forward 未证明执行 30 日 embargo。")
        if (
            fold_count is None
            or not fold_count.is_integer()
            or fold_count < 1
            or not self._is_sha256(evidence_hash)
        ):
            reasons.append("缺少可哈希的 walk-forward fold 证据。")
        passed = not reasons
        if passed:
            reasons.append("walk-forward、参数、日期、频率、universe 与市场阶段稳健性通过。")
        return self._result(
            7,
            passed,
            reasons,
            {
                "oos_positive_ratio": oos,
                "parameter_stability": param_ok,
                "date_range_robustness": dates_ok,
                "rebalance_frequency_robustness": freq_ok,
                "universe_robustness": universe_ok,
                "regime_robustness": regime_ok,
                "walk_forward_purged": purged,
                "walk_forward_purge_days": purge_days,
                "walk_forward_embargo_days": embargo_days,
                "walk_forward_fold_count": fold_count,
                "walk_forward_evidence_hash": evidence_hash,
            },
        )

    def _gate8_incremental(self, m: Mapping[str, Any]) -> GateResult:
        p = self.policy
        ret_delta = self._f(m, "master_return_delta")
        sharpe_delta = self._f(m, "sharpe_delta")
        dd_delta = self._f(m, "max_drawdown_delta")
        turnover_delta = self._f(m, "turnover_delta")
        cost_delta = self._f(m, "execution_cost_delta")
        signal_corr = self._f(m, "correlation_with_existing_signals")
        if signal_corr is None:
            signal_corr = self._f(m, "signal_corr")
        evidence_schema = str(m.get("gate8_evidence_schema", "") or "").strip()
        evidence_hash = str(m.get("gate8_evidence_hash", "") or "").strip()
        full_chain = self._strict_bool(m, "full_control_chain_evaluated")
        raw_arm_hashes = m.get("gate8_arm_hashes")
        arm_hashes = (
            {str(key): str(value or "").strip() for key, value in raw_arm_hashes.items()}
            if isinstance(raw_arm_hashes, Mapping)
            else {}
        )
        reasons: list[str] = []
        if ret_delta is None or ret_delta <= 0.0:
            reasons.append("加入组合后 Master return delta 未证明为正。")
        if sharpe_delta is None or sharpe_delta <= 0.0:
            reasons.append("Sharpe delta 未证明为正。")
        if dd_delta is None or dd_delta > p.max_drawdown_delta:
            reasons.append("max drawdown delta 不可接受。")
        if turnover_delta is None:
            reasons.append("缺少 turnover delta。")
        elif turnover_delta < 0.0:
            reasons.append("turnover delta 不能为负。")
        elif turnover_delta > p.max_turnover_delta:
            reasons.append("turnover delta 过高。")
        if cost_delta is None:
            reasons.append("缺少 execution cost delta。")
        elif cost_delta < 0.0:
            reasons.append("execution cost delta 不能为负。")
        elif ret_delta is not None and cost_delta > max(ret_delta, 0.0):
            reasons.append("execution cost delta 吃掉增量收益。")
        if signal_corr is None:
            reasons.append("缺少与现有系统信号相关性证据。")
        elif abs(signal_corr) > p.max_existing_factor_corr:
            reasons.append("与现有系统信号相关性过高，组合增量不足。")
        if evidence_schema != "factor-governance-replay-evidence.v2":
            reasons.append("Gate 8 缺少 canonical A/B/C/D replay schema。")
        if full_chain is not True:
            reasons.append("Gate 8 未走完整 deterministic control chain。")
        if not self._is_sha256(evidence_hash):
            reasons.append("Gate 8 evidence hash 缺失。")
        if set(arm_hashes) != {"A", "B", "C", "D"} or any(
            not self._is_sha256(value) for value in arm_hashes.values()
        ):
            reasons.append("Gate 8 A/B/C/D arm hashes 不完整。")
        passed = not reasons
        if passed:
            reasons.append("baseline + new factor 相对 baseline 有组合级增量。")
        return self._result(
            8,
            passed,
            reasons,
            {
                "master_return_delta": ret_delta,
                "sharpe_delta": sharpe_delta,
                "max_drawdown_delta": dd_delta,
                "turnover_delta": turnover_delta,
                "execution_cost_delta": cost_delta,
                "correlation_with_existing_signals": signal_corr,
                "gate8_evidence_schema": evidence_schema,
                "gate8_evidence_hash": evidence_hash,
                "full_control_chain_evaluated": full_chain,
                "gate8_arm_hashes": arm_hashes,
            },
        )

    def _decide(
        self,
        current_state: FactorLifecycleState,
        gates: list[GateResult],
        metrics: Mapping[str, Any],
    ) -> tuple[FactorAdmissionDecision, FactorLifecycleState, list[str]]:
        gate = {item.gate_id: item for item in gates}
        failed = [item for item in gates if not item.passed]
        reasons = [reason for item in failed[:3] for reason in item.reasons[:2]]
        if not gate[1].passed:
            target = (
                FactorLifecycleState.DEPRECATED
                if current_state == FactorLifecycleState.PRODUCTION_FACTOR
                else FactorLifecycleState.DRAFT
            )
            return (
                FactorAdmissionDecision.REJECT,
                target,
                reasons or ["Gate 1 数据安全失败，直接拒绝。"],
            )
        if not gate[2].passed:
            return FactorAdmissionDecision.REVISE, FactorLifecycleState.RESEARCH_CANDIDATE, reasons
        if not gate[3].passed:
            return (
                FactorAdmissionDecision.WATCHLIST,
                FactorLifecycleState.RESEARCH_CANDIDATE,
                reasons,
            )
        if gate[1].passed and gate[2].passed and gate[3].passed and gate[4].passed:
            all_pass = all(item.passed for item in gates)
            prod_grade_ic = (
                abs(float(metrics.get("icir", metrics.get("ir", 0.0)) or 0.0))
                >= self.policy.min_production_candidate_icir
            )
            prod_grade_hit = (
                float(metrics.get("positive_ic_ratio", metrics.get("ic_positive_rate", 0.0)) or 0.0)
                >= self.policy.min_production_positive_ic_ratio
            )
            if all_pass and prod_grade_ic and prod_grade_hit:
                return (
                    FactorAdmissionDecision.PRODUCTION_CANDIDATE,
                    FactorLifecycleState.PRODUCTION_CANDIDATE,
                    [
                        "8 道门全部通过，达到 production candidate 标准；"
                        "只有 FactorGovernanceProtocol v3 的成熟度、FDR、全链回放、"
                        "slot 预算与月末单次 transition 全部通过后才能成为 production_factor。"
                    ],
                )
            return (
                FactorAdmissionDecision.PAPER_FACTOR,
                FactorLifecycleState.PAPER_FACTOR,
                reasons or ["已通过初步回测，可进入纸面组合观察；未达到正式入库门槛。"],
            )
        if gate[1].passed and gate[2].passed and gate[3].passed:
            return (
                FactorAdmissionDecision.WATCHLIST,
                FactorLifecycleState.RESEARCH_CANDIDATE,
                reasons,
            )
        return FactorAdmissionDecision.REVISE, FactorLifecycleState.RESEARCH_CANDIDATE, reasons

    @staticmethod
    def _summary(
        decision: FactorAdmissionDecision,
        target_state: FactorLifecycleState,
        gates: list[GateResult],
        reasons: list[str],
    ) -> str:
        passed_count = sum(1 for item in gates if item.passed)
        head = f"factor_review decision={decision.value}, target_state={target_state.value}, gates_passed={passed_count}/8."
        if reasons:
            return head + " " + reasons[0]
        return head


__all__ = [
    "FactorAdmissionDecision",
    "FactorGateEvaluator",
    "FactorGovernancePolicy",
    "FactorLifecycleState",
    "FactorRecord",
    "FactorReview",
    "GateResult",
    "GateSpec",
    "GATE_BY_ID",
    "GATE_BY_KEY",
    "GATE_SPECS",
]
