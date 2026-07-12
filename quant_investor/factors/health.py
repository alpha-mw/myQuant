"""Health monitoring and de-risking rules for governed production factors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from quant_investor.factors.governance import (
    FactorAdmissionDecision,
    FactorGovernancePolicy,
    FactorLifecycleState,
    FactorRecord,
)


class FactorHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WATCHLIST = "watchlist"
    DEGRADED = "degraded"
    DATA_BLOCKED = "data_blocked"
    DEPRECATED = "deprecated"


class FactorHealthAction(str, Enum):
    KEEP = "keep"
    OBSERVE = "observe"
    WATCHLIST = "watchlist"
    REDUCE_WEIGHT = "reduce_weight"
    DEPRECATE = "deprecate"


@dataclass(frozen=True)
class FactorHealthPolicy:
    """Conservative production health thresholds.

    The automation is intentionally asymmetric: it may de-risk weak production
    factors when explicitly asked to write the registry, but it must not
    silently promote new alpha into production.
    """

    min_watch_icir: float = FactorGovernancePolicy.min_watch_icir
    min_production_icir: float = FactorGovernancePolicy.min_production_candidate_icir
    min_positive_ic_ratio: float = FactorGovernancePolicy.min_positive_ic_ratio
    min_production_positive_ic_ratio: float = (
        FactorGovernancePolicy.min_production_positive_ic_ratio
    )
    min_oos_positive_ratio: float = FactorGovernancePolicy.min_oos_positive_ratio
    min_neutralized_icir: float = FactorGovernancePolicy.min_neutralized_icir
    max_turnover: float = FactorGovernancePolicy.max_turnover
    max_capacity_pressure: float = FactorGovernancePolicy.max_capacity_pressure
    hard_min_coverage_rate: float = 0.30
    hard_max_nan_rate: float = 0.70
    reduce_after_failures: int = 2
    deprecate_after_failures: int = 3
    weight_decay: float = 0.50
    min_active_weight: float = 0.01
    max_history: int = 24


@dataclass(frozen=True)
class FactorHealthDecision:
    factor_name: str
    status: FactorHealthStatus
    action: FactorHealthAction
    consecutive_failures: int
    reasons: list[str] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    health_metrics: dict[str, Any] = field(default_factory=dict)
    evaluation_id: str = ""
    maturity_window_id: str = ""
    evaluation_hash: str = ""
    current_weight: float = 0.0
    new_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "status": self.status.value,
            "action": self.action.value,
            "consecutive_failures": int(self.consecutive_failures),
            "reasons": list(self.reasons),
            "gate_failures": list(self.gate_failures),
            "health_metrics": dict(self.health_metrics),
            "evaluation_id": self.evaluation_id,
            "maturity_window_id": self.maturity_window_id,
            "evaluation_hash": self.evaluation_hash,
            "current_weight": float(self.current_weight),
            "new_weight": float(self.new_weight),
        }


def classify_factor_health(
    record: FactorRecord,
    evaluation: Mapping[str, Any] | None,
    *,
    previous_failure_count: int = 0,
    count_failure: bool = True,
    policy: FactorHealthPolicy | None = None,
) -> FactorHealthDecision:
    """Classify one production factor health review result.

    ``count_failure=False`` is used when the same matured evaluation window has
    already been processed, preventing repeated scheduled runs from
    double-counting the same weak sample.
    """

    policy = policy or FactorHealthPolicy()
    current_weight = float(record.weight)
    reasons: list[str] = []
    gate_failures: list[str] = []
    evaluation_hash = _evaluation_id(evaluation)
    maturity_window_id = _maturity_window_id(evaluation)
    metrics: Mapping[str, Any] = {}

    if evaluation is None:
        reasons.append("missing latest factor evaluation")
        raw_status = FactorHealthStatus.DATA_BLOCKED
    else:
        metrics = dict(evaluation.get("metrics", {}) or {})
        gate_failures = _gate_failures(evaluation)
        raw_status = FactorHealthStatus.HEALTHY

        if _gate_failed(evaluation, 1):
            raw_status = FactorHealthStatus.DATA_BLOCKED
            reasons.append("gate 1 data safety failed")

        coverage = _float(metrics, "coverage_rate")
        nan_rate = _float(metrics, "nan_rate")
        if coverage is not None and coverage < policy.hard_min_coverage_rate:
            raw_status = FactorHealthStatus.DATA_BLOCKED
            reasons.append(f"coverage_rate={coverage:.2%} below hard floor")
        if nan_rate is not None and nan_rate > policy.hard_max_nan_rate:
            raw_status = FactorHealthStatus.DATA_BLOCKED
            reasons.append(f"nan_rate={nan_rate:.2%} above hard ceiling")

        _append_threshold_reason(
            reasons,
            metrics,
            "icir",
            policy.min_watch_icir,
            "below watch threshold",
        )
        _append_threshold_reason(
            reasons,
            metrics,
            "positive_ic_ratio",
            policy.min_positive_ic_ratio,
            "below watch threshold",
        )
        _append_threshold_reason(
            reasons,
            metrics,
            "oos_positive_ratio",
            policy.min_oos_positive_ratio,
            "below OOS threshold",
        )
        _append_threshold_reason(
            reasons,
            metrics,
            "neutralized_icir",
            policy.min_neutralized_icir,
            "below neutralized threshold",
        )

        top_bottom = _float(metrics, "top_bottom_spread")
        if top_bottom is not None and top_bottom <= 0.0:
            reasons.append(f"top_bottom_spread={top_bottom:.4f} is not positive")
        cost_adjusted = _float(metrics, "cost_adjusted_return")
        if cost_adjusted is not None and cost_adjusted <= 0.0:
            reasons.append(f"cost_adjusted_return={cost_adjusted:.4f} is not positive")
        turnover = _float(metrics, "turnover")
        if turnover is not None and turnover > policy.max_turnover:
            reasons.append(f"turnover={turnover:.2f}x above max {policy.max_turnover:.2f}x")
        capacity = _float(metrics, "capacity_pressure")
        if capacity is not None and capacity > policy.max_capacity_pressure:
            reasons.append(
                f"capacity_pressure={capacity:.2%} above max "
                f"{policy.max_capacity_pressure:.0%}"
            )

        decision = _review_decision(evaluation)
        if decision and decision != FactorAdmissionDecision.PRODUCTION_CANDIDATE.value:
            reasons.append(f"latest review decision={decision}")
        if gate_failures:
            reasons.append("one or more governance gates failed")

        if reasons and raw_status == FactorHealthStatus.HEALTHY:
            raw_status = FactorHealthStatus.WATCHLIST

    failing = raw_status != FactorHealthStatus.HEALTHY
    failure_count = int(previous_failure_count or 0)
    if not failing:
        duplicate_window = not count_failure and failure_count > 0
        return FactorHealthDecision(
            factor_name=record.name,
            status=FactorHealthStatus.HEALTHY,
            action=(
                FactorHealthAction.OBSERVE
                if duplicate_window
                else FactorHealthAction.KEEP
            ),
            consecutive_failures=failure_count if duplicate_window else 0,
            reasons=[],
            gate_failures=gate_failures,
            health_metrics=_health_metric_subset(metrics),
            evaluation_id=evaluation_hash,
            maturity_window_id=maturity_window_id,
            evaluation_hash=evaluation_hash,
            current_weight=current_weight,
            new_weight=current_weight,
        )

    # Missing or unsafe data is not evidence that the alpha has failed.  Keep
    # the prior alpha-failure streak unchanged and never mutate the factor
    # weight from a data-blocked observation.
    if raw_status == FactorHealthStatus.DATA_BLOCKED:
        return FactorHealthDecision(
            factor_name=record.name,
            status=FactorHealthStatus.DATA_BLOCKED,
            action=FactorHealthAction.OBSERVE,
            consecutive_failures=failure_count,
            reasons=reasons,
            gate_failures=gate_failures,
            health_metrics=_health_metric_subset(metrics),
            evaluation_id=evaluation_hash,
            maturity_window_id=maturity_window_id,
            evaluation_hash=evaluation_hash,
            current_weight=current_weight,
            new_weight=current_weight,
        )

    if count_failure:
        failure_count += 1

    if not count_failure:
        action = FactorHealthAction.OBSERVE
    elif failure_count >= policy.deprecate_after_failures:
        action = FactorHealthAction.DEPRECATE
    elif failure_count >= policy.reduce_after_failures:
        action = FactorHealthAction.REDUCE_WEIGHT
    else:
        action = FactorHealthAction.WATCHLIST

    if action == FactorHealthAction.DEPRECATE:
        status = FactorHealthStatus.DEPRECATED
        new_weight = 0.0
    elif action == FactorHealthAction.REDUCE_WEIGHT:
        status = FactorHealthStatus.DEGRADED
        new_weight = _decayed_weight(current_weight, policy)
    else:
        status = (
            raw_status
            if raw_status == FactorHealthStatus.DATA_BLOCKED
            else FactorHealthStatus.WATCHLIST
        )
        new_weight = current_weight

    return FactorHealthDecision(
        factor_name=record.name,
        status=status,
        action=action,
        consecutive_failures=failure_count,
        reasons=reasons,
        gate_failures=gate_failures,
        health_metrics=_health_metric_subset(metrics),
        evaluation_id=evaluation_hash,
        maturity_window_id=maturity_window_id,
        evaluation_hash=evaluation_hash,
        current_weight=current_weight,
        new_weight=new_weight,
    )


def apply_health_decision(
    record: FactorRecord,
    decision: FactorHealthDecision,
    *,
    reviewed_at: str,
    report_path: str = "",
    policy: FactorHealthPolicy | None = None,
) -> FactorRecord:
    """Apply a conservative health action to a registry record in place."""

    policy = policy or FactorHealthPolicy()
    metadata = dict(record.metadata or {})
    monitor = dict(metadata.get("health_monitor", {}) or {})
    active_failure_windows = active_failure_maturity_window_ids(monitor)
    history = list(monitor.get("history", []) or [])
    history.append({"reviewed_at": reviewed_at, **decision.to_dict()})
    history = history[-max(int(policy.max_history), 1) :]

    monitor.update(
        {
            "latest_reviewed_at": reviewed_at,
            "status": decision.status.value,
            "action": decision.action.value,
            "consecutive_failures": int(decision.consecutive_failures),
            "last_report_path": report_path,
            "latest_reasons": list(decision.reasons),
            "latest_health_metrics": dict(decision.health_metrics),
            "history": history,
        }
    )
    if decision.status == FactorHealthStatus.DATA_BLOCKED:
        monitor["last_data_blocked_evaluation_hash"] = decision.evaluation_hash
    else:
        monitor.update(
            {
                "last_evaluation_id": decision.evaluation_id,
                "last_maturity_window_id": decision.maturity_window_id,
                "last_evaluation_hash": decision.evaluation_hash,
            }
        )
        maturity_window_id = str(decision.maturity_window_id or "").strip()
        if decision.status == FactorHealthStatus.HEALTHY:
            if maturity_window_id not in active_failure_windows:
                active_failure_windows = []
        elif (
            maturity_window_id
            and maturity_window_id not in {"missing", "unknown"}
            and maturity_window_id not in active_failure_windows
        ):
            active_failure_windows.append(maturity_window_id)
    monitor["active_failure_maturity_window_ids"] = active_failure_windows
    metadata["health_monitor"] = monitor
    record.metadata = metadata

    if decision.action == FactorHealthAction.REDUCE_WEIGHT:
        record.weight = float(decision.new_weight)
    elif decision.action == FactorHealthAction.DEPRECATE:
        record.weight = 0.0
        record.state = FactorLifecycleState.DEPRECATED
        record.deprecated_reason = "; ".join(decision.reasons[:3]) or "factor health deprecated"

    return record


def _float(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def active_failure_maturity_window_ids(
    monitor: Mapping[str, Any],
) -> list[str]:
    """Return distinct alpha-failure windows in the current active streak."""

    if "active_failure_maturity_window_ids" in monitor:
        raw_values = monitor.get("active_failure_maturity_window_ids", [])
        if not isinstance(raw_values, (list, tuple, set)):
            raw_values = []
        return list(
            dict.fromkeys(
                value
                for item in raw_values
                if (value := str(item or "").strip())
                and value not in {"missing", "unknown"}
            )
        )

    active: list[str] = []
    history = monitor.get("history", []) or []
    if isinstance(history, list):
        for raw_item in history:
            if not isinstance(raw_item, Mapping):
                continue
            status = str(raw_item.get("status", "") or "")
            window = str(
                raw_item.get("maturity_window_id", "") or ""
            ).strip()
            if status == FactorHealthStatus.DATA_BLOCKED.value:
                continue
            if status == FactorHealthStatus.HEALTHY.value:
                if window not in active:
                    active = []
                continue
            if (
                window
                and window not in {"missing", "unknown"}
                and window not in active
            ):
                active.append(window)

    if not active and int(monitor.get("consecutive_failures", 0) or 0) > 0:
        legacy_window = str(
            monitor.get("last_maturity_window_id", "")
            or monitor.get("last_evaluation_id", "")
            or ""
        ).strip()
        if legacy_window and legacy_window not in {"missing", "unknown"}:
            active.append(legacy_window)
    return active


def _append_threshold_reason(
    reasons: list[str],
    metrics: Mapping[str, Any],
    key: str,
    minimum: float,
    label: str,
) -> None:
    value = _float(metrics, key)
    if value is not None and value < minimum:
        reasons.append(f"{key}={value:.4f} {label} {minimum:.4f}")


def _gate_failures(evaluation: Mapping[str, Any] | None) -> list[str]:
    if not evaluation:
        return []
    review = evaluation.get("review", {}) or {}
    failures: list[str] = []
    for gate in review.get("gate_results", []) or []:
        if isinstance(gate, Mapping) and not bool(gate.get("passed", False)):
            gate_id = gate.get("gate_id", "?")
            gate_key = gate.get("gate_key", "gate")
            failures.append(f"{gate_id}:{gate_key}")
    return failures


def _gate_failed(evaluation: Mapping[str, Any] | None, gate_id: int) -> bool:
    if not evaluation:
        return False
    review = evaluation.get("review", {}) or {}
    for gate in review.get("gate_results", []) or []:
        if isinstance(gate, Mapping) and int(gate.get("gate_id", 0) or 0) == gate_id:
            return not bool(gate.get("passed", False))
    return False


def _review_decision(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return ""
    review = evaluation.get("review", {}) or {}
    return str(review.get("decision", "") or "")


def _evaluation_id(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return "missing"
    diagnostics = evaluation.get("diagnostics", {}) or {}
    explicit = str(
        diagnostics.get("evaluation_hash")
        or diagnostics.get("evaluation_id")
        or ""
    )
    if explicit:
        return explicit
    end_date = str(diagnostics.get("evaluation_end_date", "") or "")
    start_date = str(diagnostics.get("analysis_start_date", "") or "")
    snapshot_id = str(diagnostics.get("snapshot_id", "") or "")
    universes = diagnostics.get("universes", []) or []
    if isinstance(universes, str):
        universes = [universes]
    universe_key = ",".join(
        sorted({str(item) for item in universes if str(item)})
    )
    decision_cost_bps = str(diagnostics.get("decision_cost_bps", "") or "")
    warmup_days = str(diagnostics.get("warmup_days", "") or "")
    implementation_hash = str(diagnostics.get("implementation_hash", "") or "")
    rankic_count = str(diagnostics.get("rankic_count", "") or "")
    horizon = str(
        (evaluation.get("metrics", {}) or {}).get("horizon_days", "") or ""
    )
    if end_date:
        return (
            f"snapshot={snapshot_id}|universes={universe_key}|"
            f"window={start_date}:{end_date}|h={horizon}|warmup={warmup_days}|"
            f"cost_bps={decision_cost_bps}|impl={implementation_hash}|"
            f"n={rankic_count}"
        )
    return str(evaluation.get("name", "") or "unknown")


def _maturity_window_id(evaluation: Mapping[str, Any] | None) -> str:
    if not evaluation:
        return "missing"
    diagnostics = evaluation.get("diagnostics", {}) or {}
    explicit = str(diagnostics.get("maturity_window_id", "") or "")
    if explicit:
        return explicit
    end_date = str(diagnostics.get("evaluation_end_date", "") or "")
    rankic_count = str(diagnostics.get("rankic_count", "") or "")
    horizon = str(
        (evaluation.get("metrics", {}) or {}).get("horizon_days", "") or ""
    )
    if end_date:
        return f"end={end_date}|h={horizon}|n={rankic_count}"
    return str(evaluation.get("name", "") or "unknown")


def _health_metric_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "coverage_rate",
        "nan_rate",
        "icir",
        "mean_rankic",
        "positive_ic_ratio",
        "oos_positive_ratio",
        "top_bottom_spread",
        "top_quantile_return",
        "turnover",
        "cost_adjusted_return",
        "capacity_pressure",
        "neutralized_icir",
        "master_return_delta",
        "sharpe_delta",
        "max_drawdown_delta",
        "turnover_delta",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _decayed_weight(current_weight: float, policy: FactorHealthPolicy) -> float:
    if abs(current_weight) <= 1e-12:
        return 0.0
    sign = 1.0 if current_weight >= 0 else -1.0
    current_magnitude = abs(current_weight)
    decayed_magnitude = current_magnitude * policy.weight_decay
    if current_magnitude > policy.min_active_weight:
        decayed_magnitude = max(decayed_magnitude, policy.min_active_weight)
    return sign * min(current_magnitude, decayed_magnitude)


__all__ = [
    "FactorHealthAction",
    "FactorHealthDecision",
    "FactorHealthPolicy",
    "FactorHealthStatus",
    "active_failure_maturity_window_ids",
    "apply_health_decision",
    "classify_factor_health",
]
