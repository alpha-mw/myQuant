"""Offline factor governance admission helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_investor.factors.schema import (
    ADMISSION_DECISION_APPROVE_PAPER_TRADING,
    ADMISSION_DECISION_APPROVE_PRODUCTION,
    ADMISSION_DECISION_DISABLE,
    ADMISSION_DECISION_NEEDS_RESEARCH,
    ADMISSION_DECISION_REJECT,
    FACTOR_STATUS_DISABLED,
    FACTOR_STATUS_PAPER_TRADING,
    FACTOR_STATUS_PRODUCTION,
    FACTOR_STATUS_REJECTED,
    FACTOR_STATUS_RESEARCH_CANDIDATE,
    FACTOR_STATUS_VALIDATED_RESEARCH,
    VALIDATION_VERDICT_FAIL,
    VALIDATION_VERDICT_PASS,
    VALIDATION_VERDICT_WARN,
    FactorAdmissionDecision,
    FactorBacktestResult,
    FactorLibraryEntry,
    FactorValidationReport,
    FactorValidationThresholds,
    ProductionFactorLibrary,
    make_admission_decision_id,
    make_production_library_id,
    make_validation_report_id,
)


def _metric_snapshot(result: FactorBacktestResult) -> dict[str, Any]:
    return {
        "sample_days": result.sample_days,
        "coverage_ratio": result.coverage_ratio,
        "missing_ratio": result.missing_ratio,
        "rank_ic_mean": result.rank_ic_mean,
        "ic_mean": result.ic_mean,
        "icir": result.icir,
        "ic_t_stat": result.ic_t_stat,
        "positive_ic_ratio": result.positive_ic_ratio,
        "top_bottom_spread": result.top_bottom_spread,
        "after_cost_top_bottom_spread": result.after_cost_top_bottom_spread,
        "before_cost_sharpe": result.before_cost_sharpe,
        "after_cost_sharpe": result.after_cost_sharpe,
        "max_drawdown": result.max_drawdown,
        "turnover_avg": result.turnover_avg,
        "monotonicity_score": result.monotonicity_score,
        "capacity_estimate": result.capacity_estimate,
    }


def _mapping_from_metadata(metadata: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    if isinstance(value, Mapping):
        return {str(item_key): item_value for item_key, item_value in value.items()}
    return {}


def _point_in_time_snapshot(result: FactorBacktestResult) -> dict[str, Any]:
    snapshot = _mapping_from_metadata(result.metadata, "point_in_time_snapshot")
    if "point_in_time_passed" in result.metadata:
        snapshot["point_in_time_passed"] = bool(result.metadata["point_in_time_passed"])
    if "passed" in result.metadata:
        snapshot["passed"] = bool(result.metadata["passed"])
    return snapshot


def _point_in_time_passed(snapshot: Mapping[str, Any]) -> bool:
    if "point_in_time_passed" in snapshot:
        return bool(snapshot["point_in_time_passed"])
    if "passed" in snapshot:
        return bool(snapshot["passed"])
    if "is_point_in_time" in snapshot:
        return bool(snapshot["is_point_in_time"])
    return False


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def evaluate_backtest_against_thresholds(
    result: FactorBacktestResult,
    thresholds: FactorValidationThresholds | None = None,
) -> FactorValidationReport:
    """Evaluate an already-produced synthetic/offline backtest result."""

    resolved_thresholds = thresholds or FactorValidationThresholds()
    gate_results: dict[str, str] = {}
    failed_gates: list[str] = []
    warning_gates: list[str] = []

    def hard_gate(name: str, passed: bool) -> None:
        gate_results[name] = VALIDATION_VERDICT_PASS if passed else VALIDATION_VERDICT_FAIL
        if not passed:
            failed_gates.append(name)

    def warning_gate(name: str, triggered: bool) -> None:
        gate_results[name] = VALIDATION_VERDICT_WARN if triggered else VALIDATION_VERDICT_PASS
        if triggered:
            warning_gates.append(name)

    hard_gate("sample_days", result.sample_days >= resolved_thresholds.min_sample_days)
    hard_gate("coverage_ratio", result.coverage_ratio >= resolved_thresholds.min_coverage_ratio)
    hard_gate(
        "rank_ic_mean",
        result.rank_ic_mean is not None
        and result.rank_ic_mean >= resolved_thresholds.min_rank_ic_mean,
    )
    hard_gate("icir", result.icir is not None and result.icir >= resolved_thresholds.min_icir)
    hard_gate(
        "ic_t_stat",
        result.ic_t_stat is not None and result.ic_t_stat >= resolved_thresholds.min_ic_t_stat,
    )
    hard_gate(
        "after_cost_sharpe",
        result.after_cost_sharpe is not None
        and result.after_cost_sharpe >= resolved_thresholds.min_after_cost_sharpe,
    )
    hard_gate(
        "positive_ic_ratio",
        result.positive_ic_ratio is not None
        and result.positive_ic_ratio >= resolved_thresholds.min_positive_ic_ratio,
    )
    if resolved_thresholds.require_positive_after_cost_spread:
        hard_gate(
            "after_cost_spread",
            result.after_cost_top_bottom_spread is not None
            and result.after_cost_top_bottom_spread > 0.0,
        )
    if resolved_thresholds.require_monotonic_quantiles:
        hard_gate(
            "monotonicity",
            result.monotonicity_score is not None and result.monotonicity_score >= 1.0,
        )

    pit_snapshot = _point_in_time_snapshot(result)
    if resolved_thresholds.require_point_in_time:
        hard_gate("point_in_time", _point_in_time_passed(pit_snapshot))
        if not pit_snapshot:
            pit_snapshot = {
                "passed": False,
                "reason": "missing_point_in_time_evidence",
            }

    if resolved_thresholds.max_drawdown is not None:
        warning_gate(
            "max_drawdown",
            result.max_drawdown is None or result.max_drawdown > resolved_thresholds.max_drawdown,
        )
    if resolved_thresholds.max_turnover is not None:
        warning_gate(
            "max_turnover",
            result.turnover_avg is None or result.turnover_avg > resolved_thresholds.max_turnover,
        )

    correlation_snapshot = _mapping_from_metadata(result.metadata, "correlation_snapshot")
    correlation_value = correlation_snapshot.get("max_correlation_with_production")
    if correlation_value is None:
        correlation_value = result.metadata.get("max_correlation_with_production")
    max_correlation = _optional_float(correlation_value)
    if max_correlation is not None:
        correlation_snapshot["max_correlation_with_production"] = max_correlation
        warning_gate(
            "production_correlation",
            max_correlation > resolved_thresholds.max_correlation_with_production,
        )

    capacity_snapshot = _mapping_from_metadata(result.metadata, "capacity_snapshot")
    if result.capacity_estimate is not None and "capacity_estimate" not in capacity_snapshot:
        capacity_snapshot["capacity_estimate"] = result.capacity_estimate

    if failed_gates:
        overall_verdict = VALIDATION_VERDICT_FAIL
        recommended_status = FACTOR_STATUS_REJECTED
        rationale = f"Validation failed hard gates: {', '.join(sorted(failed_gates))}."
    elif warning_gates:
        overall_verdict = VALIDATION_VERDICT_WARN
        recommended_status = FACTOR_STATUS_PAPER_TRADING
        rationale = f"Validation passed hard gates with warnings: {', '.join(sorted(warning_gates))}."
    else:
        overall_verdict = VALIDATION_VERDICT_PASS
        recommended_status = FACTOR_STATUS_VALIDATED_RESEARCH
        rationale = "Validation passed all hard gates."

    return FactorValidationReport(
        report_id=make_validation_report_id(
            factor_id=result.factor_id,
            factor_version=result.factor_version,
            backtest_result_id=result.result_id,
        ),
        factor_id=result.factor_id,
        factor_version=result.factor_version,
        generated_at=str(
            result.metadata.get("validation_generated_at")
            or result.end_date
            or result.start_date
            or "unspecified"
        ),
        backtest_result_id=result.result_id,
        thresholds=resolved_thresholds,
        overall_verdict=overall_verdict,
        gate_results=gate_results,
        failed_gates=failed_gates,
        warning_gates=warning_gates,
        metric_snapshot=_metric_snapshot(result),
        correlation_snapshot=correlation_snapshot,
        capacity_snapshot=capacity_snapshot,
        point_in_time_snapshot=pit_snapshot,
        recommended_status=recommended_status,
        rationale=rationale,
        metadata={
            "factor_governance_pass": "phase9_pass1",
            "source": "evaluate_backtest_against_thresholds",
        },
    )


def propose_admission_decision(
    report: FactorValidationReport,
    *,
    decided_at: str,
    decided_by: str = "system",
    metadata: Mapping[str, Any] | None = None,
) -> FactorAdmissionDecision:
    """Propose a non-production admission decision from a validation report."""

    resolved_metadata = dict(metadata or {})
    conditions: list[str] = []
    if report.overall_verdict == VALIDATION_VERDICT_PASS:
        decision = ADMISSION_DECISION_APPROVE_PAPER_TRADING
        target_status = FACTOR_STATUS_PAPER_TRADING
        rationale = "Validation passed; approve paper trading only. Production approval is manual in Pass 1."
    elif report.overall_verdict == VALIDATION_VERDICT_WARN:
        conditions = [f"resolve_warning_gate:{gate}" for gate in report.warning_gates]
        if resolved_metadata.get("allow_paper_trading_with_warnings") is True:
            decision = ADMISSION_DECISION_APPROVE_PAPER_TRADING
            target_status = FACTOR_STATUS_PAPER_TRADING
            rationale = "Validation warned; approve paper trading with explicit conditions."
        else:
            decision = ADMISSION_DECISION_NEEDS_RESEARCH
            target_status = FACTOR_STATUS_RESEARCH_CANDIDATE
            rationale = "Validation warned; needs further research before admission."
    else:
        decision = ADMISSION_DECISION_REJECT
        target_status = FACTOR_STATUS_REJECTED
        conditions = [f"fix_failed_gate:{gate}" for gate in report.failed_gates]
        rationale = "Validation failed hard gates; reject factor admission."

    return FactorAdmissionDecision(
        decision_id=make_admission_decision_id(
            factor_id=report.factor_id,
            factor_version=report.factor_version,
            decision=decision,
            decided_at=decided_at,
        ),
        factor_id=report.factor_id,
        factor_version=report.factor_version,
        validation_report_id=report.report_id,
        decision=decision,
        target_status=target_status,
        decided_at=decided_at,
        decided_by=decided_by,
        rationale=rationale,
        expires_at=resolved_metadata.get("expires_at"),
        conditions=conditions,
        metadata=resolved_metadata,
    )


def build_library_entry_from_decision(
    decision: FactorAdmissionDecision,
    *,
    owner: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FactorLibraryEntry:
    if (
        decision.target_status == FACTOR_STATUS_PRODUCTION
        and decision.decision != ADMISSION_DECISION_APPROVE_PRODUCTION
    ):
        raise ValueError("production entry requires an approve_production decision.")
    if (
        decision.target_status == FACTOR_STATUS_PAPER_TRADING
        and decision.decision != ADMISSION_DECISION_APPROVE_PAPER_TRADING
    ):
        raise ValueError("paper_trading entry requires an approve_paper_trading decision.")
    if decision.target_status == FACTOR_STATUS_REJECTED and decision.decision != ADMISSION_DECISION_REJECT:
        raise ValueError("rejected entry requires a reject decision.")
    if decision.target_status == FACTOR_STATUS_DISABLED and decision.decision != ADMISSION_DECISION_DISABLE:
        raise ValueError("disabled entry requires a disable decision.")

    status = decision.target_status
    return FactorLibraryEntry(
        factor_id=decision.factor_id,
        factor_version=decision.factor_version,
        status=status,
        admission_decision_id=decision.decision_id,
        validation_report_id=decision.validation_report_id,
        production_since=decision.decided_at if status == FACTOR_STATUS_PRODUCTION else None,
        paper_trading_since=decision.decided_at if status == FACTOR_STATUS_PAPER_TRADING else None,
        deprecated_at=None,
        disabled_at=decision.decided_at if status == FACTOR_STATUS_DISABLED else None,
        expires_at=decision.expires_at,
        last_revalidation_at=None,
        owner=owner,
        tags=list(tags or []),
        metadata=dict(metadata or {}),
    )


def build_production_factor_library(
    entries: Sequence[FactorLibraryEntry],
    *,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> ProductionFactorLibrary:
    production_entries = [
        entry for entry in entries if entry.status == FACTOR_STATUS_PRODUCTION
    ]
    seen: set[tuple[str, str]] = set()
    for entry in production_entries:
        key = (entry.factor_id, entry.factor_version)
        if key in seen:
            raise ValueError(f"Duplicate production factor entry: {entry.factor_id} {entry.factor_version}")
        seen.add(key)
    ordered = sorted(production_entries, key=lambda entry: (entry.factor_id, entry.factor_version))
    return ProductionFactorLibrary(
        library_id=make_production_library_id(ordered),
        generated_at=generated_at,
        entries=ordered,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "evaluate_backtest_against_thresholds",
    "propose_admission_decision",
    "build_library_entry_from_decision",
    "build_production_factor_library",
]
