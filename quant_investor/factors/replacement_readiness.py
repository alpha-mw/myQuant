"""Read-only governance assessment for Quant factor replacement proposals.

The helpers in this module deliberately stop before registry mutation.  They
combine strict-fresh factor-health history with prospective Quant-selection
shadow evidence and can emit only conservative governance outcomes.  A
deprecation proposal is possible only when three distinct matured alpha-failure
windows and every replacement-safety requirement are present.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "2026-07-15.quant-factor-replacement-readiness.v2"

OUTCOME_KEEP = "keep"
OUTCOME_WATCHLIST = "watchlist"
OUTCOME_REDUCE_WEIGHT_PROPOSAL = "reduce_weight_proposal"
OUTCOME_DEPRECATION_PROPOSAL = "deprecation_proposal"
OUTCOME_BLOCKED = "blocked"
ALLOWED_OUTCOMES = frozenset(
    {
        OUTCOME_KEEP,
        OUTCOME_WATCHLIST,
        OUTCOME_REDUCE_WEIGHT_PROPOSAL,
        OUTCOME_DEPRECATION_PROPOSAL,
        OUTCOME_BLOCKED,
    }
)


@dataclass(frozen=True)
class ReplacementReadinessPolicy:
    """Conservative, proposal-only replacement thresholds."""

    reduce_after_distinct_matured_failures: int = 2
    deprecate_after_distinct_matured_failures: int = 3
    min_month_end_rankic_count: int = 12
    min_nonoverlap_30d_cohort_count: int = 8
    min_candidate_coverage: float = 0.60


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _strict_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _first_integer(containers: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> int | None:
    for container in containers:
        for key in keys:
            value = _integer(container.get(key))
            if value is not None:
                return value
    return None


def _first_number(containers: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    for container in containers:
        for key in keys:
            value = _number(container.get(key))
            if value is not None:
                return value
    return None


def _timestamp_key(payload: Mapping[str, Any], fallback: int) -> tuple[str, int]:
    for key in ("generated_at", "timestamp", "as_of", "evaluation_end_date"):
        value = str(payload.get(key, "") or "")
        if value:
            return value, fallback
    snapshot = _as_mapping(payload.get("snapshot"))
    value = str(snapshot.get("latest_complete_trade_date", "") or "")
    return value, fallback


def _health_report_contract(report: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    fresh = _as_mapping(report.get("fresh_evaluation"))
    decisions = [
        _as_mapping(row) for row in _as_list(report.get("decisions")) if isinstance(row, Mapping)
    ]
    if fresh.get("requested") is not True:
        blockers.append("fresh_evaluation_not_requested")
    if fresh.get("strict") is not True:
        blockers.append("strict_fresh_evaluation_not_enabled")
    if fresh.get("atomic_success") is not True:
        blockers.append("fresh_evaluation_not_atomic_success")
    if _as_list(fresh.get("blockers")):
        blockers.append("fresh_evaluation_has_blockers")
    if not decisions:
        blockers.append("health_decisions_missing")
    nonfresh = sorted(
        {
            str(row.get("evaluation_source", "") or "missing")
            for row in decisions
            if row.get("evaluation_source") != "fresh_evaluation"
        }
    )
    if nonfresh:
        blockers.append("health_decision_sources_not_all_fresh:" + ",".join(nonfresh))
    source_counts = _as_mapping(report.get("evaluation_source_counts"))
    if source_counts:
        unexpected = sorted(
            str(key)
            for key, value in source_counts.items()
            if key != "fresh_evaluation" and _number(value) not in (None, 0.0)
        )
        if unexpected:
            blockers.append("health_source_counts_not_all_fresh:" + ",".join(unexpected))
    evaluated_count = _integer(fresh.get("evaluated_factor_count"))
    if evaluated_count is not None and evaluated_count != len(decisions):
        blockers.append("fresh_evaluated_factor_count_mismatch")
    run_status = str(report.get("run_status", "") or "")
    if run_status and run_status not in {"ok", "passed"}:
        blockers.append(f"health_run_status_not_ok:{run_status}")
    return blockers


def _factor_name(decision: Mapping[str, Any]) -> str:
    return str(decision.get("factor_name") or decision.get("name") or "").strip()


def _maturity_window_id(decision: Mapping[str, Any]) -> str:
    diagnostics = _as_mapping(decision.get("diagnostics"))
    return str(
        decision.get("maturity_window_id") or diagnostics.get("maturity_window_id") or ""
    ).strip()


def _is_data_blocked(decision: Mapping[str, Any]) -> bool:
    status = str(decision.get("status", "") or "").lower()
    action = str(decision.get("action", "") or "").lower()
    return status == "data_blocked" or action == "data_blocked"


def _is_alpha_failure(decision: Mapping[str, Any]) -> bool:
    explicit = _strict_bool(decision.get("alpha_failure"))
    if explicit is not None:
        return explicit and not _is_data_blocked(decision)
    if _is_data_blocked(decision):
        return False
    status = str(decision.get("status", "") or "").lower()
    action = str(decision.get("action", "") or "").lower()
    return status in {"watchlist", "degraded", "deprecated"} or action in {
        "watchlist",
        "reduce_weight",
        "deprecate",
    }


def _is_healthy_alpha(decision: Mapping[str, Any]) -> bool:
    return str(decision.get("status", "") or "").lower() == "healthy"


def _cohort_end_date(decision: Mapping[str, Any]) -> str:
    diagnostics = _as_mapping(decision.get("diagnostics"))
    raw_value = str(
        decision.get("evidence_end_date")
        or decision.get("evaluation_end_date")
        or diagnostics.get("evaluation_end_date")
        or ""
    ).strip()
    if not raw_value:
        for part in _maturity_window_id(decision).split("|"):
            if part.startswith("end="):
                raw_value = part.removeprefix("end=").strip()
                break
    digits = "".join(character for character in raw_value[:10] if character.isdigit())
    if len(digits) != 8:
        return ""
    try:
        datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        return ""
    return digits


def collect_health_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a cohort-ordered fresh-alpha streak, independent of rerun time."""

    report_audits: list[dict[str, Any]] = []
    factor_windows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    factor_names: set[str] = set()
    factor_data_blocked: dict[str, list[str]] = defaultdict(list)
    factor_latest_alpha_status: dict[str, str] = {}
    factor_latest_alpha_window: dict[str, str] = {}
    factor_healthy_reset_count: Counter[str] = Counter()
    factor_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    last_report_order_cohort_end: dict[str, str] = {}
    global_blockers: list[str] = []

    ordered = sorted(
        enumerate(reports),
        key=lambda item: _timestamp_key(item[1], item[0]),
    )
    for original_index, report in ordered:
        blockers = _health_report_contract(report)
        report_id = str(
            report.get("timestamp")
            or report.get("generated_at")
            or f"health_report_{original_index}"
        )
        report_audits.append(
            {
                "report_id": report_id,
                "accepted": not blockers,
                "blockers": blockers,
            }
        )
        if blockers:
            global_blockers.extend(f"{report_id}:{item}" for item in blockers)
            continue
        for decision_index, raw_decision in enumerate(_as_list(report.get("decisions"))):
            decision = _as_mapping(raw_decision)
            name = _factor_name(decision)
            if not name:
                global_blockers.append(f"{report_id}:health_decision_factor_name_missing")
                continue
            factor_names.add(name)
            window_id = _maturity_window_id(decision)
            cohort_end = _cohort_end_date(decision)
            if not cohort_end:
                global_blockers.append(
                    f"{report_id}:{name}:maturity_cohort_end_date_missing_or_invalid"
                )
                continue
            previous_end = last_report_order_cohort_end.get(name)
            if previous_end and cohort_end < previous_end:
                global_blockers.append(
                    f"{report_id}:{name}:maturity_cohort_time_regression:"
                    f"{cohort_end}<{previous_end}"
                )
            last_report_order_cohort_end[name] = max(cohort_end, previous_end or cohort_end)
            factor_observations[name].append(
                {
                    "decision": decision,
                    "maturity_window_id": window_id,
                    "report_id": report_id,
                    "cohort_end_date": cohort_end,
                    "report_order": len(report_audits),
                    "decision_index": decision_index,
                },
            )

    for name, observations in factor_observations.items():
        classifications_by_window: dict[str, str] = {}
        healthy_windows_seen: set[str] = set()
        for observation in sorted(
            observations,
            key=lambda item: (
                item["cohort_end_date"],
                item["report_order"],
                item["decision_index"],
            ),
        ):
            decision = _as_mapping(observation.get("decision"))
            report_id = str(observation.get("report_id") or "health_report")
            window_id = str(observation.get("maturity_window_id") or "")
            cohort_end = str(observation.get("cohort_end_date") or "")
            if _is_data_blocked(decision):
                factor_data_blocked[name].append(window_id or f"{report_id}:missing_window")
                continue
            if not window_id:
                global_blockers.append(f"{report_id}:{name}:maturity_window_id_missing")
                continue
            classification = (
                "healthy"
                if _is_healthy_alpha(decision)
                else "failure" if _is_alpha_failure(decision) else "unknown"
            )
            if classification == "unknown":
                global_blockers.append(f"{report_id}:{name}:alpha_status_not_classifiable")
                continue
            previous_classification = classifications_by_window.get(window_id)
            if previous_classification and previous_classification != classification:
                global_blockers.append(
                    f"{report_id}:{name}:maturity_window_classification_conflict:" f"{window_id}"
                )
                continue
            classifications_by_window[window_id] = classification
            if classification == "healthy":
                factor_windows[name] = {}
                factor_latest_alpha_status[name] = "healthy"
                factor_latest_alpha_window[name] = window_id
                if window_id not in healthy_windows_seen:
                    factor_healthy_reset_count[name] += 1
                    healthy_windows_seen.add(window_id)
                continue
            factor_latest_alpha_status[name] = "failure"
            factor_latest_alpha_window[name] = window_id
            factor_windows[name].setdefault(
                window_id,
                {
                    "maturity_window_id": window_id,
                    "cohort_end_date": cohort_end,
                    "report_id": report_id,
                    "status": str(decision.get("status", "") or ""),
                    "action": str(decision.get("action", "") or ""),
                },
            )

    return {
        "factor_names": sorted(factor_names),
        "alpha_failure_windows": {
            name: [factor_windows[name][key] for key in sorted(factor_windows[name])]
            for name in sorted(factor_names)
        },
        "data_blocked_windows": {
            name: sorted(set(values)) for name, values in sorted(factor_data_blocked.items())
        },
        "latest_determinable_alpha_status": dict(sorted(factor_latest_alpha_status.items())),
        "latest_determinable_alpha_window": dict(sorted(factor_latest_alpha_window.items())),
        "healthy_reset_count": {
            name: int(factor_healthy_reset_count.get(name, 0)) for name in sorted(factor_names)
        },
        "failure_count_semantics": (
            "cohort-end-ordered distinct alpha-failure maturity windows after the "
            "latest fresh healthy window; data-blocked windows neither count nor reset; "
            "report generation time never defines alpha chronology"
        ),
        "reports": report_audits,
        "blockers": list(dict.fromkeys(global_blockers)),
        "strict_fresh_report_count": sum(row["accepted"] for row in report_audits),
    }


def _readiness_root(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in (
        "replacement_readiness",
        "replacement_readiness_evidence",
        "readiness_evidence",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _candidate_name(payload: Mapping[str, Any]) -> str:
    root = _readiness_root(payload)
    candidate = _as_mapping(payload.get("candidate"))
    return str(
        root.get("candidate_name")
        or candidate.get("name")
        or payload.get("candidate_name")
        or payload.get("candidate")
        or ""
    ).strip()


def _maturity_counts(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    root = _readiness_root(payload)
    candidate = _as_mapping(payload.get("candidate"))
    candidate_maturity = _as_mapping(candidate.get("maturity"))
    root_maturity = _as_mapping(root.get("candidate_maturity") or root.get("maturity"))
    containers = (root_maturity, candidate_maturity, root, candidate, payload)
    monthly = _first_integer(
        containers,
        (
            "month_end_rankic_count",
            "monthly_rankic_count",
            "min_month_end_rankic_count_observed",
            "monthly_rankic_count_from_registry",
            "rank_ic_count",
            "rankic_count",
        ),
    )
    nonoverlap = _first_integer(
        containers,
        (
            "nonoverlap_30d_cohort_count",
            "current_nonoverlap_30d_cohort_count",
            "non_overlapping_30d_cohort_count",
        ),
    )
    return monthly, nonoverlap


def _candidate_coverage(payload: Mapping[str, Any], candidate_name: str) -> float | None:
    root = _readiness_root(payload)
    candidate = _as_mapping(payload.get("candidate"))
    coverage = root.get("candidate_coverage")
    coverage_mapping = _as_mapping(coverage)
    direct = _first_number(
        (coverage_mapping, root, candidate, payload),
        ("coverage_rate", "candidate_coverage_rate", "rate", "coverage"),
    )
    if direct is not None:
        return direct
    runtime = _as_mapping(payload.get("runtime_components"))
    component = _as_mapping(runtime.get(candidate_name))
    return _first_number((component,), ("coverage_rate", "coverage"))


def _gate_bool(
    value: Any,
    *,
    positive_keys: Sequence[str],
    inverse_keys: Sequence[str] = (),
) -> bool | None:
    direct = _strict_bool(value)
    if direct is not None:
        return direct
    mapping = _as_mapping(value)
    if not mapping:
        return None
    evaluated = _strict_bool(mapping.get("evaluated"))
    if evaluated is False:
        return None
    for key in positive_keys:
        result = _strict_bool(mapping.get(key))
        if result is not None:
            return result if evaluated is not False else None
    for key in inverse_keys:
        result = _strict_bool(mapping.get(key))
        if result is not None:
            return not result if evaluated is not False else None
    return None


def _present_alias_sources(
    container: Mapping[str, Any],
    keys: Sequence[str],
    *,
    prefix: str,
) -> list[tuple[str, Any]]:
    return [(f"{prefix}.{key}", container[key]) for key in keys if key in container]


def _gate_values(
    value: Any,
    *,
    positive_keys: Sequence[str],
    inverse_keys: Sequence[str] = (),
) -> list[bool]:
    direct = _strict_bool(value)
    if direct is not None:
        return [direct]
    mapping = _as_mapping(value)
    if not mapping or _strict_bool(mapping.get("evaluated")) is False:
        return []
    values: list[bool] = []
    for key in positive_keys:
        result = _strict_bool(mapping.get(key))
        if result is not None:
            values.append(result)
    for key in inverse_keys:
        result = _strict_bool(mapping.get(key))
        if result is not None:
            values.append(not result)
    return values


def _resolve_gate_aliases(
    sources: Sequence[tuple[str, Any]],
    *,
    positive_keys: Sequence[str],
    inverse_keys: Sequence[str] = (),
    conflict_code: str,
) -> tuple[bool | None, list[str]]:
    """Resolve first-present evidence and reject conflicting alias conclusions."""

    if not sources:
        return None, []
    first_values = _gate_values(
        sources[0][1],
        positive_keys=positive_keys,
        inverse_keys=inverse_keys,
    )
    chosen = first_values[0] if first_values else None
    observed: list[tuple[str, bool]] = []
    for label, value in sources:
        for result in _gate_values(
            value,
            positive_keys=positive_keys,
            inverse_keys=inverse_keys,
        ):
            observed.append((label, result))
    blockers: list[str] = []
    if len({result for _, result in observed}) > 1:
        labels = ",".join(dict.fromkeys(label for label, _ in observed))
        blockers.append(f"{conflict_code}:{labels}")
    return chosen, blockers


def _factor_evidence_map(root: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw: Any = {}
    for key in ("by_removed_factor", "factors", "factor_evidence", "replacements"):
        if key in root:
            raw = root.get(key)
            break
    result: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                result[str(key)] = value
    elif isinstance(raw, list):
        for value in raw:
            if not isinstance(value, Mapping):
                continue
            name = str(value.get("factor_name") or value.get("removed_factor") or "")
            if name:
                result[name] = value
    return result


def _find_arm_evidence(
    payload: Mapping[str, Any],
    factor_name: str,
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]], bool]:
    loo_sources: list[tuple[str, Any]] = []
    replacement_sources: list[tuple[str, Any]] = []
    has_runtime_c = False
    for raw_arm in _as_list(payload.get("arms")):
        arm = _as_mapping(raw_arm)
        if str(arm.get("removed_factor") or "") != factor_name:
            continue
        arm_type = str(arm.get("arm_type") or "")
        if arm_type == "leave_one_out":
            loo_sources.extend(
                _present_alias_sources(
                    arm,
                    ("readiness_evidence", "loo_evidence", "forward_performance"),
                    prefix="loo_arm",
                )
            )
        elif arm_type == "one_for_one_replacement":
            replacement_sources.extend(
                _present_alias_sources(
                    arm,
                    (
                        "readiness_evidence",
                        "replacement_evidence",
                        "forward_performance",
                    ),
                    prefix="replacement_arm",
                )
            )
            has_runtime_c = True
    protocol = _as_mapping(payload.get("arm_protocol"))
    runtime_recomputed = protocol.get("runtime_composite_recomputed") is True
    linear_overlay = protocol.get("gate8_linear_overlay_used")
    has_runtime_c = has_runtime_c and runtime_recomputed and linear_overlay is not True
    return loo_sources, replacement_sources, has_runtime_c


def _scope_complete(
    payload: Mapping[str, Any],
    root: Mapping[str, Any],
) -> tuple[bool | None, list[str]]:
    sources = _present_alias_sources(
        root,
        ("scope_complete", "scope"),
        prefix="replacement_readiness",
    )
    payload_scope = _as_mapping(payload.get("scope"))
    if payload_scope:
        claimed = _strict_bool(payload_scope.get("complete_production_screening_effect_claimed"))
        if claimed is not None:
            sources.append(
                (
                    "shadow.scope.complete_production_screening_effect_claimed",
                    claimed and not bool(_as_list(payload.get("scope_limitations"))),
                )
            )
    sources.extend(
        _present_alias_sources(
            payload,
            ("scope_complete",),
            prefix="shadow",
        )
    )
    return _resolve_gate_aliases(
        sources,
        positive_keys=(
            "complete",
            "passed",
            "scope_complete",
            "complete_production_screening_effect_claimed",
        ),
        conflict_code="selection_scope_alias_conflict",
    )


def _cross_branch_positive(
    payload: Mapping[str, Any],
    root: Mapping[str, Any],
) -> tuple[bool | None, list[str]]:
    sources = _present_alias_sources(
        root,
        ("cross_branch_conditional_increment", "cross_branch_increment"),
        prefix="replacement_readiness",
    )
    sources.extend(
        _present_alias_sources(
            payload,
            ("cross_branch_conditional_increment", "cross_branch_increment"),
            prefix="shadow",
        )
    )
    return _resolve_gate_aliases(
        sources,
        positive_keys=(
            "positive",
            "passed",
            "conditional_increment_positive",
            "positive_after_conditioning",
        ),
        conflict_code="cross_branch_conditional_increment_alias_conflict",
    )


def _selection_bias_acceptable(
    payload: Mapping[str, Any],
    root: Mapping[str, Any],
) -> tuple[bool | None, list[str]]:
    """Require an explicit bias-review conclusion without inventing a threshold."""

    sources = _present_alias_sources(
        root,
        ("covered_uncovered_selection_bias", "selection_bias"),
        prefix="replacement_readiness",
    )
    sources.extend(
        _present_alias_sources(
            payload,
            ("covered_uncovered_selection_bias", "selection_bias"),
            prefix="shadow",
        )
    )
    return _resolve_gate_aliases(
        sources,
        positive_keys=("acceptable",),
        conflict_code="covered_uncovered_selection_bias_alias_conflict",
    )


def _factor_replacement_gates(
    payload: Mapping[str, Any],
    factor_name: str,
) -> tuple[dict[str, bool | None], list[str]]:
    root = _readiness_root(payload)
    factor_evidence = _factor_evidence_map(root).get(factor_name, {})
    loo_arm_sources, replacement_arm_sources, runtime_c = _find_arm_evidence(payload, factor_name)
    blockers: list[str] = []

    loo_sources = (
        _present_alias_sources(
            factor_evidence,
            ("loo_deletion", "loo_deletion_not_worse", "loo_not_worse"),
            prefix=f"factor.{factor_name}",
        )
        + loo_arm_sources
    )
    loo_not_worse, conflicts = _resolve_gate_aliases(
        loo_sources,
        positive_keys=("not_worse", "passed", "loo_deletion_not_worse"),
        conflict_code=f"{factor_name}:loo_evidence_alias_conflict",
    )
    blockers.extend(conflicts)

    replacement_sources = (
        _present_alias_sources(
            factor_evidence,
            ("candidate_replacement", "replacement"),
            prefix=f"factor.{factor_name}",
        )
        + replacement_arm_sources
    )
    better_a_sources = (
        _present_alias_sources(
            factor_evidence,
            ("replacement_better_than_a",),
            prefix=f"factor.{factor_name}",
        )
        + replacement_sources
    )
    better_a, conflicts = _resolve_gate_aliases(
        better_a_sources,
        positive_keys=("better_than_a", "outperforms_a", "better_than_a_and_b", "passed"),
        conflict_code=f"{factor_name}:replacement_better_than_a_alias_conflict",
    )
    blockers.extend(conflicts)
    better_b_sources = (
        _present_alias_sources(
            factor_evidence,
            ("replacement_better_than_b",),
            prefix=f"factor.{factor_name}",
        )
        + replacement_sources
    )
    better_b, conflicts = _resolve_gate_aliases(
        better_b_sources,
        positive_keys=("better_than_b", "outperforms_b", "better_than_a_and_b", "passed"),
        conflict_code=f"{factor_name}:replacement_better_than_b_alias_conflict",
    )
    blockers.extend(conflicts)

    redundancy_sources = _present_alias_sources(
        factor_evidence,
        ("redundancy", "redundancy_evidence", "redundancy_passed"),
        prefix=f"factor.{factor_name}",
    )
    redundancy, conflicts = _resolve_gate_aliases(
        redundancy_sources,
        positive_keys=("is_redundant", "redundant", "passed"),
        conflict_code=f"{factor_name}:redundancy_evidence_alias_conflict",
    )
    blockers.extend(conflicts)
    diversifier_sources = _present_alias_sources(
        factor_evidence,
        ("diversifier_tail_protection", "diversifier_check", "tail_protection_check"),
        prefix=f"factor.{factor_name}",
    )
    diversifier, conflicts = _resolve_gate_aliases(
        diversifier_sources,
        positive_keys=(
            "safe_to_remove",
            "no_material_protection",
            "passed",
            "replacement_preserves_protection",
        ),
        inverse_keys=("material_protection_found", "protected_diversifier"),
        conflict_code=f"{factor_name}:diversifier_tail_evidence_alias_conflict",
    )
    blockers.extend(conflicts)
    runtime_explicit, conflicts = _resolve_gate_aliases(
        replacement_sources,
        positive_keys=("runtime_recomputed", "actual_runtime_replacement"),
        conflict_code=f"{factor_name}:runtime_replacement_alias_conflict",
    )
    blockers.extend(conflicts)
    if runtime_explicit is not None:
        runtime_c = runtime_c and runtime_explicit

    return (
        {
            "loo_deletion_not_worse": loo_not_worse,
            "runtime_c_replacement_executed": runtime_c,
            "replacement_better_than_a": better_a,
            "replacement_better_than_b": better_b,
            "redundancy_evidence_passed": redundancy,
            "diversifier_tail_protection_safe": diversifier,
        },
        blockers,
    )


def _decision_shadow_contract_blockers(
    payload: Mapping[str, Any],
) -> list[str]:
    """Validate the immutable, read-only shadow evidence contract."""

    blockers: list[str] = []
    status = str(payload.get("status", "") or "missing")
    if status != "passed":
        blockers.append(f"selection_shadow_status_not_passed:{status}")
    if payload.get("measurement_only") is not True:
        blockers.append("selection_shadow_not_measurement_only")
    fail_closed_blockers = payload.get("fail_closed_blockers")
    if not isinstance(fail_closed_blockers, list):
        blockers.append("selection_shadow_fail_closed_blockers_missing")
    elif fail_closed_blockers:
        blockers.append("selection_shadow_has_fail_closed_blockers")
    if payload.get("fail_closed") is True:
        blockers.append("selection_shadow_fail_closed_true")

    registry = _as_mapping(payload.get("registry"))
    if registry.get("unchanged") is not True:
        blockers.append("selection_shadow_registry_not_unchanged")
    preregistration = _as_mapping(payload.get("preregistration"))
    if preregistration.get("status") != "matched":
        blockers.append("selection_shadow_preregistration_not_matched")
    baseline_contract = _as_mapping(payload.get("baseline_contract"))
    if baseline_contract.get("status") != "matched":
        blockers.append("selection_shadow_baseline_contract_not_matched")

    runtime_parity = _as_mapping(payload.get("runtime_parity"))
    old_delta = _number(runtime_parity.get("old14_max_abs_delta"))
    candidate_delta = _number(runtime_parity.get("candidate_max_abs_delta"))
    if old_delta != 0.0:
        blockers.append("selection_shadow_old_runtime_parity_not_exact")
    if candidate_delta != 0.0:
        blockers.append("selection_shadow_candidate_runtime_parity_not_exact")

    protocol = _as_mapping(payload.get("arm_protocol"))
    if protocol.get("runtime_composite_recomputed") is not True:
        blockers.append("selection_shadow_runtime_composite_not_recomputed")
    if protocol.get("gate8_linear_overlay_used") is not False:
        blockers.append("selection_shadow_gate8_overlay_contract_not_false")
    return blockers


def _decision_shadow_ledger_identity(
    payload: Mapping[str, Any],
) -> dict[str, str]:
    registry = _as_mapping(payload.get("registry"))
    preregistration = _as_mapping(payload.get("preregistration"))
    baseline_contract = _as_mapping(payload.get("baseline_contract"))
    return {
        "registry_sha256": str(
            registry.get("sha256_after")
            or registry.get("sha256_before")
            or registry.get("sha256")
            or ""
        ),
        "preregistration_policy_sha256": str(
            preregistration.get("actual_policy_sha256")
            or preregistration.get("recorded_policy_sha256")
            or ""
        ),
        "baseline_contract_sha256": str(
            baseline_contract.get("actual_contract_sha256")
            or baseline_contract.get("recorded_contract_sha256")
            or ""
        ),
    }


def _audit_observation_ledger_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_name: str,
    decision_shadow: Mapping[str, Any],
    decision_shadow_contract_valid: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Audit ledger lineage without using ledger rows as proposal evidence."""

    audits: list[dict[str, Any]] = []
    blockers: list[str] = []
    expected = _decision_shadow_ledger_identity(decision_shadow)
    expected_schema = "2026-07-11.quant-factor-selection-observation.v2"
    for index, row in enumerate(rows):
        observation_key = str(row.get("observation_key") or f"ledger_row_{index}")
        row_blockers: list[str] = []
        if not decision_shadow_contract_valid:
            row_blockers.append("ledger_decision_shadow_contract_invalid")
        if row.get("schema_version") != expected_schema:
            row_blockers.append("ledger_schema_version_mismatch")
        if row.get("measurement_only") is not True:
            row_blockers.append("ledger_not_measurement_only")
        if row.get("registry_write") is not False:
            row_blockers.append("ledger_registry_write_contract_not_false")
        if str(row.get("candidate") or "") != candidate_name:
            row_blockers.append("ledger_candidate_mismatch")
        for key, expected_value in expected.items():
            actual_value = str(row.get(key) or "")
            if not expected_value:
                row_blockers.append(f"decision_shadow_{key}_missing")
            elif actual_value != expected_value:
                row_blockers.append(f"ledger_{key}_mismatch")
        audits.append(
            {
                "observation_key": observation_key,
                "accepted_for_audit": not row_blockers,
                "used_for_candidate_maturity": False,
                "blockers": row_blockers,
            }
        )
        blockers.extend(f"{observation_key}:{item}" for item in row_blockers)
    return audits, blockers


def collect_selection_evidence(
    shadow_reports: Sequence[Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]] = (),
    *,
    candidate_name: str | None = None,
    policy: ReplacementReadinessPolicy | None = None,
) -> dict[str, Any]:
    """Collect latest shadow safety gates and deduplicated maturity evidence."""

    policy = policy or ReplacementReadinessPolicy()
    deduplicated_rows: list[Mapping[str, Any]] = []
    seen_observations: set[str] = set()
    duplicate_observation_count = 0
    for index, row in enumerate(observation_rows):
        key = str(row.get("observation_key") or f"ledger_row_{index}")
        if key in seen_observations:
            duplicate_observation_count += 1
            continue
        seen_observations.add(key)
        deduplicated_rows.append(row)

    names = sorted(
        {_candidate_name(payload) for payload in shadow_reports if _candidate_name(payload)}
    )
    selected_candidate = str(candidate_name or "").strip()
    blockers: list[str] = []
    if not selected_candidate:
        if len(names) == 1:
            selected_candidate = names[0]
        elif not names:
            blockers.append("candidate_name_missing")
        else:
            blockers.append("candidate_name_ambiguous:" + ",".join(names))
    matching_shadow_reports = [
        payload for payload in shadow_reports if _candidate_name(payload) == selected_candidate
    ]
    decision_payload: Mapping[str, Any] = {}
    if matching_shadow_reports:
        decision_payload = max(
            enumerate(matching_shadow_reports),
            key=lambda item: _timestamp_key(item[1], item[0]),
        )[1]
    else:
        blockers.append(f"decision_selection_shadow_missing_for_candidate:{selected_candidate}")
    shadow_contract_blockers: list[str] = []
    if decision_payload:
        shadow_contract_blockers = _decision_shadow_contract_blockers(decision_payload)
        blockers.extend(shadow_contract_blockers)
    shadow_contract_valid = bool(decision_payload and not shadow_contract_blockers)
    trusted_shadow = decision_payload if shadow_contract_valid else {}

    monthly_count: int | None = None
    nonoverlap_count: int | None = None
    if trusted_shadow:
        monthly_count, nonoverlap_count = _maturity_counts(trusted_shadow)
    maturity_ready = bool(
        (monthly_count is not None and monthly_count >= policy.min_month_end_rankic_count)
        or (
            nonoverlap_count is not None
            and nonoverlap_count >= policy.min_nonoverlap_30d_cohort_count
        )
    )

    ledger_audits, ledger_blockers = _audit_observation_ledger_rows(
        deduplicated_rows,
        candidate_name=selected_candidate,
        decision_shadow=decision_payload,
        decision_shadow_contract_valid=shadow_contract_valid,
    )
    blockers.extend(ledger_blockers)

    root = _readiness_root(trusted_shadow)
    coverage = _candidate_coverage(trusted_shadow, selected_candidate)
    coverage_passed = bool(coverage is not None and coverage >= policy.min_candidate_coverage)
    scope_complete, alias_blockers = _scope_complete(trusted_shadow, root)
    blockers.extend(alias_blockers)
    cross_branch_positive, alias_blockers = _cross_branch_positive(trusted_shadow, root)
    blockers.extend(alias_blockers)
    selection_bias_acceptable, alias_blockers = _selection_bias_acceptable(trusted_shadow, root)
    blockers.extend(alias_blockers)

    factor_gates: dict[str, dict[str, bool | None]] = {}
    factor_names = set(_factor_evidence_map(root))
    factor_names.update(
        str(arm.get("removed_factor") or "")
        for arm in _as_list(trusted_shadow.get("arms"))
        if isinstance(arm, Mapping) and arm.get("removed_factor")
    )
    for factor_name in sorted(factor_names):
        gates, alias_blockers = _factor_replacement_gates(trusted_shadow, factor_name)
        factor_gates[factor_name] = gates
        blockers.extend(alias_blockers)

    return {
        "candidate_name": selected_candidate,
        "candidate_maturity": {
            "month_end_rankic_count": monthly_count,
            "nonoverlap_30d_cohort_count": nonoverlap_count,
            "min_month_end_rankic_count": policy.min_month_end_rankic_count,
            "min_nonoverlap_30d_cohort_count": policy.min_nonoverlap_30d_cohort_count,
            "threshold_logic": "month_end_rankic_count>=12 OR nonoverlap_30d_cohort_count>=8",
            "passed": maturity_ready,
        },
        "candidate_coverage": {
            "coverage_rate": coverage,
            "minimum": policy.min_candidate_coverage,
            "passed": coverage_passed,
        },
        "cross_branch_conditional_increment_positive": cross_branch_positive,
        "covered_uncovered_selection_bias_acceptable": selection_bias_acceptable,
        "scope_complete": scope_complete,
        "factor_gates": factor_gates,
        "selection_shadow_report_count": len(shadow_reports),
        "decision_shadow_contract_valid": shadow_contract_valid,
        "ledger_rows_are_maturity_evidence": False,
        "observation_ledger_row_count": len(observation_rows),
        "unique_observation_count": len(deduplicated_rows),
        "valid_ledger_audit_row_count": sum(row["accepted_for_audit"] for row in ledger_audits),
        "duplicate_observation_count": duplicate_observation_count,
        "observation_ledger_audits": ledger_audits,
        "decision_evidence_timestamp": (
            _timestamp_key(decision_payload, 0)[0] if decision_payload else ""
        ),
        "blockers": blockers,
    }


def _proposal_gate_blockers(
    selection: Mapping[str, Any],
    factor_name: str,
) -> list[str]:
    blockers: list[str] = []
    maturity = _as_mapping(selection.get("candidate_maturity"))
    coverage = _as_mapping(selection.get("candidate_coverage"))
    if maturity.get("passed") is not True:
        blockers.append("candidate_maturity_not_met")
    if coverage.get("passed") is not True:
        blockers.append("candidate_coverage_not_met")
    cross_branch = selection.get("cross_branch_conditional_increment_positive")
    if cross_branch is not True:
        blockers.append(
            "cross_branch_conditional_increment_missing"
            if cross_branch is None
            else "cross_branch_conditional_increment_not_positive"
        )
    selection_bias = selection.get("covered_uncovered_selection_bias_acceptable")
    if selection_bias is not True:
        blockers.append(
            "covered_uncovered_selection_bias_review_missing"
            if selection_bias is None
            else "covered_uncovered_selection_bias_not_acceptable"
        )
    scope = selection.get("scope_complete")
    if scope is not True:
        blockers.append(
            "selection_scope_incomplete" if scope is False else "selection_scope_missing"
        )
    gates = _as_mapping(_as_mapping(selection.get("factor_gates")).get(factor_name))
    required = (
        "loo_deletion_not_worse",
        "runtime_c_replacement_executed",
        "replacement_better_than_a",
        "replacement_better_than_b",
        "redundancy_evidence_passed",
        "diversifier_tail_protection_safe",
    )
    for key in required:
        value = gates.get(key)
        if value is not True:
            blockers.append(f"{key}_missing" if value is None else f"{key}_failed")
    blockers.extend(str(item) for item in _as_list(selection.get("blockers")))
    return list(dict.fromkeys(blockers))


def assess_replacement_readiness(
    health_reports: Sequence[Mapping[str, Any]],
    shadow_reports: Sequence[Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]] = (),
    *,
    candidate_name: str | None = None,
    factor_names: Sequence[str] | None = None,
    policy: ReplacementReadinessPolicy | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return fail-closed, proposal-only replacement readiness decisions."""

    policy = policy or ReplacementReadinessPolicy()
    health = collect_health_evidence(health_reports)
    selection = collect_selection_evidence(
        shadow_reports,
        observation_rows,
        candidate_name=candidate_name,
        policy=policy,
    )
    selected_factors = sorted(
        {str(item) for item in (factor_names or health.get("factor_names", [])) if str(item)}
    )
    global_blockers = list(health.get("blockers", []))
    if not health_reports:
        global_blockers.append("strict_fresh_health_reports_missing")
    if not selected_factors:
        global_blockers.append("factor_scope_empty")

    failures_by_factor = _as_mapping(health.get("alpha_failure_windows"))
    data_blocked_by_factor = _as_mapping(health.get("data_blocked_windows"))
    latest_alpha_status_by_factor = _as_mapping(health.get("latest_determinable_alpha_status"))
    accepted_health_factor_names = {
        str(item) for item in _as_list(health.get("factor_names")) if str(item)
    }
    decisions: list[dict[str, Any]] = []
    for factor_name in selected_factors:
        failure_rows = _as_list(failures_by_factor.get(factor_name))
        failure_ids = sorted(
            {
                str(_as_mapping(row).get("maturity_window_id") or "")
                for row in failure_rows
                if str(_as_mapping(row).get("maturity_window_id") or "")
            }
        )
        failure_count = len(failure_ids)
        data_blocked_windows = _as_list(data_blocked_by_factor.get(factor_name))
        latest_alpha_status = str(latest_alpha_status_by_factor.get(factor_name) or "unknown")
        proposal_blockers = _proposal_gate_blockers(selection, factor_name)
        decision_blockers = list(global_blockers)
        if factor_name not in accepted_health_factor_names:
            decision_blockers.append("requested_factor_missing_from_health_evidence")

        if decision_blockers:
            outcome = OUTCOME_BLOCKED
        elif data_blocked_windows and failure_count == 0:
            outcome = OUTCOME_BLOCKED
            decision_blockers.append("fresh_data_blocked_no_alpha_conclusion")
        elif failure_count >= policy.deprecate_after_distinct_matured_failures:
            if latest_alpha_status != "failure":
                outcome = OUTCOME_BLOCKED
                decision_blockers.append(
                    "latest_determinable_alpha_status_not_failure:" f"{latest_alpha_status}"
                )
            elif proposal_blockers:
                outcome = OUTCOME_BLOCKED
                decision_blockers.extend(proposal_blockers)
            else:
                outcome = OUTCOME_DEPRECATION_PROPOSAL
        elif failure_count >= policy.reduce_after_distinct_matured_failures:
            if latest_alpha_status != "failure":
                outcome = OUTCOME_BLOCKED
                decision_blockers.append(
                    "latest_determinable_alpha_status_not_failure:" f"{latest_alpha_status}"
                )
            elif proposal_blockers:
                outcome = OUTCOME_BLOCKED
                decision_blockers.extend(proposal_blockers)
            else:
                outcome = OUTCOME_REDUCE_WEIGHT_PROPOSAL
        elif failure_count > 0:
            outcome = OUTCOME_WATCHLIST
        else:
            outcome = OUTCOME_KEEP

        if outcome not in ALLOWED_OUTCOMES:  # defensive contract guard
            raise AssertionError(f"invalid replacement readiness outcome: {outcome}")
        decisions.append(
            {
                "factor_name": factor_name,
                "outcome": outcome,
                "proposal_only": outcome.endswith("_proposal"),
                "distinct_matured_alpha_failure_count": failure_count,
                "distinct_maturity_window_ids": failure_ids,
                "latest_determinable_alpha_status": latest_alpha_status,
                "duplicate_maturity_windows_do_not_increment": True,
                "data_blocked_window_count": len(data_blocked_windows),
                "data_blocked_windows_not_counted_as_alpha_failure": list(data_blocked_windows),
                "required_distinct_failures_for_reduce": (
                    policy.reduce_after_distinct_matured_failures
                ),
                "required_distinct_failures_for_deprecation": (
                    policy.deprecate_after_distinct_matured_failures
                ),
                "replacement_evidence": _as_mapping(
                    _as_mapping(selection.get("factor_gates")).get(factor_name)
                ),
                "proposal_blockers": proposal_blockers,
                "blockers": list(dict.fromkeys(decision_blockers)),
            }
        )

    outcome_counts = Counter(row["outcome"] for row in decisions)
    severity = {
        OUTCOME_KEEP: 0,
        OUTCOME_WATCHLIST: 1,
        OUTCOME_REDUCE_WEIGHT_PROPOSAL: 2,
        OUTCOME_DEPRECATION_PROPOSAL: 3,
        OUTCOME_BLOCKED: 4,
    }
    overall_outcome = (
        max((row["outcome"] for row in decisions), key=severity.get)
        if decisions
        else OUTCOME_BLOCKED
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now_iso(),
        "status": overall_outcome,
        "measurement_only": True,
        "freeze": {
            "active": False,
            "policy": "retired_v13_incubation",
        },
        "mutation_governance": {
            "mode": "read_only_proposal",
            "registry_mutation_allowed": False,
            "production_weight_change_allowed": False,
            "production_apply_authorization": "required",
        },
        "registry_update_status": "not_written_read_only_proposal",
        "candidate": {
            "name": selection.get("candidate_name", ""),
            "maturity": selection.get("candidate_maturity", {}),
            "coverage": selection.get("candidate_coverage", {}),
            "cross_branch_conditional_increment_positive": selection.get(
                "cross_branch_conditional_increment_positive"
            ),
            "covered_uncovered_selection_bias_acceptable": selection.get(
                "covered_uncovered_selection_bias_acceptable"
            ),
            "scope_complete": selection.get("scope_complete"),
        },
        "health_evidence": health,
        "selection_evidence": selection,
        "factor_decisions": decisions,
        "outcome_counts": dict(outcome_counts),
        "fail_closed": overall_outcome == OUTCOME_BLOCKED,
        "fail_closed_blockers": list(
            dict.fromkeys(
                item for decision in decisions for item in _as_list(decision.get("blockers"))
            )
        ),
        "allowed_outcomes": sorted(ALLOWED_OUTCOMES),
        "explicit_non_actions": [
            "no_registry_read_or_write",
            "no_factor_weight_change",
            "no_factor_deprecation",
            "no_production_factor_promotion",
            "no_market_or_portfolio_run",
            "no_order_broker_or_live_provider_action",
        ],
    }
