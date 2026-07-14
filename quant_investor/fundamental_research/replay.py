"""Fail-closed validation for canonical fundamental control-chain replays."""

from __future__ import annotations

from collections.abc import Mapping
from math import fsum, isclose
from typing import Any


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"fundamental control-chain replay {field} must be a mapping")
    return value


def _records(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"fundamental control-chain replay {field} must be a list of mappings")
    return value


def _symbols(records: list[Mapping[str, Any]], field: str) -> list[str]:
    symbols = [str(item.get("symbol") or "").strip() for item in records]
    if any(not symbol for symbol in symbols) or len(symbols) != len(set(symbols)):
        raise ValueError(
            f"fundamental control-chain replay {field} symbols must be nonempty and unique"
        )
    return symbols


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fundamental control-chain replay {field} must be numeric") from exc
    if not (-10.0 <= result <= 10.0):
        raise ValueError(f"fundamental control-chain replay {field} is out of bounds")
    return result


def _same(left: Any, right: Any) -> bool:
    return isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _score_action(score: float) -> str:
    if score >= 0.25:
        return "buy"
    if score <= -0.35:
        return "sell"
    return "hold"


def _action_priority(action: Any) -> int:
    priorities = {"avoid": 0, "sell": 1, "watch": 2, "hold": 2, "buy": 3}
    normalized = str(action or "").lower()
    if normalized not in priorities:
        raise ValueError("fundamental control-chain replay action is invalid")
    return priorities[normalized]


def _validate_master_disabled(decision: Mapping[str, Any]) -> None:
    master_hints = decision.get("master_hints", {})
    if master_hints not in ({}, None):
        master_hints = _mapping(master_hints, "portfolio master_hints")
        output = master_hints.get("portfolio_master_output")
        output = _mapping(output, "portfolio master output")
        if output.get("status") != "disabled_for_deterministic_counterfactual_replay":
            raise ValueError("fundamental control-chain replay contains master advisory")
    metadata = _mapping(decision.get("metadata", {}), "portfolio metadata")
    master_meta = metadata.get("portfolio_master_meta")
    if master_meta not in ({}, None):
        master_meta = _mapping(master_meta, "portfolio master metadata")
        if master_meta.get("status") != "disabled_for_deterministic_counterfactual_replay":
            raise ValueError("fundamental control-chain replay contains master advisory")


def validate_control_chain_replay(replay: Mapping[str, Any]) -> None:
    """Reject incomplete, advisory-contaminated, or internally inconsistent replay."""

    if replay.get("schema_version") != "fundamental-control-chain-replay.v1":
        raise ValueError("fundamental control-chain replay schema is invalid")
    if replay.get("measurement_only") is not True:
        raise ValueError("fundamental control-chain replay is not measurement-only")
    variant = str(replay.get("variant") or "")
    if variant not in {"with_dossier", "without_dossier"}:
        raise ValueError("fundamental control-chain replay variant is invalid")

    _mapping(replay.get("branch_summaries"), "branch_summaries")
    branches = _mapping(replay.get("branch_verdicts_by_symbol"), "branch_verdicts_by_symbol")
    bayesian_records = _records(replay.get("bayesian_records"), "bayesian_records")
    shortlist = _records(replay.get("shortlist"), "shortlist")
    ic_decisions = _records(replay.get("ic_decisions"), "ic_decisions")
    risk = _mapping(replay.get("risk_decision"), "risk_decision")
    plan = _mapping(replay.get("portfolio_plan"), "portfolio_plan")
    decision = _mapping(replay.get("portfolio_decision"), "portfolio_decision")
    if replay.get("ic_hints_by_symbol") != {}:
        raise ValueError("fundamental control-chain replay contains IC hints")
    _validate_master_disabled(decision)

    shortlist_symbols = _symbols(shortlist, "shortlist")
    bayesian_symbols = set(_symbols(bayesian_records, "bayesian_records"))
    if not set(shortlist_symbols).issubset(bayesian_symbols):
        raise ValueError("fundamental control-chain replay shortlist is not Bayesian-bound")
    if not set(shortlist_symbols).issubset(str(symbol) for symbol in branches):
        raise ValueError("fundamental control-chain replay shortlist is not branch-bound")

    branch_scores: dict[str, list[float]] = {}
    branch_confidences: dict[str, list[float]] = {}
    for symbol, branch_map in branches.items():
        if not str(symbol).strip():
            raise ValueError("fundamental control-chain replay branch symbol is empty")
        branch_map = _mapping(branch_map, "symbol branch verdicts")
        for branch_name, verdict in branch_map.items():
            verdict = _mapping(verdict, "branch verdict")
            branch_scores.setdefault(str(branch_name), []).append(
                _number(verdict.get("final_score"), "branch score")
            )
            branch_confidences.setdefault(str(branch_name), []).append(
                _number(verdict.get("final_confidence", 0.0), "branch confidence")
            )
    summaries = _mapping(replay.get("branch_summaries"), "branch_summaries")
    if set(summaries) != set(branch_scores):
        raise ValueError("fundamental control-chain replay branch summary set mismatch")
    for branch_name, scores in branch_scores.items():
        summary = _mapping(summaries[branch_name], "branch summary")
        if not _same(summary.get("final_score"), fsum(scores) / len(scores)):
            raise ValueError("fundamental control-chain replay branch summary score mismatch")
        confidences = branch_confidences[branch_name]
        if not _same(
            summary.get("final_confidence", 0.0),
            fsum(confidences) / len(confidences),
        ):
            raise ValueError("fundamental control-chain replay branch summary confidence mismatch")

    variant_count = 0
    for branch_map in branches.values():
        branch_map = _mapping(branch_map, "symbol branch verdicts")
        fundamental = branch_map.get("fundamental")
        if not isinstance(fundamental, Mapping):
            continue
        metadata = _mapping(fundamental.get("metadata", {}), "fundamental metadata")
        branch_variant = str(metadata.get("fundamental_research_variant") or "")
        if branch_variant:
            if branch_variant != variant:
                raise ValueError("fundamental control-chain replay branch variant mismatch")
            variant_count += 1
        runtime = metadata.get("fundamental_research_runtime")
        if variant == "without_dossier":
            if runtime is not None:
                raise ValueError(
                    "fundamental control-chain replay no-dossier branch contains dossier runtime"
                )
        elif branch_variant:
            runtime = _mapping(runtime, "with-dossier runtime")
            if (
                not str(runtime.get("request_id") or "")
                or not str(runtime.get("dossier_id") or "")
                or runtime.get("measurement_only") is not True
                or runtime.get("applied") is not True
                or runtime.get("counterfactual") is not False
            ):
                raise ValueError("fundamental control-chain replay with-dossier runtime is invalid")
    if variant_count == 0:
        raise ValueError("fundamental control-chain replay has no variant-bound branch")

    bayesian_by_symbol = {str(item["symbol"]): item for item in bayesian_records}
    for symbol in shortlist_symbols:
        record = bayesian_by_symbol[symbol]
        item = shortlist[shortlist_symbols.index(symbol)]
        action_score = _number(record.get("posterior_action_score"), "Bayesian action score")
        expected_alpha = _number(record.get("posterior_expected_alpha"), "Bayesian expected alpha")
        confidence = _number(record.get("posterior_confidence"), "Bayesian confidence")
        edge = _number(
            record.get("posterior_edge_after_costs", expected_alpha),
            "Bayesian edge after costs",
        )
        if _score_action(action_score) != "buy" or expected_alpha <= 0.0 or edge <= 0.0:
            raise ValueError("fundamental control-chain replay shortlist is Bayesian-ineligible")
        if (
            not _same(item.get("rank_score"), action_score)
            or not _same(item.get("confidence"), confidence)
            or not _same(item.get("expected_upside"), max(expected_alpha, 0.0))
            or str(item.get("action") or "").lower() != "buy"
        ):
            raise ValueError("fundamental control-chain replay shortlist score mismatch")
        record_meta = _mapping(record.get("metadata", {}), "Bayesian metadata")
        record_variant = str(record_meta.get("fundamental_research_variant") or "")
        if record_variant == variant:
            fundamental = _mapping(
                _mapping(branches[symbol], "symbol branch verdicts").get("fundamental"),
                "fundamental verdict",
            )
            if not _same(record_meta.get("fundamental_score"), fundamental.get("final_score")):
                raise ValueError("fundamental control-chain replay Bayesian/branch score mismatch")

    ic_symbols = _symbols(ic_decisions, "ic_decisions")
    if set(ic_symbols) != set(shortlist_symbols):
        raise ValueError("fundamental control-chain replay IC symbols mismatch shortlist")
    for item in ic_decisions:
        metadata = _mapping(item.get("metadata", {}), "IC metadata")
        if metadata.get("llm_hint_applied") is not False:
            raise ValueError("fundamental control-chain replay IC hint state is invalid")
        if metadata.get("llm_master_hint"):
            raise ValueError("fundamental control-chain replay contains master hints")
        symbol = str(item.get("symbol") or "")
        verdicts = _mapping(branches[symbol], "IC branch verdicts")
        scores = [
            _number(_mapping(value, "IC branch verdict").get("final_score"), "IC branch score")
            for value in verdicts.values()
        ]
        confidences = [
            _number(
                _mapping(value, "IC branch verdict").get("final_confidence", 0.0),
                "IC branch confidence",
            )
            for value in verdicts.values()
        ]
        expected_score = max(-1.0, min(1.0, fsum(scores) / len(scores)))
        expected_confidence = max(0.0, min(1.0, fsum(confidences) / len(confidences)))
        if not _same(item.get("final_score"), expected_score) or not _same(
            item.get("final_confidence", 0.0), expected_confidence
        ):
            raise ValueError("fundamental control-chain replay IC/branch score mismatch")
        expected_action = _score_action(expected_score)
        action_cap = str(risk.get("action_cap") or "buy").lower()
        if _action_priority(expected_action) > _action_priority(action_cap):
            expected_action = action_cap
        if str(item.get("action") or "").lower() != expected_action:
            raise ValueError("fundamental control-chain replay IC action mismatch")

    plan_weights = dict(_mapping(plan.get("target_weights"), "plan target_weights"))
    decision_weights = dict(_mapping(decision.get("target_weights"), "decision target_weights"))
    if plan_weights != decision_weights:
        raise ValueError("fundamental control-chain replay plan/decision weights mismatch")
    if plan.get("target_exposure") != decision.get("target_exposure"):
        raise ValueError("fundamental control-chain replay plan/decision exposure mismatch")
    if not set(plan_weights).issubset(shortlist_symbols):
        raise ValueError("fundamental control-chain replay weights are outside shortlist")

    decision_shortlist = _records(decision.get("shortlist"), "decision shortlist")
    if _symbols(decision_shortlist, "decision shortlist") != shortlist_symbols:
        raise ValueError("fundamental control-chain replay decision shortlist mismatch")
    risk_constraints = _mapping(decision.get("risk_constraints"), "risk_constraints")
    if risk_constraints.get("risk_decision") != risk:
        raise ValueError("fundamental control-chain replay risk decision mismatch")

    blocked = {
        str(symbol)
        for symbol in (
            list(risk.get("blocked_symbols") or [])
            + list(plan.get("blocked_symbols") or [])
            + list(plan.get("rejected_symbols") or [])
        )
    }
    positive_weights = {
        str(symbol) for symbol, weight in plan_weights.items() if float(weight or 0.0) > 0.0
    }
    if blocked & positive_weights:
        raise ValueError("fundamental control-chain replay weights include blocked symbols")

    gross = fsum(float(weight) for weight in plan_weights.values())
    if not _same(plan.get("target_exposure"), gross):
        raise ValueError("fundamental control-chain replay plan exposure/weights mismatch")
    for field in ("target_gross_exposure", "target_net_exposure"):
        if field in plan and not _same(plan[field], gross):
            raise ValueError(f"fundamental control-chain replay plan {field} mismatch")
        if field in decision and not _same(decision[field], gross):
            raise ValueError(f"fundamental control-chain replay decision {field} mismatch")
    if "cash_ratio" in plan and not _same(plan["cash_ratio"], max(0.0, 1.0 - gross)):
        raise ValueError("fundamental control-chain replay plan cash ratio mismatch")
    gross_cap = min(
        _number(risk.get("gross_exposure_cap", 1.0), "risk gross cap"),
        _number(risk.get("target_exposure_cap", 1.0), "risk target cap"),
    )
    if gross > gross_cap + 1e-9:
        raise ValueError("fundamental control-chain replay exceeds risk gross cap")
    max_weight = _number(risk.get("max_weight", 1.0), "risk max weight")
    risk_limits = dict(_mapping(risk.get("position_limits", {}), "risk position limits"))
    plan_limits = dict(_mapping(plan.get("position_limits", {}), "plan position limits"))
    for symbol, weight in plan_weights.items():
        symbol_cap = min(
            max_weight,
            _number(risk_limits.get(symbol, max_weight), "risk symbol cap"),
            _number(plan_limits.get(symbol, max_weight), "plan symbol cap"),
        )
        if float(weight) > symbol_cap + 1e-9:
            raise ValueError("fundamental control-chain replay exceeds symbol risk cap")
    if (risk.get("hard_veto") or risk.get("veto")) and positive_weights:
        raise ValueError("fundamental control-chain replay weights violate risk veto")


__all__ = ["validate_control_chain_replay"]
