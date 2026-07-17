"""Pure builders and validators for the v16 candidate-decision report.

The v16 report is deliberately isolated from the v15 reporting stack.  It
accepts only the explicit four-branch contract and can only be persisted below
``results/v16``.  Retrieval evidence is an audit annotation, never a fifth
branch or an input to branch weighting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "candidate_decision_report.v16"
ARCHITECTURE_VERSION = "16.0.0"
BRANCH_SCHEMA_VERSION = "v16.four-branch"
REPORT_PROTOCOL_VERSION = "v16"
RESULTS_NAMESPACE = "results/v16"
REPORT_FILENAME = "v16_candidate_decision_report.json"
REQUIRED_BRANCHES = ("quant", "fundamental", "macro", "llm")
MAX_IC_SELECTED = 12

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BRANCH_STATUSES = frozenset({"ready", "blocked", "unavailable"})
_RETRIEVAL_STATUSES = frozenset({"verified", "partial", "unavailable", "blocked"})
_HANDOFF_STATUSES = frozenset({"complete", "ready", "pending", "blocked", "missing"})
_EXECUTION_STATUSES = frozenset({"authorized", "no_new_risk"})
_IC_ACTIONS = frozenset({"BUY", "HOLD", "AVOID", "SELL"})
_RETRIEVAL_BRANCHES = frozenset({"quant", "fundamental", "macro"})
_WEIGHT_TOLERANCE = 1e-6


class V16CandidateReportError(ValueError):
    """Raised when a candidate report violates the v16 contract."""


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V16CandidateReportError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise V16CandidateReportError(f"{label} fields invalid: {'; '.join(details)}")


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise V16CandidateReportError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise V16CandidateReportError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise V16CandidateReportError(f"{label} must be a finite number")
    return number


def _probability(value: Any, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise V16CandidateReportError(f"{label} must be in [0, 1]")
    return number


def _hash(value: Any, *, label: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    text = str(value or "").strip().lower()
    if not _HASH_PATTERN.fullmatch(text):
        raise V16CandidateReportError(f"{label} must be a lowercase sha256")
    return text


def _strings(values: Any, *, label: str, max_items: int | None = None) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise V16CandidateReportError(f"{label} must be an array")
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            raise V16CandidateReportError(f"{label} must not contain blank values")
        if text in result:
            raise V16CandidateReportError(f"{label} must not contain duplicates")
        result.append(text)
    if max_items is not None and len(result) > max_items:
        raise V16CandidateReportError(f"{label} exceeds maximum {max_items}: got {len(result)}")
    return result


def _stable_strings(values: Any, *, label: str) -> list[str]:
    return sorted(_strings(values, label=label))


def _results_v16_relative(path: Path) -> str:
    parts = path.parts
    if ".." in parts:
        raise V16CandidateReportError("artifact path must not contain parent traversal")
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("results", "v16"):
            return Path(*parts[index:]).as_posix()
    raise V16CandidateReportError(f"artifact path must be below {RESULTS_NAMESPACE}: {path}")


def _validate_artifact_path(value: Any, *, label: str, filename: str | None = None) -> str:
    path = Path(str(value or "").strip())
    relative = _results_v16_relative(path)
    if filename is not None and path.name != filename:
        raise V16CandidateReportError(f"{label} must end with {filename}")
    return relative


def _normalize_branch_contributions(value: Any) -> dict[str, dict[str, Any]]:
    payload = _mapping(value, label="branch_contributions")
    if set(payload) != set(REQUIRED_BRANCHES):
        raise V16CandidateReportError(
            "branch_contributions must contain exactly " + ",".join(REQUIRED_BRANCHES)
        )
    result: dict[str, dict[str, Any]] = {}
    for branch in REQUIRED_BRANCHES:
        item = _mapping(payload[branch], label=f"branch_contributions.{branch}")
        _exact_keys(
            item,
            {"status", "score", "weight", "contribution", "evidence_sha256"},
            label=f"branch_contributions.{branch}",
        )
        status = str(item["status"] or "").strip().lower()
        if status not in _BRANCH_STATUSES:
            raise V16CandidateReportError(f"branch_contributions.{branch}.status is invalid")
        score = _finite_number(item["score"], label=f"{branch}.score")
        if not -1.0 <= score <= 1.0:
            raise V16CandidateReportError(f"branch_contributions.{branch}.score must be in [-1, 1]")
        weight = _probability(item["weight"], label=f"{branch}.weight")
        if not math.isclose(weight, 0.25, abs_tol=1e-12):
            raise V16CandidateReportError(f"branch_contributions.{branch}.weight must equal 0.25")
        contribution = _finite_number(item["contribution"], label=f"{branch}.contribution")
        if not math.isclose(contribution, score * weight, abs_tol=1e-12):
            raise V16CandidateReportError(
                f"branch_contributions.{branch}.contribution must equal score * weight"
            )
        result[branch] = {
            "status": status,
            "score": score,
            "weight": weight,
            "contribution": contribution,
            "evidence_sha256": _hash(item["evidence_sha256"], label=f"{branch}.evidence_sha256"),
        }
    return result


def _normalize_retrieval_evidence(value: Any) -> dict[str, Any]:
    payload = _mapping(value, label="retrieval_evidence")
    _exact_keys(
        payload,
        {"status", "items", "warnings"},
        label="retrieval_evidence",
    )
    items = payload["items"]
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise V16CandidateReportError("retrieval_evidence.items must be an array")
    normalized_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        item = _mapping(raw_item, label=f"retrieval_evidence.items[{index}]")
        _exact_keys(
            item,
            {
                "symbol",
                "branch",
                "supporting_fact_ids",
                "contradicting_fact_ids",
                "conflict_note",
            },
            label=f"retrieval_evidence.items[{index}]",
        )
        symbol = str(item["symbol"] or "").strip().upper()
        branch = str(item["branch"] or "").strip().lower()
        if not symbol:
            raise V16CandidateReportError(
                f"retrieval_evidence.items[{index}].symbol must be non-empty"
            )
        if branch not in _RETRIEVAL_BRANCHES:
            raise V16CandidateReportError(
                "retrieval evidence branch must be quant, fundamental, or macro"
            )
        conflict_note = item["conflict_note"]
        if conflict_note is not None:
            conflict_note = str(conflict_note).strip()
            if not conflict_note:
                raise V16CandidateReportError(
                    "retrieval conflict_note must be non-empty when supplied"
                )
        normalized_items.append(
            {
                "symbol": symbol,
                "branch": branch,
                "supporting_fact_ids": _strings(
                    item["supporting_fact_ids"],
                    label=f"retrieval_evidence.items[{index}].supporting_fact_ids",
                ),
                "contradicting_fact_ids": _strings(
                    item["contradicting_fact_ids"],
                    label=f"retrieval_evidence.items[{index}].contradicting_fact_ids",
                ),
                "conflict_note": conflict_note,
            }
        )
    status = str(payload["status"] or "").strip().lower()
    if status not in _RETRIEVAL_STATUSES:
        raise V16CandidateReportError("retrieval_evidence.status is invalid")
    return {
        "status": status,
        "items": normalized_items,
        "warnings": _stable_strings(payload["warnings"], label="retrieval_evidence.warnings"),
    }


def _normalize_posterior(value: Any) -> dict[str, Any]:
    payload = _mapping(value, label="posterior")
    _exact_keys(
        payload,
        {
            "posterior_win_rate",
            "posterior_expected_alpha",
            "posterior_edge_after_costs",
            "win_rate_interval_90",
            "expected_alpha_interval_90",
        },
        label="posterior",
    )
    win_rate = _probability(payload["posterior_win_rate"], label="posterior.posterior_win_rate")
    expected_alpha = _finite_number(
        payload["posterior_expected_alpha"],
        label="posterior.posterior_expected_alpha",
    )
    raw_edge = payload["posterior_edge_after_costs"]
    edge = (
        None
        if raw_edge is None
        else _finite_number(raw_edge, label="posterior.posterior_edge_after_costs")
    )

    def interval(name: str, estimate: float, *, probability: bool) -> dict[str, float]:
        raw_interval = _mapping(payload[name], label=f"posterior.{name}")
        _exact_keys(raw_interval, {"lower", "upper"}, label=f"posterior.{name}")
        lower = _finite_number(raw_interval["lower"], label=f"posterior.{name}.lower")
        upper = _finite_number(raw_interval["upper"], label=f"posterior.{name}.upper")
        if lower > upper:
            raise V16CandidateReportError(f"posterior.{name} lower must not exceed upper")
        if probability and not (0.0 <= lower <= upper <= 1.0):
            raise V16CandidateReportError("win_rate_interval_90 must be contained in [0, 1]")
        if not lower <= estimate <= upper:
            raise V16CandidateReportError(f"posterior.{name} must contain its point estimate")
        return {"lower": lower, "upper": upper}

    return {
        "posterior_win_rate": win_rate,
        "posterior_expected_alpha": expected_alpha,
        "posterior_edge_after_costs": edge,
        "win_rate_interval_90": interval("win_rate_interval_90", win_rate, probability=True),
        "expected_alpha_interval_90": interval(
            "expected_alpha_interval_90", expected_alpha, probability=False
        ),
    }


def _normalize_risk_advisor(value: Any) -> dict[str, Any]:
    payload = _mapping(value, label="risk_advisor")
    _exact_keys(
        payload,
        {"advisory_only", "warnings", "recommendations"},
        label="risk_advisor",
    )
    if payload["advisory_only"] is not True:
        raise V16CandidateReportError("RiskAdvisor must remain advisory_only")
    return {
        "advisory_only": True,
        "warnings": _stable_strings(payload["warnings"], label="risk_advisor.warnings"),
        "recommendations": _stable_strings(
            payload["recommendations"], label="risk_advisor.recommendations"
        ),
    }


def _normalize_ic(value: Any) -> dict[str, Any]:
    payload = _mapping(value, label="ic")
    _exact_keys(
        payload,
        {"menu_symbols", "actions", "selected_symbols", "cash_ratio"},
        label="ic",
    )
    menu_symbols = [
        symbol.upper() for symbol in _strings(payload["menu_symbols"], label="ic.menu_symbols")
    ]
    if len(menu_symbols) != len(set(menu_symbols)):
        raise V16CandidateReportError("ic.menu_symbols must remain unique after normalization")
    selected = [
        symbol.upper()
        for symbol in _strings(
            payload["selected_symbols"],
            label="ic.selected_symbols",
            max_items=MAX_IC_SELECTED,
        )
    ]
    if len(selected) != len(set(selected)):
        raise V16CandidateReportError("ic.selected_symbols must remain unique after normalization")
    cash_ratio = _probability(payload["cash_ratio"], label="ic.cash_ratio")
    actions = payload["actions"]
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise V16CandidateReportError("ic.actions must be an array")
    normalized_actions: list[dict[str, Any]] = []
    action_symbols: set[str] = set()
    selected_action_symbols: list[str] = []
    total_target_weight = 0.0
    for index, raw_action in enumerate(actions):
        action = _mapping(raw_action, label=f"ic.actions[{index}]")
        _exact_keys(
            action,
            {
                "symbol",
                "action",
                "selected_for_portfolio",
                "existing_weight",
                "target_weight",
                "rationale",
                "risk_acceptance_rationale",
            },
            label=f"ic.actions[{index}]",
        )
        symbol = str(action["symbol"] or "").strip().upper()
        action_name = str(action["action"] or "").strip().upper()
        rationale = str(action["rationale"] or "").strip()
        selected_for_portfolio = action["selected_for_portfolio"]
        if not isinstance(selected_for_portfolio, bool):
            raise V16CandidateReportError(
                f"ic.actions[{index}].selected_for_portfolio must be boolean"
            )
        existing_weight = _probability(
            action["existing_weight"],
            label=f"ic.actions[{index}].existing_weight",
        )
        target_weight = _probability(
            action["target_weight"], label=f"ic.actions[{index}].target_weight"
        )
        risk_rationale = action["risk_acceptance_rationale"]
        if risk_rationale is not None:
            risk_rationale = str(risk_rationale).strip()
            if not risk_rationale:
                raise V16CandidateReportError(
                    "risk_acceptance_rationale must be non-empty when supplied"
                )
        if not symbol or not rationale:
            raise V16CandidateReportError(f"ic.actions[{index}] contains a blank field")
        if action_name not in _IC_ACTIONS:
            raise V16CandidateReportError(f"ic.actions[{index}].action must be BUY/HOLD/AVOID/SELL")
        if symbol in action_symbols:
            raise V16CandidateReportError("ic.actions must contain at most one action per symbol")
        action_symbols.add(symbol)
        positive = target_weight > _WEIGHT_TOLERANCE
        if action_name == "BUY" and (not selected_for_portfolio or not positive):
            raise V16CandidateReportError("BUY requires positive selected target weight")
        if action_name == "HOLD":
            if not math.isclose(target_weight, existing_weight, abs_tol=_WEIGHT_TOLERANCE):
                raise V16CandidateReportError("HOLD target_weight must equal existing_weight")
            if selected_for_portfolio != positive:
                raise V16CandidateReportError(
                    "HOLD selection must agree with its positive existing weight"
                )
        if action_name in {"AVOID", "SELL"} and (selected_for_portfolio or positive):
            raise V16CandidateReportError(f"{action_name} requires zero unselected target weight")
        if selected_for_portfolio:
            selected_action_symbols.append(symbol)
        total_target_weight += target_weight
        normalized_actions.append(
            {
                "symbol": symbol,
                "action": action_name,
                "selected_for_portfolio": selected_for_portfolio,
                "existing_weight": existing_weight,
                "target_weight": target_weight,
                "rationale": rationale,
                "risk_acceptance_rationale": risk_rationale,
            }
        )
    if action_symbols != set(menu_symbols) or len(normalized_actions) != len(menu_symbols):
        raise V16CandidateReportError(
            "ic.actions must contain exactly one action for every menu symbol"
        )
    if selected != selected_action_symbols:
        raise V16CandidateReportError(
            "ic.selected_symbols must exactly match selected_for_portfolio actions"
        )
    if len(selected_action_symbols) > MAX_IC_SELECTED:
        raise V16CandidateReportError(f"IC positive target weights exceed {MAX_IC_SELECTED}")
    if not math.isclose(total_target_weight + cash_ratio, 1.0, abs_tol=_WEIGHT_TOLERANCE):
        raise V16CandidateReportError("IC target weights plus cash_ratio must equal 1")
    return {
        "menu_symbols": menu_symbols,
        "actions": normalized_actions,
        "selected_symbols": selected,
        "cash_ratio": cash_ratio,
    }


def _normalize_handoff(value: Any) -> dict[str, Any]:
    payload = _mapping(value, label="handoff")
    _exact_keys(
        payload,
        {"status", "artifact_path", "artifact_sha256", "blockers"},
        label="handoff",
    )
    status = str(payload["status"] or "").strip().lower()
    if status not in _HANDOFF_STATUSES:
        raise V16CandidateReportError("handoff.status is invalid")
    artifact_path = str(payload["artifact_path"] or "").strip()
    artifact_hash = payload["artifact_sha256"]
    if status in {"complete", "ready"}:
        artifact_path = _validate_artifact_path(artifact_path, label="handoff.artifact_path")
        artifact_hash = _hash(artifact_hash, label="handoff.artifact_sha256")
    else:
        if artifact_path:
            artifact_path = _validate_artifact_path(artifact_path, label="handoff.artifact_path")
        artifact_hash = _hash(artifact_hash, label="handoff.artifact_sha256", nullable=True)
    return {
        "status": status,
        "artifact_path": artifact_path or None,
        "artifact_sha256": artifact_hash,
        "blockers": _stable_strings(payload["blockers"], label="handoff.blockers"),
    }


def _normalize_eligibility(value: Any) -> dict[str, Any]:
    payload = _mapping(value, label="eligibility")
    _exact_keys(payload, {"eligible", "blockers"}, label="eligibility")
    if not isinstance(payload["eligible"], bool):
        raise V16CandidateReportError("eligibility.eligible must be boolean")
    blockers = _stable_strings(payload["blockers"], label="eligibility.blockers")
    if payload["eligible"] and blockers:
        raise V16CandidateReportError("eligible report must not contain eligibility blockers")
    return {"eligible": payload["eligible"], "blockers": blockers}


def _normalize_execution(value: Any) -> dict[str, Any]:
    payload = _mapping(value, label="execution")
    _exact_keys(
        payload,
        {"status", "new_risk_authorized", "broker_side_effects", "blockers"},
        label="execution",
    )
    status = str(payload["status"] or "").strip().lower()
    if status not in _EXECUTION_STATUSES:
        raise V16CandidateReportError("execution.status is invalid")
    if not isinstance(payload["new_risk_authorized"], bool):
        raise V16CandidateReportError("execution.new_risk_authorized must be boolean")
    if payload["broker_side_effects"] is not False:
        raise V16CandidateReportError("v16 report cannot claim broker side effects")
    if payload["new_risk_authorized"] != (status == "authorized"):
        raise V16CandidateReportError("execution status must agree with new_risk_authorized")
    return {
        "status": status,
        "new_risk_authorized": payload["new_risk_authorized"],
        "broker_side_effects": False,
        "blockers": _stable_strings(payload["blockers"], label="execution.blockers"),
    }


def _normalize_readiness(value: Any) -> dict[str, Any]:
    payload = _mapping(value, label="readiness")
    _exact_keys(
        payload,
        {
            "schema_version",
            "path",
            "sha256",
            "new_risk_authorized",
            "blockers",
            "activation_candidate",
            "activation_blockers",
        },
        label="readiness",
    )
    if payload["schema_version"] != "v16_run_readiness.v1":
        raise V16CandidateReportError("readiness must reference v16_run_readiness.v1")
    if not isinstance(payload["new_risk_authorized"], bool):
        raise V16CandidateReportError("readiness.new_risk_authorized must be boolean")
    if not isinstance(payload["activation_candidate"], bool):
        raise V16CandidateReportError("readiness.activation_candidate must be boolean")
    activation_blockers = _stable_strings(
        payload["activation_blockers"], label="readiness.activation_blockers"
    )
    if not payload["activation_candidate"] and not activation_blockers:
        raise V16CandidateReportError("non-candidate activation must include activation_blockers")
    return {
        "schema_version": "v16_run_readiness.v1",
        "path": _validate_artifact_path(
            payload["path"], label="readiness.path", filename="v16_run_readiness.json"
        ),
        "sha256": _hash(payload["sha256"], label="readiness.sha256"),
        "new_risk_authorized": payload["new_risk_authorized"],
        "blockers": _stable_strings(payload["blockers"], label="readiness.blockers"),
        "activation_candidate": payload["activation_candidate"],
        "activation_blockers": activation_blockers,
    }


def build_v16_candidate_decision_report(
    *,
    run_id: str,
    generated_at: str,
    analysis_trade_date: str,
    branch_contributions: Mapping[str, Any],
    retrieval_evidence: Mapping[str, Any],
    posterior: Mapping[str, Any],
    risk_advisor: Mapping[str, Any],
    ic: Mapping[str, Any],
    handoff: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    execution: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a normalized v16 candidate decision without external I/O."""

    run_id_text = str(run_id or "").strip()
    generated_at_text = str(generated_at or "").strip()
    trade_date_text = str(analysis_trade_date or "").strip()
    if not run_id_text or not generated_at_text or not trade_date_text:
        raise V16CandidateReportError("run_id, generated_at, and analysis_trade_date are required")
    normalized_handoff = _normalize_handoff(handoff)
    normalized_eligibility = _normalize_eligibility(eligibility)
    normalized_execution = _normalize_execution(execution)
    normalized_readiness = _normalize_readiness(readiness)
    if normalized_execution["new_risk_authorized"] != normalized_readiness["new_risk_authorized"]:
        raise V16CandidateReportError("execution and readiness new-risk authorization must agree")
    if normalized_execution["new_risk_authorized"]:
        if not normalized_readiness["activation_candidate"]:
            raise V16CandidateReportError("authorized execution requires v16 activation candidate")
        if not normalized_eligibility["eligible"]:
            raise V16CandidateReportError("authorized execution must be eligible")
        if normalized_handoff["status"] not in {"complete", "ready"}:
            raise V16CandidateReportError("authorized execution requires complete handoff")
        if normalized_readiness["blockers"] or normalized_execution["blockers"]:
            raise V16CandidateReportError("authorized execution must not contain blockers")

    report = {
        "schema_version": SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "results_namespace": RESULTS_NAMESPACE,
        "run_id": run_id_text,
        "generated_at": generated_at_text,
        "analysis_trade_date": trade_date_text,
        "branch_contributions": _normalize_branch_contributions(branch_contributions),
        "retrieval_evidence": _normalize_retrieval_evidence(retrieval_evidence),
        "posterior": _normalize_posterior(posterior),
        "risk_advisor": _normalize_risk_advisor(risk_advisor),
        "ic": _normalize_ic(ic),
        "handoff": normalized_handoff,
        "eligibility": normalized_eligibility,
        "execution": normalized_execution,
        "readiness": normalized_readiness,
    }
    validate_v16_candidate_decision_report(report)
    return report


def validate_v16_candidate_decision_report(payload: Mapping[str, Any]) -> None:
    """Validate the full version envelope and semantic v16 constraints."""

    report = _mapping(payload, label="candidate decision report")
    expected_fields = {
        "schema_version",
        "architecture_version",
        "branch_schema_version",
        "report_protocol_version",
        "results_namespace",
        "run_id",
        "generated_at",
        "analysis_trade_date",
        "branch_contributions",
        "retrieval_evidence",
        "posterior",
        "risk_advisor",
        "ic",
        "handoff",
        "eligibility",
        "execution",
        "readiness",
    }
    _exact_keys(report, expected_fields, label="candidate decision report")
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "results_namespace": RESULTS_NAMESPACE,
    }
    for field, expected in envelope.items():
        if report[field] != expected:
            raise V16CandidateReportError(
                f"{field} mismatch: expected {expected!r}, got {report[field]!r}"
            )
    identity_fields = ("run_id", "generated_at", "analysis_trade_date")
    if not all(
        isinstance(report[field], str) and report[field].strip() for field in identity_fields
    ):
        raise V16CandidateReportError("run_id, generated_at, and analysis_trade_date are required")

    normalized_sections = {
        "branch_contributions": _normalize_branch_contributions(report["branch_contributions"]),
        "retrieval_evidence": _normalize_retrieval_evidence(report["retrieval_evidence"]),
        "posterior": _normalize_posterior(report["posterior"]),
        "risk_advisor": _normalize_risk_advisor(report["risk_advisor"]),
        "ic": _normalize_ic(report["ic"]),
        "handoff": _normalize_handoff(report["handoff"]),
        "eligibility": _normalize_eligibility(report["eligibility"]),
        "execution": _normalize_execution(report["execution"]),
        "readiness": _normalize_readiness(report["readiness"]),
    }
    for field, normalized in normalized_sections.items():
        if report[field] != normalized:
            raise V16CandidateReportError(f"{field} is not canonical v16 data")
    handoff = normalized_sections["handoff"]
    eligibility = normalized_sections["eligibility"]
    execution = normalized_sections["execution"]
    readiness = normalized_sections["readiness"]
    if execution["new_risk_authorized"] != readiness["new_risk_authorized"]:
        raise V16CandidateReportError("execution and readiness new-risk authorization must agree")
    if execution["new_risk_authorized"] and (
        not readiness["activation_candidate"]
        or not eligibility["eligible"]
        or handoff["status"] not in {"complete", "ready"}
        or readiness["blockers"]
        or execution["blockers"]
    ):
        raise V16CandidateReportError("authorized report has unresolved gates")


def report_reference(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    validate_v16_candidate_decision_report(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "path": _validate_artifact_path(
            path, label="candidate report path", filename=REPORT_FILENAME
        ),
        "sha256": canonical_sha256(payload),
    }


def write_v16_candidate_decision_report(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Atomically write owner-only v16 JSON below ``results/v16``."""

    validate_v16_candidate_decision_report(payload)
    reference = report_reference(path, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise RuntimeError("v16 candidate report readback hash mismatch")
    return reference


def load_v16_candidate_decision_report(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    _validate_artifact_path(path, label="candidate report path", filename=REPORT_FILENAME)
    if path.is_symlink() or not path.is_file():
        raise V16CandidateReportError("v16 candidate report must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise V16CandidateReportError("v16 candidate report must be a JSON object")
    validate_v16_candidate_decision_report(payload)
    if canonical_sha256(payload) != str(expected_sha256 or "").strip().lower():
        raise V16CandidateReportError("v16 candidate report sha256 mismatch")
    return payload


__all__ = [
    "ARCHITECTURE_VERSION",
    "BRANCH_SCHEMA_VERSION",
    "MAX_IC_SELECTED",
    "REPORT_FILENAME",
    "REPORT_PROTOCOL_VERSION",
    "REQUIRED_BRANCHES",
    "RESULTS_NAMESPACE",
    "SCHEMA_VERSION",
    "V16CandidateReportError",
    "build_v16_candidate_decision_report",
    "canonical_sha256",
    "load_v16_candidate_decision_report",
    "report_reference",
    "validate_v16_candidate_decision_report",
    "write_v16_candidate_decision_report",
]
