"""Strict offline canonical replay for FactorGovernanceProtocol v4.

This is a new wire contract.  v2/v3 evidence is rejected and never upgraded.
Every arm binds Eligibility -> Quant -> Funnel -> CodexS1 -> Bayesian ->
RiskAdvisor -> CodexIC -> PortfolioConstructor by exact predecessor byte and
semantic hashes.  RiskAdvisor is advisory: positive portfolio weight is bound
to a CodexIC BUY, not to RiskAdvisor approval.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import statistics
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

from quant_investor.factors.governance_protocol_v4 import (
    CONTROL_CHAIN_STAGES,
    FACTOR_EVIDENCE_SCHEMA_VERSION,
    OPEN_SESSION_CALENDAR_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    TARGET_PRODUCTION_FACTOR_COUNT,
    validate_open_session_calendar_v4,
)

REPLAY_SCHEMA_VERSION = "factor-governance-canonical-replay.v4"
STAGE_SCHEMA_VERSION = "factor-governance-canonical-stage.v4"
CALENDAR_SCHEMA_VERSION = OPEN_SESSION_CALENDAR_SCHEMA_VERSION
EVIDENCE_SCHEMA_VERSION = FACTOR_EVIDENCE_SCHEMA_VERSION
ARM_NAMES = ("A", "B", "C", "D")
GENESIS_SHA256 = "0" * 64
MAX_REPLAY_FILE_BYTES = 24 * 1024 * 1024
MIN_PAIRED_COHORT_COUNT = 8
PAIRED_COHORT_OPEN_SESSION_COUNT = 30
COMPARISON_METHOD = "student_t"
COMPARISON_SIDEDNESS = "two_sided"
CONTEXT_BINDING_FIELDS = (
    "eligibility_contract_sha256",
    "calendar_sha256",
    "pit_sha256",
    "runtime_contract_sha256",
    "latest_pointer_sha256",
    "manifest_sha256",
    "market_data_input_sha256",
    "candidate_catalog_sha256",
    "screening_evidence_sha256",
    "dedup_evidence_sha256",
    "quantitative_evidence_sha256",
)


class CanonicalReplayV4Error(ValueError):
    """Raised when a v4 replay or evidence envelope fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise CanonicalReplayV4Error(f"value is not canonical JSON: {exc}") from exc


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def byte_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def stage_byte_sha256(
    *,
    arm: str,
    stage: str,
    context_sha256: str,
    predecessor: Mapping[str, Any],
    output: Any,
) -> str:
    return byte_sha256(
        {
            "schema_version": STAGE_SCHEMA_VERSION,
            "arm": arm,
            "stage": stage,
            "context_sha256": context_sha256,
            "predecessor": dict(predecessor),
            "output": output,
        }
    )


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CanonicalReplayV4Error(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        raise CanonicalReplayV4Error(f"{label} fields invalid: {';'.join(detail)}")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CanonicalReplayV4Error(f"{label} must be an exact non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CanonicalReplayV4Error(f"{label} must be lowercase SHA-256")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalReplayV4Error(f"{label} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CanonicalReplayV4Error(f"{label} must be finite numeric")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalReplayV4Error(f"{label} must be an integer")
    return value


def _iso_date(value: Any, label: str) -> str:
    text = _text(value, label)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise CanonicalReplayV4Error(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != text:
        raise CanonicalReplayV4Error(f"{label} must be a canonical ISO date")
    return text


def _same_number(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)


def _validate_open_session_calendar(
    value: Any,
    *,
    context: Mapping[str, str],
) -> dict[str, Any]:
    try:
        normalized = validate_open_session_calendar_v4(
            value,
            expected_calendar_sha256=context["calendar_sha256"],
            expected_latest_pointer_sha256=context["latest_pointer_sha256"],
            expected_manifest_sha256=context["manifest_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalReplayV4Error(str(exc)) from exc
    return {key: item for key, item in normalized.items() if key != "calendar_sha256"}


def _symbols(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise CanonicalReplayV4Error(f"{label} must be a list")
    result = [_text(item, f"{label}[]") for item in value]
    if result != sorted(result) or len(result) != len(set(result)):
        raise CanonicalReplayV4Error(f"{label} must be sorted and distinct")
    return result


def _numeric_domain(value: Any, domain: list[str], label: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(domain):
        raise CanonicalReplayV4Error(f"{label} symbol domain mismatch")
    return {symbol: _finite(value[symbol], f"{label} {symbol}") for symbol in domain}


def _hash_domain(value: Any, domain: list[str], label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(domain):
        raise CanonicalReplayV4Error(f"{label} symbol domain mismatch")
    return {symbol: _sha(value[symbol], f"{label} {symbol}") for symbol in domain}


def _validate_factor_records(
    output: Mapping[str, Any], arm: str
) -> tuple[list[str], dict[str, dict[str, str]]]:
    selected = _symbols(output["selected_factors"], f"arm {arm} selected_factors")
    records = output["factor_records"]
    if not isinstance(records, list):
        raise CanonicalReplayV4Error("factor_records must be a list")
    normalized: dict[str, dict[str, str]] = {}
    fields = {"name", "family", "slot", "state", "registry_record_sha256"}
    for index, raw in enumerate(records):
        row = _exact(raw, fields, f"arm {arm} factor_records[{index}]")
        name = _text(row["name"], "factor name")
        if name in normalized:
            raise CanonicalReplayV4Error("factor record names must be distinct")
        normalized[name] = {
            "name": name,
            "family": _text(row["family"], "factor family"),
            "slot": _text(row["slot"], "factor slot"),
            "state": _text(row["state"], "factor state"),
            "registry_record_sha256": _sha(
                row["registry_record_sha256"], "factor registry record SHA"
            ),
        }
    if set(normalized) != set(selected):
        raise CanonicalReplayV4Error(f"arm {arm} selected_factors/factor_records mismatch")
    return selected, normalized


def _validate_stage_output(
    stage: str,
    output: Any,
    *,
    arm: str,
    prior: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if stage == "eligibility":
        payload = _exact(
            output,
            {"schema_version", "eligible_symbols", "eligibility_contract_sha256"},
            "Eligibility output",
        )
        if payload["schema_version"] != "factor-governance-eligibility-output.v4":
            raise CanonicalReplayV4Error("unsupported Eligibility output schema")
        return {
            "domain": _symbols(payload["eligible_symbols"], "Eligibility symbols"),
            "eligibility_contract_sha256": _sha(
                payload["eligibility_contract_sha256"], "Eligibility contract SHA"
            ),
        }

    eligibility_domain = list((prior.get("eligibility") or {}).get("domain") or [])
    if "eligibility" not in prior:
        raise CanonicalReplayV4Error("Eligibility domain is unavailable")
    if stage == "quant":
        payload = _exact(
            output,
            {"schema_version", "scored_symbols", "selected_factors", "factor_records"},
            "Quant output",
        )
        if payload["schema_version"] != "factor-governance-quant-output.v4":
            raise CanonicalReplayV4Error("unsupported Quant output schema")
        scored = _symbols(payload["scored_symbols"], "Quant scored symbols")
        if not set(scored).issubset(eligibility_domain):
            raise CanonicalReplayV4Error("Quant scored outside Eligibility domain")
        selected, records = _validate_factor_records(payload, arm)
        return {"domain": scored, "selected_factors": selected, "factor_records": records}

    quant_domain = list((prior.get("quant") or {}).get("domain") or [])
    if "quant" not in prior:
        raise CanonicalReplayV4Error("Quant domain is unavailable")
    if stage == "funnel":
        payload = _exact(
            output,
            {"schema_version", "eligible_symbols"},
            "Funnel output",
        )
        if payload["schema_version"] != "factor-governance-funnel-output.v4":
            raise CanonicalReplayV4Error("unsupported Funnel output schema")
        domain = _symbols(payload["eligible_symbols"], "Funnel eligible symbols")
        if not set(domain).issubset(quant_domain):
            raise CanonicalReplayV4Error("Funnel contains a symbol not scored by Quant")
        return {"domain": domain}

    domain = list((prior.get("funnel") or {}).get("domain") or [])
    if "funnel" not in prior:
        raise CanonicalReplayV4Error("Funnel domain is unavailable")
    if stage == "codex_s1":
        payload = _exact(
            output,
            {"schema_version", "advisory_scores"},
            "CodexS1 output",
        )
        if payload["schema_version"] != "factor-governance-codex-s1-output.v4":
            raise CanonicalReplayV4Error("unsupported CodexS1 output schema")
        return {
            "domain": domain,
            "advisory_scores": _numeric_domain(
                payload["advisory_scores"], domain, "CodexS1 advisory scores"
            ),
        }

    if stage == "bayesian":
        payload = _exact(
            output,
            {"schema_version", "posterior_scores", "codex_s1_semantic_sha256"},
            "Bayesian output",
        )
        if payload["schema_version"] != "factor-governance-bayesian-output.v4":
            raise CanonicalReplayV4Error("unsupported Bayesian output schema")
        if payload["codex_s1_semantic_sha256"] != prior["codex_s1"]["semantic_sha256"]:
            raise CanonicalReplayV4Error("Bayesian is not bound to CodexS1 output")
        return {
            "domain": domain,
            "posterior_scores": _numeric_domain(
                payload["posterior_scores"], domain, "Bayesian posterior scores"
            ),
        }

    if stage == "risk_advisor":
        payload = _exact(
            output,
            {
                "schema_version",
                "advisory_only",
                "decisions",
                "bayesian_semantic_sha256",
            },
            "RiskAdvisor output",
        )
        if payload["schema_version"] != "factor-governance-risk-advisor-output.v4":
            raise CanonicalReplayV4Error("unsupported RiskAdvisor output schema")
        if payload["advisory_only"] is not True:
            raise CanonicalReplayV4Error("RiskAdvisor must be advisory-only")
        if payload["bayesian_semantic_sha256"] != prior["bayesian"]["semantic_sha256"]:
            raise CanonicalReplayV4Error("RiskAdvisor is not bound to Bayesian output")
        decisions = payload["decisions"]
        if not isinstance(decisions, dict) or set(decisions) != set(domain):
            raise CanonicalReplayV4Error("RiskAdvisor symbol domain mismatch")
        normalized: dict[str, str] = {}
        for symbol in domain:
            decision = _text(decisions[symbol], f"RiskAdvisor decision {symbol}")
            if decision not in {"approve", "reject", "watch"}:
                raise CanonicalReplayV4Error("RiskAdvisor decision is invalid")
            normalized[symbol] = decision
        return {"domain": domain, "decisions": normalized, "advisory_only": True}

    if stage == "codex_ic":
        payload = _exact(
            output,
            {
                "schema_version",
                "inputs",
                "input_sha256s",
                "decisions",
                "output_sha256s",
            },
            "CodexIC output",
        )
        if payload["schema_version"] != "factor-governance-codex-ic-output.v4":
            raise CanonicalReplayV4Error("unsupported CodexIC output schema")
        inputs = payload["inputs"]
        decisions = payload["decisions"]
        if not isinstance(inputs, dict) or set(inputs) != set(domain):
            raise CanonicalReplayV4Error("CodexIC input symbol domain mismatch")
        if not isinstance(decisions, dict) or set(decisions) != set(domain):
            raise CanonicalReplayV4Error("CodexIC output symbol domain mismatch")
        input_hashes = _hash_domain(payload["input_sha256s"], domain, "CodexIC input SHA")
        output_hashes = _hash_domain(payload["output_sha256s"], domain, "CodexIC output SHA")
        upstream_stages = CONTROL_CHAIN_STAGES[: CONTROL_CHAIN_STAGES.index("codex_ic")]
        expected_upstream = {
            stage_name: prior[stage_name]["semantic_sha256"] for stage_name in upstream_stages
        }
        actions: dict[str, str] = {}
        for symbol in domain:
            ic_input = _exact(
                inputs[symbol],
                {"symbol", "upstream_stage_sha256s", "ic_hints"},
                f"CodexIC input {symbol}",
            )
            if ic_input["symbol"] != symbol:
                raise CanonicalReplayV4Error("CodexIC input symbol mismatch")
            if ic_input["upstream_stage_sha256s"] != expected_upstream:
                raise CanonicalReplayV4Error("CodexIC upstream chain binding mismatch")
            if ic_input["ic_hints"] != {}:
                raise CanonicalReplayV4Error("CodexIC deterministic input requires empty hints")
            if input_hashes[symbol] != semantic_sha256(ic_input):
                raise CanonicalReplayV4Error("CodexIC input SHA mismatch")
            decision = decisions[symbol]
            if not isinstance(decision, dict):
                raise CanonicalReplayV4Error("CodexIC decision must be an object")
            if decision.get("symbol") != symbol:
                raise CanonicalReplayV4Error("CodexIC decision symbol mismatch")
            action = _text(decision.get("action"), f"CodexIC action {symbol}")
            if action not in {"buy", "hold", "sell", "watch", "avoid"}:
                raise CanonicalReplayV4Error("CodexIC action is invalid")
            if output_hashes[symbol] != semantic_sha256(decision):
                raise CanonicalReplayV4Error("CodexIC output SHA mismatch")
            actions[symbol] = action
        return {
            "domain": domain,
            "input_sha256s": input_hashes,
            "output_sha256s": output_hashes,
            "actions": actions,
        }

    payload = _exact(
        output,
        {"schema_version", "target_weights", "codex_ic_decision_sha256s"},
        "PortfolioConstructor output",
    )
    if payload["schema_version"] != "factor-governance-portfolio-output.v4":
        raise CanonicalReplayV4Error("unsupported PortfolioConstructor output schema")
    weights = payload["target_weights"]
    if not isinstance(weights, dict) or not set(weights).issubset(domain):
        raise CanonicalReplayV4Error("PortfolioConstructor contains an unknown symbol")
    normalized_weights = {
        symbol: _finite(weight, f"target weight {symbol}") for symbol, weight in weights.items()
    }
    if any(weight < 0.0 or weight > 1.0 for weight in normalized_weights.values()):
        raise CanonicalReplayV4Error("portfolio weights must be in [0,1]")
    if sum(normalized_weights.values()) > 1.0 + 1e-12:
        raise CanonicalReplayV4Error("portfolio weights exceed one")
    ic_result = prior["codex_ic"]
    ic_hashes = _hash_domain(
        payload["codex_ic_decision_sha256s"],
        domain,
        "PortfolioConstructor CodexIC decision SHA",
    )
    if ic_hashes != ic_result["output_sha256s"]:
        raise CanonicalReplayV4Error("PortfolioConstructor is not bound to CodexIC output")
    if any(
        weight > 0.0 and ic_result["actions"].get(symbol) != "buy"
        for symbol, weight in normalized_weights.items()
    ):
        raise CanonicalReplayV4Error("positive portfolio weight lacks CodexIC BUY")
    # Deliberately do not inspect RiskAdvisor approval here.  Its bytes are
    # still transitively bound through CodexIC's upstream-stage hash map.
    return {
        "domain": domain,
        "target_weights": normalized_weights,
        "codex_ic_decision_sha256s": ic_hashes,
        "positive_weight_depends_on_risk_advisor_approval": False,
    }


def _validate_paired_cohorts(
    value: Any,
    *,
    calendar_sha256: str,
    pit_sha256: str,
    calendar_open_session_dates: list[str],
) -> tuple[list[dict[str, Any]], list[float]]:
    if not isinstance(value, list) or len(value) < MIN_PAIRED_COHORT_COUNT:
        raise CanonicalReplayV4Error(
            f"comparison cohorts must contain at least {MIN_PAIRED_COHORT_COUNT} items"
        )
    normalized: list[dict[str, Any]] = []
    paired_deltas: list[float] = []
    seen_cohort_ids: set[str] = set()
    seen_open_sessions: set[str] = set()
    calendar_index = {
        session: index for index, session in enumerate(calendar_open_session_dates)
    }
    bound_universe_sha256: str | None = None
    cohort_fields = {
        "cohort_id",
        "open_session_dates",
        "universe_sha256",
        "calendar_sha256",
        "pit_sha256",
        "arms",
    }
    arm_fields = {
        "gross_return",
        "turnover",
        "cost_rate",
        "cost_return",
        "net_return",
    }
    for cohort_index, raw in enumerate(value):
        cohort = _exact(raw, cohort_fields, f"comparison cohorts[{cohort_index}]")
        cohort_id = _text(cohort["cohort_id"], f"cohort[{cohort_index}].cohort_id")
        if cohort_id in seen_cohort_ids:
            raise CanonicalReplayV4Error("comparison cohort ids must be distinct")
        seen_cohort_ids.add(cohort_id)

        raw_dates = cohort["open_session_dates"]
        if not isinstance(raw_dates, list) or len(raw_dates) != PAIRED_COHORT_OPEN_SESSION_COUNT:
            raise CanonicalReplayV4Error(
                "each comparison cohort must contain exactly 30 open_session_dates"
            )
        open_session_dates = [
            _iso_date(item, f"cohort[{cohort_index}].open_session_dates[]")
            for item in raw_dates
        ]
        if open_session_dates != sorted(open_session_dates) or len(
            set(open_session_dates)
        ) != len(open_session_dates):
            raise CanonicalReplayV4Error(
                "cohort open_session_dates must be sorted and distinct"
            )
        if any(session not in calendar_index for session in open_session_dates):
            raise CanonicalReplayV4Error(
                "cohort open_session_dates must exist in the bound calendar"
            )
        session_indexes = [calendar_index[session] for session in open_session_dates]
        if session_indexes != list(
            range(session_indexes[0], session_indexes[0] + PAIRED_COHORT_OPEN_SESSION_COUNT)
        ):
            raise CanonicalReplayV4Error(
                "cohort open_session_dates must be 30 consecutive calendar sessions"
            )
        if seen_open_sessions.intersection(open_session_dates):
            raise CanonicalReplayV4Error("comparison cohorts must not overlap")
        seen_open_sessions.update(open_session_dates)

        universe_sha256 = _sha(
            cohort["universe_sha256"], f"cohort[{cohort_index}].universe_sha256"
        )
        if bound_universe_sha256 is None:
            bound_universe_sha256 = universe_sha256
        elif universe_sha256 != bound_universe_sha256:
            raise CanonicalReplayV4Error("comparison cohort universe SHA binding mismatch")
        if cohort["calendar_sha256"] != calendar_sha256:
            raise CanonicalReplayV4Error("comparison cohort calendar SHA binding mismatch")
        if cohort["pit_sha256"] != pit_sha256:
            raise CanonicalReplayV4Error("comparison cohort PIT SHA binding mismatch")
        _sha(cohort["calendar_sha256"], f"cohort[{cohort_index}].calendar_sha256")
        _sha(cohort["pit_sha256"], f"cohort[{cohort_index}].pit_sha256")

        arms = cohort["arms"]
        if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
            raise CanonicalReplayV4Error("comparison cohort arms must be exactly A/B/C/D")
        normalized_arms: dict[str, dict[str, float]] = {}
        for arm in ARM_NAMES:
            arm_value = _exact(
                arms[arm], arm_fields, f"cohort[{cohort_index}].arms.{arm}"
            )
            gross_return = _finite(
                arm_value["gross_return"],
                f"cohort[{cohort_index}].arms.{arm}.gross_return",
            )
            turnover = _finite(
                arm_value["turnover"], f"cohort[{cohort_index}].arms.{arm}.turnover"
            )
            cost_rate = _finite(
                arm_value["cost_rate"], f"cohort[{cohort_index}].arms.{arm}.cost_rate"
            )
            cost_return = _finite(
                arm_value["cost_return"],
                f"cohort[{cohort_index}].arms.{arm}.cost_return",
            )
            net_return = _finite(
                arm_value["net_return"], f"cohort[{cohort_index}].arms.{arm}.net_return"
            )
            if turnover < 0.0 or cost_rate < 0.0:
                raise CanonicalReplayV4Error("turnover and cost_rate must be non-negative")
            expected_cost_return = turnover * cost_rate
            if not _same_number(cost_return, expected_cost_return):
                raise CanonicalReplayV4Error("comparison arm cost_return recomputation mismatch")
            expected_net_return = gross_return - expected_cost_return
            if not _same_number(net_return, expected_net_return):
                raise CanonicalReplayV4Error("comparison arm net_return recomputation mismatch")
            normalized_arms[arm] = {
                "gross_return": gross_return,
                "turnover": turnover,
                "cost_rate": cost_rate,
                "cost_return": cost_return,
                "net_return": net_return,
            }
        paired_deltas.append(
            normalized_arms["D"]["net_return"]
            - normalized_arms["A"]["net_return"]
        )
        normalized.append(
            {
                "cohort_id": cohort_id,
                "open_session_dates": open_session_dates,
                "universe_sha256": universe_sha256,
                "calendar_sha256": calendar_sha256,
                "pit_sha256": pit_sha256,
                "arms": normalized_arms,
            }
        )
    return normalized, paired_deltas


def _recompute_comparison_statistics(paired_deltas: list[float]) -> dict[str, float | int]:
    n = len(paired_deltas)
    mean = statistics.fmean(paired_deltas)
    stddev = statistics.stdev(paired_deltas)
    standard_error = stddev / math.sqrt(n)
    critical_value = float(student_t.ppf(0.975, n - 1))
    if not math.isfinite(critical_value):
        raise CanonicalReplayV4Error("Student-t critical value recomputation failed")
    return {
        "n": n,
        "mean": mean,
        "stddev": stddev,
        "standard_error": standard_error,
        "critical_value": critical_value,
        "incremental_edge_ci95_lower": mean - critical_value * standard_error,
    }


def validate_canonical_replay_v4(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact 32-stage A/B/C/D v4 replay."""

    if not isinstance(value, Mapping) or value.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise CanonicalReplayV4Error("unsupported canonical replay schema")
    payload = _exact(
        dict(value),
        {
            "schema_version",
            "protocol_version",
            "run_id",
            "as_of",
            "registry_file_sha256",
            "production_factor_set_sha256",
            "context",
            "context_sha256",
            "calendar",
            "factor_set",
            "comparison",
            "stages",
        },
        "canonical replay",
    )
    if payload["schema_version"] != REPLAY_SCHEMA_VERSION:
        raise CanonicalReplayV4Error("unsupported canonical replay schema")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise CanonicalReplayV4Error("canonical replay protocol version mismatch")
    _text(payload["run_id"], "run_id")
    _text(payload["as_of"], "as_of")
    registry_sha = _sha(payload["registry_file_sha256"], "registry_file_sha256")
    factor_set_sha = _sha(payload["production_factor_set_sha256"], "production_factor_set_sha256")
    context = _exact(
        payload["context"],
        set(CONTEXT_BINDING_FIELDS),
        "replay context",
    )
    for key, item in context.items():
        _sha(item, f"context {key}")
    calendar = _validate_open_session_calendar(payload["calendar"], context=context)
    context_sha = _sha(payload["context_sha256"], "context_sha256")
    expected_context_sha = semantic_sha256(
        {
            "registry_file_sha256": registry_sha,
            "production_factor_set_sha256": factor_set_sha,
            **dict(context),
        }
    )
    if context_sha != expected_context_sha:
        raise CanonicalReplayV4Error("canonical replay context SHA mismatch")
    factor_set = _symbols(payload["factor_set"], "factor_set")
    from quant_investor.factors.runtime import production_factor_set_sha256

    if factor_set_sha != production_factor_set_sha256(factor_set):
        raise CanonicalReplayV4Error("canonical replay production factor-set SHA mismatch")
    comparison = _exact(
        payload["comparison"],
        {
            "transition_mode",
            "incumbent",
            "challenger",
            "slot",
            "cohorts",
            "n",
            "mean",
            "stddev",
            "standard_error",
            "critical_value",
            "method",
            "sidedness",
            "incremental_edge_ci95_lower",
        },
        "comparison",
    )
    transition_mode = _text(comparison["transition_mode"], "transition_mode")
    if transition_mode not in {"add", "replace"}:
        raise CanonicalReplayV4Error("transition_mode must be add or replace")
    incumbent: str | None
    if transition_mode == "add":
        if comparison["incumbent"] is not None:
            raise CanonicalReplayV4Error("add transition incumbent must be null")
        incumbent = None
        if len(factor_set) >= TARGET_PRODUCTION_FACTOR_COUNT:
            raise CanonicalReplayV4Error("add transition requires an underfilled factor set")
    else:
        incumbent = _text(comparison["incumbent"], "incumbent")
        if len(factor_set) != TARGET_PRODUCTION_FACTOR_COUNT:
            raise CanonicalReplayV4Error("replace transition requires exactly 10 factors")
    challenger = _text(comparison["challenger"], "challenger")
    slot = _text(comparison["slot"], "slot")
    if challenger in factor_set:
        raise CanonicalReplayV4Error("challenger must not already be in the factor set")
    if transition_mode == "replace" and (
        incumbent == challenger or incumbent not in factor_set
    ):
        raise CanonicalReplayV4Error("comparison does not identify one replacement")

    cohorts, paired_deltas = _validate_paired_cohorts(
        comparison["cohorts"],
        calendar_sha256=context["calendar_sha256"],
        pit_sha256=context["pit_sha256"],
        calendar_open_session_dates=calendar["open_session_dates"],
    )
    expected_quantitative_sha256 = semantic_sha256(
        {
            "transition_mode": transition_mode,
            "incumbent": incumbent,
            "challenger": challenger,
            "slot": slot,
            "cohorts": comparison["cohorts"],
        }
    )
    if context["quantitative_evidence_sha256"] != expected_quantitative_sha256:
        raise CanonicalReplayV4Error("quantitative evidence semantic SHA mismatch")
    recomputed = _recompute_comparison_statistics(paired_deltas)
    declared_n = _integer(comparison["n"], "comparison n")
    if declared_n != recomputed["n"]:
        raise CanonicalReplayV4Error("comparison n recomputation mismatch")
    declared_statistics: dict[str, float] = {}
    for field in (
        "mean",
        "stddev",
        "standard_error",
        "critical_value",
        "incremental_edge_ci95_lower",
    ):
        declared = _finite(comparison[field], f"comparison {field}")
        expected = float(recomputed[field])
        if not _same_number(declared, expected):
            raise CanonicalReplayV4Error(f"comparison {field} recomputation mismatch")
        declared_statistics[field] = declared
    if comparison["method"] != COMPARISON_METHOD:
        raise CanonicalReplayV4Error("comparison method must be student_t")
    if comparison["sidedness"] != COMPARISON_SIDEDNESS:
        raise CanonicalReplayV4Error("comparison sidedness must be two_sided")
    edge_lower = declared_statistics["incremental_edge_ci95_lower"]
    if transition_mode == "replace" and edge_lower <= 0.0:
        raise CanonicalReplayV4Error(
            "target-10 replacement incremental edge CI95 lower must be positive"
        )

    stages = payload["stages"]
    expected_order = [(arm, stage_name) for arm in ARM_NAMES for stage_name in CONTROL_CHAIN_STAGES]
    if not isinstance(stages, list) or len(stages) != len(expected_order):
        raise CanonicalReplayV4Error("canonical replay must contain exactly 32 stages")
    fields = {
        "schema_version",
        "arm",
        "stage",
        "context_sha256",
        "byte_sha256",
        "semantic_sha256",
        "predecessor",
        "output",
    }
    results: dict[str, dict[str, Any]] = {arm: {} for arm in ARM_NAMES}
    seen_byte_hashes: set[str] = set()
    for index, raw in enumerate(stages):
        row = _exact(raw, fields, f"stage[{index}]")
        arm, stage_name = expected_order[index]
        if row["schema_version"] != STAGE_SCHEMA_VERSION:
            raise CanonicalReplayV4Error("unsupported canonical stage schema")
        if row["arm"] != arm or row["stage"] != stage_name:
            raise CanonicalReplayV4Error("stage order/identity mismatch")
        if row["context_sha256"] != context_sha:
            raise CanonicalReplayV4Error("stage context SHA mismatch")
        predecessor = _exact(
            row["predecessor"],
            {"kind", "byte_sha256", "semantic_sha256"},
            "stage predecessor",
        )
        if stage_name == CONTROL_CHAIN_STAGES[0]:
            expected_predecessor = {
                "kind": "genesis",
                "byte_sha256": GENESIS_SHA256,
                "semantic_sha256": GENESIS_SHA256,
            }
        else:
            prior_stage = CONTROL_CHAIN_STAGES[CONTROL_CHAIN_STAGES.index(stage_name) - 1]
            previous = results[arm][prior_stage]
            expected_predecessor = {
                "kind": "stage",
                "byte_sha256": previous["byte_sha256"],
                "semantic_sha256": previous["semantic_sha256"],
            }
        if predecessor != expected_predecessor:
            raise CanonicalReplayV4Error("stage predecessor byte/semantic mismatch")
        byte_sha = _sha(row["byte_sha256"], "stage byte SHA")
        if byte_sha != stage_byte_sha256(
            arm=arm,
            stage=stage_name,
            context_sha256=context_sha,
            predecessor=predecessor,
            output=row["output"],
        ):
            raise CanonicalReplayV4Error("stage byte SHA mismatch")
        if byte_sha in seen_byte_hashes:
            raise CanonicalReplayV4Error("stage byte identities must be globally unique")
        seen_byte_hashes.add(byte_sha)
        semantic_sha = _sha(row["semantic_sha256"], "stage semantic SHA")
        if semantic_sha != semantic_sha256(row["output"]):
            raise CanonicalReplayV4Error("stage semantic SHA mismatch")
        normalized = _validate_stage_output(stage_name, row["output"], arm=arm, prior=results[arm])
        results[arm][stage_name] = {
            **normalized,
            "byte_sha256": byte_sha,
            "semantic_sha256": semantic_sha,
        }

    selected = {arm: results[arm]["quant"]["selected_factors"] for arm in ARM_NAMES}
    if transition_mode == "add":
        with_challenger = sorted([*factor_set, challenger])
        expected_sets = {
            "A": factor_set,
            "B": factor_set,
            "C": with_challenger,
            "D": with_challenger,
        }
    else:
        assert incumbent is not None
        without_incumbent = sorted(name for name in factor_set if name != incumbent)
        with_challenger = sorted([*without_incumbent, challenger])
        expected_sets = {
            "A": factor_set,
            "B": without_incumbent,
            "C": with_challenger,
            "D": with_challenger,
        }
    if selected != expected_sets:
        raise CanonicalReplayV4Error("A/B/C/D factor sets do not match transition_mode")
    challenger_record = results["C"]["quant"]["factor_records"].get(challenger)
    if challenger_record is None:
        raise CanonicalReplayV4Error("comparison factor records are missing")
    if challenger_record["state"] not in {
        "shadow",
        "mature_candidate",
        "production_candidate",
    }:
        raise CanonicalReplayV4Error("challenger state is not eligible")
    if challenger_record["slot"] != slot:
        raise CanonicalReplayV4Error("challenger slot mismatch")
    incumbent_record: dict[str, str] | None = None
    if transition_mode == "add":
        if any(
            record["slot"] == slot
            for record in results["A"]["quant"]["factor_records"].values()
        ):
            raise CanonicalReplayV4Error("add transition requires an empty slot")
    else:
        assert incumbent is not None
        incumbent_record = results["A"]["quant"]["factor_records"].get(incumbent)
        if incumbent_record is None:
            raise CanonicalReplayV4Error("comparison factor records are missing")
        if incumbent_record["state"] != "production_factor":
            raise CanonicalReplayV4Error("incumbent is not a production factor")
        if (
            incumbent_record["slot"] != slot
            or incumbent_record["family"] != challenger_record["family"]
        ):
            raise CanonicalReplayV4Error("comparison family/slot mismatch")
    if any(
        results[arm]["eligibility"]["eligibility_contract_sha256"]
        != context["eligibility_contract_sha256"]
        for arm in ARM_NAMES
    ):
        raise CanonicalReplayV4Error("Eligibility output/context contract SHA mismatch")
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": payload["run_id"],
        "as_of": payload["as_of"],
        "registry_file_sha256": registry_sha,
        "production_factor_set_sha256": factor_set_sha,
        "context": dict(context),
        "context_sha256": context_sha,
        "calendar": calendar,
        "factor_set": factor_set,
        "comparison": {
            "transition_mode": transition_mode,
            "incumbent": incumbent,
            "challenger": challenger,
            "slot": slot,
            "cohorts": cohorts,
            "n": declared_n,
            "mean": declared_statistics["mean"],
            "stddev": declared_statistics["stddev"],
            "standard_error": declared_statistics["standard_error"],
            "critical_value": declared_statistics["critical_value"],
            "method": COMPARISON_METHOD,
            "sidedness": COMPARISON_SIDEDNESS,
            "incremental_edge_ci95_lower": edge_lower,
        },
        "replay_semantic_sha256": semantic_sha256(payload),
        "positive_weight_depends_on_risk_advisor_approval": False,
        "arms": results,
    }


def validate_v4_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the exact v4 evidence envelope; v2/v3 are unsupported."""

    if not isinstance(value, Mapping) or value.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise CanonicalReplayV4Error("unsupported factor governance evidence schema")
    payload = _exact(
        dict(value),
        {
            "schema_version",
            "status",
            "factor_name",
            "registry_file_sha256",
            "replay_path",
            "replay_file_sha256",
            "replay_semantic_sha256",
            "eligibility_contract_sha256",
            "calendar_sha256",
            "pit_sha256",
            "runtime_contract_sha256",
            "latest_pointer_sha256",
            "manifest_sha256",
            "market_data_input_sha256",
            "candidate_catalog_sha256",
            "screening_evidence_sha256",
            "dedup_evidence_sha256",
            "quantitative_evidence_sha256",
            "replay",
        },
        "v4 evidence",
    )
    if payload["status"] != "verified":
        raise CanonicalReplayV4Error("factor governance evidence is not verified")
    _text(payload["factor_name"], "factor_name")
    replay_path = _text(payload["replay_path"], "replay_path")
    if "\x00" in replay_path or not Path(replay_path).is_absolute():
        raise CanonicalReplayV4Error("replay_path must be an absolute path without NUL")
    for key in (
        "registry_file_sha256",
        "replay_file_sha256",
        "replay_semantic_sha256",
        *CONTEXT_BINDING_FIELDS,
    ):
        _sha(payload[key], key)
    if not isinstance(payload["replay"], Mapping):
        raise CanonicalReplayV4Error("v4 evidence replay must be an object")
    replay = validate_canonical_replay_v4(payload["replay"])
    if payload["replay_semantic_sha256"] != replay["replay_semantic_sha256"]:
        raise CanonicalReplayV4Error("v4 evidence replay semantic SHA mismatch")
    if payload["registry_file_sha256"] != replay["registry_file_sha256"]:
        raise CanonicalReplayV4Error("v4 evidence registry SHA mismatch")
    for key in CONTEXT_BINDING_FIELDS:
        if payload[key] != replay["context"][key]:
            raise CanonicalReplayV4Error(f"v4 evidence {key} mismatch")
    if payload["factor_name"] != replay["comparison"]["challenger"]:
        raise CanonicalReplayV4Error("v4 evidence factor identity mismatch")
    return dict(payload)


def _strict_json_loads(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalReplayV4Error(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalReplayV4Error(f"replay bytes are not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalReplayV4Error("replay file must contain an object")
    return value


def readback_v4_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Read one explicit 0600 replay and revalidate its exact byte graph."""

    evidence = validate_v4_evidence(value)
    path = Path(evidence["replay_path"])
    try:
        before = path.lstat()
    except OSError as exc:
        raise CanonicalReplayV4Error(f"replay path is unavailable: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CanonicalReplayV4Error("replay path must be a regular non-symlink file")
    if before.st_uid != os.getuid():
        raise CanonicalReplayV4Error("replay file owner mismatch")
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise CanonicalReplayV4Error("replay file mode must be 0600")
    if before.st_nlink != 1:
        raise CanonicalReplayV4Error("replay file link count must be one")
    if before.st_size <= 0 or before.st_size > MAX_REPLAY_FILE_BYTES:
        raise CanonicalReplayV4Error("replay file size is invalid")

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            item.st_mode,
            item.st_nlink,
        )

    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CanonicalReplayV4Error(f"replay open failed: {exc}") from exc
    try:
        opened = os.fstat(fd)
        if identity(before) != identity(opened):
            raise CanonicalReplayV4Error("replay path changed before readback")
        chunks: list[bytes] = []
        remaining = MAX_REPLAY_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        opened_after = os.fstat(fd)
    except OSError as exc:
        raise CanonicalReplayV4Error(f"replay readback failed: {exc}") from exc
    finally:
        os.close(fd)
    try:
        after = path.lstat()
    except OSError as exc:
        raise CanonicalReplayV4Error(f"replay path disappeared after readback: {exc}") from exc
    if len(raw) > MAX_REPLAY_FILE_BYTES:
        raise CanonicalReplayV4Error("replay file exceeds byte limit")
    if identity(before) != identity(after) or identity(opened) != identity(opened_after):
        raise CanonicalReplayV4Error("replay file identity changed during readback")
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != evidence["replay_file_sha256"]:
        raise CanonicalReplayV4Error("replay file SHA mismatch")
    replay_value = _strict_json_loads(raw)
    if raw != canonical_file_bytes(replay_value):
        raise CanonicalReplayV4Error(
            "replay bytes must be compact sorted canonical JSON with one newline"
        )
    replay = validate_canonical_replay_v4(replay_value)
    if replay["replay_semantic_sha256"] != evidence["replay_semantic_sha256"]:
        raise CanonicalReplayV4Error("evidence replay semantic SHA mismatch")
    return {
        "evidence": evidence,
        "replay": replay,
        "replay_file_sha256": file_sha,
        "replay_file_size": len(raw),
        "local_bytes_readback_verified": True,
        "complete_chain_hash_binding_verified": True,
        "context_bindings_readback_verified": True,
        "quantitative_evidence_hash_binding_verified": True,
        "positive_weight_depends_on_risk_advisor_approval": False,
    }


__all__ = [
    "ARM_NAMES",
    "CALENDAR_SCHEMA_VERSION",
    "CONTROL_CHAIN_STAGES",
    "CanonicalReplayV4Error",
    "EVIDENCE_SCHEMA_VERSION",
    "GENESIS_SHA256",
    "REPLAY_SCHEMA_VERSION",
    "STAGE_SCHEMA_VERSION",
    "byte_sha256",
    "canonical_file_bytes",
    "canonical_json_bytes",
    "readback_v4_evidence",
    "semantic_sha256",
    "stage_byte_sha256",
    "validate_canonical_replay_v4",
    "validate_v4_evidence",
]
