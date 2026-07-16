"""Strict, offline FactorGovernanceProtocol v3 replay validation.

The v3 replay is deliberately a new wire contract.  It never coerces or
upgrades v2 evidence and it binds the complete five-stage decision chain for
each A/B/C/D arm.  The validator is pure: it authenticates structure and
semantic/predecessor hashes but grants no production or mutation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL_VERSION = "v3"
REPLAY_SCHEMA_VERSION = "factor-governance-canonical-replay.v3"
EVIDENCE_SCHEMA_VERSION = "factor-governance-replay-evidence.v3"
STAGE_SCHEMA_VERSION = "factor-governance-canonical-stage.v3"
ARM_NAMES = ("A", "B", "C", "D")
CONTROL_CHAIN_STAGES = (
    "deterministic_funnel",
    "quant",
    "bayesian",
    "risk_guard",
    "portfolio_constructor",
)
GENESIS_SHA256 = "0" * 64


class CanonicalReplayV3Error(ValueError):
    """Raised when a v3 replay fails closed."""


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
        raise CanonicalReplayV3Error(f"value is not canonical JSON: {exc}") from exc


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CanonicalReplayV3Error(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing={','.join(missing)}")
        if unknown:
            detail.append(f"unknown={','.join(unknown)}")
        raise CanonicalReplayV3Error(f"{label} fields invalid: {';'.join(detail)}")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CanonicalReplayV3Error(f"{label} must be an exact non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CanonicalReplayV3Error(f"{label} must be lowercase SHA-256")
    return text


def _symbols(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise CanonicalReplayV3Error(f"{label} must be a list")
    result = [_text(item, f"{label}[]") for item in value]
    if result != sorted(result) or len(result) != len(set(result)):
        raise CanonicalReplayV3Error(f"{label} must be sorted and distinct")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalReplayV3Error(f"{label} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CanonicalReplayV3Error(f"{label} must be finite numeric")
    return number


def _validate_factor_records(
    output: Mapping[str, Any], arm: str
) -> tuple[list[str], dict[str, dict[str, str]]]:
    selected = _symbols(output["selected_factors"], f"arm {arm} selected_factors")
    raw_records = output["factor_records"]
    if not isinstance(raw_records, list):
        raise CanonicalReplayV3Error("factor_records must be a list")
    records: dict[str, dict[str, str]] = {}
    fields = {
        "name",
        "family",
        "slot",
        "registry_state",
        "registry_record_sha256",
    }
    for index, raw in enumerate(raw_records):
        item = _exact(raw, fields, f"arm {arm} factor_records[{index}]")
        name = _text(item["name"], "factor name")
        if name in records:
            raise CanonicalReplayV3Error("factor record names must be distinct")
        records[name] = {
            "name": name,
            "family": _text(item["family"], "factor family"),
            "slot": _text(item["slot"], "factor slot"),
            "registry_state": _text(item["registry_state"], "registry state"),
            "registry_record_sha256": _sha(
                item["registry_record_sha256"], "registry record SHA"
            ),
        }
    if set(records) != set(selected):
        raise CanonicalReplayV3Error(
            f"arm {arm} selected_factors/factor_records mismatch"
        )
    return selected, records


def _validate_output(
    stage: str,
    output: Any,
    *,
    arm: str,
    domain: list[str] | None,
    risk_decisions: Mapping[str, str] | None,
) -> dict[str, Any]:
    if stage == "deterministic_funnel":
        payload = _exact(
            output,
            {"schema_version", "eligible_symbols"},
            "deterministic funnel output",
        )
        if payload["schema_version"] != "factor-governance-funnel-output.v3":
            raise CanonicalReplayV3Error("unsupported funnel output schema")
        return {"domain": _symbols(payload["eligible_symbols"], "eligible symbols")}

    if domain is None:
        raise CanonicalReplayV3Error("eligible domain is unavailable")
    domain_set = set(domain)

    if stage == "quant":
        payload = _exact(
            output,
            {
                "schema_version",
                "eligible_symbols",
                "selected_factors",
                "factor_records",
                "branches",
                "likelihood_branches",
            },
            "quant output",
        )
        if payload["schema_version"] != "factor-governance-quant-output.v3":
            raise CanonicalReplayV3Error("unsupported quant output schema")
        if _symbols(payload["eligible_symbols"], "quant eligible symbols") != domain:
            raise CanonicalReplayV3Error("quant symbol domain mismatch")
        branches = _exact(payload["branches"], {"quant", "fundamental", "macro"}, "branches")
        branch_fields = {"ready", "object_sha256", "semantic_sha256"}
        normalized_branches: dict[str, Any] = {}
        for name in ("quant", "fundamental", "macro"):
            row = _exact(branches[name], branch_fields, f"branch {name}")
            if type(row["ready"]) is not bool:
                raise CanonicalReplayV3Error(f"branch {name}.ready must be boolean")
            normalized_branches[name] = {
                "ready": row["ready"],
                "object_sha256": _sha(row["object_sha256"], f"branch {name} object SHA"),
                "semantic_sha256": _sha(
                    row["semantic_sha256"], f"branch {name} semantic SHA"
                ),
            }
        if payload["likelihood_branches"] != ["fundamental", "quant"]:
            raise CanonicalReplayV3Error(
                "likelihood branches must be exactly fundamental/quant"
            )
        selected, records = _validate_factor_records(payload, arm)
        return {
            "domain": domain,
            "selected_factors": selected,
            "factor_records": records,
            "branches": normalized_branches,
        }

    if stage == "bayesian":
        payload = _exact(
            output,
            {"schema_version", "posterior_scores"},
            "Bayesian output",
        )
        if payload["schema_version"] != "factor-governance-bayesian-output.v3":
            raise CanonicalReplayV3Error("unsupported Bayesian output schema")
        scores = payload["posterior_scores"]
        if not isinstance(scores, dict) or set(scores) != domain_set:
            raise CanonicalReplayV3Error("Bayesian symbol domain mismatch")
        return {"domain": domain, "scores": {key: _finite(scores[key], key) for key in domain}}

    if stage == "risk_guard":
        label = "RiskGuard"
        expected_schema = "factor-governance-risk-output.v3"
        payload = _exact(output, {"schema_version", "decisions"}, f"{label} output")
        if payload["schema_version"] != expected_schema:
            raise CanonicalReplayV3Error(f"unsupported {label} output schema")
        decisions = payload["decisions"]
        if not isinstance(decisions, dict) or set(decisions) != domain_set:
            raise CanonicalReplayV3Error(f"{label} symbol domain mismatch")
        normalized: dict[str, str] = {}
        for symbol in domain:
            decision = _text(decisions[symbol], f"{label} decision {symbol}")
            if decision not in {"approved", "rejected"}:
                raise CanonicalReplayV3Error(f"{label} decision is invalid")
            normalized[symbol] = decision
        return {"domain": domain, "decisions": normalized}

    payload = _exact(
        output,
        {"schema_version", "target_weights"},
        "PortfolioConstructor output",
    )
    if payload["schema_version"] != "factor-governance-portfolio-output.v3":
        raise CanonicalReplayV3Error("unsupported PortfolioConstructor output schema")
    weights = payload["target_weights"]
    if not isinstance(weights, dict) or not set(weights).issubset(domain_set):
        raise CanonicalReplayV3Error("PortfolioConstructor contains an unknown symbol")
    normalized_weights: dict[str, float] = {
        str(symbol): _finite(value, f"weight {symbol}")
        for symbol, value in weights.items()
    }
    if any(value < 0.0 or value > 1.0 for value in normalized_weights.values()):
        raise CanonicalReplayV3Error("portfolio weights must be in [0,1]")
    if sum(normalized_weights.values()) > 1.0 + 1e-12:
        raise CanonicalReplayV3Error("portfolio weights exceed one")
    approved = {
        symbol
        for symbol, decision in dict(risk_decisions or {}).items()
        if decision == "approved"
    }
    if any(
        value > 0.0 and symbol not in approved
        for symbol, value in normalized_weights.items()
    ):
        raise CanonicalReplayV3Error("positive portfolio weight lacks RiskGuard approval")
    return {"domain": domain, "target_weights": normalized_weights}


def validate_canonical_replay_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one complete in-memory v3 replay."""

    payload = _exact(
        dict(value),
        {
            "schema_version",
            "protocol_version",
            "run_id",
            "as_of",
            "registry_file_sha256",
            "production_factor_set_sha256",
            "calendar_sha256",
            "pit_sha256",
            "context_sha256",
            "factor_set",
            "comparison",
            "stages",
        },
        "canonical replay",
    )
    if payload["schema_version"] != REPLAY_SCHEMA_VERSION:
        raise CanonicalReplayV3Error("unsupported canonical replay schema")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise CanonicalReplayV3Error("canonical replay protocol version mismatch")
    _text(payload["run_id"], "run_id")
    _text(payload["as_of"], "as_of")
    registry_file_sha = _sha(payload["registry_file_sha256"], "registry_file_sha256")
    production_factor_set_sha = _sha(
        payload["production_factor_set_sha256"], "production_factor_set_sha256"
    )
    calendar_sha = _sha(payload["calendar_sha256"], "calendar_sha256")
    pit_sha = _sha(payload["pit_sha256"], "pit_sha256")
    context_sha = _sha(payload["context_sha256"], "context_sha256")
    expected_context_sha = semantic_sha256(
        {
            "registry_file_sha256": registry_file_sha,
            "production_factor_set_sha256": production_factor_set_sha,
            "calendar_sha256": calendar_sha,
            "pit_sha256": pit_sha,
        }
    )
    if context_sha != expected_context_sha:
        raise CanonicalReplayV3Error("canonical replay context SHA mismatch")
    factor_set = _symbols(payload["factor_set"], "factor_set")
    comparison = _exact(
        payload["comparison"], {"incumbent", "challenger", "slot"}, "comparison"
    )
    incumbent = _text(comparison["incumbent"], "incumbent")
    challenger = _text(comparison["challenger"], "challenger")
    slot = _text(comparison["slot"], "slot")
    if incumbent == challenger or incumbent not in factor_set or challenger in factor_set:
        raise CanonicalReplayV3Error("comparison does not identify one replacement")

    stages = payload["stages"]
    expected_order = [(arm, stage) for arm in ARM_NAMES for stage in CONTROL_CHAIN_STAGES]
    if not isinstance(stages, list) or len(stages) != len(expected_order):
        raise CanonicalReplayV3Error("canonical replay must contain exactly 20 stages")
    stage_fields = {
        "schema_version",
        "arm",
        "stage",
        "context_sha256",
        "byte_sha256",
        "semantic_sha256",
        "predecessor",
        "output",
    }
    arm_results: dict[str, dict[str, Any]] = {arm: {} for arm in ARM_NAMES}
    seen_byte_hashes: set[str] = set()
    for index, raw in enumerate(stages):
        item = _exact(raw, stage_fields, f"stage[{index}]")
        arm, stage = expected_order[index]
        if item["schema_version"] != STAGE_SCHEMA_VERSION:
            raise CanonicalReplayV3Error("unsupported canonical stage schema")
        if item["arm"] != arm or item["stage"] != stage:
            raise CanonicalReplayV3Error("stage order/identity mismatch")
        if item["context_sha256"] != context_sha:
            raise CanonicalReplayV3Error("stage context SHA mismatch")
        byte_sha = _sha(item["byte_sha256"], "stage byte SHA")
        if byte_sha in seen_byte_hashes:
            raise CanonicalReplayV3Error("stage byte identities must be globally unique")
        seen_byte_hashes.add(byte_sha)
        semantic_sha = _sha(item["semantic_sha256"], "stage semantic SHA")
        if semantic_sha != semantic_sha256(item["output"]):
            raise CanonicalReplayV3Error("stage semantic SHA mismatch")
        predecessor = _exact(
            item["predecessor"],
            {"kind", "byte_sha256", "semantic_sha256"},
            "stage predecessor",
        )
        if stage == CONTROL_CHAIN_STAGES[0]:
            expected_predecessor = {
                "kind": "genesis",
                "byte_sha256": GENESIS_SHA256,
                "semantic_sha256": GENESIS_SHA256,
            }
        else:
            previous = arm_results[arm][CONTROL_CHAIN_STAGES[CONTROL_CHAIN_STAGES.index(stage) - 1]]
            expected_predecessor = {
                "kind": "stage",
                "byte_sha256": previous["byte_sha256"],
                "semantic_sha256": previous["semantic_sha256"],
            }
        if predecessor != expected_predecessor:
            raise CanonicalReplayV3Error("stage predecessor byte/semantic mismatch")
        prior = arm_results[arm]
        result = _validate_output(
            stage,
            item["output"],
            arm=arm,
            domain=(prior.get("deterministic_funnel") or {}).get("domain"),
            risk_decisions=(prior.get("risk_guard") or {}).get("decisions"),
        )
        arm_results[arm][stage] = {
            **result,
            "byte_sha256": byte_sha,
            "semantic_sha256": semantic_sha,
        }

    selected = {
        arm: arm_results[arm]["quant"]["selected_factors"] for arm in ARM_NAMES
    }
    expected_b = sorted(name for name in factor_set if name != incumbent)
    expected_c = sorted([*expected_b, challenger])
    if selected != {"A": factor_set, "B": expected_b, "C": expected_c, "D": expected_c}:
        raise CanonicalReplayV3Error("A/B/C/D factor sets are not one-slot replacement")
    incumbent_record = arm_results["A"]["quant"]["factor_records"].get(incumbent)
    challenger_record = arm_results["C"]["quant"]["factor_records"].get(challenger)
    if incumbent_record is None or challenger_record is None:
        raise CanonicalReplayV3Error("comparison factor records are missing")
    if incumbent_record["registry_state"] != "production_factor":
        raise CanonicalReplayV3Error("incumbent is not a production factor")
    if challenger_record["registry_state"] not in {
        "shadow",
        "mature_candidate",
        "production_candidate",
    }:
        raise CanonicalReplayV3Error("challenger registry state is not eligible")
    if (
        incumbent_record["slot"] != slot
        or challenger_record["slot"] != slot
        or incumbent_record["family"] != challenger_record["family"]
    ):
        raise CanonicalReplayV3Error("comparison factor family/slot mismatch")
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": payload["run_id"],
        "as_of": payload["as_of"],
        "comparison": {
            "incumbent": incumbent,
            "challenger": challenger,
            "slot": slot,
        },
        "registry_file_sha256": payload["registry_file_sha256"],
        "production_factor_set_sha256": payload["production_factor_set_sha256"],
        "calendar_sha256": calendar_sha,
        "pit_sha256": pit_sha,
        "replay_semantic_sha256": semantic_sha256(payload),
        "arms": arm_results,
    }


def validate_v3_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject legacy evidence and validate the exact v3 evidence envelope."""

    if not isinstance(value, Mapping) or value.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise CanonicalReplayV3Error("unsupported factor governance evidence schema")
    payload = _exact(
        dict(value),
        {
            "schema_version",
            "status",
            "factor_name",
            "registry_file_sha256",
            "replay_semantic_sha256",
            "calendar_sha256",
            "pit_sha256",
            "runtime_contract_sha256",
            "replay",
        },
        "v3 evidence",
    )
    if payload["status"] != "verified":
        raise CanonicalReplayV3Error("factor governance evidence is not verified")
    _text(payload["factor_name"], "factor_name")
    for key in (
        "registry_file_sha256",
        "replay_semantic_sha256",
        "calendar_sha256",
        "pit_sha256",
        "runtime_contract_sha256",
    ):
        _sha(payload[key], key)
    replay_value = payload["replay"]
    if not isinstance(replay_value, Mapping):
        raise CanonicalReplayV3Error("v3 evidence replay must be an object")
    replay = validate_canonical_replay_v3(replay_value)
    if payload["replay_semantic_sha256"] != replay["replay_semantic_sha256"]:
        raise CanonicalReplayV3Error("v3 evidence replay semantic SHA mismatch")
    if payload["registry_file_sha256"] != replay["registry_file_sha256"]:
        raise CanonicalReplayV3Error("v3 evidence replay registry SHA mismatch")
    if payload["calendar_sha256"] != replay["calendar_sha256"]:
        raise CanonicalReplayV3Error("v3 evidence replay calendar SHA mismatch")
    if payload["pit_sha256"] != replay["pit_sha256"]:
        raise CanonicalReplayV3Error("v3 evidence replay PIT SHA mismatch")
    if payload["factor_name"] != replay["comparison"]["challenger"]:
        raise CanonicalReplayV3Error("v3 evidence replay factor identity mismatch")
    return payload


__all__ = [
    "ARM_NAMES",
    "CONTROL_CHAIN_STAGES",
    "CanonicalReplayV3Error",
    "EVIDENCE_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "STAGE_SCHEMA_VERSION",
    "canonical_json_bytes",
    "semantic_sha256",
    "validate_canonical_replay_v3",
    "validate_v3_evidence",
]
