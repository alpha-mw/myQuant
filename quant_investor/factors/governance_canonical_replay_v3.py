"""Strict, offline FactorGovernanceProtocol v3 replay validation.

The v3 replay is deliberately a new wire contract.  It never coerces or
upgrades v2 evidence and it binds the complete six-stage runtime decision
chain for each A/B/C/D arm.  Structural validation and local byte readback are
separate operations; neither grants production mutation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "v3"
REPLAY_SCHEMA_VERSION = "factor-governance-canonical-replay.v3"
EVIDENCE_SCHEMA_VERSION = "factor-governance-replay-evidence.v3"
STAGE_SCHEMA_VERSION = "factor-governance-canonical-stage.v3"
ARM_NAMES = ("A", "B", "C", "D")
CONTROL_CHAIN_STAGES = (
    "quant",
    "deterministic_funnel",
    "bayesian",
    "risk_guard",
    "ic_coordinator",
    "portfolio_constructor",
)
GENESIS_SHA256 = "0" * 64
MAX_REPLAY_FILE_BYTES = 16 * 1024 * 1024


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


def canonical_file_bytes(value: Any) -> bytes:
    """Return the only accepted on-disk encoding for a v3 replay."""

    return canonical_json_bytes(value) + b"\n"


def byte_sha256(value: Any) -> str:
    """Hash the exact canonical newline-terminated stage bytes."""

    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def stage_byte_sha256(
    *,
    arm: str,
    stage: str,
    context_sha256: str,
    predecessor: Mapping[str, Any],
    output: Any,
) -> str:
    """Hash one canonical stage artifact without self-referential fields."""

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


def _hash_map(value: Any, *, domain: list[str], label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(domain):
        raise CanonicalReplayV3Error(f"{label} symbol domain mismatch")
    return {
        symbol: _sha(value[symbol], f"{label} {symbol}")
        for symbol in domain
    }


def _validate_output(
    stage: str,
    output: Any,
    *,
    arm: str,
    prior: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if stage == "quant":
        payload = _exact(
            output,
            {
                "schema_version",
                "scored_symbols",
                "selected_factors",
                "factor_records",
                "branches",
                "likelihood_branches",
            },
            "quant output",
        )
        if payload["schema_version"] != "factor-governance-quant-output.v3":
            raise CanonicalReplayV3Error("unsupported quant output schema")
        scored_symbols = _symbols(payload["scored_symbols"], "quant scored symbols")
        branches = _exact(
            payload["branches"],
            {"quant", "fundamental", "macro"},
            "branches",
        )
        branch_fields = {"ready", "object_sha256", "semantic_sha256"}
        normalized_branches: dict[str, Any] = {}
        for name in ("quant", "fundamental", "macro"):
            row = _exact(branches[name], branch_fields, f"branch {name}")
            if type(row["ready"]) is not bool:
                raise CanonicalReplayV3Error(f"branch {name}.ready must be boolean")
            normalized_branches[name] = {
                "ready": row["ready"],
                "object_sha256": _sha(
                    row["object_sha256"], f"branch {name} object SHA"
                ),
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
            "domain": scored_symbols,
            "selected_factors": selected,
            "factor_records": records,
            "branches": normalized_branches,
        }

    quant_domain = list((prior.get("quant") or {}).get("domain") or [])
    if stage == "deterministic_funnel":
        payload = _exact(
            output,
            {"schema_version", "eligible_symbols"},
            "deterministic funnel output",
        )
        if payload["schema_version"] != "factor-governance-funnel-output.v3":
            raise CanonicalReplayV3Error("unsupported funnel output schema")
        eligible = _symbols(payload["eligible_symbols"], "eligible symbols")
        if not set(eligible).issubset(quant_domain):
            raise CanonicalReplayV3Error(
                "deterministic funnel contains a symbol not scored by Quant"
            )
        return {"domain": eligible}

    domain = list((prior.get("deterministic_funnel") or {}).get("domain") or [])
    if "deterministic_funnel" not in prior:
        raise CanonicalReplayV3Error("eligible domain is unavailable")
    domain_set = set(domain)

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
        payload = _exact(
            output,
            {
                "schema_version",
                "decisions",
                "risk_decision",
                "risk_decision_sha256",
            },
            f"{label} output",
        )
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
        risk_decision = payload["risk_decision"]
        if not isinstance(risk_decision, dict) or not risk_decision:
            raise CanonicalReplayV3Error("RiskGuard decision payload must be an object")
        risk_decision_sha = _sha(
            payload["risk_decision_sha256"], "RiskGuard decision SHA"
        )
        if risk_decision_sha != semantic_sha256(risk_decision):
            raise CanonicalReplayV3Error("RiskGuard decision SHA mismatch")
        return {
            "domain": domain,
            "decisions": normalized,
            "risk_decision": risk_decision,
            "risk_decision_sha256": risk_decision_sha,
        }

    if stage == "ic_coordinator":
        payload = _exact(
            output,
            {
                "schema_version",
                "inputs",
                "input_sha256s",
                "decisions",
                "output_sha256s",
            },
            "ICCoordinator output",
        )
        if payload["schema_version"] != "factor-governance-ic-output.v3":
            raise CanonicalReplayV3Error("unsupported ICCoordinator output schema")
        inputs = payload["inputs"]
        decisions = payload["decisions"]
        if not isinstance(inputs, dict) or set(inputs) != domain_set:
            raise CanonicalReplayV3Error("ICCoordinator input symbol domain mismatch")
        if not isinstance(decisions, dict) or set(decisions) != domain_set:
            raise CanonicalReplayV3Error("ICCoordinator output symbol domain mismatch")
        input_hashes = _hash_map(
            payload["input_sha256s"], domain=domain, label="ICCoordinator input SHA"
        )
        output_hashes = _hash_map(
            payload["output_sha256s"], domain=domain, label="ICCoordinator output SHA"
        )
        risk_result = prior.get("risk_guard") or {}
        risk_sha = str(risk_result.get("risk_decision_sha256") or "")
        normalized_actions: dict[str, str] = {}
        for symbol in domain:
            ic_input = _exact(
                inputs[symbol],
                {"branch_verdicts", "risk_decision", "ic_hints"},
                f"ICCoordinator input {symbol}",
            )
            branches = ic_input["branch_verdicts"]
            if not isinstance(branches, dict) or set(branches) != {
                "quant",
                "fundamental",
                "macro",
            }:
                raise CanonicalReplayV3Error(
                    "ICCoordinator branch verdicts must be exactly quant/fundamental/macro"
                )
            if not all(isinstance(value, dict) for value in branches.values()):
                raise CanonicalReplayV3Error(
                    "ICCoordinator branch verdicts must be objects"
                )
            if not isinstance(ic_input["risk_decision"], dict):
                raise CanonicalReplayV3Error(
                    "ICCoordinator risk decision must be an object"
                )
            if semantic_sha256(ic_input["risk_decision"]) != risk_sha:
                raise CanonicalReplayV3Error(
                    "ICCoordinator input is not bound to RiskGuard output"
                )
            if ic_input["ic_hints"] != {}:
                raise CanonicalReplayV3Error(
                    "ICCoordinator deterministic runtime input must use empty ic_hints"
                )
            if input_hashes[symbol] != semantic_sha256(ic_input):
                raise CanonicalReplayV3Error("ICCoordinator input SHA mismatch")

            decision = decisions[symbol]
            if not isinstance(decision, dict):
                raise CanonicalReplayV3Error(
                    "ICCoordinator decision payload must be an object"
                )
            action = _text(decision.get("action"), f"ICCoordinator action {symbol}")
            if action not in {"buy", "hold", "sell", "watch", "avoid"}:
                raise CanonicalReplayV3Error("ICCoordinator action is invalid")
            decision_symbol = decision.get("symbol")
            if decision_symbol is not None and decision_symbol != symbol:
                raise CanonicalReplayV3Error("ICCoordinator decision symbol mismatch")
            if output_hashes[symbol] != semantic_sha256(decision):
                raise CanonicalReplayV3Error("ICCoordinator output SHA mismatch")
            normalized_actions[symbol] = action
        return {
            "domain": domain,
            "input_sha256s": input_hashes,
            "output_sha256s": output_hashes,
            "actions": normalized_actions,
        }

    payload = _exact(
        output,
        {"schema_version", "target_weights", "ic_decision_sha256s"},
        "PortfolioConstructor output",
    )
    if payload["schema_version"] != "factor-governance-portfolio-output.v3":
        raise CanonicalReplayV3Error("unsupported PortfolioConstructor output schema")
    weights = payload["target_weights"]
    if not isinstance(weights, dict) or not set(weights).issubset(domain_set):
        raise CanonicalReplayV3Error("PortfolioConstructor contains an unknown symbol")
    normalized_weights = {
        symbol: _finite(value, f"weight {symbol}")
        for symbol, value in weights.items()
    }
    if any(value < 0.0 or value > 1.0 for value in normalized_weights.values()):
        raise CanonicalReplayV3Error("portfolio weights must be in [0,1]")
    if sum(normalized_weights.values()) > 1.0 + 1e-12:
        raise CanonicalReplayV3Error("portfolio weights exceed one")
    risk_decisions = dict((prior.get("risk_guard") or {}).get("decisions") or {})
    approved = {
        symbol
        for symbol, decision in risk_decisions.items()
        if decision == "approved"
    }
    if any(
        value > 0.0 and symbol not in approved
        for symbol, value in normalized_weights.items()
    ):
        raise CanonicalReplayV3Error("positive portfolio weight lacks RiskGuard approval")
    ic_result = prior.get("ic_coordinator") or {}
    ic_hashes = _hash_map(
        payload["ic_decision_sha256s"],
        domain=domain,
        label="PortfolioConstructor IC decision SHA",
    )
    if ic_hashes != dict(ic_result.get("output_sha256s") or {}):
        raise CanonicalReplayV3Error(
            "PortfolioConstructor is not bound to ICCoordinator output"
        )
    actions = dict(ic_result.get("actions") or {})
    if any(
        value > 0.0 and actions.get(symbol) != "buy"
        for symbol, value in normalized_weights.items()
    ):
        raise CanonicalReplayV3Error("positive portfolio weight lacks ICCoordinator BUY")
    return {
        "domain": domain,
        "target_weights": normalized_weights,
        "ic_decision_sha256s": ic_hashes,
    }


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
            "context",
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
    _sha(payload["registry_file_sha256"], "registry_file_sha256")
    _sha(payload["production_factor_set_sha256"], "production_factor_set_sha256")
    context = _exact(
        payload["context"],
        {"calendar_sha256", "pit_sha256", "runtime_contract_sha256"},
        "replay context",
    )
    for key in ("calendar_sha256", "pit_sha256", "runtime_contract_sha256"):
        _sha(context[key], f"context {key}")
    context_sha = _sha(payload["context_sha256"], "context_sha256")
    if context_sha != semantic_sha256(context):
        raise CanonicalReplayV3Error("replay context SHA mismatch")
    factor_set = _symbols(payload["factor_set"], "factor_set")
    if payload["production_factor_set_sha256"] != semantic_sha256(factor_set):
        raise CanonicalReplayV3Error("production factor set SHA mismatch")
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
        raise CanonicalReplayV3Error("canonical replay must contain exactly 24 stages")
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
        if byte_sha != stage_byte_sha256(
            arm=arm,
            stage=stage,
            context_sha256=context_sha,
            predecessor=item["predecessor"],
            output=item["output"],
        ):
            raise CanonicalReplayV3Error("stage byte SHA mismatch")
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
            prior=prior,
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
        "registry_file_sha256": payload["registry_file_sha256"],
        "production_factor_set_sha256": payload["production_factor_set_sha256"],
        "context": dict(context),
        "factor_set": factor_set,
        "comparison": dict(comparison),
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
            "replay_path",
            "replay_file_sha256",
            "replay_semantic_sha256",
            "calendar_sha256",
            "pit_sha256",
            "runtime_contract_sha256",
        },
        "v3 evidence",
    )
    if payload["status"] != "verified":
        raise CanonicalReplayV3Error("factor governance evidence is not verified")
    _text(payload["factor_name"], "factor_name")
    replay_path = _text(payload["replay_path"], "replay_path")
    if "\x00" in replay_path:
        raise CanonicalReplayV3Error("replay_path contains a null byte")
    if not Path(replay_path).is_absolute():
        raise CanonicalReplayV3Error("replay_path must be absolute")
    for key in (
        "registry_file_sha256",
        "replay_file_sha256",
        "replay_semantic_sha256",
        "calendar_sha256",
        "pit_sha256",
        "runtime_contract_sha256",
    ):
        _sha(payload[key], key)
    return payload


def _strict_json_loads(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalReplayV3Error(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalReplayV3Error(f"replay bytes are not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalReplayV3Error("replay file must contain an object")
    return value


def readback_v3_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Read one explicit replay path and revalidate its complete byte graph.

    There is no scan, glob, latest selection or fallback.  A successful return
    authenticates only the local bytes and their v3 semantic graph; it does not
    authenticate a producer identity or authorize a registry mutation.
    """

    evidence = validate_v3_evidence(value)
    path = Path(evidence["replay_path"])
    try:
        before = path.lstat()
    except OSError as exc:
        raise CanonicalReplayV3Error(f"replay path is unavailable: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CanonicalReplayV3Error("replay path must be a regular non-symlink file")
    if before.st_uid != os.getuid():
        raise CanonicalReplayV3Error("replay file owner mismatch")
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise CanonicalReplayV3Error("replay file mode must be 0600")
    if before.st_nlink != 1:
        raise CanonicalReplayV3Error("replay file link count must be one")
    if before.st_size <= 0 or before.st_size > MAX_REPLAY_FILE_BYTES:
        raise CanonicalReplayV3Error("replay file size is invalid")

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
        raise CanonicalReplayV3Error(f"replay open failed: {exc}") from exc
    try:
        opened = os.fstat(fd)
        if identity(before) != identity(opened):
            raise CanonicalReplayV3Error("replay path changed before readback")
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
        raise CanonicalReplayV3Error(f"replay readback failed: {exc}") from exc
    finally:
        os.close(fd)
    try:
        after = path.lstat()
    except OSError as exc:
        raise CanonicalReplayV3Error(
            f"replay path disappeared after readback: {exc}"
        ) from exc
    if len(raw) > MAX_REPLAY_FILE_BYTES:
        raise CanonicalReplayV3Error("replay file exceeds byte limit")
    if identity(before) != identity(after):
        raise CanonicalReplayV3Error("replay file identity changed during readback")
    if identity(opened) != identity(opened_after):
        raise CanonicalReplayV3Error("replay file identity changed while open")
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != evidence["replay_file_sha256"]:
        raise CanonicalReplayV3Error("replay file SHA mismatch")
    replay = _strict_json_loads(raw)
    if raw != canonical_file_bytes(replay):
        raise CanonicalReplayV3Error(
            "replay bytes must be compact sorted canonical JSON with one newline"
        )
    normalized = validate_canonical_replay_v3(replay)
    if normalized["replay_semantic_sha256"] != evidence["replay_semantic_sha256"]:
        raise CanonicalReplayV3Error("evidence replay semantic SHA mismatch")
    if normalized["registry_file_sha256"] != evidence["registry_file_sha256"]:
        raise CanonicalReplayV3Error("evidence replay registry SHA mismatch")
    context = normalized["context"]
    for key in ("calendar_sha256", "pit_sha256", "runtime_contract_sha256"):
        if context[key] != evidence[key]:
            raise CanonicalReplayV3Error(f"evidence replay {key} mismatch")
    comparison = normalized["comparison"]
    known_factors = set(normalized["factor_set"])
    known_factors.update((comparison["incumbent"], comparison["challenger"]))
    if evidence["factor_name"] not in known_factors:
        raise CanonicalReplayV3Error("evidence factor is absent from the replay")
    ic_readback = {
        arm: {
            "input_sha256s": dict(normalized["arms"][arm]["ic_coordinator"]["input_sha256s"]),
            "output_sha256s": dict(normalized["arms"][arm]["ic_coordinator"]["output_sha256s"]),
        }
        for arm in ARM_NAMES
    }
    return {
        "evidence": evidence,
        "replay": normalized,
        "replay_file_sha256": file_sha,
        "replay_file_size": len(raw),
        "ic_readback": ic_readback,
        "ic_input_output_hash_binding_verified": True,
        "local_bytes_readback_verified": True,
    }


__all__ = [
    "ARM_NAMES",
    "CONTROL_CHAIN_STAGES",
    "CanonicalReplayV3Error",
    "EVIDENCE_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "STAGE_SCHEMA_VERSION",
    "byte_sha256",
    "canonical_file_bytes",
    "canonical_json_bytes",
    "readback_v3_evidence",
    "semantic_sha256",
    "stage_byte_sha256",
    "validate_canonical_replay_v3",
    "validate_v3_evidence",
]
