"""Research-only Stage 1 cutover at the formal Quant-to-Funnel boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from quant_investor.codex_review import prepare_stage1_run
from quant_investor.v16.diagnostic import V16NoAgentDiagnostic
from quant_investor.v16.stage1_contract import (
    PITFactRow,
    Stage1FactPackage,
    build_stage1_fact_package,
)

DEFAULT_CODEX_REVIEW_ROOT = "results/v16/codex_review"
DEFAULT_FACTOR_READINESS_PATH = "results/v16/factor_governance/readiness.json"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("four_branch_config.json")
DEFAULT_STAGE1_PROMPT_PATH = Path(__file__).with_name("stage1_prompt.md")
_SHA256_HEX = frozenset("0123456789abcdef")


class V16Stage1RuntimeError(RuntimeError):
    """Raised when a formal v16 Stage 1 package cannot be sealed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _semantic_sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise V16Stage1RuntimeError(f"{field_name} must be a lowercase SHA-256")
    return text


def _load_exact_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise V16Stage1RuntimeError(f"{label} is unavailable: {source}: {exc}") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V16Stage1RuntimeError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise V16Stage1RuntimeError(f"{label} must be a JSON object")
    return value


def _validate_factor_readiness(value: Mapping[str, Any]) -> str:
    expected_schema = "factor-governance-readiness.v4"
    blockers = [str(item) for item in value.get("blockers", []) or [] if str(item)]
    receipt = value.get("activation_receipt")
    receipt_map = dict(receipt) if isinstance(receipt, Mapping) else {}
    factor_count = value.get("production_factor_count")
    family_count = value.get("production_family_count")
    factor_rows = value.get("factors")
    if value.get("schema_version") != expected_schema:
        blockers.append("factor_readiness_schema_not_v4")
    if value.get("protocol_version") != "v4":
        blockers.append("factor_protocol_not_v4")
    if not isinstance(factor_count, int) or isinstance(factor_count, bool) or factor_count < 5:
        blockers.append("healthy_factor_count_below_5")
    if not isinstance(family_count, int) or isinstance(family_count, bool) or family_count < 3:
        blockers.append("healthy_factor_family_count_below_3")
    if value.get("factor_governance_ready") is not True:
        blockers.append("factor_governance_not_ready")
    if value.get("new_risk_eligible") is not True:
        blockers.append("factor_new_risk_not_eligible")
    if not isinstance(factor_rows, list):
        blockers.append("factor_v4_rows_missing")
    else:
        names = [
            str(dict(row).get("name") or "").strip()
            for row in factor_rows
            if isinstance(row, Mapping)
        ]
        runtime_hashes = [
            str(dict(row).get("runtime_contract_sha256") or "").strip()
            for row in factor_rows
            if isinstance(row, Mapping)
        ]
        if (
            len(names) != factor_count
            or len(names) != len(set(names))
            or any(not name for name in names)
            or any(
                not isinstance(row, Mapping) or dict(row).get("healthy") is not True
                for row in factor_rows
            )
        ):
            blockers.append("factor_v4_rows_not_exactly_healthy")
        else:
            from quant_investor.factors.runtime import (
                production_factor_set_sha256,
            )

            if production_factor_set_sha256(sorted(names)) != value.get(
                "production_factor_set_sha256"
            ):
                blockers.append("factor_v4_factor_set_sha_mismatch")
        if (
            len(runtime_hashes) != factor_count
            or any(
                len(item) != 64 or any(char not in _SHA256_HEX for char in item)
                for item in runtime_hashes
            )
            or _semantic_sha256(sorted(runtime_hashes)) != value.get("runtime_contracts_sha256")
        ):
            blockers.append("factor_v4_runtime_contract_set_sha_mismatch")
        factor_weights = value.get("normalized_abs_weights")
        family_weights = value.get("family_normalized_abs_weights")
        if not isinstance(factor_weights, Mapping) or not (
            set(factor_weights) == set(names)
            and all(
                isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and isfinite(float(weight))
                and 0.0 < float(weight) <= 0.20 + 1e-12
                for weight in factor_weights.values()
            )
            and abs(sum(float(weight) for weight in factor_weights.values()) - 1.0) <= 1e-9
        ):
            blockers.append("factor_v4_weight_contract_invalid")
        if not isinstance(family_weights, Mapping) or not (
            len(family_weights) == family_count
            and all(
                isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and isfinite(float(weight))
                and 0.0 < float(weight) <= 0.35 + 1e-12
                for weight in family_weights.values()
            )
            and abs(sum(float(weight) for weight in family_weights.values()) - 1.0) <= 1e-9
        ):
            blockers.append("factor_v4_family_weight_contract_invalid")
    receipt_payload = receipt_map.get("receipt")
    if receipt_map.get("valid") is not True or not isinstance(receipt_payload, Mapping):
        blockers.append("factor_v4_activation_receipt_invalid")
    receipt_sha = receipt_map.get("receipt_sha256")
    try:
        normalized_receipt_sha = _require_sha256(
            receipt_sha,
            field_name="factor activation receipt SHA",
        )
    except V16Stage1RuntimeError:
        blockers.append("factor_v4_activation_receipt_sha_invalid")
        normalized_receipt_sha = ""
    if isinstance(receipt_payload, Mapping):
        try:
            from quant_investor.factors.governance_transaction_v4 import (
                validate_activation_receipt_v4,
            )

            validated_receipt = validate_activation_receipt_v4(
                receipt_payload,
                expected_as_of=str(value.get("as_of") or ""),
                expected_protocol_hash=str(value.get("protocol_hash") or ""),
                expected_registry_file_sha256=str(value.get("registry_file_sha256") or ""),
                expected_production_factor_set_sha256=str(
                    value.get("production_factor_set_sha256") or ""
                ),
                expected_runtime_contracts_sha256=str(value.get("runtime_contracts_sha256") or ""),
            )
            if validated_receipt["receipt_sha256"] != normalized_receipt_sha:
                blockers.append("factor_v4_activation_receipt_summary_mismatch")
        except (TypeError, ValueError) as exc:
            blockers.append(f"factor_v4_activation_receipt_invalid:{exc}")
    if blockers:
        raise V16Stage1RuntimeError(
            "v16 formal Quant/Funnel is not factor-ready: " + ";".join(dict.fromkeys(blockers))
        )
    return normalized_receipt_sha


def load_v16_factor_readiness(path: str | Path) -> dict[str, Any]:
    """Load and validate the v4 receipt before formal Quant is allowed to run."""

    value = _load_exact_json(path, label="FactorGovernance v4 readiness")
    _validate_factor_readiness(value)
    return value


def _readiness_statuses(context_state: Any) -> dict[str, str]:
    readiness = getattr(context_state, "branch_data_readiness", {})
    if not isinstance(readiness, Mapping):
        raise V16Stage1RuntimeError("branch data readiness is missing")
    rows = readiness.get("readiness")
    if not isinstance(rows, Mapping):
        raise V16Stage1RuntimeError("branch data readiness rows are missing")
    statuses = {
        branch: str(dict(rows.get(branch) or {}).get("status") or "")
        for branch in ("quant", "fundamental", "macro")
    }
    blocked = [branch for branch, status in statuses.items() if status != "pass"]
    if blocked:
        raise V16Stage1RuntimeError("full-market Q/F/M facts are not ready: " + ",".join(blocked))
    return statuses


def _rank_strata(scores: Mapping[str, float]) -> dict[str, str]:
    ordered = sorted(scores, key=lambda symbol: (scores[symbol], symbol))
    size = len(ordered)
    return {
        symbol: f"quant_quintile_{min(5, (index * 5 // size) + 1)}"
        for index, symbol in enumerate(ordered)
    }


def build_stage1_package_from_market_context(
    *,
    context_state: Any,
    market: str,
    mode: str,
    pit_pointer_sha256: str,
    factor_readiness: Mapping[str, Any],
    cutoff_at: datetime,
    expires_at: datetime,
) -> Stage1FactPackage:
    """Seal full-market local facts after formal Quant and deterministic Funnel."""

    if str(market).strip().upper() != "CN":
        raise V16Stage1RuntimeError("v16 four-branch runtime supports CN only")
    if str(mode).strip().lower() == "sample":
        raise V16Stage1RuntimeError("v16 Stage 1 requires the unsampled full market")
    if cutoff_at.tzinfo is None or expires_at.tzinfo is None:
        raise V16Stage1RuntimeError("cutoff and expiry must be timezone-aware")
    factor_receipt_sha = _validate_factor_readiness(factor_readiness)
    statuses = _readiness_statuses(context_state)

    symbols = tuple(str(item).strip().upper() for item in context_state.researchable_symbols)
    if not symbols or len(symbols) != len(set(symbols)):
        raise V16Stage1RuntimeError("eligible full-market symbol set is empty or duplicated")
    quant_result = context_state.quant_result
    if getattr(quant_result, "success", False) is not True:
        raise V16Stage1RuntimeError("formal Quant result is not successful")
    raw_scores = dict(getattr(quant_result, "symbol_scores", {}) or {})
    if set(raw_scores) != set(symbols):
        missing = sorted(set(symbols) - set(raw_scores))
        extra = sorted(set(raw_scores) - set(symbols))
        raise V16Stage1RuntimeError(
            f"formal Quant score domain drift: missing={missing}, extra={extra}"
        )
    scores: dict[str, float] = {}
    for symbol, raw_score in raw_scores.items():
        score = float(raw_score)
        if not isfinite(score) or not -1.0 <= score <= 1.0:
            raise V16Stage1RuntimeError(f"formal Quant score invalid for {symbol}")
        scores[str(symbol)] = score

    funnel_symbols = tuple(str(item).strip().upper() for item in context_state.candidate_symbols)
    if not funnel_symbols or len(funnel_symbols) > 500:
        raise V16Stage1RuntimeError("deterministic Funnel must contain 1..500 symbols")
    if not set(funnel_symbols).issubset(set(symbols)):
        raise V16Stage1RuntimeError("deterministic Funnel contains an ineligible symbol")

    branch_payload = dict(getattr(context_state, "branch_data_payload", {}) or {})
    fundamentals = dict(branch_payload.get("fundamentals") or {})
    macro_facts = dict(branch_payload.get("macro_data") or {})
    if set(fundamentals) != set(symbols):
        missing = sorted(set(symbols) - set(fundamentals))
        extra = sorted(set(fundamentals) - set(symbols))
        raise V16Stage1RuntimeError(
            f"full-market Fundamental fact domain drift: missing={missing}, extra={extra}"
        )
    if not macro_facts:
        raise V16Stage1RuntimeError("sealed Macro fact package is empty")

    strata = _rank_strata(scores)
    pit_sha = _require_sha256(pit_pointer_sha256, field_name="PIT pointer SHA")
    rows: list[PITFactRow] = []
    market_states = dict(
        getattr(context_state.global_context, "metadata", {}).get("symbol_market_state", {}) or {}
    )
    for symbol in sorted(symbols):
        eligibility_receipt = _semantic_sha256(
            {
                "schema_version": "v16.eligibility-receipt.v1",
                "symbol": symbol,
                "pit_pointer_sha256": pit_sha,
                "factor_activation_receipt_sha256": factor_receipt_sha,
                "branch_readiness": statuses,
            }
        )
        fundamental = fundamentals[symbol]
        if not isinstance(fundamental, Mapping) or not fundamental:
            raise V16Stage1RuntimeError(f"Fundamental facts are empty for {symbol}")
        rows.append(
            PITFactRow(
                symbol=symbol,
                stratum=strata[symbol],
                eligibility_receipt_sha256=eligibility_receipt,
                formal_quant_score=scores[symbol],
                quant_facts={
                    "formal_score": scores[symbol],
                    "market_state": dict(market_states.get(symbol) or {}),
                    "factor_activation_receipt_sha256": factor_receipt_sha,
                },
                fundamental_facts=dict(fundamental),
                macro_facts=macro_facts,
            )
        )
    return build_stage1_fact_package(
        rows=rows,
        funnel_symbols=funnel_symbols,
        cutoff_at=cutoff_at.astimezone(timezone.utc).isoformat(),
        expires_at=expires_at.astimezone(timezone.utc).isoformat(),
        pit_pointer_sha256=pit_sha,
    )


def prepare_v16_stage1_pending(
    *,
    context_state: Any,
    market: str,
    mode: str,
    enable_agent_layer: bool,
    factor_readiness_path: str | Path = DEFAULT_FACTOR_READINESS_PATH,
    factor_readiness: Mapping[str, Any] | None = None,
    review_root: str | Path = DEFAULT_CODEX_REVIEW_ROOT,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    prompt_path: str | Path = DEFAULT_STAGE1_PROMPT_PATH,
    model_id: str,
    repo_path: str | Path = ".",
    run_id: str = "",
    now: datetime | None = None,
    expiry_hours: float = 24.0,
) -> dict[str, Any]:
    """Return a diagnostic or create exactly one S1_PREPARED research run."""

    occurred_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    resolved_run_id = str(run_id or "").strip() or (
        "v16-" + occurred_at.strftime("%Y%m%dT%H%M%S%fZ")
    )
    if not enable_agent_layer:
        return V16NoAgentDiagnostic(
            run_id=resolved_run_id,
            market=market,
            eligible_symbol_count=len(context_state.researchable_symbols),
            funnel_symbol_count=len(context_state.candidate_symbols),
            data_summary={
                "decision_protocol": "v16",
                "quant_completed_before_funnel": True,
                "codex_state_created": False,
            },
        ).to_dict()
    if not isfinite(float(expiry_hours)) or float(expiry_hours) <= 0.0:
        raise V16Stage1RuntimeError("expiry_hours must be finite and positive")
    pit_path_text = str(
        dict(getattr(context_state, "resolver_snapshot", {}) or {}).get("latest_pointer_path") or ""
    ).strip()
    if not pit_path_text:
        raise V16Stage1RuntimeError("strict PIT latest pointer path is missing")
    pit_path = Path(pit_path_text).expanduser().resolve()
    try:
        pit_sha = sha256(pit_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise V16Stage1RuntimeError(f"strict PIT latest pointer is unavailable: {exc}") from exc
    resolved_factor_readiness = (
        dict(factor_readiness)
        if factor_readiness is not None
        else load_v16_factor_readiness(factor_readiness_path)
    )
    package = build_stage1_package_from_market_context(
        context_state=context_state,
        market=market,
        mode=mode,
        pit_pointer_sha256=pit_sha,
        factor_readiness=resolved_factor_readiness,
        cutoff_at=occurred_at,
        expires_at=occurred_at + timedelta(hours=float(expiry_hours)),
    )
    prepared = prepare_stage1_run(
        root=review_root,
        run_id=resolved_run_id,
        payload=package,
        config_path=config_path,
        prompt_path=prompt_path,
        model_id=str(model_id or "").strip() or "codex-unresolved",
        pit_pointer_path=pit_path,
        repo_path=repo_path,
        now=occurred_at,
    )
    return {
        "schema_version": "v16.market-stage1-pending.v1",
        "decision_protocol": "v16",
        "status": "pending_codex_stage1",
        "formal_shortlist_generated": False,
        "new_risk_authorized": False,
        "run_id": resolved_run_id,
        "review": prepared,
        "candidate_boundaries": {
            "eligible": len(context_state.researchable_symbols),
            "funnel": len(context_state.candidate_symbols),
            "supplemental_max": 100,
            "union_max": 600,
            "menu_max": 50,
            "positive_weight_max": 12,
        },
    }


__all__ = [
    "DEFAULT_CODEX_REVIEW_ROOT",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FACTOR_READINESS_PATH",
    "DEFAULT_STAGE1_PROMPT_PATH",
    "V16Stage1RuntimeError",
    "build_stage1_package_from_market_context",
    "load_v16_factor_readiness",
    "prepare_v16_stage1_pending",
]
