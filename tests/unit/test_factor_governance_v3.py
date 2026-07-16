from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import pytest

from quant_investor.factors.factor_baseline_bootstrap import (
    FactorBaselineBootstrapError,
    build_factor_baseline_bootstrap_plan,
    validate_bootstrap_candidates,
)
from quant_investor.factors.governance_canonical_replay_v3 import (
    ARM_NAMES,
    CONTROL_CHAIN_STAGES,
    CanonicalReplayV3Error,
    semantic_sha256,
    validate_canonical_replay_v3,
    validate_v3_evidence,
)
from quant_investor.factors.governance_protocol_v3 import governance_runtime_status
from quant_investor.factors.runtime import MinedFactorRegistry


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _factor(name: str, state: str) -> dict[str, str]:
    return {
        "name": name,
        "family": "value",
        "slot": "value::primary",
        "registry_state": state,
        "registry_record_sha256": _digest(f"record:{name}"),
    }


def _replay() -> dict:
    context_sha = _digest("context")
    factor_sets = {
        "A": ["incumbent"],
        "B": [],
        "C": ["challenger"],
        "D": ["challenger"],
    }
    stages = []
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        for stage in CONTROL_CHAIN_STAGES:
            if stage == "deterministic_funnel":
                output = {
                    "schema_version": "factor-governance-funnel-output.v3",
                    "eligible_symbols": ["AAA"],
                }
            elif stage == "quant":
                selected = factor_sets[arm]
                output = {
                    "schema_version": "factor-governance-quant-output.v3",
                    "eligible_symbols": ["AAA"],
                    "selected_factors": selected,
                    "factor_records": [
                        _factor(
                            name,
                            "production_factor" if name == "incumbent" else "mature_candidate",
                        )
                        for name in selected
                    ],
                    "branches": {
                        name: {
                            "ready": True,
                            "object_sha256": _digest(f"{arm}:{name}:object"),
                            "semantic_sha256": _digest(f"{arm}:{name}:semantic"),
                        }
                        for name in ("quant", "fundamental", "macro")
                    },
                    "likelihood_branches": ["fundamental", "quant"],
                }
            elif stage == "bayesian":
                output = {
                    "schema_version": "factor-governance-bayesian-output.v3",
                    "posterior_scores": {"AAA": 0.7},
                }
            elif stage == "risk_guard":
                output = {
                    "schema_version": "factor-governance-risk-output.v3",
                    "decisions": {"AAA": "approved"},
                }
            else:
                output = {
                    "schema_version": "factor-governance-portfolio-output.v3",
                    "target_weights": {"AAA": 0.5},
                }
            byte_sha = _digest(f"{arm}:{stage}:bytes")
            semantic_sha = semantic_sha256(output)
            stages.append(
                {
                    "schema_version": "factor-governance-canonical-stage.v3",
                    "arm": arm,
                    "stage": stage,
                    "context_sha256": context_sha,
                    "byte_sha256": byte_sha,
                    "semantic_sha256": semantic_sha,
                    "predecessor": predecessor,
                    "output": output,
                }
            )
            predecessor = {
                "kind": "stage",
                "byte_sha256": byte_sha,
                "semantic_sha256": semantic_sha,
            }
    return {
        "schema_version": "factor-governance-canonical-replay.v3",
        "protocol_version": "v3",
        "run_id": "v3-test",
        "as_of": "2026-07-16",
        "registry_file_sha256": _digest("registry"),
        "production_factor_set_sha256": _digest("factor-set"),
        "context_sha256": context_sha,
        "factor_set": ["incumbent"],
        "comparison": {
            "incumbent": "incumbent",
            "challenger": "challenger",
            "slot": "value::primary",
        },
        "stages": stages,
    }


def test_v3_replay_validates_exact_five_stage_graph_and_rejects_v2() -> None:
    result = validate_canonical_replay_v3(_replay())
    assert len(CONTROL_CHAIN_STAGES) == 5
    assert set(result["arms"]) == set(ARM_NAMES)
    legacy = _replay()
    legacy["schema_version"] = "factor-governance-canonical-replay-bundle.v1"
    with pytest.raises(CanonicalReplayV3Error, match="unsupported"):
        validate_canonical_replay_v3(legacy)
    with pytest.raises(CanonicalReplayV3Error, match="unsupported"):
        validate_v3_evidence({"schema_version": "factor-governance-replay-evidence.v2"})


def test_v3_replay_enforces_symbol_domain_risk_and_portfolio_subset() -> None:
    replay = _replay()
    risk_index = next(
        index
        for index, row in enumerate(replay["stages"])
        if row["arm"] == "A" and row["stage"] == "risk_guard"
    )
    portfolio_index = risk_index + 1
    replay["stages"][risk_index]["output"]["decisions"]["AAA"] = "rejected"
    replay["stages"][risk_index]["semantic_sha256"] = semantic_sha256(
        replay["stages"][risk_index]["output"]
    )
    replay["stages"][portfolio_index]["predecessor"]["semantic_sha256"] = replay["stages"][
        risk_index
    ]["semantic_sha256"]
    with pytest.raises(CanonicalReplayV3Error, match="RiskGuard approval"):
        validate_canonical_replay_v3(replay)


def _candidate(index: int) -> dict:
    name = f"factor_{index}"
    family = f"family_{index // 2}"
    contract = {
        "schema_version": "factor-production-runtime-contract.v1",
        "factor_name": name,
    }
    return {
        "name": name,
        "family": family,
        "slot": f"{family}::slot_{index}",
        "direction": -1.0 if index % 2 else 1.0,
        "raw_weight": 1.0,
        "registry_record_sha256": _digest(f"record:{name}"),
        "runtime_contract": contract,
        "evidence": {
            "schema_version": "factor-governance-replay-evidence.v3",
            "status": "verified",
            "factor_name": name,
            "registry_file_sha256": _digest("registry"),
            "replay_semantic_sha256": _digest(f"replay:{name}"),
            "calendar_sha256": _digest("calendar"),
            "pit_sha256": _digest("pit"),
            "runtime_contract_sha256": semantic_sha256(contract),
        },
    }


def test_bootstrap_is_plan_only_and_enforces_diversity_slots_and_caps() -> None:
    validated = validate_bootstrap_candidates([_candidate(index) for index in range(6)])
    assert validated["factor_count"] == 6
    assert max(validated["absolute_weights"].values()) <= 0.20
    assert max(validated["family_absolute_weights"].values()) <= 0.35
    assert any(weight < 0 for weight in validated["signed_weights"].values())
    assert "write" not in inspect.signature(build_factor_baseline_bootstrap_plan).parameters
    duplicate = [_candidate(index) for index in range(6)]
    duplicate[1]["slot"] = duplicate[0]["slot"]
    with pytest.raises(FactorBaselineBootstrapError, match="slots"):
        validate_bootstrap_candidates(duplicate)


def test_current_registry_remains_blocked_and_unchanged() -> None:
    path = Path("quant_investor/factor_registry/mined_factors.json")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    registry = MinedFactorRegistry.load_production(path)
    status = governance_runtime_status(registry)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert status["status"] == "governance_blocked"
    assert "registry_protocol_version_mismatch" in status["blockers"]
    assert before == after
