from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

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
import quant_investor.factors.governance_protocol_v3 as protocol_v3
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


def _replay(
    *,
    challenger: str = "challenger",
    registry_file_sha256: str | None = None,
) -> dict:
    registry_sha = registry_file_sha256 or _digest("registry")
    production_factor_set_sha = _digest("factor-set")
    calendar_sha = _digest("calendar")
    pit_sha = _digest("pit")
    context_sha = semantic_sha256(
        {
            "registry_file_sha256": registry_sha,
            "production_factor_set_sha256": production_factor_set_sha,
            "calendar_sha256": calendar_sha,
            "pit_sha256": pit_sha,
        }
    )
    factor_sets = {
        "A": ["incumbent"],
        "B": [],
        "C": [challenger],
        "D": [challenger],
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
        "registry_file_sha256": registry_sha,
        "production_factor_set_sha256": production_factor_set_sha,
        "calendar_sha256": calendar_sha,
        "pit_sha256": pit_sha,
        "context_sha256": context_sha,
        "factor_set": ["incumbent"],
        "comparison": {
            "incumbent": "incumbent",
            "challenger": challenger,
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


def _verified_evidence(
    factor_name: str = "challenger",
    *,
    replay: dict | None = None,
) -> dict:
    replay_payload = replay or _replay(challenger=factor_name)
    return {
        "schema_version": "factor-governance-replay-evidence.v3",
        "status": "verified",
        "factor_name": factor_name,
        "registry_file_sha256": replay_payload["registry_file_sha256"],
        "replay_semantic_sha256": semantic_sha256(replay_payload),
        "calendar_sha256": _digest("calendar"),
        "pit_sha256": _digest("pit"),
        "runtime_contract_sha256": _digest("runtime-contract"),
        "replay": replay_payload,
    }


def test_v3_evidence_recomputes_replay_hash_and_identity() -> None:
    evidence = _verified_evidence()
    assert validate_v3_evidence(evidence)["factor_name"] == "challenger"

    forged = copy.deepcopy(evidence)
    forged["replay_semantic_sha256"] = _digest("forged")
    with pytest.raises(CanonicalReplayV3Error, match="semantic SHA mismatch"):
        validate_v3_evidence(forged)

    mismatched = copy.deepcopy(evidence)
    mismatched["factor_name"] = "different-factor"
    with pytest.raises(CanonicalReplayV3Error, match="factor identity mismatch"):
        validate_v3_evidence(mismatched)


def test_runtime_rejects_evidence_bound_to_different_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _verified_evidence()
    records = [
        SimpleNamespace(
            name=f"factor_{index}",
            weight=1.0,
            category=f"family_{index // 2}",
            metadata={
                "factor_family": f"family_{index // 2}",
                "dominant_primitive_cluster": f"slot_{index}",
            },
        )
        for index in range(6)
    ]
    manifest = {
        "production_factor_count": 6,
        "production_factor_names": [record.name for record in records],
        "production_factor_set_sha256": _digest("factor-set"),
    }
    registry = SimpleNamespace(
        metadata={
            **manifest,
            "strict_loader": True,
            "registry_sha256": _digest("registry"),
            "factor_governance_protocol_version": "v3",
            "factor_governance_protocol_hash": protocol_v3.protocol_hash(),
            "factor_governance_v3_evidence": evidence,
        },
        selectable_factors=lambda: records,
        selectable_manifest=lambda: manifest,
    )
    monkeypatch.setattr(
        protocol_v3,
        "validate_production_runtime_contracts",
        lambda *_args, **_kwargs: {
            "status": "ready",
            "contracts": {},
            "contracts_sha256": _digest("different-runtime-contract"),
            "implementation_code_sha256s": {},
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        protocol_v3,
        "validate_quant_production_activation",
        lambda *_args, **_kwargs: {"status": "ready", "blockers": []},
    )
    status = governance_runtime_status(registry)
    assert status["canonical_replay_producer_control"][
        "canonical_producer_authenticated"
    ] is True
    assert "registry_v3_evidence_runtime_contract_sha_mismatch" in status["blockers"]
    assert status["status"] == "governance_blocked"


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
    replay = _replay(challenger=name, registry_file_sha256=_digest("registry"))
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
            "replay_semantic_sha256": semantic_sha256(replay),
            "calendar_sha256": _digest("calendar"),
            "pit_sha256": _digest("pit"),
            "runtime_contract_sha256": semantic_sha256(contract),
            "replay": replay,
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

    forged = [_candidate(index) for index in range(6)]
    forged[0]["evidence"]["replay_semantic_sha256"] = _digest("arbitrary")
    with pytest.raises(FactorBaselineBootstrapError, match="semantic SHA mismatch"):
        validate_bootstrap_candidates(forged)


def test_current_registry_remains_blocked_and_unchanged() -> None:
    path = Path("quant_investor/factor_registry/mined_factors.json")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    registry = MinedFactorRegistry.load_production(path)
    status = governance_runtime_status(registry)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert status["status"] == "governance_blocked"
    assert "registry_protocol_version_mismatch" in status["blockers"]
    assert before == after
