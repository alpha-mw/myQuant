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
    canonical_file_bytes,
    readback_v3_evidence,
    semantic_sha256,
    stage_byte_sha256,
    validate_canonical_replay_v3,
    validate_v3_evidence,
)
import quant_investor.factors.governance_protocol_v3 as protocol_v3
from quant_investor.factors.governance_protocol_v3 import (
    CANONICAL_PRODUCER_AUTHENTICATION_BLOCKER,
    canonical_replay_producer_control,
    governance_runtime_status,
    protocol_policy,
)
from quant_investor.factors.runtime import (
    MinedFactorRegistry,
    production_factor_set_sha256,
)


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
    runtime_contract_sha256: str | None = None,
) -> dict:
    registry_sha = registry_file_sha256 or _digest("registry")
    production_factor_set_sha = production_factor_set_sha256(["incumbent"])
    context = {
        "calendar_sha256": _digest("calendar"),
        "pit_sha256": _digest("pit"),
        "runtime_contract_sha256": (
            runtime_contract_sha256 or _digest("runtime-contract")
        ),
    }
    context_sha = semantic_sha256(
        {
            "registry_file_sha256": registry_sha,
            "production_factor_set_sha256": production_factor_set_sha,
            **context,
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
            if stage == "quant":
                selected = factor_sets[arm]
                output = {
                    "schema_version": "factor-governance-quant-output.v3",
                    "scored_symbols": ["AAA", "BBB"],
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
            elif stage == "deterministic_funnel":
                output = {
                    "schema_version": "factor-governance-funnel-output.v3",
                    "eligible_symbols": ["AAA"],
                }
            elif stage == "bayesian":
                output = {
                    "schema_version": "factor-governance-bayesian-output.v3",
                    "posterior_scores": {"AAA": 0.7},
                }
            elif stage == "risk_guard":
                risk_decision = {
                    "schema_version": "risk-decision.v15",
                    "action_cap": "buy",
                    "blocked_symbols": [],
                }
                output = {
                    "schema_version": "factor-governance-risk-output.v3",
                    "decisions": {"AAA": "approved"},
                    "risk_decision": risk_decision,
                    "risk_decision_sha256": semantic_sha256(risk_decision),
                }
            elif stage == "ic_coordinator":
                risk_output = stages[-1]["output"]
                ic_input = {
                    "branch_verdicts": {
                        name: {
                            "schema_version": "branch-verdict.v15",
                            "branch": name,
                            "score": 0.5,
                        }
                        for name in ("quant", "fundamental", "macro")
                    },
                    "risk_decision": risk_output["risk_decision"],
                    "ic_hints": {},
                }
                ic_decision = {
                    "schema_version": "ic-decision.v15",
                    "symbol": "AAA",
                    "action": "buy",
                    "status": "success",
                }
                output = {
                    "schema_version": "factor-governance-ic-output.v3",
                    "inputs": {"AAA": ic_input},
                    "input_sha256s": {"AAA": semantic_sha256(ic_input)},
                    "decisions": {"AAA": ic_decision},
                    "output_sha256s": {"AAA": semantic_sha256(ic_decision)},
                }
            else:
                ic_output = stages[-1]["output"]
                output = {
                    "schema_version": "factor-governance-portfolio-output.v3",
                    "target_weights": {"AAA": 0.5},
                    "ic_decision_sha256s": dict(ic_output["output_sha256s"]),
                }
            byte_sha = stage_byte_sha256(
                arm=arm,
                stage=stage,
                context_sha256=context_sha,
                predecessor=predecessor,
                output=output,
            )
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
        "context": context,
        "context_sha256": context_sha,
        "factor_set": ["incumbent"],
        "comparison": {
            "incumbent": "incumbent",
            "challenger": challenger,
            "slot": "value::primary",
        },
        "stages": stages,
    }


def _rehash_arm_from(replay: dict, start_index: int) -> None:
    arm = replay["stages"][start_index]["arm"]
    for index in range(start_index, len(replay["stages"])):
        item = replay["stages"][index]
        if item["arm"] != arm:
            break
        if index > start_index:
            previous = replay["stages"][index - 1]
            item["predecessor"] = {
                "kind": "stage",
                "byte_sha256": previous["byte_sha256"],
                "semantic_sha256": previous["semantic_sha256"],
            }
        item["semantic_sha256"] = semantic_sha256(item["output"])
        item["byte_sha256"] = stage_byte_sha256(
            arm=item["arm"],
            stage=item["stage"],
            context_sha256=item["context_sha256"],
            predecessor=item["predecessor"],
            output=item["output"],
        )


def test_v3_replay_validates_exact_runtime_graph_and_rejects_v2() -> None:
    result = validate_canonical_replay_v3(_replay())
    assert CONTROL_CHAIN_STAGES == (
        "quant",
        "deterministic_funnel",
        "bayesian",
        "risk_guard",
        "ic_coordinator",
        "portfolio_constructor",
    )
    assert protocol_policy()["canonical_chain"] == list(CONTROL_CHAIN_STAGES)
    assert set(result["arms"]) == set(ARM_NAMES)
    legacy = _replay()
    legacy["schema_version"] = "factor-governance-canonical-replay-bundle.v1"
    with pytest.raises(CanonicalReplayV3Error, match="unsupported"):
        validate_canonical_replay_v3(legacy)
    with pytest.raises(CanonicalReplayV3Error, match="unsupported"):
        validate_v3_evidence({"schema_version": "factor-governance-replay-evidence.v2"})


def test_v3_replay_recomputes_factor_set_sha_from_a_arm_set() -> None:
    replay = _replay()
    replay["production_factor_set_sha256"] = production_factor_set_sha256(
        ["different-incumbent"]
    )
    replay["context_sha256"] = semantic_sha256(
        {
            "registry_file_sha256": replay["registry_file_sha256"],
            "production_factor_set_sha256": replay[
                "production_factor_set_sha256"
            ],
            **replay["context"],
        }
    )
    for stage in replay["stages"]:
        stage["context_sha256"] = replay["context_sha256"]
    with pytest.raises(CanonicalReplayV3Error, match="factor-set SHA mismatch"):
        validate_canonical_replay_v3(replay)


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
        "replay_path": (
            "/private/tmp/unavailable-factor-governance-replay.json"
        ),
        "replay_file_sha256": hashlib.sha256(
            canonical_file_bytes(replay_payload)
        ).hexdigest(),
        "replay_semantic_sha256": semantic_sha256(replay_payload),
        **replay_payload["context"],
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
        "production_factor_set_sha256": production_factor_set_sha256(
            ["incumbent"]
        ),
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
    ] is False
    assert "canonical_evidence_not_readback_bound" in status["blockers"]
    assert "registry_v3_evidence_runtime_contract_sha_mismatch" in status["blockers"]
    assert status["status"] == "governance_blocked"


def test_v3_replay_enforces_symbol_domain_risk_and_portfolio_subset() -> None:
    replay = _replay()
    risk_index = next(
        index
        for index, row in enumerate(replay["stages"])
        if row["arm"] == "A" and row["stage"] == "risk_guard"
    )
    replay["stages"][risk_index]["output"]["decisions"]["AAA"] = "rejected"
    _rehash_arm_from(replay, risk_index)
    with pytest.raises(CanonicalReplayV3Error, match="RiskGuard approval"):
        validate_canonical_replay_v3(replay)


def test_v3_replay_binds_ic_inputs_outputs_and_portfolio_consumption() -> None:
    replay = _replay()
    ic_index = next(
        index
        for index, row in enumerate(replay["stages"])
        if row["arm"] == "A" and row["stage"] == "ic_coordinator"
    )
    replay["stages"][ic_index]["output"]["inputs"]["AAA"]["branch_verdicts"][
        "quant"
    ]["score"] = 0.9
    _rehash_arm_from(replay, ic_index)
    with pytest.raises(CanonicalReplayV3Error, match="input SHA mismatch"):
        validate_canonical_replay_v3(replay)

    replay = _replay()
    ic_index = next(
        index
        for index, row in enumerate(replay["stages"])
        if row["arm"] == "A" and row["stage"] == "ic_coordinator"
    )
    decision = replay["stages"][ic_index]["output"]["decisions"]["AAA"]
    decision["action"] = "hold"
    decision_sha = semantic_sha256(decision)
    replay["stages"][ic_index]["output"]["output_sha256s"]["AAA"] = decision_sha
    replay["stages"][ic_index + 1]["output"]["ic_decision_sha256s"][
        "AAA"
    ] = decision_sha
    _rehash_arm_from(replay, ic_index)
    with pytest.raises(CanonicalReplayV3Error, match="ICCoordinator BUY"):
        validate_canonical_replay_v3(replay)


def test_v3_evidence_requires_real_canonical_readback_without_auth_claim(
    tmp_path: Path,
) -> None:
    replay = _replay()
    path = (tmp_path / "canonical-replay.v3.json").resolve()
    raw = canonical_file_bytes(replay)
    path.write_bytes(raw)
    path.chmod(0o600)
    evidence = {
        "schema_version": "factor-governance-replay-evidence.v3",
        "status": "verified",
        "factor_name": "challenger",
        "registry_file_sha256": replay["registry_file_sha256"],
        "replay_path": str(path),
        "replay_file_sha256": hashlib.sha256(raw).hexdigest(),
        "replay_semantic_sha256": semantic_sha256(replay),
        **replay["context"],
        "replay": replay,
    }

    readback = readback_v3_evidence(evidence)
    control = canonical_replay_producer_control(evidence)

    assert readback["local_bytes_readback_verified"] is True
    assert readback["ic_input_output_hash_binding_verified"] is True
    assert control["producer_implemented"] is True
    assert control["producer_available"] is True
    assert control["local_bytes_readback_verified"] is True
    assert control["ic_input_output_hash_binding_verified"] is True
    assert control["canonical_producer_authenticated"] is False
    assert control["blocker"] == CANONICAL_PRODUCER_AUTHENTICATION_BLOCKER

    path.write_bytes(raw + b" ")
    path.chmod(0o600)
    failed_control = canonical_replay_producer_control(evidence)
    assert failed_control["local_bytes_readback_verified"] is False
    assert failed_control["canonical_producer_authenticated"] is False


def _candidate(index: int) -> dict:
    name = f"factor_{index}"
    family = f"family_{index // 2}"
    contract = {
        "schema_version": "factor-production-runtime-contract.v1",
        "factor_name": name,
    }
    replay = _replay(
        challenger=name,
        registry_file_sha256=_digest("registry"),
        runtime_contract_sha256=semantic_sha256(contract),
    )
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
            "replay_path": "/private/tmp/unavailable-factor-governance-replay.json",
            "replay_file_sha256": hashlib.sha256(
                canonical_file_bytes(replay)
            ).hexdigest(),
            "replay_semantic_sha256": semantic_sha256(replay),
            **replay["context"],
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
