from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from quant_investor.factors.governance_canonical_replay_v4 import (
    ARM_NAMES,
    CONTROL_CHAIN_STAGES,
    CanonicalReplayV4Error,
    canonical_file_bytes,
    readback_v4_evidence,
    semantic_sha256,
    stage_byte_sha256,
    validate_canonical_replay_v4,
    validate_v4_evidence,
)
from quant_investor.factors.runtime import production_factor_set_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _factor_record(name: str, *, challenger: str) -> dict:
    index = int(name.split("_")[-1]) if name.startswith("factor_") else 0
    return {
        "name": name,
        "family": "family_0" if name == challenger or index == 0 else f"family_{index}",
        "slot": "family_0::slot_0" if name == challenger or index == 0 else f"family_{index}::slot",
        "state": "production_candidate" if name == challenger else "production_factor",
        "registry_record_sha256": _digest(f"record:{name}"),
    }


def _replay(*, edge_lower: float = 0.01) -> dict:
    factor_set = [f"factor_{index}" for index in range(10)]
    incumbent = "factor_0"
    challenger = "challenger"
    registry_sha = _digest("registry")
    factor_set_sha = production_factor_set_sha256(factor_set)
    context = {
        "eligibility_contract_sha256": _digest("eligibility-contract"),
        "calendar_sha256": _digest("calendar"),
        "pit_sha256": _digest("pit"),
        "runtime_contract_sha256": _digest("runtime-contract"),
    }
    context_sha = semantic_sha256(
        {
            "registry_file_sha256": registry_sha,
            "production_factor_set_sha256": factor_set_sha,
            **context,
        }
    )
    factor_sets = {
        "A": factor_set,
        "B": [name for name in factor_set if name != incumbent],
        "C": sorted([name for name in factor_set if name != incumbent] + [challenger]),
        "D": sorted([name for name in factor_set if name != incumbent] + [challenger]),
    }
    stages: list[dict] = []
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        semantic_by_stage: dict[str, str] = {}
        ic_output_hashes: dict[str, str] = {}
        for stage in CONTROL_CHAIN_STAGES:
            if stage == "eligibility":
                output = {
                    "schema_version": "factor-governance-eligibility-output.v4",
                    "eligible_symbols": ["AAA"],
                    "eligibility_contract_sha256": context["eligibility_contract_sha256"],
                }
            elif stage == "quant":
                output = {
                    "schema_version": "factor-governance-quant-output.v4",
                    "scored_symbols": ["AAA"],
                    "selected_factors": factor_sets[arm],
                    "factor_records": [
                        _factor_record(name, challenger=challenger) for name in factor_sets[arm]
                    ],
                }
            elif stage == "funnel":
                output = {
                    "schema_version": "factor-governance-funnel-output.v4",
                    "eligible_symbols": ["AAA"],
                }
            elif stage == "codex_s1":
                output = {
                    "schema_version": "factor-governance-codex-s1-output.v4",
                    "advisory_scores": {"AAA": 0.6},
                }
            elif stage == "bayesian":
                output = {
                    "schema_version": "factor-governance-bayesian-output.v4",
                    "posterior_scores": {"AAA": 0.7},
                    "codex_s1_semantic_sha256": semantic_by_stage["codex_s1"],
                }
            elif stage == "risk_advisor":
                output = {
                    "schema_version": "factor-governance-risk-advisor-output.v4",
                    "advisory_only": True,
                    "decisions": {"AAA": "reject"},
                    "bayesian_semantic_sha256": semantic_by_stage["bayesian"],
                }
            elif stage == "codex_ic":
                ic_input = {
                    "symbol": "AAA",
                    "upstream_stage_sha256s": {
                        name: semantic_by_stage[name]
                        for name in CONTROL_CHAIN_STAGES[: CONTROL_CHAIN_STAGES.index("codex_ic")]
                    },
                    "ic_hints": {},
                }
                decision = {
                    "schema_version": "codex-ic-decision.v4",
                    "symbol": "AAA",
                    "action": "buy",
                }
                ic_output_hashes = {"AAA": semantic_sha256(decision)}
                output = {
                    "schema_version": "factor-governance-codex-ic-output.v4",
                    "inputs": {"AAA": ic_input},
                    "input_sha256s": {"AAA": semantic_sha256(ic_input)},
                    "decisions": {"AAA": decision},
                    "output_sha256s": ic_output_hashes,
                }
            else:
                output = {
                    "schema_version": "factor-governance-portfolio-output.v4",
                    "target_weights": {"AAA": 0.5},
                    "codex_ic_decision_sha256s": ic_output_hashes,
                }
            semantic_sha = semantic_sha256(output)
            byte_sha = stage_byte_sha256(
                arm=arm,
                stage=stage,
                context_sha256=context_sha,
                predecessor=predecessor,
                output=output,
            )
            stages.append(
                {
                    "schema_version": "factor-governance-canonical-stage.v4",
                    "arm": arm,
                    "stage": stage,
                    "context_sha256": context_sha,
                    "byte_sha256": byte_sha,
                    "semantic_sha256": semantic_sha,
                    "predecessor": predecessor,
                    "output": output,
                }
            )
            semantic_by_stage[stage] = semantic_sha
            predecessor = {
                "kind": "stage",
                "byte_sha256": byte_sha,
                "semantic_sha256": semantic_sha,
            }
    return {
        "schema_version": "factor-governance-canonical-replay.v4",
        "protocol_version": "v4",
        "run_id": "v4-replay-test",
        "as_of": "2026-07-17",
        "registry_file_sha256": registry_sha,
        "production_factor_set_sha256": factor_set_sha,
        "context": context,
        "context_sha256": context_sha,
        "factor_set": factor_set,
        "comparison": {
            "incumbent": incumbent,
            "challenger": challenger,
            "slot": "family_0::slot_0",
            "incremental_edge_ci95_lower": edge_lower,
        },
        "stages": stages,
    }


def _evidence(replay: dict, path: str) -> dict:
    raw = canonical_file_bytes(replay)
    return {
        "schema_version": "factor-governance-replay-evidence.v4",
        "status": "verified",
        "factor_name": "challenger",
        "registry_file_sha256": replay["registry_file_sha256"],
        "replay_path": path,
        "replay_file_sha256": hashlib.sha256(raw).hexdigest(),
        "replay_semantic_sha256": semantic_sha256(replay),
        **replay["context"],
        "replay": replay,
    }


def test_v4_replay_binds_exact_chain_and_risk_advisor_is_not_positive_weight_gate() -> None:
    replay = _replay()
    normalized = validate_canonical_replay_v4(replay)
    assert normalized["positive_weight_depends_on_risk_advisor_approval"] is False
    assert all(
        normalized["arms"][arm]["risk_advisor"]["decisions"]["AAA"] == "reject" for arm in ARM_NAMES
    )
    assert all(
        normalized["arms"][arm]["portfolio_constructor"]["target_weights"]["AAA"] == 0.5
        for arm in ARM_NAMES
    )


def test_v4_target_10_replacement_requires_positive_incremental_edge_lower_bound() -> None:
    with pytest.raises(CanonicalReplayV4Error, match="incremental edge"):
        validate_canonical_replay_v4(_replay(edge_lower=0.0))


def test_v4_portfolio_positive_weight_still_requires_hash_bound_codex_ic_buy() -> None:
    replay = _replay()
    ic_index = next(
        index
        for index, row in enumerate(replay["stages"])
        if row["arm"] == "A" and row["stage"] == "codex_ic"
    )
    ic_stage = replay["stages"][ic_index]
    portfolio_stage = replay["stages"][ic_index + 1]
    decision = ic_stage["output"]["decisions"]["AAA"]
    decision["action"] = "hold"
    decision_sha = semantic_sha256(decision)
    ic_stage["output"]["output_sha256s"]["AAA"] = decision_sha
    portfolio_stage["output"]["codex_ic_decision_sha256s"]["AAA"] = decision_sha
    for index in (ic_index, ic_index + 1):
        row = replay["stages"][index]
        if index > ic_index:
            previous = replay["stages"][index - 1]
            row["predecessor"] = {
                "kind": "stage",
                "byte_sha256": previous["byte_sha256"],
                "semantic_sha256": previous["semantic_sha256"],
            }
        row["semantic_sha256"] = semantic_sha256(row["output"])
        row["byte_sha256"] = stage_byte_sha256(
            arm=row["arm"],
            stage=row["stage"],
            context_sha256=row["context_sha256"],
            predecessor=row["predecessor"],
            output=row["output"],
        )
    with pytest.raises(CanonicalReplayV4Error, match="CodexIC BUY"):
        validate_canonical_replay_v4(replay)


def test_v4_evidence_rejects_v2_v3_and_supports_exact_0600_readback(
    tmp_path: Path,
) -> None:
    with pytest.raises(CanonicalReplayV4Error, match="unsupported"):
        validate_v4_evidence({"schema_version": "factor-governance-replay-evidence.v3"})
    replay = _replay()
    path = (tmp_path / "canonical-replay.v4.json").resolve()
    path.write_bytes(canonical_file_bytes(replay))
    path.chmod(0o600)
    evidence = _evidence(replay, str(path))
    assert validate_v4_evidence(evidence)["factor_name"] == "challenger"
    readback = readback_v4_evidence(evidence)
    assert readback["complete_chain_hash_binding_verified"] is True
    assert readback["positive_weight_depends_on_risk_advisor_approval"] is False

    forged = copy.deepcopy(evidence)
    forged["replay_semantic_sha256"] = _digest("forged")
    with pytest.raises(CanonicalReplayV4Error, match="semantic SHA"):
        validate_v4_evidence(forged)
