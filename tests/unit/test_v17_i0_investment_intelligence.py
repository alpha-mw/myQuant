from __future__ import annotations

import ast
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_investor.intelligence._core import (
    NO_AUTHORITY,
    ZERO_SHA256,
    IntelligenceContractError,
    content_ref,
    seal_content_addressed,
)
from quant_investor.intelligence.bayesian import update_hypothesis, validate_bayesian_receipt
from quant_investor.intelligence.evidence import build_ai_draft, build_evidence
from quant_investor.intelligence.evidence.forward_adapter import (
    BUNDLE_VERSION,
    build_observation_evidence_bundle,
    validate_observation_evidence_bundle,
)
from quant_investor.intelligence.fusion import (
    build_fundamental_branch,
    build_quant_branch,
    fuse_research_branches,
)
from quant_investor.intelligence.hypothesis import build_hypothesis, validate_hypothesis
from quant_investor.intelligence.memory import (
    append_memory,
    memory_tip,
    validate_memory_chain,
)
from quant_investor.intelligence.package import verify_package
from quant_investor.intelligence.regime import (
    INDUSTRY_STATES,
    MARKET_STATES,
    THEME_STATES,
    build_regime_input,
    infer_multilayer_regime,
)
from quant_investor.intelligence.runtime import (
    INTELLIGENCE_RUNTIME_RECEIPT_VERSION,
    build_intelligence_runtime_receipt,
    verify_runtime_receipt,
)
from quant_investor.v17_v4_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_runtime.forward_evaluation_receipt import (
    build_existing_factor_inventory,
    build_forward_evidence_origin_inventory,
)
from quant_investor.v17_v4_runtime.orchestrator import build_forward_request

AS_OF = "2026-01-02T07:00:00Z"
LATER = "2026-01-05T07:00:00Z"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
STRATEGY = "cn-forward-research"
V4_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


def _ref(
    artifact_id: str,
    artifact_version: str,
    relative_path: str,
    *,
    byte_sha256: str = SHA_A,
    semantic_sha256: str = SHA_A,
    cutoff: str = AS_OF,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": byte_sha256,
        "cutoff": cutoff,
        "relative_path": relative_path,
        "semantic_sha256": semantic_sha256,
        "strategy_id": STRATEGY,
    }


def _source_ref(name: str = "source-a") -> dict[str, str]:
    return _ref(
        name,
        "myquant.v17.v4.research-source.v1",
        f"data/private/v17_v4_sources/{name}.json",
    )


def _evidence(
    direction: str = "POSITIVE",
    *,
    likelihood: str | None = None,
    available_at: str = AS_OF,
    name: str = "source-a",
    strength: str = "1",
    source_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ratios = {
        "CONTRARY": "0.4",
        "NEGATIVE": "0.5",
        "NEUTRAL": "1",
        "POSITIVE": "2",
    }
    return build_evidence(
        source_type="QUANT",
        direction=direction,
        likelihood_ratio=ratios[direction] if likelihood is None else likelihood,
        strength=strength,
        reason=f"{direction.lower()}-{name}",
        observed_at=AS_OF,
        available_at=available_at,
        source_ref=_source_ref(name) if source_ref is None else source_ref,
    )


def _regime_input(
    *,
    source_ref: Mapping[str, Any] | None = None,
    available_at: str = AS_OF,
) -> dict[str, Any]:
    state_sets = {
        "industry": INDUSTRY_STATES,
        "market": MARKET_STATES,
        "theme": THEME_STATES,
    }
    previous: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    emissions: dict[str, Any] = {}
    for layer, states in state_sets.items():
        share = Decimal("1") / Decimal(len(states))
        previous[layer] = {state: str(share) for state in states}
        transitions[layer] = {
            source: {target: "1" if source == target else "0" for target in states}
            for source in states
        }
        emissions[layer] = {
            state: "1" if index == 0 else "0.25" for index, state in enumerate(states)
        }
    return build_regime_input(
        previous_distributions=previous,
        transition_matrices=transitions,
        emission_likelihoods=emissions,
        source_refs=[_source_ref() if source_ref is None else source_ref],
        observed_at=AS_OF,
        available_at=available_at,
    )


def _hypothesis(positive: dict[str, Any], contrary: dict[str, Any]) -> dict[str, Any]:
    return build_hypothesis(
        thesis="Factor persistence may continue.",
        why_it_may_be_true="Observed breadth and RankIC support persistence.",
        what_would_make_it_fail="RankIC deterioration would falsify persistence.",
        supporting_evidence=[positive],
        contrary_evidence=[contrary],
        expected_window_start="2026-01-03T07:00:00Z",
        expected_window_end="2026-03-31T07:00:00Z",
        falsification_conditions=[
            {
                "metric_id": "rank_ic",
                "operator": "LT",
                "threshold": "0",
                "window_sessions": 20,
            }
        ],
        related_companies=["000001.SZ"],
        related_industries=["banks"],
        as_of=AS_OF,
    )


def _branches(
    positive: dict[str, Any], contrary: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    quant = build_quant_branch(
        factor_score="0.8",
        rank_ic="0.1",
        icir="0.2",
        exposure="0.3",
        coverage="0.9",
        confidence="0.8",
        availability="1",
        evidence=[positive],
        as_of=AS_OF,
    )
    fundamental = build_fundamental_branch(
        quality="0.6",
        earnings="0.5",
        valuation="0.4",
        industry_position="0.7",
        confidence="0.5",
        availability="0.5",
        evidence=[contrary],
        as_of=AS_OF,
    )
    return quant, fundamental


def _reseal(document: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    body = deepcopy(dict(document))
    body.pop(identity_field, None)
    body.pop("semantic_sha256", None)
    return seal_content_addressed(body, identity_field=identity_field)


def test_01_prior_deterministic() -> None:
    evidence = [_evidence()]
    first = update_hypothesis(
        hypothesis_id="hypothesis-a", prior="0.25", evidence=evidence, as_of=AS_OF
    )
    second = update_hypothesis(
        hypothesis_id="hypothesis-a", prior="0.25", evidence=evidence, as_of=AS_OF
    )
    assert first == second
    assert first["prior"] == "0.250000000000"


def test_02_positive_evidence_update() -> None:
    result = update_hypothesis(
        hypothesis_id="hypothesis-a",
        prior="0.5",
        evidence=[_evidence("POSITIVE")],
        as_of=AS_OF,
    )
    assert Decimal(result["posterior"]) > Decimal(result["prior"])


def test_03_negative_evidence_update() -> None:
    result = update_hypothesis(
        hypothesis_id="hypothesis-a",
        prior="0.5",
        evidence=[_evidence("NEGATIVE")],
        as_of=AS_OF,
    )
    assert Decimal(result["posterior"]) < Decimal(result["prior"])


def test_04_contrary_evidence_is_retained_and_reduces_posterior() -> None:
    result = update_hypothesis(
        hypothesis_id="hypothesis-a",
        prior="0.5",
        evidence=[_evidence("CONTRARY")],
        as_of=AS_OF,
    )
    assert result["direction_counts"]["CONTRARY"] == 1
    assert Decimal(result["posterior"]) < Decimal(result["prior"])


def test_05_same_evidence_same_posterior_regardless_of_input_order() -> None:
    first_evidence = _evidence(name="source-a")
    second_evidence = _evidence("NEGATIVE", name="source-b")
    first = update_hypothesis(
        hypothesis_id="hypothesis-a",
        prior="0.5",
        evidence=[first_evidence, second_evidence],
        as_of=AS_OF,
    )
    second = update_hypothesis(
        hypothesis_id="hypothesis-a",
        prior="0.5",
        evidence=[second_evidence, first_evidence],
        as_of=AS_OF,
    )
    assert first == second


def test_06_no_future_evidence() -> None:
    with pytest.raises(IntelligenceContractError, match="future evidence"):
        update_hypothesis(
            hypothesis_id="hypothesis-a",
            prior="0.5",
            evidence=[_evidence(available_at=LATER)],
            as_of=AS_OF,
        )


def test_evidence_rejects_source_not_available_at_declared_time() -> None:
    future_source = _ref(
        "source-a",
        "myquant.v17.v4.research-source.v1",
        "data/private/v17_v4_sources/source-a.json",
        cutoff=LATER,
    )
    with pytest.raises(IntelligenceContractError, match="not available at available_at"):
        _evidence(source_ref=future_source)


def test_bayesian_validator_replays_and_rejects_resealed_malformed_receipt() -> None:
    evidence = [_evidence()]
    receipt = update_hypothesis(
        hypothesis_id="hypothesis-a", prior="0.5", evidence=evidence, as_of=AS_OF
    )
    malformed = deepcopy(receipt)
    malformed.pop("posterior")
    with pytest.raises(IntelligenceContractError, match="replay mismatch"):
        validate_bayesian_receipt(
            _reseal(malformed, identity_field="receipt_id"),
            evidence=evidence,
            as_of=AS_OF,
        )


def test_07_markov_transition_deterministic() -> None:
    regime_input = _regime_input()
    first = infer_multilayer_regime(
        regime_input=regime_input,
        evidence=[_evidence()],
        as_of=AS_OF,
    )
    second = infer_multilayer_regime(
        regime_input=deepcopy(regime_input),
        evidence=[_evidence()],
        as_of=AS_OF,
    )
    assert first == second
    assert first["market_state"] == "BULL"
    assert first["industry_state"] == "EARLY_EXPANSION"
    assert first["theme_state"] == "EMERGING"


def test_08_markov_has_forward_filter_only_and_no_smoothing() -> None:
    regime_input = _regime_input()
    result = infer_multilayer_regime(
        regime_input=regime_input,
        evidence=[_evidence()],
        as_of=AS_OF,
    )
    expected = Decimal("1") / (Decimal("1") + Decimal("0.25") * 3)
    assert Decimal(result["posterior"]["market"]["BULL"]) == expected.quantize(
        Decimal("0.000000000001")
    )
    assert "smoothed_posterior" not in result


def test_09_regime_no_future_leakage() -> None:
    with pytest.raises(IntelligenceContractError, match="future evidence"):
        infer_multilayer_regime(
            regime_input=_regime_input(available_at=LATER),
            evidence=[_evidence()],
            as_of=AS_OF,
        )


def test_regime_input_rejects_source_not_available_at_declared_time() -> None:
    future_source = _ref(
        "source-a",
        "myquant.v17.v4.research-source.v1",
        "data/private/v17_v4_sources/source-a.json",
        cutoff=LATER,
    )
    with pytest.raises(IntelligenceContractError, match="not available at available_at"):
        _regime_input(source_ref=future_source)


def test_10_fusion_handles_missing_optional_branch() -> None:
    quant, _ = _branches(_evidence(), _evidence("CONTRARY", name="source-b"))
    result = fuse_research_branches(branches=[quant], as_of=AS_OF)
    assert result["normalized_weights"] == {"QUANT": "1.000000000000"}


def test_11_fusion_uses_availability_weighting() -> None:
    quant, fundamental = _branches(_evidence(), _evidence("CONTRARY", name="source-b"))
    result = fuse_research_branches(
        branches=[fundamental, quant],
        as_of=AS_OF,
    )
    assert Decimal(result["normalized_weights"]["QUANT"]) > Decimal(
        result["normalized_weights"]["FUNDAMENTAL"]
    )


def test_12_fusion_has_no_fixed_hidden_half_split() -> None:
    quant, fundamental = _branches(_evidence(), _evidence("CONTRARY", name="source-b"))
    result = fuse_research_branches(branches=[quant, fundamental], as_of=AS_OF)
    assert result["normalized_weights"]["QUANT"] != "0.500000000000"
    assert "trade_score" not in result
    assert "research_confidence_score" in result


def test_13_hypothesis_requires_supporting_and_contrary_evidence() -> None:
    positive = _evidence()
    contrary = _evidence("CONTRARY", name="source-b")
    assert _hypothesis(positive, contrary)["contrary_evidence_refs"]
    with pytest.raises(IntelligenceContractError, match="at least one evidence"):
        build_hypothesis(
            thesis="A",
            why_it_may_be_true="B",
            what_would_make_it_fail="C",
            supporting_evidence=[positive],
            contrary_evidence=[],
            expected_window_start="2026-01-03T07:00:00Z",
            expected_window_end="2026-01-04T07:00:00Z",
            falsification_conditions=[
                {
                    "metric_id": "rank_ic",
                    "operator": "LT",
                    "threshold": "0",
                    "window_sessions": 5,
                }
            ],
            related_companies=["000001.SZ"],
            related_industries=["banks"],
            as_of=AS_OF,
        )


def test_14_hypothesis_requires_machine_falsification() -> None:
    positive = _evidence()
    contrary = _evidence("CONTRARY", name="source-b")
    with pytest.raises(IntelligenceContractError, match="falsification condition"):
        build_hypothesis(
            thesis="A",
            why_it_may_be_true="B",
            what_would_make_it_fail="C",
            supporting_evidence=[positive],
            contrary_evidence=[contrary],
            expected_window_start="2026-01-03T07:00:00Z",
            expected_window_end="2026-01-04T07:00:00Z",
            falsification_conditions=[],
            related_companies=["000001.SZ"],
            related_industries=["banks"],
            as_of=AS_OF,
        )


def test_hypothesis_validator_replays_and_rejects_resealed_malformed_record() -> None:
    positive = _evidence()
    contrary = _evidence("CONTRARY", name="source-b")
    malformed = deepcopy(_hypothesis(positive, contrary))
    malformed["falsification_conditions"] = []
    with pytest.raises(IntelligenceContractError, match="falsification condition"):
        validate_hypothesis(
            _reseal(malformed, identity_field="hypothesis_id"),
            evidence=[positive, contrary],
            as_of=AS_OF,
        )


def test_15_memory_is_append_only_and_detects_deletion() -> None:
    hypothesis = _hypothesis(_evidence(), _evidence("CONTRARY", name="source-b"))
    original: tuple[dict[str, Any], ...] = ()
    appended = append_memory(
        original,
        event_type="HYPOTHESIS_CREATED",
        status="ACTIVE",
        subject_id=hypothesis["hypothesis_id"],
        summary="Created",
        artifact_refs=[content_ref(hypothesis, identity_field="hypothesis_id")],
        timestamp_value=AS_OF,
        expected_tip=ZERO_SHA256,
    )
    assert original == ()
    assert len(appended) == 1
    with pytest.raises(IntelligenceContractError, match="tail deletion"):
        validate_memory_chain((), expected_tip=memory_tip(appended))


def test_16_failed_hypothesis_is_retained() -> None:
    hypothesis = _hypothesis(_evidence(), _evidence("CONTRARY", name="source-b"))
    chain = append_memory(
        (),
        event_type="FAILED_CASE",
        status="FAILED",
        subject_id=hypothesis["hypothesis_id"],
        summary="Falsification condition was met.",
        artifact_refs=[content_ref(hypothesis, identity_field="hypothesis_id")],
        timestamp_value=AS_OF,
        expected_tip=ZERO_SHA256,
    )
    assert validate_memory_chain(chain, expected_tip=memory_tip(chain))[0]["status"] == "FAILED"


def _python_sources() -> list[Path]:
    package_root = Path(__file__).parents[2] / "quant_investor" / "intelligence"
    return sorted(package_root.rglob("*.py"))


def test_17_no_execution_or_decision_surface() -> None:
    forbidden = {"broker", "execution", "order", "portfolio", "selector", "trade"}
    assert all(value is False for key, value in NO_AUTHORITY.items() if key in forbidden)
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(token in node.name.casefold() for token in forbidden)
            for node in ast.walk(tree)
        )


def test_18_no_provider_or_model_invocation() -> None:
    forbidden_imports = {
        "anthropic",
        "openai",
        "tushare",
        "yfinance",
        "quant_investor.providers",
    }
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {str(node.module) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for name in imported
            for forbidden in forbidden_imports
        )


def test_19_v5_unchanged_dependency_boundary() -> None:
    package_root = Path(__file__).parents[2] / "quant_investor"
    intelligence_imports: set[str] = set()
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        intelligence_imports.update(
            str(node.module) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        intelligence_imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
    assert not any("v17_v5" in name for name in intelligence_imports)
    for path in package_root.rglob("*.py"):
        if "intelligence" in path.parts:
            continue
        assert "quant_investor.intelligence" not in path.read_text(
            encoding="utf-8", errors="ignore"
        )


def _artifact_ref(
    document: dict[str, Any], *, identity_field: str, relative_path: str
) -> dict[str, str]:
    return _ref(
        str(document[identity_field]),
        str(document["version"]),
        relative_path,
        byte_sha256=hashlib.sha256(canonical_resource_bytes(document)).hexdigest(),
        semantic_sha256=str(document["semantic_sha256"]),
        cutoff=str(document["cutoff"]),
    )


def _materialize(root: Path, relative_path: str, document: dict[str, Any]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_resource_bytes(document))


def _closure_artifact(
    *,
    artifact_id: str,
    version: str,
    relative_path: str,
    identity_field: str,
    cutoff: str = AS_OF,
) -> tuple[dict[str, str], tuple[str, dict[str, Any]]]:
    document = seal_semantic(
        {
            "cutoff": cutoff,
            identity_field: artifact_id,
            "strategy_id": STRATEGY,
            "version": version,
        }
    )
    reference = _artifact_ref(
        document,
        identity_field=identity_field,
        relative_path=relative_path,
    )
    return reference, (relative_path, document)


def _forward_closure(
    root: Path,
) -> tuple[
    str,
    str,
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    closure_documents: list[tuple[str, dict[str, Any]]] = []
    factor_ref, factor_document = _closure_artifact(
        artifact_id="factor-a",
        version="myquant.v17.v4.factor-definition.v1",
        relative_path="data/private/v17_v4_sources/factors/factor-a.json",
        identity_field="factor_id",
    )
    closure_documents.append(factor_document)
    source_a_ref, source_a_document = _closure_artifact(
        artifact_id="source-a",
        version="myquant.v17.v4.research-source.v1",
        relative_path="data/private/v17_v4_sources/source-a.json",
        identity_field="source_id",
    )
    closure_documents.append(source_a_document)
    source_b_ref, source_b_document = _closure_artifact(
        artifact_id="source-b",
        version="myquant.v17.v4.research-source.v1",
        relative_path="data/private/v17_v4_sources/source-b.json",
        identity_field="source_id",
    )
    closure_documents.append(source_b_document)
    label_source_ref, label_source_document = _closure_artifact(
        artifact_id="label-source",
        version="myquant.v17.v4.label-market.v1",
        relative_path="data/private/v17_v4_sources/label-market.json",
        identity_field="source_id",
        cutoff=LATER,
    )
    closure_documents.append(label_source_document)
    request = build_forward_request(
        {
            "authority": dict(V4_AUTHORITY),
            "created_at": AS_OF,
            "cutoff": AS_OF,
            "decision_session": "2026-01-02",
            "factor_refs": [factor_ref],
            "protocol_version": "myquant.v17.v4",
            "request_profile": "EXPLORE",
            "source_refs": [source_a_ref, source_b_ref],
            "strategy_id": STRATEGY,
        }
    )
    request_path = "data/private/v17_v4_runs/forward_requests/request-a.json"
    request_ref = _artifact_ref(
        request,
        identity_field="request_id",
        relative_path=request_path,
    )
    closure_documents.append((request_path, request))
    payload_json = canonical_bytes({}).decode("utf-8")
    stage_output = seal_semantic(
        {
            "authority": dict(V4_AUTHORITY),
            "completeness": "COMPLETE",
            "cutoff": AS_OF,
            "decision_session": "2026-01-02",
            "lineage_receipt_refs": [],
            "output_id": "stage-output-a",
            "payload_json": payload_json,
            "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            "protocol_version": "myquant.v17.v4",
            "recorded_at": AS_OF,
            "request_ref": request_ref,
            "stage_id": "final",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.forward-stage-output.v1",
        }
    )
    stage_output_path = (
        "results/v17_v4_shadow/forward_evidence/strategies/"
        "cn-forward-research/runs/request-a/outputs/final.json"
    )
    stage_output_ref = _artifact_ref(
        stage_output,
        identity_field="output_id",
        relative_path=stage_output_path,
    )
    closure_documents.append((stage_output_path, stage_output))
    stage_receipt = seal_semantic(
        {
            "authority": dict(V4_AUTHORITY),
            "blockers": [],
            "completeness": "COMPLETE",
            "cutoff": AS_OF,
            "decision_session": "2026-01-02",
            "execution_outcome": "SUCCEEDED",
            "output_refs": [stage_output_ref],
            "protocol_version": "myquant.v17.v4",
            "receipt_id": "stage-receipt-a",
            "recorded_at": AS_OF,
            "request_ref": request_ref,
            "stage_id": "final",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.forward-stage-receipt.v1",
        }
    )
    stage_receipt_path = (
        "results/v17_v4_shadow/forward_evidence/strategies/"
        "cn-forward-research/runs/request-a/receipts/final.json"
    )
    stage_receipt_ref = _artifact_ref(
        stage_receipt,
        identity_field="receipt_id",
        relative_path=stage_receipt_path,
    )
    closure_documents.append((stage_receipt_path, stage_receipt))
    run_path = (
        "results/v17_v4_shadow/forward_evidence/strategies/cn-forward-research/"
        "runs/request-a/run.json"
    )
    run = seal_semantic(
        {
            "authority": dict(V4_AUTHORITY),
            "broker": False,
            "completeness": "COMPLETE",
            "cutoff": AS_OF,
            "decision_session": "2026-01-02",
            "execution": False,
            "execution_outcome": "SUCCEEDED",
            "formal_activation_eligible": False,
            "global_activation_state": "INACTIVE",
            "observation_refs": [stage_output_ref],
            "observation_run_id": "observation-run-a",
            "order": False,
            "protocol_version": "myquant.v17.v4",
            "recorded_at": AS_OF,
            "research_runtime_default": False,
            "request_ref": request_ref,
            "run_state": "EXPLORE_COMPLETE",
            "stage_receipt_refs": [stage_receipt_ref],
            "strategy_id": STRATEGY,
            "trade": False,
            "version": "myquant.v17.v4.forward-observation-run.v1",
        }
    )
    run_ref = _artifact_ref(run, identity_field="observation_run_id", relative_path=run_path)
    session_path = (
        "results/v17_v4_shadow/forward_evidence/strategies/cn-forward-research/"
        "sessions/2026-01-02/request-a.json"
    )
    session = seal_semantic(
        {
            "authority": dict(V4_AUTHORITY),
            "broker": False,
            "cutoff": AS_OF,
            "decision_session": "2026-01-02",
            "execution": False,
            "formal_activation_eligible": False,
            "global_activation_state": "INACTIVE",
            "observation_run_ref": run_ref,
            "order": False,
            "protocol_version": "myquant.v17.v4",
            "published_at": AS_OF,
            "research_runtime_default": False,
            "run_state": "EXPLORE_COMPLETE",
            "session_ref_id": "forward-observation-session-a",
            "strategy_id": STRATEGY,
            "trade": False,
            "version": "myquant.v17.v4.forward-observation-session-ref.v1",
        }
    )
    observation = seal_semantic(
        {
            "authority": dict(V4_AUTHORITY),
            "canary_evidence_eligible": False,
            "completeness": "COMPLETE",
            "cutoff": AS_OF,
            "decision_session": "2026-01-02",
            "factor_ref": factor_ref,
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "observation_id": "factor-observation-a",
            "observations": [{"status": "AVAILABLE", "symbol": "000001.SZ", "value": "0.8"}],
            "production_default_eligible": False,
            "promotion_eligible": False,
            "protocol_version": "myquant.v17.v4",
            "provider_authority": False,
            "provider_invoked": False,
            "request_ref": request_ref,
            "shadow_only": True,
            "source_refs": [source_a_ref, source_b_ref],
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.factor-universe-observation.v1",
        }
    )
    observation_path = "results/v17_v4_shadow/forward_observations/factor-a.json"
    observation_ref = _artifact_ref(
        observation,
        identity_field="observation_id",
        relative_path=observation_path,
    )
    label = seal_semantic(
        {
            "authority": dict(V4_AUTHORITY),
            "canary_evidence_eligible": False,
            "completeness": "COMPLETE",
            "cost_basis_points": 20,
            "cutoff": LATER,
            "decision_session": "2026-01-02",
            "evidence_refs": [label_source_ref],
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "horizon_sessions": 1,
            "label_id": "forward-label-a",
            "label_rows": [
                {
                    "cost_adjusted_return": "0.08",
                    "industry_adjusted_return": "0.02",
                    "industry_id": "banks",
                    "industry_return": "0.08",
                    "market_adjusted_return": "0.03",
                    "market_return": "0.07",
                    "status": "AVAILABLE",
                    "symbol": "000001.SZ",
                    "total_return": "0.1",
                }
            ],
            "label_session": "2026-01-05",
            "observation_run_ref": run_ref,
            "performance_evidence_eligible": False,
            "production_default_eligible": False,
            "promotion_eligible": False,
            "protocol_version": "myquant.v17.v4",
            "provider_authority": False,
            "provider_invoked": False,
            "shadow_only": True,
            "shanghai_open_sessions": ["2026-01-02", "2026-01-05"],
            "source_lineage_sha256": SHA_C,
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.forward-label.v1",
        }
    )
    label_path = "results/v17_v4_shadow/forward_labels/label-a.json"
    label_ref = _artifact_ref(label, identity_field="label_id", relative_path=label_path)
    lineage = {
        "factor_definition_sha256": SHA_B,
        "factor_name": "factor-a",
        "factor_set_sha256": SHA_C,
        "horizon_sessions": 1,
        "quant_policy_sha256": SHA_A,
        "source_lineage_sha256": SHA_C,
    }
    origin_inventory = build_forward_evidence_origin_inventory(
        inventory_id="origin-inventory-a",
        strategy_id=STRATEGY,
        decision_session="2026-01-02",
        cutoff=LATER,
        request_ref=request_ref,
        origins=[
            {
                "evidence_ref": label_ref,
                "lineage_key": lineage,
                "origin": "2026-01-02",
            }
        ],
    )
    origin_inventory_path = "results/v17_v4_shadow/forward_evaluations/origin-a.json"
    origin_inventory_ref = _artifact_ref(
        origin_inventory,
        identity_field="inventory_id",
        relative_path=origin_inventory_path,
    )
    closure_documents.append((origin_inventory_path, origin_inventory))
    factor_inventory = build_existing_factor_inventory(
        inventory_id="factor-inventory-a",
        strategy_id=STRATEGY,
        decision_session="2026-01-02",
        cutoff=LATER,
        request_ref=request_ref,
        source_refs=[source_a_ref, source_b_ref],
        factors=[
            {
                "definition_sha256": SHA_B,
                "exposure_observation_refs": [observation_ref],
                "factor_name": "factor-a",
                "factor_ref": factor_ref,
                "lifecycle": "ACTIVE",
            }
        ],
    )
    factor_inventory_path = "results/v17_v4_shadow/forward_evaluations/inventory-a.json"
    factor_inventory_ref = _artifact_ref(
        factor_inventory,
        identity_field="inventory_id",
        relative_path=factor_inventory_path,
    )
    closure_documents.append((factor_inventory_path, factor_inventory))
    evaluation = seal_semantic(
        {
            "authority": dict(V4_AUTHORITY),
            "blockers": [],
            "canary_evidence_eligible": False,
            "completeness": "COMPLETE",
            "cutoff": LATER,
            "decision_session": "2026-01-02",
            "evidence_origin_inventory_ref": origin_inventory_ref,
            "execution_outcome": "SUCCEEDED",
            "existing_factor_inventory_ref": factor_inventory_ref,
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "label_refs": [label_ref],
            "lineage_key": lineage,
            "lineage_key_sha256": hashlib.sha256(canonical_bytes(lineage)).hexdigest(),
            "metric_rows": [{"metric_id": "rank_ic", "status": "AVAILABLE", "value": "0.1"}],
            "observation_run_ref": run_ref,
            "origin_count": 1,
            "production_default_eligible": False,
            "promotion_eligible": False,
            "protocol_version": "myquant.v17.v4",
            "provider_authority": False,
            "provider_invoked": False,
            "receipt_id": "evaluation-receipt-a",
            "receipt_type": "factor_evaluation_receipt",
            "recorded_at": LATER,
            "shadow_only": True,
            "strategy_id": STRATEGY,
            "subject_id": "factor-a",
            "version": "myquant.v17.v4.forward-evaluation-receipt.v1",
        }
    )
    evaluation_path = "results/v17_v4_shadow/forward_evaluations/evaluation-a.json"
    evaluation_ref = _artifact_ref(
        evaluation,
        identity_field="receipt_id",
        relative_path=evaluation_path,
    )
    for path, document in (
        *closure_documents,
        (run_path, run),
        (session_path, session),
        (observation_path, observation),
        (label_path, label),
        (evaluation_path, evaluation),
    ):
        _materialize(root, path, document)
    session_sha = hashlib.sha256(canonical_resource_bytes(session)).hexdigest()
    closure_refs = [
        request_ref,
        stage_receipt_ref,
        stage_output_ref,
        factor_ref,
        source_a_ref,
        source_b_ref,
        label_source_ref,
        origin_inventory_ref,
        factor_inventory_ref,
    ]
    return (
        session_path,
        session_sha,
        [observation_ref],
        [label_ref],
        [evaluation_ref],
        closure_refs,
    )


def test_forward_adapter_binds_observation_label_and_evaluation(tmp_path: Path) -> None:
    session_path, session_sha, observations, labels, evaluations, closure_refs = _forward_closure(
        tmp_path
    )
    bundle = build_observation_evidence_bundle(
        workspace_root=str(tmp_path),
        session_relative_path=session_path,
        session_byte_sha256=session_sha,
        observation_refs=observations,
        closure_refs=closure_refs,
        label_refs=labels,
        evaluation_refs=evaluations,
        as_of=LATER,
    )
    assert bundle["version"] == BUNDLE_VERSION
    assert bundle["completeness"] == "COMPLETE"
    assert bundle["observation_refs"] == observations
    assert bundle["label_refs"] == labels
    assert bundle["evaluation_refs"] == evaluations


def test_forward_adapter_blocks_supplied_invalid_optional_ref(tmp_path: Path) -> None:
    session_path, session_sha, observations, labels, _, closure_refs = _forward_closure(tmp_path)
    labels[0]["byte_sha256"] = SHA_B
    with pytest.raises(IntelligenceContractError, match="byte SHA mismatch"):
        build_observation_evidence_bundle(
            workspace_root=str(tmp_path),
            session_relative_path=session_path,
            session_byte_sha256=session_sha,
            observation_refs=observations,
            closure_refs=closure_refs,
            label_refs=labels,
            as_of=LATER,
        )


@pytest.mark.parametrize("field", ["artifact_id", "semantic_sha256"])
def test_forward_adapter_rejects_conflicting_preverified_label_ref(
    tmp_path: Path,
    field: str,
) -> None:
    session_path, session_sha, observations, labels, evaluations, closure_refs = _forward_closure(
        tmp_path
    )
    evaluation_path = tmp_path / "results/v17_v4_shadow/forward_evaluations/evaluation-a.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["label_refs"][0][field] = "forged-label-id" if field == "artifact_id" else SHA_A
    evaluation.pop("semantic_sha256")
    evaluation = seal_semantic(evaluation)
    _materialize(
        tmp_path,
        "results/v17_v4_shadow/forward_evaluations/evaluation-a.json",
        evaluation,
    )
    evaluations = [
        _artifact_ref(
            evaluation,
            identity_field="receipt_id",
            relative_path="results/v17_v4_shadow/forward_evaluations/evaluation-a.json",
        )
    ]
    with pytest.raises(IntelligenceContractError, match="supplied exactly"):
        build_observation_evidence_bundle(
            workspace_root=str(tmp_path),
            session_relative_path=session_path,
            session_byte_sha256=session_sha,
            observation_refs=observations,
            closure_refs=closure_refs,
            label_refs=labels,
            evaluation_refs=evaluations,
            as_of=LATER,
        )


def test_forward_adapter_rejects_conflicting_recursive_exact_ref(tmp_path: Path) -> None:
    session_path, session_sha, observations, _, _, closure_refs = _forward_closure(tmp_path)
    observation_path = tmp_path / "results/v17_v4_shadow/forward_observations/factor-a.json"
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["factor_ref"]["artifact_id"] = "forged-factor-id"
    observation.pop("semantic_sha256")
    observation = seal_semantic(observation)
    _materialize(
        tmp_path,
        "results/v17_v4_shadow/forward_observations/factor-a.json",
        observation,
    )
    observations = [
        _artifact_ref(
            observation,
            identity_field="observation_id",
            relative_path="results/v17_v4_shadow/forward_observations/factor-a.json",
        )
    ]
    with pytest.raises(IntelligenceContractError, match="recursive exact ref conflict"):
        build_observation_evidence_bundle(
            workspace_root=str(tmp_path),
            session_relative_path=session_path,
            session_byte_sha256=session_sha,
            observation_refs=observations,
            closure_refs=closure_refs[:6],
            as_of=AS_OF,
        )


def test_forward_adapter_rejects_dangling_recursive_closure(tmp_path: Path) -> None:
    session_path, session_sha, observations, _, _, closure_refs = _forward_closure(tmp_path)
    without_request = [
        ref
        for ref in closure_refs
        if ref["artifact_version"] != "myquant.v17.v4.forward-run-request.v1"
    ]
    with pytest.raises(IntelligenceContractError, match="undeclared or conflicting"):
        build_observation_evidence_bundle(
            workspace_root=str(tmp_path),
            session_relative_path=session_path,
            session_byte_sha256=session_sha,
            observation_refs=observations,
            closure_refs=without_request,
            as_of=AS_OF,
        )


def test_forward_adapter_rejects_hardlinked_closure_artifact(tmp_path: Path) -> None:
    session_path, session_sha, observations, _, _, closure_refs = _forward_closure(tmp_path)
    source_path = tmp_path / "data/private/v17_v4_sources/source-a.json"
    os.link(source_path, source_path.with_name("source-a-hardlink.json"))
    with pytest.raises(IntelligenceContractError, match="hard-linked"):
        build_observation_evidence_bundle(
            workspace_root=str(tmp_path),
            session_relative_path=session_path,
            session_byte_sha256=session_sha,
            observation_refs=observations,
            closure_refs=closure_refs,
            as_of=AS_OF,
        )


def test_observation_bundle_validator_rejects_resealed_authorization_drift(
    tmp_path: Path,
) -> None:
    session_path, session_sha, observations, _, _, closure_refs = _forward_closure(tmp_path)
    closure_refs = closure_refs[:6]
    bundle = build_observation_evidence_bundle(
        workspace_root=str(tmp_path),
        session_relative_path=session_path,
        session_byte_sha256=session_sha,
        observation_refs=observations,
        closure_refs=closure_refs,
        as_of=AS_OF,
    )
    malformed = deepcopy(bundle)
    malformed["authorized_evidence_refs"] = malformed["authorized_evidence_refs"][1:]
    with pytest.raises(IntelligenceContractError, match="authorization closure mismatch"):
        validate_observation_evidence_bundle(
            _reseal(malformed, identity_field="bundle_id"),
            as_of=AS_OF,
        )


def test_ai_draft_is_source_bound_and_cannot_modify_control_fields() -> None:
    draft = build_ai_draft(
        kind="SUMMARY",
        payload={"summary": "Industry evidence summary."},
        source_refs=[_source_ref()],
        generated_at=AS_OF,
        confidence="0.7",
    )
    assert draft["source_refs"] and draft["confidence"] == "0.700000000000"
    with pytest.raises(IntelligenceContractError, match="forbidden"):
        build_ai_draft(
            kind="HYPOTHESIS_DRAFT",
            payload={"nested": {"posterior_value": "0.9"}},
            source_refs=[_source_ref()],
            generated_at=AS_OF,
            confidence="0.7",
        )


def test_runtime_and_package_verify(tmp_path: Path) -> None:
    session_path, session_sha, observations, _, _, closure_refs = _forward_closure(tmp_path)
    closure_refs = closure_refs[:6]
    bundle = build_observation_evidence_bundle(
        workspace_root=str(tmp_path),
        session_relative_path=session_path,
        session_byte_sha256=session_sha,
        observation_refs=observations,
        closure_refs=closure_refs,
        as_of=AS_OF,
    )
    source_a_ref = next(ref for ref in closure_refs if ref["artifact_id"] == "source-a")
    source_b_ref = next(ref for ref in closure_refs if ref["artifact_id"] == "source-b")
    positive = _evidence(source_ref=source_a_ref)
    contrary = _evidence(
        "CONTRARY",
        name="source-b",
        source_ref=source_b_ref,
    )
    hypothesis = _hypothesis(positive, contrary)
    bayesian = update_hypothesis(
        hypothesis_id=hypothesis["hypothesis_id"],
        prior="0.5",
        evidence=[positive, contrary],
        as_of=AS_OF,
    )
    regime_input = _regime_input(source_ref=source_a_ref)
    regime = infer_multilayer_regime(
        regime_input=regime_input,
        evidence=[positive, contrary],
        as_of=AS_OF,
    )
    quant, fundamental = _branches(positive, contrary)
    fusion = fuse_research_branches(branches=[quant, fundamental], as_of=AS_OF)
    memory = append_memory(
        (),
        event_type="HYPOTHESIS_CREATED",
        status="ACTIVE",
        subject_id=hypothesis["hypothesis_id"],
        summary="Created",
        artifact_refs=[content_ref(hypothesis, identity_field="hypothesis_id")],
        timestamp_value=AS_OF,
        expected_tip=ZERO_SHA256,
    )
    receipt = build_intelligence_runtime_receipt(
        observation_bundle=bundle,
        workspace_root=str(tmp_path),
        session_relative_path=session_path,
        session_byte_sha256=session_sha,
        observation_refs=observations,
        closure_refs=closure_refs,
        evidence=[positive, contrary],
        bayesian_receipts=[bayesian],
        regime_input=regime_input,
        regime_receipt=regime,
        branches=[quant, fundamental],
        fusion_receipt=fusion,
        hypotheses=[hypothesis],
        memory_entries=memory,
        expected_memory_tip=memory_tip(memory),
        as_of=AS_OF,
    )
    assert verify_runtime_receipt(receipt) == receipt
    assert verify_package()["version"].endswith("package-manifest.v1")


def test_runtime_rejects_evidence_outside_observation_authorization(tmp_path: Path) -> None:
    session_path, session_sha, observations, _, _, closure_refs = _forward_closure(tmp_path)
    closure_refs = closure_refs[:6]
    bundle = build_observation_evidence_bundle(
        workspace_root=str(tmp_path),
        session_relative_path=session_path,
        session_byte_sha256=session_sha,
        observation_refs=observations,
        closure_refs=closure_refs,
        as_of=AS_OF,
    )
    source_a_ref = next(ref for ref in closure_refs if ref["artifact_id"] == "source-a")
    foreign_ref = _source_ref("foreign-source")
    foreign_positive = _evidence(source_ref=foreign_ref)
    forged_bundle = deepcopy(bundle)
    forged_bundle["verified_closure_refs"] = sorted(
        [*forged_bundle["verified_closure_refs"], foreign_ref],
        key=lambda ref: (ref["relative_path"].encode(), ref["byte_sha256"].encode()),
    )
    forged_bundle["authorized_evidence_refs"] = sorted(
        [*forged_bundle["authorized_evidence_refs"], foreign_ref],
        key=lambda ref: (ref["relative_path"].encode(), ref["byte_sha256"].encode()),
    )
    forged_bundle = _reseal(forged_bundle, identity_field="bundle_id")
    contrary = _evidence(
        "CONTRARY",
        name="source-b",
        source_ref=next(ref for ref in closure_refs if ref["artifact_id"] == "source-b"),
    )
    hypothesis = _hypothesis(foreign_positive, contrary)
    bayesian = update_hypothesis(
        hypothesis_id=hypothesis["hypothesis_id"],
        prior="0.5",
        evidence=[foreign_positive, contrary],
        as_of=AS_OF,
    )
    regime_input = _regime_input(source_ref=source_a_ref)
    regime = infer_multilayer_regime(
        regime_input=regime_input,
        evidence=[foreign_positive, contrary],
        as_of=AS_OF,
    )
    quant, fundamental = _branches(foreign_positive, contrary)
    fusion = fuse_research_branches(branches=[quant, fundamental], as_of=AS_OF)
    memory = append_memory(
        (),
        event_type="HYPOTHESIS_CREATED",
        status="ACTIVE",
        subject_id=hypothesis["hypothesis_id"],
        summary="Created",
        artifact_refs=[content_ref(hypothesis, identity_field="hypothesis_id")],
        timestamp_value=AS_OF,
        expected_tip=ZERO_SHA256,
    )
    with pytest.raises(IntelligenceContractError, match="exact adapter replay"):
        build_intelligence_runtime_receipt(
            observation_bundle=forged_bundle,
            workspace_root=str(tmp_path),
            session_relative_path=session_path,
            session_byte_sha256=session_sha,
            observation_refs=observations,
            closure_refs=closure_refs,
            evidence=[foreign_positive, contrary],
            bayesian_receipts=[bayesian],
            regime_input=regime_input,
            regime_receipt=regime,
            branches=[quant, fundamental],
            fusion_receipt=fusion,
            hypotheses=[hypothesis],
            memory_entries=memory,
            expected_memory_tip=memory_tip(memory),
            as_of=AS_OF,
        )


def _runtime_summary(*, timestamp_value: str = AS_OF) -> dict[str, Any]:
    versions = {
        "bayesian": "myquant.v17.research-intelligence.bayesian-evidence-receipt.v1",
        "branches": "myquant.v17.research-intelligence.branch-output.v1",
        "evidence": "myquant.v17.research-intelligence.evidence.v1",
        "fusion": ("myquant.v17.research-intelligence." "availability-aware-fusion-receipt.v1"),
        "hypotheses": "myquant.v17.research-intelligence.hypothesis.v1",
        "observation_bundle": ("myquant.v17.research-intelligence.observation-evidence-bundle.v1"),
        "regime": "myquant.v17.research-intelligence.multilayer-regime-receipt.v1",
        "regime_input": "myquant.v17.research-intelligence.regime-input.v1",
    }

    def ref(name: str) -> dict[str, str]:
        return {
            "artifact_id": f"{name}-a",
            "artifact_version": versions[name],
            "byte_sha256": SHA_A,
            "semantic_sha256": SHA_B,
        }

    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "component_refs": {
                "bayesian": [ref("bayesian")],
                "branches": [ref("branches")],
                "evidence": [ref("evidence")],
                "fusion": ref("fusion"),
                "hypotheses": [ref("hypotheses")],
                "observation_bundle": ref("observation_bundle"),
                "regime": ref("regime"),
                "regime_input": ref("regime_input"),
            },
            "memory_entry_count": 1,
            "memory_tip_sha256": SHA_C,
            "production": False,
            "research_only": True,
            "timestamp": timestamp_value,
            "version": INTELLIGENCE_RUNTIME_RECEIPT_VERSION,
        },
        identity_field="runtime_receipt_id",
    )


def test_runtime_summary_verifier_rejects_noncanonical_time_and_wrong_topology() -> None:
    with pytest.raises(IntelligenceContractError, match="UTC second timestamp"):
        verify_runtime_receipt(_runtime_summary(timestamp_value="not-a-time"))
    wrong_version = _runtime_summary()
    wrong_version["component_refs"]["fusion"]["artifact_version"] = "wrong.v1"
    wrong_version = _reseal(wrong_version, identity_field="runtime_receipt_id")
    with pytest.raises(IntelligenceContractError, match="version mismatch"):
        verify_runtime_receipt(wrong_version)
    conflicting_identity = _runtime_summary()
    conflict = deepcopy(conflicting_identity["component_refs"]["branches"][0])
    conflict["byte_sha256"] = SHA_B
    conflict["semantic_sha256"] = SHA_C
    conflicting_identity["component_refs"]["branches"].append(conflict)
    conflicting_identity = _reseal(
        conflicting_identity,
        identity_field="runtime_receipt_id",
    )
    with pytest.raises(IntelligenceContractError, match="duplicates"):
        verify_runtime_receipt(conflicting_identity)
