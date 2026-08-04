from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes, seal_semantic
from quant_investor.v17_v5_contract.validators import (
    V4_COMPATIBILITY_POLICY_BYTE_SHA256,
    V4_PACKAGE_MANIFEST_SHA256,
    V4_RUNTIME_MANIFEST_SHA256,
    V4_SOURCE_GIT_COMMIT,
)
from quant_investor.v17_v5_runtime.v4_compat_reader import (
    V4ClosureNode,
    V4CompatibilityRead,
    read_v4_artifact,
)
from quant_investor.v17_v5_runtime.v4_regime_adapter import (
    REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE,
    REGIME_EVIDENCE_V1_VERSION,
    REGIME_EVIDENCE_V2_VERSION,
    REGIME_EVIDENCE_V2_NON_DEPLOYABLE,
    REGIME_EVIDENCE_V3_NOT_FINALIZED,
    REGIME_CONTINUITY_GENESIS,
    REGIME_CONTINUITY_RECOVERY,
    REGIME_HARD_STATE_UNAVAILABLE,
    REGIME_HARD_STATE_UNKNOWN,
    V4RegimeAdapterError,
    V4RegimeEvidenceStatus,
    adapt_v4_regime_evidence,
)

CUTOFF = "2026-07-29T08:00:00Z"
RELATIVE_PATH = "data/private/v17_v4_runs/run-1/regime.json"
STRATEGY = "quant-first"

_V3_TEST_PATH = Path(__file__).with_name("test_v17_v4_regime_evidence_v3.py")
_V3_SPEC = importlib.util.spec_from_file_location("_v4_v3_regime_test_support", _V3_TEST_PATH)
assert _V3_SPEC is not None and _V3_SPEC.loader is not None
_V3_SUPPORT = importlib.util.module_from_spec(_V3_SPEC)
sys.modules[_V3_SPEC.name] = _V3_SUPPORT
_V3_SPEC.loader.exec_module(_V3_SUPPORT)


def _artifact(**overrides: Any) -> dict[str, Any]:
    payload = {
        "authority": {
            "broker": False,
            "execution": False,
            "formal_research_publication": False,
            "order": False,
            "research_runtime_default": False,
            "trade": False,
        },
        "available_at": "2026-07-29T07:59:00Z",
        "created_at": "2026-07-29T08:00:01Z",
        "cutoff": CUTOFF,
        "evidence_id": "regime-evidence-1",
        "gross_multiplier": "0.8",
        "protocol_version": "myquant.v17.v4",
        "role": "markov_evidence",
        "run_id": "run-1",
        "status": "AVAILABLE",
        "strategy_id": STRATEGY,
        "version": "myquant.v17.v4.regime-evidence.v1",
    }
    payload.update(overrides)
    payload.pop("semantic_sha256", None)
    return seal_semantic(payload)


def _write_artifact(root: Path, artifact: dict[str, Any] | None = None) -> str:
    path = root / RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_resource_bytes(_artifact() if artifact is None else artifact)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, sha256: str) -> V4CompatibilityRead:
    return read_v4_artifact(
        root,
        relative_path=RELATIVE_PATH,
        expected_byte_sha256=sha256,
        expected_strategy_id=STRATEGY,
        decision_cutoff=CUTOFF,
    )


def _read_v3(root: Path, result: Any) -> V4CompatibilityRead:
    return read_v4_artifact(
        root,
        relative_path=result.evidence_path,
        expected_byte_sha256=result.evidence_sha256,
        expected_strategy_id=str(result.document["strategy_id"]),
        decision_cutoff=str(result.document["cutoff"]),
    )


def _with_read(
    read: V4CompatibilityRead,
    *,
    document: dict[str, Any] | None = None,
    documents: dict[str, dict[str, Any]] | None = None,
    root_ref: dict[str, str] | None = None,
    closure: tuple[V4ClosureNode, ...] | None = None,
    policy_sha256: str | None = None,
) -> V4CompatibilityRead:
    return V4CompatibilityRead(
        closure=read.closure if closure is None else closure,
        compatibility_policy_byte_sha256=(
            read.compatibility_policy_byte_sha256 if policy_sha256 is None else policy_sha256
        ),
        document=read.document if document is None else document,
        documents=read.documents if documents is None else documents,
        predecessor_git_commit=read.predecessor_git_commit,
        predecessor_package_manifest_byte_sha256=(read.predecessor_package_manifest_byte_sha256),
        predecessor_package_manifest_relative_path=(
            read.predecessor_package_manifest_relative_path
        ),
        predecessor_protocol_version=read.predecessor_protocol_version,
        predecessor_runtime_manifest_byte_sha256=(read.predecessor_runtime_manifest_byte_sha256),
        predecessor_runtime_manifest_relative_path=(
            read.predecessor_runtime_manifest_relative_path
        ),
        root_ref=read.root_ref if root_ref is None else root_ref,
        terminal_bindings=read.terminal_bindings,
    )


def _node_from_ref(ref: dict[str, str]) -> V4ClosureNode:
    return V4ClosureNode(
        artifact_id=ref["artifact_id"],
        byte_sha256=ref["byte_sha256"],
        relative_path=ref["relative_path"],
        semantic_sha256=ref["semantic_sha256"],
        validation_mode="V4_REGISTERED_JSON",
        version=ref["artifact_version"],
    )


def _ref(
    artifact_id: str,
    version: str,
    path: str,
    *,
    cutoff: str = CUTOFF,
    strategy_id: str = STRATEGY,
) -> dict[str, str]:
    material = f"{artifact_id}|{version}|{path}|{cutoff}|{strategy_id}".encode()
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(material + b"|byte").hexdigest(),
        "cutoff": cutoff,
        "relative_path": path,
        "semantic_sha256": hashlib.sha256(material + b"|semantic").hexdigest(),
        "strategy_id": strategy_id,
    }


def _manual_read(
    document: dict[str, Any],
    refs: list[dict[str, str]] | None = None,
    documents: dict[str, dict[str, Any]] | None = None,
) -> V4CompatibilityRead:
    root_path = "data/private/v17_v4_sources/regime_evidence/2026-07-30/regime_evidence.v2.json"
    if document["version"] == REGIME_EVIDENCE_V1_VERSION:
        root_path = RELATIVE_PATH
    root_ref = {
        "artifact_id": str(document["evidence_id"]),
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(document)).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": root_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }
    root_node = _node_from_ref(root_ref)
    closure = [root_node, *[_node_from_ref(ref) for ref in (refs or [])]]
    return V4CompatibilityRead(
        closure=tuple(sorted(closure, key=lambda node: node.relative_path)),
        compatibility_policy_byte_sha256=V4_COMPATIBILITY_POLICY_BYTE_SHA256,
        document=document,
        documents={root_path: document, **(documents or {})},
        predecessor_git_commit=V4_SOURCE_GIT_COMMIT,
        predecessor_package_manifest_byte_sha256=V4_PACKAGE_MANIFEST_SHA256,
        predecessor_package_manifest_relative_path=(
            "quant_investor/v17_v4_contract/resources/package_manifest.v1.json"
        ),
        predecessor_protocol_version="myquant.v17.v4",
        predecessor_runtime_manifest_byte_sha256=V4_RUNTIME_MANIFEST_SHA256,
        predecessor_runtime_manifest_relative_path=(
            "quant_investor/v17_v4_contract/resources/runtime_build_manifest.v1.json"
        ),
        root_ref=root_ref,
        terminal_bindings=(),
    )


def _sealed_ref(
    document: dict[str, Any],
    *,
    artifact_id: str,
    path: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    sealed = seal_semantic(document)
    return sealed, {
        "artifact_id": artifact_id,
        "artifact_version": str(sealed["version"]),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(sealed)).hexdigest(),
        "cutoff": str(sealed["cutoff"]),
        "relative_path": path,
        "semantic_sha256": str(sealed["semantic_sha256"]),
        "strategy_id": str(sealed["strategy_id"]),
    }


def _v2_document(
    **overrides: Any,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, dict[str, Any]]]:
    calendar_sessions = overrides.pop(
        "_calendar_sessions",
        ["2026-07-29", "2026-07-30"],
    )
    calendar_path = "data/private/v17_v4_sources/regime_evidence/2026-07-30/calendar.json"
    calendar_document, calendar = _sealed_ref(
        {
            "calendar_terminal_id": "calendar-terminal",
            "cutoff": "2026-07-29T07:05:00Z",
            "open_sessions": calendar_sessions,
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.regime-calendar-terminal.v1",
        },
        artifact_id="calendar-terminal",
        path=calendar_path,
    )
    feature_path = "data/private/v17_v4_sources/regime_evidence/2026-07-30/feature.json"
    feature_document, feature = _sealed_ref(
        {
            "calendar_ref": calendar,
            "cutoff": "2026-07-29T07:05:00Z",
            "effective_session": "2026-07-30",
            "feature_snapshot_id": "feature-snapshot",
            "observed_through_session": "2026-07-29",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.regime-feature-snapshot.v1",
        },
        artifact_id="feature-snapshot",
        path=feature_path,
    )
    model = _ref(
        "model-snapshot",
        "myquant.v17.v4.regime-model-snapshot.v1",
        "data/private/v17_v4_sources/regime_evidence/2026-07-30/model.json",
    )
    transition = _ref(
        "transition-snapshot",
        "myquant.v17.v4.regime-transition-matrix-snapshot.v1",
        "data/private/v17_v4_sources/regime_evidence/2026-07-30/transition.json",
    )
    scope = _ref(
        "source-locator",
        "myquant.v17.v4.regime-source-locator-terminal.v1",
        "data/private/v17_v4_sources/regime_evidence/2026-07-30/source-locator.json",
    )
    market = _ref(
        "market-terminal",
        "myquant.v17.v4.regime-market-terminal.v1",
        "data/private/v17_v4_sources/regime_evidence/2026-07-30/market.json",
    )
    payload = {
        "authority": {
            "broker": False,
            "execution": False,
            "formal_research_publication": False,
            "order": False,
            "research_runtime_default": False,
            "trade": False,
        },
        "available_at": "2026-07-30T07:45:00Z",
        "blocker_codes": [],
        "computed_at": "2026-07-30T07:44:00Z",
        "coverage_ratio": "1.000000000000",
        "created_at": "2026-07-30T07:45:00Z",
        "cutoff": "2026-07-30T07:45:00Z",
        "decision_session": "2026-07-30",
        "effective_session": "2026-07-30",
        "evidence_id": "a" * 64,
        "feature_cutoff": "2026-07-29T07:05:00Z",
        "feature_snapshot_ref": feature,
        "formal_activation_eligible": False,
        "hard_state": "趋势上涨",
        "hard_state_derivation": "SEALED_ARGMAX_POLICY_V1",
        "inference_kind": "FILTERED_CAUSAL",
        "inference_policy_ref": {
            "byte_sha256": "006773e24f47f0b7f28d6f7707ff6f570066cb212bd83ebd9566512fda7734ef",
            "relative_path": "resources/regime_inference_policy.v1.json",
            "semantic_sha256": ("8abff276d5ed217ad2cb411e26e658ac87877e6f7b03682f502d960a8487c913"),
            "version": "myquant.v17.v4.regime-inference-policy.v1",
        },
        "lineage_phase": "BOOTSTRAP",
        "market_sample_count": 5000,
        "minimum_market_sample": 30,
        "model_id": "v17-v4-native-regime-filtered-model",
        "model_implementation_sha256": (
            "4e90e06eb340438e909b842a4f40e1dec7eb5ff3231e02c087499bda7646cc7a"
        ),
        "model_snapshot_ref": model,
        "model_training_end_session": None,
        "model_version": "PINNED_RULE_BASED_NO_TRAINING_V1",
        "no_retroactive_causal_backfill": True,
        "observed_through_session": "2026-07-29",
        "performance_evidence_eligible": False,
        "promotion_eligible": False,
        "protocol_version": "myquant.v17.v4",
        "publication_phase": "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION",
        "published_at": "2026-07-30T07:45:00Z",
        "same_session_execution_eligible": False,
        "scope_kind": "FULL_MARKET",
        "scope_ref": scope,
        "shadow_only": True,
        "smoothing_used": False,
        "source_refs": [market],
        "state_order": ["趋势上涨", "震荡低波", "震荡高波", "趋势下跌", "未知"],
        "state_probabilities": {
            "趋势上涨": "0.400000000000",
            "震荡低波": "0.300000000000",
            "震荡高波": "0.100000000000",
            "趋势下跌": "0.100000000000",
            "未知": "0.100000000000",
        },
        "status": "AVAILABLE",
        "strategy_id": STRATEGY,
        "transition_matrix_ref": transition,
        "version": REGIME_EVIDENCE_V2_VERSION,
    }
    payload.update(overrides)
    payload.pop("semantic_sha256", None)
    documents = {
        calendar_path: calendar_document,
        feature_path: feature_document,
    }
    return (
        seal_semantic(payload),
        [feature, model, transition, scope, market, calendar],
        documents,
    )


def test_adapter_normalizes_exact_v4_regime_read_without_inference(tmp_path: Path) -> None:
    read = _manual_read(_artifact())

    result = adapt_v4_regime_evidence(read)

    assert result.status == V4RegimeEvidenceStatus.UNAVAILABLE
    assert result.conditioning_eligible is False
    assert result.conditioning_ineligibility_reason == REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE
    assert result.causal_status == REGIME_HARD_STATE_UNAVAILABLE
    assert result.strategy_id == STRATEGY
    assert result.cutoff == CUTOFF
    assert result.available_at == "2026-07-29T07:59:00Z"
    assert result.created_at == "2026-07-29T08:00:01Z"
    assert result.published_at is None
    assert result.decision_session is None
    assert result.effective_session is None
    assert result.regime_state is None
    assert result.state_probabilities is None
    assert result.regime_artifact_ref == read.root_ref
    assert result.source_identity == "regime-evidence-1"
    assert result.source_role == "markov_evidence"
    assert result.source_version == "myquant.v17.v4.regime-evidence.v1"
    assert result.blockers == (
        "regime_hard_state_absent",
        "regime_posterior_absent",
        "regime_decision_session_absent",
        "regime_effective_session_absent",
        "regime_published_at_absent",
        "regime_source_refs_absent",
        REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE,
    )


def test_adapter_rejects_macro_role_even_when_v4_schema_valid(tmp_path: Path) -> None:
    read = _manual_read(_artifact(role="macro_evidence"))

    with pytest.raises(V4RegimeAdapterError, match="role is not markov_evidence"):
        adapt_v4_regime_evidence(read)


def test_adapter_rejects_policy_or_predecessor_binding_drift(tmp_path: Path) -> None:
    read = _manual_read(_artifact())
    assert read.compatibility_policy_byte_sha256 == V4_COMPATIBILITY_POLICY_BYTE_SHA256
    assert read.predecessor_git_commit == V4_SOURCE_GIT_COMMIT

    with pytest.raises(V4RegimeAdapterError, match="policy or predecessor"):
        adapt_v4_regime_evidence(_with_read(read, policy_sha256="0" * 64))


def test_adapter_rejects_root_identity_tamper(tmp_path: Path) -> None:
    read = _manual_read(_artifact())
    tampered_ref = dict(read.root_ref)
    tampered_ref["artifact_id"] = "other-regime-evidence"

    with pytest.raises(V4RegimeAdapterError, match="root reference identity mismatch"):
        adapt_v4_regime_evidence(_with_read(read, root_ref=tampered_ref))


def test_adapter_rejects_semantic_sha_tamper(tmp_path: Path) -> None:
    read = _manual_read(_artifact())
    tampered_node = V4ClosureNode(
        artifact_id=read.closure[0].artifact_id,
        byte_sha256=read.closure[0].byte_sha256,
        relative_path=read.closure[0].relative_path,
        semantic_sha256="0" * 64,
        validation_mode=read.closure[0].validation_mode,
        version=read.closure[0].version,
    )

    with pytest.raises(V4RegimeAdapterError, match="root reference identity mismatch"):
        adapt_v4_regime_evidence(_with_read(read, closure=(tampered_node,)))


def test_adapter_rejects_unregistered_posterior_shape_without_argmax(tmp_path: Path) -> None:
    read = _manual_read(_artifact())
    posterior_document = dict(read.document)
    posterior_document["version"] = "myquant.v17.v4.regime-posterior.v1"
    posterior_document["state_probabilities"] = {
        "趋势上涨": "0.600000000000",
        "趋势下跌": "0.400000000000",
    }

    with pytest.raises(V4RegimeAdapterError, match="version mismatch"):
        adapt_v4_regime_evidence(_with_read(read, document=posterior_document))


def test_adapter_verifies_v2_without_recomputing_argmax_but_rejects_conditioning() -> None:
    document, refs, documents = _v2_document(
        hard_state="趋势上涨",
        state_probabilities={
            "趋势上涨": "0.100000000000",
            "震荡低波": "0.600000000000",
            "震荡高波": "0.100000000000",
            "趋势下跌": "0.100000000000",
            "未知": "0.100000000000",
        },
    )
    read = _manual_read(document, refs, documents)

    result = adapt_v4_regime_evidence(read)

    assert result.status == V4RegimeEvidenceStatus.CONDITIONING_INELIGIBLE
    assert result.conditioning_eligible is False
    assert result.conditioning_ineligibility_reason == REGIME_EVIDENCE_V2_NON_DEPLOYABLE
    assert result.regime_state == "趋势上涨"
    assert result.hard_state == "趋势上涨"
    assert result.state_order == ("趋势上涨", "震荡低波", "震荡高波", "趋势下跌", "未知")
    assert result.state_probabilities is not None
    assert result.state_probabilities["震荡低波"] == "0.600000000000"
    assert result.inference_kind == "FILTERED_CAUSAL"
    assert result.smoothing_used is False
    assert result.publication_phase == "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION"
    assert result.scope_kind == "FULL_MARKET"
    assert result.source_version == REGIME_EVIDENCE_V2_VERSION
    assert result.source_commit == V4_SOURCE_GIT_COMMIT


def test_adapter_marks_v2_unknown_as_valid_but_conditioning_ineligible() -> None:
    document, refs, documents = _v2_document(hard_state="未知")
    result = adapt_v4_regime_evidence(_manual_read(document, refs, documents))

    assert result.status == V4RegimeEvidenceStatus.CONDITIONING_INELIGIBLE
    assert result.conditioning_eligible is False
    assert result.conditioning_ineligibility_reason == REGIME_EVIDENCE_V2_NON_DEPLOYABLE
    assert result.blockers == (
        REGIME_EVIDENCE_V2_NON_DEPLOYABLE,
        REGIME_HARD_STATE_UNKNOWN,
    )


def test_adapter_rejects_v2_posterior_sum_drift_without_normalizing() -> None:
    document, refs, documents = _v2_document(
        state_probabilities={
            "趋势上涨": "0.400000000000",
            "震荡低波": "0.300000000000",
            "震荡高波": "0.100000000000",
            "趋势下跌": "0.100000000000",
            "未知": "0.099999999999",
        }
    )

    with pytest.raises(V4RegimeAdapterError, match="sum"):
        adapt_v4_regime_evidence(_manual_read(document, refs, documents))


def test_adapter_requires_exact_pinned_v4_policy_and_model_implementation() -> None:
    document, refs, documents = _v2_document()
    bad_policy = dict(document)
    bad_policy["inference_policy_ref"] = {
        **document["inference_policy_ref"],
        "byte_sha256": "0" * 64,
    }
    bad_policy = seal_semantic(
        {key: value for key, value in bad_policy.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V4RegimeAdapterError, match="inference policy ref"):
        adapt_v4_regime_evidence(_manual_read(bad_policy, refs, documents))

    bad_model = dict(document)
    bad_model["model_implementation_sha256"] = "0" * 64
    bad_model = seal_semantic(
        {key: value for key, value in bad_model.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V4RegimeAdapterError, match="model implementation"):
        adapt_v4_regime_evidence(_manual_read(bad_model, refs, documents))


def test_adapter_rejects_v2_scope_or_smoothing_drift() -> None:
    document, refs, documents = _v2_document(smoothing_used=True)
    with pytest.raises(V4RegimeAdapterError, match="smoothing_used"):
        adapt_v4_regime_evidence(_manual_read(document, refs, documents))

    document, refs, documents = _v2_document(scope_kind="SUBSET")
    with pytest.raises(V4RegimeAdapterError, match="scope_kind"):
        adapt_v4_regime_evidence(_manual_read(document, refs, documents))


def test_adapter_requires_sealed_calendar_to_prove_previous_open_session() -> None:
    document, refs, documents = _v2_document(_calendar_sessions=["2026-07-28", "2026-07-30"])

    with pytest.raises(V4RegimeAdapterError, match="sealed calendar"):
        adapt_v4_regime_evidence(_manual_read(document, refs, documents))


def test_adapter_accepts_only_composite_finalized_contiguous_v3(
    tmp_path: Path,
) -> None:
    sessions = _V3_SUPPORT._business_sessions("2026-07-29", 8)
    factory = _V3_SUPPORT.V3Factory(tmp_path, sessions)
    genesis, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    contiguous, _ = factory.build(
        observed=sessions[1],
        effective=sessions[2],
        created_at="2026-07-30T15:21:00Z",
        prior=genesis,
    )
    read = _read_v3(tmp_path, contiguous)
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    result = adapt_v4_regime_evidence(
        read,
        checkpoint_relative_path=contiguous.chain_checkpoint_path,
        checkpoint_byte_sha256=contiguous.chain_checkpoint_sha256,
    )

    assert result.source_version == "myquant.v17.v4.regime-evidence.v3"
    assert result.finalized is True
    assert result.continuity_kind == "CONTIGUOUS"
    assert result.conditioning_eligible is True
    assert result.checkpoint_ref == contiguous.document["current_checkpoint_ref"]
    assert result.chain_digest_sha256 == contiguous.document["global_accumulator"]
    assert result.transition_commitment_sha256 == contiguous.document["record_commitment"]
    assert before == {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }


def test_adapter_rejects_v3_without_exact_checkpoint_finality(
    tmp_path: Path,
) -> None:
    sessions = _V3_SUPPORT._business_sessions("2026-07-29", 5)
    factory = _V3_SUPPORT.V3Factory(tmp_path, sessions)
    genesis, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    read = _read_v3(tmp_path, genesis)

    with pytest.raises(V4RegimeAdapterError, match=REGIME_EVIDENCE_V3_NOT_FINALIZED):
        adapt_v4_regime_evidence(read)
    with pytest.raises(V4RegimeAdapterError, match=REGIME_EVIDENCE_V3_NOT_FINALIZED):
        adapt_v4_regime_evidence(
            read,
            checkpoint_relative_path=genesis.chain_checkpoint_path,
            checkpoint_byte_sha256="0" * 64,
        )


def test_adapter_marks_genesis_and_recovery_v3_conditioning_ineligible(
    tmp_path: Path,
) -> None:
    sessions = _V3_SUPPORT._business_sessions("2026-07-29", 8)
    factory = _V3_SUPPORT.V3Factory(tmp_path, sessions)
    genesis, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    recovery, _ = factory.build(
        observed=sessions[4],
        effective=sessions[5],
        created_at=f"{sessions[4]}T08:21:00Z",
        prior=genesis,
    )

    for result, reason in (
        (genesis, REGIME_CONTINUITY_GENESIS),
        (recovery, REGIME_CONTINUITY_RECOVERY),
    ):
        normalized = adapt_v4_regime_evidence(
            _read_v3(tmp_path, result),
            checkpoint_relative_path=result.chain_checkpoint_path,
            checkpoint_byte_sha256=result.chain_checkpoint_sha256,
        )
        assert normalized.finalized is True
        assert normalized.conditioning_eligible is False
        assert normalized.conditioning_ineligibility_reason == reason


def test_adapter_fail_closes_composite_finality_conflicts_and_orphan_evidence(
    tmp_path: Path,
) -> None:
    sessions = _V3_SUPPORT._business_sessions("2026-07-29", 8)
    factory = _V3_SUPPORT.V3Factory(tmp_path, sessions)
    genesis, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    contiguous, _ = factory.build(
        observed=sessions[1],
        effective=sessions[2],
        created_at="2026-07-30T15:21:00Z",
        prior=genesis,
    )
    read = _read_v3(tmp_path, contiguous)
    checkpoint_path = contiguous.chain_checkpoint_path

    for field, value in (
        ("hard_state", "趋势下跌"),
        ("record_commitment", "0" * 64),
        ("global_accumulator", "1" * 64),
        ("segment_accumulator", "2" * 64),
        ("chain_id", "3" * 64),
        ("segment_id", "4" * 64),
        ("phase", "RECOVERY"),
    ):
        documents = copy.deepcopy(dict(read.documents))
        documents[checkpoint_path][field] = value
        with pytest.raises(
            V4RegimeAdapterError,
            match=REGIME_EVIDENCE_V3_NOT_FINALIZED,
        ):
            adapt_v4_regime_evidence(
                _with_read(read, documents=documents),
                checkpoint_relative_path=checkpoint_path,
                checkpoint_byte_sha256=contiguous.chain_checkpoint_sha256,
            )

    root_document = copy.deepcopy(dict(read.document))
    root_document["source_refs"] = root_document["source_refs"][1:]
    documents = copy.deepcopy(dict(read.documents))
    documents[contiguous.evidence_path] = root_document
    with pytest.raises(
        V4RegimeAdapterError,
        match=REGIME_EVIDENCE_V3_NOT_FINALIZED,
    ):
        adapt_v4_regime_evidence(
            _with_read(read, document=root_document, documents=documents),
            checkpoint_relative_path=checkpoint_path,
            checkpoint_byte_sha256=contiguous.chain_checkpoint_sha256,
        )

    orphan_closure = tuple(node for node in read.closure if node.relative_path != checkpoint_path)
    orphan_documents = {
        path: document for path, document in read.documents.items() if path != checkpoint_path
    }
    with pytest.raises(
        V4RegimeAdapterError,
        match=REGIME_EVIDENCE_V3_NOT_FINALIZED,
    ):
        adapt_v4_regime_evidence(
            _with_read(
                read,
                closure=orphan_closure,
                documents=orphan_documents,
            ),
            checkpoint_relative_path=checkpoint_path,
            checkpoint_byte_sha256=contiguous.chain_checkpoint_sha256,
        )


@pytest.mark.parametrize(
    ("ref_field", "historical_field"),
    [
        ("model_snapshot_ref", "training_source_refs"),
        ("transition_matrix_ref", "source_evidence_refs"),
    ],
)
def test_adapter_rejects_historical_v3_refs_without_recursive_traversal(
    tmp_path: Path,
    ref_field: str,
    historical_field: str,
) -> None:
    sessions = _V3_SUPPORT._business_sessions("2026-07-29", 5)
    factory = _V3_SUPPORT.V3Factory(tmp_path, sessions)
    genesis, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    read = _read_v3(tmp_path, genesis)
    referenced_path = genesis.document[ref_field]["relative_path"]
    documents = copy.deepcopy(dict(read.documents))
    documents[referenced_path][historical_field] = [
        {
            "artifact_id": "forbidden-historical-evidence",
            "artifact_version": "myquant.v17.v4.regime-evidence.v3",
            "byte_sha256": "0" * 64,
            "cutoff": genesis.document["cutoff"],
            "relative_path": (
                "data/private/v17_v4_sources/regime_evidence/" "forbidden-historical-evidence.json"
            ),
            "semantic_sha256": "1" * 64,
            "strategy_id": genesis.document["strategy_id"],
        }
    ]

    with pytest.raises(V4RegimeAdapterError, match="bounded direct closure mismatch"):
        adapt_v4_regime_evidence(
            _with_read(read, documents=documents),
            checkpoint_relative_path=genesis.chain_checkpoint_path,
            checkpoint_byte_sha256=genesis.chain_checkpoint_sha256,
        )
