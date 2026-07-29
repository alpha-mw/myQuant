from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes, seal_semantic
from quant_investor.v17_v5_contract.validators import (
    V4_COMPATIBILITY_POLICY_BYTE_SHA256,
    V4_SOURCE_GIT_COMMIT,
)
from quant_investor.v17_v5_runtime.v4_compat_reader import (
    V4ClosureNode,
    V4CompatibilityRead,
    read_v4_artifact,
)
from quant_investor.v17_v5_runtime.v4_regime_adapter import (
    REGIME_HARD_STATE_UNAVAILABLE,
    V4RegimeAdapterError,
    V4RegimeEvidenceStatus,
    adapt_v4_regime_evidence,
)

CUTOFF = "2026-07-29T08:00:00Z"
RELATIVE_PATH = "data/private/v17_v4_runs/run-1/regime.json"
STRATEGY = "quant-first"


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


def _with_read(
    read: V4CompatibilityRead,
    *,
    document: dict[str, Any] | None = None,
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
        documents=read.documents,
        predecessor_git_commit=read.predecessor_git_commit,
        root_ref=read.root_ref if root_ref is None else root_ref,
        terminal_bindings=read.terminal_bindings,
    )


def test_adapter_normalizes_exact_v4_regime_read_without_inference(tmp_path: Path) -> None:
    sha256 = _write_artifact(tmp_path)
    read = _read(tmp_path, sha256)

    result = adapt_v4_regime_evidence(read)

    assert result.status == V4RegimeEvidenceStatus.UNAVAILABLE
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
    )


def test_adapter_rejects_macro_role_even_when_v4_schema_valid(tmp_path: Path) -> None:
    sha256 = _write_artifact(tmp_path, _artifact(role="macro_evidence"))
    read = _read(tmp_path, sha256)

    with pytest.raises(V4RegimeAdapterError, match="role is not markov_evidence"):
        adapt_v4_regime_evidence(read)


def test_adapter_rejects_policy_or_predecessor_binding_drift(tmp_path: Path) -> None:
    sha256 = _write_artifact(tmp_path)
    read = _read(tmp_path, sha256)
    assert read.compatibility_policy_byte_sha256 == V4_COMPATIBILITY_POLICY_BYTE_SHA256
    assert read.predecessor_git_commit == V4_SOURCE_GIT_COMMIT

    with pytest.raises(V4RegimeAdapterError, match="policy or predecessor"):
        adapt_v4_regime_evidence(_with_read(read, policy_sha256="0" * 64))


def test_adapter_rejects_root_identity_tamper(tmp_path: Path) -> None:
    sha256 = _write_artifact(tmp_path)
    read = _read(tmp_path, sha256)
    tampered_ref = dict(read.root_ref)
    tampered_ref["artifact_id"] = "other-regime-evidence"

    with pytest.raises(V4RegimeAdapterError, match="root reference identity mismatch"):
        adapt_v4_regime_evidence(_with_read(read, root_ref=tampered_ref))


def test_adapter_rejects_semantic_sha_tamper(tmp_path: Path) -> None:
    sha256 = _write_artifact(tmp_path)
    read = _read(tmp_path, sha256)
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
    sha256 = _write_artifact(tmp_path)
    read = _read(tmp_path, sha256)
    posterior_document = dict(read.document)
    posterior_document["version"] = "myquant.v17.v4.regime-posterior.v1"
    posterior_document["state_probabilities"] = {
        "趋势上涨": "0.600000000000",
        "趋势下跌": "0.400000000000",
    }

    with pytest.raises(V4RegimeAdapterError, match="version mismatch"):
        adapt_v4_regime_evidence(_with_read(read, document=posterior_document))
