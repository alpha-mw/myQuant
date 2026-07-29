from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_contract.canonical import (
    load_canonical_resource,
)
import quant_investor.v17_v4_runtime.regime_evidence_v2 as subject
from quant_investor.v17_v4_runtime.source_storage import SourceStore
from tests.unit.test_v17_v4_regime_evidence_v2_producer import (
    BOOTSTRAP_CREATED,
    CUTOFF,
    NO_AUTHORITY,
    STRATEGY,
    _clock,
    _sha,
    _store_document,
    build_fixture,
    make_regime_fixture,
)


def test_exact_replay_and_status_do_not_use_current_clock_or_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    built = build_fixture(fixture)

    def forbidden(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("replay/status must not scan or consult time")

    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    replayed = subject.read_regime_evidence_v2(
        workspace_root=tmp_path,
        evidence_path=built.evidence_path,
        evidence_sha256=built.evidence_sha256,
    )
    status = subject.regime_evidence_v2_status(
        workspace_root=tmp_path,
        evidence_path=built.evidence_path,
        evidence_sha256=built.evidence_sha256,
    )
    assert replayed == built.document
    assert status["status"] == "AVAILABLE"
    assert status["hard_state"] == replayed["hard_state"]
    assert status["blocker_codes"] == []


def test_normal_inference_requires_and_uses_contiguous_sealed_prior(
    tmp_path: Path,
) -> None:
    first_fixture = make_regime_fixture(tmp_path)
    first = build_fixture(first_fixture)
    second_fixture = make_regime_fixture(
        tmp_path,
        observed="2026-07-30",
        effective="2026-07-31",
        created_at="2026-07-30T07:05:00Z",
        prior=(
            first.document,
            first.evidence_path,
            first.evidence_sha256,
        ),
    )
    second = build_fixture(second_fixture)
    expected = subject._posterior(
        previous=first.document["state_probabilities"],
        transition_matrix=subject.load_packaged_json(subject.INFERENCE_POLICY_PATH)[
            "transition_matrix"
        ],
        likelihoods=second_fixture.feature["state_likelihoods"],
    )
    assert second.document["lineage_phase"] == "NORMAL"
    assert second.document["observed_through_session"] == (first.document["effective_session"])
    assert second.document["state_probabilities"] == expected


def test_replay_rejects_semantically_resealed_posterior_and_identity(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    built = build_fixture(fixture)
    path = tmp_path / built.evidence_path
    document = load_canonical_resource(path.read_bytes())
    assert type(document) is dict
    document.pop("semantic_sha256")
    document["state_probabilities"] = {
        "趋势上涨": "0.000000000000",
        "震荡低波": "0.000000000000",
        "震荡高波": "0.000000000000",
        "趋势下跌": "0.000000000000",
        "未知": "1.000000000000",
    }
    document["hard_state"] = "未知"
    document["evidence_id"] = subject.regime_evidence_v2_id(document)
    tampered = canonical_resource_bytes(seal_semantic(document))
    path.write_bytes(tampered)
    path.chmod(0o600)

    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.read_regime_evidence_v2(
            workspace_root=tmp_path,
            evidence_path=built.evidence_path,
            evidence_sha256=_sha(tampered),
        )
    assert raised.value.blocker_code in {
        subject.IDENTITY_BLOCKER,
        subject.SEMANTIC_BLOCKER,
        subject.INPUT_TAMPER_BLOCKER,
    }


def test_replay_rejects_direct_snapshot_byte_drift(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    built = build_fixture(fixture)
    feature_path = tmp_path / fixture.feature_path
    feature = load_canonical_resource(feature_path.read_bytes())
    assert type(feature) is dict
    feature.pop("semantic_sha256")
    feature["breadth"] = "0.700000000000"
    feature["feature_snapshot_id"] = subject._snapshot_id(
        feature,
        identity_field="feature_snapshot_id",
    )
    drifted = canonical_resource_bytes(seal_semantic(feature))
    feature_path.write_bytes(drifted)
    feature_path.chmod(0o600)

    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.replay_regime_evidence_v2(
            workspace_root=tmp_path,
            evidence_path=built.evidence_path,
            evidence_sha256=built.evidence_sha256,
        )
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER


def test_path_escape_symlink_and_hardlink_are_hard_failures(
    tmp_path: Path,
) -> None:
    escape_fixture = make_regime_fixture(tmp_path / "escape")
    escape_fixture.kwargs["feature_snapshot_path"] = "data/private/v17_v4_sources/../escape.json"
    with pytest.raises(subject.RegimeEvidenceV2Error) as escape:
        subject.build_regime_evidence_v2(
            **escape_fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert escape.value.blocker_code == subject.INPUT_TAMPER_BLOCKER

    symlink_fixture = make_regime_fixture(tmp_path / "symlink")
    source = symlink_fixture.workspace / symlink_fixture.feature_path
    alias = source.with_name("feature-alias.json")
    alias.symlink_to(source)
    symlink_fixture.kwargs["feature_snapshot_path"] = str(
        alias.relative_to(symlink_fixture.workspace)
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as symlink:
        subject.build_regime_evidence_v2(
            **symlink_fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert symlink.value.blocker_code == subject.INPUT_TAMPER_BLOCKER

    hardlink_fixture = make_regime_fixture(tmp_path / "hardlink")
    hard_source = hardlink_fixture.workspace / hardlink_fixture.feature_path
    hard_alias = hard_source.with_name("feature-hardlink.json")
    os.link(hard_source, hard_alias)
    with pytest.raises(subject.RegimeEvidenceV2Error) as hardlink:
        subject.build_regime_evidence_v2(
            **hardlink_fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert hardlink.value.blocker_code == subject.INPUT_TAMPER_BLOCKER


def test_closure_rejects_partial_hidden_ref_and_identity_alias(
    tmp_path: Path,
) -> None:
    store = SourceStore(tmp_path)
    store.initialize()
    root = "data/private/v17_v4_sources/regime_inputs"
    partial = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": BOOTSTRAP_CREATED,
            "created_at": BOOTSTRAP_CREATED,
            "cutoff": CUTOFF,
            "hidden": {
                "byte_sha256": "a" * 64,
                "relative_path": f"{root}/hidden.json",
            },
            "partial_terminal_id": "partial-terminal",
            "protocol_version": subject.PROTOCOL_VERSION,
            "strategy_id": STRATEGY,
            "terminal_role": "PARTIAL",
            "version": "myquant.v17.v4.regime-partial-terminal.v1",
        }
    )
    _, partial_ref = _store_document(
        store,
        path=f"{root}/partial.json",
        document=partial,
    )
    loader = subject._ClosureLoader(store)
    with pytest.raises(subject.RegimeEvidenceV2Error) as hidden:
        loader.document(partial_ref)
    assert hidden.value.blocker_code == subject.INPUT_TAMPER_BLOCKER

    first = seal_semantic(
        {
            "alias_terminal_id": "same-identity",
            "authority": dict(NO_AUTHORITY),
            "available_at": BOOTSTRAP_CREATED,
            "created_at": BOOTSTRAP_CREATED,
            "cutoff": CUTOFF,
            "protocol_version": subject.PROTOCOL_VERSION,
            "strategy_id": STRATEGY,
            "terminal_role": "ALIAS",
            "version": "myquant.v17.v4.regime-alias-terminal.v1",
        }
    )
    second = dict(first)
    second.pop("semantic_sha256")
    second["terminal_role"] = "ALIAS_DRIFT"
    second = seal_semantic(second)
    _, first_ref = _store_document(
        store,
        path=f"{root}/alias-first.json",
        document=first,
    )
    _, second_ref = _store_document(
        store,
        path=f"{root}/alias-second.json",
        document=second,
    )
    alias_loader = subject._ClosureLoader(store)
    alias_loader.document(first_ref)
    with pytest.raises(subject.RegimeEvidenceV2Error) as alias:
        alias_loader.document(second_ref)
    assert alias.value.blocker_code == subject.INPUT_TAMPER_BLOCKER


def test_closure_node_limit_is_fail_closed(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)
    store.initialize()
    loader = subject._ClosureLoader(store)
    for index in range(subject.MAX_CLOSURE_NODES):
        document = seal_semantic(
            {
                "authority": dict(NO_AUTHORITY),
                "available_at": BOOTSTRAP_CREATED,
                "created_at": BOOTSTRAP_CREATED,
                "cutoff": CUTOFF,
                "limit_terminal_id": f"limit-{index}",
                "protocol_version": subject.PROTOCOL_VERSION,
                "strategy_id": STRATEGY,
                "terminal_role": "LIMIT",
                "version": "myquant.v17.v4.regime-limit-terminal.v1",
            }
        )
        _, reference = _store_document(
            store,
            path=("data/private/v17_v4_sources/regime_inputs/" f"limit-{index}.json"),
            document=document,
        )
        loader.document(reference)
    overflow = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": BOOTSTRAP_CREATED,
            "created_at": BOOTSTRAP_CREATED,
            "cutoff": CUTOFF,
            "limit_terminal_id": "limit-overflow",
            "protocol_version": subject.PROTOCOL_VERSION,
            "strategy_id": STRATEGY,
            "terminal_role": "LIMIT",
            "version": "myquant.v17.v4.regime-limit-terminal.v1",
        }
    )
    _, overflow_ref = _store_document(
        store,
        path=("data/private/v17_v4_sources/regime_inputs/" "limit-overflow.json"),
        document=overflow,
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        loader.document(overflow_ref)
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER


def test_direct_parquet_or_opaque_raw_reference_is_fail_closed(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)
    store.initialize()
    relative_path = "data/private/v17_v4_sources/regime_inputs/raw.parquet"
    raw = b"PAR1not-a-registered-canonical-json-artifactPAR1"
    write = store.write_exact_once(relative_path, raw)
    reference = {
        "artifact_id": "raw-market",
        "artifact_version": "myquant.v17.v4.regime-market-raw.v1",
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": CUTOFF,
        "relative_path": relative_path,
        "semantic_sha256": "a" * 64,
        "strategy_id": STRATEGY,
    }

    loader = subject._ClosureLoader(store)
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        loader(reference)
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER
    assert "canonical sealed JSON" in raised.value.detail


def test_runtime_has_no_legacy_markov_jsonl_or_scanning_imports() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "quant_investor.regime" not in imported
    assert ".jsonl" not in source
    assert ".glob(" not in source
    assert ".rglob(" not in source
    assert "provider" not in source.lower()
    assert "requests" not in imported
