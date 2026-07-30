from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes
import quant_investor.v17_v4_runtime.cli as cli
from quant_investor.v17_v4_runtime.source_storage import SourceStore

SHA = "a" * 64


def _build_argv(workspace: Path) -> list[str]:
    return [
        "regime-evidence-v3-build",
        "--workspace-root",
        str(workspace),
        "--evidence-id",
        "0" * 64,
        "--strategy-id",
        "cn-aggressive-tech-manufacturing",
        "--decision-session",
        "2026-07-30",
        "--cutoff",
        "2026-07-30T07:00:00Z",
        "--created-at",
        "2026-07-30T00:01:00Z",
        "--inference-policy-path",
        "resources/regime_inference_policy.v2.json",
        "--inference-policy-sha256",
        SHA,
        "--model-snapshot-path",
        "data/private/v17_v4_sources/regime_inputs/model.v2.json",
        "--model-snapshot-sha256",
        SHA,
        "--transition-matrix-path",
        "data/private/v17_v4_sources/regime_inputs/transition.v2.json",
        "--transition-matrix-sha256",
        SHA,
        "--feature-snapshot-path",
        "data/private/v17_v4_sources/regime_inputs/feature.v1.json",
        "--feature-snapshot-sha256",
        SHA,
    ]


def _document() -> dict[str, Any]:
    return {
        "available_at": "2026-07-30T00:01:00Z",
        "blocker_codes": [],
        "chain_id": "b" * 64,
        "created_at": "2026-07-30T00:01:00Z",
        "cutoff": "2026-07-30T07:00:00Z",
        "decision_session": "2026-07-30",
        "effective_session": "2026-07-30",
        "evidence_id": "c" * 64,
        "finalized_evidence_ordinal": 0,
        "global_accumulator": "d" * 64,
        "hard_state": "趋势上涨",
        "inference_kind": "FILTERED_CAUSAL",
        "missing_sessions": [],
        "observed_through_session": "2026-07-29",
        "phase": "GENESIS",
        "publication_phase": "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION",
        "published_at": "2026-07-30T00:01:00Z",
        "record_commitment": "e" * 64,
        "scope_kind": "FULL_MARKET",
        "segment_accumulator": "f" * 64,
        "segment_id": "1" * 64,
        "segment_index": 0,
        "segment_position": 0,
        "smoothing_used": False,
        "state_probabilities": {
            "趋势上涨": "0.400000000000",
            "震荡低波": "0.200000000000",
            "震荡高波": "0.150000000000",
            "趋势下跌": "0.150000000000",
            "未知": "0.100000000000",
        },
        "strategy_id": "cn-aggressive-tech-manufacturing",
    }


def test_v3_parser_requires_exact_current_closure_and_optional_prior_pairs() -> None:
    parser = cli._parser()
    subparsers = cast(
        Any,
        next(action for action in parser._actions if action.dest == "command"),
    )
    build = subparsers.choices["regime-evidence-v3-build"]
    required = {action.dest for action in build._actions if getattr(action, "required", False)}
    assert required == {
        "workspace_root",
        "evidence_id",
        "strategy_id",
        "decision_session",
        "cutoff",
        "created_at",
        "inference_policy_path",
        "inference_policy_sha256",
        "model_snapshot_path",
        "model_snapshot_sha256",
        "transition_matrix_path",
        "transition_matrix_sha256",
        "feature_snapshot_path",
        "feature_snapshot_sha256",
    }
    optional = {action.dest for action in build._actions} - required
    assert {
        "prior_evidence_path",
        "prior_evidence_sha256",
        "prior_checkpoint_path",
        "prior_checkpoint_sha256",
        "chain_anchor_path",
        "chain_anchor_sha256",
    } <= optional
    assert "latest" not in optional | required


def test_v3_build_wires_only_explicit_inputs_and_attests_no_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "created": True,
            "document": _document(),
            "evidence_path": "data/private/v17_v4_sources/regime_evidence/evidence.json",
            "evidence_sha256": SHA,
            "reused": False,
            "status": "AVAILABLE",
        }

    monkeypatch.setattr(cli, "build_regime_evidence_v3", fake_build)
    assert cli.main(_build_argv(tmp_path)) == 0
    body = json.loads(capsys.readouterr().out)

    assert captured["workspace_root"] == str(tmp_path.resolve())
    assert captured["prior_evidence_path"] is None
    assert captured["prior_checkpoint_path"] is None
    assert captured["chain_anchor_path"] is None
    assert body["artifact_version"] == "myquant.v17.v4.regime-evidence.v3"
    assert body["phase"] == "GENESIS"
    assert body["default_protocol_state"] == "V15_DEFAULT"
    assert body["global_activation_state"] == "INACTIVE"
    assert body["run_state"] == "INACTIVE"
    assert all(value is False for value in body["authority"].values())
    assert all(value is False for value in body["side_effects"].values())


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--prior-evidence-path", "data/private/prior.json"),
        ("--prior-checkpoint-sha256", SHA),
        ("--chain-anchor-path", "data/private/anchor.json"),
    ],
)
def test_v3_build_rejects_half_optional_pair_before_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    called = False

    def forbidden(**kwargs: Any) -> object:
        del kwargs
        nonlocal called
        called = True
        raise AssertionError("producer must not run")

    monkeypatch.setattr(cli, "build_regime_evidence_v3", forbidden)
    assert cli.main([*_build_argv(tmp_path), flag, value]) == 2
    body = json.loads(capsys.readouterr().out)
    assert called is False
    assert body["blocker_codes"] == ["REGIME_EVIDENCE_V3_EXPLICIT_PAIR_REQUIRED"]
    assert all(value is False for value in body["authority"].values())


def test_v3_status_requires_explicit_path_and_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_replay(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _document()

    monkeypatch.setattr(cli, "replay_regime_evidence_v3", fake_replay)
    assert (
        cli.main(
            [
                "regime-evidence-v3-status",
                "--workspace-root",
                str(tmp_path),
                "--artifact-path",
                "data/private/evidence.v3.json",
                "--expected-sha256",
                SHA,
            ]
        )
        == 0
    )
    body = json.loads(capsys.readouterr().out)
    assert captured == {
        "workspace_root": str(tmp_path.resolve()),
        "evidence_path": "data/private/evidence.v3.json",
        "evidence_sha256": SHA,
    }
    assert body["status"] == "AVAILABLE"
    assert body["phase"] == "GENESIS"


def test_v3_audit_consumes_one_exact_canonical_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = {
        "audit_as_of_session": "2026-07-30",
        "evidence_refs": [
            {
                "artifact_id": "c" * 64,
                "artifact_version": "myquant.v17.v4.regime-evidence.v3",
                "byte_sha256": SHA,
                "cutoff": "2026-07-30T07:00:00Z",
                "relative_path": "data/private/evidence.v3.json",
                "semantic_sha256": "b" * 64,
                "strategy_id": "cn-aggressive-tech-manufacturing",
            }
        ],
        "expected_head_path": "data/private/evidence.v3.json",
        "expected_head_sha256": SHA,
    }
    raw = canonical_resource_bytes(request)
    write = SourceStore(tmp_path).write_exact_once(
        "data/private/v17_v4_sources/regime_v3_audit/audit.json",
        raw,
    )
    captured: dict[str, Any] = {}

    def fake_audit(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "bounded": True,
            "record_count": 1,
            "status": "EXACT_CHAIN_AUDIT_VERIFIED",
        }

    monkeypatch.setattr(cli, "audit_regime_chain_v3", fake_audit)
    assert (
        cli.main(
            [
                "regime-chain-v3-audit",
                "--workspace-root",
                str(tmp_path),
                "--request-path",
                write.relative_path,
                "--request-sha256",
                write.byte_sha256,
            ]
        )
        == 0
    )
    body = json.loads(capsys.readouterr().out)
    assert captured["workspace_root"] == str(tmp_path.resolve())
    assert captured["evidence_refs"] == request["evidence_refs"]
    assert captured["expected_head_path"] == request["expected_head_path"]
    assert captured["audit_as_of_session"] == "2026-07-30"
    assert body["status"] == "EXACT_CHAIN_AUDIT_VERIFIED"
    assert body["artifact_version"] == "myquant.v17.v4.regime-evidence.v3"


def test_v3_audit_missing_request_is_structured_blocked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli.main(
            [
                "regime-chain-v3-audit",
                "--workspace-root",
                str(tmp_path),
                "--request-path",
                "data/private/missing.json",
                "--request-sha256",
                SHA,
            ]
        )
        == 2
    )
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "BLOCKED"
    assert body["blocker_codes"] == ["REGIME_EVIDENCE_V3_AUDIT_REQUEST_INVALID"]
    assert all(value is False for value in body["authority"].values())
