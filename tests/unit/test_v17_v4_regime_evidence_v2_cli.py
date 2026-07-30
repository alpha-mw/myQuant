from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import quant_investor.v17_v4_runtime.cli as cli

SHA = "a" * 64


def _build_argv(workspace: Path) -> list[str]:
    return [
        "regime-evidence-build",
        "--workspace-root",
        str(workspace),
        "--evidence-id",
        "regime-evidence-20260730",
        "--strategy-id",
        "cn-aggressive-tech-manufacturing",
        "--decision-session",
        "2026-07-30",
        "--cutoff",
        "2026-07-30T07:00:00Z",
        "--created-at",
        "2026-07-29T07:05:00Z",
        "--inference-policy-path",
        "resources/regime_inference_policy.v1.json",
        "--inference-policy-sha256",
        SHA,
        "--model-snapshot-path",
        "data/private/v17_v4_sources/regime/model.json",
        "--model-snapshot-sha256",
        SHA,
        "--transition-matrix-path",
        "data/private/v17_v4_sources/regime/transition.json",
        "--transition-matrix-sha256",
        SHA,
        "--feature-snapshot-path",
        "data/private/v17_v4_sources/regime/features.json",
        "--feature-snapshot-sha256",
        SHA,
        "--prior-evidence-path",
        (
            "data/private/v17_v4_sources/regime_evidence/"
            "cn-aggressive-tech-manufacturing/2026-07-29/regime_evidence.v2.json"
        ),
        "--prior-evidence-sha256",
        SHA,
    ]


def _document() -> dict[str, Any]:
    return {
        "available_at": "2026-07-29T07:05:00Z",
        "blocker_codes": [],
        "coverage_ratio": "1.000000000000",
        "created_at": "2026-07-29T07:05:00Z",
        "decision_session": "2026-07-30",
        "effective_session": "2026-07-30",
        "evidence_id": "regime-evidence-20260730",
        "hard_state": "趋势上涨",
        "inference_kind": "FILTERED_CAUSAL",
        "market_sample_count": 5502,
        "minimum_market_sample": 30,
        "observed_through_session": "2026-07-29",
        "publication_phase": "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION",
        "state_probabilities": {
            "趋势上涨": "0.400000000000",
            "震荡低波": "0.200000000000",
            "震荡高波": "0.150000000000",
            "趋势下跌": "0.150000000000",
            "未知": "0.100000000000",
        },
        "published_at": "2026-07-29T07:05:00Z",
        "scope_kind": "FULL_MARKET",
        "smoothing_used": False,
        "replay_result": {
            "closure_replayed": True,
            "hard_state_reclassified": False,
            "status": "EXACT_REPLAY_VERIFIED",
        },
        "strategy_id": "cn-aggressive-tech-manufacturing",
    }


def test_build_parser_has_exact_required_closure_and_optional_prior_pair() -> None:
    parser = cli._parser()
    subparsers = cast(
        Any,
        next(action for action in parser._actions if action.dest == "command"),
    )
    build = subparsers.choices["regime-evidence-build"]
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
    assert {"prior_evidence_path", "prior_evidence_sha256"} <= optional
    assert "observed_through_session" not in optional | required
    assert "effective_session" not in optional | required
    assert "output_path" not in optional | required


def test_build_wires_only_explicit_inputs_and_emits_full_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "status": "AVAILABLE",
            "evidence_id": "regime-evidence-20260730",
            "evidence_path": (
                "data/private/v17_v4_sources/regime_evidence/"
                "cn-aggressive-tech-manufacturing/2026-07-30/regime_evidence.v2.json"
            ),
            "evidence_sha256": SHA,
            "created": True,
            "reused": False,
            "document": _document(),
        }

    monkeypatch.setattr(cli, "build_regime_evidence_v2", fake_build)

    assert cli.main(_build_argv(tmp_path)) == 0
    body = json.loads(capsys.readouterr().out)

    assert captured == {
        "workspace_root": str(tmp_path.resolve()),
        "evidence_id": "regime-evidence-20260730",
        "strategy_id": "cn-aggressive-tech-manufacturing",
        "decision_session": "2026-07-30",
        "cutoff": "2026-07-30T07:00:00Z",
        "created_at": "2026-07-29T07:05:00Z",
        "inference_policy_path": "resources/regime_inference_policy.v1.json",
        "inference_policy_sha256": SHA,
        "model_snapshot_path": "data/private/v17_v4_sources/regime/model.json",
        "model_snapshot_sha256": SHA,
        "transition_matrix_path": ("data/private/v17_v4_sources/regime/transition.json"),
        "transition_matrix_sha256": SHA,
        "feature_snapshot_path": ("data/private/v17_v4_sources/regime/features.json"),
        "feature_snapshot_sha256": SHA,
        "prior_evidence_path": (
            "data/private/v17_v4_sources/regime_evidence/"
            "cn-aggressive-tech-manufacturing/2026-07-29/regime_evidence.v2.json"
        ),
        "prior_evidence_sha256": SHA,
    }
    assert body["status"] == "AVAILABLE"
    assert body["hard_state"] == "趋势上涨"
    assert sum(float(value) for value in body["state_probabilities"].values()) == pytest.approx(1.0)
    assert body["replay_result"]["status"] == "EXACT_REPLAY_VERIFIED"
    assert body["blocker_codes"] == []
    assert body["default_protocol_state"] == "V15_DEFAULT"
    assert body["global_activation_state"] == "INACTIVE"
    assert body["run_state"] == "INACTIVE"
    assert body["research_runtime_default"] is False
    assert body["factor_governance_write"] is False
    assert body["formal_activation"] is False
    assert body["promotion"] is False
    assert body["execution"] is False
    assert body["broker"] is False
    assert body["order"] is False
    assert body["trade"] is False
    assert body["selector"] is False
    assert body["provider_calls"] is False
    assert body["publication_phase"] == "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION"
    assert body["inference_kind"] == "FILTERED_CAUSAL"
    assert body["smoothing_used"] is False
    assert body["scope_kind"] == "FULL_MARKET"
    assert body["market_sample_count"] == 5502
    assert set(body["authority"]) == {
        "broker",
        "execution",
        "formal_research_publication",
        "order",
        "research_runtime_default",
        "trade",
    }
    assert all(value is False for value in body["authority"].values())
    assert all(value is False for value in body["side_effects"].values())


def test_build_rejects_half_of_optional_prior_pair_before_producer_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    def forbidden_build(**kwargs: Any) -> object:
        del kwargs
        nonlocal called
        called = True
        raise AssertionError("producer must not run")

    monkeypatch.setattr(cli, "build_regime_evidence_v2", forbidden_build)
    argv = _build_argv(tmp_path)
    del argv[-2:]
    argv.extend(["--prior-evidence-path", "data/private/v17_v4_sources/prior.json"])

    assert cli.main(argv) == 2
    body = json.loads(capsys.readouterr().out)
    assert called is False
    assert body["status"] == "BLOCKED"
    assert body["blocker_codes"] == ["PRIOR_EVIDENCE_EXPLICIT_PAIR_REQUIRED"]
    assert all(value is False for value in body["authority"].values())


def test_status_requires_exact_path_and_sha_without_latest_scanning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def forbidden_glob(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("status must not scan")

    def fake_read(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _document()

    monkeypatch.setattr(Path, "glob", forbidden_glob)
    monkeypatch.setattr(Path, "rglob", forbidden_glob)
    monkeypatch.setattr(cli, "read_regime_evidence_v2", fake_read)
    artifact_path = (
        "data/private/v17_v4_sources/regime_evidence/"
        "cn-aggressive-tech-manufacturing/2026-07-30/regime_evidence.v2.json"
    )

    assert (
        cli.main(
            [
                "regime-evidence-status",
                "--workspace-root",
                str(tmp_path),
                "--artifact-path",
                artifact_path,
                "--expected-sha256",
                SHA,
            ]
        )
        == 0
    )
    body = json.loads(capsys.readouterr().out)
    assert captured == {
        "workspace_root": str(tmp_path.resolve()),
        "evidence_path": artifact_path,
        "evidence_sha256": SHA,
    }
    assert body["status"] == "AVAILABLE"
    assert body["evidence_path"] == artifact_path
    assert body["evidence_sha256"] == SHA
    assert body["replay_result"]["hard_state_reclassified"] is False
    assert all(value is False for value in body["authority"].values())

    with pytest.raises(SystemExit) as missing_sha:
        cli.main(
            [
                "regime-evidence-status",
                "--workspace-root",
                str(tmp_path),
                "--artifact-path",
                artifact_path,
            ]
        )
    assert missing_sha.value.code == 2


def test_missing_current_closure_is_gap_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Gap(RuntimeError):
        blocker_code = "TRUE_CURRENT_CANONICAL_INPUT_GAP"

    def missing(**kwargs: Any) -> object:
        del kwargs
        raise Gap("feature snapshot is absent")

    monkeypatch.setattr(cli, "RegimeEvidenceV2InputGap", Gap)
    monkeypatch.setattr(cli, "build_regime_evidence_v2", missing)

    assert cli.main(_build_argv(tmp_path)) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "TRUE_CURRENT_CANONICAL_INPUT_GAP"
    assert body["blocker_codes"] == ["TRUE_CURRENT_CANONICAL_INPUT_GAP"]
    assert body["replay_result"]["status"] == "NOT_AVAILABLE"
    assert all(value is False for value in body["authority"].values())


def test_integrity_failure_is_blocked_exit_two_not_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Blocked(RuntimeError):
        blocker_code = "FEATURE_SNAPSHOT_SHA256_MISMATCH"

    def blocked(**kwargs: Any) -> object:
        del kwargs
        raise Blocked("feature snapshot SHA-256 mismatch")

    monkeypatch.setattr(cli, "RegimeEvidenceV2Error", Blocked)
    monkeypatch.setattr(cli, "build_regime_evidence_v2", blocked)

    assert cli.main(_build_argv(tmp_path)) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "BLOCKED"
    assert body["blocker_codes"] == ["FEATURE_SNAPSHOT_SHA256_MISMATCH"]
    assert body["status"] != "TRUE_CURRENT_CANONICAL_INPUT_GAP"
    assert all(value is False for value in body["side_effects"].values())
