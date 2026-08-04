from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

import quant_investor.v17_v4_runtime.cli as cli
import quant_investor.v17_v4_runtime.regime_evidence_v2 as regime_v2

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


def test_public_build_is_blocked_without_writes_or_producer_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = {"v2": False, "v3": False}

    def forbidden_v2(**kwargs: Any) -> object:
        del kwargs
        called["v2"] = True
        raise AssertionError("V2 producer must not run")

    def forbidden_v3(**kwargs: Any) -> object:
        del kwargs
        called["v3"] = True
        raise AssertionError("V3 producer must not run")

    def forbidden_scan(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("blocked V2 command must not scan")

    monkeypatch.setattr(regime_v2, "build_regime_evidence_v2", forbidden_v2)
    monkeypatch.setattr(cli, "build_regime_evidence_v3", forbidden_v3)
    monkeypatch.setattr(Path, "glob", forbidden_scan)
    monkeypatch.setattr(Path, "rglob", forbidden_scan)

    assert cli.main(_build_argv(tmp_path)) == 2
    body = json.loads(capsys.readouterr().out)

    assert called == {"v2": False, "v3": False}
    assert list(tmp_path.iterdir()) == []
    assert body["status"] == "REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE"
    assert body["blocker_codes"] == ["REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE"]
    assert body["requested_version"] == "myquant.v17.v4.regime-evidence.v2"
    assert body["deployment_status"] == "CONTRACT_VALIDATED_NOT_DEPLOYABLE"
    assert body["replacement_command"] == "regime-evidence-v3-build"
    assert body["artifact_created"] is False
    assert body["replay_result"]["status"] == "NOT_AVAILABLE"
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


def test_publication_block_precedes_optional_pair_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = _build_argv(tmp_path)
    del argv[-2:]
    argv.extend(["--prior-evidence-path", "data/private/v17_v4_sources/prior.json"])

    assert cli.main(argv) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE"
    assert body["artifact_created"] is False
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


def test_frozen_v1_v2_contract_policy_and_producer_bytes_are_unchanged() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "quant_investor/v17_v4_contract/schemas/regime_evidence.v1.schema.json": (
            "49d006413465d7304c621f74f9732cc0d5636c400989d25b170ee4337ff229a3"
        ),
        "quant_investor/v17_v4_contract/schemas/regime_evidence.v2.schema.json": (
            "1d2b624d63808038240d29cf27a48b62d1f3d3da32b757be8bd196916f22de8c"
        ),
        "quant_investor/v17_v4_contract/resources/regime_inference_policy.v1.json": (
            "006773e24f47f0b7f28d6f7707ff6f570066cb212bd83ebd9566512fda7734ef"
        ),
        "quant_investor/v17_v4_runtime/regime_evidence_v2.py": (
            "4e90e06eb340438e909b842a4f40e1dec7eb5ff3231e02c087499bda7646cc7a"
        ),
    }
    observed = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in expected
    }
    assert observed == expected
