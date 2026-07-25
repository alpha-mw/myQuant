from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant_investor.v17 import runtime
from quant_investor.v17.semantic import canonical_json_bytes, seal_semantic
from quant_investor.v17.state_machine import (
    EMPTY_SHA,
    V17PostCommitReadbackError,
    advance_run_state,
    initialize_run,
    load_run_ledger,
)
from quant_investor.v17.storage import file_sha256, read_json

TIMES = {
    "prepared_at": "2026-07-22T07:00:00Z",
    "deterministic_at": "2026-07-22T07:01:00Z",
    "deep_request_at": "2026-07-22T07:02:00Z",
    "response_at": "2026-07-22T07:03:00Z",
    "generated_at": "2026-07-22T07:04:00Z",
    "finalized_at": "2026-07-22T07:05:00Z",
    "failed_at": "2026-07-22T07:06:00Z",
}
TEST_IMPLEMENTATION_SHA256S = {
    relative: "9" * 64 for relative in runtime.IMPLEMENTATION_BINDING_RELATIVE_PATHS
}


@pytest.fixture(autouse=True)
def _stable_implementation_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "_compute_implementation_sha256s",
        lambda _root: dict(TEST_IMPLEMENTATION_SHA256S),
    )


def _sealed(version: str, **values: Any) -> dict[str, Any]:
    return seal_semantic(
        {
            "version": version,
            **values,
            "authority": False,
        }
    )


def _prepare_request(run_id: str) -> dict[str, Any]:
    return seal_semantic(
        {
            "version": runtime.PREPARE_REQUEST_VERSION,
            "run_id": run_id,
            "strategy_id": "cn-shadow",
            "market": "CN",
            "cutoff": TIMES["prepared_at"],
            "source_manifest_path": ("data/private/v17_sources/manifests/synthetic.json"),
            "source_manifest_sha256": "a" * 64,
            "resource_sha256s": dict(runtime.FROZEN_POLICY_RESOURCE_SHA256S),
            "schema_sha256s": dict(runtime.FROZEN_SCHEMA_SHA256S),
            "transition_times": {
                "prepared_at": TIMES["prepared_at"],
                "deterministic_at": TIMES["deterministic_at"],
                "deep_request_at": TIMES["deep_request_at"],
            },
            "authority": False,
        }
    )


def _prepare_artifacts(run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    deterministic = _sealed(
        "test.deterministic.v1",
        run_id=run_id,
        cutoff=TIMES["prepared_at"],
        sealed_symbols=["000001.SZ"],
    )
    deep_request = _sealed(
        "test.deep-request.v1",
        run_id=run_id,
        cutoff=TIMES["prepared_at"],
        symbols=["000001.SZ"],
    )
    return deterministic, deep_request


def _patch_prepare_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    run_id: str,
) -> None:
    bundle = SimpleNamespace(
        manifest_path=(repo / "data/private/v17_sources/manifests/synthetic.json"),
        manifest_byte_sha256="a" * 64,
        effective_availability_by_role={"market_snapshot": "AVAILABLE"},
        rank_unavailable_roles=(),
        portfolio_unavailable_roles=(),
    )
    deterministic, deep_request = _prepare_artifacts(run_id)
    monkeypatch.setattr(
        runtime,
        "_validate_package_bindings",
        lambda *_args, **_kwargs: {
            "resources": dict(runtime.FROZEN_POLICY_RESOURCE_SHA256S),
            "schemas": dict(runtime.FROZEN_SCHEMA_SHA256S),
        },
    )
    monkeypatch.setattr(
        runtime,
        "load_source_manifest_binding",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        runtime,
        "compute_prepare_artifacts",
        lambda *_args, **_kwargs: (deterministic, deep_request),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_deterministic_result",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_deep_request",
        lambda payload, **_kwargs: dict(payload),
    )


@pytest.mark.parametrize("crash_before", ["DETERMINISTIC_COMPLETE", "DEEP_REQUEST_READY"])
def test_prepare_resumes_each_partial_state_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_before: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = f"prepare-{crash_before.lower()}"
    request = _prepare_request(run_id)
    _patch_prepare_dependencies(monkeypatch, repo, run_id)
    request_sha = "b" * 64
    real_advance = advance_run_state
    crashed = False

    def flaky_advance(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal crashed
        if kwargs["next_state"] == crash_before and not crashed:
            crashed = True
            raise RuntimeError("synthetic crash")
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(runtime, "advance_run_state", flaky_advance)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        runtime.prepare_shadow_run(
            repo,
            request,
            request_byte_sha256=request_sha,
            expected_ledger_sha256=EMPTY_SHA,
        )

    partial, partial_sha = load_run_ledger(repo, run_id)
    expected_state = (
        "PREPARED" if crash_before == "DETERMINISTIC_COMPLETE" else "DETERMINISTIC_COMPLETE"
    )
    assert partial["state"] == expected_state

    resumed = runtime.prepare_shadow_run(
        repo,
        request,
        request_byte_sha256=request_sha,
        expected_ledger_sha256=partial_sha,
    )
    assert resumed["state"] == "DEEP_REQUEST_READY"
    before_retry = (repo / f"results/v17_shadow/runs/{run_id}/ledger.json").read_bytes()
    repeated = runtime.prepare_shadow_run(
        repo,
        request,
        request_byte_sha256=request_sha,
        expected_ledger_sha256=resumed["ledger_sha256"],
    )
    assert repeated["ledger_sha256"] == resumed["ledger_sha256"]
    assert (repo / f"results/v17_shadow/runs/{run_id}/ledger.json").read_bytes() == before_retry


def test_prepare_retry_rejects_input_and_artifact_rebinding_with_zero_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = "prepare-binding"
    request = _prepare_request(run_id)
    _patch_prepare_dependencies(monkeypatch, repo, run_id)
    prepared = runtime.prepare_shadow_run(
        repo,
        request,
        request_byte_sha256="b" * 64,
        expected_ledger_sha256=EMPTY_SHA,
    )
    ledger_path = repo / f"results/v17_shadow/runs/{run_id}/ledger.json"
    before = ledger_path.read_bytes()

    with pytest.raises(runtime.V17RuntimeInvalidEvidence, match="input binding"):
        runtime.prepare_shadow_run(
            repo,
            request,
            request_byte_sha256="c" * 64,
            expected_ledger_sha256=prepared["ledger_sha256"],
        )
    assert ledger_path.read_bytes() == before

    ledger, _ = load_run_ledger(repo, run_id)
    artifact_path = repo / ledger["artifacts"]["deterministic_result"]["relative_path"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    after_tamper = ledger_path.read_bytes()
    with pytest.raises(runtime.V17RuntimeSnapshotDrift, match="artifact"):
        runtime.prepare_shadow_run(
            repo,
            request,
            request_byte_sha256="b" * 64,
            expected_ledger_sha256=prepared["ledger_sha256"],
        )
    assert ledger_path.read_bytes() == after_tamper


def _deep_response_ledger(repo: Path, run_id: str) -> tuple[dict[str, Any], str]:
    ledger, ledger_sha = initialize_run(
        repo,
        run_id=run_id,
        strategy_id="cn-shadow",
        cutoff=TIMES["prepared_at"],
        prepared_at=TIMES["prepared_at"],
        input_bindings={
            "source_manifest_path": ("data/private/v17_sources/manifests/synthetic.json"),
            "source_manifest_sha256": "a" * 64,
            "resource_sha256s": dict(runtime.FROZEN_POLICY_RESOURCE_SHA256S),
            "schema_sha256s": dict(runtime.FROZEN_SCHEMA_SHA256S),
            "implementation_sha256s": dict(TEST_IMPLEMENTATION_SHA256S),
        },
        expected_ledger_sha256=EMPTY_SHA,
    )
    deterministic = _sealed(
        "test.deterministic.v1",
        run_id=run_id,
        cutoff=TIMES["prepared_at"],
        sealed_symbols=["000001.SZ"],
    )
    deep_request = _sealed(
        "test.deep-request.v1",
        run_id=run_id,
        cutoff=TIMES["prepared_at"],
        symbols=["000001.SZ"],
    )
    for state, timestamp, artifacts in (
        (
            "DETERMINISTIC_COMPLETE",
            TIMES["deterministic_at"],
            {"deterministic_result": deterministic},
        ),
        (
            "DEEP_REQUEST_READY",
            TIMES["deep_request_at"],
            {"deep_request": deep_request},
        ),
        (
            "DEEP_RESPONSE_RECEIVED",
            TIMES["response_at"],
            {
                "deep_evaluation": _sealed(
                    "test.deep-evaluation.v1",
                    run_id=run_id,
                    cutoff=TIMES["prepared_at"],
                )
            },
        ),
    ):
        ledger, ledger_sha = advance_run_state(
            repo,
            run_id=run_id,
            expected_ledger_sha256=ledger_sha,
            next_state=state,
            transitioned_at=timestamp,
            artifacts=artifacts,
        )
    return ledger, ledger_sha


def _deep_request_ledger(repo: Path, run_id: str) -> tuple[dict[str, Any], str]:
    ledger, ledger_sha = initialize_run(
        repo,
        run_id=run_id,
        strategy_id="cn-shadow",
        cutoff=TIMES["prepared_at"],
        prepared_at=TIMES["prepared_at"],
        input_bindings={
            "source_manifest_path": ("data/private/v17_sources/manifests/synthetic.json"),
            "source_manifest_sha256": "a" * 64,
            "resource_sha256s": dict(runtime.FROZEN_POLICY_RESOURCE_SHA256S),
            "schema_sha256s": dict(runtime.FROZEN_SCHEMA_SHA256S),
            "implementation_sha256s": dict(TEST_IMPLEMENTATION_SHA256S),
        },
        expected_ledger_sha256=EMPTY_SHA,
    )
    for state, timestamp, artifacts in (
        (
            "DETERMINISTIC_COMPLETE",
            TIMES["deterministic_at"],
            {
                "deterministic_result": _sealed(
                    "test.deterministic.v1",
                    run_id=run_id,
                    cutoff=TIMES["prepared_at"],
                    sealed_symbols=["000001.SZ"],
                )
            },
        ),
        (
            "DEEP_REQUEST_READY",
            TIMES["deep_request_at"],
            {
                "deep_request": _sealed(
                    "test.deep-request.v1",
                    run_id=run_id,
                    cutoff=TIMES["prepared_at"],
                    symbols=["000001.SZ"],
                )
            },
        ),
    ):
        ledger, ledger_sha = advance_run_state(
            repo,
            run_id=run_id,
            expected_ledger_sha256=ledger_sha,
            next_state=state,
            transitioned_at=timestamp,
            artifacts=artifacts,
        )
    return ledger, ledger_sha


def _patch_bound_prepare_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "_recompute_prepare_artifacts_from_sources",
        lambda root, ledger, bundle: (
            runtime._artifact_payload(root, ledger, "deterministic_result"),
            runtime._artifact_payload(root, ledger, "deep_request"),
        ),
    )


def test_deep_response_cannot_predate_request_ready(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = "response-predates-request"
    ledger, _ = _deep_request_ledger(repo, run_id)
    deep_request = runtime._artifact_payload(repo, ledger, "deep_request")
    response = seal_semantic(
        {
            "version": runtime.DEEP_RESPONSE_VERSION,
            "run_id": run_id,
            "cutoff": TIMES["prepared_at"],
            "review_results": [
                {
                    "symbol": "000001.SZ",
                    "status": "UNAVAILABLE",
                    "reason": "sealed evidence unavailable",
                }
            ],
            "generated_at": TIMES["deterministic_at"],
            "received_at": TIMES["response_at"],
            "authority": False,
        }
    )

    with pytest.raises(
        runtime.V17RuntimeError,
        match="generated_at precedes DEEP_REQUEST_READY",
    ):
        runtime._validate_deep_response(
            response,
            ledger=ledger,
            deep_request=deep_request,
        )


def test_receive_replays_exact_artifacts_after_postcommit_readback_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = "receive-resume"
    _, ledger_sha = _deep_request_ledger(repo, run_id)
    response = _sealed(
        "test.deep-response.v1",
        run_id=run_id,
        cutoff=TIMES["prepared_at"],
        received_at=TIMES["response_at"],
    )
    evaluation = _sealed(
        "test.deep-evaluation.v1",
        run_id=run_id,
        cutoff=TIMES["prepared_at"],
    )
    monkeypatch.setattr(
        runtime,
        "_validate_deterministic_result",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(runtime, "_revalidate_sources", lambda *_args: object())
    _patch_bound_prepare_recompute(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "_validate_deep_request",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_deep_response",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        runtime,
        "evaluate_deep_response",
        lambda *_args, **_kwargs: evaluation,
    )
    monkeypatch.setattr(
        runtime,
        "_validate_deep_evaluation",
        lambda payload, **_kwargs: dict(payload),
    )
    real_advance = advance_run_state

    def uncertain_advance(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        real_advance(*args, **kwargs)
        raise V17PostCommitReadbackError("synthetic postcommit uncertainty")

    monkeypatch.setattr(runtime, "advance_run_state", uncertain_advance)
    with pytest.raises(V17PostCommitReadbackError, match="uncertainty"):
        runtime.receive_shadow_response(
            repo,
            run_id=run_id,
            response=response,
            response_byte_sha256="d" * 64,
            expected_ledger_sha256=ledger_sha,
        )
    committed, committed_sha = load_run_ledger(repo, run_id)
    assert committed["state"] == "DEEP_RESPONSE_RECEIVED"

    monkeypatch.setattr(runtime, "advance_run_state", real_advance)
    before = (repo / f"results/v17_shadow/runs/{run_id}/ledger.json").read_bytes()
    replay = runtime.receive_shadow_response(
        repo,
        run_id=run_id,
        response=response,
        response_byte_sha256="d" * 64,
        expected_ledger_sha256=committed_sha,
    )
    assert replay["ledger_sha256"] == committed_sha
    assert (repo / f"results/v17_shadow/runs/{run_id}/ledger.json").read_bytes() == before

    with pytest.raises(runtime.V17RuntimeSnapshotDrift, match="deep_response_import"):
        runtime.receive_shadow_response(
            repo,
            run_id=run_id,
            response=response,
            response_byte_sha256="e" * 64,
            expected_ledger_sha256=committed_sha,
        )


def test_receive_revalidates_frozen_package_bindings_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = "receive-package-drift"
    ledger, ledger_sha = _deep_request_ledger(repo, run_id)
    ledger["input_bindings"]["resource_sha256s"] = {
        **runtime.FROZEN_POLICY_RESOURCE_SHA256S,
        next(iter(runtime.FROZEN_POLICY_RESOURCE_SHA256S)): "f" * 64,
    }
    with pytest.raises(runtime.V17RuntimeSnapshotDrift, match="package"):
        runtime._revalidate_ledger_package_bindings(ledger)
    assert load_run_ledger(repo, run_id)[1] == ledger_sha


@pytest.mark.parametrize("artifact_role", ["deterministic_result", "deep_request"])
def test_receive_file_commits_snapshot_terminal_without_trusting_drifted_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_role: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = f"receive-{artifact_role}-drift"
    ledger, ledger_sha = _deep_request_ledger(repo, run_id)
    artifact_path = repo / ledger["artifacts"][artifact_role]["relative_path"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    response_path = tmp_path / "response.json"
    response_sha = _write_canonical(
        response_path,
        _sealed("test.response.v1", value="unreached"),
    )
    ledger_path = repo / f"results/v17_shadow/runs/{run_id}/ledger.json"
    before = ledger_path.read_bytes()
    monkeypatch.setattr(runtime, "_revalidate_sources", lambda *_args: object())
    _patch_bound_prepare_recompute(monkeypatch)

    result = runtime.receive_shadow_response_from_file(
        repo,
        run_id=run_id,
        response_path=response_path,
        expected_response_sha256=response_sha,
        expected_ledger_sha256=ledger_sha,
        expected_latest_sha256=EMPTY_SHA,
        failed_at=TIMES["failed_at"],
    )
    assert result["state"] == "HARD_STOP_SNAPSHOT_DRIFT"
    terminal, _ = load_run_ledger(repo, run_id, verify_artifacts=False)
    output = read_json(repo / terminal["artifacts"]["terminal_output"]["relative_path"])
    assert output["blockers"] == [
        (
            f"artifact_drift:role={artifact_role}:expected="
            f"{ledger['artifacts'][artifact_role]['byte_sha256']}:observed="
            f"{file_sha256(artifact_path)}"
        )
    ]
    assert ledger_path.read_bytes() != before
    assert (repo / "results/v17_shadow/_latest/shadow.json").exists()
    assert runtime.shadow_status(repo, run_id)["state"] == "HARD_STOP_SNAPSHOT_DRIFT"


@pytest.mark.parametrize("fault", ["implementation", "package", "source"])
def test_receive_implementation_package_or_source_drift_commits_snapshot_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = f"receive-{fault}-drift"
    _, ledger_sha = _deep_request_ledger(repo, run_id)
    response_path = tmp_path / f"{fault}-response.json"
    response_sha = _write_canonical(
        response_path,
        _sealed("test.response.v1", value="unreached"),
    )
    if fault == "implementation":
        changed = dict(TEST_IMPLEMENTATION_SHA256S)
        changed[next(iter(changed))] = "8" * 64
        monkeypatch.setattr(
            runtime,
            "_compute_implementation_sha256s",
            lambda _root: changed,
        )
    elif fault == "package":
        monkeypatch.setattr(
            runtime,
            "_validate_package_bindings",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("tampered frozen bytes")),
        )
    else:
        monkeypatch.setattr(
            runtime,
            "_revalidate_sources",
            lambda *_args: (_ for _ in ()).throw(
                runtime.V17RuntimeSnapshotDrift("source object identity changed")
            ),
        )

    result = runtime.receive_shadow_response_from_file(
        repo,
        run_id=run_id,
        response_path=response_path,
        expected_response_sha256=response_sha,
        expected_ledger_sha256=ledger_sha,
        expected_latest_sha256=EMPTY_SHA,
        failed_at=TIMES["failed_at"],
    )
    assert result["state"] == "HARD_STOP_SNAPSHOT_DRIFT"


@pytest.mark.parametrize(
    ("evaluation_status", "expected_state"),
    [
        ("DEEP_RESEARCH_INVALID", "HARD_STOP_INVALID_EVIDENCE"),
        ("DEEP_RESEARCH_UNAVAILABLE", "DEEP_RESPONSE_RECEIVED"),
    ],
)
def test_invalid_deep_evaluation_hard_stops_but_explicit_unavailable_is_normal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_status: str,
    expected_state: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = f"deep-{evaluation_status.lower()}"
    _, ledger_sha = _deep_request_ledger(repo, run_id)
    response = _sealed(
        "test.deep-response.v1",
        run_id=run_id,
        cutoff=TIMES["prepared_at"],
        received_at=TIMES["response_at"],
    )
    response_path = tmp_path / f"{run_id}.json"
    response_sha = _write_canonical(response_path, response)
    evaluation = seal_semantic(
        {
            "version": runtime.DEEP_EVALUATION_VERSION,
            "run_id": run_id,
            "cutoff": TIMES["prepared_at"],
            "evaluations": [{"symbol": "000001.SZ", "status": evaluation_status}],
            "received_at": TIMES["response_at"],
            "authority": False,
        }
    )
    monkeypatch.setattr(runtime, "_revalidate_sources", lambda *_args: object())
    _patch_bound_prepare_recompute(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "_validate_deterministic_result",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_deep_request",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_deep_response",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        runtime,
        "evaluate_deep_response",
        lambda *_args, **_kwargs: evaluation,
    )

    result = runtime.receive_shadow_response_from_file(
        repo,
        run_id=run_id,
        response_path=response_path,
        expected_response_sha256=response_sha,
        expected_ledger_sha256=ledger_sha,
        expected_latest_sha256=EMPTY_SHA,
        failed_at=TIMES["failed_at"],
    )
    assert result["state"] == expected_state


def _finalization(run_id: str, *, candidate_id: str = "candidate-a") -> dict[str, Any]:
    return seal_semantic(
        {
            "version": runtime.FINALIZATION_VERSION,
            "run_id": run_id,
            "cutoff": TIMES["prepared_at"],
            "candidate_proposals": [
                {
                    "candidate_id": candidate_id,
                    "target_weights": {"000001.SZ": 0.1},
                }
            ],
            "generated_at": TIMES["generated_at"],
            "finalized_at": TIMES["finalized_at"],
            "authority": False,
        }
    )


def _patch_finalize_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_bound_prepare_recompute(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "_validate_deterministic_result",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_deep_evaluation",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(runtime, "_revalidate_sources", lambda *_args: object())
    monkeypatch.setattr(
        runtime,
        "_recompute_stored_deep_evaluation",
        lambda root, ledger, **_kwargs: runtime._artifact_payload(root, ledger, "deep_evaluation"),
    )
    monkeypatch.setattr(
        runtime,
        "compute_finalization",
        lambda *_args, **_kwargs: (
            "COMPLETE",
            {"ranked_symbols": ["000001.SZ"]},
            {"weights": {"000001.SZ": 0.1}},
            [],
            _sealed("test.portfolio-computation.v1", selected_candidate_id="candidate-a"),
        ),
    )


def _write_canonical(path: Path, payload: dict[str, Any]) -> str:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return file_sha256(path)


def test_finalize_file_recovers_after_portfolio_commit_and_binds_terminal_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = "finalize-resume"
    _, ledger_sha = _deep_response_ledger(repo, run_id)
    _patch_finalize_dependencies(monkeypatch)
    finalization = _finalization(run_id)
    path = tmp_path / "finalization.json"
    finalization_sha = _write_canonical(path, finalization)
    real_advance = advance_run_state
    failed_terminal = False

    def flaky_advance(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal failed_terminal
        if (
            kwargs["next_state"] == "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION"
            and not failed_terminal
        ):
            failed_terminal = True
            raise RuntimeError("synthetic second-step crash")
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(runtime, "advance_run_state", flaky_advance)
    result = runtime.finalize_shadow_run_from_file(
        repo,
        run_id=run_id,
        finalization_path=path,
        expected_finalization_sha256=finalization_sha,
        expected_ledger_sha256=ledger_sha,
        expected_latest_sha256=EMPTY_SHA,
        failed_at=TIMES["failed_at"],
    )
    assert result["state"] == "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION"
    ledger, _ = load_run_ledger(repo, run_id)
    assert [item["to_state"] for item in ledger["history"]][-2:] == [
        "PORTFOLIO_COMPLETE",
        "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
    ]
    output = read_json(repo / ledger["artifacts"]["terminal_output"]["relative_path"])
    assert output["source_manifest_sha256"] == ledger["input_bindings"]["source_manifest_sha256"]


def test_portfolio_resume_requires_exact_finalization_artifact_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = "finalize-binding"
    _, ledger_sha = _deep_response_ledger(repo, run_id)
    _patch_finalize_dependencies(monkeypatch)
    finalization = _finalization(run_id)
    real_advance = advance_run_state

    def stop_before_terminal(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        if kwargs["next_state"] == "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION":
            raise RuntimeError("stop before terminal")
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(runtime, "advance_run_state", stop_before_terminal)
    with pytest.raises(RuntimeError, match="stop before terminal"):
        runtime.finalize_shadow_run(
            repo,
            run_id=run_id,
            finalization=finalization,
            finalization_byte_sha256="d" * 64,
            expected_ledger_sha256=ledger_sha,
            expected_latest_sha256=EMPTY_SHA,
        )
    portfolio_ledger, portfolio_sha = load_run_ledger(repo, run_id)
    assert portfolio_ledger["state"] == "PORTFOLIO_COMPLETE"
    before = (repo / f"results/v17_shadow/runs/{run_id}/ledger.json").read_bytes()
    monkeypatch.setattr(runtime, "advance_run_state", real_advance)

    with pytest.raises(runtime.V17RuntimeSnapshotDrift, match="finalization"):
        runtime.finalize_shadow_run(
            repo,
            run_id=run_id,
            finalization=_finalization(run_id, candidate_id="candidate-b"),
            finalization_byte_sha256="e" * 64,
            expected_ledger_sha256=portfolio_sha,
            expected_latest_sha256=EMPTY_SHA,
        )
    assert (repo / f"results/v17_shadow/runs/{run_id}/ledger.json").read_bytes() == before


@pytest.mark.parametrize(
    ("failure", "expected_state"),
    [
        (
            runtime.V17RuntimeSnapshotDrift("typed drift without keyword heuristics"),
            "HARD_STOP_SNAPSHOT_DRIFT",
        ),
        (
            runtime.V17RuntimeInvalidEvidence("malformed caller payload"),
            "HARD_STOP_INVALID_EVIDENCE",
        ),
    ],
)
def test_finalize_failure_classification_is_typed_not_message_based(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_state: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_id = f"typed-{expected_state.lower()}"
    _, ledger_sha = _deep_response_ledger(repo, run_id)
    finalization = _finalization(run_id)
    path = tmp_path / f"{run_id}.json"
    finalization_sha = _write_canonical(path, finalization)
    monkeypatch.setattr(
        runtime, "finalize_shadow_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(failure)
    )

    result = runtime.finalize_shadow_run_from_file(
        repo,
        run_id=run_id,
        finalization_path=path,
        expected_finalization_sha256=finalization_sha,
        expected_ledger_sha256=ledger_sha,
        expected_latest_sha256=EMPTY_SHA,
        failed_at=TIMES["failed_at"],
    )
    assert result["state"] == expected_state


def test_terminal_source_binding_validator_rejects_mismatch() -> None:
    ledger = {"input_bindings": {"source_manifest_sha256": "a" * 64}}
    with pytest.raises(runtime.V17RuntimeSnapshotDrift, match="does not match"):
        runtime._validate_terminal_source_binding(
            {"source_manifest_sha256": "b" * 64},
            ledger=ledger,
        )
