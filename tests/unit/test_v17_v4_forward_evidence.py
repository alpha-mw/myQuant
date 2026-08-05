from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_contract.schema_validation import validate_artifact
from quant_investor.v17_v4_runtime.orchestrator import (
    ForwardEvidenceError,
    StageResult,
    build_forward_request,
    publish_forward_request,
    run_forward,
)
from quant_investor.v17_v4_runtime.source_storage import GovernedStore

CUTOFF = "2026-07-29T07:00:00Z"
SESSION = "2026-07-29"
STRATEGY_ID = "synthetic-core"
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "mainline_authority": False,
    "order": False,
    "production": False,
    "research_only": True,
    "trade": False,
}


def _ref(
    artifact_id: str,
    artifact_version: str,
    relative_path: str,
    *,
    byte_sha256: str = "a" * 64,
    semantic_sha256: str = "b" * 64,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": byte_sha256,
        "cutoff": CUTOFF,
        "relative_path": relative_path,
        "semantic_sha256": semantic_sha256,
        "strategy_id": STRATEGY_ID,
    }


def _request(profile: str = "FORWARD_EVIDENCE") -> dict[str, object]:
    return {
        "authority": dict(NO_AUTHORITY),
        "created_at": CUTOFF,
        "cutoff": CUTOFF,
        "decision_session": SESSION,
        "factor_refs": [
            _ref(
                "factor.synthetic",
                "myquant.v17.v4.synthetic-factor-set.v1",
                "data/private/v17_v4_sources/factors/synthetic.json",
            )
        ],
        "protocol_version": "myquant.v17.v4",
        "request_profile": profile,
        "source_refs": [
            _ref(
                "source.synthetic",
                "myquant.v17.v4.pit-source-manifest.v1",
                "data/private/v17_v4_sources/manifests/synthetic.json",
            )
        ],
        "strategy_id": STRATEGY_ID,
    }


def _callbacks(calls: dict[str, int] | None = None):
    stages = {
        "source": {"source": "synthetic", "pit_valid": True},
        "allocation": {"allocation": "Core"},
        "quant": {"rows": 3},
        "factor_universe_observation": {"observed": True},
        "fusion": {"rows": 2},
        "strategy_pool_observation": {"observed": True},
        "final": {"forward_label": "PENDING"},
    }

    def callback(stage: str):
        def run(_context):
            if calls is not None:
                calls[stage] = calls.get(stage, 0) + 1
            return stages[stage]

        return run

    return {stage: callback(stage) for stage in stages}


def _materialize_request_refs(
    workspace: Path,
    request: dict[str, object],
) -> dict[str, object]:
    for field in ("source_refs", "factor_refs"):
        references = request[field]
        assert isinstance(references, list)
        for reference in references:
            assert isinstance(reference, dict)
            target = workspace / reference["relative_path"]
            if target.exists():
                raw = target.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == reference["byte_sha256"]
                continue
            document = seal_semantic(
                {
                    "bound_artifact_id": reference["artifact_id"],
                    "cutoff": reference["cutoff"],
                    "strategy_id": reference["strategy_id"],
                    "version": reference["artifact_version"],
                }
            )
            raw = canonical_resource_bytes(document)
            GovernedStore(workspace).write_exact_once(
                reference["relative_path"],
                raw,
            )
            reference["byte_sha256"] = hashlib.sha256(raw).hexdigest()
            reference["semantic_sha256"] = document["semantic_sha256"]
    return request


def _publish(workspace: Path, profile: str = "FORWARD_EVIDENCE"):
    return publish_forward_request(
        workspace,
        _materialize_request_refs(workspace, _request(profile)),
    )


def _run(workspace: Path, published, **kwargs):
    return run_forward(
        workspace,
        request_path=published["request_path"],
        request_sha256=published["request_sha256"],
        factor_pointer_reread=lambda: True,
        **kwargs,
    )


def _load(workspace: Path, relative_path: str):
    return json.loads((workspace / relative_path).read_text())


def test_request_defaults_profile_before_content_addressing() -> None:
    body = _request()
    body.pop("request_profile")

    request = build_forward_request(body)
    rebuilt = build_forward_request(request)

    assert request["request_profile"] == "FORWARD_EVIDENCE"
    assert rebuilt["request_id"] == request["request_id"]
    assert rebuilt["semantic_sha256"] == request["semantic_sha256"]


def test_forward_evidence_success_is_exact_and_idempotent(tmp_path: Path):
    published = _publish(tmp_path)
    calls: dict[str, int] = {}

    first = _run(
        tmp_path,
        published,
        stage_callbacks=_callbacks(calls),
    )

    assert first["created"] is True
    assert first["run_state"] == "FORWARD_EVIDENCE_ACTIVE"
    assert first["research_only"] is True
    assert first["mainline_authority"] is False
    assert first["authority"] == NO_AUTHORITY
    assert all(value is False for value in first["side_effects"].values())
    assert first["lifecycle_labels"] == [
        "SOURCE_SNAPSHOT",
        "QUANT_COMPLETE",
        "FUNDAMENTAL_PARTIAL_ALLOWED",
        "FUSION_COMPLETE",
        "DEEP_OPTIONAL",
        "SHADOW_OBSERVATION_CREATED",
        "FORWARD_LABEL_PENDING",
    ]
    assert calls == {
        "allocation": 1,
        "factor_universe_observation": 1,
        "final": 1,
        "fusion": 1,
        "quant": 1,
        "source": 1,
        "strategy_pool_observation": 1,
    }

    second = _run(
        tmp_path,
        published,
        stage_callbacks=_callbacks(calls),
    )

    assert second["created"] is False
    assert second["session_ref"] == first["session_ref"]
    assert calls == {
        "allocation": 1,
        "factor_universe_observation": 1,
        "final": 1,
        "fusion": 1,
        "quant": 1,
        "source": 1,
        "strategy_pool_observation": 1,
    }


def test_crash_after_output_leaves_orphan_and_retry_does_not_reexecute(
    tmp_path: Path,
):
    published = _publish(tmp_path)
    calls: dict[str, int] = {}

    def crash(event, context):
        if event == "after_stage_output" and context.stage == "quant":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        _run(
            tmp_path,
            published,
            stage_callbacks=_callbacks(calls),
            event_hook=crash,
        )

    request_id = published["request_id"]
    root = tmp_path / "results/v17_v4_shadow/forward_evidence/strategies" / "synthetic-core"
    assert (root / f"runs/{request_id}/outputs/quant.json").is_file()
    assert not (root / f"runs/{request_id}/receipts/quant.json").exists()
    assert not (root / f"sessions/2026-07-29/{request_id}.json").exists()

    result = _run(
        tmp_path,
        published,
        stage_callbacks=_callbacks(calls),
    )

    assert result["run_state"] == "FORWARD_EVIDENCE_ACTIVE"
    assert calls["quant"] == 1


def test_absent_optional_stages_are_skipped_unavailable(tmp_path: Path):
    published = _publish(tmp_path)

    result = _run(
        tmp_path,
        published,
        stage_callbacks=_callbacks(),
    )

    request_id = published["request_id"]
    receipt_root = (
        "results/v17_v4_shadow/forward_evidence/strategies/"
        f"synthetic-core/runs/{request_id}/receipts"
    )
    for stage in ("fundamental", "deep", "holdings"):
        receipt = _load(tmp_path, f"{receipt_root}/{stage}.json")
        assert receipt["execution_outcome"] == "SKIPPED"
        assert receipt["completeness"] == "UNAVAILABLE"
        assert receipt["output_refs"] == []
    assert result["session_ref"]["relative_path"].endswith(f"sessions/2026-07-29/{request_id}.json")


def test_provided_invalid_optional_stage_blocks_entire_run(tmp_path: Path):
    published = _publish(tmp_path)
    callbacks = _callbacks()
    callbacks["fundamental"] = lambda _context: StageResult(
        payload={"rows": 1},
        expected_payload_sha256="0" * 64,
    )

    with pytest.raises(ForwardEvidenceError) as caught:
        _run(
            tmp_path,
            published,
            stage_callbacks=callbacks,
        )

    assert caught.value.exit_code == 2
    assert caught.value.run_state == "BLOCKED"
    request_id = published["request_id"]
    session = (
        tmp_path
        / "results/v17_v4_shadow/forward_evidence/strategies"
        / "synthetic-core"
        / "sessions/2026-07-29"
        / f"{request_id}.json"
    )
    assert not session.exists()


def test_tampered_output_blocks_session_replay(tmp_path: Path):
    published = _publish(tmp_path)
    result = _run(
        tmp_path,
        published,
        stage_callbacks=_callbacks(),
    )
    run = _load(tmp_path, result["run_ref"]["relative_path"])
    quant_ref = next(
        ref
        for ref in run["observation_refs"]
        if ref["relative_path"].endswith("/outputs/quant.json")
    )
    quant_path = tmp_path / quant_ref["relative_path"]
    original = quant_path.read_bytes()
    tampered = original.replace(b'\\"rows\\":3', b'\\"rows\\":4')
    assert tampered != original
    quant_path.write_bytes(tampered)

    with pytest.raises(ForwardEvidenceError, match="artifact_readback"):
        _run(
            tmp_path,
            published,
            stage_callbacks=_callbacks(),
        )


def test_explore_succeeds_without_forward_only_stages(tmp_path: Path):
    published = _publish(tmp_path, "EXPLORE")
    callbacks = _callbacks()
    callbacks = {
        stage: callback
        for stage, callback in callbacks.items()
        if stage
        in {
            "source",
            "allocation",
            "quant",
            "factor_universe_observation",
        }
    }

    result = _run(
        tmp_path,
        published,
        stage_callbacks=callbacks,
    )

    assert result["run_state"] == "EXPLORE_COMPLETE"


def _write_stage_input(
    workspace: Path,
    stage: str,
) -> dict[str, str]:
    request_ref = _ref(
        "request.synthetic.input",
        "myquant.v17.v4.forward-run-request.v1",
        "data/private/v17_v4_runs/forward_requests/synthetic.json",
    )
    payload = {"allocation": "Core"} if stage == "allocation" else {"input_stage": stage}
    payload_raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "completeness": "COMPLETE",
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "lineage_receipt_refs": [],
            "output_id": f"input.{stage}",
            "payload_json": payload_raw.decode(),
            "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
            "protocol_version": "myquant.v17.v4",
            "recorded_at": CUTOFF,
            "request_ref": request_ref,
            "stage_id": stage,
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.forward-stage-output.v1",
        }
    )
    raw = canonical_resource_bytes(document)
    path = "data/private/v17_v4_sources/forward_stage_inputs/" f"{stage}.json"
    result = GovernedStore(workspace).write_exact_once(path, raw)
    return _ref(
        str(document["output_id"]),
        str(document["version"]),
        path,
        byte_sha256=result.byte_sha256,
        semantic_sha256=str(document["semantic_sha256"]),
    )


def _request_with_stage_inputs(workspace: Path) -> dict[str, object]:
    stages = (
        "allocation",
        "factor_universe_observation",
        "final",
        "fusion",
        "quant",
        "source",
        "strategy_pool_observation",
    )
    refs = {stage: _write_stage_input(workspace, stage) for stage in stages}
    request = _request()
    request["factor_refs"] = [refs["source"]]
    request["source_refs"] = [refs["source"]]
    request["stage_inputs"] = [
        {
            "artifact_ref": refs[stage],
            "completeness": "COMPLETE",
            "stage_id": stage,
        }
        for stage in stages
    ]
    return request


def test_request_driven_stage_inputs_succeed_without_callbacks(
    tmp_path: Path,
) -> None:
    published = publish_forward_request(
        tmp_path,
        _request_with_stage_inputs(tmp_path),
    )

    result = run_forward(
        tmp_path,
        request_path=published["request_path"],
        request_sha256=published["request_sha256"],
    )

    assert result["run_state"] == "FORWARD_EVIDENCE_ACTIVE"
    run = _load(tmp_path, result["run_ref"]["relative_path"])
    assert run["execution_outcome"] == "SUCCEEDED"
    assert run["completeness"] == "COMPLETE"
    assert run["run_state"] == "FORWARD_EVIDENCE_ACTIVE"
    assert run["research_only"] is True
    assert run["mainline_authority"] is False
    assert run["broker"] is False
    assert run["execution"] is False
    assert run["order"] is False
    assert run["trade"] is False
    session = _load(tmp_path, result["session_ref"]["relative_path"])
    assert session["run_state"] == "FORWARD_EVIDENCE_ACTIVE"
    assert result["session_ref"]["relative_path"].startswith(
        "results/v17_v4_shadow/forward_evidence/" f"strategies/{STRATEGY_ID}/sessions/{SESSION}/"
    )
    paths = {
        published["request_path"],
        result["run_ref"]["relative_path"],
        result["session_ref"]["relative_path"],
        *(ref["relative_path"] for ref in run["observation_refs"]),
        *(ref["relative_path"] for ref in run["stage_receipt_refs"]),
    }
    for path in paths:
        validate_artifact(_load(tmp_path, path))


def test_tampered_request_stage_input_blocks(tmp_path: Path) -> None:
    request = _request_with_stage_inputs(tmp_path)
    published = publish_forward_request(tmp_path, request)
    source_ref = next(
        row["artifact_ref"] for row in request["stage_inputs"] if row["stage_id"] == "source"
    )
    source_path = tmp_path / source_ref["relative_path"]
    original = source_path.read_bytes()
    tampered = original.replace(
        b'\\"input_stage\\":\\"source\\"',
        b'\\"input_stage\\":\\"broken\\"',
    )
    assert tampered != original
    source_path.write_bytes(tampered)

    with pytest.raises(
        ForwardEvidenceError,
        match="stage_input_readback",
    ):
        run_forward(
            tmp_path,
            request_path=published["request_path"],
            request_sha256=published["request_sha256"],
        )


def test_request_stage_input_cannot_be_bypassed_by_callback(
    tmp_path: Path,
) -> None:
    published = publish_forward_request(
        tmp_path,
        _request_with_stage_inputs(tmp_path),
    )

    with pytest.raises(
        ForwardEvidenceError,
        match="stage_input_callback_conflict",
    ):
        run_forward(
            tmp_path,
            request_path=published["request_path"],
            request_sha256=published["request_sha256"],
            stage_callbacks=_callbacks(),
        )


def test_request_source_ref_is_exact_read_before_callbacks(
    tmp_path: Path,
) -> None:
    request = _materialize_request_refs(tmp_path, _request())
    published = publish_forward_request(tmp_path, request)
    source_refs = request["source_refs"]
    assert isinstance(source_refs, list)
    source_ref = source_refs[0]
    assert isinstance(source_ref, dict)
    source_path = tmp_path / source_ref["relative_path"]
    source_path.write_bytes(
        source_path.read_bytes().replace(
            b"source.synthetic",
            b"source.tampered",
        )
    )

    with pytest.raises(ForwardEvidenceError, match="request_ref_readback"):
        run_forward(
            tmp_path,
            request_path=published["request_path"],
            request_sha256=published["request_sha256"],
            stage_callbacks=_callbacks(),
            factor_pointer_reread=lambda: True,
        )
