from __future__ import annotations

import hashlib
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import quant_investor.v17_v4_runtime.eligibility_control as module
from quant_investor.v17_v4_contract import validate_artifact
from quant_investor.v17_v4_runtime.eligibility_control import (
    EligibilityCrash,
    EligibilityService,
    build_rollback_drill_receipt,
    build_validation_receipt,
)
from quant_investor.v17_v4_runtime.source_storage import EMPTY_SHA256

STRATEGY = "quant-first"
CUTOFF = "2026-07-27T08:00:00Z"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ref(
    artifact_id: str,
    version: str,
    path: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": _sha(f"bytes:{artifact_id}"),
        "cutoff": CUTOFF,
        "relative_path": path,
        "semantic_sha256": _sha(f"semantic:{artifact_id}"),
        "strategy_id": STRATEGY,
    }


def _service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> EligibilityService:
    tmp_path.chmod(0o700)
    service = EligibilityService(tmp_path, repo_root=tmp_path)
    formal_pointer_ref = _ref(
        "formal-pointer-1",
        "myquant.v17.v4.formal-active-pointer.v1",
        (
            "results/v17_v4_formal_research/strategies/"
            f"{STRATEGY}/_active.json"
        ),
    )
    monkeypatch.setattr(
        module,
        "resolve_public_run",
        lambda *_args, **_kwargs: {
            "cutoff": CUTOFF,
            "formal_active_pointer_ref": formal_pointer_ref,
        },
    )
    monkeypatch.setattr(service, "_revalidate_intent", lambda _intent: None)
    monkeypatch.setattr(
        module,
        "validate_artifact",
        lambda document: SimpleNamespace(version=document["version"]),
    )

    def store_evidence(
        strategy_id: str,
        folder: str,
        document: Mapping[str, Any],
    ) -> dict[str, str]:
        version = str(document["version"])
        identity = str(document["receipt_id"])
        return _ref(
            identity,
            version,
            (
                "results/v17_v4_formal_research/strategies/"
                f"{strategy_id}/eligibility/{folder}/{identity}.json"
            ),
        )

    monkeypatch.setattr(service, "_store_evidence", store_evidence)
    return service


def _documents() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
]:
    public = [
        {
            "cutoff": CUTOFF,
            "receipt_id": f"public-{index}",
            "strategy_id": STRATEGY,
            "version": (
                "myquant.v17.v4."
                "public-surface-compatibility-receipt.v1"
            ),
        }
        for index in range(4)
    ]
    validations = [
        {
            "cutoff": CUTOFF,
            "receipt_id": f"validation-{index}",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.validation-receipt.v1",
        }
        for index in range(5)
    ]
    rollback = {
        "cutoff": CUTOFF,
        "receipt_id": "rollback-1",
        "strategy_id": STRATEGY,
        "version": "myquant.v17.v4.rollback-drill-receipt.v1",
    }
    return public, validations, rollback


def _bootstrap_ref() -> dict[str, str]:
    return _ref(
        "bootstrap-1",
        "myquant.research-runtime.route-bootstrap-receipt.v1",
        (
            "results/research_runtime_control/bootstrap_receipts/"
            "bootstrap-1.json"
        ),
    )


def _qualify(
    service: EligibilityService,
    *,
    crash_after: str | None = None,
):
    public, validations, rollback = _documents()
    return service.qualify(
        intent_id="eligibility-1",
        strategy_id=STRATEGY,
        created_at=CUTOFF,
        expected_pointer_sha256=EMPTY_SHA256,
        selector_bootstrap_receipt_ref=_bootstrap_ref(),
        public_surface_receipts=public,
        validation_receipts=validations,
        rollback_drill_receipt=rollback,
        crash_after=crash_after,
    )


def test_eligibility_is_intent_pointer_completion_and_not_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    result = _qualify(service)
    state = service.resolve(STRATEGY)

    assert result.status == state.status == "DEFAULT_ELIGIBLE"
    assert state.pointer is not None
    assert state.pointer["state"] == "PENDING_COMPLETION"
    assert state.pointer["authority"]["research_runtime_default"] is False
    assert state.completion is not None
    assert state.completion["status"] == "DEFAULT_ELIGIBLE"
    assert state.completion["authority"]["formal_research_publication"] is True
    assert not (tmp_path / "results/research_runtime_control").exists()

    root = (
        tmp_path
        / "results/v17_v4_formal_research/strategies"
        / STRATEGY
        / "eligibility"
    )
    assert sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    ) == [
        ".active.lock",
        "_active.json",
        "completion_receipts/eligibility-1.json",
        "intents/eligibility-1.json",
    ]
    for path in root.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected


@pytest.mark.parametrize("boundary", ["intent", "cas", "readback", "completion"])
def test_eligibility_recovers_every_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    service = _service(tmp_path, monkeypatch)
    with pytest.raises(EligibilityCrash):
        _qualify(service, crash_after=boundary)

    state = service.resolve(STRATEGY)
    if boundary == "intent":
        assert state.status == "FORMAL_ACTIVE"
    elif boundary in {"cas", "readback"}:
        assert state.status == "PENDING_COMPLETION"
    else:
        assert state.status == "DEFAULT_ELIGIBLE"

    recovered = _qualify(service)
    assert recovered.status == "DEFAULT_ELIGIBLE"
    assert recovered.recovered is (boundary != "intent")


def test_validation_and_isolated_rollback_receipts_are_closed() -> None:
    validation = build_validation_receipt(
        receipt_id="validation-1",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        recorded_at=CUTOFF,
        validation_kind="V4_FULL_TESTS",
        command_id="pytest-v4",
        command_sha256=_sha("pytest-v4"),
        result_sha256=_sha("result-v4"),
        passed_count=189,
    )
    rollback = build_rollback_drill_receipt(
        receipt_id="rollback-1",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        recorded_at=CUTOFF,
        isolated_control_root_digest=_sha("isolated-root"),
        bootstrap_receipt_sha256=_sha("bootstrap"),
        cutover_receipt_sha256=_sha("cutover"),
        rollback_receipt_sha256=_sha("rollback"),
        final_selector_sha256=_sha("selector"),
    )

    assert validate_artifact(validation).version == validation["version"]
    assert validate_artifact(rollback).version == rollback["version"]
    assert rollback["production_selector_writes"] is False
