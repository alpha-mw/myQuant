from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.research.label_maturity import assess_label_maturity
from quant_investor.research.research_scheduler import run_daily_research_loop
from quant_investor.v17_v4_contract.canonical import canonical_resource_bytes
from quant_investor.v17_v4_runtime.cli_provisional import main
from quant_investor.v17_v4_runtime.provisional_forward import (
    ProvisionalForwardError,
    build_provisional_request,
)

CUTOFF = "2026-08-04T07:00:00Z"
SESSION = "2026-08-04"
STRATEGY = "v17-sanitized-loop"
SHA = "a" * 64


def _ref(name: str) -> dict[str, str]:
    return {
        "artifact_id": f"artifact-{name}",
        "artifact_version": "myquant.v17.v4.provisional-forward-input.v1",
        "byte_sha256": SHA,
        "cutoff": CUTOFF,
        "relative_path": f"data/inputs/{name}.json",
        "semantic_sha256": SHA,
        "strategy_id": STRATEGY,
    }


def _request(tmp_path: Path) -> tuple[str, str]:
    refs = [_ref(name) for name in ("a", "b", "c", "d", "e", "quant")]
    request = build_provisional_request(
        request_id="sanitized-daily-request",
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        input_refs=refs,
        quant_input_ref=refs[-1],
    )
    path = tmp_path / "data/requests/daily.json"
    path.parent.mkdir(parents=True)
    raw = canonical_resource_bytes(request)
    path.write_bytes(raw)
    return "data/requests/daily.json", hashlib.sha256(raw).hexdigest()


def _manifest_ref() -> dict[str, str]:
    return {
        "artifact_id": "forward-manifest",
        "artifact_version": "myquant.v17.v4.provisional-forward-run-manifest.v1",
        "byte_sha256": "b" * 64,
        "cutoff": CUTOFF,
        "relative_path": "results/v17_v4_shadow/provisional_forward/manifest.json",
        "semantic_sha256": "c" * 64,
        "strategy_id": STRATEGY,
    }


def test_daily_loop_writes_only_immutable_research_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request_sha = _request(tmp_path)
    monkeypatch.setattr(
        "quant_investor.research.research_scheduler.run_provisional_forward",
        lambda *args, **kwargs: {"artifact_manifest_ref": _manifest_ref()},
    )
    result = run_daily_research_loop(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    assert result["run_state"] == "RUN_SUCCESS"
    assert result["research_only"] is True
    assert result["provider_calls"] is False
    assert result["execution"] is result["broker"] is result["order"] is result["trade"] is False
    receipt = json.loads((tmp_path / result["receipt_ref"]["relative_path"]).read_bytes())
    assert receipt["forward_manifest_ref"] == _manifest_ref()
    assert receipt["historical_backfill_eligible"] is False
    assert receipt["production_governance_eligible"] is False
    files = {path.name for path in tmp_path.rglob("*.json")}
    assert any(name.startswith("memory-") for name in files)
    assert any(name.startswith("experiment-") for name in files)
    assert any(name.startswith("receipt-") for name in files)
    assert not any("source_closure" in name or "security_master" in name for name in files)


def test_partial_forward_preserves_refs_without_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request_sha = _request(tmp_path)

    def _partial(*args: object, **kwargs: object) -> dict[str, object]:
        raise ProvisionalForwardError(
            "PROVISIONAL_OPTIONAL_BRANCH_BLOCKED",
            preserved_artifact_refs=[_manifest_ref()],
        )

    monkeypatch.setattr(
        "quant_investor.research.research_scheduler.run_provisional_forward", _partial
    )
    result = run_daily_research_loop(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    assert result["run_state"] == "RUN_PARTIAL"
    receipt = json.loads((tmp_path / result["receipt_ref"]["relative_path"]).read_bytes())
    assert receipt["experiment_registry_ref"] is None
    assert receipt["preserved_artifact_refs"] == [_manifest_ref()]
    assert receipt["blocker_codes"] == ["PROVISIONAL_OPTIONAL_BRANCH_BLOCKED"]


def test_label_maturity_requires_explicit_future_calendar() -> None:
    assert (
        assess_label_maturity(decision_session=SESSION, cutoff=CUTOFF, horizon_sessions=20).status
        == "PENDING"
    )
    blocked = assess_label_maturity(
        decision_session=SESSION,
        cutoff=CUTOFF,
        horizon_sessions=1,
        future_sessions=["2026-08-05"],
    )
    assert blocked.status == "BLOCKED"
    assert blocked.blocker_codes == ("EXPLICIT_CALENDAR_REF_UNAVAILABLE",)
    matured = assess_label_maturity(
        decision_session=SESSION,
        cutoff=CUTOFF,
        horizon_sessions=1,
        future_sessions=["2026-08-05"],
        calendar_ref={"artifact_id": "calendar"},
    )
    assert matured.status == "MATURED"


def test_cli_exposes_only_explicit_daily_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path, request_sha = _request(tmp_path)
    monkeypatch.setattr(
        "quant_investor.research.research_scheduler.run_provisional_forward",
        lambda *args, **kwargs: {"artifact_manifest_ref": _manifest_ref()},
    )
    assert (
        main(
            [
                "research-daily-run",
                "--workspace-root",
                str(tmp_path),
                "--request-path",
                request_path,
                "--request-sha256",
                request_sha,
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["run_state"] == "RUN_SUCCESS"
    with pytest.raises(SystemExit):
        main(["research-daily-run", "--latest"])


def test_request_sha_mismatch_has_no_write_side_effect(tmp_path: Path) -> None:
    request_path, _ = _request(tmp_path)
    with pytest.raises(Exception, match="RESEARCH_REQUEST_SHA_MISMATCH"):
        run_daily_research_loop(str(tmp_path), request_path=request_path, request_sha256="0" * 64)
    assert not (tmp_path / "results").exists()
