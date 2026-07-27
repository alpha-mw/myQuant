from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from fastapi.testclient import TestClient

import quant_investor.cli.main as cli_main
import quant_investor.v17_v4_runtime.public_surfaces as public_surfaces
import web.routers.v17_v4_research as v17_v4_router
from quant_investor.v17_v4_contract import seal_semantic
from quant_investor.v17_v4_runtime.formal_activation import FormalState
from quant_investor.v17_v4_runtime.public_surfaces import (
    PublicSurfaceError,
    _CanaryPublicWriter,
    build_dashboard_contract_v4,
    build_public_surface_compatibility_receipts,
    publish_canary_snapshot,
    resolve_public_run,
)
from quant_investor.v17_v4_runtime.source_storage import (
    SourceExactOnceConflict,
    SourceStorageSecurityError,
)
from web.workspace_app import create_app

ROOT = Path(__file__).resolve().parents[2]
CUTOFF = "2026-07-27T08:00:00Z"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _ref(
    artifact_id: str,
    artifact_version: str,
    relative_path: str,
    *,
    byte_sha256: str = SHA_A,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": byte_sha256,
        "cutoff": CUTOFF,
        "relative_path": relative_path,
        "semantic_sha256": SHA_B,
        "strategy_id": "quant-first",
    }


def _resolved_documents() -> tuple[
    FormalState,
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    formal_ref = _ref(
        "formal-output-1",
        "myquant.v17.v4.formal-output.v1",
        "results/v17_v4_formal_research/runs/formal-output-1.json",
    )
    portfolio_ref = _ref(
        "portfolio-output-1",
        "myquant.v17.v4.portfolio-output.v1",
        "data/private/v17_v4_runs/portfolio-output-1.json",
    )
    pointer_ref = _ref(
        "formal-pointer-formal-activation-1",
        "myquant.v17.v4.formal-active-pointer.v1",
        (
            "results/v17_v4_formal_research/strategies/"
            "quant-first/_active.json"
        ),
    )
    completion_ref = _ref(
        "formal-activation-1",
        "myquant.v17.v4.formal-activation-receipt.v1",
        (
            "results/v17_v4_formal_research/strategies/quant-first/"
            "completion_receipts/formal-activation-1.json"
        ),
    )
    intent = {
        "intent_id": "formal-activation-1",
        "strategy_id": "quant-first",
        "formal_output_ref": formal_ref,
        "portfolio_output_ref": portfolio_ref,
    }
    state = FormalState(
        "FORMAL_ACTIVE",
        MappingProxyType(intent),
        MappingProxyType(
            {
                "version": "myquant.v17.v4.formal-active-pointer.v1",
            }
        ),
        MappingProxyType(
            {
                "version": (
                    "myquant.v17.v4.formal-activation-receipt.v1"
                ),
            }
        ),
    )
    formal = {
        "cutoff": CUTOFF,
        "evidence_refs": [portfolio_ref],
    }
    portfolio = {
        "cash_weight": "0.4",
        "cutoff": CUTOFF,
        "gross_weight": "0.6",
        "run_id": "run-1",
        "strategy_id": "quant-first",
        "targets": [
            {
                "current_target": "0.1" if index == 1 else "0",
                "final_target": "0.025",
                "lane": (
                    "REVIEW_ONLY_HOLDING"
                    if index == 1
                    else "SELECTION_POOL"
                ),
                "symbol": f"{index:06d}.SZ",
            }
            for index in range(1, 25)
        ],
    }
    return state, formal, portfolio, pointer_ref, completion_ref


def _fake_public_run(surface: str = "WEB") -> dict[str, Any]:
    (
        _state,
        _formal,
        portfolio,
        pointer_ref,
        completion_ref,
    ) = _resolved_documents()
    return seal_semantic(
        {
            "authority": dict(public_surfaces.PUBLICATION_AUTHORITY),
            "cash_weight": portfolio["cash_weight"],
            "cutoff": CUTOFF,
            "formal_activation_receipt_ref": completion_ref,
            "formal_active_pointer_ref": pointer_ref,
            "formal_output_ref": _ref(
                "formal-output-1",
                "myquant.v17.v4.formal-output.v1",
                "results/v17_v4_formal_research/runs/formal-output-1.json",
            ),
            "gross_weight": portfolio["gross_weight"],
            "is_default": False,
            "portfolio_output_ref": _ref(
                "portfolio-output-1",
                "myquant.v17.v4.portfolio-output.v1",
                "data/private/v17_v4_runs/portfolio-output-1.json",
            ),
            "protocol_version": "myquant.v17.v4",
            "read_only": True,
            "run_id": "run-1",
            "side_effects": dict(public_surfaces.NO_SIDE_EFFECTS),
            "state": "FORMAL_ACTIVE",
            "strategy_id": "quant-first",
            "surface": surface,
            "targets": [dict(row) for row in portfolio["targets"]],
            "version": "myquant.v17.v4.public-run-dto.v1",
            "view_label": "CANARY",
        }
    )


def test_public_projection_is_read_only_canary_and_not_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        public_surfaces,
        "_resolve_documents",
        lambda workspace_root, strategy_id: _resolved_documents(),
    )

    payload = resolve_public_run(
        tmp_path,
        strategy_id="quant-first",
        surface="CLI",
    )

    assert payload["view_label"] == "CANARY"
    assert payload["is_default"] is False
    assert payload["read_only"] is True
    assert payload["authority"]["formal_research_publication"] is True
    assert payload["authority"]["research_runtime_default"] is False
    assert all(value is False for value in payload["side_effects"].values())
    assert [row["symbol"] for row in payload["targets"][:2]] == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert not (tmp_path / "results").exists()


def test_dashboard_v4_is_separate_and_does_not_relabel_v15_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        public_surfaces,
        "resolve_public_run",
        lambda workspace_root, strategy_id, surface: _fake_public_run(
            "DASHBOARD"
        ),
    )
    payload = build_dashboard_contract_v4(
        tmp_path,
        strategy_id="quant-first",
    )
    schema_path = (
        ROOT
        / "portfolio_dashboard/schema/dashboard_contract.v4.schema.json"
    )
    schema_raw = schema_path.read_bytes()
    schema = json.loads(schema_raw)

    assert schema["properties"]["schema_version"]["const"] == (
        "dashboard_contract.v4"
    )
    assert set(schema["required"]) == set(payload)
    assert payload["schema_version"] == "dashboard_contract.v4"
    assert payload["schema_sha256"] == hashlib.sha256(
        schema_raw
    ).hexdigest()
    assert payload["v15_run_readiness"] is None
    assert payload["v17_v4_run_readiness"]["state"] == "FORMAL_ACTIVE"
    assert "themes" not in payload


def test_web_v4_route_is_get_only_and_visibly_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v17_v4_router,
        "resolve_public_run",
        lambda workspace_root, strategy_id, surface: _fake_public_run(
            "WEB"
        ),
    )
    client = TestClient(
        create_app(
            frontend_dist=Path("/tmp/non-existent-v17-v4-dist"),
            auth_token="",
        )
    )

    response = client.get("/api/v4/research-runs/quant-first")
    post = client.post("/api/v4/research-runs/quant-first", json={})

    assert response.status_code == 200
    assert response.json()["view_label"] == "CANARY"
    assert response.json()["read_only"] is True
    assert response.json()["is_default"] is False
    assert post.status_code == 405


def test_web_v4_route_fails_closed_without_formal_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise PublicSurfaceError("missing")

    monkeypatch.setattr(v17_v4_router, "resolve_public_run", blocked)
    client = TestClient(
        create_app(
            frontend_dist=Path("/tmp/non-existent-v17-v4-dist"),
            auth_token="",
        )
    )
    response = client.get("/api/v4/research-runs/quant-first")
    assert response.status_code == 409
    assert response.json() == {
        "detail": "V17 v4 FORMAL_ACTIVE run is unavailable"
    }


def test_canary_schedule_writes_only_exact_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _fake_public_run("SCHEDULE")
    monkeypatch.setattr(
        public_surfaces,
        "resolve_public_run",
        lambda workspace_root, strategy_id, surface: payload,
    )
    expected = payload["formal_active_pointer_ref"]["byte_sha256"]
    first = publish_canary_snapshot(
        tmp_path,
        strategy_id="quant-first",
        session_id="session-1",
        created_at=CUTOFF,
        expected_formal_pointer_sha256=expected,
    )
    second = publish_canary_snapshot(
        tmp_path,
        strategy_id="quant-first",
        session_id="session-1",
        created_at=CUTOFF,
        expected_formal_pointer_sha256=expected,
    )

    assert first["relative_path"] == (
        "results/v17_v4_canary/strategies/quant-first/"
        "public_snapshots/sessions/session-1.json"
    )
    assert first["created"] is True
    assert second["created"] is False
    assert first["snapshot"]["view_label"] == "CANARY"
    assert not (tmp_path / "results/research_runtime_control").exists()
    assert not (
        tmp_path / "results/v17_v4_formal_research"
    ).exists()

    with pytest.raises(PublicSurfaceError, match="FORMAL_POINTER_PREVALUE"):
        publish_canary_snapshot(
            tmp_path,
            strategy_id="quant-first",
            session_id="session-2",
            created_at=CUTOFF,
            expected_formal_pointer_sha256="c" * 64,
        )
    with pytest.raises(SourceStorageSecurityError):
        _CanaryPublicWriter(tmp_path).write_exact_once(
            "results/v17_v4_shadow/escape.json",
            b"x",
        )
    with pytest.raises(SourceExactOnceConflict):
        _CanaryPublicWriter(tmp_path).write_exact_once(
            first["relative_path"],
            b"different",
        )


def test_all_four_public_surface_receipts_bind_v15_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        public_surfaces,
        "resolve_public_run",
        lambda workspace_root, strategy_id, surface: _fake_public_run(
            "CLI"
        ),
    )

    receipts = build_public_surface_compatibility_receipts(
        ROOT,
        tmp_path,
        strategy_id="quant-first",
        created_at=CUTOFF,
    )

    assert [receipt["surface"] for receipt in receipts] == [
        "CLI",
        "DASHBOARD",
        "SCHEDULE",
        "WEB",
    ]
    assert all(receipt["status"] == "ACCEPTED" for receipt in receipts)
    assert all(receipt["explicit_opt_in"] is True for receipt in receipts)
    assert all(
        receipt["v15_default_unchanged"] is True
        for receipt in receipts
    )
    assert all(
        receipt["authority"]["research_runtime_default"] is False
        for receipt in receipts
    )
    schedule = next(
        receipt
        for receipt in receipts
        if receipt["surface"] == "SCHEDULE"
    )
    assert any(
        row["relative_path"].endswith("canary_schedule_policy.v1.json")
        for row in schedule["surface_file_refs"]
    )


def test_main_cli_v17_opt_in_never_calls_v15_market_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        public_surfaces,
        "resolve_public_run",
        lambda workspace_root, strategy_id, surface: _fake_public_run(
            "CLI"
        ),
    )

    cli_main.main(
        [
            "market",
            "analyze",
            "--market",
            "CN",
            "--decision-protocol",
            "v17-v4",
            "--v17-workspace-root",
            str(tmp_path),
            "--v17-strategy-id",
            "quant-first",
        ]
    )

    body = json.loads(capsys.readouterr().out)
    assert body["protocol_version"] == "myquant.v17.v4"
    assert body["view_label"] == "CANARY"
    assert body["side_effects"]["provider_calls"] is False


def test_main_cli_default_remains_v15(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def v15(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "quant_investor.market.analyze.run_market_analysis",
        v15,
    )
    result = cli_main.run_market_analysis(market="CN")
    assert result == {}
    assert captured == {"market": "CN"}
