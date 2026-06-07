from __future__ import annotations

from types import SimpleNamespace

from quant_investor.agent_protocol import AgentStatus, ModelRoleMetadata
from quant_investor.agents.stock_reviewers import (
    BranchOverlayPacket,
    MasterSymbolPacket,
)
from quant_investor.market.dag.decision import _is_codex_handoff_model_roles
from quant_investor.market.dag.research import (
    _build_codex_handoff_master,
    _build_codex_handoff_overlay,
    _is_codex_handoff_review,
)
from quant_investor.market import run_pipeline
from quant_investor.model_roles import ModelRoleResolution


def test_codex_handoff_overlay_preserves_base_verdict_and_packet() -> None:
    packet = BranchOverlayPacket(
        symbol="000001.SZ",
        branch_name="quant",
        base_score=0.42,
        base_confidence=0.71,
        thesis="base quant thesis",
        direction="bullish",
        action="buy",
        agreement_points=["trend confirmed"],
        conflict_points=["short-term volatility"],
        risk_points=["liquidity risk"],
        metadata={"resolver": {"source": "local_csv"}},
    )

    overlay = _build_codex_handoff_overlay(packet)

    assert overlay.status == AgentStatus.DEGRADED
    assert overlay.adjusted_score == packet.base_score
    assert overlay.adjusted_confidence == packet.base_confidence
    assert overlay.score_delta == 0.0
    assert overlay.confidence_delta == 0.0
    assert overlay.telemetry.provider == "codex"
    assert overlay.telemetry.fallback_reason == "codex_handoff_pending"
    assert overlay.metadata["codex_handoff_pending"] is True
    assert overlay.metadata["codex_review_packet"]["symbol"] == "000001.SZ"
    assert overlay.metadata["codex_review_packet"]["branch_name"] == "quant"


def test_codex_handoff_master_preserves_baseline_and_packet() -> None:
    packet = MasterSymbolPacket(
        symbol="000001.SZ",
        branch_overlay_summaries=[
            {
                "symbol": "000001.SZ",
                "branch_name": "quant",
                "adjusted_score": 0.31,
                "adjusted_confidence": 0.66,
            }
        ],
        risk_summary={"risk_flags": ["liquidity risk"]},
        baseline_score=0.31,
        baseline_confidence=0.66,
        metadata={"resolver": {"source": "local_csv"}},
    )

    hint = _build_codex_handoff_master(packet)

    assert hint.status == AgentStatus.DEGRADED
    assert hint.score_hint == packet.baseline_score
    assert hint.confidence_hint == packet.baseline_confidence
    assert hint.score_delta == 0.0
    assert hint.confidence_delta == 0.0
    assert hint.telemetry.provider == "codex"
    assert hint.telemetry.fallback_reason == "codex_handoff_pending"
    assert hint.metadata["codex_handoff_pending"] is True
    assert hint.metadata["codex_review_packet"]["baseline_score"] == 0.31


def test_codex_handoff_role_detection() -> None:
    branch_resolution = ModelRoleResolution(
        role="branch",
        resolved_model="codex-handoff",
        metadata={
            "review_layer_mode": "codex_handoff",
            "codex_handoff_pending": True,
        },
    )
    master_resolution = ModelRoleResolution(
        role="master",
        resolved_model="codex-handoff",
        metadata={
            "review_layer_mode": "codex_handoff",
            "codex_handoff_pending": True,
        },
    )
    model_roles = ModelRoleMetadata(
        resolved_branch_model="codex-handoff",
        resolved_master_model="codex-handoff",
        metadata={
            "review_layer_mode": "codex_handoff",
            "codex_handoff_pending": True,
        },
    )

    assert (
        _is_codex_handoff_review(branch_resolution, master_resolution)
        is True
    )
    assert _is_codex_handoff_model_roles(model_roles) is True


def test_unified_pipeline_keeps_requested_agent_layer_in_handoff(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYQUANT_DISABLE_LOCAL_LLM", "true")
    captured: dict[str, object] = {}

    def _fake_download_stage(**kwargs):
        return {
            "data_snapshot": {"local_latest_trade_date": "2026-06-05"}
        }, 0.0

    def _fake_role_models(**kwargs):
        branch = SimpleNamespace(
            primary_model="deepseek-chat",
            fallback_model="",
            candidate_models=["deepseek-chat"],
        )
        master = SimpleNamespace(
            primary_model="deepseek-reasoner",
            fallback_model="",
            candidate_models=["deepseek-reasoner"],
        )
        return branch, master

    def _fake_run_market_analysis(**kwargs):
        captured.update(kwargs)
        return {"results": [], "reports": {}, "analysis_meta": {}}

    monkeypatch.setattr(
        run_pipeline,
        "_run_download_stage",
        _fake_download_stage,
    )
    monkeypatch.setattr(
        run_pipeline,
        "resolve_runtime_role_models",
        _fake_role_models,
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_market_analysis",
        _fake_run_market_analysis,
    )

    run_pipeline.run_unified_pipeline(
        market="CN",
        universe="full_a",
        enable_agent_layer=True,
        skip_download=True,
        verbose=False,
    )

    assert captured["enable_agent_layer"] is True
