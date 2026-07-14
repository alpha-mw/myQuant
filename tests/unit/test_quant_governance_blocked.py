from __future__ import annotations

import pandas as pd

import quant_investor.agents.quant_agent as quant_agent_module
import quant_investor.market.dag.packets as packets_module
from quant_investor.agents.quant_agent import QuantAgent
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.factors.governance import (
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import (
    MinedFactorRegistry,
    REPORT_ONLY_SHADOW_RUNTIME_MODE,
    RuntimeFactorScore,
    score_with_mined_factors,
)


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "TEST.SZ": pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=5),
                "close": [10.0, 10.4, 10.8, 11.2, 11.6],
                "volume": [1_000_000] * 5,
            }
        )
    }


def _blocked_score(_frames):
    return RuntimeFactorScore(
        symbol_scores={"TEST.SZ": 0.0},
        factor_count=0,
        registry_metadata={"fixture": "empty"},
    )


def _ready_score(
    symbol_scores: dict[str, float],
    *,
    runtime_blockers: list[str] | None = None,
) -> RuntimeFactorScore:
    factor_name = "pv_low_dollar_volume_5d"
    return RuntimeFactorScore(
        symbol_scores=symbol_scores,
        factor_count=1,
        factors_used=[factor_name],
        factor_weights={factor_name: 0.05},
        factor_coverages={factor_name: 1.0},
        registry_metadata={
            "governance_runtime": {
                "status": "ready",
                "factor_mode": "governed_mined_factors",
                "production_eligible": True,
                "production_factor_names": [factor_name],
                "factor_runtime_contracts": {factor_name: {"fixture": True}},
                "factor_runtime_contracts_sha256": "a" * 64,
                "factor_runtime_implementation_code_sha256s": {
                    factor_name: "b" * 64
                },
                "quant_production_activation": {
                    "status": "ready",
                    "blockers": [],
                },
                "blockers": [],
            }
        },
        governance_status="ready",
        factor_mode="governed_mined_factors",
        confidence_multiplier=1.0,
        production_eligible=True,
        runtime_blockers=list(runtime_blockers or []),
    )


def test_quant_agent_does_not_resurrect_legacy_proxy_or_adjustment(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        quant_agent_module,
        "score_with_mined_factors",
        _blocked_score,
    )
    frames = _frames()
    verdict = QuantAgent().run(
        {
            "data_bundle": UnifiedDataBundle(
                market="CN",
                symbols=["TEST.SZ"],
                symbol_data=frames,
            ),
            "score_adjustment": 0.10,
            "confidence_adjustment": 0.15,
        }
    )

    assert verdict.final_score == 0.0
    assert verdict.final_confidence == 0.0
    assert verdict.metadata["factor_mode"] == "governance_blocked"
    assert "legacy_proxy_fallback" not in verdict.diagnostic_notes
    assert "governance_blocked_adjustment_ignored" in verdict.diagnostic_notes


def test_dag_quant_packet_is_zero_confidence_when_governance_blocked(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        packets_module,
        "score_with_mined_factors",
        _blocked_score,
    )

    result = packets_module._build_quant_branch_result(frames=_frames())
    verdict = packets_module._build_symbol_quant_verdict(
        symbol="TEST.SZ",
        quant_result=result,
    )

    assert result.final_score == 0.0
    assert result.final_confidence == 0.0
    assert result.symbol_scores == {"TEST.SZ": 0.0}
    assert result.metadata["factor_mode"] == "governance_blocked"
    assert result.metadata["legacy_fallback_allowed"] is False
    assert verdict.final_confidence == 0.0
    assert "legacy_proxy_fallback" not in verdict.diagnostic_notes


def test_global_quant_summary_keeps_cross_section_metrics_diagnostic_when_blocked() -> None:
    bullish_diagnostics = {
        "candidate_count": 100,
        "sample_count": 100,
        "average_return": 0.08,
        "average_volatility": 0.01,
        "breadth": 0.95,
    }
    for governance_status, factor_mode in (
        ("governance_blocked", "governance_blocked"),
        ("report_only", "historical_shadow_report_only"),
    ):
        quant_result = BranchResult(
            branch_name="quant",
            final_score=0.9,
            final_confidence=0.9,
            metadata={
                "governance_status": governance_status,
                "factor_mode": factor_mode,
                "production_eligible": False,
            },
        )

        verdict = packets_module._build_global_quant_verdict(
            cross_section_quant=bullish_diagnostics,
            symbol_count=100,
            quant_result=quant_result,
        )

        assert verdict.final_score == 0.0
        assert verdict.final_confidence == 0.0
        assert verdict.metadata["production_quant_evidence"] is False
        assert verdict.metadata["cross_section_diagnostic_only"] is True
        assert "diagnostic_breadth=0.950" in verdict.coverage_notes


def test_global_quant_summary_rejects_forged_nested_ready_claims() -> None:
    score = _ready_score(
        {f"S{index:03d}": -0.25 for index in range(100)}
    )
    quant_result = BranchResult(
        branch_name="quant",
        final_score=-0.25,
        final_confidence=0.72,
        symbol_scores=dict(score.symbol_scores),
        metadata={
            "governance_status": "ready",
            "factor_mode": "governed_mined_factors",
            "production_eligible": True,
            "mined_factor_runtime": score.to_metadata(),
        },
    )

    verdict = packets_module._build_global_quant_verdict(
        cross_section_quant={
            "candidate_count": 100,
            "sample_count": 100,
            "average_return": 0.08,
            "average_volatility": 0.01,
            "breadth": 0.95,
        },
        symbol_count=100,
        quant_result=quant_result,
    )

    assert verdict.final_score == 0.0
    assert verdict.final_confidence == 0.0
    assert verdict.metadata["production_quant_evidence"] is False


def test_quant_agent_and_dag_reject_forged_ready_score_with_runtime_blocker(
    monkeypatch,
) -> None:
    forged = _ready_score(
        {"TEST.SZ": 0.8},
        runtime_blockers=["activation_receipt_mismatch"],
    )
    monkeypatch.setattr(
        quant_agent_module,
        "score_with_mined_factors",
        lambda _frames: forged,
    )
    monkeypatch.setattr(
        packets_module,
        "score_with_mined_factors",
        lambda _frames: forged,
    )

    agent_verdict = QuantAgent().run(
        {
            "data_bundle": UnifiedDataBundle(
                market="CN",
                symbols=["TEST.SZ"],
                symbol_data=_frames(),
            )
        }
    )
    dag_result = packets_module._build_quant_branch_result(frames=_frames())

    assert agent_verdict.final_score == 0.0
    assert agent_verdict.final_confidence == 0.0
    assert agent_verdict.metadata["factor_mode"] == "governance_blocked"
    assert dag_result.final_score == 0.0
    assert dag_result.final_confidence == 0.0
    assert dag_result.metadata["factor_mode"] == "governance_blocked"


def test_runtime_metadata_is_fail_closed_for_zero_production_factors() -> None:
    metadata = RuntimeFactorScore().to_metadata()

    assert metadata["governance_status"] == "governance_blocked"
    assert metadata["factor_mode"] == "governance_blocked"
    assert metadata["confidence_multiplier"] == 0.0
    assert metadata["legacy_fallback_allowed"] is False


def test_report_only_shadow_can_score_but_never_claim_runtime_confidence() -> None:
    record = FactorRecord(
        name="historical",
        state=FactorLifecycleState.DEPRECATED,
        implementation="builtin:short_term_return",
        weight=0.05,
        deprecated_reason="historical_only",
        gate_results=[
            GateResult(
                gate_id=index,
                gate_key=f"gate_{index}",
                title=f"Gate {index}",
                passed=index != 8,
            )
            for index in range(1, 9)
        ],
    )

    score = score_with_mined_factors(
        _frames(),
        registry=MinedFactorRegistry(
            factors=[record],
            metadata={"historical_shadow_only": True},
        ),
        runtime_mode=REPORT_ONLY_SHADOW_RUNTIME_MODE,
    )
    metadata = score.to_metadata()

    assert score.factor_count == 1
    assert metadata["governance_status"] == "report_only"
    assert metadata["factor_mode"] == "historical_shadow_report_only"
    assert metadata["confidence_multiplier"] == 0.0
    assert metadata["production_eligible"] is False


def test_current_one_factor_registry_is_blocked_by_v2_runtime_contract() -> None:
    score = score_with_mined_factors(_frames())
    metadata = score.to_metadata()

    assert score.factor_count == 0
    assert metadata["governance_status"] == "governance_blocked"
    assert metadata["confidence_multiplier"] == 0.0
    assert metadata["production_eligible"] is False
    assert "canonical_full_chain_replay_producer_unavailable" in metadata[
        "runtime_blockers"
    ]


def test_quant_agent_rejects_a_report_only_score_even_when_it_has_factors(
    monkeypatch,
) -> None:
    def _report_only_score(_frames):
        return RuntimeFactorScore(
            symbol_scores={"TEST.SZ": 0.8},
            factor_count=1,
            factors_used=["historical"],
            factor_weights={"historical": 0.05},
            governance_status="report_only",
            factor_mode="historical_shadow_report_only",
            confidence_multiplier=0.0,
            production_eligible=False,
            runtime_mode=REPORT_ONLY_SHADOW_RUNTIME_MODE,
        )

    monkeypatch.setattr(
        quant_agent_module,
        "score_with_mined_factors",
        _report_only_score,
    )
    verdict = QuantAgent().run(
        {
            "data_bundle": UnifiedDataBundle(
                market="CN",
                symbols=["TEST.SZ"],
                symbol_data=_frames(),
            )
        }
    )

    assert verdict.final_score == 0.0
    assert verdict.final_confidence == 0.0
    assert verdict.metadata["factor_mode"] == "governance_blocked"
