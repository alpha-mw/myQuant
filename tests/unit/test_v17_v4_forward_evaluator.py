from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from zoneinfo import ZoneInfo

import pytest

from quant_investor.factors.forward_evaluator import (
    EvidenceStatus,
    FactorTier,
    FactorTierInput,
    ForwardDiagnosticReceipt,
    OriginObservationRecord,
    adjusted_close_simple_return,
    allocate_factor_tiers,
    annualized_rank_ic_ir,
    average_tie_ranks,
    cost_adjusted_top_bottom_quintile_return,
    deduplicate_origins,
    evaluate_factor_tier,
    label_maturity,
    market_industry_adjusted_returns,
    max_abs_existing_factor_pearson,
    open_session_freshness,
    pearson_ic,
    rank_ic_sign_stability,
    shanghai_horizon_end_sessions,
    spearman_rank_ic,
    top_quintile_capacity,
    top_quintile_turnover,
)
from quant_investor.industry import (
    IndustryContext,
    IndustryEvidence,
    IndustryEvidenceStore,
    score_industry_context,
)
from quant_investor.v17_v4_runtime.themes import (
    ThemeExposure,
    ThemeExposureType,
    score_theme_exposure,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _qualifying_receipt(
    factor_name: str = "factor_a",
    **overrides: object,
) -> ForwardDiagnosticReceipt:
    values: dict[str, object] = {
        "factor_name": factor_name,
        "definition_sha256": SHA_A,
        "factor_set_sha256": SHA_B,
        "quant_policy_sha256": SHA_C,
        "source_lineage_sha256": SHA_D,
        "horizon_sessions": 20,
        "origin_count": 60,
        "minimum_symbols_per_origin": 100,
        "mean_rank_ic": 0.0200001,
        "annualized_rank_ic_ir": 0.5,
        "cost_adjusted_group_return": 0.000001,
        "stability": 0.60,
        "max_abs_existing_factor_correlation": 0.699999,
        "freshness_open_sessions": 5,
    }
    values.update(overrides)
    return ForwardDiagnosticReceipt(**values)  # type: ignore[arg-type]


def test_tier_allocation_requires_exact_boundaries_and_production_closure() -> None:
    challenger = evaluate_factor_tier("factor_a", _qualifying_receipt())
    assert challenger.tier is FactorTier.CHALLENGER
    assert challenger.status is EvidenceStatus.AVAILABLE

    for field_name, value, blocker in (
        ("mean_rank_ic", 0.02, "mean_rank_ic_not_above_0_02"),
        (
            "max_abs_existing_factor_correlation",
            0.70,
            "max_abs_existing_factor_correlation_not_below_0_70",
        ),
        ("freshness_open_sessions", 6, "freshness_above_5_open_sessions"),
    ):
        decision = evaluate_factor_tier(
            "factor_a",
            _qualifying_receipt(**{field_name: value}),
        )
        assert decision.tier is FactorTier.EXPERIMENTAL
        assert blocker in decision.blockers

    incomplete_core = evaluate_factor_tier(
        "factor_a",
        _qualifying_receipt(),
        production_active_set_member=True,
        activation_closure=True,
        health_closure=False,
    )
    assert incomplete_core.tier is FactorTier.CHALLENGER

    core = evaluate_factor_tier(
        "factor_a",
        production_active_set_member=True,
        activation_closure=True,
        health_closure=True,
    )
    assert core.tier is FactorTier.CORE

    allocation = allocate_factor_tiers(
        [
            FactorTierInput("z_factor"),
            FactorTierInput("b_factor", _qualifying_receipt("b_factor")),
            FactorTierInput(
                "a_factor",
                production_active_set_member=True,
                activation_closure=True,
                health_closure=True,
            ),
        ]
    )
    assert allocation.core == ("a_factor",)
    assert allocation.challenger == ("b_factor",)
    assert allocation.experimental == ("z_factor",)


def test_missing_diagnostic_prerequisite_is_typed_unavailable() -> None:
    missing_receipt = evaluate_factor_tier("factor_a")
    assert missing_receipt.tier is FactorTier.EXPERIMENTAL
    assert missing_receipt.status is EvidenceStatus.UNAVAILABLE
    assert missing_receipt.blockers == ("diagnostic_receipt_missing",)

    missing_metric = evaluate_factor_tier(
        "factor_a",
        _qualifying_receipt(mean_rank_ic=None),
    )
    assert missing_metric.status is EvidenceStatus.UNAVAILABLE
    assert "missing:mean_rank_ic" in missing_metric.blockers


def test_exact_horizon_sessions_and_close_maturity() -> None:
    calendar = [(date(2026, 1, 1) + timedelta(days=offset)).isoformat() for offset in range(61)]
    ends = shanghai_horizon_end_sessions("2026-01-01", calendar)
    assert ends.status is EvidenceStatus.AVAILABLE
    assert ends.value == {
        1: "2026-01-02",
        5: "2026-01-06",
        10: "2026-01-11",
        20: "2026-01-21",
        60: "2026-03-02",
    }
    assert (
        open_session_freshness(
            "2026-01-01",
            "2026-01-06",
            calendar,
        ).value
        == 5
    )

    insufficient = shanghai_horizon_end_sessions("2026-01-01", calendar[:20])
    assert insufficient.status is EvidenceStatus.UNAVAILABLE
    assert "horizon_20_end_session_missing" in insufficient.blockers

    shanghai = ZoneInfo("Asia/Shanghai")
    before_close = label_maturity(
        "2026-07-29",
        datetime(2026, 7, 29, 14, 59, 59, tzinfo=shanghai),
    )
    at_close = label_maturity(
        "2026-07-29",
        datetime(2026, 7, 29, 15, 0, 0, tzinfo=shanghai),
    )
    assert before_close.value is False
    assert at_close.value is True


def test_return_adjustment_and_missing_prices() -> None:
    simple = adjusted_close_simple_return(100.0, 110.0)
    assert simple.value == pytest.approx(0.10)

    adjusted = market_industry_adjusted_returns(0.10, 0.03, 0.04)
    assert adjusted.value is not None
    assert adjusted.value.market_adjusted_return == pytest.approx(0.07)
    assert adjusted.value.industry_adjusted_return == pytest.approx(0.06)

    missing = adjusted_close_simple_return(None, 110.0)
    assert missing.status is EvidenceStatus.UNAVAILABLE
    assert missing.blockers == ("start_adjusted_close_missing",)


def test_information_coefficients_ties_icir_and_forward_metrics() -> None:
    pearson = pearson_ic([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert pearson.value == pytest.approx(1.0)

    ranks = average_tie_ranks([1.0, 2.0, 2.0, 4.0])
    assert ranks.value == (1.0, 2.5, 2.5, 4.0)
    spearman = spearman_rank_ic(
        [1.0, 2.0, 2.0, 4.0],
        [1.0, 3.0, 2.0, 4.0],
    )
    assert spearman.value == pytest.approx(0.9486832980505138)

    icir = annualized_rank_ic_ir([0.01, 0.02, 0.03])
    assert icir.value == pytest.approx(2.0 * math.sqrt(252.0 / 20.0))

    group_return = cost_adjusted_top_bottom_quintile_return(
        list(range(10)),
        [value / 100.0 for value in range(10)],
    )
    assert group_return.value == pytest.approx(0.078)

    turnover = top_quintile_turnover(
        ["A", "B", "C", "D"],
        ["B", "C", "D", "E"],
    )
    assert turnover.value == pytest.approx(0.25)
    assert top_quintile_capacity([100.0, 300.0]).value == pytest.approx(2.0)
    assert rank_ic_sign_stability([0.1, 0.2, -0.1, 0.3]).value == pytest.approx(0.75)
    correlation = max_abs_existing_factor_pearson(
        [1.0, 2.0, 3.0],
        {
            "inverse": [3.0, 2.0, 1.0],
            "weak": [1.0, 1.5, 3.0],
        },
    )
    assert correlation.value == pytest.approx(1.0)

    missing_ic = pearson_ic([1.0, None], [2.0, 3.0])
    assert missing_ic.status is EvidenceStatus.UNAVAILABLE


def _origin(
    reference: str,
    *,
    byte_sha256: str = SHA_D,
    semantic_sha256: str = SHA_E,
) -> OriginObservationRecord:
    return OriginObservationRecord(
        factor_name="factor_a",
        definition_sha256=SHA_A,
        factor_set_sha256=SHA_B,
        quant_policy_sha256=SHA_C,
        horizon_sessions=20,
        source_lineage_sha256=SHA_D,
        observation_ref=reference,
        observation_byte_sha256=byte_sha256,
        observation_semantic_sha256=semantic_sha256,
    )


def test_origin_dedup_keeps_ascii_minimum_and_conflicts_fail_closed() -> None:
    duplicate = deduplicate_origins([_origin("z.json"), _origin("a.json")])
    assert duplicate.status is EvidenceStatus.AVAILABLE
    assert [row.observation_ref for row in duplicate.records] == ["a.json"]
    assert duplicate.duplicates[0].kept_ref == "a.json"
    assert duplicate.duplicates[0].duplicate_refs == ("z.json",)

    conflict = deduplicate_origins([_origin("a.json"), _origin("b.json", byte_sha256=SHA_E)])
    assert conflict.status is EvidenceStatus.DUPLICATE_ORIGIN_CONFLICT
    assert conflict.blocked is True
    assert conflict.records == ()
    assert conflict.blockers == ("DUPLICATE_ORIGIN_CONFLICT",)

    missing = deduplicate_origins(
        [
            {
                "factor_name": "factor_a",
                "definition_sha256": SHA_A,
            }
        ]
    )
    assert missing.status is EvidenceStatus.UNAVAILABLE


def _industry_context(**overrides: object) -> IndustryContext:
    values: dict[str, object] = {
        "industry_id": "semiconductors",
        "cycle_stage": "expansion",
        "demand_score": 0.8,
        "supply_score": 0.7,
        "inventory_score": 0.6,
        "pricing_power": 0.5,
        "capex_score": 0.4,
        "earnings_revision": 0.9,
        "policy_score": 0.3,
        "market_confirmation": 0.2,
        "narrative_strength": 0.1,
        "crowding_risk": 0.25,
        "catalysts": ("capacity ramp",),
        "contrary_evidence": ("inventory risk",),
        "confidence": 0.8,
        "evidence_refs": ("evidence/a.json",),
    }
    values.update(overrides)
    return IndustryContext(**values)  # type: ignore[arg-type]


def test_industry_formula_validation_and_immutable_evidence_store() -> None:
    context = _industry_context()
    result = score_industry_context(context)
    expected_base = (
        0.15 * 0.8
        + 0.10 * 0.7
        + 0.10 * 0.6
        + 0.10 * 0.5
        + 0.05 * 0.4
        + 0.15 * 0.9
        + 0.10 * 0.3
        + 0.10 * 0.2
        + 0.05 * 0.1
        + 0.10 * (1.0 - 0.25)
    )
    assert result.base_score == pytest.approx(expected_base)
    assert result.score == pytest.approx(expected_base * 0.8)

    with pytest.raises(ValueError, match="demand_score"):
        _industry_context(demand_score=1.01)

    evidence = IndustryEvidence(
        industry_id="semiconductors",
        evidence_ref="evidence/a.json",
        evidence_type="filing",
        available_at="2026-07-29T07:00:00Z",
        summary="Capacity plan",
    )
    original = IndustryEvidenceStore()
    updated = original.add(evidence)
    assert original.evidence == ()
    assert updated.for_industry("semiconductors") == (evidence,)


def _theme_exposure(**overrides: object) -> ThemeExposure:
    values: dict[str, object] = {
        "symbol": "000001.SZ",
        "theme_id": "ai-infrastructure",
        "exposure_type": ThemeExposureType.DIRECT_BENEFICIARY,
        "revenue_exposure": 0.8,
        "product_exposure": 0.6,
        "customer_exposure": None,
        "supply_chain_position": "upstream",
        "confidence": 0.9,
        "evidence_refs": ("evidence/theme-a.json",),
    }
    values.update(overrides)
    return ThemeExposure(**values)  # type: ignore[arg-type]


def test_theme_type_weights_confidence_gate_and_required_component() -> None:
    direct = score_theme_exposure(_theme_exposure())
    assert direct.base_score == pytest.approx(0.7)
    assert direct.type_weight == 1.0
    assert direct.score == pytest.approx(0.7 * 0.9)

    supplier = score_theme_exposure(
        _theme_exposure(
            exposure_type=ThemeExposureType.SUPPLIER,
            confidence=0.80,
        )
    )
    assert supplier.type_weight == pytest.approx(0.70)
    assert supplier.score == pytest.approx(0.7 * 0.8 * 0.7)

    weak_supplier = score_theme_exposure(
        _theme_exposure(
            exposure_type=ThemeExposureType.SUPPLIER,
            confidence=0.799,
        )
    )
    second_order = score_theme_exposure(
        _theme_exposure(exposure_type=ThemeExposureType.SECOND_ORDER)
    )
    concept_only = score_theme_exposure(
        _theme_exposure(exposure_type=ThemeExposureType.CONCEPT_ONLY)
    )
    assert weak_supplier.score == 0.0
    assert second_order.score == 0.0
    assert concept_only.score == 0.0

    with pytest.raises(ValueError, match="at least one"):
        _theme_exposure(
            revenue_exposure=None,
            product_exposure=None,
            customer_exposure=None,
        )
