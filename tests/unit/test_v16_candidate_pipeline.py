from __future__ import annotations

import pytest

from quant_investor.v16.candidate_pipeline import (
    FormalBranchEvidence,
    LLMBranchVerdict,
    PosteriorMenuItem,
    RetrievalEvidence,
    Stage2Decision,
    build_candidate_union,
    build_posterior_menu,
    map_portfolio_capital,
    seal_four_branch_evidence,
    validate_stage1_review,
    validate_stage2_portfolio,
)


def _verdict(symbol: str) -> LLMBranchVerdict:
    return LLMBranchVerdict(
        symbol=symbol,
        raw_score=0.1,
        confidence=0.7,
        supporting_fact_ids=("pit-fact-1",),
        contradicting_fact_ids=(),
        rationale="Sealed local facts support the fourth-branch view.",
    )


def _menu_item(symbol: str, edge: float | None, win_rate: float = 0.6) -> PosteriorMenuItem:
    return PosteriorMenuItem(
        symbol=symbol,
        posterior_win_rate=win_rate,
        posterior_expected_alpha=0.01,
        posterior_edge_after_costs=edge,
    )


def test_candidate_union_enforces_500_plus_100_and_stable_deduplication() -> None:
    funnel = [f"F{i:03d}" for i in range(500)]
    supplemental = ["F001"] + [f"L{i:03d}" for i in range(99)]
    union = build_candidate_union(funnel, supplemental)

    assert len(union.funnel_symbols) == 500
    assert len(union.supplemental_symbols) == 99
    assert len(union.symbols) == 599
    assert union.symbols[:2] == ("F000", "F001")
    assert union.symbols[500] == "L000"
    assert union.source_by_symbol["F001"] == "quant_funnel"

    exact_boundary = build_candidate_union(
        funnel,
        [f"X{i:03d}" for i in range(100)],
    )
    assert len(exact_boundary.symbols) == 600

    with pytest.raises(ValueError, match="exceeds 500"):
        build_candidate_union([f"F{i:03d}" for i in range(501)], [])
    with pytest.raises(ValueError, match="exceeds 100"):
        build_candidate_union([], [f"L{i:03d}" for i in range(101)])


def test_stage1_requires_formal_llm_verdict_for_entire_sealed_union() -> None:
    union = build_candidate_union(["000001.SZ"], ["600000.SH"])
    notes = [
        RetrievalEvidence(
            symbol="000001.SZ",
            branch="fundamental",
            supporting_fact_ids=("F-1",),
            conflict_note="Cash conversion conflicts with headline profit.",
        )
    ]
    validate_stage1_review(
        union,
        llm_verdicts=[_verdict("000001.SZ"), _verdict("600000.SH")],
        retrieval_evidence=notes,
    )

    with pytest.raises(ValueError, match="symbol-set drift"):
        validate_stage1_review(
            union,
            llm_verdicts=[_verdict("000001.SZ")],
            retrieval_evidence=notes,
        )
    with pytest.raises(ValueError, match="must be quant, fundamental, or macro"):
        RetrievalEvidence(symbol="000001.SZ", branch="llm")


def test_four_branch_seal_requires_q_f_m_llm_without_neutral_substitution() -> None:
    union = build_candidate_union(["000001.SZ"], [])
    records = [
        FormalBranchEvidence(
            symbol="000001.SZ",
            branch=branch,
            raw_score=0.1,
            confidence=0.7,
            evidence_ids=(f"{branch}-evidence",),
        )
        for branch in ("quant", "fundamental", "macro", "llm")
    ]
    sealed = seal_four_branch_evidence(union, records)
    assert [item.branch for item in sealed[0].branches] == [
        "quant",
        "fundamental",
        "macro",
        "llm",
    ]

    with pytest.raises(ValueError, match="missing formal branches"):
        seal_four_branch_evidence(union, records[:-1])


def test_menu_is_edge_win_symbol_sorted_and_retains_negative_edge() -> None:
    menu = build_posterior_menu(
        [
            _menu_item("B", -0.01, 0.9),
            _menu_item("A", 0.02, 0.6),
            _menu_item("C", -0.01, 0.9),
            _menu_item("D", None, 0.99),
        ]
    )
    assert [item.symbol for item in menu] == ["A", "B", "C", "D"]
    assert any(
        item.posterior_edge_after_costs < 0 for item in menu if item.posterior_edge_after_costs
    )


def test_menu_hard_caps_at_50_without_positive_edge_filter() -> None:
    menu = build_posterior_menu([_menu_item(f"S{i:03d}", -float(i + 1)) for i in range(55)])
    assert len(menu) == 50
    assert all(item.posterior_edge_after_costs < 0 for item in menu)


def test_stage2_exact_weights_actions_and_risk_rationale() -> None:
    menu = [_menu_item("BUY1", 0.02), _menu_item("HOLD1", 0.01), _menu_item("NO1", -0.01)]
    decisions = [
        Stage2Decision(
            symbol="BUY1",
            action="BUY",
            selected_for_portfolio=True,
            target_weight=0.3,
            rationale="Best calibrated opportunity.",
            risk_acceptance_rationale="Capacity warning accepted at this explicit weight.",
        ),
        Stage2Decision(
            symbol="HOLD1",
            action="HOLD",
            selected_for_portfolio=True,
            target_weight=0.2,
            rationale="Retain the existing position.",
        ),
        Stage2Decision(
            symbol="NO1",
            action="AVOID",
            selected_for_portfolio=False,
            target_weight=0.0,
            rationale="Insufficient net edge.",
        ),
    ]
    result = validate_stage2_portfolio(
        menu,
        decisions,
        cash_ratio=0.5,
        existing_weights={"BUY1": 0.0, "HOLD1": 0.2, "NO1": 0.0},
        severe_risk_symbols={"BUY1"},
    )
    assert result.cash_ratio == 0.5
    assert result.decisions == tuple(decisions)

    without_rationale = [
        Stage2Decision(
            symbol="BUY1",
            action="BUY",
            selected_for_portfolio=True,
            target_weight=0.3,
            rationale="Best calibrated opportunity.",
        ),
        *decisions[1:],
    ]
    with pytest.raises(ValueError, match="risk_acceptance_rationale"):
        validate_stage2_portfolio(
            menu,
            without_rationale,
            cash_ratio=0.5,
            existing_weights={"BUY1": 0.0, "HOLD1": 0.2, "NO1": 0.0},
            severe_risk_symbols={"BUY1"},
        )


def test_stage2_rejects_normalisation_and_missing_holdings() -> None:
    menu = [_menu_item("A", 0.01)]
    decision = Stage2Decision(
        symbol="A",
        action="BUY",
        selected_for_portfolio=True,
        target_weight=0.4,
        rationale="Explicit IC allocation.",
    )
    with pytest.raises(ValueError, match="must equal 1"):
        validate_stage2_portfolio(
            menu,
            [decision],
            cash_ratio=0.5,
            existing_weights={"A": 0.0},
        )
    with pytest.raises(ValueError, match="incomplete"):
        validate_stage2_portfolio(
            menu,
            [decision],
            cash_ratio=0.6,
            existing_weights={},
        )


def test_stage2_positive_weight_limit_is_twelve_not_exactly_twelve() -> None:
    menu = [_menu_item(f"S{i:02d}", 0.1 - i / 1000) for i in range(13)]
    decisions = [
        Stage2Decision(
            symbol=item.symbol,
            action="BUY",
            selected_for_portfolio=True,
            target_weight=1.0 / 13.0,
            rationale="IC selected this name.",
        )
        for item in menu
    ]
    with pytest.raises(ValueError, match="exceed 12"):
        validate_stage2_portfolio(
            menu,
            decisions,
            cash_ratio=0.0,
            existing_weights={item.symbol: 0.0 for item in menu},
        )

    smaller_menu = menu[:2]
    smaller = [
        Stage2Decision(
            symbol=item.symbol,
            action="BUY",
            selected_for_portfolio=True,
            target_weight=0.25,
            rationale="IC selected this name.",
        )
        for item in smaller_menu
    ]
    validate_stage2_portfolio(
        smaller_menu,
        smaller,
        cash_ratio=0.5,
        existing_weights={item.symbol: 0.0 for item in smaller_menu},
    )


def test_capital_mapping_computes_unrounded_buy_shares_and_preserves_hold() -> None:
    menu = [_menu_item("BUY", 0.02), _menu_item("HOLD", 0.01)]
    portfolio = validate_stage2_portfolio(
        menu,
        [
            Stage2Decision(
                symbol="BUY",
                action="BUY",
                selected_for_portfolio=True,
                target_weight=0.3,
                rationale="Explicit new target.",
            ),
            Stage2Decision(
                symbol="HOLD",
                action="HOLD",
                selected_for_portfolio=True,
                target_weight=0.2,
                rationale="Keep existing position.",
            ),
        ],
        cash_ratio=0.5,
        existing_weights={"BUY": 0.0, "HOLD": 0.2},
    )
    mapped = map_portfolio_capital(
        portfolio,
        total_capital=1_000.0,
        reference_prices={"BUY": 30.0},
        existing_shares={"BUY": 0, "HOLD": 17},
    )
    assert mapped.cash_amount == 500.0
    assert mapped.positions[0].raw_target_shares == 10.0
    assert mapped.positions[1].raw_target_shares == 17.0

    with pytest.raises(ValueError, match="reference price missing"):
        map_portfolio_capital(
            portfolio,
            total_capital=1_000.0,
            reference_prices={},
            existing_shares={"BUY": 0, "HOLD": 17},
        )
