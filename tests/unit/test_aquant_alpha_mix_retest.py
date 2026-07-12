from __future__ import annotations

import json

from scripts.retest_aquant_alpha_mix_8gate import evaluate_with_myquant_gate, load_candidates


def _base_metrics(**overrides):
    metrics = {
        "no_future_leakage": True,
        "uses_availability_date": True,
        "point_in_time_rebalance": True,
        "adjusted_price_consistent": True,
        "tradability_rules_defined": True,
        "missingness_explained": True,
        "coverage_rate": 0.75,
        "nan_rate": 0.10,
        "monthly_coverage_min": 0.60,
        "max_sector_coverage_share": 0.45,
        "max_size_bucket_coverage_share": 0.55,
        "extreme_value_ratio": 0.01,
        "icir": 0.62,
        "mean_rankic": 0.035,
        "positive_ic_ratio": 0.58,
        "rankic_direction_stable": True,
        "max_single_year_ic_contribution": 0.35,
        "family_fdr_method": "benjamini_hochberg_by_family",
        "family_fdr_q_value": 0.08,
        "family_fdr_passed": True,
        "top_bottom_spread": 0.08,
        "top_quantile_return": 0.05,
        "monotonicity": 0.55,
        "long_short_from_long_side": True,
        "turnover": 4.0,
        "cost_adjusted_return": 0.04,
        "slippage_sensitivity_ok": True,
        "execution_realism": True,
        "capacity_pressure": 0.30,
        "neutralized_icir": 0.31,
        "existing_factor_corr": 0.25,
        "style_exposure_only": False,
        "oos_positive_ratio": 0.60,
        "parameter_stability": True,
        "date_range_robustness": True,
        "rebalance_frequency_robustness": True,
        "universe_robustness": True,
        "regime_robustness": True,
        "walk_forward_purged": True,
        "walk_forward_purge_days": 30,
        "walk_forward_embargo_days": 30,
        "walk_forward_fold_count": 4,
        "walk_forward_evidence_hash": "1" * 64,
        "master_return_delta": 0.025,
        "sharpe_delta": 0.10,
        "max_drawdown_delta": -0.01,
        "turnover_delta": 0.05,
        "execution_cost_delta": 0.002,
        "correlation_with_existing_signals": 0.20,
        "gate8_evidence_schema": "factor-governance-replay-evidence.v2",
        "gate8_evidence_hash": "2" * 64,
        "full_control_chain_evaluated": True,
        "gate8_arm_hashes": {
            "A": "a" * 64,
            "B": "b" * 64,
            "C": "c" * 64,
            "D": "d" * 64,
        },
    }
    metrics.update(overrides)
    return metrics


def test_retest_runner_uses_factor_gate_evaluator_for_coverage_failure():
    review = evaluate_with_myquant_gate(
        "alpha_mix_vwap40_50_ocfprofit_50",
        _base_metrics(coverage_rate=0.10, monthly_coverage_min=0.05),
    )

    assert review.decision.value == "revise"
    assert review.gate_results[1].gate_id == 2
    assert not review.gate_results[1].passed


def test_retest_runner_complete_metrics_can_reach_production_candidate():
    review = evaluate_with_myquant_gate(
        "alpha_mix_vwap40_50_ocfprofit_50",
        _base_metrics(),
    )

    assert review.decision.value == "production_candidate"
    assert len(review.gate_results) == 8
    assert all(gate.passed for gate in review.gate_results)


def test_load_candidates_marks_independent_subset(tmp_path):
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "ready_factors.json").write_text(
        json.dumps(
            [
                {
                    "name": "alpha_mix_vwap40_50_ocfprofit_50",
                    "expression": "cs_rank(fin_ocf_to_profit)",
                },
                {
                    "name": "alpha_mix_vwap80_40_ocfprofit_60",
                    "expression": "cs_rank(fin_ocf_to_profit)",
                },
            ]
        )
    )
    (audit_dir / "independent_ready_subset.json").write_text(
        json.dumps({"factor_names": ["alpha_mix_vwap40_50_ocfprofit_50"]})
    )

    candidates, independent_names = load_candidates(audit_dir, "all")

    assert independent_names == ["alpha_mix_vwap40_50_ocfprofit_50"]
    assert [candidate.name for candidate in candidates] == [
        "alpha_mix_vwap40_50_ocfprofit_50",
        "alpha_mix_vwap80_40_ocfprofit_60",
    ]
    assert [candidate.independent for candidate in candidates] == [True, False]
