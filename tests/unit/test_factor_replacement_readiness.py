from __future__ import annotations

import json

import pytest

from quant_investor.factors.replacement_readiness import (
    OUTCOME_BLOCKED,
    OUTCOME_DEPRECATION_PROPOSAL,
    OUTCOME_REDUCE_WEIGHT_PROPOSAL,
    OUTCOME_WATCHLIST,
    assess_replacement_readiness,
)
from scripts.review_quant_factor_replacement_readiness import build_parser, main

FACTOR = "old_liquidity_factor"
CANDIDATE = "fund_fin_net_profit_yoy"


def _health(
    window: str,
    *,
    status: str = "watchlist",
    atomic: bool = True,
    report_day: int | None = None,
    evidence_day: int | None = None,
) -> dict:
    action = "watchlist" if status == "watchlist" else "observe"
    default_day = int(window[-2:])
    report_day = default_day if report_day is None else report_day
    evidence_day = default_day if evidence_day is None else evidence_day
    return {
        "timestamp": f"2026-07-{report_day:02d}T00:00:00Z",
        "run_status": "ok" if atomic else "blocked",
        "evaluation_source_counts": {"fresh_evaluation": 1},
        "fresh_evaluation": {
            "requested": True,
            "strict": True,
            "atomic_success": atomic,
            "evaluated_factor_count": 1,
            "blockers": [] if atomic else ["fixture_blocked"],
        },
        "decisions": [
            {
                "factor_name": FACTOR,
                "status": status,
                "action": action,
                "evaluation_source": "fresh_evaluation",
                "maturity_window_id": window,
                "evidence_end_date": f"2026-07-{evidence_day:02d}",
            }
        ],
    }


def _complete_shadow(
    *,
    cross_branch: bool | None = True,
    selection_bias: bool | None = True,
    status: str = "passed",
    baseline_status: str = "matched",
    monthly_rankic_count: int = 12,
) -> dict:
    cross_branch_evidence = (
        None if cross_branch is None else {"evaluated": True, "positive": cross_branch}
    )
    readiness = {
        "candidate_name": CANDIDATE,
        "candidate_maturity": {
            "month_end_rankic_count": monthly_rankic_count,
            "nonoverlap_30d_cohort_count": 1,
        },
        "candidate_coverage": {"coverage_rate": 0.72},
        "scope_complete": True,
        "by_removed_factor": {
            FACTOR: {
                "loo_deletion": {"evaluated": True, "not_worse": True},
                "candidate_replacement": {
                    "evaluated": True,
                    "better_than_a": True,
                    "better_than_b": True,
                    "runtime_recomputed": True,
                },
                "redundancy": {"evaluated": True, "is_redundant": True},
                "diversifier_tail_protection": {
                    "evaluated": True,
                    "material_protection_found": False,
                },
            }
        },
    }
    if cross_branch_evidence is not None:
        readiness["cross_branch_conditional_increment"] = cross_branch_evidence
    if selection_bias is not None:
        readiness["covered_uncovered_selection_bias"] = {
            "evaluated": True,
            "acceptable": selection_bias,
        }
    return {
        "generated_at": "2026-10-31T00:00:00Z",
        "status": status,
        "fail_closed": status != "passed",
        "fail_closed_blockers": [] if status == "passed" else ["fixture_blocked"],
        "measurement_only": True,
        "registry": {
            "unchanged": True,
            "sha256_before": "registry-sha256",
            "sha256_after": "registry-sha256",
        },
        "preregistration": {
            "status": "matched",
            "actual_policy_sha256": "preregistration-sha256",
        },
        "baseline_contract": {
            "status": baseline_status,
            "actual_contract_sha256": "baseline-sha256",
        },
        "runtime_parity": {
            "old14_max_abs_delta": 0.0,
            "candidate_max_abs_delta": 0.0,
        },
        "candidate": {"name": CANDIDATE},
        "replacement_readiness": readiness,
        "arm_protocol": {
            "runtime_composite_recomputed": True,
            "gate8_linear_overlay_used": False,
        },
        "arms": [
            {"arm_type": "leave_one_out", "removed_factor": FACTOR},
            {"arm_type": "one_for_one_replacement", "removed_factor": FACTOR},
        ],
    }


def _decision(payload: dict) -> dict:
    return payload["factor_decisions"][0]


def test_duplicate_maturity_window_counts_once_and_cannot_deprecate() -> None:
    payload = assess_replacement_readiness(
        [_health("window-01"), _health("window-01")],
        [_complete_shadow()],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["distinct_matured_alpha_failure_count"] == 1
    assert decision["distinct_maturity_window_ids"] == ["window-01"]
    assert decision["outcome"] == OUTCOME_WATCHLIST


def test_data_blocked_window_is_not_counted_as_alpha_failure() -> None:
    payload = assess_replacement_readiness(
        [_health("window-01", status="data_blocked")],
        [_complete_shadow()],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["distinct_matured_alpha_failure_count"] == 0
    assert decision["data_blocked_window_count"] == 1
    assert decision["outcome"] == OUTCOME_BLOCKED
    assert "fresh_data_blocked_no_alpha_conclusion" in decision["blockers"]


def test_healthy_window_resets_prior_distinct_failure_streak() -> None:
    payload = assess_replacement_readiness(
        [
            _health("window-01"),
            _health("window-02"),
            _health("window-03", status="healthy"),
            _health("window-04"),
        ],
        [_complete_shadow()],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["distinct_matured_alpha_failure_count"] == 1
    assert decision["distinct_maturity_window_ids"] == ["window-04"]
    assert decision["latest_determinable_alpha_status"] == "failure"
    assert decision["outcome"] == OUTCOME_WATCHLIST


def test_data_blocked_window_neither_counts_nor_resets_failure_streak() -> None:
    payload = assess_replacement_readiness(
        [
            _health("window-01"),
            _health("window-02", status="data_blocked"),
            _health("window-03"),
        ],
        [_complete_shadow()],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["distinct_matured_alpha_failure_count"] == 2
    assert decision["distinct_maturity_window_ids"] == ["window-01", "window-03"]
    assert decision["data_blocked_window_count"] == 1
    assert decision["latest_determinable_alpha_status"] == "failure"
    assert decision["outcome"] == OUTCOME_REDUCE_WEIGHT_PROPOSAL


def test_late_rerun_of_older_cohorts_cannot_rebuild_post_healthy_streak() -> None:
    payload = assess_replacement_readiness(
        [
            _health("healthy-04", status="healthy", report_day=4, evidence_day=4),
            _health("window-01", report_day=5, evidence_day=1),
            _health("window-02", report_day=6, evidence_day=2),
            _health("window-03", report_day=7, evidence_day=3),
        ],
        [_complete_shadow()],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["distinct_matured_alpha_failure_count"] == 0
    assert decision["latest_determinable_alpha_status"] == "healthy"
    assert decision["outcome"] == OUTCOME_BLOCKED
    assert any("maturity_cohort_time_regression" in item for item in decision["blockers"])


def test_three_distinct_matured_failures_and_all_safety_evidence_propose_deprecation() -> None:
    payload = assess_replacement_readiness(
        [_health("window-01"), _health("window-02"), _health("window-03")],
        [_complete_shadow()],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["distinct_matured_alpha_failure_count"] == 3
    assert decision["outcome"] == OUTCOME_DEPRECATION_PROPOSAL
    assert decision["proposal_only"] is True
    assert decision["blockers"] == []
    assert payload["registry_update_status"] == "not_written_read_only_proposal"


def test_cross_branch_conditional_increment_missing_blocks_proposal() -> None:
    payload = assess_replacement_readiness(
        [_health("window-01"), _health("window-02"), _health("window-03")],
        [_complete_shadow(cross_branch=None)],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["outcome"] == OUTCOME_BLOCKED
    assert "cross_branch_conditional_increment_missing" in decision["blockers"]
    assert decision["outcome"] != OUTCOME_DEPRECATION_PROPOSAL


def test_false_boolean_aliases_cannot_be_overridden_by_later_true_aliases() -> None:
    shadow = _complete_shadow()
    readiness = shadow["replacement_readiness"]
    readiness["cross_branch_conditional_increment"] = False
    shadow["cross_branch_increment"] = {"evaluated": True, "positive": True}
    factor_evidence = readiness["by_removed_factor"][FACTOR]
    factor_evidence["loo_deletion"] = False
    factor_evidence["loo_not_worse"] = {"evaluated": True, "not_worse": True}

    payload = assess_replacement_readiness(
        [_health("window-01"), _health("window-02"), _health("window-03")],
        [shadow],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    selection = payload["selection_evidence"]
    assert selection["cross_branch_conditional_increment_positive"] is False
    assert selection["factor_gates"][FACTOR]["loo_deletion_not_worse"] is False
    assert decision["outcome"] == OUTCOME_BLOCKED
    assert any(
        "cross_branch_conditional_increment_alias_conflict" in item for item in decision["blockers"]
    )
    assert any("loo_evidence_alias_conflict" in item for item in decision["blockers"])


@pytest.mark.parametrize(
    ("selection_bias", "expected_blocker"),
    [
        (None, "covered_uncovered_selection_bias_review_missing"),
        (False, "covered_uncovered_selection_bias_not_acceptable"),
    ],
)
def test_selection_bias_missing_or_false_blocks_proposal(
    selection_bias: bool | None,
    expected_blocker: str,
) -> None:
    payload = assess_replacement_readiness(
        [_health("window-01"), _health("window-02"), _health("window-03")],
        [_complete_shadow(selection_bias=selection_bias)],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["outcome"] == OUTCOME_BLOCKED
    assert expected_blocker in decision["blockers"]
    assert decision["outcome"] != OUTCOME_DEPRECATION_PROPOSAL


def test_blocked_shadow_and_baseline_mismatch_block_deprecation_proposal() -> None:
    payload = assess_replacement_readiness(
        [_health("window-01"), _health("window-02"), _health("window-03")],
        [_complete_shadow(status="blocked", baseline_status="mismatch")],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    assert decision["outcome"] == OUTCOME_BLOCKED
    assert "selection_shadow_status_not_passed:blocked" in decision["blockers"]
    assert "selection_shadow_has_fail_closed_blockers" in decision["blockers"]
    assert "selection_shadow_baseline_contract_not_matched" in decision["blockers"]
    assert decision["outcome"] != OUTCOME_DEPRECATION_PROPOSAL


def test_forged_ledger_maturity_cannot_raise_candidate_maturity_or_propose() -> None:
    forged_ledger = {
        "schema_version": "2026-07-11.quant-factor-selection-observation.v2",
        "observation_key": "forged-ledger-row",
        "candidate": CANDIDATE,
        "monthly_rankic_count_from_registry": 999,
        "measurement_only": True,
        "registry_write": False,
        "registry_sha256": "old-registry-sha256",
        "preregistration_policy_sha256": "old-preregistration-sha256",
        "baseline_contract_sha256": "old-baseline-sha256",
    }
    payload = assess_replacement_readiness(
        [_health("window-01"), _health("window-02"), _health("window-03")],
        [_complete_shadow(monthly_rankic_count=1)],
        [forged_ledger],
        candidate_name=CANDIDATE,
    )

    decision = _decision(payload)
    maturity = payload["candidate"]["maturity"]
    assert maturity["month_end_rankic_count"] == 1
    assert maturity["passed"] is False
    assert payload["selection_evidence"]["ledger_rows_are_maturity_evidence"] is False
    assert payload["selection_evidence"]["valid_ledger_audit_row_count"] == 0
    assert decision["outcome"] == OUTCOME_BLOCKED
    assert "candidate_maturity_not_met" in decision["blockers"]
    assert any("ledger_registry_sha256_mismatch" in item for item in decision["blockers"])
    assert decision["outcome"] != OUTCOME_DEPRECATION_PROPOSAL


def test_genuine_v2_ledger_is_audited_but_never_raises_maturity() -> None:
    genuine_ledger = {
        "schema_version": "2026-07-11.quant-factor-selection-observation.v2",
        "observation_key": "genuine-v2-ledger-row",
        "candidate": CANDIDATE,
        "monthly_rankic_count_from_registry": 999,
        "measurement_only": True,
        "registry_write": False,
        "registry_sha256": "registry-sha256",
        "preregistration_policy_sha256": "preregistration-sha256",
        "baseline_contract_sha256": "baseline-sha256",
    }
    payload = assess_replacement_readiness(
        [_health("window-01"), _health("window-02"), _health("window-03")],
        [_complete_shadow(monthly_rankic_count=1)],
        [genuine_ledger],
        candidate_name=CANDIDATE,
    )

    maturity = payload["candidate"]["maturity"]
    assert maturity["month_end_rankic_count"] == 1
    assert maturity["passed"] is False
    assert payload["selection_evidence"]["valid_ledger_audit_row_count"] == 1
    assert payload["selection_evidence"]["observation_ledger_audits"] == [
        {
            "observation_key": "genuine-v2-ledger-row",
            "accepted_for_audit": True,
            "used_for_candidate_maturity": False,
            "blockers": [],
        }
    ]
    assert _decision(payload)["outcome"] == OUTCOME_BLOCKED


def test_requested_unknown_factor_is_blocked_not_kept() -> None:
    payload = assess_replacement_readiness(
        [_health("window-01")],
        [_complete_shadow()],
        candidate_name=CANDIDATE,
        factor_names=["unknown_factor"],
    )

    decision = _decision(payload)
    assert decision["factor_name"] == "unknown_factor"
    assert decision["outcome"] == OUTCOME_BLOCKED
    assert "requested_factor_missing_from_health_evidence" in decision["blockers"]


def test_cli_has_no_registry_write_or_promotion_argument() -> None:
    option_strings = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert "--registry-path" not in option_strings
    assert "--write" not in option_strings
    assert "--apply-registry-actions" not in option_strings
    assert "--promote" not in option_strings


def test_cli_writes_only_measurement_reports(tmp_path) -> None:
    health = tmp_path / "health.json"
    shadow = tmp_path / "shadow.json"
    output = tmp_path / "reports"
    health.write_text(json.dumps(_health("window-01")), encoding="utf-8")
    shadow.write_text(json.dumps(_complete_shadow()), encoding="utf-8")

    assert (
        main(
            [
                "--health-json",
                str(health),
                "--selection-shadow-json",
                str(shadow),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(
        (output / "quant_factor_replacement_readiness.json").read_text(encoding="utf-8")
    )
    assert payload["measurement_only"] is True
    assert payload["freeze"]["registry_mutation_allowed"] is False
    assert payload["factor_decisions"][0]["outcome"] == OUTCOME_WATCHLIST
    assert sorted(path.name for path in output.iterdir()) == [
        "quant_factor_replacement_readiness.json",
        "quant_factor_replacement_readiness.md",
    ]


def test_cli_returns_nonzero_for_blocked_readiness(tmp_path) -> None:
    health = tmp_path / "health.json"
    shadow = tmp_path / "shadow.json"
    output = tmp_path / "blocked-reports"
    health.write_text(
        json.dumps(_health("window-01", status="data_blocked")),
        encoding="utf-8",
    )
    shadow.write_text(json.dumps(_complete_shadow()), encoding="utf-8")

    assert (
        main(
            [
                "--health-json",
                str(health),
                "--selection-shadow-json",
                str(shadow),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )


def test_parser_rejects_registry_write_argument() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--health-json",
                "health.json",
                "--selection-shadow-json",
                "shadow.json",
                "--registry-path",
                "registry.json",
            ]
        )
