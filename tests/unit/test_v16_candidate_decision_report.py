from __future__ import annotations

import stat
from pathlib import Path

import pytest

from quant_investor.reporting.v16_candidate_decision import (
    V16CandidateReportError,
    build_v16_candidate_decision_report,
    load_v16_candidate_decision_report,
    validate_v16_candidate_decision_report,
    write_v16_candidate_decision_report,
)

SYNTHETIC_SYMBOL = "SYNTH-0001"


def _branch_contributions() -> dict[str, dict[str, object]]:
    scores = {
        "quant": 0.60,
        "fundamental": 0.40,
        "macro": -0.10,
        "llm": 0.20,
    }
    return {
        branch: {
            "status": "ready",
            "score": scores[branch],
            "weight": 0.25,
            "contribution": scores[branch] * 0.25,
            "evidence_sha256": character * 64,
        }
        for branch, character in zip(scores, ("a", "b", "c", "d"), strict=True)
    }


def _inputs() -> dict[str, object]:
    readiness = {
        "schema_version": "v16_run_readiness.v1",
        "path": "results/v16/synthetic-run/v16_run_readiness.json",
        "sha256": "e" * 64,
        "new_risk_authorized": True,
        "blockers": [],
        "activation_candidate": True,
        "activation_blockers": [],
    }
    return {
        "run_id": "synthetic-v16-run",
        "generated_at": "2026-07-17T10:00:00+08:00",
        "analysis_trade_date": "2026-07-16",
        "branch_contributions": _branch_contributions(),
        "retrieval_evidence": {
            "status": "verified",
            "items": [
                {
                    "symbol": SYNTHETIC_SYMBOL,
                    "branch": "fundamental",
                    "supporting_fact_ids": ["synthetic-fact-1"],
                    "contradicting_fact_ids": [],
                    "conflict_note": None,
                }
            ],
            "warnings": [],
        },
        "posterior": {
            "posterior_win_rate": 0.61,
            "posterior_expected_alpha": 0.08,
            "posterior_edge_after_costs": 0.05,
            "win_rate_interval_90": {"lower": 0.52, "upper": 0.70},
            "expected_alpha_interval_90": {"lower": 0.02, "upper": 0.13},
        },
        "risk_advisor": {
            "advisory_only": True,
            "warnings": ["synthetic concentration note"],
            "recommendations": ["review concentration manually"],
        },
        "ic": {
            "menu_symbols": [SYNTHETIC_SYMBOL],
            "actions": [
                {
                    "symbol": SYNTHETIC_SYMBOL,
                    "action": "BUY",
                    "selected_for_portfolio": True,
                    "existing_weight": 0.0,
                    "target_weight": 0.4,
                    "rationale": "synthetic positive edge",
                    "risk_acceptance_rationale": None,
                }
            ],
            "selected_symbols": [SYNTHETIC_SYMBOL],
            "cash_ratio": 0.6,
        },
        "handoff": {
            "status": "complete",
            "artifact_path": "results/v16/synthetic-run/handoff.json",
            "artifact_sha256": "1" * 64,
            "blockers": [],
        },
        "eligibility": {"eligible": True, "blockers": []},
        "execution": {
            "status": "authorized",
            "new_risk_authorized": True,
            "broker_side_effects": False,
            "blockers": [],
        },
        "readiness": readiness,
    }


def _report(**overrides: object) -> dict[str, object]:
    values = _inputs()
    values.update(overrides)
    return build_v16_candidate_decision_report(**values)  # type: ignore[arg-type]


def test_v16_report_contains_exact_four_branch_decision_surfaces() -> None:
    report = _report()

    assert list(report["branch_contributions"]) == [
        "quant",
        "fundamental",
        "macro",
        "llm",
    ]
    assert report["retrieval_evidence"]["items"][0]["branch"] == "fundamental"
    assert report["posterior"] == {
        "posterior_win_rate": 0.61,
        "posterior_expected_alpha": 0.08,
        "posterior_edge_after_costs": 0.05,
        "win_rate_interval_90": {"lower": 0.52, "upper": 0.70},
        "expected_alpha_interval_90": {"lower": 0.02, "upper": 0.13},
    }
    assert report["risk_advisor"]["advisory_only"] is True
    assert report["ic"]["selected_symbols"] == [SYNTHETIC_SYMBOL]
    validate_v16_candidate_decision_report(report)


def test_retrieval_evidence_cannot_become_a_weighted_branch() -> None:
    retrieval = dict(_inputs()["retrieval_evidence"])  # type: ignore[arg-type]
    retrieval["weight"] = 0.25

    with pytest.raises(V16CandidateReportError, match="unexpected=weight"):
        _report(retrieval_evidence=retrieval)


def test_every_formal_branch_weight_must_be_exactly_one_quarter() -> None:
    contributions = _branch_contributions()
    contributions["quant"]["weight"] = 0.30
    contributions["quant"]["contribution"] = 0.18
    contributions["macro"]["weight"] = 0.20
    contributions["macro"]["contribution"] = -0.02

    with pytest.raises(V16CandidateReportError, match="must equal 0.25"):
        _report(branch_contributions=contributions)


def test_retrieval_evidence_rejects_llm_or_scoring_fields() -> None:
    retrieval = dict(_inputs()["retrieval_evidence"])  # type: ignore[arg-type]
    item = dict(retrieval["items"][0])
    item["branch"] = "llm"
    retrieval["items"] = [item]
    with pytest.raises(V16CandidateReportError, match="quant, fundamental, or macro"):
        _report(retrieval_evidence=retrieval)

    item["branch"] = "quant"
    item["confidence"] = 0.9
    with pytest.raises(V16CandidateReportError, match="unexpected=confidence"):
        _report(retrieval_evidence={**retrieval, "items": [item]})


def test_ic_selected_symbols_are_capped_at_twelve() -> None:
    selected = [f"SYNTH-{index:04d}" for index in range(13)]
    ic = {
        "actions": [
            {
                "symbol": symbol,
                "action": "BUY",
                "selected_for_portfolio": True,
                "existing_weight": 0.0,
                "target_weight": 1.0 / 13.0,
                "rationale": "synthetic selection",
                "risk_acceptance_rationale": None,
            }
            for symbol in selected
        ],
        "menu_symbols": selected,
        "selected_symbols": selected,
        "cash_ratio": 0.0,
    }

    with pytest.raises(V16CandidateReportError, match="exceeds maximum 12"):
        _report(ic=ic)


def test_risk_advisor_cannot_claim_authority() -> None:
    risk_advisor = dict(_inputs()["risk_advisor"])  # type: ignore[arg-type]
    risk_advisor["advisory_only"] = False

    with pytest.raises(V16CandidateReportError, match="advisory_only"):
        _report(risk_advisor=risk_advisor)


def test_missing_cost_evidence_keeps_posterior_edge_null() -> None:
    posterior = dict(_inputs()["posterior"])  # type: ignore[arg-type]
    posterior["posterior_edge_after_costs"] = None

    report = _report(posterior=posterior)

    assert report["posterior"]["posterior_edge_after_costs"] is None


def test_hold_must_preserve_existing_weight() -> None:
    ic = {
        "menu_symbols": [SYNTHETIC_SYMBOL],
        "actions": [
            {
                "symbol": SYNTHETIC_SYMBOL,
                "action": "HOLD",
                "selected_for_portfolio": True,
                "existing_weight": 0.3,
                "target_weight": 0.4,
                "rationale": "synthetic hold",
                "risk_acceptance_rationale": None,
            }
        ],
        "selected_symbols": [SYNTHETIC_SYMBOL],
        "cash_ratio": 0.6,
    }

    with pytest.raises(V16CandidateReportError, match="existing_weight"):
        _report(ic=ic)


def test_atomic_owner_only_v16_report_and_v15_namespace_rejection(
    tmp_path: Path,
) -> None:
    report = _report()
    path = tmp_path / "results/v16/synthetic-run/v16_candidate_decision_report.json"
    reference = write_v16_candidate_decision_report(path, report)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert reference["path"] == ("results/v16/synthetic-run/v16_candidate_decision_report.json")
    assert load_v16_candidate_decision_report(path, expected_sha256=reference["sha256"]) == report
    with pytest.raises(V16CandidateReportError, match="results/v16"):
        write_v16_candidate_decision_report(
            tmp_path / "results/v15/synthetic-run/v16_candidate_decision_report.json",
            report,
        )


def test_v15_payload_cannot_masquerade_as_v16() -> None:
    report = _report()
    wrong_schema = dict(report, schema_version="candidate_decision_report.v15")
    wrong_architecture = dict(report, architecture_version="15.0.0-stable")

    with pytest.raises(V16CandidateReportError, match="schema_version mismatch"):
        validate_v16_candidate_decision_report(wrong_schema)
    with pytest.raises(V16CandidateReportError, match="architecture_version mismatch"):
        validate_v16_candidate_decision_report(wrong_architecture)


def test_no_new_risk_report_requires_activation_blocker_reason() -> None:
    readiness = {
        "schema_version": "v16_run_readiness.v1",
        "path": "results/v16/synthetic-run/v16_run_readiness.json",
        "sha256": "e" * 64,
        "new_risk_authorized": False,
        "blockers": ["activation_dashboard_gate_not_ready"],
        "activation_candidate": False,
        "activation_blockers": ["activation_dashboard_gate_not_ready"],
    }
    execution = {
        "status": "no_new_risk",
        "new_risk_authorized": False,
        "broker_side_effects": False,
        "blockers": ["activation_dashboard_gate_not_ready"],
    }
    report = _report(readiness=readiness, execution=execution)

    assert report["execution"]["status"] == "no_new_risk"
    assert report["readiness"]["activation_candidate"] is False
