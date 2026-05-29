from __future__ import annotations

from quant_investor.reporting.formal_diagnostics import (
    HoldingDecisionDiagnostic,
    ReportWarning,
    apply_report_decision_guardrail,
    build_holding_decision_diagnostics,
    collect_formal_report_warnings,
    render_holding_diagnostic_markdown_table,
    reconcile_branch_vs_final,
)


def test_report_warning_round_trip_and_validation():
    warning = ReportWarning(
        code="stale_snapshot",
        scope="global",
        source="completeness/local_snapshot",
        severity="material",
        data_date="20260424",
        affected_symbol=None,
        decision_impact="downgraded_final_label",
        human_message="local snapshot is stale",
    )

    payload = warning.to_dict()
    restored = ReportWarning.from_dict(payload)

    assert restored == warning
    assert payload["data_date"] == "2026-04-24"


def test_report_warning_invalid_choice_raises():
    try:
        ReportWarning(
            code="unknown_code",
            scope="global",
            source="source",
            severity="material",
            data_date=None,
            affected_symbol=None,
            decision_impact="downgraded_final_label",
            human_message="bad",
        )
    except ValueError as exc:
        assert "code" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid warning code")


def test_collect_warnings_for_stale_snapshot_and_all_zero_confidence():
    holdings_review = [
        {
            "symbol": "601869.SH",
            "name": "长飞光纤",
            "llm_confidence": 0.0,
            "llm_effective_calls": 1,
            "llm_confidence_source": "recommendation.confidence",
        },
        {
            "symbol": "600487.SH",
            "name": "亨通光电",
            "llm_confidence": 0.0,
            "llm_effective_calls": 1,
            "llm_confidence_source": "recommendation.confidence",
        },
    ]
    warnings = collect_formal_report_warnings(
        target_date="20260427",
        dominant_local_snapshot_date="20260424",
        completeness_state={"complete": False, "blocking_incomplete_count": 7302},
        holdings_review=holdings_review,
        review_layer_diagnostics={"effective_call_count": 2},
    )
    codes = [item.code for item in warnings]
    assert "stale_snapshot" in codes
    assert "llm_confidence_unavailable" in codes

    diagnostics = build_holding_decision_diagnostics(
        holdings_review=holdings_review,
        warnings=warnings,
        provisional_label_by_symbol={"601869.SH": "继续持有", "600487.SH": "继续持有"},
        data_date_by_symbol={"601869.SH": "20260424", "600487.SH": "20260424"},
        branch_signals_by_symbol={},
    )
    guardrail = apply_report_decision_guardrail(
        provisional_label="no_action",
        warnings=warnings,
        holding_diagnostics=diagnostics,
        llm_confidences=[0.0, 0.0],
    )

    assert guardrail.display_label in {"no_action_evidence_impaired", "hold_arbitrated"}
    assert guardrail.display_label != "no_action"


def test_collect_warnings_treats_codex_handoff_as_expected_llm_path():
    warnings = collect_formal_report_warnings(
        target_date="20260427",
        dominant_local_snapshot_date="20260424",
        completeness_state={"complete": False, "blocking_incomplete_count": 7302},
        holdings_review=[
            {
                "symbol": "601869.SH",
                "name": "长飞光纤",
                "llm_confidence": None,
                "llm_effective_calls": 0,
                "llm_confidence_source": "codex_handoff",
            }
        ],
        review_layer_diagnostics={
            "effective_call_count": 0,
            "codex_handoff": True,
            "local_llm_disabled": True,
            "fallback_reasons": ["local_llm_disabled_codex_handoff"],
        },
    )

    codes = [item.code for item in warnings]
    assert "stale_snapshot" in codes
    assert "llm_confidence_unavailable" not in codes

    diagnostics = build_holding_decision_diagnostics(
        holdings_review=[
            {
                "symbol": "601869.SH",
                "name": "长飞光纤",
                "llm_confidence": None,
                "llm_effective_calls": 0,
                "llm_confidence_source": "codex_handoff",
            }
        ],
        warnings=warnings,
        provisional_label_by_symbol={"601869.SH": "继续持有"},
        data_date_by_symbol={"601869.SH": "20260424"},
        branch_signals_by_symbol={},
    )
    guardrail = apply_report_decision_guardrail(
        provisional_label="no_action",
        warnings=warnings,
        holding_diagnostics=diagnostics,
        llm_confidences=[None],
    )

    assert guardrail.display_label == "no_action_evidence_impaired"
    assert "LLM confidence" not in guardrail.arbitration_note
    assert "全零" not in guardrail.arbitration_note


def test_intraday_previous_day_realtime_cover_keeps_clean_decision_label():
    holdings_review = [
        {
            "symbol": "601869.SH",
            "name": "长飞光纤",
            "llm_confidence": None,
            "llm_effective_calls": 0,
            "llm_confidence_source": "codex_handoff",
            "recommended_action": "继续持有",
        }
    ]
    warnings = collect_formal_report_warnings(
        target_date="20260520",
        dominant_local_snapshot_date="20260519",
        completeness_state={
            "complete": False,
            "blocking_incomplete_count": 7302,
            "strict_trade_date": "20260520",
            "stable_trade_date": "20260519",
            "freshness_mode": "strict",
            "coverage_threshold": 0.95,
            "quote_snapshot": "20260520101603",
            "categories": {
                "full_a": {
                    "expected": 5502,
                    "date_counts": {"20260519": 5433},
                }
            },
        },
        holdings_review=holdings_review,
        review_layer_diagnostics={
            "effective_call_count": 0,
            "codex_handoff": True,
            "local_llm_disabled": True,
            "fallback_reasons": ["local_llm_disabled_codex_handoff"],
        },
    )

    stale_warning = next(item for item in warnings if item.code == "stale_snapshot")
    assert stale_warning.severity == "info"
    assert stale_warning.decision_impact == "disclosure_only"
    assert "不视为决策阻断" in stale_warning.human_message

    diagnostics = build_holding_decision_diagnostics(
        holdings_review=holdings_review,
        warnings=warnings,
        provisional_label_by_symbol={"601869.SH": "继续持有"},
        data_date_by_symbol={"601869.SH": "20260519"},
        branch_signals_by_symbol={},
    )
    guardrail = apply_report_decision_guardrail(
        provisional_label="no_action",
        warnings=warnings,
        holding_diagnostics=diagnostics,
        llm_confidences=[None],
    )

    assert diagnostics[0].branch_vs_final in {"unknown", "aligned"}
    assert guardrail.display_label == "no_action"
    assert guardrail.material_warning_count == 0


def test_collect_warnings_distinguishes_provider_and_snapshot_missing():
    warnings = collect_formal_report_warnings(
        target_date="20260424",
        dominant_local_snapshot_date="20260424",
        holdings_review=[
            {"symbol": "AAA.SH", "llm_confidence": 0.4, "llm_effective_calls": 1, "llm_confidence_source": "x"},
            {"symbol": "BBB.SH", "llm_confidence": 0.4, "llm_effective_calls": 1, "llm_confidence_source": "x"},
        ],
        fundamental_coverage_by_symbol={
            "AAA.SH": {"missing_modules": ["forecast_revision", "ownership"]},
            "BBB.SH": {"missing_modules": ["management_governance"]},
        },
        enhanced_data_flags_by_symbol={
            "AAA.SH": {
                "forecast_revision": {"provider_missing": True, "missing_scope": "global"},
                "ownership": {"snapshot_missing": True, "missing_scope": "symbol"},
            },
            "BBB.SH": {
                "management_governance": {"snapshot_missing": True, "missing_scope": "symbol"},
            },
        },
        review_layer_diagnostics={"effective_call_count": 2},
    )

    provider = [item for item in warnings if item.code == "provider_missing"]
    snapshot = [item for item in warnings if item.code == "snapshot_missing"]
    assert provider
    assert snapshot
    assert provider[0].affected_symbol == "AAA.SH"
    assert "provider" in provider[0].human_message
    assert any(item.affected_symbol == "BBB.SH" for item in snapshot)


def test_collect_warnings_for_placeholder_and_retired_signal():
    warnings = collect_formal_report_warnings(
        target_date="20260424",
        dominant_local_snapshot_date="20260424",
        holdings_review=[
            {"symbol": "AAA.SH", "llm_confidence": 0.4, "llm_effective_calls": 1, "llm_confidence_source": "x"},
        ],
        branch_diagnostics={
            "AAA.SH": {
                "reviewed_branch_verdicts": {
                    "kline": {
                        "metadata": {
                            "evaluator_name": "placeholder_llm_reviewer",
                            "llm_ready": False,
                            "model_components": {"chronos": {"runtime_mode": "error_fallback"}},
                        },
                        "diagnostic_notes": ["fallback path engaged"],
                    },
                    "intelligence": {
                        "metadata": {"branch_mode": "structured_intelligence_fusion"},
                        "coverage_notes": ["legacy batch retired"],
                    },
                }
            }
        },
        review_layer_diagnostics={"effective_call_count": 1},
    )

    codes = [item.code for item in warnings]
    assert "placeholder_kline_evaluator" in codes
    assert "retired_signal_suppressed" in codes
    assert "provider_missing" not in codes


def test_reconcile_branch_vs_final_requires_arbitration_on_structured_conflict():
    branch_vs_final, note = reconcile_branch_vs_final(
        symbol="AAA.SH",
        provisional_final_label="继续持有",
        holding_review={"llm_action": "hold", "recommended_action": "继续持有"},
        branch_signals={
            "branch_overlays": {"kline": {"action": "sell"}},
            "reviewed_branch_verdicts": {"kline": {"action": "sell"}},
        },
        warnings=[],
    )

    assert branch_vs_final == "conflict_requires_arbitration"
    assert note


def test_healthy_evidence_keeps_clean_label():
    holdings = [
        {
            "symbol": "AAA.SH",
            "name": "示例",
            "llm_confidence": 0.55,
            "llm_effective_calls": 1,
            "llm_confidence_source": "recommendation.confidence",
            "recommended_action": "继续持有",
            "llm_action": "hold",
        }
    ]
    warnings = collect_formal_report_warnings(
        target_date="20260424",
        dominant_local_snapshot_date="20260424",
        holdings_review=holdings,
        review_layer_diagnostics={"effective_call_count": 1},
    )
    diagnostics = build_holding_decision_diagnostics(
        holdings_review=holdings,
        warnings=warnings,
        provisional_label_by_symbol={"AAA.SH": "继续持有"},
        data_date_by_symbol={"AAA.SH": "20260424"},
        branch_signals_by_symbol={"AAA.SH": {"reviewed_branch_verdicts": {"kline": {"action": "hold"}}}},
    )
    guardrail = apply_report_decision_guardrail(
        provisional_label="no_action",
        warnings=warnings,
        holding_diagnostics=diagnostics,
        llm_confidences=[0.55],
    )

    assert not any(item.severity == "material" for item in warnings)
    assert guardrail.display_label == "no_action"


def test_render_holding_diagnostic_markdown_table_header_and_formatting():
    table = render_holding_diagnostic_markdown_table(
        [
            HoldingDecisionDiagnostic(
                symbol="AAA.SH",
                name="甲|公司",
                data_date="20260424",
                final_label="hold_arbitrated",
                branch_vs_final="conflict_requires_arbitration",
                llm_confidence=0.253,
                warning_codes=["snapshot_missing", "stale_snapshot", "snapshot_missing"],
                decision_impact="requires_arbitration",
                arbitration_note="需要|说明",
            )
        ]
    )

    expected_header = "| symbol | name | data_date | final_label | branch_vs_final | llm_confidence | warning_codes | decision_impact | arbitration_note |"
    assert table.splitlines()[0] == expected_header
    assert "0.25" in table
    assert "snapshot_missing,stale_snapshot" in table
    assert "甲\\|公司" in table
    assert "需要\\|说明" in table
