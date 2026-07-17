from quant_investor.v16.diagnostic import V16NoAgentDiagnostic


def test_no_agent_layer_is_diagnostic_only_and_cannot_add_risk() -> None:
    payload = V16NoAgentDiagnostic(
        run_id="run-1",
        market="CN",
        eligible_symbol_count=1000,
        funnel_symbol_count=500,
        data_summary={"pit": "sealed"},
    ).to_dict()

    assert payload["status"] == "diagnostic_only"
    assert payload["formal_shortlist_generated"] is False
    assert payload["new_risk_authorized"] is False
    assert payload["target_weights"] == {}
    assert "formal_llm_branch_missing" in payload["blockers"]
