from quant_investor.agent_protocol import DataQualityIssue, ModelRoleMetadata, ShortlistItem
from quant_investor.market.dag_executor import _compact_trace_fragments


def test_compact_trace_fragments_accepts_model_role_metadata():
    fragments = _compact_trace_fragments(
        model_roles=ModelRoleMetadata(
            branch_model="deepseek-chat",
            master_model="moonshot-v1-128k",
            branch_fallback_used=True,
            master_fallback_used=False,
            master_reasoning_effort="high",
        ),
        resolver_snapshot={
            "resolution_strategy": "union",
            "directory_priority": ["hs300", "zz500"],
            "physical_directories_used_for_full_a": ["data/cn_market_full/hs300"],
        },
        data_quality_issues=[
            DataQualityIssue(path="bad.csv", message="bad csv", category="parse_error"),
        ],
        shortlist=[
            ShortlistItem(symbol="000001.SZ", company_name="Ping An", category="core"),
        ],
    )

    assert fragments["model_roles"]["branch_model"] == "deepseek-chat"
    assert fragments["model_roles"]["master_model"] == "moonshot-v1-128k"
    assert fragments["model_roles"]["branch_fallback_used"] is True
    assert fragments["resolver"]["directory_priority"] == ["hs300", "zz500"]
    assert fragments["data_quality_issue_count"] == 1
    assert fragments["selected_count"] == 1
