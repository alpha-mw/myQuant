from __future__ import annotations

from quant_investor.agent_protocol import DataQualityIssue


def test_build_data_quality_diagnostics_tracks_universe_tiers():
    from quant_investor.market.data_quality import build_data_quality_diagnostics

    diagnostics = build_data_quality_diagnostics(
        total_symbols=["000001.SZ", "600519.SH", "000858.SZ"],
        researchable_symbols=["000001.SZ", "600519.SH"],
        shortlistable_symbols=["000001.SZ"],
        final_selected_symbols=["000001.SZ"],
        quarantined_symbols=["000858.SZ"],
        issues=[
            DataQualityIssue(
                symbol="000858.SZ",
                severity="error",
                issue_type="csv_unreadable",
                message="cannot parse csv",
            )
        ],
    )

    assert diagnostics.total_universe_count == 3
    assert diagnostics.researchable_universe_count == 2
    assert diagnostics.shortlistable_universe_count == 1
    assert diagnostics.final_selected_universe_count == 1
    assert diagnostics.quarantined_symbols == ["000858.SZ"]
