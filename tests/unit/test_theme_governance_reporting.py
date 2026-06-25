from __future__ import annotations

from quant_investor.reporting.theme_governance_renderer import (
    append_theme_governance_section_once,
    render_theme_governance_markdown,
)
from quant_investor.themes.governance import evaluate_theme_governance


def test_theme_governance_markdown_is_shadow_only_and_baseline_executable() -> None:
    payload = evaluate_theme_governance(
        {
            "schema_version": "theme_rotation.v1",
            "enabled": True,
            "status": "success",
            "market": "CN",
            "universe_key": "full_a",
            "as_of": "20260618",
            "theme_scores": {
                "industry::ai": {
                    "theme_id": "industry::ai",
                    "theme_name": "AI",
                    "score": 72,
                    "raw_score": 72,
                    "smoothed_score": 61,
                    "heat_10d": 61,
                    "heat_delta_5d": 3.5,
                    "persistence_count": 6,
                    "trend_state": "warming",
                    "smoothing_status": "success",
                    "confidence": 0.66,
                    "breadth": 0.52,
                    "member_count": 18,
                    "phase": "confirmed_rotation",
                    "risk_flags": [],
                }
            },
        }
    ).to_dict()

    markdown = render_theme_governance_markdown(payload, max_rows=10)

    assert "## Theme Governance Sidecar" in markdown
    assert "shadow/governance only" in markdown
    assert "final executable decision remains baseline" in markdown
    assert "admitted_shadow" in markdown
    assert "10日热度" in markdown
    assert "5日变化" in markdown
    assert "持续天数" in markdown
    assert "| AI | admitted_shadow | 72.0 | 61.0 | 3.5 | 6 | warming |" in markdown


def test_append_theme_governance_section_once_avoids_duplicates() -> None:
    payload = {
        "schema_version": "theme_governance.v1",
        "enabled": True,
        "status": "success",
        "decisions": [
            {
                "theme_id": "industry::ai",
                "theme_name": "AI",
                "gate_label": "admitted_shadow",
                "score": 72.0,
                "raw_score": 72.0,
                "smoothed_score": 61.0,
                "heat_10d": 61.0,
                "heat_delta_5d": 3.5,
                "persistence_count": 6,
                "trend_state": "warming",
                "confidence": 0.66,
                "breadth": 0.52,
                "member_count": 18,
                "phase": "confirmed_rotation",
                "style_tag": "",
                "diagnostic_notes": [],
            }
        ],
        "summary_counts": {"admitted_shadow": 1},
        "metadata": {
            "deterministic": True,
            "no_llm": True,
            "no_network": True,
            "shadow_only": True,
        },
    }
    base = "# Daily Review\n\nBody.\n"

    once = append_theme_governance_section_once(base, payload, max_rows=10)
    twice = append_theme_governance_section_once(once, payload, max_rows=10)

    assert once.count("## Theme Governance Sidecar") == 1
    assert twice.count("## Theme Governance Sidecar") == 1
    assert twice == once
