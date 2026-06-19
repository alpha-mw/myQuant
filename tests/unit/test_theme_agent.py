from __future__ import annotations

from types import SimpleNamespace

from quant_investor.agent_protocol import ActionLabel, Direction
from quant_investor.agents.theme_agent import ThemeAgent
from quant_investor.themes.types import ThemePhase, ThemeScanResult, ThemeScore


def _theme_payload(
    *,
    symbol: str = "000001.SZ",
    symbol_score: float = 0.78,
    phase: str = "confirmed_rotation",
    theme_id: str = "industry::AI",
    theme_name: str = "AI",
    member_count: int = 10,
    risk_flags: list[str] | None = None,
    evidence: list[str] | None = None,
) -> dict[str, object]:
    return {
        "symbol_scores": {symbol: symbol_score},
        "symbol_primary_theme": {symbol: theme_id},
        "symbol_phase": {symbol: phase},
        "symbol_risk_flags": {symbol: list(risk_flags or [])},
        "theme_scores": {
            theme_id: {
                "theme_name": theme_name,
                "member_count": member_count,
                "evidence": list(evidence if evidence is not None else ["breadth=0.70"]),
            }
        },
    }


def test_theme_agent_neutral_without_theme_data():
    verdict = ThemeAgent().run({"symbol": "000001.SZ"})

    assert verdict.final_score == 0.0
    assert verdict.action == ActionLabel.HOLD
    assert verdict.metadata["theme_data_available"] is False
    assert "theme_data_unavailable" in verdict.diagnostic_notes


def test_theme_agent_bullish_for_confirmed_rotation():
    verdict = ThemeAgent().run(
        {
            "symbol": "000001.SZ",
            "theme_scan": _theme_payload(
                symbol_score=0.78,
                phase="confirmed_rotation",
                theme_id="industry::AI",
                theme_name="AI",
                member_count=12,
            ),
        }
    )

    assert verdict.final_score > 0.0
    assert verdict.direction == Direction.BULLISH
    assert verdict.metadata["primary_theme_name"] == "AI"
    assert verdict.metadata["no_llm"] is True


def test_theme_agent_overextended_caps_score():
    verdict = ThemeAgent().run(
        {
            "symbol": "000001.SZ",
            "theme_scan": _theme_payload(
                symbol_score=0.95,
                phase="overextended",
                risk_flags=["theme_overextended"],
            ),
        }
    )

    assert verdict.final_score <= 0.25
    assert "theme_overextended_no_chase" in verdict.investment_risks


def test_theme_agent_distribution_not_positive():
    verdict = ThemeAgent().run(
        {
            "symbol": "000001.SZ",
            "theme_scan": _theme_payload(
                symbol_score=0.75,
                phase="distribution",
                risk_flags=["theme_fake_breakout_risk"],
            ),
        }
    )

    assert verdict.final_score <= 0.0
    assert "theme_distribution_risk" in verdict.investment_risks


def test_theme_agent_reads_global_context_metadata_dict():
    context = SimpleNamespace(metadata={"theme_scan": _theme_payload(theme_name="Robotics")})

    verdict = ThemeAgent().run({"symbol": "000001.SZ", "global_context": context})

    assert verdict.metadata["theme_data_available"] is True
    assert verdict.metadata["primary_theme_name"] == "Robotics"


def test_theme_agent_accepts_theme_scan_result_dataclass():
    scan = ThemeScanResult(
        theme_scores={
            "industry::ai": ThemeScore(
                theme_id="industry::ai",
                theme_name="AI",
                phase=ThemePhase.EARLY_ACCELERATION,
                member_count=14,
                evidence=["momentum=0.80"],
            )
        },
        symbol_scores={"000001.SZ": 0.72},
        symbol_primary_theme={"000001.SZ": "industry::ai"},
        symbol_phase={"000001.SZ": ThemePhase.EARLY_ACCELERATION.value},
        symbol_risk_flags={"000001.SZ": []},
    )

    verdict = ThemeAgent().run({"symbol": "000001.SZ", "theme_scan": scan})

    assert verdict.metadata["primary_theme_name"] == "AI"
    assert verdict.metadata["theme_phase"] == "early_acceleration"
    assert verdict.final_score > 0.0
