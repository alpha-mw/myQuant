from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_investor.market.dag.theme_context import build_theme_governance_metadata
from quant_investor.themes.governance import (
    evaluate_theme_governance,
    load_theme_governance_registry,
)


def test_theme_governance_classifies_gate_labels() -> None:
    payload = evaluate_theme_governance(
        _rotation(
            _theme(
                "industry::ai",
                score=72,
                confidence=0.66,
                breadth=0.52,
                member_count=18,
                phase="confirmed_rotation",
                smoothed_score=62,
                persistence_count=6,
                trend_state="warming",
            ),
            _theme(
                "industry::chips",
                score=68,
                confidence=0.62,
                breadth=0.48,
                member_count=16,
                phase="overextended",
                risk_flags=["theme_overextended"],
                smoothed_score=61,
                persistence_count=5,
                trend_state="stable",
            ),
            _theme(
                "industry::robot",
                score=41,
                confidence=0.51,
                breadth=0.46,
                member_count=14,
                phase="accumulation",
            ),
            _theme(
                "industry::short",
                score=83,
                confidence=0.75,
                breadth=0.61,
                member_count=4,
                phase="confirmed_rotation",
            ),
            _theme(
                "industry::weak",
                score=29,
                confidence=0.81,
                breadth=0.62,
                member_count=12,
                phase="confirmed_rotation",
            ),
        )
    ).to_dict()

    decisions = _decisions_by_id(payload)

    assert payload["schema_version"] == "theme_governance.v1"
    assert payload["status"] == "success"
    assert payload["enabled"] is True
    assert payload["metadata"] == {
        "deterministic": True,
        "no_llm": True,
        "no_network": True,
        "shadow_only": True,
    }
    assert decisions["industry::ai"]["gate_label"] == "admitted_shadow"
    assert decisions["industry::chips"]["gate_label"] == "watchlist_strong"
    assert decisions["industry::robot"]["gate_label"] == "watchlist_rebuild"
    assert decisions["industry::short"]["gate_label"] == "rejected"
    assert decisions["industry::weak"]["gate_label"] == "rejected"
    assert payload["summary_counts"] == {
        "admitted_shadow": 1,
        "watchlist_strong": 1,
        "watchlist_rebuild": 1,
        "rejected": 2,
        "umbrella_only": 0,
        "unavailable": 0,
    }


def test_theme_governance_blocks_raw_spike_without_smoothing_confirmation() -> None:
    payload = evaluate_theme_governance(
        _rotation(
            _theme(
                "industry::ai",
                score=78,
                confidence=0.70,
                breadth=0.58,
                member_count=18,
                phase="confirmed_rotation",
            )
        )
    ).to_dict()

    decision = _decisions_by_id(payload)["industry::ai"]

    assert decision["gate_label"] == "watchlist_strong"
    assert decision["raw_score"] == 78.0
    assert decision["smoothed_score"] is None
    assert decision["trend_state"] == "insufficient_history"
    assert "raw_spike_not_confirmed" in decision["reasons"]
    assert "theme_smoothing_history_insufficient" in decision["diagnostic_notes"]


def test_theme_governance_can_use_snapshot_history_for_admission() -> None:
    current = _rotation(
        _theme(
            "industry::ai",
            score=63,
            confidence=0.66,
            breadth=0.52,
            member_count=18,
            phase="confirmed_rotation",
        )
    )
    history = [
        _rotation_with_as_of(
            _theme(
                "industry::ai",
                score=score,
                confidence=0.62,
                breadth=0.50,
                member_count=18,
                phase="confirmed_rotation",
            ),
            as_of=f"202606{day:02d}",
        )
        for day, score in enumerate([54, 55, 56, 57, 58, 59, 60, 61, 62], start=9)
    ]

    payload = evaluate_theme_governance(current, history=history).to_dict()
    decision = _decisions_by_id(payload)["industry::ai"]

    assert decision["gate_label"] == "admitted_shadow"
    assert decision["score"] == 58.5
    assert decision["raw_score"] == 63.0
    assert decision["smoothed_score"] == 58.5
    assert decision["persistence_count"] == 9
    assert decision["trend_state"] == "warming"


def test_theme_governance_uses_json_registry_for_umbrella_only(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "themes": [
                    {
                        "theme_id": "industry::ai",
                        "theme_type": "umbrella",
                        "style_tag": "broad_narrative",
                        "parent_theme": "technology",
                        "theme_name": "AI Umbrella",
                        "notes": "Too broad for direct admission.",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    registry = load_theme_governance_registry(registry_path)
    payload = evaluate_theme_governance(
        _rotation(
            _theme(
                "industry::ai",
                score=81,
                confidence=0.79,
                breadth=0.66,
                member_count=28,
                phase="confirmed_rotation",
                theme_name="AI",
            )
        ),
        registry=registry,
    ).to_dict()

    decision = _decisions_by_id(payload)["industry::ai"]

    assert registry.entries["industry::ai"].theme_type == "umbrella"
    assert registry.diagnostic_notes == []
    assert decision["gate_label"] == "umbrella_only"
    assert decision["theme_name"] == "AI Umbrella"
    assert decision["style_tag"] == "broad_narrative"
    assert decision["parent_theme"] == "technology"


def test_theme_governance_registry_malformed_falls_back(tmp_path: Path) -> None:
    registry_path = tmp_path / "bad_registry.json"
    registry_path.write_text("{bad json", encoding="utf-8")

    registry = load_theme_governance_registry(registry_path)
    payload = evaluate_theme_governance(
        _rotation(
            _theme(
                "industry::ai",
                score=72,
                confidence=0.66,
                breadth=0.52,
                member_count=18,
                phase="confirmed_rotation",
                smoothed_score=62,
                persistence_count=6,
                trend_state="warming",
            )
        ),
        registry=registry,
    ).to_dict()

    assert registry.entries == {}
    assert any("theme_governance_registry_malformed" in note for note in registry.diagnostic_notes)
    assert _decisions_by_id(payload)["industry::ai"]["gate_label"] == "admitted_shadow"
    assert any("theme_governance_registry_malformed" in note for note in payload["diagnostic_notes"])


def test_theme_governance_missing_disabled_and_error_inputs_fail_closed() -> None:
    missing = evaluate_theme_governance(None).to_dict()
    disabled = evaluate_theme_governance(
        {
            "schema_version": "theme_rotation.v1",
            "enabled": False,
            "status": "disabled",
            "market": "CN",
            "universe_key": "full_a",
            "as_of": "20260618",
        }
    ).to_dict()
    error = evaluate_theme_governance(
        {
            "schema_version": "theme_rotation.v1",
            "enabled": True,
            "status": "error",
            "market": "CN",
            "universe_key": "full_a",
            "as_of": "20260618",
            "diagnostic_notes": ["theme_scanner_error: boom"],
        }
    ).to_dict()

    assert missing["status"] == "unavailable"
    assert missing["enabled"] is False
    assert missing["decisions"] == []
    assert "theme_rotation_missing" in missing["diagnostic_notes"]

    assert disabled["status"] == "disabled"
    assert disabled["enabled"] is False
    assert disabled["decisions"] == []
    assert "theme_rotation_disabled" in disabled["diagnostic_notes"]

    assert error["status"] == "error"
    assert error["enabled"] is True
    assert error["decisions"] == []
    assert "theme_scanner_error: boom" in error["diagnostic_notes"]


def test_theme_governance_malformed_theme_values_become_unavailable() -> None:
    payload = evaluate_theme_governance(
        _rotation(
            {
                "theme_id": "industry::bad",
                "theme_name": "Bad Input",
                "score": "not-a-number",
                "confidence": 0.72,
                "breadth": 0.55,
                "member_count": 12,
                "phase": "confirmed_rotation",
            }
        )
    ).to_dict()

    decision = _decisions_by_id(payload)["industry::bad"]

    assert decision["gate_label"] == "unavailable"
    assert decision["score"] is None
    assert "malformed_theme_score" in decision["diagnostic_notes"]
    assert payload["summary_counts"]["unavailable"] == 1


def test_theme_governance_metadata_adapter_default_disabled() -> None:
    payload = build_theme_governance_metadata(
        theme_rotation=_rotation(
            _theme(
                "industry::ai",
                score=72,
                confidence=0.66,
                breadth=0.52,
                member_count=18,
                phase="confirmed_rotation",
            )
        ),
        enabled=False,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
    )

    assert payload["schema_version"] == "theme_governance.v1"
    assert payload["enabled"] is False
    assert payload["status"] == "disabled"
    assert payload["decisions"] == []
    assert payload["summary_counts"]["admitted_shadow"] == 0
    assert payload["diagnostic_notes"] == ["theme_governance_disabled"]


def _rotation(*themes: dict[str, Any]) -> dict[str, Any]:
    return _rotation_with_as_of(*themes, as_of="20260618")


def _rotation_with_as_of(*themes: dict[str, Any], as_of: str) -> dict[str, Any]:
    theme_scores = {str(theme["theme_id"]): theme for theme in themes}
    return {
        "schema_version": "theme_rotation.v1",
        "enabled": True,
        "status": "success",
        "market": "CN",
        "universe_key": "full_a",
        "as_of": as_of,
        "theme_scores": theme_scores,
        "top_themes": list(themes),
        "diagnostic_notes": [],
        "metadata": {
            "deterministic": True,
            "no_llm": True,
            "no_network": True,
        },
    }


def _theme(
    theme_id: str,
    *,
    score: float,
    confidence: float,
    breadth: float,
    member_count: int,
    phase: str,
    risk_flags: list[str] | None = None,
    theme_name: str = "",
    smoothed_score: float | None = None,
    persistence_count: int | None = None,
    trend_state: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "theme_id": theme_id,
        "theme_name": theme_name or theme_id.removeprefix("industry::"),
        "score": score,
        "confidence": confidence,
        "breadth": breadth,
        "member_count": member_count,
        "phase": phase,
        "risk_flags": list(risk_flags or []),
    }
    if smoothed_score is not None:
        payload["smoothed_score"] = smoothed_score
        payload["heat_10d"] = smoothed_score
        payload["smoothing_status"] = "success"
    if persistence_count is not None:
        payload["persistence_count"] = persistence_count
    if trend_state:
        payload["trend_state"] = trend_state
    return payload


def _decisions_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(decision["theme_id"]): dict(decision)
        for decision in payload.get("decisions", [])
    }
