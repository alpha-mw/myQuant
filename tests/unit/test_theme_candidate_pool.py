from __future__ import annotations

import importlib

import pytest

from quant_investor.agent_protocol import GlobalContext
from quant_investor.funnel.theme_candidate_pool import (
    ThemeCandidatePoolBuilder,
    ThemeGatePolicy,
    ThemePoolConfig,
)


def _config(**overrides: object) -> ThemePoolConfig:
    values = {
        "enabled": True,
        "required": True,
        "use_markov_policy": True,
        "score_source": "smoothed",
        "fallback_to_raw_score": True,
        "base_min_theme_score": 0.58,
        "base_min_symbol_score": 0.55,
        "base_top_themes": 8,
        "max_symbols_per_theme": 30,
        "residual_ratio": 0.25,
        "min_residual_symbols": 2,
        "min_admitted_themes": 2,
        "allow_unthemed_residual": False,
        "include_risk_watch": True,
        "risk_watch_max_ratio": 0.20,
        "symbol_gate_mode": "classify",
        "allowed_phases": (
            "accumulation",
            "early_acceleration",
            "confirmed_rotation",
        ),
        "blocked_phases": ("distribution",),
        "blocked_flags": (
            "theme_distribution_risk",
            "theme_fake_breakout_risk",
        ),
        "min_member_count": 3,
    }
    values.update(overrides)
    return ThemePoolConfig(**values)


def _markov(
    regime: str = "趋势上涨",
    *,
    transition_risk: float = 0.20,
    confidence: float = 0.80,
    production_eligible: bool | None = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "dominant_regime": regime,
        "transition_risk": transition_risk,
        "confidence": confidence,
    }
    if production_eligible is not None:
        payload["production_eligible"] = production_eligible
    return payload


def _theme(
    theme_id: str,
    *,
    score: float = 0.80,
    phase: str = "confirmed_rotation",
    breadth: float = 0.70,
    confidence: float = 0.80,
    member_count: int = 8,
    risk_flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "theme_id": theme_id,
        "theme_name": f"{theme_id}_name",
        "score": score,
        "phase": phase,
        "breadth": breadth,
        "confidence": confidence,
        "member_count": member_count,
        "acceleration": 0.60,
        "volume_confirmation": 0.60,
        "overextension_risk": 0.05,
        "fake_breakout_risk": 0.05,
        "risk_flags": list(risk_flags or []),
    }


def _rotation(
    *,
    theme_scores: dict[str, dict[str, object]],
    symbol_theme: dict[str, str],
    symbol_scores: dict[str, float] | None = None,
    smoothed_scores: dict[str, float] | None = None,
    symbol_phase: dict[str, str] | None = None,
    risk_flags: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    raw_scores = symbol_scores or {symbol: 0.80 for symbol in symbol_theme}
    return {
        "status": "success",
        "theme_scores": theme_scores,
        "symbol_scores": raw_scores,
        "symbol_smoothed_scores": smoothed_scores if smoothed_scores is not None else raw_scores,
        "symbol_primary_theme": symbol_theme,
        "symbol_phase": symbol_phase or {
            symbol: str(theme_scores[theme]["phase"])
            for symbol, theme in symbol_theme.items()
        },
        "symbol_risk_flags": risk_flags or {symbol: [] for symbol in symbol_theme},
    }


def _state(symbols: list[str]) -> dict[str, dict[str, float]]:
    return {
        symbol: {
            "momentum_strength": 0.70,
            "breakout_readiness": 0.65,
            "volume_confirmation": 0.60,
            "trend_stability": 0.60,
            "fake_breakout_risk": 0.05,
            "max_drawdown_pct": 0.04,
        }
        for symbol in symbols
    }


def _context(
    symbols: list[str],
    rotation: dict[str, object] | None,
    markov: dict[str, object] | None = None,
) -> GlobalContext:
    metadata: dict[str, object] = {
        "symbol_market_state": _state(symbols),
        "markov_regime": markov or _markov(),
    }
    if rotation is not None:
        metadata["theme_rotation"] = rotation
    return GlobalContext(
        universe_symbols=symbols,
        universe_tiers={"researchable": symbols},
        liquidity_filter={"liquidity_scores": {symbol: 0.90 for symbol in symbols}},
        metadata=metadata,
    )


def test_trend_up_policy_admits_accumulation_early_acceleration_confirmed_rotation() -> None:
    policy = ThemeGatePolicy.from_markov(_markov("趋势上涨"), _config())

    assert policy.regime == "趋势上涨"
    assert policy.min_theme_score == pytest.approx(0.55)
    assert policy.min_symbol_score == pytest.approx(0.52)
    assert policy.top_themes == 10
    assert policy.allowed_phases == (
        "accumulation",
        "early_acceleration",
        "confirmed_rotation",
    )
    assert policy.residual_ratio >= 0.25
    assert policy.risk_watch_max_ratio >= 0.20


def test_high_vol_policy_reduces_pressure_without_blocking_risk_flags() -> None:
    policy = ThemeGatePolicy.from_markov(_markov("震荡高波"), _config())

    assert policy.min_theme_score == pytest.approx(0.65)
    assert policy.min_symbol_score == pytest.approx(0.60)
    assert policy.top_themes == 6
    assert policy.candidate_pressure < 1.0
    assert policy.risk_watch_max_ratio <= 0.12
    assert "theme_fake_breakout_risk" in policy.blocked_flags


def test_required_theme_pool_raises_when_rotation_missing_or_not_success() -> None:
    builder = ThemeCandidatePoolBuilder(_config(required=True))

    with pytest.raises(RuntimeError, match="theme_pool_required_but_theme_rotation_not_success"):
        builder.build(
            symbols=["A"],
            global_context=_context(["A"], None),
            quant_scores={"A": 0.5},
            max_candidates=5,
        )

    with pytest.raises(RuntimeError, match="theme_pool_required_but_theme_rotation_not_success"):
        builder.build(
            symbols=["A"],
            global_context=_context(["A"], {"status": "disabled"}),
            quant_scores={"A": 0.5},
            max_candidates=5,
        )


def test_theme_pool_required_excludes_unthemed_symbols() -> None:
    symbols = ["THEMED", "UNTHEMED_HIGH"]
    rotation = _rotation(
        theme_scores={"theme": _theme("theme", score=0.90)},
        symbol_theme={"THEMED": "theme"},
        symbol_scores={"THEMED": 0.80},
    )

    output = ThemeCandidatePoolBuilder(_config()).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("震荡低波")),
        quant_scores={"THEMED": 0.2, "UNTHEMED_HIGH": 0.99},
        max_candidates=5,
    )

    assert output.symbols == ["THEMED"]
    assert output.excluded_symbols["UNTHEMED_HIGH"] == "theme_pool_missing_theme_membership"
    assert output.metadata["unthemed_exclusion_count"] == 1
    assert output.metadata["symbols"]["UNTHEMED_HIGH"]["admitted"] is False


def test_secondary_membership_can_qualify_independent_of_primary_label() -> None:
    symbols = ["MULTI"]
    rotation = _rotation(
        theme_scores={
            "primary_cold": _theme("primary_cold", score=0.20),
            "secondary_hot": _theme("secondary_hot", score=0.92),
        },
        symbol_theme={"MULTI": "primary_cold"},
        symbol_scores={"MULTI": 0.20},
    )
    rotation["symbol_theme_memberships"] = {
        "MULTI": ["primary_cold", "secondary_hot"],
    }

    output = ThemeCandidatePoolBuilder(
        _config(use_markov_policy=False, min_admitted_themes=99)
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation),
        quant_scores={"MULTI": 0.5},
        max_candidates=5,
    )

    assert output.symbols == ["MULTI"]
    assert output.metadata["forced_theme_count"] == 0
    assert output.metadata["symbols"]["MULTI"]["primary_theme_id"] == "secondary_hot"


def test_protocol_v2_formal_kill_switch_fails_closed_even_with_forged_formal_pool() -> None:
    symbols = ["AI"]
    rotation = _rotation(
        theme_scores={"tech::ai": _theme("tech::ai", score=0.92)},
        symbol_theme={"AI": "tech::ai"},
    )
    rotation["protocol_v2"] = {
        "status": "formal",
        "formal_enabled": True,
        "formal_kill_switch": True,
        "protocol_hash": "a" * 64,
        "formal_pool": ["tech::ai"],
    }

    output = ThemeCandidatePoolBuilder(
        _config(protocol_v2_formal_enabled=True)
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation),
        quant_scores={"AI": 0.9},
        max_candidates=5,
    )

    assert output.symbols == []
    assert output.metadata["protocol_v2_formal_blocker"] == (
        "theme_v2_formal_kill_switch_active"
    )
    assert output.metadata["forced_theme_count"] == 0


def test_protocol_v2_candidate_stage_uses_prequalified_pool_without_claiming_final_formal() -> None:
    symbols = ["AI"]
    rotation = _rotation(
        theme_scores={"tech::ai": _theme("tech::ai", score=0.92)},
        symbol_theme={"AI": "tech::ai"},
    )
    rotation["protocol_v2"] = {
        "status": "prequalified",
        "formal_enabled": True,
        "formal_kill_switch": False,
        "protocol_hash": "a" * 64,
        "prequalified_pool": ["tech::ai"],
        "formal_pool": [],
    }

    output = ThemeCandidatePoolBuilder(
        _config(protocol_v2_formal_enabled=True)
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation),
        quant_scores={"AI": 0.9},
        max_candidates=5,
    )

    assert output.symbols == ["AI"]
    assert output.metadata["protocol_v2_gate_stage"] == (
        "prequalified_before_downstream"
    )
    assert output.metadata["protocol_v2_final_formal_pool"] == []
    assert output.metadata["protocol_v2_formal_blocker"] == ""


def test_protocol_v2_candidate_order_uses_adjusted_prequalified_rank_not_legacy_score() -> None:
    symbols = ["LEGACY_HIGH", "PEVC_HIGH"]
    rotation = _rotation(
        theme_scores={
            "tech::legacy": _theme("tech::legacy", score=0.99),
            "tech::pevc": _theme("tech::pevc", score=0.60),
        },
        symbol_theme={
            "LEGACY_HIGH": "tech::legacy",
            "PEVC_HIGH": "tech::pevc",
        },
    )
    rotation["protocol_v2"] = {
        "status": "prequalified",
        "formal_enabled": True,
        "formal_kill_switch": False,
        "protocol_hash": "a" * 64,
        "prequalified_pool": ["tech::legacy", "tech::pevc"],
        "formal_pool": [],
        "states": {
            "tech::legacy": {
                "base_rank_score": 0.90,
                "adjusted_percentile_rank": 0.10,
            },
            "tech::pevc": {
                "base_rank_score": 0.70,
                "adjusted_percentile_rank": 0.95,
            },
        },
    }

    output = ThemeCandidatePoolBuilder(
        _config(
            protocol_v2_formal_enabled=True,
            use_markov_policy=False,
            base_top_themes=1,
        )
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation),
        quant_scores={"LEGACY_HIGH": 1.0, "PEVC_HIGH": 0.0},
        max_candidates=5,
    )

    assert output.symbols == ["PEVC_HIGH"]
    assert output.metadata["admitted_themes"][0]["theme_id"] == "tech::pevc"
    assert output.metadata["symbols"]["PEVC_HIGH"]["candidate_intent"] == (
        "theme_v2_prequalified_research_candidate"
    )


def test_positive_minimum_theme_setting_cannot_force_formal_admission() -> None:
    symbols = ["SEMI1", "SEMI2", "BIO1", "PORT1"]
    rotation = _rotation(
        theme_scores={
            "semi": _theme(
                "semi",
                phase="overextended",
                score=0.90,
                risk_flags=["theme_overextended_no_chase"],
            ),
            "bio": _theme("bio", phase="distribution", score=0.82),
            "port": _theme("port", phase="accumulation", score=0.50),
        },
        symbol_theme={
            "SEMI1": "semi",
            "SEMI2": "semi",
            "BIO1": "bio",
            "PORT1": "port",
        },
        symbol_scores={
            "SEMI1": 0.30,
            "SEMI2": 0.28,
            "BIO1": 0.25,
            "PORT1": 0.90,
        },
        symbol_phase={
            "SEMI1": "overextended",
            "SEMI2": "overextended",
            "BIO1": "distribution",
            "PORT1": "accumulation",
        },
        risk_flags={
            "SEMI1": ["theme_overextended_no_chase"],
            "SEMI2": ["theme_overextended_no_chase"],
            "BIO1": ["theme_distribution_risk"],
            "PORT1": [],
        },
    )

    output = ThemeCandidatePoolBuilder(
        _config(
            use_markov_policy=False,
            base_top_themes=0,
            min_admitted_themes=2,
            risk_watch_max_ratio=1.0,
            min_residual_symbols=0,
        )
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("震荡高波")),
        quant_scores={symbol: 0.5 for symbol in symbols},
        max_candidates=10,
    )

    assert output.metadata["natural_admitted_theme_count"] == 0
    assert output.metadata["forced_theme_count"] == 0
    assert output.metadata["admitted_themes"] == []
    assert output.symbols == []
    assert set(output.excluded_symbols) == set(symbols)


def test_distribution_and_fake_breakout_are_candidates_with_risk_watch_metadata() -> None:
    symbols = ["DIST", "FAKE", "CORE"]
    rotation = _rotation(
        theme_scores={
            "theme": _theme("theme", phase="confirmed_rotation", score=0.90),
        },
        symbol_theme={symbol: "theme" for symbol in symbols},
        symbol_scores={"DIST": 0.85, "FAKE": 0.84, "CORE": 0.90},
        symbol_phase={
            "DIST": "distribution",
            "FAKE": "confirmed_rotation",
            "CORE": "confirmed_rotation",
        },
        risk_flags={
            "DIST": ["theme_distribution_risk"],
            "FAKE": ["theme_fake_breakout_risk"],
            "CORE": [],
        },
    )

    output = ThemeCandidatePoolBuilder(_config(risk_watch_max_ratio=1.0)).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("震荡低波")),
        quant_scores={symbol: 0.5 for symbol in symbols},
        max_candidates=10,
    )

    assert {"DIST", "FAKE", "CORE"} == set(output.symbols)
    assert output.metadata["symbols"]["DIST"]["bucket"] == "risk_watch_distribution"
    assert output.metadata["symbols"]["FAKE"]["bucket"] == "risk_watch_fake_breakout"
    assert output.metadata["symbols"]["DIST"]["source"] == "risk_watch"
    assert output.metadata["symbols"]["FAKE"]["score_penalty"] > 0
    assert output.metadata["risk_watch_symbol_count"] == 2


def test_hard_filter_excludes_theme_residual_and_unthemed_quant_escape() -> None:
    symbols = ["CORE", "RESIDUAL", "UNTHEMED_HIGH"]
    rotation = _rotation(
        theme_scores={
            "core": _theme("core", score=0.92),
            "tail": _theme("tail", score=0.30),
        },
        symbol_theme={"CORE": "core", "RESIDUAL": "tail"},
        symbol_scores={"CORE": 0.90, "RESIDUAL": 0.40},
    )

    output = ThemeCandidatePoolBuilder(
        _config(
            use_markov_policy=False,
            base_top_themes=1,
            min_admitted_themes=1,
            residual_ratio=0.50,
            min_residual_symbols=1,
        )
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("趋势上涨")),
        quant_scores={"CORE": 0.2, "RESIDUAL": 0.4, "UNTHEMED_HIGH": 0.99},
        max_candidates=3,
    )

    assert "CORE" in output.symbols
    assert "RESIDUAL" not in output.symbols
    assert "UNTHEMED_HIGH" not in output.symbols
    assert output.symbols == ["CORE"]
    assert output.metadata["residual_symbol_count"] == 0
    assert output.metadata["symbols"]["RESIDUAL"]["admitted"] is False
    assert output.metadata["symbols"]["RESIDUAL"]["source"] == "none"
    assert output.excluded_symbols["RESIDUAL"] == "theme_pool_theme_not_admitted"
    assert output.excluded_symbols["UNTHEMED_HIGH"] == "theme_pool_missing_theme_membership"


def test_hard_filter_excludes_non_admitted_theme_residuals() -> None:
    symbols = ["CORE", "TAIL", "UNTHEMED_HIGH"]
    rotation = _rotation(
        theme_scores={
            "core": _theme("core", score=0.92),
            "tail": _theme("tail", score=0.88),
        },
        symbol_theme={"CORE": "core", "TAIL": "tail"},
        symbol_scores={"CORE": 0.90, "TAIL": 0.91},
    )

    output = ThemeCandidatePoolBuilder(
        _config(
            use_markov_policy=False,
            base_top_themes=1,
            min_admitted_themes=1,
            residual_ratio=1.00,
            min_residual_symbols=2,
        )
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("趋势上涨")),
        quant_scores={"CORE": 0.2, "TAIL": 0.99, "UNTHEMED_HIGH": 1.0},
        max_candidates=3,
    )

    assert output.symbols == ["CORE"]
    assert output.metadata["core_symbol_count"] == 1
    assert output.metadata["residual_symbol_count"] == 0
    assert output.metadata["residual_theme_alpha_candidates"] == []
    assert output.excluded_symbols["TAIL"] == "theme_pool_theme_not_admitted"
    assert output.excluded_symbols["UNTHEMED_HIGH"] == "theme_pool_missing_theme_membership"
    assert output.metadata["symbols"]["TAIL"]["admitted"] is False
    assert output.metadata["symbols"]["TAIL"]["source"] == "none"


def test_hard_filter_residual_zero_upper_bound_is_locked_in_metadata() -> None:
    symbols = ["CORE", "TAIL_A", "TAIL_B"]
    rotation = _rotation(
        theme_scores={
            "core": _theme("core", score=0.92),
            "tail": _theme("tail", score=0.89),
        },
        symbol_theme={"CORE": "core", "TAIL_A": "tail", "TAIL_B": "tail"},
        symbol_scores={"CORE": 0.90, "TAIL_A": 0.95, "TAIL_B": 0.94},
    )

    output = ThemeCandidatePoolBuilder(
        _config(
            use_markov_policy=False,
            base_top_themes=1,
            min_admitted_themes=1,
            residual_ratio=1.00,
            min_residual_symbols=2,
        )
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("趋势上涨")),
        quant_scores={"CORE": 0.2, "TAIL_A": 0.99, "TAIL_B": 0.98},
        max_candidates=3,
    )

    assert output.metadata["policy"]["hard_theme_constraint"] is True
    assert output.metadata["policy"]["residual_enabled"] is False
    assert output.metadata["policy"]["residual_concept"] == "disabled_by_theme_pool_hard_filter"
    assert output.metadata["residual_symbol_count"] == 0
    assert output.metadata["source_counts"].get("residual_theme", 0) == 0
    assert output.metadata["residual_theme_alpha_candidates"] == []
    assert all(source != "residual_theme" for source in output.symbol_sources.values())


def test_zero_natural_pass_keeps_formal_pool_empty_even_when_legacy_minimum_is_positive() -> None:
    symbols = ["SEMI1", "BIO1", "TAIL1"]
    rotation = _rotation(
        theme_scores={
            "semi": _theme("semi", score=0.60, phase="accumulation"),
            "bio": _theme("bio", score=0.58, phase="accumulation"),
            "tail": _theme("tail", score=0.57, phase="accumulation"),
        },
        symbol_theme={
            "SEMI1": "semi",
            "BIO1": "bio",
            "TAIL1": "tail",
        },
        symbol_scores={
            "SEMI1": 0.30,
            "BIO1": 0.28,
            "TAIL1": 0.95,
        },
        symbol_phase={
            "SEMI1": "accumulation",
            "BIO1": "accumulation",
            "TAIL1": "accumulation",
        },
    )

    output = ThemeCandidatePoolBuilder(
        _config(
            use_markov_policy=False,
            base_min_theme_score=0.70,
            base_min_symbol_score=0.65,
            base_top_themes=3,
            min_admitted_themes=2,
            residual_ratio=1.00,
            min_residual_symbols=2,
            allowed_phases=("confirmed_rotation",),
        )
    ).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("趋势下跌")),
        quant_scores={symbol: 0.5 for symbol in symbols},
        max_candidates=5,
    )

    assert output.metadata["natural_admitted_theme_count"] == 0
    assert output.metadata["forced_theme_count"] == 0
    assert output.metadata["core_symbol_count"] == 0
    assert output.metadata["residual_symbol_count"] == 0
    assert output.symbols == []
    assert output.metadata["admitted_themes"] == []
    assert output.excluded_symbols["TAIL1"] == "theme_pool_theme_not_admitted"


def test_risk_watch_ratio_cap_is_deterministic() -> None:
    symbols = ["CORE", "RISK_B", "RISK_A", "RISK_C"]
    rotation = _rotation(
        theme_scores={"theme": _theme("theme", score=0.90)},
        symbol_theme={symbol: "theme" for symbol in symbols},
        symbol_scores={symbol: 0.85 for symbol in symbols},
        symbol_phase={
            "CORE": "confirmed_rotation",
            "RISK_A": "distribution",
            "RISK_B": "distribution",
            "RISK_C": "distribution",
        },
        risk_flags={
            "CORE": [],
            "RISK_A": ["theme_distribution_risk"],
            "RISK_B": ["theme_distribution_risk"],
            "RISK_C": ["theme_distribution_risk"],
        },
    )

    output = ThemeCandidatePoolBuilder(_config(risk_watch_max_ratio=0.20)).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("震荡低波")),
        quant_scores={symbol: 0.5 for symbol in symbols},
        max_candidates=5,
    )

    selected_risk_watch = [
        symbol
        for symbol in output.symbols
        if output.metadata["symbols"][symbol]["source"] == "risk_watch"
    ]
    assert selected_risk_watch == ["RISK_A"]
    assert output.excluded_symbols["RISK_B"] == "theme_pool_risk_watch_ratio_cutoff"
    assert output.excluded_symbols["RISK_C"] == "theme_pool_risk_watch_ratio_cutoff"
    assert output.metadata["risk_watch_limit"] == 1


def test_smoothed_score_source_falls_back_to_raw_when_enabled() -> None:
    symbols = ["RAW_ONLY"]
    rotation = _rotation(
        theme_scores={"theme": _theme("theme", phase="confirmed_rotation", score=0.90)},
        symbol_theme={"RAW_ONLY": "theme"},
        symbol_scores={"RAW_ONLY": 0.90},
        smoothed_scores={},
    )

    output = ThemeCandidatePoolBuilder(_config(fallback_to_raw_score=True)).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("震荡低波")),
        quant_scores={"RAW_ONLY": 0.5},
        max_candidates=5,
    )

    assert output.symbols == ["RAW_ONLY"]
    assert output.metadata["symbols"]["RAW_ONLY"]["symbol_theme_score"] == pytest.approx(0.90)
    assert output.metadata["score_source"] == "smoothed"
    assert output.metadata["fallback_to_raw_score"] is True


def test_transition_risk_tightens_policy() -> None:
    policy = ThemeGatePolicy.from_markov(
        _markov("趋势上涨", transition_risk=0.70),
        _config(),
    )

    assert policy.min_theme_score == pytest.approx(0.60)
    assert policy.min_symbol_score == pytest.approx(0.55)
    assert policy.top_themes == 6
    assert policy.residual_ratio <= 0.15
    assert policy.risk_watch_max_ratio <= 0.10
    assert policy.candidate_pressure < 1.0


def test_theme_pool_min_admitted_themes_is_env_backed(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_investor.config as config_module

    monkeypatch.setenv("THEME_POOL_MIN_ADMITTED_THEMES", "4")
    reloaded = importlib.reload(config_module)
    assert reloaded.MAINLINE_ENV_DEFAULTS["THEME_POOL_MIN_ADMITTED_THEMES"] == "0"
    assert reloaded.Config.THEME_POOL_MIN_ADMITTED_THEMES == 4
    monkeypatch.delenv("THEME_POOL_MIN_ADMITTED_THEMES")
    importlib.reload(config_module)


def test_markov_high_vol_reduces_risk_watch_ratio_and_candidate_pressure_without_hard_excluding_theme_members() -> None:
    symbols = ["CORE", "RISKY", "TAIL"]
    rotation = _rotation(
        theme_scores={
            "core": _theme("core", score=0.92),
            "tail": _theme("tail", score=0.45),
        },
        symbol_theme={"CORE": "core", "RISKY": "core", "TAIL": "tail"},
        symbol_scores={"CORE": 0.90, "RISKY": 0.88, "TAIL": 0.60},
        symbol_phase={
            "CORE": "confirmed_rotation",
            "RISKY": "confirmed_rotation",
            "TAIL": "accumulation",
        },
        risk_flags={
            "CORE": [],
            "RISKY": ["theme_fake_breakout_risk"],
            "TAIL": [],
        },
    )

    output = ThemeCandidatePoolBuilder(_config()).build(
        symbols=symbols,
        global_context=_context(symbols, rotation, _markov("震荡高波")),
        quant_scores={symbol: 0.5 for symbol in symbols},
        max_candidates=10,
    )

    assert output.metadata["policy"]["candidate_pressure"] < 1.0
    assert output.metadata["policy"]["risk_watch_max_ratio"] <= 0.12
    assert output.metadata["effective_max_candidates"] < 10
    assert "RISKY" in output.symbols
    assert output.metadata["symbols"]["RISKY"]["bucket"] == "risk_watch_fake_breakout"
    assert output.metadata["symbols"]["RISKY"]["risk_flags"] == ["theme_fake_breakout_risk"]
