from __future__ import annotations

import importlib
import json

import pytest

from quant_investor.regime.engine import MarkovRegimeEngine
from quant_investor.regime.persistence import append_regime_signal, load_regime_history_result
from quant_investor.regime.scope import build_regime_scope
from quant_investor.regime.types import REGIME_TREND_UP, RegimeSignal


def _scope(**overrides: object) -> dict[str, object]:
    payload = build_regime_scope(
        market="CN",
        base_universe_key="full_a",
        source_universe_key="full_a",
        requested_symbol_count=50,
        source_symbol_count=50,
        explicit_symbol_count=0,
        unsampled_symbol_count=50,
        sampled=False,
        min_market_sample=30,
        source_description="fixture",
    ).to_dict()
    payload.update(overrides)
    return payload


def _signal(**overrides: object) -> RegimeSignal:
    scope_payload = _scope()
    scope_payload.pop("diagnostics", None)
    payload: dict[str, object] = {
        "as_of": "20260625",
        "market": "CN",
        "universe_key": "full_a",
        "dominant_regime": REGIME_TREND_UP,
        "probabilities": {REGIME_TREND_UP: 1.0},
        "transition_matrix": {},
        "confidence": 1.0,
        "transition_risk": 0.0,
        "risk_on_score": 0.9,
        "volatility_score": 0.1,
        "pressure_score": 0.1,
        "suggested_gross_exposure_cap": 0.60,
        "suggested_max_single_weight": 0.50,
        "turnover_cap": None,
        "feature_snapshot": {},
        "diagnostic_notes": [],
        **scope_payload,
    }
    payload.update(overrides)
    payload.pop("diagnostics", None)
    return RegimeSignal(**payload)  # type: ignore[arg-type]


def _engine_kwargs(history_path: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "market": "CN",
        "universe_key": "full_a",
        "as_of": "20260625",
        "frames": {},
        "tradability_snapshot": {
            f"{idx:06d}.SZ": {
                "market_state": {
                    "momentum_strength": 0.70,
                    "breakout_readiness": 0.65,
                    "fake_breakout_risk": 0.10,
                    "max_drawdown_pct": 0.05,
                    "liquidity_score": 0.80,
                    "volume_confirmation": 0.60,
                }
            }
            for idx in range(50)
        },
        "cross_section_quant": {
            "average_return": 0.015,
            "average_volatility": 0.015,
            "breadth": 0.70,
            "sample_count": 50,
        },
        "macro_verdict": {"final_score": 0.20, "metadata": {"target_gross_exposure": 0.70}},
        "market_snapshot": {},
        "scope": _scope(),
    }
    payload.update(overrides)
    return payload


def test_missing_history_file_returns_diagnostic(tmp_path) -> None:
    result = load_regime_history_result(
        tmp_path / "missing.jsonl",
        market="CN",
        universe_key="full_a",
        scope_key=str(_scope()["scope_key"]),
        source_universe_key="full_a",
    )

    assert result.records == []
    assert "regime_history_missing" in result.diagnostics


def test_malformed_jsonl_line_is_ignored(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    record = _signal().to_dict()
    path.write_text("not-json\n" + json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    result = load_regime_history_result(
        path,
        market="CN",
        universe_key="full_a",
        before_or_equal_as_of="20260625",
        scope_key=str(record["scope_key"]),
        source_universe_key="full_a",
    )

    assert len(result.records) == 1
    assert any(note.startswith("regime_history_malformed_line_ignored") for note in result.diagnostics)


def test_duplicate_same_day_run_rewrites_one_effective_record(tmp_path) -> None:
    history_path = tmp_path / "history.jsonl"
    engine = MarkovRegimeEngine(history_path=str(history_path))
    kwargs = _engine_kwargs(str(history_path))

    first = engine.run(**kwargs)  # type: ignore[arg-type]
    second = engine.run(**kwargs)  # type: ignore[arg-type]

    lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert first.as_of == second.as_of == "20260625"
    assert "regime_persistence_replaced_existing_record" in second.diagnostic_notes


def test_future_dated_history_is_ignored(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    scope = _scope()
    old_record = _signal(as_of="20260624", **scope).to_dict()
    future_record = _signal(as_of="20260626", **scope).to_dict()
    path.write_text(
        json.dumps(old_record, ensure_ascii=False) + "\n"
        + json.dumps(future_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = load_regime_history_result(
        path,
        market="CN",
        universe_key="full_a",
        before_or_equal_as_of="20260625",
        scope_key=str(scope["scope_key"]),
        source_universe_key="full_a",
    )

    assert [record["as_of"] for record in result.records] == ["20260624"]


def test_different_scope_history_is_ignored(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    full_scope = _scope()
    subset_scope = _scope(
        scope_key="CN:subset:custom:symbols_3",
        source_universe_key="technology_subset",
    )
    path.write_text(
        json.dumps(_signal(as_of="20260624", **full_scope).to_dict(), ensure_ascii=False) + "\n"
        + json.dumps(_signal(as_of="20260624", **subset_scope).to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = load_regime_history_result(
        path,
        market="CN",
        universe_key="full_a",
        before_or_equal_as_of="20260625",
        scope_key=str(full_scope["scope_key"]),
        source_universe_key="full_a",
    )

    assert len(result.records) == 1
    assert result.records[0]["scope_key"] == full_scope["scope_key"]


def test_legacy_ambiguous_records_are_not_used_for_scoped_production(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    legacy = _signal().to_dict()
    legacy.pop("scope_key", None)
    legacy.pop("source_universe_key", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")

    result = load_regime_history_result(
        path,
        market="CN",
        universe_key="full_a",
        before_or_equal_as_of="20260625",
        scope_key=str(_scope()["scope_key"]),
        source_universe_key="full_a",
    )

    assert result.records == []
    assert "legacy_ambiguous_regime_history_ignored" in result.diagnostics


def test_legacy_count_scoped_full_market_record_matches_stable_scope(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    legacy = _signal(
        as_of="20260624",
        scope_key="CN:full_market:full_a:symbols_5199",
        source_symbol_count=5199,
        unsampled_symbol_count=5199,
    ).to_dict()
    path.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")

    result = load_regime_history_result(
        path,
        market="CN",
        universe_key="full_a",
        before_or_equal_as_of="20260625",
        scope_key="CN:full_market:full_a",
        source_universe_key="full_a",
    )

    assert result.records == [legacy]


def test_append_replaces_same_day_legacy_count_scoped_full_market_record(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    legacy = _signal(scope_key="CN:full_market:full_a:symbols_5199").to_dict()
    path.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")

    notes = append_regime_signal(path, _signal())

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["scope_key"] == "CN:full_market:full_a"
    assert "regime_persistence_replaced_existing_record" in notes


def test_write_failure_is_reported_without_crashing_engine(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def _fail_append(path: object, signal: object) -> list[str]:
        return ["regime_persistence_write_failed:fixture"]

    monkeypatch.setattr("quant_investor.regime.engine.append_regime_signal", _fail_append)
    engine = MarkovRegimeEngine(history_path=str(tmp_path / "history.jsonl"))

    signal = engine.run(**_engine_kwargs(str(tmp_path / "history.jsonl")))  # type: ignore[arg-type]

    assert "regime_persistence_write_failed:fixture" in signal.diagnostic_notes


def test_append_collapses_existing_duplicate_records(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    append_regime_signal(path, _signal())
    notes = append_regime_signal(path, _signal())

    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert "regime_persistence_replaced_existing_record" in notes


def test_importing_persistence_module_does_not_touch_filesystem(tmp_path) -> None:
    before = sorted(tmp_path.iterdir())
    import quant_investor.regime.persistence as persistence

    importlib.reload(persistence)

    assert sorted(tmp_path.iterdir()) == before
