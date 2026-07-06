from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pandas as pd

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.market.dag.theme_context import build_theme_rotation_metadata
from quant_investor.themes import ThemeScanner


REPO_ROOT = Path(__file__).resolve().parents[2]


def _frame(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    data: dict[str, object] = {
        "trade_date": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        "close": closes,
    }
    if volumes is not None:
        data["volume"] = volumes
    return pd.DataFrame(data)


def _trend(start: float, end: float, periods: int = 30) -> list[float]:
    step = (end - start) / max(periods - 1, 1)
    return [start + step * idx for idx in range(periods)]


def _theme_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    industry_map: dict[str, str] = {}
    for idx in range(6):
        symbol = f"SEMI{idx:03d}.SZ"
        frames[symbol] = _frame(_trend(10.0, 11.5), _trend(1000.0, 2500.0))
        industry_map[symbol] = "Semiconductor"
    return frames, industry_map


def _write_policy(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "event_id": "policy-1",
        "title": "Semiconductor pilot procurement policy",
        "issuer": "State Council",
        "publish_date": "2026-06-01",
        "effective_date": "2026-06-01",
        "policy_level": "central",
        "policy_type": "funding pilot procurement standard",
        "theme_tags": ["Semiconductor"],
        "industry_tags": [],
        "symbol_tags": ["SEMI000.SZ"],
        "evidence_text": "Local fixture evidence for policy catalyst scanner integration.",
        "source_url": "local://fixture",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_policy_disabled_keeps_theme_scores_unchanged(tmp_path: Path) -> None:
    frames, industry_map = _theme_inputs()
    policy_path = _write_policy(tmp_path / "policy.jsonl")
    baseline = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
        policy_catalyst_enabled=False,
    )
    disabled = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
        policy_catalyst_enabled=False,
        policy_event_path=str(policy_path),
    )

    theme_id = "industry::semiconductor"
    assert disabled.metadata["policy_catalyst_status"] == "disabled"
    assert disabled.theme_scores[theme_id].score == baseline.theme_scores[theme_id].score
    assert disabled.symbol_scores == baseline.symbol_scores
    assert disabled.theme_scores[theme_id].policy_stage == "disabled"


def test_policy_file_missing_is_safe_unavailable_fallback(tmp_path: Path) -> None:
    frames, industry_map = _theme_inputs()
    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
        policy_catalyst_enabled=True,
        policy_event_path=str(tmp_path / "missing.jsonl"),
    )

    theme = result.theme_scores["industry::semiconductor"]
    assert result.metadata["policy_catalyst_status"] == "unavailable"
    assert "policy_event_file_missing" in result.metadata["policy_catalyst_diagnostic_notes"]
    assert theme.policy_stage == "unavailable"
    assert theme.policy_catalyst_score == 0.0


def test_policy_file_malformed_is_safe_unavailable_fallback(tmp_path: Path) -> None:
    frames, industry_map = _theme_inputs()
    malformed_path = tmp_path / "malformed.jsonl"
    malformed_path.write_text("{not valid json", encoding="utf-8")

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
        policy_catalyst_enabled=True,
        policy_event_path=str(malformed_path),
    )

    theme = result.theme_scores["industry::semiconductor"]
    assert result.metadata["policy_catalyst_status"] == "unavailable"
    assert "policy_event_file_format_error" in " ".join(
        result.metadata["policy_catalyst_diagnostic_notes"]
    )
    assert theme.policy_stage == "unavailable"
    assert theme.policy_catalyst_score == 0.0


def test_policy_enabled_boosts_theme_score_with_weight_cap(tmp_path: Path) -> None:
    frames, industry_map = _theme_inputs()
    policy_path = _write_policy(tmp_path / "policy.jsonl")
    baseline = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
        policy_catalyst_enabled=False,
    )
    boosted = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
        as_of="2026-06-15",
        policy_catalyst_enabled=True,
        policy_catalyst_weight=0.16,
        policy_lookback_days=30,
        policy_event_path=str(policy_path),
    )

    theme_id = "industry::semiconductor"
    base_score = baseline.theme_scores[theme_id].score
    boosted_score = boosted.theme_scores[theme_id].score
    assert boosted.metadata["policy_catalyst_status"] == "success"
    assert boosted.metadata["policy_catalyst_matched_theme_count"] == 1
    assert boosted_score > base_score
    assert boosted_score <= base_score + 16.0 + 1e-9
    assert boosted.theme_scores[theme_id].policy_stage == "active_catalyst"
    assert boosted.theme_scores[theme_id].policy_evidence


def test_policy_metadata_appears_in_theme_rotation_v1_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames, industry_map = _theme_inputs()
    policy_path = _write_policy(tmp_path / "policy.jsonl")
    config_module = importlib.import_module("quant_investor.config")
    monkeypatch.setattr(config_module.Config, "THEME_POLICY_CATALYST_ENABLED", True)
    monkeypatch.setattr(config_module.Config, "THEME_POLICY_CATALYST_WEIGHT", 0.16)
    monkeypatch.setattr(config_module.Config, "THEME_POLICY_LOOKBACK_DAYS", 30)
    monkeypatch.setattr(config_module.Config, "THEME_POLICY_EVENT_PATH", str(policy_path))

    payload = build_theme_rotation_metadata(
        frames=frames,
        industry_map=industry_map,
        symbol_market_state={},
        market="CN",
        universe_key="unit",
        as_of="2026-06-15",
        min_member_count=5,
    )

    theme = payload["theme_scores"]["industry::semiconductor"]
    assert payload["schema_version"] == "theme_rotation.v1"
    assert payload["metadata"]["policy_catalyst_status"] == "success"
    assert theme["policy_catalyst_score"] > 0.0
    assert theme["policy_stage"] == "active_catalyst"
    assert theme["policy_evidence"]


def test_policy_risk_flags_enter_theme_score_only(tmp_path: Path) -> None:
    frames, industry_map = _theme_inputs()
    policy_path = _write_policy(
        tmp_path / "policy.jsonl",
        issuer="Industry Association",
        policy_level="association",
        policy_type="guidance",
        theme_tags=[],
        industry_tags=["Semiconductor"],
        symbol_tags=[],
    )
    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=5,
        as_of="2026-06-15",
        policy_catalyst_enabled=True,
        policy_event_path=str(policy_path),
    )

    theme = result.theme_scores["industry::semiconductor"]
    assert "policy_weak_authority" in theme.policy_risk_flags
    assert "policy_weak_authority" in theme.risk_flags
    assert "candidate_pool" not in result.metadata


def test_policy_layer_has_no_network_imports_or_canonical_wiring() -> None:
    policy_source = (REPO_ROOT / "quant_investor" / "themes" / "policy.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(policy_source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots.isdisjoint({"aiohttp", "httpx", "requests", "tushare", "urllib", "yfinance"})
    assert "theme" not in CANONICAL_BRANCH_ORDER
    for path in (REPO_ROOT / "quant_investor" / "bayesian").rglob("*.py"):
        assert "theme_likelihood" not in path.read_text(encoding="utf-8")
