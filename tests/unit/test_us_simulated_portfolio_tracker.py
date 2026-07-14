"""US simulated portfolio v14 branch-support contract tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.market.full_report import (
    CURRENT_MARKET_REPORT_SCHEMA_ENVELOPE,
    MarketArtifactContractError,
)
from quant_investor.monitoring import us_simulated_portfolio_tracker as tracker


def _current_us_batch(**overrides):
    payload = {
        **CURRENT_MARKET_REPORT_SCHEMA_ENVELOPE,
        "category": "large_cap",
        "batch_id": 1,
        "timestamp": "20260714_120000",
        "stocks": ["EOG"],
        "stock_count": 1,
        "branches": {name: {"score": 0.1, "confidence": 0.6} for name in CANONICAL_BRANCH_ORDER},
        "recommendations": [
            {
                "symbol": "EOG",
                "consensus_score": 0.3,
                "branch_positive_count": 3,
            }
        ],
        "analysis_meta": {
            **CURRENT_MARKET_REPORT_SCHEMA_ENVELOPE,
            "market": "US",
            "universe": "full_us",
        },
    }
    payload.update(overrides)
    return payload


def _write_us_batch(project_root: Path, name: str, payload: dict) -> Path:
    output_dir = project_root / "results" / "v14" / "us_analysis_full"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_us_tracker_requires_unanimous_current_branch_support_for_add_on_buy():
    positions = pd.DataFrame([{"symbol": "CVX", "shares": 1}])
    prices = {"CVX": {"close": 100.0}}
    caps = {"CVX": 2}

    rejected_orders, _ = tracker._generate_trade_plan(
        positions,
        9_900.0,
        prices,
        {"CVX": {"consensus_score": 0.3, "branch_positive_count": 2}},
        caps,
        "均衡偏防御",
    )
    accepted_orders, _ = tracker._generate_trade_plan(
        positions,
        9_900.0,
        prices,
        {"CVX": {"consensus_score": 0.3, "branch_positive_count": 3}},
        caps,
        "均衡偏防御",
    )

    assert tracker.BRANCH_SUPPORT_DENOMINATOR == len(CANONICAL_BRANCH_ORDER) == 3
    assert tracker.REQUIRED_BUY_BRANCH_SUPPORT == len(CANONICAL_BRANCH_ORDER)
    assert rejected_orders == []
    assert len(accepted_orders) == 1
    assert "3/3 一致支持" in accepted_orders[0].reason


def test_us_tracker_has_no_retired_five_branch_denominator_text():
    source = Path(tracker.__file__).read_text(encoding="utf-8")

    assert "3/5" not in source


def test_us_tracker_requires_unanimous_support_for_new_symbol_buy():
    positions = pd.DataFrame([{"symbol": "CVX", "shares": 1}])
    prices = {
        "CVX": {"close": 100.0},
        "EOG": {"close": 100.0},
    }
    caps = {"CVX": 1}

    rejected_orders, _ = tracker._generate_trade_plan(
        positions,
        9_900.0,
        prices,
        {"EOG": {"consensus_score": 0.3, "branch_positive_count": 2}},
        caps,
        "均衡偏防御",
    )
    accepted_orders, _ = tracker._generate_trade_plan(
        positions,
        9_900.0,
        prices,
        {"EOG": {"consensus_score": 0.3, "branch_positive_count": 3}},
        caps,
        "均衡偏防御",
    )

    assert rejected_orders == []
    assert len(accepted_orders) == 1
    assert accepted_orders[0].symbol == "EOG"
    assert "3/3 一致支持" in accepted_orders[0].reason


def test_us_recommendation_loader_accepts_only_current_v14_batch(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(tracker, "PROJECT_ROOT", tmp_path)
    _write_us_batch(
        tmp_path,
        "batch_large_cap_001_20260714_120000.json",
        _current_us_batch(),
    )

    recommendations = tracker._load_latest_batch_recommendations()

    assert recommendations["EOG"]["branch_positive_count"] == 3
    assert recommendations["EOG"]["source_batch"] == ("batch_large_cap_001_20260714_120000.json")


def test_us_recommendation_loader_does_not_scan_legacy_namespace(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(tracker, "PROJECT_ROOT", tmp_path)
    legacy_dir = tmp_path / "results" / "us_analysis_full"
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / "batch_large_cap_001_20260713_120000.json"
    legacy_path.write_text(
        json.dumps(_current_us_batch()),
        encoding="utf-8",
    )

    assert tracker._load_latest_batch_recommendations() == {}


def test_us_recommendation_loader_rejects_malformed_batch_filename(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(tracker, "PROJECT_ROOT", tmp_path)
    _write_us_batch(
        tmp_path,
        "batch_legacy.json",
        _current_us_batch(),
    )

    with pytest.raises(
        MarketArtifactContractError,
        match="filename",
    ):
        tracker._load_latest_batch_recommendations()


def test_old_three_of_five_batch_is_blocked_before_it_can_trigger_buy(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(tracker, "PROJECT_ROOT", tmp_path)
    old_batch = _current_us_batch(
        architecture_version="13.0.0-stable",
        branch_schema_version="branch-schema.v13.four-branch",
    )
    old_batch["branches"]["intelligence"] = {
        "score": 0.9,
        "confidence": 0.9,
    }
    _write_us_batch(
        tmp_path,
        "batch_large_cap_001_20260713_120000.json",
        old_batch,
    )
    _write_us_batch(
        tmp_path,
        "batch_large_cap_001_20260714_120000.json",
        _current_us_batch(),
    )

    with pytest.raises(
        MarketArtifactContractError,
        match="not a current v14 batch",
    ):
        tracker._load_latest_batch_recommendations()
