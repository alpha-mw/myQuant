from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_investor.market.dag.theme_context import build_theme_rotation_metadata
from quant_investor.themes import ThemeScanner
from quant_investor.themes.replay import extract_snapshot_theme_rows


def _frame(start: float, end: float, periods: int = 30) -> pd.DataFrame:
    if periods <= 1:
        closes = [end]
    else:
        step = (end - start) / (periods - 1)
        closes = [start + step * idx for idx in range(periods)]
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
            "close": closes,
            "vol": [1000.0 + idx for idx in range(len(closes))],
            "amount": [100000.0 + idx for idx in range(len(closes))],
        }
    )


def _membership(symbol: str, *, theme_name: str = "Low-altitude Economy") -> dict[str, object]:
    return {
        "schema_version": "theme_membership.v1",
        "membership_id": f"low-altitude-{symbol}",
        "theme_id": "concept::low-altitude-economy",
        "theme_name": theme_name,
        "theme_type": "concept",
        "symbol": symbol,
        "effective_from": "2026-01-01",
        "effective_to": "",
        "confidence": 0.8,
        "source_type": "manual_review",
    }


def test_concept_memberships_are_default_off_even_when_provided():
    frames = {"000001.SZ": _frame(10.0, 11.0), "000002.SZ": _frame(10.0, 12.0)}
    industry_map = {"000001.SZ": "Machinery", "000002.SZ": "Aviation"}

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=1,
        concept_membership_enabled=False,
        theme_memberships=[_membership("000001.SZ"), _membership("000002.SZ")],
    )

    payload = result.to_dict()
    assert "concept::low-altitude-economy" not in payload["theme_scores"]
    assert payload["metadata"]["concept_membership_status"] == "disabled"
    assert payload["metadata"]["concept_membership_count"] == 0
    assert payload["symbol_theme_memberships"] == {
        "000001.SZ": ["industry::machinery"],
        "000002.SZ": ["industry::aviation"],
    }
    json.dumps(payload)


def test_concept_memberships_add_pit_theme_and_can_become_primary():
    frames = {
        "000001.SZ": _frame(10.0, 11.0),
        "000002.SZ": _frame(10.0, 13.0),
    }
    industry_map = {"000001.SZ": "Machinery", "000002.SZ": "Aviation"}

    result = ThemeScanner().scan(
        frames=frames,
        industry_map=industry_map,
        min_member_count=1,
        concept_membership_enabled=True,
        concept_primary_margin=0.01,
        theme_memberships=[_membership("000001.SZ"), _membership("000002.SZ")],
        as_of="20260115",
    )
    payload = result.to_dict()

    concept = payload["theme_scores"]["concept::low-altitude-economy"]
    assert concept["theme_type"] == "concept"
    assert concept["membership_source"] == "theme_membership.v2"
    assert concept["pit_membership"] is True
    assert payload["symbol_theme_memberships"]["000001.SZ"] == [
        "industry::machinery",
        "concept::low-altitude-economy",
    ]
    assert payload["symbol_primary_theme"]["000001.SZ"] == "concept::low-altitude-economy"
    assert payload["metadata"]["concept_membership_status"] == "success"


def test_theme_context_can_load_concept_memberships_from_jsonl(tmp_path: Path):
    membership_path = tmp_path / "theme_membership.jsonl"
    membership_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in [_membership("000001.SZ"), _membership("000002.SZ")]
        )
        + "\n",
        encoding="utf-8",
    )
    frames = {"000001.SZ": _frame(10.0, 11.0), "000002.SZ": _frame(10.0, 13.0)}
    industry_map = {"000001.SZ": "Machinery", "000002.SZ": "Aviation"}

    payload = build_theme_rotation_metadata(
        frames=frames,
        industry_map=industry_map,
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="20260115",
        min_member_count=1,
        concept_membership_enabled=True,
        concept_membership_path=membership_path,
        concept_primary_margin=0.01,
    )

    assert payload["status"] == "success"
    assert "concept::low-altitude-economy" in payload["theme_scores"]
    assert payload["symbol_theme_memberships"]["000001.SZ"][-1] == (
        "concept::low-altitude-economy"
    )
    assert payload["metadata"]["concept_membership_status"] == "success"


def test_replay_rows_preserve_concept_membership_metadata():
    rows = extract_snapshot_theme_rows(
        {
            "theme_rotation": {
                "symbol_scores": {"000001.SZ": 0.8},
                "symbol_primary_theme": {"000001.SZ": "concept::low-altitude-economy"},
                "symbol_phase": {"000001.SZ": "confirmed_rotation"},
                "symbol_risk_flags": {"000001.SZ": []},
                "symbol_theme_memberships": {
                    "000001.SZ": [
                        "industry::machinery",
                        "concept::low-altitude-economy",
                    ]
                },
                "theme_scores": {
                    "concept::low-altitude-economy": {
                        "theme_name": "Low-altitude Economy",
                        "score": 80.0,
                        "confidence": 0.7,
                        "member_count": 8,
                        "theme_type": "concept",
                        "membership_source": "theme_membership.v1",
                        "pit_membership": True,
                    }
                },
            }
        }
    )

    assert rows[0]["theme_type"] == "concept"
    assert rows[0]["membership_source"] == "theme_membership.v1"
    assert rows[0]["pit_membership"] is True
    assert rows[0]["theme_memberships"] == [
        "industry::machinery",
        "concept::low-altitude-economy",
    ]
