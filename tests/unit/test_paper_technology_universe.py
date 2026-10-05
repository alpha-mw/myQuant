"""The universe ranking must reproduce the sealed pool's own ranking."""

from __future__ import annotations

from decimal import Decimal
import glob
import json
from pathlib import Path

import pytest

from quant_investor.intelligence._common import decimal_text
from quant_investor.intelligence.daily import _signal_percentiles

ROOT = Path(__file__).resolve().parents[2]
SESSION = "20260930"


def _signal_values() -> dict:
    for path in reversed(
        sorted(glob.glob(str(ROOT / "results/factors/objects/factor.production_generation/*.json")))
    ):
        payload = json.loads(Path(path).read_text()).get("payload", {})
        if payload.get("as_of") == SESSION and "signal_values" in payload:
            return payload["signal_values"]
    pytest.skip("no factor generation for the session")


def _policy() -> dict:
    return json.loads(
        (ROOT / "results/policies/research/aggressive_tech_manufacturing/v2.json").read_text()
    )["payload"]


def _combined() -> dict[str, Decimal]:
    aliases = {"LOW": "pv_low_dollar_volume_5d", "W80": "pv_blend_volstab19x2_mom90_amihud5_w80"}
    values = _signal_values()
    percentiles = {
        alias: {
            symbol: Decimal(decimal_text(value))
            for symbol, value in _signal_percentiles(
                {
                    symbol: Decimal.from_float(float.fromhex(raw))
                    for symbol, raw in values[factor_id].items()
                }
            ).items()
        }
        for alias, factor_id in aliases.items()
    }
    weights = {row["factor_alias"]: Decimal(row["weight"]) for row in _policy()["factor_rows"]}
    return {
        symbol: sum(
            (percentiles[alias][symbol] * weights[alias] for alias in weights), Decimal("0")
        )
        for symbol in percentiles["LOW"]
    }


def test_ranking_reproduces_the_sealed_pool_top_100() -> None:
    """Same weights and same average-tie percentiles must give the sealed order."""

    pool = json.loads(
        (
            ROOT
            / "results/intelligence/research_pool/aggressive_tech_manufacturing"
            / "2026-09-30/factor_research_rank.json"
        ).read_text()
    )["payload"]["pool_rows"]
    combined = _combined()
    ordered = sorted(combined, key=lambda symbol: (-combined[symbol], symbol.encode("ascii")))[:100]
    assert [(row["symbol"], row["combined_percentile"]) for row in pool] == [
        (symbol, decimal_text(combined[symbol])) for symbol in ordered
    ]


def test_technology_universe_members_carry_a_theme_and_a_score() -> None:
    evidence = json.loads(
        (
            ROOT / "data/private/paper_evidence/20260930/paper-technology-universe.v1.json"
        ).read_text()
    )
    assert evidence["trade_date"] == SESSION
    assert evidence["symbols"]
    assert all(themes for themes in evidence["symbols"].values())
    policy_technology = set(_policy()["technology_theme_ids"])
    for themes in evidence["symbols"].values():
        assert {f"TUSHARE_DC:{code}" for code in themes} <= policy_technology
