from __future__ import annotations

import pandas as pd
import pytest

from quant_investor.market import macro_mart
from quant_investor.macro.contracts import canonical_hash
from quant_investor.macro.registry import NATIONAL_DOMAIN_WEIGHTS


TARGET = "20240510"


def _market_frame() -> pd.DataFrame:
    dates = pd.bdate_range(end="2024-05-10", periods=100)
    rows: list[dict[str, object]] = []
    for symbol_index in range(100):
        symbol = f"{symbol_index:06d}.SZ"
        daily_return = 0.001 if symbol_index % 2 == 0 else -0.001
        for date_index, trade_date in enumerate(dates):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "close": 100.0 * (1.0 + daily_return) ** date_index,
                }
            )
    return pd.DataFrame(rows)


def _provider_bundle() -> dict[str, object]:
    return {
        "fetched_at": "2024-05-10T08:30:00+00:00",
        "source": macro_mart.SOURCE_TUSHARE,
        "source_priority": macro_mart.SOURCE_TUSHARE,
        "selected_inputs": {"cn_m": {"values": {"m2_yoy": 9.0}}},
    }


def _macro_snapshot() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "macro-snapshot.v2",
        "market": "CN",
        "as_of": "2024-05-10",
        "readiness_status": "pass",
        "national_states": {
            domain: 0.0 for domain in NATIONAL_DOMAIN_WEIGHTS
        },
        "coverage": {"national": 1.0},
    }
    payload["snapshot_hash"] = canonical_hash(payload)
    return payload


def _observation_generation() -> dict[str, object]:
    return {
        "generation_id": "macro-observations-test",
        "pointer_sha256": "1" * 64,
        "parquet_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "content_set_hash": "4" * 64,
        "row_count": 39,
    }


def test_stale_terminal_symbol_does_not_change_formula_cross_section() -> None:
    active_market = _market_frame()
    stale_dates = pd.bdate_range(end="2024-05-09", periods=99)
    stale = pd.DataFrame(
        {
            "ts_code": "999999.SZ",
            "trade_date": stale_dates,
            "close": [100.0 * 1.03**index for index in range(len(stale_dates))],
        }
    )

    baseline, baseline_universe, _ = macro_mart._derive_v15_macro_frame(
        active_market,
        trade_date=TARGET,
        provider_bundle=_provider_bundle(),
        macro_snapshot=_macro_snapshot(),
        observation_generation=_observation_generation(),
    )
    with_stale, stale_universe, _ = macro_mart._derive_v15_macro_frame(
        pd.concat([active_market, stale], ignore_index=True),
        trade_date=TARGET,
        provider_bundle=_provider_bundle(),
        macro_snapshot=_macro_snapshot(),
        observation_generation=_observation_generation(),
    )

    assert with_stale.iloc[0]["macro_score"] == pytest.approx(
        baseline.iloc[0]["macro_score"]
    )
    assert with_stale.iloc[0]["liquidity_score"] == pytest.approx(
        baseline.iloc[0]["liquidity_score"]
    )
    assert stale_universe["selection_rule"] == (
        macro_mart.MARKET_FORMULA_SELECTION_RULE
    )
    assert stale_universe["target_trade_date"] == TARGET
    assert stale_universe["input_symbol_count"] == 101
    assert stale_universe["target_terminal_symbol_count"] == 100
    assert stale_universe["stale_symbol_count"] == 1
    assert stale_universe["scored_symbol_count"] == 100
    assert stale_universe["target_terminal_symbol_set_sha256"] == (
        baseline_universe["target_terminal_symbol_set_sha256"]
    )
    assert stale_universe["scored_symbol_set_sha256"] == (
        baseline_universe["scored_symbol_set_sha256"]
    )
    assert macro_mart._validate_market_formula_universe(
        stale_universe,
        trade_date=TARGET,
        require_control_binding=True,
    ) == stale_universe

    tampered = dict(stale_universe)
    tampered["stale_symbol_count"] = 2
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="formula_universe_count_mismatch",
    ):
        macro_mart._validate_market_formula_universe(
            tampered,
            trade_date=TARGET,
            require_control_binding=True,
        )
