from __future__ import annotations

import importlib
import importlib.util

from quant_investor.monitoring import us_simulated_portfolio_tracker as tracker


def test_us_simulated_tracker_helpers_are_split_and_delegated() -> None:
    spec = importlib.util.find_spec(
        "quant_investor.monitoring.us_simulated_tracker_helpers"
    )
    assert spec is not None
    helpers = importlib.import_module(
        "quant_investor.monitoring.us_simulated_tracker_helpers"
    )

    assert tracker.TradeOrder is helpers.TradeOrder
    assert tracker.DEFAULT_BASE_DIR == helpers.DEFAULT_BASE_DIR
    assert tracker.DEFAULT_NOTES_PATH == helpers.DEFAULT_NOTES_PATH
    assert tracker.THEME_BASKETS is helpers.THEME_BASKETS
    assert tracker._parse_initial_holding is helpers.parse_initial_holding
    assert tracker._parse_cap is helpers.parse_cap
    assert tracker._safe_pct is helpers.safe_pct
    assert tracker._rank_theme_strength is helpers.rank_theme_strength
    assert tracker._theme_for_symbol is helpers.theme_for_symbol
    assert tracker._format_theme_lines is helpers.format_theme_lines

    assert tracker._parse_initial_holding("cvx:4:199.71") == ("CVX", 4, 199.71)
    assert tracker._parse_cap("eog:8") == ("EOG", 8)
    assert tracker._safe_pct(5.0, 0.0) == 0.0
    assert tracker._theme_for_symbol("NVDA") == "ai"
