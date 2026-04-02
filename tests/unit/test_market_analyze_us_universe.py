from __future__ import annotations

import json

import pandas as pd

import quant_investor.market.analyze as market_analyze


def test_get_all_local_symbols_uses_curated_us_universe(monkeypatch, tmp_path):
    universe_file = tmp_path / "complete_us_universe.json"
    universe_file.write_text(
        json.dumps(
            {
                "large_cap": ["AAPL", "BRK.B"],
                "mid_cap": [],
                "small_cap": [],
            }
        ),
        encoding="utf-8",
    )

    market_root = tmp_path / "us_market_full"
    large_dir = market_root / "large_cap"
    large_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Date": "2026-03-27", "Close": 100.0}]).to_csv(large_dir / "AAPL.csv", index=False)
    pd.DataFrame([{"Date": "2026-03-27", "Close": 500.0}]).to_csv(large_dir / "BRK-B.csv", index=False)
    pd.DataFrame([{"Date": "2026-03-27", "Close": 10.0}]).to_csv(large_dir / "EXTRA.csv", index=False)

    monkeypatch.setattr(market_analyze, "US_UNIVERSE_FILE", universe_file)

    symbols = market_analyze.get_all_local_symbols(
        "large_cap",
        market="US",
        data_dir=str(market_root),
    )

    assert symbols == ["AAPL", "BRK-B"]
