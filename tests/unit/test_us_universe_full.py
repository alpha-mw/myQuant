from __future__ import annotations

import importlib
import json
from pathlib import Path

from quant_investor.market.config import normalize_universe


def test_us_normalize_universe_uses_full_us():
    assert normalize_universe("US", "full_us") == ["full_us"]
    assert normalize_universe("US", "full_market") == ["full_us"]
    assert normalize_universe("US", "all_us") == ["full_us"]


def test_us_load_universe_canonicalizes_full_us(tmp_path, monkeypatch):
    module = importlib.import_module("quant_investor.market.download_us")
    monkeypatch.setattr(module, "create_tushare_pro", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "TUSHARE_AVAILABLE", False)

    universe_path = tmp_path / "complete_us_universe.json"
    universe_path.write_text(
        json.dumps(
            {
                "large_cap": ["AAPL", "MSFT"],
                "mid_cap": ["IBM"],
                "small_cap": ["F", "GM"],
                "stats": {"total_unique": 5},
            }
        ),
        encoding="utf-8",
    )

    downloader = module.FullMarketDownloader(data_dir=str(tmp_path), years=3)
    universe = downloader.load_universe(universe_file=str(universe_path))

    assert universe["full_us"] == ["AAPL", "MSFT", "IBM", "F", "GM"]
    assert universe["full_market"] == universe["full_us"]
    assert universe["all_us"] == universe["full_us"]
    assert universe["stats"]["full_us"] == 5
