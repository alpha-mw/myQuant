from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_investor.monitoring import cn_aggressive_portfolio_tracker as tracker


def test_market_metrics_cache_api_is_split_from_tracker() -> None:
    from quant_investor.monitoring import cn_aggressive_market_metrics as metrics

    assert tracker.MarketMetricsBundle is metrics.MarketMetricsBundle
    assert (
        tracker._load_or_compute_market_metrics_bundle
        is metrics.load_or_compute_market_metrics_bundle
    )
    assert tracker.MARKET_METRICS_CATEGORIES == metrics.MARKET_METRICS_CATEGORIES


def _fake_reader() -> SimpleNamespace:
    return SimpleNamespace(
        snapshot=lambda: {
            "snapshot_id": "snap-cache",
            "latest_complete_trade_date": "20260103",
        }
    )


def _components() -> dict[str, list[str]]:
    return {
        "full_a": ["000001.SZ", "000002.SZ"],
        "hs300": ["000001.SZ"],
        "zz500": ["000002.SZ"],
        "zz1000": [],
    }


def _metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "category": "hs300",
                "ret5": 0.03,
                "ret20": 0.12,
                "ret60": 0.20,
                "close_vs_ma20": 0.05,
                "ma20_vs_ma60": 0.03,
                "ma60_vs_ma120": 0.01,
                "dd20": -0.02,
                "latest_close": 12.0,
                "stage_target_price": 14.0,
                "stage_stop_price": 10.0,
                "score_full_market": 0.9,
                "rank_full_market": 1,
            }
        ]
    )


def test_market_metrics_cache_hit_does_not_call_compute(tmp_path: Path) -> None:
    calls = {"count": 0}

    def _compute(**kwargs):
        calls["count"] += 1
        return _metrics_frame(), {"hs300": {"latest_count": 1}}

    first = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )
    assert first.cache_meta["status"] == "blocking_generated"
    assert calls["count"] == 1

    def _must_not_compute(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("cache hit should not recompute full-market metrics")

    second = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_must_not_compute,
    )
    assert second.cache_meta["status"] == "cache_hit"
    assert second.full_metrics["symbol"].tolist() == ["000001.SZ"]
    assert second.breadth["hs300"]["latest_count"] == 1


def test_market_metrics_partial_cache_is_not_a_hit(tmp_path: Path) -> None:
    calls = {"count": 0}

    def _compute(**kwargs):
        calls["count"] += 1
        return _metrics_frame(), {"hs300": {"latest_count": 1}}

    first = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )
    assert first.cache_meta["status"] == "blocking_generated"

    breadth_path = Path(first.cache_meta["cache_dir"]) / "breadth.json"
    breadth_path.unlink()

    second = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )
    assert second.cache_meta["status"] == "blocking_generated"
    assert calls["count"] == 2


def test_market_metrics_missing_parquet_is_not_a_hit(tmp_path: Path) -> None:
    calls = {"count": 0}

    def _compute(**kwargs):
        calls["count"] += 1
        return _metrics_frame(), {"hs300": {"latest_count": 1}}

    first = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )
    assert first.cache_meta["status"] == "blocking_generated"

    metrics_path = Path(first.cache_meta["cache_dir"]) / "full_metrics.parquet"
    metrics_path.unlink()

    second = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )
    assert second.cache_meta["status"] == "blocking_generated"
    assert calls["count"] == 2


@pytest.mark.parametrize(
    ("metadata_key", "bad_value"),
    [
        ("schema_version", "bad-schema"),
        ("snapshot_id", "snap-other"),
        ("analysis_trade_date", "20260102"),
        ("components_fingerprint", "bad-fingerprint"),
    ],
)
def test_market_metrics_metadata_mismatch_is_not_a_hit(
    tmp_path: Path,
    metadata_key: str,
    bad_value: str,
) -> None:
    calls = {"count": 0}

    def _compute(**kwargs):
        calls["count"] += 1
        return _metrics_frame(), {"hs300": {"latest_count": 1}}

    first = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )
    breadth_path = Path(first.cache_meta["cache_dir"]) / "breadth.json"
    payload = json.loads(breadth_path.read_text(encoding="utf-8"))
    payload[metadata_key] = bad_value
    breadth_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    second = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )

    assert second.cache_meta["status"] == "blocking_generated"
    assert calls["count"] == 2


def test_market_metrics_required_column_mismatch_is_not_a_hit(tmp_path: Path) -> None:
    calls = {"count": 0}

    def _compute(**kwargs):
        calls["count"] += 1
        return _metrics_frame(), {"hs300": {"latest_count": 1}}

    first = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )
    metrics_path = Path(first.cache_meta["cache_dir"]) / "full_metrics.parquet"
    stale_frame = pd.read_parquet(metrics_path).drop(columns=["score_full_market"])
    stale_frame.to_parquet(metrics_path, index=False)

    second = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )

    assert second.cache_meta["status"] == "blocking_generated"
    assert calls["count"] == 2


def test_market_metrics_skip_prewarm_computes_without_cache_hit(tmp_path: Path) -> None:
    calls = {"count": 0}

    def _compute(**kwargs):
        calls["count"] += 1
        return _metrics_frame(), {"hs300": {"latest_count": 1}}

    first = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        compute_fn=_compute,
    )
    assert first.cache_meta["status"] == "blocking_generated"

    second = tracker._load_or_compute_market_metrics_bundle(
        base_dir=tmp_path,
        components=_components(),
        reader=_fake_reader(),
        latest_trade_date="20260103",
        completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
        skip_prewarm=True,
        compute_fn=_compute,
    )

    assert second.cache_meta["status"] == "skipped"
    assert calls["count"] == 2


def test_market_metrics_skip_prewarm_keeps_strict_snapshot_failure(tmp_path: Path) -> None:
    reader = SimpleNamespace(snapshot=lambda: {"healthy": False, "blockers": ["snapshot blocked"]})

    def _must_not_compute(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("strict snapshot failure must fail before compute")

    with pytest.raises(RuntimeError, match="snapshot blocked"):
        tracker._load_or_compute_market_metrics_bundle(
            base_dir=tmp_path,
            components=_components(),
            reader=reader,
            latest_trade_date="20260103",
            completeness_report={"categories": {"hs300": {}, "zz500": {}, "zz1000": {}}},
            skip_prewarm=True,
            compute_fn=_must_not_compute,
        )
