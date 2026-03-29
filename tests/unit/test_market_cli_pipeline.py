from __future__ import annotations

from typing import Any

import quant_investor.cli.main as cli_main
import quant_investor.market.run_pipeline as market_pipeline


def test_market_analyze_cli_passes_agent_layer_args(monkeypatch):
    captured: dict[str, Any] = {}

    def _run_market_analysis(**kwargs):
        captured.update(kwargs)
        return {"results": {}, "reports": {}}

    monkeypatch.setattr(cli_main, "run_market_analysis", _run_market_analysis)

    cli_main.main(
        [
            "market",
            "analyze",
            "--market",
            "CN",
            "--mode",
            "sample",
            "--category",
            "hs300",
            "--no-agent-layer",
            "--agent-model",
            "deepseek-chat",
            "--master-model",
            "gpt-5.4-mini",
            "--agent-timeout",
            "30",
            "--master-timeout",
            "60",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["mode"] == "sample"
    assert captured["categories"] == ["hs300"]
    assert captured["enable_agent_layer"] is False
    assert captured["agent_model"] == "deepseek-chat"
    assert captured["master_model"] == "gpt-5.4-mini"
    assert captured["agent_timeout"] == 30.0
    assert captured["master_timeout"] == 60.0


def test_market_run_cli_dispatches_to_unified_pipeline(monkeypatch):
    captured: dict[str, Any] = {}

    def _run_market_pipeline(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "reports": {}, "download": {}, "timing": {}}

    monkeypatch.setattr(cli_main, "run_market_pipeline", _run_market_pipeline)

    cli_main.main(
        [
            "market",
            "run",
            "--market",
            "CN",
            "--category",
            "hs300",
            "--mode",
            "sample",
            "--skip-download",
            "--agent-model",
            "deepseek-chat",
            "--master-model",
            "gpt-5.4-mini",
            "--agent-timeout",
            "25",
            "--master-timeout",
            "55",
            "--years",
            "5",
            "--workers",
            "6",
            "--max-download-rounds",
            "3",
        ]
    )

    assert captured["market"] == "CN"
    assert captured["categories"] == ["hs300"]
    assert captured["mode"] == "sample"
    assert captured["skip_download"] is True
    assert captured["force_download"] is False
    assert captured["enable_agent_layer"] is True
    assert captured["agent_model"] == "deepseek-chat"
    assert captured["master_model"] == "gpt-5.4-mini"
    assert captured["agent_timeout"] == 25.0
    assert captured["master_timeout"] == 55.0
    assert captured["years"] == 5
    assert captured["workers"] == 6
    assert captured["max_download_rounds"] == 3


def test_unified_pipeline_skips_download_when_data_is_current(monkeypatch):
    captured_analysis: dict[str, Any] = {}

    class FakeDownloader:
        def __init__(self):
            self.download_calls: list[dict[str, Any]] = []
            self.forecast_refresh_calls: list[dict[str, Any]] = []

        def load_components(self):
            return {"hs300": ["000001.SZ"], "stats": {"total_unique": 1}}

        def build_completeness_report(self, *, components=None, categories=None):
            return {
                "latest_trade_date": "20260326",
                "complete": True,
                "blocking_incomplete_count": 0,
                "categories": {
                    "hs300": {
                        "expected": 1,
                        "latest_trade_date": "20260326",
                        "date_counts": {"20260326": 1},
                        "blocking_incomplete_count": 0,
                    }
                },
            }

        def build_forecast_snapshot_report(self, symbols, as_of=None):
            return {
                "requested_as_of": "2026-03-26",
                "expected": len(symbols),
                "fresh_count": len(symbols),
                "missing_symbols": [],
                "stale_symbols": [],
                "refresh_needed": False,
            }

        def refresh_forecast_snapshots(self, symbols, force_refresh=False, as_of=None):
            self.forecast_refresh_calls.append(
                {"symbols": list(symbols), "force_refresh": force_refresh, "as_of": as_of}
            )
            return {
                "before": self.build_forecast_snapshot_report(symbols),
                "after": self.build_forecast_snapshot_report(symbols),
            }

        def download_all(self, **kwargs):
            self.download_calls.append(kwargs)
            return {"timestamp": "20260326_120000"}

    downloader = FakeDownloader()

    def _run_market_analysis(**kwargs):
        captured_analysis.update(kwargs)
        return {
            "results": {"hs300": [{"batch_id": 1}]},
            "reports": {
                "summary_report": "summary.md",
                "trade_report": "trade.md",
                "trade_data": "trade.json",
                "candidate_index": "candidates.json",
            },
        }

    monkeypatch.setattr(market_pipeline, "create_downloader", lambda market, **kwargs: downloader)
    monkeypatch.setattr(market_pipeline, "get_all_local_symbols", lambda category, market=None: [])
    monkeypatch.setattr(market_pipeline, "run_market_analysis", _run_market_analysis)

    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        enable_agent_layer=True,
        agent_model="deepseek-chat",
        master_model="gpt-5.4-mini",
        agent_timeout=20.0,
        master_timeout=40.0,
        verbose=False,
    )

    assert downloader.download_calls == []
    assert downloader.forecast_refresh_calls == []
    assert output["download"]["status"] == "up_to_date"
    assert output["analysis"] == {"hs300": [{"batch_id": 1}]}
    assert captured_analysis["enable_agent_layer"] is True
    assert captured_analysis["agent_model"] == "deepseek-chat"
    assert captured_analysis["master_model"] == "gpt-5.4-mini"
    assert captured_analysis["agent_timeout"] == 20.0
    assert captured_analysis["master_timeout"] == 40.0


def test_unified_pipeline_downloads_before_analysis_when_data_is_stale(monkeypatch):
    captured_analysis: dict[str, Any] = {}

    class FakeDownloader:
        def __init__(self):
            self.download_calls: list[dict[str, Any]] = []
            self.report_calls = 0
            self.forecast_refresh_calls: list[dict[str, Any]] = []

        def load_components(self):
            return {"hs300": ["000001.SZ"], "stats": {"total_unique": 1}}

        def build_completeness_report(self, *, components=None, categories=None):
            self.report_calls += 1
            if self.report_calls == 1:
                return {
                    "latest_trade_date": "20260326",
                    "complete": False,
                    "blocking_incomplete_count": 1,
                    "categories": {
                        "hs300": {
                            "expected": 1,
                            "latest_trade_date": "20260326",
                            "date_counts": {"20260325": 1},
                            "blocking_incomplete_count": 1,
                        }
                    },
                }
            return {
                "latest_trade_date": "20260326",
                "complete": True,
                "blocking_incomplete_count": 0,
                "categories": {
                    "hs300": {
                        "expected": 1,
                        "latest_trade_date": "20260326",
                        "date_counts": {"20260326": 1},
                        "blocking_incomplete_count": 0,
                    }
                },
            }

        def build_forecast_snapshot_report(self, symbols, as_of=None):
            return {
                "requested_as_of": "2026-03-26",
                "expected": len(symbols),
                "fresh_count": len(symbols),
                "missing_symbols": [],
                "stale_symbols": [],
                "refresh_needed": False,
            }

        def refresh_forecast_snapshots(self, symbols, force_refresh=False, as_of=None):
            self.forecast_refresh_calls.append(
                {"symbols": list(symbols), "force_refresh": force_refresh, "as_of": as_of}
            )
            return {
                "before": self.build_forecast_snapshot_report(symbols),
                "after": self.build_forecast_snapshot_report(symbols),
            }

        def download_all(self, **kwargs):
            self.download_calls.append(kwargs)
            return {"timestamp": "20260326_121500"}

    downloader = FakeDownloader()

    def _run_market_analysis(**kwargs):
        captured_analysis.update(kwargs)
        return {
            "results": {"hs300": [{"batch_id": 1}]},
            "reports": {
                "summary_report": "summary.md",
                "trade_report": "trade.md",
                "trade_data": "trade.json",
                "candidate_index": "candidates.json",
            },
        }

    monkeypatch.setattr(market_pipeline, "create_downloader", lambda market, **kwargs: downloader)
    monkeypatch.setattr(market_pipeline, "get_all_local_symbols", lambda category, market=None: [])
    monkeypatch.setattr(market_pipeline, "run_market_analysis", _run_market_analysis)

    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        max_download_rounds=2,
        verbose=False,
    )

    assert len(downloader.download_calls) == 1
    assert downloader.forecast_refresh_calls == []
    assert downloader.download_calls[0]["categories"] == ["hs300"]
    assert downloader.download_calls[0]["max_rounds"] == 2
    assert output["download"]["status"] == "downloaded"
    assert output["download"]["completeness_after"]["complete"] is True
    assert captured_analysis["categories"] == ["hs300"]
    assert captured_analysis["mode"] == "sample"


def test_unified_pipeline_refreshes_forecast_snapshots_when_ohlcv_is_current(monkeypatch):
    captured_analysis: dict[str, Any] = {}

    class FakeDownloader:
        def __init__(self):
            self.download_calls: list[dict[str, Any]] = []
            self.forecast_refresh_calls: list[dict[str, Any]] = []

        def load_components(self):
            return {"hs300": ["000001.SZ", "000002.SZ"], "stats": {"total_unique": 2}}

        def build_completeness_report(self, *, components=None, categories=None):
            return {
                "latest_trade_date": "20260326",
                "complete": True,
                "blocking_incomplete_count": 0,
                "categories": {
                    "hs300": {
                        "expected": 2,
                        "latest_trade_date": "20260326",
                        "date_counts": {"20260326": 2},
                        "blocking_incomplete_count": 0,
                    }
                },
            }

        def build_forecast_snapshot_report(self, symbols, as_of=None):
            refresh_needed = not self.forecast_refresh_calls
            return {
                "requested_as_of": "2026-03-26",
                "expected": len(symbols),
                "fresh_count": 0 if refresh_needed else len(symbols),
                "missing_symbols": list(symbols) if refresh_needed else [],
                "stale_symbols": [],
                "refresh_needed": refresh_needed,
            }

        def refresh_forecast_snapshots(self, symbols, force_refresh=False, as_of=None):
            self.forecast_refresh_calls.append(
                {"symbols": list(symbols), "force_refresh": force_refresh, "as_of": as_of}
            )
            return {
                "before": {
                    "requested_as_of": "2026-03-26",
                    "expected": len(symbols),
                    "fresh_count": 0,
                    "missing_symbols": list(symbols),
                    "stale_symbols": [],
                    "refresh_needed": True,
                },
                "after": {
                    "requested_as_of": "2026-03-26",
                    "expected": len(symbols),
                    "fresh_count": len(symbols),
                    "missing_symbols": [],
                    "stale_symbols": [],
                    "refresh_needed": False,
                },
                "refreshed_symbols": list(symbols),
                "failed_symbols": [],
            }

        def download_all(self, **kwargs):
            self.download_calls.append(kwargs)
            return {"timestamp": "20260326_122000"}

    downloader = FakeDownloader()

    def _run_market_analysis(**kwargs):
        captured_analysis.update(kwargs)
        return {
            "results": {"hs300": [{"batch_id": 1}]},
            "reports": {
                "summary_report": "summary.md",
                "trade_report": "trade.md",
                "trade_data": "trade.json",
                "candidate_index": "candidates.json",
            },
        }

    monkeypatch.setattr(market_pipeline, "create_downloader", lambda market, **kwargs: downloader)
    monkeypatch.setattr(market_pipeline, "get_all_local_symbols", lambda category, market=None: [])
    monkeypatch.setattr(market_pipeline, "run_market_analysis", _run_market_analysis)

    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        verbose=False,
    )

    assert downloader.download_calls == []
    assert len(downloader.forecast_refresh_calls) == 1
    assert downloader.forecast_refresh_calls[0]["symbols"] == ["000001.SZ", "000002.SZ"]
    assert output["download"]["status"] == "refreshed_snapshots"
    assert captured_analysis["market"] == "CN"


def test_unified_pipeline_skip_download_tolerates_missing_tushare(monkeypatch):
    captured_analysis: dict[str, Any] = {}

    def _run_market_analysis(**kwargs):
        captured_analysis.update(kwargs)
        return {
            "results": {"hs300": [{"batch_id": 1}]},
            "reports": {
                "summary_report": "summary.md",
                "trade_report": "trade.md",
                "trade_data": "trade.json",
                "candidate_index": "candidates.json",
            },
        }

    def _raise_missing_tushare(market, **kwargs):
        raise RuntimeError("tushare 未安装，无法下载 A 股全市场数据")

    monkeypatch.setattr(market_pipeline, "create_downloader", _raise_missing_tushare)
    monkeypatch.setattr(market_pipeline, "run_market_analysis", _run_market_analysis)

    output = market_pipeline.run_unified_pipeline(
        market="CN",
        categories=["hs300"],
        mode="sample",
        skip_download=True,
        verbose=False,
    )

    assert output["download"]["status"] == "skipped"
    assert output["download"]["reason"] == "skip_download_dependency_unavailable"
    assert output["download"]["warning"] == "tushare 未安装，无法下载 A 股全市场数据"
    assert output["download"]["completeness_before"] is None
    assert output["analysis"] == {"hs300": [{"batch_id": 1}]}
    assert captured_analysis["market"] == "CN"
    assert captured_analysis["categories"] == ["hs300"]
