"""
数据层单元测试
测试数据获取、清洗、特征工程
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quant_investor.enhanced_data_layer import (
    EnhancedDataLayer,
    USCompositeDataSource,
    USLocalCSVDataSource,
    _normalize_ohlcv_frame,
    normalize_kline_frame_for_model,
)
from quant_investor.branch_contracts import (
    CorporateDocumentSnapshot,
    ForecastSnapshot,
    FundamentalSnapshot,
    ManagementSnapshot,
    OwnershipSnapshot,
    UnifiedDataBundle,
)
from quant_investor.fundamental_branch import FundamentalBranch


class TestDataLayer:
    """数据层测试"""
    
    def test_data_loading(self, sample_stock_data):
        """测试数据加载"""
        assert len(sample_stock_data) == 100
        assert "symbol" in sample_stock_data.columns
        assert "close" in sample_stock_data.columns
        assert sample_stock_data["close"].notna().all()
    
    def test_ohlcv_integrity(self, sample_stock_data):
        """测试OHLCV数据完整性"""
        df = sample_stock_data
        
        # 检查价格逻辑
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
    
    def test_momentum_calculation(self, sample_stock_data):
        """测试动量因子计算"""
        df = sample_stock_data.copy()
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        
        # 检查计算结果
        assert "return_5d" in df.columns
        assert "return_20d" in df.columns
        
        # 前5天应该是NaN
        assert df["return_5d"].iloc[:5].isna().all()
        # 第6天开始有值
        assert df["return_5d"].iloc[5:].notna().all()
    
    def test_volatility_calculation(self, sample_stock_data):
        """测试波动率计算"""
        df = sample_stock_data.copy()
        df["volatility"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
        
        assert "volatility" in df.columns
        # 前20天是NaN
        assert df["volatility"].iloc[:20].isna().all()
        # 波动率应为正数
        valid_vol = df["volatility"].dropna()
        assert (valid_vol >= 0).all()
    
    def test_rsi_calculation(self, sample_stock_data):
        """测试RSI计算"""
        df = sample_stock_data.copy()
        
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        assert "rsi" in df.columns
        # RSI应在0-100之间
        valid_rsi = df["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()


class TestDataCleaning:
    """数据清洗测试"""
    
    def test_missing_value_handling(self):
        """测试缺失值处理"""
        df = pd.DataFrame({
            "close": [1.0, 2.0, np.nan, 4.0, 5.0],
            "volume": [100, 200, 300, np.nan, 500],
        })
        
        # 前向填充（使用ffill()方法）
        df_filled = df.ffill()
        assert df_filled["close"].isna().sum() == 0
        
    def test_outlier_detection(self, sample_stock_data):
        """测试异常值检测"""
        df = sample_stock_data.copy()
        
        # 使用3-sigma法则检测异常值
        mean = df["close"].mean()
        std = df["close"].std()
        outliers = df[(df["close"] < mean - 3*std) | (df["close"] > mean + 3*std)]
        
        # 正常情况下异常值应该很少
        assert len(outliers) < len(df) * 0.05


class TestUSDataSources:
    """美股数据源测试"""

    def test_local_us_csv_source_loads_and_normalizes(self, tmp_path):
        """本地美股 CSV 应可被标准化读取。"""
        data_dir = tmp_path / "data" / "us_market_full" / "large_cap"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "AAPL.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "Date,Open,High,Low,Close,Volume",
                    "2026-03-18,252,255,249,250,1000",
                    "2026-03-19,249,252,247,249,1100",
                    "2026-03-20,248,249,246,248,1200",
                ]
            ),
            encoding="utf-8",
        )

        source = USLocalCSVDataSource(str(tmp_path / "data" / "us_market_full"))
        df = source.get_ohlcv("AAPL", "20260318", "20260320")

        assert not df.empty
        assert list(df.columns[:6]) == ["date", "open", "high", "low", "close", "volume"]
        assert "amount" in df.columns
        assert len(df) == 3
        assert df["date"].max().strftime("%Y-%m-%d") == "2026-03-20"

    def test_us_composite_source_prefers_local_cache(self, tmp_path):
        """美股复合数据源应优先使用本地缓存。"""
        data_dir = tmp_path / "data" / "us_market_full" / "large_cap"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "MSFT.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "Date,Open,High,Low,Close,Volume",
                    "2026-03-18,401,405,398,404,2000",
                    "2026-03-19,404,407,400,402,2100",
                    "2026-03-20,402,406,399,405,2200",
                ]
            ),
            encoding="utf-8",
        )

        source = USCompositeDataSource(local_data_dir=str(tmp_path / "data" / "us_market_full"))
        df = source.get_ohlcv("MSFT", "20260318", "20260321")

        assert not df.empty
        assert source.last_ohlcv_source == "local_csv"
        assert df["close"].iloc[-1] == pytest.approx(405.0)

    def test_normalize_ohlcv_frame_strips_timezone(self):
        """带时区时间列应被标准化为无时区，便于跨数据源对齐。"""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2026-03-19 00:00:00+00:00", "2026-03-20 00:00:00+00:00"],
                    utc=True,
                ),
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            }
        )

        normalized = _normalize_ohlcv_frame(df)

        assert not normalized.empty
        assert normalized["date"].dt.tz is None


class TestSnapshotFallbacks:
    """V9 snapshot 接口降级测试"""

    def test_forecast_snapshot_returns_unavailable_when_tushare_offline(self):
        data_layer = EnhancedDataLayer(market="CN", verbose=False)
        snapshot = data_layer.get_earnings_forecast_snapshot("000001.SZ", "2026-03-20")

        assert snapshot.available is False
        # Provider now exists (tushare) but data is unavailable when client offline
        assert snapshot.source in {"neutral", "tushare"}
        has_missing_note = any(
            "missing" in n for n in snapshot.notes
        )
        assert has_missing_note or snapshot.data_quality.get("provider_missing") is True

    def test_document_semantic_snapshot_returns_neutral_when_missing(self):
        data_layer = EnhancedDataLayer(market="CN", verbose=False)
        snapshot = data_layer.get_document_semantic_snapshot("000001.SZ", "2026-03-20")

        assert snapshot.available is False
        assert snapshot.source in {"offline_snapshot", "neutral"}
        assert "as_of" not in snapshot.data_quality
        assert "provider_missing" in snapshot.data_quality

    def test_cn_daily_kline_normalization_produces_stable_freq(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-16", "2026-03-17", "2026-03-19"]),
                "open": [10.0, 10.2, 10.3],
                "high": [10.3, 10.4, 10.5],
                "low": [9.9, 10.0, 10.1],
                "close": [10.1, 10.3, 10.4],
                "volume": [1000, 1100, 1200],
            }
        )

        normalized = normalize_kline_frame_for_model(df)

        assert isinstance(normalized.index, pd.DatetimeIndex)
        assert normalized.index.freqstr == "B"
        assert normalized.index.tz is None
        assert normalized.loc[pd.Timestamp("2026-03-18"), "close"] == pytest.approx(10.3)

    def test_provider_global_missing_and_symbol_missing_are_distinct(self):
        data_layer = EnhancedDataLayer(market="CN", verbose=False)

        class _NoForecastProvider:
            pass

        data_layer._source = _NoForecastProvider()
        global_missing = data_layer.get_earnings_forecast_snapshot("000001.SZ", "2026-03-20")

        class _EmptyForecastProvider:
            def get_earnings_forecast_snapshot(self, symbol, as_of):
                return {}

        data_layer._source = _EmptyForecastProvider()
        symbol_missing = data_layer.get_earnings_forecast_snapshot("000001.SZ", "2026-03-20")

        assert global_missing.data_quality["missing_scope"] == "global"
        assert global_missing.data_quality["provider_missing"] is True
        assert symbol_missing.data_quality["missing_scope"] == "symbol"
        assert symbol_missing.data_quality["provider_missing"] is False
        assert symbol_missing.data_quality["snapshot_missing"] is True

    def test_document_semantics_missing_only_enters_coverage(self):
        class _StubDataLayer:
            def get_point_in_time_fundamental_snapshot(self, symbol, as_of):
                return FundamentalSnapshot(
                    symbol=symbol,
                    available=True,
                    roe=0.18,
                    gross_margin=0.32,
                    profit_growth=0.15,
                    revenue_growth=0.12,
                    debt_ratio=0.28,
                    current_ratio=1.4,
                    pe=12.0,
                    pb=1.8,
                    ps=2.0,
                )

            def get_earnings_forecast_snapshot(self, symbol, as_of):
                return ForecastSnapshot(
                    symbol=symbol,
                    available=True,
                    forecast_revision=0.06,
                    eps_growth=0.14,
                    coverage_count=8,
                    provider="forecast_provider",
                )

            def get_management_snapshot(self, symbol, as_of):
                return ManagementSnapshot(
                    symbol=symbol,
                    available=True,
                    management_stability=0.6,
                    governance_score=0.5,
                    management_alignment=0.4,
                )

            def get_ownership_snapshot(self, symbol, as_of):
                return OwnershipSnapshot(
                    symbol=symbol,
                    available=True,
                    concentration_score=0.2,
                    institutional_holding_pct=0.35,
                    ownership_change_signal=0.1,
                )

            def get_document_semantic_snapshot(self, symbol, as_of):
                return CorporateDocumentSnapshot(
                    symbol=symbol,
                    available=False,
                    source="offline_snapshot",
                    data_quality={
                        "reason": "snapshot_missing",
                        "missing_scope": "symbol",
                        "provider_missing": False,
                        "snapshot_missing": True,
                    },
                )

        dates = pd.bdate_range("2026-03-01", periods=40)
        data_bundle = UnifiedDataBundle(
            market="CN",
            symbols=["000001.SZ"],
            symbol_data={
                "000001.SZ": pd.DataFrame(
                    {
                        "date": dates,
                        "close": np.linspace(10, 12, len(dates)),
                        "open": np.linspace(10, 12, len(dates)),
                        "high": np.linspace(10.2, 12.2, len(dates)),
                        "low": np.linspace(9.8, 11.8, len(dates)),
                        "volume": np.linspace(1_000_000, 1_200_000, len(dates)),
                    }
                )
            },
            metadata={"end_date": "20260320"},
        )

        branch = FundamentalBranch(
            data_layer=_StubDataLayer(),
            stock_pool=["000001.SZ"],
            enable_document_semantics=True,
        )
        result = branch.run(data_bundle)

        assert any("文档语义" in note for note in result.coverage_notes)
        assert all("provider_missing" not in risk for risk in result.investment_risks)
        assert all("snapshot_missing" not in risk for risk in result.investment_risks)
        assert all("文档语义" not in risk for risk in result.investment_risks)
