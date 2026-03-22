"""
数据层单元测试
测试数据获取、清洗、特征工程
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quant_investor.enhanced_data_layer import (
    USCompositeDataSource,
    USLocalCSVDataSource,
    _normalize_ohlcv_frame,
)


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
