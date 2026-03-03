"""
数据层单元测试
测试数据获取、清洗、特征工程
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


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
        
        # 前向填充
        df_filled = df.fillna(method="ffill")
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
