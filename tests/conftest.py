"""
Quant-Investor V7.0 测试套件
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "unified"))

# 测试配置
@pytest.fixture(scope="session")
def test_data_path():
    """测试数据路径"""
    return PROJECT_ROOT / "tests" / "data"

@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """临时测试数据库"""
    return tmp_path_factory.mktemp("test_db") / "test_stock.db"

@pytest.fixture
def sample_stock_data():
    """示例股票数据"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    
    data = {
        "symbol": ["000001.SZ"] * 100,
        "date": dates,
        "open": 10 + np.random.randn(100).cumsum() * 0.5,
        "high": 11 + np.random.randn(100).cumsum() * 0.5,
        "low": 9 + np.random.randn(100).cumsum() * 0.5,
        "close": 10 + np.random.randn(100).cumsum() * 0.5,
        "volume": np.random.randint(1000000, 10000000, 100),
        "amount": np.random.randint(10000000, 100000000, 100),
    }
    
    df = pd.DataFrame(data)
    df["high"] = df[["open", "close", "high"]].max(axis=1) + 0.1
    df["low"] = df[["open", "close", "low"]].min(axis=1) - 0.1
    
    return df
