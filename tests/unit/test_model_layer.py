"""
模型层单元测试
测试机器学习模型训练、预测、评估
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class TestModelTraining:
    """模型训练测试"""
    
    def test_random_forest_training(self):
        """测试随机森林训练"""
        np.random.seed(42)
        n_samples = 1000
        
        # 生成模拟数据
        X = np.random.randn(n_samples, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # 验证模型能拟合
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        assert r2 > 0.5  # R2应该大于0.5
        assert len(model.feature_importances_) == 5
    
    def test_model_prediction_shape(self):
        """测试模型预测输出形状"""
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        X_test = np.random.randn(20, 3)
        y_pred = model.predict(X_test)
        
        assert y_pred.shape == (20,)
    
    def test_feature_importance(self):
        """测试特征重要性"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        # 只有第一个特征与目标相关
        y = X[:, 0] * 2 + np.random.randn(100) * 0.1
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importances = model.feature_importances_
        # 第一个特征应该最重要
        assert importances[0] > importances[1]
        assert importances[0] > importances[2]


class TestModelEvaluation:
    """模型评估测试"""
    
    def test_r2_score_calculation(self):
        """测试R2分数计算"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.0])
        
        r2 = r2_score(y_true, y_pred)
        
        assert 0 < r2 <= 1  # R2应该在0-1之间
    
    def test_mse_calculation(self):
        """测试MSE计算"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.5, 2.5, 3.5])
        
        mse = mean_squared_error(y_true, y_pred)
        expected_mse = 0.25  # (0.5^2 + 0.5^2 + 0.5^2) / 3
        
        assert abs(mse - expected_mse) < 0.001
    
    def test_perfect_prediction(self):
        """测试完美预测"""
        y = np.array([1, 2, 3, 4, 5])
        
        r2 = r2_score(y, y)
        mse = mean_squared_error(y, y)
        
        assert r2 == 1.0
        assert mse == 0.0


class TestModelEnsemble:
    """模型集成测试"""
    
    def test_ensemble_prediction(self):
        """测试集成预测"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + np.random.randn(100) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练两个模型
        model1 = RandomForestRegressor(n_estimators=5, random_state=42)
        model2 = RandomForestRegressor(n_estimators=5, random_state=43)
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        # 集成预测（简单平均）
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        ensemble_pred = (pred1 + pred2) / 2
        
        # 集成预测应该合理
        assert len(ensemble_pred) == len(y_test)
        assert not np.any(np.isnan(ensemble_pred))
