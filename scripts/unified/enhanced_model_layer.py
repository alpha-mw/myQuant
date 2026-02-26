#!/usr/bin/env python3
"""
Enhanced Model Layer - 增强版模型层

支持模型:
- 传统ML: Random Forest, XGBoost, SVM
- 深度学习: LSTM, Transformer
- 强化学习: PPO, DQN (预留接口)

特性:
- 时间序列交叉验证
- 多种损失函数
- 特征重要性分析
- 模型集成
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 可选依赖
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class ModelResult:
    """模型训练结果"""
    model_name: str
    model: Any
    feature_importance: Dict[str, float] = field(default_factory=dict)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    predictions: pd.Series = field(default_factory=pd.Series)


class EnhancedModelLayer:
    """
    增强版模型层
    
    支持多种模型和训练策略
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, ModelResult] = {}
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[EnhancedModelLayer] {msg}")
    
    # ========== 数据准备 ==========
    
    def prepare_data(self, df: pd.DataFrame, 
                     feature_cols: List[str],
                     label_col: str,
                     date_col: str = 'date') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        准备训练数据
        
        Returns:
            X, y, dates
        """
        # 剔除缺失值
        valid_data = df[feature_cols + [label_col, date_col]].dropna()
        
        X = valid_data[feature_cols]
        y = valid_data[label_col]
        dates = valid_data[date_col]
        
        self._log(f"数据准备完成: {len(X)} 样本, {len(feature_cols)} 特征")
        
        return X, y, dates
    
    def time_series_split(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                          n_splits: int = 5) -> List[Tuple]:
        """
        时间序列交叉验证
        
        确保训练集在验证集之前，防止数据泄露
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        splits = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            splits.append((X_train, X_val, y_train, y_val))
        
        self._log(f"时间序列分割完成: {n_splits} 折")
        return splits
    
    # ========== 传统ML模型 ==========
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           task: str = 'regression',
                           **kwargs) -> ModelResult:
        """
        训练随机森林
        
        Args:
            task: 'regression' 或 'classification'
        """
        self._log("训练 Random Forest...")
        
        if task == 'regression':
            model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        # 预测
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # 计算指标
        if task == 'regression':
            train_metrics = {
                'mse': mean_squared_error(y_train, train_pred),
                'mae': mean_absolute_error(y_train, train_pred)
            }
            val_metrics = {
                'mse': mean_squared_error(y_val, val_pred),
                'mae': mean_absolute_error(y_val, val_pred)
            }
        else:
            train_metrics = {
                'accuracy': accuracy_score(y_train, train_pred),
                'f1': f1_score(y_train, train_pred, average='weighted')
            }
            val_metrics = {
                'accuracy': accuracy_score(y_val, val_pred),
                'f1': f1_score(y_val, val_pred, average='weighted')
            }
        
        # 特征重要性
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        result = ModelResult(
            model_name='RandomForest',
            model=model,
            feature_importance=feature_importance,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            predictions=pd.Series(val_pred, index=X_val.index)
        )
        
        self._log(f"Random Forest 训练完成: Val MSE={val_metrics.get('mse', 0):.6f}")
        
        return result
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series,
                     task: str = 'regression',
                     **kwargs) -> ModelResult:
        """训练XGBoost"""
        if not XGBOOST_AVAILABLE:
            self._log("XGBoost未安装，跳过")
            return ModelResult(model_name='XGBoost', model=None)
        
        self._log("训练 XGBoost...")
        
        if task == 'regression':
            model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train, verbose=False)
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        if task == 'regression':
            train_metrics = {'mse': mean_squared_error(y_train, train_pred)}
            val_metrics = {'mse': mean_squared_error(y_val, val_pred)}
        else:
            train_metrics = {'accuracy': accuracy_score(y_train, train_pred)}
            val_metrics = {'accuracy': accuracy_score(y_val, val_pred)}
        
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        result = ModelResult(
            model_name='XGBoost',
            model=model,
            feature_importance=feature_importance,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            predictions=pd.Series(val_pred, index=X_val.index)
        )
        
        self._log(f"XGBoost 训练完成: Val MSE={val_metrics.get('mse', 0):.6f}")
        
        return result
    
    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 task: str = 'regression',
                 **kwargs) -> ModelResult:
        """训练SVM"""
        self._log("训练 SVM...")
        
        if task == 'regression':
            model = SVR(kernel=kwargs.get('kernel', 'rbf'), C=kwargs.get('C', 1.0))
        else:
            model = SVC(kernel=kwargs.get('kernel', 'rbf'), C=kwargs.get('C', 1.0))
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        if task == 'regression':
            train_metrics = {'mse': mean_squared_error(y_train, train_pred)}
            val_metrics = {'mse': mean_squared_error(y_val, val_pred)}
        else:
            train_metrics = {'accuracy': accuracy_score(y_train, train_pred)}
            val_metrics = {'accuracy': accuracy_score(y_val, val_pred)}
        
        result = ModelResult(
            model_name='SVM',
            model=model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            predictions=pd.Series(val_pred, index=X_val.index)
        )
        
        self._log(f"SVM 训练完成: Val MSE={val_metrics.get('mse', 0):.6f}")
        
        return result
    
    # ========== 深度学习模型 ==========
    
    def build_lstm_model(self, input_shape: Tuple[int, int],
                        units: List[int] = [64, 32],
                        dropout: float = 0.2):
        """构建LSTM模型"""
        if not TF_AVAILABLE:
            return None
        
        model = keras.Sequential()
        
        for i, u in enumerate(units):
            if i == 0:
                model.add(layers.LSTM(u, return_sequences=(i < len(units) - 1),
                                     input_shape=input_shape))
            else:
                model.add(layers.LSTM(u, return_sequences=(i < len(units) - 1)))
            model.add(layers.Dropout(dropout))
        
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  epochs: int = 50,
                  batch_size: int = 32) -> ModelResult:
        """训练LSTM"""
        if not TF_AVAILABLE:
            self._log("TensorFlow未安装，跳过LSTM")
            return ModelResult(model_name='LSTM', model=None)
        
        self._log("训练 LSTM...")
        
        # 调整数据形状 (samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        if model is None:
            return ModelResult(model_name='LSTM', model=None)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        train_pred = model.predict(X_train, verbose=0).flatten()
        val_pred = model.predict(X_val, verbose=0).flatten()
        
        train_metrics = {'mse': mean_squared_error(y_train, train_pred)}
        val_metrics = {'mse': mean_squared_error(y_val, val_pred)}
        
        result = ModelResult(
            model_name='LSTM',
            model=model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            predictions=pd.Series(val_pred)
        )
        
        self._log(f"LSTM 训练完成: Val MSE={val_metrics['mse']:.6f}")
        
        return result
    
    # ========== 模型集成 ==========
    
    def ensemble_predict(self, results: List[ModelResult],
                        weights: Optional[List[float]] = None) -> pd.Series:
        """
        模型集成预测
        
        简单平均或加权平均
        """
        valid_results = [r for r in results if r.model is not None]
        
        if not valid_results:
            return pd.Series()
        
        if weights is None:
            weights = [1.0 / len(valid_results)] * len(valid_results)
        
        # 加权平均
        ensemble_pred = None
        for i, result in enumerate(valid_results):
            if ensemble_pred is None:
                ensemble_pred = result.predictions * weights[i]
            else:
                ensemble_pred += result.predictions * weights[i]
        
        return ensemble_pred
    
    # ========== 主训练入口 ==========
    
    def train_all_models(self, df: pd.DataFrame,
                        feature_cols: List[str],
                        label_col: str,
                        task: str = 'regression',
                        use_lstm: bool = False) -> Dict[str, ModelResult]:
        """
        训练所有模型
        
        Args:
            task: 'regression' (回归) 或 'classification' (分类)
            use_lstm: 是否使用LSTM (需要3D数据)
        """
        self._log(f"开始训练所有模型: {len(feature_cols)} 特征, 任务={task}")
        
        # 准备数据
        X, y, dates = self.prepare_data(df, feature_cols, label_col)
        
        # 时间序列分割
        splits = self.time_series_split(X, y, dates, n_splits=5)
        
        # 使用最后一折作为测试
        X_train, X_val, y_train, y_val = splits[-1]
        
        results = {}
        
        # 1. Random Forest
        results['rf'] = self.train_random_forest(
            X_train, y_train, X_val, y_val, task=task
        )
        
        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            results['xgb'] = self.train_xgboost(
                X_train, y_train, X_val, y_val, task=task
            )
        
        # 3. SVM
        results['svm'] = self.train_svm(
            X_train, y_train, X_val, y_val, task=task
        )
        
        # 4. LSTM
        if use_lstm and TF_AVAILABLE:
            results['lstm'] = self.train_lstm(
                X_train.values, y_train.values,
                X_val.values, y_val.values
            )
        
        # 5. 集成预测
        ensemble_pred = self.ensemble_predict(list(results.values()))
        
        if not ensemble_pred.empty:
            ensemble_mse = mean_squared_error(y_val.loc[ensemble_pred.index], ensemble_pred)
            self._log(f"集成模型 Val MSE: {ensemble_mse:.6f}")
        
        self.results = results
        return results
    
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """获取特征重要性排名"""
        importance_dict = {}
        
        for name, result in self.results.items():
            if result.feature_importance:
                for feat, imp in result.feature_importance.items():
                    if feat not in importance_dict:
                        importance_dict[feat] = []
                    importance_dict[feat].append(imp)
        
        # 平均重要性
        avg_importance = {
            feat: np.mean(imps) 
            for feat, imps in importance_dict.items()
        }
        
        ranking = pd.DataFrame([
            {'特征': feat, '平均重要性': imp}
            for feat, imp in sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        ])
        
        return ranking


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Enhanced Model Layer - 测试")
    print("=" * 80)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'date': pd.date_range('2024-01-01', periods=n_samples, freq='B'),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'label': np.random.randn(n_samples) * 0.02 + 0.001
    }
    
    df = pd.DataFrame(data)
    
    print(f"\n测试数据: {len(df)} 样本")
    
    # 训练模型
    model_layer = EnhancedModelLayer(verbose=True)
    
    results = model_layer.train_all_models(
        df,
        feature_cols=['feature1', 'feature2', 'feature3'],
        label_col='label',
        task='regression',
        use_lstm=False
    )
    
    print("\n" + "=" * 80)
    print("模型性能对比")
    print("=" * 80)
    
    for name, result in results.items():
        if result.model is not None:
            print(f"{name:10s}: Train MSE={result.train_metrics.get('mse', 0):.6f}, "
                  f"Val MSE={result.val_metrics.get('mse', 0):.6f}")
    
    # 特征重要性
    print("\n" + "=" * 80)
    print("特征重要性排名")
    print("=" * 80)
    ranking = model_layer.get_feature_importance_ranking()
    print(ranking)
