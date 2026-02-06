#!/usr/bin/env python3
"""
Quant-Investor V6.0 - 统一模型层 (Unified Model Layer)

整合V5.0的机器学习模型能力：
- XGBoost / LightGBM / Random Forest
- 时间序列交叉验证 (避免数据泄露)
- 模型评估与特征重要性分析
- 多模型集成与信号生成
- 候选股票排序与筛选

设计原则：
1. 模型训练严格遵循时间序列规则，防止前视偏差
2. 多模型集成，提高信号稳定性
3. 输出标准化的预测信号和候选股票排名
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据结构 ====================

@dataclass
class ModelResult:
    """单个模型的训练结果"""
    model_name: str
    train_score: float = 0.0
    test_score: float = 0.0
    mse: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    predictions: pd.Series = None


@dataclass
class ModelLayerOutput:
    """模型层的完整输出"""
    # 各模型结果
    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    
    # 集成预测信号
    ensemble_signal: pd.Series = None
    
    # 排序后的候选股票
    ranked_stocks: List[Dict[str, Any]] = field(default_factory=list)
    
    # 特征重要性 (集成)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # 统计摘要
    stats: Dict[str, Any] = field(default_factory=dict)


# ==================== 时间序列交叉验证 ====================

class TimeSeriesCV:
    """时间序列交叉验证器 (源自V5.0)"""
    
    def __init__(self, n_splits: int = 5, gap: int = 5):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        test_size = n // (self.n_splits + 1)
        splits = []
        
        for i in range(self.n_splits):
            test_end = n - i * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap
            
            if train_end <= 0 or test_start >= test_end:
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))
        
        return splits[::-1]


# ==================== 模型包装器 ====================

class XGBoostModel:
    """XGBoost模型包装器"""
    
    def __init__(self, **params):
        self.params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
        }
        self.params.update(params)
        self.model = None
    
    def fit(self, X, y):
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(**self.params)
            self.model.fit(X, y, verbose=False)
        except ImportError:
            # 回退到RandomForest
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
            )
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def feature_importances(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class LightGBMModel:
    """LightGBM模型包装器"""
    
    def __init__(self, **params):
        self.params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        self.params.update(params)
        self.model = None
    
    def fit(self, X, y):
        try:
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(**self.params)
            self.model.fit(X, y)
        except ImportError:
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
            )
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def feature_importances(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class RandomForestModel:
    """随机森林模型包装器"""
    
    def __init__(self, **params):
        self.params = {
            'n_estimators': 200,
            'max_depth': 8,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1,
        }
        self.params.update(params)
        self.model = None
    
    def fit(self, X, y):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def feature_importances(self):
        return self.model.feature_importances_


# ==================== 统一模型层 ====================

class UnifiedModelLayer:
    """
    V6.0 统一模型层
    
    使用多个ML模型对因子进行建模，生成预测信号和股票排名。
    """
    
    def __init__(self, verbose: bool = True, top_n_stocks: int = 10):
        self.verbose = verbose
        self.top_n_stocks = top_n_stocks
        self.models = {}
        self.scaler = StandardScaler()
    
    def predict(self, factor_matrix: pd.DataFrame, panel: pd.DataFrame,
                candidate_stocks: List[Dict] = None,
                stock_col: str = 'stock_code', date_col: str = 'date',
                target_col: str = 'returns', forward_periods: int = 5) -> ModelLayerOutput:
        """
        执行完整的模型层处理
        
        Args:
            factor_matrix: 因子矩阵 (来自因子层)
            panel: 原始面板数据
            candidate_stocks: 候选股票列表 (来自因子层)
            stock_col: 股票代码列名
            date_col: 日期列名
            target_col: 目标变量列名
            forward_periods: 前瞻期
        
        Returns:
            ModelLayerOutput: 模型层完整输出
        """
        output = ModelLayerOutput()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 V6.0 统一模型层")
            print(f"{'='*60}")
        
        # 1. 准备训练数据
        if self.verbose:
            print(f"\n  📋 准备训练数据...")
        
        train_data = self._prepare_training_data(
            factor_matrix, panel, stock_col, date_col, target_col, forward_periods
        )
        
        if train_data is None:
            if self.verbose:
                print(f"    ⚠️ 训练数据不足，跳过模型训练")
            # 直接使用因子层的候选股票
            output.ranked_stocks = candidate_stocks or []
            return output
        
        X_train, y_train, X_latest, feature_names, latest_stocks = train_data
        
        if self.verbose:
            print(f"    ✓ 训练样本: {len(X_train)}, 特征数: {X_train.shape[1]}")
            print(f"    ✓ 预测样本: {len(X_latest)}")
        
        # 2. 训练多个模型
        model_configs = {
            'XGBoost': XGBoostModel(),
            'LightGBM': LightGBMModel(),
            'RandomForest': RandomForestModel(),
        }
        
        all_predictions = {}
        all_importances = {}
        
        for name, model in model_configs.items():
            if self.verbose:
                print(f"\n  🔧 训练 {name}...")
            
            try:
                result = self._train_and_evaluate(
                    model, name, X_train, y_train, X_latest, feature_names
                )
                output.model_results[name] = result
                
                if result.predictions is not None:
                    all_predictions[name] = result.predictions
                
                if result.feature_importance:
                    all_importances[name] = result.feature_importance
                
                if self.verbose:
                    print(f"    ✓ {name}: Train R²={result.train_score:.4f}, "
                          f"Test R²={result.test_score:.4f}")
            except Exception as e:
                if self.verbose:
                    print(f"    ✗ {name} 训练失败: {e}")
        
        # 3. 集成预测
        if all_predictions:
            pred_df = pd.DataFrame(all_predictions)
            output.ensemble_signal = pred_df.mean(axis=1)
            
            if self.verbose:
                print(f"\n  🔗 集成预测完成: {len(output.ensemble_signal)} 只股票")
        
        # 4. 集成特征重要性
        if all_importances:
            combined = pd.DataFrame(all_importances)
            output.feature_importance = combined.mean(axis=1).sort_values(ascending=False).to_dict()
            
            if self.verbose:
                print(f"\n  📊 Top 10 重要特征:")
                for i, (feat, imp) in enumerate(list(output.feature_importance.items())[:10], 1):
                    print(f"    {i:2d}. {feat:<25s} importance={imp:.4f}")
        
        # 5. 排序候选股票
        output.ranked_stocks = self._rank_stocks(
            output.ensemble_signal, latest_stocks, candidate_stocks
        )
        
        # 6. 统计摘要
        output.stats = {
            "models_trained": len(output.model_results),
            "training_samples": len(X_train),
            "features_used": X_train.shape[1],
            "ranked_stocks": len(output.ranked_stocks),
            "best_model": max(output.model_results.items(), 
                            key=lambda x: x[1].test_score)[0] if output.model_results else "N/A",
        }
        
        if self.verbose:
            print(f"\n  ✅ 模型层处理完成")
            print(f"     训练模型: {output.stats['models_trained']} 个")
            print(f"     最佳模型: {output.stats['best_model']}")
            print(f"     排序股票: {output.stats['ranked_stocks']} 只")
        
        return output
    
    def _prepare_training_data(self, factor_matrix, panel, stock_col, date_col, 
                                target_col, forward_periods):
        """准备训练数据"""
        if factor_matrix is None or len(factor_matrix) == 0:
            return None
        
        # 合并因子和目标
        data = factor_matrix.copy()
        
        # 确保有目标变量
        if target_col not in data.columns and target_col in panel.columns:
            # 从panel中合并
            merge_cols = [stock_col, date_col, target_col]
            merge_cols = [c for c in merge_cols if c in panel.columns]
            if len(merge_cols) >= 3:
                data = data.merge(panel[merge_cols], on=[stock_col, date_col], how='left')
        
        if target_col not in data.columns:
            # 从Close计算
            if 'Close' in data.columns:
                data[target_col] = data.groupby(stock_col)['Close'].pct_change()
            else:
                return None
        
        # 计算前瞻收益率
        data['forward_return'] = data.groupby(stock_col)[target_col].shift(-forward_periods)
        
        # 识别特征列
        exclude_cols = {stock_col, date_col, 'stock_name', 'industry', target_col, 
                       'forward_return', 'Open', 'High', 'Low', 'Close', 'Volume',
                       'log_returns', 'turnover'}
        feature_cols = [c for c in data.columns 
                       if c not in exclude_cols and data[c].dtype in ['float64', 'float32', 'int64']]
        
        if len(feature_cols) < 3:
            return None
        
        # 分离训练数据和最新截面
        latest_date = data[date_col].max()
        
        # 训练数据: 排除最新日期（没有前瞻收益）
        train_mask = (data[date_col] < latest_date) & data['forward_return'].notna()
        train_data = data[train_mask].copy()
        
        # 最新截面: 用于预测
        latest_data = data[data[date_col] == latest_date].copy()
        
        if len(train_data) < 50 or len(latest_data) < 5:
            return None
        
        # 处理缺失值
        for col in feature_cols:
            train_data[col] = train_data[col].fillna(train_data[col].median())
            latest_data[col] = latest_data[col].fillna(latest_data[col].median())
        
        X_train = train_data[feature_cols].values
        y_train = train_data['forward_return'].values
        X_latest = latest_data[feature_cols].values
        latest_stocks = latest_data[stock_col].values
        
        # 标准化
        X_train = self.scaler.fit_transform(X_train)
        X_latest = self.scaler.transform(X_latest)
        
        # 处理NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_latest = np.nan_to_num(X_latest, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)
        
        return X_train, y_train, X_latest, feature_cols, latest_stocks
    
    def _train_and_evaluate(self, model, name, X_train, y_train, X_latest, feature_names):
        """训练和评估单个模型"""
        result = ModelResult(model_name=name)
        
        # 时间序列交叉验证
        cv = TimeSeriesCV(n_splits=3, gap=5)
        splits = cv.split(pd.DataFrame(X_train))
        
        cv_scores = []
        for train_idx, test_idx in splits:
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            
            score = r2_score(y_te, y_pred) if len(y_te) > 1 else 0
            cv_scores.append(score)
        
        result.test_score = float(np.mean(cv_scores)) if cv_scores else 0
        
        # 使用全部训练数据重新训练
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        result.train_score = float(r2_score(y_train, train_pred))
        result.mse = float(mean_squared_error(y_train, train_pred))
        
        # 预测最新截面
        latest_pred = model.predict(X_latest)
        result.predictions = pd.Series(latest_pred)
        
        # 特征重要性
        importances = model.feature_importances()
        if importances is not None:
            result.feature_importance = {
                name: float(imp) for name, imp in zip(feature_names, importances)
            }
        
        return result
    
    def _rank_stocks(self, ensemble_signal, latest_stocks, candidate_stocks):
        """排序候选股票"""
        if ensemble_signal is None or len(ensemble_signal) == 0:
            return candidate_stocks or []
        
        # 创建候选股票信息索引
        candidate_info = {}
        if candidate_stocks:
            for s in candidate_stocks:
                candidate_info[s['code']] = s
        
        # 排序
        ranked = []
        signal_values = ensemble_signal.values
        
        # 按预测信号排序
        sorted_indices = np.argsort(-signal_values)
        
        for idx in sorted_indices[:self.top_n_stocks]:
            if idx < len(latest_stocks):
                code = latest_stocks[idx]
                info = candidate_info.get(code, {})
                
                ranked.append({
                    'code': code,
                    'name': info.get('name', code),
                    'ml_signal': float(signal_values[idx]),
                    'factor_score': info.get('composite_score', 0),
                    'combined_score': float(signal_values[idx]) * 0.6 + info.get('composite_score', 0) * 0.4,
                    'industry': info.get('industry', ''),
                    'latest_price': info.get('latest_price', 0),
                })
        
        # 按综合得分排序
        ranked.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return ranked


# ==================== 便捷函数 ====================

def run_model_prediction(factor_matrix: pd.DataFrame, panel: pd.DataFrame,
                          candidate_stocks: List[Dict] = None,
                          verbose: bool = True, top_n: int = 10) -> ModelLayerOutput:
    """
    便捷函数：运行模型预测
    """
    layer = UnifiedModelLayer(verbose=verbose, top_n_stocks=top_n)
    return layer.predict(factor_matrix, panel, candidate_stocks)


if __name__ == "__main__":
    print("=" * 60)
    print("V6.0 统一模型层测试")
    print("=" * 60)
    print("模型层需要因子层的输出作为输入，请通过MasterPipeline运行完整流程。")
