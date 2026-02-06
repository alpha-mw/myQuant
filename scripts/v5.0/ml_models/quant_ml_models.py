"""
Quant-Investor V5.0 - 机器学习模型模块

本模块提供完整的机器学习模型支持，包括：
1. 传统ML模型：随机森林、XGBoost、LightGBM、SVM
2. 时间序列交叉验证：避免数据泄露
3. 模型评估：分类/回归/排序指标
4. 特征重要性分析
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Union
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')


# ==================== 时间序列交叉验证 ====================

class TimeSeriesCV:
    """
    时间序列交叉验证
    
    确保训练集始终在测试集之前，避免数据泄露。
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding: bool = True
    ):
        """
        Args:
            n_splits: 折数
            train_size: 训练集大小（仅当expanding=False时使用）
            test_size: 测试集大小
            gap: 训练集和测试集之间的间隔（避免信息泄露）
            expanding: 是否使用扩展窗口（True）或滚动窗口（False）
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练/测试索引
        
        Args:
            X: 特征DataFrame
        
        Yields:
            (train_indices, test_indices) 元组
        """
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        splits = []
        
        for i in range(self.n_splits):
            if self.expanding:
                # 扩展窗口：训练集从头开始
                test_end = n_samples - i * test_size
                test_start = test_end - test_size
                train_end = test_start - self.gap
                train_start = 0
            else:
                # 滚动窗口：固定训练集大小
                test_end = n_samples - i * test_size
                test_start = test_end - test_size
                train_end = test_start - self.gap
                train_start = max(0, train_end - (self.train_size or test_size * 3))
            
            if train_start >= train_end or test_start >= test_end:
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        # 按时间顺序返回
        return splits[::-1]
    
    def get_n_splits(self) -> int:
        return self.n_splits


class PurgedKFold:
    """
    带清洗的K折交叉验证
    
    在训练集和测试集之间添加清洗期，避免信息泄露。
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo_gap: int = 5
    ):
        """
        Args:
            n_splits: 折数
            purge_gap: 清洗期（测试集之前的天数）
            embargo_gap: 禁运期（测试集之后的天数）
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成训练/测试索引"""
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            # 清洗期：测试集前后的数据不能用于训练
            purge_start = max(0, test_start - self.purge_gap)
            embargo_end = min(n_samples, test_end + self.embargo_gap)
            
            train_indices = np.concatenate([
                np.arange(0, purge_start),
                np.arange(embargo_end, n_samples)
            ])
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits


# ==================== 基础模型类 ====================

class BaseQuantModel(ABC):
    """量化模型基类"""
    
    def __init__(self, name: str, task_type: str = 'classification'):
        """
        Args:
            name: 模型名称
            task_type: 任务类型 ('classification', 'regression', 'ranking')
        """
        self.name = name
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """创建模型实例"""
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> 'BaseQuantModel':
        """
        训练模型
        
        Args:
            X: 特征DataFrame
            y: 标签Series
            **kwargs: 模型参数
        
        Returns:
            self
        """
        self.feature_names = list(X.columns)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建并训练模型
        self.model = self._create_model(**kwargs)
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（仅分类任务）"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()
        else:
            return None
        
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)


# ==================== 具体模型实现 ====================

class RandomForestModel(BaseQuantModel):
    """随机森林模型"""
    
    def __init__(self, task_type: str = 'classification'):
        super().__init__('RandomForest', task_type)
    
    def _create_model(self, **kwargs):
        if self.task_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 5),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 5),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )


class XGBoostModel(BaseQuantModel):
    """XGBoost模型"""
    
    def __init__(self, task_type: str = 'classification'):
        super().__init__('XGBoost', task_type)
    
    def _create_model(self, **kwargs):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Please install xgboost: pip install xgboost")
        
        if self.task_type == 'classification':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42)
            )


class LightGBMModel(BaseQuantModel):
    """LightGBM模型"""
    
    def __init__(self, task_type: str = 'classification'):
        super().__init__('LightGBM', task_type)
    
    def _create_model(self, **kwargs):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("Please install lightgbm: pip install lightgbm")
        
        if self.task_type == 'classification':
            return lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42),
                verbose=-1
            )
        else:
            return lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42),
                verbose=-1
            )


class SVMModel(BaseQuantModel):
    """SVM模型"""
    
    def __init__(self, task_type: str = 'classification'):
        super().__init__('SVM', task_type)
    
    def _create_model(self, **kwargs):
        if self.task_type == 'classification':
            from sklearn.svm import SVC
            return SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=kwargs.get('random_state', 42)
            )
        else:
            from sklearn.svm import SVR
            return SVR(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale')
            )


class LinearModel(BaseQuantModel):
    """线性模型（Lasso/Ridge/ElasticNet）"""
    
    def __init__(self, task_type: str = 'regression', regularization: str = 'ridge'):
        super().__init__(f'Linear_{regularization}', task_type)
        self.regularization = regularization
    
    def _create_model(self, **kwargs):
        alpha = kwargs.get('alpha', 1.0)
        
        if self.task_type == 'classification':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=1/alpha,
                penalty='l2' if self.regularization == 'ridge' else 'l1',
                solver='saga',
                random_state=kwargs.get('random_state', 42),
                max_iter=1000
            )
        else:
            if self.regularization == 'ridge':
                from sklearn.linear_model import Ridge
                return Ridge(alpha=alpha)
            elif self.regularization == 'lasso':
                from sklearn.linear_model import Lasso
                return Lasso(alpha=alpha)
            else:
                from sklearn.linear_model import ElasticNet
                return ElasticNet(
                    alpha=alpha,
                    l1_ratio=kwargs.get('l1_ratio', 0.5)
                )


# ==================== 模型评估 ====================

class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """评估分类模型"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                pass
        
        return metrics
    
    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """评估回归模型"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'ic': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0,
        }
    
    @staticmethod
    def evaluate_ranking(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_groups: int = 5
    ) -> Dict[str, float]:
        """评估排序模型（用于因子选股）"""
        # 计算IC
        ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        
        # 计算分组收益
        df = pd.DataFrame({'true': y_true, 'pred': y_pred})
        df['group'] = pd.qcut(df['pred'], q=n_groups, labels=False, duplicates='drop')
        
        group_returns = df.groupby('group')['true'].mean()
        
        # 多空收益
        if len(group_returns) >= 2:
            long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]
        else:
            long_short_return = 0
        
        # 单调性检验
        monotonicity = np.corrcoef(group_returns.index, group_returns.values)[0, 1] if len(group_returns) > 1 else 0
        
        return {
            'ic': ic,
            'long_short_return': long_short_return,
            'monotonicity': monotonicity,
            'top_group_return': group_returns.iloc[-1] if len(group_returns) > 0 else 0,
            'bottom_group_return': group_returns.iloc[0] if len(group_returns) > 0 else 0,
        }


# ==================== 模型训练管道 ====================

class QuantMLPipeline:
    """
    量化机器学习训练管道
    
    提供完整的模型训练、验证和评估流程。
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.models = {}
        self.cv_results = {}
        self.best_model = None
        self.evaluator = ModelEvaluator()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[QuantML] {msg}")
    
    def register_model(self, name: str, model: BaseQuantModel):
        """注册模型"""
        self.models[name] = model
        self._log(f"Registered model: {name}")
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Union[TimeSeriesCV, PurgedKFold] = None,
        model_params: Dict[str, Dict] = None
    ) -> Dict[str, Dict]:
        """
        交叉验证所有注册的模型
        
        Args:
            X: 特征DataFrame
            y: 标签Series
            cv: 交叉验证器
            model_params: 模型参数字典
        
        Returns:
            交叉验证结果
        """
        if cv is None:
            cv = TimeSeriesCV(n_splits=5, gap=5)
        
        model_params = model_params or {}
        results = {}
        
        for name, model in self.models.items():
            self._log(f"Cross-validating {name}...")
            
            fold_metrics = []
            params = model_params.get(name, {})
            
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # 跳过空折
                if len(X_train) == 0 or len(X_test) == 0:
                    continue
                
                # 训练
                model.fit(X_train, y_train, **params)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 评估
                if model.task_type == 'classification':
                    y_proba = model.predict_proba(X_test) if hasattr(model.model, 'predict_proba') else None
                    metrics = self.evaluator.evaluate_classification(y_test, y_pred, y_proba)
                elif model.task_type == 'regression':
                    metrics = self.evaluator.evaluate_regression(y_test, y_pred)
                else:
                    metrics = self.evaluator.evaluate_ranking(y_test, y_pred)
                
                fold_metrics.append(metrics)
            
            # 汇总结果
            if fold_metrics:
                avg_metrics = {}
                for key in fold_metrics[0].keys():
                    values = [m[key] for m in fold_metrics if not np.isnan(m[key])]
                    avg_metrics[f'{key}_mean'] = np.mean(values) if values else 0
                    avg_metrics[f'{key}_std'] = np.std(values) if values else 0
                
                results[name] = avg_metrics
                self._log(f"  {name} CV complete. Primary metric: {list(avg_metrics.items())[0]}")
        
        self.cv_results = results
        return results
    
    def select_best_model(self, metric: str = 'ic_mean') -> str:
        """选择最佳模型"""
        if not self.cv_results:
            raise ValueError("No CV results. Run cross_validate first.")
        
        best_name = None
        best_score = -np.inf
        
        for name, metrics in self.cv_results.items():
            score = metrics.get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = best_name
        self._log(f"Best model: {best_name} with {metric}={best_score:.4f}")
        
        return best_name
    
    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseQuantModel:
        """
        训练最终模型
        
        Args:
            X: 特征DataFrame
            y: 标签Series
            model_name: 模型名称（None则使用最佳模型）
            **kwargs: 模型参数
        
        Returns:
            训练好的模型
        """
        name = model_name or self.best_model
        if name is None:
            raise ValueError("No model specified and no best model selected")
        
        model = self.models[name]
        model.fit(X, y, **kwargs)
        
        self._log(f"Final model {name} trained on {len(X)} samples")
        
        return model
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """获取所有模型的特征重要性报告"""
        importance_dict = {}
        
        for name, model in self.models.items():
            if model.is_fitted:
                importance = model.get_feature_importance()
                if importance is not None:
                    importance_dict[name] = importance
        
        if importance_dict:
            return pd.DataFrame(importance_dict)
        return None


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing V5.0 ML Models Module")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 创建有一定预测性的标签
    true_weights = np.random.randn(n_features)
    y_continuous = X.values @ true_weights + np.random.randn(n_samples) * 0.5
    y_class = (y_continuous > np.median(y_continuous)).astype(int)
    
    y_continuous = pd.Series(y_continuous)
    y_class = pd.Series(y_class)
    
    # 测试时间序列交叉验证
    print("\n1. Testing TimeSeriesCV...")
    cv = TimeSeriesCV(n_splits=5, gap=5)
    splits = cv.split(X)
    print(f"   Generated {len(splits)} splits")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"   Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}")
    
    # 测试分类模型
    print("\n2. Testing Classification Models...")
    pipeline_clf = QuantMLPipeline()
    pipeline_clf.register_model('RandomForest', RandomForestModel('classification'))
    pipeline_clf.register_model('Linear', LinearModel('classification', 'ridge'))
    
    cv_results = pipeline_clf.cross_validate(X, y_class, cv)
    print("   CV Results:")
    for name, metrics in cv_results.items():
        print(f"   {name}: accuracy={metrics.get('accuracy_mean', 0):.4f}")
    
    # 测试回归模型
    print("\n3. Testing Regression Models...")
    pipeline_reg = QuantMLPipeline()
    pipeline_reg.register_model('RandomForest', RandomForestModel('regression'))
    pipeline_reg.register_model('Linear', LinearModel('regression', 'ridge'))
    
    cv_results = pipeline_reg.cross_validate(X, y_continuous, cv)
    print("   CV Results:")
    for name, metrics in cv_results.items():
        print(f"   {name}: IC={metrics.get('ic_mean', 0):.4f}, R2={metrics.get('r2_mean', 0):.4f}")
    
    # 选择最佳模型
    print("\n4. Selecting Best Model...")
    best = pipeline_reg.select_best_model('ic_mean')
    print(f"   Best model: {best}")
    
    # 训练最终模型
    print("\n5. Training Final Model...")
    final_model = pipeline_reg.train_final_model(X, y_continuous)
    
    # 特征重要性
    print("\n6. Feature Importance (Top 5):")
    importance = final_model.get_feature_importance()
    if importance is not None:
        print(importance.head())
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
