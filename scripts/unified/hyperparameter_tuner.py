"""
Quant-Investor V7.0 超参数调优模块
使用Optuna自动优化模型参数
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# 禁用Optuna日志
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """
    超参数调优器
    
    使用示例:
        tuner = HyperparameterTuner()
        
        # 定义搜索空间
        search_space = {
            'n_estimators': (50, 500),
            'max_depth': (3, 20),
            'min_samples_split': (2, 20),
        }
        
        # 运行优化
        best_params = tuner.optimize(
            model_class=RandomForestRegressor,
            X=X_train,
            y=y_train,
            search_space=search_space,
            n_trials=100
        )
    """
    
    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        """
        初始化调优器
        
        Args:
            n_jobs: 并行作业数
            random_state: 随机种子
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.Study] = None
    
    def optimize(self,
                 model_class: type,
                 X: pd.DataFrame,
                 y: pd.Series,
                 search_space: Dict[str, tuple],
                 n_trials: int = 100,
                 cv_splits: int = 5,
                 metric: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        运行超参数优化
        
        Args:
            model_class: 模型类
            X: 特征数据
            y: 目标数据
            search_space: 搜索空间字典，格式为 {param: (min, max)}
            n_trials: 试验次数
            cv_splits: 交叉验证折数
            metric: 评估指标
            
        Returns:
            最优参数
        """
        def objective(trial: optuna.Trial) -> float:
            # 构建参数
            params = self._build_params(trial, search_space)
            params['random_state'] = self.random_state
            
            # 创建模型
            model = model_class(**params)
            
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            
            # 评估
            scores = cross_val_score(
                model, X, y, 
                cv=tscv, 
                scoring=metric,
                n_jobs=self.n_jobs
            )
            
            return scores.mean()
        
        # 创建study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # 运行优化
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 获取最优参数
        self.best_params = self.study.best_params
        
        return self.best_params
    
    def _build_params(self, trial: optuna.Trial, 
                      search_space: Dict[str, tuple]) -> Dict[str, Any]:
        """
        构建参数字典
        
        Args:
            trial: Optuna trial
            search_space: 搜索空间
            
        Returns:
            参数字典
        """
        params = {}
        
        for param_name, (min_val, max_val) in search_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                # 整数参数
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif isinstance(min_val, float) or isinstance(max_val, float):
                # 浮点数参数
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            else:
                # 分类参数
                params[param_name] = trial.suggest_categorical(param_name, [min_val, max_val])
        
        return params
    
    def get_best_model(self, model_class: type) -> Any:
        """
        获取最优模型
        
        Args:
            model_class: 模型类
            
        Returns:
            最优模型实例
        """
        if self.best_params is None:
            raise ValueError("请先运行优化")
        
        params = self.best_params.copy()
        params['random_state'] = self.random_state
        
        return model_class(**params)
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        获取优化历史
        
        Returns:
            优化历史DataFrame
        """
        if self.study is None:
            raise ValueError("请先运行优化")
        
        return self.study.trials_dataframe()
    
    def plot_optimization_history(self):
        """绘制优化历史"""
        if self.study is None:
            raise ValueError("请先运行优化")
        
        return optuna.visualization.plot_optimization_history(self.study)
    
    def plot_param_importances(self):
        """绘制参数重要性"""
        if self.study is None:
            raise ValueError("请先运行优化")
        
        return optuna.visualization.plot_param_importances(self.study)


# 预定义的搜索空间
SEARCH_SPACES = {
    'random_forest': {
        'n_estimators': (50, 500),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': ('sqrt', 'log2', None),
    },
    'gradient_boosting': {
        'n_estimators': (50, 500),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'min_samples_split': (2, 20),
        'subsample': (0.6, 1.0),
    },
    'xgboost': {
        'n_estimators': (50, 500),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 10),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
    },
}


def optimize_model(model_name: str,
                   X: pd.DataFrame,
                   y: pd.Series,
                   n_trials: int = 100) -> Dict[str, Any]:
    """
    便捷函数：优化指定模型
    
    Args:
        model_name: 模型名称 ('random_forest', 'gradient_boosting', 'xgboost')
        X: 特征数据
        y: 目标数据
        n_trials: 试验次数
        
    Returns:
        最优参数
    """
    tuner = HyperparameterTuner()
    
    # 获取搜索空间
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"未知模型: {model_name}")
    
    search_space = SEARCH_SPACES[model_name]
    
    # 获取模型类
    if model_name == 'random_forest':
        model_class = RandomForestRegressor
    elif model_name == 'gradient_boosting':
        model_class = GradientBoostingRegressor
    else:
        raise ValueError(f"暂不支持模型: {model_name}")
    
    # 运行优化
    best_params = tuner.optimize(
        model_class=model_class,
        X=X,
        y=y,
        search_space=search_space,
        n_trials=n_trials
    )
    
    print(f"\n最优参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params
