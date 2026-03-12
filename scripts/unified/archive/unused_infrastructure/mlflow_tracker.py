"""
Quant-Investor V7.0 MLOps模块
MLflow集成实现模型版本管理
"""

import os
import json
import pickle
from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path

# 尝试导入mlflow
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow未安装，MLOps功能不可用")

from config import config


class ModelRegistry:
    """
    模型注册表
    
    使用示例:
        registry = ModelRegistry()
        
        # 记录实验
        with registry.start_run("momentum_strategy"):
            model.fit(X_train, y_train)
            registry.log_model(model, "random_forest")
            registry.log_metrics({"accuracy": 0.85, "f1": 0.82})
            registry.log_params({"n_estimators": 100, "max_depth": 10})
        
        # 加载模型
        model = registry.load_model("momentum_strategy", version=1)
    """
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "quant_investor"):
        """
        初始化模型注册表
        
        Args:
            tracking_uri: MLflow跟踪URI
            experiment_name: 实验名称
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow未安装，请运行: pip install mlflow")
        
        self.tracking_uri = tracking_uri or os.environ.get('MLFLOW_TRACKING_URI', './mlruns')
        self.experiment_name = experiment_name
        self.current_run = None
        
        # 设置跟踪URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # 设置实验
        mlflow.set_experiment(experiment_name)
        
        # 创建客户端
        self.client = MlflowClient()
    
    def start_run(self, run_name: str = None, nested: bool = False):
        """
        开始运行
        
        Args:
            run_name: 运行名称
            nested: 是否嵌套运行
            
        Returns:
            MLflow运行上下文
        """
        return mlflow.start_run(run_name=run_name, nested=nested)
    
    def end_run(self):
        """结束当前运行"""
        mlflow.end_run()
    
    def log_model(self, model: Any, model_name: str, 
                  artifact_path: str = "model"):
        """
        记录模型
        
        Args:
            model: 模型对象
            model_name: 模型名称
            artifact_path: 工件路径
        """
        if not MLFLOW_AVAILABLE:
            return
        
        # 记录模型
        mlflow.sklearn.log_model(model, artifact_path)
        
        # 注册模型
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model(model_uri, model_name)
    
    def log_params(self, params: Dict[str, Any]):
        """
        记录参数
        
        Args:
            params: 参数字典
        """
        if not MLFLOW_AVAILABLE:
            return
        
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步骤
        """
        if not MLFLOW_AVAILABLE:
            return
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        记录工件
        
        Args:
            local_path: 本地路径
            artifact_path: 工件路径
        """
        if not MLFLOW_AVAILABLE:
            return
        
        mlflow.log_artifact(local_path, artifact_path)
    
    def load_model(self, model_name: str, version: Optional[int] = None,
                   stage: Optional[str] = None) -> Any:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            version: 版本号
            stage: 阶段 (staging/production)
            
        Returns:
            模型对象
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow未安装")
        
        if version is not None:
            model_uri = f"models:/{model_name}/{version}"
        elif stage is not None:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        return mlflow.sklearn.load_model(model_uri)
    
    def get_experiment_runs(self, experiment_id: Optional[str] = None) -> List[Dict]:
        """
        获取实验运行列表
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            运行列表
        """
        if not MLFLOW_AVAILABLE:
            return []
        
        experiment_id = experiment_id or self.client.get_experiment_by_name(
            self.experiment_name
        ).experiment_id
        
        runs = self.client.search_runs(experiment_ids=[experiment_id])
        
        return [{
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', ''),
            'status': run.info.status,
            'start_time': run.info.start_time,
            'params': dict(run.data.params),
            'metrics': dict(run.data.metrics),
        } for run in runs]
    
    def transition_model_stage(self, model_name: str, version: int, 
                               stage: str, archive_existing_versions: bool = False):
        """
        转换模型阶段
        
        Args:
            model_name: 模型名称
            version: 版本号
            stage: 目标阶段 (staging/production/archived)
            archive_existing_versions: 是否归档现有版本
        """
        if not MLFLOW_AVAILABLE:
            return
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
    
    def compare_models(self, model_names: List[str], metric: str = 'accuracy') -> pd.DataFrame:
        """
        比较多个模型
        
        Args:
            model_names: 模型名称列表
            metric: 比较指标
            
        Returns:
            比较结果DataFrame
        """
        if not MLFLOW_AVAILABLE:
            return pd.DataFrame()
        
        import pandas as pd
        
        results = []
        for model_name in model_names:
            versions = self.client.get_latest_versions(model_name)
            for version in versions:
                run = self.client.get_run(version.run_id)
                results.append({
                    'model_name': model_name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'metric': run.data.metrics.get(metric, None),
                    'created_time': version.creation_timestamp,
                })
        
        return pd.DataFrame(results)


class ExperimentTracker:
    """
    实验跟踪器
    简化实验记录
    """
    
    def __init__(self, experiment_name: str = "quant_investor"):
        """
        初始化实验跟踪器
        
        Args:
            experiment_name: 实验名称
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow未安装")
        
        self.registry = ModelRegistry(experiment_name=experiment_name)
        self.current_run = None
    
    def __enter__(self):
        """上下文管理器入口"""
        self.current_run = self.registry.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.registry.end_run()
        self.current_run = None
    
    def log(self, model: Any = None, params: Dict = None, 
            metrics: Dict = None, artifacts: List[str] = None):
        """
        记录实验数据
        
        Args:
            model: 模型
            params: 参数
            metrics: 指标
            artifacts: 工件路径列表
        """
        if params:
            self.registry.log_params(params)
        
        if metrics:
            self.registry.log_metrics(metrics)
        
        if model:
            self.registry.log_model(model, "model")
        
        if artifacts:
            for artifact in artifacts:
                self.registry.log_artifact(artifact)


# 便捷函数
def track_experiment(experiment_name: str = "quant_investor"):
    """
    实验跟踪装饰器
    
    使用示例:
        @track_experiment()
        def train_model(X, y, params):
            model = RandomForestRegressor(**params)
            model.fit(X, y)
            return model, {"accuracy": 0.85}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ExperimentTracker(experiment_name) as tracker:
                result = func(*args, **kwargs)
                
                # 如果返回元组 (model, metrics)
                if isinstance(result, tuple) and len(result) == 2:
                    model, metrics = result
                    tracker.log(model=model, metrics=metrics)
                
                return result
        return wrapper
    return decorator
