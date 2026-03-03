"""
Quant-Investor V7.0 SHAP模型解释模块
提供模型可解释性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 尝试导入shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP未安装，模型解释功能不可用")


class ModelExplainer:
    """
    模型解释器
    
    使用示例:
        explainer = ModelExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.explain(X_test)
        
        # 绘制特征重要性
        explainer.plot_feature_importance(feature_names)
        
        # 解释单个预测
        explainer.explain_single(X_test[0], feature_names)
    """
    
    def __init__(self, model: Any, background_data: Optional[np.ndarray] = None):
        """
        初始化解释器
        
        Args:
            model: 训练好的模型
            background_data: 背景数据（用于SHAP计算）
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP未安装，请运行: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.explainer: Optional[Any] = None
        self.shap_values: Optional[np.ndarray] = None
        
        # 创建SHAP解释器
        self._create_explainer()
    
    def _create_explainer(self):
        """创建SHAP解释器"""
        try:
            # 尝试使用TreeExplainer（适用于树模型）
            self.explainer = shap.TreeExplainer(self.model)
        except:
            try:
                # 使用KernelExplainer（通用）
                if self.background_data is not None:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict, 
                        shap.sample(self.background_data, 100)
                    )
                else:
                    print("警告: 非树模型需要提供background_data")
            except Exception as e:
                print(f"创建解释器失败: {e}")
    
    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            X: 特征数据
            
        Returns:
            SHAP值
        """
        if self.explainer is None:
            raise ValueError("解释器未创建")
        
        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # 计算SHAP值
        self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        if self.shap_values is None:
            raise ValueError("请先调用explain()计算SHAP值")
        
        # 计算平均绝对SHAP值
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # 创建DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def plot_feature_importance(self, feature_names: Optional[List[str]] = None, 
                                top_n: int = 20):
        """
        绘制特征重要性图
        
        Args:
            feature_names: 特征名称列表
            top_n: 显示前N个特征
        """
        if self.shap_values is None:
            raise ValueError("请先调用explain()计算SHAP值")
        
        # 获取特征重要性
        importance_df = self.get_feature_importance(feature_names)
        
        # 绘制
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:top_n][::-1], 
                importance_df['importance'][:top_n][::-1])
        plt.xlabel('SHAP Importance (mean |SHAP value|)')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_summary(self, X: Union[pd.DataFrame, np.ndarray],
                     feature_names: Optional[List[str]] = None):
        """
        绘制SHAP摘要图
        
        Args:
            X: 特征数据
            feature_names: 特征名称列表
        """
        if self.shap_values is None:
            self.explain(X)
        
        # 设置特征名称
        if feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        
        # 绘制摘要图
        shap.summary_plot(self.shap_values, X, show=True)
    
    def explain_single(self, x: Union[pd.Series, np.ndarray],
                       feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        解释单个预测
        
        Args:
            x: 单个样本
            feature_names: 特征名称列表
            
        Returns:
            特征贡献DataFrame
        """
        if self.explainer is None:
            raise ValueError("解释器未创建")
        
        # 转换为numpy数组
        if isinstance(x, pd.Series):
            x = x.values
        
        # 确保是2D数组
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # 计算SHAP值
        shap_values = self.explainer.shap_values(x)
        
        # 设置特征名称
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(x.shape[1])]
        
        # 创建DataFrame
        df = pd.DataFrame({
            'feature': feature_names,
            'value': x[0],
            'shap_value': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            'abs_shap': np.abs(shap_values[0] if len(shap_values.shape) > 1 else shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        return df
    
    def plot_waterfall(self, x: Union[pd.Series, np.ndarray],
                       feature_names: Optional[List[str]] = None):
        """
        绘制瀑布图（单个预测解释）
        
        Args:
            x: 单个样本
            feature_names: 特征名称列表
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP未安装")
        
        # 转换为numpy数组
        if isinstance(x, pd.Series):
            x = x.values
        
        # 确保是2D数组
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # 设置特征名称
        if feature_names is not None:
            x_display = pd.DataFrame(x, columns=feature_names)
        else:
            x_display = x
        
        # 计算SHAP值
        shap_values = self.explainer.shap_values(x)
        
        # 绘制瀑布图
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            base_values=self.explainer.expected_value,
            data=x_display.iloc[0] if isinstance(x_display, pd.DataFrame) else x[0],
            feature_names=feature_names if feature_names else [f"f{i}" for i in range(x.shape[1])]
        ))
        plt.tight_layout()
        plt.show()


def explain_model_prediction(model: Any,
                             X: Union[pd.DataFrame, np.ndarray],
                             feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    便捷函数：解释模型预测
    
    Args:
        model: 训练好的模型
        X: 特征数据
        feature_names: 特征名称列表
        
    Returns:
        特征重要性DataFrame
    """
    explainer = ModelExplainer(model)
    explainer.explain(X)
    
    importance_df = explainer.get_feature_importance(feature_names)
    
    print("\n特征重要性排名:")
    print(importance_df.head(10).to_string(index=False))
    
    return importance_df
