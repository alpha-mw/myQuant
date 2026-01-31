#!/usr/bin/env python3
"""
表达式引擎 (Expression Engine) - quant-investor V2.3

借鉴Qlib的设计，提供简洁的因子表达式语法，简化因子构建过程。

示例表达式:
- "$close / Ref($close, 5) - 1"  # 5日收益率
- "Mean($close, 20)"              # 20日均线
- "Mean($close, 20) + 2 * Std($close, 20)"  # 布林带上轨

作者: Manus AI
日期: 2026-01-31
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable


class ExpressionEngine:
    """
    表达式引擎，将字符串表达式转换为Pandas操作
    """
    
    def __init__(self):
        """初始化表达式引擎"""
        pass
        
    def evaluate(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """
        评估表达式，返回计算结果
        
        Args:
            expression: 因子表达式字符串
            data: 包含OHLCV等字段的DataFrame
            
        Returns:
            计算后的因子值Series
        """
        # 解析并计算表达式
        result = self._parse_and_compute(expression, data)
        return result
    
    def _parse_and_compute(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """解析并计算表达式"""
        # 替换字段引用
        expr = expression
        
        # 查找所有函数调用
        func_pattern = r'(\w+)\(([^)]+)\)'
        
        # 递归处理嵌套函数
        while re.search(func_pattern, expr):
            match = re.search(func_pattern, expr)
            if match:
                func_name = match.group(1)
                func_args_str = match.group(2)
                
                # 计算函数
                result_var = self._compute_function(func_name, func_args_str, data)
                
                # 生成临时变量名
                temp_var = f"__temp_{id(result_var)}__"
                data[temp_var] = result_var
                
                # 替换表达式中的函数调用为临时变量
                expr = expr[:match.start()] + f"${temp_var}" + expr[match.end():]
        
        # 现在表达式中只剩下字段引用和算术运算
        # 替换所有$field为实际的Series
        field_pattern = r'\$(\w+)'
        fields = re.findall(field_pattern, expr)
        
        # 构建计算环境
        local_vars = {}
        for field in fields:
            if field in data.columns:
                local_vars[field] = data[field]
            else:
                raise ValueError(f"字段 ${field} 不存在于数据中")
        
        # 替换表达式中的$field为变量名
        for field in fields:
            expr = expr.replace(f'${field}', field)
        
        # 评估表达式
        try:
            result = eval(expr, {"__builtins__": {}, "np": np}, local_vars)
            
            # 清理临时变量
            for col in data.columns:
                if col.startswith('__temp_'):
                    data.drop(columns=[col], inplace=True)
            
            return result
        except Exception as e:
            raise ValueError(f"表达式评估失败: {expression}\n错误: {str(e)}")
    
    def _compute_function(self, func_name: str, args_str: str, data: pd.DataFrame) -> pd.Series:
        """计算函数调用"""
        # 解析参数
        args = [arg.strip() for arg in args_str.split(',')]
        
        # 处理参数（可能是$field或数字）
        processed_args = []
        for arg in args:
            if arg.startswith('$'):
                field_name = arg[1:]
                if field_name in data.columns:
                    processed_args.append(data[field_name])
                else:
                    raise ValueError(f"字段 {arg} 不存在于数据中")
            else:
                # 尝试转换为数字
                try:
                    processed_args.append(int(arg))
                except ValueError:
                    try:
                        processed_args.append(float(arg))
                    except ValueError:
                        raise ValueError(f"无法解析参数: {arg}")
        
        # 调用对应的函数
        if func_name == 'Ref':
            return self._func_ref(*processed_args)
        elif func_name == 'Mean':
            return self._func_mean(*processed_args)
        elif func_name == 'Std':
            return self._func_std(*processed_args)
        elif func_name == 'Sum':
            return self._func_sum(*processed_args)
        elif func_name == 'Max':
            return self._func_max(*processed_args)
        elif func_name == 'Min':
            return self._func_min(*processed_args)
        elif func_name == 'Delta':
            return self._func_delta(*processed_args)
        elif func_name == 'Rank':
            return self._func_rank(*processed_args)
        elif func_name == 'Corr':
            return self._func_corr(*processed_args)
        elif func_name == 'Cov':
            return self._func_cov(*processed_args)
        else:
            raise ValueError(f"未知函数: {func_name}")
    
    # ==================== 内置函数实现 ====================
    
    def _func_ref(self, series: pd.Series, n: int) -> pd.Series:
        """引用n个周期前的数据"""
        return series.shift(n)
    
    def _func_mean(self, series: pd.Series, n: int) -> pd.Series:
        """过去n个周期的均值"""
        return series.rolling(window=n, min_periods=1).mean()
    
    def _func_std(self, series: pd.Series, n: int) -> pd.Series:
        """过去n个周期的标准差"""
        return series.rolling(window=n, min_periods=1).std()
    
    def _func_sum(self, series: pd.Series, n: int) -> pd.Series:
        """过去n个周期的总和"""
        return series.rolling(window=n, min_periods=1).sum()
    
    def _func_max(self, series: pd.Series, n: int) -> pd.Series:
        """过去n个周期的最大值"""
        return series.rolling(window=n, min_periods=1).max()
    
    def _func_min(self, series: pd.Series, n: int) -> pd.Series:
        """过去n个周期的最小值"""
        return series.rolling(window=n, min_periods=1).min()
    
    def _func_delta(self, series: pd.Series, n: int) -> pd.Series:
        """n个周期前的差分"""
        return series - series.shift(n)
    
    def _func_rank(self, series: pd.Series) -> pd.Series:
        """截面排序（百分位）"""
        return series.rank(pct=True)
    
    def _func_corr(self, series1: pd.Series, series2: pd.Series, n: int) -> pd.Series:
        """过去n个周期的相关系数"""
        return series1.rolling(window=n, min_periods=1).corr(series2)
    
    def _func_cov(self, series1: pd.Series, series2: pd.Series, n: int) -> pd.Series:
        """过去n个周期的协方差"""
        return series1.rolling(window=n, min_periods=1).cov(series2)


def demo():
    """演示表达式引擎的使用"""
    # 创建示例数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'volume': np.random.randint(1000000, 10000000, 100),
    }, index=dates)
    
    # 创建表达式引擎
    engine = ExpressionEngine()
    
    # 测试表达式
    expressions = [
        ("5日收益率", "$close / Ref($close, 5) - 1"),
        ("20日均线", "Mean($close, 20)"),
        ("布林带上轨", "Mean($close, 20) + 2 * Std($close, 20)"),
        ("5日最大值", "Max($close, 5)"),
        ("价格动量", "Delta($close, 1)"),
    ]
    
    print("表达式引擎演示")
    print("=" * 60)
    
    for name, expr in expressions:
        try:
            result = engine.evaluate(expr, data)
            print(f"\n{name}: {expr}")
            print(f"最近5个值:\n{result.tail()}")
        except Exception as e:
            print(f"\n{name}: {expr}")
            print(f"错误: {str(e)}")


if __name__ == "__main__":
    demo()
