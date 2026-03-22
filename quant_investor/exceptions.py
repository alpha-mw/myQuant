"""
Quant-Investor V7.0 自定义异常类
定义具体的异常类型，避免裸异常捕获
"""


class QuantInvestorError(Exception):
    """基础异常类"""
    pass


# 数据层异常
class DataError(QuantInvestorError):
    """数据相关错误"""
    pass


class DataSourceError(DataError):
    """数据源错误"""
    def __init__(self, source: str, message: str, original_error: Exception = None):
        self.source = source
        self.original_error = original_error
        super().__init__(f"数据源 {source} 错误: {message}")


class DataValidationError(DataError):
    """数据验证错误"""
    def __init__(self, field: str, value, expected_type):
        self.field = field
        self.value = value
        self.expected_type = expected_type
        super().__init__(f"字段 {field} 验证失败: 期望 {expected_type}, 实际 {type(value)}")


class DataNotFoundError(DataError):
    """数据未找到"""
    def __init__(self, symbol: str, date_range: str = None):
        msg = f"未找到股票 {symbol} 的数据"
        if date_range:
            msg += f" (日期范围: {date_range})"
        super().__init__(msg)


# 因子层异常
class FactorError(QuantInvestorError):
    """因子计算错误"""
    pass


class FactorCalculationError(FactorError):
    """因子计算错误"""
    def __init__(self, factor_name: str, symbol: str, error: str):
        super().__init__(f"因子 {factor_name} 计算错误 ({symbol}): {error}")


class FactorValidationError(FactorError):
    """因子验证错误"""
    def __init__(self, factor_name: str, issue: str):
        super().__init__(f"因子 {factor_name} 验证失败: {issue}")


# 模型层异常
class ModelError(QuantInvestorError):
    """模型相关错误"""
    pass


class ModelTrainingError(ModelError):
    """模型训练错误"""
    def __init__(self, model_name: str, error: str):
        super().__init__(f"模型 {model_name} 训练失败: {error}")


class ModelPredictionError(ModelError):
    """模型预测错误"""
    def __init__(self, model_name: str, error: str):
        super().__init__(f"模型 {model_name} 预测失败: {error}")


# 风控层异常
class RiskError(QuantInvestorError):
    """风险管理错误"""
    pass


class RiskCalculationError(RiskError):
    """风险计算错误"""
    def __init__(self, metric: str, error: str):
        super().__init__(f"风险指标 {metric} 计算失败: {error}")


class RiskLimitExceeded(RiskError):
    """风险限额超限"""
    def __init__(self, limit_type: str, current_value: float, limit_value: float):
        super().__init__(f"风险限额超限: {limit_type} = {current_value:.2%} (限额: {limit_value:.2%})")


# 回测异常
class BacktestError(QuantInvestorError):
    """回测错误"""
    pass


class BacktestDataError(BacktestError):
    """回测数据错误"""
    def __init__(self, symbol: str, message: str):
        super().__init__(f"回测数据错误 ({symbol}): {message}")


class BacktestExecutionError(BacktestError):
    """回测执行错误"""
    def __init__(self, message: str):
        super().__init__(f"回测执行错误: {message}")


# 配置异常
class ConfigError(QuantInvestorError):
    """配置错误"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证错误"""
    def __init__(self, field: str, message: str):
        super().__init__(f"配置验证失败 [{field}]: {message}")


class ConfigNotFoundError(ConfigError):
    """配置未找到"""
    def __init__(self, config_file: str):
        super().__init__(f"配置文件未找到: {config_file}")
