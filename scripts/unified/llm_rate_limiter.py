#!/usr/bin/env python3
"""
LLM Rate Limiter - API速率限制管理

解决 Rate limit exceeded 问题
"""

import time
import random
from typing import Optional, Callable, Any
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_minute: int = 20  # 每分钟请求数
    requests_per_day: int = 2000   # 每天请求数
    min_interval: float = 3.0      # 最小间隔(秒)
    max_retries: int = 3           # 最大重试次数
    retry_delay: float = 5.0       # 重试延迟(秒)
    exponential_backoff: bool = True  # 指数退避


class RateLimiter:
    """
    API速率限制器
    
    功能:
    1. 请求节流 - 控制请求频率
    2. 重试机制 - 自动重试失败请求
    3. 指数退避 - 避免频繁重试
    4. 多key轮询 - 支持多个API key轮换
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.request_times: list = []
        self.daily_count: int = 0
        self.last_reset: datetime = datetime.now()
        self.last_request_time: float = 0
        
        # 多key支持
        self.api_keys: list = []
        self.current_key_index: int = 0
    
    def add_api_key(self, key: str):
        """添加API key"""
        if key:
            self.api_keys.append(key)
    
    def get_current_key(self) -> Optional[str]:
        """获取当前API key"""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]
    
    def rotate_key(self):
        """轮换到下一个API key"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            print(f"[RateLimiter] 切换到API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _check_rate_limit(self) -> bool:
        """检查是否超出速率限制"""
        now = datetime.now()
        
        # 重置每日计数
        if now - self.last_reset > timedelta(days=1):
            self.daily_count = 0
            self.last_reset = now
        
        # 检查每日限制
        if self.daily_count >= self.config.requests_per_day:
            print(f"[RateLimiter] 已达到每日限制 {self.config.requests_per_day}")
            return False
        
        # 清理1分钟前的记录
        one_minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > one_minute_ago]
        
        # 检查每分钟限制
        if len(self.request_times) >= self.config.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                print(f"[RateLimiter] 达到每分钟限制，等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
        
        # 检查最小间隔
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.min_interval:
            wait = self.config.min_interval - elapsed
            time.sleep(wait)
        
        return True
    
    def _record_request(self):
        """记录请求"""
        now = datetime.now()
        self.request_times.append(now)
        self.daily_count += 1
        self.last_request_time = time.time()
    
    def call_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        带重试机制的函数调用
        
        Args:
            func: 要调用的函数
            *args, **kwargs: 函数参数
        
        Returns:
            函数返回值
        """
        for attempt in range(self.config.max_retries):
            # 检查速率限制
            if not self._check_rate_limit():
                # 尝试切换API key
                if len(self.api_keys) > 1:
                    self.rotate_key()
                    continue
                else:
                    raise Exception("Rate limit exceeded and no alternative keys available")
            
            try:
                # 记录请求
                self._record_request()
                
                # 调用函数
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # 检查是否是速率限制错误
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    if attempt < self.config.max_retries - 1:
                        # 计算重试延迟
                        if self.config.exponential_backoff:
                            delay = self.config.retry_delay * (2 ** attempt)
                        else:
                            delay = self.config.retry_delay
                        
                        # 添加随机抖动
                        delay += random.uniform(0, 1)
                        
                        print(f"[RateLimiter] 遇到速率限制，{delay:.1f}秒后重试 ({attempt + 1}/{self.config.max_retries})...")
                        time.sleep(delay)
                        
                        # 尝试切换API key
                        if len(self.api_keys) > 1:
                            self.rotate_key()
                        
                        continue
                    else:
                        print(f"[RateLimiter] 重试次数耗尽，放弃请求")
                        raise
                else:
                    # 其他错误直接抛出
                    raise
        
        raise Exception("Max retries exceeded")


class MockLLMProvider:
    """
    模拟LLM提供商
    
    当API不可用时使用，避免rate limit
    """
    
    def __init__(self):
        self.call_count = 0
    
    def call(self, prompt: str, **kwargs) -> str:
        """模拟LLM调用"""
        self.call_count += 1
        
        # 根据prompt内容返回不同的模拟响应
        import json
        
        if "财务分析" in prompt:
            return json.dumps({
                "bullish_points": ["ROE连续3年保持在18%以上", "自由现金流充裕", "当前PE 15倍低于行业平均"],
                "bearish_points": ["应收账款周转天数增加", "资本支出占营收比例上升"],
                "confidence": 0.75,
                "bias": "bullish",
                "key_factors": ["ROE稳定性", "现金流质量", "估值水平"],
                "reasoning": "财务指标整体健康，估值合理，盈利能力强"
            }, ensure_ascii=False)
        
        elif "行业研究" in prompt:
            return json.dumps({
                "bullish_points": ["行业处于成长期，未来3年CAGR预计15%", "公司市占率30%龙头地位稳固", "技术壁垒高，研发投入占比8%"],
                "bearish_points": ["新进入者增多，竞争加剧", "上游原材料价格波动大"],
                "confidence": 0.70,
                "bias": "bullish",
                "key_factors": ["行业成长性", "市场份额", "技术壁垒"],
                "reasoning": "行业前景良好，公司具有明显竞争优势"
            }, ensure_ascii=False)
        
        elif "宏观分析" in prompt:
            return json.dumps({
                "bullish_points": ["货币政策宽松，流动性充裕", "经济复苏态势明确", "行业受益于稳增长政策"],
                "bearish_points": ["通胀压力上升", "美联储可能加息", "地缘政治风险"],
                "confidence": 0.65,
                "bias": "neutral",
                "key_factors": ["货币政策", "经济周期", "通胀压力"],
                "reasoning": "宏观环境中性偏正面，但需关注通胀和外部风险"
            }, ensure_ascii=False)
        
        elif "技术分析" in prompt:
            return json.dumps({
                "bullish_points": ["股价突破前期高点，形成上升趋势", "成交量放大，资金流入明显", "MACD金叉，均线多头排列"],
                "bearish_points": ["股价接近历史高位，阻力较大", "RSI接近70，短期可能超买"],
                "confidence": 0.60,
                "bias": "bullish",
                "key_factors": ["趋势方向", "成交量", "技术指标"],
                "reasoning": "技术面偏强，上升趋势确立，但短期注意回调风险"
            }, ensure_ascii=False)
        
        elif "风险评估" in prompt:
            return json.dumps({
                "bullish_points": ["波动率20%处于可控范围", "流动性充足，日均成交额5亿以上"],
                "bearish_points": ["最大回撤可能达25%", "行业集中度高，单一行业风险"],
                "confidence": 0.65,
                "bias": "caution",
                "key_factors": ["波动率", "回撤风险", "流动性"],
                "reasoning": "风险收益比合理，但需控制仓位和设置止损"
            }, ensure_ascii=False)
        
        elif "综合决策" in prompt or "投资决策" in prompt:
            return json.dumps({
                "decision": "买入",
                "confidence": 0.72,
                "position_size": 0.15,
                "target_price": 150.0,
                "stop_loss": 120.0,
                "time_horizon": "中期",
                "logic_chain": [
                    "财务指标健康，ROE 18%，估值合理PE 15倍",
                    "行业处于成长期，公司市占率30%龙头地位",
                    "宏观环境中性偏正面，政策支持",
                    "技术面上升趋势确立，资金流入",
                    "风险可控，设置止损位保护"
                ],
                "supporting_evidence": ["连续3年ROE>18%", "行业CAGR 15%", "突破前期高点"],
                "opposing_concerns": ["应收账款增加", "竞争加剧", "短期可能超买"],
                "risk_mitigation": ["仓位控制在15%以内", "设置止损位120元", "定期跟踪财务变化"]
            }, ensure_ascii=False)
        
        elif "公司深度研究" in prompt:
            return json.dumps({
                "company_overview": "行业龙头企业，主营业务为高科技制造，具有完整的产业链布局",
                "products": {
                    "main_products": ["核心产品A", "核心产品B", "解决方案C"],
                    "tech_moat": "拥有100+核心专利，技术领先行业2-3年",
                    "rd_investment": "研发投入占比8%，高于行业平均5%"
                },
                "competition": {
                    "market_share": "30%",
                    "main_competitors": ["竞争对手X", "竞争对手Y"],
                    "advantages": ["技术领先", "品牌优势", "渠道完善"],
                    "disadvantages": ["成本偏高", "国际化程度不足"]
                },
                "industry": {
                    "lifecycle": "成长期",
                    "market_size": "1000亿",
                    "growth_rate": "15%",
                    "policy_support": "国家战略性新兴产业，享受税收优惠"
                },
                "competitor_analysis": {
                    "key_competitors": ["竞争对手X市占率20%", "竞争对手Y市占率15%"],
                    "their_strategies": ["价格战", "海外扩张"],
                    "our_differentiation": "技术差异化，高端定位"
                },
                "financial_health": {
                    "profitability": "优秀，ROE 18%",
                    "growth": "营收增长20%，利润增长25%",
                    "cashflow": "经营现金流健康，FCF/营收15%",
                    "debt": "负债率40%，财务稳健"
                }
            }, ensure_ascii=False)
        
        return "{}"


# 全局速率限制器实例
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """获取全局速率限制器"""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter()
    return _global_limiter


def configure_rate_limiter(
    requests_per_minute: int = 20,
    requests_per_day: int = 2000,
    min_interval: float = 3.0,
    max_retries: int = 3,
    api_keys: Optional[list] = None
):
    """配置速率限制器"""
    global _global_limiter
    
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_day=requests_per_day,
        min_interval=min_interval,
        max_retries=max_retries
    )
    
    _global_limiter = RateLimiter(config)
    
    if api_keys:
        for key in api_keys:
            _global_limiter.add_api_key(key)
    
    return _global_limiter


# 装饰器版本
def rate_limited(func: Callable) -> Callable:
    """速率限制装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        limiter = get_rate_limiter()
        return limiter.call_with_retry(func, *args, **kwargs)
    return wrapper


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Rate Limiter - 测试")
    print("=" * 80)
    
    # 配置速率限制器
    limiter = configure_rate_limiter(
        requests_per_minute=10,
        min_interval=2.0,
        max_retries=3
    )
    
    # 测试模拟调用
    mock = MockLLMProvider()
    
    print("\n测试速率限制...")
    for i in range(5):
        start = time.time()
        result = limiter.call_with_retry(mock.call, f"测试prompt {i}")
        elapsed = time.time() - start
        print(f"请求 {i+1}: 耗时 {elapsed:.2f}秒")
    
    print(f"\n总调用次数: {mock.call_count}")
    print("测试完成!")
