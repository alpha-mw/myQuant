"""
Quant-Investor V7.0 异步LLM调用模块
实现并发处理提高性能
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import functools


class AsyncLLMCaller:
    """
    异步LLM调用器
    
    使用示例:
        caller = AsyncLLMCaller(max_concurrent=5)
        
        # 并发调用多个请求
        prompts = ["分析股票A", "分析股票B", "分析股票C"]
        results = await caller.batch_call(prompts)
    """
    
    def __init__(self, max_concurrent: int = 5, timeout: int = 30):
        """
        初始化异步调用器
        
        Args:
            max_concurrent: 最大并发数
            timeout: 超时时间（秒）
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def call_single(self, 
                          call_func: Callable,
                          *args,
                          **kwargs) -> Any:
        """
        单个异步调用
        
        Args:
            call_func: 调用函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            调用结果
        """
        async with self.semaphore:
            # 在线程池中执行同步函数
            loop = asyncio.get_event_loop()
            
            # 使用functools.partial包装函数和参数
            partial_func = functools.partial(call_func, *args, **kwargs)
            
            # 在线程池中执行
            result = await asyncio.wait_for(
                loop.run_in_executor(None, partial_func),
                timeout=self.timeout
            )
            
            return result
    
    async def batch_call(self,
                         call_func: Callable,
                         args_list: List[tuple],
                         kwargs_list: Optional[List[dict]] = None) -> List[Any]:
        """
        批量异步调用
        
        Args:
            call_func: 调用函数
            args_list: 参数列表
            kwargs_list: 关键字参数列表
            
        Returns:
            结果列表
        """
        if kwargs_list is None:
            kwargs_list = [{} for _ in args_list]
        
        # 创建任务
        tasks = []
        for args, kwargs in zip(args_list, kwargs_list):
            task = self.call_single(call_func, *args, **kwargs)
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def call_with_retry(self,
                              call_func: Callable,
                              *args,
                              max_retries: int = 3,
                              **kwargs) -> Any:
        """
        带重试的异步调用
        
        Args:
            call_func: 调用函数
            *args: 位置参数
            max_retries: 最大重试次数
            **kwargs: 关键字参数
            
        Returns:
            调用结果
        """
        for attempt in range(max_retries):
            try:
                return await self.call_single(call_func, *args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # 指数退避


class AsyncDataFetcher:
    """
    异步数据获取器
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        初始化获取器
        
        Args:
            max_concurrent: 最大并发数
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_url(self, 
                        session: aiohttp.ClientSession,
                        url: str,
                        **kwargs) -> Dict:
        """
        异步获取URL
        
        Args:
            session: aiohttp会话
            url: URL
            **kwargs: 请求参数
            
        Returns:
            响应数据
        """
        async with self.semaphore:
            async with session.get(url, **kwargs) as response:
                return await response.json()
    
    async def batch_fetch(self,
                          urls: List[str],
                          **kwargs) -> List[Dict]:
        """
        批量异步获取
        
        Args:
            urls: URL列表
            **kwargs: 请求参数
            
        Returns:
            响应数据列表
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url, **kwargs) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results


class AsyncFactorCalculator:
    """
    异步因子计算器
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化计算器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def calculate_factors_async(self,
                                       df: Any,
                                       factor_functions: Dict[str, Callable]) -> Dict[str, Any]:
        """
        异步计算多个因子
        
        Args:
            df: 数据DataFrame
            factor_functions: 因子函数字典
            
        Returns:
            因子结果字典
        """
        loop = asyncio.get_event_loop()
        
        # 创建任务
        tasks = {}
        for name, func in factor_functions.items():
            task = loop.run_in_executor(self.executor, func, df)
            tasks[name] = task
        
        # 等待所有任务完成
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                results[name] = None
                print(f"计算因子 {name} 失败: {e}")
        
        return results
    
    def close(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)


# 便捷函数
async def async_batch_process(items: List[Any],
                               process_func: Callable,
                               max_concurrent: int = 5) -> List[Any]:
    """
    批量异步处理
    
    Args:
        items: 待处理项列表
        process_func: 处理函数
        max_concurrent: 最大并发数
        
    Returns:
        处理结果列表
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(item):
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, process_func, item)
    
    tasks = [process_with_limit(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results


def run_async(coro):
    """
    运行异步函数（同步接口）
    
    Args:
        coro: 协程
        
    Returns:
        结果
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
