"""
Quant-Investor V7.0 Redis缓存管理器
实现数据和计算结果缓存
"""

import json
import pickle
import hashlib
from typing import Any, Optional, Union
from datetime import timedelta
import redis
from config import config


class CacheManager:
    """
    缓存管理器
    
    使用示例:
        cache = CacheManager()
        
        # 缓存数据
        cache.set("stock_data:000001", data, expire=3600)
        
        # 获取数据
        data = cache.get("stock_data:000001")
    """
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 db: int = None,
                 enabled: bool = True):
        """
        初始化缓存管理器
        
        Args:
            host: Redis主机
            port: Redis端口
            db: Redis数据库
            enabled: 是否启用缓存
        """
        self.enabled = enabled
        self._redis: Optional[redis.Redis] = None
        
        if enabled:
            try:
                self._redis = redis.Redis(
                    host=host or config.REDIS_HOST,
                    port=port or config.REDIS_PORT,
                    db=db or config.REDIS_DB,
                    decode_responses=False,
                    socket_connect_timeout=5
                )
                # 测试连接
                self._redis.ping()
            except Exception as e:
                print(f"Redis连接失败: {e}")
                self._redis = None
                self.enabled = False
    
    def _make_key(self, key: str) -> str:
        """
        生成缓存键
        
        Args:
            key: 原始键
            
        Returns:
            带前缀的键
        """
        return f"quant_investor:v7:{key}"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值或默认值
        """
        if not self.enabled or self._redis is None:
            return default
        
        try:
            full_key = self._make_key(key)
            data = self._redis.get(full_key)
            
            if data is None:
                return default
            
            # 反序列化
            return pickle.loads(data)
        except Exception as e:
            print(f"缓存读取失败: {e}")
            return default
    
    def set(self, key: str, value: Any, 
            expire: Union[int, timedelta] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒）
            
        Returns:
            是否成功
        """
        if not self.enabled or self._redis is None:
            return False
        
        try:
            full_key = self._make_key(key)
            
            # 序列化
            data = pickle.dumps(value)
            
            # 设置过期时间
            if isinstance(expire, timedelta):
                expire = int(expire.total_seconds())
            
            self._redis.set(full_key, data, ex=expire)
            return True
        except Exception as e:
            print(f"缓存写入失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功
        """
        if not self.enabled or self._redis is None:
            return False
        
        try:
            full_key = self._make_key(key)
            self._redis.delete(full_key)
            return True
        except Exception as e:
            print(f"缓存删除失败: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        if not self.enabled or self._redis is None:
            return False
        
        try:
            full_key = self._make_key(key)
            return self._redis.exists(full_key) > 0
        except Exception:
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        清除匹配模式的缓存
        
        Args:
            pattern: 匹配模式
            
        Returns:
            删除的键数量
        """
        if not self.enabled or self._redis is None:
            return 0
        
        try:
            full_pattern = self._make_key(pattern)
            keys = self._redis.keys(full_pattern)
            
            if keys:
                return self._redis.delete(*keys)
            return 0
        except Exception as e:
            print(f"缓存清除失败: {e}")
            return 0
    
    def get_or_set(self, key: str, factory: callable, 
                   expire: Union[int, timedelta] = None) -> Any:
        """
        获取或设置缓存
        
        Args:
            key: 缓存键
            factory: 数据生成函数
            expire: 过期时间
            
        Returns:
            缓存值
        """
        # 尝试获取缓存
        value = self.get(key)
        
        if value is not None:
            return value
        
        # 生成数据
        value = factory()
        
        # 写入缓存
        self.set(key, value, expire)
        
        return value
    
    def cache_decorator(self, expire: Union[int, timedelta] = 3600):
        """
        缓存装饰器
        
        Args:
            expire: 过期时间（秒）
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 生成缓存键
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
                key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
                
                # 尝试获取缓存
                value = self.get(key)
                
                if value is not None:
                    return value
                
                # 执行函数
                value = func(*args, **kwargs)
                
                # 写入缓存
                self.set(key, value, expire)
                
                return value
            
            return wrapper
        return decorator


# 全局缓存实例
_cache_instance: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """获取全局缓存实例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


def cached(expire: Union[int, timedelta] = 3600):
    """
    缓存装饰器（使用全局缓存）
    
    使用示例:
        @cached(expire=3600)
        def get_stock_data(symbol: str):
            # 获取数据
            return data
    """
    return get_cache().cache_decorator(expire)
