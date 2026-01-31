#!/usr/bin/env python3
"""
特征缓存系统 (Feature Caching System) - quant-investor V2.3

自动缓存特征工程的结果，避免重复计算，显著提升性能。

作者: Manus AI
日期: 2026-01-31
"""

import os
import hashlib
import pickle
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Any
import time


class FeatureCache:
    """
    特征缓存管理器
    """
    
    def __init__(self, cache_dir: str = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径，默认为~/.quant-investor/cache/
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.quant-investor/cache/')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[FeatureCache] 缓存目录: {self.cache_dir}")
    
    def get_or_compute(
        self, 
        key: str, 
        compute_func: Callable[[], pd.DataFrame],
        force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        获取缓存或计算特征
        
        Args:
            key: 缓存键（唯一标识）
            compute_func: 计算函数，返回DataFrame
            force_recompute: 是否强制重新计算
            
        Returns:
            特征DataFrame
        """
        cache_file = self._get_cache_file(key)
        
        # 检查缓存是否存在
        if not force_recompute and cache_file.exists():
            print(f"[FeatureCache] 从缓存加载: {key}")
            return self._load_cache(cache_file)
        
        # 计算特征
        print(f"[FeatureCache] 计算特征: {key}")
        start_time = time.time()
        result = compute_func()
        elapsed_time = time.time() - start_time
        print(f"[FeatureCache] 计算完成，耗时: {elapsed_time:.2f}秒")
        
        # 保存到缓存
        self._save_cache(cache_file, result)
        print(f"[FeatureCache] 已保存到缓存")
        
        return result
    
    def _get_cache_file(self, key: str) -> Path:
        """根据键生成缓存文件路径"""
        # 使用MD5哈希生成文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _save_cache(self, cache_file: Path, data: pd.DataFrame):
        """保存缓存"""
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_cache(self, cache_file: Path) -> pd.DataFrame:
        """加载缓存"""
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def clear_cache(self, key: Optional[str] = None):
        """
        清理缓存
        
        Args:
            key: 如果指定，只清理该键的缓存；否则清理所有缓存
        """
        if key is not None:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
                print(f"[FeatureCache] 已清理缓存: {key}")
        else:
            # 清理所有缓存
            count = 0
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
                count += 1
            print(f"[FeatureCache] 已清理所有缓存，共 {count} 个文件")
    
    def list_cache(self):
        """列出所有缓存文件"""
        cache_files = list(self.cache_dir.glob('*.pkl'))
        print(f"[FeatureCache] 缓存文件列表 (共 {len(cache_files)} 个):")
        
        for cache_file in cache_files:
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            mtime = time.ctime(cache_file.stat().st_mtime)
            print(f"  - {cache_file.name}: {size_mb:.2f} MB, 修改时间: {mtime}")


def demo():
    """演示特征缓存系统的使用"""
    import numpy as np
    
    # 创建缓存管理器
    cache = FeatureCache()
    
    # 定义一个计算密集型的特征计算函数
    def compute_expensive_feature():
        print("  [模拟] 正在进行复杂计算...")
        time.sleep(2)  # 模拟耗时计算
        
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.randn(1000),
        }, index=dates)
        
        return data
    
    print("=" * 60)
    print("特征缓存系统演示")
    print("=" * 60)
    
    # 第一次计算（无缓存）
    print("\n【第一次调用】")
    key = "expensive_feature_v1"
    result1 = cache.get_or_compute(key, compute_expensive_feature)
    print(f"结果形状: {result1.shape}")
    print(f"前5行:\n{result1.head()}")
    
    # 第二次调用（从缓存加载）
    print("\n【第二次调用】")
    result2 = cache.get_or_compute(key, compute_expensive_feature)
    print(f"结果形状: {result2.shape}")
    print(f"两次结果是否相同: {result1.equals(result2)}")
    
    # 列出缓存
    print("\n【缓存列表】")
    cache.list_cache()
    
    # 清理缓存
    print("\n【清理缓存】")
    cache.clear_cache(key)
    cache.list_cache()


if __name__ == "__main__":
    demo()
