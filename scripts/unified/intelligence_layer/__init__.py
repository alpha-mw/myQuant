#!/usr/bin/env python3
"""
智能监控层 - 简化版
整合 V3.1 动态监控功能
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MonitorState:
    """监控状态"""
    last_check: str = field(default_factory=lambda: datetime.now().isoformat())
    alerts: list = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class IntelligenceMonitor:
    """智能监控器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.state = MonitorState()
        
    def check_factor_decay(self, factor_performance: Dict[str, float]) -> list:
        """检查因子衰减"""
        alerts = []
        for factor, ic in factor_performance.items():
            if ic < 0.02:
                alerts.append(f"因子 {factor} IC 衰减，当前值: {ic:.4f}")
        return alerts
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'last_check': self.state.last_check,
            'active_alerts': len(self.state.alerts),
            'metrics': self.state.metrics
        }
