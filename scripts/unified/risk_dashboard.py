"""
Quant-Investor V7.0 实时风险监控仪表盘
Streamlit-based risk dashboard
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import pandas as pd
import numpy as np


@dataclass
class RiskAlert:
    """风险预警"""
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: str
    value: float
    threshold: float


class RiskMonitor:
    """
    风险监控器
    
    实时监控风险指标并生成预警
    """
    
    def __init__(self, alert_thresholds: Optional[Dict] = None):
        """
        初始化风险监控器
        
        Args:
            alert_thresholds: 预警阈值配置
        """
        self.thresholds = alert_thresholds or {
            'var_95': 0.02,      # 2% VaR预警
            'max_drawdown': 0.15, # 15% 最大回撤预警
            'volatility': 0.40,   # 40% 波动率预警
            'concentration': 0.30, # 30% 集中度预警
        }
        self.alerts: List[RiskAlert] = []
        self.history: List[Dict] = []
    
    def check_risk_metrics(self, metrics: Dict[str, float]) -> List[RiskAlert]:
        """
        检查风险指标
        
        Args:
            metrics: 风险指标字典
            
        Returns:
            预警列表
        """
        new_alerts = []
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                if value > threshold:
                    # 确定严重级别
                    if value > threshold * 1.5:
                        severity = 'critical'
                    elif value > threshold * 1.2:
                        severity = 'high'
                    else:
                        severity = 'medium'
                    
                    alert = RiskAlert(
                        alert_type=f'{metric_name}_exceeded',
                        severity=severity,
                        message=f'{metric_name} 超过阈值: {value:.2%} > {threshold:.2%}',
                        timestamp=datetime.now().isoformat(),
                        value=value,
                        threshold=threshold
                    )
                    new_alerts.append(alert)
                    self.alerts.append(alert)
        
        # 保存历史
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'alert_count': len(new_alerts)
        })
        
        return new_alerts
    
    def get_recent_alerts(self, minutes: int = 60) -> List[RiskAlert]:
        """
        获取最近预警
        
        Args:
            minutes: 最近多少分钟
            
        Returns:
            预警列表
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert.timestamp) > cutoff
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        recent_alerts = self.get_recent_alerts(minutes=60)
        
        return {
            'total_alerts': len(self.alerts),
            'recent_alerts_1h': len(recent_alerts),
            'critical_alerts': sum(1 for a in recent_alerts if a.severity == 'critical'),
            'high_alerts': sum(1 for a in recent_alerts if a.severity == 'high'),
            'last_updated': datetime.now().isoformat(),
        }


def create_dashboard():
    """
    创建Streamlit仪表盘
    
    运行: streamlit run scripts/unified/risk_dashboard.py
    """
    if not STREAMLIT_AVAILABLE:
        print("Streamlit未安装，运行: pip install streamlit")
        return
    
    st.set_page_config(
        page_title="Quant-Investor V7.0 风险监控",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Quant-Investor V7.0 实时风险监控")
    
    # 侧边栏
    st.sidebar.header("配置")
    refresh_interval = st.sidebar.slider("刷新间隔(秒)", 5, 60, 10)
    
    # 模拟数据（实际应从数据库或API获取）
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="VaR (95%)",
            value="2.3%",
            delta="-0.1%"
        )
    
    with col2:
        st.metric(
            label="最大回撤",
            value="12.5%",
            delta="+0.5%"
        )
    
    with col3:
        st.metric(
            label="夏普比率",
            value="1.45",
            delta="+0.05"
        )
    
    with col4:
        st.metric(
            label="波动率",
            value="28.3%",
            delta="-1.2%"
        )
    
    # 风险趋势图
    st.subheader("风险指标趋势")
    
    # 模拟历史数据
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    var_data = pd.DataFrame({
        '日期': dates,
        'VaR (95%)': np.random.uniform(0.015, 0.03, 30),
        '阈值': [0.02] * 30
    })
    
    st.line_chart(var_data.set_index('日期'))
    
    # 预警列表
    st.subheader("🚨 最新预警")
    
    alerts_data = [
        {
            '时间': '2024-03-06 10:30:00',
            '类型': 'VaR超标',
            '严重级别': 'high',
            '消息': 'VaR (95%) 超过阈值: 2.3% > 2.0%'
        },
        {
            '时间': '2024-03-06 09:15:00',
            '类型': '集中度预警',
            '严重级别': 'medium',
            '消息': '单票集中度超过阈值: 25% > 20%'
        }
    ]
    
    alerts_df = pd.DataFrame(alerts_data)
    st.dataframe(alerts_df, use_container_width=True)
    
    # 组合持仓
    st.subheader("📈 组合持仓")
    
    positions = pd.DataFrame({
        '股票代码': ['000001.SZ', '000002.SZ', '600000.SH'],
        '名称': ['平安银行', '万科A', '浦发银行'],
        '权重': [0.15, 0.12, 0.10],
        '收益': [0.05, -0.02, 0.03],
        '风险贡献': [0.08, 0.06, 0.05]
    })
    
    st.dataframe(positions, use_container_width=True)
    
    # 自动刷新
    st.empty()
    st.write(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        create_dashboard()
    else:
        print("请安装Streamlit: pip install streamlit")
        print("然后运行: streamlit run risk_dashboard.py")
