"""
Quant-Investor V7.0 事件驱动架构
RabbitMQ消息队列集成
"""

import json
import asyncio
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import pika
    from pika.adapters.asyncio_connection import AsyncioConnection
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    print("RabbitMQ客户端未安装")


@dataclass
class QuantEvent:
    """量化事件"""
    event_type: str
    payload: Dict[str, Any]
    timestamp: str = None
    correlation_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class EventBus:
    """
    事件总线
    
    使用示例:
        bus = EventBus()
        
        # 订阅事件
        @bus.subscribe('factor.calculated')
        def on_factor_calculated(event):
            print(f"因子计算完成: {event.payload}")
        
        # 发布事件
        bus.publish('factor.calculated', {'factor': 'momentum', 'value': 0.5})
    """
    
    def __init__(self, host: str = 'localhost', port: int = 5672):
        """
        初始化事件总线
        
        Args:
            host: RabbitMQ主机
            port: RabbitMQ端口
        """
        self.host = host
        self.port = port
        self.connection = None
        self.channel = None
        self.subscribers: Dict[str, list] = {}
        
        if RABBITMQ_AVAILABLE:
            self._connect()
    
    def _connect(self):
        """连接RabbitMQ"""
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host, port=self.port)
            )
            self.channel = self.connection.channel()
            print(f"RabbitMQ连接成功: {self.host}:{self.port}")
        except Exception as e:
            print(f"RabbitMQ连接失败: {e}")
            self.connection = None
    
    def publish(self, event_type: str, payload: Dict[str, Any]):
        """
        发布事件
        
        Args:
            event_type: 事件类型
            payload: 事件数据
        """
        event = QuantEvent(event_type=event_type, payload=payload)
        
        # 本地订阅者
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"事件处理失败: {e}")
        
        # RabbitMQ发布
        if self.connection and RABBITMQ_AVAILABLE:
            try:
                self.channel.exchange_declare(exchange='quant_events', type='topic')
                self.channel.basic_publish(
                    exchange='quant_events',
                    routing_key=event_type,
                    body=json.dumps(asdict(event))
                )
            except Exception as e:
                print(f"RabbitMQ发布失败: {e}")
    
    def subscribe(self, event_type: str):
        """
        订阅事件装饰器
        
        Args:
            event_type: 事件类型
        """
        def decorator(func: Callable):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(func)
            return func
        return decorator
    
    def consume(self, event_type: str, callback: Callable):
        """
        从RabbitMQ消费事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if not self.connection or not RABBITMQ_AVAILABLE:
            print("RabbitMQ未连接")
            return
        
        try:
            self.channel.exchange_declare(exchange='quant_events', type='topic')
            result = self.channel.queue_declare(queue='', exclusive=True)
            queue_name = result.method.queue
            
            self.channel.queue_bind(
                exchange='quant_events',
                queue=queue_name,
                routing_key=event_type
            )
            
            def on_message(ch, method, properties, body):
                event_data = json.loads(body)
                event = QuantEvent(**event_data)
                callback(event)
            
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=on_message,
                auto_ack=True
            )
            
            print(f"开始消费事件: {event_type}")
            self.channel.start_consuming()
        except Exception as e:
            print(f"消费事件失败: {e}")
    
    def close(self):
        """关闭连接"""
        if self.connection:
            self.connection.close()


class AsyncEventBus:
    """
    异步事件总线
    """
    
    def __init__(self):
        self.subscribers: Dict[str, list] = {}
    
    async def publish(self, event_type: str, payload: Dict[str, Any]):
        """
        异步发布事件
        
        Args:
            event_type: 事件类型
            payload: 事件数据
        """
        event = QuantEvent(event_type=event_type, payload=payload)
        
        if event_type in self.subscribers:
            tasks = []
            for callback in self.subscribers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(event))
                else:
                    callback(event)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def subscribe(self, event_type: str):
        """订阅事件"""
        def decorator(func: Callable):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(func)
            return func
        return decorator


# 预定义事件类型
EVENT_TYPES = {
    # 数据层事件
    'data.fetched': '数据获取完成',
    'data.processed': '数据处理完成',
    
    # 因子层事件
    'factor.calculated': '因子计算完成',
    'factor.selected': '因子筛选完成',
    
    # 模型层事件
    'model.trained': '模型训练完成',
    'model.predicted': '模型预测完成',
    
    # 风控层事件
    'risk.calculated': '风险计算完成',
    'risk.alert': '风险预警',
    
    # 决策层事件
    'decision.made': '投资决策完成',
    'order.placed': '订单提交',
    'order.filled': '订单成交',
    
    # 系统事件
    'system.started': '系统启动',
    'system.error': '系统错误',
}
