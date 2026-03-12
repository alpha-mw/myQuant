"""
Quant-Investor V7.0 依赖注入容器
实现松耦合的架构设计
"""

from typing import Dict, Type, Any, Callable, Optional
from abc import ABC, abstractmethod
import inspect


class DIContainer:
    """
    依赖注入容器
    
    使用示例:
        container = DIContainer()
        
        # 注册服务
        container.register(DataLayerInterface, EnhancedDataLayer)
        container.register(FactorLayerInterface, FactorAnalyzer)
        
        # 解析服务
        data_layer = container.resolve(DataLayerInterface)
    """
    
    def __init__(self):
        self._registrations: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
    
    def register(self, interface: Type, implementation: Type, 
                 lifecycle: str = "transient") -> None:
        """
        注册服务
        
        Args:
            interface: 接口类型
            implementation: 实现类型
            lifecycle: 生命周期 (transient/singleton)
        """
        self._registrations[interface] = {
            "implementation": implementation,
            "lifecycle": lifecycle
        }
    
    def register_instance(self, interface: Type, instance: Any) -> None:
        """
        注册单例实例
        
        Args:
            interface: 接口类型
            instance: 实例对象
        """
        self._singletons[interface] = instance
    
    def register_factory(self, interface: Type, factory: Callable) -> None:
        """
        注册工厂函数
        
        Args:
            interface: 接口类型
            factory: 工厂函数
        """
        self._factories[interface] = factory
    
    def resolve(self, interface: Type) -> Any:
        """
        解析服务
        
        Args:
            interface: 接口类型
            
        Returns:
            服务实例
        """
        # 检查是否已注册为单例
        if interface in self._singletons:
            return self._singletons[interface]
        
        # 检查是否有工厂函数
        if interface in self._factories:
            instance = self._factories[interface]()
            return instance
        
        # 检查注册信息
        if interface not in self._registrations:
            raise KeyError(f"未注册的服务: {interface.__name__}")
        
        registration = self._registrations[interface]
        implementation = registration["implementation"]
        lifecycle = registration["lifecycle"]
        
        # 创建实例
        instance = self._create_instance(implementation)
        
        # 如果是单例，保存实例
        if lifecycle == "singleton":
            self._singletons[interface] = instance
        
        return instance
    
    def _create_instance(self, implementation: Type) -> Any:
        """
        创建实例，自动注入依赖
        
        Args:
            implementation: 实现类型
            
        Returns:
            实例对象
        """
        # 获取构造函数签名
        signature = inspect.signature(implementation.__init__)
        parameters = list(signature.parameters.items())[1:]  # 跳过self
        
        # 解析依赖
        dependencies = []
        for name, param in parameters:
            if param.annotation != inspect.Parameter.empty:
                # 有类型注解，尝试解析
                try:
                    dep = self.resolve(param.annotation)
                    dependencies.append(dep)
                except KeyError:
                    # 依赖未注册，使用默认值
                    if param.default != inspect.Parameter.empty:
                        dependencies.append(param.default)
                    else:
                        raise KeyError(f"无法解析依赖: {name}: {param.annotation}")
            elif param.default != inspect.Parameter.empty:
                # 使用默认值
                dependencies.append(param.default)
            else:
                raise KeyError(f"无法解析依赖: {name}")
        
        # 创建实例
        return implementation(*dependencies)
    
    def build_provider(self) -> "ServiceProvider":
        """
        构建服务提供者
        
        Returns:
            服务提供者
        """
        return ServiceProvider(self)


class ServiceProvider:
    """
    服务提供者
    简化服务访问
    """
    
    def __init__(self, container: DIContainer):
        self._container = container
    
    def get_service(self, interface: Type) -> Any:
        """获取服务"""
        return self._container.resolve(interface)
    
    def get_required_service(self, interface: Type) -> Any:
        """获取必需的服务，如果不存在则抛出异常"""
        return self._container.resolve(interface)


# 全局容器实例
_default_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """获取默认容器"""
    global _default_container
    if _default_container is None:
        _default_container = DIContainer()
    return _default_container


def configure_services() -> DIContainer:
    """
    配置服务
    注册所有层的服务
    """
    container = get_container()
    
    # 注册配置
    from config import Config
    container.register_instance(object, Config)
    
    return container


# 便捷函数
def register_service(interface: Type, implementation: Type, 
                     lifecycle: str = "transient") -> None:
    """注册服务到默认容器"""
    get_container().register(interface, implementation, lifecycle)


def resolve_service(interface: Type) -> Any:
    """从默认容器解析服务"""
    return get_container().resolve(interface)
