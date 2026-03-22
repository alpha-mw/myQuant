"""K线分析后端注册表。

支持四种后端：
  - heuristic: 兼容保留的启发式后端
  - kronos:    仅运行 Kronos 子模型（诊断用途）
  - chronos:   仅运行 Chronos 子模型（诊断用途）
  - hybrid:    生产主链默认后端，固定同时运行 Kronos + Chronos 并走 evaluator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import KLineBackend

_BACKEND_REGISTRY: dict[str, type[KLineBackend]] = {}


def _ensure_registry() -> None:
    if _BACKEND_REGISTRY:
        return
    from .chronos_adapter import ChronosBackend
    from .heuristic import HeuristicBackend
    from .hybrid_adapter import HybridBackend
    from .kronos_adapter import KronosBackend

    _BACKEND_REGISTRY["heuristic"] = HeuristicBackend
    _BACKEND_REGISTRY["kronos"] = KronosBackend
    _BACKEND_REGISTRY["chronos"] = ChronosBackend
    _BACKEND_REGISTRY["hybrid"] = HybridBackend


def get_backend(name: str, **kwargs: object) -> KLineBackend:
    """根据名称获取 K线分析后端实例。"""
    _ensure_registry()
    cls = _BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"未知 K线后端: {name!r}，可选: {list(_BACKEND_REGISTRY)}")
    return cls(**kwargs)


def available_backends() -> list[str]:
    _ensure_registry()
    return list(_BACKEND_REGISTRY)
