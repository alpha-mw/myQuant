"""K线分析后端注册表。

支持三种后端：
  - heuristic: 启发式动量+波动率适配（默认，无需额外依赖）
  - kronos:    Kronos 预训练 K线基础模型
  - chronos:   Amazon Chronos-2 时序基础模型
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
    from .kronos_adapter import KronosBackend

    _BACKEND_REGISTRY["heuristic"] = HeuristicBackend
    _BACKEND_REGISTRY["kronos"] = KronosBackend
    _BACKEND_REGISTRY["chronos"] = ChronosBackend


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
