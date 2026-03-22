"""
将 vendored Chronos 包注册为顶层 `chronos` 模块。

Chronos 官方源码内部仍使用 `import chronos` / `from chronos...` 形式，
这里通过 importlib 在运行时把 `quant_investor/_vendor/chronos` 暴露为
顶层包名，避免依赖仓库外代码或额外安装的 chronos 包。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_vendored_chronos() -> ModuleType:
    """优先加载 myQuant 内 vendored 的 Chronos 包。"""
    package_name = "chronos"
    package_dir = Path(__file__).resolve().parent / "chronos"
    init_file = package_dir / "__init__.py"
    package_root = package_dir.resolve()

    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", "")
        if existing_file:
            try:
                if Path(existing_file).resolve().is_relative_to(package_root):
                    return existing
            except Exception:
                if str(package_root) in str(existing_file):
                    return existing

        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {init_file} 加载 vendored Chronos 包")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module

