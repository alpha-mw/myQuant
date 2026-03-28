"""
为仓库内仅保留 `.pyc` 的模块提供导入兜底。

当前 `quant_investor/` 目录里仍有一批运行期实现只存在于 `__pycache__`。
本模块注册一个限定作用域的 finder，只在同包内找不到 `.py` 源文件时，
按 `__pycache__/<name>.<cache_tag>.pyc` 回退加载。
"""

from __future__ import annotations

from functools import lru_cache
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourcelessFileLoader
from importlib.util import module_from_spec
from pathlib import Path
import sys
from types import ModuleType


_CACHE_TAG = sys.implementation.cache_tag or "cpython-313"
_PACKAGE_ROOT = Path(__file__).resolve().parent


class _QuantInvestorSourcelessFinder(MetaPathFinder):
    """仅为 quant_investor 包提供 sourceless 导入兜底。"""

    def find_spec(self, fullname: str, path=None, target=None) -> ModuleSpec | None:
        if fullname == "quant_investor":
            return None
        if not fullname.startswith("quant_investor."):
            return None

        relative_parts = fullname.split(".")[1:]
        if not relative_parts:
            return None

        package_dir = _PACKAGE_ROOT.joinpath(*relative_parts)
        source_package = package_dir / "__init__.py"
        if package_dir.is_dir() and source_package.exists():
            return None

        module_source = _PACKAGE_ROOT.joinpath(*relative_parts[:-1], f"{relative_parts[-1]}.py")
        if module_source.exists():
            return None

        if package_dir.is_dir():
            package_pyc = package_dir / "__pycache__" / f"__init__.{_CACHE_TAG}.pyc"
            if package_pyc.exists():
                loader = SourcelessFileLoader(fullname, str(package_pyc))
                spec = ModuleSpec(fullname, loader, origin=str(package_pyc), is_package=True)
                spec.submodule_search_locations = [str(package_dir)]
                spec.cached = str(package_pyc)
                spec.has_location = True
                return spec

        module_pyc = _PACKAGE_ROOT.joinpath(
            *relative_parts[:-1],
            "__pycache__",
            f"{relative_parts[-1]}.{_CACHE_TAG}.pyc",
        )
        if module_pyc.exists():
            loader = SourcelessFileLoader(fullname, str(module_pyc))
            spec = ModuleSpec(fullname, loader, origin=str(module_pyc), is_package=False)
            spec.cached = str(module_pyc)
            spec.has_location = True
            return spec
        return None


def install_sourceless_finder() -> None:
    """注册一次限定作用域的 sourceless finder。"""

    for finder in sys.meta_path:
        if isinstance(finder, _QuantInvestorSourcelessFinder):
            return
    sys.meta_path.insert(0, _QuantInvestorSourcelessFinder())


@lru_cache(maxsize=None)
def load_shadowed_module(fullname: str, pyc_path: str | Path | None = None) -> ModuleType:
    """以私有别名直接加载仓库里的 sourceless 原始实现。"""

    if fullname == "quant_investor" or not fullname.startswith("quant_investor."):
        raise ValueError(f"unsupported module name: {fullname}")

    relative_parts = fullname.split(".")[1:]
    package_dir = _PACKAGE_ROOT.joinpath(*relative_parts)

    is_package = package_dir.is_dir()
    if pyc_path is None:
        if is_package:
            pyc_path = package_dir / "__pycache__" / f"__init__.{_CACHE_TAG}.pyc"
        else:
            pyc_path = _PACKAGE_ROOT.joinpath(
                *relative_parts[:-1],
                "__pycache__",
                f"{relative_parts[-1]}.{_CACHE_TAG}.pyc",
            )
    else:
        pyc_path = Path(pyc_path)

    if not pyc_path.exists():
        raise FileNotFoundError(f"sourceless module not found for {fullname}: {pyc_path}")

    alias = f"_quant_investor_shadow_{'_'.join(relative_parts)}"
    cached = sys.modules.get(alias)
    if cached is not None:
        return cached

    loader = SourcelessFileLoader(alias, str(pyc_path))
    spec = ModuleSpec(alias, loader, origin=str(pyc_path), is_package=is_package)
    if is_package:
        spec.submodule_search_locations = [str(package_dir)]
    spec.cached = str(pyc_path)
    spec.has_location = True

    module = module_from_spec(spec)
    sys.modules[alias] = module
    loader.exec_module(module)
    return module
