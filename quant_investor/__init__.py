"""Quant-Investor stable public Python API.

Mainline types are loaded lazily so lower-layer packages, including
``quant_investor.intelligence``, remain independently importable.  Attribute
access preserves the root-level public API without creating an eager
Intelligence-to-Mainline dependency during package initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quant_investor.pipeline import QuantInvestor

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from quant_investor.mainline import MainlineError, MainlineStore

_LAZY_MAINLINE_EXPORTS = frozenset({"MainlineError", "MainlineStore"})


def __getattr__(name: str) -> Any:
    if name not in _LAZY_MAINLINE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from quant_investor.mainline import MainlineError, MainlineStore

    exported = {
        "MainlineError": MainlineError,
        "MainlineStore": MainlineStore,
    }[name]
    globals()[name] = exported
    return exported


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["MainlineError", "MainlineStore", "QuantInvestor"]
