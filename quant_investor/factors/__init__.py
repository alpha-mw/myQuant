"""Stable Factor package.

The official governance surface lives under :mod:`quant_investor.factors.governance`.
Historical registry, WAL, and protocol-numbered APIs are intentionally absent
from this hard-cutover package.
"""

from .governance import *  # noqa: F401,F403
from .governance import __all__ as _governance_all


__all__ = list(_governance_all)
