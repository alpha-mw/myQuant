#!/usr/bin/env python3
"""Compatibility shim for the packaged daily runner implementation."""

from __future__ import annotations

import sys as _sys

from quant_investor.automation import daily_runner as _impl

if __name__ == "__main__":
    _impl._bootstrap_project_venv()
    _impl.main()
else:
    _sys.modules[__name__] = _impl
