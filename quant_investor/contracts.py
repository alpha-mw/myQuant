#!/usr/bin/env python3
"""
兼容导出层。

V9 起契约定义迁移到 `quant_investor.branch_contracts`；旧模块名继续保留，
避免 `from quant_investor.contracts import ...` 失效。
"""

from quant_investor.branch_contracts import *  # noqa: F401,F403
