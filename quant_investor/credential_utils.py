#!/usr/bin/env python3
"""
Credential helpers.

统一处理敏感凭据的读取、脱敏和客户端初始化，避免写盘持久化。
"""

from __future__ import annotations

import os
from typing import Any


def get_secret(name: str, default: str = "") -> str:
    """从环境变量读取凭据并去除首尾空白。"""
    return os.environ.get(name, default).strip()


def has_secret(name: str) -> bool:
    """判断某个凭据是否已设置。"""
    return bool(get_secret(name))


def mask_secret(value: str, keep_prefix: int = 4, keep_suffix: int = 2) -> str:
    """返回脱敏后的凭据文本。"""
    if not value:
        return ""
    if len(value) <= keep_prefix + keep_suffix:
        return "*" * len(value)
    return f"{value[:keep_prefix]}{'*' * (len(value) - keep_prefix - keep_suffix)}{value[-keep_suffix:]}"


def create_tushare_pro(ts_module: Any, token: str, http_url: str | None = None) -> Any | None:
    """
    在内存中创建 Tushare Pro 客户端。

    不调用 `ts.set_token(...)`，避免把 token 写入用户目录。
    """
    token = token.strip()
    if not token:
        return None

    pro = ts_module.pro_api(token)
    if hasattr(pro, "_DataApi__token"):
        pro._DataApi__token = token
    if http_url and hasattr(pro, "_DataApi__http_url"):
        pro._DataApi__http_url = http_url
    return pro
