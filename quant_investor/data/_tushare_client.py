"""Shared Tushare client pool with endpoint-local throttling."""

from __future__ import annotations

import logging
import time
from typing import Any

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.data._registry import get_endpoint_spec


class TushareClientPool:
    """Small singleton wrapper around ``tushare.pro_api``.

    Tests and local verification monkeypatch ``_get_pro`` directly; the live
    path is only used when callers explicitly request provider access.
    """

    _instance: "TushareClientPool | None" = None

    def __new__(cls) -> "TushareClientPool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._logger = logging.getLogger("data.TushareClientPool")
        self._pro: Any | None = None
        self._token = ""
        self._url = ""
        self._last_call_at: dict[str, float] = {}
        self._endpoint_circuit_open_until: dict[str, float] = {}
        self._endpoint_circuit_reason: dict[str, str] = {}
        self.available = True

    def _ensure_config(self) -> None:
        self._token = str(config.TUSHARE_TOKEN or "").strip()
        self._url = str(config.TUSHARE_URL or "").strip()

    def _get_pro(self) -> Any:
        self._ensure_config()
        if self._pro is not None:
            return self._pro
        if not self._token:
            self.available = False
            raise RuntimeError("missing TUSHARE_TOKEN")
        try:
            import tushare as ts  # type: ignore
        except Exception as exc:
            self.available = False
            raise RuntimeError(f"tushare import failed: {exc}") from exc
        self._pro = create_tushare_pro(ts, self._token, self._url)
        if self._pro is None:
            self.available = False
            raise RuntimeError("tushare client unavailable")
        self.available = True
        return self._pro

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "每分钟最多访问" in text or "quota" in text or "rate limit" in text

    @staticmethod
    def _opens_circuit(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(token in text for token in ("permission", "403", "invalid token", "无效的 token", "timed out", "timeout"))

    def _check_circuit(self, api_name: str) -> None:
        now = time.monotonic()
        until = float(self._endpoint_circuit_open_until.get(api_name, 0.0) or 0.0)
        if until > now:
            reason = self._endpoint_circuit_reason.get(api_name, "")
            raise RuntimeError(f"Tushare circuit open for {api_name} {until - now:.1f}s: {reason}")

    def _rate_limit(self, api_name: str) -> None:
        spec = get_endpoint_spec(api_name)
        limit = max(int(spec.rate_limit_per_min or 1), 1)
        interval = 60.0 / limit
        now = time.monotonic()
        previous = self._last_call_at.get(api_name)
        if previous is not None:
            wait = interval - (now - previous)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
        self._last_call_at[api_name] = now

    def query(self, api_name: str, wait_on_quota: bool = False, **kwargs: Any) -> Any:
        api = str(api_name)
        self._check_circuit(api)
        self._rate_limit(api)
        pro = self._get_pro()
        method = getattr(pro, api)
        try:
            return method(**kwargs)
        except Exception as exc:
            if self._is_quota_error(exc) and wait_on_quota:
                self._logger.warning("quota hit for %s; sleeping before one retry", api)
                time.sleep(65.0)
                self._last_call_at[api] = time.monotonic()
                return method(**kwargs)
            if self._opens_circuit(exc):
                self._endpoint_circuit_open_until[api] = time.monotonic() + 60.0
                self._endpoint_circuit_reason[api] = str(exc)
            raise
