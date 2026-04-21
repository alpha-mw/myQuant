"""Shared run-history access for non-web entrypoints."""

from __future__ import annotations

from importlib import import_module


_store_module = import_module("web.services.run_history_store")

RunHistoryStore = _store_module.RunHistoryStore
history_store = _store_module.history_store

__all__ = ["RunHistoryStore", "history_store"]
