"""Synthetic acceptance freezes business dates without freezing process deadlines."""

import ast
import builtins
from datetime import datetime, timezone
import importlib
from pathlib import Path
import time

from _native_synthetic_clock import MODULES, synthetic_clock


def test_all_daily_datetime_callers_are_declared():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "quant_investor/operations").glob("*.py"))
    paths.extend((root / "scripts").glob("daily_*.py"))
    for path in paths:
        tree = ast.parse(path.read_text())
        calls_clock = any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "datetime"
            and node.attr in {"now", "utcnow", "today"}
            for node in ast.walk(tree)
        )
        if calls_clock:
            assert ".".join(path.relative_to(root).with_suffix("").parts) in MODULES


def test_clock_covers_global_and_local_imports_and_restores_real_bindings():
    import _native_daily_store_fixture  # noqa: F401 - establish existing script import context

    now = datetime(2026, 8, 27, 13, 20, tzinfo=timezone.utc)
    real_monotonic, real_time, real_import = time.monotonic, time.time, builtins.__import__
    owners = [importlib.import_module(n) for n in MODULES]
    originals = {m.__name__: getattr(m, "datetime", None) for m in owners}
    with synthetic_clock(now):
        for module in owners:
            if hasattr(module, "datetime"):
                assert module.datetime.now(timezone.utc) == now
            local = builtins.__import__(
                "datetime", {"__name__": module.__name__}, None, ("datetime",)
            )
            assert local.datetime.now(timezone.utc) == now
        assert time.monotonic is real_monotonic and time.time is real_time
        assert importlib.import_module("datetime").datetime is datetime
    assert builtins.__import__ is real_import
    for module in owners:
        assert getattr(module, "datetime", None) is originals[module.__name__]
