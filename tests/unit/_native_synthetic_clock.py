"""One explicit logical clock for the installed synthetic daily acceptance chain.

Real monotonic time, filesystem timestamps, locks and release installation
verification remain real. Only named business modules' datetime binding changes.
The list includes downstream seals, not just the upstream request producers.
"""

from contextlib import ExitStack, contextmanager
import builtins
import datetime as datetime_module
from datetime import datetime
import importlib
from types import ModuleType
from unittest.mock import patch

NATIVE = (
    "quant_investor.market._calendar_fixture_capability",
    "quant_investor.market.next_session_proof",
    "quant_investor.market.next_session_failure",
    "quant_investor.cli.unified",
    "quant_investor.factors.production_authority",
    "quant_investor.factors.production_observation",
    "quant_investor.market.daily_maintenance",
    "scripts.cn_official_close_batch",
    "scripts.manage_cn_strategy_records",
    "cn_official_close_batch",
)
OPERATIONS = (
    "automatic_catchup_closure",
    "automatic_catchup_resolution",
    "catchup_binding",
    "completion_readback",
    "corporate_adapter",
    "daily_journal",
    "daily_preparation",
    "dashboard_evidence",
    "dashboard_publication_guard",
    "dashboard_serving_contract",
    "decision_publication",
    "maintenance_handoff",
    "morning_cutover_contract",
    "morning_receipt",
    "portfolio_binding",
    "registered_dashboard",
    "research_capture",
    "research_cutoff",
    "research_sources",
    "research_timing",
    "source_slot_inputs",
    "theme_acquisition",
    "theme_capture_stage",
    "theme_handoff",
    "theme_handoff_publish",
)
SCRIPTS = (
    "daily_automatic_catchup",
    "daily_bootstrap_launch",
    "daily_completion",
    "daily_dashboard_adapter",
    "daily_dashboard_capture",
    "daily_dashboard_publication",
    "daily_dashboard_sealed",
    "daily_ledger",
    "daily_materialization",
    "daily_morning_consumer",
    "daily_morning_cutover",
    "daily_morning_history",
    "daily_morning_seal",
    "daily_registered_recovery",
    "daily_store_adoption",
)
MODULES = (
    NATIVE
    + tuple("quant_investor.operations." + n for n in OPERATIONS)
    + tuple("scripts." + n for n in SCRIPTS)
)


@contextmanager
def synthetic_clock(now):
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("an aware synthetic logical clock is required")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            from datetime import timezone

            return now.astimezone(timezone.utc).replace(tzinfo=None)

        @classmethod
        def today(cls):
            return cls.now()

    modules = [importlib.import_module(name) for name in MODULES]
    local_datetime = ModuleType("datetime")
    local_datetime.__dict__.update(vars(datetime_module))
    local_datetime.datetime = Clock
    original_import = builtins.__import__

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "datetime" and globals and globals.get("__name__") in MODULES:
            return local_datetime
        return original_import(name, globals, locals, fromlist, level)

    with ExitStack() as stack:
        seen = set()
        for module in modules:
            if id(module) in seen:
                continue
            seen.add(id(module))
            if module.__name__ == "quant_investor.operations.dashboard_publication_guard":
                continue  # This owner imports datetime inside its method.
            if module.datetime is not datetime:
                raise ValueError("unexpected datetime binding: " + module.__name__)
            stack.enter_context(patch.object(module, "datetime", Clock))
        stack.enter_context(patch.object(builtins, "__import__", scoped_import))
        yield
