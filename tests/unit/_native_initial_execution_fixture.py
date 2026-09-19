"""Installed native initial EXECUTE through early handoff, with explicit auxiliary stop."""

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


@contextmanager
def synthetic_factor_clock(enabled: bool):
    """Match Factor custody to the explicit core journal clock in this fixture only."""
    if not enabled:
        yield
        return

    class FactorClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 27, 13, tzinfo=timezone.utc)

    with (
        patch("quant_investor.cli.unified.datetime", FactorClock),
        patch("quant_investor.factors.production_observation.datetime", FactorClock),
    ):
        yield


def run_initial_boundary(
    root: Path, fixture, activated: dict, *, theme_probe=False, source_complete=False
) -> dict:
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.market.daily_factor_loop import DailyFactorLoop, read_factor_loop_context
    from quant_investor.intelligence.storage import publish_theme_policy_v2
    from quant_investor.operations.daily_contract import GRAPH_SHA256
    from quant_investor.operations.daily_journal import FALSE_AUTHORITY
    from quant_investor.operations.maintenance_handoff import read_maintenance_handoff
    from scripts import daily_materialization as coordinator
    from _native_daily_calendar_fixture import synthetic_calendar_transport
    from _native_daily_maintenance_fixture import maintenance
    from _native_daily_store_fixture import NativeStoreFixture, DAYS

    workspace = root / "factor-workspace"
    day = "20260827"
    day_root = workspace / "results/operations/daily_production/CN" / day
    assert not day_root.exists()

    def ref(path):
        return {
            "path": str(path.relative_to(workspace)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def put(name, value):
        path = workspace / "initial-execute" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent
        while parent != workspace:
            parent.chmod(0o700)
            parent = parent.parent
        path.write_bytes(canonical_json_bytes(value))
        path.chmod(0o600)
        return ref(path)

    book = NativeStoreFixture(workspace, stock_symbols=fixture.symbols[:7], preserve_market=True)
    for date in DAYS[:4]:
        store_args = book.advance(date)
    installed = put("install.json", json.loads((root / "release-input.json").read_bytes()))
    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    context = put(
        "context.json",
        {
            "schema_version": "cn-daily-factor-loop.v1",
            "release_install_input_ref": installed,
            "release_repository_root": str(root / "repository"),
            "release_commit": receipt["commit"],
            "calendar_capture_parent": str(workspace / "data/private/initial-calendar-captures"),
        },
    )
    _, installation = read_factor_loop_context(
        workspace_root=str(workspace),
        context_path=str(workspace / context["path"]),
        context_sha256=context["sha256"],
    )
    release = ref(
        workspace
        / "results/factors/objects/system.release"
        / (installation["release_ref"]["byte_sha256"] + ".json")
    )
    publish_theme_policy_v2(workspace)
    preimages = {
        "store_pointer_ref": ref(book.root / "_record_store/current.v1.json"),
        "event_pointer_ref": ref(book.root / "_event_store/current.v1.json"),
        "benchmark_pointer_ref": ref(workspace / "data/parquet/cn/benchmarks/_latest.json"),
    }
    bootstrap = put(
        "bootstrap.json",
        {
            "schema_version": "cn-daily-bootstrap.v1",
            "market": "CN",
            "strategy_id": "aggressive_tech_manufacturing",
            "first_trade_date": day,
            "previous_trade_date": "20260826",
            "graph_sha256": GRAPH_SHA256,
            "factor_parent_pointer_sha256": activated["factor_pointer_byte_sha256"],
            "store_preimages": preimages,
            "authority": FALSE_AUTHORITY,
        },
    )
    initial_sources = {}
    if source_complete:
        assert theme_probe
        from _native_initial_sources_fixture import prepare_initial_sources

        initial_sources = prepare_initial_sources(root, fixture.symbols)
    recipe = put(
        "recipe.json",
        {
            "schema_version": (
                "cn-daily-execute-recipe.v2" if theme_probe else "cn-daily-execute-recipe.v1"
            ),
            **(
                {
                    "theme_acquisition_ref": put(
                        "theme-policy.json",
                        {
                            "schema_version": "cn-daily-theme-acquisition.v1",
                            "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
                            "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
                            "maximum_companies": 100,
                        },
                    )
                }
                if theme_probe
                else {}
            ),
            "market": "CN",
            "strategy_id": "aggressive_tech_manufacturing",
            "target_trade_date": day,
            "graph_sha256": GRAPH_SHA256,
            "release_ref": release,
            "release_install_ref": installed,
            "previous_completion_ref": None,
            "bootstrap_ref": bootstrap,
            "factor_loop_context_ref": context,
            "policy_refs": {
                "research": ref(
                    workspace / "results/policies/research/aggressive_tech_manufacturing/v2.json"
                ),
                "store": {"path": store_args["policy_path"], "sha256": store_args["policy_sha"]},
                "prospective": None,
            },
            "store_preimages": preimages,
            "retrospective_ref": None,
            "dashboard_sources": {
                "benchmark_ref": ref(
                    workspace / "portfolio_dashboard/inputs/cn_index_benchmark.csv"
                ),
                "risk_free_ref": ref(
                    workspace / "portfolio_dashboard/inputs/cn_govt_bond_yield.csv"
                ),
            },
            "research_sources": {
                "as_of": "2026-08-27T13:30:00Z",
                "industry_source_ref": None,
                "theme_source_ref": None,
                "exposure_rows_ref": None,
                "fundamental": {"mode": "MAINTENANCE_STAGE", "source_ref": None},
                "macro": {"mode": "MAINTENANCE_STAGE", "source_ref": None},
                **initial_sources,
            },
            "publish_current_dashboard": False,
        },
    )
    request = put(
        "request.json",
        {
            "schema_version": "cn-daily-production-request.v1",
            "market": "CN",
            "strategy_id": "aggressive_tech_manufacturing",
            "action": "EXECUTE",
            "target_trade_date": day,
            "graph_sha256": GRAPH_SHA256,
            "release_install_ref": installed,
            "recipe_ref": recipe,
            "maintenance_handoff_ref": None,
            "calendar_ref": None,
            "raw_calendar_ref": None,
            "previous_completion_ref": None,
            "day_input_refs": {},
        },
    )
    assert not day_root.exists()
    anchor = day_root / "executions" / request["sha256"] / "maintenance-handoff.v1.json"
    events = []

    class AuxiliaryStop(BaseException):
        pass

    def stop_auxiliary(ctx):
        assert anchor.is_file(), "auxiliary reached before native handoff"
        events.append("auxiliary_after_handoff")
        raise AuxiliaryStop()

    original_settle = DailyFactorLoop._settle

    def settle(loop, ref):
        assert anchor.is_file(), "settlement reached before native handoff"
        events.append("settlement_after_handoff")
        return original_settle(loop, ref)

    def maintained(**kwargs):
        events.append("maintenance_once")
        assert kwargs["run_root"] == str(workspace / "data/private/cn_daily_maintenance")
        return maintenance(
            root,
            "2026-08-27",
            core_completed=kwargs["core_completed"],
            auxiliary_callback=stop_auxiliary,
            expected_target_trade_date=kwargs.get("_expected_target_trade_date"),
            core_replay_completed=kwargs.get("_core_replay_completed"),
        )

    class SchedulerClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 27, 13, tzinfo=timezone.utc)

    print("START initial native EXECUTE before produced core/claim", flush=True)
    with (
        synthetic_factor_clock(theme_probe),
        synthetic_calendar_transport(
            fixture_source=root / "repository/tests/unit/test_tushare_calendar_authority.py",
            cutoff="2026-08-27",
        ),
        patch.object(coordinator, "datetime", SchedulerClock),
        (
            patch("quant_investor.operations.daily_journal.datetime", SchedulerClock)
            if theme_probe
            else patch.object(coordinator, "datetime", SchedulerClock)
        ),
        patch("quant_investor.market.daily_maintenance.run_cn_daily_maintenance", maintained),
        patch.object(DailyFactorLoop, "_settle", settle),
    ):
        try:
            coordinator.execute_daily_recipe(
                workspace=str(workspace), request_ref=request, synthetic=True
            )
        except AuxiliaryStop:
            pass
        else:
            raise AssertionError("expected explicit auxiliary interruption")
    assert events == ["maintenance_once", "settlement_after_handoff", "auxiliary_after_handoff"]
    checked = read_maintenance_handoff(workspace=str(workspace), handoff_ref=ref(anchor))
    assert checked["handoff"]["request_ref"]["sha256"] == request["sha256"]
    assert not (day_root / "completion.v1.json").exists()
    result = {
        "synthetic": True,
        "initial_execute_ordering_proven": True,
        "full_daily_closure": False,
        "auxiliary_interrupted": True,
        "handoff_ref": ref(anchor),
        "request_ref": request,
        "events": events,
    }
    (root / "initial-execute-boundary-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    if theme_probe:
        from _native_initial_theme_fixture import probe_initial_theme

        result["theme_probe"] = probe_initial_theme(
            root, request, ref(anchor), source_complete=source_complete
        )
    return result
