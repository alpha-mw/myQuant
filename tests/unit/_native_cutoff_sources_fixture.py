"""Native five-domain source/book fixture; core/maintenance receipts and focus PIT are controlled.

All prices, issuer facts and provider times are generated fixture data. No
installed release, provider response, whole EOD or unattended operation is claimed.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from _native_daily_store_fixture import NativeStoreFixture, DAYS
from _native_shared_macro_fixture import build_shared_macro
from _native_daily_research_inputs import augment
from _native_corporate_fixture import tracking_policy
from test_daily_evidence_research_sources import context, put as _put
from test_daily_evidence_focus_sources import pit_context
from test_daily_evidence_production_request import recipe as base_recipe, request as base_request
from quant_investor.operations.daily_contract import GRAPH_SHA256
from quant_investor.operations.daily_journal import FALSE_AUTHORITY, DailyJournal
from quant_investor.operations.core_pool import CORE_NODES
from quant_investor.operations.maintenance_handoff_contract import (
    HANDOFF_V2_FIELDS,
    HANDOFF_V3_FIELDS,
)
from quant_investor.operations.research_timing import (
    CURRENT,
    HISTORICAL,
    acquisition_deadline,
    approved_research_timing_policy,
)
from quant_investor.operations.dashboard_serving_contract import POLICY
from quant_investor.intelligence.theme_sources import THEME_SOURCE_V2
from scripts import cn_official_close_batch as native_store
from scripts.daily_production_store_adapter import prepare_store_plan


def put(root, name, value):
    ref = _put(root, name, value)
    parent = (root / name).parent
    while parent != root:
        parent.chmod(0o700)
        parent = parent.parent
    return ref


def build(
    root,
    monkeypatch,
    mode=CURRENT,
    *,
    registered_buy=False,
    new_position=False,
    exposure_catalog=False,
):
    workspace = root / "factor-workspace"
    workspace.mkdir(mode=0o700)
    put(
        workspace, "SYNTHETIC-DATA-CLOCK.json", {"synthetic": True, "real_time_oos_eligible": False}
    )
    ctx, journal = context(workspace)
    day = journal.trade_date
    assert day == "20260828"
    inputs = NativeFactorInputs(root / "market-inputs", count=100)
    prior_market = None
    for offset in range(5):
        args = inputs.day(offset)
        value = args["as_of"]
        iso = f"{value[:4]}-{value[4:6]}-{value[6:]}"
        reader, _ = strict_market_from_factor_inputs(
            workspace,
            args,
            macro_ready_layout=True,
            pit_observed_at=iso + "T00:00:00Z",
            simulated_available_at=iso + "T07:30:00Z",
        )
        if registered_buy and value == "20260827":
            prior_market = (workspace / "data/parquet/cn/_latest.json").read_bytes()
    print("NATIVE_SOURCE_FIXTURE Market/PIT ready", flush=True)
    book = NativeStoreFixture(workspace, stock_symbols=inputs.symbols[:7], preserve_market=True)
    registered, previous_completion = None, None
    if registered_buy:
        store_args, registered, previous_completion = _registered_book(
            workspace,
            book,
            prior_market,
            inputs.symbols[7] if new_position else inputs.symbols[0],
            ctx.release_ref,
            monkeypatch,
        )
    else:
        for date in DAYS:
            store_args = book.advance(date)
    macro = build_shared_macro(workspace, "2026-08-28")
    print("NATIVE_SOURCE_FIXTURE Macro closure ready", flush=True)
    (root / f"research-inputs-{day}.json").write_text(
        json.dumps({"request_ref": ctx.request_ref, "companies": ctx.companies})
    )
    expanded = augment(root, day, native_fundamental=True)
    request_fields = json.loads((workspace / expanded["request_ref"]["path"]).read_bytes())
    print("NATIVE_SOURCE_FIXTURE Industry/Fundamental ready", flush=True)

    def file_ref(path):
        path = Path(path)
        if not path.is_absolute():
            path = workspace / path
        return {
            "path": path.relative_to(workspace).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def save(name, value):
        return put(workspace, "cutoff-controls/" + name, value)

    focus = pit_context(workspace, day)
    monkeypatch.setattr(
        "quant_investor.operations.research_recipes.read_bound_focus_pit", lambda **kwargs: focus
    )
    template = save(
        "corporate-template.json",
        {
            "schema_version": "cn-corporate-action-template.v1",
            "strategy_id": "aggressive_tech_manufacturing",
            "tracking_policy_ref": tracking_policy(workspace, book),
            "named_event_refs": None,
            "anchor_reviews_ref": None,
        },
    )
    value = base_recipe()
    value.update(
        schema_version="cn-daily-execute-recipe.v5",
        target_trade_date=day,
        release_ref=ctx.release_ref,
        release_install_ref=ctx.release_ref,
        previous_completion_ref={
            "path": "results/operations/daily_production/CN/20260827/completion.v1.json",
            "sha256": "b" * 64,
        },
        factor_loop_context_ref=ctx.release_ref,
        corporate_action_template_ref=template,
        theme_acquisition_ref=None,
        dashboard_publication_policy=POLICY,
        research_timing={
            "mode": mode,
            "policy_ref": save("timing.json", approved_research_timing_policy(mode)),
            "acquisition_deadline": acquisition_deadline(day) if mode == CURRENT else None,
        },
    )
    if registered_buy:
        value.update(
            schema_version="cn-daily-execute-recipe.v6",
            registered_event_declaration_ref=registered["declaration_ref"],
            previous_completion_ref=previous_completion,
        )
    value["policy_refs"] = {
        "research": file_ref(ctx.pool["payload"]["policy_path"]),
        "store": {"path": store_args["policy_path"], "sha256": store_args["policy_sha"]},
        "prospective": None,
    }
    value["store_preimages"] = {
        "store_pointer_ref": file_ref(book.root / "_record_store/current.v1.json"),
        "event_pointer_ref": file_ref(book.root / "_event_store/current.v1.json"),
        "benchmark_pointer_ref": file_ref("data/parquet/cn/benchmarks/_latest.json"),
    }
    value["dashboard_sources"] = {
        "benchmark_ref": file_ref("portfolio_dashboard/inputs/cn_index_benchmark.csv"),
        "risk_free_ref": file_ref("portfolio_dashboard/inputs/cn_govt_bond_yield.csv"),
    }
    value["research_sources"] = {
        "as_of": None if mode == CURRENT else "2026-08-28T13:30:00Z",
        "industry_source_ref": save("industry.json", request_fields["industry_source"]),
        "theme_source_ref": save(
            "theme.json",
            {
                "schema_version": THEME_SOURCE_V2,
                "pool": request_fields["theme_source"],
                "pcb_ai_hardware": None,
            },
        ),
        "exposure_rows_ref": None,
        "fundamental": {
            "mode": "PINNED",
            "source_ref": save(
                "fundamental.json", request_fields["company_evidence"]["fundamental_source"]
            ),
        },
        "macro": {
            "mode": "PINNED",
            "source_ref": save(
                "macro.json",
                {"classification": "CANONICAL_MACRO_READY", "source": macro["closure_ref"]},
            ),
        },
    }
    if exposure_catalog:
        from copy import deepcopy
        from quant_investor.operations.exposure_catalog import SCHEMA

        rows = deepcopy(request_fields["company_evidence"]["exposure_rows"])
        unused = {
            **rows[0],
            "company_code": "999999.SZ",
            "source": {"path": "unread-future-catalog-source.json", "sha256": "a" * 64},
            "available_at": "2099-01-01T00:00:00Z",
        }
        value["research_sources"]["exposure_rows_ref"] = save(
            "exposure-catalog.json", {"schema_version": SCHEMA, "rows": [unused, *reversed(rows)]}
        )
    recipe_ref = save("recipe.json", value)
    request = base_request()
    request.update(
        target_trade_date=day, release_install_ref=ctx.release_ref, recipe_ref=recipe_ref
    )
    if mode == HISTORICAL:
        request["schema_version"] = "cn-daily-production-request.v2"
        value["publish_current_dashboard"] = False
        recipe_ref = save("recipe.json", value)
        request["recipe_ref"] = recipe_ref
    request_ref = save("request.json", request)
    execution = journal.root / "executions" / request_ref["sha256"]
    retained_request = put(
        workspace, str(execution / "inputs" / f"request-{request_ref['sha256']}.json"), request
    )
    retained_recipe = put(
        workspace, str(execution / "inputs" / f"recipe-{recipe_ref['sha256']}.json"), value
    )
    outputs = {"top100": {"manifest.json": ctx.pool_ref}}
    for alias, node in (("LOW", "low_observation"), ("W80", "w80_observation")):
        outputs[node] = {alias: file_ref(f"results/factors/observations/2026/08/28/{alias}.json")}
    nodes = {
        node: save(
            "core-" + node + ".json",
            {
                "state": "SUCCEEDED",
                "finished_at": "2026-08-28T13:15:00Z",
                "output_refs": outputs.get(node, {}),
            },
        )
        for node in CORE_NODES
    }
    core_ref = put(
        workspace,
        str(journal.root / "core-handoff.v1.json"),
        {
            "schema_version": "cn-daily-core-handoff.v1",
            "trade_date": day,
            "graph_sha256": GRAPH_SHA256,
            "release_ref": ctx.release_ref,
            "node_refs": nodes,
            "authority": FALSE_AUTHORITY,
        },
    )
    market = file_ref("data/parquet/cn/_latest.json")
    market_copy = execution / "inputs" / f"market-pointer-{market['sha256']}.json"
    journal.storage.write(str(market_copy), (workspace / market["path"]).read_bytes())
    handoff = dict.fromkeys(
        HANDOFF_V2_FIELDS if mode == CURRENT else HANDOFF_V3_FIELDS, ctx.release_ref
    )
    handoff.update(
        schema_version=(
            "cn-daily-maintenance-handoff.v2"
            if mode == CURRENT
            else "cn-daily-maintenance-handoff.v3"
        ),
        trade_date=day,
        graph_sha256=GRAPH_SHA256,
        request_ref=retained_request,
        recipe_ref=retained_recipe,
        release_ref=ctx.release_ref,
        release_install_ref=ctx.release_ref,
        core_handoff_ref=core_ref,
        factor_pointer_ref={
            "path": "controlled-factor-pointer.json",
            "sha256": ctx.pool["payload"]["factor_pointer_sha256"],
        },
        market_pointer_ref={"path": str(market_copy), "sha256": market["sha256"]},
        market_snapshot_ref=file_ref(reader.snapshot()["manifest_path"]),
        calendar_ref=file_ref(book.calendar_path),
        sealed_at="2026-08-28T13:20:00Z",
        prospective_policy_ref=None,
        authority=FALSE_AUTHORITY,
    )
    handoff_ref = put(workspace, str(execution / "maintenance-handoff.v1.json"), handoff)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 28, 13, 20, tzinfo=timezone.utc)

    with monkeypatch.context() as planner:
        planner.setattr(native_store, "datetime", Clock)
        prepared = prepare_store_plan(store_args)
    prepared_ref = {"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]}
    return {
        "workspace": workspace,
        "journal": DailyJournal(str(workspace), day),
        "ctx": ctx,
        "book": book,
        "store_args": store_args,
        "macro": macro,
        "recovered": {"handoff_ref": handoff_ref, "handoff": handoff, "recipe": value},
        "prepared": {"store_plan_ref": prepared_ref, "retained_source_pointer_ref": None},
        "registered": registered,
    }


def _registered_book(workspace, book, prior_market, symbol, release_ref, monkeypatch):
    from _registered_event_fixture import apply_buy, ref
    from _native_daily_store_fixture import write
    from test_daily_evidence_requested_session import capture
    from scripts import manage_cn_strategy_records as manager
    from scripts import daily_completion_store
    from scripts.daily_production_store_adapter import StoreCloseAdapter

    # These are original synthetic Market publication bytes, selected in fixture
    # construction before any corresponding financial registration is sealed.
    market = workspace / "data/parquet/cn/_latest.json"
    final_market = market.read_bytes()
    assert prior_market is not None
    market.write_bytes(prior_market)
    for day in DAYS[:-1]:
        arguments = book.advance(day)
    with manager._operation_lock(book.root):
        clock = datetime(2026, 8, 27, 12, 20, tzinfo=timezone.utc)
        planned = native_store.close_through_latest(
            **arguments, execute=False, prepare_only=True, now=clock
        )
        native_store.close_through_latest(
            **arguments, execute=True, expected_plan_sha=planned["plan_sha256"], now=clock
        )
    plan_ref = {"path": planned["plan_path"], "sha256": planned["plan_sha256"]}
    baseline = ref(workspace, (workspace / plan_ref["path"]).with_name("committed-pointer.v1.json"))
    fixture = apply_buy(
        workspace,
        book=book,
        baseline_ref=baseline,
        baseline_arguments=arguments,
        trade_date="2026-08-28",
        buy_symbol=symbol,
    )
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            manager, "_manager_utc_now", lambda: datetime(2026, 8, 28, 12, 20, tzinfo=timezone.utc)
        )
        declaration = manager.command_publish_registered_event_declaration(fixture["args"])
    journal = DailyJournal(str(workspace), "20260827")
    adapter = StoreCloseAdapter(
        arguments=arguments, trade_date="20260827", plan_ref=plan_ref, release_ref=release_ref
    )
    with journal.locked():
        request = adapter.template()
        journal.begin(request)
        outcome = adapter.probe(request).outcome
        terminal = journal.finish(request, state=outcome.state, output_refs=outcome.output_refs)
    recorded = {
        "native_inputs_ref": put(
            workspace,
            "cutoff-controls/previous-store-input.json",
            {
                "schema_version": "cn-daily-native-inputs.v4",
                "store_plan_ref": plan_ref,
            },
        ),
        "node_terminal_refs": {"store": terminal["terminal_ref"]},
    }
    previous = put(
        workspace, "results/operations/daily_production/CN/20260827/completion.v1.json", recorded
    )

    def prior_eod(**kwargs):
        assert kwargs["trade_date"] == "20260827" and kwargs["completion_ref"] == previous
        return {"recorded_completion": recorded}

    # The prior Store replay is native; only all-node previous EOD admission is
    # controlled, just as the outer Factor/Core/PIT boundaries of this fixture.
    monkeypatch.setattr(daily_completion_store, "inspect_recorded_completion", prior_eod)
    market.write_bytes(final_market)
    arguments = book.advance("2026-08-28", publish_events=False)
    calendar = capture("2026-08-28T13:20:00+00:00")
    raw_path = workspace / "cutoff-controls/registered-calendar.raw.json"
    raw_path.write_bytes(calendar.raw_response_bytes)
    raw_path.chmod(0o600)
    raw = ref(workspace, raw_path)
    calendar.receipt["raw_response_path"] = str(workspace / raw["path"])
    book.calendar_sha = write(book.calendar_path, calendar.receipt)
    arguments.update(
        calendar_receipt_sha=book.calendar_sha,
        registered_event_declaration_ref=declaration["declaration_ref"],
    )
    return arguments, declaration, previous
