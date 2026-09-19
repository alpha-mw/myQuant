"""Native downstream v2 integration; deliberately not initial EXECUTE ordering proof."""

import hashlib
from pathlib import Path
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import GRAPH_SHA256
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.maintenance_handoff import publish_maintenance_handoff
from scripts.daily_materialization import materialize_daily_inputs
from scripts.daily_completion import run_materialized_native_input


def close_materialized_day(
    *,
    workspace: Path,
    request: dict,
    store_args: dict,
    core: dict,
    checkpoint: dict,
    loop_context: dict,
    installed_ref: dict,
    release_ref: dict,
    parent_pointer_sha256: str,
    previous_completion_ref: dict | None = None,
    cutoff_profile: bool = False,
    book=None,
    switch_to_actual_clock=None,
) -> dict:
    day = request["expected_trade_date"]

    def ref(path):
        path = Path(path)
        if not path.is_absolute():
            path = workspace / path
        return {
            "path": str(path.relative_to(workspace)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def put(name, value):
        path = workspace / "synthetic-v2-controls" / day / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        path.write_bytes(canonical_json_bytes(value))
        path.chmod(0o600)
        return ref(path)

    record_root = store_args["record_root"]
    preimages = {
        "store_pointer_ref": ref(record_root / "_record_store/current.v1.json"),
        "event_pointer_ref": ref(record_root / "_event_store/current.v1.json"),
        "benchmark_pointer_ref": ref(workspace / "data/parquet/cn/benchmarks/_latest.json"),
    }
    for field, arg in [
        ("store_pointer_ref", "expected_store_pointer_sha"),
        ("event_pointer_ref", "expected_event_pointer_sha"),
        ("benchmark_pointer_ref", "expected_benchmark_pointer_sha"),
    ]:
        if preimages[field]["sha256"] != store_args[arg]:
            raise ValueError("native fixture preimage changed")
    bootstrap = None
    if previous_completion_ref is None:
        # The fixture's native baseline is Aug26; original Store plan may close a longer prefix.
        bootstrap = put(
            "bootstrap.json",
            dict(
                schema_version="cn-daily-bootstrap.v1",
                market="CN",
                strategy_id="aggressive_tech_manufacturing",
                first_trade_date=day,
                previous_trade_date="20260826",
                graph_sha256=GRAPH_SHA256,
                factor_parent_pointer_sha256=parent_pointer_sha256,
                store_preimages=preimages,
                authority=FALSE_AUTHORITY,
            ),
        )
    evidence = request["company_evidence"]
    recipe = put(
        "recipe.json",
        dict(
            schema_version="cn-daily-execute-recipe.v1",
            market="CN",
            strategy_id="aggressive_tech_manufacturing",
            target_trade_date=day,
            graph_sha256=GRAPH_SHA256,
            release_ref=release_ref,
            release_install_ref=installed_ref,
            previous_completion_ref=previous_completion_ref,
            bootstrap_ref=bootstrap,
            factor_loop_context_ref=loop_context,
            policy_refs={
                "research": ref("results/policies/research/aggressive_tech_manufacturing/v2.json"),
                "store": {"path": store_args["policy_path"], "sha256": store_args["policy_sha"]},
                "prospective": None,
            },
            store_preimages=preimages,
            retrospective_ref=None,
            dashboard_sources={
                "benchmark_ref": ref("portfolio_dashboard/inputs/cn_index_benchmark.csv"),
                "risk_free_ref": ref("portfolio_dashboard/inputs/cn_govt_bond_yield.csv"),
            },
            research_sources={
                "as_of": request["as_of"],
                "industry_source_ref": put("industry-source.json", request["industry_source"]),
                "theme_source_ref": put("theme-source.json", request["theme_source"]),
                "exposure_rows_ref": put("exposure-source.json", evidence["exposure_rows"]),
                "fundamental": {
                    "mode": "PINNED",
                    "source_ref": put("fundamental-source.json", evidence["fundamental_source"]),
                },
                "macro": {
                    "mode": "PINNED",
                    "source_ref": put("macro-source.json", evidence["macro_risk"]),
                },
            },
            publish_current_dashboard=True,
        ),
    )
    if cutoff_profile:
        import json
        from _native_corporate_fixture import tracking_policy
        from quant_investor.operations.research_timing import (
            CURRENT,
            acquisition_deadline,
            approved_research_timing_policy,
        )
        from quant_investor.operations.dashboard_serving_contract import POLICY
        from quant_investor.intelligence.theme_sources import THEME_SOURCE_V2
        from _native_cutoff_installed_scenario import focus_theme_source

        if book is None:
            raise ValueError("native cutoff fixture needs its native book")
        document = json.loads((workspace / recipe["path"]).read_bytes())
        document.update(
            schema_version="cn-daily-execute-recipe.v5",
            theme_acquisition_ref=None,
            corporate_action_template_ref=put(
                "corporate-template.json",
                {
                    "schema_version": "cn-corporate-action-template.v1",
                    "strategy_id": "aggressive_tech_manufacturing",
                    "tracking_policy_ref": tracking_policy(workspace, book),
                    "named_event_refs": None,
                    "anchor_reviews_ref": None,
                },
            ),
            dashboard_publication_policy=POLICY,
            research_timing={
                "mode": CURRENT,
                "policy_ref": put("timing.json", approved_research_timing_policy(CURRENT)),
                "acquisition_deadline": acquisition_deadline(day),
            },
            publish_current_dashboard=False,
        )
        document["research_sources"]["as_of"] = None
        document["research_sources"]["theme_source_ref"] = put(
            "theme-source-v2.json",
            {
                "schema_version": THEME_SOURCE_V2,
                "pool": request["theme_source"],
                "pcb_ai_hardware": focus_theme_source(workspace, day),
            },
        )
        recipe = put("recipe-cutoff.json", document)
    production = put(
        "request.json",
        dict(
            schema_version="cn-daily-production-request.v1",
            market="CN",
            strategy_id="aggressive_tech_manufacturing",
            action="EXECUTE",
            target_trade_date=day,
            graph_sha256=GRAPH_SHA256,
            release_install_ref=installed_ref,
            recipe_ref=recipe,
            maintenance_handoff_ref=None,
            calendar_ref=None,
            raw_calendar_ref=None,
            previous_completion_ref=None,
            day_input_refs={},
        ),
    )
    handoff = publish_maintenance_handoff(
        workspace=str(workspace),
        request_ref=production,
        state={"core_checkpoint_ref": checkpoint, "core_handoff_ref": core["core_handoff_ref"]},
    )
    materialized = materialize_daily_inputs(workspace=str(workspace), handoff_ref=handoff)
    if cutoff_profile and switch_to_actual_clock is not None:
        switch_to_actual_clock()
    result = run_materialized_native_input(
        workspace=str(workspace),
        input_ref=materialized["native_inputs_ref"],
        resume=True,
        synthetic=True,
    )
    return {
        "status": result,
        "handoff_ref": handoff,
        "materialization": materialized,
        "synthetic": True,
        "initial_execute_ordering_proven": False,
    }
