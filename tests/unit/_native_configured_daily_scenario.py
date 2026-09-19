"""Installed pre-Core configuration -> native producer/EOD; synthetic inputs and clocks.

Market/PIT/history are native preproduced inputs, as in the earlier full scenario.
Their maintenance NO_ACTION adapters are explicit fixture boundaries. Factor,
pool, source validation, Fundamental/Macro owners, cutoff, Decision, Store,
Dashboard and release verification are not replaced by positive stubs.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

DAY = "20260827"
NOW = datetime(2026, 8, 27, 13, 20, tzinfo=timezone.utc)


def run(root, dependency_path, *, dispatch_case=None):
    import quant_investor

    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(receipt["runtime_verification"]["import_origin"]).resolve()
    ):
        raise ValueError("verified installed package required")
    repository = root / "repository"
    sys.path.extend(
        [
            str(repository),
            str(repository / "scripts"),
            str(repository / "tests/unit"),
            str(dependency_path),
        ]
    )
    from _native_full_dag_scenario import run as seed

    from _native_synthetic_clock import synthetic_clock

    with synthetic_clock(NOW):
        state = {}

        def baseline(root, fixture, activated):
            state["book"] = _baseline_book(root, fixture)

        return seed(
            root,
            dependency_path,
            before_current_market=baseline,
            pre_core_entry=lambda root, fixture, activated: _configured(
                root, fixture, activated, book=state["book"], dispatch_case=dispatch_case
            ),
        )


def _baseline_book(root, fixture):
    from _native_daily_store_fixture import NativeStoreFixture, DAYS
    from scripts import cn_official_close_batch as store
    from scripts.manage_cn_strategy_records import _operation_lock

    workspace = root / "factor-workspace"
    book = NativeStoreFixture(workspace, stock_symbols=fixture.symbols[:7], preserve_market=True)
    for date in DAYS[:3]:
        arguments = book.advance(date)
    with _operation_lock(book.root):
        prepared = store.close_through_latest(**arguments, execute=False, prepare_only=True)
        closed = store.close_through_latest(
            **arguments, execute=True, expected_plan_sha=prepared["plan_sha256"]
        )
    (root / "configured-baseline-store.json").write_text(json.dumps(closed, indent=2) + "\n")
    print("PASS configured-baseline-store.json", flush=True)
    return book


def _configured(root, fixture, activated, *, book, dispatch_case=None):
    from _native_daily_calendar_fixture import configured_future_calendar_scope

    with configured_future_calendar_scope(root=root, trade_date=DAY):
        return _configured_inner(root, fixture, activated, book=book, dispatch_case=dispatch_case)


def _configured_inner(root, fixture, activated, *, book, dispatch_case=None):
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.market.daily_components import build_default_components
    from quant_investor.market.daily_macro_layout import select_macro_layout
    from quant_investor.operations import theme_capture_stage
    from quant_investor.operations.exposure_catalog import SCHEMA as EXPOSURE_CATALOG_SCHEMA
    from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
    from _native_daily_maintenance_fixture import maintenance
    from _native_daily_research_inputs import augment
    from _native_shared_macro_fixture import build_shared_macro
    from _native_corporate_fixture import tracking_policy
    from _daily_preparation_fixture import config as config_template
    from test_tushare_theme_capture_stable import FakeClient
    from test_daily_evidence_requested_session import capture
    from scripts.daily_source_inputs import configured_source_inputs
    from scripts.daily_production import dispatch_daily_request
    from scripts.daily_completion_replay import replay_native_completion
    from quant_investor.intelligence.storage import (
        publish_theme_policy_v2,
        approved_theme_policy_v2,
    )

    workspace = root / "factor-workspace"
    journal_root = workspace / "results/operations/daily_production/CN" / DAY

    def put(name, value):
        path = workspace / "configured-inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent
        while parent != workspace:
            parent.chmod(0o700)
            parent = parent.parent
        raw = canonical_json_bytes(value)
        path.write_bytes(raw)
        path.chmod(0o600)
        return {"path": str(path.relative_to(workspace)), "sha256": hashlib.sha256(raw).hexdigest()}

    def save(name, value):
        (root / name).write_text(json.dumps(value, indent=2, default=str) + "\n")
        print("PASS", name, flush=True)

    book.advance("2026-08-27")
    publish_theme_policy_v2(workspace)
    installed_document = json.loads((root / "release-input.json").read_bytes())
    installed = put("installation.json", installed_document)
    # The Core owner binds the native immutable release location. An identical
    # copy at a different path is not the same source reference for the DAG.
    release_raw = canonical_json_bytes(installed_document["deployed_release"])
    release_sha = hashlib.sha256(release_raw).hexdigest()
    release = {
        "path": "results/factors/objects/system.release/" + release_sha + ".json",
        "sha256": release_sha,
    }
    if (workspace / release["path"]).read_bytes() != release_raw:
        raise ValueError("configured request requires the exact native Core release source")
    loop_context = put(
        "loop.json",
        {
            "schema_version": "cn-daily-factor-loop.v2",
            "initial_calendar_receipt_ref": None,
            "next_session_calendar_mode": "SYNTHETIC_FIXTURE_ONLY",
            "release_install_input_ref": installed,
            "release_repository_root": str(root / "repository"),
            "release_commit": json.loads((root / "fixture-receipt.json").read_bytes())["commit"],
            "calendar_capture_parent": str(workspace / "data/private/configured-calendar-captures"),
        },
    )
    source_seed = put("broad-source-seed.json", {"policy": approved_theme_policy_v2()})
    (root / ("research-inputs-" + DAY + ".json")).write_text(
        json.dumps(
            {
                "synthetic": True,
                "source_scope": "PRE_CORE_BROAD_UNIVERSE",
                "companies": fixture.symbols,
                "request_ref": source_seed,
            }
        )
        + "\n"
    )
    expanded = augment(root, DAY, native_fundamental=True)
    sources = json.loads((workspace / expanded["request_ref"]["path"]).read_bytes())
    context = SimpleNamespace(
        workspace_root=workspace,
        run_root=workspace / "data/private/cn_daily_maintenance",
        target_date=DAY,
    )
    layout = select_macro_layout(context)
    macro = build_shared_macro(workspace, "2026-08-27", transaction_identity=layout.transaction_id)
    save("configured-native-macro-source.json", macro)
    policy = json.loads((workspace / book.policy_path).read_bytes())
    policy.update(
        effective_from="2026-08-01T00:00:00Z",
        event_inbox={
            "pointer_path": str(book.root.relative_to(workspace)) + "/_event_store/current.v1.json",
            "owner_append_cutoff_local": "15:30:00",
            "timezone": "Asia/Shanghai",
            "sealed_empty_inventory_is_owner_authorized_closure": True,
            "late_event_behavior": "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED",
        },
    )
    config, _ = config_template(workspace)
    config.update(
        release_ref=release,
        release_install_ref=installed,
        factor_loop_context_ref=loop_context,
        store_policy_ref=put("standing-store-policy.json", policy),
        industry_source_ref=put("industry.json", sources["industry_source"]),
        exposure_rows_ref=put(
            "exposure.json",
            {
                "schema_version": EXPOSURE_CATALOG_SCHEMA,
                "rows": sources["company_evidence"]["exposure_rows"],
            },
        ),
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
    )
    risk = workspace / "portfolio_dashboard/inputs/cn_govt_bond_yield.csv"
    config["risk_free_ref"] = {
        "path": str(risk.relative_to(workspace)),
        "sha256": hashlib.sha256(risk.read_bytes()).hexdigest(),
    }
    config_ref = put("source-config.json", config)
    assert not journal_root.exists(), "configuration must precede native daily Core"
    with patch(
        "quant_investor.operations.source_slot_inputs.acquire_close_session_authority",
        lambda **kw: capture(kw["now"].isoformat()),
    ):
        source_result = configured_source_inputs(
            workspace=str(workspace),
            config_ref=config_ref,
            release_install_ref=installed,
            synthetic=True,
            mode="provision",
        )
    save("configured-source-result.json", source_result)
    if source_result["mode"] != "REQUEST_AVAILABLE":
        raise AssertionError("configured source did not prepare an executable original request")
    request_ref = source_result["request_ref"]
    initial_request = json.loads((workspace / request_ref["path"]).read_bytes())
    if initial_request["action"] != "EXECUTE":
        raise AssertionError("initial configured request must select native bootstrap")
    native_components = build_default_components(workspace_root=workspace)

    def maintained(**kwargs):
        maintained = maintenance(
            root,
            "2026-08-27",
            core_completed=kwargs["core_completed"],
            expected_target_trade_date=kwargs.get("_expected_target_trade_date"),
            core_replay_completed=kwargs.get("_core_replay_completed"),
            macro_callback=native_components.macro_release,
            now=NOW,
        )
        return maintained["maintenance_result"]

    theme = next(
        item.split(":", 1)[1] for item in TECHNOLOGY_THEME_IDS if item.startswith("TUSHARE_DC:")
    )
    rows = {("dc_index", "ALL"): [(theme, DAY, "synthetic", "概念板块", "1")]}
    rows.update(
        {
            ("dc_member", company): [(DAY, theme, company, "synthetic")]
            for company in fixture.symbols
        }
    )
    client = FakeClient(rows)
    with (
        patch("quant_investor.market.daily_maintenance.run_cn_daily_maintenance", maintained),
        patch.object(theme_capture_stage.native, "OfficialTushareHttpsClient", lambda **kw: client),
    ):
        if dispatch_case is not None:
            return dispatch_case(
                root=root,
                workspace=workspace,
                request_ref=request_ref,
                config_ref=config_ref,
                release_install_ref=installed,
            )
        result = dispatch_daily_request(
            workspace=str(workspace),
            request_ref=request_ref,
            release_install_ref=installed,
            synthetic=True,
        )
    save("configured-native-eod-result.json", result)
    if result["business_state"] != "COMPLETE":
        raise AssertionError(
            "configured native EOD incomplete; inspect original result and journal"
        )
    completion_ref = result["days"][0]["completion_ref"]
    replay = replay_native_completion(
        workspace=str(workspace), trade_date=DAY, completion_ref=completion_ref
    )
    if not replay["native_replay_validated"] or not replay["synthetic"]:
        raise AssertionError("native completed replay/provenance did not validate")
    proof = {
        "synthetic": True,
        "production_deployed": False,
        "real_provider_calls": False,
        "request_ref": request_ref,
        "config_ref": config_ref,
        "source_result": source_result,
        "completion_ref": completion_ref,
        "replay": replay,
        "theme_transport_calls": len(client.calls),
        "pre_core_configuration": True,
        "factor_core_and_release_verifier_mocked": False,
        "limitations": [
            "synthetic prices, external transports and logical clock",
            "maintenance verifies native preproduced Market/PIT/history "
            "through fixture NO_ACTION adapters",
            "one configured bootstrap day; successor automatic/five-session "
            "and Morning acceptance remain",
        ],
    }
    save("configured-native-proof.json", proof)
    return proof


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
