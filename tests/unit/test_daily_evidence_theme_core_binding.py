"""Native recipe/pool/capture replay; core context and provider transport are fixtures."""

from datetime import datetime, timezone
from types import SimpleNamespace
import pytest
from test_daily_evidence_research_sources import context, put
from test_daily_evidence_production_request import request, recipe
from quant_investor.operations import theme_core_binding as module
from quant_investor.operations.execution_recipe import SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5
from quant_investor.operations.dashboard_serving_contract import POLICY as DASHBOARD_POLICY
from quant_investor.operations.theme_acquisition import (
    POLICY_SCHEMA,
    POLICY_SCHEMA_V2,
    POLICY_SCHEMA_V3,
)
from quant_investor.operations.research_timing import (
    CURRENT,
    approved_research_timing_policy,
    acquisition_deadline,
)
from quant_investor.factors.production_pit import FOCUS_COMPANIES
from test_daily_evidence_focus_sources import pit_context
from quant_investor.operations.daily_contract import ContractError


@pytest.mark.parametrize("fault", [None, "pool", "core_change"])
@pytest.mark.parametrize(
    "recipe_schema,focus_enabled",
    [
        (SCHEMA_V2, False),
        (SCHEMA_V2, True),
        (SCHEMA_V3, False),
        (SCHEMA_V3, True),
        (SCHEMA_V4, False),
        (SCHEMA_V4, True),
        (SCHEMA_V5, True),
    ],
)
def test_scope_is_derived_from_exact_verified_pool(
    tmp_path, monkeypatch, fault, focus_enabled, recipe_schema
):
    ctx, journal = context(tmp_path)
    day = journal.trade_date
    req, rec = request(), recipe()
    req["target_trade_date"] = rec["target_trade_date"] = day
    rec["schema_version"] = recipe_schema
    if recipe_schema in {SCHEMA_V3, SCHEMA_V4}:
        rec["corporate_action_context_ref"] = {"path": "corporate.json", "sha256": "c" * 64}
    if recipe_schema in {SCHEMA_V4, SCHEMA_V5}:
        rec["dashboard_publication_policy"] = DASHBOARD_POLICY
    rec["research_sources"]["as_of"] = ctx.request["as_of"]
    if recipe_schema == SCHEMA_V5:
        rec["corporate_action_template_ref"] = {
            "path": "corporate-template.json",
            "sha256": "c" * 64,
        }
        rec["research_sources"]["as_of"] = None
        rec["research_timing"] = {
            "mode": CURRENT,
            "policy_ref": put(tmp_path, "timing.json", approved_research_timing_policy(CURRENT)),
            "acquisition_deadline": acquisition_deadline(day),
        }
    rec["previous_completion_ref"][
        "path"
    ] = "results/operations/daily_production/CN/20260827/completion.v1.json"
    rec["theme_acquisition_ref"] = put(
        tmp_path,
        "acquisition.json",
        {
            "schema_version": (
                POLICY_SCHEMA_V3
                if recipe_schema == SCHEMA_V5
                else POLICY_SCHEMA_V2 if focus_enabled else POLICY_SCHEMA
            ),
            "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
            "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
            "maximum_companies": 100,
            **({"special_company_keyset": list(FOCUS_COMPANIES)} if focus_enabled else {}),
        },
    )
    req["recipe_ref"] = put(tmp_path, "recipe.json", rec)
    req_ref = put(tmp_path, "request.json", req)
    pool = dict(ctx.pool_ref)
    if fault == "pool":
        pool["sha256"] = "b" * 64
    terminal = put(
        tmp_path,
        "core/key/attempt-0001/terminal.json",
        {"output_refs": {"manifest.json": pool}, "finished_at": "2026-08-28T12:00:00Z"},
    )
    put(tmp_path, "core/key/request.json", {"fixture": "core request"})
    core_ref = {"path": "core.json", "sha256": "a" * 64}
    calls = []

    def core(**kwargs):
        assert kwargs["trade_date"] == day and kwargs["handoff_ref"] == core_ref
        calls.append("core")
        value = {
            "node_terminal_refs": {"top100": terminal},
            "factor_pointer_ref": {"path": "pointer.json", "sha256": "a" * 64},
        }
        if fault == "core_change" and len(calls) > 1:
            value["changed"] = True
        return value

    arguments = {
        "rank": ctx.rank,
        "expected_policy_sha256": ctx.pool["payload"]["policy_byte_sha256"],
        "policy_path": ctx.pool["payload"]["policy_path"],
    }
    pit_calls = []
    if focus_enabled:
        arguments["observations"] = ctx.pool_store._observations(ctx.rank, None)
        focus = pit_context(tmp_path, day)

        def owning_pit(**kwargs):
            observed = arguments["observations"][0]["payload"]
            assert kwargs["generation_ref"] == ctx.rank["payload"]["factor_generation_ref"]
            assert kwargs["expected_manifest_sha256"] == observed["pit_manifest_sha256"]
            assert kwargs["expected_membership_sha256"] == observed["pit_membership_sha256"]
            pit_calls.append(kwargs)
            return focus

        monkeypatch.setattr(module, "read_bound_focus_pit", owning_pit)
    monkeypatch.setattr(module, "inspect_core_handoff", core)
    monkeypatch.setattr(
        module,
        "CoreContext",
        lambda *a: SimpleNamespace(
            pool=ctx.pool_store, pool_arguments=lambda req: arguments, recheck=lambda: None
        ),
    )
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if fault:
        with pytest.raises(ContractError, match="THEME_CORE"):
            module.bind_theme_acquisition(
                workspace=str(tmp_path), request_ref=req_ref, core_handoff_ref=core_ref
            )
    else:
        result = module.bind_theme_acquisition(
            workspace=str(tmp_path), request_ref=req_ref, core_handoff_ref=core_ref
        )
        assert result["company_keyset"] == sorted(ctx.companies)
        assert result["identity"]["pool_manifest_ref"] == ctx.pool_ref
        assert result["identity"]["rank_ref"]["sha256"] == ctx.pool["payload"]["rank_byte_sha256"]
        if focus_enabled:
            assert result["special_company_keyset"] == list(FOCUS_COMPANIES)
            assert result["focus_context"] == focus and len(pit_calls) == 1
        if recipe_schema == SCHEMA_V5:
            assert "research_cutoff" not in result
            assert result["acquisition_deadline"] == acquisition_deadline(day)
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    if fault is None and recipe_schema in {SCHEMA_V3, SCHEMA_V4, SCHEMA_V5}:
        _exercise_native_handoff(tmp_path, monkeypatch, journal, req_ref, core_ref, result)


def _exercise_native_handoff(root, monkeypatch, journal, request_ref, core_ref, binding):
    from quant_investor.operations import theme_capture_stage as stage, theme_acquisition
    from quant_investor.operations import theme_handoff_publish as publisher
    from quant_investor.operations.theme_handoff_readback import read_theme_handoff
    from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
    from test_tushare_theme_capture_stable import FakeClient

    class Clock(datetime):
        hour = 13

        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 28, cls.hour, tzinfo=timezone.utc)

    for owner in (stage, theme_acquisition, publisher):
        monkeypatch.setattr(owner, "datetime", Clock)
    day = journal.trade_date
    theme = next(t.split(":", 1)[1] for t in TECHNOLOGY_THEME_IDS if t.startswith("TUSHARE_DC:"))
    companies = sorted(set(binding["company_keyset"] + binding.get("special_company_keyset", [])))
    rows = {("dc_index", "ALL"): [(theme, day, "fixture theme", "概念板块", "1")]}
    for company in companies:
        rows[("dc_member", company)] = [(day, theme, company, "fixture company")]
    client = FakeClient(rows)
    monkeypatch.setattr(stage.native, "OfficialTushareHttpsClient", lambda **kwargs: client)
    args = dict(journal=journal, request_ref=request_ref, core_handoff_ref=core_ref)
    with journal.locked():
        handoff_ref = publisher.publish_theme_handoff(**args)
    replay = read_theme_handoff(**args, handoff_ref=handoff_ref)
    assert replay["binding"] == binding
    assert replay["handoff"]["request_ref"] == request_ref
    expected_calls = 1 + len(binding["company_keyset"])
    if "special_company_keyset" in binding:
        expected_calls += 1 + len(binding["special_company_keyset"])
    assert len(client.calls) == expected_calls
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }
    Clock.hour = 17 if "acquisition_deadline" in binding else 14
    monkeypatch.setattr(
        stage.native, "capture_theme_plan", lambda **kwargs: pytest.fail("provider on replay")
    )
    with journal.locked():
        assert publisher.publish_theme_handoff(**args) == handoff_ref
    monkeypatch.setattr(journal.storage, "write", lambda *a, **k: pytest.fail("readback wrote"))
    assert read_theme_handoff(**args, handoff_ref=handoff_ref) == replay
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }
    assert len(client.calls) == expected_calls


@pytest.mark.parametrize("fault", ["corporate", "dashboard", "extra", "unknown", "legacy"])
def test_recipe_validation_precedes_core_or_acquisition(tmp_path, monkeypatch, fault):
    req, rec = request(), recipe()
    if fault != "legacy":
        rec.update(
            schema_version=SCHEMA_V4,
            corporate_action_context_ref={"path": "corporate.json", "sha256": "c" * 64},
            dashboard_publication_policy=DASHBOARD_POLICY,
            theme_acquisition_ref={"path": "acquisition.json", "sha256": "d" * 64},
        )
    if fault == "corporate":
        rec["corporate_action_context_ref"] = None
    elif fault == "dashboard":
        rec["dashboard_publication_policy"] = "unregistered-policy"
    elif fault == "extra":
        rec["ignore_missing"] = True
    elif fault == "unknown":
        rec["schema_version"] = "cn-daily-execute-recipe.v999"
    req["recipe_ref"] = put(tmp_path, "recipe.json", rec)
    req_ref = put(tmp_path, "request.json", req)

    def forbidden(**kwargs):
        pytest.fail("invalid acquisition recipe reached native core")

    monkeypatch.setattr(module, "inspect_core_handoff", forbidden)
    with pytest.raises(ContractError):
        module.bind_theme_acquisition(
            workspace=str(tmp_path),
            request_ref=req_ref,
            core_handoff_ref={"path": "core.json", "sha256": "a" * 64},
        )
