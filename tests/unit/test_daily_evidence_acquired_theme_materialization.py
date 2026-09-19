"""Native Theme capture -> materialization; core binding/transport/clock controlled."""

from datetime import datetime, timezone
import hashlib
import json
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations import theme_capture_stage as stage, theme_acquisition as claims
from quant_investor.operations import theme_handoff_readback as reader
from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
from test_tushare_theme_capture_stable import FakeClient
from test_daily_evidence_daily_materialization import context
from scripts import daily_materialization as materializer
from quant_investor.factors.production_pit import FOCUS_COMPANIES
from test_daily_evidence_research_sources import put
from test_daily_evidence_focus_sources import pit_context


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("focus_enabled", [False, True])
def test_execute_captures_native_theme_and_passive_replay_has_no_provider(
    tmp_path, monkeypatch, fallback, focus_enabled
):
    journal, recovered, _ = context(tmp_path, 2)
    recipe, handoff = recovered["recipe"], recovered["handoff"]

    def rewrite(ref, value):
        raw = canonical_json_bytes(value)
        (tmp_path / ref["path"]).write_bytes(raw)
        ref["sha256"] = hashlib.sha256(raw).hexdigest()

    policy = put(
        tmp_path,
        "fixtures/theme-policy.json",
        {
            "schema_version": claims.POLICY_SCHEMA_V2 if focus_enabled else claims.POLICY_SCHEMA,
            "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
            "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
            "maximum_companies": 100,
            **({"special_company_keyset": list(FOCUS_COMPANIES)} if focus_enabled else {}),
        },
    )
    recipe["theme_acquisition_ref"] = policy
    recipe["research_sources"]["theme_source_ref"] = None
    recipe["research_sources"]["as_of"] = "2026-08-24T13:30:00Z"
    rewrite(handoff["recipe_ref"], recipe)
    rewrite(recovered["handoff_ref"], handoff)
    companies = ["000001.SZ"]
    company_sha = hashlib.sha256(canonical_json_bytes(companies)).hexdigest()
    identity = {
        "request_ref": handoff["request_ref"],
        "core_handoff_ref": handoff["core_handoff_ref"],
        "pool_manifest_ref": {"path": "fixtures/pool.json", "sha256": "a" * 64},
        "rank_ref": {"path": "fixtures/rank.json", "sha256": "a" * 64},
        "company_set_sha256": company_sha,
        "acquisition_policy_ref": policy,
        "release_ref": handoff["release_ref"],
    }
    binding = {
        "trade_date": journal.trade_date,
        "company_keyset": companies,
        "research_cutoff": recipe["research_sources"]["as_of"],
        "core_completed_at": "2026-08-24T12:00:00Z",
        "identity": identity,
    }
    if focus_enabled:
        focus = pit_context(tmp_path, journal.trade_date)
        binding.update(
            policy_schema=claims.POLICY_SCHEMA_V2,
            focus_context=focus,
            special_company_keyset=list(FOCUS_COMPANIES),
            special_company_set_sha256=focus["company_set_sha256"],
            **{
                key: focus[key]
                for key in (
                    "pit_selection_ref",
                    "pit_generation_manifest_ref",
                    "pit_membership_ref",
                )
            },
        )
    calls = []

    def bind(**kwargs):
        assert kwargs["request_ref"] == handoff["request_ref"]
        assert kwargs["core_handoff_ref"] == handoff["core_handoff_ref"]
        calls.append("bind")
        return binding

    monkeypatch.setattr(stage, "bind_theme_acquisition", bind)
    monkeypatch.setattr(reader, "bind_theme_acquisition", bind)
    monkeypatch.setattr(
        "quant_investor.operations.theme_handoff_publish.bind_theme_acquisition", bind
    )

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 24, 13, tzinfo=timezone.utc)

    monkeypatch.setattr(stage, "datetime", Clock)
    monkeypatch.setattr(claims, "datetime", Clock)
    theme = TECHNOLOGY_THEME_IDS[0].split(":", 1)[1]
    tdx = next(
        (n.split(":", 1)[1] for n in TECHNOLOGY_THEME_IDS if n.startswith("TUSHARE_TDX:")),
        "880001.TDX",
    )
    day = journal.trade_date
    client = FakeClient(
        {
            ("dc_index", "ALL"): [(theme, day, "fixture", "概念板块", "1")],
            ("dc_member", companies[0]): [
                (day, "BK9999.DC" if fallback else theme, companies[0], "fixture")
            ],
            ("tdx_index", "ALL"): [(tdx, day, "fixture", "概念板块", 20)],
            ("tdx_member", companies[0]): [(tdx, day, companies[0], "fixture")],
        }
    )
    if focus_enabled:
        for company in FOCUS_COMPANIES:
            client.rows_by_api_and_company[("dc_member", company)] = [
                (day, "BK9999.DC" if fallback else theme, company, "fixture focus")
            ]
            client.rows_by_api_and_company[("tdx_member", company)] = [
                (tdx, day, company, "fixture focus")
            ]
    monkeypatch.setattr(stage.native, "OfficialTushareHttpsClient", lambda **kwargs: client)
    with journal.locked():
        first = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}, _execute_theme=True
        )
    assert len(client.calls) == (4 if fallback else 2) + (
        (6 if fallback else 3) if focus_enabled else 0
    )
    assert calls
    assert first.materialization_ref["path"].endswith("materialization.v2.json")
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("completed acquisition replay reached producer, writer, or lock")

    monkeypatch.setattr(stage, "capture_bound_theme", forbidden)
    monkeypatch.setattr(stage.native, "OfficialTushareHttpsClient", forbidden)
    with journal.locked():
        again = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}, _execute_theme=True
        )
    assert again.materialization_ref == first.materialization_ref
    monkeypatch.setattr(journal, "locked", forbidden)
    monkeypatch.setattr(journal.storage, "write", forbidden)
    record = materializer.verify_materialized_inputs(
        journal=journal, recovered=recovered, materialized=first, readonly=True
    )
    assert record["theme_source_handoff_ref"] is not None
    assert record["theme_source_handoff_ref"]["path"].endswith(
        "handoff.v2.json" if focus_enabled else "handoff.v1.json"
    )
    research = json.loads((tmp_path / record["research_request_ref"]["path"]).read_bytes())
    if focus_enabled:
        assert research["theme_source"]["schema_version"] == "cn-daily-theme-evidence-source.v2"
        assert research["theme_source"]["pcb_ai_hardware"] is not None
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
