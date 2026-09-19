"""Native v2 capture/handoff, with external transport and core binding controlled."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.production_pit import FOCUS_COMPANIES
from quant_investor.intelligence.theme_sources import THEME_SOURCE_V2
from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
from quant_investor.operations import theme_capture_stage as stage, theme_acquisition as claims
from quant_investor.operations import (
    theme_handoff_publish as publisher,
    theme_handoff_readback as reader,
)
from quant_investor.operations.theme_handoff import SCHEMA_V2, validate_theme_handoff
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import ContractError
from test_daily_evidence_theme_claim import identity
from test_daily_evidence_focus_sources import pit_context, inventory
from test_tushare_theme_capture_stable import FakeClient


def binding_case(root, monkeypatch):
    companies = ["000001.SZ"]
    bound = identity()
    bound["company_set_sha256"] = hashlib.sha256(canonical_json_bytes(companies)).hexdigest()
    focus = pit_context(root)
    binding = {
        "trade_date": "20260828",
        "company_keyset": companies,
        "research_cutoff": "2026-08-28T13:30:00Z",
        "core_completed_at": "2026-08-28T12:00:00Z",
        "identity": bound,
        "policy_schema": claims.POLICY_SCHEMA_V2,
        "special_company_keyset": list(FOCUS_COMPANIES),
        "special_company_set_sha256": focus["company_set_sha256"],
        "focus_context": focus,
        **{
            key: focus[key]
            for key in ("pit_selection_ref", "pit_generation_manifest_ref", "pit_membership_ref")
        },
    }
    for module in (stage, publisher, reader):
        monkeypatch.setattr(module, "bind_theme_acquisition", lambda **kwargs: deepcopy(binding))

    class Clock(datetime):
        hour = 13

        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 28, cls.hour, tzinfo=timezone.utc)

    monkeypatch.setattr(stage, "datetime", Clock)
    monkeypatch.setattr(claims, "datetime", Clock)
    monkeypatch.setattr(publisher, "datetime", Clock)
    return binding, Clock


@pytest.mark.parametrize("mode", ["dc", "fallback", "missing"])
def test_both_scopes_native_capture_and_handoff_replay(tmp_path, monkeypatch, mode):
    binding, clock = binding_case(tmp_path, monkeypatch)
    dc = next(t.split(":", 1)[1] for t in TECHNOLOGY_THEME_IDS if t.startswith("TUSHARE_DC:"))
    tdx = next(t.split(":", 1)[1] for t in TECHNOLOGY_THEME_IDS if t.startswith("TUSHARE_TDX:"))
    day = binding["trade_date"]
    data = {
        ("dc_index", "ALL"): [(dc, day, "fixture PCB", "概念板块", "1")],
        ("tdx_index", "ALL"): [(tdx, day, "fixture fallback", "概念板块", 20)],
    }
    for company in [*binding["company_keyset"], *FOCUS_COMPANIES]:
        data[("dc_member", company)] = [
            (
                day,
                "BK9999.DC" if mode == "fallback" and company == FOCUS_COMPANIES[0] else dc,
                company,
                "fixture",
            )
        ]
        data[("tdx_member", company)] = [(tdx, day, company, "fixture")]
    client = FakeClient(data)
    if mode == "missing":
        original = client.request

        def failed(*, api_name, params, expected_fields):
            if api_name in {"dc_member", "tdx_member"} and FOCUS_COMPANIES[0] in params.values():
                client.calls.append((api_name, dict(params)))
                raise RuntimeError("synthetic focus provider unavailable")
            return original(api_name=api_name, params=params, expected_fields=expected_fields)

        monkeypatch.setattr(client, "request", failed)
    monkeypatch.setattr(stage.native, "OfficialTushareHttpsClient", lambda **kwargs: client)
    journal = DailyJournal(str(tmp_path), day)
    args = {
        "journal": journal,
        "request_ref": binding["identity"]["request_ref"],
        "core_handoff_ref": binding["identity"]["core_handoff_ref"],
    }
    with journal.locked():
        ref = publisher.publish_theme_handoff(**args)
    value = json.loads((tmp_path / ref["path"]).read_bytes())
    assert ref["path"].endswith("handoff.v2.json") and value["schema_version"] == SCHEMA_V2
    assert value["company_keyset"] == binding["company_keyset"]
    assert value["special_company_keyset"] == list(FOCUS_COMPANIES)
    assert value["tdx_plan_ref"] is None
    for key in ("pit_selection_ref", "pit_generation_manifest_ref", "pit_membership_ref"):
        assert value[key] == binding[key]
    expected_calls = 5 if mode == "dc" else 7
    assert len(client.calls) == expected_calls
    expected_fallback = [] if mode == "dc" else [FOCUS_COMPANIES[0]]
    if expected_fallback:
        plan = json.loads((tmp_path / value["special_tdx_plan_ref"]["path"]).read_bytes())
        assert plan["company_keyset"] == expected_fallback
    else:
        assert value["special_tdx_plan_ref"] is None
    validate_theme_handoff(
        value,
        journal=journal,
        binding=binding,
        claim_ref=value["claim_ref"],
        fallback_company_keyset=[],
        special_fallback_company_keyset=expected_fallback,
    )
    changed = deepcopy(value)
    changed["pit_membership_ref"]["sha256"] = "f" * 64
    with pytest.raises(ContractError, match="PIT_BINDING"):
        validate_theme_handoff(
            changed,
            journal=journal,
            binding=binding,
            claim_ref=value["claim_ref"],
            fallback_company_keyset=[],
            special_fallback_company_keyset=expected_fallback,
        )
    before = inventory(tmp_path)
    clock.hour = 14
    monkeypatch.setattr(
        stage.native, "capture_theme_plan", lambda *a, **k: pytest.fail("replay provider")
    )
    with journal.locked():
        assert publisher.publish_theme_handoff(**args) == ref
    monkeypatch.setattr(journal.storage, "write", lambda *a, **k: pytest.fail("readback writer"))
    replay = reader.read_theme_handoff(**args, handoff_ref=ref)
    assert replay["descriptor"]["schema_version"] == THEME_SOURCE_V2
    assert inventory(tmp_path) == before
    assert len(client.calls) == expected_calls


def test_v2_policy_has_exact_two_company_scope_and_keeps_v1_separate():
    legacy = {
        "schema_version": claims.POLICY_SCHEMA,
        "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
        "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
        "maximum_companies": 100,
    }
    value = {
        **legacy,
        "schema_version": claims.POLICY_SCHEMA_V2,
        "special_company_keyset": list(FOCUS_COMPANIES),
    }
    assert claims.validate_theme_acquisition_policy(value) == value
    for bad in [
        {**value, "special_company_keyset": list(reversed(FOCUS_COMPANIES))},
        {**value, "special_company_keyset": [*FOCUS_COMPANIES, "000001.SZ"]},
        {**value, "schema_version": claims.POLICY_SCHEMA},
    ]:
        with pytest.raises(ContractError):
            claims.validate_theme_acquisition_policy(bad)
