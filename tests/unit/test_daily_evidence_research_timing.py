"""Exact new timing/profile wire contracts; no release or provider admission is mocked as real."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from quant_investor.operations import research_timing as timing
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.execution_recipe import (
    SCHEMA_V4,
    SCHEMA_V5,
    validate_execution_recipe,
    validate_catchup_template,
)
from quant_investor.operations.production_request import HISTORICAL_SCHEMA
from quant_investor.operations.dashboard_serving_contract import POLICY
from quant_investor.operations.theme_acquisition import (
    POLICY_SCHEMA_V2,
    POLICY_SCHEMA_V3,
    validate_theme_acquisition_policy,
    validate_theme_policy_profile,
)
from quant_investor.factors.production_pit import FOCUS_COMPANIES
from test_daily_evidence_production_request import request, recipe


def ref(path):
    return {"path": path, "sha256": "a" * 64}


def timing_recipe(mode=timing.CURRENT):
    req, value = request(), recipe()
    value.update(
        schema_version=SCHEMA_V5,
        dashboard_publication_policy=POLICY,
        corporate_action_template_ref=ref("corporate-template.json"),
        theme_acquisition_ref=ref("acquisition.json") if mode == timing.CURRENT else None,
        research_timing={
            "mode": mode,
            "policy_ref": ref("timing.json"),
            "acquisition_deadline": (
                timing.acquisition_deadline(req["target_trade_date"])
                if mode == timing.CURRENT
                else None
            ),
        },
    )
    if mode == timing.CURRENT:
        value["research_sources"]["as_of"] = None
    else:
        req["schema_version"] = HISTORICAL_SCHEMA
        value["research_sources"]["theme_source_ref"] = ref("retained-theme.json")
    return req, value


@pytest.mark.parametrize("mode", [timing.CURRENT, timing.HISTORICAL])
def test_exact_current_and_historical_recipe_and_template(mode):
    req, value = timing_recipe(mode)
    original = deepcopy(value)
    assert validate_execution_recipe(value, request=req) == original
    assert value == original
    value["previous_completion_ref"] = None
    value["store_preimages"]["store_pointer_ref"] = None
    assert validate_catchup_template(value, request=req) == value


@pytest.mark.parametrize(
    "fault", ["as_of", "deadline", "policy_ref", "mode", "extra", "wrong_request"]
)
def test_current_profile_cannot_predeclare_cutoff_or_extend_deadline(fault):
    req, value = timing_recipe()
    if fault == "as_of":
        value["research_sources"]["as_of"] = "2026-09-04T13:30:00Z"
    elif fault == "deadline":
        value["research_timing"]["acquisition_deadline"] = "2026-09-04T23:59:59Z"
    elif fault == "policy_ref":
        value["research_timing"]["policy_ref"] = None
    elif fault == "mode":
        value["research_timing"]["mode"] = "AUTO"
    elif fault == "extra":
        value["research_timing"]["allow_live"] = True
    else:
        req["schema_version"] = HISTORICAL_SCHEMA
    with pytest.raises(ContractError):
        validate_execution_recipe(value, request=req)


@pytest.mark.parametrize(
    "fault", ["deadline", "acquire", "missing_source", "wrong_cutoff", "wrong_request"]
)
def test_historical_mode_requires_exact_retained_sources(fault):
    req, value = timing_recipe(timing.HISTORICAL)
    if fault == "deadline":
        value["research_timing"]["acquisition_deadline"] = timing.acquisition_deadline(
            req["target_trade_date"]
        )
    elif fault == "acquire":
        value["theme_acquisition_ref"] = ref("acquisition.json")
    elif fault == "missing_source":
        value["research_sources"]["theme_source_ref"] = None
    elif fault == "wrong_cutoff":
        value["research_sources"]["as_of"] = "2026-09-05T13:30:00Z"
    else:
        req["schema_version"] = "cn-daily-production-request.v1"
    with pytest.raises(ContractError):
        validate_execution_recipe(value, request=req)


@pytest.mark.parametrize("mode", [timing.CURRENT, timing.HISTORICAL])
@pytest.mark.parametrize("fault", [None, "deadline", "alignment", "authority", "extra", "timezone"])
def test_timing_policy_is_code_owned_and_non_authorizing(mode, fault):
    value = timing.approved_research_timing_policy(mode)
    if fault == "deadline":
        value["acquisition_deadline_local_time"] = "23:59:58"
    elif fault == "alignment":
        value["alignment_max_seconds"] = True
    elif fault == "authority":
        value["authority"][next(iter(value["authority"]))] = True
    elif fault == "extra":
        value["retry_budget"] = 100
    elif fault == "timezone":
        value["timezone"] = "UTC"
    if fault:
        with pytest.raises(ContractError):
            timing.validate_research_timing_policy(value)
    else:
        assert timing.validate_research_timing_policy(value) == value
        assert not any(value["authority"].values())
    assert timing.acquisition_deadline("20260904") == "2026-09-04T15:59:59Z"


def test_deadline_uses_actual_microseconds_and_legacy_is_unaffected(monkeypatch):
    _, value = timing_recipe()

    class Clock(datetime):
        micros = 0

        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 4, 15, 59, 59, cls.micros, tzinfo=timezone.utc)

    monkeypatch.setattr(timing, "datetime", Clock)
    timing.assert_fresh_acquisition_open(value)
    Clock.micros = 1
    with pytest.raises(ContractError, match="DEADLINE_EXPIRED"):
        timing.assert_fresh_acquisition_open(value)
    timing.assert_fresh_acquisition_open({"schema_version": SCHEMA_V4})
    _, historical = timing_recipe(timing.HISTORICAL)
    timing.assert_fresh_acquisition_open(historical)


def test_theme_profile_cannot_be_reinterpreted_across_versions():
    base = {
        "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
        "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
        "maximum_companies": 100,
        "special_company_keyset": list(FOCUS_COMPANIES),
    }
    for schema, recipe_schema in ((POLICY_SCHEMA_V2, SCHEMA_V4), (POLICY_SCHEMA_V3, SCHEMA_V5)):
        policy = validate_theme_acquisition_policy({**base, "schema_version": schema})
        validate_theme_policy_profile({"schema_version": recipe_schema}, policy)
        with pytest.raises(ContractError, match="PROFILE_MISMATCH"):
            validate_theme_policy_profile(
                {"schema_version": SCHEMA_V4 if recipe_schema == SCHEMA_V5 else SCHEMA_V5}, policy
            )
