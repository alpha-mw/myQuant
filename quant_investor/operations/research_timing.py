"""Code-owned acquisition bounds; never grants provider or investment authority."""

from datetime import datetime, time, timezone

from quant_investor.intelligence.fundamental_time import SHANGHAI, session_date
from .daily_contract import ContractError, utc_stamp, validate_ref
from .daily_journal import FALSE_AUTHORITY

POLICY_SCHEMA = "cn-daily-research-timing-policy.v1"
CURRENT = "CURRENT_POST_ACQUISITION"
HISTORICAL = "HISTORICAL_RETAINED"


def approved_research_timing_policy(mode):
    if type(mode) is not str or mode not in {CURRENT, HISTORICAL}:
        raise ContractError("RESEARCH_TIMING_MODE_INVALID")
    return {
        "schema_version": POLICY_SCHEMA,
        "mode": mode,
        "timezone": "Asia/Shanghai",
        "acquisition_deadline_local_time": "23:59:59" if mode == CURRENT else None,
        "cutoff_rule": (
            "AFTER_VERIFIED_SOURCE_CUSTODY" if mode == CURRENT else "EXACT_RETAINED_AS_OF"
        ),
        "alignment_max_seconds": 1,
        "authority": dict(FALSE_AUTHORITY),
    }


def validate_research_timing_policy(value):
    from quant_investor.contracts import canonical_json_bytes

    if type(value) is not dict:
        raise ContractError("RESEARCH_TIMING_POLICY_INVALID")
    expected = approved_research_timing_policy(value.get("mode"))
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ContractError("RESEARCH_TIMING_POLICY_INVALID")
    return expected


def acquisition_deadline(trade_date):
    day = session_date(trade_date)
    return (
        datetime.combine(day, time(23, 59, 59), tzinfo=SHANGHAI)
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def validate_recipe_timing(recipe, *, historical_request=None):
    timing = recipe["research_timing"]
    if type(timing) is not dict or set(timing) != {"mode", "policy_ref", "acquisition_deadline"}:
        raise ContractError("RESEARCH_TIMING_FIELDS_INVALID")
    mode = timing["mode"]
    approved_research_timing_policy(mode)
    validate_ref(timing["policy_ref"])
    sources = recipe["research_sources"]
    if type(sources) is not dict or "as_of" not in sources:
        raise ContractError("RESEARCH_TIMING_SOURCES_INVALID")
    if historical_request is not None and (mode == HISTORICAL) != historical_request:
        raise ContractError("RESEARCH_TIMING_REQUEST_MODE_MISMATCH")
    if mode == CURRENT:
        if sources["as_of"] is not None or timing["acquisition_deadline"] != acquisition_deadline(
            recipe["target_trade_date"]
        ):
            raise ContractError("RESEARCH_TIMING_CURRENT_BOUND_INVALID")
    elif (
        timing["acquisition_deadline"] is not None
        or utc_stamp(sources["as_of"]).strftime("%Y%m%d") != recipe["target_trade_date"]
        or utc_stamp(sources["as_of"]).astimezone(SHANGHAI).strftime("%Y%m%d")
        != recipe["target_trade_date"]
        or recipe["theme_acquisition_ref"] is not None
        or sources.get("theme_source_ref") is None
    ):
        raise ContractError("RESEARCH_TIMING_HISTORICAL_BOUND_INVALID")
    return timing


def recipe_v4_validation_view(recipe):
    """Only reuse legacy non-temporal shape validation; never materialize this view."""
    timing = validate_recipe_timing(recipe)
    value = {
        key: field
        for key, field in recipe.items()
        if key not in {"research_timing", "corporate_action_template_ref"}
    }
    value["schema_version"] = "cn-daily-execute-recipe.v4"
    value["corporate_action_context_ref"] = recipe["corporate_action_template_ref"]
    value["research_sources"] = {
        **recipe["research_sources"],
        "as_of": (
            timing["acquisition_deadline"]
            if timing["mode"] == CURRENT
            else recipe["research_sources"]["as_of"]
        ),
    }
    return value


def theme_acquisition_bound(binding):
    """A v3 bound is an unpublished source-validation ceiling, not a Decision cutoff."""
    from .theme_acquisition import POLICY_SCHEMA_V3

    if binding.get("policy_schema") == POLICY_SCHEMA_V3:
        return binding["acquisition_deadline"]
    return binding["research_cutoff"]


def assert_fresh_acquisition_open(recipe):
    from .execution_recipe import SCHEMA_V5, SCHEMA_V6

    if recipe.get("schema_version") not in {SCHEMA_V5, SCHEMA_V6}:
        return
    timing = validate_recipe_timing(recipe)
    if timing["mode"] == CURRENT and datetime.now(timezone.utc) > utc_stamp(
        timing["acquisition_deadline"]
    ):
        raise ContractError("RESEARCH_ACQUISITION_DEADLINE_EXPIRED")
