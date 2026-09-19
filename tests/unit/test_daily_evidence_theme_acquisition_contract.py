"""Theme source mode and immutable acquisition policy have no permissive defaults."""

from copy import deepcopy
import pytest
from quant_investor.operations.theme_acquisition import (
    validate_theme_acquisition_policy,
    POLICY_SCHEMA,
)
from quant_investor.operations.execution_recipe import (
    validate_execution_recipe,
    validate_execution_recipe_v2,
    SCHEMA_V2,
)
from quant_investor.operations.daily_contract import ContractError
from test_daily_evidence_production_request import request, recipe


@pytest.mark.parametrize("mode", ["pinned", "acquire", "both", "neither"])
def test_v2_requires_exactly_one_theme_source_mode(mode):
    value = recipe()
    value["schema_version"] = SCHEMA_V2
    ref = {"path": "theme.json", "sha256": "a" * 64}
    value["theme_acquisition_ref"] = ref if mode in {"acquire", "both"} else None
    value["research_sources"]["theme_source_ref"] = ref if mode in {"pinned", "both"} else None
    if mode in {"both", "neither"}:
        with pytest.raises(ContractError, match="THEME_MODE"):
            validate_execution_recipe_v2(value, request=request())
    else:
        assert validate_execution_recipe_v2(value, request=request()) == value
    if mode in {"both", "neither"}:
        with pytest.raises(ContractError, match="THEME_MODE"):
            validate_execution_recipe(value, request=request())
    else:
        assert validate_execution_recipe(value, request=request()) == value


@pytest.mark.parametrize("fault", [None, "bool", "larger", "reverse", "url", "fallback"])
def test_policy_cannot_expand_provider_scope_or_budget(fault):
    value = {
        "schema_version": POLICY_SCHEMA,
        "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
        "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
        "maximum_companies": 100,
    }
    if fault == "bool":
        value["maximum_companies"] = True
    elif fault == "larger":
        value["maximum_companies"] = 101
    elif fault == "reverse":
        value["provider_priority"].reverse()
    elif fault == "url":
        value["url"] = "https://example.invalid"
    elif fault == "fallback":
        value["fallback_mode"] = "RETRY_ALL"
    if fault:
        with pytest.raises(ContractError, match="POLICY_INVALID"):
            validate_theme_acquisition_policy(value)
    else:
        before = deepcopy(value)
        validated = validate_theme_acquisition_policy(value)
        validated["provider_priority"].clear()
        assert value == before
