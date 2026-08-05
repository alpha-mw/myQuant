"""Migrate `mined-factor-registry.v1` records onto the v4 production schema.

`assess_factor_record_v4` needs `family`, `slot`, `weight`, `maturity`,
`bh_q_value`, `fdr_method`, `runtime_contract`, `evidence` and fresh `health`.
The registry carries none of them: every record has `family=None`, `slot=None`
and, for the candidates, `weight=0.0`, so no record has ever entered the v4
assessment and `production_factor_count` has always been 0. That is a schema
migration gap, not a statistics problem, which is why no threshold moves here.

This module owns the deterministic half -- which five factors, in which family,
at which weight, with which pinned params. Maturity, BH and fresh health are
measured separately and passed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_investor.factors.governance_protocol_v4 import (
    MAX_FACTOR_ABS_WEIGHT,
    MAX_FAMILY_ABS_WEIGHT,
    MIN_NEW_RISK_FACTOR_COUNT,
)
from quant_investor.factors.registry_v4_migration import (
    CATEGORY_TO_FAMILY,
    RESIDUAL_BASELINE_PIN,
    RegistryV4MigrationError,
    assign_production_weights,
    migrate_registry_to_v4_records,
    select_production_set,
)

REGISTRY = Path("quant_investor/factor_registry/mined_factors.json")


def _records() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["factors"]


# --- selection --------------------------------------------------------------


def test_selects_exactly_five_factors_in_five_distinct_families():
    selected = select_production_set(_records())

    assert len(selected) == MIN_NEW_RISK_FACTOR_COUNT
    families = [item["family"] for item in selected]
    assert len(set(families)) == MIN_NEW_RISK_FACTOR_COUNT


def test_selection_is_the_documented_dedup_winners():
    """Highest recorded ICIR within each family, ties broken by name."""

    selected = {item["name"] for item in select_production_set(_records())}

    assert selected == {
        "pv_low_dollar_volume_5d",
        "pv_volume_stability_90d",
        "pv_blend_volstab19x2_mom90_amihud5_w80",
        "fund_fin_net_profit_yoy",
        "formula_mom120_np_yoy_resid_w30",
    }


def test_selection_is_deterministic_under_input_reordering():
    forward = [item["name"] for item in select_production_set(_records())]
    reversed_ = [item["name"] for item in select_production_set(list(reversed(_records())))]

    assert forward == reversed_


def test_only_production_level_records_are_eligible():
    """research_candidate records carry no gate evidence at all."""

    selected = select_production_set(_records())

    for item in selected:
        assert item["state"] in {"production_factor", "production_candidate"}


def test_every_selected_factor_passed_all_eight_gates():
    for item in select_production_set(_records()):
        gates = {int(g["gate_id"]): bool(g["passed"]) for g in item["gate_results"]}
        assert sorted(gates) == list(range(1, 9))
        assert all(gates.values())


def test_selection_fails_closed_when_a_family_is_missing():
    thinned = [r for r in _records() if r["category"] != "growth"]

    with pytest.raises(RegistryV4MigrationError, match="famil"):
        select_production_set(thinned)


def test_every_category_in_the_selected_set_has_an_explicit_family():
    """The map is written out on purpose; inference would be silent drift."""

    for item in select_production_set(_records()):
        assert item["category"] in CATEGORY_TO_FAMILY


# --- weights ----------------------------------------------------------------


def test_weights_are_equal_and_sum_to_one():
    selected = select_production_set(_records())

    weighted = assign_production_weights(selected)

    assert sum(item["weight"] for item in weighted) == pytest.approx(1.0)
    assert {round(item["weight"], 10) for item in weighted} == {0.2}


def test_weights_respect_the_per_factor_cap():
    for item in assign_production_weights(select_production_set(_records())):
        assert abs(item["weight"]) <= MAX_FACTOR_ABS_WEIGHT + 1e-12


def test_weights_respect_the_per_family_cap():
    weighted = assign_production_weights(select_production_set(_records()))

    by_family: dict[str, float] = {}
    for item in weighted:
        by_family[item["family"]] = by_family.get(item["family"], 0.0) + abs(item["weight"])

    assert max(by_family.values()) <= MAX_FAMILY_ABS_WEIGHT + 1e-12


def test_two_factors_in_one_family_would_breach_the_family_cap():
    """Why the set is one-per-family rather than five-from-fewer-families."""

    assert 2 * (1.0 / MIN_NEW_RISK_FACTOR_COUNT) > MAX_FAMILY_ABS_WEIGHT


# --- record construction ----------------------------------------------------


def test_migrated_records_carry_the_fields_v4_requires():
    migrated = migrate_registry_to_v4_records(_records())

    for record in migrated:
        assert record["family"]
        assert record["slot"]
        assert record["state"] == "production_factor"
        assert record["weight"] > 0.0
        assert record["fdr_method"] == "benjamini_hochberg_by_family"


def test_slots_are_unique_and_non_empty():
    slots = [record["slot"] for record in migrate_registry_to_v4_records(_records())]

    assert all(slots)
    assert len(set(slots)) == len(slots)


def test_the_formulaic_record_is_rewritten_to_the_pinned_form():
    """Otherwise production and fresh health both refuse to run it."""

    from quant_investor.factors.governance import FactorRecord
    from quant_investor.factors.runtime import production_set_dependent_primitives

    migrated = {r["name"]: r for r in migrate_registry_to_v4_records(_records())}
    formulaic = migrated["formula_mom120_np_yoy_resid_w30"]
    params = formulaic["metadata"]["params"]

    assert params["right"] == "fin_net_profit_yoy"
    assert params["residualize_right_against"] == RESIDUAL_BASELINE_PIN
    assert production_set_dependent_primitives(FactorRecord.from_dict(formulaic)) == ()


def test_the_pin_is_the_baseline_that_was_live_at_gate_time():
    """Pinning to the gate-time baseline is what keeps the evidence valid."""

    from quant_investor.factors.residual_baseline import validate_residual_baseline

    normalized = validate_residual_baseline(RESIDUAL_BASELINE_PIN)

    assert [item["name"] for item in normalized["factors"]] == ["pv_low_dollar_volume_5d"]
    assert normalized["factors"][0]["weight"] == pytest.approx(0.05)


def test_the_blend_weight_is_preserved_from_the_source_record():
    migrated = {r["name"]: r for r in migrate_registry_to_v4_records(_records())}
    params = migrated["formula_mom120_np_yoy_resid_w30"]["metadata"]["params"]

    assert params["left_weight"] == pytest.approx(0.30)
    assert params["left"] == "momentum_120"


def test_non_formulaic_records_keep_their_params_untouched():
    source = {r["name"]: r for r in _records()}
    migrated = {r["name"]: r for r in migrate_registry_to_v4_records(_records())}

    for name in ("pv_low_dollar_volume_5d", "fund_fin_net_profit_yoy"):
        assert migrated[name]["metadata"].get("params") == source[name]["metadata"].get(
            "params"
        )


def test_every_migrated_record_is_production_executable():
    from quant_investor.factors.runtime import is_production_allowlisted_implementation

    for record in migrate_registry_to_v4_records(_records()):
        assert is_production_allowlisted_implementation(record["implementation"])


def test_migration_does_not_mutate_the_source_records():
    source = _records()
    before = json.dumps(source, sort_keys=True)

    migrate_registry_to_v4_records(source)

    assert json.dumps(source, sort_keys=True) == before


def test_gate_results_survive_the_migration_unchanged():
    source = {r["name"]: r for r in _records()}
    migrated = {r["name"]: r for r in migrate_registry_to_v4_records(_records())}

    for name, record in migrated.items():
        assert record["gate_results"] == source[name]["gate_results"]
