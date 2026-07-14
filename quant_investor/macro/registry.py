"""Versioned macro v2 indicator and industry definitions."""

from __future__ import annotations

from dataclasses import dataclass

REGISTRY_VERSION = "cn-macro-indicators.v2.1"
SCORE_MODEL_VERSION = "cn-macro-score.observer.v1"

NATIONAL_DOMAIN_WEIGHTS: dict[str, float] = {
    "growth": 0.25,
    "credit_liquidity": 0.20,
    "inflation": 0.15,
    "policy_fiscal": 0.15,
    "property": 0.10,
    "external": 0.10,
    "market_confirmation": 0.05,
}

INDUSTRY_COMPONENT_WEIGHTS: dict[str, float] = {
    "output": 0.20,
    "orders": 0.20,
    "inventory": 0.10,
    "price_margin": 0.15,
    "profits": 0.15,
    "capacity_utilization": 0.10,
    "capex": 0.05,
    "exports": 0.05,
}

INDUSTRY_CHAINS: tuple[str, ...] = (
    "semiconductor_electronics",
    "computer_communications",
    "power_equipment_new_energy",
    "autos_nev",
    "machinery_automation",
    "chemicals_new_materials",
    "metals_mining",
    "energy_utilities",
    "property_construction",
    "consumer",
    "pharma_medical",
    "export_manufacturing_logistics",
)

FRESHNESS_MAX_AGE_DAYS: dict[str, int] = {
    "daily": 3,
    "weekly": 10,
    "monthly": 50,
    "quarterly": 140,
    "annual": 400,
}

PERIOD_MAX_LAG_DAYS: dict[str, int] = {
    "daily": 7,
    "weekly": 21,
    "monthly": 75,
    "quarterly": 180,
    "annual": 500,
}


@dataclass(frozen=True)
class IndicatorDefinition:
    indicator_id: str
    domain: str
    frequency: str
    unit: str = "%"
    polarity: float = 1.0


NATIONAL_INDICATORS: tuple[IndicatorDefinition, ...] = (
    IndicatorDefinition("cn.gdp_yoy", "growth", "quarterly"),
    IndicatorDefinition("cn.industrial_value_added_yoy", "growth", "monthly"),
    IndicatorDefinition("cn.pmi_manufacturing", "growth", "monthly", unit="index"),
    IndicatorDefinition("cn.retail_sales_yoy", "growth", "monthly"),
    IndicatorDefinition("cn.fixed_asset_investment_yoy", "growth", "monthly"),
    IndicatorDefinition("cn.m1_yoy", "credit_liquidity", "monthly"),
    IndicatorDefinition("cn.m2_yoy", "credit_liquidity", "monthly"),
    IndicatorDefinition("cn.social_financing_flow", "credit_liquidity", "monthly", unit="CNY_100M"),
    IndicatorDefinition("cn.cpi_yoy", "inflation", "monthly", polarity=-0.25),
    IndicatorDefinition("cn.ppi_yoy", "inflation", "monthly"),
    IndicatorDefinition("cn.fiscal_expenditure_yoy", "policy_fiscal", "monthly"),
    IndicatorDefinition("cn.property_investment_yoy", "property", "monthly"),
    IndicatorDefinition("cn.exports_yoy", "external", "monthly"),
    IndicatorDefinition("cn.imports_yoy", "external", "monthly"),
    IndicatorDefinition("market.breadth", "market_confirmation", "daily"),
    IndicatorDefinition(
        "market.volatility_percentile",
        "market_confirmation",
        "daily",
        unit="percentile",
        polarity=-1.0,
    ),
)

_NATIONAL_BY_ID = {item.indicator_id: item for item in NATIONAL_INDICATORS}


def definition_for(indicator_id: str, frequency: str = "") -> IndicatorDefinition | None:
    direct = _NATIONAL_BY_ID.get(indicator_id)
    if direct is not None:
        return direct
    parts = indicator_id.split(".")
    if len(parts) == 3 and parts[0] == "industry" and parts[2] in INDUSTRY_COMPONENT_WEIGHTS:
        return IndicatorDefinition(
            indicator_id,
            parts[2],
            frequency or "monthly",
            unit="",
            polarity=-1.0 if parts[2] == "inventory" else 1.0,
        )
    return None
