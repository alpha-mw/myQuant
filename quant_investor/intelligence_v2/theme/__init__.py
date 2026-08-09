"""Public pure-library surface for I3 Theme Intelligence."""

from .contracts import (
    build_theme_component_policy,
    build_theme_lifecycle_policy,
    build_theme_membership_catalog,
    build_theme_registry,
    build_theme_risk_policy,
    validate_theme_component_policy,
    validate_theme_lifecycle_policy,
    validate_theme_membership_catalog,
    validate_theme_registry,
    validate_theme_risk_policy,
)
from .engine import (
    build_theme_component_receipt,
    build_theme_risk_receipt,
    resolve_theme_exposure,
    validate_theme_component_receipt,
    validate_theme_exposure_receipt,
    validate_theme_risk_receipt,
)
from .models import ThemeContractError

__all__ = [
    "ThemeContractError",
    "build_theme_component_policy",
    "build_theme_component_receipt",
    "build_theme_lifecycle_policy",
    "build_theme_membership_catalog",
    "build_theme_registry",
    "build_theme_risk_policy",
    "build_theme_risk_receipt",
    "resolve_theme_exposure",
    "validate_theme_component_policy",
    "validate_theme_component_receipt",
    "validate_theme_exposure_receipt",
    "validate_theme_lifecycle_policy",
    "validate_theme_membership_catalog",
    "validate_theme_registry",
    "validate_theme_risk_policy",
    "validate_theme_risk_receipt",
]
