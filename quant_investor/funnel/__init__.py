"""Deterministic Funnel — scalable first-pass compression of the full market."""

from quant_investor.funnel.deterministic_funnel import (
    DeterministicFunnel,
    FunnelConfig,
    FunnelOutput,
)
from quant_investor.funnel.theme_candidate_pool import (
    ThemeCandidatePoolBuilder,
    ThemeCandidatePoolOutput,
    ThemeGatePolicy,
    ThemePoolConfig,
)

__all__ = [
    "DeterministicFunnel",
    "FunnelConfig",
    "FunnelOutput",
    "ThemeCandidatePoolBuilder",
    "ThemeCandidatePoolOutput",
    "ThemeGatePolicy",
    "ThemePoolConfig",
]
