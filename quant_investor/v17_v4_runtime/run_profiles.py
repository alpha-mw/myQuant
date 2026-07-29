"""Closed run-profile definitions for V17 v4 forward evidence.

The profiles in this module are research orchestration profiles only.  They do
not select the package default, authorize a provider, or grant formal,
canary, broker, order, execution, or trade authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping


class RunProfile(str, Enum):
    EXPLORE = "EXPLORE"
    FORWARD_EVIDENCE = "FORWARD_EVIDENCE"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"


class LifecycleLabel(str, Enum):
    SOURCE_SNAPSHOT = "SOURCE_SNAPSHOT"
    QUANT_COMPLETE = "QUANT_COMPLETE"
    FUNDAMENTAL_PARTIAL_ALLOWED = "FUNDAMENTAL_PARTIAL_ALLOWED"
    FUSION_COMPLETE = "FUSION_COMPLETE"
    DEEP_OPTIONAL = "DEEP_OPTIONAL"
    SHADOW_OBSERVATION_CREATED = "SHADOW_OBSERVATION_CREATED"
    FORWARD_LABEL_PENDING = "FORWARD_LABEL_PENDING"


STAGE_ORDER: Final = (
    "source",
    "allocation",
    "quant",
    "factor_universe_observation",
    "fundamental",
    "fusion",
    "deep",
    "holdings",
    "strategy_pool_observation",
    "final",
)

STAGE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "final_ref": "final",
    }
)

STAGE_LIFECYCLE_LABELS: Final[Mapping[str, LifecycleLabel]] = MappingProxyType(
    {
        "source": LifecycleLabel.SOURCE_SNAPSHOT,
        "quant": LifecycleLabel.QUANT_COMPLETE,
        "fundamental": LifecycleLabel.FUNDAMENTAL_PARTIAL_ALLOWED,
        "fusion": LifecycleLabel.FUSION_COMPLETE,
        "deep": LifecycleLabel.DEEP_OPTIONAL,
        "factor_universe_observation": (LifecycleLabel.SHADOW_OBSERVATION_CREATED),
        "strategy_pool_observation": (LifecycleLabel.SHADOW_OBSERVATION_CREATED),
        "final": LifecycleLabel.FORWARD_LABEL_PENDING,
    }
)


@dataclass(frozen=True)
class ProfileDefinition:
    profile: RunProfile
    required_stages: frozenset[str]
    optional_stages: frozenset[str]
    delegates_to_strict_v3: bool = False

    @property
    def stages(self) -> tuple[str, ...]:
        allowed = self.required_stages | self.optional_stages
        return tuple(stage for stage in STAGE_ORDER if stage in allowed)

    def is_required(self, stage: str) -> bool:
        return stage in self.required_stages


PROFILE_DEFINITIONS: Final[Mapping[RunProfile, ProfileDefinition]] = MappingProxyType(
    {
        RunProfile.EXPLORE: ProfileDefinition(
            profile=RunProfile.EXPLORE,
            required_stages=frozenset(
                {
                    "source",
                    "allocation",
                    "quant",
                    "factor_universe_observation",
                }
            ),
            optional_stages=frozenset(
                {
                    "fusion",
                    "strategy_pool_observation",
                    "fundamental",
                    "deep",
                    "holdings",
                }
            ),
        ),
        RunProfile.FORWARD_EVIDENCE: ProfileDefinition(
            profile=RunProfile.FORWARD_EVIDENCE,
            required_stages=frozenset(
                {
                    "source",
                    "allocation",
                    "quant",
                    "factor_universe_observation",
                    "fusion",
                    "strategy_pool_observation",
                    "final",
                }
            ),
            optional_stages=frozenset(
                {
                    "fundamental",
                    "deep",
                    "holdings",
                }
            ),
        ),
        RunProfile.RELEASE_CANDIDATE: ProfileDefinition(
            profile=RunProfile.RELEASE_CANDIDATE,
            required_stages=frozenset(),
            optional_stages=frozenset(),
            delegates_to_strict_v3=True,
        ),
    }
)


def normalize_profile(value: object) -> RunProfile:
    if isinstance(value, RunProfile):
        return value
    if type(value) is not str:
        raise ValueError("run profile must be text")
    try:
        return RunProfile(value)
    except ValueError as exc:
        raise ValueError(f"unsupported run profile: {value!r}") from exc


def normalize_stage(value: object) -> str:
    if type(value) is not str:
        raise ValueError("stage name must be text")
    stage = STAGE_ALIASES.get(value, value)
    if stage not in STAGE_ORDER:
        raise ValueError(f"unsupported forward stage: {value!r}")
    return stage


def profile_definition(value: object) -> ProfileDefinition:
    return PROFILE_DEFINITIONS[normalize_profile(value)]


__all__ = [
    "LifecycleLabel",
    "PROFILE_DEFINITIONS",
    "ProfileDefinition",
    "RunProfile",
    "STAGE_ALIASES",
    "STAGE_LIFECYCLE_LABELS",
    "STAGE_ORDER",
    "normalize_profile",
    "normalize_stage",
    "profile_definition",
]
