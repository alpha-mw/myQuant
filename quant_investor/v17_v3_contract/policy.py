"""Closed source-role, terminal, action, and authority policy for v3."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from .canonical import CanonicalContractError, validate_semantic_sha
from .identities import (
    IdentityContractError,
    require_casefold_unique,
    require_opaque_id,
    require_registry_token,
    require_sha256,
)
from .resources import PackageResourceError, load_packaged_json

PROTOCOL_VERSION: Final = "myquant.v17.v3"
SOURCE_PHASES: Final = (
    "PRESELECT",
    "BRANCH",
    "FUSION_PROMOTION",
    "QUANT_TIMING_CALIBRATION",
    "FUNDAMENTAL_FORWARD_CALIBRATION",
    "DEEP",
    "PORTFOLIO",
    "SHADOW_CURRENT_PRESELECT",
    "SHADOW_CURRENT_MODEL_PORTFOLIO",
)


class PolicyContractError(ValueError):
    """Raised when a policy resource or decision is incomplete or ambiguous."""

    exit_code = 2


@dataclass(frozen=True)
class RoleRequirement:
    phase: str
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...]
    forbidden_roles: tuple[str, ...]


@dataclass(frozen=True)
class ActionDecision:
    action: str
    state: str
    allowed: bool
    read_only: bool
    targets: tuple[str, ...]
    write_namespaces: tuple[str, ...]
    formal_research_publication_authority: bool
    execution_authority: bool = False
    production_default: bool = False
    broker_authority: bool = False
    order_authority: bool = False
    trade_authority: bool = False


@dataclass(frozen=True)
class ActivationPolicy:
    statuses: tuple[str, ...]
    transitions: tuple[tuple[str, str], ...]


def _resource(name: str) -> dict[str, Any]:
    try:
        payload = load_packaged_json(f"resources/{name}.v1.json")
        validate_semantic_sha(payload)
    except (PackageResourceError, CanonicalContractError) as exc:
        raise PolicyContractError(f"invalid v3 policy resource: {name}") from exc
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise PolicyContractError(f"v3 policy protocol mismatch: {name}")
    return payload


def _require_sorted_unique_strings(value: Any, *, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or any(type(item) is not str for item in value)
        or value != sorted(value)
        or len(value) != len(set(value))
    ):
        raise PolicyContractError(f"{label} must be an ASCII-sorted unique string array")
    try:
        require_casefold_unique(value, label=label)
    except IdentityContractError as exc:
        raise PolicyContractError(str(exc)) from exc
    return tuple(value)


@lru_cache(maxsize=1)
def source_role_matrix() -> Mapping[str, RoleRequirement]:
    payload = _resource("source_role_matrix")
    if payload.get("version") != "myquant.v17.v3.source-role-matrix.v1":
        raise PolicyContractError("source role matrix version mismatch")
    registry = _require_sorted_unique_strings(payload.get("role_registry"), label="role registry")
    raw_registry = _require_sorted_unique_strings(
        payload.get("raw_role_registry"),
        label="raw role registry",
    )
    derived_registry = _require_sorted_unique_strings(
        payload.get("derived_role_registry"),
        label="derived role registry",
    )
    if set(raw_registry) & set(derived_registry) or set(raw_registry) | set(
        derived_registry
    ) != set(registry):
        raise PolicyContractError("raw and derived role registries do not partition roles")
    rows = payload.get("phases")
    if type(rows) is not list or len(rows) != len(SOURCE_PHASES):
        raise PolicyContractError("source role matrix phase table is incomplete")
    result: dict[str, RoleRequirement] = {}
    for index, (expected_phase, row) in enumerate(zip(SOURCE_PHASES, rows, strict=True)):
        if type(row) is not dict or set(row) != {
            "forbidden_roles",
            "optional_roles",
            "phase",
            "required_roles",
        }:
            raise PolicyContractError(f"source phase row {index} shape mismatch")
        if row["phase"] != expected_phase:
            raise PolicyContractError("source phase rows are not in frozen order")
        required = _require_sorted_unique_strings(
            row["required_roles"],
            label=f"{expected_phase} required roles",
        )
        optional = _require_sorted_unique_strings(
            row["optional_roles"],
            label=f"{expected_phase} optional roles",
        )
        forbidden = _require_sorted_unique_strings(
            row["forbidden_roles"],
            label=f"{expected_phase} forbidden roles",
        )
        categories = (*required, *optional, *forbidden)
        if len(categories) != len(set(categories)) or set(categories) != set(registry):
            raise PolicyContractError(
                f"{expected_phase} role truth table does not partition the closed registry"
            )
        result[expected_phase] = RoleRequirement(
            expected_phase,
            required,
            optional,
            forbidden,
        )
    return MappingProxyType(result)


def source_role_registries() -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = _resource("source_role_matrix")
    return (
        _require_sorted_unique_strings(
            payload.get("raw_role_registry"),
            label="raw role registry",
        ),
        _require_sorted_unique_strings(
            payload.get("derived_role_registry"),
            label="derived role registry",
        ),
    )


def role_requirement(phase: Any) -> RoleRequirement:
    try:
        token = require_registry_token(phase, registry=SOURCE_PHASES, label="source phase")
    except IdentityContractError as exc:
        raise PolicyContractError(str(exc)) from exc
    return source_role_matrix()[token]


def validate_source_roles(
    phase: Any,
    available_roles: Sequence[Any],
) -> tuple[str, ...]:
    requirement = role_requirement(phase)
    if isinstance(available_roles, (str, bytes, bytearray)) or not isinstance(
        available_roles, Sequence
    ):
        raise PolicyContractError("available source roles must be an array")
    roles = _require_sorted_unique_strings(list(available_roles), label="available source roles")
    known = set(
        (*requirement.required_roles, *requirement.optional_roles, *requirement.forbidden_roles)
    )
    if not set(roles).issubset(known):
        raise PolicyContractError("available source roles contain an unregistered role")
    missing = set(requirement.required_roles) - set(roles)
    forbidden = set(requirement.forbidden_roles) & set(roles)
    if missing:
        raise PolicyContractError(f"required source roles are unavailable: {sorted(missing)}")
    if forbidden:
        raise PolicyContractError(f"forbidden source roles are present: {sorted(forbidden)}")
    return roles


@lru_cache(maxsize=1)
def state_machine() -> dict[str, Any]:
    payload = _resource("state_machine")
    states = _require_sorted_unique_strings(payload.get("states"), label="states")
    terminal_classes = payload.get("terminal_classes")
    if type(terminal_classes) is not dict or set(terminal_classes) != {
        "CONTROL",
        "FORMAL",
        "HARD_STOP",
        "SHADOW",
    }:
        raise PolicyContractError("terminal class table is incomplete")
    terminals: list[str] = []
    for class_name in ("CONTROL", "FORMAL", "HARD_STOP", "SHADOW"):
        values = _require_sorted_unique_strings(
            terminal_classes[class_name],
            label=f"{class_name} terminals",
        )
        terminals.extend(values)
    if len(terminals) != len(set(terminals)) or not set(terminals).issubset(states):
        raise PolicyContractError("terminal classes overlap or contain unknown states")
    return payload


def states() -> tuple[str, ...]:
    return tuple(state_machine()["states"])


def terminal_states() -> tuple[str, ...]:
    machine = state_machine()
    values = [
        state
        for class_name in ("CONTROL", "FORMAL", "HARD_STOP", "SHADOW")
        for state in machine["terminal_classes"][class_name]
    ]
    return tuple(sorted(values))


def terminal_class(state: Any) -> str | None:
    machine = state_machine()
    try:
        token = require_registry_token(state, registry=machine["states"], label="state")
    except IdentityContractError as exc:
        raise PolicyContractError(str(exc)) from exc
    for class_name, values in machine["terminal_classes"].items():
        if token in values:
            return class_name
    return None


@lru_cache(maxsize=1)
def activation_policy() -> ActivationPolicy:
    payload = _resource("authority_policy")
    if payload.get("version") != "myquant.v17.v3.authority-policy.v1":
        raise PolicyContractError("authority policy version mismatch")
    statuses = _require_sorted_unique_strings(
        payload.get("activation_statuses"),
        label="activation statuses",
    )
    if statuses != ("ACTIVATION_REJECTED", "ACTIVE", "REVOKED"):
        raise PolicyContractError("activation status registry mismatch")
    raw_transitions = payload.get("activation_transitions")
    if type(raw_transitions) is not list:
        raise PolicyContractError("activation transition table is missing")
    transitions: list[tuple[str, str]] = []
    previous: tuple[str, str] | None = None
    allowed_from = {"ABSENT", *statuses}
    for index, row in enumerate(raw_transitions):
        if (
            type(row) is not dict
            or set(row) != {"from", "to"}
            or type(row["from"]) is not str
            or type(row["to"]) is not str
        ):
            raise PolicyContractError(f"activation transition row {index} shape mismatch")
        transition = (row["from"], row["to"])
        if (
            transition[0] not in allowed_from
            or transition[1] not in statuses
            or (previous is not None and transition <= previous)
        ):
            raise PolicyContractError("activation transition table is not closed and sorted")
        transitions.append(transition)
        previous = transition
    expected = (
        ("ABSENT", "ACTIVATION_REJECTED"),
        ("ABSENT", "ACTIVE"),
        ("ACTIVE", "REVOKED"),
    )
    if tuple(transitions) != expected:
        raise PolicyContractError("activation transition topology mismatch")
    return ActivationPolicy(statuses=statuses, transitions=tuple(transitions))


def activation_statuses() -> tuple[str, ...]:
    return activation_policy().statuses


@lru_cache(maxsize=1)
def _action_rows() -> Mapping[str, dict[str, Any]]:
    payload = _resource("action_matrix")
    rows = payload.get("actions")
    if type(rows) is not list or not rows:
        raise PolicyContractError("action matrix has no actions")
    state_registry = frozenset(states())
    activation_registry = frozenset(activation_statuses())
    activation_only_registry = activation_registry - state_registry
    activation_actions = frozenset({"ACTIVATE_FORMAL_RESEARCH", "REVOKE_FORMAL_RESEARCH"})
    result: dict[str, dict[str, Any]] = {}
    previous: str | None = None
    forbidden_action_fragments = ("BROKER", "ORDER", "TRADE", "EXECUTION")
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "action",
            "allowed_from",
            "formal_research_publication_authority",
            "read_only",
            "targets",
            "write_namespaces",
        }:
            raise PolicyContractError(f"action row {index} shape mismatch")
        action = row["action"]
        if type(action) is not str or any(
            fragment in action for fragment in forbidden_action_fragments
        ):
            raise PolicyContractError(f"action row {index} names a forbidden capability")
        if previous is not None and action <= previous:
            raise PolicyContractError("actions are not in ASCII order")
        previous = action
        if action in result:
            raise PolicyContractError(f"duplicate action: {action}")
        allowed_from = _require_sorted_unique_strings(
            row["allowed_from"],
            label=f"{action} allowed states",
        )
        targets = _require_sorted_unique_strings(row["targets"], label=f"{action} targets")
        allowed_registry = (
            activation_registry if action == "REVOKE_FORMAL_RESEARCH" else state_registry
        )
        target_registry = (
            state_registry | activation_registry if action in activation_actions else state_registry
        )
        if not set(allowed_from).issubset(allowed_registry):
            raise PolicyContractError(f"{action} references an unknown input state or status")
        if not set(targets).issubset(target_registry):
            raise PolicyContractError(f"{action} references an unknown target state or status")
        if action not in activation_actions and set(targets) & activation_only_registry:
            raise PolicyContractError(f"{action} cannot target the activation-status axis")
        writes = _require_sorted_unique_strings(
            row["write_namespaces"],
            label=f"{action} write namespaces",
        )
        if (
            type(row["read_only"]) is not bool
            or type(row["formal_research_publication_authority"]) is not bool
        ):
            raise PolicyContractError(f"{action} authority flags are not exact booleans")
        if row["read_only"] and writes:
            raise PolicyContractError(f"{action} is read-only but declares writes")
        result[action] = dict(row)
    return MappingProxyType(result)


def actions() -> tuple[str, ...]:
    return tuple(_action_rows())


def decide_action(*, action: Any, state: Any) -> ActionDecision:
    try:
        action_token = require_registry_token(action, registry=actions(), label="action")
        input_registry = (
            activation_statuses() if action_token == "REVOKE_FORMAL_RESEARCH" else states()
        )
        state_token = require_registry_token(
            state,
            registry=input_registry,
            label="state or activation status",
        )
    except IdentityContractError as exc:
        raise PolicyContractError(str(exc)) from exc
    row = _action_rows()[action_token]
    return ActionDecision(
        action=action_token,
        state=state_token,
        allowed=state_token in row["allowed_from"],
        read_only=row["read_only"],
        targets=tuple(row["targets"]),
        write_namespaces=tuple(row["write_namespaces"]),
        formal_research_publication_authority=(
            row["formal_research_publication_authority"]
            if state_token in row["allowed_from"]
            else False
        ),
    )


def validate_authority(
    authority: Any,
    *,
    formal_research_publication_authority: bool,
) -> dict[str, bool]:
    expected = {
        "broker_authority": False,
        "execution_authority": False,
        "formal_research_publication_authority": formal_research_publication_authority,
        "order_authority": False,
        "production_default": False,
        "trade_authority": False,
    }
    if type(authority) is not dict or authority != expected:
        raise PolicyContractError("authority axes do not match the closed v3 truth table")
    return dict(expected)


def validate_factor_inventory_isolation() -> dict[str, tuple[str, ...]]:
    """Prove preselector and Quant definition/family/lineage sets are disjoint."""

    preselector = _resource("preselector_policy").get("factor_inventory")
    quant = _resource("quant_branch_policy").get("factor_inventory")
    if type(preselector) is not list or type(quant) is not list:
        raise PolicyContractError("factor inventory is missing")

    def axes(rows: list[Any], *, label: str) -> dict[str, tuple[str, ...]]:
        expected = {
            "definition_sha256",
            "expression",
            "factor_id",
            "family_id",
            "implementation",
            "lineage_id",
        }
        if label == "preselector":
            expected |= {"lookback_open_days", "weight"}
        normalized: dict[str, list[str]] = {
            "definition_sha256": [],
            "factor_id": [],
            "family_id": [],
            "lineage_id": [],
        }
        previous: str | None = None
        for index, row in enumerate(rows):
            if type(row) is not dict or set(row) != expected:
                raise PolicyContractError(f"{label} factor row {index} shape mismatch")
            if (
                type(row["expression"]) is not str
                or not row["expression"]
                or type(row["implementation"]) is not str
                or not row["implementation"]
                or (
                    label == "preselector"
                    and (
                        type(row["lookback_open_days"]) is not int
                        or row["lookback_open_days"] <= 0
                        or type(row["weight"]) is not str
                    )
                )
            ):
                raise PolicyContractError(f"{label} factor definition metadata is invalid")
            factor_id = row["factor_id"]
            if type(factor_id) is not str or (previous is not None and factor_id <= previous):
                raise PolicyContractError(f"{label} factor inventory is not canonically ordered")
            previous = factor_id
            for axis in normalized:
                value = row[axis]
                if type(value) is not str:
                    raise PolicyContractError(f"{label} factor {axis} is invalid")
                try:
                    if axis == "definition_sha256":
                        require_sha256(value, label=f"{label} factor definition SHA")
                    else:
                        require_opaque_id(value, label=f"{label} factor {axis}")
                except IdentityContractError as exc:
                    raise PolicyContractError(str(exc)) from exc
                normalized[axis].append(value)
        for axis, values in normalized.items():
            if len(values) != len(set(values)):
                raise PolicyContractError(f"{label} factor {axis} is not unique")
        return {axis: tuple(values) for axis, values in normalized.items()}

    left = axes(preselector, label="preselector")
    right = axes(quant, label="quant")
    for axis in ("definition_sha256", "family_id", "lineage_id"):
        overlap = set(left[axis]) & set(right[axis])
        if overlap:
            raise PolicyContractError(
                f"preselector and Quant factor {axis} overlap: {sorted(overlap)}"
            )
    return {
        "preselector_definition_sha256s": left["definition_sha256"],
        "preselector_family_ids": left["family_id"],
        "preselector_lineage_ids": left["lineage_id"],
        "quant_definition_sha256s": right["definition_sha256"],
        "quant_family_ids": right["family_id"],
        "quant_lineage_ids": right["lineage_id"],
    }


__all__ = [
    "ActivationPolicy",
    "ActionDecision",
    "PolicyContractError",
    "PROTOCOL_VERSION",
    "RoleRequirement",
    "SOURCE_PHASES",
    "activation_policy",
    "activation_statuses",
    "actions",
    "decide_action",
    "role_requirement",
    "source_role_matrix",
    "source_role_registries",
    "state_machine",
    "states",
    "terminal_class",
    "terminal_states",
    "validate_authority",
    "validate_factor_inventory_isolation",
    "validate_source_roles",
]
