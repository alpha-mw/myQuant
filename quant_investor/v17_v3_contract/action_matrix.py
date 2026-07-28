"""Compatibility exports for the v3 action/state policy."""

from __future__ import annotations

from .policy import (
    ActivationPolicy,
    ActionDecision,
    PolicyContractError as ActionMatrixError,
    activation_policy,
    activation_statuses,
    actions,
    decide_action,
    state_machine,
    states,
    terminal_class,
    terminal_states,
)

ACTIONS = actions()
STATES = states()
TERMINAL_STATES = terminal_states()
ACTIVATION_STATUSES = activation_statuses()

__all__ = [
    "ACTIONS",
    "ACTIVATION_STATUSES",
    "STATES",
    "TERMINAL_STATES",
    "ActivationPolicy",
    "ActionDecision",
    "ActionMatrixError",
    "actions",
    "activation_policy",
    "activation_statuses",
    "decide_action",
    "state_machine",
    "states",
    "terminal_class",
    "terminal_states",
]
