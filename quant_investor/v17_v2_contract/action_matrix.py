"""Pure v17 protocol-v2 action authorization matrix.

The matrix is deliberately isolated from :mod:`quant_investor.v17`.  It is a
pre-import contract: callers classify the discovered protocol envelope, ask
for exactly one decision, and only then decide whether a runtime may be
imported.  This module performs no writes, locking, directory creation, or
runtime dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Final, Mapping

ACTION_MATRIX_SCHEMA: Final = "myquant.v17.v2.action-matrix.v1"
STATE_MACHINE_SCHEMA: Final = "myquant.v17.v2.state-machine.v1"
PROTOCOL_VERSION: Final = "myquant.v17.v2"
PACKAGE_VERSION: Final = "17.0.0"

VERSIONS: Final = ("ABSENT", "v1", "v2", "unknown", "malformed")
ACTIONS: Final = (
    "SOURCE_MAINTAIN",
    "RISK_POLICY_SEAL",
    "SHADOW_PREPARE",
    "SHADOW_RECEIVE",
    "SHADOW_FINALIZE",
    "READ_STATUS",
    "READ_ARTIFACT",
    "REPAIR_LATEST",
)
STATES: Final = (
    "MISSING",
    "UNKNOWN",
    "MALFORMED",
    "PREPARED",
    "DETERMINISTIC_COMPLETE",
    "DEEP_REQUEST_READY",
    "DEEP_RESPONSE_RECEIVED",
    "PORTFOLIO_COMPLETE",
    "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
    "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
    "SHADOW_PORTFOLIO_INFEASIBLE",
    "HARD_STOP_SNAPSHOT_DRIFT",
    "HARD_STOP_INVALID_EVIDENCE",
)
CHECKPOINTS: Final = ("PRE_IMPORT", "ACCEPTED", "INITIALIZED")

NONTERMINAL_STATES: Final = STATES[3:8]
BUSINESS_TERMINAL_STATES: Final = STATES[8:11]
HARD_STOP_TERMINAL_STATES: Final = STATES[11:13]
TERMINAL_STATES: Final = BUSINESS_TERMINAL_STATES + HARD_STOP_TERMINAL_STATES

WRITE_NAMESPACES: Final = (
    "SOURCE_OBJECTS",
    "SOURCE_MANIFESTS",
    "SOURCE_LOCATORS",
    "RUN_ROOT",
    "RUN_LEDGER",
    "RUN_LOCK",
    "RUN_EVENTS",
    "RUN_RECEIPTS",
    "MODELS",
    "OUTCOMES",
    "LATEST",
    "LATEST_LOCK",
)
RETRY_CAS_MODES: Final = (
    "NONE",
    "EMPTY",
    "ABSENT_TARGET",
    "CURRENT_MANIFEST_SHA_AND_PLAN",
    "EXACT_BYTES",
    "CURRENT_LEDGER_SHA",
    "EXACT_REPLAY",
    "CURRENT_LEDGER_AND_LATEST_SHA",
)
LATEST_EFFECTS: Final = ("UNCHANGED", "PUBLISHED", "REPAIRED")

# Filled with the byte hashes of compact, sorted-key UTF-8 resources.  The
# constants are intentionally local so importing this module does not import
# any v1 resource loader.
ACTION_MATRIX_RESOURCE_SHA256: Final = (
    "f342820aadd34005e718552e82b162b8efaee2e1046c184609d8080cf13e6434"
)
STATE_MACHINE_RESOURCE_SHA256: Final = (
    "6f21ea8b9b13ec4242730d65d0a7d48e1af7033713342e96b43bf3a80d60aee8"
)

_RESOURCE_DIRECTORY = Path(__file__).with_name("resources")
_ACTION_MATRIX_PATH = _RESOURCE_DIRECTORY / "action_matrix.v1.json"
_STATE_MACHINE_PATH = _RESOURCE_DIRECTORY / "state_machine.v1.json"
_DECISION_KEYS = frozenset(
    {
        "allowed",
        "allowed_write_namespaces",
        "outcomes",
        "read_only",
        "reason",
        "retry_cas",
    }
)
_OUTCOME_KEYS = frozenset(
    {
        "business_acceptance",
        "command_commit",
        "exit_code",
        "latest_effect",
        "target_state",
        "terminal",
    }
)
_RULE_KEYS = frozenset({"decision", "id", "match"})
_MATCH_KEYS = frozenset({"actions", "checkpoints", "states", "versions"})
_FAILURE_KEYS = frozenset(
    {
        "allowed_write_namespaces",
        "business_acceptance",
        "command_commit",
        "exit_code",
        "latest_effect",
        "receipt_effect",
        "required_next_action",
    }
)
_EXPECTED_FAILURE_SEMANTICS: Final = {
    "CAS_CONFLICT": {
        "allowed_write_namespaces": [],
        "business_acceptance": False,
        "command_commit": False,
        "exit_code": 2,
        "latest_effect": "UNCHANGED",
        "receipt_effect": "NONE",
        "required_next_action": "READ_STATUS",
    },
    "POST_INITIALIZED_UNCOMMITTED": {
        "allowed_write_namespaces": ["RUN_RECEIPTS"],
        "business_acceptance": False,
        "command_commit": False,
        "exit_code": 2,
        "latest_effect": "UNCHANGED",
        "receipt_effect": "UNPUBLISHED_NOT_COMMITTED",
        "required_next_action": "READ_STATUS",
    },
    "PRE_IMPORT_REJECTION": {
        "allowed_write_namespaces": [],
        "business_acceptance": False,
        "command_commit": False,
        "exit_code": 2,
        "latest_effect": "UNCHANGED",
        "receipt_effect": "NONE",
        "required_next_action": None,
    },
    "PRE_INITIALIZED_VALIDATION_FAILURE": {
        "allowed_write_namespaces": [],
        "business_acceptance": False,
        "command_commit": False,
        "exit_code": 2,
        "latest_effect": "UNCHANGED",
        "receipt_effect": "NONE",
        "required_next_action": None,
    },
    "TERMINAL_LATEST_PUBLICATION_FAILURE": {
        "allowed_write_namespaces": ["RUN_RECEIPTS"],
        "business_acceptance": False,
        "command_commit": True,
        "exit_code": 2,
        "latest_effect": "UNCHANGED",
        "receipt_effect": "TERMINAL_UNPUBLISHED",
        "required_next_action": "REPAIR_LATEST",
    },
}


class ActionMatrixError(ValueError):
    """The frozen action matrix is missing, malformed, or ambiguous."""


@dataclass(frozen=True)
class ActionOutcome:
    """One fully specified result permitted by an action-matrix cell."""

    target_state: str | None
    terminal: bool
    command_commit: bool
    business_acceptance: bool
    exit_code: int
    latest_effect: str


@dataclass(frozen=True)
class ActionDecision:
    """The unique authorization decision for one Cartesian matrix cell."""

    rule_id: str
    allowed: bool
    read_only: bool
    reason: str
    retry_cas: str
    allowed_write_namespaces: tuple[str, ...]
    outcomes: tuple[ActionOutcome, ...]


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ActionMatrixError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActionMatrixError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_canonical_resource(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ActionMatrixError(f"resource byte SHA-256 mismatch: {path.name}")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionMatrixError(f"invalid JSON resource: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise ActionMatrixError(f"resource root must be an object: {path.name}")
    if raw != _canonical_json_bytes(payload) + b"\n":
        raise ActionMatrixError(f"resource is not compact sorted-key JSON: {path.name}")
    return payload


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if frozenset(payload) != expected:
        raise ActionMatrixError(f"{label} keys do not match the frozen contract")


def _require_string_list(
    value: Any,
    *,
    allowed: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) for item in value)
        or len(value) != len(set(value))
    ):
        raise ActionMatrixError(f"{label} must be a non-empty unique string list")
    result = tuple(value)
    if any(item not in allowed for item in result):
        raise ActionMatrixError(f"{label} contains an out-of-domain value")
    if result != tuple(item for item in allowed if item in result):
        raise ActionMatrixError(f"{label} must follow canonical domain order")
    return result


def _parse_outcome(payload: Any, *, label: str) -> ActionOutcome:
    if not isinstance(payload, Mapping):
        raise ActionMatrixError(f"{label} must be an object")
    _require_exact_keys(payload, _OUTCOME_KEYS, label=label)
    target_state = payload["target_state"]
    if target_state is not None and target_state != "$INPUT_STATE":
        if not isinstance(target_state, str) or target_state not in STATES:
            raise ActionMatrixError(f"{label}.target_state is invalid")
    terminal = payload["terminal"]
    command_commit = payload["command_commit"]
    business_acceptance = payload["business_acceptance"]
    if (
        type(terminal) is not bool
        or type(command_commit) is not bool
        or type(business_acceptance) is not bool
    ):
        raise ActionMatrixError(f"{label} boolean fields must be exact booleans")
    exit_code = payload["exit_code"]
    if type(exit_code) is not int or exit_code not in {0, 2}:
        raise ActionMatrixError(f"{label}.exit_code must be 0 or 2")
    latest_effect = payload["latest_effect"]
    if not isinstance(latest_effect, str) or latest_effect not in LATEST_EFFECTS:
        raise ActionMatrixError(f"{label}.latest_effect is invalid")
    if latest_effect in {"PUBLISHED", "REPAIRED"} and not command_commit:
        raise ActionMatrixError(f"{label} mutates latest without a durable commit")
    if target_state in TERMINAL_STATES and terminal is not True:
        raise ActionMatrixError(f"{label} terminal target is not marked terminal")
    if target_state in NONTERMINAL_STATES and terminal is not False:
        raise ActionMatrixError(f"{label} nonterminal target is marked terminal")
    if business_acceptance and exit_code != 0:
        raise ActionMatrixError(f"{label} accepted business outcome must exit 0")
    if target_state in HARD_STOP_TERMINAL_STATES:
        if business_acceptance or not command_commit or exit_code != 2:
            raise ActionMatrixError(f"{label} hard stop semantics are invalid")
    return ActionOutcome(
        target_state=target_state,
        terminal=terminal,
        command_commit=command_commit,
        business_acceptance=business_acceptance,
        exit_code=exit_code,
        latest_effect=latest_effect,
    )


def _parse_decision(payload: Any, *, rule_id: str) -> ActionDecision:
    if not isinstance(payload, Mapping):
        raise ActionMatrixError(f"rule {rule_id} decision must be an object")
    _require_exact_keys(payload, _DECISION_KEYS, label=f"rule {rule_id} decision")
    allowed = payload["allowed"]
    read_only = payload["read_only"]
    if type(allowed) is not bool or type(read_only) is not bool:
        raise ActionMatrixError(f"rule {rule_id} decision booleans are invalid")
    if read_only and not allowed:
        raise ActionMatrixError(f"rule {rule_id} cannot reject as read-only")
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason:
        raise ActionMatrixError(f"rule {rule_id} reason is invalid")
    retry_cas = payload["retry_cas"]
    if not isinstance(retry_cas, str) or retry_cas not in RETRY_CAS_MODES:
        raise ActionMatrixError(f"rule {rule_id} retry/CAS mode is invalid")
    namespaces = payload["allowed_write_namespaces"]
    if not isinstance(namespaces, list) or len(namespaces) != len(set(namespaces)):
        raise ActionMatrixError(f"rule {rule_id} write namespaces are invalid")
    namespace_tuple = tuple(namespaces)
    if any(item not in WRITE_NAMESPACES for item in namespace_tuple):
        raise ActionMatrixError(f"rule {rule_id} write namespace is unknown")
    if namespace_tuple != tuple(item for item in WRITE_NAMESPACES if item in namespace_tuple):
        raise ActionMatrixError(f"rule {rule_id} write namespaces are not canonical")
    outcomes_payload = payload["outcomes"]
    if not isinstance(outcomes_payload, list) or not outcomes_payload:
        raise ActionMatrixError(f"rule {rule_id} outcomes are missing")
    outcomes = tuple(
        _parse_outcome(item, label=f"rule {rule_id} outcome[{index}]")
        for index, item in enumerate(outcomes_payload)
    )
    targets = tuple(outcome.target_state for outcome in outcomes)
    if len(targets) != len(set(targets)):
        raise ActionMatrixError(f"rule {rule_id} outcome targets are duplicated")
    if len(targets) > 1:
        if any(target is None or target == "$INPUT_STATE" for target in targets):
            raise ActionMatrixError(f"rule {rule_id} multi-outcome targets must be concrete")
        if targets != tuple(state for state in STATES if state in targets):
            raise ActionMatrixError(f"rule {rule_id} outcomes must follow canonical state order")
    if (not allowed or read_only) and namespace_tuple:
        raise ActionMatrixError(f"rule {rule_id} non-mutating decision permits writes")
    if read_only and any(outcome.command_commit for outcome in outcomes):
        raise ActionMatrixError(f"rule {rule_id} read-only outcome commits")
    if not allowed and any(
        outcome.command_commit or outcome.business_acceptance for outcome in outcomes
    ):
        raise ActionMatrixError(f"rule {rule_id} rejected outcome reports success")
    return ActionDecision(
        rule_id=rule_id,
        allowed=allowed,
        read_only=read_only,
        reason=reason,
        retry_cas=retry_cas,
        allowed_write_namespaces=namespace_tuple,
        outcomes=outcomes,
    )


def _validate_failure_semantics(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ActionMatrixError("failure semantics must be an object")
    if tuple(payload) != tuple(sorted(_EXPECTED_FAILURE_SEMANTICS)):
        raise ActionMatrixError("failure semantic ids drifted")
    for failure_id, raw_semantics in payload.items():
        if not isinstance(raw_semantics, Mapping):
            raise ActionMatrixError(f"failure semantic {failure_id} must be an object")
        _require_exact_keys(
            raw_semantics,
            _FAILURE_KEYS,
            label=f"failure semantic {failure_id}",
        )
        namespaces = raw_semantics["allowed_write_namespaces"]
        if (
            not isinstance(namespaces, list)
            or len(namespaces) != len(set(namespaces))
            or any(namespace not in WRITE_NAMESPACES for namespace in namespaces)
            or tuple(namespaces)
            != tuple(namespace for namespace in WRITE_NAMESPACES if namespace in namespaces)
        ):
            raise ActionMatrixError(f"failure semantic {failure_id} write namespaces are invalid")
        if (
            type(raw_semantics["business_acceptance"]) is not bool
            or type(raw_semantics["command_commit"]) is not bool
        ):
            raise ActionMatrixError(f"failure semantic {failure_id} booleans are invalid")
        if type(raw_semantics["exit_code"]) is not int or raw_semantics["exit_code"] != 2:
            raise ActionMatrixError(f"failure semantic {failure_id} exit code is invalid")
        if raw_semantics["latest_effect"] != "UNCHANGED":
            raise ActionMatrixError(f"failure semantic {failure_id} latest effect is invalid")
        if raw_semantics["receipt_effect"] not in {
            "NONE",
            "UNPUBLISHED_NOT_COMMITTED",
            "TERMINAL_UNPUBLISHED",
        }:
            raise ActionMatrixError(f"failure semantic {failure_id} receipt effect is invalid")
        required_next_action = raw_semantics["required_next_action"]
        if required_next_action is not None and required_next_action not in ACTIONS:
            raise ActionMatrixError(f"failure semantic {failure_id} next action is invalid")
    if dict(payload) != _EXPECTED_FAILURE_SEMANTICS:
        raise ActionMatrixError("failure semantics drifted")


def _rule_matches(
    match: Mapping[str, tuple[str, ...]],
    *,
    version: str,
    action: str,
    state: str,
    checkpoint: str,
) -> bool:
    return (
        version in match["versions"]
        and action in match["actions"]
        and state in match["states"]
        and checkpoint in match["checkpoints"]
    )


@lru_cache(maxsize=1)
def _load_action_matrix() -> tuple[
    tuple[tuple[str, Mapping[str, tuple[str, ...]], ActionDecision], ...],
    Mapping[str, Any],
]:
    payload = _read_canonical_resource(
        _ACTION_MATRIX_PATH,
        expected_sha256=ACTION_MATRIX_RESOURCE_SHA256,
    )
    _require_exact_keys(
        payload,
        frozenset(
            {
                "authority",
                "decisions",
                "domains",
                "failure_semantics",
                "namespace_paths",
                "package_version",
                "protocol_version",
                "rules",
                "schema",
                "version",
            }
        ),
        label="action-matrix resource",
    )
    if (
        payload["schema"] != ACTION_MATRIX_SCHEMA
        or payload["version"] != ACTION_MATRIX_SCHEMA
        or payload["package_version"] != PACKAGE_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
    ):
        raise ActionMatrixError("action-matrix identity mismatch")
    if payload["authority"] is not False:
        raise ActionMatrixError("action-matrix authority must be false")
    _validate_failure_semantics(payload["failure_semantics"])
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, Mapping) or not raw_decisions:
        raise ActionMatrixError("action-matrix decisions are missing")
    if tuple(raw_decisions) != tuple(sorted(raw_decisions)):
        raise ActionMatrixError("action-matrix decision ids must be sorted")
    decisions: dict[str, ActionDecision] = {}
    for decision_id, raw_decision in raw_decisions.items():
        if not isinstance(decision_id, str) or not decision_id:
            raise ActionMatrixError("action-matrix decision id is invalid")
        decisions[decision_id] = _parse_decision(
            raw_decision,
            rule_id=decision_id,
        )
    domains = payload["domains"]
    if not isinstance(domains, Mapping):
        raise ActionMatrixError("action-matrix domains must be an object")
    _require_exact_keys(
        domains,
        frozenset({"actions", "checkpoints", "states", "versions"}),
        label="action-matrix domains",
    )
    expected_domains = {
        "versions": list(VERSIONS),
        "actions": list(ACTIONS),
        "states": list(STATES),
        "checkpoints": list(CHECKPOINTS),
    }
    if dict(domains) != expected_domains:
        raise ActionMatrixError("action-matrix domains drifted")
    namespace_paths = payload["namespace_paths"]
    if not isinstance(namespace_paths, Mapping):
        raise ActionMatrixError("namespace paths must be an object")
    if tuple(namespace_paths) != tuple(sorted(namespace_paths)):
        raise ActionMatrixError("namespace path keys must be sorted")
    if set(namespace_paths) != set(WRITE_NAMESPACES):
        raise ActionMatrixError("namespace path set drifted")
    if any(
        not isinstance(path, str) or "/protocol-v2/" not in f"/{path.strip('/')}/"
        for path in namespace_paths.values()
    ):
        raise ActionMatrixError("every writable namespace must be protocol-v2 isolated")

    raw_rules = payload["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ActionMatrixError("action-matrix rules are missing")
    rules: list[tuple[str, Mapping[str, tuple[str, ...]], ActionDecision]] = []
    seen_rule_ids: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, Mapping):
            raise ActionMatrixError(f"rule[{index}] must be an object")
        _require_exact_keys(raw_rule, _RULE_KEYS, label=f"rule[{index}]")
        rule_id = raw_rule["id"]
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen_rule_ids:
            raise ActionMatrixError(f"rule[{index}] id is invalid")
        seen_rule_ids.add(rule_id)
        raw_match = raw_rule["match"]
        if not isinstance(raw_match, Mapping):
            raise ActionMatrixError(f"rule {rule_id} match must be an object")
        _require_exact_keys(raw_match, _MATCH_KEYS, label=f"rule {rule_id} match")
        match = {
            "versions": _require_string_list(
                raw_match["versions"],
                allowed=VERSIONS,
                label=f"rule {rule_id} versions",
            ),
            "actions": _require_string_list(
                raw_match["actions"],
                allowed=ACTIONS,
                label=f"rule {rule_id} actions",
            ),
            "states": _require_string_list(
                raw_match["states"],
                allowed=STATES,
                label=f"rule {rule_id} states",
            ),
            "checkpoints": _require_string_list(
                raw_match["checkpoints"],
                allowed=CHECKPOINTS,
                label=f"rule {rule_id} checkpoints",
            ),
        }
        decision_id = raw_rule["decision"]
        if not isinstance(decision_id, str) or decision_id not in decisions:
            raise ActionMatrixError(f"rule {rule_id} decision id is invalid")
        decision = replace(decisions[decision_id], rule_id=rule_id)
        if (
            any(checkpoint != "INITIALIZED" for checkpoint in match["checkpoints"])
            and decision.allowed_write_namespaces
        ):
            raise ActionMatrixError(f"rule {rule_id} permits writes before INITIALIZED")
        if any(checkpoint != "INITIALIZED" for checkpoint in match["checkpoints"]) and any(
            outcome.command_commit for outcome in decision.outcomes
        ):
            raise ActionMatrixError(f"rule {rule_id} commits before INITIALIZED")
        rules.append((rule_id, match, decision))

    for version, action, state, checkpoint in itertools.product(
        VERSIONS,
        ACTIONS,
        STATES,
        CHECKPOINTS,
    ):
        matching = [
            rule_id
            for rule_id, match, _decision in rules
            if _rule_matches(
                match,
                version=version,
                action=action,
                state=state,
                checkpoint=checkpoint,
            )
        ]
        if len(matching) != 1:
            raise ActionMatrixError(
                "matrix cell must match exactly one rule: "
                f"{version}/{action}/{state}/{checkpoint} -> {matching}"
            )
        selected = next(
            decision
            for _rule_id, match, decision in rules
            if _rule_matches(
                match,
                version=version,
                action=action,
                state=state,
                checkpoint=checkpoint,
            )
        )
        for outcome in selected.outcomes:
            target = state if outcome.target_state == "$INPUT_STATE" else outcome.target_state
            if target in TERMINAL_STATES and not outcome.terminal:
                raise ActionMatrixError("resolved terminal target is not terminal")
            if target in NONTERMINAL_STATES and outcome.terminal:
                raise ActionMatrixError("resolved nonterminal target is terminal")
    return tuple(rules), payload


@lru_cache(maxsize=1)
def load_state_machine_resource() -> Mapping[str, Any]:
    """Load and validate the isolated v2 state-machine contract."""

    payload = _read_canonical_resource(
        _STATE_MACHINE_PATH,
        expected_sha256=STATE_MACHINE_RESOURCE_SHA256,
    )
    _require_exact_keys(
        payload,
        frozenset(
            {
                "authority",
                "business_terminal_states",
                "hard_stop_terminal_states",
                "initial_state",
                "nonterminal_states",
                "package_version",
                "protocol_version",
                "protocol_roots",
                "schema",
                "terminal_semantics",
                "terminal_states",
                "transitions",
                "version",
            }
        ),
        label="state-machine resource",
    )
    if (
        payload["schema"] != STATE_MACHINE_SCHEMA
        or payload["version"] != STATE_MACHINE_SCHEMA
        or payload["package_version"] != PACKAGE_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
    ):
        raise ActionMatrixError("state-machine identity mismatch")
    if payload["authority"] is not False:
        raise ActionMatrixError("state-machine authority must be false")
    if payload["initial_state"] != "PREPARED":
        raise ActionMatrixError("state-machine initial state drifted")
    if payload["nonterminal_states"] != list(NONTERMINAL_STATES):
        raise ActionMatrixError("state-machine nonterminal set drifted")
    if payload["business_terminal_states"] != list(BUSINESS_TERMINAL_STATES):
        raise ActionMatrixError("state-machine business terminal set drifted")
    if payload["hard_stop_terminal_states"] != list(HARD_STOP_TERMINAL_STATES):
        raise ActionMatrixError("state-machine hard-stop set drifted")
    if payload["terminal_states"] != list(TERMINAL_STATES):
        raise ActionMatrixError("state-machine terminal set drifted")
    transitions = payload["transitions"]
    if not isinstance(transitions, Mapping) or set(transitions) != set(
        NONTERMINAL_STATES + TERMINAL_STATES
    ):
        raise ActionMatrixError("state-machine transition states drifted")
    for state, raw_targets in transitions.items():
        if not isinstance(raw_targets, list) or len(raw_targets) != len(set(raw_targets)):
            raise ActionMatrixError(f"state-machine targets invalid for {state}")
        if any(target not in NONTERMINAL_STATES + TERMINAL_STATES for target in raw_targets):
            raise ActionMatrixError(f"state-machine target unknown for {state}")
        if state in TERMINAL_STATES and raw_targets:
            raise ActionMatrixError(f"terminal state {state} is not immutable")
    terminal_semantics = payload["terminal_semantics"]
    if not isinstance(terminal_semantics, Mapping) or set(terminal_semantics) != set(
        TERMINAL_STATES
    ):
        raise ActionMatrixError("terminal semantics set drifted")
    for state in TERMINAL_STATES:
        outcome = _parse_outcome(
            terminal_semantics[state],
            label=f"terminal semantics {state}",
        )
        if outcome.target_state != state or not outcome.terminal:
            raise ActionMatrixError(f"terminal semantics target drifted for {state}")
        if outcome.latest_effect != "PUBLISHED":
            raise ActionMatrixError(f"terminal latest semantics drifted for {state}")
    roots = payload["protocol_roots"]
    if not isinstance(roots, Mapping) or set(roots) != {
        "private_source",
        "shadow_results",
    }:
        raise ActionMatrixError("state-machine protocol roots drifted")
    if roots != {
        "private_source": "data/private/v17_sources/protocol-v2",
        "shadow_results": "results/v17_shadow/protocol-v2",
    }:
        raise ActionMatrixError("state-machine protocol roots are not isolated")
    return payload


def matrix_cardinality() -> int:
    """Return the frozen Cartesian domain size (5 x 8 x 13 x 3)."""

    return len(VERSIONS) * len(ACTIONS) * len(STATES) * len(CHECKPOINTS)


def matching_rule_ids(
    *,
    version: str,
    action: str,
    state: str,
    checkpoint: str,
) -> tuple[str, ...]:
    """Return matching rule ids for audit tests; valid cells always have one."""

    _require_domain_cell(
        version=version,
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    rules, _payload = _load_action_matrix()
    return tuple(
        rule_id
        for rule_id, match, _decision in rules
        if _rule_matches(
            match,
            version=version,
            action=action,
            state=state,
            checkpoint=checkpoint,
        )
    )


def _require_domain_cell(
    *,
    version: str,
    action: str,
    state: str,
    checkpoint: str,
) -> None:
    if version not in VERSIONS:
        raise ActionMatrixError("version category is outside the frozen domain")
    if action not in ACTIONS:
        raise ActionMatrixError("action is outside the frozen domain")
    if state not in STATES:
        raise ActionMatrixError("state category is outside the frozen domain")
    if checkpoint not in CHECKPOINTS:
        raise ActionMatrixError("checkpoint is outside the frozen domain")


def decide_action(
    *,
    version: str,
    action: str,
    state: str,
    checkpoint: str,
) -> ActionDecision:
    """Return the unique, side-effect-free decision for one matrix cell."""

    _require_domain_cell(
        version=version,
        action=action,
        state=state,
        checkpoint=checkpoint,
    )
    rules, _payload = _load_action_matrix()
    matches = [
        decision
        for _rule_id, match, decision in rules
        if _rule_matches(
            match,
            version=version,
            action=action,
            state=state,
            checkpoint=checkpoint,
        )
    ]
    if len(matches) != 1:
        raise ActionMatrixError("matrix cell did not resolve to exactly one decision")
    decision = matches[0]
    resolved_outcomes = tuple(
        replace(outcome, target_state=state) if outcome.target_state == "$INPUT_STATE" else outcome
        for outcome in decision.outcomes
    )
    return replace(decision, outcomes=resolved_outcomes)


def action_matrix_resource() -> Mapping[str, Any]:
    """Return the validated canonical resource payload."""

    _rules, payload = _load_action_matrix()
    return payload


__all__ = [
    "ACTIONS",
    "ACTION_MATRIX_RESOURCE_SHA256",
    "ACTION_MATRIX_SCHEMA",
    "ActionDecision",
    "ActionMatrixError",
    "ActionOutcome",
    "BUSINESS_TERMINAL_STATES",
    "CHECKPOINTS",
    "HARD_STOP_TERMINAL_STATES",
    "LATEST_EFFECTS",
    "NONTERMINAL_STATES",
    "PROTOCOL_VERSION",
    "RETRY_CAS_MODES",
    "STATES",
    "STATE_MACHINE_RESOURCE_SHA256",
    "STATE_MACHINE_SCHEMA",
    "TERMINAL_STATES",
    "VERSIONS",
    "WRITE_NAMESPACES",
    "action_matrix_resource",
    "decide_action",
    "load_state_machine_resource",
    "matching_rule_ids",
    "matrix_cardinality",
]
