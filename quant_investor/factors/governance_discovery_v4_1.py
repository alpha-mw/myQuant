"""Pure, fail-closed FactorGovernanceProtocol v4.1 DISCOVERY contracts.

This module consumes only caller-supplied JSON-like values and pinned source
text.  It never imports or executes A_quant code, reads a registry, discovers a
provider, evaluates a factor, appends holdout data, constructs a portfolio, or
touches an execution surface.  The A_quant generator is interpreted through a
small structural AST grammar; its definitions remain ideas until every later
v4 gate is independently run under myQuant's point-in-time contract.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from quant_investor.factors.governance_cycle_state_v4_1 import (
    DISCOVERY,
    PRECOMMITTED,
    build_next_cycle_state_v4_1,
    validate_cycle_state_v4_1,
)
from quant_investor.factors.governance_screening_v4 import (
    validate_candidate_catalog_v4,
    validate_primitive_ontology_v4,
)
from quant_investor.factors.governance_source_readback_v4_1 import (
    binding_semantic_sha256_v4_1,
    cycle_root_semantic_sha256_v4_1,
    validate_cutoff_source_node_v4_1,
)
from quant_investor.factors.governance_source_v4_1 import (
    validate_design_source_node_v4_1,
)


PROTOCOL_VERSION = "v4"
AQUANT_PINNED_COMMIT = "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
AQUANT_GENERATOR_PATH = "A_quant/scripts/run_factor_batch_screen.py"
AQUANT_GENERATOR_FUNCTION = "generate_default_candidates"
EXPECTED_AQUANT_IDEA_COUNT = 100
EXPECTED_BASE_CANDIDATE_COUNT = 230

AQUANT_SOURCE_RECEIPT_SCHEMA_VERSION = (
    "factor-governance-aquant-source-receipt.v4.1"
)
SOURCE_IDEA_AUDIT_SCHEMA_VERSION = "factor-governance-source-idea-audit.v4.1"
LOCAL_COMPATIBILITY_CONTRACT_SCHEMA_VERSION = (
    "factor-governance-local-compatibility-contract.v4.1"
)
DISCOVERY_CATALOG_SCHEMA_VERSION = "factor-governance-discovery-catalog.v4.1"
STRUCTURAL_COLLISION_AUDIT_SCHEMA_VERSION = (
    "factor-governance-structural-collision-audit.v4.1"
)
DISCOVERY_SOURCE_NODE_SCHEMA_VERSION = (
    "factor-governance-discovery-source-node.v4.1"
)
DISCOVERY_READBACK_REPORT_SCHEMA_VERSION = (
    "factor-governance-discovery-readback.v4.1"
)

AQUANT_SOURCE_RECEIPT_FILENAME = "aquant_source_receipt.v4_1.json"
SOURCE_IDEA_AUDIT_FILENAME = "source_idea_audit.v4_1.json"
LOCAL_COMPATIBILITY_CONTRACT_FILENAME = "local_compatibility_contract.v4_1.json"
DISCOVERY_CATALOG_FILENAME = "discovery_catalog.v4_1.json"
STRUCTURAL_COLLISION_AUDIT_FILENAME = "structural_collision_audit.v4_1.json"
DISCOVERY_SOURCE_NODE_FILENAME = "discovery_source_node.v4_1.json"
DISCOVERY_CYCLE_STATE_FILENAME = "cycle_state.discovery.v4_1.json"
DISCOVERY_READBACK_REPORT_FILENAME = "discovery_readback_report.v4_1.json"

CANONICAL_ARTIFACT_FILENAMES = (
    AQUANT_SOURCE_RECEIPT_FILENAME,
    SOURCE_IDEA_AUDIT_FILENAME,
    LOCAL_COMPATIBILITY_CONTRACT_FILENAME,
    DISCOVERY_CATALOG_FILENAME,
    STRUCTURAL_COLLISION_AUDIT_FILENAME,
    DISCOVERY_SOURCE_NODE_FILENAME,
    DISCOVERY_CYCLE_STATE_FILENAME,
    DISCOVERY_READBACK_REPORT_FILENAME,
)
PRE_READBACK_ARTIFACT_FILENAMES = CANONICAL_ARTIFACT_FILENAMES[:-1]

LOCAL_COMPATIBILITY_CLAIM = (
    "locally_syntax_compatible_under_bound_myquant_evaluator"
)
LOCAL_COMPATIBILITY_CONTRACT_VERSION = "myquant-aquant-expression.v1"
EXPRESSION_AST_VERSION = "expression-ast.v1"
BASE_IMPLEMENTATION_FINGERPRINT_VERSION = "base-implementation.v1"
AQUANT_SOURCE_DEFINITION_VERSION = "aquant-source-definition.v1"
STRUCTURAL_FINGERPRINT_METHOD = "structural_fingerprint_only.v1"

LOCAL_ALLOWED_FUNCTION_ARITIES = {"cs_rank": 1, "ts_mean": 2}
LOCAL_ALLOWED_FIELDS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "vwap",
        "volume",
        "amount",
        "turnover_rate",
        "fin_roe",
        "fin_roa",
        "fin_debt_to_assets",
        "fin_net_profit_yoy",
        "fin_ocf_to_profit",
        "fin_fcf_to_profit",
        "fcf_to_price",
    }
)
LOCAL_ALLOWED_AST_KINDS = (
    "binary",
    "call",
    "constant",
    "name",
    "unary",
)

READINESS_DISCOVERY = "EXPLORATORY_DISCOVERY"
HOLDOUT_SEALED_NOT_APPENDED = "sealed_not_appended"
NOT_RUN = "not_run"
DISCOVERY_BLOCKERS = (
    "formal_catalog_not_materialized",
    "holdout_not_appended",
    "statistics_not_run",
    "verified_v4_replay_not_run",
    "qualification_not_evaluated",
)
MEASUREMENT_STATUS_FIELDS = (
    "statistics",
    "family_bh",
    "maturity",
    "walk_forward",
    "cost",
    "neutralization",
    "admission_duplicate_primitive",
    "high_correlation_dedup",
    "verified_v4_replay",
    "transaction_plan",
)
SIDE_EFFECT_FIELDS = (
    "registry",
    "wal",
    "budget",
    "production_receipt",
    "production_pointer",
    "proposal",
    "apply",
    "portfolio",
    "live_provider",
    "broker",
    "order",
    "trade",
    "network",
)

EXPECTED_AQUANT_SOURCE_PATHS = (
    "A_quant/app/data/schemas.py",
    "A_quant/app/factor_sandbox/expression.py",
    "A_quant/app/factor_sandbox/matrix_dataset.py",
    "A_quant/app/factor_sandbox/operators.py",
    "A_quant/docs/factor_time_alignment_policy.md",
    AQUANT_GENERATOR_PATH,
)
REQUIRED_CODE_BINDING_SUFFIXES = (
    "/quant_investor/factors/aquant_expression.py",
    "/quant_investor/factors/governance_cycle_state_v4_1.py",
    "/quant_investor/factors/governance_discovery_readback_v4_1.py",
    "/quant_investor/factors/governance_discovery_v4_1.py",
    "/quant_investor/factors/governance_screening_v4.py",
    "/quant_investor/factors/governance_source_readback_v4_1.py",
    "/quant_investor/factors/governance_source_v4_1.py",
    "/scripts/build_factor_v4_1_discovery.py",
)
PREDECESSOR_BUNDLE_FILENAMES = tuple(
    sorted(
        (
            "cutoff_input_binding.v4_1.json",
            "design_source.v4_1.json",
            "source_chain_node.v4_1.json",
            "cycle_state.precommitted.v4_1.json",
            "source_readback_report.v4_1.json",
        )
    )
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,254}")


class FactorGovernanceDiscoveryV4_1Error(ValueError):
    """Raised when a DISCOVERY source or artifact cannot be proven exactly."""


FactorGovernanceDiscoveryV41Error = FactorGovernanceDiscoveryV4_1Error


def canonical_bytes(value: Any) -> bytes:
    """Return compact sorted finite JSON bytes without a final newline."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


canonical_json_bytes = canonical_bytes


def canonical_file_bytes(value: Any) -> bytes:
    """Return the canonical owner-only artifact representation."""

    return canonical_bytes(value) + b"\n"


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def byte_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return semantic_sha256({key: value for key, value in payload.items() if key != field})


def _seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    sealed = copy.deepcopy(payload)
    sealed[field] = _self_hash(sealed, field)
    return sealed


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceDiscoveryV4_1Error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} field names must be strings"
        )
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} fields invalid: missing={missing}; unknown={unknown}"
        )
    return payload


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or value != value.strip() or (not value and not allow_empty):
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} must be an exact {'string' if allow_empty else 'non-empty string'}"
        )
    return value


def _safe_id(value: Any, label: str) -> str:
    text = _text(value, label)
    if _SAFE_ID_RE.fullmatch(text) is None or text in {".", ".."} or ".." in text:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} must be one safe identifier"
        )
    return text


def _safe_name(value: Any, label: str) -> str:
    text = _text(value, label)
    if _SAFE_NAME_RE.fullmatch(text) is None:
        raise FactorGovernanceDiscoveryV4_1Error(f"{label} is not a safe name")
    return text


def _sha(value: Any, label: str, *, nonzero: bool = True) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} must be lowercase SHA-256"
        )
    if nonzero and value == "0" * 64:
        raise FactorGovernanceDiscoveryV4_1Error(f"{label} must be nonzero")
    return value


def _oid(value: Any, label: str) -> str:
    if type(value) is not str or _OID_RE.fullmatch(value) is None or value == "0" * 40:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} must be a nonzero lowercase Git object id"
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} must be a positive integer"
        )
    return result


def _canonical_float_one(value: Any, label: str) -> float:
    if type(value) is not float or value != 1.0:
        raise FactorGovernanceDiscoveryV4_1Error(f"{label} must be canonical 1.0")
    return value


def _sorted_distinct_text_list(
    value: Any,
    label: str,
    *,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        raise FactorGovernanceDiscoveryV4_1Error(f"{label} must be a list")
    values = [_text(item, f"{label}[]") for item in value]
    if not allow_empty and not values:
        raise FactorGovernanceDiscoveryV4_1Error(f"{label} must not be empty")
    if values != sorted(values) or len(values) != len(set(values)):
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{label} must be sorted and distinct"
        )
    return values


def _finite_json(value: Any, label: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return copy.deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise FactorGovernanceDiscoveryV4_1Error(f"{label} must be finite")
        return value
    if isinstance(value, list):
        return [_finite_json(item, f"{label}[]") for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise FactorGovernanceDiscoveryV4_1Error(
                f"{label} keys must be strings"
            )
        return {key: _finite_json(item, f"{label}.{key}") for key, item in value.items()}
    raise FactorGovernanceDiscoveryV4_1Error(f"{label} must be exact JSON")


_EXTRACTED_IDEA_FIELDS = frozenset(
    {"name", "expression", "factor_type", "source_family", "rationale"}
)


def _validate_extracted_candidate(value: Any, label: str) -> dict[str, str]:
    row = _exact(value, _EXTRACTED_IDEA_FIELDS, label)
    return {
        "name": _safe_name(row["name"], f"{label}.name"),
        "expression": _text(row["expression"], f"{label}.expression"),
        "factor_type": _text(row["factor_type"], f"{label}.factor_type"),
        "source_family": _text(row["source_family"], f"{label}.source_family"),
        "rationale": _text(row["rationale"], f"{label}.rationale"),
    }


def _validate_nested_add_function(node: ast.FunctionDef) -> None:
    args = node.args
    expected_names = ["name", "expression", "family", "rationale", "factor_type"]
    expected_annotations = ["str", "str", "str", "str", "str"]
    if (
        node.name != "add"
        or node.decorator_list
        or args.posonlyargs
        or [arg.arg for arg in args.args] != expected_names
        or [
            ast.unparse(arg.annotation) if arg.annotation is not None else None
            for arg in args.args
        ]
        != expected_annotations
        or node.returns is None
        or ast.unparse(node.returns) != "None"
        or args.vararg is not None
        or args.kwonlyargs
        or args.kw_defaults
        or args.kwarg is not None
        or len(args.defaults) != 1
        or not isinstance(args.defaults[0], ast.Constant)
        or args.defaults[0].value != "alpha"
        or len(node.body) != 3
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "nested add helper does not match the allowlisted structure"
        )

    guard, seen_add, candidate_append = node.body
    valid_guard = (
        isinstance(guard, ast.If)
        and not guard.orelse
        and isinstance(guard.test, ast.Compare)
        and isinstance(guard.test.left, ast.Name)
        and guard.test.left.id == "name"
        and len(guard.test.ops) == 1
        and isinstance(guard.test.ops[0], ast.In)
        and len(guard.test.comparators) == 1
        and isinstance(guard.test.comparators[0], ast.Name)
        and guard.test.comparators[0].id == "seen"
        and len(guard.body) == 1
        and isinstance(guard.body[0], ast.Raise)
        and isinstance(guard.body[0].exc, ast.Call)
        and isinstance(guard.body[0].exc.func, ast.Name)
        and guard.body[0].exc.func.id == "ValueError"
        and len(guard.body[0].exc.args) == 1
        and ast.dump(guard.body[0].exc.args[0], include_attributes=False)
        == ast.dump(
            ast.parse('f"duplicate candidate name: {name}"', mode="eval").body,
            include_attributes=False,
        )
        and not guard.body[0].exc.keywords
        and guard.body[0].cause is None
    )
    if not valid_guard:
        raise FactorGovernanceDiscoveryV4_1Error(
            "nested add helper duplicate guard is not allowlisted"
        )

    valid_seen_add = (
        isinstance(seen_add, ast.Expr)
        and isinstance(seen_add.value, ast.Call)
        and isinstance(seen_add.value.func, ast.Attribute)
        and isinstance(seen_add.value.func.value, ast.Name)
        and seen_add.value.func.value.id == "seen"
        and seen_add.value.func.attr == "add"
        and len(seen_add.value.args) == 1
        and isinstance(seen_add.value.args[0], ast.Name)
        and seen_add.value.args[0].id == "name"
        and not seen_add.value.keywords
    )
    if not valid_seen_add:
        raise FactorGovernanceDiscoveryV4_1Error(
            "nested add helper set mutation is not allowlisted"
        )

    valid_append = (
        isinstance(candidate_append, ast.Expr)
        and isinstance(candidate_append.value, ast.Call)
        and isinstance(candidate_append.value.func, ast.Attribute)
        and isinstance(candidate_append.value.func.value, ast.Name)
        and candidate_append.value.func.value.id == "candidates"
        and candidate_append.value.func.attr == "append"
        and len(candidate_append.value.args) == 1
        and not candidate_append.value.keywords
        and isinstance(candidate_append.value.args[0], ast.Call)
        and isinstance(candidate_append.value.args[0].func, ast.Name)
        and candidate_append.value.args[0].func.id == "BatchFactorCandidate"
        and not candidate_append.value.args[0].keywords
        and [
            arg.id if isinstance(arg, ast.Name) else None
            for arg in candidate_append.value.args[0].args
        ]
        == ["name", "expression", "factor_type", "family", "rationale"]
    )
    if not valid_append:
        raise FactorGovernanceDiscoveryV4_1Error(
            "nested add helper candidate mutation is not allowlisted"
        )


def _safe_source_value(node: ast.AST, environment: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        value = node.value
        if type(value) is bool or type(value) not in (str, int, float):
            raise FactorGovernanceDiscoveryV4_1Error(
                "generator literal type is not allowlisted"
            )
        if type(value) is float and not math.isfinite(value):
            raise FactorGovernanceDiscoveryV4_1Error(
                "generator float literal must be finite"
            )
        return value
    if isinstance(node, ast.Name):
        if node.id not in environment:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"generator name is not bound: {node.id}"
            )
        return copy.deepcopy(environment[node.id])
    if isinstance(node, ast.List):
        return [_safe_source_value(item, environment) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_source_value(item, environment) for item in node.elts)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for item in node.values:
            if isinstance(item, ast.Constant) and type(item.value) is str:
                parts.append(item.value)
                continue
            if (
                not isinstance(item, ast.FormattedValue)
                or item.conversion != -1
                or item.format_spec is not None
            ):
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator f-string formatting is not allowlisted"
                )
            rendered = _safe_source_value(item.value, environment)
            if type(rendered) not in (str, int, float):
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator f-string value is not scalar"
                )
            parts.append(str(rendered))
        return "".join(parts)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "min":
            if len(node.args) != 2 or node.keywords:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator min call is not allowlisted"
                )
            min_values = [_safe_source_value(item, environment) for item in node.args]
            if any(type(value) is not int for value in min_values):
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator min arguments must be integers"
                )
            return min(min_values)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and not node.args
            and isinstance(node.func.value, (ast.Name, ast.Constant))
        ):
            template = _safe_source_value(node.func.value, environment)
            if type(template) is not str or len(node.keywords) != 1:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator format call is not allowlisted"
                )
            format_values: dict[str, int] = {}
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg in format_values:
                    raise FactorGovernanceDiscoveryV4_1Error(
                        "generator format keyword is invalid"
                    )
                value = _safe_source_value(keyword.value, environment)
                if type(value) is not int:
                    raise FactorGovernanceDiscoveryV4_1Error(
                        "generator format values must be integers"
                    )
                format_values[keyword.arg] = value
            if set(format_values) != {"w"}:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator format supports only the exact w keyword"
                )
            literal = template.replace("{w}", "")
            if "{" in literal or "}" in literal or "{w}" not in template:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator format template permits only literal {w} placeholders"
                )
            return template.replace("{w}", str(format_values["w"]))
    raise FactorGovernanceDiscoveryV4_1Error(
        f"generator value syntax is not allowlisted: {type(node).__name__}"
    )


def _bind_loop_target(
    target: ast.AST,
    value: Any,
    environment: dict[str, Any],
) -> None:
    if isinstance(target, ast.Name):
        environment[target.id] = copy.deepcopy(value)
        return
    if isinstance(target, ast.Tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(target.elts):
            raise FactorGovernanceDiscoveryV4_1Error(
                "generator tuple-unpack target does not match its value"
            )
        if any(not isinstance(item, ast.Name) for item in target.elts):
            raise FactorGovernanceDiscoveryV4_1Error(
                "generator tuple-unpack target must contain names"
            )
        for item, element in zip(target.elts, value, strict=True):
            assert isinstance(item, ast.Name)
            environment[item.id] = copy.deepcopy(element)
        return
    raise FactorGovernanceDiscoveryV4_1Error(
        "generator for-loop target is not allowlisted"
    )


def _append_from_add_call(
    call: ast.Call,
    environment: Mapping[str, Any],
    candidates: list[dict[str, str]],
    seen: set[str],
) -> None:
    if not isinstance(call.func, ast.Name) or call.func.id != "add" or call.keywords:
        raise FactorGovernanceDiscoveryV4_1Error(
            "generator expression call is not allowlisted"
        )
    if len(call.args) == 1 and isinstance(call.args[0], ast.Starred):
        values = _safe_source_value(call.args[0].value, environment)
        if not isinstance(values, tuple):
            raise FactorGovernanceDiscoveryV4_1Error(
                "starred add input must be a validated tuple"
            )
        arguments = list(values)
    else:
        if any(isinstance(item, ast.Starred) for item in call.args):
            raise FactorGovernanceDiscoveryV4_1Error(
                "mixed starred add arguments are forbidden"
            )
        arguments = [_safe_source_value(item, environment) for item in call.args]
    if len(arguments) not in (4, 5) or any(type(item) is not str for item in arguments):
        raise FactorGovernanceDiscoveryV4_1Error(
            "add requires four or five exact string arguments"
        )
    name, expression, family, rationale = arguments[:4]
    factor_type = arguments[4] if len(arguments) == 5 else "alpha"
    if name in seen:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"duplicate candidate name: {name}"
        )
    row = _validate_extracted_candidate(
        {
            "name": name,
            "expression": expression,
            "factor_type": factor_type,
            "source_family": family,
            "rationale": rationale,
        },
        "extracted candidate",
    )
    seen.add(row["name"])
    candidates.append(row)


def _interpret_generator_statements(
    statements: Sequence[ast.stmt],
    environment: dict[str, Any],
    candidates: list[dict[str, str]],
    seen: set[str],
) -> None:
    for statement in statements:
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.targets[0].id in {"candidates", "seen", "add"}
                or statement.type_comment is not None
            ):
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator assignment is not allowlisted"
                )
            name = statement.targets[0].id
            if name in environment:
                raise FactorGovernanceDiscoveryV4_1Error(
                    f"generator assignment rebind is forbidden: {name}"
                )
            environment[name] = _safe_source_value(statement.value, environment)
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            _append_from_add_call(statement.value, environment, candidates, seen)
            continue
        if isinstance(statement, ast.For):
            if statement.orelse or statement.type_comment is not None:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator for-loop extras are forbidden"
                )
            values = _safe_source_value(statement.iter, environment)
            if not isinstance(values, (list, tuple)):
                raise FactorGovernanceDiscoveryV4_1Error(
                    "generator for-loop input must be a literal sequence"
                )
            for value in values:
                iteration_environment = dict(environment)
                _bind_loop_target(statement.target, value, iteration_environment)
                _interpret_generator_statements(
                    statement.body,
                    iteration_environment,
                    candidates,
                    seen,
                )
            continue
        raise FactorGovernanceDiscoveryV4_1Error(
            f"generator statement is not allowlisted: {type(statement).__name__}"
        )


def _is_candidates_initialization(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "candidates"
        and statement.simple == 1
        and ast.unparse(statement.annotation) == "list[BatchFactorCandidate]"
        and isinstance(statement.value, ast.List)
        and not statement.value.elts
    )


def _is_seen_initialization(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "seen"
        and statement.simple == 1
        and ast.unparse(statement.annotation) == "set[str]"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "set"
        and not statement.value.args
        and not statement.value.keywords
    )


def _validate_terminal_count_guard(statement: ast.stmt) -> None:
    valid = (
        isinstance(statement, ast.If)
        and not statement.orelse
        and isinstance(statement.test, ast.Compare)
        and isinstance(statement.test.left, ast.Call)
        and isinstance(statement.test.left.func, ast.Name)
        and statement.test.left.func.id == "len"
        and len(statement.test.left.args) == 1
        and isinstance(statement.test.left.args[0], ast.Name)
        and statement.test.left.args[0].id == "candidates"
        and not statement.test.left.keywords
        and len(statement.test.ops) == 1
        and isinstance(statement.test.ops[0], ast.NotEq)
        and len(statement.test.comparators) == 1
        and isinstance(statement.test.comparators[0], ast.Constant)
        and statement.test.comparators[0].value == EXPECTED_AQUANT_IDEA_COUNT
        and len(statement.body) == 1
        and isinstance(statement.body[0], ast.Raise)
        and isinstance(statement.body[0].exc, ast.Call)
        and isinstance(statement.body[0].exc.func, ast.Name)
        and statement.body[0].exc.func.id == "AssertionError"
        and len(statement.body[0].exc.args) == 1
        and ast.dump(statement.body[0].exc.args[0], include_attributes=False)
        == ast.dump(
            ast.parse(
                'f"candidate generator produced {len(candidates)} candidates, expected 100"',
                mode="eval",
            ).body,
            include_attributes=False,
        )
        and not statement.body[0].exc.keywords
        and statement.body[0].cause is None
    )
    if not valid:
        raise FactorGovernanceDiscoveryV4_1Error(
            "generator terminal count guard is not allowlisted"
        )


def extract_aquant_candidates_from_source(source: str | bytes) -> list[dict[str, str]]:
    """Interpret exactly the pinned generator grammar without executing Python."""

    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FactorGovernanceDiscoveryV4_1Error(
                "A_quant generator source must be UTF-8"
            ) from exc
    elif type(source) is str:
        text = source
    else:
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant generator source must be exact text or bytes"
        )
    try:
        module = ast.parse(text, filename=AQUANT_GENERATOR_PATH, mode="exec")
    except SyntaxError as exc:
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant generator source is invalid Python syntax"
        ) from exc
    functions = [
        statement
        for statement in module.body
        if isinstance(statement, ast.FunctionDef)
        and statement.name == AQUANT_GENERATOR_FUNCTION
    ]
    if any(
        isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == AQUANT_GENERATOR_FUNCTION
        and statement not in functions
        for statement in module.body
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "unsupported duplicate generator construct"
        )
    if len(functions) != 1:
        raise FactorGovernanceDiscoveryV4_1Error(
            "source must contain exactly one generate_default_candidates function"
        )
    function = functions[0]
    if (
        function.decorator_list
        or function.args.posonlyargs
        or function.args.args
        or function.args.vararg is not None
        or function.args.kwonlyargs
        or function.args.kw_defaults
        or function.args.kwarg is not None
        or function.args.defaults
        or function.returns is None
        or ast.unparse(function.returns) != "list[BatchFactorCandidate]"
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "generator function signature is not allowlisted"
        )
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if type(body[0].value.value) is not str:
            raise FactorGovernanceDiscoveryV4_1Error(
                "generator leading expression must be a docstring"
            )
        body.pop(0)
    if len(body) < 6 or not _is_candidates_initialization(body.pop(0)):
        raise FactorGovernanceDiscoveryV4_1Error(
            "generator candidates initialization is not allowlisted"
        )
    if not _is_seen_initialization(body.pop(0)):
        raise FactorGovernanceDiscoveryV4_1Error(
            "generator seen initialization is not allowlisted"
        )
    add_helper = body.pop(0)
    if not isinstance(add_helper, ast.FunctionDef):
        raise FactorGovernanceDiscoveryV4_1Error("generator add helper is missing")
    _validate_nested_add_function(add_helper)
    terminal_return = body.pop()
    if (
        not isinstance(terminal_return, ast.Return)
        or not isinstance(terminal_return.value, ast.Name)
        or terminal_return.value.id != "candidates"
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "generator must terminate with exact return candidates"
        )
    terminal_guard = body.pop()
    _validate_terminal_count_guard(terminal_guard)

    environment: dict[str, Any] = {}
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    _interpret_generator_statements(body, environment, candidates, seen)
    if len(candidates) != EXPECTED_AQUANT_IDEA_COUNT:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"generator produced {len(candidates)} candidates, expected 100"
        )
    return candidates


def _normalized_expression_node(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.Name):
        return {"kind": "name", "identifier": node.id}
    if isinstance(node, ast.Constant):
        value = node.value
        if type(value) is bool or type(value) not in (int, float):
            raise FactorGovernanceDiscoveryV4_1Error(
                "expression constants must be finite numbers"
            )
        if type(value) is float and not math.isfinite(value):
            raise FactorGovernanceDiscoveryV4_1Error(
                "expression constants must be finite numbers"
            )
        return {"kind": "constant", "value": value}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return {
            "kind": "unary",
            "operator": "negate" if isinstance(node.op, ast.USub) else "positive",
            "operand": _normalized_expression_node(node.operand),
        }
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        operators = {
            ast.Add: "add",
            ast.Sub: "subtract",
            ast.Mult: "multiply",
            ast.Div: "divide",
        }
        return {
            "kind": "binary",
            "operator": operators[type(node.op)],
            "left": _normalized_expression_node(node.left),
            "right": _normalized_expression_node(node.right),
        }
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise FactorGovernanceDiscoveryV4_1Error(
                "expression calls require a simple name and positional arguments"
            )
        return {
            "kind": "call",
            "function": node.func.id,
            "arguments": [_normalized_expression_node(item) for item in node.args],
        }
    raise FactorGovernanceDiscoveryV4_1Error(
        f"expression syntax is not allowlisted: {type(node).__name__}"
    )


def normalize_expression_ast_v4_1(expression: str) -> dict[str, Any]:
    """Normalize syntax only; no algebraic or commutative rewriting occurs."""

    text = _text(expression, "expression")
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FactorGovernanceDiscoveryV4_1Error(
            "expression is not valid Python expression syntax"
        ) from exc
    return _normalized_expression_node(parsed.body)


def _expression_metadata(tree: Mapping[str, Any]) -> tuple[list[str], list[str], int]:
    names: set[str] = set()
    functions: list[tuple[str, int, Mapping[str, Any] | None]] = []

    def visit(node: Mapping[str, Any]) -> None:
        kind = node.get("kind")
        if kind == "name":
            names.add(str(node["identifier"]))
        elif kind == "constant":
            return
        elif kind == "unary":
            visit(node["operand"])
        elif kind == "binary":
            visit(node["left"])
            visit(node["right"])
        elif kind == "call":
            arguments = node["arguments"]
            final = arguments[-1] if arguments else None
            functions.append((str(node["function"]), len(arguments), final))
            for argument in arguments:
                visit(argument)
        else:
            raise FactorGovernanceDiscoveryV4_1Error(
                "normalized expression tree has an unknown kind"
            )

    visit(tree)
    lookbacks: list[int] = []
    for function, _arity, final in functions:
        if (
            function.startswith("ts_")
            and isinstance(final, Mapping)
            and final.get("kind") == "constant"
            and type(final.get("value")) is int
            and final["value"] > 0
        ):
            lookbacks.append(int(final["value"]))
    return sorted(names), sorted({item[0] for item in functions}), max(lookbacks, default=1)


def assess_local_compatibility_v4_1(
    expression: str,
    compatibility_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess syntax compatibility only; this is never runtime/PIT equivalence."""

    contract = validate_local_compatibility_contract_v4_1(
        compatibility_contract
    )
    try:
        tree = normalize_expression_ast_v4_1(expression)
    except FactorGovernanceDiscoveryV4_1Error as exc:
        return {
            "status": "incompatible",
            "reasons": [f"unsupported_syntax:{exc}"],
            "normalized_expression_ast": None,
            "input_fields": [],
            "lookback": 1,
        }
    input_fields, _functions, lookback = _expression_metadata(tree)
    reasons: set[str] = set()

    def inspect(node: Mapping[str, Any]) -> None:
        kind = node["kind"]
        if kind == "name":
            identifier = str(node["identifier"])
            if identifier not in contract["allowed_fields"]:
                reasons.add(f"unsupported_name:{identifier}")
            return
        if kind == "constant":
            return
        if kind == "unary":
            inspect(node["operand"])
            return
        if kind == "binary":
            inspect(node["left"])
            inspect(node["right"])
            return
        if kind == "call":
            function = str(node["function"])
            arguments = node["arguments"]
            expected_arity = contract["allowed_functions"].get(function)
            if expected_arity is None:
                reasons.add(f"unsupported_function:{function}")
            elif len(arguments) != expected_arity:
                reasons.add(
                    f"unsupported_arity:{function}:{len(arguments)}:{expected_arity}"
                )
            if function == "ts_mean" and (
                len(arguments) != 2
                or arguments[-1].get("kind") != "constant"
                or type(arguments[-1].get("value")) is not int
                or arguments[-1]["value"] <= 0
            ):
                reasons.add("invalid_window:ts_mean")
            for argument in arguments:
                inspect(argument)
            return
        raise FactorGovernanceDiscoveryV4_1Error(
            "normalized expression tree has an unknown kind"
        )

    inspect(tree)
    ordered_reasons = sorted(reasons)
    return {
        "status": "compatible" if not ordered_reasons else "incompatible",
        "reasons": ordered_reasons,
        "normalized_expression_ast": tree,
        "input_fields": input_fields,
        "lookback": lookback,
    }


def expression_structural_fingerprint_sha256_v4_1(
    expression: str,
    *,
    compatibility_contract_sha256: str,
    direction: float = 1.0,
) -> str:
    _canonical_float_one(direction, "direction")
    contract_sha = _sha(
        compatibility_contract_sha256, "compatibility_contract_sha256"
    )
    return semantic_sha256(
        {
            "version": EXPRESSION_AST_VERSION,
            "ast": normalize_expression_ast_v4_1(expression),
            "direction": 1.0,
            "compatibility_contract_sha256": contract_sha,
        }
    )


def base_structural_fingerprint_sha256_v4_1(
    candidate: Mapping[str, Any],
    *,
    compatibility_contract_sha256: str,
) -> str:
    contract_sha = _sha(
        compatibility_contract_sha256, "compatibility_contract_sha256"
    )
    expression = candidate.get("expression")
    if type(expression) is not str:
        raise FactorGovernanceDiscoveryV4_1Error(
            "base candidate expression must be a string"
        )
    if expression:
        return expression_structural_fingerprint_sha256_v4_1(
            expression,
            compatibility_contract_sha256=contract_sha,
            direction=_canonical_float_one(candidate.get("direction"), "base direction"),
        )
    params = _finite_json(candidate.get("params"), "base params")
    if not isinstance(params, dict):
        raise FactorGovernanceDiscoveryV4_1Error("base params must be an object")
    return semantic_sha256(
        {
            "version": BASE_IMPLEMENTATION_FINGERPRINT_VERSION,
            "implementation": _text(candidate.get("implementation"), "base implementation"),
            "params": params,
            "direction": _canonical_float_one(candidate.get("direction"), "base direction"),
            "window": params.get("window"),
            "lookback": _positive_int(candidate.get("lookback"), "base lookback"),
            "input_fields": _sorted_distinct_text_list(
                candidate.get("input_fields"), "base input_fields"
            ),
            "primitive_ids": _sorted_distinct_text_list(
                candidate.get("primitive_ids"), "base primitive_ids", allow_empty=False
            ),
            "compatibility_contract_sha256": contract_sha,
        }
    )


_AQUANT_SOURCE_FILE_FIELDS = frozenset(
    {"path", "git_mode", "blob_oid", "raw_sha256"}
)
_AQUANT_SOURCE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "source_system",
        "repository_top_level",
        "pinned_commit",
        "object_type",
        "generator_path",
        "generator_function",
        "source_files",
        "generator_candidate_count",
        "ordered_names_semantic_sha256",
        "healthy",
        "receipt_semantic_sha256",
    }
)
_LOCAL_COMPATIBILITY_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "contract_version",
        "claim",
        "evaluator_path",
        "evaluator_source_byte_sha256",
        "allowed_functions",
        "allowed_fields",
        "allowed_ast_kinds",
        "direction",
        "direction_origin",
        "runtime_equivalence_claimed",
        "data_equivalence_claimed",
        "pit_equivalence_claimed",
        "contract_semantic_sha256",
    }
)


def _normalize_aquant_source_files(
    value: Any,
    *,
    require_sorted: bool,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise FactorGovernanceDiscoveryV4_1Error("source_files must be a list")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        row = _exact(raw, _AQUANT_SOURCE_FILE_FIELDS, f"source_files[{index}]")
        path = _text(row["path"], f"source_files[{index}].path")
        if path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise FactorGovernanceDiscoveryV4_1Error(
                "A_quant source paths must be repository-relative"
            )
        if row["git_mode"] != "100644":
            raise FactorGovernanceDiscoveryV4_1Error(
                "A_quant source entries must be exact regular blobs mode 100644"
            )
        rows.append(
            {
                "path": path,
                "git_mode": "100644",
                "blob_oid": _oid(row["blob_oid"], f"source_files[{index}].blob_oid"),
                "raw_sha256": _sha(
                    row["raw_sha256"], f"source_files[{index}].raw_sha256"
                ),
            }
        )
    ordered = sorted(rows, key=lambda item: item["path"])
    paths = [item["path"] for item in ordered]
    if tuple(paths) != EXPECTED_AQUANT_SOURCE_PATHS:
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant source receipt must bind exactly the six pinned source paths"
        )
    if require_sorted and rows != ordered:
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant source files must be sorted by path"
        )
    return ordered


def build_aquant_source_receipt_v4_1(
    *,
    repository_top_level: str,
    pinned_commit: str,
    source_files: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a healthy receipt only for the exact pinned Git-object source set."""

    root = _text(repository_top_level, "repository_top_level")
    root_path = PurePosixPath(root)
    if not root_path.is_absolute() or ".." in root_path.parts:
        raise FactorGovernanceDiscoveryV4_1Error(
            "repository_top_level must be an exact absolute path"
        )
    if pinned_commit != AQUANT_PINNED_COMMIT:
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant pinned commit does not match this discovery contract"
        )
    normalized_candidates = [
        _validate_extracted_candidate(row, f"candidates[{index}]")
        for index, row in enumerate(candidates)
    ]
    names = [row["name"] for row in normalized_candidates]
    if (
        len(normalized_candidates) != EXPECTED_AQUANT_IDEA_COUNT
        or len(names) != len(set(names))
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant receipt requires exactly 100 uniquely named ideas"
        )
    normalized_files = _normalize_aquant_source_files(
        list(source_files), require_sorted=False
    )
    payload = {
        "schema_version": AQUANT_SOURCE_RECEIPT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_system": "A_quant",
        "repository_top_level": root,
        "pinned_commit": AQUANT_PINNED_COMMIT,
        "object_type": "commit",
        "generator_path": AQUANT_GENERATOR_PATH,
        "generator_function": AQUANT_GENERATOR_FUNCTION,
        "source_files": normalized_files,
        "generator_candidate_count": EXPECTED_AQUANT_IDEA_COUNT,
        "ordered_names_semantic_sha256": semantic_sha256(names),
        "healthy": True,
    }
    return validate_aquant_source_receipt_v4_1(
        _seal(payload, "receipt_semantic_sha256")
    )


def validate_aquant_source_receipt_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(value, _AQUANT_SOURCE_RECEIPT_FIELDS, "A_quant source receipt")
    canonical_bytes(payload)
    if payload["schema_version"] != AQUANT_SOURCE_RECEIPT_SCHEMA_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error("A_quant receipt schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error("A_quant receipt protocol mismatch")
    if payload["source_system"] != "A_quant":
        raise FactorGovernanceDiscoveryV4_1Error("source_system must be A_quant")
    root = _text(payload["repository_top_level"], "repository_top_level")
    root_path = PurePosixPath(root)
    if not root_path.is_absolute() or ".." in root_path.parts:
        raise FactorGovernanceDiscoveryV4_1Error(
            "repository_top_level must be an exact absolute path"
        )
    if payload["pinned_commit"] != AQUANT_PINNED_COMMIT:
        raise FactorGovernanceDiscoveryV4_1Error("A_quant pinned commit mismatch")
    if payload["object_type"] != "commit":
        raise FactorGovernanceDiscoveryV4_1Error("pinned object must be a commit")
    if (
        payload["generator_path"] != AQUANT_GENERATOR_PATH
        or payload["generator_function"] != AQUANT_GENERATOR_FUNCTION
    ):
        raise FactorGovernanceDiscoveryV4_1Error("generator identity mismatch")
    files = _normalize_aquant_source_files(
        payload["source_files"], require_sorted=True
    )
    if payload["generator_candidate_count"] != EXPECTED_AQUANT_IDEA_COUNT:
        raise FactorGovernanceDiscoveryV4_1Error(
            "receipt generator candidate count must be exactly 100"
        )
    ordered_names_sha = _sha(
        payload["ordered_names_semantic_sha256"],
        "ordered_names_semantic_sha256",
    )
    if payload["healthy"] is not True:
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant source receipt must be healthy"
        )
    observed_sha = _sha(
        payload["receipt_semantic_sha256"], "receipt_semantic_sha256"
    )
    normalized = {
        "schema_version": AQUANT_SOURCE_RECEIPT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_system": "A_quant",
        "repository_top_level": root,
        "pinned_commit": AQUANT_PINNED_COMMIT,
        "object_type": "commit",
        "generator_path": AQUANT_GENERATOR_PATH,
        "generator_function": AQUANT_GENERATOR_FUNCTION,
        "source_files": files,
        "generator_candidate_count": EXPECTED_AQUANT_IDEA_COUNT,
        "ordered_names_semantic_sha256": ordered_names_sha,
        "healthy": True,
        "receipt_semantic_sha256": observed_sha,
    }
    if observed_sha != _self_hash(normalized, "receipt_semantic_sha256"):
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant source receipt semantic SHA mismatch"
        )
    return normalized


def build_local_compatibility_contract_v4_1(
    *,
    evaluator_source_byte_sha256: str,
) -> dict[str, Any]:
    """Bind the narrow current myQuant evaluator vocabulary without equivalence."""

    payload = {
        "schema_version": LOCAL_COMPATIBILITY_CONTRACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "contract_version": LOCAL_COMPATIBILITY_CONTRACT_VERSION,
        "claim": LOCAL_COMPATIBILITY_CLAIM,
        "evaluator_path": "quant_investor/factors/aquant_expression.py",
        "evaluator_source_byte_sha256": _sha(
            evaluator_source_byte_sha256, "evaluator_source_byte_sha256"
        ),
        "allowed_functions": dict(sorted(LOCAL_ALLOWED_FUNCTION_ARITIES.items())),
        "allowed_fields": sorted(LOCAL_ALLOWED_FIELDS),
        "allowed_ast_kinds": list(LOCAL_ALLOWED_AST_KINDS),
        "direction": 1.0,
        "direction_origin": "expression_signed_ast",
        "runtime_equivalence_claimed": False,
        "data_equivalence_claimed": False,
        "pit_equivalence_claimed": False,
    }
    return validate_local_compatibility_contract_v4_1(
        _seal(payload, "contract_semantic_sha256")
    )


def validate_local_compatibility_contract_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(
        value,
        _LOCAL_COMPATIBILITY_CONTRACT_FIELDS,
        "local compatibility contract",
    )
    canonical_bytes(payload)
    expected_constants = {
        "schema_version": LOCAL_COMPATIBILITY_CONTRACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "contract_version": LOCAL_COMPATIBILITY_CONTRACT_VERSION,
        "claim": LOCAL_COMPATIBILITY_CLAIM,
        "evaluator_path": "quant_investor/factors/aquant_expression.py",
        "allowed_functions": dict(sorted(LOCAL_ALLOWED_FUNCTION_ARITIES.items())),
        "allowed_fields": sorted(LOCAL_ALLOWED_FIELDS),
        "allowed_ast_kinds": list(LOCAL_ALLOWED_AST_KINDS),
        "direction": 1.0,
        "direction_origin": "expression_signed_ast",
        "runtime_equivalence_claimed": False,
        "data_equivalence_claimed": False,
        "pit_equivalence_claimed": False,
    }
    for field, expected in expected_constants.items():
        if payload[field] != expected or (
            field == "direction" and type(payload[field]) is not float
        ):
            raise FactorGovernanceDiscoveryV4_1Error(
                f"local compatibility contract field mismatch: {field}"
            )
    evaluator_sha = _sha(
        payload["evaluator_source_byte_sha256"],
        "evaluator_source_byte_sha256",
    )
    observed_sha = _sha(
        payload["contract_semantic_sha256"], "contract_semantic_sha256"
    )
    normalized = {
        **expected_constants,
        "evaluator_source_byte_sha256": evaluator_sha,
        "contract_semantic_sha256": observed_sha,
    }
    if observed_sha != _self_hash(normalized, "contract_semantic_sha256"):
        raise FactorGovernanceDiscoveryV4_1Error(
            "local compatibility contract semantic SHA mismatch"
        )
    return normalized


_SOURCE_IDEA_ROW_FIELDS = frozenset(
    {
        "source_index",
        "candidate_id",
        "name",
        "expression",
        "factor_type",
        "source_family",
        "rationale",
        "direction",
        "direction_origin",
        "compatibility_status",
        "incompatibility_reasons",
        "normalized_expression_ast",
        "input_fields",
        "lookback",
        "structural_fingerprint_sha256",
        "catalog_role",
        "selected",
        "structural_alias_of",
    }
)
_SOURCE_IDEA_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "source_receipt_sha256",
        "compatibility_contract_sha256",
        "base_catalog_sha256",
        "total_idea_count",
        "compatible_count",
        "incompatible_count",
        "new_candidate_count",
        "structural_alias_count",
        "ordered_names_semantic_sha256",
        "compatible_ordered_names_semantic_sha256",
        "structural_alias_ordered_names_semantic_sha256",
        "ideas",
        "formal_admission_authority",
        "statistics_status",
        "audit_semantic_sha256",
    }
)
_DISCOVERY_MEMBER_FIELDS = frozenset(
    {
        "candidate_id",
        "origin",
        "name",
        "expression",
        "implementation",
        "params",
        "direction",
        "direction_origin",
        "factor_type",
        "source_family",
        "rationale",
        "lookback",
        "input_fields",
        "primitive_ids",
        "structural_fingerprint_sha256",
        "source_definition_sha256",
        "catalog_role",
        "selected",
        "structural_alias_of",
        "initial_weight",
    }
)
_DISCOVERY_CATALOG_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "base_ontology_sha256",
        "base_catalog_sha256",
        "source_receipt_sha256",
        "compatibility_contract_sha256",
        "source_idea_audit_sha256",
        "member_count",
        "selected_count",
        "base_reference_count",
        "aquant_compatible_count",
        "new_candidate_count",
        "structural_alias_count",
        "base_ordered_definition_pairs_semantic_sha256",
        "members",
        "readiness",
        "qualification",
        "formal_admission_authority",
        "initial_weight_policy",
        "statistics_status",
        "admission_duplicate_primitive_status",
        "high_correlation_dedup_status",
        "catalog_semantic_sha256",
    }
)


def _aquant_candidate_id(name: str) -> str:
    return f"aquant:{AQUANT_PINNED_COMMIT}:{name}"


def _base_candidate_id(base_catalog_sha256: str, name: str) -> str:
    return f"myquant:{base_catalog_sha256}:{name}"


def aquant_source_definition_sha256_v4_1(candidate: Mapping[str, Any]) -> str:
    row = _validate_extracted_candidate(candidate, "A_quant source definition")
    return semantic_sha256(
        {
            "version": AQUANT_SOURCE_DEFINITION_VERSION,
            "pinned_commit": AQUANT_PINNED_COMMIT,
            **row,
            "direction": 1.0,
            "direction_origin": "expression_signed_ast",
        }
    )


def _build_base_discovery_members(
    *,
    base_catalog: Mapping[str, Any],
    compatibility_contract_sha256: str,
) -> list[dict[str, Any]]:
    catalog_sha = _sha(base_catalog["semantic_sha256"], "base catalog semantic SHA")
    members: list[dict[str, Any]] = []
    for row in base_catalog["candidates"]:
        params = _finite_json(row["params"], f"base params {row['name']}")
        assert isinstance(params, dict)
        members.append(
            {
                "candidate_id": _base_candidate_id(catalog_sha, row["name"]),
                "origin": "myquant",
                "name": row["name"],
                "expression": row["expression"],
                "implementation": row["implementation"],
                "params": params,
                "direction": 1.0,
                "direction_origin": "bound_base_catalog",
                "factor_type": None,
                "source_family": row["family"],
                "rationale": None,
                "lookback": row["lookback"],
                "input_fields": list(row["input_fields"]),
                "primitive_ids": list(row["primitive_ids"]),
                "structural_fingerprint_sha256": (
                    base_structural_fingerprint_sha256_v4_1(
                        row,
                        compatibility_contract_sha256=(
                            compatibility_contract_sha256
                        ),
                    )
                ),
                "source_definition_sha256": row["definition_sha256"],
                "catalog_role": "base_reference",
                "selected": True,
                "structural_alias_of": None,
                "initial_weight": 0.0,
            }
        )
    return members


def _plan_source_ideas(
    *,
    candidates: Sequence[Mapping[str, Any]],
    source_receipt: Mapping[str, Any],
    compatibility_contract: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    receipt = validate_aquant_source_receipt_v4_1(source_receipt)
    contract = validate_local_compatibility_contract_v4_1(
        compatibility_contract
    )
    normalized_candidates = [
        _validate_extracted_candidate(row, f"candidates[{index}]")
        for index, row in enumerate(candidates)
    ]
    names = [row["name"] for row in normalized_candidates]
    if (
        len(names) != EXPECTED_AQUANT_IDEA_COUNT
        or len(names) != len(set(names))
        or semantic_sha256(names) != receipt["ordered_names_semantic_sha256"]
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "A_quant candidate sequence does not match the healthy source receipt"
        )
    contract_sha = contract["contract_semantic_sha256"]
    base_members = _build_base_discovery_members(
        base_catalog=base_catalog,
        compatibility_contract_sha256=contract_sha,
    )
    base_by_fingerprint: dict[str, list[str]] = {}
    for member in base_members:
        base_by_fingerprint.setdefault(
            member["structural_fingerprint_sha256"], []
        ).append(member["candidate_id"])
    for ids in base_by_fingerprint.values():
        ids.sort()

    provisional: list[dict[str, Any]] = []
    compatible_by_fingerprint: dict[str, list[str]] = {}
    for index, candidate in enumerate(normalized_candidates):
        assessment = assess_local_compatibility_v4_1(
            candidate["expression"], contract
        )
        fingerprint: str | None = None
        if assessment["status"] == "compatible":
            fingerprint = expression_structural_fingerprint_sha256_v4_1(
                candidate["expression"],
                compatibility_contract_sha256=contract_sha,
            )
            compatible_by_fingerprint.setdefault(fingerprint, []).append(
                candidate["name"]
            )
        provisional.append(
            {
                "source_index": index,
                "candidate_id": _aquant_candidate_id(candidate["name"]),
                **candidate,
                "direction": 1.0,
                "direction_origin": "expression_signed_ast",
                "compatibility_status": assessment["status"],
                "incompatibility_reasons": assessment["reasons"],
                "normalized_expression_ast": assessment[
                    "normalized_expression_ast"
                ],
                "input_fields": assessment["input_fields"],
                "lookback": assessment["lookback"],
                "structural_fingerprint_sha256": fingerprint,
            }
        )
    for names_for_fingerprint in compatible_by_fingerprint.values():
        names_for_fingerprint.sort()

    planned: list[dict[str, Any]] = []
    for row in provisional:
        fingerprint = row["structural_fingerprint_sha256"]
        alias_of: str | None = None
        role = "incompatible"
        selected = False
        if fingerprint is not None:
            base_matches = base_by_fingerprint.get(fingerprint, [])
            if base_matches:
                role = "structural_alias"
                alias_of = base_matches[0]
            else:
                aquant_names = compatible_by_fingerprint[fingerprint]
                if row["name"] == aquant_names[0]:
                    role = "new_candidate"
                    selected = True
                else:
                    role = "structural_alias"
                    alias_of = _aquant_candidate_id(aquant_names[0])
        planned.append(
            {
                **row,
                "catalog_role": role,
                "selected": selected,
                "structural_alias_of": alias_of,
            }
        )
    return planned


def build_source_idea_audit_v4_1(
    *,
    cycle_id: str,
    candidates: Sequence[Mapping[str, Any]],
    source_receipt: Mapping[str, Any],
    compatibility_contract: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = validate_aquant_source_receipt_v4_1(source_receipt)
    contract = validate_local_compatibility_contract_v4_1(
        compatibility_contract
    )
    planned = _plan_source_ideas(
        candidates=candidates,
        source_receipt=receipt,
        compatibility_contract=contract,
        base_catalog=base_catalog,
    )
    compatible = [row for row in planned if row["compatibility_status"] == "compatible"]
    incompatible = [row for row in planned if row["compatibility_status"] == "incompatible"]
    new_candidates = [row for row in planned if row["catalog_role"] == "new_candidate"]
    aliases = [row for row in planned if row["catalog_role"] == "structural_alias"]
    names = [row["name"] for row in planned]
    payload = {
        "schema_version": SOURCE_IDEA_AUDIT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": _safe_id(cycle_id, "cycle_id"),
        "source_receipt_sha256": receipt["receipt_semantic_sha256"],
        "compatibility_contract_sha256": contract["contract_semantic_sha256"],
        "base_catalog_sha256": _sha(
            base_catalog.get("semantic_sha256"), "base catalog semantic SHA"
        ),
        "total_idea_count": len(planned),
        "compatible_count": len(compatible),
        "incompatible_count": len(incompatible),
        "new_candidate_count": len(new_candidates),
        "structural_alias_count": len(aliases),
        "ordered_names_semantic_sha256": semantic_sha256(names),
        "compatible_ordered_names_semantic_sha256": semantic_sha256(
            [row["name"] for row in compatible]
        ),
        "structural_alias_ordered_names_semantic_sha256": semantic_sha256(
            [row["name"] for row in aliases]
        ),
        "ideas": planned,
        "formal_admission_authority": False,
        "statistics_status": NOT_RUN,
    }
    return validate_source_idea_audit_v4_1(
        _seal(payload, "audit_semantic_sha256")
    )


def _validate_source_idea_row(value: Any, index: int) -> dict[str, Any]:
    label = f"ideas[{index}]"
    row = _exact(value, _SOURCE_IDEA_ROW_FIELDS, label)
    if row["source_index"] != index or type(row["source_index"]) is not int:
        raise FactorGovernanceDiscoveryV4_1Error(
            "source idea indices must be canonical contiguous order"
        )
    name = _safe_name(row["name"], f"{label}.name")
    candidate_id = _text(row["candidate_id"], f"{label}.candidate_id")
    if candidate_id != _aquant_candidate_id(name):
        raise FactorGovernanceDiscoveryV4_1Error(
            "source idea candidate_id is not commit-qualified"
        )
    expression = _text(row["expression"], f"{label}.expression")
    factor_type = _text(row["factor_type"], f"{label}.factor_type")
    source_family = _text(row["source_family"], f"{label}.source_family")
    rationale = _text(row["rationale"], f"{label}.rationale")
    _canonical_float_one(row["direction"], f"{label}.direction")
    if row["direction_origin"] != "expression_signed_ast":
        raise FactorGovernanceDiscoveryV4_1Error(
            "source idea direction origin mismatch"
        )
    status = row["compatibility_status"]
    if status not in {"compatible", "incompatible"}:
        raise FactorGovernanceDiscoveryV4_1Error(
            "source idea compatibility status is invalid"
        )
    reasons = _sorted_distinct_text_list(
        row["incompatibility_reasons"], f"{label}.incompatibility_reasons"
    )
    tree = row["normalized_expression_ast"]
    if tree is not None:
        if not isinstance(tree, Mapping):
            raise FactorGovernanceDiscoveryV4_1Error(
                "normalized expression AST must be an object or null"
            )
        expected_tree = normalize_expression_ast_v4_1(expression)
        if tree != expected_tree:
            raise FactorGovernanceDiscoveryV4_1Error(
                "stored normalized expression AST does not match expression"
            )
        normalized_tree: dict[str, Any] | None = copy.deepcopy(dict(tree))
    else:
        normalized_tree = None
    input_fields = _sorted_distinct_text_list(
        row["input_fields"], f"{label}.input_fields"
    )
    lookback = _positive_int(row["lookback"], f"{label}.lookback")
    role = row["catalog_role"]
    if role not in {"incompatible", "new_candidate", "structural_alias"}:
        raise FactorGovernanceDiscoveryV4_1Error("source idea catalog role is invalid")
    if type(row["selected"]) is not bool:
        raise FactorGovernanceDiscoveryV4_1Error("source idea selected must be boolean")
    fingerprint = row["structural_fingerprint_sha256"]
    alias_of = row["structural_alias_of"]
    if status == "incompatible":
        if not reasons or fingerprint is not None or role != "incompatible" or row["selected"]:
            raise FactorGovernanceDiscoveryV4_1Error(
                "incompatible source idea state is inconsistent"
            )
        if alias_of is not None:
            raise FactorGovernanceDiscoveryV4_1Error(
                "incompatible source idea cannot alias a catalog member"
            )
    else:
        if reasons or tree is None:
            raise FactorGovernanceDiscoveryV4_1Error(
                "compatible source idea must have an AST and no reasons"
            )
        fingerprint = _sha(fingerprint, f"{label}.structural_fingerprint_sha256")
        if role == "new_candidate":
            if row["selected"] is not True or alias_of is not None:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "new candidate selection state is inconsistent"
                )
        elif role == "structural_alias":
            if row["selected"] is not False or type(alias_of) is not str or not alias_of:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "structural alias state is inconsistent"
                )
        else:
            raise FactorGovernanceDiscoveryV4_1Error(
                "compatible source idea must be new_candidate or structural_alias"
            )
    return {
        "source_index": index,
        "candidate_id": candidate_id,
        "name": name,
        "expression": expression,
        "factor_type": factor_type,
        "source_family": source_family,
        "rationale": rationale,
        "direction": 1.0,
        "direction_origin": "expression_signed_ast",
        "compatibility_status": status,
        "incompatibility_reasons": reasons,
        "normalized_expression_ast": normalized_tree,
        "input_fields": input_fields,
        "lookback": lookback,
        "structural_fingerprint_sha256": fingerprint,
        "catalog_role": role,
        "selected": row["selected"],
        "structural_alias_of": alias_of,
    }


def validate_source_idea_audit_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(value, _SOURCE_IDEA_AUDIT_FIELDS, "source idea audit")
    canonical_bytes(payload)
    if payload["schema_version"] != SOURCE_IDEA_AUDIT_SCHEMA_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error("source idea audit schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error("source idea audit protocol mismatch")
    cycle_id = _safe_id(payload["cycle_id"], "cycle_id")
    receipt_sha = _sha(payload["source_receipt_sha256"], "source_receipt_sha256")
    contract_sha = _sha(
        payload["compatibility_contract_sha256"],
        "compatibility_contract_sha256",
    )
    base_catalog_sha = _sha(payload["base_catalog_sha256"], "base_catalog_sha256")
    raw_ideas = payload["ideas"]
    if not isinstance(raw_ideas, list):
        raise FactorGovernanceDiscoveryV4_1Error("ideas must be a list")
    ideas = [_validate_source_idea_row(row, index) for index, row in enumerate(raw_ideas)]
    names = [row["name"] for row in ideas]
    if len(ideas) != EXPECTED_AQUANT_IDEA_COUNT or len(names) != len(set(names)):
        raise FactorGovernanceDiscoveryV4_1Error(
            "source idea audit requires exactly 100 unique ideas"
        )
    compatible = [row for row in ideas if row["compatibility_status"] == "compatible"]
    incompatible = [row for row in ideas if row["compatibility_status"] == "incompatible"]
    new_candidates = [row for row in ideas if row["catalog_role"] == "new_candidate"]
    aliases = [row for row in ideas if row["catalog_role"] == "structural_alias"]
    expected_counts = {
        "total_idea_count": len(ideas),
        "compatible_count": len(compatible),
        "incompatible_count": len(incompatible),
        "new_candidate_count": len(new_candidates),
        "structural_alias_count": len(aliases),
    }
    for field, expected in expected_counts.items():
        if payload[field] != expected or type(payload[field]) is not int:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"source idea audit count mismatch: {field}"
            )
    if len(ideas) != len(compatible) + len(incompatible):
        raise FactorGovernanceDiscoveryV4_1Error("source idea compatibility accounting fails")
    if len(compatible) != len(new_candidates) + len(aliases):
        raise FactorGovernanceDiscoveryV4_1Error("source idea role accounting fails")
    expected_name_hashes = {
        "ordered_names_semantic_sha256": semantic_sha256(names),
        "compatible_ordered_names_semantic_sha256": semantic_sha256(
            [row["name"] for row in compatible]
        ),
        "structural_alias_ordered_names_semantic_sha256": semantic_sha256(
            [row["name"] for row in aliases]
        ),
    }
    for field, expected_hash in expected_name_hashes.items():
        if _sha(payload[field], field) != expected_hash:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"source idea ordered-name SHA mismatch: {field}"
            )
    if payload["formal_admission_authority"] is not False:
        raise FactorGovernanceDiscoveryV4_1Error(
            "source idea audit cannot have formal admission authority"
        )
    if payload["statistics_status"] != NOT_RUN:
        raise FactorGovernanceDiscoveryV4_1Error("source idea statistics must be not_run")
    observed_sha = _sha(payload["audit_semantic_sha256"], "audit_semantic_sha256")
    normalized = {
        "schema_version": SOURCE_IDEA_AUDIT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "source_receipt_sha256": receipt_sha,
        "compatibility_contract_sha256": contract_sha,
        "base_catalog_sha256": base_catalog_sha,
        **expected_counts,
        **expected_name_hashes,
        "ideas": ideas,
        "formal_admission_authority": False,
        "statistics_status": NOT_RUN,
        "audit_semantic_sha256": observed_sha,
    }
    if observed_sha != _self_hash(normalized, "audit_semantic_sha256"):
        raise FactorGovernanceDiscoveryV4_1Error(
            "source idea audit semantic SHA mismatch"
        )
    return normalized


def build_discovery_catalog_v4_1(
    *,
    cycle_id: str,
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    compatibility_contract: Mapping[str, Any],
    source_idea_audit: Mapping[str, Any],
) -> dict[str, Any]:
    ontology = validate_primitive_ontology_v4(base_ontology)
    catalog = validate_candidate_catalog_v4(base_catalog, ontology=ontology)
    if len(catalog["candidates"]) != EXPECTED_BASE_CANDIDATE_COUNT:
        raise FactorGovernanceDiscoveryV4_1Error(
            "base catalog must contain exactly 230 candidates"
        )
    receipt = validate_aquant_source_receipt_v4_1(source_receipt)
    contract = validate_local_compatibility_contract_v4_1(
        compatibility_contract
    )
    audit = validate_source_idea_audit_v4_1(source_idea_audit)
    normalized_cycle_id = _safe_id(cycle_id, "cycle_id")
    if audit["cycle_id"] != normalized_cycle_id:
        raise FactorGovernanceDiscoveryV4_1Error(
            "source idea audit cycle identity mismatch"
        )
    expected_links = {
        "source_receipt_sha256": receipt["receipt_semantic_sha256"],
        "compatibility_contract_sha256": contract[
            "contract_semantic_sha256"
        ],
        "base_catalog_sha256": catalog["semantic_sha256"],
    }
    for field, expected in expected_links.items():
        if audit[field] != expected:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"source idea audit binding mismatch: {field}"
            )
    members = _build_base_discovery_members(
        base_catalog=catalog,
        compatibility_contract_sha256=contract["contract_semantic_sha256"],
    )
    for idea in audit["ideas"]:
        if idea["compatibility_status"] != "compatible":
            continue
        members.append(
            {
                "candidate_id": idea["candidate_id"],
                "origin": "aquant",
                "name": idea["name"],
                "expression": idea["expression"],
                "implementation": "aquant_expression_ast.v1",
                "params": {},
                "direction": 1.0,
                "direction_origin": "expression_signed_ast",
                "factor_type": idea["factor_type"],
                "source_family": idea["source_family"],
                "rationale": idea["rationale"],
                "lookback": idea["lookback"],
                "input_fields": idea["input_fields"],
                "primitive_ids": [],
                "structural_fingerprint_sha256": idea[
                    "structural_fingerprint_sha256"
                ],
                "source_definition_sha256": aquant_source_definition_sha256_v4_1(
                    {
                        "name": idea["name"],
                        "expression": idea["expression"],
                        "factor_type": idea["factor_type"],
                        "source_family": idea["source_family"],
                        "rationale": idea["rationale"],
                    }
                ),
                "catalog_role": idea["catalog_role"],
                "selected": idea["selected"],
                "structural_alias_of": idea["structural_alias_of"],
                "initial_weight": 0.0,
            }
        )
    members.sort(key=lambda item: item["candidate_id"])
    base_members = [row for row in members if row["catalog_role"] == "base_reference"]
    aquant_members = [row for row in members if row["origin"] == "aquant"]
    new_members = [row for row in members if row["catalog_role"] == "new_candidate"]
    aliases = [row for row in members if row["catalog_role"] == "structural_alias"]
    payload = {
        "schema_version": DISCOVERY_CATALOG_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": normalized_cycle_id,
        "base_ontology_sha256": ontology["semantic_sha256"],
        "base_catalog_sha256": catalog["semantic_sha256"],
        "source_receipt_sha256": receipt["receipt_semantic_sha256"],
        "compatibility_contract_sha256": contract["contract_semantic_sha256"],
        "source_idea_audit_sha256": audit["audit_semantic_sha256"],
        "member_count": len(members),
        "selected_count": sum(1 for row in members if row["selected"]),
        "base_reference_count": len(base_members),
        "aquant_compatible_count": len(aquant_members),
        "new_candidate_count": len(new_members),
        "structural_alias_count": len(aliases),
        "base_ordered_definition_pairs_semantic_sha256": semantic_sha256(
            [
                {
                    "name": row["name"],
                    "source_definition_sha256": row["definition_sha256"],
                }
                for row in catalog["candidates"]
            ]
        ),
        "members": members,
        "readiness": READINESS_DISCOVERY,
        "qualification": False,
        "formal_admission_authority": False,
        "initial_weight_policy": "zero_only",
        "statistics_status": NOT_RUN,
        "admission_duplicate_primitive_status": NOT_RUN,
        "high_correlation_dedup_status": NOT_RUN,
    }
    return validate_discovery_catalog_v4_1(
        _seal(payload, "catalog_semantic_sha256")
    )


def _validate_discovery_member(value: Any, index: int) -> dict[str, Any]:
    label = f"members[{index}]"
    row = _exact(value, _DISCOVERY_MEMBER_FIELDS, label)
    candidate_id = _text(row["candidate_id"], f"{label}.candidate_id")
    origin = row["origin"]
    if origin not in {"myquant", "aquant"}:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery member origin must be myquant or aquant"
        )
    name = _safe_name(row["name"], f"{label}.name")
    expression = row["expression"]
    if type(expression) is not str:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery member expression must be a string"
        )
    implementation = _text(row["implementation"], f"{label}.implementation")
    params = _finite_json(row["params"], f"{label}.params")
    if not isinstance(params, dict):
        raise FactorGovernanceDiscoveryV4_1Error("member params must be an object")
    _canonical_float_one(row["direction"], f"{label}.direction")
    direction_origin = row["direction_origin"]
    if direction_origin not in {"bound_base_catalog", "expression_signed_ast"}:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery member direction origin is invalid"
        )
    factor_type = row["factor_type"]
    if factor_type is not None:
        factor_type = _text(factor_type, f"{label}.factor_type")
    source_family = _text(row["source_family"], f"{label}.source_family")
    rationale = row["rationale"]
    if rationale is not None:
        rationale = _text(rationale, f"{label}.rationale")
    lookback = _positive_int(row["lookback"], f"{label}.lookback")
    input_fields = _sorted_distinct_text_list(
        row["input_fields"], f"{label}.input_fields"
    )
    primitive_ids = _sorted_distinct_text_list(
        row["primitive_ids"], f"{label}.primitive_ids"
    )
    fingerprint = _sha(
        row["structural_fingerprint_sha256"],
        f"{label}.structural_fingerprint_sha256",
    )
    source_definition_sha = _sha(
        row["source_definition_sha256"], f"{label}.source_definition_sha256"
    )
    role = row["catalog_role"]
    if role not in {"base_reference", "new_candidate", "structural_alias"}:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery member catalog role is invalid"
        )
    if type(row["selected"]) is not bool:
        raise FactorGovernanceDiscoveryV4_1Error("member selected must be boolean")
    alias_of = row["structural_alias_of"]
    if type(row["initial_weight"]) is not float or row["initial_weight"] != 0.0:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery member initial_weight must be canonical 0.0"
        )
    if origin == "myquant":
        if role != "base_reference" or row["selected"] is not True or alias_of is not None:
            raise FactorGovernanceDiscoveryV4_1Error(
                "myQuant base reference selection state is invalid"
            )
        if direction_origin != "bound_base_catalog" or factor_type is not None or rationale is not None:
            raise FactorGovernanceDiscoveryV4_1Error(
                "myQuant base reference provenance is invalid"
            )
        if not primitive_ids:
            raise FactorGovernanceDiscoveryV4_1Error(
                "myQuant base reference primitive_ids must not be empty"
            )
    else:
        if candidate_id != _aquant_candidate_id(name):
            raise FactorGovernanceDiscoveryV4_1Error(
                "A_quant discovery member candidate_id mismatch"
            )
        if (
            implementation != "aquant_expression_ast.v1"
            or params
            or direction_origin != "expression_signed_ast"
            or factor_type is None
            or rationale is None
            or primitive_ids
        ):
            raise FactorGovernanceDiscoveryV4_1Error(
                "A_quant discovery member provenance is invalid"
            )
        if source_definition_sha != aquant_source_definition_sha256_v4_1(
            {
                "name": name,
                "expression": expression,
                "factor_type": factor_type,
                "source_family": source_family,
                "rationale": rationale,
            }
        ):
            raise FactorGovernanceDiscoveryV4_1Error(
                "A_quant discovery member source definition SHA mismatch"
            )
        if role == "new_candidate":
            if row["selected"] is not True or alias_of is not None:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "new discovery member selection state is invalid"
                )
        elif role == "structural_alias":
            if row["selected"] is not False or type(alias_of) is not str or not alias_of:
                raise FactorGovernanceDiscoveryV4_1Error(
                    "structural alias member selection state is invalid"
                )
        else:
            raise FactorGovernanceDiscoveryV4_1Error(
                "A_quant discovery member cannot be a base reference"
            )
    return {
        "candidate_id": candidate_id,
        "origin": origin,
        "name": name,
        "expression": expression,
        "implementation": implementation,
        "params": params,
        "direction": 1.0,
        "direction_origin": direction_origin,
        "factor_type": factor_type,
        "source_family": source_family,
        "rationale": rationale,
        "lookback": lookback,
        "input_fields": input_fields,
        "primitive_ids": primitive_ids,
        "structural_fingerprint_sha256": fingerprint,
        "source_definition_sha256": source_definition_sha,
        "catalog_role": role,
        "selected": row["selected"],
        "structural_alias_of": alias_of,
        "initial_weight": 0.0,
    }


def validate_discovery_catalog_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(value, _DISCOVERY_CATALOG_FIELDS, "discovery catalog")
    canonical_bytes(payload)
    if payload["schema_version"] != DISCOVERY_CATALOG_SCHEMA_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error("discovery catalog schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error("discovery catalog protocol mismatch")
    cycle_id = _safe_id(payload["cycle_id"], "cycle_id")
    hashes = {
        field: _sha(payload[field], field)
        for field in (
            "base_ontology_sha256",
            "base_catalog_sha256",
            "source_receipt_sha256",
            "compatibility_contract_sha256",
            "source_idea_audit_sha256",
        )
    }
    raw_members = payload["members"]
    if not isinstance(raw_members, list):
        raise FactorGovernanceDiscoveryV4_1Error("members must be a list")
    members = [_validate_discovery_member(row, index) for index, row in enumerate(raw_members)]
    candidate_ids = [row["candidate_id"] for row in members]
    if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery members must be sorted by unique candidate_id"
        )
    base_members = [row for row in members if row["catalog_role"] == "base_reference"]
    aquant_members = [row for row in members if row["origin"] == "aquant"]
    new_members = [row for row in members if row["catalog_role"] == "new_candidate"]
    aliases = [row for row in members if row["catalog_role"] == "structural_alias"]
    expected_counts = {
        "member_count": len(members),
        "selected_count": sum(1 for row in members if row["selected"]),
        "base_reference_count": len(base_members),
        "aquant_compatible_count": len(aquant_members),
        "new_candidate_count": len(new_members),
        "structural_alias_count": len(aliases),
    }
    base_definition_pairs_sha = _sha(
        payload["base_ordered_definition_pairs_semantic_sha256"],
        "base_ordered_definition_pairs_semantic_sha256",
    )
    expected_base_pairs_sha = semantic_sha256(
        [
            {
                "name": row["name"],
                "source_definition_sha256": row["source_definition_sha256"],
            }
            for row in sorted(base_members, key=lambda item: item["name"])
        ]
    )
    if base_definition_pairs_sha != expected_base_pairs_sha:
        raise FactorGovernanceDiscoveryV4_1Error(
            "base ordered definition-pair SHA mismatch"
        )
    for field, expected in expected_counts.items():
        if payload[field] != expected or type(payload[field]) is not int:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"discovery catalog count mismatch: {field}"
            )
    if len(base_members) != EXPECTED_BASE_CANDIDATE_COUNT:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery catalog must retain all 230 base references"
        )
    if len(members) != len(base_members) + len(aquant_members):
        raise FactorGovernanceDiscoveryV4_1Error("discovery catalog origin accounting fails")
    if len(aquant_members) != len(new_members) + len(aliases):
        raise FactorGovernanceDiscoveryV4_1Error("discovery catalog role accounting fails")
    by_id = {row["candidate_id"]: row for row in members}
    for row in aliases:
        target = by_id.get(row["structural_alias_of"])
        if (
            target is None
            or target["selected"] is not True
            or target["structural_fingerprint_sha256"]
            != row["structural_fingerprint_sha256"]
        ):
            raise FactorGovernanceDiscoveryV4_1Error(
                "structural alias target is absent, unselected, or fingerprint-mismatched"
            )
    base_prefix = f"myquant:{hashes['base_catalog_sha256']}:"
    if any(
        row["candidate_id"] != f"{base_prefix}{row['name']}"
        for row in base_members
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "base reference candidate_id is not catalog-qualified"
        )
    expected_constants = {
        "readiness": READINESS_DISCOVERY,
        "qualification": False,
        "formal_admission_authority": False,
        "initial_weight_policy": "zero_only",
        "statistics_status": NOT_RUN,
        "admission_duplicate_primitive_status": NOT_RUN,
        "high_correlation_dedup_status": NOT_RUN,
    }
    for field, expected_constant in expected_constants.items():
        if payload[field] != expected_constant:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"discovery catalog non-formal state mismatch: {field}"
            )
    observed_sha = _sha(payload["catalog_semantic_sha256"], "catalog_semantic_sha256")
    normalized = {
        "schema_version": DISCOVERY_CATALOG_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        **hashes,
        **expected_counts,
        "base_ordered_definition_pairs_semantic_sha256": base_definition_pairs_sha,
        "members": members,
        **expected_constants,
        "catalog_semantic_sha256": observed_sha,
    }
    if observed_sha != _self_hash(normalized, "catalog_semantic_sha256"):
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery catalog semantic SHA mismatch"
        )
    return normalized


_COLLISION_GROUP_FIELDS = frozenset(
    {
        "structural_fingerprint_sha256",
        "member_candidate_ids",
        "selected_candidate_ids",
        "alias_candidate_ids",
    }
)
_STRUCTURAL_COLLISION_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "discovery_catalog_sha256",
        "method",
        "collision_group_count",
        "structural_alias_count",
        "groups",
        "admission_duplicate_primitive_status",
        "high_correlation_dedup_status",
        "formal_admission_authority",
        "audit_semantic_sha256",
    }
)


def build_structural_collision_audit_v4_1(
    *,
    cycle_id: str,
    discovery_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = validate_discovery_catalog_v4_1(discovery_catalog)
    normalized_cycle_id = _safe_id(cycle_id, "cycle_id")
    if catalog["cycle_id"] != normalized_cycle_id:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery catalog cycle identity mismatch"
        )
    by_fingerprint: dict[str, list[dict[str, Any]]] = {}
    for member in catalog["members"]:
        by_fingerprint.setdefault(
            member["structural_fingerprint_sha256"], []
        ).append(member)
    groups: list[dict[str, Any]] = []
    for fingerprint, members in by_fingerprint.items():
        aliases = sorted(
            member["candidate_id"]
            for member in members
            if member["catalog_role"] == "structural_alias"
        )
        if not aliases:
            continue
        groups.append(
            {
                "structural_fingerprint_sha256": fingerprint,
                "member_candidate_ids": sorted(
                    member["candidate_id"] for member in members
                ),
                "selected_candidate_ids": sorted(
                    member["candidate_id"] for member in members if member["selected"]
                ),
                "alias_candidate_ids": aliases,
            }
        )
    groups.sort(key=lambda item: item["structural_fingerprint_sha256"])
    payload = {
        "schema_version": STRUCTURAL_COLLISION_AUDIT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": normalized_cycle_id,
        "discovery_catalog_sha256": catalog["catalog_semantic_sha256"],
        "method": STRUCTURAL_FINGERPRINT_METHOD,
        "collision_group_count": len(groups),
        "structural_alias_count": sum(
            len(group["alias_candidate_ids"]) for group in groups
        ),
        "groups": groups,
        "admission_duplicate_primitive_status": NOT_RUN,
        "high_correlation_dedup_status": NOT_RUN,
        "formal_admission_authority": False,
    }
    return validate_structural_collision_audit_v4_1(
        _seal(payload, "audit_semantic_sha256")
    )


def validate_structural_collision_audit_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(
        value,
        _STRUCTURAL_COLLISION_AUDIT_FIELDS,
        "structural collision audit",
    )
    canonical_bytes(payload)
    if payload["schema_version"] != STRUCTURAL_COLLISION_AUDIT_SCHEMA_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error(
            "structural collision audit schema mismatch"
        )
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error(
            "structural collision audit protocol mismatch"
        )
    cycle_id = _safe_id(payload["cycle_id"], "cycle_id")
    catalog_sha = _sha(
        payload["discovery_catalog_sha256"], "discovery_catalog_sha256"
    )
    if payload["method"] != STRUCTURAL_FINGERPRINT_METHOD:
        raise FactorGovernanceDiscoveryV4_1Error(
            "structural collision method mismatch"
        )
    raw_groups = payload["groups"]
    if not isinstance(raw_groups, list):
        raise FactorGovernanceDiscoveryV4_1Error("collision groups must be a list")
    groups: list[dict[str, Any]] = []
    all_aliases: set[str] = set()
    for index, raw in enumerate(raw_groups):
        row = _exact(raw, _COLLISION_GROUP_FIELDS, f"groups[{index}]")
        fingerprint = _sha(
            row["structural_fingerprint_sha256"],
            f"groups[{index}].structural_fingerprint_sha256",
        )
        members = _sorted_distinct_text_list(
            row["member_candidate_ids"],
            f"groups[{index}].member_candidate_ids",
            allow_empty=False,
        )
        selected = _sorted_distinct_text_list(
            row["selected_candidate_ids"],
            f"groups[{index}].selected_candidate_ids",
            allow_empty=False,
        )
        aliases = _sorted_distinct_text_list(
            row["alias_candidate_ids"],
            f"groups[{index}].alias_candidate_ids",
            allow_empty=False,
        )
        member_set = set(members)
        if not set(selected) < member_set or not set(aliases) < member_set:
            raise FactorGovernanceDiscoveryV4_1Error(
                "collision group selected/alias ids must be proper member subsets"
            )
        if set(selected) & set(aliases):
            raise FactorGovernanceDiscoveryV4_1Error(
                "collision group selected and alias ids must be disjoint"
            )
        if member_set != set(selected) | set(aliases):
            raise FactorGovernanceDiscoveryV4_1Error(
                "collision group membership accounting fails"
            )
        if all_aliases.intersection(aliases):
            raise FactorGovernanceDiscoveryV4_1Error(
                "structural alias appears in multiple collision groups"
            )
        all_aliases.update(aliases)
        groups.append(
            {
                "structural_fingerprint_sha256": fingerprint,
                "member_candidate_ids": members,
                "selected_candidate_ids": selected,
                "alias_candidate_ids": aliases,
            }
        )
    if groups != sorted(groups, key=lambda item: item["structural_fingerprint_sha256"]):
        raise FactorGovernanceDiscoveryV4_1Error(
            "collision groups must be sorted by fingerprint"
        )
    fingerprints = [group["structural_fingerprint_sha256"] for group in groups]
    if len(fingerprints) != len(set(fingerprints)):
        raise FactorGovernanceDiscoveryV4_1Error(
            "collision group fingerprints must be distinct"
        )
    expected_counts = {
        "collision_group_count": len(groups),
        "structural_alias_count": len(all_aliases),
    }
    for field, expected in expected_counts.items():
        if payload[field] != expected or type(payload[field]) is not int:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"structural collision audit count mismatch: {field}"
            )
    if (
        payload["admission_duplicate_primitive_status"] != NOT_RUN
        or payload["high_correlation_dedup_status"] != NOT_RUN
        or payload["formal_admission_authority"] is not False
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "structural collision audit cannot claim formal dedup authority"
        )
    observed_sha = _sha(payload["audit_semantic_sha256"], "audit_semantic_sha256")
    normalized = {
        "schema_version": STRUCTURAL_COLLISION_AUDIT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "discovery_catalog_sha256": catalog_sha,
        "method": STRUCTURAL_FINGERPRINT_METHOD,
        **expected_counts,
        "groups": groups,
        "admission_duplicate_primitive_status": NOT_RUN,
        "high_correlation_dedup_status": NOT_RUN,
        "formal_admission_authority": False,
        "audit_semantic_sha256": observed_sha,
    }
    if observed_sha != _self_hash(normalized, "audit_semantic_sha256"):
        raise FactorGovernanceDiscoveryV4_1Error(
            "structural collision audit semantic SHA mismatch"
        )
    return normalized


_ARTIFACT_DESCRIPTOR_FIELDS = frozenset(
    {"artifact_kind", "byte_sha256", "semantic_sha256"}
)
_PREDECESSOR_BUNDLE_BINDING_FIELDS = frozenset(
    {"filename", "byte_sha256", "semantic_sha256"}
)
_CODE_BINDING_FIELDS = frozenset(
    {"absolute_path", "raw_sha256", "size_bytes"}
)
_DISCOVERY_SOURCE_NODE_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "run_id",
        "state",
        "predecessor_bundle_bindings",
        "predecessor_source_node",
        "predecessor_state",
        "base_ontology",
        "base_catalog",
        "aquant_source_receipt",
        "local_compatibility_contract",
        "source_idea_audit",
        "discovery_catalog",
        "structural_collision_audit",
        "code_bindings",
        "code_bindings_semantic_sha256",
        "holdout_status",
        "readiness",
        "qualification",
        "formal_admission_authority",
        "production_apply_enabled",
        "semantic_sha256",
    }
)


def _descriptor(
    *,
    artifact_kind: str,
    artifact: Mapping[str, Any],
    artifact_byte_sha256: str,
    semantic_sha_field: str,
    canonical_encoding: str = "newline",
) -> dict[str, str]:
    observed_byte_sha = _sha(artifact_byte_sha256, f"{artifact_kind} byte SHA")
    if canonical_encoding == "newline":
        expected_byte_sha = byte_sha256(artifact)
    elif canonical_encoding == "no_newline":
        expected_byte_sha = hashlib.sha256(canonical_bytes(artifact)).hexdigest()
    else:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"unknown canonical encoding for {artifact_kind}"
        )
    if observed_byte_sha != expected_byte_sha:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"{artifact_kind} byte SHA does not match canonical file bytes"
        )
    semantic = _sha(artifact.get(semantic_sha_field), f"{artifact_kind} semantic SHA")
    return {
        "artifact_kind": artifact_kind,
        "byte_sha256": observed_byte_sha,
        "semantic_sha256": semantic,
    }


def _normalize_artifact_descriptor(
    value: Any,
    *,
    expected_kind: str,
) -> dict[str, str]:
    row = _exact(value, _ARTIFACT_DESCRIPTOR_FIELDS, expected_kind)
    if row["artifact_kind"] != expected_kind:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"artifact descriptor kind mismatch: {expected_kind}"
        )
    return {
        "artifact_kind": expected_kind,
        "byte_sha256": _sha(row["byte_sha256"], f"{expected_kind}.byte_sha256"),
        "semantic_sha256": _sha(
            row["semantic_sha256"], f"{expected_kind}.semantic_sha256"
        ),
    }


def _normalize_predecessor_bundle_bindings(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise FactorGovernanceDiscoveryV4_1Error(
            "predecessor_bundle_bindings must be a list"
        )
    bindings: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        row = _exact(
            raw,
            _PREDECESSOR_BUNDLE_BINDING_FIELDS,
            f"predecessor_bundle_bindings[{index}]",
        )
        bindings.append(
            {
                "filename": _text(row["filename"], f"predecessor binding {index} filename"),
                "byte_sha256": _sha(
                    row["byte_sha256"], f"predecessor binding {index} byte SHA"
                ),
                "semantic_sha256": _sha(
                    row["semantic_sha256"],
                    f"predecessor binding {index} semantic SHA",
                ),
            }
        )
    if [row["filename"] for row in bindings] != list(PREDECESSOR_BUNDLE_FILENAMES):
        raise FactorGovernanceDiscoveryV4_1Error(
            "predecessor bundle bindings must be the exact sorted five-file set"
        )
    return bindings


def _normalize_code_bindings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise FactorGovernanceDiscoveryV4_1Error("code_bindings must be a list")
    bindings: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(raw, _CODE_BINDING_FIELDS, f"code_bindings[{index}]")
        absolute_path = _text(
            row["absolute_path"], f"code_bindings[{index}].absolute_path"
        )
        path = PurePosixPath(absolute_path)
        if not path.is_absolute() or ".." in path.parts:
            raise FactorGovernanceDiscoveryV4_1Error(
                "code binding paths must be exact absolute paths"
            )
        bindings.append(
            {
                "absolute_path": absolute_path,
                "raw_sha256": _sha(
                    row["raw_sha256"], f"code_bindings[{index}].raw_sha256"
                ),
                "size_bytes": _positive_int(
                    row["size_bytes"], f"code_bindings[{index}].size_bytes"
                ),
            }
        )
    paths = [row["absolute_path"] for row in bindings]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise FactorGovernanceDiscoveryV4_1Error(
            "code bindings must be sorted by distinct absolute_path"
        )
    if len(bindings) != len(REQUIRED_CODE_BINDING_SUFFIXES):
        raise FactorGovernanceDiscoveryV4_1Error(
            "code bindings must contain exactly the governed files"
        )
    for suffix in REQUIRED_CODE_BINDING_SUFFIXES:
        if sum(path.endswith(suffix) for path in paths) != 1:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"required code binding is missing or duplicated: {suffix}"
            )
    return bindings


def build_discovery_source_node_v4_1(
    *,
    cycle_id: str,
    run_id: str,
    predecessor_bundle_bindings: Sequence[Mapping[str, Any]],
    predecessor_source_node: Mapping[str, Any],
    predecessor_source_node_byte_sha256: str,
    predecessor_state: Mapping[str, Any],
    predecessor_state_byte_sha256: str,
    base_ontology: Mapping[str, Any],
    base_ontology_byte_sha256: str,
    base_catalog: Mapping[str, Any],
    base_catalog_byte_sha256: str,
    aquant_source_receipt: Mapping[str, Any],
    local_compatibility_contract: Mapping[str, Any],
    source_idea_audit: Mapping[str, Any],
    discovery_catalog: Mapping[str, Any],
    structural_collision_audit: Mapping[str, Any],
    code_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_cycle_id = _safe_id(cycle_id, "cycle_id")
    state = validate_cycle_state_v4_1(
        predecessor_state,
        expected_cycle_id=normalized_cycle_id,
        expected_state=PRECOMMITTED,
    )
    source_node = dict(predecessor_source_node)
    if (
        source_node.get("schema_version")
        != "factor-governance-cutoff-source-node.v4.1"
        or source_node.get("cycle_id") != normalized_cycle_id
        or source_node.get("protocol_version") != PROTOCOL_VERSION
        or _SHA256_RE.fullmatch(str(source_node.get("semantic_sha256", ""))) is None
        or source_node.get("semantic_sha256")
        != byte_sha256(
            {
                key: item
                for key, item in source_node.items()
                if key != "semantic_sha256"
            }
        )
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "predecessor cutoff source node is not self-consistent"
        )
    if state["source_chain_node_sha256"] != source_node["semantic_sha256"]:
        raise FactorGovernanceDiscoveryV4_1Error(
            "predecessor state does not bind the cutoff source node"
        )
    predecessor_bindings = _normalize_predecessor_bundle_bindings(
        list(predecessor_bundle_bindings)
    )
    predecessor_source_descriptor = _descriptor(
        artifact_kind="predecessor_cutoff_source_node",
        artifact=source_node,
        artifact_byte_sha256=predecessor_source_node_byte_sha256,
        semantic_sha_field="semantic_sha256",
    )
    predecessor_state_descriptor = _descriptor(
        artifact_kind="predecessor_precommitted_state",
        artifact=state,
        artifact_byte_sha256=predecessor_state_byte_sha256,
        semantic_sha_field="state_semantic_sha256",
    )
    predecessor_by_name = {row["filename"]: row for row in predecessor_bindings}
    if predecessor_by_name["source_chain_node.v4_1.json"] != {
        "filename": "source_chain_node.v4_1.json",
        "byte_sha256": predecessor_source_descriptor["byte_sha256"],
        "semantic_sha256": predecessor_source_descriptor["semantic_sha256"],
    }:
        raise FactorGovernanceDiscoveryV4_1Error(
            "predecessor bundle source-node binding mismatch"
        )
    if predecessor_by_name["cycle_state.precommitted.v4_1.json"] != {
        "filename": "cycle_state.precommitted.v4_1.json",
        "byte_sha256": predecessor_state_descriptor["byte_sha256"],
        "semantic_sha256": predecessor_state_descriptor["semantic_sha256"],
    }:
        raise FactorGovernanceDiscoveryV4_1Error(
            "predecessor bundle state binding mismatch"
        )

    ontology = validate_primitive_ontology_v4(base_ontology)
    base = validate_candidate_catalog_v4(base_catalog, ontology=ontology)
    receipt = validate_aquant_source_receipt_v4_1(aquant_source_receipt)
    contract = validate_local_compatibility_contract_v4_1(
        local_compatibility_contract
    )
    audit = validate_source_idea_audit_v4_1(source_idea_audit)
    catalog = validate_discovery_catalog_v4_1(discovery_catalog)
    collision = validate_structural_collision_audit_v4_1(
        structural_collision_audit
    )
    for artifact_cycle in (audit, catalog, collision):
        if artifact_cycle["cycle_id"] != normalized_cycle_id:
            raise FactorGovernanceDiscoveryV4_1Error(
                "discovery artifact cycle identity mismatch"
            )
    descriptors = {
        "base_ontology": _descriptor(
            artifact_kind="base_ontology",
            artifact=ontology,
            artifact_byte_sha256=base_ontology_byte_sha256,
            semantic_sha_field="semantic_sha256",
            canonical_encoding="no_newline",
        ),
        "base_catalog": _descriptor(
            artifact_kind="base_catalog",
            artifact=base,
            artifact_byte_sha256=base_catalog_byte_sha256,
            semantic_sha_field="semantic_sha256",
            canonical_encoding="no_newline",
        ),
        "aquant_source_receipt": _descriptor(
            artifact_kind="aquant_source_receipt",
            artifact=receipt,
            artifact_byte_sha256=byte_sha256(receipt),
            semantic_sha_field="receipt_semantic_sha256",
        ),
        "local_compatibility_contract": _descriptor(
            artifact_kind="local_compatibility_contract",
            artifact=contract,
            artifact_byte_sha256=byte_sha256(contract),
            semantic_sha_field="contract_semantic_sha256",
        ),
        "source_idea_audit": _descriptor(
            artifact_kind="source_idea_audit",
            artifact=audit,
            artifact_byte_sha256=byte_sha256(audit),
            semantic_sha_field="audit_semantic_sha256",
        ),
        "discovery_catalog": _descriptor(
            artifact_kind="discovery_catalog",
            artifact=catalog,
            artifact_byte_sha256=byte_sha256(catalog),
            semantic_sha_field="catalog_semantic_sha256",
        ),
        "structural_collision_audit": _descriptor(
            artifact_kind="structural_collision_audit",
            artifact=collision,
            artifact_byte_sha256=byte_sha256(collision),
            semantic_sha_field="audit_semantic_sha256",
        ),
    }
    normalized_code_bindings = _normalize_code_bindings(list(code_bindings))
    payload = {
        "schema_version": DISCOVERY_SOURCE_NODE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": normalized_cycle_id,
        "run_id": _safe_id(run_id, "run_id"),
        "state": DISCOVERY,
        "predecessor_bundle_bindings": predecessor_bindings,
        "predecessor_source_node": predecessor_source_descriptor,
        "predecessor_state": predecessor_state_descriptor,
        **descriptors,
        "code_bindings": normalized_code_bindings,
        "code_bindings_semantic_sha256": semantic_sha256(
            normalized_code_bindings
        ),
        "holdout_status": HOLDOUT_SEALED_NOT_APPENDED,
        "readiness": READINESS_DISCOVERY,
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
    }
    return validate_discovery_source_node_v4_1(
        _seal(payload, "semantic_sha256")
    )


def validate_discovery_source_node_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(value, _DISCOVERY_SOURCE_NODE_FIELDS, "discovery source node")
    canonical_bytes(payload)
    if payload["schema_version"] != DISCOVERY_SOURCE_NODE_SCHEMA_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error("discovery source node schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error("discovery source node protocol mismatch")
    cycle_id = _safe_id(payload["cycle_id"], "cycle_id")
    run_id = _safe_id(payload["run_id"], "run_id")
    if payload["state"] != DISCOVERY:
        raise FactorGovernanceDiscoveryV4_1Error("discovery source node state mismatch")
    predecessor_bindings = _normalize_predecessor_bundle_bindings(
        payload["predecessor_bundle_bindings"]
    )
    descriptor_kinds = {
        "predecessor_source_node": "predecessor_cutoff_source_node",
        "predecessor_state": "predecessor_precommitted_state",
        "base_ontology": "base_ontology",
        "base_catalog": "base_catalog",
        "aquant_source_receipt": "aquant_source_receipt",
        "local_compatibility_contract": "local_compatibility_contract",
        "source_idea_audit": "source_idea_audit",
        "discovery_catalog": "discovery_catalog",
        "structural_collision_audit": "structural_collision_audit",
    }
    descriptors = {
        field: _normalize_artifact_descriptor(payload[field], expected_kind=kind)
        for field, kind in descriptor_kinds.items()
    }
    predecessor_by_name = {row["filename"]: row for row in predecessor_bindings}
    if predecessor_by_name["source_chain_node.v4_1.json"]["byte_sha256"] != descriptors[
        "predecessor_source_node"
    ]["byte_sha256"] or predecessor_by_name["source_chain_node.v4_1.json"][
        "semantic_sha256"
    ] != descriptors["predecessor_source_node"]["semantic_sha256"]:
        raise FactorGovernanceDiscoveryV4_1Error(
            "source node predecessor bundle source descriptor mismatch"
        )
    if predecessor_by_name["cycle_state.precommitted.v4_1.json"]["byte_sha256"] != descriptors[
        "predecessor_state"
    ]["byte_sha256"] or predecessor_by_name["cycle_state.precommitted.v4_1.json"][
        "semantic_sha256"
    ] != descriptors["predecessor_state"]["semantic_sha256"]:
        raise FactorGovernanceDiscoveryV4_1Error(
            "source node predecessor bundle state descriptor mismatch"
        )
    code_bindings = _normalize_code_bindings(payload["code_bindings"])
    code_bindings_sha = _sha(
        payload["code_bindings_semantic_sha256"],
        "code_bindings_semantic_sha256",
    )
    if code_bindings_sha != semantic_sha256(code_bindings):
        raise FactorGovernanceDiscoveryV4_1Error(
            "code bindings semantic SHA mismatch"
        )
    expected_constants = {
        "holdout_status": HOLDOUT_SEALED_NOT_APPENDED,
        "readiness": READINESS_DISCOVERY,
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
    }
    for field, expected in expected_constants.items():
        if payload[field] != expected:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"discovery source node non-formal state mismatch: {field}"
            )
    observed_sha = _sha(payload["semantic_sha256"], "semantic_sha256")
    normalized = {
        "schema_version": DISCOVERY_SOURCE_NODE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "run_id": run_id,
        "state": DISCOVERY,
        "predecessor_bundle_bindings": predecessor_bindings,
        **descriptors,
        "code_bindings": code_bindings,
        "code_bindings_semantic_sha256": code_bindings_sha,
        **expected_constants,
        "semantic_sha256": observed_sha,
    }
    if observed_sha != _self_hash(normalized, "semantic_sha256"):
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery source node semantic SHA mismatch"
        )
    return normalized


def build_discovery_cycle_state_v4_1(
    *,
    predecessor_state: Mapping[str, Any],
    predecessor_state_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    cycle_id: str,
    cycle_root_sha256: str,
    discovery_source_node: Mapping[str, Any],
) -> dict[str, Any]:
    source_node = validate_discovery_source_node_v4_1(discovery_source_node)
    if source_node["cycle_id"] != _safe_id(cycle_id, "cycle_id"):
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery source node cycle identity mismatch"
        )
    return build_next_cycle_state_v4_1(
        predecessor=predecessor_state,
        predecessor_byte_sha256=predecessor_state_byte_sha256,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=expected_predecessor_semantic_sha256,
        cycle_id=cycle_id,
        cycle_root_sha256=cycle_root_sha256,
        next_state=DISCOVERY,
        source_chain_node_sha256=source_node["semantic_sha256"],
    )


_FILE_BINDING_FIELDS = frozenset(
    {
        "filename",
        "byte_sha256",
        "semantic_sha256",
        "size_bytes",
        "mode",
        "uid",
        "nlink",
    }
)
_DISCOVERY_READBACK_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "run_id",
        "readiness",
        "qualification",
        "formal_admission_authority",
        "production_apply_enabled",
        "holdout_status",
        "measurement_status",
        "blockers",
        "artifact_bindings",
        "side_effects",
        "report_semantic_sha256",
    }
)


def _normalize_file_bindings(
    value: Any,
    *,
    require_sorted: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise FactorGovernanceDiscoveryV4_1Error(
            "artifact_bindings must be a list"
        )
    bindings: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(raw, _FILE_BINDING_FIELDS, f"artifact_bindings[{index}]")
        if row["mode"] != 0o600 or type(row["mode"]) is not int:
            raise FactorGovernanceDiscoveryV4_1Error(
                "discovery artifact file mode must be integer 384"
            )
        if row["nlink"] != 1 or type(row["nlink"]) is not int:
            raise FactorGovernanceDiscoveryV4_1Error(
                "discovery artifact file nlink must be one"
            )
        bindings.append(
            {
                "filename": _text(row["filename"], f"artifact_bindings[{index}].filename"),
                "byte_sha256": _sha(
                    row["byte_sha256"], f"artifact_bindings[{index}].byte_sha256"
                ),
                "semantic_sha256": _sha(
                    row["semantic_sha256"],
                    f"artifact_bindings[{index}].semantic_sha256",
                ),
                "size_bytes": _positive_int(
                    row["size_bytes"], f"artifact_bindings[{index}].size_bytes"
                ),
                "mode": 0o600,
                "uid": _nonnegative_int(row["uid"], f"artifact_bindings[{index}].uid"),
                "nlink": 1,
            }
        )
    ordered = sorted(bindings, key=lambda item: item["filename"])
    if [row["filename"] for row in ordered] != sorted(PRE_READBACK_ARTIFACT_FILENAMES):
        raise FactorGovernanceDiscoveryV4_1Error(
            "artifact bindings must be the exact sorted first-seven file set"
        )
    if require_sorted and bindings != ordered:
        raise FactorGovernanceDiscoveryV4_1Error(
            "stored artifact bindings must be sorted by filename"
        )
    return ordered


def _normalize_side_effects(value: Any) -> dict[str, bool]:
    expected_fields = frozenset(SIDE_EFFECT_FIELDS)
    payload = _exact(value, expected_fields, "side_effects")
    if any(payload[field] is not False for field in SIDE_EFFECT_FIELDS):
        raise FactorGovernanceDiscoveryV4_1Error(
            "all discovery side effects must remain false"
        )
    return {field: False for field in SIDE_EFFECT_FIELDS}


def build_discovery_readback_report_v4_1(
    *,
    cycle_id: str,
    run_id: str,
    artifact_bindings: Sequence[Mapping[str, Any]],
    side_effects: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_side_effects = _normalize_side_effects(
        dict(side_effects)
        if side_effects is not None
        else {field: False for field in SIDE_EFFECT_FIELDS}
    )
    payload = {
        "schema_version": DISCOVERY_READBACK_REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": _safe_id(cycle_id, "cycle_id"),
        "run_id": _safe_id(run_id, "run_id"),
        "readiness": READINESS_DISCOVERY,
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
        "holdout_status": HOLDOUT_SEALED_NOT_APPENDED,
        "measurement_status": {
            field: NOT_RUN for field in MEASUREMENT_STATUS_FIELDS
        },
        "blockers": list(DISCOVERY_BLOCKERS),
        "artifact_bindings": _normalize_file_bindings(
            list(artifact_bindings), require_sorted=False
        ),
        "side_effects": normalized_side_effects,
    }
    return validate_discovery_readback_report_v4_1(
        _seal(payload, "report_semantic_sha256")
    )


def validate_discovery_readback_report_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(
        value,
        _DISCOVERY_READBACK_REPORT_FIELDS,
        "discovery readback report",
    )
    canonical_bytes(payload)
    if payload["schema_version"] != DISCOVERY_READBACK_REPORT_SCHEMA_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery readback report schema mismatch"
        )
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery readback report protocol mismatch"
        )
    cycle_id = _safe_id(payload["cycle_id"], "cycle_id")
    run_id = _safe_id(payload["run_id"], "run_id")
    expected_constants = {
        "readiness": READINESS_DISCOVERY,
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
        "holdout_status": HOLDOUT_SEALED_NOT_APPENDED,
    }
    for field, expected in expected_constants.items():
        if payload[field] != expected:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"discovery readback non-formal state mismatch: {field}"
            )
    measurement = _exact(
        payload["measurement_status"],
        frozenset(MEASUREMENT_STATUS_FIELDS),
        "measurement_status",
    )
    if any(measurement[field] != NOT_RUN for field in MEASUREMENT_STATUS_FIELDS):
        raise FactorGovernanceDiscoveryV4_1Error(
            "all discovery measurements must remain not_run"
        )
    if payload["blockers"] != list(DISCOVERY_BLOCKERS):
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery readback blockers mismatch"
        )
    bindings = _normalize_file_bindings(
        payload["artifact_bindings"], require_sorted=True
    )
    side_effects = _normalize_side_effects(payload["side_effects"])
    observed_sha = _sha(
        payload["report_semantic_sha256"], "report_semantic_sha256"
    )
    normalized = {
        "schema_version": DISCOVERY_READBACK_REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "run_id": run_id,
        **expected_constants,
        "measurement_status": {
            field: NOT_RUN for field in MEASUREMENT_STATUS_FIELDS
        },
        "blockers": list(DISCOVERY_BLOCKERS),
        "artifact_bindings": bindings,
        "side_effects": side_effects,
        "report_semantic_sha256": observed_sha,
    }
    if observed_sha != _self_hash(normalized, "report_semantic_sha256"):
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery readback report semantic SHA mismatch"
        )
    return normalized


SELF_HASH_FIELD_BY_FILENAME = {
    AQUANT_SOURCE_RECEIPT_FILENAME: "receipt_semantic_sha256",
    SOURCE_IDEA_AUDIT_FILENAME: "audit_semantic_sha256",
    LOCAL_COMPATIBILITY_CONTRACT_FILENAME: "contract_semantic_sha256",
    DISCOVERY_CATALOG_FILENAME: "catalog_semantic_sha256",
    STRUCTURAL_COLLISION_AUDIT_FILENAME: "audit_semantic_sha256",
    DISCOVERY_SOURCE_NODE_FILENAME: "semantic_sha256",
    DISCOVERY_CYCLE_STATE_FILENAME: "state_semantic_sha256",
    DISCOVERY_READBACK_REPORT_FILENAME: "report_semantic_sha256",
}


def validate_discovery_artifact_v4_1(
    filename: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Dispatch an exact canonical filename to its strict pure validator."""

    validators = {
        AQUANT_SOURCE_RECEIPT_FILENAME: validate_aquant_source_receipt_v4_1,
        SOURCE_IDEA_AUDIT_FILENAME: validate_source_idea_audit_v4_1,
        LOCAL_COMPATIBILITY_CONTRACT_FILENAME: (
            validate_local_compatibility_contract_v4_1
        ),
        DISCOVERY_CATALOG_FILENAME: validate_discovery_catalog_v4_1,
        STRUCTURAL_COLLISION_AUDIT_FILENAME: (
            validate_structural_collision_audit_v4_1
        ),
        DISCOVERY_SOURCE_NODE_FILENAME: validate_discovery_source_node_v4_1,
        DISCOVERY_CYCLE_STATE_FILENAME: lambda artifact: validate_cycle_state_v4_1(
            artifact, expected_state=DISCOVERY
        ),
        DISCOVERY_READBACK_REPORT_FILENAME: (
            validate_discovery_readback_report_v4_1
        ),
    }
    if filename not in validators:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"unsupported discovery artifact filename: {filename}"
        )
    try:
        return validators[filename](value)
    except FactorGovernanceDiscoveryV4_1Error:
        raise
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceDiscoveryV4_1Error(
            f"discovery artifact validation failed: {filename}: {exc}"
        ) from exc


def _expected_aquant_member_from_idea(idea: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": idea["candidate_id"],
        "origin": "aquant",
        "name": idea["name"],
        "expression": idea["expression"],
        "implementation": "aquant_expression_ast.v1",
        "params": {},
        "direction": 1.0,
        "direction_origin": "expression_signed_ast",
        "factor_type": idea["factor_type"],
        "source_family": idea["source_family"],
        "rationale": idea["rationale"],
        "lookback": idea["lookback"],
        "input_fields": idea["input_fields"],
        "primitive_ids": [],
        "structural_fingerprint_sha256": idea[
            "structural_fingerprint_sha256"
        ],
        "source_definition_sha256": aquant_source_definition_sha256_v4_1(
            {
                "name": idea["name"],
                "expression": idea["expression"],
                "factor_type": idea["factor_type"],
                "source_family": idea["source_family"],
                "rationale": idea["rationale"],
            }
        ),
        "catalog_role": idea["catalog_role"],
        "selected": idea["selected"],
        "structural_alias_of": idea["structural_alias_of"],
        "initial_weight": 0.0,
    }


def validate_discovery_bundle_v4_1(
    values: Mapping[str, Mapping[str, Any]],
    *,
    base_ontology: Mapping[str, Any] | None = None,
    base_catalog: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Cross-validate all eight artifacts and return normalized filename values."""

    if not isinstance(values, Mapping) or set(values) != set(CANONICAL_ARTIFACT_FILENAMES):
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery bundle must contain exactly the eight canonical artifacts"
        )
    normalized = {
        filename: validate_discovery_artifact_v4_1(filename, values[filename])
        for filename in CANONICAL_ARTIFACT_FILENAMES
    }
    receipt = normalized[AQUANT_SOURCE_RECEIPT_FILENAME]
    audit = normalized[SOURCE_IDEA_AUDIT_FILENAME]
    contract = normalized[LOCAL_COMPATIBILITY_CONTRACT_FILENAME]
    catalog = normalized[DISCOVERY_CATALOG_FILENAME]
    collision = normalized[STRUCTURAL_COLLISION_AUDIT_FILENAME]
    source_node = normalized[DISCOVERY_SOURCE_NODE_FILENAME]
    state = normalized[DISCOVERY_CYCLE_STATE_FILENAME]
    report = normalized[DISCOVERY_READBACK_REPORT_FILENAME]

    cycle_id = audit["cycle_id"]
    if any(
        artifact["cycle_id"] != cycle_id
        for artifact in (catalog, collision, source_node, state, report)
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery bundle cycle identities differ"
        )
    if source_node["run_id"] != report["run_id"]:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery bundle run identities differ"
        )
    direct_links = {
        "source_receipt_sha256": receipt["receipt_semantic_sha256"],
        "compatibility_contract_sha256": contract[
            "contract_semantic_sha256"
        ],
        "base_catalog_sha256": catalog["base_catalog_sha256"],
    }
    for field, expected in direct_links.items():
        if audit[field] != expected:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"source idea audit cross-binding mismatch: {field}"
            )
    if receipt["ordered_names_semantic_sha256"] != audit[
        "ordered_names_semantic_sha256"
    ]:
        raise FactorGovernanceDiscoveryV4_1Error(
            "receipt and audit ordered idea names differ"
        )
    catalog_links = {
        "source_receipt_sha256": receipt["receipt_semantic_sha256"],
        "compatibility_contract_sha256": contract[
            "contract_semantic_sha256"
        ],
        "source_idea_audit_sha256": audit["audit_semantic_sha256"],
    }
    for field, expected in catalog_links.items():
        if catalog[field] != expected:
            raise FactorGovernanceDiscoveryV4_1Error(
                f"discovery catalog cross-binding mismatch: {field}"
            )

    base_members = [
        member
        for member in catalog["members"]
        if member["catalog_role"] == "base_reference"
    ]
    base_by_fingerprint: dict[str, list[str]] = {}
    for member in base_members:
        base_by_fingerprint.setdefault(
            member["structural_fingerprint_sha256"], []
        ).append(member["candidate_id"])
    for member_ids in base_by_fingerprint.values():
        member_ids.sort()
    compatibility_rows: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
    compatible_names_by_fingerprint: dict[str, list[str]] = {}
    for idea in audit["ideas"]:
        assessment = assess_local_compatibility_v4_1(
            idea["expression"], contract
        )
        for field in (
            "status",
            "reasons",
            "normalized_expression_ast",
            "input_fields",
            "lookback",
        ):
            stored_field = {
                "status": "compatibility_status",
                "reasons": "incompatibility_reasons",
            }.get(field, field)
            if idea[stored_field] != assessment[field]:
                raise FactorGovernanceDiscoveryV4_1Error(
                    f"source idea compatibility was not recomputed: {idea['name']}"
                )
        fingerprint: str | None = None
        if assessment["status"] == "compatible":
            fingerprint = expression_structural_fingerprint_sha256_v4_1(
                idea["expression"],
                compatibility_contract_sha256=contract[
                    "contract_semantic_sha256"
                ],
            )
            if idea["structural_fingerprint_sha256"] != fingerprint:
                raise FactorGovernanceDiscoveryV4_1Error(
                    f"source idea fingerprint mismatch: {idea['name']}"
                )
            compatible_names_by_fingerprint.setdefault(fingerprint, []).append(
                idea["name"]
            )
        compatibility_rows.append((idea, assessment, fingerprint))
    for names in compatible_names_by_fingerprint.values():
        names.sort()
    for idea, assessment, fingerprint in compatibility_rows:
        if assessment["status"] == "incompatible":
            expected_role, expected_selected, expected_alias = (
                "incompatible",
                False,
                None,
            )
        elif base_by_fingerprint.get(str(fingerprint)):
            expected_role, expected_selected, expected_alias = (
                "structural_alias",
                False,
                base_by_fingerprint[str(fingerprint)][0],
            )
        elif idea["name"] == compatible_names_by_fingerprint[str(fingerprint)][0]:
            expected_role, expected_selected, expected_alias = (
                "new_candidate",
                True,
                None,
            )
        else:
            expected_role, expected_selected, expected_alias = (
                "structural_alias",
                False,
                _aquant_candidate_id(
                    compatible_names_by_fingerprint[str(fingerprint)][0]
                ),
            )
        if (
            idea["catalog_role"] != expected_role
            or idea["selected"] is not expected_selected
            or idea["structural_alias_of"] != expected_alias
        ):
            raise FactorGovernanceDiscoveryV4_1Error(
                f"source idea structural role was not recomputed: {idea['name']}"
            )

    actual_aquant_members = {
        member["candidate_id"]: member
        for member in catalog["members"]
        if member["origin"] == "aquant"
    }
    expected_aquant_members = {
        idea["candidate_id"]: _expected_aquant_member_from_idea(idea)
        for idea in audit["ideas"]
        if idea["compatibility_status"] == "compatible"
    }
    if actual_aquant_members != expected_aquant_members:
        raise FactorGovernanceDiscoveryV4_1Error(
            "discovery catalog A_quant members differ from the source idea audit"
        )
    expected_collision = build_structural_collision_audit_v4_1(
        cycle_id=cycle_id, discovery_catalog=catalog
    )
    if collision != expected_collision:
        raise FactorGovernanceDiscoveryV4_1Error(
            "structural collision audit differs from the discovery catalog"
        )

    descriptor_expectations = {
        "aquant_source_receipt": (
            receipt,
            "receipt_semantic_sha256",
        ),
        "local_compatibility_contract": (
            contract,
            "contract_semantic_sha256",
        ),
        "source_idea_audit": (audit, "audit_semantic_sha256"),
        "discovery_catalog": (catalog, "catalog_semantic_sha256"),
        "structural_collision_audit": (collision, "audit_semantic_sha256"),
    }
    for field, (artifact, semantic_field) in descriptor_expectations.items():
        descriptor = source_node[field]
        if (
            descriptor["byte_sha256"] != byte_sha256(artifact)
            or descriptor["semantic_sha256"] != artifact[semantic_field]
        ):
            raise FactorGovernanceDiscoveryV4_1Error(
                f"discovery source-node artifact binding mismatch: {field}"
            )
    if (
        source_node["base_ontology"]["semantic_sha256"]
        != catalog["base_ontology_sha256"]
        or source_node["base_catalog"]["semantic_sha256"]
        != catalog["base_catalog_sha256"]
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "source-node base ontology/catalog semantic links mismatch"
        )
    if (
        state["source_chain_node_sha256"] != source_node["semantic_sha256"]
        or state["predecessor"]["byte_sha256"]
        != source_node["predecessor_state"]["byte_sha256"]
        or state["predecessor"]["semantic_sha256"]
        != source_node["predecessor_state"]["semantic_sha256"]
    ):
        raise FactorGovernanceDiscoveryV4_1Error(
            "DISCOVERY state source/predecessor CAS bindings mismatch"
        )

    if (base_ontology is None) != (base_catalog is None):
        raise FactorGovernanceDiscoveryV4_1Error(
            "base_ontology and base_catalog must be supplied together"
        )
    if base_ontology is not None and base_catalog is not None:
        ontology = validate_primitive_ontology_v4(base_ontology)
        base = validate_candidate_catalog_v4(base_catalog, ontology=ontology)
        if (
            source_node["base_ontology"]["byte_sha256"]
            != hashlib.sha256(canonical_bytes(ontology)).hexdigest()
            or source_node["base_ontology"]["semantic_sha256"]
            != ontology["semantic_sha256"]
            or source_node["base_catalog"]["byte_sha256"]
            != hashlib.sha256(canonical_bytes(base)).hexdigest()
            or source_node["base_catalog"]["semantic_sha256"]
            != base["semantic_sha256"]
        ):
            raise FactorGovernanceDiscoveryV4_1Error(
                "source-node native base artifact bindings mismatch"
            )
        expected_catalog = build_discovery_catalog_v4_1(
            cycle_id=cycle_id,
            base_ontology=ontology,
            base_catalog=base,
            source_receipt=receipt,
            compatibility_contract=contract,
            source_idea_audit=audit,
        )
        if catalog != expected_catalog:
            raise FactorGovernanceDiscoveryV4_1Error(
                "discovery catalog differs from the bound base definitions"
            )

    report_bindings = {
        row["filename"]: row for row in report["artifact_bindings"]
    }
    for filename in PRE_READBACK_ARTIFACT_FILENAMES:
        artifact = normalized[filename]
        binding = report_bindings[filename]
        semantic_field = SELF_HASH_FIELD_BY_FILENAME[filename]
        if (
            binding["byte_sha256"] != byte_sha256(artifact)
            or binding["semantic_sha256"] != artifact[semantic_field]
            or binding["size_bytes"] != len(canonical_file_bytes(artifact))
        ):
            raise FactorGovernanceDiscoveryV4_1Error(
                f"readback report artifact binding mismatch: {filename}"
            )
    return normalized


def validate_discovery_bundle_values_v4_1(
    values: Mapping[str, Mapping[str, Any]],
    *,
    base_ontology: Mapping[str, Any] | None = None,
    base_catalog: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Long-form alias for :func:`validate_discovery_bundle_v4_1`."""

    return validate_discovery_bundle_v4_1(
        values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
    )
