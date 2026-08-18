"""Finite installed Factor implementations used by trusted source replay."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import textwrap
from typing import Any, Final

import pandas as pd

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes

from .bootstrap import (
    BLEND_W80,
    CANONICAL_PARQUET,
    LOW_DOLLAR_VOLUME,
    bootstrap_factor_definitions,
    compute_bootstrap_signals,
    required_source_roles_for_factor,
)
from .common import require_sha256
from .errors import FactorGovernanceError


@dataclass(frozen=True)
class InstalledFactorImplementation:
    """One closed implementation entrypoint in the installed release."""

    factor_id: str
    implementation_id: str
    module_name: str
    qualified_name: str
    family: str
    primitive: str
    direction: str
    formula: str
    normalized_expression: str
    parameters_json: str
    input_fields: tuple[str, ...]
    required_source_roles: tuple[str, ...]


def _low_dollar_volume(
    frames: Mapping[str, pd.DataFrame],
) -> pd.Series:
    return compute_bootstrap_signals(frames, source_format=CANONICAL_PARQUET)[LOW_DOLLAR_VOLUME]


def _blend_w80(
    frames: Mapping[str, pd.DataFrame],
) -> pd.Series:
    return compute_bootstrap_signals(frames, source_format=CANONICAL_PARQUET)[BLEND_W80]


_ENTRYPOINTS: Final = {
    LOW_DOLLAR_VOLUME: _low_dollar_volume,
    BLEND_W80: _blend_w80,
}

_PRIMITIVES: Final = {
    LOW_DOLLAR_VOLUME: "low_dollar_volume",
    BLEND_W80: "volstab_momentum_amihud_blend",
}

_NORMALIZED_EXPRESSIONS: Final = {
    LOW_DOLLAR_VOLUME: (
        '{"input":"amount","operator":"NEGATIVE_LOG_ROLLING_MEAN",' '"window_open_sessions":5}'
    ),
    BLEND_W80: (
        '{"amihud":{"amount_field":"amount","price_field":"adj_close",'
        '"window_open_sessions":5},"inner_blend":{"amihud_weight":'
        '"0.400000000000","momentum_weight":"0.600000000000"},"momentum":'
        '{"price_field":"adj_close","window_open_sessions":90},"operator":'
        '"RANKED_OUTER_INNER_BLEND","outer_blend":{"inner_weight":'
        '"0.200000000000","volume_stability_weight":"0.800000000000"},'
        '"volume_stability":{"base_open_sessions":19,'
        '"smoothing_open_sessions":2,"volume_field":"vol"}}'
    ),
}


def _definition_by_factor_id() -> dict[str, dict[str, object]]:
    return {
        str(row["factor_id"]): row
        for row in bootstrap_factor_definitions()
        if row["factor_id"] in _ENTRYPOINTS
    }


def _installed_implementation(factor_id: str) -> InstalledFactorImplementation:
    definitions = _definition_by_factor_id()
    definition = definitions.get(factor_id)
    entrypoint = _ENTRYPOINTS.get(factor_id)
    if definition is None or entrypoint is None:
        raise FactorGovernanceError("factor implementation is not installed")
    input_fields = definition["input_fields"]
    if not isinstance(input_fields, list) or not all(type(value) is str for value in input_fields):
        raise FactorGovernanceError("installed Factor input fields are invalid")
    return InstalledFactorImplementation(
        factor_id=factor_id,
        implementation_id=f"installed-{factor_id}",
        module_name=entrypoint.__module__,
        qualified_name=entrypoint.__qualname__,
        family=str(definition["family"]),
        primitive=_PRIMITIVES[factor_id],
        direction=str(definition["direction"]),
        formula=str(definition["formula"]),
        normalized_expression=_NORMALIZED_EXPRESSIONS[factor_id],
        parameters_json=canonical_json_bytes(definition["parameters"]).decode("utf-8"),
        input_fields=tuple(input_fields),
        required_source_roles=required_source_roles_for_factor(factor_id),
    )


def implementation_code_sha256(factor_id: str) -> str:
    entrypoint = _ENTRYPOINTS.get(factor_id)
    if entrypoint is None:
        raise FactorGovernanceError("factor implementation is not installed")
    try:
        parsed = ast.parse(textwrap.dedent(inspect.getsource(entrypoint)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise FactorGovernanceError(
            "installed Factor implementation source is unavailable"
        ) from exc
    nodes = [
        node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(nodes) != 1 or nodes[0].name != entrypoint.__name__:
        raise FactorGovernanceError("installed Factor implementation AST is ambiguous")
    body = {
        "domain": "myquant-python-ast-entrypoint",
        "module_name": entrypoint.__module__,
        "qualified_name": entrypoint.__qualname__,
        "node": ast.dump(nodes[0], annotate_fields=True, include_attributes=False),
    }
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def installed_implementation_rows(
    *,
    implementation_component_refs: Mapping[str, Mapping[str, Any]],
    factor_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return exact installed rows; no runtime registration or plugin discovery."""

    requested = set(_ENTRYPOINTS if factor_ids is None else factor_ids)
    if not requested or not requested.issubset(_ENTRYPOINTS):
        raise FactorGovernanceError("requested Factor implementation set is not installed")
    if set(implementation_component_refs) != requested:
        raise FactorGovernanceError("implementation component refs are not exact")
    from .common import validate_artifact_ref

    rows: list[dict[str, Any]] = []
    for factor_id in sorted(requested, key=lambda value: value.encode("utf-8")):
        implementation = _installed_implementation(factor_id)
        rows.append(
            {
                "factor_id": implementation.factor_id,
                "implementation_id": implementation.implementation_id,
                "implementation_component_ref": validate_artifact_ref(
                    dict(implementation_component_refs[factor_id]),
                    label=f"implementation_component_refs[{factor_id}]",
                    expected_kind="system.installed_component_manifest",
                ),
                "module_name": implementation.module_name,
                "qualified_name": implementation.qualified_name,
                "code_sha256": implementation_code_sha256(factor_id),
                "family": implementation.family,
                "primitive": implementation.primitive,
                "direction": implementation.direction,
                "formula": implementation.formula,
                "normalized_expression": implementation.normalized_expression,
                "parameters_json": implementation.parameters_json,
                "input_fields": list(implementation.input_fields),
                "required_source_roles": list(implementation.required_source_roles),
            }
        )
    return rows


def installed_semantic_row(factor_id: str) -> dict[str, Any]:
    """Return the exact semantic identity independent of its component ref."""

    implementation = _installed_implementation(factor_id)
    return {
        "factor_id": implementation.factor_id,
        "implementation_id": implementation.implementation_id,
        "module_name": implementation.module_name,
        "qualified_name": implementation.qualified_name,
        "code_sha256": implementation_code_sha256(factor_id),
        "family": implementation.family,
        "primitive": implementation.primitive,
        "direction": implementation.direction,
        "formula": implementation.formula,
        "normalized_expression": implementation.normalized_expression,
        "parameters_json": implementation.parameters_json,
        "input_fields": list(implementation.input_fields),
        "required_source_roles": list(implementation.required_source_roles),
    }


def validate_candidate_implementation(
    candidate: Mapping[str, object],
    *,
    installed_row: Mapping[str, object],
) -> None:
    """Bind a preregistered candidate to one exact installed entrypoint."""

    factor_id = str(candidate.get("factor_id"))
    if factor_id != installed_row.get("factor_id"):
        raise FactorGovernanceError("candidate factor implementation binding differs")
    expected = installed_semantic_row(factor_id)
    fields = (
        "implementation_id",
        "family",
        "primitive",
        "direction",
        "formula",
        "normalized_expression",
        "parameters_json",
        "input_fields",
    )
    if any(candidate.get(field) != expected[field] for field in fields) or candidate.get(
        "implementation_sha256"
    ) != installed_row.get("code_sha256"):
        raise FactorGovernanceError("candidate does not match the installed implementation")
    component_ref = candidate.get("implementation_component_ref")
    if component_ref != installed_row.get("implementation_component_ref"):
        raise FactorGovernanceError("candidate implementation component differs")
    for label in ("normalized_expression", "parameters_json"):
        raw = candidate.get(label)
        if type(raw) is not str:
            raise FactorGovernanceError(f"candidate {label} is invalid")
        try:
            parse_canonical_json_bytes(raw.encode("utf-8"))
        except Exception as exc:
            raise FactorGovernanceError(f"candidate {label} is not canonical JSON") from exc
    require_sha256(candidate.get("implementation_sha256"), label="implementation_sha256")


def compute_installed_signals(
    frames: Mapping[str, pd.DataFrame],
    *,
    factor_ids: Sequence[str],
) -> dict[str, pd.Series]:
    """Recompute signals only through finite entrypoints in this installed release."""

    requested = list(factor_ids)
    if not requested or len(requested) != len(set(requested)):
        raise FactorGovernanceError("factor implementation request is empty or duplicated")
    if not set(requested).issubset(_ENTRYPOINTS):
        raise FactorGovernanceError("factor implementation is not installed")
    all_signals = compute_bootstrap_signals(frames, source_format=CANONICAL_PARQUET)
    return {factor_id: all_signals[factor_id].copy() for factor_id in requested}


__all__ = [
    "InstalledFactorImplementation",
    "compute_installed_signals",
    "implementation_code_sha256",
    "installed_implementation_rows",
    "installed_semantic_row",
    "validate_candidate_implementation",
]
