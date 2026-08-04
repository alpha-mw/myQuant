"""Pure schemas for the research-only Factor v4.1 no-label diagnostic.

This module owns validation, structural source auditing, diagnostic accounting,
and the private-bundle contract.  It has no market loader and no production
state transition.  Filesystem publication is delegated to the existing shared
owner-private, no-clobber bundle helper.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_investor.factors import governance_private_bundle_io as private_io
from quant_investor.factors.governance_aquant_no_label_eval_v4_1 import (
    EXPECTED_PINNED_IDEA_COUNT,
    MATRIX_HASH_CONTRACT_VERSION,
    matrix_hash_descriptor_v4_1,
)


PROTOCOL_VERSION = "v4.1"
READINESS = "EXPLORATORY_NO_LABEL_SIGNAL_DIAGNOSTIC_ONLY"
OPERATOR_PROFILE_SCHEMA_VERSION = "factor-governance-no-label-operator-profile.v4.1"
DIAGNOSTIC_SCHEMA_VERSION = "factor-governance-no-label-signal-diagnostic.v4.1"
READBACK_SCHEMA_VERSION = "factor-governance-no-label-diagnostic-readback.v4.1"
STRUCTURAL_AUDIT_SCHEMA_VERSION = "factor-governance-no-label-source-audit.v4.1"

OPERATOR_PROFILE_FILENAME = "no_label_operator_profile.v4_1.json"
DIAGNOSTIC_FILENAME = "no_label_signal_diagnostic.v4_1.json"
READBACK_FILENAME = "no_label_diagnostic_readback.v4_1.json"
BUNDLE_INPUT_FILENAMES = (OPERATOR_PROFILE_FILENAME, DIAGNOSTIC_FILENAME)
BUNDLE_FILENAMES = (*BUNDLE_INPUT_FILENAMES, READBACK_FILENAME)
PRIVATE_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_1_no_label_diagnostic",
)

STATUS_SIGNAL_DIAGNOSTIC = "no_label_signal_eval_diagnostic"
STATUS_TURNOVER_BLOCKED = "turnover_data_blocked"
STATUS_FUNDAMENTAL_BLOCKED = "fundamental_semantic_blocked"
EXACT_STATUS_COUNTS = {
    STATUS_SIGNAL_DIAGNOSTIC: 27,
    STATUS_TURNOVER_BLOCKED: 2,
    STATUS_FUNDAMENTAL_BLOCKED: 8,
}
FUNDAMENTAL_FIELDS = frozenset(
    {
        "fcf_to_price",
        "fin_debt_to_assets",
        "fin_net_profit_yoy",
        "fin_ocf_to_profit",
        "fin_roa",
        "fin_roe",
    }
)
RAW_TABLE_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)
DERIVED_MATRIX_FIELDS = ("volume", "vwap")
EVALUATOR_MATRIX_FIELDS = (
    "amount",
    "close",
    "high",
    "low",
    "open",
    "turnover_rate",
    "volume",
    "vwap",
)

AUTHORITY_FIELDS = {
    "signal_computability_proven": False,
    "screening_authority": False,
    "screening_eligible": False,
    "bh_authority": False,
    "family_bh_authoritative": False,
    "runtime_equivalence_proven": False,
    "runtime_equivalence_verified": False,
    "qualification": False,
    "qualified": False,
    "healthy": False,
    "admission_authority": False,
    "formal_admission_authority": False,
    "proposal_authority": False,
    "proposal_eligible": False,
    "registry_authority": False,
    "registry_entry_created": False,
    "production_apply_enabled": False,
    "new_risk_eligible": False,
    "new_risk_authorized": False,
}
SIDE_EFFECT_FIELDS = {
    "wal": False,
    "budget": False,
    "proposal": False,
    "registry": False,
    "apply": False,
    "transaction": False,
    "production": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
    "network": False,
}

_FORBIDDEN_IDENTIFIER_FAMILIES = frozenset(
    {
        "backtest",
        "execution",
        "forward",
        "label",
        "provider",
        "realized",
        "registry",
        "replay",
        "return",
        "target",
    }
)
_ROLE_ALLOWED_IMPORTS = {
    "evaluator": frozenset(
        {
            "__future__",
            "ast",
            "collections.abc",
            "copy",
            "hashlib",
            "json",
            "math",
            "numpy",
            "pandas",
            "typing",
        }
    ),
    "data_builder": frozenset(
        {
            "__future__",
            "argparse",
            "collections.abc",
            "copy",
            "dataclasses",
            "hashlib",
            "json",
            "math",
            "numpy",
            "os",
            "pandas",
            "pathlib",
            "pyarrow",
            "pyarrow.dataset",
            "stat",
            "sys",
            "typing",
            "quant_investor.factors",
            "quant_investor.factors.governance_aquant_no_label_eval_v4_1",
            "quant_investor.factors.governance_discovery_v4_1",
            "quant_investor.factors.governance_no_label_diagnostic_v4_1",
            "quant_investor.factors.governance_private_bundle_io",
            "quant_investor.factors.governance_source_readback_v4_1",
            "quant_investor.factors.governance_source_v4_1",
            "quant_investor.market.pit_universe",
        }
    ),
}
_ROLE_ALLOWED_CALL_ATTRIBUTES = {
    "evaluator": frozenset(
        {
            "add",
            "append",
            "asarray",
            "ascontiguousarray",
            "astype",
            "copy",
            "deepcopy",
            "dumps",
            "encode",
            "equals",
            "get",
            "hexdigest",
            "is_numeric_dtype",
            "isfinite",
            "isnan",
            "isoformat",
            "items",
            "join",
            "mean",
            "parse",
            "rank",
            "rolling",
            "sha256",
            "to_numpy",
            "tobytes",
            "uint64",
            "update",
            "view",
            "where",
        }
    ),
    "data_builder": frozenset(
        {
            "ArgumentParser",
            "DataFrame",
            "DatetimeIndex",
            "Path",
            "S_IMODE",
            "S_ISDIR",
            "S_ISLNK",
            "S_ISREG",
            "add_argument",
            "any",
            "append",
            "as_posix",
            "astype",
            "bind_pinned_source_ideas_v4_1",
            "build_diagnostic_row_v4_1",
            "build_operator_profile_v4_1",
            "build_private_bundle_contract_v4_1",
            "build_session_scope_descriptor_v4_1",
            "build_signal_diagnostic_v4_1",
            "build_structural_no_label_audit_v4_1",
            "classify_idea_status_v4_1",
            "dataset",
            "decode",
            "deepcopy",
            "dumps",
            "duplicated",
            "encode",
            "equals",
            "errstate",
            "evaluate_pinned_idea_v4_1",
            "field",
            "full",
            "get",
            "getuid",
            "hexdigest",
            "insert",
            "is_absolute",
            "is_dir",
            "is_symlink",
            "isfinite",
            "issubset",
            "items",
            "iterdir",
            "loads",
            "lstat",
            "matrix_hash_descriptor_v4_1",
            "normpath",
            "parse_args",
            "pivot",
            "publish_private_bundle",
            "read_bytes",
            "read_parquet",
            "readback_private_bundle",
            "reindex",
            "relative_to",
            "replace",
            "resolve",
            "rglob",
            "rpartition",
            "sha256",
            "sort",
            "startswith",
            "to_datetime",
            "to_dict",
            "to_numpy",
            "to_pandas",
            "to_table",
            "validate_cutoff_source_node_v4_1",
            "validate_design_source_node_v4_1",
            "validate_pit_records_v4_1",
            "where",
            "zeros",
        }
    ),
}
_ROLE_ALLOWED_NAMED_CALLS = {
    "evaluator": frozenset(
        {
            "FactorGovernanceAquantNoLabelEvalV4_1Error",
            "_axis_sha256",
            "_candidate_definition_sha256",
            "_mask_frame",
            "_normalized_expression_node",
            "_self_hash",
            "_sha",
            "_source_definition_sha256",
            "_text",
            "_validate_axes",
            "_validate_source_receipt",
            "any",
            "canonical_json_bytes_v4_1",
            "collect",
            "dict",
            "enumerate",
            "evaluate_expression_v4_1",
            "frozenset",
            "int",
            "isinstance",
            "len",
            "list",
            "matrix_hash_descriptor_v4_1",
            "normalize_expression_ast_v4_1",
            "semantic_sha256_v4_1",
            "set",
            "sorted",
            "type",
            "visit",
        }
    ),
    "data_builder": frozenset(
        {
            "FactorV4_1SignalDiagnosticRunnerError",
            "Path",
            "SystemExit",
            "_absolute_path",
            "_binding_snapshot",
            "_canonical_json_bytes",
            "_derive_vwap",
            "_inventory_table",
            "_load_components",
            "_load_market_matrices",
            "_load_pit_records",
            "_parse_binding",
            "_parse_bindings",
            "_protected_stability",
            "_read_bundle",
            "_read_control_bindings",
            "_revalidate_prepublication_inputs",
            "_report_semantic",
            "_reproduce_session_scope",
            "_semantic_sha",
            "_sha",
            "_signature",
            "_stable_bytes",
            "_strict_json",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "int",
            "isinstance",
            "len",
            "list",
            "main",
            "object",
            "parse_args",
            "print",
            "run",
            "set",
            "sorted",
            "str",
            "sum",
            "type",
        }
    ),
}
_FRAME_NAMES = frozenset({"bars", "data", "frame", "raw", "table_rows"})
_SHA_PATTERN = re.compile(r"[0-9a-f]{64}")


class FactorGovernanceNoLabelDiagnosticV4_1Error(ValueError):
    """Raised when no-label evidence is structurally or semantically invalid."""


def canonical_json_bytes_v4_1(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_file_bytes_v4_1(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes_v4_1(value) + b"\n"


def semantic_sha256_v4_1(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_1(value)).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    if field in payload:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"self-hash field already exists: {field}"
        )
    payload[field] = semantic_sha256_v4_1(payload)
    return payload


def _validate_self_hash(value: Mapping[str, Any], field: str, context: str) -> None:
    stored = value.get(field)
    if type(stored) is not str or _SHA_PATTERN.fullmatch(stored) is None:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"{context}.{field} is not a SHA-256"
        )
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    if semantic_sha256_v4_1(payload) != stored:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"{context}.{field} self-hash mismatch"
        )


def _sha(value: Any, context: str) -> str:
    if type(value) is not str or _SHA_PATTERN.fullmatch(value) is None:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"{context} is not a SHA-256"
        )
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"{context} fields mismatch: "
            f"missing={sorted(expected - set(value))};extra={sorted(set(value) - expected)}"
        )


def _validate_binding_rows(value: Any, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"{context} must be a non-empty list"
        )
    binding_ids: list[str] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"{context} rows must be objects"
            )
        allowed = {"binding_id", "absolute_path", "byte_sha256"}
        if "semantic_sha256" in row:
            allowed.add("semantic_sha256")
        _exact_keys(row, allowed, f"{context} row")
        binding_id = row.get("binding_id")
        if type(binding_id) is not str or not binding_id:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"{context} binding_id must be a non-empty string"
            )
        binding_ids.append(binding_id)
        path_value = row.get("absolute_path")
        if type(path_value) is not str:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"{context} absolute_path must be a string"
            )
        path = Path(path_value)
        if not path.is_absolute() or Path(os.path.normpath(path)) != path:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"{context} absolute_path must be absolute and normalized"
            )
        _sha(row.get("byte_sha256"), f"{context} byte SHA")
        if "semantic_sha256" in row:
            _sha(row.get("semantic_sha256"), f"{context} semantic SHA")
    if len(binding_ids) != len(set(binding_ids)):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"{context} binding ids must be distinct"
        )


def _identifier_tokens(value: str) -> set[str]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()
    return {item for item in re.split(r"[^a-z0-9]+", snake) if item}


def _forbidden_identifier(value: str) -> str | None:
    tokens = _identifier_tokens(value)
    # ``no_label`` is the explicit name of this proof lane, not a data label.
    # Keep that exact negated marker auditable while rejecting every other use.
    if "no_label" in re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower():
        tokens.discard("label")
    elif "nolabel" in value.lower():
        tokens.discard("label")
    for family in sorted(_FORBIDDEN_IDENTIFIER_FAMILIES):
        if any(item == family or item.startswith(family) for item in tokens):
            return family
    return None


def _import_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [item.name for item in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level != 0 or node.module is None:
            return ["<relative>"]
        return [node.module]
    return []


def _negative_numeric(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float)
    )


def _expression_root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _validate_frame_column_name(value: str) -> None:
    declared = (
        set(RAW_TABLE_FIELDS)
        | set(DERIVED_MATRIX_FIELDS)
        | set(EVALUATOR_MATRIX_FIELDS)
    )
    if value not in declared:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"source accesses undeclared data column: {value}"
        )


def _audit_source(role: str, absolute_path: str, raw: bytes) -> dict[str, Any]:
    if role not in _ROLE_ALLOWED_IMPORTS:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"unsupported structural-audit role: {role}"
        )
    path = Path(absolute_path)
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "audited code path must be absolute and normalized"
        )
    try:
        source = raw.decode("utf-8")
        tree = ast.parse(source, filename=absolute_path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"audited source is not valid UTF-8 Python: {absolute_path}"
        ) from exc
    allowed_imports = _ROLE_ALLOWED_IMPORTS[role]
    allowed_attributes = _ROLE_ALLOWED_CALL_ATTRIBUTES[role]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for imported in _import_names(node):
                if imported not in allowed_imports:
                    raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                        f"source import is not allowlisted: {role}:{imported}"
                    )
                forbidden = _forbidden_identifier(imported)
                if forbidden is not None:
                    raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                        f"source import uses forbidden family: {forbidden}"
                    )
        identifier: str | None = None
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifier = node.name
        elif isinstance(node, ast.arg):
            identifier = node.arg
        if identifier is not None:
            forbidden = _forbidden_identifier(identifier)
            if forbidden is not None:
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    f"source identifier uses forbidden family: {forbidden}:{identifier}"
                )
        if isinstance(node, ast.Attribute):
            forbidden = _forbidden_identifier(node.attr)
            if forbidden is not None:
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    f"source attribute uses forbidden family: {forbidden}:{node.attr}"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attribute = node.func.attr
            if attribute in {"pct_change", "diff"}:
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    f"source call is forbidden: {attribute}"
                )
            if attribute == "shift" and any(
                _negative_numeric(argument) for argument in node.args
            ):
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    "negative shift is forbidden"
                )
            if attribute not in allowed_attributes:
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    f"source call attribute is not allowlisted: {role}:{attribute}"
                )
            if (
                attribute == "get"
                and _expression_root_name(node.func.value) in _FRAME_NAMES
            ):
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and type(argument.value) is str:
                        _validate_frame_column_name(argument.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            call_name = node.func.id
            if call_name in {"__import__", "compile", "eval", "exec"}:
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    f"source named call is forbidden: {call_name}"
                )
            if call_name not in _ROLE_ALLOWED_NAMED_CALLS[role]:
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    f"source named call is not allowlisted: {role}:{call_name}"
                )
        if (
            isinstance(node, ast.Subscript)
            and _expression_root_name(node.value) in _FRAME_NAMES
        ):
            for item in ast.walk(node.slice):
                if isinstance(item, ast.Constant) and type(item.value) is str:
                    _validate_frame_column_name(item.value)
    ast_bytes = ast.dump(tree, annotate_fields=True, include_attributes=False).encode(
        "utf-8"
    )
    return {
        "role": role,
        "absolute_path": str(path),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "ast_sha256": hashlib.sha256(ast_bytes).hexdigest(),
    }


def build_structural_no_label_audit_v4_1(
    source_files: Mapping[str, tuple[str, bytes]],
) -> dict[str, Any]:
    """Audit the exact evaluator and data-builder Python bytes using AST only."""

    if set(source_files) != {"data_builder", "evaluator"}:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "structural audit requires exact evaluator and data_builder roles"
        )
    audited = [
        _audit_source(role, source_files[role][0], source_files[role][1])
        for role in ("data_builder", "evaluator")
    ]
    payload = {
        "schema_version": STRUCTURAL_AUDIT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed_exact_ast_allowlist",
        "allowed_imports": {
            role: sorted(_ROLE_ALLOWED_IMPORTS[role]) for role in sorted(source_files)
        },
        "allowed_call_attributes": {
            role: sorted(_ROLE_ALLOWED_CALL_ATTRIBUTES[role])
            for role in sorted(source_files)
        },
        "allowed_named_calls": {
            role: sorted(_ROLE_ALLOWED_NAMED_CALLS[role])
            for role in sorted(source_files)
        },
        "raw_table_fields": list(RAW_TABLE_FIELDS),
        "derived_matrix_fields": list(DERIVED_MATRIX_FIELDS),
        "audited_code": audited,
    }
    return _seal(payload, "audit_semantic_sha256")


def validate_structural_no_label_audit_v4_1(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_version",
            "status",
            "allowed_imports",
            "allowed_call_attributes",
            "allowed_named_calls",
            "raw_table_fields",
            "derived_matrix_fields",
            "audited_code",
            "audit_semantic_sha256",
        },
        "structural audit",
    )
    _validate_self_hash(payload, "audit_semantic_sha256", "structural audit")
    if (
        payload.get("schema_version") != STRUCTURAL_AUDIT_SCHEMA_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("status") != "passed_exact_ast_allowlist"
        or payload.get("raw_table_fields") != list(RAW_TABLE_FIELDS)
        or payload.get("derived_matrix_fields") != list(DERIVED_MATRIX_FIELDS)
        or payload.get("allowed_imports")
        != {
            role: sorted(_ROLE_ALLOWED_IMPORTS[role])
            for role in ("data_builder", "evaluator")
        }
        or payload.get("allowed_call_attributes")
        != {
            role: sorted(_ROLE_ALLOWED_CALL_ATTRIBUTES[role])
            for role in ("data_builder", "evaluator")
        }
        or payload.get("allowed_named_calls")
        != {
            role: sorted(_ROLE_ALLOWED_NAMED_CALLS[role])
            for role in ("data_builder", "evaluator")
        }
    ):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "structural audit contract mismatch"
        )
    rows = payload.get("audited_code")
    if not isinstance(rows, list) or [row.get("role") for row in rows] != [
        "data_builder",
        "evaluator",
    ]:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "structural audit code inventory mismatch"
        )
    for row in rows:
        if set(row) != {
            "role",
            "absolute_path",
            "byte_sha256",
            "size_bytes",
            "ast_sha256",
        }:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "structural audit code descriptor fields mismatch"
            )
        path = Path(str(row.get("absolute_path")))
        if not path.is_absolute() or Path(os.path.normpath(path)) != path:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "structural audit code path mismatch"
            )
        if type(row.get("size_bytes")) is not int or row["size_bytes"] <= 0:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "structural audit code size must be positive"
            )
        _sha(row.get("byte_sha256"), "audited code byte SHA")
        _sha(row.get("ast_sha256"), "audited code AST SHA")
    return payload


def classify_idea_status_v4_1(idea: Mapping[str, Any]) -> str:
    fields = idea.get("input_fields")
    if not isinstance(fields, list) or any(type(item) is not str for item in fields):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "idea input_fields must be a string list"
        )
    field_set = set(fields)
    if field_set & FUNDAMENTAL_FIELDS:
        return STATUS_FUNDAMENTAL_BLOCKED
    if field_set == {"turnover_rate"}:
        return STATUS_TURNOVER_BLOCKED
    return STATUS_SIGNAL_DIAGNOSTIC


def _normalize_bindings(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"{context} must be a non-empty list"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"{context}[{index}] must be an object"
            )
        path = Path(str(raw.get("absolute_path")))
        if not path.is_absolute() or Path(os.path.normpath(path)) != path:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"{context}[{index}] path is not absolute and normalized"
            )
        row = {
            "binding_id": str(raw.get("binding_id")),
            "absolute_path": str(path),
            "byte_sha256": _sha(raw.get("byte_sha256"), f"{context} byte SHA"),
        }
        if "semantic_sha256" in raw:
            row["semantic_sha256"] = _sha(
                raw.get("semantic_sha256"), f"{context} semantic SHA"
            )
        rows.append(row)
    rows.sort(key=lambda row: row["binding_id"])
    ids = [row["binding_id"] for row in rows]
    if any(not item or item == "None" for item in ids) or len(ids) != len(set(ids)):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"{context} binding ids must be non-empty and distinct"
        )
    return rows


def _validate_authority(value: Mapping[str, Any]) -> None:
    if any(value.get(key) is not expected for key, expected in AUTHORITY_FIELDS.items()):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "no-label authority fields must all remain false"
        )


def build_operator_profile_v4_1(
    *,
    cycle_id: str,
    bound_ideas: Sequence[Mapping[str, Any]],
    source_bindings: Sequence[Mapping[str, Any]],
    code_bindings: Sequence[Mapping[str, Any]],
    structural_audit: Mapping[str, Any],
) -> dict[str, Any]:
    ideas = [copy.deepcopy(dict(item)) for item in bound_ideas]
    if len(ideas) != EXPECTED_PINNED_IDEA_COUNT:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "operator profile requires exact 37 bound ideas"
        )
    classifications = [
        {
            "candidate_id": item["candidate_id"],
            "name": item["name"],
            "status": classify_idea_status_v4_1(item),
            "source_definition_sha256": item["source_definition_sha256"],
            "normalized_ast_sha256": item[
                "full_candidate_normalized_ast_sha256"
            ],
            "catalog_definition_sha256": item["catalog_definition_sha256"],
            "mapping_semantic_sha256": item["mapping_semantic_sha256"],
            "input_fields": list(item["input_fields"]),
            "initial_weight": 0.0,
        }
        for item in ideas
    ]
    counts = {
        status: sum(row["status"] == status for row in classifications)
        for status in EXACT_STATUS_COUNTS
    }
    if counts != EXACT_STATUS_COUNTS:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "operator profile accounting must be exactly 27+2+8"
        )
    payload = {
        "schema_version": OPERATOR_PROFILE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "readiness": READINESS,
        "operator_semantics": {
            "binary_divide": {
                "implementation": "native_pandas_divide",
                "nonfinite_rewrite": False,
            },
            "ts_mean": {
                "implementation": "rolling_window_min_periods_1_mean",
                "maximum_window": 200,
            },
            "cs_rank": {
                "implementation": "row_rank_pct_true_na_option_keep",
                "axis": 1,
            },
            "vwap": {
                "derivation": "amount_thousand_cny_times_1000_divided_by_vol_lots_times_100",
                "formula": "amount_times_10_divided_by_vol",
                "zero_or_nonfinite_denominator_to_nan": True,
                "nonfinite_result_to_nan": True,
            },
            "pit_evaluation_envelope": {
                "source_matrices_masked_before_evaluation": True,
                "mask_reapplied_after_each_dataframe_node": True,
                "mask_reapplied_after_ts_mean": True,
                "mask_reapplied_before_and_after_cs_rank": True,
                "final_output_masked": True,
                "aquant_global_universe_claimed": False,
            },
        },
        "matrix_hash_contract": MATRIX_HASH_CONTRACT_VERSION,
        "raw_table_fields": list(RAW_TABLE_FIELDS),
        "derived_matrix_fields": list(DERIVED_MATRIX_FIELDS),
        "candidate_count": EXPECTED_PINNED_IDEA_COUNT,
        "status_counts": counts,
        "candidate_classifications": classifications,
        "source_bindings": _normalize_bindings(source_bindings, "source_bindings"),
        "code_bindings": _normalize_bindings(code_bindings, "code_bindings"),
        "structural_no_label_audit": validate_structural_no_label_audit_v4_1(
            structural_audit
        ),
        **AUTHORITY_FIELDS,
    }
    return validate_operator_profile_v4_1(
        _seal(payload, "operator_profile_semantic_sha256")
    )


def _reject_statistical_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            tokens = set(normalized.split("_"))
            if "score" in tokens or "qvalue" in tokens or (
                "q" in tokens and "value" in tokens
            ) or "return" in tokens:
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    f"forbidden statistical field: {key}"
                )
            if "weight" in tokens and normalized != "initial_weight":
                raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                    f"forbidden weight field: {key}"
                )
            _reject_statistical_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_statistical_keys(item)


def validate_operator_profile_v4_1(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_version",
            "cycle_id",
            "readiness",
            "operator_semantics",
            "matrix_hash_contract",
            "raw_table_fields",
            "derived_matrix_fields",
            "candidate_count",
            "status_counts",
            "candidate_classifications",
            "source_bindings",
            "code_bindings",
            "structural_no_label_audit",
            "operator_profile_semantic_sha256",
            *AUTHORITY_FIELDS,
        },
        "operator profile",
    )
    _validate_self_hash(payload, "operator_profile_semantic_sha256", "operator profile")
    _reject_statistical_keys(payload)
    if (
        payload.get("schema_version") != OPERATOR_PROFILE_SCHEMA_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("readiness") != READINESS
        or payload.get("candidate_count") != EXPECTED_PINNED_IDEA_COUNT
        or payload.get("status_counts") != EXACT_STATUS_COUNTS
        or payload.get("raw_table_fields") != list(RAW_TABLE_FIELDS)
        or payload.get("derived_matrix_fields") != list(DERIVED_MATRIX_FIELDS)
        or payload.get("matrix_hash_contract") != MATRIX_HASH_CONTRACT_VERSION
    ):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "operator profile contract mismatch"
        )
    semantics = payload.get("operator_semantics")
    if (
        not isinstance(semantics, Mapping)
        or semantics.get("binary_divide", {}).get("implementation")
        != "native_pandas_divide"
        or semantics.get("binary_divide", {}).get("nonfinite_rewrite") is not False
        or semantics.get("ts_mean", {}).get("maximum_window") != 200
        or semantics.get("pit_evaluation_envelope", {}).get(
            "mask_reapplied_after_each_dataframe_node"
        )
        is not True
    ):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "operator semantic pin mismatch"
        )
    _exact_keys(
        semantics,
        {"binary_divide", "ts_mean", "cs_rank", "vwap", "pit_evaluation_envelope"},
        "operator semantics",
    )
    nested_fields = {
        "binary_divide": {"implementation", "nonfinite_rewrite"},
        "ts_mean": {"implementation", "maximum_window"},
        "cs_rank": {"implementation", "axis"},
        "vwap": {
            "derivation",
            "formula",
            "zero_or_nonfinite_denominator_to_nan",
            "nonfinite_result_to_nan",
        },
        "pit_evaluation_envelope": {
            "source_matrices_masked_before_evaluation",
            "mask_reapplied_after_each_dataframe_node",
            "mask_reapplied_after_ts_mean",
            "mask_reapplied_before_and_after_cs_rank",
            "final_output_masked",
            "aquant_global_universe_claimed",
        },
    }
    for key, fields in nested_fields.items():
        if not isinstance(semantics[key], Mapping):
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"operator semantic {key} must be an object"
            )
        _exact_keys(semantics[key], fields, f"operator semantic {key}")
    rows = payload.get("candidate_classifications")
    if not isinstance(rows, list) or len(rows) != EXPECTED_PINNED_IDEA_COUNT:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "operator profile candidate inventory mismatch"
        )
    for row in rows:
        _exact_keys(
            row,
            {
                "candidate_id",
                "name",
                "status",
                "source_definition_sha256",
                "normalized_ast_sha256",
                "catalog_definition_sha256",
                "mapping_semantic_sha256",
                "input_fields",
                "initial_weight",
            },
            "candidate classification",
        )
        if row.get("initial_weight") != 0.0:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "operator profile initial_weight must be zero"
            )
        if row.get("status") not in EXACT_STATUS_COUNTS:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "candidate classification status mismatch"
            )
        for field in (
            "source_definition_sha256",
            "normalized_ast_sha256",
            "catalog_definition_sha256",
            "mapping_semantic_sha256",
        ):
            _sha(row.get(field), f"candidate classification {field}")
    if len({row.get("candidate_id") for row in rows}) != EXPECTED_PINNED_IDEA_COUNT:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "candidate classification ids must be distinct"
        )
    if {
        status: sum(row.get("status") == status for row in rows)
        for status in EXACT_STATUS_COUNTS
    } != EXACT_STATUS_COUNTS:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "candidate classification status accounting mismatch"
        )
    _validate_binding_rows(payload.get("source_bindings"), "source_bindings")
    _validate_binding_rows(payload.get("code_bindings"), "code_bindings")
    validate_structural_no_label_audit_v4_1(payload["structural_no_label_audit"])
    _validate_authority(payload)
    return payload


def build_diagnostic_row_v4_1(
    *,
    idea: Mapping[str, Any],
    status: str,
    signal: pd.DataFrame | None = None,
    eligibility_mask: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if status not in EXACT_STATUS_COUNTS:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            f"diagnostic row status is invalid: {status}"
        )
    expected = classify_idea_status_v4_1(idea)
    if status != expected:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "diagnostic row status differs from deterministic classification"
        )
    metrics: dict[str, Any]
    if status == STATUS_SIGNAL_DIAGNOSTIC:
        if (
            not isinstance(signal, pd.DataFrame)
            or not isinstance(eligibility_mask, pd.DataFrame)
            or not signal.index.equals(eligibility_mask.index)
            or not signal.columns.equals(eligibility_mask.columns)
        ):
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "evaluated diagnostic row requires exact signal/mask axes"
            )
        mask = eligibility_mask.to_numpy(dtype=bool, copy=True)
        values = signal.to_numpy(dtype=np.float64, copy=True)
        eligible = values[mask]
        finite_count = int(np.isfinite(eligible).sum())
        positive_inf_count = int(np.isposinf(eligible).sum())
        negative_inf_count = int(np.isneginf(eligible).sum())
        nan_count = int(np.isnan(eligible).sum())
        outside_non_nan = int((~np.isnan(values[~mask])).sum())
        if finite_count <= 0 or outside_non_nan != 0:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "evaluated row has no finite observations or violates the PIT mask"
            )
        eligible_count = int(eligible.size)
        metrics = {
            "eligible_cell_count": eligible_count,
            "finite_count": finite_count,
            "finite_ratio": finite_count / eligible_count if eligible_count else 0.0,
            "nan_count": nan_count,
            "positive_inf_count": positive_inf_count,
            "negative_inf_count": negative_inf_count,
            "outside_mask_non_nan_count": outside_non_nan,
            "signal_matrix": matrix_hash_descriptor_v4_1(signal),
        }
    else:
        if signal is not None or eligibility_mask is not None:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "blocked diagnostic rows cannot carry signal data"
            )
        metrics = {
            "eligible_cell_count": None,
            "finite_count": None,
            "finite_ratio": None,
            "nan_count": None,
            "positive_inf_count": None,
            "negative_inf_count": None,
            "outside_mask_non_nan_count": None,
            "signal_matrix": None,
        }
    payload = {
        "candidate_id": idea["candidate_id"],
        "name": idea["name"],
        "status": status,
        "input_fields": list(idea["input_fields"]),
        "source_definition_sha256": idea["source_definition_sha256"],
        "normalized_ast_sha256": idea["full_candidate_normalized_ast_sha256"],
        "catalog_definition_sha256": idea["catalog_definition_sha256"],
        "mapping_semantic_sha256": idea["mapping_semantic_sha256"],
        "initial_weight": 0.0,
        **metrics,
    }
    return _seal(payload, "row_semantic_sha256")


def validate_diagnostic_row_v4_1(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    _exact_keys(
        payload,
        {
            "candidate_id",
            "name",
            "status",
            "input_fields",
            "source_definition_sha256",
            "normalized_ast_sha256",
            "catalog_definition_sha256",
            "mapping_semantic_sha256",
            "initial_weight",
            "eligible_cell_count",
            "finite_count",
            "finite_ratio",
            "nan_count",
            "positive_inf_count",
            "negative_inf_count",
            "outside_mask_non_nan_count",
            "signal_matrix",
            "row_semantic_sha256",
        },
        "diagnostic row",
    )
    _validate_self_hash(payload, "row_semantic_sha256", "diagnostic row")
    _reject_statistical_keys(payload)
    if payload.get("status") not in EXACT_STATUS_COUNTS or payload.get(
        "initial_weight"
    ) != 0.0:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "diagnostic row status or initial_weight mismatch"
        )
    for field in (
        "source_definition_sha256",
        "normalized_ast_sha256",
        "catalog_definition_sha256",
        "mapping_semantic_sha256",
    ):
        _sha(payload.get(field), f"diagnostic row {field}")
    if payload["status"] == STATUS_SIGNAL_DIAGNOSTIC:
        if (
            type(payload.get("finite_count")) is not int
            or payload["finite_count"] <= 0
            or type(payload.get("finite_ratio")) is not float
            or not math.isfinite(payload["finite_ratio"])
            or not 0.0 <= payload["finite_ratio"] <= 1.0
            or payload.get("outside_mask_non_nan_count") != 0
            or not isinstance(payload.get("signal_matrix"), Mapping)
            or payload["signal_matrix"].get("contract")
            != MATRIX_HASH_CONTRACT_VERSION
        ):
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "evaluated diagnostic row metrics are invalid"
            )
        _exact_keys(
            payload["signal_matrix"],
            {
                "contract",
                "shape",
                "dtype",
                "date_axis_sha256",
                "symbol_axis_sha256",
                "matrix_sha256",
            },
            "signal matrix descriptor",
        )
    elif any(
        payload.get(field) is not None
        for field in (
            "eligible_cell_count",
            "finite_count",
            "finite_ratio",
            "nan_count",
            "positive_inf_count",
            "negative_inf_count",
            "outside_mask_non_nan_count",
            "signal_matrix",
        )
    ):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "blocked diagnostic row must have null metrics"
        )
    return payload


_PROFILE_DIAGNOSTIC_ALIGNMENT_FIELDS = (
    "candidate_id",
    "name",
    "status",
    "source_definition_sha256",
    "normalized_ast_sha256",
    "catalog_definition_sha256",
    "mapping_semantic_sha256",
    "input_fields",
    "initial_weight",
)


def _validate_profile_diagnostic_row_alignment(
    profile_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> None:
    if len(profile_rows) != len(diagnostic_rows):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "operator-profile/diagnostic row inventory mismatch"
        )
    for index, (profile_row, diagnostic_row) in enumerate(
        zip(profile_rows, diagnostic_rows, strict=True)
    ):
        expected = {
            field: copy.deepcopy(profile_row.get(field))
            for field in _PROFILE_DIAGNOSTIC_ALIGNMENT_FIELDS
        }
        observed = {
            field: copy.deepcopy(diagnostic_row.get(field))
            for field in _PROFILE_DIAGNOSTIC_ALIGNMENT_FIELDS
        }
        if canonical_json_bytes_v4_1(observed) != canonical_json_bytes_v4_1(
            expected
        ):
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"operator-profile/diagnostic row alignment mismatch at index {index}"
            )


def build_signal_diagnostic_v4_1(
    *,
    cycle_id: str,
    operator_profile: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    input_bindings: Sequence[Mapping[str, Any]],
    protected_stability: Sequence[Mapping[str, Any]],
    market_matrix_bindings: Sequence[Mapping[str, Any]],
    session_scope_binding: Mapping[str, Any],
    vwap_semantic_sha256: str,
) -> dict[str, Any]:
    profile = validate_operator_profile_v4_1(operator_profile)
    normalized_rows = [validate_diagnostic_row_v4_1(item) for item in rows]
    if len(normalized_rows) != EXPECTED_PINNED_IDEA_COUNT:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "signal diagnostic requires exact 37 rows"
        )
    ids = [row["candidate_id"] for row in normalized_rows]
    profile_ids = [row["candidate_id"] for row in profile["candidate_classifications"]]
    counts = {
        status: sum(row["status"] == status for row in normalized_rows)
        for status in EXACT_STATUS_COUNTS
    }
    if ids != profile_ids or counts != EXACT_STATUS_COUNTS:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "signal diagnostic row identity/accounting mismatch"
        )
    _validate_profile_diagnostic_row_alignment(
        profile["candidate_classifications"], normalized_rows
    )
    stability = [copy.deepcopy(dict(row)) for row in protected_stability]
    if not stability:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "protected stability evidence must not be empty"
        )
    for row in stability:
        before = _sha(row.get("before_sha256"), "protected before SHA")
        after = _sha(row.get("after_sha256"), "protected after SHA")
        expected = _sha(row.get("expected_sha256"), "protected expected SHA")
        if not before == after == expected:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "protected binding changed during diagnostic build"
            )
    payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "readiness": READINESS,
        "operator_profile_semantic_sha256": profile[
            "operator_profile_semantic_sha256"
        ],
        "candidate_count": EXPECTED_PINNED_IDEA_COUNT,
        "status_counts": counts,
        "rows": normalized_rows,
        "input_bindings": _normalize_bindings(input_bindings, "input_bindings"),
        "protected_stability": stability,
        "market_matrix_bindings": _normalize_bindings(
            market_matrix_bindings, "market_matrix_bindings"
        ),
        "session_scope_binding": copy.deepcopy(dict(session_scope_binding)),
        "vwap_semantic_sha256": _sha(
            vwap_semantic_sha256, "VWAP semantic SHA"
        ),
        "all_rows_accounted": True,
        "all_evaluated_rows_have_finite_observations": True,
        "side_effects": dict(SIDE_EFFECT_FIELDS),
        **AUTHORITY_FIELDS,
    }
    return validate_signal_diagnostic_v4_1(
        _seal(payload, "diagnostic_semantic_sha256")
    )


def validate_signal_diagnostic_v4_1(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_version",
            "cycle_id",
            "readiness",
            "operator_profile_semantic_sha256",
            "candidate_count",
            "status_counts",
            "rows",
            "input_bindings",
            "protected_stability",
            "market_matrix_bindings",
            "session_scope_binding",
            "vwap_semantic_sha256",
            "all_rows_accounted",
            "all_evaluated_rows_have_finite_observations",
            "side_effects",
            "diagnostic_semantic_sha256",
            *AUTHORITY_FIELDS,
        },
        "signal diagnostic",
    )
    _validate_self_hash(payload, "diagnostic_semantic_sha256", "signal diagnostic")
    _reject_statistical_keys(payload)
    if (
        payload.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("readiness") != READINESS
        or payload.get("candidate_count") != EXPECTED_PINNED_IDEA_COUNT
        or payload.get("status_counts") != EXACT_STATUS_COUNTS
        or payload.get("all_rows_accounted") is not True
        or payload.get("all_evaluated_rows_have_finite_observations") is not True
        or payload.get("side_effects") != SIDE_EFFECT_FIELDS
    ):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "signal diagnostic contract mismatch"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_PINNED_IDEA_COUNT:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "signal diagnostic row inventory mismatch"
        )
    normalized_rows = [validate_diagnostic_row_v4_1(row) for row in rows]
    candidate_ids = [row.get("candidate_id") for row in normalized_rows]
    if (
        any(type(candidate_id) is not str or not candidate_id for candidate_id in candidate_ids)
        or len(set(candidate_ids)) != EXPECTED_PINNED_IDEA_COUNT
    ):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "signal diagnostic candidate ids must be 37 distinct non-empty strings"
        )
    recomputed_counts = {
        status: sum(row["status"] == status for row in normalized_rows)
        for status in EXACT_STATUS_COUNTS
    }
    if recomputed_counts != EXACT_STATUS_COUNTS:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "signal diagnostic row status accounting mismatch"
        )
    _validate_binding_rows(payload.get("input_bindings"), "input_bindings")
    _validate_binding_rows(
        payload.get("market_matrix_bindings"), "market_matrix_bindings"
    )
    stability = payload.get("protected_stability")
    if not isinstance(stability, list) or not stability:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "protected stability inventory mismatch"
        )
    for row in stability:
        _exact_keys(
            row,
            {
                "binding_id",
                "absolute_path",
                "expected_sha256",
                "before_sha256",
                "after_sha256",
            },
            "protected stability row",
        )
        expected = _sha(row.get("expected_sha256"), "protected expected SHA")
        if not expected == row.get("before_sha256") == row.get("after_sha256"):
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                "protected stability SHA mismatch"
            )
    scope = payload.get("session_scope_binding")
    if not isinstance(scope, Mapping):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "session scope binding must be an object"
        )
    _exact_keys(
        scope,
        {
            "session_count",
            "pit_record_count",
            "component_count",
            "descriptor_semantic_sha256",
            "eligibility_matrix",
        },
        "session scope binding",
    )
    _sha(scope.get("descriptor_semantic_sha256"), "session descriptor SHA")
    matrix = scope.get("eligibility_matrix")
    if not isinstance(matrix, Mapping):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "eligibility matrix descriptor must be an object"
        )
    _exact_keys(
        matrix,
        {
            "contract",
            "shape",
            "dtype",
            "date_axis_sha256",
            "symbol_axis_sha256",
            "matrix_sha256",
        },
        "eligibility matrix descriptor",
    )
    _validate_authority(payload)
    return payload


def build_readback_report_v4_1(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(artifacts) != set(BUNDLE_INPUT_FILENAMES):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "readback inputs must be the exact two diagnostic artifacts"
        )
    profile = validate_operator_profile_v4_1(artifacts[OPERATOR_PROFILE_FILENAME])
    diagnostic = validate_signal_diagnostic_v4_1(artifacts[DIAGNOSTIC_FILENAME])
    if diagnostic["operator_profile_semantic_sha256"] != profile[
        "operator_profile_semantic_sha256"
    ]:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "diagnostic does not bind the operator profile"
        )
    _validate_profile_diagnostic_row_alignment(
        profile["candidate_classifications"], diagnostic["rows"]
    )
    bindings = [copy.deepcopy(dict(item)) for item in artifact_bindings]
    if [item.get("filename") for item in bindings] != list(BUNDLE_INPUT_FILENAMES):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "readback artifact binding order mismatch"
        )
    payload = {
        "schema_version": READBACK_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": profile["cycle_id"],
        "run_id": run_id,
        "readiness": READINESS,
        "accepted": True,
        "artifact_bindings": bindings,
        "operator_profile_semantic_sha256": profile[
            "operator_profile_semantic_sha256"
        ],
        "diagnostic_semantic_sha256": diagnostic["diagnostic_semantic_sha256"],
        "side_effects": dict(SIDE_EFFECT_FIELDS),
        **AUTHORITY_FIELDS,
    }
    return _seal(payload, "report_semantic_sha256")


def validate_readback_report_v4_1(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_version",
            "cycle_id",
            "run_id",
            "readiness",
            "accepted",
            "artifact_bindings",
            "operator_profile_semantic_sha256",
            "diagnostic_semantic_sha256",
            "side_effects",
            "report_semantic_sha256",
            *AUTHORITY_FIELDS,
        },
        "readback report",
    )
    bindings = payload.get("artifact_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(BUNDLE_INPUT_FILENAMES):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "readback artifact binding inventory mismatch"
        )
    for row in bindings:
        _exact_keys(
            row,
            {"filename", "byte_sha256", "size_bytes", "mode", "uid", "nlink"},
            "readback artifact binding",
        )
    _validate_self_hash(payload, "report_semantic_sha256", "readback report")
    expected = build_readback_report_v4_1(
        run_id=payload["run_id"],
        artifacts=artifacts,
        artifact_bindings=artifact_bindings,
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "readback report differs from exact recomputation"
        )
    _validate_authority(payload)
    return payload


def validate_bundle_values_v4_1(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(values) != set(BUNDLE_FILENAMES):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "diagnostic bundle must contain exactly three artifacts"
        )
    profile = validate_operator_profile_v4_1(values[OPERATOR_PROFILE_FILENAME])
    diagnostic = validate_signal_diagnostic_v4_1(values[DIAGNOSTIC_FILENAME])
    if diagnostic["operator_profile_semantic_sha256"] != profile[
        "operator_profile_semantic_sha256"
    ]:
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "bundle operator-profile binding mismatch"
        )
    _validate_profile_diagnostic_row_alignment(
        profile["candidate_classifications"], diagnostic["rows"]
    )
    return {
        OPERATOR_PROFILE_FILENAME: profile,
        DIAGNOSTIC_FILENAME: diagnostic,
        READBACK_FILENAME: copy.deepcopy(dict(values[READBACK_FILENAME])),
    }


def build_private_bundle_contract_v4_1(
    *, expected_artifacts: Mapping[str, Mapping[str, Any]]
) -> private_io.PrivateBundleContract:
    if set(expected_artifacts) != set(BUNDLE_INPUT_FILENAMES):
        raise FactorGovernanceNoLabelDiagnosticV4_1Error(
            "expected artifacts must be the exact operator profile and diagnostic"
        )
    expected = {
        OPERATOR_PROFILE_FILENAME: validate_operator_profile_v4_1(
            expected_artifacts[OPERATOR_PROFILE_FILENAME]
        ),
        DIAGNOSTIC_FILENAME: validate_signal_diagnostic_v4_1(
            expected_artifacts[DIAGNOSTIC_FILENAME]
        ),
    }

    def validate_artifact(
        filename: str, value: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if filename == OPERATOR_PROFILE_FILENAME:
            normalized = validate_operator_profile_v4_1(value)
        elif filename == DIAGNOSTIC_FILENAME:
            normalized = validate_signal_diagnostic_v4_1(value)
        elif filename == READBACK_FILENAME:
            normalized = copy.deepcopy(dict(value))
        else:
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"unexpected diagnostic artifact: {filename}"
            )
        if filename in expected and canonical_json_bytes_v4_1(
            normalized
        ) != canonical_json_bytes_v4_1(expected[filename]):
            raise FactorGovernanceNoLabelDiagnosticV4_1Error(
                f"diagnostic artifact differs from expected bytes: {filename}"
            )
        return normalized

    def validate_complete(
        values: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        normalized = validate_bundle_values_v4_1(values)
        report = normalized[READBACK_FILENAME]
        bindings: Any = report.get("artifact_bindings")
        validate_readback_report_v4_1(
            report,
            artifacts={key: normalized[key] for key in BUNDLE_INPUT_FILENAMES},
            artifact_bindings=bindings,
        )
        return normalized

    def build_report(
        *,
        run_id: str,
        artifacts: Mapping[str, Mapping[str, Any]],
        artifact_bindings: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return build_readback_report_v4_1(
            run_id=run_id,
            artifacts=artifacts,
            artifact_bindings=artifact_bindings,
        )

    return private_io.PrivateBundleContract(
        root_suffix=PRIVATE_ROOT_SUFFIX,
        input_filenames=BUNDLE_INPUT_FILENAMES,
        readback_report_filename=READBACK_FILENAME,
        canonicalize=canonical_file_bytes_v4_1,
        validate_artifact=validate_artifact,
        validate_complete=validate_complete,
        build_readback_report=build_report,
    )


__all__ = [
    "AUTHORITY_FIELDS",
    "BUNDLE_FILENAMES",
    "BUNDLE_INPUT_FILENAMES",
    "DERIVED_MATRIX_FIELDS",
    "DIAGNOSTIC_FILENAME",
    "EXACT_STATUS_COUNTS",
    "FUNDAMENTAL_FIELDS",
    "FactorGovernanceNoLabelDiagnosticV4_1Error",
    "OPERATOR_PROFILE_FILENAME",
    "PRIVATE_ROOT_SUFFIX",
    "RAW_TABLE_FIELDS",
    "READBACK_FILENAME",
    "READINESS",
    "SIDE_EFFECT_FIELDS",
    "STATUS_FUNDAMENTAL_BLOCKED",
    "STATUS_SIGNAL_DIAGNOSTIC",
    "STATUS_TURNOVER_BLOCKED",
    "build_diagnostic_row_v4_1",
    "build_operator_profile_v4_1",
    "build_private_bundle_contract_v4_1",
    "build_readback_report_v4_1",
    "build_signal_diagnostic_v4_1",
    "build_structural_no_label_audit_v4_1",
    "canonical_file_bytes_v4_1",
    "canonical_json_bytes_v4_1",
    "classify_idea_status_v4_1",
    "semantic_sha256_v4_1",
    "validate_bundle_values_v4_1",
    "validate_diagnostic_row_v4_1",
    "validate_operator_profile_v4_1",
    "validate_readback_report_v4_1",
    "validate_signal_diagnostic_v4_1",
    "validate_structural_no_label_audit_v4_1",
]
