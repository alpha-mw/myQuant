"""Standalone no-label evaluator for the pinned A_quant v4.1 definitions.

The evaluator deliberately does not import the generic A_quant-expression
adapter.  It implements only the syntax and the two operator semantics needed
by the exact 37 formal-catalog classification rows.  A caller must separately
bind those rows with :func:`bind_pinned_source_ideas_v4_1` before evaluation.

Pinned pandas division is native ``/``: positive and negative infinity are not
silently rewritten.  The surrounding myQuant research envelope reapplies the
exact point-in-time eligibility mask after every DataFrame-producing AST node,
including rolling and cross-sectional operators.  That envelope is a local
PIT safety constraint, not a claim about A_quant's global-universe loader.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "v4.1"
PINNED_COMMIT = "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
PINNED_GENERATOR_PATH = "A_quant/scripts/run_factor_batch_screen.py"
PINNED_GENERATOR_FUNCTION = "generate_default_candidates"
PINNED_SOURCE_FILES = {
    "A_quant/app/data/schemas.py": (
        "848f324ada44b1d6e4c944d7e156fa9901779da797c51d8076e7b56db0a55817"
    ),
    "A_quant/app/factor_sandbox/expression.py": (
        "df93622a33309aa28d065d6e8fd366de1ebf7d2be600b26170084f727a7dc936"
    ),
    "A_quant/app/factor_sandbox/matrix_dataset.py": (
        "eab9ba96576d040622ae170fc36689a4ee62b64f13a91ae0efe9ff9cd8942547"
    ),
    "A_quant/app/factor_sandbox/operators.py": (
        "367f0c68a1e6f8c2e7f0fe168c91e23d77689f101fd203889d5c5b1c2bdb80a1"
    ),
    "A_quant/docs/factor_time_alignment_policy.md": (
        "e913ac9909927652b37571ee47c15d06e77b28227e1ee1f588179b435471f083"
    ),
    "A_quant/scripts/run_factor_batch_screen.py": (
        "011b754f01db87d04f1b924025b65c6c49999de7d20cc924cc9e22812f74c312"
    ),
}

EXPECTED_SOURCE_IDEA_COUNT = 100
EXPECTED_PINNED_IDEA_COUNT = 37
EXPECTED_STRUCTURAL_ALIAS_COUNT = 6
EXPECTED_INCOMPATIBLE_COUNT = 57
EXPECTED_FORMAL_CATALOG_COUNT = 267
MAX_TS_MEAN_WINDOW = 200

ALLOWED_DATA_FIELDS = frozenset(
    {
        "amount",
        "close",
        "fcf_to_price",
        "fin_debt_to_assets",
        "fin_net_profit_yoy",
        "fin_ocf_to_profit",
        "fin_roa",
        "fin_roe",
        "high",
        "low",
        "open",
        "turnover_rate",
        "vwap",
    }
)
ALLOWED_FUNCTION_ARITIES = {"cs_rank": 1, "ts_mean": 2}
MATRIX_HASH_CONTRACT_VERSION = "factor-no-label-matrix-f64-le.v1"
_SHA256_ZERO = "0" * 64


class FactorGovernanceAquantNoLabelEvalV4_1Error(ValueError):
    """Raised when a pinned definition or no-label evaluation fails closed."""


def canonical_json_bytes_v4_1(value: Any) -> bytes:
    """Canonical finite JSON used by the source and proof bindings."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def semantic_sha256_v4_1(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_1(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str, context: str) -> None:
    stored = value.get(field)
    if type(stored) is not str or len(stored) != 64:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            f"{context}.{field} is not a SHA-256"
        )
    try:
        int(stored, 16)
    except ValueError as exc:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            f"{context}.{field} is not a SHA-256"
        ) from exc
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    if semantic_sha256_v4_1(payload) != stored:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            f"{context}.{field} self-hash mismatch"
        )


def _sha(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            f"{context} is not a SHA-256"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            f"{context} is not a SHA-256"
        ) from exc
    return value


def _text(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            f"{context} must be a non-empty string"
        )
    return value


def _normalized_expression_node(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_DATA_FIELDS:
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                f"expression name is not allowlisted: {node.id}"
            )
        return {"kind": "name", "identifier": node.id}
    if isinstance(node, ast.Constant):
        value = node.value
        if type(value) not in (int, float) or type(value) is bool:
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                "expression constants must be finite JSON numbers"
            )
        if type(value) is float and not math.isfinite(value):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                "expression constants must be finite JSON numbers"
            )
        return {"kind": "constant", "value": value}
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                "only unary minus is allowlisted"
            )
        return {
            "kind": "unary",
            "operator": "negate",
            "operand": _normalized_expression_node(node.operand),
        }
    if isinstance(node, ast.BinOp):
        operators = {
            ast.Add: "add",
            ast.Sub: "subtract",
            ast.Mult: "multiply",
            ast.Div: "divide",
        }
        operator = operators.get(type(node.op))
        if operator is None:
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                "binary operator is not allowlisted"
            )
        return {
            "kind": "binary",
            "operator": operator,
            "left": _normalized_expression_node(node.left),
            "right": _normalized_expression_node(node.right),
        }
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in ALLOWED_FUNCTION_ARITIES
            or node.keywords
            or len(node.args) != ALLOWED_FUNCTION_ARITIES.get(node.func.id)
        ):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                "expression call is not allowlisted"
            )
        arguments = [_normalized_expression_node(item) for item in node.args]
        if node.func.id == "ts_mean":
            window = arguments[1]
            if (
                window.get("kind") != "constant"
                or type(window.get("value")) is not int
                or not 1 <= window["value"] <= MAX_TS_MEAN_WINDOW
            ):
                raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                    f"ts_mean window must be a positive int <= {MAX_TS_MEAN_WINDOW}"
                )
        return {
            "kind": "call",
            "function": node.func.id,
            "arguments": arguments,
        }
    raise FactorGovernanceAquantNoLabelEvalV4_1Error(
        f"expression syntax is not allowlisted: {type(node).__name__}"
    )


def normalize_expression_ast_v4_1(expression: str) -> dict[str, Any]:
    """Parse the exact standalone AST whitelist without algebraic rewriting."""

    text = _text(expression, "expression")
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "expression is not valid Python expression syntax"
        ) from exc
    return _normalized_expression_node(parsed.body)


def normalized_ast_sha256_v4_1(expression: str) -> str:
    return semantic_sha256_v4_1(normalize_expression_ast_v4_1(expression))


def _source_definition_sha256(idea: Mapping[str, Any]) -> str:
    return semantic_sha256_v4_1(
        {
            "version": "aquant-source-definition.v1",
            "pinned_commit": PINNED_COMMIT,
            "name": idea["name"],
            "expression": idea["expression"],
            "factor_type": idea["factor_type"],
            "source_family": idea["source_family"],
            "rationale": idea["rationale"],
            "direction": 1.0,
            "direction_origin": "expression_signed_ast",
        }
    )


def _candidate_definition_sha256(candidate: Mapping[str, Any]) -> str:
    return semantic_sha256_v4_1(
        {
            key: copy.deepcopy(item)
            for key, item in candidate.items()
            if key != "definition_sha256"
        }
    )


def _validate_source_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = copy.deepcopy(dict(value))
    _self_hash(receipt, "receipt_semantic_sha256", "source receipt")
    if (
        receipt.get("schema_version")
        != "factor-governance-aquant-source-receipt.v4.1"
        or receipt.get("protocol_version") != "v4"
        or receipt.get("source_system") != "A_quant"
        or receipt.get("pinned_commit") != PINNED_COMMIT
        or receipt.get("object_type") != "commit"
        or receipt.get("healthy") is not True
        or receipt.get("generator_path") != PINNED_GENERATOR_PATH
        or receipt.get("generator_function") != PINNED_GENERATOR_FUNCTION
        or receipt.get("generator_candidate_count") != EXPECTED_SOURCE_IDEA_COUNT
    ):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "source receipt does not identify the exact pinned A_quant source"
        )
    source_files = receipt.get("source_files")
    if not isinstance(source_files, list):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "source receipt source_files must be a list"
        )
    actual = {
        _text(row.get("path"), "source file path"): _sha(
            row.get("raw_sha256"), "source file raw SHA"
        )
        for row in source_files
        if isinstance(row, Mapping)
    }
    if actual != PINNED_SOURCE_FILES or len(source_files) != len(PINNED_SOURCE_FILES):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "source receipt file inventory differs from the exact pinned source"
        )
    return receipt


def bind_pinned_source_ideas_v4_1(
    *,
    source_receipt: Mapping[str, Any],
    source_idea_audit: Mapping[str, Any],
    primitive_mapping_proof: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind the exact 37 proof rows to their source and formal definitions."""

    receipt = _validate_source_receipt(source_receipt)
    audit = copy.deepcopy(dict(source_idea_audit))
    proof = copy.deepcopy(dict(primitive_mapping_proof))
    catalog = copy.deepcopy(dict(formal_catalog))
    _self_hash(audit, "audit_semantic_sha256", "source idea audit")
    _self_hash(proof, "proof_semantic_sha256", "primitive mapping proof")
    _self_hash(catalog, "semantic_sha256", "formal catalog")
    if (
        audit.get("schema_version") != "factor-governance-source-idea-audit.v4.1"
        or proof.get("schema_version")
        != "factor-governance-primitive-mapping-proof.v4.1"
        or catalog.get("schema_version") != "factor-candidate-catalog.v4"
        or audit.get("source_receipt_sha256")
        != receipt["receipt_semantic_sha256"]
        or proof.get("source_idea_audit_sha256")
        != audit["audit_semantic_sha256"]
        or proof.get("source_candidate_count") != EXPECTED_SOURCE_IDEA_COUNT
        or proof.get("new_candidate_count") != EXPECTED_PINNED_IDEA_COUNT
        or proof.get("structural_alias_count") != EXPECTED_STRUCTURAL_ALIAS_COUNT
        or proof.get("incompatible_count") != EXPECTED_INCOMPATIBLE_COUNT
    ):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "source/proof/catalog envelope mismatch"
        )
    ideas = audit.get("ideas")
    mappings = proof.get("new_candidate_mappings")
    candidates = catalog.get("candidates")
    if (
        not isinstance(ideas, list)
        or len(ideas) != EXPECTED_SOURCE_IDEA_COUNT
        or not isinstance(mappings, list)
        or len(mappings) != EXPECTED_PINNED_IDEA_COUNT
        or not isinstance(candidates, list)
        or len(candidates) != EXPECTED_FORMAL_CATALOG_COUNT
    ):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "source/proof/catalog exact accounting mismatch"
        )
    selected = [
        row
        for row in ideas
        if isinstance(row, Mapping)
        and row.get("compatibility_status") == "compatible"
        and row.get("catalog_role") == "new_candidate"
        and row.get("selected") is True
    ]
    if len(selected) != EXPECTED_PINNED_IDEA_COUNT:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "source audit does not contain the exact 37 selected ideas"
        )
    idea_by_id = {row.get("candidate_id"): row for row in selected}
    catalog_by_name = {
        row.get("name"): row for row in candidates if isinstance(row, Mapping)
    }
    mapping_ids = [row.get("candidate_id") for row in mappings if isinstance(row, Mapping)]
    if (
        len(mapping_ids) != EXPECTED_PINNED_IDEA_COUNT
        or mapping_ids != sorted(mapping_ids)
        or len(set(mapping_ids)) != EXPECTED_PINNED_IDEA_COUNT
        or set(mapping_ids) != set(idea_by_id)
    ):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "proof mappings are not the exact sorted source-idea set"
        )

    bound: list[dict[str, Any]] = []
    for index, raw_mapping in enumerate(mappings):
        if not isinstance(raw_mapping, Mapping):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                f"new_candidate_mappings[{index}] must be an object"
            )
        mapping = copy.deepcopy(dict(raw_mapping))
        _self_hash(mapping, "mapping_semantic_sha256", f"mapping[{index}]")
        candidate_id = _text(mapping.get("candidate_id"), "candidate_id")
        idea = copy.deepcopy(dict(idea_by_id[candidate_id]))
        name = _text(mapping.get("name"), "mapping name")
        candidate = catalog_by_name.get(name)
        if not isinstance(candidate, Mapping):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                f"formal candidate is missing: {name}"
            )
        candidate = copy.deepcopy(dict(candidate))
        expected_id = f"aquant:{PINNED_COMMIT}:{name}"
        expression = _text(mapping.get("expression"), "mapping expression")
        tree = normalize_expression_ast_v4_1(expression)
        ast_sha = semantic_sha256_v4_1(tree)
        source_sha = _source_definition_sha256(idea)
        catalog_sha = _candidate_definition_sha256(candidate)
        if (
            candidate_id != expected_id
            or idea.get("candidate_id") != candidate_id
            or idea.get("name") != name
            or idea.get("expression") != expression
            or idea.get("normalized_expression_ast") != tree
            or idea.get("direction") != 1.0
            or idea.get("direction_origin") != "expression_signed_ast"
            or mapping.get("implementation") != "aquant_expression_ast.v1"
            or mapping.get("mapping_status")
            != "complete_unique_occurrence_accounting"
            or mapping.get("source_definition_sha256") != source_sha
            or mapping.get("full_candidate_normalized_ast_sha256") != ast_sha
            or candidate.get("definition_sha256") != catalog_sha
            or mapping.get("catalog_definition_sha256") != catalog_sha
            or candidate.get("name") != name
            or candidate.get("expression") != expression
            or candidate.get("implementation") != "aquant_expression_ast.v1"
            or candidate.get("input_fields") != mapping.get("input_fields")
            or candidate.get("input_fields") != idea.get("input_fields")
            or candidate.get("primitive_ids") != mapping.get("primitive_ids")
            or candidate.get("family") != mapping.get("family")
            or candidate.get("slot") != mapping.get("slot")
            or candidate.get("lookback") != idea.get("lookback")
        ):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                f"source/proof/catalog drift for {candidate_id}"
            )
        bound.append(
            {
                "candidate_id": candidate_id,
                "name": name,
                "expression": expression,
                "normalized_expression_ast": tree,
                "source_index": idea["source_index"],
                "factor_type": idea["factor_type"],
                "source_family": idea["source_family"],
                "rationale": idea["rationale"],
                "input_fields": list(idea["input_fields"]),
                "lookback": idea["lookback"],
                "source_definition_sha256": source_sha,
                "full_candidate_normalized_ast_sha256": ast_sha,
                "catalog_definition_sha256": catalog_sha,
                "mapping_semantic_sha256": mapping["mapping_semantic_sha256"],
                "initial_weight": 0.0,
            }
        )
    return bound


def _validate_axes(
    matrices: Mapping[str, pd.DataFrame], eligibility_mask: pd.DataFrame
) -> tuple[pd.DatetimeIndex, pd.Index]:
    if not isinstance(eligibility_mask, pd.DataFrame) or eligibility_mask.empty:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "eligibility_mask must be a non-empty DataFrame"
        )
    if not isinstance(eligibility_mask.index, pd.DatetimeIndex):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "date axis must be a DatetimeIndex"
        )
    if (
        not eligibility_mask.index.is_unique
        or not eligibility_mask.index.is_monotonic_increasing
        or not eligibility_mask.columns.is_unique
        or any(type(item) is not str or not item for item in eligibility_mask.columns)
    ):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "eligibility axes must be ordered, unique, and canonical"
        )
    if any(dtype != bool for dtype in eligibility_mask.dtypes):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "eligibility_mask values must be exact booleans"
        )
    if not isinstance(matrices, Mapping) or not matrices:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "matrices must be a non-empty mapping"
        )
    for field, matrix in matrices.items():
        if field not in ALLOWED_DATA_FIELDS or not isinstance(matrix, pd.DataFrame):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                f"matrix field is unsupported: {field}"
            )
        if (
            not matrix.index.equals(eligibility_mask.index)
            or not matrix.columns.equals(eligibility_mask.columns)
        ):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                f"matrix axes differ from the exact eligibility axes: {field}"
            )
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                f"matrix must be numeric: {field}"
            )
    return eligibility_mask.index.copy(), eligibility_mask.columns.copy()


def _mask_frame(value: Any, eligibility_mask: pd.DataFrame) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.where(eligibility_mask)
    return value


def evaluate_expression_v4_1(
    *,
    expression: str,
    matrices: Mapping[str, pd.DataFrame],
    eligibility_mask: pd.DataFrame,
    expected_normalized_ast: Mapping[str, Any] | None = None,
    expected_normalized_ast_sha256: str | None = None,
) -> pd.DataFrame:
    """Evaluate one whitelist expression under the exact PIT mask envelope."""

    _validate_axes(matrices, eligibility_mask)
    tree = normalize_expression_ast_v4_1(expression)
    if expected_normalized_ast is not None and tree != dict(expected_normalized_ast):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "expression normalized AST differs from the bound source AST"
        )
    if expected_normalized_ast_sha256 is not None and semantic_sha256_v4_1(
        tree
    ) != _sha(expected_normalized_ast_sha256, "expected normalized AST SHA"):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "expression normalized AST SHA differs from the bound proof"
        )
    required: set[str] = set()

    def collect(node: Mapping[str, Any]) -> None:
        kind = node["kind"]
        if kind == "name":
            required.add(node["identifier"])
        elif kind == "unary":
            collect(node["operand"])
        elif kind == "binary":
            collect(node["left"])
            collect(node["right"])
        elif kind == "call":
            for argument in node["arguments"]:
                collect(argument)

    collect(tree)
    missing = sorted(required - set(matrices))
    if missing:
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "required input matrices are missing: " + ",".join(missing)
        )

    def visit(node: Mapping[str, Any]) -> Any:
        kind = node["kind"]
        if kind == "name":
            return matrices[node["identifier"]].astype(float).where(eligibility_mask)
        if kind == "constant":
            return node["value"]
        if kind == "unary":
            return _mask_frame(-visit(node["operand"]), eligibility_mask)
        if kind == "binary":
            left = visit(node["left"])
            right = visit(node["right"])
            operator = node["operator"]
            if operator == "add":
                value = left + right
            elif operator == "subtract":
                value = left - right
            elif operator == "multiply":
                value = left * right
            else:
                value = left / right
            return _mask_frame(value, eligibility_mask)
        function = node["function"]
        arguments = node["arguments"]
        if function == "ts_mean":
            frame = visit(arguments[0])
            if not isinstance(frame, pd.DataFrame):
                raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                    "ts_mean first argument must evaluate to a DataFrame"
                )
            window = arguments[1]["value"]
            value = frame.rolling(window=window, min_periods=1).mean()
            return value.where(eligibility_mask)
        frame = visit(arguments[0])
        if not isinstance(frame, pd.DataFrame):
            raise FactorGovernanceAquantNoLabelEvalV4_1Error(
                "cs_rank argument must evaluate to a DataFrame"
            )
        masked = frame.where(eligibility_mask)
        value = masked.rank(axis=1, pct=True, na_option="keep")
        return value.where(eligibility_mask)

    evaluated = visit(tree)
    if not isinstance(evaluated, pd.DataFrame):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "expression did not evaluate to a DataFrame"
        )
    return evaluated.where(eligibility_mask)


def evaluate_pinned_idea_v4_1(
    *,
    idea: Mapping[str, Any],
    matrices: Mapping[str, pd.DataFrame],
    eligibility_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate one already-bound exact-37 row and recheck both AST bindings."""

    expression = _text(idea.get("expression"), "idea.expression")
    tree = idea.get("normalized_expression_ast")
    if not isinstance(tree, Mapping):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "idea.normalized_expression_ast must be an object"
        )
    return evaluate_expression_v4_1(
        expression=expression,
        matrices=matrices,
        eligibility_mask=eligibility_mask,
        expected_normalized_ast=tree,
        expected_normalized_ast_sha256=_sha(
            idea.get("full_candidate_normalized_ast_sha256"),
            "idea.full_candidate_normalized_ast_sha256",
        ),
    )


def _axis_sha256(values: Sequence[str], axis: str) -> str:
    return semantic_sha256_v4_1(
        {"contract": "factor-no-label-axis.v1", "axis": axis, "values": list(values)}
    )


def matrix_hash_descriptor_v4_1(value: pd.DataFrame) -> dict[str, Any]:
    """Hash exact axes and canonical little-endian float64 matrix bytes.

    NaN payload bits are normalized to one quiet-NaN representation.  Positive
    and negative infinity retain distinct IEEE-754 encodings.
    """

    if not isinstance(value, pd.DataFrame) or not isinstance(
        value.index, pd.DatetimeIndex
    ):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "matrix hash input must have a DatetimeIndex"
        )
    if any(type(item) is not str for item in value.columns):
        raise FactorGovernanceAquantNoLabelEvalV4_1Error(
            "matrix hash symbol axis must contain strings"
        )
    dates = [item.isoformat() for item in value.index]
    symbols = list(value.columns)
    array = np.asarray(value.to_numpy(dtype=np.float64, copy=True), dtype="<f8", order="C")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype="<f8")
    bits = array.view("<u8")
    bits[np.isnan(array)] = np.uint64(0x7FF8000000000000)
    header = {
        "contract": MATRIX_HASH_CONTRACT_VERSION,
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "dtype": "float64-little-endian",
        "date_axis_sha256": _axis_sha256(dates, "date"),
        "symbol_axis_sha256": _axis_sha256(symbols, "symbol"),
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes_v4_1(header))
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return {**header, "matrix_sha256": digest.hexdigest()}


def matrix_sha256_v4_1(value: pd.DataFrame) -> str:
    return matrix_hash_descriptor_v4_1(value)["matrix_sha256"]


__all__ = [
    "ALLOWED_DATA_FIELDS",
    "ALLOWED_FUNCTION_ARITIES",
    "EXPECTED_PINNED_IDEA_COUNT",
    "FactorGovernanceAquantNoLabelEvalV4_1Error",
    "MATRIX_HASH_CONTRACT_VERSION",
    "MAX_TS_MEAN_WINDOW",
    "PINNED_COMMIT",
    "PINNED_SOURCE_FILES",
    "bind_pinned_source_ideas_v4_1",
    "canonical_json_bytes_v4_1",
    "evaluate_expression_v4_1",
    "evaluate_pinned_idea_v4_1",
    "matrix_hash_descriptor_v4_1",
    "matrix_sha256_v4_1",
    "normalize_expression_ast_v4_1",
    "normalized_ast_sha256_v4_1",
    "semantic_sha256_v4_1",
]
