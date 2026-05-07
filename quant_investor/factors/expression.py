"""Safe expression sandbox for offline factor matrix research."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import quant_investor.factors.operators as operators
from quant_investor.factors.matrix import (
    FIELD_BENCHMARK_RET,
    FIELD_CLOSE,
    FIELD_RET1,
    FIELD_VWAP,
    ExpressionEvaluationResult,
    FactorMatrix,
    MatrixDataBundle,
    add_standard_derived_fields,
    compute_coverage,
    make_expression_result_id,
    make_factor_matrix_id,
)
from quant_investor.versioning import FACTOR_EXPRESSION_SCHEMA_VERSION


_BINARY_OPERATOR_NAMES = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
}

_BANNED_CALL_NAMES = {
    "__import__",
    "eval",
    "exec",
    "getattr",
    "setattr",
    "globals",
    "locals",
    "open",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    return value


def _ensure_json_serializable(value: Any, label: str) -> Any:
    safe = _json_safe(value)
    try:
        json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON-serializable values.") from exc
    return safe


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, "metadata"))


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _non_empty_str(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _slug(value: str | None) -> str:
    resolved = "none" if value is None else str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", resolved)
    return slug.strip("-") or "unknown"


def _short_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps(
        [_json_safe(part) for part in parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class FactorExpressionSpec:
    schema_version: str = FACTOR_EXPRESSION_SCHEMA_VERSION
    expression_id: str = ""
    expression: str = ""
    factor_id: str | None = None
    factor_version: str | None = None
    allowed_fields: list[str] = field(default_factory=list)
    allowed_operators: list[str] = field(default_factory=list)
    apply_universe_mask: bool = True
    apply_tradability_mask: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or FACTOR_EXPRESSION_SCHEMA_VERSION)
        self.expression_id = _non_empty_str(self.expression_id, "expression_id")
        self.expression = _non_empty_str(self.expression, "expression")
        self.factor_id = _optional_str(self.factor_id)
        self.factor_version = _optional_str(self.factor_version)
        self.allowed_fields = _ordered_unique(self.allowed_fields)
        self.allowed_operators = _ordered_unique(self.allowed_operators)
        self.apply_universe_mask = bool(self.apply_universe_mask)
        self.apply_tradability_mask = bool(self.apply_tradability_mask)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expression_id": self.expression_id,
            "expression": self.expression,
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "allowed_fields": list(self.allowed_fields),
            "allowed_operators": list(self.allowed_operators),
            "apply_universe_mask": self.apply_universe_mask,
            "apply_tradability_mask": self.apply_tradability_mask,
            "metadata": dict(_json_safe(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorExpressionSpec":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", FACTOR_EXPRESSION_SCHEMA_VERSION)),
            expression_id=str(data.get("expression_id", "")),
            expression=str(data.get("expression", "")),
            factor_id=data.get("factor_id"),
            factor_version=data.get("factor_version"),
            allowed_fields=list(data.get("allowed_fields", []) or []),
            allowed_operators=list(data.get("allowed_operators", []) or []),
            apply_universe_mask=bool(data.get("apply_universe_mask", True)),
            apply_tradability_mask=bool(data.get("apply_tradability_mask", True)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def make_expression_spec_id(
    *,
    expression: str,
    allowed_fields: Sequence[str],
    allowed_operators: Sequence[str],
) -> str:
    fields = sorted({str(field_name) for field_name in allowed_fields})
    operator_names = sorted({str(operator_name) for operator_name in allowed_operators})
    parts = [str(expression), fields, operator_names]
    return f"factor-expression-spec-{_short_hash(parts)}"


def get_default_allowed_operators() -> dict[str, Callable[..., Any]]:
    return {
        "ts_delay": operators.ts_delay,
        "ts_delta": operators.ts_delta,
        "ts_mean": operators.ts_mean,
        "ts_sum": operators.ts_sum,
        "ts_std": operators.ts_std,
        "ts_min": operators.ts_min,
        "ts_max": operators.ts_max,
        "ts_rank": operators.ts_rank,
        "ts_corr": operators.ts_corr,
        "cs_rank": operators.cs_rank,
        "cs_zscore": operators.cs_zscore,
        "cs_winsorize": operators.cs_winsorize,
        "cs_indneut": operators.cs_indneut,
        "cs_booksize": operators.cs_booksize,
        "add": operators.add,
        "sub": operators.sub,
        "mul": operators.mul,
        "div": operators.div,
        "neg": operators.neg,
        "abs": operators.abs_,
        "sign": operators.sign,
        "log": operators.log,
        "sqrt": operators.sqrt,
        "maximum": operators.maximum,
        "minimum": operators.minimum,
    }


class _ExpressionEvaluator:
    def __init__(
        self,
        *,
        spec: FactorExpressionSpec,
        bundle: MatrixDataBundle,
        context: Mapping[str, Sequence[Sequence[Any]]],
        allowed_operator_map: Mapping[str, Callable[..., Any]],
    ) -> None:
        self.spec = spec
        self.bundle = bundle
        self.context = dict(context)
        self.allowed_operator_map = dict(allowed_operator_map)
        self.allowed_fields = set(spec.allowed_fields)
        self.used_fields: set[str] = set()
        self.used_operators: set[str] = set()

    def evaluate(self) -> Any:
        try:
            tree = ast.parse(self.spec.expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid factor expression syntax: {exc.msg}") from exc
        return self._eval_node(tree)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        if isinstance(node, ast.Name):
            return self._eval_name(node)
        if isinstance(node, ast.Constant):
            return self._eval_constant(node)
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unary(node)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed in factor expressions.")
        if isinstance(node, ast.Subscript):
            raise ValueError("Subscript access is not allowed in factor expressions.")
        if isinstance(node, ast.Lambda):
            raise ValueError("Lambda expressions are not allowed in factor expressions.")
        raise ValueError(f"Disallowed expression node: {node.__class__.__name__}.")

    def _eval_name(self, node: ast.Name) -> Any:
        name = node.id
        if "__" in name:
            raise ValueError("Dunder names are not allowed in factor expressions.")
        if self.allowed_fields and name not in self.allowed_fields:
            raise ValueError(f"Field {name!r} is not in allowed_fields.")
        if name not in self.context:
            raise ValueError(f"Referenced field {name!r} is not available in the matrix bundle.")
        self.used_fields.add(name)
        return [list(row) for row in self.context[name]]

    def _eval_constant(self, node: ast.Constant) -> int | float | bool | str:
        if isinstance(node.value, (int, float, bool, str)) and node.value is not None:
            return node.value
        raise ValueError("Only int, float, bool, and str constants are allowed.")

    def _eval_binop(self, node: ast.BinOp) -> Any:
        operator_name = _BINARY_OPERATOR_NAMES.get(type(node.op))
        if operator_name is None:
            raise ValueError(f"Binary operator {node.op.__class__.__name__} is not allowed.")
        if operator_name not in self.allowed_operator_map:
            raise ValueError(f"Operator {operator_name!r} is not allowed.")
        left = self._eval_node(node.left)
        right = self._eval_node(node.right)
        self.used_operators.add(operator_name)
        return self.allowed_operator_map[operator_name](left, right)

    def _eval_unary(self, node: ast.UnaryOp) -> Any:
        if isinstance(node.op, ast.UAdd):
            return self._eval_node(node.operand)
        if not isinstance(node.op, ast.USub):
            raise ValueError(f"Unary operator {node.op.__class__.__name__} is not allowed.")
        if "neg" not in self.allowed_operator_map:
            raise ValueError("Operator 'neg' is not allowed.")
        operand = self._eval_node(node.operand)
        self.used_operators.add("neg")
        return self.allowed_operator_map["neg"](operand)

    def _eval_call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to whitelisted operators are allowed.")
        function_name = node.func.id
        if "__" in function_name:
            raise ValueError("Dunder names are not allowed in factor expressions.")
        if function_name in _BANNED_CALL_NAMES:
            raise ValueError(f"Call to {function_name!r} is not allowed.")
        if function_name not in self.allowed_operator_map:
            raise ValueError(f"Operator {function_name!r} is not allowed.")
        args = [self._eval_node(arg) for arg in node.args]
        kwargs = {keyword.arg: self._eval_keyword(keyword) for keyword in node.keywords}
        self.used_operators.add(function_name)
        if function_name == "cs_indneut":
            if len(args) != 1 or kwargs:
                raise ValueError(
                    "cs_indneut accepts only the factor matrix; "
                    "industry context is supplied internally."
                )
            return self.allowed_operator_map[function_name](
                args[0],
                self.bundle.industry_by_symbol,
                self.bundle.contract.symbols,
            )
        return self.allowed_operator_map[function_name](*args, **kwargs)

    def _eval_keyword(self, keyword: ast.keyword) -> Any:
        if keyword.arg is None:
            raise ValueError("Expanded keyword arguments are not allowed.")
        if "__" in keyword.arg:
            raise ValueError("Dunder keyword names are not allowed.")
        if not isinstance(keyword.value, ast.Constant):
            raise ValueError("Keyword arguments must be constants.")
        return self._eval_constant(keyword.value)


def _build_context(
    bundle: MatrixDataBundle,
    *,
    extra_fields: Mapping[str, Sequence[Sequence[Any]]] | None,
) -> tuple[MatrixDataBundle, dict[str, Sequence[Sequence[Any]]]]:
    enriched_bundle = add_standard_derived_fields(bundle)
    context: dict[str, Sequence[Sequence[Any]]] = {
        field_name: enriched_bundle.get_field(field_name) for field_name in enriched_bundle.fields
    }
    if extra_fields:
        for field_name, values in sorted(extra_fields.items(), key=lambda item: str(item[0])):
            name = str(field_name)
            if "__" in name:
                raise ValueError("Dunder field names are not allowed.")
            if name in context:
                raise ValueError(f"extra_fields may not override existing field {name!r}.")
            enriched_bundle.validate_shape(values, field_name=name)
            context[name] = [list(row) for row in values]
    return enriched_bundle, context


def _operator_whitelist(
    spec: FactorExpressionSpec,
    *,
    extra_operators: Mapping[str, Callable[..., Any]] | None,
) -> dict[str, Callable[..., Any]]:
    available = get_default_allowed_operators()
    if extra_operators:
        for name, fn in extra_operators.items():
            text_name = str(name)
            if "__" in text_name:
                raise ValueError("Dunder operator names are not allowed.")
            available[text_name] = fn
    requested = set(spec.allowed_operators) if spec.allowed_operators else set(available)
    unknown = sorted(requested - set(available))
    if unknown:
        raise ValueError(f"Unknown allowed operator(s): {unknown}.")
    return {name: available[name] for name in sorted(requested)}


def _derived_warnings(
    spec: FactorExpressionSpec,
    bundle: MatrixDataBundle,
    context: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    allowed_or_referenced = set(spec.allowed_fields) if spec.allowed_fields else set(context)
    if FIELD_VWAP in allowed_or_referenced and FIELD_VWAP not in context:
        warnings.append("standard_derived_field_skipped:vwap_requires_amount_and_volume")
    if FIELD_RET1 in allowed_or_referenced and FIELD_RET1 not in context:
        warnings.append("standard_derived_field_skipped:ret1_requires_close")
    if FIELD_BENCHMARK_RET in allowed_or_referenced and FIELD_BENCHMARK_RET not in context:
        if not bundle.has_field(FIELD_BENCHMARK_RET):
            warnings.append("standard_derived_field_skipped:benchmark_ret_requires_benchmark_close")
    if FIELD_CLOSE not in context and FIELD_RET1 in allowed_or_referenced:
        warnings.append("standard_input_field_missing:close")
    return warnings


def _coerce_result_matrix(result: Any, bundle: MatrixDataBundle) -> list[list[float | None]]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        raise ValueError("Factor expression must return a symbols x dates matrix.")
    bundle.validate_shape(result, field_name="expression_result")
    output: list[list[float | None]] = []
    for row in result:
        row_values: list[float | None] = []
        for value in row:
            if value is None:
                row_values.append(None)
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Factor expression output values must be numeric or None.")
            else:
                number = float(value)
                if not number == number or number in (float("inf"), float("-inf")):
                    row_values.append(None)
                else:
                    row_values.append(number)
        output.append(row_values)
    return output


def _apply_final_masks(
    values: Sequence[Sequence[float | None]],
    bundle: MatrixDataBundle,
    *,
    apply_universe_mask: bool,
    apply_tradability_mask: bool,
) -> list[list[float | None]]:
    output = [list(row) for row in values]
    masks: list[Sequence[Sequence[bool]]] = []
    if apply_universe_mask and bundle.universe_mask is not None:
        masks.append(bundle.universe_mask)
    if apply_tradability_mask and bundle.tradability_mask is not None:
        masks.append(bundle.tradability_mask)
    for mask in masks:
        bundle.validate_shape(mask, field_name="mask")
        for row_index, mask_row in enumerate(mask):
            for col_index, included in enumerate(mask_row):
                if not included:
                    output[row_index][col_index] = None
    return output


def evaluate_factor_expression(
    spec: FactorExpressionSpec,
    bundle: MatrixDataBundle,
    *,
    extra_fields: Mapping[str, Sequence[Sequence[Any]]] | None = None,
    extra_operators: Mapping[str, Callable[..., Any]] | None = None,
) -> ExpressionEvaluationResult:
    enriched_bundle, context = _build_context(bundle, extra_fields=extra_fields)
    allowed_operator_map = _operator_whitelist(spec, extra_operators=extra_operators)
    evaluator = _ExpressionEvaluator(
        spec=spec,
        bundle=enriched_bundle,
        context=context,
        allowed_operator_map=allowed_operator_map,
    )
    raw_result = evaluator.evaluate()
    result_values = _coerce_result_matrix(raw_result, enriched_bundle)
    masked_values = _apply_final_masks(
        result_values,
        enriched_bundle,
        apply_universe_mask=spec.apply_universe_mask,
        apply_tradability_mask=spec.apply_tradability_mask,
    )
    coverage_ratio, missing_ratio = compute_coverage(masked_values)
    matrix_id = make_factor_matrix_id(
        expression=spec.expression,
        symbols=enriched_bundle.contract.symbols,
        dates=enriched_bundle.contract.dates,
        factor_id=spec.factor_id,
    )
    factor_matrix = FactorMatrix(
        matrix_id=matrix_id,
        factor_id=spec.factor_id,
        factor_version=spec.factor_version,
        expression=spec.expression,
        symbols=list(enriched_bundle.contract.symbols),
        dates=list(enriched_bundle.contract.dates),
        values=masked_values,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        metadata={
            "contract_id": enriched_bundle.contract.contract_id,
            "bundle_id": enriched_bundle.bundle_id,
            "expression_id": spec.expression_id,
            "offline_only": True,
        },
    )
    return ExpressionEvaluationResult(
        result_id=make_expression_result_id(expression=spec.expression, matrix_id=matrix_id),
        expression=spec.expression,
        factor_matrix=factor_matrix,
        used_fields=sorted(evaluator.used_fields),
        used_operators=sorted(evaluator.used_operators),
        warnings=_derived_warnings(spec, bundle, context),
        metadata={
            "contract_id": enriched_bundle.contract.contract_id,
            "bundle_id": enriched_bundle.bundle_id,
            "expression_id": spec.expression_id,
            "sandbox": "ast-whitelist",
            "offline_only": True,
        },
    )


__all__ = [
    "FactorExpressionSpec",
    "make_expression_spec_id",
    "get_default_allowed_operators",
    "evaluate_factor_expression",
]
