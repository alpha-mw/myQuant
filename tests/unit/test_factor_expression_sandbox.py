from __future__ import annotations

import pytest

from quant_investor.factors.expression import (
    FactorExpressionSpec,
    evaluate_factor_expression,
    get_default_allowed_operators,
    make_expression_spec_id,
)
from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_CLOSE,
    FIELD_OPEN,
    FIELD_RET1,
    FIELD_VOLUME,
    FIELD_VWAP,
    MatrixDataBundle,
    MatrixDataContract,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)


def _allowed_ops() -> list[str]:
    return sorted(get_default_allowed_operators())


def _bundle() -> MatrixDataBundle:
    symbols = ["AAA", "BBB"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    contract = MatrixDataContract(
        contract_id=make_matrix_contract_id(
            universe="CN",
            benchmark="CSI300",
            symbols=symbols,
            dates=dates,
        ),
        universe="CN",
        benchmark="CSI300",
        symbols=symbols,
        dates=dates,
        required_fields=[FIELD_OPEN, FIELD_CLOSE, FIELD_AMOUNT, FIELD_VOLUME],
        optional_fields=[FIELD_VWAP, FIELD_RET1],
        field_sources={
            FIELD_OPEN: "fixture",
            FIELD_CLOSE: "fixture",
            FIELD_AMOUNT: "fixture",
            FIELD_VOLUME: "fixture",
        },
        point_in_time_flags={
            FIELD_OPEN: True,
            FIELD_CLOSE: True,
            FIELD_AMOUNT: True,
            FIELD_VOLUME: True,
        },
    )
    fields = {
        FIELD_OPEN: [[9.0, 10.0, 11.0], [19.0, 20.0, 21.0]],
        FIELD_CLOSE: [[10.0, 11.0, 12.0], [20.0, 22.0, 24.0]],
        FIELD_AMOUNT: [[120.0, 240.0, 390.0], [300.0, 440.0, 720.0]],
        FIELD_VOLUME: [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]],
    }
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(
            contract_id=contract.contract_id,
            field_names=fields.keys(),
        ),
        contract=contract,
        fields=fields,
        universe_mask=[[True, False, True], [True, True, True]],
        tradability_mask=[[True, True, True], [False, True, True]],
        industry_by_symbol={"AAA": "Tech", "BBB": "Tech"},
    )


def _spec(expression: str, *, masks: bool = False) -> FactorExpressionSpec:
    allowed_fields = [
        FIELD_OPEN,
        FIELD_CLOSE,
        FIELD_AMOUNT,
        FIELD_VOLUME,
        FIELD_VWAP,
        FIELD_RET1,
    ]
    return FactorExpressionSpec(
        expression_id=make_expression_spec_id(
            expression=expression,
            allowed_fields=allowed_fields,
            allowed_operators=_allowed_ops(),
        ),
        expression=expression,
        factor_id="factor-fixture",
        factor_version="v1",
        allowed_fields=allowed_fields,
        allowed_operators=_allowed_ops(),
        apply_universe_mask=masks,
        apply_tradability_mask=masks,
    )


def test_simple_function_expression_evaluates_derived_vwap() -> None:
    result = evaluate_factor_expression(_spec("sub(vwap, close)"), _bundle())

    assert result.factor_matrix.values == [[2.0, 1.0, 1.0], [10.0, 0.0, 0.0]]
    assert result.used_fields == [FIELD_CLOSE, FIELD_VWAP]
    assert result.used_operators == ["sub"]
    assert result.factor_matrix.factor_id == "factor-fixture"
    assert result.factor_matrix.coverage_ratio == 1.0


def test_binary_expression_maps_to_safe_operators() -> None:
    result = evaluate_factor_expression(_spec("vwap - close"), _bundle())

    assert result.factor_matrix.values == [[2.0, 1.0, 1.0], [10.0, 0.0, 0.0]]
    assert result.used_operators == ["sub"]


def test_expression_with_ts_delay_and_cs_rank_evaluates() -> None:
    result = evaluate_factor_expression(_spec("cs_rank(ts_delay(close, 1))"), _bundle())

    assert result.factor_matrix.values == [[None, 0.0, 0.0], [None, 1.0, 1.0]]
    assert result.used_fields == [FIELD_CLOSE]
    assert result.used_operators == ["cs_rank", "ts_delay"]


def test_cs_indneut_uses_bundle_industry_context_internally() -> None:
    result = evaluate_factor_expression(_spec("cs_indneut(close)"), _bundle())

    assert result.factor_matrix.values == [[-5.0, -5.5, -6.0], [5.0, 5.5, 6.0]]
    assert result.used_operators == ["cs_indneut"]


def test_final_output_applies_universe_and_tradability_masks() -> None:
    result = evaluate_factor_expression(_spec("vwap - close", masks=True), _bundle())

    assert result.factor_matrix.values == [[2.0, None, 1.0], [None, 0.0, 0.0]]
    assert result.factor_matrix.coverage_ratio == pytest.approx(4 / 6)
    assert result.factor_matrix.missing_ratio == pytest.approx(2 / 6)


def test_missing_referenced_field_raises() -> None:
    spec = FactorExpressionSpec(
        expression_id=make_expression_spec_id(
            expression="high",
            allowed_fields=["high"],
            allowed_operators=_allowed_ops(),
        ),
        expression="high",
        allowed_fields=["high"],
        allowed_operators=_allowed_ops(),
    )

    with pytest.raises(ValueError, match="not available"):
        evaluate_factor_expression(spec, _bundle())


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "vwap.__class__",
        "vwap[0]",
        "lambda x: x",
        "globals()",
        "eval('1+1')",
    ],
)
def test_disallowed_expressions_raise_value_error(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate_factor_expression(_spec(expression), _bundle())


def test_operator_and_field_whitelists_are_enforced() -> None:
    spec = FactorExpressionSpec(
        expression_id=make_expression_spec_id(
            expression="cs_rank(close)",
            allowed_fields=[FIELD_CLOSE],
            allowed_operators=["sub"],
        ),
        expression="cs_rank(close)",
        allowed_fields=[FIELD_CLOSE],
        allowed_operators=["sub"],
    )

    with pytest.raises(ValueError, match="Operator"):
        evaluate_factor_expression(spec, _bundle())

    spec = FactorExpressionSpec(
        expression_id=make_expression_spec_id(
            expression="vwap",
            allowed_fields=[FIELD_CLOSE],
            allowed_operators=_allowed_ops(),
        ),
        expression="vwap",
        allowed_fields=[FIELD_CLOSE],
        allowed_operators=_allowed_ops(),
    )

    with pytest.raises(ValueError, match="allowed_fields"):
        evaluate_factor_expression(spec, _bundle())
