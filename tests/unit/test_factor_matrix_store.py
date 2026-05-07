from __future__ import annotations

import pytest

from quant_investor.factors.matrix import (
    FIELD_CLOSE,
    ExpressionEvaluationResult,
    FactorMatrix,
    MatrixDataBundle,
    MatrixDataContract,
    compute_coverage,
    make_expression_result_id,
    make_factor_matrix_id,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)
from quant_investor.factors.store import FactorMatrixStore


def _contract() -> MatrixDataContract:
    symbols = ["AAA", "BBB"]
    dates = ["2026-01-01", "2026-01-02"]
    return MatrixDataContract(
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
        required_fields=[FIELD_CLOSE],
        field_sources={FIELD_CLOSE: "fixture"},
        point_in_time_flags={FIELD_CLOSE: True},
    )


def _bundle() -> MatrixDataBundle:
    contract = _contract()
    fields = {FIELD_CLOSE: [[10.0, 11.0], [20.0, 21.0]]}
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(
            contract_id=contract.contract_id,
            field_names=fields.keys(),
        ),
        contract=contract,
        fields=fields,
    )


def _factor_matrix() -> FactorMatrix:
    bundle = _bundle()
    values = bundle.get_field(FIELD_CLOSE)
    coverage_ratio, missing_ratio = compute_coverage(values)
    return FactorMatrix(
        matrix_id=make_factor_matrix_id(
            expression=FIELD_CLOSE,
            symbols=bundle.contract.symbols,
            dates=bundle.contract.dates,
        ),
        expression=FIELD_CLOSE,
        symbols=bundle.contract.symbols,
        dates=bundle.contract.dates,
        values=values,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
    )


def _expression_result() -> ExpressionEvaluationResult:
    matrix = _factor_matrix()
    return ExpressionEvaluationResult(
        result_id=make_expression_result_id(
            expression=matrix.expression,
            matrix_id=matrix.matrix_id,
        ),
        expression=matrix.expression,
        factor_matrix=matrix,
        used_fields=[FIELD_CLOSE],
        used_operators=[],
        warnings=[],
    )


def test_append_and_read_matrix_contract(tmp_path) -> None:
    store = FactorMatrixStore(tmp_path / "matrix_store")
    contract = _contract()

    store.append_matrix_contract(contract)

    assert store.read_matrix_contracts()[0].to_dict() == contract.to_dict()
    assert store.get_matrix_contract_ids() == {contract.contract_id}


def test_append_and_read_matrix_bundle(tmp_path) -> None:
    store = FactorMatrixStore(tmp_path / "matrix_store")
    bundle = _bundle()

    store.append_matrix_bundle(bundle)

    assert store.read_matrix_bundles()[0].to_dict() == bundle.to_dict()
    assert store.get_matrix_bundle_ids() == {bundle.bundle_id}


def test_append_and_read_factor_matrix(tmp_path) -> None:
    store = FactorMatrixStore(tmp_path / "matrix_store")
    matrix = _factor_matrix()

    store.append_factor_matrix(matrix)

    assert store.read_factor_matrices()[0].to_dict() == matrix.to_dict()
    assert store.get_factor_matrix_ids() == {matrix.matrix_id}


def test_append_and_read_expression_result(tmp_path) -> None:
    store = FactorMatrixStore(tmp_path / "matrix_store")
    result = _expression_result()

    store.append_expression_result(result)

    assert store.read_expression_results()[0].to_dict() == result.to_dict()
    assert store.get_expression_result_ids() == {result.result_id}


def test_duplicate_ids_raise_on_append(tmp_path) -> None:
    store = FactorMatrixStore(tmp_path / "matrix_store")
    contract = _contract()
    store.append_matrix_contract(contract)

    with pytest.raises(ValueError, match="Duplicate contract_id"):
        store.append_matrix_contract(contract)


def test_malformed_json_raises_clear_error(tmp_path) -> None:
    store = FactorMatrixStore(tmp_path / "matrix_store")
    store.matrix_contracts_path.parent.mkdir(parents=True, exist_ok=True)
    store.matrix_contracts_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_matrix_contracts()


def test_store_creates_directories_on_demand(tmp_path) -> None:
    root = tmp_path / "missing" / "matrix_store"
    store = FactorMatrixStore(root)

    assert not root.exists()
    store.append_matrix_contract(_contract())

    assert root.exists()
    assert store.matrix_contracts_path.exists()
